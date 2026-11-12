"""OLMoE-1B-7B (allenai/OLMoE-1B-7B-0924): shared pieces for the fixed-route BC study (run_olmoe_fixed_route.py).

transformers 5.19 is taken from the isolated qwen35_assets/python_deps (as qwen35_common does); the checkpoint is
in olmoe_assets/model (BF16 safetensors, revision pinned below).

Fixed route. For every window the unquantized FP32 model is run once with its own (dynamic) router; per layer and
token the selected expert ids and their routing weights (softmax over all experts in float32, top-k, NOT
renormalized: norm_topk_prob = false) are cached. A fixed-route forward then uses, in every layer, the cached ids and
weights as constants (no gradient) while hidden states, expert inputs and expert outputs are recomputed:
    y_{l,t} = sum_{e in ids(l,t)} r_FP32(l,t,e) * Expert_e(h_{l,t})
With the FP32 weights and no quantization this equals the dynamic forward; its gradients differ from the dynamic
model's (the routing weights do not depend on h), so statistics and the truth both use the fixed route.

Two implementations of a decoder layer:
  * hf_layer (OlmoeDecoderLayer of transformers, eager attention, eager experts): the dynamic FP32 reference;
  * layer_fixed (here): every operation in the dtype of its inputs (an FP64 run is FP64 throughout: transformers'
    RMSNorm, router softmax, rotary and attention softmax compute in float32 whatever the module dtype), separate
    gate / up / down linears per expert (quantization and taps per projection), transformers' expert / token order.

The full model in FP32 (27.7 GB) does not fit the GPU, so layers are streamed: the BF16 tensors of one layer are
read from the safetensors (mmap), moved to the device and cast to FP32 (exact). A backward pass recomputes each
layer from its stored input (checkpointing; equivalence checked on the tiny model and on a real-model sub-stack).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "olmoe_assets"
sys.path.insert(0, str(HERE / "qwen35_assets" / "python_deps"))
import transformers  # noqa: E402,F401
import tokenizers  # noqa: E402,F401
import safetensors  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_DIR = ASSETS / "model"
REPO, REVISION = "allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
PROJ = ("gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------------------- #
# config, tokenizer, data
# --------------------------------------------------------------------------- #
def config(model_dir=MODEL_DIR):
    from transformers import OlmoeConfig
    c = OlmoeConfig.from_pretrained(model_dir)
    c._attn_implementation = "eager"
    c._experts_implementation = "eager"
    return c


def tokenizer(model_dir=MODEL_DIR):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def wikitext(split):
    """WikiText-2 raw (gpt2_assets/data parquet), lines joined with blank lines: the text of qwen35_common."""
    import pyarrow.parquet as pq
    return "\n\n".join(pq.read_table(HERE / "gpt2_assets/data" / f"{split}-00000-of-00001.parquet")["text"].to_pylist())


def windows(split, first, n, seq, tok=None):
    """n windows of seq tokens from token `first` of the split, with each window's character span in the text."""
    tok = tok or tokenizer()
    text = wikitext(split)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    if first + n * seq > len(ids):
        raise ValueError(f"{split}: {len(ids)} tokens")
    t = np.asarray(ids[first:first + n * seq], np.int64).reshape(n, seq)
    spans = [[int(off[first + k * seq][0]), int(off[first + (k + 1) * seq - 1][1])] for k in range(n)]
    return t, spans, len(ids), len(text)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
class Checkpoint:
    """Tensors of the safetensors checkpoint by name (memory-mapped); per layer on demand."""

    def __init__(self, model_dir=MODEL_DIR):
        from safetensors import safe_open
        self.dir = Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {f: safe_open(str(self.dir / f), "pt") for f in sorted(set(self.map.values()))}

    def tensor(self, name, device="cpu", dtype=torch.float32):
        return self.handles[self.map[name]].get_tensor(name).to(device).to(dtype)

    def global_(self, device, dtype):
        return {"embed": self.tensor("model.embed_tokens.weight", device, dtype),
                "norm": self.tensor("model.norm.weight", device, dtype),
                "lm_head": self.tensor("lm_head.weight", device, dtype)}

    def layer(self, l, device, dtype, num_experts=64):
        p = f"model.layers.{l}."
        t = lambda n: self.tensor(p + n, device, dtype)
        w = {"ln1": t("input_layernorm.weight"), "ln2": t("post_attention_layernorm.weight"),
             "q": t("self_attn.q_proj.weight"), "k": t("self_attn.k_proj.weight"), "v": t("self_attn.v_proj.weight"),
             "o": t("self_attn.o_proj.weight"), "qn": t("self_attn.q_norm.weight"), "kn": t("self_attn.k_norm.weight"),
             "router": t("mlp.gate.weight")}
        for pr in PROJ:
            w[pr] = torch.stack([t(f"mlp.experts.{e}.{pr}.weight") for e in range(num_experts)])
        return w


class StateDict:
    """The same interface over a transformers OlmoeForCausalLM (tiny model): fused gate_up_proj split back."""

    def __init__(self, model):
        self.sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.inter = model.config.intermediate_size

    def _t(self, k, device, dtype):            # always a fresh leaf (callers set requires_grad on what they get)
        return self.sd[k].to(device=device, dtype=dtype, copy=True)

    def global_(self, device, dtype):
        return {"embed": self._t("model.embed_tokens.weight", device, dtype),
                "norm": self._t("model.norm.weight", device, dtype),
                "lm_head": self._t("lm_head.weight", device, dtype)}

    def layer(self, l, device, dtype, num_experts=None):
        p = f"model.layers.{l}."
        t = lambda n: self._t(p + n, device, dtype)
        gu = t("mlp.experts.gate_up_proj")
        return {"ln1": t("input_layernorm.weight"), "ln2": t("post_attention_layernorm.weight"),
                "q": t("self_attn.q_proj.weight"), "k": t("self_attn.k_proj.weight"), "v": t("self_attn.v_proj.weight"),
                "o": t("self_attn.o_proj.weight"), "qn": t("self_attn.q_norm.weight"), "kn": t("self_attn.k_norm.weight"),
                "router": t("mlp.gate.weight"), "gate_proj": gu[:, :self.inter].contiguous(),
                "up_proj": gu[:, self.inter:].contiguous(), "down_proj": t("mlp.experts.down_proj")}


# --------------------------------------------------------------------------- #
# the layer, dtype-generic
# --------------------------------------------------------------------------- #
def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope(cfg, T, device, dtype):
    """cos, sin [T, head_dim] as transformers' default rotary (computed in `dtype`)."""
    hd = cfg.hidden_size // cfg.num_attention_heads
    base = cfg.rope_parameters["rope_theta"] if getattr(cfg, "rope_parameters", None) else cfg.rope_theta
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=dtype, device=device) / hd))
    fr = torch.arange(T, device=device, dtype=dtype)[:, None] * inv[None]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos(), emb.sin()


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def attention(w, x, cfg, cos, sin):
    B, T, H = x.shape
    nh = cfg.num_attention_heads
    hd = H // nh
    q = rms(F.linear(x, w["q"]), w["qn"], cfg.rms_norm_eps).view(B, T, nh, hd).transpose(1, 2)
    k = rms(F.linear(x, w["k"]), w["kn"], cfg.rms_norm_eps).view(B, T, nh, hd).transpose(1, 2)
    v = F.linear(x, w["v"]).view(B, T, nh, hd).transpose(1, 2)
    q = q * cos + _rot(q) * sin
    k = k * cos + _rot(k) * sin
    s = torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5
    mask = torch.full((T, T), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype).triu(1)
    p = torch.softmax(s + mask, dim=-1)
    o = torch.matmul(p, v).transpose(1, 2).reshape(B, T, H)
    return F.linear(o, w["o"])


def route(w, x2d, cfg):
    """transformers' router: softmax in float32 over all experts, top-k, no renormalization (norm_topk_prob=false)."""
    logits = F.linear(x2d, w["router"])
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    val, idx = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
    if cfg.norm_topk_prob:
        raise ValueError("norm_topk_prob is false for this checkpoint")
    return idx, val.to(logits.dtype)


class ExpertHooks:
    """Per (layer, expert) quantization / capture of one expert's projections, in the order:
         xg = qa_x(x) for gate, xu = qa_x(x) for up (each consumer its own copy), g = qa_y(xg @ Wg^T),
         u = qa_y(xu @ Wu^T), a = act(g) * u, d = qa_y(qa_x(a) @ Wd^T), contribution = r * d.
       Weights are replaced by the caller (quantized copies); `act` maps (projection, slot) -> callable or None."""

    def __init__(self):
        self.act = {}            # (layer, expert, proj, slot) -> callable on the tensor
        self.capture = None      # callable(layer, expert, proj, slot, tensor) -> None (taps, may retain grad)


def experts_fixed(w, x2d, idx, rw, l, hooks=None):
    """transformers' eager expert loop with a given (fixed) route: experts in ascending order, each expert's
    (top-k position, token) pairs in that order, index_add into the output."""
    E = w["gate_proj"].shape[0]
    out = torch.zeros_like(x2d)
    mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist()
    for e in hit:
        if e == E:
            continue
        kpos, tok = torch.where(mask[e])
        xe = x2d[tok]
        a = (lambda p, s, t: hooks.act[(l, e, p, s)](t) if hooks and (l, e, p, s) in hooks.act else t)
        if hooks and hooks.capture:
            hooks.capture(l, e, "gate_proj", "x", xe)                         # shared input (total gradient)
        g = a("gate_proj", "y", F.linear(a("gate_proj", "x", xe), w["gate_proj"][e]))
        u = a("up_proj", "y", F.linear(a("up_proj", "x", xe), w["up_proj"][e]))
        h = F.silu(g) * u
        d = a("down_proj", "y", F.linear(a("down_proj", "x", h), w["down_proj"][e]))
        if hooks and hooks.capture:
            for p, s, t in (("gate_proj", "y", g), ("up_proj", "y", u), ("down_proj", "x", h), ("down_proj", "y", d)):
                hooks.capture(l, e, p, s, t)
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_fixed(w, h, cfg, cos, sin, idx, rw, l, hooks=None):
    """One decoder layer with the route (idx [B*T, k], rw [B*T, k]) given; returns (output, moe output)."""
    B, T, H = h.shape
    x = h + attention(w, rms(h, w["ln1"], cfg.rms_norm_eps), cfg, cos, sin)
    m = experts_fixed(w, rms(x, w["ln2"], cfg.rms_norm_eps).reshape(-1, H), idx, rw, l, hooks).view(B, T, H)
    return x + m, m


def layer_dynamic(w, h, cfg, cos, sin):
    """The same layer with its own router (for the route capture); returns (output, idx, rw)."""
    B, T, H = h.shape
    x = h + attention(w, rms(h, w["ln1"], cfg.rms_norm_eps), cfg, cos, sin)
    z = rms(x, w["ln2"], cfg.rms_norm_eps).reshape(-1, H)
    idx, rw = route(w, z, cfg)
    return x + experts_fixed(w, z, idx, rw, None).view(B, T, H), idx, rw


def head(g, h, cfg):
    return F.linear(rms(h, g["norm"], cfg.rms_norm_eps), g["lm_head"])


def summed_ce(logits, ids):
    """Summed next-token CE per window over positions 0..T-2 (float64 accumulation); no router auxiliary loss."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none")
    return ce.view(B, T - 1).double().sum(1)


# --------------------------------------------------------------------------- #
# transformers' layer (dynamic reference)
# --------------------------------------------------------------------------- #
class HFReference:
    """transformers' modules in FP32, one OlmoeDecoderLayer reused for every layer (weights loaded per layer)."""

    def __init__(self, cfg, device):
        from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeRotaryEmbedding
        self.cfg, self.device = cfg, device
        self.layer = OlmoeDecoderLayer(cfg, 0).to(device).eval()
        self.rotary = OlmoeRotaryEmbedding(cfg).to(device)
        for p in self.layer.parameters():
            p.requires_grad_(False)

    def load(self, w):
        L = self.layer
        with torch.no_grad():
            L.input_layernorm.weight.copy_(w["ln1"])
            L.post_attention_layernorm.weight.copy_(w["ln2"])
            for a, b in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")):
                getattr(L.self_attn, a).weight.copy_(w[b])
            L.self_attn.q_norm.weight.copy_(w["qn"])
            L.self_attn.k_norm.weight.copy_(w["kn"])
            L.mlp.gate.weight.copy_(w["router"])
            L.mlp.experts.gate_up_proj.copy_(torch.cat([w["gate_proj"], w["up_proj"]], dim=1))
            L.mlp.experts.down_proj.copy_(w["down_proj"])

    def mask_pos(self, h):
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(h.shape[1], device=h.device)[None]
        mask = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=None, past_key_values=None,
                                  position_ids=pos)
        return mask, pos, self.rotary(h, pos)

    def forward_layer(self, h, mask, pos, pe):
        """Returns (output, moe-block output) of transformers' layer."""
        cap = {}
        hk = self.layer.mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("moe", o))
        try:
            out = self.layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        finally:
            hk.remove()
        return out, cap["moe"]


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
def route_hash(idx, rw):
    import hashlib
    h = hashlib.sha256()
    for a in (idx, rw):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()
