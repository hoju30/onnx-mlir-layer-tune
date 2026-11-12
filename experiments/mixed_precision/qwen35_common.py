"""Qwen3.5-0.8B (Qwen/Qwen3.5-0.8B, text backbone only) for the native mixed-precision experiments.

transformers >= 5 (qwen3_5) lives in qwen35_assets/python_deps, isolated from the other experiments' transformers 4.57
(gpt2_assets/python_deps): import this module before anything that imports transformers.

TextExport is an export adapter: the same parameters and the same mathematics as transformers' Qwen3_5 text model
(eager attention, the torch fallback of the gated delta rule), written with ONNX-exportable, dtype-preserving ops so
that one module gives the FP32 graph that is exported and an FP64 reference that is FP64 throughout (transformers'
own code casts the delta rule, both RMSNorms and the attention softmax to float32). Differences from transformers'
code, all exact in exact arithmetic:
  - the unit lower-triangular solve of the chunked delta rule ((I + A)^-1, A strictly lower) by recursive block
    doubling, M_2s = M_s - M_s E_s M_s (E_s: the A blocks below the diagonal blocks of size s), instead of
    torch.linalg.solve_triangular (not exportable)
  - the in-chunk cumulative sum of the log decay as a product with a constant triangular matrix
  - the per-chunk outputs collected and concatenated instead of written in place
  - the causal mask a constant additive mask (no padding), rotary tables for positions 0..L-1 (text: the three
    M-RoPE position ids are equal)
The Gated DeltaNet's causal depthwise Conv1D, gating, l2-normalization and the recurrent state are kept as in
transformers; the state starts at zero in every window (no cache).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "qwen35_assets"
sys.path.insert(0, str(ASSETS / "python_deps"))
sys.path.insert(1, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402,F401  (5.x, before any other transformers on sys.path)
import tokenizers  # noqa: E402,F401  (pin the versions transformers 5 needs before other paths are inserted)
import huggingface_hub  # noqa: E402,F401
import safetensors  # noqa: E402,F401

MODEL_DIR = ASSETS / "model"
REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
CHUNK = 64


def load_hf(dtype=torch.float32):
    """transformers' model (vision tower included, unused), eager attention, eval, no gradients."""
    from transformers import Qwen3_5ForConditionalGeneration
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=dtype, attn_implementation="eager").eval()
    m.config.use_cache = False
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def hf_logits(model, ids):
    """transformers' own text forward (reference)."""
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def wikitext_ids(split):
    """WikiText-2 raw (the parquet files of the earlier experiments), joined as qwen3_prepare.py does, tokenized with
    this model's tokenizer, no special tokens."""
    import pyarrow.parquet as pq
    text = "\n\n".join(pq.read_table(HERE / "gpt2_assets/data" / f"{split}-00000-of-00001.parquet")["text"].to_pylist())
    return tokenizer().encode(text, add_special_tokens=False)


# --------------------------------------------------------------------------- #
# Export adapter
# --------------------------------------------------------------------------- #
def rms_norm(x, w, eps, zero_centered):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * (1.0 + w) if zero_centered else y * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class TextExport(torch.nn.Module):
    """input_ids [1, L] -> logits [1, L, vocab], L fixed at construction."""

    def __init__(self, hf, seq_len, dtype=torch.float32):
        super().__init__()
        lm = hf.model.language_model
        self.cfg = lm.config
        self.lm = lm
        self.lm_head = hf.lm_head
        self.L = seq_len
        c = self.cfg
        self.eps = c.rms_norm_eps
        # rotary tables (positions 0..L-1; text: the three M-RoPE sections use the same position)
        rot = int(c.head_dim * c.rope_parameters.get("partial_rotary_factor", 1.0))
        inv = 1.0 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
        fr = torch.arange(seq_len, dtype=torch.float64)[:, None] * inv[None]
        emb = torch.cat((fr, fr), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None].to(dtype), persistent=False)       # [1, 1, L, rot]
        self.register_buffer("sin", emb.sin()[None, None].to(dtype), persistent=False)
        fill = torch.finfo(dtype).min
        self.register_buffer("mask", torch.full((seq_len, seq_len), fill, dtype=dtype).triu(1)[None, None],
                             persistent=False)
        C = min(CHUNK, seq_len)
        idx = torch.arange(C)
        self.register_buffer("tri_cum", (idx[:, None] <= idx[None, :]).to(dtype), persistent=False)   # s <= t
        self.register_buffer("upper", (idx[None, :] > idx[:, None])[None, None, None], persistent=False)  # j > i
        self.register_buffer("strict_lower", (idx[:, None] > idx[None, :]).to(dtype), persistent=False)
        self.register_buffer("eye", torch.eye(C, dtype=dtype), persistent=False)
        masks, s = [], 1
        while s < C:
            m = ((idx[:, None] // (2 * s)) == (idx[None, :] // (2 * s))) & ((idx[:, None] // s) % 2 == 1) & \
                ((idx[None, :] // s) % 2 == 0)
            masks.append(m.to(dtype))
            s *= 2
        self.register_buffer("doubling", torch.stack(masks), persistent=False)                 # [log2 C, C, C]
        self.chunk = C

    # gated delta rule, chunked (transformers' torch_chunk_gated_delta_rule with an exportable solve)
    def delta_rule(self, q, k, v, g, beta):
        B, L, H, Dk = (int(d) for d in k.shape)          # Python ints: static shapes in the export
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))                                        # [B, H, L, D]
        beta, g = beta.transpose(1, 2), g.transpose(1, 2)                                       # [B, H, L]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * (Dk ** -0.5)
        C, n = self.chunk, L // self.chunk
        v_beta, k_beta = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
        q, k, k_beta, v_beta = (x.reshape(B, H, n, C, x.shape[-1]) for x in (q, k, k_beta, v_beta))
        cum = g.reshape(B, H, n, C) @ self.tri_cum                                              # in-chunk cumsum
        pair = cum.unsqueeze(-1) - cum.unsqueeze(-2)
        decay = torch.where(self.upper, torch.full_like(pair, float("-inf")), pair).exp()
        A = (k_beta @ k.transpose(-1, -2)) * decay * self.strict_lower
        M = self.eye
        for E in self.doubling:                                                                 # (I + A)^-1
            M = M - M @ (A * E) @ M
        new_values = M @ v_beta
        k_cumdecay = M @ (k_beta * cum.exp().unsqueeze(-1))
        intra = (q @ k.transpose(-1, -2)) * decay
        qd = q * cum.exp().unsqueeze(-1)
        kd = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
        chunk_decay = cum[..., -1].exp()[..., None, None]
        state = torch.zeros(B, H, Dk, v.shape[-1], dtype=v.dtype, device=v.device)
        outs = []
        for i in range(n):
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
            outs.append(qd[:, :, i] @ state + intra[:, :, i] @ v_new)
            state = state * chunk_decay[:, :, i] + kd[:, :, i].transpose(-1, -2) @ v_new
        out = torch.cat(outs, dim=2)                                                            # [B, H, L, Dv]
        return out.transpose(1, 2)                                                              # [B, L, H, Dv]

    def linear_attn(self, m, x):
        B, L = int(x.shape[0]), int(x.shape[1])
        qkv = m.in_proj_qkv(x).transpose(1, 2)
        z = m.in_proj_z(x)
        b, a = m.in_proj_b(x), m.in_proj_a(x)
        K = m.conv1d.weight.shape[-1]
        qkv = F.conv1d(qkv, m.conv1d.weight, None, padding=K - 1, groups=qkv.shape[1])[:, :, :L]
        qkv = F.silu(qkv).transpose(1, 2)
        q, k, v = torch.split(qkv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, L, -1, m.head_k_dim)
        k = k.reshape(B, L, -1, m.head_k_dim)
        v = v.reshape(B, L, -1, m.head_v_dim)
        beta = torch.sigmoid(b)
        g = -m.A_log.exp() * F.softplus(a + m.dt_bias)
        if m.num_v_heads // m.num_k_heads > 1:
            q = q.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            k = k.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        o = self.delta_rule(q, k, v, g, beta)                                                   # [B, L, H, Dv]
        zz = z.reshape(B, L, -1, m.head_v_dim)
        o = rms_norm(o, m.norm.weight, m.norm.variance_epsilon, False) * F.silu(zz)
        return m.out_proj(o.reshape(B, L, -1))

    def full_attn(self, m, x):
        B, L = int(x.shape[0]), int(x.shape[1])
        Dh = m.head_dim
        qg = m.q_proj(x).view(B, L, -1, Dh * 2)
        q, gate = qg[..., :Dh], qg[..., Dh:]                     # torch.chunk, with static slice bounds
        gate = gate.reshape(B, L, -1)
        q = rms_norm(q, m.q_norm.weight, m.q_norm.eps, True).transpose(1, 2)
        k = rms_norm(m.k_proj(x).view(B, L, -1, Dh), m.k_norm.weight, m.k_norm.eps, True).transpose(1, 2)
        v = m.v_proj(x).view(B, L, -1, Dh).transpose(1, 2)
        r = self.cos.shape[-1]
        q = torch.cat([q[..., :r] * self.cos + rotate_half(q[..., :r]) * self.sin, q[..., r:]], dim=-1)
        k = torch.cat([k[..., :r] * self.cos + rotate_half(k[..., :r]) * self.sin, k[..., r:]], dim=-1)
        n = m.num_key_value_groups
        k = k[:, :, None].expand(B, k.shape[1], n, L, Dh).reshape(B, -1, L, Dh)
        v = v[:, :, None].expand(B, v.shape[1], n, L, Dh).reshape(B, -1, L, Dh)
        w = torch.softmax((q @ k.transpose(-1, -2)) * m.scaling + self.mask, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(B, L, -1)
        return m.o_proj(o * torch.sigmoid(gate))

    def layer(self, i, h):
        layer = self.lm.layers[i]
        r = h
        x = rms_norm(h, layer.input_layernorm.weight, self.eps, True)
        if self.cfg.layer_types[i] == "linear_attention":
            x = self.linear_attn(layer.linear_attn, x)
        else:
            x = self.full_attn(layer.self_attn, x)
        h = r + x
        r = h
        x = rms_norm(h, layer.post_attention_layernorm.weight, self.eps, True)
        return r + layer.mlp(x)

    def forward(self, input_ids):
        return self.forward_from(self.lm.embed_tokens(input_ids), 0)

    def hidden_before(self, input_ids, start):
        """The residual stream entering layer `start` (for replays that change only layers >= start)."""
        h = self.lm.embed_tokens(input_ids)
        for i in range(start):
            h = self.layer(i, h)
        return h

    def forward_from(self, h, start):
        for i in range(start, len(self.lm.layers)):
            h = self.layer(i, h)
        return self.lm_head(rms_norm(h, self.lm.norm.weight, self.eps, True))


def summed_ce(logits, ids):
    """Summed next-token cross-entropy over positions 0..L-2 (the native seed)."""
    return F.cross_entropy(logits[0, :-1], ids[1:], reduction="sum")


def projection_nodes(cfg):
    """(layer, kind, parameter suffix) of every searchable projection: linear-attention layers in_proj_qkv (fused q/k/v,
    one node), in_proj_z, in_proj_b, in_proj_a, out_proj; full-attention layers q_proj (fused query and output gate,
    one node), k_proj, v_proj, o_proj; every layer gate_proj, up_proj, down_proj."""
    out = []
    for i, t in enumerate(cfg.layer_types):
        if t == "linear_attention":
            out += [(i, k, f"linear_attn.{k}") for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")]
        else:
            out += [(i, k, f"self_attn.{k}") for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
        out += [(i, k, f"mlp.{k}") for k in ("gate_proj", "up_proj", "down_proj")]
    return out


def map_projection_nodes(graph, hf, cfg, prefix="model.language_model.layers"):
    """{MatMul node name: info} of every searchable projection, found by its weight: the exporter names modules by
    their local name (e.g. /in_proj_qkv_3/MatMul), so the 2-D initializer equal to the parameter's transpose and the
    one MatMul that consumes it identify the node. `graph` must hold its initializers' data."""
    import hashlib
    from onnx import numpy_helper

    def key(a):
        a = np.ascontiguousarray(a, np.float32)
        return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()
    by_hash = {}
    for init in graph.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.ndim == 2:
            by_hash.setdefault(key(a), []).append(init.name)
    consumers = {}
    for n in graph.graph.node:
        if n.op_type == "MatMul":
            consumers.setdefault(n.input[1], []).append(n.name)
    params = dict(hf.named_parameters())
    mapping = {}
    for layer, kind, param in projection_nodes(cfg):
        pname = f"{prefix}.{layer}.{param}.weight"
        wt = params[pname].detach().float().numpy().T
        inits = by_hash.get(key(wt), [])
        hit = [c for i in inits for c in consumers.get(i, [])]
        if len(inits) != 1 or len(hit) != 1:
            raise RuntimeError(f"{pname}: initializers {inits}, MatMuls {hit}")
        mapping[hit[0]] = {"layer": layer, "kind": kind, "param": pname, "block": cfg.layer_types[layer],
                           "weight_initializer": inits[0], "shape_in_out": list(wt.shape)}
    return mapping
