#!/usr/bin/env python3
"""OLMoE-1B-7B fixed-route BC validation, first stage (olmoe_common.py has the definitions).

Question: with the FP32 expert assignment and routing weights fixed (every layer, per window), does the original BC
(C = W + A, unchanged) predict the task-loss damage of quantizing expert projections?

Stages (one --out-dir, resumable; --device cuda for the model passes):
  precheck   resources, versions and hashes, the checkpoint's config and parameter counts      (environment.json)
  tiny       correctness on a tiny random OLMoE: transformers vs this implementation, fixed vs dynamic forward,
             layer-streamed backward vs whole-graph autograd (FP64 throughout), native onnx-mlir forward and
             gradient statistics of the fixed-route selector export vs PyTorch FP64             (correctness.json)
  prepare    windows (model tokenizer, character spans), plan.json (rules frozen before any route or loss)
  capture    FP32 dynamic forward per window -> route caches (ids, weights) + manifest; real-model forward checks
  coverage   expert coverage on the calibration routes, the nine experts by the plan's rule  (expert_coverage.csv)
  bcheck     real model: layer-streamed backward vs whole-graph autograd over layers 13-15 (one window)
  tables     BC of the selected experts' projections (PyTorch, fixed route, calibration): table.json, ranges.json
  freeze     candidates (--modes), BC, resources, hashes                                          (freeze.json)
  truth      quantized fixed-route replay: diagnostic (and calibration) per-window CE changes
  analyze    ranking, pairs, single vs joint, coverage groups                           (analysis.json, pairs.csv)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import olmoe_common as oc  # noqa: F401  (transformers 5 first)
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SEQ = 128
FORMATS = ("int4", "int6", "int8", "p6e1", "p8e1")
DATA = {"calibration": ("validation", 0, 10), "diagnostic": ("validation", 1280, 30), "check": ("test", 0, 2)}


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n")


def jload(path):
    return json.loads(path.read_text())


def sha256(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def gpu_exact(device):
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def head_ce(g, h, toks, cfg, chunk=10):
    """Summed next-token CE per window, the LM head applied `chunk` windows at a time (memory)."""
    return torch.cat([oc.summed_ce(oc.head(g, h[i:i + chunk], cfg), toks[i:i + chunk]) for i in range(0, len(toks), chunk)])


# --------------------------------------------------------------------------- #
# precheck
# --------------------------------------------------------------------------- #
def stage_precheck(args, out, log):
    import shutil
    import subprocess
    import transformers
    cfg = oc.config()
    expect = {"num_hidden_layers": 16, "num_experts": 64, "num_experts_per_tok": 8, "norm_topk_prob": False}
    got = {k: getattr(cfg, k) for k in expect}
    if got != expect:
        raise SystemExit(f"config differs from the expected official one: {got}")
    ck = oc.Checkpoint()
    counts, dtypes = {}, {}
    for name, f in ck.map.items():
        s = ck.handles[f].get_slice(name)
        n = int(np.prod(s.get_shape()))
        grp = ("experts" if ".mlp.experts." in name else "router" if ".mlp.gate." in name else
               "attention" if "self_attn" in name else "embedding" if "embed" in name else "lm_head" if "lm_head" in name
               else "norms")
        counts[grp] = counts.get(grp, 0) + n
        dtypes[s.get_dtype()] = dtypes.get(s.get_dtype(), 0) + n
    root = HERE.parent.parent
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    mem = {k: int(v.split()[0]) / 2 ** 20 for k, v in (l.split(":") for l in Path("/proc/meminfo").read_text().splitlines()[:3])}
    res = {"model": {"repo": oc.REPO, "revision": oc.REVISION, "config": cfg.to_dict(),
                     "files_sha256": {f: sha256(oc.MODEL_DIR / f) for f in sorted(ck.handles) + ["config.json", "tokenizer.json"]},
                     "download_bytes": sum((oc.MODEL_DIR / f).stat().st_size for f in ck.handles),
                     "parameters": counts, "parameters_total": sum(counts.values()), "dtypes": dtypes,
                     "fp32_bytes_total": 4 * sum(counts.values()),
                     "quantized_scope": "expert gate/up/down only; router, attention, embedding, LM head, norms FP32",
                     "expert_parameters_per_layer": counts["experts"] // cfg.num_hidden_layers},
           "resources": {"ram_gb": mem, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "gpu_mem_gb": [x / 2 ** 30 for x in torch.cuda.mem_get_info()] if torch.cuda.is_available() else None,
                         "disk_free_gb": shutil.disk_usage(HERE).free / 2 ** 30, "cpus": __import__("os").cpu_count(),
                         "backends": {"pytorch_cuda": torch.cuda.is_available(), "onnx_mlir": True}},
           "plan_for_memory": "FP32 weights 27.7 GB do not fit the GPU (11.6 GB) and barely the RAM: layers are streamed "
                              "(BF16 read -> FP32 cast, exact); backward recomputes each layer from its stored input",
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__},
           "git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--short", "--untracked-files=no"),
           "code_sha256": {f: sha256(HERE / f) for f in ("olmoe_common.py", "run_olmoe_fixed_route.py", "qwen3_swiglu_common.py",
                                                         "qwen35_tiny_check.py", "run_native_bc_pipeline.py")}}
    jdump(out / "environment.json", res)
    log(f"precheck: config ok {got}; params {res['model']['parameters_total']:,} ({counts}); dtypes {dtypes}")


# --------------------------------------------------------------------------- #
# tiny-model correctness
# --------------------------------------------------------------------------- #
def tiny_model(seed=0):
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(seed)
    cfg = OlmoeConfig(vocab_size=256, hidden_size=64, intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, num_experts=8, num_experts_per_tok=2, norm_topk_prob=False,
                      rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    m = OlmoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.15)
            else:
                p.normal_(1.0, 0.1)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def forward_stream(src, cfg, ids, device, dtype, routes=None, hooks=None, keep_h=False):
    """Whole model, layer by layer. routes None: own router (dynamic), returns the routes; else fixed.
    Returns logits, routes, list of layer inputs (keep_h) and MoE outputs."""
    g = src.global_(device, dtype)
    B, T = ids.shape
    h = F.embedding(ids, g["embed"])
    cos, sin = oc.rope(cfg, T, device, dtype)
    rs, hs, moes = [], [], []
    for l in range(cfg.num_hidden_layers):
        w = src.layer(l, device, dtype, cfg.num_experts)
        if keep_h:
            hs.append(h)
        if routes is None:
            x = h + oc.attention(w, oc.rms(h, w["ln1"], cfg.rms_norm_eps), cfg, cos, sin)
            z = oc.rms(x, w["ln2"], cfg.rms_norm_eps).reshape(-1, h.shape[-1])
            idx, rw = oc.route(w, z, cfg)
            m = oc.experts_fixed(w, z, idx, rw, l, hooks).view(h.shape)
            h = x + m
            rs.append((idx, rw))
        else:
            idx, rw = routes[l]
            h, m = oc.layer_fixed(w, h, cfg, cos, sin, idx, rw, l, hooks)
        moes.append(m)
        del w
    if keep_h:
        hs.append(h)
    return oc.head(g, h, cfg), rs, hs, moes


def tiny_fixed_export(src, cfg, routes, T):
    """The fixed-route tiny model as an exportable module: each layer's experts read their routed rows through a
    constant one-hot selector S_e [n_e, T] (MatMul; exact) and write back through S_e^T after scaling by the constant
    routing weights, experts in ascending order (transformers' accumulation order)."""
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            g = src.global_("cpu", torch.float32)
            self.embed = nn.Parameter(g["embed"], requires_grad=False)
            self.norm = nn.Parameter(g["norm"], requires_grad=False)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(g["lm_head"], requires_grad=False)
            cos, sin = oc.rope(cfg, T, "cpu", torch.float32)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            self.layers = nn.ModuleList()
            self.plan = []
            for l in range(cfg.num_hidden_layers):
                w = src.layer(l, "cpu", torch.float32)
                L = nn.Module()
                for k in ("ln1", "ln2", "qn", "kn"):
                    setattr(L, k, nn.Parameter(w[k], requires_grad=False))
                for k in ("q", "k", "v", "o"):
                    lin = nn.Linear(w[k].shape[1], w[k].shape[0], bias=False)
                    lin.weight = nn.Parameter(w[k], requires_grad=False)
                    setattr(L, k, lin)
                idx, rw = routes[l]
                E = cfg.num_experts
                mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
                hit = [e for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist() if e != E]
                steps = []
                for e in hit:
                    kpos, tok = torch.where(mask[e])
                    S = torch.zeros(len(tok), T)
                    S[torch.arange(len(tok)), tok] = 1.0
                    L.register_buffer(f"S{e}", S)
                    L.register_buffer(f"St{e}", S.t().contiguous())
                    L.register_buffer(f"r{e}", rw[tok, kpos][:, None].float().contiguous())
                    for p in oc.PROJ:
                        lin = nn.Linear(w[p].shape[2], w[p].shape[1], bias=False)
                        lin.weight = nn.Parameter(w[p][e].clone(), requires_grad=False)
                        setattr(L, f"{p}{e}", lin)
                    steps.append(e)
                self.plan.append(steps)
                self.layers.append(L)

        def forward(self, ids):
            h = F.embedding(ids, self.embed)
            T, H = ids.shape[1], self.embed.shape[1]
            for L, steps in zip(self.layers, self.plan):
                w = {"q": L.q.weight, "k": L.k.weight, "v": L.v.weight, "o": L.o.weight, "qn": L.qn, "kn": L.kn}
                x = h + oc.attention(w, oc.rms(h, L.ln1, cfg.rms_norm_eps), cfg, self.cos, self.sin)
                z = oc.rms(x, L.ln2, cfg.rms_norm_eps).reshape(T, H)
                m = torch.zeros(T, H)
                for e in steps:
                    xe = torch.matmul(getattr(L, f"S{e}"), z)
                    a = F.silu(getattr(L, f"gate_proj{e}")(xe)) * getattr(L, f"up_proj{e}")(xe)
                    d = getattr(L, f"down_proj{e}")(a)
                    m = m + torch.matmul(getattr(L, f"St{e}"), d * getattr(L, f"r{e}"))
                h = x + m.view(1, T, H)
            return self.lm_head(oc.rms(h, self.norm, cfg.rms_norm_eps))
    return M().eval()


def map_expert_nodes(graph, src, cfg):
    """{MatMul node: (layer, expert, proj)} found by the transposed weight (exporter names are not unique)."""
    import hashlib
    from onnx import numpy_helper
    key = lambda a: hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes() + str(a.shape).encode()).hexdigest()
    by = {}
    for i in graph.graph.initializer:
        a = numpy_helper.to_array(i)
        if a.ndim == 2:
            by.setdefault(key(a), []).append(i.name)
    cons = {}
    for n in graph.graph.node:
        if n.op_type == "MatMul":
            cons.setdefault(n.input[1], []).append(n.name)
    out = {}
    for l in range(cfg.num_hidden_layers):
        w = src.layer(l, "cpu", torch.float32)
        for e in range(cfg.num_experts):
            for p in oc.PROJ:
                hit = [c for i in by.get(key(w[p][e].numpy().T), []) for c in cons.get(i, [])]
                if len(hit) == 1:
                    out[hit[0]] = {"layer": l, "expert": e, "proj": p}
    return out


def torch_tap_stats(src, cfg, ids, routes, dtype, sel=None):
    """Per window: tap (expert, proj, slot) per-channel sum g^2 and sum g over the routed rows (x: the expert input,
    total gradient over gate and up), and per weight the per-output-channel sum dW^2; whole-graph autograd."""
    cap = {}

    class H(oc.ExpertHooks):
        pass
    hk = H()

    def capture(l, e, p, s, t):
        if sel is None or (l, e) in sel:
            if t.requires_grad:
                t.retain_grad()
            cap[(l, e, p, s)] = t
    hk.capture = capture
    g = src.global_("cpu", dtype)
    g["embed"].requires_grad_(True)              # so that every layer's expert inputs carry gradients
    layers = [src.layer(l, "cpu", dtype) for l in range(cfg.num_hidden_layers)]
    for w in layers:
        for p in oc.PROJ:
            w[p].requires_grad_(True)

    class Src:
        def global_(self, d, t):
            return g

        def layer(self, l, d, t, n=None):
            return layers[l]
    x = torch.from_numpy(ids)[None]
    logits, _, _, _ = forward_stream(Src(), cfg, x, "cpu", dtype, routes=routes, hooks=hk)
    loss = oc.summed_ce(logits, x).sum()
    loss.backward()
    st = {}
    for (l, e, p, s), t in cap.items():
        gr = t.grad.double()
        st[(l, e, p, s)] = ((gr ** 2).sum(0).numpy(), gr.sum(0).numpy())
        if p == "gate_proj" and s == "x":
            st[(l, e, "up_proj", "x")] = st[(l, e, p, s)]
    for l, w in enumerate(layers):
        for p in oc.PROJ:
            for e in range(cfg.num_experts):
                if sel is None or (l, e) in sel:
                    st[(l, e, p, "w")] = (w[p].grad[e].double() ** 2).sum(1).numpy()
    return logits.detach(), st, float(loss), {k: (t.detach(), t.grad.detach()) for k, t in cap.items()}, \
        {(l, p): w[p].grad.detach().clone() for l, w in enumerate(layers) for p in oc.PROJ}


def streamed_tap_stats(src, cfg, ids, routes, dtype, device="cpu"):
    """The same with the layer-streamed backward (forward storing each layer's input, then per layer: recompute with
    gradients and backpropagate the next layer's input gradient)."""
    x = torch.from_numpy(ids)[None].to(device)
    with torch.no_grad():
        logits, _, hs, _ = forward_stream(src, cfg, x, device, dtype, routes=routes, keep_h=True)
    g = src.global_(device, dtype)
    hT = hs[-1].clone().requires_grad_(True)
    loss = oc.summed_ce(oc.head(g, hT, cfg), x).sum()
    loss.backward()
    gnext = hT.grad
    cos, sin = oc.rope(cfg, x.shape[1], device, dtype)
    caps, wgrads = {}, {}
    for l in reversed(range(cfg.num_hidden_layers)):
        w = src.layer(l, device, dtype, cfg.num_experts)
        for p in oc.PROJ:
            w[p].requires_grad_(True)
        hk = oc.ExpertHooks()

        def capture(ll, e, p, s, t):
            if t.requires_grad:
                t.retain_grad()
            caps[(ll, e, p, s)] = t
        hk.capture = capture
        hin = hs[l].clone().requires_grad_(True)
        out, _ = oc.layer_fixed(w, hin, cfg, cos, sin, routes[l][0], routes[l][1], l, hk)
        out.backward(gnext)
        gnext = hin.grad
        for p in oc.PROJ:
            wgrads[(l, p)] = w[p].grad.detach().clone()
    return {k: (t.detach(), t.grad.detach()) for k, t in caps.items()}, wgrads, gnext


def stage_tiny(args, out, log):
    from transformers import OlmoeForCausalLM  # noqa: F401
    res = jload(out / "correctness.json") if (out / "correctness.json").exists() else {}
    m, cfg = tiny_model()
    src = oc.StateDict(m)
    rng = np.random.default_rng(0)
    ids = [rng.integers(0, cfg.vocab_size, SEQ).astype(np.int64) for _ in range(2)]
    t0 = time.time()
    r = {}
    # (1) transformers whole model vs this implementation (dynamic routes, FP32), and the HF layer composition
    x = torch.from_numpy(np.stack(ids))
    with torch.no_grad():
        hf = m(input_ids=x, use_cache=False).logits
        mine, routes2, _, _ = forward_stream(src, cfg, x, "cpu", torch.float32)
        ref = oc.HFReference(cfg, "cpu")
        g = src.global_("cpu", torch.float32)
        h = F.embedding(x, g["embed"])
        mask, pos, pe = ref.mask_pos(h)
        for l in range(cfg.num_hidden_layers):
            ref.load(src.layer(l, "cpu", torch.float32))
            h, _ = ref.forward_layer(h, mask, pos, pe)
        comp = oc.head(g, h, cfg)
    r["hf_model_vs_hf_layer_composition_rel"] = rel(comp, hf)
    r["hf_model_vs_this_dynamic_rel"] = rel(mine, hf)
    # (2) fixed route (cached FP32 routes) vs dynamic
    per_window = []
    for k, ids_k in enumerate(ids):
        xk = torch.from_numpy(ids_k)[None]
        with torch.no_grad():
            dyn, rk, _, md = forward_stream(src, cfg, xk, "cpu", torch.float32)
            fix, _, _, mf = forward_stream(src, cfg, xk, "cpu", torch.float32, routes=rk)
        per_window.append(rk)
        r.setdefault("fixed_vs_dynamic_logits_rel", []).append(rel(fix, dyn))
        r.setdefault("fixed_vs_dynamic_moe_rel_max_layer", []).append(max(rel(a, b) for a, b in zip(mf, md)))
        r.setdefault("route_weights_require_grad", []).append(any(rw.requires_grad for _, rw in rk))
    # route caches are per window: window 1's route on window 0 must differ
    with torch.no_grad():
        swapped, _, _, _ = forward_stream(src, cfg, torch.from_numpy(ids[0])[None], "cpu", torch.float32, routes=per_window[1])
        own, _, _, _ = forward_stream(src, cfg, torch.from_numpy(ids[0])[None], "cpu", torch.float32, routes=per_window[0])
    r["swapped_route_changes_logits_rel"] = rel(swapped, own)
    log(f"tiny forward: HF vs layer composition {r['hf_model_vs_hf_layer_composition_rel']:.2e}, HF vs this "
        f"{r['hf_model_vs_this_dynamic_rel']:.2e}, fixed vs dynamic {max(r['fixed_vs_dynamic_logits_rel']):.2e}, "
        f"swapped route {r['swapped_route_changes_logits_rel']:.2e}")
    # (3) backward: FP64 whole-graph autograd vs FP64 layer-streamed; FP32 streamed vs FP64
    bw = []
    for k, ids_k in enumerate(ids):
        rk64 = [(i, w.double()) for i, w in per_window[k]]
        _, _, _, caps_full, wg_full = torch_tap_stats(src, cfg, ids_k, rk64, torch.float64)
        caps_st, wg_st, _ = streamed_tap_stats(src, cfg, ids_k, rk64, torch.float64)
        caps32, wg32, _ = streamed_tap_stats(src, cfg, ids_k, per_window[k], torch.float32)
        e64 = [rel_max(caps_st[c][1].numpy(), caps_full[c][1].numpy()) for c in caps_full] + \
              [rel_max(wg_st[c].numpy(), wg_full[c].numpy()) for c in wg_full]
        e32 = [rel_max(caps32[c][1].numpy(), caps_full[c][1].numpy()) for c in caps_full] + \
              [rel_max(wg32[c].numpy(), wg_full[c].numpy()) for c in wg_full]
        bw.append({"streamed_fp64_vs_whole_fp64_max": max(e64), "streamed_fp32_vs_whole_fp64_median": float(np.median(e32)),
                   "streamed_fp32_vs_whole_fp64_max": max(e32), "arrays": len(e64),
                   "dtype_check": str({t.dtype for t, _ in caps_full.values()})})
    r["backward"] = bw
    log("tiny backward: " + "; ".join(f"FP64 streamed vs whole max {b['streamed_fp64_vs_whole_fp64_max']:.1e}, FP32 vs FP64 "
                                      f"median {b['streamed_fp32_vs_whole_fp64_median']:.1e} max {b['streamed_fp32_vs_whole_fp64_max']:.1e}"
                                      for b in bw))
    # (4) native: fixed-route selector export per window, forward + gradient statistics vs PyTorch FP64
    if not args.skip_native:
        import onnx
        from qwen3_prepare import fold_constants
        from qwen35_tiny_check import native_stats
        from run_native_bc_pipeline import erm
        tools = erm.Tools()
        nat = []
        for k, ids_k in enumerate(ids):
            d = out / "tiny_native" / f"w{k}"
            d.mkdir(parents=True, exist_ok=True)
            mod = tiny_fixed_export(src, cfg, per_window[k], SEQ)
            xk = torch.from_numpy(ids_k)[None]
            with torch.no_grad():
                ex = mod(xk)
                ref32, _, _, _ = forward_stream(src, cfg, xk, "cpu", torch.float32, routes=per_window[k])
            torch.onnx.export(mod, (xk,), str(d / "raw.onnx"), input_names=["input_ids"], output_names=["logits"],
                              opset_version=17, dynamo=False, do_constant_folding=True)
            g_ = onnx.load(str(d / "raw.onnx"))
            fold_constants(g_)
            mapping = map_expert_nodes(g_, src, cfg)
            onnx.save(g_, str(d / "tiny.onnx"))
            stats, z, layout, info = native_stats(d / "native", d / "tiny.onnx", [ids_k], cfg.vocab_size, mapping, tools, erm)
            rk64 = [(i, w.double()) for i, w in per_window[k]]
            z64, st64, _, _, _ = torch_tap_stats(src, cfg, ids_k, rk64, torch.float64)
            errs = {}
            for node, info_ in mapping.items():
                key = (info_["layer"], info_["expert"], info_["proj"])
                errs[f"{node}|w"] = rel_max(stats[0][(node, "w")], st64[(*key, "w")])
                for s in ("x", "y"):
                    if (node, s) in stats[0]:
                        errs[f"{node}|{s}|g2"] = rel_max(stats[0][(node, s)][0], st64[(*key, s)][0])
                        errs[f"{node}|{s}|g"] = rel_max(stats[0][(node, s)][1], st64[(*key, s)][1])
            e = np.array(list(errs.values()))
            nat.append({"export_vs_fixed_route_torch_fp32_rel": rel(ex, ref32),
                        "native_logits_vs_fp64_rel": float(np.linalg.norm(z[0] - z64[0].numpy()) / np.linalg.norm(z64[0].numpy())),
                        "mapped_expert_projections": len(mapping),
                        "expected_expert_projections": int(sum(len(set(i.reshape(-1).tolist())) for i, _ in per_window[k]) * 3),
                        "arrays": len(e), "rel_max_median": float(np.median(e)), "rel_max_max": float(e.max()),
                        "worst": sorted(errs.items(), key=lambda kv: -kv[1])[:4],
                        "layout": {"weights": len(layout["weights"]), "taps": len(layout["taps"]), "points": len(layout["points"])},
                        "ops": dict(__import__("collections").Counter(n.op_type for n in g_.graph.node))})
            log(f"tiny native w{k}: export vs torch {nat[-1]['export_vs_fixed_route_torch_fp32_rel']:.1e}, logits vs FP64 "
                f"{nat[-1]['native_logits_vs_fp64_rel']:.1e}, {len(mapping)} projections, stats median "
                f"{nat[-1]['rel_max_median']:.1e} max {nat[-1]['rel_max_max']:.1e}")
        r["native"] = nat
        allm = [n["rel_max_max"] for n in nat]
        r["native_pass"] = bool(all(n["rel_max_median"] <= 1e-3 for n in nat) and max(allm) <= 5e-2
                                and all(n["native_logits_vs_fp64_rel"] <= 1e-4 for n in nat))
    r["pass"] = bool(r["hf_model_vs_hf_layer_composition_rel"] <= 1e-6 and r["hf_model_vs_this_dynamic_rel"] <= 1e-5
                     and max(r["fixed_vs_dynamic_logits_rel"]) <= 1e-6 and r["swapped_route_changes_logits_rel"] > 1e-3
                     and not any(r["route_weights_require_grad"])
                     and all(b["streamed_fp64_vs_whole_fp64_max"] <= 1e-10 for b in bw)
                     and all(b["streamed_fp32_vs_whole_fp64_median"] <= 1e-3 for b in bw)
                     and r.get("native_pass", True))
    r["seconds"] = time.time() - t0
    res["tiny"] = r
    jdump(out / "correctness.json", res)
    log(f"tiny pass {r['pass']}")


# --------------------------------------------------------------------------- #
# windows, plan
# --------------------------------------------------------------------------- #
def stage_prepare(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    tok = oc.tokenizer()
    data = {}
    for name, (split, first, n) in DATA.items():
        t, spans, ntok, nchar = oc.windows(split, first, n, SEQ, tok)
        np.save(out / f"tokens_{name}.npy", t)
        data[name] = {"split": split, "tokens": [first, first + n * SEQ], "windows": n, "char_spans": spans,
                      "split_tokens": ntok, "split_chars": nchar, "sha256": sha256(out / f"tokens_{name}.npy")}
    for a, b in (("calibration", "diagnostic"),):
        if data[a]["char_spans"][-1][1] > data[b]["char_spans"][0][0]:
            raise SystemExit("calibration and diagnostic texts overlap")
    jdump(out / "plan.json", {
        "question": "with the FP32 expert ids and routing weights fixed in every layer, does the original BC predict the "
                    "task-loss damage of quantizing expert projections?",
        "model": {"repo": oc.REPO, "revision": oc.REVISION},
        "fixed_route": "per window, the FP32 dynamic forward's top-8 ids and (unnormalized, float32) routing weights of "
                       "every layer are cached and used as constants (detached) in every fixed-route forward; hidden "
                       "states and expert inputs / outputs are recomputed; no top-k re-selection or renormalization",
        "loss": "summed next-token cross-entropy over positions 0..126 of each 128-token window; router auxiliary loss "
                "excluded; gradients of the fixed-route model",
        "data": {"source": "WikiText-2 raw parquet (gpt2_assets/data), lines joined with blank lines; the model's own "
                           "tokenizer (GPT-NeoX BPE of OLMoE), no special tokens; spans are character offsets in that text",
                 **data,
                 "note": "token offsets of other tokenizers (earlier Qwen / GPT-2 studies) do not prove text disjointness; "
                         "within this study calibration and diagnostic character spans are disjoint; the check windows "
                         "(test split) serve correctness only"},
        "scope": {"quantized": "expert gate_proj, up_proj, down_proj of the selected experts only",
                  "fp32": "router, attention, embedding, LM head, norms, every other expert"},
        "expert_selection": {
            "layers": [0, 8, 15],
            "rule": "per layer, experts with >= 1 routed calibration token, sorted by (routed tokens desc, expert id asc); "
                    "positions q*(n-1) rounded half up for q = 0.05 (high), 0.5 (mid), 0.95 (low); if two positions "
                    "coincide the later one moves one place down; calibration routes only",
            "low_coverage_flag": "routed calibration tokens < 32 or present in < 3 windows: reported as coverage "
                                 "diagnostics, not pooled with sufficiently covered experts"},
        "bc": {"formula": "C(w, a) = W(w) + A(a), unchanged: W = sum_c (G_c / n_in) sum_in dW^2 with G_c = sum over "
                          "calibration windows of the per-window per-output-channel sum of (dL/dW)^2; A = sum over the "
                          "node's input and output taps of sum_c [G2_c / n (s2_c - n mu_c^2) + GS_c mu_c^2]",
               "moe_counts": "n = the expert's routed rows over the calibration windows (no padding); G2 = sum of g^2 "
                             "over routed rows; GS = sum over windows of (sum of g over the window's routed rows)^2; no "
                             "division by activation counts, no extra routing-probability factor",
               "shared_input": "gate and up read the same expert input rows; its tap uses their total gradient (as "
                               "input_grad='total'); each consumer quantizes its own copy",
               "qdq_order": "x -> QDQ_a (per consumer) -> W_q -> QDQ_a(y) for gate / up; down: QDQ_a(silu(g)*u) -> "
                            "W_q -> QDQ_a(y); then multiply by the fixed routing weight; then index_add",
               "implementation": "PyTorch (fixed-route, layer-streamed, FP32 on the GPU); the statistics' semantics "
                                 "checked natively on a tiny OLMoE (correctness.json)",
               "unobserved": "an expert without routed calibration rows has no statistics: no score (never a free option)"},
        "candidates": {"formats": list(FORMATS), "modes": ["w (weights only; first)", "a (activations only)",
                                                           "wa (both, same format)"],
                       "per_expert": "gate / up / down alone x 5 formats, gate+up every format pair (25)",
                       "activation_modes_if": "the activation statistics check (gathered vs dense-masked recomputation "
                                              "of ranges and error sums) passes; otherwise not run"},
        "truth": "dNLL = NLL(quantized, fixed route) - NLL(FP32, same fixed route), per window, replayed from the "
                 "candidate's layer with every layer's route fixed (GPU FP32, TF32 off)",
        "analysis": {"pairs": "equal M and BOPs within (layer, expert, mode); decisive = 95% of 2000 paired window "
                              "bootstraps agree; the same resample indices for every candidate",
                     "groups": ["gate vs up same format", "same branch int vs posit", "gate+up swaps",
                                "coverage group high / mid / low", "layer"],
                     "single_vs_joint": ["BC_gate + BC_up", "diagnostic measured gate-only + up-only (same-window "
                                         "oracle)", "calibration measured singles -> diagnostic joint (reported apart)"]},
        "no_feedback": "diagnostic routes and losses are used for evaluation only (not ranges, BC, candidates or experts)",
        "seeds": {"bootstrap": 0}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("plan: " + ", ".join(f"{k} {v['split']} tokens {v['tokens']}" for k, v in data.items()))


# --------------------------------------------------------------------------- #
# route capture and real-model forward checks
# --------------------------------------------------------------------------- #
def stage_capture(args, out, log):
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    rc = out / "routes"
    rc.mkdir(exist_ok=True)
    man = jload(out / "routes_manifest.json") if (out / "routes_manifest.json").exists() else {}
    res = jload(out / "correctness.json") if (out / "correctness.json").exists() else {}
    for name in ("check", "calibration", "diagnostic"):
        if name in man:
            continue
        t0 = time.time()
        toks = torch.from_numpy(np.load(out / f"tokens_{name}.npy")).to(dev)
        B = toks.shape[0]
        g = ck.global_(dev, torch.float32)
        h = F.embedding(toks, g["embed"])
        cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
        idxs, rws = [], []
        check = {"moe_rel": [], "layer_out_rel": []} if name == "check" else None
        if check is not None:
            ref = oc.HFReference(cfg, dev)
            hr = h.clone()
            mask, pos, pe = ref.mask_pos(hr)
        with torch.no_grad():
            for l in range(cfg.num_hidden_layers):
                w = ck.layer(l, dev, torch.float32)
                hn, idx, rw = oc.layer_dynamic(w, h, cfg, cos, sin)
                if check is not None:          # transformers' layer on the same input, and its own (dynamic) route
                    ref.load(w)
                    ho, mo = ref.forward_layer(h, mask, pos, pe)
                    _, mf = oc.layer_fixed(w, h, cfg, cos, sin, idx, rw, l)
                    check["moe_rel"].append(rel(mf, mo))
                    check["layer_out_rel"].append(rel(hn, ho))
                    hr, _ = ref.forward_layer(hr, mask, pos, pe)
                h = hn
                idxs.append(idx.view(B, SEQ, -1).to(torch.int16).cpu().numpy())
                rws.append(rw.view(B, SEQ, -1).float().cpu().numpy())
                del w
            logits_dyn = oc.head(g, h, cfg)
            ce_dyn = oc.summed_ce(logits_dyn, toks).cpu().numpy()
        idx_a, rw_a = np.stack(idxs, 1), np.stack(rws, 1)          # [window, layer, T, k]
        for k in range(B):
            np.savez(rc / f"{name}_{k:03d}.npz", ids=idx_a[k], weights=rw_a[k], tokens=np.load(out / f"tokens_{name}.npy")[k])
        entry = {"windows": B, "shape_ids": list(idx_a.shape), "dtype_ids": "int16", "dtype_weights": "float32",
                 "hash": [oc.route_hash(idx_a[k], rw_a[k]) for k in range(B)], "tokens_sha256": sha256(out / f"tokens_{name}.npy"),
                 "model_revision": oc.REVISION, "seconds": time.time() - t0, "summed_ce_dynamic": ce_dyn.tolist()}
        # fixed-route replay of the same windows from the caches
        with torch.no_grad():
            h = F.embedding(toks, g["embed"])
            for l in range(cfg.num_hidden_layers):
                w = ck.layer(l, dev, torch.float32)
                idx = torch.from_numpy(idx_a[:, l].reshape(-1, cfg.num_experts_per_tok).astype(np.int64)).to(dev)
                rw = torch.from_numpy(rw_a[:, l].reshape(-1, cfg.num_experts_per_tok)).to(dev)
                h, _ = oc.layer_fixed(w, h, cfg, cos, sin, idx, rw, l)
                del w
            logits_fix = oc.head(g, h, cfg)
            ce_fix = oc.summed_ce(logits_fix, toks).cpu().numpy()
        entry["fixed_vs_dynamic_logits_rel"] = rel(logits_fix, logits_dyn)
        entry["fixed_vs_dynamic_ce_abs_max"] = float(np.abs(ce_fix - ce_dyn).max())
        entry["summed_ce_fixed"] = ce_fix.tolist()
        if check is not None:
            with torch.no_grad():
                logits_hf = oc.head(g, hr, cfg)
            check["hf_layers_vs_this_dynamic_logits_rel"] = rel(logits_dyn, logits_hf)
            check["hf_ce"] = oc.summed_ce(logits_hf, toks).cpu().numpy().tolist()
            check["this_ce"] = ce_dyn.tolist()
            check["route_differs_between_windows"] = bool((idx_a[0] != idx_a[1]).any())
            res["real_forward"] = check
            jdump(out / "correctness.json", res)
            log(f"check: HF layers vs this {check['hf_layers_vs_this_dynamic_logits_rel']:.2e}, MoE rel max "
                f"{max(check['moe_rel']):.2e}, layer out rel max {max(check['layer_out_rel']):.2e}")
        man[name] = entry
        jdump(out / "routes_manifest.json", man)
        log(f"{name}: {B} windows routes cached, fixed vs dynamic logits {entry['fixed_vs_dynamic_logits_rel']:.2e}, "
            f"CE |diff| max {entry['fixed_vs_dynamic_ce_abs_max']:.2e} ({entry['seconds']:.0f}s)")


def load_routes(out, name, cfg, dev, dtype=torch.float32):
    """[(idx [B*T, k] long, rw [B*T, k]) per layer] for every window of the set, checked against the manifest."""
    man = jload(out / "routes_manifest.json")[name]
    ids, ws = [], []
    for k in range(man["windows"]):
        z = np.load(out / "routes" / f"{name}_{k:03d}.npz")
        if oc.route_hash(z["ids"], z["weights"]) != man["hash"][k]:
            raise SystemExit(f"route cache {name} {k} does not match its manifest")
        ids.append(z["ids"])
        ws.append(z["weights"])
    ids, ws = np.stack(ids), np.stack(ws)              # [B, L, T, k]
    return [(torch.from_numpy(ids[:, l].reshape(-1, cfg.num_experts_per_tok).astype(np.int64)).to(dev),
             torch.from_numpy(ws[:, l].reshape(-1, cfg.num_experts_per_tok)).to(dev, dtype))
            for l in range(cfg.num_hidden_layers)], ids, ws


# --------------------------------------------------------------------------- #
# coverage and expert selection
# --------------------------------------------------------------------------- #
def stage_coverage(args, out, log):
    import csv
    cfg = oc.config()
    plan = jload(out / "plan.json")
    _, ids, ws = load_routes(out, "calibration", cfg, "cpu")
    B, L, T, K = ids.shape
    rows, sel = [], {}
    for l in range(L):
        cnt = {}
        for e in range(cfg.num_experts):
            m = ids[:, l] == e                                  # [B, T, K]
            n = int(m.sum())
            wv = ws[:, l][m]
            posn = np.nonzero(m.any(-1))[1] if n else np.array([])
            rows.append({"layer": l, "expert": e, "routed_tokens": n, "windows": int(m.any((1, 2)).sum()),
                         "weight_mean": float(wv.mean()) if n else None, "weight_p10": float(np.percentile(wv, 10)) if n else None,
                         "weight_p50": float(np.percentile(wv, 50)) if n else None, "weight_p90": float(np.percentile(wv, 90)) if n else None,
                         "weight_max": float(wv.max()) if n else None, "position_mean": float(posn.mean()) if n else None,
                         "share_positions_0_7": float((posn < 8).mean()) if n else None,
                         "share_top_window": float(m.any(-1).sum(1).max() / max(n, 1)) if n else None})
            cnt[e] = n
        if l in plan["expert_selection"]["layers"]:
            cov = sorted([e for e in cnt if cnt[e] >= 1], key=lambda e: (-cnt[e], e))
            n = len(cov)
            picks, used = {}, set()
            for name, q in (("high", 0.05), ("mid", 0.5), ("low", 0.95)):
                i = int(np.floor(q * (n - 1) + 0.5))
                while i in used:
                    i += 1
                used.add(i)
                e = cov[i]
                r = next(x for x in rows if x["layer"] == l and x["expert"] == e)
                picks[name] = {"expert": e, "rank": i, "routed_tokens": cnt[e], "windows": r["windows"],
                               "low_coverage": bool(cnt[e] < 32 or r["windows"] < 3)}
            sel[str(l)] = {"covered_experts": n, "picks": picks}
    with open(out / "expert_coverage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = np.array([r["routed_tokens"] for r in rows]).reshape(L, cfg.num_experts)
    summary = {"routed_slots_per_layer": int(B * T * K), "unobserved_experts_per_layer": (tot == 0).sum(1).tolist(),
               "min_max_routed_per_layer": [[int(tot[l].min()), int(tot[l].max())] for l in range(L)],
               "selection": sel}
    jdump(out / "coverage.json", summary)
    log("selection: " + "; ".join(f"L{l}: " + ", ".join(f"{k} e{v['expert']} ({v['routed_tokens']} tok, {v['windows']} win"
                                                         f"{', LOW' if v['low_coverage'] else ''})" for k, v in s["picks"].items())
                                  for l, s in sel.items()) + f"; unobserved per layer {summary['unobserved_experts_per_layer']}")


def selected(out):
    sel = jload(out / "coverage.json")["selection"]
    return [(int(l), v["expert"], name, v["low_coverage"]) for l, s in sel.items() for name, v in s["picks"].items()]


# --------------------------------------------------------------------------- #
# real-model streamed backward check
# --------------------------------------------------------------------------- #
def stage_bcheck(args, out, log):
    """Layers 13-15 (and the head) of one calibration window: whole-graph autograd with the three layers resident vs
    the layer-streamed backward (recompute per layer), FP32 on the device."""
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    routes, _, _ = load_routes(out, "calibration", cfg, dev)
    toks = torch.from_numpy(np.load(out / "tokens_calibration.npy")[:1]).to(dev)
    T = SEQ
    r1 = [(i[:T], w[:T]) for i, w in routes]                  # window 0's rows
    cos, sin = oc.rope(cfg, T, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    with torch.no_grad():
        h = F.embedding(toks, g["embed"])
        for l in range(13):
            h, _ = oc.layer_fixed(ck.layer(l, dev, torch.float32), h, cfg, cos, sin, *r1[l], l)
    h13 = h
    ws = {l: ck.layer(l, dev, torch.float32) for l in (13, 14, 15)}
    sel = {(13, int(r1[13][0][0, 0])), (15, int(r1[15][0][0, 0]))}
    capA = {}                       # weight gradients from the taps: dW = gY^T X (the tables stage's computation)

    def mk(cap):
        hk = oc.ExpertHooks()

        def c(l, e, p, s, t):
            if (l, e) in sel:
                t.retain_grad()
                cap[(l, e, p, s)] = t
        hk.capture = c
        return hk
    hin = h13.clone().requires_grad_(True)
    x = hin
    hkA = mk(capA)
    for l in (13, 14, 15):
        x, _ = oc.layer_fixed(ws[l], x, cfg, cos, sin, *r1[l], l, hkA)
    oc.summed_ce(oc.head(g, x, cfg), toks).sum().backward()
    whole = {k: t.grad.clone() for k, t in capA.items()}
    dw = lambda cap, l, e: {p: cap[(l, e, p, "y")].grad.t() @ cap[(l, e, p if p == "down_proj" else "gate_proj", "x")].detach()
                            for p in oc.PROJ}
    whole_w = {(l, e, p): v for l, e in sel for p, v in dw(capA, l, e).items()}
    whole_h = hin.grad.clone()
    # streamed
    hs = [h13]
    with torch.no_grad():
        for l in (13, 14, 15):
            hs.append(oc.layer_fixed(ws[l], hs[-1], cfg, cos, sin, *r1[l], l)[0])
    hT = hs[-1].clone().requires_grad_(True)
    oc.summed_ce(oc.head(g, hT, cfg), toks).sum().backward()
    gn = hT.grad
    capB = {}
    for j, l in reversed(list(enumerate((13, 14, 15)))):
        hi = hs[j].clone().requires_grad_(True)
        o, _ = oc.layer_fixed(ws[l], hi, cfg, cos, sin, *r1[l], l, mk(capB))
        o.backward(gn)
        gn = hi.grad
    errs = {str(k): rel(capB[k].grad, whole[k]) for k in whole}
    stream_w = {(l, e, p): v for l, e in sel for p, v in dw(capB, l, e).items()}
    errs.update({f"dW{k}": rel(stream_w[k], whole_w[k]) for k in whole_w})
    errs["h13"] = rel(gn, whole_h)
    res = jload(out / "correctness.json")
    res["real_backward_substack"] = {"layers": [13, 14, 15], "experts": sorted(sel), "rel_l2": errs, "max": max(errs.values()),
                                     "pass": bool(max(errs.values()) <= 1e-5)}
    jdump(out / "correctness.json", res)
    log(f"real backward sub-stack: max rel {max(errs.values()):.2e} over {len(errs)} arrays")


# --------------------------------------------------------------------------- #
# BC table
# --------------------------------------------------------------------------- #
def stage_tables(args, out, log):
    """Calibration statistics of the selected experts with the fixed route (PyTorch, layer-streamed, FP32):
    forward (layer inputs stored, loss), then per layer a recomputation with gradients; per routed row the tap
    values and gradients; per window the weight gradients dW = gY^T X of the window's rows."""
    from qwen3_swiglu_common import Quantizers
    from run_native_bc_pipeline import QUANT
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    sel = selected(out)
    want = {(l, e) for l, e, _, _ in sel}
    routes, ids_np, _ = load_routes(out, "calibration", cfg, dev)
    toks = torch.from_numpy(np.load(out / "tokens_calibration.npy")).to(dev)
    B = toks.shape[0]
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    t0 = time.time()
    with torch.no_grad():
        hs = [F.embedding(toks, g["embed"])]
        for l in range(cfg.num_hidden_layers):
            hs.append(oc.layer_fixed(ck.layer(l, dev, torch.float32), hs[-1], cfg, cos, sin, *routes[l], l)[0])
    hT = hs[-1].clone().requires_grad_(True)
    ce = oc.summed_ce(oc.head(g, hT, cfg), toks)
    ce.sum().backward()
    gn = hT.grad
    t_fwd = time.time() - t0
    taps = {}
    for l in reversed(range(cfg.num_hidden_layers)):
        w = ck.layer(l, dev, torch.float32)
        cap = {}
        hk = oc.ExpertHooks()

        def c(ll, e, p, s, t, cap=cap):
            if (ll, e) in want:
                t.retain_grad()
                cap[(ll, e, p, s)] = t
        hk.capture = c
        hi = hs[l].clone().requires_grad_(True)
        o, _ = oc.layer_fixed(w, hi, cfg, cos, sin, *routes[l], l, hk)
        o.backward(gn)
        gn = hi.grad
        for k, t in cap.items():
            taps[k] = (t.detach().cpu(), t.grad.detach().cpu())
        if any(ll == l for ll, _ in want):
            for e in {e for ll, e in want if ll == l}:
                for p in oc.PROJ:
                    taps[(l, e, p, "W")] = w[p][e].detach().cpu()
        del w
    t_bwd = time.time() - t0 - t_fwd
    # per expert: routed rows -> window (rows are ordered by (top-k position, token) as in the expert loop)
    q = Quantizers(out, formats=("p4e1", "p6e1", "p8e1"))
    table = {"B": {}, "C": {}}
    ranges, meta, act_check = {}, {}, []
    for l, e, cov, low in sel:
        idx = routes[l][0].cpu()
        kpos, tok = torch.where(F.one_hot(idx, num_classes=cfg.num_experts + 1).permute(2, 1, 0)[e])
        win = (tok // SEQ).numpy()
        n = len(tok)
        node = f"L{l}.E{e}"
        meta[node] = {"layer": l, "expert": e, "coverage": cov, "low_coverage": low, "routed_rows": int(n),
                      "windows": int(len(set(win.tolist())))}
        x, gx = taps[(l, e, "gate_proj", "x")]
        tensors = {("gate_proj", "x"): (x, gx), ("up_proj", "x"): (x, gx),
                   ("gate_proj", "y"): taps[(l, e, "gate_proj", "y")], ("up_proj", "y"): taps[(l, e, "up_proj", "y")],
                   ("down_proj", "x"): taps[(l, e, "down_proj", "x")], ("down_proj", "y"): taps[(l, e, "down_proj", "y")]}
        for p in oc.PROJ:
            name = f"{node}.{p}"
            W = taps[(l, e, p, "W")]
            X, _ = tensors[(p, "x")]
            _, GY = tensors[(p, "y")]
            G = torch.zeros(W.shape[0], dtype=torch.float64)
            for k in sorted(set(win.tolist())):           # per-window weight gradient, squared, summed over windows
                m = torch.from_numpy(win == k)
                dW = GY[m].double().t() @ X[m].double()
                G += (dW ** 2).sum(1)
            B_ = {"fp32": 0.0}
            for f in QUANT:
                d2 = ((q.weight(W, f) - W).double() ** 2).sum(1)
                B_[f] = float((G / W.shape[1] * d2).sum())
            C_ = {"fp32": 0.0}
            rng_ = {}
            for s in ("x", "y"):
                t, gt = tensors[(p, s)]
                rng_[s] = [float(t.min()), float(t.max())]
            for f in QUANT:
                c = 0.0
                for s in ("x", "y"):
                    t, gt = tensors[(p, s)]
                    gt = gt.double()
                    G2 = (gt ** 2).sum(0)
                    GS = torch.stack([gt[torch.from_numpy(win == k)].sum(0) for k in sorted(set(win.tolist()))]).pow(2).sum(0)
                    err = (q.act(t, f, rng_[s]) - t).double()
                    mu = err.sum(0) / n
                    c += float((G2 / n * ((err ** 2).sum(0) - n * mu ** 2)).sum() + (GS * mu ** 2).sum())
                C_[f] = c
            table["B"][name], table["C"][name], ranges[name] = B_, C_, rng_
        # activation statistics check: dense masked recomputation of the expert input range and int8 error sums
        z = (taps[(l, e, "gate_proj", "x")][0])
        act_check.append({"node": node, "rows": int(n), "rows_equal_routed": bool(z.shape[0] == n)})
    # dense masked recomputation of ranges / error sums for the expert inputs (independent of the gathered path)
    with torch.no_grad():
        dense_errs = []
        h = F.embedding(toks, g["embed"])
        for l in range(cfg.num_hidden_layers):
            w = ck.layer(l, dev, torch.float32)
            if any(ll == l for ll, _, _, _ in sel):
                x = h + oc.attention(w, oc.rms(h, w["ln1"], cfg.rms_norm_eps), cfg, cos, sin)
                zz = oc.rms(x, w["ln2"], cfg.rms_norm_eps).reshape(-1, h.shape[-1]).cpu()
                for ll, e, _, _ in sel:
                    if ll != l:
                        continue
                    msk = (routes[l][0].cpu() == e).any(-1)               # [B*T]
                    name = f"L{l}.E{e}.gate_proj"
                    lo, hi = float(zz[msk].min()), float(zz[msk].max())
                    dense_errs.append({"node": name, "range_dense": [lo, hi], "range_gathered": ranges[name]["x"],
                                       "range_equal": bool([lo, hi] == ranges[name]["x"]),
                                       "rows_dense": int(msk.sum()), "rows_gathered": meta[f"L{l}.E{e}"]["routed_rows"]})
            h = oc.layer_fixed(w, h, cfg, cos, sin, *routes[l], l)[0]
            del w
    act_ok = all(d["range_equal"] and d["rows_dense"] == d["rows_gathered"] for d in dense_errs)
    jdump(out / "table.json", {**table, "naming": "B = W (weights), C = A (activations); node L<layer>.E<expert>.<proj>",
                               "implementation": "PyTorch fixed-route FP32 (GPU), not native"})
    jdump(out / "ranges.json", ranges)
    jdump(out / "tables.json", {"experts": meta, "calibration_summed_ce_fixed": ce.detach().cpu().numpy().tolist(),
                                "activation_check": {"dense_vs_gathered": dense_errs, "pass": act_ok},
                                "seconds": {"forward": t_fwd, "backward": t_bwd, "total": time.time() - t0}})
    log(f"tables: {len(table['B'])} projections, forward {t_fwd:.0f}s backward {t_bwd:.0f}s; activation check {act_ok}")


# --------------------------------------------------------------------------- #
# candidates, freeze, truth
# --------------------------------------------------------------------------- #
def stage_freeze(args, out, log):
    from htq_pareto_search import FORMAT_BITS
    modes = args.modes.split(",")
    if "a" in modes or "wa" in modes:
        if not jload(out / "tables.json")["activation_check"]["pass"]:
            raise SystemExit("activation statistics check failed: activation modes not allowed")
    f = out / "freeze.json"
    fz = jload(f) if f.exists() else {"rounds": []}
    table = jload(out / "table.json")
    meta = jload(out / "tables.json")["experts"]
    opt = {"w": lambda x: (x, "fp32"), "a": lambda x: ("fp32", x), "wa": lambda x: (x, x)}
    cands = jload(out / "candidates.json") if (out / "candidates.json").exists() else []
    have = {c["name"] for c in cands}
    new = []
    for node, mt in meta.items():
        for mode in modes:
            sets = [{p: opt[mode](x)} for p in oc.PROJ for x in FORMATS] + \
                   [{"gate_proj": opt[mode](a), "up_proj": opt[mode](b)} for a in FORMATS for b in FORMATS]
            for fm in sets:
                name = f"{node}|{mode}|" + "|".join(f"{p.split('_')[0]}={w}/{a}" for p, (w, a) in sorted(fm.items()))
                if name in have:
                    continue
                el = 2048 * 1024
                M = sum(el * FORMAT_BITS[fm.get(p, ("fp32", "fp32"))[0]] / 8 for p in oc.PROJ)
                Bo = sum(el * mt["routed_rows"] * FORMAT_BITS[fm.get(p, ("fp32", "fp32"))[0]] * FORMAT_BITS[fm.get(p, ("fp32", "fp32"))[1]]
                         for p in oc.PROJ)
                bc = sum(table["B"][f"{node}.{p}"][w] + table["C"][f"{node}.{p}"][a] for p, (w, a) in fm.items())
                new.append({"name": name, "node": node, "layer": mt["layer"], "expert": mt["expert"], "coverage": mt["coverage"],
                            "low_coverage": mt["low_coverage"], "mode": mode,
                            "set": "gate+up" if len(fm) == 2 else next(iter(fm)), "formats": fm, "bc": float(bc),
                            "M_bytes": M, "BOPs_calibration": Bo})
    cands += new
    jdump(out / "candidates.json", cands)
    fz["rounds"].append({"modes": modes, "added": len(new), "candidates_sha256": sha256(out / "candidates.json"),
                         "table_sha256": sha256(out / "table.json"), "ranges_sha256": sha256(out / "ranges.json"),
                         "coverage_sha256": sha256(out / "coverage.json"), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    jdump(f, fz)
    log(f"frozen: +{len(new)} candidates ({modes}), {len(cands)} total")


def stage_truth(args, out, log):
    """Per start layer: the FP32 fixed-route layer inputs of every window once, then every candidate of that layer
    replayed through the remaining layers (one weight load per layer, candidates kept on the device)."""
    from qwen3_swiglu_common import Quantizers
    import copy
    fz = jload(out / "freeze.json")["rounds"][-1]
    if sha256(out / "candidates.json") != fz["candidates_sha256"] or sha256(out / "table.json") != fz["table_sha256"]:
        raise SystemExit("candidates / table changed after the freeze")
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    ranges = jload(out / "ranges.json")
    q = Quantizers(out, formats=("p6e1", "p8e1"))
    qa = copy.copy(q)
    qa.posit = {}
    for f_, lut in q.posit.items():
        qa.posit[f_] = copy.copy(lut)
        qa.posit[f_].t_values, qa.posit[f_].t_bounds = lut.t_values.to(dev), lut.t_bounds.to(dev)
    for name in args.sets.split(","):
        f = out / f"truth_{name}.json"
        done = jload(f) if f.exists() else {"rows": {}, "base": None, "seconds": 0.0}
        cands = [c for c in jload(out / "candidates.json") if c["name"] not in done["rows"]
                 and (args.modes is None or c["mode"] in args.modes.split(","))]
        if not cands:
            continue
        t0 = time.time()
        routes, _, _ = load_routes(out, name, cfg, dev)
        toks = torch.from_numpy(np.load(out / f"tokens_{name}.npy")).to(dev)
        cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
        g = ck.global_(dev, torch.float32)
        with torch.no_grad():
            hs = [F.embedding(toks, g["embed"])]
            for l in range(cfg.num_hidden_layers):
                hs.append(oc.layer_fixed(ck.layer(l, dev, torch.float32), hs[-1], cfg, cos, sin, *routes[l], l)[0])
            base = head_ce(g, hs[-1], toks, cfg).cpu().numpy()
            done["base"] = base.tolist()
            for L in sorted({c["layer"] for c in cands}):
                group = [c for c in cands if c["layer"] == L]
                H = {c["name"]: hs[L].clone() for c in group}
                for l in range(L, cfg.num_hidden_layers):
                    w = ck.layer(l, dev, torch.float32)
                    for c in group:
                        if l == L:
                            e = c["expert"]
                            saved = {}
                            hk = oc.ExpertHooks()
                            for p, (wf, af) in c["formats"].items():
                                if wf != "fp32":
                                    saved[p] = w[p][e].clone()
                                    w[p][e] = q.weight(w[p][e].cpu(), wf).to(dev)
                                if af != "fp32":
                                    for s in ("x", "y"):
                                        hk.act[(l, e, p, s)] = (lambda t, af=af, r=ranges[f"{c['node']}.{p}"][s]: qa.act(t, af, r))
                            H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *routes[l], l, hk)[0]
                            for p, v in saved.items():
                                w[p][e] = v
                        else:
                            H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *routes[l], l)[0]
                    del w
                for c in group:
                    d = head_ce(g, H.pop(c["name"]), toks, cfg).cpu().numpy() - base
                    done["rows"][c["name"]] = d.tolist()
                done["seconds"] += time.time() - t0
                t0 = time.time()
                jdump(f, done)
                log(f"{name}: layer {L}: {len(group)} candidates ({done['seconds']:.0f}s)")


def stage_analyze(args, out, log):
    import csv
    from scipy.stats import spearmanr
    from run_qwen35_bc import _pair_class
    plan = jload(out / "plan.json")
    cands = jload(out / "candidates.json")
    truth = {n: jload(out / f"truth_{n}.json")["rows"] for n in ("diagnostic", "calibration") if (out / f"truth_{n}.json").exists()}
    D = {}
    T = SEQ - 1
    for n, rows in truth.items():
        D[n] = np.array([rows[c["name"]] for c in cands if c["name"] in rows]) / T
    cands = [c for c in cands if c["name"] in truth["diagnostic"]]
    Dd = D["diagnostic"]
    nw = Dd.shape[1]
    boot = np.random.default_rng(plan["seeds"]["bootstrap"]).integers(0, nw, (2000, nw))
    m = Dd.mean(1)
    Db = np.stack([Dd[:, b].mean(1) for b in boot], 1)
    bc = np.array([c["bc"] for c in cands])
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["node"], c["mode"]), []).append(i)
    res = {"candidates": len(cands), "windows": nw, "spearman": {}, "pairs": {}, "magnitude": {}, "joint": {}}
    for (node, mode), ix in groups.items():
        res["spearman"][f"{node}|{mode}"] = {"rho": float(spearmanr(bc[ix], m[ix])[0]),
                                            "dnll_range": [float(m[ix].min()), float(m[ix].max())],
                                            "coverage": cands[ix[0]]["coverage"], "low_coverage": cands[ix[0]]["low_coverage"]}
    pairs = []
    for (node, mode), ix in groups.items():
        for a, i in enumerate(ix):
            for j in ix[a + 1:]:
                ci, cj = cands[i], cands[j]
                if ci["M_bytes"] != cj["M_bytes"] or ci["BOPs_calibration"] != cj["BOPs_calibration"]:
                    continue
                d = m[i] - m[j]
                agree = float(((Db[i] - Db[j]) * np.sign(d) > 0).mean())
                pairs.append({"i": i, "j": j, "node": node, "mode": mode, "coverage": ci["coverage"],
                              "low_coverage": ci["low_coverage"], "layer": ci["layer"],
                              "class": _pair_class(ci, cj), "truth_diff": float(d), "decisive": agree >= 0.95,
                              "bc_correct": float(0.5 if bc[i] == bc[j] else np.sign(bc[i] - bc[j]) == np.sign(d))})

    def block(ps):
        dec = [p for p in ps if p["decisive"]]
        return {"pairs": len(ps), "decisive": len(dec), "decisive_correct": float(sum(p["bc_correct"] for p in dec)),
                "decisive_accuracy": float(sum(p["bc_correct"] for p in dec) / max(len(dec), 1)),
                "all_accuracy": float(np.mean([p["bc_correct"] for p in ps])) if ps else None,
                "median_abs_diff_decisive": float(np.median([abs(p["truth_diff"]) for p in dec])) if dec else None,
                "median_abs_diff_wrong": float(np.median([abs(p["truth_diff"]) for p in dec if not p["bc_correct"]]))
                if any(not p["bc_correct"] for p in dec) else None}
    res["pairs"]["all"] = block(pairs)
    for key in ("mode", "class", "coverage", "layer", "node"):
        for v in sorted({p[key] for p in pairs}, key=str):
            res["pairs"][f"{key}={v}"] = block([p for p in pairs if p[key] == v])
    for mode in {p["mode"] for p in pairs}:
        for cov in ("high", "mid", "low"):
            res["pairs"][f"mode={mode}|coverage={cov}"] = block([p for p in pairs if p["mode"] == mode and p["coverage"] == cov])
    # magnitudes
    for (node, mode), ix in groups.items():
        lo = np.percentile(Db[ix], 2.5, axis=1)
        res["magnitude"][f"{node}|{mode}"] = {"median_dnll": float(np.median(m[ix])), "max_dnll": float(m[ix].max()),
                                              "significantly_negative": int((np.percentile(Db[ix], 97.5, axis=1) < 0).sum()),
                                              "significantly_positive": int((lo > 0).sum())}
    # single vs joint
    single = {(c["node"], c["mode"], c["set"], tuple(c["formats"][c["set"]])): i for i, c in enumerate(cands) if c["set"] != "gate+up"}
    Dc = D.get("calibration")
    names_all = [c["name"] for c in jload(out / "candidates.json")]
    cal_idx = {n: k for k, n in enumerate([n for n in names_all if "calibration" in truth and n in truth["calibration"]])}
    for (node, mode), ix in groups.items():
        js = [i for i in ix if cands[i]["set"] == "gate+up"]
        if not js:
            continue
        gs = [single[(node, mode, "gate_proj", tuple(cands[i]["formats"]["gate_proj"]))] for i in js]
        us = [single[(node, mode, "up_proj", tuple(cands[i]["formats"]["up_proj"]))] for i in js]
        add_t = m[gs] + m[us]
        add_bc = bc[gs] + bc[us]
        I = Dd[js] - Dd[gs] - Dd[us]
        jp = [p for p in pairs if p["node"] == node and p["mode"] == mode and p["class"].startswith("joint")]
        pos = {i: k for k, i in enumerate(js)}
        row = {"joint_candidates": len(js), "bc_additive_exact": bool(np.allclose(add_bc, bc[js], rtol=1e-12)),
               "rel_interaction_median": float(np.median(np.abs(I.mean(1)) / np.maximum(np.abs(m[js]), 1e-12))),
               "interaction_ci_excludes_0": int(sum((np.percentile(I[k][boot].mean(1), 2.5) > 0) or
                                                    (np.percentile(I[k][boot].mean(1), 97.5) < 0) for k in range(len(js)))),
               "spearman_bc_joint": float(spearmanr(bc[js], m[js])[0]),
               "spearman_diag_singles_sum_vs_joint": float(spearmanr(add_t, m[js])[0])}
        if jp:
            row["joint_pairs"] = len(jp)
            row["bc_accuracy"] = float(np.mean([p["bc_correct"] for p in jp]))
            row["diag_singles_sum_accuracy (same-window oracle)"] = float(np.mean(
                [np.sign(add_t[pos[p["i"]]] - add_t[pos[p["j"]]]) == np.sign(p["truth_diff"]) for p in jp]))
            if Dc is not None:
                cm = {cands[i]["name"]: float(np.mean(truth["calibration"][cands[i]["name"]])) for i in set(gs + us)
                      if cands[i]["name"] in truth["calibration"]}
                if len(cm) == len(set(gs + us)):
                    add_c = np.array([cm[cands[g_]["name"]] + cm[cands[u_]["name"]] for g_, u_ in zip(gs, us)])
                    row["calibration_singles_sum_accuracy (deployable estimate)"] = float(np.mean(
                        [np.sign(add_c[pos[p["i"]]] - add_c[pos[p["j"]]]) == np.sign(p["truth_diff"]) for p in jp]))
        res["joint"][f"{node}|{mode}"] = row
    jdump(out / "analysis.json", res)
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "node", "mode", "coverage", "class", "truth_diff", "decisive", "bc_correct"])
        for p in pairs:
            w.writerow([cands[p["i"]]["name"], cands[p["j"]]["name"], p["node"], p["mode"], p["coverage"], p["class"],
                        p["truth_diff"], p["decisive"], p["bc_correct"]])
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "node", "coverage", "mode", "set", "bc", "dnll", "ci_lo", "ci_hi", "M_bytes", "BOPs_calibration"])
        for i, c in enumerate(cands):
            w.writerow([c["name"], c["node"], c["coverage"], c["mode"], c["set"], c["bc"], m[i],
                        np.percentile(Db[i], 2.5), np.percentile(Db[i], 97.5), c["M_bytes"], c["BOPs_calibration"]])
    a = res["pairs"]["all"]
    log(f"pairs: {a['pairs']}, decisive {a['decisive']}, BC correct {a['decisive_correct']:.0f} ({a['decisive_accuracy']:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("precheck", "tiny", "prepare", "capture", "coverage", "bcheck",
                                                       "tables", "freeze", "truth", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--modes", default=None, help="freeze: comma list of w,a,wa (default w); truth: restrict")
    ap.add_argument("--sets", default="diagnostic", help="truth: diagnostic,calibration")
    ap.add_argument("--skip-native", action="store_true", help="tiny: PyTorch parts only")
    args = ap.parse_args()
    if args.stage == "freeze" and args.modes is None:
        args.modes = "w"
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    status = "ok"
    try:
        globals()[f"stage_{args.stage}"](args, out, log)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        import resource
        timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
        timing["runs"].append({"stage": args.stage, "device": args.device, "modes": args.modes, "sets": args.sets,
                               "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                               "peak_rss_gb_largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20,
                               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
