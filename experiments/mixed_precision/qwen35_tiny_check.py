#!/usr/bin/env python3
"""Native forward / backward / gradient statistics of a tiny random Qwen3.5 (same architecture: Gated DeltaNet
linear-attention layers and one gated full-attention layer, FFN, tied LM head) against PyTorch FP64 autograd.

The tiny model goes through the same export adapter (qwen35_common.TextExport) and the same statistics pass as the
0.8B model; the reference is the adapter in float64 (FP64 throughout) with autograd. Checked per window:
  logits (native vs torch FP64), every tap's per-channel sum g^2 and sum g (input x: the total gradient of the shared
  input; output y), every searched weight's per-output-channel sum dW^2.
Gate: per array max-relative error (max |a - b| / max |b|), median <= 1e-3 and max <= 5e-2 (the 0.6B gate); the
tiny model is expected far below it. Also: transformers' own tiny model against the adapter (FP32).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class Holder(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = causal.model
        self.lm_head = causal.lm_head


def tiny_model(seed=0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
        tie_word_embeddings=True, attn_output_gate=True,
        rope_parameters={"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    m = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():                        # larger than the 0.02 init, so that the gradients are not tiny
        for n, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(0, 0.2)
            elif n.endswith("norm.weight") or n.endswith("_norm.weight"):
                p.normal_(0, 0.1)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def torch_stats(holder, cfg, ids, mapping, dtype):
    """Per-window statistics of the adapter in `dtype` with autograd, in the layout's terms."""
    ad = q35.TextExport(holder, ids.shape[0], dtype=dtype)
    mods = dict(holder.named_modules())
    params = dict(holder.named_parameters())
    cap, hooks = {}, []
    for node, info in mapping.items():
        mod = mods[info["param"].removesuffix(".weight")]
        params[info["param"]].requires_grad_(True)

        def pre(m, inp, node=node):
            inp[0].retain_grad()
            cap[(node, "x")] = inp[0]

        def post(m, inp, out, node=node):
            out.retain_grad()
            cap[(node, "y")] = out
        hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    holder.model.language_model.embed_tokens.weight.requires_grad_(True)
    x = torch.from_numpy(ids)
    z = ad(x[None])
    loss = q35.summed_ce(z, x)
    loss.backward()
    for h in hooks:
        h.remove()
    st = {}
    for (node, slot), t in cap.items():
        g = t.grad[0].double()                                  # [L, C]
        st[(node, slot)] = ((g ** 2).sum(0).numpy(), g.sum(0).numpy())
    for node, info in mapping.items():
        p = params[info["param"]]
        st[(node, "w")] = (p.grad.double() ** 2).sum(1).numpy()
        p.grad = None
        p.requires_grad_(False)
    holder.model.language_model.embed_tokens.weight.grad = None
    return z.detach().double().numpy()[0], st, float(loss)


def native_stats(work, onnx_path, ids_list, vocab, mapping, tools, erm):
    import onnx  # noqa: F401
    work.mkdir(parents=True, exist_ok=True)
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / "model", onnx_path])
    layout_path, inst = work / "layout.json", work / "grads.onnx.mlir"
    erm.run([tools.opt, work / "model.onnx.mlir", "--shape-inference", "--mixedp-collect-stats", "-o", inst],
            env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": ",".join(mapping),
                 "MIXEDP_STATS_LAYOUT": str(layout_path)})
    so, info = erm.build_so(tools, work, "grads", inst, None, None, opt="O0", krnl_opt="O3")
    layout = json.loads(layout_path.read_text())
    out = []
    for k, ids in enumerate(ids_list):
        L = len(ids)
        oh = np.zeros((1, L, vocab), np.float32)
        oh[0, np.arange(L - 1), ids[1:]] = 1.0
        np.save(work / f"x{k}.npy", ids[None])
        np.save(work / f"oh{k}.npy", oh)
        res = work / f"stats{k}.npy"
        erm.run([sys.executable, HERE / "run_native_bc_pipeline.py", "--_stats-child", so, "model", tools.lib,
                 work / f"x{k}.npy", 0, work / f"oh{k}.npy", layout["result_index"], res])
        v = np.load(res)
        taps = {t["tap"]: t for t in layout["taps"]}
        st = {}
        for p in layout["points"]:
            t = taps[p["tap"]]
            o, ch = t["offset"], t["channels"]
            st[(p["node"], p["slot"])] = (v[o:o + ch], v[o + ch:o + 2 * ch])
        for w in layout["weights"]:
            st[(w["node"], "w")] = v[w["offset"]:w["offset"] + w["channels"]]
        out.append(st)
    # logits: the plain FP32 build
    so32, _ = erm.build_so(tools, work, "fp32", work / "model.onnx.mlir", None, None, opt="O0", krnl_opt="O3",
                           uses_posit=False, uses_f16=False)
    np.save(work / "xs.npy", np.stack(ids_list))
    z, _ = erm.run_so(tools, so32, work / "xs.npy", work / "z.npy", 3600)
    return out, z, layout, info


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    import onnx
    from qwen3_prepare import fold_constants
    from run_native_bc_pipeline import erm
    causal, cfg = tiny_model()
    holder = Holder(causal).eval()
    rng = np.random.default_rng(0)
    ids_list = [rng.integers(0, cfg.vocab_size, args.seq).astype(np.int64) for _ in range(2)]
    # transformers' tiny model vs the adapter (FP32)
    with torch.no_grad():
        ref = causal(input_ids=torch.from_numpy(ids_list[0])[None], use_cache=False).logits
        ad32 = q35.TextExport(holder, args.seq).eval()
        a32 = ad32(torch.from_numpy(ids_list[0])[None])
    hf_vs_adapter = float((a32 - ref).norm() / ref.norm())
    torch.onnx.export(ad32, (torch.from_numpy(ids_list[0])[None],), str(out / "raw.onnx"), input_names=["input_ids"],
                      output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=True)
    m = onnx.load(str(out / "raw.onnx"))
    fold_constants(m)
    mapping = q35.map_projection_nodes(m, holder, cfg)
    onnx.save(m, str(out / "tiny.onnx"))
    tools = erm.Tools()
    nat, z, layout, info = native_stats(out / "native", out / "tiny.onnx", ids_list, cfg.vocab_size, mapping, tools, erm)
    holder64 = Holder(causal).double()
    res = {"seq": args.seq, "hf_vs_adapter_fp32_logits_rel_l2": hf_vs_adapter, "windows": []}
    errs_all = []
    for k, ids in enumerate(ids_list):
        z64, st64, loss64 = torch_stats(holder64, cfg, ids, mapping, torch.float64)
        errs = {}
        for key, ref_v in st64.items():
            nv = nat[k][key]
            if key[1] == "w":
                errs[f"{key[0]}|w"] = rel_max(nv, ref_v)
            else:
                errs[f"{key[0]}|{key[1]}|g2"] = rel_max(nv[0], ref_v[0])
                errs[f"{key[0]}|{key[1]}|g"] = rel_max(nv[1], ref_v[1])
        e = np.array(list(errs.values()))
        errs_all += list(e)
        res["windows"].append({"logits_rel_l2_native_vs_fp64": float(np.linalg.norm(z[k] - z64) / np.linalg.norm(z64)),
                               "loss_fp64": loss64, "arrays": len(e), "rel_max_median": float(np.median(e)),
                               "rel_max_max": float(e.max()),
                               "worst": sorted(errs.items(), key=lambda kv: -kv[1])[:5]})
    e = np.array(errs_all)
    res["gate"] = {"median": float(np.median(e)), "max": float(e.max()),
                   "pass": bool(np.median(e) <= 1e-3 and e.max() <= 5e-2)}
    res["layout"] = {"weights": len(layout["weights"]), "taps": len(layout["taps"]), "points": len(layout["points"])}
    res["ops"] = dict(__import__("collections").Counter(n.op_type for n in m.graph.node))
    res["seconds"] = time.time() - t0
    (out / "tiny_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("hf_vs_adapter_fp32_logits_rel_l2", "gate", "layout")}, indent=1))
    for w in res["windows"]:
        print(json.dumps(w))


if __name__ == "__main__":
    main()
