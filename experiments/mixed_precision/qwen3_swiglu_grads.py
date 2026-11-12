#!/usr/bin/env python3
"""Qwen3-0.6B: native gradient statistics (--mixedp-collect-stats grads) against PyTorch autograd in FP32 and FP64,
per layer and field, built from scratch in a new folder (nothing reused: no earlier .so, no cached native results).

Same calibration windows (validation_tokens.npy[:--windows]), loss and seed on both sides: summed next-token
cross-entropy over positions 0..L-2, i.e. the native seed (softmax(z) - onehot) x row mask. Statistics per window:
  tap (node, x|y)  sum_pos g^2 and sum_pos g per channel (g: the total gradient of the tensor; shared inputs such as
                   gate/up's (and q/k/v's) input sum their branches)
  weight           sum_in dW^2 per output channel
Each array is compared three ways -- native vs FP64, PyTorch FP32 vs FP64, native vs PyTorch FP32 (the earlier
check's pair) -- with compare(): rel_max = max|a-b| / max|b| (the earlier gate: median <= 1e-3, max <= 5e-2),
rel_l2, cosine. PyTorch also records, per tap, the per-position share of sum g^2 and the largest |activation| (to
see whether single positions dominate), and the branch parts of the shared gate/up input gradient.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from qwen3_swiglu_common import PREP, VOCAB, compare, environment, erm, load_model, node_mapping, summed_ce, torch

HERE = Path(__file__).resolve().parent


def native_stats(out, calib, log):
    tools = erm.Tools()
    mapping = node_mapping()
    out.mkdir(parents=True)
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", out / erm.SYMBOL_POSTFIX,
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", PREP / "model.onnx"])
    base = out / "model.mlirbc"
    erm.run([tools.opt, out / "model.onnx.mlir", "--emit-bytecode", "-o", base])
    (out / "model.onnx.mlir").unlink()
    layout_path, inst = out / "grads_layout.json", out / "grads.onnx.mlirbc"
    t0 = time.time()
    erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "--emit-bytecode", "-o", inst],
            env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": ",".join(mapping),
                 "MIXEDP_STATS_LAYOUT": str(layout_path)})
    so, info = erm.build_so(tools, out, "grads", inst, None, None, opt="O0", krnl_opt="O3", constants_to_file=True,
                            bytecode=True)
    for p in out.iterdir():
        if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
            p.unlink()
    log(f"native grads .so built in {time.time() - t0:.0f}s")
    layout = json.loads(layout_path.read_text())
    np.save(out / "calib.npy", calib)
    vals, run_s = [], []
    for i, ids in enumerate(calib):
        oh = np.zeros((1, calib.shape[1], VOCAB), np.float32)
        oh[0, np.arange(calib.shape[1] - 1), ids[1:]] = 1.0
        np.save(out / f"onehot_{i}.npy", oh)
        r, dt = erm.run([sys.executable, HERE / "run_native_bc_pipeline.py", "--_stats-child", so, erm.SYMBOL_POSTFIX,
                         tools.lib, out / "calib.npy", i, out / f"onehot_{i}.npy", layout["result_index"],
                         out / f"native_w{i}.npy"])
        (out / f"onehot_{i}.npy").unlink()
        vals.append(np.load(out / f"native_w{i}.npy"))
        run_s.append(json.loads(r.stdout.strip().splitlines()[-1])["run_s"])
    log(f"native: {np.mean(run_s):.1f}s per window")
    return layout, vals, {"build_steps": info["steps"], "run_s_per_window": float(np.mean(run_s))}


def torch_stats(calib, dtype, log):
    model, wrapper = load_model(dtype)
    mapping = node_mapping()
    modules = dict(model.named_modules())
    mod = {node: modules[param.removesuffix(".weight")] for node, (_, _, param) in mapping.items()}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mod.values():
        m.weight.requires_grad_(True)
    model.model.embed_tokens.weight.requires_grad_(True)       # so layer 0's inputs carry gradients
    out = []
    t0 = time.time()
    for ids in calib:
        cap = {}

        def hook(m, inp, outp):
            for t in (inp[0], outp):
                t.retain_grad()
            cap[m] = (inp[0], outp)

        hs = [m.register_forward_hook(hook) for m in mod.values()]
        model.zero_grad(set_to_none=True)
        loss = summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))
        loss.backward()
        for h in hs:
            h.remove()
        r = {"loss": float(loss)}
        for node, m in mod.items():
            x, y = cap[m]
            for slot, t in (("x", x), ("y", y)):
                g = t.grad[0].double().numpy()                      # [L, channels]
                a = t.detach()[0].double().numpy()
                e = (g ** 2).sum(1)
                r[(node, slot)] = {"g2": (g ** 2).sum(0), "gs": g.sum(0), "pos_share": e / max(e.sum(), 1e-300),
                                   "act_absmax": float(np.abs(a).max()),
                                   "act_absmax_pos": int(np.abs(a).max(1).argmax())}
            r[(node, "w")] = {"g2": (m.weight.grad.double().numpy() ** 2).sum(1)}
            if mapping[node][1] in ("gate", "up"):       # branch part of the shared input gradient: g_y W
                gb = (y.grad[0].double() @ m.weight.detach().double()).numpy()
                r[(node, "x_branch")] = {"g2": (gb ** 2).sum(0), "gs": gb.sum(0)}
        out.append(r)
    log(f"PyTorch {dtype}: {(time.time() - t0) / len(calib):.1f}s per window")
    del model, wrapper
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)        # a new folder: no earlier .so or native results can be reused
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    env = environment(["experiments/mixed_precision/qwen3_swiglu_grads.py"])
    (out / "environment.json").write_text(json.dumps(env, indent=1) + "\n")
    calib = np.load(PREP / "validation_tokens.npy")[:args.windows]
    layout, native, ninfo = native_stats(out / "native", calib, log)
    refs = {"fp32": torch_stats(calib, torch.float32, log), "fp64": torch_stats(calib, torch.float64, log)}
    mapping = node_mapping()
    taps = {t["tap"]: t for t in layout["taps"]}
    rows = []
    for k in range(len(calib)):
        v = native[k]
        arrays = []
        for p in layout["points"]:
            t = taps[p["tap"]]
            o, c = t["offset"], t["channels"]
            arrays.append((p["node"], p["slot"], "sum_g2", v[o:o + c], "g2"))
            arrays.append((p["node"], p["slot"], "sum_g", v[o + c:o + 2 * c], "gs"))
        for w in layout["weights"]:
            arrays.append((w["node"], "w", "sum_dW2", v[w["offset"]:w["offset"] + w["channels"]], "g2"))
        for node, slot, field, nat, key in arrays:
            r64, r32 = refs["fp64"][k][(node, slot)][key], refs["fp32"][k][(node, slot)][key]
            layer, kind, _ = mapping[node]
            row = {"window": k, "layer": layer, "kind": kind, "slot": slot, "field": field, "node": node,
                   "native_vs_fp64": compare(nat, r64), "fp32_vs_fp64": compare(r32, r64),
                   "native_vs_fp32": compare(nat, r32)}
            if slot != "w":
                info = refs["fp64"][k][(node, slot)]
                row["act_absmax"], row["act_absmax_pos"] = info["act_absmax"], info["act_absmax_pos"]
                row["pos0_share_g2"] = float(info["pos_share"][0])
                row["top_pos_share_g2"] = float(info["pos_share"].max())
            rows.append(row)
    branch = []
    for k in range(len(calib)):
        for node, (layer, kind, _) in mapping.items():
            if kind in ("gate", "up"):
                b, tot = refs["fp64"][k][(node, "x_branch")], refs["fp64"][k][(node, "x")]
                branch.append({"window": k, "layer": layer, "kind": kind,
                               "branch_g2_over_total_g2": float(b["g2"].sum() / tot["g2"].sum()),
                               "branch_vs_total_gs": compare(b["gs"], tot["gs"])})
    losses = {name: [r["loss"] for r in ref] for name, ref in refs.items()}

    def summary(pair):
        e = np.array([r[pair]["rel_max"] for r in rows])
        return {"median_rel_max": float(np.median(e)), "p99_rel_max": float(np.percentile(e, 99)),
                "max_rel_max": float(e.max()), "median_rel_l2": float(np.median([r[pair]["rel_l2"] for r in rows])),
                "min_cosine": float(min(r[pair]["cosine"] for r in rows))}
    result = {"windows": len(calib), "arrays": len(rows), "losses": losses, "native": ninfo,
              "summary": {p: summary(p) for p in ("native_vs_fp64", "fp32_vs_fp64", "native_vs_fp32")},
              "gate_old_check": {"median_le_1e-3": None, "max_le_5e-2": None}}
    s = result["summary"]["native_vs_fp32"]
    result["gate_old_check"] = {"median_le_1e-3": s["median_rel_max"] <= 1e-3, "max_le_5e-2": s["max_rel_max"] <= 5e-2}
    (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "arrays.json").write_text(json.dumps(rows) + "\n")
    (out / "branch_gradients.json").write_text(json.dumps(branch) + "\n")
    log(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
