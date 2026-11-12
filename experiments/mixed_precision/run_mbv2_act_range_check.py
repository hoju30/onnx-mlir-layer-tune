#!/usr/bin/env python3
"""MobileNetV2: does the activation range calibrated on 10 images clip held-out data (invisible to the criterion),
or are the ranges stretched by a few outlier channels?

Per weighted node, its input X and output Y with the pass's per-tensor affine ranges (calibration min/max):
  coverage       fraction of held-out elements outside [lo, hi]; the per-element squared error at int4 / int8 on the
                 held-out vs the calibration images (Y of a node followed by ReLU6 measured after the ReLU6, where
                 errors in the clamped region do no harm); the share of the held-out error from clipped elements;
                 and per node the factor by which its activation Fisher cost sum_t mean(g_t^2) * sum delta_t^2 would
                 change if delta were measured on held-out data (per element) instead of the calibration images.
  concentration  per-channel max |t| on the calibration images: largest / median channel, and the median over
                 channels of (channel max - channel min) / (hi - lo) (how much of the int grid a typical channel gets).
Link to the under-estimate: Spearman across nodes between that factor and log10(dloss / Fisher) at a=int4,
activation-only (runs/mbv2_act_lowbit_check_20261004; plain and debiased).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analyze_fisher_vs_sqnr import TYPES, layer_type  # noqa: E402
from eval_random_mixed_precision import calibrate_ranges, preprocess_by_source, supported_nodes  # noqa: E402
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402

FMTS = ("int4", "int8")
RELU6_AFTER = {"stem", "ds_dw", "pw", "dw", "head"}   # timm MobileNetV2: these convs are followed by ReLU6


def chan_reduce(t, op):
    return op(t, dim=(0, 2, 3)) if t.dim() == 4 else op(t, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]][:args.num_eval]
    cx, cy = preprocess_by_source(args.images_dir, {"images": calib}, meta), np.array([e["label"] for e in calib])
    ex = preprocess_by_source(args.images_dir, {"images": held}, meta)
    all_nodes = supported_nodes(graph)
    nodes = [n for n in all_nodes if n.get("weight_const")]
    ranges = calibrate_ranges(graph, all_nodes, cx)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}

    import timm
    model = timm.create_model(args.timm_name, pretrained=True).eval()
    fold_batchnorms(model)
    layers = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(layers) == len(nodes)
    module_of = {}
    for n, m in zip(nodes, layers):
        np.testing.assert_allclose(m.weight.detach().numpy(), inits[by_name[n["name"]].input[1]], rtol=1e-4, atol=1e-5)
        module_of[n["name"]] = m
    names = list(module_of)
    kind = {n: layer_type(i, len(names)) for i, n in enumerate(names)}
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    cap = {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))

    # --- S_t = mean(g_t^2) on the calibration images (the criterion's sensitivity) ---
    g2 = {n: {"x": 0.0, "y": 0.0} for n in names}
    gcnt = {n: {"x": 0, "y": 0} for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in cap[n]]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        for i, n in enumerate(names):
            for j, s in enumerate(("x", "y")):
                g2[n][s] += float((grads[2 * i + j].double() ** 2).sum())
                gcnt[n][s] += cap[n][j].numel()
        cap.clear()
    S = {n: {s: g2[n][s] / gcnt[n][s] for s in ("x", "y")} for n in names}

    # --- error statistics on the calibration and held-out images ---
    def blank():
        return {n: {s: {"count": 0, "out": 0, **{f: 0.0 for f in FMTS}, **{f"clip_{f}": 0.0 for f in FMTS}}
                    for s in ("x", "y")} for n in names}

    stats = {"cal": blank(), "held": blank()}
    cmax = {n: {"x": None, "y": None} for n in names}
    cmin = {n: {"x": None, "y": None} for n in names}

    @torch.no_grad()
    def accumulate(split, x):
        model(torch.from_numpy(x))
        for n in names:
            for j, s in enumerate(("x", "y")):
                t = cap[n][j]
                lo, hi = ranges[tens[n][j]]
                a = stats[split][n][s]
                outside = (t < lo) | (t > hi)
                a["count"] += t.numel()
                a["out"] += int(outside.sum())
                for f in FMTS:
                    q = act_fake_quant(t, f, (lo, hi))
                    d = (q.clamp(0, 6) - t.clamp(0, 6)) if s == "y" and kind[n] in RELU6_AFTER else (q - t)
                    d2 = d.double() ** 2
                    a[f] += float(d2.sum())
                    a[f"clip_{f}"] += float(d2[outside].sum())
                if split == "cal":
                    hi_c, lo_c = chan_reduce(t, torch.amax), chan_reduce(t, torch.amin)
                    cmax[n][s] = hi_c if cmax[n][s] is None else torch.maximum(cmax[n][s], hi_c)
                    cmin[n][s] = lo_c if cmin[n][s] is None else torch.minimum(cmin[n][s], lo_c)
        cap.clear()

    accumulate("cal", cx)
    for i in range(0, len(ex), 50):
        accumulate("held", ex[i:i + 50])

    rows = []
    for n in names:
        r = {"node": n, "type": kind[n]}
        for s in ("x", "y"):
            c, h = stats["cal"][n][s], stats["held"][n][s]
            lo, hi = ranges[tens[n][s == "y"]]
            m = torch.maximum(cmax[n][s].abs(), cmin[n][s].abs())
            span = (cmax[n][s] - cmin[n][s]) / (hi - lo)
            r[s] = {"frac_out_held": h["out"] / h["count"], "frac_out_cal": c["out"] / c["count"],
                    **{f"ratio_{f}": (h[f] / h["count"]) / (c[f] / c["count"]) if c[f] > 0 else float("nan")
                       for f in FMTS},
                    **{f"clip_share_{f}": h[f"clip_{f}"] / h[f] if h[f] > 0 else 0.0 for f in FMTS},
                    "concentration": float(m.max() / m.median().clamp_min(1e-12)),
                    "median_channel_span": float(span.median()) if hi > lo else float("nan")}
        for f in FMTS:   # activation Fisher cost with held-out (per element) vs calibration errors
            cal = sum(S[n][s] * stats["cal"][n][s][f] for s in ("x", "y"))
            hld = sum(S[n][s] * stats["held"][n][s][f] / stats["held"][n][s]["count"] * stats["cal"][n][s]["count"]
                      for s in ("x", "y"))
            r[f"fisher_factor_{f}"] = hld / cal if cal > 0 else float("nan")
        rows.append(r)
    R = {r["node"]: r for r in rows}

    print(f"ranges from {len(cy)} calibration images; held-out {len(ex)} images")
    print("\n1. coverage, median over the type's nodes: outside-range fraction (held-out), per-element error held-out / "
          "calibration at int4 and int8, share of the held-out int4 error from clipped elements")
    print(f"   {'type':<6}{'n':>3} | {'X out':>8} {'X r4':>6} {'X r8':>6} {'X clip4':>8} | {'Y out':>8} {'Y r4':>6} "
          f"{'Y r8':>6} {'Y clip4':>8} | {'Fisher factor int4':>18}")
    for t in TYPES:
        ns = [n for n in names if kind[n] == t]
        med = lambda s, k: np.median([R[n][s][k] for n in ns])  # noqa: E731
        print(f"   {t:<6}{len(ns):>3} | {med('x', 'frac_out_held'):8.2e} {med('x', 'ratio_int4'):6.2f} "
              f"{med('x', 'ratio_int8'):6.2f} {med('x', 'clip_share_int4'):8.1%} | {med('y', 'frac_out_held'):8.2e} "
              f"{med('y', 'ratio_int4'):6.2f} {med('y', 'ratio_int8'):6.2f} {med('y', 'clip_share_int4'):8.1%} | "
              f"{np.median([R[n]['fisher_factor_int4'] for n in ns]):18.2f}")
    print("\n2. concentration (calibration): largest / median channel max|t|, and the median channel's span "
          "(max - min) as a fraction of the range [lo, hi] (int4 has 16 levels over the range)")
    print(f"   {'type':<6} | {'X max/med':>10} {'X med span':>11} | {'Y max/med':>10} {'Y med span':>11}")
    for t in TYPES:
        ns = [n for n in names if kind[n] == t]
        med = lambda s, k: np.median([R[n][s][k] for n in ns])  # noqa: E731
        print(f"   {t:<6} | {med('x', 'concentration'):10.1f} {med('x', 'median_channel_span'):11.1%} | "
              f"{med('y', 'concentration'):10.1f} {med('y', 'median_channel_span'):11.1%}")

    lowbit = json.loads((HERE / "runs/mbv2_act_lowbit_check_20261004/results.json").read_text())["rows"]
    print("\n3. link to the under-estimate at a=int4 (activation-only), nodes with dloss > 2 SE")
    link = {}
    for mode in ("plain", "debiased"):
        sel = [r for r in lowbit if r["bits"] == "int4" and r["mode"] == mode and r["dloss"] > 2 * r["dloss_se"]]
        under = np.array([np.log10(r["dloss"] / r["fisher"]) for r in sel])
        fac = np.array([np.log10(R[r["node"]]["fisher_factor_int4"]) for r in sel])
        rho = float(spearmanr(fac, under)[0])
        link[mode] = {"nodes": len(sel), "spearman": rho}
        print(f"   {mode:<9} {len(sel)} nodes: Spearman(Fisher factor, dloss/Fisher) {rho:+.2f}")
    head = next(n for n in names if kind[n] == "head")
    print(f"\n4. conv_head ({head}): X out {R[head]['x']['frac_out_held']:.2e}, X r4 {R[head]['x']['ratio_int4']:.2f}, "
          f"X clip share {R[head]['x']['clip_share_int4']:.1%}, X max/med {R[head]['x']['concentration']:.1f}, "
          f"X med span {R[head]['x']['median_channel_span']:.1%} | Y out {R[head]['y']['frac_out_held']:.2e}, "
          f"Y r4 {R[head]['y']['ratio_int4']:.2f}, Y max/med {R[head]['y']['concentration']:.1f} | "
          f"Fisher factor int4 {R[head]['fisher_factor_int4']:.2f} "
          f"(rank {sorted(names, key=lambda n: -R[n]['fisher_factor_int4']).index(head) + 1} of {len(names)})")
    print("   largest Fisher factors at int4: " + ", ".join(
        f"{kind[n]} {n} {R[n]['fisher_factor_int4']:.2f}" for n in sorted(names, key=lambda n: -R[n]['fisher_factor_int4'])[:6]))

    (args.out_dir / "results.json").write_text(json.dumps(
        {"num_calibration": len(cy), "num_eval": len(ex), "link": link, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
