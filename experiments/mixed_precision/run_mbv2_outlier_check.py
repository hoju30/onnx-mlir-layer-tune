#!/usr/bin/env python3
"""MobileNetV2: does the uneven per-channel range (a few outlier channels stretching a per-tensor range) cause
activation damage, and does the criterion predict that damage?

Intervention, activation-only, one node at a time (w=fp32; a= quantizes the node's input and output) at --bits:
  per-tensor   one affine range per tensor (calibration min/max), what the pass does
  per-channel  each channel its own affine range (calibration per-channel min/max), same bits and int_qparams
Removing the uneven ranges should cut the damage where they matter. Held-out loss increase over FP32, PyTorch with
the pass's fake-quant arithmetic, paired per image. Predictions (calibration images) of both modes by the per-tensor
Fisher (A, mean(g^2) * sum delta^2) and the channel bias-aware variant (C, run_mbv2_channel_fisher.py):
if the criterion captures the outlier effect, its predicted per-channel / per-tensor ratio matches the actual one.
Concentration per node as in run_mbv2_act_range_check.py.
"""
import argparse
import json
import sys
import time
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
from run_mbv2_act_lowbit_check import chan_shape, chan_sum  # noqa: E402
from run_mbv2_act_range_check import chan_reduce  # noqa: E402
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402

MODES = ("per-tensor", "per-channel")


def channel_qparams(lo, hi, bits):
    """int_qparams (eval_random_mixed_precision.py) applied per channel."""
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    lo, hi = lo.clamp(max=0.0), hi.clamp(min=0.0)
    scale = (hi - lo) / (qmax - qmin)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zp = torch.round(qmin - lo / scale).clamp(qmin, qmax)
    return scale, zp


def channel_fake_quant(t, bits, scale, zp):
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    s, z = scale.view(chan_shape(t)), zp.view(chan_shape(t))
    return (torch.clamp(torch.round(t / s + z), qmin, qmax) - z) * s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--bits", default="int4")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    bits, nb = args.bits, int(args.bits[3:])
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]][:args.num_eval]
    cx, cy = preprocess_by_source(args.images_dir, {"images": calib}, meta), np.array([e["label"] for e in calib])
    ex, ey = preprocess_by_source(args.images_dir, {"images": held}, meta), np.array([e["label"] for e in held])
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

    # --- per-channel calibration ranges ---
    with torch.no_grad():
        model(torch.from_numpy(cx))
    qp, conc = {}, {}
    for n in names:
        for j, s in enumerate(("x", "y")):
            t = cap[n][j]
            hi_c, lo_c = chan_reduce(t, torch.amax), chan_reduce(t, torch.amin)
            lo, hi = ranges[tens[n][j]]
            assert abs(float(lo_c.min()) - lo) <= 1e-3 * max(1.0, abs(lo)) and abs(float(hi_c.max()) - hi) <= 1e-3 * max(1.0, abs(hi))
            qp[(n, s)] = channel_qparams(lo_c, hi_c, nb)
            m = torch.maximum(hi_c.abs(), lo_c.abs())
            conc[(n, s)] = {"max_over_median": float(m.max() / m.median().clamp_min(1e-12)),
                            "median_span": float(((hi_c - lo_c) / (hi - lo)).median()) if hi > lo else float("nan")}
    cap.clear()

    def quant(n, j, t, mode):
        s = ("x", "y")[j]
        return act_fake_quant(t, bits, ranges[tens[n][j]]) if mode == "per-tensor" else channel_fake_quant(t, nb, *qp[(n, s)])

    # --- calibration: gradients and error statistics of both modes, for A and C ---
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, **{m: {"s1": 0.0, "s2": 0.0} for m in MODES}} for s in ("x", "y")}
          for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in cap[n]]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[2 * i + j].double(), st[n][s]
                    a["g2"] = a["g2"] + chan_sum(g ** 2)
                    a["gc"] = a["gc"] + chan_sum(g) ** 2
                    a["n"] += t.numel() // t.shape[1]
                    for mode in MODES:
                        d = (quant(n, j, t, mode) - t).double()
                        a[mode]["s1"] = a[mode]["s1"] + chan_sum(d)
                        a[mode]["s2"] = a[mode]["s2"] + chan_sum(d ** 2)
        cap.clear()
    pred = {}
    for n in names:
        for mode in MODES:
            A = C = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                S_c = a["g2"] / a["n"]
                S = float(a["g2"].sum()) / (a["n"] * len(a["g2"]))
                mu = a[mode]["s1"] / a["n"]
                A += S * float(a[mode]["s2"].sum())
                C += float((S_c * (a[mode]["s2"] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            pred[(n, mode)] = {"A": A, "C": C}
    log("calibration statistics ready")

    # --- held-out ground truth ---
    xs, ys = torch.from_numpy(ex), torch.from_numpy(ey)
    current = {}

    @torch.no_grad()
    def per_image_loss():
        return torch.cat([F.cross_entropy(model(xs[i:i + 100]), ys[i:i + 100], reduction="none")
                          for i in range(0, len(ys), 100)]).numpy()

    base = per_image_loss()
    dl = {}
    for i, n in enumerate(names):
        m = module_of[n]
        h = [m.register_forward_pre_hook(lambda mod, inp, n=n: (quant(n, 0, inp[0], current[n]),)),
             m.register_forward_hook(lambda mod, inp, out, n=n: quant(n, 1, out, current[n]))]
        for mode in MODES:
            current[n] = mode
            dl[(n, mode)] = per_image_loss() - base
        for x in h:
            x.remove()
        if (i + 1) % 10 == 0:
            log(f"ground truth: {i + 1}/{len(names)} nodes")

    rows = []
    for n in names:
        pt, pc = dl[(n, "per-tensor")], dl[(n, "per-channel")]
        rows.append({"node": n, "type": kind[n], **{f"conc_{s}": conc[(n, s)] for s in ("x", "y")},
                     **{f"dloss_{m}": float(dl[(n, m)].mean()) for m in MODES},
                     **{f"dloss_se_{m}": float(dl[(n, m)].std(ddof=1) / np.sqrt(len(ys))) for m in MODES},
                     "diff": float((pc - pt).mean()), "diff_se": float((pc - pt).std(ddof=1) / np.sqrt(len(ys))),
                     **{f"pred_{c}_{m}": pred[(n, m)][c] for c in ("A", "C") for m in MODES}})
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "PyTorch, pass fake-quant arithmetic", "bits": bits, "num_eval": len(ys), "rows": rows},
        indent=2) + "\n")
    R = {r["node"]: r for r in rows}

    print(f"\n1. {bits} activation-only damage, per-tensor -> per-channel range, by type (sums over the type's nodes); "
          f"predicted = the criterion's per-channel / per-tensor ratio")
    print(f"   {'type':<6}{'n':>3} {'Y max/med':>9} {'Y span':>7} | {'dloss per-tensor':>16} {'per-channel':>12} "
          f"{'actual ratio':>12} | {'A predicts':>10} {'C predicts':>10}")
    for t in TYPES + ("all",):
        ns = [n for n in names if t == "all" or kind[n] == t]
        a_pt, a_pc = sum(R[n]["dloss_per-tensor"] for n in ns), sum(R[n]["dloss_per-channel"] for n in ns)
        se = np.sqrt(sum(R[n]["diff_se"] ** 2 for n in ns))
        ratio = {c: sum(R[n][f"pred_{c}_per-channel"] for n in ns) / sum(R[n][f"pred_{c}_per-tensor"] for n in ns)
                 for c in ("A", "C")}
        cy_ = np.median([R[n]["conc_y"]["max_over_median"] for n in ns])
        sy = np.median([R[n]["conc_y"]["median_span"] for n in ns])
        print(f"   {t:<6}{len(ns):>3} {cy_:9.1f} {sy:7.1%} | {a_pt:16.3f} {a_pc:12.3f} "
              f"{a_pc / a_pt if a_pt > 0 else float('nan'):12.2f} | {ratio['A']:10.2f} {ratio['C']:10.2f}"
              f"   (diff {a_pc - a_pt:+.3f}±{se:.3f})")

    sig = [n for n in names if R[n]["dloss_per-tensor"] > 2 * R[n]["dloss_se_per-tensor"]
           and R[n]["dloss_per-channel"] > 2 * R[n]["dloss_se_per-channel"]]
    act = np.array([np.log10(R[n]["dloss_per-channel"] / R[n]["dloss_per-tensor"]) for n in sig])
    print(f"\n2. per node ({len(sig)} nodes with dloss > 2 SE in both modes), log10 ratio per-channel / per-tensor")
    for c in ("A", "C"):
        p = np.array([np.log10(R[n][f"pred_{c}_per-channel"] / R[n][f"pred_{c}_per-tensor"]) for n in sig])
        print(f"   {c}: Spearman(predicted, actual) {spearmanr(p, act)[0]:+.2f}; median (predicted - actual) "
              f"{np.median(p - act):+.2f}; by type: " + ", ".join(
                  f"{t} {np.median((p - act)[[kind[n] == t for n in sig]]):+.2f}" for t in TYPES if any(kind[n] == t for n in sig)))
    span = np.array([min(R[n]["conc_x"]["median_span"], R[n]["conc_y"]["median_span"]) for n in sig])
    print(f"   Spearman(actual reduction, narrowest median channel span of X/Y) {spearmanr(span, act)[0]:+.2f} "
          f"(positive = the more uneven the ranges, the more per-channel helps)")
    print("\n3. per node, largest per-tensor damage:")
    for n in sorted(names, key=lambda n: -R[n]["dloss_per-tensor"])[:10]:
        r = R[n]
        print(f"   {r['type']:<5} {n:<16} Y max/med {r['conc_y']['max_over_median']:5.1f} span {r['conc_y']['median_span']:5.1%}"
              f" | dloss {r['dloss_per-tensor']:+.3f} -> {r['dloss_per-channel']:+.3f} (diff {r['diff']:+.3f}±{r['diff_se']:.3f})"
              f" | A predicts x{r['pred_A_per-channel'] / r['pred_A_per-tensor']:.2f}, "
              f"C x{r['pred_C_per-channel'] / r['pred_C_per-tensor']:.2f}, actual "
              f"x{r['dloss_per-channel'] / r['dloss_per-tensor'] if r['dloss_per-tensor'] > 0 else float('nan'):.2f}")


if __name__ == "__main__":
    main()
