#!/usr/bin/env python3
"""MobileNetV2: is the per-tensor Fisher's under-estimate of low-bit activation damage real, and where does it come from?

Activation-only, one node at a time (w=fp32; a= quantizes the node's input and output as the pass does), at --bits
(default int6 and int4), held-out loss increase over FP32 (PyTorch forward with the pass's fake-quant arithmetic, BatchNorm
folded) against the per-tensor Fisher prediction mean(g_X^2)*sum||dX||^2 + mean(g_Y^2)*sum||dY||^2 (the current
criterion, from the calibration images).
  plain       Q(t)                          (what the pass does)
  debiased    Q(t) - mu_c, mu_c the per-channel mean of Q(t) - t on the calibration images (DFQ-style bias
              correction), with the Fisher prediction on the de-meaned error
If dloss / Fisher grows as the bits drop, low-bit activations are under-estimated. If that growth disappears for
debiased, the cause is the error's per-channel mean (a coherent bias the per-element model treats as independent
noise); if it stays, it is a finite-noise nonlinearity.
Consistency (--check-nodes): a few nodes at w=int8/a=int4 against the ORT numbers of
runs/mbv2_predictor_check_20261003 (nodes 3, 4, 5, 25, 37, 51 matched to 1e-4 on 300 images).
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

from analyze_fisher_vs_sqnr import TYPES, layer_type  # noqa: E402
from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         supported_nodes)
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402

MODES = ("plain", "debiased")


def chan_shape(t):
    return (1, -1, 1, 1) if t.dim() == 4 else (1, -1)


def chan_sum(t):
    return t.sum(dim=(0, 2, 3)) if t.dim() == 4 else t.sum(dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--bits", default="int6,int4", help="activation formats, highest first")
    ap.add_argument("--check-nodes", default="", help="weighted-node indices to check against ORT, e.g. 3,4,5,25,37,51")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    BITS = tuple(args.bits.split(","))
    lo, hi = BITS[-1], BITS[0]
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

    # --- calibration: squared gradients, per-channel error sums for the Fisher prediction and the bias correction ---
    cap = {}
    hs = [m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
          for n, m in module_of.items()]
    g2 = {n: {"x": 0.0, "y": 0.0} for n in names}
    cnt = {n: {"x": 0, "y": 0} for n in names}
    s1 = {n: {s: {b: 0.0 for b in BITS} for s in ("x", "y")} for n in names}   # per-channel sum of error
    s2 = {n: {s: {b: 0.0 for b in BITS} for s in ("x", "y")} for n in names}   # sum of squared error
    nc = {n: {"x": 0, "y": 0} for n in names}                                   # elements per channel
    for k in range(len(cy)):
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        z = model(x0)
        targets = [t for n in names for t in cap[n]]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                for j, s in enumerate(("x", "y")):
                    t = cap[n][j]
                    g2[n][s] += float((grads[2 * i + j].double() ** 2).sum())
                    cnt[n][s] += t.numel()
                    nc[n][s] += t.numel() // t.shape[1]
                    for b in BITS:
                        d = (act_fake_quant(t, b, ranges[tens[n][j]]) - t).double()
                        s1[n][s][b] = s1[n][s][b] + chan_sum(d)
                        s2[n][s][b] += float((d ** 2).sum())
        cap.clear()
    for h in hs:
        h.remove()
    mu = {n: {s: {b: (s1[n][s][b] / nc[n][s]).float() for b in BITS} for s in ("x", "y")} for n in names}
    fisher, mean_share = {}, {}
    for n in names:
        for b in BITS:
            plain = sum(g2[n][s] / cnt[n][s] * s2[n][s][b] for s in ("x", "y"))
            demeaned = {s: s2[n][s][b] - nc[n][s] * float((mu[n][s][b].double() ** 2).sum()) for s in ("x", "y")}
            fisher[(n, b, "plain")] = plain
            fisher[(n, b, "debiased")] = sum(g2[n][s] / cnt[n][s] * demeaned[s] for s in ("x", "y"))
            mean_share[(n, b)] = 1 - sum(demeaned.values()) / sum(s2[n][s][b] for s in ("x", "y"))
    log("calibration statistics ready")

    # --- held-out ground truth, one node at a time ---
    xs, ys = torch.from_numpy(ex), torch.from_numpy(ey)
    current = {}

    def pre(mod, inp, n):
        b, mode = current[n]
        q = act_fake_quant(inp[0], b, ranges[tens[n][0]])
        return (q - mu[n]["x"][b].view(chan_shape(q)) if mode == "debiased" else q,)

    def post(mod, inp, out, n):
        b, mode = current[n]
        q = act_fake_quant(out, b, ranges[tens[n][1]])
        return q - mu[n]["y"][b].view(chan_shape(q)) if mode == "debiased" else q

    @torch.no_grad()
    def per_image_loss():
        return torch.cat([F.cross_entropy(model(xs[i:i + 100]), ys[i:i + 100], reduction="none")
                          for i in range(0, len(ys), 100)]).numpy()

    base = per_image_loss()
    log(f"FP32 held-out loss {base.mean():.4f} on {len(ys)} images")

    # consistency with the ORT single-node numbers (w=int8, a=int4)
    pc = {(r["node"], r["w"], r["a"]): r for r in
          json.loads((HERE / "runs/mbv2_predictor_check_20261003/results.json").read_text())["rows"]}
    check = []
    for i in [int(i) for i in args.check_nodes.split(",") if i]:
        n, m = names[i], module_of[names[i]]
        orig = m.weight.data.clone()
        m.weight.data.copy_(torch.from_numpy(np_fake_quant_weight(orig.numpy(), "int8", None)))
        current[n] = ("int4", "plain")
        h = [m.register_forward_pre_hook(lambda mod, inp, n=n: pre(mod, inp, n)),
             m.register_forward_hook(lambda mod, inp, out, n=n: post(mod, inp, out, n))]
        d = float((per_image_loss() - base).mean())
        for x in h:
            x.remove()
        m.weight.data.copy_(orig)
        ref = pc[(n, "int8", "int4")]
        check.append({"node": n, "type": kind[n], "torch": d, "ort": ref["dloss"], "ort_se": ref["dloss_se"]})
        log(f"check {kind[n]:<4} {n}: torch dloss {d:+.4f}  ORT {ref['dloss']:+.4f}±{ref['dloss_se']:.4f}")

    rows = []
    for i, n in enumerate(names):
        m = module_of[n]
        h = [m.register_forward_pre_hook(lambda mod, inp, n=n: pre(mod, inp, n)),
             m.register_forward_hook(lambda mod, inp, out, n=n: post(mod, inp, out, n))]
        for b in BITS:
            for mode in MODES:
                current[n] = (b, mode)
                d = per_image_loss() - base
                rows.append({"node": n, "type": kind[n], "bits": b, "mode": mode, "dloss": float(d.mean()),
                             "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d))), "fisher": fisher[(n, b, mode)],
                             "mean_error_share": mean_share[(n, b)]})
        for x in h:
            x.remove()
        if (i + 1) % 10 == 0:
            log(f"ground truth: {i + 1}/{len(names)} nodes")

    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "PyTorch, pass fake-quant arithmetic", "model": args.timm_name, "num_eval": len(ys),
         "bits": BITS, "consistency_check": check, "rows": rows}, indent=2) + "\n")

    # --- summary ---
    R = {(r["node"], r["bits"], r["mode"]): r for r in rows}
    print("\n1. share of the activation error power that is the per-channel mean (calibration images), median by type")
    print("   " + " ".join(f"{t:>7}" for t in TYPES))
    for b in BITS:
        print(f"   {b}: " + " ".join(f"{np.median([mean_share[(n, b)] for n in names if kind[n] == t]):7.1%}"
                                    for t in TYPES))
    for mode in MODES:
        print(f"\n2{'a' if mode == 'plain' else 'b'}. {mode}: actual / Fisher-predicted damage, by layer type "
              f"(sum of dloss over the type's nodes / sum of Fisher; normalized to the all-node {hi} value)")
        tot6 = sum(R[(n, hi, mode)]["dloss"] for n in names) / sum(R[(n, hi, mode)]["fisher"] for n in names)
        print(f"   {'type':<6} " + " ".join(f"{b:>7}" for b in BITS) + f"   {lo}/{hi}   total dloss {hi} / {lo}")
        for t in TYPES + ("all",):
            ns = [n for n in names if t == "all" or kind[n] == t]
            ratio = {b: sum(R[(n, b, mode)]["dloss"] for n in ns) / sum(R[(n, b, mode)]["fisher"] for n in ns) / tot6
                     for b in BITS}
            print(f"   {t:<6} " + " ".join(f"{ratio[b]:7.2f}" for b in BITS) + f"   {ratio[lo] / ratio[hi]:8.2f}x"
                  f"   {sum(R[(n, hi, mode)]['dloss'] for n in ns):+.3f} / {sum(R[(n, lo, mode)]['dloss'] for n in ns):+.3f}")
    print(f"\n3. per node, {lo}: dloss plain -> debiased (nodes with plain dloss > 2 SE, largest 12)")
    big = sorted([n for n in names if R[(n, lo, "plain")]["dloss"] > 2 * R[(n, lo, "plain")]["dloss_se"]],
                 key=lambda n: -R[(n, lo, "plain")]["dloss"])[:12]
    for n in big:
        p, d = R[(n, lo, "plain")], R[(n, lo, "debiased")]
        print(f"   {kind[n]:<5} {n:<16} {p['dloss']:+.4f}±{p['dloss_se']:.4f} -> {d['dloss']:+.4f}±{d['dloss_se']:.4f}"
              f"   mean-error share {mean_share[(n, lo)]:5.1%}   dloss/Fisher {p['dloss'] / p['fisher']:.3g} -> "
              f"{d['dloss'] / d['fisher']:.3g}")


if __name__ == "__main__":
    main()
