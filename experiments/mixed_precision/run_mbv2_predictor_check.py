#!/usr/bin/env python3
"""MobileNetV2: how well do per-node predictors rank the real damage of quantizing ONE node?

Ground truth: for every weight-bearing node and each (w, a) in CONFIGS, quantize only that node (ORT QDQ
simulation, exactly the pass's arithmetic) and measure the held-out cross-entropy increase over FP32.
Predictors, from the calibration images:
  fisher        mean(g_W^2)*||dW||^2 + mean(g_X^2)*sum||dX||^2 + mean(g_Y^2)*sum||dY||^2   (needs gradients)
  joint         sum ||Y_joint - Y||^2, Y_joint = Q_a(op(Q_a(X), Q_w(W)) + b): the node's own output error with
                weight, input and output quantized together, as the pass does (forward only)
  g2y_joint     mean(g_Y^2) * sum ||Y_joint - Y||^2   (joint error weighted by the output's gradient)
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
from scipy.stats import kendalltau, spearmanr  # noqa: E402

from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402

CONFIGS = [("int4", "int8"), ("int8", "int4"), ("int6", "int6")]
PREDICTORS = ("fisher", "joint", "g2y_joint")


def node_op(m, x, w):
    return m._conv_forward(x, w, m.bias) if isinstance(m, nn.Conv2d) else F.linear(x, w, m.bias)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
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
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    wfmts, afmts = sorted({w for w, _ in CONFIGS}), sorted({a for _, a in CONFIGS})
    qw = {n: {f: torch.from_numpy(np_fake_quant_weight(m.weight.detach().numpy(), f, None)) for f in wfmts}
          for n, m in module_of.items()}
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, m in module_of.items():
        m.register_forward_hook(hook(n))
    acc = {n: {"g2w": 0.0, **{s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(afmts, 0.0)} for s in ("x", "y")},
               "joint": dict.fromkeys(CONFIGS, 0.0)} for n in module_of}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        F.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                m, a = module_of[n], acc[n]
                a["g2w"] += float((m.weight.grad.double() ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    s = a[side]
                    s["g2"] += float((t.grad.double() ** 2).sum())
                    s["count"] += t.numel()
                    for f in afmts:
                        s["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
                for w, af in CONFIGS:
                    yj = act_fake_quant(node_op(m, act_fake_quant(x, af, ranges[tens[n][0]]), qw[n][w]), af,
                                        ranges[tens[n][1]])
                    a["joint"][(w, af)] += float(((yj - y).double() ** 2).sum())
        cap.clear()
    pred = {}
    for n, m in module_of.items():
        a = acc[n]
        W = m.weight.detach().numpy().astype(np.float64)
        mg2 = {s: a[s]["g2"] / a[s]["count"] for s in ("x", "y")}
        for w, af in CONFIGS:
            dW2 = float(((qw[n][w].numpy().astype(np.float64) - W) ** 2).sum())
            pred[(n, w, af)] = {
                "fisher": a["g2w"] / W.size * dW2 + mg2["x"] * a["x"]["d2"][af] + mg2["y"] * a["y"]["d2"][af],
                "joint": a["joint"][(w, af)],
                "g2y_joint": mg2["y"] * a["joint"][(w, af)]}
    log("predictors ready on the calibration images")

    # --- ground truth: one node quantized at a time ---
    def per_image_loss(fmts):
        z = simulate_in_onnxruntime(graph, all_nodes, fmts, ranges, ex, None)
        assert np.isfinite(z).all()
        return F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey), reduction="none").numpy()

    base = per_image_loss({})
    rows = []
    for i, n in enumerate(module_of):
        for w, af in CONFIGS:
            d = per_image_loss({n: {"w": w, "a": af}}) - base
            rows.append({"node": n, "w": w, "a": af, "dloss": float(d.mean()),
                         "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d))), **pred[(n, w, af)]})
        if (i + 1) % 10 == 0:
            log(f"ground truth: {i + 1}/{len(module_of)} nodes x {len(CONFIGS)} configs")

    def agreement(sub):
        y = np.array([r["dloss"] for r in sub])
        top = set(np.argsort(-y)[:args.top_k])
        out = {}
        for p in PREDICTORS:
            x = np.array([r[p] for r in sub])
            out[p] = {"spearman": float(spearmanr(x, y)[0]), "kendall": float(kendalltau(x, y)[0]),
                      f"top{args.top_k}": len(top & set(np.argsort(-x)[:args.top_k]))}
        return out

    stats = {"all": agreement(rows)}
    for w, af in CONFIGS:
        stats[f"{w}/{af}"] = agreement([r for r in rows if (r["w"], r["a"]) == (w, af)])
    rng = np.random.default_rng(0)
    y = np.array([r["dloss"] for r in rows])
    P = {p: np.array([r[p] for r in rows]) for p in PREDICTORS}
    boot = {}
    for p in ("joint", "g2y_joint"):
        diffs = [spearmanr(P["fisher"][ix], y[ix])[0] - spearmanr(P[p][ix], y[ix])[0]
                 for ix in rng.integers(0, len(rows), (3000, len(rows)))]
        boot[f"fisher_minus_{p}"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
    significant = int(sum(r["dloss"] > 2 * r["dloss_se"] for r in rows))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "num_eval": len(ey), "configs": CONFIGS,
         "agreement": stats, "spearman_diff_ci95": boot, "pairs_above_2se": significant,
         "rows": rows}, indent=2) + "\n")

    print(f"\n{len(rows)} (node, config) pairs, {len(ey)} held-out images; {significant} pairs with dloss > 2 SE")
    for group, s in stats.items():
        print(f"  {group:<10}" + "   ".join(f"{p} rho {s[p]['spearman']:+.2f} tau {s[p]['kendall']:+.2f} "
                                           f"top{args.top_k} {s[p][f'top{args.top_k}']}" for p in PREDICTORS))
    for k, (lo, hi) in boot.items():
        print(f"  Spearman {k.replace('_minus_', ' - ')}: 95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  top {args.top_k} pairs by measured dloss (rank under each predictor):")
    ranks = {p: {id(rows[i]): rk for rk, i in enumerate(np.argsort(-P[p]), 1)} for p in PREDICTORS}
    for r in sorted(rows, key=lambda r: -r["dloss"])[:args.top_k]:
        print(f"    {r['node']:<16} w={r['w']:<5} a={r['a']:<5} dloss {r['dloss']:+.4f}±{r['dloss_se']:.4f}  "
              + "  ".join(f"{p} #{ranks[p][id(r)]}" for p in PREDICTORS))


if __name__ == "__main__":
    main()
