#!/usr/bin/env python3
"""MobileNetV2: per-tensor vs element-wise Fisher cost, the GPT-2 ablation (run_gpt2_fisher_ablation.py) on a CNN.
For every quantized tensor t of a node (weight W, input X, output Y; a= quantizes X and Y) and format f:
  scalar        C = mean(g^2) * sum_j dt_j^2          (run_mbv2_fisher_term_ablation.py's base)
  elementwise   C = sum_j g_j^2 * dt_j^2
g = dL/dt per calibration image (cross-entropy with its label), summed over the calibration images; dt = the
pass's fake-quant error. MILP at budgets relative to all-int8; ORT QDQ simulation on the held-out images.
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

from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from mp_milp import milp_select  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant, fold_batchnorms  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

VARIANTS = ("scalar", "elementwise")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--milp-time-limit", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(f) for f in args.budgets.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]]
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
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, m in module_of.items():
        m.register_forward_hook(hook(n))
    g2w = {n: torch.zeros_like(m.weight, dtype=torch.float64) for n, m in module_of.items()}   # per element
    act = {n: {s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0), "elem": dict.fromkeys(FORMATS, 0.0)}
               for s in ("x", "y")} for n in module_of}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        nn.functional.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g2w[n] += module_of[n].weight.grad.double() ** 2
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a, g2 = act[n][side], t.grad.double() ** 2
                    a["g2"] += float(g2.sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        d2 = (act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2
                        a["d2"][f] += float(d2.sum())
                        a["elem"][f] += float((g2 * d2).sum())
        cap.clear()
    T = {v: {"w": {}, "x": {}, "y": {}} for v in VARIANTS}
    for n in module_of:
        W = inits[by_name[n].input[1]]
        T["scalar"]["w"][n], T["elementwise"]["w"][n] = {"fp32": 0.0}, {"fp32": 0.0}
        for f in FORMATS[:-1]:
            dW2 = torch.from_numpy((np_fake_quant_weight(W, f, None).astype(np.float64) - W.astype(np.float64)) ** 2)
            T["scalar"]["w"][n][f] = float(g2w[n].mean()) * float(dW2.sum())
            T["elementwise"]["w"][n][f] = float((g2w[n] * dW2).sum())
        for side in ("x", "y"):
            a = act[n][side]
            T["scalar"][side][n] = {"fp32": 0.0, **{f: a["g2"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
            T["elementwise"][side][n] = {"fp32": 0.0, **{f: a["elem"][f] for f in FORMATS[:-1]}}
    log("scalar and element-wise Fisher tables ready")

    # how much element-wise alignment changes each term: ratio elementwise / scalar at int4, over the 53 nodes
    for t in ("w", "x", "y"):
        q = np.array([T["elementwise"][t][n]["int4"] / T["scalar"][t][n]["int4"] for n in module_of
                      if T["scalar"][t][n]["int4"] > 0])
        print(f"  {t}: element-wise / scalar at int4: median {np.median(q):.2f}, range {q.min():.2f}-{q.max():.2f}",
              flush=True)

    def value_fn(v):
        t = T[v]
        return lambda n, w, a: t["w"][n][w] + t["x"][n][a] + t["y"][n][a]

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    names = list(module_of)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    res, rows = {}, []
    for f in budgets:
        for v in VARIANTS:
            cfg, obj, status = milp_select(all_nodes, FORMATS, FORMATS, value_fn(v), weight_elems, flops_proxy,
                                           f * int8_bytes, f * int8_bops, time_limit=args.milp_time_limit)
            assert status in (0, 1), (v, f, status)
            size, bops = config_cost(cfg, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            correct = ((z.argmax(1) == ey) & finite).astype(float)
            loss = nn.functional.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)),
                                               torch.from_numpy(ey), reduction="none").numpy()
            res[(f, v)] = (correct, loss)
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "nonfinite_images": int((~finite).sum()), "formats": cfg})
            log(f"f={f:.2f} {v:<11} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for row in rows:
        c, l = res[(row["budget"], row["variant"])]
        cb, lb = res[(row["budget"], "scalar")]
        d1, dl = (c - cb) * 100, l - lb
        pick = next(r for r in rows if r["budget"] == row["budget"] and r["variant"] == "scalar")["formats"]
        row.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                   dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                   nodes_differing_from_scalar=sum(row["formats"][n] != pick[n] for n in names))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets, "rows": rows}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<12} {'top1':>7} {'loss':>7} {'d top1 vs scalar':>17} {'d loss':>17} "
          f"{'size':>5} {'BOPs':>5} {'nodes differ':>13}")
    for row in rows:
        d = "" if row["variant"] == "scalar" else f"{row['dtop1_pp']:+6.2f}±{row['dtop1_se']:.2f}"
        dl = "" if row["variant"] == "scalar" else f"{row['dloss']:+.4f}±{row['dloss_se']:.4f}"
        print(f"{row['budget']:5.2f} {row['variant'] + ('' if row['milp_optimal'] else '*'):<12} {row['top1']:7.1%} "
              f"{row['loss']:7.4f} {d:>17} {dl:>17} {row['size']:5.2f} {row['bops']:5.2f} "
              f"{row['nodes_differing_from_scalar']:>8d}/{len(names)}")


if __name__ == "__main__":
    main()
