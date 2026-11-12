#!/usr/bin/env python3
"""MobileNetV2: absolute-gradient and squared-gradient surrogates as the
MILP criterion -- run_gpt2_first_order.py on an ImageNet CNN, with run_mbv2_fisher_term_ablation.py's folded
weights, labels and budgets (relative to all-int8).
  2nd_tensor   mean(g^2) * sum delta^2        (run_mbv2_fisher_term_ablation.py's base)
  1st_tensor   mean|g| * sum |delta|
  1st_elem     sum |g * delta|
  abs_g_sq_error mean|g| * sum delta^2 (fixed-error sensitivity exponent diagnostic)
These are not exact loss changes; abs_g_sq_error does not preserve Fisher's scaling properties.
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

VARIANTS = ("2nd_tensor", "1st_tensor", "1st_elem")
SENSITIVITY_VARIANTS = ("abs_g_sq_error", "abs_w_sq_error", "abs_x_sq_error", "abs_y_sq_error")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--milp-time-limit", type=float, default=60.0)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="Comma-separated variants, including 2nd_tensor as reference. abs_g_sq_error "
                         "uses mean|g| with the SAME squared errors; abs_w/x/y_sq_error changes only that term.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    variants = args.variants.split(",")
    if "2nd_tensor" not in variants or any(v not in VARIANTS + SENSITIVITY_VARIANTS for v in variants):
        ap.error("--variants must include 2nd_tensor and only supported variant names")
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
    w_abs = {n: torch.zeros_like(m.weight, dtype=torch.float64) for n, m in module_of.items()}
    w_sq = dict.fromkeys(module_of, 0.0)
    act = {n: {s: {"g1": 0.0, "g2": 0.0, "count": 0, "d1": dict.fromkeys(FORMATS, 0.0),
                   "d2": dict.fromkeys(FORMATS, 0.0), "e1": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")}
           for n in module_of}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        nn.functional.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g = module_of[n].weight.grad.double()
                w_abs[n] += g.abs()
                w_sq[n] += float((g ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a, gt = act[n][side], t.grad.double()
                    a["g1"] += float(gt.abs().sum())
                    a["g2"] += float((gt ** 2).sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        d = (act_fake_quant(t, f, ranges[rng_name]) - t).double()
                        a["d1"][f] += float(d.abs().sum())
                        a["d2"][f] += float((d ** 2).sum())
                        a["e1"][f] += float((gt * d).abs().sum())
        cap.clear()
    table_variants = VARIANTS + ("abs_g_sq_error",)
    T = {v: {"w": {}, "x": {}, "y": {}} for v in table_variants}
    for n in module_of:
        W = inits[by_name[n].input[1]]
        for v in table_variants:
            T[v]["w"][n] = {"fp32": 0.0}
        for f in FORMATS[:-1]:
            dW = torch.from_numpy(np_fake_quant_weight(W, f, None).astype(np.float64) - W.astype(np.float64))
            T["2nd_tensor"]["w"][n][f] = w_sq[n] / W.size * float((dW ** 2).sum())
            T["1st_tensor"]["w"][n][f] = float(w_abs[n].sum()) / W.size * float(dW.abs().sum())
            T["1st_elem"]["w"][n][f] = float((w_abs[n] * dW).abs().sum())
            T["abs_g_sq_error"]["w"][n][f] = float(w_abs[n].sum()) / W.size * float((dW ** 2).sum())
        for side in ("x", "y"):
            a = act[n][side]
            T["2nd_tensor"][side][n] = {"fp32": 0.0, **{f: a["g2"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
            T["1st_tensor"][side][n] = {"fp32": 0.0, **{f: a["g1"] / a["count"] * a["d1"][f] for f in FORMATS[:-1]}}
            T["1st_elem"][side][n] = {"fp32": 0.0, **{f: a["e1"][f] for f in FORMATS[:-1]}}
            T["abs_g_sq_error"][side][n] = {"fp32": 0.0, **{f: a["g1"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
    for side in ("w", "x", "y"):
        T[f"abs_{side}_sq_error"] = {t: T["abs_g_sq_error" if t == side else "2nd_tensor"][t]
                                      for t in ("w", "x", "y")}
    # |g| * delta^2 isolates the gradient exponent but is not a loss-unit surrogate.
    (args.out_dir / "criterion_tables.json").write_text(json.dumps(T) + "\n")
    log("first- and second-order tables ready")

    def value_fn(v):
        t = T[v]
        return lambda n, w, a: t["w"][n][w] + t["x"][n][a] + t["y"][n][a]

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    names = list(module_of)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    res, rows = {}, []
    for f in budgets:
        for v in variants:
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
            wq = [c["w"] for c in cfg.values() if c["w"]]
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "nonfinite_images": int((~finite).sum()), "int4_weights": wq.count("int4"), "formats": cfg})
            np.savez(args.out_dir / f"metrics_f{f:g}_{v}.npz", correct=correct, loss=loss)
            log(f"f={f:.2f} {v:<10} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in rows:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "2nd_tensor")]
        d1, dl = (c - cb) * 100, l - lb
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets,
         "variants": variants, "calibration_images": len(cy), "test_images": len(ey),
         "model_dir": str(args.model_dir), "images_dir": str(args.images_dir),
         "milp_time_limit": args.milp_time_limit, "rows": rows}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<11} {'top1':>7} {'loss':>7} {'d top1 (pp)':>14} {'d loss':>17} {'size':>5} "
          f"{'BOPs':>5} {'int4 W':>7}")
    for r in rows:
        d = "" if r["variant"] == "2nd_tensor" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "2nd_tensor" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<11} {r['top1']:7.1%} "
              f"{r['loss']:7.4f} {d:>14} {dl:>17} {r['size']:5.2f} {r['bops']:5.2f} {r['int4_weights']:7d}")


if __name__ == "__main__":
    main()
