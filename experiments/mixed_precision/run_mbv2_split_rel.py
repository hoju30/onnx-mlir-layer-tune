#!/usr/bin/env python3
"""MobileNetV2: shared_v2 with the shared weight SNIP replaced by each quantized tensor's own SNIP
(run_gpt2_split_rel.py on an ImageNet CNN, with run_mbv2_fisher_term_ablation.py's folded weights and labels).
  shared_v2   S_W * (r_W + r_X + r_Y)
  split_v2    S_W * r_W + S_X * r_X + S_Y * r_Y
  split_abs   S_W * ||dW||^2 + S_X * E||dX||^2 + S_Y * E||dY||^2   (HTQ Eq. 5's absolute error, per tensor)
  fisher      mean(g^2) * ||delta||^2 per tensor (reference)
S_t = mean|t * dL/dt| per calibration image, averaged; r_t = ||Q(t) - t||^2 / ||t||^2 on full tensors.
MILP at budgets relative to all-int8; ORT QDQ simulation on the held-out images.
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

VARIANTS = ("shared_v2", "split_v2", "split_abs", "fisher")


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
    K = len(cy)
    acc = {n: {"conn_w": 0.0, "g2w": 0.0,
               **{s: {"conn": 0.0, "g2": 0.0, "count": 0, "t2": 0.0, "d2": dict.fromkeys(FORMATS, 0.0)}
                  for s in ("x", "y")}} for n in module_of}
    for k in range(K):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        nn.functional.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                W = module_of[n].weight
                g = W.grad.double()
                a = acc[n]
                a["conn_w"] += float((W.double() * g).abs().mean()) / K
                a["g2w"] += float((g ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    s, gt, td = a[side], t.grad.double(), t.double()
                    s["conn"] += float((td * gt).abs().mean()) / K
                    s["g2"] += float((gt ** 2).sum())
                    s["count"] += t.numel()
                    s["t2"] += float((td ** 2).sum())
                    for f in FORMATS[:-1]:
                        s["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
    r, F, S, E = {}, {}, {}, {}
    for n in module_of:
        W = inits[by_name[n].input[1]]
        Wd = W.astype(np.float64)
        dW2 = {f: float(((np_fake_quant_weight(W, f, None).astype(np.float64) - Wd) ** 2).sum()) for f in FORMATS[:-1]}
        a = acc[n]
        S[n] = {"w": a["conn_w"], "x": a["x"]["conn"], "y": a["y"]["conn"]}
        r[n] = {"w": {"fp32": 0.0, **{f: v / float((Wd ** 2).sum()) for f, v in dW2.items()}},
                **{s: {"fp32": 0.0, **{f: a[s]["d2"][f] / a[s]["t2"] for f in FORMATS[:-1]}} for s in ("x", "y")}}
        # absolute squared error (HTQ Eq. 5 / Fisher's error term); activations per calibration sample
        E[n] = {"w": {"fp32": 0.0, **dW2},
                **{s: {"fp32": 0.0, **{f: a[s]["d2"][f] / K for f in FORMATS[:-1]}} for s in ("x", "y")}}
        F[n] = {"w": {"fp32": 0.0, **{f: a["g2w"] / W.size * v for f, v in dW2.items()}},
                **{s: {"fp32": 0.0, **{f: a[s]["g2"] / a[s]["count"] * a[s]["d2"][f] for f in FORMATS[:-1]}}
                   for s in ("x", "y")}}
    sw = np.array([S[n]["w"] for n in module_of])
    ratio = {s: np.array([S[n][s] for n in module_of]) / sw for s in ("x", "y")}
    terms = {v: {t: sum((S[n]["w"] if v == "shared_v2" else S[n][t]) * r[n][t]["int4"] for n in module_of)
                 for t in ("w", "x", "y")} for v in ("shared_v2", "split_v2")}
    terms["split_abs"] = {t: sum(S[n][t] * E[n][t]["int4"] for n in module_of) for t in ("w", "x", "y")}
    log("tables ready")
    print(f"\nS_X/S_W median {np.median(ratio['x']):.2f} (range {ratio['x'].min():.2f}-{ratio['x'].max():.2f}), "
          f"S_Y/S_W median {np.median(ratio['y']):.2f} (range {ratio['y'].min():.2f}-{ratio['y'].max():.2f})")
    for v, t in terms.items():
        print(f"{v:<10} summed terms at int4: W {t['w']:.3g}  X {t['x']:.3g}  Y {t['y']:.3g}  "
              f"(X/W {t['x'] / t['w']:.2f}, Y/W {t['y'] / t['w']:.2f})", flush=True)

    def value_fn(v):
        if v == "fisher":
            return lambda n, w, a: F[n]["w"][w] + F[n]["x"][a] + F[n]["y"][a]
        if v == "shared_v2":
            return lambda n, w, a: S[n]["w"] * (r[n]["w"][w] + r[n]["x"][a] + r[n]["y"][a])
        if v == "split_abs":
            return lambda n, w, a: S[n]["w"] * E[n]["w"][w] + S[n]["x"] * E[n]["x"][a] + S[n]["y"] * E[n]["y"][a]
        return lambda n, w, a: S[n]["w"] * r[n]["w"][w] + S[n]["x"] * r[n]["x"][a] + S[n]["y"] * r[n]["y"][a]

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
            wq = [c["w"] for c in cfg.values() if c["w"]]
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "nonfinite_images": int((~finite).sum()), "int4_weights": wq.count("int4"), "formats": cfg})
            log(f"f={f:.2f} {v:<10} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for row in rows:
        c, l = res[(row["budget"], row["variant"])]
        cb, lb = res[(row["budget"], "shared_v2")]
        d1, dl = (c - cb) * 100, l - lb
        row.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                   dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets, "S": S,
         "term_balance_int4": terms, "rows": rows}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<11} {'top1':>7} {'loss':>7} {'d top1 vs shared_v2':>20} {'d loss':>17} "
          f"{'size':>5} {'BOPs':>5} {'int4 W':>7}")
    for row in rows:
        d = "" if row["variant"] == "shared_v2" else f"{row['dtop1_pp']:+6.2f}±{row['dtop1_se']:.2f}"
        dl = "" if row["variant"] == "shared_v2" else f"{row['dloss']:+.4f}±{row['dloss_se']:.4f}"
        print(f"{row['budget']:5.2f} {row['variant'] + ('' if row['milp_optimal'] else '*'):<11} {row['top1']:7.1%} "
              f"{row['loss']:7.4f} {d:>20} {dl:>17} {row['size']:5.2f} {row['bops']:5.2f} {row['int4_weights']:7d}")


if __name__ == "__main__":
    main()
