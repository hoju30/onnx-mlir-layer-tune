#!/usr/bin/env python3
"""MobileNetV2: per-tensor Fisher with the empirical Fisher (true labels) vs the true Fisher (sampled labels).

Both are HTQ's Eq. 5 per tensor t in {W, input X, output Y}: C_t(f) = S_t * sum ||Q_f(t) - t||^2, S_t = mean(g_t^2);
only the gradient g behind S changes:
  empirical   g = dL(x, y_label)/dt, the label of each calibration image (run_mbv2_fisher_term_ablation.py's base)
  true        g = dL(x, y~softmax(model(x)))/dt, --samples labels drawn per image from the model's own prediction
              (E_y[g g^T] is the Gauss-Newton matrix, i.e. the loss Hessian's PSD part; needs no labels)
A. Per-node prediction: against the single-node held-out loss increases of runs/mbv2_predictor_check_20261003
   (Spearman, spread and per-layer-type bias of log10(prediction / dloss), as analyze_fisher_vs_sqnr.py).
B. MILP picks at budgets relative to all-int8, ORT QDQ simulation on the held-out images, with the saved SQNR pick.
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
from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from mp_milp import milp_select  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant, fold_batchnorms  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

RUNS = HERE / "runs"
VARIANTS = ("empirical", "true")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--samples", type=int, default=16, help="labels sampled per calibration image (true Fisher)")
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
    names = list(module_of)
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            cap[name] = (inp[0], out)
        return fn

    for n, m in module_of.items():
        m.register_forward_hook(hook(n))

    # --- squared gradients: per image, the label's (empirical) and --samples sampled labels' (true), averaged ---
    gen = torch.Generator().manual_seed(0)
    g2 = {v: {n: {"w": 0.0, "x": 0.0, "y": 0.0} for n in names} for v in VARIANTS}
    count = {n: {"x": 0, "y": 0} for n in names}
    d2 = {n: {s: dict.fromkeys(FORMATS[:-1], 0.0) for s in ("x", "y")} for n in names}
    for k in range(len(cy)):
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        z = model(x0)
        targets = [n2 for n in names for n2 in (module_of[n].weight, cap[n][0], cap[n][1])]
        labels = {"empirical": [int(cy[k])],
                  "true": torch.multinomial(F.softmax(z.detach(), 1)[0], args.samples, replacement=True,
                                            generator=gen).tolist()}
        for v, ys in labels.items():
            for y in ys:
                grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([y])), targets, retain_graph=True)
                for i, n in enumerate(names):
                    for j, s in enumerate(("w", "x", "y")):
                        g2[v][n][s] += float((grads[3 * i + j].double() ** 2).sum()) / len(ys)
        with torch.no_grad():
            for n, (x, y) in cap.items():
                for s, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    count[n][s] += t.numel()
                    for f in FORMATS[:-1]:
                        d2[n][s][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
    log(f"squared gradients: {len(cy)} images, 1 label (empirical) / {args.samples} sampled labels (true) each")
    T = {v: {} for v in VARIANTS}
    for n in names:
        W = inits[by_name[n].input[1]]
        dW2 = {f: float(((np_fake_quant_weight(W, f, None).astype(np.float64) - W.astype(np.float64)) ** 2).sum())
               for f in FORMATS[:-1]}
        for v in VARIANTS:
            G = g2[v][n]
            T[v][n] = {"w": {"fp32": 0.0, **{f: G["w"] / W.size * dW2[f] for f in FORMATS[:-1]}},
                       **{s: {"fp32": 0.0, **{f: G[s] / count[n][s] * d2[n][s][f] for f in FORMATS[:-1]}}
                          for s in ("x", "y")}}

    def value_fn(v):
        t = T[v]
        return lambda n, w, a: t[n]["w"][w] + t[n]["x"][a] + t[n]["y"][a]

    # --- A. per-node prediction against the single-node ground truth ---
    pc = json.loads((RUNS / "mbv2_predictor_check_20261003/results.json").read_text())
    kind = {n: layer_type(i, len(names)) for i, n in enumerate(names)}
    rows = pc["rows"]
    pred = {v: np.array([value_fn(v)(r["node"], r["w"], r["a"]) for r in rows]) for v in VARIANTS}
    saved = np.array([r["fisher"] for r in rows])
    log(f"empirical Fisher vs predictor_check's saved fisher column: max relative difference "
        f"{np.max(np.abs(pred['empirical'] - saved) / saved):.2e}")
    y = np.array([r["dloss"] for r in rows])
    sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in rows])
    pred_stats = {}
    print(f"\nA. per-node prediction vs single-node held-out dloss ({len(rows)} pairs, {sig.sum()} above 2 SE)")
    for v in VARIANTS:
        lr = np.log10(pred[v][sig] / y[sig])
        cen = lr - np.median(lr)
        by_cfg = {f"{w}/{a}": float(spearmanr(pred[v][ix], y[ix])[0]) for w, a in pc["configs"]
                  for ix in [np.array([(r["w"], r["a"]) == (w, a) for r in rows])]}
        types = {t: float(np.median(cen[[kind[r["node"]] == t for r in np.array(rows)[sig]]]))
                 for t in TYPES if any(kind[r["node"]] == t for r in np.array(rows)[sig])}
        pred_stats[v] = {"spearman_all": float(spearmanr(pred[v], y)[0]), "spearman_by_config": by_cfg,
                         "log10_ratio_iqr": float(np.subtract(*np.percentile(lr, [75, 25]))), "type_bias": types}
        print(f"  {v:<9} Spearman all {pred_stats[v]['spearman_all']:+.3f}  "
              + "  ".join(f"{c} {s:+.3f}" for c, s in by_cfg.items())
              + f"  | log10 ratio IQR {pred_stats[v]['log10_ratio_iqr']:.2f} | type bias "
              + " ".join(f"{t} {b:+.2f}" for t, b in types.items()))

    # --- B. MILP + held-out evaluation ---
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
    res, out = {}, []
    for f in budgets:
        cfgs = {}
        for v in VARIANTS:
            cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, value_fn(v), weight_elems, flops_proxy,
                                         f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1), (v, f, status)
            cfgs[v] = ({n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}, status)
        s = next(c for c in sq if c["criterion"] == "sqnr" and abs(c["budget"] - f) < 1e-9)
        cfgs["sqnr"] = ({n: {"w": s["formats"][n]["w"], "a": s["formats"][n]["a"]} for n in names}, 0)
        for v, (cfg, status) in cfgs.items():
            size, bops = config_cost(cfg, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            correct = ((z.argmax(1) == ey) & finite).astype(float)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(ey),
                                   reduction="none").numpy()
            res[(f, v)] = (correct, loss)
            out.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                        "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                        "formats": cfg})
            log(f"f={f:.2f} {v:<9} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in out:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "empirical")]
        d1, dl = (c - cb) * 100, l - lb
        base = next(x for x in out if x["budget"] == r["budget"] and x["variant"] == "empirical")["formats"]
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                 nodes_differing_from_empirical=sum(r["formats"][n] != base[n] for n in names))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets, "samples": args.samples,
         "num_calibration": len(cy), "num_eval": len(ey), "prediction": pred_stats, "rows": out}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<10} {'top1':>7} {'loss':>7} {'d top1 vs empirical':>20} {'d loss':>17} "
          f"{'size':>5} {'BOPs':>5} {'nodes differ':>13}")
    for r in out:
        d = "" if r["variant"] == "empirical" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "empirical" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<10} {r['top1']:7.1%} "
              f"{r['loss']:7.4f} {d:>20} {dl:>17} {r['size']:5.2f} {r['bops']:5.2f} "
              f"{r['nodes_differing_from_empirical']:>8d}/{len(names)}")


if __name__ == "__main__":
    main()
