#!/usr/bin/env python3
"""MobileNetV2: does the additive (per-node sum) model pick the wrong config, despite under-estimating the damage?

At one budget, a set of plausible candidate configs (earlier criteria's MILP picks, plus near-optimal competitors:
the Fisher MILP re-solved with each node's cost randomly perturbed) is scored three ways:
  actual      held-out loss increase with the whole config quantized
  additive    sum of each node's measured single-node increase (perfect per-node prediction, no interactions)
  fisher      sum of the per-tensor Fisher costs (the criterion actually used)
and each score's ranking of the candidates is compared with the actual ranking (Spearman, regret of its pick).
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

from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from mp_milp import milp_select  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant, fold_batchnorms  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

RUNS = HERE / "runs"
EARLIER = [  # (label, results file, variant) -- MILP picks of earlier criteria that did not collapse
    ("fisher", "mbv2_fisher_term_ablation_20261002/results.json", "base"),
    ("fisher elementwise", "mbv2_fisher_elementwise_20261003/results.json", "elementwise"),
    ("fisher drop input", "mbv2_fisher_term_ablation_20261002/results.json", "drop_x"),
    ("fisher avg input", "mbv2_fisher_term_ablation_20261002/results.json", "avg_x"),
    ("fisher avg output", "mbv2_fisher_term_ablation_20261002/results.json", "avg_y"),
    ("shared_v2", "mbv2_split_abs_20261003/results.json", "shared_v2"),
    ("split_v2", "mbv2_split_abs_20261003/results.json", "split_v2"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budget", type=float, default=0.7)
    ap.add_argument("--perturbed", type=int, default=8, help="near-optimal competitors from perturbed Fisher costs")
    ap.add_argument("--sigma", type=float, default=0.5, help="log-normal sigma of the per-node cost perturbation")
    ap.add_argument("--num-eval", type=int, default=300)
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

    # --- per-tensor Fisher tables (run_mbv2_fisher_term_ablation.py's base) ---
    import timm
    model = timm.create_model(args.timm_name, pretrained=True).eval()
    fold_batchnorms(model)
    layers = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
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
    acc = {n: {"g2w": 0.0, **{s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")}}
           for n in module_of}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        F.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                acc[n]["g2w"] += float((module_of[n].weight.grad.double() ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    s = acc[n][side]
                    s["g2"] += float((t.grad.double() ** 2).sum())
                    s["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        s["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
    T = {}
    for n in module_of:
        W = inits[by_name[n].input[1]]
        a = acc[n]
        T[n] = {"w": {"fp32": 0.0, **{f: a["g2w"] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                                       - W.astype(np.float64)) ** 2).sum())
                                      for f in FORMATS[:-1]}},
                **{s: {"fp32": 0.0, **{f: a[s]["g2"] / a[s]["count"] * a[s]["d2"][f] for f in FORMATS[:-1]}}
                   for s in ("x", "y")}}

    def fisher(n, w, a):
        return T[n]["w"][w] + T[n]["x"][a] + T[n]["y"][a]

    # --- candidates at one budget ---
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    names = list(module_of)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    cands = []
    for label, path, variant in EARLIER:
        row = next(r for r in json.loads((RUNS / path).read_text())["rows"]
                   if r["variant"] == variant and abs(r["budget"] - args.budget) < 1e-9)
        cands.append((label, {n: {"w": v["w"], "a": v["a"]} for n, v in row["formats"].items()}))
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
    sq = next(c for c in sq if c["criterion"] == "sqnr" and abs(c["budget"] - args.budget) < 1e-9)
    cands.append(("sqnr", {n: {"w": v["w"], "a": v["a"]} for n, v in sq["formats"].items()}))
    rng = np.random.default_rng(0)
    for i in range(args.perturbed):
        mult = {n: float(rng.lognormal(0.0, args.sigma)) for n in names}
        cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a: mult[n] * fisher(n, w, a),
                                     weight_elems, flops_proxy, args.budget * int8_bytes, args.budget * int8_bops,
                                     time_limit=60)
        assert status in (0, 1)
        cands.append((f"fisher perturbed #{i + 1}", {n: {"w": v["w"], "a": v["a"]} for n, v in cfg.items()}))
    unique, seen = [], set()
    for label, cfg in cands:
        key = tuple(sorted((n, v["w"], v["a"]) for n, v in cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append((label, cfg))
    log(f"{len(unique)} distinct candidate configs at f={args.budget} ({len(cands) - len(unique)} duplicates dropped)")

    # --- actual, additive and fisher scores ---
    def loss(fmts):
        z = simulate_in_onnxruntime(graph, all_nodes, fmts, ranges, ex, None)
        assert np.isfinite(z).all()
        return F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey), reduction="none").numpy()

    base = loss({})
    single = {}
    rows = []
    for label, cfg in unique:
        q = {n: v for n, v in cfg.items() if n in module_of and ((v["w"] not in (None, "fp32")) or v["a"] != "fp32")}
        for n, v in q.items():
            if (n, v["w"], v["a"]) not in single:
                single[(n, v["w"], v["a"])] = float((loss({n: v}) - base).mean())
        size, bops = config_cost(cfg, weight_elems, flops_proxy)
        rows.append({"label": label, "size": size / int8_bytes, "bops": bops / int8_bops,
                     "actual": float((loss(q) - base).mean()),
                     "additive": sum(single[(n, v["w"], v["a"])] for n, v in q.items()),
                     "fisher": sum(fisher(n, v["w"], v["a"]) for n, v in q.items())})
        log(f"{label:<22} actual {rows[-1]['actual']:.3f}  additive {rows[-1]['additive']:.3f}  "
            f"({len(single)} single-node measurements so far)")
    actual = np.array([r["actual"] for r in rows])
    stats = {}
    for key in ("additive", "fisher"):
        p = np.array([r[key] for r in rows])
        pick = int(np.argmin(p))
        stats[key] = {"spearman": float(spearmanr(p, actual)[0]), "pick": rows[pick]["label"],
                      "pick_actual": float(actual[pick]), "regret": float(actual[pick] - actual.min())}
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "budget": args.budget, "num_eval": len(ey), "rows": rows,
         "stats": stats, "best_actual": rows[int(np.argmin(actual))]["label"]}, indent=2) + "\n")

    print(f"\ncandidates at f={args.budget} (all within budget), held-out loss increase on {len(ey)} images:")
    print(f"{'config':<22} {'size':>5} {'BOPs':>5} {'actual':>8} {'additive':>9} {'fisher':>9}  ranks (actual/additive/fisher)")
    ra = {key: {i: r for r, i in enumerate(np.argsort([x[key] for x in rows]), 1)} for key in ("actual", "additive", "fisher")}
    for i, r in sorted(enumerate(rows), key=lambda t: t[1]["actual"]):
        print(f"{r['label']:<22} {r['size']:5.2f} {r['bops']:5.2f} {r['actual']:8.3f} {r['additive']:9.3f} "
              f"{r['fisher']:9.4g}  {ra['actual'][i]:2d} / {ra['additive'][i]:2d} / {ra['fisher'][i]:2d}")
    print(f"best actual: {rows[int(np.argmin(actual))]['label']} ({actual.min():.3f})")
    for key, s in stats.items():
        print(f"{key:<9} Spearman with actual {s['spearman']:+.2f}; picks '{s['pick']}' (actual {s['pick_actual']:.3f}, "
              f"regret {s['regret']:+.3f})")


if __name__ == "__main__":
    main()
