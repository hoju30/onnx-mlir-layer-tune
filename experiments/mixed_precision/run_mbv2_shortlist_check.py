#!/usr/bin/env python3
"""MobileNetV2: per-tensor Fisher shortlist + a joint check on the calibration images.

Per budget, the shortlist is the per-tensor Fisher MILP pick (HTQ Eq. 5 with S = mean(g^2), as in
run_mbv2_fisher_term_ablation.py's base) plus --perturbed near-optimal competitors: the same MILP with every node's
cost multiplied by a log-normal factor. Each shortlisted config is run whole (ORT QDQ simulation, the pass's
arithmetic) on the same calibration images the criterion used, and the pick is the one with the least damage:
  kl     mean KL(FP32 softmax || quantized softmax)   (label-free)
  ce     mean cross-entropy with the labels
Compared on the held-out images with the Fisher pick, the best shortlisted config on the held-out images (oracle:
the most a shortlist can give) and the saved SQNR MILP config.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--perturbed", type=int, default=8, help="near-optimal competitors from perturbed Fisher costs")
    ap.add_argument("--sigma", type=float, default=0.5, help="log-normal sigma of the per-node cost perturbation")
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

    # --- per-tensor Fisher tables (run_mbv2_fisher_term_ablation.py's base) ---
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

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    names = list(module_of)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    log("Fisher tables ready")

    # --- whole-config damage ---
    def logits(fmts, x):
        z = simulate_in_onnxruntime(graph, all_nodes, fmts, ranges, x, None)
        assert np.isfinite(z).all()
        return torch.from_numpy(z).double()

    ref_cal = F.log_softmax(logits({}, cx), 1)

    def calib_damage(fmts):
        lq = F.log_softmax(logits(fmts, cx), 1)
        kl = float((ref_cal.exp() * (ref_cal - lq)).sum(1).mean())
        ce = float(F.nll_loss(lq, torch.from_numpy(cy)))
        return kl, ce

    def held_out(fmts):
        z = logits(fmts, ex)
        return ((z.argmax(1).numpy() == ey).astype(float),
                F.cross_entropy(z, torch.from_numpy(ey), reduction="none").numpy())

    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
    rows, summary = [], []
    for f in budgets:
        rng = np.random.default_rng(int(round(f * 100)))
        cands = []
        for i in range(args.perturbed + 1):
            mult = {n: (1.0 if i == 0 else float(rng.lognormal(0.0, args.sigma))) for n in names}
            cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a: mult[n] * fisher(n, w, a),
                                         weight_elems, flops_proxy, f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
            if all(cfg != c["formats"] for c in cands):
                cands.append({"label": "fisher" if i == 0 else f"perturbed #{i}", "formats": cfg})
        s = next(c for c in sq if c["criterion"] == "sqnr" and abs(c["budget"] - f) < 1e-9)
        sqnr = {"label": "sqnr", "formats": {n: {"w": s["formats"][n]["w"], "a": s["formats"][n]["a"]} for n in names}}
        for c in cands + [sqnr]:
            size, bops = config_cost(c["formats"], weight_elems, flops_proxy)
            assert size <= f * int8_bytes * (1 + 1e-6) and bops <= f * int8_bops * (1 + 1e-6), c["label"]
            c["kl"], c["ce"] = calib_damage(c["formats"])
            c["fisher"] = sum(fisher(n, v["w"], v["a"]) for n, v in c["formats"].items())
            c["correct"], c["loss"] = held_out(c["formats"])
            rows.append({"budget": f, "label": c["label"], "size": size / int8_bytes, "bops": bops / int8_bops,
                         "fisher": c["fisher"], "calib_kl": c["kl"], "calib_ce": c["ce"],
                         "top1": float(c["correct"].mean()), "loss": float(c["loss"].mean()), "formats": c["formats"]})
            log(f"f={f:.2f} {c['label']:<13} fisher {c['fisher']:.4g}  calib KL {c['kl']:.4f} CE {c['ce']:.4f}  "
                f"held-out top1 {c['correct'].mean():6.1%} loss {c['loss'].mean():.4f}")
        base = cands[0]
        picks = {"fisher": base, "shortlist_kl": min(cands, key=lambda c: c["kl"]),
                 "shortlist_ce": min(cands, key=lambda c: c["ce"]),
                 "oracle": min(cands, key=lambda c: c["loss"].mean()), "sqnr": sqnr}
        held_loss = np.array([c["loss"].mean() for c in cands])
        for key, c in picks.items():
            d1, dl = (c["correct"] - base["correct"]) * 100, c["loss"] - base["loss"]
            summary.append({"budget": f, "method": key, "pick": c["label"], "top1": float(c["correct"].mean()),
                            "loss": float(c["loss"].mean()), "dtop1_pp": float(d1.mean()),
                            "dtop1_se": float(d1.std(ddof=1) / np.sqrt(len(d1))), "dloss": float(dl.mean()),
                            "dloss_se": float(dl.std(ddof=1) / np.sqrt(len(dl))), "shortlist": len(cands),
                            "spearman_kl_vs_heldout": float(spearmanr([c["kl"] for c in cands], held_loss)[0]),
                            "spearman_ce_vs_heldout": float(spearmanr([c["ce"] for c in cands], held_loss)[0])})
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets, "perturbed": args.perturbed,
         "sigma": args.sigma, "num_calibration": len(cy), "num_eval": len(ey), "summary": summary,
         "candidates": rows}, indent=2) + "\n")

    print(f"\n{'f':>5} {'method':<13} {'pick':<13} {'top1':>7} {'loss':>7} {'d top1 vs fisher':>17} {'d loss':>17}")
    for r in summary:
        d = "" if r["method"] == "fisher" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["method"] == "fisher" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['method']:<13} {r['pick']:<13} {r['top1']:7.1%} {r['loss']:7.4f} {d:>17} {dl:>17}")
    print("\nshortlist size and rank agreement of the calibration check with the held-out loss (Spearman):")
    for r in summary:
        if r["method"] == "fisher":
            print(f"  f={r['budget']:.2f}: {r['shortlist']} configs, KL {r['spearman_kl_vs_heldout']:+.2f}, "
                  f"CE {r['spearman_ce_vs_heldout']:+.2f}")


if __name__ == "__main__":
    main()
