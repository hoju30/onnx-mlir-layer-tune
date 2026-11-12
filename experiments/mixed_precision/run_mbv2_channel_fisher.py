#!/usr/bin/env python3
"""MobileNetV2: per-channel activation sensitivity, and the per-channel mean (bias) of the error scored on its own.

Per activation tensor t in {input X, output Y} of a node at format f, over the calibration images (delta = Q_f(t) - t,
c a channel, mu_c the per-channel mean of delta, g = dCE/dt with the image's label):
  A  per-tensor Fisher       mean(g^2) * sum delta^2                                   (the current criterion)
  B  channel Fisher          sum_c mean_c(g^2) * sum_{j in c} delta_j^2
  C  channel, bias-aware     sum_c [ mean_c(g^2) * sum_{j in c} (delta_j - mu_c)^2
                                     + mu_c^2 * sum_images (sum_{j in c} g_j)^2 ]
C's second term is the second-order (g^T delta)^2 of a constant per-channel offset, which adds up coherently over the
channel's positions; with gradients independent across positions it equals mean_c(g^2) * n_c * mu_c^2 and C = B.
The weight term mean(g_W^2) * ||Q(W) - W||^2 is the same in all three.
1. Per-node prediction against saved single-node ground truth:
     act-only  w=fp32, a=int6/int4 (runs/mbv2_act_lowbit_check_20261004, PyTorch, 200 held-out images)
     W+A       w/a = int4/int8, int8/int4, int6/int6 (runs/mbv2_predictor_check_20261003, ORT, 300 held-out images)
   with the SQNR table and the global-lambda bias weight (run_mbv2_bias_lambda_check.py, lambda = 15) for reference.
2. MILP picks at budgets relative to all-int8, ORT QDQ simulation on the 1000 held-out images, with the saved SQNR pick.
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

from analyze_fisher_vs_sqnr import layer_type  # noqa: E402
from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from mp_milp import milp_select  # noqa: E402
from run_mbv2_act_lowbit_check import chan_sum  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant, fold_batchnorms  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

RUNS = HERE / "runs"
VARIANTS = ("A", "B", "C")
LABEL = {"A": "A per-tensor", "B": "B channel", "C": "C channel+bias", "lam15": "ref lambda=15", "sqnr": "ref SQNR"}
SHOW_TYPES = ("stem", "ds_dw", "pw", "dw", "pwl", "head")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
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
    kind = {n: layer_type(i, len(names)) for i, n in enumerate(names)}
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    AF = FORMATS[:-1]

    # --- calibration: per-channel gradient statistics and per-channel error sums ---
    cap = {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2w = dict.fromkeys(names, 0.0)
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += float((grads[3 * i].double() ** 2).sum())
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                    a["g2"] = a["g2"] + chan_sum(g ** 2)        # per channel: sum of g^2
                    a["gc"] = a["gc"] + chan_sum(g) ** 2        # per channel: (sum of g over the image's positions)^2
                    a["n"] += t.numel() // t.shape[1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + chan_sum(d)
                        a["s2"][f] = a["s2"][f] + chan_sum(d ** 2)
        cap.clear()
    CA = {v: {n: {"fp32": 0.0} for n in names} for v in VARIANTS + ("lam15",)}
    bias_share = {}
    for n in names:
        for f in AF:
            parts = {v: 0.0 for v in CA}
            bias_c = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                S_c = a["g2"] / a["n"]                          # mean_c(g^2)
                S = float(a["g2"].sum()) / (a["n"] * len(a["g2"]))
                mu = a["s1"][f] / a["n"]
                e_bias = a["n"] * mu ** 2
                e_res = a["s2"][f] - e_bias
                parts["A"] += S * float(a["s2"][f].sum())
                parts["B"] += float((S_c * a["s2"][f]).sum())
                parts["C"] += float((S_c * e_res).sum() + (a["gc"] * mu ** 2).sum())
                parts["lam15"] += S * float(e_res.sum() + 15 * e_bias.sum())
                bias_c += float((a["gc"] * mu ** 2).sum())
            for v in CA:
                CA[v][n][f] = parts[v]
            bias_share[(n, f)] = bias_c / parts["C"] if parts["C"] > 0 else 0.0
    CW = {}
    for n in names:
        W = inits[by_name[n].input[1]]
        CW[n] = {"fp32": 0.0, **{f: g2w[n] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                                - W.astype(np.float64)) ** 2).sum()) for f in AF}}
    log("calibration statistics ready")

    def cost(v, n, w, a):
        return CW[n][w] + CA[v][n][a]

    # --- 1. per-node prediction ---
    lowbit = [r for r in json.loads((RUNS / "mbv2_act_lowbit_check_20261004/results.json").read_text())["rows"]
              if r["mode"] == "plain"]
    pc = json.loads((RUNS / "mbv2_predictor_check_20261003/results.json").read_text())
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/sqnr_table.json").read_text())["table"]
    sets = {"act-only": [{"node": r["node"], "w": "fp32", "a": r["bits"], "dloss": r["dloss"], "se": r["dloss_se"],
                          "saved": r["fisher"], "group": f"a={r['bits']}"} for r in lowbit],
            "W+A": [{"node": r["node"], "w": r["w"], "a": r["a"], "dloss": r["dloss"], "se": r["dloss_se"],
                     "saved": r["fisher"], "group": f"{r['w']}/{r['a']}"} for r in pc["rows"]]}
    preds = {k: {**{v: np.array([cost(v, r["node"], r["w"], r["a"]) for r in rows]) for v in CA},
                 "sqnr": np.array([sq[r["node"]][f"{r['w']}|{r['a']}"] for r in rows])} for k, rows in sets.items()}
    for k, rows in sets.items():
        saved = np.array([r["saved"] for r in rows])
        log(f"{k}: variant A vs the saved Fisher, max relative difference {np.max(np.abs(preds[k]['A'] - saved) / saved):.1e}")

    def stats(rows, p):
        y = np.array([r["dloss"] for r in rows])
        sig = np.array([r["dloss"] > 2 * r["se"] for r in rows])
        lr = np.log10(p[sig] / y[sig])
        cen = lr - np.median(lr)
        srows = [r for r, s in zip(rows, sig) if s]
        return {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                "groups": {g: float(spearmanr(p[ix], y[ix])[0]) for g in dict.fromkeys(r["group"] for r in rows)
                           for ix in [np.array([r["group"] == g for r in rows])]},
                "type_bias": {t: float(np.median(cen[[kind[r["node"]] == t for r in srows]]))
                              for t in SHOW_TYPES if any(kind[r["node"]] == t for r in srows)}}

    pred_stats = {k: {v: stats(rows, preds[k][v]) for v in preds[k]} for k, rows in sets.items()}
    dw_share = {f: float(np.median([bias_share[(n, f)] for n in names if kind[n] == "dw"])) for f in ("int4", "int6", "int8")}
    print("\nmedian share of C's bias term in the dw activation cost: "
          + ", ".join(f"{f} {s:.1%}" for f, s in dw_share.items()))
    for k in sets:
        groups = list(pred_stats[k]["A"]["groups"])
        print(f"\n1. {k} ({len(sets[k])} single-node pairs): Spearman with dloss; 'sig' = pairs with dloss > 2 SE; "
              f"IQR of log10(pred/dloss); per-type bias (+ over-, - under-estimated)")
        print(f"   {'variant':<16}{'all':>7}{'sig':>7}" + "".join(f"{g:>11}" for g in groups) + f"{'IQR':>6}   type bias")
        for v, s in pred_stats[k].items():
            print(f"   {LABEL[v]:<16}{s['spearman']:7.3f}{s['spearman_sig']:7.3f}"
                  + "".join(f"{s['groups'][g]:11.3f}" for g in groups) + f"{s['iqr']:6.2f}   "
                  + " ".join(f"{t} {b:+.2f}" for t, b in s["type_bias"].items()))

    # --- 2. MILP + held-out evaluation ---
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    sqc = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
    res, rows = {}, []
    for f in budgets:
        cfgs = {}
        for v in VARIANTS:
            cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a, v=v: cost(v, n, w, a),
                                         weight_elems, flops_proxy, f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1), (v, f, status)
            cfgs[v] = ({n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}, status)
        s = next(c for c in sqc if c["criterion"] == "sqnr" and abs(c["budget"] - f) < 1e-9)
        cfgs["sqnr"] = ({n: {"w": s["formats"][n]["w"], "a": s["formats"][n]["a"]} for n in names}, 0)
        for v, (cfg, status) in cfgs.items():
            size, bops = config_cost(cfg, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            correct = ((z.argmax(1) == ey) & finite).astype(float)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(ey),
                                   reduction="none").numpy()
            res[(f, v)] = (correct, loss)
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "formats": cfg})
            log(f"f={f:.2f} {LABEL[v]:<16} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in rows:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "A")]
        d1, dl = (c - cb) * 100, l - lb
        base = next(x for x in rows if x["budget"] == r["budget"] and x["variant"] == "A")["formats"]
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                 nodes_differing_from_A=sum(r["formats"][n] != base[n] for n in names))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation (MILP part)", "model": args.timm_name, "budgets": budgets,
         "num_eval": len(ey), "dw_bias_term_share": dw_share, "prediction": pred_stats, "rows": rows},
        indent=2) + "\n")

    print(f"\n2. MILP picks, held-out ({len(ey)} images); differences paired per image against A")
    print(f"   {'f':>5} {'variant':<16}{'top1':>7}{'loss':>8}{'d top1 vs A':>15}{'d loss vs A':>18}{'nodes differ':>14}")
    for r in rows:
        d = "" if r["variant"] == "A" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "A" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"   {r['budget']:5.2f} {LABEL[r['variant']] + ('' if r['milp_optimal'] else '*'):<16}{r['top1']:7.1%}"
              f"{r['loss']:8.4f}{d:>15}{dl:>18}{r['nodes_differing_from_A']:>9d}/{len(names)}")
    print("   mean over budgets:")
    for v in VARIANTS[1:] + ("sqnr",):
        sel = [r for r in rows if r["variant"] == v]
        print(f"     {LABEL[v]:<16} d top1 {np.mean([r['dtop1_pp'] for r in sel]):+.2f} pp   d loss "
              f"{np.mean([r['dloss'] for r in sel]):+.4f} ± {np.sqrt(sum(r['dloss_se'] ** 2 for r in sel)) / len(sel):.4f}")


if __name__ == "__main__":
    main()
