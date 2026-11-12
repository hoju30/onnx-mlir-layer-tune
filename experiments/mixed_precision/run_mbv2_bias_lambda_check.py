#!/usr/bin/env python3
"""MobileNetV2: does weighting the per-channel mean (bias) part of the activation error make the Fisher predict better?

Per activation tensor t in {input X, output Y} of a node at format f (calibration images):
  delta = Q_f(t) - t,  mu_c = per-channel mean of delta,
  E_bias = sum_c n_c mu_c^2,  E_res = sum ||delta - mu_c||^2   (E_res + E_bias = sum ||delta||^2)
  C_A(lambda) = sum_t mean(g_t^2) * (E_res + lambda * E_bias)      (lambda = 1: the current per-tensor Fisher)
The weight term mean(g_W^2) * ||Q(W) - W||^2 is unchanged. No new held-out runs; the ground truth is saved:
  A  activation-only, one node at a time, int6 and int4 (runs/mbv2_act_lowbit_check_20261004, PyTorch, 200 images)
  B  w/a = int4/int8, int8/int4, int6/int6, one node at a time (runs/mbv2_predictor_check_20261003, ORT, 300 images)
lambda is chosen on A (best Spearman) and read off on B; the whole lambda curve is printed for both. The SQNR table
(runs/mobilenetv2_timm_ra_criteria_milp) is the forward-measured reference.
"""
import argparse
import json
import sys
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
                                         supported_nodes)
from run_mbv2_act_lowbit_check import chan_sum  # noqa: E402
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402

RUNS = HERE / "runs"
INT = ("int4", "int5", "int6", "int7", "int8")
LAMBDAS = (0, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000)
SHOW_TYPES = ("stem", "ds_dw", "pw", "dw", "pwl", "head")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    cx, cy = preprocess_by_source(args.images_dir, {"images": calib}, meta), np.array([e["label"] for e in calib])
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

    # --- calibration: squared gradients and the error split, per node, tensor and int format ---
    cap = {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2 = {n: {"w": 0.0, "x": 0.0, "y": 0.0} for n in names}
    cnt = {n: {"x": 0, "y": 0} for n in names}
    nc = {n: {"x": 0, "y": 0} for n in names}
    s1 = {n: {s: dict.fromkeys(INT, 0.0) for s in ("x", "y")} for n in names}
    s2 = {n: {s: dict.fromkeys(INT, 0.0) for s in ("x", "y")} for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2[n]["w"] += float((grads[3 * i].double() ** 2).sum())
                for j, s in enumerate(("x", "y")):
                    t = cap[n][j]
                    g2[n][s] += float((grads[3 * i + 1 + j].double() ** 2).sum())
                    cnt[n][s] += t.numel()
                    nc[n][s] += t.numel() // t.shape[1]
                    for f in INT:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        s1[n][s][f] = s1[n][s][f] + chan_sum(d)
                        s2[n][s][f] += float((d ** 2).sum())
        cap.clear()
    S = {n: {s: g2[n][s] / cnt[n][s] for s in ("x", "y")} for n in names}
    E_bias = {n: {s: {f: float((s1[n][s][f] ** 2).sum()) / nc[n][s] for f in INT} for s in ("x", "y")} for n in names}
    E_res = {n: {s: {f: s2[n][s][f] - E_bias[n][s][f] for f in INT} for s in ("x", "y")} for n in names}
    C_W = {}
    for n in names:
        W = inits[by_name[n].input[1]]
        C_W[n] = {"fp32": 0.0, **{f: g2[n]["w"] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                                    - W.astype(np.float64)) ** 2).sum()) for f in INT}}

    def C_A(n, a, lam):
        return 0.0 if a == "fp32" else sum(S[n][s] * (E_res[n][s][a] + lam * E_bias[n][s][a]) for s in ("x", "y"))

    # --- ground truth sets ---
    lowbit = [r for r in json.loads((RUNS / "mbv2_act_lowbit_check_20261004/results.json").read_text())["rows"]
              if r["mode"] == "plain"]
    pc = json.loads((RUNS / "mbv2_predictor_check_20261003/results.json").read_text())
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/sqnr_table.json").read_text())["table"]
    sets = {
        "A": [{"node": r["node"], "w": "fp32", "a": r["bits"], "dloss": r["dloss"], "dloss_se": r["dloss_se"],
               "saved_fisher": r["fisher"], "group": r["bits"]} for r in lowbit],
        "B": [{"node": r["node"], "w": r["w"], "a": r["a"], "dloss": r["dloss"], "dloss_se": r["dloss_se"],
               "saved_fisher": r["fisher"], "group": f"{r['w']}/{r['a']}"} for r in pc["rows"]],
    }

    def predict(rows, lam):
        return np.array([C_W[r["node"]][r["w"]] + C_A(r["node"], r["a"], lam) for r in rows])

    for key, rows in sets.items():
        p1, saved = predict(rows, 1), np.array([r["saved_fisher"] for r in rows])
        print(f"set {key}: lambda = 1 vs the saved Fisher: max relative difference {np.max(np.abs(p1 - saved) / saved):.1e}")
    share = {k: np.median([sum(S[r['node']][s] * E_bias[r['node']][s][r['a']] for s in ('x', 'y'))
                           / max(C_A(r['node'], r['a'], 1), 1e-300) for r in rows if r['a'] != 'fp32'])
             for k, rows in sets.items()}
    print("median share of the bias part in C_A at lambda = 1: " + ", ".join(f"set {k} {v:.1%}" for k, v in share.items()))

    def stats(rows, pred):
        y = np.array([r["dloss"] for r in rows])
        sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in rows])
        lr = np.log10(pred[sig] / y[sig])
        cen = lr - np.median(lr)
        out = {"spearman": float(spearmanr(pred, y)[0]), "spearman_sig": float(spearmanr(pred[sig], y[sig])[0]),
               "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
               "groups": {g: float(spearmanr(pred[ix], y[ix])[0]) for g in dict.fromkeys(r["group"] for r in rows)
                          for ix in [np.array([r["group"] == g for r in rows])]}}
        sig_rows = [r for r, s in zip(rows, sig) if s]
        out["type_bias"] = {t: float(np.median(cen[[kind[r["node"]] == t for r in sig_rows]]))
                            for t in SHOW_TYPES if any(kind[r["node"]] == t for r in sig_rows)}
        return out

    res = {k: {lam: stats(rows, predict(rows, lam)) for lam in LAMBDAS} for k, rows in sets.items()}
    sqnr = {k: stats(rows, np.array([sq[r["node"]][f"{r['w']}|{r['a']}"] for r in rows])) for k, rows in sets.items()}
    best_a = max(LAMBDAS, key=lambda lam: res["A"][lam]["spearman"])

    groups = {k: list(res[k][1]["groups"]) for k in sets}
    print(f"\nSpearman with the single-node dloss (A: {len(sets['A'])} pairs, B: {len(sets['B'])} pairs); "
          f"'sig' = pairs with dloss > 2 SE; IQR of log10(pred/dloss)")
    head = (f"{'lambda':>7} | {'A all':>6} {'A sig':>6} " + " ".join(f"{g:>6}" for g in groups["A"]) + f" {'IQR':>5} | "
            f"{'B all':>6} {'B sig':>6} " + " ".join(f"{g:>10}" for g in groups["B"]) + f" {'IQR':>5}")
    print(head)

    def line(label, a, b):
        return (f"{label:>7} | {a['spearman']:6.3f} {a['spearman_sig']:6.3f} " + " ".join(f"{a['groups'][g]:6.3f}" for g in groups["A"])
                + f" {a['iqr']:5.2f} | {b['spearman']:6.3f} {b['spearman_sig']:6.3f} "
                + " ".join(f"{b['groups'][g]:10.3f}" for g in groups["B"]) + f" {b['iqr']:5.2f}")
    for lam in LAMBDAS:
        print(line(f"{lam:g}" + ("*" if lam == best_a else ""), res["A"][lam], res["B"][lam]))
    print(line("SQNR", sqnr["A"], sqnr["B"]))
    print(f"* best on A; on B it gives Spearman {res['B'][best_a]['spearman']:.3f} "
          f"(lambda = 1: {res['B'][1]['spearman']:.3f}, SQNR {sqnr['B']['spearman']:.3f})")

    print("\nper-layer-type bias, median log10(pred/dloss) minus the median over all nodes (+ over-, - under-estimated)")
    for k in sets:
        for label, s in (("lambda=1", res[k][1]), (f"lambda={best_a:g}", res[k][best_a]), ("SQNR", sqnr[k])):
            print(f"  {k} {label:<11} " + "  ".join(f"{t} {b:+.2f}" for t, b in s["type_bias"].items()))

    (args.out_dir / "results.json").write_text(json.dumps(
        {"lambdas": LAMBDAS, "best_lambda_on_A": best_a, "bias_share_at_lambda_1": share,
         "sets": {"A": "runs/mbv2_act_lowbit_check_20261004 (plain)", "B": "runs/mbv2_predictor_check_20261003"},
         "results": {k: {str(lam): v for lam, v in r.items()} for k, r in res.items()}, "sqnr": sqnr}, indent=2) + "\n")


if __name__ == "__main__":
    main()
