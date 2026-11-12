#!/usr/bin/env python3
"""MobileNetV2: the channel bias-aware term of run_mbv2_channel_fisher.py's variant C, extended to the weights.

Weight rounding dW = Q_f(W) - W shifts output channel o by a constant on average,
  b_o = mean over the calibration images and positions of (conv(X, dW))_o
(DFQ's biased weight error; strongest where an output sees few weights, e.g. depthwise 3x3). With g = dCE/dY:
  coherent cost of the shift   b_o^2 * GC_o,  GC_o = sum_images (sum_positions g_o)^2
  the same under the per-element (diagonal) model   b_o^2 * G2_o,  G2_o = sum_images sum_positions g_o^2
Activation term: C's (per-channel residual + coherent per-channel mean). Weight term:
  A, C   mean(g_W^2) * ||dW||^2                                              (per-tensor, unchanged)
  C+Wb   mean(g_W^2) * ||dW||^2 + sum_o b_o^2 GC_o                           (shift added)
  C+Wx   mean(g_W^2) * ||dW||^2 + sum_o b_o^2 (GC_o - G2_o), floored at 0   (only the coherence excess, as C does
                                                                             for activations)
1. Per-node prediction against the single-node ground truth of runs/mbv2_predictor_check_20261003 (w/a = int4/int8,
   int8/int4, int6/int6; ORT, 300 held-out images), with the SQNR table for reference.
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
VARIANTS = ("A", "C", "C+Wb", "C+Wx")
SHOW_TYPES = ("stem", "ds_dw", "ds_pw", "pw", "dw", "pwl", "head", "fc")


def node_op(m, x, w):
    return m._conv_forward(x, w, None) if isinstance(m, nn.Conv2d) else F.linear(x, w)


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
    dW = {n: {f: torch.from_numpy(np_fake_quant_weight(inits[by_name[n].input[1]], f, None)
                                  - inits[by_name[n].input[1]].astype(np.float32)) for f in AF} for n in names}

    # --- calibration statistics ---
    cap = {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2w = dict.fromkeys(names, 0.0)
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    wsh = {n: dict.fromkeys(AF, 0.0) for n in names}   # per output channel: sum of conv(X, dW) over images, positions
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += float((grads[3 * i].double() ** 2).sum())
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                    a["g2"] = a["g2"] + chan_sum(g ** 2)
                    a["gc"] = a["gc"] + chan_sum(g) ** 2
                    a["n"] += t.numel() // t.shape[1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + chan_sum(d)
                        a["s2"][f] = a["s2"][f] + chan_sum(d ** 2)
                x = cap[n][0].detach()
                for f in AF:
                    wsh[n][f] = wsh[n][f] + chan_sum(node_op(module_of[n], x, dW[n][f]).double())
        cap.clear()

    CA, CW, floored = {"A": {}, "C": {}}, {v: {} for v in ("A", "Wb", "Wx")}, 0
    bias_share = {}
    for n in names:
        CA["A"][n], CA["C"][n] = {"fp32": 0.0}, {"fp32": 0.0}
        W = inits[by_name[n].input[1]]
        for v in CW:
            CW[v][n] = {"fp32": 0.0}
        y = st[n]["y"]
        for f in AF:
            a_cost = c_cost = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                S_c = a["g2"] / a["n"]
                S = float(a["g2"].sum()) / (a["n"] * len(a["g2"]))
                mu = a["s1"][f] / a["n"]
                a_cost += S * float(a["s2"][f].sum())
                c_cost += float((S_c * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            CA["A"][n][f], CA["C"][n][f] = a_cost, c_cost
            base = g2w[n] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                             - W.astype(np.float64)) ** 2).sum())
            b = wsh[n][f] / y["n"]
            CW["A"][n][f] = base
            CW["Wb"][n][f] = base + float((b ** 2 * y["gc"]).sum())
            excess = base + float((b ** 2 * (y["gc"] - y["g2"])).sum())
            floored += excess < 0
            CW["Wx"][n][f] = max(excess, 0.0)
            bias_share[(n, f)] = float((b ** 2 * y["gc"]).sum()) / CW["Wb"][n][f] if CW["Wb"][n][f] > 0 else 0.0
    log(f"calibration statistics ready ({floored} weight costs of C+Wx floored at 0)")

    def cost(v, n, w, a):
        wv = {"A": "A", "C": "A", "C+Wb": "Wb", "C+Wx": "Wx"}[v]
        return CW[wv][n][w] + CA["A" if v == "A" else "C"][n][a]

    print("median share of the weight shift term in C+Wb's weight cost at int4, by type: " + ", ".join(
        f"{t} {np.median([bias_share[(n, 'int4')] for n in names if kind[n] == t]):.1%}" for t in SHOW_TYPES))

    # --- 1. per-node prediction ---
    pc = json.loads((RUNS / "mbv2_predictor_check_20261003/results.json").read_text())
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/sqnr_table.json").read_text())["table"]
    rows = pc["rows"]
    preds = {**{v: np.array([cost(v, r["node"], r["w"], r["a"]) for r in rows]) for v in VARIANTS},
             "SQNR": np.array([sq[r["node"]][f"{r['w']}|{r['a']}"] for r in rows])}
    saved = np.array([r["fisher"] for r in rows])
    log(f"A vs the saved Fisher: max relative difference {np.max(np.abs(preds['A'] - saved) / saved):.1e}")
    y = np.array([r["dloss"] for r in rows])
    sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in rows])
    groups = [f"{w}/{a}" for w, a in pc["configs"]]
    pred_stats = {}
    print(f"\n1. W+A single-node truth ({len(rows)} pairs, {sig.sum()} with dloss > 2 SE): Spearman; IQR of "
          f"log10(pred/dloss); per-type bias (+ over-, - under-estimated)")
    print(f"   {'variant':<7}{'all':>7}{'sig':>7}" + "".join(f"{g:>11}" for g in groups) + f"{'IQR':>6}   type bias")
    for v, p in preds.items():
        lr = np.log10(p[sig] / y[sig])
        cen = lr - np.median(lr)
        srows = [r for r, s in zip(rows, sig) if s]
        st_ = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
               "groups": {g: float(spearmanr(p[ix], y[ix])[0]) for g in groups
                          for ix in [np.array([f"{r['w']}/{r['a']}" == g for r in rows])]},
               "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
               "type_bias": {t: float(np.median(cen[[kind[r["node"]] == t for r in srows]]))
                             for t in SHOW_TYPES if any(kind[r["node"]] == t for r in srows)}}
        pred_stats[v] = st_
        print(f"   {v:<7}{st_['spearman']:7.3f}{st_['spearman_sig']:7.3f}" + "".join(f"{st_['groups'][g]:11.3f}" for g in groups)
              + f"{st_['iqr']:6.2f}   " + " ".join(f"{t} {b:+.2f}" for t, b in st_["type_bias"].items()))
    wi = np.array([(r["w"], r["a"]) == ("int4", "int8") for r in rows]) & sig
    print("   weight-dominated (int4/int8, > 2 SE) per-type bias: " + " | ".join(
        f"{v}: " + " ".join(f"{t} {np.median((lambda lr: lr - np.median(lr))(np.log10(preds[v][wi] / y[wi]))[[kind[r['node']] == t for r in np.array(rows)[wi]]]):+.2f}"
                            for t in ("pw", "dw", "pwl")) for v in preds))

    # --- 2. MILP + held-out evaluation ---
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    sqc = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
    res, out = {}, []
    for f in budgets:
        cfgs = {}
        for v in VARIANTS:
            cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a, v=v: cost(v, n, w, a),
                                         weight_elems, flops_proxy, f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1), (v, f, status)
            cfgs[v] = ({n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}, status)
        s = next(c for c in sqc if c["criterion"] == "sqnr" and abs(c["budget"] - f) < 1e-9)
        cfgs["SQNR"] = ({n: {"w": s["formats"][n]["w"], "a": s["formats"][n]["a"]} for n in names}, 0)
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
            log(f"f={f:.2f} {v:<6} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in out:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "C")]
        d1, dl = (c - cb) * 100, l - lb
        base = next(x for x in out if x["budget"] == r["budget"] and x["variant"] == "C")["formats"]
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                 nodes_differing_from_C=sum(r["formats"][n] != base[n] for n in names))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation (MILP part)", "model": args.timm_name, "budgets": budgets,
         "num_eval": len(ey), "prediction": pred_stats, "rows": out}, indent=2) + "\n")
    print(f"\n2. MILP picks, held-out ({len(ey)} images); differences paired per image against C")
    print(f"   {'f':>5} {'variant':<7}{'top1':>7}{'loss':>8}{'d top1 vs C':>15}{'d loss vs C':>18}{'nodes differ':>14}")
    for r in out:
        d = "" if r["variant"] == "C" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "C" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"   {r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<7}{r['top1']:7.1%}"
              f"{r['loss']:8.4f}{d:>15}{dl:>18}{r['nodes_differing_from_C']:>9d}/{len(names)}")
    print("   mean over budgets (top-1, loss; d loss vs C):")
    for v in VARIANTS + ("SQNR",):
        sel = [r for r in out if r["variant"] == v]
        print(f"     {v:<7} top1 {np.mean([r['top1'] for r in sel]):6.1%}  loss {np.mean([r['loss'] for r in sel]):.4f}  "
              f"d loss {np.mean([r['dloss'] for r in sel]):+.4f} ± {np.sqrt(sum(r['dloss_se'] ** 2 for r in sel)) / len(sel):.4f}")


if __name__ == "__main__":
    main()
