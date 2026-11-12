#!/usr/bin/env python3
"""Per-channel weights in the full weight + activation MILP: weight term B with activation term C, vs SQNR.

Weights int4..int8 per output channel (run_w8_per_channel_check.py's arithmetic); activations the pass's a= (per-tensor
affine range) on each weight-bearing node's input and output. Criteria, per node and (w, a):
  A     weight mean(g_W^2)*||dW||^2 (per tensor) + activation A (per-tensor mean(g^2)*sum delta^2)
  AC    weight as A + activation C (per-channel residual + coherent per-channel mean, run_mbv2_channel_fisher.py)
  BC    weight sum_o mean_o(g_W^2)*||dW_o||^2 (per output channel) + activation C
  SQNR  single-node logit noise power with per-channel weights: the int-weight entries recomputed, the f16 / bf16 /
        fp32 weight entries reused from runs/criteria_cost_<model>_20261004/sqnr_table.json (they do not depend on it)
Single-node truth at (w, a) = int4/int8, int6/int8, int8/int4, int8/int6 (held-out loss increase, ORT), and MILP
picks at budgets relative to all-int8 weight bytes and BOPs, evaluated on the held-out images.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_criteria_cost_compare import common, match_modules, torch_model  # noqa: E402
from run_mbv2_act_lowbit_check import chan_sum  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant  # noqa: E402
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

AF = FORMATS[:-1]
INT = ("int4", "int5", "int6", "int7", "int8")
CONFIGS = (("int4", "int8"), ("int6", "int8"), ("int8", "int4"), ("int8", "int6"))
CRITERIA = ("A", "AC", "BC", "SQNR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--budgets", default="0.6,0.7,0.8,1.0,1.2")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--num-truth", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(b) for b in args.budgets.split(",")]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    original = erm.np_fake_quant_weight
    erm.np_fake_quant_weight = lambda w, fmt, oracle: (per_channel_int(w, int(fmt[3:])) if fmt.startswith("int")
                                                        else original(w, fmt, oracle))
    summary = {}
    for m in args.models.split(","):
        tag = m.split("/")[0]
        mdir = args.out_dir / tag
        mdir.mkdir(exist_ok=True)
        ns = SimpleNamespace(model_dir=HERE / "../imagenet100/model_dataset" / m, images_dir=args.images_dir)
        graph, meta, held, cx, cy, all_nodes, ranges = common(ns, args.num_eval)
        weighted = [n for n in all_nodes if n.get("weight_const")]
        names = [n["name"] for n in weighted]
        tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in weighted}
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ey = np.array([e["label"] for e in held])

        # --- Fisher tables (per-channel dW) ---
        model, layers = torch_model(meta)
        module_of, inits, by_name = match_modules(graph, weighted, layers)
        cap = {}
        for n, mod in module_of.items():
            mod.register_forward_hook(lambda mm, inp, o, n=n: cap.__setitem__(n, (inp[0], o)))
        g2w = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
        st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
                  for s in ("x", "y")} for n in names}
        for k in range(len(cy)):
            z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
            targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
            grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
            with torch.no_grad():
                for i, n in enumerate(names):
                    g2w[n] += grads[3 * i].double() ** 2
                    for j, s in enumerate(("x", "y")):
                        t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                        a["g2"] = a["g2"] + chan_sum(g ** 2)
                        a["gc"] = a["gc"] + chan_sum(g) ** 2
                        a["n"] += t.numel() // t.shape[1]
                        for f in AF:
                            d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                            a["s1"][f] = a["s1"][f] + chan_sum(d)
                            a["s2"][f] = a["s2"][f] + chan_sum(d ** 2)
            cap.clear()
        CW = {"A": {}, "B": {}}
        CA = {"A": {}, "C": {}}
        for n in names:
            W = inits[by_name[n].input[1]]
            G = g2w[n].numpy().reshape(W.shape[0], -1)
            CW["A"][n], CW["B"][n], CA["A"][n], CA["C"][n] = {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}
            for f in AF:
                d2 = ((erm.np_fake_quant_weight(W, f, None).astype(np.float64) - W) ** 2).reshape(W.shape[0], -1)
                CW["A"][n][f] = float(G.mean() * d2.sum())
                CW["B"][n][f] = float((G.mean(1) * d2.sum(1)).sum())
                a_cost = c_cost = 0.0
                for s in ("x", "y"):
                    a = st[n][s]
                    mu = a["s1"][f] / a["n"]
                    a_cost += float(a["g2"].sum()) / (a["n"] * len(a["g2"])) * float(a["s2"][f].sum())
                    c_cost += float((a["g2"] / a["n"] * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
                CA["A"][n][f], CA["C"][n][f] = a_cost, c_cost
        del model, g2w
        log(f"{tag}: Fisher tables ready ({len(names)} nodes)")

        # --- SQNR with per-channel weights ---
        spath = mdir / "sqnr_table_per_channel.json"
        if spath.exists():
            sq = json.loads(spath.read_text())["table"]
        else:
            old = json.loads((HERE / f"runs/criteria_cost_{tag}_20261004/sqnr_table.json").read_text())["table"]
            ref = erm.simulate_in_onnxruntime(graph, all_nodes, {}, ranges, cx, None).astype(np.float64)
            t_sq = time.time()
            sq = {}
            for i, n in enumerate(names):
                sq[n] = {k: v for k, v in old[n].items() if k.split("|")[0] not in INT}
                for w in INT:
                    for a in FORMATS:
                        z = erm.simulate_in_onnxruntime(graph, all_nodes, {n: {"w": w, "a": a}}, ranges, cx, None)
                        sq[n][f"{w}|{a}"] = float(((z - ref) ** 2).sum()) if np.isfinite(z).all() else None
                if (i + 1) % 20 == 0:
                    log(f"{tag}: SQNR int-weight entries {i + 1}/{len(names)} nodes")
            spath.write_text(json.dumps({"seconds_int_entries": time.time() - t_sq, "table": sq}) + "\n")
        log(f"{tag}: SQNR table (per-channel weights) ready")

        def cost(c, n, w, a):
            if c == "SQNR":
                return sq[n][f"{w}|{a}"]
            return CW["B" if c == "BC" else "A"][n][w] + CA["A" if c == "A" else "C"][n][a]

        def per_image(cfg, x, y):
            z = erm.simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, x, None)
            finite = np.isfinite(z).all(1)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(y),
                                   reduction="none").numpy()
            return ((z.argmax(1) == y) & finite).astype(float), loss

        # --- single-node truth ---
        tpath = mdir / "truth.json"
        if tpath.exists():
            truth = json.loads(tpath.read_text())["rows"]
        else:
            xt, yt = ex[:args.num_truth], ey[:args.num_truth]
            base = per_image({}, xt, yt)[1]
            truth = []
            for i, n in enumerate(names):
                for w, a in CONFIGS:
                    d = per_image({n: {"w": w, "a": a}}, xt, yt)[1] - base
                    truth.append({"node": n, "w": w, "a": a, "dloss": float(d.mean()),
                                  "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
                if (i + 1) % 20 == 0:
                    log(f"{tag}: truth {i + 1}/{len(names)} nodes")
            tpath.write_text(json.dumps({"num_images": len(yt), "rows": truth}, indent=1) + "\n")
        y = np.array([r["dloss"] for r in truth])
        sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in truth])
        pred_stats = {}
        for c in CRITERIA:
            p = np.array([cost(c, r["node"], r["w"], r["a"]) for r in truth], dtype=float)
            lr = np.log10(np.clip(p[sig], 1e-300, None) / y[sig])
            pred_stats[c] = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                             "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                             "by_config": {f"{w}/{a}": float(spearmanr(p[ix], y[ix])[0]) for w, a in CONFIGS
                                           for ix in [np.array([(r["w"], r["a"]) == (w, a) for r in truth])]}}

        # --- MILP picks ---
        weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
        int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
        res, rows = {}, []
        for f in budgets:
            for c in CRITERIA:
                cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a, c=c: cost(c, n, w, a),
                                             weight_elems, flops_proxy, f * int8_bytes, f * int8_bops, time_limit=60)
                assert status in (0, 1), (tag, c, f, status)
                cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
                correct, loss = per_image(cfg, ex, ey)
                res[(f, c)] = (correct, loss)
                size, bops = config_cost(cfg, weight_elems, flops_proxy)
                rows.append({"budget": f, "criterion": c, "milp_optimal": status == 0, "size": size / int8_bytes,
                             "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                             "formats": cfg})
                log(f"{tag} f={f:.2f} {c:<5} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}")
        for r in rows:
            c_, l_ = res[(r["budget"], r["criterion"])]
            for ref_c in ("A", "SQNR"):
                cb, lb = res[(r["budget"], ref_c)]
                d = l_ - lb
                r[f"vs_{ref_c}"] = {"dtop1_pp": float(((c_ - cb) * 100).mean()), "dloss": float(d.mean()),
                                    "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        fp32 = per_image({}, ex, ey)
        w8a8 = per_image({n: {"w": "int8", "a": "int8"} for n in names}, ex, ey)
        summary[tag] = {"nodes": len(names), "fp32": {"top1": float(fp32[0].mean()), "loss": float(fp32[1].mean())},
                        "w8pc_a8": {"top1": float(w8a8[0].mean()), "loss": float(w8a8[1].mean())},
                        "truth_pairs": len(truth), "truth_significant": int(sig.sum()), "prediction": pred_stats,
                        "rows": rows}
        (mdir / "results.json").write_text(json.dumps(summary[tag], indent=2) + "\n")

        print(f"\n=== {m}: per-channel weights; FP32 {fp32[0].mean():.1%}/{fp32[1].mean():.3f}, all W8pc+A8 "
              f"{w8a8[0].mean():.1%}/{w8a8[1].mean():.3f}")
        print(f"   prediction ({len(truth)} pairs, {sig.sum()} > 2 SE): Spearman / sig / IQR; per config")
        for c, s in pred_stats.items():
            print(f"     {c:<5} {s['spearman']:.3f} / {s['spearman_sig']:.3f} / {s['iqr']:.2f}   "
                  + " ".join(f"{k} {v:.3f}" for k, v in s["by_config"].items()))
        print("   MILP (top-1 / loss); mean d loss vs A and vs SQNR")
        for c in CRITERIA:
            sel = [r for r in rows if r["criterion"] == c]
            print(f"     {c:<5} " + "  ".join(f"f={r['budget']:.2f} {r['top1']:5.1%}/{r['loss']:.3f}" for r in sel)
                  + f"   mean {np.mean([r['top1'] for r in sel]):5.1%}/{np.mean([r['loss'] for r in sel]):.3f}"
                  + f"   vs A {np.mean([r['vs_A']['dloss'] for r in sel]):+.3f}"
                  + f"   vs SQNR {np.mean([r['vs_SQNR']['dloss'] for r in sel]):+.3f}")
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in summary.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
