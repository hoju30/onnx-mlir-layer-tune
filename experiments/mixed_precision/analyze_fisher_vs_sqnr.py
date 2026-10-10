#!/usr/bin/env python3
"""MobileNetV2: where do the per-tensor Fisher and SQNR MILP configs differ, and which per-node predictor is biased?

A. Allocation: at each budget, the Fisher pick and the SQNR pick (runs/mbv2_shortlist_check_20261004), per layer
   type: weight bits averaged over weight elements, activation bits averaged over the BOPs proxy, and the type's share
   of the config's weight bytes and BOPs.
B. Bias by layer type: on the single-node ground truth of runs/mbv2_predictor_check_20261003 (each node quantized
   alone, held-out loss increase), log(predictor / dloss) per (node, config), centred on its median over all nodes;
   a type with a positive offset is over-estimated by that predictor relative to the other types, negative under-.
   Only pairs whose dloss is above 2 SE.
No model run: reads saved results only.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402

from eval_random_mixed_precision import supported_nodes  # noqa: E402
from htq_pareto_search import FORMAT_BITS as B  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402

RUNS = HERE / "runs"
TYPES = ("stem", "ds_dw", "ds_pw", "pw", "dw", "pwl", "head", "fc")


def layer_type(i, n):
    if i == 0:
        return "stem"
    if i in (1, 2):
        return ("ds_dw", "ds_pw")[i - 1]
    if i == n - 2:
        return "head"
    if i == n - 1:
        return "fc"
    return ("pw", "dw", "pwl")[(i - 3) % 3]


def main():
    graph = onnx.load(str(HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra/model.onnx"))
    all_nodes = supported_nodes(graph)
    names = [n["name"] for n in all_nodes if n.get("weight_const")]
    kind = {n: layer_type(i, len(names)) for i, n in enumerate(names)}
    we, fl = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    print("layer types: " + ", ".join(f"{t} {sum(kind[n] == t for n in names)} nodes, "
                                      f"{sum(we[n] for n in names if kind[n] == t) / sum(we[n] for n in names):.0%} "
                                      f"of weights, {sum(fl[n] for n in names if kind[n] == t) / sum(fl[n] for n in names):.0%} "
                                      f"of MACs" for t in TYPES))

    # --- A. allocation ---
    rows = json.loads((RUNS / "mbv2_shortlist_check_20261004/results.json").read_text())["candidates"]
    held = {(r["budget"], r["label"]): r for r in rows}
    print("\nA. allocation, Fisher -> SQNR: weight bits (per weight element) | act bits (per BOPs) | "
          "share of bytes | share of BOPs")
    for f in sorted({r["budget"] for r in rows}):
        print(f"  f={f:.2f}  (held-out loss Fisher {held[(f, 'fisher')]['loss']:.3f}, SQNR {held[(f, 'sqnr')]['loss']:.3f})")
        cfg = {k: held[(f, k)]["formats"] for k in ("fisher", "sqnr")}
        tot = {k: (sum(we[n] * B[c[n]["w"]] for n in names), sum(fl[n] * B[c[n]["w"]] * B[c[n]["a"]] for n in names))
               for k, c in cfg.items()}
        for t in TYPES:
            ns = [n for n in names if kind[n] == t]
            out = []
            for k in ("fisher", "sqnr"):
                c = cfg[k]
                wb = sum(we[n] * B[c[n]["w"]] for n in ns) / sum(we[n] for n in ns)
                ab = sum(fl[n] * B[c[n]["w"]] * B[c[n]["a"]] for n in ns) / sum(fl[n] * B[c[n]["w"]] for n in ns)
                sb = sum(we[n] * B[c[n]["w"]] for n in ns) / tot[k][0]
                sf = sum(fl[n] * B[c[n]["w"]] * B[c[n]["a"]] for n in ns) / tot[k][1]
                out.append((wb, ab, sb, sf))
            (w0, a0, s0, f0), (w1, a1, s1, f1) = out
            diff = sum(cfg["fisher"][n] != cfg["sqnr"][n] for n in ns)
            print(f"    {t:<6} w {w0:5.2f} -> {w1:5.2f} | a {a0:5.2f} -> {a1:5.2f} | bytes {s0:5.1%} -> {s1:5.1%} | "
                  f"BOPs {f0:5.1%} -> {f1:5.1%}   ({diff}/{len(ns)} nodes differ)")

    # --- B. predictor bias by layer type ---
    pc = json.loads((RUNS / "mbv2_predictor_check_20261003/results.json").read_text())
    sq = json.loads((RUNS / "mobilenetv2_timm_ra_criteria_milp/sqnr_table.json").read_text())["table"]
    print(f"\nB. bias by layer type: median of log10(predictor / dloss) minus its median over all nodes "
          f"(+ = over-estimated, - = under-estimated); single-node truth on {pc['num_eval']} held-out images")
    for cfg in [tuple(c) for c in pc["configs"]] + [None]:
        sel = [r for r in pc["rows"] if (cfg is None or (r["w"], r["a"]) == cfg) and r["dloss"] > 2 * r["dloss_se"]]
        lr = {p: np.log10([(r["fisher"] if p == "fisher" else sq[r["node"]][f"{r['w']}|{r['a']}"]) / r["dloss"]
                           for r in sel]) for p in ("fisher", "sqnr")}
        cen = {p: v - np.median(v) for p, v in lr.items()}
        label = "all configs" if cfg is None else f"w={cfg[0]} a={cfg[1]}"
        print(f"  {label} ({len(sel)} pairs above 2 SE; spread of log10 ratio: Fisher IQR "
              f"{np.subtract(*np.percentile(lr['fisher'], [75, 25])):.2f}, SQNR {np.subtract(*np.percentile(lr['sqnr'], [75, 25])):.2f})")
        for t in TYPES:
            ix = [i for i, r in enumerate(sel) if kind[r["node"]] == t]
            if ix:
                print(f"    {t:<6} n={len(ix):2d}  Fisher {np.median(cen['fisher'][ix]):+5.2f}   "
                      f"SQNR {np.median(cen['sqnr'][ix]):+5.2f}")


if __name__ == "__main__":
    main()
