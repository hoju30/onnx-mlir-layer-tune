#!/usr/bin/env python3
"""
Non-ML greedy planner (claude.md section 9 / 11) -- the first working milestone
and TuneQn-style baseline. Consumes real measured sensitivity data (from
sweep_node_sensitivity.py) directly as "error_increase", standing in for the
ML-predicted error claude.md's original design assumed a trained cost model
would supply. Ranks per-node replacement candidates by cost_saved / error_increase
and greedily accepts within a tolerance budget.

Per-node cost/error use the simple additive formula from claude.md section 2
(`Sigma bit_width(format_i) x tensor_size_i`) for ranking -- this is node-local
so greedy scoring is well-defined. The plan's FINAL total_precision_cost is
reported using precision_cost.py's more accurate tensor-liveness computation.
"""

import argparse
import csv
import json

from precision_cost import compute_cost


def fmt_str(nbits, es):
    return f"posit_{nbits}_{es}"


def load_sensitivity(path):
    by_node = {}
    for r in csv.DictReader(open(path)):
        by_node.setdefault(r["node"], []).append({
            "nbits": int(r["nbits"]), "es": int(r["es"]),
            "delta_top1": float(r["delta_top1_vs_fp32"]),
        })
    return by_node


def plan(op_features, sensitivity_by_node, tolerance, default_bits=32, eps=1e-6):
    ops_by_name = {op["name"]: op for op in op_features["ops"]}

    candidates = []
    for node, cands in sensitivity_by_node.items():
        op = ops_by_name.get(node)
        if not op or not op["supported"]:
            continue
        tensor_size = op["weight_elems"] + op["activation_elems"]
        base_cost = tensor_size * default_bits / 8
        for c in cands:
            if c["nbits"] >= default_bits:
                continue
            cost_saved = base_cost - tensor_size * c["nbits"] / 8
            error_increase = max(c["delta_top1"], 0.0)
            candidates.append({
                "node": node, "nbits": c["nbits"], "es": c["es"],
                "cost_saved": cost_saved, "error_increase": error_increase,
                "score": cost_saved / (error_increase + eps),
            })
    candidates.sort(key=lambda c: -c["score"])

    # Known limitation (flagged for step-5 validation): cumulative_error sums each
    # accepted node's ISOLATED sensitivity delta. Composing multiple simultaneous
    # per-node changes additively is an approximation, not a proven one.
    chosen = {}
    cumulative_error = 0.0
    for c in candidates:
        if c["node"] in chosen:
            continue
        if cumulative_error + c["error_increase"] <= tolerance:
            chosen[c["node"]] = c
            cumulative_error += c["error_increase"]

    format_by_node = {n: (c["nbits"], c["es"]) for n, c in chosen.items()}
    cost = compute_cost(op_features, format_by_node, default_bits=default_bits)

    entities = {}
    for node, cands in sensitivity_by_node.items():
        op = ops_by_name.get(node)
        if not op or not op["supported"]:
            continue
        c = chosen.get(node)
        entities[node] = {
            "chosen": fmt_str(c["nbits"], c["es"]) if c else "FP32",
            "candidates": [fmt_str(x["nbits"], x["es"]) for x in cands],
            "pred_error": c["error_increase"] if c else 0.0,
            "risk_score": c["error_increase"] if c else 0.0,
            "precision_cost": cost["per_node"][node]["weight_bytes"],
        }

    return {
        "stage": "non_ml_greedy_v1",
        "strategy": "measured_sensitivity_greedy",
        "tolerance": tolerance,
        "objective": "minimize_precision_cost_under_error_tolerance",
        "formats": sorted({fmt_str(c["nbits"], c["es"]) for c in candidates} | {"FP32"}),
        "entities": entities,
        "estimated_whole_model_error": cumulative_error,
        "total_precision_cost": cost["weight_bytes"] + cost["peak_activation_bytes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--sensitivity-csv", required=True)
    ap.add_argument("--tolerance", type=float, required=True,
                     help="max cumulative top-1 accuracy-drop budget, e.g. 0.02")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    op_features = json.load(open(args.op_features))
    sensitivity_by_node = load_sensitivity(args.sensitivity_csv)
    result = plan(op_features, sensitivity_by_node, args.tolerance)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    n_chosen = sum(1 for e in result["entities"].values() if e["chosen"] != "FP32")
    print(f"plan: {n_chosen}/{len(result['entities'])} nodes converted, "
          f"estimated_whole_model_error={result['estimated_whole_model_error']:.4f}, "
          f"total_precision_cost={result['total_precision_cost']:.0f} bytes "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
