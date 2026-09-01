#!/usr/bin/env python3
"""Complete-configuration generator for the low-precision variant
(claude_lowprecision.md section 5.1), over the MNIST demo net's tunable
nodes. Candidate formats per node come from lowp_format_scope.py, i.e. the
docs/LowPrecisionFormats.md per-op scope table -- Gemm_3/Gemm_5 get all 5
non-FP32 formats, Relu_4 only bf16/f16 (MaxPool_0/Reshape_2/Softmax_6 are
untunable and always FP32).

With only 3 tunable nodes and small per-node format sets, the full search
space is 6*3*6 = 108 configurations -- small enough to enumerate exhaustively
rather than stratified-sample, so this generates the complete space (every
sample gets a real accuracy label) instead of a random subset. This still
serves section 5.1's role of feeding build_dataset_lowp.py's Real
Low-Precision Evaluator; representative configurations (all-FP32,
uniform-per-format, front/back-only) are additionally tagged by strategy
for readability, everything else is tagged "exhaustive".

Usage:
  python sample_configs_lowp.py --out docs/mnist_example/mnist_lowp_configs.json
"""
import argparse
import itertools
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import candidate_formats

TUNABLE_NODES = ["Gemm_3", "Relu_4", "Gemm_5"]
OP_TYPES = {"Gemm_3": "Gemm", "Relu_4": "Relu", "Gemm_5": "Gemm"}


def strategy_tag(formats):
    vals = set(formats.values())
    if vals == {"FP32"}:
        return "representative:all_fp32"
    if len(vals) == 1:
        return f"representative:uniform_{next(iter(vals))}"
    if formats["Gemm_3"] != "FP32" and formats["Relu_4"] == "FP32" and formats["Gemm_5"] == "FP32":
        return "representative:front_only"
    if formats["Gemm_3"] == "FP32" and formats["Relu_4"] == "FP32" and formats["Gemm_5"] != "FP32":
        return "representative:back_only"
    return "exhaustive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_node_formats = {n: candidate_formats(OP_TYPES[n]) for n in TUNABLE_NODES}
    configs = []
    for i, combo in enumerate(itertools.product(*(per_node_formats[n] for n in TUNABLE_NODES))):
        formats = dict(zip(TUNABLE_NODES, combo))
        configs.append({
            "id": f"cfg_{i:04d}",
            "strategy": strategy_tag(formats),
            "formats": formats,
        })

    with open(args.out, "w") as f:
        json.dump({
            "tunable_nodes": TUNABLE_NODES,
            "candidate_formats": per_node_formats,
            "configs": configs,
        }, f, indent=2)
    print(f"wrote {len(configs)} configs ({len(per_node_formats['Gemm_3'])}"
          f"x{len(per_node_formats['Relu_4'])}x{len(per_node_formats['Gemm_5'])} exhaustive) to {args.out}")


if __name__ == "__main__":
    main()
