#!/usr/bin/env python3
"""Run precision_cost.py's Cost Calculator (claude.md 13-15: weight storage +
peak activation memory via real tensor liveness) on a build_posit_graph.py
graph.json, by reconstructing format_by_node from each node's N/ES/is_fp32.

Also reports the FP32 baseline cost (format_by_node={}) for comparison, since
claude.md 22/RQ5/RQ6 want reduction ratios, not just absolute bytes.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precision_cost import compute_cost


def format_by_node_from_graph(graph):
    return {
        node["id"]: (node["N"], node["ES"])
        for node in graph["nodes"] if not node["is_fp32"]
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graph_file")
    ap.add_argument("--op-features", default="docs/mnist_example/mnist_op_features.json")
    args = ap.parse_args()

    with open(args.graph_file) as f:
        graph = json.load(f)
    with open(args.op_features) as f:
        op_features = json.load(f)

    format_by_node = format_by_node_from_graph(graph)
    fp32_cost = compute_cost(op_features, {})
    cost = compute_cost(op_features, format_by_node)

    print(f"graph: {args.graph_file}")
    print(f"format_by_node: {format_by_node}")
    print(f"  FP32 baseline : weight={fp32_cost['weight_bytes']:>8.0f} bytes  peak_activation={fp32_cost['peak_activation_bytes']:>8.0f} bytes")
    print(f"  this config   : weight={cost['weight_bytes']:>8.0f} bytes  peak_activation={cost['peak_activation_bytes']:>8.0f} bytes")
    print(f"  reduction     : weight={1 - cost['weight_bytes']/fp32_cost['weight_bytes']:>6.1%}         "
          f"peak_activation={1 - cost['peak_activation_bytes']/fp32_cost['peak_activation_bytes']:>6.1%}")
    print(f"  peak occurs at op index: {cost['peak_activation_at_op_idx']} (fp32: {fp32_cost['peak_activation_at_op_idx']})")


if __name__ == "__main__":
    main()
