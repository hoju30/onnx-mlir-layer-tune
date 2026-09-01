#!/usr/bin/env python3
"""Cost Calculator for the low-precision variant (claude_lowprecision.md
14-15), computed entirely from a build_lowp_graph.py graph.json that has
already had apply_config() applied. Same liveness-sweep logic as
cost_from_graph.py, just reading the "bitwidth" field apply_config() puts on
every node instead of Posit's "N"/"is_fp32" pair.
"""
import argparse
import json


def compute_cost(graph):
    op_nodes = [n for n in graph["nodes"] if n["kind"] == "op"]
    tensor_nodes = [n for n in graph["nodes"] if n["kind"] == "tensor"]
    op_index = {n["id"]: i for i, n in enumerate(op_nodes)}

    weight_bytes = sum(t["elements"] * t["bitwidth"] / 8 for t in tensor_nodes
                       if t["role"] in ("weight", "bias") and t["value_stats_available"])

    live_tensor_ids = [t["id"] for t in tensor_nodes if t["role"] in ("activation", "input")]
    producer_idx = {tid: -1 for tid in live_tensor_ids}
    last_consumer_idx = {tid: -1 for tid in live_tensor_ids}
    for e in graph["edges"]:
        if e["direction"] != "forward":
            continue
        if e["relation"] == "produces" and e["dst"] in producer_idx:
            producer_idx[e["dst"]] = op_index[e["src"]]
        elif e["relation"] == "consumes" and e["src"] in last_consumer_idx:
            last_consumer_idx[e["src"]] = max(last_consumer_idx[e["src"]], op_index[e["dst"]])
    for tid in live_tensor_ids:
        last_consumer_idx[tid] = max(last_consumer_idx[tid], producer_idx[tid])

    tensor_bytes = {t["id"]: t["elements"] * t["bitwidth"] / 8 for t in tensor_nodes if t["id"] in producer_idx}

    peak_bytes = 0.0
    peak_at = None
    for step in range(-1, len(op_nodes)):
        live = sum(tensor_bytes[tid] for tid in live_tensor_ids if producer_idx[tid] <= step <= last_consumer_idx[tid])
        if live > peak_bytes:
            peak_bytes = live
            peak_at = step

    return {"weight_bytes": weight_bytes, "peak_activation_bytes": peak_bytes, "peak_activation_at_op_idx": peak_at}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graph_file")
    args = ap.parse_args()
    with open(args.graph_file) as f:
        graph = json.load(f)
    cost = compute_cost(graph)
    print(f"weight={cost['weight_bytes']:.0f} bytes  peak_activation={cost['peak_activation_bytes']:.0f} bytes"
          f"  (peak at op idx {cost['peak_activation_at_op_idx']})")


if __name__ == "__main__":
    main()
