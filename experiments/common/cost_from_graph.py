#!/usr/bin/env python3
"""Cost Calculator (claude.md 13-15: weight storage + peak activation memory
via tensor liveness) computed ENTIRELY from a build_posit_graph.py graph.json
-- no mnist_op_features.json / original ONNX model needed.

Updated for the Op+Tensor bipartite schema: weight storage sums the
elements/bits of every "weight"/"bias" Tensor node directly (no more reading
weight_elements off an Op node); peak activation memory does a liveness
sweep over "activation"/"input" Tensor nodes, using the produces/consumes
edges to find each one's producing and last-consuming Op index.
"""
import argparse
import json


def bits_of(n):
    return n["N"] if not n["is_fp32"] else 32


def compute_cost(graph):
    op_nodes = [n for n in graph["nodes"] if n["kind"] == "op"]
    tensor_nodes = [n for n in graph["nodes"] if n["kind"] == "tensor"]
    op_index = {n["id"]: i for i, n in enumerate(op_nodes)}

    # dtype_kind excludes integer index/shape constants (e.g. Reshape's
    # target-shape operand) bucketed under role="weight" for lack of a better
    # fit in the 4-role taxonomy -- they aren't real parameter storage.
    weight_bytes = sum(t["elements"] * bits_of(t) / 8 for t in tensor_nodes
                       if t["role"] in ("weight", "bias") and t["dtype_kind"] != "int")

    live_tensor_ids = [t["id"] for t in tensor_nodes if t["role"] in ("activation", "input")]
    producer_idx = {tid: -1 for tid in live_tensor_ids}  # -1: no producing op (graph input)
    last_consumer_idx = {tid: -1 for tid in live_tensor_ids}
    for e in graph["edges"]:
        if e["direction"] != "forward":
            continue
        if e["relation"] == "produces" and e["dst"] in producer_idx:
            producer_idx[e["dst"]] = op_index[e["src"]]
        elif e["relation"] == "consumes" and e["src"] in last_consumer_idx:
            last_consumer_idx[e["src"]] = max(last_consumer_idx[e["src"]], op_index[e["dst"]])
    # a tensor with no consumer (e.g. the final output) is still live at the step that produces it
    for tid in live_tensor_ids:
        last_consumer_idx[tid] = max(last_consumer_idx[tid], producer_idx[tid])

    tensor_bytes = {t["id"]: t["elements"] * bits_of(t) / 8 for t in tensor_nodes if t["id"] in producer_idx}

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

    fp32_graph = json.loads(json.dumps(graph))
    for n in fp32_graph["nodes"]:
        n["is_fp32"] = True
    fp32_cost = compute_cost(fp32_graph)
    cost = compute_cost(graph)

    print(f"graph: {args.graph_file}  (pure IR-derived, no op_features.json)")
    print(f"  FP32 baseline : weight={fp32_cost['weight_bytes']:>8.0f} bytes  peak_activation={fp32_cost['peak_activation_bytes']:>8.0f} bytes")
    print(f"  this config   : weight={cost['weight_bytes']:>8.0f} bytes  peak_activation={cost['peak_activation_bytes']:>8.0f} bytes")
    print(f"  reduction     : weight={1 - cost['weight_bytes']/fp32_cost['weight_bytes']:>6.1%}         "
          f"peak_activation={1 - cost['peak_activation_bytes']/fp32_cost['peak_activation_bytes']:>6.1%}")
    print(f"  peak occurs at op index: {cost['peak_activation_at_op_idx']} (fp32: {fp32_cost['peak_activation_at_op_idx']})")


if __name__ == "__main__":
    main()
