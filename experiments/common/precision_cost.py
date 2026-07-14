#!/usr/bin/env python3
"""
Generalized precision-cost function: weight storage bytes + peak activation
memory bytes for an arbitrary model, given a per-node posit format assignment.

Generalizes docs/mnist_example/eval_4groups.py's compute_cost(), which hardcoded
a strictly-linear tensor chain (true for the tiny MNIST net, false for resnet18's
/ mobilenetv2's skip connections). This version does a real tensor-liveness sweep
using extract_op_features.py's per-node inputs/outputs, so it's correct for
branching graphs too.

format_by_node: {node_name: (nbits, es)} for nodes NOT in this dict (or not
"supported"), bits falls back to default_bits (FP32=32).
"""

import argparse
import json


def bits_for_node(op, format_by_node, default_bits):
    if op["supported"] and op["name"] in format_by_node:
        return format_by_node[op["name"]][0]
    return default_bits


def compute_cost(op_features, format_by_node, default_bits=32):
    ops = op_features["ops"]
    graph_inputs = set(op_features["graph_inputs"])

    weight_bytes = 0.0
    per_node = {}
    bits_by_output = {}
    for op in ops:
        bits = bits_for_node(op, format_by_node, default_bits)
        w_bytes = op["weight_elems"] * bits / 8
        weight_bytes += w_bytes
        per_node[op["name"]] = {"bits": bits, "weight_bytes": w_bytes}
        for out_name in op["outputs"]:
            bits_by_output[out_name] = bits

    # tensor liveness: [producing_op_idx, last_consuming_op_idx] for every tensor
    # that is either produced by some op or is a graph input (weights/initializers
    # are excluded -- they never appear as any op's output and are already counted
    # in weight_bytes above).
    producer_idx = {}
    for idx, op in enumerate(ops):
        for out_name in op["outputs"]:
            producer_idx[out_name] = idx
    for name in graph_inputs:
        producer_idx.setdefault(name, -1)
        bits_by_output.setdefault(name, default_bits)

    last_consumer_idx = dict(producer_idx)
    for idx, op in enumerate(ops):
        for in_name in op["inputs"]:
            if in_name in producer_idx:
                last_consumer_idx[in_name] = max(last_consumer_idx.get(in_name, idx), idx)

    tensor_bytes = {
        name: op_features_numel(ops, producer_idx, name) * bits_by_output.get(name, default_bits) / 8
        for name in producer_idx
    }

    peak_bytes = 0.0
    peak_at = None
    for t in range(-1, len(ops)):
        live_bytes = sum(
            tensor_bytes[name]
            for name in producer_idx
            if producer_idx[name] <= t <= last_consumer_idx[name]
        )
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_at = t

    return {
        "weight_bytes": weight_bytes,
        "peak_activation_bytes": peak_bytes,
        "peak_activation_at_op_idx": peak_at,
        "per_node": per_node,
    }


def op_features_numel(ops, producer_idx, tensor_name):
    idx = producer_idx[tensor_name]
    if idx == -1:
        # graph input: find its numel from the first op that consumes it
        for op in ops:
            if tensor_name in op["inputs"]:
                shape = op["input_shapes"][op["inputs"].index(tensor_name)]
                return _numel(shape)
        return 0
    op = ops[idx]
    shape = op["output_shapes"][op["outputs"].index(tensor_name)]
    return _numel(shape)


def _numel(shape):
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= d if d else 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--plan", help="precision_plan.json (entities.<node>.chosen = 'pXXeY')")
    args = ap.parse_args()

    op_features = json.load(open(args.op_features))
    format_by_node = {}
    if args.plan:
        plan = json.load(open(args.plan))
        for name, entry in plan["entities"].items():
            chosen = entry["chosen"]
            if chosen != "FP32":
                nbits = int(chosen.lstrip("p").split("e")[0])
                es = int(chosen.lstrip("p").split("e")[1])
                format_by_node[name] = (nbits, es)

    result = compute_cost(op_features, format_by_node)
    print(json.dumps({k: v for k, v in result.items() if k != "per_node"}, indent=2))


if __name__ == "__main__":
    main()
