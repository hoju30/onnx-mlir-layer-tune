#!/usr/bin/env python3
"""Op+Tensor graph extraction for the low-precision variant
(claude_lowprecision.md sections 6-10). Unlike build_posit_graph.py, this
does NOT parse a lowered IR file: low-precision lowering doesn't change the
ONNX Dialect's op types (bf16/f16 wrap the same op in Cast; int8/fp8 expand
into a QDQ subgraph only at real-compile time), so graph topology is the
same for every configuration. The graph is extracted ONCE, straight from
mnist_op_features.json + the numpy reimplementation of the net
(mnist_model.py), and a chosen configuration is applied afterward as a pure
node-feature overlay (build_graph_for_config()) -- no re-parsing per sample.

Same MNIST demo net as build_posit_graph.py (6-node chain: MaxPool_0 ->
Reshape_2 -> Gemm_3 -> Relu_4 -> Gemm_5 -> Softmax_6), same Op+Tensor
bipartite schema (Op node = one operation; Tensor node = one SSA value with
a role of activation/weight/bias/input; edges are Op--produces-->Tensor and
Tensor--consumes-->Op, each with a reverse twin for bidirectional GNN
message passing).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import FORMAT_INFO, candidate_formats
from mnist_model import forward, load_mnist, value_stats

NODE_ORDER = ["MaxPool_0", "Reshape_2", "Gemm_3", "Relu_4", "Gemm_5", "Softmax_6"]
OPERAND_ROLES = {
    "MaxPool_0": [(0, "activation")],
    "Reshape_2": [(0, "activation"), (1, "weight")],
    "Gemm_3": [(0, "activation"), (1, "weight"), (2, "bias")],
    "Relu_4": [(0, "activation")],
    "Gemm_5": [(0, "activation"), (1, "weight"), (2, "bias")],
    "Softmax_6": [(0, "activation")],
}
PRODUCER_OF = {  # which op produces the activation each node reads (None = graph input)
    "MaxPool_0": None, "Reshape_2": "MaxPool_0", "Gemm_3": "Reshape_2",
    "Relu_4": "Gemm_3", "Gemm_5": "Relu_4", "Softmax_6": "Gemm_5",
}


def log1p(x):
    return math.log(1 + x)


def build_base_graph(op_features_path, weight_source_ir, num_calibration):
    from build_posit_graph import decode_all_constants  # reuse constant decoder

    op_features = {o["name"]: o for o in json.load(open(op_features_path))["ops"]}
    arrays = decode_all_constants(weight_source_ir)
    fc1_w, fc1_b, fc2_w, fc2_b = arrays[0], arrays[1], arrays[2], arrays[3]
    weight_by_node = {
        "Gemm_3": {"weight": fc1_w, "bias": fc1_b},
        "Gemm_5": {"weight": fc2_w, "bias": fc2_b},
    }

    images, _ = load_mnist(num_calibration, train=True)
    activations = forward(images, fc1_w, fc1_b, fc2_w, fc2_b)

    n_ops = len(NODE_ORDER)
    op_topo = {name: i / (n_ops - 1) if n_ops > 1 else 0.0 for i, name in enumerate(NODE_ORDER)}

    # ---- tensors: graph_input, per-op weight/bias, per-op activation output ----
    tensors = {}
    tensors["graph_input"] = {
        "id": "graph_input", "role": "input", "producer": None, "consumers": [],
        "shape": op_features["MaxPool_0"]["input_shapes"][0],
        "value": images,  # calibration batch, not a single fixed tensor
    }
    for name in NODE_ORDER:
        key = f"op_out::{name}"
        tensors[key] = {
            "id": key, "role": "activation", "producer": name, "consumers": [],
            "shape": op_features[name]["output_shapes"][0] if op_features[name]["output_shapes"] else [],
            "value": activations.get(name),
        }
    for name, w in weight_by_node.items():
        tensors[f"weight::{name}"] = {
            "id": f"weight::{name}", "role": "weight", "producer": None, "consumers": [(name, 1)],
            "shape": list(w["weight"].shape), "value": w["weight"],
        }
        tensors[f"bias::{name}"] = {
            "id": f"bias::{name}", "role": "bias", "producer": None, "consumers": [(name, 2)],
            "shape": list(w["bias"].shape), "value": w["bias"],
        }
    tensors["reshape_target::Reshape_2"] = {
        "id": "reshape_target::Reshape_2", "role": "weight", "producer": None,
        "consumers": [("Reshape_2", 1)], "shape": [2], "value": None,  # index constant, not real param storage
    }

    # wire up consumers for graph_input / op_out tensors from OPERAND_ROLES
    for name in NODE_ORDER:
        for operand_pos, role in OPERAND_ROLES[name]:
            if role != "activation":
                continue
            producer = PRODUCER_OF[name]
            key = "graph_input" if producer is None else f"op_out::{producer}"
            tensors[key]["consumers"].append((name, operand_pos))

    # ---- op nodes (format-independent fields only) ----
    op_weight_elems = {name: 0 for name in NODE_ORDER}
    for t in tensors.values():
        if t["role"] in ("weight", "bias") and t["value"] is not None:
            op_name = t["consumers"][0][0]
            op_weight_elems[op_name] += t["value"].size

    op_nodes = []
    for name in NODE_ORDER:
        op = op_features[name]
        op_nodes.append({
            "kind": "op", "id": name, "op_type": op["type"],
            "topological_position": op_topo[name],
            "in_degree": 0, "out_degree": 0,  # filled below
            "has_weight": int(name in weight_by_node),
            "log_flops": log1p(2 * op_weight_elems[name]),
            "candidate_formats": candidate_formats(op["type"]),
        })

    # ---- tensor nodes + produces/consumes edges ----
    tensor_nodes = []
    edges = []
    for key, t in tensors.items():
        elements = int(np.prod(t["shape"])) if t["shape"] else 0
        vstats = value_stats(t["value"], "value") if t["value"] is not None else {
            "value_mean": 0.0, "value_std": 0.0, "value_p99_abs": 0.0,
            "value_zero_ratio": 0.0, "value_dynamic_range": 0.0,
        }
        topo = op_topo.get(t["producer"], 0.0) if t["producer"] else 0.0
        tensor_nodes.append({
            "kind": "tensor", "id": t["id"], "role": t["role"],
            "topological_position": topo, "rank": len(t["shape"]),
            "elements": elements, "log_elements": log1p(elements),
            **vstats, "value_stats_available": t["value"] is not None,
            "in_degree": 1 if t["producer"] is not None else 0,
            "out_degree": len(t["consumers"]),
        })
        if t["producer"] is not None:
            edges.append({"src": t["producer"], "dst": t["id"], "relation": "produces",
                          "direction": "forward", "operand_index": -1, "producer_output_index": 0})
            edges.append({"src": t["id"], "dst": t["producer"], "relation": "produces",
                          "direction": "reverse", "operand_index": -1, "producer_output_index": 0})
        for op_name, operand_pos in t["consumers"]:
            edges.append({"src": t["id"], "dst": op_name, "relation": "consumes",
                          "direction": "forward", "operand_index": operand_pos, "producer_output_index": -1})
            edges.append({"src": op_name, "dst": t["id"], "relation": "consumes",
                          "direction": "reverse", "operand_index": operand_pos, "producer_output_index": -1})

    in_degree = {n["id"]: 0 for n in op_nodes}
    out_degree = {n["id"]: 0 for n in op_nodes}
    for e in edges:
        if e["direction"] != "forward":
            continue
        if e["src"] in out_degree:
            out_degree[e["src"]] += 1
        if e["dst"] in in_degree:
            in_degree[e["dst"]] += 1
    for n in op_nodes:
        n["in_degree"], n["out_degree"] = in_degree[n["id"]], out_degree[n["id"]]

    return {"nodes": op_nodes + tensor_nodes, "edges": edges}


def apply_config(base_graph, formats):
    """Overlay a per-node format assignment (dict node_name -> format tag,
    e.g. {'Gemm_3': 'int8', 'Relu_4': 'FP32', 'Gemm_5': 'bf16'}) onto a copy
    of the base graph: every op node gets the format's categorical features
    (claude_lowprecision.md 8.4), and every tensor node inherits its
    producing (or owning, for weight/bias) op's format."""
    graph = json.loads(json.dumps(base_graph))  # deep copy
    op_by_id = {n["id"]: n for n in graph["nodes"] if n["kind"] == "op"}

    for name, node in op_by_id.items():
        fmt = formats.get(name, "FP32")
        info = FORMAT_INFO[fmt]
        node["format"] = fmt
        node.update(info)

    def owning_op(t):
        if t["role"] in ("weight", "bias"):
            return t["consumers"][0][0] if t["consumers"] else None
        return t.get("producer")

    for t in graph["nodes"]:
        if t["kind"] != "tensor":
            continue
        # tensor dict lost "producer"/"consumers" (not part of the exported
        # schema); re-derive owning op from edges instead.
        owner = None
        for e in graph["edges"]:
            if e["direction"] != "forward":
                continue
            if e["relation"] == "produces" and e["dst"] == t["id"]:
                owner = e["src"]
            elif t["role"] in ("weight", "bias") and e["relation"] == "consumes" and e["src"] == t["id"]:
                owner = e["dst"]
        fmt = op_by_id[owner]["format"] if owner in op_by_id else "FP32"
        t.update(FORMAT_INFO[fmt])
        t["format"] = fmt

    return graph


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op-features", default="docs/mnist_example/mnist_op_features.json")
    ap.add_argument("--weight-source-ir", default="docs/mnist_example/pp/m.mixed.posit.mlir")
    ap.add_argument("--num-calibration", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    graph = build_base_graph(args.op_features, args.weight_source_ir, args.num_calibration)
    with open(args.out, "w") as f:
        json.dump(graph, f, indent=2)
    n_op = sum(1 for n in graph["nodes"] if n["kind"] == "op")
    n_t = sum(1 for n in graph["nodes"] if n["kind"] == "tensor")
    print(f"wrote {args.out}: {n_op} op nodes, {n_t} tensor nodes, {len(graph['edges'])} edges (format-free base graph)")


if __name__ == "__main__":
    main()
