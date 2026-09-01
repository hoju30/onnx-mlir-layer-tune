#!/usr/bin/env python3
"""Op+Tensor graph extraction for arbitrary ONNX models (ResNet18/
MobileNetV2-scale), the general counterpart to build_lowp_graph.py's
MNIST-hardcoded version. Same rationale as that file (claude_lowprecision.md
section 6/7): low-precision lowering doesn't change ONNX Dialect op types,
so topology is configuration-invariant and is extracted ONCE from
*_op_features.json (already has shapes/weight_elems/flops/inputs/outputs
per node -- no IR parsing needed), then apply_config() overlays a chosen
configuration's formats as a pure node-feature pass.

Weight tensor value_stats are read from the ONNX model's own initializers
(exact). Activation value_stats are left zero-filled/unavailable for now --
unlike the MNIST version there's no cheap numpy reimplementation of a
53-Conv-deep net to compute them from; they can be backfilled later from
calibrate_lowprecision_general.py's calibration batch if needed (section 8.3
is an auxiliary feature, not required for the real accuracy label or cost
model this dataset's -- WeightStorage only reads weight tensors, and
PeakActivationMemory only needs each tensor's ELEMENT COUNT, not its values).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import FORMAT_INFO, candidate_formats

ZERO_VALUE_STATS = {
    "value_mean": 0.0, "value_std": 0.0, "value_p99_abs": 0.0,
    "value_zero_ratio": 0.0, "value_dynamic_range": 0.0,
}


def log1p(x):
    return math.log(1 + x)


def value_stats(arr):
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size == 0:
        return dict(ZERO_VALUE_STATS)
    return {
        "value_mean": float(flat.mean()), "value_std": float(flat.std()),
        "value_p99_abs": float(np.percentile(np.abs(flat), 99)),
        "value_zero_ratio": float((flat == 0).mean()),
        "value_dynamic_range": float(flat.max() - flat.min()),
    }


def numel(shape):
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= d if d else 1
    return n


def build_base_graph(op_features_path, onnx_path):
    op_features = json.load(open(op_features_path))
    ops = op_features["ops"]
    init_names = set(op_features.get("initializer_names", []))

    model = onnx.load(onnx_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    n_ops = len(ops)
    op_topo = {o["name"]: o["global_idx"] / (n_ops - 1) if n_ops > 1 else 0.0 for o in ops}

    # tensor_name -> producer op name (from every op's outputs)
    producer_of = {}
    for o in ops:
        for out_name in o["outputs"]:
            producer_of[out_name] = o["name"]
    # tensor_name -> [(consumer_op, operand_index)]
    consumers_of = {}
    for o in ops:
        for idx, in_name in enumerate(o["inputs"]):
            consumers_of.setdefault(in_name, []).append((o["name"], idx))

    graph_inputs = set(op_features.get("graph_inputs", []))

    # ---- tensors ----
    tensors = {}
    for name in graph_inputs:
        tensors[name] = {"id": name, "role": "input", "producer": None,
                         "consumers": consumers_of.get(name, []), "shape": [], "value": None}
    for o in ops:
        for out_name in o["outputs"]:
            if out_name in tensors:
                continue
            tensors[out_name] = {"id": out_name, "role": "activation", "producer": o["name"],
                                 "consumers": consumers_of.get(out_name, []),
                                 "shape": o["output_shapes"][0] if o["output_shapes"] else [], "value": None}
        for idx, in_name in enumerate(o["inputs"]):
            if in_name in tensors:
                continue
            if in_name in init_names:
                role = "bias" if idx >= 2 else "weight"  # Conv/Gemm convention: 0=act,1=weight,2=bias
                arr = initializers.get(in_name)
                tensors[in_name] = {"id": in_name, "role": role, "producer": None,
                                    "consumers": consumers_of.get(in_name, []),
                                    "shape": list(arr.shape) if arr is not None else [], "value": arr}
            elif in_name not in producer_of and in_name not in graph_inputs:
                # untracked input (e.g. a Reshape/Slice's constant shape operand
                # not in initializer_names) -- treat as an opaque weight-role leaf
                tensors[in_name] = {"id": in_name, "role": "weight", "producer": None,
                                    "consumers": consumers_of.get(in_name, []), "shape": [], "value": None}

    # ---- op nodes ----
    op_weight_elems = {o["name"]: o["weight_elems"] for o in ops}
    op_nodes = []
    for o in ops:
        op_nodes.append({
            "kind": "op", "id": o["name"], "op_type": o["type"],
            "topological_position": op_topo[o["name"]],
            "in_degree": 0, "out_degree": 0,  # filled below
            "has_weight": int(o["weight_elems"] > 0),
            "log_flops": log1p(o.get("flops", 0)),
            "candidate_formats": candidate_formats(o["type"]),
        })

    # ---- tensor nodes + produces/consumes edges ----
    tensor_nodes = []
    edges = []
    for key, t in tensors.items():
        elements = int(np.prod(t["shape"])) if t["shape"] else 0
        vstats = value_stats(t["value"]) if t["value"] is not None else dict(ZERO_VALUE_STATS)
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
    """Same overlay as build_lowp_graph.py: every op node gets its chosen
    format's categorical features, every tensor node inherits its owning
    (producing, or weight/bias-consuming) op's format."""
    graph = json.loads(json.dumps(base_graph))
    op_by_id = {n["id"]: n for n in graph["nodes"] if n["kind"] == "op"}

    for name, node in op_by_id.items():
        fmt = formats.get(name, "FP32")
        info = FORMAT_INFO[fmt]
        node["format"] = fmt
        node.update(info)

    for t in graph["nodes"]:
        if t["kind"] != "tensor":
            continue
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
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    graph = build_base_graph(args.op_features, args.onnx)
    with open(args.out, "w") as f:
        json.dump(graph, f, indent=2)
    n_op = sum(1 for n in graph["nodes"] if n["kind"] == "op")
    n_t = sum(1 for n in graph["nodes"] if n["kind"] == "tensor")
    print(f"wrote {args.out}: {n_op} op nodes, {n_t} tensor nodes, {len(graph['edges'])} edges (format-free base graph)")


if __name__ == "__main__":
    main()
