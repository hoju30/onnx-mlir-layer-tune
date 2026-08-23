#!/usr/bin/env python3
"""Turn one or more build_posit_graph.py graph.json files (Op+Tensor
bipartite schema) into GNN-ready HETEROGENEOUS tensors -- separate feature
matrices per node type and separate edge_index per (src_type, dst_type)
pair, matching PyG HeteroData / DGL heterograph conventions. Op and Tensor
nodes have different feature schemas (structural-only vs role+value-stats),
so they are never concatenated into one shared matrix or index space.

Per-graph output (npz):
  op_id             [n_op]        categorical op_type id (10.1) -- looked up
                                   through a trainable nn.Embedding inside
                                   the model; this script only assigns the id
  x_op              [n_op, D_op]  op structural features, standardized
  op_node_ids       [n_op]        real op names, for traceability only

  role_id           [n_tensor]        categorical role id (activation/weight/
                                       bias/input, see ROLE_VOCAB)
  dtype_id          [n_tensor]        categorical dtype_kind id (posit/float/
                                       int, see DTYPE_VOCAB) -- distinguishes
                                       real numeric tensors from integer
                                       index/shape constants bucketed under
                                       role="weight" (e.g. Reshape's shape operand)
  x_tensor          [n_tensor, D_t]   tensor rank/elements/value-stats/format
  tensor_node_ids   [n_tensor]        real SSA names (e.g. "%10"), traceability only

  edge_index_produces      [2, E_p]  (op_idx -> tensor_idx), forward "produces"
  edge_attr_produces       [E_p, 1]  producer_output_index
  edge_index_produces_rev  [2, E_p]  (tensor_idx -> op_idx), reverse message passing
  edge_attr_produces_rev   [E_p, 1]  producer_output_index (same position info, reverse direction)
  edge_index_consumes      [2, E_c]  (tensor_idx -> op_idx), forward "consumes"
  edge_attr_consumes       [E_c, 1]  operand_index
  edge_index_consumes_rev  [2, E_c]  (op_idx -> tensor_idx), reverse message passing
  edge_attr_consumes_rev   [E_c, 1]  operand_index (same position info, reverse direction)

  y                 [2]      real measured [delta_top1_vs_fp32, delta_top5_vs_fp32]
                              whole-model accuracy loss (claude.md section 12 label),
                              NaN-filled if the source graph has no "label" entry;
                              column names in target_names

Shared across all input graphs (op_vocab.json, role_vocab.json,
norm_stats.json): fixed id tables and feature mean/std, fit here across
whatever graphs are passed in -- for a real train/val/test split these must
be fit on the training split only and reused, per claude.md 10.3 and 22.

op_vocab/role_vocab are FIXED tables, not inferred per-run, so ids stay
stable across every dataset ever built with this script. New op types are
appended with the next free id; never renumber or remove existing entries.
"""
import argparse
import json
import os

import numpy as np

# ---- Op node feature schema ----
OP_STATIC_FIELDS = [
    "kernel_h", "kernel_w", "stride_h", "stride_w", "pad_total",
    "dilation_h", "dilation_w", "group",
    "topological_position", "in_degree", "out_degree", "has_weight", "log_flops",
]
OP_STANDARDIZE_FIELDS = ["kernel_h", "kernel_w", "stride_h", "stride_w", "pad_total",
                         "topological_position", "log_flops"]

# ---- Tensor node feature schema ----
TENSOR_VALUE_FIELDS = ["value_mean", "value_std", "value_p99_abs", "value_zero_ratio", "value_dynamic_range"]
TENSOR_STATIC_FIELDS = ["rank", "log_elements", "topological_position", "in_degree", "out_degree",
                        "value_stats_available"] + TENSOR_VALUE_FIELDS
TENSOR_STANDARDIZE_FIELDS = ["log_elements", "topological_position"] + TENSOR_VALUE_FIELDS

# format (10.2): N and ES are normalized against the largest candidate format (posit_32), not z-scored.
# Shared by both node types -- same formula, same meaning either way.
FORMAT_MAX_N, FORMAT_MAX_ES = 32, 2

# fixed, append-only op_type -> id table (see module docstring).
# The first 14 are every ONNX op the posit conversion pass currently
# supports (src/Conversion/ONNXToPosit/ONNXToPosit.cpp addDynamicallyLegalOp
# list == experiments/common/extract_op_features.py SUPPORTED_TYPES).
# "Softmax" is the only currently-known *unsupported* op that still shows
# up as a graph node (it stays fp32). "UNK" is the fallback bucket for any
# op_type encountered later that isn't in this table yet.
OP_VOCAB = {
    "Add": 0, "Sub": 1, "Mul": 2, "Div": 3, "Clip": 4, "Conv": 5, "Flatten": 6,
    "Gemm": 7, "MatMul": 8, "MaxPool": 9, "ReduceMean": 10, "Relu": 11,
    "Reshape": 12, "Unsqueeze": 13,
    "Softmax": 14,
    "UNK": 15,
}
# fixed 4-role taxonomy (build_posit_graph.py's OPERAND_ROLES design)
ROLE_VOCAB = {"activation": 0, "weight": 1, "bias": 2, "input": 3}
# fixed taxonomy matching build_posit_graph.py's format_of_type_str() dtype_kind
DTYPE_VOCAB = {"posit": 0, "float": 1, "int": 2}


def op_id_of(op_type):
    if op_type not in OP_VOCAB:
        print(f"warning: op_type '{op_type}' not in OP_VOCAB, mapping to UNK -- consider appending it to OP_VOCAB")
        return OP_VOCAB["UNK"]
    return OP_VOCAB[op_type]


def role_id_of(role):
    if role not in ROLE_VOCAB:
        raise ValueError(f"unknown tensor role '{role}' -- not in the fixed ROLE_VOCAB taxonomy")
    return ROLE_VOCAB[role]


def dtype_id_of(dtype_kind):
    if dtype_kind not in DTYPE_VOCAB:
        raise ValueError(f"unknown dtype_kind '{dtype_kind}' -- not in the fixed DTYPE_VOCAB taxonomy")
    return DTYPE_VOCAB[dtype_kind]


def fit_norm_stats(items, fields):
    stats = {}
    for field in fields:
        values = np.array([item[field] for item in items], dtype=np.float64)
        std = values.std()
        stats[field] = {"mean": float(values.mean()), "std": float(std) if std > 1e-12 else 1.0}
    return stats


def zscore(value, field, norm_stats):
    s = norm_stats[field]
    return (value - s["mean"]) / s["std"]


def format_features(node):
    return [node["N"] / FORMAT_MAX_N, node["ES"] / FORMAT_MAX_ES, float(node["is_fp32"])]


def vectorize_op_nodes(op_nodes, norm_stats):
    op_id = np.array([op_id_of(n["op_type"]) for n in op_nodes], dtype=np.int64)
    rows = []
    for n in op_nodes:
        static = [zscore(n[f], f, norm_stats) if f in OP_STANDARDIZE_FIELDS else float(n[f]) for f in OP_STATIC_FIELDS]
        rows.append(static + format_features(n))
    x_op = np.array(rows, dtype=np.float64).astype(np.float32)
    return op_id, x_op


def vectorize_tensor_nodes(tensor_nodes, norm_stats):
    role_id = np.array([role_id_of(n["role"]) for n in tensor_nodes], dtype=np.int64)
    dtype_id = np.array([dtype_id_of(n["dtype_kind"]) for n in tensor_nodes], dtype=np.int64)
    rows = []
    for n in tensor_nodes:
        static = [zscore(n[f], f, norm_stats) if f in TENSOR_STANDARDIZE_FIELDS else float(n[f]) for f in TENSOR_STATIC_FIELDS]
        rows.append(static + format_features(n))
    x_tensor = np.array(rows, dtype=np.float64).astype(np.float32)
    return role_id, dtype_id, x_tensor


def vectorize_graph(graph, norm_stats):
    op_nodes = [n for n in graph["nodes"] if n["kind"] == "op"]
    tensor_nodes = [n for n in graph["nodes"] if n["kind"] == "tensor"]
    op_idx_of = {n["id"]: i for i, n in enumerate(op_nodes)}
    tensor_idx_of = {n["id"]: i for i, n in enumerate(tensor_nodes)}

    op_id, x_op = vectorize_op_nodes(op_nodes, norm_stats)
    role_id, dtype_id, x_tensor = vectorize_tensor_nodes(tensor_nodes, norm_stats)

    produces_fwd_idx, produces_fwd_attr = [], []
    produces_rev_idx, produces_rev_attr = [], []
    consumes_fwd_idx, consumes_fwd_attr = [], []
    consumes_rev_idx, consumes_rev_attr = [], []
    for e in graph["edges"]:
        if e["relation"] == "produces":
            if e["direction"] == "forward":
                produces_fwd_idx.append((op_idx_of[e["src"]], tensor_idx_of[e["dst"]]))
                produces_fwd_attr.append(float(e["producer_output_index"]))
            else:
                produces_rev_idx.append((tensor_idx_of[e["src"]], op_idx_of[e["dst"]]))
                produces_rev_attr.append(float(e["producer_output_index"]))
        else:  # consumes
            if e["direction"] == "forward":
                consumes_fwd_idx.append((tensor_idx_of[e["src"]], op_idx_of[e["dst"]]))
                consumes_fwd_attr.append(float(e["operand_index"]))
            else:
                consumes_rev_idx.append((op_idx_of[e["src"]], tensor_idx_of[e["dst"]]))
                consumes_rev_attr.append(float(e["operand_index"]))

    def idx_array(pairs):
        arr = np.array(pairs, dtype=np.int64)
        return arr.T if arr.size else np.zeros((2, 0), dtype=np.int64)

    def attr_array(vals):
        arr = np.array(vals, dtype=np.float32).reshape(-1, 1)
        return arr if arr.size else np.zeros((0, 1), dtype=np.float32)

    return {
        "op_id": op_id, "x_op": x_op, "op_node_ids": np.array([n["id"] for n in op_nodes]),
        "role_id": role_id, "dtype_id": dtype_id, "x_tensor": x_tensor,
        "tensor_node_ids": np.array([n["id"] for n in tensor_nodes]),
        "edge_index_produces": idx_array(produces_fwd_idx), "edge_attr_produces": attr_array(produces_fwd_attr),
        "edge_index_produces_rev": idx_array(produces_rev_idx), "edge_attr_produces_rev": attr_array(produces_rev_attr),
        "edge_index_consumes": idx_array(consumes_fwd_idx), "edge_attr_consumes": attr_array(consumes_fwd_attr),
        "edge_index_consumes_rev": idx_array(consumes_rev_idx), "edge_attr_consumes_rev": attr_array(consumes_rev_attr),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graph_files", nargs="+", help="one or more *.graph.json files from build_posit_graph.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    graphs = []
    for path in args.graph_files:
        with open(path) as f:
            graphs.append(json.load(f))

    all_op_nodes = [n for g in graphs for n in g["nodes"] if n["kind"] == "op"]
    all_tensor_nodes = [n for g in graphs for n in g["nodes"] if n["kind"] == "tensor"]
    norm_stats = fit_norm_stats(all_op_nodes, OP_STANDARDIZE_FIELDS)
    norm_stats.update(fit_norm_stats(all_tensor_nodes, TENSOR_STANDARDIZE_FIELDS))

    op_feature_names = OP_STATIC_FIELDS + ["N_norm", "ES_norm", "is_fp32"]
    tensor_feature_names = TENSOR_STATIC_FIELDS + ["N_norm", "ES_norm", "is_fp32"]
    target_names = ["delta_top1_vs_fp32", "delta_top5_vs_fp32"]

    with open(f"{args.out_dir}/op_vocab.json", "w") as f:
        json.dump(OP_VOCAB, f, indent=2)
    with open(f"{args.out_dir}/role_vocab.json", "w") as f:
        json.dump(ROLE_VOCAB, f, indent=2)
    with open(f"{args.out_dir}/norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(f"{args.out_dir}/feature_names.json", "w") as f:
        json.dump({"op_feature_names": op_feature_names, "tensor_feature_names": tensor_feature_names}, f, indent=2)

    for path, graph in zip(args.graph_files, graphs):
        v = vectorize_graph(graph, norm_stats)
        label = graph.get("label")
        y = np.array([label["delta_top1_vs_fp32"], label["delta_top5_vs_fp32"]], dtype=np.float32) if label else np.array([np.nan, np.nan], dtype=np.float32)
        stem = path.split("/")[-1].replace(".graph.json", "")
        out_path = f"{os.path.dirname(path) or '.'}/{stem}.vectors.npz"
        np.savez(out_path, y=y, target_names=np.array(target_names),
                 op_feature_names=np.array(op_feature_names), tensor_feature_names=np.array(tensor_feature_names),
                 **v)
        print(f"wrote {out_path}: x_op {v['x_op'].shape}, x_tensor {v['x_tensor'].shape}, "
              f"produces {v['edge_index_produces'].shape[1]}, consumes {v['edge_index_consumes'].shape[1]}, y={y}")

    print(f"op_vocab (fixed): {OP_VOCAB}")
    print(f"role_vocab (fixed): {ROLE_VOCAB}")


if __name__ == "__main__":
    main()
