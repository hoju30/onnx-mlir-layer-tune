#!/usr/bin/env python3
"""Extract a Posit-Dialect Op+Tensor bipartite graph for the MNIST demo net,
following the schema in claude.md sections 6-10 and graph_description.md:

  - Op node    = one lowered operation (Posit or still-FP32 ONNX), NOT
                 constants/casts. Structural attrs only (kernel/stride/pad/
                 dilation/group, N/ES/is_fp32, log_flops) -- weight/activation
                 VALUES live on the Tensor nodes connected to it, not folded
                 into the Op's own features.
  - Tensor node = one SSA value (e.g. "%10"), used ONLY as a unique id, never
                 as an ML feature. Carries a `role` tag (activation / weight /
                 bias / input) plus rank/elements/N/ES/is_fp32/value stats.
                 The SSA name is preserved as `id` purely for traceability.
  - Edge       = Op --produces--> Tensor --consumes--> Op (never a direct
                 Op-to-Op edge -- that would duplicate the same dependency
                 the two Tensor edges already encode). Every edge has both a
                 forward and a reverse direction for bidirectional GNN
                 message passing, per claude.md section 7.3.

Constants (onnx.Constant / posit.constant) and builtin.unrealized_conversion_
cast are still not Op nodes -- a constant becomes a leaf Tensor node (role
weight/bias) instead, and casts are transparently walked through when
resolving which Tensor node an Op's operand actually refers to (so an
activation tensor doesn't fragment into multiple nodes just because a
posit<->fp32 boundary cast sits in the raw SSA graph).

The MNIST demo net is a fixed 6-node chain (MaxPool_0 -> Reshape_2 -> Gemm_3 ->
Relu_4 -> Gemm_5 -> Softmax_6), so node identity and operand roles are
recovered via a hardcoded per-op-name table rather than general SSA analysis.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posit_fakequant import decode_posit_bits, roundtrip
from mnist_model import forward, load_mnist, value_stats

CANONICAL_NODES = [
    ("MaxPool_0", "MaxPool", re.compile(r'posit\.maxpool2d|"onnx\.MaxPoolSingleOut"')),
    ("Reshape_2", "Reshape", re.compile(r'posit\.reshape\b|"onnx\.Reshape"')),
    ("Gemm_3", "Gemm", re.compile(r'posit\.gemm\b|"onnx\.Gemm"')),
    ("Relu_4", "Relu", re.compile(r'posit\.relu\b|"onnx\.Relu"')),
    ("Gemm_5", "Gemm", re.compile(r'posit\.gemm\b|"onnx\.Gemm"')),
    ("Softmax_6", "Softmax", re.compile(r'"onnx\.Softmax"')),
]
# per canonical op: [(operand_index, role), ...]. "weight" also covers
# structural constant operands with no trainable-parameter meaning (e.g.
# Reshape's target-shape operand) -- claude.md's 4-role list (activation/
# weight/bias/input) has no fifth bucket for those, and they're both
# "a static non-activation operand", so this is the closest fit.
OPERAND_ROLES = {
    "MaxPool_0": [(0, "activation")],
    "Reshape_2": [(0, "activation"), (1, "weight")],
    "Gemm_3": [(0, "activation"), (1, "weight"), (2, "bias")],
    "Relu_4": [(0, "activation")],
    "Gemm_5": [(0, "activation"), (1, "weight"), (2, "bias")],
    "Softmax_6": [(0, "activation")],
}
CAST_OP = "builtin.unrealized_conversion_cast"
POSIT_TYPE_RE = re.compile(r"!posit\.type<\s*(\d+)\s*,\s*(\d+)\s*>")
PLAIN_DTYPE_RE = re.compile(r"[xX<]([if])(\d+)>")
STMT_RE = re.compile(r"^\s*(%[\w, ]+) = (.*)$")
OPNAME_RE = re.compile(r'^"?([\w.]+)"?')
NODE_NAME_ATTR_RE = re.compile(r'onnx_node_name = "([^"]+)"')
LAST_ARROW_TYPE_RE = re.compile(r"->\s*([^\n]+)$")
LAST_COLON_TYPE_RE = re.compile(r":\s*([^\n]+)$")
SHAPE_RE = re.compile(r"tensor<((?:\d+x)+)")
HEX_CONST_RE = re.compile(r'dense<"0x([0-9A-Fa-f]+)">\s*:\s*tensor<((?:\d+x)+)f32>')
LIST_CONST_RE = re.compile(r"dense<\[([^\]]+)\]>\s*:\s*tensor<((?:\d+x)+)f32>")
POSIT_CONST_HEX_RE = re.compile(r'dense<"0x([0-9A-Fa-f]+)">\s*:\s*tensor<((?:\d+x)+)i(\d+)>')
POSIT_CONST_LIST_RE = re.compile(r"dense<\[([^\]]+)\]>\s*:\s*tensor<((?:\d+x)+)i(\d+)>")
GENERIC_INT_CONST_LIST_RE = re.compile(r"dense<\[([^\]]+)\]>\s*:\s*tensor<((?:\d+x)+)i(\d+)>")
STORAGE_DTYPE_OF_BITS = {8: np.int8, 16: np.int16, 32: np.int32}


def log1p(x):
    return math.log(1 + x)


def parse_statements(path):
    """Return dict: ssa_id -> {op_name, operands, result_type, node_name_attr, raw}."""
    stmts = {}
    order = []
    with open(path) as f:
        for line in f:
            m = STMT_RE.match(line)
            if not m:
                continue
            lhs, rhs = m.group(1), m.group(2)
            results = [r.strip() for r in lhs.split(",")]
            op_m = OPNAME_RE.match(rhs)
            op_name = op_m.group(1) if op_m else ""
            operand_ids = [t for t in re.findall(r"%[\w]+", rhs)]
            type_m = LAST_ARROW_TYPE_RE.search(rhs) or LAST_COLON_TYPE_RE.search(rhs)
            result_type = type_m.group(1) if type_m else ""
            node_name_m = NODE_NAME_ATTR_RE.search(rhs)
            info = {
                "op_name": op_name,
                "operands": operand_ids,
                "result_type": result_type,
                "node_name_attr": node_name_m.group(1) if node_name_m else None,
                "raw": rhs,
            }
            for r in results:
                stmts[r] = info
            order.append((results[0] if results else None, info))
    return stmts, order


def shape_of_type_str(type_str):
    m = SHAPE_RE.search(type_str)
    if not m:
        return []
    return [int(d) for d in m.group(1).rstrip("x").split("x")]


def format_of_type_str(type_str):
    """(N, ES, is_fp32, dtype_kind) from a tensor's own declared element
    type: a real posit<N,ES>, or otherwise its plain bit-width (f32/i64/...)
    with is_fp32=1 as a generic 'not posit-quantized' marker. dtype_kind
    distinguishes actual numeric data ("posit"/"float") from integer
    index/shape constants ("int", e.g. Reshape's target-shape operand) --
    the latter isn't real parameter storage even though it's bucketed under
    role="weight" for lack of a better fit in the 4-role taxonomy, so cost
    calculators must be able to exclude it from weight-byte accounting."""
    m = POSIT_TYPE_RE.search(type_str)
    if m:
        return int(m.group(1)), int(m.group(2)), 0, "posit"
    m = PLAIN_DTYPE_RE.search(type_str)
    if m:
        kind = "float" if m.group(1) == "f" else "int"
        return int(m.group(2)), 0, 1, kind
    return 32, 0, 1, "float"


def assign_canonical_names(order):
    """Match canonical nodes to statements, preferring the real onnx_node_name
    attribute -- the ONNXToPosit lowering patterns now propagate it onto the
    Posit op they create (see copyOnnxNodeName in Pattern/Math.cpp), so this
    is real node identity, not a guess. Sequential op-mnemonic matching is
    kept only as a fallback for IR built before that fix."""
    canonical_names = {name for name, _, _ in CANONICAL_NODES}
    canonical_of = {}
    stmt_of_canonical = {}

    for ssa_id, info in order:
        nm = info["node_name_attr"]
        if nm in canonical_names and nm not in stmt_of_canonical:
            canonical_of[ssa_id] = nm
            stmt_of_canonical[nm] = (ssa_id, info)

    pending = [n for n in CANONICAL_NODES if n[0] not in stmt_of_canonical]
    claimed_ssa = set(canonical_of)
    for ssa_id, info in order:
        if not pending or ssa_id in claimed_ssa:
            continue
        name, _, pattern = pending[0]
        if pattern.search(info["raw"]) or pattern.search(info["op_name"]):
            canonical_of[ssa_id] = name
            stmt_of_canonical[name] = (ssa_id, info)
            pending.pop(0)
    if pending:
        missing = [n for n, _, _ in pending]
        raise RuntimeError(f"could not locate node(s) {missing} in IR text")
    return canonical_of, stmt_of_canonical


def resolve_producer(ssa_id, stmts, canonical_of):
    """Walk through bookkeeping casts to find which canonical Op (if any)
    produced this SSA value. None means it's the graph input or a constant."""
    if ssa_id == "%arg0":
        return None
    if ssa_id in canonical_of:
        return canonical_of[ssa_id]
    info = stmts.get(ssa_id)
    if info is None:
        return None
    if info["op_name"] == CAST_OP and info["operands"]:
        return resolve_producer(info["operands"][0], stmts, canonical_of)
    return None


def decode_fp32_constants(path):
    """Best-effort decode of plain-FP32 dense constants (hex or literal) in an IR
    file that hasn't been posit-converted yet (e.g. m.mixed.posit.mlir), keyed by
    flattened numpy array in file order. Posit-encoded constants are skipped."""
    arrays = []
    with open(path) as f:
        for line in f:
            m = HEX_CONST_RE.search(line)
            if m:
                raw = bytes.fromhex(m.group(1))
                shape = [int(d) for d in m.group(2).rstrip("x").split("x")]
                arr = np.frombuffer(raw, dtype="<f4")
                if arr.size == int(np.prod(shape)):
                    arrays.append(arr.reshape(shape))
                continue
            m = LIST_CONST_RE.search(line)
            if m:
                shape = [int(d) for d in m.group(2).rstrip("x").split("x")]
                vals = [float(x) for x in m.group(1).split(",")]
                arr = np.array(vals, dtype="<f4")
                if arr.size == int(np.prod(shape)):
                    arrays.append(arr.reshape(shape))
    return arrays


def decode_posit_constants(path):
    """Decode posit.constant dense<...> attributes -- already-quantized iN
    two's-complement bit patterns, not FP32 -- back to real FP32 values via
    the actual Universal posit<N,ES> decoder (posit_fakequant.decode_posit_bits),
    keyed by flattened numpy array in file order."""
    arrays = []
    with open(path) as f:
        for line in f:
            if "posit.constant" not in line:
                continue
            fmt_m = POSIT_TYPE_RE.search(line)
            if not fmt_m:
                continue
            nbits, es = int(fmt_m.group(1)), int(fmt_m.group(2))

            m = POSIT_CONST_HEX_RE.search(line)
            if m:
                shape = [int(d) for d in m.group(2).rstrip("x").split("x")]
                n = int(np.prod(shape))
                raw = bytes.fromhex(m.group(1))
                if len(raw) == n * (int(m.group(3)) // 8):
                    arrays.append(decode_posit_bits(raw, n, nbits, es).reshape(shape))
                continue

            m = POSIT_CONST_LIST_RE.search(line)
            if m:
                shape = [int(d) for d in m.group(2).rstrip("x").split("x")]
                n = int(np.prod(shape))
                storage_bits = int(m.group(3))
                dtype = STORAGE_DTYPE_OF_BITS.get(storage_bits)
                if dtype is None:
                    continue
                vals = np.array([int(x) for x in m.group(1).split(",")], dtype=dtype)
                if vals.size == n:
                    arrays.append(decode_posit_bits(vals.tobytes(), n, nbits, es).reshape(shape))
    return arrays


def decode_all_constants(path):
    """Decode every dense constant in an IR file to real FP32 values, in file
    order, regardless of whether it's still a plain-FP32 onnx.Constant or has
    already been folded into a posit.constant (e.g. under
    POSIT_COMPACT_CONSTANTS) -- so weight/activation stats work on any IR
    file directly, not just ones that happen to keep plain FP32 constants."""
    fp32 = decode_fp32_constants(path)
    posit = decode_posit_constants(path)
    if not fp32:
        return posit
    if not posit:
        return fp32
    # both kinds present (a partially-converted file): merge back into file order
    order = []
    with open(path) as f:
        for line in f:
            if HEX_CONST_RE.search(line) or LIST_CONST_RE.search(line):
                order.append("fp32")
            elif "posit.constant" in line and (POSIT_CONST_HEX_RE.search(line) or POSIT_CONST_LIST_RE.search(line)):
                order.append("posit")
    fp32_it, posit_it = iter(fp32), iter(posit)
    return [next(fp32_it) if kind == "fp32" else next(posit_it) for kind in order]


def decode_int_literal_constant(raw_line):
    """Generic (non-posit, non-fp32) literal-list integer constant, e.g.
    Reshape's target-shape operand `dense<[-1, 196]> : tensor<2xi64>`."""
    m = GENERIC_INT_CONST_LIST_RE.search(raw_line)
    if not m:
        return None
    shape = [int(d) for d in m.group(2).rstrip("x").split("x")]
    vals = np.array([float(x) for x in m.group(1).split(",")], dtype=np.float64)
    return vals.reshape(shape) if vals.size == int(np.prod(shape)) else None


ZERO_VALUE_STATS = {
    "value_mean": 0.0, "value_std": 0.0, "value_p99_abs": 0.0,
    "value_zero_ratio": 0.0, "value_dynamic_range": 0.0,
}

MAXPOOL_ATTR_RE = {
    "kernel_shape": re.compile(r"kernel_shape = \[(\d+),\s*(\d+)\]"),
    "strides": re.compile(r"strides = \[(\d+),\s*(\d+)\]"),
    "pads": re.compile(r"pads = \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"),
}


def maxpool_attrs(raw):
    k = MAXPOOL_ATTR_RE["kernel_shape"].search(raw)
    s = MAXPOOL_ATTR_RE["strides"].search(raw)
    p = MAXPOOL_ATTR_RE["pads"].search(raw)
    pad_total = sum(int(p.group(i)) for i in range(1, 5)) if p else 0
    return {
        "kernel_h": int(k.group(1)) if k else 0, "kernel_w": int(k.group(2)) if k else 0,
        "stride_h": int(s.group(1)) if s else 0, "stride_w": int(s.group(2)) if s else 0,
        "pad_total": pad_total,
    }


DEFAULT_CONV_ATTRS = {
    "kernel_h": 0, "kernel_w": 0, "stride_h": 0, "stride_w": 0,
    "pad_total": 0,
}


def load_whole_model_label(measurements_path, ir_file):
    """The real claude.md section-12 regression label: measured whole-model
    accuracy loss for the exact configuration this IR file represents (as
    opposed to single-node sensitivity, which is only an auxiliary feature
    per section 8.5)."""
    with open(measurements_path) as f:
        data = json.load(f)
    key = ir_file.split("/")[-1]
    entry = data["configs"].get(key)
    if entry is None:
        return None
    return {
        "top1": entry["top1"], "top5": entry["top5"],
        "delta_top1_vs_fp32": entry["delta_top1_vs_fp32"],
        "delta_top5_vs_fp32": entry["delta_top5_vs_fp32"],
        "fp32_top1": data["fp32_baseline"]["top1"], "fp32_top5": data["fp32_baseline"]["top5"],
    }


def value_stats_for(value, fmt):
    """Quantization-aware value_stats: round-trip through the tensor's own
    (N, ES) before computing stats when it's actually posit-quantized, same
    rule for every tensor role (weight/bias/activation/input)."""
    if value is None:
        return dict(ZERO_VALUE_STATS), False
    flat = value.ravel()
    if not fmt["is_fp32"]:
        flat = roundtrip(flat, fmt["N"], fmt["ES"])
    return value_stats(flat, "value"), True


def build_tensors(node_order, stmt_of_canonical, stmts, canonical_of, op_features, weight_by_node, activations):
    """Returns (tensors: dict[key -> tensor info]) with each tensor's role,
    producer op (or None), list of (consumer_op, operand_index), shape,
    format, and raw value array (or None if unavailable)."""
    ops_by_name = {op["name"]: op for op in op_features["ops"]}
    tensors = {}

    def get_or_create(key, id_, role):
        if key not in tensors:
            tensors[key] = {"id": id_, "role": role, "producer": None, "consumers": []}
        return tensors[key]

    image_shape = ops_by_name[node_order[0]]["input_shapes"][0]
    get_or_create("graph_input", "%arg0", "input")

    for op_name in node_order:
        ssa_id, info = stmt_of_canonical[op_name]
        for operand_pos, role in OPERAND_ROLES[op_name]:
            if operand_pos >= len(info["operands"]):
                continue
            operand_ssa = info["operands"][operand_pos]
            if role == "activation":
                producer_op = resolve_producer(operand_ssa, stmts, canonical_of)
                if producer_op is None:
                    t = tensors["graph_input"]
                else:
                    key = f"op_out::{producer_op}"
                    t = get_or_create(key, stmt_of_canonical[producer_op][0], "activation")
                    t["producer"] = producer_op
            else:
                key = f"const::{operand_ssa}"
                t = get_or_create(key, operand_ssa, role)
            t["consumers"].append((op_name, operand_pos))

    # register every op's own output even if nothing consumes it (Softmax_6)
    for op_name in node_order:
        key = f"op_out::{op_name}"
        if key not in tensors:
            ssa_id, _ = stmt_of_canonical[op_name]
            tensors[key] = {"id": ssa_id, "role": "activation", "producer": op_name, "consumers": []}

    for key, t in tensors.items():
        if t["role"] == "input":
            t["shape"] = image_shape
            t["N"], t["ES"], t["is_fp32"], t["dtype_kind"] = 32, 0, 1, "float"
            t["value"] = None  # calibration image batch, not a single fixed tensor -- see note below
        elif t["producer"] is not None:
            producer_stmt = stmt_of_canonical[t["producer"]][1]
            t["shape"] = shape_of_type_str(producer_stmt["result_type"])
            t["N"], t["ES"], t["is_fp32"], t["dtype_kind"] = format_of_type_str(producer_stmt["result_type"])
            t["value"] = activations.get(t["producer"])
        else:
            op_name, operand_pos = t["consumers"][0]
            arr = None
            if t["role"] == "weight" and operand_pos == 1 and op_name in weight_by_node:
                arr = weight_by_node[op_name]["weight"]
            elif t["role"] == "bias" and operand_pos == 2 and op_name in weight_by_node:
                arr = weight_by_node[op_name]["bias"]
            elif op_name == "Reshape_2" and operand_pos == 1:
                arr = decode_int_literal_constant(stmts[t["id"]]["raw"])
            t["value"] = arr
            t["shape"] = list(arr.shape) if arr is not None else []
            const_stmt = stmts.get(t["id"])
            t["N"], t["ES"], t["is_fp32"], t["dtype_kind"] = format_of_type_str(const_stmt["raw"]) if const_stmt else (32, 0, 1, "float")
    return tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ir_file")
    ap.add_argument("--op-features", default="docs/mnist_example/mnist_op_features.json")
    ap.add_argument("--whole-model-measurements", default="docs/mnist_example/mnist_whole_model_measurements.json",
                    help="real measured whole-model accuracy per config (claude.md section 12 label)")
    ap.add_argument("--weight-source-ir", default=None,
                    help="IR file to decode weight constants from (fp32 or posit.constant, both "
                         "handled). Defaults to ir_file itself now that decode_all_constants() can "
                         "decode posit.constant directly -- no need to borrow from a sibling file.")
    ap.add_argument("--num-calibration", type=int, default=512,
                    help="MNIST train-split calibration images for computing activation stats live")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.op_features) as f:
        op_features = json.load(f)

    w_arrays = decode_all_constants(args.weight_source_ir or args.ir_file)
    weight_by_node = {}
    activations = {}
    images = None
    if len(w_arrays) >= 4:
        weight_by_node["Gemm_3"] = {"weight": w_arrays[0], "bias": w_arrays[1]}
        weight_by_node["Gemm_5"] = {"weight": w_arrays[2], "bias": w_arrays[3]}
        images, _ = load_mnist(args.num_calibration, train=True)
        activations = forward(images, w_arrays[0], w_arrays[1], w_arrays[2], w_arrays[3])

    stmts, order = parse_statements(args.ir_file)
    canonical_of, stmt_of_canonical = assign_canonical_names(order)
    node_order = [name for name, _, _ in CANONICAL_NODES]
    n_ops = len(node_order)

    tensors = build_tensors(node_order, stmt_of_canonical, stmts, canonical_of, op_features, weight_by_node, activations)
    tensors["graph_input"]["value"] = images  # the actual calibration batch

    # ---- op nodes ----
    op_topo = {name: i / (n_ops - 1) if n_ops > 1 else 0.0 for i, name in enumerate(node_order)}
    op_weight_elems = {name: 0 for name in node_order}
    for t in tensors.values():
        if t["role"] in ("weight", "bias"):
            op_name, _ = t["consumers"][0]
            if t["value"] is not None:
                op_weight_elems[op_name] = op_weight_elems.get(op_name, 0) + t["value"].size

    op_nodes = []
    for name in node_order:
        ssa_id, info = stmt_of_canonical[name]
        op = next(o for o in op_features["ops"] if o["name"] == name)
        N, ES, is_fp32, _ = format_of_type_str(info["result_type"] or info["raw"])
        conv_attrs = maxpool_attrs(info["raw"]) if name == "MaxPool_0" else dict(DEFAULT_CONV_ATTRS)
        op_nodes.append({
            "kind": "op",
            "id": name,
            "op_type": op["type"],
            "topological_position": op_topo[name],
            "in_degree": 0, "out_degree": 0,  # filled in after edges are built
            **conv_attrs,
            "dilation_h": 1, "dilation_w": 1, "group": 1,
            "has_weight": int(name in weight_by_node),
            "log_flops": log1p(2 * op_weight_elems[name]),
            "N": N, "ES": ES, "is_fp32": is_fp32,
        })

    # ---- tensor nodes + produces/consumes edges ----
    tensor_nodes = []
    edges = []
    for key, t in tensors.items():
        elements = int(np.prod(t["shape"])) if t["shape"] else 0
        vstats, available = value_stats_for(t["value"], t)
        topo = 0.0 if t["role"] in ("input", "weight", "bias") and t["producer"] is None else op_topo.get(t["producer"], 0.0)
        tensor_nodes.append({
            "kind": "tensor",
            "id": t["id"],
            "role": t["role"],
            "topological_position": topo,
            "rank": len(t["shape"]),
            "elements": elements,
            "log_elements": log1p(elements),
            "N": t["N"], "ES": t["ES"], "is_fp32": t["is_fp32"], "dtype_kind": t["dtype_kind"],
            **vstats,
            "value_stats_available": available,
            "in_degree": 1 if t["producer"] is not None else 0,
            "out_degree": len(t["consumers"]),
        })
        if t["producer"] is not None:
            edges.append({"src": t["producer"], "dst": t["id"], "relation": "produces", "direction": "forward",
                          "operand_index": -1, "producer_output_index": 0})
            edges.append({"src": t["id"], "dst": t["producer"], "relation": "produces", "direction": "reverse",
                          "operand_index": -1, "producer_output_index": 0})
        for op_name, operand_pos in t["consumers"]:
            edges.append({"src": t["id"], "dst": op_name, "relation": "consumes", "direction": "forward",
                          "operand_index": operand_pos, "producer_output_index": -1})
            edges.append({"src": op_name, "dst": t["id"], "relation": "consumes", "direction": "reverse",
                          "operand_index": operand_pos, "producer_output_index": -1})

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

    label = load_whole_model_label(args.whole_model_measurements, args.ir_file)
    graph = {"source_ir": args.ir_file, "label": label, "nodes": op_nodes + tensor_nodes, "edges": edges}
    with open(args.out, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"wrote {args.out}: {len(op_nodes)} op nodes, {len(tensor_nodes)} tensor nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
