#!/usr/bin/env python3
"""
Extract per-op structural features from an ONNX model: shapes, weight/activation
element counts, tensor connectivity (inputs/outputs).

"supported" marks all 14 op types that POSIT_NODE_FORMATS can override
(see ONNXToPosit.cpp addDynamicallyLegalOp calls):
Conv, Gemm, MatMul, Add, Sub, Mul, Div, Relu, Clip, Reshape, Unsqueeze,
Flatten, MaxPool, ReduceMean.

Usage:
  python extract_op_features.py --onnx model.onnx --out model_op_features.json
"""

import argparse
import json

import onnx
from onnx import shape_inference

SUPPORTED_TYPES = {
    "Conv", "Gemm", "MatMul",
    "Add", "Sub", "Mul", "Div",
    "Relu", "Clip", "Reshape", "Unsqueeze", "Flatten", "MaxPool", "ReduceMean",
}


def _dims(tensor_type_shape):
    out = []
    for d in tensor_type_shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") else None)
    return out


def build_shape_map(model):
    shapes = {}
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    inferred = shape_inference.infer_shapes(model)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.name not in shapes and vi.type.HasField("tensor_type"):
            shapes[vi.name] = _dims(vi.type.tensor_type.shape)
    return shapes


def numel(shape):
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= d if d else 1
    return n



def extract(model_path):
    model = onnx.load(model_path)
    shapes = build_shape_map(model)
    init_names = {init.name for init in model.graph.initializer}
    graph_input_names = [gi.name for gi in model.graph.input if gi.name not in init_names]

    records = []
    for node in model.graph.node:
        input_shapes = [shapes.get(name, []) for name in node.input]
        output_shapes = [shapes.get(name, []) for name in node.output]
        weight_elems = sum(numel(shapes.get(name, [])) for name in node.input if name in init_names)
        activation_elems = numel(output_shapes[0]) if output_shapes else 0

        records.append({
            "name": node.name,
            "type": node.op_type,
            "supported": node.op_type in SUPPORTED_TYPES,
            "weight_elems": weight_elems,
            "activation_elems": activation_elems,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
        })
    return {
        "graph_inputs": graph_input_names,
        "ops": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = extract(args.onnx)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    records = data["ops"]
    n_supported = sum(1 for r in records if r["supported"])
    print(f"wrote {len(records)} ops ({n_supported} posit-format-overridable) to {args.out}")


if __name__ == "__main__":
    main()
