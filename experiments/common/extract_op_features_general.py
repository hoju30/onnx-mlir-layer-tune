#!/usr/bin/env python3
"""Op-feature extraction for the low-precision dataset pipeline (general
models, ResNet/MobileNet/DenseNet/EfficientNet/VGG/AlexNet/Inception/
ShuffleNet/SqueezeNet scale). Standalone re-implementation: the original
resnet18_op_features.json/mobilenetv2_op_features.json (Posit era) were
produced by a script that no longer exists in the repo, so this defines the
schema build_lowp_graph_general.py / sample_configs_lowp_stratified.py /
calibrate_lowprecision_general.py actually consume, directly from the ONNX
model itself -- no dependency on the missing legacy script.

Per-op fields: name, type, global_idx, input_shapes, output_shapes,
weight_elems, activation_elems, flops, inputs, outputs.
Top-level: graph_inputs, initializer_names, ops.

flops is a MAC-based approximation (2 * weight_elems * output spatial size
for Conv/Gemm/MatMul, 0 otherwise) -- only used for the log_flops auxiliary
node feature, not for accuracy or cost, so exactness isn't required.

Usage:
  python extract_op_features_general.py --onnx model.onnx --out op_features.json
"""
import argparse
import json

import onnx
from onnx import shape_inference


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


FLOP_TYPES = {"Conv", "Gemm", "MatMul"}


def extract(model_path):
    model = onnx.load(model_path)
    shapes = build_shape_map(model)
    init_names = [init.name for init in model.graph.initializer]
    init_name_set = set(init_names)
    graph_input_names = [gi.name for gi in model.graph.input if gi.name not in init_name_set]

    records = []
    for idx, node in enumerate(model.graph.node):
        input_shapes = [shapes.get(name, []) for name in node.input]
        output_shapes = [shapes.get(name, []) for name in node.output]
        weight_elems = sum(numel(shapes.get(name, [])) for name in node.input if name in init_name_set)
        activation_elems = numel(output_shapes[0]) if output_shapes else 0
        out_spatial = numel(output_shapes[0][2:]) if output_shapes and len(output_shapes[0]) > 2 else 1
        flops = 2 * weight_elems * out_spatial if node.op_type in FLOP_TYPES else 0

        records.append({
            "global_idx": idx,
            "name": node.name,
            "type": node.op_type,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "weight_elems": weight_elems,
            "activation_elems": activation_elems,
            "flops": flops,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })
    return {
        "graph_inputs": graph_input_names,
        "initializer_names": init_names,
        "ops": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = extract(args.onnx)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {len(data['ops'])} ops to {args.out}")


if __name__ == "__main__":
    main()