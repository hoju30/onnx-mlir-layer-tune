#!/usr/bin/env python3
"""General-purpose FP32 activation extraction for the ML calibration
pipeline: runs an ONNX model via onnxruntime with EVERY node's output
exposed as a graph output, so activation calibration generalizes to any
architecture (Conv/BatchNorm/skip-connections/branches/...) without hand-
porting a numpy forward pass per model family.

This replaces the *method* used by calibrate_activations.py's forward()
(which hardcodes the MNIST 6-node linear chain in numpy) with a real ONNX
Runtime execution of the actual model -- the standard "expose all
intermediate tensors as graph outputs" technique. Use --verify-against
to cross-check against calibrate_activations.forward() on the same images;
they should agree to floating-point precision since both execute the exact
same mathematical model.
"""
import argparse
import sys
import os

import numpy as np
import onnx
import onnx.shape_inference
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def expose_all_activations(onnx_path):
    """Return (modified_model, list_of_newly_exposed_output_names)."""
    model = onnx.load(onnx_path)
    existing_outputs = {o.name for o in model.graph.output}
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info_by_name = {vi.name: vi for vi in inferred.graph.value_info}

    exposed = []
    for node in model.graph.node:
        for out_name in node.output:
            if not out_name or out_name in existing_outputs:
                continue
            vi = value_info_by_name.get(out_name)
            if vi is None:
                vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)
            existing_outputs.add(out_name)
            exposed.append(out_name)
    return model, exposed


def run_all_activations(onnx_path, images):
    """images: (N, ...) batch. Returns {tensor_name: array with batch dim}."""
    model, _ = expose_all_activations(onnx_path)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    per_image = {name: [] for name in output_names}
    for img in images:
        outs = sess.run(output_names, {in_name: img[np.newaxis].astype(np.float32)})
        for name, val in zip(output_names, outs):
            per_image[name].append(val)
    return {name: np.concatenate(vals, axis=0) for name, vals in per_image.items()}


def activations_by_node_name(onnx_path, images, op_features):
    """Map onnxruntime's tensor-name-keyed activations to node-name keys,
    using op_features.json's node.outputs[0] (same join key convention as
    build_posit_graph.py / claude.md section 6)."""
    raw = run_all_activations(onnx_path, images)
    out = {}
    for op in op_features["ops"]:
        if op["outputs"] and op["outputs"][0] in raw:
            out[op["name"]] = raw[op["outputs"][0]]
    return out


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="docs/mnist_example/mnist.onnx")
    ap.add_argument("--op-features", default="docs/mnist_example/mnist_op_features.json")
    ap.add_argument("--num-calibration", type=int, default=512)
    ap.add_argument("--verify-against-forward", action="store_true",
                     help="cross-check against calibrate_activations.forward() on the same images")
    args = ap.parse_args()

    from calibrate_activations import load_mnist, forward, stats_of
    from build_posit_graph import decode_all_constants

    with open(args.op_features) as f:
        op_features = json.load(f)

    images, _ = load_mnist(args.num_calibration, train=True)
    activations = activations_by_node_name(args.onnx, images, op_features)

    print(f"exposed {len(activations)} node activations via onnxruntime")
    for name, arr in activations.items():
        s = stats_of(arr)
        print(f"  {name:<10} mean={s['activation_mean']:.4f} std={s['activation_std']:.4f} "
              f"zero_ratio={s['activation_zero_ratio']:.3f}")

    if args.verify_against_forward:
        print("\nverifying against calibrate_activations.forward() (hand-written numpy) ...")
        w_arrays = decode_all_constants("docs/mnist_example/pp/m.mixed.posit.mlir")
        fc1_w, fc1_b, fc2_w, fc2_b = w_arrays[0], w_arrays[1], w_arrays[2], w_arrays[3]
        ref = forward(images, fc1_w, fc1_b, fc2_w, fc2_b)
        for name, ref_arr in ref.items():
            ort_arr = activations.get(name)
            if ort_arr is None:
                print(f"  {name}: MISSING from onnxruntime activations")
                continue
            max_diff = np.max(np.abs(ref_arr.reshape(ort_arr.shape) - ort_arr))
            status = "OK" if max_diff < 1e-4 else "MISMATCH"
            print(f"  {name:<10} max_abs_diff={max_diff:.2e}  {status}")


if __name__ == "__main__":
    main()
