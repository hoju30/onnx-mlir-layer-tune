#!/usr/bin/env python3
"""Calibration for the low-precision int8/fp8e4m3/fp8e5m2 formats
(claude_lowprecision.md section 5.1/20): the --convert-onnx-to-lowprecision
pass does no calibration itself -- LOWP_NODE_FORMATS must carry each
quant-requiring node's activation x_scale/x_zero_point/y_scale/y_zero_point
directly (weight/bias quant params ARE derived automatically at compile
time from the constant's own values, so they're not calibrated here).

Computed once per (node, format) on a calibration subset and cached, since
scale/zero-point only depend on the node's own activation distribution and
the target format -- not on what any other node in the configuration is
using (claude_lowprecision.md section 5.1).

Uses symmetric quantization: scale = max(abs(activation)) / QMAX,
zero_point = 0, the same convention docs/LowPrecisionFormats.md describes
for the pass's automatic weight/bias quantization.

Usage:
  python calibrate_lowprecision.py \
      --weight-source-ir docs/mnist_example/pp/m.mixed.posit.mlir \
      --num-calibration 512 \
      --out docs/mnist_example/mnist_lowp_calibration.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import QMAX
from mnist_model import forward, load_mnist

# (node, input_activation_key) -- input_activation_key is None for the graph
# input (Gemm_3 reads Reshape_2's output, which forward() calls "Reshape_2";
# Gemm_5 reads Relu_4's output).
NODE_INPUTS = {
    "Gemm_3": "Reshape_2",
    "Gemm_5": "Relu_4",
}
QUANT_FORMATS = ["int8", "fp8e4m3", "fp8e5m2"]


def scale_zero_point(arr, fmt):
    max_abs = float(abs(arr).max())
    qmax = QMAX[fmt]
    scale = max_abs / qmax if max_abs > 0 else 1.0
    zero_point = 0 if fmt == "int8" else 0.0
    return scale, zero_point


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight-source-ir", default="docs/mnist_example/pp/m.mixed.posit.mlir",
                    help="IR file to decode fc1/fc2 weight+bias constants from")
    ap.add_argument("--num-calibration", type=int, default=512)
    ap.add_argument("--out", default="docs/mnist_example/mnist_lowp_calibration.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_posit_graph import decode_all_constants  # reuse constant decoder

    arrays = decode_all_constants(args.weight_source_ir)
    fc1_w, fc1_b, fc2_w, fc2_b = arrays[0], arrays[1], arrays[2], arrays[3]

    images, _ = load_mnist(args.num_calibration, train=True)
    activations = forward(images, fc1_w, fc1_b, fc2_w, fc2_b)

    calibration = {}
    for node, input_key in NODE_INPUTS.items():
        x = activations[input_key]
        y = activations[node]
        calibration[node] = {}
        for fmt in QUANT_FORMATS:
            x_scale, x_zp = scale_zero_point(x, fmt)
            y_scale, y_zp = scale_zero_point(y, fmt)
            calibration[node][fmt] = {
                "x_scale": x_scale, "x_zero_point": x_zp,
                "y_scale": y_scale, "y_zero_point": y_zp,
            }

    with open(args.out, "w") as f:
        json.dump({"num_calibration": args.num_calibration, "source": "MNIST train split",
                   "nodes": calibration}, f, indent=2)
    print(f"wrote {args.out}")
    for node, fmts in calibration.items():
        for fmt, p in fmts.items():
            print(f"  {node} {fmt}: x_scale={p['x_scale']:.6g} y_scale={p['y_scale']:.6g}")


if __name__ == "__main__":
    main()
