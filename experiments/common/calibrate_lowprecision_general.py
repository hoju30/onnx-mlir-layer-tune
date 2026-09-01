#!/usr/bin/env python3
"""Calibration for the low-precision int8/fp8e4m3/fp8e5m2 formats on
arbitrary ONNX models (ResNet18/MobileNetV2-scale, not just the MNIST demo
net -- see calibrate_lowprecision.py for that hardcoded version).

Unlike the MNIST version, there's no numpy reimplementation of the forward
pass to read activations off directly, so this appends every tunable node's
activation input/output tensor as an extra graph output, runs the real ONNX
model once via onnxruntime over a batch of real calibration images, and
computes symmetric-quant scale/zero-point from the observed abs-max --
same convention as docs/LowPrecisionFormats.md describes for the pass's own
automatic weight/bias quantization (scale = max_abs / <format max>,
zero_point = 0).

x = the node's own activation input (input[0] -- weight/bias operands are
excluded, they aren't calibrated here since the pass derives their quant
params from the constant itself). y = the node's own output[0].

Usage:
  python calibrate_lowprecision_general.py \
      --onnx experiments/imagenet100/model_dataset/resnet18/torchvision_v1/model.onnx \
      --op-features experiments/imagenet100/model/resnet18_op_features.json \
      --data-root experiments/imagenet100/imagenet100_hf/validation \
      --num-calibration 64 \
      --out experiments/imagenet100/model/resnet18_lowp_calibration.json
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import QMAX, is_tunable

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
QUANT_FORMATS = ["int8", "fp8e4m3", "fp8e5m2"]


def preprocess(img, resize=256, crop=224):
    img = img.convert("RGB")
    w, h = img.size
    scale = resize / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def load_calibration_images(data_root, n):
    from torchvision import datasets
    ds = datasets.ImageFolder(data_root)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    return [ds[i][0] for i in idx]  # PIL images


def build_calibration_session(onnx_path, tunable_ops):
    """Return (session, input_name, {node_name: (x_tensor, y_tensor)})
    where x/y_tensor are the appended output names to read back."""
    model = onnx.load(onnx_path)
    inferred = onnx.shape_inference.infer_shapes(model)
    known_shapes = {vi.name: vi.type for vi in inferred.graph.value_info}
    known_shapes.update({vi.name: vi.type for vi in inferred.graph.input})
    known_shapes.update({vi.name: vi.type for vi in inferred.graph.output})

    existing_outputs = {o.name for o in model.graph.output}
    tap_names = {}
    for op in tunable_ops:
        x_name, y_name = op["inputs"][0], op["outputs"][0]
        tap_names[op["name"]] = (x_name, y_name)
        for t in (x_name, y_name):
            if t in existing_outputs:
                continue
            vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)
            existing_outputs.add(t)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name, tap_names


def scale_zero_point(max_abs, fmt):
    qmax = QMAX[fmt]
    scale = max_abs / qmax if max_abs > 0 else 1.0
    zero_point = 0 if fmt == "int8" else 0.0
    return scale, zero_point


def main():
    global Image
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--data-root", required=True, help="ImageFolder-style calibration image directory")
    ap.add_argument("--num-calibration", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocess", default=None,
                    help="override preprocessing variant (see preprocessing_variants.py)")
    args = ap.parse_args()

    global preprocess
    if args.preprocess:
        from preprocessing_variants import VARIANTS
        preprocess = lambda img: VARIANTS[args.preprocess](img)

    op_features = json.load(open(args.op_features))
    tunable_ops = [o for o in op_features["ops"] if is_tunable(o["type"])]
    print(f"{len(tunable_ops)} tunable nodes to calibrate", flush=True)

    print(f"loading {args.num_calibration} calibration images from {args.data_root} ...", flush=True)
    images = load_calibration_images(args.data_root, args.num_calibration)

    sess, input_name, tap_names = build_calibration_session(args.onnx, tunable_ops)

    output_names = sess.get_outputs()
    name_to_idx = {o.name: i for i, o in enumerate(output_names)}

    max_abs = {name: {"x": 0.0, "y": 0.0} for name in tap_names}
    for i, img in enumerate(images):
        x = preprocess(img)[None, ...]
        outs = sess.run(None, {input_name: x})
        for name, (x_name, y_name) in tap_names.items():
            xv = outs[name_to_idx[x_name]]
            yv = outs[name_to_idx[y_name]]
            max_abs[name]["x"] = max(max_abs[name]["x"], float(np.abs(xv).max()))
            max_abs[name]["y"] = max(max_abs[name]["y"], float(np.abs(yv).max()))
        if (i + 1) % 16 == 0 or i + 1 == len(images):
            print(f"  [{i+1}/{len(images)}]", flush=True)

    calibration = {}
    for name, ma in max_abs.items():
        calibration[name] = {}
        for fmt in QUANT_FORMATS:
            x_scale, x_zp = scale_zero_point(ma["x"], fmt)
            y_scale, y_zp = scale_zero_point(ma["y"], fmt)
            calibration[name][fmt] = {
                "x_scale": x_scale, "x_zero_point": x_zp,
                "y_scale": y_scale, "y_zero_point": y_zp,
            }

    with open(args.out, "w") as f:
        json.dump({"num_calibration": len(images), "data_root": args.data_root,
                   "nodes": calibration}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
