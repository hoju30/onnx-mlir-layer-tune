#!/usr/bin/env python3
"""Fake-quant eval of IEEE float formats vs posit, on the ImageNet100 CNN ONNX.

Formats: f32 (baseline) | fp16 | bf16 | fp8e4m3 (float8_e4m3fn) | fp8e5m2 (float8_e5m2).

--quant-mode:
  weight : only WEIGHT initializers are rounded to the format (back to fp32).
  act    : only ACTIVATIONS are rounded (every float op-output / graph-input gets a
           Cast->format->Cast->fp32 roundtrip inserted into the graph).
  both   : weights AND activations (DEFAULT) -- i.e. "full-model" fp8/fp16/bf16.

The model is then run in fp32 via ONNX Runtime (the fake-quant roundtrips simulate
the format's representable values without needing format-native compute kernels).
Uses the SAME ONNX as the posit pipeline + same validation set + preprocessing, so
Top1 is directly comparable to the posit results.

Activation note: the final graph output (logits) has no consumers so it is left
un-quantized (argmax is unaffected); every inter-layer activation IS quantized.

Example:
  gpt2/bin/python eval_fp_formats.py \
    --onnx model/imagenet100_resnet18.onnx --quant-mode both \
    --formats f32,fp16,bf16,fp8e4m3,fp8e5m2 \
    --data-root imagenet100_hf/validation --limit 5000
"""
import argparse
import copy
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torchvision import datasets, transforms

FMT_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
}
FMT_TP = {
    "fp16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
    "fp8e4m3": TensorProto.FLOAT8E4M3FN,
    "fp8e5m2": TensorProto.FLOAT8E5M2,
}


def fake_quant(arr: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "f32":
        return arr.astype(np.float32)
    t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
    return t.to(FMT_DTYPE[fmt]).to(torch.float32).numpy()


def quantize_weights(m: onnx.ModelProto, fmt: str) -> int:
    nq = 0
    for init in m.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        q = fake_quant(arr, fmt).astype(np.float32)
        init.CopyFrom(numpy_helper.from_array(q, init.name))
        nq += 1
    return nq


def quantize_activations(m: onnx.ModelProto, fmt: str) -> int:
    """Insert Cast(->fmt)->Cast(->fp32) after every float activation tensor."""
    inferred = onnx.shape_inference.infer_shapes(m)
    ftype = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        ftype[vi.name] = vi.type.tensor_type.elem_type
    init_names = {i.name for i in m.graph.initializer}

    candidates = set()
    for gi in m.graph.input:
        if gi.name not in init_names and ftype.get(gi.name) == TensorProto.FLOAT:
            candidates.add(gi.name)
    for node in m.graph.node:
        for o in node.output:
            if ftype.get(o) == TensorProto.FLOAT:
                candidates.add(o)

    target_tp = FMT_TP[fmt]
    rename = {}
    new_nodes = []
    for T in sorted(candidates):
        Tq = f"{T}__q_{fmt}"
        Tfq = f"{T}__fq_{fmt}"
        new_nodes.append(helper.make_node("Cast", [T], [Tq], to=target_tp, name=f"fqd_{Tq}"))
        new_nodes.append(helper.make_node("Cast", [Tq], [Tfq], to=TensorProto.FLOAT, name=f"fqd_{Tfq}"))
        rename[T] = Tfq

    # rewire all consumers (existing nodes only) from T to T_fq
    for node in m.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]
    m.graph.node.extend(new_nodes)
    return len(rename)


def build_quant_model(model: onnx.ModelProto, fmt: str, mode: str):
    m = copy.deepcopy(model)
    nq_w = nq_a = 0
    if fmt != "f32":
        if mode in ("weight", "both"):
            nq_w = quantize_weights(m, fmt)
        if mode in ("act", "both"):
            nq_a = quantize_activations(m, fmt)
    return m.SerializeToString(), nq_w, nq_a


def build_transform(resize, crop, mean, std):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def eval_format(model, fmt, mode, dataset, in_name, limit, progress):
    blob, nq_w, nq_a = build_quant_model(model, fmt, mode)
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    n = min(limit, len(dataset)) if limit > 0 else len(dataset)
    top1 = top5 = 0
    t0 = time.time()
    for i in range(n):
        x, y = dataset[i]
        logits = sess.run(None, {in_name: x.unsqueeze(0).numpy().astype(np.float32)})[0][0]
        top5_idx = logits.argsort()[-5:][::-1]
        top1 += int(top5_idx[0] == y)
        top5 += int(y in top5_idx)
        if progress and (i + 1) % progress == 0:
            print(f"    [{fmt}] {i+1}/{n}  top1={100*top1/(i+1):.2f}%", flush=True)
    return {
        "fmt": fmt, "nq_w": nq_w, "nq_a": nq_a, "n": n,
        "top1": 100.0 * top1 / n, "top5": 100.0 * top5 / n,
        "sec": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data-root", default="imagenet100_hf/validation")
    ap.add_argument("--formats", default="f32,fp16,bf16,fp8e4m3,fp8e5m2")
    ap.add_argument("--quant-mode", choices=["weight", "act", "both"], default="both")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--progress", type=int, default=500)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    in_name = model.graph.input[0].name
    tf = build_transform(args.resize, args.crop, args.mean, args.std)
    ds = datasets.ImageFolder(root=args.data_root, transform=tf)
    print(f"model={args.onnx}  input={in_name}  val_images={len(ds)}  classes={len(ds.classes)}")
    print(f"quant_mode={args.quant_mode}  formats={args.formats}  limit={args.limit}")

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    results = []
    for fmt in formats:
        print(f"--- evaluating {fmt} ({args.quant_mode}) ---", flush=True)
        r = eval_format(model, fmt, args.quant_mode, ds, in_name, args.limit, args.progress)
        results.append(r)
        print(f"  {fmt:8s} top1={r['top1']:.3f}%  top5={r['top5']:.3f}%  "
              f"(W_quant={r['nq_w']}, A_quant={r['nq_a']}, n={r['n']}, {r['sec']:.1f}s)", flush=True)

    print(f"\n==== SUMMARY (quant_mode={args.quant_mode}) ====")
    print(f"{'format':10s} {'top1%':>8s} {'top5%':>8s}")
    for r in results:
        print(f"{r['fmt']:10s} {r['top1']:>8.3f} {r['top5']:>8.3f}")


if __name__ == "__main__":
    main()
