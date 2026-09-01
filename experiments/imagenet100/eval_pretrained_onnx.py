#!/usr/bin/env python3
"""
Evaluate a model_dataset/*/model.onnx (ImageNet-1K pretrained, NOT
retrained -- ml_predictor_dataset.md "Retraining: No") on the clane9/
imagenet-100 validation split (5,000 images), via onnxruntime.

Reports two numbers because there's no single obvious accuracy definition
when the model's output space (1000 classes) is bigger than the eval set's
label space (100 classes):
  - unrestricted top1/top5: argmax over the full 1000-way logits, compared
    against the ImageNet-1K index the true label maps to
    (build_imagenet100_label_map.py). This is what the pretrained model
    actually does in the real 1000-class deployment setting.
  - restricted top1/top5: argmax over only the 100 logits belonging to
    imagenet-100 classes. Easier (removes 900 nuisance classes as
    confusers), and closer to the "how good is this model AS a 100-class
    classifier" question ml_predictor_dataset.md's evaluation scope implies.

Usage:
  python eval_pretrained_onnx.py --arch resnet18 --variant torchvision_v1 \
      --label-map imagenet100_label_map.json --limit 5000
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from preprocessing_variants import VARIANTS, MODEL_PREPROCESS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def top_k_hits(logits, true_idx, k):
    return true_idx in np.argsort(logits)[-k:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model-dataset-dir", default=os.path.join(HERE, "model_dataset"))
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--limit", type=int, default=5000, help="5000 = full validation split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preprocess", default=None,
                    help="override preprocessing variant (see preprocessing_variants.py); "
                         "auto-selected from MODEL_PREPROCESS for known non-standard exports")
    args = ap.parse_args()

    preprocess_fn = preprocess
    variant_name = args.preprocess or MODEL_PREPROCESS.get((args.arch, args.variant))
    if variant_name:
        preprocess_fn = VARIANTS[variant_name]
        print(f"using non-standard preprocessing: {variant_name}", flush=True)

    onnx_path = os.path.join(args.model_dataset_dir, args.arch, args.variant, "model.onnx")
    label_map = json.load(open(args.label_map))
    local_to_1k = {int(k): v for k, v in label_map["local_to_imagenet1k"].items()}
    restricted_classes = sorted(local_to_1k.values())  # the 100 valid ImageNet-1K indices
    restricted_pos = {c: i for i, c in enumerate(restricted_classes)}  # 1k-idx -> 0..99 slot

    print(f"loading {onnx_path} ...", flush=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print("loading clane9/imagenet-100 validation split ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("clane9/imagenet-100", split="validation")
    if args.limit and args.limit < len(ds):
        ds = ds.select(range(args.limit))
    print(f"  {len(ds)} images", flush=True)

    n = len(ds)
    unrestricted_top1 = unrestricted_top5 = 0
    restricted_top1 = restricted_top5 = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        img = ex["image"]
        local_label = ex["label"]
        true_1k = local_to_1k[local_label]

        x = preprocess_fn(img)[None, ...].astype(np.float32)
        logits = sess.run(None, {input_name: x})[0].reshape(-1)

        if top_k_hits(logits, true_1k, 1):
            unrestricted_top1 += 1
        if top_k_hits(logits, true_1k, 5):
            unrestricted_top5 += 1

        restricted_logits = logits[restricted_classes]
        true_restricted = restricted_pos[true_1k]
        if top_k_hits(restricted_logits, true_restricted, 1):
            restricted_top1 += 1
        if top_k_hits(restricted_logits, true_restricted, 5):
            restricted_top5 += 1

        if (i + 1) % 500 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{n}] unrestricted top1={unrestricted_top1 / (i + 1) * 100:.2f}%  "
                  f"restricted top1={restricted_top1 / (i + 1) * 100:.2f}%  "
                  f"({elapsed:.1f}s elapsed)", flush=True)

    result = {
        "arch": args.arch, "variant": args.variant, "n_images": n,
        "unrestricted_top1": unrestricted_top1 / n, "unrestricted_top5": unrestricted_top5 / n,
        "restricted_top1": restricted_top1 / n, "restricted_top5": restricted_top5 / n,
    }
    print("\n=== result ===")
    print(f"  unrestricted (argmax over all 1000 classes): "
          f"top1={result['unrestricted_top1']*100:.2f}%  top5={result['unrestricted_top5']*100:.2f}%")
    print(f"  restricted   (argmax over the 100 eval classes): "
          f"top1={result['restricted_top1']*100:.2f}%  top5={result['restricted_top5']*100:.2f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
