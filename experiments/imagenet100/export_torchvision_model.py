#!/usr/bin/env python3
"""
Export a torchvision model with its ORIGINAL ImageNet-1K pretrained weights
straight to ONNX -- no fine-tuning, no custom head. Per ml_predictor_dataset.md
"Retraining: No", the 1000-way logits are evaluated on the ImageNet-100
subset later by filtering/mapping, not by retraining a 100-class head (unlike
the older experiments/imagenet100/export_imagenet100_{resnet18,mobilenetv2}
_onnx.py scripts, which loaded a fine-tuned 100-class checkpoint).

Usage:
  python export_torchvision_model.py --model resnet18 --weights IMAGENET1K_V1 \
      --out model_dataset/resnet18/torchvision_v1/model.onnx
"""
import argparse
import json
from pathlib import Path

import torch
import torchvision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="torchvision.models factory name, e.g. resnet50")
    ap.add_argument("--weights", required=True, help="weights enum member, e.g. IMAGENET1K_V2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    weights_enum = torchvision.models.get_model_weights(args.model)
    weights = getattr(weights_enum, args.weights)
    model = torchvision.models.get_model(args.model, weights=weights)
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamo=True, external_data=False,
    )

    meta = {
        "source": "torchvision",
        "torchvision_model": args.model,
        "torchvision_weights": args.weights,
        "num_classes": len(weights.meta.get("categories", [])) or None,
        "input_shape": [1, 3, args.img_size, args.img_size],
        "categories": weights.meta.get("categories"),
    }
    with open(out_path.with_name("metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_path} ({meta['num_classes']} classes, {args.img_size}x{args.img_size})")


if __name__ == "__main__":
    main()
