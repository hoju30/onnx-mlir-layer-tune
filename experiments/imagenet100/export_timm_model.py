#!/usr/bin/env python3
"""
Export a timm model with its ORIGINAL ImageNet-1K pretrained checkpoint
straight to ONNX -- no fine-tuning. See export_torchvision_model.py's
docstring for why (ml_predictor_dataset.md "Retraining: No").

Input size is read from the checkpoint's own pretrained_cfg rather than
assumed, since timm tags (e.g. "resnet18.a1_in1k" vs ".a2_in1k") can use
different training/eval resolutions for the same architecture.

Usage:
  python export_timm_model.py --model resnet18.a1_in1k \
      --out model_dataset/resnet18/timm_a1/model.onnx
"""
import argparse
import json
from pathlib import Path

import timm
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="timm model tag, e.g. resnet50.a2_in1k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = timm.create_model(args.model, pretrained=True)
    model.eval()

    cfg = model.pretrained_cfg
    c, h, w = cfg.get("input_size", (3, 224, 224))

    dummy = torch.randn(1, c, h, w)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamo=True, external_data=False,
    )

    meta = {
        "source": "timm",
        "timm_model": args.model,
        "num_classes": getattr(model, "num_classes", None),
        "input_shape": [1, c, h, w],
        "pretrained_cfg": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, list, tuple))},
    }
    with open(out_path.with_name("metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_path} ({meta['num_classes']} classes, {c}x{h}x{w})")


if __name__ == "__main__":
    main()
