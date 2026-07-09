#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


def parse_shape(shape: str):
    dims = [int(x) for x in shape.lower().replace(",", "x").split("x") if x]
    if len(dims) != 4 or dims[0] != 1 or dims[1] != 3:
        raise ValueError(f"expected NCHW shape 1x3xHxW, got {shape}")
    return dims


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocess one ImageNet-style RGB image into NCHW float text."
    )
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--output", required=True, help="output text tensor path")
    ap.add_argument("--shape", default="1x3x224x224", help="NCHW shape")
    ap.add_argument("--resize-short", type=int, default=256)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    args = ap.parse_args()

    dims = parse_shape(args.shape)
    if dims[2] != args.crop_size or dims[3] != args.crop_size:
        raise ValueError(
            f"--shape {args.shape} must match --crop-size {args.crop_size}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize(args.resize_short),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    tensor = transform(img).numpy().astype(np.float32, copy=False)
    tensor = np.expand_dims(tensor, axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, tensor.reshape(-1), fmt="%.9g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
