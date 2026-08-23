#!/usr/bin/env python3
"""
Download one official pretrained model from the ONNX Model Zoo
(github.com/onnx/models) verbatim -- these architectures (Inception V1/V2,
ShuffleNet V1/V2, SqueezeNet, EfficientNet-Lite4) have no directly matching
torchvision/timm ImageNet-1K checkpoint per ml_predictor_dataset.md, so the
zoo's own validated .onnx export is the source of truth instead of an
export_*_model.py re-export.

Validates the download is a real ONNX model (onnx.checker) rather than an
HTML error page or a Git LFS pointer stub, since both can silently succeed
as an HTTP 200.

Usage:
  python download_onnx_zoo_model.py \
      --url https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx \
      --out model_dataset/squeezenet/onnx_model_zoo/model.onnx
"""
import argparse
import json
from pathlib import Path

import onnx
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {args.url} ...", flush=True)
    with urllib.request.urlopen(args.url, timeout=120) as resp:
        data = resp.read()

    if data[:14].lstrip().startswith(b"<") or data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{args.url} did not return a raw ONNX file "
                            f"(got HTML or an LFS pointer -- first bytes: {data[:80]!r})")

    out_path.write_bytes(data)

    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)
    inp = model.graph.input[0]
    shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    out_dim = model.graph.output[0].type.tensor_type.shape.dim
    # the class axis isn't always last (e.g. SqueezeNet's conv classifier head
    # outputs [1, 1000, 1, 1], not a flattened [1, 1000]) -- take the largest
    # non-batch dim instead of assuming a fixed axis.
    non_batch_dims = [d.dim_value for d in out_dim[1:] if d.dim_value]
    num_classes = max(non_batch_dims) if non_batch_dims else None

    meta = {
        "source": "onnx_model_zoo",
        "zoo_url": args.url,
        "input_name": inp.name,
        "input_shape": shape,
        "num_classes": num_classes,
        "size_bytes": len(data),
    }
    with open(out_path.with_name("metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_path} ({len(data) / 1e6:.1f} MB, input {inp.name}{shape}, "
          f"{num_classes} classes)")


if __name__ == "__main__":
    main()
