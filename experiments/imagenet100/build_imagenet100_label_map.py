#!/usr/bin/env python3
"""
clane9/imagenet-100 (the ml_predictor_dataset.md eval set) keeps its own
0-99 ClassLabel indexing, with each class name written as the FULL
ImageNet synset synonym list (e.g. "bonnet, poke bonnet"), while a
torchvision pretrained model's 1000-way output uses the single canonical
first-synonym name per class (e.g. "bonnet") in the model's own reference
categories list (saved as export_torchvision_model.py's metadata.json
"categories" field, e.g. model_dataset/resnet18/torchvision_v1/metadata.json).

Since no retraining happens (ml_predictor_dataset.md "Retraining: No"), a
pretrained model must be evaluated on imagenet-100 through THIS mapping:
each imagenet-100 local label maps to the ImageNet-1K class index the
model's 1000-way logits actually use.

Usage:
  python build_imagenet100_label_map.py \
      --categories-from model_dataset/resnet18/torchvision_v1/metadata.json \
      --out imagenet100_label_map.json
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories-from", required=True,
                    help="a model_dataset/*/metadata.json with a 1000-entry 'categories' list "
                         "(only torchvision exports currently save this)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    categories_1k = json.load(open(args.categories_from))["categories"]
    if len(categories_1k) != 1000:
        raise ValueError(f"{args.categories_from} categories has {len(categories_1k)} entries, expected 1000")
    idx_1k = {name: i for i, name in enumerate(categories_1k)}

    info_url = "https://datasets-server.huggingface.co/info?dataset=clane9/imagenet-100"
    with urllib.request.urlopen(info_url, timeout=30) as resp:
        info = json.load(resp)
    names_100 = info["dataset_info"]["default"]["features"]["label"]["names"]
    if len(names_100) != 100:
        raise ValueError(f"expected 100 imagenet-100 class names, got {len(names_100)}")

    mapping = {}
    unmatched = []
    for local_idx, full_name in enumerate(names_100):
        first = full_name.split(",")[0].strip()
        if first in idx_1k:
            mapping[local_idx] = idx_1k[first]
        else:
            unmatched.append((local_idx, full_name))

    if unmatched:
        raise RuntimeError(f"{len(unmatched)} imagenet-100 classes did not match any "
                           f"1000-class category name: {unmatched}")

    with open(args.out, "w") as f:
        json.dump({
            "local_to_imagenet1k": mapping,
            "imagenet100_names": names_100,
            "categories_source": args.categories_from,
        }, f, indent=2)
    print(f"wrote {args.out}: {len(mapping)}/100 imagenet-100 classes mapped to ImageNet-1K indices")


if __name__ == "__main__":
    main()
