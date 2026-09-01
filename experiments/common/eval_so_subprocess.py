#!/usr/bin/env python3
"""Run a single compiled .so's inference over a fixed image-index list and
print the restricted-top1 accuracy as JSON. Invoked as its own subprocess
by build_dataset_lowp_general.py -- NOT imported -- because
PyRuntime.OMExecutionSession leaks native memory per model loaded (confirmed:
~100-450MB per .so, never reclaimed even after the Python object is deleted).
Looping over 41 configs' .so files inside one long-lived process accumulates
this into tens of GB and eventually triggers the kernel OOM killer (observed
directly in dmesg, killing this project's own python3 process at 26-28GB
RSS). Isolating each config's eval in its own process means the OS reclaims
everything the instant this process exits, regardless of what PyRuntime
leaks internally.
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--image-indices", required=True, help="comma-separated indices into the imagenet-100 validation split")
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--preprocess", default=None,
                    help="override preprocessing variant (see preprocessing_variants.py)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(PROJECT, "build/Release/lib"))
    sys.path.insert(0, os.path.join(PROJECT, "experiments/imagenet100"))
    from PyRuntime import OMExecutionSession
    from eval_pretrained_onnx import preprocess as default_preprocess
    from datasets import load_dataset

    preprocess = default_preprocess
    if args.preprocess:
        from preprocessing_variants import VARIANTS
        preprocess = VARIANTS[args.preprocess]

    label_map = json.load(open(args.label_map))
    local_to_1k = {int(k): v for k, v in label_map["local_to_imagenet1k"].items()}
    restricted_classes = sorted(local_to_1k.values())
    restricted_pos = {c: i for i, c in enumerate(restricted_classes)}

    ds = load_dataset("clane9/imagenet-100", split="validation")
    indices = [int(i) for i in args.image_indices.split(",")]

    sess = OMExecutionSession(args.so, args.tag)
    correct = 0
    for i in indices:
        ex = ds[i]
        true_1k = local_to_1k[ex["label"]]
        x = preprocess(ex["image"])[None, ...].astype(np.float32)
        logits = sess.run([x])[0][0]
        pred = logits[restricted_classes].argmax()
        correct += (pred == restricted_pos[true_1k])

    print(json.dumps({"top1": correct / len(indices), "n": len(indices)}))


if __name__ == "__main__":
    main()