#!/usr/bin/env python3
"""Standalone calibration/verification tool: compute FP32 activation
statistics for each of the MNIST demo net's 6 canonical nodes and, via
--verify-against-test, confirm the shared numpy forward pass (mnist_model.py)
matches the real compiled model's measured accuracy.

Note: build_posit_graph.py no longer reads this script's --out JSON directly
-- it now calls mnist_model.forward() itself and applies per-node
quantization-aware stats live (round-tripping through the node's own
assigned posit format when applicable). This script remains useful as a
standalone sanity check and for producing a format-independent reference.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_posit_graph import decode_all_constants
from mnist_model import forward, load_mnist, value_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight-source-ir", default="docs/mnist_example/pp/m.mixed.posit.mlir")
    ap.add_argument("--num-calibration", type=int, default=512)
    ap.add_argument("--out", default="docs/mnist_example/mnist_activation_stats.json")
    ap.add_argument("--verify-against-test", action="store_true",
                     help="also run the full MNIST test set and compare top1 vs mnist_whole_model_measurements.json's fp32 baseline")
    args = ap.parse_args()

    arrays = decode_all_constants(args.weight_source_ir)
    fc1_w, fc1_b, fc2_w, fc2_b = arrays[0], arrays[1], arrays[2], arrays[3]

    print(f"loading {args.num_calibration} MNIST calibration images (train split) ...")
    images, _ = load_mnist(args.num_calibration, train=True)
    activations = forward(images, fc1_w, fc1_b, fc2_w, fc2_b)
    stats = {name: value_stats(act, "activation") for name, act in activations.items()}

    with open(args.out, "w") as f:
        json.dump({"num_calibration": args.num_calibration, "source": "MNIST train split", "nodes": stats}, f, indent=2)
    print(f"wrote {args.out}")
    for name, s in stats.items():
        print(f"  {name}: mean={s['activation_mean']:.4f} std={s['activation_std']:.4f} "
              f"zero_ratio={s['activation_zero_ratio']:.3f} dyn_range={s['activation_dynamic_range']:.4f}")

    if args.verify_against_test:
        print("\nverifying forward-pass reimplementation against the real measured FP32 baseline ...")
        test_images, test_labels = load_mnist(10000, train=False)
        test_activations = forward(test_images, fc1_w, fc1_b, fc2_w, fc2_b)
        preds = test_activations["Softmax_6"].argmax(axis=1)
        top1 = float((preds == test_labels).mean())
        with open("docs/mnist_example/mnist_whole_model_measurements.json") as f:
            measured = json.load(f)
        real_top1 = measured["fp32_baseline"]["top1"]
        print(f"  numpy reimplementation top1={top1*100:.2f}%  vs  real measured fp32 top1={real_top1*100:.2f}%")
        if abs(top1 - real_top1) > 0.002:
            print("  WARNING: mismatch exceeds 0.2pp -- forward-pass reimplementation may be wrong")
        else:
            print("  OK: matches within 0.2pp")


if __name__ == "__main__":
    main()
