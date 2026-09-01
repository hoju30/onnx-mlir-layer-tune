#!/usr/bin/env python3
"""Complete-configuration generator for the low-precision variant on models
too large to enumerate exhaustively (claude_lowprecision.md section 5.1),
e.g. ResNet18 (46 tunable nodes) or MobileNetV2 (63 tunable nodes) -- unlike
sample_configs_lowp.py's exhaustive product over MNIST's 3 tunable nodes,
these node counts make exhaustive enumeration impossible (6^21-ish for
ResNet18 alone), so this mirrors the original Posit sample_configs.py's
stratified random sampling + representative configurations instead.

Tunable nodes and their candidate formats come from lowp_format_scope.py,
i.e. docs/LowPrecisionFormats.md's per-op scope table -- NOT the
"supported" flag already present in *_op_features.json (that flag reflects
Posit-era eligibility from a different pass and does not apply here; e.g.
resnet18_op_features.json marks Relu unsupported even though Relu is
low-precision-tunable, and MobileNetV2's 35 Clip nodes have no
low-precision candidates at all despite being real activation ops).

Usage:
  python sample_configs_lowp_stratified.py \
      --op-features experiments/imagenet100/model/resnet18_op_features.json \
      --n-random 40 --out experiments/imagenet100/model/resnet18_lowp_configs.json
"""
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowp_format_scope import FORMATS, candidate_formats, is_tunable

NON_FP32_FORMATS = FORMATS[1:]  # bf16, f16, int8, fp8e4m3, fp8e5m2


def tunable_nodes(op_features):
    ops = op_features["ops"]
    return [(o["name"], o["type"]) for o in ops if is_tunable(o["type"])]


def most_aggressive(op_type):
    """Smallest-bitwidth format legal for this op_type: int8/fp8 (8-bit) for
    full-scope ops, bf16 (16-bit, since narrow-scope ops have no 8-bit
    option) for narrow-scope ops."""
    cands = candidate_formats(op_type)
    return "int8" if "int8" in cands else "bf16"


def representative_configs(nodes):
    names_types = nodes
    names = [n for n, _ in names_types]
    configs = [{"id": "rep_all_fp32", "strategy": "representative:all_fp32",
                "formats": {n: "FP32" for n in names}}]

    for fmt in NON_FP32_FORMATS:
        formats = {n: (fmt if fmt in candidate_formats(t) else "FP32") for n, t in names_types}
        configs.append({"id": f"rep_uniform_{fmt}", "strategy": f"representative:uniform_{fmt}",
                         "formats": formats})

    half = len(names) // 2 or 1
    configs.append({"id": "rep_front_only", "strategy": "representative:front_only",
                     "formats": {n: (most_aggressive(t) if i < half else "FP32")
                                 for i, (n, t) in enumerate(names_types)}})
    configs.append({"id": "rep_back_only", "strategy": "representative:back_only",
                     "formats": {n: (most_aggressive(t) if i >= len(names_types) - half else "FP32")
                                 for i, (n, t) in enumerate(names_types)}})
    configs.append({"id": "rep_alternating", "strategy": "representative:alternating",
                     "formats": {n: (most_aggressive(t) if i % 2 == 0 else "FP32")
                                 for i, (n, t) in enumerate(names_types)}})
    return configs


def stratified_random_configs(nodes, n_samples, seed):
    """Strata over quantized-node ratio (section 5.1); within each stratum a
    random subset of tunable nodes is chosen and each gets an independently
    random format FROM ITS OWN candidate set (so Conv/Gemm can land on
    int8/fp8 while Relu/Add only ever land on bf16/f16), so both the
    quantized-ratio and the format-mix vary across samples at the same ratio."""
    rng = random.Random(seed)
    names_types = nodes
    n_nodes = len(names_types)
    strata = [0.0, 0.25, 0.5, 0.75, 1.0]
    configs = []
    for i in range(n_samples):
        ratio = strata[i % len(strata)]
        k = round(ratio * n_nodes)
        chosen_idx = set(rng.sample(range(n_nodes), k)) if k > 0 else set()
        formats = {}
        for idx, (n, t) in enumerate(names_types):
            if idx in chosen_idx:
                formats[n] = rng.choice(candidate_formats(t)[1:])  # exclude FP32
            else:
                formats[n] = "FP32"
        configs.append({"id": f"rand_{i:04d}",
                         "strategy": f"stratified_random:ratio={ratio}",
                         "formats": formats})
    return configs


def dedupe(configs):
    seen = set()
    out, n_dropped = [], 0
    for c in configs:
        key = tuple(sorted(c["formats"].items()))
        if key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        out.append(c)
    return out, n_dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    op_features = json.load(open(args.op_features))
    nodes = tunable_nodes(op_features)

    configs = representative_configs(nodes)
    configs += stratified_random_configs(nodes, args.n_random, args.seed)
    configs, n_dropped = dedupe(configs)

    with open(args.out, "w") as f:
        json.dump({
            "tunable_nodes": [{"name": n, "type": t, "candidate_formats": candidate_formats(t)}
                              for n, t in nodes],
            "configs": configs,
        }, f, indent=2)
    print(f"{len(nodes)} tunable nodes; wrote {len(configs)} configs "
          f"({n_dropped} duplicate(s) dropped) to {args.out}")


if __name__ == "__main__":
    main()
