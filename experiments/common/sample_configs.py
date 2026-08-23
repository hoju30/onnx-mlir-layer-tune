#!/usr/bin/env python3
"""
Complete-configuration generator (claude.md section 5.1): stratified random
sampling + representative configurations over every `supported` node in an
op_features.json. This is the "Configuration Generator" box in
graph_description.md before an accuracy predictor exists to drive
surrogate-assisted NSGA-II (section 5.2) -- its only job is to produce
diverse COMPLETE per-node format assignments for build_dataset.py to
real-evaluate and turn into training samples.

Usage:
  python sample_configs.py --op-features docs/mnist_example/mnist_op_features.json \
      --risk-calibration docs/mnist_example/mnist_risk_calibration.json \
      --n-random 24 --out docs/mnist_example/mnist_sampled_configs.json
"""
import argparse
import json
import random

FORMATS = ["posit_8_1", "posit_16_1", "posit_32_1"]  # FP32 is the implicit default/fallback


def supported_nodes(op_features):
    return [(op["name"], op["type"]) for op in op_features["ops"] if op["supported"]]


def representative_configs(names):
    configs = [{"id": "rep_all_fp32", "strategy": "representative:all_fp32",
                "formats": {n: "FP32" for n in names}}]

    for fmt in FORMATS:
        tag = fmt.replace("posit_", "p").replace("_", "e")
        configs.append({"id": f"rep_uniform_{tag}", "strategy": f"representative:uniform_{fmt}",
                         "formats": {n: fmt for n in names}})

    half = len(names) // 2 or 1
    configs.append({"id": "rep_front_only", "strategy": "representative:front_only",
                     "formats": {n: ("posit_8_1" if i < half else "FP32")
                                 for i, n in enumerate(names)}})
    configs.append({"id": "rep_back_only", "strategy": "representative:back_only",
                     "formats": {n: ("posit_8_1" if i >= len(names) - half else "FP32")
                                 for i, n in enumerate(names)}})
    configs.append({"id": "rep_alternating", "strategy": "representative:alternating",
                     "formats": {n: (FORMATS[i % len(FORMATS)] if i % 2 == 0 else "FP32")
                                 for i, n in enumerate(names)}})
    return configs


def sensitivity_guided_configs(nodes, risk):
    """Least/most sensitive-by-op-type nodes (mnist_risk_calibration.json) get
    the aggressive posit8 format first -- a 'safe' and a 'stress' extreme."""
    if risk is None:
        return []
    sens = risk["sensitivity"]
    ranked = sorted(nodes, key=lambda nt: sens.get(nt[1], 0.0))
    names_by_sens = [n for n, _ in ranked]
    n = len(names_by_sens)
    k = max(1, n // 2)

    safe = {name: ("posit_8_1" if i < k else "FP32") for i, name in enumerate(names_by_sens)}
    aggressive = {name: ("posit_8_1" if i >= n - k else "FP32") for i, name in enumerate(names_by_sens)}
    return [
        {"id": "rep_sensitivity_guided_safe", "strategy": "representative:sensitivity_guided_safe",
         "formats": safe},
        {"id": "rep_sensitivity_guided_aggressive", "strategy": "representative:sensitivity_guided_aggressive",
         "formats": aggressive},
    ]


def stratified_random_configs(names, n_samples, seed):
    """Strata over quantized-node ratio (section 5.1); within each stratum a
    random subset of nodes is chosen and each gets an independently random
    format, so both the quantized-ratio and the format-mix vary across
    samples at the same ratio."""
    rng = random.Random(seed)
    n_nodes = len(names)
    strata = [0.0, 0.25, 0.5, 0.75, 1.0]
    configs = []
    for i in range(n_samples):
        ratio = strata[i % len(strata)]
        k = round(ratio * n_nodes)
        chosen = set(rng.sample(names, k)) if k > 0 else set()
        formats = {n: (rng.choice(FORMATS) if n in chosen else "FP32") for n in names}
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
    ap.add_argument("--risk-calibration", default=None)
    ap.add_argument("--n-random", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    op_features = json.load(open(args.op_features))
    nodes = supported_nodes(op_features)
    names = [n for n, _ in nodes]

    risk = json.load(open(args.risk_calibration)) if args.risk_calibration else None

    configs = representative_configs(names)
    configs += sensitivity_guided_configs(nodes, risk)
    configs += stratified_random_configs(names, args.n_random, args.seed)
    configs, n_dropped = dedupe(configs)

    with open(args.out, "w") as f:
        json.dump({"nodes": [{"name": n, "type": t} for n, t in nodes],
                   "formats_available": FORMATS + ["FP32"],
                   "configs": configs}, f, indent=2)
    print(f"wrote {len(configs)} configs ({n_dropped} duplicate(s) dropped) to {args.out}")


if __name__ == "__main__":
    main()
