#!/usr/bin/env python3
"""MobileNetV2: is the per-node perturbation SUM (what every criterion + MILP assumes) a fair model of the damage?

A. Operating points: for the per-tensor Fisher MILP configs at f = 0.60..0.80 (run_mbv2_fisher_term_ablation.py's
   base), compare the real held-out loss increase of the whole config with the sum of its nodes' single-node
   increases (each node quantized alone with its chosen formats).
B. Pairwise interactions: on 12 nodes (pw -> dw -> pwl of 4 inverted-residual blocks), all at w=int6/a=int6,
   I_ij = dL(i,j) - dL(i) - dL(j) for all 66 pairs, grouped into adjacent / same-block / different-block pairs.
Everything in ORT QDQ simulation on the same held-out images (per-image paired differences).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eval_random_mixed_precision import (calibrate_ranges, preprocess_by_source, simulate_in_onnxruntime,  # noqa: E402
                                         supported_nodes)

BLOCK_STARTS = (3, 18, 33, 48)   # weighted-node index of the pw conv of 4 inverted-residual blocks (pw, dw, pwl)
PAIR_FMT = {"w": "int6", "a": "int6"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--configs-from", type=Path, default=HERE / "runs/mbv2_fisher_term_ablation_20261002/results.json")
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]][:args.num_eval]
    cx = preprocess_by_source(args.images_dir, {"images": calib}, meta)
    ex, ey = preprocess_by_source(args.images_dir, {"images": held}, meta), np.array([e["label"] for e in held])
    all_nodes = supported_nodes(graph)
    weighted = [n["name"] for n in all_nodes if n.get("weight_const")]
    ranges = calibrate_ranges(graph, all_nodes, cx)

    def loss(fmts):
        z = simulate_in_onnxruntime(graph, all_nodes, fmts, ranges, ex, None)
        assert np.isfinite(z).all()
        return F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey), reduction="none").numpy()

    def mean_se(d):
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    base = loss({})
    single_cache = {}

    def single(name, fmt):
        key = (name, fmt["w"], fmt["a"])
        if key not in single_cache:
            single_cache[key] = loss({name: fmt}) - base
        return single_cache[key]

    # --- A. operating points ---
    rows_a = []
    configs = [r for r in json.loads(args.configs_from.read_text())["rows"] if r["variant"] == "base"]
    for r in configs:
        cfg = {n: v for n, v in r["formats"].items() if (v["w"] not in (None, "fp32")) or v["a"] != "fp32"}
        joint = loss(cfg) - base
        singles = sum(single(n, v) for n, v in cfg.items())
        j, jse = mean_se(joint)
        s, sse = mean_se(singles)
        g, gse = mean_se(joint - singles)
        rows_a.append({"budget": r["budget"], "quantized_nodes": len(cfg), "joint_dloss": j, "joint_se": jse,
                       "sum_single_dloss": s, "sum_single_se": sse, "interaction": g, "interaction_se": gse})
        log(f"A f={r['budget']:.2f}: joint dloss {j:.3f}±{jse:.3f}  sum of {len(cfg)} singles {s:.3f}±{sse:.3f}")

    # --- B. pairwise interactions ---
    idx = [b + k for b in BLOCK_STARTS for k in range(3)]
    names = [weighted[i] for i in idx]
    d1 = {n: single(n, PAIR_FMT) for n in names}
    rows_b = []
    for (i, a), (j, b) in itertools.combinations(list(zip(idx, names)), 2):
        dij = loss({a: PAIR_FMT, b: PAIR_FMT}) - base
        inter = dij - d1[a] - d1[b]
        same_block = (i - 3) // 3 == (j - 3) // 3
        kind = ("adjacent" if j - i == 1 else "same block") if same_block else "different block"
        m, se = mean_se(inter)
        rows_b.append({"i": i, "j": j, "kind": kind, "dloss_pair": float(dij.mean()), "dloss_i": float(d1[a].mean()),
                       "dloss_j": float(d1[b].mean()), "interaction": m, "interaction_se": se})
    log(f"B: {len(rows_b)} pairs measured")

    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "num_eval": len(ey), "pair_format": PAIR_FMT, "pair_node_index": idx,
         "operating_points": rows_a, "pairs": rows_b}, indent=2) + "\n")

    print("\nA. whole config vs sum of single-node losses (Fisher MILP configs):")
    print(f"{'f':>5} {'nodes':>6} {'joint dloss':>14} {'sum singles':>14} {'joint/sum':>10} {'interaction':>16}")
    for r in rows_a:
        print(f"{r['budget']:5.2f} {r['quantized_nodes']:6d} {r['joint_dloss']:8.3f}±{r['joint_se']:.3f} "
              f"{r['sum_single_dloss']:8.3f}±{r['sum_single_se']:.3f} {r['joint_dloss'] / r['sum_single_dloss']:10.2f} "
              f"{r['interaction']:+9.3f}±{r['interaction_se']:.3f}")
    print(f"\nB. pairwise interactions, {len(names)} nodes at w=int6/a=int6 (single-node dloss: "
          + ", ".join(f"#{i} {d1[n].mean():.3f}" for i, n in zip(idx, names)) + ")")
    for kind in ("adjacent", "same block", "different block"):
        g = [r for r in rows_b if r["kind"] == kind]
        inter = np.array([r["interaction"] for r in g])
        rel = np.array([r["interaction"] / (r["dloss_i"] + r["dloss_j"]) for r in g if r["dloss_i"] + r["dloss_j"] > 0])
        sig = sum(abs(r["interaction"]) > 2 * r["interaction_se"] for r in g)
        print(f"  {kind:<16} {len(g):2d} pairs: mean interaction {inter.mean():+.4f}, mean |interaction| "
              f"{np.abs(inter).mean():.4f}, median interaction/(dL_i+dL_j) {np.median(rel):+.2f}, "
              f"{sig} pairs with |I| > 2 SE")
    top = sorted(rows_b, key=lambda r: -abs(r["interaction"]))[:5]
    print("  largest |interaction|: " + "; ".join(
        f"#{r['i']}-#{r['j']} ({r['kind']}) {r['interaction']:+.3f}±{r['interaction_se']:.3f} "
        f"vs singles {r['dloss_i']:.3f}+{r['dloss_j']:.3f}" for r in top))


if __name__ == "__main__":
    main()
