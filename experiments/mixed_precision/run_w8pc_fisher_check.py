#!/usr/bin/env python3
"""Per-channel weight quantization in the scoring flow: does the Fisher weight term predict the new weight damage?

Weights int4..int8 are quantized with one scale per output channel (run_w8_per_channel_check.py's arithmetic, axis 0);
activations stay FP32. Per node and weight format, with dW = Q_pc(W) - W and g = dCE/dW on the calibration images
(PyTorch, BatchNorm folded):
  A  mean(g^2) * ||dW||^2                         per-tensor sensitivity (the current weight term)
  B  sum_o mean_o(g^2) * ||dW_o||^2               per-output-channel sensitivity
  E  sum g^2 * dW^2                               element-wise
  SQNR  ||logits - logits_fp32||^2 on the calibration images with only that node's weights quantized (reference)
Ground truth: held-out loss increase with only that node's weights quantized (per channel) at int4, int5, int6
(ORT QDQ simulation). Reported: all-pairs Spearman, IQR of log10(pred / dloss) (the calibration that decides MILP
picks), per-format Spearman, per-kind bias, and how well the int4 / int6 damage ratio is predicted.
"""
import argparse
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
from scipy.stats import spearmanr  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_criteria_cost_compare import match_modules, torch_model  # noqa: E402
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402

TRUTH_BITS = ("int4", "int5", "int6")
VARIANTS = ("A", "B", "E", "SQNR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--num-truth", type=int, default=300)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    held = [e for e in man["images"] if not e["calibration"]][:args.num_truth]
    cy, ey = np.array([e["label"] for e in calib]), np.array([e["label"] for e in held])
    original = erm.np_fake_quant_weight
    erm.np_fake_quant_weight = lambda w, fmt, oracle: (per_channel_int(w, int(fmt[3:])) if fmt.startswith("int")
                                                        else original(w, fmt, oracle))
    summary = {}
    for m in args.models.split(","):
        tag = m.split("/")[0]
        md = HERE / "../imagenet100/model_dataset" / m
        graph = onnx.load(str(md / "model.onnx"))
        meta = json.loads((md / "metadata.json").read_text())
        nodes = erm.supported_nodes(graph)
        weighted = [n for n in nodes if n.get("weight_const")]
        names = [n["name"] for n in weighted]
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta)
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ranges = erm.calibrate_ranges(graph, nodes, cx)

        # --- Fisher weight terms ---
        model, layers = torch_model(meta)
        module_of, inits, by_name = match_modules(graph, weighted, layers)
        g2 = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
        for k in range(len(cy)):
            z = model(torch.from_numpy(cx[k:k + 1]))
            grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])),
                                        [module_of[n].weight for n in names])
            for n, g in zip(names, grads):
                g2[n] += g.double() ** 2
        kind = {n: ("fc" if module_of[n].weight.dim() == 2 else "dw" if module_of[n].groups > 1
                    else "pw" if module_of[n].kernel_size == (1, 1) else "conv") for n in names}
        pred = {v: {} for v in VARIANTS}
        for n in names:
            W = inits[by_name[n].input[1]]
            G = g2[n].numpy()
            assert G.shape == W.shape, (n, G.shape, W.shape)
            Go = G.reshape(G.shape[0], -1)
            for b in TRUTH_BITS:
                d2 = ((per_channel_int(W, int(b[3:])).astype(np.float64) - W) ** 2).reshape(W.shape[0], -1)
                pred["A"][(n, b)] = float(Go.mean() * d2.sum())
                pred["B"][(n, b)] = float((Go.mean(1) * d2.sum(1)).sum())
                pred["E"][(n, b)] = float((Go * d2).sum())
        log(f"{tag}: Fisher weight terms ready ({len(names)} nodes)")
        ref_cal = erm.simulate_in_onnxruntime(graph, nodes, {}, ranges, cx, None).astype(np.float64)
        for n in names:
            for b in TRUTH_BITS:
                zc = erm.simulate_in_onnxruntime(graph, nodes, {n: {"w": b, "a": "fp32"}}, ranges, cx, None)
                pred["SQNR"][(n, b)] = float(((zc - ref_cal) ** 2).sum())
        log(f"{tag}: SQNR (weights only, per channel) ready")

        # --- ground truth ---
        tpath = args.out_dir / f"truth_{tag}.json"
        if tpath.exists():
            truth = json.loads(tpath.read_text())["rows"]
        else:
            def loss(cfg):
                z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, ex, None)
                return F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey), reduction="none").numpy()
            base = loss({})
            truth = []
            for i, n in enumerate(names):
                for b in TRUTH_BITS:
                    d = loss({n: {"w": b, "a": "fp32"}}) - base
                    truth.append({"node": n, "w": b, "dloss": float(d.mean()),
                                  "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
                if (i + 1) % 20 == 0:
                    log(f"{tag}: truth {i + 1}/{len(names)} nodes")
            tpath.write_text(json.dumps({"num_images": len(ey), "weights": "per-channel", "rows": truth}, indent=1) + "\n")

        # --- agreement ---
        y = np.array([r["dloss"] for r in truth])
        sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in truth])
        T = {(r["node"], r["w"]): r for r in truth}
        stats = {}
        for v in VARIANTS:
            p = np.array([pred[v][(r["node"], r["w"])] for r in truth])
            lr = np.log10(np.clip(p[sig], 1e-300, None) / y[sig])
            cen = lr - np.median(lr)
            srows = [r for r, s in zip(truth, sig) if s]
            # int4 / int6 damage ratio, predicted vs actual, on nodes significant at both
            both = [n for n in names if T[(n, "int4")]["dloss"] > 2 * T[(n, "int4")]["dloss_se"]
                    and T[(n, "int6")]["dloss"] > 2 * T[(n, "int6")]["dloss_se"]]
            ratio_err = [np.log10((pred[v][(n, "int4")] / pred[v][(n, "int6")])
                                  / (T[(n, "int4")]["dloss"] / T[(n, "int6")]["dloss"])) for n in both]
            stats[v] = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                        "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                        "by_bits": {b: float(spearmanr(p[ix], y[ix])[0]) for b in TRUTH_BITS
                                    for ix in [np.array([r["w"] == b for r in truth])]},
                        "kind_bias": {k: float(np.median(cen[[kind[r["node"]] == k for r in srows]]))
                                      for k in ("conv", "pw", "dw", "fc") if any(kind[r["node"]] == k for r in srows)},
                        "int4_over_int6": {"nodes": len(both),
                                           "median_log10_error": float(np.median(ratio_err)) if both else None}}
        summary[tag] = {"nodes": len(names), "pairs": len(truth), "significant": int(sig.sum()), "stats": stats,
                        "dloss_by_bits": {b: float(np.sum([T[(n, b)]["dloss"] for n in names])) for b in TRUTH_BITS}}
        print(f"\n=== {m}: {len(names)} nodes x {TRUTH_BITS} per-channel weights (activations FP32), "
              f"{len(ey)} held-out images; {sig.sum()}/{len(truth)} pairs with dloss > 2 SE; summed single-node dloss "
              + ", ".join(f"{b} {summary[tag]['dloss_by_bits'][b]:.3f}" for b in TRUTH_BITS))
        print(f"   {'':<5}{'Spearman':>9}{'sig':>7}{'IQR':>6}   " + " ".join(f"{b:>6}" for b in TRUTH_BITS)
              + "   int4/int6 ratio err (log10, n)   per-kind bias")
        for v, s in stats.items():
            r46 = s["int4_over_int6"]
            print(f"   {v:<5}{s['spearman']:9.3f}{s['spearman_sig']:7.3f}{s['iqr']:6.2f}   "
                  + " ".join(f"{s['by_bits'][b]:6.3f}" for b in TRUTH_BITS)
                  + f"   {('%+.2f' % r46['median_log10_error']) if r46['nodes'] else '  n/a'} ({r46['nodes']:2d})"
                  + "                    " + " ".join(f"{k} {x:+.2f}" for k, x in s["kind_bias"].items()))
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
