#!/usr/bin/env python3
"""GPT-2 small, W8A8 split: where does the projections' W8A8 loss come from?

On the 48 transformer projection nodes of an existing run_gpt2_criteria.py run (same ONNX model, calibration
ranges and test windows), evaluate in ORT QDQ simulation:
  W only   w=int8, a=fp32
  A only   w=fp32, a=int8   (a= quantizes each projection's input AND output, as the pass does)
  W + A    w=int8, a=int8   (must reproduce that run's "W8A8 (projections)" row)
Everything else (embeddings, tied LM head, LayerNorm, attention MatMuls) stays FP32, as in that run.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_random_mixed_precision import simulate_in_onnxruntime, supported_nodes  # noqa: E402

VARIANTS = {"FP32": None, "W only (W8)": ("int8", "fp32"), "A only (A8)": ("fp32", "int8"),
            "W + A (W8A8)": ("int8", "int8")}


def score(logits, tokens):
    """NLL / perplexity / next-token top-1 exactly as run_gpt2_criteria.py computes them."""
    pred, target = logits[:, :-1, :], tokens[:, 1:]
    nll = float(np.concatenate([logsumexp(s.astype(np.float64), axis=-1) - s.astype(np.float64)[np.arange(len(t)), t]
                                for s, t in zip(pred, target)]).mean())
    return nll, math.exp(nll), float((pred.argmax(-1) == target).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    tokens = np.load(src / "test_tokens.npy")
    mapping = json.loads((src / "node_mapping.json").read_text())
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    assert len(nodes) == 48, len(nodes)

    rows = []
    for label, wa in VARIANTS.items():
        formats = {} if wa is None else {n["name"]: {"w": wa[0], "a": wa[1]} for n in nodes}
        z = simulate_in_onnxruntime(graph, nodes, formats, ranges, tokens, None)
        assert np.isfinite(z).all(), label
        nll, ppl, top1 = score(z, tokens)
        rows.append({"variant": label, "nll": nll, "perplexity": ppl, "top1": top1})
        print(f"[{time.time() - t0:5.1f}s] {label:<14} NLL {nll:.4f}  PPL {ppl:8.2f}  next-token top1 {top1:.2%}",
              flush=True)
        del z

    prev = next(r for r in json.loads((src / "results.json").read_text())["rows"] if r["id"] == "w8a8")
    assert abs(prev["perplexity"] - rows[-1]["perplexity"]) < 1e-6 * prev["perplexity"], (prev, rows[-1])
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "predicted_tokens": int(tokens[:, 1:].size),
         "rows": rows, "w8a8_matches_source_run": True}, indent=2) + "\n")
    print(f"W + A reproduces the source run's W8A8 PPL ({prev['perplexity']:.4f}); wrote {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
