#!/usr/bin/env python3
"""GPT-2 small: quantize ONE projection's output to int8 at a time (weights and inputs FP32), measure dNLL on
the test windows, and ask how well two per-node predictors rank that damage:
  rel_err       relative output error ||y - Q(y)||^2 / ||y||^2  (run_gpt2_criteria.py's relative_wo 'a' int8)
  S * rel_err   the same times the node's SNIP connection sensitivity (that run's sensitivity.json)
Both predictors come from the source run's calibration windows; dNLL is measured on its test windows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from scipy.special import logsumexp
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_random_mixed_precision import supported_nodes  # noqa: E402
from run_gpt2_a8_side import simulate_int_act  # noqa: E402


def token_nll(logits, tokens):
    pred, target = logits[:, :-1, :].astype(np.float64), tokens[:, 1:]
    return np.concatenate([logsumexp(s, axis=-1) - s[np.arange(len(t)), t] for s, t in zip(pred, target)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    tokens = np.load(src / "test_tokens.npy")
    mapping = json.loads((src / "node_mapping.json").read_text())
    S = json.loads((src / "sensitivity.json").read_text())
    rel = {n: v["a"]["int8"] for n, v in json.loads((src / "perturbation_tables.json").read_text())["relative_wo"].items()}
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    assert len(nodes) == 48, len(nodes)

    base = token_nll(simulate_int_act(graph, nodes, 8, ranges, tokens, False, False), tokens)
    print(f"[{time.time() - t0:5.0f}s] FP32 NLL {base.mean():.4f} over {base.size} tokens", flush=True)
    rows = []
    for k, n in enumerate(nodes):
        d = token_nll(simulate_int_act(graph, [n], 8, ranges, tokens, False, True), tokens) - base
        param = mapping[n["name"]]                              # transformer.h.<i>.<attn|mlp>.<c_attn|c_proj|c_fc>.weight
        block, kind = int(param.split(".")[2]), ".".join(param.split(".")[3:5])
        rows.append({"node": n["name"], "block": block, "kind": kind, "dnll": float(d.mean()),
                     "dnll_se": float(d.std(ddof=1) / np.sqrt(d.size)), "rel_err": rel[n["name"]],
                     "sensitivity": S[n["name"]], "s_rel_err": S[n["name"]] * rel[n["name"]]})
        print(f"[{time.time() - t0:5.0f}s] {k + 1:2d}/48 h.{block:<2d} {kind:<14} dNLL {d.mean():+.4f}", flush=True)

    dnll = np.array([r["dnll"] for r in rows])
    stats = {}
    by_true = set(np.argsort(-dnll)[:args.top_k])
    for key in ("rel_err", "s_rel_err"):
        pred = np.array([r[key] for r in rows])
        stats[key] = {"spearman": float(spearmanr(pred, dnll)[0]), "kendall": float(kendalltau(pred, dnll)[0]),
                      f"top{args.top_k}_overlap": len(by_true & set(np.argsort(-pred)[:args.top_k]))}
    for key in ("rel_err", "s_rel_err", "dnll"):
        order = np.argsort(-np.array([r[key] for r in rows]))
        for rank, i in enumerate(order, 1):
            rows[i][f"rank_{key}"] = rank
    sum_dnll = float(dnll.sum())
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "fp32_nll": float(base.mean()),
         "tokens": int(base.size), "sum_single_node_dnll": sum_dnll, "rank_agreement": stats, "rows": rows},
        indent=2) + "\n")

    print(f"\nsum of the 48 single-node dNLL: {sum_dnll:.4f}")
    for key, name in (("rel_err", "relative output error"), ("s_rel_err", "relative output error x S")):
        s = stats[key]
        print(f"{name:<28} Spearman {s['spearman']:+.2f}  Kendall {s['kendall']:+.2f}  "
              f"top-{args.top_k} overlap {s[f'top{args.top_k}_overlap']}/{args.top_k}")
    print(f"\ntop {args.top_k} by measured dNLL (rank under each predictor):")
    for r in sorted(rows, key=lambda r: -r["dnll"])[:args.top_k]:
        print(f"  h.{r['block']:<2d} {r['kind']:<14} dNLL {r['dnll']:+.4f} ± {r['dnll_se']:.4f}   "
              f"rel_err rank {r['rank_rel_err']:2d}   S*rel_err rank {r['rank_s_rel_err']:2d}")


if __name__ == "__main__":
    main()
