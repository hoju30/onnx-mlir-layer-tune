#!/usr/bin/env python3
"""GPT-2 small: replace the layer-wide weight SNIP with the gradient where the error happens.

For each of the 48 projections, int8 fake-quant of its OUTPUT y (the only thing quantized in
run_gpt2_output_rank.py) gives an error delta = Q(y) - y. On the source run's calibration windows (causal
next-token CE, labels are the text itself), rank the projections by
  rel_err        ||delta||^2 / ||y||^2                                   (baseline, as before)
  S_w * rel_err  times the layer's weight SNIP mean|W dL/dW|             (current criterion, as before)
  g_a * rel_err  times mean|dL/dy| instead: one number per layer, taken at the error's location
  fisher         sum (dL/dy * delta)^2: gradient and error matched element by element (diagonal Fisher,
                 second-order estimate of the loss change)
and compare each ranking with the dNLL measured per projection by run_gpt2_output_rank.py.
delta uses the same int8 affine fake-quant and calibration ranges as the ORT simulation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

from eval_random_mixed_precision import int_qparams, supported_nodes  # noqa: E402

PREDICTORS = {"rel_err": "relative output error", "s_rel_err": "S_w x relative error",
              "ga_rel_err": "mean|dL/dy| x relative error", "fisher": "sum (dL/dy * delta)^2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--rank-run", type=Path, default=HERE / "runs/gpt2_output_rank_20261001",
                    help="run_gpt2_output_rank.py output with the measured per-projection dNLL")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)
    src = args.source_run
    mapping = json.loads((src / "node_mapping.json").read_text())          # onnx node -> torch weight name
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    S = json.loads((src / "sensitivity.json").read_text())
    calib = np.load(src / "validation_tokens.npy")
    out_tensor = {n["name"]: n["act_tensors"][-1] for n in supported_nodes(onnx.load(str(src / "model.onnx")))
                  if n["name"] in mapping}
    measured = {r["node"]: r for r in json.loads((args.rank_run / "results.json").read_text())["rows"]}
    assert len(out_tensor) == len(measured) == 48

    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    modules = dict(model.named_modules())
    captured = {}

    def hook(name):
        def fn(mod, inp, out):
            out.retain_grad()
            captured[name] = out
        return fn

    for node, param in mapping.items():
        modules[param.removesuffix(".weight")].register_forward_hook(hook(node))
    qp = {n: int_qparams(*ranges[t], 8) for n, t in out_tensor.items()}
    acc = {n: dict.fromkeys(("err2", "y2", "abs_g", "numel", "fisher"), 0.0) for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        x = torch.from_numpy(ids[None])
        loss = model(x, labels=x, use_cache=False).loss                   # mean causal next-token CE
        loss.backward()
        with torch.no_grad():
            for n, y in captured.items():
                scale, zp = qp[n]
                q = (torch.clamp(torch.round(y / scale + zp), -128, 127) - zp) * scale
                d, g = (q - y).double(), y.grad.double()
                a = acc[n]
                a["err2"] += float((d ** 2).sum())
                a["y2"] += float((y.double() ** 2).sum())
                a["abs_g"] += float(g.abs().sum())
                a["numel"] += y.numel()
                a["fisher"] += float(((g * d) ** 2).sum())
        captured.clear()
        print(f"[{time.time() - t0:4.0f}s] calibration window {k + 1}/{len(calib)}: CE {float(loss):.4f}", flush=True)

    rows = []
    for n, a in acc.items():
        m = measured[n]
        rel = a["err2"] / a["y2"]
        rows.append({"node": n, "block": m["block"], "kind": m["kind"], "dnll": m["dnll"], "dnll_se": m["dnll_se"],
                     "rel_err": rel, "rel_err_table": m["rel_err"], "s_rel_err": S[n] * rel,
                     "grad_a": a["abs_g"] / a["numel"], "ga_rel_err": a["abs_g"] / a["numel"] * rel,
                     "fisher": a["fisher"]})
    dnll = np.array([r["dnll"] for r in rows])
    true_top = set(np.argsort(-dnll)[:args.top_k])
    stats = {}
    for key in PREDICTORS:
        pred = np.array([r[key] for r in rows])
        stats[key] = {"spearman": float(spearmanr(pred, dnll)[0]), "kendall": float(kendalltau(pred, dnll)[0]),
                      f"top{args.top_k}_overlap": len(true_top & set(np.argsort(-pred)[:args.top_k]))}
        for rank, i in enumerate(np.argsort(-pred), 1):
            rows[i][f"rank_{key}"] = rank
    for rank, i in enumerate(np.argsort(-dnll), 1):
        rows[i]["rank_dnll"] = rank
    spread = {k: float(max(r[k] for r in rows) / min(r[k] for r in rows)) for k in ("grad_a",)}
    spread["S_w"] = float(max(S[r["node"]] for r in rows) / min(S[r["node"]] for r in rows))
    check = float(spearmanr([r["rel_err"] for r in rows], [r["rel_err_table"] for r in rows])[0])
    (args.out_dir / "results.json").write_text(json.dumps(
        {"calibration_windows": len(calib), "rank_agreement": stats, "max_over_min": spread,
         "rel_err_vs_saved_table_spearman": check, "rows": rows}, indent=2) + "\n")

    print(f"\nrel_err recomputed on full tensors vs the saved (subsampled) table: Spearman {check:+.3f}")
    print(f"spread across the 48 layers (max/min): weight SNIP S_w {spread['S_w']:.1f}x, "
          f"mean|dL/dy| {spread['grad_a']:.1f}x")
    print(f"\nrank agreement with the measured dNLL (48 projections, output-only int8):")
    for key, name in PREDICTORS.items():
        s = stats[key]
        print(f"  {name:<30} Spearman {s['spearman']:+.2f}  Kendall {s['kendall']:+.2f}  "
              f"top-{args.top_k} overlap {s[f'top{args.top_k}_overlap']}/{args.top_k}")
    print(f"\ntop {args.top_k} by measured dNLL, rank under each predictor:")
    print(f"  {'':<22}{'dNLL':>9}" + "".join(f"{k:>12}" for k in PREDICTORS))
    for r in sorted(rows, key=lambda r: -r["dnll"])[:args.top_k]:
        print(f"  h.{r['block']:<2d} {r['kind']:<16}{r['dnll']:+9.4f}" + "".join(f"{r['rank_' + k]:>12d}" for k in PREDICTORS))


if __name__ == "__main__":
    main()
