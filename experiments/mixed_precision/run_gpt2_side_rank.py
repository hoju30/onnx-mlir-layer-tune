#!/usr/bin/env python3
"""GPT-2 small: the weight-only and input-only counterparts of run_gpt2_output_rank.py + run_gpt2_act_grad_rank.py.

For one projection at a time, int8 fake-quant of only its WEIGHT (the pass's per-tensor symmetric quantizer) or
only its INPUT activation (the pass's affine quantizer with the calibration range), measure dNLL on the test
windows, and compare how four predictors from the calibration windows rank that damage:
  rel_err     ||delta||^2 / ||t||^2
  s_rel_err   the layer's weight SNIP S_w times rel_err
  g_rel_err   mean|dL/dt| (t = W or x) times rel_err
  fisher      sum (dL/dt * delta)^2, gradient and error matched element by element
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

from eval_random_mixed_precision import np_fake_quant_weight, simulate_in_onnxruntime, supported_nodes  # noqa: E402
from run_gpt2_a8_side import simulate_int_act  # noqa: E402
from run_gpt2_fisher_milp import act_fake_quant  # noqa: E402
from run_gpt2_output_rank import token_nll  # noqa: E402

PREDICTORS = ("rel_err", "s_rel_err", "g_rel_err", "fisher")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--fmt", default="int8")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    mapping = json.loads((src / "node_mapping.json").read_text())
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    S = json.loads((src / "sensitivity.json").read_text())
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    x_name = {n["name"]: n["act_tensors"][0] for n in nodes}
    assert len(nodes) == 48

    # --- predictors on the calibration windows ---
    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            cap[name] = inp[0]
        return fn

    for n, p in mapping.items():
        mods[p.removesuffix(".weight")].register_forward_hook(hook(n))
    gw2 = {n: torch.zeros_like(params[p], dtype=torch.float64) for n, p in mapping.items()}
    gw1 = {n: torch.zeros_like(params[p], dtype=torch.float64) for n, p in mapping.items()}
    acc = {n: dict.fromkeys(("err2", "x2", "abs_g", "numel", "fisher"), 0.0) for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, x in cap.items():
                g = params[mapping[n]].grad.double()
                gw2[n] += g ** 2
                gw1[n] += g.abs()
                d, gx, a = (act_fake_quant(x, args.fmt, ranges[x_name[n]]) - x).double(), x.grad.double(), acc[n]
                a["err2"] += float((d ** 2).sum())
                a["x2"] += float((x.double() ** 2).sum())
                a["abs_g"] += float(gx.abs().sum())
                a["numel"] += x.numel()
                a["fisher"] += float(((gx * d) ** 2).sum())
        cap.clear()
        log(f"calibration window {k + 1}/{len(calib)}: CE {float(loss.detach()):.4f}")
    pred = {"weight": {}, "input": {}}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        dW = np_fake_quant_weight(W, args.fmt, None).astype(np.float64) - W.astype(np.float64)
        rel_w = float((dW ** 2).sum() / (W.astype(np.float64) ** 2).sum())
        pred["weight"][n] = {"rel_err": rel_w, "s_rel_err": S[n] * rel_w,
                             "g_rel_err": float(gw1[n].mean()) / len(calib) * rel_w,
                             "fisher": float((torch.from_numpy(dW ** 2) * gw2[n]).sum())}
        a = acc[n]
        rel_x = a["err2"] / a["x2"]
        pred["input"][n] = {"rel_err": rel_x, "s_rel_err": S[n] * rel_x,
                            "g_rel_err": a["abs_g"] / a["numel"] * rel_x, "fisher": a["fisher"]}
    del model, gw1, gw2
    log("predictors ready")

    # --- measured single-node dNLL on the test windows ---
    base = token_nll(simulate_in_onnxruntime(graph, nodes, {}, ranges, test, None), test)
    summary = {}
    for side in ("weight", "input"):
        rows = []
        for k, n in enumerate(nodes):
            name = n["name"]
            if side == "weight":
                z = simulate_in_onnxruntime(graph, nodes, {name: {"w": args.fmt, "a": "fp32"}}, ranges, test, None)
            else:
                z = simulate_int_act(graph, [n], int(args.fmt[3:]), ranges, test, True, False)
            d = token_nll(z, test) - base
            param = mapping[name]
            rows.append({"node": name, "block": int(param.split(".")[2]), "kind": ".".join(param.split(".")[3:5]),
                         "dnll": float(d.mean()), "dnll_se": float(d.std(ddof=1) / np.sqrt(d.size)), **pred[side][name]})
            if (k + 1) % 12 == 0:
                log(f"{side}-only: {k + 1}/48 nodes measured")
        dnll = np.array([r["dnll"] for r in rows])
        true_top = set(np.argsort(-dnll)[:args.top_k])
        stats = {}
        for key in PREDICTORS:
            p = np.array([r[key] for r in rows])
            stats[key] = {"spearman": float(spearmanr(p, dnll)[0]), "kendall": float(kendalltau(p, dnll)[0]),
                          f"top{args.top_k}_overlap": len(true_top & set(np.argsort(-p)[:args.top_k]))}
            for rank, i in enumerate(np.argsort(-p), 1):
                rows[i][f"rank_{key}"] = rank
        rng = np.random.default_rng(0)
        P = {key: np.array([r[key] for r in rows]) for key in PREDICTORS}
        diffs = [spearmanr(P["fisher"][i], dnll[i])[0] - spearmanr(P["rel_err"][i], dnll[i])[0]
                 for i in rng.integers(0, 48, (5000, 48))]
        stats["fisher_minus_rel_err_ci95"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
        significant = int(sum(r["dnll"] > 2 * r["dnll_se"] for r in rows))
        summary[side] = {"stats": stats, "sum_dnll": float(dnll.sum()), "nodes_above_2se": significant, "rows": rows}

        print(f"\n== {side}-only {args.fmt}: sum of single-node dNLL {dnll.sum():.4f}; "
              f"{significant}/48 nodes with dNLL > 2 SE")
        for key in PREDICTORS:
            s = stats[key]
            print(f"  {key:<10} Spearman {s['spearman']:+.2f}  Kendall {s['kendall']:+.2f}  "
                  f"top-{args.top_k} overlap {s[f'top{args.top_k}_overlap']}/{args.top_k}")
        lo, hi = stats["fisher_minus_rel_err_ci95"]
        print(f"  Spearman(fisher) - Spearman(rel_err) 95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  top {args.top_k} by measured dNLL (rank under each predictor):")
        for r in sorted(rows, key=lambda r: -r["dnll"])[:args.top_k]:
            print(f"    h.{r['block']:<2d} {r['kind']:<14} dNLL {r['dnll']:+.4f}±{r['dnll_se']:.4f}"
                  + "".join(f"  {key} {r['rank_' + key]:2d}" for key in PREDICTORS))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "format": args.fmt, "source_run": str(src),
         "fp32_nll": float(base.mean()), **summary}, indent=2) + "\n")
    log(f"wrote {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
