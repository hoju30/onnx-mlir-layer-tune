#!/usr/bin/env python3
"""GPT-2 small: shared_v2 with the shared weight SNIP replaced by each quantized tensor's own SNIP.

HTQ's Eq. 5 with relative errors (shared_v2), per projection node and format:
  shared_v2   S_W * (r_W + r_X + r_Y)                    one connection sensitivity, the weight's
  split_v2    S_W * r_W + S_X * r_X + S_Y * r_Y          each tensor's own connection sensitivity
  split_abs   S_W * ||dW||^2 + S_X * E||dX||^2 + S_Y * E||dY||^2   (HTQ Eq. 5's absolute error, per tensor)
  fisher      mean(g^2) * ||delta||^2 per tensor         reference (run_gpt2_fisher_term_ablation.py's base)
with S_t = mean|t * dL/dt| (HTQ/SNIP multiplicative mask, per calibration window, averaged over windows) and
r_t = ||Q(t) - t||^2 / ||t||^2 on full tensors (t = W, projection input X, projection output Y; a= quantizes X
and Y). MILP at budgets relative to the projections' all-W8A8; ORT QDQ simulation on the test windows.
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
from transformers import GPT2LMHeadModel  # noqa: E402

from eval_random_mixed_precision import np_fake_quant_weight, simulate_in_onnxruntime, supported_nodes  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_gpt2_fisher_milp import FORMATS, act_fake_quant, token_nll  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

VARIANTS = ("shared_v2", "split_v2", "split_abs", "fisher")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
    ap.add_argument("--milp-time-limit", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(f) for f in args.budgets.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    mapping = json.loads((src / "node_mapping.json").read_text())
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    assert len(nodes) == 48
    seq, K = calib.shape[1], len(calib)

    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, p in mapping.items():
        mods[p.removesuffix(".weight")].register_forward_hook(hook(n))
    acc = {n: {"conn_w": 0.0, "g2w": 0.0,
               **{s: {"conn": 0.0, "g2": 0.0, "count": 0, "t2": 0.0, "d2": dict.fromkeys(FORMATS, 0.0)}
                  for s in ("x", "y")}} for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                W = params[mapping[n]]
                g = W.grad.double()
                a = acc[n]
                a["conn_w"] += float((W.double() * g).abs().mean()) / K
                a["g2w"] += float((g ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    s, gt, td = a[side], t.grad.double(), t.double()
                    s["conn"] += float((td * gt).abs().mean()) / K
                    s["g2"] += float((gt ** 2).sum())
                    s["count"] += t.numel()
                    s["t2"] += float((td ** 2).sum())
                    for f in FORMATS[:-1]:
                        s["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
        log(f"calibration window {k + 1}/{K}: CE {float(loss.detach()):.4f}")

    saved_S = json.loads((src / "sensitivity.json").read_text())
    r, F, S, E, elems, flops = {}, {}, {}, {}, {}, {}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        Wd = W.astype(np.float64)
        dW2 = {f: float(((np_fake_quant_weight(W, f, None).astype(np.float64) - Wd) ** 2).sum()) for f in FORMATS[:-1]}
        a = acc[n]
        S[n] = {"w": a["conn_w"], "x": a["x"]["conn"], "y": a["y"]["conn"]}
        r[n] = {"w": {"fp32": 0.0, **{f: v / float((Wd ** 2).sum()) for f, v in dW2.items()}},
                **{s: {"fp32": 0.0, **{f: a[s]["d2"][f] / a[s]["t2"] for f in FORMATS[:-1]}} for s in ("x", "y")}}
        # absolute squared error (HTQ Eq. 5 / Fisher's error term); activations per calibration sample
        E[n] = {"w": {"fp32": 0.0, **dW2},
                **{s: {"fp32": 0.0, **{f: a[s]["d2"][f] / K for f in FORMATS[:-1]}} for s in ("x", "y")}}
        F[n] = {"w": {"fp32": 0.0, **{f: a["g2w"] / W.size * v for f, v in dW2.items()}},
                **{s: {"fp32": 0.0, **{f: a[s]["g2"] / a[s]["count"] * a[s]["d2"][f] for f in FORMATS[:-1]}}
                   for s in ("x", "y")}}
        elems[n], flops[n] = W.size, W.size * seq
    del model
    sw = np.array([S[n]["w"] for n in mapping])
    log(f"S_W vs the source run's saved weight SNIP: max relative difference "
        f"{max(abs(S[n]['w'] / saved_S[n] - 1) for n in mapping):.1e}")
    ratio = {s: np.array([S[n][s] for n in mapping]) / sw for s in ("x", "y")}
    terms = {v: {t: sum((S[n]["w"] if v == "shared_v2" else S[n][t]) * r[n][t]["int4"] for n in mapping)
                 for t in ("w", "x", "y")} for v in ("shared_v2", "split_v2")}
    terms["split_abs"] = {t: sum(S[n][t] * E[n][t]["int4"] for n in mapping) for t in ("w", "x", "y")}
    print(f"\nS_X/S_W median {np.median(ratio['x']):.2f} (range {ratio['x'].min():.2f}-{ratio['x'].max():.2f}), "
          f"S_Y/S_W median {np.median(ratio['y']):.2f} (range {ratio['y'].min():.2f}-{ratio['y'].max():.2f})")
    for v, t in terms.items():
        print(f"{v:<10} summed terms at int4: W {t['w']:.3g}  X {t['x']:.3g}  Y {t['y']:.3g}  "
              f"(X/W {t['x'] / t['w']:.2f}, Y/W {t['y'] / t['w']:.2f})")

    def value_fn(v):
        if v == "fisher":
            return lambda n, w, a: F[n]["w"][w] + F[n]["x"][a] + F[n]["y"][a]
        if v == "shared_v2":
            return lambda n, w, a: S[n]["w"] * (r[n]["w"][w] + r[n]["x"][a] + r[n]["y"][a])
        if v == "split_abs":
            return lambda n, w, a: S[n]["w"] * E[n]["w"][w] + S[n]["x"] * E[n]["x"][a] + S[n]["y"] * E[n]["y"][a]
        return lambda n, w, a: S[n]["w"] * r[n]["w"][w] + S[n]["x"] * r[n]["x"][a] + S[n]["y"] * r[n]["y"][a]

    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    nll, rows = {}, []
    for f in budgets:
        for v in VARIANTS:
            cfg, obj, status = milp_select(nodes, FORMATS, FORMATS, value_fn(v), elems, flops, f * ref_size,
                                           f * ref_bops, time_limit=args.milp_time_limit)
            assert status in (0, 1), (v, f, status)
            size, bops = config_cost(cfg, elems, flops)
            z = simulate_in_onnxruntime(graph, nodes, cfg, ranges, test, None)
            assert np.isfinite(z).all(), (v, f)
            nll[(f, v)] = token_nll(z, test)
            m = float(nll[(f, v)].mean())
            wq = [c["w"] for c in cfg.values() if c["w"]]
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / ref_size,
                         "bops": bops / ref_bops, "nll": m, "perplexity": float(np.exp(m)),
                         "top1": float((z[:, :-1, :].argmax(-1) == test[:, 1:]).mean()),
                         "int4_weights": wq.count("int4"), "f16_activations": sum(c["a"] == "f16" for c in cfg.values()),
                         "formats": cfg})
            log(f"f={f:.2f} {v:<10} PPL {np.exp(m):9.2f}" + ("" if status == 0 else "  (MILP time limit)"))
            del z
    for row in rows:
        d = nll[(row["budget"], row["variant"])] - nll[(row["budget"], "shared_v2")]
        row["dnll_vs_shared_v2"], row["dnll_vs_shared_v2_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "budgets": budgets,
         "S": S, "term_balance_int4": terms, "rows": rows}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<11} {'PPL':>9} {'top1':>7} {'dNLL vs shared_v2':>19} {'size':>5} {'BOPs':>5} "
          f"{'int4 W':>7} {'f16 A':>6}")
    for row in rows:
        d = "" if row["variant"] == "shared_v2" else f"{row['dnll_vs_shared_v2']:+.4f}±{row['dnll_vs_shared_v2_se']:.4f}"
        print(f"{row['budget']:5.2f} {row['variant'] + ('' if row['milp_optimal'] else '*'):<11} "
              f"{row['perplexity']:9.2f} {row['top1']:7.2%} {d:>19} {row['size']:5.2f} {row['bops']:5.2f} "
              f"{row['int4_weights']:7d} {row['f16_activations']:6d}")


if __name__ == "__main__":
    main()
