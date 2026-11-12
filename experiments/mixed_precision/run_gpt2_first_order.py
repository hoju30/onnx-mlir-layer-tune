#!/usr/bin/env python3
"""GPT-2 small: gradient-based scores for the MILP criterion, with weight W, input x and output y terms.
  2nd_tensor   mean(g^2) * sum delta^2        (run_gpt2_fisher_term_ablation.py's base)
  1st_tensor   mean|g| * sum |delta|          (same structure, first order)
  1st_elem     sum |g * delta|                (first-order Taylor magnitude, element-wise)
  abs_g_sq_error mean|g| * sum delta^2        (fixed-error sensitivity exponent diagnostic)
These are surrogates, not exact loss changes; abs_g_sq_error does not preserve Fisher's scaling properties.
g is dL/dt of each calibration window's mean causal next-token CE; |g| and g^2 are accumulated over windows.
delta is the pass's
fake-quant error of t. MILP at budgets relative to the projections' all-W8A8; ORT QDQ simulation on test windows.
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

VARIANTS = ("2nd_tensor", "1st_tensor", "1st_elem")
SENSITIVITY_VARIANTS = ("abs_g_sq_error", "abs_w_sq_error", "abs_x_sq_error", "abs_y_sq_error")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
    ap.add_argument("--milp-time-limit", type=float, default=60.0)
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="Comma-separated variants, including 2nd_tensor as reference. abs_g_sq_error "
                         "uses mean|g| with the SAME squared errors; abs_w/x/y_sq_error changes only that term.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    variants = args.variants.split(",")
    if "2nd_tensor" not in variants or any(v not in VARIANTS + SENSITIVITY_VARIANTS for v in variants):
        ap.error("--variants must include 2nd_tensor and only supported variant names")
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
    seq = calib.shape[1]

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
    w_abs = {n: torch.zeros_like(params[p], dtype=torch.float64) for n, p in mapping.items()}   # sum_windows |g|
    w_sq = dict.fromkeys(mapping, 0.0)
    act = {n: {s: {"g1": 0.0, "g2": 0.0, "count": 0, "d1": dict.fromkeys(FORMATS, 0.0),
                   "d2": dict.fromkeys(FORMATS, 0.0), "e1": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")}
           for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g = params[mapping[n]].grad.double()
                w_abs[n] += g.abs()
                w_sq[n] += float((g ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a, gt = act[n][side], t.grad.double()
                    a["g1"] += float(gt.abs().sum())
                    a["g2"] += float((gt ** 2).sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        d = (act_fake_quant(t, f, ranges[rng_name]) - t).double()
                        a["d1"][f] += float(d.abs().sum())
                        a["d2"][f] += float((d ** 2).sum())
                        a["e1"][f] += float((gt * d).abs().sum())
        cap.clear()
        log(f"calibration window {k + 1}/{len(calib)}: CE {float(loss.detach()):.4f}")

    table_variants = VARIANTS + ("abs_g_sq_error",)
    T = {v: {"w": {}, "x": {}, "y": {}} for v in table_variants}
    elems, flops = {}, {}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        size = W.size
        for v in table_variants:
            T[v]["w"][n] = {"fp32": 0.0}
        for f in FORMATS[:-1]:
            dW = torch.from_numpy(np_fake_quant_weight(W, f, None).astype(np.float64) - W.astype(np.float64))
            T["2nd_tensor"]["w"][n][f] = w_sq[n] / size * float((dW ** 2).sum())
            T["1st_tensor"]["w"][n][f] = float(w_abs[n].sum()) / size * float(dW.abs().sum())
            T["1st_elem"]["w"][n][f] = float((w_abs[n] * dW.abs()).sum())
            T["abs_g_sq_error"]["w"][n][f] = float(w_abs[n].sum()) / size * float((dW ** 2).sum())
        for side in ("x", "y"):
            a = act[n][side]
            T["2nd_tensor"][side][n] = {"fp32": 0.0, **{f: a["g2"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
            T["1st_tensor"][side][n] = {"fp32": 0.0, **{f: a["g1"] / a["count"] * a["d1"][f] for f in FORMATS[:-1]}}
            T["1st_elem"][side][n] = {"fp32": 0.0, **{f: a["e1"][f] for f in FORMATS[:-1]}}
            T["abs_g_sq_error"][side][n] = {"fp32": 0.0, **{f: a["g1"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
        elems[n], flops[n] = size, size * seq
    for side in ("w", "x", "y"):
        T[f"abs_{side}_sq_error"] = {t: T["abs_g_sq_error" if t == side else "2nd_tensor"][t]
                                      for t in ("w", "x", "y")}
    # Save the exact objectives for later budget searches and allocation diagnostics.
    # |g| * delta^2 is a controlled exponent ablation, NOT a Fisher/loss-unit approximation.
    (args.out_dir / "criterion_tables.json").write_text(json.dumps(T) + "\n")
    del model, w_abs
    log("first- and second-order tables ready")

    def value_fn(v):
        t = T[v]
        return lambda n, w, a: t["w"][n][w] + t["x"][n][a] + t["y"][n][a]

    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    nll, rows = {}, []
    for f in budgets:
        for v in variants:
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
            np.save(args.out_dir / f"nll_f{f:g}_{v}.npy", nll[(f, v)])
            log(f"f={f:.2f} {v:<10} PPL {np.exp(m):9.2f}" + ("" if status == 0 else "  (MILP time limit)"))
            del z
    for r in rows:
        d = nll[(r["budget"], r["variant"])] - nll[(r["budget"], "2nd_tensor")]
        r["dnll_vs_2nd"], r["dnll_vs_2nd_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "budgets": budgets,
         "variants": variants, "calibration_windows": len(calib), "test_windows": len(test),
         "sequence_length": seq, "milp_time_limit": args.milp_time_limit, "rows": rows}, indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<11} {'PPL':>9} {'top1':>7} {'dNLL vs 2nd':>17} {'size':>5} {'BOPs':>5} "
          f"{'int4 W':>7} {'f16 A':>6}")
    for r in rows:
        d = "" if r["variant"] == "2nd_tensor" else f"{r['dnll_vs_2nd']:+.4f}±{r['dnll_vs_2nd_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<11} {r['perplexity']:9.2f} "
              f"{r['top1']:7.2%} {d:>17} {r['size']:5.2f} {r['bops']:5.2f} {r['int4_weights']:7d} {r['f16_activations']:6d}")


if __name__ == "__main__":
    main()
