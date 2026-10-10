#!/usr/bin/env python3
"""GPT-2 small: a diagonal-Fisher criterion for weights AND activations, selected by MILP and compared with the
shared criteria at equal budgets.

Per projection node and format, on the source run's calibration windows (causal next-token CE):
  fisher     w: sum (dL/dW * dW)^2, dW = the pass's per-tensor symmetric weight fake-quant error
             a: sum (dL/dx * dx)^2 + sum (dL/dy * dy)^2, the projection's input x and output y (a= quantizes both)
  shared_v2  S_w * (||dW||^2/||W||^2 + ||dx||^2/||x||^2 + ||dy||^2/||y||^2), recomputed on full tensors
             (the saved table's 200k-sample relative errors break on GPT-2's massive activations)
  shared     the saved raw-L2 table (run_gpt2_criteria.py), unchanged
Budgets are fractions of the 48 projections' all-W8A8 weight bytes and BOPs; f > 1 leaves room for f16/bf16
activations on the nodes that need them. Evaluated in ORT QDQ simulation on the source run's test windows.
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
from scipy.special import logsumexp  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

from eval_random_mixed_precision import (int_qparams, np_fake_quant_weight, simulate_in_onnxruntime,  # noqa: E402
                                         supported_nodes)
from mp_milp import milp_select  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

FORMATS = ["int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32"]
CRITERIA = ("shared", "shared_v2", "fisher")


def act_fake_quant(t, fmt, lohi):
    if fmt == "fp32":
        return t
    if fmt == "f16":
        return t.half().float()
    if fmt == "bf16":
        return t.bfloat16().float()
    bits = int(fmt[3:])
    scale, zp = int_qparams(*lohi, bits)
    return (torch.clamp(torch.round(t / scale + zp), -(1 << (bits - 1)), (1 << (bits - 1)) - 1) - zp) * scale


def token_nll(logits, tokens):
    pred, target = logits[:, :-1, :].astype(np.float64), tokens[:, 1:]
    return np.concatenate([logsumexp(s, axis=-1) - s[np.arange(len(t)), t] for s, t in zip(pred, target)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
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
    S = json.loads((src / "sensitivity.json").read_text())
    saved = json.loads((src / "perturbation_tables.json").read_text())["shared"]
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    assert len(nodes) == 48
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    seq = calib.shape[1]

    # --- gradients and errors on the calibration windows ---
    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods = dict(model.named_modules())
    params = dict(model.named_parameters())
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, p in mapping.items():
        mods[p.removesuffix(".weight")].register_forward_hook(hook(n))
    g2w = {n: torch.zeros_like(params[p], dtype=torch.float64) for n, p in mapping.items()}
    fa = {n: dict.fromkeys(FORMATS, 0.0) for n in mapping}                    # Fisher activation term
    ra = {n: {"x": dict.fromkeys(FORMATS, 0.0), "y": dict.fromkeys(FORMATS, 0.0), "px": 0.0, "py": 0.0}
          for n in mapping}                                                  # relative-error sums
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g2w[n] += params[mapping[n]].grad.double() ** 2
                r = ra[n]
                r["px"] += float((x.double() ** 2).sum())
                r["py"] += float((y.double() ** 2).sum())
                for f in FORMATS[:-1]:
                    dx = (act_fake_quant(x, f, ranges[tens[n][0]]) - x).double()
                    dy = (act_fake_quant(y, f, ranges[tens[n][1]]) - y).double()
                    fa[n][f] += float(((x.grad.double() * dx) ** 2).sum() + ((y.grad.double() * dy) ** 2).sum())
                    r["x"][f] += float((dx ** 2).sum())
                    r["y"][f] += float((dy ** 2).sum())
        cap.clear()
        log(f"calibration window {k + 1}/{len(calib)}: CE {float(loss.detach()):.4f}")

    fw, rw, elems, flops = {}, {}, {}, {}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        Wd = W.astype(np.float64)
        fw[n], rw[n] = {"fp32": 0.0}, {"fp32": 0.0}
        for f in FORMATS[:-1]:
            dW = np_fake_quant_weight(W, f, None).astype(np.float64) - Wd
            fw[n][f] = float((torch.from_numpy(dW ** 2) * g2w[n]).sum())
            rw[n][f] = float((dW ** 2).sum() / (Wd ** 2).sum())
        fa[n]["fp32"] = 0.0
        elems[n], flops[n] = W.size, W.size * seq
    del g2w, model
    log("Fisher and full-tensor relative-error tables ready")

    def value_fn(c):
        if c == "shared":
            return lambda n, w, a: S[n] * (saved[n]["w"][w] + saved[n]["a"][a])
        if c == "shared_v2":
            def v(n, w, a):
                r = ra[n]
                return S[n] * (rw[n][w] + (r["x"][a] / r["px"] if a != "fp32" else 0.0)
                               + (r["y"][a] / r["py"] if a != "fp32" else 0.0))
            return v
        return lambda n, w, a: fw[n][w] + fa[n][a]

    # --- MILP selection and ORT evaluation ---
    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    configs = [{"id": "fp32", "criterion": "FP32", "formats": {}},
               {"id": "w8a8", "criterion": "W8A8", "formats": {n["name"]: {"w": "int8", "a": "int8"} for n in nodes}}]
    for i, f in enumerate(budgets):
        for c in CRITERIA:
            cfg, obj, status = milp_select(nodes, FORMATS, FORMATS, value_fn(c), elems, flops, f * ref_size,
                                           f * ref_bops)
            assert status == 0, (c, f, status)
            size, bops = config_cost(cfg, elems, flops)
            configs.append({"id": f"{c}_{i:03d}", "criterion": c, "budget": f, "formats": cfg,
                            "size": size / ref_size, "bops": bops / ref_bops})
    nll = {}
    rows = []
    for c in configs:
        z = simulate_in_onnxruntime(graph, nodes, c["formats"], ranges, test, None)
        assert np.isfinite(z).all(), c["id"]
        nll[c["id"]] = token_nll(z, test)
        top1 = float((z[:, :-1, :].argmax(-1) == test[:, 1:]).mean())
        m = float(nll[c["id"]].mean())
        a_count = {}
        for v in c["formats"].values():
            a_count[v["a"]] = a_count.get(v["a"], 0) + 1
        rows.append({k: c[k] for k in ("id", "criterion", "budget", "size", "bops") if k in c} |
                    {"nll": m, "perplexity": float(np.exp(m)), "top1": top1, "act_formats": a_count})
        log(f"{c['id']:<14} PPL {np.exp(m):9.2f}  top1 {top1:.2%}")
        del z
    for r in rows:
        if "budget" in r and r["criterion"] != "fisher":
            d = nll[r["id"]] - nll[f"fisher_{budgets.index(r['budget']):03d}"]
            r["dnll_vs_fisher"], r["dnll_vs_fisher_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "budgets": budgets, "rows": rows,
         "configs": [{k: v for k, v in c.items() if k != "formats"} | {"formats": c["formats"]} for c in configs]},
        indent=2) + "\n")

    fp32, w8a8 = rows[0], rows[1]
    print(f"\nFP32 PPL {fp32['perplexity']:.2f} top1 {fp32['top1']:.2%} | W8A8 PPL {w8a8['perplexity']:.2f} "
          f"top1 {w8a8['top1']:.2%}   (budgets: fractions of the projections' all-W8A8 bytes and BOPs)")
    print(f"{'f':>5} {'criterion':<10} {'size':>5} {'BOPs':>5} {'PPL':>9} {'top1':>7}   {'dNLL vs fisher':>17}   act formats")
    for r in rows[2:]:
        d = f"{r['dnll_vs_fisher']:+.4f}±{r['dnll_vs_fisher_se']:.4f}" if "dnll_vs_fisher" in r else ""
        acts = ",".join(f"{k}:{v}" for k, v in sorted(r["act_formats"].items()))
        print(f"{r['budget']:5.2f} {r['criterion']:<10} {r['size']:5.2f} {r['bops']:5.2f} {r['perplexity']:9.2f} "
              f"{r['top1']:7.2%}   {d:>17}   {acts}")


if __name__ == "__main__":
    main()
