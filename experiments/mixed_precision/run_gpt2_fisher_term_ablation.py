#!/usr/bin/env python3
"""GPT-2 small: does each term of the per-tensor Fisher criterion carry useful information for the MILP?

Base criterion, per projection node n and format f (the "all scalar" variant of run_gpt2_fisher_ablation.py):
  T_w = mean(dL/dW^2) * ||Q_f(W) - W||^2
  T_x = mean(dL/dx^2) * sum ||Q_f(x) - x||^2      (projection input, over the calibration windows)
  T_y = mean(dL/dy^2) * sum ||Q_f(y) - y||^2      (projection output; a= quantizes both x and y)
Ablations of one term t in {w, x, y}:
  drop_t   the criterion ignores t entirely (for w, the MILP then sees weight quantization as free)
  avg_t    t keeps its size but loses its per-layer information: T_t[n][f] = (sum_m T_t[m][f] / sum_m numel_m)
           * numel_n, i.e. every element of every layer counts the same
Selected by MILP at the same budgets, evaluated in ORT QDQ simulation on the test windows.
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

VARIANTS = ("base", "drop_w", "drop_x", "drop_y", "avg_w", "avg_x", "avg_y")


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
    g2w = dict.fromkeys(mapping, 0.0)
    act = {n: {s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")} for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g2w[n] += float((params[mapping[n]].grad.double() ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a = act[n][side]
                    a["g2"] += float((t.grad.double() ** 2).sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        a["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
        log(f"calibration window {k + 1}/{len(calib)}: CE {float(loss.detach()):.4f}")

    T = {"w": {}, "x": {}, "y": {}}
    numel = {"w": {}, "x": {}, "y": {}}
    elems, flops = {}, {}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        mean_g2w = g2w[n] / W.size                          # per element, summed over the calibration windows
        T["w"][n] = {"fp32": 0.0, **{f: mean_g2w * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                             - W.astype(np.float64)) ** 2).sum())
                                     for f in FORMATS[:-1]}}
        numel["w"][n] = W.size
        for side in ("x", "y"):
            a = act[n][side]
            T[side][n] = {"fp32": 0.0, **{f: a["g2"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
            numel[side][n] = a["count"]
        elems[n], flops[n] = W.size, W.size * seq
    del model
    avg = {}
    for t in ("w", "x", "y"):
        total = sum(numel[t].values())
        avg[t] = {n: {f: sum(T[t][m][f] for m in mapping) / total * numel[t][n] for f in FORMATS} for n in mapping}
    log("per-tensor Fisher tables ready")

    def value_fn(variant):
        kind, _, term = variant.partition("_")
        tables = {t: (avg[t] if kind == "avg" and t == term else T[t]) for t in ("w", "x", "y")}
        use = {t: not (kind == "drop" and t == term) for t in ("w", "x", "y")}
        return lambda n, w, a: ((tables["w"][n][w] if use["w"] else 0.0) + (tables["x"][n][a] if use["x"] else 0.0)
                                + (tables["y"][n][a] if use["y"] else 0.0))

    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    nll, rows = {}, []
    for f in budgets:
        for v in VARIANTS:
            cfg, obj, status = milp_select(nodes, FORMATS, FORMATS, value_fn(v), elems, flops, f * ref_size, f * ref_bops)
            assert status == 0, (v, f, status)
            size, bops = config_cost(cfg, elems, flops)
            z = simulate_in_onnxruntime(graph, nodes, cfg, ranges, test, None)
            assert np.isfinite(z).all(), (v, f)
            nll[(f, v)] = token_nll(z, test)
            m = float(nll[(f, v)].mean())
            wq = [c["w"] for c in cfg.values() if c["w"]]
            rows.append({"budget": f, "variant": v, "size": size / ref_size, "bops": bops / ref_bops, "nll": m,
                         "perplexity": float(np.exp(m)), "top1": float((z[:, :-1, :].argmax(-1) == test[:, 1:]).mean()),
                         "int4_weights": wq.count("int4"), "f16_activations": sum(c["a"] == "f16" for c in cfg.values()),
                         "formats": cfg})
            log(f"f={f:.2f} {v:<7} PPL {np.exp(m):9.2f}")
            del z
    for r in rows:
        d = nll[(r["budget"], r["variant"])] - nll[(r["budget"], "base")]
        r["dnll_vs_base"], r["dnll_vs_base_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "budgets": budgets, "rows": rows}, indent=2) + "\n")

    print(f"\n{'f':>5} {'variant':<8} {'PPL':>9} {'top1':>7} {'dNLL vs base':>17} {'size':>5} {'BOPs':>5} "
          f"{'int4 W':>7} {'f16 A':>6}")
    for r in rows:
        d = "" if r["variant"] == "base" else f"{r['dnll_vs_base']:+.4f}±{r['dnll_vs_base_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant']:<8} {r['perplexity']:9.2f} {r['top1']:7.2%} {d:>17} "
              f"{r['size']:5.2f} {r['bops']:5.2f} {r['int4_weights']:7d} {r['f16_activations']:6d}")


if __name__ == "__main__":
    main()
