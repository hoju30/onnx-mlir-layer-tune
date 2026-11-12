#!/usr/bin/env python3
"""GPT-2 small: C (channel bias-aware Fisher) vs SQNR, as run_criteria_cost_compare.py does for the CNNs.

48 projection nodes (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) of runs/gpt2_shared_ablation_20260930 (its
ONNX model, activation ranges, 10 WikiText-2 validation windows for calibration, 32 test windows for evaluation).
A channel is a hidden feature (last dimension): C's mean error mu_c is per feature over the window's tokens, and its
coherent term sums the gradient over the tokens. Loss: causal next-token CE.
  A     per-tensor Fisher (weight and activation terms mean(g^2) * sum delta^2)
  C     per-feature residual + coherent per-feature mean (activations); weight term as A
  SQNR  sum ||logits - logits_fp32||^2 with only that node quantized (sqnr_tables.py, ORT)
Cost: each table in its own child process, wall time and RSS sampled while it is built. Prediction: single-node
test NLL increase at w/a = int4/int8, int8/int4, int6/int6 (SE over windows). MILP: budgets as fractions of the
projections' all-W8A8 weight bytes and BOPs (f > 1 leaves room for f16/bf16 activations), ORT on the test windows.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

from run_criteria_cost_compare import RssSampler  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS  # noqa: E402

CONFIGS = (("int4", "int8"), ("int8", "int4"), ("int6", "int6"))
AF = FORMATS[:-1]


def load_source(src):
    import onnx
    from eval_random_mixed_precision import supported_nodes
    graph = onnx.load(str(src / "model.onnx"))
    mapping = json.loads((src / "node_mapping.json").read_text())
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    assert len(nodes) == 48
    return graph, mapping, ranges, calib, test, nodes


def phase_fisher(args):
    import torch
    from transformers import GPT2LMHeadModel
    from eval_random_mixed_precision import np_fake_quant_weight
    from run_mbv2_fisher_term_ablation import act_fake_quant
    graph, mapping, ranges, calib, _, nodes = load_source(args.source_run)
    names = [n["name"] for n in nodes]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    module_of = {n: mods[mapping[n].removesuffix(".weight")] for n in names}

    def csum(t):   # per feature (last dimension)
        return t.sum(dim=tuple(range(t.dim() - 1)))

    sampler = RssSampler().__enter__()
    cap, seen = {}, {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2w = dict.fromkeys(names, 0.0)
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for ids in calib:
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(loss, targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += float((grads[3 * i].double() ** 2).sum())
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                    lo, hi = seen.get((n, j), (float("inf"), float("-inf")))
                    seen[(n, j)] = (min(lo, float(t.min())), max(hi, float(t.max())))
                    a["g2"] = a["g2"] + csum(g ** 2)
                    a["gc"] = a["gc"] + csum(g) ** 2          # one window: sum over its tokens, then square
                    a["n"] += t.numel() // t.shape[-1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + csum(d)
                        a["s2"][f] = a["s2"][f] + csum(d ** 2)
        cap.clear()
    tables = {"CW": {}, "CA_A": {}, "CA_C": {}}
    for n in names:
        W = params[mapping[n]].detach().numpy()
        tables["CW"][n] = {"fp32": 0.0, **{f: g2w[n] / W.size * float(((np_fake_quant_weight(W, f, None).astype(
            np.float64) - W.astype(np.float64)) ** 2).sum()) for f in AF}}
        tables["CA_A"][n], tables["CA_C"][n] = {"fp32": 0.0}, {"fp32": 0.0}
        for f in AF:
            a_cost = c_cost = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                mu = a["s1"][f] / a["n"]
                a_cost += float(a["g2"].sum()) / (a["n"] * len(a["g2"])) * float(a["s2"][f].sum())
                c_cost += float((a["g2"] / a["n"] * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            tables["CA_A"][n][f], tables["CA_C"][n][f] = a_cost, c_cost
    sampler.__exit__(None, None, None)
    mismatch = max(max(abs(seen[(n, j)][0] - ranges[tens[n][j]][0]), abs(seen[(n, j)][1] - ranges[tens[n][j]][1]))
                   / max(1e-6, ranges[tens[n][j]][1] - ranges[tens[n][j]][0]) for n in names for j in (0, 1))
    if mismatch > 1e-2:
        raise RuntimeError(f"PyTorch activations do not match the ONNX calibration ranges (rel. {mismatch:.2e})")
    kinds = {n: mapping[n].split(".", 3)[-1].removesuffix(".weight") for n in names}   # attn.c_attn, mlp.c_fc, ...
    (args.out_dir / "fisher_tables.json").write_text(json.dumps(
        {"tables": tables, "kind": kinds, "range_mismatch": mismatch,
         "cost": {**sampler.report(f"{len(calib)} windows x (forward + backward), PyTorch CPU"),
                  "threads": torch.get_num_threads()}}) + "\n")
    print(f"fisher: {len(names)} nodes, {sampler.seconds:.1f} s, peak RSS {sampler.peak:.0f} MB "
          f"(+{sampler.peak - sampler.before:.0f} MB); activation ranges match ONNX to {mismatch:.1e}")


def phase_sqnr(args):
    from sqnr_tables import sqnr_table
    graph, mapping, ranges, calib, _, nodes = load_source(args.source_run)
    with RssSampler() as sampler:
        table = sqnr_table(graph, nodes, FORMATS, FORMATS, ranges, calib, None, log=lambda m: print(m, flush=True))
    n_runs = sum(1 for r in table.values() for k in r if k != "fp32|fp32")
    (args.out_dir / "sqnr_table.json").write_text(json.dumps(
        {"table": table, "cost": sampler.report(f"{n_runs + 1} ORT runs x {len(calib)} windows")}) + "\n")
    print(f"sqnr: {len(table)} nodes, {sampler.seconds:.1f} s, peak RSS {sampler.peak:.0f} MB "
          f"(+{sampler.peak - sampler.before:.0f} MB)")


def driver(args):
    from scipy.stats import spearmanr
    from eval_random_mixed_precision import simulate_in_onnxruntime
    from mp_milp import milp_select
    from run_gpt2_fisher_milp import token_nll
    from sample_configs_sqnr import config_cost
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    for phase, out in (("fisher", "fisher_tables.json"), ("sqnr", "sqnr_table.json")):
        if not (args.out_dir / out).exists():
            with open(args.out_dir / f"{phase}.log", "w") as fh:
                subprocess.run([sys.executable, __file__, "--phase", phase, "--source-run", str(args.source_run),
                                "--out-dir", str(args.out_dir)], stdout=fh, stderr=subprocess.STDOUT, check=True)
        log((args.out_dir / f"{phase}.log").read_text().strip().splitlines()[-1])
    fis = json.loads((args.out_dir / "fisher_tables.json").read_text())
    sq = json.loads((args.out_dir / "sqnr_table.json").read_text())
    T, kind = fis["tables"], fis["kind"]

    def cost(v, n, w, a):
        if v == "SQNR":
            return sq["table"][n][f"{w}|{a}"]
        return T["CW"][n][w] + T["CA_" + v][n][a]

    graph, mapping, ranges, calib, test, nodes = load_source(args.source_run)
    names = [n["name"] for n in nodes]
    seq = test.shape[1]

    def per_window(cfg):
        z = simulate_in_onnxruntime(graph, nodes, cfg, ranges, test, None)
        assert np.isfinite(z).all()
        nll = token_nll(z, test).reshape(len(test), seq - 1)
        top1 = (z[:, :-1, :].argmax(-1) == test[:, 1:]).mean(1)
        return nll.mean(1), top1

    base, base_top1 = per_window({})
    if (args.out_dir / "truth.json").exists():
        truth = json.loads((args.out_dir / "truth.json").read_text())["rows"]
        log("single-node truth from truth.json")
    else:
        truth = []
        for i, n in enumerate(names):
            for w, a in CONFIGS:
                d = per_window({n: {"w": w, "a": a}})[0] - base
                truth.append({"node": n, "w": w, "a": a, "dloss": float(d.mean()),
                              "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
            if (i + 1) % 8 == 0:
                log(f"single-node truth: {i + 1}/{len(names)} nodes")
        (args.out_dir / "truth.json").write_text(json.dumps({"test_windows": len(test), "rows": truth}, indent=1) + "\n")
    y = np.array([r["dloss"] for r in truth])
    sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in truth])
    pred_stats = {}
    for v in ("A", "C", "SQNR"):
        p = np.array([cost(v, r["node"], r["w"], r["a"]) for r in truth], dtype=float)
        lr = np.log10(np.clip(p[sig], 1e-300, None) / y[sig])
        cen = lr - np.median(lr)
        srows = [r for r, s in zip(truth, sig) if s]
        pred_stats[v] = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                         "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                         "by_config": {f"{w}/{a}": float(spearmanr(p[ix], y[ix])[0]) for w, a in CONFIGS
                                       for ix in [np.array([(r["w"], r["a"]) == (w, a) for r in truth])]},
                         "kind_bias": {k: float(np.median(cen[[kind[r["node"]] == k for r in srows]]))
                                       for k in sorted(set(kind.values())) if any(kind[r["node"]] == k for r in srows)}}

    from transformers import GPT2LMHeadModel
    params = dict(GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True).named_parameters())
    elems = {n: params[mapping[n]].numel() for n in names}
    flops = {n: elems[n] * seq for n in names}
    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    res, rows = {}, []
    for f in [float(b) for b in args.budgets.split(",")]:
        for v in ("A", "C", "SQNR"):
            cfg, _, status = milp_select(nodes, FORMATS, FORMATS, lambda n, w, a, v=v: cost(v, n, w, a), elems, flops,
                                         f * ref_size, f * ref_bops, time_limit=60)
            assert status in (0, 1), (v, f, status)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
            nll, top1 = per_window(cfg)
            res[(f, v)] = nll
            size, bops = config_cost(cfg, elems, flops)
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / ref_size,
                         "bops": bops / ref_bops, "nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())),
                         "top1": float(top1.mean()), "formats": cfg})
            log(f"f={f:.2f} {v:<5} PPL {np.exp(nll.mean()):8.2f}  top1 {top1.mean():6.2%}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in rows:
        d = res[(r["budget"], r["variant"])] - res[(r["budget"], "SQNR")]
        r["vs_SQNR"] = {"dnll": float(d.mean()), "dnll_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    summary = {"model": "gpt2 (openai-community/gpt2)", "nodes": len(names), "test_windows": len(test),
               "fp32": {"nll": float(base.mean()), "ppl": float(np.exp(base.mean())), "top1": float(base_top1.mean())},
               "cost": {"fisher_A_and_C": fis["cost"], "sqnr": sq["cost"]}, "prediction": pred_stats, "rows": rows}
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")

    fc, sc = fis["cost"], sq["cost"]
    print(f"\n=== GPT-2 small ({len(names)} projections; FP32 PPL {np.exp(base.mean()):.2f}, top-1 {base_top1.mean():.2%})")
    for label, c in (("A/C ", fc), ("SQNR", sc)):
        print(f"cost  {label}: {c['seconds']:8.1f} s   peak RSS {c['peak_rss_mb']:6.0f} MB (+{c['increment_mb']:.0f} MB "
              f"over {c['rss_before_mb']:.0f} MB loaded)   [{c['passes']}]")
    print(f"      time ratio SQNR / C = {sc['seconds'] / fc['seconds']:.0f}x")
    print(f"prediction ({len(truth)} single-node pairs, {sig.sum()} with dloss > 2 SE): all-pairs Spearman / sig / IQR; "
          f"per config; per-kind bias")
    for v, s in pred_stats.items():
        print(f"  {v:<5} {s['spearman']:.3f} / {s['spearman_sig']:.3f} / {s['iqr']:.2f}   "
              + " ".join(f"{k} {x:.3f}" for k, x in s["by_config"].items()) + "   "
              + " ".join(f"{k} {b:+.2f}" for k, b in s["kind_bias"].items()))
    print("MILP picks, test windows: PPL / top-1 (d NLL vs SQNR)")
    for v in ("A", "C", "SQNR"):
        sel = [r for r in rows if r["variant"] == v]
        print(f"  {v:<5} " + "  ".join(f"f={r['budget']:.2f} {r['ppl']:.2f}/{r['top1']:.2%}" for r in sel)
              + ("" if v == "SQNR" else "   d NLL vs SQNR " + " ".join(
                  f"{r['vs_SQNR']['dnll']:+.3f}±{r['vs_SQNR']['dnll_se']:.3f}" for r in sel)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("driver", "fisher", "sqnr"), default="driver")
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "driver":
        driver(args)
    else:
        {"fisher": phase_fisher, "sqnr": phase_sqnr}[args.phase](args)


if __name__ == "__main__":
    main()
