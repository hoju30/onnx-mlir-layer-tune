#!/usr/bin/env python3
"""GPT-2 small: per-channel weights in the full weight + activation MILP (run_w8pc_full_compare.py for GPT-2).

The 48 projections of runs/gpt2_shared_ablation_20260930 store W as [in, out] (Gemm, no transB), so int weights are
quantized with one scale per output feature (axis 1). Activations: the pass's a= (per-tensor affine range).
  A     weight mean(g_W^2)*||dW||^2 + activation A (per tensor)
  AC    weight as A + activation C (per-feature residual + coherent per-feature mean over a window's tokens)
  BC    weight sum_o mean_o(g_W^2)*||dW_o||^2 (per output feature) + activation C
  SQNR  single-node logit noise power with per-channel weights: int-weight entries recomputed (saved after every node,
        resumable), f16 / bf16 / fp32 weight entries reused from runs/criteria_cost_gpt2_20261004/sqnr_table.json
Single-node truth at (w, a) = int4/int8, int6/int8, int8/int4, int8/int6 on the 32 test windows (SE over windows);
MILP picks at budgets relative to the projections' all-W8A8 weight bytes and BOPs, evaluated on the test windows.
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

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_gpt2_criteria_cost_compare import load_source  # noqa: E402
from run_gpt2_fisher_milp import token_nll  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant  # noqa: E402
from run_w8_per_channel_check import per_channel_int  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

AF = FORMATS[:-1]
INT = ("int4", "int5", "int6", "int7", "int8")
CONFIGS = (("int4", "int8"), ("int6", "int8"), ("int8", "int4"), ("int8", "int6"))
CRITERIA = ("A", "AC", "BC", "SQNR")


def pc_out_features(w, bits):
    """Per-channel int quantization along axis 1 (output features of a [in, out] weight)."""
    return per_channel_int(np.ascontiguousarray(np.asarray(w, np.float32).T), bits).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(b) for b in args.budgets.split(",")]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    original = erm.np_fake_quant_weight
    erm.np_fake_quant_weight = lambda w, fmt, oracle: (pc_out_features(w, int(fmt[3:])) if fmt.startswith("int")
                                                        else original(w, fmt, oracle))
    graph, mapping, ranges, calib, test, nodes = load_source(args.source_run)
    names = [n["name"] for n in nodes]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    seq = test.shape[1]

    # --- Fisher tables ---
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    module_of = {n: mods[mapping[n].removesuffix(".weight")] for n in names}
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    import onnx
    for n in names[:3]:   # the ONNX weight equals the HF weight ([in, out]), so both use the same axis
        assert np.allclose(onnx.numpy_helper.to_array(inits[by_name[n].input[1]]), params[mapping[n]].detach().numpy())

    def csum(t):
        return t.sum(dim=tuple(range(t.dim() - 1)))

    cap = {}
    for n, mm in module_of.items():
        mm.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2w = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for ids in calib:
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(loss, targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += grads[3 * i].double() ** 2
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                    a["g2"] = a["g2"] + csum(g ** 2)
                    a["gc"] = a["gc"] + csum(g) ** 2
                    a["n"] += t.numel() // t.shape[-1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + csum(d)
                        a["s2"][f] = a["s2"][f] + csum(d ** 2)
        cap.clear()
    CW, CA = {"A": {}, "B": {}}, {"A": {}, "C": {}}
    elems = {}
    for n in names:
        W = params[mapping[n]].detach().numpy()
        elems[n] = W.size
        G = g2w[n].numpy()                       # [in, out]
        CW["A"][n], CW["B"][n], CA["A"][n], CA["C"][n] = {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}
        for f in AF:
            d2 = (erm.np_fake_quant_weight(W, f, None).astype(np.float64) - W) ** 2
            CW["A"][n][f] = float(G.mean() * d2.sum())
            CW["B"][n][f] = float((G.mean(0) * d2.sum(0)).sum())
            a_cost = c_cost = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                mu = a["s1"][f] / a["n"]
                a_cost += float(a["g2"].sum()) / (a["n"] * len(a["g2"])) * float(a["s2"][f].sum())
                c_cost += float((a["g2"] / a["n"] * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            CA["A"][n][f], CA["C"][n][f] = a_cost, c_cost
    del model, g2w
    log("Fisher tables ready (A, AC, BC)")

    # --- SQNR with per-channel weights (resumable) ---
    spath = args.out_dir / "sqnr_table_per_channel.json"
    state = json.loads(spath.read_text()) if spath.exists() else {"seconds_int_entries": 0.0, "table": {}}
    old = json.loads((HERE / "runs/criteria_cost_gpt2_20261004/sqnr_table.json").read_text())["table"]
    if len(state["table"]) < len(names):
        ref = erm.simulate_in_onnxruntime(graph, nodes, {}, ranges, calib, None).astype(np.float64)
        for i, n in enumerate(names):
            if n in state["table"]:
                continue
            t_node = time.time()
            row = {k: v for k, v in old[n].items() if k.split("|")[0] not in INT}
            for w in INT:
                for a in FORMATS:
                    z = erm.simulate_in_onnxruntime(graph, nodes, {n: {"w": w, "a": a}}, ranges, calib, None)
                    row[f"{w}|{a}"] = float(((z - ref) ** 2).sum()) if np.isfinite(z).all() else None
                    del z
            state["table"][n] = row
            state["seconds_int_entries"] += time.time() - t_node
            spath.write_text(json.dumps(state) + "\n")
            log(f"SQNR int-weight entries {len(state['table'])}/{len(names)} nodes")
    sq = state["table"]
    log(f"SQNR table (per-channel weights) ready; int entries took {state['seconds_int_entries']:.0f} s")

    def cost(c, n, w, a):
        if c == "SQNR":
            return sq[n][f"{w}|{a}"]
        return CW["B" if c == "BC" else "A"][n][w] + CA["A" if c == "A" else "C"][n][a]

    def per_window(cfg):
        z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, test, None)
        assert np.isfinite(z).all()
        nll = token_nll(z, test).reshape(len(test), seq - 1)
        return nll.mean(1), (z[:, :-1, :].argmax(-1) == test[:, 1:]).mean(1)

    base, base_top1 = per_window({})
    tpath = args.out_dir / "truth.json"
    if tpath.exists():
        truth = json.loads(tpath.read_text())["rows"]
    else:
        truth = []
        for i, n in enumerate(names):
            for w, a in CONFIGS:
                d = per_window({n: {"w": w, "a": a}})[0] - base
                truth.append({"node": n, "w": w, "a": a, "dloss": float(d.mean()),
                              "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
            if (i + 1) % 8 == 0:
                log(f"truth {i + 1}/{len(names)} nodes")
        tpath.write_text(json.dumps({"test_windows": len(test), "weights": "per-channel", "rows": truth}, indent=1) + "\n")
    y = np.array([r["dloss"] for r in truth])
    sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in truth])
    pred_stats = {}
    for c in CRITERIA:
        p = np.array([cost(c, r["node"], r["w"], r["a"]) for r in truth], dtype=float)
        lr = np.log10(np.clip(p[sig], 1e-300, None) / y[sig])
        pred_stats[c] = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                         "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                         "by_config": {f"{w}/{a}": float(spearmanr(p[ix], y[ix])[0]) for w, a in CONFIGS
                                       for ix in [np.array([(r["w"], r["a"]) == (w, a) for r in truth])]}}

    flops = {n: elems[n] * seq for n in names}
    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    res, rows = {}, []
    for f in budgets:
        for c in CRITERIA:
            cfg, _, status = milp_select(nodes, FORMATS, FORMATS, lambda n, w, a, c=c: cost(c, n, w, a), elems, flops,
                                         f * ref_size, f * ref_bops, time_limit=60)
            assert status in (0, 1), (c, f, status)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
            nll, top1 = per_window(cfg)
            res[(f, c)] = nll
            size, bops = config_cost(cfg, elems, flops)
            rows.append({"budget": f, "criterion": c, "milp_optimal": status == 0, "size": size / ref_size,
                         "bops": bops / ref_bops, "nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())),
                         "top1": float(top1.mean()), "formats": cfg})
            log(f"f={f:.2f} {c:<5} PPL {np.exp(nll.mean()):8.2f}  top1 {top1.mean():6.2%}")
    for r in rows:
        for ref_c in ("A", "SQNR"):
            d = res[(r["budget"], r["criterion"])] - res[(r["budget"], ref_c)]
            r[f"vs_{ref_c}"] = {"dnll": float(d.mean()), "dnll_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    w8a8 = per_window({n: {"w": "int8", "a": "int8"} for n in names})
    summary = {"model": "gpt2", "weights": "per-channel (output features)",
               "fp32": {"ppl": float(np.exp(base.mean())), "top1": float(base_top1.mean())},
               "w8pc_a8": {"ppl": float(np.exp(w8a8[0].mean())), "top1": float(w8a8[1].mean())},
               "sqnr_int_entries_seconds": state["seconds_int_entries"], "truth_pairs": len(truth),
               "truth_significant": int(sig.sum()), "prediction": pred_stats, "rows": rows}
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    erm.np_fake_quant_weight = original

    print(f"\n=== GPT-2, per-channel weights: FP32 PPL {np.exp(base.mean()):.2f}, all W8pc+A8 PPL {np.exp(w8a8[0].mean()):.2f}")
    print(f"   prediction ({len(truth)} pairs, {sig.sum()} > 2 SE): Spearman / sig / IQR; per config")
    for c, s in pred_stats.items():
        print(f"     {c:<5} {s['spearman']:.3f} / {s['spearman_sig']:.3f} / {s['iqr']:.2f}   "
              + " ".join(f"{k} {v:.3f}" for k, v in s["by_config"].items()))
    print("   MILP (PPL / top-1); d NLL vs SQNR per budget")
    for c in CRITERIA:
        sel = [r for r in rows if r["criterion"] == c]
        print(f"     {c:<5} " + "  ".join(f"f={r['budget']:.2f} {r['ppl']:.2f}/{r['top1']:.2%}" for r in sel)
              + "   vs SQNR " + " ".join(f"{r['vs_SQNR']['dnll']:+.3f}±{r['vs_SQNR']['dnll_se']:.3f}" for r in sel))


if __name__ == "__main__":
    main()
