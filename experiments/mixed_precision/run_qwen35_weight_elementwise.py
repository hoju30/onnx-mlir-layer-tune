#!/usr/bin/env python3
"""Qwen3.5-0.8B: BC with an element-wise weight term, pre-registered against the original on fresh windows.

The original weight term of a node (native table B) is W(w) = sum_c Gbar_c sum_i dW_ci^2, with
Gbar_c = (1 / n_in) sum_i G_ci the mean of the per-element empirical Fisher G_ci = sum over calibration windows of
(dL/dW_ci)^2 over the output channel c. The element-wise term is W_elem(w) = sum_c sum_i G_ci dW_ci^2: each weight's
quantization error weighted by its own gradient square instead of its channel's mean. The activation term A (native
table C) is unchanged; BC_elem(w, a) = W_elem(w) + A(a). Only candidates that quantize weights (modes w, wa) change.

W_elem is computed in PyTorch (the native grads pass only emits per-channel sums); the same code's per-channel W is
checked against the native table. If the hypothesis passes, the term goes native in a separate step.

Stages (one --out-dir):
  plan     plan.json (hypothesis, data, pass criteria; frozen first), fresh tokens, candidates (FFN, modes w / wa)
  scores   W_elem and W on the main run's calibration windows; check W against native B; BC / BC_elem frozen
  truth    fresh-window loss changes of the candidates (GPU replay, main run ranges; checks the freeze)
  analyze  pass / fail and the secondary analyses (analysis.json, pairs.csv)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35  # noqa: F401  (transformers 5 first)
import numpy as np
import torch

from run_qwen35_bc import FFN, _pair_class, jdump, jload, replay_truth

HERE = Path(__file__).resolve().parent
MAIN = HERE / "runs/qwen35_bc_20261015"
SEQ = 128
FRESH = ("validation", 20480, 30)        # never used before: main run 0-5119, 8192-14335; recalibration 16384-17663
GATE_UP = ("single: branch choice gate-up, same format", "single: branch choice gate-up, int vs posit",
           "joint: gate/up swap")


def stage_plan(args, out, log):
    from qwen3_swiglu_common import sha256
    split, first, n = FRESH
    ids = q35.wikitext_ids(split)
    toks = np.asarray(ids[first:first + n * SEQ], np.int64).reshape(n, SEQ)
    np.save(out / "tokens_fresh.npy", toks)
    cands = [c for c in jload(MAIN / "candidates.json") if c["mode"] in ("w", "wa")]
    jdump(out / "candidates.json", cands)
    jdump(out / "plan.json", {
        "hypothesis": "the per-channel mean of the weight Fisher misweights Gate vs Up (BC over-estimates Up / Gate "
                      "in weight-only at L23 / L3 and the reverse at L22, runs/qwen35_gateup_recalib_20261015); the "
                      "element-wise term W_elem = sum_c sum_i G_ci dW_ci^2 fixes it without hurting other pairs",
        "criteria": {"BC": "native table of the main run: B[n][w] + C[n][a]",
                     "BC_elem": "W_elem[n][w] (PyTorch, the main run's 10 calibration windows) + native C[n][a]"},
        "calibration": "main run tokens_calibration.npy (validation 0-1279), the windows of the native table",
        "fresh_windows": {"split": split, "tokens": [first, first + n * SEQ], "windows": n,
                          "sha256": sha256(out / "tokens_fresh.npy"),
                          "why": "the diagnostic windows of the main run informed the hypothesis; they are secondary"},
        "candidates": {"count": len(cands), "source": "main run candidates.json, modes w and wa (FFN gate / up / down, "
                                                      "gate+up) of layers 0, 3, 12, 15, 22, 23",
                       "main_candidates_sha256": sha256(MAIN / "candidates.json")},
        "truth": "PyTorch replay from the candidate's layer, GPU FP32 (TF32 off), the main run's int activation ranges",
        "pairs": "equal M and B within (layer, mode); decisive = 95% of 2000 window bootstraps agree",
        "pass_criteria": {
            "P1": "decisive pairs (all classes, modes w + wa): BC_elem correct > BC correct, AND the paired window "
                  "bootstrap 95% interval of accuracy(BC_elem) - accuracy(BC) over all pairs has lower bound > 0",
            "P2": "pooled Gate / Up decision pairs (" + "; ".join(GATE_UP) + "), decisive: BC_elem correct >= BC correct",
            "P3": "every layer, decisive: BC_elem correct >= BC correct - 2",
            "pass": "P1 and P2 and P3 -> implement W_elem natively and compare full searches (pre-registered then); "
                    "otherwise keep the original weight term"},
        "secondary": ["the same metrics on the main run's diagnostic windows (not used for the decision)",
                      "Gate / Up log-ratio bias in weight-only per layer", "W_elem / W per node and format"],
        "seeds": {"bootstrap": 0}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan: {len(cands)} candidates, fresh tokens {first}-{first + n * SEQ}")


def stage_scores(args, out, log):
    from qwen3_swiglu_common import Quantizers, sha256
    from run_native_bc_pipeline import QUANT
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    t0 = time.time()
    plan_main = jload(MAIN / "plan.json")
    layers = {l for v in plan_main["ranking"]["layers"].values() for l in v}
    mapping = jload(MAIN / "node_mapping.json")
    nodes = {n: v for n, v in mapping.items() if v["layer"] in layers and v["kind"] in FFN}
    toks = np.load(MAIN / "tokens_calibration.npy")
    q = Quantizers(MAIN, formats=("p4e1", "p6e1", "p8e1"))
    hf = q35.load_hf(torch.float32)
    params = dict(hf.named_parameters())
    ad = q35.TextExport(hf, toks.shape[1]).eval()
    for info in nodes.values():
        params[info["param"]].requires_grad_(True)
    G = {n: torch.zeros(params[info["param"]].shape, dtype=torch.float64) for n, info in nodes.items()}
    for ids in toks:                                  # per window: (dL/dW)^2, summed over windows (native grads)
        x = torch.from_numpy(ids)
        q35.summed_ce(ad(x[None]), x).backward()
        for n, info in nodes.items():
            p = params[info["param"]]
            G[n] += p.grad.double() ** 2
            p.grad = None
    native = jload(MAIN / "table.json")
    W, W_elem, check = {}, {}, []
    with torch.no_grad():
        for n, info in nodes.items():
            w = params[info["param"]].detach()        # torch layout [out, in]; channels = rows
            Gbar = G[n].mean(1, keepdim=True)
            W[n], W_elem[n] = {"fp32": 0.0}, {"fp32": 0.0}
            for f in QUANT:
                d2 = (q.weight(w, f) - w).double() ** 2
                W[n][f] = float((Gbar * d2).sum())
                W_elem[n][f] = float((G[n] * d2).sum())
                check.append(abs(W[n][f] - native["B"][n][f]) / native["B"][n][f] if native["B"][n][f] else 0.0)
    chk = {"median": float(np.median(check)), "max": float(np.max(check))}
    if chk["median"] > 1e-3:
        raise SystemExit(f"per-channel W does not reproduce the native B: {chk}")
    jdump(out / "weight_terms.json", {"W": W, "W_elem": W_elem, "native_B_check_rel": chk,
                                      "nodes": {n: {"layer": v["layer"], "kind": v["kind"]} for n, v in nodes.items()}})
    cands = jload(out / "candidates.json")
    for c in cands:
        c["bc_elem"] = float(sum(W_elem[c["nodes"][k]][w] + native["C"][c["nodes"][k]][a] for k, (w, a) in c["formats"].items()))
    jdump(out / "candidates.json", cands)
    jdump(out / "freeze.json", {"candidates_sha256": sha256(out / "candidates.json"),
                                "weight_terms_sha256": sha256(out / "weight_terms.json"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"scores: {len(nodes)} nodes, W vs native B rel median {chk['median']:.2e} max {chk['max']:.2e}, "
        f"{time.time() - t0:.0f}s; frozen")


def stage_truth(args, out, log):
    from qwen3_swiglu_common import sha256
    fz = jload(out / "freeze.json")
    for f, k in (("candidates.json", "candidates_sha256"), ("weight_terms.json", "weight_terms_sha256")):
        if sha256(out / f) != fz[k]:
            raise SystemExit(f"{f} changed after the freeze")
    replay_truth(MAIN, jload(out / "candidates.json"), np.load(out / "tokens_fresh.npy"),
                 jload(MAIN / "ranges_by_point.json"), out / "truth_fresh.json", args.batch, log, device=args.device)


def evaluate(cands, rows, crit, boot_seed):
    """Equal-resource pairs of `cands` against the truth `rows`: decisive / all accuracy of each criterion, by
    class, mode and layer; window-bootstrap interval of the accuracy difference (BC_elem - BC) over all pairs."""
    names = [c["name"] for c in cands]
    D = np.array([rows[n] for n in names]) / (SEQ - 1)
    m, nw = D.mean(1), D.shape[1]
    boot = np.random.default_rng(boot_seed).integers(0, nw, (2000, nw))
    Db = np.stack([D[:, b].mean(1) for b in boot], 1)
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["layer"], c["mode"]), []).append(i)
    pairs = [(i, j, _pair_class(cands[i], cands[j])) for ix in groups.values() for a, i in enumerate(ix) for j in ix[a + 1:]
             if cands[i]["M_bytes"] == cands[j]["M_bytes"] and cands[i]["B_bops"] == cands[j]["B_bops"]]
    I, J = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    d = m[I] - m[J]
    dec = ((Db[I] - Db[J]) * np.sign(d)[:, None] > 0).mean(1) >= 0.95
    ok = {k: np.where(v[I] == v[J], 0.5, np.sign(v[I] - v[J]) == np.sign(d)) for k, v in crit.items()}
    ts = np.sign(Db[I] - Db[J])
    okb = {k: np.where((v[I] == v[J])[:, None], 0.5, np.sign(v[I] - v[J])[:, None] == ts) for k, v in crit.items()}

    def block(sel):
        sel = np.asarray(sel, bool)
        r = {"pairs": int(sel.sum()), "decisive": int((sel & dec).sum())}
        for k in crit:
            r[k] = {"decisive_correct": float(ok[k][sel & dec].sum()), "all_accuracy": float(ok[k][sel].mean())}
        diff = okb["BC_elem"][sel].mean(0) - okb["BC"][sel].mean(0)
        r["elem_minus_bc_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        return r
    cls = np.array([p[2] for p in pairs])
    layer = np.array([cands[i]["layer"] for i in I])
    mode = np.array([cands[i]["mode"] for i in I])
    res = {"all": block(np.ones(len(pairs), bool)), "gate_up_pooled": block(np.isin(cls, GATE_UP))}
    for k, v in (("class", cls), ("mode", mode), ("layer", layer)):
        for val in sorted(set(v.tolist()), key=str):
            res[f"{k}={val}"] = block(v == val)
    for md in ("w", "wa"):
        for l in sorted(set(layer.tolist())):
            sel = (mode == md) & (layer == l) & np.isin(cls, GATE_UP)
            if sel.any():
                res[f"gate_up|mode={md}|layer={l}"] = block(sel)
    rows_out = [[names[i], names[j], cl, float(dd), bool(de), *[float(ok[k][x]) for k in crit]]
                for x, (i, j, cl, dd, de) in enumerate(zip(I, J, cls, d, dec))]
    # Gate / Up log-ratio bias (weight-only and wa), both truths significantly > 0
    lo = np.percentile(Db, 2.5, axis=1)
    idx = {(c["layer"], c["mode"], c["set"], tuple(c["formats"][c["set"]])): i for i, c in enumerate(cands) if c["set"] != "gate+up"}
    bias = {}
    for (l, md, s, f), i in idx.items():
        if s == "gate_proj" and lo[i] > 0 and lo[j := idx[(l, md, "up_proj", f)]] > 0:
            for k, v in crit.items():
                bias.setdefault(f"L{l}|{md}", {}).setdefault(k, []).append(float(np.log(v[j] / v[i]) - np.log(m[j] / m[i])))
    res["gate_up_logratio_bias"] = {g: {k: {"n": len(e), "median": float(np.median(e))} for k, e in v.items()}
                                    for g, v in sorted(bias.items())}
    return res, rows_out


def stage_analyze(args, out, log):
    import csv
    from qwen3_swiglu_common import sha256
    plan = jload(out / "plan.json")
    cands = jload(out / "candidates.json")
    if sha256(out / "candidates.json") != jload(out / "freeze.json")["candidates_sha256"]:
        raise SystemExit("candidates.json changed after the freeze")
    crit = {"BC": np.array([c["bc"] for c in cands]), "BC_elem": np.array([c["bc_elem"] for c in cands])}
    res = {}
    for name, rows in (("fresh", jload(out / "truth_fresh.json")["rows"]), ("diagnostic_secondary", jload(MAIN / "truth.json")["rows"])):
        res[name], pr = evaluate(cands, rows, crit, plan["seeds"]["bootstrap"])
        with open(out / f"pairs_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "class", "truth_diff", "decisive", *[f"{k}_correct" for k in crit]])
            w.writerows(pr)
    f = res["fresh"]
    p1 = f["all"]["BC_elem"]["decisive_correct"] > f["all"]["BC"]["decisive_correct"] and f["all"]["elem_minus_bc_ci95"][0] > 0
    p2 = f["gate_up_pooled"]["BC_elem"]["decisive_correct"] >= f["gate_up_pooled"]["BC"]["decisive_correct"]
    p3 = all(f[k]["BC_elem"]["decisive_correct"] >= f[k]["BC"]["decisive_correct"] - 2 for k in f if k.startswith("layer="))
    res["decision"] = {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "pass": bool(p1 and p2 and p3)}
    wt = jload(out / "weight_terms.json")
    res["W_elem_over_W"] = {f"L{v['layer']}|{v['kind']}": {fm: wt["W_elem"][n][fm] / wt["W"][n][fm] for fm in ("int4", "int6", "int8", "p6e1", "p8e1")}
                            for n, v in wt["nodes"].items()}
    jdump(out / "analysis.json", res)
    a = f["all"]
    log(f"fresh: decisive {a['decisive']}, BC {a['BC']['decisive_correct']:.0f}, BC_elem {a['BC_elem']['decisive_correct']:.0f}, "
        f"diff CI {a['elem_minus_bc_ci95']}; gate/up BC {f['gate_up_pooled']['BC']['decisive_correct']:.0f} elem "
        f"{f['gate_up_pooled']['BC_elem']['decisive_correct']:.0f}; decision {res['decision']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "scores", "truth", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    if args.stage == "plan":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "plan.json").exists():
        raise SystemExit("run --stage plan first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    import resource
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
    timing["runs"].append({"stage": args.stage, "device": args.device, "wall_s": time.time() - t0,
                           "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                           "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20})
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
