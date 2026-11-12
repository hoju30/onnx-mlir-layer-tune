#!/usr/bin/env python3
"""Qwen3-0.6B: does the finite-difference curvature at the block output (FD) choose same-layer candidates better
than the native BC, across more layers, formats and Gate+Up joint quantization? Pre-registered; the original BC is
kept as the baseline (nothing in the pipeline changes).

Candidates per layer (Down FP32, the pass's per-consumer QDQ, the calibration (scale, zp) of
runs/native_eps_fix_20261014/qwen3/ranges_by_point.json, per-output-channel weights, Universal-exact posit LUTs):
  activation-only (weight FP32): Gate-only / Up-only x {int4, p4e1, int6, p6e1, int8, p8e1}; Gate+Up with
                                 Gate, Up in {int4, int6, int8, p6e1}
  weight-only (activation FP32): the same with weight formats
FD score (calibration windows 0-9 only): e = Y_hat - Y_fp32 at Y = down_proj output, the downstream replayed on
Y_fp32 +- h e, h = 1/16: sum_s (dL_s(h) + dL_s(-h)) / 2h^2 (second order of the loss along e at the 1/16 scale; the
first order is not used: it does not transfer across windows).
BC score: the native table's W + A of the quantized branches (table.json B + C).
Truth: the summed next-token CE change of the full quantization on the new validation windows 90-119 (never used by
any earlier study), downstream replay as runs/qwen3_block_criterion_20261014.
Decisions: inside each (layer, mode), every pair with equal model size and BOPs; decisive = the window bootstrap
keeps the pair's order in >= 95% of 2000 replicates and |dS| > 1.
PASS (fixed before any FD score or truth exists):
  (1) on decisive equal-resource pairs FD is right more often than BC, and the window-bootstrap 95% interval of the
      accuracy difference (equal-resource pairs with |dS| > 1, re-filtered per replicate) is above 0;
  (2) FD is not worse than BC on decisive equal-resource pairs in at least 4 of the 5 layers and in both modes.

Stages (one new --out-dir): plan, capture, scores (--layers), freeze, truth (--layers), analyze, tables.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_qwen_block_criterion import CAL, SEQ, TABLES, TAU, Block, Downstream, Setup, jdump, jload  # noqa: E402

LAYERS = (0, 7, 14, 21, 27)
SINGLE = ("int4", "p4e1", "int6", "p6e1", "int8", "p8e1")
JOINT = ("int4", "int6", "int8", "p6e1")
H = 1 / 16
NEW = (90, 120)
BOOT = 2000
DECISIVE = 0.95


def option(mode, fmt):
    return ("fp32", fmt) if mode == "a" else (fmt, "fp32")


def candidates():
    out = []
    for layer, mode in itertools.product(LAYERS, ("a", "w")):
        cs = [(f, None) for f in SINGLE] + [(None, f) for f in SINGLE] + list(itertools.product(JOINT, JOINT))
        for g, u in cs:
            name = f"L{layer}|{mode}|g={g or '-'}|u={u or '-'}"
            out.append({"name": name, "layer": layer, "mode": mode, "gate": g, "up": u,
                        "target": "both" if g and u else "gate" if g else "up"})
    return out


def resources(c):
    from htq_pareto_search import FORMAT_BITS
    bits = []
    for f in (c["gate"], c["up"]):
        w, a = option(c["mode"], f) if f else ("fp32", "fp32")
        bits.append((FORMAT_BITS[w], FORMAT_BITS[a]))
    return (sum(w for w, _ in bits), sum(w * a for w, a in bits))


def setup(out, threads):
    from qwen3_swiglu_common import PositLut
    S = Setup(out, threads)
    S.q.posit["p4e1"] = PositLut(S.q.oracle, "p4e1")
    return S


def stage_plan(args, out, log):
    from qwen3_swiglu_common import environment, sha256
    cs = candidates()
    jdump(out / "plan.json", {
        "layers": list(LAYERS), "single_formats": list(SINGLE), "joint_formats": list(JOINT), "modes": ["a", "w"],
        "candidates": len(cs), "names": [c["name"] for c in cs], "h": H, "fd": "symmetric, second order only",
        "calibration_windows": [0, CAL - 1], "new_windows": [NEW[0], NEW[1] - 1], "tau": TAU,
        "decisive": f"bootstrap order kept >= {DECISIVE} of {BOOT} and |dS| > {TAU}",
        "pass": ["decisive equal-resource pairs: FD right > BC right, and bootstrap 95% CI of the accuracy difference "
                 "(equal-resource pairs, |dS| > tau, re-filtered per replicate) above 0",
                 "FD not worse than BC on decisive equal-resource pairs in >= 4 of 5 layers and in both modes"],
        "secondary": ["all equal-resource pairs", "within-group Spearman", "Gate/Up swaps", "joint-only",
                      "same-bit int vs posit", "per layer, per mode"]})
    jdump(out / "environment.json", {**environment(["experiments/mixed_precision/run_qwen_fd_validation.py",
                                                    "experiments/mixed_precision/run_qwen_block_criterion.py"]),
                                     "table_sha256": sha256(TABLES / "table.json"),
                                     "ranges_sha256": sha256(TABLES / "ranges_by_point.json")})
    log(f"plan frozen: {len(cs)} candidates")


def stage_capture(args, out, log):
    """FP32 tensors (X, zg, zu, Y, residual before the MLP) at the five layers, calibration and new windows."""
    from qwen3_swiglu_common import PREP, summed_ce
    from run_qwen_branch_ablation import tokenize_validation
    S = Setup(out, args.threads)
    torch = S.torch
    t0 = time.time()
    toks, _ = tokenize_validation(NEW[1])
    val = np.load(PREP / "validation_tokens.npy")
    if not np.array_equal(toks[:CAL], val):
        raise RuntimeError("re-tokenized calibration windows differ from validation_tokens.npy")
    sets = {"cal": toks[:CAL], "new": toks[NEW[0]:NEW[1]]}
    cap = {}

    def keep(name):
        def f(m, i, o):
            cap[name] = (i[0], o)
        return f

    def keep_pre(name):
        def f(m, i):
            cap[name] = i[0]
        return f
    hs = []
    for l in LAYERS:
        mlp = S.mods[f"model.layers.{l}.mlp"]
        hs += [mlp.register_forward_hook(keep((l, "mlp"))), mlp.gate_proj.register_forward_hook(keep((l, "gate"))),
               mlp.up_proj.register_forward_hook(keep((l, "up"))),
               S.mods[f"model.layers.{l}.post_attention_layernorm"].register_forward_pre_hook(keep_pre((l, "hmid")))]
    info = {}
    for sname, tk in sets.items():
        store = {f"L{l}|{k}": [] for l in LAYERS for k in ("x", "zg", "zu", "y", "hmid")}
        losses = []
        with torch.no_grad():
            for ids in tk:
                t = torch.from_numpy(ids)
                losses.append(float(summed_ce(S.wrapper(t[None]), t)))
                for l in LAYERS:
                    x, y = cap[(l, "mlp")]
                    for k, v in (("x", x), ("zg", cap[(l, "gate")][1]), ("zu", cap[(l, "up")][1]), ("y", y),
                                 ("hmid", cap[(l, "hmid")])):
                        store[f"L{l}|{k}"].append(v[0].numpy().copy())
        np.savez(out / f"cache_{sname}.npz", **{k: np.stack(v) for k, v in store.items()})
        np.save(out / f"tokens_{sname}.npy", tk)
        info[sname] = {"windows": len(tk), "losses": losses}
    for h in hs:
        h.remove()
    jdump(out / "capture.json", {**info, "seconds": time.time() - t0})
    log(f"captured in {time.time() - t0:.0f}s")


def load(out, sname):
    z = np.load(out / f"cache_{sname}.npz")
    return {k: z[k] for k in z.files}, np.load(out / f"tokens_{sname}.npy")


def stage_scores(args, out, log):
    """FD second order at h on the calibration windows, every candidate of the given layers; resumable."""
    S = setup(out, args.threads)
    torch = S.torch
    cache, toks = load(out, "cal")
    for layer in [int(x) for x in args.layers.split(",")]:
        f = out / f"scores_L{layer}.json"
        done = jload(f) if f.exists() else {"rows": {}, "seconds": 0.0, "downstream_forward_windows": 0}
        t1 = time.time()
        B = Block(S, cache, layer, slice(0, CAL))
        check = float((B.y_ref - torch.from_numpy(cache[f"L{layer}|y"])).abs().max() / torch.from_numpy(cache[f"L{layer}|y"]).abs().max())
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"])
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            done["downstream_forward_windows"] += CAL
            for c in [c for c in candidates() if c["layer"] == layer]:
                if c["name"] in done["rows"]:
                    continue
                g = option(c["mode"], c["gate"]) if c["gate"] else None
                u = option(c["mode"], c["up"]) if c["up"] else None
                e = B.y(g, u) - B.y_ref
                p = D.run(hmid + B.y_ref + H * e) - base
                m = D.run(hmid + B.y_ref - H * e) - base
                done["downstream_forward_windows"] += 2 * CAL
                a2 = (p + m) / (2 * H * H)
                done["rows"][c["name"]] = {"fd": float(a2.sum()), "fd_windows": a2.tolist(),
                                           "a1_windows": ((p - m) / (2 * H)).tolist()}
                if len(done["rows"]) % 8 == 0:
                    jdump(f, done)
        done["seconds"] += time.time() - t1
        done["y_replay_rel_max"] = check
        jdump(f, done)
        log(f"layer {layer}: {len(done['rows'])} candidates scored ({time.time() - t1:.0f}s)")


def stage_freeze(args, out, log):
    from qwen3_swiglu_common import sha256
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    files = [out / f"scores_L{l}.json" for l in LAYERS]
    n = sum(len(jload(f)["rows"]) for f in files)
    if n != len(candidates()):
        raise SystemExit(f"{n} of {len(candidates())} candidates scored")
    jdump(out / "freeze.json", {"scores_sha256": {f.name: sha256(f) for f in files},
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("scores frozen")


def stage_truth(args, out, log):
    """Actual loss change of the full quantization on the new windows; needs the frozen scores."""
    from qwen3_swiglu_common import sha256
    fz = jload(out / "freeze.json")
    for name, h in fz["scores_sha256"].items():
        if sha256(out / name) != h:
            raise SystemExit(f"{name} changed after the freeze")
    S = setup(out, args.threads)
    torch = S.torch
    cache, toks = load(out, "new")
    cap = jload(out / "capture.json")["new"]["losses"]
    for layer in [int(x) for x in args.layers.split(",")]:
        f = out / f"truth_L{layer}.json"
        done = jload(f) if f.exists() else {"rows": {}, "seconds": 0.0}
        t1 = time.time()
        n = toks.shape[0]
        B = Block(S, cache, layer, slice(0, n))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"])
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            done["downstream_vs_model_loss_max_abs"] = float(np.abs(base - np.array(cap)).max())
            for c in [c for c in candidates() if c["layer"] == layer]:
                if c["name"] in done["rows"]:
                    continue
                g = option(c["mode"], c["gate"]) if c["gate"] else None
                u = option(c["mode"], c["up"]) if c["up"] else None
                done["rows"][c["name"]] = (D.run(hmid + B.y(g, u)) - base).tolist()
                if len(done["rows"]) % 8 == 0:
                    jdump(f, done)
        done["seconds"] += time.time() - t1
        jdump(f, done)
        log(f"layer {layer}: {len(done['rows'])} candidates ({time.time() - t1:.0f}s)")


def bc_score(c, table, node):
    s = 0.0
    for kind in ("gate", "up"):
        f = c[kind]
        if f:
            w, a = option(c["mode"], f)
            s += table["B"][node[(c["layer"], kind)]][w] + table["C"][node[(c["layer"], kind)]][a]
    return s


def stage_analyze(args, out, log):
    from scipy.stats import spearmanr
    from qwen3_swiglu_common import node_mapping
    table = jload(TABLES / "table.json")
    node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    cs = candidates()
    sc, tr = {}, {}
    for l in LAYERS:
        sc.update(jload(out / f"scores_L{l}.json")["rows"])
        tr.update(jload(out / f"truth_L{l}.json")["rows"])
    names = [c["name"] for c in cs]
    dl = np.array([tr[n] for n in names])                       # [cand, new windows]
    S_new = dl.sum(1)
    pred = {"FD": np.array([sc[n]["fd"] for n in names]), "BC": np.array([bc_score(c, table, node) for c in cs])}
    res = {"candidates": len(cs), "new_windows": dl.shape[1]}
    # equal-resource pairs inside (layer, mode)
    P = []
    for (l, m) in itertools.product(LAYERS, ("a", "w")):
        ix = [k for k, c in enumerate(cs) if c["layer"] == l and c["mode"] == m]
        for i, j in itertools.combinations(ix, 2):
            if resources(cs[i]) == resources(cs[j]):
                a, b = cs[i], cs[j]
                single = a["target"] != "both" and b["target"] != "both"
                fa, fb = a["gate"] or a["up"], b["gate"] or b["up"]
                kind = ("swap" if a["target"] == b["target"] == "both" and a["gate"] == b["up"] and a["up"] == b["gate"] else
                        "joint_other" if a["target"] == b["target"] == "both" else
                        "gate_vs_up" if single and a["target"] != b["target"] and fa == fb else
                        "int_vs_posit" if single and a["target"] == b["target"] else
                        "cross_single" if single else "single_vs_joint")
                P.append({"i": i, "j": j, "layer": l, "mode": m, "kind": kind, "joint": a["target"] == "both" or b["target"] == "both"})
    I = np.array([p["i"] for p in P])
    J = np.array([p["j"] for p in P])
    rng = np.random.default_rng(0)
    reps = [rng.integers(0, dl.shape[1], dl.shape[1]) for _ in range(BOOT)]
    Sb = np.array([dl[:, r].sum(1) for r in reps])               # [boot, cand]
    dS = S_new[I] - S_new[J]
    keep_frac = (np.sign(Sb[:, I] - Sb[:, J]) == np.sign(dS)).mean(0)
    decisive = (keep_frac >= DECISIVE) & (np.abs(dS) > TAU)
    right = {m: np.sign(v[I] - v[J]) == np.sign(dS) for m, v in pred.items()}

    def count(mask):
        return {m: int((r & mask).sum()) for m, r in right.items()} | {"pairs": int(mask.sum()),
                                                                          "FD_fixed": int((right["FD"] & ~right["BC"] & mask).sum()),
                                                                          "FD_broken": int((~right["FD"] & right["BC"] & mask).sum())}
    sel = {"all equal-resource (|dS|>tau)": np.abs(dS) > TAU, "decisive": decisive}
    tables = {}
    for sname, base_mask in sel.items():
        t = {"total": count(base_mask)}
        for l in LAYERS:
            t[f"L{l}"] = count(base_mask & np.array([p["layer"] == l for p in P]))
        for m in ("a", "w"):
            t[f"mode {m}"] = count(base_mask & np.array([p["mode"] == m for p in P]))
        for k in ("gate_vs_up", "swap", "int_vs_posit", "cross_single", "joint_other", "single_vs_joint"):
            t[k] = count(base_mask & np.array([p["kind"] == k for p in P]))
        tables[sname] = t
    res["pairs"] = tables
    # bootstrap of the accuracy difference (pairs re-filtered per replicate)
    diffs = []
    for b in range(BOOT):
        d = Sb[b, I] - Sb[b, J]
        k = np.abs(d) > TAU
        acc = {m: ((np.sign(v[I] - v[J]) == np.sign(d)) & k).sum() / max(k.sum(), 1) for m, v in pred.items()}
        diffs.append(acc["FD"] - acc["BC"])
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    res["bootstrap_accuracy_diff_FD_minus_BC"] = {"ci95": ci, "mean": float(np.mean(diffs))}
    # within-group Spearman (all candidates of a layer and mode)
    sp = {}
    for (l, m) in itertools.product(LAYERS, ("a", "w")):
        ix = [k for k, c in enumerate(cs) if c["layer"] == l and c["mode"] == m]
        sp[f"L{l}|{m}"] = {p: float(spearmanr(v[ix], S_new[ix]).statistic) for p, v in pred.items()}
    res["spearman"] = sp
    # pass criteria
    dec = tables["decisive"]
    c1 = dec["total"]["FD"] > dec["total"]["BC"] and ci[0] > 0
    layers_ok = sum(dec[f"L{l}"]["FD"] >= dec[f"L{l}"]["BC"] for l in LAYERS)
    modes_ok = all(dec[f"mode {m}"]["FD"] >= dec[f"mode {m}"]["BC"] for m in ("a", "w"))
    res["pass"] = {"criterion_1": bool(c1), "layers_not_worse": int(layers_ok), "modes_not_worse": bool(modes_ok),
                   "criterion_2": bool(layers_ok >= 4 and modes_ok), "PASS": bool(c1 and layers_ok >= 4 and modes_ok)}
    # cost
    res["cost"] = {"scores_seconds": {f"L{l}": jload(out / f"scores_L{l}.json")["seconds"] for l in LAYERS},
                   "scores_downstream_forward_windows": sum(jload(out / f"scores_L{l}.json")["downstream_forward_windows"] for l in LAYERS),
                   "truth_seconds": {f"L{l}": jload(out / f"truth_L{l}.json")["seconds"] for l in LAYERS},
                   "capture_seconds": jload(out / "capture.json")["seconds"]}
    res["checks"] = {"y_replay_rel_max": {f"L{l}": jload(out / f"scores_L{l}.json")["y_replay_rel_max"] for l in LAYERS},
                     "truth_downstream_vs_model": {f"L{l}": jload(out / f"truth_L{l}.json")["downstream_vs_model_loss_max_abs"] for l in LAYERS}}
    # per-pair records
    import csv
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "layer", "mode", "kind", "dS_new", "keep_frac", "decisive", "FD_right", "BC_right"])
        for k, p in enumerate(P):
            w.writerow([names[p["i"]], names[p["j"]], p["layer"], p["mode"], p["kind"], dS[k], keep_frac[k], bool(decisive[k]),
                        bool(right["FD"][k]), bool(right["BC"][k])])
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "layer", "mode", "gate", "up", "M_bits", "B_bits", "FD", "BC", "S_new"])
        for k, c in enumerate(cs):
            w.writerow([c["name"], c["layer"], c["mode"], c["gate"], c["up"], *resources(c), pred["FD"][k], pred["BC"][k], S_new[k]])
    jdump(out / "analysis.json", res)
    log(json.dumps(res["pass"]))


def stage_tables(args, out, log):
    a = jload(out / "analysis.json")
    L = [f"### 通過標準：{a['pass']}", "", f"bootstrap（FD − BC 正確率，|ΔS| > 1 的同資源配對）95% 區間 {a['bootstrap_accuracy_diff_FD_minus_BC']['ci95']}", ""]
    for sname, t in a["pairs"].items():
        L += [f"### {sname}", "", "| 類別 | 配對 | FD 對 | BC 對 | FD 修正／弄壞 |", "|---|---:|---:|---:|---|"]
        for k, v in t.items():
            if v["pairs"]:
                L.append(f"| {k} | {v['pairs']} | {v['FD']} | {v['BC']} | +{v['FD_fixed']}／−{v['FD_broken']} |")
        L.append("")
    L += ["### 組內 Spearman（新窗口 S）", "", "| 組 | FD | BC |", "|---|---:|---:|"]
    for g, v in a["spearman"].items():
        L.append(f"| {g} | {v['FD']:.3f} | {v['BC']:.3f} |")
    L += ["", f"成本：{a['cost']}", f"檢查：{a['checks']}"]
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "capture", "scores", "freeze", "truth", "analyze", "tables"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "plan":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "plan.json").exists():
        raise SystemExit("run --stage plan first")
    if args.stage == "truth" and not (out / "freeze.json").exists():
        raise SystemExit("scores are not frozen")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {}
    timing[args.stage] = timing.get(args.stage, 0.0) + time.time() - t0
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
