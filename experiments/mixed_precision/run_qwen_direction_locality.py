#!/usr/bin/env python3
"""Qwen3-0.6B Gate/Up candidates: why does the squared (Fisher-type) criterion misrank? Two hypotheses, tested on
the candidates, quantizers, cache and per-window losses of runs/qwen3_block_criterion_20261014 (post-hoc: those
windows were analyzed before).

H1  the squared proxy carries no direction and no cancellation: the loss change dL is signed (first order
    F = <gY, e> can be negative) and cancels across windows, positions and channels; a sum of squares cannot.
H2  full quantization is outside the local (second-order) approximation: dL(alpha = 1) differs from the exact local
    quadratic model of the loss along the quantization error.

Experiment 1 (all 332 candidates, 30 windows): gY on the tuning windows (one more backward pass), the exact first
order F_s = sum_{t<=126, j} gY_s e_s per window, its in-window cancellation |F_s| / sum |gY e|, the cancellation of
the actual loss across windows |S| / A (S = sum_s dL_s, A = sum_s |dL_s|), and calibration-built predictors fixed by
the Taylor expansion (no fitted coefficient):
  BC_total (native, squared)   Fsq = sum_cal F_s^2 (squared, at Y)   F_cal = sum_cal F_s (signed)
  T_Y = F_cal + 1/2 Fsq        T_BC = F_cal + 1/2 BC_total
ranked against the tuning S (signed) and A (magnitude).
Experiment 2 (activation-only candidates, tuning windows): the block output error e scaled by +-h (h = 1/8) and
replayed downstream; per window a1 = (dL(h) - dL(-h)) / 2h, a2 = (dL(h) + dL(-h)) / 2h^2 (exact local first and
second order of the loss along e), local prediction at alpha = 1: a1 + a2. Oracle predictors (from the tuning
windows themselves; explain, not deployable): O1 = sum(a1 + a2), O2 = sum a2 (squared part only), O3 = sum a1.
A candidate is "local" when |S - O1| <= 0.25 max(|S|, |O1|, TAU).

Stages (one new --out-dir): plan, grads, direction, local (--layers, resumable), analyze, tables.
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

from run_qwen_block_criterion import (CAL, LAYERS, SEQ, TAU, TUNE, Block, Downstream, Setup, candidates,  # noqa: E402
                                      jdump, jload, load_cache, load_rows)

BLOCK = HERE / "runs/qwen3_block_criterion_20261014"
H = 1 / 8
LOCAL_TOL = 0.25
P1 = ("BC_total", "Fsq", "F_cal", "T_Y", "T_BC")
P2 = ("O1_local", "O2_a2_only", "O3_a1_only", "O2_plus_cal_first_order", "BC_total", "T_BC")


def stage_plan(args, out, log):
    from qwen3_swiglu_common import environment, sha256
    plan = {"hypotheses": {"H1": "squared proxy carries no direction / cancellation", "H2": "full quantization is "
                           "outside the local second-order approximation"},
            "experiment_1": {"candidates": "all 332 (w, a, wa) of runs/qwen3_block_criterion_20261014",
                             "predictors_calibration": {"BC_total": "native W + A (squared)", "Fsq": "sum_cal F_s^2",
                                                        "F_cal": "sum_cal F_s (signed)", "T_Y": "F_cal + Fsq / 2",
                                                        "T_BC": "F_cal + BC_total / 2"},
                             "targets": {"S_tune": "sum over tuning windows of dL (signed)",
                                         "A_tune": "sum over tuning windows of |dL|"}},
            "experiment_2": {"candidates": "activation-only (140)", "windows": "tuning 10-29", "h": H,
                             "local_tolerance": LOCAL_TOL, "oracles": list(P2[:3])},
            "metrics": ["within (layer, mode) group Spearman", f"pair accuracy, |d target| > {TAU}, same pairs for "
                        "every predictor", "equal M and B pairs", "subgroups: mode, int4, local / non-local, |dS| size"],
            "post_hoc": True, "no_fitted_coefficients": True}
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    jdump(out / "plan.json", plan)
    jdump(out / "environment.json", {**environment(["experiments/mixed_precision/run_qwen_direction_locality.py",
                                                    "experiments/mixed_precision/run_qwen_block_criterion.py"]),
                                     "inputs_sha256": {p.name: sha256(p) for p in [BLOCK / "cache.npz", BLOCK / "scores.json",
                                                                                  BLOCK / "tokens.npy",
                                                                                  *[BLOCK / f"loss_L{l}.json" for l in LAYERS]]}})
    log("plan frozen")


def stage_grads(args, out, log):
    """gY at the four layers' MLP outputs for the tuning windows (and window 0 as a check against the cache)."""
    from qwen3_swiglu_common import summed_ce
    S = Setup(out, args.threads)
    torch = S.torch
    toks = np.load(BLOCK / "tokens.npy")
    cache = load_cache(BLOCK)
    S.model.model.embed_tokens.weight.requires_grad_(True)
    cap = {}

    def hook(layer):
        def f(module, inputs, output):
            output.retain_grad()
            cap[layer] = output
        return f
    hs = [S.mods[f"model.layers.{l}.mlp"].register_forward_hook(hook(l)) for l in LAYERS]
    gy = {l: [] for l in LAYERS}
    check, t0 = {}, time.time()
    for s in [0, *range(CAL, CAL + TUNE)]:
        ids = torch.from_numpy(toks[s])
        loss = summed_ce(S.wrapper(ids[None]), ids)
        loss.backward()
        for l in LAYERS:
            g = cap[l].grad[0].numpy().copy()
            if s == 0:
                ref = cache[f"L{l}|gy"][0]
                check[f"L{l}_window0_vs_cache_rel_max"] = float(np.abs(g - ref).max() / np.abs(ref).max())
            else:
                gy[l].append(g)
                y = cap[l][0].detach().numpy()
                check.setdefault(f"L{l}_y_vs_cache_rel_max", 0.0)
                ref = cache[f"L{l}|y"][s]
                check[f"L{l}_y_vs_cache_rel_max"] = max(check[f"L{l}_y_vs_cache_rel_max"],
                                                       float(np.abs(y - ref).max() / np.abs(ref).max()))
        S.model.zero_grad(set_to_none=True)
    for h in hs:
        h.remove()
    np.savez(out / "gy_tune.npz", **{f"L{l}": np.stack(v) for l, v in gy.items()})
    jdump(out / "grads.json", {"checks": check, "seconds": time.time() - t0})
    log(json.dumps(check))


def stage_direction(args, out, log):
    """Experiment 1 raw quantities: per candidate and window, F = <gY, e>, sum |gY e|, sum (gY e)^2."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    gt = np.load(out / "gy_tune.npz")
    res, t0 = {}, time.time()
    for layer in LAYERS:
        B = Block(S, cache, layer, slice(0, CAL + TUNE))
        gy = torch.from_numpy(np.concatenate([cache[f"L{layer}|gy"], gt[f"L{layer}"]])).double()[:, :SEQ - 1]
        with torch.no_grad():
            for c in [c for c in candidates() if c["layer"] == layer]:
                ge = gy * (B.y(c["gate"], c["up"]) - B.y_ref).double()[:, :SEQ - 1]
                res[c["name"]] = {"F": ge.sum((1, 2)).tolist(), "abs": ge.abs().sum((1, 2)).tolist(),
                                  "sq": (ge ** 2).sum((1, 2)).tolist()}
        log(f"layer {layer}: done ({time.time() - t0:.0f}s)")
    jdump(out / "direction.json", {"rows": res, "seconds": time.time() - t0})


def stage_local(args, out, log):
    """Experiment 2 raw runs: dL(+h e), dL(-h e) per tuning window, activation-only candidates; resumable."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")[CAL:CAL + TUNE]
    for layer in [int(x) for x in args.layers.split(",")]:
        f = out / f"local_L{layer}.json"
        done = jload(f) if f.exists() else {"rows": {}}
        t1 = time.time()
        B = Block(S, cache, layer, slice(CAL, CAL + TUNE))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"][CAL:CAL + TUNE])
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            ref = np.array(jload(BLOCK / f"loss_L{layer}.json")["fp32_replay_ce"][CAL:])
            done["base_vs_block_study_max_abs"] = float(np.abs(base - ref).max())
            for c in [c for c in candidates() if c["layer"] == layer and c["mode"] == "a"]:
                if c["name"] in done["rows"]:
                    continue
                e = B.y(c["gate"], c["up"]) - B.y_ref
                dp = D.run(hmid + B.y_ref + H * e) - base
                dm = D.run(hmid + B.y_ref - H * e) - base
                done["rows"][c["name"]] = {"plus": dp.tolist(), "minus": dm.tolist()}
                if len(done["rows"]) % 5 == 0:
                    jdump(f, done)
        done["seconds"] = done.get("seconds", 0.0) + time.time() - t1
        jdump(f, done)
        log(f"layer {layer}: {len(done['rows'])} candidates, base check {done['base_vs_block_study_max_abs']:.1e} "
            f"({time.time() - t1:.0f}s)")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    return None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y).statistic)


def act_formats(r):
    return {o.split("/")[1] for o in (r["gate"], r["up"]) if o != "-" and o.split("/")[1] != "fp32"}


def pair_table(rows, preds, target, cats):
    """Right / counted pairs per predictor and category; pairs inside (layer, mode) groups, |d target| > TAU."""
    idx = {}
    for k, r in enumerate(rows):
        idx.setdefault((r["layer"], r["mode"]), []).append(k)
    I, J = [], []
    for ks in idx.values():
        for a, b in itertools.combinations(ks, 2):
            I.append(a)
            J.append(b)
    I, J = np.array(I), np.array(J)
    dt = target[I] - target[J]
    keep = np.abs(dt) > TAU
    out = {}
    for cname, fn in cats.items():
        m = keep & np.array([fn(rows[a], rows[b], abs(d)) for a, b, d in zip(I, J, dt)])
        out[cname] = {"pairs": int(m.sum()), **{p: int(((np.sign(v[I] - v[J]) == np.sign(dt)) & m).sum())
                                                 for p, v in preds.items()}}
    return out


def within(rows, pred, target, mode=None):
    g = {}
    for k, r in enumerate(rows):
        if mode is None or r["mode"] == mode:
            g.setdefault(f"L{r['layer']}|{r['mode']}", []).append(k)
    vals = {k: spearman(pred[ix], target[ix]) for k, ix in sorted(g.items())}
    v = [x for x in vals.values() if x is not None]
    return {"mean": float(np.mean(v)), "by_group": vals}


def stage_analyze(args, out, log):
    rows = load_rows(BLOCK)
    by = {r["name"]: r for r in rows}
    D = jload(out / "direction.json")["rows"]
    res = {"post_hoc": True}
    # ---------------- experiment 1 ----------------
    F = np.array([D[r["name"]]["F"] for r in rows])                  # [cand, 30]
    ab = np.array([D[r["name"]]["abs"] for r in rows])
    dl = np.array([r["dL"] for r in rows])
    cal, tun = slice(0, CAL), slice(CAL, CAL + TUNE)
    S_t, A_t = dl[:, tun].sum(1), np.abs(dl[:, tun]).sum(1)
    big = A_t > TAU
    canc = np.abs(S_t[big]) / A_t[big]
    modes = np.array([r["mode"] for r in rows])
    has4 = np.array(["int4" in act_formats(r) or ("int4" in {o.split("/")[0] for o in (r["gate"], r["up"]) if o != "-"}) for r in rows])
    res["exp1_cancellation_across_windows"] = {
        "candidates_A_gt_tau": int(big.sum()), "abs_S_over_A_median": float(np.median(canc)),
        "share_abs_S_over_A_lt_0.5": float((canc < 0.5).mean()),
        "share_negative_S": float((S_t[big] < 0).mean()),
        "by_mode": {m: {"median": float(np.median(canc[modes[big] == m])), "share_lt_0.5": float((canc[modes[big] == m] < 0.5).mean()),
                        "share_negative_S": float((S_t[big][modes[big] == m] < 0).mean())} for m in ("w", "a", "wa")},
        "int4_vs_other": {k: {"median": float(np.median(canc[sel])), "share_lt_0.5": float((canc[sel] < 0.5).mean())}
                          for k, sel in (("int4", has4[big]), ("no_int4", ~has4[big]))}}
    Ft = F[:, tun]
    ratio_in = np.abs(Ft) / np.maximum(ab[:, tun], 1e-300)
    res["exp1_in_window_cancellation"] = {"abs_F_over_sum_abs_ge_median": float(np.median(ratio_in)),
                                          "p10": float(np.percentile(ratio_in, 10)), "p90": float(np.percentile(ratio_in, 90))}
    # first order against the actual per-window loss (tuning windows)
    w = np.abs(dl[:, tun]) > 0.05
    res["exp1_first_order_vs_actual"] = {
        "window_sign_agreement": float((np.sign(Ft[w]) == np.sign(dl[:, tun][w])).mean()),
        "share_of_abs_dL_left_after_first_order_median": float(np.median(
            np.abs(dl[:, tun] - Ft).sum(1)[big] / A_t[big])),
        "by_mode": {m: {"sign_agreement": float((np.sign(Ft[w & (modes[:, None] == m)]) == np.sign(dl[:, tun][w & (modes[:, None] == m)])).mean()),
                        "left_after_first_order_median": float(np.median((np.abs(dl[:, tun] - Ft).sum(1) / np.maximum(A_t, 1e-300))[big & (modes == m)]))}
                    for m in ("w", "a", "wa")}}
    preds = {"BC_total": np.array([r["BC_total"] for r in rows]),
             "Fsq": (F[:, cal] ** 2).sum(1), "F_cal": F[:, cal].sum(1)}
    preds["T_Y"] = preds["F_cal"] + 0.5 * preds["Fsq"]
    preds["T_BC"] = preds["F_cal"] + 0.5 * preds["BC_total"]
    cats = {"all": lambda a, b, d: True, "w": lambda a, b, d: a["mode"] == "w", "a": lambda a, b, d: a["mode"] == "a",
            "wa": lambda a, b, d: a["mode"] == "wa",
            "equal_M_B": lambda a, b, d: tuple(a["resources"]) == tuple(b["resources"]),
            "dS 1-20": lambda a, b, d: d <= 20, "dS 20-100": lambda a, b, d: 20 < d <= 100, "dS >100": lambda a, b, d: d > 100}
    res["exp1_pairs"] = {t: pair_table(rows, preds, tv, cats) for t, tv in (("S_tune", S_t), ("A_tune", A_t))}
    res["exp1_within_group"] = {t: {p: within(rows, v, tv)["mean"] for p, v in preds.items()}
                                for t, tv in (("S_tune", S_t), ("A_tune", A_t))}
    res["exp1_within_group_by_mode"] = {m: {p: within(rows, v, S_t, m)["mean"] for p, v in preds.items()} for m in ("w", "a", "wa")}
    # ---------------- experiment 2 ----------------
    loc = {}
    for l in LAYERS:
        f = out / f"local_L{l}.json"
        if f.exists():
            d = jload(f)
            loc.update(d["rows"])
            res.setdefault("exp2_base_checks", {})[f"L{l}"] = d["base_vs_block_study_max_abs"]
    if loc:
        ar = [r for r in rows if r["name"] in loc]
        k_of = {r["name"]: k for k, r in enumerate(rows)}
        P = np.array([loc[r["name"]]["plus"] for r in ar])
        M = np.array([loc[r["name"]]["minus"] for r in ar])
        a1, a2 = (P - M) / (2 * H), (P + M) / (2 * H * H)
        Fa = np.array([F[k_of[r["name"]], tun] for r in ar])
        S = np.array([r["S_tune"] for r in ar])
        O1, O2, O3 = (a1 + a2).sum(1), a2.sum(1), a1.sum(1)
        BC = np.array([r["BC_total"] for r in ar])
        TBC = np.array([preds["T_BC"][k_of[r["name"]]] for r in ar])
        dev = np.abs(S - O1) / np.maximum(np.maximum(np.abs(S), np.abs(O1)), TAU)
        local = dev <= LOCAL_TOL
        for r, lo, dv, o1 in zip(ar, local, dev, O1):
            r["local"], r["local_dev"], r["O1"] = bool(lo), float(dv), float(o1)
        sig = np.abs(a1) > 1e-3
        res["exp2_checks"] = {"a1_vs_gradient_F_rel_median": float(np.median(np.abs(a1 - Fa)[sig] / np.abs(Fa)[sig])),
                              "a1_vs_gradient_F_corr": float(np.corrcoef(a1.ravel(), Fa.ravel())[0, 1]),
                              "candidates": len(ar)}
        int4 = np.array(["int4" in act_formats(r) for r in ar])
        res["exp2_locality"] = {
            "local_share": float(local.mean()),
            "by_int4": {k: {"candidates": int(sel.sum()), "local_share": float(local[sel].mean()),
                            "S_over_O1_median": float(np.median((S / np.where(np.abs(O1) > 1e-9, O1, np.nan))[sel & (np.abs(O1) > TAU)]))}
                        for k, sel in (("int4", int4), ("no_int4", ~int4))},
            "by_layer": {f"L{l}": {"local_share": float(local[np.array([r["layer"] == l for r in ar])].mean())}
                         for l in LAYERS if any(r["layer"] == l for r in ar)},
            "nonlocal_cases": sorted([[r["name"], r["S_tune"], r["O1"], r["local_dev"]] for r in ar if not r["local"]],
                                     key=lambda x: -abs(x[1] - x[2]))[:30],
            "first_order_share_of_O1_median": float(np.median(np.abs(O3) / np.maximum(np.abs(O3) + np.abs(O2), 1e-300))),
            "a1_negative_share": float((O3 < 0).mean())}
        preds2 = {"O1_local": O1, "O2_a2_only": O2, "O3_a1_only": O3, "BC_total": BC, "T_BC": TBC}
        cats2 = {"all": lambda a, b, d: True,
                 "both local": lambda a, b, d: a["local"] and b["local"],
                 "any non-local": lambda a, b, d: not (a["local"] and b["local"]),
                 "equal_M_B": lambda a, b, d: tuple(a["resources"]) == tuple(b["resources"]),
                 "int4 both": lambda a, b, d: "int4" in act_formats(a) and "int4" in act_formats(b),
                 "int4 none": lambda a, b, d: "int4" not in act_formats(a) and "int4" not in act_formats(b),
                 "dS 1-20": lambda a, b, d: d <= 20, "dS 20-100": lambda a, b, d: 20 < d <= 100, "dS >100": lambda a, b, d: d > 100}
        # can the calibration first order stand in for the tuning windows' own? (systematic part of the direction)
        Fcal2 = np.array([F[k_of[r["name"]], cal].sum() for r in ar]) * (TUNE / CAL)
        preds2["O2_plus_cal_first_order"] = O2 + Fcal2
        res["exp2_first_order_transfer"] = {}
        for m in ("w", "a", "wa"):
            ks = [k for k, r in enumerate(rows) if r["mode"] == m]
            fc, ft = F[ks][:, cal].sum(1) * (TUNE / CAL), F[ks][:, tun].sum(1)
            bigf = np.abs(ft) > TAU
            res["exp2_first_order_transfer"][m] = {"pearson": float(np.corrcoef(fc, ft)[0, 1]), "spearman": spearman(fc, ft),
                                                   "sign_agree_abs_F_tune_gt_tau": float((np.sign(fc[bigf]) == np.sign(ft[bigf])).mean())}
        res["exp2_pairs"] = pair_table(ar, preds2, S, cats2)
        # every pair BC orders wrongly, attributed: O1 (exact local) wrong too -> non-local; O1 right but O2 (no
        # direction) wrong -> direction; O1 and O2 right -> approximation of the second-order term
        idx2 = {}
        for k, r in enumerate(ar):
            idx2.setdefault((r["layer"], r["mode"]), []).append(k)
        att = {}
        for a_, b_ in [(x, y) for ks in idx2.values() for x, y in itertools.combinations(ks, 2)]:
            dt_ = S[a_] - S[b_]
            if abs(dt_) <= TAU:
                continue
            ok = lambda v: np.sign(v[a_] - v[b_]) == np.sign(dt_)
            if ok(BC):
                continue
            cls = "non-local" if not ok(O1) else "direction" if not ok(O2) else "approximation"
            sub = "int4 both" if "int4" in act_formats(ar[a_]) and "int4" in act_formats(ar[b_]) else \
                  "int4 one" if "int4" in act_formats(ar[a_]) or "int4" in act_formats(ar[b_]) else "int4 none"
            for key in ("all", sub, f"L{ar[a_]['layer']}"):
                att.setdefault(key, {"non-local": 0, "direction": 0, "approximation": 0})[cls] += 1
        res["exp2_bc_error_attribution"] = att
        res["exp2_within_group"] = {p: within(ar, v, S) for p, v in preds2.items()}
        # the int4 Gate-vs-Up decisions
        dec = {}
        for l in LAYERS:
            g, u = f"L{l}|a|g=fp32/int4|u=-", f"L{l}|a|g=-|u=fp32/int4"
            if g in loc and u in loc:
                ig, iu = [k for k, r in enumerate(ar) if r["name"] == g][0], [k for k, r in enumerate(ar) if r["name"] == u][0]
                truth = np.sign(S[ig] - S[iu])
                dec[f"L{l}"] = {"S_gate_minus_up": float(S[ig] - S[iu]),
                                **{p: {"gate": float(v[ig]), "up": float(v[iu]), "right": bool(np.sign(v[ig] - v[iu]) == truth)}
                                   for p, v in preds2.items()},
                                "gate_local": ar[ig]["local"], "up_local": ar[iu]["local"]}
        res["exp2_int4_gate_vs_up"] = dec
    jdump(out / "analysis.json", res)
    log("analysis written")


def stage_tables(args, out, log):
    a = jload(out / "analysis.json")
    L = ["## 實驗 1：方向與抵消（全部 332 候選；事後）", ""]
    c = a["exp1_cancellation_across_windows"]
    L += [f"- 窗口間抵消（tuning，A > {TAU:g} 的 {c['candidates_A_gt_tau']} 個候選）：|S|/A 中位數 {c['abs_S_over_A_median']:.3f}，"
          f"< 0.5 的比例 {c['share_abs_S_over_A_lt_0.5']:.3f}，S < 0 的比例 {c['share_negative_S']:.3f}",
          "  - 依模式：" + "；".join(f"{m} 中位數 {v['median']:.3f}、<0.5 {v['share_lt_0.5']:.3f}、S<0 {v['share_negative_S']:.3f}"
                                 for m, v in c["by_mode"].items()),
          "  - int4 與否：" + "；".join(f"{k} 中位數 {v['median']:.3f}、<0.5 {v['share_lt_0.5']:.3f}" for k, v in c["int4_vs_other"].items())]
    w = a["exp1_in_window_cancellation"]
    L += [f"- 窗口內（位置 × channel）抵消：|F_s| / Σ|gY·e| 中位數 {w['abs_F_over_sum_abs_ge_median']:.4f}（p10 {w['p10']:.4f}，p90 {w['p90']:.4f}）"]
    f = a["exp1_first_order_vs_actual"]
    L += [f"- 一階 F_s 與實際 ΔL_s（tuning，|ΔL_s| > 0.05 的窗口）：正負號一致 {f['window_sign_agreement']:.3f}；"
          f"扣掉一階後剩下的 Σ|ΔL − F| / A 中位數 {f['share_of_abs_dL_left_after_first_order_median']:.3f}",
          "  - 依模式：" + "；".join(f"{m} 一致 {v['sign_agreement']:.3f}、剩下 {v['left_after_first_order_median']:.3f}" for m, v in f["by_mode"].items())]
    L += ["", "| 預測量（calibration） | 組內 Spearman vs S_tune | vs A_tune | w 組 vs S | a 組 vs S | wa 組 vs S |", "|---|---:|---:|---:|---:|---:|"]
    for p in P1:
        L.append(f"| {p} | {a['exp1_within_group']['S_tune'][p]:.3f} | {a['exp1_within_group']['A_tune'][p]:.3f} | "
                 + " | ".join(f"{a['exp1_within_group_by_mode'][m][p]:.3f}" for m in ("w", "a", "wa")) + " |")
    for t in ("S_tune", "A_tune"):
        T = a["exp1_pairs"][t]
        L += ["", f"| 配對（target {t}，|Δ| > {TAU:g}） | 配對數 | " + " | ".join(P1) + " |", "|---|---:|" + "---:|" * len(P1)]
        for cname, v in T.items():
            L.append(f"| {cname} | {v['pairs']} | " + " | ".join(f"{v[p]}" for p in P1) + " |")
    if "exp2_pairs" in a:
        L += ["", "## 實驗 2：局部近似範圍（activation-only；tuning；事後）", ""]
        ch, lo = a["exp2_checks"], a["exp2_locality"]
        L += [f"- 檢查：±h 對稱差分的 a1 與梯度 F 的相對差中位數 {ch['a1_vs_gradient_F_rel_median']:.3f}，相關 {ch['a1_vs_gradient_F_corr']:.4f}；"
              f"base 與 block 研究一致（最大差 {max(a['exp2_base_checks'].values()):.1e}）；候選 {ch['candidates']}",
              f"- 局部範圍內（|S − O1| ≤ {LOCAL_TOL:g}·max）的比例 {lo['local_share']:.3f}；" + "；".join(
                  f"{k} {v['local_share']:.3f}（{v['candidates']} 個，S/O1 中位數 {v['S_over_O1_median']:.2f}）" for k, v in lo["by_int4"].items()),
              "  - 依層：" + "；".join(f"{k} {v['local_share']:.3f}" for k, v in lo["by_layer"].items()),
              f"- 局部模型中一階的份量 |O3|/(|O3|+|O2|) 中位數 {lo['first_order_share_of_O1_median']:.3f}；O3 < 0 的比例 {lo['a1_negative_share']:.3f}", "",
              "| 配對（tuning S） | 配對數 | " + " | ".join(P2) + " |", "|---|---:|" + "---:|" * len(P2)]
        for cname, v in a["exp2_pairs"].items():
            L.append(f"| {cname} | {v['pairs']} | " + " | ".join(f"{v[p]}" for p in P2) + " |")
        L += ["", "| 組內 Spearman（tuning S） | " + " | ".join(P2) + " |", "|---|" + "---:|" * len(P2)]
        groups = list(a["exp2_within_group"]["O1_local"]["by_group"])
        for g in groups:
            L.append(f"| {g} | " + " | ".join(f"{a['exp2_within_group'][p]['by_group'][g]:.3f}" for p in P2) + " |")
        L.append("| 平均 | " + " | ".join(f"{a['exp2_within_group'][p]['mean']:.3f}" for p in P2) + " |")
        L += ["", "| int4 Gate vs Up | 實際 Gate − Up | " + " | ".join(P2) + " | Gate／Up 在局部範圍內 |", "|---|---:|" + "---|" * len(P2) + "---|"]
        for l, d in a["exp2_int4_gate_vs_up"].items():
            L.append(f"| {l} | {d['S_gate_minus_up']:+.0f} | " + " | ".join(
                f"{d[p]['gate']:.0f} vs {d[p]['up']:.0f}（{'對' if d[p]['right'] else '錯'}）" for p in P2) +
                f" | {d['gate_local']}／{d['up_local']} |")
        L += ["", "| BC 排錯的配對歸因 | 非局部（O1 也錯） | 方向（O1 對、O2 錯） | 二階近似（O1、O2 都對） |", "|---|---:|---:|---:|"]
        for k, v in a["exp2_bc_error_attribution"].items():
            L.append(f"| {k} | {v['non-local']} | {v['direction']} | {v['approximation']} |")
        L += ["", "| 一階項可轉移性（calibration ×2 vs tuning，全部候選） | Pearson | Spearman | 正負號一致（|F_tune| > 1） |", "|---|---:|---:|---:|"]
        for m, v in a["exp2_first_order_transfer"].items():
            L.append(f"| {m} | {v['pearson']:.3f} | {v['spearman']:.3f} | {v['sign_agree_abs_F_tune_gt_tau']:.3f} |")
        L += ["", "| 超出局部範圍的候選（依 |S − O1| 排序） | S_tune | O1 局部預測 | 偏差 |", "|---|---:|---:|---:|"]
        for n, s, o, dv in lo["nonlocal_cases"][:15]:
            L.append(f"| {n.replace('|', ' ')} | {s:+.1f} | {o:+.1f} | {dv:.2f} |")
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "grads", "direction", "local", "analyze", "tables"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--layers", default="27,14,2,0", help="local: layers to run")
    args = ap.parse_args()
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
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {}
    timing[args.stage] = timing.get(args.stage, 0.0) + time.time() - t0
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
