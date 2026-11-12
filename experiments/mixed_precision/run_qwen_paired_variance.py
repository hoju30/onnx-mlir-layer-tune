#!/usr/bin/env python3
"""Qwen3-0.6B: does pairing the gradient and the centered error at the same position (V_paired) instead of
multiplying separately pooled sums (V_old) improve the ranking of quantization candidates? A post-hoc diagnostic on
the candidates, quantizers, cache and per-window losses of runs/qwen3_block_criterion_20261014 (whose calibration and
tuning windows were already analyzed); nothing of the native criterion or the search is changed.

Symbols (report): W weight criterion, A activation criterion, C = W + A, M model size, B BOPs. In table.json the
weight criterion is stored as "B" and the activation criterion as "C" (pipeline naming).

Per layer L, branch (Gate / Up), activation tap (x: the branch's copy of the block input X, gradient = the total
input gradient g_x(gate) + g_x(up); y: the projection output zg / zu) and activation format, on the calibration
windows s (10) and all positions t (128, including the last one without a target, as the native statistics),
channels c, in float64:
  d[s,t,c] = QDQ(x) - x                        (the pass's QDQ, int (scale, zp) from the calibration ranges)
  g[s,t,c] = d(FP32 summed CE of window s) / dx  (from the cached gY through the FP32 block)
  n = S*T, mu[c] = sum_{s,t} d / n, r = d - mu
  G2[c] = sum_{s,t} g^2, GS[c] = sum_s (sum_t g)^2
  V_old    = sum_c G2[c] * sum_{s,t} r^2 / n          (the native variance term)
  U        = sum_c GS[c] * mu[c]^2                    (the native mean / bias term)
  V_paired = sum_{s,t,c} g^2 r^2                      (same position, centered; not divided by n)
  raw diagonal (earlier study) = sum g^2 d^2
  V_shuffled: r permuted uniformly over all S*T positions inside each channel (20 fixed seeds); E = V_old
Identity per channel: V_paired,c - V_old,c = n * Cov_population(g^2, r^2).
Candidate costs (summed over the quantized branches and both taps, W from table.json "B"):
  A  W + V_old + U    (= the native C = W + A)      B  W + V_paired + U
  C  W + V_old                                        D  W + V_paired
  E  W + raw diagonal (earlier study)                 shuffled: W + V_shuffled + U, W + V_shuffled
Evaluation against the existing signed summed-CE loss changes (S = sum over windows of dL; mean NLL per token =
S / (windows * 127)): inside each (layer, mode) group (activation-only main, weight+activation supplementary),
Spearman; every candidate pair with |dS| > TAU = 1, the same pairs for all methods; equal M and B pairs; fixed /
broken against A; Gate/Up decisions; INT4, single / joint and |dS| subgroups; paired window bootstrap.

Stages (one new --out-dir): freeze, compute, evaluate, tables.
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_qwen_block_criterion import CAL, LAYERS, SEQ, TABLES, TAU, TUNE, Setup, jdump, jload, load_cache, load_rows  # noqa: E402

BLOCK = HERE / "runs/qwen3_block_criterion_20261014"
DECOMP = HERE / "runs/qwen3_error_decomposition_20261014"
FORMATS = ("int4", "int8", "p6e1", "p8e1", "bf16")
SEEDS = tuple(range(20))
BOOT = 2000
METHODS = {"A": "W + V_old + U（原完整）", "B": "W + V_paired + U（新完整）", "C": "W + V_old（原 variance-only）",
           "D": "W + V_paired（新 variance-only）", "E": "W + Σg²d²（舊 raw diagonal）"}


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
def tap_terms(d, g):
    """All per-channel / per-position terms of one tap ([S, T, C] float64 tensors)."""
    S_, T_, _ = d.shape
    n = S_ * T_
    mu = d.sum((0, 1)) / n
    r = d - mu
    g2, r2 = g ** 2, r ** 2
    G2, GS = g2.sum((0, 1)), (g.sum(1) ** 2).sum(0)
    var = r2.sum((0, 1)) / n
    paired = (g2 * r2).sum((0, 1))
    return {"n": n, "V_old_c": G2 * var, "V_paired_c": paired, "U_c": GS * mu ** 2,
            "cov_c": n * ((g2 * r2).mean((0, 1)) - g2.mean((0, 1)) * r2.mean((0, 1))),
            "diag_c": (g2 * d ** 2).sum((0, 1)),
            "pos_paired": (g2 * r2).sum((0, 2)),               # [T], additive: sums to V_paired
            "pos_old_grad": (g2 * var).sum((0, 2)),            # [T], gradient side of V_old: sums to V_old
            "pos_old_err": (G2 * r2).sum((0, 2)) / n,          # [T], error side of V_old: sums to V_old
            "g2_flat": g2.reshape(n, -1), "r2_flat": r2.reshape(n, -1)}


def shuffled(g2, r2, tap_id, torch):
    """V_shuffled per seed and channel: r^2 permuted uniformly over all n positions inside each channel."""
    out = []
    for k in SEEDS:
        gen = torch.Generator().manual_seed(1_000_003 * k + tap_id)
        idx = torch.argsort(torch.rand(g2.shape, generator=gen, dtype=torch.float64), dim=0)
        out.append((g2 * torch.gather(r2, 0, idx)).sum(0))
    return torch.stack(out)                                       # [seeds, C]


def numeric_checks(torch):
    """The formulas on synthetic tensors: zero error, channel-constant error, channel-constant gradient."""
    gen = torch.Generator().manual_seed(7)
    S_, T_, C_ = 4, 16, 8
    g = torch.randn(S_, T_, C_, generator=gen, dtype=torch.float64)
    d = torch.randn(S_, T_, C_, generator=gen, dtype=torch.float64)
    z = tap_terms(torch.zeros_like(d), g)
    k = tap_terms(torch.ones_like(d) * torch.randn(C_, generator=gen, dtype=torch.float64), g)
    gc = torch.ones_like(g) * torch.randn(C_, generator=gen, dtype=torch.float64)
    c = tap_terms(d, gc)
    f = lambda t: float(t.abs().max())
    return {"zero_error": {"V_old": f(z["V_old_c"]), "V_paired": f(z["V_paired_c"]), "U": f(z["U_c"])},
            "channel_constant_error": {"V_old": f(k["V_old_c"]), "V_paired": f(k["V_paired_c"]),
                                       "U_positive": bool((k["U_c"] > 0).all())},
            "channel_constant_gradient_rel": f((c["V_paired_c"] - c["V_old_c"]) / c["V_old_c"]),
            "passed": f(z["V_old_c"]) == f(z["V_paired_c"]) == f(z["U_c"]) == 0.0
            and f(k["V_old_c"]) < 1e-24 and f(k["V_paired_c"]) < 1e-24
            and f((c["V_paired_c"] - c["V_old_c"]) / c["V_old_c"]) < 1e-12}


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_freeze(args, out, log):
    from qwen3_swiglu_common import environment, sha256
    rows = [r for r in load_rows(BLOCK) if "a" in r["mode"]]
    cfg = {"layers": list(LAYERS), "activation_formats": list(FORMATS), "taps": ["x (total input gradient)", "y"],
           "windows": {"calibration": [0, CAL - 1], "tuning": [CAL, CAL + TUNE - 1], "positions": SEQ},
           "tau_summed_ce": TAU, "seeds": list(SEEDS), "bootstrap_replicates": BOOT, "methods": METHODS,
           "candidates": {"total": len(rows), "by_group": {f"L{l}|{m}": sum(r["layer"] == l and r["mode"] == m for r in rows)
                                                           for l in LAYERS for m in ("a", "wa")},
                          "names": [r["name"] for r in rows]},
           "post_hoc": "calibration and tuning windows were analyzed by earlier studies; not an independent validation",
           "naming": {"table.json B": "W (weight criterion)", "table.json C": "A (activation criterion)",
                      "report M": "model (weight) size", "report B": "BOPs"}}
    jdump(out / "config.json", cfg)
    files = {p: sha256(p) for p in [BLOCK / "cache.npz", BLOCK / "scores.json", BLOCK / "tokens.npy",
                                    *[BLOCK / f"loss_L{l}.json" for l in LAYERS], TABLES / "table.json",
                                    TABLES / "ranges_by_point.json", DECOMP / "attribution.json", DECOMP / "variance.json"]}
    jdump(out / "environment.json", {**environment(["experiments/mixed_precision/run_qwen_paired_variance.py",
                                                    "experiments/mixed_precision/run_qwen_block_criterion.py",
                                                    "experiments/mixed_precision/run_qwen_error_decomposition.py"]),
                                     "inputs_sha256": {str(k.relative_to(HERE)): v for k, v in files.items()}})
    log(f"frozen: {len(rows)} candidates")


def stage_compute(args, out, log):
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    checks = {"synthetic": numeric_checks(torch)}
    arrays, scalars, t0, tap_id = {}, {}, time.time(), 0
    cov_err, z_scores, nonfinite = [], [], 0
    for layer in LAYERS:
        f = lambda k: torch.from_numpy(cache[f"L{layer}|{k}"][:CAL]).double()
        zg, zu, gy = f("zg"), f("zu"), f("gy")
        W = {k: S.mlp(layer, k).weight.double() for k in ("gate", "up", "down")}
        with torch.no_grad():
            gh = gy @ W["down"]
            sg = torch.sigmoid(zg)
            g_z = {"gate": gh * zu * (sg + zg * sg * (1 - sg)), "up": gh * zg * sg}
            g_x = g_z["gate"] @ W["gate"] + g_z["up"] @ W["up"]             # total input gradient
            Xf = torch.from_numpy(cache[f"L{layer}|x"][:CAL])
            Zf = {"gate": torch.from_numpy(cache[f"L{layer}|zg"][:CAL]), "up": torch.from_numpy(cache[f"L{layer}|zu"][:CAL])}
            # Gate's and Up's copies of X have one range (one tensor): one x tap per format, used by both branches
            rx = {k: S.ranges[S.node[(layer, k)]]["x"] for k in ("gate", "up")}
            assert rx["gate"] == rx["up"], (layer, rx)
            checks.setdefault("fp32_qdq_identity", []).append(bool(torch.equal(S.act(Xf, layer, "gate", "x", "fp32"), Xf)))
            for fmt in FORMATS:
                taps = {"x": ((S.act(Xf, layer, "gate", "x", fmt) - Xf).double(), g_x)}
                for kind in ("gate", "up"):
                    taps[f"y_{kind}"] = ((S.act(Zf[kind], layer, kind, "y", fmt) - Zf[kind]).double(), g_z[kind])
                for tap, (d, g) in taps.items():
                    t = tap_terms(d, g)
                    sh = shuffled(t["g2_flat"], t["r2_flat"], tap_id, torch)
                    tap_id += 1
                    key = f"L{layer}|{tap}|{fmt}"
                    for a in ("V_old_c", "V_paired_c", "U_c", "cov_c", "diag_c", "pos_paired", "pos_old_grad", "pos_old_err"):
                        arrays[f"{key}|{a}"] = t[a].numpy()
                    arrays[f"{key}|V_shuffled_cs"] = sh.numpy()
                    tot = {k: float(t[k].sum()) for k in ("V_old_c", "V_paired_c", "U_c", "diag_c")}
                    sh_tot = sh.sum(1).numpy()
                    se = float(sh_tot.std(ddof=1) / np.sqrt(len(SEEDS)))
                    scalars[key] = {"V_old": tot["V_old_c"], "V_paired": tot["V_paired_c"], "U": tot["U_c"],
                                    "raw_diag": tot["diag_c"], "V_shuffled_seeds": sh_tot.tolist(),
                                    "V_shuffled_mean": float(sh_tot.mean()), "V_shuffled_se": se,
                                    "pos0_share_paired": float(t["pos_paired"][0] / max(tot["V_paired_c"], 1e-300)),
                                    "pos0_share_old_grad": float(t["pos_old_grad"][0] / max(tot["V_old_c"], 1e-300)),
                                    "pos0_share_old_err": float(t["pos_old_err"][0] / max(tot["V_old_c"], 1e-300))}
                    lhs = t["V_paired_c"] - t["V_old_c"]
                    scale = torch.maximum(t["V_paired_c"].abs(), t["V_old_c"].abs()).max()
                    cov_err.append(float((lhs - t["cov_c"]).abs().max() / max(float(scale), 1e-300)))
                    if se > 0:
                        z_scores.append((sh_tot.mean() - tot["V_old_c"]) / se)
                    nonfinite += sum(int((~np.isfinite(arrays[f"{key}|{a}"])).sum())
                                     for a in ("V_old_c", "V_paired_c", "U_c", "diag_c", "V_shuffled_cs"))
        log(f"layer {layer}: {len(FORMATS) * 3} taps x formats ({time.time() - t0:.0f}s)")
    z = np.array(z_scores)
    checks.update({"covariance_identity_max_rel": float(max(cov_err)), "shuffle_z_abs_max": float(np.abs(z).max()),
                   "shuffle_z_abs_gt3": int((np.abs(z) > 3).sum()), "shuffle_cases": int(len(z)),
                   "shuffle_z_mean": float(z.mean()), "nonfinite": int(nonfinite),
                   "fp32_qdq_identity": all(checks["fp32_qdq_identity"])})
    np.savez_compressed(out / "terms.npz", **arrays)
    jdump(out / "terms.json", {"taps": scalars, "checks": checks, "seconds": time.time() - t0})
    log(json.dumps(checks))


def candidate_costs(out):
    from qwen3_swiglu_common import node_mapping
    T = jload(out / "terms.json")["taps"]
    table = jload(TABLES / "table.json")
    node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    rows = [r for r in load_rows(BLOCK) if "a" in r["mode"]]
    cost = {}
    for r in rows:
        Wt, parts = 0.0, {k: 0.0 for k in ("V_old", "V_paired", "U", "raw_diag")}
        sh = np.zeros(len(SEEDS))
        native_A = 0.0
        for kind in ("gate", "up"):
            if r[kind] == "-":
                continue
            w, a = r[kind].split("/")
            Wt += table["B"][node[(r["layer"], kind)]][w]
            native_A += table["C"][node[(r["layer"], kind)]][a]
            for tap in ("x", f"y_{kind}"):
                t = T[f"L{r['layer']}|{tap}|{a}"]
                for k in parts:
                    parts[k] += t[k]
                sh += np.array(t["V_shuffled_seeds"])
        cost[r["name"]] = {"W": Wt, **parts, "native_A": native_A, "BC_total": r["BC_total"],
                           "A": Wt + parts["V_old"] + parts["U"], "B": Wt + parts["V_paired"] + parts["U"],
                           "C": Wt + parts["V_old"], "D": Wt + parts["V_paired"], "E": Wt + parts["raw_diag"],
                           "shuffled_full": (Wt + sh + parts["U"]).tolist(), "shuffled_var": (Wt + sh).tolist()}
    return rows, cost


def act_formats(r):
    return {o.split("/")[1] for o in (r["gate"], r["up"]) if o != "-"}


def build_pairs(rows):
    """Every unordered candidate pair inside a (layer, mode) group, with its categories."""
    idx = {r["name"]: i for i, r in enumerate(rows)}
    P = []
    for (l, m) in itertools.product(LAYERS, ("a", "wa")):
        rs = [r for r in rows if r["layer"] == l and r["mode"] == m]
        for a, b in itertools.combinations(rs, 2):
            n4 = ("int4" in act_formats(a)) + ("int4" in act_formats(b))
            kind = {0: "both_single", 1: "single_joint", 2: "both_joint"}[(a["target"] == "both") + (b["target"] == "both")]
            branch = (a["target"] != "both" and b["target"] != "both" and a["target"] != b["target"]
                      and (a["gate"] if a["target"] == "gate" else a["up"]) == (b["gate"] if b["target"] == "gate" else b["up"]))
            swap = (a["target"] == b["target"] == "both" and a["gate"] == b["up"] and a["up"] == b["gate"] and a["gate"] != a["up"])
            P.append({"group": f"L{l}|{m}", "mode": m, "i": idx[a["name"]], "j": idx[b["name"]], "a": a["name"], "b": b["name"],
                      "equal_resource": tuple(a["resources"]) == tuple(b["resources"]),
                      "int4": {0: "none", 1: "one", 2: "both"}[n4], "single_joint": kind,
                      "gate_up_branch": bool(branch), "gate_up_swap": bool(swap)})
    return P


def accuracy(pred, target, P, mask=None):
    """Right / counted pairs: sign(pred_i - pred_j) == sign(target_i - target_j), |dtarget| > TAU."""
    i = np.array([p["i"] for p in P])
    j = np.array([p["j"] for p in P])
    dt = target[i] - target[j]
    keep = np.abs(dt) > TAU
    if mask is not None:
        keep &= mask
    right = np.sign(pred[i] - pred[j]) == np.sign(dt)
    return int((right & keep).sum()), int(keep.sum()), right, keep


def spearman_groups(pred, target, rows, mode=None):
    from scipy.stats import spearmanr
    out = {}
    for (l, m) in itertools.product(LAYERS, ("a", "wa")):
        if mode and m != mode:
            continue
        ix = [k for k, r in enumerate(rows) if r["layer"] == l and r["mode"] == m]
        out[f"L{l}|{m}"] = float(spearmanr(pred[ix], target[ix]).statistic)
    return out


def stage_evaluate(args, out, log):
    t0 = time.time()
    rows, cost = candidate_costs(out)
    names = [r["name"] for r in rows]
    pred = {m: np.array([cost[n][m] for n in names]) for m in ("A", "B", "C", "D", "E", "BC_total")}
    shf = np.array([cost[n]["shuffled_full"] for n in names])        # [cand, seeds]
    shv = np.array([cost[n]["shuffled_var"] for n in names])
    target = {"tune": np.array([r["S_tune"] for r in rows]), "cal": np.array([r["S_cal"] for r in rows])}
    dl = np.array([r["dL"] for r in rows])                           # [cand, 30]
    P = build_pairs(rows)
    res = {"candidates": len(rows), "pairs_total": len(P)}
    # baseline consistency: A vs the native activation criterion and the old evaluation
    nat = np.array([cost[n]["native_A"] for n in names])
    rec = np.array([cost[n]["V_old"] + cost[n]["U"] for n in names])
    ok = nat > 0
    res["baseline"] = {"A_activation_vs_native_C_rel_median": float(np.median(np.abs(rec[ok] - nat[ok]) / nat[ok])),
                       "A_activation_vs_native_C_rel_max": float(np.max(np.abs(rec[ok] - nat[ok]) / nat[ok])),
                       "A_vs_BC_total_rel_max": float(np.max(np.abs(pred["A"] - pred["BC_total"]) / pred["BC_total"])),
                       "W_U_identical_in_A_and_B": bool(np.allclose(pred["A"] - pred["B"],
                            np.array([cost[n]["V_old"] - cost[n]["V_paired"] for n in names]), rtol=0, atol=1e-9 * pred["A"].max()))}
    old = jload(DECOMP / "attribution.json")
    oldv = jload(DECOMP / "variance.json")["variants"]
    res["baseline"]["old_evaluation"] = {
        "attribution_A_tune_all": old["S_tune|all_pairs"]["total"]["bc_right"],
        "attribution_C_tune_all": old["S_tune|all_pairs"]["total"]["var_right"],
        "attribution_A_tune_equal": old["S_tune|equal_resource"]["total"]["bc_right"],
        "attribution_C_tune_equal": old["S_tune|equal_resource"]["total"]["var_right"],
        "variance_stage_raw_diag_tune_all": oldv["逐元素 Σg²d²（全部）"]["S_tune|all_pairs"]["right"]}
    # pair metrics
    cats = {"all": None, "equal_resource": np.array([p["equal_resource"] for p in P]),
            "activation_only": np.array([p["mode"] == "a" for p in P]),
            "weight_activation": np.array([p["mode"] == "wa" for p in P]),
            "gate_up_branch": np.array([p["gate_up_branch"] for p in P]), "gate_up_swap": np.array([p["gate_up_swap"] for p in P]),
            "int4_both": np.array([p["int4"] == "both" for p in P]), "int4_one": np.array([p["int4"] == "one" for p in P]),
            "int4_none": np.array([p["int4"] == "none" for p in P]),
            "both_single": np.array([p["single_joint"] == "both_single" for p in P]),
            "single_joint": np.array([p["single_joint"] == "single_joint" for p in P]),
            "both_joint": np.array([p["single_joint"] == "both_joint" for p in P])}
    for g in sorted({p["group"] for p in P}):
        cats[f"group {g}"] = np.array([p["group"] == g for p in P])
    pairs_out = {}
    for tname, tgt in target.items():
        i = np.array([p["i"] for p in P])
        j = np.array([p["j"] for p in P])
        adt = np.abs(tgt[i] - tgt[j])
        size = {"dS 1-5": (adt > 1) & (adt <= 5), "dS 5-20": (adt > 5) & (adt <= 20), "dS 20-100": (adt > 20) & (adt <= 100),
                "dS >100": adt > 100}
        table = {}
        flags = {}
        for cname, mask in {**cats, **size}.items():
            row = {}
            for m in ("A", "B", "C", "D", "E", "BC_total"):
                r_, n_, right, keep = accuracy(pred[m], tgt, P, mask)
                row[m] = {"right": r_, "pairs": n_}
                flags[m] = (right, keep)
            sh = [accuracy(shf[:, k], tgt, P, mask)[0] for k in range(len(SEEDS))]
            shv_ = [accuracy(shv[:, k], tgt, P, mask)[0] for k in range(len(SEEDS))]
            row["shuffled_full"] = {"mean": float(np.mean(sh)), "min": int(min(sh)), "max": int(max(sh))}
            row["shuffled_var"] = {"mean": float(np.mean(shv_)), "min": int(min(shv_)), "max": int(max(shv_))}
            for new, base in (("B", "A"), ("D", "C"), ("E", "C")):
                rn, kn = flags[new]
                rb, _ = flags[base]
                row[f"{new}_vs_{base}"] = {"fixed": int((rn & ~rb & kn).sum()), "broken": int((~rn & rb & kn).sum())}
            table[cname] = row
        pairs_out[tname] = table
        res[f"pairs|{tname}"] = table
        res[f"spearman|{tname}"] = {m: spearman_groups(pred[m], tgt, rows) for m in ("A", "B", "C", "D", "E")}
        res[f"spearman|{tname}"]["shuffled_full_mean"] = {
            g: float(np.mean([spearman_groups(shf[:, k], tgt, rows)[g] for k in range(len(SEEDS))]))
            for g in res[f"spearman|{tname}"]["A"]}
    # fixed / broken cases (B vs A, tuning and calibration), all pairs
    i = np.array([p["i"] for p in P])
    j = np.array([p["j"] for p in P])
    cases = []
    for k, p in enumerate(P):
        rec_ = {"group": p["group"], "a": p["a"], "b": p["b"], "dS_tune": float(target["tune"][i[k]] - target["tune"][j[k]]),
                "dS_cal": float(target["cal"][i[k]] - target["cal"][j[k]]), "equal_resource": p["equal_resource"],
                "int4": p["int4"], "single_joint": p["single_joint"], "gate_up_branch": p["gate_up_branch"],
                "gate_up_swap": p["gate_up_swap"]}
        for m in ("A", "B", "C", "D", "E"):
            dp = pred[m][i[k]] - pred[m][j[k]]
            rec_[f"right_{m}_tune"] = bool(np.sign(dp) == np.sign(rec_["dS_tune"])) if abs(rec_["dS_tune"]) > TAU else None
            rec_[f"right_{m}_cal"] = bool(np.sign(dp) == np.sign(rec_["dS_cal"])) if abs(rec_["dS_cal"]) > TAU else None
        cases.append(rec_)
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)
    res["changed_B_vs_A_tune"] = [c for c in cases if c["right_A_tune"] is not None and c["right_A_tune"] != c["right_B_tune"]]
    # the int4 Gate-vs-Up decisions (activation-only single branches)
    dec = {}
    for l in LAYERS:
        g, u = f"L{l}|a|g=fp32/int4|u=-", f"L{l}|a|g=-|u=fp32/int4"
        rg, ru = rows[names.index(g)], rows[names.index(u)]
        dec[f"L{l}"] = {"truth_gate_minus_up": {"cal": rg["S_cal"] - ru["S_cal"], "tune": rg["S_tune"] - ru["S_tune"]},
                        **{m: {"gate": cost[g][m], "up": cost[u][m],
                               "right_cal": bool(np.sign(cost[g][m] - cost[u][m]) == np.sign(rg["S_cal"] - ru["S_cal"])),
                               "right_tune": bool(np.sign(cost[g][m] - cost[u][m]) == np.sign(rg["S_tune"] - ru["S_tune"]))}
                           for m in ("A", "B", "C", "D", "E")}}
    res["int4_gate_vs_up"] = dec
    # paired window bootstrap (tuning and calibration windows resampled; the predictions are fixed)
    rng = np.random.default_rng(0)
    boot = {}
    for tname, sl in (("tune", slice(CAL, CAL + TUNE)), ("cal", slice(0, CAL))):
        w = dl[:, sl]
        acc = {m: [] for m in ("A", "B", "C", "D")}
        accq = {m: [] for m in ("A", "B", "C", "D")}
        sp = {m: [] for m in ("A", "B", "C", "D")}
        eq = cats["equal_resource"]
        for _ in range(BOOT):
            t = w[:, rng.integers(0, w.shape[1], w.shape[1])].sum(1)
            dt = t[i] - t[j]
            keep = np.abs(dt) > TAU
            for m in acc:
                right = np.sign(pred[m][i] - pred[m][j]) == np.sign(dt)
                acc[m].append((right & keep).sum() / keep.sum())
                accq[m].append((right & keep & eq).sum() / max((keep & eq).sum(), 1))
                sp[m].append(np.mean(list(spearman_groups(pred[m], t, rows, mode="a").values())))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        boot[tname] = {f"{a}-{b}": {"all_pairs_accuracy": ci(np.array(acc[a]) - np.array(acc[b])),
                                    "equal_resource_accuracy": ci(np.array(accq[a]) - np.array(accq[b])),
                                    "spearman_activation_only_mean": ci(np.array(sp[a]) - np.array(sp[b]))}
                       for a, b in (("B", "A"), ("D", "C"))}
    res["bootstrap"] = {"replicates": BOOT, "method": "windows resampled with replacement (paired: the same resample for "
                        "every method), pairs re-filtered by |dS| > TAU per replicate, predictions fixed", **boot}
    # costs table
    with open(out / "costs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "layer", "mode", "target", "gate", "up", "M_bits", "B_bits", "W", "V_old", "V_paired", "U",
                    "raw_diag", "native_A", "A", "B", "C", "D", "E", "S_cal", "S_tune"])
        for r in rows:
            c = cost[r["name"]]
            w.writerow([r["name"], r["layer"], r["mode"], r["target"], r["gate"], r["up"], *r["resources"], c["W"], c["V_old"],
                        c["V_paired"], c["U"], c["raw_diag"], c["native_A"], c["A"], c["B"], c["C"], c["D"], c["E"],
                        r["S_cal"], r["S_tune"]])
    res["seconds"] = time.time() - t0
    jdump(out / "summary.json", res)
    log(f"evaluated {len(rows)} candidates, {len(P)} pairs in {res['seconds']:.0f}s")


def stage_tables(args, out, log):
    s, t = jload(out / "summary.json"), jload(out / "terms.json")
    ch = t["checks"]
    L = ["### 數值檢查", "", f"- synthetic：{ch['synthetic']}",
         f"- FP32 QDQ 不改變輸入：{ch['fp32_qdq_identity']}；非有限值：{ch['nonfinite']}",
         f"- 協方差恆等式 V_paired,c − V_old,c = n·Cov(g², r²)：最大相對誤差 {ch['covariance_identity_max_rel']:.2e}",
         f"- 打亂期望：{ch['shuffle_cases']} 個 tap × 格式，(平均 − V_old)/SE 的 |z| 最大 {ch['shuffle_z_abs_max']:.2f}，"
         f"|z| > 3 有 {ch['shuffle_z_abs_gt3']} 個，z 平均 {ch['shuffle_z_mean']:+.2f}", "",
         "### baseline 一致性", ""]
    b = s["baseline"]
    L += [f"- A 的 activation 部分 vs native table C：median 相對差 {b['A_activation_vs_native_C_rel_median']:.1e}，max "
          f"{b['A_activation_vs_native_C_rel_max']:.1e}；A vs BC_total：max {b['A_vs_BC_total_rel_max']:.1e}",
          f"- A／B 的 W 與 U 完全相同：{b['W_U_identical_in_A_and_B']}",
          f"- 舊評估（tuning，全部配對）：A {b['old_evaluation']['attribution_A_tune_all']}、C "
          f"{b['old_evaluation']['attribution_C_tune_all']}、E {b['old_evaluation']['variance_stage_raw_diag_tune_all']}；相同資源 A "
          f"{b['old_evaluation']['attribution_A_tune_equal']}、C {b['old_evaluation']['attribution_C_tune_equal']}"]
    for tname, lab in (("tune", "tuning"), ("cal", "calibration")):
        T = s[f"pairs|{tname}"]
        L += ["", f"### 配對排序（{lab}，|ΔS| > {TAU:g}，所有方法同一組配對）", "",
              "| 配對類別 | 配對數 | A 原完整 | B 新完整 | B vs A 修正／弄壞 | C 原 var | D 新 var | D vs C 修正／弄壞 | E raw diag | 打亂（完整）平均［min, max］ | BC_total（native） |",
              "|---|---:|---:|---:|---|---:|---:|---|---:|---|---:|"]
        for c, r in T.items():
            n_ = r["A"]["pairs"]
            if n_ == 0:
                continue
            L.append(f"| {c} | {n_} | {r['A']['right']} | {r['B']['right']} | +{r['B_vs_A']['fixed']}／−{r['B_vs_A']['broken']} | "
                     f"{r['C']['right']} | {r['D']['right']} | +{r['D_vs_C']['fixed']}／−{r['D_vs_C']['broken']} | {r['E']['right']} | "
                     f"{r['shuffled_full']['mean']:.1f}［{r['shuffled_full']['min']}, {r['shuffled_full']['max']}］ | {r['BC_total']['right']} |")
        sp = s[f"spearman|{tname}"]
        L += ["", f"### 組內 Spearman（{lab} S）", "", "| 組 | A | B | C | D | E | 打亂（完整）平均 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for g in sp["A"]:
            L.append(f"| {g} | " + " | ".join(f"{sp[m][g]:.3f}" for m in ("A", "B", "C", "D", "E")) + f" | {sp['shuffled_full_mean'][g]:.3f} |")
        L.append("| 平均（activation-only 4 組） | " + " | ".join(
            f"{np.mean([v for g, v in sp[m].items() if g.endswith('|a')]):.3f}" for m in ("A", "B", "C", "D", "E")) + " | "
            f"{np.mean([v for g, v in sp['shuffled_full_mean'].items() if g.endswith('|a')]):.3f} |")
    L += ["", "### int4 Gate vs Up（activation-only 單支；對／錯：calibration／tuning）", "", "| 層 | 實際 Gate − Up（cal／tune） | A | B | C | D | E |", "|---|---|---|---|---|---|---|"]
    for l, d in s["int4_gate_vs_up"].items():
        mk = lambda x: ("對" if x["right_cal"] else "錯") + "／" + ("對" if x["right_tune"] else "錯")
        L.append(f"| {l} | {d['truth_gate_minus_up']['cal']:+.0f}／{d['truth_gate_minus_up']['tune']:+.0f} | " +
                 " | ".join(f"{d[m]['gate']:.0f} vs {d[m]['up']:.0f}（{mk(d[m])}）" for m in ("A", "B", "C", "D", "E")) + " |")
    L += ["", "### 逐 window paired bootstrap（95% 區間，差值）", ""]
    for tname in ("tune", "cal"):
        for k, v in s["bootstrap"][tname].items():
            L.append(f"- {tname} {k}：全部配對正確率 [{v['all_pairs_accuracy'][0]:+.4f}, {v['all_pairs_accuracy'][1]:+.4f}]，相同資源 "
                     f"[{v['equal_resource_accuracy'][0]:+.4f}, {v['equal_resource_accuracy'][1]:+.4f}]，activation-only 組內 Spearman 平均 "
                     f"[{v['spearman_activation_only_mean'][0]:+.4f}, {v['spearman_activation_only_mean'][1]:+.4f}]")
    # position 0 shares per tap (int formats + p6e1), with the global mu
    tp = t["taps"]
    L += ["", "### position 0 在 V_old／V_paired 中的份量（同一個全域 mu）", "",
          "| tap | 格式 | V_old | V_paired | V_paired／V_old | pos 0 佔 V_paired | pos 0 佔 V_old（梯度側） | pos 0 佔 V_old（誤差側） | 打亂平均／V_old |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, v in tp.items():
        L.append(f"| {key.rsplit('|', 1)[0]} | {key.rsplit('|', 1)[1]} | {v['V_old']:.4g} | {v['V_paired']:.4g} | "
                 f"{v['V_paired'] / max(v['V_old'], 1e-300):.3f} | {v['pos0_share_paired']:.3f} | {v['pos0_share_old_grad']:.3f} | "
                 f"{v['pos0_share_old_err']:.3f} | {v['V_shuffled_mean'] / max(v['V_old'], 1e-300):.4f} |")
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("freeze", "compute", "evaluate", "tables"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "freeze":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "config.json").exists():
        raise SystemExit("run --stage freeze first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {}
    timing[args.stage] = time.time() - t0
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
