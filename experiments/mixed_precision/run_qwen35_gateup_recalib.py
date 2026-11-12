#!/usr/bin/env python3
"""Qwen3.5-0.8B: are BC's Gate / Up estimates wrong, or is it the calibration sample? The original BC (unchanged)
rebuilt natively from a second, disjoint calibration batch (B) for the Gate / Up projections of the six
representative layers of runs/qwen35_bc_20261015, and both BC tables (A = that run's calibration, B = the new one)
compared with the measured damage of Gate alone, Up alone, their sum and Gate+Up quantized together:

  in sample      each BC against the loss changes on its own calibration windows (the windows it was built from)
  out of sample  each BC against the 30 diagnostic windows of runs/qwen35_bc_20261015

Every truth uses the int activation ranges of the BC it is compared with (A: that run's, B: measured on batch B).
Stages (one --out-dir):
  plan      plan.json (design and the decision rule, frozen before any new number), batch-B tokens, candidates
  table     native BC table of the 12 Gate / Up nodes from batch B (run_native_bc_pipeline.native_tables, total)
  freeze    BC_B scores of the candidates, frozen with hashes
  truth     PyTorch replays (--parts calib_b, calib_a, diag_b; --device cuda as the main run's truth)
  analyze   analysis.json, pairs.csv, report tables
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35  # noqa: F401  (transformers 5 first)
import numpy as np
import torch

from run_qwen35_bc import _pair_class, jdump, jload, replay_truth

HERE = Path(__file__).resolve().parent
MAIN = HERE / "runs/qwen35_bc_20261015"
SEQ = 128
CALIB_B = ("validation", 16384, 10)          # disjoint from every token range the main run used (<= 14336)
SETS = ("gate_proj", "up_proj", "gate+up")


def is_int_act(c):
    return any(a.startswith("int") for _, a in c["formats"].values())


def stage_plan(args, out, log):
    from qwen3_swiglu_common import sha256
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    split, first, n = CALIB_B
    ids = q35.wikitext_ids(split)
    toks = np.asarray(ids[first:first + n * SEQ], np.int64).reshape(n, SEQ)
    np.save(out / "tokens_calibration_b.npy", toks)
    cands = [c for c in jload(MAIN / "candidates.json") if c["set"] in SETS]
    jdump(out / "candidates.json", cands)
    plan = {
        "question": "are BC's Gate / Up estimates wrong (criterion), or does the calibration sample change them?",
        "criterion": "original BC unchanged: C = W + A, total input gradient, native_tables",
        "main_run": str(MAIN.relative_to(HERE)),
        "calibration_A": "main run tokens_calibration.npy (validation 0-1279)",
        "calibration_B": {"split": split, "tokens": [first, first + n * SEQ], "windows": n, "seq": SEQ,
                          "sha256": sha256(out / "tokens_calibration_b.npy"),
                          "disjoint_from": "main run calibration 0-1279, diagnostic 1280-5119, 512-token 8192-14335"},
        "nodes": "gate_proj and up_proj of layers 0, 3, 12, 15, 22, 23 (12 nodes)",
        "candidates": {"count": len(cands), "source": "main run candidates.json, sets gate / up / gate+up",
                       "main_candidates_sha256": sha256(MAIN / "candidates.json")},
        "truths": {
            "T_cal_a": "main calibration windows, ranges A (in sample for BC_A)",
            "T_cal_b": "batch-B windows, ranges B (in sample for BC_B)",
            "T_diag_a": "main truth.json (30 diagnostic windows, ranges A; GPU)",
            "T_diag_b": "30 diagnostic windows, ranges B: candidates with an int activation replayed again, the others "
                        "identical to T_diag_a (ranges do not enter)"},
        "pairs": "equal M and B within (layer, mode): single Gate vs Up (same format / int vs posit), joint swaps, "
                 "joint int vs posit in one branch, other joint; decisive = 95% of 2000 window bootstraps agree",
        "metrics": ["decisive accuracy of BC_A / BC_B on each truth, Gate/Up decision pairs pooled and by class",
                    "overlap of the decisive errors of BC_A and BC_B on the diagnostic truth",
                    "log-ratio bias e = log(BC_up / BC_gate) - log(T_up / T_gate), single same-format pairs with both "
                    "truths significantly > 0; sign per (layer, mode)",
                    "stability of log(BC_up / BC_gate) between A and B",
                    "additivity: BC_g + BC_u, T_g + T_u and T_joint"],
        "decision_rule": {
            "systematic (consider a criterion change, pre-registered separately)":
                "in-sample decisive accuracy on the pooled Gate/Up decision pairs <= 0.85 for both BC_A on T_cal_a and "
                "BC_B on T_cal_b, AND decisive errors on the diagnostic truth shared: |E_A & E_B| / min(|E_A|, |E_B|) "
                ">= 0.6, AND the in-sample log-ratio bias has the same sign under A and B in >= 2/3 of the (layer, "
                "mode) groups that have data",
            "calibration sample (no criterion change; more / other calibration)":
                "in-sample accuracy >= 0.90 for both, AND overlap < 0.5",
            "otherwise": "mixed: report which conditions hold; no criterion change from this study alone"},
        "pooled_gate_up_decision_classes": ["single: branch choice gate-up, same format",
                                            "single: branch choice gate-up, int vs posit", "joint: gate/up swap"],
        "seeds": {"bootstrap": 0},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    jdump(out / "plan.json", plan)
    log(f"plan: {len(cands)} candidates, batch B tokens {first}-{first + n * SEQ}")


def stage_table(args, out, log):
    import shutil
    from run_native_bc_pipeline import erm, native_tables, qwen35_spec
    from qwen3_swiglu_common import sha256
    layers = {l for v in jload(MAIN / "plan.json")["ranking"]["layers"].values() for l in v}
    work = out / "bc"
    work.mkdir(exist_ok=True)
    rel = Path("..") / out.name / "tokens_calibration_b.npy"          # qwen35_spec paths are relative to MAIN
    spec = qwen35_spec(work, MAIN, select=lambda i: i["kind"] in ("gate_proj", "up_proj") and i["layer"] in layers,
                       calib=str(rel))
    if sha256(spec_calib := MAIN / rel) != jload(out / "plan.json")["calibration_B"]["sha256"]:
        raise SystemExit(f"{spec_calib}: not the planned batch")
    t0 = time.time()
    B, C, _, timing = native_tables(spec, erm.Tools(), work, log, input_grad="total")
    jdump(out / "table_b.json", {"B": B, "C": C, "naming": "B = W, C = A; calibration batch B"})
    shutil.copy(work / "ranges_by_point.json", out / "ranges_b.json")
    nonfinite = [(n, f) for t in (B, C) for n, row in t.items() for f, v in row.items() if not np.isfinite(v)]
    jdump(out / "table_b_info.json", {"nodes": len(B), "timing": timing, "nonfinite": nonfinite,
                                      "seconds": time.time() - t0})
    log(f"table B: {len(B)} nodes in {timing['criterion_total_s']:.0f}s, {len(nonfinite)} non-finite")


def stage_freeze(args, out, log):
    from qwen3_swiglu_common import sha256
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    t = jload(out / "table_b.json")
    cands = jload(out / "candidates.json")
    for c in cands:
        c["bc_b"] = float(sum(t["B"][c["nodes"][k]][w] + t["C"][c["nodes"][k]][a] for k, (w, a) in c["formats"].items()))
    jdump(out / "candidates.json", cands)
    jdump(out / "freeze.json", {"candidates_sha256": sha256(out / "candidates.json"),
                                "table_b_sha256": sha256(out / "table_b.json"),
                                "ranges_b_sha256": sha256(out / "ranges_b.json"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"frozen {len(cands)} candidates")


def stage_truth(args, out, log):
    """calib_a only involves BC_A (frozen in the main run) and may run before the batch-B freeze; calib_b and
    diag_b check it."""
    from qwen3_swiglu_common import sha256
    parts = args.parts.split(",")
    cands = jload(out / "candidates.json")
    if set(parts) - {"calib_a"}:
        fz = jload(out / "freeze.json")
        for f, k in (("candidates.json", "candidates_sha256"), ("table_b.json", "table_b_sha256"),
                     ("ranges_b.json", "ranges_b_sha256")):
            if sha256(out / f) != fz[k]:
                raise SystemExit(f"{f} changed after the freeze")
    for part in parts:
        if part == "calib_a":
            job = (cands, np.load(MAIN / "tokens_calibration.npy"), jload(MAIN / "ranges_by_point.json"))
        elif part == "calib_b":
            job = (cands, np.load(out / "tokens_calibration_b.npy"), jload(out / "ranges_b.json"))
        else:
            job = ([c for c in cands if is_int_act(c)], np.load(MAIN / "tokens_diagnostic.npy"), jload(out / "ranges_b.json"))
        replay_truth(MAIN, *job, out / f"truth_{part}.json", args.batch, log, device=args.device)


def stage_analyze(args, out, log):
    import csv
    from scipy.stats import spearmanr
    cands = jload(out / "candidates.json")
    names = [c["name"] for c in cands]
    T = SEQ - 1
    main_truth = jload(MAIN / "truth.json")["rows"]
    diag_b = jload(out / "truth_diag_b.json")["rows"]
    truths = {"T_cal_a": jload(out / "truth_calib_a.json")["rows"], "T_cal_b": jload(out / "truth_calib_b.json")["rows"],
              "T_diag_a": main_truth, "T_diag_b": {n: diag_b.get(n, main_truth[n]) for n in names}}
    crit = {"BC_A": np.array([c["bc"] for c in cands]), "BC_B": np.array([c["bc_b"] for c in cands])}
    combos = [("BC_A", "T_cal_a", "in"), ("BC_B", "T_cal_b", "in"), ("BC_A", "T_diag_a", "out"), ("BC_B", "T_diag_b", "out")]
    rng = np.random.default_rng(jload(out / "plan.json")["seeds"]["bootstrap"])
    D, Db = {}, {}
    for k, rows in truths.items():
        D[k] = np.array([rows[n] for n in names]) / T                         # [candidate, window] per token
        nw = D[k].shape[1]
        boot = rng.integers(0, nw, (2000, nw))
        Db[k] = np.stack([D[k][:, b].mean(1) for b in boot], 1)
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["layer"], c["mode"]), []).append(i)
    pairs = [(i, j, _pair_class(cands[i], cands[j])) for ix in groups.values() for a, i in enumerate(ix) for j in ix[a + 1:]
             if cands[i]["M_bytes"] == cands[j]["M_bytes"] and cands[i]["B_bops"] == cands[j]["B_bops"]]
    pooled = set(jload(out / "plan.json")["pooled_gate_up_decision_classes"])
    res = {"candidates": len(cands), "pairs": len(pairs), "accuracy": {}, "overlap": {}, "bias": {}, "stability": {},
           "additivity": {}}
    rows_out = []
    errors = {}
    for bc, tk, kind in combos:
        m = D[tk].mean(1)
        out_ = {}
        for label, sel in (("pooled_gate_up", lambda p: p[2] in pooled), ("all", lambda p: True),
                           *[(cl, (lambda cl: lambda p: p[2] == cl)(cl)) for cl in sorted({p[2] for p in pairs})]):
            ps = [p for p in pairs if sel(p)]
            if not ps:
                continue
            I, J = np.array([p[0] for p in ps]), np.array([p[1] for p in ps])
            d = m[I] - m[J]
            dec = ((Db[tk][I] - Db[tk][J]) * np.sign(d)[:, None] > 0).mean(1) >= 0.95
            ok = np.sign(crit[bc][I] - crit[bc][J]) == np.sign(d)
            out_[label] = {"pairs": len(ps), "decisive": int(dec.sum()), "decisive_correct": int((ok & dec).sum()),
                           "decisive_accuracy": float((ok & dec).sum() / max(dec.sum(), 1)), "all_accuracy": float(ok.mean())}
            if label == "pooled_gate_up":
                errors[(bc, tk)] = {(names[i], names[j]) for i, j, o, de in zip(I, J, ok, dec) if de and not o}
                for i, j, o, de, dd in zip(I, J, ok, dec, d):
                    rows_out.append([bc, tk, names[i], names[j], cands[i]["layer"], cands[i]["mode"],
                                     _pair_class(cands[i], cands[j]), float(dd), bool(de), bool(o)])
        res["accuracy"][f"{bc}|{tk}|{kind}"] = out_
    ea, eb = errors[("BC_A", "T_diag_a")], errors[("BC_B", "T_diag_b")]
    res["overlap"] = {"E_A": len(ea), "E_B": len(eb), "shared": len(ea & eb),
                      "shared_over_min": len(ea & eb) / max(min(len(ea), len(eb)), 1),
                      "in_sample_E_A": len(errors[("BC_A", "T_cal_a")]), "in_sample_E_B": len(errors[("BC_B", "T_cal_b")]),
                      "in_sample_shared": len(errors[("BC_A", "T_cal_a")] & errors[("BC_B", "T_cal_b")])}
    # log-ratio bias on single same-format Gate vs Up pairs, both truths significantly > 0
    idx = {(c["layer"], c["mode"], c["set"], tuple(c["formats"][c["set"]])): i for i, c in enumerate(cands) if c["set"] != "gate+up"}
    sign_by = {}
    for bc, tk, kind in combos:
        lo = np.percentile(Db[tk], 2.5, axis=1)
        m = D[tk].mean(1)
        per = {}
        for (l, mode, s, f), i in idx.items():
            if s != "gate_proj":
                continue
            j = idx[(l, mode, "up_proj", f)]
            if lo[i] > 0 and lo[j] > 0:
                e = np.log(crit[bc][j] / crit[bc][i]) - np.log(m[j] / m[i])
                per.setdefault(f"L{l}|{mode}", []).append(float(e))
        res["bias"][f"{bc}|{tk}|{kind}"] = {g: {"n": len(v), "median_e": float(np.median(v))} for g, v in sorted(per.items())}
        sign_by[(bc, tk)] = {g: np.sign(np.median(v)) for g, v in per.items()}
    sa, sb = sign_by[("BC_A", "T_cal_a")], sign_by[("BC_B", "T_cal_b")]
    common = sorted(set(sa) & set(sb))
    res["bias_sign_agreement_in_sample"] = {"groups": len(common), "same_sign": int(sum(sa[g] == sb[g] for g in common)),
                                            "positive_A": int(sum(sa[g] > 0 for g in common)),
                                            "positive_B": int(sum(sb[g] > 0 for g in common))}
    allv = {k: [e for g in v.values() for e in [g["median_e"]]] for k, v in res["bias"].items()}
    res["bias_overall_median"] = {k: float(np.median(v)) if v else None for k, v in allv.items()}
    # stability of BC's Up / Gate ratio between the two calibrations
    lr = {bc: [] for bc in crit}
    for (l, mode, s, f), i in idx.items():
        if s == "gate_proj":
            j = idx[(l, mode, "up_proj", f)]
            for bc in crit:
                lr[bc].append(np.log(crit[bc][j] / crit[bc][i]))
    a, b = np.array(lr["BC_A"]), np.array(lr["BC_B"])
    res["stability"] = {"items": len(a), "spearman_logratio_A_vs_B": float(spearmanr(a, b)[0]),
                        "median_abs_diff_logratio": float(np.median(np.abs(a - b))),
                        "sign_flips": int(np.sum(np.sign(a) != np.sign(b))),
                        "bc_spearman_A_vs_B_all": float(spearmanr(crit["BC_A"], crit["BC_B"])[0])}
    # additivity: truth joint vs truth singles sum vs BC
    for tk in truths:
        m = D[tk].mean(1)
        rel, addok, bcok = [], [], {bc: [] for bc in crit}
        jidx = {}
        for i, c in enumerate(cands):
            if c["set"] == "gate+up":
                g = idx[(c["layer"], c["mode"], "gate_proj", tuple(c["formats"]["gate_proj"]))]
                u = idx[(c["layer"], c["mode"], "up_proj", tuple(c["formats"]["up_proj"]))]
                jidx[i] = m[g] + m[u]
                if abs(m[i]) > 1e-3:
                    rel.append(abs(m[i] - m[g] - m[u]) / abs(m[i]))
        for i, j, cl in pairs:
            if i in jidx and j in jidx:
                t = np.sign(m[i] - m[j])
                addok.append(np.sign(jidx[i] - jidx[j]) == t)
                for bc in crit:
                    bcok[bc].append(np.sign(crit[bc][i] - crit[bc][j]) == t)
        res["additivity"][tk] = {"rel_interaction_median": float(np.median(rel)), "rel_interaction_p90": float(np.percentile(rel, 90)),
                                 "joint_pairs": len(addok), "additive_oracle_accuracy": float(np.mean(addok)),
                                 **{f"{bc}_accuracy": float(np.mean(v)) for bc, v in bcok.items()}}
    # decision (plan.json rule)
    acc_in = [res["accuracy"]["BC_A|T_cal_a|in"]["pooled_gate_up"]["decisive_accuracy"],
              res["accuracy"]["BC_B|T_cal_b|in"]["pooled_gate_up"]["decisive_accuracy"]]
    ov = res["overlap"]["shared_over_min"]
    sg = res["bias_sign_agreement_in_sample"]
    systematic = all(x <= 0.85 for x in acc_in) and ov >= 0.6 and sg["groups"] and sg["same_sign"] >= 2 / 3 * sg["groups"]
    sampling = all(x >= 0.90 for x in acc_in) and ov < 0.5
    res["decision"] = {"in_sample_accuracy": acc_in, "overlap": ov, "bias_sign_same": [sg["same_sign"], sg["groups"]],
                       "verdict": "systematic" if systematic else "calibration sample" if sampling else "mixed"}
    jdump(out / "analysis.json", res)
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bc", "truth", "a", "b", "layer", "mode", "class", "truth_diff", "decisive", "bc_correct"])
        w.writerows(rows_out)
    log(f"decision: {res['decision']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "table", "freeze", "truth", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--parts", default="calib_b,calib_a,diag_b", help="truth: which replays")
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
    timing["runs"].append({"stage": args.stage, "parts": args.parts, "device": args.device,
                           "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "wall_s": time.time() - t0,
                           "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20})
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
