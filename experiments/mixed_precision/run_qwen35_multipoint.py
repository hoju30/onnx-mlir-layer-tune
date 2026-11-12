#!/usr/bin/env python3
"""Qwen3.5-0.8B: whole-configuration validation at several resource points of the existing BC frontier.

Not a criterion change. For each search scope of runs/qwen35_bc_20261015 (FFN-only, primary; FFN + attention
projections, secondary), three configurations of the existing 2-D frontier are compiled natively and evaluated with
FP32 on 64 fresh 128-token windows (two disjoint segments): an aggressive point (less resource than the knee), the
knee itself (unchanged) and a conservative point (more resource). Points are chosen by a deterministic rule on the
resource R and BC only, before any new-window result exists. The resource points are research controls; the
pipeline's automatic knee is not changed.

Stages (one --out-dir; every stage resumable):
  audit    re-derive the key numbers of the four earlier Qwen3.5 reports from their JSON outputs (audit.json)
  plan     plan.json, environment.json, the new windows, selected_points.json (formats, BC, M, BOPs, R), freeze.json
  native   compile and evaluate configurations natively (--configs; one per batch), per-window summed CE
  sim      PyTorch replay (--device cuda / cpu) of every configuration on the pre-registered window subset
  analyze  tables, paired bootstrap, cross-implementation comparison, figures (analysis.json, per_window.csv, *.png)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35  # noqa: F401  (transformers 5 first)
import numpy as np
import torch

from run_qwen35_bc import jdump, jload

HERE = Path(__file__).resolve().parent
MAIN = HERE / "runs/qwen35_bc_20261015"
PRIOR = {"recalib": HERE / "runs/qwen35_gateup_recalib_20261015", "elem": HERE / "runs/qwen35_weight_elem_20261015",
         "knee_gateup": HERE / "runs/qwen35_knee_gateup_20261015"}
SEQ = 128
SCOPES = {"ffn": "FFN-only (primary)", "ffn_attn": "FFN + attention projections (secondary)"}
POINTS = ("aggressive", "knee", "conservative")
SEGMENTS = {"V": ("validation", 131072, 32), "T": ("test", 131072, 32)}
SUBSET = list(range(0, 16)) + list(range(32, 48))       # cross-implementation subset: first 16 of each segment
USED = {"validation": [[0, 1280, "Qwen3.5 calibration A"], [1280, 5120, "diagnostic"], [8192, 10240, "512 calibration"],
                       [10240, 14336, "512 diagnostic"], [16384, 17664, "calibration B (recalibration study)"],
                       [20480, 24320, "fresh windows (element-wise weight study)"],
                       [24576, 28416, "selection windows (knee Gate/Up study)"]],
        "test": [[0, 8192, "final test (main run; re-used by the knee Gate/Up study)"]]}


def cfg_names():
    return ["fp32"] + [f"{s}_{p}" for s in SCOPES for p in POINTS]


# --------------------------------------------------------------------------- #
# audit of the earlier reports
# --------------------------------------------------------------------------- #
def stage_audit(args, out, log):
    """Each check: a number stated in an earlier report against the value recomputed from its JSON."""
    checks = []

    def chk(report, claim, stated, actual, tol=0.0015):
        ok = (stated == actual) if isinstance(stated, (str, bool)) else abs(float(stated) - float(actual)) <= tol * max(1.0, abs(float(stated)))
        checks.append({"report": report, "claim": claim, "stated": stated, "data": actual, "match": bool(ok)})
    a = jload(MAIN / "analysis.json")
    r = jload(MAIN / "results.json")
    s = jload(MAIN / "selection.json")
    p = a["pairs"]
    chk("bc", "decisive pairs", 1057, p["all"]["decisive"], 0)
    chk("bc", "BC decisive correct", 934, p["all"]["BC"]["decisive_correct"], 0)
    chk("bc", "SQNR decisive correct", 944, p["all"]["SQNR"]["decisive_correct"], 0)
    chk("bc", "BC-SQNR CI low", -0.025, round(p["all"]["BC_minus_SQNR_ci95"][0], 3), 0)
    chk("bc", "BC-SQNR CI high", 0.003, round(p["all"]["BC_minus_SQNR_ci95"][1], 3), 0)
    chk("bc", "gate-up same format decisive correct", 32, p["class=single: branch choice gate-up, same format"]["BC"]["decisive_correct"], 0)
    chk("bc", "wa x linear decisive correct", 126, p["mode=wa|block=linear_attention"]["BC"]["decisive_correct"], 0)
    chk("bc", "FP32 PPL", 43.800, round(r["fp32"]["ppl"], 3), 0)
    chk("bc", "FFN knee PPL", 44.120, round(r["ffn"]["ppl"], 3), 0)
    chk("bc", "FFN+attn knee PPL", 44.728, round(r["ffn_attn"]["ppl"], 3), 0)
    chk("bc", "FFN knee dNLL", 0.0073, round(r["ffn"]["delta_nll_vs_fp32"]["mean"], 4), 0)
    chk("bc", "FFN+attn knee dNLL", 0.0210, round(r["ffn_attn"]["delta_nll_vs_fp32"]["mean"], 4), 0)
    chk("bc", "FFN knee CI", "[+0.0036, +0.0108]", "[{:+.4f}, {:+.4f}]".format(*r["ffn"]["delta_nll_vs_fp32"]["ci95"]))
    chk("bc", "FFN+attn knee CI", "[+0.0151, +0.0267]", "[{:+.4f}, {:+.4f}]".format(*r["ffn_attn"]["delta_nll_vs_fp32"]["ci95"]))
    chk("bc", "FFN knee size", 0.826, round(s["ffn"]["knee"]["size"], 3), 0)
    chk("bc", "FFN knee BOPs", 0.877, round(s["ffn"]["knee"]["bops"], 3), 0)
    chk("bc", "FFN+attn knee size", 0.799, round(s["ffn_attn"]["knee"]["size"], 3), 0)
    chk("bc", "FFN+attn knee BOPs", 0.841, round(s["ffn_attn"]["knee"]["bops"], 3), 0)
    for v, n in (("ffn", 21), ("ffn_attn", 23)):
        chk("bc", f"{v} frontier points", n, len(jload(MAIN / f"selection_{v}.json")["frontier"]), 0)
    chk("bc", "FP32 run seconds", 692, round(r["fp32"]["run_wall_s"]), 1)
    chk("bc", "table seconds", 1854, round(jload(MAIN / "tables.json")["timing"]["criterion_total_s"]), 1)
    bk = s["ffn_attn"]["by_kind"]["down_proj"]
    chk("bc", "FFN+attn: Down mostly int7/p8e1 (section 4)", "int7/p8e1 is the single most common",
        "int7/p8e1 is the single most common" if bk.get("int7/p8e1", 0) > max(v for k, v in bk.items() if k != "int7/p8e1")
        else f"tie or not: {bk}")
    rc = jload(PRIOR["recalib"] / "analysis.json")
    chk("recalib", "BC_A in-sample pooled correct", 107, rc["accuracy"]["BC_A|T_cal_a|in"]["pooled_gate_up"]["decisive_correct"], 0)
    chk("recalib", "BC_B in-sample pooled correct", 138, rc["accuracy"]["BC_B|T_cal_b|in"]["pooled_gate_up"]["decisive_correct"], 0)
    chk("recalib", "overlap shared / E_B", "22/23", f"{rc['overlap']['shared']}/{rc['overlap']['E_B']}")
    chk("recalib", "log-ratio Spearman A vs B", 0.990, round(rc["stability"]["spearman_logratio_A_vs_B"], 3), 0)
    chk("recalib", "verdict", "mixed", rc["decision"]["verdict"])
    el = jload(PRIOR["elem"] / "analysis.json")
    chk("elem", "fresh decisive pairs", 725, el["fresh"]["all"]["decisive"], 0)
    chk("elem", "fresh BC / BC_elem correct", "650/650", f"{el['fresh']['all']['BC']['decisive_correct']:.0f}/{el['fresh']['all']['BC_elem']['decisive_correct']:.0f}")
    chk("elem", "pass", False, el["decision"]["pass"])
    ev = jload(PRIOR["knee_gateup"] / "evaluation.json")
    sel = jload(PRIOR["knee_gateup"] / "selection.json")
    chk("knee_gateup", "FFN adopted layers", "[4, 20, 21]", str(sorted(int(k) for k in sel["ffn"]["adopted"])))
    chk("knee_gateup", "FFN native corrected - knee", -0.0004, round(ev["ffn"]["native"]["corrected_minus_knee"]["mean"], 4), 0)
    chk("knee_gateup", "FFN native corrected PPL", 44.101, round(ev["ffn"]["native"]["ppl"], 3), 0)
    chk("knee_gateup", "FFN+attn adopted", 0, len(sel["ffn_attn"]["adopted"]), 0)
    chk("knee_gateup", "final-test windows were re-used (not independent)", True,
        jload(PRIOR["knee_gateup"] / "plan.json")["evaluation"].startswith("final test (main run tokens_final_test.npy"))
    mism = [c for c in checks if not c["match"]]
    jdump(out / "audit.json", {"checks": checks, "mismatches": mism, "n": len(checks)})
    log(f"audit: {len(checks)} checks, {len(mism)} mismatches: " + "; ".join(f"{c['report']}: {c['claim']} ({c['stated']} vs {c['data']})" for c in mism))


# --------------------------------------------------------------------------- #
# plan: windows, point selection, freeze
# --------------------------------------------------------------------------- #
def resources(fmts, nodes, mapping):
    from htq_pareto_search import FORMAT_BITS
    M = B = M8 = B8 = 0.0
    for n in nodes:
        e = int(np.prod(mapping[n]["shape_in_out"]))
        w, a = fmts[n]["w"], fmts[n]["a"]
        M += e * FORMAT_BITS[w] / 8
        B += e * SEQ * FORMAT_BITS[w] * FORMAT_BITS[a]
        M8 += e
        B8 += e * SEQ * 64
    return {"M_bytes": M, "BOPs": B, "M_int8_bytes": M8, "BOPs_int8": B8, "M_rel": M / M8, "BOPs_rel": B / B8,
            "R": max(M / M8, B / B8)}


def select_points(sel, names):
    """Aggressive / conservative: the frontier point (config differing from the knee) on that side of the knee whose
    R is closest to 0.8 R_knee / 1.25 R_knee (ties: the lower R). Fallback if that point is degenerate (the
    frontier's lowest-R point, or BC = 0): the nearest frontier point to the knee on that side."""
    from mp_auto import config_hash
    F, K = sel["frontier"], sel["selected"]
    h = lambda f: config_hash({n: f["formats"][n] for n in names})
    kh, rk = h(K), K["resource"]
    rmin = min(f["resource"] for f in F)
    below = [f for f in F if h(f) != kh and f["resource"] < rk]
    above = [f for f in F if h(f) != kh and f["resource"] > rk]
    out, notes = {"knee": K}, {}
    for side, pts, target in (("aggressive", below, 0.8 * rk), ("conservative", above, 1.25 * rk)):
        if not pts:
            raise SystemExit(f"no frontier point on the {side} side")
        p = min(pts, key=lambda f: (abs(f["resource"] - target), f["resource"]))
        if p["resource"] == rmin or p["objective"] == 0:
            notes[side] = "degenerate target point; fallback to the nearest point on that side"
            p = min(pts, key=lambda f: (abs(f["resource"] - rk), f["resource"]))
        out[side] = p
        notes.setdefault(side, f"closest to {target:.4f}")
    return out, notes, kh


def stage_plan(args, out, log):
    from qwen3_swiglu_common import sha256
    from mp_auto import config_hash
    import transformers
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    mapping = jload(MAIN / "node_mapping.json")
    # windows
    toks, seg_info = [], {}
    for k, (split, first, n) in SEGMENTS.items():
        ids = q35.wikitext_ids(split)
        if first + n * SEQ > len(ids):
            raise SystemExit(f"segment {k}: {split} has only {len(ids)} tokens")
        for a, b, what in USED[split]:
            if first < b and first + n * SEQ > a:
                raise SystemExit(f"segment {k} overlaps {what}")
        t = np.asarray(ids[first:first + n * SEQ], np.int64).reshape(n, SEQ)
        toks.append(t)
        seg_info[k] = {"split": split, "tokens": [first, first + n * SEQ], "windows": n, "split_tokens": len(ids)}
    toks = np.concatenate(toks)
    np.save(out / "tokens_eval.npy", toks)
    # points
    points = {}
    (out / "formats").mkdir(exist_ok=True)
    for scope in SCOPES:
        sel = jload(MAIN / f"selection_{scope}.json")
        names = sel["names"]
        knee_file = jload(MAIN / f"selected_formats_{scope}.json")
        picked, notes, kh = select_points(sel, names)
        if config_hash(knee_file) != kh:
            raise SystemExit(f"{scope}: selection.json knee differs from selected_formats_{scope}.json")
        for p in POINTS:
            f = picked[p]
            fm = {n: f["formats"][n] for n in names}
            res = resources(fm, names, mapping)
            if abs(res["M_rel"] - f["size"]) > 1e-9 or abs(res["BOPs_rel"] - f["bops"]) > 1e-9:
                raise SystemExit(f"{scope} {p}: resources do not reproduce the frontier's")
            name = f"{scope}_{p}"
            jdump(out / "formats" / f"{name}.json", fm)
            from collections import Counter
            points[name] = {"scope": scope, "point": p, "nodes": len(names), "bc_total": f["objective"],
                            "frontier_R": f["resource"], "frontier_size": f["size"], "frontier_bops": f["bops"],
                            "search_budget": f.get("search_budget"), "origin": f.get("origin"),
                            "solver_status": f.get("solver_status"), **res, "rule": notes.get(p, "the automatic knee"),
                            "formats_file": f"formats/{name}.json", "config_hash": config_hash(fm),
                            "w_formats": dict(Counter(x["w"] for x in fm.values())),
                            "a_formats": dict(Counter(x["a"] for x in fm.values()))}
        a, k, c = (points[f"{scope}_{p}"] for p in POINTS)
        for side, x in (("aggressive", a), ("conservative", c)):
            dm, db = x["M_bytes"] - k["M_bytes"], x["BOPs"] - k["BOPs"]
            x["vs_knee"] = {"M_bytes": dm, "BOPs": db, "M_rel_change": dm / k["M_bytes"], "BOPs_rel_change": db / k["BOPs"],
                            "componentwise": ("both lower" if dm < 0 and db < 0 else "both higher" if dm > 0 and db > 0
                                              else "not componentwise ordered")}
    jdump(out / "selected_points.json", points)
    log("points: " + "; ".join(f"{n} R {v['R']:.4f} (M {v['M_rel']:.3f}, BOPs {v['BOPs_rel']:.3f}) BC {v['bc_total']:.4g}"
                               for n, v in points.items()))
    # environment
    root = HERE.parent.parent
    import subprocess
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    from run_native_bc_pipeline import erm
    tools = erm.Tools()
    code = ["run_qwen35_multipoint.py", "run_qwen35_bc.py", "run_qwen35_knee_gateup.py", "run_native_bc_pipeline.py",
            "eval_random_mixed_precision.py", "evaluate_native_search_chunked.py", "qwen35_common.py",
            "qwen3_swiglu_common.py", "mp_auto.py", "mp_milp.py"]
    inputs = ["model.onnx", "node_mapping.json", "table.json", "ranges_by_point.json", "selection_ffn.json",
              "selection_ffn_attn.json", "selected_formats_ffn.json", "selected_formats_ffn_attn.json",
              "tokens_calibration.npy", "results.json", "plan.json"]
    jdump(out / "environment.json", {
        "git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--short", "--untracked-files=no"),
        "code_sha256": {f: sha256(HERE / f) for f in code},
        "main_run_inputs_sha256": {f: sha256(MAIN / f) for f in inputs},
        "onnx_mlir_opt": {"sha256": sha256(tools.opt), "mtime": Path(tools.opt).stat().st_mtime},
        "model": {"repo": q35.REPO, "revision": q35.REVISION, "config_sha256": sha256(q35.MODEL_DIR / "config.json"),
                  "tokenizer_sha256": sha256(q35.MODEL_DIR / "tokenizer.json")},
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__})
    # plan
    jdump(out / "plan.json", {
        "purpose": "whole-configuration validation of BC at three resource points per search scope; no criterion change",
        "main_run": str(MAIN.relative_to(HERE)),
        "fixed": ["BC formula, gradients and normalization (main run table.json)", "calibration windows and ranges",
                  "posit LUT and QDQ semantics", "MILP -> 2-D Pareto -> normalized chord knee (main run selection)",
                  "format grid GRID11", "repository defaults and earlier outputs (read only)"],
        "scopes": SCOPES,
        "used_token_ranges": USED,
        "eval_windows": {"segments": seg_info, "seq": SEQ, "count": int(len(toks)), "tokenizer": "model's own (Qwen3.5), "
                         "no special tokens, WikiText-2 raw parquet joined with blank lines (qwen35_common.wikitext_ids)",
                         "sha256": sha256(out / "tokens_eval.npy"), "state": "each window independent (state reset)",
                         "chosen": "fixed offsets far from every used range, before any model output; not changed later",
                         "other_models": "earlier Qwen3 / GPT-2 studies used other tokenizers on early parts of the same "
                                         "splits; they made no decisions about Qwen3.5 configurations"},
        "point_rule": select_points.__doc__,
        "points": {n: {k: v[k] for k in ("bc_total", "M_bytes", "BOPs", "M_rel", "BOPs_rel", "R", "config_hash", "rule")}
                   for n, v in points.items()},
        "native": {"settings": "as runs/qwen35_bc_20261015 evaluate: MIXEDP_WEIGHT_PER_CHANNEL=1, Krnl -O3, clang -O0, "
                               "constants to file, bytecode import, the main run's calibration ranges, posit runtime",
                   "processes": args.chunks, "outputs": "per window: summed next-token CE (float64), 127 tokens, NLL",
                   "ppl": "exp(total CE / total tokens)", "resume": "--stage native --configs <name>; done configs are skipped"},
        "metrics": {"dNLL": "per window (CE_config - CE_fp32) / 127; mean over windows (equal tokens)",
                    "bootstrap": "2000 resamples, stratified: 32 windows drawn with replacement within each segment; the "
                                 "same index draws for every configuration and difference; per segment: the same draws "
                                 "restricted to that segment", "seed": 0,
                    "adjacent": "aggressive - knee and conservative - knee, paired per window"},
        "cross_implementation": {"subset": "windows 0-15 of each segment (32 windows): indices " + str(SUBSET),
                                 "gpu": "PyTorch replay (run_qwen35_knee_gateup.Replay), RTX 3060 FP32 TF32 off, every configuration",
                                 "cpu_if": "the GPU and native subset results disagree on the order of the three points of a "
                                           "scope, or on the sign of an adjacent difference, or a point's subset dNLL differs "
                                           "by more than half the smallest native adjacent difference of its scope -> CPU "
                                           "replay of the same subset",
                                 "no_universal_floor": "the earlier 0.0012 / 0.0023 spreads are not used as a floor"},
        "interpretation": ["BC predicts aggressive > knee > conservative damage by construction of the frontier",
                           "BC is a different scale from dNLL; no absolute prediction error is computed",
                           "three points test a local trend only; the knee is not re-selected from these results",
                           "a larger aggressive dNLL alone is the expected compression cost, not a BC failure"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    jdump(out / "freeze.json", {"selected_points_sha256": sha256(out / "selected_points.json"),
                                "formats_sha256": {n: sha256(out / v["formats_file"]) for n, v in points.items()},
                                "tokens_eval_sha256": sha256(out / "tokens_eval.npy"),
                                "plan_sha256": sha256(out / "plan.json"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan frozen: {len(toks)} windows, {len(points)} points")


def check_freeze(out):
    from qwen3_swiglu_common import sha256
    fz = jload(out / "freeze.json")
    pts = jload(out / "selected_points.json")
    sp = fz["amendments"][-1]["selected_points_sha256"] if fz.get("amendments") else fz["selected_points_sha256"]
    if sha256(out / "selected_points.json") != sp or sha256(out / "tokens_eval.npy") != fz["tokens_eval_sha256"]:
        raise SystemExit("selected points or windows changed after the freeze")
    for n, v in pts.items():
        if sha256(out / v["formats_file"]) != fz["formats_sha256"][n]:
            raise SystemExit(f"{n}: formats changed after the freeze")
    return pts


# --------------------------------------------------------------------------- #
# native evaluation
# --------------------------------------------------------------------------- #
def stage_native(args, out, log):
    import os
    import onnx
    from mp_auto import config_hash
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm, qwen35_spec
    from evaluate_native_search_chunked import chunked_run_so
    from run_qwen35_bc import native_import
    from qwen3_swiglu_common import sha256
    pts = check_freeze(out)
    tools, _, base = native_import(MAIN, log)
    toks = np.load(out / "tokens_eval.npy")
    graph = onnx.load(str(MAIN / "model.onnx"), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in jload(MAIN / "ranges_by_point.json").items():
        for slot, r in slots.items():
            n = by_name[node]
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    spec = qwen35_spec(out / "unused", MAIN)
    nd = out / "native"
    nd.mkdir(exist_ok=True)
    np.save(nd / "tokens_eval.npy", toks)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    posit_obj = erm.build_posit_runtime(tools, nd)
    for name in args.configs.split(","):
        f = nd / f"{name}.json"
        if f.exists():
            log(f"{name}: done, skipped")
            continue
        t0 = time.time()
        target = nd / name
        target.mkdir(exist_ok=True)
        env = None
        if name != "fp32":
            fm = jload(out / pts[name]["formats_file"])
            cfg = {n["name"]: fm.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes}
            env = erm.node_formats_env(cfg, spec.nodes, ranges)
            same_as_main = None
            if pts[name]["point"] == "knee":
                main_env = jload(MAIN / "eval" / pts[name]["scope"] / "compile_env.json")["MIXEDP_NODE_FORMATS"]
                same_as_main = env == main_env
                if not same_as_main:
                    raise SystemExit(f"{name}: compile settings differ from the main run's knee")
            (target / "compile_env.json").write_text(json.dumps({"MIXEDP_NODE_FORMATS": env, "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                                                                 "config_hash": config_hash(fm),
                                                                 "same_as_main_run_knee": same_as_main}, indent=1) + "\n")
        so, info = erm.build_so(tools, target, name, base, env, posit_obj if env else None, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True,
                                uses_posit=None if env else False, uses_f16=None if env else False)
        build_s = time.time() - t0
        if info["warnings"]:
            raise RuntimeError(f"{name}: compilation warnings {info['warnings']}")
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        t1 = time.time()
        z, run = chunked_run_so(args.chunks)(tools, so, nd / "tokens_eval.npy", target / "logits.npy", 36000)
        run_s = time.time() - t1
        (target / "logits.npy").unlink()
        finite = bool(np.isfinite(z).all())
        ce = np.stack([cross_entropy(z[i, :-1], toks[i, 1:]) for i in range(len(toks))])     # [window, 127]
        # loss consistency: float64 log-sum-exp recomputation on two windows
        chk = []
        for i in (0, len(toks) - 1):
            zz = z[i, :-1].astype(np.float64)
            m = zz.max(1, keepdims=True)
            lse = (m[:, 0] + np.log(np.exp(zz - m).sum(1)))
            chk.append(float(np.abs(lse - zz[np.arange(SEQ - 1), toks[i, 1:]] - ce[i]).max()))
        so_sha = sha256(so)
        for p in (so, target / "model.constants.bin"):
            p.unlink(missing_ok=True)
        tot = float(ce.sum())
        res = {"config": name, "windows": int(len(toks)), "tokens_per_window": SEQ - 1,
               "summed_ce": ce.sum(1).tolist(), "nll": tot / ce.size, "ppl": float(np.exp(tot / ce.size)),
               "all_logits_finite": finite, "lse_check_max_abs": max(chk), "build_s": build_s,
               "build_steps": info["steps"], "run_wall_s": run["wall_seconds"], "run_peak_rss_gb": run["peak_rss_gb"],
               "processes": run["processes"], "stage_s": time.time() - t0, "so_sha256": so_sha}
        jdump(f, res)
        log(f"{name}: NLL {res['nll']:.5f} PPL {res['ppl']:.3f} finite {finite} (build {build_s:.0f}s, run {run_s:.0f}s, "
            f"peak {run['peak_rss_gb']:.1f} GB/process)")


# --------------------------------------------------------------------------- #
# PyTorch replay on the subset
# --------------------------------------------------------------------------- #
def stage_sim(args, out, log):
    from run_qwen35_knee_gateup import Replay
    pts = check_freeze(out)
    toks = np.load(out / "tokens_eval.npy")[SUBSET]
    f = out / f"sim_{args.device}.json"
    res = jload(f) if f.exists() else {"subset": SUBSET, "device": args.device, "summed_ce": {}, "seconds": {}}
    rp = Replay(args.device, args.threads)
    for name in cfg_names():
        if name in res["summed_ce"]:
            continue
        t0 = time.time()
        rp.set({} if name == "fp32" else jload(out / pts[name]["formats_file"]))
        res["summed_ce"][name] = rp.ce(toks).tolist()
        res["seconds"][name] = time.time() - t0
        jdump(f, res)
        log(f"{name}: subset NLL {np.sum(res['summed_ce'][name]) / (len(toks) * (SEQ - 1)):.5f} ({res['seconds'][name]:.0f}s)")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def stage_analyze(args, out, log):
    import csv
    pts = check_freeze(out)
    plan = jload(out / "plan.json")
    nat = {n: jload(out / "native" / f"{n}.json") for n in cfg_names() if (out / "native" / f"{n}.json").exists()}
    if set(nat) != set(cfg_names()):
        raise SystemExit(f"native results missing: {sorted(set(cfg_names()) - set(nat))}")
    T = SEQ - 1
    CE = {n: np.array(v["summed_ce"]) for n, v in nat.items()}
    nw = len(CE["fp32"])
    seg = {"V": np.arange(0, 32), "T": np.arange(32, 64)}
    rng = np.random.default_rng(plan["metrics"]["seed"])
    draws = np.concatenate([rng.integers(0, 32, (2000, 32)), 32 + rng.integers(0, 32, (2000, 32))], 1)   # stratified

    def stat(d, idx=None):
        """mean of per-window values d (all windows or a segment) and its bootstrap CI from the shared draws."""
        if idx is None:
            b = d[draws].mean(1)
            m = d.mean()
        else:
            sub = draws[:, :32] if idx[0] == 0 else draws[:, 32:]
            b = d[sub].mean(1)
            m = d[idx].mean()
        return {"mean": float(m), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    res = {"windows": nw, "tokens_per_window": T, "configs": {}, "adjacent": {}, "trend": {}, "tradeoff": {},
           "format_changes": {}, "cross_impl": {}}
    rows = []
    for n in cfg_names():
        d = (CE[n] - CE["fp32"]) / T
        c = {"nll": nat[n]["nll"], "ppl": nat[n]["ppl"], "all_logits_finite": nat[n]["all_logits_finite"],
             "lse_check_max_abs": nat[n]["lse_check_max_abs"],
             "per_segment_nll": {k: float(CE[n][i].sum() / (len(i) * T)) for k, i in seg.items()},
             "per_segment_ppl": {k: float(np.exp(CE[n][i].sum() / (len(i) * T))) for k, i in seg.items()}}
        if n != "fp32":
            c["dnll_vs_fp32"] = {"all": stat(d), **{k: stat(d, i) for k, i in seg.items()}}
            c.update({k: pts[n][k] for k in ("scope", "point", "nodes", "bc_total", "M_bytes", "BOPs", "M_rel", "BOPs_rel",
                                              "R", "w_formats", "a_formats")})
        res["configs"][n] = c
        for i in range(nw):
            rows.append([n, "V" if i < 32 else "T", i, float(CE[n][i]), T, float(CE[n][i] / T), float((CE[n][i] - CE["fp32"][i]) / T)])
    for scope in SCOPES:
        a, k, c = (f"{scope}_{p}" for p in POINTS)
        for label, x, y in (("aggressive_minus_knee", a, k), ("conservative_minus_knee", c, k), ("aggressive_minus_conservative", a, c)):
            d = (CE[x] - CE[y]) / T
            res["adjacent"][f"{scope}|{label}"] = {"all": stat(d), **{s: stat(d, i) for s, i in seg.items()}}
        bc = [pts[n]["bc_total"] for n in (a, k, c)]
        dm = [res["configs"][n]["dnll_vs_fp32"]["all"]["mean"] for n in (a, k, c)]
        ak, ck = res["adjacent"][f"{scope}|aggressive_minus_knee"]["all"], res["adjacent"][f"{scope}|conservative_minus_knee"]["all"]
        res["trend"][scope] = {
            "bc_order_aggr_knee_cons": bc, "bc_predicts_aggr_gt_knee_gt_cons": bool(bc[0] > bc[1] > bc[2]),
            "measured_dnll_aggr_knee_cons": dm, "measured_same_order": bool(dm[0] > dm[1] > dm[2]),
            "aggr_minus_knee_ci_excludes_0": bool(ak["ci95"][0] > 0 or ak["ci95"][1] < 0),
            "cons_minus_knee_ci_excludes_0": bool(ck["ci95"][0] > 0 or ck["ci95"][1] < 0),
            "bc_ratio_aggr_over_knee": bc[0] / bc[1], "bc_ratio_cons_over_knee": bc[2] / bc[1]}
        kp = pts[k]
        res["tradeoff"][scope] = {
            side: {"M_bytes_change": pts[n]["M_bytes"] - kp["M_bytes"], "M_rel_change": pts[n]["M_bytes"] / kp["M_bytes"] - 1,
                   "BOPs_change": pts[n]["BOPs"] - kp["BOPs"], "BOPs_rel_change": pts[n]["BOPs"] / kp["BOPs"] - 1,
                   "M_int8_frac_change": pts[n]["M_rel"] - kp["M_rel"], "BOPs_int8_frac_change": pts[n]["BOPs_rel"] - kp["BOPs_rel"],
                   "componentwise": pts[n]["vs_knee"]["componentwise"],
                   "dnll_change": res["adjacent"][f"{scope}|{side}_minus_knee"]["all"]}
            for side, n in (("aggressive", a), ("conservative", c))}
        # format changes vs the knee, by module kind
        mapping = jload(MAIN / "node_mapping.json")
        fk = jload(out / kp["formats_file"])
        for side, n in (("aggressive", a), ("conservative", c)):
            fx = jload(out / pts[n]["formats_file"])
            ch = {}
            for node in fk:
                if fk[node] != fx[node]:
                    kind = mapping[node]["kind"]
                    key = f"{fk[node]['w']}/{fk[node]['a']} -> {fx[node]['w']}/{fx[node]['a']}"
                    ch.setdefault(kind, {}).setdefault(key, []).append(mapping[node]["layer"])
            res["format_changes"][f"{scope}|{side}"] = {"nodes_changed": sum(len(v) for kd in ch.values() for v in kd.values()),
                                                        "by_kind": ch}
    # cross-implementation on the subset
    sub = np.array(SUBSET)
    sims = {d: jload(out / f"sim_{d}.json") for d in ("cuda", "cpu") if (out / f"sim_{d}.json").exists()}
    for dev, s in sims.items():
        ci = {}
        for n in cfg_names():
            sv = np.array(s["summed_ce"][n]) / T
            nv = CE[n][sub] / T
            ci[n] = {"subset_nll_native": float(nv.mean()), "subset_nll_sim": float(sv.mean()),
                     "max_abs_window_diff": float(np.abs(sv - nv).max()), "sd_window_diff": float((sv - nv).std())}
            if n != "fp32":
                ds = sv - np.array(s["summed_ce"]["fp32"]) / T
                dn = nv - CE["fp32"][sub] / T
                ci[n].update({"subset_dnll_native": float(dn.mean()), "subset_dnll_sim": float(ds.mean()),
                              "dnll_window_corr": float(np.corrcoef(ds, dn)[0, 1])})
        for scope in SCOPES:
            a, k, c = (f"{scope}_{p}" for p in POINTS)
            nat_d = [ci[n]["subset_dnll_native"] for n in (a, k, c)]
            sim_d = [ci[n]["subset_dnll_sim"] for n in (a, k, c)]
            adj_n = [nat_d[0] - nat_d[1], nat_d[2] - nat_d[1]]
            adj_s = [sim_d[0] - sim_d[1], sim_d[2] - sim_d[1]]
            ci[scope] = {"native_order": list(np.argsort(nat_d)[::-1].tolist()), "sim_order": list(np.argsort(sim_d)[::-1].tolist()),
                         "adjacent_native": adj_n, "adjacent_sim": adj_s,
                         "order_agrees": bool(np.argsort(nat_d).tolist() == np.argsort(sim_d).tolist()),
                         "adjacent_signs_agree": bool(all(np.sign(x) == np.sign(y) for x, y in zip(adj_n, adj_s))),
                         "max_point_diff": float(max(abs(x - y) for x, y in zip(nat_d, sim_d))),
                         "half_smallest_native_adjacent": float(0.5 * min(abs(x) for x in adj_n))}
            ci[scope]["cpu_check_triggered"] = bool(not ci[scope]["order_agrees"] or not ci[scope]["adjacent_signs_agree"]
                                                    or ci[scope]["max_point_diff"] > ci[scope]["half_smallest_native_adjacent"])
        res["cross_impl"][dev] = ci
    jdump(out / "analysis.json", res)
    with open(out / "per_window.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "segment", "window", "summed_ce", "tokens", "nll", "dnll_vs_fp32"])
        w.writerows(rows)
    figures(out, pts, res)
    for scope in SCOPES:
        t = res["trend"][scope]
        log(f"{scope}: BC {['%.4g' % x for x in t['bc_order_aggr_knee_cons']]}, dNLL {['%+.5f' % x for x in t['measured_dnll_aggr_knee_cons']]}, "
            f"same order {t['measured_same_order']}")


def figures(out, pts, res):
    """Per scope: BC score of every frontier point and measured dNLL (with 95% CI) of the evaluated points, against
    weight bytes and BOPs relative to all-int8, as separate panels; markers only (no lines through unevaluated
    resource values). Colors: the first three slots of the dataviz reference palette (validated all-pairs there; the
    validator script needs node, absent here), direct labels on the evaluated points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    col = {"aggressive": "#eb6834", "knee": "#2a78d6", "conservative": "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": surface, "axes.facecolor": surface, "text.color": ink})
    for scope, title in SCOPES.items():
        sel = jload(MAIN / f"selection_{scope}.json")
        F = [f for f in sel["frontier"] if f["objective"] > 0]
        fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
        for j, (key, lab) in enumerate((("size", "weight bytes / all-int8 weight bytes (searched nodes)"),
                                        ("bops", "BOPs / all-int8 BOPs (searched nodes)"))):
            rk = "M_rel" if key == "size" else "BOPs_rel"
            a = ax[0, j]
            xmax = 1.3 if key == "size" else 1.7
            shown = [f for f in F if f[key] <= xmax]
            a.scatter([f[key] for f in shown], [f["objective"] for f in shown], s=22, color="#9b9a95", zorder=2,
                      label=f"other frontier points (not evaluated; x > {xmax} not shown)")
            offs = {"aggressive": (10, 10, "left"), "knee": (-12, -22, "right"), "conservative": (10, 10, "left")}
            for p in POINTS:
                v = pts[f"{scope}_{p}"]
                a.scatter([v[rk]], [v["bc_total"]], s=80, color=col[p], zorder=3, edgecolor=surface, linewidth=2, label=p)
                dx, dy, ha = offs[p]
                a.annotate(f"{p}: BC {v['bc_total']:.3g}", (v[rk], v["bc_total"]), textcoords="offset points",
                           xytext=(dx, dy), fontsize=8, color=ink, ha=ha)
            a.set_xlim(right=xmax)
            a.set_yscale("log")
            a.set_xlabel(lab)
            a.set_ylabel("BC total (log scale)")
            a.grid(True, color=grid, linewidth=0.6)
            b = ax[1, j]
            for p in POINTS:
                n = f"{scope}_{p}"
                v, d = pts[n], res["configs"][n]["dnll_vs_fp32"]["all"]
                b.errorbar([v[rk]], [d["mean"]], yerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o",
                           color=col[p], ms=8, capsize=4, lw=2, mec=surface, mew=1.5, label=p)
                b.annotate(f"{p}\n{d['mean']:+.4f}", (v[rk], d["mean"]), textcoords="offset points", xytext=(9, -4),
                           fontsize=8, color=ink)
            b.axhline(0, color=ink2, lw=0.8)
            b.set_xlabel(lab)
            b.set_ylabel("native dNLL vs FP32 (nat/token), 95% CI")
            b.grid(True, color=grid, linewidth=0.6)
        ax[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
        ax[1, 0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle(f"Qwen3.5-0.8B, {title}: BC on the frontier (top) and native damage on 64 fresh windows (bottom)",
                     color=ink, fontsize=11)
        fig.savefig(out / f"fig_{scope}.png", dpi=150, facecolor=surface)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("audit", "plan", "native", "sim", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--configs", default=",".join(cfg_names()), help="native: comma list")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    status = "ok"
    try:
        globals()[f"stage_{args.stage}"](args, out, log)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        import resource
        timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
        timing["runs"].append({"stage": args.stage, "configs": args.configs if args.stage == "native" else None,
                               "device": args.device, "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                               "peak_rss_gb_largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
