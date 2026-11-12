#!/usr/bin/env python3
"""Cross-architecture regression of the LP-first search pipeline (pre-registered in plan.json, one --out-dir).

Problems: MobileNetV2, ResNet18, EfficientNet-B0, GPT-2, BERT, Qwen3, Qwen3.5 FFN-only, Qwen3.5 FFN + attention
(their current native BC tables, all finite options, as their pipelines search them) and OLMoE fixed route (3051
expert projections, exact dominance pruning, as its pipeline searched it). BC, tables, ranges, formats and the resource
model are unchanged; mp_milp / mp_auto are not edited; no evaluation loss enters any rule, threshold, budget or knee.

Solvers (run_olmoe_search_final.make_solver, unchanged):
  milp     corrected MILP: attempt 1 at the exact budget, the 1 ppm retry if attempt 1 is over budget, and only if the
           retry fails a repair of attempt 1 (labelled, not proven optimal)
  lp       LP relaxation + rounding + repair
  lpfirst  lp accepted if objective - LP bound <= max(1e-3 objective, 1e-6 objective scale), else milp

Stages:
  plan        plan.json (budgets of experiment A fixed here)
  a_solve     --problem P --method M --rep r: every planned budget of P solved by M (fresh process, no cache)
  a_analyze   per budget and per problem: feasibility, BC / M / B, differences to MILP, numerics, timing
  b_search    --problem P --solver S --rule orig|grouped: the auto_select copy with the refinement neighbour rule;
              every (solver, budget) is solved at most once in this round (experiment A's rep 1 and earlier B solves
              are reused, the same policy for every solver and rule); orig is checked against mp_auto.auto_select
  b_offline   order / duplicate / perturbation robustness of the knee and of the refinement interval
  b_analyze
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG = HERE / "runs/solver_regression_20261016"
FINAL = HERE / "runs/olmoe_search_final_20261016"
DENSE_KEYS = ["mobilenetv2", "resnet18", "efficientnet_b0", "gpt2", "bert", "qwen3", "qwen35_ffn", "qwen35_ffn_attn"]
PROBLEMS = DENSE_KEYS + ["olmoe"]
METHODS = ("milp", "lp", "lpfirst")
SAMPLES, REFINEMENTS = 17, 2
TL = {"dense": 10.0, "olmoe": 300.0}
OLMOE_DIAG = [0.6771270929368864, 0.707075759814072, 0.7071062114158619, 0.7383791757708325]
OLMOE_REP23 = [0.5, 0.7071067811865476, 2.8284271247461903]          # OLMoE MILP repetitions 2-3 (cost cap)
REL_TOL, ABS_TOL = 1e-3, 1e-6
SCRIPTS = ["run_lp_pipeline_regression.py", "run_olmoe_search_final.py", "run_solver_regression.py", "mp_auto.py",
           "mp_milp.py", "run_native_bc_pipeline.py", "run_olmoe_multipoint.py", "htq_pareto_search.py"]


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_problem(key, work):
    """(nodes, names, value, elems, flops, time_limit, build seconds) as the model's pipeline searches it."""
    t0 = time.perf_counter()
    if key == "olmoe":
        from run_olmoe_search_final import olmoe_problem
        nodes, names, value, elems, flops, cost = olmoe_problem()
        return nodes, names, value, elems, flops, TL["olmoe"], {**cost, "total_s": time.perf_counter() - t0}
    from run_solver_regression import dense_problem
    nodes, names, raw, elems, flops, _ = dense_problem(key, work)

    def value(n, w, a):                        # non-finite costs forbid the option, as auto_select's lookup does
        v = raw(n, w, a)
        return v if v is not None and math.isfinite(v) else None
    return nodes, names, value, elems, flops, TL["dense"], {"spec_and_table_s": time.perf_counter() - t0}


def table_path(key):
    from run_solver_regression import DENSE
    return HERE / ("runs/olmoe_multipoint_20261015/table.json" if key == "olmoe" else DENSE[key][1])


def endpoints(nodes, names, value, elems, flops):
    """auto_select's option list, endpoints, lower / upper bounds and coarse budgets (same arithmetic)."""
    from htq_pareto_search import FORMAT_BITS
    from run_native_bc_pipeline import GRID11
    ref_bytes = sum(elems[n] for n in names)
    ref_bops = sum(64 * flops[n] for n in names)
    options = {}
    for n in names:
        opts = []
        for w in GRID11:
            for a in GRID11:
                v = value(n, w, a)
                if v is None or not math.isfinite(v):
                    continue
                opts.append((w, a, float(v), elems[n] * FORMAT_BITS[w] / 8, flops[n] * FORMAT_BITS[w] * FORMAT_BITS[a]))
        options[n] = opts

    def endpoint(key):
        cfg = {}
        for n in names:
            w, a, *_ = min(options[n], key=key)
            cfg[n] = (w, a)
        obj = sum(next(o[2] for o in options[n] if (o[0], o[1]) == cfg[n]) for n in names)
        size = sum(elems[n] * FORMAT_BITS[cfg[n][0]] / 8 for n in names) / ref_bytes
        bops = sum(flops[n] * FORMAT_BITS[cfg[n][0]] * FORMAT_BITS[cfg[n][1]] for n in names) / ref_bops
        return {"objective": obj, "size": size, "bops": bops, "resource": max(size, bops)}
    cheap = endpoint(lambda o: (o[3], o[4], o[2]))
    quality = endpoint(lambda o: (o[2], max(o[3] / ref_bytes, o[4] / ref_bops)))
    lower = max(sum(min(o[3] for o in options[n]) for n in names) / ref_bytes,
                sum(min(o[4] for o in options[n]) for n in names) / ref_bops)
    upper = max(cheap["resource"], quality["resource"])
    allc = np.array([o[2] for n in names for o in options[n]])
    return {"cheap": cheap, "quality": quality, "lower": lower, "upper": upper,
            "coarse_budgets": [float(b) for b in np.geomspace(lower, upper, SAMPLES)],
            "options": int(sum(len(v) for v in options.values())), "max_abs_cost": float(np.abs(allc).max()),
            "objective_scale": float(np.abs(allc).max()) / 1e4, "span_S": cheap["objective"] - quality["objective"],
            "ref_bytes": ref_bytes, "ref_bops": ref_bops}


def group_delta(lower, upper):
    """Half the finest planned refinement spacing: coarse ratio g = (upper/lower)^(1/16), two refinements of five
    geometric points each over a neighbour interval reach a spacing of about g^(1/4)."""
    if upper <= lower:
        return 0.0
    return ((upper / lower) ** (1.0 / (4 * (SAMPLES - 1))) - 1) / 2


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    key = args.problem
    f = out / "plan_budgets.json"
    pb = jload(f) if f.exists() else {}
    if key:                                    # one problem's budgets (separate processes: transformers versions)
        nodes, names, value, elems, flops, tl, build = load_problem(key, out / "spec_work")
        ep = endpoints(nodes, names, value, elems, flops)
        if key == "olmoe":
            diag = OLMOE_DIAG
        else:
            r = jload(REG / "regress.json")[key]["variants"]
            diag = sorted({b for v in ("milp_repair", "lp_first") for b in r[v]["attempted_budgets"][17:]})
        diag = [b for b in diag if not any(math.isclose(b, c, rel_tol=1e-12) for c in ep["coarse_budgets"])]
        pb[key] = {**{k: v for k, v in ep.items()}, "diagnostic_budgets": diag, "nodes": len(names), "time_limit_s": tl,
                   "group_delta": group_delta(ep["lower"], ep["upper"]), "table": str(table_path(key).relative_to(HERE)),
                   "table_sha256": sha256(table_path(key)), "build": build}
        jdump(f, pb)
        log(f"{key}: {len(names)} nodes, {ep['options']} options, lower {ep['lower']:.4f} upper {ep['upper']:.4f}, "
            f"{len(diag)} diagnostic budgets, delta {pb[key]['group_delta']:.4f}")
        return
    missing = [p for p in PROBLEMS if p not in pb]
    if missing:
        raise SystemExit(f"budgets missing for {missing}")
    sc = {s: sha256(HERE / s) for s in SCRIPTS}
    jdump(out / "plan.json", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "fixed route only for OLMoE; no dynamic route; BC criterion, tables, calibration, ranges, formats, "
                 "resource model unchanged; mp_milp / mp_auto / pipeline defaults unchanged; no evaluation result is "
                 "used for any rule, threshold, budget or knee",
        "background_verified": {
            "1": "OLMoE fixed route: corrected MILP auto_select 3294.8 s, pure LP 14.3 s, LP first 76.5 s (one run each, "
                 "olmoe_search_final); 512 new windows GPU: LP - MILP +0.00062 [-0.00030, +0.00154] inside +-0.002 - "
                 "confirmed",
            "2": "dense regression (solver_regression): LP first changed the knee on Qwen3 (37 projections, BC +18.7%) and "
                 "Qwen3.5 FFN+attn (31, +8.2%); same knee elsewhere with 2-10 of 23 budgets accepted from LP; no loss "
                 "evaluation of the differing dense knees - confirmed. Difference: that regression's LP-first fallback "
                 "was one MILP + repair, not the corrected retry path; on the dense problems the retry never failed, so "
                 "the corrected MILP equals mp_milp there",
            "3": "near-duplicate resource points change neighbours and the refinement side; the corrected MILP did it on "
                 "OLMoE (0.7071056 vs 0.7071066) - confirmed",
            "4": "LP / MILP bounds unreliable for tiny objectives and wide cost spans (Qwen3 LP bound above the MILP "
                 "optimum at R >= 6.7; OLMoE MILP 'optimal' 1.7e-10 worse than LP's 4e-12 at R 8; HiGHS 'excessively "
                 "small costs') - confirmed",
            "5": "OLMoE evaluations: corrected GPU activation quantizer (device scale / zero-point), CPU with "
                 "OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE, GPU TF32 off; earlier and corrected results kept apart - "
                 "confirmed. The same quantizer issue exists in every GPU replay that uses "
                 "qwen3_swiglu_common.Quantizers.act (e.g. the Qwen3.5 Replay); this round applies the same correction "
                 "to any GPU replay it runs"},
        "problems": {k: {kk: v[kk] for kk in ("table", "table_sha256", "nodes", "options", "time_limit_s", "lower",
                                              "upper", "group_delta", "span_S", "objective_scale")}
                     for k, v in pb.items()},
        "budgets_file_sha256": sha256(f),
        "scripts_sha256": sc,
        "experiment_A": {
            "question": "same problem, same budget: does LP (alone / first) find an acceptable feasible configuration, "
                        "where does it save time, where does it only add an LP before the MILP, and why",
            "budgets": "per problem: the 17 coarse budgets auto_select derives from the cost table (geomspace(lower, "
                       "upper, 17)) + diagnostic budgets = the refinement budgets of the earlier MILP and LP-first runs "
                       "(solver_regression for dense problems; for OLMoE 4 of olmoe_search_final's: 0.67713, 0.70708, "
                       "0.707106, 0.73838); listed in plan_budgets.json; no method adds budgets",
            "methods": ["milp", "lp", "lpfirst"],
            "records": "per budget: path (lp / rounding / repair / milp / retry / repaired / fallback), status, BC, M, B "
                       "recomputed from the configuration, exact feasibility, |BC - BC_milp| absolute, relative and / "
                       "span_S, differing projections vs MILP, fractional LP nodes, lower bound, gap, HiGHS warnings, "
                       "repair swaps, seconds of every step",
            "span": "span_S = BC(cheapest-resource endpoint) - BC(minimum-BC endpoint), from the cost table before any "
                    "solve; span 0: differences / span reported as 0 when the difference is 0, else undefined; the "
                    "knee normalization span BC(lowest-budget MILP point) - BC(minimum-BC endpoint) reported "
                    "afterwards as a diagnostic",
            "numerics": "status optimal is not a proof; a lower bound above the objective of any exactly feasible "
                        "configuration known at that budget (any method in this round, budgets <= it, endpoints) by "
                        "more than 1e-9 max(1, |objective|) is 'suspect'; objectives below 1e-6 x objective scale are "
                        "'below solver resolution' (absolute differences only); no guaranteed error bound is claimed "
                        "from a suspect bound; the same-budget MILP comparison is research evaluation only, never an "
                        "acceptance condition",
            "acceptance_rule": "the existing LP-first rule unchanged; no new numerical rule is registered in this round",
            "timing": "same machine, nothing else heavy running, one process per (problem, method, repetition), order "
                      "rep-major then milp, lp, lpfirst; 3 repetitions, median and range; table / spec load, problem "
                      "build (incl. dominance pruning), per-call array build and every solver step timed separately; "
                      "OLMoE milp: repetition 1 at all budgets, repetitions 2-3 only at " + str(OLMOE_REP23) +
                      " (cost cap, about 70 min per full repetition); OLMoE lp / lpfirst all budgets 3 times; no warm "
                      "start or cache in experiment A",
            "stop": "OLMoE milp repetition 1 over 2.5 h: stop and report the completed budgets"},
        "experiment_B": {
            "question": "does adaptive refinement amplify small solver differences; does one neighbour rule change make "
                        "the search more stable without extra solves",
            "rules": {"orig": "mp_auto.auto_select unchanged (a copy, verified identical to mp_auto.auto_select)",
                      "grouped": "only the refinement neighbours change: frontier points (sorted by actual resource R) "
                                 "are grouped left to right, a new group starting when R > R(first point of the group) "
                                 "x (1 + delta); the knee is still chosen by the unchanged score over all frontier "
                                 "points; the refinement interval is from the highest-R point of the nearest group below "
                                 "the knee's group to the lowest-R point of the nearest group above (no group below: the "
                                 "lowest-R point of the knee's group; no group above: its highest-R point); the same "
                                 "geomspace(left, right, 5)[1:-1] budgets, the same 2 rounds, 3 solves each; no point is "
                                 "removed, averaged or added; the final selection rule is unchanged"},
            "delta": "delta = ((upper/lower)^(1/64) - 1) / 2: half the finest planned refinement spacing (coarse ratio "
                     "(upper/lower)^(1/16), two refinements reach ratio^(1/4)); actual R (not the budget) because "
                     "neighbours and scores use actual R; from the search grid only",
            "runs": "rule x solver in {orig, grouped} x {milp, lpfirst} for every problem (lp alone also run with both "
                    "rules for reference)",
            "reuse": "each (solver, budget) is solved at most once in this round: experiment A repetition 1 and earlier "
                     "B solves are reused for every solver and rule alike (deterministic solver); reported search cost = "
                     "sum of the one-time solve seconds of the budgets that run requested + the measured bookkeeping",
            "analysis": ["per round: sampled budgets, actual R, knee, neighbours (and groups), next budgets",
                         "final BC, M, B, differing projections vs orig+milp, search cost",
                         "knee score of the best point and the best point whose R differs > 0.5%",
                         "Qwen3 / Qwen3.5 FFN+attn: does grouped reduce the MILP / LP-first divergence; other problems: "
                         "unnecessary changes; lowest-budget endpoint present",
                         "union of all candidates per problem as a post-hoc reference frontier (not fed back, not called "
                         "the true Pareto frontier)"],
            "offline": "on each run's final candidates: 100 random orderings (seed 0) and all candidates duplicated must "
                       "give the same knee and refinement interval; perturbations x (1 + eps u), u ~ U(-1, 1) of R and / "
                       "or BC, eps in {1e-8, 1e-6, 1e-4}, 200 draws (seed 0): share of draws changing the knee point, "
                       "the knee group, the orig and the grouped interval; perturbed points are not real configurations"},
        "experiment_C": {
            "when": "a problem is evaluated only if the final configurations of its B runs (orig/grouped x milp/lpfirst) "
                    "are not all identical; identical configuration, ranges, semantics and execution path: no new "
                    "evaluation (OLMoE: the 512-window evidence cited if the configurations are unchanged)",
            "configs": "every distinct final configuration + FP32; reference = orig + milp (the current pipeline)",
            "freeze": "configurations and analysis rules frozen before new data is read",
            "margins": {"language models": "non-inferiority margin = 10% of the reference knee's previously measured "
                                           "native dNLL vs FP32 (Qwen3: 0.0750 on 32 test windows, "
                                           "qwen3_branch_ablation heldout -> 0.0075 nat/token; Qwen3.5 FFN+attn: 0.0323 "
                                           "on 64 windows, qwen35_multipoint -> 0.0032; Qwen3.5 FFN: 0.0098 -> 0.0010; "
                                           "GPT-2 / BERT / OLMoE: from their earlier knee evaluations if needed); "
                                           "rationale: a solver / rule change may not add more than a tenth of the "
                                           "knee's own quantization damage",
                        "CNN": "10% of the reference knee's previously measured top-1 drop and loss increase vs FP32",
                        "not available": "no margin: effect sizes and intervals only"},
            "claims": "primary: non-inferiority of each alternative to the reference (upper bound of the two-sided "
                      "(1 - 0.05/k) paired CI < margin, k = alternatives of that problem, Bonferroni); secondary: "
                      "equivalence (CI inside +-margin); 'CI contains 0' is not equivalence; differing resources are "
                      "reported as end-to-end selection comparisons",
            "sample_size": "n = ((z_{1-0.025/k} + z_0.8) SD / margin)^2 with SD the per-window SD of a paired difference "
                           "between knees of that model in earlier data (Qwen3 0.036, Qwen3.5 FFN+attn 0.030), fixed "
                           "before evaluation, capped by cost (amendment states the cap and the resulting power)",
            "backend": "native where affordable; otherwise GPU PyTorch replay labelled as such (corrected device "
                       "quantizer, TF32 off) with a native cross-check subset reported separately",
            "data": "new windows / samples, overlap with every earlier window of the same model checked"},
        "not_claimed": ["BOPs = hardware speed-up", "storage estimate = runtime memory", "results for dynamic route",
                        "a new default"]})
    log("plan written")


# --------------------------------------------------------------------------- #
# experiment A
# --------------------------------------------------------------------------- #
def stage_a_solve(args, out, log):
    from run_native_bc_pipeline import GRID11
    from run_olmoe_search_final import make_solver
    key, m, rep = args.problem, args.method, args.rep
    d = out / "A" / key
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{m}_rep{rep}.json"
    if f.exists():
        raise SystemExit(f"{f.name} exists")
    pb = jload(out / "plan_budgets.json")[key]
    budgets = pb["coarse_budgets"] + pb["diagnostic_budgets"]
    if key == "olmoe" and m == "milp" and rep > 1:
        budgets = OLMOE_REP23
    t0 = time.perf_counter()
    nodes, names, value, elems, flops, tl, build = load_problem(key, out / "spec_work")
    t_build = time.perf_counter() - t0
    ref_b, ref_o = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
    (d / "logs").mkdir(exist_ok=True)
    recs, rows = [], []
    solver = make_solver(m, recs, d / "logs", time_limit=tl, tag=f"A_rep{rep}")
    from htq_pareto_search import FORMAT_BITS as FB
    for b in budgets:
        ts = time.perf_counter()
        try:
            cfg, obj, status = solver(nodes, GRID11, GRID11, value, elems, flops, b * ref_b, b * ref_o, time_limit=tl)
            fm = {n: cfg[n] for n in names}
            size = sum(elems[n] * FB[fm[n]["w"]] / 8 for n in names) / ref_b
            bops = sum(flops[n] * FB[fm[n]["w"]] * FB[fm[n]["a"]] for n in names) / ref_o
            row = {"budget": b, "objective": obj, "status": status, "size": size, "bops": bops, "resource": max(size, bops),
                   "exact_feasible": bool(recs[-1]["feasible"]), "relative_check": bool(size <= b and bops <= b),
                   "call": recs[-1],
                   "formats": fm if rep == 1 else None}
        except RuntimeError as exc:
            row = {"budget": b, "failure": str(exc), "call": recs[-1] if recs else None}
        row["wall_s"] = time.perf_counter() - ts
        rows.append(row)
        log(f"{key} {m} rep{rep} R {b:.6f}: {row.get('call', {}).get('path') if row.get('call') else None} "
            f"obj {row.get('objective')} ({row['wall_s']:.2f}s)")
    from run_olmoe_search_final import env_info
    jdump(f, {"problem": key, "method": m, "rep": rep, "budgets": budgets, "problem_build_s": t_build, "build": build,
              "solve_total_s": float(sum(r["wall_s"] for r in rows)), "rows": rows, "env": env_info(),
              "script_sha256": sha256(HERE / "run_lp_pipeline_regression.py")})


def stage_a_analyze(args, out, log):
    pbs = jload(out / "plan_budgets.json")
    res = {}
    for key in PROBLEMS:
        d = out / "A" / key
        if not (d / "milp_rep1.json").exists():
            continue
        pb = pbs[key]
        R = {(m, r): jload(d / f"{m}_rep{r}.json") for m in METHODS for r in (1, 2, 3) if (d / f"{m}_rep{r}.json").exists()}
        rows = {m: {round(x["budget"], 15): x for x in R[(m, 1)]["rows"]} for m in METHODS if (m, 1) in R}
        scale, S = pb["objective_scale"], pb["span_S"]
        known = [(pb["cheap"]["resource"], pb["cheap"]["objective"]), (pb["quality"]["resource"], pb["quality"]["objective"])]
        for m in rows:
            known += [(x["resource"], x["objective"]) for x in rows[m].values() if "objective" in x]
        lo_mi = rows["milp"].get(round(pb["coarse_budgets"][0], 15), {}).get("objective")
        S_knee = (lo_mi - pb["quality"]["objective"]) if lo_mi is not None else None
        per_budget = []
        for bk in sorted(rows["milp"]):
            e = {"budget": bk, "coarse": any(math.isclose(bk, c, rel_tol=1e-12) for c in pb["coarse_budgets"])}
            mi = rows["milp"][bk]
            for m in rows:
                x = rows[m].get(bk)
                if x is None:
                    continue
                if "objective" not in x:
                    e[m] = {"failure": x["failure"]}
                    continue
                c = x["call"]
                steps = {s["step"]: s for s in c["steps"]}
                best_known = min([o for (rr, o) in known if rr <= bk + 1e-12])
                lb = c.get("lower_bound")
                r = {"objective": x["objective"], "size": x["size"], "bops": x["bops"], "resource": x["resource"],
                     "exact_feasible": x["exact_feasible"], "path": c["path"], "status": x["status"],
                     "lower_bound": lb, "gap_abs": c.get("gap_abs"),
                     "fractional_nodes": steps.get("round_repair", {}).get("fractional_nodes"),
                     "lp_gap": steps.get("round_repair", {}).get("gap"), "lp_tol": steps.get("round_repair", {}).get("tolerance"),
                     "lp_round_swaps": steps.get("round_repair", {}).get("swaps"),
                     "lp_rounded_feasible": steps.get("round_repair", {}).get("rounded", {}).get("feasible"),
                     "milp_repair_swaps": steps.get("repair_attempt1", {}).get("swaps"),
                     "below_solver_resolution": bool(x["objective"] < ABS_TOL * scale),
                     "highs_warnings": sorted({w for s in c["steps"] for w in s.get("highs", {}).get("warnings", [])}),
                     "bound_suspect": bool(lb is not None and (lb > x["objective"] + 1e-9 * max(1, abs(x["objective"]))
                                                                or lb > best_known + 1e-9 * max(1, abs(best_known)))),
                     "step_seconds": {s["step"]: s["seconds"] for s in c["steps"]}}
                if m != "milp" and "objective" in mi:
                    dlt = x["objective"] - mi["objective"]
                    r.update(bc_minus_milp=dlt, bc_rel_milp=(dlt / mi["objective"] if mi["objective"] > 0 else None),
                             bc_over_span=(dlt / S if S > 0 else (0.0 if dlt == 0 else None)),
                             bc_over_knee_span=(dlt / S_knee if S_knee else None),
                             differing_projections=(sum(x["formats"][n] != mi["formats"][n] for n in x["formats"])
                                                    if x.get("formats") and mi.get("formats") else None))
                e[m] = r
            per_budget.append(e)
        timing = {}
        for m in METHODS:
            reps = [R[(m, r)] for r in (1, 2, 3) if (m, r) in R]
            same = [x for x in reps if len(x["budgets"]) == len(reps[0]["budgets"])]
            tot = [x["solve_total_s"] for x in same]
            timing[m] = {"repetitions_full": len(same), "solve_total_median_s": float(np.median(tot)) if tot else None,
                         "solve_total_range_s": [float(min(tot)), float(max(tot))] if tot else None,
                         "problem_build_median_s": float(np.median([x["problem_build_s"] for x in reps])),
                         "per_step_rep1_s": {}}
            for x in reps[:1]:
                agg = {}
                for row in x["rows"]:
                    for s in (row.get("call") or {}).get("steps", []):
                        agg[s["step"]] = agg.get(s["step"], 0.0) + s["seconds"]
                    agg["array_build"] = agg.get("array_build", 0.0) + (row.get("call") or {}).get("build_s", 0.0)
                timing[m]["per_step_rep1_s"] = agg
            if key == "olmoe" and m == "milp":
                timing[m]["rep23_budgets"] = OLMOE_REP23
                timing[m]["per_budget_seconds_all_reps"] = {
                    f"{b:.6f}": [next((row["wall_s"] for row in R[("milp", r)]["rows"] if math.isclose(row["budget"], b, rel_tol=1e-12)), None)
                                 for r in (1, 2, 3) if ("milp", r) in R] for b in OLMOE_REP23}
        lf = [e["lpfirst"] for e in per_budget if "lpfirst" in e and "path" in e["lpfirst"]]
        cause = {}
        for e in per_budget:
            x = e.get("lpfirst")
            if not x or "path" not in x or not x["path"].startswith("lp->"):
                continue
            c = ("lp bound suspect" if e.get("lp", {}).get("bound_suspect") else
                 "rounding: fractional nodes" if (x.get("fractional_nodes") or 0) > 0 and (x.get("lp_round_swaps") or 0) == 0 else
                 "rounding + repair" if (x.get("lp_round_swaps") or 0) > 0 else "LP integral but gap > tol")
            cause[c] = cause.get(c, 0) + 1
        res[key] = {"nodes": pb["nodes"], "options": pb["options"], "span_S": S, "knee_span": S_knee,
                    "objective_scale": scale, "budgets": len(per_budget),
                    "lpfirst_accepted": sum(x["path"] == "lp" for x in lf), "lpfirst_fallback": sum(x["path"] != "lp" for x in lf),
                    "lpfirst_fallback_causes": cause,
                    "all_feasible": all(e[m].get("exact_feasible", False) for e in per_budget for m in METHODS if m in e),
                    "failures": {m: sum("failure" in e.get(m, {}) for e in per_budget) for m in METHODS},
                    "suspect_bounds": {m: [e["budget"] for e in per_budget if e.get(m, {}).get("bound_suspect")] for m in METHODS},
                    "highs_warnings": sorted({w for e in per_budget for m in METHODS for w in e.get(m, {}).get("highs_warnings", [])}),
                    "timing": timing, "per_budget": per_budget}
        log(f"{key}: lpfirst LP {res[key]['lpfirst_accepted']}/{len(lf)}, causes {cause}, timing "
            + ", ".join(f"{m} {timing[m]['solve_total_median_s']}" for m in METHODS))
    jdump(out / "A" / "analysis.json", res)


# --------------------------------------------------------------------------- #
# experiment B
# --------------------------------------------------------------------------- #
def groups_of(frontier, delta):
    gs = []
    for i, r in enumerate(frontier):
        if gs and r["resource"] <= frontier[gs[-1][0]]["resource"] * (1 + delta):
            gs[-1].append(i)
        else:
            gs.append([i])
    return gs


def interval(frontier, i, rule, delta):
    if rule == "orig":
        return frontier[max(0, i - 1)], frontier[min(len(frontier) - 1, i + 1)]
    gs = groups_of(frontier, delta)
    gi = next(k for k, g in enumerate(gs) if i in g)
    left = frontier[gs[gi - 1][-1]] if gi > 0 else frontier[gs[gi][0]]
    right = frontier[gs[gi + 1][0]] if gi < len(gs) - 1 else frontier[gs[gi][-1]]
    return left, right


def auto_select_rule(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy, solve, *, rule, delta,
                     samples=SAMPLES, refinements=REFINEMENTS, time_limit=60, trace=None):
    """mp_auto.auto_select with the solver injected and the refinement neighbour rule selectable (rule='orig' is the
    same algorithm); trace receives each round's knee, neighbours, groups and next budgets."""
    from htq_pareto_search import FORMAT_BITS
    from mp_auto import _frontier, _knee
    names = [n['name'] for n in nodes if n.get('weight_const')]
    ref_bytes = sum(weight_elems[n] for n in names)
    ref_bops = sum(64 * flops_proxy[n] for n in names)
    options, excluded = {}, 0
    for n in names:
        opts = []
        for w in weight_formats:
            for a in act_formats:
                v = value(n, w, a)
                if v is None or not math.isfinite(v):
                    excluded += 1
                    continue
                opts.append((w, a, float(v), weight_elems[n] * FORMAT_BITS[w] / 8,
                             flops_proxy[n] * FORMAT_BITS[w] * FORMAT_BITS[a]))
        options[n] = opts
    lookup = {(n, w, a): v for n, opts in options.items() for w, a, v, _, _ in opts}

    def cost(n, w, a):
        return lookup.get((n, w, a))

    def record(cfg, budget, status, origin):
        objective = sum(lookup[n, cfg[n]['w'], cfg[n]['a']] for n in names)
        size = sum(weight_elems[n] * FORMAT_BITS[cfg[n]['w']] / 8 for n in names) / ref_bytes
        bops = sum(flops_proxy[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']] for n in names) / ref_bops
        return dict(formats=cfg, objective=objective, size=size, bops=bops, resource=max(size, bops),
                    search_budget=budget, solver_status=status, optimal=status == 0, origin=origin)

    def assignment(key):
        cfg = {n['name']: {'w': None, 'a': 'fp32'} for n in nodes if not n.get('weight_const')}
        for n in names:
            w, a, *_ = min(options[n], key=key)
            cfg[n] = {'w': w, 'a': a}
        return cfg
    cheap = record(assignment(lambda o: (o[3], o[4], o[2])), None, None, 'cheap_endpoint')
    quality = record(assignment(lambda o: (o[2], max(o[3] / ref_bytes, o[4] / ref_bops))), None, 0, 'minimum_bc_endpoint')
    lower = max(sum(min(o[3] for o in options[n]) for n in names) / ref_bytes,
                sum(min(o[4] for o in options[n]) for n in names) / ref_bops)
    upper = max(cheap['resource'], quality['resource'])
    candidates, failures, attempted = [cheap, quality], [], []

    def solve_one(budget):
        if any(math.isclose(budget, b, rel_tol=1e-12) for b in attempted):
            return
        attempted.append(float(budget))
        try:
            cfg, _, status = solve(nodes, weight_formats, act_formats, cost, weight_elems, flops_proxy,
                                   budget * ref_bytes, budget * ref_bops, time_limit=time_limit)
        except RuntimeError as exc:
            failures.append({'budget': float(budget), 'message': str(exc)})
            return
        row = record(cfg, float(budget), status, 'milp')
        if row['resource'] > budget + 1e-12:
            raise RuntimeError('auto MILP answer exceeds its resource budget')
        candidates.append(row)

    def note(r, fr, sel, left, right, nxt):
        if trace is not None:
            i = fr.index(sel)
            trace.append({"after_round": r, "attempted_so_far": len(attempted),
                          "knee": {k: sel[k] for k in ("resource", "objective", "search_budget")},
                          "left": {k: left.get(k) for k in ("resource", "objective", "search_budget")},
                          "right": {k: right.get(k) for k in ("resource", "objective", "search_budget")},
                          "orig_neighbours": [fr[max(0, i - 1)]["resource"], fr[min(len(fr) - 1, i + 1)]["resource"]],
                          "groups": [[fr[j]["resource"] for j in g] for g in groups_of(fr, delta)],
                          "next_budgets": nxt})
    if quality['objective'] != cheap['objective'] or cheap['resource'] > lower:
        for b in np.geomspace(lower, upper, samples):
            solve_one(float(b))
        for r in range(refinements):
            frontier = _frontier(candidates)
            selected, _ = _knee(frontier)
            i = frontier.index(selected)
            left, right = interval(frontier, i, rule, delta)
            nxt = [float(b) for b in np.geomspace(left['resource'], right['resource'], 5)[1:-1]]
            note(r, frontier, selected, left, right, nxt)
            for b in nxt:
                solve_one(b)
    frontier = _frontier(candidates)
    selected, scores = _knee(frontier)
    note(refinements, frontier, selected, selected, selected, None)
    return dict(selected=selected, frontier=frontier, knee_scores=scores, candidates=candidates,
                attempted_budgets=attempted, failures=failures, bounds={'lower': lower, 'upper': upper},
                excluded_nonfinite_options=excluded)


class SolveStore:
    """One solve per (solver, budget) in this round, shared by every rule and run: experiment A's repetition 1 rows
    and earlier B solves are loaded; a new budget is solved once and stored."""

    def __init__(self, out, key, method, tl):
        from run_olmoe_search_final import make_solver
        self.f = out / "B" / key / f"store_{method}.json"
        self.f.parent.mkdir(parents=True, exist_ok=True)
        self.rows = jload(self.f)["rows"] if self.f.exists() else []
        if not self.rows:
            a = out / "A" / key / f"{method}_rep1.json"
            if a.exists():
                self.rows = [{**r, "source": "A_rep1"} for r in jload(a)["rows"]]
        self.recs = []
        self.solver = make_solver(method, self.recs, out / "B" / key, time_limit=tl, tag="B")
        self.used = []

    def __call__(self, nodes, wf, af, value, elems, flops, max_bytes, max_bops, time_limit=600, info=None):
        names = [n["name"] for n in nodes if n.get("weight_const")]
        b = max_bytes / sum(elems[n] for n in names)
        hit = next((r for r in self.rows if math.isclose(r["budget"], b, rel_tol=1e-12)), None)
        if hit is None:
            ts = time.perf_counter()
            try:
                cfg, obj, status = self.solver(nodes, wf, af, value, elems, flops, max_bytes, max_bops, time_limit=time_limit)
                hit = {"budget": b, "objective": obj, "status": status, "formats": {n: cfg[n] for n in names},
                       "call": self.recs[-1], "source": "B"}
            except RuntimeError as exc:
                hit = {"budget": b, "failure": str(exc), "call": self.recs[-1] if self.recs else None, "source": "B"}
            hit["wall_s"] = time.perf_counter() - ts
            self.rows.append(hit)
            jdump(self.f, {"rows": self.rows})
        self.used.append({"budget": b, "source": hit["source"], "wall_s": hit["wall_s"],
                          "path": (hit.get("call") or {}).get("path")})
        if "failure" in hit:
            raise RuntimeError(hit["failure"])
        cfg = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
        cfg.update(hit["formats"])
        return cfg, hit["objective"], hit["status"]


def stage_b_search(args, out, log):
    import mp_auto
    from mp_auto import config_hash
    from run_native_bc_pipeline import GRID11
    key = args.problem
    pb = jload(out / "plan_budgets.json")[key]
    nodes, names, value, elems, flops, tl, _ = load_problem(key, out / "spec_work")
    d = out / "B" / key
    d.mkdir(parents=True, exist_ok=True)
    for solver_name in args.solvers.split(","):
        store = SolveStore(out, key, solver_name, tl)
        for rule in ("orig", "grouped"):
            f = d / f"search_{solver_name}_{rule}.json"
            if f.exists():
                continue
            store.used = []
            trace = []
            t0 = time.perf_counter()
            sel = auto_select_rule(nodes, GRID11, GRID11, value, elems, flops, store, rule=rule, delta=pb["group_delta"],
                                   time_limit=tl, trace=trace)
            wall = time.perf_counter() - t0
            new_s = sum(u["wall_s"] for u in store.used if u["source"] == "B")
            cost = sum(u["wall_s"] for u in store.used)
            check = None
            if rule == "orig":                     # the copy against mp_auto.auto_select with the same (stored) solves
                orig = mp_auto.milp_select
                mp_auto.milp_select = store
                try:
                    ref = mp_auto.auto_select(nodes, GRID11, GRID11, value, elems, flops, samples=SAMPLES,
                                              refinements=REFINEMENTS, time_limit=tl)
                finally:
                    mp_auto.milp_select = orig
                check = {"same_attempted": ref["attempted_budgets"] == sel["attempted_budgets"],
                         "same_selection": config_hash({n: ref["selected"]["formats"][n] for n in names})
                         == config_hash({n: sel["selected"]["formats"][n] for n in names})}
                if not all(check.values()):
                    raise SystemExit(f"{key} {solver_name}: copy differs from mp_auto.auto_select {check}")
            K = sel["selected"]
            jdump(f, {"problem": key, "solver": solver_name, "rule": rule, "delta": pb["group_delta"],
                      "search_cost_s": cost, "new_solve_s": new_s, "bookkeeping_wall_s": wall - new_s, "used": store.used[:],
                      "trace": trace, "attempted_budgets": sel["attempted_budgets"], "failures": sel["failures"],
                      "selected": {**{k: K[k] for k in ("resource", "size", "bops", "objective", "search_budget", "origin")},
                                   "config_hash": config_hash({n: K["formats"][n] for n in names})},
                      "selected_formats": {n: K["formats"][n] for n in names},
                      "frontier": [{k: p[k] for k in ("resource", "size", "bops", "objective", "search_budget", "origin",
                                                      "solver_status")} for p in sel["frontier"]],
                      "knee_scores": sel["knee_scores"],
                      "candidates": [{**{k: p[k] for k in ("resource", "size", "bops", "objective", "search_budget",
                                                           "origin", "solver_status")},
                                      "config_hash": config_hash({n: p["formats"][n] for n in names})} for p in sel["candidates"]],
                      "check_vs_mp_auto": check})
            log(f"{key} {solver_name} {rule}: knee R {K['resource']:.6f} BC {K['objective']:.6g} budget "
                f"{K['search_budget']}, cost {cost:.1f}s (new {new_s:.1f}s), check {check}")


def stage_b_offline(args, out, log):
    from mp_auto import _frontier, _knee
    pbs = jload(out / "plan_budgets.json")
    res = {}
    rng0 = np.random.default_rng(0)
    for key in PROBLEMS:
        d = out / "B" / key
        if not d.exists():
            continue
        delta = pbs[key]["group_delta"]
        for f in sorted(d.glob("search_*.json")):
            s = jload(f)
            C = [dict(c) for c in s["candidates"]]

            def decide(cands, rule):
                fr = _frontier(cands)
                k, _ = _knee(fr)
                i = fr.index(k)
                lft, rgt = interval(fr, i, rule, delta)
                g = next(j for j, gg in enumerate(groups_of(fr, delta)) if i in gg)
                return (k["config_hash"], round(k["resource"], 12)), (lft["resource"], rgt["resource"]), \
                    [fr[j]["config_hash"] for j in groups_of(fr, delta)[g]]
            base = {r: decide(C, r) for r in ("orig", "grouped")}
            row = {"order_invariant": True, "duplicate_invariant": True, "perturbation": {}}
            rng = np.random.default_rng(0)
            for _ in range(100):
                perm = [C[j] for j in rng.permutation(len(C))]
                if any(decide(perm, r)[:2] != base[r][:2] for r in base):
                    row["order_invariant"] = False
            if any(decide(C + [dict(c) for c in C], r)[:2] != base[r][:2] for r in base):
                row["duplicate_invariant"] = False
            for what in ("R", "BC", "both"):
                for eps in (1e-8, 1e-6, 1e-4):
                    rngp = np.random.default_rng(0)
                    cnt = {"knee_point": 0, "knee_group": 0, "orig_interval": 0, "grouped_interval": 0}
                    for _ in range(200):
                        P = []
                        for c in C:
                            c2 = dict(c)
                            if what in ("R", "both"):
                                c2["resource"] = c["resource"] * (1 + eps * rngp.uniform(-1, 1))
                            if what in ("BC", "both"):
                                c2["objective"] = c["objective"] * (1 + eps * rngp.uniform(-1, 1))
                            P.append(c2)
                        o, g = decide(P, "orig"), decide(P, "grouped")
                        cnt["knee_point"] += o[0][0] != base["orig"][0][0]
                        cnt["knee_group"] += base["grouped"][0][0] not in g[2]
                        cnt["orig_interval"] += any(abs(a / b - 1) > 1e-6 for a, b in zip(o[1], base["orig"][1]))
                        cnt["grouped_interval"] += any(abs(a / b - 1) > 1e-6 for a, b in zip(g[1], base["grouped"][1]))
                    row["perturbation"][f"{what}_{eps:g}"] = {k: v / 200 for k, v in cnt.items()}
            res[f"{key}/{f.stem}"] = row
            log(f"{key} {f.stem}: order {row['order_invariant']}, dup {row['duplicate_invariant']}, "
                f"pert 1e-6 both {row['perturbation']['both_1e-06']}")
    jdump(out / "B" / "offline.json", res)


def stage_b_analyze(args, out, log):
    from mp_auto import _frontier, _knee
    pbs = jload(out / "plan_budgets.json")
    res = {}
    for key in PROBLEMS:
        d = out / "B" / key
        if not d.exists():
            continue
        runs = {f.stem.replace("search_", ""): jload(f) for f in sorted(d.glob("search_*.json"))}
        ref = runs.get("milp_orig")
        row = {"runs": {}}
        for k, s in runs.items():
            fm = s["selected_formats"]
            sc = s["knee_scores"]
            fr = s["frontier"]
            ib = max(range(len(fr)), key=lambda j: sc[j])
            runner = max((j for j in range(len(fr)) if abs(fr[j]["resource"] / fr[ib]["resource"] - 1) > 0.005),
                         key=lambda j: sc[j], default=None)
            lowest = min(pbs[key]["coarse_budgets"])
            row["runs"][k] = {"knee": s["selected"], "search_cost_s": s["search_cost_s"], "new_solve_s": s["new_solve_s"],
                              "solves": len(s["used"]), "paths": {p: sum(u["path"] == p for u in s["used"]) for p in
                                                                  sorted({u["path"] for u in s["used"] if u["path"]})},
                              "failures": s["failures"],
                              "differing_vs_milp_orig": (sum(fm[n] != ref["selected_formats"][n] for n in fm) if ref else None),
                              "score_margin_R_diff_0.5pct": (sc[ib] - sc[runner]) if runner is not None else None,
                              "lowest_budget_point_present": any(math.isclose(c["search_budget"] or -1, lowest, rel_tol=1e-12)
                                                                 for c in s["candidates"]),
                              "rounds": s["trace"], "attempted_budgets": s["attempted_budgets"]}
        allc = [c for s in runs.values() for c in s["candidates"]]
        fu = _frontier(allc)
        ku, su = _knee(fu)
        row["union"] = {"points": len(fu), "knee": {k: ku[k] for k in ("resource", "objective", "search_budget", "config_hash")},
                        "top": sorted([{"resource": p["resource"], "objective": p["objective"], "score": s_, "hash": p["config_hash"]}
                                       for p, s_ in zip(fu, su)], key=lambda x: -x["score"])[:5]}
        res[key] = row
        log(f"{key}: " + "; ".join(f"{k} R {v['knee']['resource']:.5f} BC {v['knee']['objective']:.5g} diff {v['differing_vs_milp_orig']}"
                                   f" cost {v['search_cost_s']:.1f}s" for k, v in row["runs"].items()))
    jdump(out / "B" / "analysis.json", res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--problem", default=None)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--solvers", default="milp,lpfirst,lp")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tag = "_".join(x for x in (args.stage, args.problem, args.method, f"rep{args.rep}" if args.stage == "a_solve" else None) if x)
    (out / "logs").mkdir(exist_ok=True)
    logf = open(out / "logs" / f"{tag}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    status = "ok"
    try:
        globals()[f"stage_{args.stage}"](args, out, log)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        tf = out / "timing.json"
        timing = jload(tf) if tf.exists() else {"runs": []}
        timing["runs"].append({"tag": tag, "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "script_sha256": sha256(HERE / "run_lp_pipeline_regression.py")})
        jdump(tf, timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
