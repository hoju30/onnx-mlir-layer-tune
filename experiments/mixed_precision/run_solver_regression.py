#!/usr/bin/env python3
"""Solver regression before changing the formal searcher (mp_milp.milp_select inside mp_auto.auto_select).

BC tables, option sets, the resource model, auto_select's budgets and the knee rule stay as they are; only the per-budget
solver that auto_select calls is swapped, in this process (mp_milp.py / mp_auto.py are not edited by this script):
  milp_retry   mp_milp.milp_select as it is: MILP at the exact budget; if the answer exceeds the exact check, the whole
               MILP again with the budget shrunk by 1 ppm; if that fails too, the budget is dropped
  milp_repair  one MILP at the exact budget; an overshoot is removed by option swaps (cheapest cost increase per unit of
               excess); certified lower bound = HiGHS's dual bound of that solve
  lp_repair    LP relaxation; a fractional node takes its least-resource option in the LP support; the same repair;
               certified lower bound = the LP optimum
  lp_first     lp_repair, accepted if objective - LP bound <= max(1e-3 objective, 1e-6 objective scale), else milp_repair
               (objective scale = max |cost| / 1e4, milp_select's; 1e-6 of it = HiGHS's default absolute MIP gap)

Problems: the five CNN / GPT-2 / BERT native tables, Qwen3, Qwen3.5 FFN-only and FFN + attention (all unpruned, as their
pipelines ran them, 10 s per solve) and OLMoE fixed-route (3051 projections, exact dominance pruning, 300 s per solve).

Stages (one --out-dir):
  plan      pre-registration (problems, variants, pass criteria, OLMoE evaluation windows), before any result
  regress   auto_select with every variant on every problem; feasibility, certified gaps, frontiers, knees; freezes the
            OLMoE knee configurations
  capture   (GPU) FP32 routes of the new OLMoE evaluation windows (run_olmoe_multipoint's code)
  truth     (GPU) fixed-route whole-configuration replay of FP32 and the frozen knees (run_olmoe_multipoint.replay)
  cpu       the pre-registered CPU subset of the knees
  analyze   pass / fail against plan.json
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLMOE = HERE / "runs/olmoe_multipoint_20261015"
SPEED = HERE / "runs/olmoe_milp_speed_20261016"
STAGE1 = HERE / "runs/olmoe_fixed_route_20261015"
SEQ = 128
EVAL = {"V": ("validation", 163840, 32), "T": ("test", 163840, 32)}
CPU_SUBSET = list(range(0, 8)) + list(range(32, 40))
REL_TOL, ABS_TOL = 1e-3, 1e-6           # lp_first acceptance (ABS_TOL in units of the objective scale)
P1_REL = 1e-5                          # milp_repair: objective within 1e-5 of its dual bound; knee within 1e-5
P2_KNEE = 1e-3                         # lp_first: knee BC and R within 1e-3 of the reference knee
EQUIV = 0.002                          # nat / token: equivalence margin of the paired OLMoE knee test
VARIANTS = ("milp_retry", "milp_repair", "lp_repair", "lp_first")
FFN = ("gate_proj", "up_proj", "down_proj")
IMAGES = HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images"
DENSE = {
    "mobilenetv2": ("cnn", "runs/native_auto_2d_20261011/mobilenetv2/table.json",
                    "runs/native_auto_2d_20261011/mobilenetv2/selection.json"),
    "resnet18": ("cnn", "runs/native_auto_2d_20261011/resnet18/table.json",
                 "runs/native_auto_2d_20261011/resnet18/selection.json"),
    "efficientnet_b0": ("cnn", "runs/native_auto_2d_20261011/efficientnet_b0/table.json",
                        "runs/native_auto_2d_20261011/efficientnet_b0/selection.json"),
    "gpt2": ("gpt2", "runs/native_eps_fix_20261014/gpt2/table.json", "runs/native_auto_2d_20261011/gpt2/selection.json"),
    "bert": ("bert", "runs/native_eps_fix_20261014/bert/table.json", "runs/native_auto_2d_20261011/bert/selection.json"),
    "qwen3": ("qwen3", "runs/native_eps_fix_20261014/qwen3/table.json",
              "runs/qwen3_branch_ablation_20261014/selection_total.json"),
    "qwen35_ffn": ("qwen35_ffn", "runs/qwen35_bc_20261015/table.json", "runs/qwen35_bc_20261015/selection_ffn.json"),
    "qwen35_ffn_attn": ("qwen35_all", "runs/qwen35_bc_20261015/table.json",
                        "runs/qwen35_bc_20261015/selection_ffn_attn.json"),
}
DENSE_TL, OLMOE_TL = 10.0, 300.0


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(v):
    return float(v) if v is not None and math.isfinite(v) else None


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    import olmoe_common as oc
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    tok = oc.tokenizer()
    s1 = jload(STAGE1 / "plan.json")["data"]
    mp = jload(OLMOE / "plan.json")["eval_windows"]["segments"]
    toks, segs = [], {}
    for k, (split, first, n) in EVAL.items():
        t, spans, ntok, _ = oc.windows(split, first, n, SEQ, tok)
        toks.append(t)
        used = [s1[name]["char_spans"] for name in ("calibration", "diagnostic", "check") if s1[name]["split"] == split]
        used += [[seg["char_spans"]] for seg in mp.values() if seg["split"] == split]
        for sp in used:
            if spans[0][0] < sp[-1][1] and spans[-1][1] > sp[0][0]:
                raise SystemExit(f"segment {k} overlaps an earlier window set")
        segs[k] = {"split": split, "tokens": [first, first + n * SEQ], "windows": n,
                   "char_spans": [spans[0][0], spans[-1][1]], "split_tokens": ntok}
    np.save(out / "tokens_eval.npy", np.concatenate(toks))
    jdump(out / "plan.json", {
        "question": "can mp_milp's 1 ppm re-solve be replaced by repair + certificate, and can LP + rounding + repair "
                    "(with MILP fallback) replace the MILP, without changing feasibility, the certified optimality gap "
                    "or the selected knee; is the OLMoE LP knee's measured loss equivalent to the MILP knee's?",
        "already_seen": "runs/olmoe_milp_speed_20261016: on OLMoE, milp_retry's per-budget behaviour, milp_repair at 8 "
                        "budgets, lp_repair at the 22 recorded budgets and lp_repair's auto_select (knee R 0.7071, 7 of "
                        "3051 nodes differ from the MILP point at that budget). New here: every dense problem, lp_first, "
                        "the loss of the LP knee on new windows.",
        "variants": {"milp_retry": "mp_milp.milp_select unchanged (reference of the current pipeline)",
                     "milp_repair": "one MILP at the exact budget, overshoot repaired by option swaps, lower bound = "
                                    "HiGHS dual bound",
                     "lp_repair": "LP relaxation, fractional nodes to their least-resource option in the LP support, "
                                  "repair, lower bound = LP optimum (reference only, not proposed)",
                     "lp_first": f"lp_repair accepted if objective - LP bound <= max({REL_TOL:g} objective, {ABS_TOL:g} "
                                 "x objective scale), objective scale = max |cost| / 1e4; otherwise milp_repair"},
        "problems": {**{k: {"table": v[1], "sha256": sha256(HERE / v[1]), "options": "all finite (unpruned)",
                            "time_limit_s": DENSE_TL, "recorded_selection": v[2]} for k, v in DENSE.items()},
                     "olmoe": {"table": str((OLMOE / "table.json").relative_to(HERE)),
                               "sha256": sha256(OLMOE / "table.json"),
                               "options": "exact dominance pruning (run_olmoe_multipoint.nondominated)",
                               "time_limit_s": OLMOE_TL,
                               "variants_run": ["lp_repair", "lp_first"],
                               "milp_retry": "the recorded run (selection_raw.json, 3462 s; R 0.5 failed), not re-run",
                               "milp_repair": "not run here (about 35 min); 8 budgets in the speed study",
                               "reference_knee": "the recorded MILP candidates plus the speed study's repaired R = 0.5 "
                                                 "point, knee by mp_auto._knee (the recorded refinements were placed "
                                                 "around the wrong knee R 0.569; the corrected reference has no "
                                                 "refinement around R 0.707)"}},
        "search": "mp_auto.auto_select(samples=17, refinements=2), GRID11 x GRID11, value = B + C (non-finite forbidden)",
        "criteria": {
            "P1_milp_repair (dense problems)": [
                "a: every solve returns a configuration within the exact budget; no failed budget",
                f"b: every status-0 solve: objective - dual bound <= max({P1_REL:g} objective, {ABS_TOL:g} x objective "
                "scale); no more time-limited solves than milp_retry",
                f"c: knee identical to milp_retry's, or BC and R both within {P1_REL:g} relative"],
            "P2_lp_first (dense problems and OLMoE)": [
                "a: every solve within the exact budget; no failed budget",
                "b: every accepted LP point within its acceptance tolerance; where the reference solved the same budget "
                "(milp_repair on dense problems, the recorded MILP on OLMoE), objective - reference objective <= the "
                "same tolerance",
                f"c: knee identical to the reference knee (dense: milp_repair; OLMoE: corrected recorded MILP), or BC "
                f"and R both within {P2_KNEE:g} relative",
                f"d: OLMoE: per-token dNLL(LP knee) - dNLL(MILP knee) on the new windows, 95% bootstrap CI inside "
                f"[-{EQUIV}, +{EQUIV}] (CI outside: fail; CI crossing a bound: inconclusive = not passed)"],
            "lp_repair": "reported only",
            "speed": "seconds per auto_select reported; not a criterion"},
        "integration": {"if P1 passes": "mp_milp.milp_select: the 1 ppm re-solve replaced by repair; info gains the "
                                        "lower bound, certified gap and repair steps (default behaviour changes only "
                                        "where the old code re-solved or failed)",
                        "if P2 passes": "milp_select / auto_select gain solver='lp_first' (opt-in); the default stays "
                                        "the exact MILP",
                        "then": "the integrated code re-run on qwen35_ffn and OLMoE must reproduce this harness's "
                                "configurations"},
        "olmoe_eval": {"windows": {"segments": segs, "seq": SEQ, "count": int(sum(len(t) for t in toks)),
                                   "sha256": sha256(out / "tokens_eval.npy"),
                                   "disjoint": "character spans disjoint from stage 1's calibration / diagnostic / check "
                                               "windows and the multipoint evaluation windows"},
                       "configs": "fp32; milp_knee (corrected recorded MILP knee); lp_knee (lp_first's knee); "
                                  "lp_repair's knee too if it differs from lp_first's",
                       "truth": "per window summed next-token CE, fixed FP32 route (run_olmoe_multipoint.replay), GPU "
                                "FP32, TF32 off; ranges.json of the multipoint run (hash-checked copy)",
                       "bootstrap": "2000 resamples stratified by segment, seed 0",
                       "cpu_subset": CPU_SUBSET},
        "unchanged": "BC, tables, calibration, ranges, posit LUTs, QDQ, candidate formats, resource model, auto_select "
                     "budgets and knee rule, mp_milp / mp_auto defaults, earlier outputs",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan: eval {segs}")


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def arrays(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy):
    """milp_select's option arrays (weight format outer, activation format inner, None forbids)."""
    from htq_pareto_search import FORMAT_BITS
    weighted = [n["name"] for n in nodes if n.get("weight_const")]
    opts, cost, size, bops = [], [], [], []
    for name in weighted:
        row = []
        for w in weight_formats:
            for a in act_formats:
                v = value(name, w, a)
                if v is not None:
                    row.append((w, a))
                    cost.append(v)
                    size.append(weight_elems[name] * FORMAT_BITS[w] / 8)
                    bops.append(flops_proxy[name] * FORMAT_BITS[w] * FORMAT_BITS[a])
        if not row:
            raise ValueError(f"{name}: every option is forbidden")
        opts.append(row)
    off = np.cumsum([0] + [len(r) for r in opts])
    return weighted, opts, off, np.array(cost, float), np.array(size, float), np.array(bops, float)


def constraint(off, size, bops, mb, mo):
    from scipy.optimize import LinearConstraint
    from scipy.sparse import coo_matrix
    N, n = len(off) - 1, len(size)
    grp = np.repeat(np.arange(N), np.diff(off))
    A = coo_matrix((np.concatenate([np.ones(n), size / mb, bops / mo]),
                    (np.concatenate([grp, np.full(n, N), np.full(n, N + 1)]), np.concatenate([np.arange(n)] * 3))),
                   shape=(N + 2, n)).tocsr()
    return LinearConstraint(A, np.r_[np.ones(N), -np.inf, -np.inf], np.r_[np.ones(N), 1.0, 1.0])


def feasible(pick, size, bops, mb, mo):
    return bool(size[pick].sum() <= mb and bops[pick].sum() <= mo)


def repair(pick, cost, size, bops, off, mb, mo, max_steps=1000):
    """run_olmoe_milp_speed.repair: swap options until the exact budget holds, each step the swap with the smallest
    cost increase per unit of excess removed (only swaps that lower some over-budget resource and keep the other
    within budget)."""
    pick = pick.copy()
    steps = 0
    while True:
        s, b = size[pick].sum(), bops[pick].sum()
        exs, exb = max(0.0, s - mb), max(0.0, b - mo)
        if exs == 0 and exb == 0:
            return pick, steps
        if steps >= max_steps:
            return None, steps
        best = None
        for k in range(len(off) - 1):
            p = pick[k]
            cand = np.arange(off[k], off[k + 1])
            ds, db, dc = size[cand] - size[p], bops[cand] - bops[p], cost[cand] - cost[p]
            gain = np.minimum(exs, np.maximum(0, -ds)) / max(exs, 1e-300) * (exs > 0) + \
                np.minimum(exb, np.maximum(0, -db)) / max(exb, 1e-300) * (exb > 0)
            ok = (gain > 0) & (s + ds <= max(mb, s)) & (b + db <= max(mo, b))
            if not ok.any():
                continue
            score = np.where(ok, np.maximum(dc, 0) / np.maximum(gain, 1e-300), np.inf)
            j = int(np.argmin(score))
            if best is None or score[j] < best[0]:
                best = (score[j], k, cand[j])
        if best is None:
            return None, steps
        pick[best[1]] = best[2]
        steps += 1


def make_solver(variant, records):
    """A drop-in for mp_milp.milp_select (same arguments and return) that solves with `variant` and appends one
    record per call to `records`."""
    import mp_milp
    from scipy.optimize import Bounds, milp

    def select(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy, max_bytes, max_bops,
               time_limit=600, info=None):
        t0 = time.perf_counter()
        weighted, opts, off, cost, size, bops = arrays(nodes, weight_formats, act_formats, value, weight_elems,
                                                       flops_proxy)
        ref_bytes = sum(weight_elems[n] for n in weighted)
        scale = float(np.abs(cost).max()) / 1e4 or 1.0
        rec = {"variant": variant, "budget": max_bytes / ref_bytes, "scale": scale}

        def fail(msg):
            rec.update(failure=msg, seconds=time.perf_counter() - t0)
            records.append(rec)
            raise RuntimeError(msg)

        if variant == "milp_retry":
            inf = {}
            try:
                cfg, _, status = mp_milp.milp_select(nodes, weight_formats, act_formats, value, weight_elems,
                                                     flops_proxy, max_bytes, max_bops, time_limit=time_limit, info=inf)
            except RuntimeError as exc:
                fail(str(exc))
            idx = {(k, o): off[k] + j for k in range(len(weighted)) for j, o in enumerate(opts[k])}
            pick = np.array([idx[k, (cfg[n]["w"], cfg[n]["a"])] for k, n in enumerate(weighted)])
            db = inf.get("mip_dual_bound")
            lb = db * inf["objective_scale"] if db is not None else None
            path, status = "milp_retry", int(status)
            rec.update(margin_used=bool(inf["budget_margin_used"]),
                       lower_bound_note="dual bound of the final (possibly 1 ppm shrunk) solve")
        else:
            cons = constraint(off, size, bops, max_bytes, max_bops)
            pick = None
            if variant in ("lp_repair", "lp_first"):
                t1 = time.perf_counter()
                res = milp(cost / scale, integrality=np.zeros(len(cost)), bounds=Bounds(0, 1), constraints=cons)
                rec["lp_s"] = time.perf_counter() - t1
                if res.x is not None:
                    p0, frac = [], 0
                    for k in range(len(weighted)):
                        seg = res.x[off[k]:off[k + 1]]
                        nz = np.nonzero(seg > 1e-9)[0]
                        if len(nz) > 1:
                            frac += 1
                            j = nz[np.lexsort((bops[off[k] + nz], size[off[k] + nz]))[0]]
                        else:
                            j = int(np.argmax(seg))
                        p0.append(off[k] + j)
                    p1, steps = repair(np.array(p0), cost, size, bops, off, max_bytes, max_bops)
                    lpb = float(res.fun) * scale
                    rec.update(lp_bound=lpb, fractional_nodes=frac, lp_repair_steps=steps)
                    if p1 is not None:
                        o = float(cost[p1].sum())
                        tol = max(REL_TOL * o, ABS_TOL * scale)
                        rec.update(lp_objective=o, lp_gap=o - lpb, lp_tol=tol, lp_accepted=bool(o - lpb <= tol))
                        if variant == "lp_repair" or o - lpb <= tol:
                            pick, lb, status, path = p1, lpb, 1, "lp"
                else:
                    rec["lp_message"] = str(res.message)
                if pick is None and variant == "lp_repair":
                    fail(f"LP route found no solution: {res.message}")
            if pick is None:
                t1 = time.perf_counter()
                res = milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1), constraints=cons,
                           options={"mip_rel_gap": 0.0, "time_limit": time_limit})
                rec.update(milp_s=time.perf_counter() - t1, milp_status=int(res.status), milp_message=str(res.message))
                if res.x is None:
                    fail(f"MILP found no solution: {res.message}")
                p0 = np.array([off[k] + int(np.argmax(res.x[off[k]:off[k + 1]])) for k in range(len(weighted))])
                rec["milp_first_feasible"] = feasible(p0, size, bops, max_bytes, max_bops)
                p1, steps = repair(p0, cost, size, bops, off, max_bytes, max_bops)
                rec["milp_repair_steps"] = steps
                if p1 is None:
                    fail("MILP answer could not be repaired into the budget")
                db = num(res.get("mip_dual_bound"))
                pick, lb, status = p1, (db * scale if db is not None else None), int(res.status)
                path = "milp" if variant == "milp_repair" else "lp->milp"
        obj = float(cost[pick].sum())
        rec.update(path=path, status=status, objective=obj, lower_bound=lb,
                   gap=(obj - lb) if lb is not None else None,
                   gap_rel=((obj - lb) / obj if obj > 0 else 0.0) if lb is not None else None,
                   M_over=float(size[pick].sum() / max_bytes - 1), B_over=float(bops[pick].sum() / max_bops - 1),
                   feasible=feasible(pick, size, bops, max_bytes, max_bops), seconds=time.perf_counter() - t0)
        records.append(rec)
        if info is not None:
            info.update({k: rec.get(k) for k in ("path", "status", "objective", "lower_bound", "gap", "seconds")})
        formats = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
        for k, name in enumerate(weighted):
            w, a = opts[k][pick[k] - off[k]]
            formats[name] = {"w": w, "a": a}
        return formats, obj, status
    return select


# --------------------------------------------------------------------------- #
# problems
# --------------------------------------------------------------------------- #
def dense_problem(key, work):
    import run_native_bc_pipeline as nb
    kind, table_path, _ = DENSE[key]
    if kind == "cnn":
        spec = nb.cnn_spec(key, IMAGES, 1, work, with_references=False)
    elif kind == "gpt2":
        spec = nb.gpt2_spec(work, with_references=False)
    elif kind == "bert":
        spec = nb.bert_spec(work, with_references=False)
    elif kind == "qwen3":
        spec = nb.qwen3_spec(work)
    else:
        spec = nb.qwen35_spec(work, select=(lambda info: info["kind"] in FFN) if kind == "qwen35_ffn" else None)
    table = jload(HERE / table_path)
    nodes = [n for n in spec.nodes if n["name"] in spec.names]
    missing = [n for n in spec.names if n not in table["B"]]
    if missing:
        raise SystemExit(f"{key}: table lacks {missing[:3]}")
    return nodes, spec.names, (lambda n, w, a: table["B"][n][w] + table["C"][n][a]), spec.weight_elems, spec.flops, \
        DENSE_TL


def olmoe_problem():
    import olmoe_common  # noqa: F401  (transformers 5; the dense specs import the venv's transformers, see regress)
    from run_olmoe_multipoint import nondominated, problem, pruned
    table, _, names, nodes, elems, _, flops = problem(OLMOE)
    bc = lambda n, w, a: table["B"][n][w] + table["C"][n][a]
    return nodes, names, pruned(bc, nondominated(names, bc)), elems, flops, OLMOE_TL


def run_variant(nodes, value, elems, flops, tl, variant):
    import mp_auto
    from run_native_bc_pipeline import GRID11
    recs, slog = [], []
    orig = mp_auto.milp_select
    mp_auto.milp_select = make_solver(variant, recs)
    try:
        t0 = time.perf_counter()
        sel = mp_auto.auto_select(nodes, GRID11, GRID11, value, elems, flops, samples=17, refinements=2,
                                  time_limit=tl, solver_log=slog)
        sec = time.perf_counter() - t0
    finally:
        mp_auto.milp_select = orig
    return sel, recs, sec


def point(row, names):
    from mp_auto import config_hash
    return {"config_hash": config_hash({n: row["formats"][n] for n in names}) if row.get("formats") else None,
            **{k: row.get(k) for k in ("resource", "size", "bops", "objective", "search_budget", "origin",
                                       "solver_status")}}


def knee_compare(a, b, names):
    """a against the reference b (selected rows with formats)."""
    diff = sum(a["formats"][n] != b["formats"][n] for n in names)
    return {"identical": diff == 0, "nodes_differing": int(diff),
            "bc_rel": (a["objective"] - b["objective"]) / b["objective"] if b["objective"] else None,
            "R_rel": (a["resource"] - b["resource"]) / b["resource"],
            "budget": a["search_budget"], "reference_budget": b["search_budget"]}


def same_budget(budget, rows):
    return next((r for r in rows if r.get("search_budget") is not None
                 and math.isclose(r["search_budget"], budget, rel_tol=1e-12)), None)


def recorded_check(sel, rec_path, value, names):
    """Whether the recorded selection was made on this table (every recorded MILP candidate's objective recomputed),
    and if so, whether this run's knee and frontier reproduce it."""
    if rec_path is None or not (HERE / rec_path).exists():
        return {"available": False}
    rec = jload(HERE / rec_path)
    if "selected" not in rec or "candidates" not in rec:
        return {"available": False, "reason": "no auto_select output"}
    mil = [c for c in rec["candidates"] if c.get("origin") == "milp"]
    dev = max(abs(sum(value(n, c["formats"][n]["w"], c["formats"][n]["a"]) for n in names) - c["objective"])
              / max(abs(c["objective"]), 1e-300) for c in mil)
    if dev > 1e-9:
        return {"available": True, "same_table": False, "max_objective_rel_dev": dev}
    cmp = knee_compare(sel["selected"], rec["selected"], names)
    return {"available": True, "same_table": True, "max_objective_rel_dev": dev, "knee_vs_recorded": cmp,
            "frontier_points": [len(sel["frontier"]), len(rec["frontier"])]}


def stage_regress(args, out, log):
    from mp_auto import _frontier, _knee
    if not (out / "plan.json").exists():
        raise SystemExit("plan first")
    f = out / "regress.json"
    res = jload(f) if f.exists() else {}
    work = out / "spec_work"
    work.mkdir(exist_ok=True)
    keys = args.problems.split(",") if args.problems else list(DENSE)
    if "olmoe" in keys and len(keys) > 1:
        raise SystemExit("run olmoe in its own process: the dense specs import another transformers")
    for key in keys:
        if key in res:
            continue
        t0 = time.perf_counter()
        if key == "olmoe":
            nodes, names, value, elems, flops, tl = olmoe_problem()
            variants = ("lp_repair", "lp_first")
        else:
            nodes, names, value, elems, flops, tl = dense_problem(key, work)
            variants = VARIANTS
        log(f"{key}: {len(names)} nodes, problem built in {time.perf_counter() - t0:.1f}s")
        row = {"nodes": len(names), "time_limit_s": tl, "variants": {}}
        sels = {}
        for v in variants:
            sel, recs, sec = run_variant(nodes, value, elems, flops, tl, v)
            sels[v] = sel
            ok = [r for r in recs if "failure" not in r]
            row["variants"][v] = {
                "seconds": sec, "solves": len(recs), "failures": sel["failures"],
                "all_feasible": all(r["feasible"] for r in ok), "paths": dict(Counter(r["path"] for r in ok)),
                "status_counts": dict(Counter(str(r["status"]) for r in ok)),
                "max_gap_rel": max((r["gap_rel"] for r in ok if r["gap_rel"] is not None), default=None),
                "knee": point(sel["selected"], names),
                "frontier": [point(p, names) for p in sel["frontier"]],
                "attempted_budgets": sel["attempted_budgets"], "solves_detail": recs}
            k = sel["selected"]
            log(f"  {v}: {sec:.1f}s, {len(recs)} solves, paths {row['variants'][v]['paths']}, failures "
                f"{len(sel['failures'])}, feasible {row['variants'][v]['all_feasible']}, max gap "
                f"{row['variants'][v]['max_gap_rel']}, knee R {k['resource']:.4f} BC {k['objective']:.6g}")
        if key == "olmoe":
            raw = jload(OLMOE / "selection_raw.json")
            rep = jload(SPEED / "repair.json")["0.500000"]
            pj = jload(SPEED / "problem.json")
            r05 = {"formats": None, "objective": rep["repaired_objective"], "search_budget": 0.5, "solver_status": 0,
                   "origin": "milp_repaired_speed_study",
                   "resource": max(rep["repaired_M"] / pj["ref_bytes"], rep["repaired_B"] / pj["ref_bops"])}
            mil = [c for c in raw["candidates"] if c["origin"] == "milp"]
            ref, _ = _knee(_frontier(raw["candidates"] + [r05]))
            p125 = jload(OLMOE / "configs/bc_p125.json")
            row["reference"] = {"what": "recorded MILP candidates + repaired R 0.5 point", "knee": point(ref, names),
                                "recorded_knee": point(raw["selected"], names),
                                "knee_is_bc_p125": all(ref["formats"][n] == p125[n] for n in names),
                                "recorded_seconds": raw["seconds_"]}
            for v in variants:
                sel = sels[v]
                row["variants"][v]["knee_vs_reference"] = knee_compare(sel["selected"], ref, names)
                vs = []
                for r in row["variants"][v]["solves_detail"]:
                    m = same_budget(r["budget"], mil)
                    if m is not None and "failure" not in r:
                        vs.append({"budget": r["budget"], "objective": r["objective"], "reference": m["objective"],
                                   "diff_rel": (r["objective"] - m["objective"]) / m["objective"] if m["objective"] else None,
                                   "within_tol": r["objective"] - m["objective"] <= max(REL_TOL * r["objective"],
                                                                                       ABS_TOL * r["scale"])})
                row["variants"][v]["vs_reference_same_budget"] = vs
            (out / "configs").mkdir(exist_ok=True)
            cfgs = {"milp_knee": ref, "lp_knee": sels["lp_first"]["selected"]}
            if knee_compare(sels["lp_repair"]["selected"], sels["lp_first"]["selected"], names)["nodes_differing"]:
                cfgs["lp_repair_knee"] = sels["lp_repair"]["selected"]
            for name, r in cfgs.items():
                jdump(out / "configs" / f"{name}.json", {n: r["formats"][n] for n in names})
            shutil.copyfile(OLMOE / "ranges.json", out / "ranges.json")
            if sha256(out / "ranges.json") != sha256(OLMOE / "ranges.json"):
                raise SystemExit("ranges copy differs")
            jdump(out / "freeze.json", {"configs_sha256": {n: sha256(out / "configs" / f"{n}.json") for n in cfgs},
                                        "ranges_sha256": sha256(out / "ranges.json"),
                                        "tokens_sha256": sha256(out / "tokens_eval.npy"),
                                        "plan_sha256": sha256(out / "plan.json"),
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            log(f"  reference knee R {ref['resource']:.4f} BC {ref['objective']:.6g} (= bc_p125: "
                f"{row['reference']['knee_is_bc_p125']}); frozen {list(cfgs)}")
        else:
            refsel = sels["milp_repair"]
            for v in variants:
                sel = sels[v]
                row["variants"][v]["knee_vs_milp_repair"] = knee_compare(sel["selected"], refsel["selected"], names)
                row["variants"][v]["knee_vs_milp_retry"] = knee_compare(sel["selected"], sels["milp_retry"]["selected"],
                                                                        names)
                vs = []
                for r in row["variants"][v]["solves_detail"]:
                    m = same_budget(r["budget"], [c for c in refsel["candidates"] if c["origin"] == "milp"])
                    if m is not None and "failure" not in r:
                        vs.append({"budget": r["budget"], "objective": r["objective"], "reference": m["objective"],
                                   "diff_rel": (r["objective"] - m["objective"]) / m["objective"] if m["objective"] else None,
                                   "within_tol": r["objective"] - m["objective"] <= max(REL_TOL * r["objective"],
                                                                                       ABS_TOL * r["scale"])})
                row["variants"][v]["vs_milp_repair_same_budget"] = vs
                log(f"  {v} knee vs milp_repair: {row['variants'][v]['knee_vs_milp_repair']}")
            row["recorded"] = recorded_check(sels["milp_retry"], DENSE[key][2], value, names)
            log(f"  recorded: {row['recorded']}")
        res = jload(f) if f.exists() else {}         # another process may have added problems meanwhile
        res[key] = row
        jdump(f, res)


# --------------------------------------------------------------------------- #
# OLMoE evaluation
# --------------------------------------------------------------------------- #
def check_freeze(out):
    fz = jload(out / "freeze.json")
    for k, h in fz["configs_sha256"].items():
        if sha256(out / "configs" / f"{k}.json") != h:
            raise SystemExit(f"{k} changed after the freeze")
    if sha256(out / "ranges.json") != fz["ranges_sha256"] or sha256(out / "tokens_eval.npy") != fz["tokens_sha256"]:
        raise SystemExit("ranges / tokens changed after the freeze")
    return list(fz["configs_sha256"])


def stage_capture(args, out, log):
    from run_olmoe_multipoint import stage_capture as capture
    capture(args, out, log)


def stage_truth(args, out, log):
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    from run_olmoe_multipoint import replay
    gpu_exact(args.device)
    names = check_freeze(out)
    if (out / "truth.json").exists():
        return
    rts, _, _ = load_routes(out, "eval", oc.config(), args.device)
    ce, sec = replay(out, ["fp32"] + names, np.load(out / "tokens_eval.npy"), rts, args.device, log, "truth")
    jdump(out / "truth.json", {"summed_ce": ce, "seconds": sec, "device": args.device})
    log(f"truth: {['fp32'] + names} in {sec:.0f}s")


def stage_cpu(args, out, log):
    import olmoe_common as oc
    import torch
    from run_olmoe_fixed_route import load_routes
    from run_olmoe_multipoint import replay
    names = check_freeze(out)
    if (out / "truth_cpu.json").exists():
        return
    rts, _, _ = load_routes(out, "eval", oc.config(), "cpu")
    rows = torch.cat([torch.arange(k * SEQ, (k + 1) * SEQ) for k in CPU_SUBSET])
    rts = [(i[rows], w[rows]) for i, w in rts]
    ce, sec = replay(out, names, np.load(out / "tokens_eval.npy")[CPU_SUBSET], rts, "cpu", log, "cpu")
    jdump(out / "truth_cpu.json", {"subset": CPU_SUBSET, "summed_ce": ce, "seconds": sec})
    log(f"cpu: {names} in {sec:.0f}s")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def stage_analyze(args, out, log):
    reg = jload(out / "regress.json")
    res = {"P1": {}, "P2": {}, "lp_repair": {}, "speed": {}, "olmoe_loss": None}

    def p1(key, row):
        v, r0 = row["variants"]["milp_repair"], row["variants"]["milp_retry"]
        det = [r for r in v["solves_detail"] if "failure" not in r]
        b = all(r["gap"] <= max(P1_REL * r["objective"], ABS_TOL * r["scale"]) for r in det if r["status"] == 0)
        tl_ok = v["status_counts"].get("1", 0) <= r0["status_counts"].get("1", 0)
        kc = v["knee_vs_milp_retry"]
        c = kc["identical"] or (abs(kc["bc_rel"] or 0) <= P1_REL and abs(kc["R_rel"]) <= P1_REL)
        a = v["all_feasible"] and not v["failures"]
        return {"a": a, "b": b and tl_ok, "c": c, "pass": a and b and tl_ok and c, "knee": kc,
                "repairs": sum(r.get("milp_repair_steps", 0) > 0 for r in det),
                "milp_retry_margin_solves": sum(bool(r.get("margin_used")) for r in r0["solves_detail"]),
                "milp_retry_failures": len(r0["failures"]),
                "max_gap_rel": v["max_gap_rel"]}

    def p2(key, row, v):
        x = row["variants"][v]
        det = [r for r in x["solves_detail"] if "failure" not in r]
        a = x["all_feasible"] and not x["failures"]
        acc = all(r["gap"] <= max(REL_TOL * r["objective"], ABS_TOL * r["scale"]) for r in det if r["path"] == "lp")
        vs = x.get("vs_milp_repair_same_budget", x.get("vs_reference_same_budget", []))
        b = acc and all(s["within_tol"] for s in vs)
        kc = x.get("knee_vs_milp_repair", x.get("knee_vs_reference"))
        c = kc["identical"] or (abs(kc["bc_rel"] or 0) <= P2_KNEE and abs(kc["R_rel"]) <= P2_KNEE)
        return {"a": a, "b": b, "b_accepted_within_tol": acc, "same_budget_compared": len(vs),
                "same_budget_max_diff_rel": max((s["diff_rel"] for s in vs if s["diff_rel"] is not None), default=None),
                "c": c, "knee": kc, "paths": x["paths"], "max_gap_rel_accepted": x["max_gap_rel"]}

    for key, row in reg.items():
        if key != "olmoe":
            res["P1"][key] = p1(key, row)
        res["P2"][key] = p2(key, row, "lp_first")
        res["lp_repair"][key] = p2(key, row, "lp_repair")
        res["speed"][key] = {v: x["seconds"] for v, x in row["variants"].items()}
        if key == "olmoe":
            res["speed"][key]["milp_retry_recorded"] = row["reference"]["recorded_seconds"]

    if (out / "truth.json").exists():
        tr = jload(out / "truth.json")["summed_ce"]
        T = SEQ - 1
        base = np.array(tr["fp32"])
        rng = np.random.default_rng(0)
        draws = np.concatenate([rng.integers(0, 32, (2000, 32)), 32 + rng.integers(0, 32, (2000, 32))], 1)

        def st(d):
            b = d[draws].mean(1)
            return {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    "V": float(d[:32].mean()), "T": float(d[32:].mean())}
        lo = {"fp32_nll": float(base.sum() / (len(base) * T)), "configs": {}, "paired": {}}
        for k in tr:
            if k != "fp32":
                lo["configs"][k] = {"dnll": st((np.array(tr[k]) - base) / T),
                                    "ppl": float(np.exp(np.array(tr[k]).sum() / (len(base) * T)))}
        for k in [k for k in tr if k.startswith("lp")]:
            d = st((np.array(tr[k]) - np.array(tr["milp_knee"])) / T)
            lo_, hi_ = d["ci95"]
            d["verdict"] = ("equivalent" if -EQUIV <= lo_ and hi_ <= EQUIV else
                            "different" if hi_ < -EQUIV or lo_ > EQUIV else "inconclusive")
            lo["paired"][f"{k}-milp_knee"] = d
        if (out / "truth_cpu.json").exists():
            cpu = jload(out / "truth_cpu.json")
            sub = np.array(cpu["subset"])
            c = cpu["summed_ce"]
            lo["cpu"] = {k: {"max_abs_window_diff_per_token": float(np.abs(np.array(c[k]) - np.array(tr[k])[sub]).max() / T)}
                         for k in c}
            for k in [k for k in c if k.startswith("lp")]:
                g = (np.array(tr[k])[sub] - np.array(tr["milp_knee"])[sub]) / T
                cc = (np.array(c[k]) - np.array(c["milp_knee"])) / T
                lo["cpu"][f"{k}-milp_knee"] = {"gpu_subset_mean": float(g.mean()), "cpu_subset_mean": float(cc.mean()),
                                               "max_abs_window_diff": float(np.abs(g - cc).max())}
        res["olmoe_loss"] = lo
        res["P2"]["olmoe"]["d"] = lo["paired"]["lp_knee-milp_knee"]["verdict"] == "equivalent"
    for key, r in res["P2"].items():
        r["pass"] = r["a"] and r["b"] and r["c"] and (r.get("d", False) if key == "olmoe" else True)
    res["P1_pass"] = all(r["pass"] for r in res["P1"].values())
    res["P2_pass"] = all(r["pass"] for r in res["P2"].values()) and "d" in res["P2"].get("olmoe", {})
    jdump(out / "analysis.json", res)
    for key in reg:
        log(f"{key}: P1 {res['P1'].get(key, {}).get('pass')} | P2 {res['P2'][key]['pass']} "
            f"(knee {res['P2'][key]['knee']}) | speed {res['speed'][key]}")
    if res["olmoe_loss"]:
        log(f"OLMoE loss: {json.dumps(res['olmoe_loss']['paired'])}")
    log(f"P1 pass {res['P1_pass']}, P2 pass {res['P2_pass']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "regress", "capture", "truth", "cpu", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--problems", default=None, help="regress: comma-separated subset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("capture", "truth", "cpu"):
        import torch
        torch.set_num_threads(args.threads)
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
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
        timing["runs"].append({"stage": args.stage, "problems": args.problems, "device": args.device, "status": status,
                               "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
