#!/usr/bin/env python3
"""Why is the MILP of runs/olmoe_multipoint_20261015 slow, and what is cheaper without changing the problem?

The problem is the BC selection of that run, rebuilt exactly (its table.json, the exact dominance pruning,
GRID11 x GRID11, 3051 expert projections, M <= R M8, B <= R B8), solved the way mp_milp.milp_select does
(same matrix, objective scale, mip_rel_gap 0; attempt 1 at the exact budget, attempt 2 with the budget shrunk by
1 ppm if attempt 1's answer exceeds the exact check). mp_milp / mp_auto and every earlier output stay untouched.

Stages (one --out-dir):
  problem  the arrays of the selection problem (npz) and the cost of building them as milp_select does
  diag     step 1: each attempt in its own process with HiGHS's log: presolve, root LP, first feasible
           solution, bound trajectory, total; whether attempt 2 was needed and what it cost
  repair   step 2: the exact-budget overshoot of attempt 1 repaired by the cheapest option swaps instead of
           re-solving; the R = 0.5 failure; cost against attempt 1's dual bound
  lp       step 3: LP relaxation + rounding + repair at every budget of the auto_select run, against the MILP
  lpauto   step 3: mp_auto.auto_select with the LP solver substituted (frontier, knee) against the MILP run's
  gap      step 4: MILP with mip_rel_gap > 0 at the diagnosed budgets (only if step 1 shows time after a good
           incumbent)
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "runs/olmoe_multipoint_20261015"
DIAG_BUDGETS = [0.5, 0.5221367684069359, 0.5693939156132402, 0.7071067811865476, 1.0, 1.681792830507429,
                2.8284271247461903, 4.0]


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# the problem
# --------------------------------------------------------------------------- #
def stage_problem(args, out, log):
    sys.argv = [sys.argv[0]]
    import olmoe_common  # noqa: F401
    from run_olmoe_multipoint import problem, nondominated
    from run_native_bc_pipeline import GRID11
    from htq_pareto_search import FORMAT_BITS
    t0 = time.perf_counter()
    table, meta, names, nodes, elems, rows, flops = problem(SRC)
    bc = lambda n, w, a: table["B"][n][w] + table["C"][n][a]
    t1 = time.perf_counter()
    keep = nondominated(names, bc)
    t2 = time.perf_counter()
    cost, size, bops, grp, lab = [], [], [], [], []
    for k, n in enumerate(names):            # milp_select's option order: weight format, then activation format
        for w in GRID11:
            for a in GRID11:
                if (w, a) in keep[n]:
                    cost.append(bc(n, w, a))
                    size.append(elems[n] * FORMAT_BITS[w] / 8)
                    bops.append(flops[n] * FORMAT_BITS[w] * FORMAT_BITS[a])
                    grp.append(k)
                    lab.append(f"{w}/{a}")
    t3 = time.perf_counter()
    np.savez(out / "problem.npz", cost=np.array(cost), size=np.array(size), bops=np.array(bops), grp=np.array(grp))
    ref_b = float(sum(elems[n] for n in names))
    ref_o = float(sum(64 * flops[n] for n in names))
    jdump(out / "problem.json", {"names": names, "labels": lab, "ref_bytes": ref_b, "ref_bops": ref_o,
                                 "options": len(cost), "options_unpruned": 121 * len(names), "nodes": len(names),
                                 "seconds": {"load_table": t1 - t0, "dominance_pruning": t2 - t1, "arrays": t3 - t2}})
    log(f"problem: {len(names)} nodes, {len(cost)} options; load {t1 - t0:.1f}s, pruning {t2 - t1:.1f}s, arrays {t3 - t2:.1f}s")


def load_problem(out):
    z = np.load(out / "problem.npz")
    meta = jload(out / "problem.json")
    grp = z["grp"]
    N = meta["nodes"]
    off = np.r_[0, np.cumsum(np.bincount(grp, minlength=N))]
    return z["cost"], z["size"], z["bops"], grp, off, N, meta


def matrix(size, bops, grp, N, R, meta, shrink):
    from scipy.sparse import coo_matrix
    mb, mo = R * meta["ref_bytes"], R * meta["ref_bops"]
    n = len(size)
    A = coo_matrix((np.r_[np.ones(n), size / mb, bops / mo],
                    (np.r_[grp, np.full(n, N), np.full(n, N + 1)], np.r_[np.arange(n), np.arange(n), np.arange(n)])),
                   shape=(N + 2, n)).tocsr()
    lo = np.r_[np.ones(N), -np.inf, -np.inf]
    hi = np.r_[np.ones(N), 1 - shrink, 1 - shrink]
    return A, lo, hi, mb, mo


def check(pick, size, bops, mb, mo):
    s, b = float(size[pick].sum()), float(bops[pick].sum())
    return {"M": s, "B": b, "M_over_budget": s / mb - 1, "B_over_budget": b / mo - 1, "feasible": bool(s <= mb and b <= mo)}


# --------------------------------------------------------------------------- #
# one solve in its own process (HiGHS's log on stdout)
# --------------------------------------------------------------------------- #
def worker(out, R, shrink, gap, time_limit, res_path, presolve=True):
    from scipy.optimize import Bounds, LinearConstraint, milp
    cost, size, bops, grp, off, N, meta = load_problem(out)
    t0 = time.perf_counter()
    A, lo, hi, mb, mo = matrix(size, bops, grp, N, R, meta, shrink)
    scale = float(np.abs(cost).max()) / 1e4 or 1.0
    t1 = time.perf_counter()
    print(f"#build {t1 - t0:.3f}", flush=True)
    res = milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1), constraints=LinearConstraint(A, lo, hi),
               options={"mip_rel_gap": gap, "time_limit": time_limit, "disp": True, "presolve": presolve})
    t2 = time.perf_counter()
    r = {"R": R, "shrink": shrink, "gap_setting": gap, "presolve": presolve, "status": int(res.status), "message": str(res.message),
         "build_s": t1 - t0, "solve_s": t2 - t1, "scale": scale}
    if res.x is not None:
        pick = np.array([off[k] + int(np.argmax(res.x[off[k]:off[k + 1]])) for k in range(N)])
        r.update(objective=float(cost[pick].sum()), mip_gap=res.get("mip_gap"), dual_bound=float(res.get("mip_dual_bound")) * scale
                 if res.get("mip_dual_bound") is not None else None, nodes=res.get("mip_node_count"),
                 **check(pick, size, bops, mb, mo))
        np.save(str(res_path) + ".pick.npy", pick)
    jdump(Path(res_path), r)


def run_attempt(out, R, shrink, gap, tl, tag, presolve=True):
    d = out / "logs"
    d.mkdir(exist_ok=True)
    logf, resf = d / f"{tag}.log", d / f"{tag}.json"
    t0 = time.perf_counter()
    with open(logf, "w") as fh:
        subprocess.run([sys.executable, __file__, "--worker", str(out), str(R), str(shrink), str(gap), str(tl), str(resf),
                        "1" if presolve else "0"],
                       stdout=fh, stderr=subprocess.STDOUT, check=False)
    r = jload(resf)
    r["wall_s"] = time.perf_counter() - t0
    r.update(parse_log(logf))
    jdump(resf, r)
    return r


def parse_log(path):
    """Presolve and root timings, the first incumbent, and the bound trajectory from HiGHS's MIP log."""
    traj, first, presolve_end = [], None, None
    txt = path.read_text()
    for line in txt.splitlines():
        t = line.split()
        if not t or not t[-1].endswith("s"):
            continue
        try:
            sec = float(t[-1][:-1])
        except ValueError:
            continue
        pct = [i for i, x in enumerate(t) if x.endswith("%")]
        if not pct or len(t) < pct[0] + 4:
            continue
        i = pct[0]
        src = t[0] if t[0].isalpha() else ""

        def num(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except ValueError:
                return None
        bb, bs, gp = num(t[i + 1]), num(t[i + 2]), t[i + 3]
        traj.append({"src": src, "time": sec, "best_bound": bb, "best_sol": bs, "gap": gp})
        if first is None and bs is not None:
            first = {"time": sec, "src": src, "best_sol": bs, "best_bound": bb}
    m = re.search(r"Presolve\w*:? .*?([\d.]+)s", txt)
    tot = re.search(r"Timing\s+([\d.]+)", txt)
    pres = re.search(r"Presolve.*?\n(?:.*\n)*?.*?Reductions", txt)
    red = re.search(r"Presolve reductions: rows (\d+)\(-?(\d+)\); columns (\d+)\(-?(\d+)\)", txt)
    return {"log_rows": len(traj), "first_incumbent": first, "trajectory": traj[:400],
            "highs_total_s": float(tot.group(1)) if tot else None,
            "presolve_reductions": red.groups() if red else None,
            "log_tail": txt.splitlines()[-25:]}


# --------------------------------------------------------------------------- #
# step 1
# --------------------------------------------------------------------------- #
def stage_diag(args, out, log):
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else DIAG_BUDGETS
    pre = not args.no_presolve
    f = out / ("diag.json" if pre else "diag_nopresolve.json")
    res = jload(f) if f.exists() else {}
    for R in budgets:
        key = f"{R:.6f}"
        if key in res:
            continue
        tag = "diag" if pre else "nopre"
        a1 = run_attempt(out, R, 0.0, 0.0, args.time_limit, f"{tag}_{key}_a1", pre)
        row = {"attempt1": a1}
        log(f"R {R:.4f} attempt 1: status {a1['status']} {a1['solve_s']:.0f}s, first incumbent "
            f"{a1['first_incumbent']}, feasible {a1.get('feasible')}, over M {a1.get('M_over_budget')}, B {a1.get('B_over_budget')}")
        if a1.get("objective") is None or not a1.get("feasible"):
            a2 = run_attempt(out, R, 1e-6, 0.0, args.time_limit, f"{tag}_{key}_a2", pre)
            row["attempt2"] = a2
            log(f"R {R:.4f} attempt 2 (1 ppm): status {a2['status']} {a2['solve_s']:.0f}s, feasible {a2.get('feasible')}")
        res[key] = row
        jdump(f, res)


# --------------------------------------------------------------------------- #
# step 2: repair instead of re-solve
# --------------------------------------------------------------------------- #
def repair(pick, cost, size, bops, grp, off, mb, mo, max_steps=1000):
    """Swap options until the exact budget holds, each step the swap with the smallest cost increase per unit of
    excess removed (only swaps that lower some over-budget resource and keep the other within budget)."""
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


def stage_repair(args, out, log):
    cost, size, bops, grp, off, N, meta = load_problem(out)
    tag = "nopre" if args.no_presolve else "diag"
    diag = jload(out / ("diag_nopresolve.json" if args.no_presolve else "diag.json"))
    res = {}
    for key, row in diag.items():
        a1 = row["attempt1"]
        R = a1["R"]                                   # the exact budget (the key is rounded)
        _, _, _, mb, mo = matrix(size, bops, grp, N, R, meta, 0.0)
        r = {"attempt1_feasible": a1.get("feasible"), "attempt1_s": a1["solve_s"],
             "attempt2_s": row.get("attempt2", {}).get("solve_s")}
        pf = out / "logs" / f"{tag}_{key}_a1.json.pick.npy"
        if a1.get("objective") is not None and not a1.get("feasible") and pf.exists():
            t0 = time.perf_counter()
            pk, steps = repair(np.load(pf), cost, size, bops, grp, off, mb, mo)
            r["repair_s"] = time.perf_counter() - t0
            r["repair_steps"] = steps
            if pk is not None:
                obj = float(cost[pk].sum())
                r.update(repaired_objective=obj, attempt1_objective=a1["objective"], dual_bound=a1.get("dual_bound"),
                         repaired_minus_dual_bound_rel=(obj - a1["dual_bound"]) / abs(a1["dual_bound"]) if a1.get("dual_bound") else None,
                         **{f"repaired_{k}": v for k, v in check(pk, size, bops, mb, mo).items()})
                if "attempt2" in row and row["attempt2"].get("objective") is not None:
                    r["attempt2_objective"] = row["attempt2"]["objective"]
                    r["repaired_minus_attempt2_rel"] = (obj - row["attempt2"]["objective"]) / abs(row["attempt2"]["objective"])
        res[key] = r
        log(f"R {R:.4f}: {r}")
    jdump(out / ("repair_nopresolve.json" if args.no_presolve else "repair.json"), res)


# --------------------------------------------------------------------------- #
# step 3: LP + rounding + repair
# --------------------------------------------------------------------------- #
def lp_solve(cost, size, bops, grp, off, N, meta, R):
    from scipy.optimize import Bounds, LinearConstraint, milp
    A, lo, hi, mb, mo = matrix(size, bops, grp, N, R, meta, 0.0)
    scale = float(np.abs(cost).max()) / 1e4 or 1.0
    t0 = time.perf_counter()
    res = milp(cost / scale, integrality=np.zeros(len(cost)), bounds=Bounds(0, 1), constraints=LinearConstraint(A, lo, hi))
    t1 = time.perf_counter()
    if res.x is None:
        return None, {"lp_status": int(res.status), "lp_s": t1 - t0, "message": str(res.message)}
    x = res.x
    pick, frac = [], 0
    for k in range(N):
        seg = x[off[k]:off[k + 1]]
        nz = np.nonzero(seg > 1e-9)[0]
        if len(nz) > 1:
            frac += 1                                             # fractional: its least-resource option in the LP support
            j = nz[np.lexsort((bops[off[k] + nz], size[off[k] + nz]))[0]]
        else:
            j = int(np.argmax(seg))
        pick.append(off[k] + j)
    pick = np.array(pick)
    pk, steps = repair(pick, cost, size, bops, grp, off, mb, mo)
    t2 = time.perf_counter()
    info = {"lp_status": int(res.status), "lp_s": t1 - t0, "round_repair_s": t2 - t1, "fractional_nodes": frac,
            "repair_steps": steps, "lp_bound": float(res.fun) * scale}
    if pk is None:
        return None, info
    info.update(objective=float(cost[pk].sum()), **check(pk, size, bops, mb, mo))
    return pk, info


def stage_lp(args, out, log):
    cost, size, bops, grp, off, N, meta = load_problem(out)
    raw = jload(SRC / "selection_raw.json")
    names, lab = meta["names"], meta["labels"]
    mil = {}
    for c in raw["candidates"]:
        if c["origin"] == "milp":
            mil.setdefault(round(c["search_budget"], 10), c)
    res = {}
    for log_row in raw["solver_log_"]:
        R = log_row["budget"]
        pk, info = lp_solve(cost, size, bops, grp, off, N, meta, R)
        m = mil.get(round(R, 10))
        row = {"R": R, **info, "milp_seconds": log_row.get("seconds"), "milp_margin_used": log_row.get("budget_margin_used"),
               "milp_failed": "failure" in log_row}
        if m is not None and pk is not None:
            fm = {names[k]: lab[pk[k]] for k in range(N)}
            diff = sum(fm[n] != f"{m['formats'][n]['w']}/{m['formats'][n]['a']}" for n in names)
            row.update(milp_objective=m["objective"], lp_minus_milp_rel=(info["objective"] - m["objective"]) / m["objective"],
                       nodes_differing=diff, lp_bound_vs_milp_rel=(info["lp_bound"] - m["objective"]) / m["objective"])
        res[f"{R:.6f}"] = row
        log(f"R {R:.4f}: LP {info['lp_s']:.2f}s + {info.get('round_repair_s', 0):.2f}s, frac {info.get('fractional_nodes')}, "
            f"obj {info.get('objective')}, MILP {row.get('milp_objective')} ({row.get('milp_seconds')}s), "
            f"rel {row.get('lp_minus_milp_rel')}, nodes diff {row.get('nodes_differing')}")
    jdump(out / "lp.json", res)


def stage_lpauto(args, out, log):
    """mp_auto.auto_select with milp_select replaced (in this process only) by LP + rounding + repair."""
    sys.argv = [sys.argv[0]]
    import olmoe_common  # noqa: F401
    import mp_auto
    from mp_auto import config_hash
    from run_olmoe_multipoint import problem, nondominated, pruned
    from run_native_bc_pipeline import GRID11
    from htq_pareto_search import FORMAT_BITS
    table, meta_e, names, nodes, elems, rows, flops = problem(SRC)
    bc = lambda n, w, a: table["B"][n][w] + table["C"][n][a]
    keep = nondominated(names, bc)
    cost, size, bops, grp, off, N, meta = load_problem(out)
    lab = meta["labels"]

    def lp_select(nodes_, wf, af, value, weight_elems, flops_proxy, max_bytes, max_bops, time_limit=600, info=None):
        R = max_bytes / meta["ref_bytes"]
        pk, inf = lp_solve(cost, size, bops, grp, off, N, meta, R)
        if pk is None:
            raise RuntimeError(f"LP route found no solution: {inf}")
        if info is not None:
            info.update(status=0, message="LP+repair", seconds=inf["lp_s"] + inf["round_repair_s"], mip_gap=None,
                        lp_bound=inf["lp_bound"], fractional_nodes=inf["fractional_nodes"], repair_steps=inf["repair_steps"])
        fm = {names[k]: dict(zip(("w", "a"), lab[pk[k]].split("/"))) for k in range(N)}
        return fm, float(cost[pk].sum()), 0
    mp_auto.milp_select = lp_select
    t0 = time.perf_counter()
    slog = []
    sel = mp_auto.auto_select(nodes, GRID11, GRID11, pruned(bc, keep), elems, flops, samples=17, refinements=2,
                              time_limit=300, solver_log=slog)
    sec = time.perf_counter() - t0
    raw = jload(SRC / "selection_raw.json")
    K, KM = sel["selected"], raw["selected"]
    h = lambda f: config_hash({n: f["formats"][n] for n in names})
    diff = sum(K["formats"][n] != KM["formats"][n] for n in names)
    res = {"seconds": sec, "milp_auto_select_seconds": raw.get("seconds_"), "solves": len(slog),
           "knee_lp": {k: K[k] for k in ("resource", "size", "bops", "objective")},
           "knee_milp": {k: KM[k] for k in ("resource", "size", "bops", "objective")},
           "same_knee_config": h(K) == h(KM), "knee_nodes_differing": diff,
           "frontier_lp": [{k: f[k] for k in ("resource", "objective", "origin")} for f in sel["frontier"]],
           "frontier_milp": [{k: f[k] for k in ("resource", "objective", "origin")} for f in raw["frontier"]],
           "failures": sel["failures"], "solver_log": slog}
    jdump(out / "lpauto.json", res)
    log(f"LP auto_select {sec:.0f}s ({len(slog)} solves) vs MILP {raw.get('seconds_'):.0f}s; knee LP R {K['resource']:.4f} "
        f"BC {K['objective']:.4f} vs MILP R {KM['resource']:.4f} BC {KM['objective']:.4f}; same config {res['same_knee_config']}, "
        f"{diff} nodes differ")


# --------------------------------------------------------------------------- #
# step 4
# --------------------------------------------------------------------------- #
def stage_gap(args, out, log):
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else DIAG_BUDGETS[1:6]
    f = out / "gap.json"
    res = jload(f) if f.exists() else {}
    for g in [float(x) for x in args.gaps.split(",")]:
        for R in budgets:
            key = f"{g:g}|{R:.6f}"
            if key in res:
                continue
            a = run_attempt(out, R, 0.0, g, args.time_limit, f"gap_{g:g}_{R:.6f}")
            res[key] = {k: a.get(k) for k in ("R", "gap_setting", "status", "solve_s", "objective", "dual_bound", "mip_gap",
                                              "feasible", "M_over_budget", "B_over_budget", "first_incumbent", "nodes")}
            jdump(f, res)
            log(f"gap {g:g} R {R:.4f}: {a['solve_s']:.0f}s obj {a.get('objective')} feasible {a.get('feasible')}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, o, R, shrink, gap, tl, rp, pre = sys.argv
        worker(Path(o), float(R), float(shrink), float(gap), float(tl), rp, pre == "1")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("problem", "diag", "repair", "lp", "lpauto", "gap"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--gaps", default="1e-4,1e-3")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--no-presolve", action="store_true", help="diag: HiGHS presolve off (same problem)")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
    timing["runs"].append({"stage": args.stage, "budgets": args.budgets, "wall_s": time.time() - t0,
                           "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
