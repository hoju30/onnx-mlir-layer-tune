"""Exact per-node format selection by MILP (scipy's HiGHS), as an alternative to htq_pareto_search's tree search.

Picks one (weight format, activation format) per weight-bearing node to minimize a per-(node, option) table
(the HTQ profit sens_w * w_err + sens_a * a_err, or sqnr_tables.sqnr_table) subject to
  weight bytes  sum elems * bits(w) / 8          <= max_bytes
  BOPs proxy    sum flops_proxy * bits(w) * bits(a) <= max_bops
the same cost model pareto_search uses. Activation-only nodes cost nothing there, so they stay fp32 (zero
perturbation, zero noise) and are not variables. The objective is additive, so the answer is the true
optimum (HiGHS proves it); the tree search's pruning and beam cap give no such guarantee.
"""
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from htq_pareto_search import FORMAT_BITS


def _number(v):
    return float(v) if v is not None and math.isfinite(v) else None


def milp_select(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy, max_bytes, max_bops,
                time_limit=600, info=None):
    """value(node_name, w, a) -> float, or None to forbid the option. Returns (formats, objective, status)
    with formats in leaf_config()'s shape: {node_name: {"w": fmt or None, "a": fmt}}. status is HiGHS's, except 1
    (feasible, not proven optimal) when the answer had to be repaired: attempt 1 over the exact budget and the 1 ppm
    retry without a within-budget answer. A dict passed as `info` receives HiGHS's account of the final solve
    (status, message, mip_gap, mip_dual_bound in the scaled objective, objective_scale, node count, seconds, whether
    the 1 ppm budget margin was needed, whether and with how many option swaps attempt 1's answer was repaired)."""
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
    cost, size, bops = np.array(cost), np.array(size), np.array(bops)
    rows = np.concatenate([np.repeat(np.arange(len(opts)), [len(r) for r in opts]), np.full(len(cost), len(opts)),
                           np.full(len(cost), len(opts) + 1)])
    cols = np.concatenate([np.arange(len(cost))] * 3)
    vals = np.concatenate([np.ones(len(cost)), size / max_bytes, bops / max_bops])
    A = coo_matrix((vals, (rows, cols)), shape=(len(opts) + 2, len(cost))).tocsr()
    # HiGHS keeps an absolute gap even at mip_rel_gap = 0, so scale the objective to O(1e4).
    scale = float(np.abs(cost).max()) / 1e4 or 1.0
    # Solve at the exact budget first (a config can sit exactly on it); only if HiGHS's feasibility
    # tolerance let the answer overshoot the exact check, retry with the budget shrunk by 1 ppm.
    t0 = time.perf_counter()

    def solve(shrink):
        hi = np.r_[np.ones(len(opts)), 1 - shrink, 1 - shrink]
        lo = np.r_[np.ones(len(opts)), -np.inf, -np.inf]
        res = milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(A, lo, hi), options={"mip_rel_gap": 0.0, "time_limit": time_limit})
        pick = None if res.x is None else [off[k] + int(np.argmax(res.x[off[k]:off[k + 1]])) for k in range(len(opts))]
        return res, pick

    def within(pick):
        return size[pick].sum() <= max_bytes and bops[pick].sum() <= max_bops
    res, pick = solve(0.0)
    if pick is None:
        raise RuntimeError(f"MILP found no solution: {res.message}")
    shrink, swaps = 0.0, None
    if not within(pick):
        res2, pick2 = solve(1e-6)
        if pick2 is not None and within(pick2):
            res, pick, shrink = res2, pick2, 1e-6
        else:
            # The retry can fail where attempt 1 had an answer (at the lowest budget of runs/olmoe_milp_speed_20261016
            # the 1 ppm shrink made the problem infeasible): repair attempt 1's answer into the exact budget instead of
            # dropping the budget. The repaired answer is feasible but not proven optimal (status 1).
            pick, swaps = _repair(np.array(pick), cost, size, bops, off, max_bytes, max_bops)
            if pick is None:
                raise RuntimeError("MILP answer exceeds the budget even with a 1 ppm margin and could not be repaired")
    status = int(res.status) if swaps is None else 1
    if info is not None:
        info.update(status=status, message=str(res.message), mip_gap=_number(res.get("mip_gap")),
                    mip_dual_bound=_number(res.get("mip_dual_bound")), objective_scale=scale,
                    mip_node_count=_number(res.get("mip_node_count")), seconds=time.perf_counter() - t0,
                    budget_margin_used=shrink > 0, repaired=swaps is not None, repair_swaps=swaps)
    formats = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
    for k, name in enumerate(weighted):
        w, a = opts[k][pick[k] - off[k]]
        formats[name] = {"w": w, "a": a}
    return formats, float(cost[pick].sum()), status


def _repair(pick, cost, size, bops, off, max_bytes, max_bops, max_steps=1000):
    """Option swaps until the exact budget holds, each step the swap with the smallest cost increase per unit of
    excess removed (only swaps that lower some over-budget resource and keep the other within budget). Returns
    (pick, swaps), or (None, swaps) if no such swap is left."""
    pick = pick.copy()
    steps = 0
    while True:
        s, b = size[pick].sum(), bops[pick].sum()
        exs, exb = max(0.0, s - max_bytes), max(0.0, b - max_bops)
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
            ok = (gain > 0) & (s + ds <= max(max_bytes, s)) & (b + db <= max(max_bops, b))
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
