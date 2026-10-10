"""Exact per-node format selection by MILP (scipy's HiGHS), as an alternative to htq_pareto_search's tree search.

Picks one (weight format, activation format) per weight-bearing node to minimize a per-(node, option) table
(the HTQ profit sens_w * w_err + sens_a * a_err, or sqnr_tables.sqnr_table) subject to
  weight bytes  sum elems * bits(w) / 8          <= max_bytes
  BOPs proxy    sum flops_proxy * bits(w) * bits(a) <= max_bops
the same cost model pareto_search uses. Activation-only nodes cost nothing there, so they stay fp32 (zero
perturbation, zero noise) and are not variables. The objective is additive, so the answer is the true
optimum (HiGHS proves it); the tree search's pruning and beam cap give no such guarantee.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from htq_pareto_search import FORMAT_BITS


def milp_select(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy, max_bytes, max_bops,
                time_limit=600):
    """value(node_name, w, a) -> float, or None to forbid the option. Returns (formats, objective, status)
    with formats in leaf_config()'s shape: {node_name: {"w": fmt or None, "a": fmt}}."""
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
    for shrink in (0.0, 1e-6):
        hi = np.r_[np.ones(len(opts)), 1 - shrink, 1 - shrink]
        lo = np.r_[np.ones(len(opts)), -np.inf, -np.inf]
        res = milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(A, lo, hi), options={"mip_rel_gap": 0.0, "time_limit": time_limit})
        if res.x is None:
            raise RuntimeError(f"MILP found no solution: {res.message}")
        pick = [off[k] + int(np.argmax(res.x[off[k]:off[k + 1]])) for k in range(len(opts))]
        if size[pick].sum() <= max_bytes and bops[pick].sum() <= max_bops:
            break
    else:
        raise RuntimeError("MILP answer exceeds the budget even with a 1 ppm margin")
    formats = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
    for k, name in enumerate(weighted):
        w, a = opts[k][pick[k] - off[k]]
        formats[name] = {"w": w, "a": a}
    return formats, float(cost[pick].sum()), int(res.status)
