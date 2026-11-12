"""Budget-free selection on a sampled BC/resource frontier.

The policy is explicit: resource = max(bytes / int8_bytes, BOPs / int8_BOPs).
Search bounds come from the cheapest and minimum-BC assignments, not user
budgets. Select maximum distance below the normalized endpoint chord, refining
around that knee twice. This is a heuristic tradeoff on predicted BC error,
not an accuracy constraint or an enumeration of the full 3-D Pareto frontier.
No held-out measurements enter selection. MILP status is preserved, including
time-limited incumbents and mp_milp's repaired answers (both status 1), which
must not be described as proven optimal.
"""
import hashlib
import json
import math

import numpy as np

from htq_pareto_search import FORMAT_BITS
from mp_milp import milp_select


def _frontier(rows):
    """Non-dominated points in the scalar resource / predicted-error plane."""
    out = []
    for row in sorted(rows, key=lambda r: (r['resource'], r['objective'],
                                          r['solver_status'] != 0)):
        if not out or row['objective'] < out[-1]['objective']:
            out.append(row)
    return out


def _knee(frontier):
    if len(frontier) == 1:
        return frontier[0], [0.0]
    lo, hi = frontier[0], frontier[-1]
    dx = hi['resource'] - lo['resource']
    dy = lo['objective'] - hi['objective']
    scores = [1.0 - (r['resource'] - lo['resource']) / dx
              - (r['objective'] - hi['objective']) / dy for r in frontier]
    # Ties prefer lower predicted error; a straight line has no identifiable
    # interior knee, so this deterministically chooses its high-quality end.
    index = max(range(len(frontier)), key=lambda i: (round(scores[i], 12),
                                                    -frontier[i]['objective']))
    return frontier[index], scores


def auto_select(nodes, weight_formats, act_formats, value, weight_elems,
                flops_proxy, *, samples=17, refinements=2, time_limit=60, solver_log=None):
    """Return selected config, sampled frontier, and auditable policy metadata.

`value` and the resource dictionaries have the same contract as milp_select.
None/NaN/Inf costs forbid options. BC must be nonnegative. samples and
refinements control search effort, not the user's desired resource budget.
A list passed as solver_log receives one record per MILP solve (budget and
milp_select's info: status, gap, seconds) or failure.
"""
    if samples < 3 or refinements < 0 or time_limit <= 0:
        raise ValueError('need samples >= 3, refinements >= 0, time_limit > 0')
    names = [n['name'] for n in nodes if n.get('weight_const')]
    if not names or len(names) != len(set(names)):
        raise ValueError('need uniquely named weight-bearing nodes')
    for n in names:
        if not (math.isfinite(weight_elems[n]) and weight_elems[n] > 0
                and math.isfinite(flops_proxy[n]) and flops_proxy[n] > 0):
            raise ValueError(f'{n}: resource estimates must be finite and positive')
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
                if v < 0:
                    raise ValueError(f'{n}/{w}/{a}: BC cost must be nonnegative')
                opts.append((w, a, float(v), weight_elems[n] * FORMAT_BITS[w] / 8,
                             flops_proxy[n] * FORMAT_BITS[w] * FORMAT_BITS[a]))
        if not opts:
            raise ValueError(f'{n}: every option is forbidden')
        options[n] = opts
    lookup = {(n, w, a): v for n, opts in options.items() for w, a, v, _, _ in opts}

    def cost(n, w, a):
        return lookup.get((n, w, a))

    def record(cfg, budget, status, origin):
        objective = sum(lookup[n, cfg[n]['w'], cfg[n]['a']] for n in names)
        size = sum(weight_elems[n] * FORMAT_BITS[cfg[n]['w']] / 8 for n in names) / ref_bytes
        bops = sum(flops_proxy[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']]
                   for n in names) / ref_bops
        return dict(formats=cfg, objective=objective, size=size, bops=bops,
                    resource=max(size, bops), search_budget=budget,
                    solver_status=status, optimal=status == 0, origin=origin)

    def assignment(key):
        cfg = {n['name']: {'w': None, 'a': 'fp32'} for n in nodes if not n.get('weight_const')}
        for n in names:
            w, a, *_ = min(options[n], key=key)
            cfg[n] = {'w': w, 'a': a}
        return cfg

    cheap = record(assignment(lambda o: (o[3], o[4], o[2])), None, None, 'cheap_endpoint')
    quality = record(assignment(lambda o: (o[2], max(o[3] / ref_bytes, o[4] / ref_bops))),
                     None, 0, 'minimum_bc_endpoint')
    # Independent minima provide a lower bound; for finite separable B(w)+C(a)
    # this is feasible. A general callback may make the lower bound infeasible.
    lower = max(sum(min(o[3] for o in options[n]) for n in names) / ref_bytes,
                sum(min(o[4] for o in options[n]) for n in names) / ref_bops)
    upper = max(cheap['resource'], quality['resource'])
    candidates, failures, attempted = [cheap, quality], [], []

    def solve(budget):
        if any(math.isclose(budget, b, rel_tol=1e-12) for b in attempted):
            return
        attempted.append(float(budget))
        info = {} if solver_log is not None else None
        try:
            cfg, _, status = milp_select(nodes, weight_formats, act_formats, cost,
                                         weight_elems, flops_proxy,
                                         budget * ref_bytes, budget * ref_bops,
                                         time_limit=time_limit, info=info)
        except RuntimeError as exc:
            failures.append({'budget': float(budget), 'message': str(exc)})
            if solver_log is not None:
                solver_log.append({'budget': float(budget), 'failure': str(exc)})
            return
        if solver_log is not None:
            solver_log.append({'budget': float(budget), **info})
        if status not in (0, 1):
            raise RuntimeError(f'auto MILP returned solver status {status}')
        row = record(cfg, float(budget), status, 'milp')
        if row['resource'] > budget + 1e-12:
            raise RuntimeError('auto MILP answer exceeds its resource budget')
        candidates.append(row)

    if quality['objective'] != cheap['objective'] or cheap['resource'] > lower:
        for b in np.geomspace(lower, upper, samples):
            solve(float(b))
        for _ in range(refinements):
            frontier = _frontier(candidates)
            selected, _ = _knee(frontier)
            i = frontier.index(selected)
            left = frontier[max(0, i - 1)]['resource']
            right = frontier[min(len(frontier) - 1, i + 1)]['resource']
            for b in np.geomspace(left, right, 5)[1:-1]:
                solve(float(b))
    if attempted and not any(r['origin'] == 'milp' for r in candidates):
        raise RuntimeError(f'no successful auto MILP solves: {failures}')
    frontier = _frontier(candidates)
    selected, scores = _knee(frontier)
    return dict(policy='normalized_chord_knee', selection_metric='predicted_BC',
                resource_metric='max(weight_bytes/int8_bytes, BOPs/int8_BOPs)',
                bounds={'lower': lower, 'upper': upper},
                reference={'weight_bytes': ref_bytes, 'bops': ref_bops},
                selected=selected, frontier=frontier, knee_scores=scores,
                candidates=candidates, attempted_budgets=attempted, failures=failures,
                excluded_nonfinite_options=excluded,
                samples=samples, refinements=refinements,
                accuracy_guarantee=False)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def auto_select_3d(nodes, weight_formats, act_formats, value, weight_elems,
                   flops_proxy, *, samples=17, refinements=2,
                   local_factors=(0.8, 1.0, 1.25), time_limit=10):
    """Budget-free selection over (weight size, BOPs, BC): the flow compared
against HTQ in benchmark_native_search.py (method milp_local,
runs/search_milp_htq_20261010):
  1. auto_select's common-budget sweep and knee refinements;
  2. independent size / BOPs budgets around that knee's actual resources, every
     pair of local_factors except (1, 1);
  3. the 3-D non-dominated set of all candidates and two analytic anchors
     (per node the cheapest formats; per node the minimum BC);
  4. coordinates normalized by the anchors (size and BOPs from the cheap to
     the minimum-BC anchor, BC from the minimum-BC to the cheap anchor), the
     candidate nearest the ideal point; ties: lower BC, then config hash.
Same contract as auto_select (None/NaN/Inf costs forbid options). Equal
weights and the anchor scale are a policy choice, not an accuracy guarantee.
"""
    coarse = auto_select(nodes, weight_formats, act_formats, value, weight_elems,
                         flops_proxy, samples=samples, refinements=refinements,
                         time_limit=time_limit)
    names = [n['name'] for n in nodes if n.get('weight_const')]
    ref_bytes = coarse['reference']['weight_bytes']
    ref_bops = coarse['reference']['bops']

    def cost(n, w, a):
        v = value(n, w, a)
        return float(v) if v is not None and math.isfinite(v) else None

    def record(cfg, origin, **extra):
        return dict(formats=cfg, config_hash=config_hash(cfg), origin=origin,
                    objective=sum(cost(n, cfg[n]['w'], cfg[n]['a']) for n in names),
                    size=sum(weight_elems[n] * FORMAT_BITS[cfg[n]['w']] / 8
                             for n in names) / ref_bytes,
                    bops=sum(flops_proxy[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']]
                             for n in names) / ref_bops, **extra)

    def anchor(key, origin):
        cfg = {n['name']: {'w': None, 'a': 'fp32'} for n in nodes if not n.get('weight_const')}
        for n in names:
            allowed = [(w, a) for w in weight_formats for a in act_formats
                       if cost(n, w, a) is not None]
            w, a = min(allowed, key=lambda o: key(n, *o))
            cfg[n] = {'w': w, 'a': a}
        return record(cfg, origin)

    bits = FORMAT_BITS
    anchors = [anchor(lambda n, w, a: (bits[w], bits[w] * bits[a], cost(n, w, a)), 'cheap_anchor'),
               anchor(lambda n, w, a: (cost(n, w, a), bits[w], bits[a]), 'quality_anchor')]
    lo = np.array([anchors[0]['size'], anchors[0]['bops'], anchors[1]['objective']])
    hi = np.array([anchors[1]['size'], anchors[1]['bops'], anchors[0]['objective']])
    if not np.all(hi > lo):
        raise ValueError(f'degenerate anchors (no spread on some axis): low {lo}, high {hi}')

    rows = [record(r['formats'], r['origin'], solver_status=r['solver_status'])
            for r in coarse['candidates']]
    center = coarse['selected']
    probes = []
    for fs in local_factors:
        for fp in local_factors:
            if fs == fp == 1:
                continue
            bs, bp = center['size'] * fs, center['bops'] * fp
            cfg, _, status = milp_select(nodes, weight_formats, act_formats, cost,
                                         weight_elems, flops_proxy, bs * ref_bytes,
                                         bp * ref_bops, time_limit=time_limit)
            r = record(cfg, 'local_probe', solver_status=status, size_budget=bs, bops_budget=bp)
            if status not in (0, 1) or r['size'] > bs + 1e-10 or r['bops'] > bp + 1e-10:
                raise RuntimeError(f'local probe ({fs}, {fp}): status {status} or over budget')
            rows.append(r)
            probes.append(dict(size_factor=fs, bops_factor=fp, solver_status=status))

    unique = {}
    for r in rows + anchors:
        unique.setdefault(r['config_hash'], r)
    rows = list(unique.values())
    v = np.array([[r['size'], r['bops'], r['objective']] for r in rows])
    norm = (v - lo) / (hi - lo)
    front = [i for i, x in enumerate(norm)
             if not np.any(np.all(norm <= x + 1e-10, axis=1) & np.any(norm < x - 1e-10, axis=1))]
    for i, r in enumerate(rows):
        r['selection_score'] = float(np.linalg.norm(norm[i]))
    winner = min(front, key=lambda i: (rows[i]['selection_score'], rows[i]['objective'],
                                       rows[i]['config_hash']))
    statuses = [r.get('solver_status') for r in rows if r.get('solver_status') is not None]
    return dict(policy='anchor_normalized_ideal_distance_3d', selection_metric='predicted_BC',
                resources=['weight_bytes/int8_bytes', 'BOPs/int8_BOPs'],
                reference=coarse['reference'], selected=rows[winner], candidates=rows,
                frontier_indices=front, anchors=dict(low=lo.tolist(), high=hi.tolist()),
                coarse_selected={k: center[k] for k in ('size', 'bops', 'objective')},
                coarse_attempts=len(coarse['attempted_budgets']), coarse_failures=coarse['failures'],
                local_probes=probes, time_limited_solves=sum(s == 1 for s in statuses),
                excluded_nonfinite_options=coarse['excluded_nonfinite_options'],
                accuracy_guarantee=False)
