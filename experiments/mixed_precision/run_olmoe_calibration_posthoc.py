#!/usr/bin/env python3
"""POSTHOC (written after the J results of runs/olmoe_calibration_stability_20261018 were seen; exploratory only).

Question for the next hypothesis: is the first-order direction more transferable when it is reliable inside its own
calibration batch? Reliability uses only calibration data: per-article (5 articles x 2 windows) pair differences of F
and of the measured loss, t = mean / (sd / sqrt(5)). |t| bins are the conventional [0,1), [1,2), [2,3), [3,inf) and
were not tuned on J. J is used only to evaluate. Not a method, no coefficient, no change to any frozen result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import run_olmoe_calibration_stability as cs_

BINS = [0, 1, 2, 3, np.inf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    out = ap.parse_args().out_dir
    if (out / 'posthoc_reliability.json').exists():
        raise RuntimeError('exists')
    from run_olmoe_bc_numerics import pair_list
    cands = cs_.read(out / 'candidates.json'); nms = [c['name'] for c in cands]; ix = {n: i for i, n in enumerate(nms)}
    pairs = pair_list(cands); ii = np.array([p['i'] for p in pairs]); jj = np.array([p['j'] for p in pairs])
    J = cs_.article_values(cs_.read(out / 'J/truth.json'), nms, 4); Jd = J.mean(1)[ii] - J.mean(1)[jj]
    gr = cs_.groups_of(cands)
    ch = cs_.read(out / 'choices.json')
    rows = {'score_F': [], 'measured': []}
    grp = []
    for b in cs_.BATCHES:
        s = cs_.read(out / 'scores' / f'{b}.json')
        fw = np.array([sum(np.array(s['projections'][c['node'] + '.' + p]['formats'][wf]['F_per_window'])
                           for p, (wf, af) in c['formats'].items()) for c in cands])             # [360, 10]
        fa = fw.reshape(len(cands), cs_.APB, cs_.WPA).mean(2)                                     # per article
        t = cs_.read(out / b / 'truth.json')
        ma = (np.array([t['rows'][n] for n in nms]) / cs_.TP).reshape(len(cands), cs_.APB, cs_.WPA).mean(2)
        sc = {m: np.array([r[m] for r in s['candidates']]) for m in cs_.METHODS}
        for key, A in (('score_F', fa), ('measured', ma)):
            d = A[ii] - A[jj]
            sd = d.std(1, ddof=1)
            tt = np.where(sd > 0, d.mean(1) / np.maximum(sd, 1e-300) * np.sqrt(cs_.APB), 0.0)
            prim = np.sign(sc[cs_.PRIMARY][ii] - sc[cs_.PRIMARY][jj])
            rows[key].append(np.column_stack([np.abs(tt), np.sign(d.mean(1)) == np.sign(Jd), prim == np.sign(Jd), np.abs(Jd)]))
        # choice level: groups where primary and BC differ, reliability of their F difference
        for k, g in enumerate(gr):
            a, c = ix[ch[b][cs_.PRIMARY][k]], ix[ch[b]['bc'][k]]
            if a == c:
                continue
            d = fa[a] - fa[c]
            sd = d.std(ddof=1)
            grp.append((abs(d.mean() / sd * np.sqrt(cs_.APB)) if sd > 0 else 0.0, float(J[a].mean() - J[c].mean())))
    res = {'scope': 'posthoc exploratory after J results; reliability from calibration articles only; bins fixed a priori by convention'}
    for key, R in rows.items():
        R = np.concatenate(R)
        res[key] = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (R[:, 0] >= lo) & (R[:, 0] < hi)
            res[key].append({'abs_t': [lo, None if hi == np.inf else hi], 'cells': int(m.sum()),
                             'direction_agrees_with_J': float(R[m, 1].mean()) if m.any() else None,
                             'primary_agrees_with_J': float(R[m, 2].mean()) if m.any() else None,
                             'median_abs_J_diff': float(np.median(R[m, 3])) if m.any() else None})
    G = np.array(grp)
    res['changed_groups'] = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (G[:, 0] >= lo) & (G[:, 0] < hi)
        res['changed_groups'].append({'abs_t': [lo, None if hi == np.inf else hi], 'cells': int(m.sum()),
                                      'mean_J_primary_minus_bc': float(G[m, 1].mean()) if m.any() else None,
                                      'share_primary_better': float((G[m, 1] < 0).mean()) if m.any() else None})
    cs_.write(out / 'posthoc_reliability.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
