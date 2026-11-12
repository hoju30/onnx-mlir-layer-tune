#!/usr/bin/env python3
"""Analysis-only supplement to run_olmoe_first_order_validation.py (no GPU, no new loss or gradient).

Written after H truth existed and before its analysis was viewed. Scores, choices, the primary endpoint and the
gate stay as frozen in the main script's analyze stage; this file only adds integrity checks, per-expert
attribution, per-pair historical tracking and timing. Sections named posthoc_* are exploratory diagnostics:
the calibration / D / V measured "selectors" there are reference points, not candidate methods.
"""
import argparse
import csv
import glob
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_first_order_validation as fo
from run_olmoe_bc_numerics import pair_list

SEEDS = {'D': 17041, 'V': 17042, 'H': 17043}  # identical draws to the main analyze stage


def seg_values(truth, nms):
    s = np.array([truth['rows'][n] for n in nms]) / 127
    return s.reshape(len(nms), 16, 4).mean(2)


def boot_idx(out, label):
    meta = fo.read(fo.CT / 'segments.json') if label != 'H' else fo.read(out / 'segments.json')
    meta = sorted([x for x in meta if label == 'H' or x['group'] == label],
                  key=lambda x: x['index'] if label == 'H' else x['index_in_group'])
    rng = np.random.default_rng(SEEDS[label])
    return np.concatenate([rng.choice([i for i, x in enumerate(meta) if x['split'] == sp],
                                      (10000, sum(x['split'] == sp for x in meta)), replace=True)
                           for sp in ('validation', 'test')], 1), meta


def ci(x, idx):
    """x: (..., 16 articles) -> mean and percentile CI95 over article bootstrap rows."""
    x = np.asarray(x, dtype=float)
    b = np.stack([x[..., r].mean(-1) for r in idx], -1)
    m, p = x.mean(-1), np.percentile(b, [2.5, 97.5], axis=-1)
    if np.ndim(m) == 0:
        return {'mean': float(m), 'ci95': p.tolist()}
    return {'mean': m, 'ci95': p}  # vectorised (pairs): arrays, not written to JSON directly


def excl(c):
    lo, hi = c['ci95']
    return 1 if lo > 0 else -1 if hi < 0 else 0


def integrity(out, cs, nms, scr, H):
    import olmoe_common as oc
    from run_olmoe_crosstext import used_spans
    r = {}
    plan, sf, ff = fo.read(out / 'plan.json'), fo.read(out / 'score_freeze.json'), fo.read(out / 'fresh_freeze.json')
    r['hashes'] = {'plan_matches_score_freeze': fo.sha(out / 'plan.json') == sf['plan_sha256'],
                   'scores_match_score_and_fresh_freeze': fo.sha(out / 'scores.json') == sf['scores_sha256'] == ff['scores_sha256'],
                   'tokens_match_fresh_freeze': fo.sha(out / 'H/tokens.npy') == ff['tokens_sha256'],
                   'plan_inputs_unchanged': all(fo.sha(fo.HERE / p) == h for p, h in plan['inputs'].items()),
                   'truth_script_sha256': H['script_sha256'],
                   'executed_snapshot_sha256': fo.sha(out / 'executed_script_snapshot.py'),
                   'current_main_script_sha256': fo.sha(fo.HERE / 'run_olmoe_first_order_validation.py')}
    r['candidates'] = {'equal_stage1_weight_only': cs == fo.candidates(), 'n': len(cs),
                       'all_activation_fp32': all(af == 'fp32' for c in cs for wf, af in c['formats'].values())}
    A = np.array([H['rows'][n] for n in nms])
    man = fo.read(out / 'H/routes_manifest.json')['w']
    toks = np.load(out / 'H/tokens.npy')
    files = sorted(glob.glob(str(out / 'H/routes/w_*.npz')))
    ids, rh, rt = [], True, True
    for k, f in enumerate(files):
        z = np.load(f)
        rh &= oc.route_hash(z['ids'], z['weights']) == man['hash'][k]
        rt &= bool((z['tokens'] == toks[k]).all())
        ids.append(z['ids'])
    ids = np.stack(ids)
    zero_mismatch, coverage = 0, {}
    for node in sorted({c['node'] for c in cs}):
        l, e = int(node.split('.')[0][1:]), int(node.split('.')[1][1:])
        routed = (ids[:, l] == e).any(-1).sum(-1)
        mask = np.array([c['node'] == node for c in cs])
        zero_mismatch += int(((A[mask] == 0).all(0) != (routed == 0)).sum())
        coverage[node] = {'routed_tokens': int(routed.sum()), 'windows_without_tokens': int((routed == 0).sum())}
    r['truth'] = {'names_equal_candidates': sorted(H['rows']) == sorted(nms), 'shape': list(A.shape),
                  'all_finite': bool(np.isfinite(A).all()), 'base_windows': len(H['base']),
                  'base_finite': bool(np.isfinite(H['base']).all()),
                  'base_minus_capture_dynamic_ce_max_abs': float(np.max(np.abs(np.array(H['base']) - np.array(man['summed_ce_dynamic'])))),
                  'route_files': len(files), 'route_hashes_match': bool(rh), 'route_tokens_match': bool(rt),
                  'exact_zero_windows_equal_unrouted_windows_mismatch': zero_mismatch, 'coverage': coverage,
                  'base_mean_nat_per_token': float(np.mean(H['base']) / 127)}
    segs, old = fo.read(out / 'segments.json'), fo.read(fo.CT / 'segments.json')
    used = used_spans()
    r['segments'] = {'n': len(segs), 'splits': dict(Counter(s['split'] for s in segs)),
                     'unique_titles': len({s['title'] for s in segs}),
                     'title_overlap_DV': len({s['title'] for s in segs} & {s['title'] for s in old}),
                     'article_overlap_DV': sum(o['split'] == s['split'] and o['article_chars'][0] < s['article_chars'][1] and s['article_chars'][0] < o['article_chars'][1] for s in segs for o in old),
                     'article_overlap_earlier_olmoe_sets': sum(s['article_chars'][0] < b and a < s['article_chars'][1] for s in segs for a, b in used[s['split']]),
                     'order_validation_then_test': [s['split'] for s in segs] == ['validation'] * 8 + ['test'] * 8}
    ps = fo.read(out / 'projection_scores.json')
    err = 0.0
    for c, s in zip(cs, scr):
        f = sum(ps[c['node'] + '.' + p]['formats'][wf]['first'] for p, (wf, af) in c['formats'].items())
        b = sum(ps[c['node'] + '.' + p]['formats'][wf]['bc'] for p, (wf, af) in c['formats'].items() if wf != 'fp32')
        err = max(err, abs(f - s['first']), abs(b - s['bc']) / abs(s['bc']),
                  abs(s['first_plus_half_bc'] - s['first'] - 0.5 * s['bc']), abs(s['first_plus_bc'] - s['first'] - s['bc']))
    gc = fo.read(out / 'gradient_checks.json')
    r['scores'] = {'recomposed_from_projection_scores_max_error': err, 'gradient_checks_pass': gc['pass'],
                   'bc_max_relative': gc['bc_max_relative'], 'up_case_relative': gc['up_case_relative']}
    return r


def posthoc_coverage(out):
    """Added AFTER viewing H results (largest harm came from L15.E59). Splits the frozen primary statistic by the
    stage-1 coverage tiers (high/mid/low, fixed in runs/olmoe_fixed_route_20261015 before this study). Descriptive
    only: it cannot change the primary result, gate or method."""
    if (out / 'posthoc_coverage.json').exists():
        raise RuntimeError('posthoc_coverage exists')
    import glob as g_
    cs = fo.read(out / 'candidates.json'); nms = [c['name'] for c in cs]
    ch = fo.read(out / 'choices.json')
    sel = fo.read(fo.S1 / 'coverage.json')['selection']
    tier = {f'L{l}.E{v["expert"]}': (t, v['routed_tokens'], v['windows']) for l, x in sel.items() for t, v in x['picks'].items()}
    ix = {n: i for i, n in enumerate(nms)}
    res = {'scope': 'posthoc after H results; descriptive; coverage tiers predate this study', 'experts': {}, 'tiers': {}}
    for s in ('D', 'V', 'H'):
        truth = fo.read(out / 'H/truth.json') if s == 'H' else fo.read(fo.CT / s / 'truth.json')
        seg = seg_values(truth, nms)
        idx, _ = boot_idx(out, s)
        folder = out / 'H' if s == 'H' else fo.CT / s
        ids = np.stack([np.load(f)['ids'] for f in sorted(g_.glob(str(folder / 'routes/w_*.npz')))])
        d = {m: np.array([seg[ix[a['name']]] - seg[ix[b['name']]] for a, b in zip(ch[m], ch['bc'])]) for m in fo.METHODS[1:]}
        node = np.array([x['node'] for x in ch['bc']])
        for n, (t, tok, win) in tier.items():
            l, e = int(n.split('.')[0][1:]), int(n.split('.')[1][1:])
            res['experts'].setdefault(n, {'tier': t, 'calibration_routed_tokens': tok, 'calibration_windows': win})
            res['experts'][n][s + '_routed_tokens'] = int((ids[:, l] == e).any(-1).sum())
        res['tiers'][s] = {m: {t: ci(d[m][np.isin(node, [n for n, v in tier.items() if v[0] == t])].mean(0), idx)
                               for t in ('high', 'mid', 'low')} for m in d}
    fo.write(out / 'posthoc_coverage.json', res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--posthoc-coverage', action='store_true')
    a_ = ap.parse_args(); out = a_.out_dir
    if a_.posthoc_coverage:
        return posthoc_coverage(out)
    if (out / 'supplement.json').exists():
        raise RuntimeError('supplement exists')
    t0 = time.time()
    cs = fo.read(out / 'candidates.json'); nms = [c['name'] for c in cs]
    scr = fo.read(out / 'scores.json')
    assert nms == [s['name'] for s in scr]
    vals = {m: np.array([s[m] for s in scr]) for m in fo.METHODS}
    pairs = pair_list(cs); ii = np.array([p['i'] for p in pairs]); jj = np.array([p['j'] for p in pairs])
    dirs = {m: np.sign(v[ii] - v[jj]) for m, v in vals.items()}
    byres = defaultdict(list)
    for i, c in enumerate(cs):
        byres[(c['node'], c['M_bytes'], c['BOPs_calibration'])].append(i)
    groups = [v for v in byres.values() if len(v) > 1]
    choose = lambda v: np.array([min(g, key=lambda i: (v[i], nms[i])) for g in groups])
    choices = {m: choose(vals[m]) for m in fo.METHODS}
    gnode = np.array([cs[g[0]]['node'] for g in groups]); nodes = sorted(set(gnode))
    H = fo.read(out / 'H/truth.json')
    res = {'note': 'analysis-only supplement; primary endpoint and gate are in analysis.json', 'integrity': integrity(out, cs, nms, scr, H)}

    # choices must be exactly the main analyze choices and resource-matched
    saved = fo.read(out / 'choices.json')
    res['integrity']['choices'] = {
        'equal_main_analyze_choices_json': all([x['name'] for x in saved[m]] == [nms[i] for i in choices[m]] for m in fo.METHODS),
        'all_within_group': all(choices[m][k] in g for m in fo.METHODS for k, g in enumerate(groups)),
        'groups': len(groups), 'groups_per_expert': dict(Counter(gnode.tolist())), 'pairs': len(pairs)}

    seg = {'H': seg_values(H, nms)}
    for s in ('D', 'V'):
        seg[s] = seg_values(fo.read(fo.CT / s / 'truth.json'), nms)
    calrows = fo.read(fo.S1 / 'truth_calibration.json')['rows']
    cal = np.array([np.mean(calrows[n]) / 127 for n in nms])  # 10 calibration windows, nat/token
    idx, meta = {}, {}
    for s in ('D', 'V', 'H'):
        idx[s], meta[s] = boot_idx(out, s)

    # ---- per-expert attribution of the group-mean choice difference
    attr = {}
    for s in ('D', 'V', 'H'):
        attr[s] = {}
        for m in fo.METHODS[1:]:
            d = seg[s][choices[m]] - seg[s][choices['bc']]  # (63, 16)
            tot = d.mean(0)
            x = {'overall': ci(tot, idx[s]), 'changed_groups': int((choices[m] != choices['bc']).sum()),
                 'split_means': {sp: float(tot[[k for k, y in enumerate(meta[s]) if y['split'] == sp]].mean()) for sp in ('validation', 'test')},
                 'experts': {}}
            for n in nodes:
                mk = gnode == n
                contrib = d[mk].sum(0) / len(groups)
                x['experts'][n] = {'changed_groups': int((choices[m][mk] != choices['bc'][mk]).sum()),
                                   'contribution_to_overall': ci(contrib, idx[s]),
                                   'share_of_overall_point_mean': float(contrib.mean() / tot.mean()) if tot.mean() != 0 else None,
                                   'leave_expert_out': ci(d[~mk].mean(0), idx[s])}
            gm = d.mean(1)
            ch = choices[m] != choices['bc']
            x['changed_group_point_signs'] = {'better': int((gm[ch] < 0).sum()), 'worse': int((gm[ch] > 0).sum()), 'zero': int((gm[ch] == 0).sum())}
            x['changed_group_ci_signs'] = dict(Counter(['better' if e < 0 else 'worse' if e > 0 else 'unresolved'
                                                        for e in (excl(ci(d[k], idx[s])) for k in np.where(ch)[0])]))
            attr[s][m] = x
    res['choice_attribution'] = attr

    rows = []
    for k, g in enumerate(groups):
        if choices[fo.PRIMARY][k] == choices['bc'][k]:
            continue
        a, b = choices[fo.PRIMARY][k], choices['bc'][k]
        c = cs[g[0]]
        r = {'node': c['node'], 'M_bytes': c['M_bytes'], 'BOPs_calibration': c['BOPs_calibration'], 'group_size': len(g),
             'bc_choice': nms[b], 'primary_choice': nms[a], 'calibration_measured': cal[a] - cal[b]}
        for s in ('D', 'V', 'H'):
            v = ci(seg[s][a] - seg[s][b], idx[s])
            r.update({s + '_mean': v['mean'], s + '_lo': v['ci95'][0], s + '_hi': v['ci95'][1]})
        rows.append(r)
    with open(out / 'changed_groups.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- all 909 pairs: where predictions change, who does H side with (descriptive; pairs share candidates)
    pm = {s: seg[s][ii] - seg[s][jj] for s in seg}  # (909, 16)
    pc = {s: ci(pm[s], idx[s]) for s in seg}
    pairs_out = {}
    for s in ('D', 'V', 'H'):
        sgn = np.sign(pc[s]['mean']); ex = np.array([excl({'ci95': [lo, hi]}) for lo, hi in np.array(pc[s]['ci95']).T])
        pairs_out[s] = {}
        for m in fo.METHODS[1:]:
            ch = dirs[m] != dirs['bc']
            gain = pm[s][ch] * dirs[m][ch][:, None]  # >0 where the method's changed direction matches measured loss
            pairs_out[s][m] = {'changed_pairs': int(ch.sum()),
                               'changed_by_expert': dict(Counter(np.array([p['node'] for p in pairs])[ch].tolist())),
                               'method_point_agreement_on_changed': int((dirs[m][ch] == sgn[ch]).sum()),
                               'bc_point_agreement_on_changed': int((dirs['bc'][ch] == sgn[ch]).sum()),
                               'unadjusted_ci_resolved_on_changed': {'method_right': int((ex[ch] == dirs[m][ch]).sum()),
                                                                     'bc_right': int((ex[ch] == dirs['bc'][ch]).sum()),
                                                                     'unresolved': int((ex[ch] == 0).sum())},
                               'mean_signed_gain_per_changed_pair': ci(gain.mean(0), idx[s]) if ch.any() else None}
    res['changed_pairs'] = pairs_out

    # ---- historical 28 BC-right / 7 BC-wrong (prediction change and whether H supports it)
    a = fo.read(fo.CT / 'analysis.json')
    wrong = {tuple(x[:2]) for x in a['reproducible_bc_errors_list']}
    lookup = {(nms[p['i']], nms[p['j']]): k for k, p in enumerate(pairs)}
    hist = []
    for r in csv.DictReader(open(fo.CT / 'pairs.csv')):
        if r.get('V_category') != 'confirmed_same' or (r['a'], r['b']) not in lookup:
            continue
        k = lookup[(r['a'], r['b'])]
        lab = 'bc_wrong' if (r['a'], r['b']) in wrong else 'bc_right'
        hm, (hlo, hhi) = pc['H']['mean'][k], np.array(pc['H']['ci95'])[:, k]
        he = excl({'ci95': [hlo, hhi]})
        row = {'label': lab, 'node': pairs[k]['node'], 'a': r['a'], 'b': r['b'],
               **{m + '_dir': int(dirs[m][k]) for m in fo.METHODS},
               'primary_changed': bool(dirs[fo.PRIMARY][k] != dirs['bc'][k]),
               'F_diff': float(vals['first'][ii[k]] - vals['first'][jj[k]]), 'B_diff': float(vals['bc'][ii[k]] - vals['bc'][jj[k]]),
               'calibration_measured_diff': float(cal[ii[k]] - cal[jj[k]]),
               'D_mean': pc['D']['mean'][k], 'V_mean': pc['V']['mean'][k],
               'H_mean': hm, 'H_lo': hlo, 'H_hi': hhi,
               'H_articles_a_worse': int((pm['H'][k] > 0).sum()), 'H_articles_a_better': int((pm['H'][k] < 0).sum())}
        p, b = dirs[fo.PRIMARY][k], dirs['bc'][k]
        if row['primary_changed']:
            row['H_verdict'] = ('supports_change' if he == p else 'contradicts_change' if he == b else
                                'unresolved_point_favors_' + ('primary' if np.sign(hm) == p else 'bc'))
        else:
            row['H_verdict'] = ('same_prediction_H_agrees' if he == b else 'same_prediction_H_disagrees' if he == -b else
                                'same_prediction_unresolved_point_' + ('agrees' if np.sign(hm) == b else 'disagrees'))
        hist.append(row)
    with open(out / 'historical_pairs.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    res['historical'] = {lab: {'n': sum(h['label'] == lab for h in hist),
                               'primary_changed': sum(h['label'] == lab and h['primary_changed'] for h in hist),
                               'H_verdicts': dict(Counter(h['H_verdict'] for h in hist if h['label'] == lab)),
                               'distinct_candidates': len({x for h in hist if h['label'] == lab for x in (h['a'], h['b'])}),
                               'nodes': dict(Counter(h['node'] for h in hist if h['label'] == lab))}
                         for lab in ('bc_right', 'bc_wrong')}

    # ---- posthoc: calibration-direction transfer
    ex = fo.read(out / 'exploratory_regressions.json')['pairs']
    l8 = []
    for p in ex:
        k = lookup[(p['a'], p['b'])]
        l8.append({'a': p['a'], 'b': p['b'], 'calibration_measured': float(cal[ii[k]] - cal[jj[k]]),
                   'calibration_measured_recorded': p['measured_calibration_difference'],
                   'F_diff': float(vals['first'][ii[k]] - vals['first'][jj[k]]), 'B_diff': float(vals['bc'][ii[k]] - vals['bc'][jj[k]]),
                   'primary_dir': int(dirs[fo.PRIMARY][k]), 'bc_dir': int(dirs['bc'][k]),
                   **{s: {'mean': pc[s]['mean'][k], 'ci95': np.array(pc[s]['ci95'])[:, k].tolist()} for s in ('D', 'V', 'H')},
                   'H_articles_same_sign_as_calibration': int((np.sign(pm['H'][k]) == np.sign(cal[ii[k]] - cal[jj[k]])).sum())})
    cd = np.sign(cal[ii] - cal[jj])
    transfer = {'scope': 'posthoc exploratory; not used for coefficients, data or gate', 'L8_E33_flipped_pairs': l8}
    transfer['in_sample_calibration_sign_agreement'] = {m: float((dirs[m] == cd).mean()) for m in fo.METHODS}
    transfer['calibration_measured_vs_H_sign'] = float((cd == np.sign(pc['H']['mean'])).mean())
    transfer['D_vs_H_sign'] = float((np.sign(pc['D']['mean']) == np.sign(pc['H']['mean'])).mean())
    transfer['V_vs_H_sign'] = float((np.sign(pc['V']['mean']) == np.sign(pc['H']['mean'])).mean())
    ch = dirs[fo.PRIMARY] != dirs['bc']
    hs = np.sign(pc['H']['mean'])
    transfer['primary_changed_pairs_fidelity_x_transfer'] = {
        'n': int(ch.sum()),
        'primary_matches_calibration_measured': int((dirs[fo.PRIMARY][ch] == cd[ch]).sum()),
        'calibration_measured_matches_H_point': int((cd[ch] == hs[ch]).sum()),
        'both': int(((dirs[fo.PRIMARY][ch] == cd[ch]) & (cd[ch] == hs[ch])).sum())}
    ref = {}
    for name, v in [('calibration_measured', cal), ('D_measured', seg['D'].mean(1)), ('V_measured', seg['V'].mean(1))]:
        cch = choose(v)
        ref[name] = {'changed_groups_vs_bc': int((cch != choices['bc']).sum()),
                     'H_choice_minus_bc': ci((seg['H'][cch] - seg['H'][choices['bc']]).mean(0), idx['H'])}
    transfer['reference_selectors_on_H'] = ref
    transfer['per_expert_sign_agreement_with_H'] = {
        n: {'calibration_measured': float((cd[mk] == hs[mk]).mean()), 'D': float((np.sign(pc['D']['mean'][mk]) == hs[mk]).mean()),
            'bc': float((dirs['bc'][mk] == hs[mk]).mean()), 'primary': float((dirs[fo.PRIMARY][mk] == hs[mk]).mean()), 'pairs': int(mk.sum())}
        for n in nodes for mk in [np.array([p['node'] == n for p in pairs])]}
    res['posthoc_transfer'] = transfer

    tl = [json.loads(x) for x in open(out / 'timing.jsonl')]
    res['timing'] = {'score_stage_wall_s': [x['seconds'] for x in tl if x['stage'] == 'score'],
                     'score_compute_s_from_freeze': fo.read(out / 'score_freeze.json')['seconds'],
                     'score_peak_gpu_gb': fo.read(out / 'score_freeze.json')['peak_gpu_gb'],
                     'fresh_sampling_s': [x['seconds'] for x in tl if x['stage'] == 'fresh'],
                     'truth_stage_wall_s': [x['seconds'] for x in tl if x['stage'] == 'truth'],
                     'truth_replay_s': H['seconds'],
                     'analyze_stage_s': [x['seconds'] for x in tl if x['stage'] == 'analyze'],
                     'supplement_s': time.time() - t0}
    fo.write(out / 'supplement.json', res)
    with open(out / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': 'supplement', 'seconds': time.time() - t0, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': fo.sha(__file__)}) + '\n')
    print(json.dumps({'integrity': {k: v for k, v in res['integrity'].items() if k != 'truth'}}, ensure_ascii=False)[:3000])


if __name__ == '__main__':
    main()
