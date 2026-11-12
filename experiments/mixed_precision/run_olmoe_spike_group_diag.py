#!/usr/bin/env python3
"""OLMoE Down-input activation: spike-grouped sensitivity score (post hoc, CPU-only diagnostic on existing caches).

Three scores per expert Down input and candidate format, all with the same full-channel mean error mu_c and the unchanged bias U:
  A  T_old   = sum_c mean_t(g^2) sum_t r^2 + U                                     (original BC)
  B  T_group = sum_c [mean_S(g^2) sum_S r^2 + mean_N(g^2) sum_N r^2] + U           (spike / normal groups)
  C  T_pair  = sum_c sum_t g^2 r^2 + U                                              (position-paired)
r = e - mu_c (no re-centering inside groups); S = {t : max_c |h_tc| >= 6 x median_t max_c |h_tc|} on the calibration rows.
No model inference, no search, no BC / pipeline change. The cases and losses were analysed before: exploratory, not confirmation.

Stages (one --out-dir): plan, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr

HERE = Path(__file__).resolve().parent
PGD = HERE / 'runs/olmoe_pair_gradient_diag_20261023'
PGN = HERE / 'runs/olmoe_pair_generalization_20261023'
SEQ, TP, WPA = 128, 127, 3
TAU, LOW_COVER, BC_TIE, BOOT = 6.0, 8, 0.02, 10000
SEEDS = {'bootstrap': 23901}
SAME_BIT = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def sources():
    """(group, source dir, tap metadata, formats, measured variant key) for every tap; groups from the source metadata."""
    out = []
    for t in read(PGD / 'taps.json')['taps']:
        out.append({'group': 'mechanism_8', 'src': PGD, 'tap': t, 'formats': ('int4', 'p6e1'), 'key': lambda tid, f: f'tap{tid}|{f}|F'})
    for t in read(PGN / 'selection.json')['taps']:
        g = 'generalization_general' if t['group'] == 'general' else 'generalization_spike'
        out.append({'group': g, 'src': PGN, 'tap': t, 'formats': ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1'), 'key': lambda tid, f: f'tap{tid}|{f}'})
    return out


def source_files():
    fs = [PGD / 'taps.json', PGD / 'measure.json', PGD / 'segments.json', PGD / 'mech.json', PGD / 'freeze.json',
          PGN / 'selection.json', PGN / 'measure.json', PGN / 'segments.json', PGN / 'mech.json', PGN / 'freeze.json']
    fs += sorted(PGD.glob('work/tap*.npz')) + sorted(PGN.glob('work/tap*.npz'))
    return fs


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    files = {str(p.relative_to(HERE)): sha(p) for p in source_files()}
    src = sources()
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'nature': 'post hoc exploratory CPU diagnostic on cases and losses that were already analysed; fixing the rules beforehand does not make '
                  'the old data independent confirmation data',
        'hypothesis': 'grouping each channel\'s rows into spike / normal and averaging g^2 per group keeps the between-group difference without '
                      'the full element-level gradient-error association; not assumed better, element-level information not assumed noise',
        'scope': {'groups': {'mechanism_8': '8 taps of runs/olmoe_pair_gradient_diag_20261023 (formats int4, p6e1; F condition of its measure)',
                             'generalization_general': '16 general taps of runs/olmoe_pair_generalization_20261023 (6 formats; one-tap quantization)',
                             'generalization_spike': '8 spike-stratified taps of the same round'},
                  'taps': [(s['group'], s['tap']['node']) for s in src], 'all_taps_kept': True, 'formats': 'only those in each cache',
                  'mapping': 'expert from the source metadata (taps.json / selection.json node), formats from the cache keys, windows = tok // 128',
                  'not_mixed': 'P / S / R interventions (different spike mask) are not part of the main comparison'},
        'spike': {'rule': f's_t = max_c |h_tc| on the cached FP32 calibration rows; threshold = {TAU} x median_t(s_t) of these rows; S = s_t >= '
                          'threshold, N = the rest; one mask for every format', 'difference': 'earlier rounds used the median of the token-spike '
                  'discovery (calibration + 192 extra windows); both masks are counted and compared', 'fixed': 'never changed by g, errors, loss or '
                  'rankings; no other multiples', 'degenerate': 'threshold non-positive / non-finite -> no valid grouping, T_group = T_old; an empty '
                  f'group -> T_group reduces to T_old exactly', 'low_coverage': f'a non-empty group with < {LOW_COVER} rows is flagged and reported '
                  'separately; the flag never changes weights or selects results; no epsilon replaces an empty group'},
        'formulas': {'A': 'V_old = sum_c mean_t(g^2) sum_t r^2; T_old = V_old + U', 'B': 'V_group = sum_c [mean_S(g^2) sum_S r^2 + mean_N(g^2) sum_N r^2]; '
                     'T_group = V_group + U', 'C': 'V_pair = sum_c sum_t g^2 r^2; T_pair = V_pair + U', 'common': 'e = cached Q(h) - h; mu_c = mean '
                     'over all rows; r = e - mu_c (no group re-centering); U = sum_c GS_c mu_c^2, GS_c = sum_window (sum_t in window g_tc)^2 '
                     '(summed-loss gradients, tok // 128 windows); float64; no penalty, no first-order term, no NLL rescaling'},
        'identities': {'between': 'V_group,c - V_old,c = (n_S n_N / n)(mean_S X - mean_N X)(mean_S Y - mean_N Y), X = g^2, Y = r^2',
                       'within': 'V_pair,c - V_group,c = n_S Cov_S(X, Y) + n_N Cov_N(X, Y) (population covariance; empty group term 0)',
                       'check': 'per channel, relative to the channel scale; T_old / T_pair must reproduce the source mech.json values'},
        'reporting': 'per tap and format: n, spike rows and share, group means and totals of g^2 and r^2, V_old, V_group, V_pair, U, between and '
                     'within terms (size, sign), dominant channels, top-row concentration; absolute values next to ratios',
        'ranking': {'measured': 'per article (3 windows) NLL difference of the two formats under full quantization of the tap; decisive = 95% '
                                f'article-bootstrap CI excludes 0 ({BOOT}, seed {SEEDS["bootstrap"]}), identical for all methods',
                    'primary': 'same-bit INT / Posit pairs (int4-p4e1, int6-p6e1, int8-p8e1); the mechanism_8 group has only int4 vs p6e1 (reported '
                               'as its only pair)', 'secondary': 'all other pairs', 'tie': f'|dT| / max T < {BC_TIE}: listed apart, never correct',
                    'undeterminable': 'pairs with a CI including 0 count for no method', 'classes': 'T_group vs T_old and T_group vs T_pair: fix / '
                    'new error / unchanged; experts involved; whether T_pair fixes are kept; single-expert dependence',
                    'groups_apart': 'the three groups are never pooled into a model-wide accuracy', 'dependence': 'pairs of one expert are not '
                    'independent; no inferential statistics on pooled pairs'},
        'window_stability': 'leave-one-calibration-window-out: the full-calibration threshold kept, the mask restricted to the remaining rows; mu, '
                            'U and the three scores recomputed from the remaining rows; format-order flips and score changes per method; not a '
                            'proof of cross-article or cross-calibration stability',
        'costs': 'CPU recompute time; extra statistics per expert / format (estimate, not a native build cost; grouping is not assumed faster than '
                 'pairing, whose whole-model statistics cost about 3 s)',
        'stop_rule': 'stop this direction if the gain is a single known case, if T_group only beats T_pair but not T_old, or if new errors cancel '
                     'the gain; even a favourable result is not added to the pipeline; any confirmation would be a new pre-registered round',
        'outputs': ['report.md', 'plan.json', 'plan.md', 'scores.csv', 'decomposition.csv', 'ranking_comparisons.csv', 'window_stability.csv',
                    'controls.json', 'timing.json'], 'seeds': SEEDS, 'source_hashes': files, 'code': sha(Path(__file__))}
    write(out / 'plan.json', p)
    md = ['# 分析計畫：OLMoE Down-input activation 的 spike 分組評分（事後探索性診斷）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- 性質：本輪使用已分析過的案例與 loss，是事後探索；事先固定規則也不會讓舊資料變成獨立確認資料。只做既有快取的 CPU 分析，不推論、不搜尋、不改 BC 或 pipeline。',
          '- 範圍：olmoe_pair_gradient_diag 的 8 個 tap（int4、p6e1；F 條件）、olmoe_pair_generalization 的一般組 16 個與 spike 分層組 8 個（6 格式；單 tap 量化）。三組分開；全部 tap 保留；expert 依來源 metadata 對應；不混入 P／S／R。',
          f'- spike：s_t = max_c |h_tc|（快取的 FP32 calibration rows）；門檻 = {TAU} × 這些 rows 的 median；S／N；所有格式同一 mask。與舊實驗用 discovery median 的差異會列出比較。不依 g、誤差、loss 或排序改 mask；不掃其他倍數。',
          f'- 門檻非正／非有限 → 不能分組，沿用原 BC；空組 → 分組公式精確退化為原 BC；非空組 < {LOW_COVER} rows → 低覆蓋旗標、另列（不調權重、不挑結果）；不用 epsilon。',
          '- 公式：A V_old = Σ_c mean_t(g²)Σ_t r²；B V_group = Σ_c[mean_S(g²)Σ_S r² + mean_N(g²)Σ_N r²]；C V_pair = Σ_c Σ_t g²r²；共同 μ_c（全部 rows）、r = e − μ_c、不在組內重新中心化；U = Σ_c GS_c μ_c² 不變；float64；不加係數或一階項、不以 NLL 換算。',
          '- 恆等式（逐 channel）：V_group − V_old = (n_S n_N/n)(mean_S X − mean_N X)(mean_S Y − mean_N Y)；V_pair − V_group = n_S Cov_S + n_N Cov_N（母體共變異）；T_old／T_pair 須重現來源 mech.json。',
          f'- 排序：每篇文章（3 窗口）的兩格式 NLL 差；可判定 = 文章 bootstrap 95% CI 排除 0（{BOOT} 次，seed {SEEDS["bootstrap"]}），三種方法共用；主要 = 同位元 INT／Posit（mechanism_8 只有 int4 vs p6e1，作為其唯一配對報告）；其餘次要；proxy tie（相對差 < {BC_TIE}）單獨列出、不算正確；不可判定不算任何方法排對；三組分開、同一 expert 的配對不當獨立樣本。',
          '- 窗口穩定性：逐一移除 calibration 窗口；門檻固定為完整 calibration 的值，mask 限於剩餘 rows；μ、U 與三種分數由剩餘 rows 重算；比較排序翻轉與分數變動；不能證明跨文章穩定。',
          '- 停止規則：只改善單一已知案例、只勝過 T_pair 未勝過 T_old、或新增誤排抵消收益 → 建議停止此方向；即使有利也不接入 pipeline，確認需另行登記。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'timing.json', {'plan': {'seconds': time.perf_counter() - t0}})
    log(f'plan written; {len(src)} taps; {len(files)} source files hashed')


# --------------------------------------------------------------------------- #
def scores_for(h, g, tok, errs, thr, rows=None):
    """three scores and decomposition for one tap; rows: optional boolean row selection (leave-one-window-out)."""
    if rows is not None:
        h, g, tok = h[rows], g[rows], tok[rows]
        errs = {f: e[rows] for f, e in errs.items()}
    g = g.astype(np.float64)
    n = len(tok)
    X = g ** 2
    win = tok // SEQ
    wins = sorted(set(win.tolist()))
    GS = (np.stack([g[win == k].sum(0) for k in wins]) ** 2).sum(0)
    s = np.abs(h).max(1)
    valid = bool(np.isfinite(thr) and thr > 0)
    S = s >= thr if valid else np.zeros(n, bool)
    N_ = ~S
    nS, nN = int(S.sum()), int(N_.sum())
    mX = X.mean(0)
    mXS = X[S].mean(0) if nS else np.zeros(X.shape[1])
    mXN = X[N_].mean(0) if nN else np.zeros(X.shape[1])
    res = {}
    for f, e in errs.items():
        e = e.astype(np.float64)
        mu = e.mean(0)
        Y = (e - mu) ** 2
        U_c = GS * mu ** 2
        Vo_c = mX * Y.sum(0)
        Vg_c = (mXS * Y[S].sum(0) if nS else 0.0) + (mXN * Y[N_].sum(0) if nN else 0.0) if valid else Vo_c
        Vp_c = (X * Y).sum(0)
        res[f] = {'n': n, 'nS': nS, 'nN': nN, 'mu': mu, 'U_c': U_c, 'Vo_c': Vo_c, 'Vg_c': Vg_c, 'Vp_c': Vp_c, 'X': X, 'Y': Y, 'S': S,
                  'mXS': mXS, 'mXN': mXN, 'valid': valid}
    return res


def totals(r):
    U = float(r['U_c'].sum())
    return {'V_old': float(r['Vo_c'].sum()), 'V_group': float(r['Vg_c'].sum()), 'V_pair': float(r['Vp_c'].sum()), 'U': U,
            'T_old': float(r['Vo_c'].sum()) + U, 'T_group': float(r['Vg_c'].sum()) + U, 'T_pair': float(r['Vp_c'].sum()) + U}


def article_vec(ce):
    return np.array(ce).reshape(-1, WPA).sum(1) / (WPA * TP)


def analyze(out, log):
    if (out / 'scores.csv').exists():
        raise SystemExit('exists')
    pl = read(out / 'plan.json')
    if {str(p.relative_to(HERE)): sha(p) for p in source_files()} != pl['source_hashes']:
        raise SystemExit('source files changed since the plan')
    t0 = time.perf_counter()
    tm = defaultdict(float)
    src = sources()
    mech = {PGD: read(PGD / 'mech.json')['taps'], PGN: read(PGN / 'mech.json')}
    meas = {PGD: read(PGD / 'measure.json'), PGN: read(PGN / 'measure.json')}
    srows, drows, rrows, wrows = [], [], [], []
    ctl = {'reproduce_T_old_max_rel': 0.0, 'reproduce_T_pair_max_rel': 0.0, 'identity_between_max_rel': 0.0, 'identity_within_max_rel': 0.0,
           'empty_group_equals_old': True, 'invalid_threshold_taps': [], 'mask_vs_discovery': []}
    per_tap = {}
    for sp in src:
        t, d, fm = sp['tap'], sp['src'], sp['formats']
        z = np.load(d / 'work' / f'tap{t["id"]}.npz')
        h, g, tok = z['h'], z['g'], z['tok']
        errs = {f: z[f'e_{f}'] for f in fm}
        s = np.abs(h).max(1)
        thr = TAU * float(np.median(s))
        t1 = time.perf_counter()
        r = scores_for(h, g, tok, errs, thr)
        tm['scores'] += time.perf_counter() - t1
        disc_thr = t['threshold']
        ctl['mask_vs_discovery'].append({'group': sp['group'], 'node': t['node'], 'threshold_calibration': thr, 'threshold_discovery': disc_thr,
                                         'spike_rows_calibration_median': int((s >= thr).sum()), 'spike_rows_discovery_median': int((s >= disc_thr).sum()),
                                         'masks_equal': bool(np.array_equal(s >= thr, s >= disc_thr))})
        if not (np.isfinite(thr) and thr > 0):
            ctl['invalid_threshold_taps'].append(t['node'])
        tt = {}
        for f in fm:
            rf = r[f]
            T = totals(rf)
            tt[f] = T
            # reproduction of the source scores
            src_m = mech[d][t['node']]['formats'][f]
            src_m = src_m['conditions']['F'] if 'conditions' in src_m else src_m
            ctl['reproduce_T_old_max_rel'] = max(ctl['reproduce_T_old_max_rel'], abs(T['T_old'] - src_m['T_old']) / max(abs(src_m['T_old']), 1e-300))
            ctl['reproduce_T_pair_max_rel'] = max(ctl['reproduce_T_pair_max_rel'], abs(T['T_pair'] - src_m['T_pair']) / max(abs(src_m['T_pair']), 1e-300))
            # identities
            X, Y, S = rf['X'], rf['Y'], rf['S']
            nS, nN, n = rf['nS'], rf['nN'], rf['n']
            if rf['valid'] and nS and nN:
                between = nS * nN / n * (X[S].mean(0) - X[~S].mean(0)) * (Y[S].mean(0) - Y[~S].mean(0))
                within = ((X[S] - X[S].mean(0)) * (Y[S] - Y[S].mean(0))).sum(0) + ((X[~S] - X[~S].mean(0)) * (Y[~S] - Y[~S].mean(0))).sum(0)
            else:
                between = np.zeros(X.shape[1])
                within = ((X - X.mean(0)) * (Y - Y.mean(0))).sum(0)
                if not np.allclose(rf['Vg_c'], rf['Vo_c'], rtol=0, atol=0):
                    ctl['empty_group_equals_old'] = False
            scale = max(float(np.abs(rf['Vo_c']).max()), 1e-300)
            ctl['identity_between_max_rel'] = max(ctl['identity_between_max_rel'], float(np.abs((rf['Vg_c'] - rf['Vo_c']) - between).max()) / scale)
            ctl['identity_within_max_rel'] = max(ctl['identity_within_max_rel'], float(np.abs((rf['Vp_c'] - rf['Vg_c']) - within).max()) / scale)
            # decomposition details
            bt, wt = float(between.sum()), float(within.sum())
            cb, cw = int(np.argmax(np.abs(between))), int(np.argmax(np.abs(within)))
            rows_within = np.abs((X - np.where(S[:, None], X[S].mean(0) if nS else 0, X[~S].mean(0) if nN else 0)) *
                                 (Y - np.where(S[:, None], Y[S].mean(0) if nS else 0, Y[~S].mean(0) if nN else 0))).sum(1) if (nS and nN) else np.abs((X - X.mean(0)) * (Y - Y.mean(0))).sum(1)
            top = np.sort(rows_within)[::-1]
            lowcov = bool(rf['valid'] and ((0 < nS < LOW_COVER) or (0 < nN < LOW_COVER)))
            srows.append({'group': sp['group'], 'node': t['node'], 'tap_id': t['id'], 'role': t['role'], 'format': f, 'n': n, 'spike_rows': nS,
                          'spike_share': nS / n, 'threshold': thr, 'valid_grouping': rf['valid'], 'empty_group': bool(rf['valid'] and (nS == 0 or nN == 0)),
                          'low_coverage': lowcov, **T,
                          'g2_mean_S': float(X[S].mean()) if nS else None, 'g2_mean_N': float(X[~S].mean()) if nN else None,
                          'r2_mean_S': float(Y[S].mean()) if nS else None, 'r2_mean_N': float(Y[~S].mean()) if nN else None,
                          'g2_total_S': float(X[S].sum()), 'g2_total_N': float(X[~S].sum()), 'r2_total_S': float(Y[S].sum()), 'r2_total_N': float(Y[~S].sum())})
            drows.append({'group': sp['group'], 'node': t['node'], 'format': f, 'between_total': bt, 'within_total': wt,
                          'between_rel_to_V_old': bt / T['V_old'] if T['V_old'] else None, 'within_rel_to_V_old': wt / T['V_old'] if T['V_old'] else None,
                          'between_top_channel': cb, 'between_top_channel_share': float(abs(between[cb]) / np.abs(between).sum()) if np.abs(between).sum() > 0 else None,
                          'within_top_channel': cw, 'within_top_channel_share': float(abs(within[cw]) / np.abs(within).sum()) if np.abs(within).sum() > 0 else None,
                          'within_top1_row_share': float(top[0] / top.sum()) if top.sum() > 0 else None,
                          'within_top5_rows_share': float(top[:5].sum() / top.sum()) if top.sum() > 0 else None, 'spike_rows': nS, 'low_coverage': lowcov})
        per_tap[(sp['group'], t['node'])] = {'sp': sp, 'T': tt}
        # leave-one-window-out (threshold fixed)
        t1 = time.perf_counter()
        win = tok // SEQ
        for w in sorted(set(win.tolist())):
            keep = win != w
            if keep.sum() < 2:
                continue
            rw = scores_for(h, g, tok, errs, thr, rows=keep)
            Tw = {f: totals(rw[f]) for f in fm}
            for i, f1 in enumerate(fm):
                for f2 in fm[i + 1:]:
                    row = {'group': sp['group'], 'node': t['node'], 'window_removed': int(w), 'f1': f1, 'f2': f2}
                    for mth in ('T_old', 'T_group', 'T_pair'):
                        full = tt[f1][mth] - tt[f2][mth]
                        lo = Tw[f1][mth] - Tw[f2][mth]
                        row[f'{mth}_flip'] = bool((full < 0) != (lo < 0))
                    wrows.append(row)
            for f in fm:
                wrows.append({'group': sp['group'], 'node': t['node'], 'window_removed': int(w), 'f1': f, 'f2': '', **{f'{m}_rel_change': abs(Tw[f][m] - tt[f][m]) / max(abs(tt[f][m]), 1e-300)
                                                                                                             for m in ('T_old', 'T_group', 'T_pair')}})
        tm['window_stability'] += time.perf_counter() - t1
        log(f'{sp["group"]} {t["node"]}: done')
    # rankings
    t1 = time.perf_counter()
    for (grp, node), v in per_tap.items():
        sp, tt = v['sp'], v['T']
        t, d, fm = sp['tap'], sp['src'], sp['formats']
        m = meas[d]
        z0 = article_vec(m['base_ce'])
        na = len(z0)
        idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, na, (BOOT, na))
        D = {f: article_vec(m['variants'][sp['key'](t['id'], f)]) - z0 for f in fm}
        for i, f1 in enumerate(fm):
            for f2 in fm[i + 1:]:
                dd = D[f1] - D[f2]
                b = dd[idx].mean(1)
                ci = np.percentile(b, [2.5, 97.5])
                dec = bool(ci[0] > 0 or ci[1] < 0)
                row = {'group': grp, 'node': node, 'f1': f1, 'f2': f2, 'same_bit': (f1, f2) in SAME_BIT, 'primary': (f1, f2) in SAME_BIT or grp == 'mechanism_8',
                       'measured_diff': float(dd.mean()), 'ci95_lo': float(ci[0]), 'ci95_hi': float(ci[1]), 'decisive': dec}
                for mth in ('T_old', 'T_group', 'T_pair'):
                    dT = tt[f1][mth] - tt[f2][mth]
                    mx = max(tt[f1][mth], tt[f2][mth])
                    tie = bool(mx <= 0 or abs(dT) / mx < BC_TIE)
                    row[f'{mth}_diff'] = dT
                    row[f'{mth}_tie'] = tie
                    row[f'{mth}_right'] = bool(dec and not tie and ((dT < 0) == (dd.mean() < 0)))
                rrows.append(row)
    tm['rankings'] = time.perf_counter() - t1
    summ = {}
    for grp in ('mechanism_8', 'generalization_general', 'generalization_spike'):
        for scope in ('primary', 'secondary', 'all'):
            rs = [r for r in rrows if r['group'] == grp and (scope == 'all' or (scope == 'primary') == r['primary'])]
            dec = [r for r in rs if r['decisive']]
            def cls(a, b):
                c = Counter()
                ex = defaultdict(set)
                for r in dec:
                    ra, rb = r[f'{a}_right'], r[f'{b}_right']
                    k = 'fix' if ra and not rb else 'new_error' if rb and not ra else 'both_right' if ra else 'both_not'
                    c[k] += 1
                    ex[k].add(r['node'])
                return {**dict(c), 'experts_fix': sorted(ex['fix']), 'experts_new_error': sorted(ex['new_error'])}
            summ[f'{grp}|{scope}'] = {'pairs': len(rs), 'decisive': len(dec), 'undeterminable': len(rs) - len(dec),
                                      **{f'{m}_right': sum(r[f'{m}_right'] for r in dec) for m in ('T_old', 'T_group', 'T_pair')},
                                      **{f'{m}_ties_on_decisive': sum(r[f'{m}_tie'] for r in dec) for m in ('T_old', 'T_group', 'T_pair')},
                                      'group_vs_old': cls('T_group', 'T_old'), 'group_vs_pair': cls('T_group', 'T_pair'), 'pair_vs_old': cls('T_pair', 'T_old')}
    # T_pair fixes (vs T_old) kept by T_group?
    pf = [r for r in rrows if r['decisive'] and r['T_pair_right'] and not r['T_old_right']]
    summ['pair_fixes_kept_by_group'] = {'pair_fixes': [(r['group'], r['node'], r['f1'], r['f2']) for r in pf], 'kept': [(r['group'], r['node'], r['f1'], r['f2']) for r in pf if r['T_group_right']]}
    # window stability summary
    ws = {}
    for grp in ('mechanism_8', 'generalization_general', 'generalization_spike'):
        pr_ = [r for r in wrows if r['group'] == grp and r['f2']]
        sc_ = [r for r in wrows if r['group'] == grp and not r['f2']]
        ws[grp] = {'pair_window_cases': len(pr_), **{f'{m}_flips': sum(r[f'{m}_flip'] for r in pr_) for m in ('T_old', 'T_group', 'T_pair')},
                   **{f'{m}_rel_change_median_max': [float(np.median([r[f'{m}_rel_change'] for r in sc_])), float(max(r[f'{m}_rel_change'] for r in sc_))]
                      for m in ('T_old', 'T_group', 'T_pair')},
                   'nodes_with_flips': {m: sorted({r['node'] for r in pr_ if r[f'{m}_flip']}) for m in ('T_old', 'T_group', 'T_pair')}}
    ctl['pass'] = bool(ctl['reproduce_T_old_max_rel'] < 1e-10 and ctl['reproduce_T_pair_max_rel'] < 1e-10 and ctl['identity_between_max_rel'] < 1e-9
                       and ctl['identity_within_max_rel'] < 1e-9 and ctl['empty_group_equals_old'])
    ctl['mask_summary'] = {'taps': len(ctl['mask_vs_discovery']), 'masks_equal': sum(m['masks_equal'] for m in ctl['mask_vs_discovery'])}
    for name, rows in (('scores.csv', srows), ('decomposition.csv', drows), ('ranking_comparisons.csv', rrows), ('window_stability.csv', wrows)):
        with open(out / name, 'w') as fh:
            keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
            wcsv = csv.DictWriter(fh, fieldnames=keys); wcsv.writeheader(); wcsv.writerows(rows)
    write(out / 'controls.json', ctl)
    write(out / 'summary.json', {'ranking': summ, 'window_stability': ws})
    # extra statistics size (estimate per expert Down input): grouping needs per channel sum_S g^2 (format-independent) and per format
    # sum_S e, sum_S e^2 restricted to S; pairing needs per format sum g^2 e^2 and sum g^2 e
    stats = {'group_extra_arrays_per_expert': '1 + 2 x formats per channel (sum_S g^2; sum_S e, sum_S e^2 per format) plus n_S',
             'pair_extra_arrays_per_expert': '2 x formats per channel (sum g^2 e^2, sum g^2 e)', 'channels': 1024,
             'bytes_float64_10_formats': {'group': (1 + 20) * 1024 * 8, 'pair': 20 * 1024 * 8}, 'note': 'estimates; not a native build cost'}
    tmj = read(out / 'timing.json')
    tmj['analyze'] = {'seconds_total': time.perf_counter() - t0, **dict(tm), 'stats_size': stats}
    write(out / 'timing.json', tmj)
    log(json.dumps(dr.clean({'controls': {k: ctl[k] for k in ('reproduce_T_old_max_rel', 'reproduce_T_pair_max_rel', 'identity_between_max_rel', 'identity_within_max_rel', 'empty_group_equals_old', 'pass')} | ctl['mask_summary'],
                             'ranking': {k: {kk: v[kk] for kk in ('decisive', 'T_old_right', 'T_group_right', 'T_pair_right')} for k, v in summ.items() if '|' in k},
                             'window': {g_: {k: v for k, v in w.items() if k.endswith('flips')} for g_, w in ws.items()}})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(Path(__file__))}) + '\n')


if __name__ == '__main__':
    main()
