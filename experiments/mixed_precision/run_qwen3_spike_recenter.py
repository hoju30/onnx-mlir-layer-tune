#!/usr/bin/env python3
"""Qwen3-0.6B Down input (h = SiLU(gate) * up), all 28 layers, one layer quantized at a time: does estimating the
ordinary-token mean error mu_N without the spike rows, and scoring the spike rows with their own gradients, improve the
same-bit INT / Posit ranking of the full Down-input QDQ? Spec: reports/qwen3_spike_recenter_prompt_20261026.md.
No change to the formal criterion, quantizers, search or native pipeline.

Rows (s, p) of the original 10 x 128 calibration, channels c, g = d(summed CE of window s)/dh (FP32, position 127 = 0).
Mask (all formats, from FP32 h): a_sp = max_c |h_spc|, threshold = 6 median(a), S = {a >= threshold}, N = the rest.
e = Q_f(h) - h, mu_all = mean_all(e), mu_N = mean_N(e). For a non-empty row set G and a channel vector m:
  V_G(m) = sum_c mean_G(g^2) sum_G (e - m)^2,  U_G(m) = sum_(s,c) m^2 (sum_(p in G) g)^2,  P_G(m) = sum_G g^2 (e - m)^2
  A   = V_all(mu_all) + U_all(mu_all)                     original BC
  P   = P_all(mu_all) + U_all(mu_all)                     old element-wise pairing
  G00 = V_N(mu_all) + V_S(mu_all) + U_all(mu_all)         old spike grouping
  G01 = V_N(mu_all) + P_S(mu_all) + U_all(mu_all)
  G10 = V_N(mu_N) + U_N(mu_N) + V_S(0)
  D   = V_N(mu_N) + U_N(mu_N) + P_S(0)                    the single pre-specified primary candidate
Fallback (S empty, N empty, invalid threshold): G00 / G01 / G10 / D = A (same float), P unchanged.
Truth: per-article NLL change of the full Q_f(h) of one layer on 128 new WikiText-103 articles x 3 windows (GPU replay,
the evaluator path of run_qwen_down_spike_diag). Gate: if D and A pick the same format in all 84 same-bit pairs, stop
after the cache analysis (no new articles).

Stages (one new --out-dir): plan, scores, old_replay, audit, data, eval, analyze.
CPU: ../../venv/bin/python; eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (sampler, read / write / sha / now; standard imports only)

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE, PREP = prs.NATIVE, prs.PREP
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
VDIAG = HERE / 'runs/qwen3_posit_variance_diagnostic_20261025'
FORMATS = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')
SAME_BIT = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
METHODS = ('A', 'P', 'G00', 'G01', 'G10', 'D')
SECONDARY = (('D', 'G10'), ('G01', 'G00'), ('G10', 'G00'), ('D', 'G01'), ('G00', 'A'), ('P', 'A'))
NL, SEQ, TP, CAL, C = 28, 128, 127, 10, 3072
N = CAL * SEQ
TAU, LOW_COVER, PROXY_TIE, BOOT, ALPHA = 6.0, 8, 0.02, 10000, 0.05
N_ART, WPA = 128, 3
SEEDS = {'order': 26102601, 'start': 26102602, 'bootstrap': 26102603}
OLD_BOOT_SEED = 27103
CHUNK, LM_SUB, THREADS = 32, 16, 12
TIMING_REPEATS = 5
TOL = {'old_score_rel': 1e-9, 'identity_rel': 1e-9, 'spot_h_rel': 1e-6, 'spot_g_rel': 1e-4, 'sumN_rel': 1e-9,
       'replay_identity_abs_summed_ce': 1e-6, 'gpu_qdq': 'bitwise (sha of float32 errors equal the CPU errors)'}
CODE = ['run_qwen3_spike_recenter.py', 'run_qwen_down_spike_diag.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_down_fp16_protection.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'wt_manifest_scan.py', 'run_olmoe_route_mass_restore.py',
        'run_olmoe_dynamic_route.py', 'qwen3_prepare.py']
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def log_stage(out, stage, seconds, extra=None):
    p = out / 'timing.json'
    d = read(p) if p.exists() else {}
    d[stage] = {'seconds': seconds, **(extra or {})}
    write(p, d)


def input_files():
    import run_olmoe_route_mass_restore as mr
    return ([HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] +
            [NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', PREP / 'node_mapping.json', PREP / 'validation_tokens.npy',
             PREP / 'test_tokens.npy', SPIKE / 'plan.json', SPIKE / 'freeze.json', SPIKE / 'scores.csv', SPIKE / 'mechanism.json',
             SPIKE / 'per_article_losses.csv', SPIKE / 'ranking_pairs.csv', SPIKE / 'controls_calib.json', SPIKE / 'report.md',
             VDIAG / 'report.md', HERE / 'reports/qwen3_spike_recenter_prompt_20261026.md', HERE / 'qwen35_assets/model/tokenizer.json']
            + [SPIKE / 'work' / f'{k}_L{l}.npy' for l in range(NL) for k in ('h', 'g')] + mr.WT103 + mr.WT2)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    if inputs:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def down_nodes():
    m = read(PREP / 'node_mapping.json')
    return {int(p.split('.')[2]): n for n, p in m.items() if p.split('.')[3] == 'mlp' and p.split('.')[4] == 'down_proj'}


# --------------------------------------------------------------------------- #
# scores (float64, numpy)
# --------------------------------------------------------------------------- #
def components(h, g, errs, thr, win, keep=None):
    """the ten components and six scores per format for one layer; rows = calibration (window, position)."""
    if keep is not None:
        h, g, win = h[keep], g[keep], win[keep]
        errs = {f: e[keep] for f, e in errs.items()}
    n = len(win)
    wins = np.unique(win)
    gd = g.astype(np.float64)
    X = gd ** 2
    a = np.abs(h).max(1)
    valid = bool(np.isfinite(thr) and thr > 0)
    S = (a >= thr) if valid else np.zeros(n, bool)
    nS = int(S.sum())
    nN = n - nS
    fallback = None if valid and nS > 0 and nN > 0 else ('invalid threshold' if not valid else 'S empty' if nS == 0 else 'N empty')
    mX = X.mean(0)
    GS = sum(gd[win == k].sum(0) ** 2 for k in wins)
    if fallback is None:
        XS = X[S]
        mXS, mXN = XS.mean(0), X[~S].mean(0)
        GSN = sum(gd[(win == k) & ~S].sum(0) ** 2 for k in wins)
    res = {}
    for f, e in errs.items():
        ed = e.astype(np.float64)
        mu = ed.mean(0)
        Y = (ed - mu) ** 2
        c = {'V_all(mu_all)': float((mX * Y.sum(0)).sum()), 'U_all(mu_all)': float((GS * mu ** 2).sum()), 'P_all(mu_all)': float((X * Y).sum(0).sum())}
        r = {'n': n, 'nS': nS, 'nN': nN, 'fallback': fallback, 'low_coverage': bool(0 < nS < LOW_COVER or 0 < nN < LOW_COVER), 'mu_all': mu}
        A = c['V_all(mu_all)'] + c['U_all(mu_all)']
        sc = {'A': A, 'P': c['P_all(mu_all)'] + c['U_all(mu_all)']}
        if fallback is None:
            YS = Y[S]
            eN = ed[~S]
            muN = eN.mean(0)
            eS2 = ed[S] ** 2
            c.update({'V_N(mu_all)': float((mXN * Y[~S].sum(0)).sum()), 'V_S(mu_all)': float((mXS * YS.sum(0)).sum()),
                      'P_S(mu_all)': float((XS * YS).sum()), 'V_N(mu_N)': float((mXN * ((eN - muN) ** 2).sum(0)).sum()),
                      'U_N(mu_N)': float((GSN * muN ** 2).sum()), 'V_S(0)': float((mXS * eS2.sum(0)).sum()), 'P_S(0)': float((XS * eS2).sum())})
            sc['G00'] = c['V_N(mu_all)'] + c['V_S(mu_all)'] + c['U_all(mu_all)']
            sc['G01'] = c['V_N(mu_all)'] + c['P_S(mu_all)'] + c['U_all(mu_all)']
            sc['G10'] = c['V_N(mu_N)'] + c['U_N(mu_N)'] + c['V_S(0)']
            sc['D'] = c['V_N(mu_N)'] + c['U_N(mu_N)'] + c['P_S(0)']
            # G01 - G00 = P_S(mu_all) - V_S(mu_all) and G11 - G10 = P_S(0) - V_S(0): checked at the component level against the
            # covariance forms (score-level rounding of the large totals reported separately)
            id1, id2 = float(((XS - mXS) * YS).sum()), float(((XS - mXS) * eS2).sum())
            sc1, sc2 = abs(c['V_S(mu_all)']) + abs(c['P_S(mu_all)']), abs(c['V_S(0)']) + abs(c['P_S(0)'])
            r['identity_rel'] = max(abs((c['P_S(mu_all)'] - c['V_S(mu_all)']) - id1) / max(sc1, 1e-300),
                                    abs((c['P_S(0)'] - c['V_S(0)']) - id2) / max(sc2, 1e-300))
            r['score_level_identity_rounding'] = max(abs((sc['G01'] - sc['G00']) - id1) / max(abs(sc['G01']), abs(sc['G00']), 1e-300),
                                                     abs((sc['D'] - sc['G10']) - id2) / max(abs(sc['D']), abs(sc['G10']), 1e-300))
            r['sumN_e_minus_muN_rel'] = float(np.abs((eN - muN).sum(0)).max() / max(np.abs(eN).sum(0).max(), 1e-300))
            r['mu_N'] = muN
        else:
            for k in ('G00', 'G01', 'G10', 'D'):
                sc[k] = A
            r['mu_N'] = None
        r['components'], r['scores'] = c, sc
        res[f] = r
    return res, S


def pick_tie(s1, s2):
    """smaller raw score; exactly equal -> the INT (first) format; proxy tie: relative difference < 2% (all-zero exact tie)."""
    d = s1 - s2
    if s1 == 0.0 and s2 == 0.0:
        return d, True, 0
    return d, bool(abs(d) / max(abs(s1), abs(s2)) < PROXY_TIE), (0 if d <= 0 else 1)


def kernel_A(ed, gd, win):
    X = gd ** 2
    mX = X.mean(0)
    GS = sum(gd[win == k].sum(0) ** 2 for k in range(CAL))
    out = {}
    for f, e in ed.items():
        mu = e.mean(0)
        out[f] = float((mX * ((e - mu) ** 2).sum(0)).sum()) + float((GS * mu ** 2).sum())
    return out


def kernel_D(h, ed, gd, win):
    a = np.abs(h).max(1)
    S = a >= TAU * float(np.median(a))
    X = gd ** 2
    XN, XS = X[~S], X[S]
    mXN = XN.mean(0)
    GSN = sum(gd[(win == k) & ~S].sum(0) ** 2 for k in range(CAL))
    out = {}
    for f, e in ed.items():
        eN = e[~S]
        muN = eN.mean(0)
        out[f] = float((mXN * ((eN - muN) ** 2).sum(0)).sum()) + float((GSN * muN ** 2).sum()) + float((XS * e[S] ** 2).sum())
    return out


def synthetic_checks():
    rng = np.random.default_rng(7)
    n, Cs = 4 * 16, 8
    win = np.repeat(np.arange(4), 16)
    h = rng.standard_normal((n, Cs)).astype(np.float32)
    h[[0, 16, 33]] *= 50.0
    g = rng.standard_normal((n, Cs)).astype(np.float32)
    e = rng.standard_normal((n, Cs)).astype(np.float32)
    thr = TAU * float(np.median(np.abs(h).max(1)))
    out = {}
    z, _ = components(h, g, {'z': np.zeros_like(e)}, thr, win)
    out['zero_error_all_scores_zero'] = all(v == 0.0 for v in z['z']['scores'].values())
    for name, t in (('invalid_threshold_inf', np.inf), ('invalid_threshold_zero', 0.0), ('S_empty', 1e30), ('N_empty', 1e-30)):
        r, _ = components(h, g, {'e': e}, t, win)
        s = r['e']['scores']
        out[name] = {'fallback': r['e']['fallback'], 'equal_A': all(s[k] is s['A'] for k in ('G00', 'G01', 'G10', 'D')),
                     'P_old_formula': s['P'] == r['e']['components']['P_all(mu_all)'] + r['e']['components']['U_all(mu_all)']}
    r, S = components(h, g, {'e': e}, thr, win)
    v = r['e']
    s, c = v['scores'], v['components']
    out['spike_rows'] = v['nS']
    out['identity_rel'] = v['identity_rel']
    out['score_level_identity_rounding'] = v['score_level_identity_rounding']
    out['passed'] = bool(out['zero_error_all_scores_zero'] and v['nS'] == 3 and out['identity_rel'] < 1e-12
                         and all(out[k]['equal_A'] and out[k]['P_old_formula'] for k in ('invalid_threshold_inf', 'invalid_threshold_zero', 'S_empty', 'N_empty'))
                         and [out[k]['fallback'] for k in ('invalid_threshold_inf', 'invalid_threshold_zero', 'S_empty', 'N_empty')]
                         == ['invalid threshold', 'invalid threshold', 'S empty', 'N empty'])
    return out


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    cfg = read(HERE / 'qwen3_assets/model/config.json')
    if cfg['num_hidden_layers'] != NL or sorted(down_nodes()) != list(range(NL)):
        raise SystemExit('layer count mismatch')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seeds_seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seeds_seen:
        raise SystemExit(f'seeds already used: {seeds_seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'spec': 'reports/qwen3_spike_recenter_prompt_20261026.md and the request of 2026-10-26',
        'question': 'does estimating the ordinary-token mean error mu_N without the spike rows (its own mean / variance terms) and scoring '
                    'the spike rows with their own gradients (uncentred sum g^2 e^2) improve the same-bit ranking of the full Down-input '
                    'QDQ, at low table-building cost?',
        'not_first': 'spike grouping (G00 = old T_group) and element-wise pairing (P = old T_pair) were tested in runs/qwen3_down_spike_diag_20261023 '
                     '(74/78 like A); this round changes the decomposition (mu_N for N rows, U_N with N-row gradient sums) and the spike scoring',
        'scope': f'Qwen3-0.6B, all {NL} Down inputs h = SiLU(gate) * up (config / mapping checked), one layer at a time; weights, other '
                 f'activations and the Down output FP32; formats {list(FORMATS)} + FP32 identity; original 10 x 128 calibration, native INT '
                 'range / scale / zero point, Universal posit LUT, the original loss; no change to the formal criterion, quantizers, search or native',
        'mask': {'rule': f'a_sp = max_c |h_spc| (FP32 calibration h), threshold = {TAU} x numpy median over the 1280 rows, S = a >= threshold '
                         '(old semantics), N = the rest; one mask per layer for all formats; no threshold scan, no hard-coded position 0, no use of '
                         'g, e, format or diagnostic loss', 'check': 'threshold and spike row count / positions equal the old mechanism.json',
                 'low_coverage': f'flag a non-empty group with fewer than {LOW_COVER} rows (formula unchanged)'},
        'formulas': {'V_G(m)': 'sum_c mean_G(g_c^2) sum_(s,p in G) (e - m)^2 (mean over the rows of G, position-127 rows included)',
                     'U_G(m)': 'sum_(s,c) m_c^2 [sum_(p: (s,p) in G) g]^2 (N-row gradient sums only for U_N)',
                     'P_G(m)': 'sum_((s,p) in G, c) g^2 (e - m)^2',
                     'methods': {'A': 'V_all(mu_all) + U_all(mu_all) (original BC)', 'P': 'P_all(mu_all) + U_all(mu_all) (old pairing)',
                                 'G00': 'V_N(mu_all) + V_S(mu_all) + U_all(mu_all) (old spike grouping)', 'G01': 'V_N(mu_all) + P_S(mu_all) + U_all(mu_all)',
                                 'G10': 'V_N(mu_N) + U_N(mu_N) + V_S(0)', 'D': 'V_N(mu_N) + U_N(mu_N) + P_S(0) (the single primary candidate)'},
                     'fallback': 'S empty, N empty or a non-positive / non-finite threshold: G00 / G01 / G10 / D = A (the same float) with the '
                                 'reason recorded; P keeps its old formula; no epsilon in denominators',
                     'numerics': 'float64 sums; no first-order term, HVP, FD, fitted coefficient or background update; scores are uncalibrated '
                                 'proxies (raw; any /1270 display is still a proxy)'},
        'cache': {'source': 'runs/qwen3_down_spike_diag_20261023/work/h_L*.npy, g_L*.npy (CPU FP32 forward / backward of the original '
                            'calibration)', 'checks': 'input hashes of the old plan equal today\'s files; float32 errors from the cached h equal '
                            'the old freeze error sha256; A / P / G00 reproduce the old T_old / T_pair / T_group (raw, U included, not /1270) '
                            f'within {TOL["old_score_rel"]} rel; a spot re-run of window 0 (CPU forward / backward, old code path) equals the cache '
                            f'(h within {TOL["spot_h_rel"]}, g within {TOL["spot_g_rel"]} rel); a failure means one fresh forward / backward of '
                            'the original calibration (never a refreshed or diagnostic gradient)'},
        'controls': ['zero error -> all six scores 0', f'N-row sum of (e - mu_N) ~ 0 within {TOL["sumN_rel"]} rel',
                     'fallbacks reproduce A bitwise (synthetic)', f'score identities G01 - G00 = sum_(S,c) [g^2 - mean_S(g^2)] r_all^2 and G11 - G10 = '
                     f'sum_(S,c) [g^2 - mean_S(g^2)] e^2 within {TOL["identity_rel"]} (relative to the S-group terms; score identities, not loss '
                     'interactions)', 'no "constant gradient => equal scores" control (different decompositions have different U)',
                     'posit LUT vs Universal; INT scale / zp on the tensor device; GPU QDQ equals the CPU QDQ bitwise on the calibration h',
                     'TF32 off, deterministic, FP32, eager attention, RMSNorm epsilon 1e-6', 'identity replay equals the no-hook forward within '
                     f'{TOL["replay_identity_abs_summed_ce"]} summed CE; only the Down input replaced; position-127 gradient 0',
                     'control failures fixed with an amendment; non-finite values and failures kept'],
        'gate': 'if D and A pick the same format in all 84 same-bit pairs: finish the cache analysis and report, stop (no articles, no '
                'evaluation; the single-point selection difference is 0 by definition; cross-layer allocation, cross-bit pairs and full '
                'configurations are not implied); otherwise run the new-article validation below',
        'old_replay': 'exploratory: the old per_article_losses (already seen, 128 articles of 20261023, bootstrap seed 27103) replayed for the six '
                      'methods after the freeze; kept apart from the new data; never used to choose a method, threshold or D',
        'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data (verified manifests, historical Qwen '
                 'token audit, WT2 / near-duplicate / previous-article rules, reject any window with >= 50% WT2 word 5-gram overlap), Qwen3 '
                 'tokenizer', 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS,
                 'seed_check': 'none of the seeds appears in an earlier plan.json', 'when': 'only after scores, predictions, code hashes and '
                 'plan are frozen and only if the gate says so'},
        'evaluation': {'conditions': f'{NL} layers x (FP32 identity + 6 formats), one layer quantized', 'path': 'the evaluator of '
                       'run_qwen_down_spike_diag (GPU, chunk 32, LM-head sub-batch 16, identity replay first)', 'article': f'summed CE of the {WPA} '
                       f'windows / {WPA * TP}; dNLL = quantized - same-path identity (sign kept)', 'no_recalibration': 'mu, g, range and mask never '
                       're-estimated on new text'},
        'ranking': {'pairs': '84 = 28 layers x (int4/p4e1, int6/p6e1, int8/p8e1)', 'pick': 'smaller raw score; exactly equal -> INT',
                    'tie': f'relative difference < {PROXY_TIE} = proxy tie (pick unchanged, never counted right); both zero = exact tie',
                    'decisive': f'article bootstrap {BOOT} (seed {SEEDS["bootstrap"]}, one resample for everything) 95% CI of INT - Posit dNLL '
                                'excludes 0; otherwise undetermined; the same decisive set for every method',
                    'confirmed': 'simultaneous 95% CI over the 84 pairs (max-|t| bootstrap, as the old analysis)',
                    'report': 'all 84 pairs, fixes, new errors, proxy ties, changed layers / bits, L2 and L27 4 / 6 / 8-bit one by one'},
        'primary': {'G': 'G(article) = R_D - R_A, R_M = mean over the 84 pairs of the article dNLL of the format chosen by M', 'ci': 'two-sided 95% '
                    'bootstrap', 'note': 'an average of single-layer conditions, not a 28-layer quantized model'},
        'secondary': {'family': [f'{a}-{b}' for a, b in SECONDARY], 'test': 'two-sided bootstrap p = 2 min(P(b >= 0), P(b <= 0)), Holm over the family',
                      'note': 'mechanism analysis only; never replaces D'},
        'lowo': 'leave-one-calibration-window-out: threshold of the full calibration fixed; mu, gradient sums and scores recomputed on the remaining '
                'rows; flips of each method over the 10 windows (stability description, not new-text validation)',
        'candidate_rule': ['1. the upper end of the 95% CI of G (D - A) < 0',
                           '2. on the simultaneously confirmed pairs D has more fixes than new errors vs A, and at least one of L2 int6-p6e1 / int8-p8e1 '
                           'is a confirmed fix', '3. if the new data decide L2 int4-p4e1 (raw 95% CI), D picks it correctly',
                           '4. the confirmed L2 fix keeps its choice in at least 8 of the 10 leave-one-window-out runs',
                           'all four -> only supports a separate full-configuration and native-cost study; otherwise keep the original BC and '
                           'report no gain / unstable / new errors / insufficient evidence; no threshold search, other coefficient or extra data'],
        'cost': ['cache read', 'spot forward / backward (provenance)', 'QDQ / e', 'mask', 'A and D statistic kernels (warm-up, '
                 f'{TIMING_REPEATS} repeats, median and range; shared and incremental costs apart)', 'extra statistics memory', 'LOO',
                 'exploratory replay', 'new-article evaluation', 'peak RAM / VRAM', 'Python cache replay; no native-latency or saved-pass claim'],
        'tolerances': TOL, 'smoke': smoke or 'none',
        'boundaries': 'stop afterwards: no change to the formal BC, quantizer scale, search, knee or native lowering; no fine-tuning, HVP / FD or '
                      'other model',
        'inputs': {prs.rel(p): sha(p) for p in input_files()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：Qwen3 Down input 的分組平均誤差與 spike 同位置評分', '', f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。', '',
          f'- 範圍：Qwen3-0.6B 全部 {NL} 層 Down input h，一次一層；權重、其他 activation、Down output 全 FP32；{list(FORMATS)} + identity；原 calibration、range、LUT、loss。',
          f'- 遮罩：a = max_c|h|，門檻 {TAU}×median(a)，S = a ≥ 門檻，N = 其餘；所有格式共用；與舊 mechanism.json 核對；組 < {LOW_COVER} 列標低覆蓋。',
          '- 公式：V_G(m) = Σ_c mean_G(g²)Σ_G(e−m)²；U_G(m) = Σ_(s,c) m²[Σ_(p∈G) g]²；P_G(m) = Σ_G g²(e−m)²。',
          '- 方法：A = V_all(μ_all)+U_all（原 BC）；P = P_all(μ_all)+U_all（舊配對）；G00 = V_N(μ_all)+V_S(μ_all)+U_all（舊分組）；G01 = V_N(μ_all)+P_S(μ_all)+U_all；G10 = V_N(μ_N)+U_N(μ_N)+V_S(0)；**D = V_N(μ_N)+U_N(μ_N)+P_S(0)（唯一主要候選）**。回退：S 或 N 空、門檻無效 → G00／G01／G10／D = A。',
          '- 快取：舊 down spike diag 的 h、g；核對輸入 hash、誤差 sha、舊分數（A／P／G00 = T_old／T_pair／T_group，1e-9 相對）與窗口 0 的重算。',
          '- gate：D 與 A 在 84 對的選擇完全相同 → 完成快取分析與報告後停止；否則抽 128 篇新 WT103 × 3 窗口，驗證 28 層。',
          f'- 排序：較小分數；相等選 INT；相對差 < {PROXY_TIE} 為 proxy tie；文章 bootstrap {BOOT}（seed {SEEDS["bootstrap"]}）95% CI 排除 0 才可判定；simultaneous 95% CI 判確認。',
          '- 主要終點：G = R_D − R_A（每篇 84 對所選格式的平均 ΔNLL），95% CI。次要家族（Holm）：' + '、'.join(f'{a}−{b}' for a, b in SECONDARY) + '。',
          '- 候選資格：G 上界 < 0；確認修正 > 確認新增誤排且 L2 6／8-bit 至少一對確認修正；L2 4-bit 若可判定須排對；該 L2 修正 LOO ≥ 8/10 維持。',
          '- 舊資料重播：探索性，凍結後才做，與新資料分開，不用來選方法。',
          f'- seeds：{SEEDS}（未出現在任何舊 plan）；排除 {len(used)} 篇已用文章；合格 {len(elig)} 篇。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; used {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# scores (CPU, from the cache)
# --------------------------------------------------------------------------- #
def spot_check(log_):
    """re-run window 0 with the old calibration code path (CPU, 12 threads) and compare h / g with the cache."""
    import torch
    import qwen3_swiglu_common as qc
    from run_qwen_down_spike_diag import Model
    torch.set_num_threads(THREADS)
    toks = np.load(NATIVE / 'calib_inputs.npy')
    M = Model('cpu')
    m = M.model
    m.model.embed_tokens.weight.requires_grad_(True)
    cap = {}

    def down_hook(l):
        def hook(mod, inp, outp):
            inp[0].retain_grad()
            cap[l] = inp[0]
        return hook
    hooks = [M.layers[l].mlp.down_proj.register_forward_hook(down_hook(l)) for l in range(NL)]
    ids = torch.from_numpy(toks[:1])
    loss = qc.summed_ce(M.wrapper(ids), ids[0])
    loss.backward()
    res = {}
    for l in range(NL):
        hc = np.load(SPIKE / 'work' / f'h_L{l}.npy', mmap_mode='r')[:SEQ]
        gc = np.load(SPIKE / 'work' / f'g_L{l}.npy', mmap_mode='r')[:SEQ]
        h, g = cap[l][0].detach().numpy(), cap[l].grad[0].numpy()
        res[l] = {'h_rel': float(np.abs(h - hc).max() / np.abs(hc).max()), 'g_rel': float(np.abs(g - gc).max() / np.abs(gc).max()),
                  'h_bitwise': bool(np.array_equal(h, hc)), 'g_bitwise': bool(np.array_equal(g, gc))}
    for hk in hooks:
        hk.remove()
    return {'rms_norm_eps': [float(M.layers[0].post_attention_layernorm.variance_epsilon), float(m.model.norm.variance_epsilon)],
            'attn_implementation': m.config._attn_implementation, 'window0': res}


def scores(out):
    import torch
    from run_qwen_down_spike_diag import Quant
    t_all = time.perf_counter()
    if (out / 'freeze.json').exists():
        raise SystemExit('freeze exists')
    pl = check_plan(out)
    torch.set_num_threads(THREADS)
    tm = defaultdict(float)
    dn = down_nodes()
    ranges = read(NATIVE / 'ranges_by_point.json')
    old_plan, old_fz = read(SPIKE / 'plan.json'), read(SPIKE / 'freeze.json')
    old_sc = {(int(r['layer']), r['format']): r for r in csv.DictReader(open(SPIKE / 'scores.csv'))}
    old_mech = read(SPIKE / 'mechanism.json')
    ctl = {'synthetic': synthetic_checks()}
    ctl['old_plan_inputs_unchanged'] = {k: sha(HERE / k) == v for k, v in old_plan['inputs'].items()}
    ctl['old_plan_inputs_all_unchanged'] = all(ctl['old_plan_inputs_unchanged'].values())
    t0 = time.perf_counter()
    ctl['spot_check'] = spot_check(log)
    tm['spot_forward_backward_s'] = time.perf_counter() - t0
    sp = ctl['spot_check']['window0']
    Q = Quant(out / 'work', 'cpu')
    ctl['posit_lut_mismatches'] = Q.verify()
    win = np.repeat(np.arange(CAL), SEQ)
    pos = np.tile(np.arange(SEQ), CAL)
    comp_rows, score_rows, lowo_rows, mus, masks, shas = [], [], [], {}, {}, {}
    rep = {'A_vs_T_old': 0.0, 'P_vs_T_pair': 0.0, 'G00_vs_T_group': 0.0}
    idmax, score_round, sumN_max, zero_ok, mask_ok = 0.0, 0.0, 0.0, True, {}
    kt = {'A': [], 'D': []}
    pk = {}
    for l in range(NL):
        t0 = time.perf_counter()
        h = np.load(SPIKE / 'work' / f'h_L{l}.npy')
        g = np.load(SPIKE / 'work' / f'g_L{l}.npy')
        tm['cache_read_s'] += time.perf_counter() - t0
        rx = ranges[dn[l]]['x']
        t0 = time.perf_counter()
        errs = {}
        with torch.no_grad():
            ht = torch.from_numpy(h)
            for f in FORMATS:
                errs[f] = (Q.act(ht, f, rx) - ht).numpy()
                shas[f'L{l}|{f}'] = hashlib.sha256(np.ascontiguousarray(errs[f]).tobytes()).hexdigest()
        tm['qdq_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        a = np.abs(h).max(1)
        thr = TAU * float(np.median(a))
        tm['mask_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        r, S = components(h, g, errs, thr, win)
        tm['all_components_s'] += time.perf_counter() - t0
        z, _ = components(h, g, {'fp32': np.zeros_like(errs['int4'])}, thr, win)
        zero_ok &= all(v == 0.0 for v in z['fp32']['scores'].values())
        om = old_mech[f'L{l}']
        mask_ok[f'L{l}'] = bool(float(old_sc[(l, 'int4')]['threshold']) == np.float64(thr) and om['spike_rows'] == int(S.sum())
                               and om['spike_positions'] == {str(k): v for k, v in Counter(int(p) for p in pos[S]).most_common(10)})
        masks[f'L{l}'] = S
        # timing of the A and D statistic kernels (same inputs; warm-up then repeats)
        gd = g.astype(np.float64)
        ed = {f: e.astype(np.float64) for f, e in errs.items()}
        for k, fn in (('A', lambda: kernel_A(ed, gd, win)), ('D', lambda: kernel_D(h, ed, gd, win))):
            v = fn()
            ref = {f: r[f]['scores'][k] for f in FORMATS}
            if any(abs(v[f] - ref[f]) > 1e-9 * abs(ref[f]) for f in FORMATS):
                raise SystemExit(f'kernel {k} differs from the component scores at L{l}')
            ts = []
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            kt[k].append(ts)
        del gd, ed
        for f in FORMATS:
            rf = r[f]
            o = old_sc[(l, f)]
            for k, ok in (('A_vs_T_old', ('A', 'T_old')), ('P_vs_T_pair', ('P', 'T_pair')), ('G00_vs_T_group', ('G00', 'T_group'))):
                rep[k] = max(rep[k], abs(rf['scores'][ok[0]] - float(o[ok[1]])) / abs(float(o[ok[1]])))
            if rf['fallback'] is None:
                idmax = max(idmax, rf['identity_rel'])
                score_round = max(score_round, rf['score_level_identity_rounding'])
                sumN_max = max(sumN_max, rf['sumN_e_minus_muN_rel'])
            mus[f'L{l}__{f}__mu_all'] = rf['mu_all']
            if rf['mu_N'] is not None:
                mus[f'L{l}__{f}__mu_N'] = rf['mu_N']
            comp_rows.append({'layer': l, 'format': f, 'threshold': thr, 'spike_rows': rf['nS'], 'N_rows': rf['nN'], 'fallback': rf['fallback'] or '',
                              'low_coverage': rf['low_coverage'], 'spike_positions': json.dumps(dict(Counter(int(p) for p in pos[S]).most_common(10))),
                              **{k: rf['components'].get(k, '') for k in ('V_all(mu_all)', 'U_all(mu_all)', 'P_all(mu_all)', 'V_N(mu_all)', 'V_S(mu_all)',
                                                                          'P_S(mu_all)', 'V_N(mu_N)', 'U_N(mu_N)', 'V_S(0)', 'P_S(0)')}})
            score_rows.append({'layer': l, 'format': f, **rf['scores'], **{f'{m}_per1270': rf['scores'][m] / (CAL * TP) for m in METHODS}})
            pk[(l, f)] = rf['scores']
        # leave-one-window-out (full-calibration threshold fixed)
        t0 = time.perf_counter()
        for w in range(CAL):
            rw, _ = components(h, g, errs, thr, win, keep=win != w)
            for f in FORMATS:
                lowo_rows.append({'layer': l, 'window_removed': w, 'format': f, 'fallback': rw[f]['fallback'] or '', **rw[f]['scores']})
        tm['lowo_s'] += time.perf_counter() - t0
        log(f'L{l}: S {int(S.sum())} rows; A/D ' + ', '.join(f'{f} {r[f]["scores"]["A"]:.3g}/{r[f]["scores"]["D"]:.3g}' for f in FORMATS))
    ctl.update({'old_score_reproduction_rel': rep, 'mask_equals_old': mask_ok, 'zero_error_all_zero': zero_ok, 'identity_rel_max': idmax,
                'score_level_identity_rounding_max': score_round,
                'sumN_e_minus_muN_rel_max': sumN_max, 'error_sha_equals_old_freeze': all(shas[k] == old_fz['error_sha256_cpu'][k] for k in shas),
                'position127_gradient_zero': all(bool((np.load(SPIKE / 'work' / f'g_L{l}.npy', mmap_mode='r').reshape(CAL, SEQ, -1)[:, -1] == 0).all())
                                                 for l in range(NL))})
    ctl['passed'] = bool(ctl['synthetic']['passed'] and ctl['old_plan_inputs_all_unchanged'] and ctl['error_sha_equals_old_freeze']
                         and all(v <= TOL['old_score_rel'] for v in rep.values()) and all(mask_ok.values()) and zero_ok
                         and idmax <= TOL['identity_rel'] and sumN_max <= TOL['sumN_rel'] and ctl['position127_gradient_zero']
                         and all(v[0] == 0 for v in ctl['posit_lut_mismatches'].values())
                         and all(v['h_rel'] <= TOL['spot_h_rel'] and v['g_rel'] <= TOL['spot_g_rel'] for v in sp.values())
                         and ctl['spot_check']['rms_norm_eps'] == [1e-6, 1e-6] and ctl['spot_check']['attn_implementation'] == 'eager')
    write(out / 'controls_scores.json', ctl)
    if not ctl['passed']:
        raise SystemExit('score controls failed (see controls_scores.json); nothing frozen')
    np.savez(out / 'mu.npz', **mus)
    np.savez(out / 'masks.npz', **masks)
    for name, rows in (('components.csv', comp_rows), ('scores.csv', score_rows), ('lowo.csv', lowo_rows)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    # predictions
    LO = defaultdict(dict)
    for r_ in lowo_rows:
        LO[(r_['layer'], r_['window_removed'])][r_['format']] = r_
    preds = []
    for l in range(NL):
        for f1, f2 in SAME_BIT:
            row = {'layer': l, 'f1': f1, 'f2': f2}
            for m in METHODS:
                d, tie, k = pick_tie(pk[(l, f1)][m], pk[(l, f2)][m])
                row[f'{m}_diff'], row[f'{m}_tie'], row[f'{m}_pick'] = d, tie, (f1, f2)[k]
                row[f'{m}_lowo_flips'] = sum(pick_tie(LO[(l, w)][f1][m], LO[(l, w)][f2][m])[2] != k for w in range(CAL))
            preds.append(row)
    with open(out / 'frozen_predictions.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(preds[0]))
        wr.writeheader()
        wr.writerows(preds)
    differ = [f"L{p['layer']} {p['f1']}-{p['f2']}" for p in preds if p['D_pick'] != p['A_pick']]
    gate = 'proceed: D changes picks' if differ else 'stop: D and A pick the same format in all 84 pairs'
    akt, dkt = np.array(kt['A']), np.array(kt['D'])                     # [28, repeats]
    sizes = {'A_streaming_float64_per_layer': (3 + 2 * len(FORMATS)) * C * 8, 'D_streaming_float64_per_layer': (3 + 3 * len(FORMATS)) * C * 8 + N * 4,
             'A_plus_D_float64_per_layer': (6 + 5 * len(FORMATS)) * C * 8 + N * 4,
             'note': 'A: sum g^2, sum_p g (running) and its square sum, per format sum e and sum e^2; D: the same on N rows plus per format '
                     'sum_S g^2 e^2 and the per-row a = max|h| (float32) for the median threshold; estimates of a streaming implementation, '
                     'not a native build'}
    tm.update({'kernel_A_s': {'median_sum_over_layers': float(np.median(akt, 1).sum()), 'min_sum': float(akt.min(1).sum()), 'max_sum': float(akt.max(1).sum())},
               'kernel_D_s': {'median_sum_over_layers': float(np.median(dkt, 1).sum()), 'min_sum': float(dkt.min(1).sum()), 'max_sum': float(dkt.max(1).sum())},
               'timing_protocol': f'per layer, same float64 inputs (cache read, QDQ and casts shared and excluded), one warm-up call (also checks the '
                                  f'kernel equals the component score), {TIMING_REPEATS} timed calls; median / min / max per layer summed over {NL} layers',
               'extra_statistics_bytes': sizes, 'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024})
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_scores.json'),
                                'scores_sha256': sha(out / 'scores.csv'), 'components_sha256': sha(out / 'components.csv'),
                                'predictions_sha256': sha(out / 'frozen_predictions.csv'), 'lowo_sha256': sha(out / 'lowo.csv'),
                                'mu_sha256': sha(out / 'mu.npz'), 'masks_sha256': sha(out / 'masks.npz'), 'error_sha256_cpu': shas,
                                'script_sha256': sha(HERE / 'run_qwen3_spike_recenter.py'), 'gate': gate, 'D_vs_A_differing_pairs': differ})
    log_stage(out, 'scores', time.perf_counter() - t_all, dict(tm))
    log(f'frozen; gate: {gate}; differing {differ}; controls {json.dumps({k: v for k, v in ctl.items() if k in ("old_score_reproduction_rel", "identity_rel_max")})}')


# --------------------------------------------------------------------------- #
# ranking analysis (shared by the exploratory old replay and the new articles)
# --------------------------------------------------------------------------- #
def load_preds(out):
    P_ = list(csv.DictReader(open(out / 'frozen_predictions.csv')))
    for p in P_:
        p['layer'] = int(p['layer'])
        for m in METHODS:
            p[f'{m}_diff'] = float(p[f'{m}_diff'])
            p[f'{m}_tie'] = p[f'{m}_tie'] == 'True'
            p[f'{m}_lowo_flips'] = int(p[f'{m}_lowo_flips'])
    return P_


def ranking(dn, preds, seed):
    """dn: [28, 6 (FORMATS), articles] dNLL; returns per-pair rows, counts, selection statistics."""
    na = dn.shape[-1]
    fi = {f: i for i, f in enumerate(FORMATS)}
    rng = np.random.default_rng(seed)
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na
    Dm = np.stack([dn[p['layer'], fi[p['f1']]] - dn[p['layer'], fi[p['f2']]] for p in preds])          # [84, A]  INT - Posit
    mean = Dm.mean(1)
    bs = Wb @ Dm.T
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    se = bs.std(0, ddof=1)
    tmax = np.abs((bs - mean) / np.where(se > 0, se, np.inf)).max(1)
    qsim = float(np.quantile(tmax, 1 - ALPHA))
    rows = []
    for k, p in enumerate(preds):
        dec = bool(lo[k] > 0 or hi[k] < 0)
        sl, sh = mean[k] - qsim * se[k], mean[k] + qsim * se[k]
        better = p['f1'] if mean[k] < 0 else p['f2']
        r = {'layer': p['layer'], 'pair': f"{p['f1']}-{p['f2']}", 'D_INT_minus_Posit': float(mean[k]), 'ci95': [float(lo[k]), float(hi[k])],
             'decisive': dec, 'simultaneous_ci95': [float(sl), float(sh)], 'confirmed': bool(sl > 0 or sh < 0), 'better': better if dec else None}
        for m in METHODS:
            r[m] = {'pick': p[f'{m}_pick'], 'tie': p[f'{m}_tie'], 'diff': p[f'{m}_diff'], 'lowo_flips': p[f'{m}_lowo_flips'],
                    'right': bool(dec and p[f'{m}_pick'] == better and not p[f'{m}_tie'])}
        rows.append(r)

    def counts(sel):
        rs = [r for r in rows if sel(r)]
        d = [r for r in rs if r['decisive']]
        c = {'pairs': len(rs), 'decisive': len(d), 'undetermined': len(rs) - len(d), 'confirmed': sum(r['confirmed'] for r in rs)}
        for m in METHODS:
            c[m] = {'right': sum(r[m]['right'] for r in d), 'ties_decisive': sum(r[m]['tie'] for r in d), 'ties_all': sum(r[m]['tie'] for r in rs),
                    'differing_picks_vs_A': sum(r[m]['pick'] != r['A']['pick'] for r in rs)}
            if m != 'A':
                fx = [r for r in d if r[m]['right'] and not r['A']['right']]
                ne = [r for r in d if r['A']['right'] and not r[m]['right']]
                c[m].update({'fixes': [f"L{r['layer']} {r['pair']}" for r in fx], 'new_errors': [f"L{r['layer']} {r['pair']}" for r in ne],
                             'confirmed_fixes': [f"L{r['layer']} {r['pair']}" for r in fx if r['confirmed']],
                             'confirmed_new_errors': [f"L{r['layer']} {r['pair']}" for r in ne if r['confirmed']]})
        return c
    cnt = {'all 84': counts(lambda r: True), **{f'{a}-{b}': counts(lambda r, a=a, b=b: r['pair'] == f'{a}-{b}') for a, b in SAME_BIT},
           'L2': counts(lambda r: r['layer'] == 2), 'L27': counts(lambda r: r['layer'] == 27),
           'other layers': counts(lambda r: r['layer'] not in (2, 27))}
    pickd = {m: np.stack([dn[p['layer'], fi[p[f'{m}_pick']]] for p in preds]) for m in METHODS}            # [84, A]
    R = {m: v.mean(0) for m, v in pickd.items()}
    sel = {'R_mean': {m: float(R[m].mean()) for m in METHODS}}

    def contrast(a, b):
        d = R[a] - R[b]
        bb = Wb @ d
        return {'mean': float(d.mean()), 'ci95': np.percentile(bb, [2.5, 97.5]).tolist(),
                'p_two_sided': float(min(1.0, 2 * min((bb >= 0).mean(), (bb <= 0).mean()))),
                'differing_picks': int(sum(p[f'{a}_pick'] != p[f'{b}_pick'] for p in preds)),
                'articles_better': int((d < 0).sum()), 'articles_worse': int((d > 0).sum())}
    sel['primary D-A'] = contrast('D', 'A')
    sec = {f'{a}-{b}': contrast(a, b) for a, b in SECONDARY}
    order_ = sorted(sec, key=lambda k: sec[k]['p_two_sided'])
    run_ = 0.0
    for j, k in enumerate(order_):
        run_ = max(run_, min(1.0, (len(order_) - j) * sec[k]['p_two_sided']))
        sec[k]['p_holm'] = run_
    sel['secondary'] = sec
    return rows, cnt, sel, qsim


def old_replay(out):
    t0 = time.perf_counter()
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    preds = load_preds(out)
    fi = {f: i for i, f in enumerate(FORMATS)}
    dn = np.full((NL, len(FORMATS), 128), np.nan)
    for r in csv.DictReader(open(SPIKE / 'per_article_losses.csv')):
        dn[int(r['layer']), fi[r['format']], int(r['article'])] = float(r['dnll'])
    if np.isnan(dn).any():
        raise SystemExit('old losses incomplete')
    rows, cnt, sel, qsim = ranking(dn, preds, OLD_BOOT_SEED)
    old = {(int(r['layer']), r['f1'], r['f2']): r for r in csv.DictReader(open(SPIKE / 'ranking_pairs.csv')) if r['primary'] == 'True'}
    repro = {'decisive_flags_equal_old': all((old[(r['layer'], *r['pair'].split('-'))]['decisive'] == 'True') == r['decisive'] for r in rows),
             'A_right_equals_old_T_old': all((old[(r['layer'], *r['pair'].split('-'))]['T_old_right'] == 'True') == r['A']['right'] for r in rows),
             'G00_right_equals_old_T_group': all((old[(r['layer'], *r['pair'].split('-'))]['T_group_right'] == 'True') == r['G00']['right'] for r in rows),
             'P_right_equals_old_T_pair': all((old[(r['layer'], *r['pair'].split('-'))]['T_pair_right'] == 'True') == r['P']['right'] for r in rows)}
    write(out / 'old_replay.json', {'label': 'EXPLORATORY: articles of runs/qwen3_down_spike_diag_20261023 (seen before); not a validation; never '
                                             'used to choose a method, threshold or D', 'bootstrap_seed': OLD_BOOT_SEED, 'reproduction_of_old_analysis': repro,
                                    'simultaneous_quantile': qsim, 'counts': cnt, 'selection': sel, 'pairs': rows})
    log_stage(out, 'old_replay', time.perf_counter() - t0)
    log(f'old replay (exploratory): reproduction {repro}; right {({m: cnt["all 84"][m]["right"] for m in METHODS})} of {cnt["all 84"]["decisive"]}; '
        f'D-A {sel["primary D-A"]}')


# --------------------------------------------------------------------------- #
# audit / data (only if the gate says so)
# --------------------------------------------------------------------------- #
def sampler(out, stage):
    check_plan(out)
    fz = read(out / 'freeze.json')
    if not fz['gate'].startswith('proceed'):
        raise SystemExit(f'gate: {fz["gate"]}')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    log_stage(out, stage, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# eval (GPU; the evaluator path of run_qwen_down_spike_diag; chunked, resumable)
# --------------------------------------------------------------------------- #
def eval_(out):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from run_qwen_down_spike_diag import Model, Quant, summed_ce_rows
    from run_olmoe_dynamic_route import env_info
    t_all = time.perf_counter()
    check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if not fz['gate'].startswith('proceed'):
        raise SystemExit(f'gate: {fz["gate"]}')
    if sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'segments.json') != dz['segments_sha256']:
        raise SystemExit('freeze / data changed')
    for k, f in (('scores', 'scores.csv'), ('predictions', 'frozen_predictions.csv'), ('lowo', 'lowo.csv'), ('masks', 'masks.npz')):
        if sha(out / f) != fz[f'{k}_sha256']:
            raise SystemExit(f'{f} changed after the freeze')
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    dn_ = down_nodes()
    ranges = read(NATIVE / 'ranges_by_point.json')
    work = out / 'work'
    Q = Quant(work, dev)
    t0 = time.perf_counter()
    bad = []
    with torch.no_grad():
        for l in range(NL):
            ht = torch.from_numpy(np.load(SPIKE / 'work' / f'h_L{l}.npy')).to(dev)
            for f in FORMATS:
                e = (Q.act(ht, f, ranges[dn_[l]]['x']) - ht).cpu().numpy()
                if hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() != fz['error_sha256_cpu'][f'L{l}|{f}']:
                    bad.append(f'L{l}|{f}')
    ctl = {'environment': env_info(), 'gpu_qdq_equals_cpu': not bad, 'gpu_qdq_mismatches': bad, 'qdq_check_s': time.perf_counter() - t0,
           'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled(),
           'posit_lut_mismatches': Q.verify()}
    if bad:
        write(out / 'controls_eval.json', ctl)
        raise SystemExit(f'GPU QDQ differs from CPU: {bad[:5]}')
    M = Model(dev)
    ctl['attn_implementation'] = M.model.config._attn_implementation
    ctl['rms_norm_eps'] = [float(M.layers[0].post_attention_layernorm.variance_epsilon), float(M.model.model.norm.variance_epsilon)]
    toks = np.load(out / 'tokens.npy')
    nw = len(toks)
    F7 = ('fp32',) + FORMATS
    cap, kw = {}, {}

    def pre_mid(l):
        def hook(mod, inp):
            cap[(l, 'mid')] = inp[0]
        return hook

    def pre_down(l):
        def hook(mod, inp):
            cap[(l, 'h')] = inp[0]
        return hook

    def pre_kw(mod, args, kwargs):
        kw.update(kwargs)

    def head_ce(normed, ids):
        ce = []
        for i in range(0, normed.shape[0], LM_SUB):
            z = M.model.lm_head(normed[i:i + LM_SUB])
            ce.append(summed_ce_rows(torch, z, ids[i:i + LM_SUB]))
            del z
        return torch.cat(ce)

    def forward_ce(ids):
        hs = M.model.model(input_ids=ids, attention_mask=M.wrapper.mask, use_cache=False).last_hidden_state
        return head_ce(hs, ids)

    def downstream(hidden, l, ids):
        for layer in M.layers[l + 1:]:
            o = layer(hidden, **kw)
            hidden = o[0] if isinstance(o, tuple) else o
        return head_ce(M.model.model.norm(hidden), ids)
    hdr0 = {'tokens_sha256': dz['tokens_sha256'], 'freeze_sha256': dz['freeze_sha256'], 'script_sha256': sha(HERE / 'run_qwen3_spike_recenter.py')}
    torch.cuda.reset_peak_memory_stats()
    id_diff, t_fwd, t_rep = [], 0.0, 0.0
    nchunk = -(-nw // CHUNK)
    for k, c0 in enumerate(range(0, nw, CHUNK)):
        ck = work / f'eval_chunk{k:02d}.npz'
        hdr = hdr0 | {'chunk': k, 'windows': [c0, min(nw, c0 + CHUNK)]}
        if ck.exists():
            if json.loads(str(np.load(ck)['header'])) != hdr:
                raise SystemExit(f'{ck}: checkpoint from another version / data')
            log(f'chunk {k}: reusing checkpoint')
            continue
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).to(dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            nohook = forward_ce(ids)
            hooks = [M.layers[l].post_attention_layernorm.register_forward_pre_hook(pre_mid(l)) for l in range(NL)]
            hooks += [M.layers[l].mlp.down_proj.register_forward_pre_hook(pre_down(l)) for l in range(NL)]
            hooks.append(M.layers[1].register_forward_pre_hook(pre_kw, with_kwargs=True))
            hooked = forward_ce(ids)
            for hk in hooks:
                hk.remove()
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t1
        if not torch.equal(nohook, hooked):
            raise SystemExit('hooks changed the forward')
        ce = np.full((NL, len(F7), len(ids)), np.nan)
        t1 = time.perf_counter()
        with torch.no_grad():
            for l in range(NL):
                down = M.layers[l].mlp.down_proj
                ce[l, 0] = downstream(cap[(l, 'mid')] + down(cap[(l, 'h')]), l, ids).cpu().numpy()
            dmax = float(np.abs(ce[:, 0] - nohook.cpu().numpy()).max())
            if dmax > TOL['replay_identity_abs_summed_ce']:
                write(out / 'controls_eval.json', {**ctl, 'identity_vs_nohook_max_abs': dmax})
                raise SystemExit(f'identity replay differs from the no-hook forward by {dmax}')
            for l in range(NL):
                down = M.layers[l].mlp.down_proj
                for fi_, f in enumerate(FORMATS, start=1):
                    hq = Q.act(cap[(l, 'h')], f, ranges[dn_[l]]['x'])
                    ce[l, fi_] = downstream(cap[(l, 'mid')] + down(hq), l, ids).cpu().numpy()
        torch.cuda.synchronize()
        t_rep += time.perf_counter() - t1
        cap.clear()
        np.savez(ck, ce=ce, full=nohook.cpu().numpy(), identity_max_abs=dmax, header=json.dumps(hdr))
        log(f'chunk {k + 1}/{nchunk}: identity max |dCE| {dmax:.2e}; forward {t_fwd:.0f}s, replay {t_rep:.0f}s')
    parts = [np.load(work / f'eval_chunk{k:02d}.npz') for k in range(nchunk)]
    ce = np.concatenate([p['ce'] for p in parts], axis=-1)
    full = np.concatenate([p['full'] for p in parts])
    id_diff = [float(p['identity_max_abs']) for p in parts]
    np.savez(out / 'eval_ce.npz', ce=ce, full=full, formats=np.array(F7))
    ctl.update({'identity_vs_nohook_max_abs_per_chunk': id_diff, 'identity_vs_nohook_max_abs': max(id_diff),
                'nonfinite': int((~np.isfinite(ce)).sum()), 'peak_gpu_memory_bytes_this_process': int(torch.cuda.max_memory_allocated())})
    ctl['passed'] = bool(ctl['gpu_qdq_equals_cpu'] and ctl['tf32'] == [False, False] and ctl['deterministic'] and ctl['attn_implementation'] == 'eager'
                         and ctl['rms_norm_eps'] == [1e-6, 1e-6] and all(v[0] == 0 for v in ctl['posit_lut_mismatches'].values())
                         and ctl['identity_vs_nohook_max_abs'] <= TOL['replay_identity_abs_summed_ce'])
    write(out / 'controls_eval.json', ctl)
    log_stage(out, 'eval', time.perf_counter() - t_all, {'qdq_check_s': ctl['qdq_check_s'], 'fp32_forward_s_this_process': t_fwd,
                                                         'replay_s_this_process': t_rep, 'peak_gpu_memory_bytes': ctl['peak_gpu_memory_bytes_this_process'],
                                                         'windows': nw, 'eval_ce_sha256': sha(out / 'eval_ce.npz')})
    log(f'eval done; identity max {max(id_diff):.2e}; passed {ctl["passed"]}')


# --------------------------------------------------------------------------- #
# analyze (new articles)
# --------------------------------------------------------------------------- #
def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    fz = read(out / 'freeze.json')
    for k, f in (('scores', 'scores.csv'), ('predictions', 'frozen_predictions.csv'), ('lowo', 'lowo.csv')):
        if sha(out / f) != fz[f'{k}_sha256']:
            raise SystemExit(f'{f} changed after the freeze')
    ce_ctl = read(out / 'controls_eval.json')
    if not (read(out / 'controls_scores.json')['passed'] and ce_ctl['passed']):
        raise SystemExit('controls did not pass')
    preds = load_preds(out)
    z = np.load(out / 'eval_ce.npz')
    ce = z['ce']
    segs = read(out / 'segments.json')
    na = len(segs)
    nll = ce.reshape(NL, 1 + len(FORMATS), na, WPA).sum(-1) / (WPA * TP)
    dn = nll[:, 1:] - nll[:, :1]                                                # [28, 6, A]
    rows, cnt, sel, qsim = ranking(dn, preds, SEEDS['bootstrap'])
    by = {(r['layer'], r['pair']): r for r in rows}
    # candidate rule
    c = cnt['all 84']['D']
    l2conf = [p for p in ('L2 int6-p6e1', 'L2 int8-p8e1') if p in c['confirmed_fixes']]
    r4 = by[(2, 'int4-p4e1')]
    rule = {'1_G_upper_below_0': bool(sel['primary D-A']['ci95'][1] < 0),
            '2_confirmed_fixes_gt_new_errors_and_L2_fix': bool(len(c['confirmed_fixes']) > len(c['confirmed_new_errors']) and l2conf),
            '3_L2_4bit_kept_if_decided': bool((not r4['decisive']) or r4['D']['right']),
            '4_L2_fix_lowo_stable': bool(l2conf and all(by[(2, p.split(' ')[1])]['D']['lowo_flips'] <= 2 for p in l2conf))}
    rule['candidate_for_next_round'] = bool(all(rule.values()))
    fi = {f: i for i, f in enumerate(FORMATS)}
    with open(out / 'per_article_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'article', 'title', 'nll_fp32_identity', 'nll_quantized', 'dnll'])
        for l in range(NL):
            for f in FORMATS:
                for a in range(na):
                    wr.writerow([l, f, a, segs[a]['title'], nll[l, 0, a], nll[l, fi[f] + 1, a], dn[l, fi[f], a]])
    with open(out / 'same_bit_pairs.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'pair', 'D_INT_minus_Posit', 'ci_lo', 'ci_hi', 'decisive', 'sim_lo', 'sim_hi', 'confirmed',
                     *[f'{m}_{k}' for m in METHODS for k in ('pick', 'tie', 'right', 'lowo_flips', 'diff')]])
        for r in rows:
            wr.writerow([r['layer'], r['pair'], r['D_INT_minus_Posit'], *r['ci95'], r['decisive'], *r['simultaneous_ci95'], r['confirmed'],
                         *[r[m][k] for m in METHODS for k in ('pick', 'tie', 'right', 'lowo_flips', 'diff')]])
    with open(out / 'selection.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['contrast', 'mean', 'ci_lo', 'ci_hi', 'p_two_sided', 'p_holm', 'differing_picks', 'articles_better', 'articles_worse'])
        for k, v in [('D-A (primary)', sel['primary D-A'])] + list(sel['secondary'].items()):
            wr.writerow([k, v['mean'], *v['ci95'], v['p_two_sided'], v.get('p_holm', ''), v['differing_picks'], v['articles_better'], v['articles_worse']])
    res = {'units': 'dNLL nat/token per article (3 windows, 381 targets); single-layer conditions', 'simultaneous_quantile': qsim, 'counts': cnt,
           'selection': sel, 'candidate_rule': rule, 'pairs': rows,
           'dnll_by_layer': {f'L{l}': {f: float(dn[l, fi[f]].mean()) for f in FORMATS} for l in range(NL)}}
    write(out / 'summary.json', res)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'D-A': sel['primary D-A'], 'right': {m: cnt['all 84'][m]['right'] for m in METHODS}, 'decisive': cnt['all 84']['decisive'],
                    'D_fixes': c['fixes'], 'D_new_errors': c['new_errors'], 'confirmed': [c['confirmed_fixes'], c['confirmed_new_errors']], 'rule': rule},
                   indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'scores', 'old_replay', 'audit', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note')
    a = ap.parse_args()
    out = a.out_dir.resolve()
    if a.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty')
    elif not (out / 'plan.json').exists():
        raise SystemExit('plan first')
    out.mkdir(parents=True, exist_ok=True)
    for d in ('logs', 'work'):
        (out / d).mkdir(exist_ok=True)
    LOGF = open(out / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage == 'plan':
        plan(out, a.smoke_note)
    elif a.stage in ('audit', 'data'):
        sampler(out, a.stage)
    else:
        {'scores': scores, 'old_replay': old_replay, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_spike_recenter.py')}) + '\n')


if __name__ == '__main__':
    main()
