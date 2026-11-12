#!/usr/bin/env python3
"""OLMoE spike rows: a causal intervention on their quantization error. Pre-registered in plan.json / plan.md.

Question: BC charges spike rows of some Down inputs with a high cost; does removing the quantization error of exactly those rows
change the final NLL accordingly? One expert's Down input h is quantized at a time (FP32 background, normal dynamic route,
fixed calibration range); conditions F (all rows quantized), P (spike rows kept FP32), S (only spike rows quantized), R (as many
random non-spike rows kept FP32, 3 fixed seeds), Z (tap in FP32). Protection replaces Q(h) by h on the chosen rows; no token is
removed and nothing is set to zero. BC is recomputed per condition on the calibration rows with the protected errors set to zero
(same sensitivities, same normalization). BC criterion, quantizer, search, knee and pipeline unchanged; nothing added to them.

Stages (one --out-dir): correct_prev, plan, prep, freeze, data, measure, analyze, figures.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_act_outlier_diag as od
import run_olmoe_dynamic_route as dr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss
import run_olmoe_token_spike_diag as tsd
import wt_manifest_scan as wms

HERE = Path(__file__).resolve().parent
PREV_TS, PREV_AO = HERE / 'runs/olmoe_token_spike_diag_20261023', HERE / 'runs/olmoe_act_outlier_diag_20261023'
CORR = PREV_TS / 'correction_20261023'
NOTE_TARGET = PREV_TS / 'report.md'                          # the correction note is added at the top of this report
RANGES, CALIB = ss.RANGES, ss.CALIB
ACT_CACHE, ACT_INDEX, GC, GC_INDEX, Q0K = od.ACT_CACHE, od.ACT_INDEX, od.GC, od.GC_INDEX, od.Q0K
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
FOCUS = {'L1.E9': 'known_case', 'L1.E11': 'position0', 'L1.E18': 'position0', 'L2.E30': 'position0', 'L6.E59': 'int4_improvement'}
CTRL_ANCHOR = {1: 'L1.E9', 2: 'L2.E30', 6: 'L6.E59'}       # one weak-spike control per layer, coverage-matched to this tap
FORMATS = ('int4', 'p6e1')
TAU = tsd.TAU                                                # spike row: s_t >= 6 x median_t s_t (FP32 discovery of the token-spike round)
R_SEEDS = (23341, 23342, 23343)
FOCUS_CONDS = ('F', 'P', 'S') + tuple(f'R{i}' for i in range(len(R_SEEDS)))
CTRL_CONDS = ('F', 'P', 'S')
N_ART, WPA_C, CHUNK, VAR_BATCH = 96, 3, 128, 12
BOOT, ALPHA, B_FRACTION, B_MIN_EFFECT = 10000, 0.05, 0.25, 5e-4
SEEDS = {'order': 23301, 'start': 23302, 'bootstrap': 23311, 'select': 23321}
BUDGET_MIN = {'prep': 5, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 35}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_spike_intervention.py', 'wt_manifest_scan.py', 'run_olmoe_token_spike_diag.py',
                                                  'run_olmoe_act_outlier_diag.py'] + ss.CODE + dr.SCRIPTS))}


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def node_le(node):
    l, e = node[1:].split('.E')
    return int(l), int(e)


def fp32_experts():
    with open(PREV_TS / 'discovery_experts.csv') as fh:
        return {r['node']: r for r in csv.DictReader(fh) if r['state'] == 'fp32'}


# --------------------------------------------------------------------------- #
def correct_prev(out, log):
    """article isolation: verified manifest scan, the duplicate in the token-spike round, and its results without the duplicate."""
    if (CORR / 'correction.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    CORR.mkdir(parents=True, exist_ok=True)
    orig = {f: sha(PREV_TS / f) for f in ('report.md', 'analysis.json', 'segments.json', 'measure.json', 'ranking_pairs.csv', 'per_tap.csv', 'data_freeze.json')}
    w103, wt2 = wms.corpora()
    by = {a['ch']: a for a in w103}
    old_used, _ = ss.used_wt103_all(PREV_TS)
    new_used, rows = wms.used_wt103_verified(PREV_TS, w103, wt2)
    segs = read(PREV_TS / 'segments.json')
    exact = [i for i, s in enumerate(segs) if s['content_sha256'] in new_used]
    prev_sh = [mr.shingles(by[h]['body_norm']) for h in new_used]
    near = {}
    for i, s in enumerate(segs):
        sh = mr.shingles(by[s['content_sha256']]['body_norm'])
        m = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        if m >= mr.NEARDUP:
            near[i] = m
    drop = sorted(set(exact) | set(near))
    validation = {'old_scanner_used': len(old_used), 'verified_used': len(new_used), 'verified_superset': old_used <= new_used,
                  'missing_from_old': len(new_used - old_used), 'missing_by_manifest': {r['path']: r['articles'] for r in rows
                                                                                      if r['resolved'] == 'wikitext-103' and r.get('declared_missing')},
                  'duplicates_exact': [{'index': i, 'title': segs[i]['title'], 'content_sha256': segs[i]['content_sha256'],
                                        'in_old_scanner': segs[i]['content_sha256'] in old_used} for i in exact],
                  'near_duplicates': {str(i): v for i, v in near.items()}, 'dropped_articles': drop}
    if not exact or any(d['in_old_scanner'] for d in validation['duplicates_exact']):
        raise SystemExit('the known duplicate is not caught by the verified scan only')
    keep = [a for a in range(tsd.N_ART) if a not in drop]
    res = recompute_token_spike(keep)
    corr = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cause': 'run_olmoe_selective_sensitivity.used_wt103_all classified manifests only by their '
            'source / split fields; runs/olmoe_act_outlier_diag_20261023/segments.json has neither, so its 64 WikiText-103 articles were not '
            'excluded when the token-spike round drew its confirmation articles', 'fix': 'wt_manifest_scan.used_wt103_verified (content-hash '
            'membership; manifests without hashes must be shown to be WikiText-2; unresolved manifests are errors); the old function is kept '
            'unchanged for the recorded code hashes', 'manifests': rows, 'validation': validation, 'original_hashes': orig,
            'recomputed_without_dropped': res, 'code': {f: sha(HERE / f) for f in ('run_olmoe_spike_intervention.py', 'wt_manifest_scan.py')}}
    write(CORR / 'correction.json', corr)
    write(out / 'correction_summary.json', {k: corr[k] for k in ('time', 'cause', 'fix', 'validation', 'original_hashes')})
    # note at the top of the token-spike report (original content kept below)
    rep = NOTE_TARGET.read_text()
    if '更正註記' not in rep:
        P4 = res['P4']
        kc = res['known_case']['now']
        note = ['', '> **更正註記（2026-10-23，資料隔離）**',
                f'> 本輪確認文章 #{drop[0]}「{segs[drop[0]]["title"]}」與 act_outlier 輪重複（content_sha256 相同，token 窗口不同），違反文章級隔離。',
                '> 原因：`used_wt103_all()` 只依 source／split 欄位判定來源，act_outlier 輪的 segments.json 缺這兩個欄位，因此其 64 篇 WT103 文章沒有被排除。',
                f'> 修正：`wt_manifest_scan.py`（內容 hash 判定）。排除該文章後以 {len(keep)} 篇重算：',
                f'> - p4e1 強弱配對差 {P4["p4e1"]["mean_diff"]:+.6f}，97.5% CI [{P4["p4e1"]["ci97.5"][0]:+.6f}, {P4["p4e1"]["ci97.5"][1]:+.6f}]；'
                f'int4 {P4["int4"]["mean_diff"]:+.6f}，97.5% CI [{P4["int4"]["ci97.5"][0]:+.6f}, {P4["int4"]["ci97.5"][1]:+.6f}]。',
                '> - L1.E9 上一輪誤排的重現：' + '；'.join(f'{x["f1"]} vs {x["f2"]} {x["measured_diff"]:+.5f} [{x["ci95"][0]:+.5f}, {x["ci95"][1]:+.5f}]'
                                                    f'{"（決定性、仍誤排）" if x["decisive"] and not x["bc_correct"] else "（CI 含 0）"}' for x in kc) + '。',
                f'> - 新 tap 排序：決定性 {res["ranking"]["all_new"]["decisive"]}，BC 正確 {res["ranking"]["all_new"]["correct"]}（原 61／53）；L1.E9 決定性 '
                f'{res["ranking"]["known_case"]["decisive"]}，BC 正確 {res["ranking"]["known_case"]["correct"]}（原 10／8）。誤排：'
                + '、'.join(f'{x["node"]} {x["f1"]} vs {x["f2"]}' for x in res['misranked']) + '。',
                '> - 詳見 `correction_20261023/correction.json` 與 `correction.md`；原始輸出與其 hash 保留不變（本註記改變了本檔 hash，原 hash 見 correction.json）。', '']
        lines = rep.split('\n')
        NOTE_TARGET.write_text('\n'.join(lines[:1] + note + lines[1:]))
    md = ['# 更正：token-spike 輪的文章重複', '', f'時間：{corr["time"]}', '', '## 原因', f'- {corr["cause"]}', '', '## 修正', f'- {corr["fix"]}', '',
          '## 驗證', f'- 舊掃描 {len(old_used)} 篇，驗證掃描 {len(new_used)} 篇（舊集合是子集：{validation["verified_superset"]}）；舊掃描漏掉 {len(new_used - old_used)} 篇，'
          f'全部來自缺來源欄位的 manifest：{validation["missing_by_manifest"]}。',
          f'- 已知重複：{validation["duplicates_exact"]}；驗證掃描會排除，舊掃描不會。近似重複（shingle ≥ {mr.NEARDUP}）：{validation["near_duplicates"] or "無"}。',
          '- 沒有 hash 的 manifest 全部以證據判定為 WikiText-2（WT2 文章本來就從 WT103 抽樣池整批排除）：'] + \
         [f'  - {r["path"]}：{r["evidence"]}' for r in rows if r['resolved'] == 'wikitext-2'] + \
         ['', f'## 以 {len(keep)} 篇重算（seed 不變；bootstrap 抽樣因篇數不同而不同）', '```', json.dumps(dr.clean(res), indent=1, ensure_ascii=False)[:6000], '```', '']
    (CORR / 'correction.md').write_text('\n'.join(md))
    log_stage(out, 'correct_prev', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'validation': validation, 'P4': res['P4'], 'known': res['known_case']})))


def recompute_token_spike(keep):
    """the token-spike round's confirmation statistics on the kept articles (same rules, same bootstrap seed)."""
    keep = np.array(keep)
    n_all = tsd.N_ART

    def ad(ce_v, ce_0):
        d = (np.array(ce_v) - np.array(ce_0)).reshape(n_all, tsd.WPA_C).sum(1) / (tsd.WPA_C * TP)
        return d[keep]
    saved = (tsd.article_delta, tsd.N_ART)
    tsd.article_delta, tsd.N_ART = ad, len(keep)
    try:
        per_tap, pairs = tsd.ranking(PREV_TS)
    finally:
        tsd.article_delta, tsd.N_ART = saved
    new = lambda p: p['role'] != 'known_case'
    R = {'all_new': tsd.acc_summary([p for p in pairs if new(p)]), 'known_case': tsd.acc_summary([p for p in pairs if not new(p)]),
         'same_bits_new': tsd.acc_summary([p for p in pairs if new(p) and p['same_bits']])}
    for fam in ('int-int', 'posit-posit', 'cross'):
        R[f'family:{fam}'] = tsd.acc_summary([p for p in pairs if new(p) and p['family'] == fam])
    m = read(PREV_TS / 'measure.json')
    taps = read(PREV_TS / 'selected_taps.json')['taps']
    idx = np.random.default_rng(tsd.SEEDS['bootstrap']).integers(0, len(keep), (BOOT, len(keep)))
    P4 = {}
    for f in tsd.CANDS:
        ds = []
        for k_ in range(tsd.N_PAIRS):
            st = next(t for t in taps if t['role'] == 'pair_strong' and t['pair'] == k_)
            wk = next(t for t in taps if t['role'] == 'pair_weak' and t['pair'] == k_)
            ds.append(ad(m['variants'][f'tap{st["id"]}|{f}'], m['base_ce']) - ad(m['variants'][f'tap{wk["id"]}|{f}'], m['base_ce']))
        D = np.mean(ds, 0)
        bs = D[idx].mean(1)
        P4[f] = {'mean_diff': float(D.mean()), 'ci97.5': np.percentile(bs, [1.25, 98.75]).tolist(), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(),
                 'primary': f in ('int4', 'p4e1')}
    pt = {t['tap']: t for t in per_tap}
    prevp = list(csv.DictReader(open(PREV_AO / 'ranking_pairs.csv')))
    old = [(p['f1'], p['f2']) for p in prevp if p['tap'] == '18' and p['decisive'] == 'True' and p['bc_tie'] == 'False' and p['bc_correct'] == 'False']
    kp = [p for p in pairs if p['role'] == 'known_case']
    known = {'now': [next({'f1': a, 'f2': b, 'measured_diff': p['measured_diff'], 'ci95': p['ci95'], 'decisive': p['decisive'], 'bc_correct': p['bc_correct']}
                          for p in kp if p['f1'] == a and p['f2'] == b) for a, b in old]}
    known['replicated'] = sum(x['decisive'] and not x['bc_correct'] for x in known['now'])
    mis = [{'tap': p['tap'], 'node': p['node'], 'f1': p['f1'], 'f2': p['f2'], 'diff': p['measured_diff'], 'ci95': p['ci95']} for p in pairs
           if p['decisive'] and not p['bc_tie'] and not p['bc_correct']]
    return {'articles': len(keep), 'ranking': R, 'P4': P4, 'known_case': known, 'misranked': mis,
            'per_tap_dnll': {pt[i]['node']: pt[i]['dnll'] for i in pt}}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    if not (out / 'correction_summary.json').exists():
        raise SystemExit('run correct_prev first')
    t0 = time.perf_counter()
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [Q0K, ACT_CACHE, ACT_INDEX, GC, GC_INDEX, RANGES, CALIB, PREV_TS / 'discovery_experts.csv',
                                                         PREV_TS / 'selected_taps.json', PREV_TS / 'segments.json', PREV_AO / 'segments.json',
                                                         PREV_AO / 'selected_taps.json'] + mr.WT103 + mr.WT2}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    ex = fp32_experts()
    thr = {n: TAU * float(ex[n]['s_p50']) for n in FOCUS}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'BC charges spike rows with a high cost; does removing the quantization error of exactly those rows change the final NLL '
                    'accordingly?',
        'fixed': {'model': 'OLMoE-1B-7B, original checkpoint and router', 'quantizer': 'dev_quant(corrected=True), original fixed calibration range '
                  '(ranges.json), no recalibration', 'bc': 'static sensitivities (G2 / GS) of the BC calibration; formula and normalization unchanged',
                  'not': 'no fine-tuning, no spike penalty, no first-order or per-token criterion variant, nothing added to the pipeline'},
        'isolation': {'scanner': 'wt_manifest_scan.used_wt103_verified (content-hash membership; manifests without hashes must be WikiText-2)',
                      'excluded_previous_wt103': len(used), 'manifests': len(rows), 'eligible_before_length_and_neardup': len(elig),
                      'also_excluded': 'WikiText-2 articles (normalized title and content hash), near duplicates (shingle share >= '
                                       f'{mr.NEARDUP}) of WikiText-2, of every earlier article and of accepted ones; discovery / selection data of '
                                       'this round (token-spike discovery = calibration + act-outlier windows; token-spike confirmation) are in '
                                       'the excluded manifests', 'correction': 'see correction_summary.json'},
        'taps': {'focus': FOCUS, 'known_case': 'L1.E9 is a replication (reported apart from the generalization taps L1.E11, L1.E18, L2.E30, L6.E59)',
                 'controls': 'one weak-spike control per layer {1, 2, 6}: same layer, R_token <= p50 of the token-spike eligible set (searched, '
                             'calibration rows >= 64, discovery rows >= 512, finite; taps of earlier rounds allowed, selection never uses a loss), '
                             f'nearest |log rows| to {CTRL_ANCHOR}, same q0 Down activation bit class preferred, ties at random (seed '
                             f'{SEEDS["select"]}); if a layer has no candidate <= p50, the expert with the lowest R_token of the layer (recorded); '
                             'discovery statistics only', 'no_loss': 'no confirmation loss used for any selection'},
        'formats': {'primary': list(FORMATS), 'secondary': 'none (int6 / p4e1 not run, cost control)'},
        'spike_rule': {'rule': f's_t = max_c |h[t,c]| >= {TAU} x median_t s_t of the expert in the FP32 discovery of the token-spike round',
                       'thresholds': thr, 'same_mask': 'the mask is computed from the FP32 input h at the tap, identical for every format and '
                                                       'condition of a tap (checked by row-id checksums)'},
        'conditions': {'Z': 'tap in FP32 (= the FP32 forward; checked bitwise)', 'F': 'all rows quantized', 'P': 'spike rows keep FP32 (Q(h) -> h), '
                       'others quantized', 'S': 'only spike rows quantized', 'R': f'as many random non-spike rows as spike rows keep FP32, seeds '
                       f'{list(R_SEEDS)}; per seed averaged per window before article statistics; no seed is chosen',
                       'random_rows': 'in each window, as many non-spike rows of the same expert as it has spike rows, uniformly without '
                                      'replacement (rng keyed by seed and window); if a window has fewer non-spike rows than spike rows, all of '
                                      'them plus the shortfall drawn uniformly from the other non-spike rows of the same forward chunk; '
                                      'shortfalls counted and reported', 'controls_taps': 'F, P, S only',
                       'scale': 'every condition uses the original range of the tap; P therefore does not change the step of the other rows: '
                                'this round measures the spike rows\' own error and its interplay, not the outlier-widened range'},
        'measurement': {'data': f'{N_ART} new WikiText-103 train articles x {WPA_C} windows of {SEQ}, drawn after the freeze (seeds order '
                                f'{SEEDS["order"]}, start {SEEDS["start"]})', 'background': 'FP32 weights and activations except the one '
                        'tap; normal dynamic route; layers before the tap reused from the FP32 pass', 'dnll': 'D_X = NLL(X) - NLL(Z), per article '
                        '(sum over its windows / tokens)', 'records': 'rows, spike rows, protected rows, shortfalls, mask checksums, residual '
                        'error energy on spike / other rows, range used, downstream route changes, coverage (rows, spike rows, windows, articles)'},
        'bc_counterfactual': {'rows': 'the BC calibration rows of the tap (FP32 forward of the 10 calibration windows)', 'rule': 'for each '
                              'condition the errors of the protected rows are set to 0 and S1, S2 are recomputed; n (rows) unchanged; '
                              'T = sum_c G2_c / n (S2_c - n mu_c^2) + GS_c mu_c^2 with the static G2 / GS; R uses the same random-row rule on the '
                              'calibration windows (same seeds); T_F is checked against the cached BC cost',
                              'not': 'no subtraction of old spike shares from the original score'},
        'contrasts': {'C1': 'NLL(P) - NLL(F): does protecting spike rows help', 'C2': 'NLL(P) - mean_seeds NLL(R): more than protecting as many '
                      'random rows', 'C3': 'D_F - D_P - D_S: overall non-additivity of the two error parts (not a Hessian cross term; includes '
                      'non-linearity and routing)'},
        'multiplicity': {'replication_family': 'L1.E9: C1 and C2 x 2 formats = 4 tests, Holm at FWER 0.05',
                         'generalization_family': 'L1.E11, L1.E18, L2.E30, L6.E59: C1 and C2 x 2 formats = 16 tests, Holm at FWER 0.05',
                         'p_values': f'two-sided article-bootstrap p = 2 min(P(b <= 0), P(b >= 0)), floor 1/({BOOT}+1)',
                         'C3_and_controls': '95% CIs, descriptive'},
        'categories': {'A': 'C1 < 0, Holm-significant: spike-row errors have a real effect',
                       'B': f'BC-implied effect E_BC = k (T_P - T_F) with the int4 anchor k = D_F(int4) / T_F(int4) of the same tap (only if the '
                            f'95% CI of D_F(int4) is above 0) and |E_BC| >= {B_MIN_EFFECT}; B if the whole 95% CI of C1 lies within '
                            f'+-{B_FRACTION} |E_BC|: BC predicts a large change, the measured one is small with a narrow CI',
                       'C': '95% CI of C3 excludes 0: non-additivity, mechanism not identified',
                       'D': 'C1 > 0, Holm-significant: protection hurts (favourable perturbation or cancellation possible)',
                       'E': 'none of A, B, D: undeterminable (not significant is not small)', 'scale': 'BC and NLL scales differ; no '
                       'numeric equality is required and different relative drops alone do not mean the criterion fails'},
        'bc_comparison': 'direction of T_P - T_F vs C1; format order (int4 vs p6e1) under F and under P; whether both say P is more useful '
                         'than R (T_P < mean T_R vs C2 < 0); BC non-additivity T_F - T_P - T_S vs C3',
        'statistics': {'bootstrap': f'{BOOT} article bootstraps (seed {SEEDS["bootstrap"]}); windows of an article and candidates of a tap are not '
                       'independent cases', 'stop': 'if the measurement exceeds its budget, stop before the next layer group; no articles '
                       'added, no seed or threshold change after results'},
        'budget_minutes': BUDGET_MIN, 'seeds': {**SEEDS, 'random_rows': list(R_SEEDS)}, 'inputs': inputs, 'model_files': mfiles,
        'code_hashes': code_hashes(), 'environment': dr.env_info(), 'manifest_scan': rows}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE spike rows 量化誤差的因果干預', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 問題', '- BC 對 spike rows 給出高成本；只消除這些 rows 的量化誤差，最終 NLL 是否相應改變？',
          '- 不改 BC criterion、量化器、搜尋、knee；不 fine-tune；不加 spike 懲罰；不把干預放進正式 pipeline；不加一階項或逐 token 變體。', '',
          '## 資料隔離（已修正）',
          f'- 改用 `wt_manifest_scan.used_wt103_verified`（內容 hash 判定；沒有 hash 的 manifest 必須證明是 WikiText-2，否則報錯）：排除 {len(used)} 篇歷史 WT103 文章，掃描 {len(rows)} 個 manifest。',
          '- 另排除 WikiText-2 文章與近似重複；本輪 discovery／選點用到的文章（token-spike discovery 與確認資料）都在排除清單中。更正紀錄見 correction_summary.json。', '',
          '## 選點與格式',
          f'- 重點：{", ".join(f"{k}（{v}）" for k, v in FOCUS.items())}。L1.E9 是已知案例重現，其餘是推廣檢查，分開報告。',
          f'- 對照：每層（1、2、6）一個弱 spike tap，同層、R_token ≤ p50、最接近 {CTRL_ANCHOR} 的 log rows、優先同 q0 位元類別；允許前幾輪用過的 tap；若該層沒有 ≤ p50 的候選，取該層 R_token 最低者並記錄。只用 discovery 統計，不用任何 loss。',
          f'- 主要格式：{", ".join(FORMATS)}；不跑 int6／p4e1（成本控制）。', '',
          '## spike 定義與條件',
          f'- spike row：s_t ≥ {TAU} × 該 expert 在 token-spike 輪 FP32 discovery 的 median s_t。門檻：{ {k: round(v, 4) for k, v in thr.items()} }。',
          '- 同一輸入、同一 tap 的所有格式與條件使用完全相同的 mask（以 row-id checksum 檢查）。',
          '- Z：tap 完全 FP32；F：全部量化；P：spike rows 保留 FP32；S：只量化 spike rows；R：同數量的隨機一般 rows 保留 FP32。',
          f'- R 的 seeds：{list(R_SEEDS)}；在同一窗口、同一 expert 內抽，數量等於該窗口的 spike rows；不足時用同一 chunk 的其他一般 rows 補足並報告；各 seed 先按窗口平均再做文章統計，不挑 seed。',
          '- 「保護」是把 Q(h) 換回 h；不刪 token、不設為 0。所有條件用原 range，P 不會改善其他 rows 的間距：本輪測的是 spike rows 自身誤差，不是 range 被撐大。對照 tap 只跑 F、P、S。', '',
          '## 測量',
          f'- FP32 背景、一次只干預一個 expert 的 Down 輸入、dynamic route；{N_ART} 篇新 WT103 文章 × {WPA_C} 窗口，凍結後抽取。',
          '- 記錄 rows、spike rows、被保護 rows、補足數、mask checksum、殘餘誤差、range、下游 route 變化、覆蓋文章數。', '',
          '## BC 反事實',
          '- 在 tap 的 BC calibration rows 上，把被保護 rows 的誤差設為 0，重算 S1、S2、mean、variance、bias；n 不變；用原 G2／GS 與原正規化。不從原分數直接減舊占比。T_F 與 cache 的 BC 成本比對。', '',
          '## 主要對比與多重比較',
          '- C1 = NLL(P) − NLL(F)；C2 = NLL(P) − NLL(R̄)；C3 = D_F − D_P − D_S（整體非加性診斷，不等於 Hessian 交叉項）。',
          '- 重現族（L1.E9：C1、C2 × 2 格式 = 4 檢定）與推廣族（其餘 4 個 tap：16 檢定）各自 Holm，FWER 0.05；p 值為雙尾文章 bootstrap。C3 與對照 tap 只報 95% CI。', '',
          '## 判讀類別（預先固定）',
          '- A：C1 < 0 且 Holm 顯著。D：C1 > 0 且 Holm 顯著。C：C3 的 95% CI 排除 0。',
          f'- B：以同一 tap 的 int4 為錨，k = D_F(int4) ÷ T_F(int4)（只有 D_F(int4) 的 95% CI > 0 才定義），E_BC = k·(T_P − T_F)，|E_BC| ≥ {B_MIN_EFFECT}；C1 的 95% CI 完全落在 ±{B_FRACTION}·|E_BC| 內 → BC 預測大變化而實測小且 CI 窄。',
          '- E：A、B、D 都不成立 → 無法判定（不顯著不等於效果小）。BC 與 NLL 尺度不同，不要求數值相等。', '',
          '## 統計與預算',
          f'- {BOOT} 次文章 bootstrap（seed {SEEDS["bootstrap"]}）；窗口與候選不當成獨立樣本。',
          f'- 預算（分鐘）：{BUDGET_MIN}；測量超時就在下一個層組前停止；看到結果後不加資料、不換 seed、不改門檻。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; previous WT103 {len(used)} (verified); eligible {len(elig)}; thresholds {thr}')


# --------------------------------------------------------------------------- #
def random_rows(spike, win, seed, stream, pool_key):
    """boolean mask of non-spike rows matched to the spike rows window by window (numpy arrays); shortfall from the pool."""
    sel = np.zeros(len(spike), bool)
    short = 0
    for w in np.unique(win[spike]):
        k = int((spike & (win == w)).sum())
        cand = np.flatnonzero(~spike & (win == w))
        rng = np.random.default_rng([seed, stream, int(w)])
        take = min(k, len(cand))
        if take:
            sel[rng.choice(cand, take, replace=False)] = True
        short += k - take
    if short:
        rest = np.flatnonzero(~spike & ~sel)
        rng = np.random.default_rng([seed, stream, 10 ** 6 + int(pool_key)])
        sel[rng.choice(rest, min(short, len(rest)), replace=False)] = True
    return sel, short


def select_controls():
    ex = fp32_experts()
    names, *_ = pr_static()
    searched = {n.rsplit('.', 1)[0] for n in names if n.endswith('down_proj')}
    aidx = read(ACT_INDEX)
    q0 = read(Q0K)
    prev = {t['node'] for t in read(PREV_TS / 'selected_taps.json')['taps']} | {t['node'] for t in read(PREV_AO / 'selected_taps.json')['taps']}
    f = lambda r, k: float(r[k]) if r[k] not in ('', 'nan') else float('nan')
    el = [r for r in ex.values() if r['node'] in searched and r['node'] in aidx['row'] and f(r, 'rows_cal') >= tsd.MIN_CAL_ROWS
          and f(r, 'rows') >= tsd.MIN_ALL_ROWS and math.isfinite(f(r, 'R_token')) and math.isfinite(f(r, 'R_channel'))]
    p50 = float(np.quantile([f(r, 'R_token') for r in el], tsd.WEAK_Q))
    bitclass = lambda node: 0 if od.bits(q0[f'{node}.down_proj']['a']) <= 6 else 1 if od.bits(q0[f'{node}.down_proj']['a']) <= 8 else 2
    rng = np.random.default_rng(SEEDS['select'])
    ctrls = []
    for l, anchor in sorted(CTRL_ANCHOR.items()):
        a = ex[anchor]
        cand = [r for r in el if int(r['layer']) == l and f(r, 'R_token') <= p50 and r['node'] not in FOCUS]
        fallback = not cand
        if fallback:
            lay = sorted([r for r in el if int(r['layer']) == l and r['node'] not in FOCUS], key=lambda r: (f(r, 'R_token'), r['node']))
            cand = lay[:1]
        same = [r for r in cand if bitclass(r['node']) == bitclass(anchor)]
        pick = same or cand
        d = np.array([abs(math.log(f(r, 'rows')) - math.log(f(a, 'rows'))) for r in pick])
        best = [pick[i] for i in np.flatnonzero(d == d.min())]
        r = best[int(rng.integers(len(best)))]
        ctrls.append({'node': r['node'], 'layer': l, 'anchor': anchor, 'same_bitclass': bool(same), 'log_rows_diff': float(d.min()),
                      'fallback_lowest_in_layer': fallback, 'candidates_le_p50': 0 if fallback else len(cand), 'previous_round_tap': r['node'] in prev,
                      'R_token': f(r, 'R_token'), 'R_channel': f(r, 'R_channel'), 'rows': f(r, 'rows'), 'rows_cal': f(r, 'rows_cal'),
                      'n_spike': f(r, 'n_spike'), 'q0_act_format': q0[f'{r["node"]}.down_proj']['a']})
    return ctrls, {'eligible': len(el), 'R_token_p50': p50}


def pr_static():
    import run_olmoe_proxy_refresh as pr
    return pr.problem_static()


def bc_T(err, G2, GS, n):
    """BC cost of one slot from per-row errors (rows x channels, float64 torch) with the static sensitivities."""
    S1, S2 = err.sum(0), err.pow(2).sum(0)
    mu = S1 / n
    V = (G2 / n * (S2 - n * mu ** 2)).sum()
    B = (GS * mu ** 2).sum()
    return float(V + B), float(V), float(B)


def prep(out, log):
    """controls (discovery statistics only) and the BC counterfactual per condition on the calibration rows of every tap."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'taps.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    ex = fp32_experts()
    ctrls, pool = select_controls()
    q0 = read(Q0K)
    taps = []
    for node, role in FOCUS.items():
        taps.append({'node': node, 'role': role, 'kind': 'focus', 'conds': list(FOCUS_CONDS)})
    for c in ctrls:
        taps.append({'node': c['node'], 'role': f'control_L{c["layer"]}', 'kind': 'control', 'conds': list(CTRL_CONDS), 'selection': c})
    for i, t in enumerate(taps):
        r = ex[t['node']]
        l, e = node_le(t['node'])
        t.update({'id': i, 'layer': l, 'expert': e, 'threshold': TAU * float(r['s_p50']), 'R_token': float(r['R_token']), 'R_channel': float(r['R_channel']),
                  'disc_rows': int(float(r['rows'])), 'disc_rows_cal': int(float(r['rows_cal'])), 'disc_n_spike': int(float(r['n_spike'])),
                  'q0_act_format': q0[f'{t["node"]}.down_proj']['a']})
    # BC counterfactual on the calibration rows
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    a_, b_ = ss.COLOFF['down_x']
    want = defaultdict(dict)
    for t in taps:
        want[t['layer']][t['expert']] = t
    cap = {}
    with torch.inference_mode():
        cfg, ck = oc.config(), oc.Checkpoint()
        toks = torch.from_numpy(np.load(CALIB)).to('cuda')
        cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
        g0 = ck.global_('cuda', torch.float32)
        h = F.embedding(toks, g0['embed'])
        Hd = cfg.hidden_size
        for l in range(max(want) + 1):
            w = ck.layer(l, 'cuda', torch.float32)
            B = h.shape[0]
            x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
            z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, Hd)
            idx, rw = oc.route(w, z, cfg)

            def cb(e, tok, kpos, gg, uu, hh, l=l):
                if e in want.get(l, {}):
                    cap[(l, e)] = (hh.clone(), tok.clone())
            h = x + tsd.experts_stats(w, z, idx, rw, {}, cb).view(B, SEQ, Hd)
    bcres = {}
    for t in taps:
        node, l, e = t['node'], t['layer'], t['expert']
        hh, tok = cap[(l, e)]
        n = hh.shape[0]
        s = hh.abs().max(1).values
        spike = (s >= t['threshold']).cpu().numpy()
        win = (tok // SEQ).cpu().numpy()
        rmasks = [random_rows(spike, win, sd, 0, 0) for sd in R_SEEDS]
        G2 = torch.from_numpy(np.asarray(gc[gidx['row'][node]])[0, a_:b_].astype(np.float64)).to('cuda')
        GS = torch.from_numpy(np.asarray(gc[gidx['row'][node]])[1, a_:b_].astype(np.float64)).to('cuda')
        row, n_c = np.asarray(ac[aidx['row'][node]]), int(aidx['n'][node])
        rng = ranges[f'{node}.down_proj']['x']
        res = {'n_rows': n, 'n_cache': n_c, 'spike_rows': int(spike.sum()), 'random_shortfall': [r[1] for r in rmasks], 'formats': {}}
        sp_t = torch.from_numpy(spike).to('cuda')
        for f in FORMATS:
            err = (dq.act(hh, f, rng) - hh).double()
            j = QUANT.index(f)
            S1c, S2c = row[0][j][a_:b_].astype(np.float64), row[1][j][a_:b_].astype(np.float64)
            muc = S1c / n_c
            T_cache = float((G2.cpu().numpy() / n_c * (S2c - n_c * muc ** 2)).sum() + (GS.cpu().numpy() * muc ** 2).sum())
            d = {'T_cache': T_cache}
            for cond in ('F', 'P', 'S') + tuple(f'R{i}' for i in range(len(R_SEEDS))):
                e_ = err.clone()
                if cond == 'P':
                    e_[sp_t] = 0
                elif cond == 'S':
                    e_[~sp_t] = 0
                elif cond.startswith('R'):
                    e_[torch.from_numpy(rmasks[int(cond[1:])][0]).to('cuda')] = 0
                T, V, Bv = bc_T(e_, G2, GS, n)
                d[cond] = {'T': T, 'variance': V, 'bias': Bv}
            d['T_F_vs_cache_rel'] = abs(d['F']['T'] - T_cache) / max(abs(T_cache), 1e-300)
            d['R_mean_T'] = float(np.mean([d[f'R{i}']['T'] for i in range(len(R_SEEDS))]))
            d['nonadditivity'] = d['F']['T'] - d['P']['T'] - d['S']['T']
            res['formats'][f] = d
        bcres[node] = res
    write(out / 'taps.json', {'taps': taps, 'controls_pool': pool})
    write(out / 'bc_counterfactual.json', bcres)
    log_stage(out, 'prep', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(json.dumps(dr.clean({'taps': [(t['id'], t['node'], t['role'], round(t['threshold'], 3)) for t in taps],
                             'bc': {k: {f: {c: round(v[c]['T'], 4) if isinstance(v.get(c), dict) else None for c in ('F', 'P', 'S', 'R0')} | {'rel': v['T_F_vs_cache_rel']}
                                        for f, v in r['formats'].items()} for k, r in bcres.items()}})))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    files = ('plan.json', 'taps.json', 'bc_counterfactual.json', 'correction_summary.json')
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'code': code_hashes(),
                                'note': 'taps, thresholds, formats, conditions, seeds and BC counterfactuals frozen before any confirmation article is drawn'})
    log('frozen')


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    if len(used) != pl['isolation']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    lines = mr.load_lines(mr.WT103)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['isolation']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    by = {a['ch']: a for a in w103}
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(by[h]['body_norm']) for h in used]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    segs, toks, acc_sh, skips = [], [], [], Counter()
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        ids = tok(text, add_special_tokens=False)['input_ids']
        if len(ids) < WPA_C * SEQ:
            skips['short'] += 1; continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if max(c2, cp, ca) >= mr.NEARDUP:
            skips['near_duplicate'] += 1; continue
        st = int(rs.integers(0, len(ids) - WPA_C * SEQ + 1))
        w = np.array(ids[st:st + WPA_C * SEQ], dtype=np.int64).reshape(WPA_C, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'title': a['title'], 'content_sha256': a['ch'], 'article_tokens': len(ids), 'window_tokens': [st, st + WPA_C * SEQ],
                     'source': 'wikitext-103-raw-v1 train', 'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca},
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
        if len(segs) == N_ART:
            break
    if len(segs) < N_ART or {s['content_sha256'] for s in segs} & used:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'), 'n': N_ART, 'windows_per_article': WPA_C,
                                     'skips': dict(skips), 'visited': N_ART + sum(skips.values()), 'excluded_previous_wt103': len(prev),
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json')})
    log_stage(out, 'data', time.perf_counter() - t0)
    log(f'data: {N_ART} articles; skips {dict(skips)}; excluded previous {len(prev)}')


# --------------------------------------------------------------------------- #
def experts_tok(w, x2d, idx, rw, l, tokhook, E=NE):
    """olmoe_common.experts_fixed with no QDQ hooks (FP32), same op order; tokhook[(l, e)](h, tok) replaces the Down input of one expert."""
    import torch
    import torch.nn.functional as F
    out = torch.zeros_like(x2d)
    mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist()
    for e in hit:
        if e == E:
            continue
        kpos, tok = torch.where(mask[e])
        xe = x2d[tok]
        g = F.linear(xe, w['gate_proj'][e])
        u = F.linear(xe, w['up_proj'][e])
        h = F.silu(g) * u
        if (l, e) in tokhook:
            h = tokhook[(l, e)](h, tok)
        d = F.linear(h, w['down_proj'][e])
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_step_tok(ctx, w, h, l, tokhook):
    import olmoe_common as oc
    B = h.shape[0]
    Hd = ctx.cfg.hidden_size
    x = h + oc.attention(w, oc.rms(h, w['ln1'], ctx.cfg.rms_norm_eps), ctx.cfg, ctx.cos, ctx.sin)
    z = oc.rms(x, w['ln2'], ctx.cfg.rms_norm_eps).reshape(-1, Hd)
    idx, rw = oc.route(w, z, ctx.cfg)
    return x + experts_tok(w, z, idx, rw, l, tokhook).view(B, SEQ, Hd), idx


def variants_from_tok(ctx, toks_np, start_l, H_in_l, base_ids, chunks, variants, cur):
    """od.variants_from with token-aware hooks; cur['off'] holds the first window of the chunk being run."""
    import torch
    from run_olmoe_fixed_route import head_ce
    H = {name: [x.clone() for x in H_in_l] for name, _ in variants}
    changed = {name: [0, 0] for name, _ in variants}
    with torch.no_grad():
        for l in range(start_l, NL):
            w = ctx.ck.layer(l, 'cuda', torch.float32)
            for name, hooks in variants:
                th = hooks.get(l, {})
                for ci, (a, b) in enumerate(chunks):
                    cur['off'] = a
                    hn, idx = layer_step_tok(ctx, w, H[name][ci].to('cuda'), l, th)
                    H[name][ci] = hn.cpu()
                    if l > start_l:
                        bi = base_ids[l][a:b].to('cuda').view(-1, K)
                        ch = (torch.sort(idx, -1).values != torch.sort(bi.long(), -1).values).any(-1)
                        changed[name][0] += int(ch.sum()); changed[name][1] += ch.numel()
            del w
        ce = {}
        for name, _ in variants:
            ce[name] = []
            for ci, (a, b) in enumerate(chunks):
                ce[name] += head_ce(ctx.g, H[name][ci].to('cuda'), torch.from_numpy(toks_np[a:b]).to('cuda'), ctx.cfg).cpu().numpy().tolist()
    return ce, {k: (v[0] / v[1] if v[1] else 0.0) for k, v in changed.items()}


def make_hook(dq, f, rng, thr, cond, cur, acc):
    """Down-input replacement for one condition; the spike mask comes from the FP32 input h itself."""
    import torch

    def fn(h, tok):
        s = h.abs().max(1).values
        spike = s >= thr
        gid = (cur['off'] * SEQ + tok).long()
        acc['rows'] += h.shape[0]; acc['spike_rows'] += int(spike.sum())
        acc['spike_checksum'] += int(gid[spike].sum())
        acc['spike_windows'].update((gid[spike] // SEQ).cpu().numpy().tolist())
        acc['range'] = [float(rng[0]), float(rng[1])]
        if cond == 'identity':
            return h
        q = dq.act(h, f, rng)
        if cond == 'F':
            out = q
            prot = torch.zeros_like(spike)
        elif cond == 'P':
            prot = spike
            out = torch.where(prot[:, None], h, q)
        elif cond == 'S':
            prot = ~spike
            out = torch.where(prot[:, None], h, q)
        else:
            sd = R_SEEDS[int(cond[1:])]
            win = (gid // SEQ).cpu().numpy()
            sel, short = random_rows(spike.cpu().numpy(), win, sd, 1, cur['off'])
            prot = torch.from_numpy(sel).to(h.device)
            acc['shortfall'] += short
            acc['random_checksum'] += int(gid[prot].sum())
            out = torch.where(prot[:, None], h, q)
        acc['protected_rows'] += int(prot.sum())
        r2 = (out - h).double().pow(2).sum(1)
        acc['res_spike'] += float(r2[spike].sum()); acc['res_other'] += float(r2[~spike].sum())
        acc['res_protected'] += float(r2[prot].sum())
        return out
    return fn


def measure(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'measure.json').exists():
        raise SystemExit('exists')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('data changed')
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    taps = read(out / 'taps.json')['taps']
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ctx = od.Ctx(None)
    ce0, H_in, ids, chunks = od.base_pass(ctx, toks, dq, log, 'fp32 base (Z)')
    ref = dr.forward(toks, None, dq, ctx.ranges, record=False, log=lambda m: None, label='dr fp32')
    controls = {'Z_equals_dr_forward_max_abs': float(np.max(np.abs(np.array(ce0) - np.array(ref['ce']))))}
    res = {'base_ce': ce0, 'variants': {}, 'acc': {}, 'route_changed': {}}
    by_layer = defaultdict(list)
    for tp in taps:
        by_layer[tp['layer']].append(tp)
    cur = {'off': 0}
    stopped = False
    for l in sorted(by_layer):
        if time.perf_counter() - t0 > 60 * BUDGET_MIN['measure']:
            stopped = True
            log(f'budget exceeded before layer {l}: stop')
            break
        variants, accs = [], {}
        for tp in by_layer[l]:
            rng = ctx.ranges[f'{tp["node"]}.down_proj']['x']
            conds = (['identity'] if tp == by_layer[l][0] else []) + [(f, c) for f in FORMATS for c in tp['conds']]
            for item in conds:
                f, c = ('-', 'identity') if item == 'identity' else item
                name = f'tap{tp["id"]}|{f}|{c}'
                accs[name] = {'rows': 0, 'spike_rows': 0, 'spike_checksum': 0, 'protected_rows': 0, 'shortfall': 0, 'random_checksum': 0,
                              'res_spike': 0.0, 'res_other': 0.0, 'res_protected': 0.0, 'range': None, 'spike_windows': set()}
                variants.append((name, {l: {(l, tp['expert']): make_hook(dq, f, rng, tp['threshold'], c, cur, accs[name])}}))
        for b in range(0, len(variants), VAR_BATCH):
            part = variants[b:b + VAR_BATCH]
            ce, chg = variants_from_tok(ctx, toks, l, H_in[l], ids, chunks, part, cur)
            for name, _ in part:
                res['variants'][name] = ce[name]
                res['route_changed'][name] = chg[name]
                res['acc'][name] = {**accs[name], 'spike_windows': sorted(accs[name]['spike_windows'])}
            log(f'measure: layer {l} batch {b // VAR_BATCH} ({time.perf_counter() - t0:.0f}s)')
        log(f'measure: layer {l} done ({time.perf_counter() - t0:.0f}s, {len(variants)} variants)')
    ident = {k: float(np.max(np.abs(np.array(v) - np.array(ce0)))) for k, v in res['variants'].items() if k.endswith('|identity')}
    controls['identity_max_abs'] = ident
    chk = defaultdict(set)
    for k, a in res['acc'].items():
        tp = k.split('|')[0]
        chk[tp].add((a['rows'], a['spike_rows'], a['spike_checksum']))
    controls['same_spike_mask_all_conditions'] = all(len(v) == 1 for v in chk.values())
    rchk = defaultdict(set)
    for k, a in res['acc'].items():
        p_ = k.split('|')
        if p_[2].startswith('R'):
            rchk[(p_[0], p_[2])].add(a['random_checksum'])
    controls['same_random_rows_across_formats'] = all(len(v) == 1 for v in rchk.values())
    want_prot = lambda k, a: (a['spike_rows'] if k.endswith('|P') or '|R' in k else a['rows'] - a['spike_rows'] if k.endswith('|S') else 0)
    controls['protected_rows_ok'] = all(a['protected_rows'] == want_prot(k, a) for k, a in res['acc'].items())
    controls['residual_zero_on_protected'] = all(a['res_protected'] == 0.0 for a in res['acc'].values())
    controls['single_range_per_tap'] = all(len({tuple(a['range']) for k, a in res['acc'].items() if k.startswith(f'tap{tp["id"]}|') and a['range']}) == 1 for tp in taps)
    controls['pass'] = (controls['Z_equals_dr_forward_max_abs'] == 0 and all(v == 0 for v in ident.values()) and controls['same_spike_mask_all_conditions']
                        and controls['same_random_rows_across_formats'] and controls['protected_rows_ok'] and controls['residual_zero_on_protected']
                        and controls['single_range_per_tap'])
    res['controls'] = controls
    res['stopped_early'] = stopped
    write(out / 'measure.json', res)
    log_stage(out, 'measure', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'stopped_early': stopped})
    log(json.dumps(dr.clean({'controls': controls, 'variants': len(res['variants']), 'stopped': stopped})))


# --------------------------------------------------------------------------- #
def article_vec(ce):
    return np.array(ce).reshape(N_ART, WPA_C).sum(1) / (WPA_C * TP)


def boot(d, idx):
    return d[idx].mean(1)


def pval(b):
    return max(1.0 / (BOOT + 1), min(1.0, 2 * min(float((b <= 0).mean()), float((b >= 0).mean()))))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for r, i in enumerate(order):
        if ps[i] <= ALPHA / (m - r):
            rej[i] = True
        else:
            break
    return rej.tolist()


def analyze(out, log):
    t0 = time.perf_counter()
    taps = read(out / 'taps.json')['taps']
    bc = read(out / 'bc_counterfactual.json')
    m = read(out / 'measure.json')
    segs = read(out / 'segments.json')
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
    z = article_vec(m['base_ce'])
    ci = lambda b: np.percentile(b, [2.5, 97.5]).tolist()
    rows, tests = [], []
    for tp in taps:
        for f in FORMATS:
            key = lambda c: f'tap{tp["id"]}|{f}|{c}'
            if not all(key(c) in m['variants'] for c in tp['conds']):
                continue
            D = {c: article_vec(m['variants'][key(c)]) - z for c in tp['conds']}
            if tp['kind'] == 'focus':
                D['R'] = np.mean([D[f'R{i}'] for i in range(len(R_SEEDS))], 0)
            r = {'tap': tp['id'], 'node': tp['node'], 'role': tp['role'], 'kind': tp['kind'], 'format': f}
            for c, v in D.items():
                r[f'D_{c}'] = float(v.mean()); r[f'D_{c}_ci95'] = ci(boot(v, idx))
            con = {'C1': D['P'] - D['F'], 'C3': D['F'] - D['P'] - D['S']}
            if 'R' in D:
                con['C2'] = D['P'] - D['R']
            for c, v in con.items():
                b = boot(v, idx)
                r[c] = float(v.mean()); r[f'{c}_ci95'] = ci(b); r[f'{c}_p'] = pval(b)
            if tp['kind'] == 'focus':
                for c in ('C1', 'C2'):
                    tests.append({'family': 'replication' if tp['role'] == 'known_case' else 'generalization', 'row': len(rows), 'contrast': c, 'p': r[f'{c}_p']})
            a = m['acc'][key('F')]
            r['coverage'] = {'rows': a['rows'], 'spike_rows': a['spike_rows']}
            r['route_changed'] = {c: m['route_changed'][key(c)] for c in tp['conds']}
            r['shortfall'] = {c: m['acc'][key(c)]['shortfall'] for c in tp['conds'] if c.startswith('R')}
            b_ = bc[tp['node']]['formats'][f]
            r['bc'] = {'T_cache': b_['T_cache'], 'T_F': b_['F']['T'], 'T_P': b_['P']['T'], 'T_S': b_['S']['T'], 'T_R': b_.get('R_mean_T'),
                       'T_F_vs_cache_rel': b_['T_F_vs_cache_rel'], 'nonadditivity': b_['nonadditivity'],
                       'bias_F': b_['F']['bias'], 'bias_P': b_['P']['bias'], 'bias_S': b_['S']['bias']}
            r['bc']['P_over_F'] = r['bc']['T_P'] / r['bc']['T_F'] if r['bc']['T_F'] else None
            r['bc']['S_over_F'] = r['bc']['T_S'] / r['bc']['T_F'] if r['bc']['T_F'] else None
            r['bc']['R_over_F'] = (r['bc']['T_R'] / r['bc']['T_F']) if r['bc']['T_F'] and r['bc']['T_R'] is not None else None
            rows.append(r)
    for fam in ('replication', 'generalization'):
        ts = [t for t in tests if t['family'] == fam]
        rej = holm([t['p'] for t in ts])
        for t, rj in zip(ts, rej):
            rows[t['row']][f'{t["contrast"]}_holm_reject'] = rj
            rows[t['row']][f'{t["contrast"]}_family'] = fam
    # coverage per tap on the confirmation data (articles with a spike row): from the F variant of each format
    cov = {}
    for tp in taps:
        a = m['acc'].get(f'tap{tp["id"]}|{FORMATS[0]}|F')
        cov[tp['node']] = {'rows': a['rows'] if a else None, 'spike_rows': a['spike_rows'] if a else None}
    # categories and BC comparison
    for r in rows:
        if r['kind'] != 'focus':
            continue
        cats = []
        if r.get('C1_holm_reject') and r['C1'] < 0:
            cats.append('A')
        if r.get('C1_holm_reject') and r['C1'] > 0:
            cats.append('D')
        i4 = next(x for x in rows if x['tap'] == r['tap'] and x['format'] == 'int4')
        if i4['D_F_ci95'][0] > 0 and i4['bc']['T_F'] > 0:
            k = i4['D_F'] / i4['bc']['T_F']
            E = k * (r['bc']['T_P'] - r['bc']['T_F'])
            r['B'] = {'k': k, 'E_BC': E, 'band': B_FRACTION * abs(E), 'evaluable': abs(E) >= B_MIN_EFFECT,
                      'ratio_C1_to_E_BC': r['C1'] / E if E else None}
            r['B']['holds'] = bool(r['B']['evaluable'] and -r['B']['band'] <= r['C1_ci95'][0] and r['C1_ci95'][1] <= r['B']['band'])
            if r['B']['holds']:
                cats.append('B')
        else:
            r['B'] = {'evaluable': False, 'reason': 'D_F(int4) 95% CI not above 0 at this tap: no int4 anchor'}
        if r['C3_ci95'][0] > 0 or r['C3_ci95'][1] < 0:
            cats.append('C')
        if not any(c in cats for c in ('A', 'B', 'D')):
            cats.append('E')
        r['categories'] = cats
        r['bc_vs_measured'] = {'direction_P_vs_F': {'bc': 'P lower' if r['bc']['T_P'] < r['bc']['T_F'] else 'P higher',
                                                    'measured': 'P lower' if r['C1'] < 0 else 'P higher', 'measured_decisive': r['C1_ci95'][1] < 0 or r['C1_ci95'][0] > 0},
                               'P_more_useful_than_R': {'bc': r['bc']['T_R'] is not None and r['bc']['T_P'] < r['bc']['T_R'],
                                                        'measured': r['C2'] < 0, 'measured_decisive': r['C2_ci95'][1] < 0 or r['C2_ci95'][0] > 0},
                               'nonadditivity_sign': {'bc': float(np.sign(r['bc']['nonadditivity'])), 'measured': float(np.sign(r['C3'])),
                                                      'measured_decisive': r['C3_ci95'][0] > 0 or r['C3_ci95'][1] < 0}}
    # format order under F and under P (int4 vs p6e1) per tap
    order = {}
    for tp in taps:
        rr = {x['format']: x for x in rows if x['tap'] == tp['id']}
        if len(rr) < 2:
            continue
        o = {}
        for c in ('F', 'P'):
            d = article_vec(m['variants'][f'tap{tp["id"]}|int4|{c}']) - article_vec(m['variants'][f'tap{tp["id"]}|p6e1|{c}'])
            b = boot(d, idx)
            o[c] = {'measured_int4_minus_p6e1': float(d.mean()), 'ci95': ci(b), 'decisive': ci(b)[0] > 0 or ci(b)[1] < 0,
                    'bc_int4_minus_p6e1': rr['int4']['bc'][f'T_{c}'] - rr['p6e1']['bc'][f'T_{c}']}
            o[c]['agree'] = (o[c]['measured_int4_minus_p6e1'] > 0) == (o[c]['bc_int4_minus_p6e1'] > 0)
        order[tp['node']] = o
    res = {'rows': rows, 'format_order': order, 'coverage_confirm': cov, 'articles': N_ART,
           'article_coverage': article_cov(m, taps), 'measure_controls': m['controls'], 'stopped_early': m['stopped_early'],
           'bc_reproduces_cache': {tp['node']: {f: bc[tp['node']]['formats'][f]['T_F_vs_cache_rel'] for f in FORMATS} for tp in taps},
           'bc_rows_equal_cache': {tp['node']: bc[tp['node']]['n_rows'] == bc[tp['node']]['n_cache'] for tp in taps}}
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    with open(out / 'contrasts.csv', 'w') as fh:
        keys = ['tap', 'node', 'role', 'format', 'D_F', 'D_P', 'D_S', 'D_R', 'C1', 'C1_ci95', 'C1_p', 'C1_holm_reject', 'C2', 'C2_ci95', 'C2_p', 'C2_holm_reject',
                'C3', 'C3_ci95', 'categories']
        wcsv = csv.writer(fh); wcsv.writerow(keys + ['T_F', 'T_P', 'T_S', 'T_R'])
        for r in rows:
            wcsv.writerow([r.get(k) for k in keys] + [r['bc']['T_F'], r['bc']['T_P'], r['bc']['T_S'], r['bc']['T_R']])
    with open(out / 'per_window.csv', 'w') as fh:
        keys = ['base'] + sorted(m['variants'])
        wcsv = csv.writer(fh); wcsv.writerow(['window', 'article'] + keys)
        for i in range(len(m['base_ce'])):
            wcsv.writerow([i, i // WPA_C, m['base_ce'][i]] + [m['variants'][k][i] for k in keys[1:]])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps(dr.clean([{k: r.get(k) for k in ('node', 'format', 'D_F', 'C1', 'C1_ci95', 'C1_holm_reject', 'C2', 'C2_ci95', 'C3', 'C3_ci95', 'categories')}
                             for r in rows])))


def article_cov(m, taps):
    """rows, spike rows, windows and articles with at least one spike row at each tap (confirmation data, F condition)."""
    res = {}
    for tp in taps:
        a = m['acc'].get(f'tap{tp["id"]}|{FORMATS[0]}|F')
        if a:
            res[tp['node']] = {'rows': a['rows'], 'spike_rows': a['spike_rows'], 'spike_windows': len(a['spike_windows']),
                               'spike_articles': len({w // WPA_C for w in a['spike_windows']})}
    return res


PAL = tsd.PAL


def figures(out, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    fd = out / 'figures'
    fd.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'axes.edgecolor': PAL['muted'], 'axes.labelcolor': PAL['text'],
                         'xtick.color': PAL['text2'], 'ytick.color': PAL['text2'], 'text.color': PAL['text'], 'axes.grid': True, 'grid.color': PAL['grid'],
                         'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'lines.linewidth': 2})
    an = read(out / 'analysis.json')
    rows = an['rows']
    series = (('D_F', 'F all quantized', PAL['s1']), ('D_P', 'P spike rows FP32', PAL['s2']), ('D_S', 'S only spike rows', PAL['s3']),
              ('D_R', 'R random rows FP32', '#eda100'))
    fig, axs = plt.subplots(1, len(FORMATS), figsize=(11, 0.42 * len(rows) / len(FORMATS) + 1.6), sharey=True)
    for ax, f in zip(axs, FORMATS):
        rr = [r for r in rows if r['format'] == f]
        for i, r in enumerate(rr):
            for j, (k, lab, col) in enumerate(series):
                if k not in r:
                    continue
                y = i + (j - 1.5) * 0.17
                lo, hi = r[f'{k}_ci95']
                ax.plot([lo, hi], [y, y], color=col, lw=2)
                ax.scatter([r[k]], [y], color=col, s=18, zorder=3, edgecolors=PAL['surface'], linewidths=1, label=lab if i == 0 else None)
        ax.axvline(0, color=PAL['text2'], lw=0.8)
        ax.set_yticks(range(len(rr))); ax.set_yticklabels([f'{r["node"]} ({r["role"]})' for r in rr], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('dNLL vs Z (nat/token), 95% CI'); ax.set_title(f.upper() if f.startswith('int') else f, loc='left', fontsize=10)
    hs, ls = axs[0].get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, fontsize=8, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(fd / 'conditions.png', dpi=150); plt.close(fig)
    fr = [r for r in rows if r['kind'] == 'focus']
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lab = [f'{r["node"]} {r["format"]}' for r in fr]
    x = np.arange(len(fr))
    ax.bar(x - 0.2, [1 - r['bc']['P_over_F'] for r in fr], 0.38, color=PAL['s1'], label='BC: share of T_F removed by P')
    meas = [(-r['C1'] / r['D_F']) if r['D_F'] > 0 and r['D_F_ci95'][0] > 0 else np.nan for r in fr]
    ax.bar(x + 0.2, meas, 0.38, color=PAL['s2'], label='measured: share of D_F removed by P (D_F CI > 0 only)')
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=40, ha='right', fontsize=8); ax.axhline(0, color=PAL['text2'], lw=0.8)
    ax.set_title('Protecting spike rows: BC vs measured (relative, different scales)', loc='left', fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / 'bc_vs_measured.png', dpi=150); plt.close(fig)
    log_stage(out, 'figures', time.perf_counter() - t0)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['correct_prev', 'plan', 'prep', 'freeze', 'data', 'measure', 'analyze', 'figures'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage not in ('plan', 'correct_prev') and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
