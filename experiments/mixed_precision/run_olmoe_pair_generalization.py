#!/usr/bin/env python3
"""OLMoE: do the original BC (T_old) and the position-paired score (T_pair) rank candidate formats differently, and stably, on
Down-input experts NOT chosen from known misranks? Pre-registered in plan.json / plan.md.

Scope: the Down input of single experts only (no Gate / Up shared-input attribution). Two predictors only:
  T_old  = sum_c (sum_t g^2) / n * sum_t r^2 + B       (the BC variance term + the BC bias term)
  T_pair = sum_c sum_t g^2 r^2 + B                       (same quantities paired by position), r = e - mu_c, B = sum_c GS_c mu_c^2.
Fixed: BC calibration, loss, gradient normalization, router semantics, quantizer, ranges, candidate formats; no proxy refresh, no
tuning, no spike penalty, no first-order term; search, knee and pipeline untouched; no NLL rescaling of either proxy.

Stages (one --out-dir): plan, select, grad, mech, freeze, data, measure, analyze, figures.
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
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss
import run_olmoe_spike_intervention as si
import run_olmoe_token_spike_diag as tsd
import wt_manifest_scan as wms

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
TS = RUNS / 'olmoe_token_spike_diag_20261023'
RANGES, CALIB = ss.RANGES, ss.CALIB
ACT_CACHE, ACT_INDEX, GC, GC_INDEX, Q0K = od.ACT_CACHE, od.ACT_INDEX, od.GC, od.GC_INDEX, od.Q0K
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
CANDS = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')
SAME_BIT = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
MIN_CAL_ROWS, MIN_ALL_ROWS = 64, 512
N_GENERAL, N_STRONG, STRONG_Q, WEAK_Q = 16, 4, 0.90, 0.50
TAU = tsd.TAU
BC_TIE, BOOT, ALPHA_GROUP, MIN_DECISIVE = 0.02, 10000, 0.025, 8
N_ART, WPA_C, CHUNK, VAR_BATCH = 96, 3, 128, 12
SEEDS = {'general': 23601, 'spike': 23602, 'order': 23611, 'start': 23612, 'bootstrap': 23621, 'expert_bootstrap': 23622}
BUDGET_MIN = {'select': 2, 'grad': 8, 'mech': 10, 'data': 5, 'measure': 32, 'analyze': 5, 'figures': 3, 'total_gpu': 42}
MEASURED_SOURCES = {  # experts with damage or intervention measurements in earlier rounds (excluded from the new cases)
    'olmoe_fixed_route_20261015': ('candidates.json', 'node'), 'olmoe_first_order_validation_20261017': ('candidates.json', 'node'),
    'olmoe_calibration_stability_20261018': ('candidates.json', 'node'), 'olmoe_crosstext_20261017': ('pairs.csv', 'node'),
    'olmoe_act_outlier_diag_20261023': ('selected_taps.json', 'taps'), 'olmoe_token_spike_diag_20261023': ('selected_taps.json', 'taps'),
    'olmoe_spike_intervention_20261023': ('taps.json', 'taps'), 'olmoe_pair_gradient_diag_20261023': ('taps.json', 'taps'),
    'olmoe_up_case_20261017': ('regex', None), 'olmoe_bc_numerics_20261016': ('regex', None)}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_pair_generalization.py', 'run_olmoe_pair_gradient_diag.py', 'run_olmoe_spike_intervention.py',
                                                  'wt_manifest_scan.py', 'run_olmoe_token_spike_diag.py', 'run_olmoe_act_outlier_diag.py'] + ss.CODE + dr.SCRIPTS))}


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def measured_experts():
    import re
    import glob
    res = {}
    for d, (f, key) in MEASURED_SOURCES.items():
        p = RUNS / d
        if f == 'regex':
            txt = ''.join(Path(x).read_text() for x in glob.glob(str(p / '*.json')) + glob.glob(str(p / 'report.md')))
            nodes = set(re.findall(r'\bL\d{1,2}\.E\d{1,2}\b', txt))
        elif f.endswith('.csv'):
            nodes = {r[key] for r in csv.DictReader(open(p / f))}
        else:
            d_ = read(p / f)
            nodes = {t['node'] for t in (d_[key] if key == 'taps' else d_)} if key == 'taps' else {c[key] for c in d_}
        for n in nodes:
            res.setdefault(n, []).append(d)
    return {n: sorted(v) for n, v in sorted(res.items())}


def pairs_of(cands=CANDS):
    fam = lambda f: 'int' if f.startswith('int') else 'posit'
    out = []
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            out.append({'a': a, 'b': b, 'family': 'int-int' if fam(a) == fam(b) == 'int' else 'posit-posit' if fam(a) == fam(b) == 'posit' else 'cross',
                        'same_bit': (a, b) in SAME_BIT})
    return out


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    excl = measured_experts()
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [ACT_CACHE, ACT_INDEX, GC, GC_INDEX, RANGES, CALIB, Q0K, TS / 'discovery_experts.csv']
              + mr.WT103 + mr.WT2}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'on Down-input experts not selected from known misranks, do T_old (BC) and T_pair (position-paired) rank the candidate formats '
                    'differently and stably; scope and regression risk, no further tuning on old cases',
        'scope': 'single-expert Down input (down_proj x slot) only; no Gate / Up shared-input attribution; conclusions limited to this scope',
        'methods': {'T_old': 'sum_c G2_c / n sum_t r_tc^2 + sum_c GS_c mu_c^2 (BC; equals G2/n (S2 - n mu^2) + GS mu^2)',
                    'T_pair': 'sum_c sum_t g_tc^2 r_tc^2 + the same bias term', 'r': 'e - mu_c, mu_c = sum_t e_tc / n, n = routed calibration rows',
                    'fixed': 'BC calibration (10 x 128, WikiText-2 validation), sum-loss CE, routes and routing weights constant in backward (no '
                             'router / top-k gradient), G2 / GS as the static cache, quantizer dev_quant(corrected=True), ranges.json, the 6 '
                             'candidates; no refresh, no tuning, no spike penalty, no first-order term, no NLL rescaling; search / knee / '
                             'pipeline untouched'},
        'exclusion': {'measured_experts': excl, 'count': len(excl), 'rule': 'experts with damage or intervention measurements in earlier rounds '
                      '(single-expert candidates, pairs, taps of the diagnostic rounds) are not new cases'},
        'eligibility': {'rule': f'searched expert with Down-input cache rows (act-error and gradient caches); calibration rows >= {MIN_CAL_ROWS}; '
                                f'token-spike discovery rows >= {MIN_ALL_ROWS}; finite R_token and s_p50 > 0; finite Down-input range with hi > lo; '
                                'finite cached S1 / S2 (6 candidates) and G2 / GS; not in the exclusion list',
                        'no_outcome_info': 'no loss, BC misrank or T_pair information is used'},
        'selection': {'general': f'{N_GENERAL} experts: one per layer drawn uniformly from the eligible new experts of that layer (seed '
                                 f'{SEEDS["general"]}, layers in order); a layer without eligible experts gets none and the missing slots are '
                                 'drawn uniformly from the remaining eligible pool (recorded)',
                      'spike': f'{N_STRONG} strong (R_token >= p{int(100 * STRONG_Q)} of the eligible set) drawn in random order (seed '
                               f'{SEEDS["spike"]}), at most one per layer, each matched to a weak expert (R_token <= p{int(100 * WEAK_Q)}) of the '
                               'same layer with the nearest |log discovery rows| (nearest layer if none; ties at random); disjoint from the '
                               'general sample; matching shortfalls listed', 'reporting': 'groups reported separately; pooled numbers are '
                                                                                            'descriptive only, not a whole-model average'},
        'formats': list(CANDS), 'pairs': {'primary': [list(p) for p in SAME_BIT], 'secondary': 'the other 12 of the 15 candidate pairs'},
        'controls': ['FP32 forward of the gradient path equals dr.forward on the calibration (CE)', 'G2 / GS from per-element gradients equal the '
                     'static cache', 'S1 / S2 from per-element errors equal the act-error cache (6 candidates)', 'T_old reproduces the cached BC cost',
                     'identity V_pair,c - V_old,c = n Cov_t(g^2, r^2)', 'streaming accumulators sum g^2 e^2, sum g^2 e, sum g^2 reproduce the '
                     'per-element V_pair', 'any mismatch explained before interpretation'],
        'mechanism': {'per_tap_format': 'V_pair / V_old; spike (s_t >= 6 x discovery median) vs other rows: shares of g^2, r^2, V_old, V_pair and '
                                        'covariance contributions; main channel (largest V_old,c) vs other channels', 'use': 'frozen before the '
                      'confirmation data; post-hoc links to outcomes are descriptive and may not tune the formula or define selection rules'},
        'confirmation': {'data': f'{N_ART} new WikiText-103 train articles x {WPA_C} windows of {SEQ}, drawn after the freeze (seeds order '
                                 f'{SEEDS["order"]}, start {SEEDS["start"]}); wt_manifest_scan.used_wt103_verified excludes {len(used)} earlier '
                                 'articles (all historical, discovery and confirmation manifests); WikiText-2 (calibration) and near duplicates '
                                 'excluded', 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                         'measurement': 'FP32 background, one expert Down input quantized with one candidate (6 per expert), other weights and '
                                        'activations FP32, normal dynamic route; dNLL vs FP32 per article; layers before the tap reused from the '
                                        'FP32 pass (bitwise control); at most 12 variants per layer pass', 'no_search': 'no search, no knee'},
        'evaluation': {'measured_pair': 'article-level paired bootstrap of NLL(a) - NLL(b); decisive = 95% CI excludes 0',
                       'strict_sensitivity': '99% CI; and Holm within group over the primary pairs (bootstrap p)',
                       'proxy': 'a proxy is right on a decisive pair if its sign matches; a proxy tie (|dT| / max T < 0.02) counts as not right; '
                                'both proxies on exactly the same pairs', 'classes': 'both right, fix (T_pair right, T_old not), break (T_old right, '
                                'T_pair not), both not right', 'undeterminable': 'pairs with CI including 0 never count as right; reported with coverage',
                       'dependence': 'pairs of one expert are not independent; expert-equal summaries and expert bootstrap',
                       'primary_metric': f'per group on decisive same-bit pairs: net = fixes - breaks, expert bootstrap (seed '
                                         f'{SEEDS["expert_bootstrap"]}) {100 * (1 - ALPHA_GROUP):.1f}% CI (Bonferroni over the 2 groups)',
                       'group_status': f'insufficient if fewer than {MIN_DECISIVE} decisive same-bit pairs; improve if net > 0 and the CI lower '
                                       'bound > 0; worse if net < 0 and the CI upper bound < 0; else unchanged',
                       'negative_dnll': 'kept and reported separately'},
        'verdict': {'D': 'both groups insufficient: evidence insufficient, no equivalence claim, no extra data',
                    'A': 'both groups improve and, over all 15 pairs, breaks <= fixes / 2 in each group: supports wider validation',
                    'B': 'the spike group improves, the general group does not (unchanged, worse or insufficient): local study only, no BC '
                         'replacement, no post-hoc selection rule', 'C': 'otherwise when at least one group is sufficient (fixes ~ breaks, '
                         'worse, or both improve with many all-pair breaks): keep BC, previous round = local mechanism explanation',
                    'G_only': 'only the general group improves: reported as such (not covered by the rules)',
                    'always': 'no Hessian claim, no knee claim, pipeline unchanged'},
        'costs': 'BC statistics (forward, backward, G2 / GS, S1 / S2), added pairing statistics (streaming accumulators) time and memory, storage, '
                 'final loss evaluation time; whole-model numbers are labelled estimates',
        'statistics': {'bootstrap': f'{BOOT} article bootstraps (seed {SEEDS["bootstrap"]})', 'stop': 'if the measurement exceeds its budget, stop '
                       'before the next layer group and analyse completed experts; no data added, no seed or threshold change after results'},
        'budget_minutes': BUDGET_MIN, 'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info(),
        'manifest_scan': rows}
    write(out / 'plan.json', p)
    md = ['# 預先登記：BC 與逐位置配對評分在非誤排挑選的 expert／格式上的排序差異', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 範圍與方法',
          '- 只限 expert 的 Down 輸入 activation；不含 Gate／Up 共享輸入。結論只適用於此範圍。',
          '- 只比較 T_old（BC：Σ_c G2_c/n·Σ_t r² + 偏差項）與 T_pair（Σ_c Σ_t g²r² + 相同偏差項）。',
          '- 固定 BC calibration、sum-loss、梯度正規化、router 語義（route 與權重在 backward 中固定）、量化器、range、6 個候選；不刷新、不調參、不加 spike 懲罰或一階項、不用 NLL 換算尺度；不改搜尋、knee、pipeline。', '',
          '## 排除與合格',
          f'- 前幾輪做過損傷或干預測量的 {len(excl)} 個 expert 不作新案例（清單見 plan.json）。',
          f'- 合格：有 Down 輸入 cache；calibration rows ≥ {MIN_CAL_ROWS}；discovery rows ≥ {MIN_ALL_ROWS}；R_token、range、cache 統計有限。不用任何 loss、BC 誤排或 T_pair 資訊。', '',
          '## 選點（select 階段依規則產生，之後才收集梯度）',
          f'- 一般樣本 {N_GENERAL} 個：每層從合格新 expert 隨機抽 1 個（seed {SEEDS["general"]}）；某層沒有就從其餘合格池隨機補足並記錄。',
          f'- spike 樣本：強 {N_STRONG} 個（R_token ≥ 合格集合 p90）、弱 {N_STRONG} 個（≤ p50），每個強 tap 配同層、log rows 最近的弱 tap（seed {SEEDS["spike"]}）；與一般樣本不重複；匹配不足列出。',
          '- 兩組分開報告；合併數字只作描述。', '',
          '## 格式與比較',
          f'- 格式：{", ".join(CANDS)}；主要：int4 vs p4e1、int6 vs p6e1、int8 vs p8e1；其餘 12 個配對為次要。全部結果保留。', '',
          '## 控制',
          '- 梯度路徑 CE 與 dr.forward 相同；G2／GS、S1／S2、T_old 重現 cache；恆等式成立；streaming 累加（Σg²e²、Σg²e、Σg²）與逐元素計算一致。不一致時先解釋。', '',
          '## 確認資料與測量',
          f'- {N_ART} 篇新 WT103 文章 × {WPA_C} 窗口，凍結後抽取；verified 掃描排除 {len(used)} 篇歷史文章；排除 WikiText-2 與近似重複。',
          '- FP32 背景、一次只量化一個 expert 的 Down 輸入、6 個格式、dynamic route；ΔNLL 相對 FP32。不做搜尋或 knee。', '',
          '## 評估',
          '- 決定性配對：文章 bootstrap 95% CI 排除 0；敏感度：99% CI 與組內 Holm。proxy 平手（相對差 < 0.02）算「不對」；兩個 proxy 用同一組配對。',
          '- 類別：兩者皆對、修正、弄壞、兩者皆不對；不可判定配對不算排對，另報數量與覆蓋率；同時報 expert 等權結果。',
          f'- 主要指標：各組決定性同位元配對的 修正 − 弄壞，expert bootstrap {100 * (1 - ALPHA_GROUP):.1f}% CI（兩組 Bonferroni）。',
          f'- 組別狀態：決定性同位元配對 < {MIN_DECISIVE} → 不足；淨值 > 0 且 CI 下界 > 0 → 改善；淨值 < 0 且上界 < 0 → 變差；其餘 → 不變。ΔNLL 為負的案例保留並分開解讀。', '',
          '## 結論規則',
          '- D：兩組都不足。A：兩組都改善，且全部 15 個配對中各組 弄壞 ≤ 修正/2。B：spike 組改善、一般組沒有。C：其餘（至少一組足夠）。只有一般組改善：照實報告（規則未涵蓋）。',
          '- 不論結果：不宣稱 Hessian、不宣稱 knee 改善、不改 pipeline。', '',
          f'## 預算（分鐘）：{BUDGET_MIN}；測量超時在下一個層組前停止；不追加資料、不換 seed、不改門檻。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    write(out / 'excluded_experts.json', excl)
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; excluded experts {len(excl)}; previous WT103 {len(used)}')


def select(out, log):
    if (out / 'selection.json').exists():
        raise SystemExit('exists')
    from run_native_bc_pipeline import QUANT
    t0 = time.perf_counter()
    excl = read(out / 'excluded_experts.json')
    with open(TS / 'discovery_experts.csv') as fh:
        ex = {r['node']: r for r in csv.DictReader(fh) if r['state'] == 'fp32'}
    names, *_ = pr.problem_static()
    searched = {n.rsplit('.', 1)[0] for n in names if n.endswith('down_proj')}
    aidx, gidx = read(ACT_INDEX), read(GC_INDEX)
    ac, gc = np.load(ACT_CACHE, mmap_mode='r'), np.load(GC, mmap_mode='r')
    ranges = read(RANGES)
    q0 = read(Q0K)
    a_, b_ = ss.COLOFF['down_x']
    fi = [QUANT.index(f) for f in CANDS]
    f_ = lambda r, k: float(r[k]) if r[k] not in ('', 'nan') else float('nan')
    el, reasons = [], Counter()
    for node, r in ex.items():
        if node in excl:
            reasons['excluded_measured'] += 1; continue
        if node not in searched or node not in aidx['row'] or node not in gidx['row']:
            reasons['no_cache'] += 1; continue
        if f_(r, 'rows_cal') < MIN_CAL_ROWS or f_(r, 'rows') < MIN_ALL_ROWS:
            reasons['coverage'] += 1; continue
        if not (math.isfinite(f_(r, 'R_token')) and f_(r, 's_p50') > 0):
            reasons['R_token'] += 1; continue
        rg = ranges.get(f'{node}.down_proj', {}).get('x')
        if rg is None or not all(math.isfinite(float(x)) for x in rg) or not float(rg[1]) > float(rg[0]):
            reasons['range'] += 1; continue
        row, grow = np.asarray(ac[aidx['row'][node]]), np.asarray(gc[gidx['row'][node]])
        if not (np.isfinite(row[:, fi, a_:b_]).all() and np.isfinite(grow[:, a_:b_]).all()):
            reasons['cache_nonfinite'] += 1; continue
        el.append(r)
    el.sort(key=lambda r: (int(r['layer']), int(r['expert'])))
    RT = np.array([f_(r, 'R_token') for r in el])
    hiT, loT = float(np.quantile(RT, STRONG_Q)), float(np.quantile(RT, WEAK_Q))
    rec = lambda r, grp, role, extra=None: {'node': r['node'], 'layer': int(r['layer']), 'expert': int(r['expert']), 'group': grp, 'role': role,
                                            'R_token': f_(r, 'R_token'), 'R_channel': f_(r, 'R_channel'), 'rows': int(f_(r, 'rows')),
                                            'rows_cal': int(f_(r, 'rows_cal')), 's_p50': f_(r, 's_p50'), 'threshold': TAU * f_(r, 's_p50'),
                                            'q0_act_format': q0[f'{r["node"]}.down_proj']['a'], **(extra or {})}
    rng = np.random.default_rng(SEEDS['general'])
    taps, used, empty = [], set(), []
    for l in range(NL):
        pool = [r for r in el if int(r['layer']) == l and r['node'] not in used]
        if not pool:
            empty.append(l); continue
        r = pool[int(rng.integers(len(pool)))]
        taps.append(rec(r, 'general', 'general')); used.add(r['node'])
    for _ in empty:
        pool = [r for r in el if r['node'] not in used]
        r = pool[int(rng.integers(len(pool)))]
        taps.append(rec(r, 'general', 'general_fill', {'fill_for_empty_layer': True})); used.add(r['node'])
    rng = np.random.default_rng(SEEDS['spike'])
    strong = [r for r in el if f_(r, 'R_token') >= hiT]
    weak = [r for r in el if f_(r, 'R_token') <= loT]
    got, layers = 0, set()
    for j in rng.permutation(len(strong)).tolist():
        if got == N_STRONG:
            break
        s_ = strong[j]
        if s_['node'] in used or int(s_['layer']) in layers:
            continue
        cand = [r for r in weak if r['node'] not in used]
        if not cand:
            break
        dl = min(abs(int(r['layer']) - int(s_['layer'])) for r in cand)
        cand = [r for r in cand if abs(int(r['layer']) - int(s_['layer'])) == dl]
        d = np.array([abs(math.log(f_(r, 'rows')) - math.log(f_(s_, 'rows'))) for r in cand])
        best = [cand[i] for i in np.flatnonzero(d == d.min())]
        w_ = best[int(rng.integers(len(best)))]
        taps.append(rec(s_, 'spike', 'strong', {'pair': got}))
        taps.append(rec(w_, 'spike', 'weak', {'pair': got, 'layer_diff': dl, 'log_rows_diff': float(d.min())}))
        used |= {s_['node'], w_['node']}; layers.add(int(s_['layer'])); got += 1
    for i, t in enumerate(taps):
        t['id'] = i
    write(out / 'selection.json', {'taps': taps, 'eligible': len(el), 'ineligible_reasons': dict(reasons), 'R_token_p90': hiT, 'R_token_p50': loT,
                                   'empty_layers': empty, 'strong_pool': len(strong), 'weak_pool': len(weak), 'eligible_nodes': [r['node'] for r in el]})
    log_stage(out, 'select', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'eligible': len(el), 'reasons': dict(reasons), 'empty_layers': empty, 'taps': [(t['id'], t['group'], t['role'], t['node'],
                                                                                                             round(t['R_token'], 1), t['rows']) for t in taps]})))


# --------------------------------------------------------------------------- #
def grad(out, log):
    """per-element g / h / e for every tap on the BC calibration (BC gradient path), controls, streaming accumulators and cost timing."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'grad_controls.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    sync = torch.cuda.synchronize
    taps = read(out / 'selection.json')['taps']
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    want_all = defaultdict(dict)
    for t in taps:
        want_all[t['layer']][t['expert']] = ['down_proj']
    lmin = min(want_all)
    cfg, ck = oc.config(), oc.Checkpoint()
    toks_np = np.load(CALIB)
    toks = torch.from_numpy(toks_np).to('cuda')
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g0 = ck.global_('cuda', torch.float32)
    STEQ = pr.ste_fn()
    tm = defaultdict(float)
    torch.cuda.reset_peak_memory_stats()
    sync(); t1 = time.perf_counter()
    routes, hs = [], []
    with torch.no_grad():
        h = F.embedding(toks, g0['embed'])
        hs.append(h)
        for l in range(NL):
            w = ck.layer(l, 'cuda', torch.float32)
            h, idx, rw = pr.layer_tap(w, h, cfg, cos, sin, None, None, {})
            routes.append((idx, rw)); hs.append(h)
    sync(); tm['bc_forward'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    hT = hs[-1].clone().requires_grad_(True)
    ce = head_ce(g0, hT, toks, cfg)
    ce.sum().backward()
    gn = hT.grad
    cap_all = {}
    for l in reversed(range(lmin, NL)):
        w = ck.layer(l, 'cuda', torch.float32)
        want = want_all.get(l, {})
        cap = {}
        hi = hs[l].clone().requires_grad_(True)
        o = ss.layer_tap_sel(w, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(None, STEQ), want, cap)
        o.backward(gn)
        gn = hi.grad
        for e in want:
            c = cap[e]
            cap_all[(l, e)] = (c['h'].detach().clone(), c['h'].grad.detach().clone(), c['tok'].detach().clone())
        del w, cap, o
    sync(); tm['bc_backward'] = time.perf_counter() - t1
    peak_bc = torch.cuda.max_memory_allocated() / 2 ** 30
    ref_ce = dr.forward(toks_np, None, dq, ranges, record=False, log=lambda m: None, label='dr fp32 cal')['ce']
    ctl = {'ce_vs_dr_forward_max_abs': float(np.max(np.abs(ce.detach().cpu().numpy() - np.array(ref_ce)))), 'taps': {}}
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    a_, b_ = ss.COLOFF['down_x']
    nbytes, acc_bytes = 0, 0
    torch.cuda.reset_peak_memory_stats()
    for t in taps:
        l, e, node = t['layer'], t['expert'], t['node']
        hh, gg, tok = cap_all[(l, e)]
        n = int(hh.shape[0])
        win = tok // SEQ
        wins = sorted(set(win.tolist()))
        sync(); t1 = time.perf_counter()
        gd = gg.double()
        G2 = (gd ** 2).sum(0)
        GS = torch.stack([gd[win == k].sum(0) for k in wins]).pow(2).sum(0)
        sync(); tm['bc_G2_GS'] += time.perf_counter() - t1
        grow = np.asarray(gc[gidx['row'][node]])
        row, n_c = np.asarray(ac[aidx['row'][node]]), int(aidx['n'][node])
        c = {'rows': n, 'n_cache': n_c, 'G2_equal_cache': bool(np.array_equal(G2.cpu().numpy(), grow[0, a_:b_])),
             'GS_equal_cache': bool(np.array_equal(GS.cpu().numpy(), grow[1, a_:b_])), 'formats': {}}
        rng = ranges[f'{node}.down_proj']['x']
        arrays = {'h': hh.cpu().numpy(), 'g': gg.cpu().numpy(), 'tok': tok.cpu().numpy()}
        g2 = gd ** 2
        for f in CANDS:
            sync(); t1 = time.perf_counter()
            err = (dq.act(hh, f, rng) - hh).double()
            S1, S2 = err.sum(0), err.pow(2).sum(0)
            sync(); tm['bc_S1_S2'] += time.perf_counter() - t1
            t1 = time.perf_counter()
            A = (g2 * err * err).sum(0)
            Bv = (g2 * err).sum(0)
            sync(); tm['pair_accumulators'] += time.perf_counter() - t1
            acc_bytes += 2 * A.numel() * 8
            S1n, S2n = S1.cpu().numpy(), S2.cpu().numpy()
            j = QUANT.index(f)
            S1c, S2c = row[0][j][a_:b_].astype(np.float64), row[1][j][a_:b_].astype(np.float64)
            Gc2, GSc = grow[0, a_:b_].astype(np.float64), grow[1, a_:b_].astype(np.float64)
            muc = S1c / n_c
            T_cache = float((Gc2 / n_c * (S2c - n_c * muc ** 2)).sum() + (GSc * muc ** 2).sum())
            mu = S1n / n
            G2n, GSn = G2.cpu().numpy(), GS.cpu().numpy()
            T_re = float((G2n / n * (S2n - n * mu ** 2)).sum() + (GSn * mu ** 2).sum())
            mu_t = S1 / n
            V_stream = float((A - 2 * mu_t * Bv + mu_t ** 2 * G2).sum())
            V_direct = float((g2 * (err - mu_t) ** 2).sum())
            c['formats'][f] = {'S1_equal_cache': bool(np.array_equal(S1n, S1c)), 'S2_equal_cache': bool(np.array_equal(S2n, S2c)), 'T_cache': T_cache,
                               'T_recomputed': T_re, 'T_rel': abs(T_re - T_cache) / max(abs(T_cache), 1e-300), 'V_pair_stream': V_stream,
                               'V_pair_direct': V_direct, 'stream_rel': abs(V_stream - V_direct) / max(abs(V_direct), 1e-300)}
            arrays[f'e_{f}'] = (dq.act(hh, f, rng) - hh).cpu().numpy()
        np.savez(out / 'work' / f'tap{t["id"]}.npz', **arrays)
        nbytes += sum(v.nbytes for v in arrays.values())
        ctl['taps'][node] = c
    peak_stats = torch.cuda.max_memory_allocated() / 2 ** 30
    ctl['pass'] = (ctl['ce_vs_dr_forward_max_abs'] == 0 and all(c['G2_equal_cache'] and c['GS_equal_cache'] and c['rows'] == c['n_cache']
                                                              and all(v['S1_equal_cache'] and v['S2_equal_cache'] and v['T_rel'] < 1e-12 and v['stream_rel'] < 1e-9
                                                                      for v in c['formats'].values()) for c in ctl['taps'].values()))
    res = {'controls': ctl, 'semantics': {'loss': 'head_ce(...).sum() over the 10 calibration windows', 'route': 'held constant in backward',
                                          'rows': 'torch.where(mask[e]) order, routed rows only'},
           'costs': {'seconds': dict(tm), 'peak_gpu_gb_bc_pass': peak_bc, 'peak_gpu_gb_stats': peak_stats, 'per_element_bytes': nbytes,
                     'pair_accumulator_bytes': acc_bytes, 'taps': len(taps), 'note': 'bc_forward / bc_backward cover the whole model on 10 windows '
                     '(the backward stops at the lowest tap layer); bc_G2_GS / bc_S1_S2 / pair_accumulators are summed over the taps and the 6 '
                     'candidates; per-element arrays are kept only for this diagnostic'}}
    write(out / 'grad_controls.json', res)
    log_stage(out, 'grad', time.perf_counter() - t0, {'peak_gpu_gb': max(peak_bc, peak_stats)})
    log(json.dumps(dr.clean({'pass': ctl['pass'], 'ce': ctl['ce_vs_dr_forward_max_abs'], 'costs': res['costs'],
                             'failing': [k for k, c in ctl['taps'].items() if not (c['G2_equal_cache'] and c['GS_equal_cache'] and all(v['S1_equal_cache'] and v['S2_equal_cache'] for v in c['formats'].values()))]})))


# --------------------------------------------------------------------------- #
def mech(out, log):
    """T_old / T_pair for every tap and candidate, mechanism statistics, frozen pair predictions."""
    if (out / 'predictions.json').exists():
        raise SystemExit('exists')
    if not read(out / 'grad_controls.json')['controls']['pass']:
        raise SystemExit('grad controls failed: explain first')
    t0 = time.perf_counter()
    taps = read(out / 'selection.json')['taps']
    gctl = read(out / 'grad_controls.json')['controls']['taps']
    res, preds = {}, []
    for t in taps:
        z = np.load(out / 'work' / f'tap{t["id"]}.npz')
        h, g, tok = z['h'], z['g'].astype(np.float64), z['tok']
        n = len(tok)
        win = tok // SEQ
        wins = sorted(set(win.tolist()))
        g2 = g ** 2
        G2 = g2.sum(0)
        GS = (np.stack([g[win == k].sum(0) for k in wins]) ** 2).sum(0)
        spike = np.abs(h).max(1) >= t['threshold']
        tr = {'rows': n, 'spike_rows': int(spike.sum()), 'formats': {}}
        for f in CANDS:
            e = z[f'e_{f}'].astype(np.float64)
            S1, S2 = e.sum(0), (e ** 2).sum(0)
            mu = S1 / n
            r2 = (e - mu) ** 2
            Vo_c, Vp_c, B_c = G2 / n * r2.sum(0), (g2 * r2).sum(0), GS * mu ** 2
            T_old_bc = float((G2 / n * (S2 - n * mu ** 2)).sum() + B_c.sum())
            cov = ((g2 - g2.mean(0)) * (r2 - r2.mean(0))).sum(0)
            share = lambda x, m: float(x[m].sum() / x.sum()) if x.sum() > 0 else None
            vo_rows, vp_rows = (G2 / n)[None, :] * r2, g2 * r2
            cv_rows = (g2 - g2.mean(0)) * (r2 - r2.mean(0))
            mc = int(np.argmax(Vo_c))
            oth = np.ones(len(Vo_c), bool); oth[mc] = False
            fr = {'T_old': float(Vo_c.sum() + B_c.sum()), 'T_pair': float(Vp_c.sum() + B_c.sum()), 'T_old_bcform': T_old_bc, 'V_old': float(Vo_c.sum()),
                  'V_pair': float(Vp_c.sum()), 'B': float(B_c.sum()), 'ratio': float(Vp_c.sum() / Vo_c.sum()) if Vo_c.sum() > 0 else None,
                  'cov_total': float(cov.sum()), 'identity_max_abs_rel': float(np.max(np.abs((Vp_c - Vo_c) - cov)) / max(np.max(np.abs(Vo_c)), 1e-300)),
                  'T_old_vs_cache_rel': abs(T_old_bc - gctl[t['node']]['formats'][f]['T_cache']) / max(abs(gctl[t['node']]['formats'][f]['T_cache']), 1e-300),
                  'rows_split': {nm: {'rows': int(m.sum()), 'g2_share': share(g2, m), 'r2_share': share(r2, m), 'V_old_share': share(vo_rows, m),
                                      'V_pair_share': share(vp_rows, m), 'cov_sum': float(cv_rows[m].sum())} for nm, m in (('spike', spike), ('other', ~spike))},
                  'main_channel': {'channel': mc, 'V_old_share': float(Vo_c[mc] / Vo_c.sum()), 'V_pair_share': float(Vp_c[mc] / Vp_c.sum()) if Vp_c.sum() > 0 else None,
                                   'ratio': float(Vp_c[mc] / Vo_c[mc]) if Vo_c[mc] > 0 else None,
                                   'other_ratio': float(Vp_c[oth].sum() / Vo_c[oth].sum()) if Vo_c[oth].sum() > 0 else None}}
            tr['formats'][f] = fr
        res[t['node']] = tr
        for p in pairs_of():
            To = tr['formats'][p['a']]['T_old'] - tr['formats'][p['b']]['T_old']
            Tp = tr['formats'][p['a']]['T_pair'] - tr['formats'][p['b']]['T_pair']
            mo = max(tr['formats'][p['a']]['T_old'], tr['formats'][p['b']]['T_old'])
            mp = max(tr['formats'][p['a']]['T_pair'], tr['formats'][p['b']]['T_pair'])
            preds.append({'tap': t['id'], 'node': t['node'], 'group': t['group'], 'role': t['role'], **p, 'T_old_diff': To, 'T_pair_diff': Tp,
                          'old_tie': bool(mo <= 0 or abs(To) / mo < BC_TIE), 'pair_tie': bool(mp <= 0 or abs(Tp) / mp < BC_TIE),
                          'log_ratio_a_over_b': (math.log(tr['formats'][p['a']]['ratio'] / tr['formats'][p['b']]['ratio'])
                                                 if tr['formats'][p['a']]['ratio'] and tr['formats'][p['b']]['ratio'] else None)})
    write(out / 'mech.json', res)
    write(out / 'predictions.json', {'note': 'frozen before any confirmation article', 'pairs': preds})
    with open(out / 'proxy_table.csv', 'w') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['tap', 'node', 'group', 'role', 'format', 'T_old', 'T_pair', 'V_old', 'V_pair', 'B', 'ratio', 'cov_total', 'spike_rows', 'spike_g2_share',
                       'spike_r2_share', 'main_channel', 'main_V_old_share', 'main_ratio', 'other_ratio'])
        for t in taps:
            for f in CANDS:
                fr = res[t['node']]['formats'][f]
                wcsv.writerow([t['id'], t['node'], t['group'], t['role'], f, fr['T_old'], fr['T_pair'], fr['V_old'], fr['V_pair'], fr['B'], fr['ratio'], fr['cov_total'],
                               fr['rows_split']['spike']['rows'], fr['rows_split']['spike']['g2_share'], fr['rows_split']['spike']['r2_share'],
                               fr['main_channel']['channel'], fr['main_channel']['V_old_share'], fr['main_channel']['ratio'], fr['main_channel']['other_ratio']])
    log_stage(out, 'mech', time.perf_counter() - t0)
    log(f'mech: {len(taps)} taps; max identity err {max(res[n]["formats"][f]["identity_max_abs_rel"] for n in res for f in CANDS):.2e}; '
        f'max T_old vs cache {max(res[n]["formats"][f]["T_old_vs_cache_rel"] for n in res for f in CANDS):.2e}')


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    taps = read(out / 'selection.json')['taps']
    files = ('plan.json', 'excluded_experts.json', 'selection.json', 'grad_controls.json', 'mech.json', 'predictions.json', 'proxy_table.csv') + \
        tuple(f'work/tap{t["id"]}.npz' for t in taps)
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'code': code_hashes(),
                                'note': 'selection, T_old / T_pair and all pair predictions frozen before any confirmation article is drawn'})
    log('frozen')


def data(out, log):
    """same rule as the previous rounds (verified manifest scan), this round's seeds."""
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')['confirmation']
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    if len(used) != pl['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    lines = mr.load_lines(mr.WT103)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['eligible_before_length_and_neardup']:
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


def measure(out, log):
    """FP32 background, one expert Down input quantized with one of the 6 candidates (od.make_qdq); batches of VAR_BATCH variants."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'measure.json').exists():
        raise SystemExit('exists')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('data changed')
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    taps = read(out / 'selection.json')['taps']
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ctx = od.Ctx(None)
    ce0, H_in, ids, chunks = od.base_pass(ctx, toks, dq, log, 'fp32 base')
    ref = dr.forward(toks, None, dq, ctx.ranges, record=False, log=lambda m: None, label='dr fp32')
    controls = {'base_equals_dr_forward_max_abs': float(np.max(np.abs(np.array(ce0) - np.array(ref['ce']))))}
    res = {'base_ce': ce0, 'variants': {}, 'acc': {}, 'route_changed': {}}
    by_layer = defaultdict(list)
    for tp in taps:
        by_layer[tp['layer']].append(tp)
    stopped = False
    for l in sorted(by_layer):
        if time.perf_counter() - t0 > 60 * BUDGET_MIN['measure']:
            stopped = True
            log(f'budget exceeded before layer {l}: stop')
            break
        variants, accs = [], {}
        if l == min(by_layer):
            variants.append(('control_identity', {}))
        for tp in by_layer[l]:
            rng = ctx.ranges[f'{tp["node"]}.down_proj']['x']
            for f in CANDS:
                name = f'tap{tp["id"]}|{f}'
                accs[name] = {'rows': 0, 'chE': None, 'clip': 0, 'clipE': 0.0, 'n': 0}
                variants.append((name, {l: {(l, tp['expert'], 'down_proj', 'x'): od.make_qdq(dq, f, rng, accs[name])}}))
        for b in range(0, len(variants), VAR_BATCH):
            part = variants[b:b + VAR_BATCH]
            ce, chg = od.variants_from(ctx, toks, l, H_in[l], ids, chunks, part, dq)
            for name, _ in part:
                if name == 'control_identity':
                    controls['identity_from_layer'] = {'layer': l, 'max_abs': float(np.max(np.abs(np.array(ce[name]) - np.array(ce0))))}
                    continue
                res['variants'][name] = ce[name]
                res['route_changed'][name] = chg[name]
                a = accs[name]
                res['acc'][name] = {'rows': a['rows'], 'err_energy': float(a['chE'].sum()) if a['chE'] is not None else 0.0,
                                    'clip_frac': a['clip'] / a['n'] if a['n'] else None}
        log(f'measure: layer {l} done ({time.perf_counter() - t0:.0f}s, {len(variants)} variants)')
    res['controls'] = controls
    res['stopped_early'] = stopped
    res['controls']['pass'] = controls['base_equals_dr_forward_max_abs'] == 0 and controls.get('identity_from_layer', {}).get('max_abs', 1) == 0
    write(out / 'measure.json', res)
    log_stage(out, 'measure', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'stopped_early': stopped})
    log(json.dumps(dr.clean({'controls': controls, 'variants': len(res['variants']), 'stopped': stopped})))


# --------------------------------------------------------------------------- #
def article_vec(ce):
    return np.array(ce).reshape(-1, WPA_C).sum(1) / (WPA_C * TP)


def classify(p, right_old, right_pair):
    if right_old and right_pair:
        return 'both_right'
    if right_pair:
        return 'fix'
    if right_old:
        return 'break'
    return 'both_not'


def summarize(rows, key='decisive'):
    dec = [r for r in rows if r[key]]
    c = Counter(r[f'class_{key}'] for r in dec)
    return {'pairs': len(rows), 'decisive': len(dec), 'undeterminable': len(rows) - len(dec), 'coverage': len(dec) / len(rows) if rows else None,
            'old_right': c['both_right'] + c['break'], 'pair_right': c['both_right'] + c['fix'], 'both_right': c['both_right'], 'fix': c['fix'],
            'break': c['break'], 'both_not': c['both_not'], 'old_ties': sum(r['old_tie'] for r in dec), 'pair_ties': sum(r['pair_tie'] for r in dec),
            'old_acc': (c['both_right'] + c['break']) / len(dec) if dec else None, 'pair_acc': (c['both_right'] + c['fix']) / len(dec) if dec else None}


def expert_equal(rows, key='decisive'):
    by = defaultdict(list)
    for r in rows:
        if r[key]:
            by[r['tap']].append(r)
    accs = [(np.mean([x[f'class_{key}'] in ('both_right', 'break') for x in v]), np.mean([x[f'class_{key}'] in ('both_right', 'fix') for x in v]),
             sum(x[f'class_{key}'] == 'fix' for x in v) - sum(x[f'class_{key}'] == 'break' for x in v)) for v in by.values()]
    if not accs:
        return {'experts': 0}
    return {'experts': len(accs), 'old_acc_mean': float(np.mean([a[0] for a in accs])), 'pair_acc_mean': float(np.mean([a[1] for a in accs])),
            'experts_net_positive': sum(a[2] > 0 for a in accs), 'experts_net_negative': sum(a[2] < 0 for a in accs), 'experts_net_zero': sum(a[2] == 0 for a in accs)}


def net_ci(rows, taps_in_group, key='decisive'):
    """net = fixes - breaks on decisive rows; expert bootstrap over the experts of the group (experts without decisive rows count 0)."""
    per = {t: 0 for t in taps_in_group}
    for r in rows:
        if r[key]:
            per[r['tap']] += (r[f'class_{key}'] == 'fix') - (r[f'class_{key}'] == 'break')
    v = np.array(list(per.values()), float)
    rng = np.random.default_rng(SEEDS['expert_bootstrap'])
    bs = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [100 * ALPHA_GROUP / 2, 100 * (1 - ALPHA_GROUP / 2)])
    return {'net': float(v.sum()), 'ci': [float(lo), float(hi)], 'level': 1 - ALPHA_GROUP}


def status(summ, nci):
    if summ['decisive'] < MIN_DECISIVE:
        return 'insufficient'
    if nci['net'] > 0 and nci['ci'][0] > 0:
        return 'improve'
    if nci['net'] < 0 and nci['ci'][1] < 0:
        return 'worse'
    return 'unchanged'


def analyze(out, log):
    t0 = time.perf_counter()
    taps = read(out / 'selection.json')['taps']
    m = read(out / 'measure.json')
    preds = read(out / 'predictions.json')['pairs']
    mech_ = read(out / 'mech.json')
    z = article_vec(m['base_ce'])
    na = len(z)
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, na, (BOOT, na))
    done = {t['id'] for t in taps if all(f'tap{t["id"]}|{f}' in m['variants'] for f in CANDS)}
    D = {(t['id'], f): article_vec(m['variants'][f'tap{t["id"]}|{f}']) - z for t in taps if t['id'] in done for f in CANDS}
    per_tap = []
    for t in taps:
        if t['id'] not in done:
            continue
        per_tap.append({'tap': t['id'], 'node': t['node'], 'group': t['group'], 'role': t['role'], 'R_token': t['R_token'],
                        'dnll': {f: float(D[(t['id'], f)].mean()) for f in CANDS},
                        'dnll_ci95': {f: np.percentile(D[(t['id'], f)][idx].mean(1), [2.5, 97.5]).tolist() for f in CANDS},
                        'T_old': {f: mech_[t['node']]['formats'][f]['T_old'] for f in CANDS}, 'T_pair': {f: mech_[t['node']]['formats'][f]['T_pair'] for f in CANDS},
                        'route_changed': {f: m['route_changed'][f'tap{t["id"]}|{f}'] for f in CANDS}})
    rows = []
    for p in preds:
        if p['tap'] not in done:
            continue
        d = D[(p['tap'], p['a'])] - D[(p['tap'], p['b'])]
        b = d[idx].mean(1)
        ci95, ci99 = np.percentile(b, [2.5, 97.5]).tolist(), np.percentile(b, [0.5, 99.5]).tolist()
        r = {**p, 'measured_diff': float(d.mean()), 'ci95': ci95, 'ci99': ci99, 'p': si.pval(b), 'decisive': bool(ci95[0] > 0 or ci95[1] < 0),
             'decisive99': bool(ci99[0] > 0 or ci99[1] < 0)}
        rows.append(r)
    for grp in ('general', 'spike'):
        prim = [r for r in rows if r['group'] == grp and r['same_bit']]
        rej = si.holm([r['p'] for r in prim]) if prim else []
        for r, rj in zip(prim, rej):
            r['decisive_holm'] = bool(rj)
    for r in rows:
        r.setdefault('decisive_holm', False)
        for key in ('decisive', 'decisive99', 'decisive_holm'):
            if r[key]:
                ro = (not r['old_tie']) and ((r['T_old_diff'] < 0) == (r['measured_diff'] < 0))
                rp = (not r['pair_tie']) and ((r['T_pair_diff'] < 0) == (r['measured_diff'] < 0))
                r[f'class_{key}'] = classify(r, ro, rp)
            else:
                r[f'class_{key}'] = None
    res = {'completed_taps': sorted(done), 'per_tap': per_tap, 'groups': {}}
    tid = lambda grp: [t['id'] for t in taps if t['group'] == grp and t['id'] in done]
    for grp in ('general', 'spike', 'pooled_descriptive'):
        gr = rows if grp == 'pooled_descriptive' else [r for r in rows if r['group'] == grp]
        g = {}
        for name, sel in (('same_bit', lambda r: r['same_bit']), ('all', lambda r: True), ('int-int', lambda r: r['family'] == 'int-int'),
                          ('posit-posit', lambda r: r['family'] == 'posit-posit'), ('cross', lambda r: r['family'] == 'cross'),
                          ('int4-p4e1', lambda r: (r['a'], r['b']) == ('int4', 'p4e1')), ('int6-p6e1', lambda r: (r['a'], r['b']) == ('int6', 'p6e1')),
                          ('int8-p8e1', lambda r: (r['a'], r['b']) == ('int8', 'p8e1'))):
            sub = [r for r in gr if sel(r)]
            g[name] = {k: summarize(sub, k) for k in ('decisive', 'decisive99', 'decisive_holm')} | {'expert_equal': expert_equal(sub)}
        if grp != 'pooled_descriptive':
            ids_ = tid(grp)
            g['net_same_bit'] = net_ci([r for r in gr if r['same_bit']], ids_)
            g['net_same_bit_99'] = net_ci([r for r in gr if r['same_bit']], ids_, 'decisive99')
            g['net_all'] = net_ci(gr, ids_)
            g['status'] = status(g['same_bit']['decisive'], g['net_same_bit'])
            g['status_99'] = status(g['same_bit']['decisive99'], g['net_same_bit_99'])
            g['all_breaks_le_half_fixes'] = g['all']['decisive']['break'] <= g['all']['decisive']['fix'] / 2
        res['groups'][grp] = g
    sg, ss_ = res['groups']['general']['status'], res['groups']['spike']['status']
    if sg == 'insufficient' and ss_ == 'insufficient':
        verdict = 'D'
    elif sg == 'improve' and ss_ == 'improve' and res['groups']['general']['all_breaks_le_half_fixes'] and res['groups']['spike']['all_breaks_le_half_fixes']:
        verdict = 'A'
    elif ss_ == 'improve' and sg != 'improve':
        verdict = 'B'
    elif sg == 'improve' and ss_ != 'improve':
        verdict = 'G_only'
    else:
        verdict = 'C'
    res['verdict'] = verdict
    sg9, ss9 = res['groups']['general']['status_99'], res['groups']['spike']['status_99']
    res['verdict_sensitivity_99'] = {'general': sg9, 'spike': ss9}
    # negative dNLL cases
    res['negative_dnll'] = [{'node': t['node'], 'group': t['group'], 'format': f, 'dnll': t['dnll'][f], 'ci95': t['dnll_ci95'][f]} for t in per_tap for f in CANDS
                            if t['dnll_ci95'][f][1] < 0]
    res['rows_with_negative_formats'] = sum(1 for r in rows if r['decisive'] and any(x['node'] == r['node'] and x['format'] in (r['a'], r['b']) for x in res['negative_dnll']))
    # post-hoc mechanism links (descriptive)
    disc = [r for r in rows if r['decisive'] and r['class_decisive'] in ('fix', 'break')]
    links = {'discordant': len(disc)}
    for cl in ('fix', 'break'):
        sub = [r for r in disc if r['class_decisive'] == cl and r['log_ratio_a_over_b'] is not None]
        links[cl] = {'n': len(sub), 'median_abs_log_ratio': float(np.median([abs(r['log_ratio_a_over_b']) for r in sub])) if sub else None}
    wrong_old = [r for r in rows if r['decisive'] and r['class_decisive'] in ('fix', 'both_not') and not r['old_tie']]
    over = []
    for r in wrong_old:
        better = r['a'] if r['measured_diff'] < 0 else r['b']          # measured better, but T_old said it costs more
        rat = mech_[r['node']]['formats'][better]['ratio']
        worse = r['b'] if better == r['a'] else r['a']
        over.append({'node': r['node'], 'pair': f'{r["a"]}-{r["b"]}', 'overestimated': better, 'ratio_overestimated': rat,
                     'ratio_other': mech_[r['node']]['formats'][worse]['ratio'], 'class': r['class_decisive']})
    links['old_wrong'] = {'n': len(over), 'overestimated_format_has_ratio_lt1': sum(1 for o in over if o['ratio_overestimated'] is not None and o['ratio_overestimated'] < 1),
                          'ratio_overestimated_lt_ratio_other': sum(1 for o in over if o['ratio_overestimated'] is not None and o['ratio_other'] is not None
                                                                    and o['ratio_overestimated'] < o['ratio_other']), 'cases': over}
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]
    dset = [r for r in rows if r['decisive'] and r['log_ratio_a_over_b'] is not None]
    links['by_abs_log_ratio'] = [{'bin': list(b_), 'n': len(s_), **dict(Counter(r['class_decisive'] for r in s_))}
                                 for b_ in bins for s_ in [[r for r in dset if b_[0] <= abs(r['log_ratio_a_over_b']) < b_[1]]]]
    res['mechanism_links_posthoc'] = links
    res['measure_controls'] = m['controls']
    res['stopped_early'] = m['stopped_early']
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    with open(out / 'comparisons.csv', 'w') as fh:
        keys = ['tap', 'node', 'group', 'role', 'a', 'b', 'family', 'same_bit', 'T_old_diff', 'T_pair_diff', 'old_tie', 'pair_tie', 'log_ratio_a_over_b', 'measured_diff',
                'p', 'decisive', 'decisive99', 'decisive_holm', 'class_decisive', 'class_decisive99', 'class_decisive_holm']
        wcsv = csv.writer(fh); wcsv.writerow(keys + ['ci95_lo', 'ci95_hi'])
        for r in rows:
            wcsv.writerow([r.get(k) for k in keys] + r['ci95'])
    with open(out / 'per_tap.csv', 'w') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['tap', 'node', 'group', 'role', 'R_token'] + [f'dnll_{f}' for f in CANDS] + [f'T_old_{f}' for f in CANDS] + [f'T_pair_{f}' for f in CANDS])
        for t in per_tap:
            wcsv.writerow([t['tap'], t['node'], t['group'], t['role'], t['R_token']] + [t['dnll'][f] for f in CANDS] + [t['T_old'][f] for f in CANDS]
                          + [t['T_pair'][f] for f in CANDS])
    with open(out / 'per_article.csv', 'w') as fh:
        wcsv = csv.writer(fh)
        keys = sorted(D)
        wcsv.writerow(['article', 'base_nll'] + [f'tap{k[0]}|{k[1]}' for k in keys])
        for i in range(na):
            wcsv.writerow([i, float(z[i])] + [float(D[k][i]) for k in keys])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'verdict': verdict, 'sens99': res['verdict_sensitivity_99'],
                             **{g: {'status': res['groups'][g]['status'], 'net': res['groups'][g]['net_same_bit'], 'same_bit': res['groups'][g]['same_bit']['decisive'],
                                    'all': res['groups'][g]['all']['decisive']} for g in ('general', 'spike')}})))


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
    cats = (('both_right', 'both right', PAL['muted']), ('fix', 'fix (T_pair right only)', PAL['s3']), ('break', 'break (T_old right only)', PAL['s2']),
            ('both_not', 'both not right', '#52514e'))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, grp in zip(axs, ('general', 'spike')):
        g = an['groups'][grp]
        sets = ['same_bit', 'int4-p4e1', 'int6-p6e1', 'int8-p8e1', 'int-int', 'posit-posit', 'cross', 'all']
        left = np.zeros(len(sets))
        for k, lab, col in cats:
            v = np.array([g[s]['decisive'][k] for s in sets], float)
            ax.barh(range(len(sets)), v, left=left, color=col, label=lab, height=0.6, edgecolor=PAL['surface'], linewidth=1)
            left += v
        ax.set_yticks(range(len(sets))); ax.set_yticklabels(sets); ax.invert_yaxis()
        ax.set_xlabel('decisive pairs (95% CI)'); ax.set_title(f'{grp} group: status {g["status"]}', loc='left', fontsize=10)
    hs, ls = axs[0].get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, fontsize=8, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(fd / 'outcomes.png', dpi=150); plt.close(fig)
    mech_ = read(out / 'mech.json')
    sel = read(out / 'selection.json')['taps']
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, f in enumerate(CANDS):
        xs = [i + (j - 2.5) * 0.12 for i in range(len(sel))]
        ax.scatter(xs, [mech_[t['node']]['formats'][f]['ratio'] for t in sel], s=14, color=PAL['s1'] if f.startswith('int') else PAL['s2'],
                   label=('INT' if f == 'int4' else 'Posit' if f == 'p4e1' else None), zorder=3)
    ax.axhline(1, color=PAL['text2'], lw=0.8); ax.set_yscale('log')
    ax.set_xticks(range(len(sel))); ax.set_xticklabels([f'{t["node"]} ({t["role"][0]})' for t in sel], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('V_pair / V_old (log)'); ax.set_title('Pairing ratio per expert and format (calibration, frozen)', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / 'pair_ratio.png', dpi=150); plt.close(fig)
    log_stage(out, 'figures', time.perf_counter() - t0)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'select', 'grad', 'mech', 'freeze', 'data', 'measure', 'analyze', 'figures'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
