#!/usr/bin/env python3
"""Post hoc, CPU only: is the signed first-order term S = g_u . du needed next to an output-side second-order term?
Re-aggregates saved per-window / per-position arrays of three earlier GPU runs (no model, gradient, proxy or N recomputation):
  L27 Down (weight INT8), backgrounds U / F / W: runs/qwen3_topk_ce_curvature_20261026 (C_terms.npz, eval_ce_pos.npz)
  L26 Down (weight INT6), backgrounds U / F:     runs/qwen3_l26_suffix_bridge_20261027 (C_terms.npz, eval_ce_pos.npz)
  L26 coverage groups (F, 120 groups + C0):      runs/qwen3_l26_calib_coverage_20261027 (C_values.npz, eval_ce_pos.npz)
Scorers: S, K_full (= K2_actual, 1/2 Var_p0 of the actual logit difference), S + K_full, K256 (= Khat256), S + K256; all per token
relative to the same background's activation f16. Truth: D_C (calibration) and the run's own new articles (article bootstrap of that
run). Every N used here was already seen, so all new comparisons are exploratory.

Stages (one new --out-dir): plan (fixes tolerances / rules / sources before any new comparison), analyze.
../../venv/bin/python run_qwen_first_order_necessity_diagnostic.py --stage plan --out-dir runs/qwen_first_order_necessity_diagnostic_<date>
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
L27 = HERE / 'runs/qwen3_topk_ce_curvature_20261026'
L26 = HERE / 'runs/qwen3_l26_suffix_bridge_20261027'
COV = HERE / 'runs/qwen3_l26_calib_coverage_20261027'
DIAG = HERE / 'runs/qwen3_l26_calib_diagnostic_20261027'
FORMATS = ('f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16', 'fp32')       # index order of C_values.npz
SELECT_SET = ('f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16')             # exact ties: f16, then this order
PAIRS = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
SAME_BIT = (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1'))
SCORERS = ('S', 'K_full', 'S+K_full', 'K256', 'S+K256')
KTYPES = ('full', '256')
TP, WPA, NA, BOOT = 127, 2, 128, 20000
RUNS = {'L27': {'dir': L27, 'backgrounds': ('U', 'F', 'W'), 'k_full_key': 'K2', 'weight': 'int8', 'seed': 26102683},
        'L26': {'dir': L26, 'backgrounds': ('U', 'F'), 'k_full_key': 'K2_actual', 'weight': 'int6', 'seed': 26102713},
        'L26cov': {'dir': COV, 'backgrounds': ('F',), 'k_full_key': 'K2_actual', 'weight': 'int6', 'seed': 26102726}}
TAU = 4e-7          # nat/token: |difference| <= TAU is a tie (scores and C truth)
EPS = TAU           # |dS| / (|dK| + EPS); |dK| <= TAU is reported as "K near zero", no ratio
TOL_REPRO = 1e-12   # reproduction of saved per-token scores / N means / CIs
LOGF = None


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(HERE))


def sources():
    files = []
    for d, names in ((L27, ('report.md', 'plan.json', 'C_terms.npz', 'eval_ce_pos.npz', 'N_article_D.npz', 'scores_C.csv', 'frozen_predictions.json',
                            'stats.json', 'controls_calib.json', 'controls_eval.json', 'data_freeze.json')),
                     (L26, ('report.md', 'plan.json', 'C_terms.npz', 'eval_ce_pos.npz', 'N_article_D.npz', 'scores_C.csv', 'frozen_predictions.json',
                            'stats.json', 'controls_calib.json', 'controls_eval.json', 'data_freeze.json')),
                     (COV, ('report.md', 'plan.json', 'C_values.npz', 'eval_ce_pos.npz', 'N_article_D.npz', 'scores_C.csv', 'frozen_picks.json',
                            'stats.json', 'controls_calib.json', 'controls_eval.json', 'groups.json', 'data_freeze.json', 'freeze.json')),
                     (DIAG, ('report.md', 'summary.json', 'leave_one_window_out.csv'))):
        files += [d / n for n in names]
    files += [HERE / f for f in ('run_qwen3_topk_ce_curvature.py', 'run_qwen3_norm_ce_decomposition.py', 'run_qwen3_conditional_signed_score.py',
                                 'run_qwen3_l26_suffix_bridge.py', 'run_qwen3_l26_calib_coverage.py', 'run_qwen3_l26_calib_diagnostic.py',
                                 'runs/native_eps_fix_20261014/qwen3/table.json')]
    return files


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out):
    if (out / 'analysis_plan.json').exists():
        raise SystemExit('plan exists')
    missing = [rel(p) for p in sources() if not p.exists()]
    man = {'time': now(), 'files': {rel(p): sha(p) for p in sources() if p.exists()}, 'missing': missing,
           'not available (marked, not reconstructed)': {
               'original BC per candidate': 'runs/native_eps_fix_20261014/qwen3/table.json has native C[node][a] act costs, but they are computed '
                                            'with an FP32 background and FP32 target weight, native error statistics, other units and an additive '
                                            'per-node form; they match neither F / W nor the INT8 / INT6 target weight of U; the runs only hard-code '
                                            '"BC = f16" (A\'s choice). Not compared.',
               'L26 bridge and L26 coverage W background': 'never measured', 'coverage U background': 'never measured'},
           'script_sha256': sha(Path(__file__))}
    write(out / 'source_manifest.json', man)
    pl = {'time': now(), 'status': 'post hoc exploratory: every C and N value below was already observed in the source runs; no new data, '
                                   'model inference, gradient or proxy recomputation',
          'definitions': {'S': 'saved S_u = g_u . (u_f - u_f16), u = final-norm output before the LM head; summed over the 127 target positions of '
                                'each window, divided by windows x 127 (the runs\' per-token normalization)',
                          'K_full': 'saved K2_actual (L27 key "K2") = 1/2 Var_p0 of the actual logit difference with batch-8 baseline logits; '
                                    'output-side second-order reference, not a whole-model Hessian',
                          'K256': 'saved Khat256 (top-256 by the microbatch-1 baseline p, merged tail)',
                          'D_C': 'saved D = CE(f) - CE(f16) (L27 / L26) or CE(f) - CE(f16) from C_values (coverage), same normalization',
                          'D_n': 'per article: (2 windows x 127 targets) mean of CE(f) - CE(f16), the runs\' part() function',
                          'relative': 'everything relative to the same background\'s activation f16; no mixing of backgrounds or baselines',
                          'L26 du': 'after a real block-27 replay: not a replay-free local criterion',
                          'targets': 'L27 weight int8 (A), L26 weight int6 (A); activation input and output switched together',
                          'backgrounds': {'U': 'every other projection FP32', 'F': 'other 195 projections per A '
                                          '(runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json)',
                                          'W (L27 only)': 'Down projections of layers 0-26 with A\'s weight format, activation FP32; everything else FP32'}},
          'scorers': list(SCORERS), 'core comparisons': ['K_full -> S + K_full (value of the first-order term with the full output-side second order)',
                                                         'K256 -> S + K256 (same at low cost)', 'K_full vs K256 (compression)'],
          'not a test of necessity': 'S + K vs S (only shows that adding K helps)',
          'contexts': {'A': 'L27 C (10 windows) x U / F / W, truth on its own 128 N articles (same articles for U / F / W)',
                       'B': 'L26 C (10 windows) x U / F, truth on its own 128 N articles (same for U / F)',
                       'C': 'L26 coverage: 120 groups (WT2 / WT103 x single / multi x 3 pools x 10) + C0 reference, background F; each group ranks '
                            'and picks on its own 1280 tokens; truth = its own D_C and the coverage run\'s 128 N articles; never pooled'},
          'pairs': f'the 28 pairs of {list(SELECT_SET)}; same-bit pairs {[f"{a}-{b}" for a, b in SAME_BIT]} reported separately; fp32 numerical '
                   'control only (scores listed, never in pairs or picks)',
          'pair quantities': 'dS = S(a) - S(b), dK = K(a) - K(b) (full and 256 separately), dP = dS + dK, dD_C, dD_N = mean_n[D_n(a) - D_n(b)] with CI',
          'tolerances': {'tie_tau_nat_per_token': TAU, 'reason': 'largest documented numerical discrepancy between two implementations of the same '
                                                                 'second-order term (K2_actual vs K2_affine, L27: <= 4e-7); applied to every score '
                                                                 'difference and to dD_C; fixed before any pairwise comparison',
                         'ratio': f'|dS| / (|dK| + {EPS}); if |dK| <= {TAU} the pair is "K near zero" and no ratio is used; descriptive only, no '
                                  'activation threshold', 'reproduction_abs': TOL_REPRO},
          'classification': {'order': 'sign of the difference, 0 if |d| <= tau', 'truth C': 'order of dD_C (tie -> truth tie)',
                             'truth N': 'sign of the mean if the run\'s 95% article-bootstrap CI excludes 0, else undetermined (CI rule of the runs)',
                             'categories (K vs S+K)': {'fix': 'K wrong, S+K right', 'break': 'K right, S+K wrong', 'both right': '', 'both wrong': '',
                                                       'score tie': 'K or S+K order 0', 'truth tie / undetermined': ''},
                             'flip': 'order(K) != order(S+K); listed with: dS opposite to dK, |dS| > |dK|, right on C, right on N (decisive), same reading '
                                     'under K_full and K256'},
          'statistics': {'bootstrap': f'{BOOT} article resamples with each run\'s own seed (L27 26102683, L26 26102713, coverage 26102726) and its '
                                      'bincount construction, so original CIs reproduce; one index set per run; 95% percentile CI',
                         'paired backgrounds': 'within a run U / F / W use the same articles: background differences are article-paired',
                         'across runs': 'different articles: no paired comparison, no pooling into one confirmation set',
                         'multiplicity': 'many pair CIs are exploratory descriptions, not a confirmatory multiple-comparison test',
                         'coverage': 'the 30 groups of a cell share articles: not 30 independent calibration sets; each pool listed, then the average; '
                                     'the N bootstrap reflects only N uncertainty for fixed groups and picks'},
          'selectors (secondary)': 'argmin of each scorer over the selection set, exact ties f16 first then the set order (the runs\' rule); N mean '
                                   'Delta NLL vs f16 with CI; paired S+K pick - K pick; C margin; coverage pick frequencies. K alone gives f16 = 0 and '
                                   'every other format > 0, so its argmin is f16 by construction: picks are not evidence for or against the first '
                                   'order on their own; same-bit pair orders are reported next to them',
          'background reading': 'per run: which pairs are fixed / broken under U vs F (and W), change of dS vs change of dD_C / dD_N (paired); '
                                'a net gain only in F is written "supports a background-dependent first-order value", never as a proven '
                                'error-cancellation mechanism',
          'residuals per candidate': ['D_C - (S + K_full)', 'D_C - (S + K256)', 'K_full - K256'], 'P0 / Rest': 'auxiliary decomposition only; no '
                                                                                                                'position removal or re-weighting',
          'controls': ['S + K256 reproduces the saved P256 per token (all contexts) and its frozen pick',
                       'S + K_full reproduces the saved S_u + K2_actual reference and its frozen pick',
                       'saved D_C per token and picks / N means / CIs of the original selectors reproduce',
                       'D_n reproduces the saved N_article_D.npz', 'f16: S, K, D_C, D_n exactly 0', 'P0 + Rest = All (C and N)',
                       'format / article / background indices: coverage C0 equals the L26 bridge F arrays',
                       'any failure stops the analysis before interpretation'],
          'boundaries': 'no change to the formal BC, configs, quantizers, ranges, MILP / knee; no training, gradient refresh, STE, HVP, JVP or GPU; '
                        'no activation rule for the first order; stop after the report',
          'code_sha256': sha(Path(__file__))}
    write(out / 'analysis_plan.json', pl)
    (out / 'analysis_plan.md').write_text('\n'.join([
        '# 分析計畫：一階方向資訊必要性診斷（事後探索）', '', f'時間：{pl["time"]}。以 analysis_plan.json 為準。', '',
        '- 性質：只重新聚合三個既有 run 已保存的 C／N 陣列；所有 C 與 N 先前都已看過，本輪全部新比較都是事後探索，不是確認性實驗。',
        '- 評分器：S（= g_u·Δu）、K_full（= K2_actual，輸出端二階參考，非全模型 Hessian）、S+K_full、K256（= Khat256）、S+K256；全部相對同背景 activation f16、每 token。',
        '- 核心比較：K_full → S+K_full（一階的額外價值）、K256 → S+K256（低成本版）、K_full vs K256（二階壓縮）。S+K vs S 不用來證明一階必要。',
        '- 情境：A＝L27（權重 int8）U／F／W；B＝L26（權重 int6）U／F；C＝L26 coverage 120 組＋C0（F），每組獨立排序與選擇，不合併。',
        f'- 配對：選擇集 28 對（f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16），另列同 bit 三對；fp32 只作數值控制。',
        f'- 平手容差 τ = {TAU} nat/token（分數差與 dD_C 都適用；依據：L27 K2_actual 與 K2_affine 的已知數值差 ≤ 4e-7），在任何配對比較前固定。',
        f'- 比值 |dS|/(|dK|+{EPS})；|dK| ≤ τ 記為「K 近零」不算比值；只作描述，不設啟用門檻。',
        '- N 真值：沿用各 run 的文章單位（2 窗口 × 127）、20000 次 bootstrap 與原 seed；CI 跨 0 為未定。大量配對的 CI 只是探索性描述。',
        '- 分類：修正（K 錯、S+K 對）、破壞（K 對、S+K 錯）、都對、都錯、分數平手、真值未定；全部列出，不挑案例。',
        '- 背景：同 run 內 U／F／W 使用同一批 N 文章，配對分析；只在 F 有淨改善時寫「支持背景相關的一階價值」，不寫成已證明抵銷機制。',
        '- 選擇（次要）：各評分器 argmin（精確平手 f16 優先，再依集合順序）；K 單獨使用時 f16 = 0、其他 > 0，必然選 f16，所以必須同時看同 bit 排序。',
        '- coverage：每個 pool 分開列，再列平均；30 組共享文章，不視為 30 個獨立校準集；不跨 run 合併 N。',
        '- 原 BC：native table 的背景、目標權重、誤差統計與單位都不匹配，標示為無法比較（見 source_manifest.json）。',
        '- 核對：S+K256 = 原 P256、S+K_full = 原完整二階參考、原選擇／N 平均／CI 重現、D_n = 原 N_article_D、f16 為 0、P0+Rest=All、coverage C0 = L26 F；任一失敗先停。',
        '']) + '\n')
    log(f'plan written; missing {missing}')


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def wb(seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, NA, NA), minlength=NA) for _ in range(BOOT)]) / NA


def part(ce, G):
    c = ce.reshape(NA, WPA, TP)
    return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[G] / (WPA * TP)


def per_token(a, G):
    nw = a.shape[0]
    return float({'P0': a[:, 0].sum(), 'Rest': a[:, 1:].sum(), 'All': a.sum()}[G] / (nw * TP))


def contexts():
    """[(run, background, group or None, {format: {term: [windows, 127]}})]"""
    ctx = []
    for run in ('L27', 'L26'):
        R = RUNS[run]
        z = np.load(R['dir'] / 'C_terms.npz')
        for b in R['backgrounds']:
            ctx.append((run, b, None, {f: {'S': z[f'{b}__{f}__S_u'], 'K_full': z[f'{b}__{f}__{R["k_full_key"]}'],
                                           'K256': z[f'{b}__{f}__Khat256'], 'D_C': z[f'{b}__{f}__D']} for f in FORMATS}))
    z = np.load(COV / 'C_values.npz')
    fi16 = FORMATS.index('f16')
    for g in [g_['id'] for g_ in read(COV / 'groups.json')]:
        ce = z[f'{g}__CE']
        ctx.append(('L26cov', 'F', g, {f: {'S': z[f'{g}__S_u'][i], 'K_full': z[f'{g}__K2_actual'][i], 'K256': z[f'{g}__Khat256'][i],
                                           'D_C': ce[i] - ce[fi16]} for i, f in enumerate(FORMATS)}))
    return ctx


def scores(c):
    """per-token All / P0 / Rest of every term and scorer."""
    v = {}
    for f, t in c.items():
        for G in ('P0', 'Rest', 'All'):
            S, Kf, K2, D = (per_token(t[k], G) for k in ('S', 'K_full', 'K256', 'D_C'))
            v[(f, G)] = {'S': S, 'K_full': Kf, 'K256': K2, 'S+K_full': S + Kf, 'S+K256': S + K2, 'D_C': D}
    return v


def argmin_sel(val):
    order = sorted(SELECT_SET, key=lambda f: (val[f], SELECT_SET.index(f)))
    return order[0], val[order[1]] - val[order[0]], order


def sgn(d):
    return 0 if abs(d) <= TAU else (1 if d > 0 else -1)


def category(oK, oP, T):
    if T == 0:
        return 'truth_tie_or_undetermined'
    if oK == 0 or oP == 0:
        return 'score_tie'
    return {(False, True): 'fix', (True, False): 'break', (True, True): 'both_right', (False, False): 'both_wrong'}[(oK == T, oP == T)]


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyze(out):
    t_all = time.perf_counter()
    pl = read(out / 'analysis_plan.json')
    man = read(out / 'source_manifest.json')
    if pl['code_sha256'] != sha(Path(__file__)):
        raise SystemExit('script changed since the plan (record an amendment)')
    changed = [f for f, h in man['files'].items() if sha(HERE / f) != h]
    if changed:
        raise SystemExit(f'sources changed: {changed}')
    ctl, ok = {}, True
    ctx = contexts()
    # N truth per run / background
    Wbs = {run: wb(R['seed']) for run, R in RUNS.items()}
    ND = {}
    for run, R in RUNS.items():
        z = np.load(R['dir'] / 'eval_ce_pos.npz')
        for b in R['backgrounds']:
            key = (lambda f: f) if run == 'L26cov' else (lambda f, b=b: f'{b}__{f}')
            for f in FORMATS:
                for G in ('P0', 'Rest', 'All'):
                    ND[(run, b, f, G)] = part(z[key(f)], G) - part(z[key('f16')], G)

    def st(run, v):
        bs = Wbs[run] @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'pos': int((v > 0).sum()), 'neg': int((v < 0).sum())}

    def nord(s):
        return 0 if not (s['ci95'][0] > 0 or s['ci95'][1] < 0) else (1 if s['mean'] > 0 else -1)
    # ---- controls
    worst = defaultdict(float)
    for run, R in RUNS.items():
        zN = np.load(R['dir'] / 'N_article_D.npz')
        for b in R['backgrounds']:
            for f in FORMATS:
                for G in ('P0', 'Rest', 'All'):
                    k = f'{f}__{G}' if run == 'L26cov' else f'{b}__{f}__{G}'
                    worst['N_article_D_abs'] = max(worst['N_article_D_abs'], float(np.abs(zN[k] - ND[(run, b, f, G)]).max()))
                worst['N_P0_plus_Rest_abs'] = max(worst['N_P0_plus_Rest_abs'], float(np.abs(ND[(run, b, f, 'P0')] + ND[(run, b, f, 'Rest')]
                                                                                            - ND[(run, b, f, 'All')]).max()))
            worst['N_f16_abs'] = max(worst['N_f16_abs'], float(np.abs(ND[(run, b, 'f16', 'All')]).max()))
    SC = {}
    fpL27, fpL26 = read(L27 / 'frozen_predictions.json'), read(L26 / 'frozen_predictions.json')
    fpC, stC = read(COV / 'frozen_picks.json'), read(COV / 'stats.json')
    stats_old = {'L27': read(L27 / 'stats.json'), 'L26': read(L26 / 'stats.json')}
    for run, b, g, c in ctx:
        v = scores(c)
        SC[(run, b, g)] = v
        for f in FORMATS:
            worst['P0_plus_Rest_abs'] = max(worst['P0_plus_Rest_abs'], max(abs(v[(f, 'P0')][k] + v[(f, 'Rest')][k] - v[(f, 'All')][k]) for k in v[(f, 'All')]))
        worst['f16_abs'] = max(worst['f16_abs'], max(abs(x) for x in v[('f16', 'All')].values()), max(abs(x) for x in v[('f16', 'P0')].values()))
        all_ = {f: v[(f, 'All')] for f in FORMATS}
        if run in ('L27', 'L26'):
            sp = (fpL27 if run == 'L27' else fpL26)['scores_per_token_All'][b]
            for mine, theirs in (('S+K256', 'P256'), ('S+K_full', 'Su+K2actual'), ('S', 'S_u'), ('D_C', 'D')):
                worst[f'repro_{mine}'] = max(worst[f'repro_{mine}'], max(abs(all_[f][mine] - sp[theirs][f]) for f in FORMATS))
            sel = (fpL27 if run == 'L27' else fpL26)['selectors'][b]
            roles = {'S+K256': 'primary_P256', 'S+K_full': 'reference_Su+K2actual' if run == 'L27' else 'quadratic_reference',
                     'S': 'old_Su' if run == 'L27' else 'first_order'}
            for mine, role in roles.items():
                ctl.setdefault('pick_reproduced', []).append([run, b, mine, argmin_sel({f: all_[f][mine] for f in SELECT_SET})[0] == sel[role]])
                pk = sel[role]
                s_old = stats_old[run]['picks'][b][role]['N_D_All']
                s_new = st(run, ND[(run, b, pk, 'All')])
                worst['repro_pick_N_mean_CI'] = max(worst['repro_pick_N_mean_CI'], abs(s_new['mean'] - s_old['mean']),
                                                    *[abs(x - y) for x, y in zip(s_new['ci95'], s_old['ci95'])])
        else:
            for mine, theirs in (('S+K256', 'P256'), ('S+K_full', 'Su+K2actual'), ('D_C', 'D_C')):
                pv = fpC['picks'][g][theirs]['per_token_All']
                worst[f'repro_cov_{mine}'] = max(worst[f'repro_cov_{mine}'], max(abs(all_[f][mine] - pv[f]) for f in FORMATS))
                pk = argmin_sel({f: all_[f][mine] for f in SELECT_SET})[0]
                ctl.setdefault('cov_pick_reproduced', Counter())[str(pk == fpC['picks'][g][theirs]['pick'])] += 1
                worst['repro_cov_N_mean'] = max(worst['repro_cov_N_mean'], abs(float(ND[('L26cov', 'F', pk, 'All')].mean()) - stC['groups'][g][theirs]['N_mean_D']))
    # coverage C0 = L26 bridge F (index / background / normalization check)
    zb, zc = np.load(L26 / 'C_terms.npz'), np.load(COV / 'C_values.npz')
    worst['cov_C0_vs_L26F'] = max(max(float(np.abs(zc['C0__S_u'][i] - zb[f'F__{f}__S_u']).max()), float(np.abs(zc['C0__K2_actual'][i] - zb[f'F__{f}__K2_actual']).max()),
                                      float(np.abs(zc['C0__Khat256'][i] - zb[f'F__{f}__Khat256']).max()), float(np.abs(zc['C0__CE'][i] - zb[f'F__{f}__CE']).max()))
                                  for i, f in enumerate(FORMATS))
    ctl['worst'] = dict(worst)
    ctl['checks'] = {'repro_scores': all(worst[k] <= TOL_REPRO for k in worst if k.startswith('repro_') and 'N_mean' not in k),
                     'repro_N': worst['repro_pick_N_mean_CI'] <= TOL_REPRO and worst['repro_cov_N_mean'] <= TOL_REPRO,
                     'picks': all(x[-1] for x in ctl['pick_reproduced']) and set(ctl['cov_pick_reproduced']) == {'True'},
                     'N_article_D': worst['N_article_D_abs'] <= TOL_REPRO, 'f16_zero': worst['f16_abs'] == 0 and worst['N_f16_abs'] == 0,
                     'P0_plus_Rest': worst['P0_plus_Rest_abs'] <= TOL_REPRO and worst['N_P0_plus_Rest_abs'] <= TOL_REPRO,
                     'cov_C0_equals_L26_F': worst['cov_C0_vs_L26F'] == 0}
    ctl['passed'] = bool(all(ctl['checks'].values()))
    write(out / 'controls.json', ctl)
    log(f'controls passed={ctl["passed"]} {ctl["checks"]} worst {dict(worst)}')
    if not ctl['passed']:
        raise SystemExit('controls failed: stop before interpretation')
    # ---- N pair truths per run / background
    NP = {}
    for run, R in RUNS.items():
        for b in R['backgrounds']:
            for a, c2 in PAIRS:
                NP[(run, b, a, c2)] = st(run, ND[(run, b, a, 'All')] - ND[(run, b, c2, 'All')])
            for f in SELECT_SET:
                NP[(run, b, f)] = st(run, ND[(run, b, f, 'All')])
    # ---- candidate scores
    crows = []
    for (run, b, g), v in SC.items():
        for f in FORMATS:
            for G in ('P0', 'Rest', 'All'):
                x = v[(f, G)]
                r = {'run': run, 'background': b, 'group': g or '', 'format': f, 'G': G, 'D_C': x['D_C'], 'S': x['S'], 'K_full': x['K_full'],
                     'K256': x['K256'], 'S+K_full': x['S+K_full'], 'S+K256': x['S+K256'], 'D_C-(S+K_full)': x['D_C'] - x['S+K_full'],
                     'D_C-(S+K256)': x['D_C'] - x['S+K256'], 'K_full-K256': x['K_full'] - x['K256'],
                     'N_mean_D': float(ND[(run, b, f, G)].mean())}
                crows.append(r)
    # ---- pairwise
    prows = []
    for (run, b, g), v in SC.items():
        for a, c2 in PAIRS:
            A, B = v[(a, 'All')], v[(c2, 'All')]
            dS, dD = A['S'] - B['S'], A['D_C'] - B['D_C']
            n = NP[(run, b, a, c2)]
            TC, TN = sgn(dD), nord(n)
            red = {}
            for kt in KTYPES:
                kk = 'K_full' if kt == 'full' else 'K256'
                dK = A[kk] - B[kk]
                dP = dS + dK
                oK, oP, oS = sgn(dK), sgn(dP), sgn(dS)
                red[kt] = (oK, oP)
                prows.append({'run': run, 'background': b, 'group': g or '', 'pair': f'{a}-{c2}', 'same_bit': (a, c2) in SAME_BIT or (c2, a) in SAME_BIT,
                              'K': kt, 'dS': dS, 'dK': dK, 'dP': dP, 'dD_C': dD, 'dD_N': n['mean'], 'dD_N_lo': n['ci95'][0], 'dD_N_hi': n['ci95'][1],
                              'order_S': oS, 'order_K': oK, 'order_P': oP, 'truth_C': TC, 'truth_N': TN,
                              'cat_C': category(oK, oP, TC), 'cat_N': category(oK, oP, TN),
                              'S_alone_C': 'tie' if oS == 0 or TC == 0 else ('right' if oS == TC else 'wrong'),
                              'S_alone_N': 'tie' if oS == 0 or TN == 0 else ('right' if oS == TN else 'wrong'),
                              'flip': oK != oP, 'dS_opposes_dK': oS != 0 and oK != 0 and oS == -oK, 'absdS_gt_absdK': abs(dS) > abs(dK),
                              'ratio_absdS_over_absdK': '' if abs(dK) <= TAU else abs(dS) / (abs(dK) + EPS), 'K_near_zero': abs(dK) <= TAU})
            for r in prows[-len(KTYPES):]:
                r['same_reading_full_256'] = red['full'] == red['256']
    # ---- selectors
    srows = []
    for (run, b, g), v in SC.items():
        picks = {}
        for s_ in SCORERS + ('D_C',):
            pk, mg, order = argmin_sel({f: v[(f, 'All')][s_] for f in SELECT_SET})
            picks[s_] = pk
            nst = NP[(run, b, pk)]
            srows.append({'run': run, 'background': b, 'group': g or '', 'scorer': s_, 'pick': pk, 'C_margin': mg, 'C_score_of_pick': v[(pk, 'All')][s_],
                          'C_D_of_pick': v[(pk, 'All')]['D_C'], 'order': ' '.join(order), 'N_dNLL_vs_f16': nst['mean'], 'N_lo': nst['ci95'][0],
                          'N_hi': nst['ci95'][1], 'N_harmful_articles': nst['pos'], 'paired_vs': '', 'paired_mean': '', 'paired_lo': '', 'paired_hi': ''})
        for x, y in (('S+K_full', 'K_full'), ('S+K256', 'K256'), ('S+K_full', 'S+K256')):
            p_ = st(run, ND[(run, b, picks[x], 'All')] - ND[(run, b, picks[y], 'All')])
            srows.append({'run': run, 'background': b, 'group': g or '', 'scorer': f'{x} minus {y}', 'pick': f'{picks[x]} vs {picks[y]}', 'C_margin': '',
                          'C_score_of_pick': '', 'C_D_of_pick': '', 'order': '', 'N_dNLL_vs_f16': '', 'N_lo': '', 'N_hi': '', 'N_harmful_articles': '',
                          'paired_vs': y, 'paired_mean': p_['mean'], 'paired_lo': p_['ci95'][0], 'paired_hi': p_['ci95'][1]})
    # ---- background comparison (within a run, paired articles)
    brows = []
    for run, combos in (('L27', (('U', 'F'), ('U', 'W'), ('W', 'F'))), ('L26', (('U', 'F'),))):
        for b1, b2 in combos:
            v1, v2 = SC[(run, b1, None)], SC[(run, b2, None)]
            for a, c2 in PAIRS:
                dn = st(run, (ND[(run, b2, a, 'All')] - ND[(run, b2, c2, 'All')]) - (ND[(run, b1, a, 'All')] - ND[(run, b1, c2, 'All')]))
                for kt in KTYPES:
                    kk = 'K_full' if kt == 'full' else 'K256'
                    r1 = next(r for r in prows if r['run'] == run and r['background'] == b1 and r['pair'] == f'{a}-{c2}' and r['K'] == kt)
                    r2 = next(r for r in prows if r['run'] == run and r['background'] == b2 and r['pair'] == f'{a}-{c2}' and r['K'] == kt)
                    dS1, dS2 = v1[(a, 'All')]['S'] - v1[(c2, 'All')]['S'], v2[(a, 'All')]['S'] - v2[(c2, 'All')]['S']
                    dK1, dK2 = v1[(a, 'All')][kk] - v1[(c2, 'All')][kk], v2[(a, 'All')][kk] - v2[(c2, 'All')][kk]
                    brows.append({'run': run, 'from': b1, 'to': b2, 'pair': f'{a}-{c2}', 'same_bit': (a, c2) in SAME_BIT or (c2, a) in SAME_BIT, 'K': kt,
                                  'dS_from': dS1, 'dS_to': dS2, 'delta_dS': dS2 - dS1, 'dK_from': dK1, 'dK_to': dK2, 'delta_dK': dK2 - dK1,
                                  'dD_C_from': r1['dD_C'], 'dD_C_to': r2['dD_C'], 'delta_dD_C': r2['dD_C'] - r1['dD_C'],
                                  'dD_N_from': r1['dD_N'], 'dD_N_to': r2['dD_N'], 'delta_dD_N': dn['mean'], 'delta_dD_N_lo': dn['ci95'][0],
                                  'delta_dD_N_hi': dn['ci95'][1], 'cat_C_from': r1['cat_C'], 'cat_C_to': r2['cat_C'], 'cat_N_from': r1['cat_N'],
                                  'cat_N_to': r2['cat_N']})
    # ---- summaries
    def counts(rows):
        out_ = {}
        for T in ('C', 'N'):
            c = Counter(r[f'cat_{T}'] for r in rows)
            out_[T] = dict(c)
            out_[T]['net_fix_minus_break'] = c['fix'] - c['break']
        return out_
    summ = {}
    for (run, b, g) in SC:
        if run == 'L26cov':
            continue
        for kt in KTYPES:
            rr = [r for r in prows if r['run'] == run and r['background'] == b and r['K'] == kt]
            summ[f'{run}|{b}|{kt}|28'] = counts(rr)
            summ[f'{run}|{b}|{kt}|same_bit'] = counts([r for r in rr if r['same_bit']])
            summ[f'{run}|{b}|S_alone'] = {T: dict(Counter(r[f'S_alone_{T}'] for r in rr)) for T in ('C', 'N')}
    groups = read(COV / 'groups.json')
    cell = {g_['id']: (g_['source'], g_['coverage'], g_['pool']) for g_ in groups}
    covrows = []
    for src in ('WT2', 'WT103'):
        for cov in ('single', 'multi'):
            for pool in (0, 1, 2, 'all'):
                gs = [g_ for g_, (s_, c_, p_) in cell.items() if s_ == src and c_ == cov and (pool == 'all' or p_ == pool)]
                for kt in KTYPES:
                    rr = [r for r in prows if r['run'] == 'L26cov' and r['group'] in gs and r['K'] == kt]
                    cc = counts(rr)
                    sb = counts([r for r in rr if r['same_bit']])
                    sp = [r for r in srows if r['run'] == 'L26cov' and r['group'] in gs]
                    pk = lambda s_: Counter(r['pick'] for r in sp if r['scorer'] == s_)
                    kk = 'K_full' if kt == 'full' else 'K256'
                    pair_rows = [r for r in sp if r['scorer'] == f'S+{kk} minus {kk}']
                    covrows.append({'source': src, 'coverage': cov, 'pool': pool, 'K': kt, 'groups': len(gs), 'pairs': len(rr),
                                    **{f'C_{k}': cc['C'].get(k, 0) for k in ('fix', 'break', 'both_right', 'both_wrong', 'score_tie', 'truth_tie_or_undetermined')},
                                    'C_net': cc['C']['net_fix_minus_break'],
                                    **{f'N_{k}': cc['N'].get(k, 0) for k in ('fix', 'break', 'both_right', 'both_wrong', 'score_tie', 'truth_tie_or_undetermined')},
                                    'N_net': cc['N']['net_fix_minus_break'], 'samebit_C_net': sb['C']['net_fix_minus_break'], 'samebit_N_net': sb['N']['net_fix_minus_break'],
                                    'samebit_N_fix': sb['N'].get('fix', 0), 'samebit_N_break': sb['N'].get('break', 0),
                                    f'picks_S+{kk}': json.dumps(dict(pk(f'S+{kk}'))), f'picks_{kk}': json.dumps(dict(pk(kk))), 'picks_S': json.dumps(dict(pk('S'))),
                                    'picks_D_C': json.dumps(dict(pk('D_C'))),
                                    f'mean_N_dNLL_S+{kk}_pick': float(np.mean([r['N_dNLL_vs_f16'] for r in sp if r['scorer'] == f'S+{kk}'])),
                                    f'mean_paired_S+{kk}_minus_{kk}': float(np.mean([r['paired_mean'] for r in pair_rows]))})
    for name, rows in (('candidate_scores.csv', crows), ('pairwise_diagnostics.csv', prows), ('selector_outcomes.csv', srows),
                       ('background_comparison.csv', brows), ('coverage_pool_summary.csv', covrows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    # flips (L27 / L26 contexts) for the report
    flips = [r for r in prows if r['run'] != 'L26cov' and r['flip']]
    cov_flip = Counter((cell[r['group']][0], cell[r['group']][1], r['K'], r['pair'], r['cat_N']) for r in prows
                       if r['run'] == 'L26cov' and r['group'] != 'C0' and r['flip'])
    write(out / 'summary.json', {'contexts_28_and_same_bit': summ, 'flips_L27_L26': flips,
                                 'coverage_flip_counts': [list(k) + [v] for k, v in sorted(cov_flip.items(), key=lambda kv: -kv[1])],
                                 'N_mean_by_run': {f'{run}|{b}|{f}': NP[(run, b, f)] for run, R in RUNS.items() for b in R['backgrounds'] for f in SELECT_SET}})
    secs = time.perf_counter() - t_all
    with open(out / 'timing.json', 'w') as fh:
        json.dump({'analyze_cpu_seconds': secs, 'time': now()}, fh, indent=1)
    log(f'analysis done in {secs:.1f} s; {len(prows)} pair rows, {len(flips)} flips in L27 / L26')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    out = a.out_dir.resolve()
    if a.stage == 'plan' and out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    LOGF = open(out / f'{a.stage}.log', 'a')
    t0 = time.time()
    {'plan': plan, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(Path(__file__)), 'argv': sys.argv}) + '\n')


if __name__ == '__main__':
    main()
