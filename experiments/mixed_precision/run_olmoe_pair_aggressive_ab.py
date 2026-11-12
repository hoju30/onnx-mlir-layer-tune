#!/usr/bin/env python3
"""OLMoE whole-configuration A/B at the existing aggressive resource point: original BC (A) vs BC with only the Down-input activation
variance term paired by position (B). Pre-registered in plan.json / plan.md; the default pipeline is unchanged.

A: the static BC table (runs/olmoe_proxy_refresh_20261022/table_static.json) and its historical aggressive configuration q0.
B: the paired table verified in runs/olmoe_pair_knee_ab_20261023 (tables_B.json), reused unchanged.
Limits: runs/olmoe_proxy_refresh_20261022/plan.json["budget"] (exact values). One W+A search per table with the existing LP-first rule.

Stages (one --out-dir): correct_prev, plan, prep, search, configs, freeze, data, eval, analyze, figures.
"""
import time
T_PROC = time.time()
import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_pair_knee_ab as kab
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss
import run_olmoe_token_spike_diag as tsd
import wt_manifest_scan as wms

HERE = Path(__file__).resolve().parent
PRR = HERE / 'runs/olmoe_proxy_refresh_20261022'
PRC = HERE / 'runs/olmoe_proxy_refresh_confirm_20261022'
KAB = HERE / 'runs/olmoe_pair_knee_ab_20261023'
TS = HERE / 'runs/olmoe_token_spike_diag_20261023'
TABLE_A, TABLE_B = PRR / 'table_static.json', KAB / 'tables_B.json'
Q0, BUDGET_PLAN = PRR / 'configs/q0.json', PRR / 'plan.json'
RANGES, MP = ss.RANGES, ss.MP
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CTRL, BOOT = 600, 128, 10000
SEEDS = {'order': 23801, 'start': 23802, 'bootstrap': 23811}
FRONT, BACK = (0, 1), (14, 15)
BUDGET_MIN = {'prep': 2, 'search': 15, 'data': 6, 'eval': 20, 'analyze': 5, 'total_gpu': 25}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_pair_aggressive_ab.py', 'run_olmoe_pair_knee_ab.py', 'wt_manifest_scan.py', 'run_olmoe_search_final.py',
                                                  'run_olmoe_multipoint.py', 'run_solver_regression.py', 'mp_auto.py', 'run_native_bc_pipeline.py'] + dr.SCRIPTS))}


def merge_timing(out, key, obj):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[key] = obj
    write(f, t)


def bounds(sres):
    """lower-bound bookkeeping of one solve: the original-budget LP bound, the original-budget MILP attempt and the tightened 1-ppm retry
    kept apart; the final gap is measured against the original LP bound."""
    steps = {s['step']: s for s in sres['record']['steps']}
    rr = steps.get('round_repair', {})
    lp = rr.get('lp_bound')
    obj = sres['objective']
    out = {'path': sres['path'], 'final_objective': obj, 'original_LP_bound': lp,
           'final_minus_original_LP_bound': (obj - lp) if lp is not None else None,
           'final_minus_original_LP_bound_rel': ((obj - lp) / obj) if lp is not None else None,
           'round_repair': {k: rr.get(k) for k in ('objective', 'gap', 'tolerance', 'accepted', 'feasible', 'M_over_budget', 'B_over_budget', 'fractional_nodes', 'swaps', 'seconds')},
           'lp_seconds': steps.get('lp', {}).get('seconds')}
    if 'attempt1' in steps:
        a = steps['attempt1']
        out['milp_original_budget'] = {k: a.get(k) for k in ('seconds', 'status', 'message', 'objective', 'dual_bound', 'mip_gap', 'feasible', 'M_over_budget', 'B_over_budget')}
    if 'retry_1ppm' in steps:
        r = steps['retry_1ppm']
        out['milp_retry_tightened_1ppm'] = {**{k: r.get(k) for k in ('seconds', 'status', 'message', 'objective', 'dual_bound', 'mip_gap', 'feasible', 'M_over_budget', 'B_over_budget')},
                                            'note': 'bound of the problem with both budgets tightened by 1 ppm; not a proof of optimality for the original budget'}
    if 'repair_attempt1' in steps:
        out['repair_after_retry_failure'] = steps['repair_attempt1']
    out['proved_optimal_original_budget'] = bool(sres['path'] == 'lp->milp' and out.get('milp_original_budget', {}).get('mip_gap') == 0.0
                                                  and out.get('milp_original_budget', {}).get('feasible'))
    out['tolerance_note'] = ('LP-first accepts a rounded solution if objective - LP bound <= max(1e-3 x objective, 1e-6 x scale); HiGHS feasibility '
                             'tolerances allow tiny budget violations, hence the 1-ppm retry')
    return out


# --------------------------------------------------------------------------- #
def correct_prev(out, log):
    """note in the normal-knee A/B report: B's 'gap 0 / proven optimal' referred to the 1-ppm tightened retry problem."""
    corr = KAB / 'correction_lower_bound_20261023.json'
    if corr.exists():
        raise SystemExit('exists')
    s = read(KAB / 'search.json')
    bB, bA = bounds(s['B']), bounds(s['A'])
    orig = {f: sha(KAB / f) for f in ('report.md', 'search.json', 'analysis.json')}
    write(corr, {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'original_hashes': orig, 'B': bB, 'A': bA,
                 'statement': 'B was reported as "MILP, gap 0, proven optimal". The gap-0 dual bound belongs to the retry problem with both '
                              'budgets tightened by 1 ppm; the original-budget MILP attempt returned a solution violating the BOPs budget by '
                              '4.5e-7 (relative). Against the original-budget LP bound the final B solution is within the reported difference; '
                              'optimality for the original budget is not proven. No result or conclusion changes.'})
    rep = (KAB / 'report.md').read_text()
    if '下界更正' not in rep:
        note = ['', '> **下界更正（2026-10-23）**',
                f'> 本報告把 B 寫成「MILP，gap 0，已證明最佳」，這不正確：gap 0 與 dual bound {bB["milp_retry_tightened_1ppm"]["dual_bound"]:.9f} 屬於兩個預算都收緊 1 ppm 的重試問題，不能證明原始預算問題已達最佳。',
                f'> 原始預算的 MILP 第一次求解得到的解在 BOPs 上超出 {bB["milp_original_budget"]["B_over_budget"]:.1e}（相對值），屬於不可行。',
                f'> 以原始預算的 LP 下界 {bB["original_LP_bound"]:.9f} 計，B 最終解的差距是 {bB["final_minus_original_LP_bound"]:.2e}（相對 {bB["final_minus_original_LP_bound_rel"]:.1e}）；A 的對應差距是 {bA["final_minus_original_LP_bound"]:.4f}。',
                '> 結論不變。詳見 `correction_lower_bound_20261023.json`（含原始 hash）。', '']
        lines = rep.split('\n')
        (KAB / 'report.md').write_text('\n'.join(lines[:1] + note + lines[1:]))
    merge_timing(out, 'correct_prev', {'seconds': 0.0})
    log(json.dumps(dr.clean({'B': {k: bB[k] for k in ('original_LP_bound', 'final_minus_original_LP_bound', 'final_minus_original_LP_bound_rel')},
                             'A': {k: bA[k] for k in ('original_LP_bound', 'final_minus_original_LP_bound')}})))


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    from run_olmoe_search_final import REL_TOL, ABS_TOL, TIME_LIMIT
    names, nodes, elems, flops, rows = pr.problem_static()
    budget = read(BUDGET_PLAN)['budget']
    w103, wt2 = wms.corpora()
    used, mrows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    knee_ab_in = {s['content_sha256'] for s in read(KAB / 'segments.json')} <= used
    files = [TABLE_A, TABLE_B, Q0, BUDGET_PLAN, PRR / 'search_static.json', KAB / 'freeze.json', KAB / 'search.json', PRC / 'tokens.npy', PRC / 'eval.json',
             RANGES, TS / 'discovery_experts.csv'] + mr.WT103 + mr.WT2
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hypothesis': 'at the aggressive resource point, B (Down-input variance paired) gives a lower whole-configuration NLL than A (original BC)',
        'default': 'unchanged; experimental branch only; no quantized-state refresh, no router fine-tuning, no spike-only selection, no new coefficients',
        'A': {'table': str(TABLE_A.relative_to(HERE)), 'table_sha256': sha(TABLE_A), 'historical_config': str(Q0.relative_to(HERE)), 'config_sha256': sha(Q0)},
        'B': {'table': str(TABLE_B.relative_to(HERE)), 'table_sha256': sha(TABLE_B), 'verified_in': 'runs/olmoe_pair_knee_ab_20261023 (freeze sha '
              f'{read(KAB / "freeze.json")["files"]["tables_B.json"]})', 'definition': 'only the Down-input activation variance term of every searched '
              'expert: V_old = sum_c (sum_t g^2) / n sum_t (e - mu)^2 -> V_pair = sum_c sum_t g^2 (e - mu)^2; bias term, Down output term, Gate / Up '
              'terms and shared-input attribution, all weight rows, calibration, quantizer, ranges, candidates, searched nodes and resource model '
              'unchanged; full W+A search'},
        'limits': {'source': str(BUDGET_PLAN.relative_to(HERE)) + ' ["budget"]', 'max_bytes': budget['max_bytes'], 'max_bops': budget['max_bops'], 'R': budget['R'],
                   'rule': 'exact file values; never redefined from a configuration\'s usage; no new frontier, no new knee, no other point'},
        'search': {'rule': f'kab.solve (= ss.search_table with an explicit method): dominance pruning -> LP -> round + repair, accepted if '
                           f'objective - LP bound <= max({REL_TOL} x objective, {ABS_TOL} x scale), else corrected MILP (mip_rel_gap 0, '
                           f'{TIME_LIMIT} s per attempt, at most the existing single 1-ppm retry, then repair); same for A and B; no extra solves',
                   'A_control': 'the A table search must reproduce the historical q0 (configuration equality); otherwise investigate before any '
                                'new data', 'search_limited': 'if B\'s solution scores worse on the B table than the feasible A configuration, the '
                                'comparison is labelled search-limited (not a failure of the score); no configuration chosen by evaluation NLL',
                   'bounds': 'report the original-budget LP bound, the original-budget MILP attempt, and the tightened 1-ppm retry bound '
                             'separately; the final gap is measured against the original LP bound; a retry gap 0 is not a proof for the original '
                             'budget'},
        'controls': ['A table and B table hashes equal the records', 'B weight rows and non-Down activation rows equal A; Down rows changed; '
                     'finite; complete', 'A search reproduces q0', 'A / B configurations legal and within the exact limits', 'cross scores of A '
                     'and B configurations on both tables'],
        'configs_report': 'changed projections / experts, weight-only / activation-only / both, Gate / Up / Down, bit and INT / Posit / FP '
                          'distributions, Down activation shares <= 4 / <= 5 / <= 6 bit, A\'s FP16 Down activations, size / BOPs usage and slack; '
                          f'front {FRONT} / middle / back {BACK}; if A and B are identical, report that and skip the evaluation',
        'evaluation': {'data': f'{N} new WikiText-103 train articles x {WPA} windows of {SEQ}, drawn after the configuration freeze (seeds order '
                               f'{SEEDS["order"]}, start {SEEDS["start"]}); wt_manifest_scan.used_wt103_verified (count computed, not hard-coded: '
                               f'{len(used)} at plan time; the normal-knee A/B articles included: {knee_ab_in}); WikiText-2 and near duplicates '
                               'excluded; content-hash intersection with history checked to be 0',
                       'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                       'conditions': 'FP32, A, B on the same articles; original router; normal dynamic routing; same QDQ backend',
                       'controls': 'program control: q0 on the first 128 windows of runs/olmoe_proxy_refresh_confirm_20261022 (bitwise vs its '
                                   'record); rerun of B on the first 128 windows (bitwise)',
                       'primary': f'Delta = NLL(B) - NLL(A): per article over its 4 windows, then a paired article bootstrap ({BOOT}, seed '
                                  f'{SEEDS["bootstrap"]}) 95% CI; the only primary comparison',
                       'secondary': 'token-weighted NLL and PPL, damage vs FP32, better / worse articles, relative damage change (point estimate), '
                                    'routed BOPs proxy under dynamic routes',
                       'decision': {'CI < 0': 'gain at this aggressive point confirmed; weigh the offline cost; the auto knee only in a next round; '
                                              'no default change', 'CI contains 0': 'not confirmed; keep BC; close this pairing direction for now',
                                    'CI > 0': 'worse at this point; do not adopt', 'controls fail': 'comparison limited; no criterion-quality conclusion'},
                       'cautions': 'no extra articles, no configuration or threshold change after evaluation; "not significantly worse" is not '
                                   'equivalence or adoption evidence; no subtraction across different article batches'},
        'costs': 'table read + validation + preparation; pruning + search; final evaluation; any from-scratch estimate = historical build '
                 'measurements + this round\'s search measurement, labelled; no ONNX-MLIR native or low-bit kernel claims',
        'budget_minutes': BUDGET_MIN, 'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info(),
        'manifest_scan': mrows}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE aggressive 資源點完整配置 A／B — 原 BC vs Down-input 變異項逐位置配對', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 假說與定義',
          '- 假說：在既有 aggressive 資源點，B 的完整配置 NLL 低於 A。唯一主要比較：Δ = NLL(B) − NLL(A)。',
          f'- A：{TABLE_A.relative_to(HERE)}（sha {sha(TABLE_A)[:12]}）＋歷史 q0（{Q0.relative_to(HERE)}，sha {sha(Q0)[:12]}）。',
          f'- B：{TABLE_B.relative_to(HERE)}（sha {sha(TABLE_B)[:12]}，上一輪已驗證），只改 Down-input activation 變異項；其餘全部不變；完整 W＋A 搜尋。',
          '- 不做 quantized-state refresh、router fine-tune、spike-only 選點或新係數；不改正式 pipeline。', '',
          '## 資源點',
          f'- 讀 {BUDGET_PLAN.relative_to(HERE)} 的 budget：max_bytes = {budget["max_bytes"]!r}，max_bops = {budget["max_bops"]!r}（R ≈ {budget["R"]:.6f}）；不以配置用量重定義，不建 frontier、不重選 knee、不換點。', '',
          '## 搜尋',
          f'- 同一 LP-first：支配剪枝 → LP → 取整＋修復（誤差 ≤ max({REL_TOL}·目標, {ABS_TOL}·scale) 才接受）→ 否則修正版 MILP（每次 ≤ {TIME_LIMIT} s，最多一次既有 1-ppm 重試，再修復）；A／B 相同；不額外求解。',
          '- 控制：A 表搜尋必須重現歷史 q0（不同則先調查、不開始新資料）；B 的 weight 列與非 Down activation 列等於 A；配置合法、精確資源檢查；A 表與 B 表各自評 A、B 配置。',
          '- B 的解在 B 表上若差於可行的 A 配置 → 標示「搜尋受限」，不據此判定評分失效，不以評估 NLL 選配置。',
          '- 下界：分開報告原始預算的 LP 下界、原始預算的 MILP 嘗試、收緊 1 ppm 重試的 dual bound；最終差距以原始 LP 下界計；重試的 gap 0 不代表原始問題已證明最佳。', '',
          '## 配置分析（凍結後才抽資料）',
          '- 改變的投影／expert、weight-only／activation-only／兩者、Gate／Up／Down、位元與 INT／Posit／FP 分布、Down activation ≤4／≤5／≤6 bit 比例、原 FP16 Down activation、size／BOPs 用量與餘量；若 A、B 相同則直接回報、不評估。', '',
          '## 評估',
          f'- {N} 篇新 WT103 文章 × {WPA} 窗口（seeds {SEEDS["order"]}／{SEEDS["start"]}）；verified 掃描（計畫時排除 {len(used)} 篇，含正常 knee A/B：{knee_ab_in}；資料階段重新計算並比對，不硬編碼）；排除 WikiText-2 與近似重複；核對 content hash 交集為 0。',
          '- FP32、A、B；原 router、dynamic routing、同一 QDQ 後端；程式控制（q0 在舊批次前 128 窗口）與 B 前 128 窗口重跑。',
          f'- 主要：Δ = NLL(B) − NLL(A)，每篇先彙整 4 窗口，文章 paired bootstrap（{BOOT}，seed {SEEDS["bootstrap"]}）95% CI。次要：token 加權 NLL、PPL、相對 FP32 損傷、改善／變差文章、相對損傷改善（點估計）、dynamic route 下 BOPs proxy。',
          '- 判讀：CI < 0 → 此點收益確認（權衡成本；下一輪才考慮自動 knee，不改預設）；CI 含 0 → 未確認，保留 BC，配對方向先收斂；CI > 0 → 此點退步，不接入；控制不通過 → 比較受限。', '',
          '## 成本',
          '- 分開報告讀表／驗證／準備、剪枝與搜尋、最終評估；從頭成本若引用上一輪建表，標示「歷史建表實測＋本輪搜尋實測」。不宣稱 native 整合或 kernel 加速。', '',
          f'## 預算（分鐘）：{BUDGET_MIN}；不因結果追加文章、改配置或門檻；不跨批次相減 NLL。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    merge_timing(out, 'plan', {'seconds': time.perf_counter() - t0})
    log(f'plan written; limits {budget["max_bytes"]!r} / {budget["max_bops"]!r}; excluded {len(used)}; knee A/B included {knee_ab_in}')


def prep(out, log):
    """read the tables, verify hashes and the B-vs-A structure."""
    if (out / 'prep.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    t1 = time.perf_counter()
    tA, tB = read(TABLE_A), read(TABLE_B)
    t_read = time.perf_counter() - t1
    names, *_ = pr.problem_static()
    ctl = {'A_sha_equal_plan': sha(TABLE_A) == pl['A']['table_sha256'], 'B_sha_equal_plan': sha(TABLE_B) == pl['B']['table_sha256'],
           'B_sha_equal_previous_freeze': sha(TABLE_B) == read(KAB / 'freeze.json')['files']['tables_B.json'],
           'q0_sha_equal_plan': sha(Q0) == pl['A']['config_sha256'], 'structure': kab.table_checks(tA, tB, names)}
    s = ctl['structure']
    ctl['pass'] = bool(ctl['A_sha_equal_plan'] and ctl['B_sha_equal_plan'] and ctl['B_sha_equal_previous_freeze'] and ctl['q0_sha_equal_plan']
                       and s['weight_rows_equal_A'] and s['non_down_act_rows_equal_A'] and s['down_rows_covered'] and s['finite'] and s['formats_complete'])
    write(out / 'prep.json', {'controls': ctl, 'read_s': t_read})
    merge_timing(out, 'prep', {'seconds': time.perf_counter() - t0, 'read_tables_s': t_read, 'peak_cpu_gb': ss.peak_cpu_gb()})
    if not ctl['pass']:
        raise SystemExit('table controls failed')
    log(json.dumps(dr.clean(ctl)))


def search(out, log):
    if (out / 'search.json').exists():
        raise SystemExit('exists')
    if not read(out / 'prep.json')['controls']['pass']:
        raise SystemExit('prep controls failed')
    t0 = time.perf_counter()
    names, nodes, elems, flops, rows = pr.problem_static()
    prob = (names, nodes, elems, flops)
    budget = read(out / 'plan.json')['limits']
    lim = {'max_bytes': budget['max_bytes'], 'max_bops': budget['max_bops']}
    tA, tB = read(TABLE_A), read(TABLE_B)
    q0 = read(Q0)
    fmA, sA = kab.solve(tA, prob, lim, out / 'logs', 'A')
    sA['reproduces_historical_q0'] = fmA == q0
    res = {'limits': lim, 'A': sA, 'A_bounds': bounds(sA)}
    if not sA['reproduces_historical_q0']:
        diff = [n for n in names if fmA[n] != q0[n]]
        res['A_mismatch'] = {'projections': len(diff), 'examples': {n: {'search': fmA[n], 'q0': q0[n]} for n in diff[:20]},
                             'objective_search': sA['objective'], 'objective_q0': float(sum(tA['B'][n][q0[n]['w']] + tA['C'][n][q0[n]['a']] for n in names))}
        write(out / 'search.json', dr.clean(res))
        raise SystemExit('the A search does not reproduce the historical q0: investigate before any new data')
    fmB, sB = kab.solve(tB, prob, lim, out / 'logs', 'B')
    res['B'] = sB
    res['B_bounds'] = bounds(sB)
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    from run_olmoe_multipoint import resources
    from run_native_bc_pipeline import GRID11
    res['cross_scores'] = {'A_on_A_table': obj(tA, q0), 'B_on_A_table': obj(tA, fmB), 'A_on_B_table': obj(tB, q0), 'B_on_B_table': obj(tB, fmB),
                           'note': 'absolute values of different tables are not comparable'}
    res['search_limited'] = bool(obj(tB, q0) < obj(tB, fmB))
    chk = {}
    for c, fm in (('A', q0), ('B', fmB)):
        r = resources(fm, names, elems, flops)
        chk[c] = {'resources': r, 'within_limits': r['M_bytes'] <= lim['max_bytes'] and r['BOPs'] <= lim['max_bops'],
                  'slack': {'bytes': lim['max_bytes'] - r['M_bytes'], 'bops': lim['max_bops'] - r['BOPs']},
                  'use': {'size': r['M_bytes'] / lim['max_bytes'], 'bops': r['BOPs'] / lim['max_bops']},
                  'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in fm.values()), 'complete': set(fm) == set(names)}
    res['resource_checks'] = chk
    res['pass'] = bool(sA['reproduces_historical_q0'] and all(v['within_limits'] and v['legal'] and v['complete'] for v in chk.values()))
    (out / 'configs').mkdir(exist_ok=True)
    write(out / 'configs/A.json', q0)
    write(out / 'configs/B.json', fmB)
    res['identical'] = fmB == q0
    write(out / 'search.json', res)
    merge_timing(out, 'search', {'seconds': time.perf_counter() - t0, 'A': {k: sA[k] for k in ('pruning_s', 'solve_s', 'path')},
                                 'B': {k: sB[k] for k in ('pruning_s', 'solve_s', 'path')}, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps(dr.clean({'A': {k: sA[k] for k in ('objective', 'path', 'reproduces_historical_q0', 'solve_s')}, 'A_bounds_gap': res['A_bounds']['final_minus_original_LP_bound'],
                             'B': {k: sB[k] for k in ('objective', 'path', 'solve_s')}, 'B_bounds': {k: res['B_bounds'][k] for k in ('original_LP_bound', 'final_minus_original_LP_bound', 'proved_optimal_original_budget')},
                             'cross': res['cross_scores'], 'search_limited': res['search_limited'], 'identical': res['identical'], 'pass': res['pass']})))


def configs(out, log):
    from htq_pareto_search import FORMAT_BITS
    if (out / 'config_diff.json').exists():
        raise SystemExit('exists')
    A, B = read(out / 'configs/A.json'), read(out / 'configs/B.json')
    names = sorted(A)
    fam = lambda f: 'fp32' if f == 'fp32' else 'FP' if f in ('f16', 'bf16') else 'INT' if f.startswith('int') else 'Posit'
    with open(TS / 'discovery_experts.csv') as fh:
        ex = [r for r in csv.DictReader(fh) if r['state'] == 'fp32' and float(r['rows']) >= 512]
    p90 = float(np.quantile([float(r['R_token']) for r in ex], 0.9))
    strong = {r['node'] for r in ex if float(r['R_token']) >= p90}
    grp = lambda n: 'front' if int(n.split('.')[0][1:]) in FRONT else 'back' if int(n.split('.')[0][1:]) in BACK else 'middle'
    ch = [n for n in names if A[n] != B[n]]
    kind = Counter('both' if A[n]['w'] != B[n]['w'] and A[n]['a'] != B[n]['a'] else 'weight_only' if A[n]['w'] != B[n]['w'] else 'act_only' for n in ch)
    downs = [n for n in names if n.endswith('down_proj')]

    def down_share(cf):
        b = [FORMAT_BITS[cf[n]['a']] for n in downs]
        return {f'<={k}bit': sum(x <= k for x in b) / len(b) for k in (4, 5, 6)} | {'count': len(b)}

    def all_share(cf, key):
        b = [FORMAT_BITS[cf[n][key]] for n in names]
        return {f'<={k}bit': sum(x <= k for x in b) / len(b) for k in (4, 5, 6)}
    res = {'projections': len(names), 'changed': len(ch), 'changed_experts': len({'.'.join(n.split('.')[:2]) for n in ch}), 'kind': dict(kind),
           'by_projection_type': dict(Counter(n.split('.')[-1] for n in ch)), 'by_layer_group': dict(Counter(grp(n) for n in ch)),
           'strong_spike_threshold_R_token_p90': p90, 'strong_spike_changed': sum('.'.join(n.split('.')[:2]) in strong for n in ch),
           'weight_transitions': dict(Counter(f'{A[n]["w"]}->{B[n]["w"]}' for n in ch if A[n]['w'] != B[n]['w'])),
           'act_transitions': dict(Counter(f'{A[n]["a"]}->{B[n]["a"]}' for n in ch if A[n]['a'] != B[n]['a'])),
           'down_act_low_bit_share': {'A': down_share(A), 'B': down_share(B)},
           'all_act_low_bit_share': {'A': all_share(A, 'a'), 'B': all_share(B, 'a')}, 'all_weight_low_bit_share': {'A': all_share(A, 'w'), 'B': all_share(B, 'w')},
           'formats': {c: {'w': dict(Counter(cf[n]['w'] for n in names)), 'a': dict(Counter(cf[n]['a'] for n in names)),
                           'a_family': dict(Counter(fam(cf[n]['a']) for n in names)), 'w_bits': dict(Counter(FORMAT_BITS[cf[n]['w']] for n in names)),
                           'a_bits': dict(Counter(FORMAT_BITS[cf[n]['a']] for n in names)), 'down_a': dict(Counter(cf[n]['a'] for n in downs))}
                       for c, cf in (('A', A), ('B', B))},
           'A_fp16_down_act': {n: {'A': A[n], 'B': B[n]} for n in downs if A[n]['a'] in ('f16', 'bf16')},
           'A_fp16_act_all': {n: {'A': A[n], 'B': B[n]} for n in names if A[n]['a'] in ('f16', 'bf16')},
           'changed_list': [{'name': n, 'A': A[n], 'B': B[n], 'group': grp(n), 'strong_spike': '.'.join(n.split('.')[:2]) in strong} for n in ch],
           'identical': A == B}
    write(out / 'config_diff.json', res)
    log(json.dumps(dr.clean({k: res[k] for k in ('changed', 'changed_experts', 'kind', 'by_projection_type', 'by_layer_group', 'strong_spike_changed',
                                                 'down_act_low_bit_share', 'act_transitions', 'weight_transitions', 'identical')})))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'search.json')['pass']:
        raise SystemExit('search controls failed')
    from mp_auto import config_hash
    files = ['plan.json', 'prep.json', 'search.json', 'config_diff.json', 'configs/A.json', 'configs/B.json']
    cfgs = {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                'config_hash': config_hash(read(out / f'configs/{c}.json'))} for c in ('A', 'B')}
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'configs': cfgs,
                                'tables': {'A': sha(TABLE_A), 'B': sha(TABLE_B)}, 'code': code_hashes(),
                                'note': 'tables, configurations, code and hashes frozen before any evaluation article is drawn'})
    log('frozen')


def data(out, log):
    """kab.data (verified scan, same exclusion and near-duplicate rules) with this round's seeds."""
    saved = dict(kab.SEEDS)
    kab.SEEDS.update(order=SEEDS['order'], start=SEEDS['start'])
    try:
        kab.data(out, log)
    finally:
        kab.SEEDS.clear(); kab.SEEDS.update(saved)


def eval_(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'eval.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'freeze.json')
    for c, v in fz['configs'].items():
        if sha(HERE / v['path']) != v['sha256']:
            raise SystemExit('frozen configuration changed')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('frozen data changed')
    if fz['configs']['A']['config_hash'] == fz['configs']['B']['config_hash']:
        write(out / 'eval.json', {'skipped': True, 'reason': 'A and B identical: configuration unchanged at this resource point'})
        log('identical configurations: evaluation skipped')
        return
    (out / 'work').mkdir(parents=True, exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    torch.cuda.reset_peak_memory_stats()
    old_t, old = np.load(PRC / 'tokens.npy')[:CTRL], read(PRC / 'eval.json')['ce']['q0']
    r = dr.forward(old_t, read(out / 'configs/A.json'), dq, ranges, record=False, log=log, label='program control q0')
    pc = float(np.max(np.abs(np.array(r['ce']) - np.array(old[:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs = {'fp32': dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')}
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in ('A', 'B'):
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/B.json'), dq, ranges, record=True, log=log, label='rerun B')
    ctl = {'program_control_q0_ce_max_abs': pc,
           'rerun_B_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['B']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['B']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['B']['routes']['ids'][:CTRL]))},
           'finite': all(x['finite_ce'] for x in runs.values()), 'audit': {c: runs[c]['audit'] for c in ('A', 'B')},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in ('A', 'B')},
           'loadavg_end': os.getloadavg()}
    ctl['pass'] = bool(pc == 0 and ctl['rerun_B_first_chunk']['ce_max_abs'] == 0 and ctl['rerun_B_first_chunk']['hidden_bitwise_windows'] == CTRL
                       and ctl['rerun_B_first_chunk']['routes_equal'] and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: x['ce'] for c, x in runs.items()}, 'controls': ctl, 'seconds': {c: x['seconds'] for c, x in runs.items()},
                              'rerun_s': r2['seconds'], 'program_control_s': r['seconds'], 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                              'environment': dr.env_info()})
    merge_timing(out, 'eval', {**{f'{c}_s': x['seconds'] for c, x in runs.items()}, 'rerun_s': r2['seconds'], 'program_control_s': r['seconds'],
                               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps(dr.clean({'controls_pass': ctl['pass'], 'program_control': pc})))


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    from run_olmoe_multipoint import problem
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    res = {'search': read(out / 'search.json'), 'config_diff': {k: v for k, v in read(out / 'config_diff.json').items() if k != 'changed_list'}}
    if ev.get('skipped'):
        res['evaluation'] = ev
    else:
        if not ev['controls']['pass']:
            res['comparison_limited'] = 'evaluation controls failed'
        A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
        idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))

        def st(x):
            m = x[idx].mean(1)
            return {'mean': float(x.mean()), 'ci95': np.percentile(m, [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1)),
                    'better_articles': int((x < 0).sum()), 'worse_articles': int((x > 0).sum()), 'zero_articles': int((x == 0).sum())}
        D = A['B'] - A['A']
        prim = st(D)
        lo, hi = prim['ci95']
        prim['decision'] = 'B_better_confirmed' if hi < 0 else 'B_worse_confirmed' if lo > 0 else 'not_confirmed'
        res['primary_B_minus_A'] = prim
        res['damage'] = {c: st(A[c] - A['fp32']) for c in ('A', 'B')}
        tok = N * WPA * TP
        res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
        dA = res['nll_ppl']['A']['nll'] - res['nll_ppl']['fp32']['nll']
        res['relative_damage_change_point_estimate'] = (res['nll_ppl']['B']['nll'] - res['nll_ppl']['A']['nll']) / dA if dA else None
        cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B')}
        _, _, pnames, _, elems2, rows_cal, _ = problem(MP)
        counts = {}
        for c in ('fp32', 'A', 'B'):
            ids = np.load(out / f'routes_{c}.npz')['ids'].astype(np.int64)
            counts[c] = np.stack([np.bincount(ids[:, l].ravel(), minlength=NE) for l in range(NL)])
        searched = {(int(n.split('.')[0][1:]), int(n.split('.')[1][1:])) for n in pnames}
        unsearched = [(l, e) for l in range(NL) for e in range(NE) if (l, e) not in searched]
        rowsof = lambda cnt, n: cnt[int(n.split('.')[0][1:]), int(n.split('.')[1][1:])]
        b8_s = sum(rowsof(counts['fp32'], n) * elems2[n] * 64 for n in pnames)
        cost = {}
        for c in ('A', 'B'):
            sb = float(sum(rowsof(counts[c], n) * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
            cost[c] = {'routed_BOPs_relative_to_int8_fp32_counts': sb / b8_s, 'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
                       'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
        cost['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l, e in unsearched))
        cost['route_set_change_B_vs_A'] = float((np.sort(np.load(out / 'routes_B.npz')['ids'], -1) != np.sort(np.load(out / 'routes_A.npz')['ids'], -1)).any(-1).mean())
        res['routing_cost'] = cost
        res['eval_controls'] = ev['controls']
        segs = read(out / 'segments.json')
        with open(out / 'per_article.csv', 'w') as fh:
            rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, 'B_minus_A': float(D[i])} for i, s in enumerate(segs)]
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(out / 'per_window.csv', 'w') as fh:
            rows = [{'window': k, 'article': k // WPA, **{f'summed_ce_{c}': ev['ce'][c][k] for c in ev['ce']}} for k in range(N * WPA)]
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t = read(out / 'timing.json')
    kt = read(KAB / 'analysis.json')['timing']
    res['costs'] = {'this_round': {'prep': t.get('prep'), 'search': t.get('search'), 'eval': t.get('eval'), 'data': t.get('data')},
                    'historical_build_measurements': {'A_static_build_method_median_s': kt['A']['method_s']['median'], 'B_full_build_method_median_s': kt['B']['method_s']['median'],
                                                      'B_incremental_build_method_median_s': kt['Binc']['method_s']['median'],
                                                      'source': 'runs/olmoe_pair_knee_ab_20261023 (2 interleaved runs each)'},
                    'from_scratch_estimate_s': {'A': kt['A']['method_s']['median'] + t['search']['A']['pruning_s'] + t['search']['A']['solve_s'],
                                                'B_full': kt['B']['method_s']['median'] + t['search']['B']['pruning_s'] + t['search']['B']['solve_s'],
                                                'B_incremental': kt['Binc']['method_s']['median'] + t['search']['B']['pruning_s'] + t['search']['B']['solve_s'],
                                                'label': 'historical build measurement + this round\'s search measurement; not an end-to-end timing of this round'}}
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    write(out / 'analysis.json', res)
    merge_timing(out, 'analyze', {'seconds': time.perf_counter() - t0})
    log(json.dumps(dr.clean({'primary': res.get('primary_B_minus_A'), 'nll_ppl': res.get('nll_ppl'), 'rel': res.get('relative_damage_change_point_estimate')})))


def figures(out, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = tsd.PAL
    fd = out / 'figures'
    fd.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'axes.edgecolor': PAL['muted'], 'axes.labelcolor': PAL['text'],
                         'xtick.color': PAL['text2'], 'ytick.color': PAL['text2'], 'text.color': PAL['text'], 'axes.grid': True, 'grid.color': PAL['grid'],
                         'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
    cd = read(out / 'config_diff.json')
    tr = sorted(cd['act_transitions'].items(), key=lambda x: -x[1])[:16] + [(f'w {k}', v) for k, v in sorted(cd['weight_transitions'].items(), key=lambda x: -x[1])[:10]]
    if tr:
        fig, ax = plt.subplots(figsize=(7, 0.28 * len(tr) + 1.2))
        ax.barh(range(len(tr)), [v for _, v in tr], color=[PAL['s2'] if k.startswith('w ') else PAL['s1'] for k, _ in tr], height=0.6)
        ax.set_yticks(range(len(tr))); ax.set_yticklabels([k if k.startswith('w ') else f'a {k}' for k, _ in tr], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel('projections'); ax.set_title('Format transitions A -> B at the aggressive point', loc='left', fontsize=10)
        fig.tight_layout(); fig.savefig(fd / 'transitions.png', dpi=150); plt.close(fig)
    if (out / 'per_article.csv').exists():
        d = np.array([float(r['B_minus_A']) for r in csv.DictReader(open(out / 'per_article.csv'))])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(d, bins=60, color=PAL['s1'])
        ax.axvline(0, color=PAL['text2'], lw=0.8); ax.axvline(d.mean(), color=PAL['s2'], lw=1.5, label=f'mean {d.mean():+.2e}')
        ax.set_xlabel('NLL(B) - NLL(A) per article (nat/token)'); ax.set_ylabel('articles'); ax.legend(frameon=False)
        ax.set_title('Per-article difference (600 new articles, aggressive point)', loc='left', fontsize=10)
        fig.tight_layout(); fig.savefig(fd / 'per_article_delta.png', dpi=150); plt.close(fig)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['correct_prev', 'plan', 'prep', 'search', 'configs', 'freeze', 'data', 'eval', 'analyze', 'figures'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage not in ('plan', 'correct_prev') and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    (eval_ if a.stage == 'eval' else globals()[a.stage])(a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
