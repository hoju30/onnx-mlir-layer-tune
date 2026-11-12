#!/usr/bin/env python3
"""OLMoE normal knee: all-node sensitivity-only refresh (C), next to the frozen A / B / D of
runs/olmoe_knee_selective_sensitivity_20261022. Pre-registered in plan.json / plan.md.

A static knee; B all sensitivities + activation errors in q0_knee's working state; C all sensitivities in q0_knee's
working state with the FP32 static weight / activation error statistics (normalized by the static routed rows);
D sensitivities of the q0_knee <= 4-bit projections only. A, B, D are the frozen configurations of the previous round;
C is searched once at the same exact knee limits (knee/knee.json). Same code path: run_olmoe_knee_selective_sensitivity.build
(sensitivity-only is selected by passing the static activation-error cache; the cache use is checked on every run).

Stages (one --out-dir): plan, controls, refresh (--variant B|C|D --rep k), e2e, timing (--phase refresh|e2e), check,
freeze, data, eval, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_knee_selective_sensitivity as ks
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss

HERE = Path(__file__).resolve().parent
KS = HERE / 'runs/olmoe_knee_selective_sensitivity_20261022'
SS = ks.SS
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE, D2, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.D2, ks.ACT_CACHE, ks.ACT_INDEX
Q0K, KNEE, LOWBIT = KS / 'knee/q0_knee.json', KS / 'knee/knee.json', KS / 'lowbit_nodes.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, NCMP = 10000, 0.0005, 1e-12, 3
CI_CORR = [100 * 0.05 / (2 * NCMP), 100 * (1 - 0.05 / (2 * NCMP))]      # Bonferroni over the 3 primary comparisons: 98.33%
SEEDS = {'order': 22601, 'start': 22602, 'bootstrap': 22611}
REFRESH_SCHEDULE = [('B', 1), ('C', 1), ('D', 1), ('C', 2), ('D', 2), ('B', 2), ('D', 3), ('B', 3), ('C', 3)]
CATEGORIES = ks.CATEGORIES


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_all_sensitivity.py', 'run_olmoe_knee_selective_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def previous_round():
    """the frozen A / B / D, tables, knee and targets of the previous round against its freeze.json; reuse hashes."""
    f = read(KS / 'freeze.json')
    res = {'configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in f['configs'].items()},
           'tables': {n: sha(KS / 'tables' / n) == h for n, h in f['tables'].items()},
           'knee': sha(KNEE) == f['knee_sha256'], 'lowbit': sha(LOWBIT) == f['lowbit_nodes_sha256'], 'check': sha(KS / 'check.json') == f['check_sha256'],
           'check_pass': read(KS / 'check.json')['pass'],
           'code_unchanged': {k: sha(HERE / k) == f['code'][k] for k in ('run_olmoe_knee_selective_sensitivity.py', 'run_olmoe_selective_sensitivity.py')},
           'q0_knee_equals_A': read(Q0K) == read(KS / 'configs/A.json')}
    reuse = ks.reuse_hashes()
    res['reuse_hashes_equal'] = all(v['equal'] for k, v in reuse.items() if k != 'ss_check_pass') and reuse['ss_check_pass']
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['pass'] = flat(res)
    return res


def run_record(tm, t0, extra):
    import torch
    total = time.perf_counter() - t0
    timers = dict(tm.t)
    cats = {c: sum(timers.get(k, 0.0) for k in ks_) for c, ks_ in CATEGORIES.items()}
    cats['unclassified'] = total - sum(timers.values())
    return {'total_s': total, 'method_s': total - cats['controls'], 'timers': timers, 'categories': cats,
            'process_start_to_run_s': time.time() - total - T_PROC, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'peak_cpu_gb': ss.peak_cpu_gb(), 'gpu': torch.cuda.get_device_name(0), **extra}


def cache_use(variant, timers):
    """sensitivity-only (C / D) must read the static activation-error cache and compute no activation error; B the reverse."""
    a_err, a_asm, c_rd = timers.get('act_error', 0.0), timers.get('act_assembly', 0.0), timers.get('cache_read', 0.0)
    ok = (a_err == 0 and a_asm > 0 and c_rd > 0) if variant in ('C', 'D') else (a_err > 0 and a_asm == 0 and c_rd == 0)
    return {'act_error_s': a_err, 'act_assembly_s': a_asm, 'act_cache_read_s': c_rd, 'as_expected': ok}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    prev = previous_round()
    names, nodes, elems, flops, rows = pr.problem_static()
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = ss.input_files() + [KS / f for f in ('plan.json', 'freeze.json', 'check.json', 'controls.json', 'eval.json', 'tokens.npy', 'configs/A.json',
                                                  'configs/B.json', 'configs/D.json', 'tables/static.json', 'tables/B.json', 'tables/D.json',
                                                  'knee/knee.json', 'knee/q0_knee.json', 'lowbit_nodes.json',
                                                  'work/q0_knee_calibration_routes_dr_forward.npz')] + [ACT_CACHE, ACT_INDEX]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    kn = read(KNEE)
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'At the normal knee, does an all-node sensitivity-only refresh (C: every searched projection, static error '
                    'statistics) improve the static knee, and does it separate the roles of update scope (C vs D) and activation-error '
                    'refresh (B vs C)?',
        'not_this_round': 'no new knee, no router fine-tuning, no parameter update, no BC formula change, no low-bit threshold change, '
                          'no default-pipeline change, no loss-based fallback or selection',
        'fixed': {'previous_round': str(KS.relative_to(HERE)), 'previous_round_hashes': prev,
                  'q0_knee': f'{Q0K.relative_to(HERE)} (= frozen A)', 'limits': kn['limits'],
                  'limits_note': 'exact size / BOPs limits of knee/knee.json (auto_select budget x reference); never rebuilt from a rounded R',
                  'A_B_D': 'frozen configurations of the previous round (hash-checked), not re-searched',
                  'same': 'checkpoint and router, 10 x 128 calibration, ranges, corrected quantizer, numerics, 3051 searched projections, '
                          'GRID11 x GRID11, resource model, LP-first solver with exact dominance pruning'},
        'conditions': {'A': 'static BC knee (q0_knee)', 'B': 'all-node sensitivities + activation errors (frozen)',
                       'C': 'all-node sensitivities, weight / activation error statistics static; one search at the knee limits; every '
                            'projection may change format',
                       'D': 'q0_knee <= 4-bit projections\' sensitivities, other rows static (frozen)'},
        'C_math': {'weight': 'W_new(f) = sum_c S_new,c x D_static,c(f), S_new,c = G_c / n_in (identical to B\'s weight term)',
                   'activation': 'A_new(f) = sum_slots,c [G2_new,c var_static,c(f) + GS_new,c mu_static,c(f)^2], static mean / variance '
                                 'normalized by the static routed rows n_s (never the q0_knee rows), evaluated in the BC op order '
                                 'G2 / n_s (S2 - n_s mu^2) + GS mu^2 from the cached sums',
                   'scope': 'every searched projection, every candidate format',
                   'gradients': 'one q0_knee calibration forward with its own dynamic routes; backward with the recorded routes constant, '
                                'identity STE through activation QDQ, gate / up shared input with its total gradient; no second update',
                   'fallback': 'a searched projection with no q0_knee calibration token keeps its static row (count and reason recorded)',
                   'implementation': 'run_olmoe_knee_selective_sensitivity.build with the static activation-error cache passed explicitly '
                                     'and no targets; the previous knee refresh() loads the cache only for D, so it is not reused for C; '
                                     'cache use asserted from the timers (no activation-error block, cache read and assembly blocks present)'},
        'controls': {
            '1_inputs': 'A / B / D configurations, tables, knee, targets and every reuse dependency hash equal the previous round',
            '2_forward': 'C (and the new B / D timing runs) q0_knee forward: per-window CE, final hidden sha256, routes equal a fresh '
                         'independent run_olmoe_dynamic_route.forward and the previous round\'s record',
            '3_gradients': 'per-layer input-gradient sha256 identical across every B / C / D run and equal to the previous round',
            '4_rows': 'C weight rows bitwise equal B\'s; frozen D target rows bitwise equal C\'s',
            '5_recompute': f'independent numpy recomputation of every C weight and activation row from the recorded statistics, the '
                           f'static error cache and the weight errors: max relative difference <= {TOL_REL}; normalization by the q0_knee '
                           'rows shown to differ',
            '6_validity': 'C covers exactly the 3051 searched projections; fallback, non-finite values, legal formats',
            '7_limits': 'C within the exact knee limits',
            '8_repeats': 'C repeats 1-3 identical tables / configurations (main = repeat 1); new B / D repeats equal the frozen B / D; the '
                         'end-to-end C run reproduces the static table, caches, knee and C configuration',
            '9_identical': 'identical configurations are reported and share one evaluation',
            'cache_use': 'every C / D run: act_error 0, act_assembly > 0, cache read > 0; every B run the reverse',
            'ste': 'STE unit test'},
        'timing': {'refresh': f'B / C / D, 3 each, interleaved {REFRESH_SCHEDULE}; one process per run; CUDA synchronization; no page-cache '
                              'dropping (residency recorded)',
                   'e2e': 'one end-to-end C run: static statistics (with the weight-error and activation caches) -> auto knee -> C -> search '
                          '(single measurement); historical A / B / D end-to-end times cited as historical; sums of stage medians called estimates',
                   'categories': CATEGORIES, 'separate': 'controls, repeats and the final evaluation'},
        'freeze': 'plan, code, inputs, tables, configurations and hashes frozen before any evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries, duplicate and '
                               'near-duplicate exclusion; every earlier manifest (incl. the previous knee round) excluded by scanning '
                               'runs/**/*segments*.json (only this run skipped by path)',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests, 'eligible_before_length_and_neardup': len(elig),
                       'seeds': SEEDS, 'conditions': 'FP32, A, B, C, D on the same new articles; original router; each its own dynamic route',
                       'primary': ['C - A', 'C - D', 'B - C'], 'negative': 'negative = the first condition is better',
                       'intervals': f'mean, 95% CI and Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI ({NCMP} comparisons); direction confirmed only '
                                    'by the corrected interval (upper < 0: first better; lower > 0: first worse)',
                       'context': 'B - A, D - A (95% CI)',
                       'noninferiority': f'C - B: corrected interval = the negated B - C corrected interval; upper < +{NI_MARGIN} -> C keeps B\'s '
                                         'quality within the margin; otherwise not shown',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'cautions': 'not confirmed is not equal; non-inferior is not better than static; no cross-batch NLL subtraction'},
        'conclusion_rules': ['C better than A and better than D: all-node sensitivity refresh helps at this knee; cheaper selection rules may be studied',
                             f'C better than A and non-inferior to B (+{NI_MARGIN}): weigh C\'s time saving as an optional step',
                             'B better than C: activation-error refresh adds a gain once all sensitivities are updated',
                             'C - A not confirmed: no claim that sensitivity is useless and no automatic attribution to activation errors; read '
                             'C - D and B - C together, say "cannot separate" if both include 0',
                             'never claimed: too few low-bit nodes as the only cause; all interactions separated; beyond this model / calibration / knee / GPU QDQ'],
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not prev['pass']:
        raise SystemExit('previous-round hashes differ: investigate first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 正常 knee 的全節點 sensitivity-only refresh（C）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- A／B／D 沿用上一輪凍結的配置（hash 核對）；C 用同一個 q0_knee、knee/knee.json 的精確上限重新搜尋一次，所有節點可改格式。',
          '- C：全部 3051 個搜尋投影、所有候選格式更新 sensitivity；weight／activation 誤差沿用 static，activation 以 static routed rows 正規化。',
          '- 實作：上一輪的 build，明確傳入 static activation 快取、不給 targets；每次執行以計時器驗證快取確實被使用（不能只把 variant 叫做 C）。',
          '- 控制：輸入 hash、forward、input gradient、C weight 列 = B、D 目標列 = C、numpy 獨立重算全部 C 列、合法與上限、重複一致、端到端重現。',
          f'- 計時：B／C／D 各 3 次交錯（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；C 端到端 1 次（單次實測）。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、C、D；seed 22601／22602／22611。',
          f'- 主要比較：C − A、C − D、B − C（負值代表前者較好）；95% CI 與 Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI，以校正區間確認方向。',
          f'- 另報 B − A、D − A；C − B 以 B − C 校正區間反號判讀非劣（+{NI_MARGIN}）。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; previous round ok {prev["pass"]}; previous WT103 {len(used)}; eligible {len(elig)}')


def controls(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'controls.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    q0k, ranges = read(Q0K), read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    rd = dr.forward(np.load(CALIB), q0k, dq, ranges, record=True, log=lambda m: None, label='q0_knee calibration', chunk=10)
    prevc = read(KS / 'controls.json')
    ctl = {'q0_knee_forward_ce': rd['ce'], 'q0_knee_forward_hidden_sha': rd['hidden_sha'], 'q0_knee_forward_routes_hash': dr.route_hash(rd['routes']),
           'equal_previous_round': {'ce': rd['ce'] == prevc['q0_knee_forward_ce'], 'hidden': rd['hidden_sha'] == prevc['q0_knee_forward_hidden_sha'],
                                    'routes': dr.route_hash(rd['routes']) == prevc['q0_knee_forward_routes_hash']},
           'ste_unit_test': pr.ste_unit_test(dq, ranges), 'previous_round': previous_round(),
           'page_cache_residency': ss.resident_fraction(ks.watch_files()), 'seconds': time.perf_counter() - t0}
    ctl['pass'] = bool(rd['finite_ce'] and all(ctl['equal_previous_round'].values()) and ctl['ste_unit_test']['pass'] and ctl['previous_round']['pass'])
    write(out / 'controls.json', ctl)
    log(json.dumps({'equal_previous_round': ctl['equal_previous_round'], 'pass': ctl['pass']}))


def refresh(out, log, variant, rep):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = f'refresh_{variant}_r{rep}'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(ks.watch_files())
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
        fm = read(Q0K)
        limits = read(KNEE)['limits']
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    with tm.block('cache_read_d2'):
        d2z = np.load(D2)
    cache = cidx = static_tab = targets = None
    if variant in ('C', 'D'):                    # sensitivity-only: the static activation-error statistics, explicitly
        with tm.block('cache_read'):
            cache = np.load(ACT_CACHE, mmap_mode='r')
            cidx = read(ACT_INDEX)
    if variant == 'D':
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
            targets = read(LOWBIT)['nodes']
    stats = {} if (variant == 'C' and rep == 1) else None
    r = ks.build(variant, dq, ranges, names, tm, log, fm=fm, d2z=d2z, cache=cache, cidx=cidx, static_tab=static_tab, targets=targets, stats_out=stats)
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'],
         'updated_projections': len(r['updated_projections']), 'state': f'q0_knee working state, condition {variant}'}
    fm_new, srch = ss.search_table(t, (names, nodes, elems, flops), limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm_new)
    with tm.block('controls'):
        np.savez(rd / 'q0_knee_calibration_routes.npz', **r['routes'])
        if stats is not None:
            np.savez(rd / 'target_stats.npz', **{f'{n}|{k}': v for n, d in stats.items() for k, v in d.items() if isinstance(v, np.ndarray)})
            write(rd / 'target_stats_meta.json', {n: {k: v for k, v in d.items() if not isinstance(v, np.ndarray)} for n, d in stats.items()})
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'search': srch,
                              'cache_use': cache_use(variant, tm.t), 'updated_projections': len(r['updated_projections']),
                              'fallback_static_rows': r['fallback_static_rows'], 'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'],
                              'routes_hash': dr.route_hash(r['routes']), 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'],
                              'recompute_max_abs': max(r['recompute_max_abs_per_layer']), 'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']),
                              'q0_routed_rows_by_expert': {k: v['routed_rows'] for k, v in r['meta'].items()}, 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path'], 'cache_use': rec['cache_use']}))


def e2e(out, log):
    """one process: static statistics (+ weight-error and activation caches) -> auto knee -> C (caches just written) -> search."""
    import torch
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    tag = 'e2e_C'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(ks.watch_files())
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
    prob = (names, nodes, elems, flops)
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    with tm.block('cache_build'):
        experts = sorted({'.'.join(n.split('.')[:2]) for n in names}, key=lambda s: (int(s.split('.')[0][1:]), int(s.split('.')[1][1:])))
        collect = {'row': {e: i for i, e in enumerate(experts)}, 'n': {}, 'data': np.zeros((len(experts), 2, 10, ss.NCOL), np.float64)}
    st = ks.build('static', dq, ranges, names, tm, log, collect=collect)
    t = {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'}
    marks = {'static_done_s': time.perf_counter() - t0}
    with tm.block('table_write'):
        write(rd / 'table_static.json', t)
    with tm.block('cache_write_d2'):
        np.savez(rd / 'weight_error_d2.npz', **{f'{n}|{f}': v for n, d in st['d2'].items() for f, v in d.items()})
    with tm.block('cache_write'):
        np.save(rd / 'act_err_cache.npy', collect['data'])
        write(rd / 'act_err_cache_index.json', {'experts': list(collect['row']), 'row': collect['row'], 'n': collect['n'], 'formats': list(QUANT),
                                                'columns': ss.COLS, 'stats': ['sum_err', 'sum_err_sq'], 'shape': list(collect['data'].shape), 'dtype': 'float64'})
    marks['static_written_s'] = time.perf_counter() - t0
    del st, collect
    sel, recs, slog, kept = ks.auto_search(t, prob, tm, out / 'logs', tag)
    q0k, limits = ks.knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'q0_knee.json', q0k)
    marks['knee_done_s'] = time.perf_counter() - t0
    with tm.block('cache_read_d2'):
        d2z = np.load(rd / 'weight_error_d2.npz')
    with tm.block('cache_read'):
        cache = np.load(rd / 'act_err_cache.npy', mmap_mode='r')
        cidx = read(rd / 'act_err_cache_index.json')
    r = ks.build('C', dq, ranges, names, tm, log, fm=q0k, d2z=d2z, cache=cache, cidx=cidx, static_tab=t)
    tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'],
          'updated_projections': len(r['updated_projections']), 'state': 'q0_knee working state, condition C'}
    fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', tt)
        write(rd / 'config.json', fm_new)
    rec = run_record(tm, t0, {'variant': 'C', 'page_cache_residency_before': residency, 'marks': marks, 'limits': limits,
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'auto_paths': dict(Counter(x.get('path') for x in recs)),
                              'search': srch, 'cache_use': cache_use('C', tm.t), 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'],
                              'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward'], 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'marks': marks, 'cache_use': rec['cache_use']}))


def timing(out, log, phase):
    sched = REFRESH_SCHEDULE if phase == 'refresh' else [('C', None)]
    for v, r in sched:
        name = f'refresh_{v}_r{r}' if phase == 'refresh' else 'e2e_C'
        if (out / 'work/runs' / name / 'run.json').exists():
            log(f'{name}: done already, kept')
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', phase, '--out-dir', str(out)]
        cmd += ['--variant', v, '--rep', str(r)] if phase == 'refresh' else []
        t = time.time()
        p = subprocess.run(cmd, cwd=HERE)
        log(f'{name}: exit {p.returncode}, process wall {time.time() - t:.1f}s')
        if p.returncode:
            raise SystemExit(f'{name} failed')


def check(out, log):
    t0 = time.perf_counter()
    if (out / 'check.json').exists():
        raise SystemExit('exists')
    from run_native_bc_pipeline import GRID11, QUANT
    names, nodes, elems, flops, rows = pr.problem_static()
    pl = read(out / 'plan.json')
    W = out / 'work/runs'
    rr = [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE]
    exp = rr + ['e2e_C']
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    ctl, prevc = read(out / 'controls.json'), read(KS / 'controls.json')
    st, kn = read(STATIC_TABLE), read(KNEE)
    tB, tD = read(KS / 'tables/B.json'), read(KS / 'tables/D.json')
    cB, cD = read(KS / 'configs/B.json'), read(KS / 'configs/D.json')
    tC, cC = read(W / 'refresh_C_r1/table.json'), read(W / 'refresh_C_r1/config.json')
    low = read(LOWBIT)['nodes']
    prev = previous_round()
    res['1_inputs'] = {'previous_round': prev, 'pass': prev['pass']}
    drr = dr.route_hash(dict(np.load(KS / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    c2 = {k: {'ce': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'] == prevc['q0_knee_forward_ce'],
              'hidden': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'], 'routes': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr}
          for k in rr}
    c2['e2e_C'] = {'ce': runs['e2e_C']['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden': runs['e2e_C']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha']}
    res['2_forward'] = {'runs': c2, 'pass': all(all(v.values()) for v in c2.values())}
    g_prev = read(KS / 'work/runs/refresh_B_r1/run.json')['input_grad_sha_per_layer']
    res['3_gradients'] = {'identical_all_runs_and_previous_round': all(runs[k]['input_grad_sha_per_layer'] == g_prev for k in exp),
                          'recompute_max_abs': max(runs[k]['recompute_max_abs'] for k in rr)}
    res['3_gradients']['pass'] = res['3_gradients']['identical_all_runs_and_previous_round'] and res['3_gradients']['recompute_max_abs'] == 0
    fbC = set(runs['refresh_C_r1']['fallback_static_rows'])
    res['4_rows'] = {'C_weight_rows_equal_B': all(tC['B'][n] == tB['B'][n] for n in names),
                     'D_target_rows_equal_C': all(tD['B'][n] == tC['B'][n] and tD['C'][n] == tC['C'][n] for n in low),
                     'C_act_rows_differ_from_B': sum(tC['C'][n] != tB['C'][n] for n in names),
                     'C_act_rows_differ_from_static': sum(tC['C'][n] != st['C'][n] for n in names)}
    res['4_rows']['pass'] = bool(res['4_rows']['C_weight_rows_equal_B'] and res['4_rows']['D_target_rows_equal_C'])
    z = np.load(W / 'refresh_C_r1/target_stats.npz')
    sm = read(W / 'refresh_C_r1/target_stats_meta.json')
    d2z, cache, cidx = np.load(D2), np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    relw = rela = relu = relq = 0.0
    for n in names:
        if n in fbC:
            continue
        node, p = '.'.join(n.split('.')[:2]), n.split('.')[2]
        Gc, nin, nq = z[f'{n}|Gc'], sm[n]['n_in'], sm[n]['n_q0']
        row, ns = np.asarray(cache[cidx['row'][node]]), cidx['n'][node]
        for i, f in enumerate(QUANT):
            wv = float((Gc / nin * d2z[f'{n}|{f}']).sum())
            av = au = aq = 0.0
            for s in ('x', 'y'):
                a_, b_ = ss.COLOFF[ss.SLOT_COL[(p, s)]]
                S1_, S2_ = row[0, i, a_:b_], row[1, i, a_:b_]
                G2, GS = z[f'{n}|{s}_G2'], z[f'{n}|{s}_GS']
                mu = S1_ / ns
                av += float(np.sum(G2 / ns * (S2_ - ns * mu ** 2)) + np.sum(GS * mu ** 2))
                au += float(np.sum(G2 * (S2_ / ns - mu ** 2)) + np.sum(GS * mu ** 2))
                mq = S1_ / nq
                aq += float(np.sum(G2 / nq * (S2_ - nq * mq ** 2)) + np.sum(GS * mq ** 2))
            relw, rela = max(relw, rr_(wv, tC['B'][n][f])), max(rela, rr_(av, tC['C'][n][f]))
            relu, relq = max(relu, rr_(au, tC['C'][n][f])), max(relq, rr_(aq, tC['C'][n][f]))
    res['5_recompute'] = {'rows': len(names) - len(fbC), 'weight_max_rel': relw, 'act_max_rel': rela,
                          'descriptive_user_form_max_rel': relu, 'descriptive_q0_rows_normalization_max_rel': relq,
                          'pass': relw <= TOL_REL and rela <= TOL_REL}
    res['6_validity'] = {'C_covers_exactly_searched': sorted(cC) == sorted(names) and sorted(tC['B']) == sorted(names) and sorted(tC['C']) == sorted(names),
                         'fallback_count': len(fbC), 'fallback_rows': sorted(fbC),
                         'fallback_reason': 'searched expert with no q0_knee calibration token (static row kept)',
                         'finite': all(runs[k]['grads_finite'] and runs[k]['nonfinite'] == 0 for k in rr),
                         'finite_note': 'checked on the 9 refresh runs (the end-to-end record has no finiteness fields; its C table equals C repeat 1 bitwise)',
                         'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in cC.values())}
    res['6_validity']['pass'] = bool(res['6_validity']['C_covers_exactly_searched'] and res['6_validity']['finite'] and res['6_validity']['legal'])
    lim = kn['limits']
    from run_olmoe_multipoint import resources
    rc = resources(cC, names, elems, flops)
    res['7_limits'] = {'resources': rc, 'within': rc['M_bytes'] <= lim['max_bytes'] and rc['BOPs'] <= lim['max_bops'],
                       'search_within_budget_all_runs': all(runs[k]['search']['within_budget'] for k in exp), 'e2e_limits_equal': runs['e2e_C']['limits'] == lim}
    res['7_limits']['pass'] = bool(res['7_limits']['within'] and res['7_limits']['search_within_budget_all_runs'] and res['7_limits']['e2e_limits_equal'])
    reps = {}
    for v, tref, cref in (('C', tC, cC), ('B', tB, cB), ('D', tD, cD)):
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        reps[v] = {'tables': all({kk: vv for kk, vv in read(W / k / 'table.json').items() if kk in ('B', 'C')} == {kk: tref[kk] for kk in ('B', 'C')} for k in ks_),
                   'configs': all(read(W / k / 'config.json') == cref for k in ks_)}
    reps['e2e_C'] = {'static_table': {kk: vv == st[kk] for kk, vv in read(W / 'e2e_C/table_static.json').items() if kk in ('B', 'C')},
                     'weight_error': ss.d2_equal(W / 'e2e_C/weight_error_d2.npz', D2), 'act_cache': sha(W / 'e2e_C/act_err_cache.npy') == sha(ACT_CACHE),
                     'knee': read(W / 'e2e_C/q0_knee.json') == read(Q0K), 'C_table': {kk: read(W / 'e2e_C/table.json')[kk] == tC[kk] for kk in ('B', 'C')},
                     'C_config': read(W / 'e2e_C/config.json') == cC}
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['8_repeats'] = reps
    res['8_repeats']['pass'] = flat(reps)
    cfgs = {'A': read(KS / 'configs/A.json'), 'B': cB, 'C': cC, 'D': cD}
    res['9_identical'] = {f'{a}={b}': cfgs[a] == cfgs[b] for a, b in (('C', 'A'), ('C', 'B'), ('C', 'D'))}
    res['cache_use'] = {k: runs[k]['cache_use'] for k in exp}
    eu = runs['e2e_C']['cache_use']
    res['cache_use_e2e_rule'] = ('end-to-end: its act_error block is the static-statistics stage that builds the cache; the C stage must show cache '
                                 'read and assembly blocks, and its C table must equal C repeat 1 bitwise (8_repeats)')
    res['cache_use_pass'] = bool(all(runs[k]['cache_use']['as_expected'] for k in rr) and eu['act_assembly_s'] > 0 and eu['act_cache_read_s'] > 0
                                 and all(reps['e2e_C']['C_table'].values()))
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['inputs_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['pass'] = bool(not res['missing_runs'] and all(res[k]['pass'] for k in ('1_inputs', '2_forward', '3_gradients', '4_rows', '5_recompute',
                                                                                 '6_validity', '7_limits', '8_repeats'))
                       and res['cache_use_pass'] and res['inputs_unchanged'] and res['ste'] and res['controls_stage_pass'])
    res['seconds'] = time.perf_counter() - t0
    write(out / 'check.json', res)
    log(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in res.items() if k not in ('cache_use', 'seconds')}))


def freeze(out, log):
    from mp_auto import config_hash
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'check.json')['pass']:
        raise SystemExit('controls failed: not freezing')
    (out / 'configs').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    W = out / 'work/runs'
    for c in 'ABD':
        write(out / f'configs/{c}.json', read(KS / f'configs/{c}.json'))
    write(out / 'configs/C.json', read(W / 'refresh_C_r1/config.json'))
    write(out / 'tables/C.json', read(W / 'refresh_C_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in 'ABCD'}
    names = sorted(cfg['A'])
    from run_olmoe_multipoint import resources
    _, nodes, elems, flops, _ = pr.problem_static()
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    low = set(read(LOWBIT)['nodes'])
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'C = refresh repeat 1; A / B / D = previous round frozen',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c]), 'resources': resources(cfg[c], names, elems, flops)} for c in cfg},
           'ABD_equal_previous_round': all(sha(out / f'configs/{c}.json') == read(KS / 'freeze.json')['configs'][c]['sha256'] for c in 'ABD'),
           'identical': {f'{a}={b}': cfg[a] == cfg[b] for a, b in (('C', 'A'), ('C', 'B'), ('C', 'D'), ('B', 'D'), ('A', 'B'), ('A', 'D'))},
           'pairwise_differences': {f'{a}-{b}': diff(cfg[a], cfg[b]) for a, b in (('C', 'A'), ('C', 'B'), ('C', 'D'), ('B', 'A'), ('D', 'A'), ('D', 'B'))},
           'changed_vs_A_in_lowbit_set': {c: {'lowbit': sum(cfg[c][n] != cfg['A'][n] for n in low), 'other': sum(cfg[c][n] != cfg['A'][n] for n in names if n not in low)}
                                          for c in 'BCD'},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in 'BCD'},
           'tables': {'C.json': sha(out / 'tables/C.json'), **{f'previous/{n}': h for n, h in read(KS / 'freeze.json')['tables'].items()}},
           'knee_sha256': sha(KNEE), 'lowbit_nodes_sha256': sha(LOWBIT), 'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'),
           'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_in_lowbit_set')}))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    used, _ = ss.used_wt103_all(out)
    if len(used) != pl['evaluation']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    excl, base = Counter(), []
    for a in w103:
        if a['nt'] in T2 or a['ch'] in C2:
            excl['wikitext2_title_or_content'] += 1
        elif ct[a['nt']] > 1 or cc[a['ch']] > 1:
            excl['wt103_internal_duplicate'] += 1
        else:
            base.append(a)
    prev = [a for a in base if a['ch'] in used]
    if len(prev) != len(used):
        raise SystemExit('previous articles not all found')
    excl['previous_olmoe_wt103'] = len(prev)
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['evaluation']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(a['body_norm']) for a in prev]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    segs, toks, acc_sh, skips, nd = [], [], [], Counter(), []
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc['input_ids'], enc['offset_mapping']
        if len(ids) < WPA * SEQ:
            skips['short'] += 1; continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if max(c2, cp, ca) >= mr.NEARDUP:
            skips['near_duplicate'] += 1; nd.append({'title': a['title'], 'wt2': c2, 'previous': cp, 'accepted': ca}); continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']], 'title': a['title'],
                     'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids), 'window_tokens': [st, st + WPA * SEQ],
                     'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                     'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca}, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
                     'windows': [WPA * len(segs) + i for i in range(WPA)]})
        if len(segs) == N:
            break
    if len(segs) < N or {s['content_sha256'] for s in segs} & used:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'dedup_checks.json', {'exclusions': dict(excl), 'eligible': len(elig), 'visited': N + sum(skips.values()), 'skips': dict(skips),
                                      'near_duplicates': nd, 'overlap_with_previous': 0,
                                      'max_share': {k: max(s['max_shingle_share'][k] for s in segs) for k in ('wt2', 'previous', 'accepted')}})
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'), 'n': N,
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json')})
    ss.merge_timing(out, 'data', {'seconds': time.perf_counter() - t0, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(f'data frozen: {N}; skips {dict(skips)}; excluded previous {len(prev)}')


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
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    torch.cuda.reset_peak_memory_stats()
    old_t, old = np.load(KS / 'tokens.npy')[:CTRL], read(KS / 'eval.json')['ce']
    pc = {}
    for c, fm in (('fp32', None), ('A', read(KS / 'configs/A.json')), ('B', read(KS / 'configs/B.json')), ('D', read(KS / 'configs/D.json'))):
        r = dr.forward(old_t, fm, dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs, shared, by_hash = {}, {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in 'ABCD':
        h = fz['configs'][c]['config_hash']
        if h in by_hash:
            runs[c] = runs[by_hash[h]]; shared[c] = by_hash[h]
            continue
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        by_hash[h] = c
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/C.json'), dq, ranges, record=True, log=log, label='rerun C')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_C_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['C']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['C']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['C']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in 'ABCD'},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in 'ABCD'}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_C_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_C_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_C_first_chunk']['routes_equal'] and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl,
                              'seconds': {c: runs[c]['seconds'] for c in runs if c not in shared}, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                              'environment': dr.env_info()})
    ss.merge_timing(out, 'eval', {**{f'{c}_s': runs[c]['seconds'] for c in runs if c not in shared}, 'rerun_s': r2['seconds'],
                                  'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps({'controls_pass': ctl['pass'], 'program_control': pc, 'shared': shared}))


def timing_summary(out):
    W = out / 'work/runs'
    runs = {p.name: read(p / 'run.json') for p in sorted(W.iterdir()) if (p / 'run.json').exists()}
    jl = [json.loads(x) for x in (out / 'timing.jsonl').read_text().splitlines()]
    proc = {(f'refresh_{j["variant"]}_r{j["rep"]}' if j['stage'] == 'refresh' else 'e2e_C'): j['seconds'] for j in jl if j['stage'] in ('refresh', 'e2e')}
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in 'BCD':
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        s[f'refresh_{v}'] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                             'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                             'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks_]) for t in sorted({t for k in ks_ for t in runs[k]['timers']})},
                             'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                             'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks_}}
    e = runs['e2e_C']
    s['e2e_C_measured_single'] = {'method_s': e['method_s'], 'process_wall_s': proc['e2e_C'], 'marks': e['marks'], 'categories': e['categories'],
                                  'peak_gpu_gb': e['peak_gpu_gb'], 'peak_cpu_gb': e['peak_cpu_gb'], 'page_cache_residency_before': e['page_cache_residency_before']}
    kt = read(KS / 'analysis.json')['timing']
    s['historical_previous_round'] = {'e2e_measured_single': {v: kt['e2e_measured'][v]['method_s'] for v in 'ABD'},
                                      'auto_median_s': kt['auto']['method_s']['median'],
                                      'refresh_median_s': {v: kt[f'refresh_{v}']['method_s']['median'] for v in 'BD'},
                                      'static_statistics_component': kt['total_estimate_sum_of_stage_medians_s']['static_statistics_component'],
                                      'note': 'previous round, same machine and code; historical values'}
    cm = s['refresh_C']['method_s']['median']
    s['C_total_estimate_sum_of_stage_medians_s'] = kt['total_estimate_sum_of_stage_medians_s']['static_statistics_component']['D'] + kt['auto']['method_s']['median'] + cm
    s['extra_with_caches_median_s'] = {v: s[f'refresh_{v}']['method_s']['median'] for v in 'BCD'}
    s['note'] = 'method = run time minus control blocks; the C estimate = static statistics with the activation cache (selective-sensitivity medians) + previous-round auto median + C median'
    return s


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    if not ev['controls']['pass']:
        raise SystemExit('evaluation controls failed')
    A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))

    def st(x):
        m = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(m, [2.5, 97.5]).tolist(), 'ci_bonferroni': np.percentile(m, CI_CORR).tolist(),
                'article_sd': float(x.std(ddof=1)), 'better_articles': int((x < 0).sum()), 'worse_articles': int((x > 0).sum()), 'zero_articles': int((x == 0).sum())}
    D = {'C_minus_A': A['C'] - A['A'], 'C_minus_D': A['C'] - A['D'], 'B_minus_C': A['B'] - A['C'], 'B_minus_A': A['B'] - A['A'], 'D_minus_A': A['D'] - A['A']}
    res = {'n_articles': N, 'bonferroni_level': 100 - 2 * CI_CORR[0], 'primary': {k: st(D[k]) for k in ('C_minus_A', 'C_minus_D', 'B_minus_C')},
           'context': {k: st(D[k]) for k in ('B_minus_A', 'D_minus_A')}, 'damage': {c: st(A[c] - A['fp32']) for c in 'ABCD'}}
    for k, v in res['primary'].items():
        lo, hi = v['ci_bonferroni']
        v['direction_corrected'] = 'first_better_confirmed' if hi < 0 else 'first_worse_confirmed' if lo > 0 else 'not_confirmed'
    for k, v in res['context'].items():
        lo, hi = v['ci95']
        v['direction_95'] = 'first_better_confirmed' if hi < 0 else 'first_worse_confirmed' if lo > 0 else 'not_confirmed'
    blo, bhi = res['primary']['B_minus_C']['ci_bonferroni']
    res['C_minus_B_noninferiority'] = {'mean': -res['primary']['B_minus_C']['mean'], 'ci_bonferroni': [-bhi, -blo], 'margin': NI_MARGIN,
                                       'decision': (f'C_noninferior_to_B_within_{NI_MARGIN}' if -blo < NI_MARGIN else
                                                    f'C_worse_than_B_by_more_than_{NI_MARGIN}' if -bhi > NI_MARGIN else 'not_shown')}
    p_ = {k: v['direction_corrected'] for k, v in res['primary'].items()}
    rules = []
    if p_['C_minus_A'] == 'first_better_confirmed' and p_['C_minus_D'] == 'first_better_confirmed':
        rules.append('C_better_than_A_and_D: all-node sensitivity refresh helps at this knee')
    if p_['C_minus_A'] == 'first_better_confirmed' and res['C_minus_B_noninferiority']['decision'].startswith('C_noninferior'):
        rules.append('C_better_than_A_and_noninferior_to_B: weigh the time saving')
    if p_['B_minus_C'] == 'first_better_confirmed':
        rules.append('B_better_than_C: activation-error refresh adds a gain beyond all-node sensitivities')
    if p_['C_minus_A'] != 'first_better_confirmed':
        ci_cd, ci_bc = res['primary']['C_minus_D']['ci_bonferroni'], res['primary']['B_minus_C']['ci_bonferroni']
        both0 = ci_cd[0] <= 0 <= ci_cd[1] and ci_bc[0] <= 0 <= ci_bc[1]
        rules.append('C_minus_A_not_confirmed' + ('; C-D and B-C both include 0: cannot separate scope from error refresh' if both0 else
                                                  '; read C-D and B-C intervals together'))
    res['conclusion_rules'] = rules
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in 'ABCD'}
    names = sorted(cfg['A'])
    tabs = {'static': read(STATIC_TABLE), 'B': read(KS / 'tables/B.json'), 'C': read(out / 'tables/C.json'), 'D': read(KS / 'tables/D.json')}
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    res['objectives_descriptive'] = {t: {c: obj(tabs[t], cfg[c]) for c in 'ABCD'} for t in tabs}
    res['search_C'] = {k: read(out / 'work/runs/refresh_C_r1/run.json')['search'][k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use',
                                                                                                 'weight_formats', 'act_formats', 'within_budget')}
    res['configs'] = {c: v['resources'] for c, v in fz['configs'].items()}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_in_lowbit_set', 'transitions_vs_A')}
    low = set(read(LOWBIT)['nodes'])
    lowexp = {'.'.join(n.split('.')[:2]) for n in low}
    from run_olmoe_multipoint import problem
    _, _, pnames, _, elems, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    for c in ['fp32', 'A', 'B', 'C', 'D']:
        z = np.load(out / f'routes_{shared.get(c, c)}.npz')
        ids = z['ids'].astype(np.int64)
        cnt = np.stack([np.bincount(ids[:, l].ravel(), minlength=NE) for l in range(NL)])
        counts[c] = cnt
        p = cnt / cnt.sum(1, keepdims=True)
        desc[c] = {'selected_weight_sum_mean': float(z['wsum'].mean()), 'max_expert_share_mean': float(p.max(1).mean())}
    searched = {(int(n.split('.')[0][1:]), int(n.split('.')[1][1:])) for n in pnames}
    unsearched = [(l, e) for l in range(NL) for e in range(NE) if (l, e) not in searched]
    rowsof = lambda cnt, n: cnt[int(n.split('.')[0][1:]), int(n.split('.')[1][1:])]
    elem_u = 2048 * 1024
    b8_s = sum(rowsof(counts['fp32'], n) * elems[n] * 64 for n in pnames)
    b8_all = b8_s + sum(counts['fp32'][l, e] * 3 * elem_u * 64 for l, e in unsearched)
    cost = {}
    for c in 'ABCD':
        sb = float(sum(rowsof(counts[c], n) * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
        ub = float(sum(counts[c][l, e] * 3 * elem_u * 16 * 32 for l, e in unsearched))
        cost[c] = {'searched_BOPs_actual': sb, 'relative_searched_to_int8_fp32_counts': sb / b8_s, 'unsearched_BOPs_bf16_fp32': ub,
                   'total_BOPs_actual': sb + ub, 'relative_total_to_all_int8_fp32_counts': (sb + ub) / b8_all,
                   'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
                   'routed_rows_to_lowbit_experts': int(sum(counts[c][int(e.split('.')[0][1:]), int(e.split('.')[1][1:])] for e in lowexp)),
                   'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
        if c != 'A':
            desc[c]['set_change_vs_A'] = float((np.sort(np.load(out / f'routes_{shared.get(c, c)}.npz')['ids'], -1) !=
                                                np.sort(np.load(out / f'routes_{shared.get("A", "A")}.npz')['ids'], -1)).any(-1).mean())
    cost['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l, e in unsearched))
    res['routing_cost'] = cost
    res['routing_descriptive'] = desc | {'set_change_vs_fp32': ev['controls']['routes_differ_from_fp32']}
    res['timing'] = timing_summary(out)
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    segs = read(out / 'segments.json')
    with open(out / 'per_article.csv', 'w') as fh:
        rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, **{k: float(v[i]) for k, v in D.items()}}
                for i, s in enumerate(segs)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / 'per_window.csv', 'w') as fh:
        rows = [{'window': k, 'article': k // WPA, **{f'summed_ce_{c}': ev['ce'][c][k] for c in ev['ce']}} for k in range(N * WPA)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(out / 'controls_summary.json', {'controls_stage': read(out / 'controls.json'), 'check': read(out / 'check.json'), 'evaluation': ev['controls'],
                                          'input_hashes_unchanged': res['input_hashes_unchanged']})
    ss.merge_timing(out, 'analysis', {'seconds': time.perf_counter() - t0})
    write(out / 'analysis.json', res)
    log(json.dumps({'primary': {k: (v['mean'], v['ci_bonferroni'], v['direction_corrected']) for k, v in res['primary'].items()},
                    'noninferiority': res['C_minus_B_noninferiority'], 'rules': rules}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'controls', 'refresh', 'e2e', 'timing', 'check', 'freeze', 'data', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant', choices=['B', 'C', 'D'])
    ap.add_argument('--rep', type=int)
    ap.add_argument('--phase', choices=['refresh', 'e2e'])
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    if a.stage == 'refresh':
        refresh(a.out_dir, log, a.variant, a.rep)
    elif a.stage == 'e2e':
        e2e(a.out_dir, log)
    elif a.stage == 'timing':
        timing(a.out_dir, log, a.phase)
    else:
        {'eval': eval_}.get(a.stage, globals().get(a.stage))(a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'variant': a.variant, 'rep': a.rep, 'phase': a.phase, 'seconds': time.time() - T_PROC,
                            'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
