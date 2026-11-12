#!/usr/bin/env python3
"""OLMoE normal knee: <= 5-bit selective sensitivity refresh (D5), next to the frozen A / B / D4
(runs/olmoe_knee_selective_sensitivity_20261022) and C (runs/olmoe_knee_all_sensitivity_20261022).
Pre-registered in plan.json / plan.md. The <= 5-bit threshold is a new hypothesis proposed after the D4 and C
results were seen; it is tested on new evaluation articles.

D5: targets = q0_knee projections whose weight or activation format has <= 5 bits; their weight and activation
sensitivities (every candidate format) from q0_knee's working state with the static error statistics (cache passed
explicitly, static routed-row normalization); every other row static. One search at the exact knee limits.
Same code path: run_olmoe_knee_selective_sensitivity.build (targets -> only the target taps / statistics are collected,
the full backward path is kept).

Stages (one --out-dir): plan, controls, refresh (--variant B|C|D4|D5 --rep k), e2e, timing (--phase refresh|e2e),
check, freeze, data, eval, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
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
KA = HERE / 'runs/olmoe_knee_all_sensitivity_20261022'
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE, D2, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.D2, ks.ACT_CACHE, ks.ACT_INDEX
Q0K, KNEE, LOW4 = KS / 'knee/q0_knee.json', KS / 'knee/knee.json', KS / 'lowbit_nodes.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, NCMP, LOW_MAX = 10000, 0.0005, 1e-12, 3, 5
CI_CORR = [100 * 0.05 / (2 * NCMP), 100 * (1 - 0.05 / (2 * NCMP))]      # Bonferroni over the 3 primary comparisons: 98.33%
SEEDS = {'order': 22701, 'start': 22702, 'bootstrap': 22711}
VARIANTS = ('B', 'C', 'D4', 'D5')
REFRESH_SCHEDULE = [('B', 1), ('C', 1), ('D4', 1), ('D5', 1), ('C', 2), ('D4', 2), ('D5', 2), ('B', 2), ('D4', 3), ('D5', 3), ('B', 3), ('C', 3)]
FROZEN = {'A': KS / 'configs/A.json', 'B': KS / 'configs/B.json', 'C': KA / 'configs/C.json', 'D4': KS / 'configs/D.json'}
FROZEN_TABLES = {'B': KS / 'tables/B.json', 'C': KA / 'tables/C.json', 'D4': KS / 'tables/D.json'}
CATEGORIES = ks.CATEGORIES


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_low5_sensitivity.py', 'run_olmoe_knee_selective_sensitivity.py',
                                                  'run_olmoe_knee_all_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def low5_nodes(q0):
    from htq_pareto_search import FORMAT_BITS
    return sorted(n for n, v in q0.items() if FORMAT_BITS[v['w']] <= LOW_MAX or FORMAT_BITS[v['a']] <= LOW_MAX)


def history():
    """frozen A / B / C / D4 configurations and tables, knee, D4 targets, checks and reuse dependencies vs their freeze records."""
    fs, fa = read(KS / 'freeze.json'), read(KA / 'freeze.json')
    res = {'knee_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in fs['configs'].items()},
           'knee_round_tables': {n: sha(KS / 'tables' / n) == h for n, h in fs['tables'].items()},
           'all_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in fa['configs'].items()},
           'all_round_table_C': sha(KA / 'tables/C.json') == fa['tables']['C.json'],
           'knee': sha(KNEE) == fs['knee_sha256'], 'd4_targets': sha(LOW4) == fs['lowbit_nodes_sha256'],
           'checks_pass': read(KS / 'check.json')['pass'] and read(KA / 'check.json')['pass'],
           'code_unchanged': {'run_olmoe_knee_selective_sensitivity.py': sha(HERE / 'run_olmoe_knee_selective_sensitivity.py') == fs['code']['run_olmoe_knee_selective_sensitivity.py'],
                              'run_olmoe_selective_sensitivity.py': sha(HERE / 'run_olmoe_selective_sensitivity.py') == fs['code']['run_olmoe_selective_sensitivity.py']},
           'q0_knee_equals_A': read(Q0K) == read(FROZEN['A'])}
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
    """sensitivity-only (C / D4 / D5) must read the static activation-error cache and compute no activation error; B the reverse."""
    a_err, a_asm, c_rd = timers.get('act_error', 0.0), timers.get('act_assembly', 0.0), timers.get('cache_read', 0.0)
    ok = (a_err == 0 and a_asm > 0 and c_rd > 0) if variant != 'B' else (a_err > 0 and a_asm == 0 and c_rd == 0)
    return {'act_error_s': a_err, 'act_assembly_s': a_asm, 'act_cache_read_s': c_rd, 'as_expected': ok}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    hist = history()
    names, nodes, elems, flops, rows = pr.problem_static()
    q0 = read(FROZEN['A'])
    t5, t4 = low5_nodes(q0), read(LOW4)['nodes']
    cidx = read(ACT_INDEX)
    by = lambda ns: {'by_proj': dict(Counter(n.split('.')[2] for n in ns)), 'by_layer': {str(l): v for l, v in sorted(Counter(int(n.split('.')[0][1:]) for n in ns).items())},
                     'weight_formats': dict(Counter(q0[n]['w'] for n in ns)), 'act_formats': dict(Counter(q0[n]['a'] for n in ns)),
                     'experts': len({'.'.join(n.split('.')[:2]) for n in ns}),
                     'fp32_calibration_rows_of_experts': int(sum(cidx['n'][e] for e in {'.'.join(n.split('.')[:2]) for n in ns}))}
    write(out / 'low5_nodes.json', {'definition': f'q0_knee (= frozen A) weight or activation format with <= {LOW_MAX} bits (htq_pareto_search.FORMAT_BITS)',
                                    'background': 'threshold proposed after the D4 (<= 4-bit) and C (all-node) results were seen; new evaluation articles',
                                    'q0_sha256': sha(FROZEN['A']), 'count': len(t5), 'expected': 1512, 'total': len(names),
                                    'd4_subset': set(t4) <= set(t5), 'd4_count': len(t4), 'new_5bit': len(set(t5) - set(t4)), 'outside': len(names) - len(t5),
                                    'fp32_calibration_rows_total': int(sum(cidx['n'].values())), **by(t5), 'new_5bit_detail': by(sorted(set(t5) - set(t4))),
                                    'nodes': t5})
    if len(t5) != 1512 or not set(t4) <= set(t5):
        raise SystemExit(f'D5 count {len(t5)} or D4 subset differ: check the inputs and the rule first')
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = ss.input_files() + list(FROZEN.values()) + list(FROZEN_TABLES.values()) + [
        KS / f for f in ('plan.json', 'freeze.json', 'check.json', 'controls.json', 'tables/static.json', 'knee/knee.json', 'knee/q0_knee.json',
                         'lowbit_nodes.json', 'work/q0_knee_calibration_routes_dr_forward.npz')] + [
        KA / f for f in ('plan.json', 'freeze.json', 'check.json', 'controls.json', 'eval.json', 'tokens.npy')] + [ACT_CACHE, ACT_INDEX]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'At the normal knee, does the <= 5-bit selective sensitivity refresh (D5) improve the static BC, improve on the <= 4-bit '
                    'refresh (D4), and keep the full refresh\'s (B) quality at a lower cost?',
        'background': 'post hoc hypothesis: the <= 5-bit threshold was proposed after the D4 (no gain at the knee) and C results were seen; '
                      'it is tested once on new evaluation articles; a success supports only this model / knee / calibration, not 5 bits as a '
                      'general threshold; no <= 6-bit follow-up in this round',
        'not_this_round': 'no new knee, no loss-based fallback, no fine-tuning, no parameter update, no BC formula change, no threshold change, '
                          'no default-pipeline change',
        'fixed': {'history': {'knee_round': str(KS.relative_to(HERE)), 'all_round': str(KA.relative_to(HERE)), 'hashes': hist},
                  'frozen_configs': {c: str(p.relative_to(HERE)) for c, p in FROZEN.items()}, 'limits': read(KNEE)['limits'],
                  'same': 'checkpoint and router, 10 x 128 calibration, ranges, corrected quantizer, q0_knee working state, 3051 searched '
                          'projections, GRID11 x GRID11, resource model, LP-first search with exact dominance pruning, exact knee limits'},
        'D5': {'targets': 'low5_nodes.json: FORMAT_BITS[q0.w] <= 5 or FORMAT_BITS[q0.a] <= 5 on the frozen A; recomputed, frozen',
               'targets_count': len(t5), 'd4_subset': True, 'new_5bit': len(set(t5) - set(t4)), 'outside': len(names) - len(t5),
               'update': 'weight and activation sensitivities of every candidate format of a target from one q0_knee calibration forward (own '
                         'dynamic routes) and backward (recorded routes constant, identity STE, gate / up shared input with its total gradient); '
                         'static weight / activation error statistics (cache passed explicitly, static routed-row normalization); non-targets '
                         'keep their static rows bitwise; one search at the exact limits; every projection may change format; no second refresh',
               'timing_rule': 'D5 runs collect only the target taps / statistics (run_olmoe_knee_selective_sensitivity.build with targets), never C '
                              'statistics masked afterwards'},
        'controls': {
            '1_history': 'A / B / C / D4 configurations, tables, inputs and cache hashes equal their records',
            '2_state': 'D5 (and the new B / C / D4 timing runs) forward CE, hidden sha256, routes and per-layer input-gradient sha256 equal the '
                       'same q0_knee working state (independent forward of this round and the previous records)',
            '3_targets_equal_C': 'D5 target rows bitwise equal the frozen C rows',
            '4_nontargets_static': 'D5 non-target rows bitwise equal the static table',
            '5_recompute': f'independent numpy recomputation of the D5 target rows from the recorded statistics, the static cache and the weight '
                           f'errors: max relative difference <= {TOL_REL}',
            '6_cache_use': 'every C / D4 / D5 run: no activation-error block, cache read and assembly blocks present; B the reverse; end-to-end: '
                           'cache read and assembly present and its D5 table equal to D5 repeat 1',
            '7_validity': 'fallback (count, reason), non-finite values, legal formats, exact limits',
            '8_repeats': 'identical tables / configurations across repeats (main = D5 repeat 1); new B / C / D4 runs equal the frozen ones; '
                         'end-to-end D5 reproduces the static table, caches, knee and D5 configuration',
            '9_identical': 'identical configurations are reported and share one evaluation; the threshold is never changed to create differences',
            'ste': 'STE unit test'},
        'timing': {'refresh': f'B / C / D4 / D5, 3 each, interleaved {REFRESH_SCHEDULE}; one process per run; CUDA synchronization; no page-cache '
                              'dropping (residency recorded)',
                   'e2e': 'one end-to-end D5 run: static statistics with the caches it needs -> auto knee -> D5 -> search (single measurement); '
                          'historical times and sums of stage medians reported separately',
                   'categories': CATEGORIES, 'separate': 'controls, repeats and the final evaluation'},
        'freeze': 'plan, code, inputs, tables, configurations and hashes frozen before any evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries, duplicate and near-duplicate '
                               'exclusion; every earlier manifest (incl. the two latest knee rounds) excluded by scanning runs/**/*segments*.json '
                               '(only this run skipped by path)',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests, 'eligible_before_length_and_neardup': len(elig),
                       'seeds': SEEDS, 'conditions': 'FP32, A, B, C, D4, D5 on the same new articles; original router; each its own dynamic route; no '
                                                     'cross-batch NLL subtraction',
                       'primary': {'D5 - A': 'corrected upper < 0 -> D5 improves the static BC',
                                   'D5 - D4': 'corrected upper < 0 -> D5 improves on D4',
                                   'D5 - B': f'corrected upper < +{NI_MARGIN} -> non-inferior; corrected lower > +{NI_MARGIN} -> beyond the margin; '
                                             'otherwise insufficient evidence'},
                       'intervals': f'mean, 95% CI and Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI ({NCMP} comparisons); decisions on the corrected interval',
                       'secondary': 'D5 - C, B - A, C - A, D4 - A (95% CI, labelled secondary)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'cautions': 'no threshold change, seed change or extra articles; not confirmed is not equal; non-inferior is not better than static'},
        'report_questions': ['1 D5 better than the static BC?', '2 D5 better than D4?', '3 D5 keeps B\'s quality, time saved?',
                             '4 versus C, the value of selecting about half the projections', '5 worth keeping as an optional step?'],
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not hist['pass']:
        raise SystemExit('history hashes differ: investigate first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 正常 knee 的 ≤5-bit selective sensitivity refresh（D5）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- 背景：≤5-bit 門檻是在看過 D4（≤4-bit，knee 無收益）與 C（全節點）結果後提出的新假說，本輪以新評估文章檢驗一次；成功也只支持本模型／knee／calibration，不測 ≤6-bit。',
          f'- D5 目標：凍結的 A 中 weight 或 activation ≤5 bit 的投影，{len(t5)} 個（int4 169＋int5 1343）；D4 是子集合；其外 {len(names) - len(t5)} 個。',
          '- 更新：目標投影所有候選格式的 weight／activation sensitivity（q0_knee 工作狀態），static 誤差統計（明確傳入快取、static rows 正規化）；非目標沿用 static；在 knee 精確上限重新搜尋一次。',
          '- A／B／C／D4 沿用凍結配置（hash 核對）；D5 計時只收集目標統計，不用 C 的統計遮罩。',
          f'- 計時：B／C／D4／D5 各 3 次交錯（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；D5 端到端 1 次。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、C、D4、D5；seed 22701／22702／22711。',
          f'- 主要：D5 − A、D5 − D4（校正上界 < 0 才確認改善）、D5 − B（校正上界 < +{NI_MARGIN} 才確認非劣；下界 > +{NI_MARGIN} 超過界線）；Bonferroni {100 - 2 * CI_CORR[0]:.2f}%。',
          '- 次要（95%）：D5 − C、B − A、C − A、D4 − A。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; D5 {len(t5)} (new 5-bit {len(set(t5) - set(t4))}); history ok {hist["pass"]}; previous WT103 {len(used)}; eligible {len(elig)}')


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
           'equal_previous_record': {'ce': rd['ce'] == prevc['q0_knee_forward_ce'], 'hidden': rd['hidden_sha'] == prevc['q0_knee_forward_hidden_sha'],
                                     'routes': dr.route_hash(rd['routes']) == prevc['q0_knee_forward_routes_hash']},
           'ste_unit_test': pr.ste_unit_test(dq, ranges), 'history': history(),
           'low5_recount': len(low5_nodes(q0k)), 'low5_file_count': read(out / 'low5_nodes.json')['count'],
           'page_cache_residency': ss.resident_fraction(ks.watch_files()), 'seconds': time.perf_counter() - t0}
    ctl['pass'] = bool(rd['finite_ce'] and all(ctl['equal_previous_record'].values()) and ctl['ste_unit_test']['pass'] and ctl['history']['pass']
                       and ctl['low5_recount'] == ctl['low5_file_count'])
    write(out / 'controls.json', ctl)
    log(json.dumps({'equal_previous_record': ctl['equal_previous_record'], 'pass': ctl['pass']}))


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
    if variant != 'B':                             # sensitivity-only: the static activation-error statistics, explicitly
        with tm.block('cache_read'):
            cache = np.load(ACT_CACHE, mmap_mode='r')
            cidx = read(ACT_INDEX)
    if variant in ('D4', 'D5'):
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
            targets = read(LOW4)['nodes'] if variant == 'D4' else read(out / 'low5_nodes.json')['nodes']
    stats = {} if (variant == 'D5' and rep == 1) else None
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
    """one process: static statistics (+ weight-error and activation caches) -> auto knee -> D5 (caches just written) -> search."""
    import torch
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    tag = 'e2e_D5'
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
    targets = low5_nodes(q0k)
    with tm.block('cache_read_d2'):
        d2z = np.load(rd / 'weight_error_d2.npz')
    with tm.block('cache_read'):
        cache = np.load(rd / 'act_err_cache.npy', mmap_mode='r')
        cidx = read(rd / 'act_err_cache_index.json')
    r = ks.build('D5', dq, ranges, names, tm, log, fm=q0k, d2z=d2z, cache=cache, cidx=cidx, static_tab=t, targets=targets)
    tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'],
          'updated_projections': len(r['updated_projections']), 'state': 'q0_knee working state, condition D5'}
    fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', tt)
        write(rd / 'config.json', fm_new)
    rec = run_record(tm, t0, {'variant': 'D5', 'page_cache_residency_before': residency, 'marks': marks, 'limits': limits, 'targets': len(targets),
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'auto_paths': dict(Counter(x.get('path') for x in recs)),
                              'search': srch, 'cache_use': cache_use('D5', tm.t), 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'],
                              'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward'], 'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']),
                              'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'marks': marks, 'cache_use': rec['cache_use']}))


def timing(out, log, phase):
    sched = REFRESH_SCHEDULE if phase == 'refresh' else [('D5', None)]
    for v, r in sched:
        name = f'refresh_{v}_r{r}' if phase == 'refresh' else 'e2e_D5'
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
    from run_olmoe_multipoint import resources
    names, nodes, elems, flops, rows = pr.problem_static()
    pl = read(out / 'plan.json')
    W = out / 'work/runs'
    rr = [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE]
    exp = rr + ['e2e_D5']
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    ctl, prevc = read(out / 'controls.json'), read(KS / 'controls.json')
    st, kn = read(STATIC_TABLE), read(KNEE)
    ft = {v: read(p) for v, p in FROZEN_TABLES.items()}
    fc = {v: read(p) for v, p in FROZEN.items()}
    t5, c5 = read(W / 'refresh_D5_r1/table.json'), read(W / 'refresh_D5_r1/config.json')
    low5 = read(out / 'low5_nodes.json')['nodes']
    s5 = set(low5)
    hist = history()
    res['1_history'] = {'hashes': hist, 'pass': hist['pass']}
    drr = dr.route_hash(dict(np.load(KS / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    g_prev = read(KS / 'work/runs/refresh_B_r1/run.json')['input_grad_sha_per_layer']
    c2 = {k: {'ce': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'] == prevc['q0_knee_forward_ce'],
              'hidden': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
              'routes': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr,
              'input_grad': runs[k]['input_grad_sha_per_layer'] == g_prev, 'recompute_zero': runs[k]['recompute_max_abs'] == 0} for k in rr}
    c2['e2e_D5'] = {'ce': runs['e2e_D5']['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden': runs['e2e_D5']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
                    'input_grad': runs['e2e_D5']['input_grad_sha_per_layer'] == g_prev}
    res['2_state'] = {'runs': c2, 'pass': all(all(v.values()) for v in c2.values())}
    fb5 = set(runs['refresh_D5_r1']['fallback_static_rows'])
    res['3_targets_equal_C'] = {'rows': len(s5), 'equal': all(t5['B'][n] == ft['C']['B'][n] and t5['C'][n] == ft['C']['C'][n] for n in low5 if n not in fb5)}
    res['3_targets_equal_C']['pass'] = res['3_targets_equal_C']['equal']
    res['4_nontargets_static'] = {'rows': len(names) - len(s5), 'equal': all(t5['B'][n] == st['B'][n] and t5['C'][n] == st['C'][n] for n in names if n not in s5 or n in fb5)}
    res['4_nontargets_static']['pass'] = res['4_nontargets_static']['equal']
    z = np.load(W / 'refresh_D5_r1/target_stats.npz')
    sm = read(W / 'refresh_D5_r1/target_stats_meta.json')
    d2z, cache, cidx = np.load(D2), np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    relw = rela = relu = relq = 0.0
    for n in low5:
        if n in fb5:
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
            relw, rela = max(relw, rr_(wv, t5['B'][n][f])), max(rela, rr_(av, t5['C'][n][f]))
            relu, relq = max(relu, rr_(au, t5['C'][n][f])), max(relq, rr_(aq, t5['C'][n][f]))
    res['5_recompute'] = {'rows': len(s5) - len(fb5), 'weight_max_rel': relw, 'act_max_rel': rela, 'descriptive_user_form_max_rel': relu,
                          'descriptive_q0_rows_normalization_max_rel': relq, 'pass': relw <= TOL_REL and rela <= TOL_REL}
    eu = runs['e2e_D5']['cache_use']
    res['6_cache_use'] = {'runs': {k: runs[k]['cache_use'] for k in exp}, 'e2e_rule': 'end-to-end: its act_error block is the static-statistics stage that '
                          'builds the cache; the D5 stage must show cache read and assembly blocks and its D5 table must equal D5 repeat 1'}
    lim = kn['limits']
    rc = resources(c5, names, elems, flops)
    res['7_validity'] = {'fallback_count': len(fb5), 'fallback_rows': sorted(fb5), 'fallback_reason': 'searched expert with no q0_knee calibration token (static row kept)',
                         'finite': all(runs[k]['grads_finite'] and runs[k]['nonfinite'] == 0 for k in exp),
                         'covers_exactly_searched': sorted(c5) == sorted(names) and sorted(t5['B']) == sorted(names) and sorted(t5['C']) == sorted(names),
                         'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in c5.values()), 'resources': rc,
                         'within_limits': rc['M_bytes'] <= lim['max_bytes'] and rc['BOPs'] <= lim['max_bops'],
                         'search_within_budget_all_runs': all(runs[k]['search']['within_budget'] for k in exp), 'e2e_limits_equal': runs['e2e_D5']['limits'] == lim}
    v7 = res['7_validity']
    v7['pass'] = bool(v7['finite'] and v7['covers_exactly_searched'] and v7['legal'] and v7['within_limits'] and v7['search_within_budget_all_runs'] and v7['e2e_limits_equal'])
    reps = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        tref = t5 if v == 'D5' else ft[v]
        cref = c5 if v == 'D5' else fc[v]
        reps[v] = {'tables': all({kk: vv for kk, vv in read(W / k / 'table.json').items() if kk in ('B', 'C')} == {kk: tref[kk] for kk in ('B', 'C')} for k in ks_),
                   'configs': all(read(W / k / 'config.json') == cref for k in ks_)}
    reps['e2e_D5'] = {'static_table': {kk: vv == st[kk] for kk, vv in read(W / 'e2e_D5/table_static.json').items() if kk in ('B', 'C')},
                      'weight_error': ss.d2_equal(W / 'e2e_D5/weight_error_d2.npz', D2), 'act_cache': sha(W / 'e2e_D5/act_err_cache.npy') == sha(ACT_CACHE),
                      'knee': read(W / 'e2e_D5/q0_knee.json') == read(Q0K), 'targets': runs['e2e_D5']['targets'] == len(low5),
                      'D5_table': {kk: read(W / 'e2e_D5/table.json')[kk] == t5[kk] for kk in ('B', 'C')}, 'D5_config': read(W / 'e2e_D5/config.json') == c5}
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['8_repeats'] = reps
    res['8_repeats']['pass'] = flat(reps)
    res['6_cache_use']['pass'] = bool(all(runs[k]['cache_use']['as_expected'] for k in rr) and eu['act_assembly_s'] > 0 and eu['act_cache_read_s'] > 0
                                      and all(reps['e2e_D5']['D5_table'].values()))
    res['9_identical'] = {f'D5={c}': c5 == fc[c] for c in ('A', 'B', 'C', 'D4')}
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['inputs_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['pass'] = bool(not res['missing_runs'] and all(res[k]['pass'] for k in ('1_history', '2_state', '3_targets_equal_C', '4_nontargets_static', '5_recompute',
                                                                                 '6_cache_use', '7_validity', '8_repeats'))
                       and res['inputs_unchanged'] and res['ste'] and res['controls_stage_pass'])
    res['seconds'] = time.perf_counter() - t0
    write(out / 'check.json', res)
    log(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in res.items() if k not in ('seconds',)}))


def freeze(out, log):
    from mp_auto import config_hash
    from run_olmoe_multipoint import resources
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'check.json')['pass']:
        raise SystemExit('controls failed: not freezing')
    (out / 'configs').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    W = out / 'work/runs'
    for c, p in FROZEN.items():
        write(out / f'configs/{c}.json', read(p))
    write(out / 'configs/D5.json', read(W / 'refresh_D5_r1/config.json'))
    write(out / 'tables/D5.json', read(W / 'refresh_D5_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    s4, s5 = set(read(LOW4)['nodes']), set(read(out / 'low5_nodes.json')['nodes'])
    zone = lambda n: 'le4' if n in s4 else 'new5' if n in s5 else 'rest'
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'D5 = refresh repeat 1; A / B / C / D4 = frozen records',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c]), 'resources': resources(cfg[c], names, elems, flops)} for c in cfg},
           'frozen_equal_records': {c: cfg[c] == read(p) for c, p in FROZEN.items()},
           'identical': {f'D5={c}': cfg['D5'] == cfg[c] for c in ('A', 'B', 'C', 'D4')},
           'pairwise_differences': {f'D5-{c}': diff(cfg['D5'], cfg[c]) for c in ('A', 'B', 'C', 'D4')} | {f'{c}-A': diff(cfg[c], cfg['A']) for c in ('B', 'C', 'D4')},
           'changed_vs_A_by_zone': {c: dict(Counter(zone(n) for n in names if cfg[c][n] != cfg['A'][n])) for c in ('B', 'C', 'D4', 'D5')},
           'zone_sizes': {'le4': len(s4), 'new5': len(s5 - s4), 'rest': len(names) - len(s5)},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in ('D5',)},
           'tables': {'D5.json': sha(out / 'tables/D5.json'), **{f'frozen/{v}': sha(p) for v, p in FROZEN_TABLES.items()}},
           'knee_sha256': sha(KNEE), 'low5_nodes_sha256': sha(out / 'low5_nodes.json'), 'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'),
           'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_by_zone')}))


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
    old_t, old = np.load(KA / 'tokens.npy')[:CTRL], read(KA / 'eval.json')['ce']
    pc = {}
    for c, fm in (('fp32', None), ('A', read(FROZEN['A'])), ('C', read(FROZEN['C'])), ('D', read(FROZEN['D4']))):
        r = dr.forward(old_t, fm, dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs, shared, by_hash = {}, {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in ('A', 'B', 'C', 'D4', 'D5'):
        h = fz['configs'][c]['config_hash']
        if h in by_hash:
            runs[c] = runs[by_hash[h]]; shared[c] = by_hash[h]
            continue
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        by_hash[h] = c
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/D5.json'), dq, ranges, record=True, log=log, label='rerun D5')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_D5_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['D5']['ce'][:CTRL])))),
                                    'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['D5']['hidden_sha'][:CTRL]))),
                                    'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['D5']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in ('A', 'B', 'C', 'D4', 'D5')},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean())
                                       for c in ('A', 'B', 'C', 'D4', 'D5')}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_D5_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_D5_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_D5_first_chunk']['routes_equal'] and ctl['finite']
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
    proc = {(f'refresh_{j["variant"]}_r{j["rep"]}' if j['stage'] == 'refresh' else 'e2e_D5'): j['seconds'] for j in jl if j['stage'] in ('refresh', 'e2e')}
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        s[f'refresh_{v}'] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                             'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                             'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks_]) for t in sorted({t for k in ks_ for t in runs[k]['timers']})},
                             'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                             'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks_}}
    e = runs['e2e_D5']
    s['e2e_D5_measured_single'] = {'method_s': e['method_s'], 'process_wall_s': proc['e2e_D5'], 'marks': e['marks'], 'categories': e['categories'],
                                   'peak_gpu_gb': e['peak_gpu_gb'], 'peak_cpu_gb': e['peak_cpu_gb'], 'page_cache_residency_before': e['page_cache_residency_before']}
    kt, at = read(KS / 'analysis.json')['timing'], read(KA / 'analysis.json')['timing']
    s['historical'] = {'e2e_measured_single': {'A': kt['e2e_measured']['A']['method_s'], 'B': kt['e2e_measured']['B']['method_s'],
                                               'D4': kt['e2e_measured']['D']['method_s'], 'C': at['e2e_C_measured_single']['method_s']},
                       'auto_median_s': kt['auto']['method_s']['median'],
                       'note': 'previous knee rounds, same machine and code; historical single measurements'}
    s['D5_total_estimate_sum_of_stage_medians_s'] = (kt['total_estimate_sum_of_stage_medians_s']['static_statistics_component']['D'] + kt['auto']['method_s']['median']
                                                    + s['refresh_D5']['method_s']['median'])
    s['extra_with_caches_median_s'] = {v: s[f'refresh_{v}']['method_s']['median'] for v in VARIANTS}
    s['note'] = 'method = run time minus control blocks; the D5 estimate = static statistics with caches (selective-sensitivity medians) + previous-round auto median + D5 median'
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
    pairs = {'D5_minus_A': ('D5', 'A'), 'D5_minus_D4': ('D5', 'D4'), 'D5_minus_B': ('D5', 'B'),
             'D5_minus_C': ('D5', 'C'), 'B_minus_A': ('B', 'A'), 'C_minus_A': ('C', 'A'), 'D4_minus_A': ('D4', 'A')}
    D = {k: A[a] - A[b] for k, (a, b) in pairs.items()}
    res = {'n_articles': N, 'bonferroni_level': 100 - 2 * CI_CORR[0], 'primary': {k: st(D[k]) for k in ('D5_minus_A', 'D5_minus_D4', 'D5_minus_B')},
           'secondary': {k: st(D[k]) for k in ('D5_minus_C', 'B_minus_A', 'C_minus_A', 'D4_minus_A')}, 'damage': {c: st(A[c] - A['fp32']) for c in ('A', 'B', 'C', 'D4', 'D5')}}
    for k in ('D5_minus_A', 'D5_minus_D4'):
        lo, hi = res['primary'][k]['ci_bonferroni']
        res['primary'][k]['decision'] = 'D5_better_confirmed' if hi < 0 else 'D5_worse_confirmed' if lo > 0 else 'not_confirmed'
    lo, hi = res['primary']['D5_minus_B']['ci_bonferroni']
    res['primary']['D5_minus_B']['decision'] = (f'noninferior_within_{NI_MARGIN}' if hi < NI_MARGIN else f'beyond_margin_{NI_MARGIN}' if lo > NI_MARGIN
                                                else 'insufficient_evidence')
    for k, v in res['secondary'].items():
        lo, hi = v['ci95']
        v['direction_95'] = 'first_better' if hi < 0 else 'first_worse' if lo > 0 else 'not_confirmed'
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5')}
    names = sorted(cfg['A'])
    tabs = {'static': read(STATIC_TABLE), **{v: read(p) for v, p in FROZEN_TABLES.items()}, 'D5': read(out / 'tables/D5.json')}
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    res['objectives_descriptive'] = {t: {c: obj(tabs[t], cfg[c]) for c in cfg} for t in tabs}
    res['search_D5'] = {k: read(out / 'work/runs/refresh_D5_r1/run.json')['search'][k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources',
                                                                                                   'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
    res['configs'] = {c: v['resources'] for c, v in fz['configs'].items()}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_by_zone', 'zone_sizes', 'transitions_vs_A')}
    res['targets'] = {k: v for k, v in read(out / 'low5_nodes.json').items() if k != 'nodes'}
    rD5 = read(out / 'work/runs/refresh_D5_r1/run.json')
    t5e = {'.'.join(n.split('.')[:2]) for n in read(out / 'low5_nodes.json')['nodes']}
    res['targets']['q0_knee_calibration_rows_of_target_experts'] = int(sum(rD5['q0_routed_rows_by_expert'].get(e, 0) for e in t5e))
    res['targets']['fallback_static_rows'] = rD5['fallback_static_rows']
    from run_olmoe_multipoint import problem
    _, _, pnames, _, elems, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    for c in ['fp32', 'A', 'B', 'C', 'D4', 'D5']:
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
    for c in ('A', 'B', 'C', 'D4', 'D5'):
        sb = float(sum(rowsof(counts[c], n) * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
        ub = float(sum(counts[c][l, e] * 3 * elem_u * 16 * 32 for l, e in unsearched))
        cost[c] = {'searched_BOPs_actual': sb, 'relative_searched_to_int8_fp32_counts': sb / b8_s, 'unsearched_BOPs_bf16_fp32': ub,
                   'total_BOPs_actual': sb + ub, 'relative_total_to_all_int8_fp32_counts': (sb + ub) / b8_all,
                   'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
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
    log(json.dumps({'primary': {k: (v['mean'], v['ci_bonferroni'], v['decision']) for k, v in res['primary'].items()},
                    'secondary': {k: (v['mean'], v['ci95'], v['direction_95']) for k, v in res['secondary'].items()}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'controls', 'refresh', 'e2e', 'timing', 'check', 'freeze', 'data', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant', choices=list(VARIANTS))
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
