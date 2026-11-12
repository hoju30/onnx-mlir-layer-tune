#!/usr/bin/env python3
"""OLMoE normal knee: <= 5-bit activation error-only refresh (E5), next to the frozen A / B / E / D5 (and C / D4 for
diagnostics). Pre-registered in plan.json / plan.md. The <= 5-bit scope is a new hypothesis proposed (at the user's
request) after the all-node error-only E result was seen; it is tested on new evaluation articles.

E5: targets = q0_knee (= A) projections whose weight or activation format has <= 5 bits (the D5 selection); for a target,
every candidate format's activation cost is rebuilt from the static G2 / GS cache (runs/olmoe_knee_error_only_20261022) and
the new q0_knee activation errors at the E taps (q0 routed-row normalization); every weight row and every non-target
activation row is A's, bitwise. The update is forward-only (inference_mode, no retain_grad, autograd calls counted) and
computes candidate errors only for the slots the target projections need (x_in shared by gate / up computed once).

Stages (one --out-dir): plan, controls, refresh (--variant B|E|E5 --rep k), e2e, timing (--phase refresh|e2e), check,
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
import run_olmoe_knee_error_only as ke
import run_olmoe_knee_selective_sensitivity as ks
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss

HERE = Path(__file__).resolve().parent
KS, KA, K5 = ke.KS, ke.KA, ke.K5
KE = HERE / 'runs/olmoe_knee_error_only_20261022'
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE, D2 = ks.STATIC_TABLE, ks.D2
Q0K, KNEE = KS / 'knee/q0_knee.json', KS / 'knee/knee.json'
GC, GC_INDEX = KE / 'work/runs/static_gc/grad_stats_cache.npy', KE / 'work/runs/static_gc/grad_stats_cache_index.json'
LOW5 = K5 / 'low5_nodes.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, NCMP, LOW_MAX = 10000, 0.0005, 1e-12, 3, 5
CI_CORR = [100 * 0.05 / (2 * NCMP), 100 * (1 - 0.05 / (2 * NCMP))]      # Bonferroni over the 3 primary comparisons: 98.33%
SEEDS = {'order': 22901, 'start': 22902, 'bootstrap': 22911}
VARIANTS = ('B', 'E', 'E5')
REFRESH_SCHEDULE = [('B', 1), ('E', 1), ('E5', 1), ('E', 2), ('E5', 2), ('B', 2), ('E5', 3), ('B', 3), ('E', 3)]
FROZEN = {'A': KS / 'configs/A.json', 'B': KS / 'configs/B.json', 'C': KA / 'configs/C.json', 'D4': KS / 'configs/D.json',
          'D5': K5 / 'configs/D5.json', 'E': KE / 'configs/E.json'}
FROZEN_TABLES = {'B': KS / 'tables/B.json', 'C': KA / 'tables/C.json', 'D4': KS / 'tables/D.json', 'D5': K5 / 'tables/D5.json', 'E': KE / 'tables/E.json'}
NEED = {'gate_proj': ('x_in', 'gate_y'), 'up_proj': ('x_in', 'up_y'), 'down_proj': ('down_x', 'down_y')}
CATEGORIES = ks.CATEGORIES


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_low5_error_only.py', 'run_olmoe_knee_error_only.py', 'run_olmoe_knee_selective_sensitivity.py',
                                                  'run_olmoe_knee_all_sensitivity.py', 'run_olmoe_knee_low5_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def watch_files():
    return ks.watch_files() + [GC]


def low5_nodes(q0):
    from htq_pareto_search import FORMAT_BITS
    return sorted(n for n, v in q0.items() if FORMAT_BITS[v['w']] <= LOW_MAX or FORMAT_BITS[v['a']] <= LOW_MAX)


def history():
    fe = read(KE / 'freeze.json')
    res = {'previous_rounds': ke.history(),
           'error_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in fe['configs'].items()},
           'error_round_table_E': sha(KE / 'tables/E.json') == fe['tables']['E.json'], 'error_round_check': read(KE / 'check.json')['pass'],
           'grad_cache': {'npy': sha(GC) == fe['grad_cache']['npy'], 'index': sha(GC_INDEX) == fe['grad_cache']['index']},
           'code_unchanged': {k: sha(HERE / k) == fe['code'][k] for k in ('run_olmoe_knee_error_only.py', 'run_olmoe_knee_selective_sensitivity.py',
                                                                           'run_olmoe_selective_sensitivity.py', 'run_olmoe_knee_low5_sensitivity.py')}}
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


def e5_update(fm, dq, ranges, names, tm, log, gc, gidx, static_tab, targets, save_err=None):
    """forward-only, selective: q0_knee dynamic forward (every expert computed as in q0); candidate errors only for the slots of
    the target projections; target activation rows from the static G2 / GS and the new errors (q0-row normalization); all weight
    rows and every non-target activation row from the static table."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    COLOFF, COL_SRC, SLOT_COL = ss.COLOFF, ss.COL_SRC, ss.SLOT_COL
    tset = set(targets)
    want_all = defaultdict(dict)
    for n in names:
        if n in tset:
            l, e, p = n.split('.')
            want_all[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    C_, meta, cols_done = {}, {}, Counter()
    with ke.AutogradGuard() as guard, torch.inference_mode():
        grad_enabled = torch.is_grad_enabled()
        with tm.block('setup'):
            cfg, ck = oc.config(), oc.Checkpoint()
            toks_np = np.load(CALIB)
            toks = torch.from_numpy(toks_np).to('cuda')
            nW = len(toks_np)
            cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
        with tm.block('ckpt_load'):
            g = ck.global_('cuda', torch.float32)
        ids_rec = np.zeros((nW, NL, SEQ, K), np.int16); w_rec = np.zeros((nW, NL, SEQ, K), np.float32)
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        for l in range(NL):
            wq, hk = ss.load_layer(ck, l, fm, dq, ranges, tm)
            want = want_all.get(l, {})
            obs = {}
            with tm.block('forward_routing'):
                h, idx, rw = ke.layer_observe(wq, h, cfg, cos, sin, pr.act_map(hk), want, obs)
            with tm.block('controls'):
                ids_rec[:, l] = idx.view(nW, SEQ, K).to(torch.int16).cpu().numpy(); w_rec[:, l] = rw.view(nW, SEQ, K).cpu().numpy()
            for e in sorted(want):
                node = f'L{l}.E{e}'
                if e not in obs:
                    meta[node] = {'routed_rows': 0}
                    continue
                c = obs[e]
                n = len(c['tok'])
                meta[node] = {'routed_rows': int(n)}
                needed = sorted({col for p in want[e] for col in NEED[p]}, key=list(COL_SRC).index)
                with tm.block('cache_read'):
                    grow = torch.from_numpy(np.ascontiguousarray(gc[gidx['row'][node]])).to('cuda')
                with tm.block('act_error'):
                    stats = {}
                    for col in needed:
                        p0, s0, k0 = COL_SRC[col]
                        t, rng = c[k0], ranges[f'{node}.{p0}'][s0]
                        s1s, s2s = [], []
                        for f in QUANT:
                            err = (dq.act(t, f, rng) - t).double()
                            s1s.append(err.sum(0)); s2s.append((err ** 2).sum(0))
                        stats[col] = (torch.stack(s1s), torch.stack(s2s))
                        cols_done[col] += 1
                with tm.block('act_assembly'):
                    vm = {}
                    for p in want[e]:
                        tot = None
                        for s in ('x', 'y'):
                            col = SLOT_COL[(p, s)]
                            if col not in vm:
                                a_, b_ = COLOFF[col]
                                vm[col] = ss.act_cost_from_stats(stats[col][0], stats[col][1], n, grow[0, a_:b_], grow[1, a_:b_])
                            tot = vm[col] if tot is None else tot + vm[col]
                        v = tot.cpu().numpy()
                        C_[f'{node}.{p}'] = {'fp32': 0.0, **{f: float(v[i]) for i, f in enumerate(QUANT)}}
                if save_err is not None:
                    with tm.block('controls'):
                        for col, (S1_, S2_) in stats.items():
                            save_err['data'][f'{node}|{col}'] = np.stack([S1_.cpu().numpy(), S2_.cpu().numpy()])
                        save_err['n'][node] = n
            del wq, hk, obs
            log(f'E5: layer {l} ({sum(tm.t.values()):.0f}s)')
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
        with tm.block('controls'):
            hc = h.cpu().numpy()
            hidden_sha = [hashlib.sha256(np.ascontiguousarray(hc[k]).tobytes()).hexdigest() for k in range(nW)]
    fallback = [n for n in names if n in tset and meta['.'.join(n.split('.')[:2])]['routed_rows'] == 0]
    with tm.block('table_assembly'):
        B_ = {n: static_tab['B'][n] for n in names}
        for n in names:
            if n not in tset or n in fallback:
                C_[n] = static_tab['C'][n]
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd, 'hidden_sha': hidden_sha,
            'fallback_static_rows': fallback, 'nonfinite': nonfinite, 'updated_projections': len(tset) - len(fallback), 'columns_computed': dict(cols_done),
            'gradient_free': {'autograd_calls': dict(guard.calls), 'grad_enabled_inside': grad_enabled, 'inference_mode': True}}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    hist = history()
    names, nodes, elems, flops, rows = pr.problem_static()
    q0 = read(FROZEN['A'])
    t5 = low5_nodes(q0)
    k5 = read(LOW5)['nodes']
    cidx = read(GC_INDEX)
    texp = {'.'.join(n.split('.')[:2]) for n in t5}
    need_cols = Counter(col for e in texp for col in sorted({c for n in t5 if n.startswith(e + '.') for c in NEED[n.split('.')[2]]}))
    write(out / 'low5_nodes.json', {'definition': f'q0_knee (= frozen A) weight or activation format with <= {LOW_MAX} bits (htq_pareto_search.FORMAT_BITS); '
                                                  'the D5 selection semantics', 'background': 'scope proposed at the user\'s request after the all-node error-only E '
                                                  'result was seen; new evaluation articles', 'q0_sha256': sha(FROZEN['A']), 'count': len(t5), 'expected': 1512,
                                    'equal_to_d5_targets': t5 == k5, 'total': len(names), 'outside': len(names) - len(t5),
                                    'weight_formats': dict(Counter(q0[n]['w'] for n in t5)), 'by_proj': dict(Counter(n.split('.')[2] for n in t5)),
                                    'experts': len(texp), 'slot_columns_needed': dict(need_cols), 'slot_columns_all_experts': 5 * len(cidx['experts']),
                                    'fp32_calibration_rows_of_target_experts': int(sum(cidx['n_static'][e] for e in texp)), 'nodes': t5})
    if len(t5) != 1512 or t5 != k5:
        raise SystemExit('E5 targets differ from the expected / D5 set: check inputs and rule first')
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = ss.input_files() + list(FROZEN.values()) + list(FROZEN_TABLES.values()) + [
        KS / f for f in ('freeze.json', 'check.json', 'controls.json', 'knee/knee.json', 'knee/q0_knee.json', 'work/q0_knee_calibration_routes_dr_forward.npz')] + [
        KA / 'freeze.json', K5 / 'freeze.json', LOW5] + [KE / f for f in ('freeze.json', 'check.json', 'eval.json', 'tokens.npy', 'work/runs/refresh_E_r1/new_error_stats.npy',
                                                                       'work/runs/refresh_E_r1/new_error_stats_index.json')] + [GC, GC_INDEX]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'At the normal knee, does the <= 5-bit activation error-only refresh (E5) improve the static BC, beat the all-node error-only E, '
                    'and keep the full refresh\'s (B) quality at a lower cost?',
        'background': 'new hypothesis requested by the user after the all-node error-only E (no confirmed gain, ~25 s update) was seen; tested once on '
                      'new articles; a success supports only this model / knee / calibration; no <= 6-bit or other selection rule in this round',
        'not_this_round': 'no new knee, no BC change, no training or parameter update, no second refresh, no loss-based fallback, no new search algorithm',
        'fixed': {'history': hist, 'frozen_configs': {c: str(p.relative_to(HERE)) for c, p in FROZEN.items()}, 'limits': read(KNEE)['limits'],
                  'grad_cache': f'{GC.relative_to(HERE)} (verified static G2 / GS of the error-only round)',
                  'same': 'checkpoint and router, calibration, ranges / scale, quantizer, activation taps, candidates, resource model, LP-first search, exact knee limits'},
        'E5': {'targets': 'low5_nodes.json: FORMAT_BITS[q0.w] <= 5 or FORMAT_BITS[q0.a] <= 5 on A (projection level; equal to the D5 targets), frozen',
               'targets_count': len(t5), 'outside': len(names) - len(t5),
               'weight': 'every weight row = A (static), bitwise', 'sensitivity': 'the static G2 / GS cache only; no new sensitivity',
               'activation_targets': 'every candidate format: sum_slots,c [G2_static var_new + GS_static mu_new^2], err_new = Q_f(a_q0) - a_q0 at the E taps, '
                                     'mu / var over the q0 routed rows, G2 / GS at the static scale, BC op order',
               'activation_non_targets': 'A\'s rows bitwise (a shared x_in computed for a gate / up target never rewrites the other projection\'s row)',
               'implementation': 'forward-only (inference_mode, observation path without retain_grad, autograd calls counted); the whole q0 model runs; '
                                 'candidate errors only for the slot columns of the target projections (gate: x_in, gate_y; up: x_in, up_y; down: '
                                 'down_x, down_y); never E computed and masked',
               'search': 'one LP-first search at the exact knee limits; every projection may change format; LP bound, objective, gaps; A / B / C / D4 / D5 / E / '
                         'E5 cross scores on the E5 table with feasibility; a cheaper known configuration is disclosed; the search approximation is never '
                         'claimed excluded'},
        'controls': {'1_inputs': 'inputs, configurations, gradient cache and targets hashes equal their records',
                     '2_weight_rows': 'every E5 weight row bitwise equals A', '3_target_rows': 'E5 target activation rows bitwise equal E',
                     '4_nontarget_rows': 'E5 non-target activation rows bitwise equal A',
                     '5_recompute': f'independent numpy recomputation of the target rows from the saved new errors and the static G2 / GS (relative <= {TOL_REL})',
                     '6_forward': 'forward CE, hidden sha256 and routes equal the q0_knee forward record',
                     '7_gradient_free': 'update stage: 0 autograd calls, grad disabled, no backward / sensitivity blocks',
                     '8_validity': 'fallback, finite, legal, exact limits',
                     '9_repeats': 'E5 repeats identical tables / configurations (main = repeat 1); new B / E runs equal the frozen ones; end-to-end E5 '
                                  'reproduces the static table, gradient cache, knee and E5 configuration; the expected table (E target rows + A rows) equals E5'},
        'timing': {'refresh': f'B / E / E5, 3 each, interleaved {REFRESH_SCHEDULE}; one process per run; CUDA synchronization; no page-cache dropping',
                   'e2e': 'one end-to-end E5 run: static BC + gradient cache -> auto knee -> E5 -> search (single measurement)',
                   'categories': CATEGORIES, 'separate': 'controls, repeats, evaluation; historical times and stage-sum estimates labelled'},
        'freeze': 'plan, code, inputs, tables, configurations and hashes frozen before any evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries, duplicate and near-duplicate exclusion; '
                               'every earlier manifest (incl. the error-only round) excluded by scanning runs/**/*segments*.json',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests, 'eligible_before_length_and_neardup': len(elig),
                       'seeds': SEEDS, 'conditions': 'FP32, A, B, E, D5, E5 on the same new articles; original router; each its own dynamic route; no '
                                                     'cross-batch NLL subtraction',
                       'primary': {'E5 - A': 'corrected upper < 0 -> E5 improves the static BC', 'E5 - E': 'corrected upper < 0 -> E5 better than E',
                                   'E5 - B': f'corrected upper < +{NI_MARGIN} -> non-inferior; corrected lower > +{NI_MARGIN} -> beyond the margin; otherwise insufficient'},
                       'intervals': f'mean, 95% CI, Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI ({NCMP} comparisons); decisions on the corrected interval',
                       'secondary': 'E5 - D5 (same targets: error vs sensitivity update), B - A, E - A, D5 - A (95% CI)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'cautions': 'no extra articles, seed or threshold changes; better than E is not better than A; non-inferior is not an improvement'},
        'report_questions': ['1 E5 improves the static BC?', '2 E5 better than E?', '3 E5 keeps B\'s quality?', '4 E5 vs D5 on the same targets',
                             '5 extra and total offline time saved', '6 worth keeping as an optional step?'],
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not hist['pass']:
        raise SystemExit('history hashes differ: investigate first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 正常 knee 的 ≤5-bit activation error-only refresh（E5）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- 背景：看過全節點 error-only E（未確認改善、更新約 25 秒）後，依使用者要求提出的新假說；以新文章檢驗一次；不測 ≤6-bit 或其他規則。',
          f'- E5 目標：A 中 weight 或 activation ≤5 bit 的投影 {len(t5)} 個（= D5 的目標集合）；其外 {len(names) - len(t5)} 個。',
          '- E5 成本：weight 全部 = A；目標投影的 activation（所有候選格式）= Σ(G2_static·var_new + GS_static·μ_new²)，新誤差在 E 的 tap、q0 rows 正規化；非目標 activation 列 = A（逐位元）。',
          '- 實作：forward-only（inference_mode、無 retain_grad、計數 autograd）；完整 q0 forward；只計算目標投影所需的 slot 誤差（不先算 E 再遮罩）。',
          '- 控制：輸入與快取 hash、weight 列 = A、目標列 = E、非目標列 = A、numpy 獨立重算、forward 一致、無梯度計算、合法與上限、重複一致、端到端重現。',
          f'- 計時：B／E／E5 各 3 次交錯（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；E5 端到端 1 次。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、E、D5、E5；seed 22901／22902／22911。',
          f'- 主要：E5 − A、E5 − E（校正上界 < 0 才確認改善）、E5 − B（校正上界 < +{NI_MARGIN} 非劣；下界 > +{NI_MARGIN} 超過界線）；Bonferroni {100 - 2 * CI_CORR[0]:.2f}%。',
          '- 次要（95%）：E5 − D5、B − A、E − A、D5 − A。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; E5 targets {len(t5)} (= D5 {t5 == k5}); columns {dict(need_cols)}; history ok {hist["pass"]}; previous WT103 {len(used)}; eligible {len(elig)}')


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
           'targets_recount_equal': low5_nodes(q0k) == read(out / 'low5_nodes.json')['nodes'],
           'page_cache_residency': ss.resident_fraction(watch_files()), 'seconds': time.perf_counter() - t0}
    ctl['pass'] = bool(rd['finite_ce'] and all(ctl['equal_previous_record'].values()) and ctl['ste_unit_test']['pass'] and ctl['history']['pass']
                       and ctl['targets_recount_equal'])
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
    residency = ss.resident_fraction(watch_files())
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
    extra = {}
    if variant == 'B':
        with tm.block('cache_read_d2'):
            d2z = np.load(D2)
        r = ks.build('B', dq, ranges, names, tm, log, fm=fm, d2z=d2z)
        extra = {'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'recompute_max_abs': max(r['recompute_max_abs_per_layer']), 'grads_finite': r['grads_finite']}
    else:
        with tm.block('cache_read'):
            gc = np.load(GC, mmap_mode='r')
            gidx = read(GC_INDEX)
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
            targets = read(out / 'low5_nodes.json')['nodes'] if variant == 'E5' else None
        save = {'n': {}, 'data': {}} if (variant == 'E5' and rep == 1) else None
        if variant == 'E':
            r = ke.e_update(fm, dq, ranges, names, tm, log, gc, gidx, static_tab)
        else:
            r = e5_update(fm, dq, ranges, names, tm, log, gc, gidx, static_tab, targets, save_err=save)
            extra['columns_computed'] = r['columns_computed']
            extra['updated_projections'] = r['updated_projections']
        extra['gradient_free'] = r['gradient_free']
        if save is not None:
            with tm.block('controls'):
                np.savez(rd / 'new_error_stats_targets.npz', **save['data'])
                write(rd / 'new_error_stats_targets_index.json', {'n_q0': save['n'], 'keys': 'node|column -> [sum_err, sum_err_sq] x formats x channels',
                                                                  'formats': list(__import__('run_native_bc_pipeline').QUANT)})
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': f'q0_knee working state, condition {variant}'}
    fm_new, srch = ss.search_table(t, (names, nodes, elems, flops), limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm_new)
    with tm.block('controls'):
        np.savez(rd / 'q0_knee_calibration_routes.npz', **r['routes'])
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'search': srch,
                              'cache_use': ke.cache_use('B' if variant == 'B' else 'E', tm.t), 'fallback_static_rows': r['fallback_static_rows'],
                              'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'], 'routes_hash': dr.route_hash(r['routes']), 'nonfinite': len(r['nonfinite']),
                              'q0_routed_rows_by_expert': {k: v['routed_rows'] for k, v in r['meta'].items()}, 'code': code_hashes(), **extra})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path'], 'cache_use': rec['cache_use'],
                    **({k: extra[k] for k in ('gradient_free', 'columns_computed') if k in extra})}))


def e2e(out, log):
    """one process: static BC + gradient cache -> auto knee -> E5 forward-only selective refresh -> search."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = 'e2e_E5'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(watch_files())
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
        gcol = ke.new_gcollect(names)
    st = ke.build_static(dq, ranges, names, tm, log, gcollect=gcol)
    t = {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'}
    marks = {'static_done_s': time.perf_counter() - t0}
    with tm.block('table_write'):
        write(rd / 'table_static.json', t)
    gp, gi = rd / 'grad_stats_cache.npy', rd / 'grad_stats_cache_index.json'
    with tm.block('cache_write'):
        ke.write_gc(gp, gi, gcol, {'state': 'end-to-end run'})
    marks['static_written_s'] = time.perf_counter() - t0
    del st, gcol
    sel, recs, slog, kept = ks.auto_search(t, prob, tm, out / 'logs', tag)
    q0k, limits = ks.knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'q0_knee.json', q0k)
    marks['knee_done_s'] = time.perf_counter() - t0
    targets = low5_nodes(q0k)
    with tm.block('cache_read'):
        gc = np.load(gp, mmap_mode='r')
        gidx = read(gi)
    r = e5_update(q0k, dq, ranges, names, tm, log, gc, gidx, t, targets)
    tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': 'q0_knee working state, condition E5'}
    fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', tt)
        write(rd / 'config.json', fm_new)
    rec = run_record(tm, t0, {'variant': 'E5', 'page_cache_residency_before': residency, 'marks': marks, 'limits': limits, 'targets': len(targets),
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'auto_paths': dict(Counter(x.get('path') for x in recs)),
                              'search': srch, 'gradient_free_E5_stage': r['gradient_free'], 'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward'],
                              'nonfinite': len(r['nonfinite']), 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'marks': marks, 'gradient_free': r['gradient_free']}))


def timing(out, log, phase):
    sched = REFRESH_SCHEDULE if phase == 'refresh' else [('E5', None)]
    for v, r in sched:
        name = f'refresh_{v}_r{r}' if phase == 'refresh' else 'e2e_E5'
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
    er = [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE if v != 'B']
    exp = rr + ['e2e_E5']
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    ctl, prevc = read(out / 'controls.json'), read(KS / 'controls.json')
    st, kn = read(STATIC_TABLE), read(KNEE)
    ft = {v: read(p) for v, p in FROZEN_TABLES.items()}
    fc = {v: read(p) for v, p in FROZEN.items()}
    t5, c5 = read(W / 'refresh_E5_r1/table.json'), read(W / 'refresh_E5_r1/config.json')
    low = read(out / 'low5_nodes.json')['nodes']
    lows = set(low)
    hist = history()
    res['1_inputs'] = {'history': hist, 'targets_equal_d5': low == read(LOW5)['nodes'], 'pass': hist['pass'] and low == read(LOW5)['nodes']}
    res['2_weight_rows'] = {'equal_A': all(t5['B'][n] == st['B'][n] for n in names)}
    res['2_weight_rows']['pass'] = res['2_weight_rows']['equal_A']
    fb5 = set(runs['refresh_E5_r1']['fallback_static_rows'])
    res['3_target_rows'] = {'rows': len(lows), 'equal_E': all(t5['C'][n] == ft['E']['C'][n] for n in low if n not in fb5)}
    res['3_target_rows']['pass'] = res['3_target_rows']['equal_E']
    res['4_nontarget_rows'] = {'rows': len(names) - len(lows), 'equal_A': all(t5['C'][n] == st['C'][n] for n in names if n not in lows or n in fb5)}
    res['4_nontarget_rows']['pass'] = res['4_nontarget_rows']['equal_A']
    expected = {'B': {n: st['B'][n] for n in names}, 'C': {n: (ft['E']['C'][n] if n in lows and n not in fb5 else st['C'][n]) for n in names}}
    res['expected_table_E_targets_plus_A'] = {'equal': {k: t5[k] == expected[k] for k in ('B', 'C')}}
    z = np.load(W / 'refresh_E5_r1/new_error_stats_targets.npz')
    zi = read(W / 'refresh_E5_r1/new_error_stats_targets_index.json')
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    rel5 = 0.0
    for nm in low:
        if nm in fb5:
            continue
        node, p = '.'.join(nm.split('.')[:2]), nm.split('.')[2]
        nq, grow = zi['n_q0'][node], np.asarray(gc[gidx['row'][node]])
        v = np.zeros(len(QUANT))
        for s in ('x', 'y'):
            col = ss.SLOT_COL[(p, s)]
            a_, b_ = ss.COLOFF[col]
            arr = z[f'{node}|{col}']
            G2v, GSv = grow[0, a_:b_], grow[1, a_:b_]
            for i in range(len(QUANT)):
                mu = arr[0, i] / nq
                v[i] += float(np.sum(G2v / nq * (arr[1, i] - nq * mu ** 2)) + np.sum(GSv * mu ** 2))
        rel5 = max(rel5, max(rr_(v[i], t5['C'][nm][f]) for i, f in enumerate(QUANT)))
    res['5_recompute'] = {'numpy_target_act_max_rel': rel5, 'pass': rel5 <= TOL_REL}
    drr = dr.route_hash(dict(np.load(KS / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    c6 = {k: {'ce': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'] == prevc['q0_knee_forward_ce'], 'hidden': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
              'routes': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr} for k in rr}
    c6['e2e_E5'] = {'ce': runs['e2e_E5']['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden': runs['e2e_E5']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha']}
    g_prev = read(KS / 'work/runs/refresh_B_r1/run.json')['input_grad_sha_per_layer']
    bgrad = {k: runs[k]['input_grad_sha_per_layer'] == g_prev and runs[k]['recompute_max_abs'] == 0 for k in rr if k.startswith('refresh_B')}
    res['6_forward'] = {'runs': c6, 'B_input_gradients_equal_record': bgrad, 'pass': all(all(v.values()) for v in c6.values()) and all(bgrad.values())}
    gf = {k: runs[k]['gradient_free'] for k in er}
    gf['e2e_E5'] = runs['e2e_E5']['gradient_free_E5_stage']
    res['7_gradient_free'] = {'runs': gf, 'cache_use': {k: runs[k]['cache_use'] for k in rr},
                              'columns_computed_E5': {k: runs[k]['columns_computed'] for k in rr if k.startswith('refresh_E5')},
                              'pass': all(v['autograd_calls'] == {} and v['grad_enabled_inside'] is False for v in gf.values())
                              and all(runs[k]['cache_use']['as_expected'] for k in rr)}
    lim = kn['limits']
    rc = resources(c5, names, elems, flops)
    res['8_validity'] = {'fallback_count': len(fb5), 'fallback_rows': sorted(fb5), 'fallback_reason': 'target expert with no q0_knee calibration token (static row kept)',
                         'finite': all(runs[k]['nonfinite'] == 0 for k in exp) and all(runs[k].get('grads_finite', True) for k in rr),
                         'covers_exactly_searched': sorted(c5) == sorted(names) and sorted(t5['B']) == sorted(names) and sorted(t5['C']) == sorted(names),
                         'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in c5.values()), 'resources': rc,
                         'within_limits': rc['M_bytes'] <= lim['max_bytes'] and rc['BOPs'] <= lim['max_bops'],
                         'search_within_budget_all_runs': all(runs[k]['search']['within_budget'] for k in exp), 'e2e_limits_equal': runs['e2e_E5']['limits'] == lim}
    v8 = res['8_validity']
    v8['pass'] = bool(v8['finite'] and v8['covers_exactly_searched'] and v8['legal'] and v8['within_limits'] and v8['search_within_budget_all_runs'] and v8['e2e_limits_equal'])
    reps = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        tref = t5 if v == 'E5' else ft[v]
        cref = c5 if v == 'E5' else fc[v]
        reps[v] = {'tables': all({kk: vv for kk, vv in read(W / k / 'table.json').items() if kk in ('B', 'C')} == {kk: tref[kk] for kk in ('B', 'C')} for k in ks_),
                   'configs': all(read(W / k / 'config.json') == cref for k in ks_)}
    reps['e2e_E5'] = {'static_table': {kk: vv == st[kk] for kk, vv in read(W / 'e2e_E5/table_static.json').items() if kk in ('B', 'C')},
                      'grad_cache': sha(W / 'e2e_E5/grad_stats_cache.npy') == sha(GC), 'knee': read(W / 'e2e_E5/q0_knee.json') == read(Q0K),
                      'targets': runs['e2e_E5']['targets'] == len(low), 'E5_table': {kk: read(W / 'e2e_E5/table.json')[kk] == t5[kk] for kk in ('B', 'C')},
                      'E5_config': read(W / 'e2e_E5/config.json') == c5}
    reps['expected_table'] = res['expected_table_E_targets_plus_A']['equal']
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['9_repeats'] = reps
    res['9_repeats']['pass'] = flat(reps)
    res['identical'] = {f'E5={c}': c5 == fc[c] for c in fc}
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['inputs_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['pass'] = bool(not res['missing_runs'] and all(res[k]['pass'] for k in ('1_inputs', '2_weight_rows', '3_target_rows', '4_nontarget_rows', '5_recompute',
                                                                                 '6_forward', '7_gradient_free', '8_validity', '9_repeats'))
                       and res['inputs_unchanged'] and res['ste'] and res['controls_stage_pass'])
    res['seconds'] = time.perf_counter() - t0
    write(out / 'check.json', res)
    log(json.dumps(dr.clean({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in res.items() if k not in ('seconds',)})))


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
    write(out / 'configs/E5.json', read(W / 'refresh_E5_r1/config.json'))
    write(out / 'tables/E5.json', read(W / 'refresh_E5_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E', 'E5')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    lows = set(read(out / 'low5_nodes.json')['nodes'])
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'E5 = refresh repeat 1; others = frozen records',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c]), 'resources': resources(cfg[c], names, elems, flops)} for c in cfg},
           'frozen_equal_records': {c: cfg[c] == read(p) for c, p in FROZEN.items()},
           'identical': {f'E5={c}': cfg['E5'] == cfg[c] for c in ('A', 'B', 'C', 'D4', 'D5', 'E')},
           'pairwise_differences': {f'E5-{c}': diff(cfg['E5'], cfg[c]) for c in ('A', 'B', 'E', 'D5')} | {f'{c}-A': diff(cfg[c], cfg['A']) for c in ('B', 'E', 'D5')},
           'changed_vs_A_in_target_set': {c: {'targets': sum(cfg[c][n] != cfg['A'][n] for n in lows), 'other': sum(cfg[c][n] != cfg['A'][n] for n in names if n not in lows)}
                                          for c in ('B', 'E', 'D5', 'E5')},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in ('E5',)},
           'tables': {'E5.json': sha(out / 'tables/E5.json'), **{f'frozen/{v}': sha(p) for v, p in FROZEN_TABLES.items()}},
           'grad_cache': {'npy': sha(GC), 'index': sha(GC_INDEX)}, 'low5_nodes_sha256': sha(out / 'low5_nodes.json'), 'knee_sha256': sha(KNEE),
           'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'), 'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_in_target_set')}))


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
    old_t, old = np.load(KE / 'tokens.npy')[:CTRL], read(KE / 'eval.json')['ce']
    pc = {}
    for c in ('fp32', 'A', 'B', 'E'):
        r = dr.forward(old_t, None if c == 'fp32' else read(FROZEN[c]), dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    conds = ('A', 'B', 'E', 'D5', 'E5')
    runs, shared, by_hash = {}, {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in conds:
        h = fz['configs'][c]['config_hash']
        if h in by_hash:
            runs[c] = runs[by_hash[h]]; shared[c] = by_hash[h]
            continue
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        by_hash[h] = c
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/E5.json'), dq, ranges, record=True, log=log, label='rerun E5')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_E5_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['E5']['ce'][:CTRL])))),
                                    'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['E5']['hidden_sha'][:CTRL]))),
                                    'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['E5']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in conds},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in conds}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_E5_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_E5_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_E5_first_chunk']['routes_equal'] and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl,
                              'seconds': {c: runs[c]['seconds'] for c in runs if c not in shared}, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                              'environment': dr.env_info()})
    ss.merge_timing(out, 'eval', {**{f'{c}_s': runs[c]['seconds'] for c in runs if c not in shared}, 'rerun_s': r2['seconds'],
                                  'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps(dr.clean({'controls_pass': ctl['pass'], 'program_control': pc, 'shared': shared})))


def timing_summary(out):
    W = out / 'work/runs'
    runs = {p.name: read(p / 'run.json') for p in sorted(W.iterdir()) if (p / 'run.json').exists()}
    jl = [json.loads(x) for x in (out / 'timing.jsonl').read_text().splitlines()]
    proc = {}
    for j in jl:
        if j['stage'] == 'refresh':
            proc[f'refresh_{j["variant"]}_r{j["rep"]}'] = j['seconds']
        elif j['stage'] == 'e2e':
            proc['e2e_E5'] = j['seconds']
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        s[f'refresh_{v}'] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                             'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                             'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks_]) for t in sorted({t for k in ks_ for t in runs[k]['timers']})},
                             'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                             'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks_}}
    e = runs['e2e_E5']
    s['e2e_E5_measured_single'] = {'method_s': e['method_s'], 'process_wall_s': proc['e2e_E5'], 'marks': e['marks'], 'categories': e['categories'],
                                   'peak_gpu_gb': e['peak_gpu_gb'], 'peak_cpu_gb': e['peak_cpu_gb'], 'page_cache_residency_before': e['page_cache_residency_before']}
    et = read(KE / 'analysis.json')['timing']
    s['historical'] = {'e2e_measured_single': {**et['historical']['e2e_measured_single'], 'E': et['e2e_E_measured_single']['method_s']},
                       'auto_median_s': et['historical']['auto_median_s'], 'static_gc_single': et['static_gc_single']['method_s'],
                       'note': 'previous knee rounds, same machine and code; historical single measurements'}
    comp = et['E_total_estimate_sum_of_stage_medians_s']['components']
    s['E5_total_estimate_sum_of_stage_medians_s'] = {'value': comp['static_without_caches'] + comp['gradient_cache_build_write_single'] + comp['auto_median_previous_round']
                                                     + s['refresh_E5']['method_s']['median'],
                                                     'components': {**{k: comp[k] for k in ('static_without_caches', 'gradient_cache_build_write_single', 'auto_median_previous_round')},
                                                                    'E5_median': s['refresh_E5']['method_s']['median']}}
    s['extra_with_caches_median_s'] = {v: s[f'refresh_{v}']['method_s']['median'] for v in VARIANTS}
    s['note'] = 'method = run time minus control blocks; estimates are sums of stage medians, not a measured pipeline median'
    return s


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    from run_olmoe_multipoint import resources, problem
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
    pairs = {'E5_minus_A': ('E5', 'A'), 'E5_minus_E': ('E5', 'E'), 'E5_minus_B': ('E5', 'B'),
             'E5_minus_D5': ('E5', 'D5'), 'B_minus_A': ('B', 'A'), 'E_minus_A': ('E', 'A'), 'D5_minus_A': ('D5', 'A')}
    D = {k: A[a] - A[b] for k, (a, b) in pairs.items()}
    res = {'n_articles': N, 'bonferroni_level': 100 - 2 * CI_CORR[0], 'primary': {k: st(D[k]) for k in ('E5_minus_A', 'E5_minus_E', 'E5_minus_B')},
           'secondary': {k: st(D[k]) for k in ('E5_minus_D5', 'B_minus_A', 'E_minus_A', 'D5_minus_A')},
           'damage': {c: st(A[c] - A['fp32']) for c in ('A', 'B', 'E', 'D5', 'E5')}}
    for k in ('E5_minus_A', 'E5_minus_E'):
        lo, hi = res['primary'][k]['ci_bonferroni']
        res['primary'][k]['decision'] = 'E5_better_confirmed' if hi < 0 else 'E5_worse_confirmed' if lo > 0 else 'not_confirmed'
    lo, hi = res['primary']['E5_minus_B']['ci_bonferroni']
    res['primary']['E5_minus_B']['decision'] = (f'noninferior_within_{NI_MARGIN}' if hi < NI_MARGIN else f'beyond_margin_{NI_MARGIN}' if lo > NI_MARGIN
                                                else 'insufficient_evidence')
    for k, v in res['secondary'].items():
        lo, hi = v['ci95']
        v['direction_95'] = 'first_better' if hi < 0 else 'first_worse' if lo > 0 else 'not_confirmed'
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E', 'E5')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    t5 = read(out / 'tables/E5.json')
    lim = read(KNEE)['limits']
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    s5 = read(out / 'work/runs/refresh_E5_r1/run.json')['search']
    res['search_E5'] = {k: s5[k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
    res['search_E5']['gap_abs'] = s5['objective'] - s5['lower_bound']
    res['search_E5']['gap_rel'] = (s5['objective'] - s5['lower_bound']) / s5['objective']
    cross = {}
    for c in cfg:
        r = resources(cfg[c], names, elems, flops)
        cross[c] = {'objective_on_E5_table': obj(t5, cfg[c]), 'feasible_at_knee_limits': r['M_bytes'] <= lim['max_bytes'] and r['BOPs'] <= lim['max_bops'],
                    'below_E5_solution': obj(t5, cfg[c]) < s5['objective'] and c != 'E5'}
    res['cross_scores_on_E5_table'] = cross
    res['known_configuration_cheaper_than_E5_solution'] = sorted(c for c, v in cross.items() if v['below_E5_solution'] and v['feasible_at_knee_limits'])
    res['configs'] = {c: v['resources'] for c, v in fz['configs'].items()}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_in_target_set', 'transitions_vs_A')}
    res['targets'] = {k: v for k, v in read(out / 'low5_nodes.json').items() if k != 'nodes'}
    res['targets']['columns_computed_E5_r1'] = read(out / 'work/runs/refresh_E5_r1/run.json')['columns_computed']
    _, _, pnames, _, elems2, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    conds = ('A', 'B', 'E', 'D5', 'E5')
    for c in ('fp32',) + conds:
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
    b8_s = sum(rowsof(counts['fp32'], n) * elems2[n] * 64 for n in pnames)
    b8_all = b8_s + sum(counts['fp32'][l, e] * 3 * elem_u * 64 for l, e in unsearched)
    cost = {}
    for c in conds:
        sb = float(sum(rowsof(counts[c], n) * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
        ub = float(sum(counts[c][l, e] * 3 * elem_u * 16 * 32 for l, e in unsearched))
        cost[c] = {'relative_searched_to_int8_fp32_counts': sb / b8_s, 'relative_total_to_all_int8_fp32_counts': (sb + ub) / b8_all,
                   'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
                   'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
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
    log(json.dumps(dr.clean({'primary': {k: (v['mean'], v['ci_bonferroni'], v['decision']) for k, v in res['primary'].items()},
                             'secondary': {k: (v['mean'], v['ci95'], v['direction_95']) for k, v in res['secondary'].items()},
                             'cheaper_known': res['known_configuration_cheaper_than_E5_solution']})))


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
