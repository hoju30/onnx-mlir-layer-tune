#!/usr/bin/env python3
"""OLMoE: does the low-bit selective sensitivity refresh carry over to the normal auto-selected knee?
Pre-registered in plan.json / plan.md. No BC formula change, no router training, no new model, no threshold change,
no default-pipeline change.

Knee: the corrected static BC table (runs/olmoe_proxy_refresh_20261022/table_static.json) searched with the verified
OLMoE research flow (run_olmoe_search_final.make_solver("lpfirst") inside mp_auto.auto_select, 17 coarse samples,
2 refinements, original 2-D normalized-chord knee) -> q0_knee and its exact solve limits (search_budget x auto_select
reference bytes / BOPs).
A static knee (q0_knee). B full refresh in q0_knee's working state. D sensitivity-only refresh of the projections whose
q0_knee weight or activation format has <= 4 bits (static error statistics, other rows static). B and D re-searched once
at the knee limits. Same math, gradient semantics and code path as run_olmoe_selective_sensitivity.

Stages (one --out-dir): plan, auto (--rep k), controls, refresh (--variant B|D --rep k), e2e (--variant A|B|D),
timing (--phase auto|refresh|e2e), check, freeze, data, eval, analyze.
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
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss

HERE = Path(__file__).resolve().parent
SS = HERE / 'runs/olmoe_selective_sensitivity_20261022'
SF = HERE / 'runs/olmoe_search_final_20261016'
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE = PR / 'table_static.json'
D2 = PR / 'weight_error_d2.npz'
ACT_CACHE = SS / 'work/runs/static_cache_r0/act_err_cache.npy'
ACT_INDEX = SS / 'work/runs/static_cache_r0/act_err_cache_index.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL = 10000, 0.0005, 1e-12
AGGR_DELTA_LOW = -0.0017320154321293945       # aggressive D - A point estimate (runs/olmoe_selective_sensitivity_20261022)
SEEDS = {'order': 22501, 'start': 22502, 'bootstrap': 22511}
AUTO_REPS = (1, 2, 3)
REFRESH_SCHEDULE = [('B', 1), ('D', 1), ('D', 2), ('B', 2), ('B', 3), ('D', 3)]
E2E_SCHEDULE = ['A', 'B', 'D']
CATEGORIES = dict(ss.CATEGORIES, pruning_search=('pruning', 'search', 'auto_search'))
REUSE_CODE = ['olmoe_common.py', 'run_olmoe_search_final.py', 'run_native_bc_pipeline.py', 'run_olmoe_proxy_refresh.py',
              'run_olmoe_fixed_route.py', 'run_olmoe_multipoint.py', 'mp_auto.py', 'mp_milp.py', 'run_solver_regression.py',
              'htq_pareto_search.py', 'run_olmoe_dynamic_route.py', 'run_olmoe_selective_sensitivity.py']


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_selective_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def watch_files():
    return ss.model_files()[:-1] + [D2, ACT_CACHE]


def reuse_hashes():
    """dependencies of the FP32 static statistics reused from the aggressive selective-sensitivity run, current vs recorded
    there (plan inputs / model files, freeze tables / cache / code)."""
    ssp, ssf = read(SS / 'plan.json'), read(SS / 'freeze.json')
    rel = lambda p: str(p.relative_to(HERE))
    rows = {'model_files': ({f.name: sha(f) for f in ss.model_files()}, ssp['model_files']),
            'calibration': (sha(CALIB), ssp['inputs'][rel(CALIB)]), 'ranges': (sha(RANGES), ssp['inputs'][rel(RANGES)]),
            'stage1_routes_manifest': (sha(S1 / 'routes_manifest.json'), ssp['inputs'][rel(S1 / 'routes_manifest.json')]),
            'static_table': (sha(STATIC_TABLE), ssf['tables']['static']), 'weight_error_d2': (sha(D2), ssp['inputs'][rel(D2)]),
            'act_err_cache': (sha(ACT_CACHE), ssf['cache']['act_err_cache.npy']),
            'act_err_cache_index': (sha(ACT_INDEX), ssf['cache']['act_err_cache_index.json']),
            **{f'code:{f}': (sha(HERE / f), ssf['code'][f]) for f in REUSE_CODE}}
    res = {k: {'current': a, 'recorded': b, 'equal': a == b} for k, (a, b) in rows.items()}
    res['ss_check_pass'] = read(SS / 'check.json')['pass']
    return res


def run_record(tm, t0, extra):
    import torch
    total = time.perf_counter() - t0
    timers = dict(tm.t)
    cats = {c: sum(timers.get(k, 0.0) for k in ks) for c, ks in CATEGORIES.items()}
    cats['unclassified'] = total - sum(timers.values())
    return {'total_s': total, 'method_s': total - cats['controls'], 'timers': timers, 'categories': cats,
            'process_start_to_run_s': time.time() - total - T_PROC, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'peak_cpu_gb': ss.peak_cpu_gb(), 'gpu': torch.cuda.get_device_name(0), **extra}


# --------------------------------------------------------------------------- #
# BC tables: run_olmoe_selective_sensitivity.build (same op order), plus an optional record of the target statistics
# --------------------------------------------------------------------------- #
def build(variant, dq, ranges, names, tm, log, fm=None, d2z=None, cache=None, cidx=None, static_tab=None, targets=None,
          collect=None, stats_out=None):
    """variant 'static' (fm None: FP32 reference, computes weight errors), 'B' (full statistics in fm's state), 'D'
    (cache given: sensitivity-only for targets, other rows static). stats_out (dict): per target projection, the
    per-channel weight sensitivity sum G_c and the activation gradient statistics G2 / GS of both slots (control)."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    TAPS, SLOT_COL, COLOFF, COL_SRC, NCOL = ss.TAPS, ss.SLOT_COL, ss.COLOFF, ss.COL_SRC, ss.NCOL
    with tm.block('setup'):
        cfg, ck = oc.config(), oc.Checkpoint()
        toks_np = np.load(CALIB)
        toks = torch.from_numpy(toks_np).to('cuda')
        nW = len(toks_np)
        cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    with tm.block('ckpt_load'):
        g = ck.global_('cuda', torch.float32)
    tset = set(names) if targets is None else set(targets)
    want_all = defaultdict(dict)
    for n in names:
        if n in tset:
            l, e, p = n.split('.')
            want_all[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    STEQ = pr.ste_fn()
    ids_rec = np.zeros((nW, NL, SEQ, K), np.int16); w_rec = np.zeros((nW, NL, SEQ, K), np.float32)
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = ss.load_layer(ck, l, fm, dq, ranges, tm)
            with tm.block('forward_routing'):
                h, idx, rw = pr.layer_tap(wq, h, cfg, cos, sin, None, None, pr.act_map(hk))
            routes.append((idx, rw)); hs.append(h)
            with tm.block('controls'):
                ids_rec[:, l] = idx.view(nW, SEQ, K).to(torch.int16).cpu().numpy(); w_rec[:, l] = rw.view(nW, SEQ, K).cpu().numpy()
            del wq, hk
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
        with tm.block('controls'):
            hc = h.cpu().numpy()
            hidden_sha = [hashlib.sha256(np.ascontiguousarray(hc[k]).tobytes()).hexdigest() for k in range(nW)]
    with tm.block('backward'):
        hT = hs[-1].clone().requires_grad_(True)
        ce_bwd = head_ce(g, hT, toks, cfg)
        ce_bwd.sum().backward()
        gn = hT.grad
    with tm.block('controls'):
        if [float(x) for x in ce_bwd.detach().cpu().numpy()] != ce_fwd:
            raise RuntimeError('head CE differs between forward and backward')
    B_, C_, meta, d2_new = {}, {}, {}, {}
    recompute, gn_sha = [], []
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = ss.load_layer(ck, l, fm, dq, ranges, tm)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = ss.layer_tap_sel(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(hk, STEQ), want, cap)
            o.backward(gn)
            gn = hi.grad
        with tm.block('controls'):
            recompute.append(float((o.detach() - hs[l + 1]).abs().max()))
            gnc = gn.cpu().numpy()
            gn_sha.append(hashlib.sha256(gnc.tobytes()).hexdigest())
            gfin &= bool(np.isfinite(gnc).all())
        with torch.no_grad():
            for e in sorted(want):
                node = f'L{l}.E{e}'
                if e not in cap:
                    meta[node] = {'routed_rows': 0, 'windows': 0}
                    continue
                c = cap[e]
                tok = c['tok']
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                meta[node] = {'routed_rows': int(n), 'windows': len(wins)}
                gst, errm, vmemo = {}, {}, {}
                if cache is not None:
                    with tm.block('cache_read'):
                        crow = torch.from_numpy(np.ascontiguousarray(cache[cidx['row'][node]])).to('cuda')
                    ns = cidx['n'][node]
                for p in want[e]:
                    name = f'{node}.{p}'
                    kX, kG, kx, ky = TAPS[p]
                    with tm.block('weight_sens'):
                        GY, X = c[kG].grad, c[kX].detach()
                        Gc = torch.zeros(GY.shape[1], dtype=torch.float64, device='cuda')
                        for k in wins:
                            m = win == k
                            Gc += ((GY[m].double().t() @ X[m].double()) ** 2).sum(1)
                        gcn = Gc.cpu().numpy()
                        gfin &= bool(np.isfinite(gcn).all())
                    if fm is None:
                        with tm.block('weight_error'):
                            W = wq[p][e]
                            d2_new[name] = {f: ((dq.weight(W, f) - W).double() ** 2).sum(1).cpu().numpy() for f in QUANT}
                        dd = d2_new[name]
                    else:
                        with tm.block('cache_read_d2'):
                            dd = {f: d2z[f'{name}|{f}'] for f in QUANT}
                    with tm.block('table_assembly'):
                        B_[name] = {'fp32': 0.0}
                        for f in QUANT:
                            B_[name][f] = float((gcn / X.shape[1] * dd[f]).sum())
                    with tm.block('act_grad_stats'):
                        for k_ in (kx, ky):
                            if k_ not in gst:
                                gt = c[k_].grad.double()
                                gst[k_] = ((gt ** 2).sum(0), torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0))
                    with tm.block('controls'):
                        gfin &= all(bool(torch.isfinite(v).all()) for k_ in (kx, ky) for v in gst[k_])
                        if stats_out is not None:
                            stats_out[name] = {'Gc': gcn, 'n_in': int(X.shape[1]), 'n_q0': int(n),
                                               **{f'{s}_{q}': gst[k_][j].cpu().numpy() for s, k_ in (('x', kx), ('y', ky)) for j, q in enumerate(('G2', 'GS'))}}
                    if cache is None:
                        rng = ranges[name]
                        with tm.block('act_error'):
                            for s, k_ in (('x', kx), ('y', ky)):
                                key = (k_, tuple(rng[s]))
                                if key in errm:
                                    continue
                                t = c[k_].detach()
                                G2, GS = gst[k_]
                                vals, s12 = [], []
                                for f in QUANT:
                                    err = (dq.act(t, f, rng[s]) - t).double()
                                    s1 = err.sum(0)
                                    s2 = (err ** 2).sum(0)
                                    mu = s1 / n
                                    vals.append(float((G2 / n * (s2 - n * mu ** 2)).sum() + (GS * mu ** 2).sum()))
                                    if collect is not None:
                                        s12.append((s1, s2))
                                errm[key] = (vals, s12)
                            vx, vy = errm[(kx, tuple(rng['x']))][0], errm[(ky, tuple(rng['y']))][0]
                            C_[name] = {'fp32': 0.0}
                            for i, f in enumerate(QUANT):
                                cc = 0.0
                                cc += vx[i]
                                cc += vy[i]
                                C_[name][f] = cc
                    else:
                        with tm.block('act_assembly'):
                            tot = None
                            for s, k_ in (('x', kx), ('y', ky)):
                                col = SLOT_COL[(p, s)]
                                if col not in vmemo:
                                    a_, b_ = COLOFF[col]
                                    vmemo[col] = ss.act_cost_from_stats(crow[0, :, a_:b_], crow[1, :, a_:b_], ns, *gst[k_])
                                tot = vmemo[col] if tot is None else tot + vmemo[col]
                            v = tot.cpu().numpy()
                            C_[name] = {'fp32': 0.0, **{f: float(v[i]) for i, f in enumerate(QUANT)}}
                if collect is not None:
                    with tm.block('cache_build'):
                        row = torch.empty((2, len(QUANT), NCOL), dtype=torch.float64, device='cuda')
                        for col, (p0, s0, k0) in COL_SRC.items():
                            a_, b_ = COLOFF[col]
                            s12 = errm[(k0, tuple(ranges[f'{node}.{p0}'][s0]))][1]
                            row[0, :, a_:b_] = torch.stack([x[0] for x in s12])
                            row[1, :, a_:b_] = torch.stack([x[1] for x in s12])
                        collect['data'][collect['row'][node]] = row.cpu().numpy()
                        collect['n'][node] = n
        del wq, hk, cap
        log(f'{variant}: layer {l} ({sum(tm.t.values()):.0f}s)')
    fallback = []
    if fm is not None:
        for n in names:
            if n in tset and meta['.'.join(n.split('.')[:2])]['routed_rows'] == 0:
                fallback.append(n)
        if (fallback or targets is not None) and static_tab is None:
            with tm.block('input_read'):
                static_tab = read(STATIC_TABLE)
        with tm.block('table_assembly'):
            for n in names:
                if n in fallback or n not in tset:
                    B_[n], C_[n] = static_tab['B'][n], static_tab['C'][n]
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'd2': d2_new, 'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd,
            'hidden_sha': hidden_sha, 'input_grad_sha_per_layer': gn_sha[::-1], 'recompute_max_abs_per_layer': recompute[::-1],
            'grads_finite': gfin, 'nonfinite': nonfinite, 'fallback_static_rows': fallback,
            'updated_projections': sorted(n for n in tset if n not in fallback)}


def auto_search(table, prob, tm, logdir, tag):
    """the verified OLMoE research search: LP-first per-budget solver inside mp_auto.auto_select (17 samples, 2 refinements,
    normalized-chord knee), on the exact-dominance-pruned table."""
    import mp_auto
    from run_native_bc_pipeline import GRID11
    from run_olmoe_multipoint import nondominated, pruned
    from run_olmoe_search_final import TIME_LIMIT, make_solver
    names, nodes, elems, flops = prob
    bc = lambda n, w, a: table['B'][n][w] + table['C'][n][a]
    with tm.block('pruning'):
        keep = nondominated(names, bc)
    recs, slog = [], []
    logdir.mkdir(parents=True, exist_ok=True)
    orig = mp_auto.milp_select
    mp_auto.milp_select = make_solver('lpfirst', recs, logdir, tag=tag)
    try:
        with tm.block('auto_search'):
            sel = mp_auto.auto_select(nodes, GRID11, GRID11, pruned(bc, keep), elems, flops, samples=17, refinements=2,
                                      time_limit=TIME_LIMIT, solver_log=slog)
    finally:
        mp_auto.milp_select = orig
    return sel, recs, slog, sum(len(v) for v in keep.values())


def knee_of(sel, names):
    k = sel['selected']
    if k['origin'] != 'milp' or k['search_budget'] is None:
        raise SystemExit(f'knee is an endpoint ({k["origin"]}): stop and report')
    ref = sel['reference']
    limits = {'search_budget': k['search_budget'], 'max_bytes': k['search_budget'] * ref['weight_bytes'],
              'max_bops': k['search_budget'] * ref['bops'], 'reference': ref,
              'definition': 'auto_select passes budget * reference weight bytes and budget * reference BOPs to the solver'}
    return {n: k['formats'][n] for n in names}, limits


def targets_info(q0k, names):
    low = ss.lowbit_nodes(q0k)
    cidx = read(ACT_INDEX)
    experts = sorted({'.'.join(n.split('.')[:2]) for n in low})
    return {'definition': 'q0_knee weight or activation format with <= 4 bits (htq_pareto_search.FORMAT_BITS); frozen from the '
                          'initial q0_knee', 'count': len(low), 'total': len(names), 'fraction': len(low) / len(names),
            'by_proj': dict(Counter(n.split('.')[2] for n in low)), 'by_layer': {str(l): v for l, v in sorted(Counter(int(n.split('.')[0][1:]) for n in low).items())},
            'experts': len(experts), 'experts_by_layer': {str(l): v for l, v in sorted(Counter(int(e.split('.')[0][1:]) for e in experts).items())},
            'weight_formats': dict(Counter(q0k[n]['w'] for n in low)), 'act_formats': dict(Counter(q0k[n]['a'] for n in low)),
            'fp32_calibration_rows_of_target_experts': int(sum(cidx['n'][e] for e in experts)),
            'fp32_calibration_rows_total': int(sum(cidx['n'].values())), 'nodes': low}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    names, nodes, elems, flops, rows = pr.problem_static()
    reuse = reuse_hashes()
    reuse_ok = all(v['equal'] for k, v in reuse.items() if k != 'ss_check_pass') and reuse['ss_check_pass']
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in ss.input_files() + [SS / 'plan.json', SS / 'freeze.json', SS / 'check.json',
                                                                              SS / 'tokens.npy', SS / 'eval.json', SS / 'configs/A.json',
                                                                              SS / 'configs/D.json', SS / 'lowbit_nodes.json', ACT_CACHE, ACT_INDEX,
                                                                              SS / 'work/runs/refresh_D_r1/table.json',
                                                                              SF / 'configs/lp_knee.json', SF / 'configs/milp_0707.json',
                                                                              SF / 'B/search_lpfirst.json']}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': ['1. On the normal auto-selected knee, is the low-bit sensitivity-only refresh D better than the static knee A?',
                     f'2. Is D not worse than the full refresh B by more than {NI_MARGIN} nat/token?',
                     '3. Is the extra cost worth it when the knee has fewer <= 4-bit projections?'],
        'not_this_round': 'no BC formula change, no router training, no new model, no low-bit threshold change, no new knee '
                          'grouping rule or solver threshold, no default-pipeline change',
        'knee': {'table': f'{STATIC_TABLE.relative_to(HERE)} (corrected static BC table; reproduced bitwise by the selective-'
                          'sensitivity run)',
                 'search': 'run_olmoe_search_final.make_solver("lpfirst") replacing mp_auto.milp_select inside mp_auto.auto_select '
                           '(samples 17, refinements 2, time_limit 300 s, original normalized-chord knee), on the exact-dominance-'
                           'pruned table (run_olmoe_multipoint.nondominated / pruned); same nodes, GRID11 x GRID11, resource model '
                           '(FP32 calibration routed rows) and unsearched-projection handling',
                 'selection': 'q0_knee = auto_select["selected"] of repeat 1 (pre-registered); no manual choice; if the knee is an '
                              'endpoint, stop and report',
                 'limits': 'exact common limits for B / D = selected search_budget x auto_select reference weight bytes and '
                           'x reference BOPs (what auto_select passed to the solver); never rounded R, never the configuration\'s '
                           'actual use',
                 'historical_reference': 'LP-first knee of runs/olmoe_search_final_20261016 (R 0.7070606, budget 0.70710678; '
                                         'historical table with the pre-fix int activation QDQ) and the MILP knee (R 0.7071056): '
                                         'format differences reported; a different knee is explained, never replaced',
                 'repeats': 'auto search run 3 times (repeat 1 canonical, repeats 2-3 timing); all must give the same knee'},
        'conditions': {'A': 'static knee: q0_knee', 'B': 'full refresh in q0_knee\'s working state: all sensitivities and activation '
                                                         'errors recomputed, weight errors reused from the static cache, one re-search '
                                                         'at the knee limits',
                       'D': 'low-bit sensitivity-only at the knee: targets = q0_knee weight or activation <= 4 bits (recomputed from '
                            'q0_knee, frozen, not the aggressive list); weight and activation sensitivities of every candidate format '
                            'of a target updated; static error statistics (FP32 reference, normalized by the static routed rows); '
                            'non-targets keep their static rows exactly; one re-search at the knee limits',
                       'search_scope': 'every projection may change format; no new frontier, no other knee, no loss-based fallback',
                       'identical': 'identical configurations are reported as such and share one evaluation; no threshold change '
                                    'to create differences'},
        'math': 'unchanged from run_olmoe_selective_sensitivity (plan there): summed-CE gradient scale; per-output-channel weight '
                'sensitivity S_new[c] = G_c / n_in with D_static; activation variance + coherent bias; gate / up shared input with '
                'its total gradient; identity STE through activation QDQ; q0_knee dynamic forward, recorded routes constant in the '
                'backward; no parameter update. D activation cost A_new = sum (G2_new var_static + GS_new mu_static^2), static '
                'statistics normalized by the static routed rows n_s',
        'reuse': {'what': 'FP32 static statistics of the aggressive selective-sensitivity run: static table, weight errors '
                          '(runs/olmoe_proxy_refresh_20261022/weight_error_d2.npz) and the activation-error cache '
                          f'({ACT_CACHE.relative_to(HERE)}); q0_knee gradients and activation errors are recomputed',
                  'condition': 'every dependency hash equal to the record of that run', 'hashes': reuse, 'reuse_ok': reuse_ok,
                  'build_cost': 'measured there (static plain / with cache, 3 repeats each) and re-measured here in the end-to-end '
                                'runs'},
        'code': 'run_olmoe_selective_sensitivity.build copied here unchanged in op order, with an optional record of the target '
                'statistics; regression control: the copy reproduces the aggressive D table of that run bitwise',
        'controls': {
            '1_static': 'static table equal to the corrected static table (hash; and the end-to-end static rebuilds bitwise); the '
                        'static table re-solved at the knee limits reproduces q0_knee',
            '2_forward': 'q0_knee calibration forward of every B / D run: per-window CE, final hidden sha256 and routes equal an '
                         'independent run_olmoe_dynamic_route.forward',
            '3_input_gradient': 'per-layer input-gradient sha256 identical across every B / D run; recomputation max abs 0',
            '4_unselected': 'D non-target rows bitwise equal the static table',
            '5_selected': 'D target weight rows bitwise equal B\'s; D target rows equal an independent numpy recomputation of '
                          f'the sensitivity-only formula from the recorded statistics, the cache and the weight errors (relative <= {TOL_REL})',
            '6_validity': 'finite, legal formats, within the exact knee limits',
            '7_inputs': 'checkpoint, router, ranges, quantizer code and calibration hashes unchanged (before / after)',
            '8_repeats': 'identical tables and configurations across repeats (auto, B, D, end-to-end); main = repeat 1',
            'regression': 'copied build reproduces runs/olmoe_selective_sensitivity_20261022 refresh_D_r1 table bitwise',
            'ste': 'STE unit test'},
        'timing': {'rule': 'one process per run, CUDA synchronization around every timed block, no page-cache dropping; residency '
                           '(mincore) of the checkpoint shards, the weight-error cache and the activation cache recorded before each '
                           'run; peak GPU / CPU memory',
                   'auto': f'auto search repeats {list(AUTO_REPS)} (r1 canonical, then controls, then r2, r3)',
                   'refresh': f'B / D, 3 each, order {REFRESH_SCHEDULE}', 'e2e': f'one end-to-end run each, order {E2E_SCHEDULE}: '
                   'static statistics (A: no cache written; B: + weight-error cache; D: + weight-error and activation caches) -> '
                   'auto knee search -> (B / D: refresh -> re-search); single measurements',
                   'categories': CATEGORIES,
                   'report': {'a_extra': 'B / D refresh run median with the static caches and q0_knee available',
                              'b_total_estimate': 'static statistics (selective-sensitivity medians, its fixed-budget search removed) '
                                                  '+ auto search median + refresh median: labelled as a sum of stage medians',
                              'b_total_measured': 'the end-to-end runs (single measurements)',
                              'separate': 'controls, repeats and evaluation'}},
        'freeze': 'configurations, tables, knee, limits, targets, code and input hashes frozen before any evaluation article',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries; WikiText-2 '
                               'removed; within-WT103 duplicates removed; near-duplicates (>= 10% shingles) skipped; every WikiText-103 '
                               'article of every earlier manifest excluded (runs/**/*segments*.json, only this run skipped by path)',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests,
                       'eligible_before_length_and_neardup': len(elig), 'seeds': {k: SEEDS[k] for k in ('order', 'start')},
                       'conditions': 'FP32, A, B, D; original router; each its own dynamic route; same GPU, numerics, chunk 128, '
                                     'quantizer',
                       'primary': 'Delta_low = NLL(D) - NLL(A): CI upper < 0 -> D improves the static knee; lower > 0 -> D worse; '
                                  'otherwise not confirmed',
                       'keep': f'Delta_keep = NLL(D) - NLL(B): CI upper < +{NI_MARGIN} -> D not worse than full refresh by more '
                               f'than {NI_MARGIN}; lower > +{NI_MARGIN} -> worse beyond the margin; otherwise insufficient evidence',
                       'secondary': 'B - A (does full refresh help at the knee itself)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL); no extra '
                                    'articles, threshold or configuration changes',
                       'not_improvement': 'non-inferiority is not improvement: if neither D nor B is confirmed better than A, D '
                                          'is not claimed worth adding',
                       'interpretation': [f'(1) D - A confirmed < 0 and keep upper < +{NI_MARGIN}: both criteria met',
                                          '(2) D - A not confirmed and B - A not confirmed: full refresh has no confirmed gain at the knee',
                                          '(3) D - A not confirmed but B - A confirmed < 0: full refresh helps but D did not keep it',
                                          '(4) D - A confirmed < 0: weigh its size and time cost against A (descriptive, no threshold)',
                                          f'(5) a CI containing both 0 and the aggressive effect {AGGR_DELTA_LOW:.6f}: too wide to decide'],
                       'descriptive': 'actual dynamic routed counts and BOPs proxy including the unsearched experts'},
        'not_claimed': 'other q0, calibration, resource points, models, native kernels',
        'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not reuse_ok:
        raise SystemExit('reuse hashes differ: investigate before reusing the static statistics')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 預設 knee 點的 selective sensitivity refresh', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- knee：修正後 static BC 表 + 已驗證的研究搜尋（LP 優先取代 auto_select 的每預算求解，17 樣本、2 輪 refinement、原二維 normalized-chord knee），'
          '第 1 次 auto 的 selected 為 q0_knee；不手動選點、不手動設 0.7071；若為端點則停止。',
          '- 共同上限：selected 的 search_budget × auto_select 參考 bytes／BOPs（就是求解器收到的精確上限），不用四捨五入的 R、不用配置實際用量。',
          '- A：q0_knee；B：q0_knee 工作狀態完整 refresh；D：q0_knee 中 weight 或 activation ≤4 bit 的投影只更新 sensitivity（重算並凍結目標集合），其餘沿用 static；B、D 在 knee 上限各重新搜尋一次，所有節點都可改格式。',
          '- 數學與梯度語義同上一輪；D 的 activation 成本 Σ(G2_new·var_static + GS_new·μ_static²)，以 static routed rows 正規化。',
          f'- 重用上一輪的 FP32 static 統計（static 表、ΔW、activation 快取），依賴 hash 全部一致（reuse_ok = {reuse_ok}）；q0_knee 的梯度與誤差全部重算。',
          '- 控制：static 表一致與 knee 重解、q0_knee forward 與獨立 forward 一致、input gradient 逐位元、D 非目標 = static、D 目標 = 公式（numpy 獨立重算）且 weight 列 = B、合法與上限、輸入不變、重複一致、複製的 build 重現上一輪 D 表。',
          f'- 計時：auto 3 次；B／D 各 3 次（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；端到端 A、B、D 各 1 次（實測）；另報各階段中位數相加的估算並明確標示。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、D；seed 22501／22502／22511。',
          f'- 主要：Δ_low = D − A；非劣：Δ_keep = D − B，上界 < +{NI_MARGIN}；次要：B − A。非劣不等於改善。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; reuse_ok {reuse_ok}; previous WT103 {len(used)}; eligible {len(elig)}')


def auto(out, log, rep):
    import torch
    from run_olmoe_multipoint import resources
    from run_olmoe_search_final import frontier_rounds
    from mp_auto import config_hash
    dr.gpu_setup()
    tag = f'auto_r{rep}'
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
        table = read(STATIC_TABLE)
    sel, recs, slog, kept = auto_search(table, (names, nodes, elems, flops), tm, out / 'logs', tag)
    q0k, limits = knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'selection.json', {'selection': sel, 'calls': recs, 'solver_log': slog, 'options_kept': kept})
        write(rd / 'q0_knee.json', q0k)
    rec = run_record(tm, t0, {'rep': rep, 'page_cache_residency_before': residency, 'calls': len(recs),
                              'paths': dict(Counter(r.get('path') for r in recs)), 'failures': sel['failures'],
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'size', 'bops', 'search_budget', 'origin', 'solver_status')},
                              'limits': limits, 'config_hash': config_hash(q0k), 'attempted_budgets': sel['attempted_budgets'],
                              'code': code_hashes()})
    if rep == 1:
        with tm.block('controls'):
            r = resources(q0k, names, elems, flops)
            fm_re, s_re = ss.search_table(table, (names, nodes, elems, flops), limits, pr.Timer(), out / 'logs', 'knee_resolve')
            hist = {k: read(SF / f'configs/{k}.json') for k in ('lp_knee', 'milp_0707')}
            sfl = read(SF / 'B/search_lpfirst.json')['selection']['selected']
            diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                                 'act': sum(a[n]['a'] != b[n]['a'] for n in names),
                                 'weight_transitions': dict(Counter(f'{b[n]["w"]}->{a[n]["w"]}' for n in names if a[n]['w'] != b[n]['w'])),
                                 'act_transitions': dict(Counter(f'{b[n]["a"]}->{a[n]["a"]}' for n in names if a[n]['a'] != b[n]['a']))}
            ti = targets_info(q0k, names)
        (out / 'knee').mkdir(exist_ok=True)
        write(out / 'knee/q0_knee.json', q0k)
        write(out / 'knee/knee.json', {
            'source': f'work/runs/{tag}/selection.json (auto_select selected, repeat 1)', 'limits': limits,
            'selected': rec['knee'], 'resources_exact': r, 'within_limits': r['M_bytes'] <= limits['max_bytes'] and r['BOPs'] <= limits['max_bops'],
            'config_hash': config_hash(q0k), 'weight_formats': dict(Counter(v['w'] for v in q0k.values())),
            'act_formats': dict(Counter(v['a'] for v in q0k.values())), 'frontier_rounds': frontier_rounds(sel),
            'frontier': [{k: c[k] for k in ('resource', 'objective', 'size', 'bops', 'search_budget', 'origin', 'solver_status')} for c in sel['frontier']],
            'knee_scores': sel['knee_scores'], 'attempted_budgets': sel['attempted_budgets'], 'bounds': sel['bounds'],
            'static_resolve_at_limits_reproduces_q0_knee': fm_re == q0k, 'static_resolve_objective': s_re['objective'],
            'historical': {'lp_knee_search_final': {k: sfl[k] for k in ('resource', 'objective', 'search_budget')},
                           'diff_vs_lp_knee': diff(q0k, hist['lp_knee']), 'diff_vs_milp_knee': diff(q0k, hist['milp_0707'])}})
        write(out / 'lowbit_nodes.json', ti)
        rec['knee_files_written'] = True
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'total_s': rec['total_s'], 'knee': rec['knee'], 'paths': rec['paths']}))


def controls(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'controls.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    q0k = read(out / 'knee/q0_knee.json')
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    toks = np.load(CALIB)
    rd = dr.forward(toks, q0k, dq, ranges, record=True, log=lambda m: None, label='q0_knee calibration', chunk=len(toks))
    names, nodes, elems, flops, rows = pr.problem_static()
    reg = build('D', dq, ranges, names, pr.Timer(), log, fm=read(PR / 'configs/q0.json'), d2z=np.load(D2), cache=np.load(ACT_CACHE, mmap_mode='r'),
                cidx=read(ACT_INDEX), static_tab=read(STATIC_TABLE), targets=read(SS / 'lowbit_nodes.json')['nodes'])
    ref = read(SS / 'work/runs/refresh_D_r1/table.json')
    ctl = {'q0_knee_forward_ce': rd['ce'], 'q0_knee_forward_hidden_sha': rd['hidden_sha'], 'q0_knee_forward_routes_hash': dr.route_hash(rd['routes']),
           'q0_knee_forward_finite': rd['finite_ce'], 'ste_unit_test': pr.ste_unit_test(dq, ranges),
           'regression_copy_reproduces_aggressive_D': {'B_equal': reg['B'] == ref['B'], 'C_equal': reg['C'] == ref['C']},
           'lowbit_recount': len(ss.lowbit_nodes(q0k)), 'lowbit_file_count': read(out / 'lowbit_nodes.json')['count'],
           'page_cache_residency': ss.resident_fraction(watch_files()), 'seconds': time.perf_counter() - t0}
    np.savez(out / 'work/q0_knee_calibration_routes_dr_forward.npz', **rd['routes'])
    ctl['pass'] = bool(rd['finite_ce'] and ctl['ste_unit_test']['pass'] and all(ctl['regression_copy_reproduces_aggressive_D'].values())
                       and ctl['lowbit_recount'] == ctl['lowbit_file_count'])
    write(out / 'controls.json', ctl)
    log(json.dumps({k: ctl[k] for k in ('regression_copy_reproduces_aggressive_D', 'lowbit_recount', 'pass')}))


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
        fm = read(out / 'knee/q0_knee.json')
        limits = read(out / 'knee/knee.json')['limits']
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    with tm.block('cache_read_d2'):
        d2z = np.load(D2)
    cache = cidx = static_tab = targets = None
    if variant == 'D':
        with tm.block('cache_read'):
            cache = np.load(ACT_CACHE, mmap_mode='r')
            cidx = read(ACT_INDEX)
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
            targets = read(out / 'lowbit_nodes.json')['nodes']
    stats = {} if (variant == 'D' and rep == 1) else None
    r = build(variant, dq, ranges, names, tm, log, fm=fm, d2z=d2z, cache=cache, cidx=cidx, static_tab=static_tab, targets=targets, stats_out=stats)
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
                              'updated_projections': len(r['updated_projections']), 'fallback_static_rows': r['fallback_static_rows'],
                              'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'], 'routes_hash': dr.route_hash(r['routes']),
                              'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'recompute_max_abs': max(r['recompute_max_abs_per_layer']),
                              'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']),
                              'q0_routed_rows_total': int(sum(v['routed_rows'] for v in r['meta'].values())),
                              'q0_routed_rows_by_expert': {k: v['routed_rows'] for k, v in r['meta'].items()}, 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'total_s': rec['total_s'], 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path']}))


def e2e(out, log, variant):
    """one process: static statistics (A: no cache written; B: + weight-error cache; D: + weight-error and activation caches)
    -> auto knee search -> (B / D: refresh with the caches just written -> re-search at its knee limits)."""
    import torch
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    tag = f'e2e_{variant}'
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
    collect = None
    if variant == 'D':
        with tm.block('cache_build'):
            experts = sorted({'.'.join(n.split('.')[:2]) for n in names}, key=lambda s: (int(s.split('.')[0][1:]), int(s.split('.')[1][1:])))
            collect = {'row': {e: i for i, e in enumerate(experts)}, 'n': {}, 'data': np.zeros((len(experts), 2, 10, ss.NCOL), np.float64)}
    st = build('static', dq, ranges, names, tm, log, collect=collect)
    t = {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'}
    marks = {'static_done_s': time.perf_counter() - t0}
    with tm.block('table_write'):
        write(rd / 'table_static.json', t)
    if variant in ('B', 'D'):
        with tm.block('cache_write_d2'):
            np.savez(rd / 'weight_error_d2.npz', **{f'{n}|{f}': v for n, d in st['d2'].items() for f, v in d.items()})
    if variant == 'D':
        with tm.block('cache_write'):
            np.save(rd / 'act_err_cache.npy', collect['data'])
            write(rd / 'act_err_cache_index.json', {'experts': list(collect['row']), 'row': collect['row'], 'n': collect['n'],
                                                    'formats': list(QUANT), 'columns': ss.COLS, 'stats': ['sum_err', 'sum_err_sq'],
                                                    'shape': list(collect['data'].shape), 'dtype': 'float64'})
    marks['static_written_s'] = time.perf_counter() - t0
    del st
    sel, recs, slog, kept = auto_search(t, prob, tm, out / 'logs', tag)
    q0k, limits = knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'q0_knee.json', q0k)
        write(rd / 'selection_summary.json', {'selected': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')},
                                              'limits': limits, 'attempted_budgets': sel['attempted_budgets'], 'paths': dict(Counter(r.get('path') for r in recs))})
    marks['knee_done_s'] = time.perf_counter() - t0
    extra = {}
    if variant in ('B', 'D'):
        with tm.block('cache_read_d2'):
            d2z = np.load(rd / 'weight_error_d2.npz')
        cache = cidx = targets = None
        if variant == 'D':
            with tm.block('cache_read'):
                cache = np.load(rd / 'act_err_cache.npy', mmap_mode='r')
                cidx = read(rd / 'act_err_cache_index.json')
            targets = ss.lowbit_nodes(q0k)
        r = build(variant, dq, ranges, names, tm, log, fm=q0k, d2z=d2z, cache=cache, cidx=cidx, static_tab=t, targets=targets)
        tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'],
              'updated_projections': len(r['updated_projections']), 'state': f'q0_knee working state, condition {variant}'}
        fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
        with tm.block('table_write'):
            write(rd / 'table.json', tt)
            write(rd / 'config.json', fm_new)
        extra = {'search': srch, 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward']}
    rec = run_record(tm, t0, {'variant': variant, 'page_cache_residency_before': residency, 'marks': marks,
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'limits': limits,
                              'auto_paths': dict(Counter(r_.get('path') for r_ in recs)), 'code': code_hashes(), **extra})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'total_s': rec['total_s'], 'method_s': rec['method_s'], 'marks': marks}))


def timing(out, log, phase):
    sched = {'auto': [(None, r) for r in AUTO_REPS], 'refresh': REFRESH_SCHEDULE, 'e2e': [(v, None) for v in E2E_SCHEDULE]}[phase]
    for v, r in sched:
        name = {'auto': f'auto_r{r}', 'refresh': f'refresh_{v}_r{r}', 'e2e': f'e2e_{v}'}[phase]
        if (out / 'work/runs' / name / 'run.json').exists():
            log(f'{name}: done already, kept')
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', phase, '--out-dir', str(out)]
        cmd += (['--variant', v] if v else []) + (['--rep', str(r)] if r else [])
        t = time.time()
        p = subprocess.run(cmd, cwd=HERE)
        log(f'{name}: exit {p.returncode}, process wall {time.time() - t:.1f}s')
        if p.returncode:
            raise SystemExit(f'{name} failed')


def check(out, log):
    t0 = time.perf_counter()
    if (out / 'check.json').exists():
        raise SystemExit('exists')
    from run_native_bc_pipeline import QUANT
    names, nodes, elems, flops, rows = pr.problem_static()
    pl = read(out / 'plan.json')
    W = out / 'work/runs'
    exp = [f'auto_r{r}' for r in AUTO_REPS] + [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE] + [f'e2e_{v}' for v in E2E_SCHEDULE]
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    q0k, kn, ctl = read(out / 'knee/q0_knee.json'), read(out / 'knee/knee.json'), read(out / 'controls.json')
    st = read(STATIC_TABLE)
    reuse = reuse_hashes()
    res['reuse_hashes_still_equal'] = all(v['equal'] for k, v in reuse.items() if k != 'ss_check_pass') and reuse['ss_check_pass']
    # 1. static
    e2e_static = {v: read(W / f'e2e_{v}/table_static.json') for v in E2E_SCHEDULE}
    res['1_static'] = {'table_hash_equal_corrected': sha(STATIC_TABLE) == read(SS / 'freeze.json')['tables']['static'],
                       'e2e_static_tables_bitwise': {v: t['B'] == st['B'] and t['C'] == st['C'] for v, t in e2e_static.items()},
                       'e2e_weight_error_equal': {v: ss.d2_equal(W / f'e2e_{v}/weight_error_d2.npz', D2) for v in ('B', 'D')},
                       'e2e_act_cache_equal': sha(W / 'e2e_D/act_err_cache.npy') == sha(ACT_CACHE),
                       'knee_resolve_reproduces_q0_knee': kn['static_resolve_at_limits_reproduces_q0_knee'], 'q0_knee_within_limits': kn['within_limits']}
    c1 = res['1_static']
    c1['pass'] = bool(c1['table_hash_equal_corrected'] and all(c1['e2e_static_tables_bitwise'].values()) and all(c1['e2e_weight_error_equal'].values())
                      and c1['e2e_act_cache_equal'] and c1['knee_resolve_reproduces_q0_knee'] and c1['q0_knee_within_limits'])
    # 2. forward
    rr = [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE]
    drr = dr.route_hash(dict(np.load(out / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    c2 = {k: {'ce_equal': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden_equal': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
              'routes_equal': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr} for k in rr}
    c2.update({f'e2e_{v}': {'ce_equal': runs[f'e2e_{v}']['ce_forward'] == ctl['q0_knee_forward_ce'],
                            'hidden_equal': runs[f'e2e_{v}']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha']} for v in ('B', 'D')})
    res['2_forward'] = {'runs': c2, 'pass': all(all(v.values()) for v in c2.values())}
    # 3. input gradients
    g0 = runs['refresh_B_r1']['input_grad_sha_per_layer']
    gk = rr + ['e2e_B', 'e2e_D']
    res['3_input_gradient'] = {'identical': all(runs[k]['input_grad_sha_per_layer'] == g0 for k in gk),
                               'recompute_max_abs': max(runs[k]['recompute_max_abs'] for k in rr)}
    res['3_input_gradient']['pass'] = res['3_input_gradient']['identical'] and res['3_input_gradient']['recompute_max_abs'] == 0
    # 4 / 5. D rows
    tD, tB = read(W / 'refresh_D_r1/table.json'), read(W / 'refresh_B_r1/table.json')
    low = read(out / 'lowbit_nodes.json')['nodes']
    lows = set(low)
    fbD = set(runs['refresh_D_r1']['fallback_static_rows'])
    res['4_unselected'] = {'rows': len(names) - len(lows), 'equal_static': all(tD['B'][n] == st['B'][n] and tD['C'][n] == st['C'][n] for n in names if n not in lows or n in fbD)}
    res['4_unselected']['pass'] = res['4_unselected']['equal_static']
    z = np.load(W / 'refresh_D_r1/target_stats.npz')
    sm = read(W / 'refresh_D_r1/target_stats_meta.json')
    d2z, cache, cidx = np.load(D2), np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    relw, rela, relu, relq = 0.0, 0.0, 0.0, 0.0
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    for n in low:
        if n in fbD:
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
            relw = max(relw, rr_(wv, tD['B'][n][f]))
            rela = max(rela, rr_(av, tD['C'][n][f]))
            relu = max(relu, rr_(au, tD['C'][n][f]))
            relq = max(relq, rr_(aq, tD['C'][n][f]))
    res['5_selected'] = {'rows': len(lows), 'fallback': sorted(fbD), 'weight_rows_equal_B': all(tD['B'][n] == tB['B'][n] for n in lows if n not in fbD),
                         'numpy_recompute_weight_max_rel': relw, 'numpy_recompute_act_max_rel': rela,
                         'descriptive_user_form_G2_var_plus_GS_mu2_max_rel': relu,
                         'descriptive_wrong_normalization_by_q0_rows_max_rel': relq,
                         'note': 'numpy recomputation (independent implementation, BC op-order form G2 / n_s (S2 - n_s mu^2) + GS mu^2); '
                                 'the user form G2 (S2 / n_s - mu^2) is algebraically equal (different rounding); normalizing by the '
                                 'q0_knee rows instead of n_s is shown to differ (sensitivity of this check)'}
    res['5_selected']['pass'] = bool(res['5_selected']['weight_rows_equal_B'] and relw <= TOL_REL and rela <= TOL_REL)
    # 6. validity
    lim = kn['limits']
    v6 = {k: {'finite': runs[k]['grads_finite'] and runs[k]['nonfinite'] == 0, 'legal': runs[k]['search']['all_formats_legal'],
              'within_limits': runs[k]['search']['within_budget']} for k in rr}
    v6['limits_equal_knee'] = all(runs[k]['limits'] == lim for k in [f'auto_r{r}' for r in AUTO_REPS] + [f'e2e_{v}' for v in E2E_SCHEDULE])
    res['6_validity'] = v6
    v6['pass'] = all(all(v.values()) for k, v in v6.items() if isinstance(v, dict)) and v6['limits_equal_knee']
    # 7. inputs
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['7_inputs'] = {'input_hashes_unchanged': after == read(out / 'input_hashes_before.json'),
                       'quantizer_code_equal_plan': all(runs[k]['code'][f] == pl['code_hashes'][f] for k in runs for f in dr.SCRIPTS)}
    res['7_inputs']['pass'] = res['7_inputs']['input_hashes_unchanged'] and res['7_inputs']['quantizer_code_equal_plan']
    # 8. repeats
    a1 = read(W / 'auto_r1/selection.json')['selection']
    reps = {'auto': {f'r{r}': read(W / f'auto_r{r}/q0_knee.json') == q0k and runs[f'auto_r{r}']['attempted_budgets'] == a1['attempted_budgets'] for r in AUTO_REPS},
            'e2e_knee': {v: read(W / f'e2e_{v}/q0_knee.json') == q0k for v in E2E_SCHEDULE}}
    for v in 'BD':
        ks = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        t1, c1_ = read(W / ks[0] / 'table.json'), read(W / ks[0] / 'config.json')
        reps[f'refresh_{v}'] = {'tables': all(read(W / k / 'table.json') == t1 for k in ks), 'configs': all(read(W / k / 'config.json') == c1_ for k in ks),
                                'e2e_config': read(W / f'e2e_{v}/config.json') == c1_,
                                'e2e_table': {kk: vv == t1[kk] for kk, vv in read(W / f'e2e_{v}/table.json').items() if kk in ('B', 'C')}}
    res['8_repeats'] = reps
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['8_repeats_pass'] = flat(reps)
    res['regression'] = ctl['regression_copy_reproduces_aggressive_D']
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['fallback_static_rows'] = {k: runs[k]['fallback_static_rows'] for k in rr}
    res['pass'] = bool(not res['missing_runs'] and res['reuse_hashes_still_equal'] and all(res[k]['pass'] for k in
                       ('1_static', '2_forward', '3_input_gradient', '4_unselected', '5_selected', '6_validity', '7_inputs'))
                       and res['8_repeats_pass'] and all(res['regression'].values()) and res['ste'] and res['controls_stage_pass'])
    res['seconds'] = time.perf_counter() - t0
    write(out / 'check.json', res)
    log(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in res.items()
                    if k not in ('fallback_static_rows', 'seconds', '8_repeats')} | {'8_repeats': res['8_repeats_pass']}))


def freeze(out, log):
    from mp_auto import config_hash
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'check.json')['pass']:
        raise SystemExit('controls failed: not freezing')
    W = out / 'work/runs'
    (out / 'configs').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    src = {'A': out / 'knee/q0_knee.json', 'B': W / 'refresh_B_r1/config.json', 'D': W / 'refresh_D_r1/config.json'}
    for c, p in src.items():
        write(out / f'configs/{c}.json', read(p))
    write(out / 'tables/static.json', read(STATIC_TABLE))
    for c in 'BD':
        write(out / f'tables/{c}.json', read(W / f'refresh_{c}_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in src}
    names = sorted(cfg['A'])
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    low = set(read(out / 'lowbit_nodes.json')['nodes'])
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'auto repeat 1 knee; refresh repeat 1 of B / D',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c])} for c in cfg},
           'identical': {f'{a}={b}': cfg[a] == cfg[b] for a, b in (('A', 'B'), ('A', 'D'), ('B', 'D'))},
           'pairwise_differences': {f'{a}-{b}': diff(cfg[a], cfg[b]) for a, b in (('B', 'A'), ('D', 'A'), ('D', 'B'))},
           'changed_vs_A_in_lowbit_set': {c: {'lowbit': sum(cfg[c][n] != cfg['A'][n] for n in low), 'other': sum(cfg[c][n] != cfg['A'][n] for n in names if n not in low)} for c in 'BD'},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in 'BD'},
           'tables': {p.name: sha(p) for p in sorted((out / 'tables').iterdir())}, 'knee_sha256': sha(out / 'knee/knee.json'),
           'lowbit_nodes_sha256': sha(out / 'lowbit_nodes.json'), 'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'),
           'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('identical', 'pairwise_differences')}))


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
    old_t, old = np.load(SS / 'tokens.npy')[:CTRL], read(SS / 'eval.json')['ce']
    pc = {}
    for c, fm in (('fp32', None), ('A', read(SS / 'configs/A.json')), ('D', read(SS / 'configs/D.json'))):
        r = dr.forward(old_t, fm, dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs, shared, by_hash = {}, {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in 'ABD':
        h = fz['configs'][c]['config_hash']
        if h in by_hash:
            runs[c] = runs[by_hash[h]]; shared[c] = by_hash[h]
            continue
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        by_hash[h] = c
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/D.json'), dq, ranges, record=True, log=log, label='rerun D')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_D_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['D']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['D']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['D']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in 'ABD'},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in 'ABD'}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_D_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_D_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_D_first_chunk']['routes_equal'] and ctl['finite']
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
    proc = {}
    for j in jl:
        if j['stage'] == 'auto':
            proc[f'auto_r{j["rep"]}'] = j['seconds']
        elif j['stage'] == 'refresh':
            proc[f'refresh_{j["variant"]}_r{j["rep"]}'] = j['seconds']
        elif j['stage'] == 'e2e':
            proc[f'e2e_{j["variant"]}'] = j['seconds']
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    groups = {'auto': [f'auto_r{r}' for r in AUTO_REPS], 'refresh_B': [f'refresh_B_r{r}' for r in (1, 2, 3)], 'refresh_D': [f'refresh_D_r{r}' for r in (1, 2, 3)]}
    s = {}
    for g, ks in groups.items():
        s[g] = {'method_s': med([runs[k]['method_s'] for k in ks]), 'process_wall_s': med([proc[k] for k in ks]),
                'categories': {c: med([runs[k]['categories'][c] for k in ks]) for c in list(CATEGORIES) + ['unclassified']},
                'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks]) for t in sorted({t for k in ks for t in runs[k]['timers']})},
                'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks),
                'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks}}
    s['e2e_measured'] = {v: {'method_s': runs[f'e2e_{v}']['method_s'], 'process_wall_s': proc[f'e2e_{v}'], 'marks': runs[f'e2e_{v}']['marks'],
                             'categories': runs[f'e2e_{v}']['categories'], 'peak_gpu_gb': runs[f'e2e_{v}']['peak_gpu_gb'],
                             'peak_cpu_gb': runs[f'e2e_{v}']['peak_cpu_gb'], 'page_cache_residency_before': runs[f'e2e_{v}']['page_cache_residency_before']}
                         for v in E2E_SCHEDULE}
    # static statistics from the selective-sensitivity repeats, its fixed-budget pruning / search removed
    SW = SS / 'work/runs'
    sstat = lambda v, drop: float(np.median([read(SW / f'static_{v}_r{r}/run.json')['method_s'] - sum(read(SW / f'static_{v}_r{r}/run.json')['timers'].get(t, 0.0) for t in drop)
                                             for r in (1, 2, 3)]))
    stat = {'A': sstat('plain', ('pruning', 'search', 'cache_write_d2')), 'B': sstat('plain', ('pruning', 'search')), 'D': sstat('cache', ('pruning', 'search'))}
    am, bm, dm = s['auto']['method_s']['median'], s['refresh_B']['method_s']['median'], s['refresh_D']['method_s']['median']
    s['total_estimate_sum_of_stage_medians_s'] = {'A': stat['A'] + am, 'B': stat['B'] + am + bm, 'D': stat['D'] + am + dm,
                                                  'static_statistics_component': stat,
                                                  'note': 'sum of stage medians (static from the selective-sensitivity repeats); not a measured pipeline median'}
    s['extra_with_caches_median_s'] = {'B': bm, 'D': dm}
    s['note'] = 'method = run time minus control blocks; process wall includes Python start, imports and CUDA initialization'
    return s


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    if not ev['controls']['pass']:
        raise SystemExit('evaluation controls failed')
    A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))
    st = lambda x: {'mean': float(x.mean()), 'ci95': np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1)),
                    'better_articles': int((x < 0).sum()), 'worse_articles': int((x > 0).sum()), 'zero_articles': int((x == 0).sum())}
    D = {'D_minus_A': A['D'] - A['A'], 'D_minus_B': A['D'] - A['B'], 'B_minus_A': A['B'] - A['A']}
    res = {'n_articles': N, 'primary_delta_low': st(D['D_minus_A']), 'keep_delta': st(D['D_minus_B']), 'secondary_B_minus_A': st(D['B_minus_A']),
           'damage': {c: st(A[c] - A['fp32']) for c in 'ABD'}}
    lo, hi = res['primary_delta_low']['ci95']
    res['primary_decision'] = 'D_improves_static_knee_confirmed' if hi < 0 else 'D_worse_confirmed' if lo > 0 else 'direction_not_confirmed'
    klo, khi = res['keep_delta']['ci95']
    res['keep_decision'] = (f'D_noninferior_to_full_refresh_within_{NI_MARGIN}' if khi < NI_MARGIN else
                            f'D_worse_than_full_refresh_by_more_than_{NI_MARGIN}' if klo > NI_MARGIN else 'insufficient_evidence')
    blo, bhi = res['secondary_B_minus_A']['ci95']
    res['secondary_decision'] = 'B_better_than_A_confirmed' if bhi < 0 else 'B_worse_confirmed' if blo > 0 else 'not_confirmed'
    wide = lambda ci: ci[0] <= 0 <= ci[1] and ci[0] <= AGGR_DELTA_LOW <= ci[1]
    interp = []
    if hi < 0 and khi < NI_MARGIN:
        interp.append('1_both_criteria_met')
    if hi < 0:
        interp.append('4_D_improves_weigh_size_and_cost')
    if not (hi < 0) and not (lo > 0):
        interp.append('3_full_refresh_helps_but_D_did_not_keep_it' if bhi < 0 else '2_full_refresh_has_no_confirmed_gain_at_knee')
    if wide([lo, hi]):
        interp.append('5_delta_low_CI_too_wide')
    if wide([blo, bhi]):
        interp.append('5_B_minus_A_CI_too_wide')
    res['interpretation'] = interp
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in 'ABD'}
    tabs = {c: read(out / f'tables/{c}.json') for c in ('static', 'B', 'D')}
    names = sorted(cfg['A'])
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    res['objectives'] = {t: {c: obj(tabs[t], cfg[c]) for c in 'ABD'} for t in tabs}
    W = out / 'work/runs'
    srch = {c: read(W / f'refresh_{c}_r1/run.json')['search'] for c in 'BD'}
    res['search'] = {c: {k: srch[c][k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
                     for c in 'BD'}
    res['knee'] = {k: v for k, v in read(out / 'knee/knee.json').items() if k not in ('frontier', 'knee_scores', 'frontier_rounds')}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_in_lowbit_set', 'transitions_vs_A')}
    ti = read(out / 'lowbit_nodes.json')
    rD = read(W / 'refresh_D_r1/run.json')
    low = set(ti['nodes'])
    lowexp = {'.'.join(n.split('.')[:2]) for n in low}
    res['targets'] = {k: v for k, v in ti.items() if k != 'nodes'} | {
        'q0_knee_calibration_rows_of_target_experts': int(sum(rD['q0_routed_rows_by_expert'].get(e, 0) for e in lowexp)),
        'q0_knee_calibration_rows_total': read(W / 'refresh_B_r1/run.json')['q0_routed_rows_total'],
        'target_experts_without_q0_tokens': sorted(e for e in lowexp if rD['q0_routed_rows_by_expert'].get(e, 0) == 0),
        'fallback_static_rows': rD['fallback_static_rows']}
    from run_olmoe_multipoint import problem
    _, _, pnames, _, elems, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    for c in ['fp32', 'A', 'B', 'D']:
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
    for c in 'ABD':
        sb = float(sum(rowsof(counts[c], n) * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
        ub = float(sum(counts[c][l, e] * 3 * elem_u * 16 * 32 for l, e in unsearched))
        cost[c] = {'searched_BOPs_actual': sb, 'relative_searched_to_int8_fp32_counts': sb / b8_s, 'unsearched_BOPs_bf16_fp32': ub,
                   'total_BOPs_actual': sb + ub, 'relative_total_to_all_int8_fp32_counts': (sb + ub) / b8_all,
                   'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
                   'routed_rows_to_target_experts': int(sum(counts[c][int(e.split('.')[0][1:]), int(e.split('.')[1][1:])] for e in lowexp)),
                   'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
        if c != 'A':
            desc[c]['set_change_vs_A'] = float((np.sort(np.load(out / f'routes_{shared.get(c, c)}.npz')['ids'], -1) !=
                                                np.sort(np.load(out / 'routes_A.npz')['ids'], -1)).any(-1).mean())
    cost['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l, e in unsearched))
    cost['routed_rows_total_per_condition'] = int(counts['fp32'].sum())
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
    log(json.dumps({'primary': res['primary_delta_low'], 'decision': res['primary_decision'], 'keep': res['keep_delta'],
                    'keep_decision': res['keep_decision'], 'B_minus_A': res['secondary_B_minus_A'], 'interpretation': interp}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'auto', 'controls', 'refresh', 'e2e', 'timing', 'check', 'freeze', 'data', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant')
    ap.add_argument('--rep', type=int)
    ap.add_argument('--phase', choices=['auto', 'refresh', 'e2e'])
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    if a.stage == 'auto':
        auto(a.out_dir, log, a.rep)
    elif a.stage == 'refresh':
        refresh(a.out_dir, log, a.variant, a.rep)
    elif a.stage == 'e2e':
        e2e(a.out_dir, log, a.variant)
    elif a.stage == 'timing':
        timing(a.out_dir, log, a.phase)
    else:
        {'eval': eval_}.get(a.stage, globals().get(a.stage))(a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'variant': a.variant, 'rep': a.rep, 'phase': a.phase, 'seconds': time.time() - T_PROC,
                            'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
