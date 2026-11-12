#!/usr/bin/env python3
"""OLMoE: sensitivity-only and low-bit selective sensitivity refresh of the BC proxy in q0's working state.
Pre-registered in plan.json / plan.md. Compiler analysis / automatic quantization only: no training, no parameter
update, no router fine-tuning, no BC formula change, no default integration.

A static BC: the frozen q0 of runs/olmoe_proxy_refresh_20261022 (no refresh).
B full refresh: every searched projection gets new sensitivities and new activation-error statistics in q0's working
  state (run_olmoe_proxy_refresh.build_table semantics); must reproduce that run's table_refresh.json and q1.
C sensitivity-only, all projections: new sensitivities in q0's working state; weight errors and activation-error
  statistics from the FP32 static reference:
    W[f] = sum_c S_new[c] D_static[c, f]  (S_new[c] = G_c / n_in; identical to B's weight term, which already reused D_static)
    A[f] = sum_slots sum_c (G2_new[c] var_static[c, f] + GS_new[c] mu_static[c, f]^2)
  with mu_static / var_static the per-channel mean / variance of the static errors over the static routed rows n_s,
  computed in the original BC op order as G2_new / n_s (S2 - n_s mu^2) + GS_new mu^2 from the cached per-channel
  sums S1 = sum err, S2 = sum err^2 (mu = S1 / n_s).
D low-bit sensitivity-only: as C, but only for the projections whose q0 weight or activation format has <= 4 bits
  (all candidate formats of a selected projection are recomputed); every other projection keeps its static BC row.
B, C, D: the same full q0 calibration forward (own dynamic routes), routes constant in the backward, identity STE
through activation QDQ, no optimizer; the same LP-first search, pruning, candidates and limits.

Stages (one --out-dir): plan, controls, static (--variant plain|cache --rep k), refresh (--variant B|C|D --rep k),
timing (--phase static|refresh: the pre-registered schedule, one process per run), check, freeze, data, eval, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import ctypes
import glob
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
PR = HERE / 'runs/olmoe_proxy_refresh_20261022'
PRC = HERE / 'runs/olmoe_proxy_refresh_confirm_20261022'
RFT = HERE / 'runs/olmoe_router_ft_20261022'
S1 = HERE / 'runs/olmoe_fixed_route_20261015'
MP = HERE / 'runs/olmoe_multipoint_20261015'
RANGES = MP / 'ranges.json'
CALIB = S1 / 'tokens_calibration.npy'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, LOWBIT_MAX = 10000, 0.0005, 1e-12, 4
SEEDS = {'order': 22401, 'start': 22402, 'bootstrap': 22411}
STATIC_SCHEDULE = [('cache', 0), ('plain', 1), ('cache', 1), ('cache', 2), ('plain', 2), ('plain', 3), ('cache', 3)]
REFRESH_SCHEDULE = [('B', 1), ('C', 1), ('D', 1), ('C', 2), ('D', 2), ('B', 2), ('D', 3), ('B', 3), ('C', 3)]
COLS = (('x_in', 2048), ('gate_y', 1024), ('up_y', 1024), ('down_x', 1024), ('down_y', 2048))
COLOFF, _o = {}, 0
for _k, _w in COLS:
    COLOFF[_k] = (_o, _o + _w)
    _o += _w
NCOL = _o
SLOT_COL = {('gate_proj', 'x'): 'x_in', ('up_proj', 'x'): 'x_in', ('gate_proj', 'y'): 'gate_y', ('up_proj', 'y'): 'up_y',
            ('down_proj', 'x'): 'down_x', ('down_proj', 'y'): 'down_y'}
COL_SRC = {'x_in': ('gate_proj', 'x', 'xe'), 'gate_y': ('gate_proj', 'y', 'zg'), 'up_y': ('up_proj', 'y', 'zu'),
           'down_x': ('down_proj', 'x', 'h'), 'down_y': ('down_proj', 'y', 'zd')}
# per projection: (weight-term input X, weight-term output-gradient tensor, x-slot tensor, y-slot tensor)
TAPS = {'gate_proj': ('xg', 'zg', 'xe', 'zg'), 'up_proj': ('xu', 'zu', 'xe', 'zu'), 'down_proj': ('hd', 'zd', 'h', 'zd')}
CATEGORIES = {'load_and_prep': ('setup', 'setup_inputs', 'ckpt_load', 'quant_init', 'quantize', 'input_read'),
              'forward_routing': ('forward_routing',), 'backward_recompute': ('backward',), 'weight_sensitivity': ('weight_sens',),
              'act_grad_stats': ('act_grad_stats',), 'weight_error': ('weight_error',), 'act_error': ('act_error',),
              'assembly': ('table_assembly', 'act_assembly'), 'pruning_search': ('pruning', 'search'), 'cache_build': ('cache_build',),
              'cache_write': ('cache_write', 'cache_write_d2'), 'cache_read': ('cache_read', 'cache_read_d2'),
              'output_write': ('table_write',), 'controls': ('controls',)}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def merge_timing(out, key, obj):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[key] = obj
    write(f, t)


def model_files():
    import olmoe_common as oc
    return sorted(oc.MODEL_DIR.glob('*.safetensors')) + [oc.MODEL_DIR / 'config.json']


def cache_paths(out):
    d = out / 'work/runs/static_cache_r0'
    return d / 'act_err_cache.npy', d / 'act_err_cache_index.json'


def watch_files(out):
    return model_files()[:-1] + [PR / 'weight_error_d2.npz', cache_paths(out)[0]]


def resident_fraction(paths):
    """fraction of each file's pages in the OS page cache (mincore); the cache is never dropped."""
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    ps = os.sysconf('SC_PAGESIZE')
    res = {}
    for p in map(Path, paths):
        key = str(p.relative_to(HERE)) if HERE in p.parents else p.name
        if not p.exists():
            res[key] = None
            continue
        sz = p.stat().st_size
        fd = os.open(p, os.O_RDONLY)
        try:
            a = libc.mmap(None, sz, 1, 1, fd, 0)
            vec = np.zeros((sz + ps - 1) // ps, np.uint8)
            rc = libc.mincore(a, sz, vec.ctypes.data_as(ctypes.c_void_p))
            libc.munmap(a, sz)
        finally:
            os.close(fd)
        res[key] = float((vec & 1).mean()) if rc == 0 else None
    return res


def lowbit_nodes(q0):
    from htq_pareto_search import FORMAT_BITS
    return sorted(n for n, v in q0.items() if FORMAT_BITS[v['w']] <= LOWBIT_MAX or FORMAT_BITS[v['a']] <= LOWBIT_MAX)


def used_wt103_all(out):
    """content hashes of every WikiText-103 article in an earlier manifest (runs/**/*segments*.json); only files inside
    this run's own directory are skipped (by path containment, not by a name filter)."""
    used, rows = set(), []
    for f in sorted(glob.glob(str(HERE / 'runs/**/*segments*.json'), recursive=True)):
        if Path(f).resolve().is_relative_to(out.resolve()):
            continue
        s = read(f)
        srcs = sorted({x.get('source', x.get('split', '?')) for x in s})
        wt103 = any('wikitext-103' in x for x in srcs)
        if wt103:
            used |= {x['content_sha256'] for x in s}
        rows.append({'path': str(Path(f).relative_to(HERE)), 'sha256': sha(f), 'articles': len(s), 'sources': srcs, 'wt103': wt103})
    return used, rows


def input_files():
    return [PR / f for f in ('configs/q0.json', 'configs/q1.json', 'table_static.json', 'table_refresh.json', 'weight_error_d2.npz',
                             'q0_calibration_routes.npz', 'plan.json', 'freeze.json', 'eval.json', 'tokens.npy')] + \
           [RANGES, CALIB, S1 / 'routes_manifest.json', MP / 'table.json', MP / 'table_meta.json'] + mr.WT103 + mr.WT2


CODE = ['run_olmoe_selective_sensitivity.py', 'run_olmoe_proxy_refresh.py', 'run_olmoe_route_mass_restore.py', 'run_olmoe_multipoint.py',
        'run_solver_regression.py', 'run_native_bc_pipeline.py', 'htq_pareto_search.py', 'run_olmoe_fixed_route.py']


def code_hashes():
    return {f: sha(HERE / f) for f in CODE + dr.SCRIPTS}


# --------------------------------------------------------------------------- #
# model pieces
# --------------------------------------------------------------------------- #
def experts_tap_sel(w, x2d, idx, rw, act, want, cap, E=NE):
    """run_olmoe_proxy_refresh.experts_tap (same op order); gradients retained, and tensors kept, only for the projections
    in want[e] (the update targets of expert e). retain_grad changes no value."""
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
        A = (lambda p, s, t, e=e: act[(e, p, s)](t) if (e, p, s) in act else t)
        xg = A('gate_proj', 'x', xe)
        xu = A('up_proj', 'x', xe)
        zg = F.linear(xg, w['gate_proj'][e]); g = A('gate_proj', 'y', zg)
        zu = F.linear(xu, w['up_proj'][e]); u = A('up_proj', 'y', zu)
        h = F.silu(g) * u
        hd = A('down_proj', 'x', h)
        zd = F.linear(hd, w['down_proj'][e]); d = A('down_proj', 'y', zd)
        if e in want:
            loc = {'xe': xe, 'xg': xg, 'xu': xu, 'zg': zg, 'zu': zu, 'h': h, 'hd': hd, 'zd': zd}
            grads = {k for p in want[e] for k in TAPS[p][1:]}
            for k in sorted(grads):
                loc[k].retain_grad()
            cap[e] = {**{k: loc[k] for k in grads | {TAPS[p][0] for p in want[e]}}, 'tok': tok}
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_tap_sel(w, h, cfg, cos, sin, idx, rw, act, want, cap):
    import olmoe_common as oc
    B, T, H = h.shape
    x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
    z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, H)
    return x + experts_tap_sel(w, z, idx, rw, act, want, cap).view(B, T, H)


def load_layer(ck, l, fm, dq, ranges, tm):
    import torch
    from run_olmoe_search_final import quantized_layer
    with tm.block('ckpt_load'):
        w = ck.layer(l, 'cuda', torch.float32)
    if fm is None:
        return w, None
    with tm.block('quantize'):
        wq, hk = quantized_layer(w, fm, l, dq, ranges, NE)
    return wq, hk


def act_cost_from_stats(S1_, S2_, ns, G2, GS):
    """sensitivity-only activation cost per candidate format (rows of S1_ / S2_): sum_c G2 var_s + GS mu_s^2."""
    mu = S1_ / ns
    return (G2 / ns * (S2_ - ns * mu ** 2)).sum(1) + (GS * mu ** 2).sum(1)


def build(variant, dq, ranges, names, tm, log, fm=None, d2z=None, cache=None, cidx=None, static_tab=None, targets=None,
          collect=None, extra=False):
    """variant 'static' (fm None: FP32 reference, computes weight errors), 'B' (full statistics in q0's state), 'C' / 'D'
    (cache given: sensitivity-only; D: targets = the low-bit projections, other rows static)."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
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
    # 1. forward with q0's own dynamic routes (FP32 reference for static), no grad
    ids_rec = np.zeros((nW, NL, SEQ, K), np.int16); w_rec = np.zeros((nW, NL, SEQ, K), np.float32)
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = load_layer(ck, l, fm, dq, ranges, tm)
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
    # 2. backward: head, then layer-wise recomputation with the recorded routes as constants
    with tm.block('backward'):
        hT = hs[-1].clone().requires_grad_(True)
        ce_bwd = head_ce(g, hT, toks, cfg)
        ce_bwd.sum().backward()
        gn = hT.grad
    with tm.block('controls'):
        if [float(x) for x in ce_bwd.detach().cpu().numpy()] != ce_fwd:
            raise RuntimeError('head CE differs between forward and backward')
    B_, C_, meta, d2_new, rngs, recon = {}, {}, {}, {}, {}, {}
    recompute, gn_sha = [], []
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = load_layer(ck, l, fm, dq, ranges, tm)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = layer_tap_sel(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(hk, STEQ), want, cap)
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
                            W = wq[p][e]                                   # the original FP32 checkpoint weight
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
                    if extra:
                        with tm.block('controls'):
                            rngs[name] = {s: [float(c[k_].detach().min()), float(c[k_].detach().max())] for s, k_ in (('x', kx), ('y', ky))}
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
                                    vmemo[col] = act_cost_from_stats(crow[0, :, a_:b_], crow[1, :, a_:b_], ns, *gst[k_])
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
                    if extra:
                        with tm.block('controls'):         # the sensitivity-only formula with the static gradients
                            for p in want[e]:
                                kX, kG, kx, ky = TAPS[p]
                                tot = sum(act_cost_from_stats(row[0, :, slice(*COLOFF[SLOT_COL[(p, s)]])], row[1, :, slice(*COLOFF[SLOT_COL[(p, s)]])],
                                                              n, *gst[k_]) for s, k_ in (('x', kx), ('y', ky)))
                                v = tot.cpu().numpy()
                                recon[f'{node}.{p}'] = {'fp32': 0.0, **{f: float(v[i]) for i, f in enumerate(QUANT)}}
        del wq, hk, cap
        log(f'{variant}: layer {l} ({sum(tm.t.values()):.0f}s)')
    fallback = []
    if fm is not None:
        for n in names:
            if n in tset and meta['.'.join(n.split('.')[:2])]['routed_rows'] == 0:
                fallback.append(n)
        if (fallback or targets is not None) and static_tab is None:
            with tm.block('input_read'):
                static_tab = read(PR / 'table_static.json')
        with tm.block('table_assembly'):
            for n in names:
                if n in fallback or n not in tset:
                    B_[n], C_[n] = static_tab['B'][n], static_tab['C'][n]
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'd2': d2_new, 'observed_ranges': rngs, 'recon_C': recon,
            'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd, 'hidden_sha': hidden_sha, 'input_grad_sha_per_layer': gn_sha[::-1],
            'recompute_max_abs_per_layer': recompute[::-1], 'grads_finite': gfin, 'nonfinite': nonfinite, 'fallback_static_rows': fallback,
            'updated_projections': sorted(n for n in tset if n not in fallback)}


def search_table(t, prob, budget, tm, logdir, tag):
    from run_native_bc_pipeline import GRID11
    from run_olmoe_multipoint import nondominated, pruned, resources
    from run_olmoe_search_final import make_solver
    from mp_auto import config_hash
    names, nodes, elems, flops = prob
    bc = lambda n, w, a: t['B'][n][w] + t['C'][n][a]
    with tm.block('pruning'):
        keep = nondominated(names, bc)
    recs = []
    logdir.mkdir(parents=True, exist_ok=True)
    sel = make_solver('lpfirst', recs, logdir, tag=tag)
    with tm.block('search'):
        fm, obj, status = sel(nodes, GRID11, GRID11, pruned(bc, keep), elems, flops, budget['max_bytes'], budget['max_bops'])
    fm = {n: fm[n] for n in names}
    r = resources(fm, names, elems, flops)
    return fm, {'objective': obj, 'objective_recomputed': float(sum(bc(n, fm[n]['w'], fm[n]['a']) for n in names)), 'status': status,
                'config_hash': config_hash(fm), 'record': recs[-1], 'path': recs[-1].get('path'), 'lower_bound': recs[-1].get('lower_bound'),
                'options_kept': sum(len(v) for v in keep.values()), 'options_total': 121 * len(names), 'resources': r,
                'within_budget': r['M_bytes'] <= budget['max_bytes'] and r['BOPs'] <= budget['max_bops'],
                'budget_use': {'size': r['M_bytes'] / budget['max_bytes'], 'bops': r['BOPs'] / budget['max_bops']},
                'weight_formats': dict(Counter(v['w'] for v in fm.values())), 'act_formats': dict(Counter(v['a'] for v in fm.values())),
                'all_formats_legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in fm.values())}


def run_record(tm, t0, extra):
    import torch
    total = time.perf_counter() - t0
    timers = dict(tm.t)
    cats = {c: sum(timers.get(k, 0.0) for k in ks) for c, ks in CATEGORIES.items()}
    cats['unclassified'] = total - sum(timers.values())
    return {'total_s': total, 'method_s': total - cats['controls'], 'timers': timers, 'categories': cats,
            'process_start_to_run_s': time.time() - total - T_PROC, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'peak_cpu_gb': peak_cpu_gb(), 'gpu': torch.cuda.get_device_name(0), **extra}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    names, nodes, elems, flops, rows = pr.problem_static()
    prp = read(PR / 'plan.json')
    q0 = read(PR / 'configs/q0.json')
    low = lowbit_nodes(q0)
    write(out / 'lowbit_nodes.json', {'definition': f'q0 weight or activation format with <= {LOWBIT_MAX} bits (htq_pareto_search.FORMAT_BITS)',
                                      'q0_sha256': sha(PR / 'configs/q0.json'), 'count': len(low), 'total': len(names), 'nodes': low})
    ranges = read(RANGES)
    experts = sorted({'.'.join(n.split('.')[:2]) for n in names})
    shared_ok = all(ranges[f'{e}.gate_proj']['x'] == ranges[f'{e}.up_proj']['x'] for e in experts)
    used, manifests = used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    mfiles = {f.name: sha(f) for f in model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': ['1. With the original quantization errors kept (FP32 static reference), does updating only the sensitivities '
                     'in q0\'s working state keep the gain of the full refresh?',
                     '2. Does updating only the low-bit (<= 4-bit) projections reduce time and keep the gain?'],
        'not_this_round': 'no training, no router fine-tuning, no parameter update, no BC formula change, no default '
                          'integration; no router-FT checkpoint is loaded; every working state is built from the original q0 '
                          'and the original checkpoint router',
        'budget': {k: prp['budget'][k] for k in ('max_bytes', 'max_bops', 'M_rel', 'B_rel', 'R')} |
                  {'source': 'runs/olmoe_proxy_refresh_20261022/plan.json (original common limits; not q0\'s actual use)',
                   'bops_rows': 'FP32 calibration routed rows (run_olmoe_multipoint.problem), fixed for every search'},
        'fixed': {'nodes': f'{len(names)} searched projections', 'candidates': 'GRID11 x GRID11',
                  'calibration': f'{CALIB.relative_to(HERE)} (10 x 128, sha256 {sha(CALIB)[:12]})', 'ranges': f'{RANGES.relative_to(HERE)}',
                  'quantizers': 'run_olmoe_search_final.dev_quant(corrected=True): DevQuant weights, Universal posit LUTs, corrected int '
                                'activation QDQ', 'checkpoint': 'original OLMoE checkpoint and router (hashes in model_files)',
                  'unsearched': '21 unsearched projections bf16 weight / FP32 activation', 'q0': f'{(PR / "configs/q0.json").relative_to(HERE)}',
                  'q1': f'{(PR / "configs/q1.json").relative_to(HERE)}'},
        'conditions': {
            'A': 'static BC: the original q0 (no refresh)',
            'B': 'full refresh: every searched projection gets new weight / activation sensitivities and new activation-error '
                 'statistics in q0\'s working state (run_olmoe_proxy_refresh.build_table semantics; weight errors reused from the '
                 'static build as before); must reproduce table_refresh.json and q1',
            'C': 'all sensitivity-only: new sensitivities for every searched projection; weight and activation error statistics '
                 'from the FP32 static reference',
            'D': f'low-bit sensitivity-only: new sensitivities only for the {len(low)} projections of lowbit_nodes.json (q0 weight or '
                 'activation <= 4 bits; target set frozen from the initial q0, no second round); all candidate formats of a target '
                 'are recomputed; every other projection keeps its static BC row exactly; all error statistics static'},
        'math': {
            'weight': 'S_new[c] = G_c / n_in with G_c = sum over windows of the squared per-window weight-gradient row sums '
                      '(GY^T X, GY = gradient w.r.t. the pre-y-QDQ linear output, X = the post-x-QDQ input); D_static[c, f] = '
                      'sum_j (Q_f(W) - W)^2 of the original checkpoint W (runs/olmoe_proxy_refresh_20261022/weight_error_d2.npz, '
                      'cached by the original refresh; its computation is never counted as a saving); W_new[f] = sum_c S_new[c] '
                      'D_static[c, f]. B and C share this weight term exactly (B already reused D_static).',
            'activation': 'static cache per searched expert, slot tensor (x_in shared by gate / up, gate_y, up_y, down_x, down_y), '
                          'channel and candidate format: n_s (static routed rows), S1 = sum err, S2 = sum err^2 in float64 (err = '
                          'QDQ_f(t) - t of the FP32 reference pre-QDQ tensor, fixed ranges); mu_static = S1 / n_s, var_static = '
                          'S2 / n_s - mu_static^2. G2_new[c] = sum_rows g^2 and GS_new[c] = sum_windows (sum_rows in window g)^2 '
                          'of the q0 working state (gate / up x slot: shared pre-QDQ input with its total gradient). A_new[f] = '
                          'sum_slots sum_c (G2_new var_static + GS_new mu_static^2), evaluated as G2_new / n_s (S2 - n_s mu^2) + '
                          'GS_new mu^2 (the original BC op order); n_s, not the q0 routed rows, normalizes the static sums; no '
                          'extra division by tokens; the summed-CE gradient scale of the original BC is kept.',
            'minimal_cache': 'two float64 sufficient statistics (S1, S2) per expert x 5 slot tensors x channel x 10 quantized '
                             'formats + n_s; the merged B / C of table_static.json cannot be split into these; no merged cost is '
                             'rescaled by a gradient ratio'},
        'semantics': {'forward': 'B, C, D: full q0 model on the calibration windows, q0\'s own dynamic expert ids and routing weights',
                      'backward': 'as the original proxy refresh: q0 routes constant, no router-parameter or top-k derivative, '
                                  'identity STE through activation QDQ, no optimizer, layer-wise recomputation through every expert',
                      'selective': 'D only drops tap / retain_grad, the weight-sensitivity matmuls, the gradient-statistics '
                                   'aggregation and the cost rebuild of non-target projections; the full backward path (every '
                                   'expert, no detach) is kept, so input gradients are unchanged (checked)',
                      'coverage': 'a searched projection with no q0 calibration token keeps its static row (recorded); the '
                                  'searched set never grows'},
        'controls': {
            '1_cache': 'static rebuild with the cache: table equals runs/olmoe_proxy_refresh_20261022/table_static.json (max '
                       f'relative difference <= {TOL_REL}; expected 0), q0 reproduced exactly; the sensitivity-only formula with the '
                       f'static gradients and the cached statistics reproduces the static C (max relative difference <= {TOL_REL}) and '
                       'searching that reconstructed table gives q0; gate / up x ranges identical for every expert (shared x_in); '
                       'static FP32 routes / CE equal the stage-1 caches; weight errors equal the original weight_error_d2.npz',
            '2_full': f'B table equals table_refresh.json (max relative difference <= {TOL_REL}; expected 0) and B\'s configuration '
                      'equals q1 exactly; any difference is investigated before going on',
            '3_forward': 'B, C, D q0 forward: per-window CE, final hidden states (sha256) and routes equal a separate '
                         'run_olmoe_dynamic_route.forward of q0 on the calibration windows and the recorded q0 calibration routes',
            '4_selected': 'D target rows equal C rows exactly (bitwise)',
            '5_unselected': 'D non-target rows equal the static table exactly (bitwise)',
            '6_input_gradient': 'per-layer input-gradient sha256 identical across every B / C / D run; recomputation equals the '
                                'forward (max abs 0)',
            '7_inputs': 'model files, ranges, q0 / q1, quantizer code and input hashes identical before / after',
            '8_validity': 'all statistics and costs finite; formats legal; resources within the limits',
            'repeats': 'the tables and configurations of the three timed repeats of each method must be identical; a difference '
                       'is located and reported; the pre-registered main configuration is repeat 1 (never the best one)',
            'ste': 'STE unit test (forward bitwise QDQ, backward identity)'},
        'search': {'solver': 'run_olmoe_search_final.make_solver("lpfirst") (LP -> rounding -> repair; corrected MILP fallback, '
                             '300 s), exact dominance pruning per table, exact resource check',
                   'scope': 'every search may reassign every projection (D restricts which costs are updated, not the search)'},
        'timing': {'rule': 'one process per run, CUDA synchronization around every timed block, no page-cache dropping; '
                           'page-cache residency (mincore) of the checkpoint shards, weight_error_d2.npz and the activation cache '
                           'recorded before every run; peak GPU / CPU memory and cache sizes recorded',
                   'categories': CATEGORIES, 'unclassified': 'run wall time minus all timed blocks',
                   'schedule': {'static': STATIC_SCHEDULE, 'refresh': REFRESH_SCHEDULE},
                   'schedule_note': 'static_cache r0 is the canonical cache build with the extra controls (not a timing repeat); '
                                    'repeats 1-3: static plain / with activation cache in the order P1 C1 C2 P2 P3 C3; refresh B / C / '
                                    'D in a Latin-square rotation B1 C1 D1 | C2 D2 B2 | D3 B3 C3; medians and ranges reported',
                   'costs': {'extra_with_cache': 'B / C / D refresh run (q0 forward -> table -> search), method time = run time '
                                                 'minus control blocks',
                             'from_zero': 'A: static plain (minus the weight-error cache write, which only refresh needs); B: static '
                                          'plain + B; C / D: static with activation cache + C / D; medians',
                             'not_counted': 'controls, timing repeats and the final evaluation are reported separately'}},
        'freeze': 'configurations, tables, cache spec, target list, code and input hashes frozen (freeze.json) before any '
                  'evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ} tokens; strict boundaries; WikiText-2 '
                               'removed (titles / contents); within-WT103 duplicates removed; near-duplicates (>= 10% 5-gram shingles vs '
                               'WikiText-2, earlier articles, accepted articles) skipped; every WikiText-103 article of every earlier '
                               'manifest excluded (runs/**/*segments*.json scanned, only this run skipped by path)',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests,
                       'eligible_before_length_and_neardup': len(elig),
                       'seeds': {k: SEEDS[k] for k in ('order', 'start')},
                       'conditions': 'FP32, A, B, C, D; original checkpoint router; each its own dynamic route; no FP32 route feedback; '
                                     'conditions with identical frozen configurations share one evaluation run (the forward is '
                                     'bitwise deterministic)',
                       'primary': 'Delta_low = NLL(D) - NLL(A); 95% CI upper < 0: D better than static BC confirmed; lower > 0: D '
                                  'worse confirmed; otherwise direction not confirmed',
                       'keep': f'Delta_keep = NLL(D) - NLL(B); non-inferiority margin +{NI_MARGIN} nat/token: 95% CI upper < '
                               f'+{NI_MARGIN} -> D not worse than full refresh by more than the margin; lower > +{NI_MARGIN} -> D worse '
                               'than the margin; otherwise insufficient evidence (never "equally good"); no extra articles',
                       'secondary_exploratory': 'C - A, C - B, D - C (and B - A as context)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL); per-article and '
                                    'per-window results saved',
                       'descriptive': 'format changes, resources, search objectives (own and cross tables), routed counts, actual-route '
                                      'BOPs including the unsearched experts; never used to reselect a configuration',
                       'interpretation': 'if no improvement: (a) C - A not confirmed -> sensitivity-only lost the gain; (b) C - A '
                                         'confirmed but D - A not -> low-bit selection lost it; (c) quality kept but the necessary '
                                         'backward / cache cost limits the time saving (from the timing medians)'},
        'not_claimed': 'other q0, calibration, resource points, models, native kernels; all low-bit refresh methods',
        'lowbit': {'file': 'lowbit_nodes.json', 'sha256': sha(out / 'lowbit_nodes.json'), 'count': len(low), 'expected': 1161,
                   'gate_up_x_ranges_identical': shared_ok,
                   'by_proj': dict(Counter(n.split('.')[2] for n in low)), 'by_layer': dict(Counter(int(n.split('.')[0][1:]) for n in low))},
        'tolerances': {'relative': TOL_REL, 'bitwise': 'forward, routes, input gradients, D vs C / static rows, repeats'},
        'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code': code_hashes(), 'environment': dr.env_info()}
    if len(low) != 1161 or not shared_ok:
        raise SystemExit(f'low-bit count {len(low)} or shared x ranges differ: investigate the inputs first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE sensitivity-only 與 low-bit selective sensitivity refresh', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          f'- 資源上限：直接讀原 plan（weight bytes ≤ {p["budget"]["max_bytes"]:.0f}、BOPs ≤ {p["budget"]["max_bops"]:.0f}），BOPs 用 FP32 calibration routed rows。',
          f'- 固定：{len(names)} 個搜尋投影、GRID11×GRID11、原 10×128 calibration、固定 ranges、修正後量化器、原 checkpoint 與 router；不載入 router FT。',
          '- A：原 q0；B：完整 refresh（須重現 table_refresh 與 q1）；C：全部節點只更新 sensitivity；'
          f'D：只更新 q0 中 weight 或 activation ≤4 bit 的 {len(low)} 個投影（所有候選格式），其餘沿用 static 表。',
          '- weight 項：S_new[c]·D_static[c,f]，D_static 沿用原快取（不算省時）；C 的 weight 項與 B 完全相同。',
          '- activation 項：快取 static 的 n_s、Σerr、Σerr²（float64，逐 channel、逐格式；gate／up 共用 x_in），'
          'A_new = Σ (G2_new·var_static + GS_new·μ_static²)，以原 BC 運算順序 G2/n_s·(S2 − n_s μ²) + GS·μ² 計算。',
          '- B、C、D 都跑完整 q0 forward 與完整逐層 backward；D 只省去非目標投影的 tap／梯度保留、weight sensitivity、統計與成本重建。',
          f'- 控制：static 表與 q0 重現（≤{TOL_REL} 相對差，預期 0）、B 重現 table_refresh 與 q1、forward／路由／input gradient 逐位元、'
          'D 目標列 = C、D 非目標列 = static、重複間配置一致（主要配置固定為第 1 次）。',
          '- 計時：每次一個 process；static plain／cache 各 3 次（P1 C1 C2 P2 P3 C3，前置 r0 為正式快取與控制）；'
          'B／C／D 各 3 次（B1 C1 D1｜C2 D2 B2｜D3 B3 C3）；記錄 page-cache 駐留；報告中位數與範圍。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4 窗口（掃描全部歷史 manifest 排除 {len(used)} 篇）；FP32、A、B、C、D；seed 22401／22402／22411。',
          f'- 主要：Δ_low = NLL(D) − NLL(A)；上界 < 0 → D 優於 static，下界 > 0 → D 變差，否則未確認。',
          f'- 非劣：Δ_keep = NLL(D) − NLL(B)；上界 < +{NI_MARGIN} 才算不差超過界線；否則證據不足（不寫成一樣好），不追加文章。',
          '- 次要（探索）：C−A、C−B、D−C。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    log(f'plan written; low-bit {len(low)}/{len(names)}; previous WT103 {len(used)}; eligible {len(elig)}')


def controls(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'controls.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    fm = read(PR / 'configs/q0.json')
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    toks = np.load(CALIB)
    rd = dr.forward(toks, fm, dq, ranges, record=True, log=lambda m: None, label='q0 calibration', chunk=len(toks))
    prr = np.load(PR / 'q0_calibration_routes.npz')
    low = read(out / 'lowbit_nodes.json')
    ctl = {'q0_forward_ce': rd['ce'], 'q0_forward_hidden_sha': rd['hidden_sha'], 'q0_forward_routes_hash': dr.route_hash(rd['routes']),
           'q0_forward_routes_equal_previous_refresh': bool(np.array_equal(rd['routes']['ids'], prr['ids']) and np.array_equal(rd['routes']['weights'], prr['weights'])),
           'ste_unit_test': pr.ste_unit_test(dq, ranges), 'lowbit_recount': len(lowbit_nodes(fm)), 'lowbit_file_count': low['count'],
           'lowbit_sha_matches_plan': sha(out / 'lowbit_nodes.json') == read(out / 'plan.json')['lowbit']['sha256'],
           'page_cache_residency': resident_fraction(watch_files(out)), 'seconds': time.perf_counter() - t0}
    ctl['pass'] = bool(ctl['q0_forward_routes_equal_previous_refresh'] and ctl['ste_unit_test']['pass'] and ctl['lowbit_recount'] == low['count']
                       and ctl['lowbit_sha_matches_plan'] and rd['finite_ce'])
    write(out / 'controls.json', ctl)
    log(json.dumps({k: ctl[k] for k in ('q0_forward_routes_equal_previous_refresh', 'lowbit_recount', 'pass')}))


def static(out, log, variant, rep):
    import torch
    from run_olmoe_search_final import dev_quant
    from run_olmoe_fixed_route import load_routes
    from run_native_bc_pipeline import QUANT
    import olmoe_common as oc
    dr.gpu_setup()
    tag = f'static_{variant}_r{rep}'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    pl = read(out / 'plan.json')
    residency = resident_fraction(watch_files(out))
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    collect = None
    if variant == 'cache':
        with tm.block('cache_build'):
            experts = sorted({'.'.join(n.split('.')[:2]) for n in names}, key=lambda s: (int(s.split('.')[0][1:]), int(s.split('.')[1][1:])))
            collect = {'row': {e: i for i, e in enumerate(experts)}, 'n': {}, 'data': np.zeros((len(experts), 2, 10, NCOL), np.float64)}
    r = build('static', dq, ranges, names, tm, log, collect=collect, extra=(rep == 0))
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'state': 'FP32 reference'}
    fm, srch = search_table(t, (names, nodes, elems, flops), pl['budget'], tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm)
    with tm.block('cache_write_d2'):
        np.savez(rd / 'weight_error_d2.npz', **{f'{n}|{f}': v for n, d in r['d2'].items() for f, v in d.items()})
    if collect is not None:
        with tm.block('cache_write'):
            np.save(rd / 'act_err_cache.npy', collect['data'])
            write(rd / 'act_err_cache_index.json', {'experts': list(collect['row']), 'row': collect['row'], 'n': collect['n'],
                                                    'formats': list(QUANT), 'columns': COLS,
                                                    'stats': ['sum_err', 'sum_err_sq'], 'shape': list(collect['data'].shape), 'dtype': 'float64',
                                                    'state': 'FP32 reference (static calibration routes)', 'ranges_sha256': sha(RANGES)})
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'search': srch,
                              'files_bytes': {f.name: f.stat().st_size for f in rd.iterdir() if f.is_file()},
                              'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'], 'routes_hash': dr.route_hash(r['routes']),
                              'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'recompute_max_abs': max(r['recompute_max_abs_per_layer']),
                              'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']), 'code': code_hashes()})
    if rep == 0:
        _, ids_s1, ws_s1 = load_routes(S1, 'calibration', oc.config(), 'cpu')
        s1m = read(S1 / 'routes_manifest.json')['calibration']
        hmeta = read(MP / 'table_meta.json')['experts']
        rel = max(abs(r['recon_C'][n][f] - r['C'][n][f]) / max(abs(r['C'][n][f]), 1e-300) for n in names for f in r['C'][n] if f != 'fp32')
        write(rd / 'table_recon.json', {'B': r['B'], 'C': r['recon_C'], 'note': 'C rebuilt with the sensitivity-only formula from the static '
                                                                            'gradients and the cached statistics (control 1)'})
        np.savez(rd / 'static_calibration_routes.npz', **r['routes'])
        rec['extra_controls'] = {
            'routes_equal_stage1_cache': bool(np.array_equal(r['routes']['ids'], ids_s1) and np.array_equal(r['routes']['weights'], ws_s1)),
            'ce_vs_stage1_dynamic_max_abs': float(np.max(np.abs(np.array(r['ce_forward']) - np.array(s1m['summed_ce_dynamic'])))),
            'rows_equal_historical': all(r['meta'][k]['routed_rows'] == hmeta[k]['routed_rows'] for k in r['meta']),
            'observed_ranges_equal_fixed': all(r['observed_ranges'][n] == ranges[n] for n in names),
            'all_searched_observed': all(v['routed_rows'] > 0 for v in r['meta'].values()),
            'cache_n_equal_meta': all(collect['n'][k] == r['meta'][k]['routed_rows'] for k in r['meta']),
            'recon_C_max_rel': rel}
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'total_s': rec['total_s'], 'objective': srch['objective'], 'path': srch['path']} |
                   ({'extra': rec['extra_controls']} if rep == 0 else {})))


def refresh(out, log, variant, rep):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = f'refresh_{variant}_r{rep}'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    pl = read(out / 'plan.json')
    cpath, ipath = cache_paths(out)
    residency = resident_fraction(watch_files(out))
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
        fm = read(PR / 'configs/q0.json')
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    with tm.block('cache_read_d2'):
        d2z = np.load(PR / 'weight_error_d2.npz')
    cache = cidx = static_tab = targets = None
    if variant in ('C', 'D'):
        with tm.block('cache_read'):
            cache = np.load(cpath, mmap_mode='r')
            cidx = read(ipath)
    if variant == 'D':
        with tm.block('input_read'):
            static_tab = read(PR / 'table_static.json')
            targets = read(out / 'lowbit_nodes.json')['nodes']
    r = build(variant, dq, ranges, names, tm, log, fm=fm, d2z=d2z, cache=cache, cidx=cidx, static_tab=static_tab, targets=targets)
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'],
         'updated_projections': len(r['updated_projections']), 'state': f'q0 working state, condition {variant}'}
    fm_new, srch = search_table(t, (names, nodes, elems, flops), pl['budget'], tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm_new)
    with tm.block('controls'):
        np.savez(rd / 'q0_calibration_routes.npz', **r['routes'])
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'search': srch,
                              'updated_projections': r['updated_projections'] if variant == 'D' else len(r['updated_projections']),
                              'fallback_static_rows': r['fallback_static_rows'], 'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'],
                              'routes_hash': dr.route_hash(r['routes']), 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'],
                              'recompute_max_abs': max(r['recompute_max_abs_per_layer']), 'grads_finite': r['grads_finite'],
                              'nonfinite': len(r['nonfinite']), 'q0_routed_rows_total': int(sum(v['routed_rows'] for v in r['meta'].values())),
                              'cache_bytes': {'act_err_cache.npy': cpath.stat().st_size, 'weight_error_d2.npz': (PR / 'weight_error_d2.npz').stat().st_size},
                              'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'total_s': rec['total_s'], 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path']}))


def timing(out, log, phase):
    sched = read(out / 'plan.json')['timing']['schedule'][phase]
    for v, r in sched:
        rd = out / 'work/runs' / f'{phase}_{v}_r{r}'
        if (rd / 'run.json').exists():
            log(f'{phase} {v} r{r}: done already, kept')
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', phase, '--variant', v, '--rep', str(r), '--out-dir', str(out)]
        t = time.time()
        p = subprocess.run(cmd, cwd=HERE)
        log(f'{phase} {v} r{r}: exit {p.returncode}, process wall {time.time() - t:.1f}s')
        if p.returncode:
            raise SystemExit(f'{phase} {v} r{r} failed')


def max_rel(a, b, names):
    d = 0.0
    for n in names:
        for f in a[n]:
            x, y = a[n][f], b[n][f]
            if x != y:
                d = max(d, abs(x - y) / max(abs(y), 1e-300))
    return d


def d2_equal(p, ref):
    za, zb = np.load(p), np.load(ref)
    return sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in zb.files)


def check(out, log):
    t0 = time.perf_counter()
    if (out / 'check.json').exists():
        raise SystemExit('exists')
    names, nodes, elems, flops, rows = pr.problem_static()
    pl = read(out / 'plan.json')
    W = out / 'work/runs'
    runs = {p.name: read(p / 'run.json') for p in sorted(W.iterdir()) if (p / 'run.json').exists()}
    exp = [f'static_{v}_r{r}' for v, r in STATIC_SCHEDULE] + [f'refresh_{v}_r{r}' for v, r in REFRESH_SCHEDULE]
    tabs = {k: read(W / k / 'table.json') for k in exp}
    cfgs = {k: read(W / k / 'config.json') for k in exp}
    st0, q0, q1 = read(PR / 'table_static.json'), read(PR / 'configs/q0.json'), read(PR / 'configs/q1.json')
    rf = read(PR / 'table_refresh.json')
    ctl = read(out / 'controls.json')
    res = {'missing_runs': [k for k in exp if k not in runs]}
    # 1. cache / static
    r0 = runs['static_cache_r0']
    recon = read(W / 'static_cache_r0/table_recon.json')

    class _T:
        t = defaultdict(float)

        def block(self, k):
            class _B:
                def __enter__(s): pass

                def __exit__(s, *a): pass
            return _B()
    fm_recon, s_recon = search_table({'B': recon['B'], 'C': recon['C']}, (names, nodes, elems, flops), pl['budget'], _T(), out / 'logs', 'check_recon')
    c1 = {'static_B_max_rel_vs_original': max_rel(tabs['static_cache_r0']['B'], st0['B'], names),
          'static_C_max_rel_vs_original': max_rel(tabs['static_cache_r0']['C'], st0['C'], names),
          'q0_reproduced': cfgs['static_cache_r0'] == q0, 'weight_error_equal_original': d2_equal(W / 'static_cache_r0/weight_error_d2.npz', PR / 'weight_error_d2.npz'),
          'recon_C_max_rel': r0['extra_controls']['recon_C_max_rel'], 'recon_search_gives_q0': fm_recon == q0, 'recon_objective': s_recon['objective'],
          **{k: v for k, v in r0['extra_controls'].items() if k != 'recon_C_max_rel'}}
    c1['pass'] = bool(c1['static_B_max_rel_vs_original'] <= TOL_REL and c1['static_C_max_rel_vs_original'] <= TOL_REL and c1['q0_reproduced']
                      and c1['weight_error_equal_original'] and c1['recon_C_max_rel'] <= TOL_REL and c1['recon_search_gives_q0']
                      and c1['routes_equal_stage1_cache'] and c1['ce_vs_stage1_dynamic_max_abs'] == 0 and c1['rows_equal_historical']
                      and c1['observed_ranges_equal_fixed'] and c1['all_searched_observed'] and c1['cache_n_equal_meta'])
    res['1_cache'] = c1
    # 2. full refresh
    c2 = {'B_weight_max_rel_vs_original': max_rel(tabs['refresh_B_r1']['B'], rf['B'], names),
          'B_act_max_rel_vs_original': max_rel(tabs['refresh_B_r1']['C'], rf['C'], names), 'q1_reproduced': cfgs['refresh_B_r1'] == q1}
    c2['pass'] = bool(c2['B_weight_max_rel_vs_original'] <= TOL_REL and c2['B_act_max_rel_vs_original'] <= TOL_REL and c2['q1_reproduced'])
    res['2_full'] = c2
    # 3. forward
    rr = [k for k in exp if k.startswith('refresh_')]
    prr = dr.route_hash(dict(np.load(PR / 'q0_calibration_routes.npz')))
    c3 = {k: {'ce_equal': runs[k]['ce_forward'] == ctl['q0_forward_ce'], 'hidden_equal': runs[k]['hidden_sha'] == ctl['q0_forward_hidden_sha'],
              'routes_equal': runs[k]['routes_hash'] == ctl['q0_forward_routes_hash'] == prr} for k in rr}
    res['3_forward'] = {'runs': c3, 'pass': all(all(v.values()) for v in c3.values())}
    # 4 / 5. selective rows
    low = set(read(out / 'lowbit_nodes.json')['nodes'])
    tD, tC, tB = tabs['refresh_D_r1'], tabs['refresh_C_r1'], tabs['refresh_B_r1']
    fbD = set(runs['refresh_D_r1']['fallback_static_rows'])
    res['4_selected'] = {'rows': len(low), 'fallback': sorted(fbD),
                         'equal_C': all(tD['B'][n] == tC['B'][n] and tD['C'][n] == tC['C'][n] for n in low if n not in fbD)}
    res['4_selected']['pass'] = res['4_selected']['equal_C']
    res['5_unselected'] = {'rows': len(names) - len(low), 'equal_static': all(tD['B'][n] == st0['B'][n] and tD['C'][n] == st0['C'][n] for n in names if n not in low or n in fbD)}
    res['5_unselected']['pass'] = res['5_unselected']['equal_static']
    res['C_weight_term_equals_B'] = all(tC['B'][n] == tB['B'][n] for n in names)
    # 6. input gradients
    g0 = runs['refresh_B_r1']['input_grad_sha_per_layer']
    res['6_input_gradient'] = {'identical_all_refresh_runs': all(runs[k]['input_grad_sha_per_layer'] == g0 for k in rr),
                               'recompute_max_abs': max(runs[k]['recompute_max_abs'] for k in exp)}
    res['6_input_gradient']['pass'] = res['6_input_gradient']['identical_all_refresh_runs'] and res['6_input_gradient']['recompute_max_abs'] == 0
    # 7. inputs and code
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in model_files()}}
    codes = {k: runs[k]['code'] for k in exp}
    res['7_inputs'] = {'input_hashes_unchanged': after == read(out / 'input_hashes_before.json'),
                       'same_code_all_runs': all(c == codes[exp[0]] for c in codes.values()),
                       'quantizer_code_equal_plan': all(codes[exp[0]][f] == pl['code'][f] for f in dr.SCRIPTS)}
    res['7_inputs']['pass'] = res['7_inputs']['input_hashes_unchanged'] and res['7_inputs']['quantizer_code_equal_plan']
    # 8. validity
    res['8_validity'] = {k: {'finite': runs[k]['grads_finite'] and runs[k]['nonfinite'] == 0, 'legal': runs[k]['search']['all_formats_legal'],
                             'within_budget': runs[k]['search']['within_budget']} for k in exp}
    res['8_validity']['pass'] = all(all(v.values()) for k, v in res['8_validity'].items() if k != 'pass')
    # repeats
    reps = {}
    for v in ('plain', 'cache'):
        ks = [f'static_{v}_r{r}' for vv, r in STATIC_SCHEDULE if vv == v]
        reps[f'static_{v}'] = {'tables_identical': all(tabs[k] == tabs['static_cache_r0'] for k in ks),
                               'configs_identical': all(cfgs[k] == q0 for k in ks)}
    ch = sha(W / 'static_cache_r0/act_err_cache.npy')
    reps['static_cache']['act_cache_identical'] = all(sha(W / f'static_cache_r{r}/act_err_cache.npy') == ch for r in (1, 2, 3))
    reps['weight_error_d2_identical'] = all(d2_equal(W / f'static_{v}_r{r}/weight_error_d2.npz', PR / 'weight_error_d2.npz') for v, r in STATIC_SCHEDULE if r > 0)
    for v in 'BCD':
        ks = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        reps[f'refresh_{v}'] = {'tables_identical': all(tabs[k] == tabs[ks[0]] for k in ks), 'configs_identical': all(cfgs[k] == cfgs[ks[0]] for k in ks)}
    res['repeats'] = reps
    res['repeats_pass'] = all(all(v.values()) if isinstance(v, dict) else v for v in reps.values())
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['fallback_static_rows'] = {k: runs[k]['fallback_static_rows'] for k in rr}
    res['pass'] = bool(all(res[k]['pass'] for k in ('1_cache', '2_full', '3_forward', '4_selected', '5_unselected', '6_input_gradient', '7_inputs', '8_validity'))
                       and res['repeats_pass'] and res['ste'] and res['controls_stage_pass'] and not res['missing_runs'] and res['C_weight_term_equals_B'])
    res['seconds'] = time.perf_counter() - t0
    write(out / 'check.json', res)
    log(json.dumps({k: (v['pass'] if isinstance(v, dict) and 'pass' in v else v) for k, v in res.items() if k not in ('fallback_static_rows', 'seconds')}))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'check.json')['pass']:
        raise SystemExit('controls failed: not freezing')
    W = out / 'work/runs'
    (out / 'configs').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    src = {'A': PR / 'configs/q0.json', 'B': W / 'refresh_B_r1/config.json', 'C': W / 'refresh_C_r1/config.json', 'D': W / 'refresh_D_r1/config.json'}
    tsrc = {'static': PR / 'table_static.json', 'B': W / 'refresh_B_r1/table.json', 'C': W / 'refresh_C_r1/table.json', 'D': W / 'refresh_D_r1/table.json'}
    for c, p in src.items():
        write(out / f'configs/{c}.json', read(p))
    for c, p in tsrc.items():
        write(out / f'tables/{c}.json', read(p))
    cfg = {c: read(out / f'configs/{c}.json') for c in src}
    names = sorted(cfg['A'])
    q1 = read(PR / 'configs/q1.json')
    from mp_auto import config_hash
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    cpath, ipath = cache_paths(out)
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'repeat 1 of each method (pre-registered)',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'), 'config_hash': config_hash(cfg[c])}
                       for c in cfg},
           'A_equals_original_q0': cfg['A'] == read(PR / 'configs/q0.json'), 'B_equals_q1': cfg['B'] == q1,
           'pairwise_differences': {f'{a}-{b}': diff(cfg[a], cfg[b]) for a, b in (('B', 'A'), ('C', 'A'), ('D', 'A'), ('C', 'B'), ('D', 'B'), ('D', 'C'))},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))}
                                for c in 'BCD'},
           'tables': {c: sha(out / f'tables/{c}.json') for c in tsrc},
           'cache': {'act_err_cache.npy': sha(cpath), 'act_err_cache_index.json': sha(ipath), 'bytes': cpath.stat().st_size},
           'lowbit_nodes_sha256': sha(out / 'lowbit_nodes.json'), 'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'),
           'ranges_sha256': sha(RANGES), 'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('A_equals_original_q0', 'B_equals_q1', 'pairwise_differences')}))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    used, _ = used_wt103_all(out)
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
    merge_timing(out, 'data', {'seconds': time.perf_counter() - t0, 'peak_cpu_gb': peak_cpu_gb()})
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
    old_t, old = np.load(PR / 'tokens.npy')[:CTRL], read(PR / 'eval.json')['ce']
    pc = {}
    for c, fm in (('fp32', None), ('q0', read(PR / 'configs/q0.json')), ('q1', read(PR / 'configs/q1.json'))):
        r = dr.forward(old_t, fm, dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs, shared = {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    by_hash = {}
    for c in 'ABCD':
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
           'finite': all(r['finite_ce'] for r in runs.values()),
           'audit': {c: runs[c]['audit'] for c in 'ABCD'},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in 'ABCD'}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_D_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_D_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_D_first_chunk']['routes_equal'] and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl,
                              'seconds': {c: runs[c]['seconds'] for c in runs if c not in shared}, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                              'environment': dr.env_info()})
    merge_timing(out, 'eval', {**{f'{c}_s': runs[c]['seconds'] for c in runs if c not in shared}, 'rerun_s': r2['seconds'],
                               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': peak_cpu_gb()})
    log(json.dumps({'controls_pass': ctl['pass'], 'program_control': pc, 'shared': shared}))


def timing_summary(out):
    W = out / 'work/runs'
    runs = {p.name: read(p / 'run.json') for p in sorted(W.iterdir()) if (p / 'run.json').exists()}
    jl = [json.loads(x) for x in (out / 'timing.jsonl').read_text().splitlines()]
    proc = {f'{j["stage"]}_{j["variant"]}_r{j["rep"]}': j['seconds'] for j in jl if j.get('variant')}
    groups = {'static_plain': [f'static_plain_r{r}' for r in (1, 2, 3)], 'static_cache': [f'static_cache_r{r}' for r in (1, 2, 3)],
              **{f'refresh_{v}': [f'refresh_{v}_r{r}' for r in (1, 2, 3)] for v in 'BCD'}}
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for gname, ks in groups.items():
        s[gname] = {'method_s': med([runs[k]['method_s'] for k in ks]), 'run_total_s': med([runs[k]['total_s'] for k in ks]),
                    'process_wall_s': med([proc[k] for k in ks]),
                    'categories': {c: med([runs[k]['categories'][c] for k in ks]) for c in list(CATEGORIES) + ['unclassified']},
                    'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks]) for t in sorted({t for k in ks for t in runs[k]['timers']})},
                    'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks),
                    'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks}}
    a_cost = med([runs[k]['method_s'] - runs[k]['timers'].get('cache_write_d2', 0.0) for k in groups['static_plain']])['median']
    sp, sc = s['static_plain']['method_s']['median'], s['static_cache']['method_s']['median']
    ex = {v: s[f'refresh_{v}']['method_s']['median'] for v in 'BCD'}
    s['from_zero_median_s'] = {'A': a_cost, 'B': sp + ex['B'], 'C': sc + ex['C'], 'D': sc + ex['D']}
    s['extra_with_cache_median_s'] = ex
    s['cache_build_extra_median_s'] = sc - sp
    s['canonical_cache_run'] = {k: runs['static_cache_r0'][k] for k in ('total_s', 'method_s', 'categories', 'peak_gpu_gb', 'peak_cpu_gb', 'page_cache_residency_before')}
    s['cache_bytes'] = {'act_err_cache.npy': cache_paths(out)[0].stat().st_size, 'weight_error_d2.npz': (PR / 'weight_error_d2.npz').stat().st_size}
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
    D = {'D_minus_A': A['D'] - A['A'], 'D_minus_B': A['D'] - A['B'], 'C_minus_A': A['C'] - A['A'], 'C_minus_B': A['C'] - A['B'],
         'D_minus_C': A['D'] - A['C'], 'B_minus_A': A['B'] - A['A']}
    res = {'n_articles': N, 'primary_delta_low': st(D['D_minus_A']), 'keep_delta': st(D['D_minus_B']),
           'secondary_exploratory': {k: st(D[k]) for k in ('C_minus_A', 'C_minus_B', 'D_minus_C', 'B_minus_A')},
           'damage': {c: st(A[c] - A['fp32']) for c in 'ABCD'}}
    lo, hi = res['primary_delta_low']['ci95']
    res['primary_decision'] = 'D_better_than_static_confirmed' if hi < 0 else 'D_worse_than_static_confirmed' if lo > 0 else 'direction_not_confirmed'
    lo, hi = res['keep_delta']['ci95']
    res['keep_decision'] = (f'D_noninferior_to_full_refresh_within_{NI_MARGIN}' if hi < NI_MARGIN else
                            f'D_worse_than_full_refresh_by_more_than_{NI_MARGIN}' if lo > NI_MARGIN else 'insufficient_evidence')
    for k, v in res['secondary_exploratory'].items():
        v['direction'] = 'negative_confirmed' if v['ci95'][1] < 0 else 'positive_confirmed' if v['ci95'][0] > 0 else 'not_confirmed'
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    # configurations, objectives, resources
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in 'ABCD'}
    tabs = {c: read(out / f'tables/{c}.json') for c in ('static', 'B', 'C', 'D')}
    names = sorted(cfg['A'])
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    res['objectives'] = {t: {c: obj(tabs[t], cfg[c]) for c in 'ABCD'} for t in tabs}
    W = out / 'work/runs'
    srch = {'A': read(W / 'static_cache_r0/run.json')['search'], **{c: read(W / f'refresh_{c}_r1/run.json')['search'] for c in 'BCD'}}
    res['search'] = {c: {k: srch[c][k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
                     for c in 'ABCD'}
    res['format_changes'] = {'pairwise': fz['pairwise_differences'], 'transitions_vs_A': fz['transitions_vs_A']}
    low = set(read(out / 'lowbit_nodes.json')['nodes'])
    res['format_changes']['changed_vs_A_in_lowbit_set'] = {c: {'lowbit': sum(cfg[c][n] != cfg['A'][n] for n in low),
                                                               'other': sum(cfg[c][n] != cfg['A'][n] for n in names if n not in low)} for c in 'BCD'}
    # routing cost
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
                   'routed_rows_to_lowbit_experts': int(sum(counts[c][l, e] for l, e in {(int(n.split('.')[0][1:]), int(n.split('.')[1][1:])) for n in low})),
                   'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
        if c != 'A':
            desc[c]['set_change_vs_A'] = float((np.sort(np.load(out / f'routes_{shared.get(c, c)}.npz')['ids'], -1) !=
                                                np.sort(np.load(out / 'routes_A.npz')['ids'], -1)).any(-1).mean())
    cost['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l, e in unsearched))
    res['routing_cost'] = cost
    res['routing_descriptive'] = desc | {'set_change_vs_fp32': ev['controls']['routes_differ_from_fp32']}
    res['timing'] = timing_summary(out)
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in model_files()}}
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
    merge_timing(out, 'analysis', {'seconds': time.perf_counter() - t0})
    write(out / 'analysis.json', res)
    log(json.dumps({'primary': res['primary_delta_low'], 'decision': res['primary_decision'], 'keep': res['keep_delta'], 'keep_decision': res['keep_decision']}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'controls', 'static', 'refresh', 'timing', 'check', 'freeze', 'data', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant')
    ap.add_argument('--rep', type=int)
    ap.add_argument('--phase', choices=['static', 'refresh'])
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    if a.stage == 'static':
        static(a.out_dir, log, a.variant, a.rep)
    elif a.stage == 'refresh':
        refresh(a.out_dir, log, a.variant, a.rep)
    elif a.stage == 'timing':
        timing(a.out_dir, log, a.phase)
    else:
        {'eval': eval_}.get(a.stage, globals().get(a.stage))(a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'variant': a.variant, 'rep': a.rep, 'phase': a.phase, 'seconds': time.time() - T_PROC,
                            'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
