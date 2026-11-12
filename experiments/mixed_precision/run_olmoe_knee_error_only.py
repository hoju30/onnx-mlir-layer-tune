#!/usr/bin/env python3
"""OLMoE normal knee: activation error-only refresh (E), next to the frozen A / B / C (and D4 / D5 for diagnostics).
Pre-registered in plan.json / plan.md.

E keeps the FP32 static sensitivities and refreshes only the activation quantization errors in q0_knee's working state:
  weight rows = the static table (A) unchanged;
  A_E(f) = sum_slots,c [G2_static,c var_new,c(f) + GS_static,c mu_new,c(f)^2],
  mu_new = sum err_new / n_q0, var_new = sum err_new^2 / n_q0 - mu_new^2, err_new = Q_f(a_q0) - a_q0 at the same taps as B
  (gate / up share the pre-QDQ expert input; down x = SiLU(g) u; y = pre-QDQ linear outputs), evaluated in the BC op order
  G2 / n_q0 (S2 - n_q0 mu^2) + GS mu^2.
G2_static / GS_static come from a static-build cache saved here (the FP32 build needs a backward; the E update does not).
The E update is forward-only: torch.inference_mode, an observation path without retain_grad, autograd calls counted.

Stages (one --out-dir): plan, static, controls, refresh (--variant B|C|E --rep k), e2e, timing (--phase refresh|e2e),
check, freeze, data, eval, analyze.
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
KA = HERE / 'runs/olmoe_knee_all_sensitivity_20261022'
K5 = HERE / 'runs/olmoe_knee_low5_sensitivity_20261022'
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE, D2, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.D2, ks.ACT_CACHE, ks.ACT_INDEX
Q0K, KNEE = KS / 'knee/q0_knee.json', KS / 'knee/knee.json'
C_STATS = KA / 'work/runs/refresh_C_r1'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, NCMP = 10000, 0.0005, 1e-12, 3
CI_CORR = [100 * 0.05 / (2 * NCMP), 100 * (1 - 0.05 / (2 * NCMP))]      # Bonferroni over the 3 primary comparisons: 98.33%
SEEDS = {'order': 22801, 'start': 22802, 'bootstrap': 22811}
VARIANTS = ('B', 'C', 'E')
REFRESH_SCHEDULE = [('B', 1), ('C', 1), ('E', 1), ('C', 2), ('E', 2), ('B', 2), ('E', 3), ('B', 3), ('C', 3)]
FROZEN = {'A': KS / 'configs/A.json', 'B': KS / 'configs/B.json', 'C': KA / 'configs/C.json', 'D4': KS / 'configs/D.json', 'D5': K5 / 'configs/D5.json'}
FROZEN_TABLES = {'B': KS / 'tables/B.json', 'C': KA / 'tables/C.json', 'D4': KS / 'tables/D.json', 'D5': K5 / 'tables/D5.json'}
CATEGORIES = ks.CATEGORIES


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_error_only.py', 'run_olmoe_knee_selective_sensitivity.py', 'run_olmoe_knee_all_sensitivity.py',
                                                  'run_olmoe_knee_low5_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def gc_paths(out):
    d = out / 'work/runs/static_gc'
    return d / 'grad_stats_cache.npy', d / 'grad_stats_cache_index.json'


def watch_files(out):
    return ks.watch_files() + [gc_paths(out)[0]]


def history():
    fs, fa, f5 = read(KS / 'freeze.json'), read(KA / 'freeze.json'), read(K5 / 'freeze.json')
    res = {'knee_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in fs['configs'].items()},
           'knee_round_tables': {n: sha(KS / 'tables' / n) == h for n, h in fs['tables'].items()},
           'all_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in fa['configs'].items()},
           'all_round_table_C': sha(KA / 'tables/C.json') == fa['tables']['C.json'],
           'low5_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in f5['configs'].items()},
           'low5_round_table_D5': sha(K5 / 'tables/D5.json') == f5['tables']['D5.json'],
           'knee': sha(KNEE) == fs['knee_sha256'], 'checks_pass': read(KS / 'check.json')['pass'] and read(KA / 'check.json')['pass'] and read(K5 / 'check.json')['pass'],
           'code_unchanged': {k: sha(HERE / k) == f5['code'][k] for k in ('run_olmoe_knee_selective_sensitivity.py', 'run_olmoe_selective_sensitivity.py',
                                                                            'run_olmoe_knee_all_sensitivity.py', 'run_olmoe_knee_low5_sensitivity.py')},
           'q0_knee_equals_A': read(Q0K) == read(FROZEN['A']), 'c_stats_present': (C_STATS / 'target_stats.npz').exists()}
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
    a_err, a_asm, c_rd = timers.get('act_error', 0.0), timers.get('act_assembly', 0.0), timers.get('cache_read', 0.0)
    if variant == 'C':
        ok = a_err == 0 and a_asm > 0 and c_rd > 0
    elif variant == 'B':
        ok = a_err > 0 and a_asm == 0 and c_rd == 0
    else:                                        # E: new activation errors, the static gradient-statistics cache read
        ok = a_err > 0 and a_asm > 0 and c_rd > 0 and all(timers.get(k, 0.0) == 0 for k in ('backward', 'weight_sens', 'act_grad_stats', 'cache_read_d2'))
    return {'act_error_s': a_err, 'act_assembly_s': a_asm, 'cache_read_s': c_rd, 'backward_s': timers.get('backward', 0.0),
            'weight_sens_s': timers.get('weight_sens', 0.0), 'act_grad_stats_s': timers.get('act_grad_stats', 0.0), 'as_expected': ok}


class AutogradGuard:
    """counts torch.autograd.backward / torch.autograd.grad / Tensor.backward calls while active."""
    def __enter__(self):
        import torch
        self.calls = Counter()
        self._orig = (torch.autograd.backward, torch.autograd.grad, torch.Tensor.backward)
        g = self

        def wrap(name, fn):
            def w(*a, **k):
                g.calls[name] += 1
                return fn(*a, **k)
            return w
        torch.autograd.backward = wrap('autograd.backward', self._orig[0])
        torch.autograd.grad = wrap('autograd.grad', self._orig[1])
        torch.Tensor.backward = wrap('Tensor.backward', self._orig[2])
        return self

    def __exit__(self, *a):
        import torch
        torch.autograd.backward, torch.autograd.grad, torch.Tensor.backward = self._orig


# --------------------------------------------------------------------------- #
# static build with the gradient-statistics cache (run_olmoe_knee_selective_sensitivity.build, static path, same op order)
# --------------------------------------------------------------------------- #
def build_static(dq, ranges, names, tm, log, gcollect=None):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    TAPS, COLOFF, COL_SRC, NCOL = ss.TAPS, ss.COLOFF, ss.COL_SRC, ss.NCOL
    with tm.block('setup'):
        cfg, ck = oc.config(), oc.Checkpoint()
        toks_np = np.load(CALIB)
        toks = torch.from_numpy(toks_np).to('cuda')
        nW = len(toks_np)
        cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    with tm.block('ckpt_load'):
        g = ck.global_('cuda', torch.float32)
    want_all = defaultdict(dict)
    for n in names:
        l, e, p = n.split('.')
        want_all[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    STEQ = pr.ste_fn()
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = ss.load_layer(ck, l, None, dq, ranges, tm)
            with tm.block('forward_routing'):
                h, idx, rw = pr.layer_tap(wq, h, cfg, cos, sin, None, None, pr.act_map(hk))
            routes.append((idx, rw)); hs.append(h)
            del wq, hk
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
    with tm.block('backward'):
        hT = hs[-1].clone().requires_grad_(True)
        ce_bwd = head_ce(g, hT, toks, cfg)
        ce_bwd.sum().backward()
        gn = hT.grad
    with tm.block('controls'):
        if [float(x) for x in ce_bwd.detach().cpu().numpy()] != ce_fwd:
            raise RuntimeError('head CE differs between forward and backward')
    B_, C_, meta, d2_new = {}, {}, {}, {}
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = ss.load_layer(ck, l, None, dq, ranges, tm)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = ss.layer_tap_sel(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(hk, STEQ), want, cap)
            o.backward(gn)
            gn = hi.grad
        with tm.block('controls'):
            gfin &= bool(torch.isfinite(gn).all())
        with torch.no_grad():
            for e in sorted(want):
                node = f'L{l}.E{e}'
                c = cap[e]
                tok = c['tok']
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                meta[node] = {'routed_rows': int(n), 'windows': len(wins)}
                gst, errm = {}, {}
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
                    with tm.block('weight_error'):
                        W = wq[p][e]
                        d2_new[name] = {f: ((dq.weight(W, f) - W).double() ** 2).sum(1).cpu().numpy() for f in QUANT}
                    dd = d2_new[name]
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
                    rng = ranges[name]
                    with tm.block('act_error'):
                        for s, k_ in (('x', kx), ('y', ky)):
                            key = (k_, tuple(rng[s]))
                            if key in errm:
                                continue
                            t = c[k_].detach()
                            G2, GS = gst[k_]
                            vals = []
                            for f in QUANT:
                                err = (dq.act(t, f, rng[s]) - t).double()
                                s1 = err.sum(0)
                                s2 = (err ** 2).sum(0)
                                mu = s1 / n
                                vals.append(float((G2 / n * (s2 - n * mu ** 2)).sum() + (GS * mu ** 2).sum()))
                            errm[key] = vals
                        vx, vy = errm[(kx, tuple(rng['x']))], errm[(ky, tuple(rng['y']))]
                        C_[name] = {'fp32': 0.0}
                        for i, f in enumerate(QUANT):
                            cc = 0.0
                            cc += vx[i]
                            cc += vy[i]
                            C_[name][f] = cc
                if gcollect is not None:
                    with tm.block('cache_build'):
                        row = torch.empty((2, NCOL), dtype=torch.float64, device='cuda')
                        for col, (p0, s0, k0) in COL_SRC.items():
                            a_, b_ = COLOFF[col]
                            row[0, a_:b_], row[1, a_:b_] = gst[k0]
                        gcollect['data'][gcollect['row'][node]] = row.cpu().numpy()
                        gcollect['n'][node] = n
        del wq, hk, cap
        log(f'static: layer {l} ({sum(tm.t.values()):.0f}s)')
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'd2': d2_new, 'ce_forward': ce_fwd, 'grads_finite': gfin, 'nonfinite': nonfinite}


def new_gcollect(names):
    experts = sorted({'.'.join(n.split('.')[:2]) for n in names}, key=lambda s: (int(s.split('.')[0][1:]), int(s.split('.')[1][1:])))
    return {'row': {e: i for i, e in enumerate(experts)}, 'n': {}, 'data': np.zeros((len(experts), 2, ss.NCOL), np.float64)}


def write_gc(path, ipath, gcollect, sources):
    np.save(path, gcollect['data'])
    write(ipath, {'experts': list(gcollect['row']), 'row': gcollect['row'], 'n_static': gcollect['n'], 'columns': ss.COLS,
                  'column_offsets': ss.COLOFF, 'column_source': {c: list(v) for c, v in ss.COL_SRC.items()},
                  'stats': ['G2 = sum over static routed rows of g^2', 'GS = sum over windows of (sum over the window rows of g)^2'],
                  'gradient': 'summed CE, FP32 reference, routes constant; x_in = the shared gate / up pre-QDQ input with its total gradient',
                  'shape': list(gcollect['data'].shape), 'dtype': 'float64', 'sources': sources})


# --------------------------------------------------------------------------- #
# E: forward-only activation-error refresh
# --------------------------------------------------------------------------- #
def experts_observe(w, x2d, idx, rw, act, want, obs, E=NE):
    """run_olmoe_proxy_refresh.experts_tap op order, no autograd: keeps references to the pre-QDQ slot tensors of the experts in want."""
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
            obs[e] = {'xe': xe, 'zg': zg, 'zu': zu, 'h': h, 'zd': zd, 'tok': tok}
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_observe(w, h, cfg, cos, sin, act, want, obs):
    import olmoe_common as oc
    B, T, H = h.shape
    x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
    z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, H)
    idx, rw = oc.route(w, z, cfg)
    return x + experts_observe(w, z, idx, rw, act, want, obs).view(B, T, H), idx, rw


def e_update(fm, dq, ranges, names, tm, log, gc, gidx, static_tab, save_err=None):
    """forward-only: q0_knee dynamic forward; at the B taps, per candidate format, sum err / sum err^2 over the q0 rows; activation
    cost with the static G2 / GS (q0-row normalization); weight rows = static."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    COLOFF, COL_SRC, SLOT_COL = ss.COLOFF, ss.COL_SRC, ss.SLOT_COL
    searched = defaultdict(dict)
    for n in names:
        l, e, p = n.split('.')
        searched[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    C_, meta = {}, {}
    with AutogradGuard() as guard, torch.inference_mode():
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
            want = searched.get(l, {})
            obs = {}
            with tm.block('forward_routing'):
                h, idx, rw = layer_observe(wq, h, cfg, cos, sin, pr.act_map(hk), want, obs)
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
                with tm.block('cache_read'):
                    grow = torch.from_numpy(np.ascontiguousarray(gc[gidx['row'][node]])).to('cuda')
                with tm.block('act_error'):
                    stats = {}
                    for col, (p0, s0, k0) in COL_SRC.items():
                        t, rng = c[k0], ranges[f'{node}.{p0}'][s0]
                        s1s, s2s = [], []
                        for f in QUANT:
                            err = (dq.act(t, f, rng) - t).double()
                            s1s.append(err.sum(0)); s2s.append((err ** 2).sum(0))
                        stats[col] = (torch.stack(s1s), torch.stack(s2s))
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
                        row = np.zeros((2, len(QUANT), ss.NCOL), np.float64)
                        for col, (S1_, S2_) in stats.items():
                            a_, b_ = COLOFF[col]
                            row[0, :, a_:b_], row[1, :, a_:b_] = S1_.cpu().numpy(), S2_.cpu().numpy()
                        save_err['data'][gidx['row'][node]] = row
                        save_err['n'][node] = n
            del wq, hk, obs
            log(f'E: layer {l} ({sum(tm.t.values()):.0f}s)')
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
        with tm.block('controls'):
            hc = h.cpu().numpy()
            hidden_sha = [hashlib.sha256(np.ascontiguousarray(hc[k]).tobytes()).hexdigest() for k in range(nW)]
    fallback = [n for n in names if meta['.'.join(n.split('.')[:2])]['routed_rows'] == 0]
    with tm.block('table_assembly'):
        B_ = {n: static_tab['B'][n] for n in names}
        for n in fallback:
            C_[n] = static_tab['C'][n]
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd, 'hidden_sha': hidden_sha,
            'fallback_static_rows': fallback, 'nonfinite': nonfinite,
            'gradient_free': {'autograd_calls': dict(guard.calls), 'grad_enabled_inside': grad_enabled, 'inference_mode': True}}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    hist = history()
    names, nodes, elems, flops, rows = pr.problem_static()
    ranges = read(RANGES)
    experts = sorted({'.'.join(n.split('.')[:2]) for n in names})
    shared_ok = all(ranges[f'{e}.gate_proj']['x'] == ranges[f'{e}.up_proj']['x'] for e in experts)
    prior_gc = sorted(str(p.relative_to(HERE)) for p in HERE.glob('runs/**/*grad*cache*'))
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = ss.input_files() + list(FROZEN.values()) + list(FROZEN_TABLES.values()) + [
        KS / f for f in ('freeze.json', 'check.json', 'controls.json', 'knee/knee.json', 'knee/q0_knee.json', 'work/q0_knee_calibration_routes_dr_forward.npz')] + [
        KA / f for f in ('freeze.json', 'check.json')] + [K5 / f for f in ('freeze.json', 'check.json', 'eval.json', 'tokens.npy')] + [
        C_STATS / 'target_stats.npz', C_STATS / 'target_stats_meta.json', ACT_CACHE, ACT_INDEX]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'At the normal knee, does refreshing only the activation quantization errors in q0_knee\'s working state (E, static '
                    'sensitivities, forward-only update) improve the static BC, keep the full refresh\'s quality, and beat the sensitivity-only C?',
        'not_this_round': 'no new knee, no training or parameter update, no BC formula change, no ranges / scale change, no loss-based '
                          'selection or fallback, no new search algorithm, no bit-threshold selective variants, no default-pipeline change',
        'fixed': {'history': hist, 'frozen_configs': {c: str(p.relative_to(HERE)) for c, p in FROZEN.items()}, 'limits': read(KNEE)['limits'],
                  'same': 'checkpoint and router, calibration, ranges, quantizer and numerics, 3051 searched projections and candidates, '
                          'resource model, LP-first search, exact knee limits'},
        'E': {'weight': 'the static (A) weight rows unchanged: Delta W(f) = Q_f(W) - W of the original weights, static sensitivities',
              'activation': 'A_E(f) = sum_slots,c [G2_static,c var_new,c(f) + GS_static,c mu_new,c(f)^2]; err_new(f) = Q_f(a_q0) - a_q0 at the '
                            'B taps (x: gate / up shared pre-QDQ expert input, down pre-QDQ SiLU(g) u; y: pre-QDQ linear outputs) with the fixed '
                            'ranges; mu_new = sum err / n_q0, var_new = sum err^2 / n_q0 - mu_new^2 (q0 routed rows); G2 / GS keep the static '
                            'scale; evaluated as G2 / n_q0 (S2 - n_q0 mu^2) + GS mu^2',
              'scope': 'every searched projection and candidate format; a projection whose expert has no q0 token keeps its static row (listed)',
              'state': 'one q0_knee calibration forward with its own dynamic routes; candidate formats are only observed, the forward keeps q0',
              'gradient_statistics': 'G2_static / GS_static cached in a static rebuild here (no earlier verifiable cache: '
                                     f'{prior_gc or "none found"}); never inferred from the cost tables; the rebuilt static table must equal A bitwise',
              'forward_only': 'torch.inference_mode, an observation path without retain_grad (not the backward tap function), autograd calls '
                              'counted (must be 0); no weight sensitivity or gradient statistics; the original static build still needs a backward',
              'search': 'one LP-first search at the exact knee limits; every projection may change format; LP bound, objective, gaps and the '
                        'cross scores of A / B / C / D4 / D5 / E on the E table (with feasibility) recorded; if another configuration scores below '
                        'E\'s solution on the E table, the conclusion covers error-only + the current approximate search'},
        'gate_up_x_ranges_identical': shared_ok,
        'controls': {
            '1_inputs': 'inputs and A / B / C (D4 / D5) configurations equal their records',
            '2_static_reconstruction': 'static gradient statistics + static error cache reproduce A\'s activation costs (independent numpy, '
                                       f'relative <= {TOL_REL}); the static rebuild that writes the gradient cache reproduces the A table bitwise',
            '3_weight_rows': 'E weight rows bitwise equal A',
            '4_forward': 'E q0_knee forward CE, hidden sha256 and routes equal an independent run_olmoe_dynamic_route.forward and the records',
            '5_recompute': f'independent numpy recomputation of every E activation row from the static gradient cache and the saved new '
                           f'errors (relative <= {TOL_REL})',
            '6_B_reproduction': f'the q0 gradient statistics saved by C (runs/olmoe_knee_all_sensitivity_20261022 refresh_C_r1) with the new errors '
                                f'reproduce B\'s activation costs (relative <= {TOL_REL}); control only, outside E\'s method time',
            '7_gradient_free': 'E update: 0 autograd calls, grad disabled, no backward / weight-sensitivity / gradient-statistics blocks',
            '8_validity': 'finite, legal, exact limits; fallback listed',
            '9_repeats': 'E repeats identical tables / configurations (main = repeat 1); new B / C runs equal the frozen ones; end-to-end E '
                         'reproduces the static table, gradient cache, knee and E configuration'},
        'timing': {'refresh': f'B / C / E, 3 each, interleaved {REFRESH_SCHEDULE}; one process per run; CUDA synchronization; no page-cache dropping',
                   'static_gc': 'the canonical static rebuild with the gradient cache (single run): cache build / write cost charged to E only',
                   'e2e': 'one end-to-end E run: static BC + gradient cache -> auto knee -> E forward-only refresh -> search (single measurement)',
                   'categories': CATEGORIES, 'separate': 'controls, repeats, evaluation'},
        'freeze': 'plan, code, inputs, caches, tables, configurations and hashes frozen before any evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries, duplicate and near-duplicate '
                               'exclusion; every earlier manifest (incl. the D5 round) excluded by scanning runs/**/*segments*.json',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests, 'eligible_before_length_and_neardup': len(elig),
                       'seeds': SEEDS, 'conditions': 'FP32, A, B, C, E on the same new articles; original router; each its own dynamic route',
                       'primary': {'E - A': 'corrected upper < 0 -> E better', 'E - C': 'corrected upper < 0 -> E better',
                                   'E - B': f'corrected upper < +{NI_MARGIN} -> non-inferior; corrected lower > +{NI_MARGIN} -> beyond the margin; '
                                            'otherwise insufficient evidence'},
                       'intervals': f'mean, 95% CI, Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI ({NCMP} comparisons); decisions on the corrected interval',
                       'secondary': 'B - A, C - A (95% CI)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'cautions': 'no extra articles, seed or rule changes; non-inferior is not better than static; not confirmed is not equal'},
        'report_questions': ['1 E better than the static BC?', '2 E keeps B\'s quality?', '3 extra and total offline time saved vs B / C',
                             '4 does it support the error-only direction?', '5 worth keeping as an optional step?'],
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not hist['pass'] or not shared_ok:
        raise SystemExit('history hashes or shared gate / up x ranges differ: investigate first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 正常 knee 的 activation error-only refresh（E）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- A／B／C 沿用凍結配置（D4／D5 只用於交叉評分診斷）；E 用同一個 q0_knee、knee/knee.json 的精確上限重新搜尋一次，所有節點可改格式。',
          '- E：weight 成本整張沿用 A；activation 成本 = Σ(G2_static·var_new + GS_static·μ_new²)，新誤差在 B 相同的 tap 取得、以 q0 routed rows 正規化，G2／GS 維持 static 尺度。',
          '- G2_static／GS_static：在本輪的 static 重建中補存（不從成本表反推），重建表須和 A 逐位元相同；建立／寫入／讀取成本只算給 E。',
          '- E 的更新為 forward-only：inference_mode、無 retain_grad 的觀測路徑、計數 autograd 呼叫（須為 0）；首次建立原始 BC 仍需要 backward。',
          '- 控制：static 重建、E weight 列 = A、forward 一致、numpy 獨立重算 E、以 C 存的 q0 梯度＋新誤差重現 B、無梯度計算、合法與上限、重複一致。',
          f'- 計時：B／C／E 各 3 次交錯（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；static＋梯度快取單次；E 端到端單次。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、C、E；seed 22801／22802／22811。',
          f'- 主要：E − A、E − C（校正上界 < 0 才確認較好）、E − B（校正上界 < +{NI_MARGIN} 非劣；下界 > +{NI_MARGIN} 超過界線）；Bonferroni {100 - 2 * CI_CORR[0]:.2f}%。次要：B − A、C − A（95%）。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; history ok {hist["pass"]}; prior gradient caches {prior_gc}; previous WT103 {len(used)}; eligible {len(elig)}')


def static(out, log):
    """canonical static rebuild with the gradient-statistics cache."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    rd = out / 'work/runs/static_gc'
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(watch_files(out))
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    with tm.block('cache_build'):
        gcol = new_gcollect(names)
    st = build_static(dq, ranges, names, tm, log, gcollect=gcol)
    gp, gi = gc_paths(out)
    with tm.block('cache_write'):
        write_gc(gp, gi, gcol, {'static_table_sha256': sha(STATIC_TABLE), 'calibration_sha256': sha(CALIB), 'ranges_sha256': sha(RANGES),
                                'code_sha256': sha(HERE / 'run_olmoe_knee_error_only.py')})
    with tm.block('table_write'):
        write(rd / 'table_static.json', {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'})
    ref = read(STATIC_TABLE)
    rec = run_record(tm, t0, {'page_cache_residency_before': residency, 'static_B_equal_A': st['B'] == ref['B'], 'static_C_equal_A': st['C'] == ref['C'],
                              'n_equal_act_cache': gcol['n'] == read(ACT_INDEX)['n'], 'grads_finite': st['grads_finite'], 'nonfinite': len(st['nonfinite']),
                              'ce_forward': st['ce_forward'], 'cache_bytes': gp.stat().st_size, 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({k: rec[k] for k in ('total_s', 'method_s', 'static_B_equal_A', 'static_C_equal_A', 'n_equal_act_cache', 'cache_bytes')}))


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
           'ste_unit_test': pr.ste_unit_test(dq, ranges), 'history': history(), 'static_gc_run': read(out / 'work/runs/static_gc/run.json'),
           'page_cache_residency': ss.resident_fraction(watch_files(out)), 'seconds': time.perf_counter() - t0}
    s = ctl['static_gc_run']
    ctl['pass'] = bool(rd['finite_ce'] and all(ctl['equal_previous_record'].values()) and ctl['ste_unit_test']['pass'] and ctl['history']['pass']
                       and s['static_B_equal_A'] and s['static_C_equal_A'] and s['n_equal_act_cache'] and s['grads_finite'] and s['nonfinite'] == 0)
    ctl['static_gc_run'] = {k: s[k] for k in ('static_B_equal_A', 'static_C_equal_A', 'n_equal_act_cache', 'grads_finite', 'nonfinite', 'cache_bytes')}
    write(out / 'controls.json', ctl)
    log(json.dumps({'equal_previous_record': ctl['equal_previous_record'], 'static_gc': ctl['static_gc_run'], 'pass': ctl['pass']}))


def refresh(out, log, variant, rep):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = f'refresh_{variant}_r{rep}'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(watch_files(out))
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
    if variant in ('B', 'C'):
        with tm.block('cache_read_d2'):
            d2z = np.load(D2)
        cache = cidx = None
        if variant == 'C':
            with tm.block('cache_read'):
                cache = np.load(ACT_CACHE, mmap_mode='r')
                cidx = read(ACT_INDEX)
        r = ks.build(variant, dq, ranges, names, tm, log, fm=fm, d2z=d2z, cache=cache, cidx=cidx)
        extra = {'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'recompute_max_abs': max(r['recompute_max_abs_per_layer']), 'grads_finite': r['grads_finite']}
    else:
        gp, gi = gc_paths(out)
        with tm.block('cache_read'):
            gc = np.load(gp, mmap_mode='r')
            gidx = read(gi)
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
        save = None
        if rep == 1:
            save = {'n': {}, 'data': np.zeros((len(gidx['experts']), 2, 10, ss.NCOL), np.float64)}
        r = e_update(fm, dq, ranges, names, tm, log, gc, gidx, static_tab, save_err=save)
        extra = {'gradient_free': r['gradient_free']}
        if save is not None:
            with tm.block('controls'):
                np.save(rd / 'new_error_stats.npy', save['data'])
                write(rd / 'new_error_stats_index.json', {'row': gidx['row'], 'n_q0': save['n'], 'columns': ss.COLS, 'stats': ['sum_err', 'sum_err_sq'],
                                                          'formats': list(__import__('run_native_bc_pipeline').QUANT), 'shape': list(save['data'].shape)})
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': f'q0_knee working state, condition {variant}'}
    fm_new, srch = ss.search_table(t, (names, nodes, elems, flops), limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm_new)
    with tm.block('controls'):
        np.savez(rd / 'q0_knee_calibration_routes.npz', **r['routes'])
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'search': srch, 'cache_use': cache_use(variant, tm.t),
                              'fallback_static_rows': r['fallback_static_rows'], 'ce_forward': r['ce_forward'], 'hidden_sha': r['hidden_sha'],
                              'routes_hash': dr.route_hash(r['routes']), 'nonfinite': len(r['nonfinite']),
                              'q0_routed_rows_by_expert': {k: v['routed_rows'] for k, v in r['meta'].items()}, 'code': code_hashes(), **extra})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path'], 'cache_use': rec['cache_use'],
                    **({'gradient_free': extra['gradient_free']} if variant == 'E' else {})}))


def e2e(out, log):
    """one process: static BC + gradient cache -> auto knee -> E forward-only refresh (cache just written) -> search."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = 'e2e_E'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(watch_files(out))
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
        gcol = new_gcollect(names)
    st = build_static(dq, ranges, names, tm, log, gcollect=gcol)
    t = {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'}
    marks = {'static_done_s': time.perf_counter() - t0}
    with tm.block('table_write'):
        write(rd / 'table_static.json', t)
    gp, gi = rd / 'grad_stats_cache.npy', rd / 'grad_stats_cache_index.json'
    with tm.block('cache_write'):
        write_gc(gp, gi, gcol, {'state': 'end-to-end run'})
    marks['static_written_s'] = time.perf_counter() - t0
    del st, gcol
    sel, recs, slog, kept = ks.auto_search(t, prob, tm, out / 'logs', tag)
    q0k, limits = ks.knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'q0_knee.json', q0k)
    marks['knee_done_s'] = time.perf_counter() - t0
    with tm.block('cache_read'):
        gc = np.load(gp, mmap_mode='r')
        gidx = read(gi)
    r = e_update(q0k, dq, ranges, names, tm, log, gc, gidx, t)
    tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': 'q0_knee working state, condition E'}
    fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', tt)
        write(rd / 'config.json', fm_new)
    rec = run_record(tm, t0, {'variant': 'E', 'page_cache_residency_before': residency, 'marks': marks, 'limits': limits,
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'auto_paths': dict(Counter(x.get('path') for x in recs)),
                              'search': srch, 'gradient_free_E_stage': r['gradient_free'], 'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward'],
                              'nonfinite': len(r['nonfinite']), 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps({'tag': tag, 'method_s': rec['method_s'], 'marks': marks, 'gradient_free': r['gradient_free']}))


def timing(out, log, phase):
    sched = REFRESH_SCHEDULE if phase == 'refresh' else [('E', None)]
    for v, r in sched:
        name = f'refresh_{v}_r{r}' if phase == 'refresh' else 'e2e_E'
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
    er = [f'refresh_E_r{r}' for r in (1, 2, 3)]
    exp = rr + ['e2e_E']
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    ctl, prevc = read(out / 'controls.json'), read(KS / 'controls.json')
    st, kn = read(STATIC_TABLE), read(KNEE)
    ft = {v: read(p) for v, p in FROZEN_TABLES.items()}
    fc = {v: read(p) for v, p in FROZEN.items()}
    tE, cE = read(W / 'refresh_E_r1/table.json'), read(W / 'refresh_E_r1/config.json')
    hist = history()
    res['1_inputs'] = {'history': hist, 'pass': hist['pass']}
    # 2. static reconstruction of A's activation costs
    gp, gi = gc_paths(out)
    gc, gidx = np.load(gp, mmap_mode='r'), read(gi)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)

    def act_cost_np(S1r, S2r, n, G2, GS, p):
        out_ = np.zeros(len(QUANT))
        for s in ('x', 'y'):
            a_, b_ = ss.COLOFF[ss.SLOT_COL[(p, s)]]
            G2v, GSv = G2(s, a_, b_), GS(s, a_, b_)
            for i in range(len(QUANT)):
                S1_, S2_ = S1r[i, a_:b_], S2r[i, a_:b_]
                mu = S1_ / n
                out_[i] += float(np.sum(G2v / n * (S2_ - n * mu ** 2)) + np.sum(GSv * mu ** 2))
        return out_
    rel2 = 0.0
    for nm in names:
        node, p = '.'.join(nm.split('.')[:2]), nm.split('.')[2]
        grow, arow, ns = np.asarray(gc[gidx['row'][node]]), np.asarray(ac[aidx['row'][node]]), aidx['n'][node]
        v = act_cost_np(arow[0], arow[1], ns, lambda s, a_, b_: grow[0, a_:b_], lambda s, a_, b_: grow[1, a_:b_], p)
        rel2 = max(rel2, max(rr_(v[i], st['C'][nm][f]) for i, f in enumerate(QUANT)))
    sgc = read(W / 'static_gc/run.json')
    res['2_static_reconstruction'] = {'numpy_act_max_rel': rel2, 'static_rebuild_B_equal_A': sgc['static_B_equal_A'], 'static_rebuild_C_equal_A': sgc['static_C_equal_A'],
                                      'n_static_equal_act_cache': gidx['n_static'] == aidx['n'],
                                      'pass': rel2 <= TOL_REL and sgc['static_B_equal_A'] and sgc['static_C_equal_A'] and gidx['n_static'] == aidx['n']}
    res['3_weight_rows'] = {'E_weight_rows_equal_A': all(tE['B'][n] == st['B'][n] for n in names)}
    res['3_weight_rows']['pass'] = res['3_weight_rows']['E_weight_rows_equal_A']
    drr = dr.route_hash(dict(np.load(KS / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    c4 = {k: {'ce': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'] == prevc['q0_knee_forward_ce'], 'hidden': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
              'routes': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr} for k in rr}
    c4['e2e_E'] = {'ce': runs['e2e_E']['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden': runs['e2e_E']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha']}
    g_prev = read(KS / 'work/runs/refresh_B_r1/run.json')['input_grad_sha_per_layer']
    c4_grad = {k: runs[k]['input_grad_sha_per_layer'] == g_prev and runs[k]['recompute_max_abs'] == 0 for k in rr if k not in er}
    res['4_forward'] = {'runs': c4, 'B_C_input_gradients_equal_record': c4_grad, 'pass': all(all(v.values()) for v in c4.values()) and all(c4_grad.values())}
    # 5. E recompute from the static gradient cache + new errors; 6. B from C's q0 gradient statistics + new errors
    ne, nidx = np.load(W / 'refresh_E_r1/new_error_stats.npy', mmap_mode='r'), read(W / 'refresh_E_r1/new_error_stats_index.json')
    zc, smc = np.load(C_STATS / 'target_stats.npz'), read(C_STATS / 'target_stats_meta.json')
    fbE = set(runs['refresh_E_r1']['fallback_static_rows'])
    rel5 = rel5u = rel6 = relq = 0.0
    for nm in names:
        if nm in fbE:
            continue
        node, p = '.'.join(nm.split('.')[:2]), nm.split('.')[2]
        erow, nq = np.asarray(ne[nidx['row'][node]]), nidx['n_q0'][node]
        grow = np.asarray(gc[gidx['row'][node]])
        v5 = act_cost_np(erow[0], erow[1], nq, lambda s, a_, b_: grow[0, a_:b_], lambda s, a_, b_: grow[1, a_:b_], p)
        v6 = act_cost_np(erow[0], erow[1], nq, lambda s, a_, b_: zc[f'{nm}|{s}_G2'], lambda s, a_, b_: zc[f'{nm}|{s}_GS'], p)
        ns = aidx['n'][node]
        vq = act_cost_np(erow[0], erow[1], ns, lambda s, a_, b_: grow[0, a_:b_], lambda s, a_, b_: grow[1, a_:b_], p) if ns != nq else v5
        for i, f in enumerate(QUANT):
            rel5 = max(rel5, rr_(v5[i], tE['C'][nm][f]))
            rel6 = max(rel6, rr_(v6[i], ft['B']['C'][nm][f]))
            relq = max(relq, rr_(vq[i], tE['C'][nm][f]))
        assert smc[nm]['n_q0'] == nq
    res['5_recompute'] = {'numpy_E_act_max_rel': rel5, 'descriptive_static_rows_normalization_max_rel': relq, 'pass': rel5 <= TOL_REL}
    res['6_B_reproduction'] = {'numpy_B_act_max_rel': rel6, 'pass': rel6 <= TOL_REL,
                               'note': 'C repeat 1 q0 gradient statistics (G2 / GS per projection and slot) + the new errors of E repeat 1 vs the frozen B table'}
    gf = {k: runs[k]['gradient_free'] for k in er}
    gf['e2e_E'] = runs['e2e_E']['gradient_free_E_stage']
    res['7_gradient_free'] = {'runs': gf, 'cache_use': {k: runs[k]['cache_use'] for k in rr},
                              'pass': all(v['autograd_calls'] == {} and v['grad_enabled_inside'] is False for v in gf.values())
                              and all(runs[k]['cache_use']['as_expected'] for k in rr)}
    lim = kn['limits']
    rc = resources(cE, names, elems, flops)
    res['8_validity'] = {'fallback_count': len(fbE), 'fallback_rows': sorted(fbE), 'fallback_reason': 'searched expert with no q0_knee calibration token (static row kept)',
                         'finite': all(runs[k]['nonfinite'] == 0 for k in exp) and all(runs[k].get('grads_finite', True) for k in rr),
                         'covers_exactly_searched': sorted(cE) == sorted(names) and sorted(tE['B']) == sorted(names) and sorted(tE['C']) == sorted(names),
                         'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in cE.values()), 'resources': rc,
                         'within_limits': rc['M_bytes'] <= lim['max_bytes'] and rc['BOPs'] <= lim['max_bops'],
                         'search_within_budget_all_runs': all(runs[k]['search']['within_budget'] for k in exp), 'e2e_limits_equal': runs['e2e_E']['limits'] == lim}
    v8 = res['8_validity']
    v8['pass'] = bool(v8['finite'] and v8['covers_exactly_searched'] and v8['legal'] and v8['within_limits'] and v8['search_within_budget_all_runs'] and v8['e2e_limits_equal'])
    reps = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        tref = tE if v == 'E' else ft[v]
        cref = cE if v == 'E' else fc[v]
        reps[v] = {'tables': all({kk: vv for kk, vv in read(W / k / 'table.json').items() if kk in ('B', 'C')} == {kk: tref[kk] for kk in ('B', 'C')} for k in ks_),
                   'configs': all(read(W / k / 'config.json') == cref for k in ks_)}
    reps['e2e_E'] = {'static_table': {kk: vv == st[kk] for kk, vv in read(W / 'e2e_E/table_static.json').items() if kk in ('B', 'C')},
                     'grad_cache': sha(W / 'e2e_E/grad_stats_cache.npy') == sha(gp), 'knee': read(W / 'e2e_E/q0_knee.json') == read(Q0K),
                     'E_table': {kk: read(W / 'e2e_E/table.json')[kk] == tE[kk] for kk in ('B', 'C')}, 'E_config': read(W / 'e2e_E/config.json') == cE}
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['9_repeats'] = reps
    res['9_repeats']['pass'] = flat(reps)
    res['identical'] = {f'E={c}': cE == fc[c] for c in fc}
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['inputs_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['pass'] = bool(not res['missing_runs'] and all(res[k]['pass'] for k in ('1_inputs', '2_static_reconstruction', '3_weight_rows', '4_forward', '5_recompute',
                                                                                 '6_B_reproduction', '7_gradient_free', '8_validity', '9_repeats'))
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
    write(out / 'configs/E.json', read(W / 'refresh_E_r1/config.json'))
    write(out / 'tables/E.json', read(W / 'refresh_E_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names)}
    s4 = set(read(KS / 'lowbit_nodes.json')['nodes'])
    s5 = set(read(K5 / 'low5_nodes.json')['nodes'])
    zone = lambda n: 'le4' if n in s4 else 'new5' if n in s5 else 'rest'
    gp, gi = gc_paths(out)
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'E = refresh repeat 1; A / B / C (D4 / D5) = frozen records',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c]), 'resources': resources(cfg[c], names, elems, flops)} for c in cfg},
           'frozen_equal_records': {c: cfg[c] == read(p) for c, p in FROZEN.items()},
           'identical': {f'E={c}': cfg['E'] == cfg[c] for c in ('A', 'B', 'C', 'D4', 'D5')},
           'pairwise_differences': {f'E-{c}': diff(cfg['E'], cfg[c]) for c in ('A', 'B', 'C')} | {f'{c}-A': diff(cfg[c], cfg['A']) for c in ('B', 'C')},
           'changed_vs_A_by_zone': {c: dict(Counter(zone(n) for n in names if cfg[c][n] != cfg['A'][n])) for c in ('B', 'C', 'E')},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in ('E',)},
           'tables': {'E.json': sha(out / 'tables/E.json'), **{f'frozen/{v}': sha(p) for v, p in FROZEN_TABLES.items()}},
           'grad_cache': {'npy': sha(gp), 'index': sha(gi), 'bytes': gp.stat().st_size}, 'knee_sha256': sha(KNEE),
           'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'), 'code': code_hashes()}
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
    old_t, old = np.load(K5 / 'tokens.npy')[:CTRL], read(K5 / 'eval.json')['ce']
    pc = {}
    for c in ('fp32', 'A', 'B', 'C'):
        r = dr.forward(old_t, None if c == 'fp32' else read(FROZEN[c]), dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs, shared, by_hash = {}, {}, {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in ('A', 'B', 'C', 'E'):
        h = fz['configs'][c]['config_hash']
        if h in by_hash:
            runs[c] = runs[by_hash[h]]; shared[c] = by_hash[h]
            continue
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        by_hash[h] = c
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/E.json'), dq, ranges, record=True, log=log, label='rerun E')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_E_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['E']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['E']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['E']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in ('A', 'B', 'C', 'E')},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean())
                                       for c in ('A', 'B', 'C', 'E')}}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_E_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_E_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_E_first_chunk']['routes_equal'] and ctl['finite']
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
        if j['stage'] == 'refresh':
            proc[f'refresh_{j["variant"]}_r{j["rep"]}'] = j['seconds']
        elif j['stage'] == 'e2e':
            proc['e2e_E'] = j['seconds']
        elif j['stage'] == 'static':
            proc['static_gc'] = j['seconds']
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        s[f'refresh_{v}'] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                             'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                             'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks_]) for t in sorted({t for k in ks_ for t in runs[k]['timers']})},
                             'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                             'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks_}}
    g = runs['static_gc']
    s['static_gc_single'] = {'method_s': g['method_s'], 'process_wall_s': proc.get('static_gc'), 'categories': g['categories'], 'timers': g['timers'],
                             'gradient_cache_extra_s': g['timers'].get('cache_build', 0.0) + g['timers'].get('cache_write', 0.0), 'cache_bytes': g['cache_bytes'],
                             'peak_gpu_gb': g['peak_gpu_gb'], 'peak_cpu_gb': g['peak_cpu_gb']}
    e = runs['e2e_E']
    s['e2e_E_measured_single'] = {'method_s': e['method_s'], 'process_wall_s': proc['e2e_E'], 'marks': e['marks'], 'categories': e['categories'],
                                  'peak_gpu_gb': e['peak_gpu_gb'], 'peak_cpu_gb': e['peak_cpu_gb'], 'page_cache_residency_before': e['page_cache_residency_before']}
    kt, at, ft = read(KS / 'analysis.json')['timing'], read(KA / 'analysis.json')['timing'], read(K5 / 'analysis.json')['timing']
    s['historical'] = {'e2e_measured_single': {'A': kt['e2e_measured']['A']['method_s'], 'B': kt['e2e_measured']['B']['method_s'],
                                               'C': at['e2e_C_measured_single']['method_s'], 'D4': kt['e2e_measured']['D']['method_s'],
                                               'D5': ft['e2e_D5_measured_single']['method_s']},
                       'auto_median_s': kt['auto']['method_s']['median'], 'note': 'previous knee rounds, same machine and code; historical single measurements'}
    SW = HERE / 'runs/olmoe_selective_sensitivity_20261022/work/runs'
    sp = float(np.median([read(SW / f'static_plain_r{r}/run.json')['method_s'] - sum(read(SW / f'static_plain_r{r}/run.json')['timers'].get(t, 0.0)
                                                                                     for t in ('pruning', 'search', 'cache_write_d2')) for r in (1, 2, 3)]))
    s['E_total_estimate_sum_of_stage_medians_s'] = {'value': sp + s['static_gc_single']['gradient_cache_extra_s'] + kt['auto']['method_s']['median'] + s['refresh_E']['method_s']['median'],
                                                    'components': {'static_without_caches': sp, 'gradient_cache_build_write_single': s['static_gc_single']['gradient_cache_extra_s'],
                                                                   'auto_median_previous_round': kt['auto']['method_s']['median'], 'E_median': s['refresh_E']['method_s']['median']}}
    s['extra_with_caches_median_s'] = {v: s[f'refresh_{v}']['method_s']['median'] for v in VARIANTS}
    s['note'] = 'method = run time minus control blocks; estimates are sums of stage medians, not a measured pipeline median'
    return s


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    from run_olmoe_multipoint import resources
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
    pairs = {'E_minus_A': ('E', 'A'), 'E_minus_B': ('E', 'B'), 'E_minus_C': ('E', 'C'), 'B_minus_A': ('B', 'A'), 'C_minus_A': ('C', 'A')}
    D = {k: A[a] - A[b] for k, (a, b) in pairs.items()}
    res = {'n_articles': N, 'bonferroni_level': 100 - 2 * CI_CORR[0], 'primary': {k: st(D[k]) for k in ('E_minus_A', 'E_minus_B', 'E_minus_C')},
           'secondary': {k: st(D[k]) for k in ('B_minus_A', 'C_minus_A')}, 'damage': {c: st(A[c] - A['fp32']) for c in ('A', 'B', 'C', 'E')}}
    for k in ('E_minus_A', 'E_minus_C'):
        lo, hi = res['primary'][k]['ci_bonferroni']
        res['primary'][k]['decision'] = 'E_better_confirmed' if hi < 0 else 'E_worse_confirmed' if lo > 0 else 'not_confirmed'
    lo, hi = res['primary']['E_minus_B']['ci_bonferroni']
    res['primary']['E_minus_B']['decision'] = (f'noninferior_within_{NI_MARGIN}' if hi < NI_MARGIN else f'beyond_margin_{NI_MARGIN}' if lo > NI_MARGIN
                                               else 'insufficient_evidence')
    for k, v in res['secondary'].items():
        lo, hi = v['ci95']
        v['direction_95'] = 'first_better' if hi < 0 else 'first_worse' if lo > 0 else 'not_confirmed'
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    tE = read(out / 'tables/E.json')
    lim = read(KNEE)['limits']
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    sE = read(out / 'work/runs/refresh_E_r1/run.json')['search']
    res['search_E'] = {k: sE[k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
    res['search_E']['gap_abs'] = sE['objective'] - sE['lower_bound']
    res['search_E']['gap_rel'] = (sE['objective'] - sE['lower_bound']) / sE['objective']
    cross = {}
    for c in cfg:
        r = resources(cfg[c], names, elems, flops)
        cross[c] = {'objective_on_E_table': obj(tE, cfg[c]), 'feasible_at_knee_limits': r['M_bytes'] <= lim['max_bytes'] and r['BOPs'] <= lim['max_bops'],
                    'below_E_solution': obj(tE, cfg[c]) < sE['objective'] and c != 'E'}
    res['cross_scores_on_E_table'] = cross
    res['approximate_search_flag'] = any(v['below_E_solution'] and v['feasible_at_knee_limits'] for v in cross.values())
    res['configs'] = {c: v['resources'] for c, v in fz['configs'].items()}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'changed_vs_A_by_zone', 'transitions_vs_A')}
    from run_olmoe_multipoint import problem
    _, _, pnames, _, elems2, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    for c in ['fp32', 'A', 'B', 'C', 'E']:
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
    for c in ('A', 'B', 'C', 'E'):
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
                    'approximate_search_flag': res['approximate_search_flag']})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'static', 'controls', 'refresh', 'e2e', 'timing', 'check', 'freeze', 'data', 'eval', 'analyze'], required=True)
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
