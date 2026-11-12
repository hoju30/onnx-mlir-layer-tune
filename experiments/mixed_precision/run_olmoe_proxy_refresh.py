#!/usr/bin/env python3
"""OLMoE: does one full refresh of the BC proxy in the quantized model's working state select a better configuration
than the static FP32 proxy, at the same weight-size and BOPs limits? Pre-registered in plan.json / plan.md.

A (static): BC table rebuilt here on the FP32 reference (stage-1 calibration, 10 x 128 tokens, FP32 dynamic routes)
            -> LP-first search at the limits of the aggressive configuration -> q0.
B (refresh): the BC table rebuilt once in q0's working state (q0's own dynamic routes, q0 hidden states, weight /
            activation QDQ of q0 in the forward; routes constant and identity STE through activation QDQ in the
            backward) -> same LP-first search, same limits -> q1. One refresh only; no fallback to q0 by loss.

The BC form is unchanged (run_olmoe_multipoint.stage_table): weight term B = sum_c (G_c / n_in) sum_in dW^2 with
G_c = sum over windows of the squared per-window weight-gradient row sums, dW = Q_f(W) - W of the original checkpoint
W; activation term C = sum over the x and y slots and channels of G2 / n (sum err^2 - n mu^2) + GS mu^2 (variance and
coherent-bias parts), err = QDQ_f(t) - t of the pre-QDQ activation t with the fixed ranges; gate / up share the
pre-QDQ input with its total gradient; fp32 options cost 0.

Stages (one --out-dir): plan, static, search_static, refresh, search_refresh, freeze, data, eval, analyze.
"""
import argparse
import csv
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
S1 = HERE / 'runs/olmoe_fixed_route_20261015'
MP = HERE / 'runs/olmoe_multipoint_20261015'
MPT = HERE / 'runs/olmoe_dynamic_multipoint_20261018'
RDP = HERE / 'runs/olmoe_route_dynamic_prop_20261018'
RPP = HERE / 'runs/olmoe_route_precision_protection_20261019'
MRD = HERE / 'runs/olmoe_route_mass_restore_20261018'
AGGR = MPT / 'configs/aggressive.json'
RANGES = MP / 'ranges.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N_EVAL, CHUNK, CTRL = 128, 128, 128
EPS, BOOT = 0.002, 10000
SEEDS = {'order': 22101, 'start': 22102, 'bootstrap': 22111}
PROJ = ('gate_proj', 'up_proj', 'down_proj')


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


class Timer:
    def __init__(self):
        self.t = defaultdict(float)

    def add(self, k, s):
        self.t[k] += s

    def block(self, k):
        import torch
        outer = self

        class _B:
            def __enter__(self_):
                torch.cuda.synchronize(); self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                torch.cuda.synchronize(); outer.add(k, time.perf_counter() - self_.t0)
        return _B()


def peak_cpu_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def merge_timing(out, key, obj):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[key] = obj
    write(f, t)


# --------------------------------------------------------------------------- #
# model pieces (olmoe_common.experts_fixed arithmetic with richer taps)
# --------------------------------------------------------------------------- #
def ste_fn():
    import torch

    class STEQ(torch.autograd.Function):
        """forward: the existing QDQ value exactly (computed without autograd); backward: identity (gradient w.r.t. the
        QDQ input = gradient w.r.t. its output, including clamped values). Not the derivative of the discrete map."""
        @staticmethod
        def forward(ctx, x, fn):
            with torch.no_grad():
                return fn(x)

        @staticmethod
        def backward(ctx, g):
            return g, None
    return STEQ


def act_map(hk, ste=None):
    """(e, proj, slot) -> callable, from run_olmoe_search_final.quantized_layer's hooks (optionally STE-wrapped)."""
    if hk is None:
        return {}
    if ste is None:
        return {(k[1], k[2], k[3]): f for k, f in hk.act.items()}
    return {(k[1], k[2], k[3]): (lambda t, f=f: ste.apply(t, f)) for k, f in hk.act.items()}


def experts_tap(w, x2d, idx, rw, act, cap=None, E=NE):
    """olmoe_common.experts_fixed, same op order; with cap: pre/post-QDQ tensors per expert (grads retained)."""
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
        if cap is not None:
            for t in (xe, zg, zu, h, zd):
                t.retain_grad()
            cap[e] = {'xe': xe, 'xg': xg, 'xu': xu, 'zg': zg, 'zu': zu, 'h': h, 'hd': hd, 'zd': zd, 'tok': tok}
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_tap(w, h, cfg, cos, sin, idx, rw, act, cap=None):
    import olmoe_common as oc
    B, T, H = h.shape
    x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
    z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, H)
    if idx is None:
        idx, rw = oc.route(w, z, cfg)
    return x + experts_tap(w, z, idx, rw, act, cap).view(B, T, H), idx, rw


def layer_weights(ck, l, fm, dq, ranges, tm, key):
    import torch
    from run_olmoe_search_final import quantized_layer
    w = ck.layer(l, 'cuda', torch.float32)
    if fm is None:
        return w, None
    with tm.block(key):
        wq, hk = quantized_layer(w, fm, l, dq, ranges, NE)
    return wq, hk


# --------------------------------------------------------------------------- #
# one BC table in a working state (fm None = FP32 reference)
# --------------------------------------------------------------------------- #
def build_table(out, fm, dq, ranges, names, log, label, d2=None):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    tm = Timer()
    torch.cuda.reset_peak_memory_stats()
    cfg, ck = oc.config(), oc.Checkpoint()
    toks_np = np.load(S1 / 'tokens_calibration.npy')
    toks = torch.from_numpy(toks_np).to('cuda')
    nW = len(toks_np)
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    searched = defaultdict(list)
    for n in names:
        l, e, p = n.split('.')
        searched[(int(l[1:]), int(e[1:]))].append(p)
    STEQ = ste_fn()
    # 1. forward (own dynamic routes), no grad
    ids_rec = np.zeros((nW, NL, SEQ, K), np.int16); w_rec = np.zeros((nW, NL, SEQ, K), np.float32)
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = layer_weights(ck, l, fm, dq, ranges, tm, 'quantize_forward')
            with tm.block('forward_routing'):
                h, idx, rw = layer_tap(wq, h, cfg, cos, sin, None, None, act_map(hk))
            routes.append((idx, rw)); hs.append(h)
            ids_rec[:, l] = idx.view(nW, SEQ, K).to(torch.int16).cpu().numpy(); w_rec[:, l] = rw.view(nW, SEQ, K).cpu().numpy()
            del wq, hk
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
    # 2. backward through the head
    with tm.block('backward_sensitivity'):
        hT = hs[-1].clone().requires_grad_(True)
        ce_bwd = head_ce(g, hT, toks, cfg)
        ce_bwd.sum().backward()
        gn = hT.grad
    if [float(x) for x in ce_bwd.detach().cpu().numpy()] != ce_fwd:
        raise RuntimeError('head CE differs between forward and backward')
    B_, C_, meta, rngs, recompute_diff = {}, {}, {}, {}, []
    d2_new = {} if d2 is None else None
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = layer_weights(ck, l, fm, dq, ranges, tm, 'quantize_backward_recompute')
        cap = {}
        with tm.block('backward_sensitivity'):
            hi = hs[l].clone().requires_grad_(True)
            o, _, _ = layer_tap(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], act_map(hk, STEQ), cap)
            recompute_diff.append(float((o.detach() - hs[l + 1]).abs().max()))
            o.backward(gn)
            gn = hi.grad
            gfin &= bool(torch.isfinite(gn).all())
        with torch.no_grad():
            for e in range(NE):
                node = f'L{l}.E{e}'
                if (l, e) not in searched:
                    continue
                if e not in cap:
                    meta[node] = {'routed_rows': 0, 'windows': 0}
                    continue
                c = cap[e]
                tok = c['tok']
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                meta[node] = {'routed_rows': int(n), 'windows': len(wins)}
                taps = {'gate_proj': (c['xg'], c['zg'], ('x', c['xe']), ('y', c['zg'])),
                        'up_proj': (c['xu'], c['zu'], ('x', c['xe']), ('y', c['zu'])),
                        'down_proj': (c['hd'], c['zd'], ('x', c['h']), ('y', c['zd']))}
                for p in searched[(l, e)]:
                    name = f'{node}.{p}'
                    xin, zout, sx, sy = taps[p]
                    with tm.block('backward_sensitivity'):
                        GY, X = zout.grad, xin.detach()
                        Gc = torch.zeros(GY.shape[1], dtype=torch.float64, device='cuda')
                        for k in wins:
                            m = win == k
                            Gc += ((GY[m].double().t() @ X[m].double()) ** 2).sum(1)
                        gfin &= bool(torch.isfinite(Gc).all())
                    if d2 is None:
                        with tm.block('weight_error'):
                            W = wq[p][e]                                   # fm None: the original FP32 checkpoint weight
                            d2_new[name] = {f: ((dq.weight(W, f) - W).double() ** 2).sum(1).cpu().numpy() for f in QUANT}
                        dd = d2_new[name]
                    else:
                        dd = d2[name]
                    with tm.block('table_assembly'):
                        B_[name] = {'fp32': 0.0}
                        for f in QUANT:
                            B_[name][f] = float((Gc.cpu().numpy() / X.shape[1] * dd[f]).sum())
                    rng_fixed = ranges[name]
                    rngs[name] = {s: [float(t.detach().min()), float(t.detach().max())] for s, t in (sx, sy)}
                    with tm.block('act_error'):
                        gstats = {}
                        for s, t in (sx, sy):
                            gt = t.grad.double()
                            G2 = (gt ** 2).sum(0)
                            GS = torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0)
                            gstats[s] = (t.detach(), G2, GS)
                            gfin &= bool(torch.isfinite(G2).all())
                        C_[name] = {'fp32': 0.0}
                        for f in QUANT:
                            cc = 0.0
                            for s in ('x', 'y'):
                                t, G2, GS = gstats[s]
                                err = (dq.act(t, f, rng_fixed[s]) - t).double()
                                mu = err.sum(0) / n
                                cc += float((G2 / n * ((err ** 2).sum(0) - n * mu ** 2)).sum() + (GS * mu ** 2).sum())
                            C_[name][f] = cc
        del wq, hk, cap
        log(f'{label}: layer {l} ({sum(tm.t.values()):.0f}s)')
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    res = {'B': B_, 'C': C_, 'meta': meta, 'observed_ranges': rngs, 'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd,
           'recompute_max_abs_per_layer': recompute_diff[::-1], 'grads_finite': gfin, 'nonfinite': nonfinite,
           'timers': dict(tm.t), 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': peak_cpu_gb()}
    return res, d2_new


def ste_unit_test(dq, ranges):
    """STE forward equals the existing QDQ bitwise; backward = identity; finite, nonzero."""
    import torch
    STEQ = ste_fn()
    torch.manual_seed(0)
    res = []
    name = 'L5.E3.gate_proj'
    r = ranges[name]['x']
    for f in ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1', 'f16', 'bf16'):
        x = (torch.randn(257, 2048, device='cuda') * (r[1] - r[0]) / 4).requires_grad_(True)
        y = STEQ.apply(x, lambda t, f=f: dq.act(t, f, r))
        ref = dq.act(x.detach(), f, r)
        gout = torch.randn_like(y)
        y.backward(gout)
        res.append({'format': f, 'forward_bitwise': bool(torch.equal(y.detach(), ref)), 'backward_identity': bool(torch.equal(x.grad, gout)),
                    'grad_finite_nonzero': bool(torch.isfinite(x.grad).all() and (x.grad != 0).any())})
    return {'cases': res, 'pass': all(c['forward_bitwise'] and c['backward_identity'] and c['grad_finite_nonzero'] for c in res)}


# --------------------------------------------------------------------------- #
def problem_static():
    from run_olmoe_multipoint import problem
    table, meta, names, nodes, elems, rows, flops = problem(MP)
    return names, nodes, elems, flops, rows


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    from run_olmoe_multipoint import resources
    names, nodes, elems, flops, rows = problem_static()
    fa = read(AGGR)
    r = resources(fa, names, elems, flops)
    mp = read(MPT / 'plan.json')
    if sha(AGGR) != mp['configs']['aggressive']['sha256'] or abs(r['R'] - mp['configs']['aggressive']['R_frontier']) > 1e-9:
        raise SystemExit('aggressive config / resources differ')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    used = prior_used()
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'Does one full refresh of the BC proxy in the quantized working state (q0) select a better configuration '
                    'than the static FP32 proxy at the same weight-size and BOPs limits? Offline-time cost of the refresh.',
        'not_this_round': 'no sensitivity-only / error-only ablation, no BC formula change, no router fine-tuning, no '
                          'granularity change, no first-order term, no training, no default integration; not a reproduction of '
                          'GEMQ (MoE-output proxy + router fine-tuning)',
        'budget': {'source': f'{AGGR.relative_to(HERE)} (sha256 {sha(AGGR)[:12]}), exact resources recomputed with '
                             'run_olmoe_multipoint.resources (FP32 calibration routed rows, fixed for every search)',
                   'max_bytes': r['M_bytes'], 'max_bops': r['BOPs'], 'M_rel': r['M_rel'], 'B_rel': r['B_rel'], 'R': r['R'],
                   'why': 'the resource point with the clearest dynamic-route extra damage so far'},
        'fixed': {'nodes': f'{len(names)} searched expert projections (experts with >= 1 FP32 calibration token), per expert x '
                           'gate / up / down', 'candidates': 'GRID11 x GRID11', 'calibration': 'stage-1 10 windows x 128 tokens '
                           f'(sha256 {sha(S1 / "tokens_calibration.npy")[:12]})', 'ranges': f'{RANGES.relative_to(HERE)} (fixed for both tables)',
                  'quantizers': 'run_olmoe_search_final.dev_quant(corrected=True): DevQuant weights (per-output-channel int, '
                                'Universal posit LUTs, f16 / bf16) and the corrected shared int activation QDQ',
                  'other': 'router, attention, norms, embedding, head FP32; 21 unsearched projections bf16 / FP32; BOPs use FP32 '
                           'calibration routed rows for both searches'},
        'static_A': {'table': 'rebuilt here on the FP32 reference: FP32 dynamic forward of the calibration (routes recorded), '
                              'backward of summed CE with the routes constant, BC form of run_olmoe_multipoint.stage_table',
                     'cross_check': 'against runs/olmoe_multipoint_20261015/table.json (built with the pre-fix int activation '
                                    'quantizer): relative differences reported with reasons; the main comparison uses the rebuilt table'},
        'refresh_B': {
            'forward': 'q0 configuration, calibration windows, normal dynamic routing (own ids and weights recorded), q0 weight '
                       'and activation QDQ',
            'backward': 'layer-wise recomputation with the recorded q0 routes as constants (no router-weight gradient, no '
                        'derivative of top-k switching); activation QDQ through an identity STE (forward = the existing QDQ '
                        'value bitwise via a custom autograd Function; backward passes the gradient unchanged); q0 quantized '
                        'weights as constants',
            'weight_term': 'G_c from GY = gradient w.r.t. the pre-y-QDQ linear output and X = the input the linear actually '
                           'consumes (post-x-QDQ); dW = Q_f(W) - W of the original checkpoint W, reused exactly from the static build',
            'activation_term': 'slot x: gate / up use the shared pre-QDQ expert input with its total gradient (sum over both '
                               'consumers through the STE); down uses the pre-QDQ SiLU(g) * u; slot y: pre-QDQ linear output and '
                               'its gradient; err = QDQ_f(t) - t with the fixed ranges; variance and coherent-bias parts as in the static form',
            'reference': 'all errors are relative to the pre-QDQ tensor of the working state; fp32 options cost 0',
            'coverage': 'searched node set fixed; searched experts with no q0 calibration token keep their static rows (recorded); '
                        'experts outside the search stay unsearched even if q0 routes tokens to them; n = q0 routed rows of that expert',
            'one_round': 'one refresh; no second update; no loss-based fallback to q0',
            'claim': 'a working-state refresh of the BC form, not an exact loss increment and not the GEMQ formula'},
        'search': {'solver': 'run_olmoe_search_final.make_solver("lpfirst"): LP -> rounding -> feasibility repair, accepted if the '
                             'gap to the LP bound <= max(1e-3 obj, 1e-6 scale), else the corrected MILP (1 ppm retry, repair), '
                             'time limit 300 s; exact resource check',
                   'pruning': 'exact dominance pruning of each table on its own (run_olmoe_multipoint.nondominated)',
                   'record': 'objective, LP bound, path, options, seconds, resources, differing projections, format distributions'},
        'controls': ['FP32 calibration routes / CE equal the stage-1 caches; static table vs historical table',
                     'per-layer recomputation (with grad, STE) equals the no-grad forward bitwise (static and q0)',
                     'my dynamic forward CE equals run_olmoe_dynamic_route.forward (q0, calibration) and the replay of the recorded '
                     'q0 calibration routes reproduces it',
                     'STE unit test: forward bitwise equal to the QDQ, backward identity, finite and nonzero',
                     'all proxies, gradients and losses finite; every option has a legal format and ranges',
                     'input file hashes identical before and after',
                     'evaluation: replay of q0 / q1 recorded routes bitwise, rerun of the first chunk bitwise, program control '
                     '(aggressive dynamic on old windows run_olmoe_route_dynamic_prop 0-127 reproduces its L11 CE)'],
        'freeze': 'q0, q1, both tables, ranges, sources and code hashes frozen before evaluation data are drawn',
        'evaluation': {'data': f'{N_EVAL} unused WikiText-103 train articles x 4 windows (512 windows); strict boundaries, '
                               'WikiText-2 removed, within-WT103 duplicates removed, near-duplicates (>= 10% shingles) removed, '
                               'every earlier OLMoE article excluded (route_mass_restore, route_dynamic_prop, precision_protection '
                               f'selection and final); article order seed {SEEDS["order"]}, window start seed {SEEDS["start"]}',
                       'eligible_before_length_and_neardup': len(elig),
                       'conditions': 'FP32 dynamic, q0 dynamic, q1 dynamic (each model routes itself; no FP32 route / '
                                     'proportion feedback); GPU FP32, TF32 off, chunks of 128 windows',
                       'primary': 'Delta = NLL(q1) - NLL(q0), article-paired; NLL per article = summed CE of 4 windows / 508',
                       'decision': '95% CI upper < 0: refresh improves; lower > 0: refresh worse; otherwise not confirmed; '
                                   f'equivalent only if the whole 95% CI lies in [-{EPS}, +{EPS}] (pre-fixed); not significant is not equivalent',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'no_feedback': 'no knee re-selection, no search feedback from evaluation data, no extra samples'},
        'timing': 'cold, measured separately: static build (forward / backward / weight errors / activation errors / '
                  'assembly), first search, q0 weight quantization (forward and backward recompute), q0 calibration forward / '
                  'routing, backward / sensitivity, candidate activation errors, refreshed table and second search; peak GPU / '
                  'CPU memory; baseline cost (to q0), extra cost (q0 -> q1), control / evaluation cost separately',
        'not_claimed': 'other resource points, other models, native kernels; BC as an exact loss approximation; a GEMQ '
                       'reproduction; a negative result does not refute every progressive-proxy method',
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_proxy_refresh.py', 'run_olmoe_route_mass_restore.py', 'run_olmoe_multipoint.py',
                                            'run_solver_regression.py', 'run_native_bc_pipeline.py', 'htq_pareto_search.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 完整更新一次 BC proxy 是否有收益', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          f'- 資源上限：aggressive 配置的精確 weight bytes = {r["M_bytes"]:.0f}、BOPs = {r["BOPs"]:.0f}（R {r["R"]:.7f}），BOPs 用 FP32 calibration routed rows，兩次搜尋相同。',
          f'- 固定：{len(names)} 個搜尋投影、GRID11×GRID11、原 10 個 calibration 窗口、固定 ranges、修正後量化器、router／attention／norm FP32。',
          '- A（static）：本輪在 FP32 reference 上重建 BC 表 → LP 優先搜尋 → q0；與歷史表交叉核對。',
          '- B（refresh）：q0 的 calibration dynamic forward（自身路由）→ 路由視為常數、activation QDQ 以 identity STE 反傳 → 依原 BC 形式重算 → 同一搜尋 → q1；只更新一次，不依 loss 回退。',
          '- weight 項：GY = 對 y-QDQ 前輸出的梯度、X = linear 實際吃到的（x-QDQ 後）輸入；ΔW = Q_f(W) − W（原始 W，自 static 精確重用）。',
          '- activation 項：x 槽 gate／up 用共享的 QDQ 前輸入與 total gradient，down 用 QDQ 前 SiLU(g)·u；y 槽用 QDQ 前 linear 輸出；err = QDQ_f(t) − t（固定 ranges），含 variance 與 bias 兩部分；fp32 選項成本為 0。',
          '- q0 無 calibration token 的原搜尋 expert 回退 static 表（記錄）；搜尋集合不擴張；n 為 q0 的 routed rows。',
          '- 搜尋：LP → 取整 → 修復，差距門檻未過才退回修正後 MILP；兩表各自精確支配剪枝。',
          f'- 評估：{N_EVAL} 篇未用過的 WikiText-103 文章 × 4 窗口；seed 22101／22102／22111；FP32、q0、q1 皆正常 dynamic。',
          '- 主要終點：Δ = NLL(q1) − NLL(q0)；95% CI 上界 < 0 → 改善，下界 > 0 → 較差，其他 → 無法確認；整段 95% CI 落在 ±0.002 才算等價。',
          '- 不做：部分更新消融、改 BC、router 微調、改粒度、一階項、整合預設；不宣稱重現 GEMQ。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', inputs)
    log(f'plan written; budget M_rel {r["M_rel"]:.7f} B_rel {r["B_rel"]:.7f}; eval eligible {len(elig)}')


def input_files():
    return [AGGR, RANGES, MP / 'table.json', MP / 'table_meta.json', S1 / 'tokens_calibration.npy', S1 / 'routes_manifest.json',
            MPT / 'plan.json', RDP / 'tokens.npy', RDP / 'results.json'] + mr.WT103 + mr.WT2 + \
           [MRD / 'segments.json', RDP / 'segments.json', RPP / 'selection_segments.json', RPP / 'final_segments.json']


def prior_used():
    return {s['content_sha256'] for f in (MRD / 'segments.json', RDP / 'segments.json', RPP / 'selection_segments.json', RPP / 'final_segments.json')
            for s in read(f)}


def static(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    from run_olmoe_fixed_route import load_routes
    import olmoe_common as oc
    dr.gpu_setup()
    if (out / 'table_static.json').exists():
        raise SystemExit('exists')
    (out / 'work').mkdir(exist_ok=True)
    names, nodes, elems, flops, rows = problem_static()
    ranges = read(RANGES)
    t0 = time.perf_counter()
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    t_dq = time.perf_counter() - t0
    res, d2 = build_table(out, None, dq, ranges, names, log, 'static')
    # controls vs the stage-1 caches and the historical table
    _, ids_s1, ws_s1 = load_routes(S1, 'calibration', oc.config(), 'cpu')
    s1m = read(S1 / 'routes_manifest.json')['calibration']
    hist, hmeta = read(MP / 'table.json'), read(MP / 'table_meta.json')['experts']
    relB = [abs(res['B'][n][f] - hist['B'][n][f]) / max(abs(hist['B'][n][f]), 1e-30) for n in names for f in hist['B'][n] if f != 'fp32']
    relC = {f: max(abs(res['C'][n][f] - hist['C'][n][f]) / max(abs(hist['C'][n][f]), 1e-30) for n in names) for f in hist['C'][names[0]] if f != 'fp32'}
    ctl = {'routes_equal_stage1_cache': bool(np.array_equal(res['routes']['ids'], ids_s1) and np.array_equal(res['routes']['weights'], ws_s1)),
           'ce_equal_stage1_dynamic': float(np.max(np.abs(np.array(res['ce_forward']) - np.array(s1m['summed_ce_dynamic'])))),
           'recompute_max_abs': max(res['recompute_max_abs_per_layer']), 'grads_finite': res['grads_finite'], 'nonfinite': len(res['nonfinite']),
           'rows_equal_historical': all(res['meta'][k]['routed_rows'] == hmeta[k]['routed_rows'] for k in res['meta']),
           'observed_ranges_equal_fixed': all(res['observed_ranges'][n] == ranges[n] for n in names),
           'all_searched_observed': all(v['routed_rows'] > 0 for v in res['meta'].values()),
           'B_vs_historical_max_rel': max(relB), 'C_vs_historical_max_rel_by_format': relC,
           'C_note': 'int activation formats: historical table used the pre-fix GPU int QDQ (CPU 0-dim scale division, 1-5 '
                     'code flips per window); posit / f16 / bf16 unaffected'}
    write(out / 'table_static.json', {'B': res['B'], 'C': res['C'], 'meta': res['meta'], 'state': 'FP32 reference'})
    np.savez(out / 'weight_error_d2.npz', **{f'{n}|{f}': v for n, d in d2.items() for f, v in d.items()})
    np.savez(out / 'static_calibration_routes.npz', **res['routes'])
    write(out / 'static_checks.json', ctl)
    merge_timing(out, 'static', {'devquant_init_s': t_dq, **res['timers'], 'peak_gpu_gb': res['peak_gpu_gb'], 'peak_cpu_gb': res['peak_cpu_gb']})
    log(json.dumps({k: v for k, v in ctl.items() if k != 'C_note'}))


def search(out, log, which):
    from run_native_bc_pipeline import GRID11
    from run_olmoe_multipoint import nondominated, pruned, resources
    from run_olmoe_search_final import make_solver
    from mp_auto import config_hash
    tf = out / f'table_{which}.json'
    of = out / f'search_{which}.json'
    if of.exists():
        raise SystemExit('exists')
    pl = read(out / 'plan.json')
    names, nodes, elems, flops, rows = problem_static()
    t = read(tf)
    t0 = time.perf_counter()
    bc = lambda n, w, a: t['B'][n][w] + t['C'][n][a]
    keep = nondominated(names, bc)
    t_prune = time.perf_counter() - t0
    recs = []
    (out / 'logs').mkdir(exist_ok=True)
    sel = make_solver('lpfirst', recs, out / 'logs', tag=f'proxy_{which}')
    t1 = time.perf_counter()
    fm, obj, status = sel(nodes, GRID11, GRID11, pruned(bc, keep), elems, flops, pl['budget']['max_bytes'], pl['budget']['max_bops'])
    t_solve = time.perf_counter() - t1
    fm = {n: fm[n] for n in names}
    cfgname = 'q0' if which == 'static' else 'q1'
    (out / 'configs').mkdir(exist_ok=True)
    write(out / f'configs/{cfgname}.json', fm)
    r = resources(fm, names, elems, flops)
    res = {'config': f'configs/{cfgname}.json', 'sha256': sha(out / f'configs/{cfgname}.json'), 'config_hash': config_hash(fm),
           'objective': obj, 'objective_recomputed': float(sum(bc(n, fm[n]['w'], fm[n]['a']) for n in names)), 'status': status,
           'record': recs[-1], 'options_kept': sum(len(v) for v in keep.values()), 'options_total': 121 * len(names),
           'resources': r, 'within_budget': r['M_bytes'] <= pl['budget']['max_bytes'] and r['BOPs'] <= pl['budget']['max_bops'],
           'budget_use': {'size': r['M_bytes'] / pl['budget']['max_bytes'], 'bops': r['BOPs'] / pl['budget']['max_bops']},
           'weight_formats': dict(Counter(v['w'] for v in fm.values())), 'act_formats': dict(Counter(v['a'] for v in fm.values())),
           'all_formats_legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in fm.values())}
    write(of, res)
    merge_timing(out, f'search_{which}', {'pruning_s': t_prune, 'solve_s': t_solve, 'path': recs[-1].get('path'), 'peak_cpu_gb': peak_cpu_gb()})
    log(json.dumps({k: res[k] for k in ('objective', 'status', 'within_budget', 'budget_use')} | {'path': recs[-1].get('path'), 'seconds': t_solve}))


def refresh(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'table_refresh.json').exists():
        raise SystemExit('exists')
    names, nodes, elems, flops, rows = problem_static()
    ranges = read(RANGES)
    sq = read(out / 'search_static.json')
    if sha(out / 'configs/q0.json') != sq['sha256']:
        raise SystemExit('q0 changed')
    fm = read(out / 'configs/q0.json')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    z = np.load(out / 'weight_error_d2.npz')
    d2 = defaultdict(dict)
    for k in z.files:
        n, f = k.split('|'); d2[n][f] = z[k]
    st = read(out / 'table_static.json')
    res, _ = build_table(out, fm, dq, ranges, names, log, 'refresh', d2=d2)
    # coverage: searched experts unobserved under q0 keep the static rows
    B_, C_, fallback = {}, {}, []
    for n in names:
        node = '.'.join(n.split('.')[:2])
        if res['meta'].get(node, {'routed_rows': 0})['routed_rows'] == 0:
            B_[n], C_[n] = st['B'][n], st['C'][n]
            fallback.append(n)
        else:
            B_[n], C_[n] = res['B'][n], res['C'][n]
    write(out / 'table_refresh.json', {'B': B_, 'C': C_, 'meta': res['meta'], 'fallback_static_rows': fallback, 'state': 'q0 working state'})
    np.savez(out / 'q0_calibration_routes.npz', **res['routes'])
    # controls: own dynamic forward vs run_olmoe_dynamic_route.forward, and replay of the recorded routes
    toks = np.load(S1 / 'tokens_calibration.npy')
    rd = dr.forward(toks, fm, dq, ranges, record=True, log=lambda m: None, label='q0 dynamic (dr.forward)', chunk=len(toks))
    rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(res['routes']), log=lambda m: None, label='q0 replay', chunk=len(toks))
    ste = ste_unit_test(dq, ranges)
    ctl = {'recompute_max_abs': max(res['recompute_max_abs_per_layer']), 'recompute_per_layer': res['recompute_max_abs_per_layer'],
           'own_forward_ce_vs_dr_forward_max_abs': float(np.max(np.abs(np.array(res['ce_forward']) - np.array(rd['ce'])))),
           'own_routes_equal_dr_forward': bool(np.array_equal(res['routes']['ids'], rd['routes']['ids']) and np.array_equal(res['routes']['weights'], rd['routes']['weights'])),
           'replay_ce_max_abs': float(np.max(np.abs(np.array(rp['ce']) - np.array(res['ce_forward'])))),
           'grads_finite': res['grads_finite'], 'nonfinite': len(res['nonfinite']), 'ste_unit_test': ste,
           'fallback_count': len(fallback), 'fallback_nodes': fallback,
           'q0_observed_experts': sum(1 for v in res['meta'].values() if v['routed_rows'] > 0),
           'rows_fp32_vs_q0_total': [int(sum(read(out / 'table_static.json')['meta'][k]['routed_rows'] for k in res['meta'])),
                                     int(sum(v['routed_rows'] for v in res['meta'].values()))],
           'calibration_ce_q0_nat_per_token': float(np.sum(res['ce_forward']) / (len(toks) * TP))}
    ctl['pass'] = bool(ctl['recompute_max_abs'] == 0 and ctl['own_forward_ce_vs_dr_forward_max_abs'] == 0 and ctl['own_routes_equal_dr_forward']
                       and ctl['replay_ce_max_abs'] == 0 and ctl['grads_finite'] and ctl['nonfinite'] == 0 and ste['pass'])
    write(out / 'refresh_checks.json', ctl)
    merge_timing(out, 'refresh', {**res['timers'], 'weight_error_reused_from_static': True, 'peak_gpu_gb': res['peak_gpu_gb'],
                                  'peak_cpu_gb': res['peak_cpu_gb'], 'controls_s': {'dr_forward': rd['seconds'], 'replay': rp['seconds']}})
    log(json.dumps({k: ctl[k] for k in ('recompute_max_abs', 'own_forward_ce_vs_dr_forward_max_abs', 'replay_ce_max_abs', 'fallback_count', 'pass')}))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    q0, q1 = read(out / 'configs/q0.json'), read(out / 'configs/q1.json')
    names = sorted(q0)
    diff = [n for n in names if q0[n] != q1[n]]
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'q0_sha256': sha(out / 'configs/q0.json'), 'q1_sha256': sha(out / 'configs/q1.json'),
           'identical': not diff, 'differing_projections': len(diff),
           'weight_changed': sum(q0[n]['w'] != q1[n]['w'] for n in names), 'act_changed': sum(q0[n]['a'] != q1[n]['a'] for n in names),
           'weight_transitions': dict(Counter(f'{q0[n]["w"]}->{q1[n]["w"]}' for n in names if q0[n]['w'] != q1[n]['w'])),
           'act_transitions': dict(Counter(f'{q0[n]["a"]}->{q1[n]["a"]}' for n in names if q0[n]['a'] != q1[n]['a'])),
           'diff_by_layer': dict(Counter(int(n.split('.')[0][1:]) for n in diff)), 'diff_by_proj': dict(Counter(n.split('.')[2] for n in diff)),
           'hashes': {f: sha(out / f) for f in ['table_static.json', 'table_refresh.json', 'search_static.json', 'search_refresh.json',
                                                 'static_checks.json', 'refresh_checks.json', 'plan.json']},
           'ranges_sha256': sha(RANGES), 'code': {f: sha(HERE / f) for f in ['run_olmoe_proxy_refresh.py'] + dr.SCRIPTS}}
    if not read(out / 'refresh_checks.json')['pass']:
        raise SystemExit('refresh controls failed: not freezing')
    write(out / 'freeze.json', res)
    log(json.dumps({k: res[k] for k in ('identical', 'differing_projections', 'weight_changed', 'act_changed')}))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    used = prior_used()
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    if len(prev) != len(used):
        raise SystemExit('previous articles not all found')
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != read(out / 'plan.json')['evaluation']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(a['body_norm']) for a in prev]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    segs, toks, acc_sh, skips = [], [], [], Counter()
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
            skips['near_duplicate'] += 1; continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']], 'title': a['title'],
                     'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids), 'window_tokens': [st, st + WPA * SEQ],
                     'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                     'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca}, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
        if len(segs) == N_EVAL:
            break
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'), 'n': len(segs),
                                     'skips': dict(skips), 'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json')})
    log(f'eval data frozen: {len(segs)} articles; skips {dict(skips)}')


def eval_(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'eval.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'freeze.json')
    if sha(out / 'configs/q0.json') != fz['q0_sha256'] or sha(out / 'configs/q1.json') != fz['q1_sha256'] or \
            sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('frozen inputs changed')
    toks = np.load(out / 'tokens.npy')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    torch.cuda.reset_peak_memory_stats()
    runs = {'fp32': dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')}
    for c in ('q0', 'q1'):
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
    ctl = {}
    for c in ('q0', 'q1'):
        fm = read(out / f'configs/{c}.json')
        rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs[c]['routes']), log=log, label=f'replay {c}')
        ctl[f'replay_{c}'] = dr.same(rp, runs[c]); ctl[f'replay_{c}_s'] = rp['seconds']
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/q1.json'), dq, ranges, record=True, log=log, label='rerun q1')
    ctl['rerun_q1_first_chunk'] = {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['q1']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['q1']['hidden_sha'][:CTRL])))}
    old = np.load(RDP / 'tokens.npy')[:CTRL]
    pc = dr.forward(old, read(AGGR), dq, ranges, record=False, log=log, label='program control')
    ctl['program_control_ce_max_abs_vs_previous_L11'] = float(np.max(np.abs(np.array(pc['ce']) - np.array(read(RDP / 'results.json')['ce']['L11'][:CTRL]))))
    ctl['finite'] = all(r['finite_ce'] for r in runs.values())
    ctl['routes_differ_between_q_and_fp32'] = {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in ('q0', 'q1')}
    ctl['audit'] = {c: runs[c]['audit'] for c in ('q0', 'q1')}
    ctl['pass'] = bool(ctl['replay_q0']['ce_max_abs'] == 0 and ctl['replay_q1']['ce_max_abs'] == 0
                       and ctl['replay_q0']['hidden_bitwise_windows'] == len(toks) and ctl['replay_q1']['hidden_bitwise_windows'] == len(toks)
                       and ctl['rerun_q1_first_chunk']['ce_max_abs'] == 0 and ctl['rerun_q1_first_chunk']['hidden_bitwise_windows'] == CTRL
                       and ctl['program_control_ce_max_abs_vs_previous_L11'] == 0 and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl,
                              'seconds': {c: r['seconds'] for c, r in runs.items()}, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                              'environment': dr.env_info()})
    merge_timing(out, 'eval', {'fp32_s': runs['fp32']['seconds'], 'q0_s': runs['q0']['seconds'], 'q1_s': runs['q1']['seconds'],
                               'controls_s': {'replay_q0': ctl['replay_q0_s'], 'replay_q1': ctl['replay_q1_s'], 'rerun_q1': r2['seconds'], 'program_control': pc['seconds']}})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    t0 = time.time()
    ev = read(out / 'eval.json')
    if not ev['controls']['pass']:
        raise SystemExit('evaluation controls failed')
    nA = N_EVAL
    A = {c: np.array(v).reshape(nA, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
    d = A['q1'] - A['q0']
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, nA, (BOOT, nA))
    st = lambda x: {'mean': float(x.mean()), 'ci95': np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1)),
                    'articles_positive': int((x > 0).sum())}
    res = {'n_articles': nA, 'delta_q1_minus_q0': st(d), 'damage': {c: st(A[c] - A['fp32']) for c in ('q0', 'q1')}}
    lo, hi = res['delta_q1_minus_q0']['ci95']
    res['decision'] = 'refresh_improves' if hi < 0 else 'refresh_worse' if lo > 0 else 'not_confirmed'
    res['equivalent_within_eps'] = bool(-EPS <= lo and hi <= EPS)
    tokens = nA * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for c, v in ev['ce'].items()}
    res['freeze'] = read(out / 'freeze.json')
    res['search'] = {w: {k: read(out / f'search_{w}.json')[k] for k in ('objective', 'status', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats')}
                     | {'path': read(out / f'search_{w}.json')['record'].get('path'), 'lower_bound': read(out / f'search_{w}.json')['record'].get('lower_bound')}
                     for w in ('static', 'refresh')}
    st_, rf = read(out / 'table_static.json'), read(out / 'table_refresh.json')
    q0, q1 = read(out / 'configs/q0.json'), read(out / 'configs/q1.json')
    names = sorted(q0)
    bc = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    res['cross_objectives'] = {'static_table': {'q0': bc(st_, q0), 'q1': bc(st_, q1)}, 'refresh_table': {'q0': bc(rf, q0), 'q1': bc(rf, q1)}}
    after = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    with open(out / 'per_article.csv', 'w') as fh:
        segs = read(out / 'segments.json')
        rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, 'delta_q1_minus_q0': float(d[i])} for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    res['timing'] = read(out / 'timing.json')
    res['analysis_seconds'] = time.time() - t0
    write(out / 'analysis.json', res)
    log(json.dumps({'delta': res['delta_q1_minus_q0'], 'decision': res['decision'], 'equivalent': res['equivalent_within_eps']}))


def main():
    ap = argparse.ArgumentParser()
    stages = ['plan', 'static', 'search_static', 'refresh', 'search_refresh', 'freeze', 'data', 'eval', 'analyze']
    ap.add_argument('--stage', choices=stages, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    fn = {'search_static': lambda o, l: search(o, l, 'static'), 'search_refresh': lambda o, l: search(o, l, 'refresh'), 'eval': eval_}.get(a.stage, globals().get(a.stage))
    fn(a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
