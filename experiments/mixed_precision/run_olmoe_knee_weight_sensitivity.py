#!/usr/bin/env python3
"""OLMoE normal knee: weight-sensitivity-only refresh (W), next to the frozen A / B / C (and D4 / D5 / E / E5 for
diagnostics). Pre-registered in plan.json / plan.md. Same BC criterion; only the idea of re-collecting statistics in the
quantized working state is borrowed from GEMQ (no output-error criterion, no weight-only quantization).

W: for all 3051 searched projections, the BC weight term with q0_knee sensitivities and static weight errors,
  B_i(f) = sum_c S_i,c^q0 D_i,c^static(f), S_c = (sum over windows of row sums of (GY^T X)^2) / n_in, GY = gradient w.r.t. the
  pre-output-QDQ linear output, X = the post-input-QDQ input; the activation term entirely static (A's rows).
  Expected table: weight rows = historical C (= historical B) weight rows; activation rows = static.
The refresh runs the q0_knee dynamic forward and the layer-wise backward (routes constant, identity STE) but retains
gradients only for the linear outputs needed by the weight term; no activation gradient statistics, candidate errors or
activation assembly. W+A search at the exact knee limits (weight and activation formats both re-assigned).

Stages (one --out-dir): plan, controls, refresh (--variant B|C|W --rep k), e2e, timing (--phase refresh|e2e), check,
freeze, data, eval, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
import os
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
KS, KA, K5, KE = ke.KS, ke.KA, ke.K5, HERE / 'runs/olmoe_knee_error_only_20261022'
K5E = HERE / 'runs/olmoe_knee_low5_error_only_20261022'
PR, S1, MP, RANGES, CALIB = ss.PR, ss.S1, ss.MP, ss.RANGES, ss.CALIB
STATIC_TABLE, D2, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.D2, ks.ACT_CACHE, ks.ACT_INDEX
Q0K, KNEE = KS / 'knee/q0_knee.json', KS / 'knee/knee.json'
C_STATS = KA / 'work/runs/refresh_C_r1'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
BOOT, NI_MARGIN, TOL_REL, NCMP = 10000, 0.0005, 1e-12, 3
CI_CORR = [100 * 0.05 / (2 * NCMP), 100 * (1 - 0.05 / (2 * NCMP))]      # Bonferroni over the 3 primary comparisons: 98.33%
SEEDS = {'order': 23001, 'start': 23002, 'bootstrap': 23011}
VARIANTS = ('B', 'C', 'W')
REFRESH_SCHEDULE = [('B', 1), ('C', 1), ('W', 1), ('C', 2), ('W', 2), ('B', 2), ('W', 3), ('B', 3), ('C', 3)]
FROZEN = {'A': KS / 'configs/A.json', 'B': KS / 'configs/B.json', 'C': KA / 'configs/C.json', 'D4': KS / 'configs/D.json',
          'D5': K5 / 'configs/D5.json', 'E': KE / 'configs/E.json', 'E5': K5E / 'configs/E5.json'}
FROZEN_TABLES = {'B': KS / 'tables/B.json', 'C': KA / 'tables/C.json'}
WTAPS = {'gate_proj': ('xg', 'zg'), 'up_proj': ('xu', 'zu'), 'down_proj': ('hd', 'zd')}     # (post-input-QDQ X, pre-output-QDQ linear output)
CATEGORIES = ks.CATEGORIES


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_knee_weight_sensitivity.py', 'run_olmoe_knee_error_only.py', 'run_olmoe_knee_low5_error_only.py',
                                                  'run_olmoe_knee_selective_sensitivity.py', 'run_olmoe_knee_all_sensitivity.py'] + ss.CODE + dr.SCRIPTS))}


def history():
    f5e = read(K5E / 'freeze.json')
    tB, tC = read(KS / 'tables/B.json'), read(KA / 'tables/C.json')
    res = {'earlier_rounds': ke.history(),
           'low5_error_round_configs': {c: sha(HERE / v['path']) == v['sha256'] for c, v in f5e['configs'].items()},
           'low5_error_round_check': read(K5E / 'check.json')['pass'], 'error_round_check': read(KE / 'check.json')['pass'],
           'historical_B_weight_rows_equal_C': tB['B'] == tC['B'],
           'code_unchanged': {k: sha(HERE / k) == f5e['code'][k] for k in ('run_olmoe_knee_low5_error_only.py', 'run_olmoe_knee_error_only.py',
                                                                            'run_olmoe_knee_selective_sensitivity.py', 'run_olmoe_selective_sensitivity.py')},
           'c_stats_present': (C_STATS / 'target_stats.npz').exists()}
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['pass'] = flat(res)
    return res


def run_record(tm, t0, extra):
    import torch
    total = time.perf_counter() - t0
    timers = dict(tm.t)
    cats = {c: sum(timers.get(k, 0.0) for k in ks_) for c, ks_ in CATEGORIES.items()}
    cats['unclassified'] = total - sum(timers.values())
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    return {'total_s': total, 'method_s': total - cats['controls'], 'timers': timers, 'categories': cats,
            'process_start_to_run_s': time.time() - total - T_PROC, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'peak_cpu_gb': ss.peak_cpu_gb(), 'gpu': torch.cuda.get_device_name(0), 'loadavg_after': load, **extra}


def work_check(variant, timers):
    """W: backward and weight sensitivity present, no activation statistics / errors / assembly / activation-cache read."""
    t = lambda k: timers.get(k, 0.0)
    if variant == 'W':
        ok = t('backward') > 0 and t('weight_sens') > 0 and t('act_grad_stats') == 0 and t('act_error') == 0 and t('act_assembly') == 0 and t('cache_read') == 0
    elif variant == 'C':
        ok = t('act_grad_stats') > 0 and t('act_error') == 0 and t('act_assembly') > 0 and t('cache_read') > 0
    else:
        ok = t('act_grad_stats') > 0 and t('act_error') > 0 and t('act_assembly') == 0 and t('cache_read') == 0
    return {k: t(k) for k in ('backward', 'weight_sens', 'act_grad_stats', 'act_error', 'act_assembly', 'cache_read', 'cache_read_d2')} | {'as_expected': ok}


# --------------------------------------------------------------------------- #
# W: weight-statistics-only backward
# --------------------------------------------------------------------------- #
def experts_tap_w(w, x2d, idx, rw, act, want, cap, E=NE):
    """run_olmoe_proxy_refresh.experts_tap op order; retain_grad only on the linear outputs of the target projections (weight term);
    the activation slots are not tapped. retain_grad changes no value; the backward path is unchanged."""
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
            loc = {'xg': xg, 'xu': xu, 'zg': zg, 'zu': zu, 'hd': hd, 'zd': zd}
            for p in want[e]:
                loc[WTAPS[p][1]].retain_grad()
            cap[e] = {**{k: loc[k] for p in want[e] for k in WTAPS[p]}, 'tok': tok}
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def layer_tap_w(w, h, cfg, cos, sin, idx, rw, act, want, cap):
    import olmoe_common as oc
    B, T, H = h.shape
    x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
    z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, H)
    return x + experts_tap_w(w, z, idx, rw, act, want, cap).view(B, T, H)


def build_w(fm, dq, ranges, names, tm, log, d2z, static_tab, stats_out=None):
    """q0_knee forward (own dynamic routes) and layer-wise backward (recorded routes constant, identity STE); weight rows from the q0
    weight sensitivities and the static weight errors (run_olmoe_knee_selective_sensitivity.build op order); activation rows static."""
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
    want_all = defaultdict(dict)
    for n in names:
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
    B_, meta = {}, {}
    recompute, gn_sha = [], []
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = ss.load_layer(ck, l, fm, dq, ranges, tm)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = layer_tap_w(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(hk, STEQ), want, cap)
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
                meta[node] = {'routed_rows': int(len(tok)), 'windows': len(wins)}
                for p in want[e]:
                    name = f'{node}.{p}'
                    kX, kG = WTAPS[p]
                    with tm.block('weight_sens'):
                        GY, X = c[kG].grad, c[kX].detach()
                        Gc = torch.zeros(GY.shape[1], dtype=torch.float64, device='cuda')
                        for k in wins:
                            m = win == k
                            Gc += ((GY[m].double().t() @ X[m].double()) ** 2).sum(1)
                        gcn = Gc.cpu().numpy()
                        gfin &= bool(np.isfinite(gcn).all())
                    with tm.block('cache_read_d2'):
                        dd = {f: d2z[f'{name}|{f}'] for f in QUANT}
                    with tm.block('table_assembly'):
                        B_[name] = {'fp32': 0.0}
                        for f in QUANT:
                            B_[name][f] = float((gcn / X.shape[1] * dd[f]).sum())
                    if stats_out is not None:
                        with tm.block('controls'):
                            stats_out[name] = {'Gc': gcn, 'n_in': int(X.shape[1])}
        del wq, hk, cap
        log(f'W: layer {l} ({sum(tm.t.values()):.0f}s)')
    fallback = [n for n in names if meta['.'.join(n.split('.')[:2])]['routed_rows'] == 0]
    with tm.block('table_assembly'):
        C_ = {n: static_tab['C'][n] for n in names}
        for n in fallback:
            B_[n] = static_tab['B'][n]
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not math.isfinite(v)]
    return {'B': B_, 'C': C_, 'meta': meta, 'routes': {'ids': ids_rec, 'weights': w_rec}, 'ce_forward': ce_fwd, 'hidden_sha': hidden_sha,
            'input_grad_sha_per_layer': gn_sha[::-1], 'recompute_max_abs_per_layer': recompute[::-1], 'grads_finite': gfin, 'nonfinite': nonfinite,
            'fallback_static_rows': fallback}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    hist = history()
    names, nodes, elems, flops, rows = pr.problem_static()
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = ss.input_files() + list(FROZEN.values()) + list(FROZEN_TABLES.values()) + [
        KS / f for f in ('freeze.json', 'check.json', 'controls.json', 'knee/knee.json', 'knee/q0_knee.json', 'work/q0_knee_calibration_routes_dr_forward.npz',
                         'work/runs/refresh_B_r1/run.json')] + [KA / 'freeze.json', KA / 'check.json', KE / 'freeze.json', K5 / 'freeze.json'] + [
        K5E / f for f in ('freeze.json', 'check.json', 'eval.json', 'tokens.npy')] + [C_STATS / 'target_stats.npz', C_STATS / 'target_stats_meta.json']
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'At the normal knee, with the W+A mixed-precision search, does refreshing only the BC weight-term sensitivities in q0_knee\'s '
                    'working state (W) improve the static BC, and how does it compare with also refreshing activation sensitivities (C) and errors (B)?',
        'background': 'keeps the BC criterion; borrows from GEMQ only the idea of re-collecting statistics in the quantized working state (no '
                      'output-error criterion, no weight-only quantization); not a GEMQ reproduction',
        'not_this_round': 'no <= 4 / 5-bit selection, no BC formula change, no fine-tuning or parameter update, no new knee, no second refresh, no '
                          'loss-based fallback, no new search algorithm, no default-pipeline change',
        'fixed': {'history': hist, 'frozen_configs': {c: str(p.relative_to(HERE)) for c, p in FROZEN.items()}, 'limits': read(KNEE)['limits'],
                  'same': 'checkpoint and router, calibration, ranges, quantizer, q0_knee working state, 3051 searched projections, GRID11 x GRID11, '
                          'resource model, LP-first search, exact knee limits'},
        'W': {'weight': 'B_i(f) = sum_c S_i,c^q0 D_i,c^static(f); S_c = sum over calibration windows of the per-output-channel row sums of (GY^T X)^2, '
                        'divided by n_in (GY: gradient w.r.t. the pre-output-QDQ linear output; X: the post-input-QDQ input); D_static = weight-error cache',
              'activation': 'the whole activation term static: A\'s rows (static sensitivities and static errors)',
              'expected_table': 'weight rows = historical C weight rows (= historical B weight rows, checked); activation rows = static',
              'scope': 'all 3051 searched projections; fallback for experts without q0 tokens = static rows (listed)',
              'gradient_semantics': 'q0_knee calibration forward with its own dynamic routes; layer-wise backward with the recorded routes constant and '
                                    'the identity STE through activation QDQ; no router or top-k derivative; no optimizer',
              'implementation': 'the full backward runs through every expert; retain_grad only on the linear outputs (zg / zu / zd) of the weight term; '
                                'no activation gradient statistics, candidate activation errors, activation assembly or activation-cache read '
                                '(the existing build() would still compute them, so a dedicated path is used)',
              'search': 'W+A: weight and activation formats both re-assigned; activation costs fixed does not mean activation formats fixed; one LP-first '
                        'search at the exact knee limits; LP bound, objective, gaps; A / B / C / D4 / D5 / E / E5 / W cross scores on the W table with '
                        'feasibility; a cheaper known configuration is disclosed; the search approximation is never claimed excluded'},
        'controls': {'1_inputs': 'A / B / C (and D4 / D5 / E / E5), q0, inputs and caches equal their records',
                     '2_weight_rows': 'every W weight row bitwise equals the historical C (= B) weight row',
                     '3_activation_rows': 'every W activation row bitwise equals A',
                     '4_forward': 'forward CE, hidden sha256 and routes equal an independent q0_knee forward record',
                     '5_gradients': 'per-layer input-gradient sha256 equal the historical B record; per-projection weight sensitivity G_c bitwise equal '
                                    'the C run\'s saved statistics',
                     '6_recompute': f'independent numpy recomputation of the weight rows from the saved G_c and the weight errors (relative <= {TOL_REL})',
                     '7_work': 'W runs: backward and weight sensitivity present; activation gradient statistics, candidate errors, assembly and activation '
                               'cache read absent',
                     '8_validity': 'fallback, finite, legal, exact limits',
                     '9_repeats': 'W repeats identical tables / configurations (main = repeat 1); new B / C runs equal the frozen ones; end-to-end W '
                                  'reproduces the static table, weight-error cache, knee and W configuration'},
        'timing': {'refresh': f'B / C / W, 3 each, interleaved {REFRESH_SCHEDULE}; one process per run; CUDA synchronization; no page-cache dropping; '
                              'load average recorded (interleaving does not fully remove background-load effects)',
                   'e2e': 'one end-to-end W run: static BC (with the weight-error cache) -> auto knee -> W -> search (single measurement)',
                   'categories': CATEGORIES, 'separate': 'controls, repeats, evaluation; historical times and stage-sum estimates labelled'},
        'freeze': 'plan, code, inputs, tables, configurations and hashes frozen before any evaluation article is drawn',
        'evaluation': {'data': f'{N} unused WikiText-103 train articles x {WPA} windows of {SEQ}; strict boundaries, duplicate and near-duplicate '
                               'exclusion; every earlier manifest (incl. the E5 round) excluded by scanning runs/**/*segments*.json',
                       'excluded_previous_wt103': len(used), 'manifests_scanned': manifests, 'eligible_before_length_and_neardup': len(elig),
                       'seeds': SEEDS, 'conditions': 'FP32, A, B, C, W on the same new articles; original router; each its own dynamic route',
                       'primary': {'W - A': 'corrected upper < 0 -> W better; lower > 0 -> W worse; otherwise direction not confirmed',
                                   'W - C': 'corrected upper < 0 -> W better; lower > 0 -> W worse; otherwise direction not confirmed',
                                   'W - B': f'corrected upper < +{NI_MARGIN} -> non-inferior; lower > +{NI_MARGIN} -> beyond the margin; otherwise insufficient'},
                       'interpretation': {'W - A': 'the overall effect of refreshing only the weight sensitivities',
                                          'W - C': 'with weight sensitivities refreshed, the difference made by also refreshing activation sensitivities',
                                          'W - B': 'with weight sensitivities refreshed, the difference made by also refreshing activation sensitivities and '
                                                   'errors (not attributable to the errors alone)'},
                       'intervals': f'mean, 95% CI, Bonferroni {100 - 2 * CI_CORR[0]:.2f}% CI ({NCMP} comparisons); decisions on the corrected interval',
                       'secondary': 'B - A, C - A (95% CI)',
                       'bootstrap': f'{BOOT} paired article bootstraps, seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'cautions': 'no extra articles, seed or setting changes; non-inferior is not better than static; not confirmed is not equal'},
        'report_questions': ['1 W improves the static BC?', '2 W keeps B\'s quality?', '3 evidence for also refreshing activation sensitivities (vs C)?',
                             '4 update and total offline time saved', '5 worth keeping as an optional step?'],
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    if not hist['pass']:
        raise SystemExit('history hashes differ: investigate first')
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 正常 knee 的只刷新 weight sensitivity（W）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- 背景：維持原 BC criterion，只借用 GEMQ「在量化後工作狀態重新取得統計」的想法；不是 GEMQ 復現，不改 output-error criterion，也不改成 weight-only 量化。',
          '- W：全部 3051 個投影的 weight 項用 q0_knee 的 sensitivity（原 per-channel 定義）乘 static ΔW²；activation 項整項沿用 static；預期表 = 歷史 C 的 weight 列 + static 的 activation 列。',
          '- 實作：完整 q0 forward 與逐層 backward（固定路由、STE），只對 linear 輸出保留梯度；不做 activation 梯度統計、候選誤差、組表與快取讀取。',
          '- 搜尋：W+A，在 knee 精確上限重新搜尋一次，weight 與 activation 格式都可重新分配。',
          '- 控制：hash、weight 列 = 歷史 C、activation 列 = A、forward 一致、輸入梯度與 G_c 和 C 一致、numpy 重算、無 activation 統計工作、合法與上限、重複一致、端到端重現。',
          f'- 計時：B／C／W 各 3 次交錯（{" ".join(f"{v}{r}" for v, r in REFRESH_SCHEDULE)}）；W 端到端 1 次；記錄 load average。',
          f'- 評估：{N} 篇未用過的 WikiText-103 文章 × 4（排除 {len(used)} 篇歷史文章）；FP32、A、B、C、W；seed 23001／23002／23011。',
          f'- 主要：W − A、W − C（校正上界 < 0 → W 較好；下界 > 0 → W 較差）、W − B（校正上界 < +{NI_MARGIN} 非劣；下界 > +{NI_MARGIN} 超過界線）；Bonferroni {100 - 2 * CI_CORR[0]:.2f}%。次要：B − A、C − A（95%）。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log(f'plan written; history ok {hist["pass"]}; previous WT103 {len(used)}; eligible {len(elig)}')


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
           'page_cache_residency': ss.resident_fraction(ks.watch_files()), 'loadavg': os.getloadavg(), 'seconds': time.perf_counter() - t0}
    ctl['pass'] = bool(rd['finite_ce'] and all(ctl['equal_previous_record'].values()) and ctl['ste_unit_test']['pass'] and ctl['history']['pass'])
    write(out / 'controls.json', ctl)
    log(json.dumps(dr.clean({'equal_previous_record': ctl['equal_previous_record'], 'pass': ctl['pass']})))


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
    load_before = os.getloadavg()
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
    stats = None
    if variant == 'W':
        with tm.block('input_read'):
            static_tab = read(STATIC_TABLE)
        stats = {} if rep == 1 else None
        r = build_w(fm, dq, ranges, names, tm, log, d2z, static_tab, stats_out=stats)
    else:
        cache = cidx = None
        if variant == 'C':
            with tm.block('cache_read'):
                cache = np.load(ACT_CACHE, mmap_mode='r')
                cidx = read(ACT_INDEX)
        r = ks.build(variant, dq, ranges, names, tm, log, fm=fm, d2z=d2z, cache=cache, cidx=cidx)
    t = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': f'q0_knee working state, condition {variant}'}
    fm_new, srch = ss.search_table(t, (names, nodes, elems, flops), limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', t)
        write(rd / 'config.json', fm_new)
    with tm.block('controls'):
        np.savez(rd / 'q0_knee_calibration_routes.npz', **r['routes'])
        if stats is not None:
            np.savez(rd / 'weight_sens_stats.npz', **{n: d['Gc'] for n, d in stats.items()})
            write(rd / 'weight_sens_stats_meta.json', {n: {'n_in': d['n_in']} for n, d in stats.items()})
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'page_cache_residency_before': residency, 'loadavg_before': load_before, 'search': srch,
                              'work_check': work_check(variant, tm.t), 'fallback_static_rows': r['fallback_static_rows'], 'ce_forward': r['ce_forward'],
                              'hidden_sha': r['hidden_sha'], 'routes_hash': dr.route_hash(r['routes']), 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'],
                              'recompute_max_abs': max(r['recompute_max_abs_per_layer']), 'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']),
                              'q0_routed_rows_by_expert': {k: v['routed_rows'] for k, v in r['meta'].items()}, 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps(dr.clean({'tag': tag, 'method_s': rec['method_s'], 'objective': srch['objective'], 'path': srch['path'], 'work_check': rec['work_check']})))


def e2e(out, log):
    """one process: static BC (+ weight-error cache) -> auto knee -> W (cache just written) -> search."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = 'e2e_W'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    residency = ss.resident_fraction(ks.watch_files())
    load_before = os.getloadavg()
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
    prob = (names, nodes, elems, flops)
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    st = ks.build('static', dq, ranges, names, tm, log)
    t = {'B': st['B'], 'C': st['C'], 'meta': st['meta'], 'state': 'FP32 reference'}
    marks = {'static_done_s': time.perf_counter() - t0}
    with tm.block('table_write'):
        write(rd / 'table_static.json', t)
    with tm.block('cache_write_d2'):
        np.savez(rd / 'weight_error_d2.npz', **{f'{n}|{f}': v for n, d in st['d2'].items() for f, v in d.items()})
    marks['static_written_s'] = time.perf_counter() - t0
    del st
    sel, recs, slog, kept = ks.auto_search(t, prob, tm, out / 'logs', tag)
    q0k, limits = ks.knee_of(sel, names)
    with tm.block('table_write'):
        write(rd / 'q0_knee.json', q0k)
    marks['knee_done_s'] = time.perf_counter() - t0
    with tm.block('cache_read_d2'):
        d2z = np.load(rd / 'weight_error_d2.npz')
    r = build_w(q0k, dq, ranges, names, tm, log, d2z, t)
    tt = {'B': r['B'], 'C': r['C'], 'meta': r['meta'], 'fallback_static_rows': r['fallback_static_rows'], 'state': 'q0_knee working state, condition W'}
    fm_new, srch = ss.search_table(tt, prob, limits, tm, out / 'logs', tag)
    with tm.block('table_write'):
        write(rd / 'table.json', tt)
        write(rd / 'config.json', fm_new)
    rec = run_record(tm, t0, {'variant': 'W', 'page_cache_residency_before': residency, 'loadavg_before': load_before, 'marks': marks, 'limits': limits,
                              'knee': {k: sel['selected'][k] for k in ('resource', 'objective', 'search_budget')}, 'auto_paths': dict(Counter(x.get('path') for x in recs)),
                              'search': srch, 'input_grad_sha_per_layer': r['input_grad_sha_per_layer'], 'hidden_sha': r['hidden_sha'], 'ce_forward': r['ce_forward'],
                              'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']), 'code': code_hashes()})
    write(rd / 'run.json', rec)
    log(json.dumps(dr.clean({'tag': tag, 'method_s': rec['method_s'], 'marks': marks})))


def timing(out, log, phase):
    sched = REFRESH_SCHEDULE if phase == 'refresh' else [('W', None)]
    for v, r in sched:
        name = f'refresh_{v}_r{r}' if phase == 'refresh' else 'e2e_W'
        if (out / 'work/runs' / name / 'run.json').exists():
            log(f'{name}: done already, kept')
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', phase, '--out-dir', str(out)]
        cmd += ['--variant', v, '--rep', str(r)] if phase == 'refresh' else []
        t = time.time()
        p = subprocess.run(cmd, cwd=HERE)
        log(f'{name}: exit {p.returncode}, process wall {time.time() - t:.1f}s, loadavg {os.getloadavg()}')
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
    exp = rr + ['e2e_W']
    runs = {k: read(W / k / 'run.json') for k in exp if (W / k / 'run.json').exists()}
    res = {'missing_runs': [k for k in exp if k not in runs]}
    ctl, prevc = read(out / 'controls.json'), read(KS / 'controls.json')
    st, kn = read(STATIC_TABLE), read(KNEE)
    ft = {v: read(p) for v, p in FROZEN_TABLES.items()}
    fc = {v: read(p) for v, p in FROZEN.items()}
    tW, cW = read(W / 'refresh_W_r1/table.json'), read(W / 'refresh_W_r1/config.json')
    hist = history()
    res['1_inputs'] = {'history': hist, 'pass': hist['pass']}
    fbW = set(runs['refresh_W_r1']['fallback_static_rows'])
    res['2_weight_rows'] = {'equal_historical_C': all(tW['B'][n] == ft['C']['B'][n] for n in names if n not in fbW),
                            'equal_historical_B': all(tW['B'][n] == ft['B']['B'][n] for n in names if n not in fbW)}
    res['2_weight_rows']['pass'] = res['2_weight_rows']['equal_historical_C'] and res['2_weight_rows']['equal_historical_B']
    res['3_activation_rows'] = {'equal_A': all(tW['C'][n] == st['C'][n] for n in names)}
    res['3_activation_rows']['pass'] = res['3_activation_rows']['equal_A']
    res['expected_table'] = {'B': tW['B'] == ft['C']['B'], 'C': tW['C'] == st['C']}
    drr = dr.route_hash(dict(np.load(KS / 'work/q0_knee_calibration_routes_dr_forward.npz')))
    c4 = {k: {'ce': runs[k]['ce_forward'] == ctl['q0_knee_forward_ce'] == prevc['q0_knee_forward_ce'], 'hidden': runs[k]['hidden_sha'] == ctl['q0_knee_forward_hidden_sha'],
              'routes': runs[k]['routes_hash'] == ctl['q0_knee_forward_routes_hash'] == drr} for k in rr}
    c4['e2e_W'] = {'ce': runs['e2e_W']['ce_forward'] == ctl['q0_knee_forward_ce'], 'hidden': runs['e2e_W']['hidden_sha'] == ctl['q0_knee_forward_hidden_sha']}
    res['4_forward'] = {'runs': c4, 'pass': all(all(v.values()) for v in c4.values())}
    g_prev = read(KS / 'work/runs/refresh_B_r1/run.json')['input_grad_sha_per_layer']
    zw, smw = np.load(W / 'refresh_W_r1/weight_sens_stats.npz'), read(W / 'refresh_W_r1/weight_sens_stats_meta.json')
    zc, smc = np.load(C_STATS / 'target_stats.npz'), read(C_STATS / 'target_stats_meta.json')
    gc_equal = all(np.array_equal(zw[n], zc[f'{n}|Gc']) and smw[n]['n_in'] == smc[n]['n_in'] for n in names if n not in fbW)
    res['5_gradients'] = {'input_grad_equal_record': {k: runs[k]['input_grad_sha_per_layer'] == g_prev for k in exp},
                          'recompute_max_abs': max(runs[k]['recompute_max_abs'] for k in rr), 'weight_sensitivity_equal_C': gc_equal}
    res['5_gradients']['pass'] = bool(all(res['5_gradients']['input_grad_equal_record'].values()) and res['5_gradients']['recompute_max_abs'] == 0 and gc_equal)
    d2z = np.load(D2)
    rr_ = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    relw = 0.0
    for n in names:
        if n in fbW:
            continue
        gcn, nin = zw[n], smw[n]['n_in']
        for f in QUANT:
            relw = max(relw, rr_(float(np.sum(gcn / nin * d2z[f'{n}|{f}'])), tW['B'][n][f]))
    res['6_recompute'] = {'numpy_weight_max_rel': relw, 'pass': relw <= TOL_REL}
    res['7_work'] = {'runs': {k: runs[k]['work_check'] for k in rr}, 'pass': all(runs[k]['work_check']['as_expected'] for k in rr)}
    e2t = runs['e2e_W']['timers']
    res['7_work']['e2e_note'] = 'the end-to-end act_error / act_grad_stats blocks belong to the static build; W stage adds no activation-cache read or assembly'
    res['7_work']['e2e_no_assembly_or_act_cache'] = e2t.get('act_assembly', 0.0) == 0 and e2t.get('cache_read', 0.0) == 0
    res['7_work']['pass'] = res['7_work']['pass'] and res['7_work']['e2e_no_assembly_or_act_cache']
    lim = kn['limits']
    rc = resources(cW, names, elems, flops)
    res['8_validity'] = {'fallback_count': len(fbW), 'fallback_rows': sorted(fbW), 'fallback_reason': 'searched expert with no q0_knee calibration token (static row kept)',
                         'finite': all(runs[k]['grads_finite'] and runs[k]['nonfinite'] == 0 for k in exp),
                         'covers_exactly_searched': sorted(cW) == sorted(names) and sorted(tW['B']) == sorted(names) and sorted(tW['C']) == sorted(names),
                         'legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in cW.values()), 'resources': rc,
                         'within_limits': rc['M_bytes'] <= lim['max_bytes'] and rc['BOPs'] <= lim['max_bops'],
                         'search_within_budget_all_runs': all(runs[k]['search']['within_budget'] for k in exp), 'e2e_limits_equal': runs['e2e_W']['limits'] == lim}
    v8 = res['8_validity']
    v8['pass'] = bool(v8['finite'] and v8['covers_exactly_searched'] and v8['legal'] and v8['within_limits'] and v8['search_within_budget_all_runs'] and v8['e2e_limits_equal'])
    reps = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        tref = tW if v == 'W' else ft[v]
        cref = cW if v == 'W' else fc[v]
        reps[v] = {'tables': all({kk: vv for kk, vv in read(W / k / 'table.json').items() if kk in ('B', 'C')} == {kk: tref[kk] for kk in ('B', 'C')} for k in ks_),
                   'configs': all(read(W / k / 'config.json') == cref for k in ks_)}
    reps['e2e_W'] = {'static_table': {kk: vv == st[kk] for kk, vv in read(W / 'e2e_W/table_static.json').items() if kk in ('B', 'C')},
                     'weight_error': ss.d2_equal(W / 'e2e_W/weight_error_d2.npz', D2), 'knee': read(W / 'e2e_W/q0_knee.json') == read(Q0K),
                     'W_table': {kk: read(W / 'e2e_W/table.json')[kk] == tW[kk] for kk in ('B', 'C')}, 'W_config': read(W / 'e2e_W/config.json') == cW}
    reps['expected_table'] = res['expected_table']
    flat = lambda d: all(flat(v) if isinstance(v, dict) else bool(v) for v in d.values())
    res['9_repeats'] = reps
    res['9_repeats']['pass'] = flat(reps)
    res['identical'] = {f'W={c}': cW == fc[c] for c in fc}
    after = {'inputs': {k: sha(HERE / k) for k in pl['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['inputs_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['ste'] = ctl['ste_unit_test']['pass']
    res['controls_stage_pass'] = ctl['pass']
    res['pass'] = bool(not res['missing_runs'] and all(res[k]['pass'] for k in ('1_inputs', '2_weight_rows', '3_activation_rows', '4_forward', '5_gradients',
                                                                                 '6_recompute', '7_work', '8_validity', '9_repeats'))
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
    write(out / 'configs/W.json', read(W / 'refresh_W_r1/config.json'))
    write(out / 'tables/W.json', read(W / 'refresh_W_r1/table.json'))
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E', 'E5', 'W')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    diff = lambda a, b: {'projections': sum(a[n] != b[n] for n in names), 'weight': sum(a[n]['w'] != b[n]['w'] for n in names),
                         'act': sum(a[n]['a'] != b[n]['a'] for n in names), 'weight_only': sum(a[n]['w'] != b[n]['w'] and a[n]['a'] == b[n]['a'] for n in names),
                         'act_only': sum(a[n]['a'] != b[n]['a'] and a[n]['w'] == b[n]['w'] for n in names), 'both': sum(a[n]['w'] != b[n]['w'] and a[n]['a'] != b[n]['a'] for n in names)}
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'main_configuration_rule': 'W = refresh repeat 1; others = frozen records',
           'configs': {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                           'config_hash': config_hash(cfg[c]), 'resources': resources(cfg[c], names, elems, flops)} for c in cfg},
           'frozen_equal_records': {c: cfg[c] == read(p) for c, p in FROZEN.items()},
           'identical': {f'W={c}': cfg['W'] == cfg[c] for c in ('A', 'B', 'C', 'D4', 'D5', 'E', 'E5')},
           'pairwise_differences': {f'W-{c}': diff(cfg['W'], cfg[c]) for c in ('A', 'B', 'C')} | {f'{c}-A': diff(cfg[c], cfg['A']) for c in ('B', 'C')},
           'transitions_vs_A': {c: {'weight': dict(Counter(f'{cfg["A"][n]["w"]}->{cfg[c][n]["w"]}' for n in names if cfg['A'][n]['w'] != cfg[c][n]['w'])),
                                    'act': dict(Counter(f'{cfg["A"][n]["a"]}->{cfg[c][n]["a"]}' for n in names if cfg['A'][n]['a'] != cfg[c][n]['a']))} for c in ('W',)},
           'tables': {'W.json': sha(out / 'tables/W.json'), **{f'frozen/{v}': sha(p) for v, p in FROZEN_TABLES.items()}},
           'knee_sha256': sha(KNEE), 'check_sha256': sha(out / 'check.json'), 'plan_sha256': sha(out / 'plan.json'), 'code': code_hashes()}
    write(out / 'freeze.json', res)
    log(json.dumps(dr.clean({k: res[k] for k in ('identical', 'pairwise_differences')})))


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
    old_t, old = np.load(K5E / 'tokens.npy')[:CTRL], read(K5E / 'eval.json')['ce']
    pc = {}
    for c in ('fp32', 'A', 'B', 'E5'):
        r = dr.forward(old_t, None if c == 'fp32' else read(FROZEN[c]), dq, ranges, record=False, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    conds = ('A', 'B', 'C', 'W')
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
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/W.json'), dq, ranges, record=True, log=log, label='rerun W')
    ctl = {'program_control_ce_max_abs': pc, 'shared_evaluations': shared,
           'rerun_W_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['W']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['W']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['W']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()), 'audit': {c: runs[c]['audit'] for c in conds},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in conds},
           'loadavg_end': os.getloadavg()}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_W_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_W_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_W_first_chunk']['routes_equal'] and ctl['finite']
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
            proc['e2e_W'] = j['seconds']
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in VARIANTS:
        ks_ = [f'refresh_{v}_r{r}' for r in (1, 2, 3)]
        s[f'refresh_{v}'] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                             'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                             'timers': {t: med([runs[k]['timers'].get(t, 0.0) for k in ks_]) for t in sorted({t for k in ks_ for t in runs[k]['timers']})},
                             'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                             'loadavg_before': {k: runs[k]['loadavg_before'] for k in ks_},
                             'page_cache_residency_before': {k: runs[k]['page_cache_residency_before'] for k in ks_}}
    e = runs['e2e_W']
    s['e2e_W_measured_single'] = {'method_s': e['method_s'], 'process_wall_s': proc['e2e_W'], 'marks': e['marks'], 'categories': e['categories'],
                                  'peak_gpu_gb': e['peak_gpu_gb'], 'peak_cpu_gb': e['peak_cpu_gb'], 'loadavg_before': e['loadavg_before']}
    et = read(KE / 'analysis.json')['timing']
    e5t = read(K5E / 'analysis.json')['timing']
    s['historical'] = {'e2e_measured_single': {**et['historical']['e2e_measured_single'], 'E': et['e2e_E_measured_single']['method_s'],
                                               'E5': e5t['e2e_E5_measured_single']['method_s']},
                       'auto_median_s': et['historical']['auto_median_s'], 'note': 'previous knee rounds; historical single measurements'}
    SW = HERE / 'runs/olmoe_selective_sensitivity_20261022/work/runs'
    sp = float(np.median([read(SW / f'static_plain_r{r}/run.json')['method_s'] - sum(read(SW / f'static_plain_r{r}/run.json')['timers'].get(t, 0.0)
                                                                                     for t in ('pruning', 'search')) for r in (1, 2, 3)]))
    s['W_total_estimate_sum_of_stage_medians_s'] = {'value': sp + et['historical']['auto_median_s'] + s['refresh_W']['method_s']['median'],
                                                    'components': {'static_with_weight_error_cache_write': sp, 'auto_median_previous_round': et['historical']['auto_median_s'],
                                                                   'W_median': s['refresh_W']['method_s']['median']}}
    s['extra_with_caches_median_s'] = {v: s[f'refresh_{v}']['method_s']['median'] for v in VARIANTS}
    s['note'] = 'method = run time minus control blocks; estimates are sums of stage medians, not a measured pipeline median; interleaving does not remove background load'
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
    pairs = {'W_minus_A': ('W', 'A'), 'W_minus_C': ('W', 'C'), 'W_minus_B': ('W', 'B'), 'B_minus_A': ('B', 'A'), 'C_minus_A': ('C', 'A')}
    D = {k: A[a] - A[b] for k, (a, b) in pairs.items()}
    res = {'n_articles': N, 'bonferroni_level': 100 - 2 * CI_CORR[0], 'primary': {k: st(D[k]) for k in ('W_minus_A', 'W_minus_C', 'W_minus_B')},
           'secondary': {k: st(D[k]) for k in ('B_minus_A', 'C_minus_A')}, 'damage': {c: st(A[c] - A['fp32']) for c in ('A', 'B', 'C', 'W')}}
    for k in ('W_minus_A', 'W_minus_C'):
        lo, hi = res['primary'][k]['ci_bonferroni']
        res['primary'][k]['decision'] = 'W_better_confirmed' if hi < 0 else 'W_worse_confirmed' if lo > 0 else 'direction_not_confirmed'
    lo, hi = res['primary']['W_minus_B']['ci_bonferroni']
    res['primary']['W_minus_B']['decision'] = (f'noninferior_within_{NI_MARGIN}' if hi < NI_MARGIN else f'beyond_margin_{NI_MARGIN}' if lo > NI_MARGIN
                                               else 'insufficient_evidence')
    for k, v in res['secondary'].items():
        lo, hi = v['ci95']
        v['direction_95'] = 'first_better' if hi < 0 else 'first_worse' if lo > 0 else 'not_confirmed'
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    fz = read(out / 'freeze.json')
    cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B', 'C', 'D4', 'D5', 'E', 'E5', 'W')}
    names = sorted(cfg['A'])
    _, nodes, elems, flops, _ = pr.problem_static()
    tW = read(out / 'tables/W.json')
    lim = read(KNEE)['limits']
    obj = lambda t, fm: float(sum(t['B'][n][fm[n]['w']] + t['C'][n][fm[n]['a']] for n in names))
    sW = read(out / 'work/runs/refresh_W_r1/run.json')['search']
    res['search_W'] = {k: sW[k] for k in ('objective', 'lower_bound', 'path', 'options_kept', 'resources', 'budget_use', 'weight_formats', 'act_formats', 'within_budget')}
    res['search_W']['gap_abs'] = sW['objective'] - sW['lower_bound']
    res['search_W']['gap_rel'] = (sW['objective'] - sW['lower_bound']) / sW['objective']
    cross = {}
    for c in cfg:
        r = resources(cfg[c], names, elems, flops)
        cross[c] = {'objective_on_W_table': obj(tW, cfg[c]), 'feasible_at_knee_limits': r['M_bytes'] <= lim['max_bytes'] and r['BOPs'] <= lim['max_bops'],
                    'below_W_solution': obj(tW, cfg[c]) < sW['objective'] and c != 'W'}
    res['cross_scores_on_W_table'] = cross
    res['known_configuration_cheaper_than_W_solution'] = sorted(c for c, v in cross.items() if v['below_W_solution'] and v['feasible_at_knee_limits'])
    res['configs'] = {c: v['resources'] for c, v in fz['configs'].items()}
    res['format_changes'] = {k: fz[k] for k in ('identical', 'pairwise_differences', 'transitions_vs_A')}
    _, _, pnames, _, elems2, rows_cal, _ = problem(MP)
    counts, desc = {}, {}
    shared = ev['controls']['shared_evaluations']
    conds = ('A', 'B', 'C', 'W')
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
                             'cheaper_known': res['known_configuration_cheaper_than_W_solution']})))


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
