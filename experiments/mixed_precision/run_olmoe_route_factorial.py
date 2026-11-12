#!/usr/bin/env python3
"""OLMoE aggressive BC configuration, FP32 expert ids S0 fixed: 2x2 routing-weight restore diagnostic on the same
103 WikiText-103 articles (412 windows) as runs/olmoe_route_mass_restore_20261018.

                      proportions not restored     proportions restored (FP32 w0 / m0)
  sum not restored    L01   w = p[S0]              Lprop  w0 * (m / m0)
  sum restored (m0)   Lmass w * (m0 / m)           L00    w0  (= m0 * w0 / m0)

w = p[S0] of the condition's own hidden state, m = sum(w), w0 = p0[S0], m0 = sum(w0) (float32, sums in S0 order).
L00 / L01 / Lmass are run with the unchanged run_olmoe_route_mass_restore.forward5 and must reproduce that run bitwise;
Lprop uses the same arithmetic with its own weight line. Diagnostic with FP32 information, not a deployable method.

Stages: plan, run, analyze (one --out-dir). Pre-registered in plan.json / plan.md before Lprop is computed.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
MR = HERE / 'runs/olmoe_route_mass_restore_20261018'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK, CTRL = 128, 128
CELLS = ('L01', 'Lmass', 'Lprop', 'L00')
EPS, ALPHA, BOOT = 0.002, 0.05, 10000
SEED = 18131


def forward_prop(toks_np, fm, dq, ranges, base, log=print, label='', chunk=CHUNK):
    """run_olmoe_route_mass_restore.forward5 with the Lprop weight line:
    w = p[S0] (float32), m = sum(w), w0 = p0[S0], m0 = sum(w0), scale = m / m0, weights = w0 * scale."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_olmoe_search_final import quantized_layer
    cfg, ck = oc.config(), oc.Checkpoint()
    assert not cfg.norm_topk_prob
    eps, Hd = cfg.rms_norm_eps, cfg.hidden_size
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    nW = len(toks_np)
    chunks = [(a, min(nW, a + chunk)) for a in range(0, nW, chunk)]
    rec = {'ids': np.zeros((nW, NL, SEQ, K), np.int16), 'weights': np.zeros((nW, NL, SEQ, K), np.float32)}
    extra = {'moe_norm': np.zeros((nW, NL), np.float32), 'scale': np.zeros((nW, NL, SEQ), np.float32),
             'm': np.zeros((nW, NL, SEQ), np.float32), 'm0': np.zeros((nW, NL, SEQ), np.float32), 'w_raw': np.zeros((nW, NL, SEQ, K), np.float32)}
    audit = {'router_untouched': True, 'hook_keys_are_expert_projections': True, 'hooks': 0}
    torch.cuda.synchronize(); t0 = time.time()
    H = {}
    with torch.no_grad():
        for a, b in chunks:
            H[a] = F.embedding(torch.from_numpy(toks_np[a:b]).to('cuda'), g['embed'])
        for l in range(NL):
            w = ck.layer(l, 'cuda', torch.float32)
            if fm is None:
                wq, hk = w, None
            else:
                wq, hk = quantized_layer(w, fm, l, dq, ranges, NE)
                audit['router_untouched'] &= (wq['router'] is w['router'])
                audit['hooks'] += len(hk.act)
                audit['hook_keys_are_expert_projections'] &= all(k_[0] == l and k_[2] in oc.PROJ and k_[3] in ('x', 'y') for k_ in hk.act)
            S0l = torch.from_numpy(base['S0'][:, l].reshape(-1, K).astype(np.int64)).to('cuda')
            p0l = torch.from_numpy(np.ascontiguousarray(base['p0'][:, l]).reshape(-1, NE)).to('cuda')
            for a, b in chunks:
                B = b - a
                rows = slice(a * SEQ, b * SEQ)
                h = H[a]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], eps).reshape(-1, Hd)
                logits = F.linear(z, wq['router'])
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                idx = S0l[rows]
                wcur = probs.gather(-1, idx).to(torch.float32)
                m = torch.sum(wcur, dim=-1, keepdim=True)
                w0r = p0l[rows].gather(-1, idx)
                m0 = torch.sum(w0r, dim=-1, keepdim=True)
                if not (bool(torch.isfinite(m).all()) and bool(torch.isfinite(m0).all()) and bool((m0 > 0).all())):
                    raise RuntimeError(f'Lprop: zero / non-finite sum at layer {l}')
                scale = m / m0
                rw = w0r * scale
                if not (bool(torch.isfinite(scale).all()) and bool(torch.isfinite(rw).all())):
                    raise RuntimeError(f'Lprop: non-finite scale / weights at layer {l}')
                extra['scale'][a:b, l] = scale.view(B, SEQ).cpu().numpy()
                extra['m'][a:b, l] = m.view(B, SEQ).cpu().numpy()
                extra['m0'][a:b, l] = m0.view(B, SEQ).cpu().numpy()
                extra['w_raw'][a:b, l] = wcur.view(B, SEQ, K).cpu().numpy()
                rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                moe = oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
                extra['moe_norm'][a:b, l] = moe[:, :TP].norm(dim=-1).mean(-1).cpu().numpy()
                H[a] = x + moe
            del wq, w, S0l, p0l
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'extra': extra, 'seconds': time.time() - t0, 'audit': audit, 'chunks': chunks,
            'finite_ce': bool(np.isfinite(ce).all()), 'finite_weights': bool(np.isfinite(rec['weights']).all())}


def prop_semantics(r, base):
    S0, w0 = base['S0'], base['w0']
    W, e = r['routes']['weights'], r['extra']
    W64 = W.astype(np.float64)
    ratio = W64 / np.where(w0 != 0, w0.astype(np.float64), np.nan)
    res = {'ids_equal_S0': bool(np.array_equal(r['routes']['ids'], S0)),
           'weights_equal_float32_w0_times_scale': bool(np.array_equal(W, (w0 * e['scale'][..., None]).astype(np.float32))),
           'recorded_m0_minus_float32_sum_w0_max_abs': float(np.max(np.abs(e['m0'] - w0.sum(-1, dtype=np.float32)))),
           'recorded_m_minus_float32_sum_w_raw_max_abs': float(np.max(np.abs(e['m'] - e['w_raw'].sum(-1, dtype=np.float32)))),
           'sum_minus_m_max_rel': float(np.max(np.abs(W64.sum(-1) - e['w_raw'].astype(np.float64).sum(-1)) / e['w_raw'].astype(np.float64).sum(-1))),
           'proportion_ratio_to_w0_max_rel_spread': float(np.nanmax((np.nanmax(ratio, -1) - np.nanmin(ratio, -1)) / np.nanmean(ratio, -1))),
           'sum_equals_1_share': float((np.abs(W64.sum(-1) - 1) < 1e-6).mean()),
           'scale_min': float(e['scale'].min()), 'scale_max': float(e['scale'].max()),
           'ids_in_range_and_distinct': bool(r['routes']['ids'].min() >= 0 and r['routes']['ids'].max() < NE
                                             and (np.diff(np.sort(r['routes']['ids'], -1), axis=-1) > 0).all())}
    res['pass'] = bool(res['ids_equal_S0'] and res['weights_equal_float32_w0_times_scale'] and res['recorded_m0_minus_float32_sum_w0_max_abs'] <= 1e-6
                       and res['recorded_m_minus_float32_sum_w_raw_max_abs'] <= 1e-6 and res['sum_minus_m_max_rel'] < 1e-5
                       and res['proportion_ratio_to_w0_max_rel_spread'] < 1e-5 and res['ids_in_range_and_distinct'])
    return res


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    mp = mr.read(MR / 'plan.json')
    inputs = {str(p.relative_to(HERE)): mr.sha(p) for p in [mr.CONFIG, dr.RANGES, MR / 'plan.json', MR / 'data_freeze.json', MR / 'segments.json',
                                                            MR / 'tokens.npy', MR / 'results.json', MR / 'analysis.json', MR / 'baseline_routes.npz']
              + [MR / f'routes/{c}.npz' for c in ('L00', 'L01', 'Lmass')]}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'With FP32 expert ids fixed, which part of the routing-weight change hurts: the selected sum, the '
                    'relative proportions among the selected experts, or both? 2x2 restore diagnostic on the same windows.',
        'design': {
            'cells': {'L01': 'neither restored: w = p[S0]', 'Lmass': 'sum restored: w * (m0 / m)', 'Lprop': 'proportions restored: '
                      'w0 * (m / m0) (FP32 proportions w0 / m0, the condition own current sum m)', 'L00': 'both restored: w0 '
                      '(m0 * w0 / m0 = w0 by definition; the fully fixed condition)'},
            'definitions': 'w = p[S0] from each condition own hidden state (float32); m = torch.sum(w, -1); w0 = p0[S0]; '
                           'm0 = torch.sum(w0, -1), float32, S0 order. Lprop: scale = m / m0 (float32), weights = w0 * scale '
                           '(float32; equals w0 exactly when m = m0, so quantization-off Lprop must equal FP32 bitwise). '
                           'No renormalization to 1, no logit change, no clipping / epsilon; zero or non-finite values stop the run.',
            'reuse': 'L00 / L01 / Lmass are recomputed with the unchanged run_olmoe_route_mass_restore.forward5 and must '
                     'reproduce that run bitwise (CE, ids, weights); only Lprop is new',
            'shared': 'same aggressive configuration (sha256 %s), same 103 WikiText-103 articles / 412 windows (tokens sha256 %s), '
                      'same FP32 baseline (recomputed, must match bitwise), chunks of 128 windows (tail 28), GPU FP32, TF32 off, '
                      'router FP32, 21 absent projections bf16 / FP32; GPU QDQ behaviour, not native kernels'
                      % (mp['configuration']['sha256'][:12], mr.sha(MR / 'tokens.npy')[:12]),
            'caveat': 'every cell changes later hidden states (end-to-end layerwise interventions); no "X% sum, Y% proportions" shares; '
                      'the articles were already used for Lmass / L01 / L00 (their contrasts are re-reported, not new evidence); '
                      'Lprop is the only new measurement'},
        'endpoints': {
            'primary': 'Delta_prop = CE(Lprop) - CE(L01), 95% percentile CI; negative = restoring FP32 proportions helps',
            'secondary_family': 'Bonferroni over four (two-sided 98.75% intervals): main effect of restoring proportions '
                                '= 1/2[(Lprop - L01) + (L00 - Lmass)]; main effect of restoring the sum = 1/2[(Lmass - L01) + '
                                '(L00 - Lprop)]; interaction = (L00 - Lprop) - (Lmass - L01); Residual_prop = Lprop - L00 '
                                '(equivalent within +-eps if inside, positive if lower > 0)',
            'descriptive': 'Delta_restore = Lmass - L01 and Residual = Lmass - L00 re-reported (already known), CE / PPL of all cells',
            'eps': f'{EPS} nat/token; CI containing 0 is not equivalence',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix, seed {SEED}; article = 4 windows / 508 tokens'},
        'interpretation': {
            'proportions_help': 'Delta_prop 95% CI entirely below 0: FP32 proportions reduce the weight-path damage',
            'proportions_enough': 'and Residual_prop inside +-eps (98.75%): restoring proportions alone returns to the fixed '
                                  'level within tolerance',
            'proportions_partial': 'Delta_prop below 0 but Residual_prop positive: proportions explain part only',
            'interaction': 'interaction CI excludes 0: sum and proportions do not act additively',
            'null': 'Delta_prop CI contains 0: no claim; report the excluded improvement range; then the damage is more '
                    'likely carried by later-layer cascades than by the weight values of one layer (hypothesis)',
            'scope': 'FP32 ids and FP32 w0 are used: no transfer to full dynamic routing or to a deployable fix is claimed'},
        'controls': ['FP32 baseline recomputed == run_olmoe_route_mass_restore baseline (S0, w0, p0, CE) bitwise',
                     'L00 / L01 / Lmass == that run (CE all windows, ids, weights) bitwise',
                     'Lprop: ids = S0; weights == float32(w0 * scale) exactly; recorded m0 and m equal float32 sums (<= 1e-6); '
                     'sum(weights) / m relative < 1e-5; weights / w0 constant per token (relative spread < 1e-5); sums not 1',
                     'quantization off (first chunk): Lprop == FP32 bitwise; rerun of Lprop on the first chunk bitwise; '
                     'replay of Lprop ids / weights through run_olmoe_dynamic_route.forward bitwise (all windows)',
                     'config completeness, router untouched, hooks on expert projections only, finite values',
                     'any failure: fix and record before analysing losses'],
        'auxiliary': 'per layer: selected-sum gap to m0 and proportion shift (L1 of w / sum(w) vs w0 / m0) for every cell, '
                     'Lprop scale m / m0 distribution, mean MoE output norm; descriptive only',
        'not_done': 'no BC / search / config change, no deployable sum compensation, no first-order term',
        'inputs': inputs, 'code': {f: mr.sha(HERE / f) for f in ['run_olmoe_route_factorial.py', 'run_olmoe_route_mass_restore.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    mr.write(out / 'plan.json', p)
    md = ['# 預先登記：固定 FP32 expert 選擇的 routing weight 2×2 恢復診斷', '', f'時間：{p["time"]}。以 plan.json 為準；在計算 Lprop 之前寫定。', '',
          '| | 比例不恢復 | 比例恢復（w0／m0） |', '|---|---|---|',
          '| 總和不恢復 | L01：w = p[S0] | Lprop：w0 × (m／m0) |', '| 總和恢復（m0） | Lmass：w × (m0／m) | L00：w0 |', '',
          '- w 來自各條件自身 hidden state；m = sum(w)、m0 = sum(w0)，float32、依 S0 順序；不 normalize 到 1、不改 logits、不 clip。',
          '- L00／L01／Lmass 用上一輪未改動的 forward5 重算，須與上一輪逐位元相同；只有 Lprop 是新量測。',
          '- 同一配置、同 103 篇 WikiText-103 文章（412 窗口）、同 FP32 基準；GPU QDQ，不代表 native。', '',
          '## 終點', '', '- 主要：Δ_prop = CE(Lprop) − CE(L01)，95% CI；負值 = 恢復 FP32 比例有幫助。',
          '- 次要族（Bonferroni ×4，98.75%）：比例主效應、總和主效應、交互作用、Residual_prop = Lprop − L00（±0.002 等價或 >0）。',
          '- 描述：Δ_restore、Residual（上一輪已知）、各格 CE／PPL。不做「總和占 X%、比例占 Y%」。', '',
          '## 判讀', ''] + [f'- {k}：{v}' for k, v in p['interpretation'].items()] + ['', '## 控制', ''] + [f'- {c}' for c in p['controls']] + ['']
    (out / 'plan.md').write_text('\n'.join(md))
    log('plan written')


def run(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'results.json').exists():
        raise SystemExit('results exist')
    pl = mr.read(out / 'plan.json')
    for k, v in pl['inputs'].items():
        if mr.sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    toks = np.load(MR / 'tokens.npy')
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = mr.read(mr.CONFIG), dr.read(dr.RANGES)
    cc = dr.config_checks(fm, dq, ranges)
    f, base = mr.baseline(toks, dq, ranges, log, 'baseline')
    ob, old = np.load(MR / 'baseline_routes.npz'), mr.read(MR / 'results.json')['ce']
    ctl = {'config_checks': cc, 'baseline_vs_previous': {
        'S0': bool(np.array_equal(base['S0'], ob['S0'])), 'w0': bool(np.array_equal(base['w0'], ob['w0'])),
        'p0': bool(np.array_equal(base['p0'], ob['p0'])), 'ce_max_abs': float(np.max(np.abs(np.array(f['ce']) - np.array(old['fp32']))))}}
    runs = {c: mr.forward5(toks, fm, dq, ranges, c, base=base, log=log, label=c) for c in ('L00', 'L01', 'Lmass')}
    ctl['reuse_vs_previous'] = {}
    for c, r in runs.items():
        z = np.load(MR / f'routes/{c}.npz')
        ctl['reuse_vs_previous'][c] = {'ce_max_abs': float(np.max(np.abs(np.array(r['ce']) - np.array(old[c])))),
                                       'ids_equal': bool(np.array_equal(r['routes']['ids'], z['ids'])),
                                       'weights_equal': bool(np.array_equal(r['routes']['weights'], z['weights']))}
    runs['Lprop'] = forward_prop(toks, fm, dq, ranges, base, log=log, label='Lprop')
    (out / 'routes').mkdir(exist_ok=True)
    np.savez(out / 'routes/Lprop.npz', ids=runs['Lprop']['routes']['ids'], weights=runs['Lprop']['routes']['weights'], **runs['Lprop']['extra'])
    ctl['Lprop_semantics'] = prop_semantics(runs['Lprop'], base)
    sub, sb = toks[:CTRL], {k: v[:CTRL] for k, v in base.items()}
    off = forward_prop(sub, None, dq, ranges, sb, log=log, label='off Lprop')
    ctl['Lprop_quant_off'] = mr.eq(off, f, CTRL)
    ctl['Lprop_quant_off_scale_all_one'] = bool((off['extra']['scale'] == 1.0).all())
    r2 = forward_prop(sub, fm, dq, ranges, sb, log=log, label='rerun Lprop')
    ctl['Lprop_rerun'] = mr.eq(r2, runs['Lprop'], CTRL)
    ctl['Lprop_rerun']['routes_equal'] = bool(np.array_equal(r2['routes']['weights'], runs['Lprop']['routes']['weights'][:CTRL]))
    rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs['Lprop']['routes']), log=log, label='replay Lprop')
    ctl['Lprop_replay'] = mr.eq(rp, runs['Lprop'])
    ctl['finite'] = all(r['finite_ce'] and r['finite_weights'] for r in runs.values()) and f['finite_ce']
    ctl['audit'] = {c: r['audit'] for c, r in runs.items()}
    bp = ctl['baseline_vs_previous']
    ctl['pass'] = bool(cc['pass'] and bp['S0'] and bp['w0'] and bp['p0'] and bp['ce_max_abs'] == 0
                       and all(v['ce_max_abs'] == 0 and v['ids_equal'] and v['weights_equal'] for v in ctl['reuse_vs_previous'].values())
                       and ctl['Lprop_semantics']['pass'] and mr.bitwise(ctl['Lprop_quant_off']) and ctl['Lprop_quant_off_scale_all_one']
                       and mr.bitwise(ctl['Lprop_rerun']) and ctl['Lprop_rerun']['routes_equal'] and mr.bitwise(ctl['Lprop_replay'])
                       and ctl['finite'] and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    norms = {c: r['extra']['moe_norm'] for c, r in runs.items()}
    norms['fp32'] = f['extra']['moe_norm']
    np.savez(out / 'moe_norms.npz', **norms)
    mr.write(out / 'results.json', {'ce': {'fp32': f['ce'], **{c: r['ce'] for c, r in runs.items()}}, 'controls': ctl,
                                    'seconds': {'baseline': f['seconds'], **{c: r['seconds'] for c, r in runs.items()},
                                                'Lprop_quant_off': off['seconds'], 'Lprop_rerun': r2['seconds'], 'Lprop_replay': rp['seconds']},
                                    'environment': dr.env_info()})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    t0 = time.time()
    rs = mr.read(out / 'results.json')
    if not rs['controls']['pass']:
        raise SystemExit('controls failed: do not analyse losses')
    nA = len(mr.read(MR / 'segments.json'))
    A = {c: np.array(v).reshape(nA, WPA).sum(1) / (WPA * TP) for c, v in rs['ce'].items()}
    E = {'Delta_prop': A['Lprop'] - A['L01'],
         'main_proportions': 0.5 * ((A['Lprop'] - A['L01']) + (A['L00'] - A['Lmass'])),
         'main_sum': 0.5 * ((A['Lmass'] - A['L01']) + (A['L00'] - A['Lprop'])),
         'interaction': (A['L00'] - A['Lprop']) - (A['Lmass'] - A['L01']),
         'Residual_prop': A['Lprop'] - A['L00'], 'Delta_restore': A['Lmass'] - A['L01'], 'Residual_mass': A['Lmass'] - A['L00'],
         'E_weight': A['L01'] - A['L00'], **{f'D_{c}': A[c] - A['fp32'] for c in CELLS}}
    idx = np.random.default_rng(SEED).integers(0, nA, (BOOT, nA))

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci_bonf4': np.percentile(d, [0.625, 99.375]).tolist(),
                'article_sd': float(x.std(ddof=1)), 'articles_positive': int((x > 0).sum())}
    res = {'n_articles': nA, 'contrasts': {k: st(v) for k, v in E.items()}}
    c = res['contrasts']
    lo, hi = c['Delta_prop']['ci95']
    rp = c['Residual_prop']['ci_bonf4']
    res['primary'] = 'helps' if hi < 0 else 'hurts' if lo > 0 else 'null'
    res['secondary'] = {'main_proportions': 'negative' if c['main_proportions']['ci_bonf4'][1] < 0 else 'positive' if c['main_proportions']['ci_bonf4'][0] > 0 else 'none',
                        'main_sum': 'negative' if c['main_sum']['ci_bonf4'][1] < 0 else 'positive' if c['main_sum']['ci_bonf4'][0] > 0 else 'none',
                        'interaction': 'nonzero' if (c['interaction']['ci_bonf4'][0] > 0 or c['interaction']['ci_bonf4'][1] < 0) else 'none',
                        'Residual_prop_within_eps': -EPS <= rp[0] and rp[1] <= EPS, 'Residual_prop_positive': rp[0] > 0}
    tokens = nA * WPA * TP
    res['ce_ppl'] = {k: {'ce': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for k, v in rs['ce'].items()}
    b = np.load(MR / 'baseline_routes.npz')
    w0 = b['w0'][:, :, :TP].astype(np.float64); m0 = w0.sum(-1); q0 = w0 / m0[..., None]
    W = {c_: np.load(MR / f'routes/{c_}.npz')['weights'] for c_ in ('L01', 'Lmass')}
    W['Lprop'] = np.load(out / 'routes/Lprop.npz')['weights']
    W['L00'] = b['w0']
    sc = np.load(out / 'routes/Lprop.npz')['scale'][:, :, :TP]
    nm = np.load(out / 'moe_norms.npz')
    rows = []
    for l in range(NL):
        row = {'layer': l, 'm0_mean': float(m0[:, l].mean())}
        for c_ in CELLS:
            wv = W[c_][:, l, :TP].astype(np.float64)
            row[f'{c_}_sum_minus_m0'] = float((wv.sum(-1) - m0[:, l]).mean())
            row[f'{c_}_proportion_l1'] = float(np.abs(wv / wv.sum(-1, keepdims=True) - q0[:, l]).sum(-1).mean())
            row[f'{c_}_moe_norm'] = float(nm[c_][:, l].mean())
        row.update({'fp32_moe_norm': float(nm['fp32'][:, l].mean()), 'Lprop_scale_mean': float(sc[:, l].mean()),
                    'Lprop_scale_p01': float(np.percentile(sc[:, l], 1)), 'Lprop_scale_p99': float(np.percentile(sc[:, l], 99))})
        rows.append(row)
    with open(out / 'per_layer.csv', 'w') as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    res['Lprop_scale'] = {'min': float(sc.min()), 'max': float(sc.max()), 'mean': float(sc.mean()), 'p001': float(np.percentile(sc, 0.1)),
                          'p999': float(np.percentile(sc, 99.9))}
    segs = mr.read(MR / 'segments.json')
    with open(out / 'per_article.csv', 'w') as fh:
        pa = [{'index': s['index'], 'title': s['title'], **{f'ce_{k}': float(v[i]) for k, v in A.items()},
               **{k: float(E[k][i]) for k in ('Delta_prop', 'Delta_restore', 'interaction', 'Residual_prop')}} for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(pa[0])); w_.writeheader(); w_.writerows(pa)
    res['controls'] = rs['controls']
    res['timing'] = {'run': rs['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    mr.write(out / 'analysis.json', res)
    log(json.dumps({k: [c[k]['mean'], c[k]['ci95']] for k in ('Delta_prop', 'main_proportions', 'main_sum', 'interaction', 'Residual_prop')}) + ' ' + res['primary'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'run', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'run': run, 'analyze': analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': mr.sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
