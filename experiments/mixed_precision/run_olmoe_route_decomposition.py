#!/usr/bin/env python3
"""OLMoE aggressive BC configuration: end-to-end routing interventions that separate expert selection from routing
weights. Pre-registered in plan.json / plan.md.

Per layer l of a condition's own forward (its own hidden state z_l -> FP32 router -> p = softmax over 64, S = top-8):
  L00  ids = S0, weights = p0[S0]          (fully fixed: FP32 baseline ids and probabilities)
  L01  ids = S0, weights = p[S0]           (ids fixed, weights released)
  L10  ids = S,  weights = p0[S]           (ids released, FP32 probability function)
  L11  ids = S,  weights = p[S]            (fully dynamic)
S0 / p0 (all 64 probabilities) come from the complete FP32 dynamic baseline of the same window. No renormalization
(the checkpoint has norm_topk_prob = false). The layer arithmetic is that of run_olmoe_dynamic_route.forward; every
condition's actual ids / weights are recorded and replayed through that verified forward as a control.

Stages (one --out-dir):
  plan     plan.json / plan.md (conditions, hashes, sample-size rule, statistics, controls, interpretation)
  pilot    old windows (run_olmoe_dynamic_multipoint first 128 = 32 articles): program controls + article SDs -> n
  data     n new train articles x 4 windows (strict boundaries, whole-article exclusion of every earlier set)
  run      FP32 baseline (S0, p0), L00/L01/L10/L11, controls (quantization off, reruns, replay)
  analyze  contrasts, routing statistics, per-article table
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr

HERE = Path(__file__).resolve().parent
MPT = HERE / 'runs/olmoe_dynamic_multipoint_20261018'
PREV1 = HERE / 'runs/olmoe_dynamic_route_20261018'
CONFIG = MPT / 'configs/aggressive.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK = 128
CTRL = 128                                  # control subset = the first chunk
PILOT_WINDOWS = 128                         # multipoint windows 0-127 (32 articles x 4)
CONDS = ('L00', 'L01', 'L10', 'L11')
EPS = 0.002
ALPHA = 0.05
BOOT = 10000
H_TARGET = 0.002
SEEDS = {'data': 18091, 'bootstrap': 18101}
PREV_AGGR = {'mean': 0.007528211910623012, 'ci95': [0.00606, 0.00897]}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


# --------------------------------------------------------------------------- #
def forward4(toks_np, fm, dq, ranges, mode, base=None, record_p=False, log=print, label='', chunk=CHUNK):
    """run_olmoe_dynamic_route.forward's layer-major pass with the route chosen by `mode` (see module doc);
    mode 'fp32' = own router (as L11) used for the FP32 baseline. base = {'S0': [N, L, T, k], 'p0': [N, L, T, 64]}.
    Records the ids / weights actually used (and, with record_p, the full probability vector of the own router)."""
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
    P = np.zeros((nW, NL, SEQ, NE), np.float32) if record_p else None
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
            if mode in ('L00', 'L01', 'L10'):
                S0l = torch.from_numpy(base['S0'][:, l].reshape(-1, K).astype(np.int64)).to('cuda')
                p0l = torch.from_numpy(np.ascontiguousarray(base['p0'][:, l]).reshape(-1, NE)).to('cuda')
            for a, b in chunks:
                B = b - a
                rows = slice(a * SEQ, b * SEQ)
                h = H[a]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], eps).reshape(-1, Hd)
                if mode == 'L00':
                    idx = S0l[rows]
                    rw = p0l[rows].gather(-1, idx)
                else:
                    logits = F.linear(z, wq['router'])                              # oc.route arithmetic
                    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    val, top = torch.topk(probs, K, dim=-1)
                    if mode in ('L11', 'fp32'):
                        idx, rw = top, val.to(logits.dtype)
                    elif mode == 'L01':
                        idx = S0l[rows]
                        rw = probs.gather(-1, idx).to(logits.dtype)
                    elif mode == 'L10':
                        idx = top
                        rw = p0l[rows].gather(-1, idx)
                    else:
                        raise ValueError(mode)
                    if record_p:
                        P[a:b, l] = probs.view(B, SEQ, NE).cpu().numpy()
                rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                H[a] = x + oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
            del wq, w
            if mode in ('L00', 'L01', 'L10'):
                del S0l, p0l
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'p': P, 'seconds': time.time() - t0, 'audit': audit, 'chunks': chunks,
            'finite_ce': bool(np.isfinite(ce).all()), 'finite_weights': bool(np.isfinite(rec['weights']).all())}


def eq(a, b, n=None):
    n = n or len(a['ce'])
    return {'ce_max_abs': float(np.max(np.abs(np.array(a['ce'][:n]) - np.array(b['ce'][:n])))),
            'hidden_bitwise_windows': int(sum(x == y for x, y in zip(a['hidden_sha'][:n], b['hidden_sha'][:n]))), 'windows': n}


def bitwise(e):
    return e['ce_max_abs'] == 0 and e['hidden_bitwise_windows'] == e['windows']


def semantics(runs, base):
    """recorded ids / weights against the definitions (CPU, exact)."""
    S0, p0 = base['S0'], base['p0']
    take = lambda ids: np.take_along_axis(p0, ids.astype(np.int64), axis=-1)
    r = runs
    sets = lambda ids: np.sort(ids, -1)
    res = {'L00_ids_equal_S0': bool(np.array_equal(r['L00']['routes']['ids'], S0)),
           'L00_weights_equal_p0_at_S0': bool(np.array_equal(r['L00']['routes']['weights'], take(S0))),
           'L00_weights_equal_baseline_topk_weights': bool(np.array_equal(r['L00']['routes']['weights'], base['w0'])),
           'L01_ids_equal_S0': bool(np.array_equal(r['L01']['routes']['ids'], S0)),
           'L01_weights_differ_from_p0_share': float((r['L01']['routes']['weights'] != take(S0)).mean()),
           'L10_weights_equal_p0_at_own_ids': bool(np.array_equal(r['L10']['routes']['weights'], take(r['L10']['routes']['ids']))),
           'L10_set_differs_from_S0_share': float((sets(r['L10']['routes']['ids']) != sets(S0)).any(-1).mean()),
           'L10_vs_L11_set_differs_share': float((sets(r['L10']['routes']['ids']) != sets(r['L11']['routes']['ids'])).any(-1).mean()),
           'L01_vs_L11_weights_differ_share_where_L11_set_equals_S0': None,
           'weight_sum_max': {c: float(r[c]['routes']['weights'].sum(-1).max()) for c in CONDS},
           'ids_in_range_and_distinct': {c: bool(r[c]['routes']['ids'].min() >= 0 and r[c]['routes']['ids'].max() < NE
                                                 and (np.diff(sets(r[c]['routes']['ids']), axis=-1) > 0).all()) for c in CONDS}}
    m = (sets(r['L11']['routes']['ids']) == sets(S0)).all(-1)
    if m.any():
        a = np.take_along_axis(r['L01']['routes']['weights'], np.argsort(r['L01']['routes']['ids'], -1), -1)
        b = np.take_along_axis(r['L11']['routes']['weights'], np.argsort(r['L11']['routes']['ids'], -1), -1)
        res['L01_vs_L11_weights_differ_share_where_L11_set_equals_S0'] = float((a[m] != b[m]).any(-1).mean())
    res['no_renormalization'] = all(v < 1.0 for v in res['weight_sum_max'].values())
    res['pass'] = bool(res['L00_ids_equal_S0'] and res['L00_weights_equal_p0_at_S0'] and res['L00_weights_equal_baseline_topk_weights']
                       and res['L01_ids_equal_S0'] and res['L10_weights_equal_p0_at_own_ids'] and res['no_renormalization']
                       and all(res['ids_in_range_and_distinct'].values()))
    return res


def baseline(toks, dq, ranges, log, label):
    f = forward4(toks, None, dq, ranges, 'fp32', record_p=True, log=log, label=label)
    base = {'S0': f['routes']['ids'], 'w0': f['routes']['weights'], 'p0': f['p']}
    return f, base


def run_conditions(toks, fm, dq, ranges, base, log, tag):
    return {c: forward4(toks, fm, dq, ranges, c, base=base, log=log, label=f'{tag} {c}') for c in CONDS}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    import olmoe_common as oc
    from scipy.stats import norm
    from mp_auto import config_hash
    mp = read(MPT / 'plan.json')
    if sha(CONFIG) != mp['configs']['aggressive']['sha256']:
        raise SystemExit('aggressive config hash differs from the multipoint plan')
    tok = oc.tokenizer()
    used = dr.used_spans_all()
    for f in (PREV1 / 'segments.json', MPT / 'segments.json'):
        for s in read(f):
            used[s['split']].append(tuple(s['article_chars'])); used[s['split']].append(tuple(s['window_chars']))
    arts, _, _, _ = dr.strict_articles('train', tok)
    elig = sum(1 for a in arts if a['tokens'][1] - a['tokens'][0] >= WPA * SEQ
               and not any(a['chars'][0] < u1 and a['chars'][1] > u0 for u0, u1 in used['train']))
    z3 = float(norm.ppf(1 - ALPHA / 6))
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [CONFIG, dr.RANGES, MPT / 'plan.json', MPT / 'analysis.json', MPT / 'segments.json',
                                                         MPT / 'tokens.npy', MPT / 'routes/fp32_dyn.npz', MPT / 'eval_aggressive.json',
                                                         MPT / 'capture.json', PREV1 / 'segments.json', dr.DATA / 'train-00000-of-00001.parquet']}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'For the aggressive BC configuration, does the extra dynamic-route damage come from expert selection, '
                    'routing weights, or their joint / cascading action?',
        'configuration': {'path': str(CONFIG.relative_to(HERE)), 'sha256': sha(CONFIG), 'config_hash': config_hash(read(CONFIG)),
                          'R': mp['configs']['aggressive']['R_frontier'], 'semantics': 'unchanged: 3051 searched projections, 21 '
                          'absent projections bf16 weights + FP32 activations, router / attention / norms / head FP32'},
        'conditions': {
            'baseline': 'complete FP32 dynamic forward; per layer and token S0 = top-8 ids and p0 = softmax over all 64 experts '
                        '(float32), stored in full',
            'L00': 'ids S0, weights p0[S0] (fully fixed)', 'L01': 'ids S0, weights p[S0] (p from the condition own hidden state)',
            'L10': 'ids S = top-8 of the condition own p, weights p0[S] (FP32 probability function; the gathered values and '
                   'their sum can change with S)', 'L11': 'ids S, weights p[S] (fully dynamic)',
            'shared': 'each condition runs its own complete forward with the aggressive configuration; p and S come from its '
                      'own hidden states (never from a recorded L11 route); no renormalization (norm_topk_prob false); '
                      'run_olmoe_dynamic_route.forward arithmetic (oc.route: FP32 logits, float32 softmax, topk); chunks of '
                      f'{CHUNK} windows, tail chunk = remainder, identical for every condition; GPU FP32, TF32 off; '
                      '127 predicted tokens per window; GPU QDQ behaviour, not native kernels',
            'caveat': 'the four conditions produce different downstream hidden states; effects are end-to-end interventions, '
                      'not local effects at a shared hidden state'},
        'controls': [
            'pilot (old multipoint windows 0-127): FP32 baseline S0 / top-8 weights / CE equal the multipoint capture bitwise; '
            'L00 == multipoint aggressive q_fix and L11 == q_dyn (CE bitwise)',
            'quantization off (control subset = first chunk): L00, L01, L10, L11 each == FP32 baseline (CE, hidden bitwise)',
            'reruns of every condition on the control subset bitwise (CE, hidden, ids, weights)',
            'replay: every condition\'s recorded ids / weights through run_olmoe_dynamic_route.forward (fixed routes) '
            'reproduces the condition (CE, hidden bitwise, all windows)',
            'semantics: L00 ids = S0 and weights = p0[S0] = baseline weights; L01 ids = S0; L10 weights = p0 gathered at its '
            'own ids; weight sums < 1 (no renormalization); ids in [0, 64) and distinct; L10 vs L11 and L01 vs L11 differ '
            '(evidence of own hidden states); config completeness, finite values, router untouched, 127 tokens per window',
            'any failure: fix and record before analysing losses'],
        'pilot_and_sample_size': {
            'pilot': f'old windows: run_olmoe_dynamic_multipoint windows 0-{PILOT_WINDOWS - 1} (32 articles x 4); never part of '
                     'the confirmation set',
            'rule': 'n = min(eligible, ceil((z s_max / h)^2)), z = Phi^-1(1 - 0.05/6) = %.4f (two-sided Bonferroni over the 3 '
                    'primary contrasts), h = %.3f, s_max = largest pilot article-level SD among E_weight, E_ID, I' % (z3, H_TARGET),
            'eligible_train_articles': elig,
            'note': 'if n is capped by eligibility the achievable half-width is reported; no new data source this round '
                    '(WikiText-103 overlaps WikiText-2 and would need its own registration)'},
        'data': {'source': 'WikiText-2 raw train, strict article boundaries; whole-article exclusion of every earlier OLMoE '
                           'set (stage 1, multipoint, solver regression, search_final C, numerics B, D/V, H, J, C1-C8, both '
                           'dynamic-route rounds)', 'segment': f'{WPA} consecutive 128-token windows per article, uniform start, '
                           f'seed {SEEDS["data"]}; all conditions share the windows', 'eligible': elig},
        'contrasts': {
            'per_article': 'CE(cond) = summed CE of 4 windows / 508',
            'primary_family': 'E_weight = L01 - L00; E_ID = L10 - L00; I = L11 - L10 - L01 + L00 (E_total = E_weight + E_ID + I). '
                              'Bonferroni over the three: two-sided 98.33% intervals; unadjusted 95% also shown',
            'E_total': 'L11 - L00 (replication of the extra dynamic damage), 95% CI, compared with eps (exceeds if lower > eps) '
                       'and with the previous +0.00753 [0.00606, 0.00897]',
            'secondary': 'E_weight - E_ID (paired 95% CI)', 'descriptive': 'L11 - L01, L11 - L10',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix for all conditions, seed {SEEDS["bootstrap"]}; '
                         'single source (train), no strata; PPL = exp(mean CE)',
            'signs': 'no sign is assumed; effects are not non-negative shares'},
        'interpretation': {
            'weight_first': 'E_weight Bonferroni CI lower > 0 and both E_ID and I Bonferroni CIs inside [-eps, +eps]',
            'id_first': 'E_ID Bonferroni CI lower > 0 and both E_weight and I Bonferroni CIs inside [-eps, +eps]',
            'joint': 'I Bonferroni CI excludes 0: study joint / cross-layer accumulation first, no split into two penalties',
            'overlap': 'E_weight and E_ID both Bonferroni-positive, each point >= 0.5 E_total point, I upper < 0: overlapping '
                       'effects; do not claim either explains all damage',
            'undetermined': 'otherwise: mechanisms not separated; no story from point estimates',
            'precedence': 'checked in the order weight_first, id_first, overlap (when its condition holds), joint, undetermined',
            'replication': 'E_total differing from the previous round: report sources and uncertainty, no re-picking'},
        'routing_statistics': 'per condition and layer vs the baseline (positions 0-126): set change vs S0, overlap, aligned '
                              'weight L1 vs p0[S0], change of the selected-weight sum, dispatch TVD; L10 vs L11 set divergence; '
                              'per-article route change vs loss (descriptive, no causal attribution)',
        'not_done': 'no margin-stratified release, no per-layer switching, no precision changes, no normalization change, '
                    'no BC / search / config change',
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_route_decomposition.py', 'run_olmoe_dynamic_multipoint.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE aggressive 配置的 expert 選擇／routing weight 干預', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          f'## 配置', '', f'- `{p["configuration"]["path"]}`（sha256 `{p["configuration"]["sha256"][:12]}…`，R {p["configuration"]["R"]:.4f}），與上一輪相同；未進成本表的 21 個投影仍為 bf16 權重 + FP32 activation；router FP32。', '',
          '## 四個條件（各自完整 forward；p、S 來自各自 hidden state）', '',
          '- 基準：完整 FP32 dynamic，保存每層每 token 的 S0（top-8）與 p0（全部 64 個 softmax 機率）。',
          '- L00：IDs = S0，weights = p0[S0]。', '- L01：IDs = S0，weights = p[S0]。',
          '- L10：IDs = S（自身 top-8），weights = p0[S]（固定 FP32 機率函數，取值與總和可隨 S 改變）。', '- L11：IDs = S，weights = p[S]。',
          '- 不重新 normalize；四條件共用 `run_olmoe_dynamic_route.forward` 的算術；每批 128 窗口、尾批為剩餘；結果是端到端干預，不是同一 hidden state 的局部效應。', '',
          '## 控制', ''] + [f'- {c}' for c in p['controls']] + ['',
          '## 樣本數', '', f'- Pilot：上一輪 multipoint 的窗口 0–127（32 篇舊文章，只作控制與變異估計）。',
          f'- n = min(可用篇數, ceil((z·s_max/h)²))，z = Φ⁻¹(1−0.05/6) = {z3:.4f}，h = {H_TARGET}，s_max 為 pilot 中 E_weight、E_ID、I 的最大文章級 SD；可用 {elig} 篇。',
          '- 若受可用篇數限制，報告可達精度；本輪不加新來源。', '',
          '## 對比與統計', '', '- 主要族：E_weight = L01−L00、E_ID = L10−L00、I = L11−L10−L01+L00；Bonferroni ×3（98.33% 區間），另列 95%。',
          '- E_total = L11−L00：重現指標，95% CI 對 ε = 0.002 與上一輪 +0.00753。', '- 次要：E_weight − E_ID（95%）；描述：L11−L01、L11−L10。',
          f'- {BOOT} 次 paired article bootstrap（seed {SEEDS["bootstrap"]}）。不預設正負，不當作非負占比。', '',
          '## 判讀', ''] + [f'- {k}：{v}' for k, v in p['interpretation'].items()] + ['',
          '## 不做', '', '- margin 分層、逐層開關、提高精度、改 normalization、改 BC／搜尋／配置。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    log(f'plan written; eligible {elig}')


def pilot(out, log):
    from run_olmoe_search_final import dev_quant
    from scipy.stats import norm
    dr.gpu_setup()
    if (out / 'pilot.json').exists():
        raise SystemExit('pilot exists')
    toks = np.load(MPT / 'tokens.npy')[:PILOT_WINDOWS]
    cap, ev = read(MPT / 'capture.json'), read(MPT / 'eval_aggressive.json')
    mr = dr.load_routes_npz(MPT / 'routes/fp32_dyn.npz')
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = read(CONFIG), dr.read(dr.RANGES)
    f, base = baseline(toks, dq, ranges, log, 'pilot fp32')
    runs = run_conditions(toks, fm, dq, ranges, base, log, 'pilot')
    mx = lambda a, b: float(np.max(np.abs(np.array(a) - np.array(b[:PILOT_WINDOWS]))))
    res = {'program_control': {
        'baseline_ids_equal_multipoint': bool(np.array_equal(base['S0'], mr['ids'][:PILOT_WINDOWS])),
        'baseline_weights_equal_multipoint': bool(np.array_equal(base['w0'], mr['weights'][:PILOT_WINDOWS])),
        'baseline_ce_max_abs_vs_capture': mx(f['ce'], cap['ce']),
        'baseline_hidden_bitwise_vs_capture': int(sum(x == y for x, y in zip(f['hidden_sha'], cap['hidden_sha'][:PILOT_WINDOWS]))),
        'L00_ce_max_abs_vs_q_fix': mx(runs['L00']['ce'], ev['ce']['q_fix']),
        'L11_ce_max_abs_vs_q_dyn': mx(runs['L11']['ce'], ev['ce']['q_dyn'])},
        'semantics': semantics(runs, base)}
    pc = res['program_control']
    res['pass'] = bool(pc['baseline_ids_equal_multipoint'] and pc['baseline_weights_equal_multipoint'] and pc['baseline_ce_max_abs_vs_capture'] == 0
                       and pc['baseline_hidden_bitwise_vs_capture'] == PILOT_WINDOWS and pc['L00_ce_max_abs_vs_q_fix'] == 0
                       and pc['L11_ce_max_abs_vs_q_dyn'] == 0 and res['semantics']['pass'])
    art = {c: np.array(runs[c]['ce']).reshape(-1, WPA).sum(1) / (WPA * TP) for c in CONDS}
    E = {'E_weight': art['L01'] - art['L00'], 'E_ID': art['L10'] - art['L00'], 'I': art['L11'] - art['L10'] - art['L01'] + art['L00'],
         'E_total': art['L11'] - art['L00']}
    res['pilot_estimates'] = {k: {'mean': float(v.mean()), 'article_sd': float(v.std(ddof=1))} for k, v in E.items()}
    s_max = max(res['pilot_estimates'][k]['article_sd'] for k in ('E_weight', 'E_ID', 'I'))
    z3 = float(norm.ppf(1 - ALPHA / 6))
    elig = read(out / 'plan.json')['pilot_and_sample_size']['eligible_train_articles']
    n_req = math.ceil((z3 * s_max / H_TARGET) ** 2)
    n = int(min(elig, n_req))
    res['sample_size'] = {'s_max': s_max, 'z': z3, 'n_required': n_req, 'eligible': elig, 'n_articles': n,
                          'expected_bonferroni_half_width': {k: z3 * res['pilot_estimates'][k]['article_sd'] / math.sqrt(n) for k in ('E_weight', 'E_ID', 'I')},
                          'expected_95_half_width_E_total': 1.959964 * res['pilot_estimates']['E_total']['article_sd'] / math.sqrt(n)}
    res['seconds'] = {'baseline': f['seconds'], **{c: runs[c]['seconds'] for c in CONDS}}
    res['environment'] = dr.env_info()
    write(out / 'pilot.json', res)
    log(json.dumps({'pass': res['pass'], 'sample_size': res['sample_size']}))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('data exists')
    pl, pi = read(out / 'plan.json'), read(out / 'pilot.json')
    if not pi['pass']:
        raise SystemExit('pilot controls failed')
    for k, v in pl['inputs'].items():
        if sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    n = pi['sample_size']['n_articles']
    used = dr.used_spans_all()
    for f in (PREV1 / 'segments.json', MPT / 'segments.json'):
        for s in read(f):
            used[s['split']].append(tuple(s['article_chars'])); used[s['split']].append(tuple(s['window_chars']))
    tok = oc.tokenizer()
    arts, ids, off, _ = dr.strict_articles('train', tok)
    free = [a for a in arts if a['tokens'][1] - a['tokens'][0] >= WPA * SEQ
            and not any(a['chars'][0] < u1 and a['chars'][1] > u0 for u0, u1 in used['train'])]
    if len(free) != pl['pilot_and_sample_size']['eligible_train_articles']:
        raise SystemExit('eligible count changed')
    rng = np.random.default_rng(SEEDS['data'])
    pick = sorted(rng.choice(len(free), n, replace=False).tolist())
    segs, toks = [], []
    for k in pick:
        a = free[k]
        st = int(rng.integers(a['tokens'][0], a['tokens'][1] - WPA * SEQ + 1))
        wc = [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])]
        assert a['chars'][0] <= wc[0] and wc[1] <= a['chars'][1]
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w)
        segs.append({'index': len(segs), 'split': 'train', 'title': a['title'], 'article_chars': a['chars'], 'article_tokens': a['tokens'],
                     'window_chars': wc, 'tokens': [st, st + WPA * SEQ], 'windows': [WPA * len(segs) + i for i in range(WPA)],
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
    if len({s['title'] for s in segs}) != len(segs):
        raise SystemExit('duplicate article')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'), 'pilot_sha256': sha(out / 'pilot.json'),
                                     'n_articles': n, 'eligible': len(free), 'segments_sha256': sha(out / 'segments.json'), 'tokens_sha256': sha(out / 'tokens.npy')})
    log(f'data frozen: {n} articles')


def run(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'results.json').exists():
        raise SystemExit('results exist')
    toks = np.load(out / 'tokens.npy')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('tokens changed')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = read(CONFIG), dr.read(dr.RANGES)
    if sha(CONFIG) != read(out / 'plan.json')['configuration']['sha256']:
        raise SystemExit('config changed')
    cc = dr.config_checks(fm, dq, ranges)
    f, base = baseline(toks, dq, ranges, log, 'baseline')
    np.savez(out / 'baseline_routes.npz', S0=base['S0'], w0=base['w0'], p0=base['p0'], tokens=toks)
    runs = run_conditions(toks, fm, dq, ranges, base, log, 'main')
    (out / 'routes').mkdir(exist_ok=True)
    for c in CONDS:
        np.savez(out / f'routes/{c}.npz', ids=runs[c]['routes']['ids'], weights=runs[c]['routes']['weights'])
    sem = semantics(runs, base)
    ctl = {'semantics': sem, 'config_checks': cc}
    sub_toks = toks[:CTRL]
    sub_base = {k: v[:CTRL] for k, v in base.items()}
    ctl['quant_off'] = {c: eq(forward4(sub_toks, None, dq, ranges, c, base=sub_base, log=log, label=f'off {c}'), f, CTRL) for c in CONDS}
    ctl['rerun'] = {}
    for c in CONDS:
        r2 = forward4(sub_toks, fm, dq, ranges, c, base=sub_base, log=log, label=f'rerun {c}')
        e = eq(r2, runs[c], CTRL)
        e['routes_equal'] = bool(np.array_equal(r2['routes']['ids'], runs[c]['routes']['ids'][:CTRL])
                                 and np.array_equal(r2['routes']['weights'], runs[c]['routes']['weights'][:CTRL]))
        ctl['rerun'][c] = e
    ctl['replay'] = {}
    rep_sec = {}
    for c in CONDS:
        rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs[c]['routes']), log=log, label=f'replay {c}')
        ctl['replay'][c] = eq(rp, runs[c]); rep_sec[c] = rp['seconds']
    ctl['finite'] = all(runs[c]['finite_ce'] and runs[c]['finite_weights'] for c in CONDS) and f['finite_ce']
    ctl['audit'] = {c: runs[c]['audit'] for c in CONDS}
    ctl['pass'] = bool(sem['pass'] and cc['pass'] and ctl['finite'] and all(bitwise(v) for v in ctl['quant_off'].values())
                       and all(bitwise(v) and v['routes_equal'] for v in ctl['rerun'].values()) and all(bitwise(v) for v in ctl['replay'].values())
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'results.json', {'ce': {'fp32': f['ce'], **{c: runs[c]['ce'] for c in CONDS}}, 'controls': ctl,
                                 'seconds': {'baseline': f['seconds'], **{c: runs[c]['seconds'] for c in CONDS}, 'replay': rep_sec},
                                 'environment': dr.env_info()})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    from scipy.stats import spearmanr
    t0 = time.time()
    rs = read(out / 'results.json')
    if not rs['controls']['pass']:
        raise SystemExit('controls failed: do not analyse losses')
    segs = read(out / 'segments.json'); nA = len(segs)
    art = {c: np.array(v).reshape(nA, WPA).sum(1) / (WPA * TP) for c, v in rs['ce'].items()}
    E = {'E_weight': art['L01'] - art['L00'], 'E_ID': art['L10'] - art['L00'], 'I': art['L11'] - art['L10'] - art['L01'] + art['L00'],
         'E_total': art['L11'] - art['L00'], 'E_weight_minus_E_ID': (art['L01'] - art['L00']) - (art['L10'] - art['L00']),
         'L11_minus_L01': art['L11'] - art['L01'], 'L11_minus_L10': art['L11'] - art['L10'],
         'D_L00': art['L00'] - art['fp32'], 'D_L11': art['L11'] - art['fp32']}
    rng = np.random.default_rng(SEEDS['bootstrap'])
    idx = rng.integers(0, nA, (BOOT, nA))
    lv = 100 * ALPHA / 6

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci_bonf3': np.percentile(d, [lv, 100 - lv]).tolist(),
                'article_sd': float(x.std(ddof=1)), 'articles_positive': int((x > 0).sum())}
    res = {'n_articles': nA, 'contrasts': {k: st(v) for k, v in E.items()}}
    c = res['contrasts']
    inside = lambda k: -EPS <= c[k]['ci_bonf3'][0] and c[k]['ci_bonf3'][1] <= EPS
    pos = lambda k: c[k]['ci_bonf3'][0] > 0
    res['flags'] = {'E_weight_bonf_positive': pos('E_weight'), 'E_ID_bonf_positive': pos('E_ID'),
                    'I_bonf_excludes_0': c['I']['ci_bonf3'][0] > 0 or c['I']['ci_bonf3'][1] < 0,
                    'E_weight_inside_eps': inside('E_weight'), 'E_ID_inside_eps': inside('E_ID'), 'I_inside_eps': inside('I'),
                    'E_total_exceeds_eps_95': c['E_total']['ci95'][0] > EPS,
                    'E_total_previous_point_in_new_ci95': c['E_total']['ci95'][0] <= PREV_AGGR['mean'] <= c['E_total']['ci95'][1]}
    fl = res['flags']
    if fl['E_weight_bonf_positive'] and fl['E_ID_inside_eps'] and fl['I_inside_eps']:
        res['interpretation'] = 'weight_first'
    elif fl['E_ID_bonf_positive'] and fl['E_weight_inside_eps'] and fl['I_inside_eps']:
        res['interpretation'] = 'id_first'
    elif (fl['E_weight_bonf_positive'] and fl['E_ID_bonf_positive'] and c['I']['ci_bonf3'][1] < 0
          and min(c['E_weight']['mean'], c['E_ID']['mean']) >= 0.5 * c['E_total']['mean']):
        res['interpretation'] = 'overlap'
    elif fl['I_bonf_excludes_0']:
        res['interpretation'] = 'joint'
    else:
        res['interpretation'] = 'undetermined'
    tokens = nA * WPA * TP
    res['ce_ppl'] = {k: {'ce': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for k, v in rs['ce'].items()}
    # routing statistics vs the baseline, positions 0..126
    b = np.load(out / 'baseline_routes.npz')
    S0 = b['S0'][:, :, :TP].astype(np.int64); w0 = b['w0'][:, :, :TP]
    sets = lambda a: np.sort(a, -1)
    R = {}
    for cnd in CONDS:
        z = np.load(out / f'routes/{cnd}.npz')
        ids, w = z['ids'][:, :, :TP].astype(np.int64), z['weights'][:, :, :TP]
        inter = (S0[..., :, None] == ids[..., None, :]).any(-1).sum(-1)
        layers = []
        for l in range(NL):
            n_ = S0.shape[0] * TP
            df, dq_ = np.zeros((n_, NE)), np.zeros((n_, NE))
            r_ = np.repeat(np.arange(n_), K)
            np.add.at(df, (r_, S0[:, l].reshape(-1)), w0[:, l].reshape(-1)); np.add.at(dq_, (r_, ids[:, l].reshape(-1)), w[:, l].reshape(-1))
            cf, cq = np.bincount(S0[:, l].ravel(), minlength=NE), np.bincount(ids[:, l].ravel(), minlength=NE)
            layers.append({'condition': cnd, 'layer': l, 'set_change_vs_S0': float((inter[:, l] < K).mean()), 'overlap': float(inter[:, l].mean() / K),
                           'weight_l1_vs_baseline': float(np.abs(dq_ - df).sum(1).mean()),
                           'weight_sum_change_mean': float((w[:, l].sum(-1) - w0[:, l].sum(-1)).mean()),
                           'weight_sum_change_abs_mean': float(np.abs(w[:, l].sum(-1) - w0[:, l].sum(-1)).mean()),
                           'dispatch_tvd': float(0.5 * np.abs(cq / cq.sum() - cf / cf.sum()).sum())})
        R[cnd] = {'layers': layers, 'set_change_vs_S0': float((inter < K).mean()), 'overlap': float(inter.mean() / K),
                  'per_article_set_change': (inter < K).reshape(nA, WPA, NL, TP).mean((1, 2, 3))}
    z10, z11 = np.load(out / 'routes/L10.npz'), np.load(out / 'routes/L11.npz')
    div = (sets(z10['ids'][:, :, :TP]) != sets(z11['ids'][:, :, :TP])).any(-1)
    res['routing'] = {cnd: {'set_change_vs_S0': R[cnd]['set_change_vs_S0'], 'overlap': R[cnd]['overlap']} for cnd in CONDS}
    res['routing']['L10_vs_L11_set_divergence'] = {'overall': float(div.mean()), 'per_layer': [float(div[:, l].mean()) for l in range(NL)]}
    res['routing']['per_article_spearman'] = {
        'L11_set_change_vs_E_total': float(spearmanr(R['L11']['per_article_set_change'], E['E_total']).statistic),
        'L10_set_change_vs_E_ID': float(spearmanr(R['L10']['per_article_set_change'], E['E_ID']).statistic)}
    with open(out / 'per_layer_routing.csv', 'w') as fh:
        rows = [x for cnd in CONDS for x in R[cnd]['layers']]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    with open(out / 'per_article.csv', 'w') as fh:
        rows = [{'index': s['index'], 'title': s['title'], **{f'ce_{k}': float(v[i]) for k, v in art.items()},
                 **{k: float(v[i]) for k, v in E.items() if k in ('E_weight', 'E_ID', 'I', 'E_total')},
                 **{f'set_change_{cnd}': float(R[cnd]['per_article_set_change'][i]) for cnd in ('L10', 'L11')}} for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    res['controls'] = rs['controls']
    res['timing'] = {'run': rs['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    write(out / 'analysis.json', res)
    log(json.dumps({k: [c[k]['mean'], c[k]['ci_bonf3']] for k in ('E_weight', 'E_ID', 'I', 'E_total')}) + ' ' + res['interpretation'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'pilot', 'data', 'run', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'pilot': pilot, 'data': data, 'run': run, 'analyze': analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
