#!/usr/bin/env python3
"""OLMoE aggressive BC configuration, dynamic expert selection with FP32 relative proportions restored on the set the
model selects itself (keeping the condition's own selected sum). Diagnostic with FP32 information, not deployable.

Per layer of each condition's own forward (own hidden state -> FP32 router -> p over 64 experts):
  L11     S = top-8(p), weights p[S]                                  (full dynamic, comparison)
  Ldprop  S = top-8(p), weights p0[S] * (m / m0S), m = sum p[S], m0S = sum p0[S]   (float32, sums in S order)
  L00     S0, p0[S0]                                                  (fully fixed, reference)
p0 = the complete FP32 dynamic baseline's softmax over all 64 experts at the same token / layer. No renormalization to 1,
no logit change, no clipping / epsilon. L00 / L11 / baseline use the unchanged run_olmoe_route_mass_restore.forward5.

Data: WikiText-103 raw train, same dedup rules as run_olmoe_route_mass_restore, excluding its 103 articles.
Stages: plan, pilot (old windows: run_olmoe_route_mass_restore windows 0-127), data, run, analyze.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
MR = HERE / 'runs/olmoe_route_mass_restore_20261018'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK, CTRL, PILOT_WINDOWS = 128, 128, 128
EPS, ALPHA, BOOT = 0.002, 0.05, 10000
H_TARGET, N_MIN, N_MAX = 0.0015, 64, 300
SEEDS = {'order': 18141, 'start': 18142, 'bootstrap': 18151}


def forward_dprop(toks_np, fm, dq, ranges, base, log=print, label='', chunk=CHUNK):
    """run_olmoe_route_mass_restore.forward5 arithmetic; route: S = top-8 of the own p, weights p0[S] * (m / m0S)."""
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
    extra = {'moe_norm': np.zeros((nW, NL), np.float32), 'scale': np.zeros((nW, NL, SEQ), np.float32), 'm': np.zeros((nW, NL, SEQ), np.float32),
             'm0S': np.zeros((nW, NL, SEQ), np.float32), 'w_raw': np.zeros((nW, NL, SEQ, K), np.float32)}
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
            p0l = torch.from_numpy(np.ascontiguousarray(base['p0'][:, l]).reshape(-1, NE)).to('cuda')
            for a, b in chunks:
                B = b - a
                rows = slice(a * SEQ, b * SEQ)
                h = H[a]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], eps).reshape(-1, Hd)
                logits = F.linear(z, wq['router'])
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                val, idx = torch.topk(probs, K, dim=-1)
                wcur = val.to(torch.float32)
                m = torch.sum(wcur, dim=-1, keepdim=True)
                p0S = p0l[rows].gather(-1, idx)
                m0S = torch.sum(p0S, dim=-1, keepdim=True)
                if not (bool(torch.isfinite(m).all()) and bool(torch.isfinite(m0S).all()) and bool((m0S > 0).all())):
                    raise RuntimeError(f'Ldprop: zero / non-finite sum at layer {l}')
                scale = m / m0S
                rw = p0S * scale
                if not (bool(torch.isfinite(scale).all()) and bool(torch.isfinite(rw).all())):
                    raise RuntimeError(f'Ldprop: non-finite scale / weights at layer {l}')
                extra['scale'][a:b, l] = scale.view(B, SEQ).cpu().numpy()
                extra['m'][a:b, l] = m.view(B, SEQ).cpu().numpy()
                extra['m0S'][a:b, l] = m0S.view(B, SEQ).cpu().numpy()
                extra['w_raw'][a:b, l] = wcur.view(B, SEQ, K).cpu().numpy()
                rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                moe = oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
                extra['moe_norm'][a:b, l] = moe[:, :TP].norm(dim=-1).mean(-1).cpu().numpy()
                H[a] = x + moe
            del wq, w, p0l
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'extra': extra, 'seconds': time.time() - t0, 'audit': audit, 'chunks': chunks,
            'finite_ce': bool(np.isfinite(ce).all()), 'finite_weights': bool(np.isfinite(rec['weights']).all())}


def semantics(r, base, l11=None):
    ids, W, e = r['routes']['ids'], r['routes']['weights'], r['extra']
    p0S = np.take_along_axis(base['p0'], ids.astype(np.int64), axis=-1)
    W64 = W.astype(np.float64)
    ratio = W64 / np.where(p0S != 0, p0S.astype(np.float64), np.nan)
    sets = lambda a: np.sort(a, -1)
    res = {'weights_equal_float32_p0S_times_scale': bool(np.array_equal(W, (p0S * e['scale'][..., None]).astype(np.float32))),
           'recorded_m0S_minus_float32_sum_p0S_max_abs': float(np.max(np.abs(e['m0S'] - p0S.sum(-1, dtype=np.float32)))),
           'recorded_m_minus_float32_sum_w_raw_max_abs': float(np.max(np.abs(e['m'] - e['w_raw'].sum(-1, dtype=np.float32)))),
           'sum_minus_m_max_rel': float(np.max(np.abs(W64.sum(-1) - e['w_raw'].astype(np.float64).sum(-1)) / e['w_raw'].astype(np.float64).sum(-1))),
           'proportion_ratio_to_p0S_max_rel_spread': float(np.nanmax((np.nanmax(ratio, -1) - np.nanmin(ratio, -1)) / np.nanmean(ratio, -1))),
           'w_raw_descending_topk': bool((np.diff(e['w_raw'], axis=-1) <= 0).all()),
           'set_differs_from_S0_share': float((sets(ids) != sets(base['S0'])).any(-1).mean()),
           'sum_equals_1_share': float((np.abs(W64.sum(-1) - 1) < 1e-6).mean()),
           'scale_min': float(e['scale'].min()), 'scale_max': float(e['scale'].max()),
           'ids_in_range_and_distinct': bool(ids.min() >= 0 and ids.max() < NE and (np.diff(sets(ids), axis=-1) > 0).all())}
    if l11 is not None:
        res['set_differs_from_L11_share'] = float((sets(ids) != sets(l11['routes']['ids'])).any(-1).mean())
    res['pass'] = bool(res['weights_equal_float32_p0S_times_scale'] and res['recorded_m0S_minus_float32_sum_p0S_max_abs'] <= 1e-6
                       and res['recorded_m_minus_float32_sum_w_raw_max_abs'] <= 1e-6 and res['sum_minus_m_max_rel'] < 1e-5
                       and res['proportion_ratio_to_p0S_max_rel_spread'] < 1e-5 and res['w_raw_descending_topk'] and res['ids_in_range_and_distinct'])
    return res


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    mp = mr.read(MR / 'plan.json')
    inputs = {str(p.relative_to(HERE)): mr.sha(p) for p in [mr.CONFIG, dr.RANGES, MR / 'plan.json', MR / 'segments.json', MR / 'data_freeze.json',
                                                            MR / 'tokens.npy', MR / 'results.json', MR / 'baseline_routes.npz'] + mr.WT103 + mr.WT2}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'With the model selecting its own experts, does restoring FP32 relative proportions on the selected set '
                    '(keeping the condition own selected sum) reduce the full dynamic-route loss? FP32-informed diagnostic.',
        'configuration': {'path': str(mr.CONFIG.relative_to(HERE)), 'sha256': mr.sha(mr.CONFIG), 'R': mp['configuration']['R']},
        'conditions': {
            'baseline': 'complete FP32 dynamic; S0 and p0 (float32 softmax over 64) per layer and token',
            'L11': 'S = top-8(p) of the own hidden state, weights p[S] (full dynamic; run_olmoe_route_mass_restore.forward5)',
            'Ldprop': 'S = top-8(p) of the own hidden state (torch.topk order); m = torch.sum(p[S]); p0S = p0[S] (FP32 baseline '
                      'probabilities of the experts selected now, possibly outside S0); m0S = torch.sum(p0S); scale = m / m0S; '
                      'weights = p0S * scale; float32; no renormalization to 1, no logit change, no clipping / epsilon; zero or '
                      'non-finite m0S, scale or weights stop the run. Equals FP32 exactly when quantization is off (S = S0, scale 1)',
            'L00': 'S0, p0[S0] (fully fixed; reference for the extra dynamic damage)',
            'shared': 'own complete forward per condition, aggressive configuration, router FP32, GPU FP32, TF32 off, chunks of '
                      '128 windows (tail = remainder), 127 predicted tokens per window; GPU QDQ behaviour, not native kernels',
            'caveat': 'selections and later hidden states differ between L11 and Ldprop: end-to-end intervention; uses FP32 '
                      'p0, so not deployable'},
        'data': {'source': 'Salesforce/wikitext wikitext-103-raw-v1 train (sha256 checked), strict article boundaries; WikiText-2 '
                           'title / content matches and within-WT103 duplicate groups removed (as run_olmoe_route_mass_restore); '
                           'the 103 articles of run_olmoe_route_mass_restore (also used by run_olmoe_route_factorial) excluded',
                 'sampling': f'eligible articles in a new random order (seed {SEEDS["order"]}); per article tokenize alone, skip if < 512 '
                             'tokens, skip if >= 10% of its word 5-gram shingles occur in one WikiText-2 article, one previously '
                             'used WT103 article or one already accepted article; uniform start (seed '
                             f'{SEEDS["start"]}), 4 consecutive 128-token windows; nothing depends on routes / loss'},
        'pilot_and_sample_size': {
            'pilot': f'old windows: run_olmoe_route_mass_restore windows 0-{PILOT_WINDOWS - 1} (32 articles): baseline, L00, L11 must '
                     'reproduce that run bitwise; article SD s of Ldprop - L11; never part of the confirmation set',
            'rule': f'n = min({N_MAX}, max({N_MIN}, ceil((1.96 s / {H_TARGET})^2)))'},
        'endpoints': {
            'primary': 'Delta_dprop = CE(Ldprop) - CE(L11), 95% percentile CI; negative = restoring FP32 proportions on the '
                       'dynamically selected set helps',
            'secondary_family': 'Bonferroni over two (two-sided 97.5% intervals): E_total = CE(L11) - CE(L00) (replication, '
                                'positive if lower > 0; compared with eps); Residual_dprop = CE(Ldprop) - CE(L00) (equivalent '
                                'within +-eps if inside, positive if lower > 0)',
            'eps': f'{EPS} nat/token; CI containing 0 is not equivalence',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix, seed {SEEDS["bootstrap"]}; article = 4 windows / '
                         '508 tokens; PPL = exp(mean CE)'},
        'interpretation': {
            'helps': 'Delta_dprop CI entirely below 0: FP32 proportions on the self-selected set reduce the dynamic damage',
            'enough': 'and Residual_dprop inside +-eps: back to the fixed level within tolerance',
            'partial': 'Residual_dprop positive: proportions do not remove all dynamic damage (selection / cascade remains)',
            'null': 'Delta_dprop CI contains 0: no claim; report the excluded improvement range',
            'hurts': 'Delta_dprop CI entirely above 0: reported as is',
            'scope': 'uses FP32 p0: a diagnostic of where the damage sits, not a deployable correction'},
        'controls': ['pilot: baseline (S0, w0, p0, CE), L00 and L11 CE reproduce run_olmoe_route_mass_restore windows 0-127 bitwise',
                     'Ldprop: weights == float32(p0[S] * scale) exactly; recorded m0S and m equal float32 sums (<= 1e-6); '
                     'sum(weights) / m relative < 1e-5; weights / p0[S] constant per token (spread < 1e-5); recorded current '
                     'top-8 weights descending (own topk); ids in range and distinct; share of sets differing from S0 and from L11 reported',
                     'quantization off (first chunk): Ldprop == FP32 bitwise, scale all 1; rerun of Ldprop on the first chunk '
                     'bitwise; replay of Ldprop ids / weights through run_olmoe_dynamic_route.forward bitwise (all windows)',
                     'config completeness, router untouched, hooks on expert projections only, finite values',
                     'any failure: fix and record before analysing losses'],
        'auxiliary': 'per layer: set change vs S0, proportion shift (L1 of normalized weights vs p0[S] / m0S) for L11, sum gap '
                     'm - m0S, scale distribution, MoE output norm; descriptive only',
        'not_done': 'no BC / search / config change, no deployable correction, no first-order term',
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: mr.sha(HERE / f) for f in ['run_olmoe_route_dynamic_prop.py', 'run_olmoe_route_mass_restore.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    mr.write(out / 'plan.json', p)
    md = ['# 預先登記：dynamic 選擇下恢復 FP32 相對比例', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- L11：S = 自身 top-8(p)，weights = p[S]（完整 dynamic，比較對象）。',
          '- Ldprop：S = 自身 top-8(p)；weights = p0[S] × (m／m0S)，m = Σp[S]（當下總和），m0S = Σp0[S]（FP32 對這組 expert 的機率和）；float32；不 normalize 到 1、不改 logits、不 clip；零或非有限值即停止。',
          '- L00：S0、p0[S0]（完全 fixed，參考）。', '- 各自完整 forward；aggressive 配置；router FP32；GPU QDQ。使用 FP32 p0，屬診斷，不可部署。', '',
          '## 資料', '', '- WikiText-103 train，與上兩輪相同的去重規則，另外排除已用的 103 篇；新隨機順序 seed 18141；近重複也比對已用文章。', '',
          '## 樣本數', '', f'- Pilot：上兩輪用過的窗口 0–127（32 篇舊文章）；n = min({N_MAX}, max({N_MIN}, ceil((1.96 s／{H_TARGET})²)))，s 為 Ldprop − L11 的文章級 SD。', '',
          '## 終點', '', '- 主要：Δ_dprop = CE(Ldprop) − CE(L11)，95% CI；負值 = 有幫助。',
          '- 次要（Bonferroni ×2，97.5%）：E_total = L11 − L00（重現；對 ε）；Residual_dprop = Ldprop − L00（±0.002 等價或 >0）。', '',
          '## 判讀', ''] + [f'- {k}：{v}' for k, v in p['interpretation'].items()] + ['', '## 控制', ''] + [f'- {c}' for c in p['controls']] + ['']
    (out / 'plan.md').write_text('\n'.join(md))
    log('plan written')


def pilot(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'pilot.json').exists():
        raise SystemExit('pilot exists')
    toks = np.load(MR / 'tokens.npy')[:PILOT_WINDOWS]
    old, ob = mr.read(MR / 'results.json')['ce'], np.load(MR / 'baseline_routes.npz')
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = mr.read(mr.CONFIG), dr.read(dr.RANGES)
    f, base = mr.baseline(toks, dq, ranges, log, 'pilot fp32')
    runs = {c: mr.forward5(toks, fm, dq, ranges, c, base=base, log=log, label=f'pilot {c}') for c in ('L00', 'L11')}
    runs['Ldprop'] = forward_dprop(toks, fm, dq, ranges, base, log=log, label='pilot Ldprop')
    mx = lambda a, b: float(np.max(np.abs(np.array(a) - np.array(b[:PILOT_WINDOWS]))))
    pc = {'baseline_S0': bool(np.array_equal(base['S0'], ob['S0'][:PILOT_WINDOWS])), 'baseline_w0': bool(np.array_equal(base['w0'], ob['w0'][:PILOT_WINDOWS])),
          'baseline_p0': bool(np.array_equal(base['p0'], ob['p0'][:PILOT_WINDOWS])), 'baseline_ce_max_abs': mx(f['ce'], old['fp32']),
          'L00_ce_max_abs': mx(runs['L00']['ce'], old['L00']), 'L11_ce_max_abs': mx(runs['L11']['ce'], old['L11'])}
    sem = semantics(runs['Ldprop'], base, runs['L11'])
    res = {'program_control': pc, 'semantics': sem,
           'pass': bool(pc['baseline_S0'] and pc['baseline_w0'] and pc['baseline_p0'] and pc['baseline_ce_max_abs'] == 0
                        and pc['L00_ce_max_abs'] == 0 and pc['L11_ce_max_abs'] == 0 and sem['pass'])}
    art = {c: np.array(r['ce']).reshape(-1, WPA).sum(1) / (WPA * TP) for c, r in runs.items()}
    d = art['Ldprop'] - art['L11']
    s = float(d.std(ddof=1))
    n = int(min(N_MAX, max(N_MIN, math.ceil((1.96 * s / H_TARGET) ** 2))))
    res['pilot_estimates'] = {'Delta_dprop_mean': float(d.mean()), 'Delta_dprop_article_sd': s, 'E_total_mean': float((art['L11'] - art['L00']).mean())}
    res['sample_size'] = {'s': s, 'n_required': math.ceil((1.96 * s / H_TARGET) ** 2), 'n_articles': n, 'expected_95_half_width': 1.96 * s / math.sqrt(n)}
    res['seconds'] = {'baseline': f['seconds'], **{c: r['seconds'] for c, r in runs.items()}}
    mr.write(out / 'pilot.json', res)
    log(json.dumps({'pass': res['pass'], 'sample_size': res['sample_size']}))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('data exists')
    pl, pi = mr.read(out / 'plan.json'), mr.read(out / 'pilot.json')
    if not pi['pass']:
        raise SystemExit('pilot controls failed')
    for k, v in pl['inputs'].items():
        if mr.sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    n = pi['sample_size']['n_articles']
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    cnt_t, cnt_c = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    used = {s['content_sha256'] for s in mr.read(MR / 'segments.json')}
    base_elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and cnt_t[a['nt']] == 1 and cnt_c[a['ch']] == 1]
    used_arts = [a for a in base_elig if a['ch'] in used]
    if len(used_arts) != len(used):
        raise SystemExit('previously used articles not all found')
    elig = [a for a in base_elig if a['ch'] not in used]
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(a['body_norm']) for a in used_arts]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    skips, acc_sh, toks, segs, nd = Counter(), [], [], [], []
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc['input_ids'], enc['offset_mapping']
        if len(ids) < WPA * SEQ:
            skips['short'] += 1
            continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if max(c2, cp, ca) >= mr.NEARDUP:
            skips['near_duplicate'] += 1
            nd.append({'title': a['title'], 'wt2': c2, 'previous': cp, 'accepted': ca})
            continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']], 'title': a['title'],
                     'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids), 'window_tokens': [st, st + WPA * SEQ],
                     'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])], 'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca},
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(), 'windows': [WPA * len(segs) + i for i in range(WPA)]})
        if len(segs) == n:
            break
    if len(segs) < n:
        raise SystemExit('not enough articles')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    mr.write(out / 'segments.json', segs)
    mr.write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': mr.sha(out / 'plan.json'), 'pilot_sha256': mr.sha(out / 'pilot.json'),
                                        'n_articles': n, 'eligible': len(elig), 'excluded_previous': len(used_arts), 'skips': dict(skips),
                                        'near_duplicates': nd[:20], 'segments_sha256': mr.sha(out / 'segments.json'), 'tokens_sha256': mr.sha(out / 'tokens.npy')})
    log(f'data frozen: {n} articles; skips {dict(skips)}')


def run(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'results.json').exists():
        raise SystemExit('results exist')
    if mr.sha(out / 'tokens.npy') != mr.read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('tokens changed')
    toks = np.load(out / 'tokens.npy')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = mr.read(mr.CONFIG), dr.read(dr.RANGES)
    cc = dr.config_checks(fm, dq, ranges)
    f, base = mr.baseline(toks, dq, ranges, log, 'baseline')
    np.savez(out / 'baseline_routes.npz', S0=base['S0'], w0=base['w0'], p0=base['p0'], tokens=toks, moe_norm=f['extra']['moe_norm'])
    runs = {c: mr.forward5(toks, fm, dq, ranges, c, base=base, log=log, label=c) for c in ('L00', 'L11')}
    runs['Ldprop'] = forward_dprop(toks, fm, dq, ranges, base, log=log, label='Ldprop')
    (out / 'routes').mkdir(exist_ok=True)
    for c, r in runs.items():
        np.savez(out / f'routes/{c}.npz', ids=r['routes']['ids'], weights=r['routes']['weights'], **r['extra'])
    ctl = {'config_checks': cc, 'semantics': semantics(runs['Ldprop'], base, runs['L11'])}
    sub, sb = toks[:CTRL], {k: v[:CTRL] for k, v in base.items()}
    off = forward_dprop(sub, None, dq, ranges, sb, log=log, label='off Ldprop')
    ctl['quant_off'] = mr.eq(off, f, CTRL); ctl['quant_off_scale_all_one'] = bool((off['extra']['scale'] == 1.0).all())
    r2 = forward_dprop(sub, fm, dq, ranges, sb, log=log, label='rerun Ldprop')
    ctl['rerun'] = mr.eq(r2, runs['Ldprop'], CTRL)
    ctl['rerun']['routes_equal'] = bool(np.array_equal(r2['routes']['ids'], runs['Ldprop']['routes']['ids'][:CTRL])
                                        and np.array_equal(r2['routes']['weights'], runs['Ldprop']['routes']['weights'][:CTRL]))
    rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs['Ldprop']['routes']), log=log, label='replay Ldprop')
    ctl['replay'] = mr.eq(rp, runs['Ldprop'])
    ctl['finite'] = all(r['finite_ce'] and r['finite_weights'] for r in runs.values()) and f['finite_ce']
    ctl['audit'] = {c: r['audit'] for c, r in runs.items()}
    ctl['pass'] = bool(cc['pass'] and ctl['semantics']['pass'] and mr.bitwise(ctl['quant_off']) and ctl['quant_off_scale_all_one']
                       and mr.bitwise(ctl['rerun']) and ctl['rerun']['routes_equal'] and mr.bitwise(ctl['replay']) and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    mr.write(out / 'results.json', {'ce': {'fp32': f['ce'], **{c: r['ce'] for c, r in runs.items()}}, 'controls': ctl,
                                    'seconds': {'baseline': f['seconds'], **{c: r['seconds'] for c, r in runs.items()}, 'quant_off': off['seconds'],
                                                'rerun': r2['seconds'], 'replay': rp['seconds']}, 'environment': dr.env_info()})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    t0 = time.time()
    rs = mr.read(out / 'results.json')
    if not rs['controls']['pass']:
        raise SystemExit('controls failed')
    segs = mr.read(out / 'segments.json'); nA = len(segs)
    A = {c: np.array(v).reshape(nA, WPA).sum(1) / (WPA * TP) for c, v in rs['ce'].items()}
    E = {'Delta_dprop': A['Ldprop'] - A['L11'], 'E_total': A['L11'] - A['L00'], 'Residual_dprop': A['Ldprop'] - A['L00'],
         **{f'D_{c}': A[c] - A['fp32'] for c in ('L00', 'L11', 'Ldprop')}}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, nA, (BOOT, nA))

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci_bonf2': np.percentile(d, [1.25, 98.75]).tolist(),
                'article_sd': float(x.std(ddof=1)), 'articles_positive': int((x > 0).sum())}
    res = {'n_articles': nA, 'contrasts': {k: st(v) for k, v in E.items()}}
    c = res['contrasts']
    lo, hi = c['Delta_dprop']['ci95']
    rd = c['Residual_dprop']['ci_bonf2']
    res['primary'] = 'helps' if hi < 0 else 'hurts' if lo > 0 else 'null'
    res['secondary'] = {'E_total_positive': c['E_total']['ci_bonf2'][0] > 0, 'E_total_exceeds_eps': c['E_total']['ci_bonf2'][0] > EPS,
                        'Residual_dprop_within_eps': -EPS <= rd[0] and rd[1] <= EPS, 'Residual_dprop_positive': rd[0] > 0}
    if c['E_total']['mean'] != 0:
        res['descriptive_fraction_of_E_total_removed_point'] = float(-c['Delta_dprop']['mean'] / c['E_total']['mean'])
    tokens = nA * WPA * TP
    res['ce_ppl'] = {k: {'ce': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for k, v in rs['ce'].items()}
    b = np.load(out / 'baseline_routes.npz')
    S0, p0 = b['S0'][:, :, :TP].astype(np.int64), b['p0'][:, :, :TP]
    R = {c_: np.load(out / f'routes/{c_}.npz') for c_ in ('L00', 'L11', 'Ldprop')}
    rows = []
    for l in range(NL):
        row = {'layer': l}
        for c_ in ('L11', 'Ldprop'):
            ids = R[c_]['ids'][:, l, :TP].astype(np.int64); wv = R[c_]['weights'][:, l, :TP].astype(np.float64)
            p0S = np.take_along_axis(p0[:, l], ids, axis=-1).astype(np.float64)
            row[f'{c_}_set_change_vs_S0'] = float((np.sort(ids, -1) != np.sort(S0[:, l], -1)).any(-1).mean())
            row[f'{c_}_proportion_l1_vs_p0S'] = float(np.abs(wv / wv.sum(-1, keepdims=True) - p0S / p0S.sum(-1, keepdims=True)).sum(-1).mean())
            row[f'{c_}_sum_minus_p0S_sum'] = float((wv.sum(-1) - p0S.sum(-1)).mean())
            row[f'{c_}_moe_norm'] = float(R[c_]['moe_norm'][:, l].mean())
        sc = R['Ldprop']['scale'][:, l, :TP]
        row.update({'L00_moe_norm': float(R['L00']['moe_norm'][:, l].mean()), 'fp32_moe_norm': float(b['moe_norm'][:, l].mean()),
                    'Ldprop_scale_mean': float(sc.mean()), 'Ldprop_scale_p01': float(np.percentile(sc, 1)), 'Ldprop_scale_p99': float(np.percentile(sc, 99))})
        rows.append(row)
    with open(out / 'per_layer.csv', 'w') as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    sc = R['Ldprop']['scale'][:, :, :TP]
    res['scale'] = {'min': float(sc.min()), 'max': float(sc.max()), 'mean': float(sc.mean()), 'p001': float(np.percentile(sc, 0.1)), 'p999': float(np.percentile(sc, 99.9))}
    with open(out / 'per_article.csv', 'w') as fh:
        pa = [{'index': s['index'], 'title': s['title'], **{f'ce_{k}': float(v[i]) for k, v in A.items()},
               **{k: float(E[k][i]) for k in ('Delta_dprop', 'E_total', 'Residual_dprop')}} for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(pa[0])); w_.writeheader(); w_.writerows(pa)
    res['controls'] = rs['controls']
    res['timing'] = {'run': rs['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    mr.write(out / 'analysis.json', res)
    log(json.dumps({k: [c[k]['mean'], c[k]['ci95']] for k in ('Delta_dprop', 'E_total', 'Residual_dprop')}) + ' ' + res['primary'])


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
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': mr.sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
