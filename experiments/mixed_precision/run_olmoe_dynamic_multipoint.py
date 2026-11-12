#!/usr/bin/env python3
"""OLMoE: fixed-route vs dynamic-route damage at three pre-fixed points of the corrected-MILP BC frontier
(aggressive ~R 0.55, the knee, relaxed ~R 0.82), on new train articles. Pre-registered in plan.json / plan.md.

Reuses the verified inference function of run_olmoe_dynamic_route.py (one function; only the route source differs).

Stages (one --out-dir):
  plan          configs (selection rule on existing resources), hashes, data design, sample size, statistics, controls
  data          300 new train articles x 4 windows (strict article boundaries, whole-article exclusion)
  control_prev  program control: the previous round's knee results on its first 128 windows (old data, control only)
  capture       FP32 dynamic on the new windows (ids, weights, router top-16 logits)
  eval          FP32 fixed; per config q_fix, q_dyn, q_fix with q_dyn's own routes; reruns on the control subset
  analyze       endpoints, contrasts, routing statistics, per-article / per-layer tables, figures
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
SF = HERE / 'runs/olmoe_search_final_20261016'
MP = HERE / 'runs/olmoe_multipoint_20261015'
PREV = HERE / 'runs/olmoe_dynamic_route_20261018'
SEARCH = SF / 'B/search_milp.json'
TARGETS = {'aggressive': 0.55, 'relaxed': 0.82}
CONFIGS = ('aggressive', 'knee', 'relaxed')
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK = 128
CONTROL_WINDOWS = 128                           # control subset = the first chunk (windows 0-127)
EPS = 0.002
ALPHA = 0.05
BOOT = 10000
N_MAX = 300
HALF = 0.001
SEEDS = {'data': 18071, 'bootstrap': 18081}
PREV_DELTA = {'mean': 0.0013020140224187897, 'ci95': [-0.00015459480815372648, 0.002765552452548283]}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def frontier_choice():
    """Configs from the existing corrected-MILP frontier by actual resource R only (ties: name order)."""
    from mp_auto import config_hash
    s = read(SEARCH)
    names = s['names']
    cands = []
    for k, c in enumerate(s['selection']['candidates']):
        fm = {n: c['formats'][n] for n in names}
        cands.append({'name': f'cand{k:02d}_R{c["resource"]:.7f}', 'formats': fm, 'hash': config_hash(fm), 'R': c['resource'],
                      'size': c['size'], 'bops': c['bops'], 'objective': c['objective'], 'origin': c['origin'],
                      'solver_status': c['solver_status'], 'search_budget': c['search_budget']})
    knee_hash = config_hash(dr.load_config())
    knee = [c for c in cands if c['hash'] == knee_hash]
    pick = {'knee': knee[0]}
    for k, t in TARGETS.items():
        pick[k] = min(cands, key=lambda c: (abs(c['R'] - t), c['name']))
    return pick, cands, names


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    import olmoe_common as oc
    from scipy.stats import norm
    from run_olmoe_multipoint import problem, resources
    pick, cands, names = frontier_choice()
    if len({pick[c]['hash'] for c in CONFIGS}) != 3 or not (pick['aggressive']['R'] < pick['knee']['R'] < pick['relaxed']['R']):
        raise SystemExit('selection rule failed: need three distinct configs with R_aggr < R_knee < R_relaxed')
    (out / 'configs').mkdir(exist_ok=True)
    _, _, pnames, _, elems, rows, flops = problem(MP)
    assert pnames == names or sorted(pnames) == sorted(names)
    info = {}
    for c in CONFIGS:
        fm = pick[c]['formats']
        write(out / 'configs' / f'{c}.json', fm)
        r = resources(fm, pnames, elems, flops)
        info[c] = {'path': f'configs/{c}.json', 'sha256': sha(out / 'configs' / f'{c}.json'), 'config_hash': pick[c]['hash'],
                   'frontier_entry': pick[c]['name'], 'R_frontier': pick[c]['R'], 'size_rel_frontier': pick[c]['size'],
                   'bops_rel_frontier': pick[c]['bops'], 'R_recomputed': r['R'], 'M_rel': r['M_rel'], 'B_rel': r['B_rel'],
                   'M_bytes': r['M_bytes'], 'BOPs_calibration': r['BOPs'], 'bc_objective': pick[c]['objective'],
                   'solver_status': pick[c]['solver_status'], 'search_budget': pick[c]['search_budget'], 'origin': pick[c]['origin'],
                   'binding_component': 'BOPs' if r['B_rel'] >= r['M_rel'] else 'size',
                   'weight_formats': dict(Counter(v['w'] for v in fm.values())), 'act_formats': dict(Counter(v['a'] for v in fm.values()))}
        if abs(r['R'] - pick[c]['R']) > 1e-9 or abs(r['M_rel'] - pick[c]['size']) > 1e-9 or abs(r['B_rel'] - pick[c]['bops']) > 1e-9:
            raise SystemExit(f'{c}: recomputed resources differ from the frontier record')
    if info['knee']['sha256'] != sha(dr.CONFIG) and pick['knee']['hash'] != __import__('mp_auto').config_hash(dr.load_config()):
        raise SystemExit('knee differs from the previous round')
    diff = {f'{a}_vs_{b}': {'any': sum(pick[a]['formats'][n] != pick[b]['formats'][n] for n in names),
                            'weight': sum(pick[a]['formats'][n]['w'] != pick[b]['formats'][n]['w'] for n in names),
                            'act': sum(pick[a]['formats'][n]['a'] != pick[b]['formats'][n]['a'] for n in names)}
            for a, b in (('aggressive', 'knee'), ('knee', 'relaxed'), ('aggressive', 'relaxed'))}
    prev_art = np.array([float(r['Delta_route']) for r in csv.DictReader(open(PREV / 'per_article.csv'))])
    s_prev = float(prev_art.std(ddof=1))
    z_dec = float(norm.ppf(1 - ALPHA / 3))
    n_req = math.ceil((z_dec * s_prev / HALF) ** 2)
    # eligible train articles (counting only; no route / loss)
    tok = oc.tokenizer()
    used = dr.used_spans_all()
    for s_ in read(PREV / 'segments.json'):
        used[s_['split']].append(tuple(s_['article_chars'])); used[s_['split']].append(tuple(s_['window_chars']))
    arts, _, _, rej = dr.strict_articles('train', tok)
    elig = sum(1 for a in arts if a['tokens'][1] - a['tokens'][0] >= WPA * SEQ
               and not any(a['chars'][0] < u1 and a['chars'][1] > u0 for u0, u1 in used['train']))
    n = int(min(N_MAX, elig, n_req))
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [SEARCH, dr.CONFIG, dr.RANGES, SF / 'C/freeze.json', MP / 'table.json',
                                                         MP / 'table_meta.json', PREV / 'plan.json', PREV / 'analysis.json',
                                                         PREV / 'pilot.json', PREV / 'per_article.csv', PREV / 'segments.json',
                                                         PREV / 'eval.json', PREV / 'capture.json', PREV / 'routes/fp32_dyn.npz',
                                                         PREV / 'tokens.npy', dr.DATA / 'train-00000-of-00001.parquet']}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'questions': ['does the extra dynamic-route damage grow with compression', 'does the knee result replicate on new '
                      'articles', 'does any point clearly exceed the tolerance (trigger for a routing-mechanism round)'],
        'scope': 'validation only: no BC change, no first-order term, no re-search, no config repair, no knee re-selection',
        'selection_rule': 'existing corrected-MILP frontier of runs/olmoe_search_final_20261016/B/search_milp.json (the '
                          'frontier the default pipeline produced; the previous round used its knee). aggressive = '
                          'actual resource R closest to 0.55, relaxed = closest to 0.82 (R = max(weight_bytes/int8_bytes, '
                          'BOPs/int8_BOPs) of the configuration itself, not the solver budget; ties by entry name), '
                          'knee = the exact previous configuration. Three distinct configs with R_aggr < R_knee < R_relaxed '
                          'required, else amend before any evaluation.',
        'configs': info, 'config_differences_nodes': diff,
        'absent_nodes': '21 projections of experts without calibration tokens: bf16 weights + FP32 activations in every '
                        'config (historical semantics, unchanged)',
        'quantizer': 'run_olmoe_multipoint.DevQuant + corrected int activation (run_olmoe_search_final.dev_quant corrected), '
                     'ranges runs/olmoe_search_final_20261016/C/ranges.json; router FP32, not quantized',
        'inference': f'run_olmoe_dynamic_route.forward (verified last round): GPU FP32, TF32 off, matmul precision highest; '
                     f'chunks of {CHUNK} windows in window order, the tail chunk holds the remainder (same for every '
                     'condition); 127 predicted tokens per window, summed CE float64; no padding; GPU QDQ behaviour, not '
                     'native kernels',
        'conditions': {'fp32_dyn': 'baseline, own router; ids, weights, top-16 logits recorded',
                       'fp32_fix': 'ids + weights of fp32_dyn (control)',
                       'q_fix[c]': 'config c, ids + weights of fp32_dyn', 'q_dyn[c]': 'config c, FP32 router on its own hidden states'},
        'data': {
            'source': 'WikiText-2 raw train (Salesforce/wikitext, sha256 checked), strict article boundaries (title line '
                      'with an empty line before and after; 600 train articles)',
            'exclusion': 'whole strict articles overlapping any earlier OLMoE text (every set excluded last round plus '
                         'the 100 train and 28 validation/test articles of runs/olmoe_dynamic_route_20261018)',
            'eligible_train_articles_ge_512_tokens': elig,
            'sampling': f'{n} articles uniformly without replacement, uniform start, {WPA} consecutive 128-token windows; '
                        f'seed {SEEDS["data"]}; titles, offsets, token hashes frozen before evaluation; no resampling',
            'shared': 'all three configs and all conditions use the same articles and windows'},
        'sample_size': {
            'basis': 'article-level SD of Delta_route of the previous knee round (128 articles) = %.6f' % s_prev,
            'formula': 'n = ceil((z * s / h)^2), z = Phi^-1(1 - 0.05/3) = %.4f (the Bonferroni-adjusted one-sided level of '
                       'the decision interval), h = %.3f nat/token; n = min(%d, eligible, n_required)' % (z_dec, HALF, N_MAX),
            'n_required': n_req, 'n_articles': n, 'windows': n * WPA,
            'expected_half_width': {'decision_96.7': z_dec * s_prev / math.sqrt(n), 'ci95': 1.959964 * s_prev / math.sqrt(n)},
            'caveat': 'the knee SD is a planning value only; the aggressive point may vary more. Windows per article are '
                      'not extra independent articles.'},
        'compute_budget': {'windows': n * WPA, 'passes_full': 'fp32_dyn, fp32_fix, 3 x (q_fix, q_dyn, q_fix own routes)',
                           'passes_control_subset': 'fp32 dynamic path rerun and 3 x q_dyn rerun on windows 0-127',
                           'program_control': 'previous knee round, its windows 0-127: fp32_dyn, q_fix, q_dyn must reproduce it',
                           'estimate_gpu_minutes': 30, 'max_gpu_minutes': 90},
        'endpoints': {
            'per_article': 'CE = summed CE of the 4 windows / 508; D_fixed(c) = CE(q_fix,c) - CE(fp32_dyn); '
                           'D_dynamic(c) = CE(q_dyn,c) - CE(fp32_dyn); Delta_route(c) = CE(q_dyn,c) - CE(q_fix,c)',
            'aggregation': 'equal article weights (every article 508 tokens); PPL = exp(token-weighted mean CE)',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix for every config / condition / contrast, '
                         f'seed {SEEDS["bootstrap"]}; percentile intervals',
            'A_per_config': f'eps = {EPS} nat/token (previous round; search_final / solver regression). Decision interval = '
                            'two-sided 1 - 2(0.05/3) = 96.67% percentile CI (Bonferroni over the three configs for each '
                            'one-sided TOST / exceedance test): equivalent if inside [-eps, +eps]; exceeds tolerance if '
                            'lower > +eps; better beyond tolerance if upper < -eps; otherwise uncertain. Unadjusted 90% and '
                            '95% CIs reported as description. CI containing 0 is never equivalence.',
            'B_primary_contrast': 'T = Delta_route(aggressive) - Delta_route(relaxed), paired, 95% CI; supported T > 0 if '
                                  'lower > 0 (endpoints differ only; no monotonicity claim without the knee)',
            'B_secondary': 'aggressive - knee and knee - relaxed: Bonferroni over the two (97.5% CIs), unadjusted 95% shown',
            'knee_replication': 'descriptive: new knee Delta_route vs previous +0.00130 [-0.00015, +0.00277]; consistent if '
                                'the previous point estimate lies inside the new 95% CI; same decision category reported',
            'order': 'descriptive: order of D_fixed and of D_dynamic across the three configs with paired 95% CIs of '
                     'neighbouring differences',
            'ratios': 'D_dynamic / D_fixed only as description (small-denominator ratios are not decisions)'},
        'routing_statistics': 'per config vs fp32_dyn, every layer, positions 0-126: set change, overlap, routing-weight L1 '
                              '(aligned by expert id), dispatch TVD, set change by FP32 margin quintile (logit rank 8 - '
                              'rank 9, pooled edges), FP32 rank of entering experts, per-article distribution and Spearman '
                              'with Delta_route. Association only; no causal attribution to layers or change types.',
        'controls': ['fp32_fix == fp32_dyn (CE, final hidden state bitwise, all windows)',
                     'dynamic path without quantization (rerun) == fp32_dyn on the control subset (CE, hidden, routes)',
                     'q_dyn rerun on the control subset == q_dyn (bitwise, routes equal), each config',
                     'q_fix fed with q_dyn own routes == q_dyn (bitwise, all windows), each config',
                     'finite CE / routing weights, 127 tokens per window, route ids in [0, 64), all configs complete '
                     '(explicit format list, ranges for every quantized activation, posit LUTs), router object untouched, '
                     'hooks on expert projections only',
                     'program control: previous knee fp32_dyn / q_fix / q_dyn CE on its windows 0-127 reproduced bitwise',
                     'failure: fix and record before interpreting q_dyn; do not read kernel / batching / QDQ differences as routing'],
        'interpretation': {
            '1': 'all three equivalent within eps: fixed-route approximates dynamic in the tested range only',
            '2': 'a config exceeds +eps: next round decomposes expert-id vs routing-weight effects for that config first; '
                 'no criterion change now',
            '3': 'some or all uncertain: report the precision and the excluded effect range; no pass claim, no adding samples',
            '4': 'dynamic below fixed: kept as is; check controls first, then discuss fixed-route over-statement',
            '5': 'quality order of the three configs changes under dynamic: recorded as a phenomenon; this confirmation '
                 'set is not used to pick a deployment config'},
        'figures': ['R vs D_fixed / D_dynamic with 95% CIs', 'R vs Delta_route with decision CIs, zero line and +-eps; '
                    'three points only, no fitted curve'],
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_dynamic_multipoint.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    fmt = lambda c: (f'| {c} | `{info[c]["frontier_entry"]}` | {info[c]["R_frontier"]:.7f} | {info[c]["M_rel"]:.6f} | '
                     f'{info[c]["B_rel"]:.6f} | {info[c]["bc_objective"]:.4f} | `{info[c]["sha256"][:12]}…` |')
    md = ['# 預先登記：OLMoE 多資源點 fixed-route／dynamic-route 對照', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 配置（依既有實際資源選取）', '',
          '- 來源：search_final B 修正後 MILP frontier（`B/search_milp.json`，預設 pipeline 產生；上一輪用其 knee）。',
          '- aggressive：實際 R 最接近 0.55；relaxed：最接近 0.82；knee：上一輪同一配置。R = max(權重 bytes/int8, BOPs/int8)，用配置本身的資源，不用求解預算；同距離依名稱排序。',
          '', '| 配置 | frontier 項目 | R | size/int8 | BOPs/int8 | BC | sha256 |', '|---|---|---:|---:|---:|---:|---|',
          fmt('aggressive'), fmt('knee'), fmt('relaxed'), '',
          f'- 配置間不同節點數：{json.dumps(diff, ensure_ascii=False)}。', '- 21 個無 calibration token 的投影維持 bf16 權重 + FP32 activation。', '',
          '## 資料與樣本數', '',
          f'- WikiText-2 train，嚴格文章邊界，整篇排除先前所有 OLMoE 文本（含上一輪 128 篇）；可用 {elig} 篇（≥512 token）。',
          f'- n = ceil((z·s/h)²)，z = Φ⁻¹(1−0.05/3) = {z_dec:.4f}（Bonferroni 決策區間），s = 上一輪 knee 的文章級 Δ_route SD {s_prev:.5f}，h = {HALF}；需 {n_req} 篇，取上限 {n} 篇（{n * WPA} 窗口）。',
          f'- 預期決策區間半寬 {z_dec * s_prev / math.sqrt(n):.5f}、95% 半寬 {1.959964 * s_prev / math.sqrt(n):.5f}；aggressive 的變異可能更大。',
          f'- 均勻抽樣、每篇 4 個連續窗口，seed {SEEDS["data"]}；三個配置共用同一批窗口。', '',
          '## 推論與控制', '', f'- 沿用上一輪 `forward`；每批 {CHUNK} 窗口、尾批為剩餘窗口，所有條件一致；GPU FP32、TF32 關閉。',
          '- 控制：fp32_fix = fp32_dyn（全部）；動態路徑不量化重跑 = fp32_dyn（控制子集 0–127）；各配置 q_dyn 重跑（控制子集）；各配置 q_fix 改用 q_dyn 路由 = q_dyn（全部）；配置完整性與 router 未動；重現上一輪 knee 前 128 窗口（程式控制，舊資料不進確認集）。', '',
          '## 統計', '', f'- {BOOT} 次 paired article bootstrap（seed {SEEDS["bootstrap"]}），所有配置、條件、對比共用同一組重抽。',
          f'- A：每配置 Δ_route；ε = {EPS}；決策區間為 96.67%（Bonferroni ×3）：落在 [−ε, ε] → 等價；下界 > ε → 超過容忍；上界 < −ε → 明顯較好；否則不確定。另列未校正 90%／95%。',
          '- B：主要對比 T = Δ(aggressive) − Δ(relaxed)，95% CI；次要 aggressive−knee、knee−relaxed 用 97.5%（Bonferroni ×2）。',
          '- knee 重現（描述）：新 knee 的 95% CI 是否包含上一輪點估計 +0.00130。', '',
          '## 預先判讀', ''] + [f'{k}. {v}' for k, v in p['interpretation'].items()] + ['',
          '## 不做', '', '- 不改 BC、不加一階項、不重搜、不修配置、不依結果選 knee、不加樣、不做路由機制拆解。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    log(f'plan written: n={n}; configs ' + ', '.join(f'{c} R {info[c]["R_frontier"]:.7f}' for c in CONFIGS))


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('data exists')
    pl = read(out / 'plan.json')
    for k, v in pl['inputs'].items():
        if sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    n = pl['sample_size']['n_articles']
    used = dr.used_spans_all()
    for s_ in read(PREV / 'segments.json'):
        used[s_['split']].append(tuple(s_['article_chars'])); used[s_['split']].append(tuple(s_['window_chars']))
    tok = oc.tokenizer()
    arts, ids, off, rej = dr.strict_articles('train', tok)
    free = [a for a in arts if a['tokens'][1] - a['tokens'][0] >= WPA * SEQ
            and not any(a['chars'][0] < u1 and a['chars'][1] > u0 for u0, u1 in used['train'])]
    if len(free) != pl['data']['eligible_train_articles_ge_512_tokens']:
        raise SystemExit('eligible count changed since the plan')
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
        segs.append({'index': len(segs), 'split': 'train', 'title': a['title'], 'article_chars': a['chars'],
                     'article_tokens': a['tokens'], 'window_chars': wc, 'tokens': [st, st + WPA * SEQ],
                     'windows': [WPA * len(segs) + i for i in range(WPA)], 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
    if len({s['title'] for s in segs}) != len(segs):
        raise SystemExit('duplicate article')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'), 'n_articles': n,
                                     'eligible': len(free), 'rejected_title_lines_train': len(rej), 'segments_sha256': sha(out / 'segments.json'),
                                     'tokens_sha256': sha(out / 'tokens.npy')})
    log(f'data frozen: {n} articles, {n * WPA} windows')


def load_cfg(out, c):
    pl = read(out / 'plan.json')
    if sha(out / 'configs' / f'{c}.json') != pl['configs'][c]['sha256']:
        raise SystemExit(f'{c} config changed')
    return read(out / 'configs' / f'{c}.json')


def control_prev(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'control_prev.json').exists():
        raise SystemExit('exists')
    toks = np.load(PREV / 'tokens.npy')[:CONTROL_WINDOWS]
    ev, cap = read(PREV / 'eval.json'), read(PREV / 'capture.json')
    rec = dr.load_routes_npz(PREV / 'routes/fp32_dyn.npz')
    rec = {k: rec[k][:CONTROL_WINDOWS] for k in ('ids', 'weights')}
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = load_cfg(out, 'knee'), dr.read(dr.RANGES)
    f = dr.forward(toks, None, dq, ranges, record=True, log=log, label='prev fp32_dyn')
    qf = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(rec), log=log, label='prev q_fix')
    qd = dr.forward(toks, fm, dq, ranges, record=True, log=log, label='prev q_dyn')
    mx = lambda a, b: float(np.max(np.abs(np.array(a) - np.array(b[:CONTROL_WINDOWS]))))
    res = {'fp32_dyn_ce_max_abs': mx(f['ce'], cap['ce']), 'fp32_dyn_hidden_bitwise': int(sum(x == y for x, y in zip(f['hidden_sha'], cap['hidden_sha']))),
           'fp32_dyn_routes_equal': bool(np.array_equal(f['routes']['ids'], rec['ids']) and np.array_equal(f['routes']['weights'], rec['weights'])),
           'q_fix_ce_max_abs': mx(qf['ce'], ev['ce']['q_fix']), 'q_dyn_ce_max_abs': mx(qd['ce'], ev['ce']['q_dyn']),
           'seconds': {'fp32_dyn': f['seconds'], 'q_fix': qf['seconds'], 'q_dyn': qd['seconds']}, 'environment': dr.env_info()}
    res['pass'] = (res['fp32_dyn_ce_max_abs'] == 0 and res['fp32_dyn_hidden_bitwise'] == CONTROL_WINDOWS and res['fp32_dyn_routes_equal']
                   and res['q_fix_ce_max_abs'] == 0 and res['q_dyn_ce_max_abs'] == 0)
    write(out / 'control_prev.json', res)
    log(json.dumps({'pass': res['pass']}))


def check_tokens(out):
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('tokens changed')
    return np.load(out / 'tokens.npy')


def capture(out, log):
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'capture.json').exists():
        raise SystemExit('capture exists')
    toks = check_tokens(out)
    (out / 'routes').mkdir(exist_ok=True); (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    r = dr.forward(toks, None, dq, dr.read(dr.RANGES), record=True, router_top=16, log=log, label='fp32_dyn')
    dr.save_routes(out / 'routes/fp32_dyn.npz', r['routes'], r['top'], tokens=toks)
    write(out / 'capture.json', {'ce': r['ce'], 'hidden_sha': r['hidden_sha'], 'seconds': r['seconds'], 'audit': r['audit'],
                                 'route_hash': dr.route_hash(r['routes']), 'routes_sha256': sha(out / 'routes/fp32_dyn.npz'),
                                 'tokens_sha256': sha(out / 'tokens.npy'), 'model_revision': oc.REVISION, 'chunks': r['chunks'],
                                 'finite_ce': r['finite_ce'], 'environment': dr.env_info()})


def route_ok(rec):
    return bool(rec['ids'].min() >= 0 and rec['ids'].max() < NE and np.isfinite(rec['weights']).all()
                and all(len(set(row)) == K for row in rec['ids'].reshape(-1, K)[::997]))


def eval_(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    toks = check_tokens(out)
    cap = read(out / 'capture.json')
    if sha(out / 'routes/fp32_dyn.npz') != cap['routes_sha256']:
        raise SystemExit('routes changed')
    rec = dr.load_routes_npz(out / 'routes/fp32_dyn.npz')
    assert np.array_equal(rec['tokens'], toks)
    ranges = dr.read(dr.RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    rts = dr.routes_to_dev(rec)
    nW = len(toks)
    capd = {'ce': cap['ce'], 'hidden_sha': cap['hidden_sha']}
    sub = lambda r: {'ce': r['ce'][:CONTROL_WINDOWS], 'hidden_sha': r['hidden_sha'][:CONTROL_WINDOWS]}
    if not (out / 'eval_fp32.json').exists():
        ff = dr.forward(toks, None, dq, ranges, routes=rts, log=log, label='fp32_fix')
        fr = dr.forward(toks[:CONTROL_WINDOWS], None, dq, ranges, record=True, log=log, label='fp32_dyn rerun (control subset)')
        write(out / 'eval_fp32.json', {
            'ce': {'fp32_fix': ff['ce'], 'fp32_dyn': cap['ce']},
            'controls': {'fp32_fix_vs_fp32_dyn': dr.same(ff, capd), 'fp32_dyn_rerun_subset': dr.same(fr, sub(capd)),
                         'fp32_dyn_rerun_subset_routes_equal': bool(np.array_equal(fr['routes']['ids'], rec['ids'][:CONTROL_WINDOWS])
                                                                     and np.array_equal(fr['routes']['weights'], rec['weights'][:CONTROL_WINDOWS])),
                         'fp32_routes_valid': route_ok(rec), 'finite': ff['finite_ce'] and cap['finite_ce'] and fr['finite_ce']},
            'seconds': {'fp32_dyn_capture': cap['seconds'], 'fp32_fix': ff['seconds'], 'fp32_dyn_rerun_subset': fr['seconds']}})
    for c in CONFIGS:
        if (out / f'eval_{c}.json').exists():
            continue
        fm = load_cfg(out, c)
        cc = dr.config_checks(fm, dq, ranges)
        qf = dr.forward(toks, fm, dq, ranges, routes=rts, log=log, label=f'{c} q_fix')
        qd = dr.forward(toks, fm, dq, ranges, record=True, log=log, label=f'{c} q_dyn')
        dr.save_routes(out / f'routes/q_dyn_{c}.npz', qd['routes'], tokens=toks)
        qo = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(qd['routes']), log=log, label=f'{c} q_fix own routes')
        qr = dr.forward(toks[:CONTROL_WINDOWS], fm, dq, ranges, record=True, log=log, label=f'{c} q_dyn rerun (control subset)')
        ctl = {'q_fix_own_routes_vs_q_dyn': dr.same(qo, qd), 'q_dyn_rerun_subset': dr.same(qr, sub(qd)),
               'q_dyn_rerun_subset_routes_equal': bool(np.array_equal(qr['routes']['ids'], qd['routes']['ids'][:CONTROL_WINDOWS])
                                                       and np.array_equal(qr['routes']['weights'], qd['routes']['weights'][:CONTROL_WINDOWS])),
               'q_dyn_routes_valid': route_ok(qd['routes']),
               'finite': all(x['finite_ce'] for x in (qf, qd, qo, qr)),
               'audit': {'q_fix': qf['audit'], 'q_dyn': qd['audit']}}
        ok = (cc['pass'] and ctl['q_fix_own_routes_vs_q_dyn']['hidden_bitwise_windows'] == nW and ctl['q_fix_own_routes_vs_q_dyn']['ce_max_abs'] == 0
              and ctl['q_dyn_rerun_subset']['hidden_bitwise_windows'] == CONTROL_WINDOWS and ctl['q_dyn_rerun_subset_routes_equal']
              and ctl['q_dyn_routes_valid'] and ctl['finite']
              and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] and a['nonfinite_route_weights'] == 0 for a in ctl['audit'].values()))
        write(out / f'eval_{c}.json', {'config_checks': cc, 'ce': {'q_fix': qf['ce'], 'q_dyn': qd['ce']}, 'controls': ctl, 'pass': bool(ok),
                                       'q_dyn_routes_sha256': sha(out / f'routes/q_dyn_{c}.npz'),
                                       'seconds': {'q_fix': qf['seconds'], 'q_dyn': qd['seconds'], 'q_fix_own_routes': qo['seconds'],
                                                   'q_dyn_rerun_subset': qr['seconds']}, 'environment': dr.env_info()})
        log(f'{c}: pass={ok}')


# --------------------------------------------------------------------------- #
def routing_stats(f, q, nA):
    fi, fw = f['ids'][:, :, :TP].astype(np.int64), f['weights'][:, :, :TP]
    qi, qw = q['ids'][:, :, :TP].astype(np.int64), q['weights'][:, :, :TP]
    tl, ti = f['top_logits'][:, :, :TP], f['top_ids'][:, :, :TP].astype(np.int64)
    inter = (fi[..., :, None] == qi[..., None, :]).any(-1).sum(-1)
    seteq = inter == K
    margin = tl[..., K - 1] - tl[..., K]
    qe = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    mbin = np.digitize(margin, qe)
    layers, l1_all = [], np.zeros(fi.shape[:3])
    for l in range(NL):
        cf, cq = np.bincount(fi[:, l].ravel(), minlength=NE), np.bincount(qi[:, l].ravel(), minlength=NE)
        pf, pq_ = cf / cf.sum(), cq / cq.sum()
        both = (cf > 0) | (cq > 0)
        n_ = fi.shape[0] * TP
        dense_f, dense_q = np.zeros((n_, NE)), np.zeros((n_, NE))
        r_ = np.repeat(np.arange(n_), K)
        np.add.at(dense_f, (r_, fi[:, l].reshape(-1)), fw[:, l].reshape(-1))
        np.add.at(dense_q, (r_, qi[:, l].reshape(-1)), qw[:, l].reshape(-1))
        l1 = np.abs(dense_q - dense_f).sum(1)
        l1_all[:, l] = l1.reshape(fi.shape[0], TP)
        ch = ~seteq[:, l]
        layers.append({'layer': l, 'set_change_share': float(ch.mean()), 'overlap_mean': float(inter[:, l].mean() / K),
                       'dispatch_tvd': float(0.5 * np.abs(pq_ - pf).sum()), 'max_abs_dispatch_share_change': float(np.abs(pq_ - pf)[both].max()),
                       'experts_used_fp32': int((cf > 0).sum()), 'experts_used_q': int((cq > 0).sum()),
                       'weight_l1_mean': float(l1.mean()), 'weight_l1_unchanged_set_mean': float(l1[~ch.reshape(-1)].mean()),
                       'margin_median_changed': float(np.median(margin[:, l][ch])) if ch.any() else None,
                       'margin_median_unchanged': float(np.median(margin[:, l][~ch])),
                       'change_rate_bottom_margin_quintile': float(ch[mbin[:, l] == 0].mean()) if (mbin[:, l] == 0).any() else None,
                       'change_rate_top_margin_quintile': float(ch[mbin[:, l] == 4].mean()) if (mbin[:, l] == 4).any() else None})
    enter = ~(qi[..., :, None] == fi[..., None, :]).any(-1)
    rk = qi[..., :, None] == ti[..., None, :]
    rank = np.where(rk.any(-1), rk.argmax(-1) + 1, 99)
    er = rank[enter]
    per_art = {'set_change': (~seteq).reshape(nA, WPA, NL, TP).mean((1, 2, 3)), 'overlap': (inter / K).reshape(nA, WPA, NL, TP).mean((1, 2, 3)),
               'weight_l1': l1_all.reshape(nA, WPA, NL, TP).mean((1, 2, 3))}
    summary = {'set_change_share': float((~seteq).mean()), 'overlap': float(inter.mean() / K), 'weight_l1': float(l1_all.mean()),
               'dispatch_tvd_mean_layers_1_15': float(np.mean([x['dispatch_tvd'] for x in layers[1:]])),
               'change_rate_by_margin_quintile': [float((~seteq)[mbin == b].mean()) for b in range(5)], 'margin_quintile_edges': qe.tolist(),
               'entering_expert_fp32_rank': {'swaps': int(er.size), 'rank9': float((er == 9).mean()) if er.size else None,
                                             'rank10_12': float(((er >= 10) & (er <= 12)).mean()) if er.size else None,
                                             'rank13_16': float(((er >= 13) & (er <= 16)).mean()) if er.size else None,
                                             'beyond16': float((er == 99).mean()) if er.size else None}}
    return layers, per_art, summary


def figures(out, res, pl):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    c_fix, c_dyn = '#2a78d6', '#eb6834'
    R = {c: pl['configs'][c]['R_frontier'] for c in CONFIGS}
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': ink2, 'axes.labelcolor': ink, 'xtick.color': ink2, 'ytick.color': ink2,
                         'figure.facecolor': surf, 'axes.facecolor': surf})
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for key, col, dx, lab in (('D_fixed', c_fix, -0.004, 'fixed route'), ('D_dynamic', c_dyn, 0.004, 'dynamic route')):
        for c in CONFIGS:
            e = res['per_config'][c][key]
            ax.errorbar(R[c] + dx, e['mean'], yerr=[[e['mean'] - e['ci95'][0]], [e['ci95'][1] - e['mean']]], fmt='o', ms=6, color=col,
                        elinewidth=2, capsize=0, mec=surf, mew=1.5, label=lab if c == 'aggressive' else None)
    for c in CONFIGS:
        ax.annotate(c, (R[c], res['per_config'][c]['D_dynamic']['ci95'][1]), xytext=(0, 6), textcoords='offset points', ha='center', color=ink2, fontsize=9)
    ax.set_xlabel('resource R = max(size, BOPs) / int8')
    ax.set_ylabel('damage vs FP32 (nat/token)')
    top = max(res['per_config'][c]['D_dynamic']['ci95'][1] for c in CONFIGS)
    ax.set_ylim(top=top * 1.12)
    ax.set_title('Damage by resource point (95% CI; three configs, not a curve)', color=ink, fontsize=10, loc='left', pad=10)
    ax.grid(axis='y', color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout(); fig.savefig(out / 'fig_damage_vs_R.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.axhline(0, color=ink2, lw=1)
    for y in (EPS, -EPS):
        ax.axhline(y, color=ink2, lw=1, ls=(0, (4, 3)))
    xr = max(R.values()) + 0.085
    ax.text(xr, EPS + 0.00015, '+ε = 0.002', ha='right', va='bottom', color=ink2, fontsize=9)
    ax.text(xr, -EPS + 0.00015, '−ε', ha='right', va='bottom', color=ink2, fontsize=9)
    for c in CONFIGS:
        e = res['per_config'][c]['Delta_route']
        lo, hi = e['ci_decision']
        ax.errorbar(R[c], e['mean'], yerr=[[e['mean'] - lo], [hi - e['mean']]], fmt='o', ms=6, color=c_dyn, elinewidth=2, capsize=0, mec=surf, mew=1.5)
        ax.annotate(f'{c}\n{e["mean"]:+.4f}', (R[c], hi), xytext=(0, 6), textcoords='offset points', ha='center', color=ink2, fontsize=9)
    ax.set_xlabel('resource R = max(size, BOPs) / int8')
    ax.set_ylabel('Δ_route = CE(dynamic) − CE(fixed) (nat/token)')
    top = max(res['per_config'][c]['Delta_route']['ci_decision'][1] for c in CONFIGS)
    ax.set_ylim(-EPS - 0.0008, top + 0.0022)
    ax.set_title('Extra dynamic-route damage (96.7% Bonferroni CI)', color=ink, fontsize=10, loc='left', pad=10)
    ax.grid(axis='y', color=grid, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(min(R.values()) - 0.05, max(R.values()) + 0.09)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'fig_delta_route_vs_R.png'); plt.close(fig)


def analyze(out, log):
    from scipy.stats import spearmanr
    t0 = time.time()
    pl = read(out / 'plan.json')
    e0 = read(out / 'eval_fp32.json')
    ev = {c: read(out / f'eval_{c}.json') for c in CONFIGS}
    cp = read(out / 'control_prev.json')
    c0 = e0['controls']
    fp_ok = (c0['fp32_fix_vs_fp32_dyn']['ce_max_abs'] == 0 and c0['fp32_fix_vs_fp32_dyn']['hidden_bitwise_windows'] == c0['fp32_fix_vs_fp32_dyn']['windows']
             and c0['fp32_dyn_rerun_subset']['hidden_bitwise_windows'] == CONTROL_WINDOWS and c0['fp32_dyn_rerun_subset_routes_equal']
             and c0['fp32_routes_valid'] and c0['finite'])
    if not (fp_ok and cp['pass'] and all(ev[c]['pass'] for c in CONFIGS)):
        raise SystemExit('numerical controls failed: do not interpret dynamic results')
    segs = read(out / 'segments.json'); nA = len(segs)
    art = lambda x: np.array(x).reshape(nA, WPA).sum(1) / (WPA * TP)
    base = art(e0['ce']['fp32_dyn'])
    A = {c: {'q_fix': art(ev[c]['ce']['q_fix']), 'q_dyn': art(ev[c]['ce']['q_dyn'])} for c in CONFIGS}
    Q = {c: {'D_fixed': A[c]['q_fix'] - base, 'D_dynamic': A[c]['q_dyn'] - base, 'Delta_route': A[c]['q_dyn'] - A[c]['q_fix']} for c in CONFIGS}
    rng = np.random.default_rng(SEEDS['bootstrap'])
    idx = rng.integers(0, nA, (BOOT, nA))
    lvl = 100 * ALPHA / 3

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci90': np.percentile(d, [5, 95]).tolist(),
                'ci_decision': np.percentile(d, [lvl, 100 - lvl]).tolist(), 'ci97_5': np.percentile(d, [1.25, 98.75]).tolist()}
    res = {'n_articles': nA, 'windows': nA * WPA, 'per_config': {}, 'decision_level': f'{100 - 2 * lvl:.2f}% two-sided (Bonferroni x3)'}
    tokens = nA * WPA * TP
    res['fp32'] = {'ce': float(np.sum(e0['ce']['fp32_dyn']) / tokens), 'ppl': float(np.exp(np.sum(e0['ce']['fp32_dyn']) / tokens))}
    for c in CONFIGS:
        r = {k: st(v) for k, v in Q[c].items()}
        lo, hi = r['Delta_route']['ci_decision']
        r['decision'] = ('equivalent_within_tolerance' if -EPS <= lo and hi <= EPS else 'exceeds_tolerance' if lo > EPS
                         else 'dynamic_better_beyond_tolerance' if hi < -EPS else 'uncertain')
        r['ce'] = {k: float(np.sum(ev[c]['ce'][k]) / tokens) for k in ('q_fix', 'q_dyn')}
        r['ppl'] = {k: float(np.exp(v)) for k, v in r['ce'].items()}
        r['ppl_rel_fp32'] = {k: v / res['fp32']['ppl'] - 1 for k, v in r['ppl'].items()}
        r['ratio_D_dynamic_over_D_fixed_point'] = float(Q[c]['D_dynamic'].mean() / Q[c]['D_fixed'].mean())
        r['articles_delta_positive'] = int((Q[c]['Delta_route'] > 0).sum())
        r['article_delta_sd'] = float(Q[c]['Delta_route'].std(ddof=1))
        r['R'] = pl['configs'][c]['R_frontier']
        res['per_config'][c] = r
    T = Q['aggressive']['Delta_route'] - Q['relaxed']['Delta_route']
    res['contrast_T_aggr_minus_relaxed'] = st(T)
    res['contrast_T_aggr_minus_relaxed']['supported_positive'] = res['contrast_T_aggr_minus_relaxed']['ci95'][0] > 0
    res['contrast_T_aggr_minus_relaxed']['supported_negative'] = res['contrast_T_aggr_minus_relaxed']['ci95'][1] < 0
    res['secondary'] = {}
    for a, b in (('aggressive', 'knee'), ('knee', 'relaxed')):
        x = st(Q[a]['Delta_route'] - Q[b]['Delta_route'])
        x['bonferroni_supported'] = 'positive' if x['ci97_5'][0] > 0 else 'negative' if x['ci97_5'][1] < 0 else 'none'
        res['secondary'][f'{a}_minus_{b}'] = x
    kn = res['per_config']['knee']['Delta_route']
    res['knee_replication'] = {'previous': PREV_DELTA, 'new': kn, 'previous_point_in_new_ci95': kn['ci95'][0] <= PREV_DELTA['mean'] <= kn['ci95'][1],
                               'previous_decision': 'uncertain', 'new_decision': res['per_config']['knee']['decision']}
    res['order'] = {k: {f'{a}_minus_{b}': st(Q[a][k] - Q[b][k]) for a, b in (('aggressive', 'knee'), ('knee', 'relaxed'))} for k in ('D_fixed', 'D_dynamic')}
    # routing
    f = dr.load_routes_npz(out / 'routes/fp32_dyn.npz')
    res['routing'] = {}
    per_art_rows = [{'index': s['index'], 'title': s['title'], 'ce_fp32': float(base[i])} for i, s in enumerate(segs)]
    layer_rows = []
    for c in CONFIGS:
        q = dr.load_routes_npz(out / f'routes/q_dyn_{c}.npz')
        layers, pa, summ = routing_stats(f, q, nA)
        summ['article_spearman_with_delta'] = {k: float(spearmanr(v, Q[c]['Delta_route']).statistic) for k, v in pa.items()}
        summ['article_set_change_range'] = [float(pa['set_change'].min()), float(pa['set_change'].max())]
        res['routing'][c] = {'summary': summ, 'layers': layers}
        for L_ in layers:
            layer_rows.append({'config': c, **L_})
        for i, row in enumerate(per_art_rows):
            row.update({f'{c}_D_fixed': float(Q[c]['D_fixed'][i]), f'{c}_D_dynamic': float(Q[c]['D_dynamic'][i]),
                        f'{c}_Delta_route': float(Q[c]['Delta_route'][i]), f'{c}_route_set_change': float(pa['set_change'][i]),
                        f'{c}_route_weight_l1': float(pa['weight_l1'][i])})
        del q
    with open(out / 'per_article.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_art_rows[0])); w.writeheader(); w.writerows(per_art_rows)
    with open(out / 'per_layer_routing.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(layer_rows[0])); w.writeheader(); w.writerows(layer_rows)
    res['controls'] = {'fp32': c0, 'configs': {c: ev[c]['controls'] for c in CONFIGS}, 'config_checks': {c: ev[c]['config_checks'] for c in CONFIGS},
                       'program_control_previous_round': cp}
    res['timing'] = {'fp32': e0['seconds'], 'configs': {c: ev[c]['seconds'] for c in CONFIGS}, 'program_control': cp['seconds'],
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    figures(out, res, pl)
    res['timing']['analysis_seconds'] = time.time() - t0
    write(out / 'analysis.json', res)
    log(json.dumps({c: [res['per_config'][c]['Delta_route']['mean'], res['per_config'][c]['decision']] for c in CONFIGS}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'data', 'control_prev', 'capture', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'data': data, 'control_prev': control_prev, 'capture': capture, 'eval': eval_, 'analyze': analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
