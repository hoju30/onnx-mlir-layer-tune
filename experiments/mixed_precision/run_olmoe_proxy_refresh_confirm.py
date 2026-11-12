#!/usr/bin/env python3
"""OLMoE: confirmation of the one-refresh proxy result on 600 new articles. q0 / q1 are the frozen configurations of
runs/olmoe_proxy_refresh_20261022 (no proxy rebuild, no search, no calibration / range / quantizer / config / router
change). Only the sampling uncertainty of the evaluation text is examined. Pre-registered in plan.json / plan.md.

Stages (one --out-dir): plan, data, control, eval, analyze.
"""
import argparse
import csv
import glob
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
PR = HERE / 'runs/olmoe_proxy_refresh_20261022'
RANGES = HERE / 'runs/olmoe_multipoint_20261015/ranges.json'
SEQ, TP, WPA = 128, 127, 4
N, CHUNK, CTRL = 600, 128, 128
EPS, BOOT = 0.002, 10000
SEEDS = {'order': 22201, 'start': 22202, 'bootstrap': 22211}
PREV = {'mean': -0.0019560257048358565, 'ci95': [-0.00415487388606929, 0.00017213402775174654], 'article_sd': 0.012482524851683904, 'n': 128}


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


def manifests():
    """every historical segment manifest under runs/ (WT103 ones carry content hashes; WT2 ones are covered by
    excluding WikiText-2 entirely), plus the WT2 window-set files of the early OLMoE runs."""
    rows = []
    for f in sorted(glob.glob(str(HERE / 'runs/**/*segments*.json'), recursive=True)):
        p = Path(f)
        if 'olmoe_proxy_refresh_confirm' in f:
            continue
        s = read(p)
        src = sorted({x.get('source', x.get('split', '?')) for x in s})
        rows.append({'path': str(p.relative_to(HERE)), 'sha256': sha(p), 'articles': len(s), 'sources': src,
                     'wt103': any('wikitext-103' in x for x in src), 'content_hashes': 'content_sha256' in s[0]})
    windows = ['runs/olmoe_fixed_route_20261015/plan.json', 'runs/olmoe_multipoint_20261015/plan.json',
               'runs/solver_regression_20261016/plan.json', 'runs/olmoe_search_final_20261016/C/windows.json',
               'runs/olmoe_bc_numerics_20261016/B/windows.json']
    return rows, [{'path': w, 'sha256': sha(HERE / w), 'source': 'WikiText-2 windows (excluded with all of WikiText-2)'} for w in windows]


def used_wt103():
    rows, _ = manifests()
    used = set()
    for r in rows:
        if r['wt103']:
            if not r['content_hashes']:
                raise SystemExit(f'{r["path"]}: WT103 manifest without content hashes')
            used |= {s['content_sha256'] for s in read(HERE / r['path'])}
    return used


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    fz, pp = read(PR / 'freeze.json'), read(PR / 'plan.json')
    if sha(PR / 'configs/q0.json') != fz['q0_sha256'] or sha(PR / 'configs/q1.json') != fz['q1_sha256']:
        raise SystemExit('q0 / q1 differ from the freeze manifest')
    if sha(PR / 'freeze.json') != read(PR / 'data_freeze.json')['freeze_sha256']:
        raise SystemExit('freeze manifest changed')
    # original controls (the aggregated checks.json was written afterwards; check the raw control files too)
    raw = {'static_checks': read(PR / 'static_checks.json'), 'refresh_checks_pass': read(PR / 'refresh_checks.json')['pass'],
           'eval_controls_pass': read(PR / 'eval.json')['controls']['pass']}
    rows, wins = manifests()
    used = used_wt103()
    if not {s['content_sha256'] for s in read(PR / 'segments.json')} <= used:
        raise SystemExit('the previous 128 articles are not in the exclusion set')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    src = [PR / 'configs/q0.json', PR / 'configs/q1.json', PR / 'freeze.json', PR / 'plan.json', PR / 'data_freeze.json', PR / 'eval.json',
           PR / 'analysis.json', PR / 'tokens.npy', PR / 'static_checks.json', PR / 'refresh_checks.json', PR / 'checks.json', RANGES] + mr.WT103 + mr.WT2
    inputs = {str(p.relative_to(HERE)): sha(p) for p in src}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'Does NLL(q1) - NLL(q0) of runs/olmoe_proxy_refresh_20261022 reproduce on 600 independent new articles?',
        'frozen': {'q0': {'path': 'runs/olmoe_proxy_refresh_20261022/configs/q0.json', 'sha256': fz['q0_sha256']},
                   'q1': {'path': 'runs/olmoe_proxy_refresh_20261022/configs/q1.json', 'sha256': fz['q1_sha256']},
                   'ranges': {'path': str(RANGES.relative_to(HERE)), 'sha256': sha(RANGES), 'equals_previous': sha(RANGES) == fz['ranges_sha256']},
                   'quantizer': 'run_olmoe_search_final.dev_quant(corrected=True), as the previous evaluation',
                   'previous_raw_controls': raw},
        'not_done': 'no proxy rebuild, no search, no calibration / range / quantizer / config / router change, no fine-tuning; '
                    'only the sampling uncertainty of the evaluation text (not other calibrations, budgets or datasets)',
        'data': {'n_articles': N, 'windows': N * WPA, 'source': 'Salesforce/wikitext wikitext-103-raw-v1 train (sha256 checked)',
                 'rules': ['strict article boundaries (title line with an empty line before and after)',
                           'exclude every WT103 article whose normalized title or content hash matches WikiText-2 (all splits)',
                           'exclude within-WT103 duplicate title / content groups',
                           'exclude every WT103 article of an earlier OLMoE manifest (content hash), including the 128 articles '
                           'of runs/olmoe_proxy_refresh_20261022',
                           'near-duplicates: skip if >= 10% of the word 5-gram shingles occur in one WikiText-2 article, one earlier '
                           'OLMoE WT103 article or one already accepted article',
                           'tokenize each article alone (OLMoE tokenizer, no special tokens), skip < 512 tokens'],
                 'seeds': SEEDS, 'order': f'random order of the eligible articles (seed {SEEDS["order"]}), accept until {N}; uniform '
                          f'window start (seed {SEEDS["start"]}); no dependence on loss, routes, coverage or difficulty',
                 'eligible_before_length_and_neardup': len(elig), 'excluded_previous_wt103': len(used),
                 'manifests_scanned': rows, 'wt2_window_files': wins},
        'evaluation': {'conditions': 'FP32, q0, q1 with normal dynamic routing (own ids and weights; no FP32 route / proportion '
                                     'feedback); run_olmoe_dynamic_route.forward, GPU FP32, TF32 off, chunks of 128 windows, 127 '
                                     'predicted tokens per window',
                       'program_control': 'the previous run windows 0-127 (already seen; control only): q0 / q1 / FP32 CE must '
                                          'reproduce runs/olmoe_proxy_refresh_20261022/eval.json bitwise',
                       'checks': 'finite CE, unchanged config / input hashes, quantized routes differ from FP32 (no substitution), '
                                 'rerun of the first chunk of the new data bitwise (q0 and q1)'},
        'endpoint': {'per_article': 'NLL = summed CE of the 4 windows / (4 x 127)',
                     'primary': 'Delta = NLL(q1) - NLL(q0), mean over the 600 articles',
                     'bootstrap': f'{BOOT} paired article bootstraps (one index matrix), seed {SEEDS["bootstrap"]}, 95% percentile CI',
                     'A_direction': 'upper < 0: refresh better confirmed; lower > 0: refresh worse confirmed; otherwise direction not confirmed',
                     'B_equivalence': f'whole 95% CI inside [-{EPS}, +{EPS}]: equivalence confirmed; A and B are separate and can both hold; '
                                      'not significant is not equivalent',
                     'planning': f'previous article SD {PREV["article_sd"]:.5f} -> expected half-width about 0.001 (estimate, not a guarantee)',
                     'previous': 'the 128-article result is listed as history only; never pooled into the confirmation',
                     'fixed': 'n = 600 fixed; no extra samples, no subsets, no reselection'},
        'cost': 'research validation cost only (data, FP32 / q0 / q1 evaluation, controls, analysis, memory); the 104 s / 108 s '
                'offline numbers are single measurements of the previous run and are not re-measured',
        'inputs': inputs, 'code': {f: sha(HERE / f) for f in ['run_olmoe_proxy_refresh_confirm.py', 'run_olmoe_proxy_refresh.py',
                                                              'run_olmoe_route_mass_restore.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：一次 proxy refresh 收益的新文本確認', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          f'- 固定配置：q0 `{fz["q0_sha256"][:12]}…`、q1 `{fz["q1_sha256"][:12]}…`（runs/olmoe_proxy_refresh_20261022），ranges 與修正後量化器同前；不重建、不重搜、不改任何設定。',
          f'- 資料：{N} 篇新的 WikiText-103 train 文章 × 4 窗口（{N * WPA} 窗口）；排除 WikiText-2、WT103 內部重複、所有歷史 OLMoE WT103 文章（{len(used)} 篇，含上一輪 128 篇）、近重複（≥10% 5-gram）；seed 22201／22202；可用 {len(elig)} 篇。',
          '- 掃描的歷史 manifest：' + '、'.join(r['path'] for r in rows) + '；以及 WikiText-2 窗口檔 ' + '、'.join(w['path'] for w in wins) + '（整份 WikiText-2 排除即涵蓋）。',
          '- 評估：FP32、q0、q1 皆正常 dynamic；同一 forward、chunk 128、每窗口 127 個預測 token。',
          '- 程式控制：上一輪看過的窗口 0–127 須逐位元重現上一輪 CE（只作控制）。',
          '- 主要終點：Δ = NLL(q1) − NLL(q0)（文章配對）；10000 次 bootstrap（seed 22211）。',
          '- A：95% CI 上界 < 0 → 確認 refresh 較好；下界 > 0 → 確認較差；其他 → 方向未確認。',
          '- B：整段 95% CI 落在 ±0.002 → 確認等價。A、B 分開判定，可同時成立；不顯著不等於等價。',
          '- 樣本數固定 600（預期半寬約 0.001，僅估計）；不加樣、不挑子集、不重選配置；上一輪 128 篇只並列為歷史結果，不合併。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', inputs)
    write(out / 'freeze_manifest.json', {'time': p['time'], 'q0': p['frozen']['q0'], 'q1': p['frozen']['q1'], 'ranges': p['frozen']['ranges'],
                                         'source_run_freeze_sha256': sha(PR / 'freeze.json'), 'code': p['code']})
    log(f'plan written: eligible {len(elig)}, excluded previous WT103 {len(used)}')


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    for k, v in pl['inputs'].items():
        if sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    used = used_wt103()
    if len(used) != pl['data']['excluded_previous_wt103']:
        raise SystemExit('the set of previous WT103 articles changed since the plan')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    excl = Counter()
    base = []
    for a in w103:
        if a['nt'] in T2 or a['ch'] in C2:
            excl['wikitext2_title_or_content'] += 1
        elif ct[a['nt']] > 1 or cc[a['ch']] > 1:
            excl['wt103_internal_duplicate'] += 1
        else:
            base.append(a)
    prev = [a for a in base if a['ch'] in used]
    if len(prev) != len(used):
        raise SystemExit('previous WT103 articles not all found')
    excl['previous_olmoe_wt103'] = len(prev)
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['data']['eligible_before_length_and_neardup']:
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
            skips['near_duplicate'] += 1
            nd.append({'title': a['title'], 'wt2': c2, 'previous': cp, 'accepted': ca}); continue
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
    if len(segs) < N:
        raise SystemExit('not enough articles')
    if {s['content_sha256'] for s in segs} & used or len({s['content_sha256'] for s in segs}) != N:
        raise SystemExit('overlap or duplicate')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    dedup = {'exclusions': dict(excl), 'eligible': len(elig), 'visited': N + sum(skips.values()), 'skips': dict(skips), 'near_duplicates': nd,
             'overlap_with_previous': 0, 'unique_articles': N,
             'max_share': {k: max(s['max_shingle_share'][k] for s in segs) for k in ('wt2', 'previous', 'accepted')}}
    write(out / 'dedup_checks.json', dedup)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'), 'n': N,
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json'),
                                     'dedup_sha256': sha(out / 'dedup_checks.json')})
    merge_timing(out, 'data', {'seconds': time.perf_counter() - t0, 'peak_cpu_gb': peak_cpu_gb()})
    log(f'data frozen: {N} articles; skips {dict(skips)}; exclusions {dict(excl)}')


def control(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'control.json').exists():
        raise SystemExit('exists')
    torch.cuda.reset_peak_memory_stats()
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    toks = np.load(PR / 'tokens.npy')[:CTRL]
    old = read(PR / 'eval.json')['ce']
    res, sec = {}, {}
    for c in ('fp32', 'q0', 'q1'):
        fm = None if c == 'fp32' else read(PR / f'configs/{c}.json')
        r = dr.forward(toks, fm, dq, ranges, log=log, label=f'control {c}')
        res[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
        sec[c] = r['seconds']
    out_ = {'ce_max_abs_vs_previous': res, 'pass': all(v == 0 for v in res.values()), 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
    write(out / 'control.json', out_)
    merge_timing(out, 'program_control', {'seconds': sec, 'peak_gpu_gb': out_['peak_gpu_gb']})
    log(json.dumps(out_))


def eval_(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'eval.json').exists():
        raise SystemExit('exists')
    if not read(out / 'control.json')['pass']:
        raise SystemExit('program control failed: locate the cause first')
    fz = read(out / 'freeze_manifest.json')
    if sha(PR / 'configs/q0.json') != fz['q0']['sha256'] or sha(PR / 'configs/q1.json') != fz['q1']['sha256'] or \
            sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('frozen inputs changed')
    torch.cuda.reset_peak_memory_stats()
    toks = np.load(out / 'tokens.npy')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    runs = {}
    for c in ('fp32', 'q0', 'q1'):
        fm = None if c == 'fp32' else read(PR / f'configs/{c}.json')
        runs[c] = dr.forward(toks, fm, dq, ranges, record=True, log=log, label=f'eval {c}')
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'])
    ctl, rer = {}, {}
    for c in ('q0', 'q1'):
        r2 = dr.forward(toks[:CTRL], read(PR / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'rerun {c}')
        ctl[f'rerun_{c}'] = {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs[c]['ce'][:CTRL])))),
                             'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs[c]['hidden_sha'][:CTRL]))),
                             'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs[c]['routes']['ids'][:CTRL])
                                                  and np.array_equal(r2['routes']['weights'], runs[c]['routes']['weights'][:CTRL]))}
        rer[c] = r2['seconds']
    ctl['finite'] = all(r['finite_ce'] for r in runs.values())
    srt = lambda c: np.sort(runs[c]['routes']['ids'], -1)
    ctl['route_set_differs_from_fp32'] = {c: float((srt(c) != srt('fp32')).any(-1).mean()) for c in ('q0', 'q1')}
    ctl['audit'] = {c: runs[c]['audit'] for c in ('q0', 'q1')}
    ctl['pass'] = bool(all(v['ce_max_abs'] == 0 and v['hidden_bitwise_windows'] == CTRL and v['routes_equal'] for k, v in ctl.items() if k.startswith('rerun_'))
                       and ctl['finite'] and all(v > 0 for v in ctl['route_set_differs_from_fp32'].values())
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl, 'seconds': {c: r['seconds'] for c, r in runs.items()},
                              'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'environment': dr.env_info()})
    merge_timing(out, 'eval', {'fp32_s': runs['fp32']['seconds'], 'q0_s': runs['q0']['seconds'], 'q1_s': runs['q1']['seconds'],
                               'reruns_s': rer, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': peak_cpu_gb()})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    if not (ev['controls']['pass'] and read(out / 'control.json')['pass']):
        raise SystemExit('controls failed')
    A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
    d = A['q1'] - A['q0']
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))
    st = lambda x: {'mean': float(x.mean()), 'ci95': np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1))}
    res = {'n_articles': N, 'delta_q1_minus_q0': st(d) | {'q1_better_articles': int((d < 0).sum()), 'q1_worse_articles': int((d > 0).sum()),
                                                           'ties': int((d == 0).sum())},
           'damage': {c: st(A[c] - A['fp32']) for c in ('q0', 'q1')}}
    lo, hi = res['delta_q1_minus_q0']['ci95']
    res['A_direction'] = 'refresh_better_confirmed' if hi < 0 else 'refresh_worse_confirmed' if lo > 0 else 'direction_not_confirmed'
    res['B_equivalence'] = bool(-EPS <= lo and hi <= EPS)
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    res['previous_128'] = PREV | {'note': 'history only; not pooled'}
    after = {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    segs = read(out / 'segments.json')
    with open(out / 'per_article.csv', 'w') as fh:
        rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, 'delta_q1_minus_q0': float(d[i])} for i, s in enumerate(segs)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / 'per_window.csv', 'w') as fh:
        rows = [{'window': k, 'article': k // WPA, **{f'summed_ce_{c}': ev['ce'][c][k] for c in ev['ce']}} for k in range(N * WPA)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    chk = {'program_control': read(out / 'control.json'), 'evaluation': ev['controls'], 'dedup': read(out / 'dedup_checks.json'),
           'input_hashes_unchanged': res['input_hashes_unchanged'],
           'frozen_configs_unchanged': sha(PR / 'configs/q0.json') == read(out / 'freeze_manifest.json')['q0']['sha256']
           and sha(PR / 'configs/q1.json') == read(out / 'freeze_manifest.json')['q1']['sha256']}
    chk['all_pass'] = bool(chk['program_control']['pass'] and chk['evaluation']['pass'] and chk['input_hashes_unchanged'] and chk['frozen_configs_unchanged'])
    write(out / 'checks.json', chk)
    merge_timing(out, 'analysis', {'seconds': time.perf_counter() - t0})
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    log(json.dumps({'delta': res['delta_q1_minus_q0'], 'A': res['A_direction'], 'B': res['B_equivalence']}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'data', 'control', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'data': data, 'control': control, 'eval': eval_, 'analyze': analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
