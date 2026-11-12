#!/usr/bin/env python3
"""OLMoE fixed route: is the calibration-only first-order term (and BC) stable across calibration texts, how much of
the transfer failure is approximation error vs text dependence, and does expert coverage relate to instability?
Pre-registered in plan.json / plan.md (written before any data, score or loss).

Scope: stage 1's 9 experts / 27 projections / 360 weight-only candidates / 909 same-resource pairs / 63 groups,
activation FP32, the shared (corrected) weight quantizer, single-point semantics. Each text uses its own FP32 route.
Nothing here changes the criterion or the search pipeline.

Stages (one --out-dir):
  plan      plan.json + plan.md (methods, seeds, statistics, input hashes); no data drawn
  data      J (8 validation + 8 test articles x 4 windows) and 8 calibration batches C1..C8 (5 train articles x 2
            windows each); tokens, segments.json, data_freeze.json
  capture   --set C1..C8 | J: FP32 routes per window (J refused before freeze)
  score     C0 (original calibration, implementation check) and C1..C8: B and F per projection/format, per-window F,
            routed-token coverage, quantization-error hashes
  caltruth  C1..C8: measured loss of all 360 candidates per window (diagnostic truth)
  freeze    every method's choices in every batch (+ calibration-measured selector, historical references) and hashes
  jtruth    J: 360 candidates x 64 windows (refused before freeze)
  analyze   pre-specified analyses -> analysis.json, tables.md, csv files
"""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / 'runs/olmoe_fixed_route_20261015'
CT = HERE / 'runs/olmoe_crosstext_20261017'
FO = HERE / 'runs/olmoe_first_order_validation_20261017'
DATA = HERE / 'gpt2_assets/data'
FORMATS = ('int4', 'int6', 'int8', 'p6e1', 'p8e1')
METHODS = ('bc', 'first', 'first_plus_half_bc', 'first_plus_bc')
PRIMARY = 'first_plus_half_bc'
SEQ, TP = 128, 127
NB, APB, WPA = 8, 5, 2                      # batches, articles per batch, windows per article
WPB = APB * WPA                             # 10 windows per batch
DENOM = WPB * TP                            # 1270 predicted tokens per batch (all batches equal)
BATCHES = tuple(f'C{k}' for k in range(1, NB + 1))
SEEDS = {'J': 18011, 'calibration': 18021, 'bootstrap': 18031, 'permutation': 18041}
BOOT = 10000


def read(p):
    return json.loads(Path(p).read_text())


def clean(o):
    """NaN / inf -> None, numpy scalars -> Python (JSON without NaN)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write(p, obj):
    Path(p).write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def candidates():
    return [c for c in read(S1 / 'candidates.json') if c['mode'] == 'w']


def groups_of(cs):
    by = defaultdict(list)
    for i, c in enumerate(cs):
        by[(c['node'], c['M_bytes'], c['BOPs_calibration'])].append(i)
    return [v for v in by.values() if len(v) > 1]


def choose(groups, v, nms):
    return np.array([min(g, key=lambda i: (v[i], nms[i])) for g in groups])


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise RuntimeError('plan exists')
    ins = [S1 / 'candidates.json', S1 / 'table.json', S1 / 'tokens_calibration.npy', S1 / 'routes_manifest.json',
           S1 / 'truth_calibration.json', CT / 'segments.json', CT / 'D/truth.json', CT / 'V/truth.json',
           FO / 'segments.json', FO / 'projection_scores.json', FO / 'scores.json', FO / 'choices.json', FO / 'H/truth.json',
           DATA / 'validation-00000-of-00001.parquet', DATA / 'test-00000-of-00001.parquet', DATA / 'train-00000-of-00001.parquet']
    code = ['run_olmoe_calibration_stability.py', 'olmoe_common.py', 'run_olmoe_crosstext.py', 'run_olmoe_fixed_route.py',
            'run_olmoe_bc_numerics.py', 'run_olmoe_first_order_validation.py', 'qwen3_swiglu_common.py']
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'questions': {
            'A': 'first-order approximation itself inaccurate on new calibration text',
            'B': 'approximation accurate on each batch, but the measured quantization preference differs between texts',
            'C': 'few routed tokens for an expert make scores / choices unstable (hypothesis, not assumed)'},
        'scope': '9 experts, 27 projections, 360 weight-only candidates, 909 same-resource pairs, 63 groups (stage 1). '
                 'Activation FP32; stage-1 candidates, resource model, weight quantizer (corrected shared quantizers; '
                 'weight error depends on W only). Only the text used to estimate B/F changes. Each text has its own '
                 'FP32 route, fixed when evaluating its candidates; no route is transferred between texts. No criterion '
                 'or pipeline change, no dynamic route, no full configuration.',
        'data': {
            'shortage_amendment': 'Before this plan (counting only, no route/score/loss): after excluding every earlier '
                'OLMoE window set (stage 1 calibration/diagnostic/check, multipoint, solver regression, search_final C, '
                'numerics B) and the whole articles of cross-text D/V and first-order H, WikiText-2 validation has 18 and '
                'test 27 unused articles >=256 tokens (45). J (16) + 8x5 calibration articles (40) = 56 does not fit. '
                'Amended design (fixed here, before any data is drawn): J from validation/test as before; the 8 '
                'calibration batches from the WikiText-2 raw train split (Salesforce/wikitext wikitext-2-raw-v1 '
                'train-00000-of-00001.parquet, sha256 checked against the hub LFS oid; never used by any OLMoE '
                'experiment). The original calibration (validation, first 1280 tokens) differs in split and '
                'composition: old-vs-new differences are not attributed to one factor.',
            'J': '16 articles (8 validation + 8 test), whole-article exclusion of every earlier OLMoE set incl. D/V/H, '
                 '>=512 tokens, uniform article draw without replacement, uniform start, 4 consecutive 128-token windows; '
                 f'seed {SEEDS["J"]}; same procedure as runs/olmoe_first_order_validation_20261017 fresh stage.',
            'calibration': f'{NB} batches x {APB} train articles x {WPA} consecutive windows (10 windows, {DENOM} '
                           'predicted tokens per batch). Eligible: train articles >=256 tokens whose title is unique in '
                           'train and absent from validation/test. 40 articles drawn uniformly without replacement; the '
                           'draw order assigns batches (C1 = draws 1-5, ...); uniform start per article; '
                           f'seed {SEEDS["calibration"]}. Nothing depends on routes, scores or loss.',
            'records': 'title, split, article char/token span, window char/token span, token hashes (segments.json, '
                       'data_freeze.json). No resampling; any shortage is recorded and amends this plan.',
            'C0': 'original 10 calibration windows / routes of stage 1: historical reference and implementation check only.',
            'retrospective': 'D, V (cross-text) and H (first-order validation) are already seen: retrospective only.'},
        'scores': {
            'B': 'batch BC / 1270: sum_c (G_c / n_in) sum_in dW^2, G_c = sum over windows of the per-window squared '
                 'weight gradient row sums (summed CE), re-estimated on every batch',
            'F': 'sum_s <g_s, Q(W)-W> / 1270, g_s = gradient of window s summed CE w.r.t. W (FP32 forward/backward, '
                 'FP64 contraction and accumulation)',
            'joint': 'joint Gate+Up candidates sum projection B and F',
            'methods': 'B; F; F+0.5B (primary); F+B (sensitivity). No clipping, no fitted coefficient, no posthoc '
                       'method choice. B is a squared-gradient proxy, not a Hessian; F+0.5B is not a calibrated '
                       'second-order Taylor estimate.',
            'references': 'frozen original BC choice (stage-1 table) and the previous round frozen choices '
                          '(runs/olmoe_first_order_validation_20261017/choices.json) as historical references; '
                          'calibration-measured loss of each batch as a diagnostic selector (not a method).',
            'no_observation': 'an expert with zero routed tokens in a batch is "no observation" there: all its scores '
                              'are 0 and choices fall to the name tie-break; such cells are excluded from fidelity, '
                              'stability and coverage outcomes and are counted. In the primary endpoint they stay in '
                              '(every method makes the identical tie-break choice, difference exactly 0); a secondary '
                              'version averages observed cells only.'},
        'checks': [
            'C0 rescored with the new code reproduces the previous projection F and B (relative <=1e-9) and the '
            'previous frozen choices of all four methods exactly',
            'every window has 127 predicted tokens; batch denominators 1270; F equals the mean of per-window terms',
            'joint candidate scores equal the sum of projection scores',
            'quantization error dW (sha256 per projection/format) identical across C0..C8',
            'fixed-route FP32 baseline (score forward and truth replay) equals the captured dynamic summed CE',
            'all choices within the same (expert, M_bytes, BOPs) group; candidate set equals stage 1 weight-only',
            'any failure: fix and record before interpretation'],
        'order': 'plan -> data -> capture C1..C8 -> score (C0 + C1..C8) -> caltruth C1..C8 -> freeze (all choices) -> '
                 'capture J -> J truth -> analyze. J routes and truth are refused before freeze.',
        'analyses': {
            '1_fidelity': 'per batch (and C0): sign agreement of B, F, F+0.5B, F+B with the same batch measured pair '
                          'differences over observed pairs; mean within-expert Spearman of scores vs measured candidate '
                          'loss; in-batch regret = measured loss of the choice minus the batch-measured best of the group.',
            '2_stability': 'per pair: counts over the 8 batches of +/-/tie/no-observation for every method score and the '
                           'measured difference; mean majority share; unanimous fraction; mean pairwise-batch sign '
                           'agreement. Per group: distinct choices, modal share, unanimity over observed batches; for '
                           'every method incl. BC and the measured selector.',
            '3_primary': 'Delta = J loss(F+0.5B choice) - J loss(same-batch BC choice), J loss = per-token NLL change '
                         'vs FP32 per article (4 windows). Equal-weight mean over 63 groups x 8 batches, then article '
                         f'as unit. CI95: {BOOT} two-way bootstrap (batches resampled with replacement; J articles '
                         'resampled within validation/test strata, 8 each), same draws for every comparison, seed '
                         f'{SEEDS["bootstrap"]}. Also per batch (article bootstrap), per expert (descriptive), the '
                         'article-only CI, F and F+B, and the historical references. Only 8 batches: the batch '
                         'component of the CI is coarse. Pairs are never treated as independent samples.',
            '4_transfer': 'calibration-measured selector of each batch evaluated on J. Decomposition '
                          'Delta(primary-BC) = Delta(primary-measured selector) + Delta(measured selector-BC). Pair '
                          'level, per method and batch: correct vs J sign; approximation error (score disagrees with '
                          'the batch truth and with J); transfer error (score agrees with batch truth, batch truth '
                          'disagrees with J); also |J difference|-weighted and on J-resolved pairs (unadjusted). '
                          'Descriptive variance components of per-window measured differences: between-article vs '
                          'within-article (calibration 40 articles x 2 windows; J 16 x 4).',
            '5_coverage': 'per (expert, batch): routed tokens, windows and articles with tokens, per-article tokens. '
                          'Outcomes (observed cells): (o1) share of the expert pairs whose F sign differs from the '
                          'leave-one-batch-out pooled F sign; (o2) share of its 7 groups whose primary choice differs '
                          'from the leave-one-batch-out pooled primary choice; (o3) mean J regret of the primary choice '
                          '(J loss minus the J-best candidate of the group, all 16 articles). Same o1/o2 for BC. '
                          'Association: within-expert Spearman with log(1+tokens) averaged over experts; permutation '
                          f'p (tokens permuted within expert, {BOOT}, seed {SEEDS["permutation"]}); Holm over o1-o3 for '
                          'the primary method. No threshold is chosen. Association only, not causation; stage-1 '
                          'high/mid/low tiers descriptive only (9 experts, identity confounded with coverage).',
            '6_cases': 'L15.E17 (up int8 vs p8e1 and its groups), L15.E59 (coverage vs choice / J loss), L8.E33 '
                       '(three historical pairs: score vs batch truth vs J). Cases do not replace the overall result.',
            'interpretation_guides': 'Descriptive, not gates. A is supported if the primary in-batch fidelity is '
                'clearly below the C0 value (e.g. median over batches < 0.90) or approximation errors outnumber '
                'transfer errors. B is supported if in-batch fidelity stays high while measured preferences disagree '
                'between batches / with J and transfer errors dominate. C is supported only as an association if the '
                'within-expert coverage associations point the expected way with Holm-adjusted p < 0.05.'},
        'not_done': 'no coefficient tuning on J, no coverage threshold / shrinkage / fallback, no expert exclusion, no '
                    'multi-batch averaged score presented as a method, no default pipeline change, no full configuration.',
        'execution': 'GPU FP32, TF32 off; OMP/MKL dynamic off; corrected shared quantizers; replay = '
                     'run_olmoe_crosstext.replay (stage-1 single-point semantics).',
        'inputs': {str(f.relative_to(HERE)): sha(f) for f in ins},
        'code': {f: sha(HERE / f) for f in code}}
    write(out / 'plan.json', p)
    md = ['# 預先登記：跨 calibration 文本的一階項穩定性與 expert 覆蓋診斷', '',
          f'時間：{p["time"]}。本檔為 plan.json 的摘要；以 plan.json 為準。', '',
          '## 問題', '', '- A：一階近似在新 calibration 上本身不準。', '- B：一階近似準確，但不同文本的實測量化偏好不同。',
          '- C：expert 觀測 token 少使分數與選擇不穩定（待驗證假說，不預設成立）。', '',
          '## 資料（含資料不足修訂）', '',
          '- 排除先前所有 OLMoE 資料（含 D、V、H 整篇文章）後，validation 剩 18 篇、test 剩 27 篇（≥256 token），共 45 篇；'
          'J 16 篇 + calibration 40 篇 = 56 篇不足。此計數在本 plan 前完成，未使用 route、分數或 loss。',
          '- 修訂：8 批 calibration 改由 WikiText-2 raw train（同一來源，hub LFS sha256 核對）抽取；J 仍由 validation/test 各 8 篇抽取。',
          '- 原 calibration（validation 開頭 1280 token）來源與組成不同，新舊差異不單獨歸因於「換文本」或「多樣性」。',
          f'- J：16 篇 × 4 窗口，seed {SEEDS["J"]}；calibration：8 批 × 5 篇 × 2 窗口（每批 1270 預測 token），seed {SEEDS["calibration"]}，抽樣順序決定批次。',
          '- D、V、H 已看過，只作回顧。', '',
          '## 分數', '', '- 每批重新估計 B（BC/1270）與 F（Σ_s〈g_s, Q(W)−W〉/1270，FP64 累加）；聯合候選加總。',
          '- 比較 B、F、F+0.5B（主要）、F+B（敏感度）；不截斷、不調係數、不事後選方法。B 不是 Hessian。',
          '- 某批無 token 的 expert 標為「無觀測」：不進入忠實度、穩定性與覆蓋分析；主要終點中各方法同為名稱 tie-break，差為 0，另報只含有觀測格的次要版本。', '',
          '## 主要終點', '',
          '- Δ = J loss(同批 F+0.5B 選擇) − J loss(同批 BC 選擇)；63 群組 × 8 批等權平均，再以 J 文章為單位。',
          f'- 95% CI：{BOOT} 次雙向 bootstrap（批次重抽 + J 文章依 validation/test 分層重抽），seed {SEEDS["bootstrap"]}；同時列出每批結果。只有 8 批，批次部分的不確定性估計粗糙。', '',
          '## 其他預先指定分析', '', '1. 同批忠實度；2. 跨批穩定性（含 BC 與實測）；4. 以各批實測 loss 當診斷選擇器的轉移分解；'
          '5. 同 expert 內覆蓋與不穩定的關聯（連續 token 數、置換檢定、Holm）；6. L15.E17、L15.E59、L8.E33 案例。',
          '- 判讀指引（非門檻）見 plan.json → analyses.interpretation_guides。', '',
          '## 不做', '', '- 不依 J 調係數；不加 coverage 門檻、shrinkage、fallback；不排除 expert；不把多批平均分數包裝成新方法；不改預設 pipeline；不做完整配置。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'candidates.json', candidates())
    log('plan written')


def data(out, log):
    import olmoe_common as oc
    from run_olmoe_crosstext import used_spans, articles
    if (out / 'segments.json').exists():
        raise RuntimeError('data already drawn')
    pl = read(out / 'plan.json')
    for k, v in pl['inputs'].items():
        if k.startswith('gpt2_assets') and sha(HERE / k) != v:
            raise RuntimeError('data file changed: ' + k)
    used = used_spans()
    prior = read(CT / 'segments.json') + read(FO / 'segments.json')
    titles = {s['title'] for s in prior}
    for s in prior:
        used[s['split']].append(tuple(s['article_chars']))
    tok = oc.tokenizer()
    segs, avail, jt, vt_titles = [], {}, [], set()
    rng = np.random.default_rng(SEEDS['J'])
    for split in ('validation', 'test'):
        arts, ids, off = articles(split, tok)
        vt_titles |= {a['title'] for a in arts}
        free = [a for a in arts if a['title'] not in titles and a['tokens'][1] - a['tokens'][0] >= 4 * SEQ
                and not any(a['chars'][0] < b and a['chars'][1] > a0 for a0, b in used[split])]
        avail['J_' + split] = len(free)
        if len(free) < 8:
            raise RuntimeError('insufficient unused articles for J')
        for k in sorted(rng.choice(len(free), 8, replace=False).tolist()):
            a = free[k]
            st = int(rng.integers(a['tokens'][0], a['tokens'][1] - 4 * SEQ + 1))
            wc = [int(off[st][0]), int(off[st + 4 * SEQ - 1][1])]
            assert a['chars'][0] <= wc[0] and wc[1] <= a['chars'][1]
            jt.append(np.array(ids[st:st + 4 * SEQ], dtype=np.int64).reshape(4, SEQ))
            segs.append({'set': 'J', 'split': split, 'title': a['title'], 'article_chars': a['chars'], 'window_chars': wc,
                         'tokens': [st, st + 4 * SEQ], 'index': len(jt) - 1, 'windows': [4 * (len(jt) - 1) + i for i in range(4)]})
            titles.add(a['title'])
    arts, ids, off = articles('train', tok)
    cnt = Counter(a['title'] for a in arts)
    free = [a for a in arts if cnt[a['title']] == 1 and a['title'] not in vt_titles and a['title'] not in titles
            and a['tokens'][1] - a['tokens'][0] >= WPA * SEQ]
    avail['calibration_train'] = len(free)
    avail['train_articles'] = len(arts)
    if len(free) < NB * APB:
        raise RuntimeError('insufficient train articles')
    rng = np.random.default_rng(SEEDS['calibration'])
    pick = rng.choice(len(free), NB * APB, replace=False).tolist()
    toks = {b: [] for b in BATCHES}
    for r, k in enumerate(pick):
        b, a = BATCHES[r // APB], free[k]
        st = int(rng.integers(a['tokens'][0], a['tokens'][1] - WPA * SEQ + 1))
        wc = [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])]
        assert a['chars'][0] <= wc[0] and wc[1] <= a['chars'][1]
        toks[b].append(np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ))
        n = len(toks[b]) - 1
        segs.append({'set': b, 'split': 'train', 'title': a['title'], 'article_chars': a['chars'], 'window_chars': wc,
                     'tokens': [st, st + WPA * SEQ], 'index': n, 'windows': [WPA * n + i for i in range(WPA)]})
    if len({s['title'] for s in segs}) != len(segs):
        raise RuntimeError('an article appears twice')
    for name, arr in [('J', jt)] + [(b, toks[b]) for b in BATCHES]:
        (out / name).mkdir(exist_ok=True)
        np.save(out / name / 'tokens.npy', np.concatenate(arr))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'),
                                     'available': avail, 'segments_sha256': sha(out / 'segments.json'),
                                     'tokens_sha256': {n: sha(out / n / 'tokens.npy') for n in ('J',) + BATCHES}})
    log(f'data frozen: {avail}')


def capture(out, log, name):
    from run_olmoe_crosstext import stage_capture
    if name == 'J' and not (out / 'freeze.json').exists():
        raise RuntimeError('J routes only after freeze')
    if sha(out / name / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256'][name]:
        raise RuntimeError('tokens changed')
    stage_capture(SimpleNamespace(set=name, device='cuda'), out, log)


def score(out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    from run_olmoe_crosstext import load_set_routes
    from run_olmoe_bc_numerics import quantizers
    gpu_exact('cuda')
    torch.set_num_threads(8)
    dev = 'cuda'
    (out / 'scores').mkdir(exist_ok=True)
    (out / 'work').mkdir(exist_ok=True)
    cfg, ck = oc.config(), oc.Checkpoint()
    cs = read(out / 'candidates.json')
    want = sorted({(c['layer'], c['expert']) for c in cs})
    co, si = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    q, _ = quantizers(out / 'work', dev, 'corrected')
    sets = [('C0', np.load(S1 / 'tokens_calibration.npy'), lambda: load_routes(S1, 'calibration', cfg, dev)[0],
             read(S1 / 'routes_manifest.json')['calibration']['summed_ce_dynamic'])]
    sets += [(b, np.load(out / b / 'tokens.npy'), (lambda b=b: load_set_routes(out, b, dev)),
              read(out / b / 'routes_manifest.json')['w']['summed_ce_dynamic']) for b in BATCHES]
    for name, toks_np, routes, dyn in sets:
        if (out / 'scores' / f'{name}.json').exists():
            raise RuntimeError(f'scores {name} exist')
        if name != 'C0' and sha(out / name / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256'][name]:
            raise RuntimeError('tokens changed')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        rt = routes()
        toks = torch.from_numpy(toks_np).to(dev)
        assert toks.shape == (WPB, SEQ)
        with torch.no_grad():
            hs = [F.embedding(toks, g['embed'])]
            for l in range(cfg.num_hidden_layers):
                hs.append(oc.layer_fixed(ck.layer(l, dev, torch.float32), hs[-1], cfg, co, si, *rt[l], l)[0])
        ht = hs[-1].detach().requires_grad_(True)
        logits = oc.head(g, ht, cfg)
        ce = oc.summed_ce(logits, toks)
        n_pred = int(logits[:, :-1].shape[1])
        base = ce.detach().cpu().numpy().tolist()
        gn, = torch.autograd.grad(ce.sum(), ht)
        del logits
        proj, cov, dsha = {}, {}, {}
        for l in reversed(range(cfg.num_hidden_layers)):
            w = ck.layer(l, dev, torch.float32)
            cap = {}
            hk = oc.ExpertHooks()

            def tap(ll, e, p, s, t):
                if (ll, e) in want:
                    t.retain_grad()
                    cap[(e, p, s)] = t
            hk.capture = tap
            hi = hs[l].detach().requires_grad_(True)
            o = oc.layer_fixed(w, hi, cfg, co, si, *rt[l], l, hk)[0]
            o.backward(gn)
            gn = hi.grad.detach()
            for e in sorted(e for ll, e in want if ll == l):
                node = f'L{l}.E{e}'
                routed = (rt[l][0] == e).any(-1).view(WPB, SEQ)
                pw = routed.sum(1).cpu().numpy().astype(int)
                cov[node] = {'routed_per_window': pw.tolist(), 'routed_tokens': int(pw.sum()),
                             'routed_predicting_positions': int(routed[:, :-1].sum()),
                             'windows_with_tokens': int((pw > 0).sum()),
                             'per_article': pw.reshape(APB, WPA).sum(1).tolist(),
                             'articles_with_tokens': int((pw.reshape(APB, WPA).sum(1) > 0).sum())}
                observed = (e, 'gate_proj', 'x') in cap
                if observed:
                    _, tok = torch.where(F.one_hot(rt[l][0], num_classes=cfg.num_experts + 1).permute(2, 1, 0)[e])
                    win = tok // SEQ
                for p in oc.PROJ:
                    pn = f'{node}.{p}'
                    W = w[p][e].detach().cpu()
                    deltas = {}
                    for f in FORMATS:
                        d32 = (q.weight(W, f) - W).float().contiguous()
                        dsha[f'{pn}.{f}'] = hashlib.sha256(d32.numpy().tobytes()).hexdigest()
                        deltas[f] = d32.to(device=dev, dtype=torch.float64)
                    gs = torch.zeros(W.shape, device=dev, dtype=torch.float64)
                    g2 = torch.zeros(W.shape[0], device=dev, dtype=torch.float64)
                    perw = {f: [0.0] * WPB for f in FORMATS}
                    if observed:
                        x = cap[(e, 'gate_proj', 'x')] if p != 'down_proj' else cap[(e, p, 'x')]
                        gy = cap[(e, p, 'y')].grad
                        for s in range(WPB):
                            m = win == s
                            if not bool(m.any()):
                                continue
                            gr = gy[m].double().T @ x[m].detach().double()
                            gs += gr
                            g2 += gr.square().sum(1)
                            for f, d in deltas.items():
                                perw[f][s] = float((gr * d).sum()) / TP
                    vals = {}
                    for f, d in deltas.items():
                        vals[f] = {'F': float((gs * d).sum()) / DENOM,
                                   'B': float((g2 / W.shape[1] * d.square().sum(1)).sum()) / DENOM,
                                   'F_per_window': perw[f]}
                    proj[pn] = {'observed': observed, 'formats': vals}
                    del deltas, gs, g2
            del cap, hk, o, hi, w
        torch.cuda.synchronize()
        sec = time.time() - t0
        rows = []
        for c in cs:
            f_ = sum(proj[c['node'] + '.' + p]['formats'][wf]['F'] for p, (wf, af) in c['formats'].items())
            b_ = sum(proj[c['node'] + '.' + p]['formats'][wf]['B'] for p, (wf, af) in c['formats'].items())
            rows.append({'name': c['name'], 'bc': b_, 'first': f_, 'first_plus_half_bc': f_ + 0.5 * b_, 'first_plus_bc': f_ + b_})
        write(out / 'scores' / f'{name}.json', {
            'set': name, 'seconds': sec, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'predicted_tokens_per_window': n_pred, 'denominator': DENOM,
            'base_summed_ce': base, 'base_minus_capture_max_abs': float(np.max(np.abs(np.array(base) - np.array(dyn)))),
            'coverage': cov, 'delta_sha256': dsha, 'projections': proj, 'candidates': rows})
        log(f'{name}: scores in {sec:.1f}s; baseline-capture max |diff| '
            f'{np.max(np.abs(np.array(base) - np.array(dyn))):.3g}; observed experts '
            f'{sum(v["routed_tokens"] > 0 for v in cov.values())}/9')


def caltruth(out, log):
    from run_olmoe_crosstext import replay, load_set_routes
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact('cuda')
    cs = read(out / 'candidates.json')
    for b in BATCHES:
        if (out / b / 'truth.json').exists():
            raise RuntimeError(f'{b} truth exists')
        if sha(out / b / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256'][b]:
            raise RuntimeError('tokens changed')
        r = replay(out, np.load(out / b / 'tokens.npy'), load_set_routes(out, b, 'cuda'), cs, 'cuda', log, b + ' truth',
                   work='truth_work')
        r['base_minus_capture_max_abs'] = float(np.max(np.abs(np.array(r['base']) - np.array(read(out / b / 'routes_manifest.json')['w']['summed_ce_dynamic']))))
        r['script_sha256'] = sha(__file__)
        write(out / b / 'truth.json', r)


def load_scores(out, name):
    s = read(out / 'scores' / f'{name}.json')
    return {m: np.array([r[m] for r in s['candidates']]) for m in METHODS}, s


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise RuntimeError('freeze exists')
    cs = read(out / 'candidates.json'); nms = [c['name'] for c in cs]
    gr = groups_of(cs)
    res, checks = {}, {}
    for name in ('C0',) + BATCHES:
        v, s = load_scores(out, name)
        assert [r['name'] for r in s['candidates']] == nms
        res[name] = {m: [nms[i] for i in choose(gr, v[m], nms)] for m in METHODS}
        if name != 'C0':
            t = read(out / name / 'truth.json')
            meas = np.array([np.mean(t['rows'][n]) for n in nms]) / TP
            res[name]['measured'] = [nms[i] for i in choose(gr, meas, nms)]
    # references
    orig = np.array([c['bc'] for c in cs])
    res['reference'] = {'original_bc': [nms[i] for i in choose(gr, orig, nms)],
                        **{'previous_' + m: [x['name'] for x in read(FO / 'choices.json')[m]] for m in METHODS}}
    # implementation checks on C0 against the previous round
    prev = read(FO / 'projection_scores.json')
    _, s0 = load_scores(out, 'C0')
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-30)
    checks['C0_F_max_relative'] = max(rel(s0['projections'][pn]['formats'][f]['F'], v['formats'][f]['first']) for pn, v in prev.items() for f in FORMATS)
    checks['C0_B_max_relative'] = max(rel(s0['projections'][pn]['formats'][f]['B'], v['formats'][f]['bc_reconstructed']) for pn, v in prev.items() for f in FORMATS)
    checks['C0_B_vs_original_table_max_relative'] = max(rel(s0['projections'][pn]['formats'][f]['B'], v['formats'][f]['bc']) for pn, v in prev.items() for f in FORMATS)
    checks['C0_choices_equal_previous'] = {m: res['C0'][m] == res['reference']['previous_' + m] for m in METHODS}
    checks['C0_bc_choice_equal_original_table_choice'] = res['C0']['bc'] == res['reference']['original_bc']
    dsh = {n: read(out / 'scores' / f'{n}.json')['delta_sha256'] for n in ('C0',) + BATCHES}
    checks['delta_identical_all_sets'] = all(dsh[n] == dsh['C0'] for n in dsh)
    perw = []
    for n in ('C0',) + BATCHES:
        s = read(out / 'scores' / f'{n}.json')
        checks.setdefault('predicted_tokens_per_window', []).append(s['predicted_tokens_per_window'])
        checks.setdefault('score_baseline_minus_capture_max_abs', {})[n] = s['base_minus_capture_max_abs']
        for pn, v in s['projections'].items():
            for f, x in v['formats'].items():
                perw.append(rel(np.mean(x['F_per_window']) * TP * WPB / DENOM, x['F']) if x['F'] != 0 else abs(np.mean(x['F_per_window'])))
        rows = {r['name']: r for r in s['candidates']}
        err = 0.0
        for c in cs:
            pf = sum(s['projections'][c['node'] + '.' + p]['formats'][wf]['F'] for p, (wf, af) in c['formats'].items())
            pb = sum(s['projections'][c['node'] + '.' + p]['formats'][wf]['B'] for p, (wf, af) in c['formats'].items())
            r = rows[c['name']]
            err = max(err, abs(r['first'] - pf), abs(r['bc'] - pb), abs(r['first_plus_half_bc'] - pf - 0.5 * pb), abs(r['first_plus_bc'] - pf - pb))
        checks.setdefault('joint_additivity_max_abs', {})[n] = err
    checks['F_equals_mean_per_window_max_relative'] = max(perw)
    checks['truth_baseline_minus_capture_max_abs'] = {b: read(out / b / 'truth.json')['base_minus_capture_max_abs'] for b in BATCHES}
    checks['truth_all_finite'] = {b: bool(np.isfinite(np.array([read(out / b / 'truth.json')['rows'][n] for n in nms])).all()) for b in BATCHES}
    checks['candidates_equal_stage1_weight_only'] = cs == candidates()
    checks['groups'] = len(gr)
    checks['all_choices_within_group'] = all(nms.index(x) in g_ for k_ in res for m_ in res[k_] for g_, x in zip(gr, res[k_][m_]))
    checks['pass'] = (checks['C0_F_max_relative'] <= 1e-9 and checks['C0_B_max_relative'] <= 1e-9
                      and all(checks['C0_choices_equal_previous'].values()) and checks['delta_identical_all_sets']
                      and set(checks['predicted_tokens_per_window']) == {TP}
                      and max(checks['score_baseline_minus_capture_max_abs'][n] for n in BATCHES) == 0.0
                      and max(checks['truth_baseline_minus_capture_max_abs'].values()) == 0.0
                      and all(checks['truth_all_finite'].values()) and checks['candidates_equal_stage1_weight_only']
                      and max(checks['joint_additivity_max_abs'].values()) < 1e-15
                      and checks['F_equals_mean_per_window_max_relative'] < 1e-9 and checks['all_choices_within_group'])
    write(out / 'checks.json', checks)
    if not checks['pass']:
        raise RuntimeError('implementation checks failed; see checks.json (nothing frozen)')
    write(out / 'choices.json', res)
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'),
                                'data_freeze_sha256': sha(out / 'data_freeze.json'), 'choices_sha256': sha(out / 'choices.json'),
                                'checks_sha256': sha(out / 'checks.json'),
                                'scores_sha256': {n: sha(out / 'scores' / f'{n}.json') for n in ('C0',) + BATCHES},
                                'caltruth_sha256': {b: sha(out / b / 'truth.json') for b in BATCHES}})
    log('choices frozen; checks pass')


def jtruth(out, log):
    from run_olmoe_crosstext import replay, load_set_routes
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact('cuda')
    fz = read(out / 'freeze.json')
    if sha(out / 'choices.json') != fz['choices_sha256']:
        raise RuntimeError('choices changed after freeze')
    if (out / 'J' / 'truth.json').exists():
        raise RuntimeError('J truth exists')
    if sha(out / 'J' / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']['J']:
        raise RuntimeError('tokens changed')
    r = replay(out, np.load(out / 'J/tokens.npy'), load_set_routes(out, 'J', 'cuda'), read(out / 'candidates.json'), 'cuda',
               log, 'J truth', work='truth_work')
    r['base_minus_capture_max_abs'] = float(np.max(np.abs(np.array(r['base']) - np.array(read(out / 'J/routes_manifest.json')['w']['summed_ce_dynamic']))))
    r['script_sha256'] = sha(__file__)
    write(out / 'J' / 'truth.json', r)


# --------------------------------------------------------------------------- #
# analysis helpers
def strat_idx(splits, rng, n=BOOT):
    return np.concatenate([rng.choice(np.where(np.array(splits) == sp)[0], (n, int((np.array(splits) == sp).sum())), replace=True)
                           for sp in ('validation', 'test')], 1)


def two_way(M, bidx, aidx):
    """M [batches, articles] -> bootstrap means with batches and articles resampled independently."""
    A = M[:, aidx].mean(-1)                                  # [batches, BOOT]
    return np.take_along_axis(A.T, bidx, axis=1).mean(1)


def ci(draws, point):
    return {'mean': float(point), 'ci95': np.percentile(draws, [2.5, 97.5]).tolist()}


def article_values(truth, nms, wpa):
    s = np.array([truth['rows'][n] for n in nms]) / TP
    return s.reshape(len(nms), -1, wpa).mean(2)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(np.round(x, 15))) < 2 or len(set(np.round(y, 15))) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def analyze(out, log):
    from run_olmoe_bc_numerics import pair_list
    t0 = time.time()
    fz = read(out / 'freeze.json')
    if sha(out / 'choices.json') != fz['choices_sha256']:
        raise RuntimeError('choices changed after freeze')
    cs = read(out / 'candidates.json'); nms = [c['name'] for c in cs]; ix = {n: i for i, n in enumerate(nms)}
    nodes = sorted({c['node'] for c in cs}, key=lambda n: (int(n.split('.')[0][1:]), int(n.split('.')[1][1:])))
    gr = groups_of(cs); gnode = np.array([cs[g[0]]['node'] for g in gr]); G = len(gr)
    pairs = pair_list(cs); ii = np.array([p['i'] for p in pairs]); jj = np.array([p['j'] for p in pairs])
    pnode = np.array([p['node'] for p in pairs]); P = len(pairs)
    ch = read(out / 'choices.json')
    C = {k: {m: np.array([ix[x] for x in v]) for m, v in d.items()} for k, d in ch.items()}
    S, cov, obs, meas, meas_w = {}, {}, {}, {}, {}
    for n in ('C0',) + BATCHES:
        S[n], s = load_scores(out, n)
        cov[n] = s['coverage']
        obs[n] = {nd: cov[n][nd]['routed_tokens'] > 0 for nd in nodes}
    for b in BATCHES:
        t = read(out / b / 'truth.json')
        meas_w[b] = np.array([t['rows'][n] for n in nms]) / TP        # [360, 10]
        meas[b] = meas_w[b].mean(1)
    t = read(S1 / 'truth_calibration.json')
    meas_w['C0'] = np.array([t['rows'][n] for n in nms]) / TP
    meas['C0'] = meas_w['C0'].mean(1)
    segs = read(out / 'segments.json')
    jsegs = sorted([s for s in segs if s['set'] == 'J'], key=lambda s: s['index'])
    J = article_values(read(out / 'J/truth.json'), nms, 4)               # [360, 16]
    Jm = J.mean(1)
    rng = np.random.default_rng(SEEDS['bootstrap'])
    bidx = rng.integers(0, NB, (BOOT, NB))
    aidx = strat_idx([s['split'] for s in jsegs], rng)
    pobs = {n: np.array([obs[n][x] for x in pnode]) for n in obs}
    gobs = {n: np.array([obs[n][x] for x in gnode]) for n in obs}
    res = {'counts': {'candidates': len(cs), 'pairs': P, 'groups': G, 'batches': NB, 'J_articles': J.shape[1]},
           'checks': read(out / 'checks.json')}
    sgn = np.sign

    # ---- coverage table
    res['coverage'] = {n: {nd: {k: cov[n][nd][k] for k in ('routed_tokens', 'windows_with_tokens', 'articles_with_tokens', 'per_article')}
                           for nd in nodes} for n in ('C0',) + BATCHES}
    res['no_observation_cells'] = {b: [nd for nd in nodes if not obs[b][nd]] for b in BATCHES}

    # ---- 1 fidelity
    fid = {}
    for n in ('C0',) + BATCHES:
        md = meas[n][ii] - meas[n][jj]; o = pobs[n] & (md != 0)
        fid[n] = {'observed_pairs': int(o.sum())}
        for m in METHODS:
            sd = S[n][m][ii] - S[n][m][jj]
            sp = [spearman(S[n][m][[c_['node'] == nd for c_ in cs]], meas[n][[c_['node'] == nd for c_ in cs]]) for nd in nodes if obs[n][nd]]
            sp = [x for x in sp if x is not None]
            chosen = choose(gr, S[n][m], nms)
            best = np.array([meas[n][g].min() for g in gr])
            fid[n][m] = {'sign_agreement': float((sgn(sd[o]) == sgn(md[o])).mean()),
                         'mean_within_expert_spearman': float(np.mean(sp)) if sp else None,
                         'in_batch_regret_observed_groups': float((meas[n][chosen] - best)[gobs[n]].mean())}
    res['fidelity'] = fid
    res['fidelity_summary'] = {m: {'median_over_batches': float(np.median([fid[b][m]['sign_agreement'] for b in BATCHES])),
                                   'min': float(min(fid[b][m]['sign_agreement'] for b in BATCHES)),
                                   'max': float(max(fid[b][m]['sign_agreement'] for b in BATCHES)),
                                   'C0': fid['C0'][m]['sign_agreement']} for m in METHODS}

    # ---- 2 stability across the 8 batches
    def sign_matrix(get):
        M = np.full((NB, P), np.nan)
        for k, b in enumerate(BATCHES):
            d = get(b)
            M[k, pobs[b]] = sgn(d[pobs[b]])
        return M
    sm = {m: sign_matrix(lambda b, m=m: S[b][m][ii] - S[b][m][jj]) for m in METHODS}
    sm['measured'] = sign_matrix(lambda b: meas[b][ii] - meas[b][jj])
    stab = {}
    prow = []
    for m, M in sm.items():
        npl, nmi, nti, nno = (M == 1).sum(0), (M == -1).sum(0), (M == 0).sum(0), np.isnan(M).sum(0)
        nob = npl + nmi
        maj = np.where(nob > 0, np.maximum(npl, nmi) / np.maximum(nob, 1), np.nan)
        full = nno == 0
        agree = []
        for a in range(NB):
            for b in range(a + 1, NB):
                o = ~np.isnan(M[a]) & ~np.isnan(M[b]) & (M[a] != 0) & (M[b] != 0)
                agree.append((M[a][o] == M[b][o]).mean())
        stab[m] = {'mean_majority_share': float(np.nanmean(maj)), 'unanimous_share_fully_observed': float((maj[full] == 1).mean()),
                   'fully_observed_pairs': int(full.sum()), 'mean_pairwise_batch_agreement': float(np.mean(agree)),
                   'ties': int(nti.sum()), 'no_observation_cells': int(nno.sum())}
        for k in range(P):
            if m == METHODS[0]:
                prow.append({'a': nms[ii[k]], 'b': nms[jj[k]], 'node': pnode[k]})
            prow[k].update({f'{m}_plus': int(npl[k]), f'{m}_minus': int(nmi[k]), f'{m}_tie': int(nti[k]), f'{m}_noobs': int(nno[k])})
    for k in range(P):
        prow[k]['J_mean_a_minus_b'] = float(Jm[ii[k]] - Jm[jj[k]])
    with open(out / 'pair_stability.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)
    gstab, grow = {}, []
    for m in METHODS + ('measured',):
        nd_, ms_, un_ = [], [], []
        for k in range(G):
            o = [b for b in BATCHES if gobs[b][k]]
            if not o:
                continue
            picks = Counter(int(C[b][m][k]) for b in o)
            nd_.append(len(picks)); ms_.append(max(picks.values()) / len(o)); un_.append(len(picks) == 1)
        gstab[m] = {'mean_distinct_choices': float(np.mean(nd_)), 'mean_modal_share': float(np.mean(ms_)),
                    'unanimous_groups': int(sum(un_)), 'groups_with_observation': len(nd_)}
    for k in range(G):
        r = {'node': gnode[k], 'M_bytes': cs[gr[k][0]]['M_bytes'], 'BOPs_calibration': cs[gr[k][0]]['BOPs_calibration'], 'size': len(gr[k]),
             'original_bc': nms[C['reference']['original_bc'][k]], 'previous_primary': nms[C['reference']['previous_' + PRIMARY][k]]}
        for b in BATCHES:
            r[f'{b}_bc'] = nms[C[b]['bc'][k]] if gobs[b][k] else 'NO_OBS'
            r[f'{b}_primary'] = nms[C[b][PRIMARY][k]] if gobs[b][k] else 'NO_OBS'
            r[f'{b}_measured'] = nms[C[b]['measured'][k]] if gobs[b][k] else 'NO_OBS'
        r['J_best'] = nms[gr[k][int(np.argmin(Jm[gr[k]]))]]
        grow.append(r)
    with open(out / 'group_choices.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(grow[0])); w.writeheader(); w.writerows(grow)
    res['stability'] = {'pairs': stab, 'groups': gstab}

    # ---- 3 primary + comparisons on J (and retrospective sets)
    def cell_matrix(A_, B_, T):
        """[batches, groups, articles] T(choice A) - T(choice B)."""
        return np.stack([T[A_(b)] - T[B_(b)] for b in BATCHES])
    comps = {'primary_minus_bc': (lambda b: C[b][PRIMARY], lambda b: C[b]['bc']),
             'first_minus_bc': (lambda b: C[b]['first'], lambda b: C[b]['bc']),
             'first_plus_bc_minus_bc': (lambda b: C[b]['first_plus_bc'], lambda b: C[b]['bc']),
             'measured_minus_bc': (lambda b: C[b]['measured'], lambda b: C[b]['bc']),
             'primary_minus_measured': (lambda b: C[b][PRIMARY], lambda b: C[b]['measured']),
             'bc_minus_original_bc': (lambda b: C[b]['bc'], lambda b: C['reference']['original_bc']),
             'primary_minus_original_bc': (lambda b: C[b][PRIMARY], lambda b: C['reference']['original_bc'])}
    jres = {}
    cells = {}
    for name, (fa, fb) in comps.items():
        X = cell_matrix(fa, fb, J)                                   # [8, 63, 16]
        cells[name] = X
        M = X.mean(1)                                               # [8, 16]
        d = two_way(M, bidx, aidx)
        e = {'two_way': ci(d, M.mean()), 'article_only': ci(M.mean(0)[aidx].mean(1), M.mean())}
        e['per_batch'] = {b: ci(M[k][aidx].mean(1), M[k].mean()) for k, b in enumerate(BATCHES)}
        ob = np.stack([gobs[b] for b in BATCHES])                    # [8, 63]
        Mo = np.stack([X[k][ob[k]].mean(0) for k in range(NB)])
        e['observed_cells_only'] = ci(two_way(Mo, bidx, aidx), Mo.mean())
        e['per_expert'] = {nd: ci(two_way(X[:, gnode == nd].mean(1), bidx, aidx), X[:, gnode == nd].mean()) for nd in nodes}
        e['changed_cells'] = int(sum((C[b][PRIMARY] != C[b]['bc']).sum() for b in BATCHES)) if name == 'primary_minus_bc' else None
        jres[name] = e
    for name, (fa, fb) in {'previous_primary_minus_original_bc': (lambda: C['reference']['previous_' + PRIMARY], lambda: C['reference']['original_bc'])}.items():
        X = J[fa()] - J[fb()]
        jres[name] = {'article_only': ci(X.mean(0)[aidx].mean(1), X.mean())}
    res['J'] = jres
    res['J_damage'] = {m: float(np.mean([J[C[b][m]].mean() for b in BATCHES])) for m in METHODS + ('measured',)}
    res['J_damage']['original_bc'] = float(J[C['reference']['original_bc']].mean())
    retro = {}
    for lab, path, wpa, meta in [('D', CT / 'D/truth.json', 4, sorted([s for s in read(CT / 'segments.json') if s['group'] == 'D'], key=lambda s: s['index_in_group'])),
                                 ('V', CT / 'V/truth.json', 4, sorted([s for s in read(CT / 'segments.json') if s['group'] == 'V'], key=lambda s: s['index_in_group'])),
                                 ('H', FO / 'H/truth.json', 4, sorted(read(FO / 'segments.json'), key=lambda s: s['index']))]:
        T = article_values(read(path), nms, wpa)
        r2 = np.random.default_rng(SEEDS['bootstrap'] + {'D': 1, 'V': 2, 'H': 3}[lab])
        b2, a2 = r2.integers(0, NB, (BOOT, NB)), strat_idx([s['split'] for s in meta], r2)
        retro[lab] = {}
        for name in ('primary_minus_bc', 'measured_minus_bc', 'bc_minus_original_bc'):
            fa, fb = comps[name]
            M = cell_matrix(fa, fb, T).mean(1)
            retro[lab][name] = ci(two_way(M, b2, a2), M.mean())
    res['retrospective'] = retro

    # ---- 4 transfer decomposition (pair level)
    Jd = Jm[ii] - Jm[jj]
    Jart = J[ii] - J[jj]
    Jci = np.percentile(np.stack([Jart[:, r].mean(1) for r in aidx], 1), [2.5, 97.5], axis=1)
    Jres = (Jci[0] > 0) | (Jci[1] < 0)
    trans = {}
    for m in METHODS + ('measured',):
        tot = Counter(); wt = Counter(); rs = Counter()
        for b in BATCHES:
            md = sgn(meas[b][ii] - meas[b][jj])
            sd = md if m == 'measured' else sgn(S[b][m][ii] - S[b][m][jj])
            o = pobs[b] & (sd != 0) & (md != 0) & (Jd != 0)
            right = sd == sgn(Jd)
            approx = ~right & (sd != md)
            transfer = ~right & (sd == md)
            for k_, v_ in (('correct', right), ('approximation_error', approx), ('transfer_error', transfer)):
                tot[k_] += int((v_ & o).sum()); wt[k_] += float(np.abs(Jd)[v_ & o].sum()); rs[k_] += int((v_ & o & Jres).sum())
        n = sum(tot.values()); wn = sum(wt.values()); rn = sum(rs.values())
        trans[m] = {'cells': n, 'share': {k: tot[k] / n for k in tot}, 'abs_J_weighted_share': {k: wt[k] / wn for k in wt},
                    'J_resolved_cells': rn, 'J_resolved_share': {k: rs[k] / rn for k in rs} if rn else None}
    trans['J_resolved_pairs_unadjusted'] = int(Jres.sum())
    trans['batch_truth_vs_J_sign_agreement'] = {b: float((sgn(meas[b][ii] - meas[b][jj])[pobs[b]] == sgn(Jd)[pobs[b]]).mean()) for b in BATCHES}
    trans['C0_truth_vs_J_sign_agreement'] = float((sgn(meas['C0'][ii] - meas['C0'][jj]) == sgn(Jd)).mean())
    trans['score_vs_J_sign_agreement'] = {m: float(np.mean([(sgn(S[b][m][ii] - S[b][m][jj])[pobs[b]] == sgn(Jd)[pobs[b]]).mean() for b in BATCHES])) for m in METHODS}

    def vc(Wd, wpa):
        """per pair one-way variance components: windows [pairs, windows] grouped in articles of wpa windows."""
        A_ = Wd.reshape(Wd.shape[0], -1, wpa)
        msw = A_.var(2, ddof=1).mean(1)
        msb = wpa * A_.mean(2).var(1, ddof=1)
        sa = np.maximum(0, (msb - msw) / wpa)
        return np.where(sa + msw > 0, sa / np.maximum(sa + msw, 1e-300), np.nan)
    calw = np.concatenate([meas_w[b] for b in BATCHES], 1)               # [360, 80], 40 articles x 2 windows
    jw = np.array([read(out / 'J/truth.json')['rows'][n] for n in nms]) / TP
    icc_c, icc_j = vc(calw[ii] - calw[jj], WPA), vc(jw[ii] - jw[jj], 4)
    trans['between_article_share'] = {
        'calibration_median': float(np.nanmedian(icc_c)), 'J_median': float(np.nanmedian(icc_j)),
        'per_expert': {nd: {'calibration': float(np.nanmedian(icc_c[pnode == nd])), 'J': float(np.nanmedian(icc_j[pnode == nd]))} for nd in nodes}}
    res['transfer'] = trans

    # ---- 5 coverage vs instability (within expert across batches)
    prng = np.random.default_rng(SEEDS['permutation'])
    rows5 = []
    pooled = {m: np.mean([S[b][m] for b in BATCHES], 0) for m in METHODS}
    gbest = np.array([J[g].mean(1).min() for g in gr])
    for k, b in enumerate(BATCHES):
        for nd in nodes:
            if not obs[b][nd]:
                continue
            pm, gm = pnode == nd, gnode == nd
            r = {'batch': b, 'node': nd, 'tokens': cov[b][nd]['routed_tokens'], 'articles': cov[b][nd]['articles_with_tokens'],
                 'windows': cov[b][nd]['windows_with_tokens']}
            for m in ('first', PRIMARY, 'bc'):
                loo = (pooled[m] * NB - S[b][m]) / (NB - 1)
                r[f'o1_{m}'] = float((sgn(S[b][m][ii] - S[b][m][jj])[pm] != sgn(loo[ii] - loo[jj])[pm]).mean())
                r[f'o2_{m}'] = float((C[b][m][gm] != choose([g for g, x in zip(gr, gm) if x], loo, nms)).mean())
            r['o3_primary'] = float((J[C[b][PRIMARY]].mean(1) - gbest)[gm].mean())
            r['o3_bc'] = float((J[C[b]['bc']].mean(1) - gbest)[gm].mean())
            r['delta_J_primary_minus_bc'] = float((J[C[b][PRIMARY]].mean(1) - J[C[b]['bc']].mean(1))[gm].mean())
            r['fidelity_primary'] = float((sgn(S[b][PRIMARY][ii] - S[b][PRIMARY][jj]) == sgn(meas[b][ii] - meas[b][jj]))[pm].mean())
            rows5.append(r)
    with open(out / 'coverage_outcomes.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows5[0])); w.writeheader(); w.writerows(rows5)

    def within(outcome, cover='tokens'):
        """mean over experts of the within-expert Spearman(coverage, outcome); permutation of coverage within expert.
        Spearman = Pearson of average ranks; experts with constant coverage or outcome are not usable."""
        from scipy.stats import rankdata
        by = defaultdict(list)
        for r in rows5:
            by[r['node']].append((np.log1p(r[cover]) if cover == 'tokens' else r[cover], r[outcome]))
        usable = {}
        for nd, v in by.items():
            v = np.array(v, dtype=float)
            if len(v) >= 3 and np.ptp(v[:, 0]) > 0 and np.ptp(v[:, 1]) > 0:
                rx, ry = rankdata(v[:, 0]), rankdata(v[:, 1])
                usable[nd] = ((rx - rx.mean()) / np.linalg.norm(rx - rx.mean()), (ry - ry.mean()) / np.linalg.norm(ry - ry.mean()))
        if not usable:
            return {'mean_within_expert_spearman': None, 'p_two_sided': None, 'experts_used': 0, 'per_expert': {}}
        obs_ = np.mean([float(x @ y) for x, y in usable.values()])
        null = np.zeros(BOOT)
        for x, y in usable.values():
            perm = np.argsort(prng.random((BOOT, len(x))), axis=1)
            null += x[perm] @ y
        null /= len(usable)
        p = float((np.sum(np.abs(null) >= abs(obs_) - 1e-12) + 1) / (BOOT + 1))
        return {'mean_within_expert_spearman': float(obs_), 'p_two_sided': p, 'experts_used': len(usable),
                'per_expert': {nd: float(x @ y) for nd, (x, y) in usable.items()}}
    c5 = {o: within(o) for o in ('o1_' + PRIMARY, 'o2_' + PRIMARY, 'o3_primary', 'o1_first', 'o1_bc', 'o2_bc', 'o3_bc', 'delta_J_primary_minus_bc', 'fidelity_primary')}
    ps = sorted([(c5[o]['p_two_sided'], o) for o in ('o1_' + PRIMARY, 'o2_' + PRIMARY, 'o3_primary')])
    hol, run = {}, 0.0
    for r_, (p, o) in enumerate(ps):
        run = max(run, min(1.0, p * (3 - r_)))
        hol[o] = run
    for o, v in hol.items():
        c5[o]['holm_p'] = v
    c5['articles_cover'] = {o: within(o, 'articles') for o in ('o1_' + PRIMARY, 'o2_' + PRIMARY, 'o3_primary')}
    sel = read(S1 / 'coverage.json')['selection']
    c5['stage1_tiers'] = {f'L{l}.E{v["expert"]}': t for l, x in sel.items() for t, v in x['picks'].items()}
    res['coverage_association'] = c5

    # ---- 6 cases
    cases = {}
    def pair_track(a, b_):
        i, j = ix[a], ix[b_]
        r = {'a': a, 'b': b_, 'J': ci(J[i, aidx].mean(1) - J[j, aidx].mean(1), Jm[i] - Jm[j]),
             'J_articles_a_worse': int((J[i] > J[j]).sum())}
        for n in ('C0',) + BATCHES:
            r[n] = {'tokens': cov[n][cs[i]['node']]['routed_tokens'],
                    **{m: float(S[n][m][i] - S[n][m][j]) for m in METHODS}, 'measured': float(meas[n][i] - meas[n][j])}
        r['sign_counts'] = {m: dict(Counter(int(sgn(r[b][m])) for b in BATCHES)) for m in METHODS + ('measured',)}
        return r
    cases['L15.E17_up_int8_vs_p8e1'] = pair_track('L15.E17|w|up=int8/fp32', 'L15.E17|w|up=p8e1/fp32')
    ex = read(FO / 'exploratory_regressions.json')['pairs']
    cases['L8.E33_three_pairs'] = [pair_track(p['a'], p['b']) for p in ex]
    for nd in ('L15.E17', 'L15.E59'):
        gm = gnode == nd
        cases[nd + '_groups'] = {b: {'tokens': cov[b][nd]['routed_tokens'], 'articles': cov[b][nd]['articles_with_tokens'],
                                     'changed_groups': int((C[b][PRIMARY][gm] != C[b]['bc'][gm]).sum()),
                                     'J_primary_minus_bc': ci(cells['primary_minus_bc'][k][gm].mean(0)[aidx].mean(1), cells['primary_minus_bc'][k][gm].mean()),
                                     'J_regret_primary': float((J[C[b][PRIMARY]].mean(1) - gbest)[gm].mean()),
                                     'J_regret_bc': float((J[C[b]['bc']].mean(1) - gbest)[gm].mean())}
                                 for k, b in enumerate(BATCHES)}
        cases[nd + '_groups']['C0_tokens'] = cov['C0'][nd]['routed_tokens']
    res['cases'] = cases
    res['timing'] = {'score_seconds': {n: read(out / 'scores' / f'{n}.json')['seconds'] for n in ('C0',) + BATCHES},
                     'caltruth_seconds': {b: read(out / b / 'truth.json')['seconds'] for b in BATCHES},
                     'J_truth_seconds': read(out / 'J/truth.json')['seconds'],
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else [],
                     'analysis_seconds': time.time() - t0}
    write(out / 'analysis.json', res)
    log(json.dumps({'primary': res['J']['primary_minus_bc']['two_way'], 'fidelity': res['fidelity_summary'][PRIMARY]}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'data', 'capture', 'score', 'caltruth', 'freeze', 'jtruth', 'analyze'], required=True)
    ap.add_argument('--set')
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise RuntimeError('plan required')
    if a.stage == 'capture':
        capture(a.out_dir, log, a.set)
    else:
        globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'set': a.set, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
