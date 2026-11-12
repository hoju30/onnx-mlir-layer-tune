#!/usr/bin/env python3
"""OLMoE aggressive BC configuration: fixed FP32 expert selection with the selected routing-weight sum restored to
the FP32 baseline sum (diagnostic that uses FP32 information; not a deployable fix). Pre-registered in plan.json.

Per layer of each condition's own forward (own hidden state -> FP32 router -> p over 64 experts):
  L00    ids S0, weights w0 = p0[S0]
  L01    ids S0, weights w = p[S0]
  Lmass  ids S0, weights w * (m0 / m), m = sum(w), m0 = sum(w0)  (float32; sums in S0 order; no clipping / epsilon)
  L11    ids top-8(p), weights p[ids]
S0 / p0 from the complete FP32 dynamic baseline of the same window; no renormalization anywhere (norm_topk_prob false).
Layer arithmetic = run_olmoe_dynamic_route.forward; every condition's recorded ids / weights are replayed through it.

Data: WikiText-103 raw train (WikiText-2 is exhausted), strict article boundaries, every WikiText-2 article and every
within-WikiText-103 duplicate removed by normalized title / content hash, near-duplicates removed by 5-gram shingles.

Stages: plan, pilot (old route-decomposition windows 0-127), data, run, analyze (one --out-dir).
"""
import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr

HERE = Path(__file__).resolve().parent
RD = HERE / 'runs/olmoe_route_decomposition_20261018'
MPT = HERE / 'runs/olmoe_dynamic_multipoint_20261018'
CONFIG = MPT / 'configs/aggressive.json'
WT103 = [HERE / 'gpt2_assets/data_wt103/train-00000-of-00002.parquet', HERE / 'gpt2_assets/data_wt103/train-00001-of-00002.parquet']
WT2 = [dr.DATA / f'{s}-00000-of-00001.parquet' for s in ('train', 'validation', 'test')]
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK, CTRL, PILOT_WINDOWS = 128, 128, 128
CONDS = ('L00', 'L01', 'Lmass', 'L11')
EPS, ALPHA, BOOT = 0.002, 0.05, 10000
H_TARGET, N_MIN, N_MAX = 0.0015, 64, 300
SHINGLE, NEARDUP = 5, 0.10
SEEDS = {'order': 18111, 'start': 18112, 'bootstrap': 18121}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def norm_text(s):
    s = s.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.').lower()
    return re.sub(r'[^0-9a-z]+', ' ', s).strip()


def split_articles(lines):
    """strict boundaries: ' = T = ' with an empty line before and after starts an article."""
    starts = [i for i, l in enumerate(lines) if re.match(r'^ = [^=].* = \n$', l) and (i == 0 or lines[i - 1] == '')
              and i + 1 < len(lines) and lines[i + 1] == '']
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        body = ' '.join(x.strip() for x in lines[s + 1:e] if x.strip())
        nb = norm_text(body)
        out.append({'title': lines[s].strip(' =\n'), 'line_start': s, 'line_end': e, 'nt': norm_text(lines[s].strip(' =\n')).replace(' ', ''),
                    'ch': hashlib.sha256(nb.encode()).hexdigest(), 'body_norm': nb})
    return out


def shingles(nb):
    """word 5-gram shingles as stable 64-bit blake2b hashes (no per-process string hashing)."""
    w = nb.split()
    return {int.from_bytes(hashlib.blake2b(' '.join(w[i:i + SHINGLE]).encode(), digest_size=8).digest(), 'little')
            for i in range(max(0, len(w) - SHINGLE + 1))}


def load_lines(files):
    import pyarrow.parquet as pq
    lines = []
    for f in files:
        lines += pq.read_table(f)['text'].to_pylist()
    return lines


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def forward5(toks_np, fm, dq, ranges, mode, base=None, record_p=False, log=print, label='', chunk=CHUNK):
    """run_olmoe_dynamic_route.forward layer arithmetic with the route of `mode` ('fp32' = own router baseline)."""
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
    extra = {'moe_norm': np.zeros((nW, NL), np.float32)}
    if mode == 'Lmass':
        extra.update(scale=np.zeros((nW, NL, SEQ), np.float32), m=np.zeros((nW, NL, SEQ), np.float32), m0=np.zeros((nW, NL, SEQ), np.float32),
                     w_raw=np.zeros((nW, NL, SEQ, K), np.float32))
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
            if mode in ('L00', 'L01', 'Lmass'):
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
                    logits = F.linear(z, wq['router'])
                    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    if mode in ('L11', 'fp32'):
                        val, idx = torch.topk(probs, K, dim=-1)
                        rw = val.to(logits.dtype)
                    elif mode == 'L01':
                        idx = S0l[rows]
                        rw = probs.gather(-1, idx).to(logits.dtype)
                    elif mode == 'Lmass':
                        idx = S0l[rows]
                        wcur = probs.gather(-1, idx).to(torch.float32)
                        m = torch.sum(wcur, dim=-1, keepdim=True)
                        m0 = torch.sum(p0l[rows].gather(-1, idx), dim=-1, keepdim=True)
                        if not (bool(torch.isfinite(m).all()) and bool((m > 0).all()) and bool(torch.isfinite(m0).all())):
                            raise RuntimeError(f'Lmass: zero / non-finite sum at layer {l}')
                        scale = m0 / m
                        rw = wcur * scale
                        if not (bool(torch.isfinite(scale).all()) and bool(torch.isfinite(rw).all())):
                            raise RuntimeError(f'Lmass: non-finite scale / weights at layer {l}')
                        extra['scale'][a:b, l] = scale.view(B, SEQ).cpu().numpy()
                        extra['m'][a:b, l] = m.view(B, SEQ).cpu().numpy()
                        extra['m0'][a:b, l] = m0.view(B, SEQ).cpu().numpy()
                        extra['w_raw'][a:b, l] = wcur.view(B, SEQ, K).cpu().numpy()
                    else:
                        raise ValueError(mode)
                    if record_p:
                        P[a:b, l] = probs.view(B, SEQ, NE).cpu().numpy()
                rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                moe = oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
                extra['moe_norm'][a:b, l] = moe[:, :TP].norm(dim=-1).mean(-1).cpu().numpy()
                H[a] = x + moe
            del wq, w
            if mode in ('L00', 'L01', 'Lmass'):
                del S0l, p0l
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'p': P, 'extra': extra, 'seconds': time.time() - t0, 'audit': audit,
            'chunks': chunks, 'finite_ce': bool(np.isfinite(ce).all()), 'finite_weights': bool(np.isfinite(rec['weights']).all())}


def eq(a, b, n=None):
    n = n or len(a['ce'])
    return {'ce_max_abs': float(np.max(np.abs(np.array(a['ce'][:n]) - np.array(b['ce'][:n])))),
            'hidden_bitwise_windows': int(sum(x == y for x, y in zip(a['hidden_sha'][:n], b['hidden_sha'][:n]))), 'windows': n}


def bitwise(e):
    return e['ce_max_abs'] == 0 and e['hidden_bitwise_windows'] == e['windows']


def baseline(toks, dq, ranges, log, label):
    f = forward5(toks, None, dq, ranges, 'fp32', record_p=True, log=log, label=label)
    return f, {'S0': f['routes']['ids'], 'w0': f['routes']['weights'], 'p0': f['p']}


def semantics(runs, base):
    S0, p0, w0 = base['S0'], base['p0'], base['w0']
    w0g = np.take_along_axis(p0, S0.astype(np.int64), axis=-1)
    lm = runs['Lmass']
    W = lm['routes']['weights'].astype(np.float64)
    m0_from_p0 = w0g.astype(np.float64).sum(-1)
    raw = lm['extra']['w_raw']
    ratio = W / np.where(raw != 0, raw.astype(np.float64), np.nan)
    res = {'L00_ids_equal_S0': bool(np.array_equal(runs['L00']['routes']['ids'], S0)),
           'L00_weights_equal_w0': bool(np.array_equal(runs['L00']['routes']['weights'], w0)),
           'w0_equal_p0_at_S0': bool(np.array_equal(w0, w0g)),
           'L01_ids_equal_S0': bool(np.array_equal(runs['L01']['routes']['ids'], S0)),
           'Lmass_ids_equal_S0': bool(np.array_equal(lm['routes']['ids'], S0)),
           'Lmass_recorded_m0_minus_float32_sum_of_w0_max_abs': float(np.max(np.abs(lm['extra']['m0'] - w0g.sum(-1, dtype=np.float32)))),
           'Lmass_sum_minus_m0_max_abs': float(np.max(np.abs(W.sum(-1) - m0_from_p0))),
           'Lmass_sum_minus_m0_max_rel': float(np.max(np.abs(W.sum(-1) - m0_from_p0) / m0_from_p0)),
           'Lmass_ratio_to_unscaled_max_rel_spread': float(np.nanmax((np.nanmax(ratio, -1) - np.nanmin(ratio, -1)) / np.nanmean(ratio, -1))),
           'Lmass_weights_equal_float32_w_raw_times_scale': bool(np.array_equal(lm['routes']['weights'], (raw * lm['extra']['scale'][..., None]).astype(np.float32))),
           'Lmass_m_equal_float32_sum_of_w_raw_max_abs': float(np.max(np.abs(lm['extra']['m'] - raw.sum(-1, dtype=np.float32)))),
           'baseline_sum_equals_1_share': float((np.abs(w0.astype(np.float64).sum(-1) - 1) < 1e-6).mean()),
           'Lmass_scale_min': float(lm['extra']['scale'].min()), 'Lmass_scale_max': float(lm['extra']['scale'].max()),
           'Lmass_sum_equals_1_share': float((np.abs(W.sum(-1) - 1) < 1e-6).mean()),
           'weight_sum_max': {c: float(runs[c]['routes']['weights'].sum(-1).max()) for c in CONDS},
           'ids_in_range_and_distinct': {c: bool(runs[c]['routes']['ids'].min() >= 0 and runs[c]['routes']['ids'].max() < NE
                                                 and (np.diff(np.sort(runs[c]['routes']['ids'], -1), axis=-1) > 0).all()) for c in CONDS}}
    res['pass'] = bool(res['L00_ids_equal_S0'] and res['L00_weights_equal_w0'] and res['w0_equal_p0_at_S0'] and res['L01_ids_equal_S0']
                       and res['Lmass_ids_equal_S0'] and res['Lmass_recorded_m0_minus_float32_sum_of_w0_max_abs'] <= 1e-6 and res['Lmass_sum_minus_m0_max_rel'] < 1e-5
                       and res['Lmass_ratio_to_unscaled_max_rel_spread'] < 1e-5 and res['Lmass_weights_equal_float32_w_raw_times_scale']
                       and res['Lmass_m_equal_float32_sum_of_w_raw_max_abs'] <= 1e-6
                       and all(v < 1.0 for v in res['weight_sum_max'].values()) and all(res['ids_in_range_and_distinct'].values()))
    return res


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    from scipy.stats import norm
    from mp_auto import config_hash
    mp = read(MPT / 'plan.json')
    if sha(CONFIG) != mp['configs']['aggressive']['sha256']:
        raise SystemExit('aggressive config hash differs')
    lines2 = load_lines(WT2)
    wt2 = split_articles(lines2)
    lines = load_lines(WT103)
    w103 = split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    cnt_t, cnt_c = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and cnt_t[a['nt']] == 1 and cnt_c[a['ch']] == 1]
    inv = {'wt2_articles': len(wt2), 'wt103_train_lines': len(lines), 'wt103_train_articles': len(w103),
           'excluded_wt2_title_or_content': sum(a['nt'] in T2 or a['ch'] in C2 for a in w103),
           'excluded_wt103_internal_duplicate_groups': sum(1 for a in w103 if (cnt_t[a['nt']] > 1 or cnt_c[a['ch']] > 1) and not (a['nt'] in T2 or a['ch'] in C2)),
           'eligible_before_length_and_neardup': len(elig)}
    z = float(norm.ppf(0.975))
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [CONFIG, dr.RANGES, MPT / 'plan.json', RD / 'plan.json', RD / 'analysis.json',
                                                         RD / 'results.json', RD / 'tokens.npy', RD / 'baseline_routes.npz'] + WT103 + WT2}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'With FP32 expert ids fixed, does restoring the selected routing-weight sum to the FP32 baseline sum '
                    '(keeping the quantized model own relative proportions) reduce the loss? Diagnostic with FP32 information.',
        'configuration': {'path': str(CONFIG.relative_to(HERE)), 'sha256': sha(CONFIG), 'config_hash': config_hash(read(CONFIG)),
                          'R': mp['configs']['aggressive']['R_frontier'], 'semantics': 'unchanged (3051 searched projections; 21 absent '
                          'projections bf16 weights + FP32 activations; router / attention / norms / head FP32)'},
        'conditions': {
            'baseline': 'complete FP32 dynamic; per layer and token S0 (top-8 ids) and p0 (float32 softmax over 64) stored',
            'L00': 'ids S0, weights w0 = p0[S0]', 'L01': 'ids S0, weights w = p[S0] (p from the condition own hidden state)',
            'Lmass': 'ids S0; w = p[S0] (float32); m = torch.sum(w, -1) and m0 = torch.sum(p0[S0], -1), both float32 in S0 order '
                     '(the baseline top-k order); scale = m0 / m (float32); weights = w * scale (float32). m0 is per token and '
                     'layer, not 1; no renormalization to 1, no logit change, no clipping / shrinkage / epsilon; zero or '
                     'non-finite m, m0, scale or weights stop the run',
            'L11': 'ids top-8(p), weights p[ids] (background reproduction)',
            'shared': 'own complete forward per condition with the aggressive configuration; run_olmoe_dynamic_route.forward '
                      f'arithmetic; chunks of {CHUNK} windows, tail = remainder, identical for all; GPU FP32, TF32 off; 127 '
                      'predicted tokens per window; GPU QDQ behaviour, not native kernels',
            'caveat': 'restoring the sum changes later hidden states, so later p differ from L01: an end-to-end layerwise '
                      'mass-restore intervention, not a local decomposition'},
        'data': {
            'source': 'Salesforce/wikitext wikitext-103-raw-v1 train (two parquet files, sha256 = hub LFS oids). '
                      'WikiText-2 (all splits, 720 articles) is a subset / shares articles and is exhausted for OLMoE.',
            'article_rule': 'strict boundaries (title line with an empty line before and after), as in earlier rounds',
            'dedup': 'remove every WikiText-103 article whose normalized title (lowercase alphanumerics, WikiText @-@ / @,@ / '
                     '@.@ joined) or normalized-body sha256 matches any WikiText-2 article (so every text used by an earlier '
                     'OLMoE experiment is excluded), and every member of a within-WikiText-103 duplicate title / content group',
            'inventory_before_plan': inv,
            'sampling': f'eligible articles in a random order (seed {SEEDS["order"]}); walk the order: tokenize the article text '
                        '(lines from its title line to the next title, joined by blank lines as in olmoe_common.wikitext; '
                        'OLMoE tokenizer, no special tokens); skip if < 512 tokens; skip if >= 10% of its word 5-gram shingles '
                        'occur in one WikiText-2 article or in one already accepted article (near-duplicate); accept otherwise '
                        f'with a uniform start (seed {SEEDS["start"]}) of 4 consecutive 128-token windows, until n articles. '
                        'Skips are recorded. Nothing depends on routes, loss, weight sums or difficulty.',
            'records': 'file, line range, title, normalized-title / content hashes, article token count, window token range '
                       'and char offsets within the article text, token sha256',
            'extrapolation': 'new source: results describe WikiText-103 train articles not in WikiText-2; they are not pooled '
                             'with earlier WikiText-2 confirmation sets'},
        'pilot_and_sample_size': {
            'pilot': f'old data: run_olmoe_route_decomposition windows 0-{PILOT_WINDOWS - 1} (32 WikiText-2 articles, chunk 0 '
                     'of that run): program control (L00, L01, L11, baseline reproduce bitwise) and the article SD s of '
                     'Lmass - L01; never part of the confirmation set',
            'rule': f'n = min({N_MAX}, max({N_MIN}, ceil((1.96 s / {H_TARGET})^2)))'},
        'endpoints': {
            'primary': 'Delta_restore = CE(Lmass) - CE(L01); 95% percentile CI; negative = restoring the sum helps',
            'secondary_family': 'Bonferroni over three formal secondary claims (two-sided 96.67% intervals): Residual = '
                                'CE(Lmass) - CE(L00) (equivalent within +-eps if inside, positive if lower > 0); E_weight = '
                                'CE(L01) - CE(L00) (replication if lower > 0); E_total = CE(L11) - CE(L00) (replication if '
                                'lower > 0; compared with eps too). Unadjusted 95% intervals shown.',
            'eps': f'{EPS} nat/token (earlier rounds); CI containing 0 is not equivalence',
            'aggregation': 'per article CE = summed CE of 4 windows / 508; equal article weights; PPL = exp(mean CE)',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix for every condition, seed {SEEDS["bootstrap"]}',
            'no_shares': 'no "X% from the sum, Y% from proportions" decomposition'},
        'controls': [
            'pilot: baseline S0 / w0 / CE and L00, L01, L11 CE reproduce run_olmoe_route_decomposition windows 0-127 bitwise',
            'Lmass semantics: ids = S0; recorded m0 = float32 sum of p0[S0]; |sum(weights) - m0| reported (max relative < 1e-5); '
            'weights / unscaled w constant per token (relative spread < 1e-5); sums not set to 1; scale range reported',
            'quantization off (control subset = first chunk): L00, L01, Lmass, L11 each == FP32 baseline bitwise (Lmass: '
            'scale = 1 exactly when p = p0); otherwise the difference is measured before interpretation',
            'reruns on the control subset bitwise; replay of every condition\'s recorded ids / weights through '
            'run_olmoe_dynamic_route.forward reproduces it bitwise (all windows)',
            'config completeness, router untouched, hooks on expert projections only, finite values, 127 tokens per window',
            'any failure: fix and record before analysing losses'],
        'interpretation': {
            'helps': 'Delta_restore 95% CI entirely below 0: restoring the FP32 sum helps (FP32-informed diagnostic only)',
            'insufficient': 'Residual Bonferroni-positive after improvement: restoring the sum does not remove the weight-path damage',
            'null': 'Delta_restore CI contains 0: no claim either way; report the improvement range the CI excludes',
            'harms': 'Delta_restore CI entirely above 0: restoring the sum hurts, reported as is',
            'replication': 'E_weight / E_total not replicated on the new source: stated explicitly',
            'scope': 'even if it helps, it does not carry over to full dynamic routing (FP32 ids and FP32 m0 used)'},
        'auxiliary': 'per layer: m0, m, scale distributions; L01 and Lmass weight sums; shift of relative proportions vs w0/m0; '
                     'non-finite / extreme scales and where; mean MoE branch output norm per condition (descriptive; no claim '
                     'that a smaller sum implies a smaller output norm). Descriptive only, no post-hoc subsets.',
        'not_done': 'no BC / search / config / range / quantizer / router change, no first-order term, no pipeline change',
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_route_mass_restore.py', 'run_olmoe_route_decomposition.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE aggressive 配置「固定 expert 選擇、恢復 routing weight 總和」診斷', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 條件（各自完整 forward；p 來自各自 hidden state）', '',
          '- 基準：完整 FP32 dynamic，保存 S0（top-8）與 p0（全部 64 個機率）；w0 = p0[S0]，m0 = sum(w0)（每 token、每層各自的值，不是 1）。',
          '- L00：IDs = S0，weights = w0。', '- L01：IDs = S0，weights = w = p[S0]。',
          '- Lmass：IDs = S0，weights = w × (m0 / m)，m = sum(w)；float32，sum 依 S0 順序；不 normalize 到 1、不改 logits、不 clip、不加 epsilon；零或非有限值即停止。',
          '- L11：完整 dynamic（背景重現）。', '- 恢復總和會改變後續 hidden state：這是逐層恢復總量的端到端干預，不是局部分解。', '',
          '## 資料（新來源）', '',
          f'- WikiText-103 raw train（兩個 parquet，sha256 = hub LFS oid）。嚴格文章邊界：{inv["wt103_train_articles"]} 篇。',
          f'- 排除與 WikiText-2（全部 720 篇）標準化標題或內容 hash 相同者 {inv["excluded_wt2_title_or_content"]} 篇，以及 WikiText-103 內部重複群組 {inv["excluded_wt103_internal_duplicate_groups"]} 篇；剩 {inv["eligible_before_length_and_neardup"]} 篇。',
          f'- 依隨機順序（seed {SEEDS["order"]}）逐篇：單篇 tokenize；<512 token 跳過；與任一 WikiText-2 文章或已選文章的 5-gram shingle 重疊 ≥10% 視為近重複跳過；其餘接受，起點 seed {SEEDS["start"]}，每篇 4 個連續窗口。',
          '- 新來源結果不與先前 WikiText-2 確認集合併。', '',
          '## 樣本數', '', f'- Pilot：上一輪 route-decomposition 窗口 0–127（32 篇舊文章），重現檢查並估 Lmass − L01 的文章級 SD s。',
          f'- n = min({N_MAX}, max({N_MIN}, ceil((1.96 s / {H_TARGET})²)))，於新資料評估前固定。', '',
          '## 終點與統計', '', '- 主要：Δ_restore = CE(Lmass) − CE(L01)，95% CI；負值 = 恢復總和有幫助。',
          '- 次要族（Bonferroni ×3，96.67% 區間）：Residual = Lmass − L00（±0.002 等價或 >0）；E_weight = L01 − L00（重現）；E_total = L11 − L00（重現，另對 ε）。另列 95%。',
          f'- {BOOT} 次 paired article bootstrap（seed {SEEDS["bootstrap"]}）；PPL 由平均 CE 取 exp；不做「總量占 X%、比例占 Y%」的分解。', '',
          '## 控制', ''] + [f'- {c}' for c in p['controls']] + ['', '## 判讀', ''] + [f'- {k}：{v}' for k, v in p['interpretation'].items()] + ['']
    (out / 'plan.md').write_text('\n'.join(md))
    log(f'plan written: {inv}')


def pilot(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'pilot.json').exists():
        raise SystemExit('pilot exists')
    toks = np.load(RD / 'tokens.npy')[:PILOT_WINDOWS]
    old = read(RD / 'results.json')['ce']
    ob = np.load(RD / 'baseline_routes.npz')
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = read(CONFIG), dr.read(dr.RANGES)
    f, base = baseline(toks, dq, ranges, log, 'pilot fp32')
    runs = {c: forward5(toks, fm, dq, ranges, c, base=base, log=log, label=f'pilot {c}') for c in CONDS}
    mx = lambda a, b: float(np.max(np.abs(np.array(a) - np.array(b[:PILOT_WINDOWS]))))
    res = {'program_control': {'baseline_S0_equal_old': bool(np.array_equal(base['S0'], ob['S0'][:PILOT_WINDOWS])),
                               'baseline_w0_equal_old': bool(np.array_equal(base['w0'], ob['w0'][:PILOT_WINDOWS])),
                               'baseline_p0_equal_old': bool(np.array_equal(base['p0'], ob['p0'][:PILOT_WINDOWS])),
                               'baseline_ce_max_abs': mx(f['ce'], old['fp32']),
                               **{f'{c}_ce_max_abs': mx(runs[c]['ce'], old[c]) for c in ('L00', 'L01', 'L11')}},
           'semantics': semantics(runs, base)}
    pc = res['program_control']
    res['pass'] = bool(pc['baseline_S0_equal_old'] and pc['baseline_w0_equal_old'] and pc['baseline_p0_equal_old']
                       and all(v == 0 for k, v in pc.items() if k.endswith('max_abs')) and res['semantics']['pass'])
    art = {c: np.array(runs[c]['ce']).reshape(-1, WPA).sum(1) / (WPA * TP) for c in CONDS}
    d = art['Lmass'] - art['L01']
    s = float(d.std(ddof=1))
    n = int(min(N_MAX, max(N_MIN, math.ceil((1.96 * s / H_TARGET) ** 2))))
    res['pilot_estimates'] = {'Delta_restore_mean': float(d.mean()), 'Delta_restore_article_sd': s,
                              'Residual_mean': float((art['Lmass'] - art['L00']).mean()), 'E_weight_mean': float((art['L01'] - art['L00']).mean())}
    res['sample_size'] = {'s': s, 'n_required': math.ceil((1.96 * s / H_TARGET) ** 2), 'n_articles': n,
                          'expected_95_half_width': 1.96 * s / math.sqrt(n)}
    res['seconds'] = {'baseline': f['seconds'], **{c: runs[c]['seconds'] for c in CONDS}}
    res['environment'] = dr.env_info()
    write(out / 'pilot.json', res)
    log(json.dumps({'pass': res['pass'], 'sample_size': res['sample_size'], 'semantics_pass': res['semantics']['pass']}))


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
    wt2 = split_articles(load_lines(WT2))
    lines = load_lines(WT103)
    w103 = split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    cnt_t, cnt_c = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and cnt_t[a['nt']] == 1 and cnt_c[a['ch']] == 1]
    if len(elig) != pl['data']['inventory_before_plan']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    inv = defaultdict(set)                                       # shingle -> WikiText-2 article indices
    for i, a in enumerate(wt2):
        for s_ in shingles(a['body_norm']):
            inv[s_].add(i)
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    acc, skips, acc_sh, toks, segs = [], Counter(), [], [], []
    neardup_examples = []
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc['input_ids'], enc['offset_mapping']
        if len(ids) < WPA * SEQ:
            skips['short'] += 1
            continue
        sh = shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cs = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if c2 >= NEARDUP or cs >= NEARDUP:
            skips['near_duplicate'] += 1
            neardup_examples.append({'title': a['title'], 'max_share_wt2': c2, 'max_share_accepted': cs})
            continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']],
                     'title': a['title'], 'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                     'window_tokens': [st, st + WPA * SEQ], 'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                     'max_shingle_share_wt2': c2, 'max_shingle_share_accepted': cs, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
                     'windows': [WPA * len(segs) + i for i in range(WPA)]})
        if len(segs) == n:
            break
    if len(segs) < n:
        raise SystemExit('not enough articles')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'), 'pilot_sha256': sha(out / 'pilot.json'),
                                     'n_articles': n, 'eligible': len(elig), 'visited': int(sum(skips.values()) + n), 'skips': dict(skips),
                                     'near_duplicate_examples': neardup_examples[:20], 'segments_sha256': sha(out / 'segments.json'),
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'files_sha256': {f.name: sha(f) for f in WT103}})
    log(f'data frozen: {n} articles; skips {dict(skips)}')


def run(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'results.json').exists():
        raise SystemExit('results exist')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256'] or sha(CONFIG) != read(out / 'plan.json')['configuration']['sha256']:
        raise SystemExit('tokens or config changed')
    toks = np.load(out / 'tokens.npy')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = read(CONFIG), dr.read(dr.RANGES)
    cc = dr.config_checks(fm, dq, ranges)
    f, base = baseline(toks, dq, ranges, log, 'baseline')
    np.savez(out / 'baseline_routes.npz', S0=base['S0'], w0=base['w0'], p0=base['p0'], tokens=toks, moe_norm=f['extra']['moe_norm'])
    runs = {c: forward5(toks, fm, dq, ranges, c, base=base, log=log, label=c) for c in CONDS}
    (out / 'routes').mkdir(exist_ok=True)
    for c in CONDS:
        np.savez(out / f'routes/{c}.npz', ids=runs[c]['routes']['ids'], weights=runs[c]['routes']['weights'], **runs[c]['extra'])
    ctl = {'semantics': semantics(runs, base), 'config_checks': cc}
    sub, sb = toks[:CTRL], {k: v[:CTRL] for k, v in base.items()}
    ctl['quant_off'] = {c: eq(forward5(sub, None, dq, ranges, c, base=sb, log=log, label=f'off {c}'), f, CTRL) for c in CONDS}
    ctl['rerun'] = {}
    for c in CONDS:
        r2 = forward5(sub, fm, dq, ranges, c, base=sb, log=log, label=f'rerun {c}')
        e = eq(r2, runs[c], CTRL)
        e['routes_equal'] = bool(np.array_equal(r2['routes']['ids'], runs[c]['routes']['ids'][:CTRL])
                                 and np.array_equal(r2['routes']['weights'], runs[c]['routes']['weights'][:CTRL]))
        ctl['rerun'][c] = e
    ctl['replay'], rep = {}, {}
    for c in CONDS:
        rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs[c]['routes']), log=log, label=f'replay {c}')
        ctl['replay'][c] = eq(rp, runs[c]); rep[c] = rp['seconds']
    ctl['finite'] = all(runs[c]['finite_ce'] and runs[c]['finite_weights'] for c in CONDS) and f['finite_ce']
    ctl['audit'] = {c: runs[c]['audit'] for c in CONDS}
    ctl['pass'] = bool(ctl['semantics']['pass'] and cc['pass'] and ctl['finite'] and all(bitwise(v) for v in ctl['quant_off'].values())
                       and all(bitwise(v) and v['routes_equal'] for v in ctl['rerun'].values()) and all(bitwise(v) for v in ctl['replay'].values())
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'results.json', {'ce': {'fp32': f['ce'], **{c: runs[c]['ce'] for c in CONDS}}, 'controls': ctl,
                                 'seconds': {'baseline': f['seconds'], **{c: runs[c]['seconds'] for c in CONDS}, 'replay': rep},
                                 'environment': dr.env_info()})
    log(json.dumps({'controls_pass': ctl['pass']}))


def analyze(out, log):
    t0 = time.time()
    rs = read(out / 'results.json')
    if not rs['controls']['pass']:
        raise SystemExit('controls failed: do not analyse losses')
    segs = read(out / 'segments.json'); nA = len(segs)
    art = {c: np.array(v).reshape(nA, WPA).sum(1) / (WPA * TP) for c, v in rs['ce'].items()}
    E = {'Delta_restore': art['Lmass'] - art['L01'], 'Residual': art['Lmass'] - art['L00'], 'E_weight': art['L01'] - art['L00'],
         'E_total': art['L11'] - art['L00'], 'L11_minus_Lmass': art['L11'] - art['Lmass'],
         **{f'D_{c}': art[c] - art['fp32'] for c in CONDS}}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, nA, (BOOT, nA))
    lv = 100 * ALPHA / 6

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci_bonf3': np.percentile(d, [lv, 100 - lv]).tolist(),
                'article_sd': float(x.std(ddof=1)), 'articles_positive': int((x > 0).sum())}
    res = {'n_articles': nA, 'contrasts': {k: st(v) for k, v in E.items()}}
    c = res['contrasts']
    lo, hi = c['Delta_restore']['ci95']
    res['primary'] = 'helps' if hi < 0 else 'harms' if lo > 0 else 'null'
    res['secondary'] = {'Residual_equivalent_within_eps': -EPS <= c['Residual']['ci_bonf3'][0] and c['Residual']['ci_bonf3'][1] <= EPS,
                        'Residual_positive': c['Residual']['ci_bonf3'][0] > 0, 'Residual_negative': c['Residual']['ci_bonf3'][1] < 0,
                        'E_weight_replicated_positive': c['E_weight']['ci_bonf3'][0] > 0, 'E_total_replicated_positive': c['E_total']['ci_bonf3'][0] > 0,
                        'E_total_exceeds_eps': c['E_total']['ci_bonf3'][0] > EPS}
    tokens = nA * WPA * TP
    res['ce_ppl'] = {k: {'ce': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for k, v in rs['ce'].items()}
    b = np.load(out / 'baseline_routes.npz')
    w0 = b['w0'][:, :, :TP].astype(np.float64)
    m0 = w0.sum(-1)
    q0 = w0 / m0[..., None]
    lay = []
    R = {c_: np.load(out / f'routes/{c_}.npz') for c_ in CONDS}
    sc = R['Lmass']['scale'][:, :, :TP]
    for l in range(NL):
        row = {'layer': l, 'm0_mean': float(m0[:, l].mean()), 'm0_p05': float(np.percentile(m0[:, l], 5)), 'm0_p95': float(np.percentile(m0[:, l], 95))}
        for c_ in ('L01', 'Lmass', 'L11'):
            wv = R[c_]['weights'][:, l, :TP].astype(np.float64)
            row[f'{c_}_sum_mean'] = float(wv.sum(-1).mean())
            row[f'{c_}_sum_minus_m0_mean'] = float((wv.sum(-1) - m0[:, l]).mean())
            if c_ != 'L11':
                row[f'{c_}_proportion_l1_vs_w0'] = float(np.abs(wv / wv.sum(-1, keepdims=True) - q0[:, l]).sum(-1).mean())
        row.update({'scale_mean': float(sc[:, l].mean()), 'scale_p01': float(np.percentile(sc[:, l], 1)), 'scale_p99': float(np.percentile(sc[:, l], 99)),
                    'scale_min': float(sc[:, l].min()), 'scale_max': float(sc[:, l].max()),
                    'Lmass_m_mean': float(R['Lmass']['m'][:, l, :TP].mean()),
                    **{f'moe_norm_{c_}': float(R[c_]['moe_norm'][:, l].mean()) for c_ in CONDS}, 'moe_norm_fp32': float(b['moe_norm'][:, l].mean())})
        lay.append(row)
    with open(out / 'per_layer.csv', 'w') as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(lay[0])); w_.writeheader(); w_.writerows(lay)
    res['scale'] = {'min': float(sc.min()), 'max': float(sc.max()), 'mean': float(sc.mean()), 'share_above_1': float((sc > 1).mean()),
                    'p001': float(np.percentile(sc, 0.1)), 'p999': float(np.percentile(sc, 99.9)), 'nonfinite': int((~np.isfinite(sc)).sum())}
    res['semantics'] = rs['controls']['semantics']
    with open(out / 'per_article.csv', 'w') as fh:
        rows = [{'index': s['index'], 'title': s['title'], **{f'ce_{k}': float(v[i]) for k, v in art.items()},
                 **{k: float(E[k][i]) for k in ('Delta_restore', 'Residual', 'E_weight', 'E_total')}} for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    res['controls'] = rs['controls']
    res['timing'] = {'run': rs['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    write(out / 'analysis.json', res)
    log(json.dumps({k: [c[k]['mean'], c[k]['ci95']] for k in ('Delta_restore', 'Residual', 'E_weight', 'E_total')}) + ' ' + res['primary'])


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
