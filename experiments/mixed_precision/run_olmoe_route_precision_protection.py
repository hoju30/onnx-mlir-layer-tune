#!/usr/bin/env python3
"""OLMoE aggressive BC configuration: per-layer expert-weight precision protection (int<8 -> int8 in one layer) chosen
on a selection set by a route-proportion heuristic, versus the BC configuration with the same size / BOPs budgets, on
an isolated final-test set. Normal dynamic routing everywhere; FP32 is used only as the baseline and to measure the
router-proportion drift offline. Pre-registered in plan.json / plan.md.

Stages (one --out-dir):
  plan            16 protected configs, resource table, data / selection / statistics rules, hashes
  data_selection  32 new WikiText-103 articles (deduplicated as before; every earlier OLMoE text excluded)
  control_prev    program control on old windows (run_olmoe_route_dynamic_prop windows 0-127)
  selection       FP32 baseline, aggressive, 16 protected configs on the selection set; rule -> frozen choice
  bc_solve        MILP: min original BC s.t. size <= M_target, BOPs <= P_target of the chosen config
  selection_bc    the BC config on the selection set (variance only) -> final sample size
  data_final      next n accepted articles of the same order (disjoint from selection)
  final           FP32, aggressive, chosen, BC on the final set + controls
  analyze         endpoints, routing description, figures
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
MPT = HERE / 'runs/olmoe_dynamic_multipoint_20261018'
RDP = HERE / 'runs/olmoe_route_dynamic_prop_20261018'
MRD = HERE / 'runs/olmoe_route_mass_restore_20261018'
MP = HERE / 'runs/olmoe_multipoint_20261015'
BASE_CFG = MPT / 'configs/aggressive.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
CHUNK, CTRL = 128, 128
N_SEL, N_MIN, N_MAX, H_TARGET = 32, 64, 300, 0.002
EPS, ALPHA, BOOT = 0.002, 0.05, 10000
LOWER_INT = ('int4', 'int5', 'int6', 'int7')
SEEDS = {'order': 18191, 'start': 18192, 'bootstrap': 18201}
TIME_LIMIT = 300.0


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def problem():
    from run_olmoe_multipoint import problem as mp_problem
    table, _, names, nodes, elems, _, flops = mp_problem(MP)
    bc = lambda n, w, a: table['B'][n][w] + table['C'][n][a]
    return table, names, nodes, elems, flops, bc


def resources(fm, names, elems, flops):
    from run_olmoe_multipoint import resources as mp_res
    return mp_res(fm, names, elems, flops)


def bc_total(fm, names, bc):
    return float(sum(bc(n, fm[n]['w'], fm[n]['a']) for n in names))


# --------------------------------------------------------------------------- #
def forward_measure(toks_np, fm, dq, ranges, S0=None, log=print, label='', chunk=CHUNK):
    """run_olmoe_dynamic_route.forward / route_mass_restore.forward5 'L11' arithmetic (the model routes itself);
    additionally records p[S0] (the own router probabilities at the FP32 baseline set) for the offline drift metric.
    The recorded quantities never feed back into the computation."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_olmoe_search_final import quantized_layer
    cfg, ck = oc.config(), oc.Checkpoint()
    eps, Hd = cfg.rms_norm_eps, cfg.hidden_size
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    nW = len(toks_np)
    chunks = [(a, min(nW, a + chunk)) for a in range(0, nW, chunk)]
    rec = {'ids': np.zeros((nW, NL, SEQ, K), np.int16), 'weights': np.zeros((nW, NL, SEQ, K), np.float32)}
    pS0 = np.zeros((nW, NL, SEQ, K), np.float32) if S0 is not None else None
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
            if S0 is not None:
                S0l = torch.from_numpy(S0[:, l].reshape(-1, K).astype(np.int64)).to('cuda')
            for a, b in chunks:
                B = b - a
                h = H[a]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], eps).reshape(-1, Hd)
                logits = F.linear(z, wq['router'])
                probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                val, idx = torch.topk(probs, K, dim=-1)
                rw = val.to(logits.dtype)
                if S0 is not None:
                    pS0[a:b, l] = probs.gather(-1, S0l[a * SEQ:b * SEQ]).view(B, SEQ, K).cpu().numpy()
                rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                H[a] = x + oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
            del wq, w
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'pS0': pS0, 'seconds': time.time() - t0, 'audit': audit, 'chunks': chunks,
            'finite_ce': bool(np.isfinite(ce).all()), 'finite_weights': bool(np.isfinite(rec['weights']).all())}


def drift(pS0, w0):
    """per (window, layer, position) L1 between the own and FP32 relative proportions on S0 (float64)."""
    r0 = w0.astype(np.float64) / w0.astype(np.float64).sum(-1, keepdims=True)
    rq = pS0.astype(np.float64) / pS0.astype(np.float64).sum(-1, keepdims=True)
    return np.abs(rq - r0).sum(-1)                                     # [N, L, T]


def route_summary(r, base):
    G_tok = drift(r['pS0'], base['w0'])[:, :, :TP]
    sets = lambda a: np.sort(a, -1)
    ids = r['routes']['ids'][:, :, :TP]
    S0 = base['S0'][:, :, :TP]
    return {'G': float(G_tok[:, 1:].mean()), 'G_per_layer': G_tok.mean((0, 2)).tolist(),
            'set_change_vs_S0': float((sets(ids) != sets(S0)).any(-1)[:, 1:].mean()),
            'set_change_per_layer': (sets(ids) != sets(S0)).any(-1).mean((0, 2)).tolist(),
            'weight_sum_minus_m0': float((r['routes']['weights'][:, 1:, :TP].astype(np.float64).sum(-1) - base['w0'][:, 1:, :TP].astype(np.float64).sum(-1)).mean())}


def per_article(ce, n):
    return np.array(ce).reshape(n, WPA).sum(1) / (WPA * TP)


# --------------------------------------------------------------------------- #
def build_candidates(out):
    from mp_auto import config_hash
    from run_native_bc_pipeline import GRID11
    table, names, nodes, elems, flops, bc = problem()
    base = read(BASE_CFG)
    rb = resources(base, names, elems, flops)
    (out / 'configs').mkdir(exist_ok=True)
    write(out / 'configs/aggressive.json', base)
    rows = []
    for l in range(NL):
        fm = {k: dict(v) for k, v in base.items()}
        changes = Counter()
        for k, v in fm.items():
            if k.startswith(f'L{l}.') and v['w'] in LOWER_INT:
                changes[f'{v["w"]}->int8'] += 1
                v['w'] = 'int8'
        write(out / f'configs/protect_L{l:02d}.json', fm)
        r = resources(fm, names, elems, flops)
        in_space = all(fm[n]['w'] in GRID11 and fm[n]['a'] in GRID11 and math.isfinite(bc(n, fm[n]['w'], fm[n]['a'])) for n in names)
        rows.append({'layer': l, 'config': f'configs/protect_L{l:02d}.json', 'sha256': sha(out / f'configs/protect_L{l:02d}.json'),
                     'config_hash': config_hash(fm), 'nodes_modified': int(sum(changes.values())), 'changes': dict(changes),
                     'unchanged_in_layer_formats': dict(Counter(base[k]['w'] for k in fm if k.startswith(f'L{l}.') and base[k]['w'] not in LOWER_INT)),
                     'M_bytes': r['M_bytes'], 'BOPs': r['BOPs'], 'M_rel': r['M_rel'], 'B_rel': r['B_rel'], 'R': r['R'],
                     'dM_rel': r['M_rel'] - rb['M_rel'], 'dB_rel': r['B_rel'] - rb['B_rel'], 'dR': r['R'] - rb['R'],
                     'bc_objective': bc_total(fm, names, bc), 'in_candidate_space': bool(in_space),
                     'only_layer_weights_changed': all((fm[k] == base[k]) or (k.startswith(f'L{l}.') and fm[k]['a'] == base[k]['a']) for k in fm)})
    return base, rb, rows, config_hash(base), bc_total(base, names, bc)


def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    mp = read(MPT / 'plan.json')
    if sha(BASE_CFG) != mp['configs']['aggressive']['sha256']:
        raise SystemExit('aggressive config hash differs')
    base, rb, rows, bhash, bobj = build_candidates(out)
    if abs(rb['R'] - mp['configs']['aggressive']['R_frontier']) > 1e-9:
        raise SystemExit('base resources differ from the frontier record')
    # data inventory (counting only)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    used = {s['content_sha256'] for f in (MRD / 'segments.json', RDP / 'segments.json') for s in read(f)}
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [BASE_CFG, dr.RANGES, MP / 'table.json', MP / 'table_meta.json', MPT / 'plan.json',
                                                         MRD / 'segments.json', RDP / 'segments.json', RDP / 'tokens.npy', RDP / 'results.json',
                                                         RDP / 'baseline_routes.npz', RDP / 'routes/L11.npz'] + mr.WT103 + mr.WT2}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'Can raising a few upstream expert weights to int8 reduce the router-proportion drift under normal dynamic '
                    'routing, and does a route-chosen protection beat the original-BC configuration with the same size / BOPs?',
        'base': {'path': str(BASE_CFG.relative_to(HERE)), 'sha256': sha(BASE_CFG), 'config_hash': bhash, 'R': rb['R'], 'M_rel': rb['M_rel'],
                 'B_rel': rb['B_rel'], 'bc_objective': bobj},
        'candidates': {'rule': 'copy l (l = 0..15): every L{l}.E*.{gate,up,down}_proj in the config whose weight is int4-int7 becomes '
                               'int8; int8 / bf16 / other formats, every activation format, ranges, other layers and the 21 absent '
                               'projections unchanged; non-int formats are not ordered by bits',
                       'resources': 'run_olmoe_multipoint.resources (calibration routed-row BOPs, int8 references; fixed cost '
                                    'definition, not re-estimated on new text); R = max(M/M8, B/B8)',
                       'table': rows,
                       'layer_15_note': 'evaluated, but it has no later router: improves ordinary damage without moving earlier '
                                        'routes; not a causal control'},
        'inference': 'normal dynamic routing for every quality number (forward_measure = run_olmoe_dynamic_route.forward '
                     'arithmetic; the own router chooses ids and weights); FP32 only as the baseline and for the offline drift '
                     'metric (p[S0] is recorded, never fed back); router / quantizer / ranges unchanged; GPU FP32, TF32 off, chunks '
                     'of 128 windows (tail = remainder), 127 predicted tokens per window; GPU QDQ behaviour, not native kernels',
        'data': {'source': 'Salesforce/wikitext wikitext-103-raw-v1 train (sha256 checked), strict boundaries, WikiText-2 title / '
                           'content matches and within-WT103 duplicate groups removed, every earlier OLMoE article excluded (WT2 '
                           'entirely; WT103: run_olmoe_route_mass_restore 103 + run_olmoe_route_dynamic_prop 242)',
                 'eligible_before_length_and_neardup': len(elig),
                 'order': f'one random order of the eligible articles (seed {SEEDS["order"]}); walk it: tokenize alone, skip < 512 '
                          'tokens, skip near-duplicates (>= 10% of word 5-gram shingles in one WikiText-2 article, one earlier '
                          'OLMoE WT103 article or one already accepted article); uniform start (seed '
                          f'{SEEDS["start"]}), 4 consecutive 128-token windows',
                 'selection': f'the first {N_SEL} accepted articles', 'final': 'the next n accepted articles (n fixed after '
                 'selection_bc), whole-article disjoint from selection', 'no_choice': 'nothing depends on loss, routes or difficulty'},
        'selection': {
            'evaluated': 'FP32 baseline, aggressive, 16 protected configs (dynamic route)',
            'G': 'per token (positions 0-126) and router layer l = 1..15: S0 = FP32 top-8 set; r0 = p0[S0] / sum, rq = pq[S0] / sum '
                 '(pq = the config own router probabilities at its own hidden state); G = mean of ||rq - r0||_1 over layers 1-15 '
                 'and all positions (equal weights, identical range for every config); per-layer values reported. Diagnostic, '
                 'not a validated loss proxy.',
            'rule': 'eligible: G < G_aggressive, selection CE < CE_aggressive, dR > 0; choose the max of (G_aggr - G) / dR; ties: '
                    'lower selection CE, lower R, lower layer index. No eligible config: report it and stop (no other rule, no '
                    'other protection type). Selection numbers are exploratory; the choice is frozen with its hash.',
            'check': 'protecting layer l cannot change routers at layers <= l: their p[S0] must equal the aggressive run bitwise'},
        'bc_control': {
            'problem': 'original BC cost B[n][w] + C[n][a] (run_olmoe_multipoint table), GRID11 x GRID11, exact dominance pruning '
                       '(run_olmoe_search_final.olmoe_problem), mp_milp.milp_select (integrated corrected version), HiGHS, time '
                       f'limit {TIME_LIMIT:.0f} s',
            'constraints': 'size <= M_target and BOPs <= P_target = the chosen config own bytes and BOPs (both components, not '
                           'max-normalized R)',
            'record': 'seconds, status, gap, actual resources, objective, hash, budget use; not chosen by selection loss; a status '
                      'other than proven optimal is labelled a feasible incumbent',
            'comparability': 'the chosen config must lie in the same candidate space (all formats in GRID11, finite BC); its BC '
                             'objective is compared with the MILP objective'},
        'final': {
            'configs': 'aggressive, the chosen route-focused config, the same-budget BC config (frozen); FP32 baseline',
            'sample_size': f'n = min({N_MAX}, eligible, max({N_MIN}, ceil((1.96 s / {H_TARGET})^2))), s = selection article SD of '
                           'CE(chosen) - CE(BC); the selection set took part in the choice, so s may be optimistic',
            'primary': 'Delta_primary = CE(route-focused) - CE(same-budget BC), 95% percentile CI; negative = route-focused better',
            'secondary': 'Bonferroni over two (97.5% intervals): route-focused - aggressive; BC - aggressive. Descriptive: damage '
                         'and PPL vs FP32, G, set change, weight sums',
            'bootstrap': f'{BOOT} paired article bootstraps, one index matrix, seed {SEEDS["bootstrap"]}; article = 4 windows / 508 tokens'},
        'controls': ['old windows (run_olmoe_route_dynamic_prop 0-127): FP32 baseline (S0, w0, p0, CE) and aggressive dynamic CE / '
                     'ids / weights reproduce that run bitwise',
                     'candidates differ from aggressive only in layer-l weight formats (int<8 -> int8); activations, ranges, router '
                     'untouched; config completeness for every config',
                     'quantization-off forward_measure == FP32 baseline bitwise (first chunk); reruns of the chosen and BC configs on '
                     'the first chunk bitwise; replay of their recorded ids / weights through run_olmoe_dynamic_route.forward bitwise',
                     'routers at layers <= l unchanged for protect_l (bitwise p[S0]); recorded routes differ from S0 (dynamic, no FP32 '
                     'route substitution); finite values; resource recomputation',
                     'failure: fix and record before interpretation'],
        'interpretation': {
            'A': 'route-focused better than same-budget BC (CI < 0) and better than aggressive: worth further study; improvement '
                 'not attributed to routing alone',
            'B': 'better than aggressive only: more precision helps, no support for the route-oriented choice',
            'C': 'G improves but task loss does not: G is not a sufficient selection criterion',
            'D': 'no help from protection: this protection rule is not supported (not extended to activations or other fixes)',
            'E': 'CI spans 0: uncertainty reported, no equivalence claim, no switching candidates',
            'no_change': 'even if successful, BC is not changed automatically'},
        'not_done': 'no BC formula / router / quantizer / range change, no first-order term, no FP32 feedback, no pipeline integration',
        'seeds': SEEDS, 'inputs': inputs,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_route_precision_protection.py', 'run_olmoe_route_mass_restore.py', 'mp_milp.py',
                                            'mp_auto.py', 'run_olmoe_multipoint.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE aggressive 配置的逐層權重精度保護與同資源 BC 對照', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 候選', '', '- 16 份：第 l 份把第 l 層所有 expert 投影中 int4–int7 的權重改為 int8；其餘（int8、bf16、activation、range、其他層、21 個缺席投影）不變。',
          '- 資源沿用固定成本定義（calibration routed rows）；R = max(M/M8, B/B8)。', '',
          '| 層 | 修改節點 | ΔM_rel | ΔB_rel | ΔR | R |', '|---:|---:|---:|---:|---:|---:|'] + \
         [f'| {r["layer"]} | {r["nodes_modified"]} | {r["dM_rel"]:+.5f} | {r["dB_rel"]:+.5f} | {r["dR"]:+.5f} | {r["R"]:.5f} |' for r in rows] + ['',
          '## 資料', '', f'- 去重後 WikiText-103 train，排除 WikiText-2 與先前全部 OLMoE 文章，可用 {len(elig)} 篇；單一隨機順序（seed {SEEDS["order"]}），前 {N_SEL} 篇接受者為 selection，其後 n 篇為 final（整篇隔離）。', '',
          '## Selection', '', '- 評估 FP32、aggressive、16 個保護配置（正常 dynamic route）。',
          '- G：layer 1–15、位置 0–126，S0 上 ||pq[S0]/Σ − p0[S0]/Σ||₁ 等權平均；診斷指標，非 loss proxy。',
          '- 規則：G < G_aggr、CE < CE_aggr、ΔR > 0 者中取 (G_aggr − G)/ΔR 最大；同分依 CE、R、層號。無合格者則如實報告並停止。', '',
          '## BC 同資源對照', '', '- min 原 BC，size ≤ M_target 且 BOPs ≤ P_target（兩個分量都限制）；修正後 mp_milp，時限 300 s；非證明最優則標示 incumbent。', '',
          '## Final', '', f'- 凍結 aggressive、route-focused、同資源 BC；n = min({N_MAX}, 可用, max({N_MIN}, ceil((1.96 s／{H_TARGET})²)))，s 來自 selection（可能偏樂觀）。',
          '- 主要：Δ_primary = CE(route-focused) − CE(BC)，95% CI；次要（Bonferroni ×2）：route-focused − aggressive、BC − aggressive。', '',
          '## 判讀', ''] + [f'- {k}：{v}' for k, v in p['interpretation'].items()] + ['', '## 控制', ''] + [f'- {c}' for c in p['controls']] + ['']
    (out / 'plan.md').write_text('\n'.join(md))
    log(f'plan written; eligible {len(elig)}; candidates dR ' + ' '.join(f'{r["dR"]:.4f}' for r in rows))


# --------------------------------------------------------------------------- #
def draw(out, start_index, n, tag, exclude_hashes):
    """walk the frozen order; return the accepted articles with positions >= start_index among accepted ones."""
    import olmoe_common as oc
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    prev_used = {s['content_sha256'] for f in (MRD / 'segments.json', RDP / 'segments.json') for s in read(f)}
    base_elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev_arts = [a for a in base_elig if a['ch'] in prev_used]
    if len(prev_arts) != len(prev_used):
        raise SystemExit('previous articles not all found')
    elig = [a for a in base_elig if a['ch'] not in prev_used]
    if len(elig) != read(out / 'plan.json')['data']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(a['body_norm']) for a in prev_arts]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    acc_sh, segs, toks, skips, accepted = [], [], [], Counter(), 0
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
            continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        acc_sh.append(sh)
        if accepted >= start_index:
            w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
            if a['ch'] in exclude_hashes:
                raise SystemExit('overlap with an excluded article')
            toks.append(w)
            segs.append({'index': len(segs), 'set': tag, 'order_rank': accepted, 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']],
                         'title': a['title'], 'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                         'window_tokens': [st, st + WPA * SEQ], 'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                         'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca}, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
                         'windows': [WPA * len(segs) + i for i in range(WPA)]})
        accepted += 1
        if len(segs) == n:
            break
    if len(segs) < n:
        raise SystemExit('not enough articles')
    return segs, np.concatenate(toks), dict(skips)


def data_selection(out, log):
    if (out / 'selection_segments.json').exists():
        raise SystemExit('exists')
    segs, toks, skips = draw(out, 0, N_SEL, 'selection', set())
    np.save(out / 'selection_tokens.npy', toks)
    write(out / 'selection_segments.json', segs)
    write(out / 'selection_data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'), 'n': N_SEL, 'skips': skips,
                                               'tokens_sha256': sha(out / 'selection_tokens.npy'), 'segments_sha256': sha(out / 'selection_segments.json')})
    log(f'selection data frozen: {N_SEL} articles, skips {skips}')


def control_prev(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'control_prev.json').exists():
        raise SystemExit('exists')
    toks = np.load(RDP / 'tokens.npy')[:CTRL]
    old, ob, ol = read(RDP / 'results.json')['ce'], np.load(RDP / 'baseline_routes.npz'), np.load(RDP / 'routes/L11.npz')
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    fm, ranges = read(BASE_CFG), dr.read(dr.RANGES)
    f, base = mr.baseline(toks, dq, ranges, log, 'prev fp32')
    r = forward_measure(toks, fm, dq, ranges, base['S0'], log=log, label='prev aggressive')
    mx = lambda a, b: float(np.max(np.abs(np.array(a) - np.array(b[:CTRL]))))
    res = {'baseline_S0': bool(np.array_equal(base['S0'], ob['S0'][:CTRL])), 'baseline_w0': bool(np.array_equal(base['w0'], ob['w0'][:CTRL])),
           'baseline_p0': bool(np.array_equal(base['p0'], ob['p0'][:CTRL])), 'baseline_ce_max_abs': mx(f['ce'], old['fp32']),
           'aggressive_ce_max_abs_vs_L11': mx(r['ce'], old['L11']), 'aggressive_ids_equal': bool(np.array_equal(r['routes']['ids'], ol['ids'][:CTRL])),
           'aggressive_weights_equal': bool(np.array_equal(r['routes']['weights'], ol['weights'][:CTRL])),
           'pS0_equal_p0_at_S0_for_fp32_check': None}
    off = forward_measure(toks, None, dq, ranges, base['S0'], log=log, label='prev quant off')
    res['quant_off'] = mr.eq(off, f, CTRL)
    res['quant_off_pS0_equals_w0'] = bool(np.array_equal(off['pS0'], base['w0']))
    res['pass'] = bool(res['baseline_S0'] and res['baseline_w0'] and res['baseline_p0'] and res['baseline_ce_max_abs'] == 0
                       and res['aggressive_ce_max_abs_vs_L11'] == 0 and res['aggressive_ids_equal'] and res['aggressive_weights_equal']
                       and mr.bitwise(res['quant_off']) and res['quant_off_pS0_equals_w0'])
    write(out / 'control_prev.json', res)
    log(json.dumps({'pass': res['pass']}))


def selection(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'selection.json').exists():
        raise SystemExit('exists')
    if not read(out / 'control_prev.json')['pass']:
        raise SystemExit('program control failed')
    toks = np.load(out / 'selection_tokens.npy')
    if sha(out / 'selection_tokens.npy') != read(out / 'selection_data_freeze.json')['tokens_sha256']:
        raise SystemExit('tokens changed')
    pl = read(out / 'plan.json')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = dr.read(dr.RANGES)
    f, base = mr.baseline(toks, dq, ranges, log, 'selection fp32')
    np.savez(out / 'selection_baseline.npz', S0=base['S0'], w0=base['w0'], p0=base['p0'], tokens=toks)
    nA = len(toks) // WPA
    res = {'fp32': {'ce': float(np.sum(f['ce']) / (len(toks) * TP))}, 'configs': {}}
    rec_pS0 = {}
    names = ['aggressive'] + [f'protect_L{l:02d}' for l in range(NL)]
    cc = {}
    art = {}
    for nm in names:
        fm = read(out / f'configs/{nm}.json')
        if nm != 'aggressive' and sha(out / f'configs/{nm}.json') != pl['candidates']['table'][int(nm[-2:])]['sha256']:
            raise SystemExit(f'{nm} changed')
        cc[nm] = dr.config_checks(fm, dq, ranges)
        r = forward_measure(toks, fm, dq, ranges, base['S0'], log=log, label=f'sel {nm}')
        rs_ = route_summary(r, base)
        rec_pS0[nm] = r['pS0']
        art[nm] = per_article(r['ce'], nA)
        res['configs'][nm] = {'ce': float(np.sum(r['ce']) / (len(toks) * TP)), **rs_, 'finite': r['finite_ce'] and r['finite_weights'],
                              'audit': r['audit'], 'seconds': r['seconds'], 'config_checks_pass': cc[nm]['pass'],
                              'routes_differ_from_S0': rs_['set_change_vs_S0'] > 0}
    # protected layer l cannot change routers at layers <= l
    for l in range(NL):
        nm = f'protect_L{l:02d}'
        res['configs'][nm]['routers_upto_l_unchanged'] = bool(np.array_equal(rec_pS0[nm][:, :l + 1], rec_pS0['aggressive'][:, :l + 1]))
    ga, ca = res['configs']['aggressive']['G'], res['configs']['aggressive']['ce']
    tab = {r['layer']: r for r in pl['candidates']['table']}
    elig = []
    for l in range(NL):
        x = res['configs'][f'protect_L{l:02d}']
        x.update({'dR': tab[l]['dR'], 'R': tab[l]['R'], 'dG': ga - x['G'], 'dCE': ca - x['ce'], 'score': (ga - x['G']) / tab[l]['dR'] if tab[l]['dR'] > 0 else None})
        if x['G'] < ga and x['ce'] < ca and tab[l]['dR'] > 0:
            elig.append(l)
    ok = all(v['config_checks_pass'] and v['finite'] and v['audit']['router_untouched'] for v in res['configs'].values()) and \
        all(res['configs'][f'protect_L{l:02d}']['routers_upto_l_unchanged'] for l in range(NL))
    res['controls_pass'] = bool(ok)
    res['eligible_layers'] = elig
    if not ok:
        res['choice'] = None
        write(out / 'selection.json', res)
        raise SystemExit('selection controls failed')
    if elig:
        best = min(elig, key=lambda l: (-res['configs'][f'protect_L{l:02d}']['score'], res['configs'][f'protect_L{l:02d}']['ce'], tab[l]['R'], l))
        src = out / f'configs/protect_L{best:02d}.json'
        write(out / 'configs/route_focused.json', read(src))
        res['choice'] = {'layer': best, 'config': 'configs/route_focused.json', 'sha256': sha(out / 'configs/route_focused.json'), 'source': str(src.name)}
    else:
        res['choice'] = None
    res['article_ce'] = {k: v.tolist() for k, v in art.items()}
    res['fp32_article_ce'] = per_article(f['ce'], nA).tolist()
    res['seconds_fp32'] = f['seconds']
    write(out / 'selection.json', res)
    np.savez(out / 'selection_pS0.npz', **rec_pS0)
    write(out / 'choice_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'selection_sha256': sha(out / 'selection.json'), 'choice': res['choice']})
    log(json.dumps({'eligible': elig, 'choice': res['choice']}))


def bc_solve(out, log):
    import mp_milp
    from mp_auto import config_hash
    from run_native_bc_pipeline import GRID11
    from run_olmoe_search_final import olmoe_problem
    if (out / 'bc_solve.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'choice_freeze.json')
    if not fz['choice']:
        raise SystemExit('no chosen config: stop as pre-registered')
    if sha(out / 'configs/route_focused.json') != fz['choice']['sha256']:
        raise SystemExit('chosen config changed')
    cfm = read(out / 'configs/route_focused.json')
    table, names, nodes, elems, flops, bc = problem()
    rc = resources(cfm, names, elems, flops)
    pn, pnames, value, pel, pfl, pcost = olmoe_problem()
    assert pnames == names
    kept = sum(value(n, cfm[n]['w'], cfm[n]['a']) is not None for n in names)
    info = {}
    t0 = time.perf_counter()
    fm, obj, status = mp_milp.milp_select(pn, GRID11, GRID11, value, elems, flops, rc['M_bytes'], rc['BOPs'], time_limit=TIME_LIMIT, info=info)
    sec = time.perf_counter() - t0
    fm = {n: fm[n] for n in names}
    write(out / 'configs/bc_same_budget.json', fm)
    rb = resources(fm, names, elems, flops)
    res = {'targets': {'M_bytes': rc['M_bytes'], 'BOPs': rc['BOPs'], 'R': rc['R']}, 'seconds': sec, 'status': int(status), 'info': info,
           'proven_optimal': int(status) == 0, 'label': 'optimal' if int(status) == 0 else 'feasible incumbent',
           'objective_milp': float(obj), 'objective_recomputed': bc_total(fm, names, bc), 'chosen_config_objective': bc_total(cfm, names, bc),
           'resources': rb, 'budget_use': {'size': rb['M_bytes'] / rc['M_bytes'], 'bops': rb['BOPs'] / rc['BOPs']},
           'feasible': rb['M_bytes'] <= rc['M_bytes'] * (1 + 1e-12) and rb['BOPs'] <= rc['BOPs'] * (1 + 1e-12),
           'config_hash': config_hash(fm), 'sha256': sha(out / 'configs/bc_same_budget.json'),
           'chosen_in_candidate_space': all(cfm[n]['w'] in GRID11 and cfm[n]['a'] in GRID11 for n in names),
           'chosen_options_kept_by_pruning': kept, 'nodes': len(names), 'problem_build': pcost,
           'nodes_differing_from_chosen': sum(fm[n] != cfm[n] for n in names), 'nodes_differing_from_aggressive': sum(fm[n] != read(BASE_CFG)[n] for n in names)}
    res['bc_not_worse_than_chosen_objective'] = res['objective_recomputed'] <= res['chosen_config_objective'] * (1 + 1e-9)
    write(out / 'bc_solve.json', res)
    log(json.dumps({k: res[k] for k in ('seconds', 'status', 'label', 'objective_recomputed', 'chosen_config_objective', 'budget_use', 'feasible')}))


def selection_bc(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'sample_size.json').exists():
        raise SystemExit('exists')
    bs = read(out / 'bc_solve.json')
    if sha(out / 'configs/bc_same_budget.json') != bs['sha256'] or not bs['feasible']:
        raise SystemExit('BC config changed or infeasible')
    toks = np.load(out / 'selection_tokens.npy')
    b = np.load(out / 'selection_baseline.npz')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = dr.read(dr.RANGES)
    fm = read(out / 'configs/bc_same_budget.json')
    cc = dr.config_checks(fm, dq, ranges)
    r = forward_measure(toks, fm, dq, ranges, b['S0'], log=log, label='sel bc')
    sel = read(out / 'selection.json')
    nA = len(toks) // WPA
    a_bc = per_article(r['ce'], nA)
    a_ch = np.array(sel['article_ce'][f'protect_L{sel["choice"]["layer"]:02d}'])
    d = a_ch - a_bc
    s = float(d.std(ddof=1))
    n_req = math.ceil((1.96 * s / H_TARGET) ** 2)
    pl = read(out / 'plan.json')
    n = int(min(N_MAX, pl['data']['eligible_before_length_and_neardup'] - N_SEL, max(N_MIN, n_req)))
    base = {'S0': b['S0'], 'w0': b['w0']}
    write(out / 'sample_size.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'bc_selection_ce': float(np.sum(r['ce']) / (len(toks) * TP)),
                                     'bc_selection_route': route_summary(r, base), 'config_checks_pass': cc['pass'],
                                     'selection_diff_mean': float(d.mean()), 'selection_diff_article_sd': s, 'n_required': n_req, 'n_articles': n,
                                     'expected_95_half_width': 1.96 * s / math.sqrt(n), 'note': 'selection took part in the choice: s may be optimistic',
                                     'seconds': r['seconds']})
    log(f'sample size n={n} (s={s:.5f})')


def data_final(out, log):
    if (out / 'final_segments.json').exists():
        raise SystemExit('exists')
    n = read(out / 'sample_size.json')['n_articles']
    sel = read(out / 'selection_segments.json')
    segs, toks, skips = draw(out, N_SEL, n, 'final', {s['content_sha256'] for s in sel})
    if {s['content_sha256'] for s in segs} & {s['content_sha256'] for s in sel}:
        raise SystemExit('final overlaps selection')
    np.save(out / 'final_tokens.npy', toks)
    write(out / 'final_segments.json', segs)
    write(out / 'final_data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sample_size_sha256': sha(out / 'sample_size.json'),
                                           'choice_freeze_sha256': sha(out / 'choice_freeze.json'), 'bc_solve_sha256': sha(out / 'bc_solve.json'),
                                           'n': n, 'skips_total_walk': skips, 'tokens_sha256': sha(out / 'final_tokens.npy'),
                                           'segments_sha256': sha(out / 'final_segments.json'),
                                           'frozen_configs': {c: sha(out / f'configs/{c}.json') for c in ('aggressive', 'route_focused', 'bc_same_budget')}})
    log(f'final data frozen: {n} articles')


def final(out, log):
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'final_results.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'final_data_freeze.json')
    if sha(out / 'final_tokens.npy') != fz['tokens_sha256'] or any(sha(out / f'configs/{c}.json') != h for c, h in fz['frozen_configs'].items()):
        raise SystemExit('tokens or configs changed')
    toks = np.load(out / 'final_tokens.npy')
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = dr.read(dr.RANGES)
    f, base = mr.baseline(toks, dq, ranges, log, 'final fp32')
    np.savez(out / 'final_baseline.npz', S0=base['S0'], w0=base['w0'], tokens=toks)
    runs, cc, rsum = {}, {}, {}
    (out / 'routes').mkdir(exist_ok=True)
    for c in ('aggressive', 'route_focused', 'bc_same_budget'):
        fm = read(out / f'configs/{c}.json')
        cc[c] = dr.config_checks(fm, dq, ranges)
        runs[c] = forward_measure(toks, fm, dq, ranges, base['S0'], log=log, label=f'final {c}')
        rsum[c] = route_summary(runs[c], base)
        np.savez(out / f'routes/{c}.npz', ids=runs[c]['routes']['ids'], weights=runs[c]['routes']['weights'], pS0=runs[c]['pS0'])
    ctl = {'config_checks': {c: v['pass'] for c, v in cc.items()}}
    sub = toks[:CTRL]
    off = forward_measure(sub, None, dq, ranges, base['S0'][:CTRL], log=log, label='off')
    ctl['quant_off'] = mr.eq(off, f, CTRL)
    ctl['rerun'], ctl['replay'], rep = {}, {}, {}
    for c in ('route_focused', 'bc_same_budget'):
        fm = read(out / f'configs/{c}.json')
        r2 = forward_measure(sub, fm, dq, ranges, base['S0'][:CTRL], log=log, label=f'rerun {c}')
        e = mr.eq(r2, runs[c], CTRL)
        e['routes_equal'] = bool(np.array_equal(r2['routes']['ids'], runs[c]['routes']['ids'][:CTRL]) and np.array_equal(r2['routes']['weights'], runs[c]['routes']['weights'][:CTRL]))
        ctl['rerun'][c] = e
        rp = dr.forward(toks, fm, dq, ranges, routes=dr.routes_to_dev(runs[c]['routes']), log=log, label=f'replay {c}')
        ctl['replay'][c] = mr.eq(rp, runs[c]); rep[c] = rp['seconds']
    ctl['routes_differ_from_S0'] = {c: rsum[c]['set_change_vs_S0'] > 0 for c in runs}
    ctl['finite'] = all(r['finite_ce'] and r['finite_weights'] for r in runs.values()) and f['finite_ce']
    ctl['audit'] = {c: r['audit'] for c, r in runs.items()}
    ctl['pass'] = bool(all(ctl['config_checks'].values()) and mr.bitwise(ctl['quant_off']) and all(mr.bitwise(v) and v['routes_equal'] for v in ctl['rerun'].values())
                       and all(mr.bitwise(v) for v in ctl['replay'].values()) and all(ctl['routes_differ_from_S0'].values()) and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'final_results.json', {'ce': {'fp32': f['ce'], **{c: r['ce'] for c, r in runs.items()}}, 'route': rsum, 'controls': ctl,
                                       'seconds': {'fp32': f['seconds'], **{c: r['seconds'] for c, r in runs.items()}, 'quant_off': off['seconds'],
                                                   'replay': rep}, 'environment': dr.env_info()})
    log(json.dumps({'controls_pass': ctl['pass']}))


# --------------------------------------------------------------------------- #
def figures(out, sel, pl, fr, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    ink, ink2, grid, surf = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, red, mid = '#2a78d6', '#eb6834', '#e34948', '#f0efec'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': ink2, 'axes.labelcolor': ink, 'xtick.color': ink2, 'ytick.color': ink2,
                         'figure.facecolor': surf, 'axes.facecolor': surf})
    tab = {r['layer']: r for r in pl['candidates']['table']}
    L = list(range(NL))
    dce = [sel['configs'][f'protect_L{l:02d}']['dCE'] for l in L]
    dR = [tab[l]['dR'] for l in L]
    ch = sel['choice']['layer'] if sel.get('choice') else None
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150)
    axs[0].bar(L, dce, color=[blue if l == ch else '#86b6ef' for l in L], width=0.7)
    axs[0].axhline(0, color=ink2, lw=1)
    axs[0].set_xlabel('protected layer'); axs[0].set_ylabel('selection CE improvement vs aggressive (nat/token)')
    axs[0].set_title('Selection loss improvement (exploratory)', loc='left', color=ink, fontsize=9)
    axs[1].bar(L, dR, color='#86b6ef', width=0.7)
    axs[1].set_xlabel('protected layer'); axs[1].set_ylabel('ΔR (resource increase)')
    axs[1].set_title('Resource increase of each protection', loc='left', color=ink, fontsize=9)
    for ax in axs:
        ax.grid(axis='y', color=grid, lw=0.8); ax.set_axisbelow(True); ax.set_xticks(L)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    if ch is not None:
        axs[0].annotate('chosen', (ch, dce[ch]), xytext=(0, 4), textcoords='offset points', ha='center', color=ink2, fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'fig_selection_loss_and_dR.png'); plt.close(fig)
    ga = np.array(sel['configs']['aggressive']['G_per_layer'])
    M = np.array([ga - np.array(sel['configs'][f'protect_L{l:02d}']['G_per_layer']) for l in L])[:, 1:]
    vmax = float(np.abs(M).max()) or 1e-9
    cmap = LinearSegmentedColormap.from_list('div', [red, mid, blue])
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
    ax.set_xticks(range(15)); ax.set_xticklabels(range(1, 16)); ax.set_yticks(L)
    ax.set_xlabel('router layer'); ax.set_ylabel('protected layer')
    ax.set_title('Reduction of router-proportion drift vs aggressive (blue = less drift)', loc='left', color=ink, fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label('G_aggr,l − G_cand,l', color=ink2)
    fig.tight_layout(); fig.savefig(out / 'fig_protection_route_drift.png'); plt.close(fig)
    if fr is not None and res is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150, gridspec_kw={'width_ratios': [1.1, 1]})
        cols = {'aggressive': ink2, 'route_focused': blue, 'bc_same_budget': orange}
        off = {'aggressive': 0.0, 'route_focused': -0.0012, 'bc_same_budget': 0.0012}
        for c, col in cols.items():
            e = res['damage'][c]
            axs[0].errorbar(res['R'][c] + off[c], e['mean'], yerr=[[e['mean'] - e['ci95'][0]], [e['ci95'][1] - e['mean']]], fmt='o', ms=6,
                            color=col, elinewidth=2, capsize=0, mec=surf, mew=1.5, label=c)
        axs[0].set_xlim(0.588, 0.632)
        axs[0].set_xlabel('resource R = max(size, BOPs) / int8 (route_focused / BC offset ±0.0012)')
        axs[0].set_ylabel('dynamic damage vs FP32 (nat/token)')
        axs[0].set_title('Absolute damage (95% CI; overlap ≠ no difference)', loc='left', color=ink, fontsize=9)
        axs[0].legend(frameon=False, labelcolor=ink, loc='lower left')
        keys = [('primary_route_minus_bc', 'route_focused − BC (primary, 95%)', 'ci95'),
                ('route_minus_aggressive', 'route_focused − aggressive (97.5%)', 'ci_bonf2'),
                ('bc_minus_aggressive', 'BC − aggressive (97.5%)', 'ci_bonf2')]
        for i, (k, lab, ck) in enumerate(keys):
            e = res['contrasts'][k]
            axs[1].errorbar(e['mean'], i, xerr=[[e['mean'] - e[ck][0]], [e[ck][1] - e['mean']]], fmt='o', ms=6, color=ink2 if i else blue,
                            elinewidth=2, capsize=0, mec=surf, mew=1.5)
        axs[1].axvline(0, color=ink2, lw=1)
        axs[1].set_yticks(range(3)); axs[1].set_yticklabels([k[1] for k in keys]); axs[1].invert_yaxis()
        axs[1].set_xlabel('paired CE diff (nat/token; < 0: first better)')
        axs[1].set_title('Paired differences on the final test', loc='left', color=ink, fontsize=9)
        for ax in axs:
            ax.grid(axis='x' if ax is axs[1] else 'y', color=grid, lw=0.8); ax.set_axisbelow(True)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
        fig.tight_layout(); fig.savefig(out / 'fig_final_damage_vs_R.png'); plt.close(fig)


def analyze(out, log):
    t0 = time.time()
    pl, sel = read(out / 'plan.json'), read(out / 'selection.json')
    fr = read(out / 'final_results.json')
    if not fr['controls']['pass']:
        raise SystemExit('final controls failed')
    nA = len(read(out / 'final_segments.json'))
    A = {c: per_article(v, nA) for c, v in fr['ce'].items()}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, nA, (BOOT, nA))

    def st(x):
        d = x[idx].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci_bonf2': np.percentile(d, [1.25, 98.75]).tolist(),
                'article_sd': float(x.std(ddof=1)), 'articles_positive': int((x > 0).sum())}
    E = {'primary_route_minus_bc': A['route_focused'] - A['bc_same_budget'], 'route_minus_aggressive': A['route_focused'] - A['aggressive'],
         'bc_minus_aggressive': A['bc_same_budget'] - A['aggressive']}
    res = {'n_articles': nA, 'contrasts': {k: st(v) for k, v in E.items()}, 'damage': {c: st(A[c] - A['fp32']) for c in ('aggressive', 'route_focused', 'bc_same_budget')}}
    c = res['contrasts']
    lo, hi = c['primary_route_minus_bc']['ci95']
    res['primary'] = 'route_focused_better' if hi < 0 else 'bc_better' if lo > 0 else 'uncertain'
    sec = {k: ('negative' if c[k]['ci_bonf2'][1] < 0 else 'positive' if c[k]['ci_bonf2'][0] > 0 else 'none') for k in ('route_minus_aggressive', 'bc_minus_aggressive')}
    res['secondary'] = sec
    rf_better_aggr = sec['route_minus_aggressive'] == 'negative'
    g_better = fr['route']['route_focused']['G'] < fr['route']['aggressive']['G']
    res['interpretation'] = ('A' if res['primary'] == 'route_focused_better' and rf_better_aggr else
                             'B' if rf_better_aggr else 'C' if g_better else 'D')
    res['primary_ci_spans_zero'] = res['primary'] == 'uncertain'
    tokens = nA * WPA * TP
    res['ce_ppl'] = {k: {'ce': float(np.sum(v) / tokens), 'ppl': float(np.exp(np.sum(v) / tokens))} for k, v in fr['ce'].items()}
    bs = read(out / 'bc_solve.json')
    tab = {r['layer']: r for r in pl['candidates']['table']}
    ch = sel['choice']['layer']
    res['R'] = {'aggressive': pl['base']['R'], 'route_focused': tab[ch]['R'], 'bc_same_budget': bs['resources']['R']}
    res['resources'] = {'aggressive': {'M_rel': pl['base']['M_rel'], 'B_rel': pl['base']['B_rel']},
                        'route_focused': {'M_rel': tab[ch]['M_rel'], 'B_rel': tab[ch]['B_rel']},
                        'bc_same_budget': {'M_rel': bs['resources']['M_rel'], 'B_rel': bs['resources']['B_rel']}}
    res['route'] = fr['route']
    with open(out / 'final_per_article.csv', 'w') as fh:
        segs = read(out / 'final_segments.json')
        rows = [{'index': s['index'], 'title': s['title'], **{f'ce_{k}': float(v[i]) for k, v in A.items()}, **{k: float(v[i]) for k, v in E.items()}}
                for i, s in enumerate(segs)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    with open(out / 'selection_table.csv', 'w') as fh:
        rows = [{'layer': l, **{k: tab[l][k] for k in ('nodes_modified', 'dM_rel', 'dB_rel', 'dR', 'R', 'bc_objective')},
                 **{k: sel['configs'][f'protect_L{l:02d}'][k] for k in ('ce', 'dCE', 'G', 'dG', 'score', 'set_change_vs_S0', 'weight_sum_minus_m0')}}
                for l in range(NL)]
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    figures(out, sel, pl, fr, res)
    res['controls'] = fr['controls']
    res['timing'] = {'final': fr['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    write(out / 'analysis.json', res)
    log(json.dumps({k: [c[k]['mean'], c[k]['ci95']] for k in E}) + ' ' + res['primary'] + ' ' + res['interpretation'])


def main():
    ap = argparse.ArgumentParser()
    stages = ['plan', 'data_selection', 'control_prev', 'selection', 'bc_solve', 'selection_bc', 'data_final', 'final', 'analyze']
    ap.add_argument('--stage', choices=stages, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
