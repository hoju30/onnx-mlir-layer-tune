#!/usr/bin/env python3
"""OLMoE: the frozen BC knee configuration evaluated with fixed routes (FP32 baseline ids + weights) and with dynamic
routes (the quantized model's own FP32 router on its own hidden states). Pre-registered in plan.json / plan.md.

Conditions (same windows, same chunking, same code path; only the route source differs):
  fp32_dyn    original model, own router (baseline; its ids / weights / top-16 router logits are recorded)
  fp32_fix    original model, ids + weights of fp32_dyn (implementation control)
  q_fix       BC knee configuration, ids + weights of fp32_dyn
  q_dyn       BC knee configuration, own router (FP32 router weights and arithmetic) on its own hidden states

Stages (one --out-dir):
  plan     plan.json / plan.md (config choice and reason, data design, sample-size rule, statistics, checks, hashes)
  pilot    already-seen windows (search_final C, first 128): historical reproduction, numerical controls, SD of the
           route difference -> sample size by the pre-registered rule
  data     new articles (strict article boundaries, whole-article exclusion), frozen tokens
  capture  fp32_dyn on the new windows (routes, router top-16 logits, CE)
  eval     fp32_fix, q_fix, q_dyn (+ controls) on the new windows
  analyze  endpoints, per-article table, per-layer routing statistics
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SF = HERE / 'runs/olmoe_search_final_20261016'
MP = HERE / 'runs/olmoe_multipoint_20261015'
CT = HERE / 'runs/olmoe_crosstext_20261017'
FO = HERE / 'runs/olmoe_first_order_validation_20261017'
CS = HERE / 'runs/olmoe_calibration_stability_20261018'
DATA = HERE / 'gpt2_assets/data'
CONFIG = SF / 'C/configs/milp.json'
RANGES = SF / 'C/ranges.json'
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
CHUNK = 128
EPS = 0.002
BOOT = 10000
SEEDS = {'data': 18051, 'bootstrap': 18061}
PILOT_WINDOWS = 128                      # search_final C windows 0..127 (validation, from token 196608; already seen)
N_MIN, N_MAX = 32, 128
WPA = 4
SCRIPTS = ['run_olmoe_dynamic_route.py', 'olmoe_common.py', 'run_olmoe_search_final.py', 'run_olmoe_multipoint.py',
           'run_olmoe_fixed_route.py', 'run_olmoe_crosstext.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py',
           'mp_auto.py', 'mp_milp.py']


def read(p):
    return json.loads(Path(p).read_text())


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
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
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def env_info():
    import platform
    info = {'python': sys.version.split()[0], 'executable': sys.executable, 'numpy': np.__version__, 'platform': platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32, matmul_precision=torch.get_float32_matmul_precision(),
                    threads=torch.get_num_threads())
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info['nvidia_kernel_module'] = Path('/proc/driver/nvidia/version').read_text().split('\n')[0]
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------- #
# data: strict article boundaries
# --------------------------------------------------------------------------- #
def strict_articles(split, tok):
    """WikiText-2 articles whose title line ' = T = ' has an empty line before and after it (the official 600 / 60 / 60
    articles); title-like lines inside tables or formulas (no empty neighbours) stay inside the enclosing article."""
    import olmoe_common as oc
    import pyarrow.parquet as pq
    lines = pq.read_table(DATA / f'{split}-00000-of-00001.parquet')['text'].to_pylist()
    text = oc.wikitext(split)
    starts, rejected, pos = [], [], 0
    for i, line in enumerate(lines):
        if re.match(r'^ = [^=].* = \n$', line):
            if (i == 0 or lines[i - 1] == '') and i + 1 < len(lines) and lines[i + 1] == '':
                starts.append((line.strip(' =\n'), pos))
            else:
                rejected.append({'line': i, 'text': line.strip()[:80]})
        pos += len(line) + 2
    if pos - 2 != len(text):
        raise SystemExit('article offsets do not match the joined text')
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc['input_ids'], enc['offset_mapping']
    ts = [o[0] for o in off]
    arts = []
    for k, (title, a) in enumerate(starts):
        b = starts[k + 1][1] if k + 1 < len(starts) else len(text)
        arts.append({'title': title, 'chars': [a, b], 'tokens': [bisect.bisect_left(ts, a), bisect.bisect_left(ts, b)]})
    return arts, ids, off, rejected


def used_spans_all():
    from run_olmoe_crosstext import used_spans
    used = used_spans()
    used['train'] = []
    for f in (CT / 'segments.json', FO / 'segments.json', CS / 'segments.json'):
        for s in read(f):
            used[s['split']].append(tuple(s['article_chars']))
            used[s['split']].append(tuple(s['window_chars']))
    return used


# --------------------------------------------------------------------------- #
# model passes
# --------------------------------------------------------------------------- #
def gpu_setup():
    import torch
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact('cuda')
    torch.set_num_threads(8)


def load_config():
    return read(CONFIG)


def config_checks(fm, dq, ranges):
    """config completeness: formats supported (explicit list, no silent pass-through), defaults for absent nodes."""
    import olmoe_common as oc
    cfg = oc.config()
    wf_ok = {'int4', 'int5', 'int6', 'int7', 'int8', 'p4e1', 'p6e1', 'p8e1', 'f16', 'bf16', 'fp32'}
    af_ok = {'int4', 'int5', 'int6', 'int7', 'int8', 'p4e1', 'p6e1', 'p8e1', 'f16', 'bf16', 'fp32'}
    allk = [f'L{l}.E{e}.{p}' for l in range(cfg.num_hidden_layers) for e in range(cfg.num_experts) for p in oc.PROJ]
    absent = [k for k in allk if k not in fm]
    bad = [k for k, v in fm.items() if v['w'] not in wf_ok or v['a'] not in af_ok]
    noranges = [k for k, v in fm.items() if v['a'] != 'fp32' and not all(s in ranges.get(k, {}) for s in ('x', 'y'))]
    extra = [k for k in fm if k not in set(allk)]
    posit = set(dq.dev.posit)
    missing_posit = sorted({v[s] for v in fm.values() for s in ('w', 'a') if v[s].startswith('p')} - posit)
    return {'nodes_in_config': len(fm), 'nodes_total': len(allk), 'absent_nodes_bf16_weight_fp32_act': absent,
            'unsupported_formats': bad, 'quantized_act_without_ranges': noranges, 'unknown_keys': extra,
            'posit_luts_missing': missing_posit,
            'weight_formats': dict(Counter(v['w'] for v in fm.values())), 'act_formats': dict(Counter(v['a'] for v in fm.values())),
            'pass': not bad and not noranges and not extra and not missing_posit}


def forward(toks_np, fm, dq, ranges, routes=None, record=False, router_top=0, log=print, label='', chunk=CHUNK):
    """Layer-major pass over window chunks. fm None = FP32 model. routes: per layer (idx [N*T, k] long, rw [N*T, k]
    float32) on the device -> fixed route; None -> the model's own router (oc.route) on its own hidden states.
    The layer arithmetic is oc.layer_fixed / oc.layer_dynamic written once, so both route modes share every op.
    Returns per-window summed CE, per-window sha256 of the final hidden state, recorded routes (dynamic only),
    router top-`router_top` logits / ids (dynamic only), seconds, audit."""
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
    rec = {'ids': np.zeros((nW, NL, SEQ, K), np.int16), 'weights': np.zeros((nW, NL, SEQ, K), np.float32)} if record else None
    top = {'ids': np.zeros((nW, NL, SEQ, router_top), np.int16), 'logits': np.zeros((nW, NL, SEQ, router_top), np.float32)} if router_top else None
    audit = {'router_untouched': True, 'hook_keys_are_expert_projections': True, 'hooks': 0, 'nonfinite_route_weights': 0}
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
                audit['hook_keys_are_expert_projections'] &= all(k[0] == l and k[2] in oc.PROJ and k[3] in ('x', 'y') for k in hk.act)
            for a, b in chunks:
                B = b - a
                h = H[a]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], eps).reshape(-1, Hd)
                if routes is None:
                    idx, rw = oc.route(wq, z, cfg)
                    if record:
                        rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                        rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
                        audit['nonfinite_route_weights'] += int((~torch.isfinite(rw)).sum())
                    if router_top:
                        v, i = torch.topk(F.linear(z, wq['router']), router_top, dim=-1)
                        top['ids'][a:b, l] = i.view(B, SEQ, router_top).to(torch.int16).cpu().numpy()
                        top['logits'][a:b, l] = v.view(B, SEQ, router_top).float().cpu().numpy()
                else:
                    rows = slice(a * SEQ, b * SEQ)
                    idx, rw = routes[l][0][rows], routes[l][1][rows]
                H[a] = x + oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd)
            del wq, w
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k]).tobytes()).hexdigest() for k in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'top': top, 'seconds': time.time() - t0, 'audit': audit,
            'chunks': chunks, 'finite_ce': bool(np.isfinite(ce).all())}


def routes_to_dev(rec):
    import torch
    nW = rec['ids'].shape[0]
    return [(torch.from_numpy(rec['ids'][:, l].reshape(nW * SEQ, K).astype(np.int64)).to('cuda'),
             torch.from_numpy(rec['weights'][:, l].reshape(nW * SEQ, K)).to('cuda')) for l in range(NL)]


def save_routes(path, rec, top=None, tokens=None):
    kw = {'ids': rec['ids'], 'weights': rec['weights']}
    if top is not None:
        kw.update(top_ids=top['ids'], top_logits=top['logits'])
    if tokens is not None:
        kw['tokens'] = tokens
    np.savez(path, **kw)


def load_routes_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def route_hash(rec):
    h = hashlib.sha256()
    for k in ('ids', 'weights'):
        a = np.ascontiguousarray(rec[k])
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def same(a, b):
    return {'ce_max_abs': float(np.max(np.abs(np.array(a['ce']) - np.array(b['ce'])))),
            'hidden_bitwise_windows': int(sum(x == y for x, y in zip(a['hidden_sha'], b['hidden_sha']))),
            'windows': len(a['ce'])}


# --------------------------------------------------------------------------- #
def plan(out, log):
    import olmoe_common as oc
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    sm = read(SF / 'B/search_milp.json')
    from mp_auto import config_hash
    names = sm['names']
    knee_fm = {n: sm['selection']['selected']['formats'][n] for n in names}
    fm = load_config()
    model_files = sorted((oc.MODEL_DIR).glob('*.safetensors')) + [oc.MODEL_DIR / 'config.json', oc.MODEL_DIR / 'model.safetensors.index.json']
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [CONFIG, RANGES, SF / 'C/freeze.json', SF / 'B/search_milp.json', SF / 'C/truth.json',
                                                         SF / 'C/routes_manifest.json', SF / 'C/tokens_eval.npy', SF / 'C/windows.json',
                                                         MP / 'table.json', MP / 'table_meta.json', MP / 'ranges.json',
                                                         CT / 'segments.json', FO / 'segments.json', CS / 'segments.json',
                                                         DATA / 'validation-00000-of-00001.parquet', DATA / 'test-00000-of-00001.parquet',
                                                         DATA / 'train-00000-of-00001.parquet']}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'Same BC mixed-precision configuration: fixed route (FP32 baseline ids + weights of the same text) vs '
                    'dynamic route (the quantized model routes itself with the FP32 router) - is the loss / PPL different?',
        'configuration': {
            'path': str(CONFIG.relative_to(HERE)), 'sha256': sha(CONFIG), 'config_hash_mp_auto': config_hash(fm),
            'equals_search_final_B_milp_knee': config_hash(knee_fm) == config_hash(fm),
            'knee': 'R 0.7071056, BC 18.29196 (runs/olmoe_search_final_20261016 B, corrected MILP)',
            'reason': 'The default pipeline for OLMoE is still mp_auto.auto_select with mp_milp (lp_pipeline_regression and '
                      'search_final changed no default; LP-first is a research solver). The mp_milp minimal fix integrated on '
                      '2026-10-16 (1-ppm retry, repair only when the retry fails) is exactly the corrected MILP of search_final '
                      'B, whose knee is this file; it was validated on 512 new windows (dNLL +0.0195, GPU, corrected quantizer). '
                      'Not used: multipoint bc_p125 (pre-fix path, differs in 11 projections), LP knee (research solver). '
                      'Recorded before any new evaluation; no result of this study can change it.',
            'ranges': str(RANGES.relative_to(HERE)), 'quantizer': 'run_olmoe_multipoint.DevQuant + corrected int activation '
                      '(run_olmoe_search_final.dev_quant(corrected=True)), as in the search_final C evaluation',
            'semantics': '3051 searched expert projections with their (w, a) formats; the 21 absent projections (experts '
                         'without calibration tokens) bf16 weights + FP32 activations (historical semantics, '
                         'run_olmoe_search_final.quantized_layer). Attention, norms, router, embeddings and LM head FP32. '
                         'No router quantization.'},
        'conditions': {
            'fp32_dyn': 'original model, oc.route: FP32 router logits, softmax over 64 in FP32, top-8, no renormalization '
                        '(norm_topk_prob false); records ids (int16), weights (float32, exact) and top-16 router logits',
            'fp32_fix': 'original model, ids AND weights of fp32_dyn (control)',
            'q_fix': 'configuration, ids AND weights of fp32_dyn of the same windows (historical fixed-route semantics; '
                     'all earlier OLMoE fixed-route studies fixed both ids and weights)',
            'q_dyn': 'configuration, the FP32 router on the quantized model hidden states at every layer (upstream '
                     'quantization can change later routes even though the router is FP32)',
            'shared': 'one function for all passes (oc.layer_fixed / layer_dynamic arithmetic written once); same windows, '
                      f'chunks of {CHUNK} windows, GPU FP32, TF32 off, matmul precision highest, eval (no dropout exists in '
                      'the functional model), no padding (every window 128 tokens), causal mask, 127 predicted tokens per '
                      'window, summed CE in float64'},
        'checks': [
            'pilot: fp32_dyn (capture batch shape, 64 windows) reproduces search_final C route hashes and dynamic CE bitwise; '
            'q_fix and fp32_fix with C routes (chunk 128) reproduce C truth (milp, fp32) bitwise',
            'fp32_fix == fp32_dyn (CE and final hidden state bitwise); dynamic path without quantization == fp32_dyn',
            'q_dyn rerun bitwise; q_fix fed with q_dyn own routes == q_dyn bitwise (route source is the only difference)',
            'config: 3051 nodes, supported formats only (explicit list), ranges for every quantized activation, posit LUTs '
            'present; router weight object untouched; activation hooks only on expert projections',
            'route records: tokens, layer, model revision and hashes stored; weights finite; all CE finite; 127 tokens/window',
            'any failure: fix and record before interpreting q_dyn; if bitwise equality cannot hold, quantify FP32 control '
            'and same-path rerun differences first'],
        'pilot_and_sample_size': {
            'pilot': f'search_final C windows 0-{PILOT_WINDOWS - 1} (validation, token 196608+, already seen); all controls; '
                     'Delta_route per window; no data of the new set is touched',
            'rule': 'blocks of 4 consecutive pilot windows (32 blocks, contiguous text ~ article); s = SD of block means of '
                    'Delta_route; n_articles = ceil((1.645 s / (eps/2))^2) clipped to [32, 128] (90% CI half-width <= eps/2)'},
        'data': {
            'article_rule': 'strict WikiText-2 article boundaries: a title line " = T = " counts only with an empty line '
                            'before and after (gives exactly 600 / 60 / 60 articles); title-like lines inside tables / '
                            'formulas stay in the enclosing article',
            'exclusion': 'whole strict articles overlapping any earlier OLMoE text: stage-1 calibration/diagnostic/check, '
                         'multipoint, solver regression, search_final C, numerics B, cross-text D/V, first-order H, '
                         'calibration-stability J and C1-C8 (train)',
            'availability_before_plan': 'counting only (no routes / loss): eligible >=512 tokens: validation 10, test 18, train 556',
            'sources': 'original evaluation source first: all eligible validation (10) and test (18) articles (census, no '
                       'choice); the rest (n - 28) uniformly from eligible train articles without replacement. Strata: '
                       'validation / test / train; stratified bootstrap and separate reports (original source vs train)',
            'segment': f'{WPA} consecutive 128-token windows per article at a uniform start; seed {SEEDS["data"]}; titles, '
                       'source, char/token offsets and token hashes frozen before any evaluation; no resampling'},
        'endpoints': {
            'per_article': 'CE per condition = summed CE of its 4 windows / 508 tokens',
            'D_fixed': 'CE(q_fix) - CE(fp32_dyn)', 'D_dynamic': 'CE(q_dyn) - CE(fp32_dyn)',
            'primary': 'Delta_route = CE(q_dyn) - CE(q_fix), mean over articles (equal weights: every article 508 tokens); '
                       'positive = dynamic adds damage',
            'ppl': 'exp of the token-weighted mean CE per condition (never an average of window PPLs)',
            'bootstrap': f'{BOOT} paired article bootstrap, stratified by source, same draws for every quantity, seed '
                         f'{SEEDS["bootstrap"]}; 95% and 90% percentile intervals',
            'epsilon': f'{EPS} nat/token, taken from runs/olmoe_search_final_20261016 (EQUIV) and '
                       'runs/solver_regression_20261016 (config-vs-config equivalence margin of the OLMoE knee); a research '
                       'tolerance, not a general standard (about 10% of the knee damage 0.0195)',
            'decision': 'equivalent within tolerance if the 90% CI lies in [-eps, +eps]; dynamic extra damage beyond '
                        'tolerance if the 90% CI lower bound > +eps; dynamic better beyond tolerance if the 90% CI upper '
                        'bound < -eps; otherwise uncertain. The 95% CI is reported too; containing 0 is not equivalence.',
            'secondary': 'D_fixed, D_dynamic, ratio D_dynamic / D_fixed (how much fixed route under/over-states damage), '
                         'per source, per article'},
        'routing_statistics': {
            'scope': 'q_dyn vs fp32_dyn, every layer, positions 0-126 (the 127 positions that influence the loss)',
            'set_equal': 'share of tokens whose top-8 expert sets are identical (order ignored)',
            'overlap': '|S_q intersect S_fp32| / 8, mean',
            'dispatch': 'per layer and expert token counts; total variation distance of the dispatch distributions; '
                        'experts with zero tokens in both are excluded from relative changes',
            'weight_l1': 'routing weights aligned by expert id, unselected = 0, L1 over the 64 experts, mean per token',
            'margin': 'FP32 logit(rank 8) - logit(rank 9); set-change rate by pooled margin quintiles; FP32 rank of the '
                      'experts that enter the quantized top-8',
            'per_article': 'mean set-change share / overlap / weight L1 per article vs Delta_route (Spearman, descriptive)',
            'caveat': 'descriptive only; tokens are not independent samples; more route change is not proof of more damage'},
        'not_done': 'no first-order term, no knee re-search, no config change from evaluation loss, no article filtering, '
                    'no calibration / range / quantizer change, no per-layer intervention or format ablation',
        'seeds': SEEDS, 'inputs': inputs,
        'model': {'revision': oc.REVISION, 'files_sha256': {f.name: sha(f) for f in model_files}},
        'code': {s: sha(HERE / s) for s in SCRIPTS},
        'environment': env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 固定 BC 配置的 fixed-route／dynamic-route 對照', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 配置', '', f'- `{p["configuration"]["path"]}`（sha256 `{p["configuration"]["sha256"][:12]}…`）：search_final B 修正後 MILP knee，R 0.7071056。',
          '- 理由：OLMoE 的預設 pipeline 仍是 `mp_auto.auto_select` + `mp_milp`；2026-10-16 整合的最小修正就是 search_final B 的修正後 MILP，其 knee 即此檔，並已在 512 個新窗口驗證。不用 bc_p125（修正前路徑）與 LP knee（研究求解器）。在任何新評估前記錄。',
          '- 量化：DevQuant + 修正後 int activation（與 search_final C 相同）；ranges 為 multipoint 的 ranges.json；21 個未搜尋投影為 bf16 權重 + FP32 activation；router、attention、norm、LM head 皆 FP32，不量化 router。', '',
          '## 條件', '', '- fp32_dyn：原模型自行路由（記錄 ids、weights、router top-16 logits）。', '- fp32_fix：原模型，固定 fp32_dyn 的 ids 與 weights（控制組）。',
          '- q_fix：BC 配置，固定 fp32_dyn 的 ids 與 weights（與歷史 fixed-route 相同語意：ids 與 weights 都固定）。', '- q_dyn：BC 配置，FP32 router 在量化模型自身 hidden state 上路由。',
          f'- 四者共用同一函式、同窗口、每批 {CHUNK} 窗口、GPU FP32、TF32 關閉；每窗口 127 個預測 token。', '',
          '## 檢查', ''] + [f'- {c}' for c in p['checks']] + ['',
          '## Pilot 與樣本數', '', f'- Pilot：search_final C 前 {PILOT_WINDOWS} 個窗口（已看過的資料），做歷史重現與所有數值控制，估計 Δ_route 的 4 窗口區塊 SD。',
          '- 規則：n = ceil((1.645 s / (ε/2))²)，限制在 [32, 128] 篇。', '',
          '## 資料', '', '- 嚴格文章邊界：標題行前後都必須是空行（得到 600／60／60 篇），表格或公式中的類標題行併入所在文章。',
          '- 排除先前所有 OLMoE 文本所在的整篇文章（含 D、V、H、J、C1–C8）。計數（不含 route、loss）：validation 10、test 18、train 556 篇可用（≥512 token）。',
          '- 原評估來源優先：validation 與 test 的可用文章全部納入（共 28 篇，非挑選），其餘 n−28 篇自 train 均勻抽樣。分層：validation／test／train，分層 bootstrap 並分開報告。',
          f'- 每篇 4 個連續窗口，起點均勻；seed {SEEDS["data"]}；評估前凍結。', '',
          '## 終點與統計', '', '- 每篇文章：各條件 CE = 4 窗口 summed CE / 508。D_fixed = CE(q_fix) − CE(fp32_dyn)；D_dynamic = CE(q_dyn) − CE(fp32_dyn)。',
          '- 主要終點：Δ_route = CE(q_dyn) − CE(q_fix) 的文章等權平均（每篇 508 token）。正值 = dynamic 額外損傷。',
          f'- {BOOT} 次分層 paired article bootstrap（seed {SEEDS["bootstrap"]}），95% 與 90% 區間；PPL 由平均 CE 取 exp。',
          f'- ε = {EPS} nat/token，沿用 search_final 與 solver regression 的 OLMoE 配置等價界線；為研究容忍界線，非通用標準。',
          '- 判定：90% CI ⊂ [−ε, +ε] → 容忍範圍內等價；90% CI 下界 > +ε → dynamic 額外損傷超過容忍；90% CI 上界 < −ε → dynamic 明顯較好；其餘 → 不確定。95% CI 含 0 不等於等價。', '',
          '## 路由統計（描述）', '', '- q_dyn 對 fp32_dyn，逐層，位置 0–126：top-8 集合完全相同比例；重疊率；各 expert dispatch 變化（TVD）；依 expert 對齊的權重 L1；FP32 第 8／9 名 logit margin 分位與變動率；進入 top-8 的 expert 在 FP32 的名次；每篇文章路由變動與 Δ_route 的關係。',
          '- token 不視為獨立樣本；路由改變多不代表造成損傷。', '',
          '## 不做', '', '- 不加入一階項、不重搜 knee、不依新 loss 改配置、不篩文章、不改 calibration／range／量化器、不做逐層干預或格式消融。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    log('plan written')


def stage_pilot(out, log):
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    from run_olmoe_search_final import dev_quant
    gpu_setup()
    if (out / 'pilot.json').exists():
        raise SystemExit('pilot exists')
    d = out / 'pilot'
    d.mkdir(exist_ok=True)
    toks = np.load(SF / 'C/tokens_eval.npy')[:PILOT_WINDOWS]
    man = read(SF / 'C/routes_manifest.json')['eval']
    truth = read(SF / 'C/truth.json')['summed_ce']
    fm, ranges = load_config(), read(RANGES)
    dq = dev_quant(d, 'cuda', corrected=True)
    res = {'config_checks': config_checks(fm, dq, ranges)}
    # historical reproduction
    h64 = forward(toks, None, dq, ranges, record=True, log=log, label='pilot fp32_dyn chunk64', chunk=64)
    hashes = [oc.route_hash(h64['routes']['ids'][k], h64['routes']['weights'][k]) for k in range(PILOT_WINDOWS)]
    res['historical'] = {'fp32_dyn_chunk64_route_hash_equal': int(sum(x == y for x, y in zip(hashes, man['hash'][:PILOT_WINDOWS]))),
                         'fp32_dyn_chunk64_ce_max_abs_vs_capture': float(np.max(np.abs(np.array(h64['ce']) - np.array(man['summed_ce_dynamic'][:PILOT_WINDOWS]))))}
    rts_c, _, _ = load_routes(SF / 'C', 'eval', oc.config(), 'cuda')
    rows = slice(0, PILOT_WINDOWS * SEQ)
    rts_c = [(i[rows].clone(), w[rows].clone()) for i, w in rts_c]
    qc = forward(toks, fm, dq, ranges, routes=rts_c, log=log, label='pilot q_fix C routes')
    fc = forward(toks, None, dq, ranges, routes=rts_c, log=log, label='pilot fp32_fix C routes')
    res['historical'].update({'q_fix_ce_max_abs_vs_C_truth_milp': float(np.max(np.abs(np.array(qc['ce']) - np.array(truth['milp'][:PILOT_WINDOWS])))),
                              'fp32_fix_ce_max_abs_vs_C_truth_fp32': float(np.max(np.abs(np.array(fc['ce']) - np.array(truth['fp32'][:PILOT_WINDOWS])))),
                              'q_fix_audit': qc['audit']})
    # internal controls at the main chunking
    f_dyn = forward(toks, None, dq, ranges, record=True, router_top=16, log=log, label='pilot fp32_dyn')
    rts = routes_to_dev(f_dyn['routes'])
    f_fix = forward(toks, None, dq, ranges, routes=rts, log=log, label='pilot fp32_fix')
    q_fix = forward(toks, fm, dq, ranges, routes=rts, log=log, label='pilot q_fix')
    q_dyn = forward(toks, fm, dq, ranges, record=True, log=log, label='pilot q_dyn')
    q_dyn2 = forward(toks, fm, dq, ranges, record=True, log=log, label='pilot q_dyn rerun')
    q_own = forward(toks, fm, dq, ranges, routes=routes_to_dev(q_dyn['routes']), log=log, label='pilot q_fix own routes')
    res['controls'] = {
        'fp32_fix_vs_fp32_dyn': same(f_fix, f_dyn),
        'fp32_dyn_chunk128_vs_chunk64_routes_equal_windows': int(sum(route_hash({k: v[i:i + 1] for k, v in f_dyn['routes'].items()}) ==
                                                                      route_hash({k: v[i:i + 1] for k, v in h64['routes'].items()}) for i in range(PILOT_WINDOWS))),
        'fp32_dyn_chunk128_vs_chunk64_ce_max_abs': float(np.max(np.abs(np.array(f_dyn['ce']) - np.array(h64['ce'])))),
        'q_dyn_rerun': same(q_dyn2, q_dyn), 'q_dyn_rerun_routes_equal': route_hash(q_dyn2['routes']) == route_hash(q_dyn['routes']),
        'q_fix_own_routes_vs_q_dyn': same(q_own, q_dyn),
        'all_finite': all(x['finite_ce'] for x in (f_dyn, f_fix, q_fix, q_dyn, q_dyn2, q_own)),
        'q_dyn_audit': q_dyn['audit']}
    dr = (np.array(q_dyn['ce']) - np.array(q_fix['ce'])) / TP
    blk = dr.reshape(-1, WPA).mean(1)
    s = float(blk.std(ddof=1))
    n = int(min(N_MAX, max(N_MIN, math.ceil((1.645 * s / (EPS / 2)) ** 2))))
    res['sample_size'] = {'delta_route_window_mean': float(dr.mean()), 'delta_route_window_sd': float(dr.std(ddof=1)),
                          'block_sd': s, 'blocks': len(blk), 'n_required_unclipped': math.ceil((1.645 * s / (EPS / 2)) ** 2), 'n_articles': n}
    res['pilot_D_fixed_mean'] = float(((np.array(q_fix['ce']) - np.array(f_dyn['ce'])) / TP).mean())
    res['pilot_D_dynamic_mean'] = float(((np.array(q_dyn['ce']) - np.array(f_dyn['ce'])) / TP).mean())
    res['seconds'] = {k: v['seconds'] for k, v in (('fp32_dyn_chunk64', h64), ('q_fix_C', qc), ('fp32_fix_C', fc), ('fp32_dyn', f_dyn),
                                                   ('fp32_fix', f_fix), ('q_fix', q_fix), ('q_dyn', q_dyn), ('q_dyn_rerun', q_dyn2), ('q_fix_own', q_own))}
    c = res['controls']; h = res['historical']
    res['pass'] = bool(res['config_checks']['pass'] and h['fp32_dyn_chunk64_route_hash_equal'] == PILOT_WINDOWS
                       and h['fp32_dyn_chunk64_ce_max_abs_vs_capture'] == 0 and h['q_fix_ce_max_abs_vs_C_truth_milp'] == 0
                       and h['fp32_fix_ce_max_abs_vs_C_truth_fp32'] == 0
                       and c['fp32_fix_vs_fp32_dyn']['ce_max_abs'] == 0 and c['fp32_fix_vs_fp32_dyn']['hidden_bitwise_windows'] == PILOT_WINDOWS
                       and c['q_dyn_rerun']['hidden_bitwise_windows'] == PILOT_WINDOWS and c['q_dyn_rerun_routes_equal']
                       and c['q_fix_own_routes_vs_q_dyn']['hidden_bitwise_windows'] == PILOT_WINDOWS and c['all_finite']
                       and c['q_dyn_audit']['router_untouched'] and c['q_dyn_audit']['hook_keys_are_expert_projections'])
    res['environment'] = env_info()
    write(out / 'pilot.json', res)
    log(json.dumps({'pass': res['pass'], 'sample_size': res['sample_size']}))


def stage_data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('data exists')
    pl = read(out / 'pilot.json')
    if not pl['pass']:
        raise SystemExit('pilot controls failed; fix before drawing data')
    n = pl['sample_size']['n_articles']
    for k, v in read(out / 'plan.json')['inputs'].items():
        if sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    used = used_spans_all()
    tok = oc.tokenizer()
    rng = np.random.default_rng(SEEDS['data'])
    segs, toks, avail, rejected = [], [], {}, {}
    pools = {}
    for split in ('validation', 'test', 'train'):
        arts, ids, off, rej = strict_articles(split, tok)
        rejected[split] = rej
        free = [a for a in arts if a['tokens'][1] - a['tokens'][0] >= WPA * SEQ
                and not any(a['chars'][0] < u1 and a['chars'][1] > u0 for u0, u1 in used[split])]
        avail[split] = {'strict_articles': len(arts), 'eligible': len(free)}
        pools[split] = (free, ids, off)
    n_orig = len(pools['validation'][0]) + len(pools['test'][0])
    if n < n_orig:
        raise SystemExit('n below the original-source census; rule says n >= 32 > 28')
    picks = {'validation': list(range(len(pools['validation'][0]))), 'test': list(range(len(pools['test'][0]))),
             'train': sorted(rng.choice(len(pools['train'][0]), n - n_orig, replace=False).tolist())}
    for split in ('validation', 'test', 'train'):
        free, ids, off = pools[split]
        for k in picks[split]:
            a = free[k]
            st = int(rng.integers(a['tokens'][0], a['tokens'][1] - WPA * SEQ + 1))
            wc = [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])]
            assert a['chars'][0] <= wc[0] and wc[1] <= a['chars'][1]
            w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
            toks.append(w)
            segs.append({'index': len(segs), 'split': split, 'stratum': split, 'source': 'original' if split != 'train' else 'train',
                         'title': a['title'], 'article_chars': a['chars'], 'article_tokens': a['tokens'], 'window_chars': wc,
                         'tokens': [st, st + WPA * SEQ], 'windows': [WPA * len(segs) + i for i in range(WPA)],
                         'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
    if len({(s['split'], s['title']) for s in segs}) != len(segs):
        raise SystemExit('duplicate article')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'),
                                     'pilot_sha256': sha(out / 'pilot.json'), 'n_articles': n, 'available': avail,
                                     'rejected_title_lines': {k: len(v) for k, v in rejected.items()}, 'rejected_examples': {k: v[:10] for k, v in rejected.items()},
                                     'segments_sha256': sha(out / 'segments.json'), 'tokens_sha256': sha(out / 'tokens.npy'),
                                     'strata': dict(Counter(s['stratum'] for s in segs))})
    log(f'data frozen: n={n}, strata {dict(Counter(s["stratum"] for s in segs))}, available {avail}')


def check_tokens(out):
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('tokens changed')
    return np.load(out / 'tokens.npy')


def stage_capture(out, log):
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant
    gpu_setup()
    if (out / 'capture.json').exists():
        raise SystemExit('capture exists')
    toks = check_tokens(out)
    (out / 'routes').mkdir(exist_ok=True)
    fm, ranges = load_config(), read(RANGES)
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    r = forward(toks, None, dq, ranges, record=True, router_top=16, log=log, label='fp32_dyn')
    save_routes(out / 'routes/fp32_dyn.npz', r['routes'], r['top'], tokens=toks)
    write(out / 'capture.json', {'ce': r['ce'], 'hidden_sha': r['hidden_sha'], 'seconds': r['seconds'], 'audit': r['audit'],
                                 'route_hash': route_hash(r['routes']), 'routes_sha256': sha(out / 'routes/fp32_dyn.npz'),
                                 'tokens_sha256': sha(out / 'tokens.npy'), 'model_revision': oc.REVISION, 'chunks': r['chunks'],
                                 'finite_ce': r['finite_ce'], 'environment': env_info()})


def stage_eval(out, log):
    from run_olmoe_search_final import dev_quant
    gpu_setup()
    if (out / 'eval.json').exists():
        raise SystemExit('eval exists')
    toks = check_tokens(out)
    cap = read(out / 'capture.json')
    if sha(out / 'routes/fp32_dyn.npz') != cap['routes_sha256']:
        raise SystemExit('routes changed')
    rec = load_routes_npz(out / 'routes/fp32_dyn.npz')
    assert np.array_equal(rec['tokens'], toks)
    fm, ranges = load_config(), read(RANGES)
    (out / 'work').mkdir(exist_ok=True)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    rts = routes_to_dev(rec)
    res = {'config_checks': config_checks(fm, dq, ranges)}
    runs = {}
    runs['fp32_fix'] = forward(toks, None, dq, ranges, routes=rts, log=log, label='fp32_fix')
    runs['q_fix'] = forward(toks, fm, dq, ranges, routes=rts, log=log, label='q_fix')
    runs['q_dyn'] = forward(toks, fm, dq, ranges, record=True, log=log, label='q_dyn')
    save_routes(out / 'routes/q_dyn.npz', runs['q_dyn']['routes'], tokens=toks)
    runs['q_dyn_rerun'] = forward(toks, fm, dq, ranges, record=True, log=log, label='q_dyn rerun')
    runs['q_fix_own_routes'] = forward(toks, fm, dq, ranges, routes=routes_to_dev(runs['q_dyn']['routes']), log=log, label='q_fix own routes')
    runs['fp32_dyn_path_rerun'] = forward(toks, None, dq, ranges, record=True, log=log, label='fp32_dyn rerun')
    capd = {'ce': cap['ce'], 'hidden_sha': cap['hidden_sha']}
    res['controls'] = {
        'fp32_fix_vs_fp32_dyn': same(runs['fp32_fix'], capd),
        'fp32_dyn_rerun_vs_capture': same(runs['fp32_dyn_path_rerun'], capd),
        'fp32_dyn_rerun_routes_equal': route_hash(runs['fp32_dyn_path_rerun']['routes']) == cap['route_hash'],
        'q_dyn_rerun': same(runs['q_dyn_rerun'], runs['q_dyn']),
        'q_dyn_rerun_routes_equal': route_hash(runs['q_dyn_rerun']['routes']) == route_hash(runs['q_dyn']['routes']),
        'q_fix_own_routes_vs_q_dyn': same(runs['q_fix_own_routes'], runs['q_dyn']),
        'all_finite': all(r['finite_ce'] for r in runs.values()) and cap['finite_ce'],
        'q_dyn_route_weights_finite': runs['q_dyn']['audit']['nonfinite_route_weights'] == 0,
        'audit': {k: runs[k]['audit'] for k in ('q_fix', 'q_dyn')}}
    c = res['controls']
    nW = len(toks)
    res['pass'] = bool(res['config_checks']['pass'] and c['fp32_fix_vs_fp32_dyn']['ce_max_abs'] == 0
                       and c['fp32_fix_vs_fp32_dyn']['hidden_bitwise_windows'] == nW and c['fp32_dyn_rerun_routes_equal']
                       and c['q_dyn_rerun']['hidden_bitwise_windows'] == nW and c['q_dyn_rerun_routes_equal']
                       and c['q_fix_own_routes_vs_q_dyn']['hidden_bitwise_windows'] == nW and c['all_finite']
                       and c['q_dyn_route_weights_finite'] and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in c['audit'].values()))
    res['ce'] = {k: runs[k]['ce'] for k in ('fp32_fix', 'q_fix', 'q_dyn')}
    res['ce']['fp32_dyn'] = cap['ce']
    res['seconds'] = {k: v['seconds'] for k, v in runs.items()}
    res['seconds']['fp32_dyn_capture'] = cap['seconds']
    res['q_dyn_routes_sha256'] = sha(out / 'routes/q_dyn.npz')
    res['environment'] = env_info()
    write(out / 'eval.json', res)
    log(json.dumps({'pass': res['pass']}))


# --------------------------------------------------------------------------- #
def stage_analyze(out, log):
    from scipy.stats import spearmanr
    t0 = time.time()
    ev = read(out / 'eval.json')
    if not ev['pass']:
        raise SystemExit('numerical controls failed: do not interpret q_dyn')
    segs = read(out / 'segments.json')
    nA = len(segs)
    ce = {k: np.array(v) for k, v in ev['ce'].items()}
    art = {k: v.reshape(nA, WPA).sum(1) / (WPA * TP) for k, v in ce.items()}
    strata = np.array([s['stratum'] for s in segs])
    rng = np.random.default_rng(SEEDS['bootstrap'])
    idx = np.concatenate([rng.choice(np.where(strata == st)[0], (BOOT, int((strata == st).sum())), replace=True)
                          for st in ('validation', 'test', 'train') if (strata == st).any()], 1)
    quant = {'D_fixed': art['q_fix'] - art['fp32_dyn'], 'D_dynamic': art['q_dyn'] - art['fp32_dyn'], 'Delta_route': art['q_dyn'] - art['q_fix'],
             'fp32_fix_minus_fp32_dyn': art['fp32_fix'] - art['fp32_dyn']}

    def stat(x, rows=None):
        x = x if rows is None else x[rows]
        ii = idx if rows is None else np.concatenate([rng2.choice(np.where(strata[rows] == st)[0], (BOOT, int((strata[rows] == st).sum())), replace=True)
                                                      for st in ('validation', 'test', 'train') if (strata[rows] == st).any()], 1)
        d = x[ii].mean(1)
        return {'mean': float(x.mean()), 'ci95': np.percentile(d, [2.5, 97.5]).tolist(), 'ci90': np.percentile(d, [5, 95]).tolist(), 'n': int(len(x))}
    rng2 = np.random.default_rng(SEEDS['bootstrap'] + 1)
    res = {'n_articles': nA, 'strata': dict(Counter(strata.tolist())), 'endpoints': {k: stat(v) for k, v in quant.items()}}
    dd = res['endpoints']['Delta_route']
    lo, hi = dd['ci90']
    res['decision'] = ('equivalent_within_tolerance' if -EPS <= lo and hi <= EPS else 'dynamic_extra_damage_beyond_tolerance' if lo > EPS
                       else 'dynamic_better_beyond_tolerance' if hi < -EPS else 'uncertain')
    res['ci95_excludes_zero'] = bool(dd['ci95'][0] > 0 or dd['ci95'][1] < 0)
    dfix, ddyn = quant['D_fixed'][idx].mean(1), quant['D_dynamic'][idx].mean(1)
    res['ratio_D_dynamic_over_D_fixed'] = {'point': float(quant['D_dynamic'].mean() / quant['D_fixed'].mean()), 'ci95': np.percentile(ddyn / dfix, [2.5, 97.5]).tolist()}
    tokens = nA * WPA * TP
    res['conditions'] = {k: {'ce': float(ce[k].sum() / tokens), 'ppl': float(np.exp(ce[k].sum() / tokens))} for k in ce}
    for k in ('q_fix', 'q_dyn'):
        res['conditions'][k]['ppl_rel_fp32_dyn'] = res['conditions'][k]['ppl'] / res['conditions']['fp32_dyn']['ppl'] - 1
    res['by_source'] = {}
    for name, rows in (('original_val_test', strata != 'train'), ('train', strata == 'train'), ('validation', strata == 'validation'), ('test', strata == 'test')):
        if rows.any():
            res['by_source'][name] = {k: stat(quant[k], rows) for k in ('D_fixed', 'D_dynamic', 'Delta_route')}
            res['by_source'][name]['ppl'] = {k: float(np.exp(ce[k].reshape(nA, WPA)[rows].sum() / (rows.sum() * WPA * TP))) for k in ce}

    # routing statistics (q_dyn vs fp32_dyn), positions 0..126
    f = load_routes_npz(out / 'routes/fp32_dyn.npz'); q = load_routes_npz(out / 'routes/q_dyn.npz')
    fi, fw, qi, qw = f['ids'][:, :, :TP].astype(np.int64), f['weights'][:, :, :TP], q['ids'][:, :, :TP].astype(np.int64), q['weights'][:, :, :TP]
    tl, ti = f['top_logits'][:, :, :TP], f['top_ids'][:, :, :TP].astype(np.int64)
    inter = (fi[..., :, None] == qi[..., None, :]).any(-1).sum(-1)          # [W, L, T]
    seteq = inter == K
    margin = tl[..., K - 1] - tl[..., K]
    layers = []
    qe = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    mbin = np.digitize(margin, qe)
    for l in range(NL):
        cf = np.bincount(fi[:, l].ravel(), minlength=NE); cq = np.bincount(qi[:, l].ravel(), minlength=NE)
        pf, pq_ = cf / cf.sum(), cq / cq.sum()
        both = (cf > 0) | (cq > 0)
        dense_f = np.zeros((fi.shape[0] * TP, NE), np.float64); dense_q = np.zeros_like(dense_f)
        r_ = np.repeat(np.arange(fi.shape[0] * TP), K)
        np.add.at(dense_f, (r_, fi[:, l].reshape(-1)), fw[:, l].reshape(-1)); np.add.at(dense_q, (r_, qi[:, l].reshape(-1)), qw[:, l].reshape(-1))
        l1 = np.abs(dense_q - dense_f).sum(1)
        ch = ~seteq[:, l]
        layers.append({'layer': l, 'set_equal_share': float(seteq[:, l].mean()), 'overlap_mean': float(inter[:, l].mean() / K),
                       'dispatch_tvd': float(0.5 * np.abs(pq_ - pf).sum()), 'experts_used_fp32': int((cf > 0).sum()), 'experts_used_q': int((cq > 0).sum()),
                       'max_abs_dispatch_share_change': float(np.abs(pq_ - pf)[both].max()),
                       'weight_l1_mean': float(l1.mean()), 'weight_l1_unchanged_set_mean': float(l1[~ch.reshape(-1)].mean()) if (~ch).any() else None,
                       'margin_median_changed': float(np.median(margin[:, l][ch])) if ch.any() else None,
                       'margin_median_unchanged': float(np.median(margin[:, l][~ch])),
                       'change_rate_bottom_margin_quintile': float(ch[mbin[:, l] == 0].mean()) if (mbin[:, l] == 0).any() else None,
                       'change_rate_top_margin_quintile': float(ch[mbin[:, l] == 4].mean()) if (mbin[:, l] == 4).any() else None})
    # FP32 rank of experts entering the quantized top-8
    enter = (~(qi[..., :, None] == fi[..., None, :]).any(-1))                  # [W, L, T, K] q experts not in fp32 set
    rk = (qi[..., :, None] == ti[..., None, :])                                 # [W, L, T, K, 16]
    rank = np.where(rk.any(-1), rk.argmax(-1) + 1, 99)
    er = rank[enter]
    res['routing'] = {'layers': layers, 'overall_set_equal_share': float(seteq.mean()), 'overall_overlap': float(inter.mean() / K),
                      'change_rate_by_margin_quintile': [float((~seteq)[mbin == b].mean()) for b in range(5)],
                      'margin_quintile_edges': qe.tolist(),
                      'entering_expert_fp32_rank': {'swaps': int(er.size), 'rank9': float((er == 9).mean()) if er.size else None,
                                                    'rank10_12': float(((er >= 10) & (er <= 12)).mean()) if er.size else None,
                                                    'rank13_16': float(((er >= 13) & (er <= 16)).mean()) if er.size else None,
                                                    'beyond16': float((er == 99).mean()) if er.size else None}}
    per_art = []
    chg = (~seteq).reshape(nA, WPA, NL, TP).mean((1, 2, 3))
    ov = (inter / K).reshape(nA, WPA, NL, TP).mean((1, 2, 3))
    for k_, s in enumerate(segs):
        per_art.append({'index': s['index'], 'stratum': s['stratum'], 'title': s['title'],
                        **{f'ce_{c}': float(art[c][k_]) for c in ('fp32_dyn', 'fp32_fix', 'q_fix', 'q_dyn')},
                        'D_fixed': float(quant['D_fixed'][k_]), 'D_dynamic': float(quant['D_dynamic'][k_]), 'Delta_route': float(quant['Delta_route'][k_]),
                        'route_set_change_share': float(chg[k_]), 'route_overlap': float(ov[k_])})
    with open(out / 'per_article.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_art[0])); w.writeheader(); w.writerows(per_art)
    with open(out / 'per_layer_routing.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(layers[0])); w.writeheader(); w.writerows(layers)
    res['article_route_change_vs_delta'] = {'spearman_set_change': float(spearmanr(chg, quant['Delta_route']).statistic),
                                            'spearman_overlap': float(spearmanr(ov, quant['Delta_route']).statistic),
                                            'note': 'descriptive, articles as units'}
    res['articles_delta_positive'] = int((quant['Delta_route'] > 0).sum())
    res['controls'] = ev['controls']; res['config_checks'] = ev['config_checks']
    res['timing'] = {'eval_seconds': ev['seconds'], 'analysis_seconds': time.time() - t0,
                     'stages': [json.loads(x) for x in open(out / 'timing.jsonl')] if (out / 'timing.jsonl').exists() else []}
    write(out / 'analysis.json', res)
    log(json.dumps({'Delta_route': res['endpoints']['Delta_route'], 'decision': res['decision']}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'pilot', 'data', 'capture', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'pilot': stage_pilot, 'data': stage_data, 'capture': stage_capture, 'eval': stage_eval, 'analyze': stage_analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
