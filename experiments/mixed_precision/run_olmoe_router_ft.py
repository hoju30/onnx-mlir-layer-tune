#!/usr/bin/env python3
"""OLMoE: router-only fine-tuning of the frozen q0 configuration (runs/olmoe_proxy_refresh_20261022) on the original
10 x 128 calibration tokens, evaluated against q0, q1 and FP32 on 600 new WikiText-103 articles. Pre-registered in
plan.json / plan.md. Not a GEMQ reproduction; no BC rebuild, no search, no criterion change, no refresh after FT.

Training: only model.layers.*.mlp.gate.weight (the router; not the experts' gate_proj). Each step: a normal dynamic
forward with the current routers (logits -> float32 softmax -> top-8 -> unnormalized weights), then a layer-streamed
backward that recomputes every layer with the same routers; the routing weights keep their graph (gradient to the
router and to earlier hidden states); top-k indices are discrete; activation QDQ uses the identity STE (a surrogate,
not the derivative of the quantizer); q0 expert weights are frozen constants but pass input gradients.

Stages (one --out-dir): plan, controls, train, freeze, data, eval, analyze.
"""
import argparse
import csv
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr

HERE = Path(__file__).resolve().parent
PR = HERE / 'runs/olmoe_proxy_refresh_20261022'
PRC = HERE / 'runs/olmoe_proxy_refresh_confirm_20261022'
S1 = HERE / 'runs/olmoe_fixed_route_20261015'
RANGES = HERE / 'runs/olmoe_multipoint_20261015/ranges.json'
CALIB = S1 / 'tokens_calibration.npy'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CHUNK, CTRL = 600, 128, 128
EPS, BOOT = 0.002, 10000
TRAIN = {'optimizer': 'AdamW', 'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'clip_global_norm': 1.0,
         'updates': 20, 'batch': 'all 10 calibration windows per update (one batch, no microbatching)',
         'loss': 'next-token mean cross-entropy = sum of the 10 windows summed CE / (10 x 127)', 'seed': 22321}
SEEDS = {'order': 22301, 'start': 22302, 'bootstrap': 22311, 'train': 22321}


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


def model_files():
    import olmoe_common as oc
    return sorted(oc.MODEL_DIR.glob('*.safetensors')) + [oc.MODEL_DIR / 'config.json']


def used_wt103_all():
    """content hashes of every WikiText-103 article in an earlier OLMoE manifest (runs/**/*segments*.json, this run excluded).
    run_olmoe_proxy_refresh_confirm.used_wt103 skips its own run directory, so it misses the 600 confirmation articles."""
    import glob as _g
    used, rows = set(), []
    for f in sorted(_g.glob(str(HERE / 'runs/**/*segments*.json'), recursive=True)):
        if 'olmoe_router_ft_' in f:
            continue
        s = read(f)
        srcs = sorted({x.get('source', x.get('split', '?')) for x in s})
        wt103 = any('wikitext-103' in x for x in srcs)
        if wt103:
            used |= {x['content_sha256'] for x in s}
        rows.append({'path': str(Path(f).relative_to(HERE)), 'sha256': sha(f), 'articles': len(s), 'sources': srcs, 'wt103': wt103})
    return used, rows


def input_files():
    return [PR / 'configs/q0.json', PR / 'configs/q1.json', PR / 'freeze.json', RANGES, CALIB, PR / 'eval.json', PR / 'tokens.npy',
            PRC / 'segments.json', PRC / 'plan.json'] + mr.WT103 + mr.WT2


# --------------------------------------------------------------------------- #
# model pieces
# --------------------------------------------------------------------------- #
def init_routers(ck):
    import torch
    return [ck.tensor(f'model.layers.{l}.mlp.gate.weight', 'cuda', torch.float32).clone() for l in range(NL)]


def layer_w(ck, l, fm, dq, ranges, R, tm=None):
    """FP32 layer from the checkpoint, q0 expert weights / activation hooks, router replaced by R[l] (if given)."""
    import torch
    from run_olmoe_search_final import quantized_layer

    def blk(k):
        return tm.block(k) if tm is not None else _Null()
    with blk('checkpoint_load'):
        w = ck.layer(l, 'cuda', torch.float32)
    if fm is None:
        wq, hk = dict(w), None
    else:
        with blk('q0_quantize'):
            wq, hk = quantized_layer(w, fm, l, dq, ranges, NE)
    if R is not None:
        wq['router'] = R[l]
    return wq, hk


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def forward_store(toks, fm, dq, ranges, R, cfg, ck, g, cos, sin, tm=None):
    """normal dynamic forward of the calibration batch (no grad): per-layer inputs, routes, summed CE per window."""
    import torch
    import torch.nn.functional as F
    from run_olmoe_fixed_route import head_ce
    hs, routes = [], []
    with torch.no_grad():
        h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = layer_w(ck, l, fm, dq, ranges, R, tm)
            with (tm.block('train_forward') if tm else _Null()):
                h, idx, rw = pr.layer_tap(wq, h, cfg, cos, sin, None, None, pr.act_map(hk))
            hs.append(h); routes.append((idx, rw))
            del wq, hk
        with (tm.block('train_forward') if tm else _Null()):
            ce = head_ce(g, h, toks, cfg)
    return hs, routes, ce


def backward_stream(toks, fm, dq, ranges, R, cfg, ck, g, cos, sin, hs, routes, tm=None, loss_scale=None):
    """mean-CE loss on the stored final state, then layer-wise recomputation with the same routers (routing weights
    keep their graph; activation QDQ through the identity STE); accumulates R[l].grad; returns loss and checks."""
    import torch
    import olmoe_common as oc
    STEQ = pr.ste_fn()
    nW = toks.shape[0]
    with (tm.block('train_backward') if tm else _Null()):
        hT = hs[-1].clone().requires_grad_(True)
        ce = oc.summed_ce(oc.head(g, hT, cfg), toks)
        loss = ce.sum() / (nW * TP)
        loss.backward()
        gn = hT.grad
    rec_diff, idx_mismatch = [], 0
    for l in reversed(range(NL)):
        wq, hk = layer_w(ck, l, fm, dq, ranges, R, tm)
        with (tm.block('train_backward') if tm else _Null()):
            hi = hs[l].clone().requires_grad_(True)
            o, idx, rw = pr.layer_tap(wq, hi, cfg, cos, sin, None, None, pr.act_map(hk, STEQ))
            idx_mismatch += int((idx != routes[l][0]).sum())
            rec_diff.append(float((o.detach() - hs[l + 1]).abs().max()))
            o.backward(gn)
            gn = hi.grad
        del wq, hk, o, idx, rw
    return float(loss.detach()), rec_diff[::-1], idx_mismatch, gn


def forward_eval(toks_np, fm, dq, ranges, R=None, record=False, chunk=CHUNK, log=print, label=''):
    """run_olmoe_dynamic_route.forward (layer-major over window chunks) with the router optionally replaced."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    cfg, ck = oc.config(), oc.Checkpoint()
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    nW = len(toks_np)
    chunks = [(a, min(nW, a + chunk)) for a in range(0, nW, chunk)]
    rec = {'ids': np.zeros((nW, NL, SEQ, K), np.int16), 'weights': np.zeros((nW, NL, SEQ, K), np.float32)} if record else None
    audit = {'hooks': 0, 'router_replaced': R is not None}
    torch.cuda.synchronize(); t0 = time.time()
    H = {}
    with torch.no_grad():
        for a, b in chunks:
            H[a] = F.embedding(torch.from_numpy(toks_np[a:b]).to('cuda'), g['embed'])
        for l in range(NL):
            wq, hk = layer_w(ck, l, fm, dq, ranges, R)
            audit['hooks'] += len(hk.act) if hk else 0
            act = pr.act_map(hk)
            for a, b in chunks:
                B = b - a
                H[a], idx, rw = pr.layer_tap(wq, H[a], cfg, cos, sin, None, None, act)
                if record:
                    rec['ids'][a:b, l] = idx.view(B, SEQ, K).to(torch.int16).cpu().numpy()
                    rec['weights'][a:b, l] = rw.view(B, SEQ, K).float().cpu().numpy()
            del wq, hk
            log(f'{label}: layer {l} ({time.time() - t0:.0f}s)')
        ce, hh = [], []
        for a, b in chunks:
            ce += head_ce(g, H[a], torch.from_numpy(toks_np[a:b]).to('cuda'), cfg).cpu().numpy().tolist()
            hc = H[a].cpu().numpy()
            hh += [hashlib.sha256(np.ascontiguousarray(hc[k_]).tobytes()).hexdigest() for k_ in range(len(hc))]
        torch.cuda.synchronize()
    return {'ce': ce, 'hidden_sha': hh, 'routes': rec, 'seconds': time.time() - t0, 'audit': audit, 'finite_ce': bool(np.isfinite(ce).all())}


def save_routers(path, R):
    np.savez(path, **{f'L{l}': R[l].detach().cpu().numpy() for l in range(NL)})


def load_routers(path):
    import torch
    z = np.load(path)
    return [torch.from_numpy(z[f'L{l}']).to('cuda') for l in range(NL)]


def setup(out):
    import torch
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    (out / 'work').mkdir(exist_ok=True)
    cfg, ck = oc.config(), oc.Checkpoint()
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    return cfg, ck, dq, cos, sin, g, read(RANGES), read(PR / 'configs/q0.json')


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    fz = read(PR / 'freeze.json')
    chk = {'q0': sha(PR / 'configs/q0.json') == fz['q0_sha256'], 'q1': sha(PR / 'configs/q1.json') == fz['q1_sha256'],
           'ranges': sha(RANGES) == fz['ranges_sha256'], 'calibration': sha(CALIB) == read(PR / 'plan.json')['inputs']['runs/olmoe_fixed_route_20261015/tokens_calibration.npy']}
    if not all(chk.values()):
        raise SystemExit(f'input hashes differ from the previous freeze: {chk}')
    rows, wins = __import__('run_olmoe_proxy_refresh_confirm').manifests()
    used = __import__('run_olmoe_proxy_refresh_confirm').used_wt103()
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    mfiles = {f.name: sha(f) for f in model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'With the q0 formats fixed, does training only the routers improve new-text NLL, and how does it compare '
                    'with the one-refresh configuration q1 in quality and extra offline time?',
        'scope': 'no BC rebuild, no search, no criterion change, no refresh after FT; independent router-only FT trial, not a GEMQ '
                 'reproduction; not integrated into the pipeline',
        'inputs_verified_against_previous_freeze': chk,
        'conditions': {'A': 'FP32 dynamic', 'B': 'q0 dynamic, checkpoint routers', 'C': 'q0 dynamic, routers after FT (step 20)',
                       'D': 'q1 dynamic, checkpoint routers (the full-refresh comparison)'},
        'training': {**TRAIN, 'parameters': 'model.layers.{0..15}.mlp.gate.weight only (64 x 2048 each; the router, not the '
                                            'experts gate_proj)', 'data': 'stage-1 calibration 10 x 128 tokens (sha256 '
                                            f'{sha(CALIB)[:12]})', 'fixed': 'q0 weight / activation datatypes per projection, '
                                            'the q0-quantized checkpoint expert weights, activation ranges / scales / zero-points / '
                                            'posit LUTs, attention, norms, embedding, LM head, the 21 unsearched projections '
                                            '(bf16 / FP32)',
                     'checkpoint': 'the router after the 20th update is the only main result; per-step loss, gradient norm, '
                                   'update size and time logged; nothing chosen from evaluation results',
                     'failure': 'NaN / Inf in loss or gradients stops the run and is reported; no silent change of settings'},
        'gradient': {'forward': 'normal dynamic route with the current routers every step: logits -> float32 softmax -> top-8 -> '
                                'unnormalized weights (norm_topk_prob false); no renormalization',
                     'backward': 'layer-streamed: stored layer inputs, each layer recomputed with the same routers; routing weights '
                                 'keep their graph (gradient to the router and through the router logits to earlier hidden '
                                 'states); top-k indices discrete (no gradient for expert switching); activation QDQ forward = '
                                 'corrected quantizer, backward = identity STE (surrogate); expert weights frozen but input '
                                 'gradients flow; all layer gradients computed before the single optimizer update',
                     'determinism': 'TF32 off, matmul precision highest; torch.use_deterministic_algorithms not enabled '
                                    '(index_add_); reproducibility checked by computing the initial gradient twice'},
        'controls': ['zero-update: training forward with the checkpoint routers == run_olmoe_dynamic_route.forward(q0) on the '
                     'calibration batch (CE, routes, final hidden state bitwise); recomputed layer outputs == stored outputs',
                     'gradient: a 2-layer sub-stack (layers 0-1, 2 calibration windows, q0 + STE): full autograd vs the streaming '
                     'backward, router and input gradients (max relative difference); router gradients finite and nonzero',
                     'determinism: the full initial gradient computed twice is bitwise identical',
                     'parameters: the optimizer holds exactly the 16 router tensors; checkpoint files, q0 / q1 / ranges hashes '
                     'unchanged before / after',
                     'checkpoint: saved and reloaded routers reproduce the post-training calibration CE and routes bitwise; '
                     'evaluation with routers=None still reproduces q0 (no overwrite)',
                     'evaluation: forward_eval == the previous runs (q0 / q1 / FP32 CE on runs/olmoe_proxy_refresh_20261022 '
                     'windows 0-127 bitwise); rerun of the first new chunk for q0+FT bitwise; finite CE',
                     'failure: fix, record and re-register before any conclusion'],
        'evaluation': {'data': f'{N} new WikiText-103 train articles x 4 windows (2400), strict boundaries, drawn after the freeze; '
                               'excluded: WikiText-2 (all), within-WT103 duplicates, every earlier OLMoE WT103 article (incl. the '
                               '128 proxy-refresh and 600 confirmation articles), near-duplicates (>= 10% 5-gram shingles vs '
                               'WikiText-2, earlier or accepted articles), < 512 tokens; the calibration (WikiText-2) is excluded '
                               f'with WikiText-2; order seed {SEEDS["order"]}, window start seed {SEEDS["start"]}',
                       'eligible_before_length_and_neardup': len(elig), 'excluded_previous_wt103': len(used),
                       'manifests_scanned': rows, 'wt2_window_files': wins,
                       'conditions': 'A-D on the same windows, each with its own dynamic route (no FP32 ids / weights / proportions '
                                     'fed back); GPU FP32, TF32 off, chunks of 128 windows, 127 predicted tokens per window',
                       'primary': 'Delta_FT = NLL(q0 + router FT) - NLL(q0); per article NLL = summed CE of 4 windows / 508',
                       'decision': '95% CI upper < 0: FT improves (confirmed); lower > 0: FT worse (confirmed); otherwise direction '
                                   f'not confirmed; equivalence only if the whole 95% CI lies in [-{EPS}, +{EPS}]',
                       'secondary_exploratory': 'Delta_refresh = NLL(q1) - NLL(q0); Delta_FT_vs_refresh = NLL(q0 + FT) - NLL(q1); '
                                                'damage vs FP32; 95% CIs, labelled exploratory; never replacing the primary',
                       'bootstrap': f'{BOOT} paired article bootstraps (one index matrix), seed {SEEDS["bootstrap"]}; PPL = exp(mean NLL)',
                       'fixed': 'n = 600; no extra articles, no subsets, no change of the training after seeing results'},
        'cost': 'timed with CUDA synchronization: checkpoint load and q0 quantization, training forward, backward / recomputation, '
                'optimizer update, router saving; total FT time; peak GPU / CPU; controls and evaluation separately; untimed '
                'remainder reported as unclassified. The 104 s / 108 s figures are historical single measurements.',
        'routing_cost': 'routed token counts per (layer, expert) on the evaluation windows for every condition; actual-route BOPs '
                        'proxy = sum over searched projections of rows x elements x weight bits x activation bits with each '
                        'condition own counts (all 128 positions), relative to all-int8 with the FP32 counts; tokens sent to '
                        'unsearched experts; selected-weight sums, dispatch entropy and maximum expert share (descriptive)',
        'seeds': SEEDS, 'inputs': inputs, 'model_files_sha256': mfiles,
        'code': {f: sha(HERE / f) for f in ['run_olmoe_router_ft.py', 'run_olmoe_proxy_refresh.py', 'run_olmoe_proxy_refresh_confirm.py',
                                            'run_olmoe_route_mass_restore.py'] + dr.SCRIPTS},
        'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：固定 q0、只 fine-tune router', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '- 條件：A FP32 dynamic；B q0；C q0 + router FT（第 20 步）；D q1（完整 refresh 對照）。全部用自己的 dynamic route。',
          '- 只訓練 16 個 router 矩陣（mlp.gate.weight，不是 expert 的 gate_proj）；q0 格式、量化後 expert 權重、ranges、attention／norm／embedding／LM head、未搜尋投影全部固定。',
          '- AdamW，lr 1e-4（固定），betas (0.9, 0.999)，eps 1e-8，weight decay 0，router 全體 global norm clip 1.0，20 次 update，每次用全部 10 個 calibration 窗口，loss = next-token mean CE，無 distillation／load-balancing。第 20 步為唯一主要結果。',
          '- 梯度：每步以當前 router 做 dynamic forward；逐層重算 backward，routing weights 保留梯度（傳到 router 與較前層 hidden state），top-k 為離散選擇；activation QDQ 用 identity STE（surrogate）；expert 權重固定但輸入梯度保留；所有層梯度算完才 update。',
          '- 控制：零更新與 dr.forward 逐位元一致；2 層子堆疊完整 autograd 對逐層 backward；初始梯度算兩次逐位元一致；optimizer 只含 router；checkpoint 存讀重現；評估 forward 重現前次結果。',
          f'- 評估：訓練與凍結後才抽 {N} 篇新 WikiText-103 文章 × 4 窗口；排除 WikiText-2、WT103 內部重複、所有歷史 OLMoE 文章（含前次 128 與 600 篇）、近重複；seed 22301／22302／22311。',
          '- 主要終點：ΔFT = NLL(q0+FT) − NLL(q0)；95% CI 上界 < 0 → 確認改善，下界 > 0 → 確認變差，其他 → 未確認；整段 CI 落在 ±0.002 才算等價。',
          '- 次要（探索）：Δrefresh = NLL(q1) − NLL(q0)、ΔFT_vs_refresh = NLL(q0+FT) − NLL(q1)。',
          '- 成本：同步計時各段；評估與控制另列；實際路由 BOPs proxy、routed token counts 作描述。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    log(f'plan written; eligible {len(elig)}; excluded previous {len(used)}')


def controls(out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    if (out / 'controls.json').exists():
        raise SystemExit('exists')
    cfg, ck, dq, cos, sin, g, ranges, q0 = setup(out)
    torch.manual_seed(SEEDS['train'])
    toks_np = np.load(CALIB)
    toks = torch.from_numpy(toks_np).to('cuda')
    res = {}
    t0 = time.time()
    # 1. zero-update: training forward vs run_olmoe_dynamic_route.forward (same single batch of 10 windows)
    R0 = init_routers(ck)
    hs, routes, ce = forward_store(toks, q0, dq, ranges, R0, cfg, ck, g, cos, sin)
    ref = dr.forward(toks_np, q0, dq, ranges, record=True, log=lambda m: None, label='dr q0', chunk=len(toks_np))
    hsha = [hashlib.sha256(np.ascontiguousarray(hs[-1][k].cpu().numpy()).tobytes()).hexdigest() for k in range(len(toks_np))]
    ids = np.stack([r[0].view(len(toks_np), SEQ, K).to(torch.int16).cpu().numpy() for r in routes], 1)
    ws = np.stack([r[1].view(len(toks_np), SEQ, K).float().cpu().numpy() for r in routes], 1)
    res['zero_update'] = {'ce_max_abs': float(np.max(np.abs(ce.cpu().numpy() - np.array(ref['ce'])))),
                          'hidden_bitwise_windows': int(sum(x == y for x, y in zip(hsha, ref['hidden_sha']))),
                          'routes_equal': bool(np.array_equal(ids, ref['routes']['ids']) and np.array_equal(ws, ref['routes']['weights'])),
                          'initial_mean_ce': float(ce.sum().cpu() / (len(toks_np) * TP)), 'initial_ce_per_window': ce.cpu().numpy().tolist()}
    # 2. determinism + recomputation: full initial gradient twice
    grads = []
    for rep in range(2):
        R = [r.clone().requires_grad_(True) for r in R0]
        hs, routes, _ = forward_store(toks, q0, dq, ranges, R, cfg, ck, g, cos, sin)
        loss, rec, mism, gin = backward_stream(toks, q0, dq, ranges, R, cfg, ck, g, cos, sin, hs, routes)
        grads.append([r.grad.detach().clone() for r in R])
        if rep == 0:
            res['streaming'] = {'loss': loss, 'recompute_max_abs': max(rec), 'top_k_mismatch': mism,
                                'router_grad_norm_per_layer': [float(x.norm()) for x in grads[0]],
                                'router_grad_finite': all(bool(torch.isfinite(x).all()) for x in grads[0]),
                                'router_grad_nonzero_layers': sum(bool((x != 0).any()) for x in grads[0]),
                                'input_grad_finite': bool(torch.isfinite(gin).all())}
        del hs, routes, R
    res['determinism'] = {'gradients_bitwise_equal': all(bool(torch.equal(a, b)) for a, b in zip(*grads))}
    # 3. sub-stack: full autograd vs streaming (layers 0-1, 2 windows)
    STEQ = pr.ste_fn()
    t2 = toks[:2]
    e = F.embedding(t2, g['embed'])
    Ws = [layer_w(ck, l, q0, dq, ranges, None) for l in (0, 1)]
    rvec = torch.randn(2, SEQ, cfg.hidden_size, device='cuda')
    Ra = [ck.tensor(f'model.layers.{l}.mlp.gate.weight', 'cuda', torch.float32).clone().requires_grad_(True) for l in (0, 1)]
    ea = e.clone().requires_grad_(True)
    h = ea
    for i, (wq, hk) in enumerate(Ws):
        wq = dict(wq); wq['router'] = Ra[i]
        h, _, _ = pr.layer_tap(wq, h, cfg, cos, sin, None, None, pr.act_map(hk, STEQ))
    (h * rvec).sum().backward()
    Rb = [ck.tensor(f'model.layers.{l}.mlp.gate.weight', 'cuda', torch.float32).clone().requires_grad_(True) for l in (0, 1)]
    with torch.no_grad():
        hs2 = [e]
        for i, (wq, hk) in enumerate(Ws):
            wq = dict(wq); wq['router'] = Rb[i]
            hs2.append(pr.layer_tap(wq, hs2[-1], cfg, cos, sin, None, None, pr.act_map(hk))[0])
    gn = rvec
    for i in (1, 0):
        wq, hk = Ws[i]; wq = dict(wq); wq['router'] = Rb[i]
        hi = hs2[i].clone().requires_grad_(True)
        o = pr.layer_tap(wq, hi, cfg, cos, sin, None, None, pr.act_map(hk, STEQ))[0]
        o.backward(gn); gn = hi.grad
    rel = lambda a, b: float((a - b).abs().max() / max(float(b.abs().max()), 1e-30))
    res['substack'] = {'router_grad_rel': [rel(Rb[i].grad, Ra[i].grad) for i in (0, 1)], 'input_grad_rel': rel(gn, ea.grad),
                       'router_grad_bitwise': [bool(torch.equal(Rb[i].grad, Ra[i].grad)) for i in (0, 1)],
                       'input_grad_bitwise': bool(torch.equal(gn, ea.grad)), 'router_grad_nonzero': [bool((Ra[i].grad != 0).any()) for i in (0, 1)]}
    del Ws
    res['ste_unit_test'] = pr.ste_unit_test(dq, ranges)
    s, z = res['streaming'], res['zero_update']
    res['pass'] = bool(z['ce_max_abs'] == 0 and z['hidden_bitwise_windows'] == len(toks_np) and z['routes_equal']
                       and s['recompute_max_abs'] == 0 and s['top_k_mismatch'] == 0 and s['router_grad_finite'] and s['router_grad_nonzero_layers'] == NL
                       and s['input_grad_finite'] and res['determinism']['gradients_bitwise_equal']
                       and max(res['substack']['router_grad_rel'] + [res['substack']['input_grad_rel']]) < 1e-5
                       and all(res['substack']['router_grad_nonzero']) and res['ste_unit_test']['pass'])
    res['seconds'] = time.time() - t0
    write(out / 'controls.json', res)
    log(json.dumps({'pass': res['pass'], 'zero_update': z, 'determinism': res['determinism'], 'substack': res['substack']}))


def train(out, log):
    import torch
    if (out / 'router_ft.npz').exists():
        raise SystemExit('exists')
    if not read(out / 'controls.json')['pass']:
        raise SystemExit('controls failed')
    cfg, ck, dq, cos, sin, g, ranges, q0 = setup(out)
    torch.manual_seed(SEEDS['train'])
    torch.cuda.reset_peak_memory_stats()
    toks_np = np.load(CALIB)
    toks = torch.from_numpy(toks_np).to('cuda')
    tm = pr.Timer()
    t_all = time.perf_counter()
    R = [r.requires_grad_(True) for r in init_routers(ck)]
    R_init = [r.detach().clone() for r in R]
    opt = torch.optim.AdamW(R, lr=TRAIN['lr'], betas=TRAIN['betas'], eps=TRAIN['eps'], weight_decay=TRAIN['weight_decay'])
    pids = {id(p) for gr in opt.param_groups for p in gr['params']}
    param_ok = pids == {id(r) for r in R} and len(pids) == NL and all(r.shape == (NE, cfg.hidden_size) for r in R)
    rows = []
    for step in range(1, TRAIN['updates'] + 1):
        ts = {k: v for k, v in tm.t.items()}
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        hs, routes, _ = forward_store(toks, q0, dq, ranges, R, cfg, ck, g, cos, sin, tm)
        loss, rec, mism, _ = backward_stream(toks, q0, dq, ranges, R, cfg, ck, g, cos, sin, hs, routes, tm)
        del hs, routes
        gfin = all(bool(torch.isfinite(r.grad).all()) for r in R)
        if not (math.isfinite(loss) and gfin):
            write(out / 'training_failure.json', {'step': step, 'loss': loss if math.isfinite(loss) else str(loss), 'grad_finite': gfin})
            raise SystemExit(f'non-finite loss / gradient at step {step}')
        with tm.block('optimizer_update'):
            before = [r.detach().clone() for r in R]
            gnorm = float(torch.nn.utils.clip_grad_norm_(R, TRAIN['clip_global_norm']))
            opt.step()
            upd = math.sqrt(sum(float(((r.detach() - b) ** 2).sum()) for r, b in zip(R, before)))
            upd_max = max(float((r.detach() - b).abs().max()) for r, b in zip(R, before))
        rows.append({'step': step, 'loss_before_update': loss, 'grad_global_norm_preclip': gnorm, 'clipped': gnorm > TRAIN['clip_global_norm'],
                     'update_l2': upd, 'update_max_abs': upd_max, 'recompute_max_abs': max(rec), 'top_k_mismatch': mism,
                     'seconds': time.perf_counter() - t0, **{f's_{k}': tm.t[k] - ts.get(k, 0.0) for k in tm.t}})
        log(f'step {step}: loss {loss:.6f} gnorm {gnorm:.4f} upd {upd:.3e} ({rows[-1]["seconds"]:.1f}s)')
    with tm.block('router_save'):
        save_routers(out / 'router_ft.npz', R)
    t_ft = time.perf_counter() - t_all
    with open(out / 'training_log.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # post-training calibration CE (in-memory routers) and checkpoint control (saved -> reloaded)
    with torch.no_grad():
        hs, routes, ce_post = forward_store(toks, q0, dq, ranges, [r.detach() for r in R], cfg, ck, g, cos, sin)
    ids_mem = np.stack([r[0].view(len(toks_np), SEQ, K).to(torch.int16).cpu().numpy() for r in routes], 1)
    hmem = [hashlib.sha256(np.ascontiguousarray(hs[-1][k].cpu().numpy()).tobytes()).hexdigest() for k in range(len(toks_np))]
    del hs, routes
    RL = load_routers(out / 'router_ft.npz')
    fe = forward_eval(toks_np, q0, dq, ranges, RL, record=True, chunk=len(toks_np), log=lambda m: None, label='reload')
    f0 = forward_eval(toks_np, q0, dq, ranges, None, record=False, chunk=len(toks_np), log=lambda m: None, label='q0 no override')
    ref0 = read(out / 'controls.json')['zero_update']
    ckpt = {'reload_ce_max_abs_vs_memory': float(np.max(np.abs(np.array(fe['ce']) - ce_post.cpu().numpy()))),
            'reload_routes_equal_memory': bool(np.array_equal(fe['routes']['ids'], ids_mem)),
            'reload_hidden_bitwise_windows': int(sum(x == y for x, y in zip(fe['hidden_sha'], hmem))),
            'routers_differ_from_checkpoint': [bool((a.cpu() != b.cpu()).any()) for a, b in zip(RL, R_init)],
            'no_override_ce_max_abs_vs_initial': float(np.max(np.abs(np.array(f0['ce']) - np.array(ref0['initial_ce_per_window'])))),
            'post_training_mean_ce': float(ce_post.sum().cpu() / (len(toks_np) * TP)), 'initial_mean_ce': ref0['initial_mean_ce']}
    ckpt['pass'] = bool(ckpt['reload_ce_max_abs_vs_memory'] == 0 and ckpt['reload_routes_equal_memory'] and ckpt['reload_hidden_bitwise_windows'] == len(toks_np)
                        and all(ckpt['routers_differ_from_checkpoint']) and ckpt['no_override_ce_max_abs_vs_initial'] == 0)
    drift = {'relative_frobenius_change_per_layer': [float((RL[l].cpu() - R_init[l].cpu()).norm() / R_init[l].cpu().norm()) for l in range(NL)]}
    write(out / 'train.json', {'param_control': {'optimizer_holds_only_routers': bool(param_ok), 'n_params': NL, 'shape': [NE, cfg.hidden_size]},
                               'checkpoint_control': ckpt, 'router_drift': drift, 'router_sha256': sha(out / 'router_ft.npz'),
                               'training': TRAIN, 'steps': len(rows), 'final_loss_before_update20': rows[-1]['loss_before_update']})
    merge_timing(out, 'fine_tune', {'total_s': t_ft, **dict(tm.t), 'unclassified_s': t_ft - sum(tm.t.values()),
                                    'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': peak_cpu_gb(),
                                    'checkpoint_control_s': fe['seconds'] + f0['seconds']})
    log(json.dumps({'ckpt': ckpt, 'ft_seconds': t_ft}))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    t = read(out / 'train.json')
    if not (t['checkpoint_control']['pass'] and t['param_control']['optimizer_holds_only_routers'] and read(out / 'controls.json')['pass']):
        raise SystemExit('controls failed: not freezing')
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'router_ft_sha256': sha(out / 'router_ft.npz'),
                                'q0_sha256': sha(PR / 'configs/q0.json'), 'q1_sha256': sha(PR / 'configs/q1.json'), 'ranges_sha256': sha(RANGES),
                                'plan_sha256': sha(out / 'plan.json'), 'train_sha256': sha(out / 'train.json'),
                                'code': {f: sha(HERE / f) for f in ['run_olmoe_router_ft.py', 'run_olmoe_proxy_refresh.py'] + dr.SCRIPTS}})
    log('frozen')


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    used, _ = used_wt103_all()
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
    merge_timing(out, 'data', {'seconds': time.perf_counter() - t0, 'peak_cpu_gb': peak_cpu_gb()})
    log(f'data frozen: {N}; skips {dict(skips)}')


def eval_(out, log):
    import torch
    if (out / 'eval.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'freeze.json')
    if sha(out / 'router_ft.npz') != fz['router_ft_sha256'] or sha(PR / 'configs/q0.json') != fz['q0_sha256'] or \
            sha(PR / 'configs/q1.json') != fz['q1_sha256'] or sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('frozen inputs changed')
    cfg, ck, dq, cos, sin, g, ranges, q0 = setup(out)
    q1 = read(PR / 'configs/q1.json')
    RL = load_routers(out / 'router_ft.npz')
    torch.cuda.reset_peak_memory_stats()
    # program control on already-seen windows (control only)
    old_t, old = np.load(PR / 'tokens.npy')[:CTRL], read(PR / 'eval.json')['ce']
    pc = {}
    for c, fm in (('fp32', None), ('q0', q0), ('q1', q1)):
        r = forward_eval(old_t, fm, dq, ranges, None, log=log, label=f'program control {c}')
        pc[c] = float(np.max(np.abs(np.array(r['ce']) - np.array(old[c][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs = {}
    for c, fm, R in (('fp32', None, None), ('q0', q0, None), ('q0ft', q0, RL), ('q1', q1, None)):
        runs[c] = forward_eval(toks, fm, dq, ranges, R, record=True, log=log, label=f'eval {c}')
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = forward_eval(toks[:CTRL], q0, dq, ranges, RL, record=True, log=log, label='rerun q0ft')
    ctl = {'program_control_ce_max_abs': pc,
           'rerun_q0ft_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['q0ft']['ce'][:CTRL])))),
                                      'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['q0ft']['hidden_sha'][:CTRL]))),
                                      'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['q0ft']['routes']['ids'][:CTRL]))},
           'finite': all(r['finite_ce'] for r in runs.values()),
           'q0ft_uses_trained_router': runs['q0ft']['audit']['router_replaced'] and not runs['q0']['audit']['router_replaced'],
           'q0ft_routes_differ_from_q0': float((runs['q0ft']['routes']['ids'] != runs['q0']['routes']['ids']).any(-1).mean())}
    ctl['pass'] = bool(all(v == 0 for v in pc.values()) and ctl['rerun_q0ft_first_chunk']['ce_max_abs'] == 0
                       and ctl['rerun_q0ft_first_chunk']['hidden_bitwise_windows'] == CTRL and ctl['rerun_q0ft_first_chunk']['routes_equal']
                       and ctl['finite'] and ctl['q0ft_uses_trained_router'])
    write(out / 'eval.json', {'ce': {c: r['ce'] for c, r in runs.items()}, 'controls': ctl, 'seconds': {c: r['seconds'] for c, r in runs.items()},
                              'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'environment': dr.env_info()})
    merge_timing(out, 'eval', {**{f'{c}_s': r['seconds'] for c, r in runs.items()}, 'rerun_s': r2['seconds'],
                               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'peak_cpu_gb': peak_cpu_gb()})
    log(json.dumps({'controls_pass': ctl['pass'], 'program_control': pc}))


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    if not ev['controls']['pass']:
        raise SystemExit('evaluation controls failed')
    A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))
    st = lambda x: {'mean': float(x.mean()), 'ci95': np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1)),
                    'better_articles': int((x < 0).sum()), 'worse_articles': int((x > 0).sum())}
    D = {'delta_ft': A['q0ft'] - A['q0'], 'delta_refresh': A['q1'] - A['q0'], 'delta_ft_vs_refresh': A['q0ft'] - A['q1']}
    res = {'n_articles': N, 'primary': st(D['delta_ft']), 'secondary_exploratory': {k: st(v) for k, v in D.items() if k != 'delta_ft'},
           'damage': {c: st(A[c] - A['fp32']) for c in ('q0', 'q0ft', 'q1')}}
    lo, hi = res['primary']['ci95']
    res['decision'] = 'ft_improves_confirmed' if hi < 0 else 'ft_worse_confirmed' if lo > 0 else 'direction_not_confirmed'
    res['equivalence_within_eps'] = bool(-EPS <= lo and hi <= EPS)
    tok = N * WPA * TP
    res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
    # routing cost and descriptives
    from run_olmoe_multipoint import problem
    _, _, names, _, elems, rows_cal, _ = problem(HERE / 'runs/olmoe_multipoint_20261015')
    q0, q1 = read(PR / 'configs/q0.json'), read(PR / 'configs/q1.json')
    cfgs = {'q0': q0, 'q0ft': q0, 'q1': q1}
    counts, desc = {}, {}
    for c in ('fp32', 'q0', 'q0ft', 'q1'):
        z = np.load(out / f'routes_{c}.npz')
        ids = z['ids'].astype(np.int64)
        cnt = np.stack([np.bincount(ids[:, l].ravel(), minlength=NE) for l in range(NL)])
        counts[c] = cnt
        p = cnt / cnt.sum(1, keepdims=True)
        ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
        desc[c] = {'selected_weight_sum_mean': float(z['wsum'].mean()), 'dispatch_entropy_mean_nats': float(ent.mean()),
                   'max_expert_share_mean': float(p.max(1).mean())}
        if c != 'fp32':
            desc[c]['set_change_vs_fp32'] = float((np.sort(ids, -1) != np.sort(np.load(out / 'routes_fp32.npz')['ids'].astype(np.int64), -1)).any(-1).mean())
    searched = {(int(n.split('.')[0][1:]), int(n.split('.')[1][1:])) for n in names}
    rowsof = lambda cnt, n: cnt[int(n.split('.')[0][1:]), int(n.split('.')[1][1:])]
    b8 = sum(rowsof(counts['fp32'], n) * elems[n] * 64 for n in names)
    bops = {c: {'BOPs_actual': float(sum(rowsof(counts[c], n) * elems[n] * FORMAT_BITS[cfgs[c][n]['w']] * FORMAT_BITS[cfgs[c][n]['a']] for n in names)),
                'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l in range(NL) for e in range(NE) if (l, e) not in searched))}
            for c in ('q0', 'q0ft', 'q1')}
    for c in bops:
        bops[c]['relative_to_int8_with_fp32_counts'] = bops[c]['BOPs_actual'] / b8
        bops[c]['search_proxy_BOPs_calibration_rows'] = float(sum(rows_cal[n] * elems[n] * FORMAT_BITS[cfgs[c][n]['w']] * FORMAT_BITS[cfgs[c][n]['a']] for n in names))
    bops['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l in range(NL) for e in range(NE) if (l, e) not in searched))
    res['routing_cost'] = bops
    res['routing_descriptive'] = desc
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']},
             'model_files': {f.name: sha(f) for f in model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    segs = read(out / 'segments.json')
    with open(out / 'per_article.csv', 'w') as fh:
        rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, **{k: float(v[i]) for k, v in D.items()}}
                for i, s in enumerate(segs)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / 'per_window.csv', 'w') as fh:
        rows = [{'window': k, 'article': k // WPA, **{f'summed_ce_{c}': ev['ce'][c][k] for c in ev['ce']}} for k in range(N * WPA)]
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t = read(out / 'train.json')
    write(out / 'controls_summary.json', {'controls': read(out / 'controls.json'), 'param_control': t['param_control'],
                                          'checkpoint_control': t['checkpoint_control'], 'evaluation': ev['controls'],
                                          'input_hashes_unchanged': res['input_hashes_unchanged']})
    merge_timing(out, 'analysis', {'seconds': time.perf_counter() - t0})
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    log(json.dumps({'primary': res['primary'], 'decision': res['decision'], 'secondary': res['secondary_exploratory']}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'controls', 'train', 'freeze', 'data', 'eval', 'analyze'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    t = time.time()
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    {'plan': plan, 'controls': controls, 'train': train, 'freeze': freeze, 'data': data, 'eval': eval_, 'analyze': analyze}[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
