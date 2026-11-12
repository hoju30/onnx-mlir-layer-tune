#!/usr/bin/env python3
"""Qwen3-0.6B, all 28 Down inputs: does the mean-term candidate U_token rank same-bit INT / Posit candidates better than
the original BC mean term U_old on new text and other layers? Spec: reports/qwen3_mean_u_token_validation_prompt_20261025.md.
No criterion / range / quantizer / search / pipeline change, no HVP, no fine-tuning, no native build.

Calibration (original 10 x 128 windows, FP32 graph, one layer's Down input x at a time):
  e = Q_f(x) - x, mu_c = sum_(s,p) e / 1280, g = d(summed CE of each window)/dx
  U_old = sum_(s,c) mu_c^2 (sum_p g)^2,  U_token = sum_(s,p) (sum_c g mu)^2,  F_b = sum g mu
  S_old = U_old / (2 x 1270), S_new = U_token / (2 x 1270), F = F_b / 1270
  selectors per same-bit pair: argmin S_old, argmin S_new, argmin F + S_old, argmin F + S_new (ties -> INT)
New text (128 WT103 articles x 2 windows): dL(alpha) = NLL(x_l + alpha mu) - NLL(FP32), alpha in {+1/8, -1/8, +1, -1},
the complete suffix after the intervention (attention included), every layer and format.
Path for U_token (recorded): 1 = the calibration gradients are saved after the common pass and re-read once mu is known.

Stages (one new --out-dir): plan, calib, freeze, audit, data, eval, analyze.
CPU: ../../venv/bin/python; calib / eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (sampler, read / write / sha / now; standard imports only)

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE, PREP = prs.NATIVE, prs.PREP
PMB = HERE / 'runs/qwen3_posit_mean_bias_20261025'
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
ALPHAS = (0.125, -0.125, 1.0, -1.0)
A0 = 0.125
NL, SEQ, TP, CAL = 28, 128, 127, 10
N, TARGETS = CAL * SEQ, CAL * TP
N_ART, WPA, BOOT = 128, 2, 20000
SEEDS = {'order': 32101, 'start': 32102, 'bootstrap': 32103}
CHUNK, LM_SUB = 32, 8
TIE = 1e-12
KNOWN = (2, 27)
TOL = {'replay_vs_full_abs_summed_ce': 1e-6, 'L27_reproduce_rel': 1e-4, 'alpha0_abs': 0.0, 'grad_forward_vs_nograd_abs': 1e-4,
       'gx_vs_gyW_rel': 1e-4, 'pos127_abs': 0.0}
CODE = ['run_qwen3_mean_u_token_validation.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_l27_reversal_diag.py', 'run_qwen_down_spike_diag.py',
        'run_qwen_down_fp16_protection.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'wt_manifest_scan.py',
        'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py', 'run_qwen3_posit_mean_bias.py']
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def log_stage(out, stage, seconds, extra=None):
    p = out / 'timing.json'
    d = read(p) if p.exists() else {}
    d[stage] = {'seconds': seconds, **(extra or {})}
    write(p, d)


def input_files():
    import run_olmoe_route_mass_restore as mr
    return ([HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] +
            [NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', PREP / 'node_mapping.json', PREP / 'validation_tokens.npy',
             PREP / 'test_tokens.npy', PMB / 'raw.json', PMB / 'report.md', PMB / 'summary_table.csv', PMB / 'controls.json',
             HERE / 'reports/qwen3_mean_u_token_validation_prompt_20261025.md', HERE / 'qwen35_assets/model/tokenizer.json'] + mr.WT103 + mr.WT2)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    if inputs:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def down_nodes():
    m = read(PREP / 'node_mapping.json')
    return {int(p.split('.')[2]): n for n, p in m.items() if p.split('.')[4] == 'down_proj'}


def pick(pair, s, tie):
    return pair[0] if tie or s[pair[0]] < s[pair[1]] else pair[1]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    cfg = read(HERE / 'qwen3_assets/model/config.json')
    dn = down_nodes()
    if cfg['num_hidden_layers'] != NL or sorted(dn) != list(range(NL)):
        raise SystemExit('layer count mismatch')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    prior_runs = sorted(str(p.parent.relative_to(HERE)) for p in HERE.glob('runs/**/segments.json'))
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'spec': 'reports/qwen3_mean_u_token_validation_prompt_20261025.md and the request of 2026-10-25',
        'question': 'does U_token (per-token cross-channel mean term, no cross-position aggregation) rank same-bit INT / Posit '
                    'Down-input candidates better than the original U_old on new text and all layers? extra table-building cost?',
        'structure': 'only L27 is followed by per-position operations; L0-26 are followed by attention (real cross-position '
                     'interactions possible); this round tests whether the candidate transfers, it does not assume it',
        'scope': f'Qwen3-0.6B, all {NL} Down inputs (config and mapping checked), one layer at a time; all weights and other '
                 f'activations FP32; Down output not quantized; formats {list(FORMATS)}; 84 same-bit pairs; original ranges, INT '
                 'QDQ on device, Universal posit LUT; no full quantization, configuration, weight-only, HVP, fine-tuning or native',
        'calibration': {'windows': 'original 10 x 128 (native calib_inputs.npy)', 'e': 'Q_f(x) - x', 'mu': 'sum_(s,p) e / 1280 (float64)',
                        'g': 'gradient of each window\'s summed CE w.r.t. the Down input (full FP32 backward, position 127 = 0)',
                        'U_old': 'sum_(s,c) mu_c^2 (sum_p g_spc)^2', 'U_token': 'sum_(s,p) (sum_c g_spc mu_c)^2', 'F_b': 'sum g mu',
                        'scales': 'S_old = U_old / 2540, S_new = U_token / 2540, F = F_b / 1270'},
        'selectors': {'q_old_even': 'argmin S_old', 'q_new_even': 'argmin S_new', 'q_old_signed': 'argmin F + S_old',
                      'q_new_signed': 'argmin F + S_new', 'scope': 'within each same-bit pair', 'tie': f'|S1 - S2| <= {TIE} max(1, |S1|, '
                      '|S2|) -> INT, listed', 'frozen': 'before any new article; never updated from new text'},
        'cost_path': {'chosen': 'path 1: the common calibration forward / backward saves g of every Down input (float32) to disk; '
                                'after mu is known it is re-read to compute U_token and F',
                      'alternative': 'path 2: obtain mu first, then run the gradient pass (estimated as one more forward / backward)',
                      'native_order': 'run_native_bc_pipeline.native_tables runs the gradient chain and the ranges -> errors chain '
                                      'concurrently; the gradient pass accumulates sum g^2 and per-input sum_p g without mu, so U_token '
                                      'needs either stored per-token gradients or a gradient pass after the errors pass',
                      'recorded': 'common forward / backward, QDQ / mu, U_old, U_token, F, cache write / read, cache size, peak GPU'},
        'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data (verified manifests, '
                 'historical Qwen token audit, WT2 / near-duplicate / previous-article rules, reject any window with >= 50% WT2 word '
                 '5-gram overlap)', 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                 'manifests_seen': prior_runs, 'seeds': SEEDS, 'seed_conflict_check': 'none of 32101/32102/32103 in earlier plans'},
        'evaluation': {'intervention': 'x_l + alpha mu_(l,f) at the Down input of layer l (pre-hook, then the Down linear), '
                                       'complete suffix with attention', 'alphas': list(ALPHAS), 'conditions': f'{NL} x {len(FORMATS)} x '
                                       f'{len(ALPHAS)} = {NL * len(FORMATS) * len(ALPHAS)} plus the FP32 baseline',
                       'note': 'an artificial mean shift with the calibration mu, not a full Posit quantization of new text'},
        'statistics': {'unit': 'article (2 windows averaged, 254 targets)', 'E': '[dL(+a) + dL(-a)] / 2', 'O': '[dL(+a) - dL(-a)] / 2',
                       'primary_A': 'G_even(k) = mean over P (81 pairs of L0-26) of E_(k,q_new_even)(1/8) - E_(k,q_old_even)(1/8); '
                                    '95% CI; also divided by alpha0^2 as a coefficient scale (not a deployment NLL)',
                       'secondary_B': 'G_signed(k) = mean over P of dL_(k,q_new_signed)(+1) - dL_(k,q_old_signed)(+1); 95% CI; '
                                      'Bonferroni 97.5% CIs if a joint either-improves statement is made',
                       'supplementary': 'all 84 pairs, the 3 L27 pairs, the 78 pairs without L2 and L27 (pre-specified)',
                       'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}), one index set for everything',
                       'per_pair': 'INT - Posit of E(1/8), E(1), dL(+1) with 95% CI; E(1/8): null-centred bootstrap two-sided p, Holm '
                                   'over 84; right / wrong / undetermined by the raw 95% CI (Holm also reported); fixes and new errors',
                       'other': 'O(1/8)/(1/8) vs the frozen F (finite odd response, not used to reselect); ties listed; '
                                'undetermined pairs stay in R'},
        'reading': ['G_even CI supports improvement, gains not limited to old failures, no clear new errors -> U_token is a candidate '
                    'mean-curvature term for the next round (no pipeline change)',
                    'only G_even improves -> curvature scoring progress only; first-order / finite problems remain',
                    'gains concentrated in L2 / L27 -> local conclusion only',
                    'clear new errors or unconfirmed gain -> not adopted, no coefficient scan; report failure / undetermined',
                    'even with both improving: full quantization, variance term and whole configurations are not validated'],
        'controls': ['GPU FP32, TF32 off, deterministic, INT scale / zp on the device, posit LUT vs Universal',
                     'correct layer / slot, weights FP32, no output QDQ; the mean shift uses the fixed mu; conditions reproducible',
                     'summed-CE gradients, per-target scaling only at the end; mu denominator 1280',
                     f'every calibration condition: replay from the cached unperturbed prefix vs the complete forward with the same '
                     f'intervention within {TOL["replay_vs_full_abs_summed_ce"]} summed CE per window',
                     'the intervention is W(x + alpha mu) through the Down linear (never y + alpha W mu)',
                     'alpha = 0 equals the FP32 baseline; the FP32 baseline consistent; non-finite values kept',
                     'alpha = 1/8: fraction of changed elements and rounding residual',
                     f'L27 calibration statistics reproduce runs/qwen3_posit_mean_bias_20261025 within {TOL["L27_reproduce_rel"]} rel: U and '
                     'U_token from the same g (g_y W), F with the previous round\'s gradient path (autograd through the final norm and LM '
                     'head, 2-window sub-batches); the F difference between the two gradient paths is reported, not pass / fail '
                     '(pre-plan smoke: comparing F across the two paths gave up to 4.1e-4 rel, a cancellation-amplified path difference)',
                     'mu / selectors unchanged after the freeze; checkpoints carry code / data / mu / selector hashes'],
        'tolerances': TOL,
        'smoke': smoke or 'none',
        'boundaries': 'stop after this round: no criterion change, no HVP, no variance-term work, no whole-model configuration or native run',
        'inputs': {prs.rel(p): sha(p) for p in input_files()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：U_token 平均項的跨層與新文本驗證', '', f'時間：{pl["time"]}。以 plan.json 為準；寫入後的修改須記 amendment。', '',
          f'- 範圍：Qwen3-0.6B 全部 {NL} 層 Down input，一次一層；權重與其他 activation 全 FP32；Down output 不量化；int4／p4e1、int6／p6e1、int8／p8e1，84 個同位元對。',
          '- 結構：只有 L27 之後是逐位置運算；L0–26 之後有 attention，可能有真實跨位置交互。本輪檢驗候選能否轉移，不預設它正確。',
          '- calibration（原 10×128）：e = Q(x)−x；μ = Σe/1280；g = 每窗口 summed CE 對 Down input 的梯度；U_old = Σ_(s,c)μ²(Σ_p g)²；U_token = Σ_(s,p)(Σ_c gμ)²；F_b = Σgμ；S_old = U_old/2540、S_new = U_token/2540、F = F_b/1270。',
          f'- selector（新文章前凍結）：每對內 argmin S_old、argmin S_new、argmin F+S_old、argmin F+S_new；tie（|ΔS| ≤ {TIE}·max(1,|S|)）選 INT 並列出。',
          '- 成本路徑：採路徑 1（共同前向／反向保存梯度，μ 算好後重讀）；路徑 2（先 μ 再跑梯度 pass）只作估計。native 目前梯度 pass 與誤差 pass 並行、梯度 pass 不知道 μ。',
          f'- 新文本：{N_ART} 篇新 WT103 × {WPA} 窗口；沿用前輪抽樣規則（verified 掃描排除 {len(used)} 篇、歷史 Qwen 窗口、WT2 窗口 ≥ 50% 拒收）；seed {SEEDS}。',
          f'- 介入：x_l + αμ（α = {list(ALPHAS)}），經真實 Down linear 與含 attention 的完整 suffix；{NL * len(FORMATS) * len(ALPHAS)} 個條件 + FP32 baseline。',
          '- 主要 A：G_even = mean_P[E(q_new_even) − E(q_old_even)]，α0 = 1/8，P = L0–26 的 81 對；95% CI；另報 /α0² 係數尺度。次要 B：G_signed（ΔL(+1)）；若作共同結論另報 Bonferroni 97.5% CI。補充：84 對、L27 3 對、去 L2／L27 的 78 對。',
          f'- 統計：文章為單位（兩窗口平均），paired bootstrap {BOOT} 次（seed {SEEDS["bootstrap"]}）；逐對 INT−Posit 的 E(1/8)、E(1)、ΔL(+1) 與 95% CI；E(1/8) 84 對 Holm；修正／新增誤排；各位元、各層分布；O(1/8)/(1/8) vs 凍結 F；tie 清單。',
          f'- 控制：TF32 關閉、deterministic、量化器驗證；正確 layer／slot、無 output QDQ；calibration 全部條件「重播 vs 完整前向」≤ {TOL["replay_vs_full_abs_summed_ce"]}；α=0；α=1/8 rounding；L27 重現前輪 U／U_token／F（≤ {TOL["L27_reproduce_rel"]}）；checkpoint hash。',
          '- 判讀與停止：依規格第 8 節；完成後停止，不改 criterion、不做 HVP、變異項、完整配置或 native。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; used {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# GPU helpers
# --------------------------------------------------------------------------- #
def make_core(work):
    import run_qwen_l27_reversal_diag as rd
    core = rd.Core(work)
    core.layers = core.M.layers
    core.kw = {}
    return core


def capture_prefix(core, ids):
    """no-grad FP32 forward: per-window summed CE and the hidden state entering every decoder layer."""
    torch = core.torch
    hs, hooks = {}, []
    for b, layer in enumerate(core.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, b=b: hs.__setitem__(b, a[0].detach())))
    hooks.append(core.layers[1].register_forward_pre_hook(lambda m, a, k: core.kw.update(k), with_kwargs=True))
    with torch.no_grad():
        h = core.M.model.model(input_ids=ids, attention_mask=core.M.wrapper.mask, use_cache=False).last_hidden_state
        ce = core.head_ce(h, ids, normed=True)
    for hk in hooks:
        hk.remove()
    return ce, hs


def shifted(core, layer, delta):
    """context: a pre-hook adding delta to the Down input of `layer` (the Down linear then runs as usual)."""
    class _C:
        def __enter__(s):
            s.h = core.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: (a[0] + delta,))

        def __exit__(s, *a):
            s.h.remove()
    return _C()


def replay_ce(core, hs_l, l, ids):
    torch = core.torch
    with torch.no_grad():
        h = hs_l
        for layer in core.layers[l:]:
            o = layer(h, **core.kw)
            h = o[0] if isinstance(o, tuple) else o
        return core.head_ce(h, ids)


def full_ce(core, ids):
    torch = core.torch
    with torch.no_grad():
        h = core.M.model.model(input_ids=ids, attention_mask=core.M.wrapper.mask, use_cache=False).last_hidden_state
        return core.head_ce(h, ids, normed=True)


# --------------------------------------------------------------------------- #
# calibration: statistics, selectors, controls
# --------------------------------------------------------------------------- #
def calib(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    core = make_core(out / 'work')
    torch = core.torch
    torch.cuda.reset_peak_memory_stats()
    dn = down_nodes()
    ranges = read(NATIVE / 'ranges_by_point.json')
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(toks).cuda()
    # common FP32 forward / backward: Down inputs and their gradients for every layer
    t0 = time.perf_counter()
    cap = {}
    hooks = []
    for l in range(NL):
        def hk(m, a, o, l=l):
            a[0].retain_grad()
            o.retain_grad()
            cap[l] = (a[0], o)
        hooks.append(core.layers[l].mlp.down_proj.register_forward_hook(hk))
    emb = core.M.model.model.embed_tokens(ids).detach().requires_grad_(True)
    hlast = core.M.model.model(inputs_embeds=emb, attention_mask=core.M.wrapper.mask, use_cache=False).last_hidden_state
    loss_w = []
    for j in range(0, CAL, LM_SUB):
        z = core.M.model.lm_head(hlast[j:j + LM_SUB])
        loss_w.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids[j:j + LM_SUB, 1:].reshape(-1),
                                                        reduction='none').view(z.shape[0], -1).double().sum(1))
    loss_w = torch.cat(loss_w)
    loss_w.sum().backward()
    for hk_ in hooks:
        hk_.remove()
    X = {l: cap[l][0].detach() for l in range(NL)}
    G = {l: cap[l][0].grad.detach() for l in range(NL)}
    GY = {l: cap[l][1].grad.detach() for l in (0, 13, 27)}
    Y = {l: cap[l][1].detach() for l in (0, 13, 27)}
    del cap, hlast, emb
    torch.cuda.synchronize()
    tm['common_forward_backward_s'] = time.perf_counter() - t0
    ce0, hs = capture_prefix(core, ids)
    ctl = {'grad_forward_vs_nograd_max_abs': float((loss_w.detach() - ce0).abs().max()),
           'g_position127_abs_max': max(float(G[l][:, -1].abs().max()) for l in range(NL)),
           'gx_vs_gyW_rel_max': max(float((G[l] - GY[l] @ core.layers[l].mlp.down_proj.weight).abs().max() / G[l].abs().max()) for l in GY),
           'y_equals_linear_x': all(bool(torch.equal(torch.nn.functional.linear(X[l], core.layers[l].mlp.down_proj.weight), Y[l])) for l in Y),
           'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled(),
           'posit_lut': {f: list(core.q.posit[f].verify()) for f in core.q.posit},
           'calibration_tokens_equal_prepare': bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL])),
           'weights_fp32_unmodified': 'the checkpoint weights; no weight quantization anywhere in this round'}
    # path 1: save the gradients, re-read after mu
    t0 = time.perf_counter()
    gdir = out / 'work' / 'grads'
    gdir.mkdir(exist_ok=True)
    for l in range(NL):
        np.save(gdir / f'g_L{l}.npy', G[l].cpu().numpy())
    tm['cache_write_s'] = time.perf_counter() - t0
    cache_bytes = sum((gdir / f'g_L{l}.npy').stat().st_size for l in range(NL))
    t0 = time.perf_counter()
    MU = {}
    for l in range(NL):
        rng = ranges[dn[l]]['x']
        core.ranges = {'x': rng}
        with torch.no_grad():
            for f in FORMATS:
                e = core.qact(X[l], f, 'x') - X[l]
                MU[(l, f)] = e.double().sum((0, 1)) / N
    torch.cuda.synchronize()
    tm['qdq_and_mu_s'] = time.perf_counter() - t0
    del G
    t0 = time.perf_counter()
    Gr = {l: torch.from_numpy(np.load(gdir / f'g_L{l}.npy')).cuda().double() for l in range(NL)}
    torch.cuda.synchronize()
    tm['cache_read_s'] = time.perf_counter() - t0
    sc = {}
    for l in range(NL):
        g = Gr[l]
        t0 = time.perf_counter()
        gsum2 = g.sum(1) ** 2                                          # [s, c]  (what the native gradient pass accumulates)
        for f in FORMATS:
            sc[(l, f)] = {'U_old': float((gsum2 * MU[(l, f)] ** 2).sum())}
        torch.cuda.synchronize()
        tm['U_old_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for f in FORMATS:
            sc[(l, f)]['U_token'] = float(((g * MU[(l, f)]).sum(2) ** 2).sum())
        torch.cuda.synchronize()
        tm['U_token_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for f in FORMATS:
            sc[(l, f)]['F_b'] = float((g * MU[(l, f)]).sum())
        torch.cuda.synchronize()
        tm['F_s'] += time.perf_counter() - t0
    for k, v in sc.items():
        v.update(S_old=v['U_old'] / (2 * TARGETS), S_new=v['U_token'] / (2 * TARGETS), F=v['F_b'] / TARGETS)
    # L27 reproduction of the previous round: U / U_token from the same g (g_y W); its F came from the autograd gradient
    # through the final norm and LM head in 2-window sub-batches, recomputed here by that path for an exact comparison
    prev = read(PMB / 'raw.json')
    _, mid27, x27, _ = core.capture(ids)
    W27 = core.layers[27].mlp.down_proj.weight.detach()
    g_prev_path = torch.empty_like(x27)
    for j in range(0, CAL, 2):
        xj = x27[j:j + 2].clone().requires_grad_(True)
        zz = core.M.model.lm_head(core.M.model.model.norm(mid27[j:j + 2] + torch.nn.functional.linear(xj, W27)))
        lj = torch.nn.functional.cross_entropy(zz[:, :-1].reshape(-1, zz.shape[-1]).float(), ids[j:j + 2, 1:].reshape(-1), reduction='sum')
        g_prev_path[j:j + 2], = torch.autograd.grad(lj, xj)
        del zz, lj
    rep, fpath = {}, {}
    for f in FORMATS:
        pu, pt = prev['scores'][f]['U'], prev['scores'][f]['U_token']
        pf = float(np.sum(prev['F_per_window'][f'{f}|all']))
        f_prev_path = float((g_prev_path.double() * MU[(27, f)]).sum())
        rep[f] = {'U_rel': abs(sc[(27, f)]['U_old'] - pu) / abs(pu), 'U_token_rel': abs(sc[(27, f)]['U_token'] - pt) / abs(pt),
                  'F_same_path_rel': abs(f_prev_path - pf) / abs(pf)}
        fpath[f] = {'F_this_round': sc[(27, f)]['F_b'], 'F_previous_gradient_path': f_prev_path,
                    'rel': abs(sc[(27, f)]['F_b'] - f_prev_path) / abs(f_prev_path)}
    ctl['L27_reproduction'] = rep
    ctl['L27_F_gradient_path_sensitivity'] = fpath
    del g_prev_path, mid27, x27
    # selectors
    sel, ties = {}, []
    for l in range(NL):
        for a, b in PAIRS:
            key = f'L{l}|{a}-{b}'
            row = {}
            for name, fn in (('old_even', lambda v: v['S_old']), ('new_even', lambda v: v['S_new']),
                             ('old_signed', lambda v: v['F'] + v['S_old']), ('new_signed', lambda v: v['F'] + v['S_new'])):
                s = {f: fn(sc[(l, f)]) for f in (a, b)}
                tie = abs(s[a] - s[b]) <= TIE * max(1.0, abs(s[a]), abs(s[b]))
                row[name] = pick((a, b), s, tie)
                if tie:
                    ties.append({'pair': key, 'selector': name, 'scores': s})
            sel[key] = row
    # replay vs complete forward on calibration, alpha = 0, rounding at alpha = 1/8
    t0 = time.perf_counter()
    rep_ce, full_dev, a0dev, rnd = {}, 0.0, 0.0, {}
    for l in range(NL):
        z0 = replay_ce(core, hs[l], l, ids)
        a0dev = max(a0dev, float((z0 - ce0).abs().max()))
        for f in FORMATS:
            mu32 = MU[(l, f)].float()
            for al in ALPHAS:
                delta = (al * MU[(l, f)]).float()
                with shifted(core, l, delta):
                    r_ = replay_ce(core, hs[l], l, ids)
                    c_ = full_ce(core, ids)
                full_dev = max(full_dev, float((r_ - c_).abs().max()))
                rep_ce[f'L{l}|{f}|{al:g}'] = (r_ - ce0).cpu().numpy()
            d8 = (A0 * MU[(l, f)]).float()
            xp = X[l] + d8
            ideal = X[l].double() + A0 * MU[(l, f)]
            rnd[f'L{l}|{f}'] = {'changed_fraction': float((xp != X[l]).float().mean()),
                                'rounding_rel': float((xp.double() - ideal).norm() / (A0 * MU[(l, f)]).expand_as(X[l]).norm())}
            del mu32
        log(f'calibration L{l}: replay / complete checked (max dev {full_dev:.2e})')
    torch.cuda.synchronize()
    tm['calibration_replay_check_s'] = time.perf_counter() - t0
    ctl.update({'replay_vs_full_max_abs': full_dev, 'alpha0_replay_vs_baseline_max_abs': a0dev, 'rounding_alpha_1_8': rnd,
                'rounding_min_changed_fraction': min(v['changed_fraction'] for v in rnd.values()),
                'rounding_max_rel': max(v['rounding_rel'] for v in rnd.values()), 'ties': ties,
                'nonfinite': int(sum((~np.isfinite(v)).sum() for v in rep_ce.values()))})
    ctl['passed'] = bool(ctl['grad_forward_vs_nograd_max_abs'] <= TOL['grad_forward_vs_nograd_abs'] and ctl['g_position127_abs_max'] == 0
                         and ctl['gx_vs_gyW_rel_max'] <= TOL['gx_vs_gyW_rel'] and ctl['y_equals_linear_x'] and ctl['tf32'] == [False, False]
                         and ctl['deterministic'] and all(v[0] == 0 for v in ctl['posit_lut'].values()) and ctl['calibration_tokens_equal_prepare']
                         and all(max(v.values()) <= TOL['L27_reproduce_rel'] for v in rep.values())
                         and full_dev <= TOL['replay_vs_full_abs_summed_ce'] and a0dev == 0 and ctl['nonfinite'] == 0)
    write(out / 'controls_calib.json', ctl)
    np.savez(out / 'mu.npz', **{f'L{l}__{f}': MU[(l, f)].cpu().numpy() for l in range(NL) for f in FORMATS})
    write(out / 'calibration_scores.json', {f'L{l}|{f}': sc[(l, f)] for l in range(NL) for f in FORMATS})
    write(out / 'selectors.json', {'selectors': sel, 'ties': ties, 'tie_rule': f'|S1-S2| <= {TIE} max(1,|S1|,|S2|) -> INT'})
    np.savez(out / 'calibration_losses.npz', **{k.replace('|', '__'): v for k, v in rep_ce.items()}, baseline=ce0.cpu().numpy())
    with open(out / 'calibration_scores.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'U_old', 'U_token', 'F_b', 'S_old', 'S_new', 'F'])
        for l in range(NL):
            for f in FORMATS:
                v = sc[(l, f)]
                wr.writerow([l, f, v['U_old'], v['U_token'], v['F_b'], v['S_old'], v['S_new'], v['F']])
    tm['cache_bytes'] = cache_bytes
    tm['path2_extra_gradient_pass_estimate_s'] = tm['common_forward_backward_s']
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['total_s'] = time.perf_counter() - t_all
    log_stage(out, 'calib', tm['total_s'], dict(tm))
    log(f'calibration controls passed={ctl["passed"]}; L27 rel max {max(max(v.values()) for v in rep.values()):.2e}; '
        f'replay dev {full_dev:.2e}; differing even picks {sum(r["old_even"] != r["new_even"] for r in sel.values())}')
    if not ctl['passed']:
        raise SystemExit('calibration controls failed')


def freeze(out):
    check_plan(out)
    if not read(out / 'controls_calib.json')['passed']:
        raise SystemExit('controls failed')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'mu_sha256': sha(out / 'mu.npz'),
                                'scores_sha256': sha(out / 'calibration_scores.json'), 'selectors_sha256': sha(out / 'selectors.json'),
                                'controls_sha256': sha(out / 'controls_calib.json'), 'script_sha256': sha(HERE / 'run_qwen3_mean_u_token_validation.py')})
    log('frozen')


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    log_stage(out, stage, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# eval (GPU, chunked, resumable)
# --------------------------------------------------------------------------- #
def eval_(out):
    t_all = time.perf_counter()
    check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    for k, f in (('mu_sha256', 'mu.npz'), ('selectors_sha256', 'selectors.json'), ('scores_sha256', 'calibration_scores.json')):
        if sha(out / f) != fz[k]:
            raise SystemExit(f'{f} changed after the freeze')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256']:
        raise SystemExit('data changed after the freeze')
    core = make_core(out / 'work')
    torch = core.torch
    torch.cuda.reset_peak_memory_stats()
    mz = np.load(out / 'mu.npz')
    MU = {(l, f): torch.from_numpy(mz[f'L{l}__{f}']).cuda() for l in range(NL) for f in FORMATS}
    toks = np.load(out / 'tokens.npy')
    nw = len(toks)
    hdr0 = {'tokens_sha256': dz['tokens_sha256'], 'script_sha256': sha(HERE / 'run_qwen3_mean_u_token_validation.py'), 'mu_sha256': fz['mu_sha256'],
            'selectors_sha256': fz['selectors_sha256'], 'alphas': list(ALPHAS), 'formats': list(FORMATS)}
    tm = defaultdict(float)
    for k, c0 in enumerate(range(0, nw, CHUNK)):
        ck = out / 'work' / f'eval_chunk{k:02d}.npz'
        hdr = hdr0 | {'chunk': k, 'windows': [c0, min(nw, c0 + CHUNK)]}
        if ck.exists():
            if json.loads(str(np.load(ck)['header'])) != hdr:
                raise SystemExit(f'{ck}: checkpoint from another version / data')
            log(f'chunk {k}: reusing checkpoint')
            continue
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).cuda()
        t1 = time.perf_counter()
        base, hs = capture_prefix(core, ids)
        tm['prefix_forward_s'] += time.perf_counter() - t1
        dl = np.full((NL, len(FORMATS), len(ALPHAS), len(ids)), np.nan)
        a0 = {}
        t1 = time.perf_counter()
        for l in range(NL):
            if l in (0, 13, 27):
                a0[l] = float((replay_ce(core, hs[l], l, ids) - base).abs().max())
            for fi, f in enumerate(FORMATS):
                for ai, al in enumerate(ALPHAS):
                    with shifted(core, l, (al * MU[(l, f)]).float()):
                        dl[l, fi, ai] = (replay_ce(core, hs[l], l, ids) - base).cpu().numpy()
        tm['replay_s'] += time.perf_counter() - t1
        np.savez(ck, dl=dl, base=base.cpu().numpy(), alpha0=json.dumps(a0), header=json.dumps(hdr))
        del hs
        log(f'chunk {k + 1}/{-(-nw // CHUNK)}: {time.perf_counter() - t1:.0f}s; alpha0 {a0}; nonfinite {int((~np.isfinite(dl)).sum())}')
    parts = [np.load(out / 'work' / f'eval_chunk{k:02d}.npz') for k in range(-(-nw // CHUNK))]
    dl = np.concatenate([p['dl'] for p in parts], axis=-1)
    a0all = [json.loads(str(p['alpha0'])) for p in parts]
    np.savez(out / 'eval_dl.npz', dl=dl, base=np.concatenate([p['base'] for p in parts]))
    write(out / 'controls_eval.json', {'alpha0_replay_vs_baseline': a0all, 'alpha0_max_abs': max(max(v.values()) for v in a0all),
                                       'nonfinite': int((~np.isfinite(dl)).sum()), 'passed': bool(max(max(v.values()) for v in a0all) == 0)})
    log_stage(out, 'eval', time.perf_counter() - t_all, dict(tm) | {'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
                                                                     'windows': nw, 'eval_dl_sha256': sha(out / 'eval_dl.npz')})


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[j]))
        adj[j] = run
    return adj


def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    z = np.load(out / 'eval_dl.npz')
    dl = z['dl'] / TP                                                   # [28, 6, 4, windows] per token
    na = dl.shape[-1] // WPA
    A = dl.reshape(NL, len(FORMATS), len(ALPHAS), na, WPA).mean(-1)    # article mean [28, 6, 4, articles]
    ai = {a: i for i, a in enumerate(ALPHAS)}
    fi = {f: i for i, f in enumerate(FORMATS)}
    E8 = (A[:, :, ai[A0]] + A[:, :, ai[-A0]]) / 2
    O8 = (A[:, :, ai[A0]] - A[:, :, ai[-A0]]) / 2
    E1 = (A[:, :, ai[1.0]] + A[:, :, ai[-1.0]]) / 2
    P1 = A[:, :, ai[1.0]]
    sel = read(out / 'selectors.json')['selectors']
    sc = read(out / 'calibration_scores.json')
    rng = np.random.default_rng(SEEDS['bootstrap'])
    W = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na

    def st(x, levels=(0.05,)):
        b = W @ x
        r = {'mean': float(x.mean()), 'se': float(x.std(ddof=1) / np.sqrt(na))}
        for lv in levels:
            r[f'ci{100 * (1 - lv):g}'] = np.percentile(b, [100 * lv / 2, 100 * (1 - lv / 2)]).tolist()
        r['p_null_centred'] = 1.0 if np.all(x == 0) else float((1 + np.sum(np.abs(b - x.mean()) >= abs(x.mean()))) / (BOOT + 1))
        r['decisive'] = bool(r['ci95'][0] > 0 or r['ci95'][1] < 0)
        return r
    rows = []
    for l in range(NL):
        for a, b in PAIRS:
            key = f'L{l}|{a}-{b}'
            s_ = sel[key]
            dE8, dE1, dP1 = E8[l, fi[a]] - E8[l, fi[b]], E1[l, fi[a]] - E1[l, fi[b]], P1[l, fi[a]] - P1[l, fi[b]]
            r = {'layer': l, 'pair': f'{a}-{b}', 'bits': int(a[3:]), **{f'q_{k}': v for k, v in s_.items()},
                 'S_old_int': sc[f'L{l}|{a}']['S_old'], 'S_old_posit': sc[f'L{l}|{b}']['S_old'], 'S_new_int': sc[f'L{l}|{a}']['S_new'],
                 'S_new_posit': sc[f'L{l}|{b}']['S_new'], 'F_int': sc[f'L{l}|{a}']['F'], 'F_posit': sc[f'L{l}|{b}']['F'],
                 'dE8': st(dE8), 'dE1': st(dE1), 'dP1': st(dP1)}
            better8 = a if r['dE8']['mean'] < 0 else b
            betterP = a if r['dP1']['mean'] < 0 else b
            for m in ('old_even', 'new_even'):
                r[f'{m}_status'] = ('right' if s_[m] == better8 else 'wrong') if r['dE8']['decisive'] else 'undetermined'
            for m in ('old_signed', 'new_signed'):
                r[f'{m}_status'] = ('right' if s_[m] == betterP else 'wrong') if r['dP1']['decisive'] else 'undetermined'
            r['O8_over_alpha_minus_F'] = {f: float(O8[l, fi[f]].mean() / A0 - sc[f'L{l}|{f}']['F']) for f in (a, b)}
            rows.append(r)
    for r, pa in zip(rows, holm([r['dE8']['p_null_centred'] for r in rows])):
        r['dE8']['p_holm'] = float(pa)
        r['holm_decisive'] = bool(pa < 0.05)
    # selection gains
    def chosen(metric, m, idx):
        return np.stack([metric[rows[i]['layer'], fi[rows[i][f'q_{m}']]] for i in idx])
    sets = {'P (L0-26, primary)': [i for i, r in enumerate(rows) if r['layer'] <= 26], 'all 84': list(range(len(rows))),
            'L27 (3)': [i for i, r in enumerate(rows) if r['layer'] == 27],
            'without L2 and L27 (78)': [i for i, r in enumerate(rows) if r['layer'] not in KNOWN]}
    for bits in (4, 6, 8):
        sets[f'P {bits}-bit'] = [i for i in sets['P (L0-26, primary)'] if rows[i]['bits'] == bits]
    gains = {}
    for nm, idx in sets.items():
        ge = (chosen(E8, 'new_even', idx) - chosen(E8, 'old_even', idx)).mean(0)
        gs = (chosen(P1, 'new_signed', idx) - chosen(P1, 'old_signed', idx)).mean(0)
        gains[nm] = {'pairs': len(idx), 'G_even': st(ge, (0.05, 0.025)), 'G_even_coefficient_scale_mean': float(ge.mean() / A0 ** 2),
                     'G_signed': st(gs, (0.05, 0.025)),
                     'differing_even': sum(rows[i]['q_old_even'] != rows[i]['q_new_even'] for i in idx),
                     'differing_signed': sum(rows[i]['q_old_signed'] != rows[i]['q_new_signed'] for i in idx)}
    # counts
    def counts(idx, kind):
        o, n_ = ('old_even', 'new_even') if kind == 'even' else ('old_signed', 'new_signed')
        c = {'pairs': len(idx), 'decisive': sum(rows[i][f'{o}_status'] != 'undetermined' for i in idx),
             'old_right': sum(rows[i][f'{o}_status'] == 'right' for i in idx), 'new_right': sum(rows[i][f'{n_}_status'] == 'right' for i in idx),
             'fixes': [f"L{rows[i]['layer']} {rows[i]['pair']}" for i in idx if rows[i][f'{n_}_status'] == 'right' and rows[i][f'{o}_status'] == 'wrong'],
             'new_errors': [f"L{rows[i]['layer']} {rows[i]['pair']}" for i in idx if rows[i][f'{n_}_status'] == 'wrong' and rows[i][f'{o}_status'] == 'right']}
        if kind == 'even':
            hd = [i for i in idx if rows[i]['holm_decisive']]
            c['holm_decisive'] = len(hd)
            c['holm_fixes'] = [f"L{rows[i]['layer']} {rows[i]['pair']}" for i in hd if rows[i][f'{n_}_status'] == 'right' and rows[i][f'{o}_status'] == 'wrong']
            c['holm_new_errors'] = [f"L{rows[i]['layer']} {rows[i]['pair']}" for i in hd if rows[i][f'{n_}_status'] == 'wrong' and rows[i][f'{o}_status'] == 'right']
        return c
    cnt = {nm: {'even': counts(idx, 'even'), 'signed': counts(idx, 'signed')} for nm, idx in sets.items()}
    diff_rows = [{'pair': f"L{r['layer']} {r['pair']}", 'q_old_even': r['q_old_even'], 'q_new_even': r['q_new_even'], 'dE8': r['dE8'],
                  'old_status': r['old_even_status'], 'new_status': r['new_even_status']} for r in rows if r['q_old_even'] != r['q_new_even']]
    diff_signed = [{'pair': f"L{r['layer']} {r['pair']}", 'q_old_signed': r['q_old_signed'], 'q_new_signed': r['q_new_signed'], 'dP1': r['dP1'],
                    'old_status': r['old_signed_status'], 'new_status': r['new_signed_status']} for r in rows if r['q_old_signed'] != r['q_new_signed']]
    per_layer = {f'L{l}': {'differing_even': sum(r['q_old_even'] != r['q_new_even'] for r in rows if r['layer'] == l),
                           'old_right': sum(r['old_even_status'] == 'right' for r in rows if r['layer'] == l),
                           'new_right': sum(r['new_even_status'] == 'right' for r in rows if r['layer'] == l)} for l in range(NL)}
    of = np.array([v for r in rows for v in r['O8_over_alpha_minus_F'].values()])
    Fv = np.array([sc[f"L{r['layer']}|{f}"]['F'] for r in rows for f in r['pair'].split('-')])
    res = {'units': 'nat/token, article means of 2 windows', 'alpha0': A0, 'gains': gains, 'counts': cnt, 'differing_even_pairs': diff_rows,
           'differing_signed_pairs': diff_signed, 'per_layer': per_layer, 'ties': read(out / 'selectors.json')['ties'],
           'odd_response_vs_F': {'median_abs_diff': float(np.median(np.abs(of))), 'median_abs_F': float(np.median(np.abs(Fv))),
                                 'sign_agreement': int((np.sign(of + Fv) == np.sign(Fv)).sum()), 'n': int(len(Fv))},
           'complete': bool(np.isfinite(dl).all()), 'nonfinite': int((~np.isfinite(dl)).sum())}
    g, gsg = gains['P (L0-26, primary)']['G_even'], gains['P (L0-26, primary)']['G_signed']
    res['reading'] = {'G_even': 'improvement supported (CI < 0)' if g['ci95'][1] < 0 else 'worse (CI > 0)' if g['ci95'][0] > 0 else 'not confirmed (CI covers 0)',
                      'G_signed': 'improvement supported (CI < 0)' if gsg['ci95'][1] < 0 else 'worse (CI > 0)' if gsg['ci95'][0] > 0 else 'not confirmed (CI covers 0)'}
    write(out / 'stats.json', res)
    write(out / 'pairs.json', rows)
    with open(out / 'pairs.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'pair', 'q_old_even', 'q_new_even', 'q_old_signed', 'q_new_signed', 'dE8_mean', 'dE8_lo', 'dE8_hi', 'dE8_p_holm',
                     'dE1_mean', 'dE1_lo', 'dE1_hi', 'dP1_mean', 'dP1_lo', 'dP1_hi', 'old_even', 'new_even', 'old_signed', 'new_signed',
                     'S_old_int', 'S_old_posit', 'S_new_int', 'S_new_posit', 'F_int', 'F_posit'])
        for r in rows:
            wr.writerow([r['layer'], r['pair'], r['q_old_even'], r['q_new_even'], r['q_old_signed'], r['q_new_signed'], r['dE8']['mean'], *r['dE8']['ci95'],
                         r['dE8']['p_holm'], r['dE1']['mean'], *r['dE1']['ci95'], r['dP1']['mean'], *r['dP1']['ci95'], r['old_even_status'],
                         r['new_even_status'], r['old_signed_status'], r['new_signed_status'], r['S_old_int'], r['S_old_posit'], r['S_new_int'],
                         r['S_new_posit'], r['F_int'], r['F_posit']])
    segs = read(out / 'segments.json')
    with open(out / 'per_article_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'alpha', 'article', 'title', 'dNLL'])
        for l in range(NL):
            for f in FORMATS:
                for a in ALPHAS:
                    for k in range(na):
                        wr.writerow([l, f, a, k, segs[k]['title'], A[l, fi[f], ai[a], k]])
    with open(out / 'per_window_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'alpha', 'window', 'dCE_summed'])
        for l in range(NL):
            for f in FORMATS:
                for a in ALPHAS:
                    for w in range(dl.shape[-1]):
                        wr.writerow([l, f, a, w, z['dl'][l, fi[f], ai[a], w]])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'gains': {k: {'G_even': v['G_even']['mean'], 'ci': v['G_even']['ci95'], 'G_signed': v['G_signed']['mean'],
                                  'ci_s': v['G_signed']['ci95'], 'diff': v['differing_even']} for k, v in gains.items()},
                    'reading': res['reading']}, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'freeze', 'audit', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note')
    a = ap.parse_args()
    out = a.out_dir.resolve()
    if a.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty')
    elif not (out / 'plan.json').exists():
        raise SystemExit('plan first')
    out.mkdir(parents=True, exist_ok=True)
    for d in ('logs', 'work'):
        (out / d).mkdir(exist_ok=True)
    LOGF = open(out / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage == 'plan':
        plan(out, a.smoke_note)
    elif a.stage in ('audit', 'data'):
        sampler(out, a.stage)
    else:
        {'calib': calib, 'freeze': freeze, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_mean_u_token_validation.py')}) + '\n')


if __name__ == '__main__':
    main()
