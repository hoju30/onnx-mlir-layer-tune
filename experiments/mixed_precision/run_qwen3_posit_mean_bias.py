#!/usr/bin/env python3
"""Qwen3-0.6B L27 Down input: mechanism diagnostic of the BC mean-bias term U for Posit vs INT activations.
Spec: reports/qwen3_posit_mean_bias_prompt_20261025.md. Calibration only (original 10 x 128 windows); no new articles,
no criterion / quantizer / range / search / pipeline change, no fine-tuning, no native build.

Setting: every weight and activation FP32 (L27 Down weight too), only the L27 Down input x perturbed; Down output not
quantized. For format f:  e = Q_f(x) - x,  mu_c = mean_(s,p) e (n = 10 x 128, computed once),  b = mu (every position),
r = e - b; all outside autograd. Measured (summed next-token CE per window, true FP32 Down -> residual -> final RMSNorm
-> LM head): dL_b = L(x+b)-L(x), dL_r = L(x+r)-L(x), dL_e = L(Q(x))-L(x), I_br = dL_e - dL_b - dL_r.
Scores along the same b: U = sum_(s,c) (sum_p g)^2 mu^2 (original BC bias term), U_elem = sum (g mu)^2,
U_token = sum_(s,p) (sum_c g mu)^2; F_b = g.b; K_b = 1/2 b.H b (autograd HVP, create_graph, b detached).
alpha sweep of b, P0 / Rest split of the fixed b, HVP vs central finite differences of the gradient.

Stages (one new --out-dir): plan, run, analyze. CPU: ../../venv/bin/python; run: PYTHONPATH=qwen35_assets/gpu_deps
/home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_l27_reversal_diag as rd  # noqa: E402  (Core, NODE, constants; standard imports only)

read, write, sha, now = rd.read, rd.write, rd.sha, rd.now
NATIVE, PREP = rd.NATIVE, rd.PREP
REV = HERE / 'runs/qwen3_l27_reversal_diag_20261024'
GEO = HERE / 'runs/qwen3_posit_score_geometry_allbits_20261025'
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
ALPHAS = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625)
HS = (1 / 32, 1 / 64, 1 / 128)
MASKS = ('all', 'P0', 'Rest')
CAL, SEQ, TP = 10, 128, 127
N, TARGETS = CAL * SEQ, CAL * TP
HVP_SUB = 2
TOL = {'replay_vs_full_abs': 0.0, 'old_full_loss_abs': 1e-6, 'old_U_rel': 1e-9, 'gx_vs_gyW_rel': 1e-4, 'pos127_abs': 0.0,
       'alpha0_abs': 0.0, 'U_split_rel': 1e-9, 'F_additivity_rel': 1e-6, 'K_additivity_rel': 1e-3,
       'mean_only_additivity': {'abs_summed_ce': 1e-3, 'rel': 1e-3},
       'hvp_fd': {'adjacent_rel': 0.02, 'adjacent_abs_summed_ce': 1e-3, 'vs_hvp_rel': 0.05, 'vs_hvp_abs_summed_ce': 1e-3},
       'small_alpha_support': {'alpha': 1 / 16, 'rel': 0.2, 'abs_summed_ce': 1e-3},
       'reconstruction_note': 'x+e, x+b+r vs Q(x) floating differences are reported, not pass / fail'}
CODE = ['run_qwen3_posit_mean_bias.py', 'run_qwen_l27_reversal_diag.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_down_spike_diag.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'run_olmoe_dynamic_route.py']
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def inputs():
    return [HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors')] + [
        NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', NATIVE / 'table.json', PREP / 'validation_tokens.npy',
        REV / 'calibration_dL.npz', REV / 'calibration_scores.json', REV / 'stats.json', REV / 'report.md', GEO / 'scores.csv', GEO / 'report.md',
        HERE / 'reports/qwen3_posit_mean_bias_prompt_20261025.md']


def check_plan(out):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    for f, h in pl['inputs'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'input changed: {f}')
    return pl


def plan(out, smoke):
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'spec': 'reports/qwen3_posit_mean_bias_prompt_20261025.md and the request of 2026-10-25 (the request governs)',
        'question': 'does the BC mean-bias term U misrank the Posit mean shift itself at L27 Down input, and is the mechanism (A) a '
                    'curvature-proxy mismatch along the mean direction, (B) a signed first-order effect, (C) finite-perturbation '
                    'nonlinearity, or (D) compensation between the mean and the residual error? several may hold',
        'scope': 'Qwen3-0.6B, L27 mlp.down_proj input, all weights / other activations FP32, output not quantized; formats '
                 f'{list(FORMATS)} (4-bit primary, 6 / 8-bit pre-specified controls, all reported); original 10 x 128 calibration, '
                 'native ranges, INT QDQ on device, Universal-verified posit LUT; no new articles, no generalization claim',
        'decomposition': {'e': 'Q_f(x) - x', 'mu_c': 'sum_(s,p) e / n, n = 1280, float64, computed once', 'b': 'mu_c at every (s,p)',
                          'r': 'e - b', 'autograd': 'e, mu, b, r built outside autograd and detached; no gradient through QDQ, no STE'},
        'measurements': {'conditions': ['x', 'x+b', 'x+r', 'Q(x)', 'x+e (reconstruction)', 'x+b+r (reconstruction)'],
                         'per_window': 'dL_b, dL_r, dL_e (summed next-token CE differences), I_br = dL_e - dL_b - dL_r',
                         'alpha_sweep_b': list(ALPHAS), 'derived': 'O(a) = (dL(a) - dL(-a)) / 2, E(a) = (dL(a) + dL(-a)) / 2; residual '
                                                                  'R(a) = dL(a) - a F_b - a^2 K_b; O/a vs F_b, E/a^2 vs K_b',
                         'small_alpha': 'fraction of x elements changed by x + a b and the rounding residual ||(x+ab) - x - ab|| / ||ab||',
                         'positions': 'b_P0 = M_P0 b, b_Rest = M_Rest b (same mu): F, K, dL at a = +1 and -1; U0, UR, Ucross from '
                                      'A_sc = g_(s,0,c), B_sc = sum_(p>0) g_spc'},
        'scores': {'U': 'sum_(s,c) (sum_p g_spc)^2 mu_c^2 (original BC bias term)', 'U_elem': 'sum_(s,p,c) (g_spc mu_c)^2',
                   'U_token': 'sum_(s,p) (sum_c g_spc mu_c)^2', 'F_b': 'g.b', 'K_b': '1/2 b.H_b, H_b = grad_x(g.b) by autograd (create_graph)',
                   'g': 'gradient of each window\'s summed CE w.r.t. x (g_y W_down for U, matching the original; autograd w.r.t. x '
                        'for F / K, checked equal)', 'units': 'summed CE internally; tables divide by 10 x 127 targets at the end; mu '
                                                             'keeps n = 10 x 128',
                   'note': 'U, U_elem, U_token are gradient-built proxies, not the Hessian; U\'s cross-position term is an empirical '
                           'aggregation, not a real cross-token loss interaction (only per-position operations follow the Down)'},
        'hvp_fd': {'steps': list(HS), 'rule': f'K_FD(h) = 1/2 b.[g(x+hb) - g(x-hb)] / (2h); converged if adjacent steps agree within '
                   f'{TOL["hvp_fd"]["adjacent_rel"]} rel + {TOL["hvp_fd"]["adjacent_abs_summed_ce"]} abs (summed CE) and K_FD(1/128) is within '
                   f'{TOL["hvp_fd"]["vs_hvp_rel"]} rel + {TOL["hvp_fd"]["vs_hvp_abs_summed_ce"]} abs of K_b; otherwise numerically undetermined; '
                   'every step and the vector errors reported'},
        'controls': ['complete FP32 forward = suffix replay (bitwise)', 'only the Down input perturbed: W_down unchanged FP32 (hash), y = '
                     'linear(x\', W), no output QDQ', f'old calibration full loss (reversal diag calibration_dL x|f|all|1) within '
                     f'{TOL["old_full_loss_abs"]} and old U (calibration_scores x|f|all) within {TOL["old_U_rel"]} rel',
                     'mean of r per channel ~ 0, b + r vs e, x+e / x+b+r vs Q(x) differences', f'g_x = g_y W_down within {TOL["gx_vs_gyW_rel"]}',
                     'position 127: g and H_b exactly 0', 'alpha = 0: dL exactly 0', 'TF32 off, deterministic, float32',
                     'INT scale / zp on the device; posit LUT vs Universal', 'non-finite values kept, never clipped',
                     f'U = U0 + UR + Ucross within {TOL["U_split_rel"]}; F_all = F_P0 + F_Rest; K_all = K_P0 + K_Rest; mean-only dL additivity '
                     'over positions'],
        'channels': 'per channel mu, GS = sum_s (sum_p g)^2, U_c = GS mu^2 per format and INT - Posit differences; descriptive concentration; '
                    'no mask from channel losses',
        'reading': {'A': 'U/2 and K_b order opposite, K_b supported by FD convergence and the small-alpha curve -> curvature proxy '
                         'mismatch along the mean direction; U_elem / U_token give aggregation clues only',
                    'B': 'U/2 and K_b agree but F_b or finite-perturbation response changes the dL_b order -> first-order / finite effects',
                    'C': 'mean-only damage large, full smaller, I_br changes the gap or order -> compensation between mean and residual',
                    'D': 'formats may differ; mixed results kept separate; undetermined when numerics do not separate',
                    'E': 'a U fix would not repair the position-0 variance term or the whole BC'},
        'statistics': 'the 10 calibration windows are descriptive only; no bootstrap as if independent articles',
        'seen_before_plan': ['runs/qwen3_l27_reversal_diag_20261024 report / stats (calibration and new-article results)',
                             'runs/qwen3_posit_score_geometry_allbits_20261025 report / scores.csv (calibration geometry, another session)',
                             'runs/qwen3_l27_slot_attribution_20261024 exists (directory listing only)',
                             'scratch smoke test of this script on the same calibration windows: ' + (json.dumps(read(smoke)) if smoke else 'none')],
        'tolerances': TOL, 'boundaries': 'diagnostic; no criterion change; a candidate fix needs a separately frozen plan with new '
                                         'articles, other layers and originally correct INT / Posit pairs',
        'inputs': {rd.prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：Qwen3 L27 Down input 的 Posit 平均偏差項 U 機制診斷', '',
          f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。規格：`reports/qwen3_posit_mean_bias_prompt_20261025.md` 與 2026-10-25 的需求。', '',
          '- 範圍：L27 Down input；權重與其他 activation 全 FP32；output 不量化；int4／p4e1（主）、int6／p6e1、int8／p8e1（對照）；原 10 × 128 calibration；不抽新文章、不宣稱泛化。',
          '- 分解：e = Q(x) − x；μ_c = Σ e / 1280（float64，算一次固定）；b = μ（每個位置）；r = e − b；全在 autograd 圖外。',
          '- 量測：L(x)、L(x+b)、L(x+r)、L(Q(x))、x+e 與 x+b+r 的重建；ΔL_b、ΔL_r、ΔL_e、I_br = ΔL_e − ΔL_b − ΔL_r（逐窗口）。',
          '- 分數（同一 b 方向）：U = Σ_(s,c)(Σ_p g)²μ²（原 BC）、U_elem、U_token；F_b = g·b；K_b = ½ b·H_b（autograd HVP，create_graph）。',
          f'- α 掃描：{list(ALPHAS)}；O(α)、E(α)、R(α)；小 α 的實際改變比例與 rounding 殘差。',
          '- 位置：固定 μ 的 b_P0、b_Rest：F、K、α=±1；U0、UR、Ucross（Ucross 可負）；F、K、mean-only ΔL 的位置可加性。',
          f'- HVP-FD：h = 1/32、1/64、1/128 全報；相鄰步長 {TOL["hvp_fd"]["adjacent_rel"]} 相對 + {TOL["hvp_fd"]["adjacent_abs_summed_ce"]} 絕對、FD(1/128) 對 HVP {TOL["hvp_fd"]["vs_hvp_rel"]} 相對 + {TOL["hvp_fd"]["vs_hvp_abs_summed_ce"]} 絕對（summed CE）；不收斂即數值未定。',
          '- 控制：完整前向 = 重播（逐位元）；只改 Down input、W 不變、無 output QDQ；重現舊 full loss 與舊 U；μ、r、b+r=e；g_x = g_y W；position 127 的 g、H_b = 0；α=0 → 0；TF32 關閉、deterministic、float32；INT 在裝置上；LUT 驗證；非有限值保留。',
          '- channel：每個 channel 的 μ、GS、U_c 與 INT−Posit 差（只描述，不產生 mask）。',
          '- 判讀：A 曲率 proxy 失配（U/2 與 K_b 反序且 K_b 經 FD 與小 α 支持）；B 一階／有限擾動；C 平均與剩餘補償（I_br）；D 不同格式分開；E 修 U 不等於修好 position 0 變異項或整個 BC。',
          '- 統計：10 個 calibration 窗口只作描述，不做 bootstrap。單位：內部 summed CE，表格最後除以 1270 targets；μ 分母 1280。',
          f'- 計畫前已看：reversal diag 報告與 stats、score geometry 報告與 scores.csv、slot attribution 目錄；冒煙測試：{"見 plan.json" if smoke else "無"}。', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log('plan written')


# --------------------------------------------------------------------------- #
# run (GPU)
# --------------------------------------------------------------------------- #
def run(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    core = rd.Core(out / 'work')
    torch = core.torch
    lin = torch.nn.functional.linear
    torch.cuda.reset_peak_memory_stats()
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(toks).cuda()
    W = core.W.detach()
    w_hash = hashlib.sha256(W.cpu().numpy().tobytes()).hexdigest()
    t0 = time.perf_counter()
    ce_full, mid, x, y = core.capture(ids)
    with torch.no_grad():
        base = core.head_ce(mid + lin(x, W), ids)
    gx_ref, gy = core.grads(mid, y, ids)                              # g_x = g_y W, as the original statistics
    torch.cuda.synchronize()
    tm['fp32_forward_and_gradient'] += time.perf_counter() - t0
    ctl = {'replay_equals_full_forward': bool(torch.equal(base, ce_full)), 'y_equals_linear_x': bool(torch.equal(lin(x, W), y)),
           'W_down_sha256': w_hash, 'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
           'deterministic': torch.are_deterministic_algorithms_enabled(), 'dtype': str(x.dtype),
           'posit_lut': {f: list(core.q.posit[f].verify()) for f in core.q.posit},
           'calibration_tokens_equal_prepare': bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL])),
           'output_qdq': 'none (only the Down input is replaced; y = linear(x\', W))'}
    M = {'all': torch.ones(1, SEQ, 1, dtype=torch.bool, device='cuda'),
         'P0': (torch.arange(SEQ, device='cuda') == 0)[None, :, None], 'Rest': (torch.arange(SEQ, device='cuda') > 0)[None, :, None]}
    # decomposition
    t0 = time.perf_counter()
    E, MU, B, R, Q = {}, {}, {}, {}, {}
    with torch.no_grad():
        for f in FORMATS:
            Q[f] = core.qact(x, f, 'x')
            E[f] = Q[f] - x
            MU[f] = E[f].double().sum((0, 1)) / N
            B[f] = MU[f].float().expand_as(x).contiguous()
            R[f] = (E[f].double() - MU[f]).float()
    torch.cuda.synchronize()
    tm['qdq_and_error_stats'] += time.perf_counter() - t0
    dec = {}
    for f in FORMATS:
        dec[f] = {'r_mean_abs_max': float((R[f].double().sum((0, 1)) / N).abs().max()), 'b_plus_r_minus_e_abs_max': float((B[f] + R[f] - E[f]).abs().max()),
                  'e_abs_max': float(E[f].abs().max()), 'mu_abs_max': float(MU[f].abs().max()),
                  'xe_vs_Q_abs_max': float(((x + E[f]) - Q[f]).abs().max()), 'xe_vs_Q_fraction_differing': float(((x + E[f]) != Q[f]).float().mean()),
                  'xbr_vs_Q_abs_max': float(((x + B[f] + R[f]) - Q[f]).abs().max()), 'xbr_vs_Q_fraction_differing': float(((x + B[f] + R[f]) != Q[f]).float().mean())}
    # proxy scores from g_y W (as the original BC statistics)
    t0 = time.perf_counter()
    g = gx_ref.double()
    gsum = g.sum(1)                                                     # [s, c]
    GS = (gsum ** 2).sum(0)                                             # [c]
    A_ = g[:, 0, :]
    B_ = g[:, 1:, :].sum(1)
    scores, chan = {}, {}
    for f in FORMATS:
        mu = MU[f]
        U_w = ((gsum ** 2) * mu ** 2).sum(1)                            # per window
        Ue_w = ((g * mu) ** 2).sum((1, 2))
        Ut_w = ((g * mu).sum(2) ** 2).sum(1)
        U0, UR, Uc = float((mu ** 2 * A_ ** 2).sum()), float((mu ** 2 * B_ ** 2).sum()), float(2 * (mu ** 2 * A_ * B_).sum())
        S1 = E[f].double().sum((0, 1))
        U_rd = float((GS * (S1 / N) ** 2).sum())                       # the reversal-diag arithmetic
        scores[f] = {'U': float(U_w.sum()), 'U_elem': float(Ue_w.sum()), 'U_token': float(Ut_w.sum()), 'U_per_window': U_w.tolist(),
                     'U_elem_per_window': Ue_w.tolist(), 'U_token_per_window': Ut_w.tolist(), 'U0': U0, 'UR': UR, 'Ucross': Uc,
                     'U_reversal_diag_arithmetic': U_rd}
        chan[f] = {'mu': mu.cpu().numpy(), 'U_c': (GS * mu ** 2).cpu().numpy()}
    chan['GS'] = GS.cpu().numpy()
    tm['proxy_scores'] += time.perf_counter() - t0
    # autograd gradient w.r.t. x, F along b / b_P0 / b_Rest, HVP
    t0 = time.perf_counter()
    gx_auto = torch.empty_like(x)
    Fw = {(f, m): np.zeros(CAL) for f in FORMATS for m in MASKS}
    Kw = {(f, m): np.zeros(CAL) for f in FORMATS for m in MASKS}
    Hb_all = {f: torch.empty_like(x) for f in FORMATS}
    hb_pos127 = 0.0

    def loss_of(xj, j):
        z = core.M.model.lm_head(core.M.model.model.norm(mid[j:j + HVP_SUB] + lin(xj, W)))
        return torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids[j:j + HVP_SUB, 1:].reshape(-1), reduction='sum')
    for j in range(0, CAL, HVP_SUB):
        xj = x[j:j + HVP_SUB].clone().requires_grad_(True)
        gj, = torch.autograd.grad(loss_of(xj, j), xj, create_graph=True)
        gx_auto[j:j + HVP_SUB] = gj.detach()
        for f in FORMATS:
            for m in MASKS:
                v = (B[f][j:j + HVP_SUB] * M[m]).detach()
                s = (gj * v).sum()
                Hv, = torch.autograd.grad(s, xj, retain_graph=True)
                Fw[(f, m)][j:j + HVP_SUB] = (gj.detach().double() * v.double()).sum((1, 2)).cpu().numpy()
                Kw[(f, m)][j:j + HVP_SUB] = (0.5 * (v.double() * Hv.double()).sum((1, 2))).cpu().numpy()
                if m == 'all':
                    Hb_all[f][j:j + HVP_SUB] = Hv
                    hb_pos127 = max(hb_pos127, float(Hv[:, -1].abs().max()))
                del Hv
        del gj
    torch.cuda.synchronize()
    tm['hvp'] += time.perf_counter() - t0
    ctl.update({'gx_autograd_vs_gyW_rel_max': float((gx_auto - gx_ref).abs().max() / gx_ref.abs().max()),
                'g_position127_abs_max': float(gx_auto[:, -1].abs().max()), 'Hb_position127_abs_max': hb_pos127})
    # finite differences of the gradient along b
    t0 = time.perf_counter()

    def grad_at(xp):
        out_ = torch.empty_like(xp)
        for j in range(0, CAL, HVP_SUB):
            xj = xp[j:j + HVP_SUB].clone().requires_grad_(True)
            gj, = torch.autograd.grad(loss_of(xj, j), xj)
            out_[j:j + HVP_SUB] = gj
        return out_
    fd = {}
    for f in FORMATS:
        rows = []
        for h in HS:
            gp, gm = grad_at(x + h * B[f]), grad_at(x - h * B[f])
            FD = (gp.double() - gm.double()) / (2 * h)
            K_fd = float(0.5 * (B[f].double() * FD).sum())
            Hd = Hb_all[f].double()
            rows.append({'h': h, 'K_fd': K_fd, 'vector_rel_err': float((FD - Hd).norm() / Hd.norm()),
                         'projection_err_vs_K_b': K_fd - float(Kw[(f, 'all')].sum()),
                         'changed_fraction_plus': float(((x + h * B[f]) != x).float().mean())})
        Kb = float(Kw[(f, 'all')].sum())
        adj = [abs(rows[k]['K_fd'] - rows[k + 1]['K_fd']) <= TOL['hvp_fd']['adjacent_rel'] * abs(rows[k + 1]['K_fd']) + TOL['hvp_fd']['adjacent_abs_summed_ce']
               for k in range(len(rows) - 1)]
        close = abs(rows[-1]['K_fd'] - Kb) <= TOL['hvp_fd']['vs_hvp_rel'] * abs(Kb) + TOL['hvp_fd']['vs_hvp_abs_summed_ce']
        fd[f] = {'steps': rows, 'K_b': Kb, 'adjacent_converged': adj, 'fd_matches_hvp': bool(close), 'confirmed': bool(all(adj) and close)}
    torch.cuda.synchronize()
    tm['hvp_fd_check'] += time.perf_counter() - t0
    # interventions
    t0 = time.perf_counter()
    dl = {}
    with torch.no_grad():
        L = lambda xp: (core.head_ce(mid + lin(xp, W), ids) - base).cpu().numpy()
        for f in FORMATS:
            dl[f'{f}|b'] = L(x + B[f])
            dl[f'{f}|r'] = L(x + R[f])
            dl[f'{f}|e_direct'] = L(Q[f])
            dl[f'{f}|x+e'] = L(x + E[f])
            dl[f'{f}|x+b+r'] = L(x + B[f] + R[f])
            for a in ALPHAS:
                xp = x + a * B[f]
                dl[f'{f}|alpha|{a:g}'] = L(xp)
                ab = (a * B[f]).double()
                dl[f'{f}|alpha|{a:g}|changed_fraction'] = float((xp != x).float().mean())
                dl[f'{f}|alpha|{a:g}|rounding_rel'] = float(((xp.double() - x.double()) - ab).norm() / ab.norm()) if a != 0 else 0.0
            for m in ('P0', 'Rest'):
                for a in (1.0, -1.0):
                    dl[f'{f}|{m}|{a:g}'] = L(x + a * B[f] * M[m])
    torch.cuda.synchronize()
    tm['interventions_and_curves'] += time.perf_counter() - t0
    # controls on old results
    old = np.load(REV / 'calibration_dL.npz')
    oldsc = read(REV / 'calibration_scores.json')
    ctl['old_full_loss_max_abs'] = {f: float(np.abs(dl[f'{f}|e_direct'] - old[f'x__{f}__all__1']).max()) for f in FORMATS}
    ctl['old_U_rel'] = {f: abs(scores[f]['U_reversal_diag_arithmetic'] - oldsc[f'x|{f}|all']['U']) / abs(oldsc[f'x|{f}|all']['U']) for f in FORMATS}
    ctl['U_two_arithmetics_rel'] = {f: abs(scores[f]['U'] - scores[f]['U_reversal_diag_arithmetic']) / abs(scores[f]['U']) for f in FORMATS}
    ctl['alpha0_max_abs'] = {f: float(np.abs(dl[f'{f}|alpha|0']).max()) for f in FORMATS}
    ctl['alpha1_equals_mean_only'] = {f: bool(np.array_equal(dl[f'{f}|alpha|1'], dl[f'{f}|b'])) for f in FORMATS}
    ctl['decomposition'] = dec
    ctl['U_split_rel'] = {f: abs(scores[f]['U0'] + scores[f]['UR'] + scores[f]['Ucross'] - scores[f]['U']) / abs(scores[f]['U']) for f in FORMATS}
    ctl['F_additivity_rel'] = {f: abs(Fw[(f, 'P0')].sum() + Fw[(f, 'Rest')].sum() - Fw[(f, 'all')].sum()) / max(abs(Fw[(f, 'all')].sum()), 1e-300) for f in FORMATS}
    ctl['K_additivity_rel'] = {f: abs(Kw[(f, 'P0')].sum() + Kw[(f, 'Rest')].sum() - Kw[(f, 'all')].sum()) / max(abs(Kw[(f, 'all')].sum()), 1e-300) for f in FORMATS}
    ctl['mean_only_additivity'] = {f: {a: float(abs(dl[f'{f}|P0|{a:g}'].sum() + dl[f'{f}|Rest|{a:g}'].sum() - dl[f'{f}|alpha|{a:g}'].sum()))
                                       for a in (1.0, -1.0)} for f in FORMATS}
    ctl['hvp_fd'] = {f: {'confirmed': fd[f]['confirmed'], 'adjacent': fd[f]['adjacent_converged'], 'matches_hvp': fd[f]['fd_matches_hvp']} for f in FORMATS}
    ctl['nonfinite'] = int(sum((~np.isfinite(v)).sum() for v in dl.values() if isinstance(v, np.ndarray)))
    ma = TOL['mean_only_additivity']
    ctl['passed'] = bool(
        ctl['replay_equals_full_forward'] and ctl['y_equals_linear_x'] and ctl['tf32'] == [False, False] and ctl['deterministic']
        and all(v[0] == 0 for v in ctl['posit_lut'].values()) and ctl['calibration_tokens_equal_prepare']
        and all(v <= TOL['old_full_loss_abs'] for v in ctl['old_full_loss_max_abs'].values())
        and all(v <= TOL['old_U_rel'] for v in ctl['old_U_rel'].values()) and ctl['gx_autograd_vs_gyW_rel_max'] <= TOL['gx_vs_gyW_rel']
        and ctl['g_position127_abs_max'] == 0 and ctl['Hb_position127_abs_max'] == 0 and all(v == 0 for v in ctl['alpha0_max_abs'].values())
        and all(ctl['alpha1_equals_mean_only'].values()) and all(v <= TOL['U_split_rel'] for v in ctl['U_split_rel'].values())
        and all(v <= TOL['F_additivity_rel'] for v in ctl['F_additivity_rel'].values())
        and all(v <= TOL['K_additivity_rel'] for v in ctl['K_additivity_rel'].values())
        and all(d_ <= ma['abs_summed_ce'] + ma['rel'] * abs(dl[f'{f}|alpha|{a:g}'].sum()) for f in FORMATS for a, d_ in
                ((1.0, ctl['mean_only_additivity'][f][1.0]), (-1.0, ctl['mean_only_additivity'][f][-1.0])))
        and ctl['nonfinite'] == 0)
    ctl['note'] = 'HVP-FD convergence is reported per format; a failure marks K_b numerically undetermined, it does not stop the run'
    write(out / 'controls.json', ctl)
    np.savez(out / 'losses.npz', base=base.cpu().numpy(), **{k.replace('|', '__'): np.asarray(v) for k, v in dl.items()})
    write(out / 'raw.json', {'scores': scores, 'F_per_window': {f'{f}|{m}': Fw[(f, m)].tolist() for f in FORMATS for m in MASKS},
                             'K_per_window': {f'{f}|{m}': Kw[(f, m)].tolist() for f in FORMATS for m in MASKS}, 'hvp_fd': fd})
    with open(out / 'channels.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['channel', 'GS', *[f'mu_{f}' for f in FORMATS], *[f'U_c_{f}' for f in FORMATS], *[f'U_c_{a}_minus_{b}' for a, b in PAIRS]])
        for c in range(len(chan['GS'])):
            wr.writerow([c, chan['GS'][c], *[chan[f]['mu'][c] for f in FORMATS], *[chan[f]['U_c'][c] for f in FORMATS],
                         *[chan[a]['U_c'][c] - chan[b]['U_c'][c] for a, b in PAIRS]])
    tm['total'] = time.perf_counter() - t_all
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    write(out / 'timing.json', dict(tm))
    log(f'controls passed={ctl["passed"]}; hvp-fd confirmed {ctl["hvp_fd"]}; times {dict(tm)}')
    if not ctl['passed']:
        raise SystemExit('controls failed (see controls.json); the analysis must not be read as a mechanism result')


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    check_plan(out)
    ctl, raw = read(out / 'controls.json'), read(out / 'raw.json')
    z = np.load(out / 'losses.npz')
    dl = {k.replace('__', '|'): z[k] for k in z.files}
    sc, fd = raw['scores'], raw['hvp_fd']
    F = {k: float(np.sum(v)) for k, v in raw['F_per_window'].items()}
    K = {k: float(np.sum(v)) for k, v in raw['K_per_window'].items()}
    per = lambda v: v / TARGETS
    T = {}
    for f in FORMATS:
        dLb, dLr, dLe = dl[f'{f}|b'].sum(), dl[f'{f}|r'].sum(), dl[f'{f}|e_direct'].sum()
        curve = []
        for a in ALPHAS:
            if a <= 0:
                continue
            p_, m_ = dl[f'{f}|alpha|{a:g}'].sum(), dl[f'{f}|alpha|{-a:g}'].sum()
            O, E_ = (p_ - m_) / 2, (p_ + m_) / 2
            curve.append({'alpha': a, 'dL_plus': per(p_), 'dL_minus': per(m_), 'O_over_alpha': per(O / a), 'E_over_alpha2': per(E_ / a ** 2),
                          'R_plus': per(p_ - a * F[f'{f}|all'] - a * a * K[f'{f}|all']), 'R_minus': per(m_ + a * F[f'{f}|all'] - a * a * K[f'{f}|all']),
                          'changed_fraction': float(dl[f'{f}|alpha|{a:g}|changed_fraction']), 'rounding_rel': float(dl[f'{f}|alpha|{a:g}|rounding_rel'])})
        s_ = TOL['small_alpha_support']
        last = curve[-1]
        support = abs(last['E_over_alpha2'] - per(K[f'{f}|all'])) <= s_['rel'] * abs(per(K[f'{f}|all'])) + per(s_['abs_summed_ce'])
        T[f] = {'U/2': per(sc[f]['U'] / 2), 'U_elem/2': per(sc[f]['U_elem'] / 2), 'U_token/2': per(sc[f]['U_token'] / 2),
                'F_b': per(F[f'{f}|all']), 'K_b': per(K[f'{f}|all']), 'dL_b': per(dLb), 'dL_r': per(dLr), 'dL_e': per(dLe),
                'I_br': per(dLe - dLb - dLr), 'F+K': per(F[f'{f}|all'] + K[f'{f}|all']), 'K_confirmed_by_fd': fd[f]['confirmed'],
                'small_alpha_supports_K': bool(support), 'curve': curve,
                'positions': {m: {'F': per(F[f'{f}|{m}']), 'K': per(K[f'{f}|{m}']), 'dL_plus1': per(dl[f'{f}|{m}|1'].sum()),
                                  'dL_minus1': per(dl[f'{f}|{m}|-1'].sum())} for m in ('P0', 'Rest')},
                'U_split': {k: per(sc[f][k] / 2) for k in ('U0', 'UR', 'Ucross')},
                'per_window': {'dL_b': (dl[f'{f}|b'] / TP).tolist(), 'dL_r': (dl[f'{f}|r'] / TP).tolist(), 'dL_e': (dl[f'{f}|e_direct'] / TP).tolist(),
                               'I_br': ((dl[f'{f}|e_direct'] - dl[f'{f}|b'] - dl[f'{f}|r']) / TP).tolist()}}
    sign = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
    pairs = {}
    for a, b in PAIRS:
        d = {k: T[a][k] - T[b][k] for k in ('U/2', 'U_elem/2', 'U_token/2', 'F_b', 'K_b', 'F+K', 'dL_b', 'dL_r', 'dL_e', 'I_br')}
        kconf = T[a]['K_confirmed_by_fd'] and T[b]['K_confirmed_by_fd'] and T[a]['small_alpha_supports_K'] and T[b]['small_alpha_supports_K']
        reading = []
        if sign(d['U/2']) != sign(d['K_b']):
            reading.append('A: U/2 and K_b ordered oppositely' + ('' if kconf else ' (K_b not numerically confirmed: undetermined)'))
        elif sign(d['dL_b']) != sign(d['U/2']):
            reading.append('B: U/2 and K_b agree, but F_b / finite response changes the dL_b order')
        else:
            reading.append('U/2, K_b and dL_b agree in order')
        if sign(d['dL_b']) != sign(d['dL_e']) or abs(d['I_br']) > abs(d['dL_b']):
            reading.append('C: I_br changes the gap or order between mean-only and full damage (compensation / non-additivity)')
        pairs[f'{a}-{b}'] = {'INT_minus_Posit': d, 'K_numerically_confirmed': bool(kconf), 'reading': reading,
                             'orders': {k: ('INT larger' if v > 0 else 'Posit larger' if v < 0 else 'equal') for k, v in d.items()}}
    # channel concentration
    ch = list(csv.DictReader(open(out / 'channels.csv')))
    conc = {}
    for f in FORMATS:
        u = np.array([float(r[f'U_c_{f}']) for r in ch])
        srt = np.sort(u)[::-1]
        conc[f] = {'top1_share': float(srt[0] / srt.sum()), 'top10_share': float(srt[:10].sum() / srt.sum()), 'top100_share': float(srt[:100].sum() / srt.sum()),
                   'top_channels': [int(i) for i in np.argsort(-u)[:5]]}
    res = {'units': 'nat/token (summed CE / 1270 targets); U-type scores halved as curvature-scale proxies', 'formats': T, 'pairs': pairs,
           'channel_concentration': conc, 'controls_passed': ctl['passed']}
    write(out / 'analysis.json', res)
    with open(out / 'summary_table.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        keys = ('U/2', 'U_elem/2', 'U_token/2', 'F_b', 'K_b', 'dL_b', 'dL_r', 'dL_e', 'I_br')
        wr.writerow(['format', *keys, 'K_confirmed_by_fd', 'small_alpha_supports_K'])
        for f in FORMATS:
            wr.writerow([f, *[T[f][k] for k in keys], T[f]['K_confirmed_by_fd'], T[f]['small_alpha_supports_K']])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (a, b) in zip(axes, PAIRS):
        for f, c in ((a, '#1f77b4'), (b, '#d62728')):
            al = sorted(ALPHAS)
            meas = [per(dl[f'{f}|alpha|{x_:g}'].sum()) for x_ in al]
            ax.plot(al, meas, 'o-', color=c, label=f'{f} measured dL_b(alpha)')
            grid = np.linspace(-1, 1, 101)
            ax.plot(grid, [per(g_ * F[f'{f}|all'] + g_ * g_ * K[f'{f}|all']) for g_ in grid], '--', color=c, lw=1, label=f'{f} alpha F + alpha^2 K')
            ax.plot(grid, [per(g_ * g_ * sc[f]['U'] / 2) for g_ in grid], ':', color=c, lw=1, label=f'{f} alpha^2 U/2')
        ax.axhline(0, color='k', lw=.6)
        ax.set_title(f'L27 Down input mean direction b: {a} vs {b}', fontsize=9)
        ax.set_xlabel('alpha')
        ax.set_ylabel('nat/token')
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'fig_mean_direction_curves.png', dpi=120)
    plt.close(fig)
    t = read(out / 'timing.json')
    t['analysis'] = time.perf_counter() - t0
    write(out / 'timing.json', t)
    log(json.dumps({k: {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v['INT_minus_Posit'].items()} | {'reading': v['reading']}
                    for k, v in pairs.items()}, ensure_ascii=False, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'run', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note', type=Path)
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
    else:
        {'run': run, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_posit_mean_bias.py')}) + '\n')


if __name__ == '__main__':
    main()
