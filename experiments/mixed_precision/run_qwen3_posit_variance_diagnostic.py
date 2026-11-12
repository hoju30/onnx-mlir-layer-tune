#!/usr/bin/env python3
"""Qwen3-0.6B Down input: mechanism diagnostic of the BC variance term for Posit vs INT activations (L27 primary,
L2 pre-specified control). Spec: reports/qwen3_posit_variance_diagnostic_prompt_20261025.md. Calibration only (original
10 x 128 windows); no new articles, no criterion / quantizer / range / search / pipeline change, no fine-tuning, no native.

Setting: every weight and activation FP32, only one layer's Down input x perturbed, Down output not quantized. For format f:
e = Q_f(x) - x, mu_c = sum_(s,p) e / 1280 (float64, once), b = mu_c at every (s,p), r = e - b; all outside autograd.
Fixed position directions d_m = M_m r (m = all, P0, Rest; never re-centred). Loss = summed next-token CE per window through
the real suffix: L27 = Down -> residual -> final RMSNorm -> LM head; L2 = Down -> residual -> layers 3..27 (attention and
FFN) -> final norm -> head.
  proxies  V_old = sum_c mean_(s,p)(g^2) sum_(s,p) r^2,  V_elem = sum (g r)^2,  V_token = sum_(s,p) (sum_c g r)^2,
           V_window = sum_s (sum_(p,c) g r)^2   (no U; V_elem = the old V_pair, not T_pair)
  exact    F_r = g.d,  K_r = 1/2 d.H d (autograd HVP, create_graph, d detached), b.H d (no extra HVP)
  curves   dL_r(alpha) = L(x + alpha d) - L(x), alpha in {0, +-1, +-1/2, ..., +-1/64}; O, E, R = dL - alpha F - alpha^2 K
  refs     Q(x), x + b, x + e, x + b + r (mask all): dL_e, dL_b, dL_r(1), I_br
HVP checked by central differences of the gradient (h = 1/32, 1/64, 1/128) for every r direction and a synthetic one.

Stages (one new --out-dir): plan, run, analyze. CPU: ../../venv/bin/python; run: PYTHONPATH=qwen35_assets/gpu_deps
/home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_l27_reversal_diag as rd  # noqa: E402  (Core, constants; standard imports only)

read, write, sha, now = rd.read, rd.write, rd.sha, rd.now
NATIVE, PREP = rd.NATIVE, rd.PREP
REV = HERE / 'runs/qwen3_l27_reversal_diag_20261024'
PMB = HERE / 'runs/qwen3_posit_mean_bias_20261025'
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
UTV = HERE / 'runs/qwen3_mean_u_token_validation_20261025'
UTQ = HERE / 'runs/qwen3_u_token_full_qdq_replay_20261025'
LAYERS = (27, 2)                                                       # L27 primary, L2 pre-specified control
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
MISRANKS = ((2, 'int6', 'p6e1'), (2, 'int8', 'p8e1'), (27, 'int4', 'p4e1'), (27, 'int6', 'p6e1'))
ALPHAS = (0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 1 / 16, -1 / 16, 1 / 32, -1 / 32, 1 / 64, -1 / 64)
HS = (1 / 32, 1 / 64, 1 / 128)
MASKS = ('all', 'P0', 'Rest')
CAL, SEQ, TP = 10, 128, 127
N, TARGETS = CAL * SEQ, CAL * TP
MB, LM_SUB = 1, 8
SYN_SEED = 32201
SMALL_ALPHA = 1 / 16
TOL = {'replay_vs_full_abs_summed_ce': 1e-6, 'old_loss_abs_summed_ce': 1e-6, 'old_same_backend_rel': 1e-3,
       'old_same_backend_signed_sum_rel': 1e-2, 'old_cpu_backend_rel': 1e-2,
       'gx_vs_gyW_rel': 1e-4, 'pos127_abs': 0.0, 'alpha0_abs': 0.0, 'int_device_vs_cpu_abs': 0.0,
       'batch_loss_abs_summed_ce': 1e-3, 'batch_grad_rel': 1e-3, 'V_split_rel': 1e-9, 'cov_identity_rel': 1e-9,
       'F_additivity_rel': 1e-6, 'K_additivity_rel_L27': 1e-3, 'dL_additivity_L27': {'abs_summed_ce': 1e-3, 'rel': 1e-3},
       'hvp_fd': {'adjacent_rel': 0.02, 'adjacent_abs_summed_ce': 1e-3, 'vs_hvp_rel': 0.05, 'vs_hvp_abs_summed_ce': 1e-3},
       'hvp_machinery_min_match_fraction': 0.95,
       'small_alpha_support': {'alpha': SMALL_ALPHA, 'rel': 0.2, 'abs_summed_ce': 1e-3},
       'reported_only': 'r mean, b + r - e, x+e / x+b+r vs Q(x) floating differences, V_old two arithmetics, L2 K / dL position '
                        'interactions, rounding of small alpha, the L2 synthetic direction'}
CODE = ['run_qwen3_posit_variance_diagnostic.py', 'run_qwen_l27_reversal_diag.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_down_spike_diag.py',
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
        NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', PREP / 'validation_tokens.npy', PREP / 'node_mapping.json',
        REV / 'calibration_scores.json', PMB / 'losses.npz', PMB / 'raw.json', PMB / 'report.md', SPIKE / 'scores.csv',
        SPIKE / 'ranking_pairs.csv', SPIKE / 'report.md', UTV / 'calibration_losses.npz', UTV / 'report.md', UTQ / 'same_bit_pairs.csv',
        UTQ / 'scores.csv', UTQ / 'report.md', HERE / 'reports/qwen3_posit_variance_diagnostic_prompt_20261025.md']


def check_plan(out):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    for f, h in pl['inputs'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'input changed: {f}')
    return pl


def down_node(l):
    m = read(PREP / 'node_mapping.json')
    return next(n for n, p in m.items() if p == f'model.layers.{l}.mlp.down_proj.weight')


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    from run_olmoe_dynamic_route import env_info
    st = TOL['small_alpha_support']
    pl = {
        'time': now(), 'spec': 'reports/qwen3_posit_variance_diagnostic_prompt_20261025.md and the request of 2026-10-25',
        'question': 'the four full Down-input misranks (L2 int6-p6e1, L2 int8-p8e1, L27 int4-p4e1, L27 int6-p6e1) survive U_old -> '
                    'U_token; along the residual direction r = e - mu, does the variance proxy misrank because of (1) position '
                    'pairing of gradient and error energy, (2) lost error direction / cross-channel coupling, (3) a missing signed '
                    'first-order term, or (4) finite-perturbation nonlinearity? mixtures allowed',
        'scope': 'Qwen3-0.6B; L27 Down input primary, L2 Down input pre-specified control (attention follows); all six formats at both '
                 'layers (L2 4-bit kept); one layer at a time; weights and other activations FP32; output not quantized; original 10 x 128 '
                 'calibration, native ranges, INT QDQ on the device, Universal-verified posit LUT; no new articles, no generalization '
                 'claim; no U / V / quantizer / search / knee / pipeline change',
        'decomposition': {'e': 'Q_f(x) - x', 'mu_c': 'sum_(s,p) e / 1280 (float64, once)', 'b': 'mu_c at every (s,p)', 'r': 'e - b',
                          'directions': 'd_all = r, d_P0 = M_P0 r, d_Rest = M_Rest r (fixed; no re-estimated mu, no re-centring)',
                          'autograd': 'e, mu, b, r built outside autograd and detached; no gradient through QDQ, no STE',
                          'note': 'x + r equals Q(x) - mu mathematically; the floating difference is reported'},
        'proxies': {'V_old': 'sum_c mean_(s,p)(g^2)_c * sum_(s,p) r_spc^2 (n = 1280)', 'V_elem': 'sum_(s,p,c) (g r)^2 (= old V_pair, no U)',
                    'V_token': 'sum_(s,p) (sum_c g r)^2', 'V_window': 'sum_s (sum_(p,c) g r)^2',
                    'masks': 'V_old,m = sum_c mean_all(g^2)_c sum (M_m r)^2 (original mean_all(g^2)); V_elem / V_token / V_window with M_m r',
                    'identities': 'V_elem - V_old = sum_c sum_(s,p) (g^2 - mean g^2)_c r^2 (= n sum_c cov_c(g^2, r^2)); V_token - V_elem '
                                  '= within-token cross-channel terms; V_window - V_token = cross-position empirical aggregation',
                    'background': 'U = sum_(s,c) mu^2 (sum_p g)^2 saved, never added to r scores', 'scale': 'V/2 compared with K_r',
                    'note': 'proxies, not the Hessian'},
        'exact': {'g': 'autograd gradient of each window\'s summed CE w.r.t. x through the real suffix, microbatch 1',
                  'F_r': 'g . d', 'K_r': '1/2 d . H_r, H_r = grad_x(g . d), g with create_graph, d detached; no full Hessian; signs kept',
                  'bHd': 'b . H_r recorded (mixed mean / residual second-order term; not I_br)',
                  'closures': 'L27: Down(x\'), original residual, final RMSNorm, LM head, CE; L2: Down(x\'), residual, layers 3..27 with '
                              'the captured attention kwargs, final norm, head, CE (never the L27 head-only closure)',
                  'reproduction': 'L27 F_b / K_b along b (previous round) recomputed with this path'},
        'curves': {'alphas': list(ALPHAS), 'directions': 'every layer / format / mask', 'O': '[dL(+a) - dL(-a)] / 2', 'E': '[dL(+a) + dL(-a)] / 2',
                   'R': 'dL(a) - a F_r - a^2 K_r', 'compare': 'O/a vs F_r, E/a^2 vs K_r, a F + a^2 K and a^2 V/2 vs the measured curve',
                   'note': 'alpha = 1 odd / even parts contain higher orders; r-only alpha = 1 is x + r, not Q(x)',
                   'references': 'baseline, Q(x), x + b, x + e, x + b + r (mask all): dL_e, dL_b, dL_r(1), I_br = dL_e - dL_b - dL_r(1)'},
        'positions': {'additive_both': 'F_all = F_P0 + F_Rest; V_old,all = V_old,P0 + V_old,Rest',
                      'L27': 'per-position suffix: K_all = K_P0 + K_Rest and dL_all(a) = dL_P0(a) + dL_Rest(a) within tolerance',
                      'L2': 'attention follows: K_position_cross = K_all - K_P0 - K_Rest and I_position(a) = dL_all - dL_P0 - dL_Rest '
                            'reported with sign, not pass / fail',
                      'note': 'masked r is not zero-mean; a fixed-direction split, not a recomputed BC; position 0 is not named an '
                              'attention-sink root cause; mu fixed, so position 0 acting on Rest through mu is not excluded'},
        'hvp_fd': {'steps': list(HS), 'directions': 'all 18 r directions per layer and one synthetic Gaussian direction (seed '
                   f'{SYN_SEED}, norm = ||r_int4||)', 'rule': f'K_FD(h) = 1/2 d.[g(x+hd) - g(x-hd)] / (2h); confirmed if adjacent steps '
                   f'agree within {TOL["hvp_fd"]["adjacent_rel"]} rel + {TOL["hvp_fd"]["adjacent_abs_summed_ce"]} abs (summed CE) and '
                   f'K_FD(1/128) within {TOL["hvp_fd"]["vs_hvp_rel"]} rel + {TOL["hvp_fd"]["vs_hvp_abs_summed_ce"]} abs of K_r; otherwise '
                   'numerically undetermined; vector errors of every step reported; the same rule per window is reported (where the FD '
                   'of a window has not converged over these fixed steps, that window is undetermined, the steps are not refined)'},
        'small_alpha': f'K supported if |E({st["alpha"]:g})/a^2 - K_r| <= {st["rel"]}|K_r| + {st["abs_summed_ce"]} (summed CE); F supported '
                       f'if |O({st["alpha"]:g})/a - F_r| <= {st["rel"]}|F_r| + {st["abs_summed_ce"]}; all alphas reported (no picking); '
                       'changed fraction over the direction support and rounding residual per alpha',
        'reading': {'flags': 'per layer / same-bit pair (mask all), d(X) = X_INT - X_Posit, sign() compared',
                    'A_curvature_proxy': 'sign d(V_old/2) != sign d(K_r) -> local direction-curvature proxy mismatch; supported only if K_r '
                                         'is FD-confirmed and small-alpha supported for both formats, else undetermined; V_elem / V_token / '
                                         'V_window agreeing with K_r listed as clues (pairing / channel coupling / aggregation), not root causes',
                    'B_first_order': 'sign d(K_r) != sign d(dL_r(1)) and sign d(F_r + K_r) = sign d(dL_r(1)) -> signed first order changes '
                                     'the r-only order; supported only if K and F are numerically supported; F flipping the local model '
                                     '(sign d(F+K) != sign d(K)) reported separately',
                    'C_higher_order': 'sign d(F_r + K_r) != sign d(dL_r(1)) with K and F supported -> finite-perturbation higher order; '
                                      'even part sign d(E(1)) vs d(K), odd part sign d(O(1)) vs d(F) reported',
                    'undetermined': 'formats or positions without local convergence / failing numerics stay undetermined',
                    'layers': 'L27 conclusions are not carried to L2',
                    'full': 'r-only results do not fix full e or the whole BC; dL_b, dL_e, I_br and the new-article truth (down spike '
                            'diag) kept beside every pair'},
        'controls': ['complete forward with a Down pre-hook = suffix replay (baseline, x + r, x + b, Q(x) per format) within '
                     f'{TOL["replay_vs_full_abs_summed_ce"]} summed CE', 'Down input slot, W_down FP32 unchanged (hash), y = linear(x, W), '
                     'no output QDQ', f'L27 r-only / mean-only / full losses vs runs/qwen3_posit_mean_bias_20261025 within '
                     f'{TOL["old_loss_abs_summed_ce"]}; L27 K_b within {TOL["old_same_backend_rel"]} rel, F_b within '
                     f'{TOL["old_same_backend_signed_sum_rel"]} rel (previous round: 2-window sub-batches, here microbatch 1)',
                     f'L27 V_old / V_pair vs the reversal diag (GPU) within {TOL["old_same_backend_rel"]} rel; signed-sum quantities with '
                     f'cancellation (F_b, U) within {TOL["old_same_backend_signed_sum_rel"]} rel; L2 and L27 V_old / V_pair / U vs the down spike '
                     f'diag (CPU backend) within {TOL["old_cpu_backend_rel"]} rel',
                     f'L2 mean-only (x + b) loss vs runs/qwen3_mean_u_token_validation_20261025 (L2|f|1) within {TOL["old_loss_abs_summed_ce"]}',
                     'mu, r mean ~ 0, b + r - e, x+e / x+b+r vs Q(x) (reported)', f'g_x = g_y W within {TOL["gx_vs_gyW_rel"]}',
                     'position 127: g and every H_r exactly 0', 'alpha = 0 exactly 0', 'TF32 off, deterministic, eager attention, float32',
                     'posit LUT vs Universal; INT QDQ on the device equals the CPU QDQ', f'batch sizes 1 / 2 / 10: loss within '
                     f'{TOL["batch_loss_abs_summed_ce"]} summed CE, g within {TOL["batch_grad_rel"]} rel',
                     f'V_old split within {TOL["V_split_rel"]}, covariance identity within {TOL["cov_identity_rel"]}, F additivity within '
                     f'{TOL["F_additivity_rel"]}, L27 K additivity within {TOL["K_additivity_rel_L27"]}, L27 dL additivity '
                     f'{TOL["dL_additivity_L27"]}', 'HVP machinery: every L27 direction (synthetic included) FD-confirmed in total; at both '
                     f'layers, of the (direction, window) cells whose FD converged, at least {TOL["hvp_machinery_min_match_fraction"]} match '
                     'the HVP (same rule per window); the L2 synthetic direction reported',
                     'non-finite values kept (never clipped or dropped)'],
        'float64': 'no float64 reference in this round',
        'statistics': 'the 10 calibration windows are descriptive only; no bootstrap, no generalization test',
        'seen_before_plan': ['runs/qwen3_u_token_full_qdq_replay_20261025 report / scores / same_bit_pairs', 'runs/qwen3_posit_mean_bias_20261025 '
                             'report / summary_table / raw / controls', 'runs/qwen3_mean_u_token_validation_20261025 report (and its L2 '
                             'calibration mean-shift losses)', 'runs/qwen3_down_spike_diag_20261023 report / scores / ranking_pairs',
                             'smoke: ' + (smoke or 'none')],
        'tolerances': TOL,
        'boundaries': 'diagnostic; stop afterwards: no criterion change, new articles, all-layer sweep, RMSNorm substitute, posit '
                      'scaling / bias correction, configuration search or native; one concrete low-cost hypothesis proposed from the results',
        'inputs': {rd.prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：BC 變異項的 Posit 誤排機制診斷（Qwen3 Down input，L27 主、L2 對照）', '',
          f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。', '',
          '- 範圍：L27 Down input 主分析、L2 Down input 預先指定對照；兩層六格式全報（含 L2 4-bit）；一次一層；權重與其他 activation FP32；output 不量化；原 10×128 calibration、range、LUT；不抽新文章、不宣稱泛化。',
          '- 分解：e = Q(x)−x；μ = Σe/1280（float64，一次）；b = μ；r = e−b；方向 r_all、M_P0 r、M_Rest r，不重估 μ、不重新中心化。',
          '- proxy（不加 U）：V_old = Σ_c mean(g²)Σr²、V_elem = Σ(gr)²（= 舊 V_pair）、V_token = Σ_(s,p)(Σ_c gr)²、V_window = Σ_s(Σ_(p,c) gr)²；以 V/2 對照 K_r。',
          '- 真實導數：F_r = g·d；K_r = ½ d·H_r（autograd HVP，create_graph，microbatch 1）；b·H_r 另記。L27 closure = Down→residual→final norm→head；L2 closure = 真實 L3–L27（含 attention、FFN）→norm→head。',
          f'- 曲線：α ∈ {{0, ±1, ±1/2, ±1/4, ±1/8, ±1/16, ±1/32, ±1/64}}，全部 layer／format／mask；O、E、R；參考 Q(x)、x+b、x+e、x+b+r 與 I_br。',
          f'- HVP-FD：h = 1/32、1/64、1/128，18 個 r 方向 + 1 個合成方向；相鄰 {TOL["hvp_fd"]["adjacent_rel"]} 相對 + {TOL["hvp_fd"]["adjacent_abs_summed_ce"]} 絕對、FD(1/128) 對 HVP {TOL["hvp_fd"]["vs_hvp_rel"]} 相對 + {TOL["hvp_fd"]["vs_hvp_abs_summed_ce"]} 絕對；不收斂即未定。小 α 支持：α = 1/16，{st["rel"]} 相對 + {st["abs_summed_ce"]} 絕對（summed CE）。',
          '- 位置：F、V_old 兩層都應可加；L27 的 K 與 ΔL 應可加；L2 報 K_position_cross 與 I_position(α)（保留正負號）。',
          '- 判讀旗標（INT−Posit，mask all）：A V_old/2 與 K_r 反號（K 經 FD 與小 α 支持才算支持）；B K_r 與 ΔL_r(1) 反號而 F+K 與 ΔL_r(1) 同號；C F+K 與 ΔL_r(1) 反號；數值不足即未定；L27 不推到 L2。',
          '- 控制：完整前向 = 重播；只改 Down input、W 不變、無 output QDQ；重現 L27 前輪 r／b／e loss 與 F_b／K_b、V_old／V_pair／U（reversal diag 與 spike diag）、L2 x+b loss（U_token 輪）；g_x = g_y W；position 127；α = 0；TF32 關閉、deterministic、eager；LUT、INT 裝置端；batch 1／2／10；HVP 機制（L27 全部方向含合成方向 FD 確認；兩層逐窗口 FD 已收斂者 ≥ 95% 與 HVP 一致；L2 合成方向只報告）。',
          '- 統計：10 個窗口只作描述，不做 bootstrap。單位：內部 summed CE，表格最後 ÷1270；μ 分母 1280。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log('plan written')


# --------------------------------------------------------------------------- #
# GPU helpers
# --------------------------------------------------------------------------- #
def capture(core, l, ids):
    """complete FP32 forward: per-window summed CE, the residual before layer l's MLP, its Down input / output, layer kwargs."""
    torch = core.torch
    cap, hk = {}, []
    layer = core.M.layers[l]
    hk.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: cap.__setitem__('mid', a[0].detach())))
    hk.append(layer.mlp.down_proj.register_forward_hook(lambda m, a, o: cap.update(x=a[0].detach(), y=o.detach())))
    hk.append(core.M.layers[1].register_forward_pre_hook(lambda m, a, k: cap.__setitem__('kw', dict(k)), with_kwargs=True))
    with torch.no_grad():
        h = core.M.model.model(input_ids=ids, attention_mask=core.M.wrapper.mask, use_cache=False).last_hidden_state
        ce = core.head_ce(h, ids, normed=True)
    for h_ in hk:
        h_.remove()
    return ce, cap['mid'], cap['x'], cap['y'], cap['kw']


def suffix(core, l, h, kw):
    for layer in core.M.layers[l + 1:]:
        o = layer(h, **kw)
        h = o[0] if isinstance(o, tuple) else o
    return h


def ce_nograd(core, l, mid, xp, ids, kw):
    torch = core.torch
    with torch.no_grad():
        return core.head_ce(suffix(core, l, mid + torch.nn.functional.linear(xp, core.M.layers[l].mlp.down_proj.weight), kw), ids)


def ce_grad(core, l, mid_j, xj, ids_j, kw):
    """differentiable summed CE of the windows in xj (float32 token CE summed in float64, as head_ce); also returns y."""
    torch = core.torch
    y = torch.nn.functional.linear(xj, core.M.layers[l].mlp.down_proj.weight)
    z = core.M.model.lm_head(core.M.model.model.norm(suffix(core, l, mid_j + y, kw)))
    ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids_j[:, 1:].reshape(-1), reduction='none')
    return ce.view(z.shape[0], -1).double().sum(1), y


def full_ce_hooked(core, l, ids, fn):
    """complete forward with layer l's Down input replaced by fn(original input)."""
    torch = core.torch
    hk = core.M.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a: (fn(a[0]),))
    with torch.no_grad():
        h = core.M.model.model(input_ids=ids, attention_mask=core.M.wrapper.mask, use_cache=False).last_hidden_state
        ce = core.head_ce(h, ids, normed=True)
    hk.remove()
    return ce


def grads_at(core, l, mid, xp, ids, kw, mb=MB):
    torch = core.torch
    out_, loss = torch.empty_like(xp), []
    for j in range(0, CAL, mb):
        xj = xp[j:j + mb].clone().requires_grad_(True)
        lj, _ = ce_grad(core, l, mid[j:j + mb], xj, ids[j:j + mb], kw)
        gj, = torch.autograd.grad(lj.sum(), xj)
        out_[j:j + mb] = gj
        loss.append(lj.detach())
    return out_, torch.cat(loss)


# --------------------------------------------------------------------------- #
# run (GPU)
# --------------------------------------------------------------------------- #
def run_layer(core, l, ids, tm):
    torch = core.torch
    lin = torch.nn.functional.linear
    node = down_node(l)
    core.ranges = read(NATIVE / 'ranges_by_point.json')[node]
    W = core.M.layers[l].mlp.down_proj.weight
    w_hash = hashlib.sha256(W.detach().cpu().numpy().tobytes()).hexdigest()
    t0 = time.perf_counter()
    ce_full, mid, x, y, kw = capture(core, l, ids)
    base = ce_nograd(core, l, mid, x, ids, kw)
    torch.cuda.synchronize()
    tm[f'L{l}|prefix_fp32_forward'] += time.perf_counter() - t0
    ctl = {'node': node, 'replay_baseline_vs_full_abs': float((base - ce_full).abs().max()), 'y_equals_linear_x': bool(torch.equal(lin(x, W), y)),
           'dtype': str(x.dtype), 'W_dtype': str(W.dtype)}
    M = {'all': torch.ones(1, SEQ, 1, dtype=torch.bool, device='cuda'),
         'P0': (torch.arange(SEQ, device='cuda') == 0)[None, :, None], 'Rest': (torch.arange(SEQ, device='cuda') > 0)[None, :, None]}
    # decomposition
    t0 = time.perf_counter()
    Q, E, MU, B, R = {}, {}, {}, {}, {}
    with torch.no_grad():
        for f in FORMATS:
            Q[f] = core.qact(x, f, 'x')
            E[f] = Q[f] - x
            MU[f] = E[f].double().sum((0, 1)) / N
            B[f] = MU[f].float().expand_as(x).contiguous()
            R[f] = (E[f].double() - MU[f]).float()
    torch.cuda.synchronize()
    tm[f'L{l}|qdq'] += time.perf_counter() - t0
    dec, xc = {}, x.cpu()
    for f in FORMATS:
        dec[f] = {'r_mean_abs_max': float((R[f].double().sum((0, 1)) / N).abs().max()), 'b_plus_r_minus_e_abs_max': float((B[f] + R[f] - E[f]).abs().max()),
                  'e_abs_max': float(E[f].abs().max()), 'mu_abs_max': float(MU[f].abs().max()), 'r_abs_max': float(R[f].abs().max()),
                  'r_P0_abs_max': float(R[f][:, 0].abs().max()), 'r_Rest_abs_max': float(R[f][:, 1:].abs().max()),
                  'xe_vs_Q_abs_max': float(((x + E[f]) - Q[f]).abs().max()), 'xe_vs_Q_fraction_differing': float(((x + E[f]) != Q[f]).float().mean()),
                  'xbr_vs_Q_abs_max': float(((x + B[f] + R[f]) - Q[f]).abs().max()), 'xbr_vs_Q_fraction_differing': float(((x + B[f] + R[f]) != Q[f]).float().mean())}
        if f.startswith('int'):
            qc_ = core.qact(xc, f, 'x')
            dec[f]['int_device_vs_cpu_abs_max'] = float((Q[f].cpu() - qc_).abs().max())
    # gradients (microbatch 1, create_graph) and HVPs
    dirs = {f'r|{f}|{m}': (R[f] * M[m]).contiguous() for f in FORMATS for m in MASKS}
    if l == 27:
        dirs.update({f'b|{f}|all': B[f] for f in FORMATS})            # reproduction of the previous round's F_b / K_b
    gen = torch.Generator(device='cuda').manual_seed(SYN_SEED)
    syn = torch.randn(x.shape, generator=gen, device='cuda', dtype=torch.float32)
    dirs['syn'] = (syn * (R['int4'].norm() / syn.norm())).contiguous()
    Fw = {k: np.zeros(CAL) for k in dirs}
    Kw = {k: np.zeros(CAL) for k in dirs}
    bHw = {k: np.zeros(CAL) for k in dirs if k.startswith('r|')}
    Hd = {k: torch.empty_like(x) for k in dirs if k.startswith('r|') or k == 'syn'}
    g = torch.empty_like(x)
    gyW_dev, loss_mb1, h127 = 0.0, [], 0.0
    t_g = t_h = 0.0
    for j in range(0, CAL, MB):
        t0 = time.perf_counter()
        xj = x[j:j + MB].clone().requires_grad_(True)
        lj, yj = ce_grad(core, l, mid[j:j + MB], xj, ids[j:j + MB], kw)
        gj, gyj = torch.autograd.grad(lj.sum(), (xj, yj), create_graph=True)
        torch.cuda.synchronize()
        t_g += time.perf_counter() - t0
        g[j:j + MB] = gj.detach()
        loss_mb1.append(lj.detach())
        with torch.no_grad():
            gyW_dev = max(gyW_dev, float((gj - gyj @ W).abs().max() / gj.abs().max()))
        t0 = time.perf_counter()
        for k, d in dirs.items():
            v = d[j:j + MB]
            Hv, = torch.autograd.grad((gj * v).sum(), xj, retain_graph=True)
            Fw[k][j:j + MB] = (gj.detach().double() * v.double()).sum((1, 2)).cpu().numpy()
            Kw[k][j:j + MB] = (0.5 * (v.double() * Hv.double()).sum((1, 2))).cpu().numpy()
            if k in bHw:
                f = k.split('|')[1]
                bHw[k][j:j + MB] = (B[f][j:j + MB].double() * Hv.double()).sum((1, 2)).cpu().numpy()
            if k in Hd:
                Hd[k][j:j + MB] = Hv
                h127 = max(h127, float(Hv[:, -1].abs().max()))
            del Hv
        torch.cuda.synchronize()
        t_h += time.perf_counter() - t0
        del gj, gyj, lj, yj, xj
    tm[f'L{l}|gradient_create_graph'] += t_g
    tm[f'L{l}|hvp_all_directions'] += t_h
    tm[f'L{l}|hvp_count'] += len(dirs) * CAL
    loss_mb1 = torch.cat(loss_mb1)
    # batch-size checks (first order)
    t0 = time.perf_counter()
    g2b, l2b = grads_at(core, l, mid, x, ids, kw, mb=2)
    g10, l10 = grads_at(core, l, mid, x, ids, kw, mb=CAL)
    ctl['batch'] = {'loss_mb1_vs_nograd_abs': float((loss_mb1 - base).abs().max()), 'loss_mb2_vs_nograd_abs': float((l2b - base).abs().max()),
                    'loss_mb10_vs_nograd_abs': float((l10 - base).abs().max()), 'g_mb2_vs_mb1_rel': float((g2b - g).abs().max() / g.abs().max()),
                    'g_mb10_vs_mb1_rel': float((g10 - g).abs().max() / g.abs().max())}
    del g2b, g10
    tm[f'L{l}|batch_checks'] += time.perf_counter() - t0
    ctl.update({'gx_vs_gyW_rel_max': gyW_dev, 'g_position127_abs_max': float(g[:, -1].abs().max()), 'H_position127_abs_max': h127})
    # proxies
    t0 = time.perf_counter()
    gd = g.double()
    g2 = gd ** 2
    mg2 = g2.sum((0, 1)) / N                                           # mean_(s,p) g^2 per channel (n = 1280)
    GS = (gd.sum(1) ** 2).sum(0)
    prox, chan = {}, {'mean_g2': mg2.cpu().numpy(), 'GS': GS.cpu().numpy()}
    for f in FORMATS:
        rd_ = R[f].double()
        e_ = E[f].double()
        mu = MU[f]
        prox[f'U|{f}'] = float((GS * mu ** 2).sum())
        prox[f'V_old_bc_arithmetic|{f}'] = float((mg2 * ((e_ ** 2).sum((0, 1)) - N * mu ** 2)).sum())
        for m in MASKS:
            rm = rd_ * M[m]
            gr = gd * rm
            Vc_old = mg2 * (rm ** 2).sum((0, 1))
            Vc_elem = (gr ** 2).sum((0, 1))
            prox[f'V_old|{f}|{m}'] = float(Vc_old.sum())
            prox[f'V_elem|{f}|{m}'] = float(Vc_elem.sum())
            prox[f'V_token|{f}|{m}'] = float((gr.sum(2) ** 2).sum())
            prox[f'V_window|{f}|{m}'] = float((gr.sum((1, 2)) ** 2).sum())
            prox[f'cov_term|{f}|{m}'] = float(((g2 - mg2) * rm ** 2).sum())
            prox[f'sum_r2|{f}|{m}'] = float((rm ** 2).sum())
            prox[f'V_token_per_window|{f}|{m}'] = (gr.sum(2) ** 2).sum(1).cpu().numpy().tolist()
            prox[f'sum_r2_per_window|{f}|{m}'] = (rm ** 2).sum((1, 2)).cpu().numpy().tolist()
            if m == 'all':
                chan[f'mu|{f}'] = mu.cpu().numpy()
                chan[f'sum_r2|{f}'] = (rm ** 2).sum((0, 1)).cpu().numpy()
                chan[f'V_old_c|{f}'] = Vc_old.cpu().numpy()
                chan[f'V_elem_c|{f}'] = Vc_elem.cpu().numpy()
    torch.cuda.synchronize()
    tm[f'L{l}|proxies'] += time.perf_counter() - t0
    # HVP vs central finite differences of the gradient
    t0 = time.perf_counter()
    fd = {}
    a_ = TOL['hvp_fd']
    near = lambda u, v_, rel, ab: abs(u - v_) <= rel * abs(v_) + ab
    for k in Hd:
        d = dirs[k]
        rows = []
        Kr = float(Kw[k].sum())
        Hk = Hd[k].double()
        hn = Hk.norm(dim=(1, 2))
        for h in HS:
            gp, _ = grads_at(core, l, mid, x + h * d, ids, kw)
            gm, _ = grads_at(core, l, mid, x - h * d, ids, kw)
            FD = (gp.double() - gm.double()) / (2 * h)
            Kfw = 0.5 * (d.double() * FD).sum((1, 2))
            K_fd = float(Kfw.sum())
            supp = d != 0
            rows.append({'h': h, 'K_fd': K_fd, 'vector_rel_err': float((FD - Hk).norm() / Hk.norm()) if float(Hk.norm()) > 0 else None,
                         'K_fd_minus_K_r': K_fd - Kr, 'changed_fraction_plus': float(((x + h * d) != x)[supp].float().mean()),
                         'K_fd_per_window': Kfw.cpu().numpy().tolist(),
                         'vector_rel_err_per_window': [float(u) if float(hn_) > 0 else None for u, hn_ in
                                                       zip(((FD - Hk).norm(dim=(1, 2)) / hn).cpu().numpy(), hn.cpu().numpy())]})
            del gp, gm, FD
        adj = [near(rows[i]['K_fd'], rows[i + 1]['K_fd'], a_['adjacent_rel'], a_['adjacent_abs_summed_ce']) for i in range(len(rows) - 1)]
        close = near(rows[-1]['K_fd'], Kr, a_['vs_hvp_rel'], a_['vs_hvp_abs_summed_ce'])
        # the same rule per window (machinery check: where the FD has converged it must match the HVP)
        win = []
        for w in range(CAL):
            kf = [r_['K_fd_per_window'][w] for r_ in rows]
            conv = all(near(kf[i], kf[i + 1], a_['adjacent_rel'], a_['adjacent_abs_summed_ce']) for i in range(len(kf) - 1))
            win.append({'converged': bool(conv), 'matches_hvp': bool(near(kf[-1], float(Kw[k][w]), a_['vs_hvp_rel'], a_['vs_hvp_abs_summed_ce']))})
        fd[k] = {'steps': rows, 'K_r': Kr, 'adjacent_converged': adj, 'fd_matches_hvp': bool(close), 'confirmed': bool(all(adj) and close),
                 'per_window': win}
    torch.cuda.synchronize()
    tm[f'L{l}|hvp_fd_check'] += time.perf_counter() - t0
    del Hd
    # curves and references
    t0 = time.perf_counter()
    dl, rnd = {}, {}
    L = lambda xp: (ce_nograd(core, l, mid, xp, ids, kw) - base).cpu().numpy()
    for f in FORMATS:
        dl[f'{f}|e_direct'] = L(Q[f])
        dl[f'{f}|b'] = L(x + B[f])
        dl[f'{f}|x+e'] = L(x + E[f])
        dl[f'{f}|x+b+r'] = L(x + B[f] + R[f])
        dl[f'{f}|r'] = L(x + R[f])
        for m in MASKS:
            d = dirs[f'r|{f}|{m}']
            supp = d != 0
            for a in ALPHAS:
                xp = x + a * d
                dl[f'{f}|{m}|{a:g}'] = L(xp)
                if a != 0:
                    ad = (a * d).double()
                    rnd[f'{f}|{m}|{a:g}'] = {'changed_fraction': float((xp != x)[supp].float().mean()),
                                             'rounding_rel': float(((xp.double() - x.double()) - ad).norm() / ad.norm())}
    torch.cuda.synchronize()
    tm[f'L{l}|curves_and_references'] += time.perf_counter() - t0
    # complete forward vs replay for every reference condition
    t0 = time.perf_counter()
    dev = 0.0
    for f in FORMATS:
        for key, fn in (('r', lambda a, f=f: a + R[f]), ('b', lambda a, f=f: a + B[f]), ('e_direct', lambda a, f=f: core.qact(a, f, 'x'))):
            c_ = (full_ce_hooked(core, l, ids, fn) - base).cpu().numpy()
            dev = max(dev, float(np.abs(c_ - dl[f'{f}|{key}']).max()))
    ctl['replay_vs_full_intervened_abs'] = dev
    tm[f'L{l}|replay_controls'] += time.perf_counter() - t0
    ctl.update({'W_down_sha256_before': w_hash, 'W_down_sha256_after': hashlib.sha256(W.detach().cpu().numpy().tobytes()).hexdigest(),
                'decomposition': dec, 'rounding': rnd,
                'rounding_small_alpha_min_changed_fraction': min(v['changed_fraction'] for k, v in rnd.items()),
                'rounding_max_rel': max(v['rounding_rel'] for v in rnd.values()),
                'alpha0_max_abs': max(float(np.abs(dl[f'{f}|{m}|0']).max()) for f in FORMATS for m in MASKS),
                'alpha1_all_equals_x_plus_r': all(bool(np.array_equal(dl[f'{f}|all|1'], dl[f'{f}|r'])) for f in FORMATS),
                'nonfinite': int(sum((~np.isfinite(v)).sum() for v in dl.values())) + int(sum((~np.isfinite(v)).sum() for v in Fw.values()))
                + int(sum((~np.isfinite(v)).sum() for v in Kw.values()))})
    raw = {'F_per_window': {k: v.tolist() for k, v in Fw.items()}, 'K_per_window': {k: v.tolist() for k, v in Kw.items()},
           'bH_per_window': {k: v.tolist() for k, v in bHw.items()}, 'proxies': prox, 'hvp_fd': fd}
    return ctl, raw, dl, base.cpu().numpy(), chan, MU


def run(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    core = rd.Core(out / 'work')
    torch = core.torch
    torch.cuda.reset_peak_memory_stats()
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(toks).cuda()
    ctl = {'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled(),
           'attn_implementation': core.M.model.config._attn_implementation, 'posit_lut': {f: list(core.q.posit[f].verify()) for f in core.q.posit},
           'calibration_tokens_equal_prepare': bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL])),
           'output_qdq': 'none (only the Down input is replaced; y = linear(x\', W))', 'layers': {}}
    raw, losses, chans, mus = {}, {}, {}, {}
    for l in LAYERS:
        c, raw[f'L{l}'], dl, base, chans[l], MU = run_layer(core, l, ids, tm)
        ctl['layers'][f'L{l}'] = c
        losses.update({f'L{l}|{k}': v for k, v in dl.items()})
        losses[f'L{l}|base'] = base
        mus.update({f'L{l}__{f}': MU[f].cpu().numpy() for f in FORMATS})
        log(f'L{l} done: fd confirmed {sum(v["confirmed"] for v in raw[f"L{l}"]["hvp_fd"].values())}/{len(raw[f"L{l}"]["hvp_fd"])}')
    # reproduction of old results
    old = np.load(PMB / 'losses.npz')
    prev = read(PMB / 'raw.json')
    rev = read(REV / 'calibration_scores.json')
    utv = np.load(UTV / 'calibration_losses.npz')
    spike = {(int(r['layer']), r['format']): r for r in csv.DictReader(open(SPIKE / 'scores.csv'))}
    rep = {'L27_losses_vs_posit_mean_bias_abs': {f: {k: float(np.abs(losses[f'L27|{f}|{k}'] - old[f'{f}__{k}']).max()) for k in ('r', 'b', 'e_direct')}
                                                 for f in FORMATS},
           'L27_Fb_Kb_vs_posit_mean_bias_rel': {f: {'F_b': abs(sum(raw['L27']['F_per_window'][f'b|{f}|all']) - sum(prev['F_per_window'][f'{f}|all']))
                                                    / abs(sum(prev['F_per_window'][f'{f}|all'])),
                                                    'K_b': abs(sum(raw['L27']['K_per_window'][f'b|{f}|all']) - sum(prev['K_per_window'][f'{f}|all']))
                                                    / abs(sum(prev['K_per_window'][f'{f}|all']))} for f in FORMATS},
           'L27_vs_reversal_diag_rel': {f: {'V_old': abs(raw['L27']['proxies'][f'V_old|{f}|all'] - rev[f'x|{f}|all']['V_old']) / abs(rev[f'x|{f}|all']['V_old']),
                                            'V_pair': abs(raw['L27']['proxies'][f'V_elem|{f}|all'] - (rev[f'x|{f}|all']['T_pair'] - rev[f'x|{f}|all']['U']))
                                            / abs(rev[f'x|{f}|all']['T_pair'] - rev[f'x|{f}|all']['U']),
                                            'U': abs(raw['L27']['proxies'][f'U|{f}'] - rev[f'x|{f}|all']['U']) / abs(rev[f'x|{f}|all']['U'])} for f in FORMATS},
           'vs_down_spike_diag_cpu_rel': {f'L{l}': {f: {k: abs(raw[f'L{l}']['proxies'][kk] - float(spike[(l, f)][k])) / abs(float(spike[(l, f)][k]))
                                                        for k, kk in (('V_old', f'V_old|{f}|all'), ('V_pair', f'V_elem|{f}|all'), ('U', f'U|{f}'))}
                                                    for f in FORMATS} for l in LAYERS},
           'L2_mean_only_vs_u_token_validation_abs': {f: float(np.abs(losses[f'L2|{f}|b'] - utv[f'L2__{f}__1']).max()) for f in FORMATS},
           'note': 'reversal diag: GPU, g_y W in 8-window sub-batches (T_pair = V_pair + U); down spike diag: CPU FP32 backend; '
                   'posit mean bias: same GPU path (F_b / K_b there in 2-window sub-batches, here microbatch 1)'}
    ctl['reproduction'] = rep
    t = TOL
    lc = ctl['layers']
    chk = {
        'environment': ctl['tf32'] == [False, False] and ctl['deterministic'] and ctl['attn_implementation'] == 'eager'
        and all(v[0] == 0 for v in ctl['posit_lut'].values()) and ctl['calibration_tokens_equal_prepare'],
        'replay': all(c['replay_baseline_vs_full_abs'] <= t['replay_vs_full_abs_summed_ce'] and c['replay_vs_full_intervened_abs']
                      <= t['replay_vs_full_abs_summed_ce'] for c in lc.values()),
        'slot_weights': all(c['y_equals_linear_x'] and c['W_down_sha256_before'] == c['W_down_sha256_after'] and c['dtype'] == 'torch.float32'
                            for c in lc.values()),
        'int_device': all(v['int_device_vs_cpu_abs_max'] <= t['int_device_vs_cpu_abs'] for c in lc.values() for f, v in c['decomposition'].items()
                          if f.startswith('int')),
        'gradients': all(c['gx_vs_gyW_rel_max'] <= t['gx_vs_gyW_rel'] and c['g_position127_abs_max'] <= t['pos127_abs']
                         and c['H_position127_abs_max'] <= t['pos127_abs'] for c in lc.values()),
        'batch': all(max(c['batch']['loss_mb1_vs_nograd_abs'], c['batch']['loss_mb2_vs_nograd_abs'], c['batch']['loss_mb10_vs_nograd_abs'])
                     <= t['batch_loss_abs_summed_ce'] and max(c['batch']['g_mb2_vs_mb1_rel'], c['batch']['g_mb10_vs_mb1_rel']) <= t['batch_grad_rel']
                     for c in lc.values()),
        'alpha0': all(c['alpha0_max_abs'] <= t['alpha0_abs'] and c['alpha1_all_equals_x_plus_r'] for c in lc.values()),
        'old_losses': all(v <= t['old_loss_abs_summed_ce'] for d in rep['L27_losses_vs_posit_mean_bias_abs'].values() for v in d.values())
        and all(v <= t['old_loss_abs_summed_ce'] for v in rep['L2_mean_only_vs_u_token_validation_abs'].values()),
        'old_same_backend': all(d['K_b'] <= t['old_same_backend_rel'] and d['F_b'] <= t['old_same_backend_signed_sum_rel']
                                for d in rep['L27_Fb_Kb_vs_posit_mean_bias_rel'].values())
        and all(d['V_old'] <= t['old_same_backend_rel'] and d['V_pair'] <= t['old_same_backend_rel'] and d['U'] <= t['old_same_backend_signed_sum_rel']
                for d in rep['L27_vs_reversal_diag_rel'].values()),
        'old_cpu_backend': all(v <= t['old_cpu_backend_rel'] for dl_ in rep['vs_down_spike_diag_cpu_rel'].values() for d in dl_.values() for v in d.values()),
        'nonfinite': all(c['nonfinite'] == 0 for c in lc.values())}
    # HVP machinery: L27 directions and synthetic confirmed in total; at both layers, windows whose FD converged match the HVP
    cells = [(l, k, w, c_['matches_hvp']) for l in LAYERS for k, v in raw[f'L{l}']['hvp_fd'].items() for w, c_ in enumerate(v['per_window'])
             if c_['converged']]
    ctl['hvp_machinery'] = {'L27_confirmed': sum(v['confirmed'] for v in raw['L27']['hvp_fd'].values()), 'L27_directions': len(raw['L27']['hvp_fd']),
                            'converged_window_cells': len(cells), 'matching': sum(c_[3] for c_ in cells),
                            'mismatching_cells': [f'L{c_[0]}|{c_[1]}|w{c_[2]}' for c_ in cells if not c_[3]],
                            'not_converged_cells': [f'L{l}|{k}|w{w}' for l in LAYERS for k, v in raw[f'L{l}']['hvp_fd'].items()
                                                    for w, c_ in enumerate(v['per_window']) if not c_['converged']],
                            'synthetic_confirmed': {f'L{l}': raw[f'L{l}']['hvp_fd']['syn']['confirmed'] for l in LAYERS}}
    hm = ctl['hvp_machinery']
    chk['hvp_machinery'] = bool(hm['L27_confirmed'] == hm['L27_directions'] and hm['converged_window_cells'] > 0
                                and hm['matching'] >= t['hvp_machinery_min_match_fraction'] * hm['converged_window_cells'])
    # identities / additivity
    add = {}
    for l in LAYERS:
        p_, Fk, Kk = raw[f'L{l}']['proxies'], raw[f'L{l}']['F_per_window'], raw[f'L{l}']['K_per_window']
        S = lambda k: float(np.sum(k))
        a = {}
        for f in FORMATS:
            Fa, Fp, Fr = S(Fk[f'r|{f}|all']), S(Fk[f'r|{f}|P0']), S(Fk[f'r|{f}|Rest'])
            Ka, Kp, Kr = S(Kk[f'r|{f}|all']), S(Kk[f'r|{f}|P0']), S(Kk[f'r|{f}|Rest'])
            a[f] = {'V_old_split_rel': abs(p_[f'V_old|{f}|P0'] + p_[f'V_old|{f}|Rest'] - p_[f'V_old|{f}|all']) / abs(p_[f'V_old|{f}|all']),
                    'V_old_two_arithmetics_rel': abs(p_[f'V_old|{f}|all'] - p_[f'V_old_bc_arithmetic|{f}']) / abs(p_[f'V_old|{f}|all']),
                    'cov_identity_rel': max(abs(p_[f'V_elem|{f}|{m}'] - p_[f'V_old|{f}|{m}'] - p_[f'cov_term|{f}|{m}']) / abs(p_[f'V_elem|{f}|{m}'])
                                            for m in MASKS),
                    'F_additivity_rel': abs(Fp + Fr - Fa) / max(abs(Fa), 1e-300), 'K_additivity_rel': abs(Kp + Kr - Ka) / max(abs(Ka), 1e-300),
                    'K_position_cross': Ka - Kp - Kr,
                    'I_position': {f'{al:g}': float((losses[f'L{l}|{f}|all|{al:g}'] - losses[f'L{l}|{f}|P0|{al:g}'] - losses[f'L{l}|{f}|Rest|{al:g}']).sum())
                                   for al in ALPHAS}}
        add[f'L{l}'] = a
    ctl['identities_and_additivity'] = add
    da = t['dL_additivity_L27']
    chk['identities'] = all(v['V_old_split_rel'] <= t['V_split_rel'] and v['cov_identity_rel'] <= t['cov_identity_rel']
                            and v['F_additivity_rel'] <= t['F_additivity_rel'] for d in add.values() for v in d.values())
    chk['L27_position_additivity'] = all(v['K_additivity_rel'] <= t['K_additivity_rel_L27'] and all(
        abs(v['I_position'][f'{al:g}']) <= da['abs_summed_ce'] + da['rel'] * abs(float(losses[f'L27|{f}|all|{al:g}'].sum())) for al in ALPHAS)
        for f, v in add['L27'].items())
    ctl['checks'] = chk
    ctl['passed'] = bool(all(chk.values()))
    ctl['note'] = ('HVP-FD convergence of r directions is reported per direction; a failure marks that K numerically undetermined, it '
                   'does not stop the run. L2 position interactions are reported, not pass / fail.')
    write(out / 'controls.json', ctl)
    np.savez(out / 'losses.npz', **{k.replace('|', '__'): np.asarray(v) for k, v in losses.items()})
    np.savez(out / 'mu.npz', **mus)
    write(out / 'raw.json', raw)
    for l in LAYERS:
        ch = chans[l]
        with open(out / f'channels_L{l}.csv', 'w', newline='') as fh:
            wr = csv.writer(fh)
            cols = ['mean_g2', 'GS'] + [f'{k}|{f}' for f in FORMATS for k in ('mu', 'sum_r2', 'V_old_c', 'V_elem_c')]
            wr.writerow(['channel', *cols])
            for c in range(len(ch['mean_g2'])):
                wr.writerow([c, *[ch[k][c] for k in cols]])
    tm['total'] = time.perf_counter() - t_all
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    write(out / 'timing.json', dict(tm))
    log(f'controls passed={ctl["passed"]}; checks {chk}; times {dict(tm)}')
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
    per = lambda v: v / TARGETS
    sa = TOL['small_alpha_support']
    truth = {(int(r['layer']), r['f1'], r['f2']): r for r in csv.DictReader(open(SPIKE / 'ranking_pairs.csv')) if r['primary'] == 'True'}
    T, curves, pos_rows, fd_rows, win_rows = {}, [], [], [], []
    for l in LAYERS:
        rw, p_ = raw[f'L{l}'], raw[f'L{l}']['proxies']
        F = {k: float(np.sum(v)) for k, v in rw['F_per_window'].items()}
        K = {k: float(np.sum(v)) for k, v in rw['K_per_window'].items()}
        bH = {k: float(np.sum(v)) for k, v in rw['bH_per_window'].items()}
        for f in FORMATS:
            for m in MASKS:
                k = f'r|{f}|{m}'
                S = lambda a: float(dl[f'L{l}|{f}|{m}|{a:g}'].sum())
                for a in ALPHAS:
                    if a <= 0:
                        continue
                    p, q = S(a), S(-a)
                    O_, E_ = (p - q) / 2, (p + q) / 2
                    rr = ctl['layers'][f'L{l}']['rounding']
                    curves.append({'layer': l, 'format': f, 'mask': m, 'alpha': a, 'dL_plus': per(p), 'dL_minus': per(q), 'O_over_alpha': per(O_ / a),
                                   'E_over_alpha2': per(E_ / a ** 2), 'F_r': per(F[k]), 'K_r': per(K[k]), 'R_plus': per(p - a * F[k] - a * a * K[k]),
                                   'R_minus': per(q + a * F[k] - a * a * K[k]), 'aF_plus_a2K': per(a * F[k] + a * a * K[k]),
                                   'a2_V_old_half': per(a * a * p_[f'V_old|{f}|{m}'] / 2), 'changed_fraction': rr[f'{f}|{m}|{a:g}']['changed_fraction'],
                                   'rounding_rel': rr[f'{f}|{m}|{a:g}']['rounding_rel']})
                pa, qa = S(sa['alpha']), S(-sa['alpha'])
                Ea, Oa = (pa + qa) / 2 / sa['alpha'] ** 2, (pa - qa) / 2 / sa['alpha']
                fdk = rw['hvp_fd'][k]
                T[(l, f, m)] = {
                    'V_old/2': per(p_[f'V_old|{f}|{m}'] / 2), 'V_elem/2': per(p_[f'V_elem|{f}|{m}'] / 2), 'V_token/2': per(p_[f'V_token|{f}|{m}'] / 2),
                    'V_window/2': per(p_[f'V_window|{f}|{m}'] / 2), 'F_r': per(F[k]), 'K_r': per(K[k]), 'F+K': per(F[k] + K[k]), 'bH_r': per(bH[k]),
                    'dL_r(1)': per(S(1.0)), 'dL_r(-1)': per(S(-1.0)), 'E(1)': per((S(1.0) + S(-1.0)) / 2), 'O(1)': per((S(1.0) - S(-1.0)) / 2),
                    'K_fd_confirmed': fdk['confirmed'], 'K_fd_steps': [s_['K_fd'] / TARGETS for s_ in fdk['steps']],
                    'small_alpha_K': bool(abs(Ea - K[k]) <= sa['rel'] * abs(K[k]) + sa['abs_summed_ce']),
                    'small_alpha_F': bool(abs(Oa - F[k]) <= sa['rel'] * abs(F[k]) + sa['abs_summed_ce']),
                    'E_small_over_a2': per(Ea), 'O_small_over_a': per(Oa),
                    'fd_windows_converged': sum(c_['converged'] for c_ in fdk['per_window']),
                    'fd_windows_converged_and_matching': sum(c_['converged'] and c_['matches_hvp'] for c_ in fdk['per_window']),
                    'E_over_a2_by_alpha': {f'{a:g}': per((S(a) + S(-a)) / 2 / a ** 2) for a in ALPHAS if a > 0},
                    'dL_by_alpha': {f'{a:g}': per(S(a)) for a in ALPHAS if a != 0}}
                for s_ in fdk['steps']:
                    fd_rows.append({'layer': l, 'direction': k, **s_, 'K_r': fdk['K_r'], 'adjacent_converged': fdk['adjacent_converged'],
                                    'fd_matches_hvp': fdk['fd_matches_hvp'], 'confirmed': fdk['confirmed']})
                for w in range(CAL):
                    win_rows.append({'layer': l, 'format': f, 'mask': m, 'window': w, 'F_r': rw['F_per_window'][k][w] / TP, 'K_r': rw['K_per_window'][k][w] / TP,
                                     'bH_r': rw['bH_per_window'][k][w] / TP, 'dL_r_plus1': float(dl[f'L{l}|{f}|{m}|1'][w]) / TP,
                                     'V_token_half': p_[f'V_token_per_window|{f}|{m}'][w] / 2 / TP, 'sum_r2': p_[f'sum_r2_per_window|{f}|{m}'][w]})
            T[(l, f, 'all')].update({'dL_e': per(float(dl[f'L{l}|{f}|e_direct'].sum())), 'dL_b': per(float(dl[f'L{l}|{f}|b'].sum())),
                                     'dL_x+e': per(float(dl[f'L{l}|{f}|x+e'].sum())), 'dL_x+b+r': per(float(dl[f'L{l}|{f}|x+b+r'].sum())),
                                     'U/2': per(p_[f'U|{f}'] / 2)})
            T[(l, f, 'all')]['I_br'] = T[(l, f, 'all')]['dL_e'] - T[(l, f, 'all')]['dL_b'] - T[(l, f, 'all')]['dL_r(1)']
            ad = ctl['identities_and_additivity'][f'L{l}'][f]
            pos_rows.append({'layer': l, 'format': f, **{f'{q}_{m}': T[(l, f, m)][q] for q in ('F_r', 'K_r', 'V_old/2', 'V_elem/2', 'V_token/2', 'V_window/2',
                                                                                              'dL_r(1)', 'dL_r(-1)') for m in MASKS},
                             'K_position_cross': per(ad['K_position_cross']), **{f'I_position({a})': per(v) for a, v in ad['I_position'].items()}})
        if 'b|int4|all' in F:
            for f in FORMATS:
                T[(l, f, 'all')]['F_b'] = per(F[f'b|{f}|all'])
                T[(l, f, 'all')]['K_b'] = per(K[f'b|{f}|all'])
        T[(l, 'syn', 'all')] = {'K_r': per(K['syn']), 'K_fd_confirmed': rw['hvp_fd']['syn']['confirmed']}
    sign = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)
    pairs = []
    for l in LAYERS:
        for a, b in PAIRS:
            A_, B_ = T[(l, a, 'all')], T[(l, b, 'all')]
            keys = ('V_old/2', 'V_elem/2', 'V_token/2', 'V_window/2', 'F_r', 'K_r', 'F+K', 'bH_r', 'dL_r(1)', 'E(1)', 'O(1)', 'dL_b', 'dL_e', 'I_br', 'U/2')
            d = {k: A_[k] - B_[k] for k in keys}
            kconf = A_['K_fd_confirmed'] and B_['K_fd_confirmed'] and A_['small_alpha_K'] and B_['small_alpha_K']
            fconf = A_['small_alpha_F'] and B_['small_alpha_F']
            flags = {}
            flags['A_curvature_proxy_mismatch'] = (('supported' if kconf else 'undetermined (K numerics)')
                                                   if sign(d['V_old/2']) != sign(d['K_r']) else 'no')
            flags['A_clues_agree_with_K'] = [k for k in ('V_elem/2', 'V_token/2', 'V_window/2') if sign(d[k]) == sign(d['K_r'])]
            flags['B_first_order'] = ((('supported' if kconf and fconf else 'undetermined (numerics)')
                                       if sign(d['K_r']) != sign(d['dL_r(1)']) and sign(d['F+K']) == sign(d['dL_r(1)']) else 'no'))
            flags['F_flips_local_model'] = bool(sign(d['F+K']) != sign(d['K_r']))
            flags['C_higher_order'] = ((('supported' if kconf and fconf else 'undetermined (numerics)')
                                        if sign(d['F+K']) != sign(d['dL_r(1)']) else 'no'))
            flags['C_even_part_flips'] = bool(sign(d['E(1)']) != sign(d['K_r']))
            flags['C_odd_part_flips'] = bool(sign(d['O(1)']) != sign(d['F_r']))
            flags['V_old_ranks_r_only_right'] = bool(sign(d['V_old/2']) == sign(d['dL_r(1)']))
            flags['r_only_order_equals_full_calibration'] = bool(sign(d['dL_r(1)']) == sign(d['dL_e']))
            # finite-amplitude orders at every alpha (reported, never picked)
            flags['E_order_by_alpha'] = {k: sign(A_['E_over_a2_by_alpha'][k] - B_['E_over_a2_by_alpha'][k]) for k in A_['E_over_a2_by_alpha']}
            flags['dL_order_by_alpha'] = {k: sign(A_['dL_by_alpha'][k] - B_['dL_by_alpha'][k]) for k in A_['dL_by_alpha']}
            flags['V_old_sign'] = sign(d['V_old/2'])
            tr = truth.get((l, a, b))
            pairs.append({'layer': l, 'pair': f'{a}-{b}', 'role': 'original full misrank' if (l, a, b) in MISRANKS else 'control',
                          'INT_minus_Posit': d, 'K_numerically_supported': bool(kconf), 'F_numerically_supported': bool(fconf),
                          'flags': flags, 'new_article_full_truth_INT_minus_Posit': float(tr['D_mean']) if tr else None,
                          'new_article_ci': [float(tr['ci_lo']), float(tr['ci_hi'])] if tr else None,
                          'orders': {k: ('INT larger' if v > 0 else 'Posit larger' if v < 0 else 'equal') for k, v in d.items()}})
    fmt_rows = [{'layer': l, 'format': f, **{k: v for k, v in T[(l, f, 'all')].items() if not isinstance(v, list)}} for l in LAYERS for f in FORMATS]
    res = {'units': 'nat/token (summed CE / 1270 targets); V-type proxies halved to compare with K_r', 'formats': fmt_rows, 'pairs': pairs,
           'synthetic': {f'L{l}': T[(l, 'syn', 'all')] for l in LAYERS}, 'controls_passed': ctl['passed'],
           'small_alpha_rule': sa, 'hvp_fd_rule': TOL['hvp_fd']}
    write(out / 'analysis.json', res)

    def dump(name, rows):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    dump('summary_table.csv', fmt_rows)
    dump('curves.csv', curves)
    dump('positions.csv', pos_rows)
    dump('hvp_fd.csv', fd_rows)
    dump('per_window.csv', win_rows)
    dump('pairs.csv', [{'layer': p['layer'], 'pair': p['pair'], 'role': p['role'], **{f'd_{k}': v for k, v in p['INT_minus_Posit'].items()},
                        'K_supported': p['K_numerically_supported'], 'F_supported': p['F_numerically_supported'], **p['flags'],
                        'new_article_truth': p['new_article_full_truth_INT_minus_Posit']} for p in pairs])
    # figures: measured r-only curves vs local model and V_old proxy; E/a^2 convergence to K
    for l in LAYERS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
        for col, (a, b) in enumerate(PAIRS):
            for f, c in ((a, '#1f77b4'), (b, '#d62728')):
                al = sorted(ALPHAS)
                Fk, Kk = T[(l, f, 'all')]['F_r'], T[(l, f, 'all')]['K_r']
                ax = axes[0, col]
                ax.plot(al, [per(float(dl[f'L{l}|{f}|all|{x_:g}'].sum())) for x_ in al], 'o-', color=c, ms=3, label=f'{f} measured dL_r(alpha)')
                grid = np.linspace(-1, 1, 201)
                ax.plot(grid, grid * Fk + grid ** 2 * Kk, '--', color=c, lw=1, label=f'{f} alpha F_r + alpha^2 K_r')
                ax.plot(grid, grid ** 2 * T[(l, f, 'all')]['V_old/2'], ':', color=c, lw=1, label=f'{f} alpha^2 V_old/2')
                ax2 = axes[1, col]
                pos_a = [x_ for x_ in ALPHAS if x_ > 0]
                ax2.plot(pos_a, [per((float(dl[f'L{l}|{f}|all|{x_:g}'].sum()) + float(dl[f'L{l}|{f}|all|{-x_:g}'].sum())) / 2 / x_ ** 2) for x_ in pos_a],
                         'o-', color=c, ms=3, label=f'{f} E(alpha)/alpha^2')
                ax2.axhline(Kk, color=c, ls='--', lw=1, label=f'{f} K_r')
                ax2.axhline(T[(l, f, 'all')]['V_old/2'], color=c, ls=':', lw=1, label=f'{f} V_old/2')
            axes[0, col].axhline(0, color='k', lw=.6)
            axes[0, col].set_title(f'L{l} Down input, residual direction r: {a} vs {b}', fontsize=9)
            axes[0, col].set_xlabel('alpha')
            axes[0, col].set_ylabel('nat/token')
            axes[0, col].legend(fontsize=6)
            axes[1, col].set_xscale('log', base=2)
            axes[1, col].set_yscale('symlog', linthresh=1e-4)
            axes[1, col].set_xlabel('alpha (> 0)')
            axes[1, col].set_ylabel('nat/token')
            axes[1, col].set_title('even response E(alpha)/alpha^2 vs K_r and V_old/2', fontsize=9)
            axes[1, col].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f'fig_r_curves_L{l}.png', dpi=110)
        plt.close(fig)
    tm = read(out / 'timing.json')
    tm['analysis'] = time.perf_counter() - t0
    write(out / 'timing.json', tm)
    log(json.dumps([{'layer': p['layer'], 'pair': p['pair'], 'role': p['role'], **{k: round(v, 5) for k, v in p['INT_minus_Posit'].items()},
                     'K_ok': p['K_numerically_supported'], 'F_ok': p['F_numerically_supported'], **p['flags']} for p in pairs], ensure_ascii=False, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'run', 'analyze'))
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
    else:
        {'run': run, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_posit_variance_diagnostic.py')}) + '\n')


if __name__ == '__main__':
    main()
