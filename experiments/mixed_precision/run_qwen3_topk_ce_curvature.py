#!/usr/bin/env python3
"""Qwen3-0.6B L27 Down: can a cheap top-k CE quadratic term repair the signed score's candidate ranking? Same target (A's INT8
weight), backgrounds U / F / W and nine activation formats as runs/qwen3_norm_ce_decomposition_20261026. Per window and target
position, with the baseline f16 probabilities p (float64 softmax of the gradient pass's FP32 logits), T_k = the k most probable
tokens (ties: lower token id), q = the tail mass, W = the (unquantized, tied) LM head and du = u_f - u_f16:
  a_v = W_v . du (v in T_k),  t = w_tail . du with w_tail = W^T (p * 1_tail)  (= q a_tail),  m = sum_T p_v a_v + t  (= (W^T p) . du)
  Khat_k = 1/2 [sum_T p_v (a_v - m)^2 + (t - q m)^2 / q]   (tail term 0 for q <= Q_EPS),   P_k = S_u + Khat_k
K2_affine - Khat_k = 1/2 q Var_tail(a) >= 0 (a lower bound of the affine Fisher quadratic, not of KL or the NLL damage).
Primary k = 256; k = 1024 is a calibration-only diagnostic. C = the original 10 calibration windows (scores, references,
controls, frozen selectors); N = 128 new articles x 2 windows (truth only). GPU (LMReplay QDQ); no change to the formal BC,
search, configs, quantizers or native pipeline.

Stages (one new --out-dir): plan, calib, bench, audit, data, eval, analyze. GPU stages (calib, bench, eval):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_conditional_signed_score as cs  # noqa: E402
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen3_norm_ce_decomposition as ncd  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
TARGET, ENDP, NATIVE = cs.TARGET, cs.ENDP, cs.NATIVE
PREV = HERE / 'runs/qwen3_norm_ce_decomposition_20261026'
BGS, FORMATS, SELECT_SET, POS = cs.BGS, cs.FORMATS, cs.SELECT_SET, cs.POS
SEQ, TP, CAL, BATCH, CHUNK = cs.SEQ, cs.TP, cs.CAL, cs.BATCH, cs.CHUNK
CANDS = tuple(f for f in SELECT_SET if f != 'f16')            # f16 reuses the baseline (du = 0, every K = 0) in every method
K_MAIN, K_DIAG = 256, 1024
Q_EPS = 1e-12
REF_BG, REF_WINDOWS = 'F', (0, 9)                             # float64 affine reference scope
N_ART, WPA, BOOT = 128, 2, 20000
SEEDS = {'order': 26102681, 'start': 26102682, 'bootstrap': 26102683}
BENCH = {'background': 'F', 'warmup': 5, 'reps': 20}
SEVERE = 0.01
REPRO_TERMS = ('D', 'S_z', 'S_u', 'KL', 'K2')
TOL = {'closure_vs_full_forward_abs_summed_ce': 1e-6, 'reproduce_prev_abs_per_position': 1e-12,
       'nested_abs': 1e-9, 'nested_rel': 1e-4, 'tail_identity_abs': 1e-9, 'tail_identity_rel': 1e-4,
       'cheap_vs_fullA_abs': 1e-9, 'cheap_vs_fullA_rel': 1e-4, 'fp32_vs_float64_abs': 1e-8, 'fp32_vs_float64_rel': 1e-4,
       'wbar_rel': 1e-5, 'synthetic_abs': 1e-12, 'position_sum_abs_summed': 1e-9, 'pos127_abs': 0.0, 'f16_zero_abs': 0.0}
CODE = ['run_qwen3_topk_ce_curvature.py'] + ncd.CODE
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def check_plan(out, inputs_too=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    if inputs_too:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def inputs():
    prev = [PREV / f for f in ('report.md', 'plan.json', 'scores_C.csv', 'C_decomp.npz', 'frozen_predictions.json', 'controls_calib.json',
                               'controls_H.json', 'timing.json', 'freeze.json')]
    return ncd.inputs() + prev + [HERE / 'reports/qwen3_topk_ce_curvature_prompt_20261026.md']


# --------------------------------------------------------------------------- #
# the approximation (shared by calib, bench and the synthetic controls)
# --------------------------------------------------------------------------- #
def topk_sel(p, k):
    """[P, V] -> boolean mask of exactly k tokens per row: every p > tau_k, then ties at tau_k by ascending token id."""
    import torch
    tau = torch.topk(p, k, dim=-1).values[:, -1:]
    gt = p > tau
    tie = p == tau
    need = k - gt.sum(-1, keepdim=True)
    return gt | (tie & (torch.cumsum(tie.to(torch.int64), -1) <= need))


def prep(p, W, k, sel=None):
    """baseline-only quantities for Khat_k (p float64 [P, V], W [V, d] in the GEMM dtype)."""
    sel = topk_sel(p, k) if sel is None else sel
    idx = sel.nonzero()[:, 1].view(p.shape[0], k)
    tail = p.masked_fill(sel, 0.0)
    return {'sel': sel, 'idx': idx, 'pT': p.gather(-1, idx), 'q': tail.sum(-1), 'w_tail': tail.to(W.dtype) @ W, 'WT': W[idx]}


def khat(pr, du):
    """du [C, P, d] (GEMM dtype) -> Khat [C, P] float64."""
    import torch
    aT = torch.einsum('pkd,cpd->cpk', pr['WT'], du).double()
    t = torch.einsum('pd,cpd->cp', pr['w_tail'], du).double()
    pT, q = pr['pT'], pr['q']
    m = (pT * aT).sum(-1) + t
    head = (pT * (aT - m[..., None]) ** 2).sum(-1)
    tail = torch.where(q > Q_EPS, (t - q * m) ** 2 / q.clamp_min(Q_EPS), torch.zeros_like(m))
    return 0.5 * (head + tail)


def full_terms(p, a, sels):
    """a = W du over the whole vocabulary [P, V] float64: K2_affine and, per k, Khat from the full a and the omitted tail variance."""
    import torch
    m = (p * a).sum(-1)
    dev2 = (a - m[:, None]) ** 2
    out = {'K2_aff': 0.5 * (p * dev2).sum(-1)}
    for k, sel in sels.items():
        tp = p.masked_fill(sel, 0.0)
        q = tp.sum(-1)
        abar = torch.where(q > Q_EPS, (tp * a).sum(-1) / q.clamp_min(Q_EPS), torch.zeros_like(q))
        head = 0.5 * (p.masked_fill(~sel, 0.0) * dev2).sum(-1)
        out[f'Khat{k}_fullA'] = head + 0.5 * torch.where(q > Q_EPS, q * (abar - m) ** 2, torch.zeros_like(q))
        out[f'tailvar{k}'] = 0.5 * (tp * (a - abar[:, None]) ** 2).sum(-1)
    return out


def synthetic():
    """small-vocabulary float64 checks of the same functions (CPU)."""
    import torch
    g = torch.Generator().manual_seed(0)
    V, d, P = 50, 8, 6
    W = torch.randn(V, d, generator=g, dtype=torch.float64)
    du = torch.randn(1, P, d, generator=g, dtype=torch.float64)
    lg = torch.randn(P, V, generator=g, dtype=torch.float64) * 2
    p = torch.softmax(lg, -1)
    a = du[0] @ W.T
    full = full_terms(p, a, {})
    res = {}
    kV = khat(prep(p, W, V), du)[0]
    res['k_eq_V_matches_full_variance'] = float((kV - full['K2_aff']).abs().max())
    res['k_eq_V_tail_mass'] = float(prep(p, W, V)['q'].abs().max())
    pr = prep(p, W, 10)
    k10 = khat(pr, du)[0]
    ft = full_terms(p, a, {10: pr['sel']})
    res['tail_identity'] = float((full['K2_aff'] - k10 - ft['tailvar10']).abs().max())
    res['cheap_vs_fullA'] = float((k10 - ft['Khat10_fullA']).abs().max())
    b = torch.randn(d, generator=g, dtype=torch.float64)
    res['common_logit_shift'] = float((khat(prep(p, W + b, 10), du)[0] - k10).abs().max())
    res['common_shift_K2'] = float((full_terms(p, du[0] @ (W + b).T, {})['K2_aff'] - full['K2_aff']).abs().max())
    lt = torch.zeros(1, V, dtype=torch.float64)                   # ties: all equal -> the k lowest ids
    lt[0, 7] = 1.0
    st = topk_sel(torch.softmax(lt, -1), 5)
    res['tie_rule_lowest_ids'] = bool(st[0].nonzero().flatten().tolist() == [0, 1, 2, 3, 7])
    lq = lg.clone()
    lq[:, -1] = -200.0                                            # q ~ e^-200 < Q_EPS
    pq = torch.softmax(lq, -1)
    prq = prep(pq, W, V - 1)
    kq = khat(prq, du)[0]
    res['q_eps_branch_finite'] = bool(torch.isfinite(kq).all())
    res['q_eps_branch_q_max'] = float(prq['q'].max())
    res['q_eps_branch_vs_full'] = float((kq - full_terms(pq, a, {})['K2_aff']).abs().max())
    res['nested'] = bool(torch.all(khat(prep(p, W, 5), du)[0] <= k10 + 1e-12) and torch.all(k10 <= full['K2_aff'] + 1e-12))
    res['passed'] = bool(all(res[k] <= TOL['synthetic_abs'] for k in ('k_eq_V_matches_full_variance', 'k_eq_V_tail_mass', 'tail_identity',
                                                                      'cheap_vs_fullA', 'common_logit_shift', 'common_shift_K2', 'q_eps_branch_vs_full'))
                         and res['tie_rule_lowest_ids'] and res['q_eps_branch_finite'] and res['q_eps_branch_q_max'] <= Q_EPS and res['nested'])
    return res


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or A[TARGET] != {'w': 'int8', 'a': 'f16'}:
        raise SystemExit('A is not the original knee')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {'time': now(), 'spec': 'reports/qwen3_topk_ce_curvature_prompt_20261026.md and the request of 2026-10-26 (main k = 256 fixed)',
          'scope': 'Qwen3-0.6B L27 Down only; target weight = A\'s INT8 in every background; backgrounds U / F / W, nine activation formats '
                   '(full input / output QDQ), raw x, residual r, Wq, ranges, LUT, epsilon, batch-8 / tail-batch GEMM shapes exactly as '
                   'runs/qwen3_norm_ce_decomposition_20261026 (its Env); GPU LMReplay, FP32, eager, TF32 off, deterministic',
          'formulas': {'S_u': 'g_u . (u_f - u_f16): g_u from one autograd call per window (microbatch 1), summed FP32 CE through final norm + '
                                'head only; float64 difference of the FP32 u, float64 product',
                       'p': 'float64 softmax of the FP32 logits of the same microbatch-1 baseline forward (the gradient pass); positions 0..126',
                       'T_k': 'every token with p > tau_k plus ties at tau_k by ascending token id (exactly k per position; based on p only)',
                       'Khat_k': '1/2 [sum_T p_v (a_v - m)^2 + q (a_tail - m)^2]; a_v = W_v . du (FP32 gather + batched GEMM, float64 '
                                 'reductions); q = sum of the tail p (float64, direct sum, no renormalization); q a_tail = w_tail . du with '
                                 'w_tail = W^T (p * 1_tail) (one FP32 dense GEMM per window at baseline); m = sum_T p_v a_v + q a_tail '
                                 '(= (W^T p) . du, i.e. w_bar = W_T^T p_T + w_tail); tail term = 0 when q <= ' + f'{Q_EPS}',
                       'du': 'u_f - u_f16 in FP32 for the GEMMs (S_u keeps the float64 difference)',
                       'P_k': 'S_u + Khat_k; k = 256 primary; k = 1024 C-only diagnostic (no N selector, no fallback)',
                       'references (C only)': {'K2_actual': '1/2 Var_p0(logits_f - logits_0) of the previous round (batch-8 FP32 head logits, '
                                                            'p0 from the batch-8 logits in float64)',
                                               'K2_affine': '1/2 Var_p(W du), same p as Khat, a = du W^T by one FP32 GEMM, float64 reductions',
                                               'KL, D': 'previous round (KL_64 on the batch-8 logits, D = FP32 CE difference)',
                                               'chain': 'K2_actual - K2_affine (head / subtraction / probability source), K2_affine - Khat '
                                                        '(omitted tail variance, also computed directly as 1/2 q Var_tail), KL - K2_actual '
                                                        '(quadratic remainder)'},
                       'float64 reference': f'background {REF_BG}, windows {list(REF_WINDOWS)}, all positions and formats: W, du, a, w_tail in '
                                            'float64 (same p and T_k); explicit float64 W^T p vs the implied w_bar; g_u + W_y vs W^T p reported'},
          'controls': ['LM head = untied-from-search affine Linear, no bias / softcap / temperature / hook; final norm Qwen3RMSNorm eps 1e-6',
                       f'closure = full forward (C, 3 backgrounds x 9 formats) within {TOL["closure_vs_full_forward_abs_summed_ce"]}',
                       f'previous C per-position D, S_z, S_u, KL, K2 reproduced within {TOL["reproduce_prev_abs_per_position"]}; S_u selector',
                       'gradient chain (VJP and analytic RMSNorm Jacobian, previous tolerances); position-127 g_z / g_u = 0 (Khat uses the 127 '
                       'target positions only)', 'f16: du, S, every K and D exactly 0; fp32 not forced to 0',
                       'exactly k tokens per position; T_256 subset of T_1024',
                       f'nested Khat256 <= Khat1024 <= K2_affine within {TOL["nested_abs"]} + {TOL["nested_rel"]} K2_affine (no clipping)',
                       f'K2_affine - Khat_k = tail variance within {TOL["tail_identity_abs"]} + {TOL["tail_identity_rel"]} K2_affine; cheap '
                       f'Khat = Khat from the full a within {TOL["cheap_vs_fullA_abs"]} + {TOL["cheap_vs_fullA_rel"]} Khat',
                       f'FP32 vs float64 reference within {TOL["fp32_vs_float64_abs"]} + {TOL["fp32_vs_float64_rel"]} |ref|; implied w_bar vs '
                       f'float64 W^T p within rel {TOL["wbar_rel"]}', 'synthetic vocabulary (k = V full variance, q = 0 and q < Q_EPS branches, '
                       'common logit shift, tie rule, tail identity, nesting) within ' + f'{TOL["synthetic_abs"]}',
                       f'P0 + Rest = All within {TOL["position_sum_abs_summed"]}; microbatch-1 vs batch-8 baseline logits difference reported',
                       'failures are kept; a bug fix needs an amendment with reason, time and hashes'],
          'calibration_report': 'every background / format / P0 / Rest / All: S_z, S_u, Khat256, Khat1024, K2_affine, K2_actual, KL, P256, D; '
                                'tail mass and omitted tail variance; directions vs f16 (8 non-f16), 28 pair orders of the selection set, '
                                'same-bit INT / Posit pairs, argmin and margin for S_u, P256, P1024 (diagnostic), S_u + K2_actual',
          'selectors (frozen on C)': {'set': list(SELECT_SET), 'primary': 'argmin P256,All', 'reference': 'argmin (S_u + K2_actual),All',
                                      'old': 'argmin S_u,All', 'BC control': 'f16', 'ties': 'exact equality: f16 first, then the set order',
                                      'not evaluated on N': 'P1024 (C diagnostic only)'},
          'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'use': 'truth only (per-position CE, 3 backgrounds x 9 formats); no '
                   'gradient, proxy, range, k or selector computed or changed on N', 'rule': 'run_qwen_prefix_rank_shift.data (verified '
                   'manifest exclusion, WT2 / previous / accepted near-duplicates, historical Qwen shingle audit with the subset resolution of '
                   'runs/qwen3_conditional_signed_score_20261026)', 'excluded_previous_wt103': len(used),
                   'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
          'statistics': {'unit': 'article (2 windows averaged into a 254-target contribution)', 'bootstrap': f'{BOOT} paired article resamples, '
                         f'seed {SEEDS["bootstrap"]}, one index set for every background / format / selector difference; 95% percentile CI',
                         'primary': 'F background: N mean D of the frozen P256 pick relative to f16 (= BC); confirmed improvement only if mean < 0 '
                                    'and CI upper < 0; an f16 pick is 0 and not an improvement',
                         'secondary (estimates, not success claims)': ['primary pick - old S_u pick (F)', 'primary pick - reference pick (F)',
                                                                        'U / W picks vs f16', 'F truth: U pick - F pick',
                                                                        'frozen P256 / S_u / S_u+K2_actual directions (8), 28 pairs, 3 same-bit '
                                                                        'pairs: right / wrong / undetermined by CI', 'descriptive N oracle',
                                                                        'P0 / Rest and per-article signs of every pick'],
                         'severe harm': f'mean > {SEVERE} nat/token and CI lower > 0'},
          'cost': {'protocol': f'background {BENCH["background"]}, C (10 windows), the {len(CANDS)} non-f16 selection candidates (f16 reuses the '
                               f'baseline in every method); CUDA synchronize only at region boundaries; {BENCH["warmup"]} warmups + '
                               f'{BENCH["reps"]} timed repetitions; median / p90',
                   'methods': {'Su': 'baseline gradient pass + candidates z / norm / S_u',
                               'Su+Khat256': 'Su + baseline p, top-k, q, w_tail GEMM, W_T gather (cached) + candidate batched GEMM / reductions',
                               'Su+K2full': 'Su + baseline p (FP32 copy kept) + per candidate full FP32 GEMM a = du W^T and FP32 Var_p',
                               'directCE': 'baseline batch-8 head CE + per candidate batch-8 head + FP32 CE'},
                   'levels': ['cold C pipeline (replay init, cold background set, first prefix capture, first run of each method in the fixed '
                              'order Su, Su+Khat256, Su+K2full, directCE)', 'prefix prepared: baseline + candidates', 'baseline cached: candidate '
                              'hot increment'], 'also': 'baseline sub-steps of Su+Khat256; peak VRAM increment per method; cache sizes; CPU RSS; '
                                                        'float64 diagnostics, controls and N evaluation listed apart'},
          'tolerances': TOL, 'q_eps': Q_EPS, 'smoke': smoke or 'none',
          'boundaries': 'no change to the formal BC / raw BC, quantizers, configs, MILP / knee or native; no training, STE, HVP, new layer, '
                        'multiplier, clipping penalty, tail-variance compensation, bit threshold or other k; results are L27 GPU candidate '
                        'scoring only',
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L27 Down 的 top-k CE 二次項（k = 256）', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        '- 範圍與語意：與 runs/qwen3_norm_ce_decomposition_20261026 相同（L27 Down、目標 INT8、U／F／W、九格式、batch 8／尾批）。',
        '- p：同一個 microbatch-1 baseline forward（梯度那次）的 FP32 logits 做 float64 softmax；T_k 依 p 取前 k（平手取較小 token id），不重新正規化。',
        '- Khat_k = ½[Σ_T p(a−m)² + q(a_tail−m)²]，q·a_tail = w_tail·du（w_tail = W^T(p·1_tail)，baseline 一次 FP32 GEMM），m = Σ_T p a + q a_tail；q ≤ 1e-12 時 tail 項為 0。',
        '- 主分數 P256 = S_u + Khat256；k = 1024 只在 C 作診斷，不進 N 選擇器。',
        '- C 參考：K2_actual（前輪）、K2_affine（同 p 的完整 W du）、KL、D；分開 K2_actual−K2_affine、K2_affine−Khat（tail 方差）、KL−K2_actual。',
        '- 凍結（只用 C）：primary argmin P256、reference argmin(S_u+K2_actual)、old argmin S_u、BC = f16；平手 f16 優先。',
        f'- N：新抽 {N_ART} 篇 × {WPA} 窗口，只量真值；seed {SEEDS}；排除 {len(used)} 篇已用文章。',
        '- 主要問題：F 背景 primary pick 相對 f16 的 N 平均 D；mean < 0 且 CI 上界 < 0 才算確認改善。其餘為次要估計。',
        f'- 成本：F 背景 C、{len(CANDS)} 個非 f16 候選；Su／Su+Khat256／Su+K2full／directCE；冷、含 baseline、候選熱增量；5 warmup + 20 次，median／p90。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; used {len(used)}, eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class Env(ncd.Env):
    def setup_head(self):
        hf = self.rp.ad.model
        self.nm, self.lm = hf.model.norm, hf.lm_head
        self.W = self.lm.weight.data                                   # [V, d] FP32 (tied embedding, unquantized)

    def grad_pass(self, r, z0, ids, on_logits=None):
        """one autograd call per window (microbatch 1, as the previous rounds): g_z, g_u; on_logits(j, FP32 logits [127, V]) right after."""
        torch = self.torch
        F_ = torch.nn.functional
        gz, gu = torch.empty_like(z0), torch.empty_like(z0)
        for j in range(len(r)):
            zj = z0[j:j + 1].detach().clone().requires_grad_(True)
            u0 = self.nm(r[j:j + 1] + zj)
            lg = self.lm(u0)
            loss = F_.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            gz[j:j + 1], gu[j:j + 1] = torch.autograd.grad(loss, (zj, u0))
            if on_logits is not None:
                with torch.no_grad():
                    on_logits(j, lg[0, :-1].detach())
        return gz, gu

    def norm_all(self, r, z):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(r), BATCH):
                out.append(self.nm(r[i:i + BATCH] + z[i:i + BATCH]))
        return self.torch.cat(out)


def gsum(a, G):
    return {'P0': a[:, 0], 'Rest': a[:, 1:].sum(1), 'All': a.sum(1)}[G]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def calib(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    t0 = time.perf_counter()
    syn = synthetic()
    tm['synthetic_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    env = Env(out / 'work')
    env.setup_head()
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    ctl = defaultdict(float)
    ctl['synthetic'] = syn
    ctl['head'] = env.head_checks()
    if not ctl['head']['passed'] or not syn['passed']:
        write(out / 'controls_calib.json', ctl)
        raise SystemExit(f'head or synthetic control failed: {ctl["head"]} {syn}')
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    prev = np.load(PREV / 'C_decomp.npz')
    W = env.W
    res = {}
    ctl['target_weight_int8'], ctl['ref'] = {}, {}
    ctl['kl_min'] = 0.0
    for b in BGS:
        t0 = time.perf_counter()
        cfg = cs.background(A, b)
        env.set_background(cfg)
        torch.cuda.synchronize()
        tm[f'{b}|set_background_s'] = time.perf_counter() - t0
        ctl['target_weight_int8'][b] = bool(cfg[TARGET]['w'] == 'int8' and torch.equal(env.Wq, env.rp.cache[(TARGET, 'int8')].to(env.Wq.device)))
        t0 = time.perf_counter()
        full0, x, r = env.full(cal, 'f16', capture=True)
        z = {f: env.zf(x, f) for f in FORMATS}
        torch.cuda.synchronize()
        tm[f'{b}|prefix_and_z_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        L = {}
        gz, gu = env.grad_pass(r, z['f16'], ids, on_logits=lambda j, l: L.__setitem__(j, l.clone()))
        torch.cuda.synchronize()
        tm[f'{b}|grad_pass_s'] = time.perf_counter() - t0
        e_vjp, e_ana = env.chain_check(r, z['f16'], gz, gu)
        ctl['chain_vjp_rel'] = max(ctl['chain_vjp_rel'], e_vjp)
        ctl['chain_analytic_rel'] = max(ctl['chain_analytic_rel'], e_ana)
        ctl['pos127_g_abs'] = max(ctl['pos127_g_abs'], float(gz[:, -1].abs().max()), float(gu[:, -1].abs().max()))
        # previous decomposition terms (D, S_z, S_u, KL, K2_actual, ...)
        t0 = time.perf_counter()
        acc = {f: defaultdict(list) for f in FORMATS}
        c = env.decompose(r, z, gz, gu, ids, tm, acc, f'{b}|')
        for k, v in c.items():
            ctl[k] = min(ctl[k], v) if k == 'kl_min' else max(ctl[k], v)
        R = ncd.per_pos(acc)
        tm[f'{b}|previous_decomposition_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for f in FORMATS:
            cf = env.closure(r, z[f], ids)
            ff = env.full(cal, f)
            ctl['closure_vs_full_abs'] = max(ctl['closure_vs_full_abs'], float((cf.sum(1) - ff.sum(1)).abs().max()))
        tm['closure_control_s'] += time.perf_counter() - t0
        # baseline logits: microbatch 1 (p source) vs the batch-8 closure logits (truth / KL source)
        with torch.no_grad():
            for i in range(0, CAL, BATCH):
                l8 = env.lm(env.nm(r[i:i + BATCH] + z['f16'][i:i + BATCH]))[:, :-1]
                for j in range(l8.shape[0]):
                    ctl['logits_mb1_vs_b8_abs'] = max(ctl['logits_mb1_vs_b8_abs'], float((L[i + j] - l8[j]).abs().max()))
        u = {f: env.norm_all(r, z[f]) for f in FORMATS}
        new = {f: defaultdict(list) for f in FORMATS}
        qs = {K_MAIN: [], K_DIAG: []}
        t0 = time.perf_counter()
        for j in range(CAL):
            with torch.no_grad():
                p = torch.softmax(L[j].double(), -1)
                sel = {K_MAIN: topk_sel(p, K_MAIN), K_DIAG: topk_sel(p, K_DIAG)}
                ctl['topk_count_ok'] = bool(ctl.get('topk_count_ok', True) and all(bool((s.sum(-1) == k).all()) for k, s in sel.items()))
                ctl['nested_sets_ok'] = bool(ctl.get('nested_sets_ok', True) and not bool((sel[K_MAIN] & ~sel[K_DIAG]).any()))
                pr = {k: prep(p, W, k, sel[k]) for k in sel}
                for k in sel:
                    qs[k].append(pr[k]['q'].cpu().numpy())
                du = torch.stack([u[f][j, :-1] - u['f16'][j, :-1] for f in FORMATS])
                kc = {k: khat(pr[k], du) for k in sel}
                for fi, f in enumerate(FORMATS):
                    a = (du[fi] @ W.T).double()
                    ft = full_terms(p, a, sel)
                    new[f]['Khat256'].append(kc[K_MAIN][fi].cpu().numpy())
                    new[f]['Khat1024'].append(kc[K_DIAG][fi].cpu().numpy())
                    for k_, v_ in ft.items():
                        new[f][k_].append(v_.cpu().numpy())
                    del a, ft
                if b == REF_BG and j in REF_WINDOWS:
                    t1 = time.perf_counter()
                    ref_window(env, p, sel, pr, u, j, gu, ids, ctl, kc)
                    tm['float64_reference_s'] += time.perf_counter() - t1
                del p, pr, du, kc
        torch.cuda.synchronize()
        tm[f'{b}|topk_and_references_s'] = time.perf_counter() - t0
        for f in FORMATS:
            for k_, v_ in new[f].items():
                R[f][k_] = np.stack(v_)
        for k in qs:
            R[f'q{k}'] = np.stack(qs[k])
        res[b] = R
        log(f'{b}: F-style All per token ' + ', '.join(f'{f} S_u {R[f]["S_u"].sum() / (CAL * TP):+.4f} Khat256 {R[f]["Khat256"].sum() / (CAL * TP):+.4f}'
                                                   for f in ('p8e1', 'int8')))
        del x, r, z, u, L, gz, gu
    # controls over all backgrounds
    rep = 0.0
    for b in BGS:
        for f in FORMATS:
            for t in REPRO_TERMS:
                rep = max(rep, float(np.abs(res[b][f][t] - prev[f'{b}__{f}__{t}']).max()))
    ctl['reproduce_prev_abs'] = rep
    nest = tail = cheap = -np.inf
    psum = f16 = 0.0
    nest_viol, raw = [], defaultdict(float)
    for b in BGS:
        for f in FORMATS:
            R = res[b][f]
            K2a = R['K2_aff']
            sc = TOL['nested_abs'] + TOL['nested_rel'] * np.abs(K2a)
            v1 = R['Khat256'] - R['Khat1024'] - sc
            v2 = R['Khat1024'] - K2a - sc
            nest = max(nest, float(np.max(v1)), float(np.max(v2)))
            raw['Khat256_minus_Khat1024_max'] = max(raw['Khat256_minus_Khat1024_max'], float(np.max(R['Khat256'] - R['Khat1024'])))
            raw['Khat1024_minus_K2aff_max'] = max(raw['Khat1024_minus_K2aff_max'], float(np.max(R['Khat1024'] - K2a)))
            if (v1 > 0).any() or (v2 > 0).any():
                nest_viol.append(f'{b}|{f}')
            for k in (K_MAIN, K_DIAG):
                tail = max(tail, float(np.max(np.abs(K2a - R[f'Khat{k}_fullA'] - R[f'tailvar{k}'])
                                              - TOL['tail_identity_abs'] - TOL['tail_identity_rel'] * np.abs(K2a))))
                raw['tail_identity_max_abs'] = max(raw['tail_identity_max_abs'], float(np.max(np.abs(K2a - R[f'Khat{k}_fullA'] - R[f'tailvar{k}']))))
                raw['cheap_vs_fullA_max_abs'] = max(raw['cheap_vs_fullA_max_abs'], float(np.max(np.abs(R[f'Khat{k}'] - R[f'Khat{k}_fullA']))))
                cheap = max(cheap, float(np.max(np.abs(R[f'Khat{k}'] - R[f'Khat{k}_fullA']) - TOL['cheap_vs_fullA_abs']
                                                - TOL['cheap_vs_fullA_rel'] * np.abs(R[f'Khat{k}_fullA']))))
            for t in ('S_u', 'Khat256', 'Khat1024', 'K2_aff', 'K2', 'KL', 'D'):
                a = R[t]
                psum = max(psum, abs(a[:, 0].sum() + a[:, 1:].sum() - a.sum()))
            if f == 'f16':
                f16 = max(f16, max(float(np.abs(R[t]).max()) for t in ('S_u', 'S_z', 'Khat256', 'Khat1024', 'K2_aff', 'K2', 'KL', 'D')))
    ctl['nested_excess'] = nest                 # <= 0 passes (max of value - tolerance)
    ctl['nested_violations'] = nest_viol
    ctl['raw_max'] = dict(raw)
    ctl['tail_identity_excess'] = tail
    ctl['cheap_vs_fullA_excess'] = cheap
    ctl['position_sum_abs'] = psum
    ctl['f16_abs_new_terms'] = f16
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    ref = ctl['ref']
    ctl['passed'] = bool(ctl['head']['passed'] and syn['passed'] and all(ctl['target_weight_int8'].values())
                         and ctl['closure_vs_full_abs'] <= TOL['closure_vs_full_forward_abs_summed_ce']
                         and rep <= TOL['reproduce_prev_abs_per_position'] and ctl['chain_vjp_rel'] <= ncd.TOL['chain_autograd_vjp_rel']
                         and ctl['chain_analytic_rel'] <= ncd.TOL['chain_analytic_float64_rel'] and ctl['pos127_g_abs'] == 0
                         and ctl['pos127_S_abs'] == 0 and ctl['f16_abs'] == 0 and f16 == 0 and ctl['topk_count_ok'] and ctl['nested_sets_ok'] and nest <= 0
                         and tail <= 0 and cheap <= 0 and psum <= TOL['position_sum_abs_summed'] and ref.get('fp32_vs_float64_excess', 1) <= 0
                         and ref.get('wbar_rel', 1) <= TOL['wbar_rel'] and ctl['tf32'] == [False, False] and ctl['deterministic']
                         and ctl['identity_full_abs'] <= ncd.TOL['identity_full_abs_per_position'] and ctl['kl_min'] >= ncd.TOL['kl_min_per_position'])
    np.savez(out / 'C_terms.npz', **{f'{b}__{f}__{k}': v for b in BGS for f in FORMATS for k, v in res[b][f].items()},
             **{f'{b}__q{k}': res[b][f'q{k}'] for b in BGS for k in (K_MAIN, K_DIAG)})
    summ = summarize_C(res)
    write(out / 'controls_calib.json', ctl)
    write(out / 'frozen_predictions.json', summ['frozen'])
    write(out / 'calibration_summary.json', {k: v for k, v in summ.items() if k != 'frozen'})
    with open(out / 'scores_C.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summ['rows'][0]))
        wr.writeheader()
        wr.writerows(summ['rows'])
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib', tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; picks {summ["frozen"]["selectors"]}; rep {rep:.1e}; nest {nest:.1e}; tail {tail:.1e}; '
        f'cheap {cheap:.1e}; ref {ref}')
    if not ctl['passed']:
        raise SystemExit(f'calibration controls failed: { {k: v for k, v in ctl.items() if k not in ("synthetic", "head")} }')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_calib.json'),
                                'predictions_sha256': sha(out / 'frozen_predictions.json'), 'scores_sha256': sha(out / 'scores_C.csv'),
                                'summary_sha256': sha(out / 'calibration_summary.json'), 'terms_sha256': sha(out / 'C_terms.npz'),
                                'code_sha256': {f: sha(HERE / f) for f in CODE}})


def ref_window(env, p, sel, pr, u, j, gu, ids, ctl, kc):
    """float64 affine reference for one window of the reference background (same p and T_k)."""
    torch = env.torch
    ref = ctl['ref']
    if not hasattr(env, 'W64'):
        env.W64 = env.W.double()
    W64 = env.W64
    pr64 = {k: prep(p, W64, k, sel[k]) for k in sel}
    du64 = torch.stack([u[f][j, :-1].double() - u['f16'][j, :-1].double() for f in FORMATS])
    k64 = {k: khat(pr64[k], du64) for k in sel}
    exc = ref.get('fp32_vs_float64_excess', -np.inf)
    worst = ref.get('fp32_vs_float64_max_abs', 0.0)
    for fi, f in enumerate(FORMATS):
        a64 = du64[fi] @ W64.T
        K2a64 = full_terms(p, a64, {})['K2_aff']
        K2a32 = full_terms(p, ((u[f][j, :-1] - u['f16'][j, :-1]) @ env.W.T).double(), {})['K2_aff']      # as the main loop
        for k in sel:
            d = (kc[k][fi] - k64[k][fi]).abs()
            exc = max(exc, float((d - TOL['fp32_vs_float64_abs'] - TOL['fp32_vs_float64_rel'] * k64[k][fi].abs()).max()))
            worst = max(worst, float(d.max()))
        d = (K2a32 - K2a64).abs()
        exc = max(exc, float((d - TOL['fp32_vs_float64_abs'] - TOL['fp32_vs_float64_rel'] * K2a64.abs()).max()))
        worst = max(worst, float(d.max()))
    ref['fp32_vs_float64_excess'] = exc
    ref['fp32_vs_float64_max_abs'] = worst
    wbar64 = p @ W64
    pk = pr[K_MAIN]
    implied = (pk['WT'].double() * pk['pT'][..., None]).sum(1) + pk['w_tail'].double()
    ref['wbar_rel'] = max(ref.get('wbar_rel', 0.0), float((implied - wbar64).abs().max() / wbar64.abs().max()))
    y = ids[j, 1:]
    alt = gu[j, :-1].double() + W64[y]                           # g_u + W_y (FP32 CE softmax) vs explicit float64 W^T p
    ref['gu_plus_Wy_vs_wbar64_rel'] = max(ref.get('gu_plus_Wy_vs_wbar64_rel', 0.0), float((alt - wbar64).abs().max() / wbar64.abs().max()))
    ref['windows'] = list(REF_WINDOWS)
    ref['background'] = REF_BG


def summarize_C(res):
    n = CAL * TP
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def tot(b, f, t, G='All'):
        R = res[b][f]
        if t == 'P256':
            return tot(b, f, 'S_u', G) + tot(b, f, 'Khat256', G)
        if t == 'P1024':
            return tot(b, f, 'S_u', G) + tot(b, f, 'Khat1024', G)
        if t == 'Su+K2actual':
            return tot(b, f, 'S_u', G) + tot(b, f, 'K2', G)
        if t == 'Su+K2affine':
            return tot(b, f, 'S_u', G) + tot(b, f, 'K2_aff', G)
        return float(gsum(R[t], G).sum()) / n
    cols = ('S_z', 'S_u', 'Khat256', 'Khat1024', 'K2_aff', 'K2', 'KL', 'P256', 'P1024', 'Su+K2actual', 'D', 'tailvar256', 'tailvar1024')
    rows = []
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                r = {'background': b, 'format': f, 'G': G}
                r.update({c: tot(b, f, c, G) for c in cols})
                r['K2_actual_minus_K2_affine'] = r['K2'] - r['K2_aff']
                r['K2_affine_minus_Khat256'] = r['K2_aff'] - r['Khat256']
                r['KL_minus_K2_actual'] = r['KL'] - r['K2']
                r['D_minus_P256'] = r['D'] - r['P256']
                rows.append(r)
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    nonf16 = [f for f in FORMATS if f != 'f16']
    scores = ('S_u', 'P256', 'P1024', 'Su+K2actual', 'Su+K2affine')
    rank, sel = {}, {}
    for b in BGS:
        rank[b] = {'truth_order': sorted(SELECT_SET, key=lambda f: tot(b, f, 'D'))}
        sel[b] = {}
        for s in scores:
            v = {f: tot(b, f, s) for f in SELECT_SET}
            order = sorted(SELECT_SET, key=lambda f: (v[f], SELECT_SET.index(f)))
            rank[b][s] = {'order': order, 'direction_right': {f: sgn(tot(b, f, s)) == sgn(tot(b, f, 'D')) for f in nonf16},
                          'pairs_right_of_28': sum(sgn(v[a] - v[c]) == sgn(tot(b, a, 'D') - tot(b, c, 'D')) for a, c in pairs),
                          'same_bit': {f'{i_}-{p_}': {'score': v[i_] - v[p_], 'truth': tot(b, i_, 'D') - tot(b, p_, 'D'),
                                                      'right': sgn(v[i_] - v[p_]) == sgn(tot(b, i_, 'D') - tot(b, p_, 'D'))}
                                       for i_, p_ in (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1'))},
                          'pick': order[0], 'margin_to_next': v[order[1]] - v[order[0]], 'pick_C_truth': tot(b, order[0], 'D')}
            rank[b][s]['direction_right_n'] = sum(rank[b][s]['direction_right'].values())
        sel[b] = {'primary_P256': rank[b]['P256']['pick'], 'reference_Su+K2actual': rank[b]['Su+K2actual']['pick'],
                  'old_Su': rank[b]['S_u']['pick'], 'BC': 'f16', 'diagnostic_only_P1024': rank[b]['P1024']['pick']}
    tails = {}
    for b in BGS:
        for k in (K_MAIN, K_DIAG):
            q = res[b][f'q{k}']
            tails[f'{b}|k{k}'] = {'P0_mean': float(q[:, 0].mean()), 'P0_max': float(q[:, 0].max()), 'Rest_mean': float(q[:, 1:].mean()),
                                  'Rest_median': float(np.median(q[:, 1:])), 'Rest_p90': float(np.percentile(q[:, 1:], 90)),
                                  'Rest_max': float(q[:, 1:].max()), 'min': float(q.min()), 'n_le_q_eps': int((q <= Q_EPS).sum()),
                                  'frac_gt_0.1': float((q > 0.1).mean()), 'frac_gt_0.5': float((q > 0.5).mean())}
    frozen = {'selectors': sel, 'ranking': rank, 'rule': 'argmin over the selection set, exact ties: f16 first then the set order',
              'scores_per_token_All': {b: {s: {f: tot(b, f, s) for f in FORMATS} for s in scores + ('Khat256', 'K2', 'D')} for b in BGS},
              'note': 'C only; frozen before any N data; P1024 and Su+K2affine are C diagnostics (no N selector)'}
    return {'rows': rows, 'frozen': frozen, 'tail_mass': tails}


# --------------------------------------------------------------------------- #
# timing (C, F background, after the freeze)
# --------------------------------------------------------------------------- #
def bench(out):
    t_all = time.perf_counter()
    check_plan(out)
    fz = read(out / 'freeze.json')
    if sha(out / 'frozen_predictions.json') != fz['predictions_sha256']:
        raise SystemExit('freeze changed')
    tm = {}
    t0 = time.perf_counter()
    env = Env(out / 'work')
    env.setup_head()
    torch = env.torch
    F_ = torch.nn.functional
    sync = torch.cuda.synchronize
    tm['replay_init_s'] = time.perf_counter() - t0
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    t0 = time.perf_counter()
    env.set_background(cs.background(A, BENCH['background']))
    sync()
    tm['set_background_cold_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, x, r = env.full(cal, 'f16', capture=True)
    z0 = env.zf(x, 'f16')
    sync()
    tm['prefix_capture_cold_s'] = time.perf_counter() - t0
    W, nm, lm = env.W, env.nm, env.lm
    V = W.shape[0]
    y = ids[:, 1:]

    def u0_all():
        return env.norm_all(r, z0)

    def base_su(extra=None):
        u0 = u0_all()
        gz, gu = env.grad_pass(r, z0, ids, on_logits=extra)
        return {'u0': u0, 'gu': gu}

    def base_khat():
        cache = {}

        def on(j, l):
            p = torch.softmax(l.double(), -1)
            cache[j] = prep(p, W, K_MAIN)
            del cache[j]['sel']
        st = base_su(on)
        st['pr'] = [cache[j] for j in range(CAL)]
        return st

    def base_k2():
        cache = {}

        def on(j, l):
            cache[j] = torch.softmax(l.double(), -1).float()
        st = base_su(on)
        st['p32'] = [cache[j] for j in range(CAL)]
        return st

    def base_ce():
        with torch.no_grad():
            u0 = u0_all()
            ce0 = []
            for i in range(0, CAL, BATCH):
                lg = lm(u0[i:i + BATCH])
                ce0.append(F_.cross_entropy(lg[:, :-1].reshape(-1, V), y[i:i + BATCH].reshape(-1), reduction='none').view(lg.shape[0], -1))
            return {'u0': u0, 'ce0': torch.cat(ce0)}

    def cands(st):
        with torch.no_grad():
            return torch.stack([env.norm_all(r, env.zf(x, f)) for f in CANDS])          # [C, windows, 128, d]

    def cand_su(st):
        U = cands(st)
        with torch.no_grad():
            S = (st['gu'].double() * (U.double() - st['u0'].double())).sum(-1)
        return U, S

    def cand_khat(st):
        U, S = cand_su(st)
        with torch.no_grad():
            K = torch.stack([khat(st['pr'][j], U[:, j, :-1] - st['u0'][j, :-1]) for j in range(CAL)], 1)
        return S, K

    def cand_k2(st):
        U, S = cand_su(st)
        out_ = []
        with torch.no_grad():
            for j in range(CAL):
                a = (U[:, j, :-1] - st['u0'][j, :-1]) @ W.T                                  # [C, 127, V] FP32
                pj = st['p32'][j]
                m = (a * pj).sum(-1)
                out_.append(0.5 * (((a - m[..., None]) ** 2) * pj).sum(-1))
                del a
        return S, torch.stack(out_, 1)

    def cand_ce(st):
        U = cands(st)
        out_ = []
        with torch.no_grad():
            for ci in range(len(CANDS)):
                ce = []
                for i in range(0, CAL, BATCH):
                    lg = lm(U[ci, i:i + BATCH])
                    ce.append(F_.cross_entropy(lg[:, :-1].reshape(-1, V), y[i:i + BATCH].reshape(-1), reduction='none').view(lg.shape[0], -1))
                out_.append(torch.cat(ce) - st['ce0'])
        return torch.stack(out_)

    methods = {'Su': (base_su, lambda st: cand_su(st)[1]), 'Su+Khat256': (base_khat, cand_khat), 'Su+K2full': (base_k2, cand_k2),
               'directCE': (base_ce, cand_ce)}
    cold, outs = {}, {}
    for name, (bf, cf) in methods.items():
        sync()
        t0 = time.perf_counter()
        st = bf()
        outs[name] = cf(st)
        sync()
        cold[name] = time.perf_counter() - t0
        del st
    tm['cold_first_run_s'] = cold
    tm['cold_pipeline_s'] = {k: tm['replay_init_s'] + tm['set_background_cold_s'] + tm['prefix_capture_cold_s'] + v for k, v in cold.items()}

    def timed(fn):
        for _ in range(BENCH['warmup']):
            fn()
        ts = []
        for _ in range(BENCH['reps']):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t0)
        return {'median_s': float(np.median(ts)), 'p90_s': float(np.percentile(ts, 90)), 'min_s': float(min(ts)), 'n': len(ts)}
    warm, mem = {}, {}
    for name, (bf, cf) in methods.items():
        torch.cuda.empty_cache()
        sync()
        static = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        warm[name] = {'total_baseline_plus_candidates': timed(lambda: cf(bf())), 'baseline_only': timed(bf)}
        st = bf()
        warm[name]['candidates_hot_increment'] = timed(lambda: cf(st))
        mem[name] = {'peak_bytes': int(torch.cuda.max_memory_allocated()), 'increment_over_static_bytes': int(torch.cuda.max_memory_allocated() - static)}
        del st
    # baseline sub-steps of Su+Khat256 (logits of one gradient pass cached)
    Ls = {}
    env.grad_pass(r, z0, ids, on_logits=lambda j, l: Ls.__setitem__(j, l.clone()))
    P = [torch.softmax(Ls[j].double(), -1) for j in range(CAL)]
    S = [topk_sel(P[j], K_MAIN) for j in range(CAL)]
    sub = {'grad_pass_with_u0': timed(lambda: base_su()),
           'softmax_float64': timed(lambda: [torch.softmax(Ls[j].double(), -1) for j in range(CAL)]),
           'topk_tie_mask': timed(lambda: [topk_sel(P[j], K_MAIN) for j in range(CAL)]),
           'index_pT_q': timed(lambda: [(S[j].nonzero(), P[j].masked_fill(S[j], 0.0).sum(-1)) for j in range(CAL)]),
           'w_tail_dense_gemm': timed(lambda: [P[j].masked_fill(S[j], 0.0).float() @ W for j in range(CAL)]),
           'row_gather_WT': timed(lambda: [W[S[j].nonzero()[:, 1].view(TP, K_MAIN)] for j in range(CAL)])}
    del Ls, P, S
    # consistency of the timed implementations with the frozen calibration values (descriptive)
    fp = read(out / 'frozen_predictions.json')['scores_per_token_All'][BENCH['background']]
    n = CAL * TP
    cons = {'Su': max(abs(float(outs['Su'][ci].sum()) / n - fp['S_u'][f]) for ci, f in enumerate(CANDS)),
            'Khat256': max(abs(float(outs['Su+Khat256'][1][ci].sum()) / n - fp['Khat256'][f]) for ci, f in enumerate(CANDS)),
            'K2full_fp32_vs_K2affine_float64red': None, 'directCE_vs_D': max(abs(float(outs['directCE'][ci].double().sum()) / n - fp['D'][f])
                                                                             for ci, f in enumerate(CANDS))}
    C = np.load(out / 'C_terms.npz')
    cons['K2full_fp32_vs_K2affine_float64red'] = max(abs(float(outs['Su+K2full'][1][ci].double().sum()) / n
                                                         - float(C[f'{BENCH["background"]}__{f}__K2_aff'].sum()) / n) for ci, f in enumerate(CANDS))
    sizes = {'vocab': int(V), 'hidden': int(W.shape[1]), 'k': K_MAIN, 'windows': CAL, 'target_positions': TP, 'candidates': len(CANDS),
             'WT_cache_bytes_all_windows': CAL * TP * K_MAIN * W.shape[1] * 4, 'p32_cache_bytes_all_windows': CAL * TP * V * 4,
             'w_tail_bytes_all_windows': CAL * TP * W.shape[1] * 4, 'pT_idx_bytes_all_windows': CAL * TP * K_MAIN * (8 + 8)}
    tm.update({'warm': warm, 'baseline_substeps_Su+Khat256': sub, 'gpu_memory': mem, 'consistency_per_token_abs': cons, 'sizes': sizes,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'device': torch.cuda.get_device_name(0)})
    write(out / 'bench.json', tm)
    ca.log_stage(out, 'bench', time.perf_counter() - t_all)
    log('bench: ' + ', '.join(f'{k} total {v["total_baseline_plus_candidates"]["median_s"]:.3f}s / hot {v["candidates_hot_increment"]["median_s"]:.3f}s'
                              for k, v in warm.items()))


# --------------------------------------------------------------------------- #
# new articles
# --------------------------------------------------------------------------- #
def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    cs.LOGF = LOGF
    try:
        getattr(prs, stage)(out)
    except SystemExit as e:
        if stage != 'audit' or 'unresolved' not in str(e):
            raise
        cs.resolve_subsets(out)
    ca.log_stage(out, stage, time.perf_counter() - t0)


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz_ = read(out / 'data_freeze.json')
    fz = read(out / 'freeze.json')
    if sha(out / 'tokens.npy') != dz_['tokens_sha256'] or sha(out / 'freeze.json') != dz_['freeze_sha256'] or \
            sha(out / 'frozen_predictions.json') != fz['predictions_sha256']:
        raise SystemExit('data / freeze changed')
    env = Env(out / 'work')
    torch = env.torch
    A = read(ENDP / 'configs/A_auto.json')
    tok = np.load(out / 'tokens.npy')
    res = {f'{b}__{f}': np.zeros((len(tok), TP)) for b in BGS for f in FORMATS}
    dev = 0.0
    for b in BGS:
        env.set_background(cs.background(A, b))
        for c0 in range(0, len(tok), CHUNK):
            t = tok[c0:c0 + CHUNK]
            ids = torch.from_numpy(t).cuda()
            full0, x, r = env.full(t, 'f16', capture=True)
            for f in FORMATS:
                ce = env.closure(r, env.zf(x, f), ids)
                res[f'{b}__{f}'][c0:c0 + len(t)] = ce.cpu().numpy()
                if f == 'f16':
                    dev = max(dev, float((ce.sum(1) - full0.sum(1)).abs().max()))
            del x, r
        log(f'{b} done')
    np.savez(out / 'eval_ce_pos.npz', **res)
    write(out / 'controls_eval.json', {'f16_closure_vs_full_abs': dev, 'passed': bool(dev <= TOL['closure_vs_full_forward_abs_summed_ce'])})
    ca.log_stage(out, 'eval', time.perf_counter() - t0, {'windows': len(tok)})
    if dev > TOL['closure_vs_full_forward_abs_summed_ce']:
        raise SystemExit('closure differs from the full forward on new windows')


def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    z = np.load(out / 'eval_ce_pos.npz')
    fp = read(out / 'frozen_predictions.json')
    na = N_ART
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]) / na
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def st(v):
        bs = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'median': float(np.median(v)),
                'articles_pos': int((v > 0).sum()), 'articles_neg': int((v < 0).sum()), 'min': float(v.min()), 'max': float(v.max())}

    def decisive(s):
        return s['ci95'][0] > 0 or s['ci95'][1] < 0

    def part(ce, G):
        c = ce.reshape(na, WPA, TP)
        return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[G] / (WPA * TP)
    D = {(b, f, G): part(z[f'{b}__{f}'], G) - part(z[f'{b}__f16'], G) for b in BGS for f in FORMATS for G in POS}
    sel = fp['selectors']
    sc = fp['scores_per_token_All']
    severe = lambda s: bool(s['mean'] > SEVERE and s['ci95'][0] > 0)
    picks = {}
    for b in BGS:
        picks[b] = {}
        for role in ('primary_P256', 'reference_Su+K2actual', 'old_Su', 'BC'):
            f = sel[b][role]
            s = st(D[(b, f, 'All')])
            picks[b][role] = {'format': f, 'N_D_All': s, 'N_D_P0': st(D[(b, f, 'P0')]), 'N_D_Rest': st(D[(b, f, 'Rest')]), 'severe_harm': severe(s)}
    pf = sel['F']['primary_P256']
    s_prim = st(D[('F', pf, 'All')])
    primary = {'pick': pf, 'N_D_vs_f16': s_prim, 'confirmed_improvement': bool(pf != 'f16' and s_prim['mean'] < 0 and s_prim['ci95'][1] < 0),
               'note': 'f16 pick = 0, not an improvement' if pf == 'f16' else ''}
    secondary = {}
    for b in BGS:
        pb = sel[b]['primary_P256']
        secondary[f'{b}|primary-old'] = st(D[(b, pb, 'All')] - D[(b, sel[b]['old_Su'], 'All')])
        secondary[f'{b}|primary-reference'] = st(D[(b, pb, 'All')] - D[(b, sel[b]['reference_Su+K2actual'], 'All')])
    secondary['F_truth|U_pick-F_pick'] = {'U_pick': sel['U']['primary_P256'], 'F_pick': pf,
                                          'N': st(D[('F', sel['U']['primary_P256'], 'All')] - D[('F', pf, 'All')])}
    nonf16 = [f for f in FORMATS if f != 'f16']
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    rank = {}
    for b in BGS:
        rb = {}
        for s_ in ('P256', 'S_u', 'Su+K2actual'):
            v = sc[b][s_]
            dirs = {}
            for f in nonf16:
                s = st(D[(b, f, 'All')])
                dirs[f] = 'undetermined' if not decisive(s) else ('right' if sgn(v[f]) == sgn(s['mean']) else 'wrong')
            pc = Counter()
            for a, c in pairs:
                s = st(D[(b, a, 'All')] - D[(b, c, 'All')])
                pc['undetermined' if not decisive(s) else ('right' if sgn(v[a] - v[c]) == sgn(s['mean']) else 'wrong')] += 1
            same = {}
            for i_, p_ in (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1')):
                s = st(D[(b, i_, 'All')] - D[(b, p_, 'All')])
                same[f'{i_}-{p_}'] = {'score': v[i_] - v[p_], 'N': s,
                                      'judgement': 'undetermined' if not decisive(s) else ('right' if sgn(v[i_] - v[p_]) == sgn(s['mean']) else 'wrong')}
            rb[s_] = {'directions': dirs, 'direction_counts': dict(Counter(dirs.values())), 'pairs': dict(pc), 'same_bit': same}
        rb['N_oracle_in_set_descriptive'] = min(SELECT_SET, key=lambda f: float(D[(b, f, 'All')].mean()))
        rb['N_mean_D_All'] = {f: float(D[(b, f, 'All')].mean()) for f in FORMATS}
        rank[b] = rb
    res = {'units': 'nat/token contributions relative to f16 of the same background; article bootstrap', 'primary': primary, 'picks': picks,
           'secondary': secondary, 'rankings': rank}
    write(out / 'stats.json', res)
    rows = [{'background': b, 'format': f, 'G': G, **{k: v for k, v in st(D[(b, f, G)]).items() if k != 'ci95'},
             'ci_lo': st(D[(b, f, G)])['ci95'][0], 'ci_hi': st(D[(b, f, G)])['ci95'][1]} for b in BGS for f in FORMATS for G in POS]
    with open(out / 'N_truth.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(out / 'N_article_D.npz', **{f'{b}__{f}__{G}': D[(b, f, G)] for b in BGS for f in FORMATS for G in POS})
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log(f'primary F pick {pf}: N D {s_prim["mean"]:+.5f} {s_prim["ci95"]}; confirmed {primary["confirmed_improvement"]}')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'bench', 'audit', 'data', 'eval', 'analyze'))
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
        {'calib': calib, 'bench': bench, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_topk_ce_curvature.py')}) + '\n')


if __name__ == '__main__':
    main()
