#!/usr/bin/env python3
"""Qwen3-0.6B L27 Down: does the first-order score S_z fail in the final RMSNorm or in the output CE? Same target (A's INT8 weight),
backgrounds U / F / W and nine activation formats as runs/qwen3_conditional_signed_score_20261026. With z_f = Q_y(linear(Q_x(x), Wq)),
h_f = r + z_f, u_f = final_norm(h_f), logits_f = LM_head(u_f) and one baseline autograd call per window giving g_z = dL/dz and
g_u = dL/du at f16:
  S_z = g_z . (z_f - z_0),  S_u = g_u . (u_f - u_0),  R_norm = S_u - S_z,  D = S_u + KL(p_0 || p_f)   (affine head, exact reals)
so D - S_z = R_norm + KL; floating-point residuals are listed apart from the same FP32 logits in float64:
  eps_head = S_logit,64 - S_u,  eps_CE = D_FP32CE - D_64,  D_64 = S_logit,64 + KL_64.
C = the original 10 calibration windows (scores, decomposition, frozen S_u selector); H = the previous round's 128 articles x 2
windows, already seen: local gradients per window, post hoc mechanism analysis and an exploratory selector replay only.
GPU (LMReplay QDQ); no change to the formal BC, search, configs, quantizers or native pipeline.

Stages (one new --out-dir): plan, calib (C), hdecomp (H), analyze. GPU stages (calib, hdecomp):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_conditional_signed_score as cs  # noqa: E402
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
TARGET, ENDP, NATIVE = cs.TARGET, cs.ENDP, cs.NATIVE
PREV = HERE / 'runs/qwen3_conditional_signed_score_20261026'
BGS, FORMATS, SELECT_SET, BITS, POS = cs.BGS, cs.FORMATS, cs.SELECT_SET, cs.BITS, cs.POS
SEQ, TP, CAL, N_ART, WPA, BATCH, CHUNK = cs.SEQ, cs.TP, cs.CAL, cs.N_ART, cs.WPA, cs.BATCH, cs.CHUNK
BOOT = 20000
SEEDS = {'bootstrap': 26102671}
TERMS = ('D', 'S_z', 'S_u', 'R_norm', 'KL', 'eps_head', 'eps_CE', 'D64', 'S_logit64', 'K2')
DESC = ('rms_ratio', 'dz_rel', 'du_rel')          # descriptive per position: sigma_f / sigma_0 - 1, |dz| / |z_0|, |du| / |u_0|
SEVERE = 0.01                                     # nat/token: a pick with H mean D above this (CI > 0) is called severely harmful
TOL = {'closure_vs_full_forward_abs_summed_ce': 1e-6, 'reproduce_prev_ce_abs_per_position': 1e-6,
       'reproduce_prev_S_z_D_abs_per_token': 1e-9, 'chain_autograd_vjp_rel': 1e-6, 'chain_analytic_float64_rel': 1e-4,
       'identity_D64_abs_per_position': 1e-9, 'identity_full_abs_per_position': 1e-9, 'kl_min_per_position': -1e-10,
       'position_sum_abs_summed': 1e-9, 'pos127_abs': 0.0, 'f16_zero_abs': 0.0,
       'rounding_flag_abs_summed_ce': 1e-3, 'rounding_flag_rel': 0.01}
CODE = ['run_qwen3_norm_ce_decomposition.py'] + cs.CODE
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
    prev = [PREV / f for f in ('report.md', 'scores.csv', 'frozen_predictions.json', 'stats.json', 'eval_ce_pos.npz', 'controls_calib.json',
                               'controls_eval.json', 'segments.json', 'tokens.npy', 'freeze.json', 'data_freeze.json', 'plan.json')]
    return cs.inputs() + prev + [HERE / 'reports/qwen3_norm_ce_decomposition_prompt_20261026.md']


def plan(out, smoke):
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or A[TARGET] != {'w': 'int8', 'a': 'f16'}:
        raise SystemExit('A is not the original knee')
    dfz = read(PREV / 'data_freeze.json')
    if sha(PREV / 'tokens.npy') != dfz['tokens_sha256'] or sha(PREV / 'segments.json') != dfz['segments_sha256']:
        raise SystemExit('H tokens / segments differ from the previous data freeze')
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {'time': now(), 'spec': 'reports/qwen3_norm_ce_decomposition_prompt_20261026.md and the request of 2026-10-26',
          'scope': 'Qwen3-0.6B L27 Down only; target weight = A\'s INT8 in every background; backgrounds U / F / W, formats and full input / '
                   'output QDQ semantics exactly as runs/qwen3_conditional_signed_score_20261026 (its Env: raw x before the input QDQ, '
                   'z_f in the full forward\'s batches of 8); GPU LMReplay',
          'formulas': {'z': 'z_f = Q_y,f(linear(Q_x,f(x_b), Wq)); z_0 = z_f16', 'u': 'u_f = final_RMSNorm(r_b + z_f) (eps 1e-6, FP32)',
                       'gradients': 'one autograd.grad(loss, (z_leaf, u_0)) per window (microbatch 1, as the previous round), loss = summed '
                                    'FP32 CE of 127 targets through final norm + LM head only (no STE, no QDQ backward)',
                       'S_z': 'sum g_z . (z_f - z_0) (previous convention: FP32 difference, float64 product)',
                       'S_u': 'sum g_u . (u_f - u_0) (float64 difference of the FP32 u)', 'R_norm': 'S_u - S_z (signed)',
                       'D': 'FP32 cross_entropy(logits_f) - cross_entropy(logits_0) per position (previous closure; logits in batches of 8)',
                       'float64 on the same FP32 logits': 'D_64 = CE64_f - CE64_0 (log_softmax); S_logit,64 = (p_0 - onehot(y)) . (l_f - l_0); '
                                                          'KL_64 = sum_v p_0 (logp_0 - logp_f) (baseline -> candidate, not clipped); '
                                                          'eps_head = S_logit,64 - S_u; eps_CE = D - D_64',
                       'identities': 'D_64 = S_logit,64 + KL_64; D = S_z + R_norm + KL_64 + eps_head + eps_CE (per position)',
                       'descriptive (not a score, no selector)': 'K2 = 0.5 Var_p0(l_f - l_0) (the logit-space quadratic part of KL); '
                                                                 'rms_ratio = sigma(h_f) / sigma(h_0) - 1; |dz| / |z_0|; |du| / |u_0|'},
          'groups': 'P0 = position 0, Rest = 1..126 (127 targets); position 127 has no target (gradients must be 0), kept in the 128-position '
                    'arrays',
          'data': {'C': 'runs/native_eps_fix_20261014/qwen3/calib_inputs.npy (10 x 128): scores, decomposition, frozen S_u selector',
                   'H': {'source': 'runs/qwen3_conditional_signed_score_20261026/tokens.npy (128 articles x 2 windows, same order)',
                         'tokens_sha256': dfz['tokens_sha256'], 'segments_sha256': dfz['segments_sha256'],
                         'status': 'already seen (previous round\'s truth): post hoc mechanism analysis only, not a new confirmation',
                         'local_scores': 'per window own-background baseline gradients (local S_z / S_u) for the exact per-article '
                                         'decomposition; never the 10-window calibration averages'}},
          'analysis': {'main': 'F background All: int8, p6e1, p4e1, p8e1 (C and H separately), then P0 / Rest, U / W, all formats',
                       'per item': 'D, S_z, S_u, R_norm, KL, eps_head, eps_CE (+ K2, D-S_z, D-S_u); H: article mean, median, sign counts, '
                                   'range, cancellation (|R_norm| + |KL| vs |R_norm + KL|, articles with R_norm < 0)',
                       'H bootstrap': f'{BOOT} article resamples, one index set for all backgrounds / formats, seed {SEEDS["bootstrap"]}, '
                                      'descriptive (post hoc); C has no bootstrap',
                       'reading': 'Norm correction restores a direction if sign(S_u) = sign(D) where sign(S_z) != sign(D) (C); KL dominates '
                                  'the error if KL is the larger of |R_norm|, KL and D - S_u stays wrong in sign; both may matter, per format',
                       'rankings': 'direction vs f16 (8 non-f16 formats) and pair orders over the 8-format selection set (28 pairs) for '
                                   'S_z and S_u vs C truth; frozen (C) scores vs H truth: CI-decisive directions / pairs right / wrong; '
                                   'same-bit INT / Posit pairs'},
          'selector': {'rule': f'per background argmin S_u,All on C over {list(SELECT_SET)} (a bits <= 16, target weight unchanged; exact '
                               'ties: f16, then that order); fp32 is a control, not a candidate; frozen before H',
                       'H replay (exploratory)': 'frozen S_u picks vs frozen S_z picks (previous round) vs the original BC (f16): H D and CI; '
                                                 'F truth of argmin S_u,F vs argmin S_u,U; same-bit orders; severely harmful = H mean D > '
                                                 f'{SEVERE} nat/token with CI > 0',
                       'boundary': 'an S_u improvement only justifies a new-article confirmation; no change of selector, coefficient or '
                                   'threshold after seeing H; a failure stays a failure (KL / true D are not a new proxy)'},
          'controls': ['A hash; target INT8 weight in every background; LM head = untied-from-search affine Linear (bias kept if present), '
                       'no softcap / temperature / hook, final norm = RMSNorm eps 1e-6 (abort otherwise)',
                       f'closure loss = full forward with the target at f (C: every background / format; H: f16 per chunk) within '
                       f'{TOL["closure_vs_full_forward_abs_summed_ce"]}',
                       f'C S_z and D reproduce the previous frozen_predictions within {TOL["reproduce_prev_S_z_D_abs_per_token"]} per token; H '
                       f'per-position CE reproduces the previous eval_ce_pos within {TOL["reproduce_prev_ce_abs_per_position"]}',
                       f'g_z = norm VJP of g_u (autograd, rel {TOL["chain_autograd_vjp_rel"]}) and = the analytic RMSNorm Jacobian in float64 '
                       f'(rel {TOL["chain_analytic_float64_rel"]})', 'f16: every difference, S and KL exactly 0; fp32 not forced to 0; '
                       'position-127 g_z and g_u exactly 0',
                       f'identities per position within {TOL["identity_D64_abs_per_position"]} / {TOL["identity_full_abs_per_position"]}; KL '
                       f'>= {TOL["kl_min_per_position"]}; P0 + Rest = All within {TOL["position_sum_abs_summed"]}',
                       f'rounding flag (reported, not a failure): |eps_head + eps_CE| > {TOL["rounding_flag_abs_summed_ce"]} + '
                       f'{TOL["rounding_flag_rel"]} |D| summed CE', 'TF32 off, deterministic, FP32, eager',
                       'failures are kept and reported, never rerun for a better result'],
          'costs': 'separately: replay init, background set (cold / warm), prefix capture, one baseline for both gradients, nine z, nine '
                   'RMSNorms, dot products (the S_u score path); candidate LM head + CE and float64 KL (diagnostic only); controls; peak '
                   'VRAM / RAM',
          'tolerances': TOL, 'smoke': smoke or 'none',
          'boundaries': 'no change to the formal BC, raw BC, MILP / knee, configs, quantizers or native; no new coefficient, second-order, '
                        'endpoint score, HVP, STE or fine-tuning; no new layer or article',
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L27 一階評分的 RMSNorm／CE 餘項分解', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        '- 範圍：L27 Down、目標權重 A 的 INT8；背景 U／F／W、九種格式、完整 input／output QDQ 與前輪完全相同（同 Env、同 batch 8）。',
        '- 一次 baseline autograd 取 g_z 與 g_u（每窗口 microbatch 1）；S_z = g_z·(z_f − z_0)，S_u = g_u·(u_f − u_0)，R_norm = S_u − S_z。',
        '- 真值 D = FP32 CE 差；同一份 FP32 logits 轉 float64：D_64、S_logit,64、KL_64（p_0 → p_f，不 clip）、eps_head = S_logit,64 − S_u、eps_CE = D − D_64。',
        '- 恆等式（逐位置）：D_64 = S_logit,64 + KL_64；D = S_z + R_norm + KL_64 + eps_head + eps_CE。K2 = ½Var_p0(Δlogit) 只作描述，不是分數。',
        '- C（原 10 窗口）：重現前輪 S_z／D，分解，凍結 S_u 選擇器（argmin S_u,All，a bits ≤ 16，平手保留 f16）。',
        '- H（前輪 128 篇 × 2，已看過）：逐窗口自身背景梯度的 local 分解，只作事後機制分析；凍結選擇器的探索性重播；文章 bootstrap '
        f'{BOOT} 次（seed {SEEDS["bootstrap"]}），只作描述。',
        '- 控制：LM head 為 affine、無 softcap；closure = 完整 forward；前輪重現；g_z = Norm VJP(g_u)；f16 全 0；位置 127 梯度 0；恆等式與位置加總。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log('plan written')


class Env(cs.Env):
    def head_checks(self):
        import inspect
        import transformers
        hf = self.rp.ad.model
        lm, nm = hf.lm_head, hf.model.norm
        src = inspect.getsource(type(hf).forward)
        c = {'lm_head_type': type(lm).__name__, 'lm_head_bias': lm.bias is not None,
             'lm_head_tied_to_embedding': lm.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr(),
             'lm_head_searched_or_quantized': any(v.removesuffix('.weight') == 'lm_head' for v in self.rp.param.values()),
             'hooks_lm_head_norm': [len(lm._forward_pre_hooks), len(lm._forward_hooks), len(nm._forward_pre_hooks), len(nm._forward_hooks)],
             'forward_mentions': [w for w in ('softcap', 'temperature', 'logit_scale', 'final_logit') if w in src],
             'config_final_logit_softcapping': getattr(hf.config, 'final_logit_softcapping', None), 'norm_type': type(nm).__name__,
             'norm_eps': float(nm.variance_epsilon), 'transformers': transformers.__version__}
        c['passed'] = bool(c['lm_head_type'] == 'Linear' and not c['lm_head_searched_or_quantized'] and c['hooks_lm_head_norm'] == [0, 0, 0, 0]
                           and not c['forward_mentions'] and c['config_final_logit_softcapping'] is None and c['norm_type'] == 'Qwen3RMSNorm'
                           and c['norm_eps'] == 1e-6)
        return c

    def grads(self, r, z0, ids):
        """one autograd call per window (microbatch 1, as the previous round): g_z = dL/dz at z_0 and g_u = dL/du at u_0."""
        torch = self.torch
        hf = self.rp.ad.model
        gz, gu = torch.empty_like(z0), torch.empty_like(z0)
        for j in range(len(r)):
            zj = z0[j:j + 1].detach().clone().requires_grad_(True)
            u0 = hf.model.norm(r[j:j + 1] + zj)
            lg = hf.lm_head(u0)
            loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            gz[j:j + 1], gu[j:j + 1] = torch.autograd.grad(loss, (zj, u0))
        return gz, gu

    def chain_check(self, r, z0, gz, gu):
        torch = self.torch
        nm = self.rp.ad.model.model.norm
        w, eps = nm.weight.double(), nm.variance_epsilon
        e_vjp = e_ana = 0.0
        for j in range(len(r)):
            zj = z0[j:j + 1].detach().clone().requires_grad_(True)
            v, = torch.autograd.grad(nm(r[j:j + 1] + zj), zj, grad_outputs=gu[j:j + 1])
            den = float(gz[j].abs().max())
            e_vjp = max(e_vjp, float((v - gz[j:j + 1]).abs().max()) / den)
            with torch.no_grad():
                h = (r[j] + z0[j]).double()
                s2 = h.pow(2).mean(-1, keepdim=True) + eps
                a = gu[j].double() * w
                jt = a / s2.sqrt() - h * (h * a).sum(-1, keepdim=True) / (h.shape[-1] * s2 ** 1.5)
                e_ana = max(e_ana, float((jt - gz[j].double()).abs().max()) / den)
        return e_vjp, e_ana

    def decompose(self, r, z, gz, gu, ids, tm, acc, pre):
        """per-position terms for every format, logits in batches of BATCH windows (the previous closure's GEMM shapes); float64 on
        the same FP32 logits one window at a time. acc[f][term] -> list of [windows, 127] arrays; returns control maxima."""
        torch = self.torch
        F_ = torch.nn.functional
        hf = self.rp.ad.model
        nm, lm = hf.model.norm, hf.lm_head
        eps = nm.variance_epsilon
        sync = torch.cuda.synchronize
        ctl = defaultdict(float)
        z0 = z['f16']
        for i in range(0, len(r), BATCH):
            sl = slice(i, i + BATCH)
            y = ids[sl, 1:]
            nb = y.shape[0]
            with torch.no_grad():
                t0 = time.perf_counter()
                h0 = r[sl] + z0[sl]
                u0 = nm(h0)
                l0 = lm(u0)
                V = l0.shape[-1]
                ce0 = F_.cross_entropy(l0[:, :-1].reshape(-1, V).float(), y.reshape(-1), reduction='none').view(nb, -1).double()
                lp0 = [torch.log_softmax(l0[j, :-1].double(), -1) for j in range(nb)]
                sig0 = h0.double().pow(2).mean(-1).add(eps).sqrt()
                sync()
                tm[f'{pre}baseline_head_and_float64_s'] += time.perf_counter() - t0
                for f in FORMATS:
                    t0 = time.perf_counter()
                    hf_ = r[sl] + z[f][sl]
                    uf = nm(hf_)
                    sync()
                    tm[f'{pre}candidate_norm_s'] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    dz = z[f][sl] - z0[sl]
                    Sz = (gz[sl].double() * dz.double()).sum(-1)
                    du = uf.double() - u0.double()
                    Su = (gu[sl].double() * du).sum(-1)
                    sync()
                    tm[f'{pre}dot_products_s'] += time.perf_counter() - t0
                    ctl['pos127_S_abs'] = max(ctl['pos127_S_abs'], float(Sz[:, -1].abs().max()), float(Su[:, -1].abs().max()))
                    t0 = time.perf_counter()
                    lf = lm(uf)
                    cef = F_.cross_entropy(lf[:, :-1].reshape(-1, V).float(), y.reshape(-1), reduction='none').view(nb, -1).double()
                    sync()
                    tm[f'{pre}candidate_head_ce_s'] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    rows = {k: torch.empty(nb, TP, dtype=torch.float64, device=r.device) for k in ('D64', 'S_logit64', 'KL', 'K2')}
                    for j in range(nb):
                        lf64 = lf[j, :-1].double()
                        dl = lf64 - l0[j, :-1].double()
                        lpf = torch.log_softmax(lf64, -1)
                        p0 = lp0[j].exp()
                        yj = y[j][:, None]
                        rows['D64'][j] = (-lpf.gather(-1, yj) + lp0[j].gather(-1, yj)).squeeze(-1)
                        m1 = (p0 * dl).sum(-1)
                        rows['S_logit64'][j] = m1 - dl.gather(-1, yj).squeeze(-1)
                        rows['KL'][j] = (p0 * (lp0[j] - lpf)).sum(-1)
                        rows['K2'][j] = 0.5 * ((p0 * dl * dl).sum(-1) - m1 * m1)
                        del lf64, dl, lpf, p0
                    sync()
                    tm[f'{pre}kl64_diagnostic_s'] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    D = cef - ce0
                    Sz_, Su_ = Sz[:, :-1], Su[:, :-1]
                    terms = {'D': D, 'S_z': Sz_, 'S_u': Su_, 'R_norm': Su_ - Sz_, 'KL': rows['KL'], 'eps_head': rows['S_logit64'] - Su_,
                             'eps_CE': D - rows['D64'], 'D64': rows['D64'], 'S_logit64': rows['S_logit64'], 'K2': rows['K2'],
                             'rms_ratio': (hf_.double().pow(2).mean(-1).add(eps).sqrt() / sig0 - 1)[:, :-1],
                             'dz_rel': (dz.double().norm(dim=-1) / z0[sl].double().norm(dim=-1))[:, :-1],
                             'du_rel': (du.norm(dim=-1) / u0.double().norm(dim=-1))[:, :-1]}
                    ctl['identity_D64_abs'] = max(ctl['identity_D64_abs'], float((rows['D64'] - rows['S_logit64'] - rows['KL']).abs().max()))
                    rec = terms['S_z'] + terms['R_norm'] + terms['KL'] + terms['eps_head'] + terms['eps_CE']
                    ctl['identity_full_abs'] = max(ctl['identity_full_abs'], float((D - rec).abs().max()))
                    ctl['kl_min'] = min(ctl.get('kl_min', 0.0), float(rows['KL'].min()))
                    if f == 'f16':
                        ctl['f16_abs'] = max(ctl['f16_abs'], max(float(terms[k].abs().max()) for k in TERMS + ('rms_ratio', 'dz_rel', 'du_rel')))
                    terms['CE'] = cef
                    for k, v in terms.items():
                        acc[f][k].append(v.cpu().numpy())
                    tm['bookkeeping_s'] += time.perf_counter() - t0
                    del uf, lf, cef, rows, terms, du, dz
                del l0, lp0, u0, h0
        return ctl


def per_pos(acc):
    return {f: {k: np.concatenate(v) for k, v in d.items()} for f, d in acc.items()}


def gsum(a, G):
    """[windows, 127] per-position values -> per-window sums over G."""
    return {'P0': a[:, 0], 'Rest': a[:, 1:].sum(1), 'All': a.sum(1)}[G]


def run_bg(env, toks, b, tm, acc, ctl, chunk):
    torch = env.torch
    for c0 in range(0, len(toks), chunk):
        t = toks[c0:c0 + chunk]
        ids = torch.from_numpy(t).cuda()
        t0 = time.perf_counter()
        full0, x, r = env.full(t, 'f16', capture=True)
        torch.cuda.synchronize()
        tm[f'{b}|prefix_capture_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        z = {f: env.zf(x, f) for f in FORMATS}
        torch.cuda.synchronize()
        tm[f'{b}|candidate_z_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        gz, gu = env.grads(r, z['f16'], ids)
        torch.cuda.synchronize()
        tm[f'{b}|baseline_two_gradients_s'] += time.perf_counter() - t0
        tm['baseline_windows'] += len(t)
        t0 = time.perf_counter()
        e_vjp, e_ana = env.chain_check(r, z['f16'], gz, gu)
        ctl['chain_vjp_rel'] = max(ctl['chain_vjp_rel'], e_vjp)
        ctl['chain_analytic_rel'] = max(ctl['chain_analytic_rel'], e_ana)
        ctl['pos127_g_abs'] = max(ctl['pos127_g_abs'], float(gz[:, -1].abs().max()), float(gu[:, -1].abs().max()))
        tm['chain_check_s'] += time.perf_counter() - t0
        c = env.decompose(r, z, gz, gu, ids, tm, acc, f'{b}|')
        for k, v in c.items():
            ctl[k] = min(ctl.get(k, 0.0), v) if k == 'kl_min' else max(ctl[k], v)
        yield t, full0, x, r, z


def calib(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    t0 = time.perf_counter()
    env = Env(out / 'work')
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    ctl = defaultdict(float)
    ctl['head'] = env.head_checks()
    if not ctl['head']['passed']:
        write(out / 'controls_calib.json', ctl)
        raise SystemExit(f'LM head / final norm is not the assumed affine + RMSNorm: {ctl["head"]}')
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    prevp = read(PREV / 'frozen_predictions.json')
    prevS = {(r_['background'], r_['format'], r_['G']): (r_['S'], r_['calibration_D']) for r_ in prevp['scores']}
    res, ctl['target_weight_int8'], ctl['closure_vs_full_abs'] = {}, {}, 0.0
    for b in BGS:
        t0 = time.perf_counter()
        cfg = cs.background(A, b)
        env.set_background(cfg)
        torch.cuda.synchronize()
        tm[f'{b}|set_background_cold_s'] = time.perf_counter() - t0
        ctl['target_weight_int8'][b] = bool(cfg[TARGET]['w'] == 'int8' and torch.equal(env.Wq, env.rp.cache[(TARGET, 'int8')].to(env.Wq.device)))
        acc = {f: defaultdict(list) for f in FORMATS}
        for t, full0, x, r, z in run_bg(env, cal, b, tm, acc, ctl, len(cal)):
            t0 = time.perf_counter()
            for f in FORMATS:                                        # closure vs full forward, every format (as the previous round)
                cf = env.closure(r, z[f], ids)
                ff = env.full(cal, f)
                ctl['closure_vs_full_abs'] = max(ctl['closure_vs_full_abs'], float((cf.sum(1) - ff.sum(1)).abs().max()))
            tm['closure_control_s'] += time.perf_counter() - t0
        res[b] = per_pos(acc)
    t0 = time.perf_counter()
    env.set_background(cs.background(A, 'U'))
    torch.cuda.synchronize()
    tm['U|set_background_warm_s'] = time.perf_counter() - t0
    # reproduction of the previous round's frozen S_z and calibration D; position sums
    rep, psum = 0.0, 0.0
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                s_, d_ = prevS[(b, f, G)]
                rep = max(rep, abs(gsum(res[b][f]['S_z'], G).sum() / (CAL * TP) - s_), abs(gsum(res[b][f]['D'], G).sum() / (CAL * TP) - d_))
            for k in TERMS:
                a = res[b][f][k]
                psum = max(psum, abs(a[:, 0].sum() + a[:, 1:].sum() - a.sum()))
    ctl['reproduce_prev_S_z_D_abs_per_token'] = rep
    ctl['position_sum_abs'] = psum
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    ctl['rounding_flags'] = rounding_flags(res, CAL * TP)
    ctl['passed'] = passed(ctl) and rep <= TOL['reproduce_prev_S_z_D_abs_per_token'] and ctl['closure_vs_full_abs'] <= TOL['closure_vs_full_forward_abs_summed_ce']
    np.savez(out / 'C_decomp.npz', **{f'{b}__{f}__{k}': v for b in BGS for f in FORMATS for k, v in res[b][f].items()})
    # scores, rankings and the frozen S_u selector (C only)
    sc, sel = [], {}
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                row = {'background': b, 'format': f, 'G': G}
                row.update({k: float(gsum(res[b][f][k], G).sum()) / (CAL * TP) for k in TERMS})
                sc.append(row)
        S_u = {f: float(res[b][f]['S_u'].sum()) for f in SELECT_SET}
        S_z = {f: float(res[b][f]['S_z'].sum()) for f in SELECT_SET}
        sel[b] = {}
        for nm_, S in (('S_u', S_u), ('S_z', S_z)):
            order = sorted(SELECT_SET, key=lambda f: (round(S[f], 12), SELECT_SET.index(f)))
            sel[b][nm_] = {'pick': order[0], 'order': order, 'S_pick': S[order[0]] / (CAL * TP),
                           'margin_to_next': (S[order[1]] - S[order[0]]) / (CAL * TP)}
    ctl['S_z_selector_reproduces_previous'] = all(sel[b]['S_z']['pick'] == prevp['selector'][b]['pick'] for b in BGS)
    write(out / 'controls_calib.json', ctl)
    write(out / 'frozen_predictions.json', {'scores_per_token': sc, 'selector': sel, 'bc_selector': prevp['bc_selector'],
                                            'note': 'C only; S_u selector frozen here, before any H computation of this round'})
    with open(out / 'scores_C.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sc[0]))
        wr.writeheader()
        wr.writerows(sc)
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib', tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; S_u picks {({b: sel[b]["S_u"]["pick"] for b in BGS})}; S_z picks '
        f'{({b: sel[b]["S_z"]["pick"] for b in BGS})}; rep {rep:.2e}; identity {ctl["identity_full_abs"]:.2e}; kl_min {ctl["kl_min"]:.2e}')
    if not ctl['passed']:
        raise SystemExit(f'calibration controls failed: {dict(ctl)}')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_calib.json'),
                                'predictions_sha256': sha(out / 'frozen_predictions.json'), 'scores_sha256': sha(out / 'scores_C.csv')})


def passed(ctl):
    return bool(ctl['head']['passed'] if isinstance(ctl.get('head'), dict) else True) and bool(
        ctl['chain_vjp_rel'] <= TOL['chain_autograd_vjp_rel'] and ctl['chain_analytic_rel'] <= TOL['chain_analytic_float64_rel']
        and ctl['pos127_g_abs'] == 0 and ctl['pos127_S_abs'] == 0 and ctl['f16_abs'] == 0
        and ctl['identity_D64_abs'] <= TOL['identity_D64_abs_per_position'] and ctl['identity_full_abs'] <= TOL['identity_full_abs_per_position']
        and ctl['kl_min'] >= TOL['kl_min_per_position'] and ctl.get('position_sum_abs', 0.0) <= TOL['position_sum_abs_summed']
        and all(ctl['target_weight_int8'].values()) and ctl['tf32'] == [False, False] and ctl['deterministic'])


def rounding_flags(res, n):
    fl = {}
    for b in res:
        for f in FORMATS:
            for G in POS:
                e = float(gsum(res[b][f]['eps_head'] + res[b][f]['eps_CE'], G).sum())
                d = float(gsum(res[b][f]['D'], G).sum())
                if abs(e) > TOL['rounding_flag_abs_summed_ce'] + TOL['rounding_flag_rel'] * abs(d):
                    fl[f'{b}|{f}|{G}'] = {'eps_sum_per_token': e / n, 'D_per_token': d / n}
    return fl


def hdecomp(out):
    t_all = time.perf_counter()
    check_plan(out)
    fz = read(out / 'freeze.json')
    if sha(out / 'frozen_predictions.json') != fz['predictions_sha256'] or sha(out / 'controls_calib.json') != fz['controls_sha256']:
        raise SystemExit('freeze changed')
    dfz = read(PREV / 'data_freeze.json')
    if sha(PREV / 'tokens.npy') != dfz['tokens_sha256']:
        raise SystemExit('H tokens changed')
    tm = defaultdict(float)
    t0 = time.perf_counter()
    env = Env(out / 'work')
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    A = read(ENDP / 'configs/A_auto.json')
    tok = np.load(PREV / 'tokens.npy')
    prev = np.load(PREV / 'eval_ce_pos.npz')
    ctl = defaultdict(float)
    ctl['head'] = env.head_checks()
    ctl['target_weight_int8'], ctl['f16_closure_vs_full_abs'], ctl['reproduce_prev_ce_abs'] = {}, 0.0, 0.0
    res = {}
    for b in BGS:
        t0 = time.perf_counter()
        cfg = cs.background(A, b)
        env.set_background(cfg)
        torch.cuda.synchronize()
        tm[f'{b}|set_background_cold_s'] = time.perf_counter() - t0
        ctl['target_weight_int8'][b] = bool(cfg[TARGET]['w'] == 'int8' and torch.equal(env.Wq, env.rp.cache[(TARGET, 'int8')].to(env.Wq.device)))
        acc = {f: defaultdict(list) for f in FORMATS}
        for t, full0, x, r, z in run_bg(env, tok, b, tm, acc, ctl, CHUNK):
            t0 = time.perf_counter()
            c16 = env.closure(r, z['f16'], torch.from_numpy(t).cuda())
            ctl['f16_closure_vs_full_abs'] = max(ctl['f16_closure_vs_full_abs'], float((c16.sum(1) - full0.sum(1)).abs().max()))
            tm['closure_control_s'] += time.perf_counter() - t0
            del x, r, z
        res[b] = per_pos(acc)
        for f in FORMATS:                                           # per-position CE of every format vs the previous eval
            ctl['reproduce_prev_ce_abs'] = max(ctl['reproduce_prev_ce_abs'], float(np.abs(res[b][f]['CE'] - prev[f'{b}__{f}']).max()))
        log(f'{b} done')
    psum = 0.0
    for b in BGS:
        for f in FORMATS:
            for k in TERMS:
                a = res[b][f][k]
                psum = max(psum, abs(a[:, 0].sum() + a[:, 1:].sum() - a.sum()))
    ctl['position_sum_abs'] = psum
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    ctl['rounding_flags'] = rounding_flags(res, len(tok) * TP)
    ctl['passed'] = passed(ctl) and ctl['f16_closure_vs_full_abs'] <= TOL['closure_vs_full_forward_abs_summed_ce'] and \
        ctl['reproduce_prev_ce_abs'] <= TOL['reproduce_prev_ce_abs_per_position']
    write(out / 'controls_H.json', ctl)
    np.savez(out / 'H_decomp.npz', **{f'{b}__{f}__{k}': v for b in BGS for f in FORMATS for k, v in res[b][f].items()})
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'hdecomp', tm['total_s'], dict(tm))
    log(f'H controls passed={ctl["passed"]}; rep {ctl["reproduce_prev_ce_abs"]:.2e}; identity {ctl["identity_full_abs"]:.2e}; '
        f'kl_min {ctl["kl_min"]:.2e}; flags {len(ctl["rounding_flags"])}')
    if not ctl['passed']:
        raise SystemExit(f'H controls failed (kept): { {k: v for k, v in ctl.items() if k != "rounding_flags"} }')


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    C = np.load(out / 'C_decomp.npz')
    H = np.load(out / 'H_decomp.npz')
    fp = read(out / 'frozen_predictions.json')
    na = N_ART
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]) / na
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def art(a, G):                                    # [256, 127] -> per-article nat/token contribution [128]
        c = a.reshape(na, WPA, TP)
        return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[G] / (WPA * TP)

    def st(v):
        bs = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'median': float(np.median(v)),
                'n_pos': int((v > 0).sum()), 'n_neg': int((v < 0).sum()), 'min': float(v.min()), 'max': float(v.max())}

    def decisive(s):
        return s['ci95'][0] > 0 or s['ci95'][1] < 0
    Cv = lambda b, f, k, G: float(gsum(C[f'{b}__{f}__{k}'], G).sum()) / (CAL * TP)
    Ha = lambda b, f, k, G: art(H[f'{b}__{f}__{k}'], G)
    # decomposition tables
    rowsC, rowsH = [], []
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                rc = {'background': b, 'format': f, 'G': G}
                rc.update({k: Cv(b, f, k, G) for k in TERMS})
                rc['D_minus_S_z'] = rc['D'] - rc['S_z']
                rc['D_minus_S_u'] = rc['D'] - rc['S_u']
                rc['sign_S_z_ok'] = sgn(rc['S_z']) == sgn(rc['D'])
                rc['sign_S_u_ok'] = sgn(rc['S_u']) == sgn(rc['D'])
                rc['cancellation'] = (abs(rc['R_norm']) + abs(rc['KL'])) / max(abs(rc['R_norm'] + rc['KL']), 1e-300)
                for k in ('rms_ratio', 'dz_rel', 'du_rel'):
                    a = C[f'{b}__{f}__{k}']
                    rc[f'{k}_mean'] = float({'P0': a[:, 0], 'Rest': a[:, 1:], 'All': a}[G].mean())
                rowsC.append(rc)
                rh = {'background': b, 'format': f, 'G': G}
                for k in TERMS:
                    s = st(Ha(b, f, k, G))
                    rh.update({f'{k}_mean': s['mean'], f'{k}_lo': s['ci95'][0], f'{k}_hi': s['ci95'][1], f'{k}_median': s['median'],
                               f'{k}_npos': s['n_pos'], f'{k}_nneg': s['n_neg'], f'{k}_min': s['min'], f'{k}_max': s['max']})
                rn, kl = Ha(b, f, 'R_norm', G), Ha(b, f, 'KL', G)
                rh['cancellation'] = float((np.abs(rn) + np.abs(kl)).mean() / max(abs((rn + kl).mean()), 1e-300))
                rh['articles_R_norm_negative'] = int((rn < 0).sum())
                d, sz, su = Ha(b, f, 'D', G), Ha(b, f, 'S_z', G), Ha(b, f, 'S_u', G)
                rh['local_sign_S_z_ok_articles'] = int((np.sign(sz) == np.sign(d)).sum())
                rh['local_sign_S_u_ok_articles'] = int((np.sign(su) == np.sign(d)).sum())
                for k in ('rms_ratio', 'dz_rel', 'du_rel'):
                    a = H[f'{b}__{f}__{k}']
                    rh[f'{k}_mean'] = float({'P0': a[:, 0], 'Rest': a[:, 1:], 'All': a}[G].mean())
                rowsH.append(rh)
    # rankings: directions vs f16 and pair orders over the selection set
    nonf16 = [f for f in FORMATS if f != 'f16']
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    rank = {}
    for b in BGS:
        rb = {}
        for sc in ('S_z', 'S_u'):
            dirC = {f: sgn(Cv(b, f, sc, 'All')) == sgn(Cv(b, f, 'D', 'All')) for f in nonf16}
            pairC = sum(sgn(Cv(b, a, sc, 'All') - Cv(b, c, sc, 'All')) == sgn(Cv(b, a, 'D', 'All') - Cv(b, c, 'D', 'All')) for a, c in pairs)
            dirH, pairH = {}, {'right': 0, 'wrong': 0, 'undetermined': 0}
            for f in nonf16:
                s = st(Ha(b, f, 'D', 'All'))
                dirH[f] = 'undetermined' if not decisive(s) else ('right' if sgn(Cv(b, f, sc, 'All')) == sgn(s['mean']) else 'wrong')
            for a, c in pairs:
                s = st(Ha(b, a, 'D', 'All') - Ha(b, c, 'D', 'All'))
                pairH['undetermined' if not decisive(s) else ('right' if sgn(Cv(b, a, sc, 'All') - Cv(b, c, sc, 'All')) == sgn(s['mean']) else 'wrong')] += 1
            rb[sc] = {'C_direction_right': dirC, 'C_direction_right_n': sum(dirC.values()), 'C_pairs_right_of_28': int(pairC),
                      'H_frozen_direction': dirH, 'H_frozen_pairs': pairH,
                      'C_order': sorted(SELECT_SET, key=lambda f: Cv(b, f, sc, 'All'))}
        rb['C_truth_order'] = sorted(SELECT_SET, key=lambda f: Cv(b, f, 'D', 'All'))
        rb['H_truth_order'] = sorted(SELECT_SET, key=lambda f: float(Ha(b, f, 'D', 'All').mean()))
        rank[b] = rb
    same_bit = []
    for b in BGS:
        for i_, p_ in (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1')):
            s = st(Ha(b, i_, 'D', 'All') - Ha(b, p_, 'D', 'All'))
            row = {'background': b, 'pair': f'{i_}-{p_}', 'C_truth': Cv(b, i_, 'D', 'All') - Cv(b, p_, 'D', 'All'), 'H_truth': s}
            for sc in ('S_z', 'S_u'):
                pr = Cv(b, i_, sc, 'All') - Cv(b, p_, sc, 'All')
                row[sc] = pr
                row[f'{sc}_C'] = 'right' if sgn(pr) == sgn(row['C_truth']) else 'wrong'
                row[f'{sc}_H'] = 'undetermined' if not decisive(s) else ('right' if sgn(pr) == sgn(s['mean']) else 'wrong')
            same_bit.append(row)
    # selector replay (exploratory, H already seen)
    zero = np.zeros(na)
    Dsel = lambda b, f: zero if f == 'f16' else Ha(b, f, 'D', 'All')
    sel = {}
    for b in BGS:
        s_u, s_z = fp['selector'][b]['S_u']['pick'], fp['selector'][b]['S_z']['pick']
        su, sz = st(Dsel(b, s_u)), st(Dsel(b, s_z))
        sel[b] = {'S_u_pick': s_u, 'S_u_margin': fp['selector'][b]['S_u']['margin_to_next'], 'S_u_pick_H_D': su,
                  'S_u_pick_severe': bool(su['mean'] > SEVERE and su['ci95'][0] > 0),
                  'S_z_pick': s_z, 'S_z_pick_H_D': sz, 'S_z_pick_severe': bool(sz['mean'] > SEVERE and sz['ci95'][0] > 0),
                  'S_u_minus_S_z_pick_H': st(Dsel(b, s_u) - Dsel(b, s_z)),
                  'BC_pick': fp['bc_selector']['pick'], 'S_u_minus_BC_pick_H': st(Dsel(b, s_u) - Dsel(b, fp['bc_selector']['pick'])),
                  'oracle_H_in_set': min(SELECT_SET, key=lambda f: float(Dsel(b, f).mean())),
                  'C_truth_best_in_set': min(SELECT_SET, key=lambda f: Cv(b, f, 'D', 'All') if f != 'f16' else 0.0)}
    pf, pu = fp['selector']['F']['S_u']['pick'], fp['selector']['U']['S_u']['pick']
    sel['F_truth_pickF_minus_pickU'] = {'pick_F': pf, 'pick_U': pu, 'H': st(Dsel('F', pf) - Dsel('F', pu)),
                                        'C': (Cv('F', pf, 'D', 'All') if pf != 'f16' else 0.0) - (Cv('F', pu, 'D', 'All') if pu != 'f16' else 0.0)}
    res = {'units': 'nat/token contributions relative to f16 of the same background; C = 10 windows (1270 targets), H = 128 articles '
                    '(local per-window gradients, already seen; post hoc)', 'C': rowsC, 'rankings': rank, 'same_bit': same_bit, 'selector': sel,
           'H_summary_file': 'decomposition_H.csv'}
    write(out / 'stats.json', res)
    for name, rows in (('decomposition_C.csv', rowsC), ('decomposition_H.csv', rowsH)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log('done')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'hdecomp', 'analyze'))
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
        {'calib': calib, 'hdecomp': hdecomp, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_norm_ce_decomposition.py')}) + '\n')


if __name__ == '__main__':
    main()
