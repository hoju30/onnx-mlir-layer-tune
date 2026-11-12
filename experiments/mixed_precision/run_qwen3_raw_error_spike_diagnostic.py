#!/usr/bin/env python3
"""Qwen3-0.6B L2 Down input: where does the full-QDQ damage of each INT / Posit format sit when the raw quantization error
e = Q(x) - x (never centred) is split by the frozen spike mask? Mechanism diagnostic for why D (V_N(mu_N) + U_N(mu_N) +
sum_S g^2 e^2) fixed L2 6 / 8-bit but badly under-rated L2 Posit4. Spec: reports/qwen3_raw_error_spike_diagnostic_prompt_20261026.md.
No change to the formal criterion, D, quantizers, search or native pipeline.

Only the L2 Down input x is perturbed (weights, other activations, Down output FP32). Mask: a_sp = max_c |x_spc|,
S = {a >= frozen threshold of runs/qwen3_spike_recenter_20261026 (6 x calibration median)}, N = the rest.
Directions e_S = M_S e, e_N = M_N e, e_all = e (fixed, detached; q = Q(x) computed once on the FP32 x).
  alpha = 1 (main): S-only where(S, q, x), N-only where(N, q, x), all q;  other alpha: x + alpha d
  Delta(alpha) = L(x') - L(x), L = summed next-token CE through the real Down, residual, layers 3..27 (attention, FFN), final
  norm, LM head; I_SN = Delta_all - Delta_S - Delta_N; Posit - INT pair differences D_all = D_S + D_N + D_int.
A: original 10 x 128 calibration, alpha in {0, +-2^-k, k = 0..10}, F_G = g . e_G (GPU baseline gradient), O / E responses, D
   components reproduced from the CPU cache. B: the 128 articles x 3 windows of runs/qwen3_spike_recenter_20261026 (already
   seen; post-hoc mechanism extension, not an independent confirmation), alpha = 1 partitions only, the evaluator path of
   that round (all must reproduce its L2 losses).

Stages (one new --out-dir): plan, calib, replay, analyze.
CPU: ../../venv/bin/python; calib / replay: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (read / write / sha / now / rel; standard imports only)

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE, PREP = prs.NATIVE, prs.PREP
REC = HERE / 'runs/qwen3_spike_recenter_20261026'
VDIAG = HERE / 'runs/qwen3_posit_variance_diagnostic_20261025'
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
LAYER = 2
FORMATS = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')            # the order of the recenter round (eval_ce formats[1:])
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
PARTS = ('S', 'N', 'all')
ALPHAS = (0.0,) + tuple(s * 2.0 ** -k for k in range(11) for s in (1.0, -1.0))
SEQ, TP, CAL, WPA = 128, 127, 10, 3
TARGETS = CAL * TP
CHUNK, LM_SUB = 32, 16
BOOT, BOOT_SEED = 10000, 26102611
STRIDE = 97                                                            # fixed subsample stride for B quantiles
TOL = {'replay_vs_nohook_abs_summed_ce': 0.0, 'hooked_full_vs_replay_abs_summed_ce': 1e-6, 'B_all_vs_recenter_abs_summed_ce': 1e-6,
       'B_identity_vs_recenter_abs_summed_ce': 1e-6, 'mask_equals_recenter': 'bitwise', 'D_components_rel': 1e-12,
       'gpu_qdq_of_cpu_h_vs_recenter_sha': 'bitwise', 'gx_vs_gyW_rel': 1e-4, 'pos127_abs': 0.0, 'alpha0_abs': 0.0, 'zero_direction_abs': 0.0,
       'F_additivity_rel': 1e-9, 'pair_identity_rel': 1e-12,
       'reported_only': 'GPU x vs CPU cache h, GPU g vs CPU g, x+e_G vs masked QDQ at alpha = 1 (input and loss), calibration all vs the '
                        'variance-diagnostic Q(x) loss (different LM-head sub-batch), rounding / changed fraction of small alpha'}
CODE = ['run_qwen3_raw_error_spike_diagnostic.py', 'run_qwen3_spike_recenter.py', 'run_qwen_down_spike_diag.py', 'run_qwen_prefix_rank_shift.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'run_olmoe_dynamic_route.py']
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


def inputs():
    return ([HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors')] +
            [NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', PREP / 'node_mapping.json', PREP / 'validation_tokens.npy',
             REC / 'report.md', REC / 'components.csv', REC / 'scores.csv', REC / 'masks.npz', REC / 'freeze.json', REC / 'per_article_losses.csv',
             REC / 'segments.json', REC / 'tokens.npy', REC / 'eval_ce.npz', REC / 'controls_scores.json', REC / 'controls_eval.json',
             VDIAG / 'report.md', VDIAG / 'losses.npz', SPIKE / 'work/h_L2.npy', SPIKE / 'work/g_L2.npy',
             HERE / 'reports/qwen3_raw_error_spike_diagnostic_prompt_20261026.md'])


def check_plan(out):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    for f, h in pl['inputs'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'input changed: {f}')
    return pl


def threshold():
    rows = [r for r in csv.DictReader(open(REC / 'components.csv')) if r['layer'] == str(LAYER)]
    t = {float(r['threshold']) for r in rows}
    if len(t) != 1:
        raise SystemExit('threshold not unique')
    return t.pop()


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'spec': 'reports/qwen3_raw_error_spike_diagnostic_prompt_20261026.md and the request of 2026-10-26',
        'question': 'with the raw error e = Q(x) - x split by the frozen spike mask, where does the extra ~3.54 nat/token of L2 Posit4 over INT4 '
                    'sit (spike only, ordinary tokens only, their interaction), why not for Posit6 / 8, and which response does the spike '
                    'proxy of D miss? spike, ordinary tokens, interaction or undetermined are all allowed outcomes',
        'scope': f'Qwen3-0.6B L{LAYER} Down input only; formats {list(FORMATS)} + FP32 identity; one change at a time; weights, other activations '
                 'and Down output FP32; original calibration, INT range / scale / zp, Universal posit LUT (no external scaling), RMSNorm eps, loss',
        'mask': {'threshold': threshold(), 'source': 'runs/qwen3_spike_recenter_20261026 components.csv (6 x calibration median of a = max_c |x|)',
                 'rule': 'S = {a_sp >= threshold}, N = complement; calibration mask equals recenter masks.npz bitwise; B windows use the same '
                         'threshold on their own FP32 x (no new median); no hard-coded position 0; empty S -> e_S = 0'},
        'directions': 'q = Q_f(x) once on the FP32 x; e = q - x, e_S = M_S e, e_N = M_N e, e_all = e; fixed and detached for every alpha; no '
                      'mean subtraction, no re-centring, no re-quantization of x + alpha d, no spike re-detection',
        'interventions': {'alpha_1_main': 'S-only where(S, q, x), N-only where(N, q, x), all q', 'other_alpha': 'x + alpha d',
                          'alpha_0': 'x', 'calibration_alphas': list(ALPHAS), 'B': 'baseline, S-only, N-only, all at alpha = 1 for six formats',
                          'reconstruction': 'alpha = 1 x + e_G vs the masked QDQ: input and loss differences reported',
                          'closure': 'real Down, residual, layers 3..27 (attention and FFN), final norm, LM head, next-token CE; the evaluator '
                                     f'path of the recenter round (LM-head sub-batch {LM_SUB}; B in chunks of {CHUNK} windows)'},
        'decomposition': {'I_SN': 'Delta_all - Delta_S - Delta_N (signed; attention follows, no additivity required; not a Hessian term)',
                          'pairs': 'Posit - INT (positive = Posit worse): D_all = D_S + D_N + D_int, D_int = I_SN,P - I_SN,I; every alpha and '
                                   'alpha = 1 (main); a difference decomposition under these interventions, not a unique causal share'},
        'proxy': {'D': 'C_N + P_S, C_N = V_N(mu_N) + U_N(mu_N), P_S = sum_(S,c) g^2 e^2, reproduced from the CPU cache h / g of the original '
                       'calibration (runs/qwen3_down_spike_diag_20261023/work)', 'display': 'raw; /2540 (= 1/2 x /1270) only as the curvature-'
                       'comparison convention, not a Hessian', 'note': 'C_N is a score, not the measured D_N'},
        'first_order': {'g': 'one FP32 baseline backward through the same closure on the GPU (microbatch 1); g_x = g_y W_down and position 127 '
                             'checked; compared with the CPU cache g (reported)', 'F_G': 'g . e_G per window', 'O': '[Delta(+a) - Delta(-a)] / 2',
                        'E': '[Delta(+a) + Delta(-a)] / 2', 'report': 'O/a vs F, E/a^2 across scale vs P_S/2 and C_N/2; no HVP, Hessian or extra FD; '
                        'E / O contain higher orders; unconverged = undetermined', 'perturbation': 'alpha ||d||, max |alpha d|, changed fraction '
                        'over the support of d, rounding residual per direction and alpha'},
        'statistics': 'per format and part: |x| and |e| quantiles / max, sum e^2, fraction of exactly zero outputs, Posit elements beyond the '
                      'LUT maxpos (from the actual value set), INT zero fraction measured on calibration and on B (B quantiles from a fixed '
                      f'stride-{STRIDE} subsample); explanation only',
        'B_statistics': f'per article (3 windows, 381 targets): S / N / all / I_SN and the pair decomposition; mean, median, share of articles '
                        f'with Posit worse, mask coverage; article bootstrap {BOOT} (seed {BOOT_SEED}) 95% CI labelled post-hoc descriptive; '
                        'windows are not articles; no bootstrap of the 10 calibration windows',
        'controls': ['hashes of model / range / tokens / caches; TF32 off, deterministic, FP32, eager, RMSNorm eps 1e-6',
                     'posit LUT vs Universal; INT scale / zp on the device; GPU QDQ of the CPU cache h equals the recenter error sha',
                     f'baseline replay equals the no-hook forward (abs <= {TOL["replay_vs_nohook_abs_summed_ce"]}); hooked full forward with the '
                     f'L2 Down input replaced equals the replay (<= {TOL["hooked_full_vs_replay_abs_summed_ce"]}) for S-only / N-only / all',
                     'only the L2 Down input replaced (W hash, y = linear(x, W)), no output QDQ',
                     f'D components / total and the calibration mask reproduce recenter (rel <= {TOL["D_components_rel"]}, bitwise)',
                     f'B: identity and all reproduce the recenter eval_ce of L2 (<= {TOL["B_all_vs_recenter_abs_summed_ce"]})',
                     'e_S + e_N = e bitwise; alpha = 0 and a zero direction give 0; F_all = F_S + F_N; the pair identity',
                     f'g_x = g_y W within {TOL["gx_vs_gyW_rel"]}; position-127 gradient 0',
                     'tolerances frozen before any partition loss; fixes recorded as amendments; failures and non-finite values kept'],
        'reading': ['report where the full-amplitude damage concentrates under the interventions; where the local proxy and the finite '
                    'response disagree; that a unique root cause or general fix is not proven', 'B is post-hoc (its full QDQ losses were seen)',
                    'then propose one next hypothesis and stop'],
        'tolerances': TOL, 'smoke': smoke or 'none',
        'boundaries': 'no change to the formal BC or D, no new format / threshold / coefficient, no other layer or new articles, no '
                      'configuration search, posit scaling, fine-tuning or native integration',
        'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：L2 Posit4 的原始量化誤差分區與有限幅度診斷', '', f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。', '',
          f'- 範圍：Qwen3-0.6B 只做 L{LAYER} Down input；{list(FORMATS)} + identity；權重、其他 activation、Down output FP32；原 calibration、range、LUT、RMSNorm eps、loss。',
          f'- 遮罩：a = max_c|x|，門檻 {pl["mask"]["threshold"]}（recenter 輪凍結值）；S = a ≥ 門檻；calibration 須與 masks.npz 逐位元相同；B 用同一門檻在自己的 x 上建 S。',
          '- 方向：e = Q(x) − x（原始、不中心化），e_S、e_N、e_all 固定；α = 1 主評估用 masked QDQ，其他 α 用 x + αd；α = 1 另報 x + e_G 重建差異。',
          f'- A（calibration）：α ∈ {{0, ±2^-k, k = 0…10}}；I_SN、Posit − INT 恆等分解 D_all = D_S + D_N + D_int；F_G = g·e_G（GPU baseline 梯度）、O/α、E/α²；D 分項（C_N、P_S）由 CPU 快取重現。',
          '- B（recenter 輪 128 篇 × 3 窗口，已看過，事後延伸、非獨立確認）：α = 1 的 S-only、N-only、all；all 須重現上一輪 L2 loss；每文章平均、中位數、符號比例、事後描述性 bootstrap CI。',
          '- closure：真實 Down、residual、L3–L27 全部 attention／FFN、final norm、LM head、CE（recenter 評估器路徑）。',
          '- 控制：hash、TF32 關閉、deterministic、eager、eps 1e-6；LUT、INT 裝置；replay = 無 hook 前向；hook 完整前向 = 重播；只改 L2 Down input；D 分項與遮罩重現；B 的 identity／all 重現；e_S + e_N = e；α = 0；零方向；F 可加；配對恆等；g_x = g_y W；position 127。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log('plan written')


# --------------------------------------------------------------------------- #
# GPU helpers (the evaluator path of run_qwen3_spike_recenter.eval_)
# --------------------------------------------------------------------------- #
class Env:
    def __init__(self, work):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from run_qwen_down_spike_diag import Model, Quant, summed_ce_rows
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        self.Q = Quant(work, 'cuda')
        self.M = Model('cuda')
        self.ce_rows = summed_ce_rows
        node = next(n for n, p in read(PREP / 'node_mapping.json').items() if p == f'model.layers.{LAYER}.mlp.down_proj.weight')
        self.rng = read(NATIVE / 'ranges_by_point.json')[node]['x']
        self.down = self.M.layers[LAYER].mlp.down_proj
        self.W = self.down.weight

    def q(self, x, f):
        return self.Q.act(x, f, self.rng)

    def head_ce(self, normed, ids):
        ce = []
        for i in range(0, normed.shape[0], LM_SUB):
            z = self.M.model.lm_head(normed[i:i + LM_SUB])
            ce.append(self.ce_rows(self.torch, z, ids[i:i + LM_SUB]))
            del z
        return self.torch.cat(ce)

    def forward_ce(self, ids):
        hs = self.M.model.model(input_ids=ids, attention_mask=self.M.wrapper.mask, use_cache=False).last_hidden_state
        return self.head_ce(hs, ids)

    def capture(self, ids):
        torch = self.torch
        cap, kw = {}, {}
        with torch.no_grad():
            nohook = self.forward_ce(ids)
            hooks = [self.M.layers[LAYER].post_attention_layernorm.register_forward_pre_hook(lambda m, a: cap.__setitem__('mid', a[0])),
                     self.down.register_forward_pre_hook(lambda m, a: cap.__setitem__('x', a[0])),
                     self.M.layers[1].register_forward_pre_hook(lambda m, a, k: kw.update(k), with_kwargs=True)]
            hooked = self.forward_ce(ids)
            for hk in hooks:
                hk.remove()
        if not torch.equal(nohook, hooked):
            raise SystemExit('hooks changed the forward')
        return nohook, cap['mid'], cap['x'], kw

    def replay(self, mid, xp, ids, kw):
        torch = self.torch
        with torch.no_grad():
            h = mid + self.down(xp)
            for layer in self.M.layers[LAYER + 1:]:
                o = layer(h, **kw)
                h = o[0] if isinstance(o, tuple) else o
            return self.head_ce(self.M.model.model.norm(h), ids)

    def hooked_full(self, ids, fn):
        torch = self.torch
        hk = self.down.register_forward_pre_hook(lambda m, a: (fn(a[0]),))
        with torch.no_grad():
            ce = self.forward_ce(ids)
        hk.remove()
        return ce

    def grad(self, mid, x, ids, kw):
        """g = d(summed CE of each window)/dx through the same suffix, microbatch 1; also max |g_x - g_y W| / max |g_x|."""
        torch = self.torch
        g, dev = torch.empty_like(x), 0.0
        for j in range(x.shape[0]):
            xj = x[j:j + 1].clone().requires_grad_(True)
            y = torch.nn.functional.linear(xj, self.W)
            h = mid[j:j + 1] + y
            for layer in self.M.layers[LAYER + 1:]:
                o = layer(h, **kw)
                h = o[0] if isinstance(o, tuple) else o
            z = self.M.model.lm_head(self.M.model.model.norm(h))
            ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            gx, gy = torch.autograd.grad(ce, (xj, y))
            g[j:j + 1] = gx
            dev = max(dev, float((gx - gy @ self.W).abs().max() / gx.abs().max()))
            del z, ce
        return g, dev


def part_stats(x, q, e, M, f, maxpos, sub=None):
    """explanatory statistics of one part (rows where M)."""
    import torch
    xs, es, qs = x[M], e[M], q[M]
    if xs.numel() == 0:
        return {'elements': 0}
    ax, ae = xs.abs().double().flatten(), es.abs().double().flatten()
    if sub is not None:
        ax, ae = ax[::sub], ae[::sub]
    qq = [0.5, 0.9, 0.99, 0.999]
    qt = torch.tensor(qq, dtype=torch.float64, device=x.device)
    r = {'elements': int(xs.numel()), 'x_abs_quantiles': torch.quantile(ax, qt).tolist(), 'x_abs_max': float(xs.abs().max()),
         'e_abs_quantiles': torch.quantile(ae, qt).tolist(), 'e_abs_max': float(es.abs().max()), 'sum_e2': float((es.double() ** 2).sum()),
         'zero_output_fraction': float((qs == 0).double().mean()), 'quantile_levels': qq, 'quantiles_from_subsample_stride': sub}
    if maxpos is not None:
        r['beyond_maxpos_count'] = int((xs.abs() > maxpos).sum())
        r['beyond_maxpos_fraction'] = float((xs.abs() > maxpos).double().mean())
    return r


# --------------------------------------------------------------------------- #
# calib (A)
# --------------------------------------------------------------------------- #
def calib(out):
    import run_qwen3_spike_recenter as sr
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    thr = threshold()
    t0 = time.perf_counter()
    env = Env(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    tm['model_load_s'] = time.perf_counter() - t0
    ctl = {'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled(),
           'attn_implementation': env.M.model.config._attn_implementation,
           'rms_norm_eps': [float(env.M.layers[0].post_attention_layernorm.variance_epsilon), float(env.M.model.model.norm.variance_epsilon)],
           'posit_lut_mismatches': env.Q.verify(), 'int_range_x': env.rng}
    W_hash = hashlib.sha256(env.W.detach().cpu().numpy().tobytes()).hexdigest()
    # posit value sets (maxpos from the actual candidate values)
    vals = {f: np.unique(env.Q.lut[f][1].cpu().numpy()) for f in FORMATS if f.startswith('p')}
    ctl['posit_value_sets'] = {f: {'count': int(len(v)), 'maxpos': float(np.abs(v).max()), 'minpos': float(np.abs(v[v != 0]).min()),
                                   'positive_values': [float(u) for u in v[v > 0]] if len(v) <= 64 else 'more than 64'} for f, v in vals.items()}
    maxpos = {f: (ctl['posit_value_sets'][f]['maxpos'] if f.startswith('p') else None) for f in FORMATS}
    # GPU QDQ of the CPU cache h vs the recenter error sha (elementwise)
    fz = read(REC / 'freeze.json')
    h_cpu = np.load(SPIKE / 'work/h_L2.npy')
    with torch.no_grad():
        ht = torch.from_numpy(h_cpu).cuda()
        ctl['gpu_qdq_of_cpu_h_equals_recenter'] = {f: hashlib.sha256(np.ascontiguousarray((env.q(ht, f) - ht).cpu().numpy()).tobytes()).hexdigest()
                                                   == fz['error_sha256_cpu'][f'L{LAYER}|{f}'] for f in FORMATS}
    del ht
    # capture
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(toks).cuda()
    t0 = time.perf_counter()
    nohook, mid, x, kw = env.capture(ids)
    base = env.replay(mid, x, ids, kw)
    torch.cuda.synchronize()
    tm['prefix_capture_and_baseline_s'] = time.perf_counter() - t0
    ctl['replay_baseline_vs_nohook_abs'] = float((base - nohook).abs().max())
    ctl['y_equals_linear_x'] = bool(torch.equal(env.down(x), torch.nn.functional.linear(x, env.W)))
    xc = x.reshape(-1, x.shape[-1]).cpu().numpy()
    ctl['gpu_x_vs_cpu_cache_h'] = {'max_abs': float(np.abs(xc - h_cpu).max()), 'max_rel_to_max': float(np.abs(xc - h_cpu).max() / np.abs(h_cpu).max()),
                                   'bitwise': bool(np.array_equal(xc, h_cpu))}
    a = x.abs().amax(-1)                                                   # [10, 128]
    S = a >= thr
    Nm = ~S
    ctl['mask_equals_recenter'] = bool(np.array_equal(S.reshape(-1).cpu().numpy(), np.load(REC / 'masks.npz')[f'L{LAYER}']))
    ctl['mask'] = {'threshold': thr, 'S_rows': int(S.sum()), 'S_positions': dict(Counter(int(p) for p in torch.nonzero(S)[:, 1].tolist())),
                   'S_rows_per_window': S.sum(1).tolist()}
    MS, MN = S[..., None], Nm[..., None]
    # errors
    t0 = time.perf_counter()
    q, e, d = {}, {}, {}
    with torch.no_grad():
        for f in FORMATS:
            q[f] = env.q(x, f)
            e[f] = q[f] - x
            d[(f, 'S')], d[(f, 'N')], d[(f, 'all')] = e[f] * MS, e[f] * MN, e[f]
    ctl['eS_plus_eN_equals_e'] = all(bool(torch.equal(d[(f, 'S')] + d[(f, 'N')], e[f])) for f in FORMATS)
    stats = {f: {p: part_stats(x, q[f], e[f], m_, f, maxpos[f]) for p, m_ in (('S', S), ('N', Nm), ('all', torch.ones_like(S)))} for f in FORMATS}
    torch.cuda.synchronize()
    tm['qdq_and_statistics_s'] = time.perf_counter() - t0
    # gradient (one baseline backward) and F
    t0 = time.perf_counter()
    g, gyw = env.grad(mid, x, ids, kw)
    torch.cuda.synchronize()
    tm['baseline_backward_s'] = time.perf_counter() - t0
    g_cpu = np.load(SPIKE / 'work/g_L2.npy')
    gc = g.reshape(-1, g.shape[-1]).cpu().numpy()
    ctl.update({'gx_vs_gyW_rel_max': gyw, 'g_position127_abs_max': float(g[:, -1].abs().max()),
                'gpu_g_vs_cpu_cache_g_rel_to_max': float(np.abs(gc - g_cpu).max() / np.abs(g_cpu).max())})
    F = {k: (g.double() * v.double()).sum((1, 2)).cpu().numpy() for k, v in d.items()}
    ctl['F_additivity_rel'] = float(max(abs(F[(f, 'S')].sum() + F[(f, 'N')].sum() - F[(f, 'all')].sum()) / max(abs(F[(f, 'all')].sum()), 1e-300)
                                        for f in FORMATS))
    # curves
    t0 = time.perf_counter()
    dl, pert = {}, {}
    L = lambda xp: (env.replay(mid, xp, ids, kw) - base).cpu().numpy()
    dl['alpha0'] = L(x)
    dl['zero_direction'] = L(x + 0.5 * torch.zeros_like(x))
    for f in FORMATS:
        for p in PARTS:
            dv = d[(f, p)]
            supp = dv != 0
            for al in ALPHAS:
                if al == 0:
                    continue
                xp = x + al * dv
                dl[f'{f}|{p}|{al!r}'] = L(xp)
                ad = (al * dv).double()
                pert[f'{f}|{p}|{al!r}'] = {'alpha_norm': float(ad.norm()), 'max_abs': float(ad.abs().max()),
                                           'changed_fraction': float((xp != x)[supp].float().mean()) if bool(supp.any()) else 0.0,
                                           'rounding_rel': float(((xp.double() - x.double()) - ad).norm() / ad.norm()) if float(ad.norm()) > 0 else 0.0}
        # alpha = 1 main: the masked QDQ path
        for p, xp in (('S', torch.where(MS, q[f], x)), ('N', torch.where(MN, q[f], x)), ('all', q[f])):
            dl[f'{f}|{p}|main'] = L(xp)
            rec = x + d[(f, p)]
            pert[f'{f}|{p}|reconstruction'] = {'input_max_abs_diff': float((rec - xp).abs().max()), 'input_fraction_differing': float((rec != xp).float().mean()),
                                               'loss_max_abs_diff_summed_ce': float(np.abs(dl[f'{f}|{p}|1.0'] - dl[f'{f}|{p}|main']).max())}
    torch.cuda.synchronize()
    tm['calibration_curves_s'] = time.perf_counter() - t0
    # hooked full forward = replay (alpha = 1 main, all formats and parts)
    t0 = time.perf_counter()
    dev = 0.0
    for f in FORMATS:
        for p, fn in (('S', lambda a_, f=f: torch.where((a_.abs().amax(-1) >= thr)[..., None], env.q(a_, f), a_)),
                      ('N', lambda a_, f=f: torch.where((a_.abs().amax(-1) < thr)[..., None], env.q(a_, f), a_)), ('all', lambda a_, f=f: env.q(a_, f))):
            dev = max(dev, float(np.abs((env.hooked_full(ids, fn) - nohook).cpu().numpy() - dl[f'{f}|{p}|main']).max()))
    ctl['hooked_full_vs_replay_abs'] = dev
    tm['hooked_full_checks_s'] = time.perf_counter() - t0
    ctl['W_down_sha256_unchanged'] = hashlib.sha256(env.W.detach().cpu().numpy().tobytes()).hexdigest() == W_hash
    # previous calibration losses (variance diagnostic: different LM-head sub-batch; reported)
    vz = np.load(VDIAG / 'losses.npz')
    ctl['calibration_all_vs_variance_diag_Qx_abs'] = {f: float(np.abs(dl[f'{f}|all|main'] - vz[f'L2__{f}__e_direct']).max()) for f in FORMATS}
    # D components from the CPU cache
    t0 = time.perf_counter()
    errs_cpu = {}
    import torch as t_
    from run_qwen_down_spike_diag import Quant
    Qc = Quant(out / 'work', 'cpu')
    with t_.no_grad():
        hcpu = t_.from_numpy(h_cpu)
        for f in FORMATS:
            errs_cpu[f] = (Qc.act(hcpu, f, env.rng) - hcpu).numpy()
    win = np.repeat(np.arange(CAL), SEQ)
    comp, Sc = sr.components(h_cpu, g_cpu, errs_cpu, thr, win)
    old_c = {r['format']: r for r in csv.DictReader(open(REC / 'components.csv')) if r['layer'] == str(LAYER)}
    old_s = {r['format']: r for r in csv.DictReader(open(REC / 'scores.csv')) if r['layer'] == str(LAYER)}
    prox, drep = {}, 0.0
    for f in FORMATS:
        c = comp[f]['components']
        prox[f] = {'P_S': c['P_S(0)'], 'V_N(mu_N)': c['V_N(mu_N)'], 'U_N(mu_N)': c['U_N(mu_N)'], 'C_N': c['V_N(mu_N)'] + c['U_N(mu_N)'],
                   'D': comp[f]['scores']['D'], 'A': comp[f]['scores']['A'], 'V_S(0)': c['V_S(0)']}
        for k, kk in (('P_S', 'P_S(0)'), ('V_N(mu_N)', 'V_N(mu_N)'), ('U_N(mu_N)', 'U_N(mu_N)')):
            drep = max(drep, abs(prox[f][k] - float(old_c[f][kk])) / abs(float(old_c[f][kk])))
        drep = max(drep, abs(prox[f]['D'] - float(old_s[f]['D'])) / abs(float(old_s[f]['D'])))
    ctl['D_components_reproduction_rel'] = drep
    ctl['cpu_mask_equals_recenter'] = bool(np.array_equal(Sc, np.load(REC / 'masks.npz')[f'L{LAYER}']))
    tm['proxy_reproduction_s'] = time.perf_counter() - t0
    ctl['alpha0_abs'] = float(np.abs(dl['alpha0']).max())
    ctl['zero_direction_abs'] = float(np.abs(dl['zero_direction']).max())
    ctl['nonfinite'] = int(sum((~np.isfinite(v)).sum() for v in dl.values()))
    t = TOL
    ctl['checks'] = {
        'environment': ctl['tf32'] == [False, False] and ctl['deterministic'] and ctl['attn_implementation'] == 'eager' and ctl['rms_norm_eps'] == [1e-6, 1e-6]
        and all(v[0] == 0 for v in ctl['posit_lut_mismatches'].values()),
        'gpu_qdq': all(ctl['gpu_qdq_of_cpu_h_equals_recenter'].values()),
        'replay': ctl['replay_baseline_vs_nohook_abs'] <= t['replay_vs_nohook_abs_summed_ce'] and dev <= t['hooked_full_vs_replay_abs_summed_ce'],
        'slot': ctl['y_equals_linear_x'] and ctl['W_down_sha256_unchanged'],
        'mask': ctl['mask_equals_recenter'] and ctl['cpu_mask_equals_recenter'],
        'D_components': drep <= t['D_components_rel'],
        'directions': ctl['eS_plus_eN_equals_e'] and ctl['alpha0_abs'] <= t['alpha0_abs'] and ctl['zero_direction_abs'] <= t['zero_direction_abs'],
        'gradient': gyw <= t['gx_vs_gyW_rel'] and ctl['g_position127_abs_max'] <= t['pos127_abs'] and ctl['F_additivity_rel'] <= t['F_additivity_rel'],
        'finite': ctl['nonfinite'] == 0}
    ctl['checks'] = {k: bool(v) for k, v in ctl['checks'].items()}
    ctl['passed'] = bool(all(ctl['checks'].values()))
    write(out / 'controls_calib.json', ctl)
    np.savez(out / 'calib_losses.npz', base=base.cpu().numpy(), **{k.replace('|', '__'): v for k, v in dl.items()})
    write(out / 'calib_raw.json', {'F_per_window': {f'{f}|{p}': F[(f, p)].tolist() for f in FORMATS for p in PARTS}, 'perturbation': pert,
                                   'statistics': stats, 'proxy': prox})
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    log_stage(out, 'calib', time.perf_counter() - t_all, dict(tm))
    log(f'calib controls passed={ctl["passed"]}; checks {ctl["checks"]}')
    if not ctl['passed']:
        raise SystemExit('calibration controls failed (see controls_calib.json)')


# --------------------------------------------------------------------------- #
# replay (B, alpha = 1 partitions on the recenter articles)
# --------------------------------------------------------------------------- #
def replay(out):
    t_all = time.perf_counter()
    check_plan(out)
    if not read(out / 'controls_calib.json')['passed']:
        raise SystemExit('calibration controls did not pass')
    thr = threshold()
    env = Env(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    toks = np.load(REC / 'tokens.npy')
    nw = len(toks)
    rz = np.load(REC / 'eval_ce.npz')
    rec_fmt = [str(u) for u in rz['formats']]
    rce = rz['ce'][LAYER]                                                  # [7, windows]
    maxpos = {f: float(np.abs(env.Q.lut[f][1].cpu().numpy()).max()) if f.startswith('p') else None for f in FORMATS}
    hdr0 = {'tokens_sha256': sha(REC / 'tokens.npy'), 'script_sha256': sha(HERE / 'run_qwen3_raw_error_spike_diagnostic.py'), 'threshold': thr}
    nchunk = -(-nw // CHUNK)
    for k, c0 in enumerate(range(0, nw, CHUNK)):
        ck = out / 'work' / f'replay_chunk{k:02d}.npz'
        hdr = hdr0 | {'chunk': k, 'windows': [c0, min(nw, c0 + CHUNK)]}
        if ck.exists():
            if json.loads(str(np.load(ck)['header'])) != hdr:
                raise SystemExit(f'{ck}: checkpoint from another version / data')
            log(f'chunk {k}: reusing checkpoint')
            continue
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).cuda()
        t1 = time.perf_counter()
        nohook, mid, x, kw = env.capture(ids)
        base = env.replay(mid, x, ids, kw)
        S = x.abs().amax(-1) >= thr
        MS, MN = S[..., None], (~S)[..., None]
        ce = np.full((len(FORMATS), len(PARTS), len(ids)), np.nan)
        st = {}
        with torch.no_grad():
            for fi, f in enumerate(FORMATS):
                q = env.q(x, f)
                e = q - x
                for pi, xp in enumerate((torch.where(MS, q, x), torch.where(MN, q, x), q)):
                    ce[fi, pi] = env.replay(mid, xp, ids, kw).cpu().numpy()
                for p, m_ in (('S', S), ('N', ~S)):
                    xs, es, qs = x[m_], e[m_], q[m_]
                    st[f'{f}|{p}'] = [float(xs.numel()), float((es.double() ** 2).sum()), float((qs == 0).sum()), float(xs.abs().max()) if xs.numel() else 0.0,
                                      float(es.abs().max()) if es.numel() else 0.0, float((xs.abs() > maxpos[f]).sum()) if maxpos[f] else 0.0]
                    st[f'{f}|{p}|sub_x'] = xs.abs().flatten()[::STRIDE].cpu().numpy()
                    st[f'{f}|{p}|sub_e'] = es.abs().flatten()[::STRIDE].cpu().numpy()
        torch.cuda.synchronize()
        rec_dev = {'identity': float(np.abs(base.cpu().numpy() - rce[0, c0:c0 + len(ids)]).max()),
                   'all': float(max(np.abs(ce[fi, 2] - rce[rec_fmt.index(f), c0:c0 + len(ids)]).max() for fi, f in enumerate(FORMATS)))}
        np.savez(ck, ce=ce, base=base.cpu().numpy(), nohook=nohook.cpu().numpy(), S=S.cpu().numpy(), rec_dev=json.dumps(rec_dev),
                 stats=json.dumps({k_: v for k_, v in st.items() if not k_.endswith(('sub_x', 'sub_e'))}),
                 **{k_.replace('|', '__'): v for k_, v in st.items() if k_.endswith(('sub_x', 'sub_e'))}, header=json.dumps(hdr))
        log(f'chunk {k + 1}/{nchunk}: {time.perf_counter() - t1:.0f}s; recenter dev {rec_dev}; S rows {int(S.sum())}')
    parts = [np.load(out / 'work' / f'replay_chunk{k:02d}.npz') for k in range(nchunk)]
    ce = np.concatenate([p['ce'] for p in parts], axis=-1)
    base = np.concatenate([p['base'] for p in parts])
    nohook = np.concatenate([p['nohook'] for p in parts])
    S = np.concatenate([p['S'] for p in parts])
    rec_dev = [json.loads(str(p['rec_dev'])) for p in parts]
    agg = defaultdict(lambda: np.zeros(6))
    for p in parts:
        for k_, v in json.loads(str(p['stats'])).items():
            a_ = agg[k_]
            a_[:3] += v[:3]
            a_[3], a_[4] = max(a_[3], v[3]), max(a_[4], v[4])
            a_[5] += v[5]
    bstats = {}
    for k_, v in agg.items():
        sx = np.concatenate([p[k_.replace('|', '__') + '__sub_x'] for p in parts])
        se = np.concatenate([p[k_.replace('|', '__') + '__sub_e'] for p in parts])
        bstats[k_] = {'elements': int(v[0]), 'sum_e2': v[1], 'zero_output_fraction': v[2] / max(v[0], 1), 'x_abs_max': v[3], 'e_abs_max': v[4],
                      'beyond_maxpos_fraction': v[5] / max(v[0], 1), 'x_abs_quantiles_subsample': np.quantile(sx, [0.5, 0.9, 0.99, 0.999]).tolist() if len(sx) else None,
                      'e_abs_quantiles_subsample': np.quantile(se, [0.5, 0.9, 0.99, 0.999]).tolist() if len(se) else None}
    np.savez(out / 'replay_ce.npz', ce=ce, base=base, nohook=nohook, S=S, formats=np.array(FORMATS), parts=np.array(PARTS))
    ctl = {'identity_vs_recenter_abs_max': max(r['identity'] for r in rec_dev), 'all_vs_recenter_abs_max': max(r['all'] for r in rec_dev),
           'replay_baseline_vs_nohook_abs_max': float(np.abs(base - nohook).max()), 'nonfinite': int((~np.isfinite(ce)).sum()),
           'S_rows': int(S.sum()), 'S_positions': dict(Counter(int(p) for p in np.nonzero(S)[1].tolist()).most_common(12)),
           'windows_without_S': int((S.sum(1) == 0).sum()), 'S_rows_per_window_counts': dict(Counter(int(v) for v in S.sum(1)))}
    ctl['passed'] = bool(ctl['identity_vs_recenter_abs_max'] <= TOL['B_identity_vs_recenter_abs_summed_ce'] and ctl['all_vs_recenter_abs_max']
                         <= TOL['B_all_vs_recenter_abs_summed_ce'] and ctl['replay_baseline_vs_nohook_abs_max'] <= TOL['replay_vs_nohook_abs_summed_ce']
                         and ctl['nonfinite'] == 0)
    write(out / 'controls_replay.json', ctl)
    write(out / 'replay_statistics.json', bstats)
    log_stage(out, 'replay', time.perf_counter() - t_all, {'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
                                                           'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, 'windows': nw})
    log(f'replay controls passed={ctl["passed"]}: {ctl}')
    if not ctl['passed']:
        raise SystemExit('replay controls failed')


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    check_plan(out)
    cc, cr = read(out / 'controls_calib.json'), read(out / 'controls_replay.json')
    raw = read(out / 'calib_raw.json')
    z = np.load(out / 'calib_losses.npz')
    dl = {k.replace('__', '|'): z[k] for k in z.files}
    per = lambda v: float(np.sum(v)) / TARGETS
    pos_alphas = [a for a in ALPHAS if a > 0]
    # A: curves
    curves, A1, pairsA = [], {}, []
    for f in FORMATS:
        for p in PARTS:
            Fv = float(np.sum(raw['F_per_window'][f'{f}|{p}']))
            A1[(f, p)] = per(dl[f'{f}|{p}|main'])
            for a in pos_alphas:
                dp, dm = per(dl[f'{f}|{p}|{a!r}']), per(dl[f'{f}|{p}|{-a!r}'])
                pp, pm = raw['perturbation'][f'{f}|{p}|{a!r}'], raw['perturbation'][f'{f}|{p}|{-a!r}']
                curves.append({'format': f, 'part': p, 'alpha': a, 'dL_plus': dp, 'dL_minus': dm, 'O_over_alpha': (dp - dm) / 2 / a,
                               'E_over_alpha2': (dp + dm) / 2 / a ** 2, 'F': Fv / TARGETS, 'alpha_norm': pp['alpha_norm'], 'max_abs_perturbation': pp['max_abs'],
                               'changed_fraction_plus': pp['changed_fraction'], 'changed_fraction_minus': pm['changed_fraction'],
                               'rounding_rel_plus': pp['rounding_rel'], 'rounding_rel_minus': pm['rounding_rel']})
    I = {f: A1[(f, 'all')] - A1[(f, 'S')] - A1[(f, 'N')] for f in FORMATS}
    Ia = lambda f, a: per(dl[f'{f}|all|{a!r}']) - per(dl[f'{f}|S|{a!r}']) - per(dl[f'{f}|N|{a!r}'])
    for i_, pnt in PAIRS:
        row = {'data': 'calibration', 'pair': f'{pnt}-{i_}', 'alpha': 'main (masked QDQ, 1)', 'D_S': A1[(pnt, 'S')] - A1[(i_, 'S')],
               'D_N': A1[(pnt, 'N')] - A1[(i_, 'N')], 'D_int': I[pnt] - I[i_], 'D_all': A1[(pnt, 'all')] - A1[(i_, 'all')]}
        row['identity_residual'] = row['D_all'] - row['D_S'] - row['D_N'] - row['D_int']
        pairsA.append(row)
        for a in [x_ for x_ in ALPHAS if x_ != 0]:
            r2 = {'data': 'calibration', 'pair': f'{pnt}-{i_}', 'alpha': a, 'D_S': per(dl[f'{pnt}|S|{a!r}']) - per(dl[f'{i_}|S|{a!r}']),
                  'D_N': per(dl[f'{pnt}|N|{a!r}']) - per(dl[f'{i_}|N|{a!r}']), 'D_int': Ia(pnt, a) - Ia(i_, a),
                  'D_all': per(dl[f'{pnt}|all|{a!r}']) - per(dl[f'{i_}|all|{a!r}'])}
            r2['identity_residual'] = r2['D_all'] - r2['D_S'] - r2['D_N'] - r2['D_int']
            pairsA.append(r2)
    # B: per article
    rz = np.load(out / 'replay_ce.npz')
    ce, base, Sb = rz['ce'], rz['base'], rz['S']                           # [6, 3, W], [W], [W, 128]
    na = ce.shape[-1] // WPA
    art = lambda v: v.reshape(*v.shape[:-1], na, WPA).sum(-1) / (WPA * TP)
    dB = art(ce - base)                                                    # [6, 3, A]
    IB = dB[:, 2] - dB[:, 0] - dB[:, 1]                                    # [6, A]
    fi = {f: i for i, f in enumerate(FORMATS)}
    rng = np.random.default_rng(BOOT_SEED)
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na

    def desc(v):
        b = Wb @ v
        return {'mean': float(v.mean()), 'median': float(np.median(v)), 'share_positive': float((v > 0).mean()),
                'ci95_posthoc_descriptive': np.percentile(b, [2.5, 97.5]).tolist()}
    B_fmt = {f: {'S': desc(dB[fi[f], 0]), 'N': desc(dB[fi[f], 1]), 'all': desc(dB[fi[f], 2]), 'I_SN': desc(IB[fi[f]])} for f in FORMATS}
    pairsB = []
    for i_, pnt in PAIRS:
        comp = {'D_S': dB[fi[pnt], 0] - dB[fi[i_], 0], 'D_N': dB[fi[pnt], 1] - dB[fi[i_], 1], 'D_int': IB[fi[pnt]] - IB[fi[i_]],
                'D_all': dB[fi[pnt], 2] - dB[fi[i_], 2]}
        res_ = float(np.abs(comp['D_all'] - comp['D_S'] - comp['D_N'] - comp['D_int']).max())
        pairsB.append({'data': 'B (128 recenter articles, post-hoc)', 'pair': f'{pnt}-{i_}', 'alpha': 'main (masked QDQ, 1)',
                       **{k: desc(v) for k, v in comp.items()}, 'identity_residual_max': res_})
    cov = {'articles': int(na), 'windows': int(Sb.shape[0]), 'S_rows': int(Sb.sum()), 'windows_without_S': int((Sb.sum(1) == 0).sum()),
           'articles_without_S': int((Sb.reshape(na, WPA, -1).sum((1, 2)) == 0).sum()),
           'S_positions_top': dict(Counter(int(p) for p in np.nonzero(Sb)[1].tolist()).most_common(12))}
    # proxies vs measured
    prox = raw['proxy']
    ptab = [{'format': f, 'P_S_raw': prox[f]['P_S'], 'C_N_raw': prox[f]['C_N'], 'V_N(mu_N)_raw': prox[f]['V_N(mu_N)'], 'U_N(mu_N)_raw': prox[f]['U_N(mu_N)'],
             'D_raw': prox[f]['D'], 'A_raw': prox[f]['A'], 'P_S_half_per1270': prox[f]['P_S'] / (2 * TARGETS), 'C_N_half_per1270': prox[f]['C_N'] / (2 * TARGETS),
             'D_half_per1270': prox[f]['D'] / (2 * TARGETS), 'calib_S_only': A1[(f, 'S')], 'calib_N_only': A1[(f, 'N')], 'calib_all': A1[(f, 'all')],
             'calib_I_SN': I[f], 'B_S_only': B_fmt[f]['S']['mean'], 'B_N_only': B_fmt[f]['N']['mean'], 'B_all': B_fmt[f]['all']['mean'],
             'B_I_SN': B_fmt[f]['I_SN']['mean'], 'F_S': float(np.sum(raw['F_per_window'][f'{f}|S'])) / TARGETS,
             'F_N': float(np.sum(raw['F_per_window'][f'{f}|N'])) / TARGETS, 'F_all': float(np.sum(raw['F_per_window'][f'{f}|all'])) / TARGETS} for f in FORMATS]
    # pair-identity check
    pid = max(abs(r['identity_residual']) / max(abs(r['D_all']), 1e-300) for r in pairsA)
    res = {'units': 'nat/token (calibration summed CE / 1270; B per article / 381); proxies raw (and /2540 by convention only)',
           'calibration_alpha1': {f: {p: A1[(f, p)] for p in PARTS} | {'I_SN': I[f]} for f in FORMATS}, 'B': B_fmt, 'B_coverage': cov,
           'pairs_calibration_main': [r for r in pairsA if not isinstance(r['alpha'], float)], 'pairs_B': pairsB, 'proxy_vs_measured': ptab,
           'pair_identity_rel_max_calibration': pid, 'controls': {'calib': cc['passed'], 'replay': cr['passed']},
           'posit_value_sets': cc['posit_value_sets'], 'statistics_calibration': raw['statistics'], 'statistics_B': read(out / 'replay_statistics.json')}
    write(out / 'analysis.json', res)

    def dump(name, rows):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    dump('curves.csv', curves)
    dump('paired_decomposition.csv', pairsA + [{'data': r['data'], 'pair': r['pair'], 'alpha': r['alpha'], **{k: r[k]['mean'] for k in ('D_S', 'D_N', 'D_int', 'D_all')},
                                                 'identity_residual': r['identity_residual_max']} for r in pairsB])
    dump('proxy_components.csv', ptab)
    pw = []
    for f in FORMATS:
        for p in PARTS:
            for key in ['main'] + [repr(a) for a in ALPHAS if a != 0]:
                for w in range(CAL):
                    pw.append({'format': f, 'part': p, 'alpha': key, 'window': w, 'dCE_summed': float(dl[f'{f}|{p}|{key}'][w]),
                               'F_window': raw['F_per_window'][f'{f}|{p}'][w]})
    dump('per_window.csv', pw)
    segs = read(REC / 'segments.json')
    pa = []
    for a in range(na):
        row = {'article': a, 'title': segs[a]['title'], 'S_rows': int(Sb.reshape(na, WPA, -1)[a].sum())}
        for f in FORMATS:
            row.update({f'{f}_S': dB[fi[f], 0, a], f'{f}_N': dB[fi[f], 1, a], f'{f}_all': dB[fi[f], 2, a], f'{f}_I_SN': IB[fi[f], a]})
        pa.append(row)
    dump('per_article.csv', pa)
    # figures
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    cols = {'S': '#d62728', 'N': '#1f77b4', 'all': '#2ca02c'}
    for col, (i_, pnt) in enumerate(PAIRS):
        for f, ls in ((i_, '--'), (pnt, '-')):
            for p in PARTS:
                rows = [r for r in curves if r['format'] == f and r['part'] == p]
                axes[0, col].plot([r['alpha'] for r in rows], [r['dL_plus'] for r in rows], ls, color=cols[p], marker='o', ms=2, label=f'{f} {p}: dL(+alpha)')
                axes[1, col].plot([r['alpha'] for r in rows], [r['E_over_alpha2'] for r in rows], ls, color=cols[p], marker='o', ms=2, label=f'{f} {p}: E/alpha^2')
            axes[1, col].axhline(prox[f]['P_S'] / (2 * TARGETS), color=cols['S'], ls=':' if f == pnt else '-.', lw=0.8, label=f'{f} P_S/2540')
            axes[1, col].axhline(prox[f]['C_N'] / (2 * TARGETS), color=cols['N'], ls=':' if f == pnt else '-.', lw=0.8, label=f'{f} C_N/2540')
        for r_ in (0, 1):
            axes[r_, col].set_xscale('log', base=2)
            axes[r_, col].set_yscale('symlog', linthresh=1e-4)
            axes[r_, col].set_xlabel('alpha (> 0)')
            axes[r_, col].legend(fontsize=5)
        axes[0, col].set_title(f'L2 raw-error partitions, calibration: {i_} (dashed) vs {pnt} (solid)', fontsize=8)
        axes[0, col].set_ylabel('dL(+alpha), nat/token')
        axes[1, col].set_ylabel('E(alpha)/alpha^2 (finite even response)')
    fig.tight_layout()
    fig.savefig(out / 'fig_partition_curves.png', dpi=110)
    plt.close(fig)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'calibration_alpha1': res['calibration_alpha1'], 'pairs_calibration': res['pairs_calibration_main'],
                    'pairs_B': [{k: (v['mean'] if isinstance(v, dict) else v) for k, v in r.items()} for r in pairsB]}, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'replay', 'analyze'))
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
        {'calib': calib, 'replay': replay, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_raw_error_spike_diagnostic.py')}) + '\n')


if __name__ == '__main__':
    main()
