#!/usr/bin/env python3
"""Qwen3-0.6B L26 Down with a real block-27 suffix replay: does the top-256 head score of runs/qwen3_topk_ce_curvature_20261026
keep the candidate ranking when the target is not the last layer, and does it still save time once every candidate goes through
block 27? Target /model/model/layers.26/mlp/down_proj/MatMul, weight fixed to A's INT6; backgrounds F (other 195 projections per A,
block 27 included) and U (other projections FP32); nine activation formats on the target's input and output. Per candidate:
  z26_f = Q_y,f(linear(Q_x,f(x26), W26q)),  h27_f = r26 + z26_f,  u_f = final_norm(block27(h27_f, captured kwargs)),
  logits_f = LM_head(u_f);  S_u = g_u . (u_f - u_0) with g_u from the head only (u_0 a leaf, one backward per window);
  P256 = S_u + Khat256 (the previous round's implementation and precision, unchanged).
C = the original 10 calibration windows (scores, references, controls, frozen selectors); N = 128 new articles x 2 windows (truth
only). GPU (LMReplay QDQ, block 27 with its background QDQ); no change to the formal BC, A, quantizers, MILP / knee or native.

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
import run_qwen3_topk_ce_curvature as tk  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
ENDP, NATIVE = cs.ENDP, cs.NATIVE
PREV = HERE / 'runs/qwen3_topk_ce_curvature_20261026'
LAYER = 26
TARGET = f'/model/model/layers.{LAYER}/mlp/down_proj/MatMul'
TARGET_W = 'int6'
BGS = ('F', 'U')                                               # F primary
FORMATS, SELECT_SET, POS = cs.FORMATS, cs.SELECT_SET, cs.POS
CANDS = tuple(f for f in SELECT_SET if f != 'f16')
SEQ, TP, CAL, BATCH, CHUNK = cs.SEQ, cs.TP, cs.CAL, cs.BATCH, cs.CHUNK
K = 256
Q_EPS = tk.Q_EPS
REF_BG, REF_WINDOWS = 'F', (0, 9)
N_ART, WPA, BOOT = 128, 2, 20000
SEEDS = {'order': 26102711, 'start': 26102712, 'bootstrap': 26102713}
BENCH = {'background': 'F', 'warmup': 5, 'reps': 20}
SEVERE = 0.01
TERMS = ('D', 'S_u', 'Khat256', 'K2_actual', 'K2_aff', 'KL', 'D64', 'S_logit64', 'eps_head', 'eps_CE', 'tailvar256', 'Khat256_fullA', 'CE')
TOL = {'replay_vs_full_ce_abs_per_position': 1e-6, 'replay_vs_full_tensor_abs': 1e-6, 'replay_vs_full_tensor_rel': 1e-6,
       'reproduce_prev_F_f16_ce_abs_per_position': 1e-6, 'switch_reproduce_abs_per_position': 0.0,
       'nested_abs': 1e-9, 'nested_rel': 1e-4, 'tail_identity_abs': 1e-9, 'tail_identity_rel': 1e-4,
       'cheap_vs_fullA_abs': 1e-9, 'cheap_vs_fullA_rel': 1e-4, 'fp32_vs_float64_abs': 1e-8, 'fp32_vs_float64_rel': 1e-4,
       'wbar_rel': 1e-5, 'g_u_vs_analytic_float64_rel': 1e-4, 'identity_D64_abs_per_position': 1e-9, 'kl_min_per_position': -1e-10,
       'synthetic_abs': 1e-12, 'position_sum_abs_summed': 1e-9, 'pos127_abs': 0.0, 'f16_zero_abs': 0.0}
CODE = ['run_qwen3_l26_suffix_bridge.py'] + tk.CODE
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
    prev = [PREV / f for f in ('report.md', 'plan.json', 'frozen_predictions.json', 'C_terms.npz', 'controls_calib.json', 'bench.json',
                               'stats.json', 'freeze.json', 'data_freeze.json', 'segments.json', 'tokens.npy')]
    return tk.inputs() + prev + [HERE / 'reports/qwen3_l26_suffix_bridge_prompt_20261027.md']


def background(A, b):
    """target: A's weight, activation applied by our hooks (LMReplay sees fp32 and skips it); F: other nodes per A; U: FP32."""
    return {n: ({'w': A[n]['w'], 'a': 'fp32'} if n == TARGET else dict(A[n]) if b == 'F' else {'w': 'fp32', 'a': 'fp32'}) for n in A}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or A[TARGET] != {'w': TARGET_W, 'a': 'f16'}:
        raise SystemExit('A is not the original knee or the target is not int6 / f16')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {'time': now(), 'spec': 'reports/qwen3_l26_suffix_bridge_prompt_20261027.md and the request of 2026-10-27 (L26, INT6 weight, k = 256 fixed)',
          'scope': f'Qwen3-0.6B {TARGET} only; target weight = A\'s {TARGET_W} (from the FP32 weight by the existing quantizer, cached once) in '
                   'every condition; backgrounds F (other 195 projections per A, block 27 weights and activation QDQ included; primary) and '
                   'U (other projections FP32); nine target activation formats on input and output; GPU LMReplay, FP32, eager, TF32 off, '
                   'deterministic, use_cache=False',
          'candidate path': {'z26': 'Q_y,f(linear(Q_x,f(x26), W26q)), x26 = raw Down input captured before the input QDQ, in the full forward\'s '
                                     'batches of 8 / tail 2', 'h27': 'r26 + z26 (r26 = input of layers.26.post_attention_layernorm)',
                             'block27': 'the actual decoder layer 27 with the kwargs captured from the same batch of the baseline full forward '
                                        '(attention mask, position ids / embeddings, cache position; no cache), all background hooks active',
                             'u': 'final RMSNorm once; logits = unquantized tied LM head'},
          'scores': {'g_u': 'd(summed FP32 next-token CE)/du at the f16 u_0 (leaf), head only, one autograd call per window (microbatch 1)',
                     'S_u': 'g_u . (u_f - u_0), float64 difference of the FP32 u, float64 product',
                     'Khat256 / P256': 'exactly run_qwen3_topk_ce_curvature.prep / khat (p = float64 softmax of the microbatch-1 baseline logits, '
                                       'top-256 by p with ties by token id, q direct tail sum, w_tail = W^T(p 1_tail) FP32 GEMM, q <= 1e-12 '
                                       'branch); P256 = S_u + Khat256',
                     'references (C)': 'D (FP32 CE difference, batch-8 head), K2_actual = 1/2 Var_p0(actual logit difference) with p0 from '
                                       'the batch-8 logits (float64), K2_affine = 1/2 Var_p(W du) (same p as Khat), KL_64, D_64, S_logit,64, '
                                       'eps_head = S_logit,64 - S_u, eps_CE = D - D_64; tail variance K2_affine - Khat256, quadratic '
                                       'remainder KL - K2_actual; these are head-side terms, not cross-layer propagation errors',
                     'positions': 'P0 / Rest / All are positions of the final CE only (no source-position attribution)'},
          'controls': [f'C, both backgrounds x 9 formats: replay vs complete forward (block-27 input, final u: |d| <= '
                       f'{TOL["replay_vs_full_tensor_abs"]} + {TOL["replay_vs_full_tensor_rel"]}|ref|; per-position CE <= '
                       f'{TOL["replay_vs_full_ce_abs_per_position"]})',
                       f'F / f16 per-position CE on C reproduces runs/qwen3_topk_ce_curvature_20261026 F / f16 within '
                       f'{TOL["reproduce_prev_F_f16_ce_abs_per_position"]}; F -> U -> F switch reproduces F exactly',
                       f'target weight = cached {TARGET_W} of the FP32 weight; background weights / activation map equal the configuration; hook '
                       'counts (target 2 pre + 2 post = LMReplay no-op + ours; no duplicates)', 'kwargs: no cache object, use_cache False, '
                       'tensors of the same batch', 'f16: du, S, K, D exactly 0; fp32 not forced to 0', 'LM head affine Linear, no bias / '
                       'softcap / temperature', f'g_u vs float64 W^T(p - onehot) (F, windows {list(REF_WINDOWS)}) within rel '
                       f'{TOL["g_u_vs_analytic_float64_rel"]}', 'position 127: g_u = 0, 127 target positions only',
                       'synthetic top-k checks of the previous round; Khat256 <= K2_affine, tail identity, cheap vs full-a, float64 reference '
                       f'(F, windows {list(REF_WINDOWS)}) with the previous tolerances', f'D_64 = S_logit,64 + KL_64 within '
                       f'{TOL["identity_D64_abs_per_position"]}; KL >= {TOL["kl_min_per_position"]}; P0 + Rest = All',
                       'failures are kept; bug fixes need an amendment'],
          'selectors (frozen on C)': {'set': list(SELECT_SET), 'primary': 'argmin P256,All', 'quadratic_reference': 'argmin (S_u + K2_actual),All',
                                      'measured_C_reference': 'argmin D_C,All', 'first_order': 'argmin S_u,All', 'original_config_control': 'f16',
                                      'ties': 'exact equality: f16 first, then the set order'},
          'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'use': 'truth only (per-position CE, 2 backgrounds x 9 formats); no '
                   'gradient, proxy, range, k or selector on N', 'rule': 'run_qwen_prefix_rank_shift.data with the verified manifest scan, '
                   'near-duplicate controls, historical Qwen shingle audit and the subset resolution of '
                   'runs/qwen3_conditional_signed_score_20261026', 'excluded_previous_wt103': len(used),
                   'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
          'statistics': {'unit': 'article (2 windows -> one 254-target contribution)', 'bootstrap': f'{BOOT} paired article resamples, seed '
                         f'{SEEDS["bootstrap"]}, one index set for every comparison; 95% percentile CI',
                         'primary': 'F: N mean D of the frozen primary pick vs f16; confirmed improvement only if mean < 0 and CI upper < 0; an '
                                    'f16 pick = 0 = no added quality gain',
                         'secondary': ['primary - first_order / quadratic_reference / measured_C_reference (F, U)', 'U picks',
                                       'frozen P256 / S_u / S_u+K2_actual / D_C directions (8), 28 pairs, 3 same-bit pairs vs N (CI rule)',
                                       'descriptive N oracle', 'P0 / Rest and article signs of every pick'],
                         'severe harm': f'mean > {SEVERE} and CI lower > 0', 'reading': 'different picks with a CI covering 0 = undetermined '
                                                                                        '(not equivalence)'},
          'cost': {'protocol': f'background {BENCH["background"]}, C (10 windows, batches 8 + 2), {len(CANDS)} non-f16 candidates (f16 reuses the '
                               f'baseline); every method includes per candidate target QDQ / GEMM, block 27 and the final norm; CUDA synchronize '
                               f'at region boundaries; {BENCH["warmup"]} warmups + {BENCH["reps"]} timed repetitions; median / p90',
                   'prefix': 'one complete f16 forward capturing x26, r26, block-27 kwargs, u_0 and the baseline CE (shared by all methods; '
                             'directCE reuses its CE, the gradient methods reuse u_0 and add a head forward + backward)',
                   'methods': ['suffix + S_u', 'suffix + S_u + Khat256', 'suffix + S_u + full K2 (FP32 GEMM, FP32 reductions)',
                               'suffix + direct head + CE'],
                   'levels': ['candidate hot increment (baseline cached)', 'baseline + candidates', 'prefix + baseline + candidates',
                              'reconstructed cold (replay init + cold background set + first prefix + first run of each method; not separate '
                              'cold processes)'],
                   'breakdown': 'candidate target QDQ / GEMM, block 27, norm, head part (head-only diagnostic); baseline sub-steps; peak '
                                'VRAM increment; W_T cache (all 10 windows cached, no chunking at C size); RSS'},
          'tolerances': TOL, 'q_eps': Q_EPS, 'k': K, 'smoke': smoke or 'none',
          'boundaries': 'no change to the formal BC / raw BC, A, quantizers, MILP / knee or native; no training, STE, HVP, JVP, low rank, '
                        'coefficient, other k or other target; every candidate is replayed through block 27, so this is a bridge step, '
                        'not a replay-free local criterion',
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L26 Down 的 block27 後綴重播與 top-256 評分', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        f'- 目標 {TARGET}，權重固定 A 的 {TARGET_W}；背景 F（主要，其他 195 投影依 A，含 block27 的 QDQ）與 U（其他 FP32）；九種 activation 格式（input／output）。',
        '- 候選：z26 = Q_y(linear(Q_x(x26), W26q))（batch 8／尾批 2），h27 = r26 + z26，經實際 block27（同批捕捉的 kwargs、use_cache=False）、final norm、head。',
        '- g_u 只經 head（u_0 為 leaf，每窗口一次 backward）；S_u = g_u·(u_f − u_0)；Khat256／P256 沿用前輪實作與精度，k = 256 固定。',
        '- C 參考：D、K2_actual、K2_affine、KL、D_64、S_logit,64、浮點殘差；P0／Rest 只指最終 CE 位置。',
        '- 凍結（C）：primary argmin P256、quadratic_reference argmin(S_u+K2_actual)、measured_C_reference argmin D_C、first_order argmin S_u、f16。',
        f'- N：新抽 {N_ART} 篇 × {WPA}，只量真值；seed {SEEDS}；排除 {len(used)} 篇已用文章。主要：F 背景 primary 相對 f16，mean < 0 且 CI 上界 < 0 才算確認改善。',
        f'- 成本：F、C、{len(CANDS)} 候選，四方法都含每候選 target、block27、norm；5 warmup + 20 次，median／p90；冷時間為重建值。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; used {len(used)}, eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# environment (explicitly parameterized; the shared tools are not modified)
# --------------------------------------------------------------------------- #
class Env:
    def __init__(self, work):
        from run_qwen_down_fp16_protection import replay
        self.rp, self.init_s = replay(work)
        rp = self.rp
        self.torch = rp.torch
        hf = rp.ad.model
        self.hf = hf
        self.m = rp.mod(TARGET)
        self.rng = rp.ranges[TARGET]
        self.res_mod = rp.mods[f'model.layers.{LAYER}.post_attention_layernorm']
        self.block = hf.model.layers[LAYER + 1]
        self.nm, self.lm = hf.model.norm, hf.lm_head
        self.W = self.lm.weight.data
        self.slot = ('f16', 'f16')
        self.cap = None

        def res_pre(mod, i):
            if self.cap is not None:
                self.cap['r'].append(i[0].detach().clone())

        def pre(mod, i):
            if self.cap is not None:
                self.cap['x'].append(i[0].detach().clone())
            return None if self.slot[0] == 'fp32' else (rp.qact(i[0], self.slot[0], self.rng['x']),)

        def post(mod, i, o):
            return None if self.slot[1] == 'fp32' else rp.qact(o, self.slot[1], self.rng['y'])

        def block_pre(mod, args, kwargs):
            if self.cap is not None:
                self.cap['h27'].append(args[0].detach().clone())
                self.cap['kw'].append(dict(kwargs))

        def norm_post(mod, i, o):
            if self.cap is not None:
                self.cap['u'].append(o.detach().clone())
        self.res_mod.register_forward_pre_hook(res_pre)
        self.m.register_forward_pre_hook(pre)
        self.m.register_forward_hook(post)
        self.block.register_forward_pre_hook(block_pre, with_kwargs=True)
        self.nm.register_forward_hook(norm_post)

    def set_background(self, cfg):
        self.rp.set(cfg)
        self.Wq = self.m.weight.data
        self.cfg = cfg

    def config_checks(self):
        """weights / activation map equal the configuration; hook counts."""
        rp, torch = self.rp, self.torch
        ok_w = all(torch.equal(rp.mod(n).weight.data, rp.cache[(n, (self.cfg[n]['w'] or 'fp32'))].to(rp.device)) for n in rp.param)
        ok_a = all(rp.act[n] == self.cfg[n]['a'] for n in rp.param)
        tw = bool(torch.equal(self.Wq, rp.cache[(TARGET, TARGET_W)].to(self.Wq.device)) and not torch.equal(self.Wq, rp.orig[TARGET].to(self.Wq.device)))
        hooks = Counter(f'{len(rp.mod(n)._forward_pre_hooks)},{len(rp.mod(n)._forward_hooks)}' for n in rp.param if n != TARGET)
        return {'weights_equal_config': bool(ok_w), 'activation_map_equal_config': bool(ok_a), 'target_weight_is_cached_int6': tw,
                'target_activation_in_lmreplay': rp.act[TARGET], 'hooks_other_projections': dict(hooks),
                'hooks_target': [len(self.m._forward_pre_hooks), len(self.m._forward_hooks)],
                'hooks_block27_pre': len(self.block._forward_pre_hooks), 'hooks_final_norm': [len(self.nm._forward_pre_hooks), len(self.nm._forward_hooks)],
                'hooks_lm_head': [len(self.lm._forward_pre_hooks), len(self.lm._forward_hooks)]}

    def head_checks(self):
        import inspect
        import transformers
        lm = self.lm
        src = inspect.getsource(type(self.hf).forward)
        c = {'lm_head_type': type(lm).__name__, 'lm_head_bias': lm.bias is not None,
             'lm_head_tied_to_embedding': lm.weight.data_ptr() == self.hf.model.embed_tokens.weight.data_ptr(),
             'lm_head_searched': any(v.removesuffix('.weight') == 'lm_head' for v in self.rp.param.values()),
             'forward_mentions': [w for w in ('softcap', 'temperature', 'logit_scale', 'final_logit') if w in src],
             'config_final_logit_softcapping': getattr(self.hf.config, 'final_logit_softcapping', None), 'norm_type': type(self.nm).__name__,
             'norm_eps': float(self.nm.variance_epsilon), 'transformers': transformers.__version__}
        c['passed'] = bool(c['lm_head_type'] == 'Linear' and not c['lm_head_bias'] and not c['lm_head_searched'] and not c['forward_mentions']
                           and c['config_final_logit_softcapping'] is None and c['norm_type'] == 'Qwen3RMSNorm' and c['norm_eps'] == 1e-6)
        return c

    def full(self, toks, fmt, capture=False):
        """complete forward with the target at fmt (LMReplay's batches of 8): per-position CE [windows, 127]; optional captures."""
        torch = self.torch
        self.slot = (fmt, fmt)
        self.cap = {'x': [], 'r': [], 'h27': [], 'kw': [], 'u': []} if capture else None
        x = torch.from_numpy(toks).to(self.rp.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                z = self.rp.ad(x[i:i + BATCH])
                out.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), x[i:i + BATCH, 1:].reshape(-1),
                                                             reduction='none').view(z.shape[0], -1).double())
        cap, self.cap, self.slot = self.cap, None, ('f16', 'f16')
        ce = torch.cat(out)
        if capture:
            return ce, {'x': torch.cat(cap['x']), 'r': torch.cat(cap['r']), 'h27': torch.cat(cap['h27']), 'kw': cap['kw'], 'u': torch.cat(cap['u'])}
        return ce

    def zf(self, x, f):
        q = self.rp.qact
        out = []
        for i in range(0, len(x), BATCH):
            xb = x[i:i + BATCH]
            xq = xb if f == 'fp32' else q(xb, f, self.rng['x'])
            y = self.torch.nn.functional.linear(xq, self.Wq)
            out.append(y if f == 'fp32' else q(y, f, self.rng['y']))
        return self.torch.cat(out)

    def suffix(self, r, z, kws, keep_h27=False):
        """h27 = r + z -> actual block 27 (captured kwargs of the same batch) -> final norm; batches of BATCH."""
        torch = self.torch
        us, hs = [], []
        with torch.no_grad():
            for bi, i in enumerate(range(0, len(r), BATCH)):
                h27 = r[i:i + BATCH] + z[i:i + BATCH]
                if keep_h27:
                    hs.append(h27)
                us.append(self.nm(self.block(h27, **kws[bi])))
        return (torch.cat(us), torch.cat(hs)) if keep_h27 else torch.cat(us)

    def head_ce(self, u, ids):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(u), BATCH):
                lg = self.lm(u[i:i + BATCH])
                out.append(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[i:i + BATCH, 1:].reshape(-1),
                                                             reduction='none').view(lg.shape[0], -1).double())
        return torch.cat(out)

    def grad_pass(self, u0, ids, on_logits=None):
        """g_u = d(summed FP32 CE)/du at the leaf u_0, head only, microbatch 1; on_logits(j, FP32 logits [127, V])."""
        torch = self.torch
        F_ = torch.nn.functional
        gu = torch.empty_like(u0)
        for j in range(len(u0)):
            uj = u0[j:j + 1].detach().clone().requires_grad_(True)
            lg = self.lm(uj)
            loss = F_.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            gu[j:j + 1], = torch.autograd.grad(loss, uj)
            if on_logits is not None:
                with torch.no_grad():
                    on_logits(j, lg[0, :-1].detach())
        return gu


def gsum(a, G):
    return {'P0': a[:, 0], 'Rest': a[:, 1:].sum(1), 'All': a.sum(1)}[G]


def tensor_excess(a, b):
    d = (a - b).abs()
    return float((d - TOL['replay_vs_full_tensor_abs'] - TOL['replay_vs_full_tensor_rel'] * b.abs()).max()), float(d.max())


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def head_terms(env, U, gu, L, ids, ctl, tm, b):
    """per-position head-side terms for every format (C): truth / KL / K2_actual from batch-8 logits, Khat256 / K2_affine from the
    microbatch-1 p (previous rounds' definitions)."""
    torch = env.torch
    F_ = torch.nn.functional
    W, lm = env.W, env.lm
    n = len(gu)
    res = {f: defaultdict(list) for f in FORMATS}
    u0 = U['f16']
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            sl = slice(i, i + BATCH)
            y = ids[sl, 1:]
            nb = y.shape[0]
            l0 = lm(u0[sl])
            V = l0.shape[-1]
            ce0 = F_.cross_entropy(l0[:, :-1].reshape(-1, V).float(), y.reshape(-1), reduction='none').view(nb, -1).double()
            lp0 = [torch.log_softmax(l0[j, :-1].double(), -1) for j in range(nb)]
            for f in FORMATS:
                lf = lm(U[f][sl])
                cef = F_.cross_entropy(lf[:, :-1].reshape(-1, V).float(), y.reshape(-1), reduction='none').view(nb, -1).double()
                Su = (gu[sl].double() * (U[f][sl].double() - u0[sl].double())).sum(-1)
                ctl['pos127_S_abs'] = max(ctl['pos127_S_abs'], float(Su[:, -1].abs().max()))
                rows = {k: torch.empty(nb, TP, dtype=torch.float64, device=gu.device) for k in ('D64', 'S_logit64', 'KL', 'K2_actual')}
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
                    rows['K2_actual'][j] = 0.5 * ((p0 * dl * dl).sum(-1) - m1 * m1)
                D = cef - ce0
                Su_ = Su[:, :-1]
                t = {'D': D, 'S_u': Su_, 'KL': rows['KL'], 'K2_actual': rows['K2_actual'], 'D64': rows['D64'], 'S_logit64': rows['S_logit64'],
                     'eps_head': rows['S_logit64'] - Su_, 'eps_CE': D - rows['D64'], 'CE': cef}
                ctl['identity_D64_abs'] = max(ctl['identity_D64_abs'], float((rows['D64'] - rows['S_logit64'] - rows['KL']).abs().max()))
                ctl['kl_min'] = min(ctl['kl_min'], float(rows['KL'].min()))
                for k_, v_ in t.items():
                    res[f][k_].append(v_.cpu().numpy())
                del lf, cef, rows
            del l0, lp0
    tm[f'{b}|truth_KL_K2actual_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        for j in range(n):
            p = torch.softmax(L[j].double(), -1)
            sel = tk.topk_sel(p, K)
            ctl['topk_count_ok'] = bool(ctl.get('topk_count_ok', True) and bool((sel.sum(-1) == K).all()))
            pr = tk.prep(p, W, K, sel)
            ctl['q_min'] = min(ctl.get('q_min', 1.0), float(pr['q'].min()))
            du = torch.stack([U[f][j, :-1] - u0[j, :-1] for f in FORMATS])
            kc = tk.khat(pr, du)
            for fi, f in enumerate(FORMATS):
                ft = tk.full_terms(p, (du[fi] @ W.T).double(), {K: sel})
                res[f]['Khat256'].append(kc[fi:fi + 1].cpu().numpy())
                res[f]['K2_aff'].append(ft['K2_aff'][None].cpu().numpy())
                res[f]['Khat256_fullA'].append(ft[f'Khat{K}_fullA'][None].cpu().numpy())
                res[f]['tailvar256'].append(ft[f'tailvar{K}'][None].cpu().numpy())
            if b == REF_BG and j in REF_WINDOWS:
                t1 = time.perf_counter()
                ref_window(env, p, sel, pr, kc, U, j, gu, ids, ctl)
                tm['float64_reference_s'] += time.perf_counter() - t1
            del p, pr, du, kc
    tm[f'{b}|topk_K2affine_s'] = time.perf_counter() - t0
    return {f: {k: np.concatenate(v) for k, v in d.items()} for f, d in res.items()}


def ref_window(env, p, sel, pr, kc, U, j, gu, ids, ctl):
    torch = env.torch
    ref = ctl['ref']
    if not hasattr(env, 'W64'):
        env.W64 = env.W.double()
    W64 = env.W64
    pr64 = tk.prep(p, W64, K, sel)
    du64 = torch.stack([U[f][j, :-1].double() - U['f16'][j, :-1].double() for f in FORMATS])
    k64 = tk.khat(pr64, du64)
    exc, worst = ref.get('fp32_vs_float64_excess', -np.inf), ref.get('fp32_vs_float64_max_abs', 0.0)
    for fi, f in enumerate(FORMATS):
        K2a64 = tk.full_terms(p, du64[fi] @ W64.T, {})['K2_aff']
        K2a32 = tk.full_terms(p, ((U[f][j, :-1] - U['f16'][j, :-1]) @ env.W.T).double(), {})['K2_aff']
        for a_, r_ in ((kc[fi], k64[fi]), (K2a32, K2a64)):
            d = (a_ - r_).abs()
            exc = max(exc, float((d - TOL['fp32_vs_float64_abs'] - TOL['fp32_vs_float64_rel'] * r_.abs()).max()))
            worst = max(worst, float(d.max()))
    ref['fp32_vs_float64_excess'], ref['fp32_vs_float64_max_abs'] = exc, worst
    wbar64 = p @ W64
    implied = (pr['WT'].double() * pr['pT'][..., None]).sum(1) + pr['w_tail'].double()
    ref['wbar_rel'] = max(ref.get('wbar_rel', 0.0), float((implied - wbar64).abs().max() / wbar64.abs().max()))
    y = ids[j, 1:]
    ana = wbar64 - W64[y]                                         # W^T (p - onehot) in float64
    ref['g_u_vs_analytic_rel'] = max(ref.get('g_u_vs_analytic_rel', 0.0), float((gu[j, :-1].double() - ana).abs().max() / ana.abs().max()))
    ref['windows'], ref['background'] = list(REF_WINDOWS), REF_BG


def calib(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    t0 = time.perf_counter()
    syn = tk.synthetic()
    tm['synthetic_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    env = Env(out / 'work')
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    ctl = defaultdict(float)
    ctl['kl_min'] = 0.0
    ctl['synthetic'], ctl['head'], ctl['ref'], ctl['config'] = syn, env.head_checks(), {}, {}
    if not ctl['head']['passed'] or not syn['passed']:
        write(out / 'controls_calib.json', ctl)
        raise SystemExit(f'head or synthetic control failed: {ctl["head"]} {syn}')
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    prev = np.load(PREV / 'C_terms.npz')
    res, base_ce = {}, {}
    for b in BGS:
        t0 = time.perf_counter()
        env.set_background(background(A, b))
        torch.cuda.synchronize()
        tm[f'{b}|set_background_s'] = time.perf_counter() - t0
        ctl['config'][b] = env.config_checks()
        t0 = time.perf_counter()
        ce0, cap = env.full(cal, 'f16', capture=True)
        torch.cuda.synchronize()
        tm[f'{b}|prefix_s'] = time.perf_counter() - t0
        base_ce[b] = ce0.cpu().numpy()
        kw = cap['kw']
        ctl['kwargs'] = {k: (type(v).__name__ if not hasattr(v, 'shape') else list(v.shape)) if not isinstance(v, tuple) else
                         [list(t.shape) for t in v] for k, v in kw[0].items()}
        ctl['kwargs_cache_free'] = bool(all(k_.get('past_key_values') is None and not k_.get('use_cache', False) for k_ in kw))
        ctl['kwargs_batches'] = [int(k_['position_embeddings'][0].shape[0]) for k_ in kw]
        t0 = time.perf_counter()
        z = {f: env.zf(cap['x'], f) for f in FORMATS}
        U, H27 = {}, {}
        for f in FORMATS:
            U[f], H27[f] = env.suffix(cap['r'], z[f], kw, keep_h27=True)
        torch.cuda.synchronize()
        tm[f'{b}|candidates_suffix_s'] = time.perf_counter() - t0
        # replay vs complete forward, every format
        t0 = time.perf_counter()
        for f in FORMATS:
            cef, capf = env.full(cal, f, capture=True)
            for nm_, a_, r_ in (('h27', H27[f], capf['h27']), ('u', U[f], capf['u'])):
                e, d = tensor_excess(a_, r_)
                ctl[f'replay_{nm_}_excess'] = max(ctl.get(f'replay_{nm_}_excess', -np.inf), e)
                ctl[f'replay_{nm_}_max_abs'] = max(ctl[f'replay_{nm_}_max_abs'], d)
            ctl['replay_ce_max_abs'] = max(ctl['replay_ce_max_abs'], float((env.head_ce(U[f], ids) - cef).abs().max()))
            ctl['r_plus_z_vs_block27_input'] = ctl['replay_h27_max_abs']
        tm['replay_vs_full_control_s'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        L = {}
        gu = env.grad_pass(U['f16'], ids, on_logits=lambda j, l: L.__setitem__(j, l.clone()))
        torch.cuda.synchronize()
        tm[f'{b}|grad_pass_s'] = time.perf_counter() - t0
        ctl['pos127_g_abs'] = max(ctl['pos127_g_abs'], float(gu[:, -1].abs().max()))
        res[b] = head_terms(env, U, gu, L, ids, ctl, tm, b)
        log(f'{b}: ' + ', '.join(f'{f} S_u {res[b][f]["S_u"].sum() / (CAL * TP):+.4f} Khat {res[b][f]["Khat256"].sum() / (CAL * TP):+.4f} '
                                 f'D {res[b][f]["D"].sum() / (CAL * TP):+.4f}' for f in ('p8e1', 'int8')))
        del cap, z, U, H27, L, gu
    # switch reproduction (F -> U -> F) and the previous round's F / f16
    env.set_background(background(A, 'F'))
    ctl['switch_reproduce_abs'] = float(np.abs(env.full(cal, 'f16').cpu().numpy() - base_ce['F']).max())
    ctl['config']['F_again'] = env.config_checks()
    ctl['reproduce_prev_F_f16_ce_abs'] = float(np.abs(base_ce['F'] - prev['F__f16__CE']).max())
    # term controls
    raw, exc = defaultdict(float), defaultdict(lambda: -np.inf)
    psum = f16 = 0.0
    for b in BGS:
        for f in FORMATS:
            R = res[b][f]
            K2a = R['K2_aff']
            exc['nested'] = max(exc['nested'], float(np.max(R['Khat256'] - K2a - TOL['nested_abs'] - TOL['nested_rel'] * np.abs(K2a))))
            raw['Khat256_minus_K2aff_max'] = max(raw['Khat256_minus_K2aff_max'], float(np.max(R['Khat256'] - K2a)))
            exc['tail_identity'] = max(exc['tail_identity'], float(np.max(np.abs(K2a - R['Khat256_fullA'] - R['tailvar256'])
                                                                          - TOL['tail_identity_abs'] - TOL['tail_identity_rel'] * np.abs(K2a))))
            raw['tail_identity_max_abs'] = max(raw['tail_identity_max_abs'], float(np.max(np.abs(K2a - R['Khat256_fullA'] - R['tailvar256']))))
            exc['cheap_vs_fullA'] = max(exc['cheap_vs_fullA'], float(np.max(np.abs(R['Khat256'] - R['Khat256_fullA']) - TOL['cheap_vs_fullA_abs']
                                                                            - TOL['cheap_vs_fullA_rel'] * np.abs(R['Khat256_fullA']))))
            raw['cheap_vs_fullA_max_abs'] = max(raw['cheap_vs_fullA_max_abs'], float(np.max(np.abs(R['Khat256'] - R['Khat256_fullA']))))
            for t in ('D', 'S_u', 'Khat256', 'K2_actual', 'K2_aff', 'KL'):
                a = R[t]
                psum = max(psum, abs(a[:, 0].sum() + a[:, 1:].sum() - a.sum()))
            if f == 'f16':
                f16 = max(f16, max(float(np.abs(R[t]).max()) for t in ('D', 'S_u', 'Khat256', 'K2_actual', 'K2_aff', 'KL')))
    ctl['excess'], ctl['raw_max'], ctl['position_sum_abs'], ctl['f16_abs'] = dict(exc), dict(raw), psum, f16
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    cfg_ok = all(c['weights_equal_config'] and c['activation_map_equal_config'] and c['target_weight_is_cached_int6']
                 and c['target_activation_in_lmreplay'] == 'fp32' and c['hooks_target'] == [2, 2] and c['hooks_block27_pre'] == 1
                 and c['hooks_final_norm'] == [0, 1] and c['hooks_lm_head'] == [0, 0] and c['hooks_other_projections'] == {'1,1': 195}
                 for c in ctl['config'].values())
    ref = ctl['ref']
    ctl['passed'] = bool(ctl['head']['passed'] and syn['passed'] and cfg_ok and ctl['kwargs_cache_free']
                         and ctl['replay_h27_excess'] <= 0 and ctl['replay_u_excess'] <= 0
                         and ctl['replay_ce_max_abs'] <= TOL['replay_vs_full_ce_abs_per_position']
                         and ctl['reproduce_prev_F_f16_ce_abs'] <= TOL['reproduce_prev_F_f16_ce_abs_per_position']
                         and ctl['switch_reproduce_abs'] <= TOL['switch_reproduce_abs_per_position'] and ctl['pos127_g_abs'] == 0
                         and ctl['pos127_S_abs'] == 0 and f16 == 0 and ctl['topk_count_ok'] and all(v <= 0 for v in exc.values())
                         and psum <= TOL['position_sum_abs_summed'] and ref.get('fp32_vs_float64_excess', 1) <= 0
                         and ref.get('wbar_rel', 1) <= TOL['wbar_rel'] and ref.get('g_u_vs_analytic_rel', 1) <= TOL['g_u_vs_analytic_float64_rel']
                         and ctl['identity_D64_abs'] <= TOL['identity_D64_abs_per_position'] and ctl['kl_min'] >= TOL['kl_min_per_position']
                         and ctl['tf32'] == [False, False] and ctl['deterministic'])
    np.savez(out / 'C_terms.npz', **{f'{b}__{f}__{k}': v for b in BGS for f in FORMATS for k, v in res[b][f].items()})
    summ = summarize_C(res)
    write(out / 'controls_calib.json', ctl)
    write(out / 'frozen_predictions.json', summ['frozen'])
    with open(out / 'scores_C.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summ['rows'][0]))
        wr.writeheader()
        wr.writerows(summ['rows'])
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib', tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; picks {summ["frozen"]["selectors"]}; replay h27 {ctl["replay_h27_max_abs"]:.1e} u '
        f'{ctl["replay_u_max_abs"]:.1e} ce {ctl["replay_ce_max_abs"]:.1e}; prev {ctl["reproduce_prev_F_f16_ce_abs"]:.1e}; excess {dict(exc)}; ref {ref}')
    if not ctl['passed']:
        raise SystemExit(f'calibration controls failed: { {k: v for k, v in ctl.items() if k not in ("synthetic", "head")} }')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_calib.json'),
                                'predictions_sha256': sha(out / 'frozen_predictions.json'), 'scores_sha256': sha(out / 'scores_C.csv'),
                                'terms_sha256': sha(out / 'C_terms.npz'), 'A_sha256': sha(ENDP / 'configs/A_auto.json'),
                                'code_sha256': {f: sha(HERE / f) for f in CODE}, 'inputs_sha256': read(out / 'plan.json')['inputs']})


def summarize_C(res):
    n = CAL * TP
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def tot(b, f, t, G='All'):
        if t == 'P256':
            return tot(b, f, 'S_u', G) + tot(b, f, 'Khat256', G)
        if t == 'Su+K2actual':
            return tot(b, f, 'S_u', G) + tot(b, f, 'K2_actual', G)
        if t == 'D_C':
            return tot(b, f, 'D', G)
        return float(gsum(res[b][f][t], G).sum()) / n
    cols = ('S_u', 'Khat256', 'K2_aff', 'K2_actual', 'KL', 'P256', 'Su+K2actual', 'D', 'tailvar256', 'eps_head', 'eps_CE')
    rows = []
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                r = {'background': b, 'format': f, 'G': G}
                r.update({c: tot(b, f, c, G) for c in cols})
                r['K2_actual_minus_K2_affine'] = r['K2_actual'] - r['K2_aff']
                r['K2_affine_minus_Khat256'] = r['K2_aff'] - r['Khat256']
                r['KL_minus_K2_actual'] = r['KL'] - r['K2_actual']
                r['D_minus_P256'] = r['D'] - r['P256']
                rows.append(r)
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    nonf16 = [f for f in FORMATS if f != 'f16']
    scores = ('P256', 'Su+K2actual', 'D_C', 'S_u')
    rank, sel = {}, {}
    for b in BGS:
        rank[b] = {'truth_order': sorted(SELECT_SET, key=lambda f: (tot(b, f, 'D'), SELECT_SET.index(f)))}
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
        sel[b] = {'primary_P256': rank[b]['P256']['pick'], 'quadratic_reference': rank[b]['Su+K2actual']['pick'],
                  'measured_C_reference': rank[b]['D_C']['pick'], 'first_order': rank[b]['S_u']['pick'], 'original_config_control': 'f16'}
    frozen = {'selectors': sel, 'ranking': rank, 'rule': 'argmin over the selection set, exact ties: f16 first then the set order',
              'scores_per_token_All': {b: {s: {f: tot(b, f, s) for f in FORMATS} for s in scores + ('Khat256', 'K2_actual', 'KL', 'D')} for b in BGS},
              'note': 'C only; frozen before any N data'}
    return {'rows': rows, 'frozen': frozen}


# --------------------------------------------------------------------------- #
# timing (C, F, after the freeze); every method replays block 27 per candidate
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
    torch = env.torch
    F_ = torch.nn.functional
    sync = torch.cuda.synchronize
    tm['replay_init_s'] = time.perf_counter() - t0
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    y = ids[:, 1:]
    t0 = time.perf_counter()
    env.set_background(background(A, BENCH['background']))
    sync()
    tm['set_background_cold_s'] = time.perf_counter() - t0
    W, lm = env.W, env.lm
    V = W.shape[0]

    def prefix():
        ce0, cap = env.full(cal, 'f16', capture=True)
        return {'ce0': ce0.float(), 'x': cap['x'], 'r': cap['r'], 'kw': cap['kw'], 'u0': cap['u']}
    t0 = time.perf_counter()
    P = prefix()
    sync()
    tm['prefix_cold_s'] = time.perf_counter() - t0

    def base_su(extra=None):
        return {'gu': env.grad_pass(P['u0'], ids, on_logits=extra)}

    def base_khat():
        cache = {}

        def on(j, l):
            pr = tk.prep(torch.softmax(l.double(), -1), W, K)
            del pr['sel']
            cache[j] = pr
        st = base_su(on)
        st['pr'] = [cache[j] for j in range(CAL)]
        return st

    def base_k2():
        cache = {}
        st = base_su(lambda j, l: cache.__setitem__(j, torch.softmax(l.double(), -1).float()))
        st['p32'] = [cache[j] for j in range(CAL)]
        return st

    def base_ce():
        return {}                                                   # reuses the prefix CE

    def cand_u():
        with torch.no_grad():
            return torch.stack([env.suffix(P['r'], env.zf(P['x'], f), P['kw']) for f in CANDS])

    def head_su(st, U):
        with torch.no_grad():
            return (st['gu'].double() * (U.double() - P['u0'].double())).sum(-1)

    def head_khat(st, U):
        S = head_su(st, U)
        with torch.no_grad():
            Kh = torch.stack([tk.khat(st['pr'][j], U[:, j, :-1] - P['u0'][j, :-1]) for j in range(CAL)], 1)
        return S, Kh

    def head_k2(st, U):
        S = head_su(st, U)
        o = []
        with torch.no_grad():
            for j in range(CAL):
                a = (U[:, j, :-1] - P['u0'][j, :-1]) @ W.T
                pj = st['p32'][j]
                m = (a * pj).sum(-1)
                o.append(0.5 * (((a - m[..., None]) ** 2) * pj).sum(-1))
                del a
        return S, torch.stack(o, 1)

    def head_ce(st, U):
        o = []
        with torch.no_grad():
            for ci in range(len(CANDS)):
                ce = []
                for i in range(0, CAL, BATCH):
                    lg = lm(U[ci, i:i + BATCH])
                    ce.append(F_.cross_entropy(lg[:, :-1].reshape(-1, V), y[i:i + BATCH].reshape(-1), reduction='none').view(lg.shape[0], -1))
                o.append(torch.cat(ce) - P['ce0'])
        return torch.stack(o)
    methods = {'suffix+Su': (base_su, head_su), 'suffix+Su+Khat256': (base_khat, head_khat), 'suffix+Su+K2full': (base_k2, head_k2),
               'suffix+directCE': (base_ce, head_ce)}
    cold, outs = {}, {}
    for name, (bf, hf_) in methods.items():
        sync()
        t0 = time.perf_counter()
        outs[name] = hf_(bf(), cand_u())
        sync()
        cold[name] = time.perf_counter() - t0
    tm['first_run_s'] = cold
    tm['cold_reconstructed_s'] = {k: tm['replay_init_s'] + tm['set_background_cold_s'] + tm['prefix_cold_s'] + v for k, v in cold.items()}

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
    tm['prefix_warm'] = timed(prefix)
    for name, (bf, hf_) in methods.items():
        torch.cuda.empty_cache()
        sync()
        static = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        warm[name] = {'baseline_plus_candidates': timed(lambda: hf_(bf(), cand_u())), 'baseline_only': timed(bf)}
        st = bf()
        warm[name]['candidates_hot_increment'] = timed(lambda: hf_(st, cand_u()))
        U = cand_u()
        warm[name]['head_part_only_given_u'] = timed(lambda: hf_(st, U))
        mem[name] = {'peak_bytes': int(torch.cuda.max_memory_allocated()), 'increment_over_static_bytes': int(torch.cuda.max_memory_allocated() - static)}
        del st, U
    # candidate path breakdown (common to all methods)
    Z = {f: env.zf(P['x'], f) for f in CANDS}
    H = {f: None for f in CANDS}

    def blocks():
        with torch.no_grad():
            for f in CANDS:
                for bi, i in enumerate(range(0, CAL, BATCH)):
                    H[f] = env.block(P['r'][i:i + BATCH] + Z[f][i:i + BATCH], **P['kw'][bi])
    Hf = {f: torch.cat([env.block(P['r'][i:i + BATCH] + Z[f][i:i + BATCH], **P['kw'][bi]) for bi, i in enumerate(range(0, CAL, BATCH))])
          for f in CANDS}
    breakdown = {'target_qdq_gemm': timed(lambda: [env.zf(P['x'], f) for f in CANDS]), 'block27': timed(blocks),
                 'final_norm': timed(lambda: [env.nm(Hf[f]) for f in CANDS]), 'full_candidate_u': timed(cand_u)}
    del Z, H, Hf
    Ls = {}
    env.grad_pass(P['u0'], ids, on_logits=lambda j, l: Ls.__setitem__(j, l.clone()))
    Pp = [torch.softmax(Ls[j].double(), -1) for j in range(CAL)]
    Ss = [tk.topk_sel(Pp[j], K) for j in range(CAL)]
    sub = {'head_grad_pass': timed(lambda: base_su()), 'softmax_float64': timed(lambda: [torch.softmax(Ls[j].double(), -1) for j in range(CAL)]),
           'topk_tie_mask': timed(lambda: [tk.topk_sel(Pp[j], K) for j in range(CAL)]),
           'index_pT_q': timed(lambda: [(Ss[j].nonzero(), Pp[j].masked_fill(Ss[j], 0.0).sum(-1)) for j in range(CAL)]),
           'w_tail_dense_gemm': timed(lambda: [Pp[j].masked_fill(Ss[j], 0.0).float() @ W for j in range(CAL)]),
           'row_gather_WT': timed(lambda: [W[Ss[j].nonzero()[:, 1].view(TP, K)] for j in range(CAL)])}
    del Ls, Pp, Ss
    fp = read(out / 'frozen_predictions.json')['scores_per_token_All'][BENCH['background']]
    C = np.load(out / 'C_terms.npz')
    n = CAL * TP
    bg = BENCH['background']
    cons = {'Su': max(abs(float(outs['suffix+Su'][ci].sum()) / n - fp['S_u'][f]) for ci, f in enumerate(CANDS)),
            'Khat256': max(abs(float(outs['suffix+Su+Khat256'][1][ci].sum()) / n - fp['Khat256'][f]) for ci, f in enumerate(CANDS)),
            'K2full_fp32_vs_K2affine_float64red': max(abs(float(outs['suffix+Su+K2full'][1][ci].double().sum()) / n
                                                          - float(C[f'{bg}__{f}__K2_aff'].sum()) / n) for ci, f in enumerate(CANDS)),
            'directCE_vs_D': max(abs(float(outs['suffix+directCE'][ci].double().sum()) / n - fp['D'][f]) for ci, f in enumerate(CANDS))}
    sizes = {'vocab': int(V), 'hidden': int(W.shape[1]), 'k': K, 'windows': CAL, 'target_positions': TP, 'candidates': len(CANDS),
             'WT_cache_bytes_all_windows': CAL * TP * K * W.shape[1] * 4, 'p32_cache_bytes_all_windows': CAL * TP * V * 4,
             'cache_strategy': 'all 10 C windows cached at once (no chunking); a larger workload would chunk windows'}
    tm.update({'warm': warm, 'candidate_breakdown': breakdown, 'baseline_substeps_Khat256': sub, 'gpu_memory': mem,
               'consistency_per_token_abs': cons, 'sizes': sizes, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'device': torch.cuda.get_device_name(0)})
    write(out / 'bench.json', tm)
    ca.log_stage(out, 'bench', time.perf_counter() - t_all)
    log('bench: ' + ', '.join(f'{k} base+cand {v["baseline_plus_candidates"]["median_s"]:.3f}s / hot {v["candidates_hot_increment"]["median_s"]:.3f}s'
                              for k, v in warm.items()) + f'; prefix {tm["prefix_warm"]["median_s"]:.3f}s')


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
        env.set_background(background(A, b))
        for c0 in range(0, len(tok), CHUNK):
            t = tok[c0:c0 + CHUNK]
            ids = torch.from_numpy(t).cuda()
            ce0, cap = env.full(t, 'f16', capture=True)
            for f in FORMATS:
                ce = env.head_ce(env.suffix(cap['r'], env.zf(cap['x'], f), cap['kw']), ids)
                res[f'{b}__{f}'][c0:c0 + len(t)] = ce.cpu().numpy()
                if f == 'f16':
                    dev = max(dev, float((ce - ce0).abs().max()))
            del cap
        log(f'{b} done')
    np.savez(out / 'eval_ce_pos.npz', **res)
    write(out / 'controls_eval.json', {'f16_replay_vs_full_ce_abs_per_position': dev,
                                       'passed': bool(dev <= TOL['replay_vs_full_ce_abs_per_position'])})
    ca.log_stage(out, 'eval', time.perf_counter() - t0, {'windows': len(tok)})
    if dev > TOL['replay_vs_full_ce_abs_per_position']:
        raise SystemExit('replay differs from the full forward on new windows')


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
    roles = ('primary_P256', 'quadratic_reference', 'measured_C_reference', 'first_order', 'original_config_control')
    picks = {b: {r: {'format': sel[b][r], 'N_D_All': st(D[(b, sel[b][r], 'All')]), 'N_D_P0': st(D[(b, sel[b][r], 'P0')]),
                     'N_D_Rest': st(D[(b, sel[b][r], 'Rest')])} for r in roles} for b in BGS}
    for b in BGS:
        for r in roles:
            picks[b][r]['severe_harm'] = severe(picks[b][r]['N_D_All'])
    pf = sel['F']['primary_P256']
    s_prim = st(D[('F', pf, 'All')])
    primary = {'pick': pf, 'N_D_vs_f16': s_prim, 'confirmed_improvement': bool(pf != 'f16' and s_prim['mean'] < 0 and s_prim['ci95'][1] < 0),
               'note': 'f16 pick: no added quality gain' if pf == 'f16' else ''}
    secondary = {}
    for b in BGS:
        pb = sel[b]['primary_P256']
        for r in ('first_order', 'quadratic_reference', 'measured_C_reference'):
            secondary[f'{b}|primary-{r}'] = {'formats': [pb, sel[b][r]], 'N': st(D[(b, pb, 'All')] - D[(b, sel[b][r], 'All')])}
    nonf16 = [f for f in FORMATS if f != 'f16']
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    rank = {}
    for b in BGS:
        rb = {}
        for s_ in ('P256', 'S_u', 'Su+K2actual', 'D_C'):
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
    write(out / 'stats.json', {'units': 'nat/token contributions relative to f16 of the same background; article bootstrap', 'primary': primary,
                               'picks': picks, 'secondary': secondary, 'rankings': rank})
    rows = []
    for b in BGS:
        for f in FORMATS:
            for G in POS:
                s = st(D[(b, f, G)])
                rows.append({'background': b, 'format': f, 'G': G, 'mean': s['mean'], 'ci_lo': s['ci95'][0], 'ci_hi': s['ci95'][1],
                             'median': s['median'], 'articles_pos': s['articles_pos'], 'articles_neg': s['articles_neg']})
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
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_l26_suffix_bridge.py')}) + '\n')


if __name__ == '__main__':
    main()
