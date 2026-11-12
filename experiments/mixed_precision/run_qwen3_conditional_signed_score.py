#!/usr/bin/env python3
"""Qwen3-0.6B L27 Down: a conditional, signed first-order score. In a background b (other projections U = FP32, F = per the
original BC knee A, W = only the L0-26 Down weights per A), the target keeps its INT8 weight Wq; for a candidate activation
format f the full post-QDQ output is z_b,f = Q_y,f(linear(Q_x,f(x_b), Wq)) from the raw (pre-input-QDQ) Down input x_b, and
  S_b,G(f) = sum_G g_b . (z_b,f - z_b,f16),   g_b = dL_b/dz at z_b,f16,  L_b(z) = summed CE(LM_head(final_norm(r_b + z)))
(r_b = the L27 FFN residual; only the final norm and the head follow, so no STE). One baseline gradient per background and
calibration window, reused for every candidate; G = position 0 (P0), positions 1..127 (Rest), All. Scores are frozen on the
original 10 calibration windows; 128 new articles x 2 windows only give the truth D_b,G(f) = L_b(z_b,0 + M_G dz) - L_b(z_b,0).
GPU (LMReplay QDQ); no change to the formal BC, search, configs, quantizers or native pipeline.

Stages (one new --out-dir): plan, calib, audit, data, eval, analyze. GPU stages (calib, eval):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
TARGET, ENDP, NATIVE = ca.TARGET, ca.ENDP, ca.NATIVE
SRC = HERE / 'runs/qwen3_l27_down_source_20261026'
BGS = ('U', 'F', 'W')
FORMATS = ('f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16', 'fp32')
SELECT_SET = ('f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16')      # a bits <= 16; ties: f16, then this order
BITS = {'f16': 16, 'bf16': 16, 'fp32': 32, 'p8e1': 8, 'p6e1': 6, 'p4e1': 4, 'int8': 8, 'int6': 6, 'int4': 4}
POS = ('P0', 'Rest', 'All')
ALPHAS = tuple(s * 2.0 ** -k for k in range(7) for s in (1.0, -1.0))           # +-1 .. +-1/64
SEQ, TP, CAL, N_ART, WPA, BOOT, BATCH, CHUNK = 128, 127, 10, 128, 2, 20000, 8, 32
TARGETS_CAL = CAL * TP
SEEDS = {'order': 26102661, 'start': 26102662, 'bootstrap': 26102663}
TOL = {'closure_vs_full_forward_abs_summed_ce': 1e-6, 'reproduce_down_source_abs_per_position': 1e-6, 'position_sum_rel': 1e-12,
       'direction_alpha': 1 / 64, 'direction_rel': 0.05, 'direction_abs_summed_ce': 1e-3, 'pos127_abs': 0.0}
CODE = ['run_qwen3_conditional_signed_score.py', 'run_qwen3_l27_down_source.py', 'run_qwen3_l27_context_followup.py'] + ca.CODE
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


def background(A, b):
    cfg = {}
    for n in A:
        l, p = ca.parse(n)
        if n == TARGET:
            cfg[n] = {'w': A[n]['w'], 'a': 'fp32'}                     # the target's activation is applied by our hooks
        elif b == 'F':
            cfg[n] = dict(A[n])
        elif b == 'W' and p == 'down' and l <= 26:
            cfg[n] = {'w': A[n]['w'], 'a': 'fp32'}
        else:
            cfg[n] = {'w': 'fp32', 'a': 'fp32'}
    return cfg


def inputs():
    return ca.inputs() + [SRC / 'gpu_ce_pos.npz', SRC / 'native_ce_pos.npz', SRC / 'tokens.npy', SRC / 'report.md', SRC / 'gpu_controls.json',
                          SRC / 'native_controls.json', SRC / 'work/native_tokens.npy', ENDP / 'table_A.json',
                          HERE / 'reports/qwen3_conditional_signed_score_prompt_20261026.md']


def plan(out, smoke):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or A[TARGET] != {'w': 'int8', 'a': 'f16'}:
        raise SystemExit('A is not the original knee')
    sizes = {b: {'quantized_weights': sum(background(A, b)[n]['w'] not in (None, 'fp32') for n in A if n != TARGET),
                 'quantized_activations': sum(background(A, b)[n]['a'] != 'fp32' for n in A if n != TARGET)} for b in BGS}
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {'time': now(), 'spec': 'reports/qwen3_conditional_signed_score_prompt_20261026.md and the request of 2026-10-26',
          'scope': 'Qwen3-0.6B L27 Down only; target weight = A\'s INT8 (per-channel, as LMReplay) in every background; activation candidates '
                   'on input x and output y; GPU LMReplay QDQ (native INT ranges / qparams on the device, Universal posit LUT, f16 / bf16 casts)',
          'backgrounds': {'U': 'other 195 projections FP32', 'F': 'other 195 per A (w and a)', 'W': 'only the L0-26 down_proj weights per A '
                          '(their activations and all other projections FP32; dn_w of the down_source round)'}, 'background_sizes': sizes,
          'candidates': {'formats': list(FORMATS), 'baseline': 'f16 (the zero direction)', 'note': 'fp32 activation is not zero relative to f16 '
                         'and keeps the INT8 target weight'},
          'score': {'z': 'Q_y,f(linear(Q_x,f(x_b), Wq)) from the raw Down input captured before the input QDQ, Wq the module\'s INT8 weight',
                    'g': 'autograd of the summed CE (127 targets per window) through final norm and LM head only, at the f16 output, microbatch 1',
                    'S': 'sum g . (z_f - z_f16) over P0 / Rest / All (signed, never squared or clipped), / 1270 for display',
                    'cross_pairs': '<g_U, dz_U>, <g_U, dz_F>, <g_F, dz_U>, <g_F, dz_F> (mechanism only)'},
          'calibration_truth': 'D_b,G(f) = L_b(z0 + M_G dz) - L_b(z0) on the 10 calibration windows; direction check O(a)/a vs S for a in '
                               f'{list(ALPHAS)} (post-QDQ interpolation), converged if |O(1/64)/a - S| <= {TOL["direction_rel"]}|S| + '
                               f'{TOL["direction_abs_summed_ce"]} (summed CE), else undetermined; changed fraction and rounding reported',
          'selector': f'per background argmin S_All over {list(SELECT_SET)} (a bits <= 16; exact ties: f16, then that order); margins reported; '
                      'the original BC (table_A C of the target, background-invariant raw) selection on the same set reported beside it',
          'new_articles': {'n_articles': N_ART, 'windows_per_article': WPA, 'use': 'truth only (no new gradient, score, range or prediction '
                           'update)', 'rule': 'run_qwen_prefix_rank_shift.data', 'excluded_previous_wt103': len(used),
                           'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
          'statistics': {'unit': 'article (2 windows, 254 targets)', 'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}), '
                         'one index set; 95% CI; two-sided p = 2 min tail',
                         'primary_family_holm': ['p8e1 F-U P0', 'p8e1 F-U Rest', 'p8e1 F-U All', 'p8e1 W-U P0', 'p8e1 F P0', 'p8e1 F Rest'],
                         'reading': 'direction: the frozen S (or S difference) has the sign of the new-article mean with a CI excluding 0; a CI '
                                    'covering 0 = undetermined (no misrank claim); calibration S vs calibration D separates first-order '
                                    'approximation (calibration already off) from transfer (calibration fine, new text off)',
                         'others': 'every format / background direction vs f16; same-bit INT / Posit orders; cross-bit exploratory; '
                                   'the original BC frozen order with its raw, background-invariant limits; no coefficient fit'},
          'controls': ['A hash; target INT8 weight in every background; W = only upstream Down weights', 'x captured before the input QDQ, '
                       'candidates from the same x / Wq (no double QDQ); f16 dz = 0, S = 0, D = 0',
                       f'closure loss at z_f equals the full forward with the target at f (calibration, every background / format) within '
                       f'{TOL["closure_vs_full_forward_abs_summed_ce"]} summed CE; also f16 on every new chunk',
                       f'U / F / W f16 and p8e1 per-position CE reproduce runs/qwen3_l27_down_source_20261026 (its tokens) within '
                       f'{TOL["reproduce_down_source_abs_per_position"]}; native per-position values there reported beside them',
                       'gradient only through the final norm and head (no STE); position-127 gradient 0; S_All = S_P0 + S_Rest; truth '
                       'positions add up', 'TF32 off, deterministic, FP32, eager, RMSNorm eps 1e-6'],
          'tolerances': TOL, 'smoke': smoke or 'none',
          'boundaries': 'no change to the formal BC, MILP / knee, configs, quantizers or native; no second-order, endpoint or coefficient; no '
                        'other layer, no STE; cross-layer design comes later',
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L27 Down 的條件化、帶正負號一階評分', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        '- 背景：U（其他 FP32）、F（其他依 A）、W（只有 L0–26 Down 權重依 A）；目標 L27 Down 權重固定 A 的 INT8。',
        f'- 候選：{list(FORMATS)}；f16 為零方向。z_f = Q_y(linear(Q_x(x), Wq))，x 取自 input QDQ 之前。',
        '- 分數：S_b,G(f) = Σ_G g_b·(z_f − z_f16)，g_b 為同背景 f16 輸出處的 summed-CE 梯度（只經 final norm 與 head，無 STE），每背景每窗口一次；G = P0／Rest／All。',
        '- calibration（10 窗口）建立並凍結 S、選擇器（argmin S_All，a bits ≤ 16，平手保留 f16）；同時量 calibration 真值與 α 方向檢查。',
        f'- 新文章（{N_ART} × {WPA}）只量真值 D_b,G(f)；主要家族（Holm 6）：p8e1 的 F−U P0／Rest／All、W−U P0、F 背景 P0／Rest。',
        '- 控制：closure = 完整 forward；重現 down_source 的 U／F／W f16／p8e1 逐位置 CE；f16 δz = 0；位置 127 梯度 0；S 位置可加；TF32 關閉。',
        f'- seed {SEEDS}；排除 {len(used)} 篇已用文章。冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; sizes {sizes}; used {len(used)}')


class Env:
    def __init__(self, work):
        from run_qwen_down_fp16_protection import replay
        self.rp, self.init_s = replay(work)
        self.torch = self.rp.torch
        rp = self.rp
        self.m = rp.mod(TARGET)
        self.rng = rp.ranges[TARGET]
        self.res_mod = rp.mods['model.layers.27.post_attention_layernorm']
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
        self.res_mod.register_forward_pre_hook(res_pre)
        self.m.register_forward_pre_hook(pre)
        self.m.register_forward_hook(post)

    def set_background(self, cfg):
        self.rp.set(cfg)
        self.Wq = self.m.weight.data

    def full(self, toks, fmt, capture=False):
        """full forward with the target at fmt: per-position CE [windows, 127] (LMReplay's batching); optional capture of x and r."""
        torch = self.torch
        self.slot = (fmt, fmt)
        self.cap = {'x': [], 'r': []} if capture else None
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
            return ce, torch.cat(cap['x']), torch.cat(cap['r'])
        return ce

    def zf(self, x, f):
        """the target's post-QDQ output, computed in the same window batches as the full forward (equal GEMM shapes: the coarse
        output QDQ amplifies last-bit differences of a differently shaped matmul)."""
        q = self.rp.qact
        out = []
        for i in range(0, len(x), BATCH):
            xb = x[i:i + BATCH]
            xq = xb if f == 'fp32' else q(xb, f, self.rng['x'])
            y = self.torch.nn.functional.linear(xq, self.Wq)
            out.append(y if f == 'fp32' else q(y, f, self.rng['y']))
        return self.torch.cat(out)

    def closure(self, r, z, ids):
        """per-position CE [windows, 127] of LM_head(final_norm(r + z)), batches of BATCH windows (no grad)."""
        torch = self.torch
        hf = self.rp.ad.model
        out = []
        with torch.no_grad():
            for i in range(0, len(r), BATCH):
                lg = hf.lm_head(hf.model.norm(r[i:i + BATCH] + z[i:i + BATCH]))
                out.append(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[i:i + BATCH, 1:].reshape(-1),
                                                             reduction='none').view(lg.shape[0], -1).double())
        return torch.cat(out)

    def grad(self, r, z0, ids):
        torch = self.torch
        hf = self.rp.ad.model
        g = torch.empty_like(z0)
        for j in range(len(r)):
            zj = z0[j:j + 1].detach().clone().requires_grad_(True)
            lg = hf.lm_head(hf.model.norm(r[j:j + 1] + zj))
            loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            g[j:j + 1], = torch.autograd.grad(loss, zj)
        return g


def masks(torch, device):
    p = torch.arange(SEQ, device=device)
    return {'P0': (p == 0)[None, :, None], 'Rest': (p > 0)[None, :, None], 'All': torch.ones(1, SEQ, 1, dtype=torch.bool, device=device)}


def pos_sum(ce, G):
    """[windows, 127] per-position CE -> per-window sum over G (target position p belongs to the input position p)."""
    return {'P0': ce[:, 0], 'Rest': ce[:, 1:].sum(1), 'All': ce.sum(1)}[G]


# --------------------------------------------------------------------------- #
# calibration: scores, truth, direction checks, controls; freeze
# --------------------------------------------------------------------------- #
def calib(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    env = Env(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    A = read(ENDP / 'configs/A_auto.json')
    cal = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(cal).cuda()
    M = masks(torch, 'cuda')
    ctl = {'closure_vs_full_abs': 0.0, 'f16_zero': True, 'pos127_abs': 0.0, 'position_sum_rel': 0.0, 'target_weight_int8': {}}
    S, D, Dg, curves, cross = {}, {}, {}, [], {}
    G, DZ = {}, {}
    for b in BGS:
        cfg = background(A, b)
        env.set_background(cfg)
        ctl['target_weight_int8'][b] = bool(cfg[TARGET]['w'] == 'int8' and torch.equal(env.Wq, env.rp.cache[(TARGET, 'int8')].to(env.Wq.device)))
        t0 = time.perf_counter()
        full0, x, r = env.full(cal, 'f16', capture=True)
        torch.cuda.synchronize()
        tm[f'{b}|prefix_capture_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        z = {f: env.zf(x, f) for f in FORMATS}
        torch.cuda.synchronize()
        tm[f'{b}|candidate_qdq_s'] = time.perf_counter() - t0
        z0 = z['f16']
        t0 = time.perf_counter()
        g = env.grad(r, z0, ids)
        torch.cuda.synchronize()
        tm[f'{b}|baseline_backward_s'] = time.perf_counter() - t0
        tm['baseline_backward_windows'] += CAL
        ctl['pos127_abs'] = max(ctl['pos127_abs'], float(g[:, -1].abs().max()))
        G[b] = g
        base = env.closure(r, z0, ids)
        for f in FORMATS:
            dz = z[f] - z0
            DZ[(b, f)] = dz
            if f == 'f16':
                ctl['f16_zero'] &= bool((dz == 0).all())
            t0 = time.perf_counter()
            for Gn in POS:
                S[(b, f, Gn)] = float((g.double() * (dz * M[Gn]).double()).sum())
            tm['dot_products_s'] += time.perf_counter() - t0
            ctl['position_sum_rel'] = max(ctl['position_sum_rel'], abs(S[(b, f, 'P0')] + S[(b, f, 'Rest')] - S[(b, f, 'All')]) / max(abs(S[(b, f, 'All')]), 1e-300))
            # truth on calibration and the closure control
            cf = env.closure(r, z[f], ids)
            ff = env.full(cal, f)
            ctl['closure_vs_full_abs'] = max(ctl['closure_vs_full_abs'], float((cf.sum(1) - ff.sum(1)).abs().max()))
            for Gn in POS:
                D[(b, f, Gn)] = float((pos_sum(cf, Gn) - pos_sum(base, Gn)).sum())
                Dg[(b, f, Gn)] = float((env.closure(r, z0 + dz * M[Gn], ids).sum(1) - base.sum(1)).sum())
            # direction checks in the post-QDQ space
            if f != 'f16':
                t0 = time.perf_counter()
                for Gn in POS:
                    md = dz * M[Gn]
                    if not bool(md.abs().max() > 0):
                        continue
                    vals = {}
                    for a in ALPHAS:
                        za = z0 + a * md
                        vals[a] = float((env.closure(r, za, ids).sum(1) - base.sum(1)).sum())
                        supp = md != 0
                        ad = (a * md).double()
                        curves.append({'background': b, 'format': f, 'G': Gn, 'alpha': a, 'dL': vals[a] / TARGETS_CAL,
                                       'changed_fraction': float((za != z0)[supp].float().mean()),
                                       'rounding_rel': float(((za.double() - z0.double()) - ad).norm() / ad.norm())})
                    for a in [x_ for x_ in ALPHAS if x_ > 0]:
                        O = (vals[a] - vals[-a]) / 2 / a
                        curves.append({'background': b, 'format': f, 'G': Gn, 'alpha': a, 'O_over_alpha': O / TARGETS_CAL, 'S': S[(b, f, Gn)] / TARGETS_CAL})
                tm['direction_checks_s'] += time.perf_counter() - t0
        log(f'{b}: p8e1 S P0/Rest/All ' + '/'.join(f'{S[(b, "p8e1", Gn)] / TARGETS_CAL:+.4f}' for Gn in POS)
            + ' | D ' + '/'.join(f'{D[(b, "p8e1", Gn)] / TARGETS_CAL:+.4f}' for Gn in POS))
    ctl['truth_closure_mask_vs_position_sum_abs'] = max(abs(D[k] - Dg[k]) for k in D)
    # cross pairs (mechanism only)
    for f in FORMATS:
        for gb in ('U', 'F'):
            for zb in ('U', 'F'):
                for Gn in POS:
                    cross[f'{f}|g_{gb}|dz_{zb}|{Gn}'] = float((G[gb].double() * (DZ[(zb, f)] * M[Gn]).double()).sum()) / TARGETS_CAL
    # direction convergence per (b, f, G)
    conv = {}
    for b in BGS:
        for f in FORMATS:
            if f == 'f16':
                continue
            for Gn in POS:
                rows = [c for c in curves if c['background'] == b and c['format'] == f and c['G'] == Gn and 'O_over_alpha' in c]
                if not rows:
                    conv[f'{b}|{f}|{Gn}'] = 'zero direction'
                    continue
                small = min(rows, key=lambda c: c['alpha'])
                s_ = S[(b, f, Gn)]
                conv[f'{b}|{f}|{Gn}'] = ('converged' if abs(small['O_over_alpha'] * TARGETS_CAL - s_) <= TOL['direction_rel'] * abs(s_) + TOL['direction_abs_summed_ce']
                                         else 'undetermined')
    # reproduction of the down_source round on its tokens
    t0 = time.perf_counter()
    stok = np.load(SRC / 'tokens.npy')
    sg = np.load(SRC / 'gpu_ce_pos.npz')
    rep = {}
    for b, key in (('U', 'U'), ('F', 'F'), ('W', 'dn_w')):
        env.set_background(background(A, b))
        for f in ('f16', 'p8e1'):
            ce = env.full(stok, f).cpu().numpy()
            rep[f'{b}|{f}'] = float(np.abs(ce - sg[f'{key}__{f}']).max())
    tm['reproduction_s'] = time.perf_counter() - t0
    ctl['reproduce_down_source_abs'] = rep
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    ctl['direction_convergence'] = conv
    ctl['passed'] = bool(ctl['closure_vs_full_abs'] <= TOL['closure_vs_full_forward_abs_summed_ce'] and ctl['f16_zero'] and ctl['pos127_abs'] == 0
                         and ctl['position_sum_rel'] <= TOL['position_sum_rel'] and all(ctl['target_weight_int8'].values())
                         and max(rep.values()) <= TOL['reproduce_down_source_abs_per_position'] and ctl['tf32'] == [False, False]
                         and ctl['deterministic'] and ctl['truth_closure_mask_vs_position_sum_abs'] <= 1e-6)
    # frozen predictions and selectors
    sc = [{'background': b, 'format': f, 'G': Gn, 'S': S[(b, f, Gn)] / TARGETS_CAL, 'calibration_D': D[(b, f, Gn)] / TARGETS_CAL,
           'direction_check': conv.get(f'{b}|{f}|{Gn}', 'baseline')} for b in BGS for f in FORMATS for Gn in POS]
    sel = {}
    for b in BGS:
        order = sorted(SELECT_SET, key=lambda f: (round(S[(b, f, 'All')], 12), SELECT_SET.index(f)))
        best = order[0]
        sel[b] = {'pick': best, 'S_pick': S[(b, best, 'All')] / TARGETS_CAL, 'S_f16': 0.0, 'margin_to_next': (S[(b, order[1], 'All')] - S[(b, best, 'All')]) / TARGETS_CAL,
                  'order': order}
    bc = read(ENDP / 'table_A.json')['C'][TARGET]
    bc_order = sorted(SELECT_SET, key=lambda f: (bc[f], SELECT_SET.index(f)))
    write(out / 'controls_calib.json', ctl)
    write(out / 'frozen_predictions.json', {'scores': sc, 'selector': sel, 'bc_selector': {'pick': bc_order[0], 'order': bc_order,
                                                                                          'raw_C': {f: bc[f] for f in SELECT_SET},
                                                                                          'note': 'raw, background-invariant x + y BC'},
                                            'cross_pairs_mechanism_only': cross})
    with open(out / 'scores.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sc[0]))
        wr.writeheader()
        wr.writerows(sc)
    with open(out / 'calibration_curves.csv', 'w', newline='') as fh:
        keys = sorted({k for c in curves for k in c})
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(curves)
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib', tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; selectors {({b: v["pick"] for b, v in sel.items()})}; bc {bc_order[0]}')
    if not ctl['passed']:
        raise SystemExit(f'calibration controls failed: { {k: v for k, v in ctl.items() if k != "direction_convergence"} }')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_calib.json'),
                                'predictions_sha256': sha(out / 'frozen_predictions.json'), 'scores_sha256': sha(out / 'scores.csv')})


def resolve_subsets(out):
    """an unresolved historical token file is accepted only if every one of its windows equals (token for token) a window of a resolved
    file of the same run; otherwise the audit stays failed. The shingle index already contains every window either way."""
    au = read(out / 'audit.json')
    rows = {r['path']: r for r in au['files']}
    res = {}
    for p in au['unresolved_files']:
        run = Path(p).parts[:2]
        cands = [r['path'] for r in au['files'] if r['resolved'] != 'unresolved' and Path(r['path']).parts[:2] == run and r['path'] != p]
        a = np.load(HERE / p)
        found = None
        for c in cands:
            b = np.load(HERE / c)
            if b.ndim == 2 and b.shape[1] == a.shape[1] and {w.tobytes() for w in a} <= {w.tobytes() for w in b}:
                found = c
                break
        res[p] = found
        if found:
            rows[p]['resolved'] = f'subset of {found} (every window identical)'
    au['subset_resolution'] = res
    au['unresolved_before_subset_check'] = au['unresolved_files']
    au['unresolved_files'] = [p for p, f in res.items() if f is None]
    write(out / 'audit.json', au)
    if au['unresolved_files']:
        raise SystemExit(f'unresolved historical Qwen token sources: {au["unresolved_files"]}')
    log(f'audit: subset resolution {res}')


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()

    def plan_for_sampler(o, inputs_too=True):              # the sampler reads the data fields under 'data'
        p = dict(check_plan(o, inputs_too))
        p['data'] = p['new_articles']
        return p
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = plan_for_sampler, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    try:
        getattr(prs, stage)(out)
    except SystemExit as e:
        if stage != 'audit' or 'unresolved' not in str(e):
            raise
        resolve_subsets(out)
    ca.log_stage(out, stage, time.perf_counter() - t0)


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz_ = read(out / 'data_freeze.json')
    fz = read(out / 'freeze.json')
    if sha(out / 'tokens.npy') != dz_['tokens_sha256'] or sha(out / 'freeze.json') != dz_['freeze_sha256'] or sha(out / 'frozen_predictions.json') != fz['predictions_sha256']:
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

    def st(v):
        b = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(), 'p_two_sided': float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))),
                'articles_negative': int((v < 0).sum())}

    def part(ce, Gn):
        c = ce.reshape(na, WPA, TP)
        return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[Gn] / (WPA * TP)
    D = {(b, f, Gn): part(z[f'{b}__{f}'], Gn) - part(z[f'{b}__f16'], Gn) for b in BGS for f in FORMATS for Gn in POS}
    S = {(r['background'], r['format'], r['G']): r['S'] for r in fp['scores']}
    Dcal = {(r['background'], r['format'], r['G']): r['calibration_D'] for r in fp['scores']}
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def judge(pred, s):
        dec = s['ci95'][0] > 0 or s['ci95'][1] < 0
        return 'undetermined (CI covers 0)' if not dec else ('right' if sgn(pred) == sgn(s['mean']) else 'wrong')
    fam = {'p8e1 F-U P0': (lambda: D[('F', 'p8e1', 'P0')] - D[('U', 'p8e1', 'P0')], S[('F', 'p8e1', 'P0')] - S[('U', 'p8e1', 'P0')],
                           Dcal[('F', 'p8e1', 'P0')] - Dcal[('U', 'p8e1', 'P0')]),
           'p8e1 F-U Rest': (lambda: D[('F', 'p8e1', 'Rest')] - D[('U', 'p8e1', 'Rest')], S[('F', 'p8e1', 'Rest')] - S[('U', 'p8e1', 'Rest')],
                             Dcal[('F', 'p8e1', 'Rest')] - Dcal[('U', 'p8e1', 'Rest')]),
           'p8e1 F-U All': (lambda: D[('F', 'p8e1', 'All')] - D[('U', 'p8e1', 'All')], S[('F', 'p8e1', 'All')] - S[('U', 'p8e1', 'All')],
                            Dcal[('F', 'p8e1', 'All')] - Dcal[('U', 'p8e1', 'All')]),
           'p8e1 W-U P0': (lambda: D[('W', 'p8e1', 'P0')] - D[('U', 'p8e1', 'P0')], S[('W', 'p8e1', 'P0')] - S[('U', 'p8e1', 'P0')],
                           Dcal[('W', 'p8e1', 'P0')] - Dcal[('U', 'p8e1', 'P0')]),
           'p8e1 F P0': (lambda: D[('F', 'p8e1', 'P0')], S[('F', 'p8e1', 'P0')], Dcal[('F', 'p8e1', 'P0')]),
           'p8e1 F Rest': (lambda: D[('F', 'p8e1', 'Rest')], S[('F', 'p8e1', 'Rest')], Dcal[('F', 'p8e1', 'Rest')])}
    prim = {}
    for k, (fn, pred, cal) in fam.items():
        s = st(fn())
        prim[k] = {'new_articles': s, 'frozen_S': pred, 'calibration_D': cal, 'judgement': judge(pred, s), 'calibration_sign_match': sgn(pred) == sgn(cal)}
    order = sorted(prim, key=lambda k: prim[k]['new_articles']['p_two_sided'])
    run_ = 0.0
    for j, k in enumerate(order):
        run_ = max(run_, min(1.0, (len(order) - j) * prim[k]['new_articles']['p_two_sided']))
        prim[k]['new_articles']['p_holm'] = run_
    table = []
    for b in BGS:
        for f in FORMATS:
            if f == 'f16':
                continue
            for Gn in POS:
                s = st(D[(b, f, Gn)])
                table.append({'background': b, 'format': f, 'G': Gn, 'S': S[(b, f, Gn)], 'calibration_D': Dcal[(b, f, Gn)], 'new_mean': s['mean'],
                              'new_ci_lo': s['ci95'][0], 'new_ci_hi': s['ci95'][1], 'judgement': judge(S[(b, f, Gn)], s),
                              'calibration_sign_match': sgn(S[(b, f, Gn)]) == sgn(Dcal[(b, f, Gn)])})
    pairs = []
    for b in BGS:
        for i_, p_ in (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1')):
            d = st(D[(b, i_, 'All')] - D[(b, p_, 'All')])
            pred = S[(b, i_, 'All')] - S[(b, p_, 'All')]
            pairs.append({'background': b, 'pair': f'{i_}-{p_}', 'S_diff': pred, 'new': d, 'judgement': judge(pred, d)})
    sel = {}
    for b in BGS:
        pk = fp['selector'][b]['pick']
        sel[b] = {'pick': pk, 'margin': fp['selector'][b]['margin_to_next'], 'new_D_vs_f16': st(D[(b, pk, 'All')]) if pk != 'f16' else 'f16 kept (0)',
                  'a_bits': BITS[pk], 'target_bops_ratio_vs_f16': BITS[pk] / 16,
                  'oracle_best_in_set_new_mean': min(SELECT_SET, key=lambda f: float(D[(b, f, 'All')].mean()) if f != 'f16' else 0.0)}
    sel['bc'] = {'pick': fp['bc_selector']['pick'], 'note': fp['bc_selector']['note']}
    res = {'units': 'nat/token (contributions; D = format - f16 of the same background)', 'primary': prim, 'table': table, 'same_bit_pairs': pairs,
           'selector': sel, 'cross_pairs_calibration': fp['cross_pairs_mechanism_only']}
    write(out / 'stats.json', res)
    with open(out / 'direction_table.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log('done')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'audit', 'data', 'eval', 'analyze'))
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
        {'calib': calib, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_conditional_signed_score.py')}) + '\n')


if __name__ == '__main__':
    main()
