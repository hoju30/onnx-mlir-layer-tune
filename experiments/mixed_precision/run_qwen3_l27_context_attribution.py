#!/usr/bin/env python3
"""Qwen3-0.6B: is the gain of the L27 Down activation f16 -> p8e1 (a = input x and output y) context-dependent, and which
quantized projections create the context? Same articles for every background (the earlier evidence came from three
different article batches). GPU LMReplay (PyTorch QDQ) only; native confirmation is a separate follow-up.

Configuration A = the original BC knee (hash 36205a06...). The L27 Down keeps w = int8 (A) everywhere; its a is f16 or
p8e1 (also p8e1 on x only / y only in two backgrounds). Backgrounds = which of the other 195 searched projections take
their A formats (w and a); the rest are FP32:
  U none | own L27 q/k/v/o/gate/up | early L0-13 all | late L14-26 all | attn L0-26 q/k/v/o | ffn L0-26 gate/up/down
  P = early + late (L0-26) | F = P + own (= A itself for the f16 setting)
D_b = NLL(b, p8e1) - NLL(b, f16) per article (negative = p8e1 better).

Stages (one new --out-dir): plan, controls, audit, data, eval, analyze. GPU stages (controls, eval):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (sampler, read / write / sha / now / rel)

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE = prs.NATIVE
L27C = HERE / 'runs/qwen3_l27_native_confirm_20261024'
ENDP = HERE / 'runs/qwen3_spike_endpoint_pipeline_20261026'
TARGET = '/model/model/layers.27/mlp/down_proj/MatMul'
A_HASH = '36205a0624a33d5577a1cb5c1b71919d03e400f8dbab44ac2ef11d3104df2f4f'
BACKGROUNDS = ('U', 'own', 'early', 'late', 'attn', 'ffn', 'P', 'F')
GROUPS = ('own', 'early', 'late', 'attn', 'ffn')
SETTINGS = {'f16': ('f16', 'f16'), 'p8e1': ('p8e1', 'p8e1'), 'p8e1_x': ('p8e1', 'f16'), 'p8e1_y': ('f16', 'p8e1')}
SLOT_BACKGROUNDS = ('U', 'F')
N_ART, WPA, SEQ, TP, BOOT = 128, 2, 128, 127, 20000
SEEDS = {'order': 26102631, 'start': 26102632, 'bootstrap': 26102633}
CODE = ['run_qwen3_l27_context_attribution.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_down_fp16_protection.py', 'run_lp_pipeline_eval.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'wt_manifest_scan.py', 'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py']
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


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def inputs():
    import run_olmoe_route_mass_restore as mr
    return ([HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] +
            [ENDP / 'configs/A_auto.json', L27C / 'configs/A.json', L27C / 'configs/C.json', L27C / 'tokens.npy', L27C / 'pytorch_ce.npz',
             L27C / 'report.md', NATIVE / 'ranges_by_point.json', NATIVE / 'calib_inputs.npy', HERE / 'qwen35_assets/model/tokenizer.json',
             HERE / 'runs/qwen3_l27_reversal_diag_20261024/report.md', HERE / 'runs/qwen3_prefix_rank_shift_20261024/report.md'] + mr.WT103 + mr.WT2)


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


def parse(n):
    m = re.match(r'/model/model/layers\.(\d+)/(self_attn|mlp)/(\w+)_proj/MatMul', n)
    return int(m.group(1)), m.group(3)


def member(n, b):
    """is projection n (not the target) quantized per A in background b?"""
    l, p = parse(n)
    attn = p in ('q', 'k', 'v', 'o')
    return {'U': False, 'own': l == 27, 'early': l <= 13, 'late': 14 <= l <= 26, 'attn': l <= 26 and attn, 'ffn': l <= 26 and not attn,
            'P': l <= 26, 'F': True}[b]


def background_config(A, b):
    return {n: (dict(A[n]) if n == TARGET or member(n, b) else {'w': 'fp32', 'a': 'fp32'}) for n in A}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if config_hash(A) != A_HASH or A[TARGET] != {'w': 'int8', 'a': 'f16'} or len(A) != 196:
        raise SystemExit('A is not the original knee')
    sizes = {b: sum(member(n, b) for n in A if n != TARGET) for b in BACKGROUNDS}
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'request': 'user 2026-10-26: confirm the source of the L27 context dependence first',
        'question': 'on the same articles: does the L27 Down a f16 -> p8e1 effect change with which other projections are quantized '
                    '(all FP32 / L0-26 / all, the earlier three batches gave +0.0006 / +0.0047 / +0.0085), and which group creates it?',
        'configuration': {'A': 'runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json (original BC knee, hash 36205a06...)',
                          'target': f'{TARGET}: w = int8 (A) everywhere; a = f16 or p8e1 on x and y (also p8e1 on x only / y only in U and F)',
                          'backgrounds': {'U': 'all other projections FP32', 'own': 'L27 q/k/v/o/gate/up per A', 'early': 'L0-13 all per A',
                                          'late': 'L14-26 all per A', 'attn': 'L0-26 q/k/v/o per A', 'ffn': 'L0-26 gate/up/down per A',
                                          'P': 'L0-26 all per A', 'F': 'all 195 other projections per A (F with f16 = A)'},
                          'quantized_projections_per_background': sizes},
        'conditions': f'{len(BACKGROUNDS)} backgrounds x (f16, p8e1) + {len(SLOT_BACKGROUNDS)} backgrounds x (p8e1 x-only, p8e1 y-only) = '
                      f'{len(BACKGROUNDS) * 2 + len(SLOT_BACKGROUNDS) * 2}',
        'backend': 'GPU LMReplay (PyTorch QDQ: native INT ranges / qparams on the device, Universal posit LUT, f16 cast, per-channel '
                   'weights of qwen3_swiglu_common.Quantizers), TF32 off, deterministic, FP32; the target uses its own hooks (x and y '
                   'formats set apart), checked equal to LMReplay\'s own path; not a native result',
        'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data (verified manifests, historical '
                 'Qwen token audit, WT2 / near-duplicate / previous-article rules, window WT2 >= 50% rejected)',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
        'statistics': {'unit': 'article (2 windows, 254 targets)', 'D_b': 'NLL(b, p8e1) - NLL(b, f16), negative = p8e1 better',
                       'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}), one index set; 95% percentile CI; '
                                    'two-sided p = 2 min(P(b >= 0), P(b <= 0))',
                       'primary': 'C1 = D_F - D_U (context dependence): confirmed if the 95% CI excludes 0; D_U, D_P, D_F with CIs; '
                                  'D_P - D_U and D_F - D_P reported',
                       'attribution': 'D_g - D_U for g in own / early / late / attn / ffn, Holm over the 5; a group carries part of the '
                                      'context effect if Holm p < 0.05 with the sign of C1; several groups allowed; additivity residuals '
                                      '(D_F - D_U) - sum over {own, early, late} and over {own, attn, ffn} reported (interaction, descriptive)',
                       'slots': 'in U and F: D_x (p8e1 on x only) and D_y (y only); their context changes D_x,F - D_x,U and D_y,F - D_y,U '
                                'and the xy interaction, descriptive with CIs',
                       'mechanism_descriptive': 'per background with f16: the L27 Down input / output max |.|, fraction of |y| > 4096 '
                                                '(p8e1 maxpos), position-0 share of the extremes'},
        'reading': ['C1 CI excludes 0 -> context dependence confirmed on the same articles (GPU); otherwise not confirmed and the '
                    'attribution is descriptive only', 'groups are not exclusive (attn / ffn cut L0-26 the other way); no group is called '
                    'the sole cause', 'no criterion / config / pipeline change; native confirmation of the located source is a separate step'],
        'controls': ['A reproduced: F with f16 / p8e1 on the l27 native-confirm articles equal that round\'s PyTorch A / C per window '
                     '(bitwise)', 'the target\'s own hooks equal LMReplay\'s own path for A and C on the calibration windows (bitwise)',
                     'U with the target (w, a) = fp32 equals the all-FP32 model (bitwise); the int8 target weight alone reported',
                     'every background\'s quantized projection count as planned',
                     'TF32 off, deterministic'],
        'smoke': smoke or 'none',
        'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：L27 Down f16→p8e1 效果的情境依賴與來源歸因', '', f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。', '',
          '- 問題：同一批文章上，L27 Down 的 a 由 f16 改 p8e1（x 與 y）的效果是否隨「其他投影是否量化」改變？是哪一組造成？',
          '- 配置：A = 原 BC knee（hash 36205a06…）；L27 Down 的 w 一律 int8；背景決定其他 195 個投影哪些依 A（w 與 a），其餘 FP32。',
          '- 背景：U 全 FP32｜own L27 自己的 q/k/v/o/gate/up｜early L0–13｜late L14–26｜attn L0–26 的 q/k/v/o｜ffn L0–26 的 gate/up/down｜P = L0–26｜F = 全部（F + f16 即 A）。',
          f'- 條件：8 背景 × (f16, p8e1) + U、F 的 p8e1 只在 x／只在 y = 20 個；GPU LMReplay（非 native）。',
          f'- 資料：{N_ART} 篇新 WT103 × {WPA} 窗口；排除 {len(used)} 篇已用文章；seed {SEEDS}。',
          f'- 主要：C1 = D_F − D_U（D = p8e1 − f16，負值 p8e1 較好），95% CI 排除 0 才算確認情境依賴；歸因：D_g − D_U（5 組，Holm）；加成殘差與 x／y 槽位拆分為描述。',
          '- 控制：在 L27 confirm 的文章上重現前輪 PyTorch A／C（逐位元）；目標 hook 與 LMReplay 原路徑相同；U 且目標 (w, a) 皆 fp32 = 全 FP32；TF32 關閉、deterministic。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; sizes {sizes}; used {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# GPU replay with the target's own x / y hooks
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, work):
        from run_qwen_down_fp16_protection import replay
        self.rp, self.init_s = replay(work)
        self.torch = self.rp.torch
        self.slot = None                                   # (x format, y format) of the target, applied by our hooks
        self.stats = None
        m = self.rp.mod(TARGET)
        rng = self.rp.ranges[TARGET]

        def pre(mod, i):
            if self.slot is None:
                return None
            if self.stats is not None:
                self.stats['x_absmax'] = max(self.stats.get('x_absmax', 0.0), float(i[0].abs().max()))
            return None if self.slot[0] == 'fp32' else (self.rp.qact(i[0], self.slot[0], rng['x']),)

        def post(mod, i, o):
            if self.slot is None:
                return None
            if self.stats is not None:
                a = o.abs()
                self.stats['y_absmax'] = max(self.stats.get('y_absmax', 0.0), float(a.max()))
                self.stats['y_beyond_4096'] = self.stats.get('y_beyond_4096', 0) + int((a > 4096).sum())
                self.stats['y_beyond_4096_pos0'] = self.stats.get('y_beyond_4096_pos0', 0) + int((a[:, 0] > 4096).sum())
                self.stats['y_elements'] = self.stats.get('y_elements', 0) + int(a.numel())
            return None if self.slot[1] == 'fp32' else self.rp.qact(o, self.slot[1], rng['y'])
        m.register_forward_pre_hook(pre)                    # registered after LMReplay's own hooks
        m.register_forward_hook(post)

    def run(self, cfg, slot, toks, own_hooks=True, stats=False):
        """slot = (x, y) formats of the target; own_hooks False -> LMReplay's own path (cfg[TARGET]['a'] on x and y)."""
        c = {n: dict(v) for n, v in cfg.items()}
        if own_hooks:
            c[TARGET] = {'w': cfg[TARGET]['w'], 'a': 'fp32'}       # LMReplay's hooks skip the target; ours apply the slot
            self.slot = slot
        else:
            c[TARGET] = {'w': cfg[TARGET]['w'], 'a': slot[0]}
            self.slot = None
            if slot[0] != slot[1]:
                raise ValueError('LMReplay path needs one format')
        self.rp.set(c)
        self.stats = {} if stats else None
        ce = self.rp.ce(toks)
        st, self.stats, self.slot = self.stats, None, None
        return ce, st


def controls(out):
    t0 = time.perf_counter()
    check_plan(out)
    r = Runner(out / 'work')
    A = read(ENDP / 'configs/A_auto.json')
    ref = np.load(L27C / 'pytorch_ce.npz')
    tok = np.load(L27C / 'tokens.npy')
    F = background_config(A, 'F')
    ctl = {'F_equals_A': F == A}
    ceA, _ = r.run(F, SETTINGS['f16'], tok)
    ceC, _ = r.run(F, SETTINGS['p8e1'], tok)
    ctl['reproduce_l27_confirm_pytorch'] = {'A_max_abs': float(np.abs(ceA - ref['A']).max()), 'C_max_abs': float(np.abs(ceC - ref['C']).max()),
                                            'C_minus_A_per_token': float((ceC - ceA).sum() / (TP * len(tok)))}
    cal = np.load(NATIVE / 'calib_inputs.npy')
    own = {k: r.run(F, SETTINGS[k], cal)[0] for k in ('f16', 'p8e1')}
    lm = {k: r.run(F, SETTINGS[k], cal, own_hooks=False)[0] for k in ('f16', 'p8e1')}
    ctl['own_hooks_equal_lmreplay_path'] = {k: float(np.abs(own[k] - lm[k]).max()) for k in own}
    U = background_config(A, 'U')
    allfp = r.run({n: {'w': 'fp32', 'a': 'fp32'} for n in A}, ('fp32', 'fp32'), cal)[0]
    u_fp = r.run({**U, TARGET: {'w': 'fp32', 'a': 'fp32'}}, ('fp32', 'fp32'), cal)[0]
    u_w8 = r.run(U, ('fp32', 'fp32'), cal)[0]
    ctl['U_target_fp32_equals_all_fp32_max_abs'] = float(np.abs(u_fp - allfp).max())
    ctl['U_target_int8_weight_only_minus_all_fp32_per_token'] = float((u_w8 - allfp).sum() / (TP * len(cal)))
    ctl['background_sizes'] = {b: sum(1 for n in A if n != TARGET and background_config(A, b)[n] != {'w': 'fp32', 'a': 'fp32'}) for b in BACKGROUNDS}
    ctl['tf32'] = [r.torch.backends.cuda.matmul.allow_tf32, r.torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = r.torch.are_deterministic_algorithms_enabled()
    pl = read(out / 'plan.json')
    ctl['passed'] = bool(ctl['F_equals_A'] and ctl['reproduce_l27_confirm_pytorch']['A_max_abs'] == 0 and ctl['reproduce_l27_confirm_pytorch']['C_max_abs'] == 0
                         and all(v == 0 for v in ctl['own_hooks_equal_lmreplay_path'].values()) and ctl['U_target_fp32_equals_all_fp32_max_abs'] == 0
                         and ctl['background_sizes'] == pl['configuration']['quantized_projections_per_background']
                         and ctl['tf32'] == [False, False] and ctl['deterministic'])
    write(out / 'controls.json', ctl)
    log_stage(out, 'controls', time.perf_counter() - t0, {'replay_init_s': r.init_s})
    log(f'controls passed={ctl["passed"]}: {ctl}')
    if not ctl['passed']:
        raise SystemExit('controls failed')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls.json'),
                                'script_sha256': sha(HERE / 'run_qwen3_l27_context_attribution.py')})


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('controls / freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    log_stage(out, stage, time.perf_counter() - t0)


def conditions():
    c = [(b, s) for b in BACKGROUNDS for s in ('f16', 'p8e1')]
    return c + [(b, s) for b in SLOT_BACKGROUNDS for s in ('p8e1_x', 'p8e1_y')]


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz = read(out / 'data_freeze.json')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256']:
        raise SystemExit('data / freeze changed')
    r = Runner(out / 'work')
    A = read(ENDP / 'configs/A_auto.json')
    tok = np.load(out / 'tokens.npy')
    ck = out / 'work' / 'eval_partial.npz'
    done = dict(np.load(ck)) if ck.exists() else {}
    hdr = {'tokens_sha256': dz['tokens_sha256'], 'script_sha256': sha(HERE / 'run_qwen3_l27_context_attribution.py')}
    if done and json.loads(str(done.pop('header'))) != hdr:
        raise SystemExit('checkpoint from another version / data')
    stats = read(out / 'work' / 'eval_stats.json') if (out / 'work' / 'eval_stats.json').exists() else {}
    for b, s in conditions():
        key = f'{b}__{s}'
        if key in done:
            continue
        t1 = time.perf_counter()
        ce, st = r.run(background_config(A, b), SETTINGS[s], tok, stats=(s == 'f16'))
        done[key] = ce
        if st is not None:
            stats[b] = st
            write(out / 'work' / 'eval_stats.json', stats)
        np.savez(ck, header=json.dumps(hdr), **done)
        log(f'{key}: {time.perf_counter() - t1:.0f}s')
    np.savez(out / 'eval_ce.npz', **done)
    write(out / 'mechanism_stats.json', stats)
    log_stage(out, 'eval', time.perf_counter() - t0, {'replay_init_s': r.init_s, 'conditions': len(done), 'windows': len(tok),
                                                       'eval_ce_sha256': sha(out / 'eval_ce.npz')})


def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    z = np.load(out / 'eval_ce.npz')
    na = z['U__f16'].shape[0] // WPA
    nll = {k: z[k].reshape(na, WPA).sum(1) / (WPA * TP) for k in z.files}
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]) / na

    def st(v):
        b = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(), 'p_two_sided': float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))),
                'articles_negative': int((v < 0).sum()), 'articles_positive': int((v > 0).sum())}
    D = {b: nll[f'{b}__p8e1'] - nll[f'{b}__f16'] for b in BACKGROUNDS}
    res = {'units': 'nat/token, article means of 2 windows; D = p8e1 - f16 (negative = p8e1 better)',
           'nll_f16': {b: float(nll[f'{b}__f16'].mean()) for b in BACKGROUNDS}, 'D': {b: st(D[b]) for b in BACKGROUNDS}}
    res['C1 D_F-D_U'] = st(D['F'] - D['U'])
    res['D_P-D_U'] = st(D['P'] - D['U'])
    res['D_F-D_P'] = st(D['F'] - D['P'])
    att = {g: st(D[g] - D['U']) for g in GROUPS}
    order = sorted(att, key=lambda g: att[g]['p_two_sided'])
    run_ = 0.0
    for j, g in enumerate(order):
        run_ = max(run_, min(1.0, (len(order) - j) * att[g]['p_two_sided']))
        att[g]['p_holm'] = run_
        att[g]['carries_part'] = bool(run_ < 0.05 and np.sign(att[g]['mean']) == np.sign(res['C1 D_F-D_U']['mean']))
    res['attribution'] = att
    res['additivity_residual_own_early_late'] = st((D['F'] - D['U']) - sum(D[g] - D['U'] for g in ('own', 'early', 'late')))
    res['additivity_residual_own_attn_ffn'] = st((D['F'] - D['U']) - sum(D[g] - D['U'] for g in ('own', 'attn', 'ffn')))
    slots = {}
    for b in SLOT_BACKGROUNDS:
        Dx, Dy = nll[f'{b}__p8e1_x'] - nll[f'{b}__f16'], nll[f'{b}__p8e1_y'] - nll[f'{b}__f16']
        slots[b] = {'D_x': st(Dx), 'D_y': st(Dy), 'interaction_xy': st(D[b] - Dx - Dy)}
    slots['context_change_x F-U'] = st((nll['F__p8e1_x'] - nll['F__f16']) - (nll['U__p8e1_x'] - nll['U__f16']))
    slots['context_change_y F-U'] = st((nll['F__p8e1_y'] - nll['F__f16']) - (nll['U__p8e1_y'] - nll['U__f16']))
    res['slots'] = slots
    c1 = res['C1 D_F-D_U']
    res['reading'] = {'C1': 'context dependence confirmed (CI excludes 0)' if c1['ci95'][0] > 0 or c1['ci95'][1] < 0 else 'not confirmed (CI covers 0)',
                      'groups_carrying_part': [g for g in GROUPS if att[g]['carries_part']]}
    res['mechanism_stats'] = read(out / 'mechanism_stats.json')
    write(out / 'stats.json', res)
    segs = read(out / 'segments.json')
    with open(out / 'per_article.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        keys = sorted(nll)
        wr.writerow(['article', 'title', *keys])
        for i in range(na):
            wr.writerow([i, segs[i]['title'], *[nll[k][i] for k in keys]])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'D': {b: (round(v['mean'], 5), [round(x, 5) for x in v['ci95']]) for b, v in res['D'].items()},
                    'C1': c1, 'attribution': {g: (round(v['mean'], 5), round(v['p_holm'], 4)) for g, v in att.items()},
                    'slots': {k: (v['mean'] if 'mean' in v else {kk: round(vv['mean'], 5) for kk, vv in v.items()}) for k, v in slots.items()},
                    'reading': res['reading']}, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'controls', 'audit', 'data', 'eval', 'analyze'))
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
        {'controls': controls, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_l27_context_attribution.py')}) + '\n')


if __name__ == '__main__':
    main()
