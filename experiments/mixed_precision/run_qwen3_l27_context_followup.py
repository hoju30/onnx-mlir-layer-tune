#!/usr/bin/env python3
"""Follow-up of runs/qwen3_l27_context_attribution_20261026 on new articles (GPU LMReplay, not native):
  1. position: in the all-FP32 (U) and all-quantized (F) backgrounds, the L27 Down a = p8e1 (x and y) only at position 0 or
     only at positions 1..127 (f16 elsewhere); where does the context change of its effect sit?
  2. FFN bisection: the L0-26 FFN background split by depth (L0-13 / L14-26) and by type (gate + up / down).
The U / F / ffn effects of the previous round are re-measured on the new articles (replication).
D = NLL(p8e1 setting) - NLL(f16) per article (negative = p8e1 better). Same A, target weight int8, backgrounds as before.

Stages (one new --out-dir): plan, controls, audit, data, eval, analyze. GPU stages (controls, eval):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
PREV = HERE / 'runs/qwen3_l27_context_attribution_20261026'
TARGET, ENDP, L27C, NATIVE = ca.TARGET, ca.ENDP, ca.L27C, ca.NATIVE
BACKGROUNDS = ('U', 'F', 'ffn', 'ffn_early', 'ffn_late', 'ffn_gu', 'ffn_down')
SUBGROUPS = ('ffn_early', 'ffn_late', 'ffn_gu', 'ffn_down')
POS_SETTINGS = ('p8e1_pos0', 'p8e1_rest')
N_ART, WPA, TP, BOOT = 128, 2, 127, 20000
SEEDS = {'order': 26102641, 'start': 26102642, 'bootstrap': 26102643}
CODE = ['run_qwen3_l27_context_followup.py'] + ca.CODE
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def member(n, b):
    l, p = ca.parse(n)
    ffn = p in ('gate', 'up', 'down') and l <= 26
    return {'U': False, 'F': True, 'ffn': ffn, 'ffn_early': ffn and l <= 13, 'ffn_late': ffn and l >= 14,
            'ffn_gu': ffn and p in ('gate', 'up'), 'ffn_down': ffn and p == 'down'}[b]


def background_config(A, b):
    return {n: (dict(A[n]) if n == TARGET or member(n, b) else {'w': 'fp32', 'a': 'fp32'}) for n in A}


def conditions():
    c = [(b, s) for b in BACKGROUNDS for s in ('f16', 'p8e1')]
    return c + [(b, s) for b in ('U', 'F') for s in POS_SETTINGS]


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


def plan(out, smoke):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH:
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
    pl = {'time': now(), 'request': 'user 2026-10-26: run follow-up steps 1 (position) and 2 (FFN bisection)',
          'previous': 'runs/qwen3_l27_context_attribution_20261026 (C1 -0.0075; only the L0-26 FFN group significant alone -0.0027)',
          'backgrounds': {'U': 'all other projections FP32', 'F': 'all per A', 'ffn': 'L0-26 gate/up/down per A', 'ffn_early': 'L0-13 FFN',
                          'ffn_late': 'L14-26 FFN', 'ffn_gu': 'L0-26 gate + up', 'ffn_down': 'L0-26 down'},
          'quantized_projections_per_background': sizes,
          'target': f'{TARGET}: w int8 (A); a f16 or p8e1 on x and y; in U and F also p8e1 only at position 0 (x and y rows of '
                    'position 0) or only at positions 1..127, f16 at the other positions',
          'conditions': len(conditions()),
          'statistics': {'D': 'NLL(setting) - NLL(f16) of the same background, article means of 2 windows',
                         'step1': 'D_pos0, D_rest, interaction D_all - D_pos0 - D_rest in U and F; context changes (F - U) of pos0 and rest: '
                                  'two comparisons, Holm; the position carrying the context change = Holm p < 0.05 with the sign of F - U of all',
                         'step2': 'D_g - D_U for the 4 sub-groups, Holm over 4; within-FFN additivity: (ffn - U) - (early + late) and '
                                  '- (gate/up + down) (descriptive)', 'replication': 'C1 = D_F - D_U and D_ffn - D_U re-measured (descriptive '
                                  'replication of the previous round on independent articles)',
                         'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}); 95% percentile CI; two-sided p = 2 min tail'},
          'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data', 'excluded_previous_wt103': len(used),
                   'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
          'controls': ['previous round\'s controls repeated: F f16 / p8e1 reproduce the l27-confirm PyTorch A / C bitwise; position mask '
                       '"all" equals the unmasked hooks bitwise; U with the target fp32 equals all FP32; background sizes'],
          'boundaries': 'GPU only; no criterion / config / pipeline change', 'smoke': smoke or 'none',
          'inputs': {prs.rel(p): sha(p) for p in ca.inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L27 情境效應的位置與 FFN 二分（續）', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        '- 背景：U 全 FP32、F 全依 A、ffn（L0–26 FFN）、ffn_early（L0–13 FFN）、ffn_late（L14–26 FFN）、ffn_gu（L0–26 gate+up）、ffn_down（L0–26 down）。',
        '- 目標：L27 Down w int8；a = f16／p8e1（x 與 y）；U、F 另測 p8e1 只在 position 0、只在 1–127。',
        '- 步驟 1：U、F 中 D_pos0、D_rest 與交互；位置的情境改變（F − U）兩個比較 Holm。步驟 2：4 個子組 D_g − D_U（Holm）；FFN 內加成性描述。',
        f'- 重現：新文章上重量 C1 與 ffn − U。資料：{N_ART} 篇新 WT103 × {WPA}；seed {SEEDS}。GPU LMReplay，非 native。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; sizes {sizes}; used {len(used)}')


class PosRunner(ca.Runner):
    """the previous Runner with an optional position mask on the target's x / y quantization."""

    def __init__(self, work):
        super().__init__(work)
        self.pos = 'all'
        torch = self.torch
        rp = self.rp
        rng = rp.ranges[TARGET]
        m = rp.mod(TARGET)
        # replace the parent's hooks (registered after LMReplay's) by position-aware ones
        pre_ids, post_ids = list(m._forward_pre_hooks)[1:], list(m._forward_hooks)[1:]
        for k in pre_ids:
            del m._forward_pre_hooks[k]
        for k in post_ids:
            del m._forward_hooks[k]

        def q(t, fmt, r):
            """fmt on the masked positions, f16 on the others (pos 'all': fmt everywhere)."""
            qt = rp.qact(t, fmt, r)
            if self.pos == 'all':
                return qt
            mask = torch.zeros(t.shape[1], dtype=torch.bool, device=t.device)
            mask[0] = True
            if self.pos == 'rest':
                mask = ~mask
            return torch.where(mask[None, :, None], qt, rp.qact(t, 'f16', r))

        def pre(mod, i):
            if self.slot is None or self.slot[0] == 'fp32':
                return None
            return (q(i[0], self.slot[0], rng['x']),)

        def post(mod, i, o):
            if self.slot is None or self.slot[1] == 'fp32':
                return None
            return q(o, self.slot[1], rng['y'])
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)

    def run_pos(self, cfg, setting, toks):
        if setting in POS_SETTINGS:
            self.pos = 'pos0' if setting == 'p8e1_pos0' else 'rest'
            ce, _ = self.run(cfg, ('p8e1', 'p8e1'), toks)
        else:
            self.pos = 'all'
            ce, _ = self.run(cfg, ca.SETTINGS[setting], toks)
        self.pos = 'all'
        return ce


def controls(out):
    t0 = time.perf_counter()
    check_plan(out)
    r = PosRunner(out / 'work')
    A = read(ENDP / 'configs/A_auto.json')
    ref = np.load(L27C / 'pytorch_ce.npz')
    tok = np.load(L27C / 'tokens.npy')
    F = background_config(A, 'F')
    ctl = {'F_equals_A': F == A}
    ceA = r.run_pos(F, 'f16', tok)
    ceC = r.run_pos(F, 'p8e1', tok)
    ctl['reproduce_l27_confirm_pytorch'] = {'A_max_abs': float(np.abs(ceA - ref['A']).max()), 'C_max_abs': float(np.abs(ceC - ref['C']).max())}
    cal = np.load(NATIVE / 'calib_inputs.npy')
    lm = r.run(F, ('p8e1', 'p8e1'), cal, own_hooks=False)[0]
    ctl['all_mask_equals_lmreplay_path_abs'] = float(np.abs(r.run_pos(F, 'p8e1', cal) - lm).max())
    allfp = r.run({n: {'w': 'fp32', 'a': 'fp32'} for n in A}, ('fp32', 'fp32'), cal)[0]
    U = background_config(A, 'U')
    ctl['U_target_fp32_equals_all_fp32_max_abs'] = float(np.abs(r.run({**U, TARGET: {'w': 'fp32', 'a': 'fp32'}}, ('fp32', 'fp32'), cal)[0] - allfp).max())
    ctl['background_sizes'] = {b: sum(1 for n in A if n != TARGET and member(n, b)) for b in BACKGROUNDS}
    ctl['tf32'] = [r.torch.backends.cuda.matmul.allow_tf32, r.torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = r.torch.are_deterministic_algorithms_enabled()
    pl = read(out / 'plan.json')
    ctl['passed'] = bool(ctl['F_equals_A'] and max(ctl['reproduce_l27_confirm_pytorch'].values()) == 0 and ctl['all_mask_equals_lmreplay_path_abs'] == 0
                         and ctl['U_target_fp32_equals_all_fp32_max_abs'] == 0 and ctl['background_sizes'] == pl['quantized_projections_per_background']
                         and ctl['tf32'] == [False, False] and ctl['deterministic'])
    write(out / 'controls.json', ctl)
    ca.log_stage(out, 'controls', time.perf_counter() - t0)
    log(f'controls passed={ctl["passed"]}: {ctl}')
    if not ctl['passed']:
        raise SystemExit('controls failed')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls.json')})


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('controls / freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    ca.log_stage(out, stage, time.perf_counter() - t0)


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz = read(out / 'data_freeze.json')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256']:
        raise SystemExit('data / freeze changed')
    r = PosRunner(out / 'work')
    A = read(ENDP / 'configs/A_auto.json')
    tok = np.load(out / 'tokens.npy')
    res = {}
    for b, s in conditions():
        t1 = time.perf_counter()
        res[f'{b}__{s}'] = r.run_pos(background_config(A, b), s, tok)
        log(f'{b}__{s}: {time.perf_counter() - t1:.0f}s')
    np.savez(out / 'eval_ce.npz', **res)
    ca.log_stage(out, 'eval', time.perf_counter() - t0, {'conditions': len(res), 'windows': len(tok), 'eval_ce_sha256': sha(out / 'eval_ce.npz')})


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
        return {'mean': float(v.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(),
                'p_two_sided': float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))), 'articles_negative': int((v < 0).sum())}

    def holm(d):
        order = sorted(d, key=lambda k: d[k]['p_two_sided'])
        run_ = 0.0
        for j, k in enumerate(order):
            run_ = max(run_, min(1.0, (len(order) - j) * d[k]['p_two_sided']))
            d[k]['p_holm'] = run_
        return d
    D = {k[:-len('__p8e1')]: nll[k] - nll[k.replace('__p8e1', '__f16')] for k in nll if k.endswith('__p8e1')}
    Dp = {(b, s): nll[f'{b}__{s}'] - nll[f'{b}__f16'] for b in ('U', 'F') for s in POS_SETTINGS}
    res = {'units': 'nat/token, article means of 2 windows; D = p8e1 setting - f16 (negative = p8e1 better)', 'D_all': {b: st(v) for b, v in D.items()}}
    res['replication'] = {'C1 D_F-D_U': st(D['F'] - D['U']), 'D_ffn-D_U': st(D['ffn'] - D['U'])}
    res['position'] = {b: {'D_pos0': st(Dp[(b, 'p8e1_pos0')]), 'D_rest': st(Dp[(b, 'p8e1_rest')]),
                           'interaction': st(D[b] - Dp[(b, 'p8e1_pos0')] - Dp[(b, 'p8e1_rest')])} for b in ('U', 'F')}
    pc = holm({'pos0 F-U': st(Dp[('F', 'p8e1_pos0')] - Dp[('U', 'p8e1_pos0')]), 'rest F-U': st(Dp[('F', 'p8e1_rest')] - Dp[('U', 'p8e1_rest')])})
    sgn = np.sign(res['replication']['C1 D_F-D_U']['mean'])
    for k in pc:
        pc[k]['carries_context_change'] = bool(pc[k]['p_holm'] < 0.05 and np.sign(pc[k]['mean']) == sgn)
    res['position_context_change'] = pc
    sub = holm({g: st(D[g] - D['U']) for g in SUBGROUPS})
    for g in sub:
        sub[g]['significant'] = bool(sub[g]['p_holm'] < 0.05)
    res['ffn_bisection'] = sub
    res['ffn_additivity'] = {'ffn - (early + late)': st((D['ffn'] - D['U']) - (D['ffn_early'] - D['U']) - (D['ffn_late'] - D['U'])),
                             'ffn - (gate/up + down)': st((D['ffn'] - D['U']) - (D['ffn_gu'] - D['U']) - (D['ffn_down'] - D['U']))}
    res['nll_f16'] = {b: float(nll[f'{b}__f16'].mean()) for b in BACKGROUNDS}
    write(out / 'stats.json', res)
    segs = read(out / 'segments.json')
    with open(out / 'per_article.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        keys = sorted(nll)
        wr.writerow(['article', 'title', *keys])
        for i in range(na):
            wr.writerow([i, segs[i]['title'], *[nll[k][i] for k in keys]])
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log('done')


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
                             'script_sha256': sha(HERE / 'run_qwen3_l27_context_followup.py')}) + '\n')


if __name__ == '__main__':
    main()
