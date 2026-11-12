#!/usr/bin/env python3
"""Qwen3-0.6B cross-layer confirmation of the first-order term: at pre-specified Down projections (L7, L15, L23; weight = A's
format), backgrounds U (other projections FP32) and F (other projections per A), and three independent 1280-token calibration sets
(10 WikiText-103 articles x one 128-token window each), does S = g_u . (u_f - u_0) added to the full output-side second-order term
K_full (= K2_actual, 1/2 Var_p0 of the actual logit difference) improve the same-bit candidate choices on 128 new articles?

Per target / background / calibration set: complete f16 forward capturing the Down input x_L, the residual r_L and the kwargs of every
later block; each activation format (input and output QDQ) is replayed through the real blocks L+1..27 (background QDQ active), the
final norm and the LM head. g_u from the head only (u_0 a detached leaf, microbatch 1). The scores reuse
run_qwen3_l26_calib_coverage.proxies (S_u, K2_actual; Khat256 is saved but not used). New articles: truth only, drawn after the freeze.
No change to the formal BC, A, quantizers, ranges, MILP / knee or native.

Stages (one new --out-dir): plan, audit, calibdata, calib, data, eval, analyze. GPU stages (calib, eval):
PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import hashlib
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
import run_qwen3_l26_calib_coverage as cov  # noqa: E402
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
ENDP, NATIVE = cs.ENDP, cs.NATIVE
RANGES = cov.RANGES
DIAG = HERE / 'runs/qwen_first_order_necessity_diagnostic_20261027'
COVR = HERE / 'runs/qwen3_l26_calib_coverage_20261027'
LAYERS = (7, 15, 23)
TARGETS = {L: f'/model/model/layers.{L}/mlp/down_proj/MatMul' for L in LAYERS}
BGS = ('U', 'F')
FORMATS, SELECT_SET, POS = cs.FORMATS, cs.SELECT_SET, cs.POS
SAME_BIT = (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1'))
PAIRS = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
SEQ, TP, BATCH, CHUNK = cs.SEQ, cs.TP, cs.BATCH, cs.CHUNK
NSETS, PER_SET = 3, 10
N_ART, WPA, BOOT = 128, 2, 20000
SEEDS = {'cal_order': 26102731, 'cal_start': 26102732, 'n_order': 26102733, 'n_start': 26102734, 'bootstrap': 26102735}
TAU = 4e-7                     # nat/token: score or D_C differences within TAU are ties (as runs/qwen_first_order_necessity_diagnostic)
TOL = {'replay_vs_full_ce_abs_per_position': 1e-6, 'replay_vs_full_tensor_abs': 1e-6, 'replay_vs_full_tensor_rel': 1e-6,
       'f16_replay_vs_full_ce_abs_per_position': 1e-6, 'f16_zero_abs': 0.0, 'position_sum_abs': 1e-12, 'end_reproduce_abs_per_position': 0.0}
CODE = list(dict.fromkeys(['run_qwen_first_order_crosslayer.py'] + cov.CODE))
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def check_plan(out, inputs_too=True):
    pl = read(out / 'plan.json')
    for f, h in cov.code_hashes_in_force(out).items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan / last amendment: {f} (record an amendment)')
    if inputs_too:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def inputs():
    diag = [DIAG / f for f in ('report.md', 'analysis_plan.json', 'summary.json', 'controls.json')]
    covr = [COVR / f for f in ('report.md', 'plan.json', 'calib/wt103/segments.json', 'calib/wt2/segments.json', 'segments.json', 'sampler_check.json')]
    return list(dict.fromkeys(cov.inputs() + diag + covr))


def background(A, L, b):
    """target: A's weight, activation by our hooks (LMReplay sees fp32 and skips it); F: other nodes per A; U: other nodes FP32."""
    t = TARGETS[L]
    return {n: ({'w': A[n]['w'], 'a': 'fp32'} if n == t else dict(A[n]) if b == 'F' else {'w': 'fp32', 'a': 'fp32'}) for n in A}


def pair_pick(v, a, b):
    """lower score of the pair; |difference| <= TAU is a tie -> the earlier format of the selection-set order (the Posit)."""
    if abs(v[a] - v[b]) <= TAU:
        return min((a, b), key=SELECT_SET.index)
    return a if v[a] < v[b] else b


def argmin_sel(v):
    order = sorted(SELECT_SET, key=lambda f: (v[f], SELECT_SET.index(f)))
    return order[0], v[order[1]] - v[order[0]], order


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    from run_olmoe_dynamic_route import env_info
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or len(A) != 196:
        raise SystemExit('A is not the original knee')
    seen = [str(p.relative_to(HERE)) for p in list(HERE.glob('runs/*/plan.json')) + list(HERE.glob('runs/*/analysis_plan.json'))
            if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    rf = read(RANGES)
    targets = {L: {'node': TARGETS[L], 'A': A[TARGETS[L]], 'weight': A[TARGETS[L]]['w'], 'ranges': rf[TARGETS[L]],
                   'int_qparams_scale_zero_point': {s_: cov.int_qparams(rf[TARGETS[L]][s_]) for s_ in ('x', 'y')},
                   'suffix_blocks': f'{L + 1}..27 ({27 - L} blocks)'} for L in LAYERS}
    pl = {'time': now(), 'spec': 'request of 2026-10-27 (Qwen3 first-order cross-layer confirmation); no prompt file in reports/',
          'question': 'at pre-specified Down targets, fixed backgrounds and a fixed calibration budget, does adding S to K_full improve the '
                      'same-bit candidate choices on new articles?',
          'targets': targets, 'targets_rule': 'L7, L15, L23 (0-based) fixed before any result; weight = A\'s format (int6 for all three; A\'s '
                                              'own activations p8e1 / p6e1 / p8e1 are recorded only: every score and truth is relative to the '
                                              'target activation f16); one target at a time',
          'backgrounds': {'U': 'every other projection FP32 (target weight per A)', 'F': 'every other projection weight / activation per A '
                                                                                       '(runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json)'},
          'formats': {'candidates': list(SELECT_SET), 'control': 'fp32 (target activation QDQ off; never selected)', 'slots': 'input and output '
                      'together'},
          'fixed': f'ranges {prs.rel(RANGES)} (sha256 {sha(RANGES)}), never recalibrated; Posit LUT, quantizers, FP32, eager, TF32 off, deterministic, '
                   'use_cache=False, batches of 8 / tail 2 (LMReplay)',
          'calibration data': {'sets': f'{NSETS} disjoint sets x {PER_SET} distinct WikiText-103 train articles x one 128-token window (1280 input '
                                       'tokens, 1270 targets per set); shared by every target / background / scorer; each set scored and picked on '
                                       'its own, never pooled',
                               'draw': f'run_qwen3_l26_calib_coverage.draw (the verified reimplementation of the existing sampler; it reproduced '
                                       f'runs/qwen3_l26_suffix_bridge_20261027 N exactly in runs/qwen3_l26_calib_coverage_20261027/sampler_check.json) '
                                       f'with span 1 window: >= 128 tokens, start uniform in [0, n - 128]; eligible = WikiText-103 train base minus '
                                       f'every earlier manifest article (wt_manifest_scan.used_wt103_verified: {len(used)}, including all old N and the '
                                       'coverage round\'s calibration articles); skips in order: short, near-duplicate of WT2 / of an earlier '
                                       'article / of an accepted article, historical Qwen window / article overlap (0.10), a window with >= 0.5 WT2 '
                                       f'shingle share; order seed {SEEDS["cal_order"]}, start seed {SEEDS["cal_start"]}; set k = accepted articles '
                                       f'10k..10k+9', 'files': 'calib/{tokens.npy,segments.json}, calibration_exclusions.json, calibration_data_freeze.json'},
          'new articles': {'n_articles': N_ART, 'windows_per_article': WPA, 'when': 'after the freeze of every calibration score and pick',
                           'draw': f'same draw with span 2 (>= 256 tokens); excludes the earlier manifests and the 30 calibration articles (exact and '
                                   f'near-duplicate) and WT2; seeds {SEEDS["n_order"]} / {SEEDS["n_start"]}', 'use': 'truth only: per-position CE of the 9 '
                                                                                                                     'formats for every target and background; '
                                                                                                                     'no gradient, proxy, range or pick'},
          'scoring': {'path': 'complete f16 forward (batches 8 + 2) capturing x_L (raw Down input), r_L (input of post_attention_layernorm L), the kwargs '
                              'of every block L+1..27 and u_0; candidate: z = Q_y(linear(Q_x(x_L), W_q)), h = r_L + z, every real block L+1..27 '
                              'with its own captured kwargs (all background QDQ hooks active), final norm once, LM head',
                      'S': 'g_u . (u_f - u_0); g_u = d(summed FP32 CE)/du at the detached leaf u_0, head only, microbatch 1; no STE / HVP / JVP',
                      'K_full': 'K2_actual = 1/2 Var_p0(actual logit difference) per position with the batch-8 baseline logits (float64), as '
                                'run_qwen3_l26_calib_coverage.proxies', 'D_C': 'measured CE(f) - CE(f16) (diagnostic only)',
                      'normalization': 'sum over windows and the 127 target positions / (windows x 127); everything relative to the same '
                                       'background\'s activation f16',
                      'not': 'K_full is not a whole-model Hessian; Khat256 is saved but never scored; this is a diagnostic of the value of '
                             'direction information, not a cheap local criterion or a pipeline'},
          'primary': {'choices': 'per target x background x calibration set x bit: K_full and S + K_full each pick the lower-scoring format of '
                                 '(int8, p8e1), (int6, p6e1), (int4, p4e1); a difference within TAU is a tie and goes to the Posit (earlier in the '
                                 'selection-set order)',
                      'delta': 'delta_n = mean over the 54 choices (3 targets x 2 backgrounds x 3 sets x 3 bits, equal weights) of D_n(S+K pick) - '
                               'D_n(K pick), D_n under the same target and background; the average consequence of many local choices, not a joint '
                               'configuration and not a whole-model Delta NLL',
                      'test': f'{BOOT} paired article resamples, seed {SEEDS["bootstrap"]}, 95% percentile CI; CI upper < 0 -> "under these fixed '
                              'conditions adding the first order improves the same-bit choices"; CI lower > 0 -> "worsens"; otherwise undetermined '
                              '(not equivalence); identical picks give exactly 0',
                      'always listed': 'every target, background, bit and calibration set (and every one of the 54 choices); the article '
                                       'bootstrap excludes calibration-sampling uncertainty'},
          'secondary': {'pairs': 'all 28 pairs per target / background / set: dS, dK, d(S+K), dD_C, dD_N with the paired CI; order flip; fix (K '
                                 'wrong, S+K right) / break / both right / both wrong / score tie / truth undetermined, on C (dD_C, tie TAU) and on '
                                 'N (CI rule); reported for all, same-bit, and pairs with f16 or bf16; descriptive, not confirmatory tests',
                        'background': 'delta_F,n - delta_U,n on the same articles (27 vs 27 choices), overall and per target; a difference is '
                                      'claimed only if its own CI excludes 0',
                        'unrestricted argmin': 'over the 8 candidates for K_full (picks f16 by construction: K >= 0, K(f16) = 0), S + K_full, S, '
                                               'D_C; N Delta NLL vs f16 and paired S+K - K',
                        'P0 / Rest': 'saved for mechanism notes only; no removal or re-weighting'},
          'controls': ['A hash; target weight = cached A format, LMReplay target activation fp32 (no-op), our QDQ hooks once (target 2 pre + 2 '
                       'post), one capture hook on every later block, background weights / activation map equal the configuration, other '
                       'projections 1 + 1 hooks', f'ranges equal {prs.rel(RANGES)} and unchanged after every context; Posit LUT verified',
                       f'replay vs complete forward on every calibration window (3 sets), every target / background / format: block-(L+1) input '
                       f'and u within {TOL["replay_vs_full_tensor_abs"]} + {TOL["replay_vs_full_tensor_rel"]}|ref|, CE within '
                       f'{TOL["replay_vs_full_ce_abs_per_position"]}', 'kwargs: positions 0..127, causal mask, no cache, batches 8 + 2',
                       'f16: S, K, D exactly 0; P0 + Rest = All', 'the first context re-run at the end reproduces exactly',
                       f'new articles: f16 replay vs complete forward per chunk within {TOL["f16_replay_vs_full_ce_abs_per_position"]}',
                       'failures are kept and localized before any reading; fixes need an amendment'],
          'tolerances': TOL, 'tie_tau': TAU, 'seeds': SEEDS, 'smoke': smoke or 'none',
          'cost': 'sampling / audit, prefix, candidate suffix, head CE, head gradient, full K, control forwards, new-article evaluation; peak VRAM '
                  '(torch) and RSS; a diagnostic cost, not a future BC update cost',
          'boundaries': 'no change to the formal BC, configs, quantizers, ranges, MILP / knee or native; no training, STE, HVP, JVP; no top-256 '
                        'scoring; S is never added to the original BC; no post hoc activation threshold, no best-layer deployment, no extra '
                        'experiments; L26 / L27 are history only',
          'used_wt103_count': len(used), 'used_wt103_sha256': hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest(),
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：Qwen3 一階方向資訊的跨層確認', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        f'- 目標（事先指定）：L7／L15／L23 Down，權重 = A 的格式（{", ".join(f"L{L} {targets[L]["weight"]}" for L in LAYERS)}）；A 原本的 activation 只記錄。',
        '- 背景 U（其他投影 FP32）與 F（其他投影依 A）；候選 f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16，input／output 同格式；fp32 只作控制。',
        f'- ranges 固定（{prs.rel(RANGES)}），不重新校準。',
        f'- 校準：3 組互不重疊、各 10 篇 WT103 train 文章 × 1 個 128-token 窗口；seed {SEEDS["cal_order"]}／{SEEDS["cal_start"]}；排除 {len(used)} 篇已用文章與近重複。',
        '- 評分：完整 f16 forward 捕捉 x_L、r_L、每個後續 block 的 kwargs；候選經真實 block L+1..27、final norm、head；S = g_u·Δu（只經 head），K_full = K2_actual。',
        f'- 主要：每個 目標×背景×校準組×bit 的同 bit 對，K_full 與 S+K_full 各選分數較低者（差 ≤ {TAU} 為平手 → 選 Posit）；δ_n = 54 個選擇的平均 [D_n(S+K 選) − D_n(K 選)]。',
        f'- 新文章：凍結後才抽 {N_ART} 篇 × {WPA} 窗口；{BOOT} 次文章 bootstrap（seed {SEEDS["bootstrap"]}）；CI 上界 < 0 才稱改善，下界 > 0 稱變差，否則未定。',
        '- 次要：每層／背景／bit／校準組；28 對修正／破壞（C、N）；同文章的 F−U 背景差（自己的 CI）；無限制 argmin。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; used {len(used)}; targets {[(L, targets[L]["weight"]) for L in LAYERS]}')


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def calibdata(out):
    import wt_manifest_scan as wms
    t_all = time.perf_counter()
    pl = check_plan(out)
    if (out / 'calibration_data_freeze.json').exists():
        raise SystemExit('calibration data exists')
    au = read(out / 'audit.json')
    if au['unresolved_files'] or sha(out / 'work/qwen_hist_shingles.npz') != au['index_sha256']:
        raise SystemExit('audit unresolved or index changed')
    ctx = cov.corpus_ctx(out / 'work/qwen_hist_shingles.npz')
    used, _ = wms.used_wt103_verified(out, ctx['w103'], ctx['wt2'])
    if len(used) != pl['used_wt103_count'] or hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest() != pl['used_wt103_sha256']:
        raise SystemExit('previous WT103 set changed since the plan')
    t0 = time.perf_counter()
    el = [a for a in ctx['base'] if a['ch'] not in used]
    prev_sh = [cov.art_sh(ctx, ctx['by'][h]) for h in used]
    segs, toks, sk, ex = cov.draw(ctx, el, ctx['lines103'], np.random.default_rng(SEEDS['cal_order']).permutation(len(el)),
                                  np.random.default_rng(SEEDS['cal_start']), NSETS * PER_SET, 1, ('wt2', 'previous', 'qwen', 'window_wt2'),
                                  prev_sh, [], 'wikitext-103-raw-v1 train')
    draw_s = time.perf_counter() - t0
    if len(segs) < NSETS * PER_SET:
        write(out / 'calibration_draw_failed.json', {'accepted': len(segs), 'skips': dict(sk)})
        raise SystemExit('not enough calibration articles')
    d = out / 'calib'
    d.mkdir(exist_ok=True)
    for i, sg in enumerate(segs):
        sg.update({'set': i // PER_SET, 'article_in_set': i % PER_SET, 'seed_order': SEEDS['cal_order'], 'seed_start': SEEDS['cal_start']})
    np.save(d / 'tokens.npy', np.concatenate(toks))
    write(d / 'segments.json', segs)
    write(out / 'calibration_exclusions.json', {'skips': dict(sk), 'skipped_during_draw': ex, 'previous_wt103_excluded_by_scan': len(used)})
    sa = cov.self_audit(ctx, {'calibration': d / 'tokens.npy'})
    ok = set(sa['calibration']['windows']) <= {'wikitext-103 listed in the run manifest', 'short'} and \
        len({s['content_sha256'] for s in segs}) == len(segs) and not ({s['content_sha256'] for s in segs} & used)
    write(out / 'calibration_data_freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'audit_sha256': sha(out / 'audit.json'),
                                                 'tokens_sha256': sha(d / 'tokens.npy'), 'segments_sha256': sha(d / 'segments.json'),
                                                 'exclusions_sha256': sha(out / 'calibration_exclusions.json'), 'articles': len(segs),
                                                 'sets': NSETS, 'visited': len(segs) + sum(sk.values()), 'skips': dict(sk), 'self_audit': sa,
                                                 'articles_under_256_tokens': sum(s['article_tokens'] < 2 * SEQ for s in segs),
                                                 'controls_passed': bool(ok), 'corpus_load_s': ctx['load_s'], 'draw_s': draw_s})
    ca.log_stage(out, 'calibdata', time.perf_counter() - t_all, {'corpus_load_s': ctx['load_s'], 'draw_s': draw_s})
    log(f'calibration data: {len(segs)} articles, skips {dict(sk)}, self-audit {sa}, controls {ok}')
    if not ok:
        raise SystemExit('calibration data controls failed')


def data(out):
    import wt_manifest_scan as wms
    t_all = time.perf_counter()
    pl = check_plan(out)
    fz = read(out / 'freeze.json')
    if sha(out / 'frozen_picks.json') != fz['picks_sha256'] or (out / 'segments.json').exists():
        raise SystemExit('freeze changed or data exists')
    au = read(out / 'audit.json')
    if sha(out / 'work/qwen_hist_shingles.npz') != au['index_sha256']:
        raise SystemExit('audit index changed')
    ctx = cov.corpus_ctx(out / 'work/qwen_hist_shingles.npz')
    used, _ = wms.used_wt103_verified(out, ctx['w103'], ctx['wt2'])
    if len(used) != pl['used_wt103_count'] or hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest() != pl['used_wt103_sha256']:
        raise SystemExit('previous WT103 set changed since the plan')
    cal = {s['content_sha256'] for s in read(out / 'calib/segments.json')}
    excl = used | cal
    el = [a for a in ctx['base'] if a['ch'] not in excl]
    prev_sh = [cov.art_sh(ctx, ctx['by'][h]) for h in excl]
    segs, toks, sk, ex = cov.draw(ctx, el, ctx['lines103'], np.random.default_rng(SEEDS['n_order']).permutation(len(el)),
                                  np.random.default_rng(SEEDS['n_start']), N_ART, WPA, ('wt2', 'previous', 'qwen', 'window_wt2'), prev_sh, [],
                                  'wikitext-103-raw-v1 train')
    got = {s['content_sha256'] for s in segs}
    if len(segs) < N_ART or got & excl:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'exclusions.json', {'skipped_during_draw': ex, 'previous_wt103_excluded_by_scan': sorted(used), 'calibration_excluded': sorted(cal),
                                    'wt2_and_internal_duplicates_excluded_before_draw': len(ctx['w103']) - len(ctx['base'])})
    sa = cov.self_audit(ctx, {'N': out / 'tokens.npy'})
    write(out / 'data_freeze.json', {'time': now(), 'freeze_sha256': sha(out / 'freeze.json'), 'audit_sha256': sha(out / 'audit.json'),
                                     'n_articles': N_ART, 'windows_per_article': WPA, 'eligible': len(el), 'visited': N_ART + sum(sk.values()),
                                     'skips': dict(sk), 'excluded_previous_wt103': len(used), 'excluded_calibration': len(cal),
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json'),
                                     'exclusions_sha256': sha(out / 'exclusions.json'), 'self_audit': sa, 'corpus_load_s': ctx['load_s']})
    ca.log_stage(out, 'data', time.perf_counter() - t_all, {'corpus_load_s': ctx['load_s']})
    log(f'data: {N_ART} articles; skips {dict(sk)}; self-audit {sa}')
    if set(sa['N']['windows']) - {'wikitext-103 listed in the run manifest', 'short'}:
        raise SystemExit('new-article tokens would not resolve in a later audit')


# --------------------------------------------------------------------------- #
# GPU environment: one target at a time, real suffix L+1..27
# --------------------------------------------------------------------------- #
class Env:
    def __init__(self, work):
        from run_qwen_down_fp16_protection import replay
        self.rp, self.init_s = replay(work)
        rp = self.rp
        self.torch = rp.torch
        hf = rp.ad.model
        self.hf = hf
        self.layers = hf.model.layers
        self.nm, self.lm = hf.model.norm, hf.lm_head
        self.W = self.lm.weight.data
        self.slot, self.cap, self.handles = ('f16', 'f16'), None, []

        def norm_post(mod, i, o):
            if self.cap is not None:
                self.cap['u'].append(o.detach().clone())
        self.nm.register_forward_hook(norm_post)

    def set_target(self, L):
        rp = self.rp
        for h in self.handles:
            h.remove()
        self.handles = []
        self.L, self.target = L, TARGETS[L]
        self.m = rp.mod(self.target)
        self.rng = rp.ranges[self.target]
        self.blocks = list(self.layers[L + 1:])

        def res_pre(mod, i):
            if self.cap is not None:
                self.cap['r'].append(i[0].detach().clone())

        def pre(mod, i):
            if self.cap is not None:
                self.cap['x'].append(i[0].detach().clone())
            return None if self.slot[0] == 'fp32' else (rp.qact(i[0], self.slot[0], self.rng['x']),)

        def post(mod, i, o):
            return None if self.slot[1] == 'fp32' else rp.qact(o, self.slot[1], self.rng['y'])

        def block_pre(j):
            def f(mod, args, kwargs):
                if self.cap is not None:
                    if j == 0:
                        self.cap['h'].append(args[0].detach().clone())
                    self.cap['kw'][j].append(dict(kwargs))
            return f
        self.handles += [rp.mods[f'model.layers.{L}.post_attention_layernorm'].register_forward_pre_hook(res_pre),
                         self.m.register_forward_pre_hook(pre), self.m.register_forward_hook(post)]
        self.handles += [b.register_forward_pre_hook(block_pre(j), with_kwargs=True) for j, b in enumerate(self.blocks)]

    def set_background(self, cfg):
        self.rp.set(cfg)
        self.Wq = self.m.weight.data
        self.cfg = cfg

    def config_checks(self):
        rp, torch = self.rp, self.torch
        w = self.cfg[self.target]['w']
        ok_w = all(torch.equal(rp.mod(n).weight.data, rp.cache[(n, (self.cfg[n]['w'] or 'fp32'))].to(rp.device)) for n in rp.param)
        ok_a = all(rp.act[n] == self.cfg[n]['a'] for n in rp.param)
        tw = bool(torch.equal(self.Wq, rp.cache[(self.target, w)].to(self.Wq.device)) and not torch.equal(self.Wq, rp.orig[self.target].to(self.Wq.device)))
        hooks = Counter(f'{len(rp.mod(n)._forward_pre_hooks)},{len(rp.mod(n)._forward_hooks)}' for n in rp.param if n != self.target)
        return {'target': self.target, 'weights_equal_config': bool(ok_w), 'activation_map_equal_config': bool(ok_a),
                'target_weight_is_cached_A_format': tw, 'target_weight_format': w, 'target_activation_in_lmreplay': rp.act[self.target],
                'hooks_other_projections': dict(hooks), 'hooks_target': [len(self.m._forward_pre_hooks), len(self.m._forward_hooks)],
                'hooks_later_blocks_pre': sorted({len(b._forward_pre_hooks) for b in self.blocks}),
                'hooks_earlier_blocks_pre': sorted({len(b._forward_pre_hooks) for b in self.layers[:self.L + 1]}),
                'hooks_final_norm': [len(self.nm._forward_pre_hooks), len(self.nm._forward_hooks)],
                'hooks_lm_head': [len(self.lm._forward_pre_hooks), len(self.lm._forward_hooks)]}

    def full(self, toks, fmt, capture=False):
        torch = self.torch
        self.slot = (fmt, fmt)
        self.cap = {'x': [], 'r': [], 'h': [], 'kw': [[] for _ in self.blocks], 'u': []} if capture else None
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
            return ce, {'x': torch.cat(cap['x']), 'r': torch.cat(cap['r']), 'h': torch.cat(cap['h']), 'kw': cap['kw'], 'u': torch.cat(cap['u'])}
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

    def suffix(self, r, z, kw, keep_h=False):
        """h = r + z -> every real block L+1..27 with its own captured kwargs of the same batch -> final norm."""
        torch = self.torch
        us, hs = [], []
        with torch.no_grad():
            for bi, i in enumerate(range(0, len(r), BATCH)):
                h = r[i:i + BATCH] + z[i:i + BATCH]
                if keep_h:
                    hs.append(h)
                for j, blk in enumerate(self.blocks):
                    h = blk(h, **kw[j][bi])
                us.append(self.nm(h))
        return (torch.cat(us), torch.cat(hs)) if keep_h else torch.cat(us)

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


def kwargs_ok(env, kw):
    return all(cov.kwargs_ok(env, kj) for kj in kw)


# --------------------------------------------------------------------------- #
# calibration (GPU)
# --------------------------------------------------------------------------- #
def calib(out, smoke_c0=False):
    t_all = time.perf_counter()
    check_plan(out)
    if smoke_c0:
        sets = {0: np.load(NATIVE / 'calib_inputs.npy')}
    else:
        cdf = read(out / 'calibration_data_freeze.json')
        if sha(out / 'calib/tokens.npy') != cdf['tokens_sha256'] or sha(out / 'calib/segments.json') != cdf['segments_sha256']:
            raise SystemExit('calibration data changed')
        cal = np.load(out / 'calib/tokens.npy')
        sets = {k: cal[k * PER_SET:(k + 1) * PER_SET] for k in range(NSETS)}
    tm = defaultdict(float)
    t0 = time.perf_counter()
    env = Env(out / 'work')
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    from run_olmoe_dynamic_route import env_info
    A = read(ENDP / 'configs/A_auto.json')
    rfile = read(RANGES)
    rh0 = cov.ranges_hash(env.rp.ranges)
    ctl = {'environment': env_info(), 'config': {}, 'ranges': {'file_sha256': sha(RANGES), 'lmreplay_equals_file': env.rp.ranges == rfile,
                                                               'lmreplay_ranges_hash': rh0, 'changed_after': []},
           'posit_lut_verified': {f: list(env.rp.q.posit[f].verify()) for f in ('p4e1', 'p6e1', 'p8e1')},
           'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled()}
    torch.cuda.reset_peak_memory_stats()
    res = {}
    worst, exc = defaultdict(float), defaultdict(lambda: -np.inf)
    kw_ok, shapes, first = True, set(), None

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()
    for L in LAYERS:
        env.set_target(L)
        for b in BGS:
            t0 = sync()
            env.set_background(background(A, L, b))
            tm['set_background_s'] += sync() - t0
            ctl['config'][f'L{L}|{b}'] = env.config_checks()
            for k, toks in sets.items():
                ids = torch.from_numpy(toks).cuda()
                t0 = sync()
                ce0, cap = env.full(toks, 'f16', capture=True)
                t1 = sync()
                tm[f'L{L}|prefix_s'] += t1 - t0
                kw_ok = kw_ok and kwargs_ok(env, cap['kw'])
                shapes.add((len(toks), len(cap['kw'][0]), len(cap['kw'])))
                if first is None:
                    first = (L, b, k, ce0.clone())
                U, H = {}, {}
                for f in FORMATS:
                    U[f], H[f] = env.suffix(cap['r'], env.zf(cap['x'], f), cap['kw'], keep_h=True)
                t2 = sync()
                tm[f'L{L}|candidate_suffix_s'] += t2 - t1
                ce = torch.stack([env.head_ce(U[f], ids) for f in FORMATS])
                t3 = sync()
                tm['head_ce_s'] += t3 - t2
                worst['f16_replay_vs_full_ce'] = max(worst['f16_replay_vs_full_ce'], float((ce[FORMATS.index('f16')] - ce0).abs().max()))
                Lg = {}
                gu = env.grad_pass(U['f16'], ids, on_logits=lambda j, l: Lg.__setitem__(j, l.clone()))
                worst['pos127_g_abs'] = max(worst['pos127_g_abs'], float(gu[:, -1].abs().max()))
                t4 = sync()
                tm['head_gradient_s'] += t4 - t3
                px = cov.proxies(env, U, gu, Lg, ids)
                t5 = sync()
                tm['full_K_and_S_s'] += t5 - t4
                for fi, f in enumerate(FORMATS):                    # replay vs complete forward, every format / window
                    cef, capf = env.full(toks, f, capture=True)
                    for nm_, a_, r_ in (('h', H[f], capf['h']), ('u', U[f], capf['u'])):
                        e, d = l26_excess(a_, r_)
                        exc[f'replay_{nm_}'] = max(exc[f'replay_{nm_}'], e)
                        worst[f'replay_{nm_}_max_abs'] = max(worst[f'replay_{nm_}_max_abs'], d)
                    worst['replay_ce_max_abs'] = max(worst['replay_ce_max_abs'], float((ce[fi] - cef).abs().max()))
                    del capf
                tm['control_full_forward_s'] += sync() - t5
                res[(L, b, k)] = {'CE': ce.cpu().numpy(), 'S_u': px['S_u'], 'K2_actual': px['K2_actual'], 'Khat256': px['Khat256']}
                if cov.ranges_hash(env.rp.ranges) != rh0:
                    ctl['ranges']['changed_after'].append(f'L{L}|{b}|{k}')
                del cap, U, H, Lg, gu, ce, ce0
            log(f'L{L} {b} done')
    L0, b0, k0, ce_first = first
    env.set_target(L0)
    env.set_background(background(A, L0, b0))
    ctl['end_reproduce_abs'] = float((env.full(sets[k0], 'f16') - ce_first).abs().max())
    ctl['kwargs_ok'], ctl['shapes_windows_batches_blocks'] = bool(kw_ok), sorted(shapes)
    ctl['worst'], ctl['excess'] = dict(worst), dict(exc)
    fi16 = FORMATS.index('f16')
    f16 = psum = 0.0
    for v in res.values():
        D = v['CE'] - v['CE'][fi16]
        f16 = max(f16, float(np.abs(D[fi16]).max()), *(float(np.abs(v[t][fi16]).max()) for t in ('S_u', 'K2_actual')))
        for a in (D, v['S_u'], v['K2_actual']):
            psum = max(psum, float(np.abs(a[:, :, 0].sum(1) + a[:, :, 1:].sum((1, 2)) - a.sum((1, 2))).max()))
    ctl['f16_abs'], ctl['position_sum_abs'] = f16, psum
    cfg_ok = all(c['weights_equal_config'] and c['activation_map_equal_config'] and c['target_weight_is_cached_A_format']
                 and c['target_activation_in_lmreplay'] == 'fp32' and c['hooks_target'] == [2, 2] and c['hooks_other_projections'] == {'1,1': 195}
                 and c['hooks_later_blocks_pre'] == [1] and c['hooks_earlier_blocks_pre'] == [0] and c['hooks_final_norm'] == [0, 1]
                 and c['hooks_lm_head'] == [0, 0] for c in ctl['config'].values())
    checks = {'config': cfg_ok, 'ranges_equal_file': ctl['ranges']['lmreplay_equals_file'], 'ranges_unchanged': not ctl['ranges']['changed_after'],
              'posit_lut': all(v[0] == 0 for v in ctl['posit_lut_verified'].values()), 'tf32_off': ctl['tf32'] == [False, False],
              'deterministic': ctl['deterministic'], 'kwargs': ctl['kwargs_ok'],
              'f16_replay': worst['f16_replay_vs_full_ce'] <= TOL['f16_replay_vs_full_ce_abs_per_position'],
              'replay_tensors': exc['replay_h'] <= 0 and exc['replay_u'] <= 0, 'replay_ce': worst['replay_ce_max_abs'] <= TOL['replay_vs_full_ce_abs_per_position'],
              'f16_zero': f16 <= TOL['f16_zero_abs'], 'pos127_g': worst['pos127_g_abs'] == 0, 'position_sum': psum <= TOL['position_sum_abs'],
              'end_reproduce': ctl['end_reproduce_abs'] <= TOL['end_reproduce_abs_per_position']}
    ctl['checks'] = checks
    ctl['passed'] = bool(all(checks.values()))
    tag = '_smoke_C0' if smoke_c0 else ''
    write(out / f'controls_calib{tag}.json', ctl)
    np.savez(out / f'C_values{tag}.npz', **{f'L{L}__{b}__s{k}__{t}': v[t] for (L, b, k), v in res.items() for t in v})
    picks, rows = freeze_picks(res)
    write(out / f'frozen_picks{tag}.json', picks)
    with open(out / f'scores_C{tag}.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib' + tag, tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; failed {[k for k, v in checks.items() if not v]}; worst {dict(worst)}; time {tm["total_s"]:.0f}s')
    if not ctl['passed']:
        raise SystemExit('calibration controls failed')
    if smoke_c0:
        return
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'calibration_data_freeze_sha256': sha(out / 'calibration_data_freeze.json'),
                                'controls_sha256': sha(out / 'controls_calib.json'), 'picks_sha256': sha(out / 'frozen_picks.json'),
                                'values_sha256': sha(out / 'C_values.npz'), 'scores_sha256': sha(out / 'scores_C.csv'),
                                'A_sha256': sha(ENDP / 'configs/A_auto.json'), 'ranges_sha256': sha(RANGES),
                                'code_sha256': {f: sha(HERE / f) for f in CODE}})


def l26_excess(a, b):
    d = (a - b).abs()
    return float((d - TOL['replay_vs_full_tensor_abs'] - TOL['replay_vs_full_tensor_rel'] * b.abs()).max()), float(d.max())


def tok_scores(v):
    """per-token P0 / Rest / All of S, K_full, S+K_full, D_C for every format."""
    fi16 = FORMATS.index('f16')
    D = v['CE'] - v['CE'][fi16]
    nw = D.shape[1]
    out = {}
    for i, f in enumerate(FORMATS):
        for G, sl in (('P0', np.s_[:, 0]), ('Rest', np.s_[:, 1:]), ('All', np.s_[:, :])):
            S, K, d = (float(a[i][sl].sum() / (nw * TP)) for a in (v['S_u'], v['K2_actual'], D))
            out[(f, G)] = {'S': S, 'K_full': K, 'S+K_full': S + K, 'D_C': d}
    return out


def freeze_picks(res):
    picks, rows = {}, []
    for (L, b, k), v in res.items():
        sc = tok_scores(v)
        allv = {f: sc[(f, 'All')] for f in FORMATS}
        key = f'L{L}|{b}|s{k}'
        p = {'same_bit': {}, 'argmin': {}}
        for a, c in SAME_BIT:
            p['same_bit'][f'{a}-{c}'] = {s_: pair_pick({f: allv[f][s_] for f in (a, c)}, a, c) for s_ in ('K_full', 'S+K_full', 'S', 'D_C')}
            p['same_bit'][f'{a}-{c}']['scores'] = {s_: [allv[a][s_], allv[c][s_]] for s_ in ('K_full', 'S+K_full', 'S', 'D_C')}
        for s_ in ('K_full', 'S+K_full', 'S', 'D_C'):
            pk, mg, order = argmin_sel({f: allv[f][s_] for f in SELECT_SET})
            p['argmin'][s_] = {'pick': pk, 'margin': mg, 'order': order}
        picks[key] = p
        for f in FORMATS:
            for G in POS:
                rows.append({'target': f'L{L}', 'background': b, 'set': k, 'format': f, 'G': G, **sc[(f, G)]})
    return {'rule_same_bit': f'lower score of the pair; |difference| <= {TAU} -> the Posit', 'rule_argmin': 'exact ties: f16 first, then the set order',
            'picks': picks, 'note': 'calibration only; frozen before any new article'}, rows


# --------------------------------------------------------------------------- #
# new articles (GPU)
# --------------------------------------------------------------------------- #
def eval_(out):
    t_all = time.perf_counter()
    check_plan(out)
    dz, fz = read(out / 'data_freeze.json'), read(out / 'freeze.json')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'frozen_picks.json') != fz['picks_sha256']:
        raise SystemExit('data / freeze changed')
    env = Env(out / 'work')
    torch = env.torch
    A = read(ENDP / 'configs/A_auto.json')
    rh0 = cov.ranges_hash(env.rp.ranges)
    tok = np.load(out / 'tokens.npy')
    res, dev, cfgs, tm = {}, 0.0, {}, defaultdict(float)
    torch.cuda.reset_peak_memory_stats()
    for L in LAYERS:
        env.set_target(L)
        for b in BGS:
            env.set_background(background(A, L, b))
            cfgs[f'L{L}|{b}'] = env.config_checks()
            arr = {f: np.zeros((len(tok), TP)) for f in FORMATS}
            t0 = time.perf_counter()
            for c0 in range(0, len(tok), CHUNK):
                t = tok[c0:c0 + CHUNK]
                ids = torch.from_numpy(t).cuda()
                ce0, cap = env.full(t, 'f16', capture=True)
                for f in FORMATS:
                    ce = env.head_ce(env.suffix(cap['r'], env.zf(cap['x'], f), cap['kw']), ids)
                    arr[f][c0:c0 + len(t)] = ce.cpu().numpy()
                    if f == 'f16':
                        dev = max(dev, float((ce - ce0).abs().max()))
                del cap
            torch.cuda.synchronize()
            tm[f'L{L}|{b}_s'] = time.perf_counter() - t0
            for f in FORMATS:
                res[f'L{L}__{b}__{f}'] = arr[f]
            log(f'eval L{L} {b} {tm[f"L{L}|{b}_s"]:.0f}s')
    np.savez(out / 'eval_ce_pos.npz', **res)
    ok = bool(dev <= TOL['f16_replay_vs_full_ce_abs_per_position'] and cov.ranges_hash(env.rp.ranges) == rh0
              and all(c['target_weight_is_cached_A_format'] and c['hooks_target'] == [2, 2] and c['weights_equal_config']
                      and c['activation_map_equal_config'] for c in cfgs.values()))
    write(out / 'controls_eval.json', {'f16_replay_vs_full_ce_abs_per_position': dev, 'ranges_unchanged': cov.ranges_hash(env.rp.ranges) == rh0,
                                       'config': cfgs, 'passed': ok})
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    ca.log_stage(out, 'eval', time.perf_counter() - t_all, dict(tm))
    if not ok:
        raise SystemExit('new-article controls failed')


# --------------------------------------------------------------------------- #
# analysis (CPU)
# --------------------------------------------------------------------------- #
def sgn(d):
    return 0 if abs(d) <= TAU else (1 if d > 0 else -1)


def category(oK, oP, T):
    if T == 0:
        return 'truth_tie_or_undetermined'
    if oK == 0 or oP == 0:
        return 'score_tie'
    return {(False, True): 'fix', (True, False): 'break', (True, True): 'both_right', (False, False): 'both_wrong'}[(oK == T, oP == T)]


def analyze(out):
    t_all = time.perf_counter()
    check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if sha(out / 'frozen_picks.json') != fz['picks_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256']:
        raise SystemExit('freeze / data changed')
    picks = read(out / 'frozen_picks.json')['picks']
    Cz = np.load(out / 'C_values.npz')
    z = np.load(out / 'eval_ce_pos.npz')
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, N_ART, N_ART), minlength=N_ART) for _ in range(BOOT)]) / N_ART

    def st(v):
        bs = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'pos': int((v > 0).sum()), 'neg': int((v < 0).sum())}

    def verdict(s):
        return 'improves' if s['ci95'][1] < 0 else ('worsens' if s['ci95'][0] > 0 else 'undetermined')

    def part(ce, G):
        c = ce.reshape(N_ART, WPA, TP)
        return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[G] / (WPA * TP)
    D = {(L, b, f, G): part(z[f'L{L}__{b}__{f}'], G) - part(z[f'L{L}__{b}__f16'], G) for L in LAYERS for b in BGS for f in FORMATS for G in POS}
    ctl = {'N_f16_zero': max(float(np.abs(D[(L, b, 'f16', 'All')]).max()) for L in LAYERS for b in BGS) == 0,
           'N_P0_plus_Rest': max(float(np.abs(D[(L, b, f, 'P0')] + D[(L, b, f, 'Rest')] - D[(L, b, f, 'All')]).max())
                                 for L in LAYERS for b in BGS for f in FORMATS) <= 1e-12,
           'eval_keys': sorted(z.files) == sorted(f'L{L}__{b}__{f}' for L in LAYERS for b in BGS for f in FORMATS),
           'eval_shapes': all(z[k].shape == (N_ART * WPA, TP) for k in z.files),
           'picks_keys': sorted(picks) == sorted(f'L{L}|{b}|s{k}' for L in LAYERS for b in BGS for k in range(NSETS))}
    # scores recomputed from C_values must equal the frozen ones
    sc = {}
    for L in LAYERS:
        for b in BGS:
            for k in range(NSETS):
                v = {t: Cz[f'L{L}__{b}__s{k}__{t}'] for t in ('CE', 'S_u', 'K2_actual')}
                sc[(L, b, k)] = tok_scores(v)
    ctl['frozen_scores_reproduce'] = all(abs(sc[(L, b, k)][(a, 'All')][s_] - picks[f'L{L}|{b}|s{k}']['same_bit'][f'{a}-{c}']['scores'][s_][0]) <= 1e-15
                                         for L in LAYERS for b in BGS for k in range(NSETS) for a, c in SAME_BIT for s_ in ('K_full', 'S+K_full'))
    ctl['passed'] = bool(all(ctl.values()))
    if not ctl['passed']:
        write(out / 'controls_analysis.json', ctl)
        raise SystemExit(f'analysis controls failed {ctl}')
    write(out / 'controls_analysis.json', ctl)
    # ---- primary: 54 same-bit choices
    conds, choice_rows = [], []
    for L in LAYERS:
        for b in BGS:
            for k in range(NSETS):
                for a, c in SAME_BIT:
                    p = picks[f'L{L}|{b}|s{k}']['same_bit'][f'{a}-{c}']
                    d = D[(L, b, p['S+K_full'], 'All')] - D[(L, b, p['K_full'], 'All')]
                    conds.append(((L, b, k, a), d))
                    s = st(d)
                    sK, sP = p['scores']['K_full'], p['scores']['S+K_full']
                    choice_rows.append({'target': f'L{L}', 'background': b, 'set': k, 'bit': a[3:], 'pair': f'{a}-{c}', 'K_pick': p['K_full'],
                                        'SK_pick': p['S+K_full'], 'S_pick': p['S'], 'D_C_pick': p['D_C'], 'changed': p['K_full'] != p['S+K_full'],
                                        'K_int': sK[0], 'K_posit': sK[1], 'SK_int': sP[0], 'SK_posit': sP[1],
                                        'D_C_int': p['scores']['D_C'][0], 'D_C_posit': p['scores']['D_C'][1],
                                        'N_int_minus_posit': float((D[(L, b, a, 'All')] - D[(L, b, c, 'All')]).mean()),
                                        'delta_mean': s['mean'], 'delta_lo': s['ci95'][0], 'delta_hi': s['ci95'][1]})
    delta = np.mean([d for _, d in conds], axis=0)
    prim = st(delta)
    sub = {}

    def sub_delta(pred):
        ds = [d for key, d in conds if pred(key)]
        return st(np.mean(ds, axis=0)), len(ds), sum(r['changed'] for r, (key, _) in zip(choice_rows, conds) if pred(key))
    for name, pred in ([(f'L{L}', (lambda key, L=L: key[0] == L)) for L in LAYERS] + [(b, (lambda key, b=b: key[1] == b)) for b in BGS] +
                       [(f'bit{a[3:]}', (lambda key, a=a: key[3] == a)) for a, _ in SAME_BIT] +
                       [(f'set{k}', (lambda key, k=k: key[2] == k)) for k in range(NSETS)] +
                       [(f'L{L}|{b}', (lambda key, L=L, b=b: key[0] == L and key[1] == b)) for L in LAYERS for b in BGS]):
        s, n, ch = sub_delta(pred)
        sub[name] = {'N': s, 'choices': n, 'changed_choices': ch, 'reading': verdict(s)}
    dF = np.mean([d for key, d in conds if key[1] == 'F'], axis=0)
    dU = np.mean([d for key, d in conds if key[1] == 'U'], axis=0)
    bgdiff = {'all': st(dF - dU)}
    for L in LAYERS:
        bgdiff[f'L{L}'] = st(np.mean([d for key, d in conds if key[1] == 'F' and key[0] == L], axis=0) -
                             np.mean([d for key, d in conds if key[1] == 'U' and key[0] == L], axis=0))
    for v in bgdiff.values():
        v['reading'] = 'F - U differs (CI excludes 0)' if (v['ci95'][0] > 0 or v['ci95'][1] < 0) else 'no difference shown (CI covers 0)'
    # ---- pairwise (28) per target / background / set
    NP = {(L, b, a, c): st(D[(L, b, a, 'All')] - D[(L, b, c, 'All')]) for L in LAYERS for b in BGS for a, c in PAIRS}
    prows = []
    for (L, b, k), v in sc.items():
        for a, c in PAIRS:
            A_, B_ = v[(a, 'All')], v[(c, 'All')]
            dS, dK, dD = A_['S'] - B_['S'], A_['K_full'] - B_['K_full'], A_['D_C'] - B_['D_C']
            dP = dS + dK
            n = NP[(L, b, a, c)]
            TN = 0 if not (n['ci95'][0] > 0 or n['ci95'][1] < 0) else (1 if n['mean'] > 0 else -1)
            oK, oP, oS, TC = sgn(dK), sgn(dP), sgn(dS), sgn(dD)
            prows.append({'target': f'L{L}', 'background': b, 'set': k, 'pair': f'{a}-{c}', 'same_bit': (c, a) in SAME_BIT or (a, c) in SAME_BIT,
                          'with_f16_or_bf16': bool({a, c} & {'f16', 'bf16'}), 'dS': dS, 'dK': dK, 'dP': dP, 'dD_C': dD, 'dD_N': n['mean'],
                          'dD_N_lo': n['ci95'][0], 'dD_N_hi': n['ci95'][1], 'order_S': oS, 'order_K': oK, 'order_P': oP, 'truth_C': TC, 'truth_N': TN,
                          'flip': oK != oP, 'cat_C': category(oK, oP, TC), 'cat_N': category(oK, oP, TN),
                          'S_alone_N': 'tie' if oS == 0 or TN == 0 else ('right' if oS == TN else 'wrong')})

    def counts(rows):
        r = {}
        for T in ('C', 'N'):
            cc = Counter(x[f'cat_{T}'] for x in rows)
            r[T] = dict(cc)
            r[T]['net'] = cc['fix'] - cc['break']
        return r
    pair_summary = {}
    for scope, pred in (('all', lambda r: True), ('same_bit', lambda r: r['same_bit']), ('with_f16_or_bf16', lambda r: r['with_f16_or_bf16']),
                        ('other', lambda r: not r['same_bit'] and not r['with_f16_or_bf16'])):
        pair_summary[scope] = {'total': counts([r for r in prows if pred(r)])}
        for L in LAYERS:
            for b in BGS:
                pair_summary[scope][f'L{L}|{b}'] = counts([r for r in prows if pred(r) and r['target'] == f'L{L}' and r['background'] == b])
        for k in range(NSETS):
            pair_summary[scope][f'set{k}'] = counts([r for r in prows if pred(r) and r['set'] == k])
    # ---- unrestricted argmin (secondary)
    arows = []
    for (L, b, k) in sc:
        am = picks[f'L{L}|{b}|s{k}']['argmin']
        for s_ in ('K_full', 'S+K_full', 'S', 'D_C'):
            s = st(D[(L, b, am[s_]['pick'], 'All')])
            arows.append({'target': f'L{L}', 'background': b, 'set': k, 'scorer': s_, 'pick': am[s_]['pick'], 'C_margin': am[s_]['margin'],
                          'N_vs_f16': s['mean'], 'N_lo': s['ci95'][0], 'N_hi': s['ci95'][1], 'N_harmful_articles': s['pos']})
        s = st(D[(L, b, am['S+K_full']['pick'], 'All')] - D[(L, b, am['K_full']['pick'], 'All')])
        arows.append({'target': f'L{L}', 'background': b, 'set': k, 'scorer': 'S+K_full minus K_full', 'pick': f'{am["S+K_full"]["pick"]} vs {am["K_full"]["pick"]}',
                      'C_margin': '', 'N_vs_f16': s['mean'], 'N_lo': s['ci95'][0], 'N_hi': s['ci95'][1], 'N_harmful_articles': s['pos']})
    # ---- per-candidate N table and C scores
    nrows = []
    for L in LAYERS:
        for b in BGS:
            for f in FORMATS:
                for G in POS:
                    s = st(D[(L, b, f, G)])
                    nrows.append({'target': f'L{L}', 'background': b, 'format': f, 'G': G, 'mean': s['mean'], 'ci_lo': s['ci95'][0], 'ci_hi': s['ci95'][1],
                                  'articles_pos': s['pos'], 'articles_neg': s['neg']})
    for name, rows in (('choices_same_bit.csv', choice_rows), ('pairwise_diagnostics.csv', prows), ('argmin_outcomes.csv', arows), ('N_truth.csv', nrows)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    np.savez(out / 'N_article_D.npz', **{f'L{L}__{b}__{f}__{G}': D[(L, b, f, G)] for L in LAYERS for b in BGS for f in FORMATS for G in POS},
             delta_primary=delta)
    write(out / 'stats.json', {'units': 'nat/token relative to the same target / background activation f16; article bootstrap over the 128 new articles',
                               'primary': {'definition': 'delta_n = mean over 54 same-bit choices of D_n(S+K_full pick) - D_n(K_full pick)', 'N': prim,
                                           'reading': verdict(prim), 'changed_choices': int(sum(r['changed'] for r in choice_rows)), 'choices': len(choice_rows)},
                               'subsets': sub, 'background_F_minus_U': bgdiff, 'pairs': pair_summary,
                               'N_mean_D_All': {f'L{L}|{b}|{f}': float(D[(L, b, f, 'All')].mean()) for L in LAYERS for b in BGS for f in SELECT_SET}})
    ca.log_stage(out, 'analyze', time.perf_counter() - t_all)
    log(f'primary delta {prim["mean"]:+.6f} {prim["ci95"]} -> {verdict(prim)}; changed {sum(r["changed"] for r in choice_rows)}/54; '
        f'pairs N {pair_summary["all"]["total"]["N"]}')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'audit', 'calibdata', 'calib', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note')
    ap.add_argument('--smoke-c0', action='store_true', help='calib on the old C0 windows only (smoke test), no freeze')
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
    cov.LOGF = LOGF
    t0 = time.time()
    if a.stage == 'plan':
        plan(out, a.smoke_note)
    elif a.stage == 'audit':
        cov.check_plan = check_plan
        cov.audit(out)
    elif a.stage == 'calib':
        calib(out, a.smoke_c0)
    else:
        {'calibdata': calibdata, 'data': data, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen_first_order_crosslayer.py'),
                             'argv': sys.argv}) + '\n')


if __name__ == '__main__':
    main()
