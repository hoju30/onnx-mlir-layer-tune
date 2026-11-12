#!/usr/bin/env python3
"""Qwen3-0.6B: does quantizing only the earlier transformer blocks change the measured ranking of later Down activation
candidates? A GPU FP32 + QDQ diagnostic (run_lp_pipeline_eval.LMReplay semantics); no BC / search / knee / range /
quantizer / weight change, no fine-tuning, no new criterion, no BC statistics recomputed.

For every target layer i (all 28 mlp.down_proj) two backgrounds:
  U  blocks 0..i-1 FP32,                         block i other projections FP32, blocks i+1..27 FP32
  Q  blocks 0..i-1 every searched projection as A, block i other projections FP32, blocks i+1..27 FP32
The target Down weight is A's w in both (bitwise identical); its activation `a` (input x and output y, fixed native
ranges / Posit LUT) is one of fp32 (baseline: activation QDQ off), int4, p4e1, int6, p6e1, int8, p8e1, f16.
L0 has an empty prefix (U = Q): negative control, outside the cross-layer statistics.

Selectors frozen on the original 10 calibration windows before the new articles: q_BC (native table C[node][a]),
q_U / q_Q (calibration NLL in background U / Q). Truth: 128 new WikiText-103 articles x 2 windows.

Implementation: per chunk of windows, two full forwards (all FP32, all A) capture the hidden state entering every block
(the prefix of U and Q); each condition replays blocks i..27 from it with LMReplay's hooks. On the calibration windows
every condition is also run as a complete forward and must equal the replay.

Stages (one new --out-dir): plan, calib, freeze, audit, data, eval, analyze.
CPU: ../../venv/bin/python; calib / eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FP16 = HERE / 'runs/qwen3_down_fp16_protection_20261024'
L27R = HERE / 'runs/qwen3_l27_native_confirm_20261024'
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
REF = HERE / 'runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json'
NATIVE = HERE / 'runs/native_eps_fix_20261014/qwen3'
PREP = HERE / 'runs/qwen3_prepare_20261011'
MODEL = HERE / 'qwen3_assets/model'
A_SHA = 'ac10afa96c7833cacc6f7886857a97c9ba9b4e2cc151712194373f0e9490c06b'
SETTINGS = ('fp32', 'int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1', 'f16')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
SECONDARY = ('f16', 'p8e1')
BACKS = ('U', 'Q')
NL, SEQ, TP, CAL = 28, 128, 127, 10
N_ART, WPA = 128, 2
BOOT, ALPHA, EPS = 20000, 0.05, 0.001
PROXY_TIE, NUM_TIE, LOW_MARGIN, WT2_WINDOW_MAX = 0.02, 1e-8, 0.001, 0.5
TOL_REPLAY = 1e-6                     # |replay - complete forward| summed CE per window (bitwise expected)
SEEDS = {'order': 30101, 'start': 30102, 'bootstrap': 30103}
KNOWN = (2, 14, 27)
GROUPS = {'L1-8': range(1, 9), 'L9-17': range(9, 18), 'L18-27': range(18, 28)}
CHUNK, LM_SUB = 32, 8
LAYERS_RUN = list(range(NL))          # smoke tests may restrict; the formal run uses all 28
CODE = ['run_qwen_prefix_rank_shift.py', 'run_lp_pipeline_eval.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py',
        'run_w8_per_channel_check.py', 'run_qwen_down_fp16_protection.py', 'run_qwen_down_spike_diag.py', 'wt_manifest_scan.py',
        'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py', 'htq_pareto_search.py']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')


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


def rel(p):
    return os.path.relpath(p, HERE)


def input_files():
    import run_olmoe_route_mass_restore as mr
    return ([MODEL / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] +
            [FP16 / 'configs/A.json', REF, NATIVE / 'table.json', NATIVE / 'ranges_by_point.json', NATIVE / 'calib_inputs.npy',
             PREP / 'node_mapping.json', PREP / 'validation_tokens.npy', PREP / 'test_tokens.npy', FP16 / 'segments.json',
             SPIKE / 'segments.json', L27R / 'segments.json', HERE / 'qwen35_assets/model/tokenizer.json'] + mr.WT103 + mr.WT2)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment first)')
    if inputs:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def structure():
    """node -> (layer, kind) from the mapping (layer parsed as an integer), the Down node per layer, A."""
    m = read(PREP / 'node_mapping.json')
    layer_of = {}
    for node, param in m.items():
        p = param.split('.')
        if p[0] != 'model' or p[1] != 'layers':
            raise SystemExit(f'unexpected mapping {param}')
        layer_of[node] = (int(p[2]), p[4].removesuffix('_proj'))
    down = {l: n for n, (l, k) in layer_of.items() if k == 'down'}
    return layer_of, down, read(FP16 / 'configs/A.json')


def condition_cfg(A, layer_of, down, i, s, f):
    """the complete configuration of condition (target layer i, background s, target activation f)."""
    cfg = {}
    if s == 'Q':
        cfg = {n: dict(A[n]) for n, (l, _) in layer_of.items() if l < i}
    cfg[down[i]] = {'w': A[down[i]]['w'], 'a': f}
    return cfg


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def pick(pair, score, a_orig, tie):
    """the lower-score member of a same-bit pair; a tie picks A's original a if it is in the pair, else the INT format."""
    if tie:
        return a_orig if a_orig in pair else pair[0]
    return pair[0] if score[pair[0]] < score[pair[1]] else pair[1]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke=None):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists: the plan is frozen once')
    layer_of, down, A = structure()
    cfgj = read(MODEL / 'config.json')
    chk = {'A_sha256': sha(FP16 / 'configs/A.json') == A_SHA, 'A_equals_ref_json': sha(REF) == A_SHA,
           'config_layers': cfgj['num_hidden_layers'], 'projections': len(layer_of), 'mapping_layers': sorted({l for l, _ in layer_of.values()}) == list(range(NL)),
           'seven_per_block': all(sum(1 for l, _ in layer_of.values() if l == b) == 7 for b in range(NL)),
           'A_covers_mapping': set(A) == set(layer_of), 'down_per_layer': len(down) == NL}
    if not (chk['A_sha256'] and chk['A_equals_ref_json'] and chk['config_layers'] == NL and chk['projections'] == 196
            and chk['mapping_layers'] and chk['seven_per_block'] and chk['A_covers_mapping'] and chk['down_per_layer']):
        raise SystemExit(f'structure check failed: {chk}')
    table = read(NATIVE / 'table.json')
    q_bc, bc_rows = {}, []
    for i in range(NL):
        n = down[i]
        for a, b in PAIRS:
            c = {f: table['C'][n][f] for f in (a, b)}
            tie = abs(c[a] - c[b]) <= PROXY_TIE * max(abs(c[a]), abs(c[b]), 1e-30)
            q = pick((a, b), c, A[n]['a'], tie)
            q_bc[f'L{i}|{a}-{b}'] = q
            bc_rows.append({'layer': i, 'pair': f'{a}-{b}', 'C_int': c[a], 'C_posit': c[b], 'proxy_tie': bool(tie), 'q_BC': q, 'A_a': A[n]['a']})
    write(out / 'bc_choices.json', {'rule': 'argmin native table C[node][a] (input + output); B(w) is a common constant',
                                    'rows': bc_rows, 'q_BC': q_bc})
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    prior = {k: {s['content_sha256'] for s in read(d / 'segments.json')} for k, d in (('spike', SPIKE), ('fp16', FP16), ('l27', L27R))}
    chk['prior_articles_in_scan'] = {k: (len(v), v <= used) for k, v in prior.items()}
    if not all(ok for _, ok in chk['prior_articles_in_scan'].values()):
        raise SystemExit('earlier Qwen articles are not all in the verified scan')
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    est = read(smoke) if smoke else None
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(),
        'questions': ['does quantizing only the earlier blocks change the measured relative ranking of later Down activation '
                      'candidates?', 'how does the original BC ranking fare in both backgrounds (wrong before vs wrong only '
                      'after the background change)?', 'does a calibration choice in the new background beat the old-background '
                      'choice on unseen articles, by enough to justify a cheap re-scoring study?'],
        'boundaries': 'no change to BC, search, knee, ranges, shared quantizers or weights; no fine-tuning; no spike / first-order / '
                      'FD / Hessian criterion; no BC statistics recomputed; the new-background calibration score is a diagnostic '
                      'reference, not a refreshed BC; GPU FP32 + QDQ only (no native / low-bit kernel claim); conclusions limited '
                      'to the Down activation of this model, these candidates and backgrounds',
        'structure_checks': chk,
        'backgrounds': {'U': 'blocks < i FP32; block i other projections FP32; blocks > i FP32',
                        'Q': 'blocks < i: every searched projection (attention and FFN) as A (w and a); block i other projections '
                             'FP32; blocks > i FP32; norms / embedding / LM head unchanged',
                        'target': 'Down of block i: weight A[w] (bitwise identical in U and Q and for every setting), activation = '
                                  'setting on input x and output y (LMReplay / native semantics), fixed native ranges and Posit LUT',
                        'L0': 'empty prefix: U = Q, negative control, excluded from the cross-layer statistics',
                        'layer_order': 'block ids parsed as integers from node_mapping.json'},
        'settings': list(SETTINGS), 'conditions': f'{NL} layers x 2 backgrounds x {len(SETTINGS)} settings = {NL * 2 * len(SETTINGS)}',
        'pairs': {'primary': [f'{a}-{b}' for a, b in PAIRS], 'primary_count': (NL - 1) * len(PAIRS),
                  'secondary': 'f16 vs p8e1 for the 27 non-control layers (different resources: not an equal-cost ranking)',
                  'other': 'other cross-bit comparisons descriptive only'},
        'selectors': {'q_BC': 'argmin native table C[node][a] (input + output), frozen in bc_choices.json',
                      'q_U': 'argmin calibration NLL in background U (10 original windows)',
                      'q_Q': 'argmin calibration NLL in background Q (same windows)',
                      'ties': f'BC proxy tie |C1-C2| <= {PROXY_TIE} max(|C1|,|C2|,1e-30); calibration numeric tie |dNLL| <= {NUM_TIE} '
                              'nat/token; a tie picks A\'s original a if it is in the pair, else the INT format; never new articles',
                      'low_margin': f'calibration |dNLL| < {LOW_MARGIN}: flagged, never removed; full rankings and scores kept',
                      'note': 'q_Q is not the true best candidate nor a new criterion'},
        'formulas': {'E_s(i,f,k)': 'mean NLL of article k (2 windows, 254 targets) in background s, target layer i, setting f',
                     'd_s': 'E_s(i,f,k) - E_s(i,fp32,k)', 'D_s(i,b,k)': 'E_s(i,INT_b,k) - E_s(i,Posit_b,k) (positive: Posit better)',
                     'J(i,b,k)': 'D_Q - D_U (a non-zero J is not a reversal unless the signs differ)',
                     'G_shift(k)': 'mean over the 81 (i,b) of E_Q(i,q_U,k) - E_Q(i,q_Q,k)',
                     'G_BC(k)': 'mean over the 81 (i,b) of E_Q(i,q_BC,k) - E_Q(i,q_Q,k)',
                     'I': 'single-point interventions averaged; never summed into a whole-configuration or knee gain'},
        'classification': 'per pair vs the BC prediction: both right / right then wrong / wrong then right / both wrong / proxy tie / '
                          'undetermined (95% CI of D covers 0 in a background); accuracies with denominators, undetermined counts, '
                          'coverage, and on the common decisive set',
        'statistics': {'unit': 'article (2 windows averaged first)', 'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}), '
                       'one index set for every background, setting and selector',
                       'intervals': '95% percentile CI for every D_U, D_Q, J; G_shift and G_BC also Bonferroni 97.5% (two primary metrics)',
                       'reversal_family': 'Holm (alpha .05) over 162 tests (81 pairs x D_U, D_Q): null-centred bootstrap two-sided p = '
                                          '(1 + #|T*| >= |mean|) / (B + 1), T* = bootstrap mean - observed mean; exactly zero differences '
                                          '-> p = 1; Holm-supported reversal = opposite signs and both Holm-significant',
                       'J_family': 'Holm over the 81 J tests (secondary; a significant J is not a reversal)',
                       'secondary_family': 'f16 vs p8e1: Holm over its 54 D tests and 27 J tests, separate',
                       'labels': 'observed reversal (point signs) / uncorrected-CI supported / multiple-comparison supported'},
        'epsilon': f'{EPS} nat/token: research threshold for the mean single-point gain (previous rounds\' scale), not tuned on results',
        'decision': {'G_shift 97.5% lower > eps': 'supports studying background-aware re-scoring in this scope; if G_BC also passes, also a '
                                                  'practical gap relative to the old BC',
                     'only G_BC passes': 'likely proxy ranking error; not attributable to prefix-induced rank change',
                     'credible reversals but G_shift upper < eps': 'mean gain below threshold; keep local cases; no whole-model re-scoring; '
                                                                   'interaction not ruled out',
                     'CI straddles eps or q_Q does not improve unseen data': 'undetermined / this calibration selector has no validated '
                                                                             'gain; not that every re-scoring fails',
                     'concentrated in L27 or few historical cases': 'local follow-up only',
                     'never': 'start a refresh, a whole-model search, or combine choices into a new configuration in this round'},
        'data': {'calibration': 'original 10 x 128 windows (native calib_inputs.npy): selection and numeric controls only, not 10 articles',
                 'diagnostic': f'{N_ART} new WikiText-103 train articles x {WPA} contiguous non-overlapping 128-token windows (254 targets)',
                 'draw': f'eligible = strict-boundary articles, not a WikiText-2 title / content hash, unique, not in '
                         f'wt_manifest_scan.used_wt103_verified ({len(used)} articles, including spike {len(prior["spike"])}, fp16 '
                         f'{len(prior["fp16"])}, l27 {len(prior["l27"])}); order seed {SEEDS["order"]}; skip < {WPA * SEQ} tokens; skip '
                         '>= 10% shingle share with one WikiText-2 article, a previous WikiText-103 article or an accepted article; skip '
                         '>= 10% overlap with any historical Qwen token window (audit rerun on runs/; unresolved source stops); start '
                         f'uniform (seed {SEEDS["start"]}); NEW: reject the article if either window has >= {WT2_WINDOW_MAX:.0%} of its '
                         'word 5-gram shingles in the WikiText-2 text; all rules applied before any loss',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig)},
        'controls': ['per-condition configuration, quantized-projection count per block and hash; Q quantizes only block id < i',
                     'target weight identical across U / Q / settings; every module weight is the expected cached tensor (original FP32 '
                     'or A-quantized from the original) after every switch; no re-quantization of quantized weights',
                     'L0 U = Q bitwise per window; all-FP32 replay = unhooked model bitwise; returning to a configuration reproduces bitwise',
                     'fp32 setting disables only the target activation QDQ (target w and the prefix kept)',
                     'one pre-hook and one forward hook per projection; target x / y are fixed points of the expected quantizer (fp32: not '
                     'f16-representable); int scale / zp on the device; TF32 off; float32; deterministic; Posit LUT verified',
                     'fixed native ranges; clipping / saturation statistics of the target x / y are descriptive only',
                     'non-finite values kept as failures; an incomplete main matrix stops the primary decision',
                     f'every condition on the calibration windows: replay from the prefix cache vs a complete forward, |diff| <= {TOL_REPLAY} '
                     'summed CE per window; the cache keys background, layer, data and configuration hash',
                     'BC table C (input + output) is the BC reference; no input-only score; no claim that U reproduces the BC statistics',
                     'tolerances fixed here; a failure stops before the new articles'],
        'checkpoint': 'per chunk of windows: CE array + header (tokens, script, A, ranges, table, choices, chunk range); a resumed chunk '
                      'must match; no early stop or extension by effect size',
        'time_estimate': est, 'seeds': SEEDS,
        'inputs': {rel(p): sha(p) for p in input_files()}, 'code_hashes': {f: sha(HERE / f) for f in CODE},
        'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：Qwen3 前層量化是否改變後層 Down activation 候選排名', '',
          f'時間：{pl["time"]}。以 plan.json 為準；寫入後的程式修改須記錄 amendment。依 `reports/qwen3_prefix_rank_shift_prompt_20261024.md` 實作。', '',
          '- 邊界：不改 BC、搜尋器、knee、range、共享量化函式、權重；不 fine-tune；不新增 criterion；不重算 BC 統計。新背景的 calibration 實測評分只是診斷參照，不是 refreshed BC。GPU FP32 + QDQ，不宣稱 native。',
          f'- 結構核對：A sha256 = {A_SHA[:16]}…，與 ref.json 相同；config {NL} 層；mapping 196 投影、每 block 7 個；層號由 mapping 解析為整數。',
          '- 背景：U = 前層 FP32；Q = blocks < i 的全部搜尋投影（attention 與 FFN）依 A；兩者的 block i 其他投影與 blocks > i 皆 FP32。目標 Down 權重兩背景皆為 A 的 w（逐位元相同）；activation 設定作用於 input x 與 output y，ranges／LUT 固定。L0 為負控制。',
          f'- 設定：{", ".join(SETTINGS)}；條件 {NL}×2×{len(SETTINGS)} = {NL * 2 * len(SETTINGS)}。主要比較 27 層 × 3 同位元對 = 81；f16 vs p8e1 為次要（資源不同）。',
          f'- selector（正式評估前凍結）：q_BC = argmin native C[node][a]；q_U／q_Q = 原 10 個 calibration 窗口上 U／Q 的實測 NLL 最小者。tie：BC 相對差 ≤ {PROXY_TIE}、calibration 差 ≤ {NUM_TIE} nat/token → 選 A 原 a（若在對內）否則 INT；差 < {LOW_MARGIN} 標 low margin 不刪除。',
          '- 公式：D_s = E_s(INT) − E_s(Posit)（正值 Posit 較好）；J = D_Q − D_U；G_shift = mean[E_Q(q_U) − E_Q(q_Q)]；G_BC = mean[E_Q(q_BC) − E_Q(q_Q)]（81 對單點介入的平均，不是配置收益）。',
          f'- 資料：{N_ART} 篇新 WT103 文章 × {WPA} 個連續不重疊窗口（254 targets）；verified 掃描排除 {len(used)} 篇（含 spike／fp16／l27 實驗）；排除 WT2 與近重複、歷史 Qwen 窗口重疊；新增：任一窗口與 WT2 5-gram overlap ≥ 50% 即拒收該文章；seed 順序 {SEEDS["order"]}、起點 {SEEDS["start"]}；合格 {len(elig)} 篇。',
          f'- 統計：文章為單位，paired bootstrap {BOOT} 次（seed {SEEDS["bootstrap"]}）同一組索引；D／J 報 95% CI；G_shift、G_BC 報 95% 與 Bonferroni 97.5% CI；反轉 Holm family = 162 個 D 檢驗（零中心化 bootstrap 雙尾 p）；J 81 個另一 family；f16/p8e1 另一 family。',
          f'- ε = {EPS} nat/token。決策：G_shift 97.5% 下界 > ε → 支持下一輪背景相關重新評分研究（G_BC 也過才同時支持相對舊 BC 有改善空間）；只有 G_BC → 可能是 proxy 誤差；少數可信反轉但 G_shift 上界 < ε → 保留局部；跨 ε → 未定；集中於 L27／少數舊案例 → 只局部追查。不得啟動刷新、搜尋或組新配置。',
          f'- 控制：每條件 config／hash／每 block 量化數；權重指標核對；L0 U=Q、FP32 identity = 無 hook、重切可重現；fp32 設定只關目標 a；hook 數；目標 x／y 格式；TF32 關閉、deterministic；ranges 固定；非有限值保留；calibration 上每條件「重播 vs 完整前向」≤ {TOL_REPLAY}。失敗不放寬。',
          '- checkpoint：每 chunk 存 CE 與 hash 標頭，續跑須相符；不依效應大小停樣本或擴樣本。',
          f'- 時間估計（冒煙測試，calibration）：{json.dumps(est, ensure_ascii=False) if est else "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; checks {chk}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# GPU replay
# --------------------------------------------------------------------------- #
def make_replay(work):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from run_lp_pipeline_eval import LMReplay

    class Replay(LMReplay):
        """LMReplay with device-cached weights: the original FP32 tensors and per-format quantizations of the original
        (CPU) weights, assigned by reference on every switch (no copy, never re-quantizing a quantized weight)."""

        def __init__(self, work):
            super().__init__('qwen3', 'cuda', work)
            self.dev_w = {(n, 'fp32'): self.mod(n).weight.data for n in self.param}

        def wdev(self, n, w):
            key = (n, w or 'fp32')
            if key not in self.dev_w:
                self.dev_w[key] = self.q.weight(self.orig[n], key[1]).to(self.device)
            return self.dev_w[key]

        def set(self, cfg):
            for n in self.param:
                w, a = (cfg[n]['w'], cfg[n]['a']) if n in cfg else ('fp32', 'fp32')
                self.mod(n).weight.data = self.wdev(n, w)
                self.act[n] = a

        def weights_ok(self, cfg):
            """every module weight shares the storage of the expected cached tensor; activation map = configuration."""
            return all(self.mod(n).weight.data_ptr() == self.dev_w[(n, (cfg[n]['w'] or 'fp32') if n in cfg else 'fp32')].data_ptr()
                       and self.act[n] == (cfg[n]['a'] if n in cfg else 'fp32') for n in self.param)
    t0 = time.perf_counter()
    rp = Replay(work)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rp.layers = rp.ad.model.model.layers
    rp.kw = {}
    return rp, time.perf_counter() - t0


def head_ce(rp, normed, ids):
    torch = rp.torch
    out = []
    for j in range(0, normed.shape[0], LM_SUB):
        z = rp.ad.model.lm_head(normed[j:j + LM_SUB])
        ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids[j:j + LM_SUB, 1:].reshape(-1),
                                               reduction='none')
        out.append(ce.view(z.shape[0], -1).double().sum(1))
        del z
    return torch.cat(out)


def forward_capture(rp, ids, capture=True):
    """complete forward under the current configuration; returns per-window summed CE and the hidden state entering
    every block (the prefix cache), with the decoder-layer keyword arguments."""
    torch = rp.torch
    hs, hooks = {}, []
    if capture:
        for b, layer in enumerate(rp.layers):
            hooks.append(layer.register_forward_pre_hook(lambda m, a, b=b: hs.__setitem__(b, a[0].detach())))
        hooks.append(rp.layers[1].register_forward_pre_hook(lambda m, a, k: rp.kw.update(k), with_kwargs=True))
    with torch.no_grad():
        h = rp.ad.model.model(input_ids=ids, attention_mask=rp.ad.mask, use_cache=False).last_hidden_state
        ce = head_ce(rp, h, ids)
    for hk in hooks:
        hk.remove()
    return ce, hs


def replay_ce(rp, hs_i, i, ids):
    torch = rp.torch
    with torch.no_grad():
        h = hs_i
        for layer in rp.layers[i:]:
            o = layer(h, **rp.kw)
            h = o[0] if isinstance(o, tuple) else o
        return head_ce(rp, rp.ad.model.model.norm(h), ids)


# --------------------------------------------------------------------------- #
# calibration: controls, full matrix, selectors
# --------------------------------------------------------------------------- #
def calib(out):
    t0 = time.perf_counter()
    pl = check_plan(out)
    layer_of, down, A = structure()
    rp, init_s = make_replay(out / 'work')
    torch = rp.torch
    ranges = rp.ranges
    from run_olmoe_dynamic_route import env_info
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ids = torch.from_numpy(toks).to(rp.device)
    ctl = {'environment': env_info(), 'replay_init_s': init_s, 'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
           'deterministic': torch.are_deterministic_algorithms_enabled(), 'matmul_precision': torch.get_float32_matmul_precision(),
           'posit_lut_verified': {f: list(rp.q.posit[f].verify()) for f in rp.q.posit},
           'ranges_equal_native': ranges == read(NATIVE / 'ranges_by_point.json'),
           'calibration_tokens_equal_prepare': bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL])),
           'rms_norm_eps': sorted({float(m.variance_epsilon) for m in rp.mods.values() if hasattr(m, 'variance_epsilon')})}
    hooks_now = lambda: dict(Counter(f'{len(rp.mod(n)._forward_pre_hooks)},{len(rp.mod(n)._forward_hooks)}' for n in rp.param))
    ctl['hooks_initial'] = hooks_now()
    # unhooked model vs the all-FP32 replay model
    from transformers import Qwen3ForCausalLM
    import qwen3_swiglu_common as q3
    plain = Qwen3ForCausalLM.from_pretrained(q3.MODEL_DIR, dtype=torch.float32, attn_implementation='eager').eval().to(rp.device)
    with torch.no_grad():
        ce_plain = head_ce(rp, plain.model(input_ids=ids, attention_mask=rp.ad.mask, use_cache=False).last_hidden_state, ids).cpu().numpy()
    del plain
    torch.cuda.empty_cache()
    t1 = time.perf_counter()
    rp.set({})
    ce_fp32, hsU = forward_capture(rp, ids)
    rp.set(A)
    if not rp.weights_ok(A):
        raise SystemExit('weights after set(A) differ from the expected cached tensors')
    ce_A, hsQ = forward_capture(rp, ids)
    weight_prep_s = time.perf_counter() - t1
    ctl['fp32_identity_equals_unhooked'] = bool(np.array_equal(ce_fp32.cpu().numpy(), ce_plain))
    ctl['L0_prefix_U_equals_Q'] = bool(torch.equal(hsU[0], hsQ[0]))
    ctl['tensor_hashes'] = {f'{n}|{w}': hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                            for (n, w), t in rp.dev_w.items() if n in down.values()}
    # diagnostic hooks on the target Down modules (registered after LMReplay's, so they see the QDQ'd tensors)
    seen = {}
    diag = []
    for i in range(NL):
        m = rp.mod(down[i])
        diag.append(m.register_forward_pre_hook(lambda mod, a, i=i: seen.__setitem__((i, 'x'), a[0].detach())))
        diag.append(m.register_forward_hook(lambda mod, a, o, i=i: seen.__setitem__((i, 'y'), o.detach())))

    def fixed(t, f, rng):
        return bool(torch.equal(rp.qact(t, f, rng), t))
    res = {s: np.full((NL, len(SETTINGS), CAL), np.nan) for s in BACKS}
    full = {s: np.full((NL, len(SETTINGS), CAL), np.nan) for s in BACKS}
    fmt_bad, wbad, cfg_rows, clip = [], [], [], {}
    t_replay = t_full = 0.0
    for i in LAYERS_RUN:
        n = down[i]
        for s in BACKS:
            hs = (hsU if s == 'U' else hsQ)[i]
            for fi, f in enumerate(SETTINGS):
                cfg = condition_cfg(A, layer_of, down, i, s, f)
                rp.set(cfg)
                if not rp.weights_ok(cfg):
                    wbad.append(f'L{i}|{s}|{f}')
                quantized = Counter(layer_of[x][0] for x in cfg if x != n)
                cfg_rows.append({'layer': i, 'background': s, 'setting': f, 'hash': cfg_hash(cfg), 'prefix_quantized_projections': sum(quantized.values()),
                                 'max_prefix_block': max(quantized) if quantized else None, 'per_block': dict(sorted(quantized.items())),
                                 'target': {n: cfg[n]}})
                if sum(quantized.values()) != (7 * i if s == 'Q' else 0) or (quantized and max(quantized) >= i):
                    raise SystemExit(f'L{i}|{s}: prefix wrong {dict(quantized)}')
                t1 = time.perf_counter()
                seen.clear()
                res[s][i, fi] = replay_ce(rp, hs, i, ids).cpu().numpy()
                t_replay += time.perf_counter() - t1
                t1 = time.perf_counter()
                seen.clear()
                full[s][i, fi] = forward_capture(rp, ids, capture=False)[0].cpu().numpy()
                t_full += time.perf_counter() - t1
                x, y = seen[(i, 'x')], seen[(i, 'y')]
                if f == 'fp32':
                    ok = not fixed(x, 'f16', None) and not fixed(y, 'f16', None)
                    lo_x, hi_x = ranges[n]['x']
                    lo_y, hi_y = ranges[n]['y']
                    clip[f'L{i}|{s}'] = {'x_outside_range': float(((x < lo_x) | (x > hi_x)).float().mean()), 'y_outside_range': float(((y < lo_y) | (y > hi_y)).float().mean()),
                                         'x_abs_gt_4096': float((x.abs() > 4096).float().mean()), 'y_abs_gt_4096': float((y.abs() > 4096).float().mean()),
                                         'x_absmax': float(x.abs().max()), 'y_absmax': float(y.abs().max())}
                else:
                    ok = fixed(x, f, ranges[n]['x']) and fixed(y, f, ranges[n]['y'])
                if not ok:
                    fmt_bad.append(f'L{i}|{s}|{f}')
        log(f'L{i}: replay {t_replay:.0f}s, full {t_full:.0f}s')
    for hk in diag:
        hk.remove()
    # switching back reproduces
    i_r = LAYERS_RUN[len(LAYERS_RUN) // 2]
    cfg = condition_cfg(A, layer_of, down, i_r, 'Q', 'int4')
    rp.set(cfg)
    rep = replay_ce(rp, hsQ[i_r], i_r, ids).cpu().numpy()
    ctl['switch_back_reproduces'] = bool(np.array_equal(rep, res['Q'][i_r, SETTINGS.index('int4')]))
    diffs = np.abs(np.concatenate([(res[s] - full[s])[LAYERS_RUN].ravel() for s in BACKS]))
    ctl.update({'replay_vs_full_max_abs': float(np.nanmax(diffs)), 'replay_vs_full_exact_fraction': float((diffs == 0).mean()),
                'L0_U_equals_Q_all_settings': bool(0 not in LAYERS_RUN or np.array_equal(res['U'][0], res['Q'][0])),
                'nonfinite': int(sum((~np.isfinite(res[s][LAYERS_RUN])).sum() for s in BACKS)),
                'weights_mismatch_conditions': wbad, 'format_mismatch_conditions': fmt_bad, 'hooks_final': hooks_now(),
                'target_weight_identical': all(len({rp.dev_w[(down[i], A[down[i]]['w'])].data_ptr()}) == 1 for i in range(NL)),
                'tolerance_replay': TOL_REPLAY})
    ctl['passed'] = bool(ctl['tf32'] == [False, False] and ctl['deterministic'] and all(v[0] == 0 for v in ctl['posit_lut_verified'].values())
                         and ctl['ranges_equal_native'] and ctl['calibration_tokens_equal_prepare'] and ctl['rms_norm_eps'] == [1e-6]
                         and ctl['fp32_identity_equals_unhooked'] and ctl['L0_prefix_U_equals_Q'] and ctl['L0_U_equals_Q_all_settings']
                         and ctl['switch_back_reproduces'] and ctl['replay_vs_full_max_abs'] <= TOL_REPLAY and ctl['nonfinite'] == 0
                         and not wbad and not fmt_bad and ctl['hooks_initial'] == ctl['hooks_final'] == {'1,1': 196})
    write(out / 'controls.json', ctl)
    (out / 'configs').mkdir(exist_ok=True)
    with open(out / 'configs' / 'conditions.jsonl', 'w') as fh:
        for r in cfg_rows:
            fh.write(json.dumps(r) + '\n')
    write(out / 'configs' / 'backgrounds.json', {'A_sha256': A_SHA, 'Q_prefix_rule': 'A restricted to block id < i', 'U_prefix_rule': 'none',
                                                 'target_rule': 'A[w] at the target Down, activation = setting',
                                                 'conditions': len(cfg_rows), 'conditions_sha256': sha(out / 'configs' / 'conditions.jsonl')})
    # calibration scores and selectors
    E = {s: res[s].sum(-1) / (CAL * TP) for s in BACKS}
    scores, choices, low = {}, {'q_BC': read(out / 'bc_choices.json')['q_BC'], 'q_U': {}, 'q_Q': {}}, {}
    for i in LAYERS_RUN:
        scores[f'L{i}'] = {s: {f: float(E[s][i, fi]) for fi, f in enumerate(SETTINGS)} for s in BACKS}
        for a, b in PAIRS:
            key = f'L{i}|{a}-{b}'
            for s in BACKS:
                sc = scores[f'L{i}'][s]
                tie = abs(sc[a] - sc[b]) <= NUM_TIE
                choices[f'q_{s}'][key] = pick((a, b), sc, A[down[i]]['a'], tie)
                low[f'{key}|{s}'] = {'diff_int_minus_posit': sc[a] - sc[b], 'numeric_tie': bool(tie), 'low_margin': bool(abs(sc[a] - sc[b]) < LOW_MARGIN)}
    write(out / 'calibration_scores.json', {'unit': 'mean NLL per token over the 10 calibration windows', 'scores': scores,
                                            'per_window_summed_ce': {s: res[s].tolist() for s in BACKS},
                                            'complete_forward_summed_ce': {s: full[s].tolist() for s in BACKS},
                                            'model_fp32_summed_ce': ce_fp32.cpu().numpy().tolist(), 'model_A_summed_ce': ce_A.cpu().numpy().tolist(),
                                            'clipping_descriptive': clip})
    write(out / 'choices.json', {**choices, 'margins': low, 'differ_q_U_q_Q': sorted(k for k in choices['q_U'] if choices['q_U'][k] != choices['q_Q'][k])})
    log_stage(out, 'calib', time.perf_counter() - t0, {'replay_init_s': init_s, 'weight_and_prefix_s': weight_prep_s, 'replay_s': t_replay,
                                                       'complete_forward_s': t_full, 'conditions': len(cfg_rows), 'windows': CAL,
                                                       'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated())})
    log(f'controls passed={ctl["passed"]}; replay vs full max {ctl["replay_vs_full_max_abs"]:.3g}; q_U != q_Q: {len(read(out / "choices.json")["differ_q_U_q_Q"])}')
    if not ctl['passed']:
        raise SystemExit('calibration controls failed (see controls.json)')


def freeze(out):
    pl = check_plan(out)
    if (out / 'freeze.json').exists():
        raise SystemExit('frozen')
    ctl = read(out / 'controls.json')
    if not ctl['passed']:
        raise SystemExit('controls failed')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls.json'),
                                'bc_choices_sha256': sha(out / 'bc_choices.json'), 'choices_sha256': sha(out / 'choices.json'),
                                'calibration_scores_sha256': sha(out / 'calibration_scores.json'),
                                'conditions_sha256': sha(out / 'configs' / 'conditions.jsonl'), 'script_sha256': sha(HERE / 'run_qwen_prefix_rank_shift.py')})
    log('frozen')


# --------------------------------------------------------------------------- #
# audit and data
# --------------------------------------------------------------------------- #
def audit(out):
    import run_qwen_down_fp16_protection as source
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    source.check_plan = check_plan
    source.log_stage = lambda o, s, sec, extra=None: None
    source.audit(out, log)
    au = read(out / 'audit.json')
    if au['unresolved_files']:
        raise SystemExit(f'unresolved historical Qwen token sources: {au["unresolved_files"]}')
    log_stage(out, 'audit', time.perf_counter() - t0)


def data(out):
    import pyarrow.parquet as pq
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_qwen_down_spike_diag import qwen3_tokenizer, window_shingles
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    pl = check_plan(out)
    au = read(out / 'audit.json')
    if sha(out / 'work' / 'qwen_hist_shingles.npz') != au['index_sha256']:
        raise SystemExit('audit index changed')
    hz = np.load(out / 'work' / 'qwen_hist_shingles.npz')
    hist = defaultdict(set)
    wsize = Counter(hz['window'].tolist())
    for s_, w in zip(hz['shingle'].tolist(), hz['window'].tolist()):
        hist[s_].add(w)
    U2 = set()
    for f in mr.WT2:
        U2 |= window_shingles(' '.join(pq.read_table(f)['text'].to_pylist()))
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    if len(used) != pl['data']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    lines = mr.load_lines(mr.WT103)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['data']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    by = {a['ch']: a for a in w103}
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(by[h_]['body_norm']) for h_ in used]
    tok = qwen3_tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    segs, toks, acc_sh, skips, excl = [], [], [], Counter(), []
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        enc = tok.encode(text, add_special_tokens=False)
        ids, off = enc.ids, enc.offsets
        if len(ids) < WPA * SEQ:
            skips['short'] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': 'short', 'tokens': len(ids)})
            continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        hw = Counter(w for s_ in sh for w in hist.get(s_, ()))
        qw = max((c / wsize[w] for w, c in hw.items()), default=0.0)
        qa = sum(1 for s_ in sh if s_ in hist) / len(sh) if sh else 0.0
        shares = {'wt2': c2, 'previous': cp, 'accepted': ca, 'qwen_window': qw, 'qwen_article': qa}
        reason = ('near_duplicate_wt2' if c2 >= mr.NEARDUP else 'near_duplicate_previous' if cp >= mr.NEARDUP else
                  'near_duplicate_accepted' if ca >= mr.NEARDUP else 'historical_qwen_overlap' if max(qw, qa) >= mr.NEARDUP else None)
        if reason:
            skips[reason] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': reason, 'shares': shares})
            continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        wshare = []
        for row in w:
            s2 = window_shingles(tok.decode(row.tolist()))
            wshare.append(len(s2 & U2) / len(s2) if s2 else 0.0)
        if max(wshare) >= WT2_WINDOW_MAX:
            skips['window_wt2_overlap'] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': 'window_wt2_overlap', 'shares': shares, 'window_wt2_share': wshare})
            continue
        toks.append(w)
        acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']],
                     'title': a['title'], 'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                     'window_tokens': [st, st + WPA * SEQ], 'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                     'max_shingle_share': shares, 'window_wt2_share': wshare, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
                     'windows': [WPA * len(segs) + i for i in range(WPA)]})
        if len(segs) == N_ART:
            break
    got = {s_['content_sha256'] for s_ in segs}
    prior = set().union(*[{s['content_sha256'] for s in read(d / 'segments.json')} for d in (SPIKE, FP16, L27R)])
    if len(segs) < N_ART or got & used or got & prior:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'exclusions.json', {'skipped_during_draw': excl, 'previous_wt103_excluded_by_scan': sorted(used),
                                    'wt2_and_internal_duplicates_excluded_before_draw': len(w103) - len(base)})
    write(out / 'data_freeze.json', {'time': now(), 'freeze_sha256': sha(out / 'freeze.json'), 'audit_sha256': sha(out / 'audit.json'),
                                     'n_articles': N_ART, 'windows_per_article': WPA, 'eligible': len(elig), 'visited': N_ART + sum(skips.values()),
                                     'skips': dict(skips), 'excluded_previous_wt103': len(used), 'tokens_sha256': sha(out / 'tokens.npy'),
                                     'segments_sha256': sha(out / 'segments.json'), 'exclusions_sha256': sha(out / 'exclusions.json')})
    log_stage(out, 'data', time.perf_counter() - t0)
    log(f'data: {N_ART} articles; skips {dict(skips)}')


# --------------------------------------------------------------------------- #
# eval (GPU, chunked, resumable)
# --------------------------------------------------------------------------- #
def verify_frozen(out):
    pl = check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    for k, f in (('controls_sha256', 'controls.json'), ('choices_sha256', 'choices.json'), ('bc_choices_sha256', 'bc_choices.json'),
                 ('calibration_scores_sha256', 'calibration_scores.json')):
        if sha(out / f) != fz[k]:
            raise SystemExit(f'{f} changed after the freeze')
    if sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'segments.json') != dz['segments_sha256']:
        raise SystemExit('data changed after the freeze')
    return pl, fz, dz


def eval_(out):
    t0 = time.perf_counter()
    pl, fz, dz = verify_frozen(out)
    layer_of, down, A = structure()
    rp, init_s = make_replay(out / 'work')
    torch = rp.torch
    toks = np.load(out / 'tokens.npy')
    nw = len(toks)
    hdr_base = {'tokens_sha256': dz['tokens_sha256'], 'script_sha256': sha(HERE / 'run_qwen_prefix_rank_shift.py'), 'A_sha256': A_SHA,
                'ranges_sha256': sha(NATIVE / 'ranges_by_point.json'), 'table_sha256': sha(NATIVE / 'table.json'),
                'choices_sha256': fz['choices_sha256'], 'settings': list(SETTINGS), 'layers': LAYERS_RUN}
    tm = defaultdict(float)
    for k, c0 in enumerate(range(0, nw, CHUNK)):
        ck = out / 'work' / f'eval_chunk{k:02d}.npz'
        hdr = hdr_base | {'chunk': k, 'windows': [c0, min(nw, c0 + CHUNK)]}
        if ck.exists():
            z = np.load(ck)
            if json.loads(str(z['header'])) != hdr:
                raise SystemExit(f'{ck}: checkpoint from another version / data')
            log(f'chunk {k}: reusing checkpoint')
            continue
        t1 = time.perf_counter()
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).to(rp.device)
        rp.set({})
        ce_fp32, hsU = forward_capture(rp, ids)
        rp.set(A)
        ce_A, hsQ = forward_capture(rp, ids)
        if not torch.equal(hsU[0], hsQ[0]):
            raise SystemExit('L0 prefix differs between U and Q')
        tm['prefix_forwards'] += time.perf_counter() - t1
        ce = np.full((2, NL, len(SETTINGS), len(ids)), np.nan)
        t1 = time.perf_counter()
        for i in LAYERS_RUN:
            for si, s in enumerate(BACKS):
                if i == 0 and s == 'Q':                     # empty prefix: identical inputs, verified above and on calibration
                    ce[1, 0] = ce[0, 0]
                    continue
                hs = (hsU if s == 'U' else hsQ)[i]
                for fi, f in enumerate(SETTINGS):
                    cfg = condition_cfg(A, layer_of, down, i, s, f)
                    rp.set(cfg)
                    ce[si, i, fi] = replay_ce(rp, hs, i, ids).cpu().numpy()
        tm['replay'] += time.perf_counter() - t1
        np.savez(ck, ce=ce, model_fp32=ce_fp32.cpu().numpy(), model_A=ce_A.cpu().numpy(), header=json.dumps(hdr))
        del hsU, hsQ
        log(f'chunk {k + 1}/{-(-nw // CHUNK)}: {time.perf_counter() - t1:.0f}s replay; nonfinite {int((~np.isfinite(ce)).sum())}')
    parts = [np.load(out / 'work' / f'eval_chunk{k:02d}.npz') for k in range(-(-nw // CHUNK))]
    ce = np.concatenate([p['ce'] for p in parts], axis=-1)
    np.savez(out / 'eval_ce.npz', ce=ce, model_fp32=np.concatenate([p['model_fp32'] for p in parts]),
             model_A=np.concatenate([p['model_A'] for p in parts]), backgrounds=np.array(BACKS), settings=np.array(SETTINGS))
    log_stage(out, 'eval', time.perf_counter() - t0, {'replay_init_s': init_s, **tm, 'windows': nw, 'chunk': CHUNK,
                                                      'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
                                                      'nonfinite': int((~np.isfinite(ce)).sum()), 'eval_ce_sha256': sha(out / 'eval_ce.npz')})


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for r, j in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[j]))
        adj[j] = run
    return adj


def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    pl, fz, dz = verify_frozen(out)
    layer_of, down, A = structure()
    z = np.load(out / 'eval_ce.npz')
    ce = z['ce']                                              # [2, 28, 8, windows]
    segs = read(out / 'segments.json')
    na = len(segs)
    complete = bool(np.isfinite(ce).all())
    E = ce.reshape(2, NL, len(SETTINGS), na, WPA).sum(-1) / (WPA * TP)       # [2, 28, 8, articles]
    si, fi = {s: k for k, s in enumerate(BACKS)}, {f: k for k, f in enumerate(SETTINGS)}
    ch = read(out / 'choices.json')
    bc = {r['layer'] * 10 + PAIRS.index(tuple(r['pair'].split('-'))): r for r in read(out / 'bc_choices.json')['rows']}
    rng = np.random.default_rng(SEEDS['bootstrap'])
    W = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na

    def st(x, levels=(0.05,)):
        b = W @ x
        r = {'mean': float(x.mean()), 'se': float(x.std(ddof=1) / np.sqrt(na))}
        for lv in levels:
            r[f'ci{100 * (1 - lv):g}'] = np.percentile(b, [100 * lv / 2, 100 * (1 - lv / 2)]).tolist()
        if np.all(x == 0):
            r['p_null_centred'] = 1.0
        else:
            r['p_null_centred'] = float((1 + np.sum(np.abs(b - x.mean()) >= abs(x.mean()))) / (BOOT + 1))
        r['decisive'] = bool(r['ci95'][0] > 0 or r['ci95'][1] < 0)
        return r
    rows, Dtests, Jtests, Ftests, FJtests = [], [], [], [], []
    for i in range(1, NL):
        for a, b in PAIRS:
            key = f'L{i}|{a}-{b}'
            D = {s: E[si[s], i, fi[a]] - E[si[s], i, fi[b]] for s in BACKS}
            Ds = {s: st(D[s]) for s in BACKS}
            J = st(D['Q'] - D['U'])
            r_bc = bc[i * 10 + PAIRS.index((a, b))]
            pred = 1 if r_bc['C_int'] > r_bc['C_posit'] else -1          # sign of D predicted by BC (+: Posit better)
            right = {s: bool(Ds[s]['decisive'] and not r_bc['proxy_tie'] and np.sign(Ds[s]['mean']) == pred) for s in BACKS}
            if r_bc['proxy_tie']:
                cls = 'proxy tie'
            elif not (Ds['U']['decisive'] and Ds['Q']['decisive']):
                cls = 'undetermined (' + '+'.join(s for s in BACKS if not Ds[s]['decisive']) + ')'
            else:
                cls = {(True, True): 'both right', (True, False): 'right then wrong', (False, True): 'wrong then right',
                       (False, False): 'both wrong'}[(right['U'], right['Q'])]
            row = {'layer': i, 'pair': f'{a}-{b}', 'bits': int(a[3:]), 'C_int': r_bc['C_int'], 'C_posit': r_bc['C_posit'], 'bc_pred_sign': pred,
                   'proxy_tie': r_bc['proxy_tie'], 'q_BC': ch['q_BC'][key], 'q_U': ch['q_U'][key], 'q_Q': ch['q_Q'][key],
                   'cal_margin_U': ch['margins'][f'{key}|U']['diff_int_minus_posit'], 'cal_margin_Q': ch['margins'][f'{key}|Q']['diff_int_minus_posit'],
                   'low_margin_U': ch['margins'][f'{key}|U']['low_margin'], 'low_margin_Q': ch['margins'][f'{key}|Q']['low_margin'],
                   **{f'D_{s}_{k}': v for s in BACKS for k, v in Ds[s].items()}, **{f'J_{k}': v for k, v in J.items()},
                   'bc_right_U': right['U'], 'bc_right_Q': right['Q'], 'class': cls,
                   'observed_reversal': bool(np.sign(Ds['U']['mean']) * np.sign(Ds['Q']['mean']) < 0),
                   'ci_supported_reversal': bool(Ds['U']['decisive'] and Ds['Q']['decisive'] and np.sign(Ds['U']['mean']) != np.sign(Ds['Q']['mean']))}
            rows.append(row)
            Dtests += [(len(rows) - 1, 'U', Ds['U']['p_null_centred']), (len(rows) - 1, 'Q', Ds['Q']['p_null_centred'])]
            Jtests.append((len(rows) - 1, J['p_null_centred']))
    adjD = holm([t[2] for t in Dtests])
    for (ri, s, _), pa in zip(Dtests, adjD):
        rows[ri][f'D_{s}_p_holm'] = float(pa)
    for (ri, _), pa in zip(Jtests, holm([t[1] for t in Jtests])):
        rows[ri]['J_p_holm'] = float(pa)
    for r in rows:
        r['holm_supported_reversal'] = bool(r['observed_reversal'] and r['D_U_p_holm'] < ALPHA and r['D_Q_p_holm'] < ALPHA)
    # secondary: f16 vs p8e1
    sec = []
    for i in range(1, NL):
        D = {s: E[si[s], i, fi['f16']] - E[si[s], i, fi['p8e1']] for s in BACKS}
        sec.append({'layer': i, **{f'D_{s}': st(D[s]) for s in BACKS}, 'J': st(D['Q'] - D['U']), 'A_a': A[down[i]]['a'], 'A_w': A[down[i]]['w']})
    for fam, key in ((['D_U', 'D_Q'], 'D'), (['J'], 'J')):
        ps = [r[k]['p_null_centred'] for r in sec for k in fam]
        adj = holm(ps)
        it = iter(adj)
        for r in sec:
            for k in fam:
                r[k]['p_holm_secondary'] = float(next(it))
    # selection gains (Q background; primary)
    sel = lambda q, s: np.stack([E[si[s], r['layer'], fi[r[q]]] for r in rows])              # [81, articles]
    gains = {}

    def gain(name, idx, s, x, y):
        v = (sel(x, s)[idx] - sel(y, s)[idx]).mean(0) if len(idx) else np.zeros(na)
        gains[name] = st(v, (0.05, 0.025)) | {'pairs': int(len(idx))}
    allp = np.arange(len(rows))
    gain('G_shift', allp, 'Q', 'q_U', 'q_Q')
    gain('G_BC', allp, 'Q', 'q_BC', 'q_Q')
    gain('U_background: E_U(q_U) - E_U(q_Q)', allp, 'U', 'q_U', 'q_Q')
    gain('U_background: E_U(q_BC) - E_U(q_U)', allp, 'U', 'q_BC', 'q_U')
    diff = np.array([r['q_U'] != r['q_Q'] for r in rows])
    gain('G_shift on q_U != q_Q (secondary)', allp[diff], 'Q', 'q_U', 'q_Q')
    for bits in (4, 6, 8):
        ix = allp[[r['bits'] == bits for r in rows]]
        gain(f'G_shift {bits}-bit', ix, 'Q', 'q_U', 'q_Q')
        gain(f'G_BC {bits}-bit', ix, 'Q', 'q_BC', 'q_Q')
    for g, rg in GROUPS.items():
        ix = allp[[r['layer'] in rg for r in rows]]
        gain(f'G_shift {g}', ix, 'Q', 'q_U', 'q_Q')
        gain(f'G_BC {g}', ix, 'Q', 'q_BC', 'q_Q')
    ix = allp[[r['layer'] not in KNOWN for r in rows]]
    gain('G_shift excluding L2/L14/L27', ix, 'Q', 'q_U', 'q_Q')
    gain('G_BC excluding L2/L14/L27', ix, 'Q', 'q_BC', 'q_Q')
    best = np.stack([E[1, r['layer'], fi[min((r['pair'].split('-')), key=lambda f: E[1, r['layer'], fi[f]].mean())]] for r in rows])
    gains['observed-best gap of q_Q (in-sample, optimistic)'] = st((sel('q_Q', 'Q') - best).mean(0))
    gains['observed-best gap of q_BC (in-sample, optimistic)'] = st((sel('q_BC', 'Q') - best).mean(0))
    g1, g2 = gains['G_shift'], gains['G_BC']
    dec = []
    if g1['ci97.5'][0] > EPS:
        dec.append('G_shift 97.5% lower > eps: supports a pre-registered study of background-aware re-scoring in this scope' +
                   ('; G_BC also passes: also a practical gap relative to the old BC' if g2['ci97.5'][0] > EPS else ''))
    elif g2['ci97.5'][0] > EPS:
        dec.append('only G_BC passes: likely a proxy ranking error, not attributable to prefix-induced rank change')
    if g1['ci97.5'][1] < EPS:
        dec.append('G_shift 97.5% upper < eps: mean gain below the threshold; keep local cases; no whole-model re-scoring')
    if g1['ci97.5'][0] <= EPS <= g1['ci97.5'][1]:
        dec.append('G_shift CI straddles eps: undetermined for this calibration selector')
    res = {'complete_main_matrix': complete, 'nonfinite': int((~np.isfinite(ce)).sum()), 'articles': na, 'windows_per_article': WPA,
           'bootstrap': BOOT, 'eps': EPS, 'gains': gains, 'decision': dec if complete else ['main matrix incomplete: no primary decision']}
    cls = Counter(r['class'] for r in rows)
    acc = {}
    for s in BACKS:
        dsets = [r for r in rows if r[f'D_{s}_decisive'] and not r['proxy_tie']]
        acc[s] = {'decisive': len(dsets), 'right': sum(r[f'bc_right_{s}'] for r in dsets), 'undetermined': sum(not r[f'D_{s}_decisive'] for r in rows),
                  'proxy_ties': sum(r['proxy_tie'] for r in rows), 'coverage': len(dsets) / len(rows)}
    common = [r for r in rows if r['D_U_decisive'] and r['D_Q_decisive'] and not r['proxy_tie']]
    acc['common_decisive'] = {'pairs': len(common), 'right_U': sum(r['bc_right_U'] for r in common), 'right_Q': sum(r['bc_right_Q'] for r in common)}
    res['bc_accuracy'] = acc
    res['classes'] = dict(cls)
    res['reversals'] = {'observed': [f"L{r['layer']} {r['pair']}" for r in rows if r['observed_reversal']],
                        'ci_supported': [f"L{r['layer']} {r['pair']}" for r in rows if r['ci_supported_reversal']],
                        'holm_supported': [f"L{r['layer']} {r['pair']}" for r in rows if r['holm_supported_reversal']],
                        'J_holm_significant': [f"L{r['layer']} {r['pair']}" for r in rows if r['J_p_holm'] < ALPHA]}
    res['selectors'] = {'q_U != q_Q': [f"L{r['layer']} {r['pair']}" for r in rows if r['q_U'] != r['q_Q']],
                        'q_BC != q_Q': [f"L{r['layer']} {r['pair']}" for r in rows if r['q_BC'] != r['q_Q']],
                        'low_margin_U': sum(r['low_margin_U'] for r in rows), 'low_margin_Q': sum(r['low_margin_Q'] for r in rows)}
    res['L0_control'] = {'U_equals_Q_all_settings': bool(np.array_equal(ce[0, 0], ce[1, 0])), 'note': 'copied from U after verifying the prefix bitwise'}
    res['secondary_f16_p8e1'] = sec
    res['descriptive_d_vs_fp32'] = {f'L{i}': {s: {f: float((E[si[s], i, fi[f]] - E[si[s], i, fi['fp32']]).mean()) for f in SETTINGS[1:]} for s in BACKS} for i in range(NL)}
    res['model_nll'] = {'fp32': float(z['model_fp32'].sum() / (len(z['model_fp32']) * TP)), 'A': float(z['model_A'].sum() / (len(z['model_A']) * TP))}
    write(out / 'selection_gain.json', {k: gains[k] for k in gains} | {'decision': res['decision']})
    write(out / 'stats.json', res)
    with open(out / 'pairwise_results.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if not isinstance(rows[0][k], list)] + ['D_U_ci95', 'D_Q_ci95', 'J_ci95'])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: v for k, v in r.items() if not isinstance(v, list)} | {'D_U_ci95': r['D_U_ci95'], 'D_Q_ci95': r['D_Q_ci95'], 'J_ci95': r['J_ci95']})
    with open(out / 'per_article_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['background', 'layer', 'setting', 'article', 'title', 'nll'])
        for s in BACKS:
            for i in range(NL):
                for f in SETTINGS:
                    for k in range(na):
                        wr.writerow([s, i, f, k, segs[k]['title'], E[si[s], i, fi[f], k]])
    with open(out / 'per_window_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['background', 'layer', 'setting', 'window', 'article', 'summed_ce'])
        for s in BACKS:
            for i in range(NL):
                for f in SETTINGS:
                    for w in range(ce.shape[-1]):
                        wr.writerow([s, i, f, w, w // WPA, ce[si[s], i, fi[f], w]])
    # figures
    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = {4: '#d62728', 6: '#1f77b4', 8: '#2ca02c'}
    for r in rows:
        xe = [[r['D_U_mean'] - r['D_U_ci95'][0]], [r['D_U_ci95'][1] - r['D_U_mean']]]
        ye = [[r['D_Q_mean'] - r['D_Q_ci95'][0]], [r['D_Q_ci95'][1] - r['D_Q_mean']]]
        mk = '*' if r['holm_supported_reversal'] else ('s' if r['ci_supported_reversal'] else 'o')
        ax.errorbar(r['D_U_mean'], r['D_Q_mean'], xerr=xe, yerr=ye, fmt=mk, color=colors[r['bits']], ms=6 if mk == 'o' else 11, alpha=.8, lw=.7)
        if r['observed_reversal'] or r['layer'] in KNOWN:
            ax.annotate(f"L{r['layer']}", (r['D_U_mean'], r['D_Q_mean']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.axhline(0, color='k', lw=.8)
    ax.axvline(0, color='k', lw=.8)
    ax.set_xlabel('D_U = E_U(INT) - E_U(Posit)  (nat/token; + = Posit better)')
    ax.set_ylabel('D_Q = E_Q(INT) - E_Q(Posit)')
    ax.set_title('Same-bit candidate differences, prefix FP32 (U) vs prefix A (Q)\nquadrants II/IV = reversal; * Holm-supported, square CI-supported')
    for bits, c in colors.items():
        ax.plot([], [], 'o', color=c, label=f'{bits}-bit')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'fig_scatter_DU_DQ.png', dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ['G_shift', 'G_BC', 'G_shift 4-bit', 'G_shift 6-bit', 'G_shift 8-bit', 'G_BC 4-bit', 'G_BC 6-bit', 'G_BC 8-bit',
             'G_shift L1-8', 'G_shift L9-17', 'G_shift L18-27', 'G_shift excluding L2/L14/L27', 'G_BC excluding L2/L14/L27']
    for y, nm in enumerate(names):
        g = gains[nm]
        lo, hi = g.get('ci97.5', g['ci95'])
        ax.plot([lo, hi], [y, y], color='k')
        ax.plot(g['mean'], y, 'o', color='#1f77b4' if 'shift' in nm else '#d62728')
    ax.axvline(0, color='grey', lw=.8)
    ax.axvline(EPS, color='grey', ls='--', lw=.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlabel('mean single-point gain on unseen articles (nat/token), 97.5% CI; dashed = eps')
    fig.tight_layout()
    fig.savefig(out / 'fig_selection_gain.png', dpi=130)
    plt.close(fig)
    code = {'both right': 0, 'right then wrong': 1, 'wrong then right': 2, 'both wrong': 3, 'proxy tie': 4}
    M = np.full((len(PAIRS), NL - 1), 5)
    for r in rows:
        M[PAIRS.index(tuple(r['pair'].split('-'))), r['layer'] - 1] = code.get(r['class'], 5)
    fig, ax = plt.subplots(figsize=(11, 2.4))
    cmap = matplotlib.colors.ListedColormap(['#9fd39f', '#f4a261', '#7fb3d5', '#e76f51', '#cccccc', '#ffffff'])
    ax.imshow(M, cmap=cmap, vmin=-.5, vmax=5.5, aspect='auto')
    ax.set_xticks(range(NL - 1))
    ax.set_xticklabels([f'L{i}' for i in range(1, NL)], fontsize=7)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([f'{a}-{b}' for a, b in PAIRS])
    for (yy, xx), v in np.ndenumerate(M):
        ax.text(xx, yy, ['✓✓', '✓✗', '✗✓', '✗✗', 'tie', '?'][v], ha='center', va='center', fontsize=6)
    ax.set_title('BC prediction vs measured order (U then Q)\n✓✓ both right, ✓✗ right→wrong, ✗✓ wrong→right, ✗✗ both wrong, ? undetermined', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'fig_bc_classes.png', dpi=130)
    plt.close(fig)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'gains': {k: gains[k] for k in ('G_shift', 'G_BC')}, 'classes': res['classes'], 'accuracy': acc,
                    'reversals': {k: len(v) for k, v in res['reversals'].items()}, 'decision': res['decision']}, ensure_ascii=False))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'freeze', 'audit', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-estimate', type=Path, help='plan: a JSON file with the smoke-test time estimate')
    a = ap.parse_args()
    out = a.out_dir.resolve()
    if a.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty: use a new directory')
    elif not (out / 'plan.json').exists():
        raise SystemExit('run --stage plan first')
    out.mkdir(parents=True, exist_ok=True)
    for d in ('logs', 'work'):
        (out / d).mkdir(exist_ok=True)
    LOGF = open(out / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage == 'plan':
        plan(out, a.smoke_estimate)
    else:
        {'calib': calib, 'freeze': freeze, 'audit': audit, 'data': data, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen_prefix_rank_shift.py')}) + '\n')


if __name__ == '__main__':
    main()
