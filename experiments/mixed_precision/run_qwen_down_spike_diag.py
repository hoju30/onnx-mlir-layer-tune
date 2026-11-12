#!/usr/bin/env python3
"""Qwen3-0.6B Down-input criterion, single-layer diagnostic: does the OLMoE spike grouping or the position pairing of
gradient and error improve the ranking of Down-input activation formats? (cross-model check; no BC / search change)

Per layer L (all transformer layers of the config) only the Down input is quantized:
  h = SiLU(z_gate) * z_up,  y = W_down Q(h);  every other weight / activation FP32 (Gate, Up, Down weights too).
Formats int4, int6, int8 (native calibration (scale, zp) of the Down x slot), p4e1, p6e1, p8e1 (Universal LUT), plus
the FP32 identity. Scores on the original 10 calibration windows s (128 positions t, channels c), float64:
  e = Q(h) - h, mu_c = mean_{s,t} e, r = e - mu_c, GS_c = sum_s (sum_t g)^2, U = sum_c GS_c mu_c^2
  A  T_old   = sum_c mean_{s,t}(g^2) sum_{s,t} r^2 + U
  B  T_group = sum_c [mean_S(g^2) sum_S r^2 + mean_N(g^2) sum_N r^2] + U,  S = {a_st >= 6 median(a)},
               a_st = max_c |h_stc| (calibration rows only; same mask for all formats; no re-centering)
  C  T_pair  = sum_{s,t,c} g^2 r^2 + U
g = d(summed next-token CE of window s)/dh (127 predicted positions; position 127 has g = 0 and counts in n = 1280,
as the native statistics). Truth: per-article NLL change of new WikiText-103 articles (3 windows each), GPU replay.

Stages (one new --out-dir): plan, audit, calib, data, eval, analyze.
  plan     plan.json / plan.md: hashes, layers, formats, formulas, threshold, data rule, seeds, statistics, stop rule
  audit    every historical Qwen token window decoded and matched against WikiText-2 (audit.json)
  calib    CPU FP32 forward / backward on the calibration windows, controls, errors, three scores, mechanism,
           leave-one-window-out; scores and predicted orders frozen (freeze.json) before any diagnostic loss
  data     128 new WikiText-103 articles x 3 windows, Qwen3 tokenizer, verified scan + historical Qwen overlap check
  eval     GPU (offline PyTorch QDQ, not onnx-mlir native): per layer x (FP32 identity + 6 formats) per-window CE
  analyze  rankings, article bootstrap, selection, stability, mechanism
CPU stages: ../../venv/bin/python; eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_olmoe_dynamic_route as dr  # noqa: E402  (read / write / sha / env_info; standard imports only)

NATIVE = HERE / 'runs/native_eps_fix_20261014/qwen3'
PREP = HERE / 'runs/qwen3_prepare_20261011'
MODEL = HERE / 'qwen3_assets/model'
QWEN35_TOK = HERE / 'qwen35_assets/model/tokenizer.json'
SEQ, TP, CAL, WPA, N_ART = 128, 127, 10, 3, 128
FORMATS = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')
SAME_BIT = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
TAU, LOW_COVER, PROXY_TIE, BOOT, ALPHA = 6.0, 8, 0.02, 10000, 0.05
KNOWN_LAYERS = (0, 2, 14, 27)
SEEDS = {'order': 27101, 'start': 27102, 'bootstrap': 27103}
CHUNK, LM_SUB, THREADS = 32, 16, 12
OLMOE_VOCAB = 50304
TOL = {'hook_rel': 1e-6, 'grad_rel': 1e-4, 'identity_ce_abs': 1e-3, 'native_C_median_rel': 1e-3, 'native_C_max_rel': 1e-2,
       'identity_rel': 1e-9, 'synthetic_rel': 1e-12}
BUDGET_MIN = {'plan': 3, 'audit': 5, 'calib': 15, 'data': 10, 'eval': 30, 'analyze': 5}
METHODS = ('T_old', 'T_group', 'T_pair')
CODE = ['run_qwen_down_spike_diag.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'run_w8_per_channel_check.py',
        'wt_manifest_scan.py', 'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py', 'qwen3_prepare.py']


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in CODE}


def input_files():
    import run_olmoe_route_mass_restore as mr
    return ([MODEL / f for f in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')]
            + [NATIVE / 'calib_inputs.npy', NATIVE / 'ranges_by_point.json', NATIVE / 'table.json', PREP / 'validation_tokens.npy',
               PREP / 'test_tokens.npy', PREP / 'node_mapping.json', QWEN35_TOK] + mr.WT103 + mr.WT2)


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def check_plan(out):
    pl = read(out / 'plan.json')
    if pl['code_hashes'] != code_hashes():
        raise SystemExit('code changed since the plan (record an amendment first)')
    for k, v in pl['inputs'].items():
        if sha(HERE / k) != v:
            raise SystemExit('input changed: ' + k)
    return pl


def down_nodes():
    m = read(PREP / 'node_mapping.json')
    out = {}
    for node, param in m.items():
        p = param.split('.')
        if p[4] == 'down_proj':
            out[int(p[2])] = node
    return out


def layers_from_config():
    c = read(MODEL / 'config.json')
    return list(range(int(c['num_hidden_layers']))), c


def pairs_all():
    """(f1, f2, primary) for every unordered format pair; primary = same-bit INT / Posit."""
    out = []
    for i, f1 in enumerate(FORMATS):
        for f2 in FORMATS[i + 1:]:
            out.append((f1, f2, (f1, f2) in SAME_BIT))
    return out


def qwen3_tokenizer():
    sys.path.insert(0, str(HERE / 'gpt2_assets' / 'python_deps'))
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL / 'tokenizer.json'))


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, log):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists: the plan is frozen once')
    layers, cfg = layers_from_config()
    dn = down_nodes()
    rng = read(NATIVE / 'ranges_by_point.json')
    if sorted(dn) != layers or any('x' not in rng[dn[l]] or not rng[dn[l]]['x'][1] > rng[dn[l]]['x'][0] for l in layers):
        raise SystemExit('down nodes / ranges do not cover the config layers')
    tab = read(NATIVE / 'table.json')
    if any(f not in tab['C'][dn[l]] for l in layers for f in FORMATS):
        raise SystemExit('native table lacks a candidate format')
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    P = pairs_all()
    plan_ = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'nature': 'single-layer diagnostic of the Down-input activation criterion; no change to the formal BC, search, knee or '
                  'pipeline; no fine-tuning. Offline PyTorch QDQ on GPU for the truth, not an onnx-mlir native integration.',
        'model': {'dir': str(MODEL.relative_to(HERE)), 'layers': layers, 'num_hidden_layers': cfg['num_hidden_layers'],
                  'intermediate_size': cfg['intermediate_size'], 'hidden_size': cfg['hidden_size'], 'rms_norm_eps': cfg['rms_norm_eps'],
                  'dtype': 'float32 (checkpoint bfloat16 cast exactly)', 'attention': 'eager, constant causal mask (qwen3_prepare.CausalLM)',
                  'native_reference': str(NATIVE.relative_to(HERE)) + ' (RMSNorm epsilon fixed, 2026-10-14)'},
        'tap': 'Down input h = SiLU(z_gate) * z_up (forward hook input of mlp.down_proj, after the product, before the Down '
               'linear); only this tensor is quantized, one layer at a time; Gate / Up / Down weights and every other '
               'activation FP32; Down output not quantized',
        'formats': {'candidates': list(FORMATS), 'baseline': 'fp32 identity',
                    'int': 'erm.int_qparams(lo, hi, bits) of ranges_by_point.json[down node]["x"] (native calibration range), '
                           'qwen3_swiglu_common.int_act_qdq (scale / zero-point on the tensor device)',
                    'posit': 'qwen3_swiglu_common.PositLut (Universal decision boundaries) for p4e1, p6e1, p8e1, each verified '
                             'against Universal before use; the diagnostic data never touch the ranges'},
        'scores': {'A': 'T_old = sum_c mean_{s,t}(g^2) sum_{s,t} r^2 + U',
                   'B': 'T_group = sum_c [mean_S(g^2) sum_S r^2 + mean_N(g^2) sum_N r^2] + U',
                   'C': 'T_pair = sum_{s,t,c} g^2 r^2 + U',
                   'common': 'e = Q(h) - h; mu_c = mean over all 1280 calibration rows; r = e - mu_c (no re-centering inside '
                             'groups); GS_c = sum_s (sum_t g)^2; U = sum_c GS_c mu_c^2; float64 accumulation; n = 10 windows x 128 '
                             'positions (native positions = 128; position 127 has no target and g = 0)',
                   'gradient': 'g = d(summed next-token CE of the window, positions 0..126)/dh, FP32 CPU autograd, no '
                               'coefficient, no first-order term, no conversion to NLL',
                   'not_used': 'table.json C (Down input + output) is not the original BC of this round; it is only a '
                               'consistency control (x + y reconstruction)'},
        'spike': {'row_score': 'a_st = max_c |h_stc| on the calibration rows', 'threshold': f'{TAU:g} x median_{{s,t}}(a_st)',
                  'groups': 'S = a >= threshold, N = rest; one mask for every format',
                  'fallbacks': 'non-positive / non-finite threshold -> A (recorded); empty group -> B equals A exactly; a non-empty '
                               f'group with fewer than {LOW_COVER} rows -> low-coverage flag only', 'no_scan': 'no other multiple'},
        'controls': {
            'model': 'own CausalLM copy equals qwen3_prepare.CausalLM bitwise (CPU); config epsilon 1e-6; calibration tokens = '
                     'native calib_inputs.npy = prepare validation windows 0-9',
            'quantizer': 'GPU QDQ of the calibration h equals the CPU QDQ bitwise for every layer and format (sha256 of the '
                         'float32 errors); TF32 off (matmul and cuDNN), deterministic algorithms, float32',
            'positions': 'g at position 127 exactly 0; n = 1280',
            'hook': f'captured h equals SiLU(gate(x)) * up(x) recomputed from the MLP input (rel <= {TOL["hook_rel"]:g}); Down '
                    'output equals down_proj(h)',
            'gradient': f'autograd g_h vs gY @ W_down (rel to max |g_h| <= {TOL["grad_rel"]:g})',
            'identity': 'fp32: zero error and zero scores; GPU identity replay of every layer vs the no-hook forward of the '
                        f'same chunk: max |summed CE difference| <= {TOL["identity_ce_abs"]:g} per window',
            'native': f'PyTorch A_x + A_y of every Down node vs native table C: median rel <= {TOL["native_C_median_rel"]:g}, max '
                      f'rel <= {TOL["native_C_max_rel"]:g}',
            'synthetic': 'zero error -> all terms 0; channel-constant gradient -> T_group = T_pair = T_old; empty group -> B = A; '
                         f'between / within identities (rel <= {TOL["synthetic_rel"]:g}); real-data identities rel <= {TOL["identity_rel"]:g}',
            'failure': 'any failed control stops the stage before a formal comparison; fix, record an amendment, rerun'},
        'data': {'calibration': 'unchanged: native calib_inputs.npy (10 x 128, WikiText-2 validation)',
                 'diagnostic': f'{N_ART} new WikiText-103 train articles x {WPA} contiguous non-overlapping {SEQ}-token windows '
                               '(Qwen3 tokenizer, article lines joined with blank lines, no special tokens), drawn after the freeze',
                 'draw': f'eligible = WikiText-103 strict-boundary articles, not a WikiText-2 title / content hash, unique title and '
                         f'content, not in wt_manifest_scan.used_wt103_verified; visited in permutation order (seed {SEEDS["order"]}); '
                         f'skip if fewer than {WPA * SEQ} tokens; skip if >= 10% of its word 5-gram shingles fall in one WikiText-2 '
                         'article, in a previously used WikiText-103 article or in an accepted article; skip if >= 10% of the '
                         'shingles of any historical Qwen token window (audit stage) fall in the article, or >= 10% of the '
                         'article shingles fall in the historical Qwen windows; window start uniform (seed '
                         f'{SEEDS["start"]}); the same articles for every layer and format',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                 'manifest_scan': [{k: r[k] for k in ('path', 'articles', 'resolved')} for r in rows],
                 'historical_qwen_tokens': 'every integer token array under runs/ whose path names qwen or whose max id exceeds '
                                           f'{OLMOE_VOCAB} is decoded (Qwen3 or Qwen3.5 tokenizer) and matched against WikiText-2; '
                                           'unresolved windows are reported, and every window enters the overlap exclusion'},
        'evaluation': {'unit': 'article: NLL = (sum of the 3 windows summed CE) / (3 x 127)', 'per_layer_format':
                       'dNLL = NLL(Q at layer L) - NLL(fp32 identity replay at layer L), same chunk / device / code path; '
                       'negative values kept signed',
                       'device': 'GPU float32, TF32 off, deterministic; layer L replayed from the cached residual before the MLP'},
        'ranking': {'primary': [f'L{l} {a} vs {b}' for l in layers for a, b in SAME_BIT],
                    'primary_count': len(layers) * len(SAME_BIT),
                    'secondary': f'the other {sum(not p for _, _, p in P)} format pairs per layer (cross-bit, e.g. int6 vs p4e1, never '
                                 'labelled same-bit)',
                    'truth': 'D = mean over articles of [dNLL(f1) - dNLL(f2)]',
                    'decisive': f'article bootstrap ({BOOT}, seed {SEEDS["bootstrap"]}) percentile 95% CI excludes 0; one common set '
                                'for every method; all 84 primary pairs reported (decisive or uncertain)',
                    'right': 'sign(T(f1) - T(f2)) = sign(D) on a decisive pair and not a proxy tie',
                    'proxy_tie': f'|T(f1) - T(f2)| / max(T(f1), T(f2)) < {PROXY_TIE}: reported separately, never counted right',
                    'pick': 'per primary pair each method picks the format with the lower score (a proxy tie still picks the lower; '
                            'exact equality picks the INT format)',
                    'fix / new error': 'B or C right where A is not / A right where B or C is not (decisive primary pairs)',
                    'multiple_comparisons': {
                        'overall': 'B - A and C - A: selection regret difference, Bonferroni over the 2 comparisons (97.5% '
                                   'percentile CI); bootstrap two-sided p with Holm adjustment also reported',
                        'specific_pair': 'a specific fix is called confirmed only if the simultaneous 95% CI (max-|t| article '
                                         'bootstrap over the 84 primary pairs) excludes 0; uncorrected 95% CIs are exploratory'},
                    'selection': 'R_M = mean over the 84 primary pairs of dNLL(pick_M) per article; R_B - R_A and R_C - R_A with '
                                 'article bootstrap CIs (the same resample for every layer / format / method)',
                    'independence': 'the 3 pairs of a layer are not independent samples; layer counts are descriptive',
                    'stability': 'leave-one-calibration-window-out (threshold fixed at the full-calibration value, mask on the '
                                 'remaining rows, mu / U / scores recomputed): pick flips per method'},
        'decision': {
            'support_next_round_if_all': [
                'R_M - R_A < 0 with the Bonferroni 97.5% CI upper bound < 0 (M = B or C)',
                'on decisive primary pairs fixes - new errors >= 3, fixes in >= 3 distinct layers, >= 2 of them outside the '
                f'earlier studied layers {list(KNOWN_LAYERS)}',
                ">= 2/3 of M's decisive primary fixes keep M's pick in all 10 leave-one-window-out recomputations"],
            'then': 'propose (not run) a separately registered whole W+A configuration comparison at the original Qwen3 knee '
                    'resource limits with the same MILP, on new articles',
            'otherwise': 'keep the original BC and stop; no threshold search or other multiple',
            'never': 'claim a whole-pipeline PPL improvement from single-layer results; change scores, threshold, formats or '
                     'data size after seeing the diagnostic loss'},
        'statistics_order': 'controls -> calibration scores frozen (freeze.json) -> new articles drawn -> diagnostic loss',
        'pre_plan_smoke_test': 'before this plan a scratch run (outside runs/) exercised every stage: calibration stage in full '
                               '(its controls and calibration scores were seen), data stage with other seeds (999001/999002) and 6 '
                               'articles (never evaluated), eval / analyze on calibration windows only (no new-article loss). It '
                               'found three code faults (tokenizer import order, the checks re-entering the capture hooks, '
                               'full-chunk logits out of GPU memory), fixed before this plan; nothing of the design was changed.',
        'budget_minutes': BUDGET_MIN, 'budget_rule': 'an over-budget stage is recorded; the evaluation is not cut short and no '
                                                     'partial analysis replaces the plan',
        'seeds': SEEDS, 'tolerances': TOL, 'inputs': inputs, 'code_hashes': code_hashes(),
        'environment_cpu': dr.env_info(),
        'environment_gpu': 'PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python (recorded by eval)'}
    write(out / 'plan.json', plan_)
    L = ['# 計畫：Qwen3-0.6B Down-input criterion 跨模型單點診斷', '',
         f'時間：{plan_["time"]}。以 plan.json 為準；寫入後不改分數、門檻、格式、資料量或 seed。', '',
         '- 性質：單層、單點診斷；不改正式 BC、不搜尋配置、不重選 knee、不 fine-tune。真值用 GPU 上的離線 PyTorch QDQ，不是 onnx-mlir native。',
         f'- 範圍：config 的全部 {len(layers)} 層（L0–L{layers[-1]}）的 Down input h = SiLU(z_gate)·z_up；一次只量化一層；其餘權重與 activation 全 FP32；不量化 Down output。',
         f'- 格式：{", ".join(FORMATS)}，另設 FP32 identity。INT 用 native calibration 的 Down x range／scale／zero-point；Posit 用 Universal LUT（p4e1／p6e1／p8e1 都先驗證）。',
         '- 三種分數（calibration 10 窗口 × 128 位置，n = 1280，float64；共用 μ_c 與 U；B 不在組內重新中心化）：',
         '  - A：T_old = Σ_c mean(g²)·Σ r² + U',
         '  - B：T_group = Σ_c [mean_S(g²)·Σ_S r² + mean_N(g²)·Σ_N r²] + U',
         '  - C：T_pair = Σ g²r² + U',
         f'- spike：a_st = max_c |h_stc|；門檻 = {TAU:g} × calibration median；所有格式同一 mask。門檻無效回退 A；空組 = A；非空組 < {LOW_COVER} rows 只加旗標；不掃其他倍數。',
         '- 梯度：每窗口 summed next-token CE（127 個預測位置）；position 127 的 g = 0，仍計入 n（與 native 相同）。',
         '- 控制：wrapper 與 qwen3_prepare 一致；calibration tokens 一致；GPU／CPU QDQ 逐位元一致；TF32 關閉；hook 位置；autograd 梯度 vs gY·W_down；'
         'position 127 梯度為 0；FP32 identity 零誤差、零分數、replay loss = 無 hook loss；x+y 重建 vs native table C；合成資料（零誤差、常數梯度、空組、組間／組內分解）。'
         '控制失敗即停止、修正、記 amendment。',
         '- table C（含 input＋output）不當作本輪的原 BC，只作一致性控制。',
         f'- 診斷資料：{N_ART} 篇新 WT103 文章 × {WPA} 個不重疊 128-token 窗口，Qwen3 tokenizer；verified 掃描排除 {len(used)} 篇歷史文章；排除 WT2 重複／近重複；'
         f'另解碼全部歷史 Qwen token 檔比對重疊（audit）；seed：順序 {SEEDS["order"]}、起點 {SEEDS["start"]}。凍結分數之後才抽文章。',
         '- 真值：每篇文章 NLL（3 窗口 summed CE / 381）；ΔNLL = 量化 − 同層 FP32 identity replay；負值保留。',
         f'- 主要比較：每層 int4–p4e1、int6–p6e1、int8–p8e1，共 {len(layers) * 3} 對全部報告；其他 {sum(not p for _, _, p in P)} 對／層為次要（跨位元，不標為同位元）。',
         f'- 可判定：文章 bootstrap（{BOOT} 次，seed {SEEDS["bootstrap"]}，所有層與格式同一次重抽）95% CI 排除 0；三種方法共用同一集合。proxy tie：相對差 < {PROXY_TIE}，單列、不算對。',
         '- 多重比較：B−A、C−A 的整體比較用選格式後的 ΔNLL 差（R_M − R_A），Bonferroni（2 個比較）97.5% CI，另報 Holm 校正 p；特定配對的「確認修正」需 84 對的 simultaneous 95% CI（max-|t| bootstrap）排除 0；未校正 CI 只作探索。',
         '- 穩定性：逐一移除 calibration 窗口（門檻固定），看各方法挑選是否翻轉。',
         f'- 決策：B 或 C 同時滿足 (1) R_M − R_A 的 97.5% CI 上界 < 0；(2) 可判定主要配對的修正 − 新增誤排 ≥ 3，修正分布 ≥ 3 層且至少 2 層不在 {list(KNOWN_LAYERS)}；'
         '(3) ≥ 2/3 修正在 10 次移除窗口下都不翻轉 → 只提出下一輪完整 W+A 配置比較計畫（原 knee 資源限制、同 MILP、新文章、另行登記），本輪不執行。否則保留原 BC、停止，不追加門檻搜尋。',
         '- 不得只憑單點結果宣稱整體 pipeline PPL 改善。',
         '- 計畫前的冒煙測試（scratchpad，不在 runs/）：calibration 階段完整跑過（看過其控制與 calibration 分數）；資料階段用其他 seed 抽 6 篇（未評估）；'
         'eval／analyze 只用 calibration 窗口，沒有新文章 loss。修正 3 個程式錯誤（tokenizer 匯入順序、檢查時重入 hook、整個 chunk 的 logits 超出 GPU 記憶體），設計未改。', '',
         f'## 預算（分鐘）：{BUDGET_MIN}；超時只記錄，不中途截斷評估、不以部分資料取代計畫。', '']
    (out / 'plan.md').write_text('\n'.join(L) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written: {len(layers)} layers; previous WT103 {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# audit of historical Qwen token data
# --------------------------------------------------------------------------- #
def hist_files(out):
    rows = []
    for p in sorted(HERE.glob('runs/**/*.npy')):
        if p.resolve().is_relative_to(out.resolve()) or p.stat().st_size > 200 << 20:
            continue
        try:
            a = np.load(p, mmap_mode='r')
        except Exception:
            continue
        if a.dtype.kind not in 'iu' or a.ndim != 2 or a.shape[1] not in (128, 512) or not a.size:
            continue
        rel = str(p.relative_to(HERE))
        mx = int(a.max())
        if 'qwen' in rel.lower() or mx > OLMOE_VOCAB:
            rows.append((p, rel, a.shape, mx))
    return rows


def window_shingles(text):
    import run_olmoe_route_mass_restore as mr
    return mr.shingles(mr.norm_text(text))


def audit(out, log):
    import run_olmoe_route_mass_restore as mr
    t0 = time.perf_counter()
    check_plan(out)
    tok3 = qwen3_tokenizer()
    from tokenizers import Tokenizer
    tok35 = Tokenizer.from_file(str(QWEN35_TOK))
    # tokenizer control: the Qwen3 tokenizer of this round reproduces the prepare windows (WikiText-2 validation / test)
    import pyarrow.parquet as pq
    ctl = {}
    for split, f in (('validation', 'validation_tokens.npy'), ('test', 'test_tokens.npy')):
        text = '\n\n'.join(pq.read_table(HERE / 'gpt2_assets/data' / f'{split}-00000-of-00001.parquet')['text'].to_pylist())
        ids = tok3.encode(text, add_special_tokens=False).ids
        ref = np.load(PREP / f)
        ctl[f'tokenizer_reproduces_prepare_{split}'] = bool(np.array_equal(np.array(ids[:ref.size]).reshape(ref.shape), ref))
    if not all(ctl.values()):
        raise SystemExit(f'tokenizer control failed: {ctl}')
    # WikiText-2 union of shingles (whole normalized text per split: titles and article boundaries included)
    U2 = set()
    for f in mr.WT2:
        U2 |= window_shingles(' '.join(pq.read_table(f)['text'].to_pylist()))
    rows, idx_sh, idx_win, wid = [], [], [], 0
    for p, rel, shape, mx in hist_files(out):
        a = np.load(p)
        row = {'path': rel, 'sha256': sha(p), 'shape': list(shape), 'max_id': mx}
        if 'tiny_check' in rel and mx < 1024:
            row.update(tokenizer='none', resolved='synthetic (tiny-model ids < 1024, not text)', windows=int(shape[0]))
            rows.append(row)
            continue
        if 'qwen35' in rel.lower():
            tk, name = tok35, 'qwen3.5'
        else:
            if mx >= 151936:
                raise SystemExit(f'{rel}: id {mx} outside the Qwen3 vocabulary')
            tk, name = tok3, 'qwen3'
        shares, short = [], 0
        for w in a:
            sh = window_shingles(tk.decode(w.tolist()))
            if len(sh) < 10:
                short += 1
            else:
                shares.append(len(sh & U2) / len(sh))
            for s_ in sh:
                idx_sh.append(s_)
                idx_win.append(wid)
            wid += 1
        sh_ = np.array(shares)
        row.update(tokenizer=name, windows=int(shape[0]), windows_short=short,
                   wt2_share={'min': float(sh_.min()) if len(sh_) else None, 'median': float(np.median(sh_)) if len(sh_) else None},
                   windows_below_half=int((sh_ < 0.5).sum()),
                   resolved='wikitext-2' if len(sh_) and (sh_ >= 0.5).all() else 'unresolved')
        rows.append(row)
    np.savez_compressed(out / 'work' / 'qwen_hist_shingles.npz', shingle=np.array(idx_sh, np.uint64), window=np.array(idx_win, np.int64))
    res = {'controls': ctl, 'files': rows, 'windows_indexed': wid, 'unresolved_files': [r['path'] for r in rows if r['resolved'] == 'unresolved'],
           'index_sha256': sha(out / 'work' / 'qwen_hist_shingles.npz'),
           'rule': 'resolved wikitext-2 if every window with >= 10 shingles has >= 50% of its word 5-gram shingles in the '
                   'WikiText-2 text (all splits); every window, resolved or not, enters the data-stage overlap exclusion'}
    write(out / 'audit.json', res)
    log_stage(out, 'audit', time.perf_counter() - t0)
    log(f'audit: {len(rows)} files, {wid} windows indexed, unresolved {res["unresolved_files"]}; controls {ctl}')


# --------------------------------------------------------------------------- #
# quantizers (CPU and GPU, one implementation)
# --------------------------------------------------------------------------- #
class Quant:
    """act(x, fmt, (lo, hi)): the pass's activation QDQ on x's device (int: erm.int_qparams + int_act_qdq; posit:
    PositLut boundaries / values moved to the device, the same searchsorted as PositLut.__call__)."""

    def __init__(self, work, device='cpu'):
        import torch
        import qwen3_swiglu_common as qc
        self.torch, self.qc, self.erm = torch, qc, qc.erm
        self.q = qc.Quantizers(work, formats=('p4e1', 'p6e1', 'p8e1'))
        self.lut = {f: (l.t_bounds.to(device), l.t_values.to(device)) for f, l in self.q.posit.items()}

    def verify(self):
        return {f: self.q.posit[f].verify() for f in self.q.posit}

    def act(self, x, fmt, rng):
        if fmt == 'fp32':
            return x
        if fmt.startswith('int'):
            bits = int(fmt[3:])
            scale, zp = self.erm.int_qparams(float(rng[0]), float(rng[1]), bits)
            return self.qc.int_act_qdq(x, scale, zp, bits)
        b, v = self.lut[fmt]
        return v[self.torch.searchsorted(b, self.qc._order_key_torch(x.float()), right=True)].to(x.dtype)


class Model:
    """Qwen3ForCausalLM float32 eager with the constant-causal-mask wrapper of qwen3_prepare (a copy: the GPU environment
    has no onnxruntime, which qwen3_prepare imports; equality checked in calib)."""

    def __init__(self, device='cpu'):
        import torch
        import qwen3_swiglu_common as qc  # noqa: F401  (transformers 4.57 path)
        from transformers import Qwen3ForCausalLM
        self.torch = torch
        self.model = Qwen3ForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation='eager').eval()
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)

        class CausalLM(torch.nn.Module):
            def __init__(self, model, seq):
                super().__init__()
                self.model = model
                mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)
                self.register_buffer('mask', mask[None, None], persistent=False)

            def forward(self, input_ids):
                return self.model(input_ids=input_ids, attention_mask=self.mask, use_cache=False).logits
        self.wrapper = CausalLM(self.model, SEQ).eval().to(device)
        self.layers = self.model.model.layers


def summed_ce_rows(torch, z, ids):
    """per-window summed next-token CE (positions 0..126), float64 sum of float32 token CE."""
    ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), ids[:, 1:].reshape(-1), reduction='none')
    return ce.view(z.shape[0], -1).double().sum(1)


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def scores(h, g, errs, thr, win, keep=None, timing=None):
    """three scores per format for one layer; rows = calibration (window, position); keep: optional row subset."""
    if keep is not None:
        h, g, win = h[keep], g[keep], win[keep]
        errs = {f: e[keep] for f, e in errs.items()}
    n = len(win)
    gd = g.astype(np.float64)
    X = gd ** 2
    GS = sum(gd[win == k].sum(0) ** 2 for k in np.unique(win))
    a = np.abs(h).max(1)
    valid = bool(np.isfinite(thr) and thr > 0)
    S = (a >= thr) if valid else np.zeros(n, bool)
    nS = int(S.sum())
    nN = n - nS
    mX = X.mean(0)
    grouped = valid and nS > 0 and nN > 0
    mXS, mXN = (X[S].mean(0), X[~S].mean(0)) if grouped else (None, None)
    res = {}
    for f, e in errs.items():
        t1 = time.perf_counter()
        ed = e.astype(np.float64)
        mu = ed.mean(0)
        Y = (ed - mu) ** 2
        U_c = GS * mu ** 2
        t2 = time.perf_counter()
        Vo = mX * Y.sum(0)
        t3 = time.perf_counter()
        Vg = (mXS * Y[S].sum(0) + mXN * Y[~S].sum(0)) if grouped else Vo.copy()
        t4 = time.perf_counter()
        Vp = (X * Y).sum(0)
        t5 = time.perf_counter()
        if timing is not None:
            timing['error_stats'] += t2 - t1
            timing['T_old'] += t3 - t2
            timing['T_group'] += t4 - t3
            timing['T_pair'] += t5 - t4
        U = float(U_c.sum())
        res[f] = {'n': n, 'nS': nS, 'nN': nN, 'valid': valid, 'grouped': grouped, 'S': S, 'X': X, 'Y': Y, 'mu': mu,
                  'U_c': U_c, 'Vo': Vo, 'Vg': Vg, 'Vp': Vp, 'U': U,
                  'T_old': float(Vo.sum()) + U, 'T_group': float(Vg.sum()) + U, 'T_pair': float(Vp.sum()) + U,
                  'V_old': float(Vo.sum()), 'V_group': float(Vg.sum()), 'V_pair': float(Vp.sum())}
    return res


def decompose(r):
    """between = V_group - V_old and within = V_pair - V_group per channel, from the group means; row contributions."""
    X, Y, S = r['X'], r['Y'], r['S']
    n, nS, nN = r['n'], r['nS'], r['nN']
    if r['grouped']:
        mXs, mXn, mYs, mYn = X[S].mean(0), X[~S].mean(0), Y[S].mean(0), Y[~S].mean(0)
        between = nS * nN / n * (mXs - mXn) * (mYs - mYn)
        cX = np.where(S[:, None], X - mXs, X - mXn)
        cY = np.where(S[:, None], Y - mYs, Y - mYn)
    else:
        between = np.zeros(X.shape[1])
        cX, cY = X - X.mean(0), Y - Y.mean(0)
    rowc = (cX * cY).sum(1)
    within = (cX * cY).sum(0)
    return between, within, rowc


def synthetic_checks():
    rng = np.random.default_rng(7)
    n, C = 4 * 16, 8
    win = np.repeat(np.arange(4), 16)
    h = rng.standard_normal((n, C)).astype(np.float32)
    h[[0, 16, 33]] *= 50.0                                  # three spike rows
    g = rng.standard_normal((n, C)).astype(np.float32)
    e = rng.standard_normal((n, C)).astype(np.float32)
    thr = TAU * float(np.median(np.abs(h).max(1)))
    out = {}
    z = scores(h, g, {'z': np.zeros_like(e)}, thr, win)['z']
    out['zero_error_all_zero'] = z['T_old'] == z['T_group'] == z['T_pair'] == z['U'] == 0.0
    gc = np.tile(rng.standard_normal(C).astype(np.float32), (n, 1))
    c = scores(h, gc, {'e': e}, thr, win)['e']
    out['constant_gradient_rel'] = max(abs(c['V_group'] - c['V_old']), abs(c['V_pair'] - c['V_old'])) / c['V_old']
    em = scores(h, g, {'e': e}, np.inf, win)['e']
    out['invalid_threshold_equals_old'] = em['Vg'].tobytes() == em['Vo'].tobytes() and not em['valid']
    big = scores(h, g, {'e': e}, 1e30, win)['e']
    out['empty_group_equals_old'] = big['nS'] == 0 and big['Vg'].tobytes() == big['Vo'].tobytes()
    r = scores(h, g, {'e': e}, thr, win)['e']
    b, w, _ = decompose(r)
    sc = float(np.abs(r['Vo']).max())
    out['spike_rows'] = r['nS']
    out['between_identity_rel'] = float(np.abs(r['Vg'] - r['Vo'] - b).max() / sc)
    out['within_identity_rel'] = float(np.abs(r['Vp'] - r['Vg'] - w).max() / sc)
    out['passed'] = bool(out['zero_error_all_zero'] and out['constant_gradient_rel'] < TOL['synthetic_rel'] and out['invalid_threshold_equals_old']
                         and out['empty_group_equals_old'] and r['nS'] == 3 and out['between_identity_rel'] < TOL['synthetic_rel']
                         and out['within_identity_rel'] < TOL['synthetic_rel'])
    return out


def pick_and_tie(T1, T2):
    d = T1 - T2
    tie = abs(d) / max(abs(T1), abs(T2), 1e-300) < PROXY_TIE
    return d, bool(tie)


# --------------------------------------------------------------------------- #
# calib
# --------------------------------------------------------------------------- #
def calib(out, log):
    import torch
    import qwen3_swiglu_common as qc
    if (out / 'freeze.json').exists():
        raise SystemExit('freeze exists')
    t0 = time.perf_counter()
    pl = check_plan(out)
    torch.set_num_threads(THREADS)
    layers = pl['model']['layers']
    dn = down_nodes()
    ranges = read(NATIVE / 'ranges_by_point.json')
    table = read(NATIVE / 'table.json')
    work = out / 'work'
    tm = defaultdict(float)
    ctl = {'synthetic': synthetic_checks()}
    if not ctl['synthetic']['passed']:
        raise SystemExit(f'synthetic control failed: {ctl["synthetic"]}')
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ctl['calibration_tokens_equal_prepare'] = bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL]) and toks.shape == (CAL, SEQ))
    M = Model('cpu')
    m = M.model
    ctl['rms_norm_eps'] = [float(l.post_attention_layernorm.variance_epsilon) for l in M.layers][:1] + [float(m.model.norm.variance_epsilon)]
    from qwen3_prepare import CausalLM as PrepCausalLM
    with torch.no_grad():
        z_own = M.wrapper(torch.from_numpy(toks[:1]))
        z_ref = PrepCausalLM(m, SEQ).eval()(torch.from_numpy(toks[:1]))
    ctl['wrapper_equals_prepare_bitwise'] = bool(torch.equal(z_own, z_ref))
    Q = Quant(work, 'cpu')
    ctl['posit_lut_mismatches'] = Q.verify()
    # FP32 forward / backward with hooks
    t1 = time.perf_counter()
    m.model.embed_tokens.weight.requires_grad_(True)
    cap = {}

    def mlp_pre(l):
        def hook(mod, inp):
            cap[(l, 'x')] = inp[0]
        return hook

    def down_hook(l):
        def hook(mod, inp, outp):
            inp[0].retain_grad()
            outp.retain_grad()
            cap[(l, 'h')], cap[(l, 'y')] = inp[0], outp
        return hook
    hooks = []
    for l in layers:
        hooks += [M.layers[l].mlp.register_forward_pre_hook(mlp_pre(l)), M.layers[l].mlp.down_proj.register_forward_hook(down_hook(l))]
    H = {l: np.zeros((CAL * SEQ, 3072), np.float32) for l in layers}
    G = {l: np.zeros((CAL * SEQ, 3072), np.float32) for l in layers}
    Yo = {l: np.zeros((CAL * SEQ, 1024), np.float32) for l in layers}
    GY = {l: np.zeros((CAL * SEQ, 1024), np.float32) for l in layers}
    hook_rel, y_rel, grad_rel, last_zero, losses = 0.0, 0.0, 0.0, True, []
    for s in range(CAL):
        ids = torch.from_numpy(toks[s:s + 1])
        loss = qc.summed_ce(M.wrapper(ids), ids[0])
        loss.backward()
        losses.append(float(loss.detach()))
        sl = slice(s * SEQ, (s + 1) * SEQ)
        with torch.no_grad():
            for l in layers:
                x, h, y = cap[(l, 'x')], cap[(l, 'h')], cap[(l, 'y')]
                mlp = M.layers[l].mlp
                lin = torch.nn.functional.linear                 # module calls would re-enter the capture hooks
                href = mlp.act_fn(lin(x, mlp.gate_proj.weight)) * lin(x, mlp.up_proj.weight)
                hook_rel = max(hook_rel, float((h - href).abs().max() / href.abs().max()))
                y_rel = max(y_rel, float((y - lin(h, mlp.down_proj.weight)).abs().max() / y.abs().max()))
                gh, gy = h.grad[0], y.grad[0]
                grad_rel = max(grad_rel, float((gh - gy @ mlp.down_proj.weight).abs().max() / gh.abs().max()))
                last_zero &= bool((gh[-1] == 0).all())
                H[l][sl], G[l][sl], Yo[l][sl], GY[l][sl] = h[0].numpy(), gh.numpy(), y[0].numpy(), gy.numpy()
        m.zero_grad(set_to_none=True)
        cap.clear()
    for hk in hooks:
        hk.remove()
    m.model.embed_tokens.weight.requires_grad_(False)
    with torch.no_grad():
        nohook = [float(qc.summed_ce(M.wrapper(torch.from_numpy(toks[s:s + 1])), torch.from_numpy(toks[s]))) for s in range(CAL)]
    tm['calibration_forward_backward'] = time.perf_counter() - t1
    ctl.update({'hook_h_rel_max': hook_rel, 'hook_y_rel_max': y_rel, 'grad_vs_gy_Wdown_rel_max': grad_rel, 'grad_last_position_zero': last_zero,
                'loss_with_hooks_vs_no_hooks_max_abs': float(np.abs(np.array(losses) - np.array(nohook)).max()), 'calibration_losses': losses})
    win = np.repeat(np.arange(CAL), SEQ)
    pos = np.tile(np.arange(SEQ), CAL)
    tokflat = toks.reshape(-1)
    tok3 = qwen3_tokenizer()
    srows, drows, lrows, mech, shas, nat = [], [], [], {}, {}, []
    id_rel = {'between': 0.0, 'within': 0.0}
    fp32_zero = True
    for l in layers:
        node = dn[l]
        rx, ry = ranges[node]['x'], ranges[node]['y']
        h, g = H[l], G[l]
        t1 = time.perf_counter()
        errs = {}
        with torch.no_grad():
            ht = torch.from_numpy(h)
            for f in FORMATS:
                e = (Q.act(ht, f, rx) - ht).numpy()
                errs[f] = e
                shas[f'L{l}|{f}'] = hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest()
            e0 = (Q.act(ht, 'fp32', rx) - ht).numpy()
        tm['qdq'] += time.perf_counter() - t1
        a = np.abs(h).max(1)
        thr = TAU * float(np.median(a))
        r = scores(h, g, errs, thr, win, timing=tm)
        z0 = scores(h, g, {'fp32': e0}, thr, win)['fp32']
        fp32_zero &= bool(not e0.any() and z0['T_old'] == z0['T_group'] == z0['T_pair'] == 0.0)
        # native consistency: x + y slots with the original formula vs table C (not the baseline of this round)
        with torch.no_grad():
            yt = torch.from_numpy(Yo[l])
            ey = {f: (Q.act(yt, f, ry) - yt).numpy() for f in FORMATS}
        ryy = scores(Yo[l], GY[l], ey, np.inf, win)
        for f in FORMATS:
            mine = r[f]['T_old'] + ryy[f]['T_old']
            nat.append({'layer': l, 'format': f, 'pytorch_x_plus_y': mine, 'native_C': table['C'][node][f],
                        'rel': abs(mine - table['C'][node][f]) / abs(table['C'][node][f]), 'x_share': r[f]['T_old'] / mine})
        S = r[FORMATS[0]]['S']
        nS = int(S.sum())
        sp_tok = Counter(tok3.decode([int(t)]) for t in tokflat[S])
        mech[f'L{l}'] = {'threshold': thr, 'median_a': float(np.median(a)), 'max_a': float(a.max()), 'spike_rows': nS, 'coverage': nS / len(a),
                         'low_coverage': bool(0 < nS < LOW_COVER), 'valid_threshold': bool(np.isfinite(thr) and thr > 0),
                         'spike_positions': dict(Counter(int(p) for p in pos[S]).most_common(10)),
                         'spike_position0_rows': int((S & (pos == 0)).sum()), 'spike_tokens': dict(sp_tok.most_common(10)),
                         'g2_row_mean': {'S': float((g[S].astype(np.float64) ** 2).mean()) if nS else None,
                                         'N': float((g[~S].astype(np.float64) ** 2).mean())},
                         'g2_row_mean_quantiles_N': np.quantile((g[~S].astype(np.float64) ** 2).mean(1), [0.5, 0.9, 0.99]).tolist(),
                         'g2_row_mean_S_values': sorted(((g[S].astype(np.float64) ** 2).mean(1)).tolist(), reverse=True)[:20],
                         'formats': {}}
        for f in FORMATS:
            rf = r[f]
            b, w, rowc = decompose(rf)
            sc = max(float(np.abs(rf['Vo']).max()), 1e-300)
            id_rel['between'] = max(id_rel['between'], float(np.abs(rf['Vg'] - rf['Vo'] - b).max()) / sc)
            id_rel['within'] = max(id_rel['within'], float(np.abs(rf['Vp'] - rf['Vg'] - w).max()) / sc)
            order = np.argsort(-np.abs(rowc))
            tops = np.abs(rowc)[order]
            pair_row = (rf['X'] * rf['Y']).sum(1)
            cb, cw = int(np.argmax(np.abs(b))), int(np.argmax(np.abs(w)))
            mech[f'L{l}']['formats'][f] = {
                'r2_row_mean': {'S': float(rf['Y'][S].mean()) if nS else None, 'N': float(rf['Y'][~S].mean())},
                'between': float(b.sum()), 'within': float(w.sum()),
                'between_top_channel': cb if nS else None,
                'between_top_channel_share': float(abs(b[cb]) / np.abs(b).sum()) if np.abs(b).sum() > 0 else None,
                'within_top_channel': cw, 'within_top_channel_share': float(abs(w[cw]) / np.abs(w).sum()) if np.abs(w).sum() > 0 else None,
                'within_top1_row_share': float(tops[0] / tops.sum()) if tops.sum() > 0 else None,
                'within_top5_rows_share': float(tops[:5].sum() / tops.sum()) if tops.sum() > 0 else None,
                'within_top_rows': [{'window': int(win[i]), 'position': int(pos[i]), 'token': tok3.decode([int(tokflat[i])]), 'spike': bool(S[i]),
                                     'contribution': float(rowc[i])} for i in order[:5]],
                'pair_position0_share': float(pair_row[pos == 0].sum() / max(pair_row.sum(), 1e-300)),
                'pair_spike_share': float(pair_row[S].sum() / max(pair_row.sum(), 1e-300))}
            srows.append({'layer': l, 'format': f, 'n': rf['n'], 'spike_rows': nS, 'low_coverage': bool(0 < nS < LOW_COVER),
                          'grouped': rf['grouped'], 'threshold': thr, 'T_old': rf['T_old'], 'T_group': rf['T_group'], 'T_pair': rf['T_pair'],
                          'V_old': rf['V_old'], 'V_group': rf['V_group'], 'V_pair': rf['V_pair'], 'U': rf['U']})
            drows.append({'layer': l, 'format': f, 'spike_rows': nS, 'between': float(b.sum()), 'within': float(w.sum()),
                          'between_rel_V_old': float(b.sum()) / rf['V_old'], 'within_rel_V_old': float(w.sum()) / rf['V_old'],
                          **{k: mech[f'L{l}']['formats'][f][k] for k in ('between_top_channel', 'between_top_channel_share', 'within_top_channel',
                                                                          'within_top_channel_share', 'within_top1_row_share', 'within_top5_rows_share',
                                                                          'pair_position0_share', 'pair_spike_share')}})
        # leave-one-window-out (threshold fixed)
        t1 = time.perf_counter()
        for wdrop in range(CAL):
            rw = scores(h, g, errs, thr, win, keep=win != wdrop)
            for f in FORMATS:
                lrows.append({'layer': l, 'window_removed': wdrop, 'format': f, 'spike_rows': rw[f]['nS'],
                              **{mth: rw[f][mth] for mth in METHODS}})
        tm['leave_one_window_out'] += time.perf_counter() - t1
        log(f'L{l}: spike rows {nS} (thr {thr:.3g}); ' + ', '.join(f'{f} {r[f]["T_old"]:.3g}/{r[f]["T_group"]:.3g}/{r[f]["T_pair"]:.3g}' for f in FORMATS))
    natr = np.array([x['rel'] for x in nat])
    ctl.update({'fp32_identity_zero_error_and_scores': fp32_zero, 'identity_between_rel_max': id_rel['between'],
                'identity_within_rel_max': id_rel['within'],
                'native_C_x_plus_y': {'median_rel': float(np.median(natr)), 'max_rel': float(natr.max()),
                                      'worst': sorted(nat, key=lambda x: -x['rel'])[:5]}})
    ctl['passed'] = bool(ctl['calibration_tokens_equal_prepare'] and ctl['wrapper_equals_prepare_bitwise'] and ctl['rms_norm_eps'] == [1e-6, 1e-6]
                         and all(v[0] == 0 for v in ctl['posit_lut_mismatches'].values()) and hook_rel <= TOL['hook_rel'] and y_rel <= TOL['hook_rel']
                         and grad_rel <= TOL['grad_rel'] and last_zero and ctl['loss_with_hooks_vs_no_hooks_max_abs'] == 0.0 and fp32_zero
                         and id_rel['between'] <= TOL['identity_rel'] and id_rel['within'] <= TOL['identity_rel']
                         and ctl['native_C_x_plus_y']['median_rel'] <= TOL['native_C_median_rel'] and ctl['native_C_x_plus_y']['max_rel'] <= TOL['native_C_max_rel'])
    write(out / 'controls_calib.json', ctl)
    if not ctl['passed']:
        raise SystemExit('calibration controls failed (see controls_calib.json); no freeze')
    # caches for the GPU quantizer check (float32 h per layer) and re-analysis
    t1 = time.perf_counter()
    for l in layers:
        np.save(work / f'h_L{l}.npy', H[l])
        np.save(work / f'g_L{l}.npy', G[l])
    tm['cache_write'] = time.perf_counter() - t1
    for name, rows in (('scores.csv', srows), ('decomposition.csv', drows), ('lowo.csv', lrows), ('native_consistency.csv', nat)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    write(out / 'mechanism.json', mech)
    # predictions frozen
    T = {(r['layer'], r['format']): r for r in srows}
    LO = defaultdict(dict)
    for r in lrows:
        LO[(r['layer'], r['window_removed'])][r['format']] = r
    preds = []
    for l in layers:
        for f1, f2, prim in pairs_all():
            row = {'layer': l, 'f1': f1, 'f2': f2, 'primary': prim}
            for mth in METHODS:
                d, tie = pick_and_tie(T[(l, f1)][mth], T[(l, f2)][mth])
                row[f'{mth}_diff'], row[f'{mth}_tie'] = d, tie
                row[f'{mth}_pick'] = f1 if d <= 0 else f2
                row[f'{mth}_lowo_flips'] = sum(((LO[(l, w)][f1][mth] - LO[(l, w)][f2][mth]) <= 0) != (d <= 0) for w in range(CAL))
            preds.append(row)
    with open(out / 'predictions.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(preds[0]))
        wr.writeheader()
        wr.writerows(preds)
    sizes = {'cache_h_g_bytes_per_layer': int(H[layers[0]].nbytes + G[layers[0]].nbytes),
             'extra_stats_per_layer_float64_6_formats': {'group': int((1 + 1 + 2 * len(FORMATS)) * 3072 * 8),
                                                         'pair': int(2 * len(FORMATS) * 3072 * 8),
                                                         'note': 'group: sum_S g^2 per channel and n_S, per format sum_S e and sum_S e^2; '
                                                                 'pair: per format sum g^2 e^2 and sum g^2 e; estimates, not a native build'}}
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'),
                                'controls_sha256': sha(out / 'controls_calib.json'), 'scores_sha256': sha(out / 'scores.csv'),
                                'predictions_sha256': sha(out / 'predictions.csv'), 'lowo_sha256': sha(out / 'lowo.csv'),
                                'error_sha256_cpu': shas, 'script_sha256': sha(HERE / 'run_qwen_down_spike_diag.py')})
    log_stage(out, 'calib', time.perf_counter() - t0, {k: v for k, v in tm.items()} | {'sizes': sizes})
    log(f'calib frozen; controls {json.dumps({k: v for k, v in ctl.items() if k not in ("calibration_losses", "native_C_x_plus_y", "synthetic")})}; '
        f'native x+y {ctl["native_C_x_plus_y"]["median_rel"]:.2e}/{ctl["native_C_x_plus_y"]["max_rel"]:.2e}')


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def data(out, log):
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
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
        reason = ('near_duplicate_wt2' if c2 >= mr.NEARDUP else 'near_duplicate_previous' if cp >= mr.NEARDUP else
                  'near_duplicate_accepted' if ca >= mr.NEARDUP else 'historical_qwen_overlap' if max(qw, qa) >= mr.NEARDUP else None)
        if reason:
            skips[reason] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': reason,
                         'shares': {'wt2': c2, 'previous': cp, 'accepted': ca, 'qwen_window': qw, 'qwen_article': qa}})
            continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w)
        acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'file_lines': [a['line_start'], a['line_end']],
                     'title': a['title'], 'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                     'window_tokens': [st, st + WPA * SEQ], 'window_chars_in_article': [int(off[st][0]), int(off[st + WPA * SEQ - 1][1])],
                     'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca, 'qwen_window': qw, 'qwen_article': qa},
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(), 'windows': [WPA * len(segs) + i for i in range(WPA)]})
        if len(segs) == N_ART:
            break
    if len(segs) < N_ART or {s_['content_sha256'] for s_ in segs} & used:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'exclusions.json', {'skipped_during_draw': excl, 'previous_wt103_excluded_by_scan': sorted(used),
                                    'wt2_and_internal_duplicates_excluded_before_draw': len(w103) - len(base)})
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'),
                                     'audit_sha256': sha(out / 'audit.json'), 'n_articles': N_ART, 'windows_per_article': WPA,
                                     'eligible': len(elig), 'visited': N_ART + sum(skips.values()), 'skips': dict(skips),
                                     'excluded_previous_wt103': len(used), 'tokens_sha256': sha(out / 'tokens.npy'),
                                     'segments_sha256': sha(out / 'segments.json'), 'exclusions_sha256': sha(out / 'exclusions.json')})
    log_stage(out, 'data', time.perf_counter() - t0)
    log(f'data: {N_ART} articles; skips {dict(skips)}')


# --------------------------------------------------------------------------- #
# eval (GPU)
# --------------------------------------------------------------------------- #
def eval_(out, log):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    t0 = time.perf_counter()
    pl = check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'segments.json') != dz['segments_sha256']:
        raise SystemExit('freeze / data changed')
    for k in ('scores', 'predictions', 'lowo'):
        if sha(out / f'{k}.csv') != fz[f'{k}_sha256']:
            raise SystemExit(f'{k}.csv changed after the freeze')
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    layers = pl['model']['layers']
    dn = down_nodes()
    ranges = read(NATIVE / 'ranges_by_point.json')
    work = out / 'work'
    env = dr.env_info()
    env['cuda_mem_free_total'] = list(torch.cuda.mem_get_info())
    Q = Quant(work, dev)
    # control: GPU QDQ = CPU QDQ (sha of float32 errors) on the calibration h of every layer and format
    t1 = time.perf_counter()
    bad = []
    with torch.no_grad():
        for l in layers:
            ht = torch.from_numpy(np.load(work / f'h_L{l}.npy')).to(dev)
            for f in FORMATS:
                e = (Q.act(ht, f, ranges[dn[l]]['x']) - ht).cpu().numpy()
                if hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() != fz['error_sha256_cpu'][f'L{l}|{f}']:
                    bad.append(f'L{l}|{f}')
    ctl = {'environment': env, 'gpu_qdq_equals_cpu': not bad, 'gpu_qdq_mismatches': bad, 'qdq_check_s': time.perf_counter() - t1,
           'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
           'deterministic': torch.are_deterministic_algorithms_enabled()}
    if bad:
        write(out / 'controls_eval.json', ctl)
        raise SystemExit(f'GPU QDQ differs from CPU: {bad[:5]}')
    M = Model(dev)
    toks = np.load(out / 'tokens.npy')
    nw = len(toks)
    F7 = ('fp32',) + FORMATS
    cap = {}
    kw = {}

    def pre_mid(l):
        def hook(mod, inp):
            cap[(l, 'mid')] = inp[0]
        return hook

    def pre_down(l):
        def hook(mod, inp):
            cap[(l, 'h')] = inp[0]
        return hook

    def pre_kw(mod, args, kwargs):
        kw.update(kwargs)

    def head_ce(normed, ids):
        """LM head and summed CE in sub-batches of LM_SUB windows (full-chunk logits do not fit the GPU)."""
        ce = []
        for i in range(0, normed.shape[0], LM_SUB):
            z = M.model.lm_head(normed[i:i + LM_SUB])
            ce.append(summed_ce_rows(torch, z, ids[i:i + LM_SUB]))
            del z
        return torch.cat(ce)

    def forward_ce(ids):
        """the model's own forward (decoder + final norm, constant causal mask as the wrapper), then head_ce."""
        hs = M.model.model(input_ids=ids, attention_mask=M.wrapper.mask, use_cache=False).last_hidden_state
        return head_ce(hs, ids)

    def downstream(hidden, l, ids):
        for layer in M.layers[l + 1:]:
            o = layer(hidden, **kw)
            hidden = o[0] if isinstance(o, tuple) else o
        return head_ce(M.model.model.norm(hidden), ids)
    ce = np.full((len(layers), len(F7), nw), np.nan)
    full = np.full(nw, np.nan)
    id_diff = []
    t_fwd = t_rep = 0.0
    torch.cuda.reset_peak_memory_stats()
    for c0 in range(0, nw, CHUNK):
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).to(dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            nohook = forward_ce(ids)
            hooks = [M.layers[l].post_attention_layernorm.register_forward_pre_hook(pre_mid(l)) for l in layers]
            hooks += [M.layers[l].mlp.down_proj.register_forward_pre_hook(pre_down(l)) for l in layers]
            hooks.append(M.layers[1].register_forward_pre_hook(pre_kw, with_kwargs=True))
            hooked = forward_ce(ids)
            for hk in hooks:
                hk.remove()
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t1
        full[c0:c0 + len(ids)] = nohook.cpu().numpy()
        if not torch.equal(nohook, hooked):
            raise SystemExit('hooks changed the forward')
        t1 = time.perf_counter()
        with torch.no_grad():
            for li, l in enumerate(layers):           # identity first: the control precedes any quantized replay
                down = M.layers[l].mlp.down_proj
                ce[li, 0, c0:c0 + len(ids)] = downstream(cap[(l, 'mid')] + down(cap[(l, 'h')]), l, ids).cpu().numpy()
            dmax = float(np.abs(ce[:, 0, c0:c0 + len(ids)] - full[c0:c0 + len(ids)]).max())
            id_diff.append(dmax)
            if dmax > TOL['identity_ce_abs']:
                write(out / 'controls_eval.json', {**ctl, 'identity_vs_nohook_max_abs': id_diff})
                raise SystemExit(f'identity replay differs from the no-hook forward by {dmax}')
            for li, l in enumerate(layers):
                down = M.layers[l].mlp.down_proj
                for fi, f in enumerate(FORMATS, start=1):
                    hq = Q.act(cap[(l, 'h')], f, ranges[dn[l]]['x'])
                    ce[li, fi, c0:c0 + len(ids)] = downstream(cap[(l, 'mid')] + down(hq), l, ids).cpu().numpy()
        torch.cuda.synchronize()
        t_rep += time.perf_counter() - t1
        cap.clear()
        np.savez(work / 'eval_partial.npz', ce=ce, full=full)
        log(f'chunk {c0 // CHUNK + 1}/{-(-nw // CHUNK)}: identity max |dCE| {dmax:.2e}; forward {t_fwd:.0f}s, replay {t_rep:.0f}s')
    if np.isnan(ce).any() or np.isnan(full).any():
        raise SystemExit('missing values')
    np.savez(out / 'eval_ce.npz', ce=ce, full=full, layers=np.array(layers), formats=np.array(F7))
    ctl.update({'identity_vs_nohook_max_abs_per_chunk': id_diff, 'identity_vs_nohook_max_abs': max(id_diff),
                'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()), 'passed': True})
    write(out / 'controls_eval.json', ctl)
    log_stage(out, 'eval', time.perf_counter() - t0, {'qdq_check_s': ctl['qdq_check_s'], 'fp32_forward_s': t_fwd, 'replay_s': t_rep,
                                                      'peak_gpu_memory_bytes': ctl['peak_gpu_memory_bytes'], 'windows': nw,
                                                      'eval_ce_sha256': sha(out / 'eval_ce.npz')})
    log(f'eval done: forward {t_fwd:.0f}s, replay {t_rep:.0f}s, identity max {max(id_diff):.2e}')


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out, log):
    t0 = time.perf_counter()
    pl = check_plan(out)
    fz = read(out / 'freeze.json')
    for k in ('scores', 'predictions', 'lowo'):
        if sha(out / f'{k}.csv') != fz[f'{k}_sha256']:
            raise SystemExit(f'{k}.csv changed after the freeze')
    layers = pl['model']['layers']
    cc, ce_ctl, au = read(out / 'controls_calib.json'), read(out / 'controls_eval.json'), read(out / 'audit.json')
    if not (cc['passed'] and ce_ctl['passed']):
        raise SystemExit('controls did not pass')
    write(out / 'controls.json', {'audit_tokenizer': au['controls'], 'historical_qwen_unresolved_files': au['unresolved_files'],
                                  'calibration': cc, 'evaluation': ce_ctl,
                                  'statement': 'offline PyTorch QDQ diagnostic (calibration CPU, truth GPU); not an onnx-mlir native '
                                               'integration; table C (input + output) used only as a consistency control'})
    z = np.load(out / 'eval_ce.npz')
    ce, full = z['ce'], z['full']
    F7 = [str(x) for x in z['formats']]
    segs = read(out / 'segments.json')
    na = len(segs)
    art = lambda v: v.reshape(*v.shape[:-1], na, WPA).sum(-1) / (WPA * TP)       # windows -> article NLL
    nll = art(ce)                                   # [L, F7, A]
    dn = nll[:, 1:] - nll[:, :1]                    # [L, 6, A] dNLL vs the same layer's identity replay
    fi = {f: i for i, f in enumerate(FORMATS)}
    li = {l: i for i, l in enumerate(layers)}
    S_ = {(int(r['layer']), r['format']): r for r in csv.DictReader(open(out / 'scores.csv'))}
    preds = list(csv.DictReader(open(out / 'predictions.csv')))
    for p in preds:
        p['layer'] = int(p['layer'])
        p['primary'] = p['primary'] == 'True'
        for m in METHODS:
            p[f'{m}_diff'] = float(p[f'{m}_diff'])
            p[f'{m}_tie'] = p[f'{m}_tie'] == 'True'
            p[f'{m}_lowo_flips'] = int(p[f'{m}_lowo_flips'])
    # per article / per window outputs
    with open(out / 'per_article_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'article', 'title', 'nll_fp32_identity', 'nll_quantized', 'dnll', 'T_old', 'T_group', 'T_pair'])
        for l in layers:
            for f in FORMATS:
                s_ = S_[(l, f)]
                for a in range(na):
                    wr.writerow([l, f, a, segs[a]['title'], nll[li[l], 0, a], nll[li[l], fi[f] + 1, a], dn[li[l], fi[f], a], s_['T_old'], s_['T_group'], s_['T_pair']])
    with open(out / 'per_window_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', 'window', 'article', 'ce_fp32_identity', 'ce_quantized', 'dce', 'ce_no_hook_forward', 'T_old', 'T_group', 'T_pair'])
        for l in layers:
            for f in FORMATS:
                s_ = S_[(l, f)]
                for w in range(ce.shape[2]):
                    wr.writerow([l, f, w, w // WPA, ce[li[l], 0, w], ce[li[l], fi[f] + 1, w], ce[li[l], fi[f] + 1, w] - ce[li[l], 0, w], full[w],
                                 s_['T_old'], s_['T_group'], s_['T_pair']])
    # bootstrap weights: one article resample for every layer / format / pair / method
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na   # [B, A]
    Dm = np.stack([dn[li[p['layer']], fi[p['f1']]] - dn[li[p['layer']], fi[p['f2']]] for p in preds])      # [P, A]
    mean = Dm.mean(1)
    bs = Wb @ Dm.T                                                                                           # [B, P]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    se = bs.std(0, ddof=1)
    prim = np.array([p['primary'] for p in preds])
    tmax = np.abs((bs[:, prim] - mean[prim]) / se[prim]).max(1)
    qsim = float(np.quantile(tmax, 1 - ALPHA))
    rows = []
    for k, p in enumerate(preds):
        dec = bool(lo[k] > 0 or hi[k] < 0)
        r = {'layer': p['layer'], 'f1': p['f1'], 'f2': p['f2'], 'primary': p['primary'], 'D_mean': float(mean[k]), 'ci95': [float(lo[k]), float(hi[k])],
             'decisive': dec, 'articles_f1_worse': int((Dm[k] > 0).sum())}
        if p['primary']:
            sl, sh = mean[k] - qsim * se[k], mean[k] + qsim * se[k]
            r['simultaneous_ci95'] = [float(sl), float(sh)]
            r['confirmed'] = bool(sl > 0 or sh < 0)
        for m in METHODS:
            sign_ok = np.sign(p[f'{m}_diff']) == np.sign(mean[k])
            r[m] = {'diff': p[f'{m}_diff'], 'tie': p[f'{m}_tie'], 'pick': p[f'{m}_pick'], 'lowo_flips': p[f'{m}_lowo_flips'],
                    'right': bool(dec and sign_ok and not p[f'{m}_tie']), 'sign_agrees_point': bool(sign_ok)}
        rows.append(r)

    def counts(sel):
        rs = [r for r in rows if sel(r)]
        d = [r for r in rs if r['decisive']]
        c = {'pairs': len(rs), 'decisive': len(d), 'uncertain': len(rs) - len(d)}
        for m in METHODS:
            c[f'{m}_right'] = sum(r[m]['right'] for r in d)
            c[f'{m}_ties_decisive'] = sum(r[m]['tie'] for r in d)
            c[f'{m}_ties_all'] = sum(r[m]['tie'] for r in rs)
            c[f'{m}_sign_agrees_point_all'] = sum(r[m]['sign_agrees_point'] for r in rs)
        for m in ('T_group', 'T_pair'):
            fx = [r for r in d if r[m]['right'] and not r['T_old']['right']]
            ne = [r for r in d if r['T_old']['right'] and not r[m]['right']]
            c[f'{m}_vs_old'] = {'fixes': len(fx), 'new_errors': len(ne), 'net': len(fx) - len(ne),
                                'fix_pairs': [f"L{r['layer']} {r['f1']}-{r['f2']}" for r in fx],
                                'new_error_pairs': [f"L{r['layer']} {r['f1']}-{r['f2']}" for r in ne],
                                'fix_layers': sorted({r['layer'] for r in fx}), 'new_error_layers': sorted({r['layer'] for r in ne}),
                                'confirmed_fixes': [f"L{r['layer']} {r['f1']}-{r['f2']}" for r in fx if r.get('confirmed')],
                                'confirmed_new_errors': [f"L{r['layer']} {r['f1']}-{r['f2']}" for r in ne if r.get('confirmed')],
                                'fixes_stable_lowo': sum(r[m]['lowo_flips'] == 0 for r in fx)}
        return c
    res = {'controls': {'freeze_hashes_ok': True}, 'simultaneous_quantile': qsim,
           'counts': {'primary': counts(lambda r: r['primary']), 'secondary': counts(lambda r: not r['primary']),
                      **{f'primary {a}-{b}': counts(lambda r, a=a, b=b: r['primary'] and r['f1'] == a and r['f2'] == b) for a, b in SAME_BIT},
                      'primary known layers': counts(lambda r: r['primary'] and r['layer'] in KNOWN_LAYERS),
                      'primary other layers': counts(lambda r: r['primary'] and r['layer'] not in KNOWN_LAYERS)}}
    # selection regret on the primary pairs
    pp = [k for k, p in enumerate(preds) if p['primary']]
    pick = {m: np.stack([dn[li[preds[k]['layer']], fi[preds[k][f'{m}_pick']]] for k in pp]) for m in METHODS}   # [84, A]
    oracle_pick = np.stack([dn[li[preds[k]['layer']], fi[preds[k]['f1'] if mean[k] <= 0 else preds[k]['f2']]] for k in pp])
    R = {m: v.mean(0) for m, v in pick.items()}                                                               # [A]
    R['oracle_in_sample'] = oracle_pick.mean(0)
    sel = {m: {'mean': float(R[m].mean())} for m in R}
    pvals = {}
    for m in ('T_group', 'T_pair'):
        d = R[m] - R['T_old']
        b = Wb @ d
        p2 = float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean())))
        pvals[m] = p2
        sel[f'{m}-T_old'] = {'mean': float(d.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(),
                             'ci97.5_bonferroni': np.percentile(b, [1.25, 98.75]).tolist(), 'p_two_sided': p2,
                             'differing_picks': int(sum(preds[k][f'{m}_pick'] != preds[k]['T_old_pick'] for k in pp)),
                             'articles_better': int((d < 0).sum()), 'articles_worse': int((d > 0).sum())}
    order_ = sorted(pvals, key=pvals.get)
    for i, m in enumerate(order_):
        sel[f'{m}-T_old']['p_holm'] = float(min(1.0, max(pvals[mm] * (len(order_) - j) for j, mm in enumerate(order_[:i + 1]))))
    for (a, b_) in SAME_BIT:
        ks = [i for i, k in enumerate(pp) if preds[k]['f1'] == a and preds[k]['f2'] == b_]
        for m in ('T_group', 'T_pair'):
            d = pick[m][ks].mean(0) - pick['T_old'][ks].mean(0)
            bb = Wb @ d
            sel[f'{m}-T_old {a}-{b_}'] = {'mean': float(d.mean()), 'ci95': np.percentile(bb, [2.5, 97.5]).tolist(),
                                         'differing_picks': int(sum(preds[pp[i]][f'{m}_pick'] != preds[pp[i]]['T_old_pick'] for i in ks))}
    res['selection'] = sel
    # decision
    dec = {}
    for m in ('T_group', 'T_pair'):
        c = res['counts']['primary'][f'{m}_vs_old']
        outside = [l for l in c['fix_layers'] if l not in KNOWN_LAYERS]
        c1 = sel[f'{m}-T_old']['ci97.5_bonferroni'][1] < 0
        c2 = c['net'] >= 3 and len(c['fix_layers']) >= 3 and len(outside) >= 2
        c3 = c['fixes'] > 0 and c['fixes_stable_lowo'] >= 2 / 3 * c['fixes']
        dec[m] = {'regret_confirmed': bool(c1), 'spread': bool(c2), 'stable': bool(c3), 'fix_layers_outside_known': outside,
                  'support_next_round': bool(c1 and c2 and c3)}
    res['decision'] = dec
    # descriptive: layers with net > 0 / < 0
    per_layer = {}
    for l in layers:
        d = [r for r in rows if r['primary'] and r['layer'] == l and r['decisive']]
        per_layer[f'L{l}'] = {'decisive': len(d), **{m: sum(r[m]['right'] for r in d) for m in METHODS}}
    res['per_layer_primary'] = per_layer
    res['dnll_summary'] = {f: {'mean_over_layers': float(dn[:, fi[f]].mean()), 'min_layer_mean': float(dn[:, fi[f]].mean(1).min()),
                               'max_layer_mean': float(dn[:, fi[f]].mean(1).max()),
                               'negative_layer_means': int((dn[:, fi[f]].mean(1) < 0).sum())} for f in FORMATS}
    res['dnll_by_layer'] = {f'L{l}': {f: float(dn[li[l], fi[f]].mean()) for f in FORMATS} for l in layers}
    write(out / 'ranking.json', {'pairs': rows, 'summary': res})
    with open(out / 'ranking_pairs.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'f1', 'f2', 'primary', 'D_mean', 'ci_lo', 'ci_hi', 'decisive', 'sim_lo', 'sim_hi', 'confirmed',
                     *[f'{m}_{k}' for m in METHODS for k in ('diff', 'tie', 'pick', 'right', 'lowo_flips')]])
        for r in rows:
            wr.writerow([r['layer'], r['f1'], r['f2'], r['primary'], r['D_mean'], *r['ci95'], r['decisive'],
                         *(r.get('simultaneous_ci95') or ['', '']), r.get('confirmed', ''),
                         *[r[m][k] for m in METHODS for k in ('diff', 'tie', 'pick', 'right', 'lowo_flips')]])
    write(out / 'summary.json', res)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'primary': {k: v for k, v in res['counts']['primary'].items() if not isinstance(v, dict)},
                    'group_vs_old': res['counts']['primary']['T_group_vs_old'], 'pair_vs_old': res['counts']['primary']['T_pair_vs_old'],
                    'selection': {k: v for k, v in sel.items() if '-' in k and ' ' not in k}, 'decision': dec}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'audit', 'calib', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty: use a new directory')
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / 'plan.json').exists():
        raise SystemExit('run --stage plan first')
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'work').mkdir(exist_ok=True)
    logf = open(out / 'logs' / f'{args.stage}.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    t0 = time.time()
    {'plan': plan, 'audit': audit, 'calib': calib, 'data': data, 'eval': eval_, 'analyze': analyze}[args.stage](out, log)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': args.stage, 'seconds': time.time() - t0, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                             'script_sha256': sha(HERE / 'run_qwen_down_spike_diag.py')}) + '\n')


if __name__ == '__main__':
    main()
