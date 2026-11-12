#!/usr/bin/env python3
"""OLMoE token-level activation spikes at the Down input and BC ranking: a diagnostic. Pre-registered in plan.json / plan.md.

Follows runs/olmoe_act_outlier_diag_20261023 (channel outliers). Question: does OLMoE show the token-level spikes of
"Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs" (arXiv 2405.14428) at the Down input
h = SiLU(gate) * up, do they increase quantization error and measured damage, and does BC mis-rank more there.
No criterion, quantizer, search, knee or pipeline change; no fine-tuning; no QFeM / QFeP / SmoothQuant; diagnostic only.

Metrics (never mixed up): per token s_t = max_c |h[t, c]|; per expert R_token = max_t s_t / median_t s_t;
R_channel = max_c(max_t |h[t, c]|) / median_c(max_t |h[t, c]|) (the previous round's R_cross).

Stages (one --out-dir): plan, discovery, select, freeze, data, measure, analyze, figures.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_act_outlier_diag as od
import run_olmoe_dynamic_route as dr
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss

HERE = Path(__file__).resolve().parent
PREV = HERE / 'runs/olmoe_act_outlier_diag_20261023'
RANGES, CALIB = ss.RANGES, ss.CALIB
ACT_CACHE, ACT_INDEX, GC, GC_INDEX, Q0K = od.ACT_CACHE, od.ACT_INDEX, od.GC, od.GC_INDEX, od.Q0K
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
NCAL = 10                                    # calibration windows come first in the discovery batch
CANDS = od.CANDS
TAU, TAU_STRONG, TAU_EXTRA = 6.0, 20.0, 100.0  # spike token: s_t >= TAU x median_t s_t (expert, FP32 discovery)
OUT_MULT = od.OUT_MULT                        # outlier channel: max_t |h_c| >= 6 x median_c (as in the previous round)
MIN_CAL_ROWS, MIN_ALL_ROWS, N0, NSUB, WIN_MIN = 64, 512, 256, 200, 8
STRONG_Q, WEAK_Q, N_PAIRS, N_DIS, MAX_STRONG_PER_LAYER = 0.90, 0.50, 4, 2, 2
KNOWN = 'L1.E9'
FRONT, BACK = (0, 1), (14, 15)
N_ART, WPA_C, CHUNK, VAR_BATCH = 96, 3, 128, 12
BOOT, BC_TIE = 10000, 0.02
SEEDS = {'order': 23201, 'start': 23202, 'bootstrap': 23211, 'tap_bootstrap': 23212, 'select': 23221, 'subsample': 23231}
BUDGET_MIN = {'discovery': 10, 'select': 4, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 40}
SAME_BIT = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_token_spike_diag.py', 'run_olmoe_act_outlier_diag.py'] + ss.CODE + dr.SCRIPTS))}


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def input_files():
    return [Q0K, ACT_CACHE, ACT_INDEX, GC, GC_INDEX, RANGES, CALIB, PREV / 'tokens.npy', PREV / 'segments.json', PREV / 'data_freeze.json',
            PREV / 'selected_taps.json', PREV / 'ranking_pairs.csv', PREV / 'report.md'] + mr.WT103 + mr.WT2


def disc_tokens():
    """discovery windows: the BC calibration (10 x 128, WikiText-2 validation) then the previous round's confirmation windows
    (64 articles x 3, WikiText-103 train; results already seen -> discovery only)."""
    return np.concatenate([np.load(CALIB), np.load(PREV / 'tokens.npy')]).astype(np.int64)


def token_category(text, tid, eos):
    if tid == eos:
        return 'eos'
    if '\n' in text:
        return 'newline'
    st = text.strip()
    if st == '':
        return 'space'
    if all(unicodedata.category(ch).startswith('P') for ch in st):
        return 'punct'
    if st.isdigit():
        return 'digit'
    return 'word'


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    prev_taps = read(PREV / 'selected_taps.json')['taps']
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'reference': 'Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs, arXiv 2405.14428 (token-wise max of the down-projection '
                     'input, max / median ratio); only the measurement idea is used, not QFeM / QFeP',
        'questions': ['1 do token spikes exist at the OLMoE Down input (per expert, FP32 and q0)', '2 where (layers, experts, tokens, positions)',
                      '3 how do token spikes differ from channel outliers', '4 do spikes increase quantization error and measured damage',
                      '5 does BC mis-rank more where spikes are', '6 is L1.E9 an isolated case', '7 is an intervention round needed for causality'],
        'fixed': {'model': 'OLMoE-1B-7B, original checkpoint and router', 'quantizer': 'dev_quant(corrected=True), fixed calibration ranges (ranges.json)',
                  'q0': str(Q0K.relative_to(HERE)), 'bc': 'static caches (activation errors and G2 / GS from the BC calibration)',
                  'not': 'no criterion / quantizer / search / knee / pipeline change, no fine-tuning, no QFeM / QFeP / SmoothQuant, no new method'},
        'quantization_facts': {'int_activation': 'one scalar scale / zero-point per physical QDQ position from the FP32-calibration [min, max] widened to '
                                                 'contain 0; per-tensor, so a spike in calibration widens the step of every token and channel, and a spike '
                                                 'beyond the calibration range is clipped', 'posit_activation': 'direct LUT rounding, no scale, no range; '
                                                 'the INT range mechanism does not apply', 'bc': 'per-channel G2 / GS and per-channel error sums S1 / S2 '
                                                 'over the calibration rows; T = sum_c G2_c / n (S2_c - n mu_c^2) + GS_c mu_c^2'},
        'data_use': {'discovery': f'BC calibration ({NCAL} x {SEQ}, WikiText-2 validation, contiguous) + the previous round\'s confirmation windows '
                                  '(64 WikiText-103 articles x 3; results already seen) = 202 windows; FP32 and q0 forward-only, own dynamic routes',
                     'bc_decomposition': 'calibration windows only (the rows BC uses)',
                     'confirmation': f'{N_ART} new WikiText-103 train articles x {WPA_C} windows of {SEQ}, drawn after the freeze; every earlier manifest '
                                     f'excluded ({len(used)} articles), duplicate and near-duplicate (shingle) exclusion against WikiText-2, earlier '
                                     'articles and accepted ones', 'eligible_before_length_and_neardup': len(elig), 'manifests_scanned': manifests,
                     'excluded_previous_wt103': len(used), 'no_extra_data': 'no articles added after results, whatever the significance'},
        'metrics': {'s_t': 'max_c |h[t, c]| for every (layer, expert, routed token); argmax channel c*_t; second-largest |h| of the row; ||h_t||^2',
                    'R_token': 'max_t s_t / median_t s_t per expert (all its discovery tokens); undefined if the median is 0 or rows < 2',
                    'R_channel': 'max_c(max_t |h[t,c]|) / median_c(max_t |h[t,c]|) per expert, same tokens; R_channel_cal on the calibration rows only '
                                 '(= the previous round\'s R_cross); identity: log R_token - log R_channel = log median_c(max_t) - log median_t(max_c)',
                    'coverage': f'rows per expert (all / calibration / previous) and windows; R_token@{N0} = mean over {NSUB} subsamples of {N0} rows '
                                f'without replacement (rows >= {MIN_ALL_ROWS}; seed {SEEDS["subsample"]}); spike rate per 1000 rows; '
                                'Spearman of rows with each metric',
                    'per_window': f'R_token within each window where the expert has >= {WIN_MIN} rows; median and max over windows; windows with a spike',
                    'spike': f'spike token: s_t >= {TAU} x median_t s_t (primary); strong spike >= {TAU_STRONG}x; extreme >= {TAU_EXTRA}x; '
                             'energy share of spike rows in sum ||h_t||^2',
                    'layer_level': 'MoE analogue of the dense per-layer token scale: s_t^layer = max over the token\'s 8 routed experts of s_t; '
                                   'R_token^layer = max / median over all tokens of the layer (secondary, labelled)',
                    'identity': 'for spikes: source, article title, article / split token offset, window, window position, token id and text, '
                                'category (eos / newline / space / punct / digit / word; OLMoE has no BOS token and windows carry no special tokens), '
                                'first occurrence of the id in the window, expert, top-k slot and routing weight, c*, s_t, SiLU(g) and up at c*; '
                                'window position 0 is not the article start and not a BOS; expert-internal row order is not a position',
                    'identity_reporting': 'token ids, window positions and contexts reported separately with base rates (lift); co-occurrence is not '
                                          'a cause', 'gate_up': 'for strong spikes: log|SiLU(g)| and log|up| at c* minus the expert medians of the '
                                          'same quantities at each token\'s own c*; a source description, not evidence of error amplification',
                    'layers': f'layer x expert tables of R_token, R_channel, max |h|, rows; per-layer distributions over experts (not only maxima); '
                              f'front {FRONT} vs back {BACK} vs middle described, all layers kept'},
        'errors': {'candidates': list(CANDS), 'per_expert': 'FP32 discovery, fixed ranges: error energy and per-element MSE of spike vs normal rows; '
                                                           'INT clipping fraction and clipped energy for spike vs normal rows (calibration vs '
                                                           'previous-source rows separately); 2 x 2 energy shares spike x outlier channel',
                   'bc_token_decomposition': 'calibration rows, FP32: the variance part splits exactly over rows as v_t = sum_c G2_c / n (e_tc - mu_c)^2 '
                                             '(mu from all rows); the bias part sum_c GS_c mu_c^2 has cross-row terms and is only allocated as '
                                             'b_t = sum_c GS_c mu_c e_tc / n (sums exactly, may be negative); neither is a per-token true BC cost; '
                                             'recomputed S1 / S2 / n checked against the static cache'},
        'selection': {'eligible': f'Down-input taps of searched experts with calibration rows >= {MIN_CAL_ROWS}, discovery rows >= {MIN_ALL_ROWS}, '
                                  f'finite R_token and R_channel (FP32); the {len(prev_taps)} experts used in the previous round excluded',
                      'pairs': f'{N_PAIRS} token-strong taps (R_token >= p{int(100 * STRONG_Q)}) drawn in random order (seed {SEEDS["select"]}), at most '
                               f'{MAX_STRONG_PER_LAYER} per layer; each matched to a token-weak tap (R_token <= p{int(100 * WEAK_Q)}): same layer if available '
                               'else nearest layer, then same q0 Down activation bit class (<= 6 / 7-8 / >= 16), then nearest |log rows|, ties at random; '
                               'mismatches listed', 'disagreement': f'{N_DIS} token-dominant (R_token >= p90, lowest R_channel percentile) and {N_DIS} '
                               'channel-dominant (R_channel >= p90, lowest R_token percentile), deterministic', 'known_case': f'{KNOWN} reported separately '
                               '(previous round\'s misrank cluster); not in any pooled new-case statistic', 'no_bc_info': 'no BC-error or ranking '
                               'information used for the new taps', 'frozen_per_tap': 'spike threshold TAU x median s_t and outlier channels from FP32 '
                               'discovery; BC predictions T(f) from the static caches'},
        'measurement': {'main': 'FP32 background, one Down-input tap quantized with one candidate, dynamic route, dNLL = NLL(tap) - NLL(FP32); layers '
                                'before the tap reused from the FP32 pass (bitwise control: base vs dr.forward, identity variant); downstream routing '
                                'changes recorded; at the tap on the confirmation rows: R_token, spike rows by the frozen threshold, error energy of '
                                'spike vs normal rows, clipping, spike x outlier-channel energy',
                        'size': f'{N_PAIRS * 2 + N_DIS * 2 + 1} taps x {len(CANDS)} = {(N_PAIRS * 2 + N_DIS * 2 + 1) * len(CANDS)} variants on {N_ART} x {WPA_C} '
                                'windows; from the previous round (0.75 s per variant and remaining layer on 192 windows) about 8-23 min',
                        'batches': f'at most {VAR_BATCH} variants per layer pass (memory)'},
        'primary': {'P1': 'same-bit INT vs Posit pairs (int4-p4e1, int6-p6e1, int8-p8e1): BC accuracy on decisive pairs, new taps by role, L1.E9 apart',
                    'P2': 'all 15 pairs per tap: BC accuracy for INT-INT, Posit-Posit, cross-family',
                    'P3': 'counts of decisive, undeterminable (CI includes 0) and BC-tie pairs',
                    'P4': f'token strong vs weak over the {N_PAIRS} matched pairs: (a) mean over pairs of article-level dNLL(strong) - dNLL(weak) for int4 '
                          'and p4e1, article bootstrap, Bonferroni 2 tests -> 97.5% CI; conditional on the selected taps; (b) misranking rate '
                          'strong - weak with a tap bootstrap (descriptive, low power)',
                    'P5': 'R_token vs R_channel: Spearman over the 12 new taps with int4 / p4e1 dNLL and misranking rate (descriptive); the 4 '
                          'disagreement taps reported one by one',
                    'L1.E9': 'replication of the previous round\'s 4 decisive misranks (int4/int6/int8 vs p6e1, int8 vs p8e1): each pair decisive and '
                             'still mis-ranked on the new data or not'},
        'statistics': {'decisive_pair': '95% article-bootstrap CI of the measured difference excludes 0 (classification, not a test); sensitivity at 99%',
                       'bc_tie': f'|T1 - T2| / max(T1, T2) < {BC_TIE}: excluded from accuracy, counted', 'undeterminable': 'pairs with CI including 0 '
                       'are never counted as correct; pairs with |diff| < 1e-6 counted', 'dependence': 'pairs of one tap are not independent cases; '
                       'tap-level summaries and tap bootstraps for strong vs weak', 'bootstrap': f'{BOOT} article bootstraps (seed {SEEDS["bootstrap"]}); '
                       f'tap bootstrap seed {SEEDS["tap_bootstrap"]}', 'ratios': 'no strong / weak ratios restricted to positive dNLL'},
        'stop': 'if the measurement exceeds its budget, stop before the next layer group and analyse the completed taps',
        'no_feedback': 'no tap, candidate, threshold, seed or data change after confirmation results',
        'budget_minutes': BUDGET_MIN, 'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE token-level activation spikes 與 BC 排序（診斷）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 目的與限制',
          '- 參考 arXiv 2405.14428 的 token-wise 尺度指標，只借用量測方式，不實作 QFeM／QFeP。',
          '- 分開回答：token spike 是否存在；是否增加量化誤差與實測損傷；BC 是否因此更容易排錯。',
          '- 不修改 criterion、量化器、搜尋、knee 或 pipeline；不 fine-tune；不加入 SmoothQuant 等方法；本輪沒有干預。', '',
          '## 資料用途',
          f'- discovery：BC calibration（{NCAL} × {SEQ}，WikiText-2 validation）＋上一輪確認資料（64 篇 WikiText-103 × 3 窗口，結果已看過）＝ 202 窗口；FP32 與 q0。',
          '- BC token 分解只用 calibration 窗口（BC 實際使用的 rows）。',
          f'- 確認：凍結後才抽 {N_ART} 篇新 WikiText-103 文章 × {WPA_C} 窗口；排除 {len(used)} 篇歷史文章，並排除重複與近似重複。不因結果增加資料。', '',
          '## 指標（不可混稱）',
          '- s_t = max_c |h[t,c]|；R_token = max_t s_t ÷ median_t s_t（每個 expert）。',
          '- R_channel = max_c(max_t |h[t,c]|) ÷ median_c(max_t |h[t,c]|)；R_channel_cal 只用 calibration（＝上一輪 R_cross）。',
          '- 兩者分子相同；log R_token − log R_channel = log median_c(max_t) − log median_t(max_c)。',
          f'- 覆蓋量：每個 expert 的 rows（全部／calibration／上一輪資料）與窗口數；R_token@{N0}＝{NSUB} 次 {N0}-row 子樣本的平均；spike 率；rows 與各指標的 Spearman。',
          f'- 窗口內 R_token（該窗口 rows ≥ {WIN_MIN}）的中位數與最大值。',
          f'- spike token：s_t ≥ {TAU} × median_t s_t（主要）；強 spike ≥ {TAU_STRONG}×；極端 ≥ {TAU_EXTRA}×。',
          '- 層級 MoE 類比（次要）：s_t^layer = token 的 8 個 expert 中最大的 s_t；R_token^layer = 該層所有 token 的 max ÷ median。',
          '- 身分：來源、文章、偏移、窗口位置、token ID 與文字、類別、窗口內首次出現、expert、top-k 位置與權重、c*、SiLU(g) 與 up。',
          '- OLMoE 沒有 BOS，窗口不含特殊 token；窗口位置 0 不是文章開頭，也不是 BOS。token ID、位置、上下文分開報告，並附 base rate。',
          f'- 層分布：layer × expert 的 R_token、R_channel、最大 |h|、rows；每層畫分布，不只畫最大值；前段 {FRONT}、後段 {BACK} 與中段都保留。', '',
          '## 誤差與 BC 分解',
          f'- 候選：{", ".join(CANDS)}；固定量化器與 range。spike 與一般 rows 的 MSE、誤差能量占比；INT clipping（依來源分開）；spike × outlier channel 的 2×2 占比。',
          '- BC 變異項可精確拆到 row：v_t = Σ_c G2_c/n·(e_tc − μ_c)²。偏差項 Σ_c GS_c μ_c² 有跨 row 項，只能「分攤」成 b_t = Σ_c GS_c μ_c e_tc/n（加總精確、可為負）。兩者都不是每個 token 的真實 BC 成本。重算的 S1／S2／n 與 static cache 比對。', '',
          '## 選點（不使用任何 BC 誤排資訊）',
          f'- 合格：searched expert 的 Down 輸入，calibration rows ≥ {MIN_CAL_ROWS}、discovery rows ≥ {MIN_ALL_ROWS}；排除上一輪用過的 {len(prev_taps)} 個 expert。',
          f'- 配對：隨機（seed {SEEDS["select"]}）抽 {N_PAIRS} 個 token 強 tap（R_token ≥ p90，每層最多 {MAX_STRONG_PER_LAYER} 個），各配一個 token 弱 tap（≤ p50）：同層優先，否則最近層；再同 q0 Down activation 位元類別；再最近 log rows。不匹配處列出。',
          f'- 不一致案例：{N_DIS} 個 token 主導（R_token ≥ p90 中 R_channel 百分位最低者）、{N_DIS} 個 channel 主導（R_channel ≥ p90 中 R_token 百分位最低者）。',
          f'- 已知案例 {KNOWN} 單獨列出，不併入新案例的統計。', '',
          '## 確認測量',
          f'- FP32 背景，一次只量化一個 Down 輸入 tap，dynamic route，6 個候選；共 {(N_PAIRS * 2 + N_DIS * 2 + 1) * len(CANDS)} 個 variant；依上一輪耗時估計 8–23 分鐘。',
          '- 在 tap 上記錄確認資料的 R_token、凍結門檻下的 spike rows、spike／一般 rows 的誤差能量與 clipping。', '',
          '## 主要比較',
          '- P1 同位元 INT／Posit 排序；P2 全部候選（INT-INT、Posit-Posit、跨家族）；P3 決定性／不可判定／BC 平手的數量。',
          f'- P4 token 強 vs 弱（{N_PAIRS} 組配對）：(a) int4 與 p4e1 的平均配對差 ΔNLL(強) − ΔNLL(弱)，文章 bootstrap，Bonferroni 2 個檢定 → 97.5% CI，結論只限選中的 tap；(b) 誤排率差（tap bootstrap，描述性）。',
          '- P5 R_token vs R_channel：12 個新 tap 上與 int4／p4e1 ΔNLL、誤排率的 Spearman（描述性）；4 個不一致案例逐一報告。',
          f'- {KNOWN}：上一輪 4 個決定性誤排是否在新資料上重現（逐對）。', '',
          '## 統計',
          f'- 決定性配對 = 95% 文章 bootstrap CI 排除 0（分類規則，不是檢定）；99% 作為敏感度分析。BC 平手 = 相對差 < {BC_TIE}，不計入準確率。',
          '- 不可判定的配對不算排對；同一 tap 的配對不是獨立案例；不使用只取 ΔNLL 為正的強弱比值。',
          f'- {BOOT} 次文章 bootstrap（seed {SEEDS["bootstrap"]}）；tap bootstrap seed {SEEDS["tap_bootstrap"]}。',
          f'- 預算（分鐘）：{BUDGET_MIN}；測量超過預算則在下一個層組前停止，只分析已完成的 tap。看到確認結果後不改 tap、候選、門檻、seed 或資料。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; previous WT103 {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
def experts_stats(w, x2d, idx, rw, act, cb, E=NE):
    """run_olmoe_knee_error_only.experts_observe op order, no autograd; cb(e, tok, kpos, g, u, h) with post-QDQ g / u and the Down input h."""
    import torch
    import torch.nn.functional as F
    out = torch.zeros_like(x2d)
    mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist()
    for e in hit:
        if e == E:
            continue
        kpos, tok = torch.where(mask[e])
        xe = x2d[tok]
        A = (lambda p, s, t, e=e: act[(e, p, s)](t) if (e, p, s) in act else t)
        g = A('gate_proj', 'y', F.linear(A('gate_proj', 'x', xe), w['gate_proj'][e]))
        u = A('up_proj', 'y', F.linear(A('up_proj', 'x', xe), w['up_proj'][e]))
        h = F.silu(g) * u
        cb(e, tok, kpos, g, u, h)
        d = A('down_proj', 'y', F.linear(A('down_proj', 'x', h), w['down_proj'][e]))
        out.index_add_(0, tok, d * rw[tok, kpos, None])
    return out


def window_meta(toks):
    """per discovery window: source, article title, token offset of the window start (calibration: in the WikiText-2 validation split;
    previous: in the article); calibration titles from the ' = Title = ' header lines of the decoded contiguous text."""
    import olmoe_common as oc
    tk = oc.tokenizer()
    cal = toks[:NCAL].reshape(-1)
    pieces = [tk.decode([int(t)]) for t in cal]
    starts = np.cumsum([0] + [len(x) for x in pieces])
    text = ''.join(pieces)
    heads = [(m.start(1), m.group(1).strip()) for m in re.finditer(r'(?:^|\n) = ([^=\n][^\n]*?) = \n', text)]
    cal_title = []
    for i in range(len(cal)):
        t = [h for c, h in heads if c <= starts[i + 1]]
        cal_title.append(t[-1] if t else None)
    segs = read(PREV / 'segments.json')
    meta = []
    for wi in range(len(toks)):
        if wi < NCAL:
            meta.append({'win': wi, 'source': 'cal', 'title': cal_title[wi * SEQ], 'start_offset': wi * SEQ, 'offset_kind': 'wt2_validation_split_token'})
        else:
            a, w = divmod(wi - NCAL, 3)
            meta.append({'win': wi, 'source': 'prev', 'title': segs[a]['title'], 'start_offset': segs[a]['window_tokens'][0] + w * SEQ,
                         'offset_kind': 'article_token', 'article': a})
    return meta, cal_title


def discovery(out, log):
    """forward-only over the 202 discovery windows in the FP32 and q0 states; per (layer, expert, routed token) Down-input records."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant, quantized_layer
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'discovery_channels.npz').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    toks = disc_tokens()
    meta, cal_title = window_meta(toks)
    write(out / 'discovery_windows.json', {'windows': meta, 'cal_token_titles': cal_title, 'tokens_sha256': hashlib.sha256(toks.tobytes()).hexdigest()})
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    fi = [QUANT.index(f) for f in CANDS]
    a_, b_ = ss.COLOFF['down_x']
    ints = [j for j, f in enumerate(CANDS) if f.startswith('int')]
    chan, bcctl = {}, {}
    for state in ('fp32', 'q0'):
        fm = None if state == 'fp32' else read(Q0K)
        R = defaultdict(list)
        cal_rec = defaultdict(list)
        cm_all, cm_cal, nodes = [], [], []
        with torch.inference_mode():
            cfg, ck = oc.config(), oc.Checkpoint()
            T = torch.from_numpy(toks).to('cuda')
            cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
            g0 = ck.global_('cuda', torch.float32)
            h = F.embedding(T, g0['embed'])
            Hd = cfg.hidden_size
            for l in range(NL):
                w = ck.layer(l, 'cuda', torch.float32)
                wq, hk = (w, None) if fm is None else quantized_layer(w, fm, l, dq, ranges, NE)
                act = pr.act_map(hk)
                B = h.shape[0]
                x = h + oc.attention(wq, oc.rms(h, wq['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
                z = oc.rms(x, wq['ln2'], cfg.rms_norm_eps).reshape(-1, Hd)
                idx, rw = oc.route(wq, z, cfg)

                def cb(e, tok, kpos, gg, uu, hh, l=l, rw=rw):
                    node = f'L{l}.E{e}'
                    n = int(tok.numel())
                    ab = hh.abs()
                    s, cs = ab.max(1)
                    top2 = ab.topk(2, 1).values[:, 1] if hh.shape[1] > 1 else torch.zeros_like(s)
                    ar = torch.arange(n, device=hh.device)
                    cal = tok < NCAL * SEQ
                    cmax = ab.max(0).values
                    nodes.append(node); cm_all.append(cmax.cpu().numpy())
                    cm_cal.append(ab[cal].max(0).values.cpu().numpy() if bool(cal.any()) else np.full(hh.shape[1], np.nan, np.float32))
                    R['layer'].append(np.full(n, l, np.uint8)); R['expert'].append(np.full(n, e, np.uint8))
                    R['win'].append((tok // SEQ).cpu().numpy().astype(np.int16)); R['pos'].append((tok % SEQ).cpu().numpy().astype(np.uint8))
                    R['kslot'].append(kpos.cpu().numpy().astype(np.uint8)); R['rw'].append(rw[tok, kpos].float().cpu().numpy())
                    R['s'].append(s.cpu().numpy()); R['cstar'].append(cs.cpu().numpy().astype(np.int16)); R['top2'].append(top2.cpu().numpy())
                    R['gsilu'].append(F.silu(gg)[ar, cs].cpu().numpy()); R['up'].append(uu[ar, cs].cpu().numpy())
                    R['hn2'].append(hh.double().pow(2).sum(1).float().cpu().numpy())
                    if state != 'fp32':
                        return
                    med_c = float(cmax.median())
                    O = (cmax >= OUT_MULT * med_c) if med_c > 0 else torch.zeros_like(cmax, dtype=torch.bool)
                    rng = ranges.get(f'{node}.down_proj', {}).get('x')
                    e2r = np.full((n, len(CANDS)), np.nan, np.float32); e2o = np.full((n, len(CANDS)), np.nan, np.float32)
                    cln = np.zeros((n, len(ints)), np.int16); clE = np.full((n, len(ints)), np.nan, np.float32)
                    in_cache = node in aidx['row'] and node in gidx['row'] and bool(cal.any())
                    if in_cache:
                        row, grow, n_c = np.asarray(ac[aidx['row'][node]]), np.asarray(gc[gidx['row'][node]]), int(aidx['n'][node])
                        G2 = torch.from_numpy(grow[0, a_:b_].astype(np.float64)).to('cuda'); GS = torch.from_numpy(grow[1, a_:b_].astype(np.float64)).to('cuda')
                        ncal = int(cal.sum())
                        ctl = {'n_own': ncal, 'n_cache': n_c, 'S1_maxdiff_rel': [], 'S2_maxdiff_rel': [], 'T_own': [], 'T_cache': [], 'v_sum': [], 'b_sum': []}
                        cr = {'win': (tok[cal] // SEQ).cpu().numpy().astype(np.int16), 'pos': (tok[cal] % SEQ).cpu().numpy().astype(np.uint8),
                              'v': np.zeros((ncal, len(CANDS))), 'b': np.zeros((ncal, len(CANDS)))}
                    if rng is not None:
                        for j, f in enumerate(CANDS):
                            q = dq.act(hh, f, rng)
                            err = (q - hh).double()
                            e2 = err.pow(2)
                            e2r[:, j] = e2.sum(1).float().cpu().numpy()
                            e2o[:, j] = e2[:, O].sum(1).float().cpu().numpy()
                            if f.startswith('int'):
                                clip = od.int_clip_mask(hh, f, rng)
                                ji = ints.index(j)
                                cln[:, ji] = clip.sum(1).cpu().numpy().astype(np.int16)
                                clE[:, ji] = (e2 * clip).sum(1).float().cpu().numpy()
                            if in_cache:
                                ec = err[cal]
                                S1, S2 = ec.sum(0), ec.pow(2).sum(0)
                                S1c = torch.from_numpy(row[0][fi[j]][a_:b_].astype(np.float64)).to('cuda')
                                S2c = torch.from_numpy(row[1][fi[j]][a_:b_].astype(np.float64)).to('cuda')
                                ctl['S1_maxdiff_rel'].append(float((S1 - S1c).abs().max() / S1c.abs().max().clamp_min(1e-300)))
                                ctl['S2_maxdiff_rel'].append(float((S2 - S2c).abs().max() / S2c.abs().max().clamp_min(1e-300)))
                                mu = S1 / ncal
                                v = ((ec - mu).pow(2) * (G2 / ncal)).sum(1)
                                bb = (ec * (mu * GS / ncal)).sum(1)
                                cr['v'][:, j], cr['b'][:, j] = v.cpu().numpy(), bb.cpu().numpy()
                                muc = S1c / n_c
                                ctl['T_cache'].append(float((G2 / n_c * (S2c - n_c * muc ** 2)).sum() + (GS * muc ** 2).sum()))
                                ctl['T_own'].append(float((G2 / ncal * (S2 - ncal * mu ** 2)).sum() + (GS * mu ** 2).sum()))
                                ctl['v_sum'].append(float(v.sum())); ctl['b_sum'].append(float(bb.sum()))
                    R['e2'].append(e2r); R['e2O'].append(e2o); R['clipn'].append(cln); R['clipE'].append(clE)
                    if in_cache:
                        bcctl[node] = ctl
                        cal_rec['layer'].append(np.full(len(cr['win']), l, np.uint8)); cal_rec['expert'].append(np.full(len(cr['win']), e, np.uint8))
                        for k_ in ('win', 'pos', 'v', 'b'):
                            cal_rec[k_].append(cr[k_])

                h = x + experts_stats(wq, z, idx, rw, act, cb).view(B, SEQ, Hd)
                del wq, hk, w, x, z, idx, rw, act
                log(f'discovery {state}: layer {l} ({time.perf_counter() - t0:.0f}s)')
        np.savez(out / 'work' / f'records_{state}.npz', **{k: np.concatenate(v) for k, v in R.items()})
        if state == 'fp32':
            np.savez(out / 'work' / 'cal_bc_records.npz', **{k: np.concatenate(v) for k, v in cal_rec.items()})
        chan[f'{state}_nodes'] = np.array(nodes); chan[f'{state}_cmax_all'] = np.stack(cm_all); chan[f'{state}_cmax_cal'] = np.stack(cm_cal)
    np.savez(out / 'discovery_channels.npz', **chan)
    write(out / 'discovery_bc_control.json', bcctl)
    import torch as _t
    log_stage(out, 'discovery', time.perf_counter() - t0, {'peak_gpu_gb': _t.cuda.max_memory_allocated() / 2 ** 30})
    log(f'discovery done: {len(chan["fp32_nodes"])} fp32 experts, {len(chan["q0_nodes"])} q0 experts')


# --------------------------------------------------------------------------- #
def load_records(out, state):
    z = np.load(out / 'work' / f'records_{state}.npz')
    return {k: z[k] for k in z.files}


def group_index(rec):
    """record indices per (layer, expert), in record order."""
    key = rec['layer'].astype(np.int64) * NE + rec['expert'].astype(np.int64)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    return {(int(g[0]) // NE, int(g[0]) % NE): o for g, o in zip(np.split(ks, cut), np.split(order, cut))}


def expert_table(out, state, rec, gi, chan):
    """per expert activation metrics (no error terms)."""
    rng = np.random.default_rng(SEEDS['subsample'])
    nodes = list(chan[f'{state}_nodes'])
    cmi = {n: i for i, n in enumerate(nodes)}
    rows = []
    for (l, e), ix in sorted(gi.items()):
        node = f'L{l}.E{e}'
        s = rec['s'][ix].astype(np.float64)
        win = rec['win'][ix]
        n = len(ix)
        med = float(np.median(s))
        cm = chan[f'{state}_cmax_all'][cmi[node]].astype(np.float64)
        cmc = chan[f'{state}_cmax_cal'][cmi[node]].astype(np.float64)
        mc, mcc = float(np.median(cm)), (float(np.nanmedian(cmc)) if np.isfinite(cmc).any() else float('nan'))
        r = {'state': state, 'node': node, 'layer': l, 'expert': e, 'rows': n, 'rows_cal': int((win < NCAL).sum()), 'rows_prev': int((win >= NCAL).sum()),
             'windows': int(len(np.unique(win))), 's_p50': med, 's_p90': float(np.percentile(s, 90)), 's_p99': float(np.percentile(s, 99)), 's_max': float(s.max()),
             'R_token': float(s.max() / med) if med > 0 and n >= 2 else float('nan'), 'R_channel': float(cm.max() / mc) if mc > 0 else float('nan'),
             'R_channel_cal': float(np.nanmax(cmc) / mcc) if mcc and mcc > 0 else float('nan'),
             'median_cmax': mc, 'n_outlier_channels': int((cm >= OUT_MULT * mc).sum()) if mc > 0 else 0}
        for tau, nm in ((TAU, 'spike'), (TAU_STRONG, 'strong'), (TAU_EXTRA, 'extreme')):
            m = s >= tau * med if med > 0 else np.zeros(n, bool)
            r[f'n_{nm}'] = int(m.sum())
            r[f'{nm}_per_1000'] = 1000.0 * m.sum() / n
            if nm == 'spike':
                hn = rec['hn2'][ix].astype(np.float64)
                r['spike_energy_share'] = float(hn[m].sum() / hn.sum()) if hn.sum() > 0 else float('nan')
                r['spike_windows'] = int(len(np.unique(win[m])))
                r['spike_at_pos0'] = int((m & (rec['pos'][ix] == 0)).sum())
        r['rows_pos0'] = int((rec['pos'][ix] == 0).sum())
        if n >= MIN_ALL_ROWS:
            sub = np.array([rng.choice(n, N0, replace=False) for _ in range(NSUB)])
            ssub = s[sub]
            msub = np.median(ssub, 1)
            r[f'R_token_at_{N0}'] = float(np.mean(ssub.max(1) / msub)) if (msub > 0).all() else float('nan')
        else:
            r[f'R_token_at_{N0}'] = float('nan')
        o = np.argsort(win, kind='stable')
        ws, ss_ = win[o], s[o]
        cut = np.flatnonzero(np.diff(ws)) + 1
        wr = [float(x.max() / np.median(x)) for x in np.split(ss_, cut) if len(x) >= WIN_MIN and np.median(x) > 0]
        r['R_token_window_median'] = float(np.median(wr)) if wr else float('nan')
        r['R_token_window_max'] = float(max(wr)) if wr else float('nan')
        r['windows_evaluated'] = len(wr)
        rows.append(r)
    return rows


def layer_level(rec, nw):
    """MoE analogue: per layer, s_t^layer = max over the token's routed experts of s_t."""
    res = {}
    for l in range(NL):
        m = rec['layer'] == l
        key = rec['win'][m].astype(np.int64) * SEQ + rec['pos'][m]
        sl = np.zeros(nw * SEQ)
        np.maximum.at(sl, key, rec['s'][m].astype(np.float64))
        med = float(np.median(sl))
        j = int(np.argmax(sl))
        jj = np.flatnonzero(m)[np.flatnonzero(key == j)]
        best = jj[np.argmax(rec['s'][jj])]
        res[l] = {'R_token_layer': float(sl.max() / med) if med > 0 else float('nan'), 'median': med, 'max': float(sl.max()),
                  'max_win': j // SEQ, 'max_pos': j % SEQ, 'max_expert': int(rec['expert'][best])}
    return res


def select(out, log):
    """pre-registered selection on the FP32 discovery metrics; expert tables for both states; BC predictions from the static caches."""
    from scipy.stats import rankdata
    from run_native_bc_pipeline import QUANT
    if (out / 'selected_taps.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    chan = np.load(out / 'discovery_channels.npz')
    tabs = {}
    for state in ('fp32', 'q0'):
        rec = load_records(out, state)
        gi = group_index(rec)
        tabs[state] = expert_table(out, state, rec, gi, chan)
        write(out / f'layer_level_{state}.json', layer_level(rec, len(disc_tokens())))
        del rec
        log(f'select: {state} expert table ({time.perf_counter() - t0:.0f}s)')
    allrows = tabs['fp32'] + tabs['q0']
    with open(out / 'discovery_experts.csv', 'w') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(allrows[0])); wcsv.writeheader(); wcsv.writerows(allrows)
    names, *_ = pr.problem_static()
    searched = {n.rsplit('.', 1)[0] for n in names if n.endswith('down_proj')}
    aidx, gidx = read(ACT_INDEX), read(GC_INDEX)
    q0 = read(Q0K)
    prev_nodes = {t['node'] for t in read(PREV / 'selected_taps.json')['taps']}
    fp = {r['node']: r for r in tabs['fp32']}
    el = [r for r in tabs['fp32'] if r['node'] in searched and r['node'] in aidx['row'] and r['node'] not in prev_nodes and r['node'] != KNOWN
          and r['rows_cal'] >= MIN_CAL_ROWS and r['rows'] >= MIN_ALL_ROWS and math.isfinite(r['R_token']) and math.isfinite(r['R_channel'])]
    RT, RC = np.array([r['R_token'] for r in el]), np.array([r['R_channel'] for r in el])
    pT, pC = rankdata(RT) / len(RT), rankdata(RC) / len(RC)
    for r, a, b in zip(el, pT, pC):
        r['pct_R_token'], r['pct_R_channel'] = float(a), float(b)
    hiT, loT, hiC = np.quantile(RT, STRONG_Q), np.quantile(RT, WEAK_Q), np.quantile(RC, STRONG_Q)
    strong = sorted([r for r in el if r['R_token'] >= hiT], key=lambda r: r['node'])
    weak = sorted([r for r in el if r['R_token'] <= loT], key=lambda r: r['node'])
    bitclass = lambda node: 0 if od.bits(q0[f'{node}.down_proj']['a']) <= 6 else 1 if od.bits(q0[f'{node}.down_proj']['a']) <= 8 else 2
    rng = np.random.default_rng(SEEDS['select'])
    taps, used, per_layer = [], set(), Counter()

    def add(r, role, pair=None, match=None):
        taps.append({'id': len(taps), 'role': role, 'pair': pair, 'node': r['node'], 'layer': int(r['layer']), 'expert': int(r['expert']), 'proj': 'down_proj',
                     'slot': 'x', 'column': 'down_x', 'rows_all': r['rows'], 'rows_cal': r['rows_cal'], 'R_token': r['R_token'], 'R_channel': r['R_channel'],
                     'R_channel_cal': r['R_channel_cal'], f'R_token_at_{N0}': r[f'R_token_at_{N0}'], 'pct_R_token': r.get('pct_R_token'),
                     'pct_R_channel': r.get('pct_R_channel'), 'n_spike': r['n_spike'], 'n_strong': r['n_strong'], 'n_outlier_channels': r['n_outlier_channels'],
                     'q0_act_format': q0[f'{r["node"]}.down_proj']['a'], 'q0_weight_format': q0[f'{r["node"]}.down_proj']['w'], 'match': match})
        used.add(r['node'])
    got = 0
    for j in rng.permutation(len(strong)).tolist():
        if got == N_PAIRS:
            break
        sr = strong[j]
        if sr['node'] in used or per_layer[sr['layer']] >= MAX_STRONG_PER_LAYER:
            continue
        cand = [r for r in weak if r['node'] not in used]
        if not cand:
            break
        dl = min(abs(r['layer'] - sr['layer']) for r in cand)
        cand = [r for r in cand if abs(r['layer'] - sr['layer']) == dl]
        same = [r for r in cand if bitclass(r['node']) == bitclass(sr['node'])]
        pick = same or cand
        d = np.array([abs(math.log(r['rows']) - math.log(sr['rows'])) for r in pick])
        best = [pick[i] for i in np.flatnonzero(d == d.min())]
        wr = best[int(rng.integers(len(best)))]
        add(sr, 'pair_strong', got)
        add(wr, 'pair_weak', got, {'layer_diff': dl, 'same_bitclass': bool(same), 'log_rows_diff': float(d.min())})
        per_layer[sr['layer']] += 1
        got += 1
    tokdom = sorted([r for r in el if r['R_token'] >= hiT and r['node'] not in used], key=lambda r: (r['pct_R_channel'], r['node']))[:N_DIS]
    for r in tokdom:
        add(r, 'token_dominant')
    chdom = sorted([r for r in el if r['R_channel'] >= hiC and r['node'] not in used], key=lambda r: (r['pct_R_token'], r['node']))[:N_DIS]
    for r in chdom:
        add(r, 'channel_dominant')
    add(fp[KNOWN], 'known_case')
    # frozen per-tap quantities and BC predictions (static caches)
    ac, gc = np.load(ACT_CACHE, mmap_mode='r'), np.load(GC, mmap_mode='r')
    fi = [QUANT.index(f) for f in CANDS]
    a_, b_ = ss.COLOFF['down_x']
    cmi = {n: i for i, n in enumerate(chan['fp32_nodes'])}
    for tp in taps:
        node = tp['node']
        r = fp[node]
        tp['spike_threshold'] = TAU * r['s_p50']
        tp['strong_threshold'] = TAU_STRONG * r['s_p50']
        cm = chan['fp32_cmax_all'][cmi[node]]
        tp['outlier_channels'] = np.flatnonzero(cm >= OUT_MULT * np.median(cm)).tolist()
        row, grow, ns = np.asarray(ac[aidx['row'][node]]), np.asarray(gc[gidx['row'][node]]), aidx['n'][node]
        V, Bc = od.bc_slot_costs(grow[0, a_:b_], grow[1, a_:b_], row[0][fi][:, a_:b_], row[1][fi][:, a_:b_], ns)
        tp['bc'] = {f: {'T': float(V[j].sum() + Bc[j].sum()), 'variance': float(V[j].sum()), 'bias': float(Bc[j].sum())} for j, f in enumerate(CANDS)}
    pools = {'eligible': len(el), 'R_token_p90': float(hiT), 'R_token_p50': float(loT), 'R_channel_p90': float(hiC), 'strong_pool': len(strong),
             'weak_pool': len(weak), 'excluded_previous_experts': sorted(prev_nodes)}
    write(out / 'selected_taps.json', {'rule': read(out / 'plan.json')['selection'], 'pools': pools, 'taps': taps})
    log_stage(out, 'select', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'taps': [(t['id'], t['role'], t['node'], round(t['R_token'], 1), round(t['R_channel'], 1), t['q0_act_format']) for t in taps], 'pools': pools})))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    files = ('plan.json', 'selected_taps.json', 'discovery_experts.csv', 'discovery_channels.npz', 'discovery_bc_control.json', 'discovery_windows.json',
             'layer_level_fp32.json', 'layer_level_q0.json', 'work/records_fp32.npz', 'work/records_q0.npz', 'work/cal_bc_records.npz')
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'code': code_hashes(),
           'note': 'taps, spike thresholds, outlier channels, BC predictions and candidates frozen before any confirmation article is drawn'}
    write(out / 'freeze.json', res)
    log('frozen')


def data(out, log):
    """od.data with this round's size and seeds."""
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    used, _ = ss.used_wt103_all(out)
    if len(used) != pl['data_use']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    elig = [a for a in base if a['ch'] not in used]
    if len(prev) != len(used) or len(elig) != pl['data_use']['eligible_before_length_and_neardup']:
        raise SystemExit('eligible / previous counts changed')
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(a['body_norm']) for a in prev]
    tok = oc.tokenizer()
    order = np.random.default_rng(SEEDS['order']).permutation(len(elig))
    rs = np.random.default_rng(SEEDS['start'])
    segs, toks, acc_sh, skips = [], [], [], Counter()
    for k in order.tolist():
        a = elig[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        ids = tok(text, add_special_tokens=False)['input_ids']
        if len(ids) < WPA_C * SEQ:
            skips['short'] += 1; continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if max(c2, cp, ca) >= mr.NEARDUP:
            skips['near_duplicate'] += 1; continue
        st = int(rs.integers(0, len(ids) - WPA_C * SEQ + 1))
        w = np.array(ids[st:st + WPA_C * SEQ], dtype=np.int64).reshape(WPA_C, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'title': a['title'], 'content_sha256': a['ch'], 'article_tokens': len(ids), 'window_tokens': [st, st + WPA_C * SEQ],
                     'source': 'wikitext-103 train', 'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca},
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
        if len(segs) == N_ART:
            break
    if len(segs) < N_ART or {s['content_sha256'] for s in segs} & used:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'), 'n': N_ART, 'windows_per_article': WPA_C,
                                     'skips': dict(skips), 'visited': N_ART + sum(skips.values()), 'excluded_previous_wt103': len(prev),
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json')})
    log_stage(out, 'data', time.perf_counter() - t0)
    log(f'data: {N_ART} articles; skips {dict(skips)}; excluded previous {len(prev)}')


# --------------------------------------------------------------------------- #
def make_qdq_tok(dq, f, rng, acc, thr, O, keep_s):
    """QDQ of the Down input of one expert; accumulates on the confirmation rows: rows, row maxima (once per tap), error energy of spike vs
    normal rows (frozen threshold), outlier-channel energy, INT clipping per row class."""
    import torch

    def fn(t):
        q = dq.act(t, f, rng)
        e2 = (q - t).double().pow(2)
        s = t.abs().max(1).values
        sp = s >= thr
        rE = e2.sum(1)
        acc['rows'] += t.shape[0]; acc['spike_rows'] += int(sp.sum())
        acc['E'] += float(rE.sum()); acc['E_spike'] += float(rE[sp].sum())
        if len(O):
            eO = e2[:, O].sum(1)
            acc['E_O'] += float(eO.sum()); acc['E_spike_O'] += float(eO[sp].sum())
        if keep_s:
            acc['s'].append(s.cpu().numpy())
        if f.startswith('int'):
            clip = od.int_clip_mask(t, f, rng)
            cr = clip.sum(1)
            acc['clip'] += int(cr.sum()); acc['clip_spike'] += int(cr[sp].sum())
            acc['clipE'] += float((e2 * clip).sum()); acc['clipE_spike'] += float((e2 * clip).sum(1)[sp].sum())
            acc['n_el'] += clip.numel(); acc['n_el_spike'] += int(sp.sum()) * t.shape[1]
        return q
    return fn


def measure(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'measure.json').exists():
        raise SystemExit('exists')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('data changed')
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    taps = read(out / 'selected_taps.json')['taps']
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ctx = od.Ctx(None)
    ce0, H_in, ids, chunks = od.base_pass(ctx, toks, dq, log, 'fp32 base')
    ref = dr.forward(toks, None, dq, ctx.ranges, record=False, log=lambda m: None, label='dr fp32')
    controls = {'base_equals_dr_forward_max_abs': float(np.max(np.abs(np.array(ce0) - np.array(ref['ce']))))}
    res = {'base_ce': ce0, 'variants': {}, 'acc': {}, 'route_changed': {}}
    by_layer = defaultdict(list)
    for tp in taps:
        by_layer[tp['layer']].append(tp)
    stopped = False
    for l in sorted(by_layer):
        if time.perf_counter() - t0 > 60 * BUDGET_MIN['measure']:
            stopped = True
            log(f'budget exceeded before layer {l}: stop')
            break
        variants, accs = [], {}
        if l == min(by_layer):
            variants.append(('control_identity', {}))
        for tp in by_layer[l]:
            rng = ctx.ranges[f'{tp["node"]}.down_proj']['x']
            O = torch.tensor(tp['outlier_channels'], dtype=torch.long, device='cuda')
            for f in CANDS:
                name = f'tap{tp["id"]}|{f}'
                accs[name] = {'rows': 0, 'spike_rows': 0, 'E': 0.0, 'E_spike': 0.0, 'E_O': 0.0, 'E_spike_O': 0.0, 's': [], 'clip': 0, 'clip_spike': 0,
                              'clipE': 0.0, 'clipE_spike': 0.0, 'n_el': 0, 'n_el_spike': 0}
                fn = make_qdq_tok(dq, f, rng, accs[name], tp['spike_threshold'], O, f == 'int4')
                variants.append((name, {l: {(l, tp['expert'], 'down_proj', 'x'): fn}}))
        for b in range(0, len(variants), VAR_BATCH):
            part = variants[b:b + VAR_BATCH]
            ce, chg = od.variants_from(ctx, toks, l, H_in[l], ids, chunks, part, dq)
            for name, _ in part:
                if name == 'control_identity':
                    controls['identity_from_layer'] = {'layer': l, 'max_abs': float(np.max(np.abs(np.array(ce[name]) - np.array(ce0))))}
                    continue
                res['variants'][name] = ce[name]
                res['route_changed'][name] = chg[name]
                a = dict(accs[name])
                s = np.concatenate(a.pop('s')) if a['s'] else None
                if s is not None:
                    a['R_token_confirm'] = float(s.max() / np.median(s)) if np.median(s) > 0 else None
                    a['s_p50_confirm'], a['s_max_confirm'] = float(np.median(s)), float(s.max())
                else:
                    a.pop('s', None)
                res['acc'][name] = a
        log(f'measure: layer {l} done ({time.perf_counter() - t0:.0f}s, {len(variants)} variants)')
    res['controls'] = controls
    res['stopped_early'] = stopped
    res['controls']['pass'] = controls['base_equals_dr_forward_max_abs'] == 0 and controls.get('identity_from_layer', {}).get('max_abs', 1) == 0
    write(out / 'measure.json', res)
    log_stage(out, 'measure', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'stopped_early': stopped})
    log(json.dumps(dr.clean({'controls': controls, 'variants': len(res['variants']), 'stopped': stopped})))


# --------------------------------------------------------------------------- #
def article_delta(ce_v, ce_0):
    d = np.array(ce_v) - np.array(ce_0)
    return d.reshape(N_ART, WPA_C).sum(1) / (WPA_C * TP)


def ranking(out):
    taps = read(out / 'selected_taps.json')['taps']
    m = read(out / 'measure.json')
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
    fam = lambda f: 'int' if f.startswith('int') else 'posit'
    per_tap, pairs = [], []
    for tp in taps:
        if not all(f'tap{tp["id"]}|{f}' in m['variants'] for f in CANDS):
            continue
        D = {f: article_delta(m['variants'][f'tap{tp["id"]}|{f}'], m['base_ce']) for f in CANDS}
        T = {f: tp['bc'][f]['T'] for f in CANDS}
        for i, f1 in enumerate(CANDS):
            for f2 in CANDS[i + 1:]:
                diff = D[f1] - D[f2]
                bs = diff[idx].mean(1)
                ci, ci99 = np.percentile(bs, [2.5, 97.5]), np.percentile(bs, [0.5, 99.5])
                tie = abs(T[f1] - T[f2]) / max(T[f1], T[f2]) < BC_TIE if max(T[f1], T[f2]) > 0 else True
                pairs.append({'tap': tp['id'], 'role': tp['role'], 'node': tp['node'], 'f1': f1, 'f2': f2,
                              'family': 'int-int' if fam(f1) == fam(f2) == 'int' else 'posit-posit' if fam(f1) == fam(f2) == 'posit' else 'cross',
                              'same_bits': (f1, f2) in SAME_BIT, 'measured_diff': float(diff.mean()), 'ci95': ci.tolist(), 'ci99': ci99.tolist(),
                              'decisive': bool(ci[0] > 0 or ci[1] < 0), 'decisive99': bool(ci99[0] > 0 or ci99[1] < 0), 'bc_tie': bool(tie),
                              'bc_correct': bool((T[f1] < T[f2]) == (diff.mean() < 0))})
        from scipy.stats import spearmanr
        dn = {f: float(D[f].mean()) for f in CANDS}
        dec = [p for p in pairs if p['tap'] == tp['id'] and p['decisive'] and not p['bc_tie']]
        a4 = m['acc'][f'tap{tp["id"]}|int4']
        per_tap.append({'tap': tp['id'], 'role': tp['role'], 'pair': tp['pair'], 'node': tp['node'], 'layer': tp['layer'], 'R_token': tp['R_token'],
                        'R_channel': tp['R_channel'], 'R_channel_cal': tp['R_channel_cal'], f'R_token_at_{N0}': tp[f'R_token_at_{N0}'], 'rows_all': tp['rows_all'],
                        'n_spike_disc': tp['n_spike'], 'n_outlier_channels': tp['n_outlier_channels'], 'q0_act_format': tp['q0_act_format'],
                        'dnll': dn, 'dnll_ci95': {f: np.percentile(D[f][idx].mean(1), [2.5, 97.5]).tolist() for f in CANDS}, 'bc_T': T,
                        'spearman_bc_vs_dnll': float(spearmanr([T[f] for f in CANDS], [dn[f] for f in CANDS]).statistic),
                        'decisive_pairs': len(dec), 'misranked': sum(not p['bc_correct'] for p in dec),
                        'misrank_rate': (sum(not p['bc_correct'] for p in dec) / len(dec)) if dec else None,
                        'route_changed': {f: m['route_changed'][f'tap{tp["id"]}|{f}'] for f in CANDS},
                        'confirm': {'rows': a4['rows'], 'R_token': a4.get('R_token_confirm'), 'spike_rows': a4['spike_rows'],
                                    **{f: {k: m['acc'][f'tap{tp["id"]}|{f}'][k] for k in ('E', 'E_spike', 'E_O', 'E_spike_O', 'clip', 'clip_spike', 'clipE',
                                                                                           'clipE_spike', 'n_el', 'n_el_spike')} for f in CANDS}}})
    return per_tap, pairs


def acc_summary(pairs, dkey='decisive'):
    dec = [p for p in pairs if p[dkey] and not p['bc_tie']]
    return {'pairs': len(pairs), 'decisive': sum(p[dkey] for p in pairs), 'undeterminable': sum(not p[dkey] for p in pairs),
            'bc_ties': sum(p['bc_tie'] for p in pairs), 'decisive_bc_ties': sum(p[dkey] and p['bc_tie'] for p in pairs), 'evaluated': len(dec),
            'correct': sum(p['bc_correct'] for p in dec), 'accuracy': (sum(p['bc_correct'] for p in dec) / len(dec)) if dec else None,
            'abs_diff_lt_1e-6': sum(abs(p['measured_diff']) < 1e-6 for p in pairs)}


def q(xs, p):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else None


def discovery_summary(out):
    """activation, identity, error and BC-decomposition summaries from the discovery records."""
    from scipy.stats import spearmanr
    import olmoe_common as oc
    tk = oc.tokenizer()
    eos = tk.eos_token_id
    with open(out / 'discovery_experts.csv') as fh:
        er = list(csv.DictReader(fh))
    for r in er:
        for k, v in r.items():
            if k not in ('state', 'node'):
                r[k] = float(v) if v not in ('', 'nan') else float('nan')
    toks = disc_tokens()
    wmeta = read(out / 'discovery_windows.json')
    first = np.ones(toks.shape, bool)
    for w in range(len(toks)):
        seen = set()
        for p_ in range(SEQ):
            first[w, p_] = int(toks[w, p_]) not in seen
            seen.add(int(toks[w, p_]))
    text_cache = {}

    def ttext(i):
        if i not in text_cache:
            text_cache[i] = tk.decode([int(i)])
        return text_cache[i]
    res = {}
    for state in ('fp32', 'q0'):
        rs = [r for r in er if r['state'] == state]
        big = [r for r in rs if r['rows'] >= MIN_ALL_ROWS]
        e = {'experts': len(rs), 'experts_rows_ge_min': len(big), 'rows_median': q([r['rows'] for r in rs], 50)}
        for k in ('R_token', 'R_channel', 'R_channel_cal', f'R_token_at_{N0}', 's_max', 'R_token_window_median', 'R_token_window_max', 'spike_per_1000',
                  'spike_energy_share', 'n_outlier_channels'):
            v = [r[k] for r in big]
            e[k] = {'p50': q(v, 50), 'p90': q(v, 90), 'p99': q(v, 99), 'max': q(v, 100)}
        e['experts_with_spike'] = {nm: sum(r[f'n_{nm}'] > 0 for r in big) for nm in ('spike', 'strong', 'extreme')}
        e['spike_events'] = {nm: int(sum(r[f'n_{nm}'] for r in rs)) for nm in ('spike', 'strong', 'extreme')}
        e['rows_total'] = int(sum(r['rows'] for r in rs))
        e['spearman_rows'] = {k: float(spearmanr([r['rows'] for r in big], [r[k] for r in big], nan_policy='omit').statistic)
                              for k in ('R_token', f'R_token_at_{N0}', 'R_channel', 'spike_per_1000')}
        e['spearman_R_token_R_channel'] = float(spearmanr([r['R_token'] for r in big], [r['R_channel'] for r in big], nan_policy='omit').statistic)
        e['spearman_R_token_at_N0_R_channel'] = float(spearmanr([r[f'R_token_at_{N0}'] for r in big], [r['R_channel'] for r in big], nan_policy='omit').statistic)
        by_l = {}
        for l in range(NL):
            v = [r for r in big if int(r['layer']) == l]
            by_l[l] = {'experts': len(v), **{k: {'p50': q([r[k] for r in v], 50), 'p90': q([r[k] for r in v], 90), 'max': q([r[k] for r in v], 100)}
                                             for k in ('R_token', 'R_channel', f'R_token_at_{N0}', 's_max', 'spike_per_1000')},
                       'experts_with_strong_spike': sum(r['n_strong'] > 0 for r in v), 'experts_with_extreme_spike': sum(r['n_extreme'] > 0 for r in v)}
        e['by_layer'] = by_l
        grp = lambda ls: [r for r in big if int(r['layer']) in ls]
        e['front_middle_back'] = {nm: {'experts': len(grp(ls)), 'R_token_p50': q([r['R_token'] for r in grp(ls)], 50),
                                       'R_token_p90': q([r['R_token'] for r in grp(ls)], 90), f'R_token_at_{N0}_p50': q([r[f'R_token_at_{N0}'] for r in grp(ls)], 50),
                                       'R_channel_p50': q([r['R_channel'] for r in grp(ls)], 50), 'strong_spike_experts': sum(r['n_strong'] > 0 for r in grp(ls))}
                                  for nm, ls in (('front', FRONT), ('middle', tuple(range(FRONT[-1] + 1, BACK[0]))), ('back', BACK))}
        e['layer_level'] = read(out / f'layer_level_{state}.json')
        # identity of spike rows
        rec = load_records(out, state)
        gi = group_index(rec)
        med = np.zeros(len(rec['s']))
        lg_med, lu_med = np.zeros(len(rec['s'])), np.zeros(len(rec['s']))
        for (l, ex), ix in gi.items():
            med[ix] = np.median(rec['s'][ix])
            lg_med[ix] = np.median(np.log(np.abs(rec['gsilu'][ix]) + 1e-30))
            lu_med[ix] = np.median(np.log(np.abs(rec['up'][ix]) + 1e-30))
        ratio = np.where(med > 0, rec['s'] / np.maximum(med, 1e-30), 0)
        tid = toks[rec['win'].astype(np.int64), rec['pos'].astype(np.int64)]
        cat_of = {}
        cats = np.array([cat_of.setdefault(int(t), token_category(ttext(int(t)), int(t), eos)) for t in tid])
        fo = first[rec['win'].astype(np.int64), rec['pos'].astype(np.int64)]
        src = np.where(rec['win'] < NCAL, 'cal', 'prev')
        ident = {}
        for nm, tau in (('spike', TAU), ('strong', TAU_STRONG), ('extreme', TAU_EXTRA)):
            m = ratio >= tau
            n_all, n_sp = len(m), int(m.sum())
            d = {'events': n_sp}
            if n_sp:
                d['pos0'] = {'share_spikes': float((rec['pos'][m] == 0).mean()), 'share_all_rows': float((rec['pos'] == 0).mean())}
                d['pos_buckets'] = {b: {'share_spikes': float(((rec['pos'][m] >= lo) & (rec['pos'][m] < hi)).mean()),
                                        'share_all_rows': float(((rec['pos'] >= lo) & (rec['pos'] < hi)).mean())}
                                    for b, (lo, hi) in {'0': (0, 1), '1-3': (1, 4), '4-15': (4, 16), '16-63': (16, 64), '64-127': (64, 128)}.items()}
                d['category'] = {c: {'share_spikes': float((cats[m] == c).mean()), 'share_all_rows': float((cats == c).mean()),
                                     'lift': float((cats[m] == c).mean() / max((cats == c).mean(), 1e-12))} for c in sorted(set(cats))}
                d['first_occurrence'] = {'share_spikes': float(fo[m].mean()), 'share_all_rows': float(fo.mean())}
                d['source'] = {s_: {'spikes': int((src[m] == s_).sum()), 'rows': int((src == s_).sum())} for s_ in ('cal', 'prev')}
                ct_s, ct_a = Counter(tid[m].tolist()), Counter(tid.tolist())
                d['top_token_ids'] = [{'id': int(i), 'text': ttext(int(i)), 'category': cat_of[int(i)], 'spikes': c, 'rows': ct_a[i],
                                       'spike_rate': c / ct_a[i]} for i, c in ct_s.most_common(20)]
                d['distinct_ids'] = len(ct_s)
                d['experts'] = int(len(set(zip(rec['layer'][m].tolist(), rec['expert'][m].tolist()))))
                d['by_layer'] = {int(l): int(c) for l, c in Counter(rec['layer'][m].tolist()).items()}
                d['top2_over_s_p50'] = float(np.median(rec['top2'][m] / np.maximum(rec['s'][m], 1e-30)))
                dg = np.log(np.abs(rec['gsilu'][m]) + 1e-30) - lg_med[m]
                du = np.log(np.abs(rec['up'][m]) + 1e-30) - lu_med[m]
                d['gate_up'] = {'dlog_gate_p50': float(np.median(dg)), 'dlog_up_p50': float(np.median(du)), 'share_gate_larger': float((dg > du).mean()),
                                'share_both_positive': float(((dg > 0) & (du > 0)).mean()), 'share_gate_only': float(((dg > 0) & (du <= 0)).mean()),
                                'share_up_only': float(((du > 0) & (dg <= 0)).mean())}
            ident[nm] = d
        e['identity'] = ident
        if state == 'fp32':
            m = ratio >= TAU_STRONG
            top = np.flatnonzero(m)[np.argsort(-ratio[m])][:40]
            lst = []
            for i in top.tolist():
                wi, p_ = int(rec['win'][i]), int(rec['pos'][i])
                wm = wmeta['windows'][wi]
                lst.append({'layer': int(rec['layer'][i]), 'expert': int(rec['expert'][i]), 'ratio': float(ratio[i]), 's': float(rec['s'][i]),
                            'cstar': int(rec['cstar'][i]), 'source': wm['source'], 'title': wmeta['cal_token_titles'][wi * SEQ + p_] if wi < NCAL else wm['title'],
                            'offset': wm['start_offset'] + p_, 'offset_kind': wm['offset_kind'], 'win': wi, 'pos': p_, 'token_id': int(tid[i]),
                            'text': ttext(int(tid[i])), 'category': cats[i], 'first_in_window': bool(fo[i]), 'kslot': int(rec['kslot'][i]), 'rw': float(rec['rw'][i]),
                            'silu_gate': float(rec['gsilu'][i]), 'up': float(rec['up'][i]), 'dlog_gate': float(np.log(abs(rec['gsilu'][i]) + 1e-30) - lg_med[i]),
                            'dlog_up': float(np.log(abs(rec['up'][i]) + 1e-30) - lu_med[i]),
                            'context': tk.decode(toks[wi, max(0, p_ - 6):p_].tolist()) + ' [[' + ttext(int(tid[i])) + ']] ' + tk.decode(toks[wi, p_ + 1:p_ + 4].tolist())})
            e['top_strong_spikes'] = lst
            with open(out / 'spike_tokens_fp32.csv', 'w') as fh:
                mm = np.flatnonzero(ratio >= TAU)
                wcsv = csv.writer(fh)
                wcsv.writerow(['layer', 'expert', 'source', 'win', 'pos', 'offset', 'token_id', 'text', 'category', 'first_in_window', 'kslot', 'rw', 's', 'ratio',
                               'cstar', 'silu_gate', 'up'])
                for i in mm.tolist():
                    wi, p_ = int(rec['win'][i]), int(rec['pos'][i])
                    wcsv.writerow([int(rec['layer'][i]), int(rec['expert'][i]), src[i], wi, p_, wmeta['windows'][wi]['start_offset'] + p_, int(tid[i]),
                                   ttext(int(tid[i])), cats[i], bool(fo[i]), int(rec['kslot'][i]), float(rec['rw'][i]), float(rec['s'][i]), float(ratio[i]),
                                   int(rec['cstar'][i]), float(rec['gsilu'][i]), float(rec['up'][i])])
            # errors: spike vs normal rows, per expert then summarized
            sp = ratio >= TAU
            errs = {}
            for j, f in enumerate(CANDS):
                shares, mse_ratio, cells = [], [], np.zeros(4)
                for (l, ex), ix in gi.items():
                    e2 = rec['e2'][ix, j].astype(np.float64)
                    if not np.isfinite(e2).all() or e2.sum() <= 0 or len(ix) < MIN_ALL_ROWS:
                        continue
                    msk = sp[ix]
                    if msk.any() and (~msk).any():
                        shares.append(e2[msk].sum() / e2.sum())
                        mse_ratio.append((e2[msk].mean()) / max(e2[~msk].mean(), 1e-300))
                    eo = rec['e2O'][ix, j].astype(np.float64)
                    cells += [eo[msk].sum(), (e2 - eo)[msk].sum(), eo[~msk].sum(), (e2 - eo)[~msk].sum()]
                d = {'experts_with_spikes': len(shares), 'spike_err_share_p50_p90': [q(shares, 50), q(shares, 90)],
                     'mse_ratio_spike_over_normal_p50_p90': [q(mse_ratio, 50), q(mse_ratio, 90)],
                     'pooled_2x2_share': dict(zip(('spike_outlierch', 'spike_otherch', 'normal_outlierch', 'normal_otherch'), (cells / cells.sum()).tolist())),
                     'spike_row_share': float(sp.mean())}
                if f.startswith('int'):
                    ji = [x for x in CANDS if x.startswith('int')].index(f)
                    cn, cE = rec['clipn'][:, ji].astype(np.float64), rec['clipE'][:, ji].astype(np.float64)
                    ok = np.isfinite(cE)
                    d['clipping'] = {}
                    for s_ in ('cal', 'prev'):
                        for nm_, mm_ in (('spike', sp), ('normal', ~sp)):
                            sel = ok & (src == s_) & mm_
                            d['clipping'][f'{s_}|{nm_}'] = {'rows': int(sel.sum()), 'rows_with_clip': int((cn[sel] > 0).sum()),
                                                             'clip_frac_elements': float(cn[sel].sum() / max(sel.sum() * 1024, 1)),
                                                             'clipped_energy_share': float(cE[sel].sum() / max(rec['e2'][sel, j].astype(np.float64).sum(), 1e-300))}
                errs[f] = d
            e['errors'] = errs
        del rec
        res[state] = e
    # BC token decomposition on the calibration rows
    cr = np.load(out / 'work' / 'cal_bc_records.npz')
    recf = load_records(out, 'fp32')
    gi = group_index(recf)
    med = {k: float(np.median(recf['s'][ix])) for k, ix in gi.items()}
    mc = recf['win'] < NCAL
    key = {(int(l), int(ex), int(w), int(p_)): float(s) for l, ex, w, p_, s in zip(recf['layer'][mc], recf['expert'][mc], recf['win'][mc], recf['pos'][mc], recf['s'][mc])}
    ctl = read(out / 'discovery_bc_control.json')
    sp_c = np.array([key[(int(l), int(ex), int(w), int(p_))] >= TAU * med[(int(l), int(ex))] for l, ex, w, p_ in zip(cr['layer'], cr['expert'], cr['win'], cr['pos'])])
    ek = cr['layer'].astype(np.int64) * NE + cr['expert']
    bcd = {}
    for j, f in enumerate(CANDS):
        vs, bsh, bias_share = [], [], []
        for k_ in np.unique(ek):
            mm = ek == k_
            node = f'L{k_ // NE}.E{k_ % NE}'
            if node not in ctl or ctl[node]['n_own'] < MIN_CAL_ROWS:
                continue
            v, b = cr['v'][mm, j], cr['b'][mm, j]
            Tt = v.sum() + b.sum()
            if Tt > 0:
                bias_share.append(b.sum() / Tt)
            if sp_c[mm].any() and v.sum() > 0:
                vs.append(v[sp_c[mm]].sum() / v.sum())
                bsh.append(b[sp_c[mm]].sum() / b.sum() if b.sum() != 0 else float('nan'))
        bcd[f] = {'experts_with_cal_spikes': len(vs), 'spike_variance_share_p50_p90': [q(vs, 50), q(vs, 90)],
                  'spike_bias_allocation_share_p50_p90': [q(bsh, 50), q(bsh, 90)], 'bias_share_of_T_p50_p90': [q(bias_share, 50), q(bias_share, 90)]}
    bcd['cal_spike_row_share'] = float(sp_c.mean())
    vals = [c for c in ctl.values()]
    bcd['control'] = {'experts': len(vals), 'n_equal': sum(c['n_own'] == c['n_cache'] for c in vals),
                      'S1_maxdiff_rel_max': max(max(c['S1_maxdiff_rel']) for c in vals if c['S1_maxdiff_rel']),
                      'S2_maxdiff_rel_max': max(max(c['S2_maxdiff_rel']) for c in vals if c['S2_maxdiff_rel']),
                      'T_own_vs_cache_rel_max': max(abs(a - b) / max(abs(b), 1e-300) for c in vals for a, b in zip(c['T_own'], c['T_cache'])),
                      'v_plus_b_vs_T_own_rel_max': max(abs(v + b - t) / max(abs(t), 1e-300) for c in vals for v, b, t in zip(c['v_sum'], c['b_sum'], c['T_own'])),
                      'per_format': {f: {'S1_maxdiff_rel_p50_max': [q([c['S1_maxdiff_rel'][j] for c in vals], 50), q([c['S1_maxdiff_rel'][j] for c in vals], 100)],
                                         'S2_maxdiff_rel_p50_max': [q([c['S2_maxdiff_rel'][j] for c in vals], 50), q([c['S2_maxdiff_rel'][j] for c in vals], 100)],
                                         'T_rel_p50_max': [q([abs(c['T_own'][j] - c['T_cache'][j]) / max(abs(c['T_cache'][j]), 1e-300) for c in vals], 50),
                                                           q([abs(c['T_own'][j] - c['T_cache'][j]) / max(abs(c['T_cache'][j]), 1e-300) for c in vals], 100)]}
                                     for j, f in enumerate(CANDS)}}
    res['bc_token_decomposition'] = bcd
    # FP32 vs q0
    fpm = {r['node']: r for r in er if r['state'] == 'fp32' and r['rows'] >= MIN_ALL_ROWS}
    q0m = {r['node']: r for r in er if r['state'] == 'q0' and r['rows'] >= MIN_ALL_ROWS}
    com = [n for n in fpm if n in q0m]
    res['fp32_vs_q0'] = {'R_token_spearman': float(spearmanr([fpm[n]['R_token'] for n in com], [q0m[n]['R_token'] for n in com], nan_policy='omit').statistic),
                         'R_channel_spearman': float(spearmanr([fpm[n]['R_channel'] for n in com], [q0m[n]['R_channel'] for n in com], nan_policy='omit').statistic)}
    return res


def analyze(out, log):
    from scipy.stats import spearmanr
    t0 = time.perf_counter()
    taps = read(out / 'selected_taps.json')['taps']
    res = {'discovery': discovery_summary(out)}
    log(f'analyze: discovery summary ({time.perf_counter() - t0:.0f}s)')
    per_tap, pairs = ranking(out)
    res['per_tap'] = per_tap
    new = lambda p: p['role'] != 'known_case'
    R = {'all_new': acc_summary([p for p in pairs if new(p)]), 'all_new_99': acc_summary([p for p in pairs if new(p)], 'decisive99'),
         'known_case': acc_summary([p for p in pairs if not new(p)])}
    for role in ('pair_strong', 'pair_weak', 'token_dominant', 'channel_dominant'):
        R[f'role:{role}'] = acc_summary([p for p in pairs if p['role'] == role])
    for fam in ('int-int', 'posit-posit', 'cross'):
        R[f'family:{fam}'] = acc_summary([p for p in pairs if new(p) and p['family'] == fam])
        R[f'family:{fam}|known'] = acc_summary([p for p in pairs if not new(p) and p['family'] == fam])
    R['same_bits'] = acc_summary([p for p in pairs if new(p) and p['same_bits']])
    for role in ('pair_strong', 'pair_weak', 'token_dominant', 'channel_dominant', 'known_case'):
        R[f'same_bits|{role}'] = acc_summary([p for p in pairs if p['role'] == role and p['same_bits']])
    for a, b in SAME_BIT:
        R[f'same_bits:{a}-{b}'] = acc_summary([p for p in pairs if new(p) and p['f1'] == a and p['f2'] == b])
    res['ranking'] = R
    # P4: matched pairs
    m = read(out / 'measure.json')
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
    pt = {t['tap']: t for t in per_tap}
    P4 = {'pairs': []}
    for f in ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1'):
        ds = []
        for k_ in range(N_PAIRS):
            st = next((t for t in taps if t['role'] == 'pair_strong' and t['pair'] == k_), None)
            wk = next((t for t in taps if t['role'] == 'pair_weak' and t['pair'] == k_), None)
            if st is None or wk is None or st['id'] not in pt or wk['id'] not in pt:
                continue
            d = article_delta(m['variants'][f'tap{st["id"]}|{f}'], m['base_ce']) - article_delta(m['variants'][f'tap{wk["id"]}|{f}'], m['base_ce'])
            ds.append(d)
            P4['pairs'].append({'pair': k_, 'format': f, 'strong': st['node'], 'weak': wk['node'], 'diff': float(d.mean()),
                                'ci95': np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist()})
        if ds:
            D = np.mean(ds, 0)
            bs = D[idx].mean(1)
            P4[f] = {'n_pairs': len(ds), 'mean_diff': float(D.mean()), 'ci97.5': np.percentile(bs, [1.25, 98.75]).tolist(),
                     'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'primary': f in ('int4', 'p4e1')}
    S = [t['tap'] for t in per_tap if t['role'] == 'pair_strong' and t['misrank_rate'] is not None]
    Wk = [t['tap'] for t in per_tap if t['role'] == 'pair_weak' and t['misrank_rate'] is not None]
    if S and Wk:
        rng = np.random.default_rng(SEEDS['tap_bootstrap'])
        dd = [np.mean([pt[i]['misrank_rate'] for i in rng.choice(S, len(S))]) - np.mean([pt[i]['misrank_rate'] for i in rng.choice(Wk, len(Wk))]) for _ in range(BOOT)]
        P4['misrank'] = {'strong_taps_evaluated': len(S), 'weak_taps_evaluated': len(Wk),
                         'diff': float(np.mean([pt[i]['misrank_rate'] for i in S]) - np.mean([pt[i]['misrank_rate'] for i in Wk])),
                         'ci95': np.percentile(dd, [2.5, 97.5]).tolist()}
    else:
        P4['misrank'] = {'strong_taps_evaluated': len(S), 'weak_taps_evaluated': len(Wk), 'diff': None}
    res['P4'] = P4
    # P5
    nt = [t for t in per_tap if t['role'] != 'known_case']
    P5 = {}
    for x in ('R_token', 'R_channel', f'R_token_at_{N0}'):
        for y in ('int4', 'p4e1', 'misrank_rate'):
            pts = [(t[x], t['misrank_rate'] if y == 'misrank_rate' else t['dnll'][y]) for t in nt
                   if (t['misrank_rate'] is not None or y != 'misrank_rate') and t[x] is not None and math.isfinite(t[x])]
            P5[f'{x}|{y}'] = {'n': len(pts), 'spearman': float(spearmanr(*zip(*pts)).statistic) if len(pts) > 2 else None}
    res['P5'] = P5
    # known case replication
    prevp = list(csv.DictReader(open(PREV / 'ranking_pairs.csv')))
    old = [(p['f1'], p['f2']) for p in prevp if p['tap'] == '18' and p['decisive'] == 'True' and p['bc_tie'] == 'False' and p['bc_correct'] == 'False']
    kp = [p for p in pairs if p['role'] == 'known_case']
    res['known_case'] = {'previous_misranked_pairs': old, 'now': [next(({'f1': a, 'f2': b, 'measured_diff': p['measured_diff'], 'ci95': p['ci95'],
                                                                        'decisive': p['decisive'], 'bc_correct': p['bc_correct']} for p in kp if p['f1'] == a and p['f2'] == b), None)
                                                                  for a, b in old]}
    res['known_case']['replicated'] = sum(1 for x in res['known_case']['now'] if x and x['decisive'] and not x['bc_correct'])
    res['near_zero'] = {'taps_all_dnll_ci_include_0': sum(all(c[0] <= 0 <= c[1] for c in t['dnll_ci95'].values()) for t in per_tap)}
    res['confirm_vs_discovery_R_token_spearman'] = float(spearmanr([t['R_token'] for t in per_tap], [t['confirm']['R_token'] for t in per_tap]).statistic)
    with open(out / 'ranking_pairs.csv', 'w') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(pairs[0])); wcsv.writeheader(); wcsv.writerows(pairs)
    with open(out / 'per_tap.csv', 'w') as fh:
        flat = [{'tap': t['tap'], 'role': t['role'], 'pair': t['pair'], 'node': t['node'], 'R_token': t['R_token'], 'R_channel': t['R_channel'],
                 'R_channel_cal': t['R_channel_cal'], f'R_token_at_{N0}': t[f'R_token_at_{N0}'], 'R_token_confirm': t['confirm']['R_token'],
                 'rows_all': t['rows_all'], 'q0_act_format': t['q0_act_format'], 'spearman': t['spearman_bc_vs_dnll'], 'decisive_pairs': t['decisive_pairs'],
                 'misranked': t['misranked'], **{f'dnll_{f}': t['dnll'][f] for f in CANDS}, **{f'bcT_{f}': t['bc_T'][f] for f in CANDS}} for t in per_tap]
        wcsv = csv.DictWriter(fh, fieldnames=list(flat[0])); wcsv.writeheader(); wcsv.writerows(flat)
    with open(out / 'per_window.csv', 'w') as fh:
        keys = ['base'] + sorted(m['variants'])
        wcsv = csv.writer(fh); wcsv.writerow(['window', 'article'] + keys)
        for i in range(len(m['base_ce'])):
            wcsv.writerow([i, i // WPA_C, m['base_ce'][i]] + [m['variants'][k][i] for k in keys[1:]])
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['measure_controls'] = m['controls']
    res['stopped_early'] = m['stopped_early']
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'ranking_new': R['all_new'], 'known': R['known_case'], 'P4_int4': P4.get('int4'), 'P4_p4e1': P4.get('p4e1'),
                             'known_replicated': res['known_case']['replicated']})))


# --------------------------------------------------------------------------- #
PAL = {'surface': '#fcfcfb', 'text': '#0b0b0b', 'text2': '#52514e', 'grid': '#e4e3df', 's1': '#2a78d6', 's2': '#eb6834', 's3': '#1baf7a', 'muted': '#b9b8b2'}


def figures(out, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    fd = out / 'figures'
    fd.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'axes.edgecolor': PAL['muted'], 'axes.labelcolor': PAL['text'],
                         'xtick.color': PAL['text2'], 'ytick.color': PAL['text2'], 'text.color': PAL['text'], 'axes.grid': True, 'grid.color': PAL['grid'],
                         'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'lines.linewidth': 2})
    with open(out / 'discovery_experts.csv') as fh:
        er = [r for r in csv.DictReader(fh)]
    for r in er:
        for k, v in r.items():
            if k not in ('state', 'node'):
                r[k] = float(v) if v not in ('', 'nan') else float('nan')
    fp = [r for r in er if r['state'] == 'fp32' and r['rows'] >= MIN_ALL_ROWS]
    rng = np.random.default_rng(0)
    # 1 per-layer distributions
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, k, ttl in ((axs[0], 'R_token', 'R_token (token max / median)'), (axs[1], 'R_channel', 'R_channel (channel max / median)')):
        for l in range(NL):
            v = np.array([r[k] for r in fp if int(r['layer']) == l and math.isfinite(r[k])])
            ax.scatter(l + rng.uniform(-0.25, 0.25, len(v)), v, s=8, color=PAL['s1'], alpha=0.35, linewidths=0)
            ax.plot([l - 0.3, l + 0.3], [np.median(v)] * 2, color=PAL['text'], lw=2)
        ax.set_yscale('log'); ax.set_xticks(range(NL)); ax.set_xlabel('layer'); ax.set_title(ttl, loc='left', fontsize=10)
    axs[0].set_ylabel('ratio (log scale)')
    fig.text(0.01, 0.01, f'FP32, discovery 202 windows; experts with >= {MIN_ALL_ROWS} rows; dots = experts, bar = layer median', color=PAL['text2'], fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(fd / 'layer_distributions.png', dpi=150); plt.close(fig)
    # 2 layer-level MoE analogue
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for st, col, lab in (('fp32', PAL['s1'], 'FP32'), ('q0', PAL['s2'], 'q0')):
        ll = read(out / f'layer_level_{st}.json')
        ax.plot(range(NL), [ll[str(l)]['R_token_layer'] for l in range(NL)], color=col, marker='o', ms=4, label=lab)
    ax.set_yscale('log'); ax.set_xticks(range(NL)); ax.set_xlabel('layer'); ax.set_ylabel('R_token^layer (log)')
    ax.set_title('Layer-level token scale (max over a token\'s 8 experts)', loc='left', fontsize=10); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fd / 'layer_level.png', dpi=150); plt.close(fig)
    # 3 R_token vs R_channel with selected taps
    taps = read(out / 'selected_taps.json')['taps']
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter([r['R_channel'] for r in fp], [r['R_token'] for r in fp], s=6, color=PAL['muted'], alpha=0.6, linewidths=0, label='all experts')
    groups = (('pair_strong', 'pair_weak'), ('token_dominant',), ('channel_dominant',), ('known_case',))
    for (g, col, mk, lab) in zip(groups, (PAL['s1'], PAL['s2'], PAL['s2'], PAL['s3']), ('o', '^', 's', 'o'),
                                 ('matched pairs', 'token-dominant', 'channel-dominant', 'L1.E9 (known)')):
        tt = [t for t in taps if t['role'] in g]
        ax.scatter([t['R_channel'] for t in tt], [t['R_token'] for t in tt], s=44, marker=mk, color=col, edgecolors=PAL['surface'], linewidths=1.5,
                   label=lab, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('R_channel'); ax.set_ylabel('R_token')
    ax.set_title('Token vs channel ratio per expert (FP32)', loc='left', fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / 'R_token_vs_R_channel.png', dpi=150); plt.close(fig)
    # 4 coverage dependence
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k in ((axs[0], 'R_token'), (axs[1], f'R_token_at_{N0}')):
        ax.scatter([r['rows'] for r in fp], [r[k] for r in fp], s=6, color=PAL['s1'], alpha=0.4, linewidths=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('rows of the expert (discovery)'); ax.set_ylabel(k)
        ax.set_title(f'{k} vs coverage', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(fd / 'coverage.png', dpi=150); plt.close(fig)
    # 5 confirmation dNLL per tap
    if (out / 'analysis.json').exists():
        an = read(out / 'analysis.json')
        pts = an['per_tap']
        n = len(pts)
        cols = 5
        rows_ = int(math.ceil(n / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.2 * rows_), squeeze=False)
        for i, t in enumerate(pts):
            ax = axs[i // cols][i % cols]
            for j, f in enumerate(CANDS):
                col = PAL['s1'] if f.startswith('int') else PAL['s2']
                lo, hi = t['dnll_ci95'][f]
                ax.plot([j, j], [lo, hi], color=col, lw=2)
                ax.scatter([j], [t['dnll'][f]], s=22, color=col, zorder=3, edgecolors=PAL['surface'], linewidths=1)
            ax.axhline(0, color=PAL['text2'], lw=0.8)
            ax.set_xticks(range(len(CANDS))); ax.set_xticklabels(CANDS, rotation=60, fontsize=7)
            ax.set_title(f'{t["node"]} ({t["role"].replace("_", " ")})', fontsize=8, loc='left')
            ax.tick_params(axis='y', labelsize=7)
        for i in range(n, rows_ * cols):
            axs[i // cols][i % cols].axis('off')
        fig.text(0.01, 0.005, 'dNLL vs FP32 (nat/token), 95% article-bootstrap CI; blue = INT, orange = Posit', color=PAL['text2'], fontsize=8)
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(fd / 'confirm_dnll.png', dpi=150); plt.close(fig)
    # 6 spike rate by window position
    rec = load_records(out, 'fp32')
    gi = group_index(rec)
    med = np.zeros(len(rec['s']))
    for k_, ix in gi.items():
        med[ix] = np.median(rec['s'][ix])
    sp = rec['s'] >= TAU * np.maximum(med, 1e-30)
    st_ = rec['s'] >= TAU_STRONG * np.maximum(med, 1e-30)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    pos = rec['pos'].astype(int)
    for msk, col, lab in ((sp, PAL['s1'], f'spike (>= {TAU:g}x)'), (st_, PAL['s2'], f'strong (>= {TAU_STRONG:g}x)')):
        rate = np.bincount(pos, weights=msk.astype(float), minlength=SEQ) / np.bincount(pos, minlength=SEQ)
        ax.plot(range(SEQ), 1000 * rate, color=col, label=lab, lw=1.5)
    ax.set_yscale('symlog', linthresh=0.1); ax.set_xlabel('position in window'); ax.set_ylabel('spikes per 1000 routed rows')
    ax.set_title('Spike rate by window position (FP32, all layers)', loc='left', fontsize=10); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fd / 'spike_rate_by_position.png', dpi=150); plt.close(fig)
    log_stage(out, 'figures', time.perf_counter() - t0)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'discovery', 'select', 'freeze', 'data', 'measure', 'analyze', 'figures'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
