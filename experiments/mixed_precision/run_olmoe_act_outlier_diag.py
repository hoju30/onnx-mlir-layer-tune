#!/usr/bin/env python3
"""OLMoE activation outliers and BC ranking errors: a diagnostic. Pre-registered in plan.json / plan.md.

Separates (1) whether activations have a few extreme channels / tokens, (2) whether these dominate the quantization
error and the measured NLL damage, (3) whether BC mis-ranks candidate formats there. No criterion change, no fine-tuning,
no new knee, no weight-only pipeline, no new quantization method; diagnostic oracles only.

Quantization facts used (checked in the code): INT activation QDQ = one scalar scale / zero-point per physical QDQ position
(expert x projection x slot), affine from the fixed FP32 calibration [min, max] widened to contain 0 (per-tensor scale; the
corrected device arithmetic); Posit activation QDQ = direct LUT rounding of the raw values (no scale, no range). BC uses
per-channel statistics (G2, GS, error sums per channel); per-channel statistics are not per-channel scales.
Physical QDQ positions per expert: gate.x and up.x (two copies of the same input xe, same range), gate.y, up.y, down.x
(SiLU(g) u), down.y. BC charges gate.x and up.x with the total gradient of xe (both paths).

Stages (one --out-dir): plan, discovery, select, pathgrad, freeze, data, measure, q0local, trigger, intervene, analyze.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_dynamic_route as dr
import run_olmoe_knee_error_only as ke
import run_olmoe_knee_selective_sensitivity as ks
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss

HERE = Path(__file__).resolve().parent
KS, KA, KE = ke.KS, ke.KA, HERE / 'runs/olmoe_knee_error_only_20261022'
KW = HERE / 'runs/olmoe_knee_weight_sensitivity_20261023'
RANGES, CALIB, MP = ss.RANGES, ss.CALIB, ss.MP
STATIC_TABLE, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.ACT_CACHE, ks.ACT_INDEX
GC, GC_INDEX = KE / 'work/runs/static_gc/grad_stats_cache.npy', KE / 'work/runs/static_gc/grad_stats_cache_index.json'
Q0_ERR, Q0_ERR_INDEX = KE / 'work/runs/refresh_E_r1/new_error_stats.npy', KE / 'work/runs/refresh_E_r1/new_error_stats_index.json'
C_STATS = KA / 'work/runs/refresh_C_r1'
Q0K = KS / 'knee/q0_knee.json'
REFRESH_B_TABLE = KS / 'tables/B.json'
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
CANDS = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')
TAP_TYPES = {'gate.x': ('gate_proj', 'x', 'xe', 'x_in'), 'up.x': ('up_proj', 'x', 'xe', 'x_in'), 'gate.y': ('gate_proj', 'y', 'zg', 'gate_y'),
             'up.y': ('up_proj', 'y', 'zu', 'up_y'), 'down.x': ('down_proj', 'x', 'h', 'down_x'), 'down.y': ('down_proj', 'y', 'zd', 'down_y')}
OUT_MULT, MIN_ROWS, STRONG_Q, WEAK_Q, N_PER_STRATUM = 6.0, 64, 0.90, 0.50, 2
N_ART, WPA_C, CHUNK = 64, 3, 128
BOOT, BC_TIE = 10000, 0.02
SEEDS = {'order': 23101, 'start': 23102, 'bootstrap': 23111, 'tap_bootstrap': 23112, 'select': 23121, 'random_bypass': 23131}
BUDGET_MIN = {'discovery': 5, 'select': 2, 'pathgrad': 3, 'data': 4, 'measure': 25, 'q0local': 5, 'intervene': 12, 'analyze': 3, 'total_gpu': 60}
TRIGGER = {'T1': 'for int4 / int6, the outlier channels carry >= 50% of the tap error energy (FP32 discovery) in at least half of the strong taps',
           'T2': 'median over the matched (strong, weak) pairs of measured int4 dNLL(strong) / dNLL(weak) >= 2 (pairs with both dNLL > 0)',
           'T3': 'decisive-pair misranking rate strong - weak >= 0.10 with a 95% tap-bootstrap CI excluding 0'}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_act_outlier_diag.py', 'run_olmoe_knee_error_only.py', 'run_olmoe_knee_selective_sensitivity.py']
                                                 + ss.CODE + dr.SCRIPTS))}


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def node_of(n):
    return '.'.join(n.split('.')[:2])


def bits(f):
    from htq_pareto_search import FORMAT_BITS
    return FORMAT_BITS[f]


def int_clip_mask(t, f, rng):
    """the elements the corrected int QDQ clamps (same scale / zero-point tensors as the quantizer)."""
    import torch
    import eval_random_mixed_precision as erm
    b = int(f[3:])
    qmin, qmax = -(1 << (b - 1)), (1 << (b - 1)) - 1
    scale, zp = erm.int_qparams(float(rng[0]), float(rng[1]), b)
    s = torch.tensor(np.float32(scale), dtype=t.dtype, device=t.device)
    z = torch.tensor(np.float32(zp), dtype=t.dtype, device=t.device)
    u = torch.round(t / s + z)
    return (u < qmin) | (u > qmax)


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    names, nodes, elems, flops, rows = pr.problem_static()
    used, manifests = ss.used_wt103_all(out)
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    w103 = mr.split_articles(mr.load_lines(mr.WT103))
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    files = [Q0K, KS / 'knee/knee.json', STATIC_TABLE, REFRESH_B_TABLE, ACT_CACHE, ACT_INDEX, GC, GC_INDEX, Q0_ERR, Q0_ERR_INDEX,
             C_STATS / 'target_stats.npz', C_STATS / 'target_stats_meta.json', RANGES, CALIB] + mr.WT103 + mr.WT2
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'questions': ['1 where do OLMoE activation outliers occur, how many channels / tokens', '2 do they dominate the quantization error',
                      '3 do they dominate the measured NLL damage', '4 does BC mis-rank candidates more often there', '5 do controls support a causal reading',
                      '6 range / granularity problem, BC statistics approximation, or undetermined'],
        'fixed': {'model': 'OLMoE-1B-7B, original checkpoint and router', 'calibration': 'stage-1 10 x 128 (discovery)', 'quantizer': 'dev_quant(corrected=True)',
                  'q0': f'{Q0K.relative_to(HERE)} (normal knee)', 'not': 'no criterion change, no fine-tuning, no knee re-selection, no weight-only pipeline, '
                  'no new quantization method (SmoothQuant etc.), nothing added to the default pipeline'},
        'quantization_granularity': {'int_activation': 'one scalar scale and zero-point per physical QDQ position (expert x projection x slot), affine from '
                                                       'the fixed FP32-calibration [min, max] widened to contain 0; per-tensor, not per-channel',
                                     'posit_activation': 'direct posit LUT rounding of raw values; no scale, no range',
                                     'bc_statistics': 'per-channel G2 / GS and per-channel error sums; per-channel statistics, per-tensor scales',
                                     'separate_analysis': 'INT and Posit analysed separately; INT range / clipping mechanisms are not applied to Posit'},
        'physical_positions': {'per_expert': list(TAP_TYPES), 'shared_input': 'gate.x and up.x quantize their own copies of the same input xe with the '
                               'same range; activation statistics of xe are computed once',
                               'bc_attribution': 'BC charges both gate.x and up.x with the total gradient of xe (sum of both paths); the physical gate.x '
                                                 'QDQ only feeds the gate path; a path-specific gradient is computed for the selected x taps as a '
                                                 'diagnostic (not a criterion change)'},
        'outlier_metrics': {'per_channel': 'max|a|, min, max, RMS, mean, p99 of |a| (kthvalue; rows >= 100 else missing), whether max|a| occurs at a '
                                           'position-0 token',
                            'cross_channel': 'R_cross = max_c max|a_c| / median_c max|a_c| (undefined if the median is 0)',
                            'within_channel': 'R_within_c = max|a_c| / RMS_c for channels with RMS_c > 1e-12 x max RMS',
                            'outlier_channel': f'max|a_c| >= {OUT_MULT} x median_c max|a_c| (fixed threshold)',
                            'tokens': 'share of sum a^2 in the top 1% rows (ceil), share in position-0 rows (token index % 128 == 0), position-0 row count',
                            'states': 'FP32 reference and q0 working state on the calibration windows (forward only, own dynamic routes)',
                            'sample_size': f'expert rows reported for every tensor; taps with < {MIN_ROWS} FP32 rows are not eligible for selection'},
        'error_metrics': {'candidates': list(CANDS), 'per_tensor': 'MSE, mean error, error variance; INT clipping fraction and clipped share of the error '
                                                                 'energy; share of error energy in the outlier channels, in the top 1% channels, '
                                                                 'in the top 1% rows and in position-0 rows',
                          'bc_decomposition': 'per physical slot: variance part sum_c G2/n (S2 - n mu^2) and bias part sum_c GS mu^2, and their shares '
                                              'in the outlier channels; FP32: static caches; q0: error sums of the error-only round and q0 gradient '
                                              'statistics of the all-node round (both verified there)',
                          'caution': 'predicted-cost shares are not measured-damage shares; large magnitude is not importance'},
        'selection': {'eligible': f'searched experts, FP32 calibration rows >= {MIN_ROWS}, R_cross finite', 'strata': '6 tap types x outlier class',
                      'strong': f'R_cross >= {int(100 * STRONG_Q)}th percentile within the tap type (eligible)', 'weak': f'R_cross <= {int(100 * WEAK_Q)}th percentile',
                      'per_stratum': N_PER_STRATUM, 'total': 6 * 2 * N_PER_STRATUM,
                      'rule': f'per type, {N_PER_STRATUM} strong taps drawn at random (seed {SEEDS["select"]}); each matched to a weak tap of the same type with the '
                              'same q0 activation bit class of its projection (<= 6 / 7-8 / >= 16 bits) if available, else any, at the nearest log rows; '
                              'no expert used twice; no BC-error information used',
                      'outlier_channels_of_a_tap': 'from FP32 discovery (calibration) only'},
        'bc_prediction': {'fp32_background': 'per physical slot, per candidate: T(f) = sum_c G2/n_s (S2 - n_s mu^2) + GS mu^2 from the static caches '
                                             '(x taps: total gradient, as BC); path-specific variant for x taps reported separately',
                          'tie': f'BC tie if |T1 - T2| / max(T1, T2) < {BC_TIE}'},
        'confirmation': {'data': f'{N_ART} unused WikiText-103 train articles x {WPA_C} windows of {SEQ}; strict boundaries, duplicate and near-duplicate '
                                 'exclusion; every earlier manifest excluded; drawn after the freeze', 'excluded_previous_wt103': len(used),
                         'eligible_before_length_and_neardup': len(elig), 'manifests_scanned': manifests},
        'measurement': {'main': 'FP32 background (weights and all other activations FP32), one physical tap quantized with one candidate; normal dynamic route; '
                                'dNLL = NLL(tap) - NLL(FP32) on the confirmation articles; layers before the tap reused from the FP32 pass (bitwise '
                                'control); routing changes downstream recorded; INT clipping on confirmation recorded at the tap',
                        'ranking': 'per tap, all 15 candidate pairs: BC order vs measured order; decisive pair = 95% article-bootstrap CI of the measured '
                                   'difference excludes 0; accuracy on decisive pairs without BC tie; INT-INT, Posit-Posit, cross, same-bit pairs separately; '
                                   'near-zero measured differences counted; per-tap Spearman between T and dNLL',
                        'outlier_link': 'misranking rate strong vs weak (tap bootstrap); Spearman of R_cross with per-tap misranking (descriptive)',
                        'q0_background': 'local replacement of one projection\'s activation format (its x and y QDQ) in q0, 4 cases (first selected strong and '
                                         'weak tap of the gate/up types and of the down types), alternatives int6 and p6e1 (int8 / p8e1 if equal to the '
                                         'current format); compared with the static-table projection cost difference; marginal effect in the full '
                                         'configuration, reported separately from the FP32 single-tap results',
                        'route': 'dynamic route everywhere; no fixed-route variant in this round'},
        'intervention': {'trigger': TRIGGER, 'rule': 'run only if T1, T2 or T3 holds (evaluated after the main measurement, recorded before running)',
                         'cases': 'strong taps with >= 1 outlier channel; int4 and int6 (Posit excluded: no range)',
                         'variants': {'I1': 'outlier-channel bypass: QDQ with the original range, then the outlier channels set to their FP32 values',
                                      'I2': f'random bypass: the same number of non-outlier channels, seed {SEEDS["random_bypass"]}',
                                      'I2b': 'typical bypass: the same number of channels with max|a| closest to the median',
                                      'I3': 'I1 plus range recalibration of the other channels from the FP32 discovery min / max of the non-outlier channels'},
                         'reading': 'I1 vs I2 / I2b: damage from the outlier channels\' own error; I3 vs I1: damage from the outlier-widened shared range; '
                                    'oracles only, chosen from discovery data only'},
        'statistics': {'bootstrap': f'{BOOT} article bootstraps (seed {SEEDS["bootstrap"]}); taps are the units for strong vs weak (seed {SEEDS["tap_bootstrap"]})',
                       'no_feedback': 'no candidate, tap or threshold changes after seeing confirmation results'},
        'budget_minutes': BUDGET_MIN, 'stop': 'if the main measurement exceeds its budget, stop after the current layer group and analyse the completed taps',
        'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info()}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE activation outlier 與 BC 排序誤差（診斷）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 不預設因果', '- 分開回答：outlier 是否存在、是否主導量化誤差與實測 NLL 損傷、BC 是否因此排錯。',
          '- 不修改 criterion、不 fine-tune、不重選 knee、不改成 weight-only、不加入新量化方法；干預只是診斷 oracle。', '',
          '## 量化粒度（程式確認）', '- INT activation：每個實體 QDQ 位置（expert × 投影 × x／y 槽）一組 scalar scale／zero-point，來自 FP32 calibration 的 [min, max]（含 0）；是 per-tensor，不是 per-channel scale。',
          '- Posit activation：直接對原值做 LUT 捨入，沒有 scale 也沒有 range；INT 的 range／clipping 機制不套用到 Posit。',
          '- BC 使用逐 channel 的 G2、GS 與誤差統計（per-channel 統計 ≠ per-channel scale）。',
          '- 實體位置：gate.x、up.x（同一輸入 xe 的兩份副本、range 相同；xe 的統計只算一次）、gate.y、up.y、down.x、down.y。BC 對 gate.x 與 up.x 都用 xe 的 total gradient；選中的 x tap 另算單一路徑梯度作診斷。', '',
          '## Outlier 指標（門檻固定）', f'- 逐 channel：max|a|、min、max、RMS、mean、|a| 的 p99（rows ≥ 100）、max 是否出現在 position 0。',
          f'- 跨 channel：R_cross = max_c max|a_c| ÷ median_c max|a_c|；channel 內：max|a_c| ÷ RMS_c（RMS 近零者排除）。',
          f'- outlier channel：max|a_c| ≥ {OUT_MULT} × median_c max|a_c|。token：前 1% rows 與 position-0 rows 的 Σa² 占比。',
          f'- 狀態：FP32 與 q0，calibration 窗口；每個張量報告 expert token 數；FP32 rows < {MIN_ROWS} 不參與選點。', '',
          '## 誤差', f'- 候選：{", ".join(CANDS)}。逐張量：MSE、平均誤差、變異；INT clipping 比例與 clipped 誤差占比；outlier channel、前 1% channel、前 1% token、position-0 的誤差能量占比。',
          '- BC 分解：各實體 slot 的變異項 Σ G2/n·(S2 − n·μ²) 與偏差項 Σ GS·μ²，以及 outlier channel 的占比（預測成本占比 ≠ 實測損傷占比）。', '',
          '## 選點（不使用任何 BC 誤排資訊）', f'- 6 種 tap 類型 × 強（R_cross ≥ 同類 90 百分位）／弱（≤ 中位數），每格 {N_PER_STRATUM} 個，共 {6 * 2 * N_PER_STRATUM} 個。',
          f'- 每類隨機抽 {N_PER_STRATUM} 個強 tap（seed {SEEDS["select"]}），各配一個同類弱 tap：優先同 q0 activation 位元類別（≤6／7–8／≥16），再取 log rows 最近者；不重複使用 expert。',
          '- tap 的 outlier channel 只由 FP32 discovery（calibration）決定。', '',
          '## 資料切分', f'- discovery = 原 calibration（BC 也用它）；確認 = 凍結後才抽的 {N_ART} 篇新 WikiText-103 文章 × {WPA_C} 窗口（排除 {len(used)} 篇歷史文章）。', '',
          '## 主測量與排序', '- FP32 背景、一次只量化一個實體 tap、dynamic route；ΔNLL = NLL(tap) − NLL(FP32)；tap 之前的層重用 FP32 結果（逐位元控制）；記錄下游路由變化與確認資料上的 INT clipping。',
          f'- 每個 tap 15 個候選配對：BC 排序 vs 實測排序；決定性配對 = 實測差的 95% 文章 bootstrap CI 排除 0；BC 平手 = 相對差 < {BC_TIE}；INT-INT、Posit-Posit、跨家族、同位元分開；報告實測近零配對數與每個 tap 的 Spearman。',
          '- outlier 與誤排的關聯：強 vs 弱的誤排率（以 tap 為單位 bootstrap）；R_cross 與誤排率的 Spearman（描述）。',
          '- q0 背景：4 個代表案例的投影層級 activation 格式替換（int6、p6e1，若等於現行格式則改 int8／p8e1）；與 static 表的成本差比較；這是完整配置中的邊際效果，與 FP32 單點結果分開報告。', '',
          '## 干預（只有觸發才做）', f'- 觸發（任一）：T1 {TRIGGER["T1"]}；T2 {TRIGGER["T2"]}；T3 {TRIGGER["T3"]}。',
          '- 案例：至少有 1 個 outlier channel 的強 tap；int4、int6（Posit 沒有 range，不做）。',
          '- I1 outlier channel 誤差 bypass；I2 相同數量的隨機一般 channel bypass；I2b 最接近中位數幅度的 channel bypass；I3 在 I1 之上，其餘 channel 用不含 outlier channel 的 discovery range 重新校準。',
          '- I1 對 I2／I2b：outlier channel 自身誤差的損傷；I3 對 I1：outlier 撐大共用 range 的損傷；兩者分開解讀。', '',
          '## 統計與預算', f'- {BOOT} 次文章 bootstrap（seed {SEEDS["bootstrap"]}）；強弱比較以 tap 為單位（seed {SEEDS["tap_bootstrap"]}）。',
          f'- 預算（分鐘）：{BUDGET_MIN}；主測量超過預算則在目前層組結束後停止，只分析已完成的 tap。看到確認結果後不改候選、tap 或門檻。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; previous WT103 {len(used)}; eligible {len(elig)}')


def discovery(out, log):
    """forward-only pass over the calibration windows in the FP32 and q0 states; activation and error statistics per expert tensor."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_search_final import dev_quant, quantized_layer
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'discovery_tensors.csv').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    names, nodes, elems, flops, rows_ = pr.problem_static()
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    gidx = read(GC_INDEX)
    experts = gidx['experts']
    erow = {e: i for i, e in enumerate(experts)}
    searched = defaultdict(dict)
    for n in names:
        l, e, p = n.split('.')
        searched[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    rows = []
    for state in ('fp32', 'q0'):
        fm = None if state == 'fp32' else read(Q0K)
        arr = {k: np.full((len(experts), ss.NCOL), np.nan, np.float32) for k in ('cmax', 'cmin', 'cmaxs', 'rms', 'mean', 'p99')}
        arr['pos0_at_max'] = np.zeros((len(experts), ss.NCOL), np.bool_)
        nrows, npos0 = np.zeros(len(experts), np.int64), np.zeros(len(experts), np.int64)
        with torch.inference_mode():
            cfg, ck = oc.config(), oc.Checkpoint()
            toks = torch.from_numpy(np.load(CALIB)).to('cuda')
            cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
            g = ck.global_('cuda', torch.float32)
            h = F.embedding(toks, g['embed'])
            for l in range(NL):
                w = ck.layer(l, 'cuda', torch.float32)
                wq, hk = (w, None) if fm is None else quantized_layer(w, fm, l, dq, ranges, NE)
                obs = {}
                want = searched.get(l, {})
                h, idx, rw = ke.layer_observe(wq, h, cfg, cos, sin, pr.act_map(hk), want, obs)
                for e in sorted(want):
                    node = f'L{l}.E{e}'
                    i = erow[node]
                    if e not in obs:
                        continue
                    c = obs[e]
                    tok = c['tok']
                    pos0 = (tok % SEQ) == 0
                    n = int(len(tok))
                    nrows[i], npos0[i] = n, int(pos0.sum())
                    for col, (p0, s0, k0) in ss.COL_SRC.items():
                        a_, b_ = ss.COLOFF[col]
                        t = c[k0]
                        C = t.shape[1]
                        ab = t.abs()
                        cmax, amax = ab.max(0)
                        rms = t.pow(2).mean(0).sqrt()
                        arr['cmax'][i, a_:b_] = cmax.cpu().numpy(); arr['rms'][i, a_:b_] = rms.cpu().numpy()
                        arr['mean'][i, a_:b_] = t.mean(0).cpu().numpy()
                        arr['cmin'][i, a_:b_] = t.min(0).values.cpu().numpy(); arr['cmaxs'][i, a_:b_] = t.max(0).values.cpu().numpy()
                        if n >= 100:
                            arr['p99'][i, a_:b_] = ab.kthvalue(int(math.ceil(0.99 * n)), dim=0).values.cpu().numpy()
                        arr['pos0_at_max'][i, a_:b_] = pos0[amax].cpu().numpy()
                        med = float(cmax.median())
                        valid = rms > 1e-12 * float(rms.max()) if float(rms.max()) > 0 else torch.zeros_like(rms, dtype=torch.bool)
                        rw_ = (cmax[valid] / rms[valid]) if bool(valid.any()) else None
                        outm = (cmax >= OUT_MULT * med) if med > 0 else torch.zeros_like(cmax, dtype=torch.bool)
                        rowE = t.double().pow(2).sum(1)
                        totE = float(rowE.sum())
                        k1 = max(1, int(math.ceil(0.01 * n)))
                        r = {'state': state, 'node': node, 'layer': l, 'expert': e, 'tensor': k0, 'column': col, 'rows': n, 'pos0_rows': int(pos0.sum()), 'channels': C,
                             'cmax_max': float(cmax.max()), 'cmax_median': med, 'R_cross': float(cmax.max()) / med if med > 0 else float('nan'),
                             'R_within_max': float(rw_.max()) if rw_ is not None else float('nan'), 'R_within_median': float(rw_.median()) if rw_ is not None else float('nan'),
                             'n_outlier_channels': int(outm.sum()), 'outlier_max_at_pos0': int((outm & pos0[amax]).sum()),
                             'top1pct_rows_energy_share': float(rowE.topk(k1).values.sum()) / totE if totE > 0 else float('nan'),
                             'pos0_energy_share': float(rowE[pos0].sum()) / totE if totE > 0 else float('nan')}
                        rng = ranges[f'{node}.{p0}'][s0]
                        r['range_lo'], r['range_hi'] = float(rng[0]), float(rng[1])
                        kc = max(1, int(math.ceil(0.01 * C)))
                        for f in CANDS:
                            q = dq.act(t, f, rng)
                            err = (q - t).double()
                            e2 = err.pow(2)
                            chE, rE = e2.sum(0), e2.sum(1)
                            tot = float(chE.sum())
                            mse = tot / (n * C)
                            me = float(err.mean())
                            r[f'{f}_mse'], r[f'{f}_mean'], r[f'{f}_var'] = mse, me, mse - me * me
                            r[f'{f}_outlier_err_share'] = float(chE[outm].sum()) / tot if tot > 0 else float('nan')
                            r[f'{f}_top1pct_channel_err_share'] = float(chE.topk(kc).values.sum()) / tot if tot > 0 else float('nan')
                            r[f'{f}_top1pct_row_err_share'] = float(rE.topk(k1).values.sum()) / tot if tot > 0 else float('nan')
                            r[f'{f}_pos0_err_share'] = float(rE[pos0].sum()) / tot if tot > 0 else float('nan')
                            if f.startswith('int'):
                                clip = int_clip_mask(t, f, rng)
                                r[f'{f}_clip_frac'] = float(clip.double().mean())
                                r[f'{f}_clip_err_share'] = float(e2[clip].sum()) / tot if tot > 0 else float('nan')
                        rows.append(r)
                del wq, hk, obs, w
                log(f'discovery {state}: layer {l} ({time.perf_counter() - t0:.0f}s)')
        np.savez(out / f'discovery_channels_{state}.npz', rows=nrows, pos0_rows=npos0, **arr)
    with open(out / 'discovery_tensors.csv', 'w') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0])); wcsv.writeheader(); wcsv.writerows(rows)
    write(out / 'discovery_meta.json', {'experts': experts, 'columns': ss.COLS, 'column_offsets': ss.COLOFF, 'column_tensor': {c: v[2] for c, v in ss.COL_SRC.items()},
                                        'candidates': CANDS, 'outlier_mult': OUT_MULT})
    log_stage(out, 'discovery', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(f'discovery done: {len(rows)} tensor rows')


def load_discovery(out):
    with open(out / 'discovery_tensors.csv') as fh:
        rows = list(csv.DictReader(fh))
    conv = lambda v: float(v) if v not in ('', 'nan') else float('nan')
    for r in rows:
        for k, v in list(r.items()):
            if k not in ('state', 'node', 'tensor', 'column'):
                r[k] = conv(v)
    return rows


def bc_slot_costs(G2, GS, S1, S2, n):
    """per candidate (rows of S1 / S2): variance part and bias part per channel."""
    mu = S1 / n
    V = G2[None] / n * (S2 - n * mu ** 2)
    Bc = GS[None] * mu ** 2
    return V, Bc


def select(out, log):
    """pre-registered tap selection on the FP32 discovery statistics; BC predictions and decompositions from the caches."""
    from run_native_bc_pipeline import QUANT
    if (out / 'selected_taps.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    rows = load_discovery(out)
    q0 = read(Q0K)
    meta = read(out / 'discovery_meta.json')
    erow = {e: i for i, e in enumerate(meta['experts'])}
    disc = {s: np.load(out / f'discovery_channels_{s}.npz') for s in ('fp32', 'q0')}
    fp = {(r['node'], r['tensor']): r for r in rows if r['state'] == 'fp32'}
    # eligible taps per type
    rng = np.random.default_rng(SEEDS['select'])
    bitclass = lambda f: 0 if bits(f) <= 6 else 1 if bits(f) <= 8 else 2
    taps, used_e = [], set()
    pool_info = {}
    for tt, (p, s, k, col) in TAP_TYPES.items():
        el = [r for (node, ten), r in fp.items() if ten == k and r['rows'] >= MIN_ROWS and math.isfinite(r['R_cross'])]
        Rs = np.array([r['R_cross'] for r in el])
        hi_t, lo_t = np.quantile(Rs, STRONG_Q), np.quantile(Rs, WEAK_Q)
        strong = sorted([r for r in el if r['R_cross'] >= hi_t], key=lambda r: r['node'])
        weak = sorted([r for r in el if r['R_cross'] <= lo_t], key=lambda r: r['node'])
        pool_info[tt] = {'eligible': len(el), 'strong_threshold': float(hi_t), 'weak_threshold': float(lo_t), 'strong_pool': len(strong), 'weak_pool': len(weak)}
        order = rng.permutation(len(strong)).tolist()
        got = 0
        for j in order:
            if got == N_PER_STRATUM:
                break
            sr = strong[j]
            if sr['node'] in used_e:
                continue
            bc_s = bitclass(q0[f'{sr["node"]}.{p}']['a'])
            cand = [r for r in weak if r['node'] not in used_e and r['node'] != sr['node']]
            same = [r for r in cand if bitclass(q0[f'{r["node"]}.{p}']['a']) == bc_s]
            pick_from = same or cand
            if not pick_from:
                continue
            d = np.array([abs(math.log(r['rows']) - math.log(sr['rows'])) for r in pick_from])
            best = [pick_from[i] for i in np.flatnonzero(d == d.min())]
            wr = best[int(rng.integers(len(best)))]
            for cls, r in (('strong', sr), ('weak', wr)):
                taps.append({'id': len(taps), 'pair': f'{tt}#{got}', 'type': tt, 'class': cls, 'node': r['node'], 'layer': int(r['layer']), 'expert': int(r['expert']),
                             'proj': p, 'slot': s, 'tensor': k, 'column': col, 'rows_fp32': int(r['rows']), 'R_cross': r['R_cross'], 'n_outlier_channels': int(r['n_outlier_channels']),
                             'q0_act_format': q0[f'{r["node"]}.{p}']['a'], 'q0_weight_format': q0[f'{r["node"]}.{p}']['w'], 'bit_class_matched': bool(same) if cls == 'weak' else None})
                used_e.add(r['node'])
            got += 1
    # BC predictions (FP32 static caches) and decompositions; outlier channels from FP32 discovery
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    fi = [QUANT.index(f) for f in CANDS]
    for tp in taps:
        node, col = tp['node'], tp['column']
        a_, b_ = ss.COLOFF[col]
        i = erow[node]
        cmax = disc['fp32']['cmax'][i, a_:b_]
        med = float(np.median(cmax))
        outm = cmax >= OUT_MULT * med
        tp['outlier_channels'] = np.flatnonzero(outm).tolist()
        row, grow, ns = np.asarray(ac[aidx['row'][node]]), np.asarray(gc[gidx['row'][node]]), aidx['n'][node]
        V, Bc = bc_slot_costs(grow[0, a_:b_], grow[1, a_:b_], row[0][fi][:, a_:b_], row[1][fi][:, a_:b_], ns)
        tp['bc'] = {f: {'T': float(V[j].sum() + Bc[j].sum()), 'variance': float(V[j].sum()), 'bias': float(Bc[j].sum()),
                        'outlier_share_of_T': float((V[j][outm].sum() + Bc[j][outm].sum()) / (V[j].sum() + Bc[j].sum())) if (V[j].sum() + Bc[j].sum()) > 0 else float('nan')}
                    for j, f in enumerate(CANDS)}
    # BC decomposition for every expert and physical slot, both states (descriptive)
    zc, smc = np.load(C_STATS / 'target_stats.npz'), read(C_STATS / 'target_stats_meta.json')
    qe, qidx = np.load(Q0_ERR, mmap_mode='r'), read(Q0_ERR_INDEX)
    drows = []
    for node in meta['experts']:
        i = erow[node]
        row, grow, ns = np.asarray(ac[aidx['row'][node]]), np.asarray(gc[gidx['row'][node]]), aidx['n'][node]
        qrow, nq = np.asarray(qe[qidx['row'][node]]), qidx['n_q0'].get(node, 0)
        for tt, (p, s, k, col) in TAP_TYPES.items():
            a_, b_ = ss.COLOFF[col]
            for state in ('fp32', 'q0'):
                cm = disc[state]['cmax'][i, a_:b_]
                med = float(np.nanmedian(cm)) if np.isfinite(cm).any() else float('nan')
                outm = cm >= OUT_MULT * med if med > 0 else np.zeros_like(cm, bool)
                if state == 'fp32':
                    V, Bc = bc_slot_costs(grow[0, a_:b_], grow[1, a_:b_], row[0][fi][:, a_:b_], row[1][fi][:, a_:b_], ns)
                else:
                    if nq == 0:
                        continue
                    nm = f'{node}.{p}'
                    V, Bc = bc_slot_costs(zc[f'{nm}|{s}_G2'], zc[f'{nm}|{s}_GS'], qrow[0][fi][:, a_:b_], qrow[1][fi][:, a_:b_], nq)
                for j, f in enumerate(CANDS):
                    Tt = float(V[j].sum() + Bc[j].sum())
                    drows.append({'state': state, 'node': node, 'tap': tt, 'format': f, 'T': Tt, 'variance': float(V[j].sum()), 'bias': float(Bc[j].sum()),
                                  'bias_share': float(Bc[j].sum()) / Tt if Tt > 0 else float('nan'),
                                  'outlier_share_of_T': float(V[j][outm].sum() + Bc[j][outm].sum()) / Tt if Tt > 0 else float('nan'),
                                  'n_outlier_channels': int(outm.sum())})
    with open(out / 'bc_decomposition.csv', 'w') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(drows[0])); wcsv.writeheader(); wcsv.writerows(drows)
    write(out / 'selected_taps.json', {'rule': read(out / 'plan.json')['selection'], 'pools': pool_info, 'taps': taps})
    log_stage(out, 'select', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'taps': len(taps), 'by_type_class': dict(Counter(f"{t['type']}|{t['class']}" for t in taps)), 'pools': pool_info})))


def pathgrad(out, log):
    """FP32 backward (routes constant) for the selected experts: total G2 / GS per selected slot (control against the cache) and the
    path-specific gradient of gate.x / up.x (gradient through that consumer only) for a diagnostic BC variant."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'pathgrad.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    taps = read(out / 'selected_taps.json')['taps']
    ranges = read(RANGES)
    want_all = defaultdict(dict)
    for tp in taps:
        want_all[tp['layer']][tp['expert']] = ['gate_proj', 'up_proj', 'down_proj']
    cfg, ck = oc.config(), oc.Checkpoint()
    toks = torch.from_numpy(np.load(CALIB)).to('cuda')
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g = ck.global_('cuda', torch.float32)
    STEQ = pr.ste_fn()
    routes, hs = [], []
    with torch.no_grad():
        h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            w = ck.layer(l, 'cuda', torch.float32)
            h, idx, rw = pr.layer_tap(w, h, cfg, cos, sin, None, None, {})
            routes.append((idx, rw)); hs.append(h)
    hT = hs[-1].clone().requires_grad_(True)
    head_ce(g, hT, toks, cfg).sum().backward()
    gn = hT.grad
    stats = {}
    for l in reversed(range(NL)):
        w = ck.layer(l, 'cuda', torch.float32)
        want = want_all.get(l, {})
        cap = {}
        hi = hs[l].clone().requires_grad_(True)
        o = ss.layer_tap_sel(w, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(None, STEQ), want, cap)
        o.backward(gn)
        gn = hi.grad
        with torch.no_grad():
            for e in want:
                c = cap[e]
                win = c['tok'] // SEQ
                wins = sorted(set(win.tolist()))
                gs = lambda gt: ((gt ** 2).sum(0), torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0))
                st = {k: gs(c[k].grad.double()) for k in ('xe', 'zg', 'zu', 'h', 'zd')}
                pg = (c['zg'].grad @ w['gate_proj'][e]).double()
                pu = (c['zu'].grad @ w['up_proj'][e]).double()
                st['gate_path'], st['up_path'] = gs(pg), gs(pu)
                xsum = (pg + pu - c['xe'].grad.double()).abs().max() / c['xe'].grad.double().abs().max()
                stats[f'L{l}.E{e}'] = {k: (v[0].cpu().numpy(), v[1].cpu().numpy()) for k, v in st.items()} | {'path_sum_rel_err': float(xsum)}
        del w, cap
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    fi = [QUANT.index(f) for f in CANDS]
    res, ctl = {}, {}
    for tp in taps:
        node, col, k = tp['node'], tp['column'], tp['tensor']
        a_, b_ = ss.COLOFF[col]
        grow = np.asarray(gc[gidx['row'][node]])
        st = stats[node]
        ctl[tp['id']] = {'G2_equal_cache': bool(np.array_equal(st[k][0], grow[0, a_:b_])), 'GS_equal_cache': bool(np.array_equal(st[k][1], grow[1, a_:b_])),
                         'path_sum_rel_err': st['path_sum_rel_err']}
        if tp['type'] in ('gate.x', 'up.x'):
            G2p, GSp = st['gate_path' if tp['type'] == 'gate.x' else 'up_path']
            row, ns = np.asarray(ac[aidx['row'][node]]), aidx['n'][node]
            V, Bc = bc_slot_costs(G2p, GSp, row[0][fi][:, a_:b_], row[1][fi][:, a_:b_], ns)
            res[tp['id']] = {f: {'T_path': float(V[j].sum() + Bc[j].sum()), 'variance': float(V[j].sum()), 'bias': float(Bc[j].sum()),
                                 'ratio_total_to_path': tp['bc'][f]['T'] / float(V[j].sum() + Bc[j].sum()) if float(V[j].sum() + Bc[j].sum()) > 0 else float('nan')}
                             for j, f in enumerate(CANDS)}
    write(out / 'pathgrad.json', {'path_specific_bc': res, 'controls': ctl,
                                  'pass': all(v['G2_equal_cache'] and v['GS_equal_cache'] and v['path_sum_rel_err'] < 1e-5 for v in ctl.values())})
    log_stage(out, 'pathgrad', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(json.dumps(dr.clean({'pass': all(v['G2_equal_cache'] and v['GS_equal_cache'] for v in ctl.values()), 'max_path_sum_rel_err': max(v['path_sum_rel_err'] for v in ctl.values())})))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    if not read(out / 'pathgrad.json')['pass']:
        raise SystemExit('pathgrad controls failed')
    res = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in ('plan.json', 'selected_taps.json', 'pathgrad.json', 'discovery_tensors.csv',
                                                                                         'bc_decomposition.csv', 'discovery_channels_fp32.npz', 'discovery_channels_q0.npz')},
           'code': code_hashes(), 'note': 'taps, outlier channels, BC predictions and candidates frozen before any confirmation article is drawn'}
    write(out / 'freeze.json', res)
    log('frozen')


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')
    used, _ = ss.used_wt103_all(out)
    if len(used) != pl['confirmation']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    wt2 = mr.split_articles(mr.load_lines(mr.WT2))
    lines = mr.load_lines(mr.WT103)
    w103 = mr.split_articles(lines)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    elig = [a for a in base if a['ch'] not in used]
    if len(prev) != len(used) or len(elig) != pl['confirmation']['eligible_before_length_and_neardup']:
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
        enc = tok(text, add_special_tokens=False)
        ids = enc['input_ids']
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
                     'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca}, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
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
# forward passes on the confirmation windows
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, fm=None):
        import torch
        import olmoe_common as oc
        from run_olmoe_search_final import dev_quant
        self.cfg, self.ck = oc.config(), oc.Checkpoint()
        self.cos, self.sin = oc.rope(self.cfg, SEQ, 'cuda', torch.float32)
        self.g = self.ck.global_('cuda', torch.float32)
        self.ranges = read(RANGES)
        self.fm = fm
        self.dq = None

    def layer(self, l, dq):
        import torch
        from run_olmoe_search_final import quantized_layer
        w = self.ck.layer(l, 'cuda', torch.float32)
        if self.fm is None:
            return w, {}
        wq, hk = quantized_layer(w, self.fm, l, dq, self.ranges, NE)
        return wq, dict(hk.act)


def layer_step(ctx, wq, h, l, acts):
    """run_olmoe_dynamic_route.forward's per-layer arithmetic (dynamic route); acts: (l, e, p, s) -> callable."""
    import olmoe_common as oc
    hk = oc.ExpertHooks()
    hk.act = acts
    B = h.shape[0]
    Hd = ctx.cfg.hidden_size
    x = h + oc.attention(wq, oc.rms(h, wq['ln1'], ctx.cfg.rms_norm_eps), ctx.cfg, ctx.cos, ctx.sin)
    z = oc.rms(x, wq['ln2'], ctx.cfg.rms_norm_eps).reshape(-1, Hd)
    idx, rw = oc.route(wq, z, ctx.cfg)
    return x + oc.experts_fixed(wq, z, idx, rw, l, hk).view(B, SEQ, Hd), idx


def base_pass(ctx, toks_np, dq, log, label):
    """full pass; returns per-window summed CE, the per-layer input hidden states (CPU, per chunk) and the per-layer route ids."""
    import torch
    import torch.nn.functional as F
    from run_olmoe_fixed_route import head_ce
    nW = len(toks_np)
    chunks = [(a, min(nW, a + CHUNK)) for a in range(0, nW, CHUNK)]
    with torch.no_grad():
        H = [F.embedding(torch.from_numpy(toks_np[a:b]).to('cuda'), ctx.g['embed']) for a, b in chunks]
        H_in, ids = [], []
        for l in range(NL):
            wq, acts = ctx.layer(l, dq)
            H_in.append([h.cpu() for h in H])
            idl = []
            for ci, (a, b) in enumerate(chunks):
                H[ci], idx = layer_step(ctx, wq, H[ci], l, acts)
                idl.append(idx.view(b - a, SEQ, K).cpu())
            ids.append(torch.cat(idl))
            del wq, acts
            log(f'{label}: layer {l}')
        ce = []
        for ci, (a, b) in enumerate(chunks):
            ce += head_ce(ctx.g, H[ci], torch.from_numpy(toks_np[a:b]).to('cuda'), ctx.cfg).cpu().numpy().tolist()
    return ce, H_in, ids, chunks


def variants_from(ctx, toks_np, start_l, H_in_l, base_ids, chunks, variants, dq):
    """variants: list of (name, {l: {(l, e, p, s): fn}}) started at layer start_l from the cached inputs; dynamic route at every layer;
    returns per-variant per-window summed CE and the fraction of (token, layer) expert sets differing from the base after start_l."""
    import torch
    from run_olmoe_fixed_route import head_ce
    H = {name: [h.clone() for h in H_in_l] for name, _ in variants}
    changed = {name: [0, 0] for name, _ in variants}
    with torch.no_grad():
        for l in range(start_l, NL):
            wq, acts = ctx.layer(l, dq)
            for name, hooks in variants:
                a_ = dict(acts)
                a_.update(hooks.get(l, {}))
                for ci, (a, b) in enumerate(chunks):
                    hn, idx = layer_step(ctx, wq, H[name][ci].to('cuda'), l, a_)
                    H[name][ci] = hn.cpu()
                    if l > start_l:
                        bi = base_ids[l][a:b].to('cuda').view(-1, K)
                        ch = (torch.sort(idx, -1).values != torch.sort(bi.long(), -1).values).any(-1)
                        changed[name][0] += int(ch.sum()); changed[name][1] += ch.numel()
            del wq, acts
        ce = {}
        for name, _ in variants:
            ce[name] = []
            for ci, (a, b) in enumerate(chunks):
                ce[name] += head_ce(ctx.g, H[name][ci].to('cuda'), torch.from_numpy(toks_np[a:b]).to('cuda'), ctx.cfg).cpu().numpy().tolist()
    return ce, {k: (v[0] / v[1] if v[1] else 0.0) for k, v in changed.items()}


def make_qdq(dq, f, rng, acc=None, bypass=None, rng_other=None):
    """QDQ of one physical tap; bypass: channel indices set to their FP32 values; rng_other: range for the non-bypassed channels;
    acc accumulates rows, error energy per channel and INT clipping on the confirmation data."""
    import torch

    def fn(t):
        q = dq.act(t, f, rng_other if rng_other is not None else rng)
        if bypass is not None and len(bypass):
            q = q.clone()
            q[:, bypass] = t[:, bypass]
        if acc is not None:
            e2 = (q - t).double().pow(2)
            acc['rows'] += t.shape[0]
            acc['chE'] = e2.sum(0) if acc['chE'] is None else acc['chE'] + e2.sum(0)
            if f.startswith('int'):
                clip = int_clip_mask(t, f, rng_other if rng_other is not None else rng)
                if bypass is not None and len(bypass):
                    clip[:, bypass] = False
                acc['clip'] += int(clip.sum()); acc['clipE'] += float(e2[clip].sum()); acc['n'] += clip.numel()
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
    ctx = Ctx(None)
    ce0, H_in, ids, chunks = base_pass(ctx, toks, dq, log, 'fp32 base')
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
            rng = ctx.ranges[f'{tp["node"]}.{tp["proj"]}'][tp['slot']]
            for f in CANDS:
                name = f'tap{tp["id"]}|{f}'
                accs[name] = {'rows': 0, 'chE': None, 'clip': 0, 'clipE': 0.0, 'n': 0}
                variants.append((name, {l: {(l, tp['expert'], tp['proj'], tp['slot']): make_qdq(dq, f, rng, accs[name])}}))
        ce, chg = variants_from(ctx, toks, l, H_in[l], ids, chunks, variants, dq)
        for name, _ in variants:
            if name == 'control_identity':
                controls['identity_from_layer'] = {'layer': l, 'max_abs': float(np.max(np.abs(np.array(ce[name]) - np.array(ce0))))}
                continue
            res['variants'][name] = ce[name]
            res['route_changed'][name] = chg[name]
            a = accs[name]
            tp = taps[int(name.split('|')[0][3:])]
            chE = a['chE'].cpu().numpy() if a['chE'] is not None else None
            tot = float(chE.sum()) if chE is not None else 0.0
            oc_ = tp['outlier_channels']
            res['acc'][name] = {'rows': a['rows'], 'err_energy': tot, 'outlier_err_share': float(chE[oc_].sum()) / tot if tot > 0 and oc_ else (0.0 if tot > 0 else None),
                                'clip_frac': a['clip'] / a['n'] if a['n'] else None, 'clip_err_share': a['clipE'] / tot if a['n'] and tot > 0 else None}
        log(f'measure: layer {l} done ({time.perf_counter() - t0:.0f}s, {len(variants)} variants)')
    res['controls'] = controls
    res['stopped_early'] = stopped
    res['controls']['pass'] = controls['base_equals_dr_forward_max_abs'] == 0 and controls.get('identity_from_layer', {}).get('max_abs', 1) == 0
    write(out / 'measure.json', res)
    log_stage(out, 'measure', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'stopped_early': stopped})
    log(json.dumps(dr.clean({'controls': controls, 'variants': len(res['variants']), 'stopped': stopped})))


def q0local(out, log):
    """q0 background: one projection's activation format (x and y QDQ) replaced; marginal effect in the full configuration."""
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'q0local.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    taps = read(out / 'selected_taps.json')['taps']
    q0 = read(Q0K)
    st, rb = read(STATIC_TABLE), read(REFRESH_B_TABLE)
    cases = []
    for group in (('gate.x', 'up.x', 'gate.y', 'up.y'), ('down.x', 'down.y')):
        for cls in ('strong', 'weak'):
            tp = next(t for t in taps if t['type'] in group and t['class'] == cls)
            nm = f'{tp["node"]}.{tp["proj"]}'
            cur = q0[nm]['a']
            alts = [('int6' if cur != 'int6' else 'int8'), ('p6e1' if cur != 'p6e1' else 'p8e1')]
            cases.append({'tap': tp['id'], 'projection': nm, 'layer': tp['layer'], 'expert': tp['expert'], 'proj': tp['proj'], 'class': cls, 'current': cur, 'alternatives': alts,
                          'bc_static_delta': {a: st['C'][nm][a] - st['C'][nm][cur] for a in alts}, 'bc_q0_refresh_delta': {a: rb['C'][nm][a] - rb['C'][nm][cur] for a in alts}})
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ctx = Ctx(q0)
    ce0, H_in, ids, chunks = base_pass(ctx, toks, dq, log, 'q0 base')
    ref = dr.forward(toks, q0, dq, ctx.ranges, record=False, log=lambda m: None, label='dr q0')
    res = {'cases': cases, 'base_ce': ce0, 'variants': {}, 'route_changed': {}, 'controls': {'base_equals_dr_forward_max_abs': float(np.max(np.abs(np.array(ce0) - np.array(ref['ce']))))}}
    by_layer = defaultdict(list)
    for c in cases:
        by_layer[c['layer']].append(c)
    for l in sorted(by_layer):
        variants = [('control_identity', {})] if l == min(by_layer) else []
        for c in by_layer[l]:
            for a in c['alternatives']:
                hooks = {(l, c['expert'], c['proj'], s): (lambda t, a=a, r=ctx.ranges[c['projection']][s]: dq.act(t, a, r)) for s in ('x', 'y')}
                variants.append((f'case{c["tap"]}|{a}', {l: hooks}))
        ce, chg = variants_from(ctx, toks, l, H_in[l], ids, chunks, variants, dq)
        for name, _ in variants:
            if name == 'control_identity':
                res['controls']['identity_from_layer'] = {'layer': l, 'max_abs': float(np.max(np.abs(np.array(ce[name]) - np.array(ce0))))}
                continue
            res['variants'][name] = ce[name]
            res['route_changed'][name] = chg[name]
        log(f'q0local: layer {l} done ({time.perf_counter() - t0:.0f}s)')
    res['controls']['pass'] = res['controls']['base_equals_dr_forward_max_abs'] == 0 and res['controls'].get('identity_from_layer', {}).get('max_abs', 1) == 0
    write(out / 'q0local.json', res)
    log_stage(out, 'q0local', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(json.dumps(dr.clean(res['controls'])))


# --------------------------------------------------------------------------- #
def article_delta(ce_v, ce_0):
    d = np.array(ce_v) - np.array(ce_0)
    return d.reshape(N_ART, WPA_C).sum(1) / (WPA_C * TP)


def ranking(out):
    """pairwise BC vs measured ordering per tap (FP32 background)."""
    taps = read(out / 'selected_taps.json')['taps']
    pg = read(out / 'pathgrad.json')['path_specific_bc']
    m = read(out / 'measure.json')
    idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
    fam = lambda f: 'int' if f.startswith('int') else 'posit'
    per_tap, pairs = [], []
    for tp in taps:
        names = [f'tap{tp["id"]}|{f}' for f in CANDS]
        if not all(n in m['variants'] for n in names):
            continue
        D = {f: article_delta(m['variants'][f'tap{tp["id"]}|{f}'], m['base_ce']) for f in CANDS}
        dn = {f: float(D[f].mean()) for f in CANDS}
        T = {f: tp['bc'][f]['T'] for f in CANDS}
        Tp = {f: pg[str(tp['id'])][f]['T_path'] for f in CANDS} if str(tp['id']) in pg else None
        for i, f1 in enumerate(CANDS):
            for f2 in CANDS[i + 1:]:
                diff = D[f1] - D[f2]
                ci = np.percentile(diff[idx].mean(1), [2.5, 97.5])
                decisive = bool(ci[0] > 0 or ci[1] < 0)
                tie = abs(T[f1] - T[f2]) / max(T[f1], T[f2]) < BC_TIE if max(T[f1], T[f2]) > 0 else True
                correct = (T[f1] < T[f2]) == (diff.mean() < 0)
                rec = {'tap': tp['id'], 'type': tp['type'], 'class': tp['class'], 'f1': f1, 'f2': f2, 'family': 'int-int' if fam(f1) == fam(f2) == 'int' else
                       'posit-posit' if fam(f1) == fam(f2) == 'posit' else 'cross', 'same_bits': bits(f1) == bits(f2), 'measured_diff': float(diff.mean()),
                       'ci95': ci.tolist(), 'decisive': decisive, 'bc_tie': bool(tie), 'bc_correct': bool(correct)}
                if Tp is not None:
                    tiep = abs(Tp[f1] - Tp[f2]) / max(Tp[f1], Tp[f2]) < BC_TIE if max(Tp[f1], Tp[f2]) > 0 else True
                    rec['path_bc_tie'], rec['path_bc_correct'] = bool(tiep), bool((Tp[f1] < Tp[f2]) == (diff.mean() < 0))
                pairs.append(rec)
        from scipy.stats import spearmanr
        rho = spearmanr([T[f] for f in CANDS], [dn[f] for f in CANDS]).statistic
        ci_t = {f: np.percentile(D[f][idx].mean(1), [2.5, 97.5]).tolist() for f in CANDS}
        dec = [p for p in pairs if p['tap'] == tp['id'] and p['decisive'] and not p['bc_tie']]
        per_tap.append({'tap': tp['id'], 'type': tp['type'], 'class': tp['class'], 'node': tp['node'], 'R_cross': tp['R_cross'], 'rows_fp32': tp['rows_fp32'],
                        'n_outlier_channels': tp['n_outlier_channels'], 'dnll': dn, 'dnll_ci95': ci_t, 'bc_T': T, 'spearman_bc_vs_dnll': float(rho),
                        'decisive_pairs': len(dec), 'misranked_decisive': sum(not p['bc_correct'] for p in dec),
                        'misrank_rate': (sum(not p['bc_correct'] for p in dec) / len(dec)) if dec else None,
                        'route_changed': {f: m['route_changed'][f'tap{tp["id"]}|{f}'] for f in CANDS},
                        'confirmation_tap': {f: m['acc'][f'tap{tp["id"]}|{f}'] for f in CANDS}})
    return per_tap, pairs


def acc_summary(pairs, key='bc_correct', tie='bc_tie'):
    dec = [p for p in pairs if p['decisive'] and not p[tie]]
    return {'pairs': len(pairs), 'decisive': sum(p['decisive'] for p in pairs), 'bc_ties': sum(p[tie] for p in pairs),
            'not_decisive': sum(not p['decisive'] for p in pairs), 'evaluated': len(dec),
            'accuracy': (sum(p[key] for p in dec) / len(dec)) if dec else None}


def trigger(out, log):
    if (out / 'trigger.json').exists():
        raise SystemExit('exists')
    rows = load_discovery(out)
    taps = read(out / 'selected_taps.json')['taps']
    fp = {(r['node'], r['tensor']): r for r in rows if r['state'] == 'fp32'}
    strong = [t for t in taps if t['class'] == 'strong']
    t1_vals = [max(fp[(t['node'], t['tensor'])]['int4_outlier_err_share'], fp[(t['node'], t['tensor'])]['int6_outlier_err_share']) for t in strong if t['n_outlier_channels'] > 0]
    t1 = sum(v >= 0.5 for v in t1_vals) >= len(strong) / 2
    per_tap, pairs = ranking(out)
    pt = {p['tap']: p for p in per_tap}
    ratios = []
    for t in strong:
        w = next(x for x in taps if x['pair'] == t['pair'] and x['class'] == 'weak')
        if t['id'] in pt and w['id'] in pt:
            a, b = pt[t['id']]['dnll']['int4'], pt[w['id']]['dnll']['int4']
            if a > 0 and b > 0:
                ratios.append(a / b)
    t2 = bool(ratios) and float(np.median(ratios)) >= 2
    rng = np.random.default_rng(SEEDS['tap_bootstrap'])
    S = [i for i in pt if pt[i]['class'] == 'strong' and pt[i]['misrank_rate'] is not None]
    Wk = [i for i in pt if pt[i]['class'] == 'weak' and pt[i]['misrank_rate'] is not None]
    if S and Wk:
        dd = []
        for _ in range(BOOT):
            s = rng.choice(S, len(S)); w = rng.choice(Wk, len(Wk))
            dd.append(np.mean([pt[i]['misrank_rate'] for i in s]) - np.mean([pt[i]['misrank_rate'] for i in w]))
        diff = float(np.mean([pt[i]['misrank_rate'] for i in S]) - np.mean([pt[i]['misrank_rate'] for i in Wk]))
        ci = np.percentile(dd, [2.5, 97.5]).tolist()
        t3 = diff >= 0.10 and ci[0] > 0
    else:
        diff, ci, t3 = None, None, False
    res = {'T1': {'outlier_err_share_strong_taps': t1_vals, 'holds': bool(t1)}, 'T2': {'ratios': ratios, 'median': float(np.median(ratios)) if ratios else None, 'holds': bool(t2)},
           'T3': {'misrank_diff_strong_minus_weak': diff, 'ci95': ci, 'holds': bool(t3)}, 'run_intervention': bool(t1 or t2 or t3),
           'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write(out / 'trigger.json', res)
    log(json.dumps(dr.clean(res)))


def intervene(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'intervene.json').exists():
        raise SystemExit('exists')
    trg = read(out / 'trigger.json')
    if not trg['run_intervention']:
        write(out / 'intervene.json', {'skipped': True, 'reason': 'no pre-registered trigger held'})
        log('intervention skipped (no trigger)')
        return
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    taps = read(out / 'selected_taps.json')['taps']
    meta = read(out / 'discovery_meta.json')
    erow = {e: i for i, e in enumerate(meta['experts'])}
    disc = np.load(out / 'discovery_channels_fp32.npz')
    cases = [t for t in taps if t['class'] == 'strong' and t['n_outlier_channels'] > 0]
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ctx = Ctx(None)
    ce0, H_in, ids, chunks = base_pass(ctx, toks, dq, log, 'fp32 base (intervention)')
    rngs = np.random.default_rng(SEEDS['random_bypass'])
    res = {'base_ce': ce0, 'cases': [], 'variants': {}}
    by_layer = defaultdict(list)
    for tp in cases:
        by_layer[tp['layer']].append(tp)
    for l in sorted(by_layer):
        variants = []
        for tp in by_layer[l]:
            a_, b_ = ss.COLOFF[tp['column']]
            i = erow[tp['node']]
            cmax, cmin, cmaxs = disc['cmax'][i, a_:b_], disc['cmin'][i, a_:b_], disc['cmaxs'][i, a_:b_]
            O = np.array(tp['outlier_channels'])
            others = np.setdiff1d(np.arange(b_ - a_), O)
            R = np.sort(rngs.choice(others, len(O), replace=False))
            med = np.median(cmax)
            Ty = np.sort(others[np.argsort(np.abs(cmax[others] - med), kind='stable')[:len(O)]])
            rng = ctx.ranges[f'{tp["node"]}.{tp["proj"]}'][tp['slot']]
            rng3 = [float(cmin[others].min()), float(cmaxs[others].max())]
            res['cases'].append({'tap': tp['id'], 'node': tp['node'], 'type': tp['type'], 'k': len(O), 'outlier': O.tolist(), 'random': R.tolist(), 'typical': Ty.tolist(),
                                 'range': rng, 'range_recalibrated': rng3})
            key = (l, tp['expert'], tp['proj'], tp['slot'])
            Ot, Rt, Tyt = (torch.tensor(x, device='cuda', dtype=torch.long) for x in (O, R, Ty))
            for f in ('int4', 'int6'):
                variants += [(f'tap{tp["id"]}|{f}|I1', {l: {key: make_qdq(dq, f, rng, bypass=Ot)}}),
                             (f'tap{tp["id"]}|{f}|I2', {l: {key: make_qdq(dq, f, rng, bypass=Rt)}}),
                             (f'tap{tp["id"]}|{f}|I2b', {l: {key: make_qdq(dq, f, rng, bypass=Tyt)}}),
                             (f'tap{tp["id"]}|{f}|I3', {l: {key: make_qdq(dq, f, rng, bypass=Ot, rng_other=rng3)}})]
        ce, chg = variants_from(ctx, toks, l, H_in[l], ids, chunks, variants, dq)
        for name, _ in variants:
            res['variants'][name] = ce[name]
        log(f'intervene: layer {l} done ({time.perf_counter() - t0:.0f}s)')
    res['controls'] = {'base_equals_measure_base': ce0 == read(out / 'measure.json')['base_ce']}
    write(out / 'intervene.json', res)
    log_stage(out, 'intervene', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(json.dumps(dr.clean(res['controls'])))


def analyze(out, log):
    from scipy.stats import spearmanr
    t0 = time.perf_counter()
    rows = load_discovery(out)
    taps = read(out / 'selected_taps.json')['taps']
    res = {}
    # Q1 / Q2: discovery summaries
    q = lambda xs, p: float(np.nanpercentile(xs, p)) if len(xs) else None
    summ = {}
    for state in ('fp32', 'q0'):
        for k in ('xe', 'zg', 'zu', 'h', 'zd'):
            rs = [r for r in rows if r['state'] == state and r['tensor'] == k and r['rows'] > 0]
            Rc = [r['R_cross'] for r in rs if math.isfinite(r['R_cross'])]
            e = {'tensors': len(rs), 'rows_median': q([r['rows'] for r in rs], 50), 'rows_lt_min': sum(r['rows'] < MIN_ROWS for r in rs),
                 'R_cross_p50_p90_p99_max': [q(Rc, 50), q(Rc, 90), q(Rc, 99), max(Rc) if Rc else None],
                 'share_R_cross_ge_10': sum(x >= 10 for x in Rc) / len(Rc) if Rc else None,
                 'n_outlier_channels_p50_p90_max': [q([r['n_outlier_channels'] for r in rs], 50), q([r['n_outlier_channels'] for r in rs], 90),
                                                    max(r['n_outlier_channels'] for r in rs)],
                 'tensors_with_outlier_channels': sum(r['n_outlier_channels'] > 0 for r in rs),
                 'R_within_max_p50_p90': [q([r['R_within_max'] for r in rs], 50), q([r['R_within_max'] for r in rs], 90)],
                 'top1pct_rows_energy_share_p50_p90': [q([r['top1pct_rows_energy_share'] for r in rs], 50), q([r['top1pct_rows_energy_share'] for r in rs], 90)],
                 'pos0_energy_share_p50_p90': [q([r['pos0_energy_share'] for r in rs], 50), q([r['pos0_energy_share'] for r in rs], 90)],
                 'pos0_row_share_p50': q([r['pos0_rows'] / r['rows'] for r in rs], 50),
                 'outlier_channel_max_at_pos0_share': (sum(r['outlier_max_at_pos0'] for r in rs) / max(1, sum(r['n_outlier_channels'] for r in rs)))}
            for f in CANDS:
                e[f] = {kk: [q([r[f'{f}_{kk}'] for r in rs if r['n_outlier_channels'] > 0 or kk != 'outlier_err_share'], 50),
                             q([r[f'{f}_{kk}'] for r in rs if r['n_outlier_channels'] > 0 or kk != 'outlier_err_share'], 90)]
                        for kk in ('outlier_err_share', 'top1pct_channel_err_share', 'top1pct_row_err_share', 'pos0_err_share')
                        + (('clip_frac', 'clip_err_share') if f.startswith('int') else ())}
            by_layer = defaultdict(list)
            for r in rs:
                if math.isfinite(r['R_cross']):
                    by_layer[int(r['layer'])].append(r['R_cross'])
            e['R_cross_median_by_layer'] = {str(l): float(np.median(v)) for l, v in sorted(by_layer.items())}
            summ[f'{state}|{k}'] = e
    res['discovery_summary'] = summ
    fpm = {(r['node'], r['tensor']): r for r in rows if r['state'] == 'fp32'}
    q0m = {(r['node'], r['tensor']): r for r in rows if r['state'] == 'q0'}
    common = [k for k in fpm if k in q0m and math.isfinite(fpm[k]['R_cross']) and math.isfinite(q0m[k]['R_cross'])]
    res['fp32_vs_q0_R_cross_spearman'] = float(spearmanr([fpm[k]['R_cross'] for k in common], [q0m[k]['R_cross'] for k in common]).statistic)
    # BC decomposition summaries
    with open(out / 'bc_decomposition.csv') as fh:
        drows = list(csv.DictReader(fh))
    bsum = {}
    for state in ('fp32', 'q0'):
        for f in CANDS:
            rs = [r for r in drows if r['state'] == state and r['format'] == f]
            bs = [float(r['bias_share']) for r in rs if r['bias_share'] not in ('nan', '')]
            os_ = [float(r['outlier_share_of_T']) for r in rs if r['outlier_share_of_T'] not in ('nan', '') and int(r['n_outlier_channels']) > 0]
            bsum[f'{state}|{f}'] = {'slots': len(rs), 'bias_share_p50_p90': [q(bs, 50), q(bs, 90)], 'outlier_share_of_T_p50_p90': [q(os_, 50), q(os_, 90)],
                                    'slots_with_outlier_channels': len(os_)}
    res['bc_decomposition_summary'] = bsum
    # ranking
    per_tap, pairs = ranking(out)
    res['per_tap'] = per_tap
    res['ranking'] = {'all': acc_summary(pairs)}
    for grp, sel in (('strong', lambda p: p['class'] == 'strong'), ('weak', lambda p: p['class'] == 'weak'), ('int-int', lambda p: p['family'] == 'int-int'),
                     ('posit-posit', lambda p: p['family'] == 'posit-posit'), ('cross', lambda p: p['family'] == 'cross'), ('same_bits', lambda p: p['same_bits'])):
        res['ranking'][grp] = acc_summary([p for p in pairs if sel(p)])
    for t in TAP_TYPES:
        res['ranking'][f'type:{t}'] = acc_summary([p for p in pairs if p['type'] == t])
    xp = [p for p in pairs if 'path_bc_correct' in p]
    res['ranking']['x_taps_bc_total_gradient'] = acc_summary(xp)
    res['ranking']['x_taps_bc_path_gradient'] = acc_summary(xp, 'path_bc_correct', 'path_bc_tie')
    res['near_zero'] = {'pairs_not_decisive': sum(not p['decisive'] for p in pairs),
                        'pairs_abs_diff_lt_1e-6': sum(abs(p['measured_diff']) < 1e-6 for p in pairs),
                        'taps_all_dnll_ci_include_0': sum(all(c[0] <= 0 <= c[1] for c in t['dnll_ci95'].values()) for t in per_tap)}
    rates = [(t['R_cross'], t['misrank_rate']) for t in per_tap if t['misrank_rate'] is not None]
    res['outlier_vs_misrank_spearman'] = float(spearmanr(*zip(*rates)).statistic) if len(rates) > 2 else None
    res['trigger'] = read(out / 'trigger.json') if (out / 'trigger.json').exists() else None
    # q0 local
    if (out / 'q0local.json').exists():
        ql = read(out / 'q0local.json')
        idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
        qrows = []
        for c in ql['cases']:
            for a in c['alternatives']:
                d = article_delta(ql['variants'][f'case{c["tap"]}|{a}'], ql['base_ce'])
                qrows.append({'tap': c['tap'], 'projection': c['projection'], 'class': c['class'], 'current': c['current'], 'alternative': a,
                              'bc_static_delta': c['bc_static_delta'][a], 'bc_q0_refresh_delta': c['bc_q0_refresh_delta'][a], 'measured_dnll': float(d.mean()),
                              'ci95': np.percentile(d[idx].mean(1), [2.5, 97.5]).tolist(), 'route_changed': ql['route_changed'][f'case{c["tap"]}|{a}']})
        res['q0_local'] = {'rows': qrows, 'controls': ql['controls'],
                           'sign_agreement_static': sum((r['bc_static_delta'] > 0) == (r['measured_dnll'] > 0) for r in qrows),
                           'decisive': sum(r['ci95'][0] > 0 or r['ci95'][1] < 0 for r in qrows)}
    # intervention
    if (out / 'intervene.json').exists():
        iv = read(out / 'intervene.json')
        if not iv.get('skipped'):
            m = read(out / 'measure.json')
            idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N_ART, (BOOT, N_ART))
            ir = []
            for c in iv['cases']:
                for f in ('int4', 'int6'):
                    base = article_delta(m['variants'][f'tap{c["tap"]}|{f}'], m['base_ce'])
                    row = {'tap': c['tap'], 'type': c['type'], 'k': c['k'], 'format': f, 'base_dnll': float(base.mean())}
                    for v in ('I1', 'I2', 'I2b', 'I3'):
                        d = article_delta(iv['variants'][f'tap{c["tap"]}|{f}|{v}'], iv['base_ce'])
                        row[f'{v}_dnll'] = float(d.mean())
                        row[f'{v}_minus_base_ci95'] = np.percentile((d - base)[idx].mean(1), [2.5, 97.5]).tolist()
                    for a, b in (('I1', 'I2'), ('I1', 'I2b'), ('I3', 'I1')):
                        da = article_delta(iv['variants'][f'tap{c["tap"]}|{f}|{a}'], iv['base_ce'])
                        db = article_delta(iv['variants'][f'tap{c["tap"]}|{f}|{b}'], iv['base_ce'])
                        row[f'{a}_minus_{b}'] = float((da - db).mean())
                        row[f'{a}_minus_{b}_ci95'] = np.percentile((da - db)[idx].mean(1), [2.5, 97.5]).tolist()
                    ir.append(row)
            res['intervention'] = {'rows': ir, 'controls': iv['controls']}
        else:
            res['intervention'] = iv
    with open(out / 'ranking_pairs.csv', 'w') as fh:
        if pairs:
            wcsv = csv.DictWriter(fh, fieldnames=list(pairs[0])); wcsv.writeheader(); wcsv.writerows(pairs)
    with open(out / 'per_tap.csv', 'w') as fh:
        flat = [{'tap': t['tap'], 'type': t['type'], 'class': t['class'], 'node': t['node'], 'R_cross': t['R_cross'], 'rows_fp32': t['rows_fp32'],
                 'n_outlier_channels': t['n_outlier_channels'], 'spearman': t['spearman_bc_vs_dnll'], 'decisive_pairs': t['decisive_pairs'], 'misrank_rate': t['misrank_rate'],
                 **{f'dnll_{f}': t['dnll'][f] for f in CANDS}, **{f'bcT_{f}': t['bc_T'][f] for f in CANDS}} for t in per_tap]
        if flat:
            wcsv = csv.DictWriter(fh, fieldnames=list(flat[0])); wcsv.writeheader(); wcsv.writerows(flat)
    m = read(out / 'measure.json')
    with open(out / 'per_window.csv', 'w') as fh:
        keys = ['base'] + sorted(m['variants'])
        wcsv = csv.writer(fh); wcsv.writerow(['window', 'article'] + keys)
        for i in range(len(m['base_ce'])):
            wcsv.writerow([i, i // WPA_C, m['base_ce'][i]] + [m['variants'][k][i] for k in keys[1:]])
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['measure_controls'] = m['controls']
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'ranking': res['ranking']['all'], 'strong': res['ranking']['strong'], 'weak': res['ranking']['weak'],
                             'outlier_vs_misrank_spearman': res['outlier_vs_misrank_spearman']})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'discovery', 'select', 'pathgrad', 'freeze', 'data', 'measure', 'q0local', 'trigger', 'intervene', 'analyze'], required=True)
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
