#!/usr/bin/env python3
"""OLMoE spike rows: does BC's channel-aggregated sensitivity ignore the position correspondence of gradients and quantization
errors? A diagnostic pairing score, pre-registered in plan.json / plan.md.

On the BC calibration rows of the Down input h (FP32 working state, original BC gradient semantics), per element
g[t,c] = dloss/dh[t,c] and e[t,c] = Q(h[t,c]) - h[t,c] from the same forward. With mu_c = mean_t e, r = e - mu:
  V_old  = sum_c (sum_t g^2) / n * sum_t r^2     (the BC variance term)
  V_pair = sum_c sum_t g^2 r^2                     (diagnostic: the same quantities paired by position)
  T_old = V_old + B, T_pair = V_pair + B with the unchanged BC bias term B = sum_c GS_c mu_c^2;
  per channel V_pair,c - V_old,c = n Cov_t(g^2, r^2) (population covariance).
Nothing else changes (no uncentered e^2, no new bias, no range change, no new loss, no coefficient, no first-order term).
The official BC criterion, quantizer, search and knee are unchanged; the pairing score is never used by the pipeline.

Stages (one --out-dir): plan, grad, mech, freeze, data, measure, analyze, figures.
"""
import time
T_PROC = time.time()
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_olmoe_act_outlier_diag as od
import run_olmoe_dynamic_route as dr
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss
import run_olmoe_spike_intervention as si
import run_olmoe_token_spike_diag as tsd
import wt_manifest_scan as wms

HERE = Path(__file__).resolve().parent
PREV = HERE / 'runs/olmoe_spike_intervention_20261023'
TS = HERE / 'runs/olmoe_token_spike_diag_20261023'
RANGES, CALIB = ss.RANGES, ss.CALIB
ACT_CACHE, ACT_INDEX, GC, GC_INDEX = od.ACT_CACHE, od.ACT_INDEX, od.GC, od.GC_INDEX
SEQ, TP, K, NL, NE = 128, 127, 8, 16, 64
FORMATS = si.FORMATS                       # ('int4', 'p6e1')
R_SEEDS = si.R_SEEDS
KEY_CHANNELS = {'L1.E9': 915, 'L1.E11': 808, 'L1.E18': 627, 'L2.E30': 742, 'L6.E59': 750}   # from the token-spike discovery
KEY_CELLS = (('L1.E9', 'p6e1'), ('L1.E18', 'p6e1'), ('L2.E30', 'int4'))
MISMATCH_FACTOR, SHUF_N, SHUF_SEED, KEY_MIN = 1.5, 200, 23401, 2
N_ART, WPA_C = si.N_ART, si.WPA_C          # 96 x 3, as the previous round
BOOT, ALPHA = 10000, 0.05
SEEDS = {'order': 23501, 'start': 23502, 'bootstrap': 23511, 'shuffle': SHUF_SEED}
BUDGET_MIN = {'grad': 5, 'mech': 5, 'data': 5, 'measure': 25, 'analyze': 5, 'figures': 3, 'total_gpu': 35}
PRIMARY = [{'id': 'P1', 'node': 'L1.E9', 'type': 'format', 'cond': 'F', 'a': 'int4', 'b': 'p6e1'},
           {'id': 'P2', 'node': 'L1.E18', 'type': 'format', 'cond': 'F', 'a': 'int4', 'b': 'p6e1'},
           {'id': 'P3', 'node': 'L2.E30', 'type': 'cond', 'format': 'int4', 'a': 'P', 'b': 'R'},
           {'id': 'P4', 'node': 'L2.E30', 'type': 'cond', 'format': 'p6e1', 'a': 'P', 'b': 'F'}]


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_pair_gradient_diag.py', 'run_olmoe_spike_intervention.py', 'wt_manifest_scan.py',
                                                  'run_olmoe_token_spike_diag.py', 'run_olmoe_act_outlier_diag.py'] + ss.CODE + dr.SCRIPTS))}


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def comparisons(taps):
    """primary (pre-registered) and secondary comparisons; value(a) - value(b), lower NLL / lower cost is better."""
    sec = []
    prim = {(p['node'], p['type'], p.get('cond'), p.get('format'), p['a'], p['b']) for p in PRIMARY}
    for t in taps:
        n = t['node']
        cand = [{'node': n, 'type': 'format', 'cond': c, 'a': 'int4', 'b': 'p6e1'} for c in ('F', 'P')]
        for f in FORMATS:
            cand += [{'node': n, 'type': 'cond', 'format': f, 'a': 'P', 'b': 'F'}, {'node': n, 'type': 'cond', 'format': f, 'a': 'S', 'b': 'F'}]
            if t['kind'] == 'focus':
                cand.append({'node': n, 'type': 'cond', 'format': f, 'a': 'P', 'b': 'R'})
        for c in cand:
            if (c['node'], c['type'], c.get('cond'), c.get('format'), c['a'], c['b']) not in prim:
                sec.append({**c, 'id': f'S{len(sec) + 1}'})
    return PRIMARY, sec


def cell(c, side):
    """(format, condition) of side 'a' or 'b' of a comparison."""
    return (c[side], c['cond']) if c['type'] == 'format' else (c['format'], c[side])


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    taps_prev = read(PREV / 'taps.json')
    write(out / 'taps.json', taps_prev)
    taps = taps_prev['taps']
    prim, sec = comparisons(taps)
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    prev_ok = {s['content_sha256'] for s in read(PREV / 'segments.json')} <= used
    inputs = {str(p.relative_to(HERE)): sha(p) for p in [ACT_CACHE, ACT_INDEX, GC, GC_INDEX, RANGES, CALIB, PREV / 'taps.json',
                                                         PREV / 'bc_counterfactual.json', PREV / 'measure.json', PREV / 'analysis.json',
                                                         PREV / 'segments.json', TS / 'discovery_windows.json'] + mr.WT103 + mr.WT2}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'does BC\'s channel-aggregated sensitivity ignore the position correspondence of gradients and quantization errors on '
                    'spike rows, causing candidate misranking?',
        'limits': 'official BC criterion, quantizer, search and knee unchanged; no fine-tuning, no spike penalty, no first-order term; only a '
                  'position-paired diagnostic score is added; no INT4 measured NLL is used to scale BC; orderings and relative importance are '
                  'compared, the proxy is not required to equal NLL',
        'taps': {'list': [(t['node'], t['role']) for t in taps], 'source': 'runs/olmoe_spike_intervention_20261023/taps.json (copied unchanged: '
                 'thresholds, conditions, controls)', 'key_channels': KEY_CHANNELS, 'L6.E59': 'kept and interpreted separately (negative int4 '
                 'damage; BC-type scores are non-negative)'},
        'formats': list(FORMATS), 'spike_rule': 'unchanged: s_t >= 6 x discovery median s_t (thresholds in taps.json)',
        'conditions': 'F, P, S, R (3 seeds, same random-row rule and seeds as the previous round), Z; protection replaces Q(h) by h; fixed '
                      'calibration range',
        'gradients': {'data': 'the BC calibration (10 x 128, WikiText-2 validation)', 'state': 'FP32', 'loss': 'sum over windows and predicted '
                      'tokens of next-token CE (head_ce(...).sum()), as the static BC gradient cache', 'route': 'routes and routing weights '
                      'recorded in a no-grad FP32 forward and held constant in the backward (no router / top-k gradient), as the BC cache',
                      'window_accumulation': 'G2_c = sum_t g^2; GS_c = sum_window (sum_{t in window} g)^2 (window = 128-token sequence)',
                      'pairing': 'g and h (hence e) taken from the same recomputed forward of each layer; rows = tokens routed to the '
                                 'expert (torch.where order), no unexecuted positions', 'no_recompute': 'gradients are not recomputed per '
                      'condition; only the errors change'},
        'formulas': {'V_old': 'sum_c (sum_t g_tc^2) / n * sum_t r_tc^2 (= the BC variance term G2/n (S2 - n mu^2))',
                     'V_pair': 'sum_c sum_t g_tc^2 r_tc^2', 'B': 'sum_c GS_c mu_c^2 (unchanged)', 'T': 'T_old = V_old + B, T_pair = V_pair + B',
                     'identity': 'per channel V_pair,c - V_old,c = n Cov_t(g^2, r^2) (population covariance), checked numerically',
                     'conditions': 'protected rows get e = 0; mu, r, S1, S2, B recomputed; g fixed; R = mean over the 3 seeds',
                     'checked_against_implementation': 'V_old in the S2 - n mu^2 form reproduces the previous counterfactual T; the r^2 form is '
                                                       'used for the pairing and the identity'},
        'controls': ['FP32 forward of the gradient path equals the previous capture path (h bitwise) and dr.forward (CE)',
                     'G2 / GS from the per-element gradients equal the static cache (bitwise expected)',
                     'S1 / S2 from the per-element errors equal the static cache; T_old(F) reproduces the cached BC cost',
                     'T_old per condition reproduces the previous round\'s BC counterfactual', 'any mismatch is explained before interpretation'],
        'mechanism': {'data_use': 'calibration gradients and errors; the previous round\'s measured results are discovery only (already seen)',
                      'per_cell': 'for each tap and format at F: spike vs other rows (g^2 and r^2 shares and medians, covariance contribution, '
                                  'contributions to V_old and V_pair), key channel vs other channels, per-window contribution to V_pair - '
                                  'V_old and leave-one-window-out ratio, within-channel shuffle of g^2 across rows '
                                  f'({SHUF_N} permutations, seed {SHUF_SEED}) as the position-free reference',
                      'mismatch_cell': f'V_pair / V_old outside [1/{MISMATCH_FACTOR}, {MISMATCH_FACTOR}] and V_pair outside the central 95% of '
                                       'its shuffle distribution', 'mismatch_evident': f'in >= {KEY_MIN} of the key cells {list(KEY_CELLS)}',
                      'previous_data': 'agreement of T_old / T_pair signs with the previous round\'s measured comparisons is reported as '
                                       'mechanism analysis on existing data, not as validation'},
        'confirmation': {'data': f'{N_ART} new WikiText-103 train articles x {WPA_C} windows of {SEQ}, drawn after the freeze (seeds order '
                                 f'{SEEDS["order"]}, start {SEEDS["start"]}); wt_manifest_scan.used_wt103_verified: {len(used)} earlier '
                                 f'articles excluded (the previous round\'s included: {prev_ok}), WikiText-2 and near duplicates excluded',
                         'eligible_before_length_and_neardup': len(elig), 'excluded_previous_wt103': len(used),
                         'measurement': 'the previous round\'s measure stage unchanged (same taps, formats, conditions, seeds, FP32 background, '
                                        'dynamic route, bitwise controls)', 'no_refit': 'no proxy recomputation or threshold change with '
                                                                                         'confirmation data'},
        'comparisons': {'primary': prim, 'secondary': sec, 'definition': 'value(a) - value(b); measured = article-level NLL; proxy = T'},
        'statistics': {'bootstrap': f'{BOOT} article bootstraps (seed {SEEDS["bootstrap"]}); two-sided p = 2 min(P(b <= 0), P(b >= 0)), floor '
                       f'1/({BOOT}+1)', 'primary': f'Holm over the 4 primary comparisons, FWER {ALPHA}; decisive = rejected',
                       'secondary': '95% CI excluding 0, descriptive, no multiplicity control', 'agreement': 'a proxy is right on a comparison '
                       'only if the measured difference is decisive and has the proxy\'s sign; undeterminable comparisons are never counted; '
                       'comparisons of one tap are not independent successes'},
        'verdict': {'order': 'C, then D, then A, else B',
                    'C': 'no evident mismatch (rule above): the lead is not supported',
                    'D': 'T_pair fixes >= 1 primary comparison that T_old gets wrong, but breaks >= 1 primary or >= 2 decisive secondary '
                         'comparisons that T_old gets right: local explanation only',
                    'A': 'evident mismatch and S_pair > S_old on the decisive primary comparisons, with no primary break and <= 1 secondary '
                         'break: supports position aggregation as part of the cause, worth further study',
                    'B': 'evident mismatch but no improvement of the primary orderings: no criterion-relevant cause shown',
                    'always': 'no Hessian claim, no knee claim, the pipeline stays unchanged'},
        'resources': 'time, peak GPU memory and storage of the per-element statistics recorded, with an estimate for a per-channel '
                     'accumulator version', 'budget_minutes': BUDGET_MIN, 'seeds': {**SEEDS, 'random_rows': list(R_SEEDS)},
        'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info(), 'manifest_scan': rows}
    write(out / 'plan.json', p)
    md = ['# 預先登記：BC 的 channel 彙整是否忽略了 spike token 上梯度與誤差的位置對應（診斷）', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 界線',
          '- 不改正式 BC criterion、量化器、搜尋、knee；不 fine-tune、不加 spike 懲罰、不加一階項；只新增逐位置配對的診斷分數，pipeline 不使用它。',
          '- 不用 INT4 實測 NLL 換算 BC 尺度；比較排序與干預效果的相對重要性，不要求 proxy 等於 NLL。', '',
          '## tap、格式、條件（沿用上一輪，不調整）',
          f'- tap：{", ".join(f"{t[0]}（{t[1]}）" for t in p["taps"]["list"])}。關鍵 channel：{KEY_CHANNELS}。L6.E59 保留並分開解讀。',
          f'- 格式：{", ".join(FORMATS)}；spike 門檻沿用 taps.json；條件 F、P、S、R（3 seeds）、Z。', '',
          '## 梯度與誤差（BC calibration，FP32）',
          '- loss：所有窗口、所有預測 token 的 CE 加總（與 BC cache 相同的 sum-loss）；route 與權重在 no-grad forward 記錄、backward 中固定（無 router／top-k 梯度）。',
          '- G2_c = Σ_t g²，GS_c = Σ_窗口 (Σ_t g)²；g 與 h（因此 e）來自同一次 forward；只含被路由到 expert 的 rows。',
          '- 控制：梯度路徑的 h 與上一輪擷取路徑逐位元相同、CE 與 dr.forward 相同；G2／GS 重現 cache；S1／S2 與 T_old(F) 重現 cache；各條件的 T_old 重現上一輪 BC 反事實。不一致時先解釋差異。', '',
          '## 公式（只改位置配對）',
          '- r = e − μ_c；V_old = Σ_c (Σ_t g²)/n · Σ_t r²（= BC 的 G2/n·(S2 − nμ²)）；V_pair = Σ_c Σ_t g² r²；B = Σ_c GS_c μ_c²（不變）。',
          '- T_old = V_old + B；T_pair = V_pair + B；逐 channel 驗證 V_pair,c − V_old,c = n·Cov_t(g², r²)（母體共變異）。',
          '- 各條件：被保護 rows 的誤差設為 0，重算 μ、r、S1、S2、B；梯度固定；R 取 3 seeds 平均。', '',
          '## 機制分析（既有資料，只作 discovery）',
          f'- 每個 tap × 格式（F）：spike／一般 rows 的 g²、r² 占比與中位數、共變異貢獻、對 V_old／V_pair 的貢獻；關鍵 channel vs 其餘；各窗口對 V_pair − V_old 的貢獻與 leave-one-window-out；同 channel 內打亂 g² 的對照（{SHUF_N} 次，seed {SHUF_SEED}）。',
          f'- 單格「位置錯配」：V_pair/V_old 落在 [1/{MISMATCH_FACTOR}, {MISMATCH_FACTOR}] 之外，且 V_pair 落在打亂分布的中間 95% 之外。「明顯錯配」：{list(KEY_CELLS)} 中至少 {KEY_MIN} 格成立。',
          '- 與上一輪實測比較只是機制分析，不算驗證。', '',
          '## 確認（凍結後的新文章）',
          f'- {N_ART} 篇新 WT103 文章 × {WPA_C} 窗口；verified 掃描排除 {len(used)} 篇（含上一輪：{prev_ok}）、WikiText-2、近似重複；測量沿用上一輪 measure 階段，不改。',
          '- 主要比較：P1 L1.E9 F 下 int4 vs p6e1；P2 L1.E18 F 下 int4 vs p6e1；P3 L2.E30／int4 P vs R；P4 L2.E30／p6e1 P vs F。',
          f'- 主要比較用 Holm（4 個，FWER {ALPHA}），文章 bootstrap 雙尾 p；次要比較只報 95% CI。不可判定的比較不算 proxy 排對；同一 tap 的比較不是獨立成功。', '',
          '## 判讀（依序 C → D → A → B）',
          '- C：沒有明顯錯配。D：T_pair 修正至少 1 個主要比較，但破壞 ≥1 個主要或 ≥2 個決定性次要比較。A：明顯錯配且 T_pair 在決定性主要比較上對得比 T_old 多、沒有破壞主要比較、次要破壞 ≤ 1。B：其餘（有錯配但排序沒改善）。',
          '- 不論結果：不宣稱算到 Hessian、不宣稱 knee 改善、正式 pipeline 不變；記錄逐 token 統計的時間、記憶體、儲存量。', '',
          f'## 預算（分鐘）：{BUDGET_MIN}；不因結果加資料、換 seed 或改門檻。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work').mkdir(parents=True, exist_ok=True)
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; {len(prim)} primary, {len(sec)} secondary comparisons; excluded {len(used)}; previous round excluded: {prev_ok}')


# --------------------------------------------------------------------------- #
def grad(out, log):
    """per-element g and e at the Down input of every tap on the BC calibration (FP32, BC gradient semantics) + controls."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_olmoe_search_final import dev_quant
    from run_native_bc_pipeline import QUANT
    dr.gpu_setup()
    if (out / 'grad_controls.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    taps = read(out / 'taps.json')['taps']
    ranges = read(RANGES)
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    want_all = defaultdict(dict)
    for t in taps:
        want_all[t['layer']][t['expert']] = ['down_proj']
    lmin = min(want_all)
    cfg, ck = oc.config(), oc.Checkpoint()
    toks_np = np.load(CALIB)
    toks = torch.from_numpy(toks_np).to('cuda')
    cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    g0 = ck.global_('cuda', torch.float32)
    STEQ = pr.ste_fn()
    tm = {}
    # (1) the previous round's capture path (h only)
    t1 = time.perf_counter()
    ref_h = {}
    with torch.inference_mode():
        h = F.embedding(toks, g0['embed'])
        for l in range(max(want_all) + 1):
            w = ck.layer(l, 'cuda', torch.float32)
            B = h.shape[0]
            x = h + oc.attention(w, oc.rms(h, w['ln1'], cfg.rms_norm_eps), cfg, cos, sin)
            z = oc.rms(x, w['ln2'], cfg.rms_norm_eps).reshape(-1, cfg.hidden_size)
            idx, rw = oc.route(w, z, cfg)

            def cb(e, tok, kpos, gg, uu, hh, l=l):
                if e in want_all.get(l, {}):
                    ref_h[(l, e)] = (hh.clone(), tok.clone())
            h = x + tsd.experts_stats(w, z, idx, rw, {}, cb).view(B, SEQ, cfg.hidden_size)
    tm['reference_capture'] = time.perf_counter() - t1
    # (2) the BC gradient path (pathgrad semantics)
    t1 = time.perf_counter()
    routes, hs = [], []
    with torch.no_grad():
        h = F.embedding(toks, g0['embed'])
        hs.append(h)
        for l in range(NL):
            w = ck.layer(l, 'cuda', torch.float32)
            h, idx, rw = pr.layer_tap(w, h, cfg, cos, sin, None, None, {})
            routes.append((idx, rw)); hs.append(h)
    tm['forward'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    hT = hs[-1].clone().requires_grad_(True)
    ce = head_ce(g0, hT, toks, cfg)
    ce.sum().backward()
    gn = hT.grad
    cap_all = {}
    for l in reversed(range(lmin, NL)):
        w = ck.layer(l, 'cuda', torch.float32)
        want = want_all.get(l, {})
        cap = {}
        hi = hs[l].clone().requires_grad_(True)
        o = ss.layer_tap_sel(w, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(None, STEQ), want, cap)
        o.backward(gn)
        gn = hi.grad
        for e in want:
            c = cap[e]
            cap_all[(l, e)] = (c['h'].detach().clone(), c['h'].grad.detach().clone(), c['tok'].detach().clone())
        del w, cap, o
    tm['backward'] = time.perf_counter() - t1
    ref_ce = dr.forward(toks_np, None, dq, ranges, record=False, log=lambda m: None, label='dr fp32 cal')['ce']
    ctl = {'ce_vs_dr_forward_max_abs': float(np.max(np.abs(ce.detach().cpu().numpy() - np.array(ref_ce)))), 'taps': {}}
    gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
    a_, b_ = ss.COLOFF['down_x']
    t1 = time.perf_counter()
    nbytes = 0
    for t in taps:
        l, e, node = t['layer'], t['expert'], t['node']
        hh, gg, tok = cap_all[(l, e)]
        rh, rtok = ref_h[(l, e)]
        win = tok // SEQ
        wins = sorted(set(win.tolist()))
        gd = gg.double()
        G2 = (gd ** 2).sum(0)
        GS = torch.stack([gd[win == k].sum(0) for k in wins]).pow(2).sum(0)
        grow = np.asarray(gc[gidx['row'][node]])
        row, n_c = np.asarray(ac[aidx['row'][node]]), int(aidx['n'][node])
        c = {'rows': int(hh.shape[0]), 'n_cache': n_c, 'h_equal_previous_capture': bool(torch.equal(hh, rh)) and bool(torch.equal(tok, rtok)),
             'G2_equal_cache': bool(np.array_equal(G2.cpu().numpy(), grow[0, a_:b_])), 'GS_equal_cache': bool(np.array_equal(GS.cpu().numpy(), grow[1, a_:b_])),
             'formats': {}}
        rng = ranges[f'{node}.down_proj']['x']
        arrays = {'h': hh.cpu().numpy(), 'g': gg.cpu().numpy(), 'tok': tok.cpu().numpy()}
        for f in FORMATS:
            err = (dq.act(hh, f, rng) - hh).double()
            S1, S2 = err.sum(0).cpu().numpy(), err.pow(2).sum(0).cpu().numpy()
            j = QUANT.index(f)
            S1c, S2c = row[0][j][a_:b_].astype(np.float64), row[1][j][a_:b_].astype(np.float64)
            Gc2, GSc = grow[0, a_:b_].astype(np.float64), grow[1, a_:b_].astype(np.float64)
            muc = S1c / n_c
            T_cache = float((Gc2 / n_c * (S2c - n_c * muc ** 2)).sum() + (GSc * muc ** 2).sum())
            mu = S1 / c['rows']
            G2n, GSn = G2.cpu().numpy(), GS.cpu().numpy()
            T_re = float((G2n / c['rows'] * (S2 - c['rows'] * mu ** 2)).sum() + (GSn * mu ** 2).sum())
            c['formats'][f] = {'S1_equal_cache': bool(np.array_equal(S1, S1c)), 'S2_equal_cache': bool(np.array_equal(S2, S2c)),
                               'S1_maxdiff_rel': float(np.abs(S1 - S1c).max() / max(np.abs(S1c).max(), 1e-300)),
                               'S2_maxdiff_rel': float(np.abs(S2 - S2c).max() / max(np.abs(S2c).max(), 1e-300)),
                               'T_cache': T_cache, 'T_recomputed': T_re, 'T_rel': abs(T_re - T_cache) / max(abs(T_cache), 1e-300)}
            arrays[f'e_{f}'] = err.cpu().numpy()
        np.savez(out / 'work' / f'tap{t["id"]}.npz', **arrays)
        nbytes += sum(v.nbytes for v in arrays.values())
        ctl['taps'][node] = c
    tm['errors_and_controls'] = time.perf_counter() - t1
    ctl['pass'] = (ctl['ce_vs_dr_forward_max_abs'] == 0 and all(c['h_equal_previous_capture'] and c['G2_equal_cache'] and c['GS_equal_cache'] and c['rows'] == c['n_cache']
                                                              and all(v['T_rel'] < 1e-12 for v in c['formats'].values()) for c in ctl['taps'].values()))
    res = {'controls': ctl, 'semantics': {'loss': 'head_ce(...).sum() over the 10 calibration windows', 'route': 'held constant in backward',
                                          'ste': 'pr.ste_fn (identity; no QDQ in FP32)', 'rows': 'torch.where(mask[e]) order, routed rows only'},
           'resources': {'seconds': tm, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'stored_bytes': nbytes,
                         'per_element_bytes_per_tap_mean': nbytes / len(taps)}}
    write(out / 'grad_controls.json', res)
    log_stage(out, 'grad', time.perf_counter() - t0, {'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30})
    log(json.dumps(dr.clean({'pass': ctl['pass'], 'ce': ctl['ce_vs_dr_forward_max_abs'], 'taps': {k: {'h': v['h_equal_previous_capture'], 'G2': v['G2_equal_cache'],
                                                                                                        'GS': v['GS_equal_cache'], 'n': (v['rows'], v['n_cache']),
                                                                                                        **{f: (x['S1_equal_cache'], x['S2_equal_cache'], x['T_rel'])
                                                                                                           for f, x in v['formats'].items()}} for k, v in ctl['taps'].items()},
                             'seconds': tm})))


# --------------------------------------------------------------------------- #
def masks(t, h, tok):
    s = np.abs(h).max(1)
    spike = s >= t['threshold']
    win = tok // SEQ
    rm = [si.random_rows(spike, win, sd, 0, 0)[0] for sd in R_SEEDS]
    return spike, win, rm


def scores(e, g2, G2, GS, n, wins_idx):
    """T_old (both forms), T_pair and parts for one error matrix (rows x channels)."""
    S1, S2 = e.sum(0), (e ** 2).sum(0)
    mu = S1 / n
    r2 = (e - mu) ** 2
    Vo_c = G2 / n * r2.sum(0)
    Vp_c = (g2 * r2).sum(0)
    B_c = GS * mu ** 2
    T_bcform = float((G2 / n * (S2 - n * mu ** 2)).sum() + B_c.sum())
    return {'V_old': float(Vo_c.sum()), 'V_pair': float(Vp_c.sum()), 'B': float(B_c.sum()), 'T_old': float(Vo_c.sum() + B_c.sum()),
            'T_pair': float(Vp_c.sum() + B_c.sum()), 'T_old_bcform': T_bcform}, (mu, r2, Vo_c, Vp_c)


def mech(out, log):
    """mechanism analysis on the calibration gradients / errors; frozen proxy predictions for every comparison."""
    if (out / 'mech.json').exists():
        raise SystemExit('exists')
    if not read(out / 'grad_controls.json')['controls']['pass']:
        raise SystemExit('grad controls failed: explain first')
    t0 = time.perf_counter()
    taps = read(out / 'taps.json')['taps']
    prevbc = read(PREV / 'bc_counterfactual.json')
    wmeta = read(TS / 'discovery_windows.json')
    cal_title = [wmeta['cal_token_titles'][w * SEQ] for w in range(10)]
    res, proxy = {}, {}
    for t in taps:
        z = np.load(out / 'work' / f'tap{t["id"]}.npz')
        h, g, tok = z['h'], z['g'].astype(np.float64), z['tok']
        n = len(tok)
        spike, win, rm = masks(t, h, tok)
        g2 = g ** 2
        G2 = g2.sum(0)
        wins = sorted(set(win.tolist()))
        GS = np.stack([g[win == k].sum(0) for k in wins]) ** 2
        GS = GS.sum(0)
        tr = {'rows': n, 'spike_rows': int(spike.sum()), 'spike_rows_previous': prevbc[t['node']]['spike_rows'], 'formats': {}}
        kc = KEY_CHANNELS.get(t['node'])
        for fi, f in enumerate(FORMATS):
            e0 = z[f'e_{f}']
            conds = {'F': np.zeros(n, bool), 'P': spike, 'S': ~spike, **{f'R{i}': m for i, m in enumerate(rm)}}
            fr = {'conditions': {}}
            for cn, prot in conds.items():
                if cn.startswith('R') and t['kind'] != 'focus':
                    continue
                e = e0.copy()
                e[prot] = 0.0
                sc, parts = scores(e, g2, G2, GS, n, wins)
                sc['T_old_vs_previous_rel'] = abs(sc['T_old_bcform'] - prevbc[t['node']]['formats'][f][cn]['T']) / max(abs(prevbc[t['node']]['formats'][f][cn]['T']), 1e-300)
                fr['conditions'][cn] = sc
                if cn == 'F':
                    mu, r2, Vo_c, Vp_c = parts
            if t['kind'] == 'focus':
                for k_ in ('T_old', 'T_pair', 'V_old', 'V_pair', 'B'):
                    fr['conditions'].setdefault('R', {})[k_] = float(np.mean([fr['conditions'][f'R{i}'][k_] for i in range(len(R_SEEDS))]))
            # identity per channel
            cov = ((g2 - g2.mean(0)) * (r2 - r2.mean(0))).sum(0)
            d = Vp_c - Vo_c
            fr['identity_max_abs_rel'] = float(np.max(np.abs(d - cov)) / max(np.max(np.abs(Vo_c)), 1e-300))
            # spike vs other rows
            def share(x, m):
                return float(x[m].sum() / x.sum()) if x.sum() > 0 else None
            vo_rows = (G2 / n)[None, :] * r2
            vp_rows = g2 * r2
            cv_rows = (g2 - g2.mean(0)) * (r2 - r2.mean(0))
            sp = {}
            for nm, m in (('spike', spike), ('other', ~spike)):
                sp[nm] = {'rows': int(m.sum()), 'g2_share': share(g2, m), 'r2_share': share(r2, m), 'V_old_share': share(vo_rows, m), 'V_pair_share': share(vp_rows, m),
                          'cov_sum': float(cv_rows[m].sum()), 'g2_median_elem': float(np.median(g2[m])) if m.any() else None,
                          'r2_median_elem': float(np.median(r2[m])) if m.any() else None,
                          'g2_mean_row': float(g2[m].sum(1).mean()) if m.any() else None, 'r2_mean_row': float(r2[m].sum(1).mean()) if m.any() else None}
            fr['rows_split'] = sp
            fr['V_old'], fr['V_pair'] = float(Vo_c.sum()), float(Vp_c.sum())
            fr['ratio'] = fr['V_pair'] / fr['V_old'] if fr['V_old'] > 0 else None
            if kc is not None:
                oth = np.ones(g2.shape[1], bool); oth[kc] = False
                fr['key_channel'] = {'channel': kc, 'V_old': float(Vo_c[kc]), 'V_pair': float(Vp_c[kc]), 'ratio': float(Vp_c[kc] / Vo_c[kc]) if Vo_c[kc] > 0 else None,
                                     'V_old_share_of_tap': float(Vo_c[kc] / Vo_c.sum()), 'V_pair_share_of_tap': float(Vp_c[kc] / Vp_c.sum()),
                                     'spike_g2_share': share(g2[:, kc], spike), 'spike_r2_share': share(r2[:, kc], spike),
                                     'spike_g2_mean': float(g2[spike, kc].mean()) if spike.any() else None, 'other_g2_mean': float(g2[~spike, kc].mean()),
                                     'spike_r2_mean': float(r2[spike, kc].mean()) if spike.any() else None, 'other_r2_mean': float(r2[~spike, kc].mean()),
                                     'rest_V_old': float(Vo_c[oth].sum()), 'rest_V_pair': float(Vp_c[oth].sum()),
                                     'rest_ratio': float(Vp_c[oth].sum() / Vo_c[oth].sum()) if Vo_c[oth].sum() > 0 else None}
            # windows
            dwin = {int(k): float(((g2 - G2[None, :] / n) * r2)[win == k].sum()) for k in wins}
            tot = sum(abs(v) for v in dwin.values())
            loo = {}
            for k in wins:
                m = win != k
                if m.sum() < 2:
                    continue
                e = z[f'e_{f}'][m]
                nn = int(m.sum())
                mu_ = e.sum(0) / nn
                r2_ = (e - mu_) ** 2
                vo = (g2[m].sum(0) / nn * r2_.sum(0)).sum()
                vp = (g2[m] * r2_).sum()
                loo[int(k)] = float(vp / vo) if vo > 0 else None
            fr['windows'] = {'delta_by_window': dwin, 'max_abs_share': max(abs(v) for v in dwin.values()) / tot if tot > 0 else None,
                             'loo_ratio': loo, 'loo_ratio_range': [min(v for v in loo.values() if v is not None), max(v for v in loo.values() if v is not None)] if loo else None,
                             'titles': {int(k): cal_title[k] for k in wins}}
            # within-channel shuffle of g^2
            shuf = []
            for s_ in range(SHUF_N):
                rng = np.random.default_rng([SHUF_SEED, t['id'], fi, s_])
                shuf.append(float((rng.permuted(g2, axis=0) * r2).sum()))
            shuf = np.array(shuf)
            lo, hi = np.percentile(shuf, [2.5, 97.5])
            fr['shuffle'] = {'mean': float(shuf.mean()), 'p2.5': float(lo), 'p97.5': float(hi), 'V_pair_percentile': float((shuf < fr['V_pair']).mean()),
                             'ratio_to_shuffle_mean': fr['V_pair'] / float(shuf.mean()) if shuf.mean() > 0 else None}
            fr['mismatch'] = bool(fr['ratio'] is not None and (fr['ratio'] > MISMATCH_FACTOR or fr['ratio'] < 1 / MISMATCH_FACTOR)
                                  and (fr['V_pair'] < lo or fr['V_pair'] > hi))
            tr['formats'][f] = fr
            for cn, sc in fr['conditions'].items():
                proxy[(t['node'], f, cn)] = (sc['T_old'], sc['T_pair'])
        res[t['node']] = tr
        log(f'mech: {t["node"]} ({time.perf_counter() - t0:.0f}s)')
    key = {f'{n}|{f}': res[n]['formats'][f]['mismatch'] for n, f in KEY_CELLS}
    evident = sum(key.values()) >= KEY_MIN
    prim, sec = comparisons(taps)
    preds = []
    for c in prim + sec:
        fa, ca = cell(c, 'a'); fb, cb = cell(c, 'b')
        To = proxy[(c['node'], fa, ca)][0] - proxy[(c['node'], fb, cb)][0]
        Tp = proxy[(c['node'], fa, ca)][1] - proxy[(c['node'], fb, cb)][1]
        preds.append({**c, 'T_old_diff': To, 'T_pair_diff': Tp, 'old_says_a_better': To < 0, 'pair_says_a_better': Tp < 0})
    prev = previous_measured(taps, prim + sec)
    for p_ in preds:
        m = prev[p_['id']]
        p_['previous'] = m
        if m['decisive']:
            p_['previous_old_right'] = (m['diff'] < 0) == p_['old_says_a_better']
            p_['previous_pair_right'] = (m['diff'] < 0) == p_['pair_says_a_better']
    write(out / 'mech.json', {'taps': res, 'key_cells': key, 'mismatch_evident': evident})
    write(out / 'predictions.json', {'note': 'frozen before any confirmation article; previous-round agreement is mechanism analysis on seen data',
                                     'comparisons': preds})
    log_stage(out, 'mech', time.perf_counter() - t0)
    log(json.dumps(dr.clean({'key_cells': key, 'evident': evident, 'ratios': {f'{n}|{f}': res[n]['formats'][f]['ratio'] for n in res for f in FORMATS},
                             'previous_agreement': {'old': sum(p_.get('previous_old_right', False) for p_ in preds),
                                                    'pair': sum(p_.get('previous_pair_right', False) for p_ in preds),
                                                    'decisive': sum(p_['previous']['decisive'] for p_ in preds)}})))


def article_vec(ce):
    return np.array(ce).reshape(-1, WPA_C).sum(1) / (WPA_C * TP)


def cond_vec(m, tap_id, f, cond):
    if cond == 'R':
        return np.mean([article_vec(m['variants'][f'tap{tap_id}|{f}|R{i}']) for i in range(len(R_SEEDS))], 0)
    return article_vec(m['variants'][f'tap{tap_id}|{f}|{cond}'])


def measured(m, taps, comps, seed):
    na = len(m['base_ce']) // WPA_C
    idx = np.random.default_rng(seed).integers(0, na, (BOOT, na))
    byn = {t['node']: t for t in taps}
    res = {}
    for c in comps:
        t = byn[c['node']]
        fa, ca = cell(c, 'a'); fb, cb = cell(c, 'b')
        d = cond_vec(m, t['id'], fa, ca) - cond_vec(m, t['id'], fb, cb)
        b = d[idx].mean(1)
        ci = np.percentile(b, [2.5, 97.5]).tolist()
        res[c['id']] = {'diff': float(d.mean()), 'ci95': ci, 'p': si.pval(b), 'decisive': bool(ci[0] > 0 or ci[1] < 0)}
    return res


def previous_measured(taps, comps):
    """the previous round's measured comparisons (its own article bootstrap seed); discovery only."""
    return measured(read(PREV / 'measure.json'), taps, comps, si.SEEDS['bootstrap'])


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    files = ('plan.json', 'taps.json', 'grad_controls.json', 'mech.json', 'predictions.json') + tuple(f'work/tap{i}.npz' for i in range(len(read(out / 'taps.json')['taps'])))
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'code': code_hashes(),
                                'note': 'diagnostic formula, mechanism tables and proxy predictions frozen before any confirmation article is drawn'})
    log('frozen')


def data(out, log):
    """the previous round's data stage with this round's seeds and plan keys (verified manifest scan)."""
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')['confirmation']
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    if len(used) != pl['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    lines = mr.load_lines(mr.WT103)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    prev = [a for a in base if a['ch'] in used]
    elig = [a for a in base if a['ch'] not in used]
    if len(elig) != pl['eligible_before_length_and_neardup']:
        raise SystemExit('eligible count changed')
    by = {a['ch']: a for a in w103}
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    prev_sh = [mr.shingles(by[h]['body_norm']) for h in used]
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
                     'source': 'wikitext-103-raw-v1 train', 'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca},
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


def measure(out, log):
    si.measure(out, log)


# --------------------------------------------------------------------------- #
def analyze(out, log):
    t0 = time.perf_counter()
    taps = read(out / 'taps.json')['taps']
    m = read(out / 'measure.json')
    preds = read(out / 'predictions.json')['comparisons']
    mech_ = read(out / 'mech.json')
    meas = measured(m, taps, preds, SEEDS['bootstrap'])
    prim = [p for p in preds if p['id'].startswith('P')]
    rej = si.holm([meas[p['id']]['p'] for p in prim])
    rows = []
    for p in preds:
        mm = meas[p['id']]
        dec = rej[prim.index(p)] if p['id'].startswith('P') else mm['decisive']
        r = {**{k: p[k] for k in ('id', 'node', 'type', 'a', 'b')}, 'cond': p.get('cond'), 'format': p.get('format'), 'T_old_diff': p['T_old_diff'],
             'T_pair_diff': p['T_pair_diff'], 'measured': mm, 'decisive': bool(dec), 'previous': p['previous']}
        if dec:
            r['old_right'] = (mm['diff'] < 0) == p['old_says_a_better']
            r['pair_right'] = (mm['diff'] < 0) == p['pair_says_a_better']
        rows.append(r)
    P = [r for r in rows if r['id'].startswith('P')]
    S = [r for r in rows if r['id'].startswith('S')]
    S_old = sum(r.get('old_right', False) for r in P)
    S_pair = sum(r.get('pair_right', False) for r in P)
    fixes = [r['id'] for r in P if r['decisive'] and r['pair_right'] and not r['old_right']]
    pbreak = [r['id'] for r in P if r['decisive'] and r['old_right'] and not r['pair_right']]
    sbreak = [r['id'] for r in S if r['decisive'] and r['old_right'] and not r['pair_right']]
    sfix = [r['id'] for r in S if r['decisive'] and r['pair_right'] and not r['old_right']]
    if not mech_['mismatch_evident']:
        verdict = 'C'
    elif fixes and (pbreak or len(sbreak) >= 2):
        verdict = 'D'
    elif S_pair > S_old and not pbreak and len(sbreak) <= 1:
        verdict = 'A'
    else:
        verdict = 'B'
    l6 = [r for r in rows if r['node'] == 'L6.E59']
    res = {'rows': rows, 'primary_holm_reject': rej, 'S_old': S_old, 'S_pair': S_pair, 'primary_fixes': fixes, 'primary_breaks': pbreak,
           'secondary_breaks': sbreak, 'secondary_fixes': sfix, 'secondary_decisive': sum(r['decisive'] for r in S),
           'secondary_old_right': sum(r.get('old_right', False) for r in S), 'secondary_pair_right': sum(r.get('pair_right', False) for r in S),
           'mismatch_evident': mech_['mismatch_evident'], 'key_cells': mech_['key_cells'], 'verdict': verdict,
           'L6.E59_separate': [{k: r[k] for k in ('id', 'type', 'a', 'b', 'cond', 'format', 'T_old_diff', 'T_pair_diff', 'measured', 'decisive')} | {'old_right': r.get('old_right'), 'pair_right': r.get('pair_right')} for r in l6],
           'measure_controls': m['controls'], 'stopped_early': m['stopped_early']}
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    res['timing'] = read(out / 'timing.json')
    write(out / 'analysis.json', res)
    with open(out / 'comparisons.csv', 'w') as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(['id', 'node', 'type', 'cond', 'format', 'a', 'b', 'T_old_diff', 'T_pair_diff', 'diff', 'ci95_lo', 'ci95_hi', 'p', 'decisive', 'old_right', 'pair_right',
                       'prev_diff', 'prev_decisive'])
        for r in rows:
            wcsv.writerow([r['id'], r['node'], r['type'], r['cond'], r['format'], r['a'], r['b'], r['T_old_diff'], r['T_pair_diff'], r['measured']['diff'],
                           r['measured']['ci95'][0], r['measured']['ci95'][1], r['measured']['p'], r['decisive'], r.get('old_right'), r.get('pair_right'),
                           r['previous']['diff'], r['previous']['decisive']])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps(dr.clean({k: res[k] for k in ('S_old', 'S_pair', 'primary_fixes', 'primary_breaks', 'secondary_breaks', 'secondary_fixes', 'secondary_decisive',
                                                 'secondary_old_right', 'secondary_pair_right', 'mismatch_evident', 'verdict')}
                            | {'primary': [(r['id'], r['measured']['diff'], r['measured']['ci95'], r['decisive'], r.get('old_right'), r.get('pair_right')) for r in P]})))


PAL = tsd.PAL


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
    mech_ = read(out / 'mech.json')
    taps = read(out / 'taps.json')['taps']
    # 1 V_pair / V_old with the shuffle interval
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels, x = [], 0
    for t in taps:
        for f in FORMATS:
            fr = mech_['taps'][t['node']]['formats'][f]
            sh = fr['shuffle']
            lo, hi = sh['p2.5'] / fr['V_old'], sh['p97.5'] / fr['V_old']
            col = PAL['s1'] if f == 'int4' else PAL['s2']
            ax.plot([x, x], [lo, hi], color=PAL['muted'], lw=6, solid_capstyle='butt')
            ax.scatter([x], [fr['ratio']], color=col, s=30, zorder=3, edgecolors=PAL['surface'], linewidths=1)
            labels.append(f'{t["node"]} {f}'); x += 1
    ax.axhline(1, color=PAL['text2'], lw=0.8)
    for y in (MISMATCH_FACTOR, 1 / MISMATCH_FACTOR):
        ax.axhline(y, color=PAL['text2'], lw=0.6, ls='--')
    ax.set_yscale('log'); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('V_pair / V_old (log)'); ax.set_title('Position pairing vs channel aggregation (F, calibration); grey = shuffle 95%', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(fd / 'pair_ratio.png', dpi=150); plt.close(fig)
    # 2 spike vs other: g2 share and r2 share for focus taps
    fr_ = [(t['node'], f, mech_['taps'][t['node']]['formats'][f]['rows_split']) for t in taps if t['kind'] == 'focus' for f in FORMATS]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(len(fr_))
    ax.bar(xs - 0.27, [s['spike']['g2_share'] for _, _, s in fr_], 0.26, color=PAL['s1'], label='spike rows: share of sum g^2')
    ax.bar(xs, [s['spike']['r2_share'] for _, _, s in fr_], 0.26, color=PAL['s2'], label='spike rows: share of sum r^2')
    ax.bar(xs + 0.27, [s['spike']['V_pair_share'] for _, _, s in fr_], 0.26, color=PAL['s3'], label='spike rows: share of V_pair')
    ax.set_xticks(xs); ax.set_xticklabels([f'{n} {f}' for n, f, _ in fr_], rotation=45, ha='right', fontsize=7)
    ax.set_title('Where gradients and errors sit (calibration, F)', loc='left', fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fd / 'spike_shares.png', dpi=150); plt.close(fig)
    log_stage(out, 'figures', time.perf_counter() - t0)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'grad', 'mech', 'freeze', 'data', 'measure', 'analyze', 'figures'], required=True)
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
