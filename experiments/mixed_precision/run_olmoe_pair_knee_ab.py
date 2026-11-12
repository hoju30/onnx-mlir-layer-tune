#!/usr/bin/env python3
"""OLMoE whole-configuration A/B at the normal knee: original BC (A) vs BC with the Down-input activation variance term paired by
position (B). Pre-registered in plan.json / plan.md; the default pipeline keeps the original BC.

A: the normal knee configuration (q0_knee) and the static BC table.
B: the same table except, for every searched expert's Down input (down_proj x slot), the variance term
     V_old = sum_c (sum_t g^2) / n * sum_t (e - mu_c)^2     ->     V_pair = sum_c sum_t g^2 (e - mu_c)^2
   computed with streaming sums  V_pair,c = sum g^2 e^2 - 2 mu_c sum g^2 e + mu_c^2 sum g^2. Bias term, Down output term, Gate / Up terms
   and the shared-input attribution, all weight rows, quantizer, ranges, calibration, resource model, legality and searched nodes
   unchanged; full candidate grid; FP32 stays at cost 0. One W+A search at A's exact knee limits; no new frontier, no new knee.

Stages (one --out-dir): plan, build (--variant A | B | Binc, --rep), search, configs, freeze, data, eval, analyze, figures.
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

import run_olmoe_act_outlier_diag as od
import run_olmoe_dynamic_route as dr
import run_olmoe_knee_selective_sensitivity as ks
import run_olmoe_proxy_refresh as pr
import run_olmoe_route_mass_restore as mr
import run_olmoe_selective_sensitivity as ss
import run_olmoe_token_spike_diag as tsd
import wt_manifest_scan as wms

HERE = Path(__file__).resolve().parent
KS = ks.HERE / 'runs/olmoe_knee_selective_sensitivity_20261022'
KW = HERE / 'runs/olmoe_knee_weight_sensitivity_20261023'
TS = HERE / 'runs/olmoe_token_spike_diag_20261023'
RANGES, CALIB, MP = ss.RANGES, ss.CALIB, ss.MP
STATIC_TABLE, ACT_CACHE, ACT_INDEX = ks.STATIC_TABLE, ks.ACT_CACHE, ks.ACT_INDEX
GC, GC_INDEX = od.GC, od.GC_INDEX
A_CONFIG, KNEE = KS / 'configs/A.json', KS / 'knee/knee.json'
SEQ, TP, K, NL, NE, WPA = 128, 127, 8, 16, 64, 4
N, CTRL = 600, 128
BOOT, TOL_REL = 10000, 1e-12
SEEDS = {'order': 23701, 'start': 23702, 'bootstrap': 23711}
VALIDATE = ('L1.E9', 'L2.E30', 'L9.E8', 'L0.E28', 'L15.E36')      # pre-specified direct per-element checks of the streaming sums
BUILD_SCHEDULE = [('A', 1), ('B', 1), ('Binc', 1), ('A', 2), ('B', 2), ('Binc', 2)]
FRONT, BACK = (0, 1), (14, 15)
BUDGET_MIN = {'builds': 12, 'search': 20, 'data': 6, 'eval': 20, 'analyze': 5, 'total_gpu': 50}
CATEGORIES = {'load_and_prep': ('setup', 'setup_inputs', 'ckpt_load', 'quant_init', 'cache_read', 'input_read'),
              'forward_routing': ('forward_routing',), 'backward': ('backward',), 'weight_sensitivity': ('weight_sens',),
              'weight_error': ('weight_error',), 'act_grad_stats': ('act_grad_stats',), 'act_error': ('act_error',),
              'pair_stats': ('pair_stats',), 'assembly': ('table_assembly',), 'table_write': ('table_write',), 'controls': ('controls',)}


def read(p):
    return dr.read(p)


def write(p, obj):
    dr.write(p, obj)


def sha(p):
    return dr.sha(p)


def code_hashes():
    return {f: sha(HERE / f) for f in sorted(set(['run_olmoe_pair_knee_ab.py', 'run_olmoe_knee_error_only.py', 'run_olmoe_knee_selective_sensitivity.py',
                                                  'wt_manifest_scan.py', 'run_olmoe_search_final.py', 'run_olmoe_multipoint.py', 'run_solver_regression.py']
                                                 + ss.CODE + dr.SCRIPTS))}


def merge_timing(out, key, obj):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[key] = obj
    write(f, t)


def run_record(tm, t0, extra):
    import torch
    total = time.perf_counter() - t0
    timers = dict(tm.t)
    cats = {c: sum(timers.get(k, 0.0) for k in ks_) for c, ks_ in CATEGORIES.items()}
    cats['unclassified'] = total - sum(timers.values())
    return {'total_s': total, 'method_s': total - cats['controls'], 'timers': timers, 'categories': cats, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
            'peak_cpu_gb': ss.peak_cpu_gb(), 'gpu': torch.cuda.get_device_name(0), 'loadavg_before': os.getloadavg(), **extra}


# --------------------------------------------------------------------------- #
def plan(out, log):
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    t0 = time.perf_counter()
    from run_native_bc_pipeline import GRID11, QUANT
    from run_olmoe_search_final import REL_TOL, ABS_TOL, TIME_LIMIT
    names, nodes, elems, flops, rows = pr.problem_static()
    lim = read(KNEE)['limits']
    w103, wt2 = wms.corpora()
    used, mrows = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    downs = [n for n in names if n.endswith('down_proj')]
    files = [STATIC_TABLE, A_CONFIG, KNEE, KS / 'knee/q0_knee.json', ACT_CACHE, ACT_INDEX, GC, GC_INDEX, RANGES, CALIB, KW / 'tokens.npy', KW / 'eval.json',
             TS / 'discovery_experts.csv'] + mr.WT103 + mr.WT2
    inputs = {str(p.relative_to(HERE)): sha(p) for p in files}
    mfiles = {f.name: sha(f) for f in ss.model_files()}
    p = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'does replacing only the Down-input activation variance term of BC by its position-paired form change the knee configuration '
                    'and turn into a whole-configuration NLL / PPL gain, and what is the full offline cost',
        'default': 'the official pipeline keeps the original BC; this is a separate experimental branch; no fine-tuning, no proxy refresh',
        'A': {'config': str(A_CONFIG.relative_to(HERE)), 'config_sha256': sha(A_CONFIG), 'table': str(STATIC_TABLE.relative_to(HERE)),
              'table_sha256': sha(STATIC_TABLE)},
        'B': {'replaced': f'for every searched expert ({len(downs)} down_proj projections) the Down-input (down_proj x slot, h = SiLU(g) u) '
                          'activation variance term: V_old = sum_c G2_c / n sum_t r^2 -> V_pair = sum_c sum_t g^2 r^2, r = e - mu_c',
              'unchanged': 'bias term sum_c GS_c mu_c^2 of the same slot; Down output (y) term; Gate / Up x and y terms and the shared-input '
                           'attribution; every weight row; quantizer, ranges, calibration; resource model, legality, searched nodes',
              'grid': f'the full candidate grid {list(QUANT)} (+ fp32 at cost 0), activation and weight grids GRID11 = {list(GRID11)}',
              'uniform': 'applied to all searched experts\' Down inputs; no spike-only rule, no expert chosen from misranks or loss',
              'streaming': 'per expert and format: sum_t g^2 e^2, sum_t g^2 e with G2 = sum_t g^2 and mu_c = S1_c / n from the same rows; '
                           'V_pair,c = sum g^2 e^2 - 2 mu_c sum g^2 e + mu_c^2 G2_c; no per-token tensors stored',
              'numerics': 'float64 accumulation; no clamping or other protection; negative V_pair,c (only possible by rounding) counted and '
                          'reported with their magnitude; a negative total stops the run', 'validation': f'direct per-element sum_t g^2 (e - mu)^2 for '
                          f'{list(VALIDATE)} (pre-specified) against the streaming value'},
        'gradient_semantics': 'as the static BC: FP32 calibration (10 x 128), sum-loss next-token CE, routes and routing weights recorded in a '
                              'no-grad forward and constant in backward (no router / top-k gradient), STE identity for QDQ (FP32: none); g, h, e '
                              'from the same recomputed layer forward',
        'limits': {'source': str(KNEE.relative_to(HERE)), 'max_bytes': lim['max_bytes'], 'max_bops': lim['max_bops'], 'exact': 'read from the file, '
                   'not rounded'},
        'search': {'method': f'ss.search_table: dominance pruning, then make_solver("lpfirst"): LP, round + repair, accepted if objective - LP '
                             f'bound <= max({REL_TOL} x objective, {ABS_TOL} x scale), else corrected MILP (mip_rel_gap 0, time limit {TIME_LIMIT} '
                             's, 1-ppm retry, repair); identical for A and B', 'A_control': 'searching the rebuilt A table at the limits must '
                             'reproduce q0_knee exactly (config hash); otherwise investigate before any comparison',
                   'approximation_rule': 'if a configuration known to be feasible at the limits (A) has a lower objective on the B table than B\'s '
                                         'solution, B is re-solved once with the corrected MILP (same time limit); the lower-objective feasible '
                                         'solution on the B table is used and both are reported; if B still scores worse than A on the B table '
                                         'the comparison is labelled search-limited; the A configuration is never substituted for B; no '
                                         'evaluation NLL is used', 'no_new_knee': 'one search per table at the fixed limits; no frontier, no auto '
                                                                                  'knee'},
        'controls': ['the A rebuild reproduces the static table bitwise; the A search reproduces q0_knee', 'G2 / GS, S1 / S2 and n equal the '
                     'static caches (Binc)', 'B weight rows bitwise equal A', 'B activation rows of non-Down projections bitwise equal A', 'per Down '
                     'projection: A row = vx_old + vy and B row = vx_pair + vy with the input / output terms recorded separately',
                     'finite, legal, all searched Down projections covered', 'Binc (incremental) equals the full B build', 'lower candidate cost is '
                     'not quality: no conclusion from cost tables'],
        'configs_report': 'changed projections (weight only / activation only / both), format and bit distributions, INT / Posit / FP families, '
                          f'front {FRONT} / middle / back {BACK}, strong-spike (R_token >= p90 of the token-spike discovery, rows >= 512) vs other '
                          'experts, A\'s FP16 activations, resource use; if A and B are identical, conclude "configuration unchanged at this '
                          'resource point" without a full evaluation',
        'evaluation': {'data': f'{N} new WikiText-103 train articles x {WPA} windows of {SEQ}, drawn after the freeze (seeds order {SEEDS["order"]}, '
                               f'start {SEEDS["start"]}); wt_manifest_scan.used_wt103_verified excludes {len(used)} earlier articles; WikiText-2 '
                               '(calibration) and near duplicates excluded; content-hash intersection checked to be empty',
                       'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                       'conditions': 'FP32, A, B on the same articles; original checkpoint and router; each its own dynamic route; same backend',
                       'primary': 'Delta = NLL(B) - NLL(A), article-level paired bootstrap (10000, seed '
                                  f'{SEEDS["bootstrap"]}) 95% CI; the only primary comparison', 'secondary': 'NLL, PPL (token-weighted), damage '
                       'vs FP32, better / worse articles, relative damage change, routed BOPs proxy and unsearched-expert use (diagnostic)',
                       'decision': {'CI < 0': 'quality gain at this resource point confirmed; weigh size vs extra offline cost; next: other points '
                                              'and the auto knee; no default change', 'CI contains 0': 'gain not confirmed; keep BC; no "not '
                                              'worse" argument for adoption', 'CI > 0': 'do not adopt; examine allocation vs local scores; no '
                                              're-report after configuration fixes', 'unfair': 'resource or search unfairness -> comparison limited'},
                       'controls': 'A program control on 128 windows of the weight-sensitivity round (bitwise vs its record); rerun of B on '
                                   'the first 128 windows (bitwise)'},
        'costs': {'separate': ['original BC statistics (A build)', 'added pairing statistics (pair_stats timer; B - A)', 'table assembly and '
                               'storage', 'pruning and search', 'input -> final configuration (sum of measured stages; labelled)', 'final '
                               'NLL evaluation (separate)'], 'repeats': f'builds interleaved {BUILD_SCHEDULE}, one process each, same cache '
                  'conditions (no page-cache dropping)', 'incremental_vs_scratch': 'Binc reuses the static table and caches (forward + backward + '
                  'Down-input pairing only); B builds the whole table from the checkpoint', 'not_claimed': 'no ONNX-MLIR native integration, '
                  'no low-bit kernel speed from simulated inference'},
        'no_feedback': 'no budget, candidate, configuration or data change after evaluation; no extra articles',
        'budget_minutes': BUDGET_MIN, 'seeds': SEEDS, 'inputs': inputs, 'model_files': mfiles, 'code_hashes': code_hashes(), 'environment': dr.env_info(),
        'manifest_scan': mrows}
    write(out / 'plan.json', p)
    md = ['# 預先登記：OLMoE 完整配置 A／B — 原 BC vs 只替換 Down 輸入 activation 的逐位置配對評分', '', f'時間：{p["time"]}。以 plan.json 為準。', '',
          '## 定義',
          f'- A：正常 knee 配置（{A_CONFIG.relative_to(HERE)}）與原 static BC 表。',
          f'- B：所有 {len(downs)} 個搜尋 expert 的 Down 輸入（down_proj x）activation 變異項由 Σ_c G2_c/n·Σ_t r² 改為 Σ_c Σ_t g²r²；偏差項、Down 輸出項、Gate／Up 項與共享輸入歸屬、全部 weight 列、量化器、range、calibration、資源模型、合法性、搜尋節點都不變；完整候選網格；FP32 成本 0。統一套用，不選 spike。',
          '- streaming：Σg²e²、Σg²e 與 G2、μ（S1/n）同 rows；V_pair,c = Σg²e² − 2μΣg²e + μ²G2；float64；不做任何截斷；負值只計數報告，總和為負則停止。預先指定直接逐元素驗證：' + ', '.join(VALIDATE) + '。', '',
          '## 搜尋',
          f'- A 的精確上限：max_bytes = {lim["max_bytes"]!r}，max_bops = {lim["max_bops"]!r}（直接讀檔）。',
          f'- 兩張表用同一個 ss.search_table（支配剪枝 → LP → 四捨五入＋修復，誤差 ≤ max({REL_TOL}·目標, {ABS_TOL}·scale) 才接受，否則修正版 MILP，時限 {TIME_LIMIT} s）；只搜一次，不建 frontier、不重選 knee。',
          '- A 控制：重建的 A 表必須逐位元等於 static 表；A 表的搜尋必須重現 q0_knee。',
          '- 近似處理：若 A 配置在 B 表上的目標值低於 B 的解，B 用修正版 MILP 重解一次，取 B 表上較低者並同時報告；仍高於 A 則標示「搜尋受限」。永不以 A 配置代替 B；不使用任何評估 NLL。', '',
          '## 控制',
          '- G2／GS、S1／S2、n 與 cache 一致；B 的 weight 列與非 Down activation 列逐位元等於 A；Down 列拆成輸入／輸出項分別記錄；有限、合法、覆蓋全部；增量版 Binc 等於完整 B。', '',
          '## 評估（配置凍結後才抽資料）',
          f'- {N} 篇新 WT103 文章 × {WPA} 窗口；verified 掃描排除 {len(used)} 篇；排除 WikiText-2 與近似重複；核對 content hash 交集為 0。',
          '- FP32、A、B 同一批文章、原 router、dynamic route、同一量化後端。',
          '- 主要：Δ = NLL(B) − NLL(A)，文章 bootstrap 95% CI；唯一的主要比較。次要：NLL、PPL、相對 FP32 損傷、改善／變差文章數、相對損傷改善、路由 BOPs 與未搜尋 expert 用量。',
          '- 決策：CI < 0 → 此資源點品質收益確認（權衡額外 offline 成本，下一輪再驗其他點與自動 knee，不改預設）；CI 含 0 → 未確認，保留 BC；CI > 0 → 不接入；資源或搜尋不公平 → 比較受限。', '',
          '## 成本',
          f'- 分開量原 BC 統計、配對統計、組表與儲存、剪枝與搜尋、輸入到配置的總時間、最終評估；建表交錯 {" ".join(f"{v}{r}" for v, r in BUILD_SCHEDULE)}；Binc 是重用 cache 的增量成本，B 是從頭建表成本。不宣稱 ONNX-MLIR native 或低位元 kernel 加速。', '',
          f'## 預算（分鐘）：{BUDGET_MIN}；看到評估結果後不改預算、候選、配置或資料。', '']
    (out / 'plan.md').write_text('\n'.join(md))
    write(out / 'input_hashes_before.json', {'inputs': inputs, 'model_files': mfiles})
    (out / 'work/runs').mkdir(parents=True, exist_ok=True)
    merge_timing(out, 'plan', {'seconds': time.perf_counter() - t0})
    log(f'plan written; {len(downs)} Down projections; limits {lim["max_bytes"]!r} / {lim["max_bops"]!r}; excluded {len(used)}')


# --------------------------------------------------------------------------- #
def build_full(dq, ranges, names, tm, log, pair):
    """run_olmoe_knee_error_only.build_static (same op order and table arithmetic); with pair=True the Down-input paired term is also
    computed (separate 'pair_stats' block, own QDQ pass) and a B table assembled."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    TAPS = ss.TAPS
    with tm.block('setup'):
        cfg, ck = oc.config(), oc.Checkpoint()
        toks_np = np.load(CALIB)
        toks = torch.from_numpy(toks_np).to('cuda')
        cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    with tm.block('ckpt_load'):
        g = ck.global_('cuda', torch.float32)
    want_all = defaultdict(dict)
    for n in names:
        l, e, p = n.split('.')
        want_all[int(l[1:])].setdefault(int(e[1:]), []).append(p)
    STEQ = pr.ste_fn()
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            wq, hk = ss.load_layer(ck, l, None, dq, ranges, tm)
            with tm.block('forward_routing'):
                h, idx, rw = pr.layer_tap(wq, h, cfg, cos, sin, None, None, pr.act_map(hk))
            routes.append((idx, rw)); hs.append(h)
            del wq, hk
        with tm.block('forward_routing'):
            ce_fwd = head_ce(g, h, toks, cfg).cpu().numpy().tolist()
    with tm.block('backward'):
        hT = hs[-1].clone().requires_grad_(True)
        ce_bwd = head_ce(g, hT, toks, cfg)
        ce_bwd.sum().backward()
        gn = hT.grad
    B_, C_, CB, meta, terms, val, negs = {}, {}, {}, {}, {}, {}, {'channels': 0, 'min': 0.0, 'neg_total': 0}
    gfin = True
    for l in reversed(range(NL)):
        wq, hk = ss.load_layer(ck, l, None, dq, ranges, tm)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = ss.layer_tap_sel(wq, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(hk, STEQ), want, cap)
            o.backward(gn)
            gn = hi.grad
        with tm.block('controls'):
            gfin &= bool(torch.isfinite(gn).all())
        with torch.no_grad():
            for e in sorted(want):
                node = f'L{l}.E{e}'
                c = cap[e]
                tok = c['tok']
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                meta[node] = {'routed_rows': int(n), 'windows': len(wins)}
                gst, errm = {}, {}
                for p in want[e]:
                    name = f'{node}.{p}'
                    kX, kG, kx, ky = TAPS[p]
                    with tm.block('weight_sens'):
                        GY, X = c[kG].grad, c[kX].detach()
                        Gc = torch.zeros(GY.shape[1], dtype=torch.float64, device='cuda')
                        for k in wins:
                            m = win == k
                            Gc += ((GY[m].double().t() @ X[m].double()) ** 2).sum(1)
                        gcn = Gc.cpu().numpy()
                        gfin &= bool(np.isfinite(gcn).all())
                    with tm.block('weight_error'):
                        W = wq[p][e]
                        dd = {f: ((dq.weight(W, f) - W).double() ** 2).sum(1).cpu().numpy() for f in QUANT}
                    with tm.block('table_assembly'):
                        B_[name] = {'fp32': 0.0}
                        for f in QUANT:
                            B_[name][f] = float((gcn / X.shape[1] * dd[f]).sum())
                    with tm.block('act_grad_stats'):
                        for k_ in (kx, ky):
                            if k_ not in gst:
                                gt = c[k_].grad.double()
                                gst[k_] = ((gt ** 2).sum(0), torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0))
                    with tm.block('controls'):
                        gfin &= all(bool(torch.isfinite(v).all()) for k_ in (kx, ky) for v in gst[k_])
                    rng = ranges[name]
                    with tm.block('act_error'):
                        for s, k_ in (('x', kx), ('y', ky)):
                            key = (k_, tuple(rng[s]))
                            if key in errm:
                                continue
                            t = c[k_].detach()
                            G2, GS = gst[k_]
                            vals = []
                            for f in QUANT:
                                err = (dq.act(t, f, rng[s]) - t).double()
                                s1 = err.sum(0)
                                s2 = (err ** 2).sum(0)
                                mu = s1 / n
                                vals.append(float((G2 / n * (s2 - n * mu ** 2)).sum() + (GS * mu ** 2).sum()))
                            errm[key] = vals
                        vx, vy = errm[(kx, tuple(rng['x']))], errm[(ky, tuple(rng['y']))]
                        C_[name] = {'fp32': 0.0}
                        for i, f in enumerate(QUANT):
                            cc = 0.0
                            cc += vx[i]
                            cc += vy[i]
                            C_[name][f] = cc
                    if pair and p == 'down_proj':
                        vxp = pair_terms(c[kx].detach(), c[kx].grad, gst[kx], rng['x'], n, dq, QUANT, tm, negs)
                        CB[name] = {'fp32': 0.0}
                        for i, f in enumerate(QUANT):
                            cc = 0.0
                            cc += vxp[i]
                            cc += vy[i]
                            CB[name][f] = cc
                        terms[name] = {'vx_old': vx, 'vy': vy, 'vx_pair': vxp}
                        if node in VALIDATE:
                            with tm.block('controls'):
                                val[node] = direct_check(c[kx].detach(), c[kx].grad, gst[kx], rng['x'], n, dq, QUANT, vxp)
        del wq, hk, cap
        log(f'build: layer {l} ({sum(tm.t.values()):.0f}s)')
    tab = {'B': B_, 'C': C_, 'meta': meta}
    tabB = None
    if pair:
        tabB = {'B': B_, 'C': {n: (CB[n] if n in CB else C_[n]) for n in names}, 'meta': meta}
    nonfinite = [(n, f) for t in ([tab] + ([tabB] if tabB else [])) for part in ('B', 'C') for n, r in t[part].items() for f, v in r.items() if not math.isfinite(v)]
    return {'A_table': tab, 'B_table': tabB, 'terms': terms, 'validation': val, 'negative_channels': negs, 'ce_forward': ce_fwd, 'grads_finite': gfin,
            'nonfinite': nonfinite}


def pair_terms(t, gx, gstat, rng, n, dq, QUANT, tm, negs):
    """streaming paired Down-input term per format: sum_c [sum g^2 e^2 - 2 mu sum g^2 e + mu^2 G2] + sum_c GS mu^2."""
    with tm.block('pair_stats'):
        g2 = gx.double() ** 2
        G2, GS = gstat
        vals = []
        for f in QUANT:
            err = (dq.act(t, f, rng) - t).double()
            mu = err.sum(0) / n
            Vc = (g2 * err * err).sum(0) - 2 * mu * (g2 * err).sum(0) + mu ** 2 * G2
            neg = Vc < 0
            if bool(neg.any()):
                negs['channels'] += int(neg.sum()); negs['min'] = min(negs['min'], float(Vc.min()))
            tot = float(Vc.sum())
            if tot < 0:
                negs['neg_total'] += 1
            vals.append(tot + float((GS * mu ** 2).sum()))
    return vals


def direct_check(t, gx, gstat, rng, n, dq, QUANT, vxp):
    g2 = gx.double() ** 2
    G2, GS = gstat
    res = {}
    for i, f in enumerate(QUANT):
        err = (dq.act(t, f, rng) - t).double()
        mu = err.sum(0) / n
        Vd = float((g2 * (err - mu) ** 2).sum()) + float((GS * mu ** 2).sum())
        res[f] = {'direct': Vd, 'streaming': vxp[i], 'rel': abs(Vd - vxp[i]) / max(abs(Vd), 1e-300)}
    return res


def build_incremental(dq, ranges, names, tm, log, static_tab):
    """reuse the static table and caches: one FP32 forward + backward retaining only the Down inputs; Down-input paired term (streaming);
    Down output term from the static caches; B table assembled from the static table."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    from run_native_bc_pipeline import QUANT
    with tm.block('setup'):
        cfg, ck = oc.config(), oc.Checkpoint()
        toks = torch.from_numpy(np.load(CALIB)).to('cuda')
        cos, sin = oc.rope(cfg, SEQ, 'cuda', torch.float32)
    with tm.block('ckpt_load'):
        g = ck.global_('cuda', torch.float32)
    with tm.block('cache_read'):
        ac, aidx = np.load(ACT_CACHE, mmap_mode='r'), read(ACT_INDEX)
        gc, gidx = np.load(GC, mmap_mode='r'), read(GC_INDEX)
    downs = [n for n in names if n.endswith('down_proj')]
    want_all = defaultdict(dict)
    for n in downs:
        l, e, p = n.split('.')
        want_all[int(l[1:])][int(e[1:])] = ['down_proj']
    STEQ = pr.ste_fn()
    routes, hs = [], []
    with torch.no_grad():
        with tm.block('forward_routing'):
            h = F.embedding(toks, g['embed'])
        hs.append(h)
        for l in range(NL):
            with tm.block('ckpt_load'):
                w = ck.layer(l, 'cuda', torch.float32)
            with tm.block('forward_routing'):
                h, idx, rw = pr.layer_tap(w, h, cfg, cos, sin, None, None, {})
            routes.append((idx, rw)); hs.append(h)
            del w
    with tm.block('backward'):
        hT = hs[-1].clone().requires_grad_(True)
        head_ce(g, hT, toks, cfg).sum().backward()
        gn = hT.grad
    a_, b_ = ss.COLOFF['down_x']
    ya, yb = ss.COLOFF['down_y']
    from run_native_bc_pipeline import QUANT as Q
    fi = [Q.index(f) for f in QUANT]
    CB, terms, negs, ctl = {}, {}, {'channels': 0, 'min': 0.0, 'neg_total': 0}, {'n_equal': True, 'G2_equal': True, 'GS_equal': True, 'S1_equal': True, 'S2_equal': True}
    for l in reversed(range(NL)):
        with tm.block('ckpt_load'):
            w = ck.layer(l, 'cuda', torch.float32)
        want = want_all.get(l, {})
        cap = {}
        with tm.block('backward'):
            hi = hs[l].clone().requires_grad_(True)
            o = ss.layer_tap_sel(w, hi, cfg, cos, sin, routes[l][0], routes[l][1], pr.act_map(None, STEQ), want, cap)
            o.backward(gn)
            gn = hi.grad
        with torch.no_grad():
            for e in sorted(want):
                node = f'L{l}.E{e}'
                name = f'{node}.down_proj'
                c = cap[e]
                tok = c['tok']
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                with tm.block('act_grad_stats'):
                    gt = c['h'].grad.double()
                    G2 = (gt ** 2).sum(0)
                    GS = torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0)
                with tm.block('cache_read'):
                    grow = np.asarray(gc[gidx['row'][node]])
                    arow = np.asarray(ac[aidx['row'][node]])
                    n_c = int(aidx['n'][node])
                with tm.block('controls'):
                    ctl['n_equal'] &= n == n_c
                    ctl['G2_equal'] &= bool(np.array_equal(G2.cpu().numpy(), grow[0, a_:b_]))
                    ctl['GS_equal'] &= bool(np.array_equal(GS.cpu().numpy(), grow[1, a_:b_]))
                rng = ranges[name]['x']
                vxp = pair_terms(c['h'].detach(), c['h'].grad, (G2, GS), rng, n, dq, QUANT, tm, negs)
                with tm.block('table_assembly'):
                    G2y, GSy = grow[0, ya:yb], grow[1, ya:yb]
                    S1y, S2y = arow[0][fi][:, ya:yb], arow[1][fi][:, ya:yb]
                    vy = [float((G2y / n_c * (S2y[i] - n_c * (S1y[i] / n_c) ** 2)).sum() + (GSy * (S1y[i] / n_c) ** 2).sum()) for i in range(len(QUANT))]
                    G2x, GSx = grow[0, a_:b_], grow[1, a_:b_]
                    S1x, S2x = arow[0][fi][:, a_:b_], arow[1][fi][:, a_:b_]
                    vx = [float((G2x / n_c * (S2x[i] - n_c * (S1x[i] / n_c) ** 2)).sum() + (GSx * (S1x[i] / n_c) ** 2).sum()) for i in range(len(QUANT))]
                    CB[name] = {'fp32': 0.0}
                    for i, f in enumerate(QUANT):
                        cc = 0.0
                        cc += vxp[i]
                        cc += vy[i]
                        CB[name][f] = cc
                    terms[name] = {'vx_old_cache': vx, 'vy_cache': vy, 'vx_pair': vxp}
                with tm.block('controls'):
                    for i, f in enumerate(QUANT):
                        err = (dq.act(c['h'].detach(), f, rng) - c['h'].detach()).double()
                        ctl['S1_equal'] &= bool(np.array_equal(err.sum(0).cpu().numpy(), arow[0][fi[i]][a_:b_]))
                        ctl['S2_equal'] &= bool(np.array_equal((err ** 2).sum(0).cpu().numpy(), arow[1][fi[i]][a_:b_]))
        del w, cap
        log(f'Binc: layer {l} ({sum(tm.t.values()):.0f}s)')
    with tm.block('table_assembly'):
        tabB = {'B': static_tab['B'], 'C': {n: (CB[n] if n in CB else static_tab['C'][n]) for n in names}, 'meta': {'incremental': True}}
    return {'B_table': tabB, 'terms': terms, 'negative_channels': negs, 'cache_controls': ctl}


def build(out, log, variant, rep):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    tag = f'build_{variant}_r{rep}'
    rd = out / 'work/runs' / tag
    if (rd / 'run.json').exists():
        raise SystemExit('exists')
    rd.mkdir(parents=True, exist_ok=True)
    tm = pr.Timer()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with tm.block('setup_inputs'):
        names, nodes, elems, flops, rows = pr.problem_static()
        ranges = read(RANGES)
    with tm.block('quant_init'):
        dq = dev_quant(out / 'work', 'cuda', corrected=True)
    extra = {}
    if variant in ('A', 'B'):
        r = build_full(dq, ranges, names, tm, log, pair=(variant == 'B'))
        with tm.block('table_write'):
            write(rd / 'table_A.json', r['A_table'])
            if variant == 'B':
                write(rd / 'table_B.json', r['B_table'])
                write(rd / 'down_terms.json', r['terms'])
        extra = {'grads_finite': r['grads_finite'], 'nonfinite': len(r['nonfinite']), 'ce_forward': r['ce_forward'], 'validation': r['validation'],
                 'negative_channels': r['negative_channels']}
    else:
        with tm.block('input_read'):
            st = read(STATIC_TABLE)
        r = build_incremental(dq, ranges, names, tm, log, st)
        with tm.block('table_write'):
            write(rd / 'table_B.json', r['B_table'])
            write(rd / 'down_terms.json', r['terms'])
        extra = {'cache_controls': r['cache_controls'], 'negative_channels': r['negative_channels']}
    rec = run_record(tm, t0, {'variant': variant, 'rep': rep, 'code': code_hashes(), **extra})
    write(rd / 'run.json', rec)
    log(json.dumps(dr.clean({k: rec[k] for k in ('total_s', 'method_s', 'peak_gpu_gb')} | {'pair_stats_s': rec['timers'].get('pair_stats'),
                                                                                         'negative_channels': extra.get('negative_channels')})))


# --------------------------------------------------------------------------- #
def solve(t, prob, budget, logdir, tag, method='lpfirst'):
    """ss.search_table with an explicit solver method (identical otherwise)."""
    from run_native_bc_pipeline import GRID11
    from run_olmoe_multipoint import nondominated, pruned, resources
    from run_olmoe_search_final import make_solver
    from mp_auto import config_hash
    names, nodes, elems, flops = prob
    bc = lambda n, w, a: t['B'][n][w] + t['C'][n][a]
    t0 = time.perf_counter()
    keep = nondominated(names, bc)
    t_prune = time.perf_counter() - t0
    recs = []
    logdir.mkdir(parents=True, exist_ok=True)
    sel = make_solver(method, recs, logdir, tag=tag)
    t1 = time.perf_counter()
    fm, obj, status = sel(nodes, GRID11, GRID11, pruned(bc, keep), elems, flops, budget['max_bytes'], budget['max_bops'])
    t_solve = time.perf_counter() - t1
    fm = {n: fm[n] for n in names}
    r = resources(fm, names, elems, flops)
    return fm, {'method': method, 'objective': obj, 'objective_recomputed': float(sum(bc(n, fm[n]['w'], fm[n]['a']) for n in names)), 'status': status,
                'config_hash': config_hash(fm), 'path': recs[-1].get('path'), 'lower_bound': recs[-1].get('lower_bound'), 'record': recs[-1],
                'gap_abs': (obj - recs[-1]['lower_bound']) if recs[-1].get('lower_bound') is not None else None,
                'options_kept': sum(len(v) for v in keep.values()), 'resources': r,
                'within_budget': r['M_bytes'] <= budget['max_bytes'] and r['BOPs'] <= budget['max_bops'],
                'budget_use': {'size': r['M_bytes'] / budget['max_bytes'], 'bops': r['BOPs'] / budget['max_bops']},
                'slack': {'bytes': budget['max_bytes'] - r['M_bytes'], 'bops': budget['max_bops'] - r['BOPs']},
                'all_formats_legal': all(v['w'] in GRID11 and v['a'] in GRID11 for v in fm.values()), 'pruning_s': t_prune, 'solve_s': t_solve}


def table_checks(tA, tB, names):
    from run_native_bc_pipeline import QUANT
    downs = [n for n in names if n.endswith('down_proj')]
    return {'weight_rows_equal_A': tB['B'] == tA['B'], 'non_down_act_rows_equal_A': all(tB['C'][n] == tA['C'][n] for n in names if not n.endswith('down_proj')),
            'down_rows_covered': all(n in tB['C'] for n in downs), 'down_rows_changed': sum(tB['C'][n] != tA['C'][n] for n in downs), 'downs': len(downs),
            'finite': all(math.isfinite(v) for part in ('B', 'C') for r in tB[part].values() for v in r.values()),
            'formats_complete': all(set(tB['C'][n]) == {'fp32', *QUANT} for n in names)}


def search(out, log):
    if (out / 'search.json').exists():
        raise SystemExit('exists')
    t0 = time.perf_counter()
    names, nodes, elems, flops, rows = pr.problem_static()
    prob = (names, nodes, elems, flops)
    lim = read(KNEE)['limits']
    W = out / 'work/runs'
    tA = read(W / 'build_A_r1/table_A.json')
    tB = read(W / 'build_B_r1/table_B.json')
    st = read(STATIC_TABLE)
    tBi = read(W / 'build_Binc_r1/table_B.json')
    ctl = {'A_rebuild_equals_static': {k: tA[k] == st[k] for k in ('B', 'C')}, 'A_rebuild_r2_equals_r1': read(W / 'build_A_r2/table_A.json') == tA,
           'B_build_A_part_equals_static': {k: read(W / 'build_B_r1/table_A.json')[k] == st[k] for k in ('B', 'C')},
           'B_r2_equals_r1': read(W / 'build_B_r2/table_B.json') == tB, 'B_table_checks': table_checks(st, tB, names),
           'Binc_r2_equals_r1': read(W / 'build_Binc_r2/table_B.json') == tBi}
    rel = max(abs(tBi['C'][n][f] - tB['C'][n][f]) / max(abs(tB['C'][n][f]), 1e-300) for n in names if n.endswith('down_proj') for f in tB['C'][n] if f != 'fp32')
    ctl['Binc_vs_B_down_rows_max_rel'] = rel
    ctl['Binc_equals_B_other_rows'] = all(tBi['C'][n] == tB['C'][n] for n in names if not n.endswith('down_proj')) and tBi['B'] == tB['B']
    terms = read(W / 'build_B_r1/down_terms.json')
    from run_native_bc_pipeline import QUANT
    ctl['down_split_reproduces_A'] = max(abs(0.0 + terms[n]['vx_old'][i] + terms[n]['vy'][i] - st['C'][n][f]) for n in terms for i, f in enumerate(QUANT))
    ctl['pass'] = bool(all(ctl['A_rebuild_equals_static'].values()) and ctl['A_rebuild_r2_equals_r1'] and all(ctl['B_build_A_part_equals_static'].values())
                       and ctl['B_r2_equals_r1'] and ctl['Binc_r2_equals_r1'] and ctl['B_table_checks']['weight_rows_equal_A']
                       and ctl['B_table_checks']['non_down_act_rows_equal_A'] and ctl['B_table_checks']['down_rows_covered'] and ctl['B_table_checks']['finite']
                       and ctl['B_table_checks']['formats_complete'] and ctl['Binc_equals_B_other_rows'] and rel <= TOL_REL and ctl['down_split_reproduces_A'] == 0)
    if not ctl['pass']:
        write(out / 'search_controls_failed.json', dr.clean(ctl))
        raise SystemExit('table controls failed: investigate before searching')
    write(out / 'tables_B.json', tB)
    res = {'table_controls': ctl}
    fmA, sA = solve(st, prob, lim, out / 'logs', 'A')
    q0 = read(A_CONFIG)
    sA['reproduces_q0_knee'] = fmA == q0
    res['A'] = sA
    if not sA['reproduces_q0_knee']:
        write(out / 'search.json', dr.clean(res))
        raise SystemExit('the A search does not reproduce q0_knee: investigate (version, solver, tie-breaking) before any comparison')
    fmB, sB = solve(tB, prob, lim, out / 'logs', 'B')
    objB = lambda fm: float(sum(tB['B'][n][fm[n]['w']] + tB['C'][n][fm[n]['a']] for n in names))
    objA = lambda fm: float(sum(st['B'][n][fm[n]['w']] + st['C'][n][fm[n]['a']] for n in names))
    res['B_lpfirst'] = sB
    rule = {'A_config_on_B_table': objB(q0), 'B_solution_on_B_table': sB['objective'], 'triggered': objB(q0) < sB['objective']}
    chosen, chosen_s = fmB, sB
    if rule['triggered']:
        fmM, sM = solve(tB, prob, lim, out / 'logs', 'B_milp', method='milp')
        res['B_milp'] = sM
        rule['milp_objective'] = sM['objective']
        if sM['within_budget'] and sM['objective'] < sB['objective']:
            chosen, chosen_s = fmM, sM
        rule['still_above_A_config'] = objB(q0) < chosen_s['objective']
    rule['search_limited'] = bool(rule.get('still_above_A_config', False))
    res['approximation_rule'] = rule
    res['B'] = {'chosen': chosen_s['method'], **chosen_s}
    res['cross_scores'] = {'A_on_A_table': objA(q0), 'B_on_A_table': objA(chosen), 'A_on_B_table': objB(q0), 'B_on_B_table': objB(chosen),
                           'note': 'absolute numbers of different tables are not comparable'}
    (out / 'configs').mkdir(exist_ok=True)
    write(out / 'configs/A.json', q0)
    write(out / 'configs/B.json', chosen)
    res['identical'] = chosen == q0
    write(out / 'search.json', res)
    merge_timing(out, 'search', {'seconds': time.perf_counter() - t0, 'A': {k: sA[k] for k in ('pruning_s', 'solve_s', 'path')},
                                 'B': {k: chosen_s[k] for k in ('pruning_s', 'solve_s', 'path')}, 'B_milp': {k: res['B_milp'][k] for k in ('pruning_s', 'solve_s', 'path')} if 'B_milp' in res else None,
                                 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps(dr.clean({'A': {k: sA[k] for k in ('objective', 'path', 'gap_abs', 'reproduces_q0_knee', 'budget_use')},
                             'B': {k: chosen_s[k] for k in ('objective', 'path', 'gap_abs', 'budget_use', 'within_budget')}, 'rule': rule, 'identical': res['identical'],
                             'Binc_rel': rel})))


def configs(out, log):
    from htq_pareto_search import FORMAT_BITS
    if (out / 'config_diff.json').exists():
        raise SystemExit('exists')
    A, B = read(out / 'configs/A.json'), read(out / 'configs/B.json')
    names = sorted(A)
    fam = lambda f: 'fp32' if f == 'fp32' else 'FP' if f in ('f16', 'bf16') else 'INT' if f.startswith('int') else 'Posit'
    with open(TS / 'discovery_experts.csv') as fh:
        ex = [r for r in csv.DictReader(fh) if r['state'] == 'fp32' and float(r['rows']) >= 512]
    p90 = float(np.quantile([float(r['R_token']) for r in ex], 0.9))
    strong = {r['node'] for r in ex if float(r['R_token']) >= p90}
    grp = lambda n: 'front' if int(n.split('.')[0][1:]) in FRONT else 'back' if int(n.split('.')[0][1:]) in BACK else 'middle'
    ch = [n for n in names if A[n] != B[n]]
    kind = Counter('both' if A[n]['w'] != B[n]['w'] and A[n]['a'] != B[n]['a'] else 'weight_only' if A[n]['w'] != B[n]['w'] else 'act_only' for n in ch)
    res = {'projections': len(names), 'changed': len(ch), 'changed_experts': len({'.'.join(n.split('.')[:2]) for n in ch}), 'kind': dict(kind),
           'by_projection_type': dict(Counter(n.split('.')[-1] for n in ch)), 'by_layer_group': dict(Counter(grp(n) for n in ch)),
           'strong_spike_threshold_R_token_p90': p90, 'strong_spike_changed': sum('.'.join(n.split('.')[:2]) in strong for n in ch),
           'strong_spike_projections': sum('.'.join(n.split('.')[:2]) in strong for n in names),
           'weight_transitions': dict(Counter(f'{A[n]["w"]}->{B[n]["w"]}' for n in ch if A[n]['w'] != B[n]['w'])),
           'act_transitions': dict(Counter(f'{A[n]["a"]}->{B[n]["a"]}' for n in ch if A[n]['a'] != B[n]['a'])),
           'formats': {c: {'w': dict(Counter(cf[n]['w'] for n in names)), 'a': dict(Counter(cf[n]['a'] for n in names)),
                           'a_family': dict(Counter(fam(cf[n]['a']) for n in names)), 'w_bits': dict(Counter(FORMAT_BITS[cf[n]['w']] for n in names)),
                           'a_bits': dict(Counter(FORMAT_BITS[cf[n]['a']] for n in names)),
                           'down_a': dict(Counter(cf[n]['a'] for n in names if n.endswith('down_proj')))} for c, cf in (('A', A), ('B', B))},
           'A_fp16_act_projections': {n: {'A': A[n], 'B': B[n]} for n in names if A[n]['a'] in ('f16', 'bf16')},
           'changed_list': [{'name': n, 'A': A[n], 'B': B[n], 'group': grp(n), 'strong_spike': '.'.join(n.split('.')[:2]) in strong} for n in ch],
           'identical': A == B}
    write(out / 'config_diff.json', res)
    log(json.dumps(dr.clean({k: res[k] for k in ('changed', 'changed_experts', 'kind', 'by_projection_type', 'by_layer_group', 'strong_spike_changed', 'act_transitions',
                                                 'weight_transitions', 'identical')})))


def freeze(out, log):
    if (out / 'freeze.json').exists():
        raise SystemExit('exists')
    from mp_auto import config_hash
    W = out / 'work/runs'
    files = ['plan.json', 'search.json', 'config_diff.json', 'configs/A.json', 'configs/B.json', 'tables_B.json'] + \
        [str(p.relative_to(out)) for p in sorted(W.glob('build_*/*.json'))]
    cfgs = {c: {'path': str((out / f'configs/{c}.json').resolve().relative_to(HERE)), 'sha256': sha(out / f'configs/{c}.json'),
                'config_hash': config_hash(read(out / f'configs/{c}.json'))} for c in ('A', 'B')}
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'files': {f: sha(out / f) for f in files}, 'configs': cfgs, 'code': code_hashes(),
                                'note': 'tables, configurations, code and hashes frozen before any evaluation article is drawn'})
    log('frozen')


def data(out, log):
    import olmoe_common as oc
    if (out / 'segments.json').exists():
        raise SystemExit('exists')
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    pl = read(out / 'plan.json')['evaluation']
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
        if len(ids) < WPA * SEQ:
            skips['short'] += 1; continue
        sh = mr.shingles(a['body_norm'])
        hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
        c2 = max(hits.values()) / len(sh) if hits and sh else 0.0
        cp = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        ca = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        if max(c2, cp, ca) >= mr.NEARDUP:
            skips['near_duplicate'] += 1; continue
        st = int(rs.integers(0, len(ids) - WPA * SEQ + 1))
        w = np.array(ids[st:st + WPA * SEQ], dtype=np.int64).reshape(WPA, SEQ)
        toks.append(w); acc_sh.append(sh)
        segs.append({'index': len(segs), 'source': 'wikitext-103-raw-v1 train', 'title': a['title'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                     'window_tokens': [st, st + WPA * SEQ], 'max_shingle_share': {'wt2': c2, 'previous': cp, 'accepted': ca},
                     'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest()})
        if len(segs) == N:
            break
    inter = {s['content_sha256'] for s in segs} & used
    if len(segs) < N or inter:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'), 'n': N, 'windows_per_article': WPA,
                                     'skips': dict(skips), 'visited': N + sum(skips.values()), 'excluded_previous_wt103': len(prev),
                                     'content_hash_intersection_with_history': len(inter), 'tokens_sha256': sha(out / 'tokens.npy'),
                                     'segments_sha256': sha(out / 'segments.json')})
    merge_timing(out, 'data', {'seconds': time.perf_counter() - t0, 'peak_cpu_gb': ss.peak_cpu_gb()})
    log(f'data: {N} articles; skips {dict(skips)}; excluded previous {len(prev)}; intersection {len(inter)}')


def eval_(out, log):
    import torch
    from run_olmoe_search_final import dev_quant
    dr.gpu_setup()
    if (out / 'eval.json').exists():
        raise SystemExit('exists')
    fz = read(out / 'freeze.json')
    for c, v in fz['configs'].items():
        if sha(HERE / v['path']) != v['sha256']:
            raise SystemExit('frozen configuration changed')
    if sha(out / 'tokens.npy') != read(out / 'data_freeze.json')['tokens_sha256']:
        raise SystemExit('frozen data changed')
    if fz['configs']['A']['config_hash'] == fz['configs']['B']['config_hash']:
        write(out / 'eval.json', {'skipped': True, 'reason': 'A and B identical: configuration unchanged at this resource point'})
        log('identical configurations: evaluation skipped')
        return
    dq = dev_quant(out / 'work', 'cuda', corrected=True)
    ranges = read(RANGES)
    torch.cuda.reset_peak_memory_stats()
    old_t, old = np.load(KW / 'tokens.npy')[:CTRL], read(KW / 'eval.json')['ce']
    r = dr.forward(old_t, read(out / 'configs/A.json'), dq, ranges, record=False, log=log, label='program control A')
    pc = float(np.max(np.abs(np.array(r['ce']) - np.array(old['A'][:CTRL]))))
    toks = np.load(out / 'tokens.npy')
    runs = {}
    runs['fp32'] = dr.forward(toks, None, dq, ranges, record=True, log=log, label='eval fp32')
    np.savez(out / 'routes_fp32.npz', ids=runs['fp32']['routes']['ids'], wsum=runs['fp32']['routes']['weights'].sum(-1))
    for c in ('A', 'B'):
        runs[c] = dr.forward(toks, read(out / f'configs/{c}.json'), dq, ranges, record=True, log=log, label=f'eval {c}')
        np.savez(out / f'routes_{c}.npz', ids=runs[c]['routes']['ids'], wsum=runs[c]['routes']['weights'].sum(-1))
    r2 = dr.forward(toks[:CTRL], read(out / 'configs/B.json'), dq, ranges, record=True, log=log, label='rerun B')
    ctl = {'program_control_A_ce_max_abs': pc,
           'rerun_B_first_chunk': {'ce_max_abs': float(np.max(np.abs(np.array(r2['ce']) - np.array(runs['B']['ce'][:CTRL])))),
                                   'hidden_bitwise_windows': int(sum(x == y for x, y in zip(r2['hidden_sha'], runs['B']['hidden_sha'][:CTRL]))),
                                   'routes_equal': bool(np.array_equal(r2['routes']['ids'], runs['B']['routes']['ids'][:CTRL]))},
           'finite': all(x['finite_ce'] for x in runs.values()), 'audit': {c: runs[c]['audit'] for c in ('A', 'B')},
           'routes_differ_from_fp32': {c: float((np.sort(runs[c]['routes']['ids'], -1) != np.sort(runs['fp32']['routes']['ids'], -1)).any(-1).mean()) for c in ('A', 'B')},
           'loadavg_end': os.getloadavg()}
    ctl['pass'] = bool(pc == 0 and ctl['rerun_B_first_chunk']['ce_max_abs'] == 0 and ctl['rerun_B_first_chunk']['hidden_bitwise_windows'] == CTRL
                       and ctl['rerun_B_first_chunk']['routes_equal'] and ctl['finite']
                       and all(a['router_untouched'] and a['hook_keys_are_expert_projections'] for a in ctl['audit'].values()))
    write(out / 'eval.json', {'ce': {c: x['ce'] for c, x in runs.items()}, 'controls': ctl, 'seconds': {c: x['seconds'] for c, x in runs.items()},
                              'rerun_s': r2['seconds'], 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30, 'environment': dr.env_info()})
    merge_timing(out, 'eval', {**{f'{c}_s': x['seconds'] for c, x in runs.items()}, 'rerun_s': r2['seconds'], 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                               'peak_cpu_gb': ss.peak_cpu_gb()})
    log(json.dumps(dr.clean({'controls_pass': ctl['pass'], 'program_control': pc})))


def timing_summary(out):
    W = out / 'work/runs'
    runs = {p.name: read(p / 'run.json') for p in sorted(W.iterdir()) if (p / 'run.json').exists()}
    jl = [json.loads(x) for x in (out / 'timing.jsonl').read_text().splitlines()]
    proc = {f'build_{j["variant"]}_r{j["rep"]}': j['seconds'] for j in jl if j['stage'] == 'build'}
    med = lambda xs: {'median': float(np.median(xs)), 'min': float(min(xs)), 'max': float(max(xs)), 'values': [float(x) for x in xs]}
    s = {}
    for v in ('A', 'B', 'Binc'):
        ks_ = [f'build_{v}_r{r}' for r in (1, 2)]
        s[v] = {'method_s': med([runs[k]['method_s'] for k in ks_]), 'process_wall_s': med([proc[k] for k in ks_]),
                'categories': {c: med([runs[k]['categories'][c] for k in ks_]) for c in list(CATEGORIES) + ['unclassified']},
                'peak_gpu_gb': max(runs[k]['peak_gpu_gb'] for k in ks_), 'peak_cpu_gb': max(runs[k]['peak_cpu_gb'] for k in ks_),
                'loadavg_before': {k: runs[k]['loadavg_before'] for k in ks_}}
    t = read(out / 'timing.json')
    sr = t.get('search', {})
    s['search'] = sr
    s['added_pair_stats_in_B_build_median_s'] = s['B']['categories']['pair_stats']['median']
    s['B_minus_A_method_median_s'] = s['B']['method_s']['median'] - s['A']['method_s']['median']
    s['scratch_input_to_config_s'] = {'value': s['B']['method_s']['median'] + sr.get('B', {}).get('pruning_s', 0) + sr.get('B', {}).get('solve_s', 0),
                                      'note': 'B build (checkpoint -> table, median) + pruning + search at A\'s fixed limits (single); an auto-knee frontier '
                                              '(not run here) would be added in a from-scratch pipeline that also chooses the limits'}
    s['incremental_input_to_config_s'] = {'value': s['Binc']['method_s']['median'] + sr.get('B', {}).get('pruning_s', 0) + sr.get('B', {}).get('solve_s', 0),
                                          'note': 'reuses the static table and caches'}
    s['A_reference_input_to_config_s'] = {'value': s['A']['method_s']['median'] + sr.get('A', {}).get('pruning_s', 0) + sr.get('A', {}).get('solve_s', 0)}
    s['evaluation_separate'] = t.get('eval')
    return s


def analyze(out, log):
    from htq_pareto_search import FORMAT_BITS
    from run_olmoe_multipoint import problem
    t0 = time.perf_counter()
    ev = read(out / 'eval.json')
    res = {'search': read(out / 'search.json'), 'config_diff': {k: v for k, v in read(out / 'config_diff.json').items() if k != 'changed_list'}}
    if ev.get('skipped'):
        res['evaluation'] = ev
    else:
        if not ev['controls']['pass']:
            raise SystemExit('evaluation controls failed')
        A = {c: np.array(v).reshape(N, WPA).sum(1) / (WPA * TP) for c, v in ev['ce'].items()}
        idx = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))

        def st(x):
            m = x[idx].mean(1)
            return {'mean': float(x.mean()), 'ci95': np.percentile(m, [2.5, 97.5]).tolist(), 'article_sd': float(x.std(ddof=1)),
                    'better_articles': int((x < 0).sum()), 'worse_articles': int((x > 0).sum()), 'zero_articles': int((x == 0).sum())}
        D = A['B'] - A['A']
        prim = st(D)
        lo, hi = prim['ci95']
        prim['decision'] = 'B_better_confirmed' if hi < 0 else 'B_worse_confirmed' if lo > 0 else 'not_confirmed'
        res['primary_B_minus_A'] = prim
        res['damage'] = {c: st(A[c] - A['fp32']) for c in ('A', 'B')}
        tok = N * WPA * TP
        res['nll_ppl'] = {c: {'nll': float(np.sum(v) / tok), 'ppl': float(np.exp(np.sum(v) / tok))} for c, v in ev['ce'].items()}
        dA = res['nll_ppl']['A']['nll'] - res['nll_ppl']['fp32']['nll']
        res['relative_damage_change'] = (res['nll_ppl']['B']['nll'] - res['nll_ppl']['A']['nll']) / dA if dA else None
        cfg = {c: read(out / f'configs/{c}.json') for c in ('A', 'B')}
        _, _, pnames, _, elems2, rows_cal, _ = problem(MP)
        counts = {}
        for c in ('fp32', 'A', 'B'):
            ids = np.load(out / f'routes_{c}.npz')['ids'].astype(np.int64)
            counts[c] = np.stack([np.bincount(ids[:, l].ravel(), minlength=NE) for l in range(NL)])
        searched = {(int(n.split('.')[0][1:]), int(n.split('.')[1][1:])) for n in pnames}
        unsearched = [(l, e) for l in range(NL) for e in range(NE) if (l, e) not in searched]
        rowsof = lambda cnt, n: cnt[int(n.split('.')[0][1:]), int(n.split('.')[1][1:])]
        b8_s = sum(rowsof(counts['fp32'], n) * elems2[n] * 64 for n in pnames)
        cost = {}
        for c in ('A', 'B'):
            sb = float(sum(rowsof(counts[c], n) * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))
            cost[c] = {'routed_BOPs_relative_to_int8_fp32_counts': sb / b8_s, 'tokens_to_unsearched_experts': int(sum(counts[c][l, e] for l, e in unsearched)),
                       'search_proxy_BOPs_calibration_rows': float(sum(rows_cal[n] * elems2[n] * FORMAT_BITS[cfg[c][n]['w']] * FORMAT_BITS[cfg[c][n]['a']] for n in pnames))}
        cost['fp32_tokens_to_unsearched_experts'] = int(sum(counts['fp32'][l, e] for l, e in unsearched))
        cost['route_set_change_B_vs_A'] = float((np.sort(np.load(out / 'routes_B.npz')['ids'], -1) != np.sort(np.load(out / 'routes_A.npz')['ids'], -1)).any(-1).mean())
        res['routing_cost'] = cost
        res['eval_controls'] = ev['controls']
        segs = read(out / 'segments.json')
        with open(out / 'per_article.csv', 'w') as fh:
            rows = [{'index': s['index'], 'title': s['title'], **{f'nll_{c}': float(A[c][i]) for c in A}, 'B_minus_A': float(D[i])} for i, s in enumerate(segs)]
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(out / 'per_window.csv', 'w') as fh:
            rows = [{'window': k, 'article': k // WPA, **{f'summed_ce_{c}': ev['ce'][c][k] for c in ev['ce']}} for k in range(N * WPA)]
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    res['timing'] = timing_summary(out)
    after = {'inputs': {k: sha(HERE / k) for k in read(out / 'input_hashes_before.json')['inputs']}, 'model_files': {f.name: sha(f) for f in ss.model_files()}}
    res['input_hashes_unchanged'] = after == read(out / 'input_hashes_before.json')
    write(out / 'analysis.json', res)
    merge_timing(out, 'analyze', {'seconds': time.perf_counter() - t0})
    log(json.dumps(dr.clean({'primary': res.get('primary_B_minus_A'), 'nll_ppl': res.get('nll_ppl'), 'relative_damage_change': res.get('relative_damage_change')})))


PAL = tsd.PAL


def figures(out, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fd = out / 'figures'
    fd.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor': PAL['surface'], 'axes.facecolor': PAL['surface'], 'axes.edgecolor': PAL['muted'], 'axes.labelcolor': PAL['text'],
                         'xtick.color': PAL['text2'], 'ytick.color': PAL['text2'], 'text.color': PAL['text'], 'axes.grid': True, 'grid.color': PAL['grid'],
                         'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
    cd = read(out / 'config_diff.json')
    tr = sorted(cd['act_transitions'].items(), key=lambda x: -x[1])[:16] + [(f'w {k}', v) for k, v in sorted(cd['weight_transitions'].items(), key=lambda x: -x[1])[:10]]
    if tr:
        fig, ax = plt.subplots(figsize=(7, 0.28 * len(tr) + 1.2))
        ax.barh(range(len(tr)), [v for _, v in tr], color=[PAL['s2'] if k.startswith('w ') else PAL['s1'] for k, _ in tr], height=0.6)
        ax.set_yticks(range(len(tr))); ax.set_yticklabels([k if k.startswith('w ') else f'a {k}' for k, _ in tr], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel('projections'); ax.set_title('Format transitions A -> B (a = activation, w = weight)', loc='left', fontsize=10)
        fig.tight_layout(); fig.savefig(fd / 'transitions.png', dpi=150); plt.close(fig)
    if (out / 'per_article.csv').exists():
        d = np.array([float(r['B_minus_A']) for r in csv.DictReader(open(out / 'per_article.csv'))])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(d, bins=60, color=PAL['s1'])
        ax.axvline(0, color=PAL['text2'], lw=0.8); ax.axvline(d.mean(), color=PAL['s2'], lw=1.5, label=f'mean {d.mean():+.2e}')
        ax.set_xlabel('NLL(B) - NLL(A) per article (nat/token)'); ax.set_ylabel('articles'); ax.legend(frameon=False)
        ax.set_title('Per-article difference (600 new articles)', loc='left', fontsize=10)
        fig.tight_layout(); fig.savefig(fd / 'per_article_delta.png', dpi=150); plt.close(fig)
    log(f'figures: {sorted(p.name for p in fd.iterdir())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['plan', 'build', 'search', 'configs', 'freeze', 'data', 'eval', 'analyze', 'figures'], required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--variant', choices=['A', 'B', 'Binc'])
    ap.add_argument('--rep', type=int, default=1)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M:%S'), msg, flush=True)
    if a.stage != 'plan' and not (a.out_dir / 'plan.json').exists():
        raise SystemExit('plan required')
    if a.stage == 'build':
        build(a.out_dir, log, a.variant, a.rep)
    elif a.stage == 'eval':
        eval_(a.out_dir, log)
    else:
        globals()[a.stage](a.out_dir, log)
    with open(a.out_dir / 'timing.jsonl', 'a') as f:
        f.write(json.dumps({'stage': a.stage, 'variant': a.variant, 'rep': a.rep, 'seconds': time.time() - T_PROC, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'script_sha256': sha(__file__)}) + '\n')


if __name__ == '__main__':
    main()
