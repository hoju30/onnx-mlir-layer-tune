#!/usr/bin/env python3
"""Qwen3-0.6B: is the FP16 activation protection of the L2 / L27 Down projections in the original BC knee necessary?
A fixed-configuration local intervention (no criterion change, no search, no knee re-selection, no fine-tuning, the
original calibration).

Configurations (runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json = A):
  A  unchanged                       B  L2 Down a: f16 -> p8e1
  C  L27 Down a: f16 -> p8e1         D  both
`a` is the projection's activation format: the pass (MIXEDP_NODE_FORMATS a=) and run_lp_pipeline_eval.LMReplay apply it
to the projection input (x range) and to its output (y range); weights (w) are unchanged and identical.
Truth: per-article NLL on new WikiText-103 articles, GPU PyTorch QDQ replay (LMReplay); native onnx-mlir cross-check on
the 10 original calibration windows. Resources: the pipeline's resource model (mp_auto) on run_solver_regression's
Qwen3 problem (native_eps_fix table, spec elements / FLOPs).

Stages (one new --out-dir): plan, controls, native, freeze, audit, data, eval, analyze.
CPU stages: ../../venv/bin/python; controls / eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import time
T_PROC = time.time()
import argparse
import copy
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
import run_olmoe_dynamic_route as dr  # noqa: E402  (read / write / sha / env_info)

REF = HERE / 'runs/lp_pipeline_regression_20261016/C/qwen3'
SEARCH = HERE / 'runs/lp_pipeline_regression_20261016/B/qwen3/search_milp_orig.json'
NATIVE = HERE / 'runs/native_eps_fix_20261014/qwen3'
PREP = HERE / 'runs/qwen3_prepare_20261011'
MODEL = HERE / 'qwen3_assets/model'
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
NODES = {'L2': '/model/model/layers.2/mlp/down_proj/MatMul', 'L27': '/model/model/layers.27/mlp/down_proj/MatMul'}
EXPECT_A = {'w': 'int8', 'a': 'f16'}
NEW_A = 'p8e1'
CONFIGS = {'A': (), 'B': ('L2',), 'C': ('L27',), 'D': ('L2', 'L27')}
SEQ, TP, CAL, WPA, N_ART = 128, 127, 10, 3, 512
DELTA, BOOT, ALPHA, K = 0.001, 20000, 0.05, 3
SEEDS = {'order': 28101, 'start': 28102, 'bootstrap': 28103}
NATIVE_CFGS, NATIVE_WINDOWS = ('FP32', 'A', 'B', 'C', 'D'), CAL
OLMOE_VOCAB = 50304
BUDGET_MIN = {'plan': 5, 'controls': 10, 'native': 30, 'freeze': 1, 'audit': 5, 'data': 15, 'eval': 20, 'analyze': 5}
CODE = ['run_qwen_down_fp16_protection.py', 'run_lp_pipeline_eval.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py',
        'run_w8_per_channel_check.py', 'run_native_bc_pipeline.py', 'run_solver_regression.py', 'mp_auto.py', 'htq_pareto_search.py',
        'evaluate_native_search_chunked.py', 'wt_manifest_scan.py', 'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py',
        'run_qwen_down_spike_diag.py']


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
            + [REF / 'configs/ref.json', REF / 'freeze.json', SEARCH, NATIVE / 'table.json', NATIVE / 'ranges_by_point.json',
               NATIVE / 'calib_inputs.npy', NATIVE / 'model.mlirbc', PREP / 'model.onnx', PREP / 'model.onnx.data',
               PREP / 'node_mapping.json', PREP / 'validation_tokens.npy', PREP / 'test_tokens.npy', SPIKE / 'segments.json',
               SPIKE / 'tokens.npy', HERE / 'qwen35_assets/model/tokenizer.json'] + mr.WT103 + mr.WT2)


def log_stage(out, stage, seconds, extra=None):
    f = out / 'timing.json'
    t = read(f) if f.exists() else {}
    t[stage] = {'seconds': seconds, 'budget_min': BUDGET_MIN.get(stage), 'over_budget': seconds > 60 * BUDGET_MIN.get(stage, 1e9), **(extra or {})}
    write(f, t)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    if pl['code_hashes'] != code_hashes():
        raise SystemExit('code changed since the plan (record an amendment first)')
    if inputs:
        for k, v in pl['inputs'].items():
            if sha(HERE / k) != v:
                raise SystemExit('input changed: ' + k)
    for nm, h in pl['configs_sha256'].items():
        if sha(out / 'configs' / f'{nm}.json') != h:
            raise SystemExit(f'config {nm} changed')
    return pl


def configs(out):
    return {nm: read(out / 'configs' / f'{nm}.json') for nm in CONFIGS}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# plan: configurations, diffs, resources
# --------------------------------------------------------------------------- #
def resources(cfgs, work):
    """the pipeline's resource model (mp_auto.auto_select's record) on run_solver_regression's Qwen3 problem."""
    from htq_pareto_search import FORMAT_BITS
    from run_solver_regression import dense_problem
    nodes, names, value, elems, flops, _ = dense_problem('qwen3', work)
    table = read(NATIVE / 'table.json')
    ref_bytes = sum(elems[n] for n in names)
    ref_bops = sum(64 * flops[n] for n in names)
    res = {'searched_projections': len(names), 'int8_weight_bytes': ref_bytes, 'int8_bops': ref_bops,
           'formula': {'size': 'sum elems * bits(w) / 8 / sum elems', 'bops': 'sum flops * bits(w) * bits(a) / sum 64 flops',
                       'resource': 'max(size, bops)', 'objective': 'sum table B[w] + C[a] (native_eps_fix table.json)',
                       'flops': 'spec weight elements x sequence length 128 (run_native_bc_pipeline.qwen3_spec)'},
           'configs': {}}
    for nm, cfg in cfgs.items():
        wb = sum(elems[n] * FORMAT_BITS[cfg[n]['w']] / 8 for n in names)
        bops = sum(flops[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']] for n in names)
        obj = sum(value(n, cfg[n]['w'], cfg[n]['a']) for n in names)
        res['configs'][nm] = {
            'weight_bytes': wb, 'bops': bops, 'size': wb / ref_bytes, 'bops_rel': bops / ref_bops, 'resource': max(wb / ref_bytes, bops / ref_bops),
            'dominant': 'size' if wb / ref_bytes > bops / ref_bops else 'bops', 'objective': obj,
            'modified_nodes': {k: {'w': cfg[n]['w'], 'a': cfg[n]['a'], 'bops': flops[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']],
                                   'B_w': table['B'][n][cfg[n]['w']], 'C_a': table['C'][n][cfg[n]['a']],
                                   'elems': elems[n], 'flops': flops[n]} for k, n in NODES.items()}}
    A = res['configs']['A']
    for nm, r in res['configs'].items():
        r['vs_A'] = {'weight_bytes': r['weight_bytes'] - A['weight_bytes'], 'bops': r['bops'] - A['bops'],
                     'bops_pct_of_A_total': 100 * (r['bops'] - A['bops']) / A['bops'], 'bops_rel': r['bops_rel'] - A['bops_rel'],
                     'resource': r['resource'] - A['resource'], 'objective': r['objective'] - A['objective'],
                     'modified_projection_bops': sum(r['modified_nodes'][k]['bops'] - A['modified_nodes'][k]['bops'] for k in NODES)}
    sel = read(SEARCH)['selected']
    res['control_A_reproduces_search'] = {
        'size': [A['size'], sel['size']], 'bops': [A['bops_rel'], sel['bops']], 'resource': [A['resource'], sel['resource']],
        'objective': [A['objective'], sel['objective']],
        'passed': bool(abs(A['size'] - sel['size']) <= 1e-12 and abs(A['bops_rel'] - sel['bops']) <= 1e-12
                       and abs(A['resource'] - sel['resource']) <= 1e-12 and abs(A['objective'] - sel['objective']) <= 1e-9 * sel['objective'])}
    return res


def plan(out, log):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists: the plan is frozen once')
    (out / 'configs').mkdir(exist_ok=True)
    A = read(REF / 'configs/ref.json')
    rz = read(REF / 'freeze.json')
    sel = read(SEARCH)
    chk = {'ref_file_sha256_equals_regression_freeze': sha(REF / 'configs/ref.json') == rz['configs_sha256']['ref'],
           'ref_hash_equals_regression_freeze': config_hash(A) == rz['configs']['ref']['hash'],
           'ref_equals_milp_orig_selection': config_hash(A) == config_hash(sel['selected_formats']) == sel['selected']['config_hash'],
           'L2_L27_down_are_int8_f16': all(A[n] == EXPECT_A for n in NODES.values()), 'projections': len(A)}
    if not all(v for k, v in chk.items() if k != 'projections'):
        raise SystemExit(f'reference configuration differs from the expected source: {chk}')
    cfgs = {}
    for nm, mods in CONFIGS.items():
        c = copy.deepcopy(A)
        for k in mods:
            c[NODES[k]]['a'] = NEW_A
        cfgs[nm] = c
        write(out / 'configs' / f'{nm}.json', c)
    diffs = {}
    for nm, c in cfgs.items():
        d = [{'node': n, 'field': f, 'A': A[n][f], nm: c[n][f]} for n in A for f in ('w', 'a') if A[n][f] != c[n][f]]
        diffs[nm] = d
    chk['diffs_as_designed'] = (len(diffs['A']) == 0 and len(diffs['B']) == 1 and len(diffs['C']) == 1 and len(diffs['D']) == 2
                                and all(x['field'] == 'a' for v in diffs.values() for x in v)
                                and all(c[n]['w'] == A[n]['w'] for c in cfgs.values() for n in A) and set(cfgs['A']) == set(A))
    if not chk['diffs_as_designed']:
        raise SystemExit(f'config diff not as designed: {diffs}')
    t1 = time.perf_counter()
    res = resources(cfgs, out / 'work')
    res['seconds'] = time.perf_counter() - t1
    write(out / 'resources.json', res)
    if not res['control_A_reproduces_search']['passed']:
        raise SystemExit(f'resource model does not reproduce the search: {res["control_A_reproduces_search"]}')
    w103, wt2 = wms.corpora()
    used, rows = wms.used_wt103_verified(out, w103, wt2)
    spike_hashes = {s['content_sha256'] for s in read(SPIKE / 'segments.json')}
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    elig = [a for a in base if a['ch'] not in used]
    chk['spike_diag_128_in_verified_scan'] = bool(spike_hashes <= used and len(spike_hashes) == 128)
    if not chk['spike_diag_128_in_verified_scan']:
        raise SystemExit('the 128 articles of qwen3_down_spike_diag are not all excluded by the verified scan')
    inputs = {str(p.relative_to(HERE)): sha(p) for p in input_files()}
    rc = res['configs']
    plan_ = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'the original BC knee protects the L2 and L27 Down activations with f16; does replacing only these '
                    'with p8e1 lower the resource cost while keeping the added quality loss within a preset tolerance?',
        'nature': 'fixed-configuration local intervention; no criterion change, no search, no knee re-selection, no '
                  'fine-tuning, no new calibration; truth by GPU PyTorch QDQ (LMReplay), native cross-check on the 10 '
                  'calibration windows only',
        'reference_checks': chk,
        'configs': {nm: {'modified': [{'node': NODES[k], 'slot': 'a (projection input x and output y)', 'from': 'f16', 'to': NEW_A}
                                      for k in mods], 'hash': config_hash(cfgs[nm])} for nm, mods in CONFIGS.items()},
        'configs_sha256': {nm: sha(out / 'configs' / f'{nm}.json') for nm in CONFIGS},
        'fp32_baseline': 'every projection fp32 (descriptive only)',
        'semantics': {'native': 'MIXEDP_NODE_FORMATS a= applies to every float operand other than the weight and to the '
                                'result (ONNXToMixedPrecision.cpp); for MatMul: input x and output y',
                      'pytorch': 'run_lp_pipeline_eval.LMReplay: forward pre-hook QDQ of x (x range), forward hook QDQ of '
                                 'y (y range), per-output-channel int weights (Quantizers.weight), posit LUT on device',
                      'note': 'the previous round quantized the Down input only; this round replaces the whole a (x and y)'},
        'resources': {'method': res['formula'], 'A_reproduces_search': res['control_A_reproduces_search'],
                      'summary': {nm: {k: rc[nm][k] for k in ('size', 'bops_rel', 'resource', 'dominant', 'objective')} for nm in rc},
                      'statement': 'BOPs is a proxy: no claim of GPU QDQ latency or low-bit kernel speed-up; weight bytes '
                                   'are unchanged by construction (only a changes) and verified'},
        'controls': [
            'config diff: B / C one a each, D two; every w identical; quantized weight hashes identical across A-D',
            'int QDQ with scale / zero-point on the tensor device (LMReplay.qact); FP32, TF32 off (matmul and cuDNN), '
            'deterministic algorithms, highest matmul precision',
            'posit LUT verification (LMReplay); actual formats at the L2 / L27 Down input and output hooks match the '
            'configuration (idempotent under the expected quantizer; f16 not posit-only); one pre-hook and one forward '
            'hook per projection throughout; activation map equals the configuration after every switch',
            'FP32 identity replay equals an unhooked model bitwise; A repeated on old windows reproduces bitwise; two '
            'interleaved configuration orders give bitwise identical results',
            'original RMSNorm epsilon (config 1e-6), Qwen3 tokenizer, sequence length 128, summed next-token CE '
            '(127 positions), native calibration ranges',
            'native cross-check (run_lp_pipeline_eval.stage_c_native procedure: Krnl -O3, clang -O0, constants to file, '
            'per-channel weights) of FP32 / A / B / C / D on the 10 calibration windows, A run twice; compared with the '
            'PyTorch QDQ on the same windows, absolute NLL and the paired effects B-A, C-A, D-A',
            'any failed control stops before the freeze; locate, fix, record an amendment'],
        'backend_rule': f'for each of B-A, C-A, D-A on the 10 calibration windows: native - PyTorch paired-effect difference '
                        f'per window; if |mean| + 2 SE >= delta ({DELTA}) for any comparison, the backend difference can '
                        'change a delta-level reading and every conclusion is limited to the PyTorch QDQ diagnostic; even '
                        'otherwise, 10 calibration windows are not a native confirmation of the new-article result',
        'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'seq': SEQ,
                 'draw': f'WikiText-103 train, strict-boundary articles, not a WikiText-2 title / content hash, unique; not in '
                         f'wt_manifest_scan.used_wt103_verified (includes the 128 articles of qwen3_down_spike_diag_20261023); '
                         f'permutation order seed {SEEDS["order"]}; skip < {WPA * SEQ} tokens; skip >= 10% shingle share with '
                         'one WikiText-2 article, a previously used WikiText-103 article or an accepted article; skip >= 10% '
                         'overlap with any historical Qwen token window (audit, rerun on the current runs/); window start '
                         f'uniform (seed {SEEDS["start"]}); Qwen3 tokenizer; the same articles for every configuration',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                 'no_change': 'no article added or removed after any new loss; insufficient power -> undetermined'},
        'statistics': {
            'article_nll': 'sum of the 3 windows summed CE / 381',
            'primary': 'd_X = NLL(X) - NLL(A) for X in B, C, D',
            'bootstrap': f'paired article bootstrap, {BOOT} replicates (seed {SEEDS["bootstrap"]}), one resample for every '
                         'configuration',
            'intervals': f'95% percentile CI and the Bonferroni (3 comparisons) two-sided {100 * (1 - ALPHA / K):.3f}% CI',
            'delta': f'{DELTA} nat/token (this round\'s tolerance; not derived from new loss; not the 0.0075 of the solver '
                     'comparison)',
            'decision': {'corrected upper < 0': 'quality improvement supported',
                         'corrected upper < delta': 'added loss within the tolerance (if the corrected CI is all positive: '
                                                    'small quality drop within the preset tolerance, not "no loss")',
                         'corrected lower > delta': 'added loss exceeds the tolerance',
                         'otherwise': 'undetermined for non-inferiority',
                         'note': 'a CI covering 0 is not equivalence'},
            'interaction': 'I = NLL(D) - NLL(B) - NLL(C) + NLL(A) per article; mean and 95% CI; secondary; CI covering 0 '
                           'means interaction not confirmed (not proof of additivity); not attributed to Gate/Up, '
                           'position 0 or a second-order approximation',
            'fp32': 'NLL(config) - NLL(FP32) descriptive only'},
        'decision_rule': {'within_tolerance_and_cheaper': 'only "these selected positions show an opportunity to lower the '
                                                          'protection"; automatic identification from calibration is a later '
                                                          'study; no hard-coding of L2 / L27 into the pipeline',
                          'exceeds': 'keep f16; single-point INT / Posit misranks do not overturn the original configuration',
                          'undetermined': 'keep the original configuration and report the uncertainty',
                          'never': 'change defaults, start a search, add Posit6 / other layers / configurations after new '
                                   'loss, change ranges or scales from new data, call a GPU QDQ result a native gain'},
        'pre_plan_smoke_test': 'before this plan a scratch run (outside runs/) exercised every stage on old data: the GPU '
                               'controls on the 10 calibration windows (their FP32 / A / B / C / D PyTorch NLL were seen: B '
                               'and D about +0.3 nat/token over A, C about -0.005), native FP32 / A on 2 calibration windows, '
                               'data with other seeds (999101 / 999102, 6 articles, never evaluated), eval / analyze on 9 '
                               'calibration windows. One fault fixed (native scope text with missing comparisons). No new '
                               'article was evaluated; the configurations, delta, data size and rules come from the request '
                               'and were not changed.',
        'budget_minutes': BUDGET_MIN, 'seeds': SEEDS, 'inputs': inputs, 'code_hashes': code_hashes(),
        'environment_cpu': dr.env_info()}
    write(out / 'plan.json', plan_)
    L = ['# 計畫：Qwen3-0.6B Down activation FP16 保護是否必要', '',
         f'時間：{plan_["time"]}。以 plan.json 為準；寫入後不改配置、資料量、seed、δ 或決策規則。', '',
         '- 性質：固定配置的局部介入；不改 criterion、不搜尋、不重選 knee、不 fine-tune、不換 calibration。真值是 GPU PyTorch QDQ（LMReplay）；native 只在 10 個 calibration 窗口交叉驗證。',
         f'- A = `runs/lp_pipeline_regression_20261016/C/qwen3/configs/ref.json`（= MILP 原規則 knee，hash {config_hash(A)[:16]}…）；核對 L2、L27 Down 皆為 w=int8、a=f16。',
         '- B：L2 Down a f16→p8e1；C：L27 Down a f16→p8e1；D：兩者。a 同時套用 projection input（x range）與 output（y range），與 native pass 相同；w 全部不變。',
         f'- 配置 hash：' + '、'.join(f'{nm} {plan_["configs"][nm]["hash"][:12]}' for nm in CONFIGS) + '。另評估 FP32（只作描述）。',
         f'- 資源（原 pipeline 成本函式，A 重現 knee 的 size／BOPs／R／BC）：' + '；'.join(
             f'{nm} size {rc[nm]["size"]:.6f}、BOPs {rc[nm]["bops_rel"]:.6f}、R {rc[nm]["resource"]:.6f}、BC {rc[nm]["objective"]:.3f}' for nm in CONFIGS) + '。BOPs 只是 proxy。',
         '- 控制：config diff 與量化權重 hash；裝置上的 INT scale／zp、TF32 關閉、deterministic；Posit LUT 驗證；L2／L27 Down 的 input／output 實際格式；'
         'hook 不累積、切換不殘留；FP32 identity = 無量化模型；舊窗口重複 A 與交錯順序逐位元一致；原 epsilon／tokenizer／長度／loss／range；'
         'native（FP32／A／B／C／D，A 跑兩次）vs PyTorch 在 10 個 calibration 窗口的絕對 NLL 與 B−A、C−A、D−A paired 效應。控制失敗先定位，不進正式比較。',
         f'- 後端規則：任一比較的「native − PyTorch」paired 效應差 |平均| + 2 SE ≥ δ → 全部結論限定為 PyTorch QDQ 診斷；即使低於 δ，10 個 calibration 窗口也不是 native 對新文章結果的確認。',
         f'- 資料：{N_ART} 篇新 WT103 文章 × {WPA} 個不重疊 128-token 窗口（Qwen3 tokenizer）；verified 掃描排除 {len(used)} 篇（含 spike 診斷的 128 篇）；'
         f'排除 WT2 重複／近重複；重跑歷史 Qwen token audit 並排除重疊；seed 順序 {SEEDS["order"]}、起點 {SEEDS["start"]}；凍結後才抽；不依新 loss 增減文章。',
         f'- 統計：文章 NLL = 3 窗口 summed CE / 381；d_B、d_C、d_D = NLL(X) − NLL(A)；paired 文章 bootstrap {BOOT} 次（seed {SEEDS["bootstrap"]}），同一重抽；95% CI 與 Bonferroni 98.333% CI。',
         f'- δ = {DELTA} nat/token（本輪選擇，非由新 loss 推得，不沿用 0.0075）。校正上界 < 0 → 支持改善；< δ → 新增損傷在容許內（CI 全正時寫「小幅下降但在容許內」）；校正下界 > δ → 超過容許；其餘未定。CI 跨 0 不等於等效。',
         '- 次要：I = NLL(D) − NLL(B) − NLL(C) + NLL(A) 的平均與 95% CI；CI 跨 0 只表示未確認交互作用；不歸因 Gate/Up、position 0 或二階近似。',
         '- 決策：在容許內且降低資源 → 只說「這些已選定位置存在降低保護的機會」，不硬編碼進 pipeline；超標 → 保留 f16；未定 → 保留原配置。不改預設、不啟動搜尋、不把 GPU QDQ 結果稱為 native 收益。',
         '- 計畫前冒煙測試（scratchpad，不在 runs/，只用舊資料）：GPU 控制在 10 個 calibration 窗口上跑過，已看到 FP32／A／B／C／D 的 PyTorch NLL（B、D 約比 A 高 0.3 nat/token，C 約低 0.005）；'
         'native 只建 FP32／A、2 個窗口；資料階段用其他 seed（999101／999102）抽 6 篇未評估；eval／analyze 只用 9 個 calibration 窗口。修正 1 個程式錯誤。沒有評估任何新文章；配置、δ、資料量與規則來自需求，未因此更改。', '',
         f'## 預算（分鐘）：{BUDGET_MIN}', '']
    (out / 'plan.md').write_text('\n'.join(L) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0, {'config_preparation_and_resources_s': res['seconds']})
    log(f'plan written; checks {chk}; resources ' + json.dumps({nm: [rc[nm]['size'], rc[nm]['bops_rel'], rc[nm]['resource'], rc[nm]['objective']] for nm in rc}))


# --------------------------------------------------------------------------- #
# GPU replay helpers
# --------------------------------------------------------------------------- #
def replay(work, device='cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from run_lp_pipeline_eval import LMReplay
    t0 = time.perf_counter()
    rp = LMReplay('qwen3', device, work)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return rp, time.perf_counter() - t0


def weights_hash(rp):
    h = hashlib.sha256()
    for n in sorted(rp.param):
        h.update(rp.mod(n).weight.data.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def controls(out, log):
    t0 = time.perf_counter()
    pl = check_plan(out)
    cf = configs(out)
    rp, init_s = replay(out / 'work')
    torch = rp.torch
    from run_olmoe_dynamic_route import env_info
    toks = np.load(NATIVE / 'calib_inputs.npy')
    ctl = {'environment': env_info(), 'replay_init_s': init_s,
           'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
           'deterministic': torch.are_deterministic_algorithms_enabled(), 'matmul_precision': torch.get_float32_matmul_precision()}
    ctl['posit_lut_verified'] = {f: list(rp.q.posit[f].verify()) for f in ('p8e1',)}
    ctl['rms_norm_eps'] = sorted({float(m.variance_epsilon) for nm_, m in rp.mods.items() if hasattr(m, 'variance_epsilon')})
    ctl['calibration_tokens_equal_prepare'] = bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL]))
    ctl['ranges_equal_native'] = rp.ranges == read(NATIVE / 'ranges_by_point.json')
    hook_counts = lambda: Counter((len(rp.mod(n)._forward_pre_hooks), len(rp.mod(n)._forward_hooks)) for n in rp.param)
    ctl['hooks_initial'] = {str(k): v for k, v in hook_counts().items()}
    # weights and activation maps per configuration
    t1 = time.perf_counter()
    wh, amap = {}, {}
    for nm in ('A', 'B', 'C', 'D'):
        rp.set(cf[nm])
        wh[nm] = weights_hash(rp)
        amap[nm] = all(rp.act[n] == cf[nm][n]['a'] for n in rp.param) and set(rp.param) == set(cf[nm])
    rp.set({})
    wh['FP32'] = weights_hash(rp)
    amap['FP32'] = all(rp.act[n] == 'fp32' for n in rp.param)
    ctl['quantized_weights_sha256'] = wh
    ctl['weights_identical_A_to_D'] = len({wh[k] for k in 'ABCD'}) == 1 and wh['A'] != wh['FP32']
    ctl['activation_map_equals_config'] = amap
    ctl['weight_quantization_s'] = time.perf_counter() - t1
    # actual formats at the L2 / L27 Down input and output
    seen = {}

    def chk_pre(key):
        def hook(mod, inp):
            seen.setdefault(key, []).append(('x', inp[0].detach()))
        return hook

    def chk_post(key):
        def hook(mod, inp, outp):
            seen.setdefault(key, []).append(('y', outp.detach()))
        return hook
    diag = []
    for k, n in NODES.items():
        m = rp.mod(n)
        diag += [m.register_forward_pre_hook(chk_pre(k)), m.register_forward_hook(chk_post(k))]
    vals = rp.lut['p8e1'].t_values

    def fmt_of(t):
        f16_fixed = bool(torch.equal(t.to(torch.float16).to(t.dtype), t))
        p8_fixed = bool(torch.equal(rp.lut['p8e1'](t), t))
        return 'p8e1' if p8_fixed else 'f16' if f16_fixed else 'fp32'
    fmt_res = {}
    with torch.no_grad():
        for nm in ('FP32', 'A', 'B', 'C', 'D'):
            rp.set({} if nm == 'FP32' else cf[nm])
            seen.clear()
            rp.ad(torch.from_numpy(toks[:2]).to(rp.device))
            for k, n in NODES.items():
                exp = 'fp32' if nm == 'FP32' else cf[nm][n]['a']
                got = {slot: fmt_of(t) for slot, t in seen[k]}
                fmt_res[f'{nm}|{k}'] = {'expected': exp, 'x': got['x'], 'y': got['y'], 'ok': got['x'] == exp and got['y'] == exp}
    for hk in diag:
        hk.remove()
    ctl['down_hook_formats'] = fmt_res
    ctl['hooks_after_checks'] = {str(k): v for k, v in hook_counts().items()}
    # FP32 identity vs an unhooked model
    from transformers import Qwen3ForCausalLM
    import qwen3_swiglu_common as q3
    plain = Qwen3ForCausalLM.from_pretrained(q3.MODEL_DIR, dtype=torch.float32, attn_implementation='eager').eval().to(rp.device)
    plain.config.use_cache = False

    def ce_plain(t):
        x = torch.from_numpy(t).to(rp.device)
        r = []
        with torch.no_grad():
            for i in range(0, len(x), 8):
                z = plain(input_ids=x[i:i + 8], attention_mask=rp.ad.mask, use_cache=False).logits
                r.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), x[i:i + 8, 1:].reshape(-1),
                                                           reduction='none').view(z.shape[0], -1).double().sum(1))
        return torch.cat(r).cpu().numpy()
    base = ce_plain(toks)
    del plain
    torch.cuda.empty_cache()
    t1 = time.perf_counter()
    runs = defaultdict(list)
    for order in (('FP32', 'A', 'B', 'C', 'D', 'A'), ('D', 'C', 'B', 'A', 'FP32', 'A')):
        for nm in order:
            rp.set({} if nm == 'FP32' else cf[nm])
            runs[nm].append(rp.ce(toks))
    ctl['repeat_order_s'] = time.perf_counter() - t1
    ctl['fp32_identity_equals_unhooked'] = bool(all(np.array_equal(r, base) for r in runs['FP32']))
    ctl['reproducible_bitwise'] = {nm: bool(all(np.array_equal(r, v[0]) for r in v)) for nm, v in runs.items()}
    ctl['repeats'] = {nm: len(v) for nm, v in runs.items()}
    ctl['hooks_final'] = {str(k): v for k, v in hook_counts().items()}
    ctl['pytorch_calibration_summed_ce'] = {nm: v[0].tolist() for nm, v in runs.items()}
    ctl['pytorch_calibration_nll'] = {nm: float(v[0].sum() / (CAL * TP)) for nm, v in runs.items()}
    ctl['passed'] = bool(ctl['tf32'] == [False, False] and ctl['deterministic'] and ctl['posit_lut_verified']['p8e1'][0] == 0
                         and ctl['rms_norm_eps'] == [1e-6] and ctl['calibration_tokens_equal_prepare'] and ctl['ranges_equal_native']
                         and ctl['weights_identical_A_to_D']
                         and all(amap.values()) and all(v['ok'] for v in fmt_res.values())
                         and ctl['hooks_initial'] == ctl['hooks_after_checks'] == ctl['hooks_final'] == {'(1, 1)': len(rp.param)}
                         and ctl['fp32_identity_equals_unhooked'] and all(ctl['reproducible_bitwise'].values()))
    write(out / 'controls_gpu.json', ctl)
    log_stage(out, 'controls', time.perf_counter() - t0, {'replay_init_incl_lut_s': init_s, 'weight_quantization_s': ctl['weight_quantization_s'],
                                                          'repeat_order_s': ctl['repeat_order_s']})
    log(f'controls passed={ctl["passed"]}; ' + json.dumps({k: v for k, v in ctl.items() if k not in ('environment', 'pytorch_calibration_summed_ce', 'down_hook_formats')}))
    if not ctl['passed']:
        raise SystemExit('GPU controls failed')


# --------------------------------------------------------------------------- #
# native cross-check (CPU, onnx-mlir), as run_lp_pipeline_eval.stage_c_native
# --------------------------------------------------------------------------- #
def native(out, log):
    import onnx
    from evaluate_native_search_chunked import chunked_run_so
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm, imported_model, qwen3_spec
    t0 = time.perf_counter()
    pl = check_plan(out)
    cg = read(out / 'controls_gpu.json')
    if not cg['passed']:
        raise SystemExit('GPU controls first')
    cf = configs(out)
    nd = out / 'native'
    nd.mkdir(exist_ok=True)
    toks = np.load(NATIVE / 'calib_inputs.npy')[:NATIVE_WINDOWS]
    np.save(nd / 'tokens.npy', toks)
    tools = erm.Tools()
    base = imported_model(NATIVE)
    graph = onnx.load(str(PREP / 'model.onnx'), load_external_data=False)
    spec = qwen3_spec(nd / 'unused')
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in read(NATIVE / 'ranges_by_point.json').items():
        n = by_name[node]
        for slot, r in slots.items():
            ranges[n.output[0] if slot == 'y' else n.input[SLOT_INPUT[slot]]] = tuple(r)
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    t1 = time.perf_counter()
    posit_obj = erm.build_posit_runtime(tools, nd)
    posit_s = time.perf_counter() - t1
    tm = read(nd / 'timing.json') if (nd / 'timing.json').exists() else {}
    tm['posit_runtime_s'] = posit_s
    res = {'env_entries_modified': {}}
    for nm in NATIVE_CFGS:
        env = None
        if nm != 'FP32':
            env = erm.node_formats_env({n['name']: cf[nm].get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}, spec.nodes, ranges)
            entries = {e.split(':')[0]: e.split(':') for e in env.split(',')}
            bad = [n for n in cf[nm] if f"a={cf[nm][n]['a']}" not in entries[n] or f"w={cf[nm][n]['w']}" not in entries[n]]
            if bad or len(entries) != len(cf[nm]):
                raise SystemExit(f'{nm}: compile environment differs from the configuration at {bad[:3]}')
            res['env_entries_modified'][nm] = {k: ':'.join(entries[n]) for k, n in NODES.items()}
        if (nd / f'{nm}.json').exists():                    # resumable: a finished configuration is not rebuilt
            continue
        t1 = time.perf_counter()
        target = nd / nm
        target.mkdir(exist_ok=True)
        if env:
            (target / 'compile_env.json').write_text(json.dumps({'MIXEDP_NODE_FORMATS': env, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'}) + '\n')
        so, info = erm.build_so(tools, target, nm, base, env, posit_obj if env else None, opt='O0', krnl_opt='O3',
                                constants_to_file=True, bytecode=True, uses_posit=None if env else False, uses_f16=None if env else False)
        build_s = time.perf_counter() - t1
        if info['warnings']:
            raise SystemExit(f'{nm}: compilation warnings {info["warnings"]}')
        for p in target.iterdir():
            if p.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                p.unlink()
        runs = []
        for rep in range(2 if nm == 'A' else 1):
            z, run = chunked_run_so(4)(tools, so, nd / 'tokens.npy', target / 'logits.npy', 36000)
            (target / 'logits.npy').unlink()
            runs.append({'summed_ce': [float(cross_entropy(z[i, :-1], toks[i, 1:]).sum()) for i in range(len(toks))],
                         'finite': bool(np.isfinite(z).all()), 'run_wall_s': run['wall_seconds']})
        for p in (so, target / 'model.constants.bin'):
            p.unlink(missing_ok=True)
        r = {'config': nm, 'summed_ce': runs[0]['summed_ce'], 'finite': runs[0]['finite'], 'build_s': build_s,
             'build_steps': info.get('steps'), 'run_wall_s': [x['run_wall_s'] for x in runs]}
        if nm == 'A':
            r['repeat_summed_ce'] = runs[1]['summed_ce']
            r['repeat_max_abs'] = float(np.abs(np.array(runs[1]['summed_ce']) - np.array(runs[0]['summed_ce'])).max())
        write(nd / f'{nm}.json', r)
        tm[f'{nm}_build_s'], tm[f'{nm}_run_s'] = build_s, sum(r['run_wall_s'])
        write(nd / 'timing.json', tm)
        log(f'native {nm}: NLL {sum(r["summed_ce"]) / (len(toks) * TP):.5f} (build {build_s:.0f}s, run {sum(r["run_wall_s"]):.0f}s)')
    nat = {nm: np.array(read(nd / f'{nm}.json')['summed_ce']) / TP for nm in NATIVE_CFGS}
    pt = {nm: np.array(cg['pytorch_calibration_summed_ce'][nm])[:len(toks)] / TP for nm in NATIVE_CFGS}
    cmp_ = {'windows': int(len(toks)), 'unit': 'nat/token per window (summed CE / 127)', 'absolute': {}, 'paired': {}}
    for nm in nat:
        cmp_['absolute'][nm] = {'native_nll': float(nat[nm].mean()), 'pytorch_nll': float(pt[nm].mean()),
                                'max_abs_window_diff': float(np.abs(nat[nm] - pt[nm]).max())}
    within = True
    for x in [c for c in ('B', 'C', 'D') if c in nat]:
        en, ep = nat[x] - nat['A'], pt[x] - pt['A']
        d = en - ep
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        ok = abs(float(d.mean())) + 2 * se < DELTA
        within &= ok
        cmp_['paired'][f'{x}-A'] = {'native_mean': float(en.mean()), 'native_se': float(en.std(ddof=1) / np.sqrt(len(en))),
                                    'pytorch_mean': float(ep.mean()), 'pytorch_se': float(ep.std(ddof=1) / np.sqrt(len(ep))),
                                    'difference_mean': float(d.mean()), 'difference_se': se, 'max_abs_window_difference': float(np.abs(d).max()),
                                    'sign_agree_windows': int((np.sign(en) == np.sign(ep)).sum()), 'per_window_corr':
                                        float(np.corrcoef(en, ep)[0, 1]) if en.std() > 0 and ep.std() > 0 else None,
                                    'below_delta_rule': bool(ok)}
    cmp_['native_A_repeat_max_abs_summed_ce'] = read(nd / 'A.json')['repeat_max_abs']
    cmp_['backend_consistent_at_delta'] = bool(within and all(c in nat for c in ('B', 'C', 'D')))
    cmp_['scope'] = ('native and PyTorch paired effects agree within delta on the 10 calibration windows (still not a native '
                     'confirmation of the new-article result)' if cmp_['backend_consistent_at_delta'] else
                     'backend difference can change a delta-level reading: every conclusion is limited to the PyTorch QDQ diagnostic')
    cmp_['onnx_mlir_opt_sha256'] = sha(tools.opt)
    cmp_['env_entries_modified'] = res.get('env_entries_modified')
    write(out / 'native_crosscheck.json', cmp_)
    log_stage(out, 'native', time.perf_counter() - t0, tm)
    log(json.dumps({k: v for k, v in cmp_.items() if k in ('absolute', 'paired', 'backend_consistent_at_delta', 'native_A_repeat_max_abs_summed_ce')}))


def freeze(out, log):
    t0 = time.perf_counter()
    pl = check_plan(out)
    if (out / 'freeze.json').exists():
        raise SystemExit('frozen')
    cg, nc = read(out / 'controls_gpu.json'), read(out / 'native_crosscheck.json')
    if not cg['passed']:
        raise SystemExit('controls failed')
    nat_ok = all(read(out / 'native' / f'{nm}.json')['finite'] for nm in ('FP32', 'A', 'B', 'C', 'D'))
    if not nat_ok:
        raise SystemExit('native results not finite')
    write(out / 'controls.json', {'gpu': {k: v for k, v in cg.items() if k != 'pytorch_calibration_summed_ce'}, 'native': nc,
                                  'resources_reproduce_search': read(out / 'resources.json')['control_A_reproduces_search'],
                                  'plan_reference_checks': pl['reference_checks']})
    write(out / 'freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'plan_sha256': sha(out / 'plan.json'),
                                'configs_sha256': pl['configs_sha256'], 'config_hashes': {nm: v['hash'] for nm, v in pl['configs'].items()},
                                'controls_gpu_sha256': sha(out / 'controls_gpu.json'), 'native_crosscheck_sha256': sha(out / 'native_crosscheck.json'),
                                'resources_sha256': sha(out / 'resources.json'), 'backend_scope': nc['scope'],
                                'script_sha256': sha(HERE / 'run_qwen_down_fp16_protection.py')})
    log_stage(out, 'freeze', time.perf_counter() - t0)
    log(f'frozen; backend: {nc["scope"]}')


# --------------------------------------------------------------------------- #
# audit and data
# --------------------------------------------------------------------------- #
def audit(out, log):
    import pyarrow.parquet as pq
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_qwen_down_spike_diag import hist_files, qwen3_tokenizer, window_shingles
    t0 = time.perf_counter()
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    tok3 = qwen3_tokenizer()
    from tokenizers import Tokenizer
    tok35 = Tokenizer.from_file(str(HERE / 'qwen35_assets/model/tokenizer.json'))
    ctl = {}
    for split, f in (('validation', 'validation_tokens.npy'), ('test', 'test_tokens.npy')):
        text = '\n\n'.join(pq.read_table(HERE / 'gpt2_assets/data' / f'{split}-00000-of-00001.parquet')['text'].to_pylist())
        ids = tok3.encode(text, add_special_tokens=False).ids
        ref = np.load(PREP / f)
        ctl[f'tokenizer_reproduces_prepare_{split}'] = bool(np.array_equal(np.array(ids[:ref.size]).reshape(ref.shape), ref))
    if not all(ctl.values()):
        raise SystemExit(f'tokenizer control failed {ctl}')
    U2 = set()
    for f in mr.WT2:
        U2 |= window_shingles(' '.join(pq.read_table(f)['text'].to_pylist()))
    w103, _ = wms.corpora()
    by_ch = {a['ch']: a for a in w103}
    rows, idx_sh, idx_win, wid = [], [], [], 0
    for p, rel, shape, mx in hist_files(out):
        a = np.load(p)
        row = {'path': rel, 'sha256': sha(p), 'shape': list(shape), 'max_id': mx}
        if 'tiny_check' in rel and mx < 1024:
            row.update(tokenizer='none', resolved='synthetic (tiny-model ids < 1024, not text)', windows=int(shape[0]))
            rows.append(row)
            continue
        tk, name = (tok35, 'qwen3.5') if 'qwen35' in rel.lower() else (tok3, 'qwen3')
        if name == 'qwen3' and mx >= 151936:
            raise SystemExit(f'{rel}: id {mx} outside the Qwen3 vocabulary')
        seg = p.parent / 'segments.json'
        listed = [mr.shingles(by_ch[s['content_sha256']]['body_norm']) for s in read(seg) if s.get('content_sha256') in by_ch] if seg.exists() else []
        res_ = Counter()
        shares = []
        for w in a:
            sh = window_shingles(tk.decode(w.tolist()))
            for s_ in sh:
                idx_sh.append(s_)
                idx_win.append(wid)
            wid += 1
            if len(sh) < 10:
                res_['short'] += 1
                continue
            s2 = len(sh & U2) / len(sh)
            shares.append(s2)
            if s2 >= 0.5:
                res_['wikitext-2'] += 1
            elif listed and max(len(sh & o) / len(sh) for o in listed) >= 0.5:
                res_['wikitext-103 listed in the run manifest'] += 1
            else:
                res_['unresolved'] += 1
        row.update(tokenizer=name, windows=int(shape[0]), window_sources=dict(res_), wt2_share_min=float(min(shares)) if shares else None,
                   resolved='unresolved' if res_['unresolved'] else ' + '.join(sorted(k for k in res_ if k != 'short')))
        rows.append(row)
    np.savez_compressed(out / 'work' / 'qwen_hist_shingles.npz', shingle=np.array(idx_sh, np.uint64), window=np.array(idx_win, np.int64))
    res = {'controls': ctl, 'files': rows, 'windows_indexed': wid, 'unresolved_files': [r['path'] for r in rows if r['resolved'] == 'unresolved'],
           'index_sha256': sha(out / 'work' / 'qwen_hist_shingles.npz'),
           'rule': 'a window is WikiText-2 if >= 50% of its word 5-gram shingles are in the WikiText-2 text, else WikiText-103 if '
                   '>= 50% are in an article listed (content hash) in the segments.json of its run; every window enters the '
                   'overlap exclusion of the data stage'}
    write(out / 'audit.json', res)
    log_stage(out, 'audit', time.perf_counter() - t0)
    log(f'audit: {len(rows)} files, {wid} windows; unresolved {res["unresolved_files"]}; ' +
        json.dumps(Counter(r['resolved'] for r in rows)))


def data(out, log):
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_qwen_down_spike_diag import qwen3_tokenizer
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
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    if len(used) != pl['data']['excluded_previous_wt103']:
        raise SystemExit('previous WT103 set changed since the plan')
    spike = {s['content_sha256'] for s in read(SPIKE / 'segments.json')}
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
    got = {s_['content_sha256'] for s_ in segs}
    if len(segs) < N_ART or got & used or got & spike:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'exclusions.json', {'skipped_during_draw': excl, 'previous_wt103_excluded_by_scan': sorted(used),
                                    'spike_diag_articles_excluded': sorted(spike),
                                    'wt2_and_internal_duplicates_excluded_before_draw': len(w103) - len(base)})
    write(out / 'data_freeze.json', {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'freeze_sha256': sha(out / 'freeze.json'),
                                     'audit_sha256': sha(out / 'audit.json'), 'n_articles': N_ART, 'windows_per_article': WPA,
                                     'eligible': len(elig), 'visited': N_ART + sum(skips.values()), 'skips': dict(skips),
                                     'excluded_previous_wt103': len(used), 'tokens_sha256': sha(out / 'tokens.npy'),
                                     'segments_sha256': sha(out / 'segments.json'), 'exclusions_sha256': sha(out / 'exclusions.json')})
    log_stage(out, 'data', time.perf_counter() - t0)
    log(f'data: {N_ART} articles; skips {dict(skips)}')


# --------------------------------------------------------------------------- #
# eval (GPU) and analyze
# --------------------------------------------------------------------------- #
def eval_(out, log):
    t0 = time.perf_counter()
    pl = check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'segments.json') != dz['segments_sha256']:
        raise SystemExit('freeze / data changed')
    cf = configs(out)
    rp, init_s = replay(out / 'work')
    toks = np.load(out / 'tokens.npy')
    ce, secs = {}, {}
    for nm in ('FP32', 'A', 'B', 'C', 'D'):
        t1 = time.perf_counter()
        rp.set({} if nm == 'FP32' else cf[nm])
        if nm != 'FP32' and not all(rp.act[n] == cf[nm][n]['a'] for n in rp.param):
            raise SystemExit('activation map differs from the configuration')
        ce[nm] = rp.ce(toks)
        secs[nm] = time.perf_counter() - t1
        np.savez(out / 'work' / 'eval_partial.npz', **ce)
        log(f'{nm}: NLL {ce[nm].sum() / (len(toks) * TP):.6f} ({secs[nm]:.0f}s)')
    np.savez(out / 'eval_ce.npz', **ce)
    log_stage(out, 'eval', time.perf_counter() - t0, {'replay_init_s': init_s, 'per_config_s': secs, 'windows': int(len(toks)),
                                                      'peak_gpu_memory_bytes': int(rp.torch.cuda.max_memory_allocated()),
                                                      'eval_ce_sha256': sha(out / 'eval_ce.npz')})


def analyze(out, log):
    t0 = time.perf_counter()
    pl = check_plan(out)
    fz = read(out / 'freeze.json')
    z = np.load(out / 'eval_ce.npz')
    ce = {k: z[k] for k in z.files}
    segs = read(out / 'segments.json')
    na = len(segs)
    art = {k: v.reshape(na, WPA).sum(1) / (WPA * TP) for k, v in ce.items()}
    with open(out / 'per_window_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['window', 'article', 'title', *[f'ce_{k}' for k in ('FP32', 'A', 'B', 'C', 'D')]])
        for w in range(len(ce['A'])):
            wr.writerow([w, w // WPA, segs[w // WPA]['title'], *[ce[k][w] for k in ('FP32', 'A', 'B', 'C', 'D')]])
    with open(out / 'per_article_losses.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['article', 'title', 'content_sha256', *[f'nll_{k}' for k in ('FP32', 'A', 'B', 'C', 'D')], 'd_B', 'd_C', 'd_D', 'I'])
        for a in range(na):
            wr.writerow([a, segs[a]['title'], segs[a]['content_sha256'], *[art[k][a] for k in ('FP32', 'A', 'B', 'C', 'D')],
                         art['B'][a] - art['A'][a], art['C'][a] - art['A'][a], art['D'][a] - art['A'][a],
                         art['D'][a] - art['B'][a] - art['C'][a] + art['A'][a]])
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]).astype(np.float64) / na
    lv = ALPHA / K

    def st(x, levels=(ALPHA, lv)):
        b = Wb @ x
        r = {'mean': float(x.mean()), 'sd_articles': float(x.std(ddof=1)), 'se': float(x.std(ddof=1) / np.sqrt(len(x))),
             'articles_positive': int((x > 0).sum()), 'articles_negative': int((x < 0).sum())}
        for l_ in levels:
            r[f'ci{100 * (1 - l_):.3f}'] = [float(np.percentile(b, 100 * l_ / 2)), float(np.percentile(b, 100 * (1 - l_ / 2)))]
        return r
    res = {'backend': 'GPU PyTorch QDQ replay (LMReplay; TF32 off, deterministic)', 'backend_scope': fz['backend_scope'],
           'articles': na, 'windows': int(len(ce['A'])), 'delta': DELTA, 'bootstrap': BOOT,
           'nll': {k: float(v.mean()) for k, v in art.items()}, 'ppl': {k: float(np.exp(v.mean())) for k, v in art.items()},
           'vs_fp32': {k: st(art[k] - art['FP32'], (ALPHA,)) for k in ('A', 'B', 'C', 'D')}, 'comparisons': {}}
    ck = f'ci{100 * (1 - lv):.3f}'
    for x in ('B', 'C', 'D'):
        s = st(art[x] - art['A'])
        lo, hi = s[ck]
        if hi < 0:
            dec = 'quality improvement supported (corrected upper < 0)'
        elif hi < DELTA:
            dec = ('small quality drop within the preset tolerance (corrected CI all positive, upper < delta)' if lo > 0 else
                   'added loss within the tolerance (corrected upper < delta)')
        elif lo > DELTA:
            dec = 'added loss exceeds the tolerance (corrected lower > delta)'
        else:
            dec = 'undetermined for non-inferiority'
        s['decision'] = dec
        s['corrected_level'] = ck
        res['comparisons'][f'{x}-A'] = s
    res['interaction'] = st(art['D'] - art['B'] - art['C'] + art['A'], (ALPHA,))
    res['interaction']['reading'] = ('I > 0 confirmed' if res['interaction']['ci95.000'][0] > 0 else
                                     'I < 0 confirmed' if res['interaction']['ci95.000'][1] < 0 else 'interaction not confirmed (not proof of additivity)')
    res['window_level_sd'] = {f'{x}-A': float(((ce[x] - ce['A']) / TP).std(ddof=1)) for x in ('B', 'C', 'D')}
    rs = read(out / 'resources.json')['configs']
    res['resources'] = {nm: {k: rs[nm][k] for k in ('size', 'bops_rel', 'resource', 'dominant', 'objective')} | {'vs_A': rs[nm]['vs_A']} for nm in rs}
    write(out / 'stats.json', res)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({k: res[k] for k in ('nll', 'comparisons', 'interaction')}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'controls', 'native', 'freeze', 'audit', 'data', 'eval', 'analyze'))
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
    {'plan': plan, 'controls': controls, 'native': native, 'freeze': freeze, 'audit': audit, 'data': data, 'eval': eval_,
     'analyze': analyze}[args.stage](out, log)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': args.stage, 'seconds': time.time() - t0, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                             'script_sha256': sha(HERE / 'run_qwen_down_fp16_protection.py')}) + '\n')


if __name__ == '__main__':
    main()
