#!/usr/bin/env python3
"""Qwen3-0.6B: native (onnx-mlir) independent confirmation of the L27 Down activation f16 -> p8e1 replacement.

Fixed complete configurations from runs/qwen3_down_fp16_protection_20261024: A (original BC knee) and C (only the L27
Down `a` f16 -> p8e1; `a` quantizes the projection input AND output). No criterion change, no search, no knee
re-selection, no fine-tuning, no calibration change, no pipeline change. Primary evidence: native C - A on 128 new
WikiText-103 articles (one 128-token window each); secondary: GPU PyTorch QDQ (LMReplay) on the same windows.

Native runs: one compiled .so per subprocess (OM_CONSTANT_PATH isolation), batch 1, float32 logits streamed into
run_native_bc_pipeline.cross_entropy (float64 log-sum-exp, 127 next-token positions); per-window checkpoints with a
header of hashes (tokens, .so, constants, script, configuration, index range) that a resumed run must match.

Stages (one new --out-dir): plan, build, controls, audit, data, native_eval, pytorch_eval, analyze (worker: internal).
CPU stages: ../../venv/bin/python; pytorch_eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PREV = HERE / 'runs/qwen3_down_fp16_protection_20261024'
SPIKE = HERE / 'runs/qwen3_down_spike_diag_20261023'
NATIVE = HERE / 'runs/native_eps_fix_20261014/qwen3'
PREP = HERE / 'runs/qwen3_prepare_20261011'
L27 = '/model/model/layers.27/mlp/down_proj/MatMul'
L2 = '/model/model/layers.2/mlp/down_proj/MatMul'
N_ART, WPA, SEQ, TP, BOOT, VOCAB = 128, 1, 128, 127, 20000, 151936
DELTA, TOL = 0.001, 1e-8
WORKERS = 4
SEEDS = {'order': 29101, 'start': 29102, 'bootstrap': 29103}
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE'}
CODE = [Path(__file__).name, 'run_qwen_down_fp16_protection.py', 'run_qwen_down_spike_diag.py', 'run_lp_pipeline_eval.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'run_native_bc_pipeline.py', 'wt_manifest_scan.py',
        'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py', 'run_solver_regression.py', 'mp_auto.py',
        'htq_pareto_search.py']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


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


def check_plan(out, inputs=True):
    p = read(out / 'plan.json')
    for name, h in p['code_hashes'].items():
        if sha(HERE / name) != h:
            raise SystemExit(f'Code changed after registration: {name} (record an amendment first)')
    for name, h in p['configs_sha256'].items():
        if sha(out / 'configs' / f'{name}.json') != h:
            raise SystemExit(f'Config changed: {name}')
    if inputs:
        for path, h in p['inputs'].items():
            if sha(HERE / path) != h:
                raise SystemExit(f'Input changed: {path}')
    return p


def tool_files():
    import eval_random_mixed_precision as erm
    t = erm.Tools()
    return t, {'onnx_mlir': t.onnx_mlir, 'onnx_mlir_opt': t.opt, 'mlir_translate': t.translate, 'clangxx': t.clangxx,
               'cruntime': t.cruntime, 'posit_runtime_src': t.posit_rt, 'pyruntime_py': t.lib / 'PyRuntime.py',
               'pyruntime_c': next(t.lib.glob('PyRuntimeC*.so'))}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out):
    import wt_manifest_scan as wms
    from collections import Counter
    import run_qwen_down_fp16_protection as src
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('Already registered; never overwrite a plan')
    for folder in ('configs', 'work', 'native'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name in ('A', 'C'):
        shutil.copyfile(PREV / 'configs' / f'{name}.json', out / 'configs' / f'{name}.json')
    a, c = [read(out / 'configs' / f'{n}.json') for n in ('A', 'C')]
    pp = read(PREV / 'plan.json')
    changed = [n for n in a if a[n] != c[n]]
    chk = {'configs_equal_previous_files': all(sha(out / 'configs' / f'{n}.json') == pp['configs_sha256'][n] for n in ('A', 'C')),
           'config_hashes_equal_previous': all(src.config_hash(x) == pp['configs'][n]['hash'] for n, x in (('A', a), ('C', c))),
           'projections': len(a), 'same_nodes': set(a) == set(c),
           'only_difference_L27_down_a': changed == [L27] and a[L27] == {'w': 'int8', 'a': 'f16'} and c[L27] == {'w': 'int8', 'a': 'p8e1'},
           'all_weights_equal': all(a[n]['w'] == c[n]['w'] for n in a),
           'L2_down_f16_in_both': a[L2] == c[L2] == {'w': 'int8', 'a': 'f16'}}
    if not (all(v for k, v in chk.items() if k != 'projections') and chk['projections'] == 196):
        raise SystemExit(f'configuration check failed: {chk}')
    t1 = time.perf_counter()
    res = src.resources({'A': a, 'C': c}, out / 'work')
    prev = read(PREV / 'resources.json')['configs']
    keys = ('weight_bytes', 'bops', 'size', 'bops_rel', 'resource', 'objective')
    res['check_vs_previous_round'] = {n: {k: [res['configs'][n][k], prev[n][k]] for k in keys} for n in ('A', 'C')}
    res['check_passed'] = bool(res['control_A_reproduces_search']['passed'] and
                               all(res['configs'][n][k] == prev[n][k] for n in ('A', 'C') for k in keys))
    res['seconds'] = time.perf_counter() - t1
    write(out / 'resources.json', res)
    if not res['check_passed']:
        raise SystemExit('resource model does not reproduce the previous round')
    w103, wt2 = wms.corpora()
    used, manifests = wms.used_wt103_verified(out, w103, wt2)
    t2, h2 = {x['nt'] for x in wt2}, {x['ch'] for x in wt2}
    nt, nh = Counter(x['nt'] for x in w103), Counter(x['ch'] for x in w103)
    eligible = [x for x in w103 if x['nt'] not in t2 and x['ch'] not in h2 and nt[x['nt']] == nh[x['ch']] == 1 and x['ch'] not in used]
    prev512 = {x['content_sha256'] for x in read(PREV / 'segments.json')}
    spike128 = {x['content_sha256'] for x in read(SPIKE / 'segments.json')}
    chk['previous_512_in_verified_scan'] = len(prev512) == 512 and prev512 <= used
    chk['spike_128_in_verified_scan'] = len(spike128) == 128 and spike128 <= used
    if not (chk['previous_512_in_verified_scan'] and chk['spike_128_in_verified_scan']):
        raise SystemExit(f'isolation check failed: {chk}')
    _, tf = tool_files()
    paths = [PREV / 'configs/A.json', PREV / 'configs/C.json', PREV / 'plan.json', PREV / 'resources.json', PREV / 'segments.json',
             PREV / 'tokens.npy', PREV / 'native/A.json', PREV / 'native/C.json', PREV / 'native/A/compile_env.json',
             PREV / 'native/C/compile_env.json', PREV / 'native_crosscheck.json', PREV / 'stats.json', SPIKE / 'segments.json',
             NATIVE / 'calib_inputs.npy', NATIVE / 'model.mlirbc', NATIVE / 'ranges_by_point.json', NATIVE / 'table.json',
             PREP / 'model.onnx', PREP / 'model.onnx.data', PREP / 'node_mapping.json', PREP / 'validation_tokens.npy',
             PREP / 'test_tokens.npy', HERE / 'qwen3_assets/model/model.safetensors', HERE / 'qwen3_assets/model/config.json',
             HERE / 'qwen3_assets/model/tokenizer.json', HERE / 'qwen35_assets/model/tokenizer.json', *tf.values()]
    import run_olmoe_route_mass_restore as mr
    paths += mr.WT103 + mr.WT2
    rel = lambda x: os.path.relpath(x, HERE)
    pw = read(PREV / 'stats.json')
    pn = read(PREV / 'native_crosscheck.json')['paired']['C-A']
    sd_pt, sd_nat = pw['window_level_sd']['C-A'], pn['native_se'] * np.sqrt(10)
    mde = {k: float(2.8016 * v / np.sqrt(N_ART)) for k, v in (('pytorch_window_sd', sd_pt), ('native_calibration_sd', sd_nat))}
    p = {'time': now(),
         'question': 'Does replacing only the L27 Down activation f16 with p8e1 (input and output) improve native quality on '
                     'independent articles, as the PyTorch QDQ diagnostic of the previous round found?',
         'scope': 'fixed complete configurations A and C of the previous round; no criterion / search / knee / fine-tuning / '
                  'calibration / pipeline change; no L2 replacement, no Posit6 or other candidate; FP32 not required',
         'checks': chk,
         'configs': {'A': {'source': rel(PREV / 'configs/A.json'), 'hash': src.config_hash(a)},
                     'C': {'source': rel(PREV / 'configs/C.json'), 'hash': src.config_hash(c)},
                     'difference': {'node': L27, 'field': 'a', 'A': 'f16', 'C': 'p8e1', 'w': 'int8 in both',
                                    'semantics': 'a quantizes the projection input (x range) and output (y range)'}},
         'configs_sha256': {n: sha(out / 'configs' / f'{n}.json') for n in ('A', 'C')},
         'resources': {n: {k: res['configs'][n][k] for k in keys + ('dominant',)} for n in ('A', 'C')} |
                      {'C_vs_A': res['configs']['C']['vs_A'], 'scope': 'searched range: the 196 transformer projections',
                       'statement': 'BOPs is a proxy; not a measured kernel speed-up; weight bytes unchanged'},
         'compile': {'krnl': 'O3', 'clang': 'O0', 'constants_to_file': True, 'bytecode': True, 'MIXEDP_WEIGHT_PER_CHANNEL': '1',
                     'base': rel(NATIVE / 'model.mlirbc'), 'ranges': rel(NATIVE / 'ranges_by_point.json'),
                     'env': 'MIXEDP_NODE_FORMATS regenerated from each configuration (erm.node_formats_env, native ranges) '
                            'must equal the previous round\'s compile_env.json exactly; every one of the 196 entries checked',
                     'retention': '.so, model.constants.bin, compile_env.json, build steps and hashes kept after the run'},
         'controls': ['config diff: only L27 Down a', 'compile environment of all 196 projections equals the configuration',
                      f'calibration windows 0-1: native A / C summed CE equal the previous native/A.json, C.json within {TOL}',
                      f'A window 0 again in another fresh process within {TOL}',
                      'failure: locate and fix (amendment); never widen the tolerance; no formal comparison before passing'],
         'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'sequence': SEQ, 'excluded_previous_wt103': len(used),
                  'eligible_before_length_and_neardup': len(eligible),
                  'rule': 'WikiText-103 train strict-boundary articles; not a WikiText-2 title / content hash; unique; not in '
                          'wt_manifest_scan.used_wt103_verified (includes the 128 of qwen3_down_spike_diag and the 512 of '
                          'qwen3_down_fp16_protection); skip >= 10% shingle share with one WikiText-2 article, a previous '
                          'WikiText-103 article or an accepted article; skip >= 10% overlap with any historical Qwen token '
                          'window (audit rerun on the current runs/; an unresolved source stops the stage); one window per '
                          'article, start uniform; Qwen3 tokenizer',
                  'sampler': 'run_qwen_down_fp16_protection.audit / data with N_ART=128, WPA=1 and this round\'s seeds',
                  'controls_windows': 'calibration windows are controls only, never pooled with the new articles',
                  'no_pooling': 'not pooled with the previous 512 x 3 windows for the primary inference'},
         'seeds': SEEDS,
         'execution': {'workers': WORKERS, 'threads_per_worker': THREAD_ENV, 'isolation': 'one compiled .so per subprocess',
                       'loss': 'float32 native logits -> run_native_bc_pipeline.cross_entropy (float64 log-sum-exp), 127 '
                               'next-token positions, summed',
                       'resume': 'per-window JSON lines after a header of tokens / .so / constants / script hashes, '
                                 'configuration and index range; a resumed worker must match the header; a truncated last '
                                 'line is discarded and recomputed; complete.json only when every window is present once; '
                                 'no window is skipped for any reason',
                       'memory': 'VmHWM per worker recorded'},
         'statistics': {'article_nll': 'summed CE of the window / 127', 'primary': 'd_native = NLL_native(C) - NLL_native(A)',
                        'bootstrap': f'paired article bootstrap, {BOOT} replicates, seed {SEEDS["bootstrap"]}',
                        'interval': 'two-sided 95% percentile CI (one primary comparison, no multiplicity correction)',
                        'delta': f'{DELTA} nat/token (previous round)',
                        'decision': {'upper < 0': 'native quality improvement supported',
                                     'lower > 0 and upper < delta': 'small quality drop, within the preset tolerance',
                                     '0 <= upper < delta otherwise': 'native non-inferiority supported; improvement not confirmed',
                                     'lower > delta': 'added loss exceeds the tolerance',
                                     'otherwise': 'undetermined; keep A'},
                        'notes': 'a CI covering 0 is not equivalence; no significant improvement is not proof of no gain',
                        'secondary': 'PyTorch QDQ A / C on the identical windows; d_pytorch and b = d_native - d_pytorch per '
                                     'article: mean, paired 95% CI, correlation of the two effects, sign agreement; does not '
                                     'change the native reading; no backend is called buggy from a difference'},
         'power_context': {'sd_assumptions': {'pytorch_window_sd_C-A_previous': sd_pt, 'native_calibration_window_sd_C-A': sd_nat},
                           'minimum_detectable_effect_80pct_two_sided': mde,
                           'expected': 'previous PyTorch article effect -0.0085 (3 windows); native on 10 calibration windows '
                                       '-0.0018 +- 0.0045. If the native effect is near -0.002 the chance to confirm an '
                                       'improvement is low; the fixed N is a cost decision, not a power guarantee'},
         'decision_rule': 'even if native succeeds: record as an independent confirmation of a selected case; no change to the '
                          'formal configuration, criterion or pipeline; automatic identification from calibration and '
                          'cross-layer / cross-model validity are later studies. If not confirmed: keep the original '
                          'configuration; no added samples, no other layer as a substitute.',
         'pre_plan_observations': 'previous reports and outputs only; before this plan a scratch smoke run (outside runs/) '
                                  'built A / C, ran the control stage, a resume / lock test and native / PyTorch evaluation on '
                                  'calibration windows 0-9 only (all of them already measured natively and in PyTorch by the '
                                  'previous round), and exercised audit / data with other seeds (999201 / 999202, 4 articles, '
                                  'never evaluated); fixes: none to the design, one added (a lock against two writers of one '
                                  'checkpoint); no new article was evaluated',
         'code_hashes': {n: sha(HERE / n) for n in CODE}, 'inputs': {rel(x): sha(x) for x in paths}}
    write(out / 'plan.json', p)
    write(out / 'historical_manifests.json', manifests)
    r = p['resources']
    md = ['# 計畫：Qwen3-0.6B L27 Down activation 的 native 獨立確認', '',
          f'時間：{p["time"]}。以 plan.json 為準；寫入後的程式修改須記錄 amendment。', '',
          '- 問題：只把 L27 Down 的 activation 由 f16 改成 p8e1（input 與 output 都量化），在 onnx-mlir 編譯模型、新文章上是否重現前輪 PyTorch QDQ 的品質改善。',
          '- 不改 criterion、不搜尋、不重選 knee、不 fine-tune、不換 calibration、不改 pipeline；不測 L2、不加 Posit6 或其他候選；FP32 非必要條件。',
          f'- 配置：A、C 直接取自前輪（檔案 sha256 與前輪相同；hash A {p["configs"]["A"]["hash"][:12]}…、C {p["configs"]["C"]["hash"][:12]}…）；唯一差異 {L27} 的 a：f16→p8e1，w 都是 int8；L2 Down 仍為 f16。',
          f'- 資源（原成本函式重算並與前輪逐值核對）：A size {r["A"]["size"]:.6f}、BOPs {r["A"]["bops_rel"]:.6f}、R {r["A"]["resource"]:.6f}；C size {r["C"]["size"]:.6f}、BOPs {r["C"]["bops_rel"]:.6f}、R {r["C"]["resource"]:.6f}；'
          f'搜尋範圍總 BOPs {r["C_vs_A"]["bops_pct_of_A_total"]:+.3f}%，weight bytes 不變。BOPs 是 proxy。',
          '- 編譯：Krnl O3、clang O0、constants to file、MLIR bytecode、MIXEDP_WEIGHT_PER_CHANNEL=1、native_eps_fix 的 model.mlirbc 與 ranges；'
          '由配置重新產生 MIXEDP_NODE_FORMATS，須與前輪 compile_env.json 逐字相同（196 個條目逐一核對）。保留 .so、constants、編譯紀錄與 hash。',
          f'- 控制：config diff 只有 L27 Down a；calibration 窗口 0–1 的 native A／C summed CE 須重現前輪（≤ {TOL}）；A 的窗口 0 在另一個新 process 重跑（≤ {TOL}）。失敗先定位，不放寬容差。',
          f'- 執行：{WORKERS} 個隔離 worker，每個 process 只載入一個 .so，OMP／MKL／BLAS threads = 1；float32 logits 串流進 run_native_bc_pipeline.cross_entropy（float64），127 個位置；'
          '逐窗口 checkpoint，標頭含 tokens／.so／constants／程式 hash、配置與窗口範圍，續跑必須相符；截斷的最後一行丟棄重算；不跳過任何窗口；記錄 VmHWM。',
          f'- 資料：{N_ART} 篇新 WT103 文章 × {WPA} 個 128-token 窗口（Qwen3 tokenizer）；verified 掃描排除 {len(used)} 篇（含 spike 診斷 128 篇、前輪 512 篇）；排除 WT2 重複／近重複、歷史 Qwen 窗口重疊、本輪內部近重複；'
          f'歷史來源無法解析即停止；seed：順序 {SEEDS["order"]}、起點 {SEEDS["start"]}；控制窗口不計入正式評估；不與前輪 512×3 合併做主要推論。',
          f'- 主要統計：每篇 NLL = summed CE / 127；d_native = C − A；paired 文章 bootstrap {BOOT} 次（seed {SEEDS["bootstrap"]}），雙尾 95% CI；δ = {DELTA}。',
          '  上界 < 0 → 支持 native 品質改善；CI 全正且上界 < δ → 小幅變差但在容許內；上界 < δ（其他）→ 支持非劣、改善未確認；下界 > δ → 超過容許；其餘 → 未定，保留 A。CI 跨 0 不等於等效；未顯著不等於沒有收益。',
          '- 次要：同一批窗口的 PyTorch QDQ A／C（修正後量化器、TF32 關閉、deterministic、FP32）；b = d_native − d_pytorch 的平均與 95% CI、兩後端效應相關、逐文章符號一致率。不改變主要判讀，也不據差異宣稱某後端有錯。',
          f'- 檢定力背景（估計，非保證）：80% 檢定力可偵測約 {mde["native_calibration_sd"]:.4f}（native calibration SD）至 {mde["pytorch_window_sd"]:.4f}（PyTorch 逐窗口 SD）nat/token；前輪 native 在 calibration 上只有 −0.0018 ± 0.0045，若 native 效應約 −0.002，確認改善的機率低。樣本數固定，不依途中結果增減。',
          '- 決策：即使 native 成功，只記為已選定案例的獨立確認，不改正式配置或 criterion；未確認則保留原配置，不加樣本、不以其他層替代。',
          '- 計畫前觀察：只讀前輪報告與輸出；另在 scratchpad（不在 runs/）做冒煙測試：建 A／C，跑控制階段、續跑與鎖定測試，以及 native／PyTorch 在 calibration 窗口 0–9 上的評估（前輪皆已量過）；'
          '以其他 seed（999201／999202）測試 audit／data，抽 4 篇未評估。設計未改，只加入「同一 checkpoint 只允許一個寫入者」的鎖。沒有評估任何新文章。', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0, {'resources_s': res['seconds']})
    log(f'Registered {N_ART} articles; excluded {len(used)} previous; checks {chk}')


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(out):
    import onnx
    import eval_random_mixed_precision as erm
    from run_native_bc_pipeline import SLOT_INPUT, qwen3_spec
    check_plan(out)
    t0 = time.perf_counter()
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    tools, tf = tool_files()
    root = out / 'native'
    root.mkdir(exist_ok=True)
    spec = qwen3_spec(out / 'work' / 'spec')
    graph = onnx.load(str(PREP / 'model.onnx'), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in read(NATIVE / 'ranges_by_point.json').items():
        n = by_name[node]
        for slot, r in slots.items():
            ranges[n.output[0] if slot == 'y' else n.input[SLOT_INPUT[slot]]] = tuple(r)
    t1 = time.perf_counter()
    obj = erm.build_posit_runtime(tools, root)
    builds = {'posit_runtime': {'object': os.path.relpath(obj, HERE), 'sha256': sha(obj), 'seconds': time.perf_counter() - t1},
              'tools_sha256': {k: sha(v) for k, v in tf.items()}, 'flags': {'krnl_opt': 'O3', 'opt': 'O0', 'constants_to_file': True,
                                                                           'bytecode': True, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'}}
    env_check = {}
    for name in ('A', 'C'):
        target = root / name
        target.mkdir(exist_ok=True)
        cfg = read(out / 'configs' / f'{name}.json')
        env_new = erm.node_formats_env({n['name']: cfg.get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}, spec.nodes, ranges)
        prev = read(PREV / 'native' / name / 'compile_env.json')
        entries = {x.split(':')[0]: x.split(':')[1:] for x in env_new.split(',')}
        per_node = all(f'w={v["w"]}' in entries[n] and f'a={v["a"]}' in entries[n] for n, v in cfg.items())
        env_check[name] = {'entries': len(entries), 'equals_previous_compile_env': env_new == prev['MIXEDP_NODE_FORMATS'],
                           'previous_weight_per_channel': prev.get('MIXEDP_WEIGHT_PER_CHANNEL'), 'all_196_match_config': per_node,
                           'L27_entry': ':'.join([L27] + entries[L27]), 'L2_entry': ':'.join([L2] + entries[L2])}
        if not (len(entries) == len(cfg) == 196 and per_node and env_new == prev['MIXEDP_NODE_FORMATS'] and prev.get('MIXEDP_WEIGHT_PER_CHANNEL') == '1'):
            raise SystemExit(f'{name}: compile environment check failed {env_check[name]}')
        write(target / 'compile_env.json', {'MIXEDP_NODE_FORMATS': env_new, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'})
        start = time.perf_counter()
        so, info = erm.build_so(tools, target, name, NATIVE / 'model.mlirbc', env_new, obj, opt='O0', krnl_opt='O3',
                                constants_to_file=True, bytecode=True)
        if info['warnings']:
            raise SystemExit(f'{name}: compilation warnings {info["warnings"]}')
        for f in target.iterdir():                          # compiler intermediates only; the .so and constants stay
            if f.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                f.unlink()
        builds[name] = {'so': os.path.relpath(so, HERE), 'so_sha256': sha(so), 'constants_sha256': sha(target / 'model.constants.bin'),
                        'compile_env_sha256': sha(target / 'compile_env.json'), 'seconds': time.perf_counter() - start, 'steps': info['steps']}
        write(out / 'builds.json', builds | {'compile_env_checks': env_check})
        log(f'Built {name} in {builds[name]["seconds"]:.1f}s')
    log_stage(out, 'native_build', time.perf_counter() - t0, {k: builds[k]['seconds'] for k in ('A', 'C')} | {'posit_runtime_s': builds['posit_runtime']['seconds']})


# --------------------------------------------------------------------------- #
# native runs
# --------------------------------------------------------------------------- #
def worker(args):
    """one process, one .so: per-window summed CE appended after a hash header; resumable."""
    sys.path.insert(0, args.lib)
    from PyRuntime import OMExecutionSession
    from run_native_bc_pipeline import cross_entropy
    import fcntl
    hdr = json.loads(Path(args.header).read_text())
    if sha(args.tokens) != hdr['tokens_sha256'] or sha(Path(__file__).resolve()) != hdr['script_sha256']:
        raise SystemExit('worker: tokens or script differ from the header')
    path = Path(args.result)
    lock = open(str(path) + '.lock', 'w')                   # one writer per part file (e.g. an orphan of a killed run)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'worker: {path} is being written by another process')
    done = set()
    if path.exists():
        good = []
        for line in path.read_text().split('\n'):
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:                    # an interrupted append: drop it and recompute
                break
        if not good or good[0].get('header') != hdr:
            raise SystemExit(f'worker: checkpoint {path} belongs to another version / data')
        path.write_text(''.join(json.dumps(r) + '\n' for r in good))
        done = {r['index'] for r in good[1:]}
    else:
        path.write_text(json.dumps({'header': hdr}) + '\n')
    x = np.load(args.tokens, mmap_mode='r')
    sess = OMExecutionSession(args.so, 'model')
    t_all = time.perf_counter()
    n_run = 0
    for i in range(hdr['start'], hdr['stop']):
        if i in done:
            continue
        t0 = time.perf_counter()
        w = np.ascontiguousarray(x[i:i + 1])
        z = np.array(sess.run([w])[0][0])
        if z.dtype != np.float32 or z.shape != (SEQ, VOCAB) or not np.isfinite(z).all():
            raise SystemExit(f'worker: bad logits at {i}: {z.dtype} {z.shape}')
        ce = float(cross_entropy(z[:-1], np.asarray(x[i, 1:])).sum())
        row = {'index': i, 'summed_ce': ce, 'window_sha256': hashlib.sha256(w.tobytes()).hexdigest(), 'seconds': time.perf_counter() - t0}
        with path.open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())
        n_run += 1
        del z
    peak_kb = next(int(l.split()[1]) for l in open('/proc/self/status') if l.startswith('VmHWM'))
    write(Path(str(path) + '.stats.json'), {'windows_run_this_process': n_run, 'seconds': time.perf_counter() - t_all,
                                            'peak_rss_gb': peak_kb / 1048576, 'threads': {k: os.environ.get(k) for k in THREAD_ENV},
                                            'pid': os.getpid(), 'time': now()})


def run_native(out, name, tokens, label, workers=WORKERS):
    import eval_random_mixed_precision as erm
    tools = erm.Tools()
    builds = read(out / 'builds.json')
    so = HERE / builds[name]['so']
    if sha(so) != builds[name]['so_sha256'] or sha(so.parent / 'model.constants.bin') != builds[name]['constants_sha256']:
        raise SystemExit(f'{name}: .so / constants differ from builds.json')
    target = out / 'native_results' / label / name
    target.mkdir(parents=True, exist_ok=True)
    x = np.load(tokens, mmap_mode='r')
    base = {'tokens_sha256': sha(tokens), 'so_sha256': builds[name]['so_sha256'], 'constants_sha256': builds[name]['constants_sha256'],
            'script_sha256': sha(Path(__file__).resolve()), 'config': name, 'config_sha256': sha(out / 'configs' / f'{name}.json')}
    comp = target / 'complete.json'
    if comp.exists():
        c = read(comp)
        if {k: c['header'][k] for k in base} != base:
            raise SystemExit(f'{comp}: from another version / data')
        log(f'{label} {name}: reusing complete results')
        return np.array(c['summed_ce'])
    idxs = [v for v in np.array_split(np.arange(len(x)), min(workers, len(x))) if len(v)]
    env = dict(os.environ, **THREAD_ENV)

    def one(k, inds):
        hdr = base | {'start': int(inds[0]), 'stop': int(inds[-1]) + 1, 'part': k, 'workers': len(idxs)}
        hp = target / f'part{k}.header.json'
        if hp.exists() and read(hp) != hdr:
            raise SystemExit(f'{hp}: header differs')
        write(hp, hdr)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', 'worker', '--out-dir', str(out), '--so', str(so),
               '--lib', str(tools.lib), '--tokens', str(tokens), '--result', str(target / f'part{k}.jsonl'), '--header', str(hp)]
        with (target / f'part{k}.log').open('a') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(idxs)) as pool:
        for f in as_completed([pool.submit(one, k, v) for k, v in enumerate(idxs)]):
            f.result()
    rows, stats = [], []
    for k in range(len(idxs)):
        lines = [json.loads(l) for l in (target / f'part{k}.jsonl').read_text().splitlines() if l]
        if lines[0]['header'] != read(target / f'part{k}.header.json'):
            raise SystemExit('header mismatch after the run')
        rows += lines[1:]
        stats.append(read(str(target / f'part{k}.jsonl') + '.stats.json'))
    rows.sort(key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(len(x))):
        raise SystemExit(f'{label} {name}: windows missing or duplicated')
    for r in rows:
        if r['window_sha256'] != hashlib.sha256(np.ascontiguousarray(x[r['index']:r['index'] + 1]).tobytes()).hexdigest():
            raise SystemExit('window hash mismatch')
    ce = np.array([r['summed_ce'] for r in rows])
    write(comp, {'time': now(), 'header': base, 'windows': len(x), 'summed_ce': ce.tolist(), 'wall_s': time.perf_counter() - t0,
                 'sum_window_seconds': sum(r['seconds'] for r in rows), 'workers': len(idxs), 'worker_stats': stats})
    log(f'{label} {name}: {len(x)} windows in {time.perf_counter() - t0:.1f}s (peak RSS {max(s["peak_rss_gb"] for s in stats):.2f} GB)')
    return ce


def controls(out):
    p = check_plan(out)
    t0 = time.perf_counter()
    if (out / 'freeze.json').exists():
        raise SystemExit('frozen')
    builds = read(out / 'builds.json')
    tokens = out / 'work' / 'control_tokens.npy'
    np.save(tokens, np.load(NATIVE / 'calib_inputs.npy')[:2])
    result = {'config_checks': p['checks'], 'compile_env_checks': builds['compile_env_checks']}
    for name in ('A', 'C'):
        ce = run_native(out, name, tokens, 'controls', 2)
        ref = np.array(read(PREV / 'native' / f'{name}.json')['summed_ce'])[:2]
        err = float(np.abs(ce - ref).max())
        result[name] = {'summed_ce': ce.tolist(), 'previous': ref.tolist(), 'max_abs_error': err, 'passed': err <= TOL}
    repeat = out / 'work' / 'repeat_tokens.npy'
    np.save(repeat, np.load(tokens)[:1])
    ce = run_native(out, 'A', repeat, 'repeat', 1)
    result['repeat_A_window0'] = {'summed_ce': float(ce[0]), 'max_abs_error': float(abs(ce[0] - result['A']['summed_ce'][0]))}
    result['repeat_A_window0']['passed'] = result['repeat_A_window0']['max_abs_error'] <= TOL
    result['tolerance'] = TOL
    result['passed'] = bool(all(result[n]['passed'] for n in ('A', 'C')) and result['repeat_A_window0']['passed']
                            and all(v['equals_previous_compile_env'] and v['all_196_match_config'] for v in builds['compile_env_checks'].values()))
    write(out / 'controls.json', result)
    log_stage(out, 'controls', time.perf_counter() - t0)
    if not result['passed']:
        raise SystemExit(f'Native control failed: {result}')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'builds_sha256': sha(out / 'builds.json'),
                                'controls_sha256': sha(out / 'controls.json'), 'configs_sha256': p['configs_sha256'],
                                'so_sha256': {n: builds[n]['so_sha256'] for n in ('A', 'C')},
                                'constants_sha256': {n: builds[n]['constants_sha256'] for n in ('A', 'C')}})
    log(f'controls passed; frozen')


# --------------------------------------------------------------------------- #
# data (the previous round's audited sampler, this round's sizes and seeds)
# --------------------------------------------------------------------------- #
def reuse_sampler(out, stage):
    import run_qwen_down_fp16_protection as source
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('controls / freeze first')
    t0 = time.perf_counter()
    source.check_plan = check_plan
    source.log_stage = lambda o, s, sec, extra=None: None
    source.N_ART, source.WPA, source.SEEDS = N_ART, WPA, SEEDS
    getattr(source, stage)(out, log)
    if stage == 'audit':
        au = read(out / 'audit.json')
        if au['unresolved_files']:
            raise SystemExit(f'unresolved historical Qwen token sources: {au["unresolved_files"]} (investigate before sampling)')
    else:
        segs = read(out / 'segments.json')
        got = {s['content_sha256'] for s in segs}
        prev512 = {x['content_sha256'] for x in read(PREV / 'segments.json')}
        spike = {x['content_sha256'] for x in read(SPIKE / 'segments.json')}
        toks = np.load(out / 'tokens.npy')
        if len(got) != N_ART or got & prev512 or got & spike or toks.shape != (N_ART * WPA, SEQ):
            raise SystemExit('data isolation / shape check failed')
    log_stage(out, stage, time.perf_counter() - t0)


def verify_data(out):
    check_plan(out)
    d = read(out / 'data_freeze.json')
    for key, file in [('freeze', 'freeze.json'), ('tokens', 'tokens.npy'), ('segments', 'segments.json')]:
        if sha(out / file) != d[key + '_sha256']:
            raise SystemExit(f'{file} changed after the data freeze')
    fz = read(out / 'freeze.json')
    if sha(out / 'builds.json') != fz['builds_sha256'] or sha(out / 'controls.json') != fz['controls_sha256']:
        raise SystemExit('builds / controls changed after the freeze')


def native_eval(out):
    verify_data(out)
    t0 = time.perf_counter()
    vals, tm = {}, {}
    for name in ('A', 'C'):
        t1 = time.perf_counter()
        vals[name] = run_native(out, name, out / 'tokens.npy', 'fresh', WORKERS)
        tm[name] = time.perf_counter() - t1
    np.savez(out / 'native_ce.npz', **vals)
    stats = {n: read(out / 'native_results/fresh' / n / 'complete.json') for n in ('A', 'C')}
    log_stage(out, 'native_eval', time.perf_counter() - t0, {'per_config_wall_s': tm, 'workers': WORKERS, 'threads': THREAD_ENV,
                                                             'peak_rss_gb': {n: max(s['peak_rss_gb'] for s in stats[n]['worker_stats']) for n in stats},
                                                             'native_ce_sha256': sha(out / 'native_ce.npz')})


def pytorch_eval(out):
    verify_data(out)
    from run_qwen_down_fp16_protection import replay
    from run_olmoe_dynamic_route import env_info
    t0 = time.perf_counter()
    rp, init = replay(out / 'work')
    torch = rp.torch
    tok = np.load(out / 'tokens.npy')
    vals, secs = {}, {}
    for name in ('A', 'C'):
        t1 = time.perf_counter()
        cfg = read(out / 'configs' / f'{name}.json')
        rp.set(cfg)
        if not all(rp.act[n] == cfg[n]['a'] for n in rp.param):
            raise SystemExit('activation map differs from the configuration')
        vals[name] = rp.ce(tok)
        secs[name] = time.perf_counter() - t1
    np.savez(out / 'pytorch_ce.npz', **vals)
    env = env_info()
    log_stage(out, 'pytorch_eval', time.perf_counter() - t0, {'initialization_s': init, 'per_config_s': secs, 'environment': env,
                                                              'deterministic': torch.are_deterministic_algorithms_enabled(),
                                                              'pytorch_ce_sha256': sha(out / 'pytorch_ce.npz')})


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    verify_data(out)
    t0 = time.perf_counter()
    raw = {k: dict(np.load(out / f'{k}_ce.npz')) for k in ('native', 'pytorch')}
    nll = {k: {c: v.reshape(N_ART, WPA).sum(1) / (TP * WPA) for c, v in d.items()} for k, d in raw.items()}
    rng = np.random.default_rng(SEEDS['bootstrap'])
    weights = np.stack([np.bincount(rng.integers(0, N_ART, N_ART), minlength=N_ART) for _ in range(BOOT)]) / N_ART

    def summary(v):
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'se': float(v.std(ddof=1) / np.sqrt(len(v))),
                'ci95': np.percentile(weights @ v, [2.5, 97.5]).tolist(), 'median': float(np.median(v)),
                'articles_better': int((v < 0).sum()), 'articles_worse': int((v > 0).sum()), 'articles_zero': int((v == 0).sum())}
    diffs = {k: d['C'] - d['A'] for k, d in nll.items()}
    b = diffs['native'] - diffs['pytorch']
    res = {'articles': N_ART, 'windows_per_article': WPA, 'delta': DELTA, 'primary': 'native C-A',
           'nll': {k: {c: float(v.mean()) for c, v in d.items()} for k, d in nll.items()},
           'ppl': {k: {c: float(np.exp(v.mean())) for c, v in d.items()} for k, d in nll.items()},
           'comparisons': {k: summary(v) for k, v in diffs.items()},
           'backend_difference_of_effects': summary(b) | {
               'correlation_of_effects': float(np.corrcoef(diffs['native'], diffs['pytorch'])[0, 1]),
               'sign_agreement': int((np.sign(diffs['native']) == np.sign(diffs['pytorch'])).sum()),
               'absolute_nll_difference_native_minus_pytorch': {c: summary(nll['native'][c] - nll['pytorch'][c]) for c in ('A', 'C')}}}
    lo, hi = res['comparisons']['native']['ci95']
    res['decision'] = ('native quality improvement supported (upper < 0)' if hi < 0 else
                       'small quality drop, within the preset tolerance (CI all positive, upper < delta)' if lo > 0 and hi < DELTA else
                       'native non-inferiority supported; improvement not confirmed (upper < delta)' if hi < DELTA else
                       'added loss exceeds the tolerance (lower > delta)' if lo > DELTA else 'undetermined; keep A')
    segs = read(out / 'segments.json')
    order = np.argsort(diffs['native'])
    ext = lambda ix: [{'article': int(i), 'title': segs[i]['title'], 'd_native': float(diffs['native'][i]),
                       'd_pytorch': float(diffs['pytorch'][i]), 'nll_native_A': float(nll['native']['A'][i])} for i in ix]
    res['extremes'] = {'most_improved': ext(order[:5]), 'most_worsened': ext(order[::-1][:5])}
    res['distribution_native'] = {'quantiles_5_25_50_75_95': np.quantile(diffs['native'], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                                  'abs_gt_0.01': int((np.abs(diffs['native']) > 0.01).sum()), 'abs_gt_0.05': int((np.abs(diffs['native']) > 0.05).sum())}
    rs = read(out / 'resources.json')['configs']
    res['resources'] = {n: {k: rs[n][k] for k in ('weight_bytes', 'bops', 'size', 'bops_rel', 'resource', 'dominant', 'objective')} for n in ('A', 'C')}
    res['resources']['C_vs_A'] = rs['C']['vs_A']
    write(out / 'stats.json', res)
    with (out / 'per_article_losses.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['article', 'title', 'content_sha256', 'window_tokens', 'summed_ce_native_A', 'summed_ce_native_C', 'summed_ce_pytorch_A',
                    'summed_ce_pytorch_C', 'nll_native_A', 'nll_native_C', 'nll_pytorch_A', 'nll_pytorch_C', 'd_native', 'd_pytorch', 'b'])
        for i, s in enumerate(segs):
            w.writerow([i, s['title'], s['content_sha256'], s['window_tokens'], *[raw[k][c][i] for k in ('native', 'pytorch') for c in ('A', 'C')],
                        *[nll[k][c][i] for k in ('native', 'pytorch') for c in ('A', 'C')], diffs['native'][i], diffs['pytorch'][i], b[i]])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'decision': res['decision'], 'comparisons': res['comparisons'], 'backend': res['backend_difference_of_effects']}, ensure_ascii=False))


def main():
    global LOGF
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--stage', required=True, choices=['plan', 'build', 'controls', 'audit', 'data', 'native_eval', 'pytorch_eval', 'analyze', 'worker'])
    for key in ('so', 'lib', 'tokens', 'result', 'header'):
        p.add_argument('--' + key)
    a = p.parse_args()
    out = a.out_dir.resolve()
    if a.stage == 'worker':
        worker(a)
        return
    if a.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty: use a new directory')
    elif not (out / 'plan.json').exists():
        raise SystemExit('run --stage plan first')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'work').mkdir(exist_ok=True)
    LOGF = open(out / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage in ('audit', 'data'):
        reuse_sampler(out, a.stage)
    else:
        globals()[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(Path(__file__).resolve())}) + '\n')


if __name__ == '__main__':
    main()
