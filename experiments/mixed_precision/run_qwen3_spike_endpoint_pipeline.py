#!/usr/bin/env python3
"""Qwen3-0.6B: (1) confirm where the L2 Posit4 finite-amplitude damage sits (dense alpha of the spike rows, element
partitions), (2) one pre-fixed endpoint-sensitivity candidate E for every Down input with spike rows, the full MILP +
original knee search, native (onnx-mlir) compilation and a new-article native NLL / PPL A/B.
Spec: reports/qwen3_spike_endpoint_pipeline_prompt_20261026.md. The formal criterion, quantizers, MILP and knee are not
changed; the candidate tables and code live only in this round.

Down input x of layer l, original calibration (10 x 128), S = rows with max_c |x| >= 6 x median (N the rest), q = Q_f(x),
e = q - x, g0 = d(summed CE)/dx at x, g1_f = d(summed CE)/dx1 at x1 = where(S, q_f, x) (same GPU FP32 closure):
  C_N  = sum_c mean_N(g0^2) sum_N (e - mu_N)^2 + sum_(s,c) mu_N^2 (sum_(p in N_s) g0)^2
  P0   = sum_(S,c) g0^2 e^2,   Pend = 1/2 sum_(S,c) (g0^2 + g1^2) e^2
  B: C_x = the original BC formula on the GPU g0 / x / e;  D: C_x = C_N + P0;  E: C_x = C_N + Pend
Tables: A = native table.json; B / D / E replace only the Down x term of the Down nodes with spike rows:
C_total = C_y,old (native, re-run original statistics) + C_x,new. Search: mp_auto.auto_select (MILP, 17 samples, 2
refinements, time limit 10 s, normalized-chord knee), caps at A_auto's two actual resources, a pre-fixed aggressive point.
Evaluation: native compiled models on 128 new WikiText-103 articles (one 128-token window each), GPU LMReplay secondary.
"Mixed table building (PyTorch E term) + full-configuration native evaluation", not a compiler-native statistic.

Stages (one new --out-dir): plan, phase1, gpu_stats, native_xy, tables, search, build, controls, audit, data,
native_eval, pytorch_eval, analyze. GPU stages: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python;
the others ../../venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (sampler, read / write / sha / now / rel; standard imports only)

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE, PREP = prs.NATIVE, prs.PREP
REC = HERE / 'runs/qwen3_spike_recenter_20261026'
RAW = HERE / 'runs/qwen3_raw_error_spike_diagnostic_20261026'
FP16 = HERE / 'runs/qwen3_down_fp16_protection_20261024'
LPREG = HERE / 'runs/lp_pipeline_regression_20261016/B/qwen3/search_milp_orig.json'
NL, SEQ, TP, CAL = 28, 128, 127, 10
CNT = CAL * SEQ
FORMATS6 = ('int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'))
GRID11 = ('int4', 'int5', 'int6', 'int7', 'int8', 'p4e1', 'p6e1', 'p8e1', 'f16', 'bf16', 'fp32')
QUANT = GRID11[:-1]
TAU = 6.0
DENSE_ALPHAS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0)
SAMPLES, REFINEMENTS, TIME_LIMIT = 17, 2, 10.0
N_ART, WPA, BOOT, DELTA = 128, 1, 20000, 0.001
SEEDS = {'order': 26102621, 'start': 26102622, 'bootstrap': 26102623}   # the spec's 26102611 is the raw-error round's bootstrap seed
NATIVE_WORKERS, STATS_WORKERS, LM_SUB = 6, 4, 16
HIST_KNEE = {'size': 0.7598214285714285, 'bops': 0.8779761904761905, 'resource': 0.8779761904761905,
             'config_hash': '36205a0624a33d5577a1cb5c1b71919d03e400f8dbab44ac2ef11d3104df2f4f'}
TOL = {'phase1_reproduce_abs_summed_ce': 0.0, 'mask_equals_recenter': 'bitwise per layer', 'native_xy_sum_rel': 1e-9,
       'A_table_equals_native': 'bitwise', 'S_empty_nodes_equal_A': 'bitwise', 'gx_vs_gyW_rel': 1e-4, 'pos127_abs': 0.0,
       'phase1_vs_gpu_stats_L2_rel': 1e-12, 'knee_reproduction': 'config hash equal to the historical knee (else record and locate)',
       'native_control_abs_summed_ce': 1e-8, 'fp32_native_vs_gpu_abs_summed_ce_reported': 'reported only',
       'caps': 'size and BOPs of a cap solve <= the cap exactly (mp_milp exact check)'}
CODE = ['run_qwen3_spike_endpoint_pipeline.py', 'run_qwen3_raw_error_spike_diagnostic.py', 'run_qwen3_spike_recenter.py',
        'run_qwen_l27_native_confirm.py', 'run_qwen_down_fp16_protection.py', 'run_qwen_down_spike_diag.py', 'run_native_bc_pipeline.py',
        'run_solver_regression.py', 'run_lp_pipeline_eval.py', 'mp_auto.py', 'mp_milp.py', 'htq_pareto_search.py',
        'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'wt_manifest_scan.py', 'run_qwen_prefix_rank_shift.py',
        'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py']
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


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def down(l):
    return f'/model/model/layers.{l}/mlp/down_proj/MatMul'


def inputs():
    import run_olmoe_route_mass_restore as mr
    files = [HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] + [
        NATIVE / f for f in ('calib_inputs.npy', 'ranges_by_point.json', 'table.json', 'criterion.json', 'model.mlirbc')] + [
        NATIVE / 'stats_grads/grads.so', NATIVE / 'stats_grads/grads_layout.json', NATIVE / 'stats_errors6/errors6.so',
        NATIVE / 'stats_errors6/errors6_layout.json', PREP / 'node_mapping.json', PREP / 'validation_tokens.npy', PREP / 'model.onnx',
        REC / 'masks.npz', REC / 'components.csv', REC / 'report.md', RAW / 'calib_losses.npz', RAW / 'report.md', RAW / 'curves.csv',
        RAW / 'proxy_components.csv', LPREG, FP16 / 'native/A.json', FP16 / 'native/FP32.json', FP16 / 'configs/A.json',
        HERE / 'reports/qwen3_spike_endpoint_pipeline_prompt_20261026.md', HERE / 'qwen35_assets/model/tokenizer.json']
    return files + mr.WT103 + mr.WT2


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


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(), 'spec': 'reports/qwen3_spike_endpoint_pipeline_prompt_20261026.md and the request of 2026-10-26 (full pipeline authorized)',
        'phase1': {'scope': 'L2 Down input only; weights, other activations and the Down output FP32; original calibration, INT qparams, '
                            'Universal LUT, FP32 / eager / RMSNorm eps 1e-6', 'mask': 'S = {max_c |x| >= 6 x median} on the FP32 x, '
                            'checked equal to the recenter masks', 'directions': 'q = Q_f(x), e = q - x fixed; no re-centring or re-quantization',
                   'dense_alpha': {'formats': list(FORMATS6), 'alphas': list(DENSE_ALPHAS), 'alpha_1': 'masked QDQ where(S, q, x)',
                                   'others': 'x + alpha M_S e', 'check': 'alpha 0.5 and 1 equal runs/qwen3_raw_error_spike_diagnostic_20261026'},
                   'element_partitions': {'formats': ['p4e1', 'p6e1', 'p8e1'], 'maxpos': 'from the actual value set',
                                          'H': 'per S row the element(s) with the largest |x| (ties included)',
                                          'O_f': 'S and |x| > maxpos_f', 'R_f': 'S without O_f', 'alpha': 1.0, 'I_OR': 'Delta_S - Delta_O - Delta_R',
                                          'note': 'H is not necessarily inside O; H / O / R are not three exclusive groups'},
                   'L2_preview': 'C_x,B / C_x,D / C_x,E of the six formats at L2 and their same-bit picks (calibration diagnostic only); '
                                 '1/2 (g0 + g1).e_S vs the measured S-only dNLL (diagnostic, never a search cost)',
                   'closure': 'real Down, residual, layers 3..27 (attention, FFN), final norm, LM head, next-token CE'},
        'candidate_E': {'scope': 'every Down input whose calibration S is non-empty (N non-empty, valid threshold); all 10 quantized '
                                 'formats of GRID11; fp32 = 0; S empty / N empty / invalid threshold -> the A value',
                        'formulas': {'C_N': 'sum_c mean_N(g0^2) sum_N (e - mu_N)^2 + sum_(s,c) mu_N^2 [sum_(p in N_s) g0]^2 (N row count incl. '
                                            'the last position; gradient sums over N only)', 'P0': 'sum_(S,c) g0^2 e^2',
                                     'Pend': '1/2 sum_(S,c) (g0^2 + g1^2) e^2, g1 at x1 = where(S, q_f, x) (detached leaf), same closure / '
                                             'summed CE, other weights and activations FP32', 'C_x_D': 'C_N + P0', 'C_x_E': 'C_N + Pend',
                                     'C_x_B': 'original BC: sum_c G2/n (S2 - n mu^2) + sum_c GS mu^2 on the GPU g0 / x / e (n = 1280)'},
                        'fixed': '0.5 fixed, no coefficient scan, no measured loss in the cost, float64 sums; formats with e_S = 0 skip the '
                                 'endpoint backward (recorded)', 'not': 'not the trapezoid loss integral, not a Hessian'},
        'tables': {'A': 'native table.json (unchanged)', 'B': 'Down x of nodes with S: C_x_B', 'D': 'C_x_D', 'E': 'C_x_E',
                   'composition': 'C_total = C_y,old (native, re-run of the original grads / errors-shard-6 statistics .so on the '
                                  'calibration inputs) + C_x,new; checked C_x,old + C_y,old = C_old within '
                                  f'{TOL["native_xy_sum_rel"]} rel; never C_total minus a PyTorch C_x', 'unchanged': 'Down y, all other '
                                  'C entries, all B (weight) terms, formats, qparams and resources'},
        'search': {'policy': 'mp_auto.auto_select (MILP, normalized-chord knee), samples 17, refinements 2, time limit 10 s, GRID11 x GRID11, '
                             'the Qwen3 problem of run_solver_regression.dense_problem; non-finite costs forbid options',
                   'A_auto_check': f'equals the historical knee (hash {HIST_KNEE["config_hash"][:12]}, size / BOPs / R '
                                   f'{HIST_KNEE["size"]:.6f} / {HIST_KNEE["bops"]:.6f} / {HIST_KNEE["resource"]:.6f}); otherwise recorded and located',
                   'configs': {'A_auto': 'A automatic knee', 'E_auto': 'E automatic knee', 'B_cap / D_cap / E_cap': 'milp_select on B / D / E '
                               'with A_auto\'s actual weight bytes and BOPs as two caps', 'A_cap': 'control: A at the same caps (evaluated '
                               'only if it differs from A_auto)', 'A_aggr': 'on A\'s frontier, R < R_Aknee, R closest (log) to sqrt(R_min '
                               'R_Aknee) (ties: lower R, then config hash); none if the knee has the lowest R', 'E_aggr': 'E at A_aggr\'s caps'},
                   'dedupe': 'one compile / evaluation per distinct config hash; identical configs give identical differences'},
        'native': {'compile': 'erm.build_so: bytecode, constants to file, Krnl O3, clang O0, MIXEDP_WEIGHT_PER_CHANNEL=1, node formats '
                              'env from each config (all 196 checked); FP32 plain build; .so / constants / compile_env kept',
                   'controls': 'calibration windows 0-1: A_auto (hash of the historical knee) and FP32 equal the native results of '
                               'runs/qwen3_down_fp16_protection_20261024 (native/A.json, FP32.json) within 1e-8; A window 0 in another '
                               'process within 1e-8; FP32 native vs GPU reported',
                   'eval': f'run_qwen_l27_native_confirm.run_native workers ({NATIVE_WORKERS} isolated processes, 1 thread each), float32 '
                           'logits -> float64 CE over 127 positions, resumable checkpoints'},
        'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data (verified manifests, historical '
                 'Qwen token audit, WT2 / near-duplicate / previous-article rules, window WT2 >= 50% rejected)',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS,
                 'seed_note': 'the suggested 26102611 / 26102612 / 26102613 were not used: 26102611 is the bootstrap seed of '
                              'runs/qwen3_raw_error_spike_diagnostic_20261026; replaced before any data by 26102621 / 22 / 23 (unused)',
                 'when': 'after every table, search trace and config is frozen (freeze.json)'},
        'statistics': {'nll': 'summed CE / 127 per article; PPL = exp(mean NLL)',
                       'primary': {'P1': 'native NLL(E_cap) - NLL(A_auto)', 'P2': 'native NLL(E_auto) - NLL(A_auto)'},
                       'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]}), one index set for all configs; 95% '
                                    'and Bonferroni 97.5% two-sided percentile CIs', 'identical_configs': 'difference 0 by identity',
                       'reading': {'P1': 'upper (97.5%) < 0 and both caps respected -> same-cap quality improvement supported',
                                   'P2': 'quality and the two resources reported apart; improvement only if neither resource increases '
                                         'and upper < 0', 'noninferiority': f'upper < delta = {DELTA} -> non-inferior (not better)'},
                       'secondary_family_holm': ['E_aggr - A_aggr', 'B_cap - A_auto', 'D_cap - B_cap', 'E_cap - D_cap'],
                       'gpu': 'LMReplay GPU QDQ on the same windows; per-article effect differences, correlation, sign agreement; native '
                              'is the primary evidence'},
        'costs': ['native table reuse / re-run of the original statistics (bridge cost, not E\'s necessary cost)', 'GPU capture / g0',
                  'mask / mu / error statistics', 'endpoint backwards per (layer, format), skipped zero-error or empty groups',
                  'B / D / E incremental table time', 'searches (cold)', 'compiles (selected model and all research models)',
                  'evaluation', 'peak RAM / VRAM'],
        'tolerances': TOL, 'smoke': smoke or 'none',
        'boundaries': 'formal defaults, quantizers, MILP and knee unchanged; candidate tables only in this round; no other formula, '
                      'coefficient, channel mask or layer list after phase 1; failures recorded with amendments',
        'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：Qwen3 spike 端點敏感度候選的機制確認與完整 native pipeline A/B', '', f'時間：{pl["time"]}。以 plan.json 為準；寫入後修改須記 amendment。', '',
          '- 第一階段（只在原 10×128 calibration）：L2 Down input，六格式 S-only 的 α ∈ ' + str(list(DENSE_ALPHAS)) + '（α = 1 用 masked QDQ）；p4e1／p6e1／p8e1 的元素分區 H（每行最大 |x|）、O（S 中 |x| > maxpos）、R（S 其餘），I_OR = Δ_S − Δ_O − Δ_R；L2 的 B／D／E x-only 分數預覽與 ½(g0+g1)·e_S 診斷。',
          '- 候選 E（唯一）：所有有 spike 行的 Down input、10 個量化格式；C_N = V_N(μ_N) + U_N(μ_N)；Pend = ½Σ_S(g0² + g1²)e²，g1 在 x1 = where(S, q, x) 的同一 GPU closure；C_x,E = C_N + Pend；對照 D（C_N + P0）、B（GPU 原 BC C_x）。',
          '- 表：A = native table.json；B／D／E 只替換有 spike 行的 Down x 項，C_total = C_y,old（重跑原 native 統計）+ C_x,new；先驗證 C_x,old + C_y,old = C_old。',
          f'- 搜尋：mp_auto.auto_select（MILP、17 點、2 輪細化、time limit 10 s、原 knee）；A_auto 須重現歷史 knee；E_auto；B／D／E_cap 用 A_auto 實際 bytes 與 BOPs 兩個上限；A_aggr（A frontier 上 R 最接近 √(R_min·R_Aknee)）與 E_aggr；A_cap 為控制。',
          '- native：bytecode、constants to file、Krnl O3、clang O0、per-channel weight；控制窗口重現前輪 native A／FP32（1e-8）；去重後編譯與評估。',
          f'- 資料：凍結後抽 {N_ART} 篇新 WT103 × {WPA} 窗口；seed {SEEDS}；排除 {len(used)} 篇已用文章。',
          f'- 主要：P1 = E_cap − A_auto、P2 = E_auto − A_auto；文章 bootstrap {BOOT}，95% 與 Bonferroni 97.5% CI；δ = {DELTA} 只稱非劣。次要 Holm：E_aggr − A_aggr、B_cap − A_auto、D_cap − B_cap、E_cap − D_cap。GPU LMReplay 為後端對照。',
          f'- 冒煙測試：{smoke or "無"}', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; used {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# GPU helpers
# --------------------------------------------------------------------------- #
class GEnv:
    """the recenter / raw-error evaluator path generalized to every layer (LM-head sub-batch 16)."""

    def __init__(self, work):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        import eval_random_mixed_precision as erm
        import qwen3_swiglu_common as qc
        from run_qwen_down_spike_diag import Model, Quant, summed_ce_rows
        self.torch, self.erm, self.qc = torch, erm, qc
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        self.Q = Quant(work, 'cuda')
        self.M = Model('cuda')
        self.ce_rows = summed_ce_rows
        mp = read(PREP / 'node_mapping.json')
        rng = read(NATIVE / 'ranges_by_point.json')
        self.rng = {l: rng[next(n for n, p in mp.items() if p == f'model.layers.{l}.mlp.down_proj.weight')]['x'] for l in range(NL)}

    def q(self, x, f, l):
        torch = self.torch
        if f == 'fp32':
            return x
        if f.startswith('int'):
            bits = int(f[3:])
            scale, zp = self.erm.int_qparams(float(self.rng[l][0]), float(self.rng[l][1]), bits)
            return self.qc.int_act_qdq(x, scale, zp, bits)
        if f == 'f16':
            return x.to(torch.float16).to(x.dtype)
        if f == 'bf16':
            return x.to(torch.bfloat16).to(x.dtype)
        return self.Q.act(x, f, self.rng[l])

    def head_ce(self, normed, ids):
        ce = []
        for i in range(0, normed.shape[0], LM_SUB):
            z = self.M.model.lm_head(normed[i:i + LM_SUB])
            ce.append(self.ce_rows(self.torch, z, ids[i:i + LM_SUB]))
            del z
        return self.torch.cat(ce)

    def forward_ce(self, ids):
        hs = self.M.model.model(input_ids=ids, attention_mask=self.M.wrapper.mask, use_cache=False).last_hidden_state
        return self.head_ce(hs, ids)

    def capture(self, ids):
        torch = self.torch
        cap, kw = {}, {}
        with torch.no_grad():
            nohook = self.forward_ce(ids)
            hooks = [self.M.layers[1].register_forward_pre_hook(lambda m, a, k: kw.update(k), with_kwargs=True)]
            for l in range(NL):
                hooks.append(self.M.layers[l].post_attention_layernorm.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(('mid', l), a[0])))
                hooks.append(self.M.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(('x', l), a[0])))
            hooked = self.forward_ce(ids)
            for hk in hooks:
                hk.remove()
        if not torch.equal(nohook, hooked):
            raise SystemExit('hooks changed the forward')
        return nohook, cap, kw

    def replay(self, l, mid, xp, ids, kw):
        torch = self.torch
        with torch.no_grad():
            h = mid + self.M.layers[l].mlp.down_proj(xp)
            for layer in self.M.layers[l + 1:]:
                o = layer(h, **kw)
                h = o[0] if isinstance(o, tuple) else o
            return self.head_ce(self.M.model.model.norm(h), ids)

    def grad(self, l, mid, x, ids, kw):
        """d(summed CE of each window)/dx through the real suffix of layer l, microbatch 1; max |g_x - g_y W| / max |g_x|."""
        torch = self.torch
        W = self.M.layers[l].mlp.down_proj.weight
        g, dev = torch.empty_like(x), 0.0
        for j in range(x.shape[0]):
            xj = x[j:j + 1].detach().clone().requires_grad_(True)
            y = torch.nn.functional.linear(xj, W)
            h = mid[j:j + 1] + y
            for layer in self.M.layers[l + 1:]:
                o = layer(h, **kw)
                h = o[0] if isinstance(o, tuple) else o
            z = self.M.model.lm_head(self.M.model.model.norm(h))
            ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), ids[j:j + 1, 1:].reshape(-1), reduction='none').double().sum()
            gx, gy = torch.autograd.grad(ce, (xj, y))
            g[j:j + 1] = gx
            dev = max(dev, float((gx - gy @ W).abs().max() / gx.abs().max()))
            del z, ce
        return g, dev


def mask_of(x):
    """S from the FP32 x [10, 128, C]: a = max_c |x| per row, threshold = 6 x numpy median over the 1280 rows (the recenter rule)."""
    a = x.abs().amax(-1).reshape(-1).cpu().numpy()
    thr = TAU * float(np.median(a))
    valid = bool(np.isfinite(thr) and thr > 0)
    S = (a >= thr) if valid else np.zeros(len(a), bool)
    return S, thr, valid


def components(env, g0, x, e, Sr, g1=None):
    """float64 C_x terms of one (layer, format) on the GPU; Sr: bool [10, 128]."""
    torch = env.torch
    gd, ed = g0.double(), e.double()
    X = gd ** 2
    G2 = X.sum((0, 1))
    GS = (gd.sum(1) ** 2).sum(0)
    S1, S2 = ed.sum((0, 1)), (ed ** 2).sum((0, 1))
    mu = S1 / CNT
    B = float((G2 / CNT * (S2 - CNT * mu ** 2)).sum() + (GS * mu ** 2).sum())
    Nm = (~Sr)[..., None].double()
    Sm = Sr[..., None].double()
    nN = float((~Sr).sum())
    muN = (ed * Nm).sum((0, 1)) / nN
    VN = float(((X * Nm).sum((0, 1)) / nN * (((ed - muN) ** 2) * Nm).sum((0, 1))).sum())
    GSN = (((gd * Nm).sum(1)) ** 2).sum(0)
    UN = float((GSN * muN ** 2).sum())
    eS2 = (ed ** 2) * Sm
    P0 = float((X * eS2).sum())
    r = {'C_x_B': B, 'V_N(mu_N)': VN, 'U_N(mu_N)': UN, 'C_N': VN + UN, 'P0': P0, 'C_x_D': VN + UN + P0,
         'G2_sum': float(G2.sum()), 'S1': S1, 'S2': S2, 'G2': G2, 'GS': GS}
    if g1 is not None:
        g1d = g1.double()
        Pend = float((0.5 * (X + g1d ** 2) * eS2).sum())
        r.update({'Pend': Pend, 'C_x_E': VN + UN + Pend, 'P1_endpoint_only': float(((g1d ** 2) * eS2).sum()),
                  'trapezoid_diag': float((0.5 * (gd + g1d) * ed * Sm).sum()), 'F0_S': float((gd * ed * Sm).sum())})
    return r


# --------------------------------------------------------------------------- #
# phase 1 (L2, calibration)
# --------------------------------------------------------------------------- #
def phase1(out):
    t_all = time.perf_counter()
    check_plan(out)
    tm = defaultdict(float)
    env = GEnv(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    l = 2
    ids = torch.from_numpy(np.load(NATIVE / 'calib_inputs.npy')).cuda()
    t0 = time.perf_counter()
    nohook, cap, kw = env.capture(ids)
    mid, x = cap[('mid', l)], cap[('x', l)]
    base = env.replay(l, mid, x, ids, kw)
    tm['capture_s'] = time.perf_counter() - t0
    S, thr, valid = mask_of(x)
    Sr = torch.from_numpy(S.reshape(CAL, SEQ)).cuda()
    MS = Sr[..., None]
    ctl = {'replay_vs_nohook_abs': float((base - nohook).abs().max()), 'threshold': thr,
           'mask_equals_recenter': bool(np.array_equal(S, np.load(REC / 'masks.npz')['L2'])), 'S_rows': int(S.sum())}
    raw = np.load(RAW / 'calib_losses.npz')
    L = lambda xp: (env.replay(l, mid, xp, ids, kw) - base).cpu().numpy()
    dense, pert, repro = {}, {}, 0.0
    q, e = {}, {}
    t0 = time.perf_counter()
    for f in FORMATS6:
        q[f] = env.q(x, f, l)
        e[f] = q[f] - x
        eS = e[f] * MS
        for a in DENSE_ALPHAS:
            xp = x if a == 0 else (torch.where(MS, q[f], x) if a == 1 else x + a * eS)
            dense[f'{f}|{a!r}'] = L(xp)
            if a not in (0.0,):
                ad = (xp.double() - x.double())
                pert[f'{f}|{a!r}'] = {'norm': float(ad.norm()), 'max_abs': float(ad.abs().max()),
                                      'rounding_rel': float(((x + a * eS).double() - x.double() - a * eS.double()).norm() / (a * eS.double()).norm())}
        repro = max(repro, float(np.abs(dense[f'{f}|0.5'] - raw[f'{f}__S__0.5']).max()), float(np.abs(dense[f'{f}|1.0'] - raw[f'{f}__S__main']).max()))
    tm['dense_alpha_s'] = time.perf_counter() - t0
    ctl['reproduce_raw_error_round_abs'] = repro
    ctl['alpha0_abs'] = max(float(np.abs(dense[f'{f}|0.0']).max()) for f in FORMATS6)
    # element partitions at alpha = 1
    t0 = time.perf_counter()
    ax = x.abs()
    rowmax = ax.amax(-1, keepdim=True)
    H = MS & (ax == rowmax)
    parts = {}
    for f in ('p4e1', 'p6e1', 'p8e1'):
        maxpos = float(env.Q.lut[f][1].abs().max())
        O = MS & (ax > maxpos)
        R = MS & ~O
        res = {'maxpos': maxpos, 'H_elements': int(H.sum()), 'O_elements': int(O.sum()), 'R_elements': int(R.sum()),
               'H_inside_O': int((H & O).sum()), 'S_elements': int(MS.expand_as(x).sum())}
        for nm, m in (('H', H), ('O', O), ('R', R), ('S', MS.expand_as(x))):
            res[f'dL_{nm}'] = L(torch.where(m, q[f], x)).tolist()
        res['I_OR_summed'] = float(np.sum(res['dL_S']) - np.sum(res['dL_O']) - np.sum(res['dL_R']))
        res['H_rows_channels'] = Counter(int(c) for c in torch.nonzero(H)[:, 2].tolist()).most_common(8)
        parts[f] = res
    tm['partitions_s'] = time.perf_counter() - t0
    # L2 preview: B / D / E x-only and the trapezoid diagnostic
    t0 = time.perf_counter()
    g0, gyw = env.grad(l, mid, x, ids, kw)
    tm['g0_s'] = time.perf_counter() - t0
    ctl['gx_vs_gyW_rel'] = gyw
    ctl['g_pos127_abs'] = float(g0[:, -1].abs().max())
    prev = {}
    for f in FORMATS6:
        t1 = time.perf_counter()
        g1, gyw1 = env.grad(l, mid, torch.where(MS, q[f], x), ids, kw)
        tm['g1_s'] += time.perf_counter() - t1
        c = components(env, g0, x, e[f], Sr, g1)
        prev[f] = {k: v for k, v in c.items() if isinstance(v, float)} | {'S_only_dNLL_per_token': float(np.sum(dense[f'{f}|1.0'])) / (CAL * TP)}
        ctl['gx_vs_gyW_rel'] = max(ctl['gx_vs_gyW_rel'], gyw1)
    picks = {}
    for a, b in PAIRS:
        picks[f'{a}-{b}'] = {m: (a if prev[a][k] <= prev[b][k] else b) for m, k in (('B', 'C_x_B'), ('D', 'C_x_D'), ('E', 'C_x_E'))}
    ctl['passed'] = bool(ctl['replay_vs_nohook_abs'] == 0 and ctl['mask_equals_recenter'] and repro <= TOL['phase1_reproduce_abs_summed_ce']
                         and ctl['alpha0_abs'] == 0 and ctl['gx_vs_gyW_rel'] <= TOL['gx_vs_gyW_rel'] and ctl['g_pos127_abs'] == 0)
    np.savez(out / 'phase1_losses.npz', base=base.cpu().numpy(), **{k.replace('|', '__'): v for k, v in dense.items()})
    write(out / 'phase1.json', {'controls': ctl, 'perturbation': pert, 'partitions': parts, 'L2_preview': prev, 'L2_preview_picks': picks,
                                'note': 'calibration diagnostic only; x-only scores (the full C also has the y term); not a final quality'})
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    log_stage(out, 'phase1', time.perf_counter() - t_all, dict(tm))
    log(f'phase1 controls passed={ctl["passed"]}; picks {picks}')
    if not ctl['passed']:
        raise SystemExit('phase 1 controls failed')


# --------------------------------------------------------------------------- #
# GPU statistics of every Down input
# --------------------------------------------------------------------------- #
def gpu_stats(out):
    t_all = time.perf_counter()
    check_plan(out)
    if not read(out / 'phase1.json')['controls']['passed']:
        raise SystemExit('phase 1 controls did not pass')
    tm = defaultdict(float)
    cnt = Counter()
    env = GEnv(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    ids = torch.from_numpy(np.load(NATIVE / 'calib_inputs.npy')).cuda()
    t0 = time.perf_counter()
    nohook, cap, kw = env.capture(ids)
    torch.cuda.synchronize()
    tm['capture_all_layers_s'] = time.perf_counter() - t0
    rmask = np.load(REC / 'masks.npz')
    res, layers, ctl = {}, {}, {'mask_equals_recenter': {}, 'gx_vs_gyW_rel': 0.0, 'g_pos127_abs': 0.0}
    stats_np = {}
    for l in range(NL):
        x, mid = cap[('x', l)], cap[('mid', l)]
        t0 = time.perf_counter()
        S, thr, valid = mask_of(x)
        tm['mask_s'] += time.perf_counter() - t0
        ctl['mask_equals_recenter'][f'L{l}'] = bool(np.array_equal(S, rmask[f'L{l}']))
        nS = int(S.sum())
        fb = None if valid and 0 < nS < CNT else ('invalid threshold' if not valid else 'S empty' if nS == 0 else 'N empty')
        layers[f'L{l}'] = {'threshold': thr, 'S_rows': nS, 'fallback': fb,
                           'S_positions': dict(Counter(int(i) % SEQ for i in np.nonzero(S)[0]).most_common(10))}
        if fb:
            cnt['fallback_layers'] += 1
            continue
        Sr = torch.from_numpy(S.reshape(CAL, SEQ)).cuda()
        MS = Sr[..., None]
        t0 = time.perf_counter()
        g0, d0 = env.grad(l, mid, x, ids, kw)
        torch.cuda.synchronize()
        tm['g0_s'] += time.perf_counter() - t0
        cnt['g0_backward_windows'] += CAL
        ctl['gx_vs_gyW_rel'] = max(ctl['gx_vs_gyW_rel'], d0)
        ctl['g_pos127_abs'] = max(ctl['g_pos127_abs'], float(g0[:, -1].abs().max()))
        for f in QUANT:
            t0 = time.perf_counter()
            q = env.q(x, f, l)
            e = q - x
            torch.cuda.synchronize()
            tm['qdq_s'] += time.perf_counter() - t0
            g1 = None
            if bool((e * MS).abs().max() > 0):
                t0 = time.perf_counter()
                g1, d1 = env.grad(l, mid, torch.where(MS, q, x), ids, kw)
                torch.cuda.synchronize()
                tm['g1_endpoint_s'] += time.perf_counter() - t0
                tm[f'g1_endpoint_s|L{l}|{f}'] = time.perf_counter() - t0
                cnt['g1_backward_windows'] += CAL
                cnt['endpoint_backward_conditions'] += 1
                ctl['gx_vs_gyW_rel'] = max(ctl['gx_vs_gyW_rel'], d1)
            else:
                cnt['endpoint_skipped_zero_spike_error'] += 1
            t0 = time.perf_counter()
            c = components(env, g0, x, e, Sr, g1)
            if g1 is None:
                c.update({'Pend': c['P0'], 'C_x_E': c['C_x_D'], 'endpoint_skipped': True})
            torch.cuda.synchronize()
            tm['components_s'] += time.perf_counter() - t0
            stats_np[f'L{l}__{f}__S1'], stats_np[f'L{l}__{f}__S2'] = c['S1'].cpu().numpy(), c['S2'].cpu().numpy()
            if f == QUANT[0]:
                stats_np[f'L{l}__G2'], stats_np[f'L{l}__GS'] = c['G2'].cpu().numpy(), c['GS'].cpu().numpy()
            res[f'L{l}|{f}'] = {k: v for k, v in c.items() if not hasattr(v, 'shape')}
        log(f'L{l}: S {nS}; ' + ', '.join(f'{f} B/D/E {res[f"L{l}|{f}"]["C_x_B"]:.4g}/{res[f"L{l}|{f}"]["C_x_D"]:.4g}/{res[f"L{l}|{f}"]["C_x_E"]:.4g}'
                                          for f in ('int4', 'p4e1', 'int8', 'p8e1')))
    # phase-1 consistency at L2 (the same closure)
    p1 = read(out / 'phase1.json')['L2_preview']
    ctl['phase1_vs_gpu_stats_L2_rel'] = max(abs(p1[f][k] - res[f'L2|{f}'][k]) / max(abs(p1[f][k]), 1e-300)
                                            for f in FORMATS6 for k in ('C_x_B', 'C_x_D', 'C_x_E'))
    ctl['passed'] = bool(all(ctl['mask_equals_recenter'].values()) and ctl['gx_vs_gyW_rel'] <= TOL['gx_vs_gyW_rel'] and ctl['g_pos127_abs'] == 0
                         and ctl['phase1_vs_gpu_stats_L2_rel'] <= TOL['phase1_vs_gpu_stats_L2_rel'])
    np.savez(out / 'gpu_slot_stats.npz', **stats_np)
    write(out / 'gpu_components.json', {'layers': layers, 'components': res, 'counts': dict(cnt), 'controls': ctl})
    with open(out / 'cost_components.csv', 'w', newline='') as fh:
        keys = ['C_x_B', 'C_N', 'V_N(mu_N)', 'U_N(mu_N)', 'P0', 'Pend', 'P1_endpoint_only', 'C_x_D', 'C_x_E', 'trapezoid_diag', 'F0_S']
        wr = csv.writer(fh)
        wr.writerow(['layer', 'format', *keys])
        for k, v in res.items():
            l_, f = k.split('|')
            wr.writerow([l_, f, *[v.get(kk, '') for kk in keys]])
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    log_stage(out, 'gpu_stats', time.perf_counter() - t_all, {k: v for k, v in tm.items() if '|' not in k} | {'counts': dict(cnt)})
    write(out / 'endpoint_timing.json', {k: v for k, v in tm.items() if '|' in k})
    log(f'gpu_stats controls passed={ctl["passed"]}; counts {dict(cnt)}')
    if not ctl['passed']:
        raise SystemExit('gpu stats controls failed')


# --------------------------------------------------------------------------- #
# native x / y split of the original Down C (re-run of the original statistics .so)
# --------------------------------------------------------------------------- #
def native_xy(out):
    import run_native_bc_pipeline as nb
    import eval_random_mixed_precision as erm
    t_all = time.perf_counter()
    check_plan(out)
    work = out / 'work' / 'native_stats'
    work.mkdir(parents=True, exist_ok=True)
    tools = erm.Tools()
    toks = np.load(NATIVE / 'calib_inputs.npy')
    inputs_path = work / 'calib_inputs.npy'
    np.save(inputs_path, toks)
    onehot = work / 'onehot'
    onehot.mkdir(exist_ok=True)
    for i in range(CAL):
        t = np.zeros((1, SEQ, 151936), np.float32)
        t[0, np.arange(SEQ - 1), toks[i, 1:]] = 1.0
        np.save(onehot / f'{i}.npy', t)
    links = {}
    for kind, name in (('grads', 'grads'), ('errors6', 'errors6')):
        d = work / f'stats_{kind}'
        d.mkdir(exist_ok=True)
        src = NATIVE / f'stats_{kind}'
        for f in (f'{name}.so', 'model.constants.bin', f'{name}_layout.json'):
            if not (d / f).exists():
                os.symlink(src / f, d / f)
        links[kind] = (d / f'{name}.so', read(d / f'{name}_layout.json'))
    budget = nb.MemoryBudget(0.8 * nb.mem_available_gb())
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=STATS_WORKERS) as pool, ThreadPoolExecutor(max_workers=2) as chains:
        fg = chains.submit(nb.stats_runs, pool, tools, (links['grads'][0], erm.SYMBOL_POSTFIX), links['grads'][1], inputs_path, CAL, budget, onehot)
        fe = chains.submit(nb.stats_runs, pool, tools, (links['errors6'][0], erm.SYMBOL_POSTFIX), links['errors6'][1], inputs_path, CAL, budget)
        (vg, ig), (ve, ie) = fg.result(), fe.result()
    secs = time.perf_counter() - t0
    shutil.rmtree(onehot)
    lay_g, lay_e = links['grads'][1], links['errors6'][1]
    g2, gc, pos = {}, {}, {}
    taps = {t['tap']: t for t in lay_g['taps']}
    for p in lay_g['points']:
        t, key = taps[p['tap']], (p['node'], p['slot'])
        o, c = t['offset'], t['channels']
        g2[key] = vg[:, o:o + c].sum(0)
        gc[key] = (vg[:, o + c:o + 2 * c] ** 2).sum(0)
        pos[key] = t['positions']
    s1, s2 = {}, {}
    taps = {t['tap']: t for t in lay_e['taps']}
    for p in lay_e['points']:
        t, key = taps[p['tap']], (p['node'], p['slot'])
        c = t['channels']
        s1[key] = {f: ve[:, t['errors'][f]:t['errors'][f] + c].sum(0) for f in QUANT}
        s2[key] = {f: ve[:, t['errors'][f] + c:t['errors'][f] + 2 * c].sum(0) for f in QUANT}

    def a_cost(key, f):
        cnt_ = CAL * pos[key]
        mu = s1[key][f] / cnt_
        return float((g2[key] / cnt_ * (s2[key][f] - cnt_ * mu ** 2)).sum() + (gc[key] * mu ** 2).sum())
    table = read(NATIVE / 'table.json')
    xy, worst = {}, 0.0
    for l in range(NL):
        n = down(l)
        xy[n] = {'fp32': {'x': 0.0, 'y': 0.0}}
        for f in QUANT:
            cx, cy = a_cost((n, 'x'), f), a_cost((n, 'y'), f)
            xy[n][f] = {'x': cx, 'y': cy, 'old_total': table['C'][n][f], 'sum_minus_old': cx + cy - table['C'][n][f]}
            worst = max(worst, abs(cx + cy - table['C'][n][f]) / abs(table['C'][n][f]))
    # GPU vs native x-slot statistics of the Down inputs with spike rows (bridge description)
    gs = np.load(out / 'gpu_slot_stats.npz')
    bridge = {}
    for l in range(NL):
        if f'L{l}__G2' not in gs.files:
            continue
        n = down(l)
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
        bridge[f'L{l}'] = {'G2_rel': rel(gs[f'L{l}__G2'], g2[(n, 'x')]), 'GS_rel': rel(gs[f'L{l}__GS'], gc[(n, 'x')]),
                           'S1_rel_max': max(rel(gs[f'L{l}__{f}__S1'], s1[(n, 'x')][f]) for f in QUANT),
                           'S2_rel_max': max(rel(gs[f'L{l}__{f}__S2'], s2[(n, 'x')][f]) for f in QUANT)}
    ctl = {'x_plus_y_vs_table_rel_max': worst, 'passed': bool(worst <= TOL['native_xy_sum_rel'])}
    write(out / 'native_xy.json', {'C': xy, 'controls': ctl, 'gpu_vs_native_x_slot': bridge,
                                   'runs': {'grads': ig, 'errors6': ie, 'wall_s': secs}})
    log_stage(out, 'native_xy', time.perf_counter() - t_all, {'stats_runs_wall_s': secs, 'grads': ig, 'errors6': ie})
    log(f'native_xy: x + y vs table rel max {worst:.2e}; bridge {bridge.get("L2")}')
    if not ctl['passed']:
        raise SystemExit('native x + y does not reproduce table.json')


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def tables(out):
    t0 = time.perf_counter()
    check_plan(out)
    A = read(NATIVE / 'table.json')
    xy = read(out / 'native_xy.json')['C']
    gc = read(out / 'gpu_components.json')
    T = {}
    rep = {}
    for name, key in (('B', 'C_x_B'), ('D', 'C_x_D'), ('E', 'C_x_E')):
        tb = json.loads(json.dumps(A))
        changed = []
        for l in range(NL):
            n = down(l)
            if gc['layers'][f'L{l}']['fallback']:
                continue
            for f in QUANT:
                tb['C'][n][f] = xy[n][f]['y'] + gc['components'][f'L{l}|{f}'][key]
            changed.append(n)
        T[name] = tb
        rep[name] = {'changed_nodes': changed, 'others_bitwise_equal_A': all(tb['C'][n] == A['C'][n] for n in A['C'] if n not in changed)
                     and tb['B'] == A['B']}
    write(out / 'table_A.json', A)
    for name, tb in T.items():
        write(out / f'table_{name}.json', tb)
    ctl = {'A_equals_native_table': read(out / 'table_A.json') == A, 'replacements': rep}
    ctl['passed'] = bool(ctl['A_equals_native_table'] and all(v['others_bitwise_equal_A'] for v in rep.values()))
    write(out / 'tables_controls.json', ctl)
    log_stage(out, 'tables', time.perf_counter() - t0)
    log(f'tables: changed nodes {len(rep["E"]["changed_nodes"])}; passed {ctl["passed"]}')
    if not ctl['passed']:
        raise SystemExit('table controls failed')


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def problem(work):
    from run_solver_regression import dense_problem
    nodes, names, _, elems, flops, _ = dense_problem('qwen3', work)
    return nodes, names, elems, flops


def value_of(tb):
    def value(n, w, a):
        v = tb['B'][n][w] + tb['C'][n][a]
        return v if v is not None and math.isfinite(v) else None
    return value


def resources_of(cfg, names, elems, flops):
    from htq_pareto_search import FORMAT_BITS
    ref_b, ref_o = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
    wb = sum(elems[n] * FORMAT_BITS[cfg[n]['w']] / 8 for n in names)
    bo = sum(flops[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']] for n in names)
    return {'weight_bytes': wb, 'bops': bo, 'size': wb / ref_b, 'bops_rel': bo / ref_o, 'resource': max(wb / ref_b, bo / ref_o)}


def search(out):
    import mp_auto
    from mp_milp import milp_select
    t_all = time.perf_counter()
    check_plan(out)
    if not read(out / 'tables_controls.json')['passed']:
        raise SystemExit('tables did not pass')
    (out / 'configs').mkdir(exist_ok=True)
    (out / 'search').mkdir(exist_ok=True)
    nodes, names, elems, flops = problem(out / 'work' / 'spec')
    tb = {k: read(out / f'table_{k}.json') for k in 'ABDE'}
    tm, recs = {}, {}

    def auto(k):
        slog = []
        t0 = time.perf_counter()
        sel = mp_auto.auto_select(nodes, GRID11, GRID11, value_of(tb[k]), elems, flops, samples=SAMPLES, refinements=REFINEMENTS,
                                  time_limit=TIME_LIMIT, solver_log=slog)
        tm[f'{k}_auto_s'] = time.perf_counter() - t0
        write(out / 'search' / f'{k}_auto.json', {'selected': {kk: v for kk, v in sel['selected'].items() if kk != 'formats'},
                                                  'frontier': [{kk: v for kk, v in r.items() if kk != 'formats'} | {'config_hash': config_hash(r['formats'])}
                                                               for r in sel['frontier']], 'knee_scores': sel['knee_scores'],
                                                  'attempted_budgets': sel['attempted_budgets'], 'failures': sel['failures'], 'bounds': sel['bounds'],
                                                  'solver_log': slog, 'seconds': tm[f'{k}_auto_s']})
        return sel

    def cap(k, caps, tag):
        info = {}
        t0 = time.perf_counter()
        cfg, obj, status = milp_select(nodes, GRID11, GRID11, value_of(tb[k]), elems, flops, caps['weight_bytes'], caps['bops'],
                                       time_limit=TIME_LIMIT, info=info)
        tm[f'{tag}_s'] = time.perf_counter() - t0
        r = resources_of(cfg, names, elems, flops)
        recs[tag] = {'status': status, 'objective': obj, 'info': info, **r, 'caps': caps,
                     'within_caps': bool(r['weight_bytes'] <= caps['weight_bytes'] and r['bops'] <= caps['bops'])}
        return cfg
    selA = auto('A')
    selE = auto('E')
    cfgs = {'A_auto': selA['selected']['formats'], 'E_auto': selE['selected']['formats']}
    rA = resources_of(cfgs['A_auto'], names, elems, flops)
    capsA = {'weight_bytes': rA['weight_bytes'], 'bops': rA['bops']}
    for k in ('A', 'B', 'D', 'E'):
        cfgs[f'{k}_cap'] = cap(k, capsA, f'{k}_cap')
    # pre-fixed aggressive point from A's frontier
    fr = selA['frontier']
    knee_r = selA['selected']['resource']
    below = [r for r in fr if r['resource'] < knee_r]
    aggr = None
    if below:
        target = math.sqrt(fr[0]['resource'] * knee_r)
        aggr = min(below, key=lambda r: (round(abs(math.log(r['resource'] / target)), 12), r['resource'], config_hash(r['formats'])))
        cfgs['A_aggr'] = aggr['formats']
        ra = resources_of(aggr['formats'], names, elems, flops)
        cfgs['E_aggr'] = cap('E', {'weight_bytes': ra['weight_bytes'], 'bops': ra['bops']}, 'E_aggr')
    for k, c in cfgs.items():
        write(out / 'configs' / f'{k}.json', c)
    write(out / 'configs' / 'FP32.json', {})
    res = {k: resources_of(c, names, elems, flops) | {'config_hash': config_hash(c),
                                                       'objective_by_table': {t: sum(value_of(tb[t])(n, c[n]['w'], c[n]['a']) for n in names) for t in 'ABDE'}}
           for k, c in cfgs.items()}
    hashes = {k: v['config_hash'] for k, v in res.items()}
    uniq = {}
    for k in ('A_auto', 'E_auto', 'B_cap', 'D_cap', 'E_cap', 'A_aggr', 'E_aggr', 'A_cap'):
        if k in hashes:
            uniq.setdefault(hashes[k], k)
    eval_names = sorted(set(uniq.values()) - ({'A_cap'} if hashes.get('A_cap') == hashes['A_auto'] else set()))
    diffs = {k: {n: {'A_auto': cfgs['A_auto'][n], k: c[n]} for n in names if c[n] != cfgs['A_auto'][n]} for k, c in cfgs.items() if k != 'A_auto'}
    knee_ok = {'config_hash_equal': hashes['A_auto'] == HIST_KNEE['config_hash'], 'size': [rA['size'], HIST_KNEE['size']],
               'bops': [rA['bops_rel'], HIST_KNEE['bops']], 'resource': [rA['resource'], HIST_KNEE['resource']]}
    write(out / 'search_summary.json', {'resources': res, 'config_hashes': hashes, 'unique_representatives': uniq, 'evaluate': eval_names,
                                        'identical_to': {k: uniq[h] for k, h in hashes.items()}, 'cap_solves': recs, 'aggressive': {
                                            'R_min': fr[0]['resource'], 'R_knee': knee_r, 'target': math.sqrt(fr[0]['resource'] * knee_r) if below else None,
                                            'selected_resource': aggr['resource'] if aggr else None, 'available': bool(below)},
                                        'A_auto_vs_history': knee_ok, 'diffs_vs_A_auto': diffs, 'seconds': tm})
    log_stage(out, 'search', time.perf_counter() - t_all, tm)
    log(f'search: hashes {({k: v[:10] for k, v in hashes.items()})}; evaluate {eval_names}; knee reproduces {knee_ok["config_hash_equal"]}')


# --------------------------------------------------------------------------- #
# native build / controls / evaluation (run_qwen_l27_native_confirm's runner)
# --------------------------------------------------------------------------- #
def build(out):
    import onnx
    import eval_random_mixed_precision as erm
    from run_native_bc_pipeline import SLOT_INPUT, qwen3_spec
    import run_qwen_l27_native_confirm as l27
    t_all = time.perf_counter()
    check_plan(out)
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    ss = read(out / 'search_summary.json')
    tools, tf = l27.tool_files()
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
    builds = read(out / 'builds.json') if (out / 'builds.json').exists() else {}
    builds['posit_runtime'] = {'object': os.path.relpath(obj, HERE), 'sha256': sha(obj), 'seconds': time.perf_counter() - t1}
    builds['tools_sha256'] = {k: sha(v) for k, v in tf.items()}
    env_check = builds.get('compile_env_checks', {})
    for name in ['FP32'] + ss['evaluate']:
        if name in builds and (HERE / builds[name]['so']).exists() and sha(HERE / builds[name]['so']) == builds[name]['so_sha256']:
            log(f'{name}: reusing build')
            continue
        target = root / name
        target.mkdir(exist_ok=True)
        cfg = read(out / 'configs' / f'{name}.json')
        if name == 'FP32':
            env_new = None
        else:
            env_new = erm.node_formats_env({n['name']: cfg.get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}, spec.nodes, ranges)
            entries = {x_.split(':')[0]: x_.split(':')[1:] for x_ in env_new.split(',')}
            ok = all(f'w={v["w"]}' in entries[n] and f'a={v["a"]}' in entries[n] for n, v in cfg.items())
            env_check[name] = {'entries': len(entries), 'all_196_match_config': ok and len(entries) == len(cfg) == 196}
            if not env_check[name]['all_196_match_config']:
                raise SystemExit(f'{name}: compile environment check failed')
            if name == 'A_auto':
                prev = read(FP16 / 'native/A/compile_env.json')
                env_check[name]['equals_fp16_round_A_compile_env'] = env_new == prev['MIXEDP_NODE_FORMATS']
        write(target / 'compile_env.json', {'MIXEDP_NODE_FORMATS': env_new, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'})
        t1 = time.perf_counter()
        so, info = erm.build_so(tools, target, name, NATIVE / 'model.mlirbc', env_new, obj if env_new else None, opt='O0', krnl_opt='O3',
                                constants_to_file=True, bytecode=True, uses_posit=None if env_new else False,
                                uses_f16=None if env_new else False)
        if info['warnings']:
            raise SystemExit(f'{name}: compilation warnings {info["warnings"]}')
        for f in target.iterdir():
            if f.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                f.unlink()
        builds[name] = {'so': os.path.relpath(so, HERE), 'so_sha256': sha(so), 'constants_sha256': sha(target / 'model.constants.bin'),
                        'compile_env_sha256': sha(target / 'compile_env.json'), 'seconds': time.perf_counter() - t1, 'steps': info['steps']}
        builds['compile_env_checks'] = env_check
        write(out / 'builds.json', builds)
        log(f'built {name} in {builds[name]["seconds"]:.0f}s')
    log_stage(out, 'build', time.perf_counter() - t_all, {k: v['seconds'] for k, v in builds.items() if isinstance(v, dict) and 'seconds' in v})


def controls(out):
    import run_qwen_l27_native_confirm as l27
    t0 = time.perf_counter()
    check_plan(out)
    if (out / 'freeze.json').exists():
        raise SystemExit('frozen')
    ss = read(out / 'search_summary.json')
    builds = read(out / 'builds.json')
    tokens = out / 'work' / 'control_tokens.npy'
    np.save(tokens, np.load(NATIVE / 'calib_inputs.npy')[:2])
    res = {'compile_env_checks': builds['compile_env_checks']}
    for name in ['FP32'] + ss['evaluate']:
        ce = l27.run_native(out, name, tokens, 'controls', 2)
        res[name] = {'summed_ce': ce.tolist()}
    for name, ref in (('FP32', FP16 / 'native/FP32.json'), ('A_auto', FP16 / 'native/A.json')):
        if name == 'A_auto' and ss['config_hashes']['A_auto'] != HIST_KNEE['config_hash']:
            res[name]['reference'] = 'A_auto differs from the historical knee: no stored native reference'
            continue
        r = np.array(read(ref)['summed_ce'])[:2]
        res[name]['previous'] = r.tolist()
        res[name]['max_abs_error'] = float(np.abs(np.array(res[name]['summed_ce']) - r).max())
    rep = out / 'work' / 'repeat_tokens.npy'
    np.save(rep, np.load(tokens)[:1])
    ce = l27.run_native(out, 'A_auto', rep, 'repeat', 1)
    res['repeat_A_window0_abs'] = float(abs(ce[0] - res['A_auto']['summed_ce'][0]))
    res['FP32_native_vs_gpu'] = 'reported by the pytorch_eval stage (GPU environment), controls_gpu.json'
    tol = TOL['native_control_abs_summed_ce']
    res['passed'] = bool(res['FP32'].get('max_abs_error', 1) <= tol and res['A_auto'].get('max_abs_error', 0) <= tol
                         and res['repeat_A_window0_abs'] <= tol and all(v['all_196_match_config'] for v in builds['compile_env_checks'].values()))
    write(out / 'controls_native.json', res)
    log_stage(out, 'controls', time.perf_counter() - t0)
    if not res['passed']:
        raise SystemExit(f'native controls failed: {res}')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'builds_sha256': sha(out / 'builds.json'),
                                'controls_sha256': sha(out / 'controls_native.json'), 'search_summary_sha256': sha(out / 'search_summary.json'),
                                'tables_sha256': {k: sha(out / f'table_{k}.json') for k in 'ABDE'},
                                'configs_sha256': {p.stem: sha(p) for p in sorted((out / 'configs').glob('*.json'))},
                                'phase1_sha256': sha(out / 'phase1.json'), 'gpu_components_sha256': sha(out / 'gpu_components.json')})
    log('native controls passed; frozen')


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    log_stage(out, stage, time.perf_counter() - t0)


def verify_frozen(out):
    check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if sha(out / 'freeze.json') != dz['freeze_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256']:
        raise SystemExit('freeze / data changed')
    if sha(out / 'builds.json') != fz['builds_sha256'] or any(sha(out / 'configs' / f'{k}.json') != v for k, v in fz['configs_sha256'].items()):
        raise SystemExit('builds / configs changed after the freeze')
    return fz


def native_eval(out):
    import run_qwen_l27_native_confirm as l27
    t0 = time.perf_counter()
    verify_frozen(out)
    ss = read(out / 'search_summary.json')
    vals, tm = {}, {}
    for name in ['FP32'] + ss['evaluate']:
        t1 = time.perf_counter()
        vals[name] = l27.run_native(out, name, out / 'tokens.npy', 'fresh', NATIVE_WORKERS)
        tm[name] = time.perf_counter() - t1
    np.savez(out / 'native_ce.npz', **vals)
    log_stage(out, 'native_eval', time.perf_counter() - t0, {'per_config_wall_s': tm, 'workers': NATIVE_WORKERS,
                                                             'native_ce_sha256': sha(out / 'native_ce.npz')})


def pytorch_eval(out):
    from run_qwen_down_fp16_protection import replay
    t0 = time.perf_counter()
    verify_frozen(out)
    ss = read(out / 'search_summary.json')
    rp, init = replay(out / 'work')
    tok = np.load(out / 'tokens.npy')
    vals = {}
    for name in ['FP32'] + ss['evaluate']:
        cfg = read(out / 'configs' / f'{name}.json')
        rp.set(cfg)
        if cfg and not all(rp.act[n] == cfg[n]['a'] for n in rp.param):
            raise SystemExit('activation map differs from the configuration')
        vals[name] = rp.ce(tok)
    np.savez(out / 'pytorch_ce.npz', **vals)
    ctok = np.load(out / 'work' / 'control_tokens.npy')
    cn = read(out / 'controls_native.json')
    gpu_ctl = {}
    for name in ['FP32'] + ss['evaluate']:
        rp.set(read(out / 'configs' / f'{name}.json'))
        g = rp.ce(ctok)
        gpu_ctl[name] = {'gpu_summed_ce': g.tolist(), 'native_summed_ce': cn[name]['summed_ce'],
                         'native_minus_gpu_abs_max': float(np.abs(np.array(cn[name]['summed_ce']) - g).max())}
    write(out / 'controls_gpu.json', {'note': 'calibration control windows 0-1, reported only', 'configs': gpu_ctl})
    log_stage(out, 'pytorch_eval', time.perf_counter() - t0, {'initialization_s': init})


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    t0 = time.perf_counter()
    verify_frozen(out)
    ss = read(out / 'search_summary.json')
    ident = ss['identical_to']
    raw = {k: dict(np.load(out / f'{k}_ce.npz')) for k in ('native', 'pytorch')}
    get = lambda be, name: raw[be][ident.get(name, name)] / TP
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, N_ART, N_ART), minlength=N_ART) for _ in range(BOOT)]) / N_ART

    def st(v, same=False):
        b = Wb @ v
        r = {'mean': float(v.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(), 'ci97.5': np.percentile(b, [1.25, 98.75]).tolist(),
             'p_two_sided': 1.0 if same or np.all(v == 0) else float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))),
             'articles_better': int((v < 0).sum()), 'articles_worse': int((v > 0).sum()), 'identical_configs': bool(same)}
        return r

    def cmp(a, b, be):
        same = ident.get(a, a) == ident.get(b, b)
        return st(get(be, a) - get(be, b), same)
    res = {'nll': {be: {n: float(get(be, n).mean()) for n in ['FP32'] + list(ident)} for be in raw},
           'ppl': {be: {n: float(np.exp(get(be, n).mean())) for n in ['FP32'] + list(ident)} for be in raw},
           'delta_vs_fp32': {be: {n: float((get(be, n) - get(be, 'FP32')).mean()) for n in ident} for be in raw}}
    prim = {'P1 E_cap-A_auto': ('E_cap', 'A_auto'), 'P2 E_auto-A_auto': ('E_auto', 'A_auto')}
    sec = {'E_aggr-A_aggr': ('E_aggr', 'A_aggr'), 'B_cap-A_auto': ('B_cap', 'A_auto'), 'D_cap-B_cap': ('D_cap', 'B_cap'),
           'E_cap-D_cap': ('E_cap', 'D_cap')}
    res['primary'] = {k: {be: cmp(a, b, be) for be in raw} for k, (a, b) in prim.items()}
    secr = {k: {be: cmp(a, b, be) for be in raw} for k, (a, b) in sec.items() if a in ident and b in ident}
    order = sorted(secr, key=lambda k: secr[k]['native']['p_two_sided'])
    run_ = 0.0
    for j, k in enumerate(order):
        run_ = max(run_, min(1.0, (len(order) - j) * secr[k]['native']['p_two_sided']))
        secr[k]['native']['p_holm'] = run_
    res['secondary'] = secr
    if 'A_cap' in ident:
        res['control_A_cap-A_auto'] = {be: cmp('A_cap', 'A_auto', be) for be in raw}
    rs = ss['resources']
    p1 = res['primary']['P1 E_cap-A_auto']['native']
    p2 = res['primary']['P2 E_auto-A_auto']['native']
    caps_ok = rs['E_cap']['weight_bytes'] <= rs['A_auto']['weight_bytes'] and rs['E_cap']['bops'] <= rs['A_auto']['bops']
    res['reading'] = {
        'P1': ('identical configurations (difference 0)' if p1['identical_configs'] else
               'same-cap quality improvement supported' if p1['ci97.5'][1] < 0 and caps_ok else
               'non-inferior (upper < delta), improvement not shown' if p1['ci97.5'][1] < DELTA else
               'worse (lower > 0)' if p1['ci97.5'][0] > 0 else 'undetermined'),
        'P2': ('identical configurations (difference 0)' if p2['identical_configs'] else
               'quality improvement and no resource increase' if p2['ci97.5'][1] < 0 and rs['E_auto']['weight_bytes'] <= rs['A_auto']['weight_bytes']
               and rs['E_auto']['bops'] <= rs['A_auto']['bops'] else
               'quality CI below 0 but a resource increased' if p2['ci97.5'][1] < 0 else
               'non-inferior (upper < delta), improvement not shown' if p2['ci97.5'][1] < DELTA else
               'worse (lower > 0)' if p2['ci97.5'][0] > 0 else 'undetermined'),
        'caps_respected_E_cap': bool(caps_ok)}
    # backend agreement of the effects
    agree = {}
    for k, (a, b) in {**prim, **sec}.items():
        if a not in ident or b not in ident or ident[a] == ident[b]:
            continue
        dn, dp = get('native', a) - get('native', b), get('pytorch', a) - get('pytorch', b)
        agree[k] = {'correlation': float(np.corrcoef(dn, dp)[0, 1]) if dn.std() > 0 and dp.std() > 0 else None,
                    'sign_agreement': int((np.sign(dn) == np.sign(dp)).sum()), 'native_minus_gpu': st(dn - dp)}
    res['backend_agreement'] = agree
    res['resources'] = rs
    write(out / 'stats.json', res)
    segs = read(out / 'segments.json')
    with open(out / 'per_article_nll.csv', 'w', newline='') as fh:
        names = ['FP32'] + list(ident)
        wr = csv.writer(fh)
        wr.writerow(['article', 'title', *[f'{be}_{n}' for be in raw for n in names]])
        for i, s in enumerate(segs):
            wr.writerow([i, s['title'], *[float(get(be, n)[i]) for be in raw for n in names]])
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({'primary': {k: v['native'] for k, v in res['primary'].items()}, 'reading': res['reading'],
                    'secondary': {k: v['native'] for k, v in secr.items()}}, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'phase1', 'gpu_stats', 'native_xy', 'tables', 'search', 'build', 'controls',
                                                       'audit', 'data', 'native_eval', 'pytorch_eval', 'analyze'))
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
        globals()[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(HERE / 'run_qwen3_spike_endpoint_pipeline.py')}) + '\n')


if __name__ == '__main__':
    main()
