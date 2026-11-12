#!/usr/bin/env python3
"""Qwen3-0.6B L26 Down (weight = A's INT6, background F): with the calibration budget fixed at 10 x 128 = 1280 tokens and the
native ranges fixed, does spreading the 10 windows over 10 articles instead of one article change the measured-loss format choice
and its consequence on new articles? Sources WikiText-2 validation and WikiText-103 train; per source 3 pools x 10 articles x one
contiguous 1280-token span; the same 100 windows of a pool form 10 single-article groups (M[a, 0..9]) and 10 multi-article groups
(row j = M[(j - r) mod 10, j]), so every window keeps its batch row and both coverages see identical text.

Selector (primary): argmin of the measured calibration CE difference d_C(f) over f16, p8e1, p6e1, p4e1, int8, int6, int4, bf16
(exact ties: f16, then that order). Existing proxies (S_u + Khat256 = P256 and S_u + K2_actual, the implementations of
run_qwen3_l26_suffix_bridge / run_qwen3_topk_ce_curvature) are computed on the same windows and reported separately. Evaluator =
the L26 wrapper (complete f16 forward capturing x26 / r26 / block-27 kwargs, candidate replay through the real block 27, final
norm, head). New articles: 128 WikiText-103 train articles x 2 windows, drawn after every pick is frozen; truth only.
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
import run_qwen3_l26_suffix_bridge as l26  # noqa: E402
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen3_topk_ce_curvature as tk  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
ENDP, NATIVE = cs.ENDP, cs.NATIVE
L26 = HERE / 'runs/qwen3_l26_suffix_bridge_20261027'
DIAG = HERE / 'runs/qwen3_l26_calib_diagnostic_20261027'
RANGES = NATIVE / 'ranges_by_point.json'
WT2_VAL = HERE / 'gpt2_assets/data/validation-00000-of-00001.parquet'
TARGET, TARGET_W = l26.TARGET, l26.TARGET_W
FORMATS, SELECT_SET, POS = cs.FORMATS, cs.SELECT_SET, cs.POS
SEQ, TP, BATCH, CHUNK = cs.SEQ, cs.TP, cs.BATCH, cs.CHUNK
K = l26.K
CW = 10                                                        # windows per calibration group: 1280 tokens, 1270 targets
SOURCES = ('WT2', 'WT103')
SRC_LABEL = {'WT2': 'wikitext-2-raw-v1 validation', 'WT103': 'wikitext-103-raw-v1 train'}
POOLS, PER_POOL = 3, 10
COVS = ('single', 'multi')
C0_TITLE = 'Homarus gammarus'
N_ART, WPA, BOOT = 128, 2, 20000
SEEDS = {'wt2_order': 26102721, 'wt103_order': 26102722, 'span_start': 26102723, 'n_order': 26102724, 'n_start': 26102725,
         'bootstrap': 26102726}
SELECTORS = ('D_C', 'P256', 'Su+K2actual')                       # D_C primary; the proxies are separate secondary selectors
TERMS = ('CE', 'S_u', 'Khat256', 'K2_actual')
CONTROL_GROUPS = ['C0'] + [f'{s}.p{p}.{c}.{x}0' for s in SOURCES for p in range(POOLS) for c, x in (('single', 'a'), ('multi', 'r'))]
TOL = {'replay_vs_full_ce_abs_per_position': 1e-6, 'replay_vs_full_tensor_abs': l26.TOL['replay_vs_full_tensor_abs'],
       'replay_vs_full_tensor_rel': l26.TOL['replay_vs_full_tensor_rel'], 'f16_replay_vs_full_ce_abs_per_position': 1e-6,
       'reproduce_prev_C0_ce_abs_per_position': 1e-6, 'reproduce_prev_C0_proxy_abs_per_position': 1e-9,
       'shared_window_ce_abs_per_position': 1e-6, 'shared_window_proxy_abs_per_position': 1e-9, 'pooled_table_abs': 1e-12,
       'position_sum_abs': 1e-12, 'f16_zero_abs': 0.0, 'end_reproduce_C0_abs_per_position': 0.0}
CODE = list(dict.fromkeys(['run_qwen3_l26_calib_coverage.py', 'run_qwen_down_spike_diag.py'] + l26.CODE))
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def code_hashes_in_force(out):
    pl = read(out / 'plan.json')
    am = out / 'amendments.json'
    return read(am)[-1]['code_hashes'] if am.exists() else pl['code_hashes']


def check_plan(out, inputs_too=True):
    pl = read(out / 'plan.json')
    for f, h in code_hashes_in_force(out).items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan / last amendment: {f} (record an amendment)')
    if inputs_too:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def inputs():
    prev = [L26 / f for f in ('report.md', 'plan.json', 'frozen_predictions.json', 'C_terms.npz', 'controls_calib.json', 'freeze.json',
                              'data_freeze.json', 'segments.json', 'tokens.npy', 'eval_ce_pos.npz', 'stats.json', 'audit.json',
                              'work/qwen_hist_shingles.npz')]
    diag = [DIAG / f for f in ('report.md', 'source_audit.json', 'summary.json', 'calibration_windows.csv', 'leave_one_window_out.csv',
                               'analysis_scope.json')]
    return list(dict.fromkeys(l26.inputs() + prev + diag + [RANGES, HERE / 'run_qwen3_l26_calib_diagnostic.py',
                                                              HERE / 'reports/qwen3_l26_calib_coverage_prompt_20261027.md']))


def ranges_hash(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()


def int_qparams(rng):
    import eval_random_mixed_precision as erm
    return {f'int{b}': [float(v) for v in erm.int_qparams(float(rng[0]), float(rng[1]), b)] for b in (8, 6, 4)}


def group_ids():
    ids = ['C0']
    for s in SOURCES:
        for p in range(POOLS):
            ids += [f'{s}.p{p}.single.a{a}' for a in range(PER_POOL)] + [f'{s}.p{p}.multi.r{r}' for r in range(PER_POOL)]
    return ids


def cell_groups(s, c, pool=None):
    return [f'{s}.p{p}.{c}.{"a" if c == "single" else "r"}{i}' for p in (range(POOLS) if pool is None else (pool,)) for i in range(PER_POOL)]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_olmoe_dynamic_route import env_info
    from run_qwen_down_spike_diag import qwen3_tokenizer
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH or A[TARGET] != {'w': TARGET_W, 'a': 'f16'} or len(A) != 196:
        raise SystemExit('A is not the original knee or the target is not int6 / f16')
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig103 = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    val = mr.load_lines([WT2_VAL])
    arts2 = mr.split_articles(val)
    if arts2[0]['title'] != C0_TITLE:
        raise SystemExit('WT2 validation does not start with the C0 article')
    tok = qwen3_tokenizer()
    ntok2 = [len(tok.encode('\n\n'.join(val[a['line_start']:a['line_end']]), add_special_tokens=False).ids) for a in arts2[1:]]
    feas = {'wt2_validation_articles': len(arts2), 'wt2_after_excluding_C0': len(arts2) - 1,
            'wt2_with_at_least_1280_tokens_after_excluding_C0': int(sum(n >= CW * SEQ for n in ntok2)), 'needed_per_source': POOLS * PER_POOL,
            'wt103_eligible_before_length_and_neardup': len(elig103), 'note': 'metadata / token lengths only; no model loss'}
    if feas['wt2_with_at_least_1280_tokens_after_excluding_C0'] < POOLS * PER_POOL:
        write(out / 'feasibility_failed.json', feas)
        raise SystemExit(f'not enough WT2 articles: {feas}')
    pl = {'time': now(), 'spec': 'reports/qwen3_l26_calib_coverage_prompt_20261027.md and the request of 2026-10-27 (calibration coverage '
                                 'diagnostic; the request also asks for the existing proxy scores, leave-one-window-out, rank agreement and '
                                 'peak memory, all reported separately from the primary measured-loss selector)',
          'spec_conflict': 'the prompt file says not to compute new S_u / Khat; the request of 2026-10-27 asks for the existing proxy scores '
                           'next to the measured-loss selection; resolved: the proxies are computed on the same windows as secondary, '
                           'separately labelled selectors and never enter the primary selector, picks or comparison',
          'question': 'with 1280 calibration tokens and fixed ranges, does multi-article coverage (10 articles x 1 window) instead of '
                      'single-article coverage (1 article x 10 contiguous windows) lower the new-article damage of the frozen measured-loss '
                      'pick, WT2 validation fixed (primary); WT103 train as pre-registered replication / secondary',
          'scope': f'Qwen3-0.6B {TARGET} only; weight = A\'s {TARGET_W} (cached once from the FP32 weight by the existing quantizer); other 195 '
                   'projections per runs/qwen3_spike_endpoint_pipeline_20261026/configs/A_auto.json (background F, block 27 included); '
                   'target activation input and output in one format; LMReplay QDQ with the native ranges file (fixed, never recalibrated), '
                   'Posit LUT, FP32, eager, TF32 off, deterministic, use_cache=False; batches of 8 / tail 2',
          'ranges': {'file': prs.rel(RANGES), 'sha256': sha(RANGES), 'target_x': read(RANGES)[TARGET]['x'], 'target_y': read(RANGES)[TARGET]['y'],
                     'target_int_qparams_scale_zero_point': {s_: int_qparams(read(RANGES)[TARGET][s_]) for s_ in ('x', 'y')},
                     'rule': 'every group and the new articles use this one file (LMReplay ranges, checked equal to the file and re-hashed '
                             'after every group); no range, scale or zero point from any new text'},
          'feasibility': feas,
          'calibration data': {
              'C0': 'runs/native_eps_fix_20261014/qwen3/calib_inputs.npy (first 1280 tokens of the joined WT2 validation = Homarus gammarus); '
                    'historical reference only: same evaluator, not part of any pool or comparison',
              'pools': f'{POOLS} pools x {PER_POOL} distinct articles per source; the {POOLS} pools of a source share no article; '
                       'articles accepted in draw order, pool p = accepted articles 10p..10p+9 (article index a = order within the pool)',
              'WT2': 'only gpt2_assets/data/validation-00000-of-00001.parquet, split by top-level title (run_olmoe_route_mass_restore.'
                     'split_articles, strict boundaries); Homarus gammarus (C0) excluded; eligible = every other article; order = '
                     f'permutation(seed {SEEDS["wt2_order"]}); skip if < 1280 tokens, near-duplicate (word 5-gram shingle share >= '
                     f'{0.10}) of C0 or of an accepted calibration article; historical Qwen overlap is recorded, not excluded (WT2 is the '
                     'existing development corpus, not claimed new)',
              'WT103': 'only the WikiText-103 train parquets with the verified metadata; eligible = run_qwen_prefix_rank_shift.data base '
                       '(not a WT2 title / content hash, unique title and hash) minus every article of an earlier manifest '
                       f'(wt_manifest_scan.used_wt103_verified: {len(used)}); order = permutation(seed {SEEDS["wt103_order"]}); skip, in this '
                       'order: < 1280 tokens, near-duplicate of WT2, of an earlier article, of an accepted calibration article (both sources), '
                       'historical Qwen window / article overlap (all thresholds 0.10), any of the 10 windows with >= 0.5 WT2 shingle share '
                       '(the existing sampler rules)',
              'span': f'text = "\\n\\n".join(article rows) (the existing sampler convention), tokenizer add_special_tokens=False, no BOS / '
                      f'template; start uniform in [0, n_tokens - 1280] (rng = default_rng([{SEEDS["span_start"]}, 0]) for WT2 and '
                      f'[{SEEDS["span_start"]}, 1] for WT103, drawn after the article-level checks as in the existing sampler); 1280 '
                      'contiguous tokens -> 10 non-overlapping 128-token windows M[a, j]; positions restart at 0 in every window',
              'groups': 'per pool: single a = M[a, 0..9]; multi r: row j = M[(j - r) mod 10, j], j = 0..9 (row order never shuffled); every '
                        'window appears once in the single and once in the multi groups of its pool, at the same row j (same batch, same '
                        'row); 2 sources x 3 pools x 20 = 120 groups of 1280 tokens + C0',
              'manifests': 'calib/wt2/{tokens.npy,segments.json}, calib/wt103/{tokens.npy,segments.json} (content hashes, raw rows, chars, '
                           'token starts, token hashes, seeds), groups.json, calibration_data_freeze.json -- all before any model computation',
              'sampler check': 'before the pools: the reimplemented draw must reproduce runs/qwen3_l26_suffix_bridge_20261027 N (seeds, used '
                               'set as of that round, its audit index) token for token and segment for segment; failure stops the round'},
          'calibration measurement': {
              'evaluator': 'run_qwen3_l26_suffix_bridge.Env: complete f16 forward of the group (batches 8 + 2) capturing x26, r26, the '
                           'block-27 kwargs and u_0; each format replayed through the target QDQ / GEMM, the real block 27, final norm, '
                           'head CE (batch 8)',
              'd_C': 'd_C(f) = mean over the 1270 targets of CE(f) - CE(f16); pick = argmin over the selection set; exact ties f16 first then '
                     'the set order; no gain threshold, penalty or guard',
              'saved': 'per window / position CE of all 9 formats, d_C, full order, margin to the second, INT8 positive / negative windows',
              'proxies (secondary, separate)': 'g_u = head-only gradient at u_0 (microbatch 1); S_u = g_u . (u_f - u_0); Khat256 = '
                                               'run_qwen3_topk_ce_curvature.prep / khat (p = float64 softmax of the microbatch-1 logits); '
                                               'K2_actual = 1/2 Var_p0(logit difference) with the batch-8 logits (float64), exactly the '
                                               'previous round; selectors argmin P256 = S_u + Khat256 and argmin S_u + K2_actual',
              'leave-one-window-out': 'per group and selector: delete one of the 10 windows, re-aggregate the saved per-window values '
                                      '(1143 targets), argmin; no new forward, gradient or range',
              'not done': 'no pooling of groups into a larger calibration set, no use of N to choose a group / pool / source'},
          'controls': [f'ranges: LMReplay ranges equal {prs.rel(RANGES)} (sha256) and unchanged after every group; target int scale / zero '
                       'point recorded; A / INT6 / hooks / LUT as the previous round (target LMReplay hook no-op, our QDQ once)',
                       f'C0: per-position CE of all formats reproduces runs/qwen3_l26_suffix_bridge_20261027 F within '
                       f'{TOL["reproduce_prev_C0_ce_abs_per_position"]}, proxies within {TOL["reproduce_prev_C0_proxy_abs_per_position"]}; '
                       'C0 pick int8; C0 leave-one-window-out reproduces runs/qwen3_l26_calib_diagnostic_20261027 (D 7 int8 / 3 f16, '
                       'S_u+K2 7 / 3, P256 10 / 0)',
                       f'f16 replay vs complete forward CE for every group (<= {TOL["f16_replay_vs_full_ce_abs_per_position"]}); all 9 formats '
                       f'replay vs complete forward (block-27 input, final u: abs {TOL["replay_vs_full_tensor_abs"]} + rel '
                       f'{TOL["replay_vs_full_tensor_rel"]}; CE {TOL["replay_vs_full_ce_abs_per_position"]}) on {CONTROL_GROUPS}',
                       'kwargs: position_ids = cache_position = 0..127, causal mask, no cache, use_cache False, batch shapes 8 + 2',
                       'partition: each pool\'s 100 windows used exactly once by the single and once by the multi groups, same row; every '
                       'multi group has 10 distinct articles; token hashes match the manifests',
                       f'shared windows: every format, every window, single vs multi per-position CE <= {TOL["shared_window_ce_abs_per_position"]} '
                       f'(proxies <= {TOL["shared_window_proxy_abs_per_position"]}); bitwise equality recorded; a failure is localized, '
                       'never read as a coverage effect',
                       f'pooled table: per pool the mean d_C of the 10 single groups equals that of the 10 multi groups (<= '
                       f'{TOL["pooled_table_abs"]}) with the same argmin (a mathematical control, not a result)',
                       'f16 D exactly 0; fp32 only switches the target activation off; target weight INT6; P0 + Rest = All; C0 re-run at '
                       'the end reproduces the first C0 baseline exactly',
                       'failures are kept; implementation fixes need an amendment with time, reason and hashes'],
          'new articles': {'n_articles': N_ART, 'windows_per_article': WPA, 'when': 'after the freeze of all 121 groups\' picks',
                           'rule': 'the existing sampler rules (>= 256 tokens, start uniform, the same skips) with the verified manifest scan, '
                                   'the historical Qwen index of this round\'s audit, and in addition every calibration article of this '
                                   'round (WT103 content hashes added to the excluded set and its near-duplicate reference; WT2 articles '
                                   'are excluded by the WT2 rules)', 'seeds': {k: SEEDS[k] for k in ('n_order', 'n_start')},
                           'use': 'truth only: per-position CE of the 9 formats in background F; no range, proxy, group or pick update',
                           'previous N': 'runs/qwen3_l26_suffix_bridge_20261027 N (128 articles) was already seen (that round and the post hoc '
                                         'diagnostic): excluded and not used, not even as exploratory evidence'},
          'statistics': {'unit': 'article: 2 windows -> mean over 254 targets; D_n(f) = CE(f) - CE(f16)',
                         'L': 'L_n(s, c) = mean over the 3 pools x 10 frozen groups of cell (s, c) of D_n(pick); the average consequence of '
                              'those frozen picks, not an ensemble configuration and not a selector on 600 windows',
                         'primary': 'delta_n = L_n(WT2, multi) - L_n(WT2, single), D_C picks; improvement only if mean < 0 and CI upper < 0 '
                                    '("in these three WT2 pools, multi-article coverage lowers the average new-article damage of the frozen '
                                    'picks"); CI covering 0 = undetermined, not equivalence; identical pick frequencies give exactly 0',
                         'bootstrap': f'{BOOT} paired article resamples, seed {SEEDS["bootstrap"]}, one index set for every comparison; 95% '
                                      'percentile CI; it reflects only N uncertainty for these fixed pools and picks (groups share articles, '
                                      'no group-level test)',
                         'secondary (effects with descriptive CI)': [
                             'WT103 multi - single; every source x pool multi - single', 'WT103 - WT2 at the same coverage (N is WT103 only)',
                             'every cell vs f16 (= L) and vs the C0 frozen int8', 'pick frequencies, C margins, mean N damage and harmful '
                             'articles per group and cell', 'regret vs the N-mean best format (descriptive only)',
                             'P0 / Rest C -> N shift of every format per source and C0',
                             'leave-one-window-out pick-change rate per cell (D_C and proxies)',
                             'rank agreement of each group\'s calibration order with N: Kendall tau-b over the 8 formats, and right / wrong '
                             'on the N-decisive pairs of 28', 'proxy selectors (P256, S_u + K2_actual): the same L / delta, reported '
                                                              'separately from the measured-loss selector'],
                         'reading': ['coverage "helps this L26 diagnostic" only if the primary improvement holds AND the WT2 multi '
                                     'leave-one-window-out change rate is lower than single', 'lower change rate without the N improvement = '
                                     'calibration scores more stable, not better prediction', 'a pick back to f16 = harm avoided, not a gain '
                                     'over f16', 'source and coverage differences are described separately; WT2 vs WT103 differs in corpus '
                                     'split, history and article population, so no claim isolates article diversity beyond these pools',
                                     'nothing here changes or validates the BC criterion, other layers or the search pipeline']},
          'cost': 'metadata / tokenize / audit / sampling times; unique windows (600 + C0) vs replayed windows; per group prefix, candidate '
                  'suffix, head CE, gradient + proxies, controls; new-article evaluation; peak VRAM (torch) and peak RSS; this is a '
                  'diagnostic of measured loss, not the cost of a future BC / P256 update',
          'tolerances': TOL, 'smoke': smoke or 'none',
          'boundaries': 'no change to the formal BC, sensitivities or quantization errors, A, quantizers, ranges, MILP / knee or native; no '
                        'training, STE, HVP, JVP, margin or confidence rule; no L25 or other node; no deployment of the best pool',
          'used_wt103_count': len(used), 'used_wt103_sha256': hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest(),
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：L26 校準文章覆蓋與來源（固定 1280 tokens、固定 ranges）', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        f'- 目標 {TARGET}，權重 A 的 {TARGET_W}，背景 F；activation input／output 同格式；選擇集 {", ".join(SELECT_SET)}（fp32 僅數值控制）。',
        f'- ranges 固定為 {prs.rel(RANGES)}（sha256 {pl["ranges"]["sha256"][:12]}…），所有組與新文章共用，不重算。',
        f'- 來源 WT2 validation／WT103 train，各 {POOLS} pool × {PER_POOL} 篇（pool 間不重複），每篇一段連續 1280 tokens → 10 窗口 M[a,j]。',
        '- 單篇組 a = M[a,0..9]；多篇組 r 的第 j 列 = M[(j−r) mod 10, j]；兩種覆蓋用完全相同的 100 窗口、同 batch 列。',
        f'- 種子：{SEEDS}。WT2 排除 C0（{C0_TITLE}），歷史重疊只記錄；WT103 沿用既有 sampler 規則並排除所有已用文章（{len(used)}）。',
        '- 主要選擇：校準實測 d_C argmin（平手 f16 優先）；proxy（P256、S_u+K2_actual）另列，不混為一法。留一窗口只重聚合已存數值。',
        f'- 全部 121 組 pick 凍結後才抽 {N_ART} 篇新 WT103 文章 × {WPA} 窗口（排除所有校準文章與舊 N），只量 F 背景真值。',
        f'- 主要比較：delta_n = L_n(WT2,multi) − L_n(WT2,single)，{BOOT} 次文章 bootstrap（seed {SEEDS["bootstrap"]}）；mean<0 且 CI 上界<0 才算改善。',
        '- 「覆蓋有幫助」需主要改善成立且 WT2 multi 的留一窗口改選率低於 single；只更穩不等於更準。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; used {len(used)}, WT103 eligible {len(elig103)}, feasibility {feas}')


# --------------------------------------------------------------------------- #
# historical Qwen token audit (the rule of run_qwen_down_fp16_protection.audit, copied because that function requires the picks
# freeze; here it precedes the calibration draw) + run_qwen3_conditional_signed_score.resolve_subsets
# --------------------------------------------------------------------------- #
def audit(out):
    import pyarrow.parquet as pq
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_qwen_down_spike_diag import hist_files, qwen3_tokenizer, window_shingles
    t0 = time.perf_counter()
    check_plan(out)
    if (out / 'audit.json').exists():
        raise SystemExit('audit exists')
    tok3 = qwen3_tokenizer()
    from tokenizers import Tokenizer
    tok35 = Tokenizer.from_file(str(HERE / 'qwen35_assets/model/tokenizer.json'))
    ctl = {}
    for split, f in (('validation', 'validation_tokens.npy'), ('test', 'test_tokens.npy')):
        text = '\n\n'.join(pq.read_table(HERE / 'gpt2_assets/data' / f'{split}-00000-of-00001.parquet')['text'].to_pylist())
        ids = tok3.encode(text, add_special_tokens=False).ids
        ref = np.load(prs.PREP / f)
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
        tkz, name = (tok35, 'qwen3.5') if 'qwen35' in rel.lower() else (tok3, 'qwen3')
        if name == 'qwen3' and mx >= 151936:
            raise SystemExit(f'{rel}: id {mx} outside the Qwen3 vocabulary')
        seg = p.parent / 'segments.json'
        listed = [mr.shingles(by_ch[s['content_sha256']]['body_norm']) for s in read(seg) if s.get('content_sha256') in by_ch] if seg.exists() else []
        res_ = Counter()
        shares = []
        for w in a:
            sh = window_shingles(tkz.decode(w.tolist()))
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
                   'overlap exclusion of the calibration and new-article draws',
           'note': 'copy of run_qwen_down_fp16_protection.audit (that function requires freeze.json; here the audit precedes the '
                   'calibration draw); files inside this run are skipped'}
    write(out / 'audit.json', res)
    if res['unresolved_files']:
        cs.LOGF = LOGF
        cs.resolve_subsets(out)
    ca.log_stage(out, 'audit', time.perf_counter() - t0)
    log(f'audit: {len(rows)} files, {wid} windows; ' + json.dumps(Counter(r['resolved'] for r in rows)))


# --------------------------------------------------------------------------- #
# article draws (the rules of run_qwen_prefix_rank_shift.data, generalized to a span of `span` windows and other sources)
# --------------------------------------------------------------------------- #
def corpus_ctx(audit_npz):
    import pyarrow.parquet as pq
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    from run_qwen_down_spike_diag import qwen3_tokenizer, window_shingles
    t0 = time.perf_counter()
    hz = np.load(audit_npz)
    hist = defaultdict(set)
    wsize = Counter(hz['window'].tolist())
    for s_, w in zip(hz['shingle'].tolist(), hz['window'].tolist()):
        hist[s_].add(w)
    U2 = set()
    for f in mr.WT2:
        U2 |= window_shingles(' '.join(pq.read_table(f)['text'].to_pylist()))
    w103, wt2 = wms.corpora()
    inv = defaultdict(set)
    for i, a in enumerate(wt2):
        for s_ in mr.shingles(a['body_norm']):
            inv[s_].add(i)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    base = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1]
    return {'hist': hist, 'wsize': wsize, 'U2': U2, 'w103': w103, 'wt2': wt2, 'inv': inv, 'base': base, 'by': {a['ch']: a for a in w103},
            'lines103': mr.load_lines(mr.WT103), 'tok': qwen3_tokenizer(), 'sh_cache': {}, 'load_s': time.perf_counter() - t0}


def art_sh(ctx, a):
    import run_olmoe_route_mass_restore as mr
    c = ctx['sh_cache']
    if a['ch'] not in c:
        c[a['ch']] = mr.shingles(a['body_norm'])
    return c[a['ch']]


def draw(ctx, arts, lines, order, rs, n_need, span, rules, prev_sh, acc_sh, source, ref_sh=None):
    """rules (checked in this order): 'wt2' near-duplicate of a WT2 article, 'ref' near-duplicate of the C0 article, 'previous' of an
    earlier article, 'accepted' of an accepted article (acc_sh, shared across draws), 'qwen' historical Qwen window / article overlap,
    'window_wt2' a window with >= WT2_WINDOW_MAX WikiText-2 shingle share; the start is drawn after the article-level checks."""
    import run_olmoe_route_mass_restore as mr
    from run_qwen_down_spike_diag import window_shingles
    tok, inv, hist, wsize, U2 = ctx['tok'], ctx['inv'], ctx['hist'], ctx['wsize'], ctx['U2']
    segs, toks, skips, excl = [], [], Counter(), []
    for k in order.tolist():
        a = arts[k]
        text = '\n\n'.join(lines[a['line_start']:a['line_end']])
        enc = tok.encode(text, add_special_tokens=False)
        ids, off = enc.ids, enc.offsets
        if len(ids) < span * SEQ:
            skips['short'] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': 'short', 'tokens': len(ids)})
            continue
        sh = art_sh(ctx, a)
        shares = {}
        if 'wt2' in rules:
            hits = Counter(i for s_ in sh for i in inv.get(s_, ()))
            shares['wt2'] = max(hits.values()) / len(sh) if hits and sh else 0.0
        if 'ref' in rules:
            shares['C0'] = len(sh & ref_sh) / len(sh) if sh else 0.0
        if 'previous' in rules:
            shares['previous'] = max((len(sh & o) / len(sh) for o in prev_sh), default=0.0) if sh else 0.0
        shares['accepted'] = max((len(sh & o) / len(sh) for o in acc_sh), default=0.0) if sh else 0.0
        hw = Counter(w for s_ in sh for w in hist.get(s_, ()))
        shares['qwen_window'] = max((c / wsize[w] for w, c in hw.items()), default=0.0)
        shares['qwen_article'] = sum(1 for s_ in sh if s_ in hist) / len(sh) if sh else 0.0
        reason = ('near_duplicate_wt2' if 'wt2' in rules and shares['wt2'] >= mr.NEARDUP else
                  'near_duplicate_C0' if 'ref' in rules and shares['C0'] >= mr.NEARDUP else
                  'near_duplicate_previous' if 'previous' in rules and shares['previous'] >= mr.NEARDUP else
                  'near_duplicate_accepted' if shares['accepted'] >= mr.NEARDUP else
                  'historical_qwen_overlap' if 'qwen' in rules and max(shares['qwen_window'], shares['qwen_article']) >= mr.NEARDUP else None)
        if reason:
            skips[reason] += 1
            excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': reason, 'shares': shares})
            continue
        st = int(rs.integers(0, len(ids) - span * SEQ + 1))
        w = np.array(ids[st:st + span * SEQ], dtype=np.int64).reshape(span, SEQ)
        wshare = None
        if 'window_wt2' in rules:
            wshare = []
            for row in w:
                s2 = window_shingles(tok.decode(row.tolist()))
                wshare.append(len(s2 & U2) / len(s2) if s2 else 0.0)
            if max(wshare) >= prs.WT2_WINDOW_MAX:
                skips['window_wt2_overlap'] += 1
                excl.append({'title': a['title'], 'content_sha256': a['ch'], 'reason': 'window_wt2_overlap', 'shares': shares,
                             'window_wt2_share': wshare})
                continue
        toks.append(w)
        acc_sh.append(sh)
        if 'wt2' in rules and 'previous' in rules:                 # the existing N manifest layout (field order kept)
            shares = {'wt2': shares['wt2'], 'previous': shares['previous'], 'accepted': shares['accepted'],
                      'qwen_window': shares['qwen_window'], 'qwen_article': shares['qwen_article']}
        segs.append({'index': len(segs), 'source': source, 'file_lines': [a['line_start'], a['line_end']],
                     'title': a['title'], 'normalized_title': a['nt'], 'content_sha256': a['ch'], 'article_tokens': len(ids),
                     'window_tokens': [st, st + span * SEQ], 'window_chars_in_article': [int(off[st][0]), int(off[st + span * SEQ - 1][1])],
                     'max_shingle_share': shares, 'window_wt2_share': wshare, 'tokens_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
                     'windows': [span * len(segs) + i for i in range(span)]})
        if len(segs) == n_need:
            break
    return segs, toks, skips, excl


def sampler_check(ctx):
    """the reimplemented draw must reproduce the previous round's new articles (seeds, used set and audit index of that round)."""
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    pl, dz = read(L26 / 'plan.json'), read(L26 / 'data_freeze.json')
    used_then, _ = wms.used_wt103_verified(L26, ctx['w103'], ctx['wt2'])
    au = read(L26 / 'audit.json')
    if sha(L26 / 'work/qwen_hist_shingles.npz') != au['index_sha256']:
        raise SystemExit('previous audit index changed')
    hz = np.load(L26 / 'work/qwen_hist_shingles.npz')
    hist = defaultdict(set)
    for s_, w in zip(hz['shingle'].tolist(), hz['window'].tolist()):
        hist[s_].add(w)
    cx = dict(ctx, hist=hist, wsize=Counter(hz['window'].tolist()))
    elig = [a for a in ctx['base'] if a['ch'] not in used_then]
    prev_sh = [art_sh(ctx, ctx['by'][h]) for h in used_then]
    seeds = pl['data']['seeds']
    segs, toks, skips, _ = draw(cx, elig, ctx['lines103'], np.random.default_rng(seeds['order']).permutation(len(elig)),
                                np.random.default_rng(seeds['start']), pl['data']['n_articles'], pl['data']['windows_per_article'],
                                ('wt2', 'previous', 'qwen', 'window_wt2'), prev_sh, [], 'wikitext-103-raw-v1 train')
    ref_tok, ref_seg = np.load(L26 / 'tokens.npy'), read(L26 / 'segments.json')
    res = {'used_then': len(used_then), 'used_then_expected': dz['excluded_previous_wt103'], 'eligible_then': len(elig),
           'eligible_then_expected': dz['eligible'], 'tokens_equal': bool(np.array_equal(np.concatenate(toks), ref_tok)),
           'segments_equal': json.loads(json.dumps(segs)) == ref_seg, 'skips': dict(skips), 'skips_expected': dz['skips'],
           'seconds': time.perf_counter() - t0}
    res['passed'] = bool(res['used_then'] == res['used_then_expected'] and res['eligible_then'] == res['eligible_then_expected']
                         and res['tokens_equal'] and res['segments_equal'] and res['skips'] == res['skips_expected'])
    return res


def self_audit(ctx, files):
    """the future audit (run_qwen_down_fp16_protection.audit rule) must resolve every window of this round's token files
    ({tag: path})."""
    import run_olmoe_route_mass_restore as mr
    from run_qwen_down_spike_diag import window_shingles
    res = {}
    for tag, p in files.items():
        seg = p.parent / 'segments.json'
        listed = [mr.shingles(ctx['by'][s['content_sha256']]['body_norm']) for s in read(seg) if s.get('content_sha256') in ctx['by']]
        c = Counter()
        for w in np.load(p):
            sh = window_shingles(ctx['tok'].decode(w.tolist()))
            if len(sh) < 10:
                c['short'] += 1
            elif len(sh & ctx['U2']) / len(sh) >= 0.5:
                c['wikitext-2'] += 1
            elif listed and max(len(sh & o) / len(sh) for o in listed) >= 0.5:
                c['wikitext-103 listed in the run manifest'] += 1
            else:
                c['unresolved'] += 1
        res[tag] = {'path': prs.rel(p), 'windows': dict(c)}
    return res


def make_groups(titles):
    """C0 + per source / pool: single a = M[a, 0..9], multi r: row j = M[(j - r) mod 10, j]; rows = [source, row of its token file]."""
    groups = [{'id': 'C0', 'source': 'WT2', 'coverage': 'C0', 'pool': None, 'rows': [['C0', j] for j in range(CW)],
               'articles': [C0_TITLE] * CW, 'note': 'runs/native_eps_fix_20261014/qwen3/calib_inputs.npy; historical reference only'}]
    for s in SOURCES:
        for p in range(POOLS):
            M = [[(p * PER_POOL + a) * CW + j for j in range(CW)] for a in range(PER_POOL)]
            for a in range(PER_POOL):
                groups.append({'id': f'{s}.p{p}.single.a{a}', 'source': s, 'coverage': 'single', 'pool': p, 'index': a,
                               'rows': [[s, M[a][j]] for j in range(CW)], 'articles': [titles[s][p * PER_POOL + a]] * CW,
                               'article_in_pool': [a] * CW})
            for r in range(PER_POOL):
                aa = [(j - r) % PER_POOL for j in range(CW)]
                groups.append({'id': f'{s}.p{p}.multi.r{r}', 'source': s, 'coverage': 'multi', 'pool': p, 'index': r,
                               'rows': [[s, M[aa[j]][j]] for j in range(CW)], 'articles': [titles[s][p * PER_POOL + a] for a in aa],
                               'article_in_pool': aa})
    return groups


def calibdata(out):
    import run_olmoe_route_mass_restore as mr
    import wt_manifest_scan as wms
    t_all = time.perf_counter()
    pl = check_plan(out)
    if (out / 'calibration_data_freeze.json').exists():
        raise SystemExit('calibration data exists')
    au = read(out / 'audit.json')
    if au['unresolved_files'] or sha(out / 'work/qwen_hist_shingles.npz') != au['index_sha256']:
        raise SystemExit('audit unresolved or index changed')
    tm = {}
    ctx = corpus_ctx(out / 'work/qwen_hist_shingles.npz')
    tm['corpus_load_s'] = ctx['load_s']
    used, scan_rows = wms.used_wt103_verified(out, ctx['w103'], ctx['wt2'])
    if len(used) != pl['used_wt103_count'] or hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest() != pl['used_wt103_sha256']:
        raise SystemExit('previous WT103 set changed since the plan')
    chk = sampler_check(ctx)
    tm['sampler_check_s'] = chk['seconds']
    write(out / 'sampler_check.json', chk)
    log(f'sampler check: {chk}')
    if not chk['passed']:
        raise SystemExit('the reimplemented draw does not reproduce the previous round')
    # WT2 validation pools
    t0 = time.perf_counter()
    val = mr.load_lines([WT2_VAL])
    arts2 = mr.split_articles(val)
    c0 = arts2[0]
    if c0['title'] != C0_TITLE:
        raise SystemExit('C0 article')
    acc_sh = []
    el2 = arts2[1:]
    seg2, tok2, sk2, ex2 = draw(ctx, el2, val, np.random.default_rng(SEEDS['wt2_order']).permutation(len(el2)),
                                np.random.default_rng([SEEDS['span_start'], 0]), POOLS * PER_POOL, CW, ('ref',), [], acc_sh,
                                SRC_LABEL['WT2'], ref_sh=art_sh(ctx, c0))
    tm['wt2_draw_s'] = time.perf_counter() - t0
    # WT103 train pools
    t0 = time.perf_counter()
    el103 = [a for a in ctx['base'] if a['ch'] not in used]
    prev_sh = [art_sh(ctx, ctx['by'][h]) for h in used]
    seg103, tok103, sk103, ex103 = draw(ctx, el103, ctx['lines103'], np.random.default_rng(SEEDS['wt103_order']).permutation(len(el103)),
                                        np.random.default_rng([SEEDS['span_start'], 1]), POOLS * PER_POOL, CW,
                                        ('wt2', 'previous', 'qwen', 'window_wt2'), prev_sh, acc_sh, SRC_LABEL['WT103'])
    tm['wt103_draw_s'] = time.perf_counter() - t0
    if len(seg2) < POOLS * PER_POOL or len(seg103) < POOLS * PER_POOL:
        write(out / 'calibration_draw_failed.json', {'wt2': len(seg2), 'wt103': len(seg103), 'skips': [dict(sk2), dict(sk103)]})
        raise SystemExit('not enough calibration articles')
    files = {}
    for s, segs, toks, seed in (('WT2', seg2, tok2, SEEDS['wt2_order']), ('WT103', seg103, tok103, SEEDS['wt103_order'])):
        d = out / 'calib' / s.lower()
        d.mkdir(parents=True, exist_ok=True)
        for i, sg in enumerate(segs):
            p, a = divmod(i, PER_POOL)
            sg.update({'pool': p, 'article_in_pool': a, 'windows': [i * CW + j for j in range(CW)], 'seed_order': seed,
                       'seed_start': [SEEDS['span_start'], SOURCES.index(s)], 'span_tokens': CW * SEQ,
                       'window_token_starts': [sg['window_tokens'][0] + j * SEQ for j in range(CW)],
                       'window_tokens_sha256': [hashlib.sha256(w.tobytes()).hexdigest() for w in toks[i]]})
            if s == 'WT2':
                sg['raw_rows_in_validation_parquet'] = sg['file_lines']
        arr = np.concatenate(toks)
        np.save(d / 'tokens.npy', arr)
        write(d / 'segments.json', segs)
        files[s] = d / 'tokens.npy'
    groups = make_groups({'WT2': [sg['title'] for sg in seg2], 'WT103': [sg['title'] for sg in seg103]})
    if [g['id'] for g in groups] != group_ids():
        raise SystemExit('group order')
    arrs = {'C0': np.load(NATIVE / 'calib_inputs.npy'), 'WT2': np.load(files['WT2']), 'WT103': np.load(files['WT103'])}
    for g in groups:
        g['tokens_sha256'] = hashlib.sha256(np.stack([arrs[s_][r_] for s_, r_ in g['rows']]).tobytes()).hexdigest()
    write(out / 'groups.json', groups)
    # partition controls (before any model computation)
    part = {}
    for s in SOURCES:
        for p in range(POOLS):
            sing = [tuple(r_) + (j,) for g in groups if g['source'] == s and g['pool'] == p and g['coverage'] == 'single' for j, r_ in enumerate(g['rows'])]
            mult = [tuple(r_) + (j,) for g in groups if g['source'] == s and g['pool'] == p and g['coverage'] == 'multi' for j, r_ in enumerate(g['rows'])]
            ok_multi = all(len(set(g['article_in_pool'])) == PER_POOL for g in groups if g['source'] == s and g['pool'] == p and g['coverage'] == 'multi')
            part[f'{s}.p{p}'] = {'single_windows_once': len(sing) == len(set(sing)) == PER_POOL * CW,
                                 'single_equals_multi_with_rows': sorted(sing) == sorted(mult), 'multi_10_distinct_articles': ok_multi}
    part_ok = all(all(v.values()) for v in part.values())
    ids_all = [sg['content_sha256'] for sg in seg2 + seg103]
    distinct = len(set(ids_all)) == len(ids_all) and c0['ch'] not in ids_all and not (set(sg['content_sha256'] for sg in seg103) & used)
    tm['self_audit_s'] = -time.perf_counter()
    sa = self_audit(ctx, files)
    tm['self_audit_s'] += time.perf_counter()
    sa_ok = set(sa['WT2']['windows']) <= {'wikitext-2', 'short'} and set(sa['WT103']['windows']) <= {'wikitext-103 listed in the run manifest', 'short'}
    write(out / 'calibration_exclusions.json', {'WT2': {'skips': dict(sk2), 'skipped_during_draw': ex2},
                                                 'WT103': {'skips': dict(sk103), 'skipped_during_draw': ex103,
                                                           'previous_wt103_excluded_by_scan': len(used)}})
    fz = {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'audit_sha256': sha(out / 'audit.json'),
          'sampler_check_sha256': sha(out / 'sampler_check.json'), 'groups_sha256': sha(out / 'groups.json'),
          'files': {prs.rel(p): sha(p) for p in (files['WT2'], files['WT103'], files['WT2'].parent / 'segments.json',
                                                 files['WT103'].parent / 'segments.json', out / 'calibration_exclusions.json')},
          'C0_tokens_sha256': sha(NATIVE / 'calib_inputs.npy'), 'ranges_sha256': sha(RANGES), 'A_sha256': sha(ENDP / 'configs/A_auto.json'),
          'articles': {'WT2': len(seg2), 'WT103': len(seg103)}, 'unique_windows': len(arrs['WT2']) + len(arrs['WT103']),
          'groups': len(groups), 'tokens_per_group': CW * SEQ, 'targets_per_group': CW * TP,
          'visited': {'WT2': len(seg2) + sum(sk2.values()), 'WT103': len(seg103) + sum(sk103.values())},
          'skips': {'WT2': dict(sk2), 'WT103': dict(sk103)}, 'partition': part, 'partition_passed': part_ok, 'articles_distinct': distinct,
          'self_audit': sa, 'self_audit_passed': sa_ok, 'seeds': SEEDS, 'timing': tm}
    write(out / 'calibration_data_freeze.json', fz)
    ca.log_stage(out, 'calibdata', time.perf_counter() - t_all, tm)
    log(f'calibration data: WT2 {len(seg2)} (skips {dict(sk2)}), WT103 {len(seg103)} (skips {dict(sk103)}); partition {part_ok}; '
        f'distinct {distinct}; self-audit {sa}')
    if not (part_ok and distinct and sa_ok):
        raise SystemExit('calibration data controls failed')


# --------------------------------------------------------------------------- #
# calibration measurement (GPU)
# --------------------------------------------------------------------------- #
def proxies(env, U, gu, L, ids):
    """S_u, K2_actual (batch-8 logits) and Khat256 (microbatch-1 p): the formulas and precision of
    run_qwen3_l26_suffix_bridge.head_terms. Returns {term: [formats, windows, 127]}."""
    torch = env.torch
    W, lm = env.W, env.lm
    n = len(gu)
    u0 = U['f16']
    res = {t: np.zeros((len(FORMATS), n, TP)) for t in ('S_u', 'K2_actual', 'Khat256')}
    with torch.no_grad():
        for i in range(0, n, BATCH):
            sl = slice(i, i + BATCH)
            nb = len(gu[sl])
            l0 = lm(u0[sl])
            lp0 = [torch.log_softmax(l0[j, :-1].double(), -1) for j in range(nb)]
            for fi, f in enumerate(FORMATS):
                lf = lm(U[f][sl])
                Su = (gu[sl].double() * (U[f][sl].double() - u0[sl].double())).sum(-1)
                k2 = torch.empty(nb, TP, dtype=torch.float64, device=gu.device)
                for j in range(nb):
                    dl = lf[j, :-1].double() - l0[j, :-1].double()
                    p0 = lp0[j].exp()
                    m1 = (p0 * dl).sum(-1)
                    k2[j] = 0.5 * ((p0 * dl * dl).sum(-1) - m1 * m1)
                res['S_u'][fi, i:i + nb] = Su[:, :-1].cpu().numpy()
                res['K2_actual'][fi, i:i + nb] = k2.cpu().numpy()
                del lf
            del l0, lp0
        for j in range(n):
            p = torch.softmax(L[j].double(), -1)
            pr = tk.prep(p, W, K, tk.topk_sel(p, K))
            du = torch.stack([U[f][j, :-1] - u0[j, :-1] for f in FORMATS])
            res['Khat256'][:, j] = tk.khat(pr, du).cpu().numpy()
            del p, pr, du
    return res


def kwargs_ok(env, kw):
    torch = env.torch
    ar = torch.arange(SEQ, device='cuda')
    mask = torch.full((SEQ, SEQ), torch.finfo(torch.float32).min, device='cuda').triu(1)[None, None]
    return all(k_.get('past_key_values') is None and not k_.get('use_cache', False) and torch.equal(k_['position_ids'][0], ar)
               and torch.equal(k_['cache_position'], ar) and torch.equal(k_['attention_mask'], mask) for k_ in kw)


def calib(out, smoke_c0_only=False):
    t_all = time.perf_counter()
    check_plan(out)
    cdf = read(out / 'calibration_data_freeze.json')
    for f, h in cdf['files'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'calibration data changed: {f}')
    if sha(out / 'groups.json') != cdf['groups_sha256']:
        raise SystemExit('groups changed')
    groups = read(out / 'groups.json')
    if smoke_c0_only:
        groups = groups[:1]
    arrs = {'C0': np.load(NATIVE / 'calib_inputs.npy'), 'WT2': np.load(out / 'calib/wt2/tokens.npy'), 'WT103': np.load(out / 'calib/wt103/tokens.npy')}
    tm = defaultdict(float)
    t0 = time.perf_counter()
    env = l26.Env(out / 'work')
    torch = env.torch
    tm['replay_init_s'] = time.perf_counter() - t0
    from run_olmoe_dynamic_route import env_info
    ctl = {'environment': env_info(), 'head': env.head_checks()}
    A = read(ENDP / 'configs/A_auto.json')
    t0 = time.perf_counter()
    env.set_background(l26.background(A, 'F'))
    torch.cuda.synchronize()
    tm['set_background_s'] = time.perf_counter() - t0
    ctl['config'] = env.config_checks()
    rfile = read(RANGES)
    rh0 = ranges_hash(env.rp.ranges)
    ctl['ranges'] = {'file_sha256': sha(RANGES), 'lmreplay_equals_file': env.rp.ranges == rfile, 'lmreplay_ranges_hash': rh0,
                     'target_range_object_is_lmreplay': env.rng is env.rp.ranges[TARGET], 'target_x': env.rng['x'], 'target_y': env.rng['y'],
                     'target_int_qparams': {s_: int_qparams(env.rng[s_]) for s_ in ('x', 'y')}, 'changed_after_group': []}
    ctl['posit_lut_verified'] = {f: list(env.rp.q.posit[f].verify()) for f in ('p4e1', 'p6e1', 'p8e1')}
    ctl['tf32'] = [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = torch.are_deterministic_algorithms_enabled()
    torch.cuda.reset_peak_memory_stats()
    ce_all, px_all = {}, {}
    worst = defaultdict(float)
    exc = defaultdict(lambda: -np.inf)
    kw_ok, kw_shapes = True, set()
    c0_first = None

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()
    for gi, g in enumerate(groups):
        toks = np.stack([arrs[s_][r_] for s_, r_ in g['rows']])
        if hashlib.sha256(toks.tobytes()).hexdigest() != g['tokens_sha256']:
            raise SystemExit(f'{g["id"]}: tokens differ from groups.json')
        ids = torch.from_numpy(toks).cuda()
        ctrl = g['id'] in CONTROL_GROUPS
        t0 = sync()
        ce0, cap = env.full(toks, 'f16', capture=True)
        t1 = sync()
        tm['prefix_s'] += t1 - t0
        kw_ok = kw_ok and kwargs_ok(env, cap['kw'])
        kw_shapes.add((len(toks), len(cap['kw'])))
        if g['id'] == 'C0' and c0_first is None:
            c0_first = ce0.clone()
        U, H27 = {}, {}
        for f in FORMATS:
            z = env.zf(cap['x'], f)
            if ctrl:
                U[f], H27[f] = env.suffix(cap['r'], z, cap['kw'], keep_h27=True)
            else:
                U[f] = env.suffix(cap['r'], z, cap['kw'])
        t2 = sync()
        tm['candidate_suffix_s'] += t2 - t1
        ce = torch.stack([env.head_ce(U[f], ids) for f in FORMATS])
        t3 = sync()
        tm['head_ce_s'] += t3 - t2
        worst['f16_replay_vs_full_ce'] = max(worst['f16_replay_vs_full_ce'], float((ce[FORMATS.index('f16')] - ce0).abs().max()))
        L = {}
        gu = env.grad_pass(U['f16'], ids, on_logits=lambda j, l: L.__setitem__(j, l.clone()))
        worst['pos127_g_abs'] = max(worst['pos127_g_abs'], float(gu[:, -1].abs().max()))
        t4 = sync()
        tm['grad_pass_s'] += t4 - t3
        px = proxies(env, U, gu, L, ids)
        t5 = sync()
        tm['proxies_s'] += t5 - t4
        if ctrl:
            for fi, f in enumerate(FORMATS):
                cef, capf = env.full(toks, f, capture=True)
                for nm_, a_, r_ in (('h27', H27[f], capf['h27']), ('u', U[f], capf['u'])):
                    e, d = l26.tensor_excess(a_, r_)
                    exc[f'replay_{nm_}'] = max(exc[f'replay_{nm_}'], e)
                    worst[f'replay_{nm_}_max_abs'] = max(worst[f'replay_{nm_}_max_abs'], d)
                worst['replay_ce_max_abs'] = max(worst['replay_ce_max_abs'], float((ce[fi] - cef).abs().max()))
                del capf
            tm['control_full_forward_s'] += sync() - t5
        ce_all[g['id']] = ce.cpu().numpy()
        px_all[g['id']] = px
        if ranges_hash(env.rp.ranges) != rh0 or env.rng is not env.rp.ranges[TARGET]:
            ctl['ranges']['changed_after_group'].append(g['id'])
        del cap, U, H27, L, gu, ce, ce0
        if gi % 20 == 0:
            log(f'group {gi + 1}/{len(groups)} {g["id"]} done')
    # state drift: C0 again at the end
    ce_end = env.full(arrs['C0'], 'f16')
    ctl['end_reproduce_C0_abs'] = float((ce_end - c0_first).abs().max())
    ctl['config_end'] = env.config_checks()
    ctl['kwargs_ok'], ctl['batch_shapes'] = bool(kw_ok), sorted(kw_shapes)
    ctl['worst'], ctl['excess'] = dict(worst), dict(exc)
    # C0 reproduces the previous round (F)
    prev = np.load(L26 / 'C_terms.npz')
    fi16 = FORMATS.index('f16')
    ctl['reproduce_prev_C0_ce_abs'] = max(float(np.abs(ce_all['C0'][fi] - prev[f'F__{f}__CE']).max()) for fi, f in enumerate(FORMATS))
    ctl['reproduce_prev_C0_proxy_abs'] = {t: max(float(np.abs(px_all['C0'][t][fi] - prev[f'F__{f}__{t}']).max()) for fi, f in enumerate(FORMATS))
                                          for t in ('S_u', 'Khat256', 'K2_actual')}
    ctl['f16_abs'] = max(float(np.abs(px_all[g][t][fi16]).max()) for g in px_all for t in ('S_u', 'Khat256', 'K2_actual'))
    # shared windows and pooled tables
    if not smoke_c0_only:
        ctl.update(pool_controls(groups, ce_all, px_all))
    psum = 0.0
    for g in ce_all:
        D = ce_all[g] - ce_all[g][fi16]
        psum = max(psum, float(np.abs(D[:, :, 0].sum((1,)) + D[:, :, 1:].sum((1, 2)) - D.sum((1, 2))).max()))
    ctl['position_sum_abs'] = psum
    cfg_ok = all(c['weights_equal_config'] and c['activation_map_equal_config'] and c['target_weight_is_cached_int6']
                 and c['target_activation_in_lmreplay'] == 'fp32' and c['hooks_target'] == [2, 2] and c['hooks_block27_pre'] == 1
                 and c['hooks_final_norm'] == [0, 1] and c['hooks_lm_head'] == [0, 0] and c['hooks_other_projections'] == {'1,1': 195}
                 for c in (ctl['config'], ctl['config_end']))
    checks = {'head': ctl['head']['passed'], 'config': cfg_ok, 'ranges_equal_file': ctl['ranges']['lmreplay_equals_file'],
              'ranges_unchanged': not ctl['ranges']['changed_after_group'], 'target_range_object': ctl['ranges']['target_range_object_is_lmreplay'],
              'posit_lut': all(v[0] == 0 for v in ctl['posit_lut_verified'].values()), 'tf32_off': ctl['tf32'] == [False, False],
              'deterministic': ctl['deterministic'], 'kwargs': ctl['kwargs_ok'], 'batch_shapes': ctl['batch_shapes'] == [(CW, 2)],
              'f16_replay': worst['f16_replay_vs_full_ce'] <= TOL['f16_replay_vs_full_ce_abs_per_position'],
              'replay_tensors': exc['replay_h27'] <= 0 and exc['replay_u'] <= 0,
              'replay_ce': worst['replay_ce_max_abs'] <= TOL['replay_vs_full_ce_abs_per_position'],
              'reproduce_prev_C0_ce': ctl['reproduce_prev_C0_ce_abs'] <= TOL['reproduce_prev_C0_ce_abs_per_position'],
              'reproduce_prev_C0_proxy': max(ctl['reproduce_prev_C0_proxy_abs'].values()) <= TOL['reproduce_prev_C0_proxy_abs_per_position'],
              'f16_zero': ctl['f16_abs'] <= TOL['f16_zero_abs'], 'pos127_g': worst['pos127_g_abs'] == 0,
              'position_sum': psum <= TOL['position_sum_abs'], 'end_reproduce_C0': ctl['end_reproduce_C0_abs'] <= TOL['end_reproduce_C0_abs_per_position']}
    if not smoke_c0_only:
        checks.update({'shared_window_ce': ctl['shared_window_ce_max_abs'] <= TOL['shared_window_ce_abs_per_position'],
                       'shared_window_proxy': ctl['shared_window_proxy_max_abs'] <= TOL['shared_window_proxy_abs_per_position'],
                       'pooled_table': max(ctl['pooled_table_max_abs'].values()) <= TOL['pooled_table_abs'],
                       'pooled_pick': all(a == b for a, b in ctl['pooled_picks_single_multi'].values())})
    np.savez(out / 'C_values.npz', **{f'{g}__CE': v for g, v in ce_all.items()},
             **{f'{g}__{t}': v[t] for g, v in px_all.items() for t in ('S_u', 'Khat256', 'K2_actual')})
    summ = summarize_C(ce_all, px_all, [g['id'] for g in groups])
    ctl['C0_picks'] = summ['picks']['C0']
    ctl['C0_lowo'] = {s_: Counter(summ['lowo']['C0'][s_]['picks']) for s_ in SELECTORS}
    dg = {}
    with open(DIAG / 'leave_one_window_out.csv') as fh:
        for r in csv.DictReader(fh):
            if r['background'] == 'F':
                dg.setdefault({'D': 'D_C', 'P256': 'P256', 'Su+K2': 'Su+K2actual'}.get(r['score'], r['score']), []).append(r['pick'])
    ctl['C0_lowo_reproduces_diagnostic'] = {s_: dg[s_] == summ['lowo']['C0'][s_]['picks'] for s_ in SELECTORS}
    checks['C0_pick_int8'] = summ['picks']['C0']['D_C']['pick'] == 'int8'
    checks['C0_lowo_diagnostic'] = all(ctl['C0_lowo_reproduces_diagnostic'].values())
    ctl['checks'] = checks
    ctl['passed'] = bool(all(checks.values()))
    write(out / 'controls_calib.json', ctl)
    write(out / 'frozen_picks.json', {'rule': 'argmin over the selection set; exact ties: f16 first, then the set order', 'selection_set': list(SELECT_SET),
                                      'primary_selector': 'D_C', 'separate_secondary_selectors': ['P256', 'Su+K2actual'],
                                      'picks': summ['picks'], 'lowo': summ['lowo'], 'note': 'C only; frozen before any new article'})
    for name, rows in (('scores_C.csv', summ['rows']), ('windows_C.csv', summ['wrows'])):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    tm['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated())
    tm['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tm['groups'], tm['replayed_group_windows'] = len(groups), len(groups) * CW
    tm['total_s'] = time.perf_counter() - t_all
    ca.log_stage(out, 'calib' + ('_smoke_C0' if smoke_c0_only else ''), tm['total_s'], dict(tm))
    log(f'calib controls passed={ctl["passed"]}; failed {[k for k, v in checks.items() if not v]}; C0 picks '
        f'{ {s_: summ["picks"]["C0"][s_]["pick"] for s_ in SELECTORS} }; worst {dict(worst)}')
    if not ctl['passed']:
        raise SystemExit('calibration controls failed')
    if smoke_c0_only:
        return
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'calibration_data_freeze_sha256': sha(out / 'calibration_data_freeze.json'),
                                'controls_sha256': sha(out / 'controls_calib.json'), 'picks_sha256': sha(out / 'frozen_picks.json'),
                                'values_sha256': sha(out / 'C_values.npz'), 'scores_sha256': sha(out / 'scores_C.csv'),
                                'windows_sha256': sha(out / 'windows_C.csv'), 'A_sha256': sha(ENDP / 'configs/A_auto.json'),
                                'ranges_sha256': sha(RANGES), 'code_sha256': {f: sha(HERE / f) for f in CODE}})


def pool_controls(groups, ce_all, px_all):
    """every shared window: single vs multi per-position values; per pool: pooled d_C tables of the two coverages."""
    gid = {g['id']: g for g in groups}
    sh_ce, sh_px, bitwise, pooled, pooled_pick = 0.0, 0.0, True, {}, {}
    for s in SOURCES:
        for p in range(POOLS):
            for a in range(PER_POOL):
                for j in range(CW):
                    r = (j - a) % PER_POOL
                    gs, gm = f'{s}.p{p}.single.a{a}', f'{s}.p{p}.multi.r{r}'
                    if gid[gs]['rows'][j] != gid[gm]['rows'][j]:
                        raise SystemExit(f'layout {gs} {gm} {j}')
                    sh_ce = max(sh_ce, float(np.abs(ce_all[gs][:, j] - ce_all[gm][:, j]).max()))
                    bitwise = bitwise and bool(np.array_equal(ce_all[gs][:, j], ce_all[gm][:, j]))
                    for t in ('S_u', 'Khat256', 'K2_actual'):
                        sh_px = max(sh_px, float(np.abs(px_all[gs][t][:, j] - px_all[gm][t][:, j]).max()))
            tabs = {c: np.mean([dC_table(ce_all[g_]) for g_ in cell_groups(s, c, p)], axis=0) for c in COVS}
            pooled[f'{s}.p{p}'] = float(np.abs(tabs['single'] - tabs['multi']).max())
            pooled_pick[f'{s}.p{p}'] = [argmin_sel(dict(zip(FORMATS, tabs[c]))) for c in COVS]
    return {'shared_window_ce_max_abs': sh_ce, 'shared_window_proxy_max_abs': sh_px, 'shared_window_ce_bitwise': bitwise,
            'pooled_table_max_abs': pooled, 'pooled_picks_single_multi': pooled_pick}


def dC_table(ce):
    """[formats, windows, 127] CE -> d_C per format (mean over all targets, relative to f16)."""
    D = ce - ce[FORMATS.index('f16')]
    return D.sum((1, 2)) / (D.shape[1] * TP)


def argmin_sel(v):
    order = sorted(SELECT_SET, key=lambda f: (v[f], SELECT_SET.index(f)))
    return order[0]


def score_windows(ce, px, sel):
    """per-window sums [formats, windows] of the selector's per-position score (relative to f16 for D_C)."""
    if sel == 'D_C':
        D = ce - ce[FORMATS.index('f16')]
        return D.sum(2)
    t = {'P256': 'Khat256', 'Su+K2actual': 'K2_actual'}[sel]
    return (px['S_u'] + px[t]).sum(2)


def summarize_C(ce_all, px_all, gids):
    picks, lowo, rows, wrows = {}, {}, [], []
    fi = {f: i for i, f in enumerate(FORMATS)}
    for g in gids:
        ce, px = ce_all[g], px_all[g]
        nw = ce.shape[1]
        picks[g], lowo[g] = {}, {}
        for s_ in SELECTORS:
            Sw = score_windows(ce, px, s_)
            v = {f: float(Sw[fi[f]].sum() / (nw * TP)) for f in FORMATS}
            order = sorted(SELECT_SET, key=lambda f: (v[f], SELECT_SET.index(f)))
            picks[g][s_] = {'pick': order[0], 'order': order, 'margin_to_second': v[order[1]] - v[order[0]], 'second': order[1],
                            'per_token_All': v}
            lp, lm_ = [], []
            for w in range(nw):
                keep = [x for x in range(nw) if x != w]
                vv = {f: float(Sw[fi[f], keep].sum() / (len(keep) * TP)) for f in FORMATS}
                od = sorted(SELECT_SET, key=lambda f: (vv[f], SELECT_SET.index(f)))
                lp.append(od[0])
                lm_.append(vv[od[1]] - vv[od[0]])
            lowo[g][s_] = {'picks': lp, 'margins': lm_, 'changes': int(sum(x != order[0] for x in lp))}
        D = ce - ce[fi['f16']]
        wint8 = D[fi['int8']].sum(1) / TP
        picks[g]['int8_windows_pos_neg'] = [int((wint8 > 0).sum()), int((wint8 < 0).sum())]
        for f in FORMATS:
            r = {'group': g, 'format': f}
            for G, sl in (('P0', np.s_[:, 0:1]), ('Rest', np.s_[:, 1:]), ('All', np.s_[:, :])):
                r[f'd_C_{G}'] = float(D[fi[f]][sl].sum() / (nw * TP))
                for t in ('S_u', 'Khat256', 'K2_actual'):
                    r[f'{t}_{G}'] = float(px_all[g][t][fi[f]][sl].sum() / (nw * TP))
            r['P256_All'], r['Su+K2actual_All'] = r['S_u_All'] + r['Khat256_All'], r['S_u_All'] + r['K2_actual_All']
            rows.append(r)
            for w in range(nw):
                wrows.append({'group': g, 'window_row': w, 'format': f, 'D_window': float(D[fi[f], w].sum() / TP),
                              'P0': float(D[fi[f], w, 0] / TP), 'S_u': float(px['S_u'][fi[f], w].sum() / TP),
                              'Khat256': float(px['Khat256'][fi[f], w].sum() / TP), 'K2_actual': float(px['K2_actual'][fi[f], w].sum() / TP)})
    return {'picks': picks, 'lowo': lowo, 'rows': rows, 'wrows': wrows}


# --------------------------------------------------------------------------- #
# new articles (after the freeze)
# --------------------------------------------------------------------------- #
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
    ctx = corpus_ctx(out / 'work/qwen_hist_shingles.npz')
    used, _ = wms.used_wt103_verified(out, ctx['w103'], ctx['wt2'])
    if len(used) != pl['used_wt103_count'] or hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest() != pl['used_wt103_sha256']:
        raise SystemExit('previous WT103 set changed since the plan')
    cal103 = {s['content_sha256'] for s in read(out / 'calib/wt103/segments.json')}
    cal2 = {s['content_sha256'] for s in read(out / 'calib/wt2/segments.json')}
    excl_all = used | cal103
    elig = [a for a in ctx['base'] if a['ch'] not in excl_all]
    prev_sh = [art_sh(ctx, ctx['by'][h]) for h in excl_all]
    segs, toks, skips, excl = draw(ctx, elig, ctx['lines103'], np.random.default_rng(SEEDS['n_order']).permutation(len(elig)),
                                   np.random.default_rng(SEEDS['n_start']), N_ART, WPA, ('wt2', 'previous', 'qwen', 'window_wt2'),
                                   prev_sh, [], 'wikitext-103-raw-v1 train')
    got = {s['content_sha256'] for s in segs}
    old_n = {s['content_sha256'] for s in read(L26 / 'segments.json')}
    if len(segs) < N_ART or got & excl_all or got & cal2 or got & old_n:
        raise SystemExit('not enough articles or overlap')
    np.save(out / 'tokens.npy', np.concatenate(toks))
    write(out / 'segments.json', segs)
    write(out / 'exclusions.json', {'skipped_during_draw': excl, 'previous_wt103_excluded_by_scan': sorted(used),
                                    'calibration_wt103_excluded': sorted(cal103), 'calibration_wt2_articles_excluded_by_wt2_rules': sorted(cal2),
                                    'wt2_and_internal_duplicates_excluded_before_draw': len(ctx['w103']) - len(ctx['base'])})
    sa = self_audit(ctx, {'N': out / 'tokens.npy'})
    write(out / 'data_freeze.json', {'time': now(), 'freeze_sha256': sha(out / 'freeze.json'), 'audit_sha256': sha(out / 'audit.json'),
                                     'n_articles': N_ART, 'windows_per_article': WPA, 'eligible': len(elig), 'visited': N_ART + sum(skips.values()),
                                     'skips': dict(skips), 'excluded_previous_wt103': len(used), 'excluded_calibration_wt103': len(cal103),
                                     'tokens_sha256': sha(out / 'tokens.npy'), 'segments_sha256': sha(out / 'segments.json'),
                                     'exclusions_sha256': sha(out / 'exclusions.json'), 'self_audit': sa, 'seeds': {k: SEEDS[k] for k in ('n_order', 'n_start')},
                                     'corpus_load_s': ctx['load_s']})
    ca.log_stage(out, 'data', time.perf_counter() - t_all, {'corpus_load_s': ctx['load_s']})
    log(f'data: {N_ART} articles; skips {dict(skips)}; self-audit {sa}')
    if set(sa['N']['windows']) - {'wikitext-103 listed in the run manifest', 'short'}:
        raise SystemExit('new-article tokens would not resolve in a later audit')


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz_ = read(out / 'data_freeze.json')
    fz = read(out / 'freeze.json')
    if sha(out / 'tokens.npy') != dz_['tokens_sha256'] or sha(out / 'freeze.json') != dz_['freeze_sha256'] or \
            sha(out / 'frozen_picks.json') != fz['picks_sha256']:
        raise SystemExit('data / freeze changed')
    env = l26.Env(out / 'work')
    torch = env.torch
    env.set_background(l26.background(read(ENDP / 'configs/A_auto.json'), 'F'))
    cfg = env.config_checks()
    torch.cuda.reset_peak_memory_stats()
    rh0 = ranges_hash(env.rp.ranges)
    tok = np.load(out / 'tokens.npy')
    res = {f: np.zeros((len(tok), TP)) for f in FORMATS}
    dev = 0.0
    for c0 in range(0, len(tok), CHUNK):
        t = tok[c0:c0 + CHUNK]
        ids = torch.from_numpy(t).cuda()
        ce0, cap = env.full(t, 'f16', capture=True)
        for f in FORMATS:
            ce = env.head_ce(env.suffix(cap['r'], env.zf(cap['x'], f), cap['kw']), ids)
            res[f][c0:c0 + len(t)] = ce.cpu().numpy()
            if f == 'f16':
                dev = max(dev, float((ce - ce0).abs().max()))
        del cap
    np.savez(out / 'eval_ce_pos.npz', **res)
    ok = bool(dev <= TOL['f16_replay_vs_full_ce_abs_per_position'] and ranges_hash(env.rp.ranges) == rh0 and env.rp.ranges == read(RANGES)
              and cfg['target_weight_is_cached_int6'] and cfg['hooks_target'] == [2, 2])
    write(out / 'controls_eval.json', {'f16_replay_vs_full_ce_abs_per_position': dev, 'ranges_unchanged_and_equal_file': ranges_hash(env.rp.ranges) == rh0,
                                       'config': cfg, 'passed': ok})
    ca.log_stage(out, 'eval', time.perf_counter() - t0, {'windows': len(tok), 'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
                                                         'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    if not ok:
        raise SystemExit('new-article controls failed')


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def kendall_tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a == b:
                c += 1
            else:
                d += 1
    den = np.sqrt((c + d + tx) * (c + d + ty))
    return float((c - d) / den) if den else float('nan')


def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    fz, dz = read(out / 'freeze.json'), read(out / 'data_freeze.json')
    if sha(out / 'frozen_picks.json') != fz['picks_sha256'] or sha(out / 'tokens.npy') != dz['tokens_sha256']:
        raise SystemExit('freeze / data changed')
    z = np.load(out / 'eval_ce_pos.npz')
    fp = read(out / 'frozen_picks.json')
    picks = fp['picks']
    Cv = np.load(out / 'C_values.npz')
    na = N_ART
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, na, na), minlength=na) for _ in range(BOOT)]) / na
    sgn = lambda v: 1 if v > 0 else (-1 if v < 0 else 0)

    def st(v):
        bs = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(), 'median': float(np.median(v)),
                'articles_pos': int((v > 0).sum()), 'articles_neg': int((v < 0).sum()), 'articles_zero': int((v == 0).sum()),
                'min': float(v.min()), 'max': float(v.max())}

    def decisive(s):
        return s['ci95'][0] > 0 or s['ci95'][1] < 0

    def part(ce, G):
        c = ce.reshape(na, WPA, TP)
        return {'P0': c[:, :, 0].sum(1), 'Rest': c[:, :, 1:].sum((1, 2)), 'All': c.sum((1, 2))}[G] / (WPA * TP)
    D = {(f, G): part(z[f], G) - part(z['f16'], G) for f in FORMATS for G in POS}
    nmean = {f: float(D[(f, 'All')].mean()) for f in SELECT_SET}
    oracle = min(SELECT_SET, key=lambda f: (nmean[f], SELECT_SET.index(f)))
    pairs = [(SELECT_SET[i], SELECT_SET[j]) for i in range(len(SELECT_SET)) for j in range(i + 1, len(SELECT_SET))]
    pair_st = {(a, c): st(D[(a, 'All')] - D[(c, 'All')]) for a, c in pairs}
    # per group consequences and rank agreement
    grows, gres = [], {}
    for g in group_ids():
        gres[g] = {}
        for s_ in SELECTORS:
            pk = picks[g][s_]
            v = pk['per_token_All']
            dn = D[(pk['pick'], 'All')]
            rc = Counter()
            for a, c in pairs:
                ps = pair_st[(a, c)]
                rc['undetermined' if not decisive(ps) else ('right' if sgn(v[a] - v[c]) == sgn(ps['mean']) else 'wrong')] += 1
            gres[g][s_] = {'pick': pk['pick'], 'N_mean_D': float(dn.mean()), 'N_harmful_articles': int((dn > 0).sum()),
                           'regret_vs_N_mean_best': float(dn.mean()) - nmean[oracle],
                           'kendall_tau_b_C_vs_N': kendall_tau_b([v[f] for f in SELECT_SET], [nmean[f] for f in SELECT_SET]),
                           'decisive_pairs': dict(rc), 'C_margin': pk['margin_to_second'], 'lowo_changes': fp['lowo'][g][s_]['changes']}
            grp = g.split('.')
            grows.append({'group': g, 'source': 'WT2' if g == 'C0' else grp[0], 'pool': '' if g == 'C0' else grp[1],
                          'coverage': 'C0' if g == 'C0' else grp[2], 'selector': s_, 'pick': pk['pick'], 'second': pk['second'],
                          'C_pick_d': v[pk['pick']], 'C_margin': pk['margin_to_second'], 'C_int8_d': v['int8'],
                          'int8_windows_pos': picks[g]['int8_windows_pos_neg'][0], 'int8_windows_neg': picks[g]['int8_windows_pos_neg'][1],
                          'lowo_changes': fp['lowo'][g][s_]['changes'], 'lowo_picks': ' '.join(fp['lowo'][g][s_]['picks']),
                          'N_mean_D': gres[g][s_]['N_mean_D'], 'N_harmful_articles': gres[g][s_]['N_harmful_articles'],
                          'regret': gres[g][s_]['regret_vs_N_mean_best'], 'kendall_tau_b': gres[g][s_]['kendall_tau_b_C_vs_N'],
                          'pairs_right': rc['right'], 'pairs_wrong': rc['wrong'], 'pairs_undetermined': rc['undetermined']})

    def Lcell(s, c, sel, pool=None):
        return np.mean([D[(picks[g][sel]['pick'], 'All')] for g in cell_groups(s, c, pool)], axis=0)
    cells = {}
    for sel in SELECTORS:
        for s in SOURCES:
            for c in COVS:
                gs = cell_groups(s, c)
                Lv = Lcell(s, c, sel)
                cells[f'{sel}|{s}|{c}'] = {
                    'L_vs_f16': st(Lv), 'L_minus_C0_int8': st(Lv - D[('int8', 'All')]),
                    'pick_frequency': dict(Counter(picks[g][sel]['pick'] for g in gs)),
                    'pick_frequency_by_pool': {p: dict(Counter(picks[g][sel]['pick'] for g in cell_groups(s, c, p))) for p in range(POOLS)},
                    'C_margin_mean': float(np.mean([picks[g][sel]['margin_to_second'] for g in gs])),
                    'C_margin_min': float(np.min([picks[g][sel]['margin_to_second'] for g in gs])),
                    'lowo_change_rate': float(np.mean([fp['lowo'][g][sel]['changes'] for g in gs]) / CW),
                    'lowo_groups_with_any_change': int(sum(fp['lowo'][g][sel]['changes'] > 0 for g in gs)),
                    'group_N_mean_D_mean': float(np.mean([gres[g][sel]['N_mean_D'] for g in gs])),
                    'group_N_harmful_articles_mean': float(np.mean([gres[g][sel]['N_harmful_articles'] for g in gs])),
                    'regret_mean': float(np.mean([gres[g][sel]['regret_vs_N_mean_best'] for g in gs])),
                    'kendall_tau_b_mean': float(np.nanmean([gres[g][sel]['kendall_tau_b_C_vs_N'] for g in gs])),
                    'decisive_pairs_right_mean': float(np.mean([gres[g][sel]['decisive_pairs'].get('right', 0) for g in gs])),
                    'decisive_pairs_wrong_mean': float(np.mean([gres[g][sel]['decisive_pairs'].get('wrong', 0) for g in gs])),
                    'articles_with_L_positive': int((Lv > 0).sum())}
    comps = {}
    for sel in SELECTORS:
        for s in SOURCES:
            comps[f'{sel}|{s}|multi-single'] = st(Lcell(s, 'multi', sel) - Lcell(s, 'single', sel))
            for p in range(POOLS):
                comps[f'{sel}|{s}.p{p}|multi-single'] = st(Lcell(s, 'multi', sel, p) - Lcell(s, 'single', sel, p))
                for c in COVS:
                    comps[f'{sel}|{s}.p{p}|{c}_L'] = st(Lcell(s, c, sel, p))
        for c in COVS:
            comps[f'{sel}|WT103-WT2|{c}'] = st(Lcell('WT103', c, sel) - Lcell('WT2', c, sel))
    prim = comps['D_C|WT2|multi-single']
    lowo2 = {c: cells[f'D_C|WT2|{c}']['lowo_change_rate'] for c in COVS}
    improved = bool(prim['mean'] < 0 and prim['ci95'][1] < 0)
    primary = {'definition': 'delta_n = L_n(WT2, multi) - L_n(WT2, single), D_C picks', 'N': prim, 'confirmed_improvement': improved,
               'identical_pick_frequencies': cells['D_C|WT2|multi']['pick_frequency'] == cells['D_C|WT2|single']['pick_frequency'],
               'lowo_change_rate_WT2': lowo2, 'lowo_lower_for_multi': bool(lowo2['multi'] < lowo2['single']),
               'reading': ('coverage helps this L26 diagnostic (within these WT2 pools)' if improved and lowo2['multi'] < lowo2['single'] else
                           'N improvement without lower leave-one-window-out change rate' if improved else
                           'calibration picks more stable, no confirmed N improvement' if lowo2['multi'] < lowo2['single'] else
                           'no confirmed N improvement and no lower change rate')}
    # C -> N shift per format (pooled per source; single and multi pooled tables are identical by construction) and C0
    shift = {}
    for s in SOURCES + ('C0',):
        gs = ['C0'] if s == 'C0' else cell_groups(s, 'single')
        for f in FORMATS:
            for G, sl in (('P0', np.s_[:, 0]), ('Rest', np.s_[:, 1:]), ('All', np.s_[:, :])):
                cvals = [(Cv[f'{g}__CE'][FORMATS.index(f)] - Cv[f'{g}__CE'][FORMATS.index('f16')])[sl].sum() / (CW * TP) for g in gs]
                shift[f'{s}|{f}|{G}'] = {'C': float(np.mean(cvals)), 'N': float(D[(f, G)].mean()), 'N_minus_C': float(D[(f, G)].mean() - np.mean(cvals))}
    nrows = []
    for f in FORMATS:
        for G in POS:
            s = st(D[(f, G)])
            nrows.append({'format': f, 'G': G, 'mean': s['mean'], 'ci_lo': s['ci95'][0], 'ci_hi': s['ci95'][1], 'median': s['median'],
                          'articles_pos': s['articles_pos'], 'articles_neg': s['articles_neg']})
    write(out / 'stats.json', {'units': 'nat/token relative to f16 (background F, target weight INT6); article bootstrap over the 128 new articles',
                               'primary': primary, 'cells': cells, 'comparisons': comps, 'groups': gres,
                               'N_mean_D_All': nmean, 'N_mean_best_in_set_descriptive': oracle,
                               'N_pairs': {f'{a}-{c}': v for (a, c), v in pair_st.items()}, 'C_to_N_shift': shift,
                               'C0': {s_: gres['C0'][s_] for s_ in SELECTORS}})
    for name, rows in (('N_truth.csv', nrows), ('groups_summary.csv', grows)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    np.savez(out / 'N_article_D.npz', **{f'{f}__{G}': D[(f, G)] for f in FORMATS for G in POS})
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log(f'primary WT2 multi-single {prim["mean"]:+.5f} {prim["ci95"]}; improved {improved}; lowo {lowo2}; '
        f'freq single {cells["D_C|WT2|single"]["pick_frequency"]} multi {cells["D_C|WT2|multi"]["pick_frequency"]}')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'audit', 'calibdata', 'calib', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note')
    ap.add_argument('--smoke-c0-only', action='store_true', help='calib on C0 only, no freeze (smoke test)')
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
    elif a.stage == 'calib':
        calib(out, a.smoke_c0_only)
    else:
        {'audit': audit, 'calibdata': calibdata, 'data': data, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_l26_calib_coverage.py'),
                             'argv': sys.argv}) + '\n')


if __name__ == '__main__':
    main()
