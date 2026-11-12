#!/usr/bin/env python3
"""Original native BC: article-resampling rank stability versus held-out GPU QDQ loss.

Only new experiment artifacts are written. Native statistics use the original
compiled FP32 graphs; no derivative approximation, quantizer, or BC modification.
The truth experiment quantizes one Down activation (both slots), all weights FP32.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_first_order_crosslayer as cross
import run_qwen3_l26_calib_coverage as cov
import run_qwen_prefix_rank_shift as prs

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
NATIVE = cross.NATIVE
RANGES = cross.RANGES
LAYERS = (7, 15, 23)
TARGETS = {L: f'/model/model/layers.{L}/mlp/down_proj/MatMul' for L in LAYERS}
FORMATS = ('f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16', 'fp32')
SELECT = FORMATS[:-1]
SAME = (('int8', 'p8e1'), ('int6', 'p6e1'), ('int4', 'p4e1'))
PAIRS = [(a, b) for i, a in enumerate(SELECT) for b in SELECT[i + 1:]]
SEEDS = {'cal_order': 26102741, 'cal_start': 26102742,
         'n_order': 26102743, 'n_start': 26102744,
         'bootstrap': 26102745, 'cal_bootstrap': 26102746}
NSETS, PER_SET, N_ART, WPA, SEQ, TP = 3, 10, 128, 2, 128, 127
CBOOT, NBOOT = 5000, 20000
REL_TIE, STABLE = 1e-10, 0.95
RUNTIME = HERE.parents[1] / 'build/Release/lib'
CODE = list(dict.fromkeys([Path(__file__).name, 'run_qwen_first_order_crosslayer.py',
                          'run_native_bc_pipeline.py'] + cross.CODE))


def log(s):
    print(f'{now()} {s}', flush=True)


def csvwrite(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def code_hashes():
    return {p: sha(HERE / p) for p in CODE}


def check_plan(out, inputs_too=True):
    p = read(out / 'plan.json')
    am = out / 'amendments.json'
    h = read(am)[-1]['code_hashes'] if am.exists() else p['code_hashes']
    if code_hashes() != h:
        raise SystemExit('code changed; record an amendment before continuing')
    if inputs_too:
        for fn, hh in p['inputs'].items():
            if sha(HERE / fn) != hh:
                raise SystemExit(f'input changed: {fn}')
        for fn, m in p['large_inputs'].items():
            s = (HERE / fn).stat()
            if s.st_size != m['size'] or s.st_mtime_ns != m['mtime_ns']:
                raise SystemExit(f'large input changed: {fn}')
    return p


def plan(out):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    for p in list((HERE / 'runs').glob('*/plan.json')) + list((HERE / 'runs').glob('*/analysis_plan.json')):
        if p.parent != out and any(str(v) in p.read_text() for v in SEEDS.values()):
            raise SystemExit(f'seed collision: {p}')
    w103, wt2 = wms.corpora()
    used, manifests = wms.used_wt103_verified(out, w103, wt2)
    files = [RANGES, NATIVE / 'table.json', NATIVE / 'calib_inputs.npy',
             NATIVE / 'stats_grads/grads_layout.json', NATIVE / 'stats_grads/grads.so',
             NATIVE / 'stats_errors6/errors6_layout.json', NATIVE / 'stats_errors6/errors6.so',
             HERE / 'qwen3_assets/model/config.json', HERE / 'qwen3_assets/model/tokenizer.json']
    large = [NATIVE / 'stats_grads/model.constants.bin', NATIVE / 'stats_errors6/model.constants.bin',
             HERE / 'qwen3_assets/model/model.safetensors']
    pl = {'time': now(), 'question': 'Does original BC rank stability under article resampling distinguish correct from incorrect rankings?',
          'targets': TARGETS, 'background': 'all projections and target weights FP32; quantize only target activation input/output together',
          'criterion': 'original run_native_bc_pipeline.py a_cost, no formula change; native grads and errors6 .so reused',
          'formula': 'sum_slots,channels [G2/n*(S2-n*mu^2) + GS2*mu^2], mu=S1/n, n=N*128; '
                     'G2=sum_samples sum_positions(g^2), GS2=sum_samples(sum_positions g)^2; '
                     'S1/S2 are summed signed/squared quantization errors on original FP32 activations',
          'resampling': 'resample ten articles with replacement, jointly reaggregate G2, GS2, S1, S2 and recompute mu/variance; '
                        'never average single-article BC scores; scores divided by N for presentation, not comparable to NLL units',
          'formats': FORMATS, 'same_bit': SAME, 'seeds': SEEDS,
          'calibration': '3 disjoint sets x 10 new WT103 articles x one 128-token window, shared by all three targets',
          'data_protocol': 'reuse audited crosslayer calibdata/data samplers; exclude all history, WT2 and exact/near duplicates; '
                           'new N after BC ranking/stability freeze: 128 new articles x 2 windows',
          'cal_bootstrap': {'draws': CBOOT, 'unit': 'article (one window)', 'seed': SEEDS['cal_bootstrap'],
                            'same_resamples_for_targets': True, 'stability': 'fraction of resamples retaining the original pair direction; '
                            'ties get zero agreement; bootstrap fraction is not probability of correctness',
                            'stable_threshold': STABLE, 'threshold_purpose': 'descriptive bins only, no abstention or deployment rule'},
          'primary': '27 same-bit rows (3 targets x 3 sets x 3 bit); report stable/unstable x correct/wrong/undetermined on N. '
                     'Continuous stability-versus-correctness and normalized margins are descriptive. '
                     'Only 9 unique target/pair truths, shared across calibration sets: no independent-binomial significance test. '
                     'If every row is stable, there is no stable/unstable contrast, not proof that confidence is useful.',
          'secondary': 'all 28 pairs x 3 targets x 3 sets; three-set agreement; leave-one-article-out; C exact loss vs BC and N; '
                       'variance/bias components and calibration-window variation; do not tune cutoffs or reweight positions',
          'truth': 'GPU FP32+QDQ, existing crosslayer Env with target weight FP32; no training, no derivatives on N. '
                   'Report relative-f16 NLL for candidates; pairwise N differences independent of reference choice. '
                   'This is native BC with GPU truth, not an end-to-end native validation.',
          'N_statistics': {'bootstrap': NBOOT, 'unit': 'article, two windows averaged', 'seed': SEEDS['bootstrap'],
                           'ci': '95% paired percentile intervals; crosses zero -> undetermined; '
                                 'Holm-adjusted paired sign-flip p-values additionally reported for 9 unique same-bit pairs. '
                                 'Other pair intervals exploratory, no claims of independent replication.'},
          'numeric': {'relative_score_tie': REL_TIE, 'native_C0_table_rtol': 1e-8,
                      'native_C0_table_atol': 1e-10, 'replay_ce_abs': 1e-6, 'replay_tensor_abs_rel': [1e-6, 1e-6],
                      'native_gpu_baseline_ce_max_abs': 1e-3,
                      'note': 'baseline backend agreement does not guarantee candidate QDQ agreement'},
          'controls': 'native ten-window C0 aggregation must reproduce stored C table for all 3 targets/8 formats; '
                      'duplicate/repeated-article aggregation checks; ranges/qparams fixed; all calibration replay/full forwards equal; '
                      'FP32 target weights verified; holdout never used to pick; checks must pass before interpretation',
          'native_execution': {'workers': 4, 'samples': 'C0 10 (reproduction only), new calibration 30',
                               'modes': ['grads', 'errors6'], 'per_article_stats_saved': True},
          'boundaries': 'no change to BC, ranges, quantizers, search, one-first-order methods or other experiments; '
                        'no calibration-size change, no automatic confidence guard; no claim that instability alone proves the cause',
          'used_wt103_count': len(used), 'used_wt103_sha256': hashlib.sha256('\n'.join(sorted(used)).encode()).hexdigest(),
          'inputs': {prs.rel(p): sha(p) for p in files},
          'large_inputs': {prs.rel(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                                       'sha256': sha(p)} for p in large},
          'code_hashes': code_hashes(), 'smoke': 'none; plan precedes all new model results'}
    write(out / 'plan.json', pl)
    write(out / 'historical_manifests.json', manifests)
    (out / 'plan.md').write_text('# 原 BC 排名可靠性診斷\n\n' +
        f'預先登記時間：{pl["time"]}。完整規格見 plan.json。\n\n' +
        '- L7／L15／L23 Down，權重與其他投影全部 FP32，只量化目標 activation input/output。\n'
        '- 固定原 ranges。使用原 native 梯度與 errors6 共享庫，重現原 C0 成本表。\n'
        '- 新校準 3 組 × 10 篇 × 128 tokens；保留逐文章原始統計。\n'
        '- 5000 次文章重抽樣，每次重新聚合 BC 的平均／變異項；不可平均逐文章 BC。\n'
        '- 95% 保持原方向稱穩定，僅作描述分類，不部署信心規則。\n'
        '- BC 排名／穩定性全部凍結後，新抽 128 篇 × 2 窗口測 GPU QDQ 真值。\n'
        '- 主要 27 個同 bit 判讀只有 9 個不同 target/pair 真值；不視為 27 個獨立實驗。\n'
        '- 區分穩定且對、穩定但錯、不穩且對、不穩且錯、真值未定。\n'
        '- 不更改 criterion 或搜尋。Native BC + GPU 真值的後端限制必須揭露。\n')
    log(f'plan saved; exclude {len(used)} earlier articles')


def configure_helpers(out):
    cross.check_plan = check_plan
    cross.SEEDS = SEEDS
    cross.NSETS, cross.PER_SET, cross.N_ART, cross.WPA = NSETS, PER_SET, N_ART, WPA
    cross.log = log
    cov.check_plan = check_plan
    cov.log = log


def native_child(out, label, idx, mode):
    sys.path.insert(0, str(RUNTIME))
    from PyRuntime import OMExecutionSession
    td = out / 'calib/tokens.npy' if label == 'C' else NATIVE / 'calib_inputs.npy'
    tokens = np.load(td)[idx]
    sub = NATIVE / ('stats_grads' if mode == 'grads' else 'stats_errors6')
    stem = 'grads' if mode == 'grads' else 'errors6'
    lay = read(sub / f'{stem}_layout.json')
    sess = OMExecutionSession(str(sub / f'{stem}.so'), 'model')
    ins = [np.ascontiguousarray(tokens[None])]
    if mode == 'grads':
        target = np.zeros((1, SEQ, 151936), np.float32)
        target[0, np.arange(TP), tokens[1:]] = 1
        ins.append(target)
    t0 = time.perf_counter()
    values = sess.run(ins)
    elapsed = time.perf_counter() - t0
    v = np.asarray(values[lay['result_index']], np.float64).reshape(-1)
    taps = {t['tap']: t for t in lay['taps']}
    points = {(p['node'], p['slot']): taps[p['tap']] for p in lay['points']}
    result = {}
    for L, node in TARGETS.items():
        for slot in ('x', 'y'):
            t = points[(node, slot)]
            c = t['channels']
            assert t['positions'] == SEQ
            pre = f'L{L}__{slot}'
            if mode == 'grads':
                k = t['offset']
                result[pre + '__g2'] = v[k:k+c]
                result[pre + '__gs'] = v[k+c:k+2*c]
            else:
                for f in SELECT:
                    k = t['errors'][f]
                    result[pre + '__' + f + '__s1'] = v[k:k+c]
                    result[pre + '__' + f + '__s2'] = v[k+c:k+2*c]
    lg = np.asarray(values[0])[0, :TP].astype(np.float64)
    mx = lg.max(-1)
    result['baseline_ce'] = mx + np.log(np.exp(lg-mx[:, None]).sum(-1)) - lg[np.arange(TP), tokens[1:]]
    result['elapsed_s'] = np.array(elapsed)
    result['peak_rss_bytes'] = np.array(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    path = out / 'native' / f'{label}_{idx:02d}_{mode}.npz'
    np.savez(path, **result)
    log(f'native {label} {idx} {mode} {elapsed:.1f}s')


def native(out):
    check_plan(out)
    df = read(out / 'calibration_data_freeze.json')
    assert df['controls_passed'] and sha(out / 'calib/tokens.npy') == df['tokens_sha256']
    (out / 'native').mkdir(exist_ok=True)
    jobs = [(label, i, m) for label, n in [('C0', 10), ('C', 30)] for i in range(n) for m in ('grads', 'errors6')]
    def run(j):
        label, i, mode = j
        dest = out / 'native' / f'{label}_{i:02d}_{mode}.npz'
        if dest.exists():
            raise RuntimeError(f'output already exists: {dest}')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        cmd = [sys.executable, __file__, '--stage', 'native-child', '--out-dir', str(out),
               '--label', label, '--index', str(i), '--mode', mode]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out / 'logs' / f'{label}_{i:02d}_{mode}.log').write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(f'native child failed {j}: {p.stderr[-2000:]}')
        z = np.load(dest)
        return {'label': label, 'index': i, 'mode': mode, 'run_s': float(z['elapsed_s']),
                'rss': int(z['peak_rss_bytes'])}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            r = f.result()
            rows.append(r)
            log(f'native {len(rows)}/{len(jobs)}: {r["label"]} {r["index"]} {r["mode"]} {r["run_s"]:.1f}s')
    write(out / 'native_execution.json', rows)


def load_stats(out, label, indices, L):
    z = {}
    for slot in ('x', 'y'):
        p = f'L{L}__{slot}'
        g, e = [], []
        for i in indices:
            with np.load(out / 'native' / f'{label}_{i:02d}_grads.npz') as a:
                g.append((a[p+'__g2'].copy(), a[p+'__gs'].copy() ** 2))
            with np.load(out / 'native' / f'{label}_{i:02d}_errors6.npz') as a:
                e.append((np.stack([a[p+'__'+f+'__s1'] for f in SELECT]),
                          np.stack([a[p+'__'+f+'__s2'] for f in SELECT])))
        z[slot] = {'g2': np.stack([a[0] for a in g]), 'gc': np.stack([a[1] for a in g]),
                   's1': np.stack([a[0] for a in e]), 's2': np.stack([a[1] for a in e])}
    return z


def bc(z, weights):
    """Exact original aggregation with multiplicities; [draw, format] total BC."""
    w = np.atleast_2d(weights).astype(np.float64)
    count = w.sum(1)[:, None] * SEQ
    var, bias = np.zeros((len(w), len(SELECT))), np.zeros((len(w), len(SELECT)))
    for st in z.values():
        G, GS = w @ st['g2'], w @ st['gc']
        for fi in range(len(SELECT)):
            mu = (w @ st['s1'][:, fi]) / count
            er2 = w @ st['s2'][:, fi]
            var[:, fi] += (G / count * (er2 - count * mu ** 2)).sum(1)
            bias[:, fi] += (GS * mu ** 2).sum(1)
    return var + bias, var, bias


def normalized_gap(a, b):
    return (a-b) / (np.abs(a)+np.abs(b)+1e-30)


def direction(a, b):
    return np.where(np.abs(normalized_gap(a, b)) <= REL_TIE, 0, np.sign(a-b)).astype(int)


def freeze(out):
    check_plan(out)
    if (out / 'freeze.json').exists():
        raise SystemExit('freeze exists')
    native_ref = read(NATIVE / 'table.json')['C']
    worst = 0.0
    checks, reproductions = {}, []
    for L in LAYERS:
        z0 = load_stats(out, 'C0', range(10), L)
        v = bc(z0, np.ones(10))[0][0]
        ref = np.array([native_ref[TARGETS[L]][f] for f in SELECT])
        rel = np.abs(v-ref) / np.maximum(np.abs(ref), 1e-30)
        worst = max(worst, float(rel.max()))
        checks[f'L{L}_C0_reproduces_native_table'] = bool(np.allclose(v, ref, rtol=1e-8, atol=1e-10))
        for j, f in enumerate(SELECT):
            reproductions.append({'layer': L, 'format': f, 'new_C0': float(v[j]), 'stored_C0': float(ref[j]), 'relative_error': float(rel[j])})
        checks[f'L{L}_duplicate_scaling'] = bool(np.allclose(bc(z0, np.ones(10)*2)[0][0], 2*v, rtol=1e-12, atol=1e-12))
        ids = np.array([0, 0, 1, 2, 2, 2, 5, 7, 8, 9])
        zd = {slot: {k: a[ids] for k, a in s.items()} for slot, s in z0.items()}
        checks[f'L{L}_multiplicity_matches_explicit_duplicates'] = bool(np.allclose(
            bc(z0, np.bincount(ids, minlength=10))[0], bc(zd, np.ones(10))[0], rtol=1e-12, atol=1e-12))
    write(out / 'controls_native.json', {'checks': checks, 'passed': all(checks.values()), 'max_relative_C0_error': worst})
    csvwrite(out / 'native_C0_reproduction.csv', reproductions)
    if not all(checks.values()):
        raise SystemExit('native reproduction/aggregation checks failed')
    rng = np.random.default_rng(SEEDS['cal_bootstrap'])
    weights = np.stack([np.stack([np.bincount(x, minlength=10) for x in rng.integers(0, 10, (CBOOT, 10))]) for _ in range(NSETS)])
    np.savez(out / 'bootstrap_weights.npz', weights=weights)
    rows, scores, boot_save = [], [], {}
    for si in range(NSETS):
        for L in LAYERS:
            z = load_stats(out, 'C', range(si*10, si*10+10), L)
            val, var, bias = (a[0] for a in bc(z, np.ones(10)))
            wb = weights[si]
            boot = np.concatenate([bc(z, wb[i:i+128])[0] for i in range(0, CBOOT, 128)])
            boot_save[f'L{L}__s{si}'] = boot
            loo = bc(z, np.ones((10, 10))-np.eye(10))[0]
            for fi, f in enumerate(SELECT):
                scores.append({'layer': L, 'set': si, 'format': f, 'BC': float(val[fi]),
                               'variance': float(var[fi]), 'bias': float(bias[fi]), 'BC_per_window': float(val[fi]/10)})
            for a, b in PAIRS:
                ia, ib = SELECT.index(a), SELECT.index(b)
                sg = int(direction(val[ia], val[ib]))
                dirs = direction(boot[:, ia], boot[:, ib])
                gap = normalized_gap(boot[:, ia], boot[:, ib])
                agree = float(np.mean(dirs == sg)) if sg else 0.0
                r = {'layer': L, 'set': si, 'a': a, 'b': b,
                     'same_bit': any({a, b} == set(p) for p in SAME),
                     'BC_a': float(val[ia]), 'BC_b': float(val[ib]), 'direction': sg,
                     'pick': a if sg < 0 else b if sg > 0 else 'tie',
                     'normalized_gap': float(normalized_gap(val[ia], val[ib])),
                     'gap_boot_lo': float(np.quantile(gap, .025)), 'gap_boot_hi': float(np.quantile(gap, .975)),
                     'p_a_lower': float(np.mean(dirs < 0)), 'p_tie': float(np.mean(dirs == 0)),
                     'stability': agree, 'stable': bool(sg and agree >= STABLE),
                     'leave_one_changed': int(np.sum(direction(loo[:, ia], loo[:, ib]) != sg))}
                rows.append(r)
            log(f'BC freeze L{L} calibration set {si}')
    csvwrite(out / 'scores_C.csv', scores)
    csvwrite(out / 'rank_stability_C.csv', rows)
    np.savez(out / 'bootstrap_scores.npz', **boot_save)
    write(out / 'frozen_picks.json', {'time': now(), 'pairs': rows, 'scores': scores,
                                    'note': 'native original BC only, frozen before any new N; no abstention policy'})
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'),
                               'picks_sha256': sha(out / 'frozen_picks.json'),
                               'calibration_data_freeze_sha256': sha(out / 'calibration_data_freeze.json'),
                               'native_controls_sha256': sha(out / 'controls_native.json'),
                               'native_stats': {p.name: sha(p) for p in sorted((out / 'native').glob('*.npz'))}})


def truth(out, label):
    check_plan(out)
    fz = read(out / 'freeze.json')
    assert sha(out / 'frozen_picks.json') == fz['picks_sha256']
    if label == 'N':
        dz = read(out / 'data_freeze.json')
        assert sha(out / 'tokens.npy') == dz['tokens_sha256']
        assert sha(out / 'freeze.json') == dz['freeze_sha256']
        toks = np.load(out / 'tokens.npy')
    else:
        toks = np.load(out / 'calib/tokens.npy')
    env = cross.Env(out / 'work')
    torch = env.torch
    torch.cuda.reset_peak_memory_stats()
    rf = read(RANGES)
    assert env.rp.ranges == rf
    checks = {'ranges_equal': True, 'target_weights_fp32': True, 'all_other_weights_fp32': True,
              'all_base_activations_fp32': True, 'posit_lut': all(lut.verify()[0] == 0 for lut in env.rp.q.posit.values())}
    dev, tensor_excess, native_dev = 0.0, -float('inf'), 0.0
    result, timings = {}, {}
    for L in LAYERS:
        env.set_target(L)
        env.set_background({})
        checks['target_weights_fp32'] &= bool(torch.equal(env.Wq, env.rp.orig[TARGETS[L]].to(env.Wq.device)))
        checks['all_other_weights_fp32'] &= all(torch.equal(env.rp.mod(n).weight.data, env.rp.orig[n].to(env.rp.device)) for n in env.rp.param)
        checks['all_base_activations_fp32'] &= all(v == 'fp32' for v in env.rp.act.values())
        ce_arrays = {f: np.zeros((len(toks), TP), np.float64) for f in FORMATS}
        step = 10 if label == 'C' else 32
        t0 = time.perf_counter()
        for start in range(0, len(toks), step):
            t = toks[start:start+step]
            ids = torch.from_numpy(t).to(env.rp.device)
            ce0, cap = env.full(t, 'f16', capture=True)
            for f in FORMATS:
                u, h = env.suffix(cap['r'], env.zf(cap['x'], f), cap['kw'], keep_h=True)
                ce = env.head_ce(u, ids)
                ce_arrays[f][start:start+len(t)] = ce.cpu().numpy()
                if f == 'f16':
                    dev = max(dev, float((ce-ce0).abs().max()))
                if label == 'C':
                    direct, dc = env.full(t, f, capture=True)
                    dev = max(dev, float((ce-direct).abs().max()))
                    for a, b in [(u, dc['u']), (h, dc['h'])]:
                        tensor_excess = max(tensor_excess, float(((a-b).abs()-1e-6-1e-6*b.abs()).max()))
                    del dc, direct
                del ce, u, h
            del cap
        torch.cuda.synchronize()
        timings[f'L{L}_s'] = time.perf_counter()-t0
        result.update({f'L{L}__{f}': a for f, a in ce_arrays.items()})
        if label == 'C':
            ref = np.stack([np.load(out / 'native' / f'C_{i:02d}_errors6.npz')['baseline_ce'] for i in range(len(toks))])
            native_dev = max(native_dev, float(np.max(np.abs(ce_arrays['fp32']-ref))))
        checks['ranges_unchanged'] = env.rp.ranges == rf
        log(f'GPU {label} L{L} done, {timings[f"L{L}_s"]:.1f}s')
    checks['replay_ce'] = dev <= 1e-6
    if label == 'C':
        checks['replay_tensors'] = tensor_excess <= 0
        checks['native_gpu_baseline'] = native_dev <= 1e-3
    np.savez(out / f'{label}_ce_pos.npz', **result)
    write(out / f'controls_truth_{label}.json', {'checks': checks, 'passed': all(checks.values()),
          'max_replay_ce_abs': dev, 'max_tensor_excess': tensor_excess if label == 'C' else None,
          'native_gpu_baseline_ce_max_abs': native_dev if label == 'C' else None,
          'timing': timings, 'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
          'scope': 'GPU FP32+QDQ truth, not native candidate truth'})
    if not all(checks.values()):
        raise SystemExit(f'GPU controls failed: {checks}')


def analyze(out):
    check_plan(out)
    assert read(out / 'controls_truth_C.json')['passed'] and read(out / 'controls_truth_N.json')['passed']
    assert sha(out / 'frozen_picks.json') == read(out / 'freeze.json')['picks_sha256']
    frozen = read(out / 'frozen_picks.json')
    cn, nn = np.load(out / 'C_ce_pos.npz'), np.load(out / 'N_ce_pos.npz')
    rng = np.random.default_rng(SEEDS['bootstrap'])
    weights = rng.multinomial(N_ART, np.full(N_ART, 1/N_ART), size=NBOOT)/N_ART
    flips = rng.choice([-1.0, 1.0], size=(NBOOT, N_ART))
    pair_truth, truth_rows, pair_rows = {}, [], []
    for L in LAYERS:
        base = nn[f'L{L}__f16'].reshape(N_ART, WPA, TP).mean((1, 2))
        for f in FORMATS:
            d = nn[f'L{L}__{f}'].reshape(N_ART, WPA, TP).mean((1, 2)) - base
            ci = np.quantile(weights @ d, [.025, .975])
            truth_rows.append({'layer': L, 'format': f, 'D_mean': float(d.mean()), 'ci_lo': float(ci[0]), 'ci_hi': float(ci[1])})
        for a, b in PAIRS:
            d = (nn[f'L{L}__{a}']-nn[f'L{L}__{b}']).reshape(N_ART, WPA, TP).mean((1, 2))
            ci = np.quantile(weights @ d, [.025, .975])
            sg = -1 if ci[1] < 0 else 1 if ci[0] > 0 else 0
            p = float((1+np.sum(np.abs(flips@d/N_ART) >= abs(d.mean())))/(NBOOT+1))
            pair_truth[(L,a,b)] = {'N_gap': float(d.mean()), 'N_lo': float(ci[0]), 'N_hi': float(ci[1]), 'truth_N': sg, 'p_signflip': p}
    unique = [k for k in pair_truth if any(set(k[1:]) == set(p) for p in SAME)]
    ordered = sorted(unique, key=lambda k: pair_truth[k]['p_signflip'])
    prev = 0.0
    for j,k in enumerate(ordered):
        prev = max(prev, min(1.0, pair_truth[k]['p_signflip']*(len(ordered)-j)))
        pair_truth[k]['p_holm_9'] = prev
    for r in frozen['pairs']:
        L, si, a, b = r['layer'], r['set'], r['a'], r['b']
        tr = pair_truth[(L,a,b)]
        dC = float((cn[f'L{L}__{a}'][si*10:(si+1)*10]-cn[f'L{L}__{b}'][si*10:(si+1)*10]).mean())
        truthC = int(np.sign(dC)) if abs(dC)>4e-7 else 0
        outcome = 'undetermined' if not tr['truth_N'] else 'score_tie' if not r['direction'] else 'correct' if r['direction']==tr['truth_N'] else 'wrong'
        pair_rows.append({**r, **tr, 'p_holm_9': tr.get('p_holm_9', None), 'C_loss_gap': dC,
                          'truth_C': truthC, 'correct_on_C': bool(truthC and truthC==r['direction']), 'outcome_N': outcome})
    csvwrite(out / 'N_truth.csv', truth_rows)
    csvwrite(out / 'pairwise_results.csv', pair_rows)
    buckets = {}
    for scope, rr in [('same_bit', [r for r in pair_rows if r['same_bit']]), ('all_pairs', pair_rows)]:
        buckets[scope] = {'rows': len(rr), 'counts': dict(Counter(('stable' if r['stable'] else 'unstable')+'_'+r['outcome_N'] for r in rr)),
                         'stable': sum(r['stable'] for r in rr),
                         'mean_stability': float(np.mean([r['stability'] for r in rr])),
                         'min_stability': float(min(r['stability'] for r in rr))}
    consistency = []
    for L in LAYERS:
        for a,b in PAIRS:
            rr=[r for r in pair_rows if r['layer']==L and r['a']==a and r['b']==b]
            consistency.append({'layer': L, 'a': a, 'b': b, 'same_bit': rr[0]['same_bit'],
                                'directions': [r['direction'] for r in rr], 'all_three_agree': len({r['direction'] for r in rr})==1,
                                'truth_N': rr[0]['truth_N'], 'outcomes': [r['outcome_N'] for r in rr]})
    write(out / 'summary.json', {'buckets': buckets, 'across_sets': consistency,
                                'same_bit_unique_truths': [{'layer': k[0], 'a': k[1], 'b': k[2], **pair_truth[k]} for k in unique],
                                'interpretation_scope': 'original native BC; GPU QDQ truth; fresh articles; descriptive stability calibration only'})
    log(json.dumps(buckets, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['plan','audit','calibdata','native','native-child','freeze','data','truth-C','truth-N','analyze'], required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--label', choices=['C','C0'])
    p.add_argument('--index', type=int)
    p.add_argument('--mode', choices=['grads','errors6'])
    a=p.parse_args(); out=a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for n in ['work','logs']:
        (out/n).mkdir(exist_ok=True)
    t0=time.perf_counter()
    if a.stage=='native-child':
        native_child(out,a.label,a.index,a.mode); return
    configure_helpers(out)
    if a.stage=='plan': plan(out)
    elif a.stage=='audit': cov.audit(out)
    elif a.stage=='calibdata': cross.calibdata(out)
    elif a.stage=='native': native(out)
    elif a.stage=='freeze': freeze(out)
    elif a.stage=='data': cross.data(out)
    elif a.stage=='truth-C': truth(out,'C')
    elif a.stage=='truth-N': truth(out,'N')
    elif a.stage=='analyze': analyze(out)
    duration=time.perf_counter()-t0
    jf=out/'execution_times.json'
    timing=read(jf) if jf.exists() else {}
    timing[a.stage]={'seconds':duration,'time':now(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write(jf,timing)
    log(f'{a.stage} complete in {duration:.1f}s')


if __name__=='__main__':
    main()
