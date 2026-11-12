#!/usr/bin/env python3
"""Preregistered L27 Down X/Y QDQ attribution. Diagnostics only; no pipeline changes.

Stages: plan, calib, freeze, audit, data, eval, analyze. CUDA stages: calib/eval.
See reports/qwen3_l27_slot_attribution_prompt_20261024.md for the scientific protocol.
"""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
import run_qwen_prefix_rank_shift as pr

HERE = Path(__file__).resolve().parent
PREFIX = HERE / 'runs/qwen3_prefix_rank_shift_20261024'
PROTOCOL = HERE / 'reports/qwen3_l27_slot_attribution_prompt_20261024.md'
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1')
SLOTS = ('X', 'Y', 'XY')
SEEDS = {'order': 31101, 'start': 31102, 'bootstrap': 31103}
N, WPA, SEQ, TP, BOOT, CHUNK = 128, 2, 128, 127, 20000, 8
read, write, sha, log, now = pr.read, pr.write, pr.sha, pr.log, pr.now


def conditions():
    rows = []
    for w in ('W0', 'W8'):
        rows.append({'w': w, 'slot': 'base', 'format': 'fp32', 'mask': 'all'})
        for f in FORMATS:
            for s in SLOTS:
                rows.append({'w': w, 'slot': s, 'format': f, 'mask': 'all'})
    for f in FORMATS[:2]:
        for s in SLOTS:
            for m in ('P0', 'Rest'):
                rows.append({'w': 'W8', 'slot': s, 'format': f, 'mask': m})
    for r in rows:
        r['id'] = '|'.join(r[k] for k in ('w', 'slot', 'format', 'mask'))
    assert len(rows) == 38 and len({r['id'] for r in rows}) == 38
    return rows


COND = conditions()
INDEX = {r['id']: i for i, r in enumerate(COND)}


def idx(w, s, f='fp32', mask='all'):
    return INDEX['|'.join((w, s, f, mask))]


def csvwrite(p, rows):
    with open(p, 'w') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed after plan: {f}; record an amendment before proceeding')
    if inputs:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def plan(out):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan exists')
    _, down, A = pr.structure()
    assert sha(pr.FP16 / 'configs/A.json') == pr.A_SHA
    assert A[down[27]] == {'w': 'int8', 'a': 'f16'}
    w103, wt2 = wms.corpora()
    used, manifests = wms.used_wt103_verified(out, w103, wt2)
    prior = {d.name: {s['content_sha256'] for s in read(d / 'segments.json')}
             for d in (PREFIX, pr.FP16, pr.L27R, pr.SPIKE)}
    assert all(h <= used for h in prior.values())
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1
            and cc[a['ch']] == 1 and a['ch'] not in used]
    codes = sorted(set(pr.CODE + [Path(__file__).name, str(PROTOCOL.relative_to(HERE))]))
    inputs = pr.input_files() + [PREFIX / f for f in ('segments.json', 'calibration_scores.json', 'controls.json')]
    pl = {'time': now(), 'protocol_sha256': sha(PROTOCOL), 'target': down[27],
          'scope': 'Qwen3-0.6B last Down only; other graph FP32. W0 original weight, W8 original per-channel INT8. '
                   'X=input-only, Y=output-only, XY=actual serial QDQ. GPU float32 + QDQ, not native improvement.',
          'conditions': COND, 'seeds': SEEDS,
          'data': {'articles': N, 'windows_per_article': WPA, 'tokens_per_window': SEQ,
                   'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig),
                   'prior_checks': {k: len(h) for k, h in prior.items()},
                   'sampler': 'unchanged prefix-rank sampler; new seeds; article/word-fivegram historical audit; WT2 window >=50% rejected'},
          'statistics': {'bootstrap': BOOT, 'unit': 'article mean of two windows, each 127 CE targets',
                         'primary': ['W8 D_X int4-p4', 'W8 D_Y int4-p4', 'W8 D_XY int4-p4', 'W8 J int4-p4'],
                         'CI': '95% and primary-family Bonferroni 98.75%; identical paired bootstrap indices',
                         'secondary': 'W0, six-bit, W8-W0, positional masks; all 95%; six positional D tests Holm',
                         'epsilon_nat_per_token': 0.001},
          'tolerances': {'full_vs_suffix_summed_ce': 1e-6, 'native_BC_relative': .01,
                         'mask_summed_ce_identity': 1e-6, 'mask_proxy_relative_identity': 1e-10,
                         'chain_rule_relative': 1e-6},
          'code_hashes': {f: sha(HERE / f) for f in codes},
          'inputs': {pr.rel(f): sha(f) for f in inputs},
          'historical_manifests': manifests,
          'known_before_plan': 'Old input proxy prefers int4; output proxy prefers posit4; total prefers int4. '
                               'Previous U combined measured int4-p4 = +.387073. Old input-only FP32 weight measured Posit4 better. '
                               'No new article losses examined. Scope/conditions/seeds fixed by written protocol.'}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text(PROTOCOL.read_text() + '\n\n## This execution\n\n'
                                + f'Frozen {pl["time"]}; script hashes in plan.json. 38 conditions; seeds {SEEDS}.\n')
    write(out / 'conditions.json', COND)
    pr.log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan frozen: {len(used)} prior WT103 articles excluded, {len(elig)} eligible')


def replay(out):
    rp, init = pr.make_replay(out / 'work')
    _, down, _ = pr.structure()
    rp.target = down[27]
    rp.set({})
    rp.ad.requires_grad_(False)
    rp.target_weights = {'W0': rp.wdev(rp.target, 'fp32'), 'W8': rp.wdev(rp.target, 'int8')}
    assert rp.mod(rp.target).bias is None
    return rp, init


def qmask(rp, t, fmt, slot, mask):
    qt = rp.qact(t, fmt, rp.ranges[rp.target][slot])
    if mask == 'all':
        return qt
    selected = rp.torch.arange(t.shape[1], device=t.device) == 0
    if mask == 'Rest':
        selected = ~selected
    return rp.torch.where(selected[None, :, None], qt, t)


def capture(rp, ids):
    cap = {}
    hooks = [rp.layers[27].post_attention_layernorm.register_forward_pre_hook(
                 lambda m, a: cap.__setitem__('residual', a[0].detach())),
             rp.mod(rp.target).register_forward_pre_hook(lambda m, a: cap.__setitem__('x', a[0].detach()))]
    try:
        with rp.torch.no_grad():
            h = rp.ad.model.model(input_ids=ids, attention_mask=rp.ad.mask, use_cache=False).last_hidden_state
            ce = pr.head_ce(rp, h, ids)
    finally:
        for h in hooks:
            h.remove()
    return cap['x'], cap['residual'], ce


def suffix(rp, x, residual, ids, c):
    if 'X' in c['slot']:
        x = qmask(rp, x, c['format'], 'x', c['mask'])
    y = rp.torch.nn.functional.linear(x, rp.target_weights[c['w']])
    if 'Y' in c['slot']:
        y = qmask(rp, y, c['format'], 'y', c['mask'])
    return pr.head_ce(rp, rp.ad.model.model.norm(residual + y), ids)


def complete(rp, ids, c):
    """Independent complete model route; LMReplay hooks remain inactive, only specified target hooks apply."""
    rp.set({rp.target: {'w': 'int8' if c['w'] == 'W8' else 'fp32', 'a': 'fp32'}})
    hooks = []
    if 'X' in c['slot']:
        hooks.append(rp.mod(rp.target).register_forward_pre_hook(
            lambda m, a: (qmask(rp, a[0], c['format'], 'x', c['mask']),)))
    if 'Y' in c['slot']:
        hooks.append(rp.mod(rp.target).register_forward_hook(
            lambda m, a, o: qmask(rp, o, c['format'], 'y', c['mask'])))
    try:
        return pr.forward_capture(rp, ids, capture=False)[0]
    finally:
        for h in hooks:
            h.remove()


def mask_identity(ce):
    maximum = 0.0
    for f in FORMATS[:2]:
        for s in SLOTS:
            r = ce[idx('W8', s, f)] - ce[idx('W8', s, f, 'P0')] - ce[idx('W8', s, f, 'Rest')] + ce[idx('W8', 'base')]
            maximum = max(maximum, float(np.abs(r).max()))
    return maximum


def bc_scores(rp, out):
    torch = rp.torch
    rp.set({})
    toks = np.load(pr.NATIVE / 'calib_inputs.npy')
    vals = {k: [] for k in ('x', 'y', 'gx', 'gy')}
    chain, last = 0., True
    t0 = time.perf_counter()
    for row in toks:
        ids = torch.from_numpy(row[None]).to(rp.device)
        x0, residual, _ = capture(rp, ids)
        x = x0.detach().requires_grad_(True)
        y = torch.nn.functional.linear(x, rp.target_weights['W0'])
        loss = pr.head_ce(rp, rp.ad.model.model.norm(residual + y), ids).sum()
        gx, gy = torch.autograd.grad(loss, (x, y))
        chain = max(chain, float((gx - gy @ rp.target_weights['W0']).abs().max() / gx.abs().max()))
        last &= bool((gx[:, -1] == 0).all() and (gy[:, -1] == 0).all())
        for k, t in (('x', x), ('y', y), ('gx', gx), ('gy', gy)):
            vals[k].append(t.detach().cpu().numpy())
        del x, y, gx, gy, loss
    vals = {k: np.concatenate(v) for k, v in vals.items()}
    np.savez_compressed(out / 'calibration_tensors.npz', **vals)
    backward_s = time.perf_counter() - t0
    rows, totals, algebra = [], {}, 0.
    for slot in ('x', 'y'):
        t = vals[slot]
        g = vals['g' + slot].astype(np.float64)
        G2 = (g ** 2).sum((0, 1))
        GS = (g.sum(1) ** 2).sum(0)
        totals[slot] = {}
        for f in FORMATS:
            with torch.no_grad():
                tt = torch.from_numpy(t).to(rp.device)
                e = (rp.qact(tt, f, rp.ranges[rp.target][slot]) - tt).cpu().numpy().astype(np.float64)
            bymask, mu = {}, {}
            for mask in ('all', 'P0', 'Rest'):
                em = e.copy()
                if mask == 'P0':
                    em[:, 1:] = 0
                elif mask == 'Rest':
                    em[:, :1] = 0
                mu[mask] = em.mean((0, 1))
                var = (em ** 2).mean((0, 1)) - mu[mask] ** 2
                V, B = G2 * var, GS * mu[mask] ** 2
                bymask[mask] = {'variance': float(V.sum()), 'activation_bias': float(B.sum()), 'total': float((V + B).sum()),
                                'error_square_sum': float((em ** 2).sum())}
                for c in range(t.shape[-1]):
                    rows.append({'slot': slot, 'format': f, 'mask': mask, 'channel': c,
                                 'G2': G2[c], 'GS': GS[c], 'mean_error': mu[mask][c],
                                 'variance': V[c], 'activation_bias': B[c], 'total': V[c] + B[c]})
            cross = float((2 * (GS - G2) * mu['P0'] * mu['Rest']).sum())
            residual = bymask['all']['total'] - bymask['P0']['total'] - bymask['Rest']['total'] - cross
            algebra = max(algebra, abs(residual) / max(abs(bymask['all']['total']), 1))
            totals[slot][f] = {**bymask, 'cross': cross, 'identity_residual': residual}
    csvwrite(out / 'bc_channels.csv', rows)
    table = read(pr.NATIVE / 'table.json')['C'][rp.target]
    native = {f: {'native': table[f], 'reproduced': totals['x'][f]['all']['total'] + totals['y'][f]['all']['total']} for f in FORMATS}
    for f, r in native.items():
        r['relative_error'] = abs(r['reproduced'] - r['native']) / abs(r['native'])
    ordering = sorted(FORMATS, key=lambda f: native[f]['native']) == sorted(FORMATS, key=lambda f: native[f]['reproduced'])
    write(out / 'bc_scores.json', {'scores': totals, 'native_comparison': native, 'native_order_matches': ordering,
                                  'gradient_chain_max_relative': chain, 'gradient_last_zero': last,
                                  'mask_identity_max_relative': algebra, 'backward_seconds': backward_s})
    return native, ordering, chain, last, algebra, backward_s


def calib(out):
    pl = check_plan(out)
    t0 = time.perf_counter()
    rp, init = replay(out)
    torch = rp.torch
    ids = torch.from_numpy(np.load(pr.NATIVE / 'calib_inputs.npy')).to(rp.device)
    x, r, base = capture(rp, ids)
    ce, cf = [], []
    for c in COND:
        with torch.no_grad():
            ce.append(suffix(rp, x, r, ids, c).cpu().numpy())
            cf.append(complete(rp, ids, c).cpu().numpy())
    ce, cf = np.array(ce), np.array(cf)
    np.savez_compressed(out / 'calibration_losses.npz', suffix=ce, complete=cf)
    old = np.array(read(PREFIX / 'calibration_scores.json')['per_window_summed_ce']['U'])[27]
    olderr = {f: float(np.max(np.abs(ce[idx('W8', 'XY', f)] - old[pr.SETTINGS.index(f)]))) for f in FORMATS}
    # Independent existing input-only hook semantics, with the old helper's quantizer.
    from run_qwen_down_spike_diag import Quant
    oldq = Quant(out / 'work', 'cuda')
    quantizer_diff = {}
    with torch.no_grad():
        for f in FORMATS:
            a = oldq.act(x, f, rp.ranges[rp.target]['x'])
            b = rp.qact(x, f, rp.ranges[rp.target]['x'])
            quantizer_diff[f] = float((a - b).abs().max())
    native, order, chain, last, algebra, backward_s = bc_scores(rp, out)
    key = f'{rp.target}|int8'
    whash = hashlib.sha256(rp.target_weights['W8'].cpu().numpy().tobytes()).hexdigest()
    oldwhash = read(PREFIX / 'controls.json')['tensor_hashes'][key]
    tol = pl['tolerances']
    ctl = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'dtype': str(x.dtype), 'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
           'deterministic': torch.are_deterministic_algorithms_enabled(), 'conditions': len(COND),
           'full_vs_suffix_max_summed_ce': float(np.abs(ce - cf).max()),
           'full_vs_suffix_exact_fraction': float((ce == cf).mean()),
           'W0_base_identity': bool(np.array_equal(base.cpu().numpy(), ce[idx('W0', 'base')])),
           'W8_XY_vs_previous_max_summed_ce': olderr, 'W0_X_old_quantizer_max_abs': quantizer_diff,
           'W8_weight_sha256': whash, 'W8_weight_matches_previous': whash.startswith(oldwhash),
           'bias_is_none': rp.mod(rp.target).bias is None,
           'posit_verified': {f: list(q.verify()) for f, q in rp.q.posit.items()},
           'native_BC': native, 'native_BC_order_matches': order,
           'gradient_chain_max_relative': chain, 'gradient_last_zero': last,
           'BC_mask_identity_max_relative': algebra, 'NLL_mask_identity_max_summed_ce': mask_identity(ce),
           'nonfinite': int((~np.isfinite(ce)).sum() + (~np.isfinite(cf)).sum()),
           'hooks_final': dict(Counter(f'{len(rp.mod(n)._forward_pre_hooks)},{len(rp.mod(n)._forward_hooks)}' for n in rp.param))}
    ctl['passed'] = bool(ctl['full_vs_suffix_max_summed_ce'] <= tol['full_vs_suffix_summed_ce']
                         and ctl['W0_base_identity'] and max(olderr.values()) <= tol['full_vs_suffix_summed_ce']
                         and max(quantizer_diff.values()) == 0 and ctl['W8_weight_matches_previous']
                         and ctl['tf32'] == [False, False] and ctl['deterministic']
                         and all(v[0] == 0 for v in ctl['posit_verified'].values()) and order
                         and max(v['relative_error'] for v in native.values()) <= tol['native_BC_relative']
                         and chain <= tol['chain_rule_relative'] and last and algebra <= tol['mask_proxy_relative_identity']
                         and ctl['NLL_mask_identity_max_summed_ce'] <= tol['mask_summed_ce_identity']
                         and ctl['nonfinite'] == 0 and ctl['hooks_final'] == {'1,1': 196})
    write(out / 'controls.json', ctl)
    pr.log_stage(out, 'calib', time.perf_counter() - t0, {'model_init_s': init, 'backward_s': backward_s,
                                                        'peak_gpu_bytes': int(torch.cuda.max_memory_allocated())})
    log(f'calibration controls passed={ctl["passed"]}; full/replay max={ctl["full_vs_suffix_max_summed_ce"]}; '
        f'native BC max rel={max(v["relative_error"] for v in native.values()):.6f}')
    if not ctl['passed']:
        raise SystemExit('controls failed; do not evaluate fresh data')


def freeze(out):
    check_plan(out)
    assert read(out / 'controls.json')['passed']
    if (out / 'freeze.json').exists():
        raise SystemExit('already frozen')
    files = ['plan.json', 'conditions.json', 'controls.json', 'bc_scores.json', 'bc_channels.csv',
             'calibration_losses.npz', 'calibration_tensors.npz']
    write(out / 'freeze.json', {'time': now(), 'files': {f: sha(out / f) for f in files}})


def verify(out):
    check_plan(out)
    for f, h in read(out / 'freeze.json')['files'].items():
        assert sha(out / f) == h, f
    df = read(out / 'data_freeze.json')
    for f, k in (('freeze.json', 'freeze_sha256'), ('tokens.npy', 'tokens_sha256'),
                 ('segments.json', 'segments_sha256'), ('audit.json', 'audit_sha256'), ('exclusions.json', 'exclusions_sha256')):
        assert sha(out / f) == df[k], f
    return df


def inherited(out, stage):
    pr.check_plan = check_plan
    pr.SEEDS = SEEDS
    pr.N_ART, pr.WPA = N, WPA
    getattr(pr, stage)(out)
    if stage == 'data':
        new = {s['content_sha256'] for s in read(out / 'segments.json')}
        old = {s['content_sha256'] for s in read(PREFIX / 'segments.json')}
        assert not new & old and len(new) == N


def evaluate(out):
    df = verify(out)
    t0 = time.perf_counter()
    toks = np.load(out / 'tokens.npy')
    assert toks.shape == (N * WPA, SEQ)
    rp, init = replay(out)
    torch = rp.torch
    ckdir = out / 'checkpoints'
    ckdir.mkdir(exist_ok=True)
    identity = {'data': sha(out / 'data_freeze.json'), 'freeze': sha(out / 'freeze.json'),
                'script': sha(Path(__file__)), 'conditions': sha(out / 'conditions.json')}
    chunks, times = [], []
    for st in range(0, len(toks), CHUNK):
        npz, meta = ckdir / f'{st:04d}.npz', ckdir / f'{st:04d}.json'
        if npz.exists() and meta.exists():
            m = read(meta)
            assert m['identity'] == identity and sha(npz) == m['sha256']
            ce = np.load(npz)['summed_ce']
            times.append(m['seconds'])
        else:
            t1 = time.perf_counter()
            ids = torch.from_numpy(toks[st:st + CHUNK]).to(rp.device)
            x, residual, base = capture(rp, ids)
            with torch.no_grad():
                ce = np.array([suffix(rp, x, residual, ids, c).cpu().numpy() for c in COND])
            assert np.array_equal(ce[idx('W0', 'base')], base.cpu().numpy())
            if not np.isfinite(ce).all():
                np.savez_compressed(ckdir / f'{st:04d}_FAILED.npz', summed_ce=ce)
                raise SystemExit('nonfinite matrix; preserved failed chunk')
            assert mask_identity(ce) <= 1e-6
            np.savez_compressed(npz, summed_ce=ce)
            sec = time.perf_counter() - t1
            write(meta, {'identity': identity, 'start': st, 'stop': st + len(ids), 'sha256': sha(npz), 'seconds': sec})
            times.append(sec)
        chunks.append(ce)
        log(f'eval {st + ce.shape[1]}/{len(toks)} windows; {sum(times):.1f}s computed')
    ce = np.concatenate(chunks, axis=1)
    np.savez_compressed(out / 'losses.npz', summed_ce=ce, article_nll=ce.reshape(38, N, WPA).mean(2) / TP)
    write(out / 'eval_controls.json', {'finite': bool(np.isfinite(ce).all()), 'conditions': 38, 'windows': len(toks),
                                       'mask_identity_max_summed_ce': mask_identity(ce), 'identity': identity,
                                       'losses_sha256': sha(out / 'losses.npz')})
    pr.log_stage(out, 'eval', time.perf_counter() - t0, {'model_init_s': init, 'chunks_computed_s': sum(times),
                                                       'peak_gpu_bytes': int(torch.cuda.max_memory_allocated())})


def analyze(out):
    verify(out)
    t0 = time.perf_counter()
    assert sha(out / 'losses.npz') == read(out / 'eval_controls.json')['losses_sha256']
    z = np.load(out / 'losses.npz')
    E, CE = z['article_nll'], z['summed_ce']
    ix = np.random.default_rng(SEEDS['bootstrap']).integers(0, N, (BOOT, N))

    def stat(d, primary=False):
        b = d[ix].mean(1)
        mean = float(d.mean())
        s = {'mean': mean, 'ci95': np.percentile(b, [2.5, 97.5]).tolist(),
             'p_centered_bootstrap': float((1 + (np.abs(b - mean) >= abs(mean)).sum()) / (BOOT + 1)),
             'positive_articles': int((d > 0).sum()), 'negative_articles': int((d < 0).sum())}
        if primary:
            s['ci98.75_bonferroni'] = np.percentile(b, [.625, 99.375]).tolist()
        return s

    damage, D, I, J, pair = {}, {}, {}, {}, {}
    for w in ('W0', 'W8'):
        for f in FORMATS:
            for s in SLOTS:
                d = E[idx(w, s, f)] - E[idx(w, 'base')]
                damage[f'{w}|{s}|{f}'] = stat(d)
            I[w, f] = E[idx(w, 'XY', f)] - E[idx(w, 'X', f)] - E[idx(w, 'Y', f)] + E[idx(w, 'base')]
        for bits, fi, fp in ((4, 'int4', 'p4e1'), (6, 'int6', 'p6e1')):
            for s in SLOTS:
                D[w, s, bits] = E[idx(w, s, fi)] - E[idx(w, s, fp)]
                pair[f'{w}|{s}|{bits}'] = stat(D[w, s, bits], w == 'W8' and bits == 4)
            J[w, bits] = I[w, fi] - I[w, fp]
            assert np.max(np.abs(D[w, 'XY', bits] - D[w, 'X', bits] - D[w, 'Y', bits] - J[w, bits])) < 1e-12
    masks = {}
    for s in SLOTS:
        for m in ('P0', 'Rest'):
            d = E[idx('W8', s, 'int4', m)] - E[idx('W8', s, 'p4e1', m)]
            masks[f'{s}|{m}'] = stat(d)
    order = sorted(masks, key=lambda k: masks[k]['p_centered_bootstrap'])
    running = 0.
    for j, k in enumerate(order):
        running = max(running, min(1., (len(order) - j) * masks[k]['p_centered_bootstrap']))
        masks[k]['p_holm_6'] = running
    res = {'unit': 'nat/token; positive INT-minus-Posit means Posit is better', 'articles': N, 'windows': N * WPA,
           'baseline_nll': {w: float(E[idx(w, 'base')].mean()) for w in ('W0', 'W8')},
           'damage': damage, 'pairs': pair,
           'interactions': {f'{w}|{f}': stat(v) for (w, f), v in I.items()},
           'interaction_ranking_J': {f'{w}|{b}': stat(v, w == 'W8' and b == 4) for (w, b), v in J.items()},
           'weight_background_difference': {f'{s}|{b}': stat(D['W8', s, b] - D['W0', s, b]) for s in SLOTS for b in (4, 6)},
           'masks': masks, 'mask_identity_max_summed_ce': mask_identity(CE)}
    write(out / 'stats.json', res)
    seg = read(out / 'segments.json')
    csvwrite(out / 'article_losses.csv', [{'condition': c['id'], 'article': k, 'title': seg[k]['title'], 'nll': E[i, k],
                                          'delta_nll_same_weight': E[i, k] - E[idx(c['w'], 'base'), k]}
                                         for i, c in enumerate(COND) for k in range(N)])
    csvwrite(out / 'window_losses.csv', [{'condition': c['id'], 'window': k, 'article': k // WPA, 'summed_ce': CE[i, k]}
                                        for i, c in enumerate(COND) for k in range(N * WPA)])
    csvwrite(out / 'interaction.csv', [{'w': w, 'bits': b, 'article': k, 'D_X': D[w, 'X', b][k],
                                       'D_Y': D[w, 'Y', b][k], 'D_XY': D[w, 'XY', b][k], 'J': J[w, b][k]}
                                      for w in ('W0', 'W8') for b in (4, 6) for k in range(N)])
    bc = read(out / 'bc_scores.json')['scores']
    fmt = lambda v: f'{v:.6f}'
    ci = lambda a: '[' + ', '.join(fmt(v) for v in a) + ']'
    md = ['# Qwen3-0.6B L27 Down input/output 誤排定位', '',
          '128 篇全新 WT103 × 2 窗口；38 條件。GPU FP32 + QDQ 局部診斷；沒有修改正式 criterion 或 pipeline。', '',
          'INT−Posit 為正：Posit4 損傷較低。BC 以原 FP32 calibration 計算；W8=target INT8、W0=target FP32，其他運算 FP32。', '',
          '| 介入 | 原 BC INT−Posit（偏好） | W8 實測 INT4−Posit4 | 98.75% CI（主要 family） | W0 實測 | 95% CI |',
          '|---|---:|---:|---|---:|---|']
    for s in SLOTS:
        cb = sum(bc[slot]['int4']['all']['total'] - bc[slot]['p4e1']['all']['total'] for slot in ('x', 'y') if slot.upper() in s)
        a, b = pair[f'W8|{s}|4'], pair[f'W0|{s}|4']
        md.append(f'| {s} | {cb:.3f}（{"INT4" if cb < 0 else "Posit4"}） | {fmt(a["mean"])} | {ci(a["ci98.75_bonferroni"])} | {fmt(b["mean"])} | {ci(b["ci95"])} |')
    md += ['', '## 串聯介入交互作用', '', 'I_f = d_XY,f − d_X,f − d_Y,f；J=I_INT−I_Posit；D_XY=D_X+D_Y+J。', '',
           '| 背景 | 位元 | J | CI |', '|---|---:|---:|---|']
    for k, v in res['interaction_ranking_J'].items():
        md.append(f'| {k.split("|")[0]} | {k.split("|")[1]} | {fmt(v["mean"])} | {ci(v.get("ci98.75_bonferroni", v["ci95"]))} |')
    md += ['', '## 位置介入（W8，INT4−Posit4）', '', '| 介入 | mask | 差距 | 95% CI | Holm p（6 tests） |', '|---|---|---:|---|---:|']
    for k, v in masks.items():
        md.append(f'| {k.split("|")[0]} | {k.split("|")[1]} | {fmt(v["mean"])} | {ci(v["ci95"])} | {v["p_holm_6"]:.6g} |')
    md += ['', f'逐窗口位置加總殘差最大 {res["mask_identity_max_summed_ce"]:.3g} summed CE。此位置加總不同於 X/Y 串聯交互作用。', '',
           '## 邊界與資料', '',
           '- 本輪可定位 slot／位置失配，不能單憑這些介入區分 g² 曲率、一階項或大擾動非線性。',
           '- INT4／Posit4 局部問題不能直接解釋原 knee 的 FP16→Posit8 現象。',
           '- 完整次要分析（六位元、權重背景差、每格式損傷）在 stats.json；逐文章與逐窗口值分別在 article_losses.csv、window_losses.csv。',
           '- 計畫、程式／資料 hashes、排除紀錄、校準控制與執行時間皆保留於本目錄。', '']
    (out / 'report.md').write_text('\n'.join(md))
    pr.log_stage(out, 'analyze', time.perf_counter() - t0)
    log('analysis and numerical report complete')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--stage', choices=('plan', 'calib', 'freeze', 'audit', 'data', 'eval', 'analyze'), required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'work').mkdir(exist_ok=True)
    with open(out / 'run.log', 'a') as lf:
        pr.LOGF = lf
        log(f'start stage {args.stage}')
        if args.stage in ('audit', 'data'):
            inherited(out, args.stage)
        else:
            {'plan': plan, 'calib': calib, 'freeze': freeze, 'eval': evaluate, 'analyze': analyze}[args.stage](out)


if __name__ == '__main__':
    main()
