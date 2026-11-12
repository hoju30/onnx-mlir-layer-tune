#!/usr/bin/env python3
"""Follow-up of runs/cnn_baselines_20261029 (ResNet18, A8 fixed): is the auto-shell advantage of HAWQ / HTQ only more
resources? BC's own MILP pick at the exact R of each of their auto-shell picks (caps size and BOPs <= R x all-INT8) is
evaluated on the same held-out images and compared with that pick (exact McNemar, Holm over the 2 comparisons, paired
bootstrap CI). Post hoc relative to the main run (its results were seen), pre-registered before these 2 evaluations.

Stages: plan, evaluate, analyze (../../venv/bin/python, CPU). Nothing of the main run is modified; outputs go to
<run>/followup_matched/.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cnn_baselines as rc  # noqa: E402

RUN = rc.RUN
PAIRS = ('HAWQ', 'HTQ')                      # auto-shell picks whose R sets the caps for BC


def out_dir(run):
    d = run / 'followup_matched'
    d.mkdir(parents=True, exist_ok=True)
    return d


def cfg_id(fmt):
    return hashlib.sha256(json.dumps(fmt, sort_keys=True).encode()).hexdigest()[:16]


def plan(run):
    from mp_milp import milp_select
    d = out_dir(run)
    if (d / 'plan.json').exists():
        raise SystemExit('plan exists')
    cf = rc.read(run / 'configs.json')
    nodes, names, elems, macs = rc.node_list()
    ref_b, ref_o = sum(elems.values()), sum(64 * m for m in macs.values())
    v = rc.value_fn(rc.read(run / 'scores/BC_rep1.json')['cost'])
    comps = []
    for m in PAIRS:
        b = next(r for r in cf['rows'] if r['group'] == 'B' and r['method'] == m)
        R = b['resources']['R']
        caps = {'R': R, 'max_weight_bytes': R * ref_b, 'max_bops': R * ref_o}
        fmt, obj, st = milp_select(nodes, rc.WFMT, rc.AFMT, v, elems, macs, caps['max_weight_bytes'], caps['max_bops'],
                                   time_limit=rc.MILP_TIME_LIMIT)
        fmt = {n: fmt[n] for n in names}
        res = rc.resources(rc.full_cfg(fmt, nodes), names, elems, macs)
        valid = res['weight_bytes'] <= caps['max_weight_bytes'] * (1 + 1e-12) and res['bops'] <= caps['max_bops'] * (1 + 1e-12)
        if not valid or st != 0:
            raise SystemExit(f'BC at R of {m}: status {st}, valid {valid}')
        comps.append({'name': f'{m}_auto_vs_BC_at_same_R', 'method': m, 'method_config_id': b['config_id'],
                      'method_resources': b['resources'], 'caps': caps, 'bc_formats': fmt, 'bc_config_id': cfg_id(fmt),
                      'bc_objective': obj, 'bc_resources': res, 'bc_config_already_evaluated': cfg_id(fmt) in cf['unique']})
    out = {'time': rc.now(), 'question': 'auto shell: are the higher top-1 of the HAWQ and HTQ picks (R 0.672 / 0.673 vs BC '
           '0.625) only more resources? BC re-searched at the exact R of each pick',
           'status': 'post hoc relative to the main run (all its results seen); pre-registered before these evaluations',
           'comparisons': comps,
           'statistics': 'per comparison: method pick minus BC at the same caps on the same 3572 images; top-1 (primary), '
                         'exact two-sided McNemar, Holm over the 2 comparisons at alpha 0.05; paired bootstrap 95% CI '
                         f'({rc.BOOT} resamples, seed {rc.BOOT_SEED}) for top-1 and CE, descriptive; no equivalence margin '
                         '(non-significant = undetermined)',
           'reading': 'method better -> its auto-shell advantage is not only resources; BC better or undetermined with the '
                      'method not better -> no evidence beyond resources',
           'evaluation': 'same native compile / chunked evaluation as the main evaluate stage (O3/O3, per-channel weights, '
                         'fixed A8 ranges, 12 processes); the main run configs.json is not modified',
           'code_sha256': {f: rc.sha(HERE / f) for f in ('run_cnn_baselines_matched.py', 'run_cnn_baselines.py', 'mp_milp.py')},
           'main_run_hashes': {'configs.json': rc.sha(run / 'configs.json'), 'scores/BC_rep1.json': rc.sha(run / 'scores/BC_rep1.json'),
                               'summary.json': rc.sha(run / 'summary.json')}}
    rc.write(d / 'plan.json', out)
    for c in comps:
        print(c['name'], 'R %.4f' % c['caps']['R'], 'BC config', c['bc_config_id'], 'R %.4f' % c['bc_resources']['R'],
              'evaluated before' if c['bc_config_already_evaluated'] else 'new')


def check(run):
    p = rc.read(out_dir(run) / 'plan.json')
    for f, h in p['code_sha256'].items():
        if rc.sha(HERE / f) != h:
            raise SystemExit(f'code changed since the follow-up plan: {f}')
    return p


def evaluate(run, chunks=12):
    import run_native_bc_pipeline as nbp
    from evaluate_native_search_chunked import chunked_run_so
    erm = nbp.erm
    p = check(run)
    d0 = out_dir(run)
    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    work = run / 'work/eval'                  # imported model of the main evaluate stage (unchanged ONNX)
    nodes, names, elems, macs = rc.node_list()
    ranges = rc.tensor_ranges(nodes)
    labels = np.load(run / 'assets/eval_labels.npy')
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    for c in p['comparisons']:
        cid = c['bc_config_id']
        if c['bc_config_already_evaluated'] or (d0 / f'eval/{cid}/result.json').exists():
            continue
        d = d0 / f'eval/{cid}'
        d.mkdir(parents=True, exist_ok=True)
        env = erm.node_formats_env(rc.full_cfg(c['bc_formats'], nodes), nodes, ranges)
        t0 = time.perf_counter()
        so, info = erm.build_so(tools, d, cid, nbp.imported_model(work), env, None, opt='O3', krnl_opt='O3',
                                constants_to_file=True, bytecode=True)
        t_comp = time.perf_counter() - t0
        if info['warnings']:
            raise SystemExit(f'{cid}: compile warnings {info["warnings"]}')
        for q in d.iterdir():
            if q.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                q.unlink()
        z, ev = chunked_run_so(chunks)(tools, so, run / 'assets/eval_inputs.npy', d / 'logits.npy', 36000)
        if not np.isfinite(z).all():
            raise SystemExit(f'{cid}: non-finite logits')
        loss = nbp.cross_entropy(z, labels)
        top5 = np.argsort(-z, 1)[:, :5]
        np.savez_compressed(d / 'per_image.npz', loss=loss, pred=z.argmax(1), top5=top5,
                            correct1=(z.argmax(1) == labels), correct5=(top5 == labels[:, None]).any(1))
        np.save(d / 'logits.npy', z.astype(np.float32))
        rc.write(d / 'result.json', {'config_id': cid, 'env': env, 'top1': float((z.argmax(1) == labels).mean()),
                                     'top5': float((top5 == labels[:, None]).any(1).mean()), 'loss': float(loss.mean()),
                                     'n': int(len(labels)), 'compile_s': t_comp, 'eval': ev})
        print(f'{rc.now()} eval {cid}: top1 {float((z.argmax(1) == labels).mean()):.4f} (compile {t_comp:.0f}s, '
              f'eval {ev["wall_seconds"]:.0f}s)', flush=True)


def analyze(run):
    from scipy.stats import binomtest
    p = check(run)
    d0 = out_dir(run)

    def load(cid):
        base = run / f'eval/{cid}' if (run / f'eval/{cid}/result.json').exists() else d0 / f'eval/{cid}'
        return np.load(base / 'per_image.npz'), rc.read(base / 'result.json')
    rows = []
    for c in p['comparisons']:
        (a, ea), (b, eb) = load(c['method_config_id']), load(c['bc_config_id'])
        d1 = a['correct1'].astype(float) - b['correct1'].astype(float)
        dl = a['loss'] - b['loss']
        mw, bw = int((a['correct1'] & ~b['correct1']).sum()), int((~a['correct1'] & b['correct1']).sum())
        rows.append({'name': c['name'], 'R_cap': c['caps']['R'], 'method': c['method'], 'method_config_id': c['method_config_id'],
                     'bc_config_id': c['bc_config_id'], 'method_top1': ea['top1'], 'bc_top1': eb['top1'],
                     'method_top5': ea['top5'], 'bc_top5': eb['top5'], 'method_loss': ea['loss'], 'bc_loss': eb['loss'],
                     'method_resources': {k: c['method_resources'][k] for k in ('weight_MB', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')},
                     'bc_resources': {k: c['bc_resources'][k] for k in ('weight_MB', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')},
                     'bc_w_hist': c['bc_resources']['w_hist'],
                     'd_top1_pp': 100 * d1.mean(), 'd_top1_ci_pp': [100 * x for x in rc.boot_ci(d1)],
                     'd_loss': float(dl.mean()), 'd_loss_ci': rc.boot_ci(dl),
                     'mcnemar_method_right_BC_wrong': mw, 'mcnemar_method_wrong_BC_right': bw,
                     'mcnemar_p': binomtest(mw, mw + bw, 0.5).pvalue if mw + bw else 1.0})
    adj = rc.holm(np.array([r['mcnemar_p'] for r in rows]))
    for r, h in zip(rows, adj):
        r['holm_p'] = float(h)
        r['verdict'] = ('method better' if r['d_top1_pp'] > 0 else 'BC better') if h < 0.05 else 'undetermined'
    rc.write(d0 / 'summary.json', {'time': rc.now(), 'rows': rows, 'holm_family': len(rows)})
    for r in rows:
        print(r['name'], f"method {100 * r['method_top1']:.2f} BC {100 * r['bc_top1']:.2f} d {r['d_top1_pp']:+.2f} "
              f"[{r['d_top1_ci_pp'][0]:+.2f}, {r['d_top1_ci_pp'][1]:+.2f}] McNemar {r['mcnemar_method_right_BC_wrong']}/"
              f"{r['mcnemar_method_wrong_BC_right']} Holm {r['holm_p']:.3g} {r['verdict']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'evaluate', 'analyze'))
    ap.add_argument('--run-dir', type=Path, default=RUN)
    a = ap.parse_args()
    {'plan': plan, 'evaluate': evaluate, 'analyze': analyze}[a.stage](a.run_dir.resolve())


if __name__ == '__main__':
    main()
