#!/usr/bin/env python3
"""Audit resource redundancy and strict projection loss on cached native BC tables.

This compares final candidate sets, not HTQ's heuristic per-layer pruning.
Independent budget probes are local to the existing selected configuration.
No held-out accuracy or new model execution enters this analysis.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/native_projection_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from run_native_bc_pipeline import HERE, CNNS, GRID11, erm, onnx
from sample_configs_htq import compute_weight_elems_and_flops_proxy
from htq_pareto_search import FORMAT_BITS
from mp_milp import milp_select


def metadata(tag):
    if tag in CNNS:
        path = HERE / '../imagenet100/model_dataset' / CNNS[tag] / 'model.onnx'
    else:
        path = HERE / 'runs' / ('gpt2_shared_ablation_20260930' if tag == 'gpt2'
                               else 'bert_sst2_20261006') / 'model.onnx'
    graph = onnx.load(str(path))
    nodes = [n for n in erm.supported_nodes(graph) if n.get('weight_const')]
    if tag == 'gpt2':
        mapping = json.loads((path.parent / 'node_mapping.json').read_text())
        nodes = [n for n in nodes if n['name'] in mapping]
    if tag in CNNS:
        elems, flops = compute_weight_elems_and_flops_proxy(graph, nodes)
    else:
        inits = {i.name: i for i in graph.graph.initializer}
        by_name = {n.name: n for n in graph.graph.node}
        elems = {n['name']: int(np.prod(inits[by_name[n['name']].input[1]].dims)) for n in nodes}
        tokens = np.load(path.parent / ('validation_tokens.npy' if tag == 'gpt2' else 'calib_tokens.npy'))
        flops = {n: elems[n] * (tokens.shape[1] if tag == 'gpt2' or by_name[n].op_type == 'MatMul' else 1)
                 for n in elems}
    return nodes, elems, flops


def nondominated(rows, axes):
    values = np.array([[r[a] for a in axes] for r in rows], dtype=float)
    # Normalize to make the numerical tolerance independent of BC units.
    scale = np.ptp(values, axis=0)
    scale[scale == 0] = 1
    values = (values - values.min(axis=0)) / scale
    return [i for i, v in enumerate(values)
            if not np.any(np.all(values <= v + 1e-10, axis=1)
                          & np.any(values < v - 1e-10, axis=1))]


def statistics(rows):
    three = set(nondominated(rows, ('size', 'bops', 'objective')))
    projection = set(nondominated(rows, ('size', 'objective'))) | set(nondominated(rows, ('bops', 'objective')))
    # Equal projected coordinates can retain a point dominated on the omitted axis.
    projection &= three
    missed = sorted(three - projection)
    examples = []
    for i in missed:
        r = rows[i]
        feasible = [rows[j] for j in projection
                    if rows[j]['size'] <= r['size'] + 1e-10 and rows[j]['bops'] <= r['bops'] + 1e-10]
        best = min((x['objective'] for x in feasible), default=None)
        examples.append(dict(index=i, size=r['size'], bops=r['bops'], objective=r['objective'],
                             best_projected_bc_at_same_budgets=best,
                             projected_bc_ratio=(best / r['objective'] if best is not None and r['objective'] > 0 else None)))
    s, p = np.array([[r['size'], r['bops']] for r in rows]).T
    return dict(unique_configs=len(rows), spearman=float(spearmanr(s, p).statistic),
                frontier_3d=len(three), projection_union_on_3d=len(projection),
                missed_count=len(missed), missed=examples,
                frontier_3d_indices=sorted(three), projection_indices=sorted(projection))


def dedup(rows):
    out = {}
    for row in rows:
        key = json.dumps(row['formats'], sort_keys=True)
        out.setdefault(key, row)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cached-dir', type=Path, default=Path('/tmp/native_auto_cached_checks'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--time-limit', type=float, default=15)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    # Check the key geometric case independently of model data.
    toy = [dict(size=.4, bops=1, objective=.8), dict(size=1, bops=.4, objective=.8),
           dict(size=.7, bops=.7, objective=1)]
    assert nondominated(toy, ('size', 'bops', 'objective')) == [0, 1, 2]
    assert set(nondominated(toy, ('size', 'objective'))) | set(nondominated(toy, ('bops', 'objective'))) == {0, 1}
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    local_fig, local_axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, local_ax, tag in zip(axes.flat, local_axes.flat, [*CNNS, 'gpt2', 'bert']):
        print(f'{tag}: loading cached tables', flush=True)
        cached = json.loads((args.cached_dir / f'{tag}.json').read_text())
        table = json.loads((HERE / 'runs/native_full_20261009' / tag / 'table.json').read_text())
        B, C = table['B'], table['C']
        nodes, elems, flops = metadata(tag)
        names = [n['name'] for n in nodes]
        assert set(names) == set(B)
        ref_s, ref_p = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
        assert math.isclose(ref_s, cached['reference']['weight_bytes'])
        assert math.isclose(ref_p, cached['reference']['bops'])

        def cost(n, w, a):
            if B[n][w] is None or C[n][a] is None:
                return None
            v = B[n][w] + C[n][a]
            return v if math.isfinite(v) else None

        def coordinates(cfg):
            return dict(size=sum(elems[n] * FORMAT_BITS[cfg[n]['w']] / 8 for n in names) / ref_s,
                        bops=sum(flops[n] * FORMAT_BITS[cfg[n]['w']] * FORMAT_BITS[cfg[n]['a']] for n in names) / ref_p,
                        objective=sum(cost(n, cfg[n]['w'], cfg[n]['a']) for n in names))

        old = dedup(cached['candidates'])
        for r in old:
            for k, v in coordinates(r['formats']).items():
                assert math.isclose(v, r[k], rel_tol=1e-9, abs_tol=1e-15), (tag, k, v, r[k])
        center = cached['selected']
        new, failures = [], []
        started = time.monotonic()
        for fs in (.8, 1., 1.25):
            for fp in (.8, 1., 1.25):
                if fs == fp == 1:
                    continue
                bs, bp = center['size'] * fs, center['bops'] * fp
                t0 = time.monotonic()
                try:
                    cfg, obj, status = milp_select(nodes, GRID11, GRID11, cost, elems, flops,
                                                   bs * ref_s, bp * ref_p, time_limit=args.time_limit)
                    r = dict(formats=cfg, **coordinates(cfg), solver_status=status,
                             size_budget=bs, bops_budget=bp, origin='independent_probe', seconds=time.monotonic()-t0)
                    assert status in (0, 1)
                    assert r['size'] <= bs + 1e-10 and r['bops'] <= bp + 1e-10
                    assert math.isclose(obj, r['objective'], rel_tol=1e-9, abs_tol=1e-15)
                    new.append(r)
                    print(f'{tag}: ({fs}, {fp}) status={status} BC={obj:.6g}', flush=True)
                except RuntimeError as exc:
                    failures.append(dict(size_budget=bs, bops_budget=bp, error=str(exc)))
                    print(f'{tag}: ({fs}, {fp}) failed: {exc}', flush=True)
        rows = dedup(old + new)
        local = [r for r in rows if .75 * center['size'] <= r['size'] <= 1.3 * center['size']
                 and .75 * center['bops'] <= r['bops'] <= 1.3 * center['bops']]
        stat = statistics(rows)
        result = dict(model=tag, cached=statistics(old), augmented=stat,
                      near_selected=statistics(local) if len(local) >= 3 else None,
                      probe_seconds=time.monotonic()-started, successful_probes=len(new),
                      optimal_probes=sum(r['solver_status'] == 0 for r in new), failures=failures,
                      original_selected=center, candidates=rows,
                      limitations=['sampled frontier only', 'local budget probes only',
                                   'strict final-set projection, not HTQ per-layer heuristic',
                                   'BC proxy, no accuracy evaluation'])
        (args.out_dir / f'{tag}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        summary[tag] = {k: v for k, v in result.items() if k not in ('candidates', 'original_selected')}
        ax.scatter([r['size'] for r in old], [r['bops'] for r in old], label='Common budget', alpha=.6)
        ax.scatter([r['size'] for r in new], [r['bops'] for r in new], marker='x', label='Independent probes')
        missed = [rows[i['index']] for i in stat['missed']]
        ax.scatter([r['size'] for r in missed], [r['bops'] for r in missed], s=90, facecolors='none',
                   edgecolors='red', label='3D points lost by projection')
        ax.set(xscale='log', yscale='log', xlabel='Size / int8 size', ylabel='BOPs / int8 BOPs', title=tag)
        ax.grid(alpha=.2)
        for group, marker, label in ((old, 'o', 'Common budget'), (new, 'x', 'Independent probes')):
            local_ax.scatter([r['size'] / center['size'] for r in group],
                             [r['bops'] / center['bops'] for r in group], marker=marker, label=label)
        local_ax.scatter([r['size'] / center['size'] for r in missed],
                         [r['bops'] / center['bops'] for r in missed], s=110,
                         facecolors='none', edgecolors='red', label='Lost by projection')
        local_ax.scatter([1], [1], marker='*', s=110, color='black', label='Original selection')
        local_ax.set(xlim=(.75, 1.3), ylim=(.75, 1.3), xlabel='Size / original selected size',
                     ylabel='BOPs / original selected BOPs',
                     title=f"{tag}: local Spearman = {result['near_selected']['spearman']:.2f}")
        local_ax.grid(alpha=.25)
        print(f'{tag}: 3D={stat["frontier_3d"]}, lost={stat["missed_count"]}, rho={stat["spearman"]:.3f}', flush=True)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'resource_projection.png', dpi=160)
    local_axes.flat[-1].axis('off')
    local_axes.flat[-1].legend(*local_axes.flat[0].get_legend_handles_labels(), loc='center')
    local_fig.tight_layout()
    local_fig.savefig(args.out_dir / 'resource_projection_local.png', dpi=160)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
