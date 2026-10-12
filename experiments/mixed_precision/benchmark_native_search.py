#!/usr/bin/env python3
"""Controlled native BC search comparison, followed by compiled held-out evaluation.

Search runs in isolated, one-CPU processes. Both methods share tables, options,
analytical anchors, and a frozen normalized ideal-distance selection policy.
The existing HTQ port (including its set-union order) is used without alteration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ('mobilenetv2', 'resnet18', 'efficientnet_b0', 'gpt2', 'bert')
GRID = ('int4', 'int5', 'int6', 'int7', 'int8', 'p4e1', 'p6e1', 'p8e1', 'f16', 'bf16', 'fp32')


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def cfg_key(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


class Problem:
    def __init__(self, data):
        self.data = data
        self.nodes, self.elems, self.flops = data['nodes'], data['elems'], data['flops']
        self.names = [n['name'] for n in self.nodes]
        self.B, self.C = data['B'], data['C']
        self.ref_s = sum(self.elems[n] for n in self.names)
        self.ref_p = sum(64 * self.flops[n] for n in self.names)
        assert all(math.isfinite(self.cost(n, w, a)) and self.cost(n, w, a) >= 0
                   for n in self.names for w in GRID for a in GRID)
        from htq_pareto_search import FORMAT_BITS
        self.bits = FORMAT_BITS
        cheap = {n: dict(zip(('w', 'a'), min(((w, a) for w in GRID for a in GRID),
                    key=lambda x: (FORMAT_BITS[x[0]], FORMAT_BITS[x[0]] * FORMAT_BITS[x[1]], self.cost(n, *x)))))
                 for n in self.names}
        quality = {n: dict(zip(('w', 'a'), min(((w, a) for w in GRID for a in GRID),
                    key=lambda x: (self.cost(n, *x), FORMAT_BITS[x[0]], FORMAT_BITS[x[1]])))) for n in self.names}
        self.anchors = [self.record(cheap, 'cheap_anchor'), self.record(quality, 'quality_anchor')]
        self.lo = np.array([self.anchors[0]['size'], self.anchors[0]['bops'], self.anchors[1]['objective']])
        self.hi = np.array([self.anchors[1]['size'], self.anchors[1]['bops'], self.anchors[0]['objective']])
        assert np.all(self.hi > self.lo)

    def cost(self, n, w, a):
        return self.B[n][w] + self.C[n][a]

    def record(self, cfg, origin, **extra):
        return dict(formats=cfg, config_hash=cfg_key(cfg), origin=origin,
                    objective=sum(self.cost(n, cfg[n]['w'], cfg[n]['a']) for n in self.names),
                    size=sum(self.elems[n] * self.bits[cfg[n]['w']] / 8 for n in self.names) / self.ref_s,
                    bops=sum(self.flops[n] * self.bits[cfg[n]['w']] * self.bits[cfg[n]['a']] for n in self.names) / self.ref_p,
                    **extra)

    def select(self, rows):
        unique = {}
        for r in rows + self.anchors:
            unique.setdefault(r['config_hash'], r)
        rows = list(unique.values())
        v = np.array([[r['size'], r['bops'], r['objective']] for r in rows])
        norm = (v - self.lo) / (self.hi - self.lo)
        front = [i for i, x in enumerate(norm) if not np.any(np.all(norm <= x + 1e-10, axis=1)
                                                             & np.any(norm < x - 1e-10, axis=1))]
        for i, r in enumerate(rows):
            r['selection_score'] = float(np.linalg.norm(norm[i]))
        winner = min(front, key=lambda i: (rows[i]['selection_score'], rows[i]['objective'], rows[i]['config_hash']))
        return rows[winner], rows, front


def search_worker(args):
    # The persisted protocol is authoritative for resumed benchmark processes.
    protocol = json.loads((args.out_dir / 'protocol.json').read_text())
    args.search_cap = protocol['search_cap_seconds']
    if args.method == 'htq_fast':
        args.beam = 200
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    p = Problem(json.loads((args.out_dir / args.model / 'problem.json').read_text()))
    from htq_pareto_search import pareto_search, leaf_config
    from mp_auto import auto_select
    from mp_milp import milp_select
    rows, logs = [], []
    def expired(*_):
        raise TimeoutError('common search wall-time cap reached')
    signal.signal(signal.SIGALRM, expired)
    started = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, args.search_cap)
    completed = False
    details = {}
    try:
        if args.method == 'milp_local':
            coarse = auto_select(p.nodes, GRID, GRID, p.cost, p.elems, p.flops,
                                 samples=17, refinements=2, time_limit=10)
            rows = [p.record(r['formats'], r['origin'], solver_status=r['solver_status']) for r in coarse['candidates']]
            center = coarse['selected']
            details['coarse_selected'] = {k: center[k] for k in ('size', 'bops', 'objective')}
            details['coarse_attempts'] = len(coarse['attempted_budgets'])
            details['coarse_failures'] = coarse['failures']
            for fs in (.8, 1., 1.25):
                for fp in (.8, 1., 1.25):
                    if fs == fp == 1:
                        continue
                    bs, bp = center['size'] * fs, center['bops'] * fp
                    cfg, obj, status = milp_select(p.nodes, GRID, GRID, p.cost, p.elems, p.flops,
                                                   bs * p.ref_s, bp * p.ref_p, time_limit=10)
                    r = p.record(cfg, 'local_probe', solver_status=status, size_budget=bs, bops_budget=bp)
                    assert status in (0, 1) and r['size'] <= bs + 1e-10 and r['bops'] <= bp + 1e-10
                    rows.append(r)
        else:
            ones = dict.fromkeys(p.names, 1.)
            leaves = pareto_search(p.nodes, GRID, GRID, ones, ones, p.B, p.C, p.elems, p.flops,
                                    beam_width=args.beam, log=logs.append)
            rows = [p.record(leaf_config(l), 'htq_leaf') for l in leaves]
            for r, l in zip(rows, leaves):
                assert math.isclose(r['objective'], l.profit, rel_tol=1e-10)
            details['leaves'] = len(leaves)
            details['beam_width'] = args.beam
        selected, rows, frontier = p.select(rows)
        completed = True
    except TimeoutError as exc:
        details['error'] = str(exc)
        selected, rows, frontier = p.select(rows)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = dict(method=args.method, repeat=args.repeat, completed=completed,
                  seconds=time.perf_counter()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), selected=selected, candidates=rows, frontier_indices=frontier,
                  anchors=dict(low=p.lo.tolist(), high=p.hi.tolist()), details=details, logs=logs,
                  selection_policy='fixed_anchor_equal_weight_normalized_ideal_distance', search_cap=args.search_cap)
    dump(args.out_dir / args.model / f'{args.method}_r{args.repeat}.json', result)
    print(json.dumps({k: result[k] for k in ('method', 'completed', 'seconds', 'peak_rss_mb')}), flush=True)


def prepare(args):
    from analyze_resource_projection import metadata
    args.out_dir.mkdir(parents=True, exist_ok=False)
    src = args.out_dir / 'source'
    src.mkdir()
    import shutil
    for name in ('benchmark_native_search.py', 'mp_auto.py', 'mp_milp.py', 'htq_pareto_search.py',
                 'run_native_bc_pipeline.py', 'eval_random_mixed_precision.py', 'analyze_resource_projection.py'):
        shutil.copy2(HERE / name, src / name)
    for tag in args.models.split(','):
        work = args.out_dir / tag
        work.mkdir()
        nodes, elems, flops = metadata(tag)
        table_path = HERE / 'runs/native_full_20261009' / tag / 'table.json'
        table = json.loads(table_path.read_text())
        dump(work / 'problem.json', dict(nodes=nodes, elems=elems, flops=flops, **table,
                                         table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest()))
    dump(args.out_dir / 'protocol.json', dict(models=args.models.split(','), repeats=args.repeats,
         search_cap_seconds=args.search_cap, beam_width=args.beam, formats=GRID,
         criterion='native B(w)+C(a)', held_out_used_for_selection=False,
         selection_policy='equal-weight Euclidean distance, common cheap/minimum-BC analytical anchor ranges',
         milp_search='17 common budgets + 2 knee refinements (up to 6) + 8 independent local budgets',
         local_factors=[.8, 1, 1.25], evaluated_repeat=0,
         primary_comparison='automatic selected configuration from repeat 0, fixed before seeing accuracy',
         resource_control='additional MILP solve at HTQ repeat-0 actual size and BOPs, outside search timing',
         timing='fresh process, one CPU affinity, excludes imports/preparation; includes search and final selection',
         htq_note='existing port, unchanged 1/20 pruning and set union; criterion replaced with the same BC tables'))


def searches(args):
    for rep in range(args.repeats):
        for tag in args.models.split(','):
            methods = ('milp_local', 'htq', 'htq_fast') if rep % 2 == 0 else ('htq_fast', 'htq', 'milp_local')
            for method in methods:
                path = args.out_dir / tag / f'{method}_r{rep}.json'
                if path.exists():
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', 'worker', '--out-dir', str(args.out_dir),
                       '--model', tag, '--method', method, '--repeat', str(rep), '--beam', str(args.beam),
                       '--search-cap', str(args.search_cap)]
                print(f'{tag} repeat {rep} {method}', flush=True)
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.search_cap+120)
                (args.out_dir / tag / f'{method}_r{rep}.log').write_text(r.stdout + r.stderr)
                if r.returncode:
                    raise RuntimeError(r.stderr[-4000:])
                result = json.loads(path.read_text())
                print(f"  completed={result['completed']} {result['seconds']:.2f}s "
                      f"BC={result['selected']['objective']:.6g} S={result['selected']['size']:.3f} P={result['selected']['bops']:.3f}", flush=True)


def evaluate(args):
    from run_native_bc_pipeline import cnn_spec, gpt2_spec, bert_spec, erm, onnx, SLOT_INPUT
    from mp_milp import milp_select
    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    for tag in args.models.split(','):
        work = args.out_dir / tag
        search = {m: json.loads((work / f'{m}_r0.json').read_text()) for m in ('milp_local', 'htq', 'htq_fast')}
        if not all(s['completed'] for s in search.values()):
            raise RuntimeError(f'{tag}: an incomplete primary search cannot be evaluated as a completed method')
        p = Problem(json.loads((work / 'problem.json').read_text()))
        chosen = {m: s['selected'] for m, s in search.items()}
        h = chosen['htq']
        tm = time.perf_counter()
        max_size = sum(p.elems[n] * p.bits[h['formats'][n]['w']] / 8 for n in p.names)
        max_bops = sum(p.flops[n] * p.bits[h['formats'][n]['w']] * p.bits[h['formats'][n]['a']] for n in p.names)
        cfg, obj, status = milp_select(p.nodes, GRID, GRID, p.cost, p.elems, p.flops,
                                       max_size, max_bops, time_limit=60)
        chosen['milp_at_htq_resources'] = p.record(cfg, 'resource_matched_diagnostic', solver_status=status,
                                                  solve_seconds=time.perf_counter()-tm)
        assert status in (0, 1)
        assert chosen['milp_at_htq_resources']['size'] <= h['size']+1e-10
        assert chosen['milp_at_htq_resources']['bops'] <= h['bops']+1e-10
        assert obj <= h['objective'] * (1+1e-6) + 1e-15, 'matched solve must improve or match feasible HTQ BC'
        dump(work / 'evaluation_choices.json', chosen)
        images = HERE / 'runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images'
        spec = gpt2_spec(work, False) if tag == 'gpt2' else bert_spec(work, False) if tag == 'bert' else cnn_spec(tag, images, 1000, work, False)
        np.save(work / 'eval_inputs.npy', spec.eval_inputs)
        if tag in ('mobilenetv2', 'resnet18', 'efficientnet_b0'):
            manifest = json.loads((images / 'manifest.json').read_text())
            labels = np.array([e['label'] for e in manifest['images'] if not e['calibration']][:1000])
        elif tag == 'bert':
            labels = np.load(HERE / 'runs/bert_sst2_20261006/val_labels.npy')
        else:
            labels = spec.eval_inputs[:, 1:]
        np.save(work / 'eval_labels.npy', labels)
        old = HERE / 'runs/native_full_20261009' / tag
        graph = onnx.load(str(spec.model_path), load_external_data=False)
        by_name = {n.name: n for n in graph.graph.node}
        points = json.loads((old / 'ranges_by_point.json').read_text())
        ranges = {}
        for name, slots in points.items():
            for slot, r in slots.items():
                n = by_name[name]
                tensor = n.output[0] if slot == 'y' else n.input[SLOT_INPUT[slot]]
                assert ranges.setdefault(tensor, r) == r
        posit_obj = erm.build_posit_runtime(tools, work)

        def run_one(method, row):
            target = work / method
            target.mkdir(exist_ok=True)
            if (target / 'result.json').exists():
                cached = json.loads((target / 'result.json').read_text())
                if cached['selected']['config_hash'] != row['config_hash']:
                    raise RuntimeError(f'{tag}/{method}: cached result belongs to a different configuration')
                return cached
            print(f'{tag}: compiling {method}', flush=True)
            full = {n['name']: row['formats'].get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}
            env = erm.node_formats_env(full, spec.nodes, ranges)
            dump(target / 'compile_env.json', dict(MIXEDP_NODE_FORMATS=env, MIXEDP_WEIGHT_PER_CHANNEL='1'))
            tc = time.perf_counter()
            so, info = erm.build_so(tools, target, 'selected', old / 'model.onnx.mlir', env, posit_obj,
                                    constants_to_file=True)
            compile_s = time.perf_counter()-tc
            if info['warnings']:
                raise RuntimeError(f'{tag}/{method}: compiler warnings {info["warnings"]}')
            print(f'{tag}: evaluating {method}', flush=True)
            z, ev = erm.run_so(tools, so, work / 'eval_inputs.npy', target / 'logits.npy', 36000)
            assert np.isfinite(z).all(), f'{tag}/{method}: nonfinite logits'
            metrics = spec.metrics(z)
            per_item = metrics.pop('per_item')
            np.save(target / 'per_item.npy', per_item)
            correct = (z[:, :-1].argmax(-1) == labels).mean(1) if tag == 'gpt2' else (z.argmax(1) == labels).astype(float)
            np.save(target / 'correct.npy', correct)
            assert all(math.isfinite(x) for x in metrics.values()) and np.isfinite(per_item).all()
            result = dict(method=method, selected=row, metrics=metrics, num_eval=len(spec.eval_inputs),
                          all_logits_finite=True, compile_seconds=compile_s, eval=ev, compile_info=info,
                          evaluation_used_for_selection=False)
            dump(target / 'result.json', result)
            print(f'{tag}/{method}: {metrics}', flush=True)
            # Preserve executable, constants, logits and evidence; remove reproducible large intermediate IR.
            for path in target.iterdir():
                if path.suffix in ('.mlir', '.ll', '.bc'):
                    path.unlink()
            return result

        unique, aliases = {}, {}
        for method, row in chosen.items():
            key = row['config_hash']
            if key in unique:
                aliases[method] = unique[key]
            else:
                unique[key] = method
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {m: pool.submit(run_one, m, chosen[m]) for m in unique.values()}
            results = {m: f.result() for m, f in jobs.items()}
        for method, original in aliases.items():
            results[method] = dict(results[original], method=method, selected=chosen[method], artifact_alias=original)
        dump(work / 'evaluation.json', results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=('prepare', 'search', 'worker', 'evaluate'), required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--model', default='mobilenetv2')
    ap.add_argument('--method', choices=('milp_local', 'htq', 'htq_fast'), default='milp_local')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--beam', type=int, default=2000)
    ap.add_argument('--search-cap', type=float, default=60)
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    {'prepare': prepare, 'search': searches, 'worker': search_worker, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
