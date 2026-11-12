#!/usr/bin/env python3
"""Controlled shared vs shared_v2 MILP selection, followed by compiled QDQ evaluation."""
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

import numpy as np
import onnx

from check_shared_fix import tap_samples
from eval_random_mixed_precision import calibrate_ranges, preprocess_by_source, supported_nodes
from htq_perturbation import activation_perturbation, weight_perturbation
from mp_milp import milp_select
from sample_configs_htq import DEFAULT_ORT_TRAINING_PYTHON, compute_weight_elems_and_flops_proxy
from sample_configs_sqnr import config_cost

HERE = Path(__file__).resolve().parent
FORMATS = ['int4', 'int5', 'int6', 'int7', 'int8', 'f16', 'bf16', 'fp32']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.item()) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--images-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    model_dir, source, out = (p.resolve() for p in (args.model_dir, args.images_dir, args.out_dir))
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = json.loads((source / 'manifest.json').read_text())
    calib = [e for e in manifest['images'] if e['calibration']][:10]
    held = [e for e in manifest['images'] if not e['calibration']][:1000]
    assert len(calib) == 10 and len(held) == 1000
    selected = {e['file'] for e in calib + held}
    assert len(selected) == 1010
    entries = [e for e in manifest['images'] if e['file'] in selected]
    images = out / 'images'
    images.mkdir()
    for e in entries:
        (images / e['file']).symlink_to(source / e['file'])
    save(images / 'manifest.json', dict(manifest, num_images=1010, num_calibration=10, images=entries,
         selection='First 10 calibration and first 1000 held-out entries of source manifest',
         source_manifest=str(source / 'manifest.json')))
    snapshot = out / 'source'
    snapshot.mkdir()
    hashes = {}
    for name in ['run_shared_vs_v2.py', 'check_shared_fix.py', 'htq_snip_ort.py',
                 'htq_perturbation.py', 'mp_milp.py', 'sample_configs_htq.py',
                 'sample_configs_sqnr.py', 'htq_pareto_search.py', 'eval_random_mixed_precision.py']:
        shutil.copy2(HERE / name, snapshot / name)
        hashes[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    budgets = [.60, .65, .70, .75, .80]
    save(out / 'experiment.json', {'model_dir': str(model_dir), 'model_sha256': hashlib.sha256(
        (model_dir / 'model.onnx').read_bytes()).hexdigest(), 'source_hashes': hashes,
        'calibration': [e['file'] for e in calib], 'held_out': [e['file'] for e in held],
        'formats': FORMATS, 'budgets': budgets, 'solver': 'MILP; status 0 required',
        'shared': 'S * (raw weight L2 + raw output activation L2)',
        'shared_v2': 'S * (relative squared weight + input + output noise)',
        'evaluation': 'onnx-mlir compiled QDQ and same-config ORT simulation check'})
    model = onnx.load(str(model_dir / 'model.onnx'))
    meta = json.loads((model_dir / 'metadata.json').read_text())
    cx = preprocess_by_source(images, {'images': calib}, meta)
    nodes = supported_nodes(model)
    weighted = [n for n in nodes if n.get('weight_const')]
    ranges = calibrate_ranges(model, nodes, cx)
    np.save(out / 'calib_inputs.npy', cx)
    np.save(out / 'calib_labels.npy', [e['label'] for e in calib])
    subprocess.run([str(DEFAULT_ORT_TRAINING_PYTHON), str(HERE / 'htq_snip_ort.py'),
        '--onnx', str(model_dir / 'model.onnx'), '--inputs', str(out / 'calib_inputs.npy'),
        '--labels', str(out / 'calib_labels.npy'), '--mode', 'shared',
        '--out', str(out / 'sensitivity.json')], check=True)
    sensitivity = json.loads((out / 'sensitivity.json').read_text())['weight']
    samples = tap_samples(model, sorted({t for n in weighted for t in n['act_tensors']}), cx)
    inits = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    tables = {'shared': {}, 'shared_v2': {}}
    for n in weighted:
        name = n['name']
        x, y = n['act_tensors'][0], n['act_tensors'][-1]
        W = onnx.numpy_helper.to_array(inits[by_name[name].input[1]]).astype(np.float64)
        norms = [np.linalg.norm(v) for v in (W, samples[x], samples[y])]
        if not all(np.isfinite(v) and v > 0 for v in norms):
            raise ValueError(f'{name}: undefined relative error for zero/nonfinite signal')
        we = {f: weight_perturbation(W, f, None) for f in FORMATS}
        xe = {f: activation_perturbation(samples[x], f, ranges[x], None) for f in FORMATS}
        ye = {f: activation_perturbation(samples[y], f, ranges[y], None) for f in FORMATS}
        tables['shared'][name] = {'w': we, 'a': ye}
        tables['shared_v2'][name] = {
            'w': {f: (we[f] / norms[0])**2 for f in FORMATS},
            'a': {f: (xe[f] / norms[1])**2 + (ye[f] / norms[2])**2 for f in FORMATS}}
    save(out / 'perturbation_tables.json', tables)
    elems, flops = compute_weight_elems_and_flops_proxy(model, nodes)
    ref_size = sum(elems[n['name']] for n in weighted)
    ref_bops = sum(64 * flops[n['name']] for n in weighted)
    configs = []
    for i, budget in enumerate(budgets):
        for criterion, table in tables.items():
            def value(name, w, a):
                return sensitivity[name] * (table[name]['w'][w] + table[name]['a'][a])
            formats, objective, status = milp_select(nodes, FORMATS, FORMATS, value, elems, flops,
                                                     budget * ref_size, budget * ref_bops)
            assert status == 0, (criterion, budget, status)
            for n in nodes:
                formats[n['name']]['op'] = n['op']
            size, bops = config_cost(formats, elems, flops)
            configs.append({'id': f'{criterion}_{i:03d}', 'criterion': criterion, 'budget': budget,
                'objective': objective, 'solver_status': status, 'cost_weight_bytes': size,
                'bops_proxy': bops, 'size_vs_int8': size/ref_size, 'bops_vs_int8': bops/ref_bops,
                'formats': formats})
            save(out / 'configs.json', {'configs': configs})
            print(f'{criterion} f={budget:.2f}: optimal; size={size/ref_size:.4f}, BOPs={bops/ref_bops:.4f}', flush=True)
    print(f'Configuration generation: {time.time()-started:.1f}s; starting compiled evaluation', flush=True)
    subprocess.run([sys.executable, str(HERE / 'eval_random_mixed_precision.py'),
        '--model-dir', str(model_dir), '--images-dir', str(images), '--num-images', '1010',
        '--num-calibration', '10', '--data-seed', str(manifest['seed']), '--seeds', '',
        '--formats', ','.join(FORMATS), '--htq-configs', str(out / 'configs.json'),
        '--jobs', str(args.jobs), '--out-dir', str(out / 'compiled')], check=True)
    results = json.loads((out / 'compiled/results.json').read_text())
    rows = []
    for c in configs:
        r = results['variants'][c['id']]
        assert 'error' not in r, r
        assert r['held_out']['n'] == 1000 and r['sites_match'] and not r['warnings'], r
        assert r['non_finite_rows'] == 0, r
        rows.append({'criterion': c['criterion'], 'budget': c['budget'],
            'top1': r['held_out']['top1'], 'top5': r['held_out']['top5'],
            'size_vs_int8': c['size_vs_int8'], 'bops_vs_int8': c['bops_vs_int8']})
    save(out / 'summary.json', {'fp32_top1': results['variants']['onnx_mlir_fp32']['held_out']['top1'],
        'num_calibration': 10, 'num_eval': 1000, 'rows': rows, 'wall_seconds': time.time()-started})
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Completed:', out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
