#!/usr/bin/env python3
"""Evaluate saved shared/v2 configs with ORT while native compiled evaluation runs."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import onnx
from eval_random_mixed_precision import preprocess_by_source, supported_nodes, simulate_in_onnxruntime

out = Path(sys.argv[1]).resolve()
settings = json.loads((out / 'experiment.json').read_text())
model_dir = Path(settings['model_dir'])
model = onnx.load(str(model_dir / 'model.onnx'))
nodes = supported_nodes(model)
manifest = json.loads((out / 'images/manifest.json').read_text())
held = [e for e in manifest['images'] if not e['calibration']]
assert len(held) == 1000
inputs = preprocess_by_source(out / 'images', {'images': held}, json.loads((model_dir / 'metadata.json').read_text()))
labels = np.array([e['label'] for e in held])
ranges = json.loads((out / 'compiled/calibration.json').read_text())['min_max']
configs = json.loads((out / 'configs.json').read_text())['configs']
result = {'backend': 'ONNX Runtime QDQ simulation', 'num_eval': 1000, 'num_calibration': 10, 'rows': []}
t0 = time.time()
for c in [{'id': 'fp32', 'formats': {}}] + configs:
    z = simulate_in_onnxruntime(model, nodes, c['formats'], ranges, inputs, None)
    finite = np.isfinite(z).all(1)
    top1 = float(((z.argmax(1) == labels) & finite).mean())
    np.save(out / f"sim_logits_{c['id']}.npy", z)
    if c['id'] == 'fp32':
        result['fp32_top1'] = top1
    else:
        result['rows'].append({'id': c['id'], 'criterion': c['criterion'], 'budget': c['budget'],
            'top1': top1, 'top5': float(((np.argsort(-z, axis=1)[:, :5] == labels[:, None]).any(1) & finite).mean()),
            'non_finite_rows': int((~finite).sum()), 'size_vs_int8': c['size_vs_int8'],
            'bops_vs_int8': c['bops_vs_int8']})
    result['wall_seconds'] = time.time() - t0
    (out / 'simulation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{c['id']}: top1={top1:.1%}, nonfinite={int((~finite).sum())}, elapsed={time.time()-t0:.1f}s", flush=True)
