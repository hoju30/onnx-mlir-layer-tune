#!/usr/bin/env python3
"""Compile and evaluate further candidates of a finished run_native_auto.py search, to see the held-out trade-off
along the frontier around (and away from) the automatic pick.

Candidates come from <run-dir>/<model>/selection.json: for each --sizes entry "size" or "size:bops", the frontier
candidate nearest it (by size, then lower BC; or by |size - s| + |bops - b|). They are built and evaluated exactly as the pick (run_native_auto.compile_and_eval: the
same int ranges, per-channel weights, build levels, chunked evaluation) into <run-dir>/<model>/frontier_<size>/, and
compared per item with the run's FP32 and selected models. This is a report of the trade-off only: held-out results
must not feed back into the choice of the pick or the policy's parameters without a separate validation split.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import onnx
from run_native_auto import compile_and_eval
from run_native_bc_pipeline import bert_spec, cnn_spec, erm, gpt2_spec, qwen3_spec
from run_native_calib_check import SLOT_INPUT


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--sizes', required=True,
                    help='comma-separated targets "size" or "size:bops" (relative to int8)')
    ap.add_argument('--images-dir', type=Path, default=here / 'runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images')
    ap.add_argument('--num-eval', type=int, default=1000)
    ap.add_argument('--eval-chunks', type=int, default=6)
    ap.add_argument('--parallel', type=int, default=2, help='candidates compiled / evaluated at once')
    args = ap.parse_args()
    work = args.run_dir.resolve() / args.model
    t0 = time.time()

    def log(msg):
        print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)

    tag = args.model
    spec = (gpt2_spec(work, with_references=False) if tag == 'gpt2' else
            bert_spec(work, with_references=False) if tag == 'bert' else
            qwen3_spec(work) if tag == 'qwen3' else
            cnn_spec(tag, args.images_dir, args.num_eval, work, with_references=False))
    selection = json.loads((work / 'selection.json').read_text())
    front = [selection['candidates'][i] for i in selection['frontier_indices']]
    picks = {}
    for target in args.sizes.split(','):
        if ':' in target:
            ts, tb = map(float, target.split(':'))
            row = min(front, key=lambda r: (abs(r['size'] - ts) + abs(r['bops'] - tb), r['objective']))
            picks[f'frontier_{row["size"]:.3f}_{row["bops"]:.3f}'] = row
        else:
            row = min(front, key=lambda r: (abs(r['size'] - float(target)), r['objective']))
            picks[f'frontier_{row["size"]:.3f}'] = row
    by_name = {n.name: n for n in onnx.load(str(spec.model_path), load_external_data=False).graph.node}
    ranges = {}
    for node, slots in json.loads((work / 'ranges_by_point.json').read_text()).items():
        for slot, r in slots.items():
            n = by_name[node]
            t = n.output[0] if slot == 'y' else n.input[SLOT_INPUT[slot]]
            assert ranges.setdefault(t, tuple(r)) == tuple(r)
    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    posit_obj = erm.build_posit_runtime(tools, work)
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    fp32_item = np.load(work / 'fp32' / 'per_item.npy')
    sel_item = np.load(work / 'selected' / 'per_item.npy')

    def one(name, row):
        cfg = {n['name']: row['formats'].get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}
        env = erm.node_formats_env(cfg, spec.nodes, ranges)
        log(f'{name}: compiling (size {row["size"]:.3f}, BOPs {row["bops"]:.3f}, BC {row["objective"]:.5g})')
        result, item = compile_and_eval(spec, tools, work, name, env, posit_obj, args.eval_chunks)

        def paired(base):
            d = item - base
            return {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d)))}
        out = {'size': row['size'], 'bops': row['bops'], 'objective': row['objective'],
               'selection_score': row['selection_score'], 'origin': row['origin'],
               'metrics': result['metrics'], 'compile_s': result['compile_s'],
               'eval_s': result['eval']['wall_seconds'], 'loss_delta_vs_fp32': paired(fp32_item),
               'loss_delta_vs_selected': paired(sel_item)}
        log(f'{name}: {result["metrics"]}; vs FP32 {out["loss_delta_vs_fp32"]["mean"]:+.4f}')
        return name, out

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = dict(pool.map(lambda kv: one(*kv), picks.items()))
    path = work / 'frontier_eval.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
