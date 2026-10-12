#!/usr/bin/env python3
"""Native BC -> automatic resource/format choice -> compiled model, without a user budget.

The pipeline of NativeMixedPrecisionPipeline.md: native BC table (run_native_bc_pipeline.native_tables), MILP
common-budget sweep on max(size, BOPs) and the normalized chord knee of that frontier (mp_auto.auto_select), then the
pick compiled (constants to file) and evaluated on the held-out data, next to the FP32 model compiled the same way.
--policy 3d adds independent local size/BOPs budgets, a 3-D Pareto set and the anchor-normalized ideal-distance pick
(mp_auto.auto_select_3d); it is not the default because its BC axis is scaled by the per-node cheapest assignment,
which on Qwen3-0.6B (int4 + p4e1, BC 2e5) squashed every usable configuration into <1% of the axis and picked int4
weights everywhere (PPL 108 vs FP32 49.9; the 2-D knee: 53.5).

Supported model/data assets are those of run_native_bc_pipeline.py (mobilenetv2, resnet18, efficientnet_b0, gpt2,
bert, qwen3). Selection uses calibration BC only; held-out metrics are reported after the decision and never used to
choose it. The policy is not an accuracy guarantee.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from evaluate_native_search_chunked import chunked_run_so
from mp_auto import auto_select, auto_select_3d
from run_native_bc_pipeline import (CNNS, GRID11, bert_spec, cnn_spec, erm,
                                    gpt2_spec, native_tables, qwen3_spec)

MODELS = (*CNNS, 'gpt2', 'bert', 'qwen3')


def compile_and_eval(spec, tools, work, name, env, posit_obj, chunks):
    """Build (constants to file, own folder) and evaluate one configuration; env None is the FP32 model."""
    target = work / name
    target.mkdir(exist_ok=True)
    t0 = time.time()
    so, info = erm.build_so(tools, target, name, work / f'{erm.SYMBOL_POSTFIX}.onnx.mlir', env, posit_obj,
                            opt=spec.eval_levels[1], krnl_opt=spec.eval_levels[0], constants_to_file=True)
    compile_s = time.time() - t0
    if info['warnings']:
        raise RuntimeError(f'{name}: compilation emitted warnings: {info["warnings"]}')
    for p in target.iterdir():                   # keep the .so and its constants
        if p.suffix in ('.mlir', '.ll', '.bc'):
            p.unlink()
    z, ev = chunked_run_so(chunks)(tools, so, work / 'eval_inputs.npy', target / 'logits.npy', 36000)
    if not np.isfinite(z).all():
        raise RuntimeError(f'{name}: non-finite logits; no successful result reported')
    metrics = spec.metrics(z)
    per_item = metrics.pop('per_item')
    np.save(target / 'per_item.npy', per_item)
    (target / 'logits.npy').unlink()             # GPT-2 / Qwen3: GBs; per_item keeps what the report needs
    if not all(np.isfinite(v) for v in metrics.values()):
        raise RuntimeError(f'{name}: non-finite evaluation metrics')
    return dict(metrics=metrics, compile_s=compile_s, eval=ev, model_library=str(so.resolve())), per_item


def run_auto(spec, tools, work, log, policy='2d', eval_chunks=1, fp32_baseline=True, time_limit=10):
    B, C, ranges, timing = native_tables(spec, tools, work, log)
    (work / 'table.json').write_text(json.dumps({'B': B, 'C': C}) + '\n')
    nonfinite = sum(not np.isfinite(v) for t in (B, C) for row in t.values() for v in row.values())
    log(f'{spec.tag}: native BC table in {timing["criterion_total_s"]:.0f}s; {nonfinite} non-finite entries '
        f'(forbidden options)')
    t0 = time.time()
    selected_nodes = [n for n in spec.nodes if n['name'] in spec.names]
    select = auto_select_3d if policy == '3d' else auto_select
    selection = select(selected_nodes, GRID11, GRID11, lambda n, w, a: B[n][w] + C[n][a],
                       spec.weight_elems, spec.flops, time_limit=time_limit)
    timing['auto_search_s'] = time.time() - t0
    (work / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')
    chosen = selection['selected']
    posit = {k: sum(f[k].startswith('p') for f in chosen['formats'].values()) for k in ('w', 'a')}
    log(f"{spec.tag}: automatic choice ({selection['policy']}) size={chosen['size']:.3f}, BOPs={chosen['bops']:.3f} "
        f"relative to int8; BC={chosen['objective']:.6g}; posit w/a {posit['w']}/{posit['a']}; search "
        f"{timing['auto_search_s']:.1f}s")
    cfg = {n['name']: chosen['formats'].get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}
    env = erm.node_formats_env(cfg, spec.nodes, ranges)
    (work / 'selected_formats.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (work / 'compile_env.json').write_text(json.dumps(
        {'MIXEDP_NODE_FORMATS': env, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'}, indent=2) + '\n')
    np.save(work / 'eval_inputs.npy', spec.eval_inputs)
    posit_obj = erm.build_posit_runtime(tools, work)
    previous = os.environ.get('MIXEDP_WEIGHT_PER_CHANNEL')
    try:
        os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
        result, per_item = compile_and_eval(spec, tools, work, 'selected', env, posit_obj, eval_chunks)
    finally:
        if previous is None:
            os.environ.pop('MIXEDP_WEIGHT_PER_CHANNEL', None)
        else:
            os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = previous
    timing['selected_compile_s'], timing['selected_eval_s'] = result['compile_s'], result['eval']['wall_seconds']
    log(f'{spec.tag}: selected {result["metrics"]} (compile {result["compile_s"]:.0f}s, '
        f'eval {result["eval"]["wall_seconds"]:.0f}s)')
    out = dict(policy=selection['policy'], selected=chosen, metrics=result['metrics'], timing=timing,
               model_library=result['model_library'], num_eval=len(spec.eval_inputs),
               all_logits_finite=True, accuracy_guarantee=False, evaluation_used_for_selection=False)
    if fp32_baseline:
        base, base_item = compile_and_eval(spec, tools, work, 'fp32', None, None, eval_chunks)
        d = per_item - base_item                 # per-item loss, selected - FP32 (lower is better)
        out['fp32'] = dict(metrics=base['metrics'], compile_s=base['compile_s'],
                           eval_s=base['eval']['wall_seconds'])
        out['loss_delta_vs_fp32'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
        log(f'{spec.tag}: FP32 {base["metrics"]}; selected - FP32 per-item loss '
            f'{d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}')
    (work / 'results.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='mobilenetv2', help='comma-separated: ' + ', '.join(MODELS))
    ap.add_argument('--images-dir', type=Path,
                    default=here / 'runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images')
    ap.add_argument('--num-eval', type=int, default=1000, help='CNN evaluation size only; never affects selection')
    ap.add_argument('--policy', choices=('2d', '3d'), default='2d')
    ap.add_argument('--time-limit', type=float, default=10, help='seconds per MILP solve')
    ap.add_argument('--eval-chunks', type=int, default=1, help='processes per held-out evaluation')
    ap.add_argument('--no-fp32-baseline', action='store_true')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    models = args.models.split(',')
    if args.num_eval <= 0 or args.eval_chunks <= 0 or any(m not in MODELS for m in models):
        ap.error('need positive num-eval / eval-chunks and supported model names')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        ap.error('out-dir must be empty to keep previous experiment artifacts intact')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)

    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    summary = {}
    for tag in models:
        work = args.out_dir / tag
        work.mkdir()
        spec = (gpt2_spec(work, with_references=False) if tag == 'gpt2' else
                bert_spec(work, with_references=False) if tag == 'bert' else
                qwen3_spec(work) if tag == 'qwen3' else
                cnn_spec(tag, args.images_dir, args.num_eval, work, with_references=False))
        summary[tag] = run_auto(spec, tools, work, log, args.policy, args.eval_chunks,
                                not args.no_fp32_baseline, args.time_limit)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
