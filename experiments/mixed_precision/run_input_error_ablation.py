#!/usr/bin/env python3
"""Remove only input activation error from shared_v2; keep actual input QDQ."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import onnx

from check_shared_fix import tap_samples
from eval_random_mixed_precision import (Tools, SYMBOL_POSTFIX, build_so, expected_sites,
    node_formats_env, preprocess_by_source, run, run_so, simulate_in_onnxruntime, supported_nodes)
from htq_perturbation import activation_perturbation
from mp_milp import milp_select
from sample_configs_htq import compute_weight_elems_and_flops_proxy
from sample_configs_sqnr import config_cost


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda v: v.item()) + '\n')


def report(out, settings, results):
    text = ['# ResNet18：是否納入 input activation error 的消融', '',
        '兩組都用相同權重 sensitivity，實際量化都包含 weight、input 與 output。唯一修改是 criterion 是否納入 input 誤差。', '',
        '| 設定 | 內容 |', '|---|---|', '| 校準／held-out | 10／1,000 張 ImageNet-1K validation 子集 |',
        '| FP32 top-1 | 70.6% |', '| 求解器 | MILP，所有配置必須證明最優且符合預算 |',
        '| W/A 候選 | int4–8、f16、bf16、fp32（各 8 種） |',
        '| W＋O | 相對平方 weight 誤差＋相對平方 output 誤差 |',
        '| W＋I＋O | shared_v2：再加相對平方 input 誤差 |', '',
        '兩者共同乘上 S_i = mean(abs(W_i * dL/dW_i))。W＋O 僅在評分時省略 input；模型執行時 input QDQ 保持開啟。', '']
    for backend, label in [('simulation', 'ORT 模擬'), ('compiled', 'ONNX-MLIR 編譯後 QDQ')]:
        rows = results.get(backend, [])
        text += [f'## {label}', '', '| f | W＋O top-1 | W＋I＋O top-1 | 加入 input 後差異（百分點） |', '|---|---:|---:|---:|']
        for row in sorted(rows, key=lambda r: r['budget']):
            text.append(f"| {row['budget']:.2f} | {100*row['output_only_top1']:.1f}% | {100*row['full_top1']:.1f}% | {100*(row['full_top1']-row['output_only_top1']):+.1f} |")
        if len(rows) < len(settings['budgets']):
            text += ['', f'目前完成 {len(rows)}/{len(settings["budgets"])} 組；未完成的結果不列入表格。']
        if rows:
            with (out / f'{backend}_comparison.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        text += ['']
    text += ['完整 shared_v2 結果沿用同一模型、資料、敏感度與格式選單的前一輪實驗；重建的輸入／輸出誤差已與原始表核對。兩種後端分開比較。', '',
        '相同的是資源上限，實際權重大小與 BOPs 用量可能不同。此為固定校準資料的單次比較，不代表加入 input 對所有模型或預算一定有利。', '',
        '[設定與來源](experiment.json) · [原始結果](results.json) · [配置與成本](configs.json)']
    (out / 'report.md').write_text('\n'.join(text) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    base, out = args.baseline.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads((base / 'experiment.json').read_text())
    model_dir = Path(settings['model_dir'])
    assert hashlib.sha256((model_dir / 'model.onnx').read_bytes()).hexdigest() == settings['model_sha256']
    model = onnx.load(str(model_dir / 'model.onnx'))
    nodes = supported_nodes(model)
    weighted = [n for n in nodes if n.get('weight_const')]
    sensitivity = json.loads((base / 'sensitivity.json').read_text())['weight']
    previous = json.loads((base / 'perturbation_tables.json').read_text())
    ranges = json.loads((base / 'compiled/calibration.json').read_text())['min_max']
    cx = np.load(base / 'calib_inputs.npy')
    samples = tap_samples(model, sorted({t for n in weighted for t in n['act_tensors']}), cx)
    formats = settings['formats']
    table = {}
    for n in weighted:
        name, x, y = n['name'], n['act_tensors'][0], n['act_tensors'][-1]
        xn, yn = np.linalg.norm(samples[x]), np.linalg.norm(samples[y])
        assert xn > 0 and yn > 0
        ye = previous['shared'][name]['a']
        output = {a: float((ye[a]/yn)**2) for a in formats}
        for a in formats:
            np.testing.assert_allclose(activation_perturbation(samples[y], a, ranges[y], None), ye[a], rtol=1e-6, atol=1e-12)
            xe = activation_perturbation(samples[x], a, ranges[x], None)
            full = float((xe/xn)**2 + (ye[a]/yn)**2)
            np.testing.assert_allclose(full, previous['shared_v2'][name]['a'][a], rtol=1e-6, atol=1e-12)
        table[name] = {'w': previous['shared_v2'][name]['w'], 'a': output}
    save(out / 'perturbation_tables.json', table)
    shutil.copy2(base / 'sensitivity.json', out / 'sensitivity.json')
    (out / 'images').symlink_to(base / 'images', target_is_directory=True)
    save(out / 'experiment.json', dict(settings, baseline=str(base),
        change='Remove only normalized squared input error from criterion; actual input QDQ unchanged',
        activation_table_reconstruction_verified=True))
    snapshot = out / 'source'; snapshot.mkdir()
    for name in ['run_input_error_ablation.py', 'mp_milp.py', 'check_shared_fix.py', 'htq_perturbation.py', 'eval_random_mixed_precision.py']:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    elems, flops = compute_weight_elems_and_flops_proxy(model, nodes)
    ref_size = sum(elems[n['name']] for n in weighted)
    ref_bops = sum(64*flops[n['name']] for n in weighted)
    configs = []
    def value(name, w, a):
        return sensitivity[name] * (table[name]['w'][w] + table[name]['a'][a])
    for i, budget in enumerate(settings['budgets']):
        cfg, obj, status = milp_select(nodes, formats, formats, value, elems, flops, budget*ref_size, budget*ref_bops)
        assert status == 0
        for n in nodes: cfg[n['name']]['op'] = n['op']
        size, bops = config_cost(cfg, elems, flops)
        configs.append({'id': f'output_only_{i:03d}', 'criterion': 'shared_v2_output_only', 'budget': budget,
            'solver_status': status, 'objective': obj, 'size_vs_int8': size/ref_size, 'bops_vs_int8': bops/ref_bops, 'formats': cfg})
    save(out / 'configs.json', {'configs': configs})
    manifest = json.loads((base / 'images/manifest.json').read_text())
    held = [e for e in manifest['images'] if not e['calibration']]
    assert [e['file'] for e in held] == settings['held_out'] and len(held) == 1000
    labels = np.array([e['label'] for e in held])
    inputs = preprocess_by_source(base / 'images', {'images': held}, json.loads((model_dir / 'metadata.json').read_text()))
    old_sim = json.loads((base / 'simulation_results.json').read_text())
    old_compiled = json.loads((base / 'summary.json').read_text())
    results = {'baseline': str(base), 'simulation': [], 'compiled': [], 'status': 'simulation_running'}
    def row(c, z, backend):
        assert np.isfinite(z).all()
        correct = z.argmax(1) == labels
        old = old_sim if backend == 'simulation' else old_compiled
        full = next(r for r in old['rows'] if r['criterion']=='shared_v2' and r['budget']==c['budget'])
        return {'budget': c['budget'], 'output_only_top1': float(correct.mean()), 'full_top1': full['top1'],
            'output_only_top5': float((np.argsort(-z,axis=1)[:,:5]==labels[:,None]).any(1).mean()),
            'full_top5': full['top5'], 'output_only_size': c['size_vs_int8'], 'full_size': full['size_vs_int8'],
            'output_only_bops': c['bops_vs_int8'], 'full_bops': full['bops_vs_int8']}
    for c in configs:
        z = simulate_in_onnxruntime(model, nodes, c['formats'], ranges, inputs, None)
        np.save(out / f"sim_logits_{c['id']}.npy", z)
        r = row(c, z, 'simulation'); results['simulation'].append(r)
        save(out / 'results.json', results); report(out, settings, results)
        print(f"ORT f={c['budget']:.2f}: W+O={r['output_only_top1']:.1%}, W+I+O={r['full_top1']:.1%}", flush=True)
    results['status'] = 'compiled_running' if args.compile else 'simulation_complete'
    save(out / 'results.json', results)
    if not args.compile: return
    tools = Tools()
    work = Path(tempfile.mkdtemp(prefix='input-ablation-'))
    np.save(work / 'inputs.npy', inputs)
    basic = work / f'{SYMBOL_POSTFIX}.onnx.mlir'
    run([tools.onnx_mlir, '--EmitONNXBasic', '-o', work / SYMBOL_POSTFIX,
         '--mlir-elide-resource-strings-if-larger=1000000000', '--mlir-elide-elementsattrs-if-larger=1000000000', model_dir/'model.onnx'])
    print('Starting native compiled evaluation; simulation complete.', flush=True)
    def compiled(c):
        t = time.time()
        env = node_formats_env(c['formats'], nodes, ranges)
        so, info = build_so(tools, work, c['id'], basic, env, None)
        expected, ir = expected_sites(c['formats'], nodes), info['ir_after_pass']
        assert not info['warnings'], info['warnings']
        assert expected['int']==ir['onnx.Round'] and 2*expected['half']==ir['onnx.Cast']
        z, runtime = run_so(tools, so, work/'inputs.npy', work/f"{c['id']}.npy", 7200)
        np.save(out / f"compiled_logits_{c['id']}.npy", z)
        r = row(c, z, 'compiled')
        save(out / f"{c['id']}_compiled_info.json", dict(info, runtime=runtime, elapsed=time.time()-t))
        return r
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(compiled,c) for c in configs]):
                r = future.result(); results['compiled'].append(r)
                save(out/'results.json',results); report(out,settings,results)
                print(f"Compiled f={r['budget']:.2f}: W+O={r['output_only_top1']:.1%}, W+I+O={r['full_top1']:.1%}",flush=True)
        results['status']='complete'
        shutil.rmtree(work)
    except Exception as exc:
        results['status']='compiled_failed'; results['error']=str(exc)
        raise
    finally:
        save(out/'results.json',results); report(out,settings,results)


if __name__ == '__main__':
    main()
