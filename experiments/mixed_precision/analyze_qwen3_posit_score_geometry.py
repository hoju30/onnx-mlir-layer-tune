#!/usr/bin/env python3
"""Exploratory score geometry on EXISTING L27 calibration tensors; no model inference.

This is not a preregistered validation, a replacement criterion, or pipeline change.
No new article losses or selectors are created. Every format/mask/slot is reported.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'runs/qwen3_l27_slot_attribution_20261024'
TRUTH = HERE / 'runs/qwen3_l27_reversal_diag_20261024'
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1')
MASKS = ('all', 'P0', 'Rest')
TP = 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit('use a new output directory; existing exploration is preserved')
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    import torch
    import qwen3_swiglu_common as common
    from run_qwen_down_spike_diag import Quant
    from run_qwen_prefix_rank_shift import sha, write, now
    t0 = time.perf_counter()
    torch.set_num_threads(6)
    source_file = SOURCE / 'calibration_tensors.npz'
    z = np.load(source_file)
    ranges = json.loads((HERE / 'runs/native_eps_fix_20261014/qwen3/ranges_by_point.json').read_text())
    node = '/model/model/layers.27/mlp/down_proj/MatMul'
    q = Quant(work, 'cpu')
    verified = q.verify()
    assert all(v[0] == 0 for v in verified.values())
    truth = json.loads((TRUTH / 'stats.json').read_text())
    truthrows = {(r['slot'], r['format'], r['mask']): r for r in truth['conditions'] if r['data'] == 'calibration'}
    masks = {'all': np.ones((1, 128, 1), bool), 'P0': (np.arange(128) == 0)[None, :, None],
             'Rest': (np.arange(128) != 0)[None, :, None]}
    rows, values = [], {}
    reconstruction = 0.
    for slot in ('x', 'y'):
        t = z[slot]
        g = z['g' + slot].astype(np.float64)
        assert t.shape == g.shape and t.shape[:2] == (10, 128)
        n, targets = t.shape[0] * t.shape[1], t.shape[0] * TP
        G2 = (g * g).sum((0, 1))
        GS = (g.sum(1) ** 2).sum(0)
        for f in FORMATS:
            with torch.no_grad():
                tt = torch.from_numpy(t)
                e0 = (q.act(tt, f, ranges[node][slot]) - tt).numpy().astype(np.float64)
            for mask in MASKS:
                e = e0 * masks[mask]
                ge = g * e
                diagonal = float((ge * ge).sum() * .5 / targets)
                token = float((ge.sum(2) ** 2).sum() * .5 / targets)
                channel = float((ge.sum(1) ** 2).sum() * .5 / targets)
                window = float((ge.sum((1, 2)) ** 2).sum() * .5 / targets)
                mu = e.mean((0, 1))
                variance = float((G2 * ((e * e).mean((0, 1)) - mu * mu)).sum() / targets)
                bias = float((GS * mu * mu).sum() / targets)
                marginal = float((G2 * (e * e).mean((0, 1))).sum() * .5 / targets)
                mean_correction = float(((GS - G2) * mu * mu).sum() * .5 / targets)
                row = {'slot': slot, 'format': f, 'mask': mask,
                       'BC': variance + bias, 'BC_variance': variance, 'BC_bias': bias,
                       'BC_half_marginal_error_square': marginal,
                       'BC_half_mean_correction': mean_correction,
                       'diagonal_minus_BC_half_position_pairing': diagonal - marginal,
                       'diagonal_minus_BC_half_remove_mean_correction': -mean_correction,
                       'diagonal': diagonal, 'token': token, 'channel': channel, 'window': window,
                       'cross_channel_same_token': token - diagonal,
                       'cross_position_same_channel': channel - diagonal,
                       'cross_position_cross_channel': window - token - channel + diagonal,
                       'linear_g_dot_e': float(ge.sum() / targets)}
                row['decomposition_residual'] = window - sum(row[k] for k in (
                    'diagonal', 'cross_channel_same_token', 'cross_position_same_channel', 'cross_position_cross_channel'))
                reconstruction = max(reconstruction, abs(row['decomposition_residual']))
                assert abs(.5 * row['BC'] - marginal - mean_correction) < 1e-10
                assert abs(diagonal - .5 * row['BC'] - (diagonal - marginal) + mean_correction) < 1e-10
                tr = truthrows[(slot, f, {'all': 'all', 'P0': 'pos0', 'Rest': 'rest'}[mask])]
                row['observed_calibration_dNLL'] = tr['dL']
                row['observed_calibration_even_response'] = tr['b']
                row['reference_BC'] = tr['BC']
                rows.append(row)
                values[slot, f, mask] = row
    with open(out / 'scores.csv', 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = []
    for slot in ('x', 'y'):
        for bits, fi, fp in ((4, 'int4', 'p4e1'), (6, 'int6', 'p6e1'), (8, 'int8', 'p8e1')):
            for mask in MASKS:
                a, b = values[slot, fi, mask], values[slot, fp, mask]
                pairs.append({'slot': slot, 'bits': bits, 'mask': mask, **{
                    key + '_INT_minus_Posit': a[key] - b[key] for key in (
                        'BC', 'diagonal', 'token', 'channel', 'window', 'observed_calibration_dNLL',
                        'observed_calibration_even_response')}})
    with open(out / 'pairs.csv', 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    # The saved GPU tensors and CPU-QDQ reference differ very slightly at thresholds.
    rel = max(abs(r['BC'] - r['reference_BC']) / max(abs(r['reference_BC']), 1e-15) for r in rows)
    additive = max(abs(values[s, f, 'all'][k] - values[s, f, 'P0'][k] - values[s, f, 'Rest'][k])
                   for s in ('x', 'y') for f in FORMATS for k in ('token', 'diagonal'))
    controls = {'LUT_verification': verified, 'max_BC_relative_difference_vs_previous': rel,
                'geometry_decomposition_max_residual': reconstruction, 'position_additivity_residual': additive}
    assert reconstruction < 1e-10 and additive < 1e-10 and rel < .01
    write(out / 'controls.json', controls)
    write(out / 'provenance.json', {'time': now(), 'kind': 'posthoc on already-observed calibration only; not validation',
                                    'source_tensors': str(source_file), 'source_sha256': sha(source_file),
                                    'reference_stats_sha256': sha(TRUTH / 'stats.json'), 'script_sha256': sha(Path(__file__)),
                                    'torch': torch.__version__, 'numpy': np.__version__, 'seconds': time.perf_counter() - t0,
                                    'no_model_inference': True, 'no_new_loss_evaluation': True})
    md = ['# L27 Posit 評分：相關性聚合事後探索', '',
          '使用已保存的原 10 個 calibration 窗口 x/y/g；沒有新增模型 inference、沒有新文章 loss、沒有修改正式 criterion。', '',
          '定義 a_spc=g_spc e_spc。比較四種未中心化的經驗 Fisher 聚合，全部有 1/2，並除以 1270 targets。', '',
          '- diagonal：½Σ_spc a_spc²。', '- token：½Σ_sp(Σ_c a_spc)²，保留同一 token 內跨 channel 項。',
          '- channel：½Σ_sc(Σ_p a_spc)²，保留同一 channel 內跨位置項。',
          '- window：½Σ_s(Σ_pc a_spc)²，同時保留兩種相關性。', '',
          'BC 是原公式，與上述四種方法的中心化／偏差處理不同；本探索只用四種方法彼此比較聚合效應。它們都不是真正 Hessian。', '',
          'scores.csv 另列 BC/2 的邊際誤差平方項 M、均值修正 A：BC/2=M+A；diagonal−BC/2=(diagonal−M)−A。因此 BC 到 diagonal 同時改位置配對與均值修正，不能全部歸因於恢復相關性。', '',
          '本例在最後 Down 之後只有逐位置運算。window/channel 分數加入的跨位置項是經驗 Fisher 聚合項，不能解讀成實際存在的跨 token 損傷交互。', '',
          'INT−Posit 為正代表偏好 Posit。最後一欄是已看過的 calibration 實測差距；不把本探索稱為 held-out 改善。', '',
          '| slot | bits | mask | BC | diagonal | token | channel | window | 實測 ΔNLL 差 |',
          '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        md.append('| ' + ' | '.join([str(p['slot']), str(p['bits']), p['mask']] + [f'{p[k + "_INT_minus_Posit"]:.6f}'
                    for k in ('BC', 'diagonal', 'token', 'channel', 'window', 'observed_calibration_dNLL')]) + ' |')
    md += ['', '完整格式、mask 與分項見 scores.csv；代數控制見 controls.json。', '',
           '這些分數只用來選下一個假說；不得依本批已觀察 loss 擬合 Posit 專用係數或改正式選擇器。', '']
    (out / 'report.md').write_text('\n'.join(md))
    print(json.dumps({'output': str(out), 'controls': controls, 'input_all_pairs': [p for p in pairs if p['slot'] == 'x' and p['mask'] == 'all']}, indent=2))


if __name__ == '__main__':
    main()
