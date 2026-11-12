#!/usr/bin/env python3
"""Post hoc CPU analysis of saved L26 results; no inference, fitting or new data."""
import argparse
import bisect
from collections import Counter
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'runs/qwen3_l26_suffix_bridge_20261027'
FORMATS = ['f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16']
SCORES = ['D', 'S_u', 'Su+K2', 'P256']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pick(scores, indices):
    # Fixed format order gives f16 priority for exact ties.
    return min(FORMATS, key=lambda fmt: float(scores[fmt][list(indices)].mean()))


def source_audit():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    parquet = HERE / 'gpt2_assets/data/validation-00000-of-00001.parquet'
    tokenizer = HERE / 'qwen3_assets/model/tokenizer.json'
    native = HERE / 'runs/native_eps_fix_20261014/qwen3/calib_inputs.npy'
    prepared = HERE / 'runs/qwen3_prepare_20261011/validation_tokens.npy'
    lines = pq.read_table(parquet)['text'].to_pylist()
    text = '\n\n'.join(lines)
    enc = Tokenizer.from_file(str(tokenizer)).encode(text, add_special_tokens=False)
    calib = np.load(native)
    assert np.array_equal(calib, np.load(prepared))
    assert np.array_equal(calib.ravel(), np.asarray(enc.ids[:calib.size]))
    heads = [i for i, line in enumerate(lines) if re.fullmatch(r'= [^=].* =', line.strip())]
    row_starts = [0]
    for line in lines[:-1]:
        row_starts.append(row_starts[-1] + len(line) + 2)
    end = enc.offsets[calib.size - 1][1]
    assert heads[0] == 1 and end < row_starts[heads[1]]
    segments = json.loads((SOURCE / 'segments.json').read_text())
    return {
        'calib_equals_prepare_validation': True, 'calib_equals_first_joined_WT2_validation_tokens': True,
        'calibration_tokens': int(calib.size), 'calibration_windows': int(calib.shape[0]),
        'calibration_article_count': 1, 'first_article': lines[heads[0]].strip().strip('= '),
        'calib_character_span_half_open': [enc.offsets[0][0], end],
        'calib_raw_rows_inclusive': [0, bisect.bisect_right(row_starts, end - 1) - 1],
        'next_article': lines[heads[1]].strip().strip('= '), 'next_article_raw_row': heads[1],
        'next_article_char_start': row_starts[heads[1]],
        'tokens_before_second_article': sum(start < row_starts[heads[1]] for start, stop in enc.offsets),
        'N_articles': len(segments), 'N_source_counts': dict(Counter(s['source'] for s in segments)),
        'hashes': {str(p.relative_to(HERE)): sha(p) for p in (parquet, tokenizer, native, prepared)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory is not empty; use a new directory.')
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'source_audit.json', source_audit())
    paths = [SOURCE / name for name in ('C_terms.npz', 'eval_ce_pos.npz', 'segments.json', 'freeze.json', 'data_freeze.json')]
    paths += [HERE / 'runs/native_eps_fix_20261014/qwen3/calib_inputs.npy', Path(__file__)]
    dump(out / 'analysis_scope.json', {
        'time': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        'status': 'post hoc: both calibration and N outcomes were already observed',
        'operations': 'read saved arrays; fixed-score deletion and exhaustive subset aggregation only',
        'not_done': 'no inference; no new article; no gradient/range/proxy recomputation; no threshold fitting',
        'unit': 'nat/token relative to same-background target activation f16; target weight int6',
        'dependence': '10 consecutive calibration windows are not 10 independent articles; no calibration bootstrap or CI',
        'positions': 'final CE positions, not source-perturbation positions',
        'source_hashes': {str(p.relative_to(HERE)): sha(p) for p in paths},
    })
    C, N = np.load(SOURCE / 'C_terms.npz'), np.load(SOURCE / 'eval_ce_pos.npz')
    assert C['F__int8__D'].shape == (10, 127)
    assert N['F__int8'].shape == (256, 127)
    groups = {'P0': np.array([0]), 'Rest': np.arange(1, 127), 'All': np.arange(127)}
    groups.update({f'Rest_q{i + 1}': ids for i, ids in enumerate(np.array_split(np.arange(1, 127), 4))})
    window_rows, deletion_rows, subset_rows = [], [], []
    summary = {}
    for bg in ('F', 'U'):
        terms, scores, npos, nd = {}, {}, {}, {}
        for fmt in FORMATS:
            key = f'{bg}__{fmt}__'
            terms[fmt] = {t: C[key + t] for t in ('D', 'S_u', 'Khat256', 'K2_actual', 'K2_aff', 'KL')}
            terms[fmt]['P256'] = terms[fmt]['S_u'] + terms[fmt]['Khat256']
            terms[fmt]['Su+K2'] = terms[fmt]['S_u'] + terms[fmt]['K2_actual']
            terms[fmt]['bias_P256_minus_D'] = terms[fmt]['P256'] - terms[fmt]['D']
            npos[fmt] = (N[f'{bg}__{fmt}'] - N[f'{bg}__f16']).reshape(128, 2, 127).mean(axis=1)
            nd[fmt] = npos[fmt].mean(axis=1)
            for j in range(10):
                row = {'background': bg, 'format': fmt, 'window_0_based': j}
                for term in SCORES + ['Khat256', 'bias_P256_minus_D']:
                    row[term] = float(terms[fmt][term][j].mean())
                for group, ids in groups.items():
                    row[f'D_{group}'] = float(terms[fmt]['D'][j, ids].sum() / 127)
                window_rows.append(row)
        for score in SCORES:
            scores[score] = {fmt: terms[fmt][score].mean(axis=1) for fmt in FORMATS}
        bg_summary = {'full_C_picks': {}, 'deletion': {}, 'subsets': {}, 'formats': {}}
        for score in SCORES:
            vals = scores[score]
            bg_summary['full_C_picks'][score] = pick(vals, range(10))
            choices, heldout_losses = [], []
            for j in range(10):
                idx = [i for i in range(10) if i != j]
                selected = pick(vals, idx)
                choices.append(selected)
                heldout_losses.append(float(terms[selected]['D'][j].mean()))
                deletion_rows.append({'background': bg, 'score': score, 'deleted_window_0_based': j,
                                      'pick': selected, 'int8_score_remaining': float(vals['int8'][idx].mean()),
                                      'heldout_true_D': heldout_losses[-1], 'already_seen_N_mean_D': float(nd[selected].mean())})
            bg_summary['deletion'][score] = {'counts': dict(Counter(choices)), 'picks_by_deleted_window': choices,
                                            'heldout_D_mean_descriptive': float(np.mean(heldout_losses))}
            bg_summary['subsets'][score] = {}
            for size in range(1, 11):
                counts = Counter()
                for idx in itertools.combinations(range(10), size):
                    selected = pick(vals, idx)
                    counts[selected] += 1
                    subset_rows.append({'background': bg, 'score': score, 'subset_size': size,
                                        'window_indices': ' '.join(map(str, idx)), 'pick': selected,
                                        'already_seen_N_mean_D': float(nd[selected].mean())})
                bg_summary['subsets'][score][size] = {'count': sum(counts.values()), 'picks': dict(counts)}
        for fmt in FORMATS:
            t = terms[fmt]
            d = t['D'].mean(axis=1)
            bias = (t['P256'] - t['D']).mean(axis=1)
            fs = {'C_D_mean': float(d.mean()), 'C_D_window_sd_ddof1': float(d.std(ddof=1)),
                  'C_windows_positive': int((d > 0).sum()), 'C_windows_negative': int((d < 0).sum()),
                  'N_D_mean': float(nd[fmt].mean()), 'N_articles_positive': int((nd[fmt] > 0).sum()),
                  'N_articles_negative': int((nd[fmt] < 0).sum()),
                  'N_CE_positions_positive_mean_D': int((npos[fmt].mean(axis=0) > 0).sum()),
                  'bias_P256_minus_D': {'mean': float(bias.mean()), 'min': float(bias.min()), 'max': float(bias.max()),
                                       'windows_negative': int((bias < 0).sum())},
                  'tail_omission': float((t['K2_aff'] - t['Khat256']).mean()),
                  'KL_minus_K2': float((t['KL'] - t['K2_actual']).mean()), 'groups': {}}
            for group, ids in groups.items():
                cval, nval = float(t['D'][:, ids].sum(axis=1).mean() / 127), float(npos[fmt][:, ids].sum(axis=1).mean() / 127)
                fs['groups'][group] = {'C': cval, 'N': nval, 'shift_N_minus_C': nval - cval,
                                       'CE_position_indices': ids.tolist()}
            assert abs(fs['groups']['P0']['C'] + fs['groups']['Rest']['C'] - fs['C_D_mean']) < 1e-12
            assert abs(fs['groups']['P0']['N'] + fs['groups']['Rest']['N'] - fs['N_D_mean']) < 1e-12
            bg_summary['formats'][fmt] = fs
        summary[bg] = bg_summary
    dump(out / 'summary.json', summary)
    csv_write(out / 'calibration_windows.csv', window_rows)
    csv_write(out / 'leave_one_window_out.csv', deletion_rows)
    csv_write(out / 'all_fixed_score_subsets.csv', subset_rows)
    fs = summary['F']['formats']['int8']
    print(json.dumps({'out': str(out), 'F_full_picks': summary['F']['full_C_picks'],
                      'F_deletion': summary['F']['deletion'],
                      'F_int8': {k: v for k, v in fs.items() if k != 'groups'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
