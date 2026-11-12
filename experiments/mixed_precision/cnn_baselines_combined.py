#!/usr/bin/env python3
"""Three-model summary of the CNN baseline runs (ResNet18, MobileNetV2, EfficientNet-B0).

Reads only the finished runs' summary.json (and ResNet18's followup_matched/summary.json); writes
runs/cnn_baselines_20261029_combined/{combined.json, combined_tables.md}. No new measurement.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {'ResNet18': HERE / 'runs/cnn_baselines_20261029',
        'MobileNetV2': HERE / 'runs/cnn_baselines_20261029_mobilenetv2',
        'EfficientNet-B0': HERE / 'runs/cnn_baselines_20261029_efficientnet_b0'}
OUT = HERE / 'runs/cnn_baselines_20261029_combined'
METHODS = ('BC', 'SQNR', 'HTQ', 'HAWQ', 'InfoQ')
POINTS = ('loose', 'knee', 'aggressive')


def ci(v, nd=2):
    return f'[{v[0]:+.{nd}f}, {v[1]:+.{nd}f}]'


def main():
    OUT.mkdir(exist_ok=True)
    comb, md = {}, []
    for label, run in RUNS.items():
        p = run / 'summary.json'
        if not p.exists():
            comb[label] = None
            continue
        s = json.loads(p.read_text())
        tc = s.get('table_matched_R')
        if tc is None and (run / 'followup_matched/summary.json').exists():
            tc = json.loads((run / 'followup_matched/summary.json').read_text())['rows']
            tc_src = 'followup_matched (post hoc, pre-registered)'
        else:
            tc_src = 'pre-registered C' if tc else None
        st = s['score_timing']
        bc_t = st['BC']['median_s']
        comb[label] = {'run': str(run.relative_to(HERE)), 'n_eval_images': s['n_eval_images'], 'fp32': s['fp32'],
                       'int8_ref': s['int8_ref'], 'caps': s['caps'], 'table1': s['table1'], 'table2': s['table2'],
                       'matched_R': tc, 'matched_R_source': tc_src,
                       'cost': {m: {'reps': v['reps'], 'median_s': v['median_s'], 'range_s': v['range_s'],
                                    'ratio_vs_BC': v['median_s'] / bc_t} for m, v in st.items() if v['reps']},
                       'methods_run': [m for m in METHODS if st.get(m, {}).get('reps')]}
    done = {k: v for k, v in comb.items() if v}

    md.append('## Table 1: same caps (method - BC, top-1 pp, 95% bootstrap CI, Holm-adjusted McNemar)\n')
    md.append('| model | point | R cap | ' + ' | '.join(m for m in METHODS[1:]) + ' |')
    md.append('|---|---|---|' + '---|' * (len(METHODS) - 1))
    for label, c in done.items():
        for p in POINTS:
            rows = {r['method']: r for r in c['table1'] if r['point'] == p}
            cells = []
            for m in METHODS[1:]:
                r = rows.get(m)
                if r is None:
                    cells.append('not run')
                elif r['same_config_as_BC']:
                    cells.append('same config')
                else:
                    cells.append(f"{r['d_top1_pp_vs_BC']:+.2f} {ci(r['d_top1_ci_pp'])} {r['verdict']}")
            md.append(f"| {label} | {p} | {rows['BC']['cap_R']:.4f} | " + ' | '.join(cells) + ' |')

    md.append('\n## Table 2: common auto shell (top-1 %, R)\n')
    md.append('| model | ' + ' | '.join(METHODS) + ' |')
    md.append('|---|' + '---|' * len(METHODS))
    for label, c in done.items():
        rows = {r['method']: r for r in c['table2']}
        md.append(f'| {label} | ' + ' | '.join(f"{100 * rows[m]['top1']:.2f} (R {rows[m]['R']:.3f})" if m in rows
                                                else 'not run' for m in METHODS) + ' |')

    md.append('\n## Table C: auto pick - BC re-searched at the same R (top-1 pp)\n')
    md.append('| model | method | R | method top-1 | BC top-1 | diff [CI] | verdict | source |')
    md.append('|---|---|---|---|---|---|---|---|')
    for label, c in done.items():
        for r in c['matched_R'] or []:
            same = ' (same config)' if r.get('same_config') else ''
            md.append(f"| {label} | {r['method']} | {r['R_cap']:.4f} | {100 * r['method_top1']:.2f} | "
                      f"{100 * r['bc_top1']:.2f} | {r['d_top1_pp']:+.2f} {ci(r['d_top1_ci_pp'])}{same} | "
                      f"{r['verdict']} | {c['matched_R_source']} |")

    md.append('\n## Table 3: offline scoring cost (median table time, s; ratio vs BC; repetitions)\n')
    md.append('| model | ' + ' | '.join(METHODS) + ' |')
    md.append('|---|' + '---|' * len(METHODS))
    for label, c in done.items():
        md.append(f'| {label} | ' + ' | '.join(f"{c['cost'][m]['median_s']:.1f} ({c['cost'][m]['ratio_vs_BC']:.2f}x, "
                                                f"n={c['cost'][m]['reps']})" if m in c['cost'] else 'not run'
                                                for m in METHODS) + ' |')
    (OUT / 'combined.json').write_text(json.dumps(comb, indent=1) + '\n')
    (OUT / 'combined_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('missing:', [k for k, v in comb.items() if not v])


if __name__ == '__main__':
    main()
