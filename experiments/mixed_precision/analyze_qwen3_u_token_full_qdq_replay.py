#!/usr/bin/env python3
"""Posthoc replay of V_old+U_token against EXISTING full Down-input QDQ losses.

No new model inference, criterion edit, selector fitting, or held-out confirmation.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE / 'runs/qwen3_down_spike_diag_20261023'
NEW = HERE / 'runs/qwen3_mean_u_token_validation_20261025'
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1')
PAIRS = tuple((f'int{b}', f'p{b}e1') for b in (4, 6, 8))


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def writecsv(path, data):
    with open(path, 'w') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit('use a new output directory')
    out.mkdir(parents=True)
    start = time.perf_counter()
    inputs = [OLD / 'scores.csv', OLD / 'per_article_losses.csv', OLD / 'ranking_pairs.csv',
              OLD / 'plan.json', NEW / 'calibration_scores.csv', NEW / 'plan.json']
    write(out / 'analysis_plan.json', {
        'kind': 'posthoc exploratory replay; all loss data previously observed',
        'scope': 'all-FP32 model except one Down INPUT full QDQ; not x+y QDQ or full configuration',
        'before_this_script': 'score-only inspection already found 0/84 same-bit argmin changes; '
                              'old-U backend max relative difference .0028985 and 0 same-bit bridge changes',
        'methods': {'old': 'V_old_CPU+U_old_CPU (=T_old)',
                    'bridge': 'V_old_CPU+U_old_new_GPU', 'new': 'V_old_CPU+U_token_new_GPU'},
        'boundaries': 'no first-order term, coefficients, thresholds, range changes, new losses, or LP search',
        'primary': 'all 84 same-bit pairs, raw argmin; 2% margin flags and fixed INT fallback also shown',
        'secondary': '336 cross-bit comparisons, ranking only; unequal resource costs, no deployment benefit claim',
        'uncertainty': 'reuse original source pair CI/decisive labels; primary gain CI exactly [0,0] if all picks identical',
        'inputs': {str(p.relative_to(HERE)): sha(p) for p in inputs}, 'script_sha256': sha(Path(__file__))})
    a = {(int(r['layer']), r['format']): r for r in rows(OLD / 'scores.csv')}
    b = {(int(r['layer']), r['format']): r for r in rows(NEW / 'calibration_scores.csv')}
    keys = {(l, f) for l in range(28) for f in FORMATS}
    assert set(a) == set(b) == keys
    old, bridge, new = {}, {}, {}
    score_rows, identity = [], 0.
    for k in sorted(keys):
        V, U = float(a[k]['V_old']), float(a[k]['U'])
        Ug, Ut = float(b[k]['U_old']), float(b[k]['U_token'])
        identity = max(identity, abs(V + U - float(a[k]['T_old'])) / max(abs(V + U), 1e-30))
        old[k], bridge[k], new[k] = V + U, V + Ug, V + Ut
        assert abs(float(b[k]['S_new']) * 2540 - Ut) < 1e-9 * max(abs(Ut), 1.)
        score_rows.append({'layer': k[0], 'format': k[1], 'V_old': V, 'U_old_source': U,
                           'U_old_new_backend': Ug, 'U_token': Ut, 'old_total': old[k],
                           'bridge_total': bridge[k], 'new_total': new[k],
                           'U_backend_relative_difference': abs(Ug - U) / max(abs(U), 1e-30)})
    assert identity < 1e-12
    max_backend = max(r['U_backend_relative_difference'] for r in score_rows)
    assert max_backend < .01
    writecsv(out / 'scores.csv', score_rows)

    losses = rows(OLD / 'per_article_losses.csv')
    articles = sorted({int(r['article']) for r in losses})
    assert articles == list(range(128)) and len(losses) == 28 * 6 * 128
    L = {(int(r['layer']), r['format'], int(r['article'])): float(r['dnll']) for r in losses}
    assert len(L) == len(losses) and np.isfinite(list(L.values())).all()
    pair_source = {(int(r['layer']), frozenset((r['f1'], r['f2']))): r for r in rows(OLD / 'ranking_pairs.csv')}

    def pick(C, l, f1, f2):
        return min((f1, f2), key=lambda f: C[l, f])

    def margin(C, l, f1, f2):
        c1, c2 = C[l, f1], C[l, f2]
        return abs(c1 - c2) / max(abs(c1), abs(c2), 1e-30)

    primary, secondary, article_selected = [], [], []
    for l in range(28):
        for f1, f2 in itertools.combinations(FORMATS, 2):
            same = (f1, f2) in PAIRS
            src = pair_source[l, frozenset((f1, f2))]
            lo, hi = float(src['ci_lo']), float(src['ci_hi'])
            if src['f1'] != f1:
                lo, hi = -hi, -lo
            D = np.array([L[l, f1, k] - L[l, f2, k] for k in articles])
            decisive = lo > 0 or hi < 0
            better = f1 if D.mean() < 0 else f2
            po, pb, pn = (pick(C, l, f1, f2) for C in (old, bridge, new))
            ro = 'right' if po == better else 'wrong'
            rn = 'right' if pn == better else 'wrong'
            if not decisive:
                ro = rn = 'undetermined'
            gain = np.array([L[l, pn, k] - L[l, po, k] for k in articles])
            row = {'layer': l, 'f1': f1, 'f2': f2, 'old_pick': po, 'bridge_pick': pb, 'new_pick': pn,
                   'changed': po != pn, 'backend_bridge_changed': po != pb,
                   'old_relative_margin': margin(old, l, f1, f2),
                   'new_relative_margin': margin(new, l, f1, f2),
                   'old_2percent_tie': margin(old, l, f1, f2) <= .02,
                   'new_2percent_tie': margin(new, l, f1, f2) <= .02,
                   'source_D_int_or_f1_minus_f2': float(D.mean()), 'source_ci_lo': lo, 'source_ci_hi': hi,
                   'source_decisive': decisive, 'old_status': ro, 'new_status': rn,
                   'source_mean_new_minus_old': float(gain.mean())}
            (primary if same else secondary).append(row)
            if same:
                for k in articles:
                    article_selected.append({'layer': l, 'f1': f1, 'f2': f2, 'article': k,
                                              'old_pick': po, 'new_pick': pn,
                                              'old_dnll': L[l, po, k], 'new_dnll': L[l, pn, k],
                                              'new_minus_old': L[l, pn, k] - L[l, po, k]})
    assert len(primary) == 84 and len(secondary) == 336
    writecsv(out / 'same_bit_pairs.csv', primary)
    writecsv(out / 'cross_bit_pairs_exploratory.csv', secondary)
    writecsv(out / 'per_article_selected_losses.csv', article_selected)
    Rold = np.array([np.mean([r['old_dnll'] for r in article_selected if r['article'] == k]) for k in articles])
    Rnew = np.array([np.mean([r['new_dnll'] for r in article_selected if r['article'] == k]) for k in articles])
    gain = Rnew - Rold
    identical = bool(np.array_equal(Rnew, Rold))
    info = {'primary_pairs': 84, 'source_articles': 128, 'source_windows_per_article': 3,
            'same_bit_choice_changes': sum(r['changed'] for r in primary),
            'same_bit_bridge_changes': sum(r['backend_bridge_changed'] for r in primary),
            'old_2percent_ties': sum(r['old_2percent_tie'] for r in primary),
            'new_2percent_ties': sum(r['new_2percent_tie'] for r in primary),
            'source_decisive': sum(r['source_decisive'] for r in primary),
            'old_right': sum(r['old_status'] == 'right' for r in primary),
            'new_right': sum(r['new_status'] == 'right' for r in primary),
            'remaining_errors': [f'L{r["layer"]} {r["f1"]}-{r["f2"]}' for r in primary if r['new_status'] == 'wrong'],
            'R_old': float(Rold.mean()), 'R_new': float(Rnew.mean()), 'G_new_minus_old': float(gain.mean()),
            'article_gain_identically_zero': identical,
            'G_ci95': [0., 0.] if identical else None,
            'cross_bit_changes': [r for r in secondary if r['changed']],
            'cross_bit_bridge_changes': [r for r in secondary if r['backend_bridge_changed']],
            'T_old_identity_max_relative': identity, 'U_backend_max_relative': max_backend,
            'inference_runs': 0, 'kind': 'exploratory replay of previously observed loss data'}
    assert info['same_bit_choice_changes'] == 0 and identical
    write(out / 'summary.json', info)
    write(out / 'timing.json', {'seconds': time.perf_counter() - start, 'inference_runs': 0})
    md = ['# 原變異項＋U_token：完整 Down-input QDQ 既有資料回放', '',
          '這是已觀察資料上的探索性回放。沒有新 inference、沒有新文章、沒有修改正式 criterion。', '',
          'full QDQ 指每次一層的完整 Down input 量化誤差 e；不是只有平均分量，也不是 input/output 一起量化或完整模型配置。', '',
          '方法：原 C=V_old+U_old；新 C=V_old+U_token。全部是未正規化分數，不加 F、不調係數。', '',
          '原 V/完整 QDQ 損傷來自 qwen3_down_spike_diag_20261023；U_token 來自 qwen3_mean_u_token_validation_20261025 的原 calibration。', '',
          '| 項目 | 結果 |', '|---|---|',
          f'| 同位元選擇改變 | {info["same_bit_choice_changes"]}/84 |',
          f'| 舊資料可判定配對 | {info["source_decisive"]}/84 |',
          f'| 原／新排對 | {info["old_right"]}/{info["source_decisive"]} → {info["new_right"]}/{info["source_decisive"]} |',
          f'| 兩方法所選損傷平均 | {info["R_old"]:.9f} → {info["R_new"]:.9f} nat/token |',
          f'| 所選損傷差 new−old | {info["G_new_minus_old"]:.9f}，逐文章恰為零 |',
          f'| 2% margin tie（原／新） | {info["old_2percent_ties"]}/{info["new_2percent_ties"]} |', '',
          '仍存在的原始明確誤排：' + '、'.join(info['remaining_errors']) + '。', '',
          '因選擇完全相同，對這84個凍結的同位元選擇，換新文章也不會產生兩方法的選擇收益差。無需為此重跑GPU。', '',
          '## 跨後端控制', '',
          f'舊／新版 U_old 最大相對差 {max_backend:.6%}；T_old=V_old+U_old 最大相對殘差 {identity:.3g}。',
          f'橋接 V_old+新版U_old 的同位元選擇改變 {info["same_bit_bridge_changes"]}/84；用來區分方法更換與後端數值差異。', '',
          '## 跨位元描述性結果', '',
          f'336 個跨位元對比中有 {len(info["cross_bit_changes"])} 個排序改變；橋接控制有 {len(info["cross_bit_bridge_changes"])} 個。完整值見 cross_bit_pairs_exploratory.csv。',
          '資源不等，不能將跨位元選擇當成同成本品質收益；這只是代理排序診斷。', '',
          '## 可下的結論與限制', '',
          '- U_token 的 mean-only 改善沒有改變此批完整 Down-input QDQ 的同位元選擇。',
          '- 這不代表 U 的機制分析無效，也不代表所有 Posit 候選都不準；原變異項仍可能主導總分。',
          '- 不是獨立泛化驗證：舊 loss 已被反覆觀察；原 CI/可判定標記直接沿用來源，不假裝是新盲測。',
          '- 同位元選擇不變，不代表 LP/MILP 配置不變：絕對分數、跨位元及跨層資源取捨仍可能改變，尚未搜尋／驗證。',
          '- 不直接整合 pipeline，也不宣稱模型PPL收益。後續若檢查配置，需要完整 x/y 成本表與資源限制，不能把本 x-only 表直接當 native C。', '',
          '產物：analysis_plan.json、scores.csv、same_bit_pairs.csv、cross_bit_pairs_exploratory.csv、per_article_selected_losses.csv、summary.json、timing.json。', '']
    (out / 'report.md').write_text('\n'.join(md))
    print(json.dumps({k: v for k, v in info.items() if k not in ('cross_bit_changes', 'cross_bit_bridge_changes')}, indent=2, ensure_ascii=False))
    print('cross-bit changed:', len(info['cross_bit_changes']))


if __name__ == '__main__':
    main()
