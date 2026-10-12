#!/usr/bin/env python3
"""Summarize frozen-search compiled evaluations, with paired uncertainty."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/native_search_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


def paired(work, evaluation, method, baseline='htq'):
    def arrays(m):
        actual = evaluation[m].get('artifact_alias', m)
        return np.load(work / actual / 'per_item.npy'), np.load(work / actual / 'correct.npy')
    a, ac = arrays(method)
    b, bc = arrays(baseline)
    assert a.shape == b.shape and ac.shape == bc.shape
    rng = np.random.default_rng(20261010)
    means, accuracy, ppl = [], [], []
    for _ in range(50):
        idx = rng.integers(0, len(a), size=(100, len(a)))
        means.extend((a[idx]-b[idx]).mean(axis=1))
        accuracy.extend((ac[idx]-bc[idx]).mean(axis=1))
        if work.name == 'gpt2':
            ppl.extend(np.exp(a[idx].mean(axis=1))-np.exp(b[idx].mean(axis=1)))
    result = dict(loss_delta=float((a-b).mean()), loss_delta_ci95=np.quantile(means, [.025, .975]).tolist(),
                  accuracy_delta_pp=100*float((ac-bc).mean()),
                  accuracy_delta_pp_ci95=(100*np.quantile(accuracy, [.025, .975])).tolist(),
                  n=len(a), bootstrap_replicates=5000, bootstrap_unit='window' if work.name=='gpt2' else 'example')
    if ppl:
        result.update(ppl_delta=float(np.exp(a.mean())-np.exp(b.mean())), ppl_delta_ci95=np.quantile(ppl,[.025,.975]).tolist())
    else:
        wins, losses = int(np.sum(ac > bc)), int(np.sum(ac < bc))
        result.update(discordant_wins=wins, discordant_losses=losses,
                      mcnemar_exact_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out_dir', type=Path)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    protocol = json.loads((out / 'protocol.json').read_text())
    summary = {}
    report = ['# MILP 局部補搜 vs HTQ 投影剪枝：受控比較', '',
              '搜尋日期：2026-10-10；模型評估：2026-10-10～11；獨立核驗：2026-10-11（Asia/Taipei）。搜尋使用同一份 native BC 表，最後以 ONNX-MLIR 編譯模型評估。', '',
              '## 固定協議', '',
              '- 五模型、每層 11×11 格式、相同可調節點與資源估算。HTQ 使用專案既有移植版：1/20 投影剪枝、beam=2000，criterion 改用相同 BC，以隔離搜尋差異。',
              '- MILP：17 個共同預算、兩輪 knee 細化（最多 6 次）、原 coarse knee 附近 8 個獨立 size/BOPs 預算。候選最後做三維非支配篩選。',
              '- 兩邊加入相同的廉價／最低 BC 解析端點，以這些端點固定各軸正規化範圍；最終都取三軸等權 Euclidean ideal-distance 最小者。不是用各自候選範圍重新定義尺度，也不使用測試 accuracy 選點。',
              '- fresh process、同一 CPU affinity、搜尋順序交替，重複三次。耗時含候選搜尋、重建與最終選點，不含共同 BC 校準、載入、編譯及測試。',
              '- 最終共同搜尋上限 120 秒。先前 60 秒 pilot 的 EfficientNet/BERT HTQ 逾時紀錄保留並另行重跑；其他早期完成紀錄的 60 秒上限未觸發。上限調整及小 beam 實驗均在 accuracy 評估之前決定。',
              '- 事先固定評估 repeat 0。另求一次 MILP，使用 HTQ repeat-0 配置的實際 size/BOPs 作上限，作為資源一致診斷；這次診斷不計入原搜尋時間。',
              '- CNN：相同 1000 張 held-out ImageNet 圖；BERT：872 個 SST-2 樣本；GPT-2：32 個 128-token windows，以 PPL 為主指標。所有配置使用相同 O3 編譯與 per-channel weights，常數存外部檔案。',
              '- 95% CI 是相同樣本配對 bootstrap，5000 次、固定 seed；GPT-2 以 window 為單位。區間未做跨模型多重比較校正，也不涵蓋校準集或搜尋選點的不確定性。', '',
              '## 搜尋時間與重複性', '',
              '| 模型 | MILP 中位秒數 [min,max] | HTQ 中位秒數 [min,max] | HTQ/MILP | MILP/HTQ 不同選點數 |',
              '|---|---:|---:|---:|---:|']
    for tag in protocol['models']:
        work = out / tag
        evaluation = json.loads((work / 'evaluation.json').read_text())
        search = {m: [json.loads((work / f'{m}_r{i}.json').read_text()) for i in range(protocol['repeats'])]
                  for m in ('milp_local', 'htq', 'htq_fast')}
        assert all(r['completed'] for runs in search.values() for r in runs)
        timing = {m: dict(median=float(np.median([r['seconds'] for r in runs])),
                          min=min(r['seconds'] for r in runs), max=max(r['seconds'] for r in runs),
                          unique_picks=len({r['selected']['config_hash'] for r in runs}),
                          scores=[r['selected']['selection_score'] for r in runs],
                          peak_rss_mb=max(r['peak_rss_mb'] for r in runs)) for m, runs in search.items()}
        ratio = timing['htq']['median']/timing['milp_local']['median']
        comparison = {m: paired(work, evaluation, m) for m in ('milp_local', 'milp_at_htq_resources', 'htq_fast')}
        comparison['milp_vs_htq_fast'] = paired(work, evaluation, 'milp_local', 'htq_fast')
        summary[tag] = dict(timing=timing, htq_over_milp_seconds=ratio, evaluation=evaluation, paired=comparison)
        m, h = timing['milp_local'], timing['htq']
        report.append(f"| {tag} | {m['median']:.2f} [{m['min']:.2f},{m['max']:.2f}] | {h['median']:.2f} [{h['min']:.2f},{h['max']:.2f}] | {ratio:.2f}× | {m['unique_picks']}/{h['unique_picks']} |")
    report += ['', '## 最終模型：事先固定的自動選點', '',
               '| 模型 | MILP accuracy / PPL | HTQ accuracy / PPL | 差值與配對 95% CI | MILP size/BOPs | HTQ size/BOPs |',
               '|---|---:|---:|---|---:|---:|']
    for tag, d in summary.items():
        m, h = d['evaluation']['milp_local'], d['evaluation']['htq']
        c = d['paired']['milp_local']
        if tag == 'gpt2':
            vm, vh = m['metrics']['ppl'], h['metrics']['ppl']
            delta, ci, unit = c['ppl_delta'], c['ppl_delta_ci95'], 'PPL'
        else:
            key = 'accuracy' if tag=='bert' else 'top1'
            vm, vh = 100*m['metrics'][key], 100*h['metrics'][key]
            delta, ci, unit = c['accuracy_delta_pp'], c['accuracy_delta_pp_ci95'], 'pp'
        sm, sh = m['selected'], h['selected']
        report.append(f"| {tag} | {vm:.3f} | {vh:.3f} | {delta:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {unit} | {sm['size']:.4f}/{sm['bops']:.4f} | {sh['size']:.4f}/{sh['bops']:.4f} |")
    report += ['', 'accuracy 越高越好、GPT-2 PPL 越低越好。差值一律 MILP−HTQ。資源值相對可調節點的 all-int8；不是整個模型磁碟檔案大小。', '',
               '## 相同 HTQ 資源上限的診斷', '',
               '| 模型 | matched MILP accuracy / PPL | HTQ accuracy / PPL | 差值與配對 95% CI | matched BC / HTQ BC | solver status |',
               '|---|---:|---:|---|---:|---:|']
    for tag, d in summary.items():
        m, h = d['evaluation']['milp_at_htq_resources'], d['evaluation']['htq']
        c = d['paired']['milp_at_htq_resources']
        if tag == 'gpt2':
            vm, vh = m['metrics']['ppl'], h['metrics']['ppl']
            delta, ci, unit = c['ppl_delta'], c['ppl_delta_ci95'], 'PPL'
        else:
            key = 'accuracy' if tag=='bert' else 'top1'
            vm, vh = 100*m['metrics'][key], 100*h['metrics'][key]
            delta, ci, unit = c['accuracy_delta_pp'], c['accuracy_delta_pp_ci95'], 'pp'
        ratio = m['selected']['objective']/h['selected']['objective']
        report.append(f"| {tag} | {vm:.3f} | {vh:.3f} | {delta:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {unit} | {ratio:.4f} | {m['selected']['solver_status']} |")
    report += ['', '## HTQ 較小 beam 的敏感度實驗', '',
               'beam=200 在查看任何 held-out 結果之前加入，使用相同三次計時及同一固定選點政策。', '',
               '| 模型 | beam=200 搜尋中位秒數 | accuracy / PPL | size/BOPs | BC |',
               '|---|---:|---:|---:|---:|']
    for tag, d in summary.items():
        r=d['evaluation']['htq_fast']; metric='ppl' if tag=='gpt2' else 'accuracy' if tag=='bert' else 'top1'
        value=r['metrics'][metric]*(1 if tag=='gpt2' else 100)
        report.append(f"| {tag} | {d['timing']['htq_fast']['median']:.2f} | {value:.3f} | {r['selected']['size']:.4f}/{r['selected']['bops']:.4f} | {r['selected']['objective']:.6g} |")
    validation_path = out / 'independent_validation.json'
    execution_path = out / 'execution_verification.json'
    if validation_path.exists() and execution_path.exists():
        validated = json.loads(validation_path.read_text())
        executed = json.loads(execution_path.read_text())
        assert len(validated) == 20 and all(r['metrics_recomputed'] and r['finite'] for r in validated)
        assert len(executed) == 20 and all(r['passed'] for r in executed)
        report += ['', '## 獨立核驗', '',
                   '- 20 組配置的全部 raw logits 均有限；重新計算 loss、accuracy/PPL、per-item correctness，與保存結果一致。',
                   '- 20 個 .so 各抽第 0、N//2、N−1 筆重新執行，共 60 次樣本推論，輸出與保存 logits 逐值一致（本次最大絕對差 0）。',
                   '- 所有評估配置 hash 與事先固定的選點一致，編譯 warning 為零；五個資源一致 MILP 都是 status=0 並符合兩項上限。',
                   '- 詳見 `independent_validation.json`、`execution_verification.json` 與各配置的驗證輸出。']
    report += ['', '## 解讀限制', '',
               '- 資源一致診斷使用不超過 HTQ 的兩資源上限，不保證兩邊實際用量完全一樣。這是額外求解，不能拿它的 accuracy 冒充原自動搜尋結果。',
               '- HTQ 原移植版使用 Python object set 合併候選，會影響後續 tie 順序；三次不同選點數是本次觀測，不是多 seed accuracy 實驗。',
               '- 只比較這份 HTQ Python 移植版及 beam=200/2000，不代表所有 HTQ 實作、所有 beam 或完整原論文 pipeline。兩方法都使用 BC，未比較原 HTQ criterion。',
               '- 固定等權理想點選點是一項政策選擇；BC 是 accuracy 的代理。測試結果只適用於本次資料與政策，不能推論 BC 改善必然帶來 accuracy 改善。',
               '- 模型採 FP32 計算加 QDQ，BOPs/size 是估算值；本報告未宣稱實際低位元硬體延遲、能耗或壓縮比例。', '',
               '## 產物', '',
               '- `summary.json`：完整計時、資源、metrics、配對區間。',
               '- 各模型 `*_r*.json`：候選、選點、耗時、狀態；`evaluation_choices.json`：事前鎖定的評估配置。',
               '- 各配置資料夾：可執行 .so、常數檔、logits、per-item loss/correctness、編譯與評估紀錄。',
               '- `source/`：本次使用腳本快照；`protocol.json`：比較協議。', '']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (out / 'report.md').write_text('\n'.join(report))
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(summary)); width=.25
    for i,(method,label) in enumerate([('milp_local','MILP + local'),('htq','HTQ beam 2000'),('htq_fast','HTQ beam 200')]):
        axes[0].bar(x+(i-1)*width,[d['timing'][method]['median'] for d in summary.values()],width,label=label)
    axes[0].set(xticks=x,xticklabels=list(summary),ylabel='Search seconds (median)',title='Same CPU / tables / selection policy')
    axes[0].tick_params(axis='x',rotation=20);axes[0].legend()
    ratios=[d['evaluation']['milp_local']['selected']['objective']/d['evaluation']['htq']['selected']['objective'] for d in summary.values()]
    axes[1].bar(x,ratios);axes[1].axhline(1,color='black',linestyle='--')
    axes[1].set(xticks=x,xticklabels=list(summary),ylabel='MILP BC / HTQ BC',title='Automatic picks (resources may differ)')
    axes[1].tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(out/'search_comparison.png',dpi=160)
    for tag,d in summary.items():
        print(tag, 'speed ratio',round(d['htq_over_milp_seconds'],2),
              'auto', d['evaluation']['milp_local']['metrics'], 'HTQ',d['evaluation']['htq']['metrics'],flush=True)


if __name__ == '__main__':
    main()
