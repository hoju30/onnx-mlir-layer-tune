# Qwen3-0.6B：三種 Gate＋Up 排序的既有資料重分析

日期：2026-10-15。這是事後診斷，沒有執行新的模型 forward/backward，也没有修改 criterion 或搜尋 pipeline。

先前 block criterion 的 additive 是誤差評分函數 phi(e_gate)+phi(e_up)，並非實測 task-loss 損傷相加。此處補上三種直接比較。

資料：原 qwen3_fd_validation_20261015，160 個聯合候選、5 層 × activation-only/weight-only，Down 與其餘節點 FP32。BC 使用 calibration 0–9；實測單獨與聯合損傷都使用同一批 validation 90–119 的30個窗口。

| 排序依據 | 10組平均 Spearman，對實測聯合損傷 | 穩定同資源配對排對 | 穩定格式互換配對排對 |
|---|---:|---:|---:|
| BC Gate＋Up 分數和 | 0.672353 | 130/164 | 19/32 |
| 實測 Gate-only＋Up-only 損傷和 | 0.974706 | 158/164 | 32/32 |
| 實測 Gate＋Up 聯合損傷 | 排序真值 | 排序真值 | 排序真值 |

穩定配對沿用原 pairs.csv：同層、同模式、同 weight-size/BOPs，實測聯合 summed CE 差 >1，且原逐窗口 bootstrap 的順序保持率 ≥95%。只保留兩個候選都是聯合配置的配對。Spearman 先在每個層×模式的16個聯合候選內計算，再平均10組；沒有混入單獨候選以提高相關。

BC排錯34個；替換為同窗口實測單點損傷和，修正30個、仍錯4個，另外弄錯2個原本BC正確的配對。

判讀：在這批局部兩分支候選中，真實單點損傷相加已能很好地保留聯合損傷的排序，BC單點代理／資料轉移是較直接的瓶頸；非加成性仍會使部分配對排錯。

限制：第二列是使用評估資料真實損傷的 oracle，不能當可部署的低成本 criterion。第一列到第二列的差距同時包含代理評分與校準資料轉移，不能單獨歸因於公式。結果不代表全模型多層配置的主要誤差比例，不能把修正配對數當因果百分比，也沒有新增獨立驗證或宣稱最終PPL改善。

重算依據與逐候選／配對值：analysis.json、source_manifest.json、candidates.csv、pairs.csv。每個來源檔案的SHA256已保存。
