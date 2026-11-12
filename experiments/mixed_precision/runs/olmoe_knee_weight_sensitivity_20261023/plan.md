# 預先登記：OLMoE 正常 knee 的只刷新 weight sensitivity（W）

時間：2026-10-23T00:25:12+0800。以 plan.json 為準。

- 背景：維持原 BC criterion，只借用 GEMQ「在量化後工作狀態重新取得統計」的想法；不是 GEMQ 復現，不改 output-error criterion，也不改成 weight-only 量化。
- W：全部 3051 個投影的 weight 項用 q0_knee 的 sensitivity（原 per-channel 定義）乘 static ΔW²；activation 項整項沿用 static；預期表 = 歷史 C 的 weight 列 + static 的 activation 列。
- 實作：完整 q0 forward 與逐層 backward（固定路由、STE），只對 linear 輸出保留梯度；不做 activation 梯度統計、候選誤差、組表與快取讀取。
- 搜尋：W+A，在 knee 精確上限重新搜尋一次，weight 與 activation 格式都可重新分配。
- 控制：hash、weight 列 = 歷史 C、activation 列 = A、forward 一致、輸入梯度與 G_c 和 C 一致、numpy 重算、無 activation 統計工作、合法與上限、重複一致、端到端重現。
- 計時：B／C／W 各 3 次交錯（B1 C1 W1 C2 W2 B2 W3 B3 C3）；W 端到端 1 次；記錄 load average。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 5474 篇歷史文章）；FP32、A、B、C、W；seed 23001／23002／23011。
- 主要：W − A、W − C（校正上界 < 0 → W 較好；下界 > 0 → W 較差）、W − B（校正上界 < +0.0005 非劣；下界 > +0.0005 超過界線）；Bonferroni 98.33%。次要：B − A、C − A（95%）。
