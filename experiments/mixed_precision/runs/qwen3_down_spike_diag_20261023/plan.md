# 計畫：Qwen3-0.6B Down-input criterion 跨模型單點診斷

時間：2026-10-23T23:45:45+0800。以 plan.json 為準；寫入後不改分數、門檻、格式、資料量或 seed。

- 性質：單層、單點診斷；不改正式 BC、不搜尋配置、不重選 knee、不 fine-tune。真值用 GPU 上的離線 PyTorch QDQ，不是 onnx-mlir native。
- 範圍：config 的全部 28 層（L0–L27）的 Down input h = SiLU(z_gate)·z_up；一次只量化一層；其餘權重與 activation 全 FP32；不量化 Down output。
- 格式：int4, int6, int8, p4e1, p6e1, p8e1，另設 FP32 identity。INT 用 native calibration 的 Down x range／scale／zero-point；Posit 用 Universal LUT（p4e1／p6e1／p8e1 都先驗證）。
- 三種分數（calibration 10 窗口 × 128 位置，n = 1280，float64；共用 μ_c 與 U；B 不在組內重新中心化）：
  - A：T_old = Σ_c mean(g²)·Σ r² + U
  - B：T_group = Σ_c [mean_S(g²)·Σ_S r² + mean_N(g²)·Σ_N r²] + U
  - C：T_pair = Σ g²r² + U
- spike：a_st = max_c |h_stc|；門檻 = 6 × calibration median；所有格式同一 mask。門檻無效回退 A；空組 = A；非空組 < 8 rows 只加旗標；不掃其他倍數。
- 梯度：每窗口 summed next-token CE（127 個預測位置）；position 127 的 g = 0，仍計入 n（與 native 相同）。
- 控制：wrapper 與 qwen3_prepare 一致；calibration tokens 一致；GPU／CPU QDQ 逐位元一致；TF32 關閉；hook 位置；autograd 梯度 vs gY·W_down；position 127 梯度為 0；FP32 identity 零誤差、零分數、replay loss = 無 hook loss；x+y 重建 vs native table C；合成資料（零誤差、常數梯度、空組、組間／組內分解）。控制失敗即停止、修正、記 amendment。
- table C（含 input＋output）不當作本輪的原 BC，只作一致性控制。
- 診斷資料：128 篇新 WT103 文章 × 3 個不重疊 128-token 窗口，Qwen3 tokenizer；verified 掃描排除 7721 篇歷史文章；排除 WT2 重複／近重複；另解碼全部歷史 Qwen token 檔比對重疊（audit）；seed：順序 27101、起點 27102。凍結分數之後才抽文章。
- 真值：每篇文章 NLL（3 窗口 summed CE / 381）；ΔNLL = 量化 − 同層 FP32 identity replay；負值保留。
- 主要比較：每層 int4–p4e1、int6–p6e1、int8–p8e1，共 84 對全部報告；其他 12 對／層為次要（跨位元，不標為同位元）。
- 可判定：文章 bootstrap（10000 次，seed 27103，所有層與格式同一次重抽）95% CI 排除 0；三種方法共用同一集合。proxy tie：相對差 < 0.02，單列、不算對。
- 多重比較：B−A、C−A 的整體比較用選格式後的 ΔNLL 差（R_M − R_A），Bonferroni（2 個比較）97.5% CI，另報 Holm 校正 p；特定配對的「確認修正」需 84 對的 simultaneous 95% CI（max-|t| bootstrap）排除 0；未校正 CI 只作探索。
- 穩定性：逐一移除 calibration 窗口（門檻固定），看各方法挑選是否翻轉。
- 決策：B 或 C 同時滿足 (1) R_M − R_A 的 97.5% CI 上界 < 0；(2) 可判定主要配對的修正 − 新增誤排 ≥ 3，修正分布 ≥ 3 層且至少 2 層不在 [0, 2, 14, 27]；(3) ≥ 2/3 修正在 10 次移除窗口下都不翻轉 → 只提出下一輪完整 W+A 配置比較計畫（原 knee 資源限制、同 MILP、新文章、另行登記），本輪不執行。否則保留原 BC、停止，不追加門檻搜尋。
- 不得只憑單點結果宣稱整體 pipeline PPL 改善。
- 計畫前的冒煙測試（scratchpad，不在 runs/）：calibration 階段完整跑過（看過其控制與 calibration 分數）；資料階段用其他 seed 抽 6 篇（未評估）；eval／analyze 只用 calibration 窗口，沒有新文章 loss。修正 3 個程式錯誤（tokenizer 匯入順序、檢查時重入 hook、整個 chunk 的 logits 超出 GPU 記憶體），設計未改。

## 預算（分鐘）：{'plan': 3, 'audit': 5, 'calib': 15, 'data': 10, 'eval': 30, 'analyze': 5}；超時只記錄，不中途截斷評估、不以部分資料取代計畫。

