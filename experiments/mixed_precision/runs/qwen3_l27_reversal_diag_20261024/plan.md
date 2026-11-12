# 計畫：Qwen3 L27 Down 的 BC 預測與實測誤差為何反轉

時間：2026-10-24T15:46:33+08:00。以 plan.json 為準。

- 問題：BC 表 C[L27 Down]：int4 6482 < p4e1 7177、f16 0.0003 ≪ p8e1 160；實測卻是 p4e1、p8e1 較好。拆解：槽位（input x／output y）、位置（0／其餘）、誤差項（一階／曲率），以及 BC 公式在自己的 calibration 上是否就反轉（近似誤差）或只在新文本反轉（轉移）。
- 設定：與 BC 的 C 項相同，除 L27 Down activation QDQ 外全部 FP32（含權重）。L27 是最後一層，只重算 Down、final norm、LM head。
- 條件：7 格式 × x、y 各 3 遮罩 × α（1、−1、0.5、−0.5）+ xy × 3 遮罩（α=1）= 189。ranges、LUT 固定。
- 量測：dL(α)；a = (dL(1) − dL(−1))/2（奇次／線性部分）；b = (dL(1) + dL(−1))/2（偶次／曲率部分）；三次檢查。
- in-sample 分數（同一批窗口的 FP32 前向／反向）：BC = T_old(e·m)（native 公式）及其 V_old、U；V_pair；F = Σg·e（一階預測）；½Σ_w(g_w·e)²（經驗 Fisher 的二階預測）；½T_old（把 BC 當成 b 的預測）。
- 判讀規則：成對 D = dL(INT 或 f16) − dL(Posit)（正值 Posit 較好）。某槽位的實測 D 與其 BC 預測反號 → 該槽位承載反轉；BC 去掉 position 0 後與實測同號 → position 0 解釋反轉；D_b 與 BC 同號而 D、D_a 反號 → 曲率對、一階項翻盤；D_b 與 BC 反號 → 曲率近似錯。calibration 與新文章的 in-sample BC 都反轉 → 公式近似誤差；只有新文章 → 轉移。
- 不確定性：新文章以文章 paired bootstrap（20000 次，seed 31103）95% CI；CI 跨 0 即未定；calibration 只作描述。
- 控制：同路徑 FP32 = 完整前向（逐位元）；α=1 全遮罩 = QDQ；calibration in-sample BC x+y vs native C[L27] 相對差 ≤ 0.01；梯度 position 127 = 0；LUT 驗證；TF32 關閉、deterministic。
- 資料：原 10 個 calibration 窗口；128 篇新 WT103 文章 × 2 窗口（沿用前輪抽樣規則；verified 掃描排除 8617 篇，含前輪 128 篇）；seed 順序 31101、起點 31102。
- 邊界：只做診斷，不提出 criterion 修改、不改 pipeline。
- 計畫前冒煙測試（scratchpad，不在 runs/）：calib 跑過原 10 個 calibration 窗口，eval／analyze 跑過重複填充成 256 窗口的 calibration（沒有新文章）；已看過 calibration 端的量測與 in-sample BC 項；設計未改。

