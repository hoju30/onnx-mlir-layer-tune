# 計畫：ResNet18 weight mixed-precision 的 baseline 比較（A8 固定）

時間：2026-10-29T00:25:04+08:00。以 plan.json 為準。

- 範圍：ResNet18／ImageNet；activation 全部 INT8（固定 ranges）；weight 候選 int4／int6／int8（per-channel 對稱）；搜尋 19 個內層 conv，conv1 與 fc 固定 INT8。
- 五種評分：BC（native weight criterion）、SQNR（模型輸出 logit 雜訊，不是 weight SQNR）、HTQ（隨機初始化 SNIP × HTQ 自己的 per-tensor 擾動）、HAWQ（CLADO 附帶實作：Hessian trace／參數數 × ‖ΔW‖²）、InfoQ（SMI 評分的無訓練受控比較）。
- 校準：同一個 1000 張的池（InfoQ 的選類規則，10 類 × 100 張 train），各方法取前 k 張：{'BC': 10, 'SQNR': 10, 'HTQ': 32, 'HAWQ': 64, 'InfoQ': 1000}；A8 ranges 用既有 10 張 validation 的 native ranges，不重算。
- 評估：全新的 ImageNet validation shard（data/validation-00001-of-00014.parquet）全部圖片，native 編譯評估；配置凍結後才評估。
- 搜尋：共同 MILP（dense 模型的正式規則）；LP-first 在每個上限當對照；共同自動外殼 = mp_auto.auto_select。
- 資源點（A）：由 BC 的 calibration-only 自動 frontier 決定：knee R_k；寬鬆 R_k + (1 − R_k)/2；激進 R_min + (R_k − R_min)/2；size 與 BOPs 都 ≤ R × all-INT8；精確上限寫入 caps.json 後才搜尋其他方法。
- 統計：每點 4 個方法各與 BC 比 top-1（共 12 個比較），exact McNemar + Holm（α = 0.05）；配對 bootstrap 20000 次 95% CI 只作描述；不設等效界線，未顯著＝未定。
- 計時：BC、SQNR、HTQ、HAWQ 交錯重複 3 次取中位數與範圍；InfoQ 至少 1 次；裝置不同時只作實測 pipeline 成本比較。
- 冒煙測試：planned after this plan: scratch run directory with symlinked calibration pool and a fake 12-image evaluation set taken from the pool (never the held-out shard) to exercise every stage; fixes are recorded in amendments.json

