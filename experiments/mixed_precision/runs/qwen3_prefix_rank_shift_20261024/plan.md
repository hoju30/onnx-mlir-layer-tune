# 計畫：Qwen3 前層量化是否改變後層 Down activation 候選排名

時間：2026-10-24T04:45:55+08:00。以 plan.json 為準；寫入後的程式修改須記錄 amendment。依 `reports/qwen3_prefix_rank_shift_prompt_20261024.md` 實作。

- 邊界：不改 BC、搜尋器、knee、range、共享量化函式、權重；不 fine-tune；不新增 criterion；不重算 BC 統計。新背景的 calibration 實測評分只是診斷參照，不是 refreshed BC。GPU FP32 + QDQ，不宣稱 native。
- 結構核對：A sha256 = ac10afa96c7833ca…，與 ref.json 相同；config 28 層；mapping 196 投影、每 block 7 個；層號由 mapping 解析為整數。
- 背景：U = 前層 FP32；Q = blocks < i 的全部搜尋投影（attention 與 FFN）依 A；兩者的 block i 其他投影與 blocks > i 皆 FP32。目標 Down 權重兩背景皆為 A 的 w（逐位元相同）；activation 設定作用於 input x 與 output y，ranges／LUT 固定。L0 為負控制。
- 設定：fp32, int4, p4e1, int6, p6e1, int8, p8e1, f16；條件 28×2×8 = 448。主要比較 27 層 × 3 同位元對 = 81；f16 vs p8e1 為次要（資源不同）。
- selector（正式評估前凍結）：q_BC = argmin native C[node][a]；q_U／q_Q = 原 10 個 calibration 窗口上 U／Q 的實測 NLL 最小者。tie：BC 相對差 ≤ 0.02、calibration 差 ≤ 1e-08 nat/token → 選 A 原 a（若在對內）否則 INT；差 < 0.001 標 low margin 不刪除。
- 公式：D_s = E_s(INT) − E_s(Posit)（正值 Posit 較好）；J = D_Q − D_U；G_shift = mean[E_Q(q_U) − E_Q(q_Q)]；G_BC = mean[E_Q(q_BC) − E_Q(q_Q)]（81 對單點介入的平均，不是配置收益）。
- 資料：128 篇新 WT103 文章 × 2 個連續不重疊窗口（254 targets）；verified 掃描排除 8489 篇（含 spike／fp16／l27 實驗）；排除 WT2 與近重複、歷史 Qwen 窗口重疊；新增：任一窗口與 WT2 5-gram overlap ≥ 50% 即拒收該文章；seed 順序 30101、起點 30102；合格 18906 篇。
- 統計：文章為單位，paired bootstrap 20000 次（seed 30103）同一組索引；D／J 報 95% CI；G_shift、G_BC 報 95% 與 Bonferroni 97.5% CI；反轉 Holm family = 162 個 D 檢驗（零中心化 bootstrap 雙尾 p）；J 81 個另一 family；f16/p8e1 另一 family。
- ε = 0.001 nat/token。決策：G_shift 97.5% 下界 > ε → 支持下一輪背景相關重新評分研究（G_BC 也過才同時支持相對舊 BC 有改善空間）；只有 G_BC → 可能是 proxy 誤差；少數可信反轉但 G_shift 上界 < ε → 保留局部；跨 ε → 未定；集中於 L27／少數舊案例 → 只局部追查。不得啟動刷新、搜尋或組新配置。
- 控制：每條件 config／hash／每 block 量化數；權重指標核對；L0 U=Q、FP32 identity = 無 hook、重切可重現；fp32 設定只關目標 a；hook 數；目標 x／y 格式；TF32 關閉、deterministic；ranges 固定；非有限值保留；calibration 上每條件「重播 vs 完整前向」≤ 1e-06。失敗不放寬。
- checkpoint：每 chunk 存 CE 與 hash 標頭，續跑須相符；不依效應大小停樣本或擴樣本。
- 時間估計（冒煙測試，calibration）：{"source": "scratch smoke test 2026-10-24 04:33-04:45 (outside runs/)", "calib_stage_s": 268.9, "calib_replay_s": 80.3, "calib_complete_forward_s": 165.9, "eval_one_chunk_32_windows_s": 264.9, "eval_peak_gpu_bytes": 7011010048, "formal_eval_estimate_min": 35.3, "seen_before_plan": "the full calibration stage on the original 10 windows (controls, all 448 calibration scores, the frozen-to-be choices: 5 pairs with q_U != q_Q); an eval chunk and the analysis on calibration windows tiled to 32 (no new article); audit and a 4-article data draw with other seeds (999301/999302), not evaluated. Code fixes: weight identity check by data_ptr, Q prefix-count check, figure title. Nothing of the design was changed."}

