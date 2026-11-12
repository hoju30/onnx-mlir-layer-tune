# 預先登記：OLMoE aggressive 配置「固定 expert 選擇、恢復 routing weight 總和」診斷

時間：2026-10-18T20:41:02+0800。以 plan.json 為準。

## 條件（各自完整 forward；p 來自各自 hidden state）

- 基準：完整 FP32 dynamic，保存 S0（top-8）與 p0（全部 64 個機率）；w0 = p0[S0]，m0 = sum(w0)（每 token、每層各自的值，不是 1）。
- L00：IDs = S0，weights = w0。
- L01：IDs = S0，weights = w = p[S0]。
- Lmass：IDs = S0，weights = w × (m0 / m)，m = sum(w)；float32，sum 依 S0 順序；不 normalize 到 1、不改 logits、不 clip、不加 epsilon；零或非有限值即停止。
- L11：完整 dynamic（背景重現）。
- 恢復總和會改變後續 hidden state：這是逐層恢復總量的端到端干預，不是局部分解。

## 資料（新來源）

- WikiText-103 raw train（兩個 parquet，sha256 = hub LFS oid）。嚴格文章邊界：28472 篇。
- 排除與 WikiText-2（全部 720 篇）標準化標題或內容 hash 相同者 613 篇，以及 WikiText-103 內部重複群組 464 篇；剩 27395 篇。
- 依隨機順序（seed 18111）逐篇：單篇 tokenize；<512 token 跳過；與任一 WikiText-2 文章或已選文章的 5-gram shingle 重疊 ≥10% 視為近重複跳過；其餘接受，起點 seed 18112，每篇 4 個連續窗口。
- 新來源結果不與先前 WikiText-2 確認集合併。

## 樣本數

- Pilot：上一輪 route-decomposition 窗口 0–127（32 篇舊文章），重現檢查並估 Lmass − L01 的文章級 SD s。
- n = min(300, max(64, ceil((1.96 s / 0.0015)²)))，於新資料評估前固定。

## 終點與統計

- 主要：Δ_restore = CE(Lmass) − CE(L01)，95% CI；負值 = 恢復總和有幫助。
- 次要族（Bonferroni ×3，96.67% 區間）：Residual = Lmass − L00（±0.002 等價或 >0）；E_weight = L01 − L00（重現）；E_total = L11 − L00（重現，另對 ε）。另列 95%。
- 10000 次 paired article bootstrap（seed 18121）；PPL 由平均 CE 取 exp；不做「總量占 X%、比例占 Y%」的分解。

## 控制

- pilot: baseline S0 / w0 / CE and L00, L01, L11 CE reproduce run_olmoe_route_decomposition windows 0-127 bitwise
- Lmass semantics: ids = S0; recorded m0 = float32 sum of p0[S0]; |sum(weights) - m0| reported (max relative < 1e-5); weights / unscaled w constant per token (relative spread < 1e-5); sums not set to 1; scale range reported
- quantization off (control subset = first chunk): L00, L01, Lmass, L11 each == FP32 baseline bitwise (Lmass: scale = 1 exactly when p = p0); otherwise the difference is measured before interpretation
- reruns on the control subset bitwise; replay of every condition's recorded ids / weights through run_olmoe_dynamic_route.forward reproduces it bitwise (all windows)
- config completeness, router untouched, hooks on expert projections only, finite values, 127 tokens per window
- any failure: fix and record before analysing losses

## 判讀

- helps：Delta_restore 95% CI entirely below 0: restoring the FP32 sum helps (FP32-informed diagnostic only)
- insufficient：Residual Bonferroni-positive after improvement: restoring the sum does not remove the weight-path damage
- null：Delta_restore CI contains 0: no claim either way; report the improvement range the CI excludes
- harms：Delta_restore CI entirely above 0: restoring the sum hurts, reported as is
- replication：E_weight / E_total not replicated on the new source: stated explicitly
- scope：even if it helps, it does not carry over to full dynamic routing (FP32 ids and FP32 m0 used)
