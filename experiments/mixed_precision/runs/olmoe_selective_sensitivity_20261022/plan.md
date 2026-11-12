# 預先登記：OLMoE sensitivity-only 與 low-bit selective sensitivity refresh

時間：2026-10-22T05:08:39+0800。以 plan.json 為準。

- 資源上限：直接讀原 plan（weight bytes ≤ 3804495872、BOPs ≤ 39226380910592），BOPs 用 FP32 calibration routed rows。
- 固定：3051 個搜尋投影、GRID11×GRID11、原 10×128 calibration、固定 ranges、修正後量化器、原 checkpoint 與 router；不載入 router FT。
- A：原 q0；B：完整 refresh（須重現 table_refresh 與 q1）；C：全部節點只更新 sensitivity；D：只更新 q0 中 weight 或 activation ≤4 bit 的 1161 個投影（所有候選格式），其餘沿用 static 表。
- weight 項：S_new[c]·D_static[c,f]，D_static 沿用原快取（不算省時）；C 的 weight 項與 B 完全相同。
- activation 項：快取 static 的 n_s、Σerr、Σerr²（float64，逐 channel、逐格式；gate／up 共用 x_in），A_new = Σ (G2_new·var_static + GS_new·μ_static²)，以原 BC 運算順序 G2/n_s·(S2 − n_s μ²) + GS·μ² 計算。
- B、C、D 都跑完整 q0 forward 與完整逐層 backward；D 只省去非目標投影的 tap／梯度保留、weight sensitivity、統計與成本重建。
- 控制：static 表與 q0 重現（≤1e-12 相對差，預期 0）、B 重現 table_refresh 與 q1、forward／路由／input gradient 逐位元、D 目標列 = C、D 非目標列 = static、重複間配置一致（主要配置固定為第 1 次）。
- 計時：每次一個 process；static plain／cache 各 3 次（P1 C1 C2 P2 P3 C3，前置 r0 為正式快取與控制）；B／C／D 各 3 次（B1 C1 D1｜C2 D2 B2｜D3 B3 C3）；記錄 page-cache 駐留；報告中位數與範圍。
- 評估：600 篇未用過的 WikiText-103 文章 × 4 窗口（掃描全部歷史 manifest 排除 1874 篇）；FP32、A、B、C、D；seed 22401／22402／22411。
- 主要：Δ_low = NLL(D) − NLL(A)；上界 < 0 → D 優於 static，下界 > 0 → D 變差，否則未確認。
- 非劣：Δ_keep = NLL(D) − NLL(B)；上界 < +0.0005 才算不差超過界線；否則證據不足（不寫成一樣好），不追加文章。
- 次要（探索）：C−A、C−B、D−C。
