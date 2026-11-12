# 預先登記：OLMoE 預設 knee 點的 selective sensitivity refresh

時間：2026-10-22T11:07:22+0800。以 plan.json 為準。

- knee：修正後 static BC 表 + 已驗證的研究搜尋（LP 優先取代 auto_select 的每預算求解，17 樣本、2 輪 refinement、原二維 normalized-chord knee），第 1 次 auto 的 selected 為 q0_knee；不手動選點、不手動設 0.7071；若為端點則停止。
- 共同上限：selected 的 search_budget × auto_select 參考 bytes／BOPs（就是求解器收到的精確上限），不用四捨五入的 R、不用配置實際用量。
- A：q0_knee；B：q0_knee 工作狀態完整 refresh；D：q0_knee 中 weight 或 activation ≤4 bit 的投影只更新 sensitivity（重算並凍結目標集合），其餘沿用 static；B、D 在 knee 上限各重新搜尋一次，所有節點都可改格式。
- 數學與梯度語義同上一輪；D 的 activation 成本 Σ(G2_new·var_static + GS_new·μ_static²)，以 static routed rows 正規化。
- 重用上一輪的 FP32 static 統計（static 表、ΔW、activation 快取），依賴 hash 全部一致（reuse_ok = True）；q0_knee 的梯度與誤差全部重算。
- 控制：static 表一致與 knee 重解、q0_knee forward 與獨立 forward 一致、input gradient 逐位元、D 非目標 = static、D 目標 = 公式（numpy 獨立重算）且 weight 列 = B、合法與上限、輸入不變、重複一致、複製的 build 重現上一輪 D 表。
- 計時：auto 3 次；B／D 各 3 次（B1 D1 D2 B2 B3 D3）；端到端 A、B、D 各 1 次（實測）；另報各階段中位數相加的估算並明確標示。
- 評估：600 篇未用過的 WikiText-103 文章 × 4（排除 2474 篇歷史文章）；FP32、A、B、D；seed 22501／22502／22511。
- 主要：Δ_low = D − A；非劣：Δ_keep = D − B，上界 < +0.0005；次要：B − A。非劣不等於改善。
