# 分析計畫：OLMoE Down-input activation 的 spike 分組評分（事後探索性診斷）

時間：2026-10-23T18:10:37+0800。以 plan.json 為準。

- 性質：本輪使用已分析過的案例與 loss，是事後探索；事先固定規則也不會讓舊資料變成獨立確認資料。只做既有快取的 CPU 分析，不推論、不搜尋、不改 BC 或 pipeline。
- 範圍：olmoe_pair_gradient_diag 的 8 個 tap（int4、p6e1；F 條件）、olmoe_pair_generalization 的一般組 16 個與 spike 分層組 8 個（6 格式；單 tap 量化）。三組分開；全部 tap 保留；expert 依來源 metadata 對應；不混入 P／S／R。
- spike：s_t = max_c |h_tc|（快取的 FP32 calibration rows）；門檻 = 6.0 × 這些 rows 的 median；S／N；所有格式同一 mask。與舊實驗用 discovery median 的差異會列出比較。不依 g、誤差、loss 或排序改 mask；不掃其他倍數。
- 門檻非正／非有限 → 不能分組，沿用原 BC；空組 → 分組公式精確退化為原 BC；非空組 < 8 rows → 低覆蓋旗標、另列（不調權重、不挑結果）；不用 epsilon。
- 公式：A V_old = Σ_c mean_t(g²)Σ_t r²；B V_group = Σ_c[mean_S(g²)Σ_S r² + mean_N(g²)Σ_N r²]；C V_pair = Σ_c Σ_t g²r²；共同 μ_c（全部 rows）、r = e − μ_c、不在組內重新中心化；U = Σ_c GS_c μ_c² 不變；float64；不加係數或一階項、不以 NLL 換算。
- 恆等式（逐 channel）：V_group − V_old = (n_S n_N/n)(mean_S X − mean_N X)(mean_S Y − mean_N Y)；V_pair − V_group = n_S Cov_S + n_N Cov_N（母體共變異）；T_old／T_pair 須重現來源 mech.json。
- 排序：每篇文章（3 窗口）的兩格式 NLL 差；可判定 = 文章 bootstrap 95% CI 排除 0（10000 次，seed 23901），三種方法共用；主要 = 同位元 INT／Posit（mechanism_8 只有 int4 vs p6e1，作為其唯一配對報告）；其餘次要；proxy tie（相對差 < 0.02）單獨列出、不算正確；不可判定不算任何方法排對；三組分開、同一 expert 的配對不當獨立樣本。
- 窗口穩定性：逐一移除 calibration 窗口；門檻固定為完整 calibration 的值，mask 限於剩餘 rows；μ、U 與三種分數由剩餘 rows 重算；比較排序翻轉與分數變動；不能證明跨文章穩定。
- 停止規則：只改善單一已知案例、只勝過 T_pair 未勝過 T_old、或新增誤排抵消收益 → 建議停止此方向；即使有利也不接入 pipeline，確認需另行登記。
