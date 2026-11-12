# 計畫：Qwen3 Down input 的分組平均誤差與 spike 同位置評分

時間：2026-10-26T02:30:23+08:00。以 plan.json 為準；寫入後修改須記 amendment。

- 範圍：Qwen3-0.6B 全部 28 層 Down input h，一次一層；權重、其他 activation、Down output 全 FP32；['int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1'] + identity；原 calibration、range、LUT、loss。
- 遮罩：a = max_c|h|，門檻 6.0×median(a)，S = a ≥ 門檻，N = 其餘；所有格式共用；與舊 mechanism.json 核對；組 < 8 列標低覆蓋。
- 公式：V_G(m) = Σ_c mean_G(g²)Σ_G(e−m)²；U_G(m) = Σ_(s,c) m²[Σ_(p∈G) g]²；P_G(m) = Σ_G g²(e−m)²。
- 方法：A = V_all(μ_all)+U_all（原 BC）；P = P_all(μ_all)+U_all（舊配對）；G00 = V_N(μ_all)+V_S(μ_all)+U_all（舊分組）；G01 = V_N(μ_all)+P_S(μ_all)+U_all；G10 = V_N(μ_N)+U_N(μ_N)+V_S(0)；**D = V_N(μ_N)+U_N(μ_N)+P_S(0)（唯一主要候選）**。回退：S 或 N 空、門檻無效 → G00／G01／G10／D = A。
- 快取：舊 down spike diag 的 h、g；核對輸入 hash、誤差 sha、舊分數（A／P／G00 = T_old／T_pair／T_group，1e-9 相對）與窗口 0 的重算。
- gate：D 與 A 在 84 對的選擇完全相同 → 完成快取分析與報告後停止；否則抽 128 篇新 WT103 × 3 窗口，驗證 28 層。
- 排序：較小分數；相等選 INT；相對差 < 0.02 為 proxy tie；文章 bootstrap 10000（seed 26102603）95% CI 排除 0 才可判定；simultaneous 95% CI 判確認。
- 主要終點：G = R_D − R_A（每篇 84 對所選格式的平均 ΔNLL），95% CI。次要家族（Holm）：D−G10、G01−G00、G10−G00、D−G01、G00−A、P−A。
- 候選資格：G 上界 < 0；確認修正 > 確認新增誤排且 L2 6／8-bit 至少一對確認修正；L2 4-bit 若可判定須排對；該 L2 修正 LOO ≥ 8/10 維持。
- 舊資料重播：探索性，凍結後才做，與新資料分開，不用來選方法。
- seeds：{'order': 26102601, 'start': 26102602, 'bootstrap': 26102603}（未出現在任何舊 plan）；排除 8873 篇已用文章；合格 18522 篇。
- 冒煙測試：scratch smoke 2026-10-26 02:26-02:29 (outside runs/) of the plan and scores stages on the cached original calibration (no old-article replay, no new data). The first attempt crashed on a 0/0 in the score-level identity rounding report for the all-zero error control; a 1e-300 floor was added (no formula change). The second run passed every control and was fully seen: components, six scores for all 28 layers, the 84 frozen predictions and the gate (D differs from A only at L2 int4-p4e1, int6-p6e1, int8-p8e1, all switching to Posit; L27 and other layers unchanged), LOO flips and timings. Before the smoke the mask-position comparison was fixed to string keys and the identity check was placed at the component level (P_S - V_S vs the covariance form; score-level rounding reported). Nothing of the formulas, methods, mask, tolerances or decision rules was changed after seeing these outputs


## Amendment 2026-10-26 02:36

第一次正式 scores 只在來源抽查失敗（窗口 0 的 CPU 重算與快取差 h 2.3e-3、g 4.1e-3 相對）；同次的誤差 sha 與舊分數重現皆精確，未凍結。之後三個獨立 process 重算與快取逐位元相同，判為偶發的 CPU 數值不確定性。程式、容差與規則不變，原樣重跑；失敗紀錄保留為 controls_scores_attempt1_failed.json。腳本 sha 0017dd4b5b735ff9。
