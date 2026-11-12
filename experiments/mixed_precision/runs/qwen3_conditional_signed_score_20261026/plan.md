# 計畫：L27 Down 的條件化、帶正負號一階評分

時間：2026-10-26T19:28:13+08:00。以 plan.json 為準。

- 背景：U（其他 FP32）、F（其他依 A）、W（只有 L0–26 Down 權重依 A）；目標 L27 Down 權重固定 A 的 INT8。
- 候選：['f16', 'p8e1', 'p6e1', 'p4e1', 'int8', 'int6', 'int4', 'bf16', 'fp32']；f16 為零方向。z_f = Q_y(linear(Q_x(x), Wq))，x 取自 input QDQ 之前。
- 分數：S_b,G(f) = Σ_G g_b·(z_f − z_f16)，g_b 為同背景 f16 輸出處的 summed-CE 梯度（只經 final norm 與 head，無 STE），每背景每窗口一次；G = P0／Rest／All。
- calibration（10 窗口）建立並凍結 S、選擇器（argmin S_All，a bits ≤ 16，平手保留 f16）；同時量 calibration 真值與 α 方向檢查。
- 新文章（128 × 2）只量真值 D_b,G(f)；主要家族（Holm 6）：p8e1 的 F−U P0／Rest／All、W−U P0、F 背景 P0／Rest。
- 控制：closure = 完整 forward；重現 down_source 的 U／F／W f16／p8e1 逐位置 CE；f16 δz = 0；位置 127 梯度 0；S 位置可加；TF32 關閉。
- seed {'order': 26102661, 'start': 26102662, 'bootstrap': 26102663}；排除 9513 篇已用文章。冒煙測試：scratch smoke 19:21-19:27 of plan and calib only (10 calibration windows; the down_source articles for the reproduction control); the first calib failed one control (closure vs full forward 0.0041 summed CE: the candidate matmul ran on all 10 windows at once while the forward runs batches of 8, and the coarse output QDQ amplified last-bit differences); fixed before this plan by computing z in the same window batches (no formula change). The second calib passed every control and its outputs were seen: all calibration scores, calibration truths, direction checks and the selectors (int8 in U / F / W; BC f16). No new article seen; nothing else changed


## Amendment 2026-10-26 19:36（calibration 凍結後、抽新文章前）

歷史 token audit 把 down_source 輪 work/native_tokens.npy（同輪已解析 tokens.npy 的每篇第 1 窗口）判為未解析；抽樣器讀 plan["data"] 而本計畫寫在 "new_articles"。修改：未解析檔須每個窗口與同 run 已解析檔逐 token 相同才接受，否則仍停止；抽樣器以 new_articles 作為 data。公式、分數、預測、選擇器、容差、統計與 seed 不變。腳本 sha 5e7d080bdc79 → eaf7e35ee710。
