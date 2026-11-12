# 計畫：Qwen3 spike 端點敏感度候選的機制確認與完整 native pipeline A/B

時間：2026-10-26T04:43:43+08:00。以 plan.json 為準；寫入後修改須記 amendment。

- 第一階段（只在原 10×128 calibration）：L2 Down input，六格式 S-only 的 α ∈ [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0]（α = 1 用 masked QDQ）；p4e1／p6e1／p8e1 的元素分區 H（每行最大 |x|）、O（S 中 |x| > maxpos）、R（S 其餘），I_OR = Δ_S − Δ_O − Δ_R；L2 的 B／D／E x-only 分數預覽與 ½(g0+g1)·e_S 診斷。
- 候選 E（唯一）：所有有 spike 行的 Down input、10 個量化格式；C_N = V_N(μ_N) + U_N(μ_N)；Pend = ½Σ_S(g0² + g1²)e²，g1 在 x1 = where(S, q, x) 的同一 GPU closure；C_x,E = C_N + Pend；對照 D（C_N + P0）、B（GPU 原 BC C_x）。
- 表：A = native table.json；B／D／E 只替換有 spike 行的 Down x 項，C_total = C_y,old（重跑原 native 統計）+ C_x,new；先驗證 C_x,old + C_y,old = C_old。
- 搜尋：mp_auto.auto_select（MILP、17 點、2 輪細化、time limit 10 s、原 knee）；A_auto 須重現歷史 knee；E_auto；B／D／E_cap 用 A_auto 實際 bytes 與 BOPs 兩個上限；A_aggr（A frontier 上 R 最接近 √(R_min·R_Aknee)）與 E_aggr；A_cap 為控制。
- native：bytecode、constants to file、Krnl O3、clang O0、per-channel weight；控制窗口重現前輪 native A／FP32（1e-8）；去重後編譯與評估。
- 資料：凍結後抽 128 篇新 WT103 × 1 窗口；seed {'order': 26102621, 'start': 26102622, 'bootstrap': 26102623}；排除 9001 篇已用文章。
- 主要：P1 = E_cap − A_auto、P2 = E_auto − A_auto；文章 bootstrap 20000，95% 與 Bonferroni 97.5% CI；δ = 0.001 只稱非劣。次要 Holm：E_aggr − A_aggr、B_cap − A_auto、D_cap − B_cap、E_cap − D_cap。GPU LMReplay 為後端對照。
- 冒煙測試：scratch smoke 2026-10-26 04:34-04:43 (outside runs/) of plan, phase1, gpu_stats, native_xy, tables and search on the original calibration only (no new article, no native compile or evaluation). Before it, the suggested seeds were found used (26102611 = bootstrap seed of the raw-error round) and replaced by 26102621-23. Seen: phase-1 dense alpha (p4e1 S-only 0.032 / 0.027 / 0.020 / 0.023 / 0.032 / 0.068 / 0.099 / 0.258 / 0.975 / 2.562 / 3.969 over alpha 0.5..1; p6e1 0.050 at 1), element partitions (p4e1 H 0.026, O 4.020, R -0.000; p6e1 H 0.021, O 0.049; p8e1 O empty), the L2 x-only preview (B int4 / int6 / int8, D p4e1 / p6e1 / p8e1, E int4 / int6 / p8e1), the GPU components of all 10 spike layers, native x + y = table (rel 0), the GPU-vs-native x-slot differences, and the search: A_auto = historical knee; E_auto differs in 7 projections (same R, size 0.761012 vs 0.759821); A_cap = B_cap = A_auto; D_cap = E_cap (7 projections from A_auto); A_aggr R 0.707068 and E_aggr differ. Nothing of the formula, tables, search rules, configs to compare, statistics or tolerances was changed after seeing these outputs


## Amendment 2026-10-26 04:58

controls 階段在 CPU 環境呼叫 GPU LMReplay（只報告的 FP32 native 對 GPU 比較）而停止；native 控制已全部跑完，尚未凍結或抽資料。該比較移到 pytorch_eval（GPU 環境），寫成 controls_gpu.json，並擴及所有評估配置；native 控制、容差、凍結與其他階段不變。腳本 sha e4d3fec4ae08 → 3fdb7ea7a7fa。
