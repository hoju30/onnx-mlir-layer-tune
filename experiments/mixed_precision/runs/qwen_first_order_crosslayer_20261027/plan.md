# 計畫：Qwen3 一階方向資訊的跨層確認

時間：2026-10-27T15:40:47+08:00。以 plan.json 為準。

- 目標（事先指定）：L7／L15／L23 Down，權重 = A 的格式（L7 int6, L15 int6, L23 int6）；A 原本的 activation 只記錄。
- 背景 U（其他投影 FP32）與 F（其他投影依 A）；候選 f16、p8e1、p6e1、p4e1、int8、int6、int4、bf16，input／output 同格式；fp32 只作控制。
- ranges 固定（runs/native_eps_fix_20261014/qwen3/ranges_by_point.json），不重新校準。
- 校準：3 組互不重疊、各 10 篇 WT103 train 文章 × 1 個 128-token 窗口；seed 26102731／26102732；排除 10055 篇已用文章與近重複。
- 評分：完整 f16 forward 捕捉 x_L、r_L、每個後續 block 的 kwargs；候選經真實 block L+1..27、final norm、head；S = g_u·Δu（只經 head），K_full = K2_actual。
- 主要：每個 目標×背景×校準組×bit 的同 bit 對，K_full 與 S+K_full 各選分數較低者（差 ≤ 4e-07 為平手 → 選 Posit）；δ_n = 54 個選擇的平均 [D_n(S+K 選) − D_n(K 選)]。
- 新文章：凍結後才抽 128 篇 × 2 窗口；20000 次文章 bootstrap（seed 26102735）；CI 上界 < 0 才稱改善，下界 > 0 稱變差，否則未定。
- 次要：每層／背景／bit／校準組；28 對修正／破壞（C、N）；同文章的 F−U 背景差（自己的 CI）；無限制 argmin。
- 冒煙測試：2026-10-27 15:38-15:40 in the scratchpad (outside runs/) with this script: plan + calib --smoke-c0 on the old C0 windows (runs/native_eps_fix_20261014/qwen3/calib_inputs.npy) for L7 / L15 / L23 x U / F: every control passed (replay vs complete forward bitwise for every format, configs / hooks, f16 zero, P0+Rest, end reproduction); the C0 scores and picks were saved there but not inspected; timings were seen (75 s). No calibration set or new article of this round was drawn. A random-number offline test of freeze_picks / analyze. No code change after the smoke.

