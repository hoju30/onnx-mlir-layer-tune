# 計畫：L26 Down 的 block27 後綴重播與 top-256 評分

時間：2026-10-27T01:41:11+08:00。以 plan.json 為準。

- 目標 /model/model/layers.26/mlp/down_proj/MatMul，權重固定 A 的 int6；背景 F（主要，其他 195 投影依 A，含 block27 的 QDQ）與 U（其他 FP32）；九種 activation 格式（input／output）。
- 候選：z26 = Q_y(linear(Q_x(x26), W26q))（batch 8／尾批 2），h27 = r26 + z26，經實際 block27（同批捕捉的 kwargs、use_cache=False）、final norm、head。
- g_u 只經 head（u_0 為 leaf，每窗口一次 backward）；S_u = g_u·(u_f − u_0)；Khat256／P256 沿用前輪實作與精度，k = 256 固定。
- C 參考：D、K2_actual、K2_affine、KL、D_64、S_logit,64、浮點殘差；P0／Rest 只指最終 CE 位置。
- 凍結（C）：primary argmin P256、quadratic_reference argmin(S_u+K2_actual)、measured_C_reference argmin D_C、first_order argmin S_u、f16。
- N：新抽 128 篇 × 2，只量真值；seed {'order': 26102711, 'start': 26102712, 'bootstrap': 26102713}；排除 9769 篇已用文章。主要：F 背景 primary 相對 f16，mean < 0 且 CI 上界 < 0 才算確認改善。
- 成本：F、C、7 候選，四方法都含每候選 target、block27、norm；5 warmup + 20 次，median／p90；冷時間為重建值。
- 冒煙測試：One scratchpad smoke on C only (no N), 01:36-01:40, plan + calib + bench with this exact script: every control passed on the first attempt (replay vs complete forward h27 / u / CE all 0; previous F / f16 CE reproduced exactly). All C results were seen: F picks P256 = Su+K2actual = D_C = int8 (C truth -0.0009), first order int6; U picks p8e1 for the three, first order int6; P256 orders equal the C truth order (28/28) in both backgrounds. Timings were seen (F, 10 windows, 7 candidates, baseline + candidates median: suffix+Su 0.227 s, +Khat256 0.534 s, +K2full 0.891 s, +directCE 0.570 s; prefix 0.625 s). No design change after the smoke.

