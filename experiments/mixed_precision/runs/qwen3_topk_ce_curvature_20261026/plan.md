# 計畫：L27 Down 的 top-k CE 二次項（k = 256）

時間：2026-10-26T23:06:04+08:00。以 plan.json 為準。

- 範圍與語意：與 runs/qwen3_norm_ce_decomposition_20261026 相同（L27 Down、目標 INT8、U／F／W、九格式、batch 8／尾批）。
- p：同一個 microbatch-1 baseline forward（梯度那次）的 FP32 logits 做 float64 softmax；T_k 依 p 取前 k（平手取較小 token id），不重新正規化。
- Khat_k = ½[Σ_T p(a−m)² + q(a_tail−m)²]，q·a_tail = w_tail·du（w_tail = W^T(p·1_tail)，baseline 一次 FP32 GEMM），m = Σ_T p a + q a_tail；q ≤ 1e-12 時 tail 項為 0。
- 主分數 P256 = S_u + Khat256；k = 1024 只在 C 作診斷，不進 N 選擇器。
- C 參考：K2_actual（前輪）、K2_affine（同 p 的完整 W du）、KL、D；分開 K2_actual−K2_affine、K2_affine−Khat（tail 方差）、KL−K2_actual。
- 凍結（只用 C）：primary argmin P256、reference argmin(S_u+K2_actual)、old argmin S_u、BC = f16；平手 f16 優先。
- N：新抽 128 篇 × 2 窗口，只量真值；seed {'order': 26102681, 'start': 26102682, 'bootstrap': 26102683}；排除 9641 篇已用文章。
- 主要問題：F 背景 primary pick 相對 f16 的 N 平均 D；mean < 0 且 CI 上界 < 0 才算確認改善。其餘為次要估計。
- 成本：F 背景 C、7 個非 f16 候選；Su／Su+Khat256／Su+K2full／directCE；冷、含 baseline、候選熱增量；5 warmup + 20 次，median／p90。
- 冒煙測試：Two scratchpad smokes on C only (no N). Smoke 1 (23:00-23:01, plan + calib): every control passed on the first attempt; all C results were seen: frozen picks P256 = p8e1, Su+K2actual = p8e1, old S_u = int8 in U / F / W; P256 and P1024 orders equal the C truth order (28/28 pairs) in all three backgrounds. After it only the control reporting changed (excess maxima start at -inf, raw maximum differences saved); formulas, k, tolerances and selectors unchanged. Smoke 2 (23:02-23:05, plan + calib + bench): passed; the timings were seen (F, 10 windows, 7 candidates, median total / hot: Su 0.149 / 0.023 s, Su+Khat256 0.455 / 0.029 s, Su+K2full 0.813 / 0.565 s, directCE 0.557 / 0.490 s). No design change after the smokes.

