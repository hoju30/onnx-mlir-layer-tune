# 計畫：L27 一階評分的 RMSNorm／CE 餘項分解

時間：2026-10-26T20:07:47+08:00。以 plan.json 為準。

- 範圍：L27 Down、目標權重 A 的 INT8；背景 U／F／W、九種格式、完整 input／output QDQ 與前輪完全相同（同 Env、同 batch 8）。
- 一次 baseline autograd 取 g_z 與 g_u（每窗口 microbatch 1）；S_z = g_z·(z_f − z_0)，S_u = g_u·(u_f − u_0)，R_norm = S_u − S_z。
- 真值 D = FP32 CE 差；同一份 FP32 logits 轉 float64：D_64、S_logit,64、KL_64（p_0 → p_f，不 clip）、eps_head = S_logit,64 − S_u、eps_CE = D − D_64。
- 恆等式（逐位置）：D_64 = S_logit,64 + KL_64；D = S_z + R_norm + KL_64 + eps_head + eps_CE。K2 = ½Var_p0(Δlogit) 只作描述，不是分數。
- C（原 10 窗口）：重現前輪 S_z／D，分解，凍結 S_u 選擇器（argmin S_u,All，a bits ≤ 16，平手保留 f16）。
- H（前輪 128 篇 × 2，已看過）：逐窗口自身背景梯度的 local 分解，只作事後機制分析；凍結選擇器的探索性重播；文章 bootstrap 20000 次（seed 26102671），只作描述。
- 控制：LM head 為 affine、無 softcap；closure = 完整 forward；前輪重現；g_z = Norm VJP(g_u)；f16 全 0；位置 127 梯度 0；恆等式與位置加總。
- 冒煙測試：2026-10-26 20:06-20:07 scratchpad smoke (plan + calib on C only; H not computed): all calibration controls passed on the first attempt (closure vs full 0; previous S_z / D reproduced within 5.6e-17 per token; chain VJP 0, analytic float64 1.8e-7; identities 3.6e-14; KL min 0; f16 0; position-127 0; no rounding flag). All C decompositions and the S_u selector (int8 in U / F / W) were seen; no design change after the smoke.

