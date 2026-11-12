# 計畫：L2 Posit4 的原始量化誤差分區與有限幅度診斷

時間：2026-10-26T03:58:49+08:00。以 plan.json 為準；寫入後修改須記 amendment。

- 範圍：Qwen3-0.6B 只做 L2 Down input；['int4', 'int6', 'int8', 'p4e1', 'p6e1', 'p8e1'] + identity；權重、其他 activation、Down output FP32；原 calibration、range、LUT、RMSNorm eps、loss。
- 遮罩：a = max_c|x|，門檻 11.29648733139038（recenter 輪凍結值）；S = a ≥ 門檻；calibration 須與 masks.npz 逐位元相同；B 用同一門檻在自己的 x 上建 S。
- 方向：e = Q(x) − x（原始、不中心化），e_S、e_N、e_all 固定；α = 1 主評估用 masked QDQ，其他 α 用 x + αd；α = 1 另報 x + e_G 重建差異。
- A（calibration）：α ∈ {0, ±2^-k, k = 0…10}；I_SN、Posit − INT 恆等分解 D_all = D_S + D_N + D_int；F_G = g·e_G（GPU baseline 梯度）、O/α、E/α²；D 分項（C_N、P_S）由 CPU 快取重現。
- B（recenter 輪 128 篇 × 3 窗口，已看過，事後延伸、非獨立確認）：α = 1 的 S-only、N-only、all；all 須重現上一輪 L2 loss；每文章平均、中位數、符號比例、事後描述性 bootstrap CI。
- closure：真實 Down、residual、L3–L27 全部 attention／FFN、final norm、LM head、CE（recenter 評估器路徑）。
- 控制：hash、TF32 關閉、deterministic、eager、eps 1e-6；LUT、INT 裝置；replay = 無 hook 前向；hook 完整前向 = 重播；只改 L2 Down input；D 分項與遮罩重現；B 的 identity／all 重現；e_S + e_N = e；α = 0；零方向；F 可加；配對恆等；g_x = g_y W；position 127。
- 冒煙測試：scratch smoke 2026-10-26 03:53-03:58 (outside runs/) of the plan and calib stages on the 10 calibration windows only (no B article). The first calib attempt stopped on a JSON error (a numpy bool in the control checks); the checks and F_additivity_rel are now cast to Python bool / float (no formula or tolerance change). The second attempt passed every control and its outputs were seen: alpha = 1 calibration partitions (int4/6/8 S-only 0.002-0.014, N-only 0.312, all 0.313-0.327; p4e1 S-only 3.969, N-only 0.027, all 3.945; p6e1 0.050 / 0.002 / 0.052; p8e1 -0.003 / -0.001 / -0.004 nat/token), the posit value sets (maxpos 16 / 256 / 4096), GPU vs CPU x / g differences, D reproduction and timings; the curves and F were computed but not inspected. The B replay was not run in the smoke; nothing of the question, formulas, masks, alpha grid, tolerances or reading was changed after seeing these outputs

