# 計畫：BC 變異項的 Posit 誤排機制診斷（Qwen3 Down input，L27 主、L2 對照）

時間：2026-10-25T05:09:37+08:00。以 plan.json 為準；寫入後修改須記 amendment。

- 範圍：L27 Down input 主分析、L2 Down input 預先指定對照；兩層六格式全報（含 L2 4-bit）；一次一層；權重與其他 activation FP32；output 不量化；原 10×128 calibration、range、LUT；不抽新文章、不宣稱泛化。
- 分解：e = Q(x)−x；μ = Σe/1280（float64，一次）；b = μ；r = e−b；方向 r_all、M_P0 r、M_Rest r，不重估 μ、不重新中心化。
- proxy（不加 U）：V_old = Σ_c mean(g²)Σr²、V_elem = Σ(gr)²（= 舊 V_pair）、V_token = Σ_(s,p)(Σ_c gr)²、V_window = Σ_s(Σ_(p,c) gr)²；以 V/2 對照 K_r。
- 真實導數：F_r = g·d；K_r = ½ d·H_r（autograd HVP，create_graph，microbatch 1）；b·H_r 另記。L27 closure = Down→residual→final norm→head；L2 closure = 真實 L3–L27（含 attention、FFN）→norm→head。
- 曲線：α ∈ {0, ±1, ±1/2, ±1/4, ±1/8, ±1/16, ±1/32, ±1/64}，全部 layer／format／mask；O、E、R；參考 Q(x)、x+b、x+e、x+b+r 與 I_br。
- HVP-FD：h = 1/32、1/64、1/128，18 個 r 方向 + 1 個合成方向；相鄰 0.02 相對 + 0.001 絕對、FD(1/128) 對 HVP 0.05 相對 + 0.001 絕對；不收斂即未定。小 α 支持：α = 1/16，0.2 相對 + 0.001 絕對（summed CE）。
- 位置：F、V_old 兩層都應可加；L27 的 K 與 ΔL 應可加；L2 報 K_position_cross 與 I_position(α)（保留正負號）。
- 判讀旗標（INT−Posit，mask all）：A V_old/2 與 K_r 反號（K 經 FD 與小 α 支持才算支持）；B K_r 與 ΔL_r(1) 反號而 F+K 與 ΔL_r(1) 同號；C F+K 與 ΔL_r(1) 反號；數值不足即未定；L27 不推到 L2。
- 控制：完整前向 = 重播；只改 Down input、W 不變、無 output QDQ；重現 L27 前輪 r／b／e loss 與 F_b／K_b、V_old／V_pair／U（reversal diag 與 spike diag）、L2 x+b loss（U_token 輪）；g_x = g_y W；position 127；α = 0；TF32 關閉、deterministic、eager；LUT、INT 裝置端；batch 1／2／10；HVP 機制（L27 全部方向含合成方向 FD 確認；兩層逐窗口 FD 已收斂者 ≥ 95% 與 HVP 一致；L2 合成方向只報告）。
- 統計：10 個窗口只作描述，不做 bootstrap。單位：內部 summed CE，表格最後 ÷1270；μ 分母 1280。
- 冒煙測試：two scratch smokes on the same 10 calibration windows (outside runs/). Smoke 1 (2026-10-25 04:55-04:59) with the first tolerances: 3 control failures - batch g (L2 mb10 vs mb1 1.05e-4 rel > 1e-4), same-backend reproduction (L27 F_b up to 2.6e-3 rel from the 2-window vs microbatch-1 path and U up to 2.5e-4 rel, both signed sums with cancellation; > 1e-4), and the L2 synthetic direction FD; seen: all L27 directions FD-confirmed; at L2 every direction touching Rest positions failed FD over h = 1/32..1/128 while P0 directions passed; per-window E(alpha)/alpha^2 converged to the HVP in most windows but window 0 (and several p4e1 windows) changed sharply between alpha = 1/32 and 1/64 (real narrow-scale nonlinearity, about 0.04 summed CE >> rounding); a probe (scratchpad probe_l2.py) showed position 0 Down-input spikes 3000-3800 in every window, the L2 INT range [-3012, 3787], so INT4/6/8 flush ordinary activations to 0 (INT r ~ -x at Rest). Changes before this plan: FD judged also per window and an HVP machinery control (all L27 directions confirmed; converged window cells >= 95% matching the HVP), the L2 synthetic direction reported only, batch g tolerance 1e-3, same-backend tolerance 1e-3 with 1e-2 for the signed sums F_b / U, V_pair compared as T_pair - U, finite-amplitude orders at every alpha reported. Smoke 2 (05:05-05:09) with this script: every control passed (280 / 280 converged window cells match the HVP, 100 L2 cells not converged); all calibration results, flags and figures were seen; the formal run repeats the same deterministic computation on the same windows and is not an independent check

