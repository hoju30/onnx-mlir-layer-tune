# 計畫：Qwen3 L27 Down input 的 Posit 平均偏差項 U 機制診斷

時間：2026-10-25T01:07:09+08:00。以 plan.json 為準；寫入後修改須記 amendment。規格：`reports/qwen3_posit_mean_bias_prompt_20261025.md` 與 2026-10-25 的需求。

- 範圍：L27 Down input；權重與其他 activation 全 FP32；output 不量化；int4／p4e1（主）、int6／p6e1、int8／p8e1（對照）；原 10 × 128 calibration；不抽新文章、不宣稱泛化。
- 分解：e = Q(x) − x；μ_c = Σ e / 1280（float64，算一次固定）；b = μ（每個位置）；r = e − b；全在 autograd 圖外。
- 量測：L(x)、L(x+b)、L(x+r)、L(Q(x))、x+e 與 x+b+r 的重建；ΔL_b、ΔL_r、ΔL_e、I_br = ΔL_e − ΔL_b − ΔL_r（逐窗口）。
- 分數（同一 b 方向）：U = Σ_(s,c)(Σ_p g)²μ²（原 BC）、U_elem、U_token；F_b = g·b；K_b = ½ b·H_b（autograd HVP，create_graph）。
- α 掃描：[0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625]；O(α)、E(α)、R(α)；小 α 的實際改變比例與 rounding 殘差。
- 位置：固定 μ 的 b_P0、b_Rest：F、K、α=±1；U0、UR、Ucross（Ucross 可負）；F、K、mean-only ΔL 的位置可加性。
- HVP-FD：h = 1/32、1/64、1/128 全報；相鄰步長 0.02 相對 + 0.001 絕對、FD(1/128) 對 HVP 0.05 相對 + 0.001 絕對（summed CE）；不收斂即數值未定。
- 控制：完整前向 = 重播（逐位元）；只改 Down input、W 不變、無 output QDQ；重現舊 full loss 與舊 U；μ、r、b+r=e；g_x = g_y W；position 127 的 g、H_b = 0；α=0 → 0；TF32 關閉、deterministic、float32；INT 在裝置上；LUT 驗證；非有限值保留。
- channel：每個 channel 的 μ、GS、U_c 與 INT−Posit 差（只描述，不產生 mask）。
- 判讀：A 曲率 proxy 失配（U/2 與 K_b 反序且 K_b 經 FD 與小 α 支持）；B 一階／有限擾動；C 平均與剩餘補償（I_br）；D 不同格式分開；E 修 U 不等於修好 position 0 變異項或整個 BC。
- 統計：10 個 calibration 窗口只作描述，不做 bootstrap。單位：內部 summed CE，表格最後除以 1270 targets；μ 分母 1280。
- 計畫前已看：reversal diag 報告與 stats、score geometry 報告與 scores.csv、slot attribution 目錄；冒煙測試：見 plan.json。

