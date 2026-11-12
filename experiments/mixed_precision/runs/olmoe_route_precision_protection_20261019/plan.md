# 預先登記：OLMoE aggressive 配置的逐層權重精度保護與同資源 BC 對照

時間：2026-10-19T01:05:51+0800。以 plan.json 為準。

## 候選

- 16 份：第 l 份把第 l 層所有 expert 投影中 int4–int7 的權重改為 int8；其餘（int8、bf16、activation、range、其他層、21 個缺席投影）不變。
- 資源沿用固定成本定義（calibration routed rows）；R = max(M/M8, B/B8)。

| 層 | 修改節點 | ΔM_rel | ΔB_rel | ΔR | R |
|---:|---:|---:|---:|---:|---:|
| 0 | 192 | +0.02880 | +0.02400 | +0.02880 | 0.62340 |
| 1 | 191 | +0.02733 | +0.02242 | +0.02732 | 0.62193 |
| 2 | 189 | +0.02655 | +0.02263 | +0.02655 | 0.62115 |
| 3 | 191 | +0.02675 | +0.02256 | +0.02675 | 0.62135 |
| 4 | 192 | +0.02700 | +0.02119 | +0.02700 | 0.62160 |
| 5 | 186 | +0.02565 | +0.02060 | +0.02564 | 0.62025 |
| 6 | 189 | +0.02610 | +0.02174 | +0.02610 | 0.62070 |
| 7 | 192 | +0.02638 | +0.02116 | +0.02638 | 0.62098 |
| 8 | 189 | +0.02548 | +0.02184 | +0.02548 | 0.62008 |
| 9 | 191 | +0.02507 | +0.02147 | +0.02507 | 0.61967 |
| 10 | 189 | +0.02393 | +0.02182 | +0.02392 | 0.61853 |
| 11 | 188 | +0.02413 | +0.02065 | +0.02413 | 0.61873 |
| 12 | 191 | +0.02421 | +0.02072 | +0.02421 | 0.61881 |
| 13 | 190 | +0.02462 | +0.02083 | +0.02462 | 0.61922 |
| 14 | 188 | +0.02307 | +0.01947 | +0.02306 | 0.61767 |
| 15 | 188 | +0.02163 | +0.01867 | +0.02163 | 0.61623 |

## 資料

- 去重後 WikiText-103 train，排除 WikiText-2 與先前全部 OLMoE 文章，可用 27050 篇；單一隨機順序（seed 18191），前 32 篇接受者為 selection，其後 n 篇為 final（整篇隔離）。

## Selection

- 評估 FP32、aggressive、16 個保護配置（正常 dynamic route）。
- G：layer 1–15、位置 0–126，S0 上 ||pq[S0]/Σ − p0[S0]/Σ||₁ 等權平均；診斷指標，非 loss proxy。
- 規則：G < G_aggr、CE < CE_aggr、ΔR > 0 者中取 (G_aggr − G)/ΔR 最大；同分依 CE、R、層號。無合格者則如實報告並停止。

## BC 同資源對照

- min 原 BC，size ≤ M_target 且 BOPs ≤ P_target（兩個分量都限制）；修正後 mp_milp，時限 300 s；非證明最優則標示 incumbent。

## Final

- 凍結 aggressive、route-focused、同資源 BC；n = min(300, 可用, max(64, ceil((1.96 s／0.002)²)))，s 來自 selection（可能偏樂觀）。
- 主要：Δ_primary = CE(route-focused) − CE(BC)，95% CI；次要（Bonferroni ×2）：route-focused − aggressive、BC − aggressive。

## 判讀

- A：route-focused better than same-budget BC (CI < 0) and better than aggressive: worth further study; improvement not attributed to routing alone
- B：better than aggressive only: more precision helps, no support for the route-oriented choice
- C：G improves but task loss does not: G is not a sufficient selection criterion
- D：no help from protection: this protection rule is not supported (not extended to activations or other fixes)
- E：CI spans 0: uncertainty reported, no equivalence claim, no switching candidates
- no_change：even if successful, BC is not changed automatically

## 控制

- old windows (run_olmoe_route_dynamic_prop 0-127): FP32 baseline (S0, w0, p0, CE) and aggressive dynamic CE / ids / weights reproduce that run bitwise
- candidates differ from aggressive only in layer-l weight formats (int<8 -> int8); activations, ranges, router untouched; config completeness for every config
- quantization-off forward_measure == FP32 baseline bitwise (first chunk); reruns of the chosen and BC configs on the first chunk bitwise; replay of their recorded ids / weights through run_olmoe_dynamic_route.forward bitwise
- routers at layers <= l unchanged for protect_l (bitwise p[S0]); recorded routes differ from S0 (dynamic, no FP32 route substitution); finite values; resource recomputation
- failure: fix and record before interpretation
