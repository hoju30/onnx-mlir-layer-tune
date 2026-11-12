# 預先登記：固定 FP32 expert 選擇的 routing weight 2×2 恢復診斷

時間：2026-10-18T21:11:15+0800。以 plan.json 為準；在計算 Lprop 之前寫定。

| | 比例不恢復 | 比例恢復（w0／m0） |
|---|---|---|
| 總和不恢復 | L01：w = p[S0] | Lprop：w0 × (m／m0) |
| 總和恢復（m0） | Lmass：w × (m0／m) | L00：w0 |

- w 來自各條件自身 hidden state；m = sum(w)、m0 = sum(w0)，float32、依 S0 順序；不 normalize 到 1、不改 logits、不 clip。
- L00／L01／Lmass 用上一輪未改動的 forward5 重算，須與上一輪逐位元相同；只有 Lprop 是新量測。
- 同一配置、同 103 篇 WikiText-103 文章（412 窗口）、同 FP32 基準；GPU QDQ，不代表 native。

## 終點

- 主要：Δ_prop = CE(Lprop) − CE(L01)，95% CI；負值 = 恢復 FP32 比例有幫助。
- 次要族（Bonferroni ×4，98.75%）：比例主效應、總和主效應、交互作用、Residual_prop = Lprop − L00（±0.002 等價或 >0）。
- 描述：Δ_restore、Residual（上一輪已知）、各格 CE／PPL。不做「總和占 X%、比例占 Y%」。

## 判讀

- proportions_help：Delta_prop 95% CI entirely below 0: FP32 proportions reduce the weight-path damage
- proportions_enough：and Residual_prop inside +-eps (98.75%): restoring proportions alone returns to the fixed level within tolerance
- proportions_partial：Delta_prop below 0 but Residual_prop positive: proportions explain part only
- interaction：interaction CI excludes 0: sum and proportions do not act additively
- null：Delta_prop CI contains 0: no claim; report the excluded improvement range; then the damage is more likely carried by later-layer cascades than by the weight values of one layer (hypothesis)
- scope：FP32 ids and FP32 w0 are used: no transfer to full dynamic routing or to a deployable fix is claimed

## 控制

- FP32 baseline recomputed == run_olmoe_route_mass_restore baseline (S0, w0, p0, CE) bitwise
- L00 / L01 / Lmass == that run (CE all windows, ids, weights) bitwise
- Lprop: ids = S0; weights == float32(w0 * scale) exactly; recorded m0 and m equal float32 sums (<= 1e-6); sum(weights) / m relative < 1e-5; weights / w0 constant per token (relative spread < 1e-5); sums not 1
- quantization off (first chunk): Lprop == FP32 bitwise; rerun of Lprop on the first chunk bitwise; replay of Lprop ids / weights through run_olmoe_dynamic_route.forward bitwise (all windows)
- config completeness, router untouched, hooks on expert projections only, finite values
- any failure: fix and record before analysing losses
