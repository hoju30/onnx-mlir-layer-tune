# 預先登記：dynamic 選擇下恢復 FP32 相對比例

時間：2026-10-18T21:24:18+0800。以 plan.json 為準。

- L11：S = 自身 top-8(p)，weights = p[S]（完整 dynamic，比較對象）。
- Ldprop：S = 自身 top-8(p)；weights = p0[S] × (m／m0S)，m = Σp[S]（當下總和），m0S = Σp0[S]（FP32 對這組 expert 的機率和）；float32；不 normalize 到 1、不改 logits、不 clip；零或非有限值即停止。
- L00：S0、p0[S0]（完全 fixed，參考）。
- 各自完整 forward；aggressive 配置；router FP32；GPU QDQ。使用 FP32 p0，屬診斷，不可部署。

## 資料

- WikiText-103 train，與上兩輪相同的去重規則，另外排除已用的 103 篇；新隨機順序 seed 18141；近重複也比對已用文章。

## 樣本數

- Pilot：上兩輪用過的窗口 0–127（32 篇舊文章）；n = min(300, max(64, ceil((1.96 s／0.0015)²)))，s 為 Ldprop − L11 的文章級 SD。

## 終點

- 主要：Δ_dprop = CE(Ldprop) − CE(L11)，95% CI；負值 = 有幫助。
- 次要（Bonferroni ×2，97.5%）：E_total = L11 − L00（重現；對 ε）；Residual_dprop = Ldprop − L00（±0.002 等價或 >0）。

## 判讀

- helps：Delta_dprop CI entirely below 0: FP32 proportions on the self-selected set reduce the dynamic damage
- enough：and Residual_dprop inside +-eps: back to the fixed level within tolerance
- partial：Residual_dprop positive: proportions do not remove all dynamic damage (selection / cascade remains)
- null：Delta_dprop CI contains 0: no claim; report the excluded improvement range
- hurts：Delta_dprop CI entirely above 0: reported as is
- scope：uses FP32 p0: a diagnostic of where the damage sits, not a deployable correction

## 控制

- pilot: baseline (S0, w0, p0, CE), L00 and L11 CE reproduce run_olmoe_route_mass_restore windows 0-127 bitwise
- Ldprop: weights == float32(p0[S] * scale) exactly; recorded m0S and m equal float32 sums (<= 1e-6); sum(weights) / m relative < 1e-5; weights / p0[S] constant per token (spread < 1e-5); recorded current top-8 weights descending (own topk); ids in range and distinct; share of sets differing from S0 and from L11 reported
- quantization off (first chunk): Ldprop == FP32 bitwise, scale all 1; rerun of Ldprop on the first chunk bitwise; replay of Ldprop ids / weights through run_olmoe_dynamic_route.forward bitwise (all windows)
- config completeness, router untouched, hooks on expert projections only, finite values
- any failure: fix and record before analysing losses
