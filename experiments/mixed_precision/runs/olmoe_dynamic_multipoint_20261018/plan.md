# 預先登記：OLMoE 多資源點 fixed-route／dynamic-route 對照

時間：2026-10-18T06:10:31+0800。以 plan.json 為準。

## 配置（依既有實際資源選取）

- 來源：search_final B 修正後 MILP frontier（`B/search_milp.json`，預設 pipeline 產生；上一輪用其 knee）。
- aggressive：實際 R 最接近 0.55；relaxed：最接近 0.82；knee：上一輪同一配置。R = max(權重 bytes/int8, BOPs/int8)，用配置本身的資源，不用求解預算；同距離依名稱排序。

| 配置 | frontier 項目 | R | size/int8 | BOPs/int8 | BC | sha256 |
|---|---|---:|---:|---:|---:|---|
| aggressive | `cand03_R0.5946031` | 0.5946031 | 0.594600 | 0.594603 | 42.1827 | `e51dd10717a1…` |
| knee | `cand20_R0.7071056` | 0.7071056 | 0.690593 | 0.707106 | 18.2920 | `8ebcc86eea94…` |
| relaxed | `cand05_R0.8408957` | 0.8408957 | 0.791544 | 0.840896 | 9.1118 | `8e98082ed18a…` |

- 配置間不同節點數：{"aggressive_vs_knee": {"any": 2576, "weight": 2280, "act": 759}, "knee_vs_relaxed": {"any": 2687, "weight": 2457, "act": 727}, "aggressive_vs_relaxed": {"any": 3036, "weight": 3015, "act": 1352}}。
- 21 個無 calibration token 的投影維持 bf16 權重 + FP32 activation。

## 資料與樣本數

- WikiText-2 train，嚴格文章邊界，整篇排除先前所有 OLMoE 文本（含上一輪 128 篇）；可用 456 篇（≥512 token）。
- n = ceil((z·s/h)²)，z = Φ⁻¹(1−0.05/3) = 2.1280（Bonferroni 決策區間），s = 上一輪 knee 的文章級 Δ_route SD 0.00858，h = 0.001；需 334 篇，取上限 300 篇（1200 窗口）。
- 預期決策區間半寬 0.00105、95% 半寬 0.00097；aggressive 的變異可能更大。
- 均勻抽樣、每篇 4 個連續窗口，seed 18071；三個配置共用同一批窗口。

## 推論與控制

- 沿用上一輪 `forward`；每批 128 窗口、尾批為剩餘窗口，所有條件一致；GPU FP32、TF32 關閉。
- 控制：fp32_fix = fp32_dyn（全部）；動態路徑不量化重跑 = fp32_dyn（控制子集 0–127）；各配置 q_dyn 重跑（控制子集）；各配置 q_fix 改用 q_dyn 路由 = q_dyn（全部）；配置完整性與 router 未動；重現上一輪 knee 前 128 窗口（程式控制，舊資料不進確認集）。

## 統計

- 10000 次 paired article bootstrap（seed 18081），所有配置、條件、對比共用同一組重抽。
- A：每配置 Δ_route；ε = 0.002；決策區間為 96.67%（Bonferroni ×3）：落在 [−ε, ε] → 等價；下界 > ε → 超過容忍；上界 < −ε → 明顯較好；否則不確定。另列未校正 90%／95%。
- B：主要對比 T = Δ(aggressive) − Δ(relaxed)，95% CI；次要 aggressive−knee、knee−relaxed 用 97.5%（Bonferroni ×2）。
- knee 重現（描述）：新 knee 的 95% CI 是否包含上一輪點估計 +0.00130。

## 預先判讀

1. all three equivalent within eps: fixed-route approximates dynamic in the tested range only
2. a config exceeds +eps: next round decomposes expert-id vs routing-weight effects for that config first; no criterion change now
3. some or all uncertain: report the precision and the excluded effect range; no pass claim, no adding samples
4. dynamic below fixed: kept as is; check controls first, then discuss fixed-route over-statement
5. quality order of the three configs changes under dynamic: recorded as a phenomenon; this confirmation set is not used to pick a deployment config

## 不做

- 不改 BC、不加一階項、不重搜、不修配置、不依結果選 knee、不加樣、不做路由機制拆解。
