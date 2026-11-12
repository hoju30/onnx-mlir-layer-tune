# 預先登記：OLMoE aggressive 配置的 expert 選擇／routing weight 干預

時間：2026-10-18T17:15:49+0800。以 plan.json 為準。

## 配置

- `runs/olmoe_dynamic_multipoint_20261018/configs/aggressive.json`（sha256 `e51dd10717a1…`，R 0.5946），與上一輪相同；未進成本表的 21 個投影仍為 bf16 權重 + FP32 activation；router FP32。

## 四個條件（各自完整 forward；p、S 來自各自 hidden state）

- 基準：完整 FP32 dynamic，保存每層每 token 的 S0（top-8）與 p0（全部 64 個 softmax 機率）。
- L00：IDs = S0，weights = p0[S0]。
- L01：IDs = S0，weights = p[S0]。
- L10：IDs = S（自身 top-8），weights = p0[S]（固定 FP32 機率函數，取值與總和可隨 S 改變）。
- L11：IDs = S，weights = p[S]。
- 不重新 normalize；四條件共用 `run_olmoe_dynamic_route.forward` 的算術；每批 128 窗口、尾批為剩餘；結果是端到端干預，不是同一 hidden state 的局部效應。

## 控制

- pilot (old multipoint windows 0-127): FP32 baseline S0 / top-8 weights / CE equal the multipoint capture bitwise; L00 == multipoint aggressive q_fix and L11 == q_dyn (CE bitwise)
- quantization off (control subset = first chunk): L00, L01, L10, L11 each == FP32 baseline (CE, hidden bitwise)
- reruns of every condition on the control subset bitwise (CE, hidden, ids, weights)
- replay: every condition's recorded ids / weights through run_olmoe_dynamic_route.forward (fixed routes) reproduces the condition (CE, hidden bitwise, all windows)
- semantics: L00 ids = S0 and weights = p0[S0] = baseline weights; L01 ids = S0; L10 weights = p0 gathered at its own ids; weight sums < 1 (no renormalization); ids in [0, 64) and distinct; L10 vs L11 and L01 vs L11 differ (evidence of own hidden states); config completeness, finite values, router untouched, 127 tokens per window
- any failure: fix and record before analysing losses

## 樣本數

- Pilot：上一輪 multipoint 的窗口 0–127（32 篇舊文章，只作控制與變異估計）。
- n = min(可用篇數, ceil((z·s_max/h)²))，z = Φ⁻¹(1−0.05/6) = 2.3940，h = 0.002，s_max 為 pilot 中 E_weight、E_ID、I 的最大文章級 SD；可用 156 篇。
- 若受可用篇數限制，報告可達精度；本輪不加新來源。

## 對比與統計

- 主要族：E_weight = L01−L00、E_ID = L10−L00、I = L11−L10−L01+L00；Bonferroni ×3（98.33% 區間），另列 95%。
- E_total = L11−L00：重現指標，95% CI 對 ε = 0.002 與上一輪 +0.00753。
- 次要：E_weight − E_ID（95%）；描述：L11−L01、L11−L10。
- 10000 次 paired article bootstrap（seed 18101）。不預設正負，不當作非負占比。

## 判讀

- weight_first：E_weight Bonferroni CI lower > 0 and both E_ID and I Bonferroni CIs inside [-eps, +eps]
- id_first：E_ID Bonferroni CI lower > 0 and both E_weight and I Bonferroni CIs inside [-eps, +eps]
- joint：I Bonferroni CI excludes 0: study joint / cross-layer accumulation first, no split into two penalties
- overlap：E_weight and E_ID both Bonferroni-positive, each point >= 0.5 E_total point, I upper < 0: overlapping effects; do not claim either explains all damage
- undetermined：otherwise: mechanisms not separated; no story from point estimates
- precedence：checked in the order weight_first, id_first, overlap (when its condition holds), joint, undetermined
- replication：E_total differing from the previous round: report sources and uncertainty, no re-picking

## 不做

- margin 分層、逐層開關、提高精度、改 normalization、改 BC／搜尋／配置。
