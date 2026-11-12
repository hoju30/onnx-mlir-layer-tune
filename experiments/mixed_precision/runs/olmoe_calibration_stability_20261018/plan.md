# 預先登記：跨 calibration 文本的一階項穩定性與 expert 覆蓋診斷

時間：2026-10-18T00:35:03+0800。本檔為 plan.json 的摘要；以 plan.json 為準。

## 問題

- A：一階近似在新 calibration 上本身不準。
- B：一階近似準確，但不同文本的實測量化偏好不同。
- C：expert 觀測 token 少使分數與選擇不穩定（待驗證假說，不預設成立）。

## 資料（含資料不足修訂）

- 排除先前所有 OLMoE 資料（含 D、V、H 整篇文章）後，validation 剩 18 篇、test 剩 27 篇（≥256 token），共 45 篇；J 16 篇 + calibration 40 篇 = 56 篇不足。此計數在本 plan 前完成，未使用 route、分數或 loss。
- 修訂：8 批 calibration 改由 WikiText-2 raw train（同一來源，hub LFS sha256 核對）抽取；J 仍由 validation/test 各 8 篇抽取。
- 原 calibration（validation 開頭 1280 token）來源與組成不同，新舊差異不單獨歸因於「換文本」或「多樣性」。
- J：16 篇 × 4 窗口，seed 18011；calibration：8 批 × 5 篇 × 2 窗口（每批 1270 預測 token），seed 18021，抽樣順序決定批次。
- D、V、H 已看過，只作回顧。

## 分數

- 每批重新估計 B（BC/1270）與 F（Σ_s〈g_s, Q(W)−W〉/1270，FP64 累加）；聯合候選加總。
- 比較 B、F、F+0.5B（主要）、F+B（敏感度）；不截斷、不調係數、不事後選方法。B 不是 Hessian。
- 某批無 token 的 expert 標為「無觀測」：不進入忠實度、穩定性與覆蓋分析；主要終點中各方法同為名稱 tie-break，差為 0，另報只含有觀測格的次要版本。

## 主要終點

- Δ = J loss(同批 F+0.5B 選擇) − J loss(同批 BC 選擇)；63 群組 × 8 批等權平均，再以 J 文章為單位。
- 95% CI：10000 次雙向 bootstrap（批次重抽 + J 文章依 validation/test 分層重抽），seed 18031；同時列出每批結果。只有 8 批，批次部分的不確定性估計粗糙。

## 其他預先指定分析

1. 同批忠實度；2. 跨批穩定性（含 BC 與實測）；4. 以各批實測 loss 當診斷選擇器的轉移分解；5. 同 expert 內覆蓋與不穩定的關聯（連續 token 數、置換檢定、Holm）；6. L15.E17、L15.E59、L8.E33 案例。
- 判讀指引（非門檻）見 plan.json → analyses.interpretation_guides。

## 不做

- 不依 J 調係數；不加 coverage 門檻、shrinkage、fallback；不排除 expert；不把多批平均分數包裝成新方法；不改預設 pipeline；不做完整配置。
