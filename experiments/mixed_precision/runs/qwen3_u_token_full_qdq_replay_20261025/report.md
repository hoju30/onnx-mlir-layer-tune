# 原變異項＋U_token：完整 Down-input QDQ 既有資料回放

這是已觀察資料上的探索性回放。沒有新 inference、沒有新文章、沒有修改正式 criterion。

full QDQ 指每次一層的完整 Down input 量化誤差 e；不是只有平均分量，也不是 input/output 一起量化或完整模型配置。

方法：原 C=V_old+U_old；新 C=V_old+U_token。全部是未正規化分數，不加 F、不調係數。

原 V/完整 QDQ 損傷來自 qwen3_down_spike_diag_20261023；U_token 來自 qwen3_mean_u_token_validation_20261025 的原 calibration。

| 項目 | 結果 |
|---|---|
| 同位元選擇改變 | 0/84 |
| 舊資料可判定配對 | 78/84 |
| 原／新排對 | 74/78 → 74/78 |
| 兩方法所選損傷平均 | 0.024324350 → 0.024324350 nat/token |
| 所選損傷差 new−old | 0.000000000，逐文章恰為零 |
| 2% margin tie（原／新） | 0/0 |

仍存在的原始明確誤排：L2 int6-p6e1、L2 int8-p8e1、L27 int4-p4e1、L27 int6-p6e1。

因選擇完全相同，對這84個凍結的同位元選擇，換新文章也不會產生兩方法的選擇收益差。無需為此重跑GPU。

## 跨後端控制

舊／新版 U_old 最大相對差 0.289850%；T_old=V_old+U_old 最大相對殘差 0。
橋接 V_old+新版U_old 的同位元選擇改變 0/84；用來區分方法更換與後端數值差異。

## 跨位元描述性結果

336 個跨位元對比中有 0 個排序改變；橋接控制有 0 個。完整值見 cross_bit_pairs_exploratory.csv。
資源不等，不能將跨位元選擇當成同成本品質收益；這只是代理排序診斷。

## 可下的結論與限制

- U_token 的 mean-only 改善沒有改變此批完整 Down-input QDQ 的同位元選擇。
- 這不代表 U 的機制分析無效，也不代表所有 Posit 候選都不準；原變異項仍可能主導總分。
- 不是獨立泛化驗證：舊 loss 已被反覆觀察；原 CI/可判定標記直接沿用來源，不假裝是新盲測。
- 同位元選擇不變，不代表 LP/MILP 配置不變：絕對分數、跨位元及跨層資源取捨仍可能改變，尚未搜尋／驗證。
- 不直接整合 pipeline，也不宣稱模型PPL收益。後續若檢查配置，需要完整 x/y 成本表與資源限制，不能把本 x-only 表直接當 native C。

產物：analysis_plan.json、scores.csv、same_bit_pairs.csv、cross_bit_pairs_exploratory.csv、per_article_selected_losses.csv、summary.json、timing.json。
