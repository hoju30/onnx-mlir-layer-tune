# 預先登記：OLMoE aggressive 資源點完整配置 A／B — 原 BC vs Down-input 變異項逐位置配對

時間：2026-10-23T17:20:48+0800。以 plan.json 為準。

## 假說與定義
- 假說：在既有 aggressive 資源點，B 的完整配置 NLL 低於 A。唯一主要比較：Δ = NLL(B) − NLL(A)。
- A：runs/olmoe_proxy_refresh_20261022/table_static.json（sha 6d9c52ef0dcb）＋歷史 q0（runs/olmoe_proxy_refresh_20261022/configs/q0.json，sha 2fe46627745c）。
- B：runs/olmoe_pair_knee_ab_20261023/tables_B.json（sha 0e77ca315f73，上一輪已驗證），只改 Down-input activation 變異項；其餘全部不變；完整 W＋A 搜尋。
- 不做 quantized-state refresh、router fine-tune、spike-only 選點或新係數；不改正式 pipeline。

## 資源點
- 讀 runs/olmoe_proxy_refresh_20261022/plan.json 的 budget：max_bytes = 3804495872.0，max_bops = 39226380910592（R ≈ 0.594603）；不以配置用量重定義，不建 frontier、不重選 knee、不換點。

## 搜尋
- 同一 LP-first：支配剪枝 → LP → 取整＋修復（誤差 ≤ max(0.001·目標, 1e-06·scale) 才接受）→ 否則修正版 MILP（每次 ≤ 300.0 s，最多一次既有 1-ppm 重試，再修復）；A／B 相同；不額外求解。
- 控制：A 表搜尋必須重現歷史 q0（不同則先調查、不開始新資料）；B 的 weight 列與非 Down activation 列等於 A；配置合法、精確資源檢查；A 表與 B 表各自評 A、B 配置。
- B 的解在 B 表上若差於可行的 A 配置 → 標示「搜尋受限」，不據此判定評分失效，不以評估 NLL 選配置。
- 下界：分開報告原始預算的 LP 下界、原始預算的 MILP 嘗試、收緊 1 ppm 重試的 dual bound；最終差距以原始 LP 下界計；重試的 gap 0 不代表原始問題已證明最佳。

## 配置分析（凍結後才抽資料）
- 改變的投影／expert、weight-only／activation-only／兩者、Gate／Up／Down、位元與 INT／Posit／FP 分布、Down activation ≤4／≤5／≤6 bit 比例、原 FP16 Down activation、size／BOPs 用量與餘量；若 A、B 相同則直接回報、不評估。

## 評估
- 600 篇新 WT103 文章 × 4 窗口（seeds 23801／23802）；verified 掃描（計畫時排除 7121 篇，含正常 knee A/B：True；資料階段重新計算並比對，不硬編碼）；排除 WikiText-2 與近似重複；核對 content hash 交集為 0。
- FP32、A、B；原 router、dynamic routing、同一 QDQ 後端；程式控制（q0 在舊批次前 128 窗口）與 B 前 128 窗口重跑。
- 主要：Δ = NLL(B) − NLL(A)，每篇先彙整 4 窗口，文章 paired bootstrap（10000，seed 23811）95% CI。次要：token 加權 NLL、PPL、相對 FP32 損傷、改善／變差文章、相對損傷改善（點估計）、dynamic route 下 BOPs proxy。
- 判讀：CI < 0 → 此點收益確認（權衡成本；下一輪才考慮自動 knee，不改預設）；CI 含 0 → 未確認，保留 BC，配對方向先收斂；CI > 0 → 此點退步，不接入；控制不通過 → 比較受限。

## 成本
- 分開報告讀表／驗證／準備、剪枝與搜尋、最終評估；從頭成本若引用上一輪建表，標示「歷史建表實測＋本輪搜尋實測」。不宣稱 native 整合或 kernel 加速。

## 預算（分鐘）：{'prep': 2, 'search': 15, 'data': 6, 'eval': 20, 'analyze': 5, 'total_gpu': 25}；不因結果追加文章、改配置或門檻；不跨批次相減 NLL。
