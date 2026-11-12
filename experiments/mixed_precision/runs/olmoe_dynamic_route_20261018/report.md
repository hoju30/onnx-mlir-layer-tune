# OLMoE：固定 BC knee 配置的 fixed-route／dynamic-route 對照

日期：2026-10-18
腳本：`run_olmoe_dynamic_route.py`（plan → pilot → data → capture → eval → analyze）
輸出：本目錄

**本輪沒有改動的東西**
- 配置、criterion、搜尋、量化器、range、calibration。
- 沒有加入一階項，也沒有做逐層干預。

**流程**
- 從 plan 到分析，每個階段用的都是同一份腳本（sha `cdaad80e…`），沒有任何預登記修訂。

**標記**
- [預先登記]：寫在 `plan.json` 的分析。
- [描述]：不做推論檢定的描述統計。
- [推論]：由資料推得、尚未直接驗證的判斷。

## 結論

**同一個 BC knee 配置改用 dynamic route 後，損傷點估計多約 10%：+0.0013 nat/token。但以預先登記的 ε = 0.002 判定為「不確定」：既不能宣稱等價，也沒有證據顯示超過容忍界線。**

**主要終點** [預先登記]
- Δ_route = CE(q_dyn) − CE(q_fix) = **+0.00130 nat/token**，128 篇文章、512 個窗口。
- 95% CI [−0.00015, +0.00277]；90% CI [+0.00006, +0.00254]。
- 90% CI 超出 +ε，但下界沒有超過 +ε，所以判定為「不確定」。
- 95% CI 含 0，不代表兩者相同。

| 條件 | CE（nat/token） | PPL | 相對 FP32 |
|---|---:|---:|---:|
| FP32 dynamic（基準） | 2.98001 | 19.688 | — |
| FP32 fixed（控制） | 2.98001 | 19.688 | 逐位元相同 |
| BC 量化 fixed | 2.99295 | 19.944 | +1.30% |
| BC 量化 dynamic | 2.99425 | 19.970 | +1.43% |

| 指標 | 平均 | 95% CI | 90% CI |
|---|---:|---|---|
| D_fixed = CE(q_fix) − CE(FP32) | +0.01294 | [+0.01091, +0.01508] | [+0.01126, +0.01472] |
| D_dynamic = CE(q_dyn) − CE(FP32) | +0.01424 | [+0.01196, +0.01666] | [+0.01233, +0.01625] |
| **Δ_route** | **+0.00130** | [−0.00015, +0.00277] | [+0.00006, +0.00254] |
| D_dynamic / D_fixed | 1.10 | [0.99, 1.23] | — |

**五個問題**

1. **現有 BC knee 在 dynamic route 下是否仍可接受？**
   - 損傷從 PPL +1.30% 變成 +1.43%，仍和 fixed-route 驗證時同一量級。
   - 但在 ε = 0.002 下無法確認等價。「可接受」要看你能容忍多少額外損傷：若容忍 +0.003 nat/token，95% 上界 +0.0028 在界內；若容忍 0.002，目前不足以下結論。
2. **fixed-route 低估或高估了多少損傷？**
   - 點估計是低估約 10%（0.0013／0.0142）。
   - 區間顯示可能從沒有低估，到低估約 20%（比例 CI [0.99, 1.23]）。
   - 本輪資料不支持「fixed-route 高估」。
3. **路由改變是普遍現象，還是集中在少數層或文章？** 普遍，但幅度小、而且集中在選擇邊界附近的 token。
   - 第 1–15 層，每層有 21–29% 的 token 換掉至少 1 個 expert。第 0 層完全不變，因為第 0 層 router 的輸入還沒經過任何量化。
   - 換掉的 expert 平均只占 top-8 的約 3%（重疊率 96–97%）。
   - 各 expert 的負載幾乎不變：dispatch 分布的 TVD 只有 0.2–0.4%。
   - 128 篇文章的路由變動比例都在 0.22–0.29 之間，沒有特別集中的文章或層。
4. **有沒有數值或執行路徑差異干擾結論？** 沒有找到。
   - 所有數值控制都逐位元通過，包括歷史重現、FP32 fixed 和 dynamic 一致、重跑可重現、路徑等價（見下節）。
   - 每個窗口的 CE 仍受已知的量化混沌放大影響（1 ulp 擾動造成單窗口 SD 約 0.0095），這已包含在 bootstrap 變異中。
5. **下一步該驗證其他資源點，還是先研究路由相關損傷？**
   - 建議先在**其他預先固定的資源點**（特別是更激進的點）重複同一個 fixed／dynamic 對照，並用更大的樣本。
   - 在 knee 上，額外損傷只有約 10%，而且無法和 0 或 ε 明確區分，現在不值得先做機制拆解。
   - 若較激進的點出現明確超過 ε 的額外損傷，再預先登記「拆開 expert ID 變動與 routing weight 變動」的干預實驗。詳見最後一節。

**結論的範圍**：只限於本模型、這一個 knee 配置、本輪資料與 GPU FP32 推論設定。不能推論到其他資源點，也不能說明 BC 能預測 dynamic-route 下的候選排序。

## 配置與語意（在任何新評估前記錄於 `plan.json`）

**配置**
- 檔案：`runs/olmoe_search_final_20261016/C/configs/milp.json`，sha256 `8ebcc86e…`。
- 它是 search_final B 的修正後 MILP knee：R 0.7071056，BC 18.29196。程式核對過，和 B 的 `search_milp.json` 選出的格式 hash 相同。

**選它的理由**
- OLMoE 的預設 pipeline 仍是 `mp_auto.auto_select` 加 `mp_milp`。
- 2026-10-16 整合進 `mp_milp` 的最小修正，就是 search_final B 的修正後 MILP。
- 這個 knee 已在 512 個新窗口上驗證過：dNLL +0.0195。
- 沒選的版本：bc_p125（修正前的路徑，有 11 個投影不同）、LP knee（研究用求解器）。

**量化**
- `DevQuant` 加上修正後的 int activation 量化，和 search_final C 的評估相同。
- ranges：`runs/olmoe_search_final_20261016/C/ranges.json`，等於 multipoint 的 `ranges.json`。
- 3051 個節點全部套用，格式都在明確的支援清單內，所有量化 activation 都有 range，posit LUT 齊全。
- 另外 21 個沒有 calibration token 的投影，用 bf16 權重和 FP32 activation（歷史語意）。
- 每層 6102 個 activation hook，全部只掛在 expert 投影上。router 權重物件沒有被替換，沒有新增 router 量化。

**四個條件**
- 四者共用同一個函式，只有路由來源不同。
- 同樣的窗口、每批 128 個窗口、GPU FP32、TF32 關閉、matmul precision highest。
- 沒有 padding，causal mask，每個窗口 127 個預測 token，summed CE 用 float64。

| 條件 | 權重 | expert ID 與 routing weight |
|---|---|---|
| fp32_dyn | 原模型 | 自行路由：FP32 logits，對 64 個 expert 做 FP32 softmax，取 top-8，不重新 normalize（和 transformers 相同）。記錄 ID、權重、top-16 logits |
| fp32_fix | 原模型 | **ID 與權重**都用 fp32_dyn 的紀錄 |
| q_fix | BC 配置 | **ID 與權重**都用 fp32_dyn 的紀錄 |
| q_dyn | BC 配置 | FP32 router 在量化模型自己的 hidden state 上計算 |

**歷史語意核對**：先前所有 OLMoE fixed-route 研究都同時固定 ID 和權重（`load_routes` 的 (idx, rw) 加上 `layer_fixed`），和本輪的 q_fix 定義相同，可以直接比較。

## 數值控制（`pilot.json`、`eval.json`）

全部通過。

| 檢查 | Pilot（已看過的 128 個窗口） | 新資料（512 個窗口） |
|---|---|---|
| 重現歷史路由擷取（相同批次形狀 64） | 128／128 個窗口的 route hash 相同；CE 差 0 | — |
| 重現歷史 C truth（同一批次形狀） | q_fix 對 milp、fp32_fix 對 fp32，CE 差都是 0 | — |
| fp32_fix vs fp32_dyn | CE 差 0；最終 hidden state 128／128 逐位元相同 | 差 0；512／512 相同 |
| dynamic 路徑不量化 vs 擷取 | — | 差 0；512／512 相同；路由逐位元相同 |
| q_dyn 重跑 | 128／128 相同；路由相同 | 512／512 相同；路由相同 |
| q_fix 改用 q_dyn 自己的路由 vs q_dyn | 128／128 相同 | 512／512 相同 |
| 有限值、每窗口 127 token、routing weight 有限 | 是 | 是 |

- 「把 q_dyn 自己的路由餵給 fixed 路徑，結果逐位元等於 q_dyn」表示兩個條件之間只有路由來源不同，沒有 batching、kernel 或 QDQ 的差異。
- **批次形狀**：同一批窗口改用 64 和 128 個一批，FP32 的路由權重最後幾個位元就會不同，每窗口 CE 差 ≤ 1.7e-4（約 1.3e-6 nat/token）。本輪所有條件都用相同的批次形狀，避免了這個干擾。
- **環境**：torch 2.14.0+cu130，RTX 3060。nvidia-smi 回報驅動與函式庫版本不一致（使用者層 595.99、核心模組 595.84），但上述歷史重現逐位元成功，表示數值沒有受影響。

## 資料（`segments.json`、`data_freeze.json`）

**文章邊界規則**（預先登記）
- 只有前後都是空行的 `" = 標題 = "` 才算文章開頭。這樣 train／validation／test 恰好是官方的 600／60／60 篇。
- 寬鬆規則會多抓 train 29 處、test 2 處表格小標或公式。這些現在都併入所屬的文章。

**排除**
- 和先前任何 OLMoE 文本重疊的整篇文章都排除，包括：
  - stage 1 calibration、diagnostic、check；
  - multipoint、solver regression、search_final C、numerics B；
  - D、V、H、J、C1–C8。
- 排除後可用（≥512 token）：validation 10、test 18、train 556。

**樣本數**（預先登記的規則）
- Pilot（search_final C 前 128 個窗口，已看過的資料）中，Δ_route 的 4 窗口區塊 SD 為 0.0101。
- 要讓 90% CI 半寬 ≤ ε/2，需要 274 篇。超過上限，取上限 128 篇。
- 實際 90% CI 半寬約 0.0012：比 ε 小，但比 ε/2 大。

**來源**
- 依原評估來源優先的原則，validation 10 篇和 test 18 篇全部納入（普查，沒有挑選）。
- 其餘 100 篇從 train 均勻抽樣，seed 18051。
- 每篇 4 個連續窗口。
- 分層 bootstrap 依 validation／test／train 三層。

**注意**
- train 占 100／128 篇，整體結果主要反映 train 文章。
- validation 和 test 在排除後已全部用完，之後的 OLMoE 確認集只能用 train。

**分來源結果**

| 來源 | 篇數 | D_fixed | D_dynamic | Δ_route [95% CI] |
|---|---:|---:|---:|---|
| 原評估來源（validation＋test） | 28 | +0.0132 | +0.0135 | +0.00025 [−0.0030, +0.0032] |
| train | 100 | +0.0129 | +0.0145 | +0.00160 [−0.0001, +0.0033] |
| validation | 10 | +0.0155 | +0.0159 | +0.0004 [−0.0048, +0.0038] |
| test | 18 | +0.0119 | +0.0121 | +0.0002 [−0.0040, +0.0042] |

- 原評估來源只有 28 篇，區間很寬，無法判斷是否和 train 不同。
- 點估計的差異（+0.0003 對 +0.0016）不能解讀為來源效應。

## 逐篇文章（`per_article.csv`）[描述]

- 128 篇的 Δ_route：SD 0.0086；中位數 +0.0012；範圍 −0.021 到 +0.024；74 篇為正。
- 不是由少數文章主導：
  - 最大的 5 個正值只占全部正值總和的 18%；
  - 去掉最大的一篇後，平均是 +0.0011。
- 各篇的路由變動比例（每層、每個 token 換掉 expert 的比例）都在 0.22–0.29 之間。
- 它和 Δ_route 的 Spearman 是 −0.12：路由改變多的文章，損傷並沒有比較大。這是描述，不是檢定。

## 逐層路由變動（`per_layer_routing.csv`）[描述]

範圍：q_dyn 對 FP32 dynamic，位置 0–126。

| 層 | top-8 集合相同 | 重疊率 | dispatch TVD | 權重 L1 | 集合相同時的權重 L1 | FP32 margin 中位數（變動／不變） | 最低／最高 margin 五分位的變動率 |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 100% | 1.000 | 0 | 0 | 0 | —／0.048 | 0／0 |
| 1 | 78.8% | 0.973 | 0.0029 | 0.015 | 0.006 | 0.011／0.048 | 47%／0.4% |
| 3 | 74.4% | 0.967 | 0.0036 | 0.023 | 0.009 | 0.018／0.065 | 52%／1.2% |
| 5 | 71.3% | 0.962 | 0.0040 | 0.026 | 0.011 | 0.021／0.069 | 56%／1.9% |
| 8 | 72.4% | 0.964 | 0.0035 | 0.029 | 0.012 | 0.024／0.084 | 58%／1.5% |
| 10 | 78.0% | 0.971 | 0.0021 | 0.028 | 0.014 | 0.028／0.111 | 56%／1.3% |
| 12 | 78.4% | 0.972 | 0.0019 | 0.031 | 0.017 | 0.031／0.124 | 54%／1.7% |
| 15 | 77.0% | 0.970 | 0.0019 | 0.035 | 0.019 | 0.035／0.142 | 58%／2.6% |

完整 16 層見 CSV。

**整體**
- 76.8% 的 (token, 層) 保留完全相同的 top-8 集合；重疊率 97.0%。
- 每層 64 個 expert 在兩種條件下都有被使用，最大的負載占比變化 ≤ 0.22%。

**集中在選擇邊界**
- 依 FP32 第 8／9 名的 logit margin 分成五分位，變動率依序為 50%、35%、20%、9%、1.7%。
- 進入量化 top-8 的 expert，在 FP32 中原本排第 9 的占 70.6%，第 10–12 名占 27.5%，第 17 名以後只占 0.3%。

**權重**
- 即使 top-8 集合不變，權重 L1 仍有 0.006–0.019，而且隨深度增加。
- 也就是說，除了換掉 expert，routing weight 本身也在改變。

[推論]
- 路由改變是普遍、小幅、而且集中在邊界附近 token 的現象。
- 可能的原因是量化誤差在上游累積，讓 router 的輸入偏移，翻轉了原本就接近平手的選擇。
- 本輪沒有拆開「換 expert」和「改權重」各自造成的 loss，無法說明 +0.0013 來自哪一部分。

## 時間成本

所有條件都在同一套 GPU 程式中執行，以下只是本次實驗的成本，不代表部署時的 latency。

| 項目 | 時間 |
|---|---|
| baseline 與路由擷取（FP32 dynamic，512 個窗口） | 28.8 秒（stage 34.4 秒） |
| fixed-route 評估 | q_fix 62.3 秒；fp32_fix 控制 28.5 秒 |
| dynamic-route 評估 | q_dyn 62.4 秒 |
| 控制組 | q_dyn 重跑 62.4 秒；q_fix 改用 q_dyn 路由 62.3 秒；FP32 dynamic 重跑 28.7 秒 |
| pilot（已看過的 128 個窗口，9 次） | 147 秒 |
| 抽資料、plan | 9.4 秒、11.4 秒（含模型檔 hash） |
| 統計分析 | 2.4 秒 |

q_fix 和 q_dyn 的耗時幾乎相同。router 的計算量很小，在這個實作下看不出差異。

## 建議的下一步（假說，需另行預先登記）

**優先：在其他預先固定的資源點重做 fixed／dynamic 對照**
- 資源點：例如 search_final B MILP frontier 上已凍結的 R 0.5–0.6 和 0.8–0.84 點。
- 同一套程式和控制。
- 新的 train 文章：validation/test 已用完。
- 樣本：pilot 的規則估計，要達到 ε/2 精度約需 274 篇，可以考慮每篇用更多窗口。
- 問題：額外損傷是否隨壓縮程度增加。

**若出現明確超過 ε 的額外損傷：拆解干預**

| 條件 | expert ID | routing weight |
|---|---|---|
| (a) | FP32 | 量化模型對同一組 ID 重新計算 |
| (b) | 量化模型 | FP32 router 對這些 ID 的值 |
| (c) | 依 FP32 margin 分層，只放開 margin 小的 token 的路由 | |

另外再預先登記 ε 和主要比較。

**目前不建議**：依路由統計直接修改配置或 criterion；逐層格式消融。

## 已確立與未確定

**已確立**
- 程式路徑正確：
  - fixed 和 dynamic 在不量化時逐位元一致；
  - 兩者之間只有路由來源不同；
  - 歷史 fixed-route 結果可逐位元重現。
- 在這個 knee 上，dynamic route 的損傷點估計比 fixed route 多 +0.0013 nat/token（約 10%），PPL 從 +1.30% 變成 +1.43%。
- 路由改變普遍存在於第 1–15 層：每層約 21–29% 的 token 換掉至少 1 個 expert，但重疊率約 97%，負載幾乎不變，而且集中在 FP32 選擇邊界附近的 token。

**未確定**
- 在 ε = 0.002 下，dynamic 和 fixed 是否等價（判定「不確定」）。
- 額外損傷是否在原評估來源（validation/test）和 train 上不同：原來源只有 28 篇，區間太寬。
- 其他資源點、其他模型，以及原生（native）推論。
- 額外損傷有多少來自換 expert、多少來自權重改變。
- BC 是否能預測 dynamic-route 下的候選排序。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_dynamic_route_20261018      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
../../venv/bin/python run_olmoe_dynamic_route.py --stage plan --out-dir $O
$GPU run_olmoe_dynamic_route.py --stage pilot --out-dir $O       # 約 2.5 分鐘
../../venv/bin/python run_olmoe_dynamic_route.py --stage data --out-dir $O
$GPU run_olmoe_dynamic_route.py --stage capture --out-dir $O
$GPU run_olmoe_dynamic_route.py --stage eval --out-dir $O        # 約 5 分鐘
../../venv/bin/python run_olmoe_dynamic_route.py --stage analyze --out-dir $O
```

**環境**
- GPU：torch 2.14.0+cu130，RTX 3060，TF32 關閉，matmul precision highest。
- CPU：torch 2.14.0+cpu，用於 plan、data、analyze。
- 需要 `gpt2_assets/data/train-00000-of-00001.parquet`（sha256 `e83889ba…`）。

## 檔案

- **預先登記與凍結**：`plan.json`、`plan.md`、`pilot.json`、`segments.json`、`data_freeze.json`、`tokens.npy`。
- **路由**：`routes/fp32_dyn.npz`（ID、權重、top-16 logits、tokens）、`routes/q_dyn.npz`、`capture.json`。
- **評估與分析**：`eval.json`（各條件 CE、控制、稽核）、`analysis.json`、`per_article.csv`、`per_layer_routing.csv`。
- **紀錄**：`timing.jsonl`、`logs/`。
