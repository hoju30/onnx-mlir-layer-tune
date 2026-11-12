# OLMoE：固定 q0、只 fine-tune router

日期：2026-10-22
腳本：`run_olmoe_router_ft.py`（plan → controls → train → freeze → data → eval → analyze）
輸出：本目錄

**本輪的範圍**
- 這是獨立的 router-only FT 試驗，不是 GEMQ 的重現。
- 沒有重建 BC 表、沒有重新搜尋、沒有修改 criterion，也沒有做「FT 後再 refresh」；沒有整合到預設 pipeline。
- 全模型是 PyTorch GPU 上的 QDQ，不代表 native kernel 的結果。

## 結論

**依預先登記的主要終點，router FT 明確改善了 q0：ΔFT = −0.0824 nat/token，95% CI [−0.0856, −0.0793]。但 q0+FT 甚至比原始 FP32 模型還好 0.048 nat/token，所以這個收益主要不是「修復量化損傷」。最可能的解釋是 router 適應了 WikiText 的文本分布與格式：calibration 和評估文章都是 WikiText 格式的 Wikipedia 文本。這個收益能否推廣到其他文本，本輪完全沒有測。**

| 條件 | NLL | PPL | 相對 FP32 的損傷 [95% CI] |
|---|---:|---:|---|
| A：FP32 dynamic | 2.95227 | 19.149 | — |
| B：q0（router 未訓練） | 2.98663 | 19.819（+3.50%） | +0.03436 [+0.03283, +0.03590] |
| **C：q0 + router FT（第 20 步）** | **2.90420** | **18.251（−4.69%）** | **−0.04807 [−0.05155, −0.04461]** |
| D：q1（完整 refresh，router 未訓練） | 2.98452 | 19.777（+3.28%） | +0.03225 [+0.03075, +0.03385] |

600 篇新的 WikiText-103 文章 × 4 個窗口；四個條件都用自己的 dynamic route。

| 對比 | 平均 | 95% CI | 較好／較差的文章數 |
|---|---:|---|---|
| **ΔFT = NLL(q0+FT) − NLL(q0)（主要）** | **−0.08243** | **[−0.08556, −0.07925]** | 597／3 |
| Δrefresh = NLL(q1) − NLL(q0)（次要，探索） | −0.00211 | [−0.00305, −0.00116] | 351／249 |
| ΔFT_vs_refresh = NLL(q0+FT) − NLL(q1)（次要，探索） | −0.08032 | [−0.08341, −0.07714] | 594／6 |

**判定**
- 主要終點：**確認 router FT 改善**（CI 上界 < 0）。
- 等價判定（±0.002）不適用，因為 CI 遠在範圍之外。

## 回答五個問題

**1. router-only FT 是否在新文章上改善 q0？**
- 依預先登記的終點，是，而且幾乎每篇文章都改善（597／600）。
- 但這個結論必須和下面的疑點一起看：q0+FT 比 FP32 還好 0.048 nat/token。對一個量化後的模型而言，這不可能只是修復量化損傷。
- [推論] router 在 calibration 上學到了 WikiText 特有的分布與格式（例如 ` @-@ `、標點前的空格），而評估文章也來自同一個語料家族，所以收益在評估上也出現。
- 支持這個推論的觀察：
  - calibration loss 從 2.550 降到 1.105（20 步），遠低於 FP32 在 calibration 上的 CE；
  - 改善幅度（0.082）是 q0 量化損傷（0.034）的 2.4 倍。
- 本輪**沒有**測 FP32 + router FT 這個對照，也沒有測非 WikiText 的文本，所以無法把收益拆成「量化修復」和「領域適應」兩部分；也無法排除在其他文本上變差的可能。

**2. 效果量、信賴區間和 PPL 差多少？**
- ΔFT = −0.0824 nat/token，95% CI [−0.0856, −0.0793]，文章級 SD 0.039。
- PPL 從 19.819 降到 18.251（−7.9%）。

**3. 在同一批文章上，和 q1 相比如何？**
- q1 比 q0 好 0.0021 [0.0012, 0.0030]，和前一輪確認實驗的 −0.0012 方向一致；這是第三批獨立文章。
- q0+FT 比 q1 好 0.080。
- 但兩者性質不同：refresh 是同一個量化流程下的配置選擇；FT 則多了一個在 calibration 文本上的訓練目標。直接比大小，會把領域適應也算成 FT 的優點。這兩項都只是探索性比較。

**4. 額外 offline 時間與實際路由的計算成本如何？**
- **FT 的額外時間**：342.6 秒（單次量測）。

  | 項目 | 秒 | 占比 |
  |---|---:|---:|
  | checkpoint 逐層載入 | 126.1 | 37% |
  | q0 量化 | 140.8 | 41% |
  | 訓練 forward | 28.7 | 8% |
  | backward／重算 | 46.6 | 14% |
  | optimizer update | 0.14 | — |
  | router 存檔 | 0.01 | — |
  | 未分類 | 0.23 | — |

  每一步約 17 秒，其中 78% 花在重複載入和量化：每步要逐層載入並量化整個模型兩次（forward 一次、逐層 backward 一次），20 步共 40 次。若能讓量化後的 expert 權重常駐，可以大幅縮短；但以 FP32 存放約需 25.8 GB（16 層 × 64 experts × 3 個投影，推算值），超過 GPU 的 12 GB，也會占去機器 39 GB RAM 的大半。
- **和歷史數字比**：前次 refresh 的額外約 108 秒是在不同條件下的單次量測，所以不宣稱精確倍數；大致上 FT 的額外成本約為 refresh 的 3 倍。
- **峰值記憶體**：GPU 4.75 GB；CPU RSS 27.4 GB（含 mmap 的 checkpoint 頁面；機器總 RAM 約 39 GB）。
- **實際路由的計算成本**（評估的 2400 個窗口，全部 128 個位置）：

  | | 只算搜尋節點，相對全 int8 | 含未搜尋 expert，相對全 int8 | 送到未搜尋 expert 的 token-slot |
  |---|---:|---:|---:|
  | q0 | 0.5611 | 0.5911 | 158,213 |
  | q0+FT | 0.5598 | 0.5927（+0.27%） | 172,288（+9%） |
  | q1 | 0.5617 | 0.5917 | 157,926 |
  | FP32 路由 | — | — | 154,735 |

  - 搜尋用的 BOPs proxy（以 FP32 calibration 的 routed rows 計算）：q0 和 q0+FT 都是 3.9221e13（router 不影響這個 proxy），q1 是 3.9225e13。
  - 格式和儲存大小相同，但 FT 後的 router 把更多 token 送到未搜尋的 bf16／FP32 expert，所以實際計算量略增。不能宣稱「計算成本完全不變」。
  - 「含未搜尋 expert」那一欄是分析後才補算的描述性數字（`routing_cost_full.json`）。

**5. 是否值得下一輪測「router FT 後再更新 proxy」？**
- **現在還不值得**。要先確認 FT 收益的性質：
  1. **同配方的 FP32 + router FT 對照**：在同一批評估文章上，比較 (q0+FT − q0) 和 (FP32+FT − FP32)，才能分出量化特有的部分。
  2. **非 WikiText 的評估文本**：例如其他網頁或程式語料，檢查有沒有領域外的退化。
  3. **calibration 和評估的格式分開**：例如改用非 WikiText 的 calibration。
- 若量化特有的部分確實存在，再考慮 FT 之後的 refresh。FT 會大幅改變路由：50% 的 (token, 層) expert 集合和 FP32 不同，q0 只有 30%。這時 FP32 路由下建的 BC proxy 可能已經不符合實際的工作狀態。

## 訓練設定（預先登記，`plan.json`）

**參數與資料**
- 只更新 16 個 router 矩陣 `model.layers.*.mlp.gate.weight`（64 × 2048），不是 expert 的 `gate_proj`。
- 固定的東西：q0 每個投影的格式、量化後的 expert 權重、ranges、scale、zero-point、posit LUT、attention、norm、embedding、LM head、21 個未搜尋投影。
- 資料：stage 1 的 10 × 128 calibration tokens（WikiText-2 validation 開頭）。

**優化**
- AdamW，lr 1e-4（固定），betas (0.9, 0.999)，eps 1e-8，weight decay 0。
- 所有 router 一起做 global norm clip 1.0。
- 20 次 update，每次用全部 10 個窗口（一個 batch，不切 microbatch）。
- loss = next-token mean CE（Σ summed CE／1270）；沒有 distillation、load-balancing 或額外的 router loss。
- seed 22321。
- **第 20 步的 router 是唯一的主要結果**，沒有依評估結果挑步數或設定。

**梯度**
- 每一步都用當前 router 做 dynamic forward：FP32 logits → float32 softmax → top-8 → 不重新 normalize 的權重。
- backward 是逐層重算：每層用同一組 router 重算一次。
- **routing weight 保留梯度連結**，梯度會傳到 router，也會經 router logits 傳回較前層的 hidden state。
- top-k 的 index 是離散選擇，不對 expert 切換本身求導。
- activation QDQ 的 forward 是修正後的量化器，backward 是 identity STE。這是 surrogate，不是量化器的真實導數。
- expert 權重固定，但輸入梯度照常傳遞。
- 所有層的梯度都算完後，才做一次 optimizer update。

**訓練過程**（`training_log.csv`；都是 calibration 上 update 前的 loss）

| 步 | loss | 梯度 norm（clip 前） | update L2 |
|---:|---:|---:|---:|
| 1 | 2.5501 | 1.71 | 0.145 |
| 5 | 2.1222 | 1.13 | 0.121 |
| 10 | 1.7348 | 0.97 | 0.112 |
| 15 | 1.4236 | 0.94 | 0.106 |
| 20 | 1.1560 | 0.80 | 0.100 |

- 第 20 次 update 之後，calibration loss 為 1.1046。
- 前 9 步的梯度 norm > 1，被 clip。
- 每一步的逐層重算和 forward 的差都是 0，top-k 不一致的數量也是 0。
- 各層 router 的相對變化（Frobenius）為 4.3%–10.4%，大致越深越大：第 0 層最小，第 14 層最大。
- 只有 training loss 下降不能算成功；本輪的成功判定依據是新文章上的 NLL。

## 控制（全部通過；`controls.json`、`train.json`、`eval.json`、`controls_summary.json`）

| 控制 | 結果 |
|---|---|
| 零更新：訓練 forward（checkpoint router）vs `run_olmoe_dynamic_route.forward(q0)`，同一 batch | CE 差 0；10／10 個窗口的 hidden state 逐位元相同；路由逐位元相同 |
| 梯度：2 層子堆疊（第 0–1 層、2 個窗口、q0＋STE），完整 autograd vs 逐層 backward | router 梯度和輸入梯度都**逐位元相同**；router 梯度不為 0 |
| 初始梯度算兩次 | 逐位元相同 |
| 全模型的逐層 backward | 重算差 0；top-k 一致；16 層 router 梯度都有限且不為 0 |
| STE 單元測試 | 通過 |
| optimizer 參數 | 只含 16 個 router 張量 |
| 檔案 hash | checkpoint 檔、q0、q1、ranges、資料檔前後一致 |
| checkpoint 存讀 | 重新載入的 router 重現訓練後的 calibration CE、路由和 hidden state（逐位元）；16 層 router 都和原 checkpoint 不同；不覆寫 router 時仍逐位元重現 q0 |
| 評估：程式控制（`proxy_refresh` 的窗口 0–127，只作控制） | FP32、q0、q1 的 CE 和上一輪逐位元相同 |
| 評估：q0+FT 第一批重跑 | 逐位元相同 |
| 評估：路由 | q0+FT 確實使用訓練後的 router：89% 的 (token, 層) 和 q0 不同；CE 都有限 |

## 修訂（`amendments.json`）

- plan 原本記錄「排除 674 篇歷史 WikiText-103 文章」，應為 1274 篇。
  - 原因：沿用的 manifest 掃描函式會略過路徑含 `olmoe_proxy_refresh_confirm` 的檔案（那原是為了跳過確認實驗自己的輸出目錄），因此漏掉了確認實驗的 600 篇。
  - plan 的規則文字本來就要求排除這 600 篇；錯的只是實作。
  - 修正發生在**抽任何評估文章之前**（controls 階段只用 calibration 和舊窗口）。
  - 原 plan 保存為 `plan_original_20261022T0354.json`。
- 腳本 sha：plan 和 controls 階段是 `ef06f5fe…`；修正後的 train、freeze、data、eval、analyze 是 `c5614198…`。修正只影響抽資料的程式。
- `routing_cost_full.json` 是分析後補算的描述性檔案，不參與任何判定。

## 資料（`segments.json`、`dedup_checks.json`）

**來源**：WikiText-103 raw train，使用嚴格文章邊界。

**排除**

| 理由 | 篇數 |
|---|---:|
| 和 WikiText-2 重疊 | 613 |
| WikiText-103 內部重複 | 464 |
| 所有歷史 OLMoE 用過的 WikiText-103 文章 | 1274 |

1274 篇的組成：mass_restore 103、dynamic_prop 242、precision_protection 的 selection 32 與 final 169、proxy_refresh 128、proxy_refresh_confirm 600。calibration 屬於 WikiText-2，已隨 WikiText-2 一起排除。

**抽樣**
- 在 router 凍結（`freeze.json`）之後才抽。
- 依 seed 22301 的順序看了 619 篇：接受 600 篇；跳過 18 篇近重複、1 篇不足 512 token。窗口起點用 seed 22302。
- 和歷史文章的重疊為 0。

## 路由描述（評估資料，`analysis.json` → `routing_descriptive`）

| | 所選 8 個權重的總和 | dispatch entropy（nats） | 最大 expert 占比 | expert 集合和 FP32 不同 |
|---|---:|---:|---:|---:|
| FP32 | 0.4199 | 4.0322 | 0.0476 | — |
| q0 | 0.4185 | 4.0320 | 0.0479 | 30.4% |
| q0+FT | 0.4198 | 4.0323 | 0.0487 | 49.8% |
| q1 | 0.4186 | 4.0319 | 0.0478 | 29.9% |

- FT 後，所選權重的總和回到 FP32 的水準，各 expert 負載的集中程度幾乎不變。
- 但 expert 選擇和 FP32 的差異大幅增加。
- 以上都是描述，沒有拿來挑 checkpoint。

## 成本總表

| 項目 | 時間 |
|---|---|
| router FT（不含控制） | 342.6 秒 |
| 控制 stage（零更新、梯度子堆疊、決定性） | 58.3 秒 |
| checkpoint 存讀控制 | 16.3 秒 |
| 抽資料 | 72.1 秒 |
| 評估：FP32 | 123.4 秒 |
| 評估：q0、q0+FT、q1 | 各約 268 秒 |
| 評估：重跑 | 20.7 秒 |
| 評估：程式控制 | 約 50 秒 |
| 評估 stage 合計 | 1006 秒 |
| 分析 | 13.3 秒 |

建立 q0 約 104 秒、refresh 額外約 108 秒，這兩個數字是前次的單次歷史量測，本輪沒有在相同條件下重新計時。

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_router_ft_20261022      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
CPU=../../venv/bin/python; S=run_olmoe_router_ft.py
$CPU $S --stage plan --out-dir $O
$GPU $S --stage controls --out-dir $O     # 約 1 分鐘
$GPU $S --stage train --out-dir $O        # 約 6 分鐘
$CPU $S --stage freeze --out-dir $O
$CPU $S --stage data --out-dir $O
$GPU $S --stage eval --out-dir $O         # 約 17 分鐘
$CPU $S --stage analyze --out-dir $O
```

## 檔案

- **預先登記與修訂**：`plan.json`、`plan.md`、`plan_original_20261022T0354.json`、`amendments.json`、`input_hashes_before.json`、`freeze.json`。
- **控制**：`controls.json`、`train.json`、`controls_summary.json`、`eval.json` → `controls`。
- **訓練**：`training_log.csv`、`router_ft.npz`（訓練後的 router）。
- **資料**：`segments.json`、`tokens.npy`、`dedup_checks.json`、`data_freeze.json`。
- **結果**：`routes_{fp32,q0,q0ft,q1}.npz`、`per_window.csv`、`per_article.csv`、`analysis.json`、`routing_cost_full.json`。
- **時間**：`timing.json`、`timing.jsonl`；`logs/`。
