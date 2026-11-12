# OLMoE aggressive 配置：模型自選 expert、恢復 FP32 相對比例，對照完整 dynamic

日期：2026-10-18
腳本：`run_olmoe_route_dynamic_prop.py`（plan → pilot → data → run → analyze）
輸出：本目錄

**本輪的性質與範圍**
- 這是使用 FP32 資訊（p0）的**診斷**，不是可部署的修正。
- 本輪測的是 GPU 上的 QDQ 行為，不代表 native kernel 的結果。
- BC、搜尋、配置（aggressive，sha256 `e51dd107…`）、量化器、router 都沒有改動。

**流程**
- 預先登記 `plan.json` 和 `plan.md` 在任何新資料評估之前寫定，沒有修訂。
- 每個階段用的都是同一份腳本（sha `55c87c7e…`）。

**標記**
- [預先登記]：寫在 `plan.json` 的分析。
- [描述]：不做推論檢定的描述統計。

## 結論

**模型自己選 expert 時，只要把所選 expert 之間的權重比例恢復成 FP32 的（總和保持當下的值），完整 dynamic 的損傷就明確降低。但仍有一小部分損傷留下來。**

| 條件 | expert 選擇 | routing weight | CE 損傷 vs FP32 | PPL |
|---|---|---|---:|---:|
| FP32 | 自選 | p0[S0] | — | 19.600 |
| L00（完全固定，參考） | S0 | p0[S0] | +0.0274 | 20.145 |
| **L11（完整 dynamic）** | 自選 S | p[S] | +0.0340 | 20.279 |
| **Ldprop** | 自選 S | p0[S] × (m／m0S) | +0.0292 | 20.181 |

- p 是該條件自己的 router 機率；m = Σp[S]，是當下的權重總和。
- p0[S] 是 FP32 基準在**同一組被選中的 expert** 上的機率；m0S = Σp0[S]。

**對比**

| 對比 | 定義 | 平均 | 95% CI | Bonferroni ×2（97.5%） | 判定 |
|---|---|---:|---|---|---|
| **Δ_dprop（主要）** | Ldprop − L11 | **−0.00481** | **[−0.00624, −0.00342]** | — | **有幫助** |
| E_total | L11 − L00 | +0.00662 | [+0.00519, +0.00807] | [+0.00497, +0.00828] | 重現，而且超過 ε |
| Residual_dprop | Ldprop − L00 | +0.00181 | [+0.00063, +0.00299] | [+0.00048, +0.00315] | 明確為正，也不在 ±0.002 內 |

- 單位 nat/token；新的 242 篇 WikiText-103 文章，968 個窗口。
- 負值代表 Ldprop 比完整 dynamic 好。

**依預先判讀**
- **helps**：成立。Δ_dprop 的 95% CI 全部小於 0。
- **enough**：不成立。Residual_dprop 不在 ±0.002 內。
- **partial**：成立。恢復比例後，相對完全 fixed 仍有明確的剩餘損傷，可能來自 expert 選擇的改變，或其後各層的連鎖影響。

**描述**
- 依點估計，完整 dynamic 相對 fixed 的額外損傷 0.0066 中，約 0.0048 在恢復比例後消失。
- 這只是點估計的比例，不是因果的占比分解。

## 和上一輪（expert 固定）的一致性 [描述]

| | expert 固定為 S0（`runs/olmoe_route_factorial_20261018`，103 篇） | 模型自選（本輪，242 篇） |
|---|---|---|
| 恢復比例的效果 | Lprop − L01 = −0.0026 [−0.0041, −0.0010] | Ldprop − L11 = −0.0048 [−0.0062, −0.0034] |
| 恢復比例後，相對完全 fixed 的剩餘 | +0.0013，區間含 0 | +0.0018，明確為正 |

- 兩輪都顯示：權重比例的偏移是 dynamic 損傷的主要部分之一。
- 兩輪的文章、樣本數和比較基準都不同，不直接合併或相減。

## 數值與語意控制（`results.json` → `controls`）

全部通過。

| 控制 | 結果 |
|---|---|
| 程式控制（pilot：上兩輪用過的窗口 0–127，舊資料） | 基準的 S0、w0、p0 和 CE，以及 L00、L11 的 CE，都和 `runs/olmoe_route_mass_restore_20261018` 逐位元相同 |
| Ldprop 的權重 | 等於 float32(p0[S] × scale)，逐位元相同 |
| Ldprop 的比例 | 權重 ÷ p0[S] 在每個 token 內的相對離散 ≤ 1.2e-7，即 FP32 的比例 |
| Ldprop 的總和 | 和當下 m 的最大相對差 2.8e-7；記錄的 m 和 m0S 都等於 float32 加總（差 ≤ 1.2e-7）；總和 = 1 的比例是 0 |
| 自選 expert 的證據 | 記錄的當下 top-8 權重都是遞減排列（自身的 topk）；29% 的 (token, 層) 選擇和 S0 不同，18% 和 L11 不同 |
| 關閉量化（第一批 128 個窗口） | Ldprop 和 FP32 逐位元相同；scale 全部恰好是 1 |
| 重跑（第一批 128 個窗口） | 逐位元相同，ID 和權重也相同 |
| 重播：Ldprop 的 ID 和權重交給 `run_olmoe_dynamic_route.forward` | 968／968 逐位元相同 |
| 其他 | 配置完整、router 未被替換、hook 只在 expert 投影、所有值有限 |

**scale（m／m0S）**
- 平均 0.999；0.1% 和 99.9% 分位數為 0.89 和 1.11。
- 極端值 0.40 和 3.92，沒有非有限值。

## 資料（`data_freeze.json`、`segments.json`）

**來源與去重**
- WikiText-103 raw train，和上兩輪相同的嚴格文章邊界與去重規則：排除和 WikiText-2 標題或內容相同的文章，以及 WikiText-103 內部重複的文章。
- 另外排除上兩輪用過的 103 篇。剩下 27,292 篇可用。

**抽樣**
- 依新的隨機順序（seed 18141）逐篇處理，接受了 242 篇。
- 9 篇因近重複被跳過（≥10% 的 5-gram 出現在某篇 WikiText-2 文章、先前用過的文章或已選文章中）。
- 已接受文章的最大重疊：和 WikiText-2 為 9.9%，和先前用過的文章為 5.2%，彼此之間為 8.2%。
- 和上兩輪的 103 篇重疊 0 篇。

**樣本數**
- pilot（32 篇舊文章）中，Ldprop − L11 的文章級 SD 為 0.0119。
- n = ⌈(1.96·s／0.0015)²⌉ = 242 篇。
- 實際 95% CI 半寬約 0.0014，達到目標。

## 逐層描述（`per_layer.csv`）[描述]

| 層 | 選擇和 S0 不同（L11／Ldprop） | 比例偏移 L1，相對 p0[S]（L11／Ldprop） | 總和 − m0S（L11／Ldprop） | MoE 輸出 norm：FP32／L00／L11／Ldprop |
|---|---|---|---|---|
| 1 | 28.1%／28.1% | 0.025／0 | +0.0002／+0.0002 | 0.219／0.214／0.214／0.214 |
| 4 | 34.8%／34.0% | 0.038／0 | +0.0004／+0.0003 | 0.286／0.286／0.287／0.286 |
| 8 | 35.5%／34.2% | 0.042／0 | −0.0005／−0.0006 | 0.484／0.482／0.481／0.481 |
| 12 | 28.7%／27.1% | 0.045／0 | −0.0005／−0.0006 | 1.869／1.863／1.860／1.861 |
| 15 | 30.4%／27.9% | 0.050／0 | −0.0017／−0.0019 | 6.472／6.474／6.450／6.453 |

- 完整 dynamic 下，所選 expert 之間的比例偏移隨深度增加，約 0.025 到 0.050。
- 恢復比例後，深層的選擇改變也略為減少。可能的原因是 hidden state 比較接近 FP32，之後的 router 選擇也就比較接近。
- 總和缺口和 MoE 輸出 norm 在兩個條件之間幾乎相同，所以改善不是來自輸出大小。

## 範圍與限制

- Ldprop 使用 FP32 基準在同一 token、同一層的機率 p0，部署時沒有這個資訊。本輪說明的是損傷**在哪裡**，不是提出修正方法。
- 只有一個模型、一個配置（aggressive）、WikiText-103 文章、GPU QDQ。
- 恢復比例會改變之後各層的 hidden state 和選擇，所以這是端到端的干預。剩餘的 +0.0018 不能直接歸因於「expert 選擇改變」本身。

## 時間成本

所有條件都在同一套 GPU 程式中執行，以下只是本次實驗的成本，不代表部署時的 latency。

| 項目 | 時間 |
|---|---|
| FP32 基準（968 個窗口，含全部 64 個機率） | 52.7 秒 |
| L00／L11／Ldprop | 115.7／122.0／121.9 秒 |
| 控制：關閉量化（子集）／重跑（子集）／重播（全部） | 10.5／22.6／118.8 秒 |
| run stage 合計 | 574 秒 |
| pilot（舊資料）／抽資料 | 83.0／33.9 秒 |
| 統計分析 | 2.6 秒 |

## 重現

```bash
cd experiments/mixed_precision
O=runs/olmoe_route_dynamic_prop_20261018      # 新實驗請改用新目錄
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
../../venv/bin/python run_olmoe_route_dynamic_prop.py --stage plan --out-dir $O
$GPU run_olmoe_route_dynamic_prop.py --stage pilot --out-dir $O     # 約 1.5 分鐘
../../venv/bin/python run_olmoe_route_dynamic_prop.py --stage data --out-dir $O
$GPU run_olmoe_route_dynamic_prop.py --stage run --out-dir $O       # 約 10 分鐘
../../venv/bin/python run_olmoe_route_dynamic_prop.py --stage analyze --out-dir $O
```

## 檔案

- **預先登記與凍結**：`plan.json`、`plan.md`、`pilot.json`、`segments.json`、`data_freeze.json`、`tokens.npy`。
- **路由**：`baseline_routes.npz`（S0、w0、全部 64 個 p0、MoE norm）、`routes/{L00,L11,Ldprop}.npz`（Ldprop 另有 scale、m、m0S、w_raw）。
- **結果與分析**：`results.json`、`analysis.json`、`per_layer.csv`、`per_article.csv`。
- **紀錄**：`timing.jsonl`、`logs/`。
