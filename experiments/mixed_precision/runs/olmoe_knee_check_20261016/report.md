# OLMoE 固定路由：MILP knee 和 LP＋修復 knee 的實際損傷、模型大小與 BOPs

日期：2026-10-16　腳本：`experiments/mixed_precision/run_olmoe_knee_check.py`　輸出：本目錄
預先登記：`plan.json`。這次是量測，沒有任何選擇，也沒有通過或不通過的標準。


## 更正與後續（2026-10-16）

以下是報告完成後，由後續實驗發現的更正。原文保留不改，閱讀時請以本節為準。

- 第 1 節的推論「兩配置的真實差距小於實作雜訊」，是依據一次後來證實無法重現的 CPU 執行（負載造成動態執行緒），不再成立。
- 「MILP 0.707」（`bc_p125`）不是完整重跑後的 knee：完整重跑的 knee 和它差 11 個投影。
- LP knee 和完整重跑的 MILP knee，已在 512 個新窗口上判定等價：配對差 +0.00062，CI [−0.0003, +0.0015]（`runs/olmoe_search_final_20261016`）。

**GPU 量化器（2026-10-16 修正）**：本報告的 GPU 重播所用的 int 活化量化器（`qwen3_swiglu_common.int_act_qdq`），在 GPU 上是「乘以 scale 的倒數」，和 CPU 及 QuantizeLinear 的真除法相差最後一個位元，每個窗口約有 1–5 個量化值不同。量化流程會把這種差異放大成單窗口約 0.01 nat/token 的變動；以 64 個窗口的配對平均來看，偏移約 1e-3（`runs/olmoe_search_final_20261016` 實驗 A）。共用函式已修正並驗證（`runs/fix_integration_20261016`），本報告的數字沒有重算。

## 配置（`plan.json` 中的身分檢查全部通過）

| 名稱 | 來源 | 說明 |
|---|---|---|
| FP32 | — | 所有張量用 FP32（BF16 checkpoint 轉成 FP32），FP32 路由 |
| MILP 0.707 | `runs/olmoe_multipoint_20261015/configs/bc_p125.json` | 修正後的 MILP knee（R 0.7071） |
| LP knee | `runs/solver_regression_20261016/configs/lp_knee.json` | LP 鬆弛＋取整＋修復選出的 knee；和 LP 優先選出的相同 |

兩者只差 7 個節點（3051 個中）：

| 節點 | MILP 0.707（w / a） | LP knee（w / a） | calibration 路由列數 |
|---|---|---|---:|
| L0.E5.gate | int5 / p6e1 | int5 / int7 | 144 |
| L5.E3.gate | int5 / p6e1 | int5 / int7 | 4 |
| L6.E14.gate | int4 / p6e1 | int5 / p6e1 | 19 |
| L7.E43.gate | int6 / p8e1 | int5 / p8e1 | 331 |
| L7.E62.down | int4 / int7 | int5 / int7 | 16 |
| L8.E8.down | int4 / int7 | int5 / int6 | 12 |
| L11.E53.up | int5 / p6e1 | int5 / p8e1 | 21 |

## 1. 實際量化損傷（`truth.json`、`analysis.json`）

每個窗口 128 個 token，算 127 個下一個 token 預測的 CE 總和；每一層都用快取的 FP32 路由；在 GPU 上以 FP32 計算，TF32 關閉。

| 窗口 | FP32 NLL / PPL | MILP 0.707：dNLL（95% CI）/ PPL | LP knee：dNLL（95% CI）/ PPL | LP − MILP（配對） |
|---|---|---|---|---|
| A：multipoint 評估窗口（64） | 2.7701 / 15.960 | +0.0175 [+0.0137, +0.0209] / 16.241（+1.76%） | +0.0175 [+0.0136, +0.0213] / 16.242 | **+0.0000** [−0.0025, +0.0025] |
| B：回歸實驗的新窗口（64） | 3.0864 / 21.898 | +0.0098 [+0.0067, +0.0133] / 22.114（+0.99%） | +0.0107 [+0.0074, +0.0143] / 22.134 | +0.0009 [−0.0014, +0.0031] |
| **合併（128）** | 2.9282 / 18.695 | **+0.0136** [+0.0113, +0.0161] / 18.952（+1.37%） | **+0.0141** [+0.0116, +0.0166] / 18.960（+1.42%） | **+0.0005** [−0.0012, +0.0022] |

dNLL 的單位是 nat/token。bootstrap 為 2000 次，依 A-V、A-T、B-V、B-T 四段分層抽樣，seed 0。

- **[實際量測] 兩個配置的損傷相同，在量測解析度內分不出差別**：合併 128 個窗口後，配對差是 +0.0005 nat/token，CI 包含 0。A 組正好是 0.0000，B 組是 +0.0009。
- 合併 CI 的上界 +0.0022 略超過回歸實驗用的 ±0.002 等價範圍，所以嚴格來說，仍無法證明兩者在 ±0.002 內等價。
- **[實際量測] 重播是決定性的**：A 組的 FP32 和 MILP 0.707、B 組的三個配置，每個窗口的 CE 總和都和先前的量測逐位元相同（最大差 0.0）。
- **[推論]** 先前的 CPU 交叉檢查顯示，同一配置在 CPU 和 GPU 上的單窗口差異可達 0.026 nat/token（`runs/solver_regression_20261016`）。只差 7 個節點的兩個配置，它們的真實差距小於這個實作雜訊。

## 2. 模型大小（`resources.json`）

**搜尋範圍**：3051 個 expert 投影，knee 預算約束的就是這兩個量。

| | M（權重 bytes） | M / M8 | B（calibration 列數） / B8 | R |
|---|---:|---:|---:|---:|
| MILP 0.707 | 4,418,174,976 | 0.690511 | 0.707107（貼著預算） | 0.707107 |
| LP knee | 4,418,699,264（**+524,288 B，+0.012%**） | 0.690593 | 0.707061（−0.0065%） | 0.707061 |

兩者都在 R = 0.70710678 的精確預算內。差別是：LP knee 的權重多 0.5 MiB，BOPs 少 0.0065%。

**整個模型的權重儲存**：共 6.92B 參數，其中 3072 個 expert 投影；未被選到的是 7 個 expert（21 個投影）。

| | expert 權重 | int 每 channel scale（FP32） | 其他張量（FP32：attention、router、norm、embedding、LM head） | 合計 | 相對 FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 | 25.770 GB | 0 | 1.907 GB | **27.677 GB** | 100% |
| MILP 0.707 | 4.506 GB | 16.6 MB | 1.907 GB | **6.4297 GB** | 23.23% |
| LP knee | 4.507 GB | 16.6 MB | 1.907 GB | **6.4303 GB** | 23.23% |

- GB = 1e9 bytes。未被選到的 21 個投影在兩個量化配置中都是 bf16。
- 只看 expert 的部分（含 scale），兩者都是 FP32 的 17.55%。
- 兩個量化配置的權重格式分布：

| 權重格式 | int4 | int5 | int6 | int7 | int8 | bf16 |
|---|---:|---:|---:|---:|---:|---:|
| MILP 0.707 | 172 | 1339 | 1390 | 125 | 18 | 28 |
| LP knee | 169 | 1343 | 1389 | 125 | 18 | 28 |

## 3. BOPs（評估 token 上實際執行的量）

計算方式：
- **expert 部分**：每個 token 在每一層路由到的 8 個 expert，乘上元素數、權重位元、活化位元。路由取自評估窗口實際的路由。
- **FP32 線性層**：attention 的 q/k/v/o、router、LM head，以 32 × 32 位元計。
- **不計入**：attention 的分數和 value 乘積。它們是活化 × 活化，在所有配置中都是 FP32，三者相同。

| | expert BOPs / token（A / B） | 整個模型線性層 BOPs / token（A / B） | expert 相對 FP32 | 整個模型相對 FP32 |
|---|---:|---:|---:|---:|
| FP32 | 8.246e11 / 8.246e11 | 1.2072e12 / 1.2072e12 | 100% | 100% |
| MILP 0.707 | 3.6049e10 / 3.6182e10 | 4.1857e11 / 4.1870e11 | 4.372% / 4.388% | 34.674% / 34.685% |
| LP knee | 3.6053e10 / 3.6178e10 | 4.1857e11 / 4.1870e11 | 4.372% / 4.387% | 34.674% / 34.685% |

- LP knee 對 MILP 0.707：expert BOPs 在 A 組 +0.0099%、B 組 −0.0106%；整個模型 ±0.001%。
- expert 的 4.37% 換算成相對全 int8，約為 0.70，和預算 R 0.707 一致。評估窗口的路由分布和 calibration 不同，所以會有小差異。
- **[實際量測]** 量化後的模型中，沒有搜尋的 FP32 線性層（attention 和 LM head）佔整個模型 BOPs 的 91%。所以整個模型的 BOPs 只降到 34.7%；要再降，得把 attention 和 LM head 也納入搜尋。

## 結論

- **[實際量測] LP＋修復選出的 knee，在損傷、大小、BOPs 上都和 MILP 0.707 等同到量測解析度**：
  - 損傷：合併 128 個窗口的配對差 +0.0005 nat/token，CI [−0.0012, +0.0022]，包含 0。
  - 大小：權重多 0.5 MiB（+0.012%）。
  - BOPs：差異在 ±0.01% 以內。
- 兩者相對 FP32 的損傷都約 +0.014 nat/token（PPL +1.4%）。整個模型的權重從 27.68 GB 降到 6.43 GB（23.2%），線性層 BOPs 降到 34.7%。
- 這個結果只說明 OLMoE 上的這兩個 knee。它不改變回歸實驗的判定：LP 優先在 Qwen3 和 Qwen3.5 FFN+attn 上選出不同的 knee，所以沒有整合。

## 重現

```
PY=../../venv/bin/python; GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
O=runs/olmoe_knee_check_20261016     # 新實驗請換新目錄
$PY run_olmoe_knee_check.py --stage plan --out-dir $O
$PY run_olmoe_knee_check.py --stage resources --out-dir $O
$GPU run_olmoe_knee_check.py --stage truth --out-dir $O --device cuda
$PY run_olmoe_knee_check.py --stage analyze --out-dir $O
```

實際耗時：plan 0.3 s、resources 2.5 s、truth 63 s（每組窗口 28 s）、analyze < 1 s。

## 檔案

- `plan.json`、`configs/`（兩個配置的副本，有雜湊）、`ranges.json`（multipoint 的副本）。
- `resources.json`：搜尋範圍的 M／B、整個模型的儲存、評估 token 上的 BOPs、7 個不同節點。
- `truth.json`：A、B 兩組窗口的每窗口 CE 總和；`analysis.json`：dNLL、PPL、配對差、決定性檢查。
- `*.log`、`timing.json`。
