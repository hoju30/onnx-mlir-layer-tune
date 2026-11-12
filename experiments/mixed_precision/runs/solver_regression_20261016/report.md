# 求解器回歸比較：MILP＋修復、LP 優先，能否取代現行 MILP

日期：2026-10-16　腳本：`experiments/mixed_precision/run_solver_regression.py`　輸出：本目錄
預先登記：`plan.json`（05:48 寫入，在任何回歸結果之前；OLMoE 的 knee 配置在 05:51 評估前以 `freeze.json` 凍結）


## 更正與後續（2026-10-16）

以下是報告完成後，由後續實驗發現的更正。原文保留不改，閱讀時請以本節為準。

- OLMoE 配對 loss「無法判定」，以及「[推論] 重播的實作雜訊大於 ±0.002」這兩點，已由後續實驗取代：
  - 這裡的 CPU 子集結果無法重現（負載造成動態執行緒）；
  - GPU 量化器的除法路徑和 CPU 不同（已修正）；
  - 在 512 個新窗口上，LP knee 和完整重跑的 MILP knee 配對差 +0.00062，CI [−0.0003, +0.0015]，判定等價（`runs/olmoe_search_final_20261016`）。
- 這裡用的 OLMoE 參考 knee（`bc_p125`）不是完整重跑的結果。
- 本報告提議的「最小修正」已整合進 `mp_milp`（`runs/fix_integration_20261016`）。
- dense 部分在 `runs/lp_pipeline_regression_20261016` 有延伸：近重複點分組規則能消除 Qwen3 和 Qwen3.5 FFN+attn 的 LP 優先分歧；原規則下，LP 優先在這兩個模型選出的 knee 實測較差。

**GPU 量化器（2026-10-16 修正）**：本報告的 GPU 重播所用的 int 活化量化器（`qwen3_swiglu_common.int_act_qdq`），在 GPU 上是「乘以 scale 的倒數」，和 CPU 及 QuantizeLinear 的真除法相差最後一個位元，每個窗口約有 1–5 個量化值不同。量化流程會把這種差異放大成單窗口約 0.01 nat/token 的變動；以 64 個窗口的配對平均來看，偏移約 1e-3（`runs/olmoe_search_final_20261016` 實驗 A）。共用函式已修正並驗證（`runs/fix_integration_20261016`），本報告的數字沒有重算。

## 結論

**兩個方案都沒有通過預先登記的標準，所以沒有整合。** `mp_milp.py`、`mp_auto.py` 和所有既有輸出都沒有修改。

| 方案 | 可行性 | 下界差距 | knee | OLMoE 配對 loss | 判定 |
|---|---|---|---|---|---|
| P1：MILP＋修復（取代 1 ppm 重解） | 8/8 dense 全部在精確預算內，無失敗 | **3/8 不通過**（5 個預算的證明差距超過 1e-5；Qwen3 R 4.0 修復後比重解差 10%） | 8/8 和現行完全相同 | （不適用） | **未通過** |
| P2：LP 優先（差距 ≤ 1e-3 才接受，否則 MILP＋修復） | 9/9 全部在精確預算內，無失敗 | **Qwen3 不通過**（R ≥ 6.7 的 LP「下界」比真正最佳值還高） | **2/9 不同**：Qwen3（37 個節點，BC +18.7%）、Qwen3.5 FFN+attn（31 個節點，BC +8.2%） | **無法判定**：+0.0009，CI [−0.0013, +0.0031] 跨過 +0.002 | **未通過** |

[實際量測] 所有失敗都不在 knee 附近的解品質本身，而是以下三件事：

1. **knee 規則很平**。所有問題中，最佳 knee 分數和次佳點的差距只有 1e-5 到 5e-4。LP 取整後剩餘的資源差，就足以讓第二輪 refinement 往另一側取樣。
2. **下界不是嚴格保證**。目標值遠小於成本尺度時（Qwen3 的成本跨 9–10 個數量級），HiGHS 的容許誤差會蓋過真實值，不論 LP 或 MILP 的下界都是如此。
3. **OLMoE 重播的實作雜訊比等價範圍還大**。同一配置在 CPU 和 GPU 上的單窗口差異可達 0.026 nat/token。

**[推論] 建議**：
- 先採用一個**比預先登記更保守的最小修正**：保留現行的 1 ppm 重解，只有在重解也失敗時（例如 OLMoE R 0.5），才改用修復第一次的答案。只要現行程式成功，結果就完全不變，所以這次 8 個 dense 問題的結果按構造不會改；它修掉的就是 OLMoE 的下界失敗。但這個版本沒有預先登記，要不要整合請你決定。
- LP 優先若要再試，需要另外預先登記新設計（見最後一節），不能事後放寬這次的標準。

## 1. 問題與方案

- **問題**：各模型目前的正式 BC 表，GRID11 × GRID11，`mp_auto.auto_select`（17 個樣本 + 2 輪 refinement，normalized chord knee）。
  - dense：MobileNetV2、ResNet18、EfficientNet-B0（`native_auto_2d_20261011`）；GPT-2、BERT、Qwen3（`native_eps_fix_20261014`）；Qwen3.5 FFN-only、FFN+attention（`qwen35_bc_20261015`）。不刪減選項，每次求解時限 10 s，和正式流程相同。
  - OLMoE：3051 個 expert 投影，經過精確的支配刪除，時限 300 s。
- **方案**：只在本行程內替換 auto_select 呼叫的求解器，其他完全不變。
  - `milp_retry`：現行 `mp_milp.milp_select`。
  - `milp_repair`：只在精確預算求解一次；超出時用換選項修復；下界取 HiGHS 的 dual bound。
  - `lp_repair`：LP 鬆弛＋取整＋修復；下界取 LP 最佳值。只作參考。
  - `lp_first`：先用 `lp_repair`，若 目標值 − LP 下界 ≤ max(1e-3·目標值, 1e-6·目標尺度) 就接受，否則改跑 `milp_repair`。
- **OLMoE 的參考 knee**：原始 MILP 紀錄的候選點，加上速度實驗中修復後的 R 0.5 點，用 `_knee` 重新計算，得到 R 0.7071，等於 `bc_p125`（已驗證相同）。
  - OLMoE 不重跑 `milp_retry`（原本就要 58 分鐘）；`milp_repair` 也沒跑（約 35 分鐘）。

## 2. 回歸結果（`regress.json`、`analysis.json`）

| 問題 | 節點 | 現行 knee R / BC | lp_first knee R / BC | lp_first 路徑（LP / 退回 MILP） | 時間：retry / repair / lp_first |
|---|---:|---|---|---|---|
| MobileNetV2 | 53 | 0.7303 / 0.7577 | 相同 | 3 / 20 | 4.7 / 4.7 / 4.8 s |
| ResNet18 | 21 | 0.7912 / 0.3044 | 相同 | 3 / 20 | 2.3 / 2.3 / 2.3 s |
| EfficientNet-B0 | 82 | 0.7505 / 0.3556 | 相同 | 7 / 16 | 13.7 / 11.8 / 11.3 s |
| GPT-2 | 48 | 1.1884 / 21.64 | 相同 | 3 / 20 | 3.6 / 3.4 / 3.6 s |
| BERT | 74 | 1.3987 / 2.03e-6 | 相同 | 2 / 21 | 6.1 / 6.1 / 6.4 s |
| Qwen3 | 196 | 0.8780 / 330.5 | **0.8406 / 392.3** | 10 / 13 | 26.5 / 24.7 / 18.8 s |
| Qwen3.5 FFN | 72 | 0.8767 / 35.44 | 相同 | 6 / 17 | 5.4 / 5.1 / 5.3 s |
| Qwen3.5 FFN+attn | 186 | 0.8409 / 81.58 | **0.8228 / 88.30** | 9 / 14 | 27.7 / 23.9 / 18.9 s |
| OLMoE | 3051 | 0.7071 / 18.292（修正後） | 0.7071 / 18.297（7 個節點不同） | 21 / 2 | 3462 s（紀錄）/ — / **54 s** |

**重現性檢查**：MobileNetV2、ResNet18、EfficientNet-B0、GPT-2、Qwen3、Qwen3.5 兩個版本的 BC 表和原始紀錄相同。在這 7 個問題上，`milp_retry` 選出的 knee 和原始紀錄完全一致，驗證了這個測試框架。BERT 的表已因 eps 修正而改變，無法比對。

**[實際量測] P1：MILP＋修復**
- **可行性**：8/8，每次求解都在精確預算內；knee 8/8 和現行完全相同。
- **證明差距**：有 6 次求解需要修復；現行版本在同樣 6 個預算上也都需要重解。

| 問題 | R | 修復後目標值 | dual bound | 差距 / 容許 | 現行重解的目標值 |
|---|---:|---:|---:|---|---:|
| Qwen3.5 FFN | 4.0 | 0.007551 | 0.002577 | 0.0050 / 1.4e-5 ✗ | 0.007551（相同） |
| Qwen3 | 4.0 | **0.06251** | 0.05263 | 0.0099 / 6.3e-4 ✗ | **0.05683**（修復差 10%） |
| Qwen3.5 FFN+attn | 0.7071 / 1.414 / 4.0 | 和重解相同 | — | 1.4–3.2 倍容許 ✗ | 相同 |
| EfficientNet-B0 | 0.621 | 0.84379 | 0.843783 | 6.4e-6 / 8.4e-6 ✓ | 0.843793 |

- **原因**：在精確預算上，HiGHS 利用 1e-6 的可行性容許，找到一個便宜很多、但實際超出預算的解，例如 Qwen3.5 FFN R 4.0 的 0.00258。因此它的 dual bound 只是這個放寬後問題的下界，對精確預算來說太鬆，無法證明修復後的解接近最佳。
- 貪婪修復通常和重解得到同樣的配置；但在 Qwen3 R 4.0 選到較差的配置。這些點都在 R ≥ 0.7 且遠離 knee，不影響 knee。

**[實際量測] P2：LP 優先**
- **knee 移動的機制（Qwen3）**：
  - 粗掃描在 R 0.8409 的 LP 解取整後，實際資源只有 0.8406。它和第一輪 refinement 在 0.8408 的 MILP 解幾乎重疊，knee 分數只差 1e-6。
  - knee 停在 0.8406 這一點後，它的右鄰居變成 0.8408，所以第二輪 refinement 在 0.771–0.8408 之間取樣。
  - MILP 版本則在 0.771–0.917 之間取樣，找到 R 0.878（分數 0.92113，比 0.92079 高 3.5e-4）。LP 版本從頭到尾沒有取樣到這一點。
- **Qwen3.5 FFN+attn**：LP 版本往左邊取樣，反而找到分數更高的點（R 0.8228，0.92641 對 MILP knee 的 0.92633）。兩邊都是同一段平坦區域裡的合理取樣，但 BC 差 8%。
- **LP 下界不嚴格（Qwen3）**：R 6.73 時 LP「下界」0.001926 高於 MILP 最佳值 0.001021；R 8 時 LP 給 0.00193，但全 FP32（目標值 0）就是可行解。
  - 原因：Qwen3 的 max|cost| 約 6e6，目標尺度 624；LP 最佳性的容許是 1e-7 × 尺度，每個節點的誤差就和這些小目標值同一量級。
  - 這些點遠離 knee，對 knee 沒有影響；但「保證差距」這個說法在小目標值上並不成立。
- **OLMoE**：
  - 可行性 23/23；和原始 MILP 同預算的 16 個點都在容許內，最大差 5.5e-4。
  - knee 和修正後的 MILP knee 在同一個預算：BC 高 0.03%，R 低 6.5e-5，7 個節點不同。
  - 只有 R 3.36 和 R 6.73 兩點退回 MILP，分別 24 s 和 15 s。

**knee 分數有多平**（`milp_repair`，最佳點和 R 差 > 0.5% 的次佳點之分數差）：ResNet18 9.8e-6、MobileNetV2 5.7e-5、OLMoE 1.1e-4（lp_first）、EfficientNet 1.3e-4、Qwen3.5 FFN+attn 1.7e-4、BERT 2.1e-4、GPT-2 3.3e-4、Qwen3 3.5e-4、Qwen3.5 FFN 4.9e-4。也就是說，**哪個點被取樣到，就幾乎決定了 knee**。另外 6 個 dense 問題的 knee 能保持相同，是因為 LP 優先在大多數預算都退回 MILP，得到完全相同的點。

## 3. OLMoE 的 LP knee 配對 loss 評估（`truth.json`、`truth_cpu.json`）

64 個新窗口：validation 和 test 各從 token 163840 起取 32 個，和 stage 1 及 multipoint 的所有窗口在字元範圍上不重疊。固定使用 FP32 路由，GPU FP32 重播，TF32 關閉。

| 配置 | dNLL vs FP32（nat/token） | 95% CI | PPL |
|---|---:|---|---:|
| milp_knee（= bc_p125） | +0.0098 | [+0.0069, +0.0130] | 22.11 |
| lp_knee | +0.0107 | [+0.0075, +0.0143] | 22.13 |
| **lp − milp（配對）** | **+0.0009** | **[−0.0013, +0.0031]** | V −0.0003 / T +0.0022 |

FP32 的 NLL 為 3.086（PPL 21.90）。

- **[實際量測] 判定：無法判定（inconclusive）**。CI 跨過 +0.002，依預先登記不算通過。
- **[實際量測] CPU 交叉檢查**（16 個窗口）：
  - 同一配置在 CPU 和 GPU 之間的單窗口差異最大 0.026（milp_knee）和 0.018（lp_knee）nat/token。
  - 配對差在這 16 個窗口上：GPU +0.00006，CPU +0.0076。
  - 兩個 knee 雖然只差 7 個節點，GPU 上配對差的單窗口標準差仍有 0.0092。
- **[推論]** 這個重播的實作雜訊（浮點順序不同，造成量化取整翻轉）大約和 ±0.002 的等價範圍同一量級甚至更大。所以這個比較的解析度不足以證明兩個 knee 在 ±0.002 內等價。也沒有證據顯示 LP knee 比較差：GPU 上 V 段為負、T 段為正，CPU 子集則偏正。
- milp_knee 在新窗口上的 dNLL +0.0098，比 multipoint 窗口上的 +0.0175 小，因為是不同的文字。

## 4. 已知限制

- OLMoE 沒有完整跑 `milp_repair` 的 auto_select（約 35 分鐘）。參考 knee 的 refinement 是在錯誤的 knee（0.569）周圍做的；R 0.707 附近沒有 MILP refinement。
- dense 模型的 knee 不同時，只比較了預測的 BC，沒有做 native 編譯評估（不在預先登記範圍內）。
- 在 Qwen3.5 FFN+attn 的 MILP 求解中，HiGHS 在 stderr 印出大量 `HighsMipSolverData::transformNewIntegerFeasibleSolution tmpSolver.run();` 除錯訊息；求解結果都可行，和現行版本相同。
- 時間是在另一個回歸行程同時執行時量測的（dense 和 OLMoE 並行），只供參考。

## 5. 若要再試（每一項都需要新的預先登記）

1. **最小修正**（建議先做）：保留 1 ppm 重解；只在重解失敗時才修復第一次的答案。凡是現行版本成功的地方，結果都相同。
2. **LP 只用在粗掃描，refinement 一律用 MILP**：knee 附近的點和現行完全一致。OLMoE 的時間預估是粗掃描 17 個 LP（約 15 s）加上 6 個 refinement MILP（各 1–4 分鐘），比 58 分鐘少很多，但不會到 1 分鐘。
3. **LP 取整後補「填滿」步驟**，把剩餘資源用掉，避免出現資源略低的近重複點；同時把接受門檻改成相對於 frontier 的 BC 跨度，而不是 max|cost|。
4. **knee 規則本身的穩健性**：分數差只有 1e-4 的層級，在 Qwen3 上選擇會在 BC 差 18% 的兩點之間跳動。這是 criterion 以外、值得單獨處理的問題。

## 重現

```
PY=../../venv/bin/python; GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
O=runs/solver_regression_20261016     # 新實驗請換新目錄
$PY run_solver_regression.py --stage plan --out-dir $O
$GPU run_solver_regression.py --stage capture --out-dir $O --device cuda
$PY run_solver_regression.py --stage regress --out-dir $O --problems qwen35_ffn,mobilenetv2,resnet18,efficientnet_b0,gpt2,bert,qwen3,qwen35_ffn_attn
$PY run_solver_regression.py --stage regress --out-dir $O --problems olmoe        # 獨立行程（transformers 版本）
$GPU run_solver_regression.py --stage truth --out-dir $O --device cuda
$PY run_solver_regression.py --stage cpu --out-dir $O
$PY run_solver_regression.py --stage analyze --out-dir $O
```

實際耗時：regress dense 245 s（另有 qwen35_ffn 試跑 18 s）、OLMoE 74 s、capture 9 s、truth 33 s、cpu 213 s。

## 檔案

- `plan.json`：預先登記；`freeze.json`：OLMoE knee 配置、ranges、窗口的雜湊。
- `regress.json`：每個問題、每種方案的每次求解（路徑、狀態、目標值、下界、差距、超出量、時間）、frontier、knee 和比對結果。
- `analysis.json`：P1／P2 的逐條判定、OLMoE loss 和 CPU 檢查。
- `configs/milp_knee.json`、`configs/lp_knee.json`；`truth.json`（GPU，64 個窗口）、`truth_cpu.json`（CPU，16 個窗口）。
- `tokens_eval.npy`、`routes/`、`routes_manifest.json`：新的評估窗口和 FP32 路由。
- `*.log`、`*.out`、`timing.json`。
