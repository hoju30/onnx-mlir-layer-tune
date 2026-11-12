# Qwen3-0.6B：Gate／Up 聯合 block criterion（觀察邊界在 down_proj 輸出 Y）

日期：2026-10-14　腳本：`experiments/mixed_precision/run_qwen_block_criterion.py`　輸出：本目錄

## 結論

1. **預先登記的假設不成立。** 預先登記的主要指標是 tuning 資料上、同層同模式組內的 Spearman：
   - 在同一個評分函數 phi 下，完整 SwiGLU 重播的 exact joint 分數沒有比單獨成本相加（additive）排得更好：
     - mse：−0.007，bootstrap 95% CI [−0.012, −0.003]；
     - gw：−0.001 [−0.004, +0.002]；
     - dir：−0.030 [−0.031, −0.016]。
   - 三種 output criterion 都比原本逐節點的 BC 差，差 0.073–0.120，CI 都不含 0。
   - 相同資源的成對決策：BC_total 對 70%、BC_branch 對 71%，output criterion 是 56–65%。
   - Gate／Up 互換（相同資源）的選擇，所有預測量都接近隨機（39–56%），包括 BC。
2. **聯合交互作用的方向抓得到，但它太小，不影響排序。**
   - I_output = C_exact − C_additive 能預測完整模型的非加成性 N = S_joint − S_gate − S_up 的方向：
     - 正負號一致（|N| > 1 時）：tuning 73/97（mse、gw），calibration 64–67/87；
     - Spearman：0.44（tuning）。
   - 只用線性交叉項（linear − additive）時只有 57–60/97，所以有資訊的是完整 SwiGLU 的有限擾動非線性。
   - 但 N 相對整體損傷很小。例如 L2 activation-only 的 Gate INT4／Up INT8：N = −24，S = 1263。mse 與 gw 的 I_output 相對 additive 的中位數都在 ±0.3% 內，所以聯合重播幾乎不改變候選的排序。
3. **失敗的主因（事後診斷，不在原計畫內）：position 0 的 outlier。**
   - L2 與 L27 的 Y 分別有 99.99% 與 72% 的能量在 position 0（attention sink 的巨大激活）。候選的 output MSE 中位數有 99.9%（L2）與 90%（L27）來自 position 0，但那裡的誤差幾乎不反映在 loss 上。
   - 排除 position 0 後：
     - 組內 Spearman：mse 0.634 → 0.779，gw 0.638 → 0.761，都高於 BC_total 0.717；
     - 相同資源配對：mse 234/302（77%），BC_total 211/302（70%）。
   - 但 exact 對 additive 仍然只有 +0.003（mse）與 +0.0004（gw）。改善來自 outlier 的處理，不是聯合建模。
   - 這是看過 tuning 結果之後才發現的，必須用新的窗口重新驗證，不能當成已證實的結論。
4. **native 核對**：4 個不同 datatype 的聯合配置由 onnx-mlir 編譯執行，和 PyTorch 模擬的 S 相對差 ≤ 2e-5（大損傷配置）；含 posit 的配置，每窗口差 ≤ 0.11 summed CE（第 5 節）。
5. **是否接入 MILP：不建議。** 聯合 option 沒有排序收益，而且 option 數會從每層 2 × 121 增加到 121² = 14641，MSE 與 BC 的尺度也不一致。值得追的方向是 node 層級的 outlier 處理（第 7 節）。

## 1. 方法與觀察邊界

**block（layer L）**：zg = X·Wg，zu = X·Wu，h = SiLU(zg) ⊙ zu，Y = h·Wd。觀察邊界是 Y，也就是 down_proj 的輸出；Down 固定 FP32。

**量化語意**（沿用 pass 的 per-consumer QDQ）：
- **激活**：Gate 或 Up 的激活格式 a，會量化該分支自己的那份 X，以及自己的 projection output（zg 或 zu），位置在 SiLU 之前。
- **int 激活參數**：(scale, zp) 來自 `runs/native_eps_fix_20261014/qwen3/ranges_by_point.json`（修正 epsilon 後的 calibration range）。和 errors layouts 的紀錄比對 32 組，0 個不一致。
- **權重**：每個輸出 channel 對稱量化，用 `std::lround`。
- **posit**：用和 Universal 逐值相同的 LUT。

**候選**（`plan.json`，在任何 loss 評估前凍結）：
- 層：0、2、14、27。
- 每層 83 個候選，共 332 個：

| 模式 | 每支的選項 |
|---|---|
| w（weight-only） | int4、int6（knee 的權重格式）、int8、p6e1 |
| a（activation-only） | int4、int8、p6e1、p8e1（knee 的激活格式）、bf16 |
| wa（weight＋activation） | int4/int4、int6/p8e1（knee 附近）、int8/int8、p6e1/p6e1 |

- 每組（層 × 模式）包含：Gate-only、Up-only、所有 Gate × Up 組合（含所有不同 datatype 的組合）。

**評分 phi**（e = Y_hat − Y_fp32）：
- **共同範圍**：calibration 窗口 0–9、position 0–126、全部 1024 個 channel，float64 累加。position 127 沒有 target，在 causal 模型中也不影響任何 loss，所以排除；其 gY 經檢查全為 0。
- **mse** = 範圍內 e² 的平均。
- **gw** = Σ gY² e²。gY 是每窗口 summed CE 對 Y 的梯度。
- **dir** = Σ_s (Σ_{t,j} gY e)²：每窗口一階項的平方。Down 是線性的 FP32，所以 ⟨gY, ΔY⟩ = ⟨gh, Δh⟩，這項沒有因為邊界移到 Down 之後而增加新資訊。

**結構**（同一個 phi 下）：
- additive = phi(eg) + phi(eu)；
- linear = phi(eg + eu)；
- exact = phi(egu)；
- 單支候選的三者相同。

**外部 baseline**：BC_total（`table.json`）與 BC_branch（`table_branch.json`），量化分支的 B[w] + C[a] 相加。

**loss 影響**（PyTorch 模擬）：
- 快取第 L 層 MLP 之前的 residual h_mid，在 h_mid + Y 上重播第 L+1 層之後的 layers、final norm、LM head 與 CE。
- ΔL_s = CE_s(Y_hat) − CE_s(Y_fp32)，每窗口一個值。
- 三個指標（下文分開使用，不混稱）：
  - **S** = Σ_s ΔL_s，帶號；
  - **|S|** = abs(Σ_s ΔL_s)；
  - **A** = Σ_s abs(ΔL_s)。

**資料角色**：
- calibration 窗口 0–9：range、快取、分數表。
- tuning 窗口 10–29：驗證；同一個 validation split，不用於任何建表或選擇。
- test split：完全沒有讀取。

### FP32 重播檢查（候選量化之前）

| layer | zg rel_max | Y（由 X）rel_max | Y（由快取 zg/zu）rel_max | 下游重播 vs 模型 loss 最大差 | Y 能量在 position 0 的比例 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.4e-07 | 3.5e-07 | 2.8e-07 | 1.9e-04 | 0.010 |
| 2 | 4.3e-07 | 6.8e-07 | 6.1e-07 | 1.4e-04 | 0.9999 |
| 14 | 4.3e-07 | 7.5e-07 | 4.4e-07 | 1.3e-04 | 0.000 |
| 27 | 3.7e-07 | 4.8e-07 | 3.8e-07 | 9.1e-05 | 0.723 |

**邊界確認**：Gate、Up 的輸入就是 MLP 的輸入，Down 的輸出就是 MLP 的輸出（逐值相等）。下游重播與完整模型的每窗口 loss 差 ≤ 1.9e-4，每窗口 loss 約 444。

**和先前模擬交叉比對**：本研究的下游重播 ΔL，和第 5 階段完整模型模擬（forward hooks，batch 1）的 108 組相同格式候選相比，calibration S 最大絕對差 0.05；|S| > 1 時最大相對差 4.6e-3。

## 2. Additive／linear joint／exact joint 的比較

**組內 Spearman**（12 組 = 4 層 × 3 模式，每組含單獨與聯合候選；mean／median）：

| 預測量 | tuning S | calibration S | tuning A | tuning abs(S) |
|---|---|---|---:|---:|
| mse.additive | 0.641／0.770 | 0.573／0.749 | 0.862 | 0.790 |
| mse.linear | 0.630／0.747 | 0.561／0.723 | 0.861 | 0.785 |
| mse.exact | 0.634／0.764 | 0.566／0.724 | 0.866 | 0.789 |
| gw.additive | 0.639／0.810 | 0.596／0.726 | 0.875 | 0.799 |
| gw.linear | 0.637／0.792 | 0.595／0.729 | 0.877 | 0.802 |
| gw.exact | 0.638／0.808 | 0.597／0.728 | 0.878 | 0.803 |
| dir.additive | 0.628／0.747 | 0.576／0.724 | 0.856 | 0.788 |
| dir.linear | 0.601／0.713 | 0.550／0.715 | 0.846 | 0.782 |
| dir.exact | 0.597／0.694 | 0.548／0.706 | 0.841 | 0.778 |
| BC_total | **0.717**／0.855 | 0.667／0.811 | 0.879 | 0.806 |
| BC_branch | 0.711／0.838 | 0.668／0.828 | 0.875 | 0.804 |

**每組比較**（tuning S；bootstrap 是對 20 個 tuning 窗口重抽 2000 次）：

| 比較 | 平均差 | 較好／較差組數 | bootstrap 95% CI |
|---|---:|---|---|
| mse.exact − mse.additive | −0.007 | 1／7 | [−0.012, −0.003] |
| gw.exact − gw.additive | −0.001 | 1／8 | [−0.004, +0.002] |
| dir.exact − dir.additive | −0.030 | 0／12 | [−0.031, −0.016] |
| gw.exact − BC_total | −0.079 | 0／11 | [−0.087, −0.060] |
| gw.exact − BC_branch | −0.073 | 2／9 | [−0.083, −0.055] |
| mse.exact − BC_total | −0.084 | 2／9 | [−0.101, −0.062] |

bootstrap 是凍結計畫之後、看到 tuning 結果之前才加入的不確定性估計。

**只看聯合候選的組內 Spearman**（tuning S mean）：
- mse：additive 0.667、linear 0.649、exact 0.656；
- gw：0.663、0.657、0.659；
- dir：0.640、0.592、0.590；
- BC_total 0.742。
- 結論相同：exact 沒有勝過 additive。

**相同資源的成對決策**：同組內 weight bytes 與 BOPs 都相同的配對，排除 |ΔS| ≤ 1 的 tie。

| 預測量 | 全部配對 tuning | calibration | Gate／Up 互換 tuning | calibration |
|---|---|---|---|---|
| mse.additive／exact | 184／180 of 302 | 164／157 of 299 | 43／43 of 82 | 41／40 of 83 |
| gw.additive／exact | 196／189 of 302 | 181／175 of 299 | 43／41 of 82 | 44／43 of 83 |
| dir.additive／exact | 186／170 of 302 | 176／164 of 299 | 36／32 of 82 | 39／35 of 83 |
| BC_total | 211 of 302 | 196 of 299 | 45 of 82 | 45 of 83 |
| BC_branch | 214 of 302 | 200 of 299 | 46 of 82 | 47 of 83 |

**非加成性 N = S_joint − S_gate − S_up**（228 個聯合候選）：

| phi | 資料 | N > 1 | N < −1 | Spearman(exact − additive, N) | Spearman(linear − additive, N) | 正負號一致 exact − additive | linear − additive |
|---|---|---:|---:|---:|---:|---|---|
| mse | tuning | 45 | 52 | 0.440 | 0.197 | 73/97 | 60/97 |
| gw | tuning | 45 | 52 | 0.438 | 0.120 | 73/97 | 57/97 |
| dir | tuning | 45 | 52 | 0.206 | 0.028 | 61/97 | 51/97 |
| mse | calibration | 37 | 50 | 0.396 | 0.233 | 64/87 | 55/87 |
| gw | calibration | 37 | 50 | 0.450 | 0.207 | 67/87 | 53/87 |

**I_output = C_exact − C_additive**（相對 additive 的中位數；範圍）：
- w：mse +0.000（−0.61～+0.44）；
- a：mse −0.000（−0.43～+1.23）；
- wa：gw −0.003（−0.43～+0.61）。
- 多數聯合候選的交互項接近 0；少數極端值可以到 ±100%，但都在格式強弱差異之下。

**pooled Spearman**（全部 332 個候選，只供參考，主要反映格式強弱）：tuning S 上，mse 約 0.57、gw 約 0.69、dir 約 0.68、BC_total 0.71。

## 3. 不同 datatype 的結果（摘錄；完整見 `candidates.csv`）

S 是 summed CE 的變化，tuning 共 20 窗口 × 127 tokens。

| 候選 | S_cal | S_tune | N_tune | mse exact／additive | gw exact／additive | BC_total |
|---|---:|---:|---:|---|---|---:|
| L0 a：Gate INT4／Up INT8 | +1232.8 | +997.1 | +10.7 | 0.0064／0.0064 | 730／728 | 1240 |
| L0 a：Gate INT8／Up INT4 | +176.6 | +676.4 | +54.4 | 0.0082／0.0082 | 953／952 | 1330 |
| L0 a：Gate P6／Up INT8 | +6.40 | −4.31 | −0.24 | 1.1e-4／1.1e-4 | 13.5／13.4 | 26.4 |
| L0 a：Gate INT8／Up P6 | +0.06 | +8.56 | −0.07 | 1.0e-4／1.0e-4 | 11.9／12.0 | 21.9 |
| L2 a：Gate INT4／Up INT8 | +587.2 | +1263.3 | −23.9 | 299／299 | 1.53e4／1.54e4 | 4760 |
| L2 a：Gate INT8／Up INT4 | +1948.3 | +3794.8 | −11.6 | 306／306 | 1.58e4／1.59e4 | 3150 |
| L2 a：Gate P6／Up INT8 | +22.6 | +30.3 | +1.9 | 4.66／4.63 | 326／321 | 220 |
| L2 a：Gate INT8／Up P6 | +7.07 | +1.51 | +1.17 | 6.52／6.35 | 353／344 | 127 |
| L14 a：Gate INT4／Up INT8 | +37.3 | +33.5 | −1.22 | 0.092／0.092 | 74.1／74.1 | 167 |
| L14 a：Gate INT8／Up INT4 | +63.9 | +102.7 | +0.74 | 0.144／0.144 | 117／117 | 184 |
| L27 a：Gate INT4／Up INT8 | +914.3 | +1777.1 | −61.0 | 152／154 | 3060／3100 | 7170 |
| L27 a：Gate INT8／Up INT4 | +824.5 | +1653.1 | −51.3 | 90.0／90.9 | 2590／2630 | 7730 |
| L27 a：Gate P6／Up INT8 | +50.3 | +113.8 | −11.1 | 38.5／39.1 | 207／213 | 952 |
| L27 a：Gate INT8／Up P6 | +26.1 | +51.2 | +0.30 | 49.0／48.7 | 231／234 | 543 |
| L27 wa：Gate INT8／Up P6 | +41.4 | +82.3 | −19.2 | 40.3／44.1 | 215／231 | 923 |

**觀察**
- **互換方向會大幅改變損傷，但沒有任何準則能穩定判斷。**
  - L0 激活：Gate INT4／Up INT8 的 S_cal 1233，互換後只有 177。mse、gw、BC 都判斷相反。
  - L2 正好相反：Gate INT8／Up INT4 比較傷（3795 vs 1263）。mse、gw 判斷對，但差距只有 2–3%；BC 判斷錯。
- **Posit6 的組合損傷小。** 在 L0、L14，損傷落在雜訊附近，例如 S_tune 為 −4.3 與 +8.6。
- **非加成性在 L27 int4 激活明顯為負（抵消）**：N 為 −51 到 −62，exact 也小於 additive（方向一致），但相對 S 只有約 3%。

## 4. Calibration／tuning 預測能力與負結果的原因

**雜訊參考**：以 calibration 實測 S 本身當預測量，對 tuning S 的組內 Spearman 是 0.845，相同資源配對對 249/302。這不是嚴格上限：10 個窗口的 S 也有雜訊，部分組的準則甚至超過它。

**樣本量**
- tuning 有 20 個窗口。bootstrap 顯示「exact 不優於 additive」與「output criterion 不如 BC」這兩個結論穩定：CI 都很窄；除了 gw 的 exact − additive 是 [−0.004, +0.002]，其餘都不含 0。
- calibration 與 tuning 的結論方向一致。

**事後診斷**（`explore.json`，不在計畫內，只用來解釋）：

| 預測量（tuning S，組內 Spearman mean） | 值 |
|---|---:|
| mse.exact／gw.exact（計畫內） | 0.634／0.638 |
| mse.exact 排除 position 0 | **0.779**（additive 0.777） |
| gw.exact 排除 position 0 | **0.761**（additive 0.760） |
| BC_total | 0.717 |
| 帶號一階 F = Σ_s⟨gY, e⟩（calibration 窗口） | 0.672 |
| 參考：calibration 實測 S | 0.845 |

- **outlier**：
  - 候選的 output MSE 中位數有 99.9%（L2）與 90%（L27）來自 position 0；L0 是 0.9%，L14 約 0。
  - 差異集中在這兩層，例如 L27 weight-only 組：mse.exact −0.55、BC −0.07、排除 position 0 後 +0.42，calibration S 參考值 0.99。
  - 即使用 gY² 加權（gw），position 0 也沒有被壓低。可見逐元素梯度加權沒有反映 sink 位置誤差對 loss 的實際影響；可能是 RMSNorm 與 attention 對這個位置的非線性或飽和，本研究沒有進一步驗證。
- **排除 outlier 後，聯合仍沒有幫助**：exact − additive 只有 +0.003（mse）與 +0.0004（gw）；相同資源配對 234 vs 233（mse）、229 vs 229（gw）。
- **下游曲率**：dir（一階平方）在所有組內最差（exact 0.597），而且 exact 比 additive 更差。每窗口的一階項會互相抵消，平方後放大雜訊。帶號一階 F 與 calibration S 的 pooled Spearman 是 0.82，|F|/|S| 中位數 0.66，所以一階項只解釋部分損傷，其餘來自曲率。
- **output metric 的限制**：Y 上的局部誤差即使算得精確，也要經過 27 − L 層與 LM head 才變成 NLL。本研究只證明聯合局部誤差沒有帶來更好的 loss 排序，不代表局部誤差算錯：FP32 重播與交叉比對都通過。

## 5. Native 核對（onnx-mlir 編譯執行；其餘結果都是 PyTorch 模擬）

4 個預先指定的不同 datatype 聯合配置，由 onnx-mlir 實際編譯：
- 編譯方式：`--convert-onnx-to-mixed-precision`、per-channel int 權重、Krnl -O3／clang -O0、相同的 calibration ranges；只有該層 Gate／Up 量化，其他節點 FP32。
- 在同樣 30 個窗口上執行，每窗口取 summed CE。
- FP32 基準重用 `runs/qwen3_branch_ablation_20261014` 的 native FP32：同一個 `onnx-mlir-opt` sha256、同樣的窗口，程式會檢查。

| 候選 | native S_cal | 模擬 S_cal | native S_tune | 模擬 S_tune | 每窗口最大 abs 差 | finite | warnings |
|---|---:|---:|---:|---:|---:|---|---:|
| L2 a：Gate INT4／Up INT8 | +587.1581 | +587.1609 | +1263.3158 | +1263.2983 | 0.018 | True | 0 |
| L2 a：Gate INT8／Up INT4 | +1948.3043 | +1948.3069 | +3794.8114 | +3794.8168 | 0.004 | True | 0 |
| L27 a：Gate P6／Up INT8 | +50.2649 | +50.2850 | +113.7786 | +113.8362 | 0.108 | True | 0 |
| L14 wa：Gate INT8／Up P6 | +8.6812 | +8.6644 | +19.1345 | +19.0774 | 0.028 | True | 0 |

- 模擬與 native 在有明顯損傷的配置上相對差 ≤ 2e-5。
- 含 posit 的配置，每窗口最大差 0.03–0.11 summed CE，約每 token 1e-3 以下，遠小於格式之間的差異。
- 前面各節的 PyTorch 模擬 ΔL 可以代表 onnx-mlir 實際執行的結果，包括 Gate／Up 互換造成的 3 倍差異。

## 6. 額外耗時與限制

| 階段 | 內容 | 時間 |
|---|---|---:|
| cache | FP32 30 窗口（10 個含 backward）、4 層張量、重播檢查 | 73 秒（前向／反向 27 秒） |
| table | 332 個候選的 block 分數（calibration 10 窗口） | **25 秒**（每層約 6 秒，每候選約 0.07 秒） |
| loss | 驗證用的下游重播，332 候選 × 30 窗口（L0 1124、L2 1078、L14 669、L27 268 秒） | 3139 秒 |
| explore | 事後診斷 | 93 秒 |
| native | 4 個配置 × 30 窗口，同時 2 個（build 50–76 秒，run 411–517 秒） | 1111 秒 |

- **建表成本**：block 重播本身很便宜。但若接入搜尋，每層的聯合 option 會是 (11 × 11)² = 14641 個。以每候選 0.07 秒估算，每層約 17 分鐘，28 層約 8 小時（PyTorch、CPU）；native 統計還需要新的實作。
- **驗證成本**：loss 欄是驗證成本，不屬於方法本身。

**限制**
- 只有 4 層、每層 83 個候選；Down 固定 FP32；其他層都是 FP32，沒有測和其他節點同時量化。
- tuning 只有 20 個窗口，與 calibration 同屬 validation split。
- 全模型 ΔL 是 PyTorch 模擬，只有 4 個配置用 native 核對。
- 排除 position 0 的改善是事後發現，需要新窗口驗證。
- I_output 的方向預測力，是在 int4 等損傷大的格式中最明顯。

## 7. 是否值得接入 MILP

**不建議接入。** 理由：
1. **沒有排序收益**：exact joint 的 block 成本在組內排序、相同資源選擇、互換決策上都沒有勝過 additive，也都不如目前的 BC。
2. **成本與規模**：(Gate format, Up format) 當成一個 option 時，每層 option 數從 2 × 121 變成 121² = 14641。MILP 仍是線性的，但變數多約 60 倍，建表約需 8 小時（估計）。
3. **尺度不一致**：output MSE 與原本的 BC 量綱不同。若只替換 Gate／Up 的成本，必須重新校準和其他節點 BC 的相對尺度；本研究沒有可用的校準依據。
4. **Down 也量化時**，block 邊界與跨邊界交互作用都需要重新檢查。

**值得追的方向**（node 層級，不需要 block option）：

| 方向 | 做法 | 理由 |
|---|---|---|
| 確認 outlier 效應 | 預先登記「排除 position 0」的 output 與 BC 變體，在新的 validation 窗口（例如 30–59）上驗證 | 本研究的改善是事後發現 |
| 檢查 BC 本身 | 看 BC 的激活統計是否也被 position 0 主導 | 若是，改成在 node 層級處理 sink 位置，例如排除或分開統計 |
| 只把 I_output 當診斷 | 用來標記可能互相抵消或放大的 Gate／Up 組合，不加進成本 | 它能預測方向，但量級小 |

## 8. 重跑

在 `experiments/mixed_precision/` 下執行：

```bash
O=runs/qwen3_block_criterion_20261014       # 新實驗請換一個新的空目錄
PY=../../venv/bin/python
$PY run_qwen_block_criterion.py --stage cache   --out-dir $O
$PY run_qwen_block_criterion.py --stage plan    --out-dir $O     # 凍結候選與指標
$PY run_qwen_block_criterion.py --stage table   --out-dir $O
$PY run_qwen_block_criterion.py --stage loss    --out-dir $O --layers 27,14
$PY run_qwen_block_criterion.py --stage loss    --out-dir $O --layers 2,0
$PY run_qwen_block_criterion.py --stage analyze --out-dir $O
$PY run_qwen_block_criterion.py --stage native  --out-dir $O
$PY run_qwen_block_criterion.py --stage explore --out-dir $O     # 事後診斷
$PY run_qwen_block_criterion.py --stage tables  --out-dir $O
```

**輸出檔案**

| 檔案 | 內容 |
|---|---|
| `cache.npz` | FP32 張量快取，741 MB |
| `baseline.json` | 重播檢查與每窗口 loss |
| `plan.json` | 凍結的計畫 |
| `scores.json` | 分數表 |
| `loss_L*.json` | 每候選 30 窗口的 ΔL |
| `analysis.json`、`candidates.csv`、`per_window.csv` | 分析結果與原始數據 |
| `explore.json` | 事後診斷 |
| `native.json`、`native/*/result.json`、`native/*/compile_env.json` | native 結果與編譯環境 |
| `environment*.json` | 版本與 hash |
| `tables.md` | 報告用的表格 |
