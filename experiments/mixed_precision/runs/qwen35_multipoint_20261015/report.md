# Qwen3.5-0.8B：多資源點完整配置驗證（BC frontier 上的 aggressive／knee／conservative）

日期：2026-10-15　腳本：`experiments/mixed_precision/run_qwen35_multipoint.py`　輸出：本目錄
來源（只讀）：`runs/qwen35_bc_20261015`（成本表、搜尋、frontier、knee）

本輪**沒有修改** BC 公式、梯度與正規化、calibration 資料與 range、posit LUT 與 QDQ 語意、MILP → 2-D Pareto → normalized chord knee 的預設流程、格式集合，也沒有覆寫任何既有產物。三個資源點只是研究用的控制條件，正式 pipeline 仍然自動選 knee，不需要人工輸入預算。

## 結論

標記說明：**[量測]** 實際量到；**[推論]** 由量測推得、未直接驗證；**[未驗證]** 本輪沒有資料。

1. **BC 的完整配置趨勢是否合理？是。** [量測]
   - 兩個搜尋範圍裡，BC 預測的損傷順序都是 aggressive > knee > conservative，native 實測的順序也一樣。
   - 「aggressive 比 knee 差」在兩個範圍都明確：區間不含 0，兩個評估區段各自也成立，三種實作的方向一致。
   - 「conservative 比 knee 好」：
     - FFN + attention 明確（−0.0176 [−0.0248, −0.0107]）；
     - FFN-only 未解析（−0.0024 [−0.0066, +0.0016]），兩個區段甚至方向不一致。
2. **knee 附近有沒有值得處理的穩定問題？沒有。** [量測]
   - 三個點的實測順序沒有和 BC 衝突。
   - 實測的「損傷比值」只有一處落在 BC 比值的區間外：FFN-only conservative／knee，實測 0.76 [0.49, 1.27]，BC 是 0.43。
   - 這表示 FFN-only 往保守方向的實測收益比 BC 預期小，但也沒有證據顯示 knee 選錯方向。[推論] 而且這個比較要先假設 BC 與 ΔNLL 成正比，而這一點本身沒有驗證。
3. **要不要把原因診斷集中到激進配置？沒有這個理由。** [量測]
   - aggressive 的損傷較大，但 BC 也預測它較大。
   - 實測的 aggressive／knee 比值：FFN-only 3.27 [2.29, 5.40]、FFN + attention 1.91 [1.61, 2.32]，BC 比值 2.53／2.02 都在區間內。
   - 這是正常的壓縮代價，不是 BC 失準的證據。
4. **下一步**：做同資源的 baseline 比較（§7），不做一階／二階／擾動縮放分析。這輪在完整配置層級沒有找到 BC 的失敗案例，針對失敗案例的分析沒有對象。

**這輪支持的**
- 原版 BC 在 Qwen3.5 上的完整配置趨勢與實測一致。
- 自動 knee 落在「再壓縮就明顯變差」的位置。

**這輪排除的**
- 「BC 在比 knee 更激進的配置失準」：三個點上沒有看到。
- 「knee 附近有需要修正的系統性錯誤」：沒有看到。

**還不能宣稱的**
- knee 是實測的全域最佳點：三個點只能驗證局部趨勢。
- BC 選出的配置比 SQNR 或 HTQ 更好，或在硬體上更快：本輪沒有執行任何 baseline，FP32+QDQ 的執行時間也不代表低位元 kernel 的速度。
- conservative 改善的真實幅度：三種實作在子集上的方向一致，但幅度差了好幾倍（§4）。

## 0. 既有報告與資料的核對（`audit.json`）

從 JSON 重算四份報告的 36 個關鍵數字。

**35 個一致**，包括：
- 決定性配對 934/1057、SQNR 944；
- knee 的 PPL、ΔNLL 與區間；
- frontier 點數；
- recalibration 的 107／138、22/23、Spearman 0.990；
- 逐元素研究的 650/650；
- knee Gate／Up 研究的 L4／L20／L21、−0.0004、PPL 44.101。

**1 個不一致**（以資料為準）
- `qwen35_bc_20261015/report.md` §4 寫「Down 多為 int7/p8e1」。
- FFN-only knee 是對的（15/24），但 FFN + attention knee 的 Down 是 int7/p8e1 和 int6/p8e1 各 10 個（平手），並不是「多為 int7/p8e1」。

**另外確認**：knee Gate／Up 研究的評估重用了主實驗的 final test（test token 0–8191），所以那次不算新的獨立驗證。本輪改用全新的窗口。

## 1. 設計（`plan.json`、`freeze.json`：先凍結，才跑新窗口）

**評估窗口**：兩個不重疊的區段，各 32 個 128-token 窗口，共 64 個。
- 區段 V：validation token 131072–135167
- 區段 T：test token 131072–135167
- 用的是模型自己的 tokenizer，不加特殊 token，每個窗口獨立（狀態重置）。
- 位置是在看到任何結果之前，以遠離所有已用範圍的固定偏移選的；沒有依模型表現挑選，也沒有更換。

**已用過的範圍**（Qwen3.5 tokenizer）
- validation：0–1279（calibration A）、1280–5119（diagnostic）、8192–14335（512 tokens）、16384–17663（calibration B）、20480–24319（逐元素研究）、24576–28415（knee Gate／Up 選擇）；
- test：0–8191（final test）。
- 早期的 Qwen3／GPT-2 研究用其他 tokenizer 看過這兩個 split 的前段，但沒有對 Qwen3.5 的配置做過任何決定。

**選點規則**（只用 R 與 BC；R = max(M/M₈, B/B₈)）
- 讀 `selection_{ffn,ffn_attn}.json` 的實際 frontier，不重建成本表，也不重新搜尋。
- **aggressive**：R < R_knee 且配置不同於 knee 的 frontier 點中，R 最接近 0.8 R_knee 的（平手取較低 R）。
- **conservative**：R > R_knee 的點中，最接近 1.25 R_knee 的。
- 若選到退化點（frontier 最低 R 端點，或 BC = 0），改取該側最接近 knee 的點（本輪沒有觸發）。
- knee 原封不動，並核對它和 `selected_formats_*.json` 是同一個配置。

**native 評估**
- 與主實驗 evaluate 完全相同的設定：MIXEDP_WEIGHT_PER_CHANNEL=1、Krnl -O3、clang -O0、constants to file、bytecode import、主實驗的 calibration range、posit runtime、5 個 process。
- 主實驗評估後已刪除 .so，所以 7 個模型（FP32 + 6 個配置）全部用同一套設定重新編譯。兩個 knee 產生的 `MIXEDP_NODE_FORMATS` 與主實驗存的 `compile_env.json` 逐字相同。
- 每個窗口輸出 summed CE（float64）、有效 token 127 個、NLL；PPL = exp(總 CE／總 token)。

**統計**
- 2000 次分層 bootstrap：每次在兩個區段內各重抽 32 個窗口。
- 所有配置與差值共用同一組抽樣索引（seed 0）；單一區段的結果取同一組抽樣中該區段的部分。

**跨實作核對**
- 預先指定的子集：每個區段的前 16 個窗口，共 32 個。
- GPU 模擬（RTX 3060，FP32，關閉 TF32）跑全部 7 個模型。
- 若 GPU 與 native 的順序或相鄰差的方向不一致，或任一點的差異超過該範圍最小 native 相鄰差的一半，就在同一子集上再跑 CPU 模擬。
- 不使用先前的 0.0012／0.0023 當通用雜訊下限。

**修正紀錄**（`freeze.json` 的 `amendments`）
- `stage_plan` 拆包三個點的順序寫錯，導致 `selected_points.json` 的描述欄位 `vs_knee`（相對 knee 的資源變化）比錯了點。
- 在 GPU 子集模擬之後、任何 native 評估之前修正，只重算這個欄位。
- 選出的配置、格式（hash 逐一核對相同）、BC、M、BOPs、R、窗口都沒有變。

## 2. 選出的配置（`selected_points.json`）

M 是被搜尋權重的 bytes；BOPs 是代理指標，等於 Σ（MAC × 權重位元 × 激活位元），用 128-token 窗口計算。比例都相對於同一範圍的全 int8。

**FFN-only**（72 個節點；全 int8：264.2 MB、2165 GBOPs）

| 點 | 選點依據 | M | M／int8 | BOPs | BOPs／int8 | R | BC 總分 | 權重格式 | 激活格式 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| aggressive | 最接近 0.701 | 186.7 MB | 0.707 | 1530 G | 0.707 | 0.707 | 89.7 | int5 34、int6 29、int7 9 | p8e1 66、p6e1 5、f16 1 |
| knee | 自動 knee | 218.4 MB | 0.826 | 1898 G | 0.877 | 0.877 | 35.4 | int6 34、int7 32、int8 6 | p8e1 68、f16 4 |
| conservative | 最接近 1.096 | 245.9 MB | 0.931 | 2571 G | 1.188 | 1.188 | 15.2 | int7 40、int8 32 | p8e1 53、f16 19 |

**FFN + attention**（186 個節點；全 int8：497.6 MB、4077 GBOPs）

| 點 | 選點依據 | M | M／int8 | BOPs | BOPs／int8 | R | BC 總分 | 權重格式 | 激活格式 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| aggressive | 最接近 0.673 | 351.9 MB | 0.707 | 2882 G | 0.707 | 0.707 | 164.8 | int5 75、int6 69、int7 29、int8 13 | p8e1 138、f16 38、p6e1 7、int8 3 |
| knee | 自動 knee | 397.6 MB | 0.799 | 3428 G | 0.841 | 0.841 | 81.6 | int6 82、int7 64、int8 31、int5 9 | p8e1 136、f16 47、int8 3 |
| conservative | 最接近 1.051 | 437.1 MB | 0.878 | 4076 G | 1.000 | 1.000 | 47.7 | int7 93、int8 59、int6 32、bf16 2 | p8e1 124、f16 62 |

**資源的逐項關係**
- 兩個範圍的 aggressive，M 與 BOPs 都低於 knee；conservative 都高於 knee。所以三點在兩項資源上都有逐項的支配關係，沒有「R 較大但其中一項下降」的情況。
- 兩個範圍的 R 分母與可量化範圍不同（72 vs 186 個節點），normalized resource 不能跨範圍比較。

## 3. 結果（native，64 個新窗口）

FP32：NLL 3.65697，PPL 38.744（V 3.8617、T 3.4523）。7 個模型的 logits 全部有限；用 float64 log-sum-exp 重算兩個窗口的 CE，與評估值完全一致（差 0）。逐窗口資料在 `per_window.csv`。

### A. 每個範圍的表

**FFN-only（主要）**

| 點 | NLL | PPL | ΔNLL vs FP32 [95% CI] | 區段 V | 區段 T |
|---|---:|---:|---|---|---|
| aggressive | 3.68888 | 40.000 | +0.0319 [+0.0260, +0.0380] | +0.0272 [+0.0180, +0.0365] | +0.0367 [+0.0292, +0.0451] |
| knee | 3.66674 | 39.124 | +0.0098 [+0.0057, +0.0137] | +0.0100 [+0.0045, +0.0153] | +0.0095 [+0.0033, +0.0155] |
| conservative | 3.66437 | 39.032 | +0.0074 [+0.0052, +0.0095] | +0.0044 [+0.0013, +0.0073] | +0.0105 [+0.0074, +0.0135] |

| 相鄰差（配對） | 合併 | 區段 V | 區段 T |
|---|---|---|---|
| aggressive − knee | **+0.0221 [+0.0161, +0.0286]** | +0.0172 [+0.0085, +0.0263] | +0.0271 [+0.0182, +0.0359] |
| conservative − knee | −0.0024 [−0.0066, +0.0016] | −0.0057 [−0.0117, +0.0002] | +0.0009 [−0.0053, +0.0065] |

**FFN + attention（次要）**

| 點 | NLL | PPL | ΔNLL vs FP32 [95% CI] | 區段 V | 區段 T |
|---|---:|---:|---|---|---|
| aggressive | 3.71873 | 41.212 | +0.0618 [+0.0524, +0.0713] | +0.0556 [+0.0445, +0.0667] | +0.0679 [+0.0532, +0.0840] |
| knee | 3.68927 | 40.016 | +0.0323 [+0.0254, +0.0390] | +0.0315 [+0.0221, +0.0415] | +0.0331 [+0.0241, +0.0425] |
| conservative | 3.67166 | 39.317 | +0.0147 [+0.0101, +0.0193] | +0.0144 [+0.0084, +0.0207] | +0.0150 [+0.0080, +0.0218] |

| 相鄰差（配對） | 合併 | 區段 V | 區段 T |
|---|---|---|---|
| aggressive − knee | **+0.0295 [+0.0214, +0.0378]** | +0.0240 [+0.0129, +0.0352] | +0.0349 [+0.0228, +0.0467] |
| conservative − knee | **−0.0176 [−0.0248, −0.0107]** | −0.0171 [−0.0284, −0.0066] | −0.0181 [−0.0274, −0.0087] |

**統計的限制**
- 64 個窗口來自兩段連續文字，同一段內的窗口相鄰，可能有主題相關；bootstrap 把窗口當成可交換的單位，會低估這種相依性。
- 區間只涵蓋窗口抽樣，不涵蓋跨實作差異（§4）。

**窗口集合的差異**
- 同一個 knee 在主實驗 final test 上的 ΔNLL 是 0.0073（FFN-only）與 0.0210（FFN + attention），在這批新窗口上是 0.0098 與 0.0323。[量測]
- 這批窗口的 FP32 NLL 比 final test 低（3.657 vs 3.780），但量化損傷較大；兩個新區段之間則相當一致。

### B. BC 預測與實測的趨勢

| | BC：aggr／knee／cons | 實測 ΔNLL：aggr／knee／cons | 順序一致 | aggr − knee | cons − knee |
|---|---|---|---|---|---|
| FFN-only | 89.7／35.4／15.2 | 0.0319／0.0098／0.0074 | 是 | 區間 > 0 | 區間含 0 |
| FFN + attention | 164.8／81.6／47.7 | 0.0618／0.0323／0.0147 | 是 | 區間 > 0 | 區間 < 0 |

**描述性比值**（相對 knee）
- BC 的尺度與 ΔNLL 不同，本輪沒有定義或校準兩者的換算，所以不計算絕對誤差。
- 下表的比較隱含「BC 與 ΔNLL 成正比」的假設，這個假設沒有驗證。

| | BC 比值 | 實測比值 [95% CI] |
|---|---:|---|
| FFN-only aggressive／knee | 2.53 | 3.27 [2.29, 5.40] |
| FFN-only conservative／knee | 0.43 | 0.76 [0.49, 1.27] |
| FFN + attention aggressive／knee | 2.02 | 1.91 [1.61, 2.32] |
| FFN + attention conservative／knee | 0.58 | 0.45 [0.31, 0.61] |

只有 FFN-only conservative 的 BC 比值落在實測區間外：BC 預期損傷減少 57%，實測約 24%，而且區間含 0 與增加。三個點的 Spearman 沒有統計意義，本報告不用它。

### C. knee 的折衷（相對 knee）

| | M | BOPs | ΔNLL 變化 [95% CI] |
|---|---|---|---|
| FFN-only aggressive | −31.7 MB（−14.5%；M／int8 0.826 → 0.707） | −19.4%（0.877 → 0.707） | +0.0221 [+0.0161, +0.0286]；aggressive 總損傷約為 knee 的 3.3 倍 |
| FFN-only conservative | +27.5 MB（+12.6%；0.826 → 0.931） | +35.4%（0.877 → 1.188） | −0.0024 [−0.0066, +0.0016] |
| FFN + attention aggressive | −45.7 MB（−11.5%；0.799 → 0.707） | −15.9%（0.841 → 0.707） | +0.0295 [+0.0214, +0.0378]；總損傷約為 knee 的 1.9 倍 |
| FFN + attention conservative | +39.5 MB（+9.9%；0.799 → 0.878） | +18.9%（0.841 → 1.000） | −0.0176 [−0.0248, −0.0107]；總損傷約為 knee 的 0.45 倍 |

**[量測]**
- **FFN-only**：往激進方向省 15–19% 資源，損傷變成約 3 倍；往保守方向多用 13% M、35% BOPs，損傷的減少在統計上分辨不出來。knee 正好在「再省就明顯變差、再加也幾乎沒收益」的位置。
- **FFN + attention**：往保守方向多用 10% M、19% BOPs，換到明確的損傷下降，但幅度有實作依賴（§4）。

**[推論]**
- 兩個範圍的 knee 都位於合理的折衷區域。是否值得多付這些資源，是使用者的取捨，不是 BC 的錯。
- 本輪沒有根據評估結果重選 knee；三個點也不能證明現有 knee 是實測的全域最佳。

### D. aggressive 的格式變動（相對 knee；`analysis.json` 的 `format_changes`）

**FFN-only**：72 個節點全部變動，基本上是每個節點降一階。
- 權重 int6→int5：Gate 16+2、Up 9+1、Down 4；int7→int6：Gate 6、Up 8、Down 13；int7→int5：Down 2；int8→int7：Up 4、Down 1（另有 1 個 Down int8/f16→int7/f16）。
- 激活：p8e1→p6e1 共 5 個（Gate 2、Up 3）；Down 有 3 個 f16→p8e1。
- int5 權重共 34 個。

**FFN + attention**：149/186 個節點變動，同樣是全面降一階（int6→int5、int7→int6 為主），涵蓋所有模組種類；`in_proj_a`／`in_proj_b` 的權重也各降一階。

**[未驗證]**：低位元（int5）、跨層交互作用、大擾動，都只是 aggressive 損傷較大的可能來源。BC 並沒有排序錯誤：比值在區間內，順序一致。所以這輪沒有「BC 在激進端失效」的證據，aggressive 較大的損傷是預期中的壓縮代價。

## 4. 跨實作核對（子集：每個區段的前 16 個窗口，共 32 個；`sim_cuda.json`、`sim_cpu.json`）

**觸發 CPU 核對**（預先登記的規則 (c)）
- GPU 和 native 的三點順序與相鄰差方向，兩個範圍都一致。
- 但單點差異超過「最小 native 相鄰差的一半」：FFN-only 0.0019 > 0.0012，FFN + attention 0.0071 > 0.0054。

**子集 ΔNLL vs FP32**（native／GPU／CPU）

| 配置 | native | GPU | CPU | 單窗口最大差（GPU、CPU vs native） | 逐窗口相關（GPU、CPU） |
|---|---:|---:|---:|---|---|
| FFN aggressive | 0.0318 | 0.0299 | 0.0307 | 0.036、0.027 | 0.75、0.76 |
| FFN knee | 0.0088 | 0.0099 | 0.0091 | 0.034、0.037 | 0.36、0.54 |
| FFN conservative | 0.0064 | 0.0065 | 0.0029 | 0.014、0.024 | 0.73、0.52 |
| FFN+attn aggressive | 0.0550 | 0.0527 | 0.0567 | 0.045、0.048 | 0.89、0.81 |
| FFN+attn knee | 0.0237 | 0.0166 | 0.0191 | 0.057、0.042 | 0.68、0.78 |
| FFN+attn conservative | 0.0129 | 0.0139 | 0.0160 | 0.039、0.037 | 0.53、0.48 |

FP32 本身：GPU 與 native 的單窗口差 ≤ 2.6e-6，CPU ≤ 6.7e-5。

**子集相鄰差**（native／GPU／CPU）

| | aggressive − knee | conservative − knee |
|---|---|---|
| FFN-only | +0.0230／+0.0200／+0.0215 | −0.0024／−0.0035／−0.0063 |
| FFN + attention | +0.0314／+0.0361／+0.0375 | **−0.0107／−0.0027／−0.0031** |

**判讀**
- **窗口抽樣的不確定性** [量測]：§3 的 bootstrap 區間。
- **跨實作差異** [量測]：同一個量化配置，在三種實作之間的單窗口差最大 0.06 nat／token，32 個窗口的平均可差 0.007（FFN + attention knee）。FP32 本身一致，所以差異來自量化。
- **語意錯誤的可能** [推論]：
  - FFN + attention 的 conservative − knee，兩個模擬彼此接近（−0.0027、−0.0031），native 較大（−0.0107），主要因為 native 的 knee 比兩個模擬高 0.005–0.007。
  - 但在主實驗的 final test 上，native 的 knee 反而比兩個模擬低（0.0210 vs 0.0213、0.0233）。
  - 偏差方向不固定，比較像實作雜訊，不像固定的語意偏差。本輪無法完全排除語意差異，它未經驗證。
- **對結論的影響**：
  - 所有「方向」結論（順序、aggressive 較差、conservative 不比 knee 差）在三種實作中都成立；
  - FFN + attention conservative 收益的**幅度**不穩定：native 64 個窗口是 −0.018，兩個模擬在子集上約 −0.003。

## 5. 成本（`timing.json`）

| 項目 | 時間 | 備註 |
|---|---:|---|
| 重用的既有工作 | — | 主實驗的成本表（1854 s）與搜尋（53 s）；本輪沒有重做 |
| 配置提取與凍結（plan） | 3 s | 只讀 frontier |
| 舊報告核對（audit） | < 1 s | |
| native 編譯 | FP32 39 s；配置 52–65 s | 每個配置一次，共 7 次 |
| native 評估（64 窗口，5 個 process） | 622–708 s／模型 | 7 個模型合計 4672 s |
| native stage 合計 | 5170 s（約 86 分鐘） | 7 次獨立執行，各 693–777 s |
| GPU 模擬（7 × 32 窗口） | 45 s | |
| CPU 模擬（7 × 32 窗口） | 335 s | 依規則觸發 |
| 分析與圖 | 2 s | |

- **峰值記憶體**：
  - native 執行時每個 process 7.7 GB；
  - native stage 的父 process 15.6 GB（logits 約 8 GB）；
  - 最大子 process（編譯）9.0–11.7 GB；
  - CPU 模擬 15.2 GB，GPU 模擬（host 端）10.1 GB。
- **失敗與重試**：native、sim、analyze 都沒有失敗或重試。唯一的修正是 §1 的描述欄位錯誤，不影響任何量測。
- 評估時間不屬於搜尋時間。FP32+QDQ 的執行時間（各配置相近）不代表低位元 kernel 的硬體速度。

## 6. 判斷

**[量測]**
- 兩個範圍中，BC 的損傷順序與 native 實測一致。
- aggressive 比 knee 明確變差；FFN + attention 的 conservative 明確變好，FFN-only 則分辨不出。
- 三種實作在方向上一致，在幅度上可差到 0.007（32 窗口平均）。

**[推論]**
- BC 的 frontier 在 knee 附近的局部趨勢可信。knee 位於「往下壓縮代價高」的區域。
- FFN-only 往保守方向的實測收益小於 BC 的比例預期，這只會讓 knee 顯得更合理，不會更不合理。

**[未驗證]**
- knee 是否為實測全域最佳。
- 其他資源點的 BC 準度。
- 跨實作差異的根本原因。
- BC 與其他 criterion 在同資源下的優劣。

## 7. 建議的下一步：同資源 baseline 比較（本輪沒有啟動）

這輪沒有完整配置層級的 BC 失敗案例，所以不建議做一階／二階／擾動縮放分析。能直接回答「BC 選出的配置好不好」的，是同資源下和其他 criterion 比較。

**公平比較需要的條件**
- **範圍**：FFN-only（72 個節點），也就是主要分析的範圍。FFN + attention 可以之後再做。
- **候選集合**：相同的 GRID11 × GRID11 格式、相同的 per-consumer QDQ、range 與 posit LUT。SQNR 要用 PyTorch 重播實作（主實驗已用），因為 onnxruntime 無法模擬 posit 激活。
- **資源限制**：用同樣的 MILP、同樣的 R 定義，在本輪的三個預算（R = 0.707、0.877、1.188）各解一次 SQNR 配置，與 BC 配置同 M、同 BOPs 上限比較；另外也比較各 criterion 自己的 knee。
- **評估**：native 編譯；用**另一批新窗口**，因為本輪的 64 個窗口已經看過，不能再當獨立驗證；子集同時跑 GPU／CPU 模擬核對；判準與區間方法事先登記。
- **成本估計**（外推）：
  - SQNR 表（72 節點 × 120 個 (w,a) 選項 × 10 個 calibration 窗口）：GPU 約 45 分鐘，CPU 約 14 小時；
  - 3 個預算的 MILP：約 30 s；
  - native 評估 3 個 SQNR 配置與重評 BC 配置（新窗口）：約 6 × 12 分鐘；
  - 合計約 2–2.5 小時，GPU 可用時。
  - HTQ 原版 criterion 需要另外移植與估計，不在這個估計內。

## 8. 限制

- 每個範圍只有三個點，只能驗證局部趨勢，不能描繪完整的實測曲線。
- 64 個窗口來自兩段連續文字；bootstrap 不涵蓋段內相依性，也不涵蓋跨實作差異。
- 跨實作核對只用了 32 個窗口的子集。
- FFN + attention 的資源分母與 FFN-only 不同，兩個範圍不能用 normalized resource 比較誰更省。
- BC 與 ΔNLL 的比值比較假設兩者成正比，這個假設未驗證。
- 沒有執行任何 SQNR／HTQ baseline。

## 9. 重跑與檔案

```bash
cd experiments/mixed_precision
PY=../../venv/bin/python; O=runs/qwen35_multipoint_20261015      # 新實驗請換新目錄
GPU="env PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$PY  run_qwen35_multipoint.py --stage audit   --out-dir $O
$PY  run_qwen35_multipoint.py --stage plan    --out-dir $O     # 凍結後拒絕重跑
$GPU run_qwen35_multipoint.py --stage sim     --out-dir $O --device cuda
for c in fp32 ffn_knee ffn_aggressive ffn_conservative ffn_attn_knee ffn_attn_aggressive ffn_attn_conservative; do
  $PY run_qwen35_multipoint.py --stage native --out-dir $O --configs $c    # 已完成的配置會略過（續跑）
done
$PY  run_qwen35_multipoint.py --stage sim     --out-dir $O --device cpu    # 依規則觸發
$PY  run_qwen35_multipoint.py --stage analyze --out-dir $O
```

| 檔案 | 內容 |
|---|---|
| `plan.json` | 範圍、已用 token 範圍、新窗口、選點規則、指標、bootstrap、跨實作規則、編譯設定與續跑方式 |
| `environment.json` | git、程式與輸入檔 hash、模型 revision、版本 |
| `selected_points.json`、`formats/*.json` | 六個配置的格式、BC、M、BOPs、R、相對 knee 的變化 |
| `freeze.json` | 凍結 hash 與修正紀錄（`amendments`） |
| `tokens_eval.npy` | 64 個評估窗口 |
| `native/*.json`、`native/*/compile_env.json` | 逐窗口 summed CE、NLL、PPL、編譯／執行時間與設定 |
| `per_window.csv` | 每個模型、每個窗口的 summed CE、token、NLL、ΔNLL |
| `sim_cuda.json`、`sim_cpu.json` | 子集的模擬結果 |
| `analysis.json` | A–D 的所有數字與跨實作比較 |
| `audit.json` | 舊報告核對 |
| `fig_ffn.png`、`fig_ffn_attn.png` | BC 與實測 ΔNLL 對資源（size、BOPs 分開；只畫點，不畫曲線） |
| `timing.json`、`*.log`、`native.out` | 每次執行的時間、狀態與峰值記憶體 |
