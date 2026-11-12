# EfficientNet-B0 weight mixed-precision 的 baseline 比較（A8 固定）

日期：2026-10-29 執行，2026-11-12 撰寫報告。程式：`run_cnn_baselines.py --model efficientnet_b0`。計畫：`plan.json`／`plan.md`（10:40:17，早於任何評分），修訂見 `amendments.json`（4 筆）。

**範圍限定**：
- 只適用於 timm `efficientnet_b0.ra_in1k`／ImageNet、activation 固定 INT8 per-tensor、weight 候選 INT4／6／8 的 weight mixed precision。
- 比較的是「四種評分在同一個搜尋器下選出的配置」，不是各論文的完整原生 pipeline。HTQ、HAWQ 都是受控改寫，不是原論文復現。
- **InfoQ 未執行**：InfoQ 沒有這個模型的設定（沒有 observer、也沒有 layer-to-block 對應）。依 2026-10-29 使用者的決定標示為未執行，沒有用其他分數頂替它的名稱。
- 方法定義、量化器、資源公式、統計方式都和 ResNet18 run 相同，見 `../cnn_baselines_20261029/report.md`。本報告只寫不同之處與結果。

## 先讀這段：比較所在的區間

**A8 固定在這個模型上代價很大。**
- FP32 top-1 78.28%；all-INT8（W8A8）參考只有 44.40%，掉了 33.9 pp。
- 所以下面所有的比較，都是在 activation 已經嚴重損傷的模型上比較 weight 分配。
- 先前的 native 自動選點（`runs/native_auto_2d_20261011/efficientnet_b0`，可選 FP16／Posit activation）只在 24／82 個節點用 INT8 activation，其餘用 f16（21 個）、p8e1（26 個）、p6e1 等，top-1 77.4%（另一個 1000 張的評估集，不能直接相比）。這和「EfficientNet 對 INT8 per-tensor activation 很敏感」一致。
- 本 run 沒有診斷 INT8 損傷來自哪裡，例如 SiLU／SE 的 range，或 depthwise 每通道 range 不均（見先前 MobileNetV2 的 outlier range 結果）。

**在這個區間裡，單一節點的改變就造成約 0.7 pp 的變動。**
- BC 在 HAWQ 自動點的 R（0.641016）重新搜尋，和 BC 自己的 knee（R 0.641028）只差一個節點：`blocks.0.0.se.conv_expand` int6 → int4。
- 兩者的 top-1 差 +0.73 pp [−0.17, +1.62]（McNemar 150/124，p = 0.13，事後、描述性）。
- 另外，寬鬆點的 BC 配置（大部分是 int6／int8 weight）比 all-INT8 參考高 +1.06 pp [+0.17, +1.99]（163/125，未校正 p = 0.029，事後、描述性）：weight bit 降低反而略好。
- 所以 ±1 pp 級的方法差異，要當成接近此區間的配置雜訊來讀。

## 結論

- **相同資源上限下，沒有方法顯著勝過 BC，但 BC 也不是每個點都最好。**
  - SQNR：寬鬆點顯著較差（−1.48 pp，Holm p = 0.027），knee 與激進點未定（−0.50、−0.25 pp）。
  - HTQ：三個點都顯著較差（−2.35／−15.87／−9.74 pp）。
  - HAWQ：寬鬆、knee 未定（−0.48、−0.53 pp）。**激進點 HAWQ 的點估計比 BC 高 +1.09 pp [+0.11, +2.04]**，CE 也較低（−0.050 [−0.073, −0.027]），但 Holm p = 0.15，判定未定。
  - 沒有任何方法和 BC 選出同一配置。ResNet18 上 SQNR／HAWQ 三點都同配置，這裡不再如此。
- **共同自動外殼下，四個方法的 R 幾乎相同（0.636–0.641）**，不像 ResNet18 有資源差異。
  - BC 39.84%、SQNR 39.56%、HAWQ 39.31%、HTQ 23.49%。
  - 同 R 補比（C，預先登記）：SQNR −0.70 pp 未定；HTQ −13.38 pp、HAWQ −1.26 pp，BC 顯著較好。
  - 但 HAWQ 這一項要打折扣：對比的 BC 配置恰好是上面那組「只差一個節點就 +0.73 pp」的配置。和 BC 自己的 knee 比，HAWQ 只差 −0.53 pp（表 1 knee，未定）。
- **離線成本**：
  - 表格時間中位數：HTQ 1.1 s、SQNR 28.0 s（0.63 倍）、BC 44.8 s（冷啟動；熱啟動推估約 5.1 s）、HAWQ 3512 s（78 倍）。
  - BC 的冷啟動時間有 89% 是 grads .so 的建置（39.6 s）。
  - HAWQ 必須切 batch 才放得進 12 GB GPU，跑了 2283 個 Hutchinson probe。它在此模型上是唯一點估計曾勝過 BC 的方法，但成本高兩個數量級。
- **建議：BC 準則不改。**
  - 沒有方法在相同上限下顯著勝過 BC。
  - 唯一的正向點估計（HAWQ 激進點）Holm 後未定，量級也落在上面說的配置雜訊範圍內。
  - 更大的問題是 A8 per-tensor 本身：在這個模型上，weight 分配的品質差異被 34 pp 的 activation 損傷蓋過。下一步若要在 EfficientNet 上分辨 weight 評分，應先換到 activation 可用的設定（例如允許 FP16／per-channel activation），而不是改 BC。

## 六個問題

1. **相同資源上限下，BC 的 accuracy 是否有競爭力？**
   有。9 個比較中，BC 顯著較好的有 4 個（SQNR 寬鬆點、HTQ 三點），其餘 5 個未定，沒有方法顯著勝過 BC。唯一的反向點估計是 HAWQ 激進點的 +1.09 pp（Holm p = 0.15）。
2. **更準的方法需要多少額外成本？**
   沒有方法顯著更準。點估計較高的只有 HAWQ 激進點，它的成本是 BC 的 78 倍（GPU，64 張圖，2283 個 probe）。
3. **自動選出的 BOPs／size 差多少？**
   四個方法的 R 都受 size 綁住，彼此差不到 0.005。

   | 方法 | size | BOPs | MB | GBOPs |
   |---|---|---|---|---|
   | BC | 0.6410 | 0.6205 | 3.357 | 15.32 |
   | SQNR | 0.6409 | 0.6295 | 3.356 | 15.54 |
   | HTQ | 0.6362 | 0.6357 | 3.331 | 15.70 |
   | HAWQ | 0.6410 | 0.6359 | 3.357 | 15.70 |

   BOPs 有差：SQNR、HTQ、HAWQ 比 BC 多 1.5–2.5% BOPs。在 size 綁住時，BC 用最少的 BOPs。
4. **表面優勢是否只是用了更多資源？**
   這次自動外殼裡沒有方法的 top-1 高於 BC，所以沒有需要解釋的表面優勢。同 R 補比（C）的結果見表 C：SQNR 未定，HTQ、HAWQ 都是 BC 顯著較好，但 HAWQ 那項對 BC 配置的選擇很敏感（見「先讀這段」）。
5. **要修改 BC 還是保留？**
   保留、不改。理由見結論。與 ResNet18 不同的是，這裡 BC 和 HAWQ 選出不同配置（Spearman ρ：int4 0.86、int6 0.87，ResNet18 為 0.97）。差異可能來自 HAWQ 的每層 trace 與 BC 的每通道 mean g²，但差異的方向不一致：knee 處 BC 點估計較好，激進點 HAWQ 較好。
6. **HAWQ 的負 trace 有沒有影響？**（本模型新增）
   兩個節點的平均 trace 為負，依預設規則成本設為 0：
   - `node_Conv_779`（conv_stem）本來就固定 INT8，沒有影響。
   - `node_Conv_787`（blocks.1.0.conv_dw）在搜尋範圍內：成本 0 使 HAWQ 在 knee 與自動點都把它放在 int4，BC 則放 int8。它的 trace 只用了 3 個 probe 就達到收斂判準（平均 −39.6、標準差 51.9），估計很不穩。
   - 這個節點對 HAWQ knee 結果的貢獻沒有單獨量測。

## 設定（與 ResNet18 的差異）

- **模型**：
  - timm `efficientnet_b0.ra_in1k`，82 個 weight node，ONNX 已把 BN 折入 conv。
  - SE 的 `conv_reduce`／`conv_expand` 沒有 BN。
  - PyTorch 評分器從 HF hub 的 `model.safetensors` 載入同一個 checkpoint。
- **搜尋空間**：80 個節點；`node_Conv_779`（conv_stem）與 `node_linear`（classifier）對所有方法固定 INT8 weight。
- **前處理**：timm 設定（bicubic、crop 0.875）。
- **Activation ranges**：固定沿用 `runs/native_auto_2d_20261011/efficientnet_b0/ranges_by_point.json`（sha256 `9b932ece…`），來自 10 張 validation 校準圖，從未重算。
- **資源**：
  - all-INT8 參考為 5,236,192 bytes、24.69 GBOPs。
  - 三個上限實際綁住的都是 size（ResNet18 綁的是 BOPs）。
- **校準池與評估集**：與 ResNet18 run 是同一批，以 symlink 共用並核對 sha256。
  - 池：ImageNet train 10 類 × 100 張；各方法取前 k 張：BC 10、SQNR 10、HTQ 32、HAWQ 64。
  - 評估：`validation-00001-of-00014` 的 3572 張。
- **A 組資源點**（由 BC 的 calibration-only 自動 frontier 決定，在其他方法搜尋前寫入 `caps.json`）：
  - R_min（全 int4）= 0.6223；knee R_k = 0.6410。
  - 寬鬆 R_k + (1 − R_k)/2 = 0.8205；激進 R_min + (R_k − R_min)/2 = 0.6317。
- **HAWQ**：
  - 預先設定為 pyhessian dataloader 模式：同樣 64 張圖，切成 4 個 batch of 16，每個 probe 的 Hv 依 batch 大小平均，等於平均 CE 的 Hessian。
  - 原因是單一 64 張 batch 在冒煙測試中超出 12 GB GPU。
  - 其餘參數同 ResNet18：maxIter 100、tol 1e-3、seed 0。
- **統計**：
  - 每點 3 個方法各和 BC 比，共 9 個比較，用 exact McNemar 加 Holm（α = 0.05）。
  - **C 組**為預先登記的第二個家族：BC 在 SQNR／HTQ／HAWQ 各自自動點的精確 R 下用同一個 MILP 重新搜尋，3 個比較另做 Holm。
  - 配對 bootstrap（20000 次）的 95% CI 只作描述；不設等效界線。

FP32：top-1 78.28%、top-5 93.17%、CE 0.9417。all-INT8 參考：44.40%、67.67%、2.7660。

## 對照與完整性

| 檢查 | 結果 |
|---|---|
| checkpoint 折 BN 與 ONNX initializer 比對 | weight 最大差 3.7e-6、bias 3.7e-6 |
| torch 與 onnxruntime 的 FP32 logits（前 10 張池圖） | 最大差 1.0e-5，top-1 10/10 |
| 前處理重現既有 native 校準輸入 | 一致（最大差 0） |
| BC driver 重建既有 B 表 | 82 個節點，最大相對差 0 |
| 評分表跨重複是否可重現 | BC、SQNR 為 0（3 次），HTQ 1.9e-7（GPU，3 次），HAWQ 只跑 1 次（未檢查） |
| 每個上限另解 LP-first 對照 | 12/12 與 MILP 一致；MILP status 全為 0 |
| INT8 參考用 12／6 個程序各評估一次 | logits 逐位元相同 |
| HAWQ trace | 82 層全部收斂，每層 2–89 個 probe（共 2283）；2 層負 trace（見第 6 題） |

ResNet18 run 做過的「用第 2、3 次的表重新搜尋」對照，這次沒有做。

## 表 1：相同資源上限（A，主要）

Top-1、Top-5 單位為 %，CE 單位為 nats。Δ 都是「方法減 BC」，以同一批 3572 張圖片配對計算。McNemar 欄是「方法對 BC 錯／方法錯 BC 對」的張數。

| 點 | 方法 | 配置 | Top-1 | Top-5 | CE | 較 FP32 下降 (pp) | Δtop-1 (pp) [95% CI] | McNemar | Holm p | 判定 | ΔCE [95% CI] | int8/6/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 寬鬆 0.8205 | BC | `c386b4c3` | 45.46 | 68.03 | 2.7227 | 32.81 | — | — | — | — | — | 40/40/2 |
| | SQNR | `040c66cd` | 43.98 | 66.55 | 2.8212 | 34.29 | −1.48 [−2.52, −0.48] | 141 / 194 | 0.027 | BC 較好 | +0.099 [+0.075, +0.123] | 34/45/3 |
| | HTQ | `144b81bc` | 43.11 | 66.66 | 2.8243 | 35.16 | −2.35 [−3.30, −1.40] | 113 / 197 | 1.5e-5 | BC 較好 | +0.102 [+0.081, +0.122] | 62/14/6 |
| | HAWQ | `11531c7e` | 44.99 | 67.95 | 2.7535 | 33.29 | −0.48 [−1.48, +0.50] | 155 / 172 | 1 | 未定 | +0.031 [+0.010, +0.052] | 38/41/3 |
| knee 0.6410 | BC | `7b02cf34` | 39.84 | 63.19 | 3.0540 | 38.44 | — | — | — | — | — | 10/21/51 |
| | SQNR | `f207e00c` | 39.33 | 62.79 | 3.1467 | 38.94 | −0.50 [−1.60, +0.59] | 188 / 206 | 1 | 未定 | +0.093 [+0.065, +0.120] | 12/23/47 |
| | HTQ | `15f7b8c8` | 23.96 | 44.29 | 4.2692 | 54.31 | −15.87 [−17.30, −14.45] | 101 / 668 | 2e-102 | BC 較好 | +1.215 [+1.165, +1.266] | 28/18/36 |
| | HAWQ | `854bcda4` | 39.31 | 62.57 | 3.0953 | 38.97 | −0.53 [−1.51, +0.48] | 156 / 175 | 1 | 未定 | +0.041 [+0.019, +0.063] | 12/17/53 |
| 激進 0.6317 | BC | `2686307a` | 31.55 | 53.89 | 3.6708 | 46.72 | — | — | — | — | — | 8/11/63 |
| | SQNR | `a3783dd0` | 31.30 | 53.81 | 3.7280 | 46.98 | −0.25 [−1.29, +0.78] | 170 / 179 | 1 | 未定 | +0.057 [+0.029, +0.085] | 5/14/63 |
| | HTQ | `d2792c27` | 21.81 | 41.13 | 4.4743 | 56.47 | −9.74 [−11.09, −8.40] | 138 / 486 | 2e-45 | BC 較好 | +0.804 [+0.754, +0.852] | 20/16/46 |
| | HAWQ | `12073890` | 32.64 | 55.18 | 3.6213 | 45.63 | +1.09 [+0.11, +2.04] | 173 / 134 | 0.15 | 未定 | −0.050 [−0.073, −0.027] | 8/12/62 |

- 未校正 McNemar p：HAWQ 激進點 0.030、SQNR 寬鬆點 0.0044。
- CE 的 CI 在 SQNR、HAWQ 的寬鬆與 knee 都不含 0（BC 較好），在 HAWQ 激進點也不含 0（HAWQ 較好）。CE 只作描述。

## 表 2：共同自動外殼（B，描述性）

| 方法 | 配置 | Top-1 | Top-5 | CE | 較 FP32 下降 (pp) | MB | GBOPs | size ratio | BOPs ratio | R | int8/6/4 | frontier 點數 | 離線總時間 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BC | `7b02cf34` | 39.84 | 63.19 | 3.0540 | 38.44 | 3.357 | 15.32 | 0.6410 | 0.6205 | 0.6410 | 10/21/51 | 22 | 63.8 |
| SQNR | `965c9e44` | 39.56 | 63.49 | 3.1267 | 38.72 | 3.356 | 15.54 | 0.6409 | 0.6295 | 0.6409 | 10/27/45 | 23 | 47.5 |
| HTQ | `b5696365` | 23.49 | 45.10 | 4.2835 | 54.79 | 3.331 | 15.70 | 0.6362 | 0.6357 | 0.6362 | 21/22/39 | 23 | 22.5 |
| HAWQ | `854bcda4` | 39.31 | 62.57 | 3.0953 | 38.97 | 3.357 | 15.70 | 0.6410 | 0.6359 | 0.6410 | 12/17/53 | 21 | 3529.4 |

- 離線總時間 = 評分程序 wall 的中位數（HAWQ 用唯一一次的表格時間）+ 自動外殼搜尋 + 最終編譯，不含最終評估。
- BC 的自動點就是 A 組 knee 的配置；HAWQ 的自動點也和它自己的 A 組 knee 配置相同。

## 表 C：自動點 vs BC 在同一 R 下（預先登記，第二家族）

| 方法 | R 上限 | 方法 top-1 | BC 同 R top-1 | Δtop-1 (pp) [95% CI] | McNemar | Holm p（3 個） | 判定 | ΔCE [95% CI] | BC 配置 |
|---|---|---|---|---|---|---|---|---|---|
| SQNR | 0.6409 | 39.56 | 40.26 | −0.70 [−1.74, +0.34] | 173 / 198 | 0.21 | 未定 | +0.111 [+0.084, +0.137] | `6b04536d`，11/23/48 |
| HTQ | 0.6362 | 23.49 | 36.87 | −13.38 [−14.81, −11.98] | 128 / 606 | 1e-74 | BC 較好 | +0.983 [+0.934, +1.032] | `584e9b04`，8/17/57 |
| HAWQ | 0.6410 | 39.31 | 40.57 | −1.26 [−2.21, −0.28] | 132 / 177 | 0.024 | BC 較好 | +0.044 [+0.021, +0.066] | `37ccf44e`，10/20/52 |

- BC 在三個幾乎相同的 R 下（knee 0.64103、SQNR 的 0.64095、HAWQ 的 0.64102）得到 39.84／40.26／40.57%。這個 0.7 pp 的分散就是本區間的配置雜訊量級。
- HAWQ 的「BC 較好」用的是其中最高的那一組；和 BC 自己的 knee 比只差 −0.53 pp（未定）。因此不把這一項當成 HAWQ 分配較差的強證據。
- HTQ 差距（−13 pp）遠大於這個雜訊。

## 表 3：方法差異（與 ResNet18 的差異）

| 方法 | int6/int4 成本比（中位數 [範圍]） | 與 BC 的 Spearman ρ（int4 / int6，80 個節點） | 資料 | 裝置 |
|---|---|---|---|---|
| BC | 0.048 [0.032, 0.081] | — | 10 張 | CPU（native） |
| SQNR | 0.044 [−0.084, 0.227] | 0.98 / 0.76 | 10 張 | CPU（ORT，247 次） |
| HTQ | 0.192 [0.172, 0.314] | 0.73 / 0.72 | 32 張 | GPU |
| HAWQ | 0.048 [0.044, 0.070] | 0.86 / 0.87 | 64 張（4 × 16） | GPU |

- SQNR 的 int6 成本有些節點低於 int8（比值為負）。這是 10 張圖的單節點 logit 雜訊估計雜訊；int6 的排序和 BC 相關只有 0.76。
- BC 與 HAWQ 用同一個量化器的 ‖ΔW‖²。ResNet18 上每層的 int6/int4 比值都約 0.0516；這裡 HAWQ 的每層比值在 0.044–0.070 之間，BC 因為每通道加權而範圍更寬（0.032–0.081）。兩者的差異來自「每層 trace／參數數」對「每通道 mean g²」。
- HTQ 的分配和其他方法明顯不同：knee 處它把 13/32 個 SE conv 留在 int8（BC 為 1），把 9/16 個 depthwise 留在 int8（BC 為 3）。這和它的大幅劣勢有關聯，但原因沒有單獨檢驗。可能的因素包括 per-tensor 評分量化器、未平方擾動、隨機初始化網路的 SNIP。
- Spearman ρ 是撰寫報告時事後計算的，只作描述。

## 資源明細

- 18 個評估過的配置與 FP32，逐配置的 bytes、BOPs、int8/6/4 與完整格式字串在 `resources.csv`。82 個節點的格式字串太長，不在此列出。
- 不在搜尋範圍內、所有配置都相同的部分：FP32 bias（折 BN 後）125,392 bytes；per-channel weight scale（FP32）125,392 bytes；activation scale／zero point 785 bytes。對稱 weight 的 zero point 為 0，不另外儲存。

## 計時

| 方法 | 裝置 | 張數 | 表格時間中位數 [範圍] (s) | 程序 wall (s) | 相對 BC（表格） | 第 1 次的階段 |
|---|---|---|---|---|---|---|
| BC | CPU，native | 10 | 44.76 [44.08, 45.16]（3 次） | 46.1 [45.5, 46.6]（第 2、3 次） | 1 | weights 0.41；grads .so build 39.64；grads runs 4.92（1.54 s／張） |
| SQNR | CPU，ORT | 10 | 28.05 [26.97, 28.26]（3 次） | 29.7 [29.3, 30.1] | 0.63 | 247 個單節點 QDQ 模型 × 10 張，26.9 |
| HTQ | GPU | 32 | 1.10 [1.10, 1.10]（3 次） | 3.6 | 0.02 | 準備 0.58、SNIP 0.35、擾動 0.16 |
| HAWQ | GPU | 64 | 3511.5（1 次） | — | 78.5 | 準備 1.26、Hessian trace 3510.2（2283 probes） |

- **重複次數**依修訂 2：單次 ≥ 10 分鐘的評分只跑 1 次（< 30 分鐘才跑 2 次）。所以 HAWQ 只有 1 次，沒有範圍，也沒有跨重複的可重現性檢查。
- **BC 冷熱啟動**：每次都重建 grads .so（冷啟動）。熱啟動推估約 5.1 s（總時間減 grads build），這是推估值，沒有另外量測。
- **搜尋**：MILP 每個上限 0.007–0.34 s；自動外殼 1.1–2.4 s。
- **編譯與評估**：最終編譯每個配置 9–17 s；最終評估每個配置 62–98 s（3572 張、12 個程序），不計入離線成本。

## 圖

- `top1_vs_size.png`、`top1_vs_bops.png`：實心點是 A 組，空心點是 B 組，藍色黑框圓點是 C 組的 BC（標註「BC @ R of …」），x 是 all-INT8 參考，虛線是 FP32。
- `quality_vs_offline_time.png`：B 組自動點的 top-1 對離線時間（對數軸）。

## 限制

- 只有 EfficientNet-B0、只有 A8 固定的 weight mixed precision。最主要的限制是 all-INT8 參考已經掉了 34 pp，結論不能外推到 activation 可用的設定。
- 評估集只有一個 shard（3572 張）。在這個區間裡，單一節點的改變就造成約 0.7 pp 的變動，±1 pp 的差異難以和配置雜訊分開。
- HAWQ 只跑了 1 次，GPU 非確定性對它的配置是否有影響沒有檢查；它有一個搜尋節點因負 trace 被設為 0 成本。
- HTQ 的評分量化器（per-tensor）和部署量化器（per-channel）不同，這是原方法本身的設計。
- InfoQ 未執行。三模型合併時，EfficientNet-B0 沒有 InfoQ 的數字。

## 修訂與冒煙測試揭露

`amendments.json` 共 4 筆，每筆都記錄了時間、原因和程式雜湊：
1. **2026-10-29 11:21:19**：冒煙測試後的修正，都在任何正式評分之前。內容：
   - HTQ venv 沒有 onnx 時，固定節點改從 plan.json 讀取。
   - 依模型預設 HAWQ 的批次方式：EfficientNet-B0 用 batch of 16 的 dataloader 模式。
   - 圖的標籤錯開。
2. **2026-10-29 12:26:55**：計時重複規則，在任何正式評分之前（見「計時」一節）。
3. **2026-11-12 22:03**（評估與分析之後）：只改圖的呈現。「BC @ R of …」標籤排成一欄並加引線；時間圖的標籤上下交錯。
4. **2026-11-12 22:03**（評估與分析之後）：只改圖的呈現，加大標籤間距。

第 3、4 筆修訂後重跑 analyze，`summary.json`、`summary_table1.csv`、`summary_table2.csv`、`summary_table_matched_R.csv`、`resources.csv` 的 sha256 都和修訂前相同。

冒煙測試在 scratch 目錄進行：校準池用 symlink，評估集是池裡最後 12 張的假評估集，從未碰過 held-out shard，每個階段都跑過一次。冒煙用的校準資料與正式相同，所以正式評估前已經知道成本表、上限和配置（全都是 calibration-only），但 held-out shard 在配置凍結前從未讀取。

## 重現

在 `experiments/mixed_precision` 下執行（`plan`、`assets`、`controls`、`score_bc_control` 已完成，依序重跑即可）：

```bash
M=efficientnet_b0; CPU=../../venv/bin/python; GPU=/home/hoju/test/HTQ/.venv/bin/python
$CPU run_cnn_baselines.py --stage score_bc   --model $M
$CPU run_cnn_baselines.py --stage score_sqnr --model $M
$GPU run_cnn_baselines.py --stage score_htq  --model $M
$GPU run_cnn_baselines.py --stage score_hawq --model $M
$CPU run_cnn_baselines.py --stage caps       --model $M
$CPU run_cnn_baselines.py --stage search     --model $M
$CPU run_cnn_baselines.py --stage evaluate   --model $M
$CPU run_cnn_baselines.py --stage timing     --model $M --rep 2
$CPU run_cnn_baselines.py --stage timing     --model $M --rep 3
$CPU run_cnn_baselines.py --stage analyze    --model $M
```

本 run 目錄 3.5 GB。
