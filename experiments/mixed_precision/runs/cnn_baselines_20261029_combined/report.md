# CNN weight mixed-precision baseline 比較：三模型總結（A8 固定）

日期：2026-11-12。資料來源：三個 run 的 `summary.json`。由 `cnn_baselines_combined.py` 彙整成 `combined.json`／`combined_tables.md`，沒有新的量測。

| 模型 | run | 狀態 |
|---|---|---|
| ResNet18 | `runs/cnn_baselines_20261029` | 完成，五種評分（含 InfoQ）＋同 R 補測（事後登記） |
| EfficientNet-B0 | `runs/cnn_baselines_20261029_efficientnet_b0` | 完成，四種評分（InfoQ 未執行）＋預先登記的同 R 比較（C） |
| MobileNetV2 | `runs/cnn_baselines_20261029_mobilenetv2` | **未完成、已凍結**：只有評分表，沒有評估，不納入下列結果 |

**共同範圍**：
- activation 固定 INT8 per-tensor，weight 候選 INT4／6／8，per-channel 對稱量化。
- 共同 MILP（HiGHS）搜尋；A 組上限由 BC 的 calibration-only 自動 frontier 決定。
- 同一批 3572 張全新 held-out 圖片，onnx-mlir native 評估。
- HTQ、HAWQ、InfoQ 都是受控改寫，不是原論文復現。細節見各 run 的 report.md。

## 結論

1. **相同資源上限下，沒有任何方法顯著勝過 BC。**
   - 共 21 個比較：ResNet18 12 個、EfficientNet-B0 9 個。
   - BC 顯著較好 7 個：ResNet18 的 InfoQ 三點；EfficientNet-B0 的 HTQ 三點與 SQNR 寬鬆點。
   - 同一配置 7 個，全在 ResNet18。其餘 7 個未定。
   - 唯一的反向點估計是 EfficientNet-B0 激進點 HAWQ +1.09 pp [+0.11, +2.04]，Holm p = 0.15，未定。
2. **SQNR 是最接近 BC 的對手，而且離線成本較低。**
   - ResNet18 三點同配置。
   - EfficientNet-B0 寬鬆點顯著較差（−1.48 pp），其餘兩點未定。
   - 表格時間是 BC 冷啟動的 0.70／0.63 倍。
3. **HAWQ 在 ResNet18 上與 BC 等價，但成本隨模型急增。**
   - ResNet18 三點同配置，成本 1.7 倍。
   - EfficientNet-B0 配置不同、結果互有高低（knee −0.53、激進 +1.09，都未定），成本 78 倍（2283 個 probe，3512 s）。
   - EfficientNet-B0 上有一個 depthwise 層因負 trace 被設為 0 成本，HAWQ 把它放在 int4。
4. **HTQ 最便宜，但在 EfficientNet-B0 上大幅落後**（knee −15.9 pp）。ResNet18 上落後較小，都未定（−0.20／−0.70 pp）。
5. **InfoQ（只有 ResNet18）**：三點都顯著較差（−1.1 到 −3.9 pp），成本 37 倍。
6. **自動外殼的表面優勢都不是更好的分配。**
   - ResNet18：HAWQ／HTQ 的 top-1 較高，但多用約 7.5% BOPs；BC 在同一 R 下重新搜尋後差距消失（+0.11／−0.14 pp，未定）。
   - EfficientNet-B0：四個方法 R 幾乎相同，沒有方法的 top-1 高於 BC。
7. **BC 在這兩個 CNN 上不是最便宜的評分。**
   - 冷啟動時，BC 的時間大部分花在 grads .so 建置：ResNet18 6.1／16.1 s、EfficientNet-B0 39.6／44.8 s。
   - 熱啟動推估約 9.9 s 與 5.1 s，在 EfficientNet-B0 上會低於 SQNR（28 s）。這是推估值，沒有量測。
8. **建議：BC 準則不改。** 沒有方法在相同上限下勝過它；比它便宜的 SQNR 不比它好；比它貴的 HAWQ 也沒有穩定地更好。

## 解讀時的限制

- **EfficientNet-B0 的比較區間很差**：all-INT8 參考 44.40%，FP32 為 78.28%，activation 已經損傷 34 pp。
  - 在這個區間裡，BC 自己在幾乎相同的 R 下只差一個節點的兩組配置，就差了 0.73 pp。
  - ±1 pp 級的差異（SQNR、HAWQ）要當成接近配置雜訊來讀；只有 HTQ 的差距明顯超出這個範圍。
- 每個模型只有一個評估 shard（3572 張），McNemar 對 1 pp 以下的差異檢定力有限。「未定」不代表等效。
- 只有 A8 固定的 weight mixed precision。結論不外推到 W+A 混合精度、activation 可選 FP16／Posit、LLM 或 MoE。
- ResNet18 的同 R 補測是在看過主要結果之後才登記的（事後）；EfficientNet-B0 的 C 是預先登記的。
- 各方法的校準張數不同（BC 10、SQNR 10、HTQ 32、HAWQ 64、InfoQ 1000），差異算在成本裡。
- HAWQ 在 EfficientNet-B0 只跑 1 次（依計時規則修訂），沒有範圍，也沒有可重現性檢查。

## 表 1：相同資源上限（方法 − BC，top-1 pp [95% CI]，Holm 後判定）

| 模型 | 點 | R 上限 | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|---|
| ResNet18 | 寬鬆 | 0.8124 | 同一配置 | −0.20 [−0.70, +0.31] 未定 | 同一配置 | −1.12 [−1.88, −0.39] BC 較好 |
| ResNet18 | knee | 0.6247 | 同一配置 | −0.70 [−1.54, +0.14] 未定 | 同一配置 | −3.89 [−4.87, −2.94] BC 較好 |
| ResNet18 | 激進 | 0.5787 | 同一配置 | 同一配置 | 同一配置 | −3.39 [−4.37, −2.44] BC 較好 |
| EfficientNet-B0 | 寬鬆 | 0.8205 | −1.48 [−2.52, −0.48] BC 較好 | −2.35 [−3.30, −1.40] BC 較好 | −0.48 [−1.48, +0.50] 未定 | 未執行 |
| EfficientNet-B0 | knee | 0.6410 | −0.50 [−1.60, +0.59] 未定 | −15.87 [−17.30, −14.45] BC 較好 | −0.53 [−1.51, +0.48] 未定 | 未執行 |
| EfficientNet-B0 | 激進 | 0.6317 | −0.25 [−1.29, +0.78] 未定 | −9.74 [−11.09, −8.40] BC 較好 | +1.09 [+0.11, +2.04] 未定 | 未執行 |

BC 本身的 top-1：
- ResNet18：69.82／67.75／65.82%（寬鬆／knee／激進）。
- EfficientNet-B0：45.46／39.84／31.55%。

## 表 2：共同自動外殼（top-1 %，R）

| 模型 | BC | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|
| ResNet18 | 67.75（R 0.625） | 67.53（R 0.623） | 68.11（R 0.673） | 68.53（R 0.672） | 63.05（R 0.594） |
| EfficientNet-B0 | 39.84（R 0.641） | 39.56（R 0.641） | 23.49（R 0.636） | 39.31（R 0.641） | 未執行 |

## 表 C：自動點 vs BC 在同一 R 下重新搜尋（方法 − BC）

| 模型 | 方法 | R | 方法 top-1 | BC top-1 | Δ (pp) [95% CI] | 判定 | 登記方式 |
|---|---|---|---|---|---|---|---|
| ResNet18 | HAWQ | 0.6716 | 68.53 | 68.42 | +0.11 [−0.56, +0.76] | 未定 | 事後登記 |
| ResNet18 | HTQ | 0.6734 | 68.11 | 68.25 | −0.14 [−0.81, +0.56] | 未定 | 事後登記 |
| EfficientNet-B0 | SQNR | 0.6409 | 39.56 | 40.26 | −0.70 [−1.74, +0.34] | 未定 | 預先登記 |
| EfficientNet-B0 | HTQ | 0.6362 | 23.49 | 36.87 | −13.38 [−14.81, −11.98] | BC 較好 | 預先登記 |
| EfficientNet-B0 | HAWQ | 0.6410 | 39.31 | 40.57 | −1.26 [−2.21, −0.28] | BC 較好* | 預先登記 |

\* 對比的 BC 配置比 BC 自己的 knee（R 幾乎相同、只差一個節點）高 0.73 pp。和 knee 比，HAWQ 只差 −0.53 pp（未定），所以這一項不當作強證據。

## 表 3：離線評分成本（表格時間中位數，相對 BC 冷啟動）

| 模型 | BC | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|
| ResNet18 | 16.1 s（1，n=3） | 11.3 s（0.70，n=3） | 0.8 s（0.05，n=3） | 27.3 s（1.69，n=3） | 600.7 s（37.3，n=2） |
| EfficientNet-B0 | 44.8 s（1，n=3） | 28.0 s（0.63，n=3） | 1.1 s（0.02，n=3） | 3511.5 s（78.5，n=1） | 未執行 |
| MobileNetV2（未完成，n=1） | 21.0 s | 9.4 s（0.45） | 1.0 s（0.05） | 1358.7 s（64.7） | 中斷 |

- BC、SQNR 在 12 核 CPU；HTQ、HAWQ、InfoQ 在 RTX 3060。這是實測 pipeline 成本，不是演算法加速比。
- MobileNetV2 那列只是單次的評分時間，沒有對應的準確度結果。
