# ResNet18 weight mixed-precision 的 baseline 比較（A8 固定）

日期：2026-10-29。程式：`run_cnn_baselines.py`、`cnn_baselines_infoq.py`。計畫：`plan.json`／`plan.md`（00:25:05，早於任何評分），修訂見 `amendments.json`（4 筆）。

**範圍限定**：只適用於 ResNet18／ImageNet、activation 固定 INT8、weight 候選 INT4／6／8 的 weight mixed precision；比較的是「五種評分在同一個搜尋器下選出的配置」，不是各論文的完整原生 pipeline。HTQ、HAWQ、InfoQ 都是受控改寫，不是原論文復現；InfoQ 一律稱為「InfoQ 評分／分配的無訓練受控比較」。

## 結論

- **相同資源上限下，BC 在三個點都是觀察到最好的 top-1。**
  - SQNR、HAWQ 在三個點都選出與 BC 完全相同的配置；HTQ 在激進點也與 BC 相同。
  - HTQ 在寬鬆點 −0.20 pp、knee −0.70 pp，Holm 後都未定，但 CE 的 bootstrap CI 都不含 0（描述性）。
  - InfoQ 在三個點都顯著較差：−1.12／−3.89／−3.39 pp，Holm p = 0.037／7e-14／8e-11。
- **離線成本：BC 在 ResNet18 上不是最便宜的。**
  - 表格時間中位數：HTQ 0.78 s、SQNR 11.3 s、BC 16.1 s（冷啟動）、HAWQ 27.3 s、InfoQ 601 s。
  - 與 BC 相比：SQNR 用 0.70 倍時間、在 CPU 上得到同樣三組配置；HAWQ 要 1.7 倍、用 GPU；InfoQ 要 37 倍，而且更差。
- **共同自動外殼下，HAWQ（68.53%）與 HTQ（68.11%）的 top-1 高於 BC（67.75%）**，但它們的 knee 落在 R 0.672／0.673，比 BC 的 0.625 多用約 7.5% BOPs（HAWQ 另多 10.7% bytes）。
  - 補測讓 BC 在同一個 R 重新搜尋：BC 在 R 0.6716 得到 68.42%，HAWQ 只多 +0.11 pp [−0.56, +0.76]，未定。BC 在 R 0.6734 得到 68.25%，HTQ 少 −0.14 pp [−0.81, +0.56]，未定。
  - 所以自動外殼裡的表面優勢來自 knee 選在較高資源點，不是分配得比較好。這是缺乏優勢的證據，不是等效宣稱。
- **建議：BC 準則不改。** MobileNetV2 下一個最值得測的是 HAWQ。

## 六個問題

1. **相同資源上限下，BC 的 accuracy 是否有競爭力？**
   有。三個點上 BC 都等於或高於其他四種方法的點估計，沒有任何方法顯著勝過 BC。12 個比較中有 7 個是同一配置（差值恰為 0，這是「同一配置」，不是統計上的等效）；HTQ 的 2 個未定；InfoQ 的 3 個是 BC 顯著較好。
2. **更準的方法需要多少額外成本？**
   在相同上限下沒有比 BC 更準的方法，所以沒有「為準確度付出的額外成本」可算。和 BC 選出同一配置的方法成本為：
   - SQNR：0.70 倍（CPU）
   - HAWQ：1.7 倍（GPU，要 64 張圖）

   InfoQ 多花 37 倍時間，結果反而更差。
3. **自動選出的 BOPs／size 差多少？**
   各方法自動選點相對 all-INT8 的比值如下（size／BOPs）：

   | 方法 | size | BOPs |
   |---|---|---|
   | BC | 0.570 | 0.625 |
   | SQNR | 0.567 | 0.623 |
   | HTQ | 0.573 | 0.673 |
   | HAWQ | 0.631 | 0.672 |
   | InfoQ | 0.538 | 0.594 |

   - HAWQ 比 BC 多 +0.71 MB 與 +5.45 GBOPs，HTQ 多 +5.65 GBOPs。
   - InfoQ 比 BC 少 3.6 GBOPs，但 top-1 低 4.7 pp。
   - 差異來自 knee 規則落在各自 frontier 的不同位置。BC 與 HAWQ 在三個上限選同一配置、自動 knee 卻不同，說明這是 frontier 形狀（成本尺度）的差異，不是排序差異。
4. **表面優勢是否只是用了更多資源？**
   是。HAWQ／HTQ 的自動外殼優勢都伴隨 +7.5～7.8% BOPs。補測讓 BC 在它們各自的 R 下重新搜尋：
   - BC 在 R 0.6716 得 68.42%，HAWQ 為 68.53%，差 +0.11 pp [−0.56, +0.76]，McNemar 73/69，Holm p = 1。
   - BC 在 R 0.6734 得 68.25%，HTQ 為 68.11%，差 −0.14 pp [−0.81, +0.56]，McNemar 75/80，Holm p = 1。

   兩者都是未定。原本相對 BC knee 的 +0.78／+0.36 pp，在相同 R 下大多消失了。再加上 A 組的三個上限，沒有任何證據顯示 HAWQ／HTQ 在相同資源下分配得比 BC 好。補測是在主要結果揭曉後才預先登記的，屬於事後比較（見「補測」一節）。
5. **要修改 BC 還是保留？**
   保留、不改。相同上限下沒有方法勝過 BC，HAWQ（同為 ‖ΔW‖² × 二階權重）也選出同一配置，沒有證據顯示準則需要修正。
   - 自動外殼的差異屬於 knee 規則，不是準則本身；依 2026-10-11 的決定，knee 規則也不在本輪更動範圍。
   - BC 在 ResNet18 上的離線成本優勢並不存在（SQNR、HTQ 更快）。它的成本優勢要在節點或候選格式較多的模型上才會出現，見先前 criteria cost 的比較。
6. **哪個 baseline 最值得接著測 MobileNetV2？**
   HAWQ。理由：
   - 它是唯一和 BC 函數形式相同（‖Q_b(W) − W‖² × 敏感度）、只差「每層 Hessian trace／參數數」對「每通道 mean g²」的方法。
   - ResNet18 的通道統計較均勻，兩者分不開（ρ 0.97，三個上限同配置）。MobileNetV2 的 depthwise 層每通道 range 很不均（先前的 outlier range 結果），正好能測出每通道解析度是否重要。
   - SQNR 在 MobileNetV2 已和 BC 比過（compiled B+C ≈ SQNR）；HTQ 雖便宜，但用的是 per-tensor、未平方的擾動，在本輪已見劣勢；InfoQ 成本最高、結果最差。
   - 建議在 MobileNetV2 一開始就把「BC 在其他方法自動點 R 下的配置」排進預先登記，不要事後補測。

## 設定

- **模型**：torchvision resnet18 IMAGENET1K_V1；ONNX 已把 BN 折入 conv。PyTorch 評分器載入同一個 checkpoint；折疊與 logits 對照見下文。
- **搜尋空間**：19 個內層 conv。conv1（`node_Conv_291`）與 fc（`node_linear`）對所有方法固定 INT8 weight。
- **Weight 與 activation**：
  - weight 候選 int4／int6／int8，per-channel 對稱 max-abs，round half away from zero（`MIXEDP_WEIGHT_PER_CHANNEL=1`）。
  - 21 個 weight node 的輸入與輸出 activation 一律 INT8 per-tensor 非對稱，ranges 固定沿用 `runs/native_auto_2d_20261011/resnet18/ranges_by_point.json`。
  - bias 為 FP32；沒有 FP 或 Posit 候選。
- **資源**：
  - bytes = Σ N·b/8；BOPs = Σ MAC·b_w·b_a（MAC 不乘 2）。
  - 比值以 all-INT8 為基準（11,678,912 bytes；116.10 GBOPs）；R = max(size ratio, BOPs ratio)。
  - 超過上限的配置一律無效（實際沒有出現）。
- **校準**：
  - 同一個 1000 張的池：ImageNet train，benjamin-paine/imagenet-1k-256x256，依 InfoQ 自己的選類規則取 10 類 × 100 張，輪流排序。
  - 各方法取前 k 張：BC 10、SQNR 10、HTQ 32、HAWQ 64、InfoQ 1000（MQBench 校準另用前 64 張）。
  - 資料量差異算在方法成本裡。
- **評估**：
  - 全新的 held-out shard：benjamin-paine/imagenet-1k `validation-00001-of-00014`，共 3572 張，先前沒有任何 run 用過。
  - 用 onnx-mlir native 編譯（O3/O3）評估，切成 12 個程序。
  - 配置凍結後才評估（`configs_freeze.json`）。
- **搜尋**：
  - 共同 MILP：`mp_milp.milp_select`，HiGHS，是 dense 模型的正式規則。
  - 每個上限都另外解 LP-first 當對照，15/15 一致；MILP 的 status 全部是 0。
  - 自動外殼：`mp_auto.auto_select`（17 個預算、2 次細化、chord knee）。
- **A 組資源點**：由 BC 的 calibration-only 自動 frontier 決定，在其他方法搜尋前就寫入 `caps.json`。
  - R_min（全 int4）= 0.5327；knee R_k = 0.6247。
  - 寬鬆 R_k + (1 − R_k)/2 = 0.8124；激進 R_min + (R_k − R_min)/2 = 0.5787。
  - size 與 BOPs 都要 ≤ R × all-INT8。三個點實際綁住的都是 BOPs。
- **統計**：
  - 主要指標 top-1；每點 4 個方法各和 BC 比，共 12 個比較，用 exact McNemar 加 Holm（α = 0.05）。
  - 配對 bootstrap（20000 次，seed 26102901）的 95% CI 只作描述。
  - 不設等效界線：未顯著就是「未定」。

FP32：top-1 70.63%、top-5 88.52%、CE 1.2519。all-INT8 參考：70.52%、88.58%、1.2646（下降 0.11 pp）。

## 對照與完整性

| 檢查 | 結果 |
|---|---|
| checkpoint 折 BN 與 ONNX initializer 比對 | weight 最大差 1.19e-7、bias 1.12e-7 |
| torch 與 onnxruntime 的 FP32 logits（前 10 張池圖） | 最大差 9.06e-6，top-1 10/10 |
| 前處理重現既有 native 校準輸入 | 一致 |
| BC driver 重建既有 B 表（`controls_bc_reconstruction.json`） | 最大相對差 0 |
| 評分表跨重複是否可重現 | BC、SQNR、HTQ 為 0（3 次），InfoQ 為 0（2 次），HAWQ 4.7e-8（GPU 非確定性） |
| 用第 2、3 次的表重新搜尋 3 個上限與自動外殼（36 次） | 全部與第 1 次相同（`controls_reps_resources.json`） |
| 依 plan.json 的 elems／MACs 獨立重算資源 | 與 runner 一致，全部在上限內；固定節點都是 int8，activation 都是 int8 |
| INT8 參考用 12／6 個程序各評估一次 | logits 逐位元相同（`controls_eval.json`） |
| HAWQ trace | 21 層全部收斂，共 353 個 probe（每層 4–56），沒有負 trace，不需動用「負值 → 0」規則 |
| InfoQ | 沒有讀作者快取（RUNS/），沒有 QAT |

## 表 1：相同資源上限（A，主要）

Top-1、Top-5 單位為 %，CE 單位為 nats。Δ 都是「方法減 BC」，以同一批 3572 張圖片配對計算。McNemar 欄位是「方法對 BC 錯／方法錯 BC 對」的張數。

| 點 | 方法 | 配置 | Top-1 | Top-5 | CE | 較 FP32 下降 (pp) | Δtop-1 (pp) [95% CI] | McNemar | Holm p | 判定 | ΔCE [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 寬鬆 | BC | `6b2f3d39` | 69.82 | 88.30 | 1.2792 | 0.81 | — | — | — | — | — |
| 寬鬆 | SQNR | `6b2f3d39` | 69.82 | 88.30 | 1.2792 | 0.81 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| 寬鬆 | HTQ | `625987a4` | 69.62 | 88.05 | 1.2853 | 1.01 | −0.20 [−0.70, +0.31] | 40 / 47 | 1 | 未定 | +0.0061 [+0.0008, +0.0115] |
| 寬鬆 | HAWQ | `6b2f3d39` | 69.82 | 88.30 | 1.2792 | 0.81 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| 寬鬆 | InfoQ | `2ddc2417` | 68.70 | 88.05 | 1.2990 | 1.93 | −1.12 [−1.88, −0.39] | 71 / 111 | 0.037 | BC 較好 | +0.0198 [+0.0086, +0.0309] |
| knee | BC | `76495e9f` | 67.75 | 87.04 | 1.3574 | 2.88 | — | — | — | — | — |
| knee | SQNR | `76495e9f` | 67.75 | 87.04 | 1.3574 | 2.88 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| knee | HTQ | `0ca4ff23` | 67.05 | 86.67 | 1.3999 | 3.58 | −0.70 [−1.54, +0.14] | 108 / 133 | 1 | 未定 | +0.0425 [+0.0265, +0.0586] |
| knee | HAWQ | `76495e9f` | 67.75 | 87.04 | 1.3574 | 2.88 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| knee | InfoQ | `8accb273` | 63.86 | 84.74 | 1.5253 | 6.77 | −3.89 [−4.87, −2.94] | 92 / 231 | 7.3e-14 | BC 較好 | +0.1679 [+0.1454, +0.1905] |
| 激進 | BC | `7831791c` | 65.82 | 85.95 | 1.4547 | 4.82 | — | — | — | — | — |
| 激進 | SQNR | `7831791c` | 65.82 | 85.95 | 1.4547 | 4.82 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| 激進 | HTQ | `7831791c` | 65.82 | 85.95 | 1.4547 | 4.82 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| 激進 | HAWQ | `7831791c` | 65.82 | 85.95 | 1.4547 | 4.82 | 0 | 0 / 0 | 1 | 同一配置 | 0 |
| 激進 | InfoQ | `20655846` | 62.43 | 84.04 | 1.6002 | 8.20 | −3.39 [−4.37, −2.44] | 97 / 218 | 8.4e-11 | BC 較好 | +0.1455 [+0.1235, +0.1677] |

A 組的資源與利用率如下。「上限 R」是該點的上限；「利用率」是用量除以上限。

| 點 | 方法 | 配置 | MB | MiB | GBOPs | size ratio | BOPs ratio | R | 上限 R | size 利用率 | BOPs 利用率 | int8/6/4 層數 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 寬鬆 | BC = SQNR = HAWQ | `6b2f3d39` | 8.969 | 8.554 | 93.90 | 0.7680 | 0.8088 | 0.8088 | 0.8124 | 94.5% | 99.56% | 8/13/0 |
| 寬鬆 | HTQ | `625987a4` | 8.389 | 8.000 | 93.90 | 0.7183 | 0.8088 | 0.8088 | 0.8124 | 88.4% | 99.56% | 9/11/1 |
| 寬鬆 | InfoQ | `2ddc2417` | 8.942 | 8.528 | 93.90 | 0.7656 | 0.8088 | 0.8088 | 0.8124 | 94.2% | 99.56% | 13/4/4 |
| knee | BC = SQNR = HAWQ | `76495e9f` | 6.657 | 6.349 | 72.53 | 0.5700 | 0.6247 | 0.6247 | 0.6247 | 91.2% | 100.00% | 4/8/9 |
| knee | HTQ | `0ca4ff23` | 6.602 | 6.296 | 72.53 | 0.5653 | 0.6247 | 0.6247 | 0.6247 | 90.5% | 100.00% | 4/8/9 |
| knee | InfoQ | `8accb273` | 7.008 | 6.684 | 72.43 | 0.6001 | 0.6238 | 0.6238 | 0.6247 | 96.1% | 99.86% | 5/5/11 |
| 激進 | BC = SQNR = HTQ = HAWQ | `7831791c` | 6.223 | 5.935 | 67.08 | 0.5328 | 0.5778 | 0.5778 | 0.5787 | 92.1% | 99.85% | 5/3/13 |
| 激進 | InfoQ | `20655846` | 6.251 | 5.961 | 67.08 | 0.5352 | 0.5778 | 0.5778 | 0.5787 | 92.5% | 99.85% | 5/3/13 |

同一個點上各方法用到的 BOPs 幾乎相同（綁住的都是 BOPs），所以 A 組的差異是資源分配的差異，不是資源量的差異。

## 表 2：共同自動外殼（B，描述性）

每個方法用自己的成本表跑同一個自動外殼（掃描 → Pareto → chord knee），然後評估各自選出的配置。這張表不做檢定，要和表 1 一起讀。

| 方法 | 配置 | Top-1 | Top-5 | CE | 較 FP32 下降 (pp) | MB | GBOPs | size ratio | BOPs ratio | R | int8/6/4 | frontier 點數 | 離線總時間 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BC | `76495e9f` | 67.75 | 87.04 | 1.3574 | 2.88 | 6.657 | 72.53 | 0.5700 | 0.6247 | 0.6247 | 4/8/9 | 23 | 20.2 |
| SQNR | `8acc2f56` | 67.53 | 87.01 | 1.3588 | 3.11 | 6.622 | 72.32 | 0.5670 | 0.6229 | 0.6229 | 2/10/9 | 23 | 15.4 |
| HTQ | `7aeda055` | 68.11 | 87.35 | 1.3301 | 2.52 | 6.693 | 78.18 | 0.5731 | 0.6734 | 0.6734 | 6/8/7 | 20 | 5.8 |
| HAWQ | `2a9ca782` | 68.53 | 87.04 | 1.3456 | 2.10 | 7.371 | 77.98 | 0.6311 | 0.6716 | 0.6716 | 3/12/6 | 22 | 31.4 |
| InfoQ | `47800044` | 63.05 | 84.18 | 1.5768 | 7.59 | 6.288 | 68.93 | 0.5384 | 0.5937 | 0.5937 | 5/4/12 | 20 | 604.4 |

- 離線總時間 = 評分程序 wall 的中位數 + 自動外殼搜尋 + 最終編譯，不含最終評估。
- BC 的自動點就是 A 組 knee 那組配置（knee 上限本來就是從它定出來的）。

## 表 3：方法差異

| 方法 | 成本（加總後最小化） | 使用的量化器 | 資料 | 裝置 | int6/int4 成本比（中位數） | 與 BC 的 Spearman ρ（int4 / int6） | 改寫與原始差異 |
|---|---|---|---|---|---|---|---|
| BC | Σ_c mean(g²)_c · ‖ΔW_c‖²（正式 native B：grads pass + weights pass） | 部署用的 per-channel | 10 張，真實標籤 | CPU（onnx-mlir native） | 0.0516 | — | 無：就是正式定義，重建對照 max rel 0 |
| SQNR | N = Σ‖z_q − z_FP32‖²：單節點 QDQ（該節點 weight 為 b、x/y 為 A8，其他節點 FP32），10 張圖的 logit 雜訊加總 | 部署用的 per-channel | 10 張 | CPU（onnxruntime，64 次） | 0.0665 | 0.96 / 0.84 | 這是模型輸出雜訊，不是 weight SQNR。可加性假設是「單節點雜訊相加 = 總 logit 雜訊」，等同 −SQNR 差一個常數。該節點 A8 的雜訊對同一節點的三個 weight 格式是同一個常數，不影響選擇 |
| HTQ | SNIP 分數 × ‖W − Q_b(W)‖_F（未平方） | HTQ 自己的 per-tensor 非對稱，作用在未折疊的 checkpoint weight | 32 張 | GPU | 0.245（≈ 0.0516 開根號） | 0.78 / 0.76 | 照 HTQ 原邏輯：隨機重新初始化（torch seed 0，xavier_normal_）、train mode、對原始 logits 算 `F.nll_loss`。差異：batch 改用共同池前 32 張（原本是 32 張 random-crop train 圖）、checkpoint 共用、候選限縮成共同的三種。評分量化器（per-tensor）和部署量化器（per-channel）不一致，這是原方法本身的設計 |
| HAWQ | mean Tr(H)/n × ‖Q_b(W) − W‖² | 部署用的 per-channel（作用在未折疊 weight，與 BN scale 可交換，等於部署誤差映回去） | 64 張，CE，eval mode | GPU | 0.0516 | 0.97 / 0.97 | CLADO_SampleRun 的 `prep_hawq.estimate_trace`（pyhessian Hutchinson，Rademacher probes，maxIter 100、tol 1e-3、seed 0）加上 `optimize.py` 的成本式。不是 HAWQ-V3 的 ILP 或它的量化器。沒有負 trace |
| InfoQ | −score；score = −(Σ\|ΔMI_XL\| + Σ\|ΔMI_LY\|)/ΣMI_8 × 1/b，int8 成本為 0 | MQBench Academic 背景量化（per-tensor 對稱 MSE weight、8-bit EMA-MSE activation、校準時重估 BN） | 1000 張 SMI 加 64 張 MQBench 校準 | GPU | 0.519（0.098–1.20） | 0.61 / −0.04 | 見下方說明 |

InfoQ 的改寫細節：
- 模組照原樣使用：InfoQ 自己的 `uniform_quantized_models`／`mpq_quantize`、`calculate_smi`（k-SMI，KSG，m = 100）、`get_smi_scores`。
- 只替換資料來源，另用 stub 取代 `SMI.vae`：原模組在 import 時會載入整個資料集，而 X 用的是 DINOv2，不需要 VAE。
- Observer 從 gluon 名稱對應到 torchvision 的 `layerX_Y_relu_1_post_act_fake_quantizer`。XL 是 layer3.0、layer3.1、layer4.0 的 block 輸出；LY 是 flatten 與 fc。
- 標籤用 InfoQ 自己的 `get_targets` 重新映射成 0..9。
- 忠於原碼的一個怪處：DINOv2 路徑對 [0,1] 張量做了兩次 rescale（像素 std 約 0.13），所以 MI(X;L) ≈ 2e-4，分數主要由 L;Y 項決定。
- `layer3.0.downsample.0` 的成本非單調：int4 0.00158 < int6 0.00189。
- 沒有 QAT、沒有作者快取，也不計算原腳本裡的 accuracy 欄（它不進入分數）。

各方法的成本結構：
- BC 與 HAWQ 用同一個量化器的 ‖ΔW‖²，所以每層的 int6/int4 比值完全相同（約 0.0516，也就是步長比的平方）。兩者只差在每層的敏感度權重，排序相關 0.97，因此三個上限都選出同一配置。
- HTQ 的擾動沒有平方，int6 相對 int4 的「省下量」被低估，選擇和其他方法不同。
- InfoQ 的 int6 排序和其他方法幾乎無關（ρ ≈ 0）。

Spearman ρ 是在本報告撰寫時事後計算的，只作描述。

## 資源明細（所有評估過的配置）

配置字串的欄位順序：conv1、L1.0c1、L1.0c2、L1.1c1、L1.1c2、L2.0c1、L2.0c2、L2.0d、L2.1c1、L2.1c2、L3.0c1、L3.0c2、L3.0d、L3.1c1、L3.1c2、L4.0c1、L4.0c2、L4.0d、L4.1c1、L4.1c2、fc（數字是 weight bit；activation 全部 8）。

| 配置 | weight bits | 使用者 |
|---|---|---|
| `6b2f3d39` | 886868686666866668668 | BC、SQNR、HAWQ：A 寬鬆 |
| `76495e9f` | 864646686464644648448 | BC、SQNR、HAWQ：A knee；BC：B |
| `7831791c` | 864646484444844448448 | BC、SQNR、HTQ、HAWQ：A 激進 |
| `625987a4` | 888868686666866668648 | HTQ：A 寬鬆 |
| `0ca4ff23` | 866666484464644648448 | HTQ：A knee |
| `2ddc2417` | 886888686884844868848 | InfoQ：A 寬鬆 |
| `8accb273` | 844646686484444448648 | InfoQ：A knee |
| `20655846` | 844646684444844448448 | InfoQ：A 激進 |
| `8acc2f56` | 864646666464644646448 | SQNR：B |
| `7aeda055` | 886666686464844648448 | HTQ：B |
| `2a9ca782` | 866646686466644666448 | HAWQ：B |
| `47800044` | 844646686444844448448 | InfoQ：B |

HTQ 在 knee 和 BC 的 int8/6/4 層數相同，但分配不同：它把 L1.0c2、L1.1c2 提到 int6，把 L2.0c2、L2.1c1 降到 int4。

不在搜尋範圍內的資源另外列出，所有配置都相同：
- FP32 bias（折 BN 後）：23,200 bytes。
- per-channel weight scale（FP32）：23,200 bytes。
- activation scale／zero point：195 bytes。
- 對稱 weight 的 zero point 為 0，不另外儲存。

Artifact 大小所有配置都相同：.so 134,632 bytes、constants 46,729,888 bytes。這是 FP32 加 QDQ 的 artifact，不是 packed 低 bit 的大小，不可拿來當模型大小。逐配置的數字在 `resources.csv`，含 `total_quantized_bytes` = weight + bias + scale + activation qparams。

## 計時

計時方式：
- BC、SQNR、HTQ、HAWQ 各跑 3 次，交錯執行、每次輪換方法順序，每次都是新的子程序；InfoQ 跑 2 次（單次 < 30 分鐘，依計畫加跑一次）。
- 表格時間取自各評分內部計時；程序 wall 包含 Python 啟動與 CUDA 初始化。
- 範圍校準是固定沿用的，所有方法都是 0。
- 裝置不同（BC、SQNR 在 12 核 CPU；HTQ、HAWQ、InfoQ 在 RTX 3060），所以這是實測 pipeline 成本，不是演算法加速比。

| 方法 | 裝置 | 張數 | 表格時間中位數 [範圍] (s) | 程序 wall 中位數 [範圍] (s) | 相對 BC（表格） | 第 1 次的階段 |
|---|---|---|---|---|---|---|
| BC | CPU，native | 10 | 16.11 [16.00, 16.29] | 17.59 [17.51, 17.80] | 1 | import 0.20、bytecode 0.18；weights pass 0.83 與（grads .so build 6.12 + grads runs 9.50）並行 |
| SQNR | CPU，ORT | 10 | 11.31 [10.85, 11.41] | 12.76 [12.64, 13.87] | 0.70 | 64 個單節點 QDQ 模型 × 10 張，10.68 |
| HTQ | GPU | 32 | 0.78 [0.75, 1.07] | 3.18 [3.15, 3.73] | 0.05 | 準備 0.33、SNIP 0.57、擾動 0.17 |
| HAWQ | GPU | 64 | 27.30 [27.30, 27.39] | 28.78 [28.76, 28.86] | 1.69 | 準備 1.05、Hessian trace 26.2（353 probes） |
| InfoQ | GPU | 1000 + 64 | 600.7 [598.4, 603.0] | 601.8 [599.4, 604.2] | 37.3 | 準備 4.9、DINOv2 嵌入 8.9、背景量化模型 7.7、SMI 576.8（39 次量測：全 8-bit 參考 1 次，加 19 層 × 2 bits，每次 8.7–17.1 s） |

- **BC 冷熱啟動**：每次重複都在全新目錄重建 grads .so，所以上表是冷啟動。熱啟動（重用 .so）沒有另外量測；用各階段時間推估約 9.9 s（總時間減 grads build），這是推估值。
- **搜尋**：MILP 每個上限 4–11 ms；自動外殼 0.13–0.15 s（23 次求解）。
- **編譯與評估**：最終編譯每個配置 1–3 s。最終評估每個配置約 500 s（3572 張、12 個程序），不計入離線成本。

## 圖

- `top1_vs_size.png`、`top1_vs_bops.png`：實心點是 A 組（相同上限），空心點是 B 組（自動外殼）；x 表示 all-INT8 參考，虛線是 FP32。同一配置的點以同心、不同大小的標記重疊，旁邊有「same config」文字標註。
- `quality_vs_offline_time.png`：B 組自動選點的 top-1 對離線時間（對數軸），每點標出 R。它混合了資源差異，要搭配表 1 一起讀。

類別色用 dataviz 預設的 1–5 號色，並以標記形狀作第二編碼。調色盤驗證程式沒有執行（環境裡沒有 node），只靠目視檢查。

## 補測：BC 在 HAWQ／HTQ 自動點的同一 R 下

這是事後比較：主要結果都已看過，但在這 2 次評估之前已預先登記，計畫在 `followup_matched/plan.json`（2026-10-29 10:11:40，含程式與主 run 的雜湊）。

- 方法：讓 BC 的成本表在 HAWQ（R = 0.6716）與 HTQ（R = 0.6734）自動選點的精確 R 下，用同一個 MILP 重新搜尋，兩個上限都是 R × all-INT8 的 size 與 BOPs。評估方式與主 run 相同（native O3/O3、12 個程序、同一批 3572 張）。
- 統計：每個比較都是方法減 BC；exact McNemar，Holm 只在這 2 個比較內校正；bootstrap 設定相同。
- 主 run 的 `configs.json` 沒有修改。程式：`run_cnn_baselines_matched.py`；輸出：`followup_matched/`。

| 比較 | R 上限 | 方法 top-1 | BC 同 R top-1 | Δtop-1 (pp) [95% CI] | McNemar | Holm p | 判定 | ΔCE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| HAWQ 自動點 vs BC | 0.6716 | 68.53 | 68.42 | +0.11 [−0.56, +0.76] | 73 / 69 | 1 | 未定 | +0.0085 [−0.0011, +0.0180] |
| HTQ 自動點 vs BC | 0.6734 | 68.11 | 68.25 | −0.14 [−0.81, +0.56] | 75 / 80 | 1 | 未定 | −0.0066 [−0.0161, +0.0029] |

| 配置 | weight bits（欄位順序同資源明細） | MB | GBOPs | size ratio | BOPs ratio | int8/6/4 |
|---|---|---|---|---|---|---|
| BC 在 R 0.6716：`df4229c5` | 866666686466644646448 | 6.790 | 77.98 | 0.5814 | 0.6716 | 3/12/6 |
| HAWQ 自動點：`2a9ca782` | 866646686466644666448 | 7.371 | 77.98 | 0.6311 | 0.6716 | 3/12/6 |
| BC 在 R 0.6734：`9758013d` | 866666686466844648448 | 6.831 | 78.18 | 0.5849 | 0.6734 | 5/10/6 |
| HTQ 自動點：`7aeda055` | 886666686464844648448 | 6.693 | 78.18 | 0.5731 | 0.6734 | 6/8/7 |

- BC 在同一 R 下用的 BOPs 和對手相同；bytes 比 HAWQ 少 0.58 MB，比 HTQ 多 0.14 MB。
- BC 自己的 frontier 從 knee（R 0.6247，67.75%）到 R 0.6716（68.42%），多 4.7% BOPs，換到 +0.67 pp。

## 事後檢查（評估之後，只作描述，不改變結論的判定）

- **相同 R 純搜尋**（`posthoc_matched_R_search.py`／`.json`，沒有新評估）：用各方法自動點的 R 當上限，讓每個成本表各解一次。
  - 在 SQNR 的 R（0.6229）下，BC 與 HAWQ 選同一配置；在 HAWQ 的 R（0.6716）、HTQ 的 R（0.6734）、InfoQ 的 R（0.5937）下，BC 與 HAWQ 選的配置不同，而且都沒有評估過。
  - 所以 BC 與 HAWQ 不是在所有預算都等價，只是在本輪三個上限上恰好相同。在 HAWQ／HTQ 的 R 下，BC 的配置後來經補測評估過（見上一節）。
- **成本排序相關**（Spearman，19 層）：見表 3。

## 限制

- 只有 ResNet18、只有 A8 固定的 weight mixed precision；結論不外推到 W+A、LLM 或 MoE。
- 評估集只有一個 shard（3572 張）。McNemar 對 −0.7 pp 級的差異檢定力有限：HTQ 的「未定」不代表與 BC 相同。
- HTQ、InfoQ 的評分量化器（per-tensor）和部署量化器（per-channel）不同，這是忠於原方法的結果，但它們的劣勢有一部分可能來自這個不一致，本輪無法分離。
- InfoQ 的結果只代表本受控改寫，受 observer 對應、DINOv2 雙重 rescale、MQBench 背景量化、標籤映射等因素影響，不代表原論文的 QAT pipeline。
- 第 4 題的補測是在主要結果之後才登記的事後比較，而且只有 2 個點；它的結論是「沒有優勢的證據」，不是「等效」。
- 各方法的校準張數不同（10／10／32／64／1000），這是預先決定的「各用原生張數」，資料量差異算在成本裡。

## 修訂與冒煙測試揭露

`amendments.json` 共 4 筆，每筆都記錄了時間、原因和程式雜湊：
1. **00:25:54**：模型的 batch 固定為 1，onnxruntime logits 改成逐張計算。當時只看過資產。
2. **03:07:58**：冒煙測試後的修正，都在任何正式評分之前。內容：
   - pyhessian 路徑；mmap 切片複製。
   - InfoQ adapter：`SMI.vae` stub、block 輸出 observer 改為 `relu_1`（第一次誤取第一個 ReLU）、改用 `get_targets`（原本沒映射，MI(L;Y) ≈ 0）、加 `--max-layers`。
   - InfoQ venv 移除 CUDA 13 殘留。
   - 圖例位置。
3. **評估與分析之後**：只改圖的呈現。同配置的標記改成同心並加標註；離線時間圖移除會蓋住 InfoQ 點的圖例。
4. **評估與分析之後**：只改圖的呈現。knee 的標註移到點的右下方。

第 3、4 筆修訂後重跑 analyze，`summary.json`、`summary_table1.csv`、`summary_table2.csv`、`resources.csv` 的 sha256 都和修訂前相同。

冒煙測試在 scratch 目錄進行：校準池用 symlink，評估集是從池裡取的 12 張假評估集，從未碰過 held-out shard，每個階段都跑過一次（InfoQ 只跑部分層）。冒煙用的校準資料與正式相同，所以正式評估前已經知道成本表、上限和配置（全都是 calibration-only），但 held-out shard 在配置凍結前從未讀取。沒有在 held-out 資料上調整任何東西。

## 環境副作用

- InfoQ 的 `SMI.vae` 在 import 時呼叫 `load_imagenet_256x256_ds()`，在加入 stub 之前已經下載了整個 imagenet-1k-256x256。這在 `~/.cache/huggingface` 新增約 38 GB：
  - `hub/blobs` 19.8 GB（parquet）。
  - `datasets/benjamin-paine___imagenet-1k-256x256` 19 GB（arrow）。
  - 本 run 使用的是 `assets/` 裡自己的副本，不依賴這份快取。
  - 2026-10-29 經使用者同意後刪除：只刪該資料集專屬的 45 個 blob（19.76 GB，以及它們的 .lock／.refs）、資料集 repo 目錄和 arrow 快取。
  - DINOv2（`facebook/dinov2-small`）和其他模型的快取都保留，連結全部有效。
  - 清理後 `~/.cache/huggingface` 為 437 MB，磁碟可用空間從 28 GB 增加到 65 GB。
- InfoQ 隔離 venv：`/home/hoju/test/InfoQ/.venv_infoq`（12 GB），版本凍結在 `logs/infoq_venv_freeze.txt`；沒有執行 `setup.sh`，也沒有改動共用環境。
- 本 run 目錄 8.1 GB，其中 `assets/` 6.7 GB：train shards 3.2 GB、`eval_inputs.npy` 2.1 GB。

## 重現

在 `experiments/mixed_precision` 下執行：

```bash
R=runs/cnn_baselines_20261029
../../venv/bin/python run_cnn_baselines.py --stage plan --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage assets --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage controls --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage score_bc_control --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage score_bc --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage score_sqnr --run-dir $R
/home/hoju/test/HTQ/.venv/bin/python run_cnn_baselines.py --stage score_htq --run-dir $R
/home/hoju/test/HTQ/.venv/bin/python run_cnn_baselines.py --stage score_hawq --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage score_infoq --run-dir $R        # 會啟動 InfoQ 隔離 venv
../../venv/bin/python run_cnn_baselines.py --stage caps --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage search --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage evaluate --run-dir $R
../../venv/bin/python run_cnn_baselines.py --stage timing --run-dir $R --rep 2      # 以及 --rep 3
../../venv/bin/python run_cnn_baselines.py --stage score_infoq --run-dir $R --rep 2
../../venv/bin/python run_cnn_baselines.py --stage analyze --run-dir $R
../../venv/bin/python $R/controls_reps_resources.py
../../venv/bin/python $R/posthoc_matched_R_search.py
../../venv/bin/python run_cnn_baselines_matched.py --stage plan        # 補測：預先登記
../../venv/bin/python run_cnn_baselines_matched.py --stage evaluate
../../venv/bin/python run_cnn_baselines_matched.py --stage analyze
```

主要檔案：`summary.json`、`summary_table1.csv`、`summary_table2.csv`、`resources.csv`、`caps.json`、`configs.json`、`configs_freeze.json`、`scores/*_rep*.json`、`timing.jsonl`、`eval/<config>/{result.json,per_image.npz,logits.npy}`。
