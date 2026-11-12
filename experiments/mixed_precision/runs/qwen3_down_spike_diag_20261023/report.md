# Qwen3-0.6B Down-input criterion 跨模型單點診斷：spike 分組與同位置配對

日期：2026-10-23～24。腳本：`run_qwen_down_spike_diag.py`（sha256 `68199cdb21f1890f…`）。計畫：`plan.json`／`plan.md`（2026-10-23 23:45:45 寫入）。

## 性質與範圍

- **單層、單點診斷。** 每次只量化一層的 Down input，L0–L27 全部 28 層都做（層數由 config 驗證）。
- **沒有改動：** 正式 BC、配置搜尋、knee 選擇；也沒有做 fine-tuning。
- **真值來源：** GPU 上的離線 PyTorch QDQ 驗證，不是 onnx-mlir native 整合。
- **calibration 端：**
  - 用 CPU FP32 前向／反向；
  - 使用修正 RMSNorm epsilon 後的 native 參考 `runs/native_eps_fix_20261014/qwen3`；
  - 沿用原 10 個 calibration 窗口，沒有增加或更換。
- **舊實驗不可直接沿用：**
  - 舊的 Qwen paired-variance 實驗只涵蓋 L0／L2／L14／L27 的 Gate／Up x／y，不是本輪的 Down input；
  - 舊 Setup 只初始化 p6e1／p8e1。
  - 本輪自行建立 p4e1／p6e1／p8e1 三個 LUT，各自與 Universal 逐點驗證，不一致數為 0。

## 結論摘要

**新資料支持的結果**（128 篇新 WT103 文章 × 3 窗口）

84 個同位元主要配對中，78 對可判定（文章 bootstrap 95% CI 排除 0）。

| 方法 | 可判定配對排對數 | 相對 A 修正 | 相對 A 新增誤排 |
|---|---|---|---|
| A（原 BC） | 74 / 78 | — | — |
| B（spike 分組） | 74 / 78 | 0 | 0 |
| C（同位置配對） | 74 / 78 | 0 | 0 |

- **三種方法共同排錯的 4 對都在 L2 與 L27。** 這 4 對在 84 對的 simultaneous 95% CI 下也排除 0：
  - L2 int6–p6e1、L2 int8–p8e1；
  - L27 int4–p4e1、L27 int6–p6e1。
- **按預測選格式的實際損傷：**
  - B 的 84 個選擇與 A 完全相同。這在凍結預測時就已確定，所以差值恰為 0。
  - C 只在 L27 int8–p8e1 一對選得不同。R_C − R_A = −2.7e-5 nats，95% CI [−5.5e-5, +1.7e-6]；Bonferroni 97.5% CI [−6.0e-5, +6.0e-6]；Holm p = 0.13。未確認。

**決策**
- 預先登記的三個條件（選格式收益確認、修正分布 ≥ 3 層、穩定），B 與 C 都一個也沒有滿足。
- 依規則：保留原 BC，停止本輪，不追加門檻搜尋，不提出下一輪 W+A 配置比較。
- 本輪是單點結果，不代表任何整體 pipeline PPL 的變化。

**未確認、樣本不足或屬於次要分析的結果**
- 次要的跨位元配對中，B 修正 2 對，C 修正 6 對，分布在 L7、L19、L20、L22、L23、L27。
- 兩者都有 1 個「新增誤排」（L2 int6–int8）。這一對的正負號其實正確，只是相對差 1.2%／1.4%，低於 2%，依規則算 proxy tie。
- 這些是未校正的探索性描述，不進入決策。

**事後機制解釋**
- 剩下的錯誤集中在 position 0 的 massive activation（attention sink）。
- 在 L2、L27，即使用同位置配對的梯度，二階局部估計仍嚴重高估 Posit 在 position 0 的大誤差。
- B 與 C 把高估比例大幅縮小（L2 int6／p6e1 的分數比：A 699 倍 → B 14.8 倍 → C 9.1 倍），但沒有任何一對翻過來。
- 18 層的 calibration rows 沒有 spike，B 在這些層精確等於 A。

## 1. 設定

### 量化位置與格式

- 位置：h = SiLU(z_gate) · z_up，在乘法之後、Down linear 之前。
- 只量化 h，其餘權重（包括 Gate／Up／Down）與 activation 全部 FP32，Down output 不量化。
- 候選格式：int4、int6、int8、p4e1、p6e1、p8e1，另設 FP32 identity。
- INT 用 native calibration 的 Down x range 與 `erm.int_qparams`，以及修正後的 `int_act_qdq`（scale／zp 放在 tensor 所在裝置）。
- Posit 用 Universal 的 decision-boundary LUT。
- diagnostic 資料完全沒有用來重算 range。

### 三種分數

在原 10 個 calibration 窗口 × 128 位置上計算，n = 1280，float64，三種方法共用 μ_c 與 U。

- **A** T_old = Σ_c mean(g²)·Σ r² + U
- **B** T_group = Σ_c [mean_S(g²)·Σ_S r² + mean_N(g²)·Σ_N r²] + U
  - S：a_st = max_c |h_stc| ≥ 6 × median 的 rows；
  - 所有格式同一個 mask；組內不重新中心化。
- **C** T_pair = Σ g² r² + U

### 梯度

- g 是每窗口 summed next-token CE（127 個預測位置）對 h 的梯度。
- position 127 沒有預測目標，g = 0，但仍計入 n，與 native 的 positions = 128 相同。

### 真值

- 文章 NLL = 3 個窗口的 summed CE 之和 ÷ 381。
- ΔNLL = 量化後 NLL − 同一層 FP32 identity replay 的 NLL。在同一 chunk、同一裝置、同一程式路徑上計算，負值保留原號。
- 配對差 D = mean_a [ΔNLL(f1) − ΔNLL(f2)]。
- 文章 bootstrap：10000 次，seed 27103。每次重抽文章時，所有層、格式、配對、方法都用同一組重抽。

## 2. 控制

全部通過：`controls.json`，由 `controls_calib.json`、`controls_eval.json` 與 `audit.json` 合併。

| 控制 | 結果 |
|---|---|
| 自建 CausalLM 與 qwen3_prepare.CausalLM | logits 逐位元相同 |
| RMSNorm epsilon（config 與模組） | 1e-6 |
| calibration tokens | = native calib_inputs.npy = prepare validation 窗口 0–9 |
| Posit LUT（p4e1／p6e1／p8e1）vs Universal | 每個格式約 150 萬點，0 不一致 |
| hook 位置 | 捕捉到的 h 與 SiLU(gate(x))·up(x) 重算值逐位元相同；Down output = down_proj(h) 逐位元相同 |
| autograd g_h vs gY·W_down | 最大相對差 0 |
| position 127 的梯度 | 全部為 0 |
| 有 hook 與無 hook 的 calibration loss | 差 0 |
| FP32 identity | 誤差為 0；A／B／C／U 全為 0 |
| GPU QDQ vs CPU QDQ（28 層 × 6 格式的誤差 sha256） | 全部相同 |
| GPU 設定 | TF32 關閉（matmul 與 cuDNN）、deterministic algorithms、float32、matmul precision highest |
| GPU identity replay vs 同 chunk 無 hook 前向（12 chunk × 28 層，每窗口 summed CE） | 最大差 0（逐位元） |
| 合成資料 | 零誤差 → 0；常數梯度 → B = C = A（3.9e-16）；空組與無效門檻 → B 逐位元等於 A；分解恆等式 ≤ 6.5e-16 |
| 真實資料恆等式（V_group − V_old = 組間項；V_pair − V_group = 組內 covariance） | ≤ 2.6e-14 |
| PyTorch x+y 重建 vs native table C（28 層 × 6 格式） | median 相對差 6.2e-5，max 2.5e-3（L27 p4e1） |

補充說明：
- **table C 只作一致性控制。** 它包含 Down 的 input 與 output 兩個槽位，所以不能當作本輪的原 BC。本輪的 A 只用 x 槽位。
- **GPU 環境。** nvidia-smi 回報 NVML library 595.99 與 kernel module 595.84 版本不符，但 CUDA runtime 與 torch 2.14.0+cu130 正常運作。依指示沒有更動驅動。

## 3. 資料

### 歷史 Qwen token 資料核對（`audit.json`）

- **掃描範圍：** runs/ 下所有整數 token 陣列，條件是路徑含 qwen、或最大 id 超過 OLMoE 詞彙。共 49 個檔：
  - 43 個是文字檔：Qwen3 或 Qwen3.5 tokenizer 解碼後，每個窗口至少 96% 的 word 5-gram 落在 WikiText-2 文字中，判定為 WikiText-2；
  - 6 個是 tiny-check 的合成 id（< 1024，不是文字）；
  - 無法解析的檔：0。
- **tokenizer 控制：** 本輪使用的 Qwen3 tokenizer 能逐位元重現 prepare 的 validation 與 test 窗口。
- **重疊排除：** 全部 2084 個歷史窗口都納入抽樣時的重疊排除。規則是：任一歷史窗口有 ≥ 10% 的 shingles 落在候選文章中，或候選文章有 ≥ 10% 的 shingles 落在歷史窗口中，就排除該文章。

### 新文章（`segments.json`、`exclusions.json`、`data_freeze.json`）

- 128 篇 WikiText-103 train 文章 × 3 個連續、不重疊的 128-token 窗口。
- 抽樣條件：
  - 合格池 19674 篇；
  - verified 掃描排除 7721 篇歷史文章；
  - 抽取前已排除 WikiText-2 標題或內容重複，以及內部重複，共 1077 篇；
  - seed：順序 27101、起點 27102。
- 共訪問 144 篇，跳過 16 篇：
  - 與歷史 WT103 文章近重複：9 篇；
  - 與歷史 Qwen 窗口重疊：6 篇（窗口 share 0.108–0.382，例如 Saint Paul, Minnesota、Farm to Market Road 1528）；
  - 與 WT2 近重複：1 篇（USS Chew）。
- 接受文章的最大 share：WT2 0.097、歷史 WT103 0.0996、歷史 Qwen 窗口 0.090。
- 同一批文章用於所有層與所有格式。

## 4. 凍結預測（看任何新 loss 之前；`predictions.csv`、`freeze.json`）

- 84 個主要配對中：
  - B 的選擇與 A 完全相同（84/84）；
  - C 只有 1 對不同（L27 int8–p8e1：A、B 選 int8，C 選 p8e1）。
- 主要配對的 proxy tie 數：0。
- 次要的 336 對中，B 有 2 對選擇與 A 不同，C 有 8 對。
- 逐一移除 calibration 窗口時，三種方法在 84 個主要配對上都沒有任何選擇翻轉。
- 因此在量測新 loss 之前就已確定：B 不可能改變任何主要配對的結果，C 最多只能影響 1 對。

## 5. 排序結果

### 5.1 全部 84 個主要配對

- D = ΔNLL(INT) − ΔNLL(Posit)，單位 ×1e-3 nats/token；正值代表 INT 損傷較大。
- **粗體**：可判定（未校正 95% CI 排除 0）。†：simultaneous 95% CI 也排除 0。
- 三種方法在每一對的判斷都相同：✓ 排對、✗ 排錯、· 不確定（不計入對錯）。
- L27 int8–p8e1 例外：A／B 選 int8，C 選 p8e1；該對不確定。

| 層 | calib spike rows | int4−p4e1 D [95% CI] | | int6−p6e1 D [95% CI] | | int8−p8e1 D [95% CI] | |
|---|---|---|---|---|---|---|---|
| L0 | 0 | **+273.9 [+226.9, +330.2]** † | ✓ | **+34.71 [+29.78, +39.67]** † | ✓ | **+3.674 [+2.007, +5.264]** † | ✓ |
| L1 | 1（低） | **+2697.7 [+2589.3, +2804.2]** † | ✓ | **+72.68 [+66.04, +79.35]** † | ✓ | **+7.080 [+4.996, +9.145]** † | ✓ |
| L2 | 10 | **−3591.8 [−3661.0, −3523.6]** † | ✓ | **+218.0 [+201.7, +234.6]** † | ✗ | **+277.8 [+261.7, +294.5]** † | ✗ |
| L3 | 0 | **+153.0 [+142.3, +163.6]** † | ✓ | **+28.42 [+23.67, +33.34]** † | ✓ | **+3.063 [+1.546, +4.587]** † | ✓ |
| L4 | 0 | **+105.6 [+96.0, +114.9]** † | ✓ | **+29.94 [+25.33, +34.48]** † | ✓ | **+2.361 [+0.980, +3.699]** | ✓ |
| L5 | 1（低） | **+111.4 [+103.9, +118.9]** † | ✓ | **+22.30 [+18.48, +26.16]** † | ✓ | **+3.175 [+1.808, +4.580]** † | ✓ |
| L6 | 0 | **+82.6 [+75.8, +89.4]** † | ✓ | **+29.02 [+25.27, +32.74]** † | ✓ | **+1.813 [+0.798, +2.804]** † | ✓ |
| L7 | 45 | **+72.3 [+65.8, +78.7]** † | ✓ | **+60.83 [+56.49, +65.26]** † | ✓ | **+11.89 [+9.63, +14.12]** † | ✓ |
| L8 | 0 | **+37.5 [+32.6, +42.4]** † | ✓ | **+19.47 [+16.44, +22.46]** † | ✓ | **+1.371 [+0.496, +2.235]** | ✓ |
| L9 | 0 | **+32.9 [+27.8, +38.0]** † | ✓ | **+6.42 [+4.33, +8.51]** † | ✓ | +0.554 [−0.133, +1.212] | · |
| L10 | 28 | **+74.1 [+68.2, +80.3]** † | ✓ | **+39.17 [+34.23, +44.20]** † | ✓ | **+2.070 [+0.937, +3.197]** † | ✓ |
| L11 | 44 | **+32.3 [+28.6, +35.9]** † | ✓ | **+13.19 [+10.69, +15.75]** † | ✓ | **+2.892 [+2.000, +3.771]** † | ✓ |
| L12 | 0 | **+24.9 [+21.6, +28.3]** † | ✓ | **+7.32 [+5.73, +8.87]** † | ✓ | **+0.690 [+0.301, +1.064]** † | ✓ |
| L13 | 0 | **+7.86 [+4.96, +10.78]** † | ✓ | **+3.19 [+2.15, +4.26]** † | ✓ | +0.179 [−0.132, +0.480] | · |
| L14 | 4（低） | **+15.6 [+12.0, +19.1]** † | ✓ | **+5.96 [+3.85, +8.09]** † | ✓ | **+1.703 [+1.168, +2.253]** † | ✓ |
| L15 | 0 | **+29.6 [+26.4, +32.8]** † | ✓ | +1.66 [−0.14, +3.48] | · | **+0.561 [+0.056, +1.095]** | ✓ |
| L16 | 0 | **+31.9 [+27.8, +36.0]** † | ✓ | +0.20 [−1.49, +1.87] | · | **+0.842 [+0.359, +1.305]** † | ✓ |
| L17 | 0 | **+38.1 [+33.5, +42.7]** † | ✓ | **+6.44 [+3.70, +9.20]** † | ✓ | **+2.212 [+1.422, +2.992]** † | ✓ |
| L18 | 0 | **+30.3 [+26.5, +34.2]** † | ✓ | **+3.30 [+0.91, +5.64]** | ✓ | **+1.083 [+0.472, +1.703]** † | ✓ |
| L19 | 0 | **+45.6 [+40.8, +50.3]** † | ✓ | **+18.74 [+15.76, +21.79]** † | ✓ | **+2.967 [+2.142, +3.790]** † | ✓ |
| L20 | 0 | **+62.6 [+57.1, +68.2]** † | ✓ | **+22.31 [+19.10, +25.51]** † | ✓ | **+3.641 [+2.848, +4.439]** † | ✓ |
| L21 | 0 | **+38.8 [+33.3, +44.4]** † | ✓ | **+17.85 [+15.37, +20.29]** † | ✓ | **+3.941 [+2.888, +4.953]** † | ✓ |
| L22 | 0 | **+27.7 [+22.0, +33.3]** † | ✓ | **+7.30 [+4.70, +9.95]** † | ✓ | **+3.795 [+2.672, +4.891]** † | ✓ |
| L23 | 1（低） | **+27.1 [+21.0, +33.1]** † | ✓ | +1.39 [−1.02, +3.71] | · | **+2.569 [+1.594, +3.533]** † | ✓ |
| L24 | 0 | **+22.7 [+17.4, +28.0]** † | ✓ | **+7.09 [+5.27, +8.90]** † | ✓ | **+2.152 [+1.425, +2.878]** † | ✓ |
| L25 | 0 | **+28.6 [+23.5, +33.8]** † | ✓ | **+8.94 [+7.14, +10.68]** † | ✓ | **+2.970 [+2.156, +3.812]** † | ✓ |
| L26 | 10 | **+172.9 [+162.2, +183.7]** † | ✓ | **+46.22 [+41.89, +50.55]** † | ✓ | **+9.678 [+8.047, +11.226]** † | ✓ |
| L27 | 10 | **+249.1 [+234.4, +264.0]** † | ✗ | **+154.5 [+145.8, +163.2]** † | ✗ | +2.266 [−0.144, +4.643] | ·（C 選 p8e1） |

### 5.2 計數

| 範圍 | 配對數 | 可判定 | A 對 | B 對 | C 對 | B vs A 修正／新增 | C vs A 修正／新增 |
|---|---|---|---|---|---|---|---|
| 主要：全部 | 84 | 78 | 74 | 74 | 74 | 0／0 | 0／0 |
| 主要：int4–p4e1 | 28 | 28 | 27 | 27 | 27 | 0／0 | 0／0 |
| 主要：int6–p6e1 | 28 | 25 | 23 | 23 | 23 | 0／0 | 0／0 |
| 主要：int8–p8e1 | 28 | 25 | 24 | 24 | 24 | 0／0 | 0／0 |
| 主要：舊研究層 L0／L2／L14／L27 | 12 | 11 | 7 | 7 | 7 | 0／0 | 0／0 |
| 主要：其他層 | 72 | 67 | 67 | 67 | 67 | 0／0 | 0／0 |
| 次要（跨位元，探索） | 336 | 307 | 294 | 295 | 299 | 2／1* | 6／1* |

- \* 次要的「新增誤排」是 L2 int6–int8（D = +11.4e-3）。B 與 C 的正負號正確，但相對差只有 1.2%／1.4%，依預先規則算 proxy tie。
- 可判定配對上的 proxy tie：主要 0；次要 B 1、C 1，就是上面這一對。
- **三種方法共同排錯的 4 個主要配對**

  | 配對 | 實際 | Posit／INT 分數比（A → B → C） |
  |---|---|---|
  | L2 int6–p6e1 | INT 損傷大 0.218 | 699 → 14.8 → 9.1 |
  | L2 int8–p8e1 | INT 損傷大 0.278 | 171 → 2.6 → 1.6 |
  | L27 int4–p4e1 | INT 損傷大 0.249 | 4.1 → 3.2 → 1.9 |
  | L27 int6–p6e1 | INT 損傷大 0.155 | 3.6 → 2.5 → 1.2 |

  比值 > 1 代表預測 Posit 較差，也就是排錯。B、C 都朝正確方向移動，但沒有一對低於 1。

### 5.3 次要配對的修正（未校正，探索性；跨位元，不是同位元比較）

| 配對 | D (×1e-3) [95% CI] | A | B | C |
|---|---|---|---|---|
| L7 int8–p4e1 | +3.40 [+0.36, +6.32] | ✗ | ✓ | ✓ |
| L27 int4–p6e1 | +462.2 [+448.3, +475.7] | ✗ | ✓ | ✓ |
| L19 int6–p4e1 | −5.12 [−9.02, −1.32] | ✗ | ✗ | ✓ |
| L20 int6–p4e1 | −6.06 [−10.65, −1.48] | ✗ | ✗ | ✓ |
| L22 int8–p6e1 | −2.57 [−4.15, −1.03] | ✗ | ✗ | ✓ |
| L23 int8–p6e1 | −2.27 [−3.92, −0.72] | ✗ | ✗ | ✓ |

- 這 6 對在逐窗口移除下，C 都沒有翻轉。
- 但計畫沒有對次要配對做 simultaneous 校正，它們也不進入決策。
- 它們都是跨位元配對，資源不同，不能代表同位元下的選格式收益。

## 6. 按預測選格式的實際損傷

R_M 定義為每篇文章在 84 個主要配對上，所選格式 ΔNLL 的平均（nats/token）。

| 量 | 值 |
|---|---|
| R_A（原 BC） | 0.024324 |
| R_B | 0.024324（84 個選擇全同） |
| R_C | 0.024297 |
| 事後 oracle（同一資料上依實測選；只作尺度參考，偏樂觀） | 0.013589 |
| R_B − R_A | 0（95% 與 97.5% CI 都是 [0, 0]） |
| R_C − R_A | −2.70e-5；95% CI [−5.53e-5, +1.71e-6]；Bonferroni 97.5% CI [−6.01e-5, +5.95e-6]；雙尾 p = 0.064，Holm p = 0.128；68 篇變好、60 篇變差 |
| R_C − R_A（只在 int8–p8e1 這 28 對上平均） | −8.09e-5，95% CI [−1.66e-4, +5.1e-6] |

- 原 BC 與 oracle 的差 0.0107，幾乎全部來自 L2、L27 那 4 對大幅誤排。B 與 C 都沒有取回任何一部分。
- C 唯一不同的選擇是 L27 int8 → p8e1。兩者的 ΔNLL 點估計都是負值：int8 −0.21e-3，p8e1 −2.47e-3，C 選了點估計較低的一個。
  - 這也說明非負的 criterion 並不必然排不出負 ΔNLL 的順序。
  - 但這一對不確定，而且收益未通過多重比較，不能宣稱已確認修正。

## 7. 穩定性（逐一移除 calibration 窗口，門檻固定）

- 84 個主要配對：A、B、C 都沒有任何選擇翻轉。所以 4 個共同誤排是穩定的錯，不是窗口雜訊。
- 次要配對中有翻轉的對數：A 1、B 3、C 2。
- 這只反映單一 calibration 文章內的窗口敏感度，不能證明跨文章穩定。

## 8. 機制（事後解釋，只用來說明結果，不據此改分數；`mechanism.json`、`decomposition.csv`）

### spike 覆蓋

| 類別 | 層 | spike rows |
|---|---|---|
| 空組（B 精確等於 A） | 18 層：L0、L3、L4、L6、L8、L9、L12、L13、L15–L22、L24、L25 | 0 |
| 低覆蓋（< 8 rows） | L1、L5、L23 | 各 1 |
| 低覆蓋 | L14 | 4 |
| 有效分組 | L2、L26、L27 | 各 10，全部是 position 0 |
| 有效分組 | L7、L10、L11 | 45、28、44，主要是 ' the' 等常見 token；沒有 position 0 |

### position 0（L2、L26、L27）

**L2**
- 範圍與大小：Down x range [−3012, 3787]；row max |h| 的 median 只有 1.88，position 0 卻達 3787。
- 梯度：spike rows 的 g² 比 N 組小約 100 倍（1.7e-4 vs 1.7e-2）。
- Posit 誤差：spike rows 的 r² 極大，例如 p6e1 為 7.7e3 vs 0.48。A 用全體平均的 g² 去乘，嚴重高估 Posit；B 的組間項把 p6e1 從 1.17e6 降到 1.6e4。
- INT 的實際損傷：int4／int6／int8 的 ΔNLL 都約 0.28–0.29，與位元數幾乎無關。range 被 position 0 撐開後，一般值的量化幾乎全失；p8e1 只有 1e-4。

**L27**
- 範圍與大小：range [−1682, 721]；row max |h| 的 median 188。
- Posit 的 V_pair 有 65–84% 來自 position 0。
- U 在 int4／p4e1 的 T_old 中佔 44%／49%。

**共同解讀**
- 即使梯度在同位置配對，二階局部估計仍把 position 0 的 Posit 大誤差算得太重；實測的 loss 響應小得多。
- 這和先前 Qwen 研究的觀察一致：
  - `runs/qwen3_block_criterion_20261014`：position-0 sink outlier 是主要失敗來源；
  - `runs/qwen3_paired_variance_20261014`：L27 position 0 的 loss 響應約為 0 或為負。
- B 與 C 縮小了高估，但修正量不足以翻轉排序。

### 中層 spike（L7、L10、L11）

- spike rows 的 g² 約小 3–4 倍。
- Posit 的 r² 在 spike rows 大 17–76 倍（依層與格式），而且組間項集中在單一 channel，約 99%：L7 c2、L10 c17、L11 c43。
- B 因此大幅降低 Posit 分數，例如 L7 p4e1 從 267 降到 73。但同位元的主要配對原本就排對，所以結果不變。

### 組內項（C − B）

- 在 INT 格式上，組內項幾乎都為正（83/84），而且分散：
  - 組內項／V_old 的中位數 0.10，最大 1.43；
  - 前 5 rows 的份量中位數 0.09，最大 0.42；
  - 主導 channel 的份量最大 0.05。
- 例如 L27 int6 的 T_pair 是 T_old 的 2.2 倍。這來自 g² 與 r² 在一般 rows 間的正相關，不是 spike：L27 INT 的 V_pair 只有 0.1–0.4% 來自 position 0。
- Posit 的組內項比較集中：前 5 rows 份量的中位數 0.23，最大 0.78。

## 9. 成本（`timing.json`）

本輪是離線 PyTorch 診斷。以下時間都不是未來 native pipeline 的建表成本，評估時間也不能算進建表成本。

| 項目 | 時間／大小 |
|---|---|
| calibration FP32 前向／反向（CPU 12 threads，10 窗口，捕捉 28 層的 h、g_h、Down output 與其梯度） | 24.4 s |
| 候選 QDQ（28 層 × 6 格式，CPU） | 6.3 s |
| 誤差統計（μ、r²） | 1.86 s |
| 分數計算：A／B／C（28 層 × 6 格式，float64，不含上兩項） | 0.26 s／0.27 s／1.26 s |
| 逐窗口移除（10 次重算） | 38.7 s |
| calib 階段總計（含控制與 native 一致性） | 92.7 s |
| diagnostic 評估（GPU，384 窗口 × 28 層 × 7 種，含 identity） | 1309 s；其中 FP32 前向 21.6 s、replay 1277 s；GPU 峰值 8.1 GB |
| 診斷快取（h 與 g，float32） | 每層 31.5 MB，共 881 MB（work/，只供本診斷） |
| native 額外統計量估計（每層 3072 channel、6 格式、float64） | B 約 344 KB／層（Σ_S g²、n_S、每格式 Σ_S e 與 Σ_S e²）；C 約 295 KB／層（每格式 Σ g²e² 與 Σ g²e）。未實作，未量測 native 成本 |

## 10. 回答

**1. 三種方法各自排對多少？**
- 主要配對：78 對可判定，A、B、C 都排對 74 對；proxy tie 0。
- 次要配對：307 對可判定，A 294、B 295、C 299。次要只作探索描述。

**2. B／C 相對 A 修正多少、新增多少誤排？**
- 主要配對：都是 0 修正、0 新增。
- 次要配對：B +2／−1，C +6／−1；兩者的 −1 都是 proxy tie，不是正負號錯。

**3. 修正分布在哪些層與位元？**
- 主要配對：沒有修正。
- 次要配對（未校正）：L7 int8–p4e1、L27 int4–p6e1（B 與 C）；L19、L20 int6–p4e1；L22、L23 int8–p6e1（只有 C）。

**4. 是否只集中在 L0、L2、L27 等舊案例？**
- 剩下的誤排全部在 L2 與 L27：4 對同位元配對，simultaneous CI 確認，三種方法都排錯。
- B 與 C 沒有修正它們，只縮小了誤差幅度。
- 在其他 24 層，同位元配對原本就 67／67 全對，沒有可修正的空間。

**5. 相同位元下，按各方法挑格式的實際 NLL 是否改善？**
- 沒有可信收益。
- B 的選擇與 A 完全相同，收益為 0。
- C 只改 1 對；R_C − R_A = −2.7e-5，Bonferroni 97.5% CI 含 0，Holm p = 0.13。

**決策：** 保留原 BC，停止本輪；不追加門檻或倍數搜尋，不提出下一輪完整 W+A 配置比較。不得據此宣稱整體 pipeline PPL 有任何改善。

## 11. 冒煙測試、修改紀錄與重現

**冒煙測試（計畫前，scratchpad，不在 runs/）**
- calibration 階段完整跑過，看過其控制與 calibration 分數。
- 資料階段用其他 seed（999001／999002）抽了 6 篇，沒有評估。
- eval／analyze 只用 calibration 窗口，沒有產生任何新文章的 loss。
- 修正了 3 個程式錯誤：tokenizer 匯入順序、檢查時重入 hook、整個 chunk 的 logits 超出 GPU 記憶體。
- 設計沒有改。計畫寫入後腳本未再變更，每個階段都檢查了計畫中的程式與輸入 hash，因此沒有 amendment。

```
O=runs/qwen3_down_spike_diag_20261023
CPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE ../../venv/bin/python"
GPU="env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python"
$CPU run_qwen_down_spike_diag.py --stage plan    --out-dir $O     # 只能寫一次；新實驗請換新目錄
$CPU run_qwen_down_spike_diag.py --stage audit   --out-dir $O
$CPU run_qwen_down_spike_diag.py --stage calib   --out-dir $O     # 控制 + 凍結分數與預測
$CPU run_qwen_down_spike_diag.py --stage data    --out-dir $O
$GPU run_qwen_down_spike_diag.py --stage eval    --out-dir $O
$CPU run_qwen_down_spike_diag.py --stage analyze --out-dir $O
```

**hash（sha256 前 16 碼；腳本與計畫為完整碼）**

| 檔案 | sha256 |
|---|---|
| 腳本 | `68199cdb21f1890f7df790e67b17a80e8cb5487b6edd80e25e014f5ca41d033a` |
| plan.json | `701cb1264dfe3803801747d9090b950d50017f617c0a8e39f266a789ee4ae215` |
| plan.md | `dd8fb82e10e283b7e3985fe1b6bb66d529068ebeac376e8278b42068fb6d8bc5` |
| freeze.json | `5b3160a01cef8694` |
| scores.csv | `6fb4bfa2e5bb8d66` |
| predictions.csv | `f08665c9ad224007` |
| lowo.csv | `0a8b9123bf1efc2a` |
| tokens.npy | `65c1229910ec5e3a` |
| segments.json | `5cdcda14f049af7f` |
| exclusions.json | `7f38966d543001d3` |
| audit.json | `45e175a84f8548fa` |
| eval_ce.npz | `11dd272ca024a1ab` |
| per_article_losses.csv | `efdf4685ee32adb5` |
| per_window_losses.csv | `a390be8e0908d76a` |
| ranking.json | `a3311dc674654ca1` |
| summary.json | `93adb1a4189bfcd7` |
| controls.json | `adc5731b4f972be3` |
| timing.json | `b0a17617d8c05e9d` |

模型與輸入檔的 hash 記錄在 `plan.json["inputs"]`。

**輸出檔案**
- plan.md／plan.json
- controls.json，以及 controls_calib.json、controls_eval.json
- audit.json
- scores.csv、decomposition.csv、mechanism.json、predictions.csv、lowo.csv、native_consistency.csv
- tokens.npy、segments.json、exclusions.json、data_freeze.json
- eval_ce.npz、per_article_losses.csv、per_window_losses.csv
- ranking.json、ranking_pairs.csv、summary.json
- timing.json、timing.jsonl、logs/
