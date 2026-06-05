# Code Changes & Bug Fixes

---

## 2026-06-02

### Fix: qdq p8e1/p8e2 missing p8e0 helper symbol

**File**: `onnx-mlir/src/bash/build_model11_sos.sh`

**Symptom**:
```
dlopen failed: undefined symbol: _mlir_ciface_posit_dequantize_linear_axis_f32_p8e0
```
當使用 `--posit-source qdq --runtime-format-scope single` build p8e1 或 p8e2 時發生。

**Root cause**: case 分支只對 `p4e*|p5e*|p6e*|p7e*|p9e*` 加入 `-DPOSIT_RUNTIME_FMT_P8E0=1`，漏掉 p8e1/p8e2。QDQ per-axis weight dequantization 在 single format scope 下仍需要 p8e0 helper symbol。

**Fix**:
```bash
# Before
p4e*|p5e*|p6e*|p7e*|p9e*)
# After
p4e*|p5e*|p6e*|p7e*|p9e*|p8e1|p8e2)
```

**Action required**: qdq-p8e1.so / qdq-p8e2.so 需重新 build。

---

## 2026-06-05

### ROOT CAUSE FIX: bias 用 per-channel ALPS 但 Add kernel 只 per-tensor 解碼

**這是 MobileNetV2 nqdq-p8 崩塌（Top1=0）的真正根因。**

**症狀**: nqdq-p8e1 Top1=0%、預測幾乎常數。逐層 trace 對照 p16（正常）vs p8（崩）發現：expand gemm 後的 bias add，p16 把均值從 -6.76 拉到 +0.05（bias 貢獻 +6.8 置中），但 p8 只到 -5.79（bias 只貢獻 +0.3）→ **bias 損失 95%** → activation 均值停在很負 → ReLU6 把幾乎全部歸零 → feature 崩塌。

**根因**: 
- bias 常數在 build 時被 `shouldUsePerAxisConstMetadata` 判為 per-axis（per-channel）→ 編成 per-channel ALPS（每 channel 不同 theta）
- 但消費 bias 的 elemwise Add kernel（`posit_runtime.cpp:4968`）**只做 per-tensor metadata lookup**（`lookupTensorGPMetadata`），不做 per-channel
- → bias 用 per-channel theta 編碼、用 per-tensor metadata 解碼 → **encode/decode 不匹配** → bias 解成垃圾（損失 95%，不是精度 4% rounding）

**為什麼是 95% 不是 4%**: 純 p8 rounding 對 bias≈6.8 誤差才 ~4%；95% 損失證明是編解碼方式對不上（companding 不一致），不是精度不足。

**Fix 第一版（per-tensor ALPS）— 失敗**: 把 bias 從 per-axis 改 scalar（per-tensor）ALPS。trace 顯示 add 仍只 +0.32（bias 還是損失 95%）。原因：bias 的**per-tensor 動態範圍極端**（單一 bias vector 內達 137~5197 倍，值範圍 ±53，最小非零 0.0045），per-tensor ALPS grid search 挑了 theta≈theta_min(1e-4) 想涵蓋整個範圍，反而把所有 bias 值推進 posit 極小區壓爛。

**Fix 第二版（DIRECT，正確）**: bias-only 常數強制 **direct posit（完全不套 ALPS）**。
- `Math.cpp` 新增 `isBiasOnlyConstant()`：所有 use 都是 conv_bias/gemm_bias 才回 true
- `createCompactPositConstantIfEnabled`: bias → 跳過 scalarDecision（也跳過 per-axis）→ 落到 direct 編碼分支
- direct posit 是浮點式相對精度，每個 bias 值各給 ~幾% 精度（6.835→7 誤差 2%），足夠置中 activation
- 對應 INT8 把 bias 存 int32（高精度），不壓低位元

**為何 direct 對、ALPS 錯**: bias 動態範圍 5197 倍，一個 ALPS theta 蓋不住（顧大值就壓爛小值，反之亦然）。posit direct 本身就是 tapered 相對精度，天然適合高動態範圍的 bias。weight 維持 per-channel ALPS（動態範圍小、且 conv/gemm 走 per-channel decode）。

**需重 build**: onnx-mlir-opt + nqdq .so。

### Fix 第三版（真正根因）：bias 被「預先廣播」成完整常數，認不出

**第二版（direct）為何還是失敗**: trace 顯示即使原始 [16] bias 改 direct，add 仍只 +0.28。查 IR 發現——bias [16] 在 lowering 時被**展開廣播成 [1,16,12544] 完整常數**（200704 值）餵給 posit.add。這個展開常數**不是 Conv input[2]**（是 Add 運算元）→ `isBiasOnlyConstant` / `getConvOrGemmPerAxisConstantUseRole` 認不出（回 nullopt）→ 落到 scalar ALPS path → 被壓爛。我第一、二版只修到原始 [16] bias，但 add 實際用的是展開後的常數。

**根本原則 Fix**: **只有 Conv/Gemm WEIGHT 才套 ALPS，其他常數一律 direct**。
- `Math.cpp` 新增 `hasConvGemmWeightUse()`：常數有 conv_weight/gemm_weight use 才回 true
- `createCompactPositConstantIfEnabled`: `scalarDecision` 改成 `if (!perAxisDecision && isWeight)` ——非 weight 常數（含展開 bias）跳過 ALPS → direct
- 展開的 bias [1,16,12544] 餵 posit.add，無 weight use → isWeight=false → direct ✅

**驗證**: p16-plain 正常（bias p16）、p8+ALPS-bias 崩（bias 被 ALPS 壓爛）、INT8 85%（bias int32）。三方一致指向 bias 需要高精度/direct。bias 數值實測 ±53、per-tensor 動態範圍 5197 倍。原始 bias [16] 被廣播折疊成 [1,16,12544] 常數（200704 值）。

**v3 trace 驗證成功（2026-06-05）**: biasdirect_v2 build，第一個 bias add 從「gemm -5.18 → add -4.90（bias +0.28）」恢復成「gemm -5.19 → add +1.99（bias +7.18）」——bias 貢獻接近 p16 的 +6.8，add 均值轉正，ReLU6 不再歸零。崩塌根源解決。完整 Top1 eval 進行中。

## 2026-06-04

### Fix + New: clamp collect 改 per-part raw samples + merge + 整合 pipeline bash

**Problem**: 先前 `POSIT_COLLECT_LAYER_RANGES_FILE` 在 program exit 寫單一檔（atexit）。但 dataset runner 是「一張圖一個 process」（run_time.cpp 的 `--image` 只收單張），parallel 跑法下每個 process 覆蓋同一檔，500 張只留最後 1 張統計 → 壞的。

**Fix（posit_runtime.cpp，需重 build .so，不用 onnx-mlir-opt）**:
- `writeLayerRangeCSV` 改成寫 **raw reservoir 取樣值**（可 merge），不再寫 percentile。格式：`key,op,total_seen,n_samples,v0,v1,...`
- reservoir 上限 `kLayerRangesReservoirSize` 從 50000 降到 8000（per-image per-process，避免 part 檔過大）
- 設計改為 mirror ALPS offline：每張圖寫自己的 part 檔，再由 Python merge

**新增檔案**:
- `ImageNet100/merge_layer_ranges_to_clamp.py`：讀所有 `part_*.csv` raw samples → 每 key 合併（global-cap reservoir）→ 算 percentile → clamp [lo,hi] + theta → 輸出 clamp CSV
- `onnx-mlir/src/bash/run_clamp_pipeline.sh`：一鍵 pipeline wrapper
  - Step1 COLLECT：對指定 posit suffix 跑 N 張（parallel，f32 math），每張寫 `parts/part_i.csv`
  - Step2 MERGE：呼叫 merge python → `clamp_<suffix>/clamp.csv`
  - Step3 EVAL：用 `POSIT_OUTPUT_CLAMP_FILE` 跑完整 dataset eval（delegate 給 time_model11_dataset_parallel.sh）

**使用方式（env 控制）**:
```bash
CLAMP_DIR=/home/lai/onnx_mlir/ImageNet100/clamp_mobilenetv2_v1 \
CLAMP_COLLECT_LIMIT=500 \
CLAMP_PERCENTILE=99 \
CLAMP_THETA_MODE=auto \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/run_clamp_pipeline.sh \
  --model-name imagenet100_mobilenetv2 \
  --out-dir <nqdq build dir> \
  --image-dir /home/lai/onnx_mlir/ImageNet100/imagenet100_hf/validation \
  --image-preprocess-script /home/lai/onnx_mlir/ImageNet100/preprocess_imagenet100_tensor.py \
  --shape 1x3x224x224 \
  --suffixes nqdq-p8e1 \
  --jobs 25 --limit 5000 \
  --warmup 0 --iters 1 --no-benchmark --quire off \
  --output-alps-auto off --record-preds on
```
其他 env: `CLAMP_COLLECT_JOBS` `CLAMP_SUFFIX` `CLAMP_MARGIN` `CLAMP_FIXED_THETA` `CLAMP_GAMMA` `CLAMP_SYMMETRIC` `CLAMP_SKIP_COLLECT`（重用既有 clamp.csv）`CLAMP_VERSION_A`（=1 只輸出 key,lo,hi，配合 ALPS 使用）。

### clamp 與 ALPS 並用：用 Version A

ALPS 與 clamp 目的重疊但作用不同：ALPS 設 companding theta（精度分佈），clamp 夾範圍（防漂移）。修 ReLU6 collapse 的關鍵是夾範圍。

- **ALPS + clamp Version A**（推薦）：ALPS 編碼，clamp 只夾範圍並用既有 ALPS meta re-encode（theta=0 → 不覆蓋 ALPS theta）。merge 加 `--version-a`，pipeline 設 `CLAMP_VERSION_A=1`
- **ALPS + clamp Version B**（衝突）：clamp 自己的 theta 會覆蓋 ALPS theta（applyPositOutputClamp 的重註冊邏輯），ALPS 白做
- merge script 新增 `--version-a`：只輸出 `key,lo,hi`（loader 視為 Version A）

### Fix: clamp 不可套用在最終分類器 logits 層

**症狀**: nqdq + weight ALPS + clamp Version B 跑 100 張，Top1 仍 ≈0%。

**診斷（非程式 bug，是套用範圍錯）**: clamp.csv 的最終分類器 gemm2d（key `6244868847920025062`, 輸出 100 類）被夾到 `[-20, -1.31]`。每張圖勝出類別的 logit 落在跨 100 類分佈的最高 1%，p99 上界把識別正確答案的峰值砍平 → argmax 全亂。INT8 不會這樣因為它最後一層 calibration 涵蓋完整 logit 範圍。

**Fix**: merge script 新增 `--exclude-keys`（pipeline env `CLAMP_EXCLUDE_KEYS`），排除分類器層不 clamp。MobileNetV2 分類器 key（此 build）= `6244868847920025062`，可用 `nm + DOT_PROBE` 找出 gemm2d 輸出=100 的 key（不同 build key 不同）。

**通則**: 最終 logits 層永遠不要 clamp（需要完整範圍取 argmax）。

### New: theta-mode median（修 MobileNet 高動態範圍 activation）

**數據發現**: 收集的 per-layer activation 統計顯示——中位數 |x|≈0.75~3.25（≈posit 甜蜜點），但 max 達 40~64，**max/median = 13~53 倍**（MobileNetV2 的 6× expand 層 + ReLU6 造成的極端動態範圍）。

**問題**: `theta=1/max`（auto/from_max）被離群值綁架，把典型值 x≈1 壓成 `asinh(1/64)≈0.016`，推離 posit 甜蜜點 → 帶判別訊號的典型值失去精度。ResNet 無此問題（動態範圍緊）。INT8 用 per-tensor uniform affine 不受影響。

**Fix**: merge script 新增 `--theta-mode median`（pipeline `CLAMP_THETA_MODE=median`）+ `--theta-target`（pipeline `CLAMP_THETA_TARGET`，預設 1.0）。
- `theta = sinh(target) / median_abs`，讓典型值落在 posit 甜蜜點（|y|≈1），離群值靠 asinh 飽和 + clamp [lo,hi] 上限。
- 實測 theta 提升 6~12 倍（0.04→0.52 等）。

**驗證**: per-channel activation（方向 B）**非必要**——INT8 QDQ 的 activation 是 per-tensor（`per_channel=True` 只作用在 weight）。INT8 per-tensor 能到 85%，所以問題在 per-tensor 編碼方式（posit tapered+theta vs uniform affine），不在粒度。A（median theta）是對的方向。

**Action required**: 用前需重 build nqdq .so（現有為 6/1 舊版，不含新 collect 程式碼）。

### Fix: clamp Version B metadata 一致性（已驗證修正）

**Bug（確認屬實，靠程式碼追蹤）**: `applyPositOutputClamp` Version B 用新 theta re-encode posit bits，但 producing kernel（conv2d:6005 / gemm / clip:6166）已先註冊自己的 outMeta。下游 op 透過 `lookupTensorGPMetadata(tensorMetaPtr(input))` 解碼，拿到舊 outMeta，但 bits 是新 theta 編的 → 解碼錯誤。

**Fix（posit_runtime.cpp，applyPositOutputClamp 結尾）**:
```cpp
if (versionB)
  registerTensorGPMetadata(tensorMetaPtr(t), encodeMeta);
```
re-encode 後重新註冊 encodeMeta，下游用相同 meta 解碼。`tensorMetaPtr` 回傳 data pointer（穩定），kernel 註冊與 clamp 重註冊命中同一 key，clamp 在 kernel 之後跑所以正確覆蓋。

**驗證方式**: 程式碼追蹤 + compile OK。尚未 runtime 實測（需重 build .so）。

### New: POSIT_STORE_DQ_AS_POSIT — qdq Variant B persistent（INT8 → posit bits 持久存儲）

**Problem**: 在 qdq source 路徑中，`posit_dequantize_linear_f32_ref` 輸出 f32，posit bits 只是函式內部瞬時的（transient）中間值。Layer boundary 實際存儲的是 f32，不像 INT8 QDQ 是實際存 int8 bits。

**Fix（build-time env var）**:
- **`POSIT_STORE_DQ_AS_POSIT=1`**（或 `ONNX_MLIR_POSIT_STORE_DQ_AS_POSIT=1`）
- 設定後：`boundaryF32Flow=true` 的 DQ 節點改用 `posit_dequantize_linear_ref`（輸出 posit bits，不是 f32），再插入 `posit.to_f32` 讓下游 onnx.Conv 收到 f32
- 效果：`INT8 → posit_dequantize_linear_ref → [posit bits buffer, 跨 layer 持久存在] → posit.to_f32（當 Conv 需要時才 decode）→ f32`
- 與 INT8 QDQ 的對比：INT8 存 int8 bits；本方法存 posit bits（p4=4bits/value, p8=8bits/value）

**Changed files**（需重 build `onnx-mlir-opt`，再重 build `.so`）:

| 檔案 | 改動位置 | 說明 |
|------|----------|------|
| `src/Conversion/ONNXToPosit/ONNXToPosit.cpp` | `runOnOperation` | 讀 `POSIT_STORE_DQ_AS_POSIT` env var；pass 給 `populateONNXToPositConversionPattern()` |
| `src/Conversion/ONNXToPosit/Pattern/Math.cpp` | `ONNXDequantizeLinearOpLowering` | 新增 `preferStoreAsPosit_` member；在 `boundaryF32Flow=true` 路徑，若 flag 設定，改用 `positOutType` 輸出（呼叫 `posit_dequantize_linear_ref`），再 `toFinalOutputType` 插入 posit.to_f32；`populateONNXToPositConversionPattern()` 簽名加新參數 |

**Parameter 設定位置**: **Build time**

### New: POSIT_PREFER_DIRECT_FROM_QDQ — qdq Variant A（直接 posit，不經 INT8）

**Problem**: qdq source 的 DequantizeLinear 一律走 INT8 → posit → f32 路徑（Variant B）。若想讓 activation 從 orig_f32 直接 posit round-trip 而不經過 INT8 量化（Variant A），需要改 lowering 邏輯。

**Root cause**: `ONNXDequantizeLinearOpLowering::matchAndRewrite` 中，
`rewriteDirectFromQSource()` 只在 `!boundaryF32Flow`（DQ 輸出是 posit type）時被呼叫。對 qdq source 而言，DQ 輸出永遠是 f32（下游 `onnx.Conv` 需要 f32），所以 `boundaryF32Flow=true`，direct posit 路徑永遠被跳過。

**Fix（build-time env var）**:
- **`POSIT_PREFER_DIRECT_FROM_QDQ=1`**（或 `ONNX_MLIR_POSIT_PREFER_DIRECT_FROM_QDQ=1`）
- 設定後：即使 `boundaryF32Flow=true` 也嘗試 `rewriteDirectFromQSource()`
- 效果：`orig_f32 → posit.from_f32(qalign_key) → posit → posit.to_f32 → f32`（posit round-trip，不觸碰 INT8）
- 下游 `onnx.Conv` 仍然接收 f32，只是值來自 posit-rounded orig f32 而非 int8 linear DQ

**Changed files**（需重 build `onnx-mlir-opt`，再重 build `.so`）:

| 檔案 | 改動位置 | 說明 |
|------|----------|------|
| `src/Conversion/ONNXToPosit/ONNXToPosit.cpp` | pass class + `runOnOperation` | 新增 `preferDirectPositFromQDQ` bool member；讀 `POSIT_PREFER_DIRECT_FROM_QDQ` env var；pass 給 `populateONNXToPositConversionPattern()` |
| `src/Conversion/ONNXToPosit/Pattern/Math.cpp` | `ONNXDequantizeLinearOpLowering` | 新增 `preferDirectPositFromQDQ` parameter + member；condition 改為 `(preferDirectPositFromQDQ || !boundaryF32Flow) && preferDirectF32FromQDQ`；`populateONNXToPositConversionPattern()` 簽名加新參數 |

**Parameter 設定位置**: **Build time**（在跑 onnx-mlir-opt 的 shell 環境中設定，由 build_model11_sos.sh 呼叫 onnx-mlir-opt 時生效）。

### New: self-calibrated per-layer clamp + ALPS re-encode (Version B)

**Problem**: nqdq-p8e1 MobileNetV2 Top1≈0%，posit 精度誤差跨層累積，ReLU6 將 ~98% activation 歸零。目標：不依賴 QDQ ONNX，從 nqdq-f32 自行收集每層 activation 分佈，推算 calibrated range + ALPS theta，在每個 conv/gemm 輸出後做 clamp + ALPS re-encode。

**實作（posit_runtime.cpp）**:

1. **`PositOutputClampEntry` 新增 theta/gamma 欄位** (`posit_runtime.cpp:242`)
   - 原本：`struct PositOutputClampEntry { float lo, hi; }`
   - 現在：`struct PositOutputClampEntry { float lo, hi, theta=0, gamma=1; }`
   - theta > 0 → Version B（新 ALPS re-encode）；theta == 0 → Version A（原有行為）

2. **`applyPositOutputClamp` 支援 Version B** (`posit_runtime.cpp` 附近 3740 行)
   - 若 clamp.theta > 0：decode（用原 metadata）→ clamp → re-encode（用新 theta/gamma 的 ALPS TensorGPMetadata）
   - 這模擬了 INT8 的 per-layer requantize：每層輸出被壓縮進 calibrated range 再重新編碼

3. **`POSIT_COLLECT_LAYER_RANGES_FILE` 收集機制**
   - 新增 `PositLayerRangeReservoir`：per-key reservoir sampler（最多 50000 samples，Vitter Algorithm R）
   - `collectLayerRangeValues<Fmt>()` 在每個 conv2d / gemm 計算後收集 decoded output values
   - `writeLayerRangeCSV()`：program exit 時寫出 `key,op,n_samples,min,p50,p95,p99,p99.9,max`
   - 新增 `#include <random>`

4. **`loadPositOutputClampTable` 解析 optional theta/gamma 欄位**
   - 原格式：`key,lo,hi`（Version A）
   - 新格式：`key,lo,hi,theta,gamma`（Version B）
   - 自動計算並打印 versionA 和 versionB entry 數量

**新增 Python scripts**:

- `ImageNet100/gen_nqdq_layer_clamp_csv.py`：從 INT8 QDQ ONNX 萃取 calibrated range（依賴 QDQ model）
- `ImageNet100/gen_nqdq_layer_clamp_from_ranges.py`：從 `POSIT_COLLECT_LAYER_RANGES_FILE` 輸出推算 clamp + ALPS theta（不依賴 QDQ model）
  - 參數：`--percentile`（預設 99）、`--margin`、`--theta-mode`（auto/from_max/fixed）、`--symmetric`

**使用流程**:

```bash
# Step 1: 收集 nqdq-f32 的 activation 分佈（乾淨的 f32 統計）
POSIT_QOP_F32_MATH=on \
POSIT_COLLECT_LAYER_RANGES_FILE=/tmp/mobilenetv2_layer_ranges.csv \
bash time_model11_dataset_parallel.sh ... --suffixes nqdq-p8e1 --limit 500

# Step 2: 從統計推算 clamp + theta
python3 gen_nqdq_layer_clamp_from_ranges.py \
    --ranges /tmp/mobilenetv2_layer_ranges.csv \
    --output mobilenetv2_nqdq_selfcalibrated_clamp.csv \
    --percentile 99 --theta-mode auto

# Step 3: 用 Version B clamp 跑 posit eval
POSIT_OUTPUT_CLAMP_FILE=mobilenetv2_nqdq_selfcalibrated_clamp.csv \
bash time_model11_dataset_parallel.sh ... --suffixes nqdq-p8e1 --output-alps-auto off
```

**Status**: compile OK（2026-06-03），待測試。

### New: per-layer activation clamp for nqdq MobileNetV2

**Problem**: nqdq-p8e1 MobileNetV2 Top1≈0%。Posit 精度誤差導致 project conv 輸出超出 INT8 calibrated range，殘差連接後進入 expand conv → depthwise conv，最終 ReLU6 input 負漂嚴重，~98% activation 歸零。

**Approach**: 使用現有 `POSIT_OUTPUT_CLAMP_FILE` 機制（`posit_runtime.cpp:3706`），對每一層 conv2d / gemm3d 輸出，在計算後立即 decode → clamp to INT8 calibrated [lo, hi] → re-encode to posit。這模擬了 INT8 QDQ 的 per-layer QuantizeLinear 語義。

**實作（無 C++ 改動，只新增 Python 腳本）**:

- **腳本**: `ImageNet100/gen_nqdq_layer_clamp_csv.py`
  - 讀取 INT8 QDQ ONNX model（`imagenet100_mobilenetv2-int8-qdq.onnx`）
  - 找出每個 Conv → QuantizeLinear pair 的 scale/zp，換算 [lo, hi]
  - 讀取 DOT_PROBE CSV（`/tmp/probe_with_clips.csv`）取得 qalignKey
  - 按執行順序 match（共 52 個 conv2d + gemm3d ops，全部 OK）
  - 輸出 clamp CSV

- **產出 CSV**: `ImageNet100/mobilenetv2_nqdq_perlayer_clamp_v1.csv`
  - 52 個 entries：ReLU6 後的 expand/depthwise conv 輸出夾到 [0, 6]，project conv 夾到 INT8 calibrated 較寬範圍（例如 [-82.8, 79.6]）

**使用方式（runtime，不需重 build）**:
```bash
POSIT_OUTPUT_CLAMP_FILE=/home/lai/onnx_mlir/ImageNet100/mobilenetv2_nqdq_perlayer_clamp_v1.csv \
bash time_model11_dataset_parallel.sh ... --output-alps-auto off
```

**Status**: 待測試。預期可改善 nqdq-p8e1 Top1 超過目前 1.52%。

---

## 2026-05-21

### Fix: per-channel ALPS metadata fallback corruption

**File**: `onnx-mlir/src/posit_runtime.cpp`

**Symptom**: nqdq-p8e* MobileNetV2 logits 呈現週期性 pattern（n%4==0/1 全為 0，n%4==3 固定為 -0.96875）。Top1 接近 0%。

**Root cause**: MobileNetV2 有 52 個 Conv，總輸出 channel 17056，但 .so 只有 ~14536 個 per-channel metadata 註冊。缺少 metadata 的 channel 繼承了前一個錯誤 channel 的 ALPS/GP 參數。

**Fix**: 加入 `gTensorGPSpecialChannelMetaPtrs` / `gTensorGPAlpsChannelMetaPtrs` tracking set。缺失 channel 改為 fallback 到 direct/default posit，不再借用其他 channel 的 metadata。

**Action required**: 重新 build .so（不需重新 cmake）。

---

## 已知未解問題

### ReLU6 activation sparsity collapse（nqdq path）

**Symptom**: nqdq-p8e1 MobileNetV2 Top1=0%（sampled_v1 驗證）。

**Root cause**: posit 精度誤差使 activation 往負值漂移 → ReLU6 clip 後 ~98% 為零 → GlobalAveragePool 全零 → 分類失敗。ResNet18 不受影響（用 ReLU，無上界 clip）。

**Current direction**: 改用 qdq source path（`--posit-source qdq`），INT8 calibrated range 避免 unbounded drift。
