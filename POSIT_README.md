# ONNX-MLIR Posit / ALPS 實驗版說明

本 branch（`tungtung9487/posit-work-20260329`）在官方 ONNX-MLIR 上加入 **Posit 數值格式**與 **ALPS（asinh companding）** 的 lowering pipeline，用來把 ONNX 模型編譯成以 posit 運算的 `.so`，並研究低位元 posit 對 CNN（MobileNetV2 / ResNet18）的精度影響。

> 官方說明見 `README.md`；本檔只講 posit 相關用法。改動細節見 `src/CHANGES.md`。

---

## 0. 最重要：怎麼看每一層 IR（onnx → posit → krnl → llvm → ll）

這是大多數人最想看的部分：一個 ONNX op 如何一階一階被 lower。**只需要 build 出 `onnx-mlir-opt` 和 `onnx-mlir`（見第 2 節），不需要資料集**。

### 0.1 一鍵 dump 全部階段

repo 內附 `src/mnist-12.onnx` 可直接用（不需外部模型）：

```bash
OPT=build/Release/bin/onnx-mlir-opt
MLIR=build/Release/bin/onnx-mlir            # 用來產生 onnx.mlir
TRANSLATE=../llvm-project/build/bin/mlir-translate
MODEL=src/mnist-12.onnx
FMT=p8e1                                    # posit 格式：p8e0/p8e1/p8e2/p16e1/p32e1...

# Stage 0: ONNX 模型 → ONNX dialect IR
$MLIR --EmitONNXIR -o /tmp/m $MODEL          # 產生 /tmp/m.onnx.mlir

# Stage 1: ONNX → Posit dialect  (nqdq 路徑：權重/activation 直接轉 posit)
env ONNX_MLIR_POSIT_FORCE_NQDQ=1 POSIT_COMPACT_CONSTANTS=1 \
  $OPT /tmp/m.onnx.mlir \
  --shape-inference --convert-onnx-to-posit --posit-format=$FMT \
  -o /tmp/m.posit.mlir

# Stage 2: Posit → Krnl dialect (展開成 runtime call + loop)
$OPT /tmp/m.posit.mlir \
  --canonicalize --shape-inference \
  --convert-onnx-to-krnl --convert-posit-to-krnl --canonicalize \
  -o /tmp/m.krnl.mlir

# Stage 3: Krnl → LLVM dialect
$OPT /tmp/m.krnl.mlir \
  --convert-krnl-to-affine --convert-krnl-to-llvm --reconcile-unrealized-casts \
  -o /tmp/m.llvm.mlir

# Stage 4: LLVM dialect → LLVM IR (.ll)
$TRANSLATE --mlir-to-llvmir /tmp/m.llvm.mlir -o /tmp/m.ll
```

產出：
| 檔案 | 階段 | 看什麼 |
|------|------|--------|
| `/tmp/m.onnx.mlir` | ONNX dialect | 原始 ONNX op（`onnx.Conv`, `onnx.Gemm`…）|
| `/tmp/m.posit.mlir` | Posit dialect | `posit.conv2d`, `posit.gemm`, `posit.from_f32`, `posit.constant`（含 ALPS metadata `compand_theta`）|
| `/tmp/m.krnl.mlir` | Krnl | `func.call @_mlir_ciface_posit_*`（runtime call）+ loop |
| `/tmp/m.llvm.mlir` | LLVM dialect | `llvm.*` op |
| `/tmp/m.ll` | LLVM IR | 最終 LLVM IR |

### 0.2 想看「每個 pass 之後」的 IR（逐 pass 追蹤）

加 `--mlir-print-ir-after=<pass>` 或把每個 pass 的 IR 各自存成檔：

```bash
# 印出指定 pass 之後的 IR 到 stderr
$OPT /tmp/m.onnx.mlir --shape-inference --convert-onnx-to-posit --posit-format=p8e1 \
  --mlir-print-ir-after=convert-onnx-to-posit 2>&1 | less

# 把整條 pipeline 每個 pass 的 IR 各存一檔到目錄
$OPT /tmp/m.onnx.mlir --shape-inference --convert-onnx-to-posit --posit-format=p8e1 \
  --mlir-print-ir-tree-dir=/tmp/ir_after_each_pass
```

### 0.3 用 MobileNetV2 / ResNet18

repo 內已附現成的 ONNX-IR（可跳過 Stage 0）：

```bash
src/mobilenetv2-12.onnx.mlir        # MobileNetV2 (nqdq/f32)
src/mobilenetv2-12-qdq.onnx.mlir    # MobileNetV2 (INT8 QDQ)
src/resnet50-v1-12-qdq.onnx.mlir    # ResNet50 (INT8 QDQ)
```

直接從 Stage 1 開始即可，例如：

```bash
env ONNX_MLIR_POSIT_FORCE_NQDQ=1 POSIT_COMPACT_CONSTANTS=1 \
  $OPT src/mobilenetv2-12.onnx.mlir \
  --shape-inference --convert-onnx-to-posit --posit-format=p8e1 \
  -o /tmp/mbv2.posit.mlir
```

本實驗用的 ImageNet100 MobileNetV2 / ResNet18 模型（`imagenet100_mobilenetv2.onnx`、`imagenet100_resnet18.onnx` 及其 `-int8-qdq` 版本）與資料集 **不在本 repo**（資料集約 1.3GB、`.so` 單檔達 650MB，皆超過 GitHub 上限）。**完整重現流程**（HuggingFace 資料集下載 → f32 訓練 → ONNX 匯出 → int8 QDQ 量化 → build/評估）與所需腳本見
[`experiments/imagenet100/README.md`](experiments/imagenet100/README.md)。上述 IR dump 流程對任何 ONNX 模型都適用。

---

## 1. 取得程式碼

```bash
git clone -b tungtung9487/posit-work-20260329 https://github.com/tungtung9487/onnx-mlir.git
cd onnx-mlir
```

---

## 2. Build

### 2.1 依賴

- **LLVM/MLIR**（`llvm-project`）：依官方 ONNX-MLIR 流程 build，放在 `../llvm-project`
- **Universal posit 庫**：用附的腳本安裝（會放到 `src/.deps/`，**已被 gitignore，不在 repo 內**）
  ```bash
  bash src/bash/install_posit_deps.sh
  ```

### 2.2 編譯

```bash
cmake --build build --target onnx-mlir-opt onnx-mlir -- -j4
```
產出 `build/Release/bin/onnx-mlir-opt`、`build/Release/bin/onnx-mlir`。
`mlir-translate` 來自 `../llvm-project/build/bin/`。

> 只要做第 0 節的 IR 觀察，build 出 `onnx-mlir-opt` + `onnx-mlir` 就夠了。

---

## 3. 編譯成可執行 `.so` 並跑（需要模型 + 資料集）

> 如何取得資料集與模型（下載 / f32 訓練 / ONNX 匯出 / int8 QDQ 量化）見
> [`experiments/imagenet100/README.md`](experiments/imagenet100/README.md)。

一鍵 build 各種 posit 格式的 `.so`（含 ALPS 等選項）：

```bash
bash src/bash/build_model11_sos.sh \
  --model-name <name> --nqdq-onnx <model.onnx> --out-dir <dir> \
  --posit-formats p8e1 --posit-source nqdq \
  --runtime-format-scope single --runtime-qalign-mode alps-only
```

跑資料集評估：`src/bash/time_model11_dataset_parallel.sh`（用法見腳本 `--help` 與 `src/CHANGES.md`）。

---

## 4. 關鍵概念

| 名詞 | 說明 |
|------|------|
| **nqdq path** | f32 模型直接轉 posit（權重/activation 都是 posit）|
| **qdq path** | INT8 QDQ 模型轉 posit（activation 經 INT8 再轉）|
| **ALPS** | asinh companding：`y=asinh(θ·x)`，把值壓進 posit 高精度區。**只套用在 conv/gemm 權重**；bias 一律 direct（見 `src/CHANGES.md` 的 bias 根因修正）|
| **weight ALPS** | build 時 grid search（`ONNX_MLIR_POSIT_CONST_ALPS=1`），baked 進 `.so` |
| **runtime output ALPS** | runtime 對 activation 輸出做 ALPS（`--output-alps-auto on`）|

詳細改動與除錯歷程見 **`src/CHANGES.md`**。
