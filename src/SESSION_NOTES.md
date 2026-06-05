# SESSION_NOTES (Posit + ONNX-MLIR)

Last updated: 2026-05-21
Owner: tungtung9487

## 1) Current Git Branches (for resume on another machine)

- onnx-mlir fork:
  - Repo: https://github.com/tungtung9487/onnx-mlir
  - Branch: `tungtung9487/posit-work-20260329`
  - Commit: `04391764`
- llvm-project fork:
  - Repo: https://github.com/tungtung9487/llvm-project
  - Branch: `tungtung9487/llvm-snapshot-20260329`
  - Commit (HEAD when pushed): `113f01aa82d0`

## 2) Machine + Toolchain Actually Used

Current machine:

- CPU: `28 cores`
- RAM: `32G`
- Practical build policy:
  - `cmake --build . --parallel 8`  # 保守但不會太慢，避免整台機器卡死
  - `LLVM_PARALLEL_COMPILE_JOBS=8`  # 編譯可稍微開大
  - `LLVM_PARALLEL_LINK_JOBS=1`     # link 很吃 RAM，先保守

Current toolchain paths:

- `ONNX_MLIR_BIN=/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir`
- `ONNX_MLIR_OPT_BIN=/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt`
- `MLIR_TRANSLATE_BIN=/home/lai/onnx_mlir/llvm-project/build/bin/mlir-translate`
- `CLANGXX_BIN=/home/lai/onnx_mlir/llvm-project/build/bin/clang++`

Important:

- `mlir-translate` 必須和目前 LLVM build 同一套。  
  舊版 `/usr/local/bin/mlir-translate` 或系統版 `clang++` 可能吃不下新語法，例如 `inbounds|nuw`。
- 目前 wrapper / build script 都已經會優先使用上面這套工具。

## 3) LLVM Build Notes

不要直接用：

```bash
cmake --build . -- ${MAKEFLAGS}  # Ninja 仍可能自動開太多 jobs，容易卡住 / 爆 RAM
```

建議用：

```bash
cmake --build . --parallel 8  # 這台 28C / 32G 的實測保守值
```

如果是在 configure LLVM 時就想限制 compile / link 併行數：

```bash
cmake -G Ninja ../llvm \
  ... \
  -DLLVM_PARALLEL_COMPILE_JOBS=8 \  # 同時 compile 工作數
  -DLLVM_PARALLEL_LINK_JOBS=1       # 同時 link 工作數，先保守
```

常見卡住原因：

- compile jobs 開太多，CPU 100%
- 多個 `clang++` 同時跑，RAM 被吃滿後進 swap
- LLVM / MLIR 大型 target 在 link 時瞬間記憶體尖峰太高

## 4) onnx-mlir Build Missing Packages / Problems Seen on This Machine

### 4.1 Python headers 缺失

手動安裝：

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3.12-dev libpython3.12-dev  # 補 Python.h / pyconfig.h
```

### 4.2 Abseil 缺失

手動安裝：

```bash
sudo apt-get install -y libabsl-dev  # CMake 會在 configure 時用到
```

### 4.3 Full `cmake --build .` 可能遇到的兩個常見問題

1. `docs/doc_example` 因 `python onnx / numpy` 缺失失敗
- 目前 repo 已調整成：若沒有 Python `onnx`，預設 build 直接略過 `docs/doc_example`
- 如果要真的 build docs example，可再自行補：

```bash
sudo apt-get install -y python3-numpy python3-onnx  # 若想連 docs 範例也一起建
```

2. Python extension 連結到 static protobuf 時出現 `-fPIC` / shared lib 問題
- 目前 repo 已調整為使用 shared protobuf
- 關鍵 configure 選項：

```bash
-DONNX_USE_PROTOBUF_SHARED_LIBS=ON  # 避免 Python extension 連到不適合的 static protobuf
```

## 5) Current Canonical Model Artifact Location

之後一律以這個資料夾為準：

```text
src/temp/model/
```

目前保留且可用的標準 `EmitONNXIR` `.onnx.mlir` 檔：

- `src/temp/model/mnist-12.onnx.mlir`
- `src/temp/model/mnist-12-int8.onnx.mlir`
- `src/temp/model/mobilenetv2-12.onnx.mlir`
- `src/temp/model/mobilenetv2-12-qdq.onnx.mlir`
- `src/temp/model/resnet50-v1-12.onnx.mlir`
- `src/temp/model/resnet50-v1-12-qdq.onnx.mlir`

已清掉：

- `*.emitir.onnx.mlir`
- `*.current.onnx.mlir`
- `*.tmp`
- 像 `mobilenetv2-12-qdq.onnx.mlir.onnx.mlir` 這種重複命名檔

## 6) Canonical ONNX -> ONNX MLIR Generation

重要原則：

- 一律使用 `--EmitONNXIR`
- 一律補上兩個 elide 參數
- `-o` 請給 base name，不要把 `.onnx.mlir` 直接塞進 `-o`

### 6.1 Generic pattern

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # 產生 ONNX dialect MLIR，而不是 Basic 版本
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 避免大型常數直接塞爆字串顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 避免大型 dense attr 直接展開
  -o <out_base> \                                           # 只給 base name，輸出會自動是 <out_base>.onnx.mlir
  <model.onnx>                                              # 原始 ONNX 模型
```

### 6.2 ResNet50 non-QDQ

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # 產生正確 ONNXIR 版 MLIR
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o resnet50-v1-12 \                                       # 會輸出 resnet50-v1-12.onnx.mlir
  resnet50-v1-12.onnx                                       # non-qdq ONNX
```

### 6.3 ResNet50 QDQ

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # ResNet50 QDQ 必須走 IR 路，Basic 版會出錯
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o resnet50-v1-12-qdq \                                   # 會輸出 resnet50-v1-12-qdq.onnx.mlir
  resnet50-v1-12-qdq.onnx                                   # qdq ONNX
```

### 6.4 MobileNet

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # 使用 ONNXIR 版 MLIR
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o mobilenetv2-12 \                                       # 會輸出 mobilenetv2-12.onnx.mlir
  mobilenetv2-12.onnx                                       # non-qdq ONNX

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # qdq 也走 ONNXIR
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o mobilenetv2-12-qdq \                                   # 會輸出 mobilenetv2-12-qdq.onnx.mlir
  mobilenetv2-12-qdq.onnx                                   # qdq ONNX
```

### 6.5 MNIST

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # 產生標準 ONNXIR MLIR
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o mnist-12 \                                             # 會輸出 mnist-12.onnx.mlir
  mnist-12.onnx                                             # non-qdq ONNX

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
  --EmitONNXIR \                                            # qdq / int8 版本也用同一路生成
  --mlir-elide-resource-strings-if-larger=1000000000 \      # 大常數省略顯示
  --mlir-elide-elementsattrs-if-larger=1000000000 \         # 大 dense attr 省略顯示
  -o mnist-12-int8 \                                        # 會輸出 mnist-12-int8.onnx.mlir
  mnist-12-int8.onnx                                        # qdq / int8 ONNX
```

## 7) Single-Step Lowering Flow (for debugging)

### 7.1 ONNX / Posit -> Krnl

```bash
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt <model>.onnx.mlir \
  --shape-inference \                                       # 先做 shape inference
  --convert-onnx-to-posit \                                 # ONNX -> Posit
  --convert-posit-to-krnl \                                 # Posit -> Krnl
  --canonicalize \                                          # 清理 IR
  --convert-onnx-to-krnl \                                  # 剩餘 ONNX -> Krnl
  --convert-posit-to-krnl \                                 # 再跑一次 Posit -> Krnl，避免殘留
  --canonicalize \                                          # 再整理一次 IR
  --posit-format=p8e0 \                                     # 指定 posit 格式
  -o <model>.krnl.mlir                                      # 輸出 Krnl MLIR
```

### 7.2 Krnl -> LLVM dialect

```bash
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt <model>.krnl.mlir \
  --canonicalize \                                          # 先整理 IR
  --convert-krnl-to-affine \                                # 避免 krnl.define_loops 直接卡住
  --convert-krnl-to-llvm \                                  # Krnl -> LLVM dialect
  --reconcile-unrealized-casts \                            # 把 unrealized_conversion_cast 收乾淨
  -o <model>.llvm.mlir                                      # 輸出 LLVM dialect MLIR
```

### 7.3 LLVM dialect -> LLVM IR

```bash
/home/lai/onnx_mlir/llvm-project/build/bin/mlir-translate \
  --mlir-to-llvmir <model>.llvm.mlir \                      # LLVM dialect -> LLVM IR
  -o <model>.ll                                             # 輸出 .ll
```

### 7.4 .ll -> .so

由 wrapper script 自動完成：

- 用 `clang++`
- 連上 `src/posit_runtime.cpp`
- 連上 `libcruntime`
- 輸出 `.so`

## 8) Current Wrapper Behavior

目前 wrapper 已調整為固定使用 `src/temp/model/*.onnx.mlir`：

- `src/bash/build_mobilenet11_sos.sh`
- `src/bash/build_mobilenet_extra_sos.sh`
- `src/bash/build_resnet50_11_sos.sh`
- `src/bash/build_resnet50_extra_sos.sh`
- `src/bash/build_fasterrcnn11_sos.sh`
- `src/bash/build_fasterrcnn_extra_sos.sh`

也就是說：

- 不再依賴不存在的 `datasets/models/onnx`
- 不再優先使用舊的 `emitir/current/tmp` 檔
- 不再從 `src/*.onnx.mlir` 根目錄去抓
- 之後一律從 `src/temp/model` 的標準檔名抓

## 9) Root Cause Notes for ResNet50 QDQ

曾經出錯的檔：

- `src/temp/model/resnet50-v1-12-qdq.onnx.mlir` (舊版)

錯誤現象：

- `type of return operand 0 ('memref<?x1000xf32>') doesn't match function result type ('tensor<?x1000xf32>')`

實測結論：

- 舊版 `resnet50-v1-12-qdq.onnx.mlir` 行為與 fresh `EmitONNXBasic` 很像，會在 stage A 就失敗
- `resnet50-v1-12-qdq.current.onnx.mlir` / `resnet50-v1-12-qdq.emitir.onnx.mlir` / fresh `EmitONNXIR` 都能通過相同 stage A
- 因此目前標準 `resnet50-v1-12-qdq.onnx.mlir` 已重新用 `EmitONNXIR` 重生

結論：

- 這次先不用動 `Conversion`
- 先修 script 路徑 / import mode / 輸入 MLIR 來源就能解掉主要問題

## 10) How to Build the 11 `.so` Set

### 10.1 MobileNet 11-set

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_temp
# ./temp/mobilenet11_temp          -> 輸出目錄
# 預設會生成 qdq 9 個 posit 格式 + qdq-f32 + nqdq-f32 + run_time_sp
```

只測一個格式：

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_temp --posit-formats p8e0
# --posit-formats p8e0            -> 只先生成一個 posit 格式，方便 smoke test
```

### 10.2 ResNet50 11-set

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_resnet50_11_sos.sh ./temp/resnet50_11_temp
# ./temp/resnet50_11_temp         -> 輸出目錄
# 目前已驗證 p8e0 smoke test 可過
```

只測一個格式：

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_resnet50_11_sos.sh ./temp/resnet50_11_temp --posit-formats p8e0
# --posit-formats p8e0            -> 只先跑一個格式，快速驗證路徑是否正確
```

### 10.2b FasterRCNN 11-set

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_fasterrcnn11_sos.sh ./temp/fasterrcnn11_new --posit-formats p8e0
# 預設只建 posit .so（若要補建 qdq-f32/nqdq-f32，可加 --with-f32-baselines）
```

### 10.2c FasterRCNN extra-set

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_fasterrcnn_extra_sos.sh ./temp/fasterrcnn_extra_temp --posit-formats p9e3
# --posit-formats 可指定單一 extra 格式做 smoke
```

### 10.3 Generic core script

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/build_model11_sos.sh \
  --model-name mobilenetv2-12 \                             # 最終輸出檔名前綴
  --qdq-mlir ./temp/model/mobilenetv2-12-qdq.onnx.mlir \   # qdq MLIR
  --nqdq-mlir ./temp/model/mobilenetv2-12.onnx.mlir \      # non-qdq MLIR
  --backend universal \                                     # 使用 universal posit backend
  --shape-info "0:1x3x224x224" \                            # input 0 的 shape
  --out-dir ./temp/mobilenet11_temp \                       # 輸出目錄
  --posit-formats p8e0                                      # 只先跑一個 posit 格式
```

### 10.4 MNIST note

MNIST 的 `.onnx.mlir` 目前是正確 `EmitONNXIR` 產物，但不代表 11-set 一定能過。  
實測已知 generic path 仍可能卡在：

- `posit.dequantize_linear`

因此：

- `MNIST .onnx.mlir` 是正確的
- 但 MNIST 後續 lowering 可能仍失敗，這是已知例外

## 11) Smoke Tests Verified in This Session

已成功：

- `bash ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_smoketest4 --posit-formats p8e0`
- `bash ./bash/build_resnet50_11_sos.sh ./temp/resnet50_11_smoketest3 --posit-formats p8e0`

輸出示例：

- `mobilenetv2-12-qdq-p8e0.so`
- `mobilenetv2-12-qdq-f32.so`
- `mobilenetv2-12-nqdq-f32.so`
- `resnet50-v1-12-qdq-p8e0.so`
- `resnet50-v1-12-qdq-f32.so`
- `resnet50-v1-12-nqdq-f32.so`
- `run_time_sp`

MNIST：

- `.onnx.mlir` 正確
- generic 11-set path 仍可能失敗，先視為已知例外

## 12) Imagenette / Label Notes

目前 `src/temp` 內和 MobileNet / ResNet dataset 驗證直接相關的檔案為：

- `src/temp/imagenette_val_224/`                      # 1000 個 `1x3x224x224` txt 輸入
- `src/temp/imagenette_val_224_labels.txt`            # 1000 筆 `img_xxxxx.txt class_id`
- `src/temp/imagenette_val_224_manifest.tsv`          # 每個 txt 對應到哪張官方原圖
- `src/temp/imagenette2_320_subset_labels_full.txt`   # 官方下載三類子集的完整 label 清單
- `src/temp/imagenette_label_compare.txt`             # 現行 1000 筆與官方子集差異摘要

目前結論：

- `src/temp/imagenette_val_224/` + `src/temp/imagenette_val_224_labels.txt` 的格式是正確的
- 這份資料可直接餵給：
  - `time_model_single_format.sh`
  - `time_mobilenet11_dataset_parallel.sh`
  - `time_resnet50_11_dataset_parallel.sh`
- 這份 `imagenette_val_224_labels.txt` 是「和目前 1000 個 txt 完全對齊」的正確 label 檔
- 但它不是 fast.ai 官方直接附的一份 `label.txt`

已驗證事項：

- `wc -l src/temp/imagenette_val_224_labels.txt` 為 `1000`
- `src/temp/imagenette_val_224/` 內 txt 檔數為 `1000`
- `time_model_single_format.sh --model-name mobilenetv2-12 --format p8e0 ... --label-map ./temp/imagenette_val_224_labels.txt` 可正常讀取 label，且 `samples_missing_label=0`

重要說明：

- 目前這批 `src/temp/imagenette_val_224` 是「依官方 Imagenette 下載內容重建，並和現有 label-map 對齊」的 1000 筆子集
- 它是可用於目前驗證流程的正確資料格式
- 但不能證明和使用者早期手上的原始 `image1000` 逐張完全一致

官方資源現況：

- 官方 Imagenette 頁面提供資料集下載連結
- 官方 validation label 主要是由目錄名代表類別，而不是另外附一份 `img_xxxxx.txt class_id`
- 官方另外有一個 noisy labels CSV：
  - `noisy_imagenette.csv`
- 這個 CSV 是 noisy label 訓練資料描述，不是目前這種 validation txt 對照檔

因此目前對 label 的最準確描述是：

- `src/temp/imagenette_val_224_labels.txt` 對「目前 src/temp/imagenette_val_224 這批 1000 個 txt」來說是正確的
- 它是重建 / 對齊後的 label-map，不是官方 tarball 直接附的一份 validation label txt

### 12.1 MobileNet dataset parallel 指令

目前行為已改成動態 task queue：

- 每個 task = 一個 `(format, txt)`
- 任一 worker 空出來就立即撿下一個 task
- `--jobs` 只控制同時最多幾個 worker，不再是「一張圖整輪跑完才換下一張」
- format 會依 posit 優先度排序，較高 precision 先跑，例如：
  - `p32e2 > p32e1 > p32e0 > p16e2 > ... > p8e0 > qdq-f32 > nqdq-f32`
- 預設 `--task-progress on`
  - 終端機顯示：
    - `{task 37/1100} format=p32e1 sample=4/100`
- 若改成 `--task-progress off`
  - 終端機顯示：
    - `{done p32e2}`
    - `{done qdq-f32}`

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \                       # 11 個 .so 與 log 所在目錄
  --txt-dir ./temp/imagenette_val_224 \                   # 1000 個 1x3x224x224 txt 輸入
  --label-map ./temp/imagenette_val_224_labels.txt \      # 與 txt 對齊的 label 檔
  --jobs 12 \                                              # 平行 worker 數，這台 28C / 32G 可用 12
  --limit 100 \                                            # 只跑前 100 筆資料
  --warmup 0 \                                             # 不做 warmup
  --iters 1 \                                              # 每筆只跑 1 次
  --no-benchmark \                                         # 不做 benchmark，只做功能 / 正確性驗證
  --progress 1 \                                            # task-progress=on 時每 1 個 task 印一次
  --task-progress on                                        # on=細粒度 task 進度，off=格式完成進度
  --suffixes qdq-p8e0                                       # 只跑某一格式

bash ./bash/time_resnet50_11_dataset_parallel.sh --out-dir ./temp/resnet50_11_new --txt-dir ./temp/imagenette_val_224 --label-map ./temp/imagenette_val_224_labels.txt --jobs 25 --limit 1000 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

### 12.2 MobileNet extra dataset parallel 指令

`time_mobilenet_extra_dataset_parallel.sh` 目前也已改成同一套動態排程，不再是舊版「每個 format 各開一支外層 script」。

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/time_mobilenet_extra_dataset_parallel.sh \
  --out-dir ./temp/mobilenet_extra_temp \                  # extra .so 與 run_time_sp 所在目錄
  --txt-dir ./temp/imagenette_val_224 \                   # 1000 個 1x3x224x224 txt 輸入
  --label-map ./temp/imagenette_val_224_labels.txt \      # 與 txt 對齊的 label 檔
  --jobs 12 \                                              # 平行 worker 數
  --limit 100 \                                            # 只跑前 100 筆資料
  --warmup 0 \                                             # 不做 warmup
  --iters 1 \                                              # 每筆只跑 1 次
  --no-benchmark \                                         # 不做 benchmark，只做功能 / 正確性驗證
  --progress 1 \                                            # task-progress=on 時每 1 個 task 印一次
  --task-progress off                                       # off 時改印 {done p9e3} 這種格式完成提示
```

若只想先測少數 extra format：

```bash
bash ./bash/time_mobilenet_extra_dataset_parallel.sh \
  --out-dir ./temp/mobilenet_extra_temp \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --formats p9e3,p9e2,p7e3 \                               # 只挑幾個 extra format 先測
  --jobs 6 \
  --limit 20 \
  --no-benchmark
```

### 12.2b FasterRCNN dataset parallel 指令（11 / extra）

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

# 11-set（預設 baseline=none）
bash ./bash/time_fasterrcnn11_dataset_parallel.sh \
  --out-dir ./temp/fasterrcnn11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --jobs 1 \
  --limit 10 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --timeout-sec 20

# extra-set（可用 --formats 指定 extra 格式）
bash ./bash/time_fasterrcnn_extra_dataset_parallel.sh \
  --out-dir ./temp/fasterrcnn_extra_temp \
  --txt-dir ./temp/imagenette_val_224 \
  --formats p9e3 \
  --jobs 1 \
  --limit 10 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --timeout-sec 20
```

### 12.3 Single-format smoke test 指令

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/time_model_single_format.sh \
  --model-name mobilenetv2-12 \                            # 模型名稱
  --format p8e0 \                                          # 指定單一 posit 格式
  --out-dir ./temp/mobilenet11_new \                       # .so 所在目錄
  --txt-dir ./temp/imagenette_val_224 \                   # txt 輸入資料
  --shape 1x3x224x224 \                                    # 輸入 shape
  --limit 5 \                                              # 先做 5 筆 smoke test
  --label-map ./temp/imagenette_val_224_labels.txt \      # label 對照
  --baseline none \                                        # 不做 baseline 比較
  --no-benchmark \                                         # 不做 benchmark，只驗證功能
  --progress 1 \                                           # 每筆都印進度
  --log-file ./temp/mobilenet11_new/p8e0_limit5_label_nobase.log
```

## 13) Quick Resume Checklist (new machine)

1. Clone both forks + checkout branches in section 1.
2. Build LLVM / MLIR / onnx-mlir with moderate parallelism:

```bash
cmake --build . --parallel 8  # 這台 28C / 32G 的保守值
```

3. 補齊缺件：

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3.12-dev libpython3.12-dev libabsl-dev
```

4. 確認 toolchain：

```bash
echo "$ONNX_MLIR_BIN"       # 應指向 onnx-mlir build/bin/onnx-mlir
echo "$ONNX_MLIR_OPT_BIN"   # 應指向 onnx-mlir build/bin/onnx-mlir-opt
echo "$MLIR_TRANSLATE_BIN"  # 應指向 llvm-project/build/bin/mlir-translate
```

5. 重新生成模型 MLIR：

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model
# 依 section 6 重新跑 --EmitONNXIR
```

6. 使用 wrapper build `.so`：

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src
bash ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_temp --posit-formats p8e0
bash ./bash/build_resnet50_11_sos.sh ./temp/resnet50_11_temp --posit-formats p8e0 
實際使用：
    ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_new
    ./bash/build_resnet50_11_sos.sh ./temp/resnet50_11_new
    ./bash/build_mnist11_sos.sh ./temp/mnist_new    --without-f32-baselines可關f32nqdq qdq int8   
```

7. dataset compare / runtime test：

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src

bash ./bash/time_model_single_format.sh \
  --model-name mobilenetv2-12 \                             # 模型名稱
  --format p8e0 \                                           # 要測的 posit 格式
  --out-dir ./temp/mobilenet11_temp \                       # .so 所在目錄
  --txt-dir ./temp/imagenette_val_224 \                     # 測試資料
  --shape 1x3x224x224 \                                     # 輸入 shape
  --limit 100 \                                             # 只跑前 100 筆
  --label-map ./temp/imagenette_val_224_labels.txt \        # label 對照檔
  --baseline none \                                         # 不做 baseline 比較
  --no-benchmark \                                           # 只跑功能，不做 benchmark
  --progress 1 \                                            # 每 1 筆印一次進度
  --log-file ./temp/mobilenet11_temp/p8e0_limit100.log      # 輸出 log
```

## 14) YOLO / GPT-2 / Detection 模型補充 (2026-03-31)

### 14.1 新增模型與來源

本次新增並放在 `src/temp/model`：

- `gpt2-10.onnx`（文字生成 / next-token 預測）
- `yolov4.onnx`（物件偵測）
- `FasterRCNN-12.onnx`（物件偵測）
- `FasterRCNN-12-qdq.onnx`（FasterRCNN 同規格 qdq）

完整來源 URL 與 SHA256 寫在：

- `src/temp/model/SOURCE.md`
- `src/temp/model/MODEL_SHA256_20260331.txt`

### 14.2 生成 `.onnx.mlir`（統一 EmitONNXIR）

```bash
cd /home/lai/onnx_mlir/onnx-mlir/src/temp/model

/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir --EmitONNXIR FasterRCNN-12.onnx -o FasterRCNN-12
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir --EmitONNXIR FasterRCNN-12-qdq.onnx -o FasterRCNN-12-qdq
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir --EmitONNXIR yolov4.onnx -o yolov4
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir --EmitONNXIR gpt2-10.onnx -o gpt2-10
```

### 14.3 目前建議的「同規格 qdq 成對」清單

- `resnet50-v1-12` <-> `resnet50-v1-12-qdq`
- `mobilenetv2-12` <-> `mobilenetv2-12-qdq`
- `FasterRCNN-12` <-> `FasterRCNN-12-qdq`

`yolov4` 與 `gpt2-10` 目前這批來源尚無同名 qdq 對應，建議先用 non-qdq 做功能驗證。

## 15) MNIST QDQ：原本為何卡住、這次怎麼修 (2026-03-31)

### 15.1 原本卡住的主要原因

1. 純 onnx-mlir 路徑在 `mnist-12-int8.onnx.mlir` 會在 `ONNXDequantizeLinear` 崩潰  
   （`ONNXToKrnl/Math/Elementwise.cpp` assertion，因 per-axis/scalar 路徑不一致）。

2. Posit 路徑早期卡在 `onnx.DequantizeLinear` 無法 legalize（QDQ 分解後的 per-axis 權重/scale）。

3. `QLinearMatMul` 分支原本只接受 scalar qparams，MNIST 的 `BScale=tensor<10xf32>` 會失敗。

4. 分解後還會殘留 `onnx.QuantizeLinear`，後續 mixed pipeline 可能在 `ONNXQuantizeLinearOpLowering` 崩潰。

### 15.2 本次實作修補點

已修改：

- `src/Conversion/ONNXToPosit/ONNXToPosit.cpp`
- `src/Conversion/ONNXToPosit/Pattern/Math.cpp`

核心修法：

1. 先把 QDQ 特定 op 分解成標準 ONNX
- `onnx.QLinearConv -> DQ + Conv + Q`
- `onnx.Custom(function_name="QLinearAdd") -> DQ + Add + Q`

2. 修正分解時的 attribute 型別
- `onnx.Cast` 的 `to` 改為 `TypeAttr(f32)`（不是 enum int）
- `axis/saturate` 改為 `si64`

3. `ONNXDequantizeLinear` 降低：
- 常數折疊 case 允許 fallback 到 runtime case（讓 per-axis 可合法化）
- 新增 `DQ(MaxPool(Q(...)))` 直轉 posit maxpool，避免留下 Q island
- `DQ(QLinearMatMul(...))` 支援 per-axis qparams，並優先吃掉上游 `Q(x)` 回到 fractional/posit domain

4. `QLinearConv` bias 盡量在分解階段直接 fold 成 f32 constant  
   避免產生不必要的 scalar posit cast/broadcast 鏈。

### 15.3 驗證結果

成功生成：

- `src/temp/mnist11_fix_try2/mnist-12-qdq-p8e0.so`

回歸 smoke test（避免影響既有模型）：

- `src/temp/mobilenet11_regression_after_mnistfix/mobilenetv2-12-qdq-p8e0.so`
- `src/temp/resnet50_11_regression_after_mnistfix/resnet50-v1-12-qdq-p8e0.so`

## 16) FasterRCNN 現況與下一步 (2026-03-31)

### 16.1 先用「純 onnx-mlir lowering」比對的結論

對 `FasterRCNN-12.onnx.mlir` / `FasterRCNN-12-qdq.onnx.mlir`：

```bash
onnx-mlir-opt <model>.onnx.mlir \
  --shape-inference \
  --convert-onnx-to-krnl \
  --canonicalize \
  -o <model>.krnl.mlir

onnx-mlir-opt <model>.krnl.mlir \
  --canonicalize \
  --convert-krnl-to-affine \
  --convert-krnl-to-llvm \
  --reconcile-unrealized-casts \
  -o <model>.llvm.mlir
```

兩者都在 stage2 失敗，核心卡點是：

- `failed to legalize operation 'onnx.RoiAlign'`

也就是：`RoiAlign` 是目前 core onnx-mlir 路徑的真實 blocker。

### 16.2 Posit pipeline 目前狀態

在本次 MNIST 修補後，FasterRCNN 可先走到更後面，但 `build_model11_sos.sh` 目前仍卡在：

- `krnl->llvm` 階段出現 `tensor.dim` on posit tensor 殘留（`tensor<...x!posit.type<...>>`）

目前結論：

- 「先分解 QDQ 成標準 ONNX」這招可複用（已經讓路徑前進）
- 但 FasterRCNN 要完全打通，仍需要：
  1. 處理 `onnx.RoiAlign`（core 或 posit 對應 lowering）
  2. 清掉/合法化 `tensor.dim` on posit tensor 的殘留

## 17) 模型可用性快速檢查結果 (2026-03-31)

### 17.1 Level A：`shape-inference`（語法/基本可讀）

以下 `.onnx.mlir` 全部 PASS：

- `FasterRCNN-12.onnx.mlir`
- `FasterRCNN-12-qdq.onnx.mlir`
- `gpt2-10.onnx.mlir`
- `mnist-12.onnx.mlir`
- `mnist-12-int8.onnx.mlir`
- `mobilenetv2-12.onnx.mlir`
- `mobilenetv2-12-qdq.onnx.mlir`
- `resnet50-v1-12.onnx.mlir`
- `resnet50-v1-12-qdq.onnx.mlir`
- `yolov4.onnx.mlir`

### 17.2 Level B：core full path（`onnx -> krnl -> llvm`）

PASS：

- `gpt2-10.onnx.mlir`
- `mnist-12.onnx.mlir`
- `mobilenetv2-12.onnx.mlir`
- `mobilenetv2-12-qdq.onnx.mlir`
- `resnet50-v1-12.onnx.mlir`
- `resnet50-v1-12-qdq.onnx.mlir`
- `yolov4.onnx.mlir`

FAIL：

- `FasterRCNN-12.onnx.mlir`（`onnx.RoiAlign`）
- `FasterRCNN-12-qdq.onnx.mlir`（`onnx.RoiAlign`）
- `mnist-12-int8.onnx.mlir`（純 onnx path 的 `DequantizeLinear` 相關 crash）

### 17.3 建議的 preflight 慣例

新增模型時先跑：

```bash
onnx-mlir-opt <model>.onnx.mlir \
  --shape-inference \
  --convert-onnx-to-krnl \
  --canonicalize \
  -o /tmp/<model>.krnl.mlir

onnx-mlir-opt /tmp/<model>.krnl.mlir \
  --canonicalize \
  --convert-krnl-to-affine \
  --convert-krnl-to-llvm \
  --reconcile-unrealized-casts \
  -o /tmp/<model>.llvm.mlir
```

若這兩步都過，代表這份 `.onnx.mlir` 在 core onnx-mlir 路徑上沒有明顯未支援 OP 殘留。

## 18) FasterRCNN Runtime Guard / Fallback Log（2026-04-01）

本次已在 `src/posit_runtime.cpp` 新增三類防護的「觸發紀錄」：

- `memref_guard`：descriptor 合法性與 offset/overflow 防護被觸發
- `mapped_range`：descriptor 推估可讀大小與實際 mapped memory 不一致
- `quire_fallback`：Universal quire 計算拋例外後改走安全 fallback

### 18.1 Log 會長怎樣

執行時若有觸發，stderr 會出現：

```text
[PFALLBACK] kind=memref_guard count=...
[PFALLBACK] kind=mapped_range count=...
[PFALLBACK] kind=quire_fallback count=...
```

程序結束時會印總結：

```text
[PFALLBACK_SUMMARY] memref_guard=... mapped_range=... quire=...
```

可用環境變數控制：

- `POSIT_FALLBACK_LOG=0`：關閉逐筆 fallback log（summary 仍會在有觸發時輸出）
- `POSIT_FALLBACK_LOG_LIMIT=20`：限制最多印幾筆 `PFALLBACK`（預設 20）

### 18.2 為什麼會觸發（目前定位）

目前觀察到的主要根因偏向「轉換/descriptor 階段」而不是原始輸入圖片本身：

1. `mapped_range` 觸發原因  
   在 FasterRCNN 的某些中間張量，memref descriptor 宣告的 shape/stride 對應到的可存取範圍，明顯大於實際已配置記憶體。  
   例如 gdb 看到 `B: [1,64,43008]` 推得需要更大範圍，但實際在 `m.data + 172032` 就已越界。

2. `memref_guard` 觸發原因  
   多半是上述 descriptor 不一致延伸造成（offset/sizes/strides 組合不合理，或 index 累加可能溢位）。

3. `quire_fallback` 觸發原因  
   主要出現在 Universal quire 遇到 NaR/非實數運算時拋 `operand_is_nar`。  
   這通常是上游中間值已異常（由 descriptor/資料流問題放大），不是 Imagenette/MNIST 原始資料本身直接導致。

結論：

- 不是「原始資料天生就壞」為主因。
- 主要是某些模型在 posit pipeline / runtime 的中間張量描述與實際 buffer 對不上，才導致後續防護與 fallback 被觸發。

## 19) Descriptor Verifier + 分階段錯誤紀錄（2026-04-01）

### 19.1 PositToKrnl 內建 descriptor verifier

已在 `src/Conversion/PositToKrnl/PositToKrnl.cpp` 新增 `verifyPositRuntimeDescriptorCalls()`：

- 只檢查 `_mlir_ciface_posit_from_f32_*` / `_mlir_ciface_posit_to_f32_*` 這兩類 runtime bridge call。
- 驗證每個 call 的參數是否來自 ranked memref、是否 identity layout、元素型別是否符合（f32 / iN）。
- destination 參數要求為 `memref.alloc` 來源，避免不受控 descriptor 來源。
- 驗證失敗會在 lowering 階段直接 `emitError`，不是等 runtime 才炸。

### 19.2 descriptor 正規化 rewrite（避免壞 descriptor）

在 cast-chain rewrite 中新增兩類修正：

1. source 正規化：
   - 若 source memref 為 non-identity layout（例如 subview/strided），先 materialize 成 contiguous identity memref，再餵 runtime bridge。

2. destination 正規化：
   - destination 若是 non-identity layout，先用 identity memref alloc 作為 runtime 寫入目標，最後再 cast 回原需求型別。

這樣可把最容易出問題的 offset/stride descriptor 先收斂到可控形式。

### 19.3 build pipeline 分階段錯誤紀錄

`src/bash/build_model11_sos.sh` 支援分階段 log + pass-IR 輸出（每個 format 各自一份），
但為了省磁碟空間，現在預設 **關閉**；需要時才開：

- 開啟：`--keep-stage-logs`
- 關閉（預設）：`--no-stage-logs`

開啟後會產生：

- `stage_logs/<model>-qdq-<fmt>/01_03.posit_pipeline.log`
- `stage_logs/<model>-qdq-<fmt>/04.krnl_to_llvm.log`
- `stage_logs/<model>-qdq-<fmt>/05.mlir_translate.log`
- `stage_logs/<model>-qdq-<fmt>/06.link_so.log`
- `stage_logs/<model>-qdq-<fmt>/01_03.pass_ir/`
- `stage_logs/<model>-qdq-<fmt>/04.pass_ir/`

其中 `01_03.pass_ir` / `04.pass_ir` 會保留 pass-after IR（用 `--mlir-print-ir-after=...`），可直接定位哪一個 pass 開始壞掉。

## 20) 11-build 預設 F32 Baseline（2026-04-02）

11 系列 build 目前統一預設「會一起產生」：

- `<model>-qdq-f32.so`
- `<model>-nqdq-f32.so`
- `<model>-qdq-p*e*.so`（posit 格式）

目前狀態：

- `build_mobilenet11_sos.sh`：預設含 f32 baseline
- `build_resnet50_11_sos.sh`：預設含 f32 baseline
- `build_mnist11_sos.sh`：已改為預設含 f32 baseline
- `build_fasterrcnn11_sos.sh`：已改為預設含 f32 baseline
- `build_model11_sos.sh`：預設為「不嚴格 QDQ 模式」（優先 direct f32->posit from Q source）

如需只建 posit（不建 qdq-f32 / nqdq-f32）：

- 在 wrapper 加 `--without-f32-baselines`
- 或直接傳遞 `--skip-f32-baselines` 到 `build_model11_sos.sh`

如需切換 QDQ 模式：

- 嚴格模式：`--strict-qdq-mode`（同時開啟 `--align-to-int8-qdomain`）
- 不嚴格模式：`--non-strict-qdq-mode`（預設）
- 若只想要求 q-domain materialization（但不切換嚴格模式）可單獨加 `--align-to-int8-qdomain`
- 例如（嚴格）：

```bash
bash ./bash/build_mobilenet11_sos.sh ./temp/mobilenet11_new --strict-qdq-mode
```

後續新增模型的 11-wrapper 原則：

- 預設不要加 `--skip-f32-baselines`
- 讓 `build_model11_sos.sh` 的預設行為保留 f32 baseline，避免 dataset compare / Top1 驗證時缺基準。

## 21) Per-QDQ / Per-Channel Alpha 校正流程（2026-04-08）

### 21.1 目前狀態（預設行為）

- 11-build 預設為不嚴格 QDQ：不需手動加參數。
- 若要回到嚴格 ONNX QDQ lowering，請加 `--strict-qdq-mode`。
- `alpha` 校正表不會自動套用；要在執行 runtime 時指定環境變數：
  - `POSIT_QALIGN_FILE=<qalign_csv>`

### 21.2 新模型如何使用 alpha（標準 4 步）

1. 先只編 p8e*：

```bash
bash ./bash/build_<model>11_sos.sh <out_dir> --posit-formats p8e0,p8e1,p8e2
```

2. 收集 QALIGN 樣本（先以 p8e0 為例）：

```bash
POSIT_QALIGN_COLLECT_FILE=<out_dir>/qalign_collect.csv \
POSIT_QALIGN_COLLECT_PER_CALL=256 \
POSIT_QALIGN_COLLECT_MAX_PER_BUCKET=1024 \
<out_dir>/run_time_sp <out_dir>/<model>-qdq-p8e0.so <input.txt> \
  --shape <N>x<C>x<H>x<W> --out-type p8e0 --entry _mlir_ciface_main_graph
```

3. 從 collect 檔產生校正表：

```bash
bash ./bash/calibrate_qalign_pernode.sh \
  --collect-csv <out_dir>/qalign_collect.csv \
  --out-dir <out_dir>/calibration
```

會產出：

- `<out_dir>/calibration/qalign_p8e0.csv`
- `<out_dir>/calibration/qalign_p8e1.csv`
- `<out_dir>/calibration/qalign_p8e2.csv`

4. 套用 alpha 再跑：

```bash
POSIT_QALIGN_FILE=<out_dir>/calibration/qalign_p8e0.csv \
<out_dir>/run_time_sp <out_dir>/<model>-qdq-p8e0.so <input.txt> \
  --shape <N>x<C>x<H>x<W> --out-type p8e0 --entry _mlir_ciface_main_graph
```

`p8e1/p8e2` 同理，換對應 `.so` 與 `qalign_p8e1.csv / qalign_p8e2.csv`。

### 21.3 「產生 alpha 後要收集什麼數據」是什麼意思？

`POSIT_QALIGN_COLLECT_FILE` 會記錄每個 bucket（`qalign_key + channel`）的樣本值：

- `key`：QDQ 節點 key（編譯時生成）
- `channel`：channel id（`-1` 表 per-tensor/per-node；`>=0` 表 per-channel）
- `seen` / `sample_count`
- `samples`：該 bucket 的實際輸入樣本（runtime 觀測值）

校正工具會用這些樣本，對候選 alpha 做誤差評估，選出每個 bucket 的 alpha。

### 21.4 為什麼不能只在「編譯階段」直接決定 alpha？

原因是 alpha 依賴「實際輸入分佈」：

- 編譯階段只有 graph + 常數權重，拿不到 deployment 時 activation 分佈。
- 同一模型在不同資料集/前處理下，最佳 alpha 可能不同。
- 所以目前設計是 profile-guided：先跑代表性資料 collect，再離線校正，再套用。

補充：

- 理論上可做「純編譯期 heuristic alpha」，但通常不穩定，效果常不如資料驅動校正。
- 若未來要全自動，可做成「編譯後自動跑 calibration dataset -> 產生 qalign 表 -> 部署時載入」流程。

### 21.5 一鍵 Bash 流程（11 / extra / single）

新增腳本：

- `src/bash/qalign_pipeline_11.sh`
- `src/bash/qalign_pipeline_extra.sh`
- `src/bash/qalign_pipeline_single.sh`
- 核心流程（供 wrapper 呼叫）：`src/bash/qalign_pipeline_model.sh`

1) 11 流程（推薦）

```bash
bash ./bash/qalign_pipeline_11.sh \
  --model-name mobilenetv2-12 \
  --out-dir ./temp/mobilenet11_new \
  --input-txt ./temp/imagenette_val_224/img_00000.txt \
  --shape 1x3x224x224 \
  --collect-format p8e0 \
  --iters 10
```

2) extra 流程（可同時建 p8 + extra 格式）

```bash
bash ./bash/qalign_pipeline_extra.sh \
  --model-name resnet50-v1-12 \
  --out-dir ./temp/resnet50_extra_temp \
  --input-txt ./temp/imagenette_val_224/img_00000.txt \
  --shape 1x3x224x224 \
  --collect-format p8e1 \
  --iters 10
```

3) single 流程（已有 runner + so 時）

```bash
bash ./bash/qalign_pipeline_single.sh \
  --runner ./temp/mobilenet11_new/run_time_sp \
  --so ./temp/mobilenet11_new/mobilenetv2-12-qdq-p8e0.so \
  --input-txt ./temp/imagenette_val_224/img_00000.txt \
  --shape 1x3x224x224 \
  --out-type p8e0 \
  --iters 10
```

single 多圖 collect（新支援）：

```bash
bash ./bash/qalign_pipeline_single.sh \
  --runner ./temp/mobilenet11_new/run_time_sp \
  --so ./temp/mobilenet11_new/mobilenetv2-12-qdq-p8e0.so \
  --txt-dir ./temp/imagenette_val_224 \
  --limit 100 \
  --shape 1x3x224x224 \
  --out-type p8e0 \
  --iters 1
```

補充：

- `qalign_pipeline_11.sh` / `qalign_pipeline_extra.sh` 支援在 `--` 後面加 build script 額外參數。
- `qalign_pipeline_single.sh` 不做 build，只做 collect/calibrate/apply-check。

### 21.6 一鍵整合到 time_model11_dataset_parallel.sh（2026-04-08）

現在 `time_model11_dataset_parallel.sh` 直接內建 qalign 前置流程（方法一：先收斂再固定）：

- 預設 `--qalign-auto on`
- 預設 `--qalign-formats p8e0,p8e1,p8e2`
- 預設 `--qalign-mode alps`（可切 `off|alpha|alps`）
- 新增 `--qalign-force-compand on|off`（預設跟隨 `--qalign-mode`；可手動覆蓋）
- 預設 `--qalign-format-jobs 1`（預設不會同時跑多個格式，避免吃太多資源）
- qalign 每個格式在收集多張圖時，會自動用 `--jobs` 做動態平行（誰空閒誰接下一張）
- 會先做 qalign collect/calibrate 到收斂，再進 dataset 正式統計
- 正式跑每個格式時，自動套用對應 `qalign_*.csv`

重要澄清（避免混淆）：

- `qalign_pipeline_11.sh` 預設只處理 `p8e0,p8e1,p8e2`（用來收集/校正 alpha）。
- 正式 dataset 驗證仍是跑完整 11 格式（`p8/p16/p32 + qdq-f32 + nqdq-f32`）。
- 若目前 `out_dir` 只有 `p8e*` 的 `.so`，先補跑一次 `build_*11_sos.sh` 產生完整 11 個 `.so`，再跑 `time_*11_dataset_parallel.sh`。

最小範例（mobilenet）：

```bash
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 12 --limit 100 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

若要關掉 ALPS（切回 alpha/off）：

```bash
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --qalign-mode off \
  --jobs 12 --limit 100 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

若要保留 `--qalign-mode alps`，但校正階段改成「可回退到 off baseline」（不強制 ALPS-only）：

```bash
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --qalign-mode alps \
  --qalign-force-compand off \
  --jobs 12 --limit 100 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

只想做 p8e0 的 qalign：

```bash
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --suffixes qdq-p8e0,qdq-f32,nqdq-f32 \
  --qalign-formats p8e0 \
  --jobs 12 --limit 100 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

若已有現成 qalign CSV（省時間，不重收集）：

```bash
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --qalign-auto off \
  --qalign-csv-dir ./temp/mobilenet11_new/qalign_calibration \
  --jobs 12 --limit 100 --warmup 0 --iters 1 --no-benchmark --progress 1 --task-progress on
```

可調收斂參數：

- `--qalign-batch-size`（每輪新增樣本數，預設 25）
- `--qalign-format-jobs`（qalign 收集同時跑幾個格式，預設 1）
- `--qalign-limit`（qalign 最多看幾張，預設 1000）
- `--qalign-new-bucket-ratio`（新 bucket 比例門檻，預設 0.01）
- `--qalign-mae-delta`（相鄰輪 avg mae_improve 變化門檻，預設 0.0001）
- `--qalign-min-rounds` / `--qalign-stable-rounds`（預設 2 / 2）

### 21.7 論文 1909.03831 參考版 QALIGN（2026-04-13）

參考論文《Training Deep Neural Networks Using Posit Number System》（arXiv:1909.03831）後，
已把「distribution-based shifting」概念加進 qalign 校正器：

- 論文基準公式（以 bucket 樣本 `x` 計）：
  - `center = round(mean(log2(|x0|)))`，`x0 = clip(|x|, minpos, maxpos)`
  - `Sf = 2^(center + sigma)`（論文 `sigma` 預設為 2）
  - 對應到 runtime qalign 實作的 `alpha`：`alpha = 1 / Sf`
- 校正器預設改為「hybrid」：
  - 同時比較 legacy MAE 最佳 alpha 與論文 alpha（含鄰近候選）
  - 用 mixed score 選最佳（避免強制套用論文公式造成退化）

校正器檔案：

- `src/temp/probe/qalign_calibrate_from_collect.cpp`

可調環境變數（`calibrate_qalign_pernode.sh` 執行時生效）：

- `QALIGN_PAPER_SIGMA`（預設 `2`）
- `QALIGN_AUTO_SIGMA`（預設 `off`；可設 `global` / `per-format` 自動搜尋最佳 `sigma`）
- `QALIGN_SIGMA_MIN`（預設 `-4`，自動搜尋下界）
- `QALIGN_SIGMA_MAX`（預設 `8`，自動搜尋上界）
- `QALIGN_SCORE_MAE_WEIGHT`（預設 `0.25`，其餘權重給 weighted-MAE）
- `QALIGN_SCORE_ALPHA_REG`（預設 `0.0`，可加 alpha 正則）
- `QALIGN_FORCE_PAPER_ALPHA`（預設 `off`，開啟可做「純論文公式」A/B）

detail CSV（`qalign_*_detail.csv`）目前會包含：

- `sigma_used`（該格式/該 bucket 實際使用的 sigma）
- `sf_paper`（論文式 `Sf=2^(center+sigma)`）
- `center_log2`（`round(mean(log2(|x0|)))`）

執行範例（collect 後做 calibration）：

```bash
# A) 固定 sigma（舊行為）
QALIGN_AUTO_SIGMA=off \
QALIGN_PAPER_SIGMA=2 \
bash ./bash/calibrate_qalign_pernode.sh \
  --collect-csv ./temp/mobilenet11_alpha_test/qalign_collect_single.csv \
  --out-dir ./temp/mobilenet11_alpha_test/qalign_calibration_fixed

# B) 自動搜尋單一 global sigma（p8e0/p8e1/p8e2 共用）
QALIGN_AUTO_SIGMA=global \
QALIGN_SIGMA_MIN=-4 \
QALIGN_SIGMA_MAX=8 \
bash ./bash/calibrate_qalign_pernode.sh \
  --collect-csv ./temp/mobilenet11_alpha_test/qalign_collect_single.csv \
  --out-dir ./temp/mobilenet11_alpha_test/qalign_calibration_global

# C) 自動搜尋 per-format sigma（各格式可不同）
QALIGN_AUTO_SIGMA=per-format \
QALIGN_SIGMA_MIN=-4 \
QALIGN_SIGMA_MAX=8 \
bash ./bash/calibrate_qalign_pernode.sh \
  --collect-csv ./temp/mobilenet11_alpha_test/qalign_collect_single.csv \
  --out-dir ./temp/mobilenet11_alpha_test/qalign_calibration_per_format
```

本機實測（mobilenet `qalign_collect_single.csv`）：

- hybrid（預設）：
  - `p8e0/p8e1/p8e2` 的 `non_identity_alpha` 皆為 `0`
  - 代表此資料分佈下，`alpha=1` 仍是最佳解
- force-paper（`QALIGN_FORCE_PAPER_ALPHA=on`）：
  - `non_identity_alpha` 明顯增加（約 51~54/65）
  - 但 `avg_mae_improve` / `avg_wmae_improve` 皆為負（整體退化）
  - 單圖 smoke 也出現 top1 變差（baseline/hybrid: `342`，force-paper: `522`）

結論：

- 這篇論文的 shifting 公式已可直接套用到現有流程做 A/B。
- 但在目前 mobilenet 這批 collect 分佈上，強制論文 alpha 會退化，故預設維持 hybrid/保守選擇。

### 21.8 Paper-style ALPS Compander（2026-04-13 / updated 2026-04-24）

目前 ALPS 路徑已改成更接近論文的 paper-style compander。

以目前 runtime / calibration 的實作來說：

- `alpha` 欄位實際代表 `theta`
- `beta` 欄位實際代表 `gamma`
- `mode=alps` 時，走：
  - `y = asinh(theta * x) / gamma`
  - `y_q = snap_posit(y)`
  - `x' = sinh(gamma * y_q) / theta`
  - 最終存回 tensor 前，仍會再 snap 一次 posit

也就是目前量測/近似路徑是：

- `x -> y -> y_q -> x' -> final_posit`

用途：把值先搬到壓縮域 `y` 後再做 posit snap，讓低精度 posit 在某些 bucket 上更接近原值。

校正目標（2026-04-24 更新）：

- calibrator 現在優先最小化的是 **`|x_dq - x|`**
- 也就是 paper-style ALPS 會拿
  - `x_dq = sinh(gamma * y_q) / theta`
  與原始 `x` 比較
- 不再把最後 `final_posit = snap_posit(x_dq)` 的誤差當成主要校正目標
- 這樣更接近 int8 QDQ 的語義：
  - `Q` 的輸出看 `q`
  - `DQ` 的輸出看 `dq`
  - 主運算若在浮點域，應優先讓 `dq` 逼近原始 `x`

校正器新增環境變數（`calibrate_qalign_pernode.sh`）：

- `QALIGN_COMPAND_MODE=off|alps`（預設 `alps`）
- `QALIGN_COMPAND_THETA_MIN`（預設 `0.25`）
- `QALIGN_COMPAND_THETA_MAX`（預設 `16`）
- `QALIGN_COMPAND_THETA_STEPS`（預設 `17`，以 log2 網格搜尋 theta）
- `QALIGN_COMPAND_GAMMA_TARGET`（預設 `1.0`；把壓縮後的 `|y|` 對齊到的目標區間）
- `QALIGN_COMPAND_GAMMA_PERCENTILE`（預設 `0.99`；用哪個 percentile 估 `gamma`）
- `QALIGN_COMPAND_MIN_GAIN`（預設 `0`，至少改善多少分數才採用 compander）
- `QALIGN_FORCE_COMPAND=on|off`（預設 `off`；`on` 代表 ALPS-only，不再與 off baseline 比較）
- `QALIGN_CALIB_JOBS`（預設 `0`=自動用 CPU thread 數，做 bucket-level 平行校正）

注意：

- `time_model11_dataset_parallel.sh` 的 `--qalign-mode alps` 會把校正預設切到
  `force_compand=on`。若要保留「ALPS 不好就回退到 off baseline」的行為，
  請明確使用 `--qalign-force-compand off`。
- 2026-04-25 起，若 shell 先 `export QALIGN_FORCE_COMPAND=off|on`，
  `time_model11_dataset_parallel.sh` 也會把它當成預設 override；但命令列
  `--qalign-force-compand ...` 仍有最高優先權。

相容舊參數：

- `QALIGN_COMPAND_BETA_MIN/MAX/STEPS` 仍可用，但現在只是 `THETA_MIN/MAX/STEPS` 的 alias

輸出 CSV 格式更新（相容舊版）：

- 新版：`key,channel,alpha,beta,mode`
- 舊版：`key,channel,alpha` 仍可讀取（beta/mode 會走預設）

目前欄位語意：

- `alpha = theta`
- `beta = gamma`
- `mode = alps|off`

效能更新（2026-04-14）：

- `qalign_pipeline_single.sh` 在 `out-type=p8e*` 時，校正只輸出當前格式，不再每輪重算 `p8e0+p8e1+p8e2`。
- `qalign_calibrate_from_collect.cpp` 新增 bucket-level 多執行緒平行（`QALIGN_CALIB_JOBS`）。
- 若要手動限制校正輸出格式，可用：
  - `calibrate_qalign_pernode.sh --formats p8e0`

Runtime 端也已支援：

- 讀取 `POSIT_QALIGN_FILE` 時，若有第 4/5 欄就套用 `beta/mode`。
- 仍可用環境變數全域覆蓋（沒有 CSV 欄位時）：
  - `POSIT_QALIGN_COMPAND_MODE`（或 `..._P8E0/_P8E1/_P8E2`）
  - `POSIT_QALIGN_COMPAND_BETA`（或 `..._P8E0/_P8E1/_P8E2`）

注意：

- runtime 這裡 `POSIT_QALIGN_COMPAND_BETA=*` 現在語意是 **gamma**
- `theta` 則來自 qalign CSV 的 `alpha` 欄位
- 也就是為了相容現有 pipeline，名字仍是 `alpha/beta`，但 paper-style ALPS 下實際代表的是 `theta/gamma`

快速範例：

```bash
QALIGN_COMPAND_MODE=alps \
QALIGN_COMPAND_THETA_MIN=0.25 \
QALIGN_COMPAND_THETA_MAX=16 \
QALIGN_COMPAND_THETA_STEPS=9 \
QALIGN_COMPAND_GAMMA_TARGET=1.0 \
QALIGN_COMPAND_GAMMA_PERCENTILE=0.99 \
bash ./bash/qalign_pipeline_single.sh \
  --runner ./temp/mobilenet11_new/run_time_sp \
  --so ./temp/mobilenet11_new/mobilenetv2-12-qdq-p8e0.so \
  --txt-dir ./temp/imagenette_val_224 \
  --limit 4 --jobs 2 \
  --shape 1x3x224x224 --out-type p8e0 \
  --collect-csv ./temp/mobilenet11_new/alps_collect.csv \
  --calib-dir ./temp/mobilenet11_new/alps_calib \
  --apply-check on --no-benchmark
```

single-op probe 補充：

- `alps_single_op_probe.cpp` 現在第 3 個參數可用 `auto`，會優先挑第一個 `mode=alps` 的 key。
- 若手動指定的 key 不存在，probe 會列出目前 qalign CSV 裡前幾個可用 key，避免沿用舊 key（例如上一輪 collect/calibrate 的 key）時直接卡住。

範例：

```bash
/tmp/alps_single_op_probe.paper_alps_test \
  ./temp/mobilenet_qdomain/qdq_probe_input_conv0_failfast.csv \
  ./temp/mobilenet11_total_0421/qalign_auto/mobilenetv2-12/p8e0/calib_upto_75/qalign_p8e0.csv \
  auto \
  /tmp/qdq_probe_input_conv0_paper_alps.csv \
  /tmp/qdq_probe_input_conv0_paper_alps.txt
```

雙 reference collect（2026-04-24）：

- runtime collect CSV 現在可寫成：
  - `key,channel,seen_orig,sample_count_orig,orig_samples,seen_dq,sample_count_dq,dq_samples`
- `orig_samples`：
  - 來自 `posit_from_f32` 邊界看到的原始值
- `dq_samples`：
  - 來自 strict QDQ dequantize kernel 算出的 `int8_x_dq`
- calibrator 讀到雙 reference 時，會用：
  - `input = dq_samples`
  - `target = orig_samples`
- 也就是 ALPS / off baseline 都是套在 `int8_x_dq -> posit` 這條 runtime 路徑上，
  但評分時用原始 pre-Q `x` 當目標。
- 舊 collect 格式 `key,channel,seen,sample_count,samples` 仍可讀，會自動退化成單 reference 模式。
- `qalign_calibrate_from_collect.cpp` parser 已補強 `CRLF/空白 trim`。
  - 若 `round_1.log` 出現
    `terminate called after throwing an instance of 'std::invalid_argument'`
    `what(): stod`
    常見原因是 collect CSV 以 `\r\n` 結尾，空 samples 欄位被讀成 `"\r"`。
  - 目前已可容忍這種格式，不需要手動轉檔。

mobilenet non-QDQ f32 shape 回歸修補（2026-04-24）：

- 症狀：
  - `mobilenetv2-12.onnx.mlir -> nqdq-f32.so` 在 optimize 階段報：
    `expected result type with size = dynamic instead of 1280 in dim = 1`
- 直接原因：
  - `ONNX Reshape` lowering 經由 `emitMemRefReinterpretCastOp(...)`
    產生 `memref.reinterpret_cast`
  - 之後又用 `newView.setType(outputType)` 強行把結果型別改成較靜態的
    `memref<?x1280xf32>`
  - 但 op 本身仍攜帶 dynamic size operands，導致 memref verifier 不接受
- 修法：
  - [ONNXToKrnlCommon.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp)
    的 `emitMemRefReinterpretCastOp(...)` 改成：
    - 不再直接 `setType(outputType)`
    - 若 `reinterpret_cast` 結果與 `outputType` 可 `memref.cast`，就插入 `memref.cast`
    - 否則保留原本 verifier-friendly 型別
  - [DialectBuilder.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Dialect/Mlir/DialectBuilder.cpp)
    的 `reinterpretCast(...)` 也只在 size 真正 fold 成 `IntegerAttr` 時，才把對應 output dim 標成靜態
- 驗證：
  - 以下指令已可成功：
    ```bash
    /home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir \
      /home/lai/onnx_mlir/onnx-mlir/src/temp/model/mobilenetv2-12.onnx.mlir \
      -O3 \
      -L /home/lai/onnx_mlir/onnx-mlir/build/Release/lib \
      -o /tmp/mobilenetv2-12-nqdq-f32.recheck
    ```
- 目前剩餘獨立問題：
  - `mobilenetv2-12-qdq.onnx.mlir -> qdq-f32.so` 仍可能報
    `failed to legalize operation 'affine.delinearize_index'`
  - 這是另一個 legalize 問題，與上面的 `dynamic instead of 1280` 已分離。

### 21.9 QDQ 全邊界可校正流程（2026-04-15）

本次更新重點：

- QDQ 邊界不再只在少數入口校正；現在會盡量在每個 QDQ 邊界保留 `qalign_key`。
- 若來源已是 `posit`，維持 passthrough（不額外插 `to_f32 -> from_f32`），避免重複轉換造成額外擾動。
- 常數 DQ 折疊路徑也補上 `qalign_key`，避免全掉到 `key=0`。

目前建議流程（mobilenet）：

```bash
# 一鍵：編譯 + collect/calibrate + 正式 dataset 驗證
QALIGN_FORCE_PAPER_ALPHA=on \
QALIGN_COMPAND_MODE=off \
QALIGN_AUTO_SIGMA=per-format \
QALIGN_SIGMA_MIN=-4 \
QALIGN_SIGMA_MAX=8 \
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_paper_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 1000 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on \
  --qalign-auto on \
  --qalign-formats p8e0,p8e1,p8e2
```

```bash
# 二段式（先收集/校正，再固定 CSV 做正式跑）
# 1) 先跑 collect+calibrate
QALIGN_FORCE_PAPER_ALPHA=on \
QALIGN_COMPAND_MODE=off \
QALIGN_AUTO_SIGMA=per-format \
QALIGN_SIGMA_MIN=-4 \
QALIGN_SIGMA_MAX=8 \
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_paper_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 256 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on \
  --qalign-auto on \
  --qalign-formats p8e0,p8e1,p8e2

# 2) 固定使用已產生的 qalign CSV 跑正式統計
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_paper_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 1000 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on \
  --qalign-auto off \
  --qalign-csv-dir ./temp/mobilenet11_paper_new/qalign_auto/mobilenetv2-12/p8e0/calib_upto_96
```

補充：

- 若想連 `key=0` 也納入校正，可加：`QALIGN_SKIP_KEY0=off`。
- 若要回到 strict QDQ 語義再做 A/B，比較時用 build script 加 `--strict-qdq-mode`。

### 21.10 Mixed Precision（只開混合精度，不用 alpha/alps）

新增 runtime 開關：

- `POSIT_MIXED_PRECISION_P16=on|off`（預設 `off`）
  - `on`：在 `conv2d/gemm` 的 dot-product 中，對低位寬格式（`p4~p9`）改用 **p16e1 domain** 做乘加，再一次量化回原目標格式。
  - `off`：維持原本路徑（quire / posit-domain fallback）。
- `POSIT_MIXED_PRECISION_P16_OPS=conv2d,gemm`（預設兩者都開）
  - 可只開一種，例如 `conv2d` 或 `gemm`。

這個開關與 qalign 獨立，可直接做「混合精度 only」測試。

範例（關掉 alpha/alps，只開 mixed precision）：

```bash
POSIT_MIXED_PRECISION_P16=on \
POSIT_MIXED_PRECISION_P16_OPS=conv2d,gemm \
POSIT_QALIGN_FILE= \
POSIT_QALIGN_ALPHA=1 \
POSIT_QALIGN_COMPAND_MODE=off \
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_alps_new \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 1000 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on \
  --qalign-auto off
```

### 21.11 QDQ 對齊 f32 計算路徑（conv2d/gemm）

新增 runtime 開關（`posit_runtime.cpp`）：

- `POSIT_QDQ_F32_MATH=on|off`（預設 `off`）
  - `on`：`conv2d/gemm` 的 dot + epilogue 用 f32 域計算，再一次量化回目標 posit 格式。
  - `off`：維持原本 posit/quire（或 mixed-p16）路徑。
- `POSIT_QDQ_F32_MATH_OPS=conv2d,gemm`（預設兩者都開）
  - 可只開其一（例如只開 `conv2d`）。

A/B 測試（同一組 `.so`、同一批資料，只改環境變數）：

```bash
# A: 原路徑
POSIT_QDQ_F32_MATH=off \
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_total \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 1000 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on --qalign-auto off

# B: QDQ 對齊 f32 計算
POSIT_QDQ_F32_MATH=on \
POSIT_QDQ_F32_MATH_OPS=conv2d,gemm \
bash ./bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_total \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 25 --limit 1000 --warmup 0 --iters 1 \
  --no-benchmark --progress 1 --task-progress on --qalign-auto off
```

比較方式：

- 精度：看 `Top1 / Top5 / MAE`（`*.dataset_parallel.log` 與 `*.format_summary.B.log`）。
- 速度：看 `avg latency (us)` 與整體 wall time。

## 22) GPT-2 (onnx-community / HuggingFace) 現況

### 22.1 目前使用的 GPT-2 模型位置

```text
/home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model.onnx
/home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model_int8.onnx
/home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model_quantized.onnx
```

說明：

- `model.onnx`：non-QDQ / f32 ONNX
- `model_int8.onnx`：QDQ INT8 ONNX（目前 `11` 流程使用這份）
- `model_quantized.onnx`：同目錄保留，但目前 `11` 流程不是用它

### 22.2 GPT-2 11-build wrapper

已新增：

```text
/home/lai/onnx_mlir/onnx-mlir/src/bash/build_gpt2_hf_11_sos.sh
```

用途：

- 包住 `build_model11_sos.sh`
- 預設吃：
  - `model_int8.onnx` 當 QDQ 輸入
  - `model.onnx` 當 NQDQ 輸入
- 預設輸出到：
  - `/home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11`

最短可用編譯指令：

```bash
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_gpt2_hf_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11 \
  --posit-formats p8e0 \
  --strict-qdq-mode
```

若要保留 IR / stage logs：

```bash
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_gpt2_hf_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11 \
  --posit-formats p8e0 \
  --strict-qdq-mode \
  --keep-ir \
  --ir-dir /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11/ir_debug \
  --keep-stage-logs
```

### 22.3 這次 GPT-2 `.so` 產出成功時實際修到的地方

這批修改的目標是讓 GPT-2 這種 dynamic-shape / reshape-heavy 模型可以一路 lower 到 `.so`，主要是解掉：

- unranked / not-ranked memref 在 elementwise 與 reshape lowering 期間造成的 crash
- `onnx.SplitV13` 沒被 lower 導致 conversion 卡住
- `UnrealizedConversionCastOp` 殘留到 LLVM translation，讓 `mlir-translate` 失敗

關鍵修改檔案：

- `[Elementwise.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/Math/Elementwise.cpp:108)`
  - 新增 `getRankedOutputMemRefType(...)`
  - 讓 unary / binary / variadic / where elementwise lowering 遇到 unranked tensor/memref 時，先 materialize 成可用的 ranked memref
- `[Reshape.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Dialect/ONNX/ONNXOps/Tensor/Reshape.cpp:34)`
  - 修 ONNX Reshape shape helper，避免直接假設 input 一定有 rank
  - `0` / `-1` 的 reshape 語義改成保守且安全的處理
- `[Split.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/Tensor/Split.cpp:120)`
  - 補 `ONNXSplitV13Op` lowering
- `[ONNXToKrnlCommon.hpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp:490)`
  - 宣告 `populateLoweringONNXSplitV13OpPattern(...)`
- `[ConvertONNXToKrnl.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp:264)`
  - 註冊 `SplitV13` lowering pattern
- `[DialectBuilder.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Dialect/Mlir/DialectBuilder.cpp:1575)`
  - flatten 路徑改優先走 `reinterpretCast`，避免 dynamic model 殘留 problematic reshape / cast chain
- `[ONNXToKrnlCommon.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp:34)`
  - `OnnxToKrnlBuilder::reshape(...)` 優先使用 memref reinterpret-cast
- `[ONNXToKrnlCommon.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp:575)`
  - `TensorType -> MemRef` conversion 改得更安全，可處理 unranked tensor
- `[ONNXToKrnlCommon.cpp](/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp:612)`
  - source / target materialization 遇到 memref-compatible type 時，優先用 `memref.cast`
  - 這是解掉 `mlir-translate` 被 `unrealized_conversion_cast` 卡住的關鍵之一

### 22.4 目前已成功產出的 GPT-2 `.so`

資料夾：

```text
/home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11
```

目前可看到：

- `gpt2-hf-debug-qdq-f32.so`
- `gpt2-hf-debug-nqdq-f32.so`
- `gpt2-hf-debug-qdq-p8e0.so`
- `run_time_sp`

### 22.5 GPT-2 smoke runner（統一 f32 / posit）

已更新：

```text
/home/lai/onnx_mlir/ImageNet100/gpt2_posit_smoke.py
```

現在這支 runner 會自動判斷：

- 若 `.so` 有 `_mlir_ciface_main_graph`
  - 走 posit 的 memref ciface 路徑
- 若 `.so` 有 `run_main_graph`
  - 走 OMTensor / OMTensorList 路徑

另外已新增 bash wrapper：

```text
/home/lai/onnx_mlir/ImageNet100/run_gpt2_hf_smoke.sh
```

## 23) ImageNet100 ResNet18（PyTorch / torchvision export）Posit bugfix 記錄

說明：

- 舊的 modelzoo / 早期 ResNet18 筆記已移除。
- 目前 ResNet18 的 canonical source 改成：
  - `/home/lai/onnx_mlir/ImageNet100/export_resnet18_torchvision_onnx.py`
  - 匯出的 fp32 ONNX：`/home/lai/onnx_mlir/ImageNet100/model/imagenet100_resnet18.onnx`
  - 對應的 int8 QDQ：`/home/lai/onnx_mlir/ImageNet100/model/imagenet100_resnet18-int8-qdq.onnx`
- 這組模型 metadata 與目前 ResNet50 比較接近：
  - `producer_name = pytorch`
  - `producer_version = 2.11.0+cpu`
  - `graph_name = main_graph`
  - `opset = 20`

### 23.1 QDQ 路線：`EmitONNXBasic` import 會在 `posit/onnx -> krnl` 階段 segfault

錯誤現象：

```text
Segmentation fault (core dumped)
... onnx-mlir-opt <model>.posit.mlir
    --convert-posit-to-krnl
    --canonicalize
    --convert-onnx-to-krnl
    --convert-posit-to-krnl
    --canonicalize
```

重現條件：

- `build_model11_sos.sh`
- `--model-name imagenet100_resnet18`
- `--posit-source qdq`
- import mode 為 `basic`

根因：

- 新的 PyTorch-exported ResNet18，在 QDQ 路線下若先用 `--EmitONNXBasic` 匯入，後續 mixed Posit/ONNX lowering 會少掉部分對 QDQ graph 很重要的 ONNXIR 細節。
- 直接結果是：
  - `convert-onnx-to-posit` 可以過，
  - 但第二段 `convert-posit-to-krnl + convert-onnx-to-krnl` 會在 `onnx-mlir-opt` 內部 segfault。
- 同一顆模型若改成 `--EmitONNXIR` 匯入，再走相同 lowering 流程，`qdq-p8e0.so` 可以成功 build 並成功執行。

修法：

- 將 `build_model11_sos.sh` 的 ONNX import 預設由：

```text
--EmitONNXBasic
```

  改成：

```text
--EmitONNXIR
```

- 保留 `--use-onnx-ir` 選項相容，但現在不再需要額外顯式開啟。

關鍵檔案：

- [build_model11_sos.sh](/home/lai/onnx_mlir/onnx-mlir/src/bash/build_model11_sos.sh)

驗證結果：

- `qdq p8e0` smoke build 成功：
  - `imagenet100_resnet18-qdq-p8e0.so`
- `run_time_sp` smoke 也成功：
  - `MAIN type=f32 avg=3.50775e+06 us (iters=1, warmup=0)`

代表性意義：

- 這是 **import form（Basic vs ONNXIR）會直接影響後續 mixed lowering 穩定性** 的案例。
- 問題不在 QDQ 演算法本身，而在 import 之後提供給 lowering 的 graph 細節是否足夠完整。

### 23.2 NQDQ 路線：runtime 選錯 entrypoint，導致先前 crash / 表面上像卡住

錯誤現象：

- `nqdq p8e0` 可以順利 build 出 `.so`
- 但 `run_time_sp` 執行時：
  - 舊版會在錯誤 entrypoint 附近 crash
  - 修到一半後會看起來像「卡住」，其實是跑進了錯誤 wrapper / 不相容入口

根因：

- `run_time_sp` 會用 `dlsym` 解析模型入口。
- 新的 ResNet18 `.so` 同時匯出：
  - `run_main_graph`
  - `run_main_graph_imagenet100_resnet18-nqdq-import`
  - `_mlir_ciface_main_graph_imagenet100_resnet18-nqdq-import`
  - `omQueryEntryPoints`
- 舊邏輯在查詢 `omQueryEntryPoints()` 後，太早接受裸的 `run_main_graph`，沒有優先使用對應的 `_mlir_ciface_main_graph_*`。
- 結果是 `run_time_sp` 用錯 ABI 風格呼叫模型入口，導致 crash，或讓使用者誤以為模型卡住。

修法：

- 在 `run_time.cpp` 的 `queryEntrypointCandidates(...)` 中：
  - 先根據 `omQueryEntryPoints()` 返回的 `run_main_graph*` / `main_graph*` 名稱，
  - 優先推導對應的 `_mlir_ciface_main_graph*`
  - 最後才把原始的 `run_main_graph*` 名稱當 fallback。

關鍵檔案：

- [run_time.cpp](/home/lai/onnx_mlir/onnx-mlir/src/run_time.cpp)

驗證結果：

- 目前 `run_time_sp` 會明確印出：

```text
CONFIG main_entry_resolved=_mlir_ciface_main_graph_imagenet100_resnet18-nqdq-import
```

- `nqdq p8e0` smoke runtime 成功：
  - `MAIN type=f32 avg=4.16687e+07 us`
  - `top1=417 top5=[417,418,446,794,645]`

補充：

- 這顆 ResNet18 的 posit runtime 很慢，若不手動指定：
  - `--warmup 0`
  - `--iters 1`
  表面上會像「卡住」，其實是在跑預設的較大量 benchmark 迴圈。

代表性意義：

- 這是 **entrypoint symbol name 正確，但 ABI 風格仍可能選錯** 的典型 runtime bug。
- 也說明：
  - `.so` 能成功 link，
  - 不代表 host runner 一定會用對入口來呼叫它。

最短可用執行指令：

```bash
bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_hf_smoke.sh \
  /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11/gpt2-hf-debug-qdq-f32.so \
  15496
```

```bash
bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_hf_smoke.sh \
  /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11/gpt2-hf-debug-nqdq-f32.so \
  15496
```

### 22.6 目前實際 smoke 結果

已實測可跑：

- `gpt2-hf-debug-qdq-f32.so`
  - `mode=omtensor-run_main_graph`
  - `logits_shape=(1, 1, 50257)`
  - `top1=11`
- `gpt2-hf-debug-nqdq-f32.so`
  - `mode=omtensor-run_main_graph`
  - `logits_shape=(1, 1, 50257)`
  - `top1=11`

目前仍有 runtime 問題：

- `gpt2-hf-debug-qdq-p8e0.so`
  - 可以成功編出 `.so`
  - 但第一 token smoke 目前仍會在 runtime 端碰到：

```text
[PFALLBACK] kind=memref_guard count=1 detail=negative size
```

也就是說：

- **compile path 已修通**
- **f32 baseline 執行已可用**
- **posit GPT-2 runtime 還需要再追一層 descriptor / memref guard 問題**

### 22.7 若要直接用完整 generic 指令重編 GPT-2

這是目前已成功產出 `.so` 的等價 generic 寫法：

```bash
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_model11_sos.sh \
  --model-name gpt2-hf-debug \
  --qdq-onnx /home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model_int8.onnx \
  --nqdq-onnx /home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model.onnx \
  --out-dir /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11 \
  --posit-formats p8e0 \
  --strict-qdq-mode
```

### 22.8 GPT-2 文字輸入 / 文字輸出測試

已新增：

```text
/home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.py
```

用途：

- 可直接輸入 `prompt`，輸出生成文字
- 可對一段文字做 teacher-forcing score
- 支援：
  - `.so`（目前以 `run_main_graph` / OMTensor 形式的 GPT-2 f32 `.so` 為主）
  - `.onnx`（可直接走 ONNX Runtime）

注意：

- 這支 script 內建的是 **ASCII / English 友善的 GPT-2 BPE fallback tokenizer**
- 適合目前 GPT-2 英文 prompt smoke / score
- 若之後環境有 `transformers + tokenizers + regex`，仍可再換成官方 tokenizer 做更完整一致性驗證
- `gpt2-hf-debug-qdq-p8e0.so` 目前仍可能因既有 runtime 問題失敗；建議先拿：
  - `gpt2-hf-debug-qdq-f32.so`
  - `gpt2-hf-debug-nqdq-f32.so`
  - `model.onnx`
  - `model_int8.onnx`
  做文字測試

文字生成範例：

```bash
bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.sh \
  --model /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11/gpt2-hf-debug-qdq-f32.so \
  --mode generate \
  --prompt "Hello, my name is" \
  --max-new-tokens 16
```

```bash
bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.sh \
  --model /home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community/onnx/model_int8.onnx \
  --mode generate \
  --prompt "Hello, my name is" \
  --max-new-tokens 16
```

paper-like score / perplexity-like 範例：

```bash
bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.sh \
  --model /home/lai/onnx_mlir/ImageNet100/build_gpt2_hf_posit11/gpt2-hf-debug-qdq-f32.so \
  --mode score \
  --text "The capital of Japan is Tokyo."
```

目前已實測：

- `gpt2-hf-debug-qdq-f32.so`
- `model_int8.onnx`

在 prompt `"Hello, my name is"` 下，兩者都生成：

```text
Hello, my name is John. I'm a man of many names. I'm
```

### 22.8.1 GPT-2 專用虛擬環境 python

GPT-2 相關 bash wrapper 現在預設會使用：

```text
/home/lai/onnx_mlir/ImageNet100/gpt2/bin/python
```

已套用到：

- `/home/lai/onnx_mlir/ImageNet100/run_gpt2_hf_smoke.sh`
- `/home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.sh`

若之後要換成別的虛擬環境，不用改 script，本次執行前覆寫即可：

```bash
GPT2_VENV_PYTHON=/path/to/venv/bin/python bash /home/lai/onnx_mlir/ImageNet100/run_gpt2_text_eval.sh ...
```

備註：

- 目前 `ImageNet100/gpt2` 已有 `transformers / tokenizers / regex`
- 若要跑 `.onnx` 路線，這個 venv 還需要有 `onnxruntime`
- `resnet / mobilenet` 流程不受影響，仍維持原本直接執行

### 22.9 LLM 量化論文常見測法（目前先仿照的高層版本）

參考方向：

- GPTQ（arXiv:2210.17323）
- SmoothQuant（arXiv:2211.10438）
- LLM.int8()（arXiv:2208.07339）
- AWQ（arXiv:2306.00978）

這類工作常見不是只看「生成文字好不好看」，而是分成幾層：

- **Language modeling**
  - 常看 `perplexity (PPL)`
  - 常見資料集：WikiText-2、C4、PTB、LAMBADA
- **Zero-shot downstream tasks**
  - 常見：PIQA、HellaSwag、BoolQ、WinoGrande、ARC
- **Efficiency**
  - latency / throughput / memory

目前我們先在本地能直接落地的版本是：

- `generate`
  - 看 prompt -> generated text
- `score`
  - 看 teacher-forcing 下的 `avg_nll / ppl / next_token_top1_acc`

這比單看最後生成句子更穩，因為量化模型常常只差一點 logits，最後整句就會分岔。

### 22.10 QAlign bucket probe（先看前幾個 off/alps bucket）

新增：

- `/home/lai/onnx_mlir/onnx-mlir/src/temp/probe/qalign_bucket_probe.cpp`
- `/home/lai/onnx_mlir/onnx-mlir/src/bash/probe_qalign_buckets.sh`

用途：

- 直接讀：
  - `collect_upto_*.csv`
  - `qalign_p8e*.csv`
  - `qalign_p8e*_detail.csv`
- 先把前幾個 bucket 的樣本展開成逐點表
- 適合先看：
  - `mode=off` 的 bucket
  - 或少數 `mode=alps` bucket

目前 wrapper 預設路徑直接對：

- `src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/...`

預設跑法：

```bash
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/probe_qalign_buckets.sh
```

預設輸出：

- `/home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/probe_first_off_buckets.csv`
- `/home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/probe_first_off_buckets.txt`

可切成看 `alps` bucket：

```bash
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/probe_qalign_buckets.sh \
  --mode alps \
  --format p8e2 \
  --collect-csv /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e2/collect_upto_75.csv \
  --qalign-csv /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e2/calib_upto_75/qalign_p8e2.csv \
  --detail-csv /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e2/calib_upto_75/qalign_p8e2_detail.csv \
  --out-prefix /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e2/probe_first_alps_buckets
```

目前 probe 會輸出：

- `int8_x_dq`
- `runtime_scaled`
- `runtime_y_q`
- `runtime_x_dq`
- `runtime_final`

備註：

- 如果 collect CSV 是舊格式
  - header: `key,channel,seen,sample_count,samples`
  - 那它只有單一路樣本，probe 目前把它當成 `int8_x_dq`
  - 所以 `orig_x` 會是空的
- 也就是說：
  - 這個 probe 適合先看「runtime 這組 alpha/theta 怎麼把 bucket sample 映到最後數值」
  - 但**不能**單靠這個檔還原每個點的 `scale/zp`

若要直接看某個已知 QDQ 節點的 `x / q / x_dq / scale / zp`：

- 看現有專用 probe：
  - `/home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet_qdomain/qdq_probe_input_conv0.txt`
  - `/home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_paper_alps_probe/qdq_probe_input_conv0_paper_alps.csv`

目前 `qalign_bucket_probe.cpp` 採用的是 **runtime 現行公式**，不是只照 paper 註解抄公式：

- `mode=off`
  - 近似看成：`final = cast_p8( cast_p8(alpha * x_dq) / alpha )`
- `mode=alps`
  - 依現行 runtime：
    - `scaled = alpha * x_dq`
    - `y_q = cast_p8( asinh(alpha * scaled) / beta )`
    - `runtime_x_dq = sinh(beta * y_q) / alpha`
    - `final = cast_p8(runtime_x_dq)`

這樣可以直接對照目前 runtime 真正在做的事，而不是只對 paper-style 名義公式。

### 22.10.1 theta/gamma naming + runtime probe

之後新生成的校正輸出欄位名稱改成：

- `theta`（舊名 `alpha`）
- `gamma`（舊名 `beta`）

目前已更新：

- `qalign_p8e*.csv`
  - header 會是：`# key,channel,theta,gamma,mode`
- `qalign_p8e*_detail.csv`
  - 主要欄位會是：
    - `theta`
    - `gamma`
    - `theta_legacy`
    - `theta_paper`

相容性：

- runtime / probe parser 仍接受舊欄名 `alpha/beta`
- 所以舊的 calibration 輸出檔仍可繼續使用

runtime 內建 probe：

- `posit_runtime.cpp` 現在新增了 qalign runtime sample probe
- 目的是下次直接跑 dataset pipeline 時，順手輸出類似
  - `qdq_probe_input_conv0_paper_alps.csv`
  的小型抽樣檔

啟用方式：

```bash
QALIGN_RUNTIME_PROBE=on \
QALIGN_RUNTIME_PROBE_LIMIT=8 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

若只想看某些 bucket / op key：

```bash
QALIGN_RUNTIME_PROBE=on \
QALIGN_RUNTIME_PROBE_KEYS=1684092134847042904,454808415879048355 \
QALIGN_RUNTIME_PROBE_LIMIT=8 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

可再指定只看哪一路：

- `QALIGN_RUNTIME_PROBE_SOURCE=orig`
- `QALIGN_RUNTIME_PROBE_SOURCE=dq`
- 不設則兩邊都可記

目前 `time_model11_dataset_parallel.sh` 會在 `QALIGN_RUNTIME_PROBE=on` 時，自動為每個 `.so` 設：

- `POSIT_QALIGN_PROBE_FILE=<out_dir>/<model>-<suffix>.qalign_runtime_probe.csv`

runtime probe CSV 欄位：

- `source_kind`
- `format`
- `key`
- `channel`
- `sample_index`
- `theta`
- `gamma`
- `mode`
- `orig_x`
- `int8_x_dq`
- `q_value`
- `scale`
- `zp`
- `runtime_scaled`
- `runtime_y_q`
- `runtime_x_dq`
- `runtime_final`

22.10.2 ALPS theta^2 bug fix

問題：

- `qalign_calibrate_from_collect.cpp`
- `posit_runtime.cpp`

在 ALPS 路徑都曾經先做 `theta * x`，再呼叫 `asinh(theta * input)`，等效變成：

- `y = asinh(theta^2 * x) / gamma`

這會讓 `theta != 1` 時誤差明顯放大，等於 calibration 與 runtime 都沒有真正實作 paper ALPS 的公式。

本次修正後，兩邊都改成一致的 paper 版本：

- `y = asinh(theta * x) / gamma`
- `y_q = cast_p8(y)`
- `x_dq = sinh(gamma * y_q) / theta`

修正點：

- `src/temp/probe/qalign_calibrate_from_collect.cpp`
  - `alignedDQVal(...)` 的 ALPS 路徑改為 `compandAlps(x, theta) / gamma`
  - 不再把已經乘過 `theta` 的值再次送進 `asinh(theta * ...)`
- `src/posit_runtime.cpp`
  - `applyQAlignScaleSnap(...)` 的 ALPS 路徑同樣改為 `qalignCompandAlps(v, theta) / gamma`
  - runtime probe 仍保留 `runtime_scaled = theta * x`，方便對照外層縮放量，但 compand 本身不再重複乘一次 `theta`

影響：

- 之後重新 calibration 時，`theta` 掃描會真正反映 ALPS 的設計意義
- 之後重新 build 出來的新 `.so`，runtime 的 ALPS 行為會與 calibration 一致
- 舊的 `.so` 不會自動帶到這次修正，必須重編

注意：

- 目前 runtime 端知道的是 `qalign key`
- 還**不知道原始 ONNX node name**
- 所以這版 probe 的「某些 op」是以 `key` 為選擇單位
- 若之後需要 `key -> onnx_node_name` 對照，還要再補 compile/lowering 端的 mapping 輸出

22.10.3 CSV-only runtime + gamma grid search

runtime：

- `posit_runtime.cpp` 的 qalign 參數解析改成 **CSV-only**
- 不再使用 `POSIT_QALIGN_THETA / ALPHA / GAMMA / COMPAND_MODE` 這類 global fallback
- 若 bucket 沒有對應的 CSV entry，runtime 直接退回 identity/off：
  - `theta = 1`
  - `gamma = 0`
  - `mode = off`
- 目的是避免 miss bucket 時，意外吃到 `ALPS theta=1 gamma=1` 之類的全域設定

dataset runner：

- `time_model11_dataset_parallel.sh` 不再主動傳 `POSIT_QALIGN_COMPAND_MODE`
- runtime 套用 qalign 時只信 `POSIT_QALIGN_FILE`

calibration：

- `qalign_calibrate_from_collect.cpp` 新增：
  - `QALIGN_COMPAND_GAMMA_TARGET_LIST`
  - `QALIGN_COMPAND_GAMMA_PERCENTILE_LIST`
  - 以及無 prefix 的 alias：
    - `GAMMA_TARGET_LIST`
    - `GAMMA_PERCENTILE_LIST`
- ALPS 搜尋不再只用單一 `gamma_target / percentile`
- 現在會對每個 `theta` 做 grid search，選出該 bucket 最佳的：
  - `theta`
  - `gamma`
  - `gamma_target`
  - `gamma_percentile`

score：

- 原本 score 主要是 `MAE + weighted-MAE`
- 現在新增 paper-inspired downstream proxy：
  - `pm_proxy = noise_power / signal_power`
  - 可視為 inverse-SQNR / Eq.(12) 方向的 proxy
- 新增 env：
  - `QALIGN_SCORE_DOWNSTREAM_WEIGHT`
- detail CSV 新增欄位：
  - `pm_proxy_base`
  - `pm_proxy_aligned`
  - `pm_proxy_improve`
  - `gamma_target_used`
  - `gamma_percentile_used`

限制：

- 目前這條主線仍是「standard posit runtime + ALPS compander/qalign」
- 還**不是**完整的 generalized posit ALPS runtime
- 論文中的 `rs/sc` 若要真的進 runtime/MAC，還需要額外實作 generalized posit 的編碼/解碼/運算路徑，不能只靠目前的 qalign CSV 直接等價取代

22.10.4 Exploratory rs/sc outputs

目的：

- 先把 ALPS 論文裡的 `rs/sc` 選擇邏輯接進 calibration 分析輸出
- 讓後續可以先觀察每個 bucket / op 傾向的 generalized posit 參數
- **目前不改 runtime numerical path**

新增 detail CSV 欄位：

- `gp_rs_local`
- `gp_sc_local`
- `gp_rs_from_ref`
- `gp_sc_from_ref`
- `gp_ref_key`
- `gp_ref_channel`
- `gp_kurtosis_target`
- `gp_kurtosis_gap`
- `gp_mean_abs_target`
- `gp_mean_abs_gap`

意義：

- `gp_rs_local / gp_sc_local`
  - 直接根據目前 bucket 的樣本分布估一組 paper-inspired generalized posit 參數
  - `rs` 用樣本 excess kurtosis 與近似 generalized posit value-set 的 kurtosis 做匹配
  - `sc` 用樣本 mean-abs 與近似 generalized posit value-set 的 mean-abs 做匹配
- `gp_rs_from_ref / gp_sc_from_ref`
  - 以 reference bucket 為基準，照論文 Eq.(13)-(16) 的 error-gain 比例做 propagation
  - 目前 `E` 的 proxy 使用 `pm_proxy_aligned`

reference bucket：

- 若有設：
  - `QALIGN_GP_REFERENCE_KEY`
  - `QALIGN_GP_REFERENCE_CHANNEL`
  則用指定 bucket
- 否則預設選 sample count 最大的 bucket

重要限制：

- 目前 generalized posit value-set 是用 **approximate decoder** 估計
- 這能先提供 `rs/sc` 的觀察方向，但**不等於**完整論文中的 hardware-accurate generalized posit implementation
- 若之後要正式採用 `rs/sc`，下一階段要把：
  - encode/decode
  - quantize path
  - MAC/quire
  全部改成 generalized posit 版本

22.10.5 Experimental generalized posit runtime branch

目的：

- 直接在 `posit_runtime.cpp` 裡加一條 **experimental generalized posit** numerical path
- 讓同一個 `.so` 可以用環境變數決定：
  - 哪些格式維持標準 posit
  - 哪些格式改走 approximate generalized posit
- 目前建議只先測：
  - `p8e0`
  - `p8e1`

啟用方式：

- `POSIT_GP_EXPERIMENTAL_FORMATS`
  - 逗號分隔的格式清單
  - 例如：
    - `p8e0`
    - `p8e0,p8e1`
- 相容 alias：
  - `POSIT_GP_FORMATS`

可控制參數：

- 全域預設：
  - `POSIT_GP_RS`
  - `POSIT_GP_SC`
- 8-bit 共用預設：
  - `POSIT_GP_RS_P8`
  - `POSIT_GP_SC_P8`
- 各格式覆蓋：
  - `POSIT_GP_RS_P8E0`
  - `POSIT_GP_SC_P8E0`
  - `POSIT_GP_RS_P8E1`
  - `POSIT_GP_SC_P8E1`
  - `POSIT_GP_RS_P8E2`
  - `POSIT_GP_SC_P8E2`
- 自動抓取來源偏好：
  - `POSIT_GP_AUTO_SOURCE=from_ref`
  - 或 `POSIT_GP_AUTO_SOURCE=local`

優先順序：

- `POSIT_GP_RS_P8E* / POSIT_GP_SC_P8E*`
- 然後 `POSIT_GP_RS_P8 / POSIT_GP_SC_P8`
- 最後 `POSIT_GP_RS / POSIT_GP_SC`
- 如果以上都**沒有設定**，則會嘗試從 `POSIT_QALIGN_FILE` 同目錄下的
  `qalign_<format>_detail.csv` 自動抓一組 `rs/sc`
  - 預設優先 `gp_rs_from_ref / gp_sc_from_ref`
  - 若找不到，退回 `gp_rs_local / gp_sc_local`
  - 選法是以 `sample_count` 加權後的多數組合
  - 若 detail CSV 不存在、是空的、或沒有 `gp_rs/gp_sc` 欄位，則退回標準預設：
    - `rs = 7`
    - `sc = 0`

目前 runtime 行為：

- 對啟用 experimental GP 的 `p8e0/p8e1/p8e2`
  - `fromDouble` 會量化到 approximate generalized posit value-set
  - `toDouble` 會用 approximate generalized posit decoder 反解
  - `add/sub/mul/div` 會走：
    - double-domain arithmetic
    - 再 requantize 回 generalized posit value-set
- 這表示：
  - 這是一條 **真的會影響數值結果** 的 runtime branch
  - 不是只有 analysis / CSV 顯示

目前刻意做的保守處理：

- 若某個 p8 format 啟用 experimental GP，對應的 dot accumulator 會**停用 quire**
- 原因是目前 quire 還是標準 posit 語義
- 若直接和 generalized posit value-set 混用，語義不一致，容易導致結果更難解釋

runtime probe：

- `POSIT_QALIGN_PROBE_FILE` 產生的 probe CSV 目前會多三個欄位：
  - `gp_enabled`
  - `gp_rs`
  - `gp_sc`
- 可以直接確認：
  - 這次 run 是否真的啟用 generalized posit branch
  - 該格式實際吃到哪組 `rs/sc`

使用範例：

只指定格式，讓 runtime 自動抓 `rs/sc`：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

只對 `p8e0` 啟用：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
POSIT_GP_RS_P8E0=7 \
POSIT_GP_SC_P8E0=-3 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

同時對 `p8e0,p8e1` 啟用：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1 \
POSIT_GP_RS_P8E0=7 \
POSIT_GP_SC_P8E0=-3 \
POSIT_GP_RS_P8E1=7 \
POSIT_GP_SC_P8E1=-3 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

和 qalign runtime probe 一起開：

```bash
QALIGN_RUNTIME_PROBE=on \
QALIGN_RUNTIME_PROBE_LIMIT=8 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

注意：

- 若你想走「只輸入 `POSIT_GP_EXPERIMENTAL_FORMATS=p8e0`」的模式，
  就要確保這次使用的 final `qalign_*_detail.csv` 是**新版**，裡面有：
  - `gp_rs_from_ref`
  - `gp_sc_from_ref`
  - 或 `gp_rs_local`
  - `gp_sc_local`
- 像較早期的 `mobilenet11_dualref_0426` 那批 detail CSV 還沒有這些欄位，
  那 runtime 會安全退回 `rs=7, sc=0`

重要限制：

- 這條 branch 目前是 **approximate generalized posit runtime**
- 還**不是**完整論文版本的 generalized posit
- 目前尚未改到：
  - 真正的 generalized posit bit-level encode/decode 規格驗證
  - generalized posit quire / MAC
  - lowering 端依 `rs/sc` 生成不同數值 kernel
- 所以它適合拿來做：
  - `p8e0/p8e1` 的先行 A/B
  - 觀察 `rs/sc` 是否有潛力改善 accuracy
- 但不應直接當成最終 paper-faithful implementation

22.10.6 Per-layer gp metadata + Algorithm-1-aligned sc options

這次把 `rs/sc` 的 calibration 輸出從「只有 per-format 摘要」往前推成：

- 每個 qalign bucket / layer 都會有自己的 `gp_rs/gp_sc` 建議
- 並且直接寫進最終的：
  - `qalign_p8e0.csv`
  - `qalign_p8e1.csv`
  - `qalign_p8e2.csv`

新版 qalign CSV 欄位：

```text
# key,channel,theta,gamma,mode,gp_rs,gp_sc,gp_source
```

其中：

- `gp_rs`
- `gp_sc`

是這個 bucket 最終建議要套用的 generalized posit 參數

- `gp_source`
  - `from_ref`
  - 或 `local`

控制來源：

- `QALIGN_GP_APPLY_SOURCE=from_ref`
  - 預設，較接近 paper 的 layer-to-layer propagation 想法
- `QALIGN_GP_APPLY_SOURCE=local`
  - 每個 bucket 直接用自己的 local 建議

Algorithm 1 的 `sc` metric：

- 論文 Algorithm 1 是用 `mean(Wr)`，不是 `mean(abs(Wr))`
- 這次 calibration 支援三種模式：
  - `QALIGN_GP_SC_METRIC=paper_mean`
    - 直接用 `mean(Wr)`，最接近論文
  - `QALIGN_GP_SC_METRIC=meanabs`
    - 用 `mean(abs(Wr))`
  - `QALIGN_GP_SC_METRIC=compare_best`
    - 兩種都算，挑 gap 較小的

detail CSV 會同時保留：

- `gp_sc_local_mean`
- `gp_sc_local_meanabs`
- `gp_sc_metric_used`

所以可以直接比較：

- paper mean 選到的 `sc`
- mean-abs 選到的 `sc`
- 這次最終採用的是哪一條

新的 generalized posit detail 欄位重點：

- `gp_rs_local`
- `gp_sc_local`
- `gp_sc_local_mean`
- `gp_sc_local_meanabs`
- `gp_rs_from_ref`
- `gp_sc_from_ref`
- `gp_rs_applied`
- `gp_sc_applied`
- `gp_sc_metric_used`
- `gp_apply_source`
- `gp_err_gain_mode`
- `gp_mean_target`
- `gp_mean_gap`
- `gp_mean_abs_target`
- `gp_mean_abs_gap`
- `gp_err_gain_pm_proxy`
- `gp_err_gain_eq22`

kurtosis 的穩定化：

- `QALIGN_GP_RS_MIN`
- `QALIGN_GP_RS_MAX`
- `QALIGN_GP_SC_MIN`
- `QALIGN_GP_SC_MAX`
- `QALIGN_GP_KURTOSIS_WINSOR_PCT`

說明：

- `QALIGN_GP_KURTOSIS_WINSOR_PCT`
  - 會先對樣本做 winsorization，再算 excess kurtosis
  - 用來降低 sample 數少或 outlier 很強時的敏感度
- 若不想 winsorize，設 `0`

`Ewl / EAl` 代理值：

- 先前 `gpRsFromRef / gpScFromRef` 是拿 `pm_proxy_aligned` 當 error-gain proxy
- 這次新增：
  - `QALIGN_GP_ERR_GAIN_MODE=eq22`
  - `QALIGN_GP_ERR_GAIN_MODE=pm_proxy`

其中：

- `eq22`
  - 預設
  - 用 generalized posit quantization 的實際模擬結果估：
    - `|eps_p| / |eps_fx|`
  - 比直接拿 `pm_proxy_aligned` 更接近 Appendix Eq.(22) 的 error-gain 形式
- `pm_proxy`
  - 保留舊路徑做比較

runtime 自動抓取：

- 如果只設：
  - `POSIT_GP_EXPERIMENTAL_FORMATS=p8e0`
- runtime 會優先從 `POSIT_QALIGN_FILE` 指到的新版 `qalign_p8e0.csv`
  讀 `gp_rs/gp_sc`
- 若主 CSV 沒有這兩欄，才退回 `qalign_p8e0_detail.csv`

重要限制：

- 目前 runtime 可以：
  - 正確讀到 per-layer `gp_rs/gp_sc` metadata
  - 在 probe 與 auto-pick 上保留這些值
- 但 **完整的 per-layer generalized posit arithmetic** 仍然沒有完全實現
  - 現在的 experimental GP numerical path 仍是 format-global
  - 若要讓每一層真的用不同 `rs/sc` 做 decode / MAC / quire
  - 還需要在 lowering / tensor metadata / runtime storage semantics 再往下改

也就是說：

- 這次已經做到：
  - per-layer 建議值蒐集
  - per-layer CSV 輸出
  - runtime 自動讀取
  - 一鍵 collect + apply artifact 準備好
- 但若要 paper 那種「每層都真的用不同 generalized posit arithmetic」，
  還需要下一階段的 runtime / codegen 改造

22.10.7 Full per-layer generalized posit runtime path

這次把前一節卡住的三段補上：

1. lowering

- `ONNXToPosit/Pattern/Math.cpp`
  - `Add/Sub/Mul/Div/Relu/Clip/MaxPool/Conv/Gemm/ReduceMean`
    現在都會帶一個 layer-level `qalign_key`
  - 這個 key 用 op location + neutral qalign signature 生成
  - 目的不是重用 DQ key，而是讓「每個產生 posit tensor 的 op」
    都能有穩定的 per-layer key

2. tensor metadata

- `posit_runtime.cpp`
  - 新增 runtime tensor metadata registry
  - key: output memref `data` pointer
  - value:
    - `enabled`
    - `rs`
    - `sc`
    - `qalignKey`
- `posit_from_f32` / `dequantize_linear` 會在寫出 tensor 後註冊 metadata
- `add/relu/gemm/conv/maxpool/clip/reduce_mean`
  會在產生 output tensor 後延續或覆寫 metadata

3. runtime storage semantics

- `posit_to_f32` 現在會依 input tensor metadata 解碼
- `add/sub/mul/div`
  - 若任一 input / output 啟用 GP metadata
  - 會改走：
    - decode(A, rs/sc_A)
    - decode(B, rs/sc_B)
    - real-domain op
    - encode(out, rs/sc_out)
- `relu/maxpool/clip/reduce_mean`
  - 同樣依 tensor metadata 做 decode / encode
- `gemm/conv`
  - 若任一 input / output 啟用 GP metadata
  - 會改走 metadata-aware double accumulation
  - 再依 output tensor 的 `rs/sc` re-encode
  - 這樣每層真的可以用不同 generalized posit storage semantics

4. collect path

- 只要 op 有 layer-level `qalign_key`
  - runtime 會對 output tensor 做抽樣
  - 同時寫：
    - `orig` = op 真實輸出值
    - `dq` = 依該 layer `rs/sc` 存回後再解碼的值
- 目前已接到：
  - `add/sub/mul/div`
  - `relu`
  - `clip`
  - `maxpool`
  - `reduce_mean`
  - `gemm`
  - `conv`
- 所以下一次重新跑 auto-qalign / calibration 時，
  這些新 layer key 不再只有 runtime consume path，校正器也能看到資料

目前的套用規則：

- runtime 先看 `POSIT_QALIGN_FILE`
  - 若該 `qalign_p8e*.csv` 有：
    - `gp_rs`
    - `gp_sc`
  - 就以 `qalign_key` 對應到的 layer-level key 為主
- 若該 key 沒有 `gp_rs/gp_sc`
  - 才退回 input inherit
- 若 input 也沒有
  - 才退回 format-level `POSIT_GP_EXPERIMENTAL_FORMATS`

注意：

- 這條 runtime 現在是「完整 per-layer generalized posit arithmetic path」
  的 codegen / metadata / storage 路徑
- 但 generalized posit value-set 仍然是目前 runtime 裡的
  approximate 8-bit decoder / encoder
- 也就是說：
  - per-layer `rs/sc` 已經真的會影響每層算術與存回
  - 但它還不是另外實作一套 hardware-accurate generalized posit quire

重跑方式：

1. 先重編新的 runtime / `.so`

2. 若你已經有新版 qalign CSV：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

3. 若要同時測多個格式：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

4. 若要看 runtime 確認值：

```bash
POSIT_DESC_DEBUG=1 \
QALIGN_RUNTIME_PROBE=on \
QALIGN_RUNTIME_PROBE_LIMIT=8 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/time_mobilenet11_dataset_parallel.sh ...
```

目前最重要的使用前提：

- `POSIT_QALIGN_FILE` 指到的 `qalign_p8e*.csv`
  最好是新版，且帶：
  - `gp_rs`
  - `gp_sc`
  - `gp_source`
- 若沒有這三欄，runtime 仍可跑
  但該 layer 會退回 inherit / format-level 設定

## nqdq compile-time weight ALPS

目標：

- 只在 `--posit-source nqdq` 的 compact constant 路徑啟用
- compile-time 直接對 float weight tensor 做：
  - direct：`orig_x -> posit`
  - ALPS：`orig_x -> y = asinh(theta*x)/gamma -> y_q(raw bits)`
- 若 `ALPS` 重建出的 `x_dq = sinh(gamma*y_q)/theta` 比 direct posit decode 更接近 `orig_x`
  才採用 ALPS
- 不動 `--posit-source qdq` / `strict-qdq-mode` 路徑

主要實作位置：

- `/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToPosit/Pattern/Math.cpp`
  - `buildTimeWeightAlpsDecision(...)`
  - `createCompactPositConstantIfEnabled(...)`
- `/home/lai/onnx_mlir/onnx-mlir/src/Conversion/PositToKrnl/Pattern/Math.cpp`
  - `posit.constant -> krnl.global`
  - 若 constant 帶 `compand_mode/theta/gamma`，lowering 時註冊到 runtime metadata
- `/home/lai/onnx_mlir/onnx-mlir/src/posit_runtime.cpp`
  - `TensorGPMetadata` 新增：
    - `compandMode`
    - `theta`
    - `gamma`
  - `decodeTensorValue<Fmt>(bits, meta)` 若看到 ALPS metadata：
    - 先把 raw posit bits decode 成 `y_q`
    - 再回 `x_dq = sinh(gamma*y_q)/theta`

目前行為：

- `direct` baseline：
  - `orig_x -> posit raw bits -> decode`
  - score = tensor-level `MAE(orig_x, direct_decode)`
- `ALPS` candidate：
  - 搜 `theta`
  - 由 tensor values 估 `gamma`
  - 試 `gammaBase * {0.5, 1.0, 2.0}`
  - score = tensor-level `MAE(orig_x, x_dq)`
- 若 `candidate_score + min_gain < direct_score`
  - 存 raw bits = `y_q`
  - 並把
    - `compand_mode = 1`
    - `compand_theta = theta`
    - `compand_gamma = gamma`
    附在 `posit.constant`
- 否則維持原本 direct raw-bit constant

啟用 env：

```bash
ONNX_MLIR_POSIT_CONST_ALPS=1
```

可調參數：

```bash
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.25
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=4
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=9
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.0
```

除錯 env：

```bash
POSIT_CONST_DEBUG=1
```

可看到像這樣的訊息：

```text
[posit-const] build-time ALPS start ...
[posit-const] build-time ALPS selected nbits=8 es=0 direct_score=... chosen_score=... theta=... gamma=...
```

驗證重點：

1. `onnx-mlir-opt --convert-onnx-to-posit --posit-format=p8e0` 的輸出 MLIR
   應可看到：
   - `posit.constant`
   - `compand_mode = 1`
   - `compand_theta = ...`
   - `compand_gamma = ...`

2. `--posit-source nqdq` 的 mobilenet build 已驗到 ALPS 真的被選上

3. `--posit-source qdq --strict-qdq-mode` guard build 仍可成功
   - 代表這次沒有動到 qdq-posit 路徑

### nqdq build-time ALPS + generalized posit (`RS/SC`)

這次已把 `RS/SC` 接進 **`--posit-source nqdq` 的 build-time ALPS constant 路徑**。

高層行為：

- 只影響 `nqdq` compact constant
- 不動 `qdq-posit`
- 若 build-time constant 同時啟用 generalized posit：
  - direct baseline 會用 generalized posit 的 value-set 做 `orig_x -> raw bits -> decode`
  - ALPS candidate 也會用 generalized posit 的 `y_q` value-set 做：
    - `orig_x -> y = asinh(theta*x)/gamma -> y_q(raw bits)`
    - `x_dq = sinh(gamma*y_q)/theta`
  - compile-time 會比較：
    - `MAE(orig_x, direct_decode)`
    - `MAE(orig_x, x_dq)`
  - 若 ALPS 更好才採用

build-time 主要實作位置：

- `/home/lai/onnx_mlir/onnx-mlir/src/Conversion/ONNXToPosit/Pattern/Math.cpp`
  - `resolveBuildTimeGPConfig(...)`
  - `generalizedP8RoundToDoubleDispatch(...)`
  - `buildTimeWeightAlpsDecision(...)`
  - `createCompactPositConstantIfEnabled(...)`

runtime / lowering 主要位置：

- `/home/lai/onnx_mlir/onnx-mlir/src/Conversion/PositToKrnl/Pattern/Math.cpp`
  - 若 `posit.constant` 帶：
    - `gp_enabled`
    - `gp_rs`
    - `gp_sc`
    - `compand_mode`
    - `compand_theta`
    - `compand_gamma`
  - lowering 會呼叫 `posit_register_tensor_constmeta_*`
- `/home/lai/onnx_mlir/onnx-mlir/src/posit_runtime.cpp`
  - `decodeTensorValue<Fmt>(bits, meta)`
    - 若是 generalized posit + ALPS constant：
      1. 先用 `rs/sc` decode raw bits 成 generalized-posit `y_q`
      2. 再做 `x_dq = sinh(gamma*y_q)/theta`
  - ALPS metadata path 目前後續運算是 `gp_metadata_f32`
    - decode 到實值後用 `f32`
    - 最後再 encode 回輸出格式（例如 `p8e0/p8e1/p8e2`）

目前支援範圍：

- generalized posit build-time constant 目前實作在 `p8` 系列
  - `p8e0`
  - `p8e1`
  - `p8e2`
- `p16/p32` 仍可帶 ALPS metadata
  但 `RS/SC` 的 generalized posit compile-time value-set 目前沒有另外做

啟用 generalized posit build-time constant 的 env：

```bash
ONNX_MLIR_POSIT_CONST_ALPS=1
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0
POSIT_GP_RS_P8E0=7
POSIT_GP_SC_P8E0=0
```

也可直接指定其他格式，例如：

```bash
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1
POSIT_GP_RS_P8E0=7
POSIT_GP_SC_P8E0=-3
POSIT_GP_RS_P8E1=7
POSIT_GP_SC_P8E1=-2
```

compile-time constant 專用的更明確開關：

```bash
ONNX_MLIR_POSIT_CONST_GP=1
ONNX_MLIR_POSIT_CONST_GP_FORMATS=p8e0,p8e1,p8e2
```

目前 build-time GP 的 `rs/sc` 讀取優先順序：

1. `ONNX_MLIR_POSIT_CONST_GP_RS_<FMT>` / `...SC_<FMT>`
2. `POSIT_CONST_GP_RS_<FMT>` / `...SC_<FMT>`
3. `POSIT_GP_RS_<FMT>` / `...SC_<FMT>`
4. `..._P8`
5. generic `..._RS` / `..._SC`

build-time `rs/sc` 小範圍 sweep：

- 目前 build-time constant 已支援在 compile-time 一次試多組 `rs/sc`
- 行為是：
  1. 先看 standard direct（無 GP）
  2. 再看每組 `GP direct`
  3. 再看每組 `GP + ALPS(theta/gamma)`
  4. 最後選整個 tensor 上 `MAE(orig_x, reconstructed)` 最小的那條
- 只有 `ALPS` 仍受 `MIN_GAIN` 約束
  - `GP direct` / `standard direct` 是直接拿全域較佳者

可用 env：

```bash
POSIT_GP_RS_VALUES_P8E0=7,6
POSIT_GP_SC_VALUES_P8E0=0,-3
```

也支援：

```bash
ONNX_MLIR_POSIT_CONST_GP_RS_VALUES_P8E0=7,6
ONNX_MLIR_POSIT_CONST_GP_SC_VALUES_P8E0=0,-3
```

若不想手列清單，也可用 range：

```bash
POSIT_GP_RS_MIN_P8=6
POSIT_GP_RS_MAX_P8=7
POSIT_GP_SC_MIN_P8=-3
POSIT_GP_SC_MAX_P8=0
```

或 format-specific：

```bash
POSIT_GP_RS_MIN_P8E0=6
POSIT_GP_RS_MAX_P8E0=7
POSIT_GP_SC_MIN_P8E0=-3
POSIT_GP_SC_MAX_P8E0=0
```

目前實測 debug 可看到：

```text
gp_candidate_count=4
```

例如：

```bash
ONNX_MLIR_POSIT_CONST_ALPS=1 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
POSIT_GP_RS_VALUES_P8E0=7,6 \
POSIT_GP_SC_VALUES_P8E0=0,-3 \
POSIT_CONST_DEBUG=1 \
/home/lai/onnx_mlir/onnx-mlir/build/Release/bin/onnx-mlir-opt \
  /home/lai/onnx_mlir/onnx-mlir/src/temp/model/mobilenetv2-12.onnx.mlir \
  --mlir-disable-threading \
  --shape-inference \
  --convert-onnx-to-posit \
  --posit-format=p8e0 \
  -o /tmp/mobilenet11_nqdq_gp_sweep_check.posit.mlir
```

驗證結果：

1. `rs=7, sc=0` 時，build-time generalized posit decode/encode 已對到 standard posit
   - `decode_mismatch_count=0 / 256`
   - `roundtrip_mismatch_count=0 / 255`

2. custom `rs/sc` 小測試已驗到代表點合理，例如 `p8e1 rs=5 sc=-2`：
   - `x=-3.5 -> raw=146 -> dq=-3.5`
   - `x=-1   -> raw=160 -> dq=-1`
   - `x=-0.125 -> raw=208 -> dq=-0.125`
   - `x=0.125  -> raw=48  -> dq=0.125`
   - `x=1      -> raw=96  -> dq=1`

3. `mobilenet nqdq p8e0` 的 MLIR / LLVM stage 已驗到：
   - `posit.constant` 會帶：
     - `gp_enabled = 1`
     - `gp_rs = ...`
     - `gp_sc = ...`
     - `compand_mode = 1`
     - `compand_theta = ...`
     - `compand_gamma = ...`
   - `convert-krnl-to-llvm` 後可看到：
     - `posit_register_tensor_constmeta_p8e0(...)`

使用範例：

```bash
ONNX_MLIR_POSIT_CONST_ALPS=1 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0 \
POSIT_GP_RS_P8E0=7 \
POSIT_GP_SC_P8E0=0 \
POSIT_CONST_DEBUG=1 \
POSIT_RUNTIME_CPP_PATH=/home/lai/onnx_mlir/onnx-mlir/src/posit_runtime.cpp \
RUN_TIME_CPP_PATH=/home/lai/onnx_mlir/onnx-mlir/src/run_time.cpp \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet11_sos.sh \
  /tmp/mobilenet11_nqdq_gp_alps_smoke \
  --posit-source nqdq \
  --posit-formats p8e0 \
  --skip-f32-baselines
```

### Single-format runtime + build-side mixed + qalign ALPS-only

這次另外補了三個控制面，目標是：

- `.so` 只編本 format，減少 runtime 體積
- mixed promoted dot path 改成 build 時固定，不再只能靠 runtime env
- qalign 校正器可切成只留 ALPS，不再走 legacy/paper alpha 那條

新增的 `build_model11_sos.sh` 參數：

```bash
--runtime-format-scope full|single
--runtime-qalign-mode full|alps-only
--runtime-mixed-accum runtime|off|p16e2|p32e2
```

語意：

- `--runtime-format-scope single`
  - runtime 只 export / compile 當前 format
  - 例如 `--posit-formats p8e0` 時，只保留 `p8e0`
- `--runtime-qalign-mode alps-only`
  - compile-time 關掉舊的 runtime qalign 非-ALPS theta-only 行為
  - non-ALPS qalign 視為 identity
- `--runtime-mixed-accum`
  - `runtime`：保留原本 runtime env 控制 mixed
  - `off`：完全不走 mixed promoted accumulation
  - `p16e2`：低位元格式固定走 `p16e2` promoted mixed path
  - `p32e2`：低位元格式固定走 `p32e2` promoted mixed path

qalign 校正器新增：

```bash
QALIGN_CALIB_MODE=full|alps-only
```

語意：

- `full`
  - 保留目前 legacy alpha / paper alpha / ALPS 的完整混合選擇
- `alps-only`
  - 跳過 legacy / paper alpha
  - 只比較：
    - 原始 off baseline
    - ALPS candidate
  - 若 ALPS 不更好，仍保留 strict fallback

最小 smoke：

```bash
timeout 180s bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet11_sos.sh \
  /tmp/mobilenet11_singlefmt_smoke2 \
  --posit-source nqdq \
  --posit-formats p8e0 \
  --skip-f32-baselines \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off
```

結果：

- 成功產出：
  - `/tmp/mobilenet11_singlefmt_smoke2/mobilenetv2-12-nqdq-p8e0.so`

qalign ALPS-only smoke：

```bash
QALIGN_CALIB_MODE=alps-only \
QALIGN_COMPAND_MODE=alps \
QALIGN_FORCE_COMPAND=off \
QALIGN_OUTPUT_FORMATS=p8e0 \
bash /home/lai/onnx_mlir/onnx-mlir/src/bash/calibrate_qalign_pernode.sh \
  --collect-csv /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/collect_upto_25.csv \
  --out-dir /tmp/qalign_alps_only_smoke \
  --formats p8e0
```

console output：

```text
[sigma-auto][disabled] calib_mode=alps-only use fixed sigma=2
[calib][p8e0] sigma=2 buckets=105 jobs=28 non_identity_alpha=0
```

這組 single-format runtime 的體積對比：

- 舊 full-runtime `p8e0`
  - `/home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_nqdq_posit_final_alps/mobilenetv2-12-nqdq-p8e0.so`
  - `9075344` bytes
- 新 single-format `p8e0`
  - `/tmp/mobilenet11_singlefmt_smoke/mobilenetv2-12-nqdq-p8e0.so`
  - `3937472` bytes

dynamic symbols：

- old: `2007`
- new: `194`

### Runtime output ALPS (`p8e0/p8e1/p8e2` only)

新增 build 端參數：

```bash
--runtime-output-alps off|sampled|full
```

語意：

- `off`
  - 維持目前行為，output encode 回一般 `p8` / `GP`
- `sampled`
  - 只對 `p8e0/p8e1/p8e2`
  - 在 `gp_metadata_f32` 路徑上，先暫存整個 output tensor 的 `f32` 結果
  - 用抽樣值搜尋新的 `theta/gamma(/rs/sc)`
  - 再把整個 output tensor encode 成新的 ALPS raw bits + metadata
- `full`
  - 同上，但搜尋時用整個 output tensor 全量值

目前這條 output-ALPS 路徑：

- 只接在 `p8e0/p8e1/p8e2`
- 只接在 `meta_f32` 路徑
- 搜尋目標是 tensor-level `MAE(orig_f32_output, decode(encode(orig_f32_output)))`
- 如果沒有另外指定 runtime-output ALPS 的搜尋範圍，會沿用 build 時 bake 進 runtime 的：
  - `ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN/MAX/STEPS`
  - `ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET/PERCENTILE`
  - `ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN`
  - `ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES`
  - `POSIT_GP_RS_VALUES_P8*`
  - `POSIT_GP_SC_VALUES_P8*`

實作重點：

- `encodeTensorValue(...)` 現在如果 output metadata 帶 `compand_mode=alps`
  - 會先做：
    - `y = asinh(theta * x) / gamma`
  - 再用標準 posit 或 GP(`rs/sc`) 把 `y` encode 成 raw bits
- `chooseOutputTensorGPMetadata(...)` 改成會繼承完整 special metadata
  - 不再只看 `gp enabled`
  - 所以 output 的 ALPS metadata 可以往後續 op 傳

### Runtime per-channel constmeta fallback bug（2026-05-21）

症狀：

- ImageNet100 MobileNetV2 local-trained model：
  - `nqdq-f32` / `qdq-f32` 在同一批 validation image 上 Top1 正常
  - `nqdq-p8e0/p8e1/p8e2` 即使 `--output-alps-auto off` 仍接近全 0
- 單張 `validation/000/000000.jpg` logits：
  - `nqdq-f32` 正確 class `0` 是 Top1
  - `nqdq-p8e0` 把 class `0` 壓到很低，Top1 變成 class `18/19/47/50` 一類的錯誤 class
- `POSIT_TRACE=1` 顯示早期 Conv/Gemm activation 很快打到 `±64` 等粗粒度邊界。
- ResNet18 同路線仍可接受，表示不是整個 posit runtime 壞掉，而是 MobileNetV2 對 per-channel metadata / depthwise-pointwise 結構更敏感。

關鍵診斷：

```bash
# 查 MobileNetV2 f32 ONNX Conv output channels
python3 - <<'PY'
import onnx
from onnx import numpy_helper
m=onnx.load('/home/lai/onnx_mlir/ImageNet100/model/imagenet100_mobilenetv2.onnx')
init={x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
conv=[]
for n in m.graph.node:
    if n.op_type=='Conv' and n.input[1] in init:
        conv.append((n.name,n.input[1],init[n.input[1]].shape))
print('conv count',len(conv),'sum OC',sum(s[0] for _,_,s in conv))
PY

# 查 p8 runtime .so 內 constmeta 註冊數
for f in p8e0 p8e1 p8e2; do
  so=/home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps/imagenet100_mobilenetv2-nqdq-${f}.so
  echo ${f}
  objdump -d "$so" | rg -c "call.*posit_register_tensor_constmeta_${f}"
  objdump -d "$so" | rg -c "call.*posit_register_tensor_constmeta_channel_${f}"
done
```

當時觀察：

- MobileNetV2 Conv count = `52`
- Conv weight output-channel sum = `17056`
- `p8e0/p8e1/p8e2` `.so` 裡的 per-channel constmeta call 約 `14536`
- 代表不是每個 channel 都有自己的 ALPS/GP metadata。

根因：

- `registerTensorConstMetadataForChannel(...)` 先前會在 base tensor metadata 無效時，把某個 channel 的 metadata 同步寫進 tensor base metadata。
- `lookupTensorGPMetadataForChannel(...)` 若找不到該 channel，會 fallback 到 tensor base metadata。
- 因此「沒有 per-channel metadata 的 channel」可能偷用第一個被註冊 channel 的 ALPS/GP 參數。
- MobileNetV2 depthwise / pointwise channel 數很多，這種錯用 channel metadata 會快速污染後續 activation。
- `--output-alps-auto off` 只是不載入 runtime activation/output ALPS CSV，不能避免 build-time weight constmeta 被錯誤 fallback。

修正：

- `src/posit_runtime.cpp`
  - 新增 per-channel metadata tracking：
    - `gTensorGPSpecialChannelMetaPtrs`
    - `gTensorGPAlpsChannelMetaPtrs`
  - `registerTensorGPMetadataForChannel(...)`
    - 註冊 channel metadata 時，只記錄此 tensor 有 special/ALPS channel metadata。
  - `registerTensorConstMetadataForChannel(...)`
    - 若 base metadata 尚未存在，base 改成 direct/default metadata。
    - 不再把某個 channel 的 ALPS/GP metadata 寫成整個 tensor base metadata。
  - Conv/Gemm runtime：
    - 會檢查 input/weight/bias tensor 是否有 per-channel metadata。
    - 有 per-channel metadata 時仍走 metadata-aware decode path。
    - 沒註冊 metadata 的 channel 會 fallback 到 direct/default posit，而不是偷用別的 channel。

目前語意：

- 有 per-channel ALPS/GP metadata 的 channel：
  - 用自己的 `theta/gamma/rs/sc` decode。
- 沒有 per-channel metadata 的 channel：
  - 用一般 direct/default posit。
- 整個 tensor 有 per-tensor metadata 時：
  - 才使用 per-tensor metadata。

驗證：

```bash
# C++ compile check（不是完整模型 build，只確認 posit_runtime.cpp 可編）
/usr/bin/clang++ -std=c++20 -O0 -fPIC -c \
  /home/lai/onnx_mlir/onnx-mlir/src/posit_runtime.cpp \
  -o /tmp/posit_runtime_syntax_check.o \
  -I/home/lai/onnx_mlir/onnx-mlir/build/include \
  -I/home/lai/onnx_mlir/onnx-mlir/src \
  -I/home/lai/onnx_mlir/llvm-project/mlir/include \
  -I/home/lai/onnx_mlir/llvm-project/build/tools/mlir/include \
  -I/home/lai/onnx_mlir/llvm-project/llvm/include \
  -I/home/lai/onnx_mlir/llvm-project/build/include \
  -I/home/lai/onnx_mlir/onnx-mlir/src/.deps/universal/include/sw \
  -DPOSIT_USE_UNIVERSAL \
  -DPOSIT_RUNTIME_SINGLE_FORMAT=1 \
  -DPOSIT_RUNTIME_FMT_P8E1=1 \
  -DPOSIT_BUILD_MIXED_OFF=1 \
  -DPOSIT_RUNTIME_QALIGN_ALPS_ONLY=1 \
  -DPOSIT_RUNTIME_OUTPUT_ALPS_OFFLINE=1
```

下一步驗證建議：

```bash
POSIT_FORMATS=p8e1 \
INCLUDE_F32_BASELINES=1 \
POSIT_CONST_ALPS_JOBS=25 \
POSIT_CONST_DEBUG=1 \
ONNX_MLIR_POSIT_CONST_ALPS=1 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.0078125 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=256 \
ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=113 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.95 \
ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
POSIT_GP_EXPERIMENTAL_FORMATS=p8e0,p8e1,p8e2 \
POSIT_GP_RS_VALUES_P8=7,6,5 \
POSIT_GP_SC_VALUES_P8=3,-3 \
bash /home/lai/onnx_mlir/ImageNet100/build_imagenet100_mobilenetv2_11_sos.sh \
  /home/lai/onnx_mlir/ImageNet100/build_posit11_mobilenetv2_alps_fixcheck \
  --posit-source nqdq \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \
  --runtime-output-alps offline
```

最小 smoke：

```bash
timeout 240s bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_mobilenet11_sos.sh \
  /tmp/mobilenet11_output_alps_smoke \
  --posit-source nqdq \
  --posit-formats p8e0 \
  --skip-f32-baselines \
  --runtime-format-scope single \
  --runtime-qalign-mode alps-only \
  --runtime-mixed-accum off \
  --runtime-output-alps sampled
```

結果：

- 成功產出：
  - `/tmp/mobilenet11_output_alps_smoke/mobilenetv2-12-nqdq-p8e0.so`

補充：

- 原本 `run_time_sp` 只會硬找 `_mlir_ciface_main_graph`，
  對像 `mobilenetv2-12-nqdq-p8e0.so` 這種實際匯出：
  - `_mlir_ciface_main_graph_mobilenetv2-12`
  - `run_main_graph`
  - `main_graph_mobilenetv2-12`
  的 `.so` 會誤判成 entrypoint mismatch。
- 2026-05-05 已修成 runner 自動 fallback：
  - `_mlir_ciface_main_graph`
  - `_mlir_ciface_main_graph_<derived-model-stem>`
  - `run_main_graph`
  - `run_main_graph_<derived-model-stem>`
  - `main_graph`
  - `main_graph_<derived-model-stem>`
- `derived-model-stem` 會從 `.so` 檔名推，例如：
  - `mobilenetv2-12-nqdq-p8e0.so` -> `mobilenetv2-12`
  - `mobilenetv2-12-qdq-f32.so` -> `mobilenetv2-12-qdq-f32` 與 `mobilenetv2-12`
- 若使用者明確指定 `--entry` 或 cmp spec 的 `:entry`，
  runner 仍保持精確模式，不會偷偷 fallback。

驗證：

- `nqdq p8e0`
```bash
timeout 60s /tmp/mobilenet11_output_alps_smoke_fix2/run_time_sp \
  /tmp/mobilenet11_output_alps_smoke_fix2/mobilenetv2-12-nqdq-p8e0.so \
  --shape 1x3x224x224 --zeros --warmup 0 --iters 1 --no-benchmark
```
  - 會顯示：
    - `CONFIG main_entry_resolved=_mlir_ciface_main_graph_mobilenetv2-12`

- `qdq-f32`
```bash
timeout 60s /tmp/mobilenet11_output_alps_smoke_fix2/run_time_sp \
  /home/lai/onnx_mlir/onnx-mlir/src/temp/mobilenet11_nqdq_posit_final_alps/mobilenetv2-12-qdq-f32.so \
  --shape 1x3x224x224 --zeros --warmup 0 --iters 1 --no-benchmark
```
  - 會顯示：
    - `CONFIG main_entry_resolved=_mlir_ciface_main_graph_mobilenetv2-12-qdq-f32`
