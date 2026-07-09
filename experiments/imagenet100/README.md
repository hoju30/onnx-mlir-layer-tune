# ImageNet100 模型準備與評估流程

本目錄放 ImageNet100 posit 實驗用的 **模型準備（資料集下載 → f32 訓練 → ONNX 匯出
→ int8 QDQ 量化）** 與 **評估腳本**。posit `.so` 的 build / lowering 說明見 repo 根目錄
的 [`POSIT_README.md`](../../POSIT_README.md)。

> **為什麼資料集與模型不在 repo？** 資料集約 1.3GB、`.so` 單檔可達 650MB，都超過
> GitHub 檔案上限，因此不上傳。資料集本來就公開在 HuggingFace，模型可由以下流程完全
> 重現。以下所有指令都在 `ImageNet100/`（放這些腳本、資料集、模型的工作目錄）下執行。

## 0. 環境需求

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision timm datasets onnx onnxruntime numpy pillow
```

- 下載資料集：`datasets`（HuggingFace）
- 訓練 / 匯出：`torch`、`torchvision`、`timm`
- int8 量化 / 評估：`onnx`、`onnxruntime`

## 1. 重抓資料集（HuggingFace ImageNet-100）

```bash
python3 download_dataset.py
```

- 來源：HuggingFace dataset **`clane9/imagenet-100`**（`load_dataset`）。
- 產出：`./imagenet100_hf/{train,validation}/<類別 000..099>/<編號>.jpg`
  （100 類，validation 每類 50 張 = 5000 張，依類別資料夾排序）。
- 這就是 posit runtime `--image-dir` 與訓練 `--train-dir/--val-dir` 用的目錄。

## 2. f32 訓練（原始浮點模型）

用 timm backbone 在 ImageNet100 上 fine-tune：

```bash
# MobileNetV2 (timm mobilenetv2_100)
python3 train_imagenet100_mobilenetv2.py \
  --train-dir imagenet100_hf/train \
  --val-dir   imagenet100_hf/validation \
  --epochs 5 --batch-size 96 --lr 1e-3 --img-size 224 \
  --output-dir imagenet100_mobilenetv2_out

# ResNet18 (timm resnet18)
python3 train_imagenet100_resnet18.py \
  --train-dir imagenet100_hf/train \
  --val-dir   imagenet100_hf/validation \
  --epochs 5 --batch-size 64 --lr 1e-3 --img-size 224 \
  --output-dir imagenet100_resnet18_out
```

產出 `--output-dir` 下的 `best.pth`（最佳權重）與 `class_map.json`（類別對應）。
（`train_imagenet100_resnet50.py` 為 ResNet50 舊流程，非目前主線。）

## 3. 匯出 ONNX（f32）

```bash
python3 export_imagenet100_mobilenetv2_onnx.py \
  --ckpt imagenet100_mobilenetv2_out/best.pth \
  --class-map imagenet100_mobilenetv2_out/class_map.json \
  --onnx-out model/imagenet100_mobilenetv2.onnx --img-size 224

python3 export_imagenet100_resnet18_onnx.py \
  --ckpt imagenet100_resnet18_out/best.pth \
  --class-map imagenet100_resnet18_out/class_map.json \
  --onnx-out model/imagenet100_resnet18.onnx --img-size 224
```

產出 `model/imagenet100_mobilenetv2.onnx` / `model/imagenet100_resnet18.onnx`
（posit nqdq 路徑的輸入）。

## 4. int8 QDQ 量化

ONNX Runtime static QDQ（`QuantFormat.QDQ`、activation `int8`、weight per-channel、
`CalibrationMethod.MinMax`）。校準集用 `val_224_txt/`（validation 影像預處理後的 tensor
txt，前 `--limit` 筆）。

**最簡單：由 build wrapper 自動量化**（若 `-int8-qdq.onnx` 不存在就自動產生後再 build
posit `.so`）：

```bash
bash build_imagenet100_mobilenetv2_11_sos.sh <out_dir> --posit-source qdq ...
# 校準參數可用環境變數覆寫：CALIB_LIMIT(256) CALIB_METHOD(minmax) CALIB_ACT_TYPE(int8)
# 已存在時會略過；FORCE_REGEN_QDQ=1 可強制重產。
```

**或手動呼叫**：

```bash
python3 quantize_imagenet100_qdq.py \
  --model-in  model/imagenet100_mobilenetv2.onnx \
  --model-out model/imagenet100_mobilenetv2-int8-qdq.onnx \
  --txt-dir val_224_txt --input-name input --shape 1x3x224x224 \
  --limit 256 --method minmax --activation-type int8 --per-channel
```

產出 `model/imagenet100_*-int8-qdq.onnx`（posit qdq 路徑的輸入）。

## 5. 編成 posit `.so` 並評估

- 一鍵 build（含 f32 baseline 與各 posit 格式）：
  `build_imagenet100_mobilenetv2_11_sos.sh` / `build_imagenet100_resnet18_11_sos.sh`
  （底層呼叫 `src/bash/build_model11_sos.sh`；ALPS 等選項見 `POSIT_README.md`
  與 `CHANGES.md`）。
- 資料集平行評估：`src/bash/time_model11_dataset_parallel.sh`。
- f32 / int8 / fp16 / bf16 / fp8 的 ONNX Runtime 參考評估：
  `eval_onnx_imagenet100.py`、`eval_fp_formats.py`。
- 前處理（Resize256→CenterCrop224→ToTensor→ImageNet mean/std）：
  `preprocess_imagenet100_tensor.py`（runtime `--image-preprocess-script` 也用它）。

## 6. GPT-2 文字評估（perplexity / top-1）

posit GPT-2 `.so` 的 WikiText-2 perplexity 評估（結果會自動記錄成 TSV，
欄位含 ppl / top1 / 時間，見檔頭說明）：

```bash
POSIT_OMP_THREADS=24 python3 run_gpt2_text_eval.py \
  --model <gpt2 .so 或 .onnx> --mode score \
  --text-file eval_text/wikitext2_test.txt --max-tokens 1024 --progress 128
# 視窗平行版（記憶體有界、可多核）：run_gpt2_text_eval_parallel.py --window 128 --jobs 8
```

> 完整逐項改動、除錯歷程與已知陷阱見本目錄的 `CHANGES.md`。
