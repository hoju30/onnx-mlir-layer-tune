# Load Model Dataset

## Dataset Scope
- **Task:** CNN-based Image Classification
- **Pretrained Dataset:** ImageNet-1K
- **Evaluation Dataset:** `clane9/imagenet-100`
- **Retraining:** No
- **Model Sources:**
  - Torchvision
  - Hugging Face / timm
  - ONNX Model Zoo
- **Model Instance Definition:** Architecture + Pretrained Checkpoint
- **Duplicate Rule:**
  - Same architecture + different pretrained weights → 保留
  - Same architecture + same pretrained weights mirrored on different platforms → 去重

---

## Model List

| # | Source | Architecture | Pretrained Checkpoint | Note |
|---:|---|---|---|---|
| 1 | Torchvision | ResNet18 | `IMAGENET1K_V1` | ResNet18 Weight A |
| 2 | Hugging Face / timm | ResNet18 | `resnet18.a1_in1k` | ResNet18 Weight B |
| 3 | Hugging Face / timm | ResNet18 | `resnet18.a2_in1k` | ResNet18 Weight C |
| 4 | Torchvision | ResNet50 | `IMAGENET1K_V1` | ResNet50 Weight A |
| 5 | Torchvision | ResNet50 | `IMAGENET1K_V2` | ResNet50 Weight B |
| 6 | Hugging Face / timm | ResNet50 | `resnet50.a1_in1k` | ResNet50 Weight C |
| 7 | Hugging Face / timm | ResNet50 | `resnet50.a2_in1k` | ResNet50 Weight D |
| 8 | Torchvision | MobileNetV2 | `IMAGENET1K_V1` | MobileNetV2 Weight A |
| 9 | Torchvision | MobileNetV2 | `IMAGENET1K_V2` | MobileNetV2 Weight B |
| 10 | Hugging Face / timm | MobileNetV2 | `mobilenetv2_100.ra_in1k` | MobileNetV2 Weight C |
| 11 | Torchvision | DenseNet121 | `IMAGENET1K_V1` | DenseNet Weight A |
| 12 | Hugging Face / timm | DenseNet121 | `densenet121.ra_in1k` | DenseNet Weight B |
| 13 | Torchvision | EfficientNet-B0 | `IMAGENET1K_V1` | EfficientNet Weight A |
| 14 | Hugging Face / timm | EfficientNet-B0 | `efficientnet_b0.ra_in1k` | EfficientNet Weight B |
| 15 | Torchvision | VGG16 | `IMAGENET1K_V1` | Sequential CNN |
| 16 | Torchvision | AlexNet | `IMAGENET1K_V1` | Traditional CNN |
| 17 | ONNX Model Zoo | Inception V1 | Official Pretrained ONNX | Multi-branch |
| 18 | ONNX Model Zoo | Inception V2 | Official Pretrained ONNX | Multi-branch + BN |
| 19 | ONNX Model Zoo | ShuffleNet V1 | Official Pretrained ONNX | Lightweight CNN |
| 20 | ONNX Model Zoo | ShuffleNet V2 | Official Pretrained ONNX | Lightweight CNN |
| 21 | ONNX Model Zoo | SqueezeNet | Official Pretrained ONNX | Fire Module |
| 22 | ONNX Model Zoo | EfficientNet-Lite4 | Official Pretrained ONNX | MBConv / Lightweight |

---

## Dataset Summary

| Item | Count |
|---|---:|
| CNN Architectures | 12 |
| Pretrained Model Instances | 22 |
| Evaluation Dataset | ImageNet-100 |
| Evaluation Images | 5,000 |
| Retraining | No |

---

## Model Dataset Structure

```text
model_dataset/
│
├── resnet18/
│   ├── torchvision_v1/
│   ├── timm_a1/
│   └── timm_a2/
│
├── resnet50/
│   ├── torchvision_v1/
│   ├── torchvision_v2/
│   ├── timm_a1/
│   └── timm_a2/
│
├── mobilenetv2/
│   ├── torchvision_v1/
│   ├── torchvision_v2/
│   └── timm_ra/
│
├── densenet121/
│   ├── torchvision_v1/
│   └── timm_ra/
│
├── efficientnet_b0/
│   ├── torchvision_v1/
│   └── timm_ra/
│
├── vgg16/
│   └── torchvision_v1/
│
├── alexnet/
│   └── torchvision_v1/
│
├── inception_v1/
│   └── onnx_model_zoo/
│
├── inception_v2/
│   └── onnx_model_zoo/
│
├── shufflenet_v1/
│   └── onnx_model_zoo/
│
├── shufflenet_v2/
│   └── onnx_model_zoo/
│
├── squeezenet/
│   └── onnx_model_zoo/
│
└── efficientnet_lite4/
    └── onnx_model_zoo/