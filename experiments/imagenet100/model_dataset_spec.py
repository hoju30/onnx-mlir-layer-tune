#!/usr/bin/env python3
"""
Single source of truth for the 22-model dataset defined in
ml_predictor_dataset.md: every (architecture, pretrained checkpoint) pair
that feeds the mixed-posit tuning dataset (claude.md section 22
"Experimental Design"), and the model_dataset/<arch>/<variant>/ directory
each one's exported/downloaded ONNX file belongs in.

Per ml_predictor_dataset.md: retraining is explicitly NOT part of this
dataset -- every entry loads its original ImageNet-1K pretrained weights
as-is and is exported/downloaded verbatim; evaluation on the ImageNet-100
subset (clane9/imagenet-100) happens by filtering/mapping the 1000-way
logits down to those 100 classes at eval time, not by fine-tuning a
100-class head.
"""

MODELS = [
    # torchvision -----------------------------------------------------------
    {"arch": "resnet18", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "resnet18", "torchvision_weights": "IMAGENET1K_V1",
     "note": "ResNet18 Weight A"},
    {"arch": "resnet50", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "resnet50", "torchvision_weights": "IMAGENET1K_V1",
     "note": "ResNet50 Weight A"},
    {"arch": "resnet50", "variant": "torchvision_v2", "source": "torchvision",
     "torchvision_model": "resnet50", "torchvision_weights": "IMAGENET1K_V2",
     "note": "ResNet50 Weight B"},
    {"arch": "mobilenetv2", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "mobilenet_v2", "torchvision_weights": "IMAGENET1K_V1",
     "note": "MobileNetV2 Weight A"},
    {"arch": "mobilenetv2", "variant": "torchvision_v2", "source": "torchvision",
     "torchvision_model": "mobilenet_v2", "torchvision_weights": "IMAGENET1K_V2",
     "note": "MobileNetV2 Weight B"},
    {"arch": "densenet121", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "densenet121", "torchvision_weights": "IMAGENET1K_V1",
     "note": "DenseNet Weight A"},
    {"arch": "efficientnet_b0", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "efficientnet_b0", "torchvision_weights": "IMAGENET1K_V1",
     "note": "EfficientNet Weight A"},
    {"arch": "vgg16", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "vgg16", "torchvision_weights": "IMAGENET1K_V1",
     "note": "Sequential CNN"},
    {"arch": "alexnet", "variant": "torchvision_v1", "source": "torchvision",
     "torchvision_model": "alexnet", "torchvision_weights": "IMAGENET1K_V1",
     "note": "Traditional CNN"},

    # Hugging Face / timm -----------------------------------------------------
    {"arch": "resnet18", "variant": "timm_a1", "source": "timm",
     "timm_model": "resnet18.a1_in1k", "note": "ResNet18 Weight B"},
    {"arch": "resnet18", "variant": "timm_a2", "source": "timm",
     "timm_model": "resnet18.a2_in1k", "note": "ResNet18 Weight C"},
    {"arch": "resnet50", "variant": "timm_a1", "source": "timm",
     "timm_model": "resnet50.a1_in1k", "note": "ResNet50 Weight C"},
    {"arch": "resnet50", "variant": "timm_a2", "source": "timm",
     "timm_model": "resnet50.a2_in1k", "note": "ResNet50 Weight D"},
    {"arch": "mobilenetv2", "variant": "timm_ra", "source": "timm",
     "timm_model": "mobilenetv2_100.ra_in1k", "note": "MobileNetV2 Weight C"},
    {"arch": "densenet121", "variant": "timm_ra", "source": "timm",
     "timm_model": "densenet121.ra_in1k", "note": "DenseNet Weight B"},
    {"arch": "efficientnet_b0", "variant": "timm_ra", "source": "timm",
     "timm_model": "efficientnet_b0.ra_in1k", "note": "EfficientNet Weight B"},

    # ONNX Model Zoo (official pretrained ONNX, no PyTorch equivalent used) ---
    {"arch": "inception_v1", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/inception_and_googlenet/inception_v1/model/inception-v1-12.onnx",
     "note": "Multi-branch"},
    {"arch": "inception_v2", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
     "note": "Multi-branch + BN"},
    {"arch": "shufflenet_v1", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/shufflenet/model/shufflenet-9.onnx",
     "note": "Lightweight CNN"},
    {"arch": "shufflenet_v2", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/shufflenet/model/shufflenet-v2-12.onnx",
     "note": "Lightweight CNN"},
    {"arch": "squeezenet", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/squeezenet/model/squeezenet1.0-12.onnx",
     "note": "Fire Module"},
    {"arch": "efficientnet_lite4", "variant": "onnx_model_zoo", "source": "onnx_model_zoo",
     "zoo_url": "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                "classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
     "note": "MBConv / Lightweight"},
]

assert len(MODELS) == 22, f"expected 22 model instances, got {len(MODELS)}"
# ml_predictor_dataset.md's own Dataset Summary table says 12, but its 22-row
# Model List actually names 13 distinct architectures (ResNet18 and ResNet50
# counted separately, Inception V1/V2 and ShuffleNet V1/V2 each counted
# separately) -- the row-level table is treated as ground truth here since
# that's what actually drives model_dataset/'s directory structure.
N_DISTINCT_ARCHITECTURES = len({m["arch"] for m in MODELS})


def out_path(model_dataset_dir, entry):
    return f"{model_dataset_dir}/{entry['arch']}/{entry['variant']}/model.onnx"
