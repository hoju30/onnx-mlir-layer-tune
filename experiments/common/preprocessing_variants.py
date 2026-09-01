"""Shared preprocessing variants, empirically determined (no internet access
to check official docs) for models whose export convention differs from the
standard torchvision/timm ImageNet normalization that eval_pretrained_onnx.py
defaults to. Each was validated on a 40-image sample against several
candidates -- see the project notes for the probe results:

  inception_v1: zoo_raw_bgr_chw (70.0%)   squeezenet: zoo_raw_bgr_chw (70.0%)
  shufflenet_v1: standard_chw (72.5%)     shufflenet_v2: standard_chw (77.5%)
  efficientnet_lite4: tf_neg1_1_nhwc (77.5%)
  inception_v2: UNRESOLVED (best guess only 2.5%) -- excluded from the dataset.
"""
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def standard_chw(img):
    img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def zoo_raw_bgr_chw(img):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    return arr.transpose(2, 0, 1)


def tf_neg1_1_nhwc(img):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr  # already HWC (this model's input layout is NHWC)


VARIANTS = {
    "standard_chw": standard_chw,
    "zoo_raw_bgr_chw": zoo_raw_bgr_chw,
    "tf_neg1_1_nhwc": tf_neg1_1_nhwc,
}

# Which variant each ONNX Model Zoo (arch, variant) pair actually needs.
MODEL_PREPROCESS = {
    ("inception_v1", "onnx_model_zoo"): "zoo_raw_bgr_chw",
    ("shufflenet_v1", "onnx_model_zoo"): "standard_chw",
    ("shufflenet_v2", "onnx_model_zoo"): "standard_chw",
    ("squeezenet", "onnx_model_zoo"): "zoo_raw_bgr_chw",
    ("efficientnet_lite4", "onnx_model_zoo"): "tf_neg1_1_nhwc",
}