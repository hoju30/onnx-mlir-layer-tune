import argparse
import json
from pathlib import Path

import timm
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./imagenet100_mobilenetv2_out/best.pth")
    p.add_argument("--class-map", type=str, default="./imagenet100_mobilenetv2_out/class_map.json")
    p.add_argument("--onnx-out", type=str, default="./model/imagenet100_mobilenetv2.onnx")
    p.add_argument("--model-name", type=str, default="mobilenetv2_100")
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.class_map, "r") as f:
        class_map = json.load(f)

    num_classes = len(class_map["classes"])
    model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_out = Path(args.onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_out),
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
    )
    print("saved to", onnx_out)


if __name__ == "__main__":
    main()
