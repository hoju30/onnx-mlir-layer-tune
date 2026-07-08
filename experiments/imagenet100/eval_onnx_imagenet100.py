import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on ImageNet-100 validation set")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./imagenet100_hf/validation",
        help="Path to ImageFolder-style validation directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help='ONNX Runtime providers, e.g. "CPUExecutionProvider" or "CUDAExecutionProvider CPUExecutionProvider"',
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved for future DataLoader version; currently unused",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Model input spatial size after crop",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Resize shorter side before center crop",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="eval_results_imagenet100.json",
        help="Where to save summary JSON",
    )
    return parser.parse_args()


def build_transform(resize_size: int, input_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int):
    topk = np.argsort(logits, axis=1)[:, -k:][:, ::-1]
    correct = 0
    for i in range(labels.shape[0]):
        if labels[i] in topk[i]:
            correct += 1
    return correct


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_root = Path(args.data_root)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    transform = build_transform(
        resize_size=args.resize_size,
        input_size=args.input_size,
        mean=args.mean,
        std=args.std,
    )

    dataset = datasets.ImageFolder(root=str(data_root), transform=transform)

    print("=" * 80)
    print("Dataset Info")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Num classes: {len(dataset.classes)}")
    print(f"Num samples: {len(dataset.samples)}")
    print(f"First classes: {dataset.classes[:10]}")

    session = ort.InferenceSession(str(model_path), providers=args.providers)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name
    input_shape = input_meta.shape
    input_type = input_meta.type

    print("\n" + "=" * 80)
    print("ONNX Runtime Session Info")
    print("=" * 80)
    print(f"Providers: {session.get_providers()}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Input type: {input_type}")
    print(f"Output name: {output_name}")

    # preload all (path, label) so order is stable
    samples = dataset.samples

    total_images = 0
    top1_correct = 0
    top5_correct = 0
    total_infer_time = 0.0

    start_total = time.perf_counter()

    def sample_generator():
        for path, label in samples:
            try:
                img = Image.open(path).convert("RGB")
                tensor = transform(img).numpy().astype(np.float32)
                yield path, tensor, label
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")

    for batch_idx, batch in enumerate(batched(sample_generator(), args.batch_size), start=1):
        batch_paths = [x[0] for x in batch]
        batch_images = np.stack([x[1] for x in batch], axis=0)
        batch_labels = np.array([x[2] for x in batch], dtype=np.int64)

        infer_start = time.perf_counter()
        outputs = session.run([output_name], {input_name: batch_images})
        infer_end = time.perf_counter()

        logits = outputs[0]
        if not isinstance(logits, np.ndarray):
            logits = np.asarray(logits)

        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits [N, C], got shape {logits.shape}")

        batch_infer_time = infer_end - infer_start
        total_infer_time += batch_infer_time

        top1_correct += topk_accuracy(logits, batch_labels, k=1)
        top5_correct += topk_accuracy(logits, batch_labels, k=min(5, logits.shape[1]))
        total_images += batch_labels.shape[0]

        if batch_idx % 50 == 0 or total_images == len(samples):
            current_top1 = 100.0 * top1_correct / max(total_images, 1)
            current_top5 = 100.0 * top5_correct / max(total_images, 1)
            avg_ms = (total_infer_time / max(total_images, 1)) * 1000.0
            print(
                f"[{total_images:5d}/{len(samples)}] "
                f"Top-1: {current_top1:.4f}% | "
                f"Top-5: {current_top5:.4f}% | "
                f"Avg infer: {avg_ms:.4f} ms/image"
            )

    end_total = time.perf_counter()
    total_wall_time = end_total - start_total

    top1_acc = 100.0 * top1_correct / max(total_images, 1)
    top5_acc = 100.0 * top5_correct / max(total_images, 1)
    avg_infer_ms = (total_infer_time / max(total_images, 1)) * 1000.0

    summary = {
        "model": str(model_path),
        "data_root": str(data_root),
        "providers": session.get_providers(),
        "num_classes": len(dataset.classes),
        "total_images": total_images,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "average_inference_time_ms_per_image": avg_infer_ms,
        "total_wall_time_sec": total_wall_time,
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": input_shape,
        "input_type": input_type,
    }

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Backend: ONNX Runtime")
    print(f"Providers: {session.get_providers()}")
    print(f"Data root: {data_root}")
    print(f"Total Images Processed: {total_images}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}%")
    print(f"Top-5 Accuracy: {top5_acc:.4f}%")
    print(f"Average Inference Time: {avg_infer_ms:.4f} ms/image")
    print(f"Total Wall Time: {total_wall_time:.2f} seconds")

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()