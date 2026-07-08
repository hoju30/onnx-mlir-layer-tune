import argparse
import copy
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--output-dir", type=str, default="./imagenet100_resnet18_out")
    p.add_argument("--model-name", type=str, default="resnet18")
    return p.parse_args()


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item())
        return res


def run_eval(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0.0
    top5 = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            bsz = labels.size(0)
            total += bsz
            total_loss += loss.item() * bsz
            a1, a5 = accuracy(logits, labels, topk=(1, 5))
            top1 += a1
            top5 += a5

    return {
        "loss": total_loss / total,
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tf)

    assert len(train_ds.classes) == len(val_ds.classes), "train/val class count mismatch"
    num_classes = len(train_ds.classes)
    print("num_classes:", num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_top1 = 0.0
        train_top5 = 0.0

        start = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bsz = labels.size(0)
            train_total += bsz
            train_loss += loss.item() * bsz
            a1, a5 = accuracy(logits, labels, topk=(1, 5))
            train_top1 += a1
            train_top5 += a5

        scheduler.step()
        val_metrics = run_eval(model, val_loader, device, criterion)
        elapsed = time.perf_counter() - start

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_top1": 100.0 * train_top1 / train_total,
            "train_top5": 100.0 * train_top5 / train_total,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "elapsed_sec": elapsed,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train top1 {epoch_result['train_top1']:.2f}% | "
            f"val top1 {epoch_result['val_top1']:.2f}% | "
            f"val top5 {epoch_result['val_top5']:.2f}%"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best.pth")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    class_map = {
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
    }
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print("best val top1:", best_top1)
    print("saved:", out_dir / "best.pth")


if __name__ == "__main__":
    main()
