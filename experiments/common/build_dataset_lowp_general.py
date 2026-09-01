#!/usr/bin/env python3
"""Dataset builder for the low-precision variant on arbitrary ONNX models
(ResNet18/MobileNetV2-scale), the general counterpart to build_dataset_lowp.py's
MNIST-hardcoded version.

Real-compiles and runs every sampled configuration (Real Low-Precision
Evaluator, claude_lowprecision.md section 20) via onnx-mlir, same as the
MNIST version, but with one crucial difference established experimentally:
bf16/f16 have NO vectorized Krnl kernel for Gemm/Conv/MatMul in this
codebase (only FP32/FP64 do; int8/fp8 use their own dedicated
QLinearConv/QLinearMatMul kernels and are fast, ~0.3s/image on ResNet18).
bf16/f16 run ~20s/image regardless of --march/--O3 (confirmed: identical
timing with SIMD forced on, and a same-pipeline pure-FP32 control matches
the standard onnx-mlir --EmitLib driver's speed, so the pipeline itself
isn't the bottleneck -- see project notes). Properly fixing this needs a new
vectorized bf16 Gemm/Conv/MatMul Krnl lowering kernel, out of scope here.

So this uses an ADAPTIVE per-config image count: configs containing no
bf16/f16 assignment get --n-fast images (cheap, int8/fp8/FP32 are all fast);
configs containing any bf16/f16 assignment get only --n-slow images (keeps
wall-clock bounded while still covering the full 6-format research space
per the project's explicit choice to accept noisier bf16/f16 accuracy
numbers rather than drop those formats or invest in a new kernel).

Usage:
  python build_dataset_lowp_general.py \
      --onnx experiments/imagenet100/model_dataset/resnet18/torchvision_v1/model.onnx \
      --op-features experiments/imagenet100/model/resnet18_op_features.json \
      --configs experiments/imagenet100/model/resnet18_lowp_configs.json \
      --calibration experiments/imagenet100/model/resnet18_lowp_calibration.json \
      --fp32-baseline experiments/imagenet100/model_dataset/resnet18/torchvision_v1/fp32_baseline.json \
      --label-map experiments/imagenet100/imagenet100_label_map.json \
      --out-dir experiments/imagenet100/model/resnet18_dataset_lowp \
      --n-fast 200 --n-slow 8
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON)

QUANT_FORMATS = {"int8", "fp8e4m3", "fp8e5m2"}
SLOW_FORMATS = {"bf16", "f16"}

ONNX_MLIR = os.path.join(PROJECT, "build/Release/bin/onnx-mlir")
OPT = os.path.join(PROJECT, "build/Release/bin/onnx-mlir-opt")
TRANSLATE = "/home/hoju/test/llvm-project-1053047/build/bin/mlir-translate"
CLANGXX = "/home/hoju/test/llvm-project-1053047/build/bin/clang++"
CRUNTIME = os.path.join(PROJECT, "build/Release/lib/libcruntime.a")


def node_formats_str(formats, calibration):
    parts = []
    for node, fmt in formats.items():
        if fmt == "FP32":
            continue
        if fmt in QUANT_FORMATS:
            p = calibration["nodes"][node][fmt]
            parts.append(f"{node}:{fmt}:{p['x_scale']}:{p['x_zero_point']}:{p['y_scale']}:{p['y_zero_point']}")
        else:
            parts.append(f"{node}:{fmt}")
    return ",".join(parts)


def is_slow_config(formats):
    return any(f in SLOW_FORMATS for f in formats.values())


def config_desc(formats):
    quantized = {n: f for n, f in formats.items() if f != "FP32"}
    if not quantized:
        return "all FP32"
    n = len(quantized)
    sample = "; ".join(f"{k}->{v}" for k, v in list(quantized.items())[:3])
    return f"{n} node(s) tuned: {sample}{' ...' if n > 3 else ''}"


EVAL_CHUNK_SIZE = 20  # measured: OMExecutionSession.run() leaks ~305MB PER CALL
                       # (confirmed directly: 9.2GB RSS after 30 calls in one
                       # process, and separately via dmesg -- a fresh
                       # subprocess doing 200 sess.run() calls was itself
                       # OOM-killed at 26GB). Per-config subprocess isolation
                       # alone isn't enough; each subprocess must also only
                       # run a bounded number of images before exiting so the
                       # OS can reclaim the leak. 20 calls -> ~6GB peak, safe.


def eval_so(so_path, tag, image_indices, label_map_path, preprocess_variant=None):
    """Runs inference in FRESH subprocesses (eval_so_subprocess.py), chunked
    to EVAL_CHUNK_SIZE images per subprocess -- see the native per-call leak
    note above. Returns the overall restricted-top1 across all images."""
    script = os.path.join(COMMON, "eval_so_subprocess.py")
    correct_total = 0
    n_total = 0
    for start in range(0, len(image_indices), EVAL_CHUNK_SIZE):
        chunk = image_indices[start:start + EVAL_CHUNK_SIZE]
        idx_str = ",".join(str(i) for i in chunk)
        cmd = [sys.executable, script, "--so", so_path, "--tag", tag,
               "--image-indices", idx_str, "--label-map", label_map_path]
        if preprocess_variant:
            cmd += ["--preprocess", preprocess_variant]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print("eval_so_subprocess STDERR:", r.stderr[-3000:])
            raise RuntimeError(f"eval_so_subprocess failed for {so_path}")
        result = json.loads(r.stdout.strip().splitlines()[-1])
        correct_total += round(result["top1"] * result["n"])
        n_total += result["n"]
    return correct_total / n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--fp32-baseline", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-fast", type=int, default=200)
    ap.add_argument("--n-slow", type=int, default=8)
    ap.add_argument("--tag", default="m")
    ap.add_argument("--preprocess", default=None,
                    help="override preprocessing variant (see preprocessing_variants.py)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from lowp_lower_general import import_and_convopt, lower_and_compile
    from build_lowp_graph_general import build_base_graph, apply_config
    from cost_from_graph_lowp import compute_cost

    configs = json.load(open(args.configs))["configs"]
    calibration = json.load(open(args.calibration))
    fp32_baseline = json.load(open(args.fp32_baseline))
    label_map = json.load(open(args.label_map))
    local_to_1k = {int(k): v for k, v in label_map["local_to_imagenet1k"].items()}
    restricted_classes = sorted(local_to_1k.values())
    restricted_pos = {c: i for i, c in enumerate(restricted_classes)}

    print("loading clane9/imagenet-100 validation split (index count only) ...", flush=True)
    from datasets import load_dataset
    ds_len = len(load_dataset("clane9/imagenet-100", split="validation"))

    def pick_indices(n):
        return np.linspace(0, ds_len - 1, n).astype(int).tolist()  # spread across all 100 classes

    fast_indices = pick_indices(args.n_fast)
    slow_indices = pick_indices(args.n_slow)
    fp32_restricted_top1 = fp32_baseline["restricted_top1"]

    print("importing model + applying decompose/conv-opt (once) ...", flush=True)
    convopt_path = import_and_convopt(args.onnx, args.out_dir, ONNX_MLIR, OPT, tag=args.tag)

    print("extracting shared base graph (topology is configuration-invariant) ...", flush=True)
    base_graph = build_base_graph(args.op_features, args.onnx)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = []
    done_ids = set()
    if os.path.exists(manifest_path):
        prev = json.load(open(manifest_path))
        manifest = prev["samples"]
        done_ids = {s["id"] for s in manifest}
        print(f"resuming: {len(done_ids)} config(s) already done, skipping them", flush=True)

    for cfg in configs:
        cid = cfg["id"]
        if cid in done_ids:
            continue
        slow = is_slow_config(cfg["formats"])
        n_images = args.n_slow if slow else args.n_fast
        image_indices = slow_indices if slow else fast_indices
        print(f"\n=== [{cid}] {cfg['strategy']}: {config_desc(cfg['formats'])} "
              f"({'SLOW' if slow else 'fast'}, n={n_images}) ===", flush=True)

        skip_reason = None
        if not any(f != "FP32" for f in cfg["formats"].values()):
            top1 = fp32_restricted_top1
            print(f"  all FP32 -> reusing fp32_baseline.json restricted_top1={top1*100:.2f}%", flush=True)
        else:
            nqf = node_formats_str(cfg["formats"], calibration)
            out_base = os.path.join(args.out_dir, cid)
            try:
                t0 = time.time()
                so_path = lower_and_compile(convopt_path, nqf, out_base, OPT, TRANSLATE, CLANGXX, CRUNTIME)
                print(f"  compiled in {time.time()-t0:.1f}s", flush=True)
                t0 = time.time()
                top1 = eval_so(so_path, args.tag, image_indices, args.label_map, args.preprocess)
                print(f"  restricted top1={top1*100:.2f}% (delta={(fp32_restricted_top1-top1)*100:+.2f}pp) "
                      f"[{time.time()-t0:.1f}s for {n_images} images]", flush=True)
            except subprocess.TimeoutExpired as e:
                skip_reason = f"compile/eval timed out after {e.timeout}s (pathological input -- see " \
                              "COMPILE_TIMEOUT_S note in lowp_lower_general.py)"
                top1 = None
                print(f"  SKIPPING: {skip_reason}", flush=True)
            except RuntimeError as e:
                skip_reason = f"compile/eval failed: {e}"
                top1 = None
                print(f"  SKIPPING: {skip_reason}", flush=True)

        graph = apply_config(base_graph, cfg["formats"])
        graph_out = os.path.join(args.out_dir, f"{cid}.graph.json")
        with open(graph_out, "w") as f:
            json.dump(graph, f, indent=2)
        cost = compute_cost(graph)

        manifest.append({
            "id": cid, "strategy": cfg["strategy"], "formats": cfg["formats"],
            "graph_file": graph_out, "n_images": n_images, "slow_config": slow,
            "skip_reason": skip_reason,
            "restricted_top1": top1,
            "delta_restricted_top1_vs_fp32": (fp32_restricted_top1 - top1) if top1 is not None else None,
            "weight_bytes": cost["weight_bytes"],
            "peak_activation_bytes": cost["peak_activation_bytes"],
        })
        # checkpoint after every sample so a slow bf16/f16 run can be
        # interrupted without losing already-completed configs
        with open(manifest_path, "w") as f:
            json.dump({"fp32_baseline": fp32_baseline, "samples": manifest}, f, indent=2)

    print(f"\nwrote manifest for {len(manifest)} samples to {manifest_path}")


if __name__ == "__main__":
    main()
