#!/usr/bin/env python3
"""Dataset builder for the low-precision variant (claude_lowprecision.md
section 16, mirroring build_dataset.py): turns a sample_configs_lowp.py
config list into the labeled dataset the GNN accuracy predictor trains on.

For every configuration it actually compiles+runs it (Real Low-Precision
Evaluator, section 20) via --convert-onnx-to-lowprecision, records the real
measured whole-model accuracy as the regression label, overlays the chosen
formats onto the ONE shared base graph (build_lowp_graph.py -- topology is
configuration-invariant here, unlike Posit, so it is extracted once up
front rather than per sample), and computes static cost
(cost_from_graph_lowp.py).

int8/fp8e4m3/fp8e5m2 nodes need calibration scale/zero-point baked into
LOWP_NODE_FORMATS (the pass does no calibration itself) -- these come from
calibrate_lowprecision.py's output, keyed by (node, format).

Usage:
  python sample_configs_lowp.py --out docs/mnist_example/mnist_lowp_configs.json
  python calibrate_lowprecision.py --out docs/mnist_example/mnist_lowp_calibration.json
  python build_dataset_lowp.py \
      --configs docs/mnist_example/mnist_lowp_configs.json \
      --calibration docs/mnist_example/mnist_lowp_calibration.json \
      --op-features docs/mnist_example/mnist_op_features.json \
      --out-dir docs/mnist_example/dataset_lowp --limit 2000
"""
import argparse
import json
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON)

QUANT_FORMATS = {"int8", "fp8e4m3", "fp8e5m2"}


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


def config_desc(formats):
    quantized = {n: f for n, f in formats.items() if f != "FP32"}
    if not quantized:
        return "all FP32"
    return "; ".join(f"{n} -> {f}" for n, f in quantized.items())


def build_and_eval_mnist(mv, cfg, calibration, dataset_dir, images, labels, fp32):
    cid = cfg["id"]
    formats = cfg["formats"]
    nqf = node_formats_str(formats, calibration)

    ll_path = os.path.join(dataset_dir, f"{cid}.ll")
    so_path = os.path.join(dataset_dir, f"{cid}.so")

    if not nqf:
        # no node converted: reuse the plain fp32 baseline result directly.
        return {"top1": fp32["top1"], "top5": fp32["top5"]}

    onnx_ir = os.path.join(mv.PP, "m.onnx.mlir")
    mv.lowprecision_lower_to_ll(onnx_ir, ll_path, env={"LOWP_NODE_FORMATS": nqf})
    mv.compile_so_lowprecision(ll_path, so_path)
    return mv.eval_so(so_path, images, labels, "m")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--op-features", default="docs/mnist_example/mnist_op_features.json")
    ap.add_argument("--weight-source-ir", default="docs/mnist_example/pp/m.mixed.posit.mlir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=2000,
                    help="MNIST test images to evaluate per config (10000 = full test set)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(PROJECT, "docs", "mnist_example"))
    import eval_variants as mv
    from build_lowp_graph import build_base_graph, apply_config
    from cost_from_graph_lowp import compute_cost

    configs = json.load(open(args.configs))["configs"]
    calibration = json.load(open(args.calibration))

    images, labels = mv.load_mnist_test()
    if args.limit:
        images, labels = images[:args.limit], labels[:args.limit]

    print("=== building fp32 baseline ===", flush=True)
    fp32_so = os.path.join(mv.OUT, "mnist_fp32.so")
    if not os.path.exists(fp32_so):
        mv.run([mv.ONNXMLIR, "--EmitLib", "-o", fp32_so.replace(".so", ""), mv.ONNX])
    fp32 = mv.eval_so(fp32_so, images, labels, "")

    print("=== extracting shared base graph (topology is configuration-invariant) ===", flush=True)
    base_graph = build_base_graph(args.op_features, args.weight_source_ir, num_calibration=512)

    manifest = []
    for cfg in configs:
        cid = cfg["id"]
        print(f"\n=== [{cid}] {cfg['strategy']}: {config_desc(cfg['formats'])} ===", flush=True)
        r = build_and_eval_mnist(mv, cfg, calibration, args.out_dir, images, labels, fp32)
        print(f"  top1={r['top1']*100:.2f}% (delta={(fp32['top1']-r['top1'])*100:+.2f}pp)", flush=True)

        graph = apply_config(base_graph, cfg["formats"])
        graph_out = os.path.join(args.out_dir, f"{cid}.graph.json")
        with open(graph_out, "w") as f:
            json.dump(graph, f, indent=2)
        cost = compute_cost(graph)

        manifest.append({
            "id": cid, "strategy": cfg["strategy"], "formats": cfg["formats"],
            "graph_file": graph_out,
            "top1": r["top1"], "top5": r["top5"],
            "delta_top1_vs_fp32": fp32["top1"] - r["top1"],
            "delta_top5_vs_fp32": fp32["top5"] - r["top5"],
            "weight_bytes": cost["weight_bytes"],
            "peak_activation_bytes": cost["peak_activation_bytes"],
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"fp32_baseline": fp32, "samples": manifest}, f, indent=2)
    print(f"\nwrote manifest for {len(manifest)} samples to {manifest_path}")


if __name__ == "__main__":
    main()
