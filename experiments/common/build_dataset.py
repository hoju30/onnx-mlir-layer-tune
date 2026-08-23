#!/usr/bin/env python3
"""
Dataset builder (claude.md section 16): turns a sample_configs.py config list
into the labeled dataset the GNN accuracy predictor (section 11) trains on.
For every complete configuration it actually compiles+runs it (Real Posit
Evaluator, section 20), builds its Posit-Dialect graph (build_posit_graph.py),
computes static cost (cost_from_graph.py), records the real measured
whole-model accuracy loss as the regression label, and finally vectorizes
every graph in the batch TOGETHER (vectorize_posit_graph.py fits norm stats
jointly across all graphs passed to it, so it must run once over the whole
batch, not per-sample).

Only --scale mnist is implemented. A config whose non-FP32 nodes span more
than one distinct (nbits, es) needs the full/mixed posit runtime (every
format's kernels compiled in) instead of the single-format runtime the
existing MNIST fast-path scripts hardcode for speed (eval_variants.compile_so,
sweep_node_sensitivity._compile_so_for_format) -- a single-format runtime
link-fails on a genuinely mixed IR (undefined _mlir_ciface_posit_from_f32_p*e*
symbols for whichever formats it wasn't built with). This module's own
_compile_so_mixed() builds the full runtime on demand for those samples
(mirrors build_model11_sos.sh's default --runtime-format-scope=full); every
such sample is flagged mixed_runtime=true in the manifest, but its accuracy
is exact real-evaluated, not approximated.

Usage:
  python sample_configs.py --op-features docs/mnist_example/mnist_op_features.json \
      --risk-calibration docs/mnist_example/mnist_risk_calibration.json \
      --n-random 24 --out docs/mnist_example/mnist_sampled_configs.json

  python build_dataset.py --scale mnist \
      --configs docs/mnist_example/mnist_sampled_configs.json \
      --op-features docs/mnist_example/mnist_op_features.json \
      --measurements docs/mnist_example/mnist_whole_model_measurements.json \
      --out-dir docs/mnist_example/dataset --limit 2000
"""
import argparse
import json
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMMON)


def node_formats_str(formats):
    parts = []
    for node, fmt in formats.items():
        if fmt == "FP32":
            continue
        nbits, es = fmt.replace("posit_", "").split("_")
        parts.append(f"{node}:{nbits}:{es}")
    return ",".join(parts)


def used_formats(formats):
    out = set()
    for fmt in formats.values():
        if fmt != "FP32":
            nbits, es = fmt.replace("posit_", "").split("_")
            out.add((int(nbits), int(es)))
    return out


def config_desc(formats):
    quantized = {n: f for n, f in formats.items() if f != "FP32"}
    if not quantized:
        return "all FP32"
    return "; ".join(f"{n} -> {f}" for n, f in quantized.items())


def _compile_so_mixed(mv, ll_path, so_path):
    """Full/mixed posit runtime (no POSIT_RUNTIME_SINGLE_FORMAT / BUILD_MIXED_OFF):
    every format's kernels are compiled in, needed when a config's non-FP32
    nodes span more than one distinct (nbits, es) -- a single-format runtime
    (eval_variants.compile_so / sweep_node_sensitivity._compile_so_for_format)
    can only resolve one format's _mlir_ciface_posit_from_f32_p*e* symbols and
    link-fails on a genuinely mixed IR. This matches build_model11_sos.sh's
    default --runtime-format-scope=full behavior, just done directly in
    Python for the MNIST fast path instead of shelling out to that script.
    Slower to compile than the single-format path, so only used when needed."""
    rt_obj = so_path + ".runtime.o"
    common = [
        "-DPOSIT_USE_UNIVERSAL",
        f"-I{mv.UNIVERSAL_I}", f"-I{mv.MLIR_I}", "-I/usr/local/include",
    ]
    mv.run([mv.CLANGXX, "-std=c++20", "-O3", "-fPIC", "-c",
            "-ffunction-sections", "-fdata-sections",
            "-fvisibility=hidden", "-fvisibility-inlines-hidden",
            mv.RUNTIME_CPP, *common, "-o", rt_obj])
    mv.run([mv.CLANGXX, "-std=c++20", "-O3", "-fPIC", "-shared",
            "-Wl,--gc-sections",
            ll_path, rt_obj, *common, mv.CRUNTIME,
            "-o", so_path])
    os.remove(rt_obj)
    return so_path


def build_and_eval_mnist(mv, cfg, dataset_dir, images, labels, fp32):
    """Returns (posit_mlir_path, {"top1":..., "top5":...}, mixed_runtime)."""
    fmts = used_formats(cfg["formats"])
    cid = cfg["id"]

    if not fmts:
        # no node converted: reuse the plain ONNX IR verbatim under a
        # sample-specific name so its whole_model_measurements.json key
        # (basename) doesn't collide with other consumers of m.onnx.mlir.
        posit_mlir = os.path.join(dataset_dir, f"{cid}.posit.mlir")
        with open(os.path.join(mv.PP, "m.onnx.mlir")) as src, open(posit_mlir, "w") as dst:
            dst.write(src.read())
        return posit_mlir, {"top1": fp32["top1"], "top5": fp32["top5"]}, False

    from sweep_node_sensitivity import _compile_so_for_format

    nqf = node_formats_str(cfg["formats"])
    nbits, es = max(fmts, key=lambda x: x[0])
    mixed_runtime = len(fmts) > 1

    onnx_ir = os.path.join(mv.PP, "m.onnx.mlir")
    ll_path = os.path.join(dataset_dir, f"{cid}.ll")
    posit_mlir = os.path.join(dataset_dir, f"{cid}.posit.mlir")
    mv.posit_lower_to_ll(
        onnx_ir, ll_path,
        env={"POSIT_NODE_FORMATS": nqf, "POSIT_COMPACT_CONSTANTS": "1"},
        extra_opt_flags=["--shape-inference", "--convert-onnx-to-posit",
                          f"--posit-format=p{nbits}e{es}"])

    so_path = os.path.join(dataset_dir, f"{cid}.so")
    if mixed_runtime:
        _compile_so_mixed(mv, ll_path, so_path)
    else:
        _compile_so_for_format(mv, ll_path, so_path, nbits, es)
    r = mv.eval_so(so_path, images, labels, "m")
    return posit_mlir, r, mixed_runtime


def update_measurements(path, ir_basename, so_name, desc, r, fp32):
    with open(path) as f:
        data = json.load(f)
    data["configs"][ir_basename] = {
        "so": so_name,
        "config_desc": desc,
        "top1": r["top1"], "top5": r["top5"],
        "delta_top1_vs_fp32": fp32["top1"] - r["top1"],
        "delta_top5_vs_fp32": fp32["top5"] - r["top5"],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["mnist"], default="mnist")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=2000,
                    help="MNIST test images to evaluate per config (10000 = full test set)")
    ap.add_argument("--vectorize-out-dir", default=None, help="defaults to --out-dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vec_out_dir = args.vectorize_out_dir or args.out_dir
    os.makedirs(vec_out_dir, exist_ok=True)

    sys.path.insert(0, os.path.join(PROJECT, "docs", "mnist_example"))
    import eval_variants as mv
    from cost_from_graph import compute_cost

    configs = json.load(open(args.configs))["configs"]
    fp32 = json.load(open(args.measurements))["fp32_baseline"]

    images, labels = mv.load_mnist_test()
    if args.limit:
        images, labels = images[:args.limit], labels[:args.limit]

    manifest = []
    graph_files = []
    for cfg in configs:
        cid = cfg["id"]
        print(f"\n=== [{cid}] {cfg['strategy']} ===", flush=True)
        posit_mlir, r, mixed_runtime = build_and_eval_mnist(
            mv, cfg, args.out_dir, images, labels, fp32)

        ir_basename = os.path.basename(posit_mlir)
        so_name = ir_basename.replace(".posit.mlir", ".so")
        update_measurements(args.measurements, ir_basename, so_name,
                            config_desc(cfg["formats"]), r, fp32)
        print(f"  top1={r['top1']*100:.2f}% "
              f"(delta={(fp32['top1'] - r['top1'])*100:+.2f}pp)"
              f"{'  [full/mixed runtime]' if mixed_runtime else ''}", flush=True)

        graph_out = os.path.join(args.out_dir, f"{cid}.graph.json")
        subprocess.run([sys.executable, os.path.join(COMMON, "build_posit_graph.py"),
                        posit_mlir, "--op-features", args.op_features,
                        "--whole-model-measurements", args.measurements,
                        "--out", graph_out], check=True, cwd=PROJECT)
        graph_files.append(graph_out)

        graph = json.load(open(graph_out))
        cost = compute_cost(graph)
        cost_out = os.path.join(args.out_dir, f"{cid}.cost.json")
        with open(cost_out, "w") as f:
            json.dump(cost, f, indent=2)

        manifest.append({
            "id": cid, "strategy": cfg["strategy"], "formats": cfg["formats"],
            "ir_file": posit_mlir, "graph_file": graph_out, "cost_file": cost_out,
            "top1": r["top1"], "top5": r["top5"],
            "delta_top1_vs_fp32": fp32["top1"] - r["top1"],
            "delta_top5_vs_fp32": fp32["top5"] - r["top5"],
            "weight_bytes": cost["weight_bytes"],
            "peak_activation_bytes": cost["peak_activation_bytes"],
            "mixed_runtime": mixed_runtime,
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest for {len(manifest)} samples to {manifest_path}")

    print(f"\n=== vectorizing {len(graph_files)} graphs together (shared norm stats) ===", flush=True)
    subprocess.run([sys.executable, os.path.join(COMMON, "vectorize_posit_graph.py"),
                    *graph_files, "--out-dir", vec_out_dir], check=True, cwd=PROJECT)


if __name__ == "__main__":
    main()
