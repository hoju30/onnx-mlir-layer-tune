#!/usr/bin/env python3
"""
Per-layer posit-format sensitivity sweep: for every posit-format-overridable node
(see extract_op_features.py's "supported" flag), build a variant with only that
node converted (POSIT_NODE_FORMATS=node:nbits:es, everything else stays FP32),
measure real accuracy vs an all-FP32 baseline, and record the delta.

This is the (layer, format) -> measured_error dataset that plan_precision.py's
greedy planner consumes directly, and that a future ML cost model would train on.

Two backends:
  --scale mnist       fast path, reuses docs/mnist_example/eval_variants.py.
  --scale imagenet100 wraps src/bash/build_model11_sos.sh /
                       src/bash/time_model11_dataset_parallel.sh unmodified.

Usage:
  python sweep_node_sensitivity.py --scale mnist \
      --op-features docs/mnist_example/mnist_op_features.json \
      --out docs/mnist_example/mnist_sensitivity_table.csv

  python sweep_node_sensitivity.py --scale imagenet100 --model-name resnet18 \
      --op-features experiments/imagenet100/model/resnet18_op_features.json \
      --onnx experiments/imagenet100/model/imagenet100_resnet18.onnx \
      --out experiments/imagenet100/resnet18_sensitivity_table.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FORMATS = [(8, 1), (16, 1), (32, 1)]

CSV_FIELDS = ["node", "type", "nbits", "es", "top1", "top5",
              "delta_top1_vs_fp32", "delta_top5_vs_fp32"]


def parse_formats(spec):
    out = []
    for tok in spec.split(","):
        n, e = tok.strip().lower().lstrip("p").split("e")
        out.append((int(n), int(e)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MNIST backend
# ─────────────────────────────────────────────────────────────────────────────

def _compile_so_for_format(mv, ll_path, so_path, nbits, es):
    """Like eval_variants.compile_so, but selects the single-format runtime macro
    matching (nbits, es) instead of hardcoding p8e1 -- a format-map sweep point
    needs the runtime compiled for whichever format that one node actually uses."""
    rt_obj = so_path + ".runtime.o"
    fmt_macro = f"POSIT_RUNTIME_FMT_P{nbits}E{es}"
    common = [
        "-DPOSIT_USE_UNIVERSAL",
        "-DPOSIT_RUNTIME_SINGLE_FORMAT=1", f"-D{fmt_macro}=1",
        "-DPOSIT_BUILD_MIXED_OFF=1",
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


def sweep_mnist(op_features_path, formats, out_csv, limit=None):
    sys.path.insert(0, os.path.join(PROJECT, "docs", "mnist_example"))
    import eval_variants as mv

    sweep_dir = os.path.join(mv.PP, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    onnx_ir = os.path.join(mv.PP, "m.onnx.mlir")

    print("=== loading MNIST test set ===", flush=True)
    images, labels = mv.load_mnist_test()
    if limit:
        images, labels = images[:limit], labels[:limit]
    print(f"  loaded {len(images)} test images", flush=True)

    fp32_so = os.path.join(mv.OUT, "mnist_fp32.so")
    if not os.path.exists(fp32_so):
        print("[build] fp32 baseline ...", flush=True)
        mv.run([mv.ONNXMLIR, "--EmitLib", "-o", fp32_so.replace(".so", ""), mv.ONNX])
    baseline = mv.eval_so(fp32_so, images, labels, "")
    print(f"  fp32 baseline: top1={baseline['top1']*100:.2f}% top5={baseline['top5']*100:.2f}%",
          flush=True)

    ops = json.load(open(op_features_path))["ops"]
    targets = [op for op in ops if op["supported"]]

    rows = []
    for op in targets:
        name = op["name"]
        for nbits, es in formats:
            tag = f"{name}_p{nbits}e{es}"
            ll_path = os.path.join(sweep_dir, f"{tag}.ll")
            so_path = os.path.join(sweep_dir, f"{tag}.so")
            print(f"[sweep] {tag} ...", flush=True)
            mv.posit_lower_to_ll(
                onnx_ir, ll_path,
                env={"POSIT_NODE_FORMATS": f"{name}:{nbits}:{es}",
                     "POSIT_COMPACT_CONSTANTS": "1"},
                extra_opt_flags=["--shape-inference", "--convert-onnx-to-posit",
                                 f"--posit-format=p{nbits}e{es}"])
            _compile_so_for_format(mv, ll_path, so_path, nbits, es)
            r = mv.eval_so(so_path, images, labels, "m")
            row = {
                "node": name, "type": op["type"], "nbits": nbits, "es": es,
                "top1": r["top1"], "top5": r["top5"],
                "delta_top1_vs_fp32": baseline["top1"] - r["top1"],
                "delta_top5_vs_fp32": baseline["top5"] - r["top5"],
            }
            rows.append(row)
            print(f"    top1={r['top1']*100:.2f}% "
                  f"(delta={row['delta_top1_vs_fp32']*100:+.2f}pp)", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} sweep points to {out_csv}", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet100 backend
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def sweep_imagenet100(model_name, onnx_path, op_features_path, formats, out_csv,
                       out_dir, txt_dir, limit, jobs):
    build_script = os.path.join(PROJECT, "src/bash/build_model11_sos.sh")
    time_script = os.path.join(PROJECT, "src/bash/time_model11_dataset_parallel.sh")
    os.makedirs(out_dir, exist_ok=True)

    ops = json.load(open(op_features_path))["ops"]
    targets = [op for op in ops if op["supported"]]

    # one shared fp32 baseline .so, built once (a --posit-formats value is required
    # by the script even though only the f32 baseline output is actually used here)
    print("[build] nqdq-f32 baseline ...", flush=True)
    placeholder_fmt = f"p{formats[0][0]}e{formats[0][1]}"
    _run([build_script, "--model-name", model_name, "--nqdq-onnx", onnx_path,
          "--out-dir", out_dir, "--posit-source", "nqdq",
          "--posit-formats", placeholder_fmt, "--no-stage-logs"])
    baseline_so = os.path.join(out_dir, f"{model_name}-nqdq-f32.so")
    suffixes = ["nqdq-f32"]

    for op in targets:
        name = op["name"]
        for nbits, es in formats:
            fmt = f"p{nbits}e{es}"
            node_suffix = f"nqdq-{name}-{fmt}"  # matches time script's {model_name}-{suffix}.so lookup
            env = dict(os.environ)
            env["POSIT_NODE_FORMATS"] = f"{name}:{nbits}:{es}"
            print(f"[build] {node_suffix} ...", flush=True)
            _run([build_script, "--model-name", model_name, "--nqdq-onnx", onnx_path,
                  "--out-dir", out_dir, "--posit-source", "nqdq",
                  "--posit-formats", fmt, "--skip-f32-baselines",
                  "--continue-on-posit-fail", "--no-stage-logs"], env=env)
            built = os.path.join(out_dir, f"{model_name}-nqdq-{fmt}.so")
            renamed = os.path.join(out_dir, f"{model_name}-{node_suffix}.so")
            if os.path.exists(built):
                os.replace(built, renamed)
                suffixes.append(node_suffix)
            else:
                print(f"    WARNING: build did not produce {built}, skipping", flush=True)

    log_prefix = os.path.join(out_dir, f"{model_name}-11")
    _run([time_script, "--model-name", model_name, "--out-dir", out_dir,
          "--txt-dir", txt_dir, "--limit", str(limit), "--jobs", str(jobs),
          "--suffixes", ",".join(suffixes), "--baseline", "nqdq-f32",
          "--qalign-auto", "off",
          "--format-log-b", f"{log_prefix}.format_summary.B.log"])

    # parse the per-format summary log (real header sample:
    # "#ts_ms\tkey\tformat\t...\tgt_top1_pct\tgt_top5_pct") into the MNIST-compatible schema
    rows = []
    summary_path = f"{log_prefix}.format_summary.B.log"
    with open(summary_path) as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        parsed = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]

    # gt_top1_pct/gt_top5_pct in the raw log are percentages (0-100); normalize to
    # fractions (0-1) so this CSV's units match the MNIST backend's top1/top5.
    baseline_fields = next((r for r in parsed if r.get("format") == "nqdq-f32"), None)
    baseline_top1 = float(baseline_fields["gt_top1_pct"]) / 100.0 if baseline_fields else None
    baseline_top5 = float(baseline_fields["gt_top5_pct"]) / 100.0 if baseline_fields else None

    for fields in parsed:
        suffix = fields.get("format", "")
        if suffix == "nqdq-f32":
            continue
        # suffix format: "nqdq-{node_name}-p{nbits}e{es}"
        without_prefix = suffix.removeprefix("nqdq-")
        node, fmt = without_prefix.rsplit("-", 1)
        nbits_es = fmt.lstrip("p").split("e")
        top1 = float(fields["gt_top1_pct"]) / 100.0
        top5 = float(fields["gt_top5_pct"]) / 100.0
        rows.append({
            "node": node, "type": "", "nbits": nbits_es[0], "es": nbits_es[1],
            "top1": top1, "top5": top5,
            "delta_top1_vs_fp32": (baseline_top1 - top1) if baseline_top1 is not None else "",
            "delta_top5_vs_fp32": (baseline_top5 - top5) if baseline_top5 is not None else "",
            "raw_format_summary_fields": json.dumps(fields),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS + ["raw_format_summary_fields"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} sweep points to {out_csv} "
          f"(raw per-format log: {summary_path})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["mnist", "imagenet100"], required=True)
    ap.add_argument("--op-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", default="p8e1,p16e1,p32e1")
    ap.add_argument("--limit", type=int, default=None)
    # imagenet100-only
    ap.add_argument("--model-name")
    ap.add_argument("--onnx")
    ap.add_argument("--out-dir")
    ap.add_argument("--txt-dir", default=os.path.join(PROJECT, "experiments/imagenet100/val_224_txt"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    formats = parse_formats(args.formats)

    if args.scale == "mnist":
        sweep_mnist(args.op_features, formats, args.out, limit=args.limit)
    else:
        assert args.model_name and args.onnx and args.out_dir, \
            "--scale imagenet100 requires --model-name --onnx --out-dir"
        sweep_imagenet100(args.model_name, args.onnx, args.op_features, formats,
                           args.out, args.out_dir, args.txt_dir,
                           limit=args.limit or 300, jobs=args.jobs)


if __name__ == "__main__":
    main()
