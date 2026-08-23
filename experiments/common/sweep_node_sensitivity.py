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

By default the same --formats list is measured on every node. Pass
--candidates <topk_candidates.json> (see topk_candidates.py) to measure only
each node's Top-K formats instead -- sweep cost then scales with K per node
rather than |formats| x all nodes.

Usage:
  python sweep_node_sensitivity.py --scale mnist \
      --op-features docs/mnist_example/mnist_op_features.json \
      --out docs/mnist_example/mnist_sensitivity_table.csv

  python sweep_node_sensitivity.py --scale mnist \
      --op-features docs/mnist_example/mnist_op_features.json \
      --candidates docs/mnist_example/mnist_topk_candidates.json \
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


def load_candidates(path):
    """topk_candidates.py output -> {node: [(nbits, es), ...]}, FP32 excluded
    (FP32 is the always-available default; there's nothing to build/measure)."""
    data = json.load(open(path))
    return {
        node: [(c["nbits"], c["es"]) for c in entry["candidates"]
               if c.get("format") != "FP32"]
        for node, entry in data["nodes"].items()
    }


def formats_for(name, formats, formats_by_node):
    if formats_by_node is not None:
        return formats_by_node.get(name, [])
    return formats


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


def load_done(out_csv):
    """(node, nbits, es) pairs already recorded in an existing out_csv, so a
    sweep interrupted partway (e.g. killed) can resume instead of losing
    everything and redoing already-measured variants."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv) as f:
        return {(r["node"], int(r["nbits"]), int(r["es"])) for r in csv.DictReader(f)}


def sweep_mnist(op_features_path, formats, out_csv, limit=None, formats_by_node=None):
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

    done = load_done(out_csv)
    if done:
        print(f"  resuming: {len(done)} (node, nbits, es) points already in {out_csv}", flush=True)
    write_header = not os.path.exists(out_csv)

    rows = []
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for op in targets:
            name = op["name"]
            for nbits, es in formats_for(name, formats, formats_by_node):
                if (name, nbits, es) in done:
                    continue
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
                w.writerow(row)
                f.flush()
                print(f"    top1={r['top1']*100:.2f}% "
                      f"(delta={row['delta_top1_vs_fp32']*100:+.2f}pp)", flush=True)

    print(f"\nwrote {len(rows)} new sweep points to {out_csv} "
          f"({len(done) + len(rows)} total)", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet100 backend
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# build_model11_sos.sh tries to rpath-embed libomp automatically by looking under
# "<clang++ dir>/../runtimes/runtimes-bins/openmp/runtime/src", which doesn't
# exist for this LLVM build layout -- so it silently skips the rpath and the
# compiled .so ends up with an unresolved libomp.so dependency (dlopen fails at
# runtime, only for posit variants; the plain FP32 .so has no OpenMP dependency).
# Setting LD_LIBRARY_PATH here works around it without touching the shared script.
_OMP_LIB_DIR = "/home/hoju/test/llvm-project/build/lib"


def _env_with_omp_lib_dir(base_env=None):
    env = dict(base_env if base_env is not None else os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_OMP_LIB_DIR}:{existing}" if existing else _OMP_LIB_DIR
    return env


def sweep_imagenet100(model_name, onnx_path, qdq_onnx_path, op_features_path, formats, out_csv,
                       out_dir, txt_dir, limit, jobs, formats_by_node=None,
                       input_shape="1x3x224x224", label_map=None,
                       image_dir=None, image_preprocess_script=None):
    """txt_dir (val_224_txt) has no reliable ground-truth label association --
    its filenames turned out to be a QDQ-calibration dump, not class-labeled
    (verified: an identity-prefix label-map gave 0.67% Top1 on a model measured
    at ~86% elsewhere). Pass image_dir instead (e.g. imagenet100_hf/validation,
    class-per-subfolder) for real folder-derived ground truth -- no --label-map
    needed then, time_model11_dataset_parallel.sh derives it from the path."""
    build_script = os.path.join(PROJECT, "src/bash/build_model11_sos.sh")
    time_script = os.path.join(PROJECT, "src/bash/time_model11_dataset_parallel.sh")
    os.makedirs(out_dir, exist_ok=True)

    ops = json.load(open(op_features_path))["ops"]
    targets = [op for op in ops if op["supported"]]

    # one shared fp32 baseline .so, built once (a --posit-formats value is required
    # by the script even though only the f32 baseline output is actually used here).
    # build_model11_sos.sh hard-requires --qdq-mlir/--qdq-onnx even when only the
    # nqdq posit path is used (posit-source=nqdq) -- it imports both unconditionally.
    # --skip-output-alps-calib-tool saves ~120s/build: that tool is only ever invoked
    # by time_model11_dataset_parallel.sh's --output-alps-auto path, which we never pass.
    print("[build] nqdq-f32 baseline ...", flush=True)
    placeholder_fmt = f"p{formats[0][0]}e{formats[0][1]}"
    _run([build_script, "--model-name", model_name, "--nqdq-onnx", onnx_path,
          "--qdq-onnx", qdq_onnx_path,
          "--out-dir", out_dir, "--posit-source", "nqdq",
          "--posit-formats", placeholder_fmt, "--no-stage-logs",
          "--skip-output-alps-calib-tool"], env=_env_with_omp_lib_dir())
    baseline_so = os.path.join(out_dir, f"{model_name}-nqdq-f32.so")
    suffixes = ["nqdq-f32"]

    for op in targets:
        name = op["name"]
        for nbits, es in formats_for(name, formats, formats_by_node):
            fmt = f"p{nbits}e{es}"
            node_suffix = f"nqdq-{name}-{fmt}"  # matches time script's {model_name}-{suffix}.so lookup
            env = _env_with_omp_lib_dir()
            env["POSIT_NODE_FORMATS"] = f"{name}:{nbits}:{es}"
            print(f"[build] {node_suffix} ...", flush=True)
            _run([build_script, "--model-name", model_name, "--nqdq-onnx", onnx_path,
                  "--qdq-onnx", qdq_onnx_path,
                  "--out-dir", out_dir, "--posit-source", "nqdq",
                  "--posit-formats", fmt, "--skip-f32-baselines",
                  "--continue-on-posit-fail", "--no-stage-logs",
                  "--skip-output-alps-calib-tool"], env=env)
            built = os.path.join(out_dir, f"{model_name}-nqdq-{fmt}.so")
            renamed = os.path.join(out_dir, f"{model_name}-{node_suffix}.so")
            if os.path.exists(built):
                os.replace(built, renamed)
                suffixes.append(node_suffix)
            else:
                print(f"    WARNING: build did not produce {built}, skipping", flush=True)

    log_prefix = os.path.join(out_dir, f"{model_name}-11")
    time_cmd = [time_script, "--model-name", model_name, "--out-dir", out_dir,
                "--limit", str(limit), "--jobs", str(jobs),
                "--suffixes", ",".join(suffixes), "--baseline", "nqdq-f32",
                "--qalign-auto", "off", "--shape", input_shape,
                "--format-log-b", f"{log_prefix}.format_summary.B.log"]
    if image_dir:
        time_cmd += ["--image-dir", image_dir,
                     "--image-preprocess-script", image_preprocess_script]
    else:
        time_cmd += ["--txt-dir", txt_dir]
        if label_map:
            # without this, gt_top1_pct/gt_top5_pct (and thus delta_top1_vs_fp32) are
            # always nan -- the script only computes GT accuracy when a label-map is given
            time_cmd += ["--label-map", label_map]
    _run(time_cmd, env=_env_with_omp_lib_dir())

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
    ap.add_argument("--candidates",
                     help="topk_candidates.json (see topk_candidates.py) -- if given, "
                          "measures each node's Top-K formats instead of a uniform --formats list")
    ap.add_argument("--limit", type=int, default=None)
    # imagenet100-only
    ap.add_argument("--model-name")
    ap.add_argument("--onnx")
    ap.add_argument("--qdq-onnx",
                     help="build_model11_sos.sh hard-requires this even for a pure nqdq sweep -- "
                          "point it at the model's existing int8-qdq.onnx export")
    ap.add_argument("--out-dir")
    ap.add_argument("--txt-dir",
                     help="DO NOT use for GT accuracy: val_224_txt's filenames don't reliably "
                          "encode class (it's a QDQ-calibration dump) -- prefer --image-dir")
    ap.add_argument("--image-dir",
                     default=os.path.join(PROJECT, "experiments/imagenet100/imagenet100_hf/validation"),
                     help="class-per-subfolder image dir (real ground truth, folder-derived) -- "
                          "the working default for imagenet100 GT accuracy")
    ap.add_argument("--image-preprocess-script",
                     default=os.path.join(PROJECT, "experiments/imagenet100/preprocess_imagenet100_tensor.py"),
                     help="time_model11_dataset_parallel.sh's own default points at a path from "
                          "a different machine's layout (../../../ImageNet100/...) that doesn't "
                          "exist here -- always pass this explicitly for --image-dir mode")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--input-shape", default="1x3x224x224",
                     help="time_model11_dataset_parallel.sh's own --shape default is "
                          "1x1x28x28 (MNIST-shaped) -- must override for imagenet100 models "
                          "or the runtime silently falls back to random input and can segfault")
    ap.add_argument("--label-map",
                     help="only used with --txt-dir (ignored with --image-dir, which derives "
                          "labels from folder structure instead)")
    args = ap.parse_args()

    formats = parse_formats(args.formats)
    formats_by_node = load_candidates(args.candidates) if args.candidates else None

    if args.scale == "mnist":
        sweep_mnist(args.op_features, formats, args.out, limit=args.limit,
                    formats_by_node=formats_by_node)
    else:
        assert args.model_name and args.onnx and args.qdq_onnx and args.out_dir, \
            "--scale imagenet100 requires --model-name --onnx --qdq-onnx --out-dir"
        assert args.txt_dir or args.image_dir, \
            "--scale imagenet100 requires --txt-dir or --image-dir (--image-dir is the default)"
        sweep_imagenet100(args.model_name, args.onnx, args.qdq_onnx, args.op_features, formats,
                           args.out, args.out_dir, args.txt_dir,
                           limit=args.limit or 300, jobs=args.jobs,
                           formats_by_node=formats_by_node, input_shape=args.input_shape,
                           label_map=args.label_map,
                           image_dir=args.image_dir if not args.txt_dir else None,
                           image_preprocess_script=args.image_preprocess_script)


if __name__ == "__main__":
    main()
