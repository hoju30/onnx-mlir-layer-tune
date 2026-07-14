#!/usr/bin/env python3
"""
Validate a precision_plan.json by building the mixed-precision model and
measuring real accuracy. If the actual drop exceeds tolerance, roll back the
least-valuable node one format step at a time until within budget.

Rollback strategy:
  - Which node: lowest cost_saved/pred_error score (least valuable quantization)
  - How far:    upgrade to next less-aggressive posit format first (p8e1→p16e1→p32e1),
                only fall back to FP32 if already at the least aggressive format

Compensates for plan_precision.py's additive approximation: isolated per-node
deltas don't always sum linearly when multiple nodes are quantized together.

Usage:
  python validate_plan.py --scale mnist \
      --plan precision_plan.json --fp32-top1 0.9910 \
      --out precision_plan_validated.json

  python validate_plan.py --scale imagenet100 --model-name resnet18 \
      --onnx experiments/imagenet100/model/imagenet100_resnet18.onnx \
      --out-dir /tmp/validate --plan precision_plan.json \
      --fp32-top1 0.715 --out precision_plan_validated.json
"""

import argparse
import copy
import json
import os
import subprocess
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def node_formats_str(plan):
    parts = []
    for node, entry in plan["entities"].items():
        chosen = entry["chosen"]
        if chosen == "FP32":
            continue
        nbits, es = chosen.replace("posit_", "").split("_")
        parts.append(f"{node}:{nbits}:{es}")
    return ",".join(parts)


def chosen_formats(plan):
    fmts = set()
    for entry in plan["entities"].values():
        if entry["chosen"] != "FP32":
            nbits, es = entry["chosen"].replace("posit_", "").split("_")
            fmts.add((int(nbits), int(es)))
    return fmts


def _next_format(entry):
    """Return the next less-aggressive format from candidates, or 'FP32'."""
    chosen = entry["chosen"]
    chosen_nbits = int(chosen.replace("posit_", "").split("_")[0])
    cands_sorted = sorted(
        entry["candidates"],
        key=lambda c: int(c.replace("posit_", "").split("_")[0])
    )
    for c in cands_sorted:
        if int(c.replace("posit_", "").split("_")[0]) > chosen_nbits:
            return c
    return "FP32"


def _value_score(entry):
    """cost_saved/pred_error — lower means less valuable, roll back first."""
    chosen = entry["chosen"]
    nbits = int(chosen.replace("posit_", "").split("_")[0])
    precision_cost = entry.get("precision_cost", 0)
    cost_saved = precision_cost * (32 / nbits - 1)  # savings vs FP32
    pred_error = max(entry.get("pred_error", 0), 1e-9)
    return cost_saved / pred_error


def rollback_step(plan):
    """Upgrade the least-valuable posit node one format step toward FP32.
    Returns (node, old_format, new_format), or None if nothing left to roll back."""
    candidates = [(node, entry)
                  for node, entry in plan["entities"].items()
                  if entry["chosen"] != "FP32"]
    if not candidates:
        return None
    node, entry = min(candidates, key=lambda x: _value_score(x[1]))
    old_fmt = entry["chosen"]
    new_fmt = _next_format(entry)
    plan["entities"][node]["chosen"] = new_fmt
    return node, old_fmt, new_fmt


def n_posit_nodes(plan):
    return sum(1 for e in plan["entities"].values() if e["chosen"] != "FP32")


# ─────────────────────────────────────────────────────────────────────────────
# MNIST backend
# ─────────────────────────────────────────────────────────────────────────────

def eval_plan_mnist(plan, images, labels, sweep_dir, iteration):
    sys.path.insert(0, os.path.join(PROJECT, "docs", "mnist_example"))
    import eval_variants as mv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sweep_node_sensitivity import _compile_so_for_format

    nqf = node_formats_str(plan)
    fmts = chosen_formats(plan)

    if not fmts:
        fp32_so = os.path.join(mv.OUT, "mnist_fp32.so")
        if not os.path.exists(fp32_so):
            mv.run([mv.ONNXMLIR, "--EmitLib", "-o", fp32_so.replace(".so", ""), mv.ONNX])
        return mv.eval_so(fp32_so, images, labels, "")

    if len(fmts) > 1:
        # MNIST single-format runtime limitation: use highest precision as runtime.
        # Accuracy result is approximate; ImageNet100 backend handles mixed correctly.
        print(f"  WARNING: mixed formats {fmts} in plan -- MNIST runtime compiled for "
              f"highest precision only; result is approximate.", flush=True)

    nbits, es = max(fmts, key=lambda x: x[0])
    tag = f"validate_iter{iteration}"
    ll_path = os.path.join(sweep_dir, f"{tag}.ll")
    so_path = os.path.join(sweep_dir, f"{tag}.so")
    onnx_ir = os.path.join(mv.PP, "m.onnx.mlir")

    mv.posit_lower_to_ll(
        onnx_ir, ll_path,
        env={"POSIT_NODE_FORMATS": nqf, "POSIT_COMPACT_CONSTANTS": "1"},
        extra_opt_flags=["--shape-inference", "--convert-onnx-to-posit",
                         f"--posit-format=p{nbits}e{es}"])
    _compile_so_for_format(mv, ll_path, so_path, nbits, es)
    return mv.eval_so(so_path, images, labels, "m")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet100 backend
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def eval_plan_imagenet100(plan, model_name, onnx_path, out_dir, txt_dir,
                           limit, jobs, iteration, fp32_top1):
    fmts = chosen_formats(plan)
    if not fmts:
        # everything rolled back to FP32 -- no build needed
        return {"top1": fp32_top1, "top5": None}

    build_script = os.path.join(PROJECT, "src/bash/build_model11_sos.sh")
    time_script  = os.path.join(PROJECT, "src/bash/time_model11_dataset_parallel.sh")

    nqf = node_formats_str(plan)
    nbits, es = min(fmts, key=lambda x: x[0])  # lowest-precision as placeholder tag
    placeholder = f"p{nbits}e{es}"
    suffix = f"validate-iter{iteration}"

    env = dict(os.environ)
    env["POSIT_NODE_FORMATS"] = nqf
    _run([build_script, "--model-name", model_name, "--nqdq-onnx", onnx_path,
          "--out-dir", out_dir, "--posit-source", "nqdq",
          "--posit-formats", placeholder, "--skip-f32-baselines",
          "--continue-on-posit-fail", "--no-stage-logs"], env=env)

    built  = os.path.join(out_dir, f"{model_name}-nqdq-{placeholder}.so")
    target = os.path.join(out_dir, f"{model_name}-{suffix}.so")
    if not os.path.exists(built):
        raise RuntimeError(f"build did not produce {built}")
    os.replace(built, target)

    log_prefix = os.path.join(out_dir, f"{model_name}-validate-iter{iteration}")
    _run([time_script, "--model-name", model_name, "--out-dir", out_dir,
          "--txt-dir", txt_dir, "--limit", str(limit), "--jobs", str(jobs),
          "--suffixes", f"nqdq-f32,{suffix}", "--baseline", "nqdq-f32",
          "--qalign-auto", "off",
          "--format-log-b", f"{log_prefix}.format_summary.B.log"])

    summary_path = f"{log_prefix}.format_summary.B.log"
    with open(summary_path) as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]

    row = next((r for r in rows if r.get("format") == suffix), None)
    if not row:
        raise RuntimeError(f"suffix {suffix!r} not found in {summary_path}")
    return {
        "top1": float(row["gt_top1_pct"]) / 100.0,
        "top5": float(row["gt_top5_pct"]) / 100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["mnist", "imagenet100"], required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--fp32-top1", type=float, required=True,
                    help="fp32 baseline top-1 as fraction 0-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-name")
    ap.add_argument("--onnx")
    ap.add_argument("--out-dir")
    ap.add_argument("--txt-dir",
                    default=os.path.join(PROJECT, "experiments/imagenet100/val_224_txt"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    plan = copy.deepcopy(json.load(open(args.plan)))
    tolerance = plan["tolerance"]
    rollback_log = []

    if args.scale == "mnist":
        sys.path.insert(0, os.path.join(PROJECT, "docs", "mnist_example"))
        import eval_variants as mv
        images, labels = mv.load_mnist_test()
        if args.limit:
            images, labels = images[:args.limit], labels[:args.limit]
        sweep_dir = os.path.join(mv.PP, "sweep")
        os.makedirs(sweep_dir, exist_ok=True)
    else:
        assert args.model_name and args.onnx and args.out_dir, \
            "--scale imagenet100 requires --model-name --onnx --out-dir"
        os.makedirs(args.out_dir, exist_ok=True)

    # upper bound: each node can be stepped through at most len(candidates) formats
    max_iters = sum(len(e["candidates"]) for e in plan["entities"].values()) + 1
    for iteration in range(max_iters):
        n = n_posit_nodes(plan)
        print(f"\n[validate] iter {iteration}: {n} posit node(s)", flush=True)

        if args.scale == "mnist":
            r = eval_plan_mnist(plan, images, labels, sweep_dir, iteration)
        else:
            r = eval_plan_imagenet100(plan, args.model_name, args.onnx, args.out_dir,
                                      args.txt_dir, args.limit or 300, args.jobs,
                                      iteration, args.fp32_top1)

        actual_drop = args.fp32_top1 - r["top1"]
        print(f"  top1={r['top1']*100:.2f}%  "
              f"actual_drop={actual_drop*100:+.2f}pp  "
              f"tolerance={tolerance*100:.2f}pp", flush=True)

        plan["actual_top1"] = r["top1"]
        plan["actual_drop"] = actual_drop

        if actual_drop <= tolerance:
            print(f"  PASS — within tolerance after {len(rollback_log)} rollback(s)", flush=True)
            break

        step = rollback_step(plan)
        if step is None:
            print("  nothing left to roll back", flush=True)
            break
        node, old_fmt, new_fmt = step
        print(f"  OVER budget — {node}: {old_fmt} → {new_fmt}", flush=True)
        rollback_log.append({
            "iteration": iteration,
            "node": node,
            "from": old_fmt,
            "to": new_fmt,
            "actual_drop": actual_drop,
        })

    plan["rollback_log"] = rollback_log
    plan["n_posit_nodes_final"] = n_posit_nodes(plan)

    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nwrote validated plan ({n_posit_nodes(plan)} posit nodes) to {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
