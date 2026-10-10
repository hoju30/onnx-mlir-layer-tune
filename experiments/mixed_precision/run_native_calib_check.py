#!/usr/bin/env python3
"""M1 of the native criterion pipeline: activation calibration ranges computed by onnx-mlir itself.

The FP32 model is instrumented with --mixedp-collect-stats (MIXEDP_STATS_MODE=ranges: per-tensor min/max at every point
--convert-onnx-to-mixed-precision can quantize, appended as one 'mixedp_stats' result), compiled and run with PyRuntime
on the calibration images; the per-image (min, max) are reduced over the images and mapped back to ONNX tensor names
through the layout JSON the pass writes. Checked against eval_random_mixed_precision.calibrate_ranges (onnxruntime, used
here only as an offline reference): same tensor set, range differences, and whether the int4/int8 activation params
(int_qparams) that MIXEDP_NODE_FORMATS would carry change. Exits 1 if the tensor sets differ, a range differs by more
than --tol (relative to the reference range width) or is not finite, any int4/int8 zero-point changes, or the pass
skipped supported nodes nested in If/Loop/Scan bodies ('skipped_nested'). The last one needs its own check: both
sides miss those nodes -- eval_random_mixed_precision.supported_nodes, which the onnxruntime reference uses, only
walks the top-level graph -- so the tensor sets still agree.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_w8_per_channel_check import MODELS  # noqa: E402

SLOT_INPUT = {"x": 0, "a": 0, "b": 1}


def native_ranges(tools, work, model_path, calib):
    """(ranges by ONNX tensor name, layout, timings) from the instrumented, compiled FP32 graph."""
    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])
    layout_path, inst = work / "ranges_layout.json", work / "ranges.onnx.mlir"
    _, t_pass = erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "-o", inst],
                        env={"MIXEDP_STATS_MODE": "ranges", "MIXEDP_STATS_LAYOUT": str(layout_path)})
    postfix = f"{erm.SYMBOL_POSTFIX}_ranges"              # callers may load other .so of this model alongside
    erm.set_symbol_postfix(inst, postfix)
    t0 = time.time()
    so, _ = erm.build_so(tools, work, "ranges", inst, None, None)
    t_build = time.time() - t0
    layout = json.loads(layout_path.read_text())

    sys.path.insert(0, str(tools.lib))
    from PyRuntime import OMExecutionSession
    sess = OMExecutionSession(str(so), postfix)
    t0 = time.time()
    taps = np.stack([np.asarray(sess.run([np.ascontiguousarray(x[None])])[layout["result_index"]])
                     for x in calib]).reshape(len(calib), layout["num_taps"], 2)
    t_run = (time.time() - t0) / len(calib)
    lo, hi = taps[:, :, 0].min(0), taps[:, :, 1].max(0)

    by_name = {n.name: n for n in onnx.load(str(model_path), load_external_data=False).graph.node}
    ranges = {}
    for p in layout["points"]:
        n = by_name[p["node"]]
        t = n.output[0] if p["slot"] == "y" else n.input[SLOT_INPUT[p["slot"]]]
        r = (float(lo[p["tap"]]), float(hi[p["tap"]]))
        assert ranges.setdefault(t, r) == r, (t, ranges[t], r)   # one ONNX tensor -> one tap
    return ranges, layout, {"pass_s": t_pass, "build_s": t_build, "run_s_per_image": t_run}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tol", type=float, default=1e-4, help="max range difference relative to the ORT range width")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib_entries = [e for e in man["images"] if e["calibration"]]
    summary, failures = {}, []
    for m in args.models.split(","):
        tag = m.split("/")[0]
        work = args.out_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        model_path = HERE / "../imagenet100/model_dataset" / m / "model.onnx"
        graph = onnx.load(str(model_path))
        meta = json.loads((model_path.parent / "metadata.json").read_text())
        cpath = work / "calib_inputs.npy"         # preprocessing is data preparation, cached once
        if not cpath.exists():
            np.save(cpath, erm.preprocess_by_source(args.images_dir, {"images": calib_entries}, meta)
                    .astype(np.float32))
        cx = np.load(cpath)

        native, layout, timing = native_ranges(tools, work, model_path, cx)
        nodes = erm.supported_nodes(graph)
        ref = erm.calibrate_ranges(graph, nodes, cx)                     # offline reference only
        common = sorted(set(native) & set(ref))
        rows = []
        for t in common:
            (nl, nh), (rl, rh) = native[t], ref[t]
            width = max(rh - rl, 1e-30)
            row = {"tensor": t, "native": [nl, nh], "ort": [rl, rh],
                   "rel_diff": max(abs(nl - rl), abs(nh - rh)) / width}
            for bits in (4, 8):
                (ns, nz), (rs, rz) = erm.int_qparams(nl, nh, bits), erm.int_qparams(rl, rh, bits)
                row[f"int{bits}_zp_same"] = nz == rz
                row[f"int{bits}_scale_rel_diff"] = abs(ns - rs) / rs
            rows.append(row)
        rel = np.array([r["rel_diff"] for r in rows])
        nonfinite = sorted(t for t in common if not np.isfinite([*native[t], *ref[t]]).all())
        s = {"points": len(layout["points"]), "taps": layout["num_taps"], "tensors_native": len(native),
             "tensors_ort": len(ref), "only_native": sorted(set(native) - set(ref)),
             "only_ort": sorted(set(ref) - set(native)), "skipped_nested": layout.get("skipped_nested", []),
             "nonfinite_ranges": nonfinite, "max_rel_diff": float(np.max(rel)) if len(rel) else 0.0,
             "median_rel_diff": float(np.median(rel)) if len(rel) else 0.0, "exact": int((rel == 0).sum()),
             **{f"int{b}_zp_mismatch": int(sum(not r[f"int{b}_zp_same"] for r in rows)) for b in (4, 8)},
             **{f"int{b}_max_scale_rel_diff": float(max(r[f"int{b}_scale_rel_diff"] for r in rows)) for b in (4, 8)},
             "timing": timing}
        bad = ([f"tensor sets differ (only native {s['only_native']}, only ORT {s['only_ort']})"]
               if s["only_native"] or s["only_ort"] else []) + \
              ([f"max rel diff {s['max_rel_diff']:.2e} > tol {args.tol:.0e}"] if not s["max_rel_diff"] <= args.tol
               else []) + \
              ([f"non-finite ranges on {nonfinite}"] if nonfinite else []) + \
              ([f"{len(s['skipped_nested'])} supported nodes inside If/Loop/Scan bodies were not calibrated: "
                f"{s['skipped_nested']}"] if s["skipped_nested"] else []) + \
              [f"int{b} zero-point changed on {s[f'int{b}_zp_mismatch']} tensors" for b in (4, 8)
               if s[f"int{b}_zp_mismatch"]]
        s["pass"] = not bad
        failures += [f"{tag}: {b}" for b in bad]
        summary[tag] = s
        (work / "result.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1) + "\n")
        log(f"{tag}: {'PASS' if not bad else 'FAIL'}  {s['points']} points / {s['taps']} taps / {len(common)} "
            f"tensors; only native "
            f"{len(s['only_native'])}, only ORT {len(s['only_ort'])}; range rel diff max {s['max_rel_diff']:.2e} "
            f"median {s['median_rel_diff']:.2e} ({s['exact']} exact); zp mismatch int4 {s['int4_zp_mismatch']} "
            f"int8 {s['int8_zp_mismatch']}; build {timing['build_s']:.1f}s, run {timing['run_s_per_image']:.3f}s/img")
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))
    log(f"all {len(summary)} models pass (tol {args.tol:.0e})")


if __name__ == "__main__":
    main()
