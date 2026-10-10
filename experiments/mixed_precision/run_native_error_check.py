#!/usr/bin/env python3
"""M4 of the native criterion pipeline: per-channel activation quantization error statistics computed by onnx-mlir.

--mixedp-collect-stats with MIXEDP_STATS_MODE=errors adds, at every quantization point t and every candidate activation
format, d = Q(t) - t with Q the pass's own activation QDQ (int: the calibration range from MIXEDP_STATS_RANGES, here
the M1 native ranges; posit: posit.round_f32; f16/bf16: casts), reduced per channel to sum(d) and sum(d^2) -- the
s1 / s2 of the C term. Checked on the calibration images against numpy on the very same FP32 tensors, taken from a
MIXEDP_STATS_MODE=values build of the same graph (so the reference sees exactly the values the errors build
quantized): int as the float32 Div/Add/Round(half even)/Clip/Sub/Mul sequence, f16 / bf16 round to nearest even, posit
through Universal (PositOracle). Also: the int (scale, zero point) the pass derived must equal
eval_random_mixed_precision.int_qparams on the same range -- the values MIXEDP_NODE_FORMATS will carry. Exits 1 on any
int-parameter mismatch or an s1 / s2 error (max|native - ref| / max|ref| per tap and format, summed over the images)
above --tol-s1 / --tol-s2.

The native sums are float32 ReduceSums over up to 112 x 112 = 12544 positions per channel, so they differ from the
float64 reference by up to ~12544 * 2^-24 = 7.5e-4 relative (MobileNetV2: <= 2.1e-4). On the 112 x 112 taps they equal
a float32 sequential re-summation of the reference's own float32 d exactly; elementwise QDQ bit-exactness is
test/mixed_precision/test_mixed_precision_runtime.py's job. A NaN anywhere (non-finite statistics) fails the
check: maxima propagate NaN and every threshold test is 'not value <= tol'. Supported nodes the pass skipped inside
If/Loop/Scan bodies fail it too.
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
from run_native_calib_check import SLOT_INPUT, native_ranges  # noqa: E402
from run_w8_per_channel_check import MODELS  # noqa: E402

FORMATS = ("int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16")
f32 = np.float32


def q_int(x, scale, zp, bits):
    """The pass's int activation QDQ ops, in float32 like the compiled code."""
    s, z = f32(scale), f32(zp)
    t = np.rint(x / s + z)                                       # onnx.Round: half to even
    return ((np.clip(t, f32(-(1 << (bits - 1))), f32((1 << (bits - 1)) - 1)) - z) * s).astype(f32)


def q_bf16(x):
    u = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype("<u4").view("<f4")


def build(tools, work, base, tag, env, opt="O3"):
    layout_path, inst = work / f"{tag}_layout.json", work / f"{tag}.onnx.mlir"
    erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "-o", inst],
            env={**env, "MIXEDP_STATS_LAYOUT": str(layout_path)})
    postfix = f"{erm.SYMBOL_POSTFIX}_{tag}"            # both .so are loaded into this process
    erm.set_symbol_postfix(inst, postfix)
    t0 = time.time()
    so, _ = erm.build_so(tools, work, tag, inst, None, None, uses_posit=True, uses_f16=True, opt=opt)
    return (so, postfix), json.loads(layout_path.read_text()), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--models", default="mobilenetv2/timm_ra")
    ap.add_argument("--tol-s1", type=float, default=1e-3)
    ap.add_argument("--tol-s2", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    oracle = erm.PositOracle(tools, args.out_dir)
    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    sys.path.insert(0, str(tools.lib))
    from PyRuntime import OMExecutionSession
    summary, failures = {}, []
    for m in args.models.split(","):
        tag = m.split("/")[0]
        work = args.out_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        model_path = HERE / "../imagenet100/model_dataset" / m / "model.onnx"
        graph = onnx.load(str(model_path), load_external_data=False)
        meta = json.loads((model_path.parent / "metadata.json").read_text())
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta).astype(np.float32)

        # M1 native ranges, keyed by (node, slot) for the pass
        ranges, _, _ = native_ranges(tools, work, model_path, cx)
        by_name = {n.name: n for n in graph.graph.node}
        base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
        so_v, lay_v, t_v = build(tools, work, base, "values", {"MIXEDP_STATS_MODE": "values"})
        rng = {}
        for p in lay_v["points"]:
            n = by_name[p["node"]]
            t = n.output[0] if p["slot"] == "y" else n.input[SLOT_INPUT[p["slot"]]]
            rng.setdefault(p["node"], {})[p["slot"]] = list(ranges[t])
        (work / "ranges_by_point.json").write_text(json.dumps(rng) + "\n")
        so_e, lay_e, t_e = build(tools, work, base, "errors",
                                 {"MIXEDP_STATS_MODE": "errors", "MIXEDP_STATS_FORMATS": ",".join(FORMATS),
                                  "MIXEDP_STATS_RANGES": str(work / "ranges_by_point.json")},
                             opt="O0")   # thousands of small loops in one function: LLVM -O1/-O3 take 20+ min
        log(f"{tag}: built values ({t_v:.0f}s) and errors ({t_e:.0f}s) .so; {len(lay_e['taps'])} taps x "
            f"{len(FORMATS)} formats")

        # int params: pass vs int_qparams on the same range
        first_point = {}
        for p in lay_e["points"]:
            first_point.setdefault(p["tap"], p)
        param_mismatch = []
        for t in lay_e["taps"]:
            p = first_point[t["tap"]]
            lo, hi = rng[p["node"]][p["slot"]]
            for f, (scale, zp) in t["int_params"].items():
                ps, pz = erm.int_qparams(lo, hi, int(f[3:]))
                if scale != ps or zp != pz:
                    param_mismatch.append({"tap": t["tap"], "format": f, "pass": [scale, zp], "python": [ps, pz]})

        sv, se = OMExecutionSession(str(so_v[0]), so_v[1]), OMExecutionSession(str(so_e[0]), so_e[1])
        nat = {(t["tap"], f): [0.0, 0.0] for t in lay_e["taps"] for f in FORMATS}
        ref = {k: [0.0, 0.0] for k in nat}
        t_run = 0.0
        for x in cx:
            vals = np.asarray(sv.run([np.ascontiguousarray(x[None])])[lay_v["result_index"]])
            t1 = time.time()
            stats = np.asarray(se.run([np.ascontiguousarray(x[None])])[lay_e["result_index"]], np.float64)
            t_run += time.time() - t1
            for tv, te in zip(lay_v["taps"], lay_e["taps"]):
                shape = tv["shape"]
                v = vals[tv["offset"]:tv["offset"] + int(np.prod(shape))].reshape(shape)
                ax, c = te["channel_axis"], te["channels"]
                other = tuple(d for d in range(v.ndim) if d != ax)
                for f in FORMATS:
                    if f.startswith("int"):
                        q = q_int(v, *te["int_params"][f], int(f[3:]))
                    elif f == "f16":
                        q = v.astype(np.float16).astype(f32)
                    elif f == "bf16":
                        q = q_bf16(v).reshape(v.shape)
                    else:
                        q = oracle(v, f)
                    d = q.astype(np.float64) - v.astype(np.float64)
                    o = te["errors"][f]
                    k = (te["tap"], f)
                    nat[k][0] = nat[k][0] + stats[o:o + c]
                    nat[k][1] = nat[k][1] + stats[o + c:o + 2 * c]
                    ref[k][0] = ref[k][0] + d.sum(axis=other)
                    ref[k][1] = ref[k][1] + (d * d).sum(axis=other)

        def rel(a, b):                                 # NaN-propagating: non-finite statistics fail below
            return float(np.max(np.abs(a - b)) / np.max([np.max(np.abs(b)), 1e-300]))

        worst = {f: {"s1": 0.0, "s2": 0.0} for f in FORMATS}
        for (tap, f), (n1, n2) in nat.items():
            r1, r2 = ref[(tap, f)]
            worst[f]["s1"] = float(np.max([worst[f]["s1"], rel(n1, r1)]))
            worst[f]["s2"] = float(np.max([worst[f]["s2"], rel(n2, r2)]))
        s = {"taps": len(lay_e["taps"]), "formats": list(FORMATS), "int_param_mismatches": len(param_mismatch),
             "worst": worst, "errors_run_s_per_image": t_run / len(cx), "build_s": {"values": t_v, "errors": t_e}}
        bad = ([f"{len(param_mismatch)} int params differ from int_qparams, e.g. {param_mismatch[:2]}"]
               if param_mismatch else []) + \
              [f"{f} s1 rel err {w['s1']:.2e} > {args.tol_s1:.0e}" for f, w in worst.items()
               if not w["s1"] <= args.tol_s1] + \
              [f"{f} s2 rel err {w['s2']:.2e} > {args.tol_s2:.0e}" for f, w in worst.items()
               if not w["s2"] <= args.tol_s2] + \
              ([f"supported nodes inside If/Loop/Scan bodies not covered: {lay_e['skipped_nested']}"]
               if lay_e.get("skipped_nested") else [])
        s["pass"] = not bad
        failures += [f"{tag}: {b}" for b in bad]
        summary[tag] = s
        (work / "result.json").write_text(json.dumps(s, indent=1) + "\n")
        log(f"{tag}: {'PASS' if not bad else 'FAIL'}  int param mismatches {len(param_mismatch)}; errors .so "
            f"{s['errors_run_s_per_image']:.3f}s/img")
        log("   worst s1 / s2 rel err: " + "  ".join(f"{f} {w['s1']:.1e}/{w['s2']:.1e}" for f, w in worst.items()))
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
