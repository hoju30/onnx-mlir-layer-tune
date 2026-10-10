#!/usr/bin/env python3
"""M2 of the native criterion pipeline: per-output-channel weight quantization errors computed by onnx-mlir itself.

--mixedp-collect-stats with MIXEDP_STATS_MODE=weights (and MIXEDP_WEIGHT_PER_CHANNEL=1, as the compiled comparisons use)
writes, at compile time, sum over each output channel of (Q(W) - W)^2 for every weight-bearing node and candidate
format, Q being the pass's own fakeQuantWeight. Checked against the numpy computation the weight terms A/B were built
from so far (run_w8pc_full_compare.py: per_channel_int for intN, eval_random_mixed_precision.np_fake_quant_weight for
f16/bf16 and, through Universal's PositOracle, posit) on the ONNX initializers -- an offline reference only. The
reference quantizes intN along each node's output-channel axis derived here from the ONNX attributes (Conv 0, Gemm
transB=1 -> 0 / transB=0 -> 1, MatMul last), which the pass's own axis must equal. Exits 1 if a weighted node is
missing, an axis differs, or a per-channel error differs by more than --tol (relative).

Non-finite errors (e.g. f16 overflowing a weight beyond 65504: the pass writes null, numpy gets inf) are compared
explicitly: a channel finite on one side only is a mismatch; channels non-finite on both sides are counted and fail
the check too unless --allow-nonfinite (they agree, but such a format makes that weight unusable).
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
from onnx import numpy_helper  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402

FORMATS = ("int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16")


def output_channel_axis(node, W):
    if node.op_type == "Conv":
        return 0
    if node.op_type == "Gemm":
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b else 1
    return W.ndim - 1                                    # MatMul: [..., K, N]


def int_along(W, bits, ax):
    """per_channel_int (one scale per index along axis 0) applied along `ax`."""
    return np.moveaxis(per_channel_int(np.moveaxis(W, ax, 0), bits), 0, ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--onnx", default="", help="extra models as tag=path.onnx,... (e.g. GPT-2/BERT, whose Gemm/MatMul "
                                                 "weights have a non-zero output-channel axis)")
    ap.add_argument("--tol", type=float, default=1e-9, help="max relative per-channel difference")
    ap.add_argument("--allow-nonfinite", action="store_true",
                    help="accept channels whose error is non-finite on both sides")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    oracle = erm.PositOracle(tools, args.out_dir)
    summary, failures = {}, []
    models = [(m.split("/")[0], HERE / "../imagenet100/model_dataset" / m / "model.onnx")
              for m in args.models.split(",") if m]
    models += [(t, Path(p)) for t, p in (e.split("=", 1) for e in args.onnx.split(",") if e)]
    for tag, model_path in models:
        work = args.out_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
        erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                 "--mlir-elide-resource-strings-if-larger=1000000000",
                 "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])
        out_json = work / "weight_err.json"
        _, t_pass = erm.run([tools.opt, base, "--mixedp-collect-stats", "-o", work / "unchanged.onnx.mlir"],
                            env={"MIXEDP_STATS_MODE": "weights", "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                                 "MIXEDP_STATS_FORMATS": ",".join(FORMATS), "MIXEDP_STATS_LAYOUT": str(out_json)})
        native = json.loads(out_json.read_text())

        graph = onnx.load(str(model_path))
        inits = {i.name: numpy_helper.to_array(i) for i in graph.graph.initializer}
        by_name = {n.name: n for n in graph.graph.node}
        weighted = [n["name"] for n in erm.supported_nodes(graph) if n.get("weight_const")]
        rows, worst, axis_mismatch = [], {f: 0.0 for f in FORMATS}, []
        nonfinite = {"both": 0, "native_only": 0, "reference_only": 0}
        t_ref = time.time()
        for e in native["nodes"]:
            node = by_name[e["node"]]
            W = inits[node.input[1]].astype(np.float32)
            ax = output_channel_axis(node, W)
            if e["channel_axis"] != ax:
                axis_mismatch.append({"node": e["node"], "native": e["channel_axis"], "expected": ax})
                continue
            for f in FORMATS:
                q = int_along(W, int(f[3:]), ax) if f.startswith("int") else erm.np_fake_quant_weight(W, f, oracle)
                d2 = (q.astype(np.float64) - W.astype(np.float64)) ** 2
                ref = np.moveaxis(d2, ax, 0).reshape(W.shape[ax], -1).sum(1)
                nat = np.array([np.nan if v is None else v for v in e["err2"][f]], dtype=np.float64)
                fn, fr = np.isfinite(nat), np.isfinite(ref)
                nonfinite["both"] += int((~fn & ~fr).sum())
                nonfinite["native_only"] += int((~fn & fr).sum())
                nonfinite["reference_only"] += int((fn & ~fr).sum())
                rel = np.zeros(len(ref))
                both = fn & fr
                rel[both] = np.abs(nat[both] - ref[both]) / np.maximum(np.abs(ref[both]), 1e-300)
                rel[fn != fr] = np.inf                    # finite on one side only
                worst[f] = float(np.max([worst[f], rel.max()]))
                rows.append({"node": e["node"], "format": f, "max_rel_diff": float(rel.max()),
                             "exact_channels": int((nat == ref).sum()), "channels": len(ref)})
        t_ref = time.time() - t_ref
        s = {"nodes_native": len(native["nodes"]), "nodes_onnx_weighted": len(weighted),
             "missing": sorted(set(weighted) - {e["node"] for e in native["nodes"]}), "skipped": native["skipped"],
             "axes": sorted({e["channel_axis"] for e in native["nodes"]}), "axis_mismatch": axis_mismatch,
             "max_rel_diff_by_format": worst, "max_rel_diff": float(np.max(list(worst.values()))),
             "nonfinite_channels": nonfinite,
             "exact_fraction": sum(r["exact_channels"] for r in rows) / sum(r["channels"] for r in rows),
             "pass_seconds": t_pass, "numpy_reference_seconds": t_ref}
        bad = ([f"missing nodes {s['missing']}"] if s["missing"] else []) + \
              ([f"axis mismatch {axis_mismatch}"] if axis_mismatch else []) + \
              ([f"max rel diff {s['max_rel_diff']:.2e} > tol {args.tol:.0e}"] if not s["max_rel_diff"] <= args.tol
               else []) + \
              ([f"non-finite error on one side only: {nonfinite}"]
               if nonfinite["native_only"] or nonfinite["reference_only"] else []) + \
              ([f"{nonfinite['both']} channels with a non-finite error (both sides; --allow-nonfinite to accept)"]
               if nonfinite["both"] and not args.allow_nonfinite else [])
        s["pass"] = not bad
        failures += [f"{tag}: {b}" for b in bad]
        summary[tag] = s
        (work / "result.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1) + "\n")
        log(f"{tag}: {'PASS' if not bad else 'FAIL'}  {s['nodes_native']}/{s['nodes_onnx_weighted']} weighted nodes, "
            f"missing {len(s['missing'])}, skipped {len(s['skipped'])}, axes {s['axes']}, axis mismatches "
            f"{len(axis_mismatch)}; max rel diff {s['max_rel_diff']:.2e} (exact channels "
            f"{s['exact_fraction']:.4f}); non-finite channels {nonfinite}; pass {t_pass:.1f}s vs numpy reference "
            f"{t_ref:.1f}s")
        log("   by format: " + "  ".join(f"{f} {v:.1e}" for f, v in worst.items()))
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))
    log(f"all {len(models)} models pass (tol {args.tol:.0e})")


if __name__ == "__main__":
    main()
