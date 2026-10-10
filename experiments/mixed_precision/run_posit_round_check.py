#!/usr/bin/env python3
"""posit.round_f32 (inlined table lookup) vs the posit.from_f32/to_f32 runtime path on a real model.

Every supported node of the model gets a=<fmt> (weights untouched), compiled twice through
--convert-onnx-to-mixed-precision: with the default posit activation path and with MIXEDP_POSIT_ACTIVATION=runtime.
Both run on the same held-out images (eval_random_mixed_precision.run_so, a fresh process each); the logits must be
bit-identical, and the per-image time shows what the lookup saves. Exits 1 on any mismatch.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--model", default="mobilenetv2/timm_ra")
    ap.add_argument("--formats", default="p8e1,p4e1")
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    work = args.out_dir / args.model.split("/")[0]
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    model_path = HERE / "../imagenet100/model_dataset" / args.model / "model.onnx"
    graph = onnx.load(str(model_path))
    meta = json.loads((model_path.parent / "metadata.json").read_text())
    man = json.loads((args.images_dir / "manifest.json").read_text())
    held = [e for e in man["images"] if not e["calibration"]][:args.num_eval]
    inputs = work / "inputs.npy"
    np.save(inputs, erm.preprocess_by_source(args.images_dir, {"images": held}, meta).astype(np.float32))
    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])
    posit_obj = erm.build_posit_runtime(tools, work)
    nodes = erm.supported_nodes(graph)

    summary, failures = {}, []
    for fmt in args.formats.split(","):
        env = ",".join(f"{n['name']}:a={fmt}" for n in nodes)
        res = {}
        for path in ("lut", "runtime"):
            os.environ["MIXEDP_POSIT_ACTIVATION"] = path        # inherited by erm.run's compiler steps
            t = time.time()
            so, info = erm.build_so(tools, work, f"{fmt}_{path}", base, env, posit_obj)
            build_s = time.time() - t
            logits, stats = erm.run_so(tools, so, inputs, work / f"{fmt}_{path}_logits.npy", timeout=7200)
            ir = info["ir_after_pass"]
            res[path] = {"logits": logits, "build_s": build_s, "run_s_per_image": stats["seconds"] / stats["n"],
                         "round_f32_ops": ir["posit.round_f32"], "from_f32_ops": ir["posit.from_f32"]}
            log(f"{fmt} {path}: {ir['posit.round_f32']} round_f32 / {ir['posit.from_f32']} from_f32 ops; "
                f"build {build_s:.1f}s, {res[path]['run_s_per_image']:.3f}s/img")
        a, b = res["lut"]["logits"], res["runtime"]["logits"]
        same = a.view(np.uint32) == b.view(np.uint32)
        same |= np.isnan(a) & np.isnan(b)
        s = {"nodes": len(nodes), "mismatching_logits": int((~same).sum()), "logits": int(same.size),
             "lut_top1_agrees_runtime": float((a.argmax(1) == b.argmax(1)).mean()),
             "speedup": res["runtime"]["run_s_per_image"] / res["lut"]["run_s_per_image"],
             **{f"{p}_{k}": v for p in res for k, v in res[p].items() if k != "logits"}}
        summary[fmt] = s
        if s["mismatching_logits"]:
            failures.append(f"{fmt}: {s['mismatching_logits']}/{s['logits']} logits differ")
        log(f"{fmt}: {s['logits'] - s['mismatching_logits']}/{s['logits']} logits bit-identical; "
            f"runtime {s['runtime_run_s_per_image']:.3f}s/img vs lookup {s['lut_run_s_per_image']:.3f}s/img "
            f"({s['speedup']:.1f}x)")
    os.environ.pop("MIXEDP_POSIT_ACTIVATION", None)
    (work / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
