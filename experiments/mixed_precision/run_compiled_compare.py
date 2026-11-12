#!/usr/bin/env python3
"""Compiled (onnx-mlir) evaluation of the weight-B + activation-C picks vs the SQNR picks, per-channel weights.

The configs are the MILP picks saved by run_w8pc_full_compare.py (criteria "BC" and "SQNR", every budget). Each is
compiled through --convert-onnx-to-mixed-precision with MIXEDP_WEIGHT_PER_CHANNEL=1 (per-output-channel intN weight
scales; activations: the pass's per-tensor affine a=), lowered, linked and run with PyRuntime on the held-out images
(eval_random_mixed_precision.py's build_so / run_so). References: plain onnx-mlir FP32 (no pass) and every node at
w=int8, a=int8. Every compiled result is checked against the onnxruntime QDQ simulation with the same per-channel
weight arithmetic (run_w8_per_channel_check.per_channel_int).
  --check   only FP32 and all-W8 (a=fp32) on a few images, per-channel and per-tensor, against the simulation
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--configs-from", type=Path, default=HERE / "runs/w8pc_full_compare_20261005")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="configs compiled and run in parallel")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    original = erm.np_fake_quant_weight
    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    held = [e for e in man["images"] if not e["calibration"]][:args.num_eval]
    ey = np.array([e["label"] for e in held])
    summary = {}
    for m in args.models.split(","):
        tag = m.split("/")[0]
        mdir = args.out_dir / tag
        work = mdir / "work"
        work.mkdir(parents=True, exist_ok=True)
        model_path = HERE / "../imagenet100/model_dataset" / m / "model.onnx"
        graph = onnx.load(str(model_path))
        meta = json.loads((model_path.parent / "metadata.json").read_text())
        nodes = erm.supported_nodes(graph)
        names = [n["name"] for n in nodes if n.get("weight_const")]
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta)
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ranges = erm.calibrate_ranges(graph, nodes, cx)
        inputs_path = work / "inputs.npy"
        np.save(inputs_path, ex.astype(np.float32))
        base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
        if not base.exists():
            erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                     "--mlir-elide-resource-strings-if-larger=1000000000",
                     "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])

        def full(cfg):   # every supported node, as node_formats_env needs
            return {n["name"]: cfg.get(n["name"], {"w": None, "a": "fp32"}) for n in nodes}

        if args.check:
            w8 = full({n: {"w": "int8", "a": "fp32"} for n in names})
            configs = [("fp32", None, True), ("w8_per_channel", w8, True), ("w8_per_tensor", w8, False)]
        else:
            saved = json.loads((args.configs_from / tag / "results.json").read_text())["rows"]
            configs = [("fp32", None, True),
                       ("w8a8", full({n: {"w": "int8", "a": "int8"} for n in names}), True)]
            for r in saved:
                if r["criterion"] in ("BC", "SQNR"):
                    configs.append((f"{r['criterion']}_f{r['budget']:.2f}", full(r["formats"]), True))
        rows = {}

        def set_weights(per_channel):
            if per_channel:
                os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
                erm.np_fake_quant_weight = lambda w, fmt, oracle: (per_channel_int(w, int(fmt[3:])) if fmt.startswith("int")
                                                                    else original(w, fmt, oracle))
            else:
                os.environ.pop("MIXEDP_WEIGHT_PER_CHANNEL", None)
                erm.np_fake_quant_weight = original

        def compile_and_run(name, cfg):
            env = None if cfg is None else erm.node_formats_env(cfg, nodes, ranges)
            tc = time.time()
            so, info = erm.build_so(tools, work, name, base, env, None)
            compile_s = time.time() - tc
            logits, ev = erm.run_so(tools, so, inputs_path, work / f"{name}.logits.npy", 36000)
            for p in work.glob(f"{name}.*"):
                if p.suffix in (".mlir", ".ll", ".so", ".npy") and p != base:
                    p.unlink()
            return logits, ev, compile_s, info

        compiled = {}
        if args.check:   # mixes per-channel and per-tensor builds: one at a time
            for name, cfg, per_channel in configs:
                set_weights(per_channel)
                compiled[name] = compile_and_run(name, cfg)
        else:            # all per-channel: compile + run in parallel (each .so runs single-threaded)
            set_weights(True)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {name: pool.submit(compile_and_run, name, cfg) for name, cfg, _ in configs}
                for name, fut in futures.items():
                    compiled[name] = fut.result()
                    log(f"{tag} {name}: compiled and run")
        for name, cfg, per_channel in configs:
            set_weights(per_channel)
            logits, ev, compile_s, info = compiled[name]
            sim = erm.simulate_in_onnxruntime(graph, nodes, cfg or {}, ranges, ex, None)
            finite = np.isfinite(logits).all(1)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], logits, 0.0)), torch.from_numpy(ey),
                                   reduction="none").numpy()
            correct = ((logits.argmax(1) == ey) & finite).astype(float)
            d = np.abs(logits - sim)
            rows[name] = {"top1": float(correct.mean()), "loss": float(loss.mean()), "compile_seconds": compile_s,
                          "run_seconds_per_image": ev["seconds"] / ev["n"], "nonfinite_images": int((~finite).sum()),
                          "vs_simulation": {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                                            "same_top1": float((logits.argmax(1) == sim.argmax(1)).mean()),
                                            "simulated_top1": float((sim.argmax(1) == ey).mean())},
                          "warnings": len(info["warnings"]), "_correct": correct, "_loss": loss}
            log(f"{tag} {name:<16} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}  | vs ORT sim: max |dz| "
                f"{d.max():.2e}, same top-1 {rows[name]['vs_simulation']['same_top1']:.1%} | compile {compile_s:.0f}s, "
                f"run {ev['seconds'] / ev['n'] * 1000:.0f} ms/img")
        os.environ.pop("MIXEDP_WEIGHT_PER_CHANNEL", None)
        erm.np_fake_quant_weight = original
        if not args.check:
            budgets = sorted({float(k.split("_f")[1]) for k in rows if "_f" in k})
            for b in budgets:
                bc, sq = rows[f"BC_f{b:.2f}"], rows[f"SQNR_f{b:.2f}"]
                dl, d1 = bc["_loss"] - sq["_loss"], (bc["_correct"] - sq["_correct"]) * 100
                bc["vs_SQNR"] = {"dtop1_pp": float(d1.mean()), "dloss": float(dl.mean()),
                                 "dloss_se": float(dl.std(ddof=1) / np.sqrt(len(dl)))}
        summary[tag] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rows.items()}
        (mdir / "results.json").write_text(json.dumps(summary[tag], indent=2) + "\n")
        if not args.check:
            print(f"\n=== {m} (onnx-mlir compiled, per-channel weights, {len(ey)} held-out images): FP32 "
                  f"{rows['fp32']['top1']:.1%}/{rows['fp32']['loss']:.3f}, all W8A8 {rows['w8a8']['top1']:.1%}/{rows['w8a8']['loss']:.3f}")
            for b in budgets:
                bc, sq = rows[f"BC_f{b:.2f}"], rows[f"SQNR_f{b:.2f}"]
                print(f"   f={b:.2f}  B+C {bc['top1']:6.1%}/{bc['loss']:.3f}   SQNR {sq['top1']:6.1%}/{sq['loss']:.3f}   "
                      f"B+C - SQNR: top-1 {bc['vs_SQNR']['dtop1_pp']:+.1f} pp, loss {bc['vs_SQNR']['dloss']:+.4f}"
                      f"±{bc['vs_SQNR']['dloss_se']:.4f}")
            mb = np.mean([rows[f"BC_f{b:.2f}"]["top1"] for b in budgets])
            ms = np.mean([rows[f"SQNR_f{b:.2f}"]["top1"] for b in budgets])
            print(f"   mean  B+C {mb:.1%}/{np.mean([rows[f'BC_f{b:.2f}']['loss'] for b in budgets]):.3f}   SQNR "
                  f"{ms:.1%}/{np.mean([rows[f'SQNR_f{b:.2f}']['loss'] for b in budgets]):.3f}   max |compiled - simulated| "
                  f"{max(r['vs_simulation']['max_abs_diff'] for r in rows.values()):.2e}")
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
