#!/usr/bin/env python3
"""M3 of the native criterion pipeline: the gradient statistics of the B + C criterion computed by onnx-mlir itself.

--mixedp-collect-stats with MIXEDP_STATS_MODE=grads appends the FP32 graph's own backward pass (MixedPrecisionBackward:
softmax cross-entropy of the logits against a 'mixedp_onehot' input) and reduces, per calibration image,
  every quantization point:  sum(g^2) and sum(g) per channel   (g = total gradient of the tensor)
  every weight:              sum(dW^2) per output channel
into one 'mixedp_stats' result. Over the calibration images this gives the statistics run_w8pc_full_compare.py took
from PyTorch autograd: per channel g2 = sum_img sum_pos g^2, gc = sum_img (sum_pos g)^2, and per output channel
sum_img sum_i dW^2. They are checked here against that PyTorch computation (timm/torchvision model with folded
BatchNorm, forward hooks on the matched modules -- an offline reference only) for every weight-bearing node's input
x, output y and weight. The error of an array is max|native - ref| / max|ref|.

Exact agreement is not expected everywhere: where a ReLU6/ReLU input lies within float rounding of a kink (0 or 6),
the two forward passes (~1e-5 apart) can put it on opposite sides, and that one element's gradient is then zero on
one side only. On MobileNetV2 that happens to ~1 element per image in a handful of the 10 images; the image's
statistics upstream of it then differ by up to ~1e-2 while every other image agrees to ~1e-6. A wrong gradient rule
would instead be off on every image. So per array: the median over images of the per-image error must stay below
--tol-median (catches rule errors), and the error of the statistics summed over the images -- what the criterion
uses -- below --tol-total. Exits 1 otherwise, and also on supported nodes the pass skipped inside If/Loop/Scan
bodies. A NaN anywhere (non-finite statistics) fails the check: maxima propagate NaN and every threshold test is
'not value <= tol'.
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


def native_stats(tools, work, model_path, cx, cy, num_classes):
    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])
    layout_path, inst = work / "grads_layout.json", work / "grads.onnx.mlir"
    _, t_pass = erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "-o", inst],
                        env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_LAYOUT": str(layout_path)})
    t0 = time.time()
    so, _ = erm.build_so(tools, work, "grads", inst, None, None)
    t_build = time.time() - t0
    layout = json.loads(layout_path.read_text())

    sys.path.insert(0, str(tools.lib))
    from PyRuntime import OMExecutionSession
    sess = OMExecutionSession(str(so), erm.SYMBOL_POSTFIX)
    per_img = []                                        # per image: {("g2"|"gc", tap) | ("w", node): array}
    logits = []
    t0 = time.time()
    for x, y in zip(cx, cy):
        onehot = np.zeros((1, num_classes), np.float32)
        onehot[0, y] = 1.0
        outs = sess.run([np.ascontiguousarray(x[None]), onehot])
        logits.append(np.asarray(outs[0])[0])
        v = np.asarray(outs[layout["result_index"]], dtype=np.float64)
        d = {}
        for t in layout["taps"]:
            o, c = t["offset"], t["channels"]
            d[("g2", t["tap"])] = v[o:o + c]
            d[("gc", t["tap"])] = v[o + c:o + 2 * c] ** 2
        for w in layout["weights"]:
            o, c = w["offset"], w["channels"]
            d[("w", w["node"])] = v[o:o + c]
        per_img.append(d)
    t_run = (time.time() - t0) / len(cx)
    return layout, per_img, np.stack(logits), {"pass_s": t_pass, "build_s": t_build, "run_s_per_image": t_run}


def torch_stats(graph, meta, weighted, cx, cy):
    """run_w8pc_full_compare.py's PyTorch statistics (offline reference)."""
    import torch
    import torch.nn.functional as F
    from run_criteria_cost_compare import match_modules, torch_model
    from run_mbv2_act_lowbit_check import chan_sum
    model, layers = torch_model(meta)
    module_of, _, _ = match_modules(graph, weighted, layers)
    names = [n["name"] for n in weighted]
    cap = {}
    for n, mod in module_of.items():
        mod.register_forward_hook(lambda mm, inp, o, n=n: cap.__setitem__(n, (inp[0], o)))
    per_img = []                                        # per image: {("g2"|"gc", node, slot) | ("w", node): array}
    logits = []
    for x, y in zip(cx, cy):
        z = model(torch.from_numpy(x[None]).requires_grad_(True))
        logits.append(z.detach().numpy()[0])
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(y)])), targets)
        d = {}
        with torch.no_grad():
            for i, n in enumerate(names):
                gw = grads[3 * i].double() ** 2
                d[("w", n)] = gw.reshape(gw.shape[0], -1).sum(1).numpy()
                for j, s in enumerate(("x", "y")):
                    g = grads[3 * i + 1 + j].double()
                    d[("g2", n, s)] = chan_sum(g ** 2).numpy()
                    d[("gc", n, s)] = chan_sum(g).numpy() ** 2
        per_img.append(d)
        cap.clear()
    return per_img, np.stack(logits)


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / np.max([np.max(np.abs(b)), 1e-300]))   # NaN-propagating


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--models", default="mobilenetv2/timm_ra")
    ap.add_argument("--tol-median", type=float, default=1e-3, help="median over images of the per-image error")
    ap.add_argument("--tol-total", type=float, default=5e-2, help="error of the statistics summed over the images")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    cy = np.array([e["label"] for e in calib])
    summary, failures = {}, []
    for m in args.models.split(","):
        tag = m.split("/")[0]
        work = args.out_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        model_path = HERE / "../imagenet100/model_dataset" / m / "model.onnx"
        graph = onnx.load(str(model_path))
        meta = json.loads((model_path.parent / "metadata.json").read_text())
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta).astype(np.float32)
        weighted = [n for n in erm.supported_nodes(graph) if n.get("weight_const")]

        layout, nat, nat_logits, timing = native_stats(tools, work, model_path, cx, cy, 1000)
        log(f"{tag}: native stats ({len(layout['taps'])} taps, {len(layout['weights'])} weights); pass "
            f"{timing['pass_s']:.1f}s, build {timing['build_s']:.1f}s, run {timing['run_s_per_image']:.3f}s/img")
        ref, ref_logits = torch_stats(graph, meta, weighted, cx, cy)

        tap_of = {(p["node"], p["slot"]): p["tap"] for p in layout["points"]}
        rows = []
        for n in weighted:
            name = n["name"]
            pairs = {"w": (("w", name), ("w", name))}
            for s in ("x", "y"):
                for f in ("g2", "gc"):
                    pairs[f"{s}_{f}"] = ((f, tap_of[(name, s)]), (f, name, s))
            row = {"node": name}
            for k, (kn, kr) in pairs.items():
                errs = [rel_err(a[kn], b[kr]) for a, b in zip(nat, ref)]
                row[k] = {"median": float(np.median(errs)), "max_image": float(max(errs)),
                          "total": rel_err(sum(a[kn] for a in nat), sum(b[kr] for b in ref))}
            rows.append(row)
        keys = ("w", "x_g2", "x_gc", "y_g2", "y_gc")
        worst = {m: {k: float(np.max([r[k][m] for r in rows])) for k in keys}
                 for m in ("median", "total", "max_image")}
        s = {"weighted_nodes": len(rows), "worst": worst,
             "logits_rel_err": rel_err(nat_logits, ref_logits), "timing": timing}
        bad = [f"{k} {m} rel err {v:.2e} > tol {tol:.0e} (node {max(rows, key=lambda r: r[k][m])['node']})"
               for m, tol in (("median", args.tol_median), ("total", args.tol_total))
               for k, v in worst[m].items() if not v <= tol]
        if layout.get("skipped_nested"):
            bad.append(f"supported nodes inside If/Loop/Scan bodies not covered: {layout['skipped_nested']}")
        s["pass"] = not bad
        failures += [f"{tag}: {b}" for b in bad]
        summary[tag] = s
        (work / "result.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1) + "\n")
        log(f"{tag}: {'PASS' if not bad else 'FAIL'}  logits rel err {s['logits_rel_err']:.1e}")
        for m in ("median", "total", "max_image"):
            log(f"   worst {m:<9} " + "  ".join(f"{k} {v:.1e}" for k, v in worst[m].items()))
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
