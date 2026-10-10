#!/usr/bin/env python3
"""MobileNetV2: run_gpt2_fisher_term_ablation.py on an ImageNet CNN. Does each term of the per-tensor Fisher
criterion carry useful information for the MILP?

Base, per weight-bearing node n and format f:
  T_w = mean(dL/dW^2) * ||Q_f(W) - W||^2,  T_x = mean(dL/dx^2) * sum ||Q_f(x) - x||^2,  T_y likewise for y
with W, x, y exactly the ONNX node's weight, input and output: the timm model gets its BatchNorms folded into
the convolutions (as the exported ONNX has them), and every folded weight is checked against the ONNX
initializer. Gradients: cross-entropy with the calibration images' labels, one backward pass per image.
Ablations of one term t: drop_t (ignored) or avg_t (same total size, no per-layer information).
MILP at budgets relative to all-int8, ORT QDQ simulation on the held-out images.
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from eval_random_mixed_precision import (calibrate_ranges, int_qparams, np_fake_quant_weight,  # noqa: E402
                                         preprocess_by_source, simulate_in_onnxruntime, supported_nodes)
from mp_milp import milp_select  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

FORMATS = ["int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32"]
VARIANTS = ("base", "drop_w", "drop_x", "drop_y", "avg_w", "avg_x", "avg_y")


def act_fake_quant(t, fmt, lohi):
    if fmt == "fp32":
        return t
    if fmt == "f16":
        return t.half().float()
    if fmt == "bf16":
        return t.bfloat16().float()
    bits = int(fmt[3:])
    scale, zp = int_qparams(*lohi, bits)
    return (torch.clamp(torch.round(t / scale + zp), -(1 << (bits - 1)), (1 << (bits - 1)) - 1) - zp) * scale


def fold_batchnorms(model):
    """Fold every BatchNorm into the convolution right before it (same parent, definition order), and make the
    BatchNorm an identity, keeping timm's fused activation."""
    for parent in model.modules():
        conv = None
        for child in parent.children():
            if isinstance(child, nn.Conv2d):
                conv = child
            elif isinstance(child, nn.BatchNorm2d) and conv is not None:
                s = child.weight / torch.sqrt(child.running_var + child.eps)
                bias = conv.bias if conv.bias is not None else torch.zeros_like(child.running_mean)
                with torch.no_grad():
                    conv.weight.mul_(s.view(-1, 1, 1, 1))
                    conv.bias = nn.Parameter(child.bias + (bias - child.running_mean) * s)
                    child.weight.fill_(1.0)
                    child.bias.zero_()
                    child.running_mean.zero_()
                    child.running_var.fill_(1.0 - child.eps)
                conv = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8", help="fractions of all-int8 weight bytes and BOPs")
    ap.add_argument("--milp-time-limit", type=float, default=60.0,
                    help="seconds per MILP; on timeout the best incumbent is kept and flagged (the avg_* variants "
                         "make many options cost the same, which HiGHS can take very long to prove optimal)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(f) for f in args.budgets.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]]
    cx = preprocess_by_source(args.images_dir, {"images": calib}, meta)
    cy = np.array([e["label"] for e in calib])
    ex = preprocess_by_source(args.images_dir, {"images": held}, meta)
    ey = np.array([e["label"] for e in held])
    all_nodes = supported_nodes(graph)
    nodes = [n for n in all_nodes if n.get("weight_const")]
    ranges = calibrate_ranges(graph, all_nodes, cx)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}

    # --- torch model with the ONNX graph's folded weights, node by node ---
    import timm
    model = timm.create_model(args.timm_name, pretrained=True).eval()
    fold_batchnorms(model)
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(layers) == len(nodes), (len(layers), len(nodes))
    module_of = {}
    for n, (mname, m) in zip(nodes, layers):
        w = inits[by_name[n["name"]].input[1]]
        assert w.shape == tuple(m.weight.shape), (n["name"], mname, w.shape, tuple(m.weight.shape))
        np.testing.assert_allclose(m.weight.detach().numpy(), w, rtol=1e-4, atol=1e-5, err_msg=f"{n['name']} vs {mname}")
        module_of[n["name"]] = m
    ref = simulate_in_onnxruntime(graph, all_nodes, {}, ranges, cx, None)
    with torch.no_grad():
        mine = model(torch.from_numpy(cx)).numpy()
    np.testing.assert_allclose(mine, ref, rtol=1e-3, atol=2e-3)
    log(f"{len(nodes)} weighted ONNX nodes matched to folded torch layers; FP32 logits agree "
        f"(max |diff| {np.abs(mine - ref).max():.1e}); {len(cy)} calibration, {len(ey)} held-out images")

    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, m in module_of.items():
        m.register_forward_hook(hook(n))
    g2w = dict.fromkeys(module_of, 0.0)
    act = {n: {s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")} for n in module_of}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)       # so the first conv's input has a gradient
        loss = nn.functional.cross_entropy(model(x0), torch.tensor(cy[k:k + 1]))
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g2w[n] += float((module_of[n].weight.grad.double() ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a = act[n][side]
                    a["g2"] += float((t.grad.double() ** 2).sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        a["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
        cap.clear()
    log("gradients on the calibration images done")

    T = {"w": {}, "x": {}, "y": {}}
    numel = {"w": {}, "x": {}, "y": {}}
    for n, m in module_of.items():
        W = inits[by_name[n].input[1]]
        T["w"][n] = {"fp32": 0.0, **{f: g2w[n] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                                   - W.astype(np.float64)) ** 2).sum())
                                     for f in FORMATS[:-1]}}
        numel["w"][n] = W.size
        for side in ("x", "y"):
            a = act[n][side]
            T[side][n] = {"fp32": 0.0, **{f: a["g2"] / a["count"] * a["d2"][f] for f in FORMATS[:-1]}}
            numel[side][n] = a["count"]
    avg = {t: {n: {f: sum(T[t][m][f] for m in module_of) / sum(numel[t].values()) * numel[t][n] for f in FORMATS}
               for n in module_of} for t in ("w", "x", "y")}

    def value_fn(variant):
        kind, _, term = variant.partition("_")
        tables = {t: (avg[t] if kind == "avg" and t == term else T[t]) for t in ("w", "x", "y")}
        use = {t: not (kind == "drop" and t == term) for t in ("w", "x", "y")}
        return lambda n, w, a: ((tables["w"][n][w] if use["w"] else 0.0) + (tables["x"][n][a] if use["x"] else 0.0)
                                + (tables["y"][n][a] if use["y"] else 0.0))

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    names = list(module_of)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    base_logits = simulate_in_onnxruntime(graph, all_nodes, {}, ranges, ex, None)
    fp32 = {"top1": float((base_logits.argmax(1) == ey).mean()),
            "loss": float(nn.functional.cross_entropy(torch.from_numpy(base_logits), torch.from_numpy(ey)))}
    log(f"FP32 held-out top1 {fp32['top1']:.1%}, loss {fp32['loss']:.4f}")
    res, rows = {}, []
    for f in budgets:
        for v in VARIANTS:
            cfg, obj, status = milp_select(all_nodes, FORMATS, FORMATS, value_fn(v), weight_elems, flops_proxy,
                                           f * int8_bytes, f * int8_bops, time_limit=args.milp_time_limit)
            assert status in (0, 1), (v, f, status)              # 1: time limit, best incumbent kept
            size, bops = config_cost(cfg, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            correct = (z.argmax(1) == ey) & finite
            loss = nn.functional.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)),
                                               torch.from_numpy(ey), reduction="none").numpy()
            res[(f, v)] = (correct.astype(float), loss)
            wq = [c["w"] for c in cfg.values() if c["w"]]
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0,
                         "size": size / int8_bytes, "bops": bops / int8_bops,
                         "top1": float(correct.mean()), "loss": float(loss.mean()), "int4_weights": wq.count("int4"),
                         "int8plus_activations": sum(c["a"] in ("int8", "f16", "bf16") for n_, c in cfg.items()
                                                     if n_ in module_of), "formats": cfg})
            log(f"f={f:.2f} {v:<7} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit: best incumbent)"))
    for r in rows:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "base")]
        d1, dl = (c - cb) * 100, l - lb
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "fp32": fp32, "budgets": budgets, "rows": rows},
        indent=2) + "\n")

    print(f"\nFP32 top1 {fp32['top1']:.1%} loss {fp32['loss']:.4f} on {len(ey)} held-out images; "
          f"* = MILP stopped at the {args.milp_time_limit:.0f}s limit (best incumbent, optimality not proven)")
    print(f"{'f':>5} {'variant':<8} {'top1':>7} {'loss':>7} {'d top1 (pp)':>14} {'d loss':>17} {'size':>5} {'BOPs':>5} "
          f"{'int4 W':>7} {'A>=8bit':>8}")
    for r in rows:
        d = "" if r["variant"] == "base" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "base" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<8} {r['top1']:7.1%} {r['loss']:7.4f} {d:>14} {dl:>17} "
              f"{r['size']:5.2f} {r['bops']:5.2f} {r['int4_weights']:7d} {r['int8plus_activations']:8d}")


if __name__ == "__main__":
    main()
