#!/usr/bin/env python3
"""Does the better ranking of the per-channel Fisher weight term (B) give a better weight-bit allocation?

Weights quantized per output channel (run_w8_per_channel_check.py's arithmetic for int4..int8; f16/bf16/fp32 as the
pass); activations FP32, so the MILP allocates weight bits only, under a weight-byte budget (fraction of all-int8).
Criteria (dW = Q_pc(W) - W, g = dCE/dW on the calibration images):
  A     mean(g^2) * ||dW||^2                     per-tensor sensitivity
  B     sum_o mean_o(g^2) * ||dW_o||^2           per-output-channel sensitivity
  SQNR  ||logits - logits_fp32||^2 on the calibration images with only that node's weights quantized (reference)
Each MILP pick is evaluated in ORT QDQ simulation on the held-out images; differences paired per image.
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
import torch.nn.functional as F  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from htq_pareto_search import FORMAT_BITS  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_criteria_cost_compare import match_modules, torch_model  # noqa: E402
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402

WF = ["int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32"]
CRITERIA = ("A", "B", "SQNR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(b) for b in args.budgets.split(",")]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    held = [e for e in man["images"] if not e["calibration"]]
    cy, ey = np.array([e["label"] for e in calib]), np.array([e["label"] for e in held])
    original = erm.np_fake_quant_weight
    erm.np_fake_quant_weight = lambda w, fmt, oracle: (per_channel_int(w, int(fmt[3:])) if fmt.startswith("int")
                                                        else original(w, fmt, oracle))
    out = {}
    for m in args.models.split(","):
        tag = m.split("/")[0]
        md = HERE / "../imagenet100/model_dataset" / m
        graph = onnx.load(str(md / "model.onnx"))
        meta = json.loads((md / "metadata.json").read_text())
        nodes = erm.supported_nodes(graph)
        weighted = [n for n in nodes if n.get("weight_const")]
        names = [n["name"] for n in weighted]
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta)
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ranges = erm.calibrate_ranges(graph, nodes, cx)

        model, layers = torch_model(meta)
        module_of, inits, by_name = match_modules(graph, weighted, layers)
        g2 = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
        for k in range(len(cy)):
            z = model(torch.from_numpy(cx[k:k + 1]))
            for n, g in zip(names, torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])),
                                                       [module_of[n].weight for n in names])):
                g2[n] += g.double() ** 2
        kind = {n: ("fc" if module_of[n].weight.dim() == 2 else "dw" if module_of[n].groups > 1
                    else "pw" if module_of[n].kernel_size == (1, 1) else "conv") for n in names}
        T = {c: {n: {"fp32": 0.0} for n in names} for c in CRITERIA}
        for n in names:
            W = inits[by_name[n].input[1]]
            Go = g2[n].numpy().reshape(W.shape[0], -1)
            for f in WF[:-1]:
                q = erm.np_fake_quant_weight(W, f, None)
                d2 = ((q.astype(np.float64) - W) ** 2).reshape(W.shape[0], -1)
                T["A"][n][f] = float(Go.mean() * d2.sum())
                T["B"][n][f] = float((Go.mean(1) * d2.sum(1)).sum())
        ref = erm.simulate_in_onnxruntime(graph, nodes, {}, ranges, cx, None).astype(np.float64)
        for n in names:
            for f in WF[:-1]:
                zc = erm.simulate_in_onnxruntime(graph, nodes, {n: {"w": f, "a": "fp32"}}, ranges, cx, None)
                T["SQNR"][n][f] = float(((zc - ref) ** 2).sum())
        log(f"{tag}: A, B and SQNR weight tables ready ({len(names)} nodes)")

        weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, nodes)
        int8_bytes = sum(weight_elems[n] for n in names)

        def evaluate(cfg):
            z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(ey),
                                   reduction="none").numpy()
            return ((z.argmax(1) == ey) & finite).astype(float), loss

        fp32 = evaluate({})
        w8 = evaluate({n: {"w": "int8", "a": "fp32"} for n in names})
        res, rows = {}, []
        for f in budgets:
            for c in CRITERIA:
                cfg, _, status = milp_select(nodes, WF, ["fp32"], lambda n, w, a, c=c: T[c][n][w], weight_elems,
                                             flops_proxy, f * int8_bytes, 1e300, time_limit=60)
                assert status in (0, 1), (tag, c, f, status)
                cfg = {n: {"w": cfg[n]["w"], "a": "fp32"} for n in names}
                correct, loss = evaluate(cfg)
                res[(f, c)] = (correct, loss)
                bits = {k: float(sum(weight_elems[n] * FORMAT_BITS[cfg[n]["w"]] for n in names if kind[n] == k)
                                 / max(1, sum(weight_elems[n] for n in names if kind[n] == k)))
                        for k in ("conv", "pw", "dw", "fc") if any(kind[n] == k for n in names)}
                rows.append({"budget": f, "criterion": c, "milp_optimal": status == 0, "top1": float(correct.mean()),
                             "loss": float(loss.mean()), "bits_by_kind": bits,
                             "formats": {n: cfg[n]["w"] for n in names}})
                log(f"{tag} f={f:.2f} {c:<5} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}")
        for r in rows:
            c, l = res[(r["budget"], r["criterion"])]
            for ref_c in ("A", "SQNR"):
                cb, lb = res[(r["budget"], ref_c)]
                d = l - lb
                r[f"vs_{ref_c}"] = {"dtop1_pp": float(((c - cb) * 100).mean()), "dloss": float(d.mean()),
                                    "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
            base = next(x for x in rows if x["budget"] == r["budget"] and x["criterion"] == "A")["formats"]
            r["nodes_differing_from_A"] = sum(r["formats"][n] != base[n] for n in names)
        out[tag] = {"nodes": len(names), "fp32": {"top1": float(fp32[0].mean()), "loss": float(fp32[1].mean())},
                    "w8_per_channel": {"top1": float(w8[0].mean()), "loss": float(w8[1].mean())}, "rows": rows}
        print(f"\n=== {m}: weight-only allocation, per-channel weights, activations FP32; FP32 top-1 "
              f"{fp32[0].mean():.1%}, all-W8 per-channel {w8[0].mean():.1%} (loss {w8[1].mean():.3f})")
        print(f"   {'f':>5} {'crit':<5}{'top1':>7}{'loss':>8}{'d loss vs A':>18}{'d loss vs SQNR':>18}{'differ':>8}   weight bits by kind")
        for r in rows:
            dA = "" if r["criterion"] == "A" else f"{r['vs_A']['dloss']:+.4f}±{r['vs_A']['dloss_se']:.4f}"
            dS = "" if r["criterion"] == "SQNR" else f"{r['vs_SQNR']['dloss']:+.4f}±{r['vs_SQNR']['dloss_se']:.4f}"
            print(f"   {r['budget']:5.2f} {r['criterion']:<5}{r['top1']:7.1%}{r['loss']:8.4f}{dA:>18}{dS:>18}"
                  f"{r['nodes_differing_from_A']:>8d}   " + " ".join(f"{k} {b:.2f}" for k, b in r["bits_by_kind"].items()))
        for c in CRITERIA:
            sel = [r for r in rows if r["criterion"] == c]
            print(f"   mean {c:<5} top1 {np.mean([r['top1'] for r in sel]):6.1%}  loss {np.mean([r['loss'] for r in sel]):.4f}"
                  + ("" if c == "A" else f"  d loss vs A {np.mean([r['vs_A']['dloss'] for r in sel]):+.4f} ± "
                     f"{np.sqrt(sum(r['vs_A']['dloss_se'] ** 2 for r in sel)) / len(sel):.4f}"))
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
