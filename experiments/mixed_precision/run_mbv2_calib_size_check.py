#!/usr/bin/env python3
"""MobileNetV2: does more calibration data fix the activation ranges (per-tensor and per-channel min/max)?

Calibration sets: the original 10 images, plus the first N-10 images of HTQ/imagenet1k_2050 that no run uses
(disjoint from the held-out images). For each N: ranges by min/max (per tensor, as the pass; per channel, as
run_mbv2_outlier_check.py), then on the same held-out images
  coverage   fraction of held-out elements outside the ranges (all weighted nodes' X and Y), and for the classifier
             the fraction of images whose true-class logit is above its channel's calibrated max
  damage     int4 activation-only loss increase over FP32 of a few nodes (one at a time), both range modes
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from analyze_fisher_vs_sqnr import layer_type  # noqa: E402
from eval_random_mixed_precision import preprocess_by_source  # noqa: E402
from run_mbv2_act_range_check import chan_reduce  # noqa: E402
from run_mbv2_fisher_term_ablation import act_fake_quant, fold_batchnorms  # noqa: E402
from run_mbv2_outlier_check import channel_fake_quant, channel_qparams  # noqa: E402

DAMAGE_NODES = (0, 3, 49, 51, 52)   # stem, pw (node_Conv_815), last dw (node_Conv_907), conv_head, classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--pool-dir", type=Path, default=Path("/home/hoju/test/HTQ/imagenet1k_2050/images"))
    ap.add_argument("--sizes", default="10,50,200,1040")
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sizes = [int(s) for s in args.sizes.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    meta = json.loads((args.model_dir / "metadata.json").read_text())
    run = json.loads((args.images_dir / "manifest.json").read_text())
    pool = json.loads((args.pool_dir / "manifest.json").read_text())
    used = {e["file"] for e in run["images"]}
    calib = [e for e in run["images"] if e["calibration"]]
    extra = [e for e in pool["images"] if e["file"] not in used]
    held = [e for e in run["images"] if not e["calibration"]][:args.num_eval]
    assert max(sizes) <= len(calib) + len(extra)
    cal_x = torch.from_numpy(preprocess_by_source(args.pool_dir, {"images": calib + extra[:max(sizes) - len(calib)]}, meta))
    ex = torch.from_numpy(preprocess_by_source(args.pool_dir, {"images": held}, meta))
    ey = torch.tensor([e["label"] for e in held])
    log(f"{len(cal_x)} calibration candidates ({len(extra)} unused pool images, "
        f"{len({e['label'] for e in extra})} classes); {len(ey)} held-out images")

    model = __import__("timm").create_model(args.timm_name, pretrained=True).eval()
    fold_batchnorms(model)
    layers = [m for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    kind = {i: layer_type(i, len(layers)) for i in range(len(layers))}
    cap = {}
    for i, m in enumerate(layers):
        m.register_forward_hook(lambda mod, inp, out, i=i: cap.__setitem__(i, (inp[0], out)))

    @torch.no_grad()
    def channel_ranges(x):
        lo, hi = {}, {}
        for b in range(0, len(x), 50):
            model(x[b:b + 50])
            for i, (tx, ty) in cap.items():
                for j, t in enumerate((tx, ty)):
                    l, h = chan_reduce(t, torch.amin), chan_reduce(t, torch.amax)
                    lo[(i, j)] = l if (i, j) not in lo else torch.minimum(lo[(i, j)], l)
                    hi[(i, j)] = h if (i, j) not in hi else torch.maximum(hi[(i, j)], h)
        return lo, hi

    with torch.no_grad():
        z_held = model(ex)
        held_acts = {k: v for k, v in cap.items()}
    base = F.cross_entropy(z_held, ey, reduction="none")
    rows = []
    for n_cal in sizes:
        lo, hi = channel_ranges(cal_x[:n_cal])
        cov = {}
        for (i, j), l in lo.items():
            t, h = held_acts[i][j], hi[(i, j)]
            shp = (1, -1, 1, 1) if t.dim() == 4 else (1, -1)
            cov[(i, j)] = (float(((t < l.min()) | (t > h.max())).float().mean()),
                           float(((t < l.view(shp)) | (t > h.view(shp))).float().mean()))
        fc = len(layers) - 1
        true_clip = float((z_held[torch.arange(len(ey)), ey] > hi[(fc, 1)][ey]).float().mean())
        conv = [k for k in cov if kind[k[0]] not in ("head", "fc")]
        row = {"n_cal": n_cal, "classes": len({e["label"] for e in (calib + extra)[:n_cal]}),
               "out_conv_median": [float(np.median([cov[k][m] for k in conv])) for m in (0, 1)],
               "out_head": {s: cov[(fc - 1, j)] for j, s in enumerate("XY")},
               "out_fc": {s: cov[(fc, j)] for j, s in enumerate("XY")}, "fc_true_logit_clipped": true_clip,
               "damage": {}}
        for i in DAMAGE_NODES:
            m = layers[i]
            for mode in ("per-tensor", "per-channel"):
                def q(t, j, mode=mode, i=i):
                    if mode == "per-tensor":
                        return act_fake_quant(t, "int4", (float(lo[(i, j)].min()), float(hi[(i, j)].max())))
                    return channel_fake_quant(t, 4, *channel_qparams(lo[(i, j)], hi[(i, j)], 4))
                hs = [m.register_forward_pre_hook(lambda mod, inp, q=q: (q(inp[0], 0),)),
                      m.register_forward_hook(lambda mod, inp, out, q=q: q(out, 1))]
                with torch.no_grad():
                    z = model(ex)
                for h in hs:
                    h.remove()
                d = F.cross_entropy(z, ey, reduction="none") - base
                row["damage"][f"{kind[i]}#{i} {mode}"] = {"dloss": float(d.mean()),
                                                         "se": float(d.std() / np.sqrt(len(d))),
                                                         "top1": float((z.argmax(1) == ey).float().mean())}
        cap.clear()
        rows.append(row)
        log(f"N={n_cal} done")
    (args.out_dir / "results.json").write_text(json.dumps({"num_eval": len(ey), "rows": rows}, indent=2) + "\n")

    print(f"\n1. coverage on {len(ey)} held-out images: fraction outside the calibrated range (per-tensor / per-channel)")
    print(f"   {'N':>5} {'classes':>8} | {'conv median':>21} | {'head X':>19} {'head Y':>19} | {'fc X':>19} {'fc Y':>19} | "
          f"{'fc true logit clipped':>21}")
    for r in rows:
        f2 = lambda p: f"{p[0]:.1e} / {p[1]:.1e}"  # noqa: E731
        print(f"   {r['n_cal']:5d} {r['classes']:8d} | {f2(r['out_conv_median']):>21} | {f2(r['out_head']['X']):>19} "
              f"{f2(r['out_head']['Y']):>19} | {f2(r['out_fc']['X']):>19} {f2(r['out_fc']['Y']):>19} | "
              f"{r['fc_true_logit_clipped']:21.0%}")
    print(f"\n2. int4 activation-only damage (dloss on held-out; FP32 top-1 {(z_held.argmax(1) == ey).float().mean():.1%})")
    keys = list(rows[0]["damage"])
    print(f"   {'node / range':<26}" + "".join(f"{'N=' + str(r['n_cal']):>18}" for r in rows))
    for k in keys:
        print(f"   {k:<26}" + "".join(f"{r['damage'][k]['dloss']:+9.3f}±{r['damage'][k]['se']:.3f}  " for r in rows))
    print(f"   {'fc per-channel top-1':<26}" + "".join(f"{r['damage']['fc#52 per-channel']['top1']:18.1%}" for r in rows))


if __name__ == "__main__":
    main()
