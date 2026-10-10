#!/usr/bin/env python3
"""W8 per-channel vs per-tensor weights, activations FP32: do the all-zero channels and the accuracy recover together?

The pass quantizes a weight per tensor (symmetric int8, scale = max|W| / 127, round half away from zero). After
BatchNorm folding, a depthwise conv's channels differ in scale so much that whole output channels can round to
zero. Variants (only the weights of the weight-bearing nodes quantized; ORT QDQ simulation, held-out images):
  per-tensor            the pass's arithmetic
  per-channel           the same arithmetic with one scale per output channel (axis 0: Conv [out, ...], Gemm transB=1)
  per-channel, fix-k    per-channel only on the layers with at least k all-zero channels under per-tensor
All-zero channel: every weight of the output channel quantizes to exactly 0 (the channel keeps only its bias).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402

MODELS = ("efficientnet_b0/timm_ra", "mobilenetv2/timm_ra", "resnet18/torchvision_v1")


def per_channel_int(w, bits):
    """np_fake_quant_weight's int arithmetic, one scale per output channel (axis 0)."""
    w = np.asarray(w, np.float32)
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    flat = w.reshape(w.shape[0], -1)
    maxabs = np.abs(flat).max(1).astype(np.float32)
    scale = np.where(maxabs > 0, maxabs / np.float32(qmax), np.float32(1)).astype(np.float32)
    t = (flat / scale[:, None]).astype(np.float64)
    q = np.clip(np.sign(t) * np.floor(np.abs(t) + 0.5), qmin, qmax)
    return (q.astype(np.float32) * scale[:, None]).astype(np.float32).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    man = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    held = [e for e in man["images"] if not e["calibration"]]
    ey = np.array([e["label"] for e in held])
    original = erm.np_fake_quant_weight
    per_channel_keys = set()

    def patched(w, fmt, oracle):
        if fmt == "int8" and np.asarray(w).tobytes() in per_channel_keys:
            return per_channel_int(w, 8)
        return original(w, fmt, oracle)

    erm.np_fake_quant_weight = patched
    out = {}
    for m in MODELS:
        md = HERE / "../imagenet100/model_dataset" / m
        graph = onnx.load(str(md / "model.onnx"))
        meta = json.loads((md / "metadata.json").read_text())
        nodes = erm.supported_nodes(graph)
        inits = {i.name: i for i in graph.graph.initializer}
        by_name = {n.name: n for n in graph.graph.node}
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta)
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ranges = erm.calibrate_ranges(graph, nodes, cx)
        weighted = [n for n in nodes if n.get("weight_const")]
        layers = []
        for n in weighted:
            W = onnx.numpy_helper.to_array(inits[by_name[n["name"]].input[1]])
            zero = {}
            for mode, q in (("per-tensor", original(W, "int8", None)), ("per-channel", per_channel_int(W, 8))):
                zero[mode] = int((np.abs(q.reshape(q.shape[0], -1)).max(1) == 0).sum())
            rel = {mode: float(((q.astype(np.float64) - W) ** 2).sum() / (W.astype(np.float64) ** 2).sum())
                   for mode, q in (("per-tensor", original(W, "int8", None)), ("per-channel", per_channel_int(W, 8)))}
            layers.append({"node": n["name"], "shape": list(W.shape), "channels": int(W.shape[0]),
                           "zero_channels": zero, "rel_error": rel, "key": W.tobytes()})
        w8 = {n["name"]: {"w": "int8", "a": "fp32"} for n in weighted}

        def evaluate(cfg, pc_nodes):
            per_channel_keys.clear()
            per_channel_keys.update(l["key"] for l in layers if l["node"] in pc_nodes)
            z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, ex, None)
            loss = F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey)).item()
            return float((z.argmax(1) == ey).mean()), loss

        dead = {k: {l["node"] for l in layers if l["zero_channels"]["per-tensor"] >= k} for k in (1, 3)}
        dead[10] = {l["node"] for l in layers if l["zero_channels"]["per-tensor"] >= 0.1 * l["channels"]}
        runs = [("FP32", {}, set()), ("W8 per-tensor", w8, set()),
                (f"W8 per-channel only on the {len(dead[10])} layers with >=10% zero channels", w8, dead[10]),
                (f"W8 per-channel only on the {len(dead[1])} layers with any zero channel", w8, dead[1]),
                ("W8 per-channel (all layers)", w8, {l["node"] for l in layers})]
        res = []
        for label, cfg, pc in runs:
            top1, loss = evaluate(cfg, pc)
            zc = sum(l["zero_channels"]["per-channel" if l["node"] in pc else "per-tensor"] for l in layers) if cfg else 0
            res.append({"setting": label, "top1": top1, "loss": loss, "zero_channels": zc})
        tot = sum(l["channels"] for l in layers)
        out[m] = {"total_channels": tot, "runs": res,
                  "layers": [{k: v for k, v in l.items() if k != "key"} for l in layers]}
        print(f"\n=== {m}: {len(layers)} weighted nodes, {tot} output channels; activations FP32, "
              f"{len(ey)} held-out images")
        for r in res:
            print(f"  {r['setting']:<62} all-zero channels {r['zero_channels']:4d} ({r['zero_channels'] / tot:5.1%})   "
                  f"top-1 {r['top1']:6.1%}   loss {r['loss']:.3f}")
        worst = sorted(layers, key=lambda l: -l["zero_channels"]["per-tensor"] / l["channels"])[:5]
        print("  worst layers under per-tensor (all-zero channels, relative weight error per-tensor -> per-channel):")
        for l in worst:
            print(f"    {l['node']:<18} {str(tuple(l['shape'])):<18} {l['zero_channels']['per-tensor']:4d}/{l['channels']:<4d} "
                  f"({l['zero_channels']['per-tensor'] / l['channels']:5.1%})   rel. error "
                  f"{l['rel_error']['per-tensor']:.2e} -> {l['rel_error']['per-channel']:.2e}")
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
