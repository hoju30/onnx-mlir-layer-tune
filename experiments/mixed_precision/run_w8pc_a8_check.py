#!/usr/bin/env python3
"""With the weights fixed (W8 per-output-channel), how much do int8 activations still cost, and where?

Weights of the weight-bearing nodes: int8, one scale per output channel (run_w8_per_channel_check.py). Activations:
the pass's a=int8 (affine per-tensor range from the calibration images) on each such node's input and output.
  W8pc                 activations FP32 (the baseline after the weight fix)
  W8pc + A8            int8 activations on every weight-bearing node
  W8pc + A8 on <group> int8 activations on one layer group only (the others FP32): stem, dw, pw-expand, pw-project,
                       se (1x1 convs on a 1x1 input), head (last conv), fc
  W8pt + A8            the pass as it is (per-tensor weights), for reference
ORT QDQ simulation on the held-out images.
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
from run_w8_per_channel_check import MODELS, per_channel_int  # noqa: E402

GROUPS = ("stem", "dw", "pw-expand", "pw-project", "se", "head", "fc")


def node_groups(graph, weighted):
    """Layer group of each weight-bearing node, from its weight and the inferred input shape."""
    inferred = onnx.shape_inference.infer_shapes(graph)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(inferred.graph.value_info) + list(inferred.graph.input)}
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    convs = [n for n in weighted if by_name[n["name"]].op_type == "Conv"]
    out = {}
    for n in weighted:
        node = by_name[n["name"]]
        W = onnx.numpy_helper.to_array(inits[node.input[1]])
        group = next((onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "group"), 1)
        x = shapes.get(node.input[0], [])
        if node.op_type != "Conv":
            out[n["name"]] = "fc"
        elif W.shape[1] == 3 and n is convs[0]:
            out[n["name"]] = "stem"
        elif group > 1:
            out[n["name"]] = "dw"
        elif n is convs[-1]:
            out[n["name"]] = "head"
        elif len(x) == 4 and x[2] == 1 and x[3] == 1:
            out[n["name"]] = "se"
        else:
            out[n["name"]] = "pw-expand" if W.shape[0] > W.shape[1] else "pw-project"
    return out


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
    use_per_channel = [True]

    def patched(w, fmt, oracle):
        return per_channel_int(w, 8) if fmt == "int8" and use_per_channel[0] else original(w, fmt, oracle)

    erm.np_fake_quant_weight = patched
    out = {}
    for m in MODELS:
        md = HERE / "../imagenet100/model_dataset" / m
        graph = onnx.load(str(md / "model.onnx"))
        meta = json.loads((md / "metadata.json").read_text())
        nodes = erm.supported_nodes(graph)
        weighted = [n for n in nodes if n.get("weight_const")]
        groups = node_groups(graph, weighted)
        cx = erm.preprocess_by_source(args.images_dir, {"images": calib}, meta)
        ex = erm.preprocess_by_source(args.images_dir, {"images": held}, meta)
        ranges = erm.calibrate_ranges(graph, nodes, cx)

        def run(act_nodes, per_channel=True):
            use_per_channel[0] = per_channel
            cfg = {n["name"]: {"w": "int8", "a": "int8" if n["name"] in act_nodes else "fp32"} for n in weighted}
            z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(ey)).item()
            return {"top1": float(((z.argmax(1) == ey) & finite).mean()), "loss": loss}

        z = erm.simulate_in_onnxruntime(graph, nodes, {}, ranges, ex, None)
        fp32 = {"top1": float((z.argmax(1) == ey).mean()),
                "loss": F.cross_entropy(torch.from_numpy(z), torch.from_numpy(ey)).item()}
        names = {n["name"] for n in weighted}
        res = {"FP32": fp32, "W8pc": run(set()), "W8pc + A8": run(names),
               **{f"W8pc + A8 on {g}": run({n for n in names if groups[n] == g}) for g in GROUPS
                  if any(groups[n] == g for n in names)},
               "W8pt + A8 (pass as is)": run(names, per_channel=False)}
        counts = {g: sum(groups[n] == g for n in names) for g in GROUPS}
        out[m] = {"groups": groups, "group_counts": counts, "results": res}
        base = res["W8pc"]
        print(f"\n=== {m} ({len(names)} weight-bearing nodes: "
              + ", ".join(f"{g} {c}" for g, c in counts.items() if c) + f"; {len(ey)} held-out images)")
        for k, r in res.items():
            d = "" if k in ("FP32", "W8pc") else f"   vs W8pc: top-1 {100 * (r['top1'] - base['top1']):+6.1f} pp, loss {r['loss'] - base['loss']:+.3f}"
            print(f"  {k:<28} top-1 {r['top1']:6.1%}  loss {r['loss']:.3f}{d}")
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
