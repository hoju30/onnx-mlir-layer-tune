#!/usr/bin/env python3
"""HTQ-style SNIP-sensitivity-guided Pareto config generator for
--convert-onnx-to-mixed-precision, filling the gap
eval_random_mixed_precision.py's own random_config()/uniform_config() leave
open: a sensitivity-guided search instead of a random or uniform per-node
format assignment (claude_lowprecision.md section 24's still-missing
"sensitivity-guided greedy search" baseline).

Pipeline: compute SNIP sensitivity directly on the ONNX graph via
onnxruntime-training's gradient-graph builder, run as a subprocess under a
separate venv (htq_snip_ort.py, see its own docstring for why) -> build
per-(node, format) perturbation tables from real weights/calibration
activations (htq_perturbation.py) -> run the generalized HTQ Pareto tree search
(htq_pareto_search.py) -> pick --top-n evenly-spaced frontier points -> write
them in the exact per-node shape random_config()/uniform_config() already
produce, so eval_random_mixed_precision.py's --htq-configs flag can evaluate
them with zero changes to its own compile/run/measure code.

Sensitivity comes out already keyed by ONNX node name -- no PyTorch model and
no PyTorch<->ONNX node-name mapping is needed at all (an earlier version of
this pipeline did that; onnxruntime-training's gradient graph, built straight
from the ONNX graph, made it unnecessary).

Usage:
  python sample_configs_htq.py \\
      --model-dir experiments/imagenet100/model_dataset/mobilenetv2/timm_ra \\
      --download-shard --top-n 8 --out htq_pareto_configs.json
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_ORT_TRAINING_PYTHON = HERE.parent.parent / "venv-ort-training" / "bin" / "python3"

from eval_random_mixed_precision import (  # noqa: E402
    FORMATS, Tools, PositOracle, calibrate_ranges, download_shard,
    prepare_images, preprocess_by_source, supported_nodes,
)
from htq_perturbation import build_perturbation_tables  # noqa: E402
from htq_pareto_search import leaf_config, pareto_search  # noqa: E402


def tap_output_samples(model, nodes, calib_inputs, max_samples=200_000, seed=0):
    """Per supported node's own output tensor, a flattened (and, if huge,
    subsampled) array of real values over the calibration images -- the
    activation-perturbation counterpart of calibrate_ranges()'s min/max, which
    only gives bounds, not the values needed to actually fake-quant-and-diff."""
    import onnx
    import onnxruntime as ort
    outs_needed = sorted({n["act_tensors"][-1] for n in nodes})
    inits = {i.name for i in model.graph.initializer}
    graph_in = {i.name for i in model.graph.input} - inits
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for t in outs_needed:
        if t not in have and t not in graph_in:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    collected = {t: [] for t in outs_needed}
    for x in calib_inputs:
        result = dict(zip(out_names, sess.run(None, {in_name: x[None]})))
        for t in outs_needed:
            if t in result:
                collected[t].append(np.asarray(result[t], np.float32).ravel())
            elif t in graph_in:
                collected[t].append(np.asarray(x, np.float32).ravel())
    rng = np.random.default_rng(seed)
    samples = {}
    for t, chunks in collected.items():
        arr = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        samples[t] = rng.choice(arr, max_samples, replace=False) if arr.size > max_samples else arr
    return samples


def compute_weight_elems_and_flops_proxy(model, nodes):
    """weight_elems: {node_name: int}. flops_proxy: {node_name: float}, a MACs
    proxy (weight_elems x output-spatial-size for Conv, weight_elems for
    Gemm/MatMul, 0 otherwise) -- not exact FLOPs, just enough to rank BOPs the
    same way HTQ's own bits_w x bits_a x MACs does."""
    import onnx.numpy_helper as numpy_helper
    import onnx.shape_inference
    inits = {i.name: i for i in model.graph.initializer}
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    onnx_nodes_by_name = {n.name: n for n in model.graph.node}

    weight_elems, flops_proxy = {}, {}
    for n in nodes:
        name = n["name"]
        if not n.get("weight_const"):
            flops_proxy[name] = 0.0
            continue
        onnx_node = onnx_nodes_by_name[name]
        w = numpy_helper.to_array(inits[onnx_node.input[1]])
        weight_elems[name] = int(w.size)
        if n["op"] == "Conv":
            out_shape = shapes.get(n["output"], [])
            spatial = 1
            for d in out_shape[2:]:
                spatial *= d
            flops_proxy[name] = w.size * spatial
        else:  # Gemm / MatMul
            flops_proxy[name] = float(w.size)
    return weight_elems, flops_proxy


def pick_evenly_spaced(leaves, top_n):
    ordered = sorted(leaves, key=lambda x: x.cost)
    if top_n >= len(ordered):
        return ordered
    if top_n <= 1:
        return ordered[:1]
    idxs = sorted({round(i * (len(ordered) - 1) / (top_n - 1)) for i in range(top_n)})
    return [ordered[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="dir with model.onnx + metadata.json")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight-formats", default=",".join(FORMATS))
    ap.add_argument("--act-formats", default=",".join(FORMATS))
    ap.add_argument("--prune-fraction", type=float, default=1 / 20)
    ap.add_argument("--beam-width", type=int, default=2000,
                     help="hard cap on the Pareto frontier kept after each layer (see "
                          "htq_pareto_search.pareto_search's docstring: HTQ's own pruning "
                          "fraction alone does not converge on a 60+-node real network)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--parquet", type=Path)
    src.add_argument("--download-shard", action="store_true")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--num-calibration", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--ort-training-python", type=Path, default=DEFAULT_ORT_TRAINING_PYTHON,
                     help="python interpreter of the separate venv-ort-training venv "
                          "(see htq_snip_ort.py's docstring for why it must be separate)")
    ap.add_argument("--sensitivity-mode", choices=["shared", "split_connection", "split_direct"],
                     default="split_connection",
                     help="shared = HTQ baseline (weight-only connection sensitivity reused for "
                          "both terms); split_connection = independent weight*dL/dweight and "
                          "activation*dL/dactivation; split_direct = independent raw dL/dW, dL/dA "
                          "(see htq_snip_ort.py's docstring for the exact definitions)")
    args = ap.parse_args()

    weight_formats = args.weight_formats.split(",")
    act_formats = args.act_formats.split(",")
    model_path = args.model_dir / "model.onnx"
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    images_dir = args.images_dir or HERE / "runs" / f"{args.model_dir.parent.name}_{args.model_dir.name}" / "images"
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="htq-"))
    work.mkdir(parents=True, exist_ok=True)

    print("== loading data ==", flush=True)
    parquet = args.parquet
    if args.download_shard and not (images_dir / "manifest.json").exists():
        parquet = download_shard(work)
    manifest = prepare_images(images_dir, parquet, args.num_images, args.num_calibration, args.data_seed)
    labels = np.array([e["label"] for e in manifest["images"]])
    calib_mask = np.array([e["calibration"] for e in manifest["images"]])
    inputs = preprocess_by_source(images_dir, manifest, meta)
    calib_inputs, calib_labels = inputs[calib_mask], labels[calib_mask]
    print(f"  {len(labels)} images, {int(calib_mask.sum())} for calibration/SNIP", flush=True)

    print("== loading model ==", flush=True)
    import onnx
    onnx_model = onnx.load(str(model_path))
    nodes = supported_nodes(onnx_model)
    print(f"  {len(nodes)} pass-supported nodes ({sum(1 for n in nodes if n.get('weight_const'))} weight-bearing)",
          flush=True)

    print("== SNIP sensitivity (onnxruntime-training gradient graph, native ONNX node names) ==", flush=True)
    calib_inputs_path, calib_labels_path = work / "calib_inputs.npy", work / "calib_labels.npy"
    np.save(calib_inputs_path, calib_inputs)
    np.save(calib_labels_path, calib_labels)
    sensitivity_path = work / "htq_sensitivity.json"
    subprocess.run([str(args.ort_training_python), str(HERE / "htq_snip_ort.py"),
                    "--onnx", str(model_path), "--inputs", str(calib_inputs_path),
                    "--labels", str(calib_labels_path), "--mode", args.sensitivity_mode,
                    "--out", str(sensitivity_path)], check=True)
    sensitivity_json = json.loads(sensitivity_path.read_text())
    sensitivity_w, sensitivity_a = sensitivity_json["weight"], sensitivity_json["activation"]
    print(f"  scored {len(sensitivity_w)} weight and {len(sensitivity_a)} activation "
          f"sensitivities (mode={args.sensitivity_mode})", flush=True)

    print("== calibration + perturbation tables ==", flush=True)
    ranges = calibrate_ranges(onnx_model, nodes, calib_inputs)
    activation_samples = tap_output_samples(onnx_model, nodes, calib_inputs)
    import onnx.numpy_helper as numpy_helper
    inits = {i.name: i for i in onnx_model.graph.initializer}
    onnx_nodes_by_name = {n.name: n for n in onnx_model.graph.node}
    weights_by_node = {
        n["name"]: numpy_helper.to_array(inits[onnx_nodes_by_name[n["name"]].input[1]])
        for n in nodes if n.get("weight_const")
    }
    tools = Tools()
    posit_oracle = PositOracle(tools, work)
    all_formats = sorted(set(weight_formats) | set(act_formats))
    weight_table, activation_table = build_perturbation_tables(
        nodes, all_formats, weights_by_node, activation_samples, ranges, posit_oracle)

    print("== weight bytes / BOPs proxy ==", flush=True)
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(onnx_model, nodes)

    print("== Pareto tree search ==", flush=True)
    leaves = pareto_search(nodes, weight_formats, act_formats, sensitivity_w, sensitivity_a, weight_table,
                            activation_table, weight_elems, flops_proxy,
                            prune_fraction=args.prune_fraction, beam_width=args.beam_width, log=print)
    print(f"  {len(leaves)} frontier points", flush=True)

    picked = pick_evenly_spaced(leaves, args.top_n)
    op_by_name = {n["name"]: n["op"] for n in nodes}
    configs = []
    for i, leaf in enumerate(picked):
        formats = leaf_config(leaf)
        for name, f in formats.items():
            f["op"] = op_by_name[name]
        configs.append({
            "id": f"htq_pareto_{i:03d}",
            "cost_weight_bytes": leaf.cost,
            "bops_proxy": leaf.bops,
            "predicted_perturbation": leaf.profit,
            "formats": formats,
        })

    args.out.write_text(json.dumps({"configs": configs}, indent=1))
    print(f"wrote {len(configs)} configs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
