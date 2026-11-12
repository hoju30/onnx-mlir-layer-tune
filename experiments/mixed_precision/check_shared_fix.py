#!/usr/bin/env python3
"""Can the HTQ "shared" criterion (one connection sensitivity S_i per node for both terms) be made to work with
--convert-onnx-to-mixed-precision? Candidate fixes, all scored by S_i * (weight term + activation term):
  shared      as sample_configs_htq.py: plain L2 of W - Q(W), plus plain L2 of the node's OUTPUT activation error
  shared_v1   activation term over every tensor the pass quantizes for a=: the node's input(s) AND its output
  shared_v2   v1's tensors, but every term a relative noise ||x - Q(x)||^2 / ||x||^2 (squared as in HTQ's Eq. 5),
              so weight and activation terms share one unit and no longer scale with tensor or sample counts
  shared_v3   every term referred to the node's OUTPUT on the real calibration inputs x, relative to ||y||^2:
              ||op(x, W - Q(W))||^2 + ||op(x - Q_a(x), W)||^2 + ||y - Q_a(y)||^2, y = op(x, W, b)
  v3_noS      shared_v3 without S_i: does the connection sensitivity add anything?
  sqnr        reference: summed single-node logit noise (sqnr_tables.py), reused from --sqnr-table
Each is solved exactly by MILP at the same budgets and evaluated in onnxruntime simulation (no onnx-mlir compile).
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_random_mixed_precision import (  # noqa: E402
    Tools, PositOracle, calibrate_ranges, int_qparams, np_fake_quant_weight, preprocess_by_source,
    simulate_in_onnxruntime, supported_nodes,
)
from htq_perturbation import activation_perturbation, weight_perturbation  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from sample_configs_htq import DEFAULT_ORT_TRAINING_PYTHON, compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

CRITERIA = ("shared", "shared_v1", "shared_v2", "shared_v3", "v3_noS", "sqnr")


def fake_quant_act(x, fmt, lo_hi):
    """The pass's activation fake-quant, as activation_perturbation() applies it, returning the tensor."""
    import torch
    if fmt == "fp32":
        return x
    if fmt.startswith("int"):
        bits = int(fmt[3:])
        scale, zp = int_qparams(*lo_hi, bits)
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        return (torch.clamp(torch.round(x / scale + zp), qmin, qmax) - zp) * scale
    if fmt == "f16":
        return x.half().float()
    if fmt == "bf16":
        return x.bfloat16().float()
    raise ValueError(f"{fmt}: not simulated here")


def output_referred_noise(model, weighted, calib_inputs, ranges, wformats, aformats, oracle):
    """{node: (N_w {w: ...}, N_a {a: ...})}: each quantization's noise power at the node's output over the
    calibration images, relative to the output's own power."""
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    graph_in = {i.name for i in model.graph.input} - set(inits)
    m = onnx.ModelProto()
    m.CopyFrom(model)
    wanted = sorted({n["act_tensors"][0] for n in weighted} - graph_in)
    have = {o.name for o in m.graph.output}
    for t in wanted:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name, out_names = sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]
    per_image = []
    for x in calib_inputs:
        res = dict(zip(out_names, sess.run(None, {in_name: x[None]})))
        res[in_name] = x[None]
        per_image.append(res)

    out = {}
    for n in weighted:
        node = by_name[n["name"]]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        W = torch.from_numpy(inits[node.input[1]].astype(np.float32))
        b = torch.from_numpy(inits[node.input[2]].astype(np.float32)) if len(node.input) > 2 else None
        if node.op_type == "Conv":
            pads = attrs.get("pads", [0, 0, 0, 0])
            assert pads[:2] == pads[2:], f"{n['name']}: asymmetric pads"
            op = lambda x, w, bias=None: F.conv2d(x, w, bias, stride=attrs.get("strides", [1, 1]),  # noqa: E731
                                                  padding=pads[:2], dilation=attrs.get("dilations", [1, 1]),
                                                  groups=attrs.get("group", 1))
        else:
            assert node.op_type == "Gemm" and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
            wt = (lambda w: w) if attrs.get("transB", 0) else (lambda w: w.t())
            op = lambda x, w, bias=None: F.linear(x.flatten(1), wt(w), bias)  # noqa: E731
        dW = {f: W - torch.from_numpy(np_fake_quant_weight(W.numpy(), f, oracle).astype(np.float32))
              for f in wformats if f != "fp32"}
        xname, yname = n["act_tensors"][0], n["act_tensors"][-1]
        p_y, p_w, p_a = 0.0, dict.fromkeys(dW, 0.0), dict.fromkeys((f for f in aformats if f != "fp32"), 0.0)
        with torch.no_grad():
            for res in per_image:
                x = torch.from_numpy(np.asarray(res[xname], np.float32))
                y = op(x, W, b)
                p_y += float((y.double() ** 2).sum())
                for f, d in dW.items():
                    p_w[f] += float((op(x, d).double() ** 2).sum())
                for f in p_a:
                    p_a[f] += float((op(x - fake_quant_act(x, f, ranges[xname]), W).double() ** 2).sum())
                    p_a[f] += float(((y - fake_quant_act(y, f, ranges[yname])).double() ** 2).sum())
        out[n["name"]] = ({"fp32": 0.0, **{f: v / p_y for f, v in p_w.items()}},
                          {"fp32": 0.0, **{f: v / p_y for f, v in p_a.items()}})
    return out


def tap_samples(model, tensors, calib_inputs, max_samples=200_000, seed=0):
    """{tensor: flattened (subsampled) calibration values} for any activation tensors, graph input included."""
    import onnx
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(model)
    graph_in = {i.name for i in model.graph.input} - {i.name for i in model.graph.initializer}
    have = {o.name for o in m.graph.output}
    for t in tensors:
        if t not in have and t not in graph_in:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name, out_names = sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]
    chunks = {t: [] for t in tensors}
    for x in calib_inputs:
        res = dict(zip(out_names, sess.run(None, {in_name: x[None]})))
        for t in tensors:
            chunks[t].append(np.asarray(x if t in graph_in else res[t], np.float32).ravel())
    rng = np.random.default_rng(seed)
    out = {}
    for t, c in chunks.items():
        arr = np.concatenate(c)
        out[t] = rng.choice(arr, max_samples, replace=False) if arr.size > max_samples else arr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--sqnr-table", type=Path, required=True, help="sqnr_table.json for the same model/images/menu")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--ort-training-python", type=Path, default=DEFAULT_ORT_TRAINING_PYTHON)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(f) for f in args.budgets.split(",")]
    work = Path(tempfile.mkdtemp(prefix="shared-fix-"))
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    import onnx
    model = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]][:args.num_eval]
    cx = preprocess_by_source(args.images_dir, {"images": calib}, meta)
    ex = preprocess_by_source(args.images_dir, {"images": held}, meta)
    ey = np.array([e["label"] for e in held])
    nodes = supported_nodes(model)
    ranges = calibrate_ranges(model, nodes, cx)
    oracle = PositOracle(Tools(), work)
    sq = json.loads(args.sqnr_table.read_text())
    assert sq["settings"]["calibration"] == [e["file"] for e in calib], "SQNR table built on other calibration images"
    wformats, aformats, sqnr = sq["settings"]["weight_formats"], sq["settings"]["act_formats"], sq["table"]

    np.save(work / "x.npy", cx)
    np.save(work / "y.npy", np.array([e["label"] for e in calib]))
    subprocess.run([str(args.ort_training_python), str(HERE / "htq_snip_ort.py"), "--onnx",
                    str(args.model_dir / "model.onnx"), "--inputs", str(work / "x.npy"), "--labels",
                    str(work / "y.npy"), "--mode", "shared", "--out", str(work / "s.json")],
                   check=True, stdout=subprocess.DEVNULL)
    S = json.loads((work / "s.json").read_text())["weight"]

    weighted = [n for n in nodes if n.get("weight_const")]
    inits = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    samples = tap_samples(model, sorted({t for n in weighted for t in n["act_tensors"]}), cx)
    tab = {}
    for n in weighted:
        name, (x, y) = n["name"], (n["act_tensors"][0], n["act_tensors"][-1])
        W = onnx.numpy_helper.to_array(inits[by_name[name].input[1]]).astype(np.float64)
        w_err = {f: weight_perturbation(W, f, oracle) for f in wformats}
        in_err = {f: activation_perturbation(samples[x], f, ranges[x], oracle) for f in aformats}
        out_err = {f: activation_perturbation(samples[y], f, ranges[y], oracle) for f in aformats}
        wn, xn, yn = np.linalg.norm(W), np.linalg.norm(samples[x]), np.linalg.norm(samples[y])
        tab[name] = {
            "shared": ({f: e for f, e in w_err.items()}, {f: e for f, e in out_err.items()}),
            "shared_v1": ({f: e for f, e in w_err.items()}, {f: in_err[f] + out_err[f] for f in aformats}),
            "shared_v2": ({f: (e / wn) ** 2 for f, e in w_err.items()},
                          {f: (in_err[f] / xn) ** 2 + (out_err[f] / yn) ** 2 for f in aformats}),
        }
    for name, noise in output_referred_noise(model, weighted, cx, ranges, wformats, aformats, oracle).items():
        tab[name]["shared_v3"] = tab[name]["v3_noS"] = noise
    log(f"{len(weighted)} weighted nodes: shared sensitivity and per-variant tables ready")

    def value_fn(c):
        if c == "sqnr":
            return lambda name, w, a: sqnr[name][f"{w}|{a}"]
        s = (lambda name: 1.0) if c == "v3_noS" else (lambda name: S.get(name, 0.0))
        return lambda name, w, a: s(name) * (tab[name][c][0][w] + tab[name][c][1][a])

    names = [n["name"] for n in weighted]
    diag = {}
    print("\nHow each criterion ranks the 53 nodes vs the SQNR-measured noise (Spearman), and its term balance:")
    for c in CRITERIA[:-1]:
        v = value_fn(c)
        rw = spearmanr([v(n, "int4", "fp32") for n in names], [sqnr[n]["int4|fp32"] for n in names])[0]
        ra = spearmanr([v(n, "fp32", "int4") for n in names], [sqnr[n]["fp32|int4"] for n in names])[0]
        ratio = sum(v(n, "fp32", "int4") for n in names) / sum(v(n, "int4", "fp32") for n in names)
        diag[c] = {"spearman_weight_int4": rw, "spearman_act_int4": ra, "act_over_weight_at_int4": ratio}
        print(f"  {c:<10} weight int4 {rw:+.2f}   activation int4 {ra:+.2f}   activation/weight term at int4 {ratio:8.2f}x")

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(model, nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    op_by_name = {n["name"]: n["op"] for n in nodes}
    ref = simulate_in_onnxruntime(model, nodes, {}, ranges, ex, oracle)
    fp32_top1 = float((ref.argmax(1) == ey).mean())
    configs, rows = [], []
    print(f"\nMILP pick per budget, onnxruntime-simulated top-1 on {len(ey)} held-out images (fp32 {fp32_top1:.1%});"
          " size/BOPs used as fractions of all-int8")
    print(f"{'f':>5}" + "".join(f"{c:>22}" for c in CRITERIA))
    for i, f in enumerate(budgets):
        cells = []
        for c in CRITERIA:
            formats, objective, status = milp_select(nodes, wformats, aformats, value_fn(c), weight_elems,
                                                     flops_proxy, f * int8_bytes, f * int8_bops)
            assert status == 0, f"{c} f={f}: MILP status {status}"
            for name, fmt in formats.items():
                fmt["op"] = op_by_name[name]
            size, bops = config_cost(formats, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(model, nodes, formats, ranges, ex, oracle)
            top1 = float(((z.argmax(1) == ey) & np.isfinite(z).all(1)).mean())
            configs.append({"id": f"{c}_{i:03d}", "criterion": c, "budget": f, "cost_weight_bytes": size,
                            "bops_proxy": bops, "objective": objective, "formats": formats})
            rows.append({"criterion": c, "budget": f, "sim_top1": top1, "size_vs_int8": size / int8_bytes,
                         "bops_vs_int8": bops / int8_bops})
            cells.append(f"{top1:6.1%} ({size / int8_bytes:.2f}/{bops / int8_bops:.2f})")
        print(f"{f:5.2f}" + "".join(f"{x:>22}" for x in cells), flush=True)
    (args.out_dir / "configs.json").write_text(json.dumps({"configs": configs}, indent=1) + "\n")
    (args.out_dir / "results.json").write_text(json.dumps(
        {"fp32_sim_top1": fp32_top1, "num_eval": len(ey), "diagnostics": diag, "rows": rows}, indent=1) + "\n")
    log(f"wrote {args.out_dir}/results.json and configs.json")


if __name__ == "__main__":
    main()
