#!/usr/bin/env python3
"""GPT-2 small, A8 split by side: is the activation loss from quantizing the projections' inputs or outputs?

Weights stay FP32. On the 48 projection nodes of an existing run_gpt2_criteria.py run (same ONNX model,
calibration ranges and test windows), int8 activation fake-quant on
  input only | output only | input + output (the pass's a=int8; must reproduce run_gpt2_w8a8_split.py's "A only")
simulate_in_onnxruntime always quantizes both sides for a=, so this splices the same Div/Add/Round/Clip/Sub/Mul
chain itself, with the same int_qparams and calibration ranges, on the chosen side(s) only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_random_mixed_precision import int_qparams, supported_nodes  # noqa: E402
from run_gpt2_w8a8_split import score  # noqa: E402

SIDES = {"FP32": (False, False), "input only": (True, False), "output only": (False, True),
         "input + output": (True, True)}


def simulate_int_act(model, nodes, bits, ranges, inputs, quant_in, quant_out):
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    by_name = {n["name"]: n for n in nodes}
    count = [0]

    def fresh(prefix):
        count[0] += 1
        return f"__{prefix}{count[0]}"

    def const(value):
        name = fresh("c")
        g.initializer.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        return name

    def fake_quant(src, dst, lohi):
        scale, zp = int_qparams(*lohi, bits)
        s, z = const(np.float32(scale)), const(np.float32(zp))
        lo, hi = const(np.float32(-(1 << (bits - 1)))), const(np.float32((1 << (bits - 1)) - 1))
        t = [fresh("t") for _ in range(5)]
        return [helper.make_node("Div", [src, s], [t[0]]), helper.make_node("Add", [t[0], z], [t[1]]),
                helper.make_node("Round", [t[1]], [t[2]]), helper.make_node("Clip", [t[2], lo, hi], [t[3]]),
                helper.make_node("Sub", [t[3], z], [t[4]]), helper.make_node("Mul", [t[4], s], [dst])]

    new_nodes = []
    for orig in g.node:
        info = by_name.get(orig.name)
        if info is None or not (quant_in or quant_out):
            new_nodes.append(orig)
            continue
        n = onnx.NodeProto()
        n.CopyFrom(orig)
        pre, post = [], []
        tensors = info["act_tensors"]
        if quant_in:
            for i, t in enumerate(tensors[:-1]):
                dst = f"{t}__q{i}_{orig.name}"
                pre += fake_quant(n.input[i], dst, ranges[t])
                n.input[i] = dst
        if quant_out:
            y = n.output[0]
            raw = f"{y}__raw_{orig.name}"
            n.output[0] = raw
            post = fake_quant(raw, y, ranges[tensors[-1]])
        new_nodes += pre + [n] + post
    del g.node[:]
    g.node.extend(new_nodes)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.stack([sess.run(None, {name: x[None]})[0][0] for x in inputs]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--split-run", type=Path, default=HERE / "runs/gpt2_w8a8_split_20260930",
                    help="run_gpt2_w8a8_split.py output whose 'A only' row input + output must reproduce")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    tokens = np.load(src / "test_tokens.npy")
    mapping = json.loads((src / "node_mapping.json").read_text())
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    assert len(nodes) == 48, len(nodes)

    rows = []
    for label, (qi, qo) in SIDES.items():
        z = simulate_int_act(graph, nodes, 8, ranges, tokens, qi, qo)
        assert np.isfinite(z).all(), label
        nll, ppl, top1 = score(z, tokens)
        rows.append({"variant": label, "nll": nll, "perplexity": ppl, "top1": top1})
        print(f"[{time.time() - t0:5.1f}s] {label:<15} NLL {nll:.4f}  PPL {ppl:8.2f}  next-token top1 {top1:.2%}",
              flush=True)
        del z

    prev = next(r for r in json.loads((args.split_run / "results.json").read_text())["rows"]
                if r["variant"].startswith("A only"))
    assert abs(prev["perplexity"] - rows[-1]["perplexity"]) < 1e-6 * prev["perplexity"], (prev, rows[-1])
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "weights": "fp32", "activation_format": "int8",
         "source_run": str(src), "predicted_tokens": int(tokens[:, 1:].size), "rows": rows,
         "input_output_matches_a_only": True}, indent=2) + "\n")
    print(f"input + output reproduces the 'A only' PPL ({prev['perplexity']:.4f}); wrote {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
