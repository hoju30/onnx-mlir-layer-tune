#!/usr/bin/env python3
"""Qwen3-0.6B forward pass, layer by layer: onnx-mlir (the build levels of the statistics graphs: Krnl -O3, clang
-O0), onnxruntime and PyTorch FP32 against PyTorch FP64, on the MLP tensors of all 28 layers:
  X (gate/up input, post-attention RMSNorm), zg (gate output), zu (up output), h (down input = SiLU(zg) * zu),
  yd (down output) and the logits.
The tensors are added as outputs of a copy of the exported ONNX (same external weights), so onnx-mlir and
onnxruntime run exactly the graph the statistics graphs are built from. Per tensor and window: compare() of each
backend against FP64, the largest |value| and its position.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from qwen3_swiglu_common import PREP, compare, erm, load_model, node_mapping, torch

HERE = Path(__file__).resolve().parent


def tap_names(graph):
    """(layer, tensor) -> ONNX tensor name, from the gate / up / down nodes."""
    by_name = {n.name: n for n in graph.graph.node}
    names = {}
    for node, (layer, kind, _) in node_mapping().items():
        n = by_name[node]
        if kind == "gate":
            names[(layer, "X")], names[(layer, "zg")] = n.input[0], n.output[0]
        elif kind == "up":
            names[(layer, "zu")] = n.output[0]
        elif kind == "down":
            names[(layer, "h")], names[(layer, "yd")] = n.input[0], n.output[0]
    return names


def child(so, lib, inputs, out):
    sys.path.insert(0, lib)
    from PyRuntime import OMExecutionSession
    sess = OMExecutionSession(so, erm.SYMBOL_POSTFIX)
    x = np.load(inputs)
    res = {}
    for i in range(len(x)):
        for j, v in enumerate(sess.run([np.ascontiguousarray(x[i:i + 1])])):
            res[f"w{i}_o{j}"] = np.asarray(v)
    np.savez(out, **res)


def main():
    if sys.argv[1:2] == ["--_child"]:
        return child(*sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    import onnx
    graph = onnx.load(str(PREP / "model.onnx"), load_external_data=False)
    names = tap_names(graph)
    keys = sorted(names)
    for k in keys:
        graph.graph.output.append(onnx.helper.make_tensor_value_info(names[k], onnx.TensorProto.FLOAT, None))
    (out / "model.onnx.data").hardlink_to(PREP / "model.onnx.data")   # onnxruntime rejects data outside the folder
    onnx.save(graph, str(out / "model_taps.onnx"))
    calib = np.load(PREP / "validation_tokens.npy")[:args.windows]
    np.save(out / "calib.npy", calib)
    outputs = ["logits"] + [f"L{l}.{t}" for l, t in keys]

    # onnx-mlir, the statistics builds' levels
    tools = erm.Tools()
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", out / erm.SYMBOL_POSTFIX, out / "model_taps.onnx"])
    erm.run([tools.opt, out / "model.onnx.mlir", "--emit-bytecode", "-o", out / "model.mlirbc"])
    (out / "model.onnx.mlir").unlink()
    so, info = erm.build_so(tools, out, "fp32", out / "model.mlirbc", None, None, opt="O0", krnl_opt="O3",
                            constants_to_file=True, bytecode=True)
    for p in out.iterdir():
        if p.suffix in (".mlir", ".ll", ".bc") or p.name.startswith("fp32.") and p.suffix == ".mlirbc":
            p.unlink()
    erm.run([sys.executable, __file__, "--_child", so, tools.lib, out / "calib.npy", out / "native.npz"])
    nat = np.load(out / "native.npz")
    native = [{o: nat[f"w{i}_o{j}"][0] for j, o in enumerate(outputs)} for i in range(len(calib))]
    log(f"onnx-mlir: built ({ {k: round(v) for k, v in info['steps'].items()} }) and run")

    import onnxruntime as ort
    so_opts = ort.SessionOptions()
    so_opts.intra_op_num_threads = 8
    so_opts.log_severity_level = 3
    sess = ort.InferenceSession(str(out / "model_taps.onnx"), so_opts, providers=["CPUExecutionProvider"])
    ortv = [dict(zip(outputs, (v[0] for v in sess.run(None, {"input_ids": calib[i:i + 1]})))) for i in range(len(calib))]
    del sess
    (out / "model.onnx.data").unlink()        # onnx.load refuses external data with several hard links
    log("onnxruntime run")

    def torch_run(dtype):
        model, wrapper = load_model(dtype)
        mods = dict(model.named_modules())
        res = []
        for ids in calib:
            cap = {}
            hooks = []
            for l in range(28):
                mlp = mods[f"model.layers.{l}.mlp"]
                hooks.append(mlp.gate_proj.register_forward_hook(
                    lambda m, i, o, l=l: cap.update({f"L{l}.X": i[0], f"L{l}.zg": o})))
                hooks.append(mlp.up_proj.register_forward_hook(lambda m, i, o, l=l: cap.update({f"L{l}.zu": o})))
                hooks.append(mlp.down_proj.register_forward_hook(
                    lambda m, i, o, l=l: cap.update({f"L{l}.h": i[0], f"L{l}.yd": o})))
            with torch.no_grad():
                z = wrapper(torch.from_numpy(ids[None]))
            for h in hooks:
                h.remove()
            r = {k: v[0].double().numpy() for k, v in cap.items()}
            r["logits"] = z[0].double().numpy()
            res.append(r)
        del model, wrapper
        return res
    t32, t64 = torch_run(torch.float32), torch_run(torch.float64)
    log("PyTorch FP32 / FP64 run")

    rows = []
    for i in range(len(calib)):
        for o in outputs:
            ref = t64[i][o]
            row = {"window": i, "tensor": o, "absmax": float(np.abs(ref).max()),
                   "absmax_pos": int(np.abs(ref).reshape(ref.shape[0], -1).max(1).argmax()),
                   "absmax_channel": int(np.abs(ref).reshape(ref.shape[0], -1).max(0).argmax())}
            for name, src in (("native", native), ("ort", ortv), ("torch32", t32)):
                row[f"{name}_vs_fp64"] = compare(src[i][o], ref)
            # positions other than the largest one
            p = row["absmax_pos"]
            mask = np.ones(ref.shape[0], bool)
            mask[p] = False
            row["native_vs_fp64_without_absmax_pos"] = compare(native[i][o][mask], ref[mask])
            rows.append(row)
    (out / "forward.json").write_text(json.dumps(rows) + "\n")
    (out / "build.json").write_text(json.dumps(info["steps"]) + "\n")
    log("done")


if __name__ == "__main__":
    main()
