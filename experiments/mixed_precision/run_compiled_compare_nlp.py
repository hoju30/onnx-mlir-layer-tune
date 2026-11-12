#!/usr/bin/env python3
"""Compiled (onnx-mlir) evaluation of the weight-B + activation-C picks vs the SQNR picks for GPT-2 and BERT.

Same as run_compiled_compare.py for the language models: every config is compiled through
--convert-onnx-to-mixed-precision with MIXEDP_WEIGHT_PER_CHANNEL=1, lowered, linked and run with PyRuntime, and
checked against the onnxruntime QDQ simulation with the same per-channel weight arithmetic.
  gpt2  picks from runs/gpt2_w8pc_full_compare_20261005; 32 WikiText-2 test windows; next-token NLL / perplexity
  bert  picks from runs/bert_w8pc_compare_20261006; the 872 SST-2 validation sentences; accuracy / cross-entropy
References: plain onnx-mlir FP32 (no pass) and every weight-bearing node at w=int8, a=int8.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_gpt2_fisher_milp import token_nll  # noqa: E402
from run_w8_per_channel_check import per_channel_int  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("gpt2", "bert"), required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    work = args.out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    if args.task == "gpt2":
        src = HERE / "runs/gpt2_shared_ablation_20260930"
        model_path = src / "model.onnx"
        graph = onnx.load(str(model_path))
        mapping = json.loads((src / "node_mapping.json").read_text())
        ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
        x = np.load(src / "test_tokens.npy")
        nodes = [n for n in erm.supported_nodes(graph) if n["name"] in mapping]
        saved = json.loads((HERE / "runs/gpt2_w8pc_full_compare_20261005/results.json").read_text())["rows"]
        crit_key = "criterion"
    else:
        prep = HERE / "runs/bert_sst2_20261006"
        model_path = prep / "model.onnx"
        graph = onnx.load(str(model_path))
        x, labels = np.load(prep / "val_tokens.npy"), np.load(prep / "val_labels.npy")
        nodes = erm.supported_nodes(graph)
        ranges = erm.calibrate_ranges(graph, nodes, np.load(prep / "calib_tokens.npy"))
        saved = json.loads((HERE / "runs/bert_w8pc_compare_20261006/results.json").read_text())["rows"]
        crit_key = "criterion"
    weighted = [n for n in nodes if n.get("weight_const")]
    names = [n["name"] for n in weighted]
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    axis = {}
    for n in weighted:          # the pass's per-channel axis: Conv/Gemm(transB=1) 0, Gemm(transB=0)/MatMul last
        node = by_name[n["name"]]
        tb = next((onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "transB"), 0)
        axis[inits[node.input[1]].tobytes()] = 0 if (node.op_type == "Gemm" and tb) else 1
    original = erm.np_fake_quant_weight

    def pc_weight(w, fmt, oracle):
        if not fmt.startswith("int"):
            return original(w, fmt, oracle)
        w = np.asarray(w, np.float32)
        bits = int(fmt[3:])
        return per_channel_int(w, bits) if axis[w.tobytes()] == 0 else per_channel_int(np.ascontiguousarray(w.T), bits).T

    erm.np_fake_quant_weight = pc_weight
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    inputs_path = work / "inputs.npy"
    np.save(inputs_path, x)
    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    if not base.exists():
        erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                 "--mlir-elide-resource-strings-if-larger=1000000000",
                 "--mlir-elide-elementsattrs-if-larger=1000000000", model_path], timeout=3600)
    log(f"{args.task}: {len(names)} weight-bearing nodes, {len(x)} evaluation inputs, imported")

    def full(cfg):
        return {n["name"]: cfg.get(n["name"], {"w": None, "a": "fp32"}) for n in nodes}

    configs = [("fp32", None), ("w8a8", full({n: {"w": "int8", "a": "int8"} for n in names}))]
    budgets = sorted({r["budget"] for r in saved})
    for r in saved:
        if r[crit_key] in ("BC", "SQNR"):
            configs.append((f"{r[crit_key]}_f{r['budget']:.2f}", full(r["formats"])))

    def metrics(z):
        if args.task == "gpt2":
            nll = token_nll(z, x).reshape(len(x), x.shape[1] - 1).mean(1)
            return {"per_item": nll, "nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())),
                    "top1": float((z[:, :-1, :].argmax(-1) == x[:, 1:]).mean())}
        finite = np.isfinite(z).all(1)
        ce = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(labels),
                             reduction="none").numpy()
        return {"per_item": ce, "ce": float(ce.mean()), "accuracy": float(((z.argmax(1) == labels) & finite).mean())}

    def compile_and_run(name, cfg):
        env = None if cfg is None else erm.node_formats_env(cfg, nodes, ranges)
        tc = time.time()
        so, info = erm.build_so(tools, work, name, base, env, None)
        compile_s = time.time() - tc
        z, ev = erm.run_so(tools, so, inputs_path, work / f"{name}.logits.npy", 36000)
        for p in work.glob(f"{name}.*"):
            if p.suffix in (".mlir", ".ll", ".so", ".npy") and p != base:
                p.unlink()
        m = metrics(z)
        sim = erm.simulate_in_onnxruntime(graph, nodes, cfg or {}, ranges, x, None)
        ms = metrics(sim)
        d = np.abs(z - sim)
        res = {k: v for k, v in m.items() if k != "per_item"}
        res.update(compile_seconds=compile_s, run_seconds_per_input=ev["seconds"] / ev["n"],
                   warnings=len(info["warnings"]),
                   vs_simulation={"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                                  **{f"simulated_{k}": v for k, v in ms.items() if k != "per_item"}})
        log(f"{args.task} {name:<12} " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()
                                                    if k in ("nll", "ppl", "top1", "ce", "accuracy"))
            + f"  | sim " + "  ".join(f"{k} {v:.4f}" for k, v in ms.items() if k in ("ppl", "accuracy", "ce"))
            + f" | max |dz| {d.max():.2e} | compile {compile_s:.0f}s, run {ev['seconds'] / ev['n']:.2f} s/input")
        return res, m["per_item"]

    results, per_item = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {name: pool.submit(compile_and_run, name, cfg) for name, cfg in configs}
        for name, fut in futures.items():
            results[name], per_item[name] = fut.result()
    key = "nll" if args.task == "gpt2" else "ce"
    for b in budgets:
        d = per_item[f"BC_f{b:.2f}"] - per_item[f"SQNR_f{b:.2f}"]
        results[f"BC_f{b:.2f}"]["vs_SQNR"] = {f"d{key}": float(d.mean()), f"d{key}_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    main_metric = "ppl" if args.task == "gpt2" else "accuracy"
    print(f"\n=== {args.task} (onnx-mlir compiled, per-channel weights): FP32 {main_metric} "
          f"{results['fp32'][main_metric]:.4f}, all W8A8 {results['w8a8'][main_metric]:.4f}")
    for b in budgets:
        bc, sq = results[f"BC_f{b:.2f}"], results[f"SQNR_f{b:.2f}"]
        print(f"   f={b:.2f}  B+C {main_metric} {bc[main_metric]:.4f} {key} {bc[key]:.4f}   SQNR {main_metric} "
              f"{sq[main_metric]:.4f} {key} {sq[key]:.4f}   B+C - SQNR d{key} {bc['vs_SQNR'][f'd{key}']:+.4f}"
              f"±{bc['vs_SQNR'][f'd{key}_se']:.4f}")


if __name__ == "__main__":
    main()
