#!/usr/bin/env python3
"""The B + C criterion end to end in onnx-mlir on MobileNetV2, ResNet-18, EfficientNet-B0, GPT-2 and BERT, its picks
compiled and evaluated, with the time of every stage.

Statistics, all from onnx-mlir builds of the FP32 model on the calibration inputs (no onnxruntime / PyTorch):
  ranges   --mixedp-collect-stats ranges   per-tensor min/max (M1), also the int activation params of the picks
  weights  --mixedp-collect-stats weights  per-output-channel ||Q(W) - W||^2 per format, compile time (M2)
  grads    --mixedp-collect-stats grads    per-channel sum g^2, sum g per tap; sum dW^2 per output channel (M3)
  errors   --mixedp-collect-stats errors   per-channel sum d, sum d^2 per tap and activation format (M4)
grads and errors are restricted (MIXEDP_STATS_NODES) to the nodes the selection assigns formats to. The builds run
concurrently, store constants to file, and every statistics run is its own process (see native_tables). The loss is the
cross-entropy of the model's output: image / sentence label, or (GPT-2) the next token at every position but the last.
Cost per such node and (w, a), as run_w8pc_full_compare.py's BC:
  B(w) = sum_o mean_o(g_W^2) ||Q_w(W) - W||_o^2          (g_W^2 summed over the calibration inputs)
  C(a) = sum_{x, y} sum_c [ g2_c / n (s2_c - n mu_c^2) + gc_c mu_c^2 ],  mu_c = s1_c / n
MILP picks (mp_milp) at budgets relative to all-int8 weight bytes and BOPs, on the 8-format grid of the earlier runs
and on the 11-format grid with p4e1/p6e1/p8e1, compiled with per-channel int weights and evaluated on the held-out
inputs (CNNs: top-1 / cross-entropy on 1000 images; GPT-2: perplexity on 32 test windows; BERT: SST-2 accuracy /
cross-entropy on 872 sentences).

Offline references (not part of the pipeline): the earlier PyTorch BC table rebuilt here (table agreement; GPT-2's
used the mean token loss, so it is scaled by (tokens - 1)^2 to the summed loss used here), the earlier BC picks and
the compiled results of the earlier BC and SQNR picks.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_native_calib_check import SLOT_INPUT  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

GRID8 = ("int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32")
GRID11 = ("int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16", "fp32")
QUANT = GRID11[:-1]
CNNS = {"mobilenetv2": "mobilenetv2/timm_ra", "resnet18": "resnet18/torchvision_v1",
        "efficientnet_b0": "efficientnet_b0/timm_ra"}
# GPT-2 / BERT imports and the big statistics builds outlast eval_random_mixed_precision.run's 900 s default
erm.run.__defaults__ = (None, 7200, None)  # env, timeout, cwd


@dataclass
class Spec:
    tag: str
    model_path: Path
    budgets: list
    calib: np.ndarray                 # calibration inputs (one model input, batch dim added per item)
    onehot: Callable                  # index -> one-hot target shaped like the model output
    eval_inputs: np.ndarray
    metrics: Callable                 # logits -> {"per_item": lower-is-better loss, metric: value}
    report: tuple                     # metric names to print, the first one higher-is-better for CNN/BERT
    names: list                       # nodes the MILP assigns formats to
    nodes: list                       # supported nodes passed to milp_select / node_formats_env
    weight_elems: dict
    flops: dict
    torch_ref: Callable               # () -> (B, C) on this pipeline's scale
    old_picks: dict                   # budget -> formats of the earlier BC run
    old_compiled: dict                # "BC_f0.60" / "SQNR_f0.60" -> metrics
    workers: int


def cross_entropy(z, y):
    z = z.astype(np.float64)
    m = z.max(-1, keepdims=True)
    return (np.log(np.exp(z - m).sum(-1)) + m[..., 0]) - np.take_along_axis(z, y[..., None], -1)[..., 0]


# --------------------------------------------------------------------------- #
# Model specs
# --------------------------------------------------------------------------- #
def cnn_spec(tag, images_dir, num_eval, work, with_references=True):
    model_dir = HERE / "../imagenet100/model_dataset" / CNNS[tag]
    graph = onnx.load(str(model_dir / "model.onnx"))
    meta = json.loads((model_dir / "metadata.json").read_text())
    man = json.loads((images_dir / "manifest.json").read_text())
    calib = [e for e in man["images"] if e["calibration"]]
    held = [e for e in man["images"] if not e["calibration"]][:num_eval]
    cx = erm.preprocess_by_source(images_dir, {"images": calib}, meta).astype(np.float32)
    cy = np.array([e["label"] for e in calib])
    ey = np.array([e["label"] for e in held])
    nodes = erm.supported_nodes(graph)
    names = [n["name"] for n in nodes if n.get("weight_const")]
    elems, flops = compute_weight_elems_and_flops_proxy(graph, nodes)

    def metrics(z):
        finite = np.isfinite(z).all(1)
        loss = cross_entropy(np.where(finite[:, None], z, 0.0), ey)
        return {"per_item": loss, "top1": float(((z.argmax(1) == ey) & finite).mean()), "loss": float(loss.mean())}

    old = json.loads((HERE / "runs/compiled_compare_20261006/results.json").read_text())[tag] if with_references else {}
    picks = {r["budget"]: r["formats"] for r in json.loads(
        (HERE / "runs/w8pc_full_compare_20261005" / tag / "results.json").read_text())["rows"] if r["criterion"] == "BC"} if with_references else {}
    return Spec(tag, model_dir / "model.onnx", [0.6, 0.7, 0.8, 1.0, 1.2], cx,
                lambda i: np.eye(1000, dtype=np.float32)[[cy[i]]],
                erm.preprocess_by_source(images_dir, {"images": held}, meta).astype(np.float32), metrics,
                ("top1", "loss"), names, nodes, elems, flops,
                lambda: cnn_torch_tables(model_dir, graph, cx, cy), picks, old, 10)


def gpt2_spec(work, with_references=True):
    from run_gpt2_fisher_milp import token_nll
    src = HERE / "runs/gpt2_shared_ablation_20260930"
    graph = onnx.load(str(src / "model.onnx"))
    mapping = json.loads((src / "node_mapping.json").read_text())
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in erm.supported_nodes(graph) if n["name"] in mapping]
    names = [n["name"] for n in nodes]
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    seq, vocab = calib.shape[1], 50257
    elems = {n: int(np.prod(inits[by_name[n].input[1]].dims)) for n in names}
    flops = {n: elems[n] * seq for n in names}

    def onehot(i):                    # next-token targets; the last position has none
        t = np.zeros((1, seq, vocab), np.float32)
        t[0, np.arange(seq - 1), calib[i, 1:]] = 1.0
        return t

    def metrics(z):
        nll = token_nll(z, test).reshape(len(test), seq - 1).mean(1)
        return {"per_item": nll, "ppl": float(np.exp(nll.mean())), "nll": float(nll.mean()),
                "top1": float((z[:, :-1, :].argmax(-1) == test[:, 1:]).mean())}

    old = json.loads((HERE / "runs/compiled_compare_gpt2_20261006/results.json").read_text()) if with_references else {}
    picks = {r["budget"]: r["formats"] for r in json.loads(
        (HERE / "runs/gpt2_w8pc_full_compare_20261005/results.json").read_text())["rows"] if r["criterion"] == "BC"} if with_references else {}
    return Spec("gpt2", src / "model.onnx", [0.8, 1.0, 1.25, 1.5], calib, onehot, test, metrics, ("ppl", "nll"),
                names, nodes, elems, flops, lambda: gpt2_torch_tables(src, graph, nodes, mapping, calib),
                picks, old, 6)


def bert_spec(work, with_references=True):
    prep = HERE / "runs/bert_sst2_20261006"
    graph = onnx.load(str(prep / "model.onnx"))
    cx, cy = np.load(prep / "calib_tokens.npy"), np.load(prep / "calib_labels.npy")
    vx, vy = np.load(prep / "val_tokens.npy"), np.load(prep / "val_labels.npy")
    nodes = erm.supported_nodes(graph)
    names = [n["name"] for n in nodes if n.get("weight_const")]
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    elems = {n: int(np.prod(inits[by_name[n].input[1]].dims)) for n in names}
    flops = {n: elems[n] * (cx.shape[1] if by_name[n].op_type == "MatMul" else 1) for n in names}

    def metrics(z):
        finite = np.isfinite(z).all(1)
        ce = cross_entropy(np.where(finite[:, None], z, 0.0), vy)
        return {"per_item": ce, "accuracy": float(((z.argmax(1) == vy) & finite).mean()), "ce": float(ce.mean())}

    old = json.loads((HERE / "runs/compiled_compare_bert_20261006/results.json").read_text()) if with_references else {}
    picks = {r["budget"]: r["formats"] for r in json.loads(
        (HERE / "runs/bert_w8pc_compare_20261006/results.json").read_text())["rows"] if r["criterion"] == "BC"} if with_references else {}
    return Spec("bert", prep / "model.onnx", [0.6, 0.7, 0.8, 1.0, 1.2], cx,
                lambda i: np.eye(2, dtype=np.float32)[[cy[i]]], vx, metrics, ("accuracy", "ce"), names, nodes,
                elems, flops, lambda: bert_torch_tables(graph, nodes, names, cx, cy), picks, old, 10)


# --------------------------------------------------------------------------- #
# Native statistics and table
# --------------------------------------------------------------------------- #
# (Krnl lowering -O, clang -O) per statistics build, by model family; see native_tables
STATS_LEVELS = {"conv": {"ranges": ("O1", "O1"), "grads": ("O1", "O1"), "errors": ("O3", "O0")},
                "matmul": {"ranges": ("O3", "O0"), "grads": ("O3", "O0"), "errors": ("O3", "O0")}}
ERRORS_SHARD_NODES = 8


class MemoryBudget:
    """Admits concurrent builds / runs while the sum of their estimated peak memory fits in `gb` GiB; a job larger
    than the whole budget waits until it runs alone."""

    def __init__(self, gb):
        self.total = self.free = gb
        self.cv = threading.Condition()

    @contextmanager
    def hold(self, gb):
        gb = min(gb, self.total)
        with self.cv:
            self.cv.wait_for(lambda: self.free >= gb)
            self.free -= gb
        try:
            yield
        finally:
            with self.cv:
                self.free += gb
                self.cv.notify_all()


def mem_available_gb():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1048576


def build_gb(base):
    """Peak memory of an onnx-mlir-opt step on the imported model: ~2x its text (GPT-2: 1.3 GB -> 2.6 GB)."""
    return 2.1 * base.stat().st_size / 2 ** 30 + 0.3


def stats_build(tools, work, base, name, env, levels, budget):
    """Instrument the imported model for one statistics mode and compile it, constants to file, at levels = (Krnl
    -O, clang -O), in its own folder work/stats_<name> (OM_CONSTANT_PATH: one such .so per process; for the same
    reason the import's symbol postfix stays). Returns ((so, postfix), layout, step seconds)."""
    d = work / f"stats_{name}"
    d.mkdir(exist_ok=True)
    layout_path, inst = d / f"{name}_layout.json", d / f"{name}.onnx.mlir"
    errors = env["MIXEDP_STATS_MODE"] == "errors"
    with budget.hold(build_gb(base)):
        _, t_pass = erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "-o", inst],
                            env={**env, "MIXEDP_STATS_LAYOUT": str(layout_path)})
        so, info = erm.build_so(tools, d, name, inst, None, None, uses_posit=errors, uses_f16=errors,
                                opt=levels[1], krnl_opt=levels[0], constants_to_file=True)
    for p in d.glob(f"{name}.*"):              # keep the .so and its constants file
        if p.suffix in (".mlir", ".ll", ".bc"):
            p.unlink()
    layout = json.loads(layout_path.read_text())
    if layout.get("skipped_nested"):
        sys.exit(f"supported nodes inside If/Loop/Scan bodies get no statistics: {layout['skipped_nested']}")
    return (so, erm.SYMBOL_POSTFIX), layout, {"stats_pass": t_pass, **info["steps"]}


def stats_runs(pool, tools, so_postfix, layout, inputs_path, n, budget, onehot_dir=None):
    """The statistics result of the .so on each of the n calibration inputs, [n, num_values] float64: one process
    per input, as many at a time as `pool` has workers and `budget` admits (a run peaks at ~2x the constants file,
    which it maps, plus 0.3 GB)."""
    so, postfix = so_postfix
    need = 2 * sum(p.stat().st_size for p in so.parent.glob("*.constants.bin")) / 2 ** 30 + 0.3

    def one(i):
        out = so.parent / f"out_{i}.npy"
        with budget.hold(need):
            r, _ = erm.run([sys.executable, __file__, "--_stats-child", so, postfix, tools.lib, inputs_path, i,
                            onehot_dir / f"{i}.npy" if onehot_dir else "-", layout["result_index"], out])
        v = np.load(out)
        out.unlink()
        return v, json.loads(r.stdout.strip().splitlines()[-1])

    res = list(pool.map(one, range(n)))
    return np.stack([v for v, _ in res]), {"run_s_per_input": float(np.mean([m["run_s"] for _, m in res])),
                                           "peak_rss_gb": float(max(m["peak_rss_gb"] for _, m in res))}


def child_stats(so, postfix, lib, inputs_path, index, onehot_path, result_index, out_path):
    sys.path.insert(0, lib)
    from PyRuntime import OMExecutionSession
    x = np.load(inputs_path, mmap_mode="r")[int(index)]
    ins = [np.ascontiguousarray(x[None])] + ([np.load(onehot_path)] if onehot_path != "-" else [])
    sess = OMExecutionSession(so, postfix)
    t0 = time.time()
    v = np.asarray(sess.run(ins)[int(result_index)], np.float64)
    t_run = time.time() - t0
    np.save(out_path, v)
    peak_kb = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(json.dumps({"run_s": t_run, "peak_rss_gb": peak_kb / 1048576}))


def native_tables(spec, tools, work, log, workers=os.cpu_count()):
    """B[node][w], C[node][a] for spec.names, the int ranges by tensor, and the time of every stage.

    After the import, three independent chains run concurrently:
      weights                                          (one onnx-mlir-opt pass)
      grads build  -> grads runs
      ranges build -> ranges runs -> errors builds (need the int ranges) -> errors runs
    errors is split into shards of ERRORS_SHARD_NODES nodes, built and run concurrently: its lowering to LLVM is
    superlinear in the graph (EfficientNet-B0: 82 nodes 72 s, 20 nodes 7 s), while each shard only adds one more
    forward pass per input. Every statistics run is its own process (one .so, one calibration input), at most
    `workers` at a time, and builds and runs are admitted while their estimated peak memory fits in 80% of the
    available memory. The builds store constants to file (no weights in the .ll); their -O levels (STATS_LEVELS)
    depend on the model: Krnl -O3 tiles and vectorizes MatMul (GPT-2 forward 24 s -> 4 s per input) but makes the
    lowering to LLVM slower, clang -O0 links in seconds (-O1: 30 s more on GPT-2, 20+ min on a whole errors graph) but
    runs convolutions slowly (ResNet-18 grads 42 s vs 7 s per input). Models with Conv nodes build grads and ranges at
    (-O1, -O1), MatMul models at (-O3, -O0); the errors shards are (-O3, -O0) for both."""
    timing, stages = {}, {}
    t_all = time.time()
    budget = MemoryBudget(0.8 * mem_available_gb())

    def stage(name, fn, *a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        stages[name] = [t0 - t_all, time.time() - t_all]
        timing[f"{name}_s"] = time.time() - t0
        return r

    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    stage("import", erm.run, [tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                              "--mlir-elide-resource-strings-if-larger=1000000000",
                              "--mlir-elide-elementsattrs-if-larger=1000000000", spec.model_path])
    graph = onnx.load(str(spec.model_path), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    n_cal = len(spec.calib)
    inputs_path = work / "calib_inputs.npy"
    np.save(inputs_path, spec.calib)
    onehot_dir = work / "stats_grads" / "onehot"
    onehot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cal):
        np.save(onehot_dir / f"{i}.npy", spec.onehot(i))
    levels = STATS_LEVELS["conv" if any(n.op_type == "Conv" for n in graph.graph.node) else "matmul"]
    k = -(-len(spec.names) // ERRORS_SHARD_NODES)
    shards = [spec.names[j::k] for j in range(k)]        # interleaved: early and late layers in every shard
    run_info = {}

    def weights():
        wjson = work / "weight_err.json"
        with budget.hold(build_gb(base)):
            erm.run([tools.opt, base, "--mixedp-collect-stats", "-o", work / "unchanged.onnx.mlir"],
                    env={"MIXEDP_STATS_MODE": "weights", "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                         "MIXEDP_STATS_FORMATS": ",".join(QUANT), "MIXEDP_STATS_LAYOUT": str(wjson)})
        (work / "unchanged.onnx.mlir").unlink()
        return {e["node"]: e for e in json.loads(wjson.read_text())["nodes"]}

    def grads_chain():
        so, lay, timing["grads_build_steps"] = stage(
            "grads_build", stats_build, tools, work, base, "grads",
            {"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": ",".join(spec.names)}, levels["grads"], budget)
        v, run_info["grads"] = stage("grads_runs", stats_runs, runs, tools, so, lay, inputs_path, n_cal, budget,
                                     onehot_dir)
        return lay, v

    def errors_shard(j):
        t0 = time.time()
        so, lay, steps = stats_build(tools, work, base, f"errors{j}",
                                     {"MIXEDP_STATS_MODE": "errors", "MIXEDP_STATS_FORMATS": ",".join(QUANT),
                                      "MIXEDP_STATS_RANGES": str(work / "ranges_by_point.json"),
                                      "MIXEDP_STATS_NODES": ",".join(shards[j])}, levels["errors"], budget)
        t1 = time.time()
        v, info = stats_runs(runs, tools, so, lay, inputs_path, n_cal, budget)
        return lay, v, {"build_s": t1 - t0, "runs_s": time.time() - t1, "steps": steps, **info}

    def ranges_errors_chain():
        so, lay_r, timing["ranges_build_steps"] = stage("ranges_build", stats_build, tools, work, base, "ranges",
                                                        {"MIXEDP_STATS_MODE": "ranges"}, levels["ranges"], budget)
        v, run_info["ranges"] = stage("ranges_runs", stats_runs, runs, tools, so, lay_r, inputs_path, n_cal, budget)
        taps = v.reshape(n_cal, lay_r["num_taps"], 2)
        lo, hi = taps[:, :, 0].min(0), taps[:, :, 1].max(0)
        ranges, rng = {}, {}
        for p in lay_r["points"]:
            n = by_name[p["node"]]
            t = n.output[0] if p["slot"] == "y" else n.input[SLOT_INPUT[p["slot"]]]
            r = (float(lo[p["tap"]]), float(hi[p["tap"]]))
            assert ranges.setdefault(t, r) == r, (t, ranges[t], r)   # one ONNX tensor -> one tap
            rng.setdefault(p["node"], {})[p["slot"]] = list(r)
        (work / "ranges_by_point.json").write_text(json.dumps(rng) + "\n")
        with ThreadPoolExecutor(max_workers=k) as pool:
            res = stage("errors", lambda: list(pool.map(errors_shard, range(k))))
        slowest = max(range(k), key=lambda j: res[j][2]["build_s"])
        timing.update({"errors_shards": k, "errors_build_s": res[slowest][2]["build_s"],
                       "errors_build_steps": res[slowest][2]["steps"],
                       "errors_runs_s": max(m["runs_s"] for _, _, m in res)})
        run_info["errors"] = {"run_s_per_input": float(np.mean([m["run_s_per_input"] for _, _, m in res])),
                              "peak_rss_gb": max(m["peak_rss_gb"] for _, _, m in res)}
        return ranges, [(lay, v) for lay, v, _ in res]

    with ThreadPoolExecutor(max_workers=workers) as runs, ThreadPoolExecutor(max_workers=3) as chains:
        fw = chains.submit(stage, "weights", weights)
        fg, fe = chains.submit(grads_chain), chains.submit(ranges_errors_chain)
        werr, (lay_g, vg), (ranges, errors) = fw.result(), fg.result(), fe.result()
    for p in onehot_dir.glob("*.npy"):
        p.unlink()
    for kind, info in run_info.items():
        timing[f"{kind}_run_s_per_input"] = info["run_s_per_input"]
        timing[f"{kind}_peak_rss_gb"] = info["peak_rss_gb"]
    log(f"  stats: grads build {timing['grads_build_s']:.0f}s + runs {timing['grads_runs_s']:.0f}s | ranges build "
        f"{timing['ranges_build_s']:.0f}s + runs {timing['ranges_runs_s']:.0f}s, errors {k} shards "
        f"{timing['errors_s']:.0f}s (slowest build {timing['errors_build_s']:.0f}s, runs {timing['errors_runs_s']:.0f}s) "
        f"| weights {timing['weights_s']:.0f}s; {len(lay_g['taps'])} taps, {workers} run workers, memory budget "
        f"{budget.total:.0f} GB")

    # per quantization point (node, slot): sums over the calibration inputs
    t0 = time.time()
    g2, gc, pos, s1, s2 = {}, {}, {}, {}, {}
    taps = {t["tap"]: t for t in lay_g["taps"]}
    for p in lay_g["points"]:
        t, key = taps[p["tap"]], (p["node"], p["slot"])
        o, c = t["offset"], t["channels"]
        g2[key] = vg[:, o:o + c].sum(0)
        gc[key] = (vg[:, o + c:o + 2 * c] ** 2).sum(0)      # per-input channel sum of g, squared, then summed
        pos[key] = t["positions"]
    for lay_e, ve in errors:
        taps = {t["tap"]: t for t in lay_e["taps"]}
        for p in lay_e["points"]:
            t, key = taps[p["tap"]], (p["node"], p["slot"])
            c = t["channels"]
            assert c == len(g2[key]) and t["positions"] == pos[key], ("grads / errors tap mismatch", key)
            s1[key] = {f: ve[:, t["errors"][f]:t["errors"][f] + c].sum(0) for f in QUANT}
            s2[key] = {f: ve[:, t["errors"][f] + c:t["errors"][f] + 2 * c].sum(0) for f in QUANT}
    assert set(s1) == set(g2), "grads and errors builds disagree on the quantization points"
    g2w = {w["node"]: vg[:, w["offset"]:w["offset"] + w["channels"]].sum(0) for w in lay_g["weights"]}
    epc = {w["node"]: w["elements_per_channel"] for w in lay_g["weights"]}
    B, C = {}, {}
    for n in spec.names:
        G = g2w[n] / epc[n]
        err2 = {f: np.array([np.inf if v is None else v for v in werr[n]["err2"][f]]) for f in QUANT}  # null: overflow
        B[n] = {"fp32": 0.0, **{f: float((G * err2[f]).sum()) for f in QUANT}}
        C[n] = {"fp32": 0.0}
        for f in QUANT:
            cost = 0.0
            for key in ((n, "x"), (n, "y")):
                cnt = n_cal * pos[key]
                mu = s1[key][f] / cnt
                cost += float((g2[key] / cnt * (s2[key][f] - cnt * mu ** 2)).sum() + (gc[key] * mu ** 2).sum())
            C[n][f] = cost
    timing["table_s"] = time.time() - t0
    timing["criterion_total_s"] = time.time() - t_all
    timing["stages"] = stages
    return B, C, ranges, timing


# --------------------------------------------------------------------------- #
# Offline references: the earlier PyTorch BC tables
# --------------------------------------------------------------------------- #
def _bc_from_stats(names, st, g2w_rows, d2_rows, AF):
    """B and C from per-node PyTorch statistics: g2w_rows[n] [out, ...] per-channel sums of g_W^2 rows,
    d2_rows[n][f] the matching per-channel weight errors."""
    B, C = {}, {}
    for n in names:
        G = g2w_rows[n]
        B[n], C[n] = {"fp32": 0.0}, {"fp32": 0.0}
        for f in AF:
            B[n][f] = float((G.mean(1) * d2_rows[n][f].sum(1)).sum())
            c = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                mu = a["s1"][f] / a["n"]
                c += float((a["g2"] / a["n"] * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            C[n][f] = c
    return B, C


def _hooked_stats(names, module_of, run_loss, samples, y_of, ranges, tens, csum, AF):
    import torch
    from run_mbv2_fisher_term_ablation import act_fake_quant
    cap = {}
    hooks = [module_of[n].register_forward_hook(lambda mm, inp, o, n=n: cap.__setitem__(n, (inp[0], o)))
             for n in names]
    g2w = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for k in range(len(samples)):
        loss = run_loss(k)
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(loss, targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += grads[3 * i].double() ** 2
                x, out = cap[n]
                for j, (s, t) in enumerate((("x", x), ("y", y_of(n, out)))):
                    g, a = grads[3 * i + 1 + j].double(), st[n][s]
                    a["g2"] = a["g2"] + csum(g ** 2)
                    a["gc"] = a["gc"] + csum(g) ** 2
                    a["n"] += t.numel() // t.shape[csum.axis]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + csum(d)
                        a["s2"][f] = a["s2"][f] + csum(d ** 2)
        cap.clear()
    for h in hooks:
        h.remove()
    return g2w, st


def cnn_torch_tables(model_dir, graph, cx, cy):
    import torch
    import torch.nn.functional as F
    from run_criteria_cost_compare import match_modules, torch_model
    from run_mbv2_act_lowbit_check import chan_sum
    from run_w8_per_channel_check import per_channel_int
    meta = json.loads((model_dir / "metadata.json").read_text())
    all_nodes = erm.supported_nodes(graph)
    ranges = erm.calibrate_ranges(graph, all_nodes, cx)
    weighted = [n for n in all_nodes if n.get("weight_const")]
    names = [n["name"] for n in weighted]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in weighted}
    model, layers = torch_model(meta)
    module_of, inits, by_name = match_modules(graph, weighted, layers)
    chan_sum.axis = 1
    g2w, st = _hooked_stats(names, module_of, lambda k: F.cross_entropy(
        model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True)), torch.tensor([int(cy[k])])),
        cx, lambda n, o: o, ranges, tens, chan_sum, GRID8[:-1])
    rows, d2 = {}, {}
    for n in names:
        W = inits[by_name[n].input[1]]
        rows[n] = g2w[n].numpy().reshape(W.shape[0], -1)
        d2[n] = {f: ((per_channel_int(W, int(f[3:])) if f.startswith("int") else erm.np_fake_quant_weight(W, f, None))
                     .astype(np.float64) - W).reshape(W.shape[0], -1) ** 2 for f in GRID8[:-1]}
    return _bc_from_stats(names, st, rows, d2, GRID8[:-1])


def gpt2_torch_tables(src, graph, nodes, mapping, calib):
    import torch
    from transformers import GPT2LMHeadModel
    from run_w8_per_channel_check import per_channel_int
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    names = [n["name"] for n in nodes]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    module_of = {n: mods[mapping[n].removesuffix(".weight")] for n in names}

    def csum(t):
        return t.sum(dim=tuple(range(t.dim() - 1)))
    csum.axis = -1

    def run_loss(k):
        ids = torch.from_numpy(calib[k][None])
        return model(ids, labels=ids, use_cache=False).loss * (calib.shape[1] - 1)   # summed token loss

    g2w, st = _hooked_stats(names, module_of, run_loss, calib, lambda n, o: o, ranges, tens, csum, GRID8[:-1])
    rows, d2 = {}, {}
    for n in names:
        W = params[mapping[n]].detach().numpy()                     # [in, out]: output channels on axis 1
        rows[n] = g2w[n].numpy().T
        d2[n] = {f: (((per_channel_int(np.ascontiguousarray(W.T), int(f[3:])).T if f.startswith("int")
                       else erm.np_fake_quant_weight(W, f, None)).astype(np.float64) - W) ** 2).T
                 for f in GRID8[:-1]}
    return _bc_from_stats(names, st, rows, d2, GRID8[:-1])


def bert_torch_tables(graph, nodes, names, cx, cy):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import BertForSequenceClassification, BertTokenizer
    from run_bert_w8pc_compare import InputIdsOnly, torch_weight_q
    from run_criteria_cost_compare import match_modules
    assets = HERE / "bert_assets" / "model"
    tok = BertTokenizer.from_pretrained(assets)
    bert = BertForSequenceClassification.from_pretrained(assets, attn_implementation="eager").eval()
    model = InputIdsOnly(bert, tok.pad_token_id).eval()
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    weighted = [n for n in nodes if n.get("weight_const")]
    op_of = {n["name"]: n["op"] for n in weighted}
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in weighted}
    ranges = erm.calibrate_ranges(graph, nodes, cx)
    module_of, _, _ = match_modules(graph, weighted, [m for m in model.modules() if isinstance(m, nn.Linear)])

    def csum(t):
        return t.sum(dim=tuple(range(t.dim() - 1)))
    csum.axis = -1
    g2w, st = _hooked_stats(names, module_of, lambda k: F.cross_entropy(
        model(torch.from_numpy(cx[k:k + 1])), torch.tensor([int(cy[k])])), cx,
        lambda n, o: o - module_of[n].bias if op_of[n] == "MatMul" else o, ranges, tens, csum, GRID8[:-1])
    rows, d2 = {}, {}
    for n in names:
        W = module_of[n].weight                                      # [out, in]
        rows[n] = g2w[n].numpy()
        d2[n] = {f: (torch_weight_q(W, f).double().numpy() - W.detach().double().numpy()) ** 2 for f in GRID8[:-1]}
    return _bc_from_stats(names, st, rows, d2, GRID8[:-1])


# --------------------------------------------------------------------------- #
def milp_picks(spec, B, C):
    """MILP picks on grid8 and grid11 at every budget: {(grid, budget): formats}, one result row each."""
    ref_bytes = sum(spec.weight_elems[n] for n in spec.names)
    ref_bops = sum(64 * spec.flops[n] for n in spec.names)

    def cost(n, w, a):                         # a non-finite cost (e.g. f16 overflow) forbids the option
        v = B[n][w] + C[n][a]
        return v if np.isfinite(v) else None

    picks, rows = {}, []
    for grid_name, grid in (("grid8", GRID8), ("grid11", GRID11)):
        for f in spec.budgets:
            cfg, obj, status = milp_select(spec.nodes, grid, grid, cost, spec.weight_elems, spec.flops,
                                           f * ref_bytes, f * ref_bops, time_limit=60)
            assert status in (0, 1), (grid_name, f, status)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in spec.names}
            size, bops = config_cost(cfg, spec.weight_elems, spec.flops)
            rows.append({"grid": grid_name, "budget": f, "objective": obj, "size": size / ref_bytes,
                         "bops": bops / ref_bops, "formats": cfg,
                         "posit_weights": sum(c["w"].startswith("p") for c in cfg.values()),
                         "posit_activations": sum(c["a"].startswith("p") for c in cfg.values())})
            picks[(grid_name, f)] = cfg
    return picks, rows


def compare_criterion(spec, B, C, ranges_by_point, ref_dir):
    """The table, int ranges and MILP picks of this run against an earlier run's (ref_dir/table.json,
    ranges_by_point.json): relative table differences, range differences relative to the range width, and the
    number of nodes whose (w, a) pick is the same at every grid and budget."""
    ref = json.loads((ref_dir / "table.json").read_text())
    out = {}
    for name, nat, old in (("B", B, ref["B"]), ("C", C, ref["C"])):
        a = np.array([nat[n][f] for n in spec.names for f in QUANT])
        r = np.array([old[n][f] for n in spec.names for f in QUANT])
        fin = np.isfinite(a) & np.isfinite(r)
        rel = np.abs(a[fin] - r[fin]) / np.maximum(np.abs(r[fin]), 1e-30)
        out[name] = {"entries": int(len(a)), "nonfinite_mismatch": int((np.isfinite(a) != np.isfinite(r)).sum()),
                     "median_rel": float(np.median(rel)), "max_rel": float(np.max(rel))}
    old_rng = json.loads((ref_dir / "ranges_by_point.json").read_text())
    d = []
    for n, slots in ranges_by_point.items():
        for s, (lo, hi) in slots.items():
            olo, ohi = old_rng[n][s]
            d += [abs(lo - olo) / max(ohi - olo, 1e-30), abs(hi - ohi) / max(ohi - olo, 1e-30)]
    out["ranges_max_rel_width"] = float(max(d))
    new_picks, _ = milp_picks(spec, B, C)
    old_picks, _ = milp_picks(spec, ref["B"], ref["C"])
    out["same_picks"] = {f"{g}_f{f:.2f}": sum(new_picks[(g, f)][n] == old_picks[(g, f)][n] for n in spec.names)
                         for (g, f) in new_picks}
    out["nodes"] = len(spec.names)
    return out


def run_model(spec, tools, work, log):
    B, C, ranges, timing = native_tables(spec, tools, work, log)
    log(f"{spec.tag}: native B + C table in {timing['criterion_total_s']:.0f}s")
    (work / "table.json").write_text(json.dumps({"B": B, "C": C}) + "\n")

    t0 = time.time()
    tB, tC = spec.torch_ref()
    t_ref = time.time() - t0
    from scipy.stats import spearmanr
    agree = {}
    for name, nat, ref in (("B", B, tB), ("C", C, tC)):
        a = np.array([nat[n][f] for n in spec.names for f in GRID8[:-1]])
        r = np.array([ref[n][f] for n in spec.names for f in GRID8[:-1]])
        rel = np.abs(a - r) / np.maximum(np.abs(r), 1e-30)
        agree[name] = {"entries": int(len(a)), "median_rel": float(np.median(rel)),
                       "p99_rel": float(np.percentile(rel, 99)), "max_rel": float(np.max(rel)),
                       "spearman": float(spearmanr(a, r)[0])}
        log(f"  {name} vs PyTorch table: median rel {agree[name]['median_rel']:.1e}, p99 {agree[name]['p99_rel']:.1e}, "
            f"max {agree[name]['max_rel']:.1e}, Spearman {agree[name]['spearman']:.6f}")

    t0 = time.time()
    picks, rows = milp_picks(spec, B, C)
    for row in rows:
        if row["grid"] == "grid8" and row["budget"] in spec.old_picks:
            row["same_as_old_bc"] = sum(row["formats"][n] == spec.old_picks[row["budget"]][n] for n in spec.names)
    timing["milp_s"] = time.time() - t0
    for r in rows:
        if "same_as_old_bc" in r:
            log(f"  grid8 f={r['budget']:.2f}: identical to the old BC pick on {r['same_as_old_bc']}/{len(spec.names)}")

    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    posit_obj = erm.build_posit_runtime(tools, work)
    inputs_path = work / "eval_inputs.npy"
    np.save(inputs_path, spec.eval_inputs)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"

    def full(cfg):
        return {n["name"]: cfg.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes}

    def compile_and_run(name, cfg):
        env = erm.node_formats_env(full(cfg), spec.nodes, ranges)
        tc = time.time()
        so, info = erm.build_so(tools, work, name, base, env, posit_obj)
        compile_s = time.time() - tc
        z, ev = erm.run_so(tools, so, inputs_path, work / f"{name}.logits.npy", 36000)
        for p in work.glob(f"{name}.*"):
            if p.suffix in (".mlir", ".ll", ".bc", ".so") and p != base:
                p.unlink()
        m = spec.metrics(z)
        np.save(work / f"{name}.per_item.npy", m.pop("per_item"))
        (work / f"{name}.logits.npy").unlink()
        return m, compile_s, ev["seconds"] / ev["n"], len(info["warnings"])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=spec.workers) as pool:
        futures = {k: pool.submit(compile_and_run, f"{k[0]}_f{k[1]:.2f}", cfg) for k, cfg in picks.items()}
        results = {k: fut.result() for k, fut in futures.items()}
    timing["eval_wall_s"] = time.time() - t0
    os.environ.pop("MIXEDP_WEIGHT_PER_CHANNEL", None)
    for row in rows:
        m, compile_s, run_s, warn = results[(row["grid"], row["budget"])]
        row.update(m)
        row.update({"compile_s": compile_s, "run_s_per_item": run_s, "pass_warnings": warn})
        for crit in ("BC", "SQNR"):
            oc = spec.old_compiled.get(f"{crit}_f{row['budget']:.2f}")
            if oc:
                row[f"old_{crit}"] = {k: oc[k] for k in spec.report if k in oc}
    timing["eval_compile_s_mean"] = float(np.mean([r["compile_s"] for r in rows]))
    timing["eval_run_s_per_item_mean"] = float(np.mean([r["run_s_per_item"] for r in rows]))
    timing["torch_reference_s"] = t_ref
    paired = {}
    for f in spec.budgets:
        d = np.load(work / f"grid11_f{f:.2f}.per_item.npy") - np.load(work / f"grid8_f{f:.2f}.per_item.npy")
        paired[f] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    out = {"timing": timing, "table_vs_pytorch": agree, "grid11_minus_grid8": paired, "rows": rows}
    (work / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def stats_line(t):
    return (f"criterion {t['criterion_total_s']:.0f}s (import {t['import_s']:.0f}; grads build {t['grads_build_s']:.0f} "
            f"+ runs {t['grads_runs_s']:.0f} | ranges build {t['ranges_build_s']:.0f} + runs {t['ranges_runs_s']:.0f}, "
            f"errors x{t['errors_shards']} {t['errors_s']:.0f} | weights {t['weights_s']:.0f})")


def main():
    if sys.argv[1:2] == ["--_stats-child"]:
        return child_stats(*sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--models", default="mobilenetv2,resnet18,efficientnet_b0,gpt2,bert")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--criterion-only", action="store_true",
                    help="only the native table; compare it, its int ranges and its MILP picks with --compare-dir")
    ap.add_argument("--compare-dir", type=Path, default=HERE / "runs/native_full_20261009")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    summary = {}
    for tag in args.models.split(","):
        work = args.out_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        tm = time.time()
        refs = not args.criterion_only
        spec = (gpt2_spec(work, refs) if tag == "gpt2" else bert_spec(work, refs) if tag == "bert"
                else cnn_spec(tag, args.images_dir, args.num_eval, work, refs))
        if args.criterion_only:
            B, C, _, t = native_tables(spec, tools, work, log)
            (work / "table.json").write_text(json.dumps({"B": B, "C": C}) + "\n")
            ref_dir = args.compare_dir / tag
            cmp = compare_criterion(spec, B, C, json.loads((work / "ranges_by_point.json").read_text()), ref_dir)
            old_t = json.loads((ref_dir / "results.json").read_text())["timing"]["criterion_total_s"]
            summary[tag] = {"timing": t, "previous_criterion_total_s": old_t, "vs_previous": cmp}
            (work / "criterion.json").write_text(json.dumps(summary[tag], indent=1) + "\n")
            log(f"{tag}: {stats_line(t)}; previous {old_t:.0f}s")
            log(f"  vs previous table: B median/max rel {cmp['B']['median_rel']:.1e}/{cmp['B']['max_rel']:.1e}, "
                f"C {cmp['C']['median_rel']:.1e}/{cmp['C']['max_rel']:.1e}, ranges {cmp['ranges_max_rel_width']:.1e} "
                f"of width; same picks on {min(cmp['same_picks'].values())}/{cmp['nodes']} nodes at worst "
                f"({sum(v == cmp['nodes'] for v in cmp['same_picks'].values())}/{len(cmp['same_picks'])} grid x budget "
                f"identical)")
            continue
        out = run_model(spec, tools, work, log)
        out["timing"]["model_wall_s"] = time.time() - tm
        (work / "results.json").write_text(json.dumps(out, indent=1) + "\n")
        summary[tag] = {k: v for k, v in out.items() if k != "rows"}
        t = out["timing"]
        log(f"{tag}: {stats_line(t)}, MILP {t['milp_s']:.1f}s, eval {t['eval_wall_s']:.0f}s "
            f"(compile {t['eval_compile_s_mean']:.0f}s, run {t['eval_run_s_per_item_mean']:.3f}s/item); "
            f"model total {t['model_wall_s']:.0f}s")
        rep = spec.report
        print(f"\n=== {tag}: {rep[0]} / {rep[1]} (compiled); grid11 - grid8 paired {rep[1] if tag != 'gpt2' else 'nll'}")
        print("   budget    grid8 native      grid11 native (posit w/a)    old BC            old SQNR          grid11-grid8")
        for f in spec.budgets:
            r8 = next(r for r in out["rows"] if r["grid"] == "grid8" and r["budget"] == f)
            r11 = next(r for r in out["rows"] if r["grid"] == "grid11" and r["budget"] == f)

            def fmt(m):
                return f"{m[rep[0]]:.4g}/{m[rep[1]]:.4g}" if m else "-"
            pr = out["grid11_minus_grid8"][f]
            print(f"   f={f:.2f}   {fmt(r8):<17} {fmt(r11):<15} ({r11['posit_weights']}/{r11['posit_activations']})"
                  f"          {fmt(r8.get('old_BC')):<17} {fmt(r8.get('old_SQNR')):<17} {pr['mean']:+.4f}±{pr['se']:.4f}")
        print(flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
