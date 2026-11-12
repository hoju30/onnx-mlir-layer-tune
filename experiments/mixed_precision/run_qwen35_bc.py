#!/usr/bin/env python3
"""Qwen3.5-0.8B (dense hybrid: Gated DeltaNet + full attention), text backbone: the native onnx-mlir BC criterion and
automatic pipeline of Qwen3-0.6B carried over unchanged (C_i(w, a) = W_i(w) + A_i(a), total input gradient, MILP ->
2-D Pareto -> normalized chord knee). Stages (one --out-dir, resumable):

  prepare      plan.json (frozen before any candidate loss), tokens, the export adapter (qwen35_common.TextExport)
               exported to ONNX at the sequence length, constant folding, node mapping, op inventory, ORT check
  correctness  native forward / backward / table statistics (--parts forward,backward,stats) against PyTorch
  tables       native BC table (run_native_bc_pipeline.native_tables, input_grad="total")
  search       mp_auto.auto_select on the native table (FFN-only; FFN + attention projections)
  freeze       the ranking candidates and their BC scores, frozen with hashes
  sqnr         the SQNR reference of the candidates (calibration only), frozen before the truth
  truth        diagnostic-window loss changes of the candidates (checks the freezes; --device cuda: GPU replay)
  analyze      ranking metrics, joint vs single, device cross-check (analysis.json, *.csv)
  evaluate     selected configurations and FP32 compiled natively, final-test PPL / NLL (--versions)
  seq512       limited 512-token check (--parts export,forward,table|torch_table,stats,scores,truth,analyze)
The report (report.md) is written from these outputs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35  # noqa: F401  (transformers 5 first)
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEQ = 128
DATA = {"calibration": ("validation", 0, 10), "diagnostic": ("validation", 1280, 30), "final_test": ("test", 0, 64)}
DATA_512 = {"calibration_512": ("validation", 8192, 4), "diagnostic_512": ("validation", 10240, 8)}
FFN = ("gate_proj", "up_proj", "down_proj")
RANK_FORMATS = ("int4", "int6", "int8", "p6e1", "p8e1")


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n")


def jload(path):
    return json.loads(path.read_text())


def representative_layers(layer_types):
    """Early / middle / late of each block type: the first, the middle (index n // 2) and the last of its layers."""
    out = {}
    for t in ("linear_attention", "full_attention"):
        ls = [i for i, x in enumerate(layer_types) if x == t]
        out[t] = [ls[0], ls[len(ls) // 2], ls[-1]]
    return out


def stage_prepare(args, out, log):
    from qwen3_swiglu_common import sha256
    t0 = time.time()
    hf = q35.load_hf(torch.float32)
    cfg = hf.model.language_model.config
    reps = representative_layers(cfg.layer_types)
    plan = {
        "model": {"repo": q35.REPO, "revision": q35.REVISION, "text_backbone_only": True, "vision_tower": "not used",
                  "mtp_layer": "not used (not in the text forward)"},
        "architecture": {"layers": cfg.num_hidden_layers, "layer_types": cfg.layer_types, "hidden": cfg.hidden_size,
                         "intermediate": cfg.intermediate_size, "vocab": cfg.vocab_size, "tie_embeddings": True},
        "data": {"source": "WikiText-2 raw parquet (gpt2_assets/data), this model's tokenizer, no special tokens",
                 "seq_len": SEQ, **{k: {"split": s, "first_token": f, "windows": n, "tokens": [f, f + n * SEQ]}
                                    for k, (s, f, n) in DATA.items()},
                 "seq_512": {k: {"split": s, "first_token": f, "windows": n, "tokens": [f, f + n * 512]}
                             for k, (s, f, n) in DATA_512.items()},
                 "non_overlap": "calibration, diagnostic and the 512-token windows are disjoint token ranges of the "
                                "validation split; final test is the test split"},
        "quantization": {"searched": "MatMul projections (FFN gate/up/down; attention projections in the second "
                                     "search version); fused projections (in_proj_qkv, q_proj with its output gate) "
                                     "are one node each", "fp32_fixed": ["embedding / tied LM head", "all RMSNorms",
                         "causal Conv1D", "gated delta rule (decay, beta, l2norm, recurrent state)", "softmax, rotary",
                         "attention output gate", "in-chunk triangular solve"],
                         "semantics": "the pass's per-consumer QDQ, int (scale, zp) from the calibration ranges, "
                                      "per-output-channel int weights, Universal-exact posit"},
        "ranking": {"layers": reps, "rule": "first, middle (index n//2) and last layer of each block type",
                    "modules": ["gate", "up", "down", "gate+up"], "modes": ["w", "a", "wa"],
                    "formats": list(RANK_FORMATS), "joint": "gate+up, every pair of formats (incl. mixed)",
                    "metrics": ["within (layer, mode, module set) Spearman", "equal M and B pair accuracy",
                                "loss gap of BC's pick to the best tested equal-resource candidate",
                                "joint vs single non-additivity", "per-window dNLL, paired window bootstrap (2000)"]},
        "correctness_gates": {"forward_logits_rel_l2_vs_torch_fp32": 1e-4, "window_summed_ce_abs": 0.05,
                              "gradient_median_rel_max": 1e-3, "gradient_max_rel_max": 5e-2,
                              "stats_rel_median": 1e-3},
        "search": {"policy": "mp_auto.auto_select (2-D, normalized chord knee)", "samples": 17, "refinements": 2,
                   "time_limit_s": 10, "grid": "GRID11", "versions": ["FFN-only", "FFN + attention projections"]},
        "seeds": {"bootstrap": 0}, "evaluation": "final test only after the configuration is frozen",
    }
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    jdump(out / "plan.json", plan)
    # tokens
    ids = {s: q35.wikitext_ids(s) for s in ("validation", "test")}
    sets = {}
    for k, (s, f, n) in {**DATA, **DATA_512}.items():
        L = 512 if k.endswith("_512") else SEQ
        a = np.asarray(ids[s][f:f + n * L], np.int64).reshape(n, L)
        np.save(out / f"tokens_{k}.npy", a)
        sets[k] = {"shape": list(a.shape), "sha256": sha256(out / f"tokens_{k}.npy")}
    log(f"tokens: validation {len(ids['validation'])}, test {len(ids['test'])}")
    # export
    import onnx
    from qwen3_prepare import fold_constants
    import shutil
    ad = q35.TextExport(hf, SEQ).eval()
    x = torch.from_numpy(np.load(out / "tokens_calibration.npy")[:1])
    t1 = time.time()
    tmp = out / "export_tmp"
    tmp.mkdir()
    torch.onnx.export(ad, (x,), str(tmp / "model_raw.onnx"), input_names=["input_ids"], output_names=["logits"],
                      opset_version=17, dynamo=False, do_constant_folding=True, external_data=True)
    export_s = time.time() - t1
    m = onnx.load(str(tmp / "model_raw.onnx"))
    shutil.rmtree(tmp)
    folded = fold_constants(m)
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    mapping = q35.map_projection_nodes(m, hf, cfg)
    onnx.save(m, str(out / "model.onnx"), save_as_external_data=True, location="model.onnx.data")
    jdump(out / "node_mapping.json", mapping)
    # ORT against the PyTorch adapter
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    z = sess.run(None, {"input_ids": x.numpy()})[0]
    with torch.no_grad():
        ref = ad(x).numpy()
    ort_rel = float(np.linalg.norm(z - ref) / np.linalg.norm(ref))
    jdump(out / "prepare.json", {"tokens": sets, "export_s": export_s, "folded": folded, "ops": dict(ops),
                                 "nodes": len(m.graph.node), "searched_nodes": len(mapping),
                                 "ort_vs_torch_logits_rel_l2": ort_rel, "seconds": time.time() - t0})
    from qwen3_swiglu_common import environment
    import transformers
    jdump(out / "environment.json", {**environment(["experiments/mixed_precision/run_qwen35_bc.py",
                                                    "experiments/mixed_precision/qwen35_common.py"]),
                                     "qwen35": {"repo": q35.REPO, "revision": q35.REVISION,
                                                "transformers": transformers.__version__,
                                                "config_sha256": sha256(q35.MODEL_DIR / "config.json"),
                                                "weights_sha256": sha256(q35.MODEL_DIR / "model.safetensors-00001-of-00001.safetensors"),
                                                "tokenizer_sha256": sha256(q35.MODEL_DIR / "tokenizer.json"),
                                                "model_onnx_sha256": sha256(out / "model.onnx")},
                                     "qwen3_baseline": {"table_sha256": sha256(HERE / "runs/native_eps_fix_20261014/qwen3/table.json")}})
    log(f"exported {len(m.graph.node)} nodes ({dict(ops)}), {len(mapping)} searched, ORT rel_l2 {ort_rel:.2e}")


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def native_import(out, log, model="model.onnx", sub="native"):
    """onnx-mlir --EmitONNXBasic of model.onnx, as bytecode (as native_tables does), once."""
    from run_native_bc_pipeline import erm
    tools = erm.Tools()
    work = out / sub
    work.mkdir(exist_ok=True)
    base = work / f"{erm.SYMBOL_POSTFIX}.mlirbc"
    if not base.exists():
        t0 = time.time()
        text = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
        erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                 "--mlir-elide-resource-strings-if-larger=1000000000",
                 "--mlir-elide-elementsattrs-if-larger=1000000000", out / model])
        erm.run([tools.opt, text, "--emit-bytecode", "-o", base])
        text.unlink()
        log(f"import + bytecode {time.time() - t0:.0f}s")
    return tools, work, base


def torch_reference(out, sname, dtype):
    """Per-window logits (FP32: as exported) and summed CE of the adapter, cached."""
    f = out / "torch_ref" / f"{sname}_{str(dtype).split('.')[-1]}.npz"
    if f.exists():
        z = np.load(f)
        return z["logits"], z["ce"]
    f.parent.mkdir(exist_ok=True)
    toks = np.load(out / f"tokens_{sname}.npy")
    hf = q35.load_hf(dtype)
    ad = q35.TextExport(hf, toks.shape[1], dtype=dtype).eval()
    logits, ce = [], []
    with torch.no_grad():
        for t in toks:
            x = torch.from_numpy(t)
            z = ad(x[None])
            ce.append(float(q35.summed_ce(z, x)))
            logits.append(z[0].float().numpy())
    np.savez(f, logits=np.stack(logits), ce=np.array(ce))
    return np.stack(logits), np.array(ce)


def forward_check(out, tools, work, base, sname, chunks, log):
    """The plain FP32 build (as evaluated) on the windows tokens_<sname>.npy against the adapter in FP32 / FP64."""
    from run_native_bc_pipeline import cross_entropy, erm
    from evaluate_native_search_chunked import chunked_run_so
    t0 = time.time()
    target = work / "fp32"
    target.mkdir(exist_ok=True)
    so, info = erm.build_so(tools, target, "fp32", base, None, None, opt="O0", krnl_opt="O3",
                            constants_to_file=True, bytecode=True, uses_posit=False, uses_f16=False)
    build_s = time.time() - t0
    for p in target.iterdir():
        if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
            p.unlink()
    toks = np.load(out / f"tokens_{sname}.npy")
    np.save(target / "inputs.npy", toks)
    z, ev = chunked_run_so(chunks)(tools, so, target / "inputs.npy", target / "logits.npy", 36000)
    (target / "logits.npy").unlink()
    ce = np.array([float(cross_entropy(z[s, :-1], toks[s, 1:]).sum()) for s in range(len(toks))])
    ref32, ce32 = torch_reference(out, sname, torch.float32)
    ref64, ce64 = torch_reference(out, sname, torch.float64)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    res = {
        "windows": len(toks), "build_s": build_s, "build_steps": info["steps"], "warnings": info["warnings"],
        "run": ev, "finite": bool(np.isfinite(z).all()),
        "logits_rel_l2_vs_torch_fp32": [rel(z[s], ref32[s]) for s in range(len(toks))],
        "logits_rel_l2_vs_torch_fp64": [rel(z[s], ref64[s]) for s in range(len(toks))],
        "torch_fp32_vs_fp64_rel_l2": [rel(ref32[s], ref64[s]) for s in range(len(toks))],
        "summed_ce_native": ce.tolist(), "summed_ce_torch_fp32": ce32.tolist(), "summed_ce_torch_fp64": ce64.tolist(),
        "ce_abs_diff_native_fp32_max": float(np.abs(ce - ce32).max()),
        "ce_abs_diff_native_fp64_max": float(np.abs(ce - ce64).max()),
        "seconds": time.time() - t0}
    gate = jload(out / "plan.json")["correctness_gates"]
    res["pass"] = bool(max(res["logits_rel_l2_vs_torch_fp32"]) <= gate["forward_logits_rel_l2_vs_torch_fp32"]
                       and res["ce_abs_diff_native_fp32_max"] <= gate["window_summed_ce_abs"])
    for p in (so, target / "model.constants.bin"):
        p.unlink(missing_ok=True)
    log(f"forward {sname}: build {build_s:.0f}s, logits rel_l2 vs fp32 max {max(res['logits_rel_l2_vs_torch_fp32']):.2e}, "
        f"vs fp64 max {max(res['logits_rel_l2_vs_torch_fp64']):.2e}, CE |diff| max {res['ce_abs_diff_native_fp32_max']:.4f}, "
        f"pass {res['pass']}")
    return res


def stage_correctness(args, out, log):
    res = jload(out / "correctness.json") if (out / "correctness.json").exists() else {}
    tools, work, base = native_import(out, log)
    if "forward" in args.parts and "forward" not in res:
        res["forward"] = forward_check(out, tools, work, base, "calibration", 5, log)
        jdump(out / "correctness.json", res)
    if "backward" in args.parts and "backward" not in res:
        res["backward"] = backward_check(out, tools, work, base, log, windows=2)
        jdump(out / "correctness.json", res)
    if "stats" in args.parts and "stats" not in res:
        res["stats"] = stats_check(out, log)
        jdump(out / "correctness.json", res)


def torch_bc_table(out, nodes, toks, ranges=None):
    """The native BC table's statistics and entries for `nodes`, from the adapter in FP32 with autograd on the
    windows `toks`: per-tensor ranges, per-channel gradient sums of each tap (x: total gradient of the shared input),
    per-channel activation quantization error sums (int formats with `ranges`; None: the ranges measured here, as
    the native ranges pass does, in a first forward pass), per-output-channel weight gradients and quantization
    errors; B[n][f], C[n][f] by run_native_bc_pipeline.native_tables' formulas, for every quantized GRID11 format."""
    from qwen3_swiglu_common import Quantizers
    from run_native_bc_pipeline import QUANT
    q = Quantizers(out, formats=("p4e1", "p6e1", "p8e1"))
    hf = q35.load_hf(torch.float32)
    mods, params = dict(hf.named_modules()), dict(hf.named_parameters())
    ad = q35.TextExport(hf, toks.shape[1]).eval()
    lo, hi, G2, GS, S1, S2, Gw = {}, {}, {}, {}, {}, {}, {}

    def capture(ids, grad):
        cap, hooks = {}, []
        for n, info in nodes.items():
            mod = mods[info["param"].removesuffix(".weight")]

            def pre(m, inp, n=n):
                if grad:
                    inp[0].retain_grad()
                cap[(n, "x")] = inp[0]

            def post(m, inp, o, n=n):
                if grad:
                    o.retain_grad()
                cap[(n, "y")] = o
            hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
        x = torch.from_numpy(ids)
        z = ad(x[None])
        if grad:
            q35.summed_ce(z, x).backward()
        for h in hooks:
            h.remove()
        return cap
    for ids in toks:                                     # ranges over every window, as the native ranges pass
        with torch.no_grad():
            for key, t in capture(ids, False).items():
                lo[key] = min(lo.get(key, np.inf), float(t.min()))
                hi[key] = max(hi.get(key, -np.inf), float(t.max()))
    rng = ranges or {n: {s: [lo[(n, s)], hi[(n, s)]] for s in ("x", "y")} for n in nodes}
    hf.model.language_model.embed_tokens.weight.requires_grad_(True)
    for n, info in nodes.items():
        params[info["param"]].requires_grad_(True)
    for ids in toks:
        cap = capture(ids, True)
        with torch.no_grad():
            for key, t in cap.items():
                a, g = t.detach()[0], t.grad[0].double()
                G2[key] = G2.get(key, 0) + (g ** 2).sum(0)
                GS[key] = GS.get(key, 0) + g.sum(0) ** 2
                for f in QUANT:
                    e = (q.act(a, f, rng[key[0]][key[1]]) - a).double()
                    S1.setdefault(key, {})[f] = S1.get(key, {}).get(f, 0) + e.sum(0)
                    S2.setdefault(key, {})[f] = S2.get(key, {}).get(f, 0) + (e ** 2).sum(0)
            for n, info in nodes.items():
                p = params[info["param"]]
                Gw[n] = Gw.get(n, 0) + (p.grad.double() ** 2).sum(1)
                p.grad = None
        hf.model.language_model.embed_tokens.weight.grad = None
    cnt = len(toks) * toks.shape[1]
    B, C, err2 = {}, {}, {}
    with torch.no_grad():
        for n, info in nodes.items():
            w = params[info["param"]].detach()
            B[n], C[n], err2[n] = {"fp32": 0.0}, {"fp32": 0.0}, {}
            for f in QUANT:
                e2 = ((q.weight(w, f) - w).double() ** 2).sum(1)
                err2[n][f] = e2.numpy()
                B[n][f] = float((Gw[n] / w.shape[1] * e2).sum())
                c = 0.0
                for sl in ("x", "y"):
                    mu = S1[(n, sl)][f] / cnt
                    c += float((G2[(n, sl)] / cnt * (S2[(n, sl)][f] - cnt * mu ** 2)).sum() + (GS[(n, sl)] * mu ** 2).sum())
                C[n][f] = c
    return {"B": B, "C": C, "err2": err2, "ranges": {n: {s: [lo[(n, s)], hi[(n, s)]] for s in ("x", "y")} for n in nodes},
            "ranges_used": rng}


def stats_check(out, log, nodes=None, toks="tokens_calibration.npy", table="table.json",
                ranges="ranges_by_point.json", werr="bc/weight_err.json"):
    """End-to-end check of the native table (after the tables stage) on every projection of the representative
    layers: the adapter (FP32, autograd) recomputes on the calibration windows the native statistics -- per-tensor
    ranges, per-channel gradient sums (total gradient of a shared input), per-channel activation quantization error
    sums (int formats with the native ranges), per-output-channel weight quantization errors and weight gradients --
    and from them the table entries B[n][f], C[n][f] for every quantized GRID11 format, with the native formulas."""
    from run_native_bc_pipeline import QUANT
    t0 = time.time()
    plan, mapping = jload(out / "plan.json"), jload(out / "node_mapping.json")
    layers = {l for v in plan["ranking"]["layers"].values() for l in v}
    nodes = nodes or {n: v for n, v in mapping.items() if v["layer"] in layers}
    layers = {v["layer"] for v in nodes.values()}
    table, ranges = jload(out / table), jload(out / ranges)
    werr = {e["node"]: e for e in jload(out / werr)["nodes"]}
    toks = np.load(out / toks)
    ref = torch_bc_table(out, nodes, toks, ranges)
    rows, rng_err, werr_err = [], [], []
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    for n, info in nodes.items():
        for f in QUANT:
            nat = np.array([np.inf if v is None else v for v in werr[n]["err2"][f]])
            e2 = ref["err2"][n][f]
            werr_err.append(float(np.abs(nat - e2).max() / max(float(np.abs(e2).max()), 1e-300)))
            b_ref, c_ref = ref["B"][n][f], ref["C"][n][f]
            rows.append({"node": n, "layer": info["layer"], "kind": info["kind"], "format": f,
                         "B_native": table["B"][n][f], "B_torch": b_ref, "B_rel": rel(table["B"][n][f], b_ref),
                         "C_native": table["C"][n][f], "C_torch": c_ref, "C_rel": rel(table["C"][n][f], c_ref)})
        for sl in ("x", "y"):
            r = ranges[n][sl]
            lo, hi = ref["ranges"][n][sl]
            rng_err.append(max(abs(r[0] - lo), abs(r[1] - hi)) / (r[1] - r[0]))
    b = np.array([r["B_rel"] for r in rows])
    c = np.array([r["C_rel"] for r in rows])
    from scipy.stats import spearmanr
    rank = []
    for n in nodes:
        for k in ("B", "C"):
            nat = [r[f"{k}_native"] for r in rows if r["node"] == n]
            ref = [r[f"{k}_torch"] for r in rows if r["node"] == n]
            rank.append(float(spearmanr(nat, ref)[0]))
    gate = plan["correctness_gates"]["stats_rel_median"]
    res = {"nodes": len(nodes), "layers": sorted(layers), "formats": list(QUANT), "windows": int(len(toks)),
           "reference": "adapter FP32 autograd; int activation formats use the native ranges (checked separately)",
           "B_rel": {"median": float(np.median(b)), "p99": float(np.percentile(b, 99)), "max": float(b.max())},
           "C_rel": {"median": float(np.median(c)), "p99": float(np.percentile(c, 99)), "max": float(c.max())},
           "worst_C": sorted(rows, key=lambda r: -r["C_rel"])[:5], "worst_B": sorted(rows, key=lambda r: -r["B_rel"])[:5],
           "ranges_err_over_width": {"median": float(np.median(rng_err)), "max": float(max(rng_err))},
           "weight_err2_rel_max": {"median": float(np.median(werr_err)), "max": float(max(werr_err))},
           "within_node_format_spearman_min": float(np.nanmin(rank)),
           "rows": rows, "seconds": time.time() - t0}
    res["pass"] = bool(res["B_rel"]["median"] <= gate and res["C_rel"]["median"] <= gate
                       and res["ranges_err_over_width"]["max"] <= 1e-4)
    log(f"stats: {len(nodes)} nodes x {len(QUANT)} formats, B rel median {res['B_rel']['median']:.2e} max "
        f"{res['B_rel']['max']:.2e}, C rel median {res['C_rel']['median']:.2e} max {res['C_rel']['max']:.2e}, ranges "
        f"max {res['ranges_err_over_width']['max']:.2e} of width, weight err2 max {res['weight_err2_rel_max']['max']:.2e}, "
        f"format-rank Spearman min {res['within_node_format_spearman_min']:.3f}, pass {res['pass']}")
    return res


def backward_check(out, tools, work, base, log, windows=2):
    """Native gradient statistics (grads mode, the 186 searched nodes, the statistics build levels of a MatMul
    model) on calibration windows against the adapter's autograd in FP64 (and FP32 for scale)."""
    from run_native_bc_pipeline import erm
    from qwen35_tiny_check import torch_stats, rel_max
    mapping = jload(out / "node_mapping.json")
    t0 = time.time()
    d = work / "correct_grads"
    d.mkdir(exist_ok=True)
    layout_path, inst = d / "grads_layout.json", d / "grads.onnx.mlirbc"
    _, t_pass = erm.run([tools.opt, base, "--shape-inference", "--mixedp-collect-stats", "--emit-bytecode", "-o", inst],
                        env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": ",".join(mapping),
                             "MIXEDP_STATS_LAYOUT": str(layout_path)})
    so, info = erm.build_so(tools, d, "grads", inst, None, None, opt="O0", krnl_opt="O3", constants_to_file=True,
                            bytecode=True, uses_posit=False, uses_f16=False)
    for p in d.iterdir():
        if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
            p.unlink()
    build_s = time.time() - t0
    layout = jload(layout_path)
    toks = np.load(out / "tokens_calibration.npy")[:windows]
    vocab = jload(out / "plan.json")["architecture"]["vocab"]
    nat, run_s = [], []
    for k, ids in enumerate(toks):
        oh = np.zeros((1, ids.shape[0], vocab), np.float32)
        oh[0, np.arange(ids.shape[0] - 1), ids[1:]] = 1.0
        np.save(d / "x.npy", toks)
        np.save(d / f"oh{k}.npy", oh)
        r, dt = erm.run([sys.executable, HERE / "run_native_bc_pipeline.py", "--_stats-child", so, erm.SYMBOL_POSTFIX,
                         tools.lib, d / "x.npy", k, d / f"oh{k}.npy", layout["result_index"], d / f"stats{k}.npy"])
        run_s.append(json.loads(r.stdout.strip().splitlines()[-1]))
        v = np.load(d / f"stats{k}.npy")
        taps = {t["tap"]: t for t in layout["taps"]}
        st = {}
        for p in layout["points"]:
            t = taps[p["tap"]]
            o, ch = t["offset"], t["channels"]
            st[(p["node"], p["slot"])] = (v[o:o + ch], v[o + ch:o + 2 * ch])
        for w in layout["weights"]:
            st[(w["node"], "w")] = v[w["offset"]:w["offset"] + w["channels"]]
        nat.append(st)
        (d / f"oh{k}.npy").unlink()
    log(f"native grads: build {build_s:.0f}s, run {np.mean([r['run_s'] for r in run_s]):.0f}s per window")
    res = {"build_s": build_s, "build_steps": info["steps"], "pass_s": t_pass, "runs": run_s, "windows": int(windows),
           "layout": {"weights": len(layout["weights"]), "taps": len(layout["taps"]), "points": len(layout["points"])}}
    for dname, dtype in (("fp64", torch.float64), ("fp32", torch.float32)):
        hf = q35.load_hf(dtype)
        errs_all, per = [], []
        for k, ids in enumerate(toks):
            _, st, loss = torch_stats(hf, hf.model.language_model.config, ids, mapping, dtype)
            errs = {}
            for key, ref_v in st.items():
                nv = nat[k][key]
                if key[1] == "w":
                    errs[f"{key[0]}|w"] = rel_max(nv, ref_v)
                else:
                    errs[f"{key[0]}|{key[1]}|g2"] = rel_max(nv[0], ref_v[0])
                    errs[f"{key[0]}|{key[1]}|g"] = rel_max(nv[1], ref_v[1])
            e = np.array(list(errs.values()))
            errs_all += list(e)
            per.append({"loss": loss, "arrays": len(e), "median": float(np.median(e)), "max": float(e.max()),
                        "worst": sorted(errs.items(), key=lambda kv: -kv[1])[:6]})
        e = np.array(errs_all)
        res[f"native_vs_torch_{dname}"] = {"median": float(np.median(e)), "p99": float(np.percentile(e, 99)),
                                           "max": float(e.max()), "windows": per}
        del hf
        log(f"native vs torch {dname}: median {np.median(e):.2e}, p99 {np.percentile(e, 99):.2e}, max {e.max():.2e}")
    gate = jload(out / "plan.json")["correctness_gates"]
    f64 = res["native_vs_torch_fp64"]
    res["pass"] = bool(f64["median"] <= gate["gradient_median_rel_max"] and f64["max"] <= gate["gradient_max_rel_max"])
    res["seconds"] = time.time() - t0
    return res


# --------------------------------------------------------------------------- #
# Tables, candidates, freeze
# --------------------------------------------------------------------------- #
def stage_tables(args, out, log):
    """The native BC table of all 186 searched projections (run_native_bc_pipeline.native_tables, unchanged:
    C = W + A, total input gradient, the calibration windows)."""
    import shutil
    from run_native_bc_pipeline import erm, native_tables, qwen35_spec
    tools = erm.Tools()
    work = out / "bc"
    work.mkdir(exist_ok=True)
    spec = qwen35_spec(work, out)
    t0 = time.time()
    B, C, ranges, timing = native_tables(spec, tools, work, log, input_grad="total")
    jdump(out / "table.json", {"B": B, "C": C, "naming": "B = W (weight criterion), C = A (activation criterion)"})
    shutil.copy(work / "ranges_by_point.json", out / "ranges_by_point.json")
    nonfinite = {n: [f for f, v in {**B[n], **{f"a:{k}": x for k, x in C[n].items()}}.items() if not np.isfinite(v)]
                 for n in B}
    jdump(out / "tables.json", {"timing": timing, "nodes": len(B), "seconds": time.time() - t0,
                                "nonfinite_options": {n: v for n, v in nonfinite.items() if v}})
    log(f"table: {len(B)} nodes in {timing['criterion_total_s']:.0f}s")


def ranking_candidates(out):
    """FFN candidates at the representative layers (plan.json): gate / up / down alone and gate+up, modes w, a, wa,
    formats RANK_FORMATS (gate+up: every pair, mixed included). An option is (weight format, activation format)."""
    plan = jload(out / "plan.json")
    mapping = jload(out / "node_mapping.json")
    node = {(v["layer"], v["kind"]): n for n, v in mapping.items()}
    layers = [l for t in ("linear_attention", "full_attention") for l in plan["ranking"]["layers"][t]]
    opt = {"w": lambda f: (f, "fp32"), "a": lambda f: ("fp32", f), "wa": lambda f: (f, f)}
    cands = []
    for l in layers:
        for mode in ("w", "a", "wa"):
            for kind in FFN:
                for f in RANK_FORMATS:
                    cands.append({"layer": l, "mode": mode, "set": kind, "formats": {kind: opt[mode](f)}})
            for fg in RANK_FORMATS:
                for fu in RANK_FORMATS:
                    cands.append({"layer": l, "mode": mode, "set": "gate+up",
                                  "formats": {"gate_proj": opt[mode](fg), "up_proj": opt[mode](fu)}})
    for c in cands:
        c["block"] = mapping[node[(c["layer"], "gate_proj")]]["block"]
        c["nodes"] = {k: node[(c["layer"], k)] for k in c["formats"]}
        c["name"] = f"L{c['layer']}|{c['mode']}|" + "|".join(f"{k.split('_')[0]}={w}/{a}" for k, (w, a) in sorted(c["formats"].items()))
    return cands


def stage_freeze(args, out, log):
    from qwen3_swiglu_common import sha256
    from htq_pareto_search import FORMAT_BITS
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    table = jload(out / "table.json")
    cands = ranking_candidates(out)
    mapping = jload(out / "node_mapping.json")
    for c in cands:
        c["bc"] = float(sum(table["B"][c["nodes"][k]][w] + table["C"][c["nodes"][k]][a] for k, (w, a) in c["formats"].items()))
        size = bops = 0.0
        for kind in FFN:                       # resources of the layer's FFN: untouched nodes count as fp32
            n = [x for x, v in mapping.items() if v["layer"] == c["layer"] and v["kind"] == kind][0]
            w, a = c["formats"].get(kind, ("fp32", "fp32"))
            e = int(np.prod(mapping[n]["shape_in_out"]))
            size += e * FORMAT_BITS[w] / 8
            bops += e * SEQ * FORMAT_BITS[w] * FORMAT_BITS[a]
        c["M_bytes"], c["B_bops"] = size, bops
    jdump(out / "candidates.json", cands)
    jdump(out / "freeze.json", {"candidates": len(cands), "candidates_sha256": sha256(out / "candidates.json"),
                                "table_sha256": sha256(out / "table.json"),
                                "ranges_sha256": sha256(out / "ranges_by_point.json"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"frozen {len(cands)} candidates")


# --------------------------------------------------------------------------- #
# Truth: diagnostic-window loss changes (PyTorch simulation of the pass's QDQ)
# --------------------------------------------------------------------------- #
def stage_truth(args, out, log):
    """Every frozen candidate on the diagnostic windows: the adapter (FP32) with the candidate's FFN projections of
    one layer quantized as the pass does (per-output-channel int weights, int activations with the calibration
    (scale, zp) of ranges_by_point.json, Universal-exact posit LUTs, each consumer quantizing its own input copy),
    replayed from that layer on the cached residual stream; summed next-token CE per window, minus FP32's."""
    from qwen3_swiglu_common import sha256
    fz = jload(out / "freeze.json")
    for f, k in (("candidates.json", "candidates_sha256"), ("table.json", "table_sha256"), ("ranges_by_point.json", "ranges_sha256")):
        if sha256(out / f) != fz[k]:
            raise SystemExit(f"{f} changed after the freeze")
    if not (out / "sqnr_freeze.json").exists() or sha256(out / "sqnr.json") != jload(out / "sqnr_freeze.json")["sqnr_sha256"]:
        raise SystemExit("freeze the SQNR reference first (--stage sqnr)")
    cands = [c for c in jload(out / "candidates.json") if (args.layers is None or c["layer"] in args.layers)
             and (args.names is None or c["name"] in args.names)]
    replay_truth(out, cands, np.load(out / "tokens_diagnostic.npy"), jload(out / "ranges_by_point.json"),
                 out / args.truth_file, args.batch, log, device=args.device)


def replay_truth(out, cands, toks, ranges, f, batch, log, mapping="node_mapping.json", device="cpu"):
    """Summed next-token CE change per window of each candidate, replayed from its layer on the FP32 residual
    stream; resumable (f). device="cuda": the same FP32 arithmetic on the GPU (TF32 off), the posit LUTs moved
    there after their CPU verification; the device is recorded per row."""
    from qwen3_swiglu_common import Quantizers
    mapping = jload(out / mapping)
    q = Quantizers(out, formats=("p6e1", "p8e1"))
    for fmt, lut in q.posit.items():
        bad, n = lut.verify()
        if bad:
            raise SystemExit(f"posit LUT {fmt}: {bad} of {n} values differ from Universal")
    import copy
    qa = copy.copy(q)                       # activations on the device; weights are quantized on the CPU (numpy)
    qa.posit = {}
    for fmt, lut in q.posit.items():
        qa.posit[fmt] = copy.copy(lut)
        qa.posit[fmt].t_values, qa.posit[fmt].t_bounds = lut.t_values.to(device), lut.t_bounds.to(device)
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    hf = q35.load_hf(torch.float32)
    ad = q35.TextExport(hf, toks.shape[1]).eval().to(device)
    mods = dict(hf.named_modules())
    toks = torch.from_numpy(toks).to(device)
    done = jload(f) if f.exists() else {"rows": {}, "base": {}, "seconds": 0.0}
    done.setdefault("device", {})
    t0 = time.time()
    chunk = batch

    def ce_of(h, start):
        r = []
        for i in range(0, toks.shape[0], chunk):
            z = ad.forward_from(h[i:i + chunk], start)
            ids = toks[i:i + chunk]
            r.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), ids[:, 1:].reshape(-1),
                                                       reduction="none").view(z.shape[0], -1).double().sum(1))
        return torch.cat(r).cpu().numpy()
    with torch.no_grad():
        for layer in sorted({c["layer"] for c in cands}):
            todo = [c for c in cands if c["layer"] == layer and c["name"] not in done["rows"]]
            if not todo:
                continue
            h = torch.cat([ad.hidden_before(toks[i:i + chunk], layer) for i in range(0, toks.shape[0], chunk)])
            base = ce_of(h, layer)
            done["base"][str(layer)] = base.tolist()
            for c in todo:
                saved, hooks = {}, []
                for kind, (w, a) in c["formats"].items():
                    n = c["nodes"][kind]
                    mod = mods[mapping[n]["param"].removesuffix(".weight")]
                    if w != "fp32":
                        saved[kind] = mod.weight.data
                        mod.weight.data = q.weight(mod.weight.data, w).to(device)
                    if a != "fp32":
                        rx, ry = ranges[n]["x"], ranges[n]["y"]
                        hooks.append(mod.register_forward_pre_hook(lambda m, i, a=a, rx=rx: (qa.act(i[0], a, rx),)))
                        hooks.append(mod.register_forward_hook(lambda m, i, o, a=a, ry=ry: qa.act(o, a, ry)))
                try:
                    dl = ce_of(h, layer) - base
                finally:
                    for hk in hooks:
                        hk.remove()
                    for kind, w0 in saved.items():
                        mods[mapping[c["nodes"][kind]]["param"].removesuffix(".weight")].weight.data = w0
                done["rows"][c["name"]] = dl.tolist()
                done["device"][c["name"]] = device
                if len(done["rows"]) % 10 == 0:
                    done["seconds"] = done.get("seconds", 0.0) + time.time() - t0
                    t0 = time.time()
                    jdump(f, done)
            log(f"layer {layer}: {len(todo)} candidates, {len(done['rows'])} done, {done['seconds'] + time.time() - t0:.0f}s")
    done["seconds"] = done.get("seconds", 0.0) + time.time() - t0
    jdump(f, done)


# --------------------------------------------------------------------------- #
# SQNR reference (calibration only, frozen before the truth) and ranking analysis
# --------------------------------------------------------------------------- #
def stage_sqnr(args, out, log):
    """The SQNR criterion of sqnr_tables.py (N = sum over the calibration windows of ||logits - logits_fp32||^2 with
    one node quantized; a set of nodes scores the sum of its nodes' N) for the FFN options of the ranking candidates,
    by the same PyTorch replay as the truth (sqnr_tables.py's onnxruntime simulation has no posit activations). A
    reference only: not used by the search. Must run before the truth stage."""
    from qwen3_swiglu_common import Quantizers, sha256
    if (out / "truth.json").exists():
        raise SystemExit("the SQNR reference must be frozen before any diagnostic forward")
    fz = jload(out / "freeze.json")
    if sha256(out / "candidates.json") != fz["candidates_sha256"]:
        raise SystemExit("candidates.json changed after the freeze")
    cands = jload(out / "candidates.json")
    ranges, mapping = jload(out / "ranges_by_point.json"), jload(out / "node_mapping.json")
    opts = sorted({(c["layer"], kind, w, a) for c in cands for kind, (w, a) in c["formats"].items()})
    if args.layers is not None:
        opts_run = [o for o in opts if o[0] in args.layers]
    else:
        opts_run = opts
    q = Quantizers(out, formats=("p6e1", "p8e1"))
    hf = q35.load_hf(torch.float32)
    ad = q35.TextExport(hf, SEQ).eval()
    mods = dict(hf.named_modules())
    node = {(v["layer"], v["kind"]): n for n, v in mapping.items()}
    toks = torch.from_numpy(np.load(out / "tokens_calibration.npy"))
    f = out / "sqnr_partial.json"
    done = jload(f) if f.exists() else {"N": {}, "seconds": 0.0}
    t0 = time.time()
    chunk = args.batch
    with torch.no_grad():
        ref = torch.cat([ad(toks[i:i + chunk]) for i in range(0, len(toks), chunk)])
        for layer in sorted({o[0] for o in opts_run}):
            if all(f"L{l}|{kind}|{w}/{a}" in done["N"] for (l, kind, w, a) in opts_run if l == layer):
                continue
            h = torch.cat([ad.hidden_before(toks[i:i + chunk], layer) for i in range(0, len(toks), chunk)])
            for (l, kind, w, a) in [o for o in opts_run if o[0] == layer]:
                key = f"L{l}|{kind}|{w}/{a}"
                if key in done["N"]:
                    continue
                n = node[(l, kind)]
                mod = mods[mapping[n]["param"].removesuffix(".weight")]
                w0, hooks = mod.weight.data, []
                if w != "fp32":
                    mod.weight.data = q.weight(w0, w)
                if a != "fp32":
                    hooks.append(mod.register_forward_pre_hook(lambda m, i, a=a, r=ranges[n]["x"]: (q.act(i[0], a, r),)))
                    hooks.append(mod.register_forward_hook(lambda m, i, o, a=a, r=ranges[n]["y"]: q.act(o, a, r)))
                try:
                    tot = 0.0
                    for i in range(0, len(toks), chunk):
                        tot += float(((ad.forward_from(h[i:i + chunk], layer) - ref[i:i + chunk]).double() ** 2).sum())
                finally:
                    for hk in hooks:
                        hk.remove()
                    mod.weight.data = w0
                done["N"][key] = tot
                if len(done["N"]) % 10 == 0:
                    done["seconds"] += time.time() - t0
                    t0 = time.time()
                    jdump(f, done)
            log(f"sqnr layer {layer} done ({len(done['N'])}/{len(opts)})")
    done["seconds"] += time.time() - t0
    jdump(f, done)
    missing = [o for o in opts if f"L{o[0]}|{o[1]}|{o[2]}/{o[3]}" not in done["N"]]
    if missing:
        log(f"sqnr: {len(missing)} options left; rerun for the remaining layers")
        return
    scores = {c["name"]: float(sum(done["N"][f"L{c['layer']}|{k}|{w}/{a}"] for k, (w, a) in c["formats"].items()))
              for c in cands}
    jdump(out / "sqnr.json", {"definition": "sum over calibration windows of ||logits - logits_fp32||^2 per quantized "
                                            "node, summed over the candidate's nodes (sqnr_tables.py)",
                              "scores": scores, "per_option": done["N"], "seconds": done["seconds"]})
    jdump(out / "sqnr_freeze.json", {"sqnr_sha256": sha256(out / "sqnr.json"), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})


def _pair_class(ci, cj):
    if ci["set"] != "gate+up" and cj["set"] != "gate+up":
        if ci["set"] == cj["set"]:
            return "single: int vs posit, same branch"
        fi, fj = ci["formats"][ci["set"]], cj["formats"][cj["set"]]
        b = "-".join(sorted(k.split("_")[0] for k in (ci["set"], cj["set"])))
        return f"single: branch choice {b}, same format" if fi == fj else f"single: branch choice {b}, int vs posit"
    if ci["set"] == "gate+up" and cj["set"] == "gate+up":
        gi, ui = ci["formats"]["gate_proj"], ci["formats"]["up_proj"]
        gj, uj = cj["formats"]["gate_proj"], cj["formats"]["up_proj"]
        if gi == uj and ui == gj:
            return "joint: gate/up swap"
        if gi == gj or ui == uj:
            return "joint: int vs posit in one branch"
        return "joint: other"
    return "single vs joint"


def stage_analyze(args, out, log):
    """The frozen BC scores (and the frozen SQNR reference) against the diagnostic truth. Every interval is a paired
    window bootstrap (2000 resamples of the 30 windows, every candidate's mean recomputed on the same resample): it
    covers window noise only; pairs and candidates share windows and are not independent samples."""
    from scipy.stats import spearmanr
    import csv
    cands = jload(out / "candidates.json")
    truth = jload(out / "truth.json")
    T = SEQ - 1
    names = [c["name"] for c in cands]
    D = np.array([truth["rows"][n] for n in names]) / T            # [candidate, window] per-token dNLL
    nw = D.shape[1]
    boot = np.random.default_rng(jload(out / "plan.json")["seeds"]["bootstrap"]).integers(0, nw, (2000, nw))
    Dm = D.mean(1)
    Db = np.stack([D[:, b].mean(1) for b in boot], 1)                # [candidate, 2000]
    crit = {"BC": np.array([c["bc"] for c in cands])}
    if (out / "sqnr.json").exists():
        sq = jload(out / "sqnr.json")["scores"]
        crit["SQNR"] = np.array([sq[n] for n in names])
    idx = {n: i for i, n in enumerate(names)}
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["layer"], c["mode"]), []).append(i)
    res = {"windows": nw, "tokens_per_window": T, "criteria": list(crit), "spearman": {}, "pairs": {}, "pick_gap": {},
           "joint": {}}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    # Spearman within (layer, mode): all 40 candidates, singles (15), joint (25)
    for (l, m), ix in groups.items():
        row = {}
        for sub, sel in (("all", ix), ("single", [i for i in ix if cands[i]["set"] != "gate+up"]),
                         ("joint", [i for i in ix if cands[i]["set"] == "gate+up"])):
            for k, v in crit.items():
                rho = spearmanr(v[sel], Dm[sel])[0]
                rb = [spearmanr(v[sel], Db[sel, b])[0] for b in range(0, 2000, 10)]
                row[f"{k}_{sub}"] = {"rho": float(rho), "ci95": ci(rb)}
        res["spearman"][f"L{l}|{m}"] = row
    # equal-resource pairs: same layer and mode, equal M (bytes) and B (BOPs) of the layer's FFN
    pairs = []
    for (l, m), ix in groups.items():
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if cands[i]["M_bytes"] != cands[j]["M_bytes"] or cands[i]["B_bops"] != cands[j]["B_bops"]:
                    continue
                d = Dm[i] - Dm[j]
                agree = float(((Db[i] - Db[j]) * np.sign(d) > 0).mean())
                pairs.append({"i": i, "j": j, "layer": l, "mode": m, "block": cands[i]["block"],
                              "class": _pair_class(cands[i], cands[j]), "truth_diff": float(d),
                              "decisive": bool(agree >= 0.95), "boot_agree": agree,
                              **{f"{k}_correct": float(0.5 if v[i] == v[j] else (np.sign(v[i] - v[j]) == np.sign(d)))
                                 for k, v in crit.items()}})
    if any(p["class"] == "single vs joint" for p in pairs):
        raise RuntimeError("unexpected equal-resource single/joint pair")
    P = np.array([[p["i"], p["j"]] for p in pairs])

    def acc_block(sel, label):
        if not sel:
            return
        sp = [pairs[k] for k in sel]
        Pi, Pj = P[sel, 0], P[sel, 1]
        out_ = {"pairs": len(sp), "decisive": int(sum(p["decisive"] for p in sp))}
        for k, v in crit.items():
            cs = np.sign(v[Pi] - v[Pj])
            ts = np.sign(Db[Pi] - Db[Pj])                            # [pairs, 2000]
            acc_b = np.where(cs[:, None] == 0, 0.5, (cs[:, None] == ts)).mean(0)
            out_[k] = {"accuracy": float(np.mean([p[f"{k}_correct"] for p in sp])), "ci95": ci(acc_b),
                       "decisive_correct": int(sum(p[f"{k}_correct"] for p in sp if p["decisive"]))}
        if len(crit) == 2:
            cs = {k: np.sign(v[Pi] - v[Pj]) for k, v in crit.items()}
            ts = np.sign(Db[Pi] - Db[Pj])
            diff = (np.where(cs["BC"][:, None] == 0, 0.5, cs["BC"][:, None] == ts)
                    - np.where(cs["SQNR"][:, None] == 0, 0.5, cs["SQNR"][:, None] == ts)).mean(0)
            out_["BC_minus_SQNR_ci95"] = ci(diff)
        res["pairs"][label] = out_
    allk = list(range(len(pairs)))
    acc_block(allk, "all")
    for key in ("mode", "block", "class", "layer"):
        for val in sorted({p[key] for p in pairs}, key=str):
            acc_block([k for k in allk if pairs[k][key] == val], f"{key}={val}")
    for m in ("w", "a", "wa"):
        for blk in ("linear_attention", "full_attention"):
            acc_block([k for k in allk if pairs[k]["mode"] == m and pairs[k]["block"] == blk], f"mode={m}|block={blk}")
    # loss gap of each criterion's pick to the best tested equal-resource candidate (classes with >= 2 members)
    classes = {}
    for i, c in enumerate(cands):
        classes.setdefault((c["layer"], c["mode"], c["M_bytes"], c["B_bops"]), []).append(i)
    classes = {k: v for k, v in classes.items() if len(v) >= 2}
    for (l, m) in groups:
        cls = [v for k, v in classes.items() if k[:2] == (l, m)]
        row = {"classes": len(cls), "best_mean_dnll": float(np.mean([Dm[v].min() for v in cls])),
               "random_pick_gap": float(np.mean([Dm[v].mean() - Dm[v].min() for v in cls]))}
        for k, v in crit.items():
            picks = [ix[int(np.argmin(v[ix]))] for ix in cls]
            gap = np.mean([Dm[p] - Dm[ix].min() for p, ix in zip(picks, cls)])
            gb = np.mean([Db[p] - Db[ix].min(0) for p, ix in zip(picks, cls)], 0)
            row[k] = {"gap": float(gap), "ci95": ci(gb), "picked_best": int(sum(Dm[p] == Dm[ix].min() for p, ix in zip(picks, cls)))}
        res["pick_gap"][f"L{l}|{m}"] = row
    # joint vs single: interaction I = dL(gate=fg, up=fu) - dL(gate=fg) - dL(up=fu), per window
    single = {(c["layer"], c["mode"], c["set"], tuple(c["formats"][c["set"]])): i for i, c in enumerate(cands)
              if c["set"] != "gate+up"}
    jrows = []
    for i, c in enumerate(cands):
        if c["set"] != "gate+up":
            continue
        g = single[(c["layer"], c["mode"], "gate_proj", tuple(c["formats"]["gate_proj"]))]
        u = single[(c["layer"], c["mode"], "up_proj", tuple(c["formats"]["up_proj"]))]
        I = D[i] - D[g] - D[u]
        Ib = I[boot].mean(1)
        jrows.append({"i": i, "g": g, "u": u, "layer": c["layer"], "mode": c["mode"], "I": float(I.mean()),
                      "I_ci95": ci(Ib), "rel": float(I.mean() / Dm[i]) if Dm[i] != 0 else None,
                      "I_half1": float(I[:nw // 2].mean()), "I_half2": float(I[nw // 2:].mean())})
    for (l, m) in groups:
        js = [r for r in jrows if (r["layer"], r["mode"]) == (l, m)]
        ji = [r["i"] for r in js]
        add = np.array([Dm[r["g"]] + Dm[r["u"]] for r in js])
        h1, h2 = np.array([r["I_half1"] for r in js]), np.array([r["I_half2"] for r in js])
        row = {"candidates": len(js), "rel_interaction_median": float(np.median([r["rel"] for r in js if r["rel"] is not None])),
               "ci_excludes_0_pos": int(sum(r["I_ci95"][0] > 0 for r in js)),
               "ci_excludes_0_neg": int(sum(r["I_ci95"][1] < 0 for r in js)),
               "split_half_corr": float(np.corrcoef(h1, h2)[0, 1]) if h1.std() > 0 and h2.std() > 0 else None,
               "split_half_sign_agree": float(np.mean(np.sign(h1) == np.sign(h2))),
               "spearman_additive_truth_vs_joint_truth": float(spearmanr(add, Dm[ji])[0])}
        # joint equal-resource pairs: accuracy of the additive oracle (true single dL summed) = ceiling of any
        # additive criterion
        jp = [p for p in pairs if (p["layer"], p["mode"]) == (l, m) and p["class"].startswith("joint")]
        if jp:
            pos = {r["i"]: k for k, r in enumerate(js)}
            ok = [np.sign(add[pos[p["i"]]] - add[pos[p["j"]]]) == np.sign(p["truth_diff"]) for p in jp]
            row["joint_pairs"] = len(jp)
            row["additive_oracle_accuracy"] = float(np.mean(ok))
            for k in crit:
                row[f"{k}_accuracy"] = float(np.mean([p[f"{k}_correct"] for p in jp]))
        res["joint"][f"L{l}|{m}"] = row
    # device cross-check (truth_device_plan.json): the CPU replay of a pre-declared subset against the GPU truth
    if (out / "truth_cpu_crosscheck.json").exists():
        cpu = jload(out / "truth_cpu_crosscheck.json")["rows"]
        sub = [i for i, n in enumerate(names) if n in cpu]
        Dc = np.array([cpu[names[i]] for i in sub]) / T
        Dg = D[sub]
        dm = Dg.mean(1) - Dc.mean(1)
        rel = np.abs(dm) / np.maximum(np.abs(Dc.mean(1)), 1e-12)
        pos = {i: k for k, i in enumerate(sub)}
        sp = [p for p in pairs if p["i"] in pos and p["j"] in pos]
        Dcb = np.stack([Dc[:, b].mean(1) for b in boot], 1)
        rows = []
        for p in sp:
            a, b = pos[p["i"]], pos[p["j"]]
            dc = Dc[a].mean() - Dc[b].mean()
            dec_c = float(((Dcb[a] - Dcb[b]) * np.sign(dc) > 0).mean()) >= 0.95
            rows.append({"same_order": bool(np.sign(dc) == np.sign(p["truth_diff"])), "gpu_decisive": p["decisive"],
                         "cpu_decisive": dec_c, "BC_correct_gpu": p["BC_correct"],
                         "BC_correct_cpu": float(np.sign(crit["BC"][p["i"]] - crit["BC"][p["j"]]) == np.sign(dc))})
        res["device_crosscheck"] = {
            "candidates": len(sub), "pairs": len(sp),
            "per_token_mean_abs_diff": {"median": float(np.median(np.abs(dm))), "max": float(np.abs(dm).max())},
            "relative_mean_diff": {"median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)), "max": float(rel.max())},
            "per_window_abs_diff_max": float(np.abs(Dg - Dc).max()),
            "pairs_same_order": int(sum(r["same_order"] for r in rows)),
            "gpu_decisive_pairs": int(sum(r["gpu_decisive"] for r in rows)),
            "gpu_decisive_and_same_order": int(sum(r["same_order"] and r["gpu_decisive"] for r in rows)),
            "cpu_decisive_pairs": int(sum(r["cpu_decisive"] for r in rows)),
            "BC_accuracy_gpu_truth": float(np.mean([r["BC_correct_gpu"] for r in rows])),
            "BC_accuracy_cpu_truth": float(np.mean([r["BC_correct_cpu"] for r in rows])),
            "worst": sorted([{"name": names[i], "gpu": float(Dg[k].mean()), "cpu": float(Dc[k].mean())}
                             for k, i in enumerate(sub)], key=lambda r: -abs(r["gpu"] - r["cpu"]))[:5]}
        log(f"device cross-check: {len(sub)} candidates, {len(sp)} pairs, same order "
            f"{res['device_crosscheck']['pairs_same_order']}, BC accuracy GPU/CPU truth "
            f"{res['device_crosscheck']['BC_accuracy_gpu_truth']:.3f}/{res['device_crosscheck']['BC_accuracy_cpu_truth']:.3f}")
    jdump(out / "analysis.json", res)
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "layer", "block", "mode", "set", "M_bytes", "B_bops", *crit, "dnll_per_token",
                    "ci_lo", "ci_hi"])
        for i, c in enumerate(cands):
            w.writerow([c["name"], c["layer"], c["block"], c["mode"], c["set"], c["M_bytes"], c["B_bops"],
                        *[float(v[i]) for v in crit.values()], Dm[i], *np.percentile(Db[i], [2.5, 97.5])])
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = ["layer", "mode", "block", "class", "truth_diff", "decisive", "boot_agree", *[f"{k}_correct" for k in crit]]
        w.writerow(["a", "b", *keys])
        for p in pairs:
            w.writerow([names[p["i"]], names[p["j"]], *[p[k] for k in keys]])
    with open(out / "joint.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["joint", "layer", "mode", "dnll_joint", "dnll_gate", "dnll_up", "I", "I_lo", "I_hi", "I_half1", "I_half2"])
        for r in jrows:
            w.writerow([names[r["i"]], r["layer"], r["mode"], Dm[r["i"]], Dm[r["g"]], Dm[r["u"]], r["I"], *r["I_ci95"],
                        r["I_half1"], r["I_half2"]])
    a = res["pairs"]["all"]
    log("pairs: " + ", ".join(f"{k} {a[k]['accuracy']:.3f} {a[k]['ci95']}" for k in crit) + f" ({a['pairs']} pairs, "
        f"{a['decisive']} decisive)")


# --------------------------------------------------------------------------- #
# Limited 512-token check
# --------------------------------------------------------------------------- #
def stage_seq512(args, out, log):
    """plan.json data.seq_512, parts (--parts): export (the adapter at 512 tokens, same mapping rule), forward (native
    FP32 on the 4 calibration_512 windows against PyTorch), table (native BC of the representative layers' FFN
    projections from those windows), stats (that table end to end against PyTorch), scores (the frozen candidates'
    BC-512 scores, frozen), truth (diagnostic_512 replay, the 512-token ranges; --layers batches)."""
    import shutil
    from qwen3_swiglu_common import sha256
    d = out / "seq512"
    d.mkdir(exist_ok=True)
    parts = args.parts.split(",")
    f = d / "seq512.json"
    res = jload(f) if f.exists() else {}
    plan = jload(out / "plan.json")
    layers = {l for v in plan["ranking"]["layers"].values() for l in v}
    if "export" in parts and "export" not in res:
        import onnx
        from qwen3_prepare import fold_constants
        t0 = time.time()
        hf = q35.load_hf(torch.float32)
        ad = q35.TextExport(hf, 512).eval()
        x = torch.from_numpy(np.load(out / "tokens_calibration_512.npy")[:1])
        tmp = d / "export_tmp"
        tmp.mkdir()
        torch.onnx.export(ad, (x,), str(tmp / "model_raw.onnx"), input_names=["input_ids"], output_names=["logits"],
                          opset_version=17, dynamo=False, do_constant_folding=True, external_data=True)
        m = onnx.load(str(tmp / "model_raw.onnx"))
        shutil.rmtree(tmp)
        fold_constants(m)
        mapping = q35.map_projection_nodes(m, hf, hf.model.language_model.config)
        onnx.save(m, str(d / "model.onnx"), save_as_external_data=True, location="model.onnx.data")
        jdump(d / "node_mapping.json", mapping)
        from collections import Counter
        ops = Counter(n.op_type for n in m.graph.node)
        ref_ops = jload(out / "prepare.json")["ops"]
        res["export"] = {"nodes": len(m.graph.node), "ops": dict(ops), "searched": len(mapping),
                         "same_op_counts_as_128": dict(ops) == ref_ops, "op_types_equal_to_128": set(ops) == set(ref_ops),
                         "model_onnx_sha256": sha256(d / "model.onnx"),
                         "seconds": time.time() - t0}
        jdump(f, res)
        log(f"export 512: {len(m.graph.node)} nodes, same op types as 128: {res['export']['op_types_equal_to_128']} "
            f"(counts differ: the delta rule unrolls 8 chunks instead of 2)")
    if "forward" in parts and "forward" not in res:
        tools, work, base = native_import(out, log, model="seq512/model.onnx", sub="seq512/native")
        res["forward"] = forward_check(out, tools, work, base, "calibration_512", 4, log)
        jdump(f, res)
    m512 = jload(d / "node_mapping.json")
    sel = {n: v for n, v in m512.items() if v["kind"] in FFN and v["layer"] in layers}
    if "table" in parts and "table" not in res:
        from run_native_bc_pipeline import erm, native_tables, qwen35_spec
        work = d / "bc"
        work.mkdir(exist_ok=True)
        spec = qwen35_spec(work, out, select=lambda info: info["kind"] in FFN and info["layer"] in layers,
                           model="seq512/model.onnx", mapping="seq512/node_mapping.json",
                           calib="tokens_calibration_512.npy", test="tokens_diagnostic_512.npy")
        # the grads build at Krnl -O3 ran out of memory at 512 tokens (onnx-mlir-opt 33 GB RSS in Krnl -> LLVM on
        # this 39 GB machine, seq512.json table_attempts); -O1 as the CNN statistics builds: a lowering level only
        import dataclasses
        levels = {**spec.stats_levels, "grads": tuple(args.grads_levels.split(","))}
        spec = dataclasses.replace(spec, stats_levels=levels)
        t0 = time.time()
        B, C, _, timing = native_tables(spec, erm.Tools(), work, log, input_grad="total")
        jdump(d / "table.json", {"B": B, "C": C, "naming": "B = W, C = A; 512-token calibration windows"})
        shutil.copy(work / "ranges_by_point.json", d / "ranges_by_point.json")
        res["table"] = {"nodes": len(B), "timing": timing, "seconds": time.time() - t0, "stats_levels": levels}
        jdump(f, res)
        log(f"table 512: {len(B)} nodes in {timing['criterion_total_s']:.0f}s")
    if "torch_table" in parts and not (d / "table.json").exists():
        # fallback (seq512.json table_attempts): the same statistics and table entries from PyTorch (torch_bc_table,
        # checked against the native 128-token table in seq512/torch_table_check_128.json), ranges measured on the
        # same 512-token calibration windows
        t0 = time.time()
        ref = torch_bc_table(out, sel, np.load(out / "tokens_calibration_512.npy"))
        jdump(d / "table.json", {"B": ref["B"], "C": ref["C"], "naming": "B = W, C = A; 512-token calibration windows",
                                 "source": "PyTorch torch_bc_table (FP32 adapter, autograd), not native: the native "
                                           "512-token statistics builds ran out of memory"})
        jdump(d / "ranges_by_point.json", ref["ranges"])
        res["torch_table"] = {"nodes": len(sel), "seconds": time.time() - t0}
        jdump(f, res)
        log(f"512-token table (PyTorch): {len(sel)} nodes in {time.time() - t0:.0f}s")
    if "stats" in parts and "stats" not in res:
        res["stats"] = stats_check(out, log, nodes=sel, toks="tokens_calibration_512.npy", table="seq512/table.json",
                                   ranges="seq512/ranges_by_point.json", werr="seq512/bc/weight_err.json")
        jdump(f, res)
    if "scores" in parts and not (d / "scores_freeze.json").exists():
        t512 = jload(d / "table.json")
        by = {(v["layer"], v["kind"]): n for n, v in m512.items()}
        cands = jload(out / "candidates.json")
        for c in cands:
            c["nodes"] = {k: by[(c["layer"], k)] for k in c["formats"]}
            c["bc512"] = float(sum(t512["B"][c["nodes"][k]][w] + t512["C"][c["nodes"][k]][a]
                                   for k, (w, a) in c["formats"].items()))
        jdump(d / "candidates_512.json", cands)
        jdump(d / "scores_freeze.json", {"candidates_512_sha256": sha256(d / "candidates_512.json"),
                                         "table_sha256": sha256(d / "table.json"),
                                         "ranges_sha256": sha256(d / "ranges_by_point.json"),
                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log(f"scores 512: {len(cands)} candidates frozen")
    if "truth" in parts:
        fz = jload(d / "scores_freeze.json")
        if sha256(d / "candidates_512.json") != fz["candidates_512_sha256"]:
            raise SystemExit("candidates_512.json changed after the freeze")
        cands = [c for c in jload(d / "candidates_512.json") if args.layers is None or c["layer"] in args.layers]
        replay_truth(out, cands, np.load(out / "tokens_diagnostic_512.npy"), jload(d / "ranges_by_point.json"),
                     d / "truth.json", args.batch, log, mapping="seq512/node_mapping.json", device=args.device)
    if "analyze" in parts:
        from scipy.stats import spearmanr
        cands = jload(d / "candidates_512.json")
        t512, t128 = jload(d / "truth.json")["rows"], jload(out / "truth.json")["rows"]
        cands = [c for c in cands if c["name"] in t512]
        D512 = np.array([t512[c["name"]] for c in cands]) / 511
        D128 = np.array([t128[c["name"]] for c in cands]) / (SEQ - 1)
        m512, m128 = D512.mean(1), D128.mean(1)
        boot = np.random.default_rng(plan["seeds"]["bootstrap"]).integers(0, D512.shape[1], (2000, D512.shape[1]))
        Db = np.stack([D512[:, b].mean(1) for b in boot], 1)
        groups = {}
        for i, c in enumerate(cands):
            groups.setdefault((c["layer"], c["mode"]), []).append(i)
        res_a = {"candidates": len(cands), "windows_512": int(D512.shape[1]), "spearman": {}, "pairs": {}}
        crit = {"BC512": np.array([c["bc512"] for c in cands]), "BC128": np.array([c["bc"] for c in cands])}
        for (l, m), ix in sorted(groups.items()):
            res_a["spearman"][f"L{l}|{m}"] = {
                **{k: float(spearmanr(v[ix], m512[ix])[0]) for k, v in crit.items()},
                "BC512_vs_BC128": float(spearmanr(crit["BC512"][ix], crit["BC128"][ix])[0]),
                "truth512_vs_truth128": float(spearmanr(m512[ix], m128[ix])[0]),
                "per_token_ratio_512_over_128_median": float(np.median(m512[ix] / m128[ix]))}
        pairs = [(i, j, cands[i]["mode"]) for ix in groups.values() for a, i in enumerate(ix) for j in ix[a + 1:]
                 if cands[i]["M_bytes"] == cands[j]["M_bytes"] and cands[i]["B_bops"] == cands[j]["B_bops"]]
        for mode in ("all", "w", "a", "wa"):
            sp = [(i, j) for i, j, m in pairs if mode in ("all", m)]
            if not sp:
                continue
            I, J = np.array(sp).T
            dec = (np.sign(Db[I] - Db[J]) == np.sign(m512[I] - m512[J])[:, None]).mean(1) >= 0.95
            row = {"pairs": len(sp), "decisive": int(dec.sum())}
            for k, v in crit.items():
                ok = np.sign(v[I] - v[J]) == np.sign(m512[I] - m512[J])
                row[k] = {"accuracy": float(ok.mean()), "decisive_correct": int(ok[dec].sum())}
            ok = np.sign(m128[I] - m128[J]) == np.sign(m512[I] - m512[J])
            row["truth128_as_predictor"] = {"accuracy": float(ok.mean()), "decisive_correct": int(ok[dec].sum())}
            res_a["pairs"][mode] = row
        jdump(d / "analysis.json", res_a)
        log("seq512 pairs: " + json.dumps(res_a["pairs"].get("all")))


# --------------------------------------------------------------------------- #
# Search and evaluation
# --------------------------------------------------------------------------- #
VERSIONS = {"ffn": lambda info: info["kind"] in FFN, "ffn_attn": lambda info: True}


def stage_search(args, out, log):
    """mp_auto.auto_select (run_native_auto.py's 2-D default) on the native table, per version of the searched
    set; non-finite table entries forbid their option."""
    from mp_auto import auto_select, config_hash
    from run_native_bc_pipeline import GRID11, qwen35_spec
    table = jload(out / "table.json")
    mapping = jload(out / "node_mapping.json")
    res = {}
    for ver, sel in VERSIONS.items():
        spec = qwen35_spec(out / "bc", out, select=sel)
        log_ = []
        t0 = time.perf_counter()
        selection = auto_select(spec.nodes, GRID11, GRID11, lambda n, w, a: table["B"][n][w] + table["C"][n][a],
                                spec.weight_elems, spec.flops, samples=17, refinements=2, time_limit=10, solver_log=log_)
        search_s = time.perf_counter() - t0
        chosen = selection["selected"]
        fmts = {n: chosen["formats"][n] for n in spec.names}
        jdump(out / f"selection_{ver}.json", dict(selection, solver_log=log_, search_s=search_s, names=spec.names))
        jdump(out / f"selected_formats_{ver}.json", fmts)
        statuses = [r.get("status") for r in log_ if "status" in r]
        res[ver] = {"nodes": len(spec.names), "search_s": search_s, "milp_solves": len(log_),
                    "status_counts": {str(k): statuses.count(k) for k in sorted(set(statuses))},
                    "failures": selection["failures"], "excluded_nonfinite_options": selection["excluded_nonfinite_options"],
                    "knee": {k: chosen[k] for k in ("size", "bops", "resource", "objective", "solver_status", "origin",
                                                    "search_budget")},
                    "config_hash": config_hash(fmts),
                    "format_counts": {"w": dict(__import__("collections").Counter(f["w"] for f in fmts.values())),
                                      "a": dict(__import__("collections").Counter(f["a"] for f in fmts.values()))},
                    "by_kind": {k: dict(__import__("collections").Counter(f"{fmts[n]['w']}/{fmts[n]['a']}" for n in spec.names
                                                                         if mapping[n]["kind"] == k))
                                for k in sorted({mapping[n]["kind"] for n in spec.names})}}
        log(f"{ver}: {len(spec.names)} nodes, knee size {chosen['size']:.3f} BOPs {chosen['bops']:.3f}, "
            f"{len(log_)} MILP solves {res[ver]['status_counts']}, {search_s:.1f}s")
    # what stays FP32 in every version: the text backbone's parameters outside the searched projections
    from safetensors import safe_open
    searched = {v["param"] for v in mapping.values()}
    fixed, total = {}, 0
    for f in sorted(q35.MODEL_DIR.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                if not k.startswith("model.language_model.") or ".mtp" in k:
                    continue
                n = int(np.prod(st.get_slice(k).get_shape()))
                total += n
                if k not in searched:
                    group = k.split(".")[-2] if ".layers." in k else k.removeprefix("model.language_model.")
                    fixed[group] = fixed.get(group, 0) + n
    res["fp32_fixed"] = {"text_backbone_parameters": total,
                         "searched_ffn_attn_parameters": total - sum(fixed.values()),
                         "fixed_parameters_by_group": fixed,
                         "also_fp32": "LM head (tied to embed_tokens), every non-projection op (norms, conv1d, recurrence, "
                                      "softmax, rope, gates)"}
    jdump(out / "selection.json", res)


def stage_evaluate(args, out, log):
    """The frozen selections and FP32, compiled by onnx-mlir as run_native_auto.py does (per-channel int weights,
    Krnl -O3 / clang -O0, constants to file, the calibration ranges), on the final-test windows."""
    import os
    import onnx
    from mp_auto import config_hash
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm, qwen35_spec
    from evaluate_native_search_chunked import chunked_run_so
    from qwen3_swiglu_common import sha256
    tools, work, base = native_import(out, log)
    toks = np.load(out / "tokens_final_test.npy")
    graph = onnx.load(str(out / "model.onnx"), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in jload(out / "ranges_by_point.json").items():
        for slot, r in slots.items():
            n = by_name[node]
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    spec = qwen35_spec(out / "bc", out)
    ev_dir = out / "eval"
    ev_dir.mkdir(exist_ok=True)
    np.save(ev_dir / "final_test.npy", toks)
    posit_obj = erm.build_posit_runtime(tools, ev_dir)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    results = jload(out / "results.json") if (out / "results.json").exists() else {}
    for ver in ["fp32", *VERSIONS]:
        if ver in results or (args.versions and ver not in args.versions):
            continue
        t0 = time.time()
        target = ev_dir / ver
        target.mkdir(exist_ok=True)
        env = None
        if ver != "fp32":
            fmts = jload(out / f"selected_formats_{ver}.json")
            cfg = {n["name"]: fmts.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes}
            env = erm.node_formats_env(cfg, spec.nodes, ranges)
            (target / "compile_env.json").write_text(json.dumps({"MIXEDP_NODE_FORMATS": env, "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                                                                 "config_hash": config_hash(fmts)}, indent=1) + "\n")
        so, info = erm.build_so(tools, target, ver, base, env, posit_obj if env else None, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True,
                                uses_posit=None if env else False, uses_f16=None if env else False)
        build_s = time.time() - t0
        if info["warnings"]:
            raise RuntimeError(f"{ver}: compilation warnings {info['warnings']}")
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, ev = chunked_run_so(args.chunks)(tools, so, ev_dir / "final_test.npy", target / "logits.npy", 36000)
        (target / "logits.npy").unlink()
        finite = bool(np.isfinite(z).all())
        nll = np.stack([cross_entropy(z[i, :-1], toks[i, 1:]) for i in range(len(toks))])
        results[ver] = {"ppl": float(np.exp(nll.mean())), "nll": float(nll.mean()), "per_window_nll": nll.mean(1).tolist(),
                        "all_logits_finite": finite, "build_s": build_s, "run_wall_s": ev["wall_seconds"],
                        "run_peak_rss_gb": ev["peak_rss_gb"], "so_sha256": sha256(so)}
        for p in (so, target / "model.constants.bin"):
            p.unlink(missing_ok=True)
        jdump(out / "results.json", results)
        log(f"{ver}: PPL {results[ver]['ppl']:.3f}, NLL {results[ver]['nll']:.4f}, finite {finite} "
            f"(build {build_s:.0f}s, run {ev['wall_seconds']:.0f}s)")
    if not all(v in results for v in ["fp32", *VERSIONS]):
        return
    fp = np.array(results["fp32"]["per_window_nll"])
    for ver in VERSIONS:
        d = np.array(results[ver]["per_window_nll"]) - fp
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        results[ver]["delta_nll_vs_fp32"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                             "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    jdump(out / "results.json", results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("prepare", "correctness", "tables", "freeze", "sqnr", "truth",
                                                       "analyze", "search", "evaluate", "seq512"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--parts", default="forward", help="correctness / seq512: comma list of parts")
    ap.add_argument("--batch", type=int, default=10, help="truth: windows per forward")
    ap.add_argument("--grads-levels", default="O3,O0", help="seq512 table: (Krnl -O, clang -O) of the grads build")
    ap.add_argument("--device", default="cpu", help="truth replay: cpu or cuda (FP32, TF32 off)")
    ap.add_argument("--truth-file", default="truth.json", help="truth: output file (e.g. a CPU cross-check)")
    ap.add_argument("--names", type=lambda v: v.split(";"), default=None, help="truth: only these candidates")
    ap.add_argument("--layers", type=lambda v: [int(x) for x in v.split(",")], default=None,
                    help="sqnr / truth / seq512 truth: only these layers (a resumable batch)")
    ap.add_argument("--chunks", type=int, default=5, help="evaluate: processes per compiled run")
    ap.add_argument("--versions", type=lambda v: v.split(","), default=None, help="evaluate: only these (fp32, ffn, ffn_attn)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    if args.stage == "prepare":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "plan.json").exists():
        raise SystemExit("run --stage prepare first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    import resource
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {}
    timing[args.stage] = timing.get(args.stage, 0.0) + time.time() - t0
    runs = timing.setdefault("runs", [])          # every invocation: wall time and peak RSS (this process; the
    runs.append({"stage": args.stage, "parts": args.parts, "layers": args.layers,     # largest child process)
                 "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "wall_s": time.time() - t0,
                 "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                 "peak_rss_gb_largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20})
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
