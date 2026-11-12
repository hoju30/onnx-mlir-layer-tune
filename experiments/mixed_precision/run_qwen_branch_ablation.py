#!/usr/bin/env python3
"""Qwen3-0.6B: with the pass's per-consumer QDQ, does the per-consumer (branch) input sensitivity pick whole-model
configurations with lower quantization damage than the total sensitivity of the shared tensor, at the same resources?

Cost tables (reused, not rebuilt): runs/native_eps_fix_20261014/qwen3/{table,table_branch}.json, one native grads
build of the epsilon-fixed model (RMSNorm eps 1e-6); B = weight criterion W (identical in both), C = activation
criterion A with the input term from the shared tensor's total gradient (table.json) or from each consumer's own
input-edge gradient (table_branch.json). Activation cost variants, all with the same B:
  total             C of table.json everywhere
  ffn_branch        branch rows for gate_proj / up_proj only
  attention_branch  branch rows for q_proj / k_proj / v_proj only
  all_branch        C of table_branch.json (o_proj / down_proj rows are identical in both tables: single consumer)
Search as run_native_auto.py --policy 2d: mp_auto.auto_select on GRID11 x GRID11, value B[w] + C[a], resource
max(size, BOPs) relative to all-int8, normalized chord knee, samples 17, refinements 2, MILP time limit 10 s.

Stages (all in --out-dir, each reads what the earlier ones wrote):
  check     asset consistency (tables, node set, formats, ranges, calibration tokens, model epsilon, compiler build
            order), the tuning windows (validation-split windows after the calibration ones) and PyTorch FP32
            per-window losses of the calibration + tuning windows
  selftest  the variant tables replace only the intended rows; a toy search follows the table it is given
  search    automatic knee per variant; matched-resource controls: MILP with total and with all_branch costs under
            the exact (weight bytes, BOPs) of the total knee and of the all_branch knee
  eval      each distinct configuration (and FP32) compiled as run_native_auto.py does (per-channel int weights,
            Krnl -O3 / clang -O0, constants to file) with the calibration ranges of the tables, run on the
            calibration and tuning windows: summed next-token CE per window
  analyze   paired per-window comparisons, node differences, CSV / JSON
  decide    freezes the decision text and the held-out configurations (needs analysis.json)
  heldout   only after decide: the frozen total / all_branch knees and FP32 on the test windows (report only)
  tables    the report's Markdown tables (tables.md)
Data roles: calibration windows 0-9 of the validation split built the tables and the int ranges; tuning windows
10-29 of the same split are used for no table, range or search; the test split is held out until `decide`.
"""
import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

from htq_pareto_search import FORMAT_BITS  # noqa: E402
from mp_auto import auto_select, config_hash  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from qwen3_swiglu_common import PREP, environment, node_mapping, sha256  # noqa: E402
from run_native_bc_pipeline import GRID11, SLOT_INPUT, cross_entropy, erm, imported_model  # noqa: E402

TABLES = HERE / "runs/native_eps_fix_20261014/qwen3"
SEARCH = dict(samples=17, refinements=2, time_limit=10.0)   # run_native_auto.py defaults (--policy 2d)
VARIANTS = {"total": (), "all_branch": ("q", "k", "v", "gate", "up"),
            "ffn_branch": ("gate", "up"), "attention_branch": ("q", "k", "v")}
CAL_WINDOWS, TUNE_WINDOWS, SEQ = 10, 20, 128
KEEP = ("fp32", "knee_total", "knee_all_branch")          # builds kept for the held-out stage
EXTRA_FILES = ["experiments/mixed_precision/run_qwen_branch_ablation.py", "experiments/mixed_precision/mp_auto.py",
               "experiments/mixed_precision/mp_milp.py", "src/Dialect/ONNX/Transforms/Recompose.cpp",
               "experiments/mixed_precision/evaluate_native_search_chunked.py"]


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Problem: nodes, resources, cost tables
# --------------------------------------------------------------------------- #
def problem():
    """The 196 projections as qwen3_spec defines them (nodes, weight elements, FLOP proxy = elements x sequence
    length), read from the ONNX graph without its 3 GB of weights; `check` asserts equality with qwen3_spec."""
    import onnx
    graph = onnx.load(str(PREP / "model.onnx"), load_external_data=False)
    mapping = json.loads((PREP / "node_mapping.json").read_text())
    nodes = [n for n in erm.supported_nodes(graph) if n["name"] in mapping]
    names = [n["name"] for n in nodes]
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    elems = {n: int(np.prod(inits[by_name[n].input[1]].dims)) for n in names}
    flops = {n: elems[n] * SEQ for n in names}
    kinds = {n: v[1] for n, v in node_mapping().items()}
    return dict(graph=graph, nodes=nodes, names=names, elems=elems, flops=flops, kinds=kinds, by_name=by_name)


def load_tables(tables=TABLES):
    t, b = jload(tables / "table.json"), jload(tables / "table_branch.json")
    return t["B"], t["C"], b["B"], b["C"]


def variant_tables(C_total, C_branch, kinds):
    """Activation cost table per variant: the branch row for the node kinds the variant lists, else the total row.
    all_branch is table_branch.json itself; `selftest` checks it equals the row-wise construction."""
    out = {v: {n: dict(C_branch[n] if kinds[n] in sel else C_total[n]) for n in C_total}
           for v, sel in VARIANTS.items() if v != "all_branch"}
    out["all_branch"] = {n: dict(C_branch[n]) for n in C_branch}
    return out


def resources(cfg, names, elems, flops):
    """Exact weight bytes and BOPs, summed as milp_select sums them (node order; integer-valued, exact in float64)."""
    size = np.array([elems[n] * FORMAT_BITS[cfg[n]["w"]] / 8 for n in names]).sum()
    bops = np.array([flops[n] * FORMAT_BITS[cfg[n]["w"]] * FORMAT_BITS[cfg[n]["a"]] for n in names]).sum()
    return float(size), float(bops)


def objective(cfg, names, B, C):
    return float(sum(B[n][cfg[n]["w"]] + C[n][cfg[n]["a"]] for n in names))


def check_config(cfg, names, B, C):
    """Every searched node has a grid format pair with a finite cost; nothing else is assigned a weight format."""
    missing = [n for n in names if n not in cfg]
    bad = [n for n in names if n in cfg and (cfg[n]["w"] not in GRID11 or cfg[n]["a"] not in GRID11)]
    nonfinite = [n for n in names if n in cfg and not bad and not np.isfinite(B[n][cfg[n]["w"]] + C[n][cfg[n]["a"]])]
    extra = [n for n, f in cfg.items() if n not in names and f["w"] is not None]
    if missing or bad or nonfinite or extra:
        raise RuntimeError(f"bad configuration: missing {missing[:3]}, bad formats {bad[:3]}, "
                           f"non-finite {nonfinite[:3]}, extra {extra[:3]}")


# --------------------------------------------------------------------------- #
# Stage check
# --------------------------------------------------------------------------- #
def tokenize_validation(windows):
    """The validation split tokenized exactly as qwen3_prepare.py does, as `windows` 128-token windows."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE / "qwen3_assets" / "model")
    text = "\n\n".join(pq.read_table(HERE / "gpt2_assets/data" / "validation-00000-of-00001.parquet")["text"]
                       .to_pylist())
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) >= windows * SEQ, (len(ids), windows)
    return np.asarray(ids[:windows * SEQ], np.int64).reshape(windows, SEQ), len(ids)


def model_epsilons(graph):
    """Scalar float constants feeding Add nodes whose other input is a ReduceMean (the RMSNorm variance + eps)."""
    from onnx import numpy_helper
    consts = {i.name: numpy_helper.to_array(i) for i in graph.graph.initializer if not i.external_data}
    producers = {o: n for n in graph.graph.node for o in n.output}
    for n in graph.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    eps = []
    for n in graph.graph.node:
        if n.op_type != "Add" or len(n.input) != 2:
            continue
        for k in (0, 1):
            c, other = n.input[k], n.input[1 - k]
            p = producers.get(other)
            if c in consts and consts[c].size == 1 and p is not None and p.op_type == "ReduceMean":
                eps.append(float(consts[c].reshape(())))
    return eps


def stage_check(args, out, log):
    from run_native_bc_pipeline import qwen3_spec
    P = problem()
    names, kinds = P["names"], P["kinds"]
    checks = {}
    B, C, Bb, Cb = load_tables()
    checks["nodes"] = {"searched": len(names), "table_B": len(B), "table_C": len(C), "branch_B": len(Bb),
                       "branch_C": len(Cb), "same_set": set(names) == set(B) == set(C) == set(Bb) == set(Cb),
                       "by_kind": {k: sum(v == k for v in kinds.values()) for k in sorted(set(kinds.values()))}}
    checks["formats"] = {"grid11": list(GRID11), "all_rows_grid11": all(
        set(t[n]) == set(GRID11) for t in (B, C, Bb, Cb) for n in t)}
    vals = np.array([v for t in (B, C, Bb, Cb) for row in t.values() for v in row.values()])
    checks["values"] = {"entries": int(vals.size), "nonfinite": int((~np.isfinite(vals)).sum()),
                        "negative": int((vals < 0).sum()),
                        "fp32_zero": all(t[n]["fp32"] == 0.0 for t in (B, C, Bb, Cb) for n in t)}
    checks["weight_criterion_identical"] = B == Bb
    differs = {n for n in names if C[n] != Cb[n]}
    checks["activation_rows_differing"] = {
        "nodes": len(differs), "by_kind": {k: sum(kinds[n] == k for n in differs) for k in sorted(set(kinds.values()))},
        "only_shared_input_kinds": all(kinds[n] in VARIANTS["all_branch"] for n in differs),
        "all_shared_input_nodes_differ": all(n in differs for n in names if kinds[n] in VARIANTS["all_branch"]),
        "o_down_identical": all(C[n] == Cb[n] for n in names if kinds[n] in ("o", "down"))}
    # ranges: one (min, max) per tensor, covering every activation tensor of every searched node
    rbp = jload(TABLES / "ranges_by_point.json")
    ranges = tensor_ranges(P, rbp)
    act = {t for n in P["nodes"] for t in n["act_tensors"]}
    checks["ranges"] = {"points": sum(len(v) for v in rbp.values()), "tensors": len(ranges),
                        "searched_nodes_covered": all(n in rbp for n in names),
                        "activation_tensors_missing": sorted(act - set(ranges))[:5],
                        "finite_ordered": all(np.isfinite(lo) and np.isfinite(hi) and lo <= hi
                                              for lo, hi in ranges.values())}
    # calibration tokens: the tables' calibration inputs are the first 10 validation windows
    val = np.load(PREP / "validation_tokens.npy")
    cal = np.load(TABLES / "calib_inputs.npy")
    toks, total_ids = tokenize_validation(CAL_WINDOWS + TUNE_WINDOWS)
    checks["tokens"] = {"tables_calib_equals_validation_tokens": bool(np.array_equal(cal.reshape(val.shape), val)),
                        "retokenized_first_10_equal": bool(np.array_equal(toks[:CAL_WINDOWS], val)),
                        "validation_split_tokens": total_ids,
                        "tuning_windows": [CAL_WINDOWS, CAL_WINDOWS + TUNE_WINDOWS - 1],
                        "tuning_overlaps_calibration": False}
    if not checks["tokens"]["retokenized_first_10_equal"]:
        raise RuntimeError("re-tokenized validation windows differ from validation_tokens.npy")
    np.save(out / "eval_tokens.npy", toks)                 # 0-9 calibration, 10-29 tuning
    eps = model_epsilons(P["graph"])
    checks["model_epsilon"] = {"rmsnorm_adds": len(eps), "values": sorted(set(eps)),
                               "all_1e-6": all(abs(e - 1e-6) <= 1e-12 for e in eps)}
    # compiler: built after every source change, before the tables; the epsilon fix in the source
    tools = erm.Tools()
    root = HERE.parent.parent
    src = {f: (root / f).stat().st_mtime for f in (
        "src/Dialect/ONNX/Transforms/Recompose.cpp", "src/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.cpp",
        "src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.cpp",
        "src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.hpp")}
    opt_mtime = Path(tools.opt).stat().st_mtime
    table_mtime = min((TABLES / f).stat().st_mtime for f in ("table.json", "table_branch.json",
                                                             "ranges_by_point.json"))
    checks["compiler"] = {"onnx_mlir_opt_sha256": sha256(tools.opt), "onnx_mlir_opt_mtime": opt_mtime,
                          "built_after_sources": all(opt_mtime > m for m in src.values()),
                          "tables_after_build": table_mtime > opt_mtime,
                          "epsilon_fix_in_source": "getScalarValue<double>(epsilonOp)" in
                          (root / "src/Dialect/ONNX/Transforms/Recompose.cpp").read_text()}
    # the light problem() equals qwen3_spec's node set and resource proxies
    spec = qwen3_spec(out)
    checks["spec_equal"] = {"names": spec.names == names, "elems": spec.weight_elems == P["elems"],
                            "flops": spec.flops == P["flops"], "nodes": spec.nodes == P["nodes"]}
    del spec
    assets = {"tables_dir": str(TABLES), **{f: sha256(TABLES / f) for f in (
        "table.json", "table_branch.json", "ranges_by_point.json", "calib_inputs.npy", "criterion.json")},
        "model.mlirbc": sha256(TABLES / "model.mlirbc"), "node_mapping.json": sha256(PREP / "node_mapping.json"),
        "test_tokens.npy": sha256(PREP / "test_tokens.npy"), "eval_tokens.npy": sha256(out / "eval_tokens.npy")}
    ok = (checks["nodes"]["same_set"] and checks["formats"]["all_rows_grid11"] and not checks["values"]["nonfinite"]
          and not checks["values"]["negative"] and checks["weight_criterion_identical"]
          and checks["activation_rows_differing"]["only_shared_input_kinds"]
          and checks["activation_rows_differing"]["o_down_identical"] and checks["ranges"]["searched_nodes_covered"]
          and not checks["ranges"]["activation_tensors_missing"] and checks["ranges"]["finite_ordered"]
          and checks["tokens"]["tables_calib_equals_validation_tokens"] and checks["model_epsilon"]["all_1e-6"]
          and checks["compiler"]["built_after_sources"] and checks["compiler"]["tables_after_build"]
          and checks["compiler"]["epsilon_fix_in_source"] and all(checks["spec_equal"].values()))
    checks["all_passed"] = bool(ok)
    jdump(out / "assets.json", {"assets": assets, "checks": checks})
    log(f"asset checks: {'passed' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError(json.dumps(checks, indent=1))
    # PyTorch FP32 per-window losses: the reference the native FP32 build is compared with (epsilon in effect)
    import torch
    from qwen3_swiglu_common import load_model, summed_ce
    t0 = time.time()
    _, wrapper = load_model(torch.float32)
    losses = []
    with torch.no_grad():
        for ids in toks:
            losses.append(float(summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))))
    jdump(out / "torch_fp32.json", {"summed_ce": losses, "seconds": time.time() - t0})
    log(f"PyTorch FP32: calibration summed CE {sum(losses[:CAL_WINDOWS]):.4f}, tuning "
        f"{sum(losses[CAL_WINDOWS:]):.4f} ({time.time() - t0:.0f}s)")


def tensor_ranges(P, rbp):
    ranges = {}
    for node, slots in rbp.items():
        n = P["by_name"][node]
        for slot, r in slots.items():
            t = n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]
            assert ranges.setdefault(t, tuple(r)) == tuple(r), (t, ranges[t], r)
    return ranges


# --------------------------------------------------------------------------- #
# Stage selftest
# --------------------------------------------------------------------------- #
def run_auto(nodes, names, B, C, elems, flops, **search):
    """auto_select with value B[w] + C[a] of the given tables (run_native_auto.py's value), timed, solver log."""
    log_ = []
    t0 = time.perf_counter()
    sel = auto_select(nodes, GRID11, GRID11, lambda n, w, a: B[n][w] + C[n][a], elems, flops,
                      solver_log=log_, **search)
    return sel, log_, time.perf_counter() - t0


def stage_selftest(args, out, log):
    P = problem()
    names, kinds = P["names"], P["kinds"]
    B, C, Bb, Cb = load_tables()
    V = variant_tables(C, Cb, kinds)
    res = {}
    for v, sel in VARIANTS.items():               # rows: branch exactly where listed, total elsewhere
        wrong = [n for n in names if V[v][n] != (Cb[n] if kinds[n] in sel else C[n])]
        res[f"rows_{v}"] = {"wrong_rows": len(wrong), "branch_rows": sum(V[v][n] == Cb[n] != C[n] for n in names)}
    res["all_branch_is_table_branch"] = V["all_branch"] == Cb
    res["union_of_ffn_attention_is_all"] = all(
        V["all_branch"][n] == (V["ffn_branch"][n] if kinds[n] in ("gate", "up") else V["attention_branch"][n])
        for n in names)
    # toy search: three nodes; the branch table makes node b's low-bit activations expensive, so a search reading
    # the branch table must move bits to b's activation, one reading the total table must not
    toy = [{"name": x, "weight_const": True} for x in "abc"]
    te = {x: 1000 for x in "abc"}
    tf = {x: 128000 for x in "abc"}
    tb = {x: {f: (0.0 if f == "fp32" else 2.0 ** -FORMAT_BITS[f]) for f in GRID11} for x in "abc"}
    tc = {x: {f: (0.0 if f == "fp32" else 4.0 ** -FORMAT_BITS[f]) for f in GRID11} for x in "abc"}
    tc_branch = {x: dict(tc[x]) for x in "abc"}
    tc_branch["b"] = {f: (0.0 if f == "fp32" else 1e3 * 4.0 ** -FORMAT_BITS[f]) for f in GRID11}
    st, _, _ = run_auto(toy, list("abc"), tb, tc, te, tf, samples=5, refinements=1, time_limit=10)
    sb, _, _ = run_auto(toy, list("abc"), tb, tc_branch, te, tf, samples=5, refinements=1, time_limit=10)
    bits = lambda s, x: FORMAT_BITS[s["selected"]["formats"][x]["a"]]
    res["toy"] = {"total_b_act_bits": bits(st, "b"), "branch_b_act_bits": bits(sb, "b"),
                  "branch_reads_its_table": bits(sb, "b") > bits(st, "b"),
                  "objective_total_recomputed": abs(objective(st["selected"]["formats"], "abc", tb, tc)
                                                    - st["selected"]["objective"]) < 1e-12,
                  "objective_branch_recomputed": abs(objective(sb["selected"]["formats"], "abc", tb, tc_branch)
                                                     - sb["selected"]["objective"]) < 1e-12}
    ok = (all(r["wrong_rows"] == 0 for k, r in res.items() if k.startswith("rows_"))
          and res["all_branch_is_table_branch"] and res["union_of_ffn_attention_is_all"]
          and res["toy"]["branch_reads_its_table"] and res["toy"]["objective_total_recomputed"]
          and res["toy"]["objective_branch_recomputed"])
    res["all_passed"] = bool(ok)
    jdump(out / "selftest.json", res)
    log(f"selftest: {'passed' if ok else 'FAILED'} {json.dumps(res)}")
    if not ok:
        raise RuntimeError("selftest failed")


# --------------------------------------------------------------------------- #
# Stage search
# --------------------------------------------------------------------------- #
def stage_search(args, out, log):
    P = problem()
    nodes, names, kinds, elems, flops = P["nodes"], P["names"], P["kinds"], P["elems"], P["flops"]
    B, C, _, Cb = load_tables()
    V = variant_tables(C, Cb, kinds)
    shas = {"table.json": sha256(TABLES / "table.json"), "table_branch.json": sha256(TABLES / "table_branch.json")}
    configs, summary = {}, {"search_settings": dict(SEARCH, grid=list(GRID11), policy="2d normalized chord knee",
                                                     value="B[w] + C_variant[a]"), "tables": shas, "variants": {}}
    for v in VARIANTS:
        sel, slog, secs = run_auto(nodes, names, B, V[v], elems, flops, **SEARCH)
        for row in sel["candidates"]:                      # the search's numbers come from this variant's table
            check_config(row["formats"], names, B, V[v])
            if abs(objective(row["formats"], names, B, V[v]) - row["objective"]) > 1e-9 * max(1, row["objective"]):
                raise RuntimeError(f"{v}: candidate objective differs from its table")
        k = sel["selected"]
        size, bops = resources(k["formats"], names, elems, flops)
        statuses = [r.get("status") for r in slog if "status" in r]
        gaps = [r["mip_gap"] for r in slog if r.get("mip_gap") is not None]
        info = {"rows_from_branch": list(VARIANTS[v]), "search_s": secs, "milp_solves": len(slog),
                "status_counts": {str(s): statuses.count(s) for s in sorted(set(statuses))},
                "max_mip_gap": max(gaps) if gaps else None, "failures": sel["failures"],
                "knee": {"size": k["size"], "bops": k["bops"], "resource": k["resource"],
                         "objective": k["objective"], "solver_status": k["solver_status"], "origin": k["origin"],
                         "search_budget": k["search_budget"], "weight_bytes": size, "bops_abs": bops,
                         "config_hash": config_hash(k["formats"])},
                "frontier": [{kk: r[kk] for kk in ("size", "bops", "resource", "objective", "solver_status",
                                                     "origin", "search_budget")} for r in sel["frontier"]]}
        summary["variants"][v] = info
        jdump(out / f"selection_{v}.json", dict(sel, solver_log=slog, search_s=secs, variant=v, tables=shas,
                                                 rows_from_branch=list(VARIANTS[v])))
        configs[f"knee_{v}"] = {"formats": k["formats"], "source": f"automatic knee, {v} costs"}
        log(f"search {v}: knee size {k['size']:.4f} BOPs {k['bops']:.4f}, objective {k['objective']:.6g}, "
            f"{len(slog)} MILP solves {info['status_counts']}, {secs:.1f}s")
    # matched-resource controls: both tables under each knee's exact (bytes, BOPs)
    ref_bytes, ref_bops = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
    controls = {}
    for point in ("total", "all_branch"):
        cap_bytes, cap_bops = resources(configs[f"knee_{point}"]["formats"], names, elems, flops)
        for v in ("total", "all_branch"):
            info = {}
            t0 = time.perf_counter()
            cfg, obj, status = milp_select(nodes, GRID11, GRID11, lambda n, w, a: B[n][w] + V[v][n][a], elems, flops,
                                           cap_bytes, cap_bops, time_limit=SEARCH["time_limit"], info=info)
            secs = time.perf_counter() - t0
            check_config(cfg, names, B, V[v])
            size, bops = resources(cfg, names, elems, flops)
            if status not in (0, 1) or size > cap_bytes or bops > cap_bops:
                raise RuntimeError(f"control at {point} knee with {v}: status {status}, over the limits")
            label = f"ctrl_{point}_{v}"
            controls[label] = {"point": point, "costs": v, "cap_weight_bytes": cap_bytes, "cap_bops": cap_bops,
                               "weight_bytes": size, "bops": bops, "size": size / ref_bytes, "bops_rel": bops / ref_bops,
                               "objective": obj, "objective_recomputed": objective(cfg, names, B, V[v]),
                               "objective_of_knee_same_point_under_these_costs": objective(
                                   configs[f"knee_{point}"]["formats"], names, B, V[v]),
                               "status": status, "solver": info, "seconds": secs, "config_hash": config_hash(cfg)}
            configs[label] = {"formats": cfg, "source": f"MILP with {v} costs at the {point} knee's resources"}
            log(f"control {label}: status {status}, gap {info['mip_gap']}, size {size / ref_bytes:.4f} "
                f"BOPs {bops / ref_bops:.4f}, {secs:.1f}s")
    summary["controls"] = controls
    for label, c in configs.items():
        c["config_hash"] = config_hash(c["formats"])
        c["weight_bytes"], c["bops_abs"] = resources(c["formats"], names, elems, flops)
        c["size"], c["bops"] = c["weight_bytes"] / ref_bytes, c["bops_abs"] / ref_bops
        c["objective_by_variant"] = {v: objective(c["formats"], names, B, V[v]) for v in VARIANTS}
    summary["config_labels"] = {lab: c["config_hash"] for lab, c in configs.items()}
    summary["distinct_configs"] = len({c["config_hash"] for c in configs.values()})
    summary["reference"] = {"int8_weight_bytes": ref_bytes, "int8_bops": ref_bops}
    jdump(out / "configs.json", configs)
    jdump(out / "search.json", summary)
    log(f"{len(configs)} configurations, {summary['distinct_configs']} distinct")


# --------------------------------------------------------------------------- #
# Stage eval
# --------------------------------------------------------------------------- #
def stage_eval(args, out, log):
    from evaluate_native_search_chunked import chunked_run_so
    P = problem()
    configs = jload(out / "configs.json")
    labels = args.labels.split(",") if args.labels else ["fp32", *configs]
    rbp = jload(TABLES / "ranges_by_point.json")
    ranges = tensor_ranges(P, rbp)
    toks = np.load(out / "eval_tokens.npy")
    inputs = out / "eval_inputs.npy"
    if not inputs.exists():
        np.save(inputs, toks)
    tools = erm.Tools()
    base = imported_model(TABLES)
    posit_obj = erm.build_posit_runtime(tools, out)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"      # as run_native_auto.py's evaluated builds
    by_hash = {}
    for lab in labels:
        h = "fp32" if lab == "fp32" else configs[lab]["config_hash"]
        by_hash.setdefault(h, []).append(lab)
    env_record = {"onnx_mlir_opt_sha256": sha256(tools.opt), "base_model": str(base),
                  "ranges_sha256": sha256(TABLES / "ranges_by_point.json"),
                  "eval_tokens_sha256": sha256(out / "eval_tokens.npy"), "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                  "levels": {"krnl": "O3", "clang": "O0"}, "constants_to_file": True, "bytecode": True}

    def one(h):
        target = out / "eval" / h[:16]
        result = target / "result.json"
        if result.exists():
            r = jload(result)
            if r["environment"] == env_record:
                log(f"eval {h[:16]} {by_hash[h]}: reused (same hash and environment)")
                return r
        target.mkdir(parents=True, exist_ok=True)
        env = None
        if h != "fp32":
            full = {n["name"]: configs[by_hash[h][0]]["formats"][n["name"]] for n in P["nodes"]}
            env = erm.node_formats_env(full, P["nodes"], ranges)
            if len(env.split(",")) != len(P["names"]):
                raise RuntimeError("compile environment does not list every searched node")
            (target / "compile_env.json").write_text(json.dumps(
                {"MIXEDP_NODE_FORMATS": env, "MIXEDP_WEIGHT_PER_CHANNEL": "1"}, indent=1) + "\n")
        t = time.time()
        so, info = erm.build_so(tools, target, "m", base, env, posit_obj, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True,
                                uses_posit=None if env else False, uses_f16=None if env else False)
        build_s = time.time() - t
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, ev = chunked_run_so(args.chunks)(tools, so, inputs, target / "logits.npy", 36000)
        finite = bool(np.isfinite(z).all())
        ce = [float(cross_entropy(z[s, :-1], toks[s, 1:]).sum()) for s in range(len(toks))]
        (target / "logits.npy").unlink()
        keep = any(lab in KEEP for lab in by_hash[h])
        so_sha = sha256(so)
        if not keep:
            for p in (so, target / "model.constants.bin"):
                p.unlink(missing_ok=True)
        r = {"config_hash": h, "labels": by_hash[h], "summed_ce": ce, "all_logits_finite": finite,
             "warnings": info["warnings"], "build_s": build_s, "build_steps": info.get("steps"),
             "eval_wall_s": ev["wall_seconds"], "eval_cpu_s": ev["seconds"], "eval_processes": ev["processes"],
             "peak_rss_gb": ev["peak_rss_gb"], "so_sha256": so_sha, "kept_for_heldout": keep,
             "environment": env_record}
        jdump(result, r)
        log(f"eval {h[:16]} {by_hash[h]}: calibration CE {sum(ce[:CAL_WINDOWS]):.4f}, tuning "
            f"{sum(ce[CAL_WINDOWS:]):.4f}, finite {finite}, warnings {len(info['warnings'])} "
            f"(build {build_s:.0f}s, run {ev['wall_seconds']:.0f}s)")
        return r
    jdump(out / f"environment_eval_{int(time.time())}.json", environment(EXTRA_FILES))
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, list(by_hash)))


# --------------------------------------------------------------------------- #
# Stage analyze
# --------------------------------------------------------------------------- #
def paired(d):
    """Mean, standard error and two-sided 95% t interval of per-window differences; wins = windows below 0."""
    from scipy import stats
    d = np.asarray(d, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    q = float(stats.t.ppf(0.975, n - 1))
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else None
    return {"n": n, "mean": float(d.mean()), "se": se, "ci95": [float(d.mean() - q * se), float(d.mean() + q * se)],
            "below_zero": int((d < 0).sum()), "t_test_p": p}


def results_by_label(out, configs):
    """Evaluation results per configuration label, matched by configuration hash (a label that names the same
    configuration as an evaluated one shares its result)."""
    by_hash = {}
    for p in (out / "eval").glob("*/result.json"):
        r = jload(p)
        by_hash[r["config_hash"]] = (p.parent, r)
    res = {"fp32": by_hash["fp32"][1]} if "fp32" in by_hash else {}
    for lab, c in configs.items():
        if c["config_hash"] in by_hash:
            res[lab] = by_hash[c["config_hash"]][1]
    return res


def node_diff(a, b, names, kinds, layers=None):
    diff = [n for n in names if a[n] != b[n]]
    out = {"nodes": len(diff), "by_kind": {k: sum(kinds[n] == k for n in diff)
                                           for k in ("q", "k", "v", "o", "gate", "up", "down")},
           "weight_changed": sum(a[n]["w"] != b[n]["w"] for n in diff),
           "activation_changed": sum(a[n]["a"] != b[n]["a"] for n in diff)}
    if layers is not None:
        out["list"] = [[layers[n], kinds[n], a[n]["w"], a[n]["a"], b[n]["w"], b[n]["a"]] for n in diff]
    return out


def stage_analyze(args, out, log):
    P = problem()
    names, kinds = P["names"], P["kinds"]
    layers = {n: v[0] for n, v in node_mapping().items()}
    configs, search = jload(out / "configs.json"), jload(out / "search.json")
    torch_ref = jload(out / "torch_fp32.json")["summed_ce"]
    res = results_by_label(out, configs)
    if "fp32" not in res:
        raise RuntimeError("no FP32 evaluation")
    tokens = SEQ - 1
    sets = {"calibration": slice(0, CAL_WINDOWS), "tuning": slice(CAL_WINDOWS, CAL_WINDOWS + TUNE_WINDOWS),
            "all": slice(0, CAL_WINDOWS + TUNE_WINDOWS)}
    fp = np.array(res["fp32"]["summed_ce"])
    tr = np.array(torch_ref)
    rows, per_window = {}, []
    for lab in ["fp32", *configs]:
        if lab not in res:
            continue
        ce = np.array(res[lab]["summed_ce"])
        c = configs.get(lab, {})
        row = {"config_hash": res[lab]["config_hash"], "size": c.get("size"), "bops": c.get("bops"),
               "all_logits_finite": res[lab]["all_logits_finite"], "warnings": len(res[lab]["warnings"]),
               "build_s": res[lab]["build_s"], "eval_wall_s": res[lab]["eval_wall_s"]}
        for s, sl in sets.items():
            x = ce[sl]
            row[s] = {"summed_ce": float(x.sum()), "mean_nll": float(x.sum() / (len(x) * tokens)),
                      "ppl": float(np.exp(x.sum() / (len(x) * tokens)))}
            if lab != "fp32":
                row[s]["delta_vs_fp32_mean_nll"] = paired((x - fp[sl]) / tokens)
        rows[lab] = row
        for i, v in enumerate(ce):
            per_window.append({"label": lab, "config_hash": res[lab]["config_hash"],
                               "set": "calibration" if i < CAL_WINDOWS else "tuning", "window": i,
                               "summed_ce": float(v), "mean_nll": float(v / tokens)})
    comparisons = {}

    def compare(name, a, b, note):
        if a not in res or b not in res:
            return
        A, Bv = np.array(res[a]["summed_ce"]), np.array(res[b]["summed_ce"])
        comparisons[name] = {"a": a, "b": b, "note": note,
                             "resources": {"a": [configs[a]["size"], configs[a]["bops"]],
                                           "b": [configs[b]["size"], configs[b]["bops"]]},
                             "node_diff": node_diff(configs[a]["formats"], configs[b]["formats"], names, kinds,
                                                    layers),
                             **{s: paired((Bv[sl] - A[sl]) / tokens) for s, sl in sets.items()}}
    compare("knee: all_branch - total", "knee_total", "knee_all_branch", "automatic knees (resources may differ)")
    compare("knee: ffn_branch - total", "knee_total", "knee_ffn_branch", "automatic knees")
    compare("knee: attention_branch - total", "knee_total", "knee_attention_branch", "automatic knees")
    compare("at total knee: branch - total", "ctrl_total_total", "ctrl_total_all_branch",
            "same caps: the total knee's weight bytes and BOPs")
    compare("at branch knee: branch - total", "ctrl_all_branch_total", "ctrl_all_branch_all_branch",
            "same caps: the all_branch knee's weight bytes and BOPs")
    compare("total: control at total knee - knee", "knee_total", "ctrl_total_total", "same costs, caps vs common budget")
    compare("branch: control at branch knee - knee", "knee_all_branch", "ctrl_all_branch_all_branch",
            "same costs, caps vs common budget")
    numerics = {"native_fp32_minus_torch_fp32_summed_ce": {s: float((fp[sl] - tr[sl]).sum()) for s, sl in sets.items()},
                "max_abs_per_window": float(np.abs(fp - tr).max())}
    solves = {}
    for v in search["variants"]:                  # the solve that produced the knee, and the solves near it
        sel = jload(out / f"selection_{v}.json")
        kb = sel["selected"]["search_budget"]
        near = [r for r in sel["solver_log"] if "mip_gap" in r and r["budget"] <= 2 * kb]
        solves[v] = {"knee_solve": next((r for r in sel["solver_log"] if r["budget"] == kb), None),
                     "max_gap_budget_le_2x_knee": max((r["mip_gap"] for r in near if r["mip_gap"] is not None),
                                                      default=None),
                     "attempted_budgets": sel["attempted_budgets"]}
    analysis = {"rows": rows, "comparisons": comparisons, "numerics": numerics,
                "search": {v: {**{k: search["variants"][v][k] for k in ("search_s", "milp_solves", "status_counts",
                                                                        "max_mip_gap", "knee")}, **solves[v]}
                           for v in search["variants"]},
                "controls": search["controls"], "tokens_per_window": tokens,
                "windows": {"calibration": [0, CAL_WINDOWS - 1],
                            "tuning": [CAL_WINDOWS, CAL_WINDOWS + TUNE_WINDOWS - 1]}}
    jdump(out / "analysis.json", analysis)
    import csv
    with open(out / "per_window.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_window[0]))
        w.writeheader()
        w.writerows(per_window)
    with open(out / "comparisons.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "set", "n", "mean_dnll", "se", "ci_lo", "ci_hi", "windows_b_lower", "t_test_p",
                    "nodes_changed"])
        for name, c in comparisons.items():
            for s in sets:
                x = c[s]
                w.writerow([name, s, x["n"], x["mean"], x["se"], *x["ci95"], x["below_zero"], x["t_test_p"],
                            c["node_diff"]["nodes"]])
    with open(out / "configs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "config_hash", "size", "bops", "calibration_mean_nll", "tuning_mean_nll",
                    "calibration_ppl", "tuning_ppl", "all_logits_finite", "warnings"])
        for lab, r in rows.items():
            w.writerow([lab, r["config_hash"], r["size"], r["bops"], r["calibration"]["mean_nll"],
                        r["tuning"]["mean_nll"], r["calibration"]["ppl"], r["tuning"]["ppl"],
                        r["all_logits_finite"], r["warnings"]])
    log(json.dumps({k: {s: round(v[s]["mean"], 5) for s in sets} for k, v in comparisons.items()}, indent=1))


# --------------------------------------------------------------------------- #
# Stages decide / heldout
# --------------------------------------------------------------------------- #
def stage_decide(args, out, log):
    if not args.decision:
        raise SystemExit("--decision TEXT is required")
    if (out / "decision.json").exists():
        raise SystemExit("decision.json exists: a decision is frozen once")
    configs = jload(out / "configs.json")
    jdump(out / "decision.json", {"decision": args.decision, "time": datetime.datetime.now().isoformat(),
                                  "analysis_sha256": sha256(out / "analysis.json"),
                                  "heldout_configs": {lab: configs[lab]["config_hash"] for lab in KEEP if lab != "fp32"},
                                  "heldout_used_for_selection": False})
    log("decision frozen")


def stage_heldout(args, out, log):
    from evaluate_native_search_chunked import chunked_run_so
    d = out / "decision.json"
    if not d.exists():
        raise SystemExit("held-out evaluation needs a frozen decision (stage decide) first")
    dec = jload(d)
    if dec["analysis_sha256"] != sha256(out / "analysis.json"):
        raise SystemExit("analysis.json changed after the decision")
    test = np.load(PREP / "test_tokens.npy")
    inputs = out / "heldout_inputs.npy"
    np.save(inputs, test)
    tools = erm.Tools()
    configs = jload(out / "configs.json")
    for lab, h in dec["heldout_configs"].items():
        if configs[lab]["config_hash"] != h:
            raise SystemExit(f"{lab}: configuration changed after the decision")
    res = results_by_label(out, configs)

    def one(lab):
        r = res[lab]
        folder = out / "eval" / r["config_hash"][:16]
        so = folder / "m.so"
        if not r["kept_for_heldout"] or sha256(so) != r["so_sha256"]:
            raise RuntimeError(f"{lab}: the evaluated build is not available unchanged")
        z, ev = chunked_run_so(args.chunks)(tools, so, inputs, folder / "heldout_logits.npy", 36000)
        finite = bool(np.isfinite(z).all())
        nll = np.stack([cross_entropy(z[i, :-1], test[i, 1:]) for i in range(len(test))])   # [windows, 127]
        (folder / "heldout_logits.npy").unlink()
        return lab, {"config_hash": r["config_hash"], "per_window_mean_nll": nll.mean(1).tolist(),
                     "nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())), "all_logits_finite": finite,
                     "eval_wall_s": ev["wall_seconds"]}
    with ThreadPoolExecutor(args.parallel) as pool:
        out_rows = dict(pool.map(one, KEEP))
    fp = np.array(out_rows["fp32"]["per_window_mean_nll"])
    for lab in KEEP[1:]:
        out_rows[lab]["delta_vs_fp32"] = paired(np.array(out_rows[lab]["per_window_mean_nll"]) - fp)
    out_rows["branch_minus_total"] = paired(np.array(out_rows["knee_all_branch"]["per_window_mean_nll"])
                                            - np.array(out_rows["knee_total"]["per_window_mean_nll"]))
    jdump(out / "heldout.json", {"decision_time": dec["time"], "test_windows": len(test),
                                 "test_tokens_sha256": sha256(PREP / "test_tokens.npy"),
                                 "used_for_selection": False, "results": out_rows})
    log(json.dumps({k: (v.get("ppl") if "ppl" in v else v["mean"]) for k, v in out_rows.items()}))


def stage_tables(args, out, log):
    """The report's tables (Markdown) from analysis.json, search.json and, if present, heldout.json."""
    a, s = jload(out / "analysis.json"), jload(out / "search.json")
    f = lambda x, p=4: "—" if x is None else f"{x:.{p}g}"
    ci = lambda x: f"{x['mean']:+.5f} ± {x['se']:.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}] ({x['below_zero']}/{x['n']})"
    lines = ["### search", "", "| 成本 | knee size | knee BOPs | max(size,BOPs) | 目標值（自身成本） | 搜尋秒數 | MILP 次數 | "
             "status | knee 那次 gap | knee 附近最大 gap |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for v, x in a["search"].items():
        k = x["knee"]
        lines.append(f"| {v} | {k['size']:.5f} | {k['bops']:.5f} | {k['resource']:.5f} | {k['objective']:.6g} | "
                     f"{x['search_s']:.1f} | {x['milp_solves']} | {x['status_counts']} | "
                     f"{f(x['knee_solve']['mip_gap'], 3) if x['knee_solve'] else '—'} | "
                     f"{f(x['max_gap_budget_le_2x_knee'], 3)} |")
    lines += ["", "### controls", "", "| 控制點 | 成本 | 上限 size／BOPs | 實際 size／BOPs | status | gap | 目標值 | "
              "該點 knee 在此成本下的目標值 | 秒數 |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    ref = s["reference"]
    for lab, c in a["controls"].items():
        lines.append(f"| {c['point']} knee | {c['costs']} | {c['cap_weight_bytes'] / ref['int8_weight_bytes']:.6f}／"
                     f"{c['cap_bops'] / ref['int8_bops']:.6f} | {c['size']:.6f}／{c['bops_rel']:.6f} | {c['status']} | "
                     f"{f(c['solver']['mip_gap'], 3)} | {c['objective']:.6g} | "
                     f"{c['objective_of_knee_same_point_under_these_costs']:.6g} | {c['seconds']:.1f} |")
    lines += ["", "### configurations", "", "| 標籤 | hash | size | BOPs | calibration mean NLL（PPL） | tuning mean NLL（PPL） | "
              "Δ vs FP32 calibration | Δ vs FP32 tuning | finite | warnings | build 秒 | run 秒 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for lab, r in a["rows"].items():
        d = lambda st: f"{r[st]['delta_vs_fp32_mean_nll']['mean']:+.5f} ± {r[st]['delta_vs_fp32_mean_nll']['se']:.5f}" \
            if "delta_vs_fp32_mean_nll" in r[st] else "—"
        lines.append(f"| {lab} | {r['config_hash'][:12]} | {f(r['size'], 5)} | {f(r['bops'], 5)} | "
                     f"{r['calibration']['mean_nll']:.5f}（{r['calibration']['ppl']:.3f}） | "
                     f"{r['tuning']['mean_nll']:.5f}（{r['tuning']['ppl']:.3f}） | {d('calibration')} | {d('tuning')} | "
                     f"{r['all_logits_finite']} | {r['warnings']} | {r['build_s']:.0f} | {r['eval_wall_s']:.0f} |")
    lines += ["", "### comparisons (b − a, mean NLL per token; mean ± SE [95% CI] (windows b lower / n))", "",
              "| 比較 | 改選節點（q/k/v/o/gate/up/down） | a size／BOPs | b size／BOPs | calibration | tuning | 全部 30 |",
              "|---|---|---|---|---|---|---|"]
    for name, c in a["comparisons"].items():
        k = c["node_diff"]["by_kind"]
        lines.append(f"| {name} | {c['node_diff']['nodes']}（{'/'.join(str(k[x]) for x in ('q', 'k', 'v', 'o', 'gate', 'up', 'down'))}） | "
                     f"{c['resources']['a'][0]:.4f}／{c['resources']['a'][1]:.4f} | "
                     f"{c['resources']['b'][0]:.4f}／{c['resources']['b'][1]:.4f} | {ci(c['calibration'])} | "
                     f"{ci(c['tuning'])} | {ci(c['all'])} |")
    n = a["numerics"]
    lines += ["", f"native FP32 − PyTorch FP32 summed CE：calibration {n['native_fp32_minus_torch_fp32_summed_ce']['calibration']:+.4f}，"
              f"tuning {n['native_fp32_minus_torch_fp32_summed_ce']['tuning']:+.4f}，每窗口最大 abs {n['max_abs_per_window']:.4f}"]
    if (out / "heldout.json").exists():
        h = jload(out / "heldout.json")["results"]
        lines += ["", "### held-out", "", "| 配置 | test PPL | mean NLL | Δ vs FP32（per-window mean NLL） | finite |",
                  "|---|---:|---:|---|---|"]
        for lab in KEEP:
            r = h[lab]
            lines.append(f"| {lab} | {r['ppl']:.3f} | {r['nll']:.5f} | {ci(r['delta_vs_fp32']) if 'delta_vs_fp32' in r else '—'} | "
                         f"{r['all_logits_finite']} |")
        lines.append(f"\nbranch − total（per-window mean NLL）：{ci(h['branch_minus_total'])}")
    text = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=("check", "selftest", "search", "eval", "analyze", "decide", "heldout", "tables"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--labels", help="eval: comma-separated configuration labels (default: all, and fp32)")
    ap.add_argument("--chunks", type=int, default=5, help="processes per configuration run")
    ap.add_argument("--parallel", type=int, default=2, help="configurations built / run at once")
    ap.add_argument("--decision", help="decide: the policy conclusion, frozen before the held-out evaluation")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "check":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "assets.json").exists():
        raise SystemExit("run --stage check first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    if args.stage in ("check", "search"):
        jdump(out / f"environment_{args.stage}.json", environment(EXTRA_FILES))
    {"check": stage_check, "selftest": stage_selftest, "search": stage_search, "eval": stage_eval,
     "analyze": stage_analyze, "decide": stage_decide, "heldout": stage_heldout,
     "tables": stage_tables}[args.stage](args, out, log)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
