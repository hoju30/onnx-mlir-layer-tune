#!/usr/bin/env python3
"""Phase 1 of the SQNR criterion in this flow: is it worth replacing the HTQ criterion and/or its tree search?

At the same budgets, three ways to pick a per-node format config:
  htq_tree   the current flow: --sensitivity-mode criterion + htq_pareto_search's tree search
             (sample_configs_htq.py); per budget, its lowest-profit frontier leaf within the budget
  htq_milp   the same criterion, exact optimum by MILP (mp_milp.py)
  sqnr_milp  the SQNR criterion (sqnr_tables.py), exact optimum by MILP
Budget f: weight bytes <= f * all-int8 and BOPs proxy <= f * all-int8xint8, as in compare_search_2000.py
(sample_configs_htq.py's --top-n evenly spaced points mostly land above all-int8, where the frontier runs on
to fp32 and every config is near-lossless). All configs are evaluated with simulate_in_onnxruntime on held-out
images (paired bootstrap on the same images), and written in sample_configs_htq.py's format so that
eval_random_mixed_precision.py --htq-configs can compile and run any of them through onnx-mlir unchanged.

Posit activations are left out of the menu for all three: onnxruntime cannot simulate them. Images come
from an existing manifest (default: /home/hoju/test/HTQ/imagenet1k_2050, 50 calibration + 2000 held-out);
the first --num-calibration calibration images are used, as this flow's default of 10.

Usage:
  ../../venv/bin/python3 sample_configs_sqnr.py \\
      --model-dir ../imagenet100/model_dataset/resnet18/torchvision_v1
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_random_mixed_precision import (  # noqa: E402
    FORMATS, POSIT_FORMATS, Tools, PositOracle, calibrate_ranges, preprocess_by_source,
    simulate_in_onnxruntime, supported_nodes,
)
from htq_perturbation import build_perturbation_tables, weight_perturbation  # noqa: E402
from htq_pareto_search import FORMAT_BITS, leaf_config, pareto_search  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from sample_configs_htq import (  # noqa: E402
    DEFAULT_ORT_TRAINING_PYTHON, compute_weight_elems_and_flops_proxy, tap_output_samples,
)
from sqnr_tables import sqnr_table  # noqa: E402

METHODS = ("htq_tree", "htq_milp", "sqnr_milp")


def config_cost(formats, weight_elems, flops_proxy):
    """(weight bytes, BOPs proxy) of a config, as pareto_search and mp_milp count them."""
    weighted = [(n, f) for n, f in formats.items() if f["w"] is not None]
    return (sum(weight_elems[n] * FORMAT_BITS[f["w"]] / 8 for n, f in weighted),
            sum(flops_proxy[n] * FORMAT_BITS[f["w"]] * FORMAT_BITS[f["a"]] for n, f in weighted))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="dir with model.onnx + metadata.json")
    ap.add_argument("--images-dir", type=Path, default=Path("/home/hoju/test/HTQ/imagenet1k_2050/images"))
    ap.add_argument("--num-calibration", type=int, default=10)
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--budgets", default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
                    help="fractions of the all-int8 weight bytes and BOPs proxy")
    ap.add_argument("--sensitivity-mode", choices=["shared", "split_connection", "split_direct"],
                    default="split_direct")
    ap.add_argument("--htq-activation-term", choices=["samples", "weights"], default="samples",
                    help="samples: this flow's ||s - Q_a(s)|| on real activation samples; weights: original HTQ "
                         "(HTQ/pareto_frontier.py), ||W - Q_a(W)|| on the node's own weight tensor")
    ap.add_argument("--weight-formats", default=",".join(FORMATS))
    ap.add_argument("--act-formats", default=",".join(f for f in FORMATS if f not in POSIT_FORMATS))
    ap.add_argument("--prune-fraction", type=float, default=1 / 20)
    ap.add_argument("--beam-width", type=int, default=2000)
    ap.add_argument("--ort-training-python", type=Path, default=DEFAULT_ORT_TRAINING_PYTHON)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default runs/<arch>_<variant>_sqnr_phase1 (sqnr_table.json there is reused if present)")
    args = ap.parse_args()
    weight_formats, act_formats = args.weight_formats.split(","), args.act_formats.split(",")
    out = args.out_dir or HERE / "runs" / f"{args.model_dir.parent.name}_{args.model_dir.name}_sqnr_phase1"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sqnr-"))
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    import onnx
    model = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]][:args.num_calibration]
    held = [e for e in manifest["images"] if not e["calibration"]][:args.num_eval]
    calib_inputs = preprocess_by_source(args.images_dir, {"images": calib}, meta)
    calib_labels = np.array([e["label"] for e in calib])
    eval_inputs = preprocess_by_source(args.images_dir, {"images": held}, meta)
    eval_labels = np.array([e["label"] for e in held])
    nodes = supported_nodes(model)
    log(f"{args.model_dir}: {len(nodes)} supported nodes ({sum(1 for n in nodes if n.get('weight_const'))} with "
        f"weights); {len(calib)} calibration, {len(held)} held-out images; "
        f"weights {weight_formats}, activations {act_formats}")

    ranges = calibrate_ranges(model, nodes, calib_inputs)
    posit_oracle = PositOracle(Tools(), work)

    # --- current flow: HTQ criterion + tree search (as sample_configs_htq.py) ---
    np.save(work / "calib_inputs.npy", calib_inputs)
    np.save(work / "calib_labels.npy", calib_labels)
    subprocess.run([str(args.ort_training_python), str(HERE / "htq_snip_ort.py"), "--onnx",
                    str(args.model_dir / "model.onnx"), "--inputs", str(work / "calib_inputs.npy"),
                    "--labels", str(work / "calib_labels.npy"), "--mode", args.sensitivity_mode,
                    "--out", str(work / "sensitivity.json")], check=True, stdout=subprocess.DEVNULL)
    sens = json.loads((work / "sensitivity.json").read_text())
    sens_w, sens_a = sens["weight"], sens["activation"]
    inits = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    weights = {n["name"]: onnx.numpy_helper.to_array(inits[by_name[n["name"]].input[1]])
               for n in nodes if n.get("weight_const")}
    weight_table, activation_table = build_perturbation_tables(
        nodes, sorted(set(weight_formats) | set(act_formats)), weights,
        tap_output_samples(model, nodes, calib_inputs), ranges, posit_oracle)
    if args.htq_activation_term == "weights":
        # Original HTQ never looks at an activation: its "activation perturbation" is the WEIGHT tensor quantized
        # in the activation's format. Activation-only nodes keep their (unused: they stay fp32) sample-based entry.
        for name, w in weights.items():
            activation_table[name] = {a: weight_perturbation(w, a, posit_oracle) for a in act_formats}
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(model, nodes)
    leaves = pareto_search(nodes, weight_formats, act_formats, sens_w, sens_a, weight_table, activation_table,
                           weight_elems, flops_proxy, prune_fraction=args.prune_fraction, beam_width=args.beam_width)
    weighted = [n["name"] for n in nodes if n.get("weight_const")]
    int8_bytes, int8_bops = sum(weight_elems[n] for n in weighted), sum(64 * flops_proxy[n] for n in weighted)
    budgets = [float(f) for f in args.budgets.split(",")]
    log(f"HTQ ({args.sensitivity_mode}, activation term on {args.htq_activation_term}) tree search: {len(leaves)} leaves; all-int8 {int8_bytes / 2**20:.2f} MB, "
        f"{int8_bops / 1e9:.1f} GBOPs proxy")

    # --- SQNR table (the slow part; cached) ---
    table_path = out / "sqnr_table.json"
    settings = {"model": str(args.model_dir), "calibration": [e["file"] for e in calib],
                "weight_formats": weight_formats, "act_formats": act_formats}
    cached = json.loads(table_path.read_text()) if table_path.exists() else None
    if cached and cached["settings"] == settings:
        sqnr = cached["table"]
        log("SQNR table reused from sqnr_table.json")
    else:
        sqnr = sqnr_table(model, nodes, weight_formats, act_formats, ranges, calib_inputs, posit_oracle,
                          log=lambda m: log(m))
        table_path.write_text(json.dumps({"settings": settings, "table": sqnr}) + "\n")

    def htq_value(name, w, a):
        return sens_w.get(name, 0.0) * weight_table[name][w] + sens_a.get(name, 0.0) * activation_table[name][a]

    def sqnr_value(name, w, a):
        return sqnr[name][f"{w}|{a}"]

    op_by_name = {n["name"]: n["op"] for n in nodes}
    configs = []
    for i, f in enumerate(budgets):
        max_bytes, max_bops = f * int8_bytes, f * int8_bops
        ok = [l for l in leaves if l.cost <= max_bytes and l.bops <= max_bops]
        if not ok:
            log(f"f={f}: no tree-search leaf within budget, skipped")
            continue
        leaf = min(ok, key=lambda l: l.profit)
        chosen = {"htq_tree": (leaf_config(leaf), leaf.profit)}
        for method, value in (("htq_milp", htq_value), ("sqnr_milp", sqnr_value)):
            formats, objective, status = milp_select(nodes, weight_formats, act_formats, value, weight_elems,
                                                     flops_proxy, max_bytes, max_bops)
            assert status == 0, f"{method} f={f}: MILP not proven optimal (status {status})"
            chosen[method] = (formats, objective)
        # The MILP optimum of the HTQ profit can never be worse than the tree search's leaf.
        assert chosen["htq_milp"][1] <= leaf.profit * (1 + 1e-9), "MILP lost to the tree search on its own objective"
        for method, (formats, objective) in chosen.items():
            for name, fmt in formats.items():
                fmt["op"] = op_by_name[name]
            size, bops = config_cost(formats, weight_elems, flops_proxy)
            configs.append({"id": f"{method}_{i:03d}", "method": method, "budget": f,
                            "cost_weight_bytes": size, "bops_proxy": bops,
                            "objective": objective, "formats": formats})
    (out / "configs.json").write_text(json.dumps({"configs": configs}, indent=1) + "\n")
    log(f"wrote {len(configs)} configs to {out / 'configs.json'}")

    # --- evaluate everything in onnxruntime simulation on the held-out images ---
    reference = simulate_in_onnxruntime(model, nodes, {}, ranges, eval_inputs, posit_oracle)
    correct = {"fp32": reference.argmax(1) == eval_labels}
    for c in configs:
        logits = simulate_in_onnxruntime(model, nodes, c["formats"], ranges, eval_inputs, posit_oracle)
        correct[c["id"]] = (logits.argmax(1) == eval_labels) & np.isfinite(logits).all(1)
        c["top1"] = float(correct[c["id"]].mean())
    log(f"evaluated {len(configs)} configs on {len(held)} images (fp32 top-1 {correct['fp32'].mean():.2%})")

    resample = np.random.default_rng(0).integers(0, len(held), (args.boot, len(held)))

    def paired(a, b):
        d = correct[a].astype(float) - correct[b].astype(float)
        return float(d.mean()), np.percentile(d[resample].mean(1), [2.5, 97.5]).tolist()

    rows = []
    print(f"\n{'budget':>6}" + "".join(f"{m:>11}" for m in METHODS)
          + f"{'sqnr − htq_tree':>26}{'sqnr − htq_milp':>26}{'htq_milp − htq_tree':>26}")
    for f in sorted({c["budget"] for c in configs}):
        i = budgets.index(f)
        ids = {m: f"{m}_{i:03d}" for m in METHODS}
        row = {"budget": f,
               **{m: next(c["top1"] for c in configs if c["id"] == ids[m]) for m in METHODS},
               "sqnr_minus_tree": paired(ids["sqnr_milp"], ids["htq_tree"]),
               "sqnr_minus_htq_milp": paired(ids["sqnr_milp"], ids["htq_milp"]),
               "milp_minus_tree": paired(ids["htq_milp"], ids["htq_tree"])}
        rows.append(row)
        cells = [f"{d * 100:+6.2f} [{lo * 100:+5.1f},{hi * 100:+5.1f}]"
                 for d, (lo, hi) in (row["sqnr_minus_tree"], row["sqnr_minus_htq_milp"], row["milp_minus_tree"])]
        print(f"{f:6.2f}" + "".join(f"{row[m]:>11.2%}" for m in METHODS) + "".join(f"{c:>26}" for c in cells))
    print("  mean" + "".join(f"{np.mean([r[m] for r in rows]):>11.2%}" for m in METHODS))
    (out / "results.json").write_text(json.dumps(
        {"settings": {**settings, "sensitivity_mode": args.sensitivity_mode,
                      "htq_activation_term": args.htq_activation_term, "num_eval": len(held),
                      "budgets": budgets, "beam_width": args.beam_width},
         "fp32_top1": float(correct["fp32"].mean()), "rows": rows,
         "configs": [{k: v for k, v in c.items() if k != "formats"} for c in configs]}, indent=1) + "\n")
    log(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
