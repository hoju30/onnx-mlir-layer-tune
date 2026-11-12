#!/usr/bin/env python3
"""With a criterion that works in this flow (check_shared_fix.py's shared_v2), is HTQ's tree search
(htq_pareto_search.py) still worse than MILP, and does that depend on how many formats each node may pick?

For a small menu (int4/int6/int8 for weight and activation, 9 options per node) and the full non-posit menu
(8 x 8 = 64), both searches get the same objective and budgets (weight bytes and BOPs proxy <= f * all-int8).
Reported per budget: the tree's objective as a multiple of the MILP optimum, the cost each actually uses, and
the onnxruntime-simulated top-1 of both picks on the held-out images.
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

from check_shared_fix import tap_samples  # noqa: E402
from eval_random_mixed_precision import (  # noqa: E402
    Tools, PositOracle, calibrate_ranges, preprocess_by_source, simulate_in_onnxruntime, supported_nodes,
)
from htq_pareto_search import leaf_config, pareto_search  # noqa: E402
from htq_perturbation import activation_perturbation, weight_perturbation  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from sample_configs_htq import DEFAULT_ORT_TRAINING_PYTHON, compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

MENUS = {"small (3x3)": ["int4", "int6", "int8"],
         "full (8x8)": ["int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--beam-width", type=int, default=2000)
    ap.add_argument("--ort-training-python", type=Path, default=DEFAULT_ORT_TRAINING_PYTHON)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(f) for f in args.budgets.split(",")]
    work = Path(tempfile.mkdtemp(prefix="tree-milp-"))
    t0 = time.time()

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
    np.save(work / "x.npy", cx)
    np.save(work / "y.npy", np.array([e["label"] for e in calib]))
    subprocess.run([str(args.ort_training_python), str(HERE / "htq_snip_ort.py"), "--onnx",
                    str(args.model_dir / "model.onnx"), "--inputs", str(work / "x.npy"), "--labels",
                    str(work / "y.npy"), "--mode", "shared", "--out", str(work / "s.json")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    S = json.loads((work / "s.json").read_text())["weight"]

    # shared_v2 tables (check_shared_fix.py): relative noise, activation term over the node's input and output
    weighted = [n for n in nodes if n.get("weight_const")]
    inits = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    samples = tap_samples(model, sorted({t for n in weighted for t in n["act_tensors"]}), cx)
    formats = MENUS["full (8x8)"]
    Nw, Na = {}, {}
    for n in weighted:
        name, x, y = n["name"], n["act_tensors"][0], n["act_tensors"][-1]
        W = onnx.numpy_helper.to_array(inits[by_name[name].input[1]]).astype(np.float64)
        Nw[name] = {f: (weight_perturbation(W, f, oracle) / np.linalg.norm(W)) ** 2 for f in formats}
        Na[name] = {f: (activation_perturbation(samples[x], f, ranges[x], oracle) / np.linalg.norm(samples[x])) ** 2
                    + (activation_perturbation(samples[y], f, ranges[y], oracle) / np.linalg.norm(samples[y])) ** 2
                    for f in formats}
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(model, nodes)
    names = [n["name"] for n in weighted]
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)

    def value(name, w, a):
        return S.get(name, 0.0) * (Nw[name][w] + Na[name][a])

    def top1(fmts):
        z = simulate_in_onnxruntime(model, nodes, fmts, ranges, ex, oracle)
        return float(((z.argmax(1) == ey) & np.isfinite(z).all(1)).mean())

    fp32 = top1({})
    print(f"[{time.time() - t0:4.0f}s] shared_v2 tables ready; fp32 simulated top-1 {fp32:.1%} on {len(ey)} images",
          flush=True)
    rows = []
    for menu, fm in MENUS.items():
        t = time.time()
        leaves = pareto_search(weighted, fm, fm, S, S, Nw, Na, weight_elems, flops_proxy,
                               beam_width=args.beam_width)
        tree_s = time.time() - t
        print(f"\n== menu {menu}: {len(fm) ** 2} options/node, tree search {tree_s:.1f}s, {len(leaves)} leaves")
        print(f"{'f':>5} {'tree obj / MILP obj':>20} {'tree size/BOPs':>16} {'MILP size/BOPs':>16} "
              f"{'tree top1':>10} {'MILP top1':>10}")
        for f in budgets:
            ok = [l for l in leaves if l.cost <= f * int8_bytes and l.bops <= f * int8_bops]
            milp_fmts, milp_obj, status = milp_select(nodes, fm, fm, value, weight_elems, flops_proxy,
                                                      f * int8_bytes, f * int8_bops)
            assert status == 0
            ms, mb = config_cost(milp_fmts, weight_elems, flops_proxy)
            row = {"menu": menu, "budget": f, "tree_seconds": tree_s, "milp_objective": milp_obj,
                   "milp_size": ms / int8_bytes, "milp_bops": mb / int8_bops, "milp_top1": top1(milp_fmts)}
            if ok:
                leaf = min(ok, key=lambda l: l.profit)
                tree_fmts = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
                tree_fmts.update(leaf_config(leaf))
                row.update(tree_objective=leaf.profit, tree_size=leaf.cost / int8_bytes,
                           tree_bops=leaf.bops / int8_bops, tree_top1=top1(tree_fmts))
                cells = (f"{leaf.profit / milp_obj:20.2f}x", f"{row['tree_size']:.2f}/{row['tree_bops']:.2f}",
                         f"{row['tree_top1']:10.1%}")
            else:
                cells = (f"{'no leaf in budget':>21}", "-", f"{'-':>10}")
            rows.append(row)
            print(f"{f:5.2f} {cells[0]} {cells[1]:>16} {row['milp_size']:.2f}/{row['milp_bops']:.2f}".ljust(62)
                  + f"{cells[2]} {row['milp_top1']:10.1%}", flush=True)
    (args.out_dir / "results.json").write_text(json.dumps(
        {"fp32_sim_top1": fp32, "num_eval": len(ey), "beam_width": args.beam_width, "rows": rows}, indent=1) + "\n")
    print(f"\n[{time.time() - t0:4.0f}s] wrote {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
