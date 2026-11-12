#!/usr/bin/env python3
"""Compare selection criteria at equal budgets, each solved exactly by MILP (mp_milp.py; no tree search):
  shared / split_connection / split_direct   HTQ profit sens_w * w_err + sens_a * a_err (htq_snip_ort.py modes,
                                              activation error on real activation samples)
  sqnr                                        summed single-node logit noise power (sqnr_tables.py)
Budget f: weight bytes <= f * all-int8 and BOPs proxy <= f * all-int8xint8, as in sample_configs_sqnr.py.
Writes the configs in sample_configs_htq.py's format, so eval_random_mixed_precision.py --htq-configs compiles
and runs them through onnx-mlir unchanged.

Posit formats are left out of the default menu: onnxruntime cannot simulate posit activations, and posit
weights make each SQNR-table simulation ~12x slower (0.62 s vs 0.05 s per call on mobilenetv2).
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
    FORMATS, POSIT_FORMATS, Tools, PositOracle, calibrate_ranges, preprocess_by_source, supported_nodes,
)
from htq_perturbation import build_perturbation_tables  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from sample_configs_htq import (  # noqa: E402
    DEFAULT_ORT_TRAINING_PYTHON, compute_weight_elems_and_flops_proxy, tap_output_samples,
)
from sample_configs_sqnr import config_cost  # noqa: E402
from sqnr_tables import sqnr_table  # noqa: E402

HTQ_MODES = ("shared", "split_connection", "split_direct")
CRITERIA = HTQ_MODES + ("sqnr",)
NON_POSIT = ",".join(f for f in FORMATS if f not in POSIT_FORMATS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="dir with model.onnx + metadata.json")
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="manifest.json + images; every calibration-flagged image is used for calibration")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8",
                    help="fractions of the all-int8 weight bytes and BOPs proxy")
    ap.add_argument("--weight-formats", default=NON_POSIT)
    ap.add_argument("--act-formats", default=NON_POSIT)
    ap.add_argument("--ort-training-python", type=Path, default=DEFAULT_ORT_TRAINING_PYTHON)
    ap.add_argument("--out-dir", type=Path, required=True, help="configs.json (+ cached sqnr_table.json) go here")
    args = ap.parse_args()
    weight_formats, act_formats = args.weight_formats.split(","), args.act_formats.split(",")
    budgets = [float(f) for f in args.budgets.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="criteria-"))
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    import onnx
    model = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    calib_inputs = preprocess_by_source(args.images_dir, {"images": calib}, meta)
    calib_labels = np.array([e["label"] for e in calib])
    nodes = supported_nodes(model)
    ranges = calibrate_ranges(model, nodes, calib_inputs)
    posit_oracle = PositOracle(Tools(), work)
    log(f"{args.model_dir}: {len(nodes)} supported nodes ({sum(1 for n in nodes if n.get('weight_const'))} with "
        f"weights), {len(calib)} calibration images; weights {weight_formats}, activations {act_formats}")

    # --- HTQ profits: one SNIP sensitivity run per mode, shared perturbation tables ---
    np.save(work / "calib_inputs.npy", calib_inputs)
    np.save(work / "calib_labels.npy", calib_labels)
    sens = {}
    for mode in HTQ_MODES:
        subprocess.run([str(args.ort_training_python), str(HERE / "htq_snip_ort.py"), "--onnx",
                        str(args.model_dir / "model.onnx"), "--inputs", str(work / "calib_inputs.npy"),
                        "--labels", str(work / "calib_labels.npy"), "--mode", mode,
                        "--out", str(work / f"sensitivity_{mode}.json")], check=True, stdout=subprocess.DEVNULL)
        sens[mode] = json.loads((work / f"sensitivity_{mode}.json").read_text())
    inits = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    weights = {n["name"]: onnx.numpy_helper.to_array(inits[by_name[n["name"]].input[1]])
               for n in nodes if n.get("weight_const")}
    weight_table, activation_table = build_perturbation_tables(
        nodes, sorted(set(weight_formats) | set(act_formats)), weights,
        tap_output_samples(model, nodes, calib_inputs), ranges, posit_oracle)
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(model, nodes)
    log("SNIP sensitivities (3 modes) and perturbation tables done")

    # --- SQNR table (the slow part; cached) ---
    table_path = args.out_dir / "sqnr_table.json"
    settings = {"model": str(args.model_dir), "calibration": [e["file"] for e in calib],
                "weight_formats": weight_formats, "act_formats": act_formats}
    cached = json.loads(table_path.read_text()) if table_path.exists() else None
    if cached and cached["settings"] == settings:
        sqnr = cached["table"]
        log("SQNR table reused")
    else:
        sqnr = sqnr_table(model, nodes, weight_formats, act_formats, ranges, calib_inputs, posit_oracle,
                          log=lambda m: log(m))
        table_path.write_text(json.dumps({"settings": settings, "table": sqnr}) + "\n")

    def value_fn(criterion):
        if criterion == "sqnr":
            return lambda name, w, a: sqnr[name][f"{w}|{a}"]
        sw, sa = sens[criterion]["weight"], sens[criterion]["activation"]
        return lambda name, w, a: sw.get(name, 0.0) * weight_table[name][w] + sa.get(name, 0.0) * activation_table[name][a]

    # --- exact selection per criterion and budget ---
    weighted = [n["name"] for n in nodes if n.get("weight_const")]
    int8_bytes, int8_bops = sum(weight_elems[n] for n in weighted), sum(64 * flops_proxy[n] for n in weighted)
    op_by_name = {n["name"]: n["op"] for n in nodes}
    configs = []
    for i, f in enumerate(budgets):
        for criterion in CRITERIA:
            formats, objective, status = milp_select(nodes, weight_formats, act_formats, value_fn(criterion),
                                                     weight_elems, flops_proxy, f * int8_bytes, f * int8_bops)
            assert status == 0, f"{criterion} f={f}: MILP not proven optimal (status {status})"
            for name, fmt in formats.items():
                fmt["op"] = op_by_name[name]
            size, bops = config_cost(formats, weight_elems, flops_proxy)
            configs.append({"id": f"{criterion}_{i:03d}", "criterion": criterion, "budget": f,
                            "cost_weight_bytes": size, "bops_proxy": bops, "size_vs_int8": size / int8_bytes,
                            "bops_vs_int8": bops / int8_bops, "objective": objective, "formats": formats})
            log(f"f={f:.2f} {criterion:<17} size {size / int8_bytes:.3f} BOPs {bops / int8_bops:.3f} x all-int8")
    (args.out_dir / "configs.json").write_text(json.dumps({"configs": configs}, indent=1) + "\n")
    log(f"wrote {len(configs)} configs to {args.out_dir / 'configs.json'}")


if __name__ == "__main__":
    main()
