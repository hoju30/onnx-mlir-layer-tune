"""Per-(node, weight format, activation format) output-noise table: the SQNR selection criterion.

For every weight-bearing pass-supported node and every (w, a) in the menu, simulate the model in
onnxruntime with only that node quantized (simulate_in_onnxruntime: the pass's own fake-quant arithmetic)
on the calibration images, and record N = sum ||logits - logits_fp32||^2. A config's criterion is the sum
of its nodes' N: the total logit noise power if each node's noise simply added, i.e. -SQNR up to a constant.
Prototyped in /home/hoju/test/HTQ (decision_compare.py), where it beat htq_shared / split_connection /
split_direct on average for resnet18 and mobilenetv2 at the exact optimum of each criterion.

Activation-only nodes are not measured: the cost model (pareto_search, mp_milp) charges them no bytes and
no BOPs, so every search leaves them at fp32 anyway. Posit activations cannot be in the menu: onnxruntime
has no way to simulate them.
"""
import numpy as np

from eval_random_mixed_precision import simulate_in_onnxruntime


def sqnr_table(model, nodes, weight_formats, act_formats, ranges, calib_inputs, posit_oracle, log=print):
    """{node_name: {"w|a": N or None}} for weight-bearing nodes; None marks a (w, a) whose simulated logits
    were not finite, which a search must never pick."""
    if any(a.startswith("p") for a in act_formats):
        raise ValueError("posit activations cannot be simulated in onnxruntime; drop them from the menu")
    reference = simulate_in_onnxruntime(model, nodes, {}, ranges, calib_inputs, posit_oracle).astype(np.float64)
    weighted = [n for n in nodes if n.get("weight_const")]
    table = {}
    for k, n in enumerate(weighted):
        row = {}
        for w in weight_formats:
            for a in act_formats:
                if w == "fp32" and a == "fp32":
                    row[f"{w}|{a}"] = 0.0
                    continue
                cfg = {n["name"]: {"w": w, "a": a}}
                z = simulate_in_onnxruntime(model, nodes, cfg, ranges, calib_inputs, posit_oracle)
                row[f"{w}|{a}"] = float(((z - reference) ** 2).sum()) if np.isfinite(z).all() else None
        table[n["name"]] = row
        log(f"  [{k + 1}/{len(weighted)}] {n['name']}: {sum(v is None for v in row.values())} non-finite options")
    return table
