#!/usr/bin/env python3
"""Per-(node, format) perturbation estimate, replacing HTQ/utils/pareto_utils.py's
compute_perturbation. HTQ's own function fits one symmetric affine/uniform
quantizer to the tensor regardless of target format -- correct for int, wrong
for f16/bf16/posit (non-uniform / differently-shaped rounding). This instead
reuses eval_random_mixed_precision.py's own format-correct fake-quant
primitives (np_fake_quant_weight, and the same per-family branches
simulate_in_onnxruntime already uses for activations) and takes the same final
L2-norm reduction HTQ's compute_perturbation uses.

Perturbation is computed once per (node, format) here -- not once per tree
branch -- exactly like HTQ/pareto_frontier.py precomputes its sen_result/
sen_a_result tables before the tree search starts.
"""
import numpy as np

from eval_random_mixed_precision import int_qparams, np_fake_quant_weight


def weight_perturbation(weight, fmt, posit_oracle):
    if fmt == "fp32":
        return 0.0
    q = np_fake_quant_weight(weight, fmt, posit_oracle)
    return float(np.linalg.norm(weight.astype(np.float64) - q.astype(np.float64)))


def activation_perturbation(samples, fmt, lo_hi, posit_oracle):
    """samples: a real activation tensor's calibration-set values (from
    calibrate_ranges()'s own tapped tensors). lo_hi: that tensor's (min, max)
    over the calibration set, needed for int formats' affine scale/zero-point
    (mirrors simulate_in_onnxruntime's fake_quant int branch)."""
    if fmt == "fp32":
        return 0.0
    samples = np.asarray(samples, np.float32)
    if fmt.startswith("int"):
        bits = int(fmt[3:])
        scale, zp = int_qparams(*lo_hi, bits)
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        q = (np.clip(np.round(samples / scale + zp), qmin, qmax) - zp) * scale
    elif fmt == "f16":
        q = samples.astype(np.float16).astype(np.float32)
    elif fmt == "bf16":
        import torch
        q = torch.from_numpy(samples).to(torch.bfloat16).to(torch.float32).numpy()
    elif fmt.startswith("p"):
        q = posit_oracle(samples, fmt)
    else:
        raise ValueError(fmt)
    return float(np.linalg.norm(samples.astype(np.float64) - q.astype(np.float64)))


def build_perturbation_tables(nodes, formats, weights_by_node, activation_samples_by_tensor,
                               ranges, posit_oracle):
    """nodes: supported_nodes() output. weights_by_node: {node_name: np.ndarray}
    for weight-bearing nodes only. activation_samples_by_tensor:
    {tensor_name: np.ndarray} (a stacked batch of calibration-set values) for
    every tensor in every node's act_tensors. Returns
    (weight_table, activation_table), each {node_name: {format: perturbation}};
    weight_table only has entries for nodes with a weight_const.
    """
    weight_table, activation_table = {}, {}
    for n in nodes:
        name = n["name"]
        if n.get("weight_const"):
            w = weights_by_node[name]
            weight_table[name] = {fmt: weight_perturbation(w, fmt, posit_oracle) for fmt in formats}
        # One activation tensor's samples represent the whole node for pruning
        # purposes (its own output, the one every downstream op actually
        # consumes) -- matches how eval_random_mixed_precision.py itself
        # calibrates ranges per-tensor, not per-node.
        out_tensor = n["act_tensors"][-1]
        samples = activation_samples_by_tensor[out_tensor]
        lo_hi = ranges[out_tensor]
        activation_table[name] = {
            fmt: activation_perturbation(samples, fmt, lo_hi, posit_oracle) for fmt in formats
        }
    return weight_table, activation_table
