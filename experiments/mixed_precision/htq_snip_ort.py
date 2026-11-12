#!/usr/bin/env python3
"""Standalone SNIP-style sensitivity computation via onnxruntime-training's
on-device-training artifacts, entirely at the ONNX graph level -- no PyTorch
model, no PyTorch<->ONNX node-name mapping (supersedes the earlier
htq_sensitivity.py + htq_node_mapping.py approach).

Runs under a SEPARATE venv (../../venv-ort-training/, sibling to the repo's
own venv/) because onnxruntime-training-cpu is an alternative distribution of
the *same* `onnxruntime` package namespace the rest of this repo's experiments
use (confirmed by inspecting the wheel: both ship files under a top-level
`onnxruntime/` dir), so installing it into the main venv would corrupt
whichever one installed second rather than coexisting. sample_configs_htq.py
calls this script as a subprocess with that venv's python interpreter and
reads back the JSON it writes.

Three modes, all computed from ONE shared graph and ONE forward+backward pass
per calibration image (validated first in PyTorch directly on HTQ/sensitivity.py
before porting here -- see /home/hoju/test/HTQ/test_sensitivity_direct_grad.py):

- --mode shared (baseline, HTQ's own mechanism): only the WEIGHT gets a
  MULTIPLICATIVE mask (Mul(weight, mask), mask all-ones); the resulting score
  is mean(abs(mask.grad)) = mean(abs(weight * dL/dweight)) -- SNIP's
  connection sensitivity. Reused for BOTH the weight and activation
  perturbation terms downstream (no separate activation sensitivity).
- --mode split_connection: in ADDITION to the weight mask, every supported
  node's own input activation tensor(s) also get an independent MULTIPLICATIVE
  mask (binary elementwise ops -- Add/Sub/Mul/Div -- get one per operand,
  everything else gets one), giving activation * dL/dactivation as a SEPARATE
  connection-sensitivity score per node.
- --mode split_direct: raw dL/dW and dL/dA, NOT multiplied by the tensor's own
  value.
    - weight: no wrapper node needed at all -- the weight initializer itself is
      marked requires_grad directly, so its own gradient IS dL/dW.
    - activation: an ADDITIVE, ZERO-initialized mask (Add(activation, mask))
      instead of a multiplicative one. d(activation+mask)/dmask is identically
      1, so mask.grad IS dL/dactivation directly, at any mask value -- unlike
      recovering dL/dA from the connection-sensitivity product by dividing by
      A, which blows up (0/0) at every exactly-zero activation (extremely
      common right after Relu/Clip).

The exported graphs this repo works with have a fixed batch dimension of 1 (no
dynamic_axes), so unlike HTQ's original single multi-image mini-batch, this
runs one forward+backward per calibration image (module.lazy_reset_grad()
between each) and averages each mask's per-image mean(abs(grad)) across the
calibration set.

Usage:
  ../../venv-ort-training/bin/python3 htq_snip_ort.py \\
      --onnx model.onnx --inputs calib_inputs.npy --labels calib_labels.npy \\
      --mode split_connection --out sensitivity.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnx.shape_inference
from onnx import helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_random_mixed_precision import BINARY_OPS, WEIGHT_OPS, supported_nodes  # noqa: E402

MODES = ("shared", "split_connection", "split_direct")


def _infer_tensor_shapes(model):
    """{tensor_name: [dims]} for every graph input and every producer's output
    (dynamic/unknown dims -- there shouldn't be any, these graphs have a fixed
    batch=1 -- fall back to 1 rather than crashing)."""
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    return shapes


def insert_masks(model, nodes, mode):
    """Mutates `model` in place. Walks every ORIGINAL node (in order,
    unsupported nodes like Clip/ReduceMean/Reshape passed through untouched)
    and, for each node in `nodes` (supported_nodes() output), splices in
    whatever `mode` needs (see module docstring for the three modes' exact
    mechanics). New mask-carrying nodes are inserted immediately before their
    consumer, not at the front: an activation mask multiplies/adds onto a
    RUNTIME tensor some earlier node produced, so it must come after that
    producer to stay topologically valid (a weight mask only depends on
    initializers, so front-insertion would also be valid for it, but the
    single unified insertion point here is simpler and correct for both).

    Returns (weight_grad_names: {node_name: tensor_name_to_read_.grad_from},
             activation_mask_names: {node_name: [mask_name, ...]}).
    weight_grad_names' values are a mask name for shared/split_connection, or
    the ORIGINAL weight initializer's own name for split_direct (no wrapper
    node -- see docstring).
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    supported_by_name = {n["name"]: n for n in nodes}
    need_activation = mode in ("split_connection", "split_direct")
    shapes = _infer_tensor_shapes(model) if need_activation else {}

    weight_grad_names = {}
    activation_mask_names = {}
    new_nodes = []
    for onnx_node in list(graph.node):
        n = supported_by_name.get(onnx_node.name)
        if n is None:
            new_nodes.append(onnx_node)
            continue

        extra_before = []
        if need_activation:
            n_act_inputs = 2 if n["op"] in BINARY_OPS else 1
            a_names = []
            for i in range(n_act_inputs):
                act_name = onnx_node.input[i]
                shape = shapes.get(act_name)
                if shape is None:
                    raise ValueError(f"no inferred shape for activation tensor {act_name!r} "
                                      f"(input {i} of node {n['name']!r}); shape inference incomplete")
                mask_name = f"htq_amask__{n['name']}__{i}"
                masked_name = f"htq_amasked__{n['name']}__{i}"
                if mode == "split_connection":
                    fill, op = np.ones(shape, dtype=np.float32), "Mul"
                else:  # split_direct: additive, zero-initialized -- see docstring
                    fill, op = np.zeros(shape, dtype=np.float32), "Add"
                graph.initializer.append(numpy_helper.from_array(fill, name=mask_name))
                extra_before.append(helper.make_node(op, [act_name, mask_name], [masked_name],
                                                       name=f"htq_amask_op__{n['name']}__{i}"))
                onnx_node.input[i] = masked_name
                a_names.append(mask_name)
            activation_mask_names[n["name"]] = a_names

        if n.get("weight_const"):
            weight_name = onnx_node.input[1]
            if mode == "split_direct":
                # No wrapper node: the weight IS the parameter we mark
                # trainable, so its own gradient is dL/dW directly.
                weight_grad_names[n["name"]] = weight_name
            else:
                w = numpy_helper.to_array(inits[weight_name])
                mask_name = f"htq_wmask__{n['name']}"
                masked_name = f"htq_wmasked__{n['name']}"
                graph.initializer.append(numpy_helper.from_array(np.ones_like(w), name=mask_name))
                extra_before.append(helper.make_node("Mul", [weight_name, mask_name], [masked_name],
                                                       name=f"htq_wmul__{n['name']}"))
                onnx_node.input[1] = masked_name
                weight_grad_names[n["name"]] = mask_name

        new_nodes.extend(extra_before)
        new_nodes.append(onnx_node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return weight_grad_names, activation_mask_names


def compute_sensitivity(onnx_path, inputs, labels, work_dir, mode):
    model = onnx.load(onnx_path)
    nodes = supported_nodes(model)
    weight_nodes = [n for n in nodes if n["op"] in WEIGHT_OPS and n.get("weight_const")]

    weight_grad_names, activation_mask_names = insert_masks(
        model, weight_nodes if mode == "shared" else nodes, mode)

    trainable = set(weight_grad_names.values())
    for names in activation_mask_names.values():
        trainable.update(names)
    frozen_params = sorted({i.name for i in model.graph.initializer} - trainable)
    onnx.checker.check_model(model)

    from onnxruntime.training import artifacts
    from onnxruntime.training.api import CheckpointState, Module

    artifacts.generate_artifacts(
        model,
        requires_grad=sorted(trainable),
        frozen_params=frozen_params,
        loss=artifacts.LossType.CrossEntropyLoss,
        optimizer=artifacts.OptimType.SGD,
        artifact_directory=work_dir,
    )
    ckpt = CheckpointState.load_checkpoint(os.path.join(work_dir, "checkpoint"))
    module = Module(os.path.join(work_dir, "training_model.onnx"), ckpt,
                     os.path.join(work_dir, "eval_model.onnx"))

    running_w = dict.fromkeys(weight_grad_names, 0.0)
    running_a = dict.fromkeys(activation_mask_names, 0.0)
    module.train()
    for i in range(len(inputs)):
        module.lazy_reset_grad()
        loss = module(inputs[i:i + 1], labels[i:i + 1])
        print(f"  image {i}: loss={loss}", flush=True)
        for node_name, grad_name in weight_grad_names.items():
            grad = np.asarray(ckpt.parameters[grad_name].grad)
            running_w[node_name] += float(np.abs(grad).mean())
        for node_name, mask_list in activation_mask_names.items():
            vals = [float(np.abs(np.asarray(ckpt.parameters[m].grad)).mean()) for m in mask_list]
            running_a[node_name] += sum(vals) / len(vals)

    n = len(inputs)
    weight_sensitivity = {name: total / n for name, total in running_w.items()}
    if mode == "shared":
        activation_sensitivity = dict(weight_sensitivity)
    else:
        activation_sensitivity = {name: total / n for name, total in running_a.items()}
    return weight_sensitivity, activation_sensitivity


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--inputs", required=True, help=".npy, float32 [N,C,H,W]")
    ap.add_argument("--labels", required=True, help=".npy, int64 [N]")
    ap.add_argument("--mode", choices=MODES, default="split_connection",
                     help="shared = HTQ's original weight-only connection sensitivity, reused "
                          "for both terms; split_connection = independent weight*dL/dweight and "
                          "activation*dL/dactivation; split_direct = independent raw dL/dW and dL/dA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=None, help="default: a temp dir, cleaned up on exit")
    args = ap.parse_args()

    inputs = np.load(args.inputs).astype(np.float32)
    labels = np.load(args.labels).astype(np.int64)
    work = args.work_dir or tempfile.mkdtemp(prefix="htq-snip-ort-")
    os.makedirs(work, exist_ok=True)
    try:
        weight_sensitivity, activation_sensitivity = compute_sensitivity(
            args.onnx, inputs, labels, work, args.mode)
    finally:
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)

    with open(args.out, "w") as f:
        json.dump({"weight": weight_sensitivity, "activation": activation_sensitivity}, f, indent=1)
    print(f"wrote sensitivity for {len(weight_sensitivity)} weight-bearing nodes and "
          f"{len(activation_sensitivity)} nodes' activations to {args.out} (mode={args.mode})", flush=True)


if __name__ == "__main__":
    main()
