#!/usr/bin/env python3
"""
4-group MNIST posit precision experiment:
  A: fp32                      — baseline, every layer fp32
  B: GEMM1 posit8,1            — only Gemm_3 (fc1) in posit8,1, rest fp32
  C: GEMM2 posit8,1            — only Gemm_5 (fc2) in posit8,1, rest fp32
  D: non-param layers posit8,1 — MaxPool_0, Reshape_2, Relu_4, Softmax_6 in posit8,1
                                  (Softmax_6 has no posit lowering pattern in this
                                   branch, so it stays fp32 regardless of selection)

Metrics per group: top-1 accuracy, top-5 accuracy, weight size, peak activation memory.
Weight size is computed analytically as num_elements x bit_width / 8, per the
precision-cost formula in claude.md section 16.

Peak activation memory follows the standard sequential-inference convention (as
used e.g. in MCUNet/TinyEngine): at the point op_i executes, its input tensor
(op_i-1's output) and its output tensor must both be live at once, and everything
else can be freed — so peak = max over ops of (bytes(input_i) + bytes(output_i)).
This is NOT measured from .so file size, which contains runtime code and doesn't
reflect activation memory at all.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_variants import (  # noqa: E402  (path insert must run first)
    PP, OUT, ONNX, ONNXMLIR,
    load_mnist_test, posit_lower_to_ll, compile_so, eval_so, run,
)

# ── static graph description (from docs/mnist_example/pp/m.onnx.mlir) ──────────
# name -> (weight_element_count, output_activation_element_count, posit_lowering_supported)
# Ops are listed in graph order — needed for the peak-activation sliding window below.
NODES = {
    "MaxPool_0": (0,           1 * 1 * 14 * 14, True),   # MaxPoolSingleOut
    "Reshape_2": (0,           1 * 196,         True),
    "Gemm_3":    (128 * 196 + 128, 1 * 128,     True),   # fc1: weight 128x196 + bias 128
    "Relu_4":    (0,           1 * 128,         True),
    "Gemm_5":    (10 * 128 + 10,   1 * 10,      True),   # fc2: weight 10x128 + bias 10
    "Softmax_6": (0,           1 * 10,          False),  # no ONNXSoftmaxOpLowering pattern
}

# model input tensor ("image"): 1x1x28x28, always fp32 — no op produces it, so it's
# never posit-converted in any of the 4 groups.
INPUT_ELEMS = 1 * 1 * 28 * 28

# label -> (so basename, selective node set)
GROUPS = {
    "A_fp32":          ("mnist_fp32",            set()),
    "B_gemm1_p8e1":    ("mnist_sel_gemm3_p8e1",  {"Gemm_3"}),
    "C_gemm2_p8e1":    ("mnist_sel_gemm5_p8e1",  {"Gemm_5"}),
    "D_nonparam_p8e1": ("mnist_sel_nonparam_p8e1",
                        {"MaxPool_0", "Reshape_2", "Relu_4", "Softmax_6"}),
}


def compute_cost(selective_nodes):
    """Analytical (weight_bytes, peak_activation_bytes) for a given selective-node set."""
    weight_bytes = 0.0
    bits_of = {}
    for name, (n_weight, n_act, supported) in NODES.items():
        is_posit = supported and name in selective_nodes
        bits = 8 if is_posit else 32
        weight_bytes += n_weight * bits / 8
        bits_of[name] = bits

    # tensor chain: input image -> MaxPool_0 out -> Reshape_2 out -> ... -> Softmax_6 out
    order = list(NODES.keys())
    elem_chain = [INPUT_ELEMS] + [NODES[n][1] for n in order]
    bits_chain = [32] + [bits_of[n] for n in order]  # input tensor is always fp32

    peak_bytes = 0.0
    for i in range(1, len(elem_chain)):
        pair_bytes = (elem_chain[i - 1] * bits_chain[i - 1]
                      + elem_chain[i] * bits_chain[i]) / 8
        peak_bytes = max(peak_bytes, pair_bytes)

    return weight_bytes, peak_bytes


def build_group(label, so_base, selective_nodes):
    onnx_ir = os.path.join(PP, "m.onnx.mlir")
    so_path = os.path.join(OUT, so_base + ".so")

    if os.path.exists(so_path):
        print(f"[build] {label}: {so_base}.so already exists, skipping", flush=True)
        return so_path, ("" if label == "A_fp32" else "m")

    if label == "A_fp32":
        print(f"[build] {label}: onnx-mlir --EmitLib ...", flush=True)
        run([ONNXMLIR, "--EmitLib", "-o", so_path.replace(".so", ""), ONNX])
        return so_path, ""

    ll_path = os.path.join(OUT, so_base + ".ll")
    print(f"[build] {label}: selective posit8,1 on {sorted(selective_nodes)} ...",
          flush=True)
    posit_lower_to_ll(
        onnx_ir, ll_path,
        env={"POSIT_SELECTIVE_NODES": ",".join(sorted(selective_nodes)),
             "POSIT_COMPACT_CONSTANTS": "1"},
        extra_opt_flags=["--shape-inference", "--convert-onnx-to-posit",
                          "--posit-format=p8e1"])
    compile_so(ll_path, so_path)
    return so_path, "m"


if __name__ == "__main__":
    print("=== loading MNIST test set ===")
    images, labels = load_mnist_test()
    print(f"  loaded {len(images)} test images")

    rows = []
    for label, (so_base, selective_nodes) in GROUPS.items():
        print(f"\n=== group {label} ===")
        so_path, tag = build_group(label, so_base, selective_nodes)
        weight_bytes, peak_act_bytes = compute_cost(selective_nodes)
        r = eval_so(so_path, images, labels, tag)
        r["label"] = label
        r["weight_kb"] = weight_bytes / 1024
        r["peak_act_kb"] = peak_act_bytes / 1024
        rows.append(r)
        print(f"  top-1={r['top1']*100:.2f}%  top-5={r['top5']*100:.2f}%"
              f"  weight={r['weight_kb']:.2f}KB  peak_activation={r['peak_act_kb']:.3f}KB"
              f"  (.so size={r['size_kb']}KB)")

    print("\n=== summary ===")
    hdr = f"{'group':<20} {'top-1':>8} {'top-5':>8} {'weight(KB)':>12} {'peak_act(KB)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<20} {r['top1']*100:>7.2f}% {r['top5']*100:>7.2f}% "
              f"{r['weight_kb']:>12.2f} {r['peak_act_kb']:>14.3f}")
