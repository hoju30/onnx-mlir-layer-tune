"""Real-compile lowering helpers for the low-precision variant on arbitrary
ONNX models (ResNet18/MobileNetV2-scale). Adds two things the MNIST-era
eval_variants.py's posit_lower_to_ll didn't need: --decompose-onnx
--conv-opt-onnx before the LOWP conversion (Conv-heavy nets need the
Convolution-Optimization-for-CPU pass to get a fast Conv lowering at all --
confirmed experimentally: identical bf16 timing with/without it, but FP32
through this same pipeline matches the standard onnx-mlir --EmitLib driver's
speed only once decompose+conv-opt is applied) and --O3 on the ONNX-to-Krnl
stage (enables tiling; --march does NOT help bf16/f16 specifically -- see
project notes, their Gemm/Conv/MatMul Krnl lowering has no vectorized bf16
kernel at all, only FP32/FP64, so bf16/f16 stay near-scalar-loop slow
regardless of SIMD flags. int8/fp8 use their own dedicated
QLinearConv/QLinearMatMul kernels and are fast.).
"""
import os
import subprocess

# Some (model, format-combination) inputs make a specific compiler pass hang
# indefinitely rather than fail -- confirmed directly: onnx-mlir-opt
# --convert-krnl-to-affine --convert-krnl-to-llvm on MobileNetV2's
# rep_uniform_int8 config ran for 18.5 CPU-HOURS with zero progress before
# being killed manually. Every subprocess call here MUST have a hard
# timeout so one pathological config can't block the entire pipeline
# forever -- a timeout raises subprocess.TimeoutExpired, which callers
# should catch and treat as "skip this config", not silently hang.
COMPILE_TIMEOUT_S = 600


def run(cmd, env=None, check=True, timeout=COMPILE_TIMEOUT_S):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return r


def import_and_convopt(onnx_path, out_dir, onnx_mlir_bin, opt_bin, tag="model"):
    """Import ONNX -> ONNX Dialect, then apply decompose+conv-opt ONCE
    (topology/node-names are shared by every config, so this only needs to
    run once per model, not per sampled configuration)."""
    base = os.path.join(out_dir, tag)
    run([onnx_mlir_bin, "--EmitONNXIR", "-o", base,
         "--mlir-elide-resource-strings-if-larger=1000000000",
         "--mlir-elide-elementsattrs-if-larger=1000000000", onnx_path])
    onnx_mlir_path = base + ".onnx.mlir"
    convopt_path = os.path.join(out_dir, f"{tag}.convopt.mlir")
    run([opt_bin, onnx_mlir_path, "--shape-inference", "--decompose-onnx",
         "--conv-opt-onnx", "--shape-inference", "--canonicalize",
         "-o", convopt_path])
    return convopt_path


def lower_and_compile(convopt_path, lowp_node_formats, out_base,
                      opt_bin, translate_bin, clangxx_bin, cruntime):
    """LOWP conversion (skipped if lowp_node_formats is empty, i.e. an
    all-FP32 configuration) -> Krnl (--O3) -> LLVM dialect -> LLVM IR ->
    shared library."""
    lowp_mlir = out_base + ".lowp.mlir"
    krnl_mlir = out_base + ".krnl.mlir"
    llvm_mlir = out_base + ".llvm.mlir"
    ll_path = out_base + ".ll"
    so_path = out_base + ".so"

    if os.path.exists(so_path):
        # a prior run got interrupted after compiling but before eval/manifest
        # write finished -- the compile (the expensive step under contention,
        # 90-270s+) doesn't need to be redone.
        return so_path

    src_for_krnl = convopt_path
    if lowp_node_formats:
        run([opt_bin, convopt_path, "--shape-inference",
             "--convert-onnx-to-lowprecision", "-o", lowp_mlir],
            env={"LOWP_NODE_FORMATS": lowp_node_formats})
        src_for_krnl = lowp_mlir

    run([opt_bin, src_for_krnl, "--canonicalize", "--shape-inference", "--O3",
         "--convert-onnx-to-krnl", "--canonicalize", "-o", krnl_mlir])
    run([opt_bin, krnl_mlir, "--convert-krnl-to-affine", "--convert-krnl-to-llvm",
         "--reconcile-unrealized-casts", "-o", llvm_mlir])
    run([translate_bin, "--mlir-to-llvmir", llvm_mlir, "-o", ll_path])
    run([clangxx_bin, "-std=c++20", "-O3", "-fPIC", "-shared",
         ll_path, cruntime, "-o", so_path])
    return so_path
