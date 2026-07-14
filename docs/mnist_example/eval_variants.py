#!/usr/bin/env python3
"""
Compare 4 MNIST inference variants:
  1. fp32         — standard onnx-mlir --EmitLib (fp32 baseline)
  2. p8e1-nqdq    — full posit p8e1 (ONNX_MLIR_POSIT_FORCE_NQDQ=1)
  3. p8e1-sel     — selective posit: only Gemm_3 in p8e1, rest fp32
  4. p8e1-alps    — full posit p8e1 with ALPS weight quantization

Outputs: top-1, top-5 accuracy on MNIST test set (10,000 images), .so size.
"""

import os, sys, subprocess, time
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PP          = os.path.join(PROJECT, "docs/mnist_example/pp")
OUT         = os.path.join(PROJECT, "docs/mnist_example/pp")
ONNX        = os.path.join(PROJECT, "docs/mnist_example/mnist.onnx")
OPT         = os.path.join(PROJECT, "build/Release/bin/onnx-mlir-opt")
ONNXMLIR    = os.path.join(PROJECT, "build/Release/bin/onnx-mlir")
TRANSLATE   = "/home/hoju/test/llvm-project-1053047/build/bin/mlir-translate"
CLANGXX     = "/home/hoju/test/llvm-project-1053047/build/bin/clang++"
RUNTIME_CPP = os.path.join(PROJECT, "src/posit_runtime.cpp")
CRUNTIME    = os.path.join(PROJECT, "build/Release/lib/libcruntime.a")
SOFTPOSIT_A = os.path.join(PROJECT, "src/.deps/softposit-px1/libsoftposit.a")
SOFTPOSIT_I = os.path.join(PROJECT, "src/.deps/SoftPosit/source/include")
UNIVERSAL_I = os.path.join(PROJECT, "src/.deps/universal/include/sw")
MLIR_I      = "/home/hoju/test/llvm-project-1053047/mlir/include"
PYRUNTIME   = os.path.join(PROJECT, "build/Release/lib")
MNIST_DIR   = "/tmp/mnist_data"

os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PYRUNTIME)

# ── download MNIST ─────────────────────────────────────────────────────────────

def load_mnist_test():
    """Return (images, labels) where images: (10000,1,28,28) float32 normalised."""
    import torchvision, torchvision.transforms as T
    print("  loading via torchvision (download if needed) ...", flush=True)
    ds = torchvision.datasets.MNIST(
        MNIST_DIR, train=False, download=True,
        transform=T.Compose([T.ToTensor(),
                              T.Normalize((0.1307,), (0.3081,))]))
    imgs = np.stack([np.array(x[0]) for x in ds])   # (10000,1,28,28)
    lbls = np.array([x[1] for x in ds])
    return imgs.astype(np.float32), lbls

# ── lowering helpers ───────────────────────────────────────────────────────────
def run(cmd, env=None, check=True):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return r

def posit_lower_to_ll(onnx_mlir_in, ll_out, env=None, extra_opt_flags=None):
    """Run the 4-stage posit lowering pipeline → .ll"""
    posit   = ll_out.replace(".ll", ".posit.mlir")
    krnl    = ll_out.replace(".ll", ".krnl.mlir")
    llvm    = ll_out.replace(".ll", ".llvm.mlir")

    opt_flags = ["--shape-inference", "--convert-onnx-to-posit", "--posit-format=p8e1"]
    if extra_opt_flags:
        opt_flags = extra_opt_flags

    # Stage 1: ONNX dialect → Posit dialect
    run([OPT, onnx_mlir_in] + opt_flags + ["-o", posit], env=env)
    # Stage 2: Posit → Krnl
    run([OPT, posit,
         "--canonicalize", "--shape-inference",
         "--convert-onnx-to-krnl", "--convert-posit-to-krnl", "--canonicalize",
         "-o", krnl])
    # Stage 3: Krnl → LLVM dialect
    run([OPT, krnl,
         "--convert-krnl-to-affine", "--convert-krnl-to-llvm",
         "--reconcile-unrealized-casts",
         "-o", llvm])
    # Stage 4: LLVM dialect → LLVM IR
    run([TRANSLATE, "--mlir-to-llvmir", llvm, "-o", ll_out])
    return ll_out

def compile_so(ll_path, so_path):
    """Compile .ll + posit_runtime.cpp → .so using Universal backend."""
    rt_obj = so_path + ".runtime.o"
    common = [
        "-DPOSIT_USE_UNIVERSAL",
        "-DPOSIT_RUNTIME_SINGLE_FORMAT=1", "-DPOSIT_RUNTIME_FMT_P8E1=1",
        "-DPOSIT_BUILD_MIXED_OFF=1",
        f"-I{UNIVERSAL_I}", f"-I{MLIR_I}", "-I/usr/local/include",
    ]
    # Compile runtime
    run([CLANGXX, "-std=c++20", "-O3", "-fPIC", "-c",
         "-ffunction-sections", "-fdata-sections",
         "-fvisibility=hidden", "-fvisibility-inlines-hidden",
         RUNTIME_CPP, *common, "-o", rt_obj])
    # Link .so
    run([CLANGXX, "-std=c++20", "-O3", "-fPIC", "-shared",
         "-Wl,--gc-sections",
         ll_path, rt_obj, *common, CRUNTIME,
         "-o", so_path])
    os.remove(rt_obj)
    return so_path

# ── inference ─────────────────────────────────────────────────────────────────
def run_so(so_path, images, tag=""):
    """Run inference via OMExecutionSession. tag="" = auto-detect from filename."""
    from PyRuntime import OMExecutionSession
    if tag:
        sess = OMExecutionSession(so_path, tag)
    else:
        sess = OMExecutionSession(so_path)
    preds = []
    for img in images:
        out = sess.run([img.reshape(1, 1, 28, 28)])
        preds.append(out[0])
    return np.concatenate(preds, axis=0)

def top_k(logits, labels, k):
    topk = np.argsort(logits, axis=1)[:, -k:]
    correct = sum(labels[i] in topk[i] for i in range(len(labels)))
    return correct / len(labels)

def eval_so(so_path, images, labels, tag=""):
    t0 = time.time()
    logits = run_so(so_path, images, tag)
    elapsed = time.time() - t0
    t1 = top_k(logits, labels, 1)
    t5 = top_k(logits, labels, 5)
    sz = os.path.getsize(so_path)
    return {"top1": t1, "top5": t5, "size_kb": sz // 1024,
            "time_s": elapsed, "name": os.path.basename(so_path)}

# ── build variants ─────────────────────────────────────────────────────────────
def build_all():
    onnx_ir = os.path.join(PP, "m.onnx.mlir")

    # ── 1. fp32 baseline ──────────────────────────────────────────────────────
    fp32_so = os.path.join(OUT, "mnist_fp32.so")
    if not os.path.exists(fp32_so):
        print("[build] fp32 via onnx-mlir --EmitLib ...", flush=True)
        run([ONNXMLIR, "--EmitLib", "-o", fp32_so.replace(".so",""), ONNX])
    else:
        print("[build] fp32 .so already exists, skipping", flush=True)

    # ── 2. full p8e1 nqdq (no ALPS) ───────────────────────────────────────────
    p8e1_ll = os.path.join(OUT, "mnist_p8e1.ll")
    p8e1_so = os.path.join(OUT, "mnist_p8e1.so")
    if not os.path.exists(p8e1_so):
        print("[build] full p8e1 nqdq ...", flush=True)
        posit_lower_to_ll(
            onnx_ir, p8e1_ll,
            env={"ONNX_MLIR_POSIT_FORCE_NQDQ": "1", "POSIT_COMPACT_CONSTANTS": "1"})
        compile_so(p8e1_ll, p8e1_so)
    else:
        print("[build] p8e1 .so already exists, skipping", flush=True)

    # ── 3. selective: only Gemm_3 in p8e1 ─────────────────────────────────────
    sel_ll = os.path.join(OUT, "mnist_sel_gemm3_p8e1.ll")
    sel_so = os.path.join(OUT, "mnist_sel_gemm3_p8e1.so")
    if not os.path.exists(sel_so):
        print("[build] selective Gemm_3 p8e1 ...", flush=True)
        posit_lower_to_ll(
            onnx_ir, sel_ll,
            env={"POSIT_SELECTIVE_NODES": "Gemm_3",
                 "POSIT_COMPACT_CONSTANTS": "1"},
            extra_opt_flags=["--shape-inference", "--convert-onnx-to-posit",
                             "--posit-format=p8e1"])
        compile_so(sel_ll, sel_so)
    else:
        print("[build] sel_gemm3 .so already exists, skipping", flush=True)

    # ── 4. full p8e1 with ALPS ─────────────────────────────────────────────────
    alps_ll = os.path.join(OUT, "mnist_p8e1_alps.ll")
    alps_so = os.path.join(OUT, "mnist_p8e1_alps.so")
    if not os.path.exists(alps_so):
        print("[build] full p8e1 ALPS ...", flush=True)
        posit_lower_to_ll(
            onnx_ir, alps_ll,
            env={
                "ONNX_MLIR_POSIT_FORCE_NQDQ": "1",
                "POSIT_COMPACT_CONSTANTS": "1",
                "ONNX_MLIR_POSIT_CONST_ALPS": "1",
                "ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN": "0.0078125",
                "ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX": "16",
                "ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS": "25",
                "ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET": "1.0",
                "ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE": "0.95",
            })
        compile_so(alps_ll, alps_so)
    else:
        print("[build] alps .so already exists, skipping", flush=True)

    return fp32_so, p8e1_so, sel_so, alps_so

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== loading MNIST test set ===")
    images, labels = load_mnist_test()
    print(f"  loaded {len(images)} test images")

    print("\n=== building variants ===")
    fp32_so, p8e1_so, sel_so, alps_so = build_all()

    print("\n=== evaluating ===")
    # (label, so_path, tag) — fp32 auto-detects tag from filename; posit uses "m"
    variants = [
        ("fp32",           fp32_so, ""),
        ("p8e1-nqdq",      p8e1_so, "m"),
        ("p8e1-sel-Gemm3", sel_so,  "m"),
        ("p8e1-alps",      alps_so, "m"),
    ]

    rows = []
    for label, so, tag in variants:
        print(f"  [{label}] running inference on {len(images)} images ...", flush=True)
        r = eval_so(so, images, labels, tag)
        r["label"] = label
        rows.append(r)
        print(f"    top-1={r['top1']*100:.2f}%  top-5={r['top5']*100:.2f}%"
              f"  size={r['size_kb']}KB  time={r['time_s']:.1f}s")

    print("\n=== summary ===")
    print(f"{'variant':<22} {'top-1':>8} {'top-5':>8} {'size(KB)':>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r['label']:<22} {r['top1']*100:>7.2f}% {r['top5']*100:>7.2f}% {r['size_kb']:>10}")
