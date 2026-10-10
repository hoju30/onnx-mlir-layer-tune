#!/usr/bin/env python3
"""Runtime tests for --convert-onnx-to-mixed-precision.

Unlike the lit tests (which check the IR the pass produces), these compile the
rewritten model all the way to a shared library, EXECUTE it, and compare the
numbers against independent references:

  * INT / f16 / bf16 references are recomputed with numpy (float32 rules),
  * posit references come from sw::universal::posit<N,ES> (the same library the
    Posit runtime is built on), through a tiny helper compiled on the fly.

Why this exists: two of the worst bugs in this pass were invisible to
"looks right in the IR" checks -- the posit activation round trip was silently
folded away later in the pipeline (compiled model bit-identical to FP32), and
the posit weight table disagreed with Universal. Only executing the result
catches that class of bug.

Run (from this directory):
    python3 -m unittest -v test_mixed_precision_runtime

Needs numpy, a built onnx-mlir (onnx-mlir, onnx-mlir-opt, PyRuntimeC,
libcruntime.a) and the LLVM build it was configured against (mlir-translate,
clang++). Tool locations are discovered from <build>/CMakeCache.txt; override
with ONNX_MLIR_ROOT / ONNX_MLIR_BUILD / LLVM_BUILD_BIN / UNIVERSAL_ROOT.
Anything missing => the affected tests are SKIPPED with the reason, never
silently passed. The posit tests additionally need the Universal headers.
"""
import glob
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
f32 = np.float32 if np is not None else None

# --------------------------------------------------------------------------- #
# Environment discovery
# --------------------------------------------------------------------------- #
ROOT = os.environ.get("ONNX_MLIR_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.environ.get("ONNX_MLIR_BUILD") or os.path.join(ROOT, "build")


def _cache_var(name):
    try:
        with open(os.path.join(BUILD, "CMakeCache.txt")) as f:
            for line in f:
                m = re.match(rf"^{name}:[A-Z]+=(.*)$", line.strip())
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def _first(*paths):
    return next((p for p in paths if p and os.path.exists(p)), None)


def _discover():
    d = {}
    bins = glob.glob(os.path.join(BUILD, "*", "bin"))
    d["bin"] = next((b for b in bins if os.path.exists(os.path.join(b, "onnx-mlir-opt"))), None)
    d["lib"] = os.path.join(os.path.dirname(d["bin"]), "lib") if d["bin"] else None
    llvm_dir = _cache_var("LLVM_DIR")
    mlir_dir = _cache_var("MLIR_DIR")
    llvm_bin = os.environ.get("LLVM_BUILD_BIN") or (
        os.path.abspath(os.path.join(llvm_dir, "..", "..", "..", "bin")) if llvm_dir else None)
    d["translate"] = _first(llvm_bin and os.path.join(llvm_bin, "mlir-translate"))
    d["clangxx"] = _first(llvm_bin and os.path.join(llvm_bin, "clang++"), "/usr/bin/clang++")
    d["mlir_inc"] = (os.path.abspath(os.path.join(mlir_dir, "..", "..", "..", "..", "mlir", "include"))
                     if mlir_dir else None)
    uni = os.environ.get("UNIVERSAL_ROOT") or _cache_var("UNIVERSAL_ROOT")
    d["universal_inc"] = os.path.join(uni, "include", "sw") if uni else None
    d["posit_rt"] = os.path.join(ROOT, "src", "posit_runtime.cpp")
    d["mnist"] = os.path.join(ROOT, "docs", "mnist_example", "mnist.onnx")
    return d


ENV = _discover()
_TMP = None
_RT_OBJ = {}
_ORACLE = None


def _missing(*keys):
    return [k for k in keys if not ENV.get(k) or not os.path.exists(ENV[k])]


def require(*keys):
    if np is None:
        raise unittest.SkipTest("numpy not installed")
    miss = _missing(*keys)
    if miss:
        raise unittest.SkipTest("missing prerequisite(s): " + ", ".join(miss))
    if not os.path.exists(os.path.join(ENV["lib"], "libcruntime.a")):
        raise unittest.SkipTest("libcruntime.a not built")
    if not glob.glob(os.path.join(ENV["lib"], "PyRuntimeC*.so")):
        raise unittest.SkipTest("PyRuntimeC not built (make PyRuntimeC)")


def tmpdir():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="mixedprecision-rt-")
    return _TMP


def run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-2500:]}")
    return r


# --------------------------------------------------------------------------- #
# Building and running a model
# --------------------------------------------------------------------------- #
HDR = ('module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", '
       'llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "t"} {')
POSIT_MACRO = {"p4e1": "POSIT_RUNTIME_FMT_P4E1", "p6e1": "POSIT_RUNTIME_FMT_P6E1", "p8e1": "POSIT_RUNTIME_FMT_P8E1"}


def wrap(func_text):
    return f'{HDR}\n{func_text}\n  "onnx.EntryPoint"() <{{func = @main_graph}}> : () -> ()\n}}\n'


def hexblob(a):
    return "0x" + np.ascontiguousarray(a, dtype="<f4").tobytes().hex().upper()


def _posit_runtime_obj(macro):
    """posit_runtime.cpp is huge; build each single-format object once."""
    if macro not in _RT_OBJ:
        common = ["-DPOSIT_USE_UNIVERSAL", "-DPOSIT_RUNTIME_SINGLE_FORMAT=1", f"-D{macro}=1",
                  "-DPOSIT_BUILD_MIXED_OFF=1", f"-I{ENV['universal_inc']}", f"-I{ENV['mlir_inc']}",
                  "-I/usr/local/include"]
        obj = os.path.join(tmpdir(), f"posit_rt_{macro}.o")
        run([ENV["clangxx"], "-std=c++20", "-O3", "-fPIC", "-c", ENV["posit_rt"], *common, "-o", obj])
        _RT_OBJ[macro] = (obj, common)
    return _RT_OBJ[macro]


def build_so(tag, mlir_text, node_formats, posit_fmt=None, extra_env=None):
    """ONNX-dialect module -> mixed-precision pass -> Krnl -> LLVM -> .so."""
    opt = os.path.join(ENV["bin"], "onnx-mlir-opt")
    d = tmpdir()
    src, mixed, krnl, llvm, ll, so = (os.path.join(d, f"{tag}.{x}") for x in
                                      ["onnx.mlir", "mixed.mlir", "krnl.mlir", "llvm.mlir", "ll", "so"])
    with open(src, "w") as f:
        f.write(mlir_text)
    run([opt, src, "--shape-inference", "--convert-onnx-to-mixed-precision", "-o", mixed],
        env={"MIXEDP_NODE_FORMATS": node_formats, **(extra_env or {})})
    # --convert-posit-to-krnl is required whenever a posit format is used (docs/MixedPrecisionFormats.md).
    run([opt, mixed, "--canonicalize", "--shape-inference", "--convert-onnx-to-krnl",
         "--convert-posit-to-krnl", "--canonicalize", "-o", krnl])
    run([opt, krnl, "--convert-krnl-to-affine", "--convert-krnl-to-llvm",
         "--reconcile-unrealized-casts", "-o", llvm])
    run([ENV["translate"], "--mlir-to-llvmir", llvm, "-o", ll])
    cruntime = os.path.join(ENV["lib"], "libcruntime.a")
    if posit_fmt:
        obj, common = _posit_runtime_obj(POSIT_MACRO[posit_fmt])
        run([ENV["clangxx"], "-std=c++20", "-O3", "-fPIC", "-shared", ll, obj, *common, cruntime, "-o", so])
    else:
        run([ENV["clangxx"], "-std=c++20", "-O3", "-fPIC", "-shared", ll, cruntime, "-o", so])
    return so


def execute_outputs(so, *inputs):
    sys.path.insert(0, ENV["lib"])
    from PyRuntime import OMExecutionSession
    return OMExecutionSession(so, "t").run([np.ascontiguousarray(x, dtype=f32) for x in inputs])


def execute(so, *inputs):
    return execute_outputs(so, *inputs)[0]


# --------------------------------------------------------------------------- #
# Independent references
# --------------------------------------------------------------------------- #
def q_int_act(x, s, zp, bits):
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    t = np.rint(np.asarray(x, dtype=f32) / f32(s) + f32(zp))          # onnx.Round: ties to even
    return ((np.clip(t, f32(qmin), f32(qmax)) - f32(zp)) * f32(s)).astype(f32)


def q_int_weight(w, bits):
    w = np.asarray(w, dtype=f32)
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    fin = np.isfinite(w)
    if not fin.any():
        return w.copy()
    scale = f32(np.abs(w[fin]).max()) / f32(qmax)
    if not np.isfinite(scale) or scale == 0:
        return w.copy()
    t = (w[fin] / scale).astype(np.float64)
    q = np.where(t >= 0, np.floor(t + 0.5), np.ceil(t - 0.5))          # std::lround: half away from zero
    with np.errstate(over="ignore"):
        dq = (np.clip(q, qmin, qmax).astype(f32) * scale).astype(f32)
    out = w.copy()
    out[fin] = np.where(np.isfinite(dq), dq, w[fin])
    return out


def q_bf16(x):
    u = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype("<u4").view("<f4")


def q_f16(x):
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=f32).astype(np.float16).astype(f32)


_ORACLE_SRC = r'''
#include <universal/number/posit/posit.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
using sw::universal::posit;
template <int N, int ES> void go(const char *i, const char *o) {
  FILE *f = fopen(i, "rb"); std::vector<float> v; float x;
  while (fread(&x, 4, 1, f) == 1) v.push_back(x); fclose(f);
  for (auto &e : v) { posit<N, ES> p; p = e; e = static_cast<float>(p); }
  f = fopen(o, "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f);
}
int main(int, char **a) {
  int n = atoi(a[1]), es = atoi(a[2]);
  if (n == 4 && es == 1) go<4, 1>(a[3], a[4]);
  else if (n == 6 && es == 1) go<6, 1>(a[3], a[4]);
  else if (n == 8 && es == 1) go<8, 1>(a[3], a[4]);
  else return 1;
}
'''


def q_posit(x, fmt):
    """Universal's float -> posit<N,ES> -> float, elementwise."""
    global _ORACLE
    if _ORACLE is None:
        src = os.path.join(tmpdir(), "oracle.cpp")
        with open(src, "w") as f:
            f.write(_ORACLE_SRC)
        _ORACLE = os.path.join(tmpdir(), "oracle")
        run([ENV["clangxx"], "-std=c++20", "-O2", f"-I{ENV['universal_inc']}", src, "-o", _ORACLE])
    n, es = int(fmt[1]), int(fmt[3])
    x = np.ascontiguousarray(x, dtype="<f4")
    fi, fo = os.path.join(tmpdir(), "o_in.bin"), os.path.join(tmpdir(), "o_out.bin")
    x.tofile(fi)
    run([_ORACLE, str(n), str(es), fi, fo])
    return np.fromfile(fo, dtype="<f4").reshape(x.shape)


def quant(kind, x, fmt, params=None):
    """kind 'w' = weight, 'a' = activation. params = (scale, zero_point) for INT activations."""
    if fmt.startswith("int"):
        bits = int(fmt[3:])
        return q_int_weight(x, bits) if kind == "w" else q_int_act(x, params[0], params[1], bits)
    if fmt == "bf16":
        return q_bf16(x)
    if fmt == "f16":
        return q_f16(x)
    if fmt.startswith("p"):
        return q_posit(x, fmt)
    return np.asarray(x, dtype=f32)


def posit_fmt_of(*fmts):
    return next((f for f in fmts if f.startswith("p")), None)


def bits_of(a):
    b = np.ascontiguousarray(a, dtype="<f4").view("<u4").copy()
    b[np.asarray(a) == 0] = 0                                           # -0 == +0
    return b


def assert_bit_exact(tc, got, expected, label, mask=None):
    got, expected = np.asarray(got), np.asarray(expected)
    tc.assertEqual(got.shape, expected.shape, f"{label}: output shape")
    got, expected = got.reshape(-1), expected.reshape(-1)
    mask = np.ones(got.shape, bool) if mask is None else mask
    both_nan = np.isnan(got) & np.isnan(expected)
    bad = (bits_of(got) != bits_of(expected)) & ~both_nan & mask
    tc.assertEqual(int(bad.sum()), 0,
                   f"{label}: {int(bad.sum())}/{int(mask.sum())} mismatches, e.g. "
                   f"got={got[bad][:3]} expected={expected[bad][:3]}")


# --------------------------------------------------------------------------- #
# 1. Activation path, bit-exact (Relu: the pass quantizes its operand and result)
# --------------------------------------------------------------------------- #
ACT_XS, ACT_XZP, ACT_YS, ACT_YZP = 0.37, 3, 0.61, -2


def _relu_module(k):
    return wrap(f'''  func.func @main_graph(%arg0: tensor<1x{k}xf32> {{onnx.name = "x"}}) -> (tensor<1x{k}xf32> {{onnx.name = "y"}}) {{
    %0 = "onnx.Relu"(%arg0) {{onnx_node_name = "Relu_1"}} : (tensor<1x{k}xf32>) -> tensor<1x{k}xf32>
    return %0 : tensor<1x{k}xf32>
  }}''')


def _activation_inputs(seed=7):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(0, 2, 40000), rng.uniform(-1000, 1000, 20000), rng.normal(0, 1e-3, 10000),
        rng.uniform(-3.0e4, 3.0e4, 10000),
        np.array([0.0, -0.0, 1e-30, -1e-30, 65504.0, -65504.0, 1.0, -1.0])]).astype(f32)


def _mul_one_module(k):
    """Mul(x, 1.0): with a=p<N>e<ES> the pass rounds x, the exact constant 1.0 and the product; posit
    rounding is idempotent, so y = q(x) for every input -- negatives included, unlike a Relu."""
    return wrap(f'''  func.func @main_graph(%arg0: tensor<1x{k}xf32> {{onnx.name = "x"}}) -> (tensor<1x{k}xf32> {{onnx.name = "y"}}) {{
    %one = onnx.Constant dense<1.0> : tensor<f32>
    %0 = "onnx.Mul"(%arg0, %one) {{onnx_node_name = "Mul_1"}} : (tensor<1x{k}xf32>, tensor<f32>) -> tensor<1x{k}xf32>
    return %0 : tensor<1x{k}xf32>
  }}''')


_CORPUS = {}


def posit_corpus(bits):
    """(inputs, Universal's float -> posit<bits,1> -> float of them): the corpus
    test_mixed_precision_weights.py checks the folded posit weights on, from posit_weight_reference.cpp
    (every encoding and its float neighbours, every adjacent midpoint and its neighbours,
    zero/subnormal/extreme/non-finite values, random raw bits and in-range values)."""
    if bits not in _CORPUS:
        d = tmpdir()
        exe = os.path.join(d, "posit_corpus_reference")
        if not os.path.exists(exe):
            run([ENV["clangxx"], "-std=c++20", "-O2", f"-I{ENV['universal_inc']}",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "posit_weight_reference.cpp"), "-o", exe])
        inp, exp = (os.path.join(d, f"corpus_p{bits}e1_{x}.bin") for x in ("input", "expected"))
        run([exe, str(bits), inp, exp])
        _CORPUS[bits] = (np.fromfile(inp, dtype=f32), np.fromfile(exp, dtype=f32))
    return _CORPUS[bits]


class ActivationExactness(unittest.TestCase):
    """Executed activation QDQ == independent reference, bit for bit."""

    @classmethod
    def setUpClass(cls):
        require("bin", "lib", "translate", "clangxx")

    def test_int_activation(self):
        for bits in (4, 5, 6, 7, 8):
            with self.subTest(fmt=f"int{bits}"):
                x = _activation_inputs()
                # exact rounding ties for this format: x/s + zp lands on k + 0.5
                ks = np.arange(-(1 << (bits - 1)) - 2, (1 << (bits - 1)) + 3)
                x = np.concatenate([x, ((ks + 0.5 - ACT_XZP) * ACT_XS).astype(f32)])
                so = build_so(f"act_int{bits}", _relu_module(len(x)),
                              f"Relu_1:a=int{bits}:{ACT_XS}:{ACT_XZP}:{ACT_YS}:{ACT_YZP}")
                got = execute(so, x.reshape(1, -1))
                exp = q_int_act(np.maximum(q_int_act(x, ACT_XS, ACT_XZP, bits), 0), ACT_YS, ACT_YZP, bits)
                assert_bit_exact(self, got, exp.reshape(1, -1), f"int{bits}")

    def test_f16_activation(self):
        x = np.concatenate([_activation_inputs(),
                            np.array([65519.9, 65520.0, 5.96e-8, 3e-8, 6.1e-5, 1e-5], dtype=f32)])
        so = build_so("act_f16", _relu_module(len(x)), "Relu_1:a=f16")
        exp = q_f16(np.maximum(q_f16(x), 0))
        assert_bit_exact(self, execute(so, x.reshape(1, -1)), exp.reshape(1, -1), "f16")

    def test_bf16_activation(self):
        x = _activation_inputs()
        so = build_so("act_bf16", _relu_module(len(x)), "Relu_1:a=bf16")
        assert_bit_exact(self, execute(so, x.reshape(1, -1)),
                         q_bf16(np.maximum(q_bf16(x), 0)).reshape(1, -1), "bf16")

    def test_posit_activation(self):
        """Regression: the posit round trip must survive the whole lowering pipeline
        (it used to be folded to the identity) AND agree with Universal exactly."""
        require("universal_inc", "mlir_inc", "posit_rt")
        rng = np.random.default_rng(3)
        for fmt in ("p4e1", "p6e1", "p8e1"):
            with self.subTest(fmt=fmt):
                x = np.concatenate([
                    rng.normal(0, 3, 30000), rng.uniform(-5000, 5000, 20000), rng.normal(0, 1e-3, 10000),
                    np.array([0.001, 0.75, -16.0, 5.0, 3e38, -3e38, 1e-40, -1e-40, 0.0])]).astype(f32)
                so = build_so(f"act_{fmt}", _relu_module(len(x)), f"Relu_1:a={fmt}", posit_fmt=fmt)
                got = execute(so, x.reshape(1, -1))
                exp = np.maximum(q_posit(x, fmt), 0)
                assert_bit_exact(self, got, exp.reshape(1, -1), fmt)
                # Guard the guard: on the positive inputs (relu maps every negative to 0 on both sides, so
                # they prove nothing) the quantized value must differ from the input for most values,
                # otherwise "matches Universal" could just mean "did nothing".
                pos = x > 0
                changed = int((bits_of(exp[pos]) != bits_of(x[pos])).sum())
                self.assertGreater(changed, int(0.9 * pos.sum()))

    def test_posit_activation_universal_corpus(self):
        """Both posit activation paths agree with Universal bit for bit on the full weight-LUT corpus
        (~132K inputs per format): posit.round_f32's inlined table lookup (the default) and the
        posit.from_f32/to_f32 runtime calls (MIXEDP_POSIT_ACTIVATION=runtime)."""
        require("universal_inc", "mlir_inc", "posit_rt")
        for bits in (4, 6, 8):
            fmt = f"p{bits}e1"
            x, exp = posit_corpus(bits)
            for path in ("lut", "runtime"):
                with self.subTest(fmt=fmt, path=path):
                    so = build_so(f"act_corpus_{fmt}_{path}", _mul_one_module(len(x)), f"Mul_1:a={fmt}",
                                  posit_fmt=fmt, extra_env={"MIXEDP_POSIT_ACTIVATION": path})
                    assert_bit_exact(self, execute(so, x.reshape(1, -1)), exp.reshape(1, -1),
                                     f"{fmt} {path}, {len(x)} inputs")


# --------------------------------------------------------------------------- #
# 2. The feature itself: ONE node, weight and activation in DIFFERENT formats
# --------------------------------------------------------------------------- #
ACT_PARAMS = (0.05, 2, 0.4, -1)   # x_scale, x_zp, y_scale, y_zp for INT activations


def _node_fmt(name, wf, af):
    s = f"{name}:w={wf}:a={af}"
    return s + (":" + ":".join(str(v) for v in ACT_PARAMS) if af.startswith("int") else "")


def _act_params(af):
    return (ACT_PARAMS[:2], ACT_PARAMS[2:]) if af.startswith("int") else (None, None)


def assert_finite_close(tc, got, exp, magnitude, label):
    tc.assertEqual(np.shape(got), np.shape(exp), f"{label}: output shape")
    tc.assertTrue(np.all(np.isfinite(got)), f"{label}: nonfinite actual output")
    tc.assertTrue(np.all(np.isfinite(exp)), f"{label}: nonfinite reference output")
    tc.assertTrue(np.all(np.isfinite(magnitude)), f"{label}: nonfinite error bound")
    got, exp = np.asarray(got, np.float64).reshape(-1), np.asarray(exp, np.float64).reshape(-1)
    tol = 1e-5 * np.maximum(np.asarray(magnitude, np.float64).reshape(-1), 1e-30)
    bad = np.abs(got - exp) > tol
    tc.assertEqual(int(bad.sum()), 0, f"{label}: {int(bad.sum())} outputs off, got={got[bad][:3]} exp={exp[bad][:3]}")


def _check_close(tc, got, exp, magnitude, fp32, label):
    assert_finite_close(tc, got, exp, magnitude, label)
    got = np.asarray(got, np.float64).reshape(-1)
    # not a no-op: quantization must move the result well away from plain FP32
    tc.assertGreater(float(np.max(np.abs(got - np.asarray(fp32, np.float64).reshape(-1)))), 1e-3,
                     f"{label}: result equals FP32 -- quantization had no effect")


PAIRS = [("int4", "int8"), ("p6e1", "int6"), ("bf16", "p8e1"), ("p4e1", "p4e1"),
         ("int8", "f16"), ("f16", "bf16"), ("p8e1", "p6e1"), ("int5", "int7")]


class WeightAndActivationDiffer(unittest.TestCase):
    """The reference matmul/conv accumulates in float64 while the compiled code
    accumulates in float32, so these compare to a tight relative tolerance (not
    bit-exact); the quantizers themselves are exactly referenced."""

    @classmethod
    def setUpClass(cls):
        require("bin", "lib", "translate", "clangxx")

    def _need_posit(self, wf, af):
        if posit_fmt_of(wf, af):
            require("universal_inc", "mlir_inc", "posit_rt")

    def test_matmul(self):
        K, N = 64, 16
        for wf, af in PAIRS:
            with self.subTest(weight=wf, activation=af):
                self._need_posit(wf, af)
                rng = np.random.default_rng(sum(map(ord, wf + af)))
                W = rng.normal(0, 1.5, (K, N)).astype(f32)
                A = rng.normal(0, 3, (1, K)).astype(f32)
                mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x{K}xf32> {{onnx.name = "x"}}) -> (tensor<1x{N}xf32> {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(W)}"> : tensor<{K}x{N}xf32>
    %0 = "onnx.MatMul"(%arg0, %w) {{onnx_node_name = "M"}} : (tensor<1x{K}xf32>, tensor<{K}x{N}xf32>) -> tensor<1x{N}xf32>
    return %0 : tensor<1x{N}xf32>
  }}''')
                so = build_so(f"mm_{wf}_{af}", mod, _node_fmt("M", wf, af), posit_fmt=posit_fmt_of(af))
                got = execute(so, A)
                px, py = _act_params(af)
                Aq, Wq = quant("a", A, af, px), quant("w", W, wf)
                ref = Aq.astype(np.float64) @ Wq.astype(np.float64)
                exp = quant("a", ref.astype(f32), af, py)
                mag = np.abs(Aq).astype(np.float64) @ np.abs(Wq).astype(np.float64)
                _check_close(self, got, exp, mag, A.astype(np.float64) @ W.astype(np.float64), f"{wf}/{af}")

    def test_conv(self):
        """Conv has a 4-D weight and its own attributes; same rewrite, checked by execution."""
        C, H, O, KH = 2, 6, 3, 3
        HO = H - KH + 1
        for wf, af in [("int4", "int8"), ("p6e1", "bf16"), ("bf16", "p8e1"), ("int6", "f16")]:
            with self.subTest(weight=wf, activation=af):
                self._need_posit(wf, af)
                rng = np.random.default_rng(sum(map(ord, "conv" + wf + af)))
                W = rng.normal(0, 1.0, (O, C, KH, KH)).astype(f32)
                X = rng.normal(0, 2, (1, C, H, H)).astype(f32)
                mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x{C}x{H}x{H}xf32> {{onnx.name = "x"}}) -> (tensor<1x{O}x{HO}x{HO}xf32> {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(W)}"> : tensor<{O}x{C}x{KH}x{KH}xf32>
    %b = "onnx.NoValue"() {{value}} : () -> none
    %0 = "onnx.Conv"(%arg0, %w, %b) {{onnx_node_name = "C", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [{KH}, {KH}], pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1]}} : (tensor<1x{C}x{H}x{H}xf32>, tensor<{O}x{C}x{KH}x{KH}xf32>, none) -> tensor<1x{O}x{HO}x{HO}xf32>
    return %0 : tensor<1x{O}x{HO}x{HO}xf32>
  }}''')
                so = build_so(f"conv_{wf}_{af}", mod, _node_fmt("C", wf, af), posit_fmt=posit_fmt_of(af))
                got = execute(so, X)
                px, py = _act_params(af)
                Xq, Wq = quant("a", X, af, px).astype(np.float64), quant("w", W, wf).astype(np.float64)
                ref = np.zeros((1, O, HO, HO))
                mag = np.zeros((1, O, HO, HO))
                for o in range(O):
                    for i in range(HO):
                        for j in range(HO):
                            patch = Xq[0, :, i:i + KH, j:j + KH]
                            ref[0, o, i, j] = (patch * Wq[o]).sum()
                            mag[0, o, i, j] = (np.abs(patch) * np.abs(Wq[o])).sum()
                exp = quant("a", ref.astype(f32), af, py)
                fp32 = np.zeros_like(ref)
                for o in range(O):
                    for i in range(HO):
                        for j in range(HO):
                            fp32[0, o, i, j] = (X[0, :, i:i + KH, j:j + KH].astype(np.float64) * W[o]).sum()
                _check_close(self, got, exp, mag, fp32, f"conv {wf}/{af}")


# --------------------------------------------------------------------------- #
# 3. Ops that only the IR had been looked at for: binary elementwise, pooling
# --------------------------------------------------------------------------- #
class BinaryElementwiseAndPooling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require("bin", "lib", "translate", "clangxx")

    def test_binary_ops_int8_bit_exact(self):
        """Add/Sub/Mul/Div: each operand round-trips with ITS OWN (scale, zp), the result with a third pair."""
        k = 4096
        rng = np.random.default_rng(11)
        A = rng.normal(0, 3, (1, k)).astype(f32)
        B = (rng.uniform(1.0, 5.0, (1, k)) * rng.choice([-1, 1], (1, k))).astype(f32)   # keep Div away from 0
        pa, pb, py = (0.05, 2), (0.06, -1), (0.11, 3)
        ops = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}
        for op, fn in ops.items():
            with self.subTest(op=op):
                mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x{k}xf32> {{onnx.name = "a"}}, %arg1: tensor<1x{k}xf32> {{onnx.name = "b"}}) -> (tensor<1x{k}xf32> {{onnx.name = "y"}}) {{
    %0 = "onnx.{op}"(%arg0, %arg1) {{onnx_node_name = "N"}} : (tensor<1x{k}xf32>, tensor<1x{k}xf32>) -> tensor<1x{k}xf32>
    return %0 : tensor<1x{k}xf32>
  }}''')
                fmt = f"N:a=int8:{pa[0]}:{pa[1]}:{pb[0]}:{pb[1]}:{py[0]}:{py[1]}"
                so = build_so(f"bin_{op}", mod, fmt)
                got = execute(so, A, B)
                qa, qb = q_int_act(A, *pa, 8), q_int_act(B, *pb, 8)
                with np.errstate(all="ignore"):
                    exp = q_int_act(fn(qa, qb), *py, 8)
                assert_bit_exact(self, got, exp, op, mask=np.isfinite(np.asarray(exp).reshape(-1)))

    def test_binary_add_posit(self):
        require("universal_inc", "mlir_inc", "posit_rt")
        k = 4096
        rng = np.random.default_rng(12)
        A = rng.normal(0, 3, (1, k)).astype(f32)
        B = rng.normal(0, 3, (1, k)).astype(f32)
        mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x{k}xf32> {{onnx.name = "a"}}, %arg1: tensor<1x{k}xf32> {{onnx.name = "b"}}) -> (tensor<1x{k}xf32> {{onnx.name = "y"}}) {{
    %0 = "onnx.Add"(%arg0, %arg1) {{onnx_node_name = "N"}} : (tensor<1x{k}xf32>, tensor<1x{k}xf32>) -> tensor<1x{k}xf32>
    return %0 : tensor<1x{k}xf32>
  }}''')
        so = build_so("bin_add_p6e1", mod, "N:a=p6e1", posit_fmt="p6e1")
        exp = q_posit(q_posit(A, "p6e1") + q_posit(B, "p6e1"), "p6e1")
        assert_bit_exact(self, execute(so, A, B), exp, "Add/p6e1")

    def test_pooling_int8(self):
        rng = np.random.default_rng(13)
        X = rng.normal(0, 2, (1, 2, 4, 4)).astype(f32)
        p = (0.05, 1, 0.02, -2)                                   # x_scale, x_zp, y_scale, y_zp
        fmt = lambda n: f"{n}:a=int8:{p[0]}:{p[1]}:{p[2]}:{p[3]}"
        qx = q_int_act(X, p[0], p[1], 8).astype(np.float64)
        cases = {
            "GlobalAveragePool": ("tensor<1x2x1x1xf32>", "", qx.mean(axis=(2, 3), keepdims=True)),
            "AveragePool": ("tensor<1x2x2x2xf32>", 'kernel_shape = [2, 2], strides = [2, 2]',
                            qx.reshape(1, 2, 2, 2, 2, 2).mean(axis=(3, 5))),
        }
        for op, (out_ty, attrs, ref) in cases.items():
            with self.subTest(op=op):
                attr_txt = f", {attrs}" if attrs else ""
                mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x2x4x4xf32> {{onnx.name = "x"}}) -> ({out_ty} {{onnx.name = "y"}}) {{
    %0 = "onnx.{op}"(%arg0) {{onnx_node_name = "P"{attr_txt}}} : (tensor<1x2x4x4xf32>) -> {out_ty}
    return %0 : {out_ty}
  }}''')
                so = build_so(f"pool_{op}", mod, fmt("P"))
                got = execute(so, X)
                exp = q_int_act(ref.astype(f32), p[2], p[3], 8)
                self.assertEqual(got.shape, exp.shape)
                self.assertTrue(np.all(np.isfinite(got)))
                # The compiled code accumulates the mean in float32, the reference in float64. That can
                # flip an output across a rounding boundary by exactly one output step; anything beyond
                # a single such element is a real mismatch.
                diff = np.abs(np.asarray(got, np.float64).reshape(-1) - np.asarray(exp, np.float64).reshape(-1))
                off = diff > 1e-6
                self.assertLessEqual(int(off.sum()), 1, f"{op}: {int(off.sum())}/{diff.size} outputs differ")
                self.assertLessEqual(float(diff.max()), p[2] * 1.0001 + 1e-6, f"{op}: differs by more than one step")


    def test_maxpool_single_out_bit_exact(self):
        """MaxPool's max is exact (no accumulation), so unlike the averaging pools this can be
        compared bit for bit. Covers padding and overlapping windows, i.e. that the rebuilt op keeps
        its attributes, with INT8 / bf16 / posit activations. (ceil_mode is preserved too, but that
        is checked at the IR level in maxpool.mlir: with a ceil geometry the expected output shape
        depends on framework rules for windows that start in the padding, which would test
        onnx-mlir's MaxPool lowering rather than this pass.)"""
        x_shape = (1, 2, 7, 7)
        X = np.random.default_rng(21).normal(0, 2, x_shape).astype(f32)
        geometries = {
            "k2s2": dict(kernel=(2, 2), strides=(2, 2), pads=(0, 0, 0, 0), ceil=0),
            "k3s2_pad1": dict(kernel=(3, 3), strides=(2, 2), pads=(1, 1, 1, 1), ceil=0),
        }
        act_cases = [("int8", (0.05, 1, 0.02, -2)), ("bf16", None), ("p8e1", None)]
        for gname, g in geometries.items():
            for af, params in act_cases:
                with self.subTest(geometry=gname, activation=af):
                    if posit_fmt_of(af):
                        require("universal_inc", "mlir_inc", "posit_rt")
                    kh, kw = g["kernel"]
                    sh, sw = g["strides"]
                    pt, pl, pb, pr = g["pads"]
                    rnd = math.ceil if g["ceil"] else math.floor
                    oh = int(rnd((x_shape[2] + pt + pb - kh) / sh)) + 1
                    ow = int(rnd((x_shape[3] + pl + pr - kw) / sw)) + 1
                    out_ty = f"tensor<1x2x{oh}x{ow}xf32>"
                    mod = wrap(f'''  func.func @main_graph(%arg0: tensor<1x2x7x7xf32> {{onnx.name = "x"}}) -> ({out_ty} {{onnx.name = "y"}}) {{
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {{onnx_node_name = "MP", kernel_shape = [{kh}, {kw}], strides = [{sh}, {sw}], pads = [{pt}, {pl}, {pb}, {pr}], ceil_mode = {g["ceil"]} : si64}} : (tensor<1x2x7x7xf32>) -> {out_ty}
    return %0 : {out_ty}
  }}''')
                    fmt = "MP:a=" + af + ("" if params is None else ":" + ":".join(str(v) for v in params))
                    so = build_so(f"maxpool_{gname}_{af}", mod, fmt, posit_fmt=posit_fmt_of(af))
                    got = execute(so, X)
                    px, py = (params[:2], params[2:]) if params else (None, None)
                    pooled = maxpool_reference(quant("a", X, af, px), g["kernel"], g["strides"], g["pads"], g["ceil"])
                    exp = quant("a", pooled, af, py)
                    self.assertEqual(np.shape(got), np.shape(exp))
                    assert_bit_exact(self, got, exp, f"maxpool {gname} {af}")
                    # not a no-op: quantizing changes the result compared with pooling the raw input
                    raw = maxpool_reference(X, g["kernel"], g["strides"], g["pads"], g["ceil"])
                    self.assertFalse(np.array_equal(got, raw), f"maxpool {gname} {af}: quantization had no effect")


# --------------------------------------------------------------------------- #
# 4. Operator attributes, untouched biases, and independently rewritten branches
# --------------------------------------------------------------------------- #
def maxpool_reference(x, kernel, strides, pads, ceil_mode):
    """NCHW max pooling; padded positions never win (they hold -inf)."""
    n, c, h, w = x.shape
    (kh, kw), (sh, sw), (pt, pl, pb, pr) = kernel, strides, pads
    rnd = math.ceil if ceil_mode else math.floor
    oh = int(rnd((h + pt + pb - kh) / sh)) + 1
    ow = int(rnd((w + pl + pr - kw) / sw)) + 1
    hp, wp = max(h + pt + pb, (oh - 1) * sh + kh), max(w + pl + pr, (ow - 1) * sw + kw)
    padded = np.full((n, c, hp, wp), -np.inf, dtype=x.dtype)
    padded[:, :, pt:pt + h, pl:pl + w] = x
    out = np.empty((n, c, oh, ow), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = padded[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw].max(axis=(2, 3))
    return out


def tensor_type(a):
    return "tensor<" + "x".join(map(str, np.shape(a))) + "xf32>"


def conv_reference(x, w, bias, pads, strides, group):
    """NCHW grouped cross-correlation, independently accumulated in float64."""
    top, left, bottom, right = pads
    sh, sw = strides
    x = np.pad(np.asarray(x, np.float64), ((0, 0), (0, 0), (top, bottom), (left, right)))
    w, bias = np.asarray(w, np.float64), np.asarray(bias, np.float64)
    batch, _, height, width = x.shape
    outputs, channels_per_group, kh, kw = w.shape
    oh, ow = (height - kh) // sh + 1, (width - kw) // sw + 1
    out = np.empty((batch, outputs, oh, ow), np.float64)
    for n in range(batch):
        for o in range(outputs):
            first_channel = (o // (outputs // group)) * channels_per_group
            for i in range(oh):
                for j in range(ow):
                    patch = x[n, first_channel:first_channel + channels_per_group,
                              i * sh:i * sh + kh, j * sw:j * sw + kw]
                    out[n, o, i, j] = np.sum(patch * w[o], dtype=np.float64) + bias[o]
    return out


class OperatorVariantsAndIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require("bin", "lib", "translate", "clangxx")

    def test_int7_weights(self):
        # An identity runtime input exposes every stored weight in the output.
        # max=63 makes scale=1, so these are exact half-away-from-zero ties.
        ties = np.array([[63, 1.5, -1.5, 0.5, -0.5, 0],
                         [-63, 62.5, -62.5, 2.49, -2.49, -0.0]], dtype=f32)
        cases = {"ties": ties, "nonunit_scale": ties * f32(0.027),
                 "all_zero": np.zeros_like(ties)}
        for name, w in cases.items():
            with self.subTest(case=name):
                ty = tensor_type(w)
                mod = wrap(f'''  func.func @main_graph(%x: tensor<2x2xf32> {{onnx.name = "x"}}) -> ({ty} {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(w)}"> : {ty}
    %y = "onnx.MatMul"(%x, %w) {{onnx_node_name = "W"}} : (tensor<2x2xf32>, {ty}) -> {ty}
    return %y : {ty}
  }}''')
                expected = q_int_weight(w, 7)
                got = execute(build_so("int7_weight_" + name, mod, "W:w=int7"), np.eye(2, dtype=f32))
                assert_bit_exact(self, got, expected, "int7 " + name)
                if name != "all_zero":
                    self.assertFalse(np.array_equal(expected, w), "test input must expose quantization")

    def test_nonzero_bias_is_not_input_quantized(self):
        # With x=0, the answer isolates the bias. Using different input/output
        # scales distinguishes preserved FP32 bias from accidentally QDQ'ing it
        # with the input scale (or fake-quantizing it as a weight).
        bias = np.array([0.13, -0.29, 0.61], dtype=f32)
        for op in ("Gemm", "Conv"):
            with self.subTest(op=op):
                if op == "Gemm":
                    x, w = np.zeros((2, 2), f32), np.array([[0.4, -0.7, 1.1], [0.8, 0.2, -0.3]], f32)
                    out_shape, attrs = (2, 3), 'alpha = 1.0 : f32, beta = 1.0 : f32'
                    raw_bias = np.broadcast_to(bias, out_shape)
                else:
                    x = np.zeros((1, 2, 2, 2), f32)
                    w = np.array([0.4, -0.7, 1.1, 0.8, 0.2, -0.3], f32).reshape(3, 2, 1, 1)
                    out_shape = (1, 3, 2, 2)
                    attrs = 'kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1], group = 1 : si64'
                    raw_bias = np.broadcast_to(bias.reshape(1, 3, 1, 1), out_shape)
                xt, wt, bt, yt = tensor_type(x), tensor_type(w), tensor_type(bias), tensor_type(np.empty(out_shape))
                mod = wrap(f'''  func.func @main_graph(%x: {xt} {{onnx.name = "x"}}) -> ({yt} {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(w)}"> : {wt}
    %b = onnx.Constant dense<"{hexblob(bias)}"> : {bt}
    %y = "onnx.{op}"(%x, %w, %b) {{onnx_node_name = "B", {attrs}}} : ({xt}, {wt}, {bt}) -> {yt}
    return %y : {yt}
  }}''')
                expected = q_int_act(raw_bias, 0.015625, 0, 8)
                wrong_input = q_int_act(q_int_act(raw_bias, 0.5, 0, 8), 0.015625, 0, 8)
                wrong_weight = q_int_act(q_int_weight(raw_bias, 4), 0.015625, 0, 8)
                self.assertFalse(np.array_equal(expected, wrong_input))
                self.assertFalse(np.array_equal(expected, wrong_weight))
                so = build_so("bias_preserved_" + op, mod, "B:w=int4:a=int8:0.5:0:0.015625:0")
                assert_bit_exact(self, execute(so, x), expected, op + " FP32 bias")

    def test_gemm_transpose_alpha_beta(self):
        a = np.array([[0.13, -1.71, 0.63], [1.29, 0.41, -0.87]], f32)
        logical_w = np.array([[0.31, -0.47, 1.7, 0.93], [-1.03, 0.79, 0.17, -0.23],
                              [0.61, 0.37, -0.89, 1.13]], f32)
        cases = [(0, 0, 1.0, 1.0, np.array([0.137, -0.291, 0.613, -0.071], f32)),
                 (1, 0, 0.75, -1.25, np.array([[0.137, -0.291, 0.613, -0.071]], f32)),
                 (0, 1, -1.5, 0.375, np.array([0.137], f32)),
                 (1, 1, 0.625, 1.75, np.array([[0.137, -0.291, 0.613, -0.071],
                                               [-0.173, 0.419, -0.537, 0.239]], f32))]
        for ta, tb, alpha, beta, bias in cases:
            with self.subTest(transA=ta, transB=tb, alpha=alpha, beta=beta):
                x, w = a.T.copy() if ta else a, logical_w.T.copy() if tb else logical_w
                xt, wt, bt = tensor_type(x), tensor_type(w), tensor_type(bias)
                mod = wrap(f'''  func.func @main_graph(%x: {xt} {{onnx.name = "x"}}) -> (tensor<2x4xf32> {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(w)}"> : {wt}
    %b = onnx.Constant dense<"{hexblob(bias)}"> : {bt}
    %y = "onnx.Gemm"(%x, %w, %b) {{onnx_node_name = "G", transA = {ta} : si64, transB = {tb} : si64, alpha = {alpha} : f32, beta = {beta} : f32}} : ({xt}, {wt}, {bt}) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }}''')
                wq = q_int_weight(w, 7).astype(np.float64)
                wq = wq.T if tb else wq
                ref = alpha * (a.astype(np.float64) @ wq) + beta * bias.astype(np.float64)
                mag = abs(alpha) * (np.abs(a).astype(np.float64) @ np.abs(wq)) + abs(beta) * np.abs(bias)
                plain = alpha * (a.astype(np.float64) @ logical_w.astype(np.float64)) + beta * bias
                so = build_so(f"gemm_attrs_{ta}_{tb}", mod, "G:w=int7")
                _check_close(self, execute(so, x), ref, mag, plain, f"Gemm transA={ta}/transB={tb}")
                if bias.size > 1:
                    # A scalar equals its own absmax and can survive INT weight
                    # quantization unchanged; vector/matrix biases detect it.
                    wrong_bias = alpha * (a.astype(np.float64) @ wq) + beta * q_int_weight(bias, 7)
                    self.assertGreater(float(np.max(np.abs(ref - wrong_bias))), 1e-4)

    def test_conv_group_padding_stride(self):
        # Two groups plus asymmetric padding and a 2-D stride exercise channel
        # partitioning and both spatial boundaries in one small model.
        rng = np.random.default_rng(29)
        x = rng.uniform(-1, 1, (1, 4, 5, 6)).astype(f32)
        w = (rng.integers(-55, 56, (4, 2, 3, 2)) / 64).astype(f32)
        w.flat[0] = f32(0.875)  # INT4 scale exactly 0.125; accumulation is exact.
        bias = np.array([-0.013, 0.027, 0.041, -0.057], f32)
        pads, strides, group = (1, 0, 0, 1), (2, 2), 2
        qx, qw = q_int_act(x, 0.125, 3, 8), q_int_weight(w, 4)
        raw = conv_reference(qx, qw, bias, pads, strides, group)
        expected = q_int_act(raw.astype(f32), 0.0625, -2, 8)
        xt, wt, bt, yt = tensor_type(x), tensor_type(w), tensor_type(bias), tensor_type(expected)
        mod = wrap(f'''  func.func @main_graph(%x: {xt} {{onnx.name = "x"}}) -> ({yt} {{onnx.name = "y"}}) {{
    %w = onnx.Constant dense<"{hexblob(w)}"> : {wt}
    %b = onnx.Constant dense<"{hexblob(bias)}"> : {bt}
    %y = "onnx.Conv"(%x, %w, %b) {{onnx_node_name = "C", auto_pad = "NOTSET", group = 2 : si64, kernel_shape = [3, 2], pads = [1, 0, 0, 1], strides = [2, 2], dilations = [1, 1]}} : ({xt}, {wt}, {bt}) -> {yt}
    return %y : {yt}
  }}''')
        so = build_so("conv_group_pad_stride", mod, "C:w=int4:a=int8:0.125:3:0.0625:-2")
        got = execute(so, x)
        self.assertTrue(np.all(np.isfinite(got)))
        assert_bit_exact(self, got, expected, "Conv group/padding/stride")
        plain = conv_reference(x, w, bias, pads, strides, group)
        self.assertGreater(float(np.max(np.abs(got - plain))), 1e-3)

    def test_binary_broadcast(self):
        a = np.array([-1.13, 0.31, 1.79, -0.43, 0.87, -1.37, 0.53, 1.11], f32).reshape(2, 1, 4)
        b = np.array([1.13, -1.79, 2.31], f32).reshape(1, 3, 1)
        pa, pb, py = (0.25, 1), (0.5, -2), (0.125, 3)
        qa, qb = q_int_act(a, *pa, 8), q_int_act(b, *pb, 8)
        self.assertTrue(np.all(qb != 0))
        for op, fn in {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}.items():
            with self.subTest(op=op):
                mod = wrap(f'''  func.func @main_graph(%a: tensor<2x1x4xf32> {{onnx.name = "a"}}, %b: tensor<1x3x1xf32> {{onnx.name = "b"}}) -> (tensor<2x3x4xf32> {{onnx.name = "y"}}) {{
    %y = "onnx.{op}"(%a, %b) {{onnx_node_name = "N"}} : (tensor<2x1x4xf32>, tensor<1x3x1xf32>) -> tensor<2x3x4xf32>
    return %y : tensor<2x3x4xf32>
  }}''')
                fmt = f"N:a=int8:{pa[0]}:{pa[1]}:{pb[0]}:{pb[1]}:{py[0]}:{py[1]}"
                so = build_so("broadcast_" + op, mod, fmt)
                expected = q_int_act(fn(qa, qb), *py, 8)
                self.assertTrue(np.all(np.isfinite(expected)))
                assert_bit_exact(self, execute(so, a, b), expected, op + " multidirectional broadcast")
                self.assertFalse(np.array_equal(expected, fn(a, b)))

    def test_shared_weight_and_input_branches(self):
        a = np.array([[0.13, -1.71, 0.63, 1.29], [0.41, -0.87, 1.17, -0.33]], f32)
        w = np.array([[0.31, -0.47, 1.7], [-1.03, 0.79, 0.17],
                      [0.61, 0.37, -0.89], [-0.23, 0.93, 1.13]], f32)
        selected = '%q = "onnx.MatMul"(%x, %w) {onnx_node_name = "Selected"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>'
        untouched = '%plain = "onnx.MatMul"(%x, %w) {onnx_node_name = "Untouched"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>'
        qa, qw = q_int_act(a, 0.125, 0, 8), q_int_weight(w, 7)
        raw_q = qa.astype(np.float64) @ qw.astype(np.float64)
        qresult = q_int_act(raw_q.astype(f32), 0.25, 0, 8)
        expected_q = q_bf16(np.maximum(q_bf16(qresult), 0))
        expected_plain = a.astype(np.float64) @ w.astype(np.float64)
        magnitude_plain = np.abs(a).astype(np.float64) @ np.abs(w).astype(np.float64)
        expected_input = np.maximum(a, 0)
        expected_join = expected_q.astype(np.float64) + expected_plain
        # The fixture must distinguish both accidental input quantization and
        # accidental in-place replacement of the shared weight constant.
        for contaminated in (a.astype(np.float64) @ qw.astype(np.float64),
                             qa.astype(np.float64) @ w.astype(np.float64)):
            self.assertGreater(float(np.max(np.abs(contaminated - expected_plain))), 1e-3)
        for first in (True, False):
            with self.subTest(selected_first=first):
                nodes = selected + "\n    " + untouched if first else untouched + "\n    " + selected
                mod = wrap(f'''  func.func @main_graph(%x: tensor<2x4xf32> {{onnx.name = "x"}}) -> (tensor<2x3xf32> {{onnx.name = "q"}}, tensor<2x3xf32> {{onnx.name = "plain"}}, tensor<2x4xf32> {{onnx.name = "input_branch"}}, tensor<2x3xf32> {{onnx.name = "join"}}) {{
    %w = onnx.Constant dense<"{hexblob(w)}"> : tensor<4x3xf32>
    {nodes}
    %rq = "onnx.Relu"(%q) {{onnx_node_name = "SelectedRelu"}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %ri = "onnx.Relu"(%x) {{onnx_node_name = "UntouchedInput"}} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %join = "onnx.Add"(%rq, %plain) {{onnx_node_name = "Join"}} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %rq, %plain, %ri, %join : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x4xf32>, tensor<2x3xf32>
  }}''')
                fmt = "Selected:w=int7:a=int8:0.125:0:0.25:0,SelectedRelu:a=bf16"
                outs = execute_outputs(build_so(f"shared_branches_{int(first)}", mod, fmt), a)
                self.assertEqual(len(outs), 4)
                assert_bit_exact(self, outs[0], expected_q, "selected chain")
                assert_finite_close(self, outs[1], expected_plain, magnitude_plain, "untouched shared weight")
                assert_bit_exact(self, outs[2], expected_input, "untouched shared input")
                assert_finite_close(self, outs[3], expected_join,
                                    np.abs(expected_q) + magnitude_plain, "joined consumers")


# --------------------------------------------------------------------------- #
# 5. Whole-model regression on the shipped mnist example
# --------------------------------------------------------------------------- #
class MnistModelLevel(unittest.TestCase):
    """The exact scenario that exposed the worst bug: quantizing ONLY the
    activation of a real model with posit must change the outputs."""

    @classmethod
    def setUpClass(cls):
        require("bin", "lib", "translate", "clangxx", "mnist")
        d = tmpdir()
        base = os.path.join(d, "mnist")
        run([os.path.join(ENV["bin"], "onnx-mlir"), "--EmitONNXIR",
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", "-o", base, ENV["mnist"]])
        with open(base + ".onnx.mlir") as f:
            cls.ir = f.read()
        run([os.path.join(ENV["bin"], "onnx-mlir"), "--EmitLib", "-o", os.path.join(d, "mnist_fp32"), ENV["mnist"]])
        cls.fp32_so = os.path.join(d, "mnist_fp32.so")
        rng = np.random.default_rng(0)
        cls.imgs = rng.normal(0, 1, (4, 1, 1, 28, 28)).astype(f32)
        cls.tag = "mnist"

    def _fp32(self, img):
        sys.path.insert(0, ENV["lib"])
        from PyRuntime import OMExecutionSession
        return OMExecutionSession(self.fp32_so).run([img])[0]

    def _mixed(self, fmt, posit):
        m = re.search(r'"onnx-mlir.symbol-postfix" = "([^"]*)"', self.ir)
        tag = m.group(1) if m else ""
        so = build_so("mnist_" + re.sub(r"\W", "_", fmt), self.ir, fmt, posit_fmt=posit)
        sys.path.insert(0, ENV["lib"])
        from PyRuntime import OMExecutionSession
        sess = OMExecutionSession(so, tag)
        return lambda img: sess.run([img])[0]

    def test_activation_only_posit_is_not_a_noop(self):
        require("universal_inc", "mlir_inc", "posit_rt")
        run_mixed = self._mixed("Gemm_3:a=p8e1", "p8e1")
        differs = 0
        for img in self.imgs:
            ref, got = self._fp32(img), run_mixed(img)
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertEqual(int(np.argmax(ref)), int(np.argmax(got)))
            differs += int(not np.array_equal(ref, got))
        self.assertEqual(differs, len(self.imgs),
                         "activation-only posit output was bit-identical to FP32 -- the round trip was folded away")

    def test_maxpool_node_is_quantized(self):
        """MaxPool_0 in the real model is an onnx.MaxPoolSingleOut. If the pass skipped it (as it used
        to, with a warning), the output would be bit-identical to FP32."""
        self.assertIn("onnx.MaxPoolSingleOut", self.ir)
        run_mixed = self._mixed("MaxPool_0:a=int8:0.05:0:0.05:0", None)
        for img in self.imgs:
            ref, got = self._fp32(img), run_mixed(img)
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertAlmostEqual(float(np.sum(got)), 1.0, places=3)
            self.assertFalse(np.array_equal(ref, got), "MaxPool_0 was not quantized")
            self.assertEqual(int(np.argmax(ref)), int(np.argmax(got)))

    def test_mixed_formats_stay_close_to_fp32(self):
        run_mixed = self._mixed("Gemm_3:w=int4:a=int8:0.1:0:0.1:0,Relu_4:a=bf16,Gemm_5:w=int6", None)
        for img in self.imgs:
            ref, got = self._fp32(img), run_mixed(img)
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertAlmostEqual(float(np.sum(got)), 1.0, places=3)            # still a softmax distribution
            self.assertEqual(int(np.argmax(ref)), int(np.argmax(got)))
            self.assertFalse(np.array_equal(ref, got))                             # quantization is in effect
            # Gross-breakage bound only (int4 weights are aggressive, so this is deliberately loose);
            # exact numerics are covered by the bit-exact / reference tests above.
            self.assertLess(float(np.max(np.abs(ref - got))), 0.25)


def tearDownModule():
    import shutil
    if _TMP and not os.environ.get("KEEP_TEST_TMP"):
        shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
