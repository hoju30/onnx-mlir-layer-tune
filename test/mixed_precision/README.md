# Mixed-precision regression tests

The maintained tests are in this repository. The external
`mixedprecision-tests-draft` directory is an earlier snapshot.

## Run all tests

From the repository root, using an existing configured build:

```bash
cmake --build build --target check-onnx-mixed-precision
```

The target builds ONNX-MLIR tools/runtime, runs the focused lit suite, and runs
both Python numerical modules through CTest. A skipped test, missing dependency,
empty suite, or missing CTest registration is a failure. The CTest timeout is
ten minutes. Use the Python environment configured in CMake, with NumPy installed
and compatible with the built PyRuntimeC extension.

The IR tests are also part of `check-onnx-lit`. On native Linux x86-64, the full
suite is a dependency of `check-onnx-backend-numerical`. CI can invoke the focused
target directly. Numerical CTest registration is restricted to native Linux
x86-64 because the runtime fixtures use that target triple and data layout.
Explicitly requesting numerical tests elsewhere fails with an explanation.

```bash
cmake --build build --target check-onnx-mixed-precision-lit
cmake --build build --target check-onnx-mixed-precision-numerical
ctest --test-dir build -R '^MixedPrecisionNumerical$' --output-on-failure --no-tests=error
```

The direct CTest command assumes tools/runtime have already been built. For a
selected module or class, use the same Python interpreter configured in CMake:

```bash
python3 test/mixed_precision/run_tests.py test_mixed_precision_weights
python3 test/mixed_precision/run_tests.py test_mixed_precision_runtime.ActivationExactness
```

Direct runs accept `ONNX_MLIR_ROOT`, `ONNX_MLIR_BUILD`, `LLVM_BUILD_BIN`, and
`UNIVERSAL_ROOT` to override tool discovery. Ordinary `python -m unittest` can
report unavailable dependencies as SKIP; use `run_tests.py` for strict validation.
`KEEP_TEST_TMP=1` retains runtime model artifacts for debugging.

## Coverage

- `../mlir/conversion/onnx_to_mixed_precision/`: folded weights, QDQ dataflow,
  posit lowering, format/parameter validation, nonzero bias, and independent
  consumers of shared values. Invalid settings must warn without crashing or
  inserting QDQ. The uniqueness assertion rejects duplicated `qalign_key` values.
- `test_mixed_precision_weights.py`: reads the actual constant consumed by
  MatMul after the compiler pass. `posit_weight_reference.cpp` uses Universal
  directly without including or copying the pass. P4E1/P6E1/P8E1 cover every
  encoding, finite-value neighbors, adjacent midpoints and their FP32 neighbors,
  zero/subnormal/extreme/nonfinite values, and fixed random raw-bit/in-range
  samples: 395,478 inputs. INT4 through INT8 cover explicit rounding ties and
  neighbors, all-zero weights, scale underflow, maximum finite values, and
  mixed/all-nonfinite tensors. This module needs neither PyRuntimeC nor previous
  audit data in /tmp.
- `test_mixed_precision_runtime.py`: compiles through ONNX/Krnl/LLVM to shared
  libraries and executes through OMExecutionSession. NumPy supplies INT/F16/BF16
  references; a separate Universal helper supplies posit references. Coverage
  includes activation accuracy, weight/activation format combinations, nonzero
  bias, Gemm transpose/alpha/beta, grouped convolution with padding/stride,
  broadcasting, shared weights and multi-output branches, pooling (average,
  global average and max, the latter bit-exact), and MNIST.

## Pass criteria and limits

Activation/elementwise results are compared by FP32 bits, treating signed zeros
as equal and NaNs as equivalent where applicable. MatMul/Conv/Gemm comparisons
allow bounded differences for FP32 versus FP64 accumulation and validate shapes
and nonfinite results. Pooling allows at most one output to differ by one output
quantization step because accumulation can cross a rounding boundary.

INT weights use half-away-from-zero rounding; INT activations use ties-to-even.
The weight edge tests expect original FP32 values to be preserved when scale
calculation underflows or dequantization overflows. This fallback does not mean
those exceptional inputs have been successfully quantized to INT.

Execution needs native Linux x86-64, NumPy, onnx-mlir, onnx-mlir-opt,
mlir-translate, a C++20 compiler, libcruntime.a, and PyRuntimeC. Posit tests also
need Universal headers and the runtime source. Posit execution covers only
P4E1/P6E1/P8E1. Other posit formats, every possible shape/attribute combination,
NaN/Inf activation inputs, and interactions with other precision passes are not
exhaustively covered.

MNIST uses four fixed random inputs, not the labeled evaluation dataset. It
checks finite/changed outputs, prediction agreement with FP32, and coarse
probability bounds; it does not measure dataset accuracy.
