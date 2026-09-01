<!--- SPDX-License-Identifier: Apache-2.0 -->

# Low-precision numeric formats (`--convert-onnx-to-lowprecision`)

This is a `onnx-mlir-opt`-only pass (like the existing Posit pass, it is not
wired into the main `onnx-mlir` driver) that retypes individually-named ONNX
nodes to a chosen low-precision numeric format, driven by the
`LOWP_NODE_FORMATS` environment variable. It is independent of, and can be
chained with, the existing `--convert-onnx-to-posit` pass
(`POSIT_NODE_FORMATS`/`POSIT_SELECTIVE_NODES`): each pass only touches the
nodes named in its own environment variable, so a single model can mix Posit
formats and the formats below on different nodes.

## Supported formats

| Tag | Format | Mechanism |
|---|---|---|
| `bf16` | bfloat16 | `onnx.Cast` retype in place (op itself is untouched) |
| `f16` | float16 | `onnx.Cast` retype in place (op itself is untouched) |
| `int8` | INT8 (QDQ) | `QuantizeLinear`/`QLinearConv`/`QLinearMatMul`/`DequantizeLinear` |
| `fp8e4m3` | float8 E4M3FN | Quantize/dequantize boundary + `QLinearMatMul` (opset 21) |
| `fp8e5m2` | float8 E5M2 | Quantize/dequantize boundary + `QLinearMatMul` (opset 21) |

MXFP8 and other sub-byte/microscaling formats are explicitly out of scope
(no MLIR builtin type exists for them yet).

## `LOWP_NODE_FORMATS` syntax

```
LOWP_NODE_FORMATS=<node_name>:<format>[:<x_scale>:<x_zero_point>:<y_scale>:<y_zero_point>],...
```

- `bf16`/`f16` take no extra parameters.
- `int8`/`fp8e4m3`/`fp8e5m2` require exactly four extra parameters: the
  activation (not weight) quantization scale/zero-point for the node's input
  and output. These aren't calibrated automatically -- there's no
  calibration pass here, so they must be supplied externally (e.g. from a
  separate Python calibration step). For `int8` the zero-points are
  integers; for the fp8 formats they're real values (in practice almost
  always `0.0`).
- Weight (and, for INT8 Conv, bias) quantization parameters are **not**
  supplied via the env var: they're derived automatically at compile time
  from the weight/bias constant's own values (per-tensor symmetric
  min/max: `scale = max(abs(W)) / <format max>`, `zero_point = 0`), since
  they're static and don't need calibration.

Example: `LOWP_NODE_FORMATS=Conv_1:int8:0.02:0:0.05:0,Gemm_3:fp8e4m3:1.0:0:1.0:0,Relu_4:f16`

Unrecognized format tags are skipped (so a mixed-format env var doesn't
break on a future tag), and a node named for a format its op type doesn't
support is left untouched with a warning rather than crashing the pass.

## Per-op scope

| Op | bf16/f16 | int8 | fp8e4m3/fp8e5m2 |
|---|---|---|---|
| `onnx.Conv` | yes | yes (see below) | yes (see below) |
| `onnx.Gemm` | yes | yes | yes |
| `onnx.MatMul` | yes | yes | yes |
| `onnx.Relu` | yes | yes (QDQ round-trip only, see below) | yes (QDQ round-trip only, see below) |
| `onnx.Add`/`Sub`/`Mul`/`Div` | yes | yes (QDQ round-trip only, see below) | yes (QDQ round-trip only, see below) |
| `onnx.AveragePool`/`GlobalAveragePool` | no | yes (QDQ round-trip only, see below) | yes (QDQ round-trip only, see below) |
| `onnx.MaxPool` | no | no (known bug, see below) | no (known bug, see below) |

**bf16/f16**: a plain `onnx.Cast` in front of/behind the node's float
operands/results; the op itself is untouched, since ONNX's own type
constraints for these ops already legally accept bf16/f16 alongside
f32/f64.

**int8 Conv**: rewritten to `QuantizeLinear -> QLinearConv -> DequantizeLinear`.
`QLinearConv`'s Krnl lowering (`src/Conversion/ONNXToKrnl/Math/QLinearConv.cpp`,
not present upstream -- this project added it) implements the actual int8x8
-> int32 accumulate + rescale/saturate arithmetic via an im2col unfold reusing
`QLinearMatMul`'s kernel. Supports **`group >= 1`, including depthwise**
(each group is unfolded and matmul'd independently, then concatenated back
along the output-channel axis); 4D NCHW tensors; `auto_pad == NOTSET`; all
shape-affecting attributes and dims statically known.

**int8 Gemm/MatMul**: rewritten to use `QLinearMatMul` (opset 10). `alpha`
folds into the activation's dequant scale; `transA` inserts a real runtime
`onnx.Transpose`; `transB` is folded directly into the (compile-time
constant) weight data; Gemm's optional `C` operand is added in f32 after
dequantization (scaled by `beta`). B (the weight) must be a compile-time
`onnx.Constant` -- there's no calibration path for a non-constant weight.

**fp8e4m3/fp8e5m2 Gemm/MatMul**: same structure as the int8 case, but the
activation quantize/dequantize boundary is built from `onnx.Div`/`Add`/`Cast`
and `onnx.Cast`/`Sub`/`Mul` rather than literal `QuantizeLinear`/
`DequantizeLinear` (whose Krnl lowering asserts int8/uint8-only and was left
unmodified), and the matmul uses `QLinearMatMul`'s opset-21 float8 branch
(this project bumped `QLinearMatMul` from opset 10 to `[21, 10]` specifically
to unlock float8 T1/T2/T3; the old opset-10, int8-only schema is preserved as
`ONNXQLinearMatMulV10Op` so existing int8 QDQ models are unaffected). F8
values have no native LLVM IR float representation (LLVM lowers F8 to a
same-width integer type), so decode/encode is done via bit-manipulation
arith ops in `MathBuilder::cast`, not real float ops.

**fp8e4m3/fp8e5m2 Conv**: `onnx.QLinearConv` has no float8-capable opset
version to target (unlike `QLinearMatMul`), so this does its own im2col
decomposition directly at the ONNX-graph level (in `ONNXToLowPrecision.cpp`,
not inside a Krnl lowering pattern) and feeds the result straight into
`QLinearMatMul`. The unfold itself runs on the *original f32* activation
(`onnx.Slice`/`Concat`'s ONNX type constraints don't include float8 at all),
quantizing only the final unfolded `[M, K]` matrix; the weight is quantized
and permuted directly in C++ over the constant's data at compile time (no
runtime ops needed for it). Also supports **`group >= 1`, including
depthwise**, same restrictions otherwise (4D NCHW, `auto_pad == NOTSET`,
static shapes, weight a compile-time constant). Unlike the int8 kernel, the
weight/output scale here stay a single per-tensor scalar shared by every
group (no per-channel scale support), and bias -- being a real f32 value,
not QLinearConv's pre-quantized i32 bias -- is added once after all groups'
results are concatenated, rather than per group before rescaling.

**int8/fp8e4m3/fp8e5m2 Relu/Add/Sub/Mul/Div**: there's no ONNX quantized-
arithmetic op for these (no `QLinearRelu`/`QLinearAdd`/etc.), so instead of
routing through a real low-bit kernel, each operand is independently
quantized then immediately dequantized (a QDQ round trip simulating the
precision loss of storing it in that format at that scale/zero-point), the
*same* op is applied unchanged to the round-tripped operands (so the
arithmetic itself always happens in real f32), and the result gets one more
QDQ round trip. `onnx.Relu` (1 operand) takes the usual 4 params; the binary
ops (2 independently-scaled operands, since they're typically fed by two
different upstream tensors) take 6:
`<a_scale>:<a_zero_point>:<b_scale>:<b_zero_point>:<y_scale>:<y_zero_point>`.

> **Pipeline note**: always insert a `--canonicalize` between
> `--convert-onnx-to-lowprecision` and `--convert-onnx-to-krnl` (in addition
> to the usual one after `--convert-onnx-to-krnl`), e.g.:
> `--convert-onnx-to-lowprecision --canonicalize --convert-onnx-to-krnl --canonicalize`.
> This was found to matter in two independent cases, so treat it as a
> standing recommendation rather than something to add only after hitting a
> failure: (1) fp8e4m3/fp8e5m2 with two or more QDQ-round-tripped nodes in
> the same function -- fp8's QDQ round trip has no literal
> `QuantizeLinear`/`DequantizeLinear` to use (unlike int8), so it expands to
> plain `onnx.Div`/`Add`/`Cast`/`Sub`/`Mul`, which goes through mainline's
> generic elementwise Krnl lowering; that lowering assumes canonicalization
> has already hoisted constants, a precondition a single QDQ chain (or
> int8's literal-op path) never happened to violate before, and without the
> extra `--canonicalize` `--convert-onnx-to-krnl` can fail to verify (a
> dominance violation) on the second QDQ chain. (2) A rebuilt multi-result
> op (int8 `onnx.MaxPool` specifically) failing to legalize during
> `--convert-onnx-to-krnl` -- also fixed by the same extra `--canonicalize`,
> though `onnx.MaxPool` is excluded from this pass regardless for an
> unrelated reason, see below.

**int8/fp8e4m3/fp8e5m2 AveragePool/GlobalAveragePool**: same QDQ-round-trip
mechanism as Relu/Add/Sub/Mul/Div above (1 real operand, 4 params). Averaging
already-quantized-then-dequantized values isn't bit-identical to averaging
in a truly native low-precision accumulator, but it's the same simulation
tradeoff already accepted for Add/Mul/Div.

> **`onnx.MaxPool` is deliberately not supported** for int8/fp8, despite
> structurally fitting the same pattern (it just has a fixed ODS arity of 2
> results -- `Y` and `Indices`, the latter `NoneType` when unused -- which
> the elementwise QDQ machinery does handle generically). There's an
> unresolved, reproducible bug: the rebuilt `onnx.MaxPool` fails to legalize
> during `--convert-onnx-to-krnl` *only* when `--convert-krnl-to-llvm` is
> also present later in the same `onnx-mlir-opt` invocation -- even though
> that pass runs strictly after `--convert-onnx-to-krnl`, and the identical
> IR legalizes fine when round-tripped through a text dump into a second,
> separate `onnx-mlir-opt` invocation. Root cause not yet found (something
> in how CLI pass flags interact, not a `--canonicalize`-placement issue
> like the fp8 case above). Naming a `MaxPool` node in `LOWP_NODE_FORMATS`
> is safely ignored (warns and leaves the node untouched), it doesn't
> silently produce wrong output.

## Known limitations

- No calibration: activation scale/zero-point must be supplied externally
  via the env var for every int8/fp8 node.
- INT8/FP8 Gemm/MatMul require the weight (`B`) to be a compile-time
  constant.
- FP8 weight/output quantization is per-tensor only (no per-channel scale),
  for both Conv and Gemm/MatMul.
- Only `onnx-mlir-opt` wires this pass in; the main `onnx-mlir` driver does
  not (matches the existing Posit pass's precedent).
