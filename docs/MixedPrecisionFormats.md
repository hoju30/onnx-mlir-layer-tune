<!--- SPDX-License-Identifier: Apache-2.0 -->

# Weight/activation-separated mixed-precision formats (`--convert-onnx-to-mixed-precision`)

This is an `onnx-mlir-opt`-only pass (like the existing Posit and
low-precision passes, it is not wired into the main `onnx-mlir` driver)
that lets a single node's **weight** and **activation** independently
simulate storage in different numeric formats, driven by the
`MIXEDP_NODE_FORMATS` environment variable. It is independent of, and can
be chained with, `--convert-onnx-to-lowprecision` (`LOWP_NODE_FORMATS`) and
`--convert-onnx-to-posit` (`POSIT_NODE_FORMATS`): each pass only touches
the nodes named in its own environment variable, and neither of the other
two passes is modified by this one.

## Why this pass exists

`LOWP_NODE_FORMATS` and `POSIT_NODE_FORMATS` both assign **one format to
an entire node** -- weight and activation together. This pass instead lets
weight and activation pick independent formats on the same node (e.g.
`Conv_1`'s weight in `int4` while its activation stays `p8e1`), from a
larger candidate set: `int4, int5, int6, int7, int8` (parametrized integer
bit width), `p<N>e<ES>` (posit, e.g. `p4e1`/`p6e1`/`p8e1`), `f16`, `bf16`,
`fp32`.

Every format here is realized purely via **QDQ (quantize-then-immediately-
dequantize) simulation**: the real op always runs on real f32 operands;
only the *storage* of a value is simulated as if it had been rounded
through the target format. This generalizes the QDQ-round-trip idea
`--convert-onnx-to-lowprecision` already uses for `Relu`/`Add`/`Sub`/`Mul`/
`Div` to also cover `Conv`/`Gemm`/`MatMul` (instead of routing them through
real `QLinearConv`/`QLinearMatMul` kernels) and to a much larger format set
that has no native ONNX quantized op or Krnl kernel to route through at
all. Because of this, there is a single, uniform rewrite for every
supported op -- unlike `--convert-onnx-to-lowprecision`'s per-format-family
split into real-kernel-building functions.

## `MIXEDP_NODE_FORMATS` syntax

```
MIXEDP_NODE_FORMATS=<node_name>:w=<weight_fmt>:a=<activation_fmt>[:<params>],...
```

- `w=<weight_fmt>` is only meaningful for ops with a real weight operand
  (`Conv`, `Gemm`, `MatMul`); omitted or `fp32` leaves the weight constant
  untouched. Naming `w=` on an op with no weight operand is a no-op --
  warned and ignored, not an error.
- `a=<activation_fmt>` applies to every other float operand and to the
  result.
- `<weight_fmt>`/`<activation_fmt>` each parse as `int4`..`int8`, `p<N>e<ES>`
  (e.g. `p4e1`, `p6e1`, `p8e1` -- any `N`/`ES` the Posit runtime resolves,
  not just these three), `f16`, `bf16`, or `fp32`.
- Extra numeric params are needed **only** when `activation_fmt` is one of
  `int4..int8` (needs calibration, same convention as `LOWP_NODE_FORMATS`):
  4 params (`x_scale:x_zero_point:y_scale:y_zero_point`) for single-input
  ops, 6 (`a_scale:a_zero_point:b_scale:b_zero_point:y_scale:y_zero_point`)
  for the two-input elementwise ops. Posit/f16/bf16/fp32 activation formats
  take no params (posit self-scales; f16/bf16/fp32 need no external scale
  at all). Weight is never calibrated externally -- its quantization
  parameters are always derived from the constant's own values (or, for
  posit, its own LUT lookup), exactly like weight/bias handling in
  `--convert-onnx-to-lowprecision`.
- Invalid parameters are rejected with a warning and that node's activation
  is left untouched (never silently built into the IR): the param count must
  match, every scale must be a positive normal FP32 value, and every
  zero-point must be finite -- both checked *after* narrowing to FP32, since
  the values are baked in as FP32 constants (`1e-100` would become `0`,
  `1e100` would become `Inf`). A non-numeric field is treated as NaN and
  rejected the same way. An unrecognized format tag (a typo such as `fp16`
  for `f16`) or an illegal posit `ES` (must satisfy `ES < N`) is likewise
  warned about rather than silently applying no quantization.

Example: `MIXEDP_NODE_FORMATS=Conv_1:w=int4:a=p8e1,Gemm_3:w=p6e1:a=int8:0.05:0:0.1:0,Relu_4:a=bf16`

## Per-op scope

`Conv`, `Gemm`, `MatMul`, `Relu`, `Add`, `Sub`, `Mul`, `Div`, `AveragePool`,
`GlobalAveragePool` and `MaxPoolSingleOut` (the single-result MaxPool that
the ONNX importer produces whenever the optional `Indices` output is unused --
i.e. the MaxPool actually found in models). The two-result `onnx.MaxPool`
(`Y` plus `Indices`) is **not** supported, but not because of this pass:
onnx-mlir's own `--convert-onnx-to-krnl` never lowers that form, so a
`MaxPool` of that shape fails later with `failed to legalize operation
'onnx.MaxPool'` even when no quantization pass has touched it. (That is the
real cause of the failure `--convert-onnx-to-lowprecision` attributes to its
"rebuilt multi-result MaxPool".)

Because every format here is QDQ-simulated rather than routed through a real
per-format kernel, there is no per-op format eligibility table the way
`LowPrecisionFormats.md` has one -- any format in the candidate set is legal
on any supported op.

Bias (`Conv`'s 3rd operand / `Gemm`'s optional `C`) is **never** quantized;
it always stays real f32, matching the "arithmetic stays real f32"
philosophy and avoiding `QLinearConv`-style pre-quantized-bias handling
(there is no real `QLinearConv` being built here at all).

## How each format family is realized

**Weight** (compile-time `onnx.Constant`, folded directly into the
constant's data, no runtime cost):
- `int{4..8}`: symmetric quant, `scale = max_abs / (2^(bits-1)-1)`, round
  + clamp, rescale back to f32 -- same convention as
  `--convert-onnx-to-lowprecision`'s automatic weight quantization,
  generalized to a parametrized bit width.
- `p<N>e<ES>` with `N<=8` (`p4e1`/`p6e1`/`p8e1`): **LUT-accelerated and
  bit-exact with `sw::universal::posit<N,ES>`**. Posit rounding is not
  "nearest representable value": it rounds on the encoded bit string, a
  nonzero input never underflows to 0 (it lands on `minpos`), values past
  `maxpos` saturate, and NaN/`+-Inf` become NaR (which decodes to NaN).
  Universal's float -> posit conversion is however a monotone step
  function, so for the sorted finite representable values there is one exact
  decision boundary per adjacent pair. Those boundaries are found once per
  `(N,ES)` by bisecting over the ordered float bit patterns with Universal
  itself as the oracle, then each weight element costs one binary search over
  at most `2^N-2` boundaries instead of a general posit encode. NaR is
  excluded from the value table (it decodes to NaN and would break the
  ordering). Verified against Universal on ~600K inputs across the three
  formats (all representable values, every adjacent midpoint and its float
  neighbours, geometric midpoints, subnormals, +-0, +-FLT_MAX, NaN/Inf,
  random bit patterns) with zero mismatches. Non-finite weight elements
  become NaN, as in Universal (unlike the INT path, which passes them
  through unchanged). Only wired up for `N<=8` and the `(N,ES)` pairs
  `(4,1)`, `(6,1)`, `(8,1)`; any other posit weight format warns and leaves
  the weight untouched.
- `f16`/`bf16`: `APFloat` round trip through the target type's own float
  semantics (MLIR's own correct-rounding machinery, same technique already
  used for compile-time FP8 weight quantization).
- `fp32`: identity, weight left untouched.

**Activation** (runtime tensor, values only known at execution -- real IR
ops inserted, brackets a round trip back to f32 before the real op runs):
- `int{4..8}`: `Div(x,scale) -> Add(zero_point) -> Round -> Clip(qmin,qmax)
  -> Sub(zero_point) -> Mul(scale)`, built from plain `onnx.Div`/`Add`/
  `Round`/`Clip`/`Sub`/`Mul` -- never touches a real low-bit type at all
  (deliberately avoids `onnx.QuantizeLinear`/`DequantizeLinear`, whose Krnl
  lowering is hardcoded to int8/uint8 only, so it cannot represent
  int4..int7).
- `p<N>e<ES>`: one `posit.round_f32 {nbits, es}` (a Posit dialect op:
  f32 -> posit<N,ES> -> f32 as a single fake-quantization step, the posit
  value never stored). `--convert-posit-to-krnl` lowers it to an **inlined
  table lookup** -- the same Universal-derived decision boundaries the
  weight path uses (`src/Dialect/Posit/PositLut.hpp`), stored as two
  `krnl.global` tables, searched with `k` unrolled compare/select steps per
  element (thresholds padded to `2^k - 1`) -- so it needs no posit runtime
  call and no posit storage buffer. Only `(4,1)`, `(6,1)`, `(8,1)` have a
  table. Bit-exact with Universal: checked on the weight path's full
  corpus (~132K inputs per format) by
  `test/mixed_precision/test_mixed_precision_runtime.py`, and with every
  supported MobileNetV2 node at `a=p8e1` / `a=p4e1` the 50,000 logits of 50
  images were bit-identical to the runtime path below at 9.2x / 13.2x its
  speed (`experiments/mixed_precision/run_posit_round_check.py`).
  `MIXEDP_POSIT_ACTIVATION=runtime` -- and any posit format without a table
  -- keeps the original pair instead: `posit.from_f32` then `posit.to_f32`
  (the Posit dialect's own cast ops, the same building block
  `--convert-onnx-to-posit` uses), executed by `src/posit_runtime.cpp`.
  Each inserted `posit.from_f32` carries a nonzero `qalign_key` attribute:
  without it, `--convert-posit-to-krnl` folds a directly-adjacent
  `to_f32(from_f32(x))` straight to `x`, which silently turns the whole
  simulated round trip into a no-op (an activation-only posit run came out
  bit-identical to FP32 before this was fixed). An unregistered key is
  numerically transparent in the runtime, so it only disables that fold.
- `f16`/`bf16`: `onnx.Cast` down then immediately back up to f32 (bracket,
  not a sustained retype -- unlike `--convert-onnx-to-lowprecision`'s
  bf16/f16 path, which keeps the whole op running in bf16/f16, this pass
  always returns to f32 before the real op runs, for consistency with
  every other format here).
- `fp32`: identity/no-op.

## Pipeline note

Because `p<N>e<ES>` activation formats emit Posit dialect ops
(`posit.round_f32`, or `posit.from_f32`/`to_f32`) into otherwise-ONNX-dialect IR, any pipeline that uses a posit format
on any node (via `w=` or `a=`) must also run
`--convert-posit-to-krnl` in addition to `--convert-onnx-to-krnl`, even if
`--convert-onnx-to-posit` itself was never invoked -- e.g.:

```
onnx-mlir-opt --convert-onnx-to-mixed-precision --canonicalize \
  --convert-onnx-to-krnl --convert-posit-to-krnl --canonicalize ...
```

This mirrors the existing precedent that `LOWP_NODE_FORMATS` and
`POSIT_NODE_FORMATS` are already designed to coexist on different nodes of
the same model.

## Running a whole model

`experiments/mixed_precision/eval_random_mixed_precision.py` runs a real
ImageNet classifier (timm MobileNetV2) through this pass with a random
per-layer weight/activation assignment, calibrates the `int` activation
scale/zero-point from min/max over a few images, and reports top-1/top-5. Three
integration pitfalls surfaced while doing that; they apply to any real model:

- **Run the pass on the freshly imported graph.** `onnx-mlir --EmitONNXIR`
  first rewrites every 1x1 `Conv` into `Reshape`/`MatMul`/`Add` and renames the
  pieces (`node_Conv_813` -> `node_Conv_813_73/_38/_17`; the weight becomes the
  MatMul's *first* operand), so the node names in `MIXEDP_NODE_FORMATS` -- and
  any per-ONNX-node calibration -- stop matching. Import with
  `--EmitONNXBasic`, run `--shape-inference --convert-onnx-to-mixed-precision`,
  and only then the ONNX-level passes `--EmitONNXIR` would have run (the list in
  `addONNXToMLIRPasses`, `src/Compiler/CompilerPasses.cpp`; also
  `--standard-func-return`, which is what turns `onnx.Return` into `return`).
- **Use the complete Krnl-to-LLVM pass list, including
  `--buffer-deallocation-pipeline`** (`KRNL_TO_LLVM_PASSES` in the script). The
  shorter `--convert-krnl-to-affine --convert-krnl-to-llvm` recipe of
  `experiments/common/lowp_lower_general.py` emits no `dealloc` at all, so every
  inference leaks all of its intermediate buffers: 64 MB per MobileNetV2 image
  for plain FP32, over 100 MB with QDQ chains (13 GB after 100 images).
- **`f16` activations need F16C (`clang++ -mf16c`).** They are runtime
  `f32 -> f16 -> f32` conversions, which on baseline x86-64 become calls to
  `__truncsfhf2`/`__extendhfsf2`. `libcruntime.a` defines those in
  `src/Runtime/OMTensor.inc` with the old convention (`uint16_t
  __truncsfhf2(float)`, `float __extendhfsf2(uint16_t)`: half in an integer
  register) while current clang passes and returns `half` in `xmm0`; the
  generated code feeds the result of one straight into the other in `xmm0`, and
  the runtime's `__extendhfsf2` reads `edi`. The values are stale-register
  garbage and differ from run to run (zeros, NaN, or occasionally the right
  answer). With F16C, LLVM emits `vcvtps2ph`/`vcvtph2ps` inline and no libcall
  remains. This is independent of this pass: a hand-written `onnx.Cast` pair
  fails the same way, so any f16 cast in generated code is exposed (the f16
  path of `--convert-onnx-to-lowprecision` casts too; not tested here). `bf16`
  activations were not affected in the same tests, and `f16` *weights* are
  folded at compile time, so they are not affected either.

## Collecting statistics in the compiled FP32 graph (`--mixedp-collect-stats`)

A second `onnx-mlir-opt` pass instruments the **FP32** graph instead of
quantizing it, so the statistics a mixed-precision assignment needs come from
the same compiled graph the mixed-precision builds start from, not from a
separate runtime. It visits every node the pass above could quantize (a named
op from the [Per-op scope](#per-op-scope) list, whether or not it appears in
`MIXEDP_NODE_FORMATS`) and taps exactly the points that pass would QDQ: each
activation operand (the weight excluded) and the result. A value seen by
several points -- a result feeding another supported node -- is tapped once,
right after it is defined.

- `MIXEDP_STATS_MODE`:
  - `ranges` (default) -- per tap, the tensor's `(min, max)` over all of its
    elements, i.e. the per-tensor range the `int` activation parameters are
    derived from.
  - `weights` -- compile time only, the IR is left untouched: for every
    weight-bearing node with a compile-time f32 weight and every candidate
    format, the squared weight quantization error summed per output channel,
    `sum_{i in o} (Q(W)_i - W_i)^2`, where `Q` is this pass's own weight
    fake-quant (per-channel `int` scales under `MIXEDP_WEIGHT_PER_CHANNEL=1`,
    the posit LUT, APFloat `f16`/`bf16`) and the channel axis is the one
    `MIXEDP_WEIGHT_PER_CHANNEL` uses. `MIXEDP_STATS_FORMATS` (comma-separated)
    picks the formats; default `int4..int8,p4e1,p6e1,p8e1,f16,bf16`. Nodes
    whose weight operand is not a constant are listed under `skipped`.
  - `grads` -- appends the graph's own **backward pass**
    (`src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.cpp`: reverse
    sweep over the function body emitting each op's vector-Jacobian product as
    ONNX ops; Conv, Gemm/MatMul, Add/Sub/Mul, Relu, Clip, Sigmoid,
    ReduceMean/GlobalAveragePool, Reshape/Flatten; static shapes, batch 1;
    Conv/Gemm biases computed at run time get their gradient too, and a
    ReduceMean with no axes and `noop_with_empty_axes = 1` passes it through).
    The loss is softmax cross-entropy of result 0 against a new function
    argument `mixedp_onehot` of the same shape (seed `softmax(z) - onehot`).
    Per tap, with `g` the tensor's total gradient, `sum(g^2)` and `sum(g)`
    over every axis but the channel axis (1 for NCHW, else the last; a rank-1
    tensor is its own per-channel sum); per
    weight-bearing node with a compile-time weight, `sum(dW^2)` per output
    channel. The Conv input gradient is a stride-1 Conv of the zero-inserted
    `dY` with the spatially flipped, per-group transposed weight (folded at
    compile time) -- not `onnx.ConvTranspose`, whose decomposition does not
    handle groups; the weight gradient is a Conv of `X` with `dY` as the
    kernel (strides and dilations swapped), or a MatMul for unpadded 1x1
    convs. Checked against PyTorch autograd on MobileNetV2
    (`experiments/mixed_precision/run_native_grad_check.py`): images agree to
    ~1e-6 except where a ReLU6 input within float rounding of 0 falls on the
    other side of the kink (~1 element in some images).
  - `errors` -- per tap and candidate activation format
    (`MIXEDP_STATS_FORMATS`), `d = Q(t) - t` with `Q` exactly the activation
    QDQ `--convert-onnx-to-mixed-precision` inserts (`posit.round_f32`, cast
    pairs, the int QDQ chain), reduced to `sum(d)` and `sum(d^2)` per channel
    and inserted right after `t` is defined. Int formats need the tensor's
    calibration range: `MIXEDP_STATS_RANGES` names a JSON
    `{node: {slot: [min, max]}}` (e.g. from a `ranges` run), from which the pass
    derives `(scale, zero point)` with the same arithmetic
    `experiments/mixed_precision/eval_random_mixed_precision.int_qparams` uses
    for `MIXEDP_NODE_FORMATS`.
  - `criterion` -- `grads` and `errors` in one graph.
  - `values` -- (for testing) every tap's tensor, flattened.

  The `errors` graph has thousands of small loops in one function: lower it
  with `--O0` and link with `clang++ -O0` (about 2 minutes for MobileNetV2 and
  10 formats; at `-O1`/`-O3` LLVM takes over 20 minutes). Every `.so` built from
  the same imported model exports `main_graph_<postfix>`; to load two of them
  into one process (e.g. a `grads` and an `errors` build), give each its own
  `"onnx-mlir.symbol-postfix"` module attribute, or the second silently runs
  the first one's graph.
- `MIXEDP_STATS_LAYOUT` (required): path of the JSON file the pass writes. In
  `ranges` mode it lists every point as `{node, op, slot, tap}`, with `slot` =
  `x`/`y` for unary ops and `a`/`b`/`y` for the binary elementwise ops (the
  same order as the `int` activation params), plus `result_index`, `num_taps`
  and `values_per_tap`. In `weights` mode it holds the errors themselves
  (`nodes[].err2[<format>]`, one value per output channel). In `grads` mode it
  adds `input_name`/`input_index` (the one-hot argument), `taps[]` and
  `weights[]` with each segment's `offset` and `channels` in `mixedp_stats`.

All taps are concatenated into **one** extra f32 result named `mixedp_stats`
(`values_per_tap` values per tap, in tap order) appended after the model's own
outputs. Run it in place of `--convert-onnx-to-mixed-precision` on the freshly
imported graph, then lower as usual:

```
MIXEDP_STATS_LAYOUT=layout.json onnx-mlir-opt model.onnx.mlir \
  --shape-inference --mixedp-collect-stats -o model.stats.onnx.mlir
```

The calibration range of a tensor is then the elementwise min/max of the
per-image `mixedp_stats` over the calibration images
(`experiments/mixed_precision/run_native_calib_check.py`).

## Known limitations

- No calibration: activation `int4..int8` scale/zero-point must be
  supplied externally via the env var, same as `LOWP_NODE_FORMATS`.
  `experiments/mixed_precision/eval_random_mixed_precision.py` shows one way:
  min/max of every node's own input/output tensors over a few images, turned
  into an affine (asymmetric) scale/zero-point.
- `intN` weight quantization is **per-tensor** by default: one scale,
  `max|w| / (2^(N-1)-1)`, for the whole constant. That is hard on nets whose
  output channels differ in magnitude by orders of magnitude (BN-folded
  depthwise convolutions). In timm's `mobilenetv2_100.ra_in1k` the per-channel
  `max|w|` inside a single layer spans up to 1.7e8x, so per-tensor int8 rounds
  196 of the 7136 depthwise output channels entirely to zero (int4: 1593), and on
  100 ImageNet validation images `int4`..`int6` weights alone give <=1% top-1
  (`int7` 52%, `int8` 73%, FP32 81%). Set `MIXEDP_WEIGHT_PER_CHANNEL=1` to use one
  scale per output channel instead (same rounding, `max|w_c| / (2^(N-1)-1)` per
  channel): axis 0 for `Conv` and for `Gemm` with `transB=1`, axis 1 for `Gemm`
  without it, the last axis of a `MatMul`'s constant right operand. Other weight
  formats are unaffected. On 1000 ImageNet validation images, all-`int8` weights
  with FP32 activations go from 0.1% to 77.3% top-1 on timm
  `efficientnet_b0.ra_in1k` (FP32 77.4%) and from 70.6% to 71.6% on
  `mobilenetv2_100.ra_in1k` (FP32 72.1%); the compiled logits match an
  onnxruntime simulation of the same arithmetic to 1e-5
  (`experiments/mixed_precision/run_compiled_compare.py --check`).
- Weight quantization requires the weight operand to be a compile-time
  `onnx.Constant`; a non-constant weight leaves the weight untouched
  (warned, not an error).
- Posit weight quantization is only implemented for `p4e1`/`p6e1`/`p8e1`
  (the LUT needs `N`/`ES` as compile-time template parameters). Other
  posit formats accepted by the parser -- and available for *activation*
  through the Posit runtime -- warn and leave the weight untouched, so they
  must not be read as "quantized". A format the parser accepts but the Posit
  runtime does not export (e.g. `p8e3`) is not caught here and fails later,
  at link time, with an undefined `posit_*` symbol.
- Only `onnx-mlir-opt` wires this pass in; the main `onnx-mlir` driver does
  not (matches the existing Posit and low-precision passes' precedent).
- The two-result `onnx.MaxPool` (with an `Indices` result) is not supported:
  onnx-mlir cannot lower that form at all, with or without this pass (see
  "Per-op scope"). The single-result `onnx.MaxPoolSingleOut`, which is what
  models contain, is supported.
