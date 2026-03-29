# MobileNet/ResNet Lowering Fix Log

Date: 2026-03-04

## Scope
- Target models:
  - `mobilenetv2-12.onnx.mlir`
  - `mobilenetv2-12-qdq.onnx.mlir`
  - `resnet50-v1-12.onnx.mlir`
  - `resnet50-v1-12-qdq.onnx.mlir`
- Goal: make the current `onnx-mlir-opt` flow reach `.ll` generation.

## Important Issues Found

1. ONNXToPosit failed on unsupported ops (`onnx.Clip`, `onnx.Dim`) for large models.
- Root cause:
  - The pass forced conversion of selected ONNX ops, but large models include dynamic-shape parts and unsupported op patterns.
- Fix:
  - Added dynamic-shape fallback in `Conversion/ONNXToPosit/ONNXToPosit.cpp`.
  - If dynamic tensor shapes are detected, ONNXToPosit returns early (no conversion), so fallback ONNX lowering can proceed.

2. PositToKrnl failed when ONNXToPosit fallback left only ONNX ops.
- Error pattern:
  - `onnx.Constant` reported as illegal during `--convert-posit-to-krnl`.
- Root cause:
  - PositToKrnl attempted type-conversion even when no `posit.*` ops existed.
- Fix:
  - Added early no-op guard in `Conversion/PositToKrnl/PositToKrnl.cpp`:
    - If module has no `posit` dialect ops, return immediately.

3. Krnl->LLVM failed directly with `krnl.define_loops`.
- Error pattern:
  - `failed to legalize operation 'krnl.define_loops'`.
- Root cause:
  - `--convert-krnl-to-llvm` alone is not sufficient for Krnl loop-form IR from ONNXToKrnl fallback path.
- Fix (pipeline):
  - Add `--canonicalize --convert-krnl-to-affine` before `--convert-krnl-to-llvm`.

4. QDQ LLVM dialect still had `builtin.unrealized_conversion_cast` (`i8 -> ui8`).
- Error pattern:
  - `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`.
- Fix (pipeline):
  - Add `--reconcile-unrealized-casts` after `--convert-krnl-to-llvm`.

5. `mlir-translate` tool version mismatch.
- Error pattern:
  - Parse error on `llvm.getelementptr inbounds|nuw`.
- Root cause:
  - `/usr/local/bin/mlir-translate` is older than the MLIR dialect printed by current build.
- Fix:
- Use matching translator:
  - `/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate`

6. `build_model11_sos.sh` pipeline skipped `--convert-onnx-to-krnl`.
- Error pattern:
  - Reported as `failed to legalize operation 'func.func'` in `<model>.onnx.mlir`.
- Root cause:
  - Script chained `--convert-onnx-to-posit --convert-posit-to-krnl` directly to `--convert-krnl-to-llvm`.
  - When ONNXToPosit fell back (dynamic shapes), ONNX ops remained and were never lowered to Krnl.
- Fix:
  - Split script pipeline into two stages and add missing pass:
    - Stage A: `shape-inference -> onnx-to-posit -> posit-to-krnl -> onnx-to-krnl`
    - Stage B: `canonicalize -> krnl-to-affine -> krnl-to-llvm -> reconcile-unrealized-casts`

7. `build_model11_sos.sh` default binary/runtime paths were brittle.
- Error pattern:
  - `onnx-mlir not found/executable: <repo>/onnx-mlir`
  - `run_time.cpp: No such file or directory`
  - SoftPosit header error from wrong runtime source.
- Fix:
  - Added binary auto-detection:
    - `<repo>/build/Release/bin`, `<repo>/build/Debug/bin`, then legacy paths.
  - Added `mlir-translate` auto-detection with preferred matching LLVM build path.
  - Runtime source selection now prefers `src/posit_runtime.cpp` and `src/run_time.cpp`.

8. Wrapper scripts assumed ONNX dataset path only.
- Affected:
  - `build_mobilenet11_sos.sh`
  - `build_resnet50_11_sos.sh`
- Root cause:
  - Hardcoded `datasets/models/onnx/*.onnx`; local `.onnx.mlir` in `src/` was ignored.
- Fix:
  - Wrapper now auto-selects source:
    - If dataset ONNX exists: use `--qdq-onnx/--nqdq-onnx`
    - Else: use `--qdq-mlir/--nqdq-mlir` from `src/`

## Verified Working Pipeline

1) ONNX/Posit/Krnl stage:

```bash
../build/Release/bin/onnx-mlir-opt \
  --posit-format=p8e0 \
  --convert-onnx-to-posit \
  --convert-posit-to-krnl \
  --convert-onnx-to-krnl \
  <model>.onnx.mlir \
  -o /tmp/<model>.krnl.mlir
```

2) Krnl -> LLVM dialect:

```bash
../build/Release/bin/onnx-mlir-opt \
  --canonicalize \
  --convert-krnl-to-affine \
  --convert-krnl-to-llvm \
  --reconcile-unrealized-casts \
  /tmp/<model>.krnl.mlir \
  -o /tmp/<model>.llvm.mlir
```

3) LLVM dialect -> `.ll`:

```bash
/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate \
  --mlir-to-llvmir \
  /tmp/<model>.llvm.mlir \
  -o /tmp/<model>.ll
```

## Notes
- For static-shape models (e.g. the previous MNIST static path), ONNXToPosit/PositToKrnl behavior is preserved.
- For dynamic-shape models, current behavior intentionally falls back to ONNX path for robustness.

## Dynamic-Shape (No Fallback) Investigation

Date: 2026-03-06

Goal:
- Remove ONNXToPosit dynamic fallback and keep Posit path active for dynamic-shape models.

What was tested:
1) Disable dynamic fallback in `convert-onnx-to-posit`.
2) Improve two conversion points in `Conversion/ONNXToPosit/Pattern/Math.cpp`:
   - Conv optional bias: derive OC from weight shape first.
   - MatMul->Gemm: use `C` as a 1-element zero posit tensor (broadcastable), so dynamic output shape is not required for constant materialization.

Observed blockers:
1) `convert-onnx-to-posit --convert-posit-to-krnl` fails with mixed ONNX+Posit IR:
   - Example error: `onnx.Dim` illegal during `convert-posit-to-krnl`.
   - Root cause: PositToKrnl type conversion expects full tensor->memref legality, but dynamic model still contains many ONNX tensor ops.

2) Reordered pipeline attempt
   - `convert-onnx-to-posit -> convert-onnx-to-krnl -> convert-posit-to-krnl`
   - failed in ONNXToKrnl shape helper (`onnx.Unsqueeze`), with assertion from shape computation.
   - Root cause: ONNX shape helper path is sensitive to inserted cast structures from partial Posit rewriting.

Conclusion:
- With current architecture, "no fallback + partial Posit conversion on dynamic graphs" is not stable yet.
- A robust no-fallback solution requires larger refactor:
  - either full ONNX op coverage in ONNXToPosit for dynamic QDQ graphs,
  - or mixed-graph-safe PositToKrnl/type-conversion strategy that can coexist with remaining ONNX tensor ops.

## 2026-03-16: No-Int8-QDQ Path Stabilization

Goal:
- Fully avoid ONNX QDQ int8 lowering path while keeping `f32 + posit` flow legal for dynamic-shape Mobilenet/Resnet.

Key issues found and fixes:
1) `convert-onnx-to-posit` failed on `onnx.DequantizeLinear`
- Fix:
  - Robustly match `DQ(Q(...))` and `DQ(QLinearMatMul(...))` with `stripUnrealizedCast`.
  - Make rewrite path always legalizable (no remaining illegal `onnx.DequantizeLinear`).

2) `convert-onnx-to-krnl` crashed in DimAnalysis (`cast<RankedTensorType>`)
- Root cause:
  - Dynamic graphs can keep unranked tensors around `onnx.Reshape`.
- Fix:
  - Guard reshape-special-case in `Dialect/ONNX/ONNXDimAnalysis.cpp`:
    - use `dyn_cast<RankedTensorType>`
    - skip special-case when data/output are unranked.

3) `convert-onnx-to-krnl` crashed in elementwise fusion / type conversion on unranked tensors
- Root cause:
  - Fusion helper and krnl type conversion assumed ranked tensors.
- Fix:
  - In `Conversion/ONNXToKrnl/Math/Elementwise.cpp`, skip fusion if def/use output is not ranked.
  - Avoid introducing unstable fake-QDQ ONNX elementwise chains for this path.

4) Ensure no QDQ int8 ONNX ops remain after ONNX->Posit stage
- Verified:
  - `onnx.QuantizeLinear`, `onnx.DequantizeLinear`, `onnx.QLinearMatMul` are removed in generated `*.pos.mlir` for Mobilenet/Resnet.

Status:
- Verified on local models:
  - `mobilenetv2-12-qdq.onnx.mlir`
  - `resnet50-v1-12-qdq.onnx.mlir`
- Both pass:
  - `convert-onnx-to-posit -> convert-posit-to-krnl -> convert-onnx-to-krnl -> convert-posit-to-krnl`
  - and `krnl -> llvm`.

## 2026-03-16: p8e0 SO Rebuild + Fractional Q-Domain Update

Goal:
- Confirm `p8e0` can generate a valid `.so` and run `limit=1`.
- Preserve fractional value in `DQ(Q(x))` path (avoid early `round/clamp` to int when targeting posit flow).

Changes:
1) `DQ(Q(x))` behavior update
- File:
  - `Conversion/ONNXToPosit/Pattern/Math.cpp`
- Change:
  - In `ONNXDequantizeLinearOpLowering`, `Case A: DQ(Q(...))` now defaults to:
    - `Q` input tensor (`f32/posit`) -> `posit.from_f32`
  - This keeps fractional quant-domain information (example: `7.2` is not collapsed to `7` at this stage).
  - Strict ONNX QDQ round/clamp path is still kept in code behind local switch:
    - `preferFractionalQDomain = true` (current default).

2) Rebuild verification for broken tiny `p8e0.so`
- Symptom observed before rebuild:
  - `temp/mobilenet11_temp/mobilenetv2-12-qdq-p8e0.so` was abnormally small (~460KB), causing runtime symbol issues.
- Rebuilt with full pipeline:
  - ONNX->Posit->Krnl
  - Krnl->LLVM
  - LLVM dialect -> `.ll`
  - `.ll` + `posit_runtime.cpp` -> `.so`
- Result after rebuild:
  - `temp/mobilenet11_temp/ir/mobilenetv2-12-qdq-p8e0.llvm.mlir` ~25MB
  - `temp/mobilenet11_temp/mobilenetv2-12-qdq-p8e0.so` ~11MB
  - `_mlir_ciface_main_graph` symbol exists.

3) `limit=1` run verification
- Command:
  - `./time_model_single_format.sh --model-name mobilenetv2-12 --format p8e0 --out-dir ./temp/mobilenet11_temp --txt-dir ./temp/imagenette_val_224 --label-map ./temp/imagenette_val_224_labels.txt --baseline nqdq-f32 --limit 1 --warmup 0 --iters 1 --timeout-sec 0 --progress 1 --no-benchmark`
- Result:
  - `samples_used=1`
  - `samples_failed=0`
  - run completed successfully (no segfault / no missing main symbol).

## 2026-03-16: Reduce f32<->posit Ping-Pong (Keep Needed Low-Precision Sections)

Goal:
- Keep only necessary low-precision path behavior while reducing redundant `posit.to_f32` / `posit.from_f32` round-trips.
- Preserve successful `.so` generation and `limit=1` execution.

Problems observed:
1) Naive "only Conv/MatMul/Gemm illegal" strategy made conversion worse.
- Symptom:
  - `from_f32` and `to_f32` counts increased due mixed ONNX/Posit boundaries.
- Measured on `mobilenetv2-12-qdq-p8e0.krnl.mlir`:
  - before tweak: `from=169`, `to=100`, local ping-pong pairs (`to->from` within ~12 lines) = `63`
  - naive partial-convert attempt: `from=236`, `to=167`, ping-pong=`91`
- Conclusion:
  - Partial conversion at wrong boundaries creates more cast-chain bouncing.

2) Redundant round-trip pattern existed in PositToKrnl lowering:
- Pattern:
  - `posit.from_f32(posit.to_f32(x))`
  - `posit.to_f32(posit.from_f32(x))`
- These can be folded safely at lowering time.

Applied fix:
1) Keep quantized subgraph conversion coverage (restore full ONNX op illegal set used by quantized region), but keep ONNX constants legal in ONNXToPosit pass.
2) Add explicit round-trip folding in `Conversion/PositToKrnl/Pattern/Math.cpp`:
   - In `PositFromF32OpLowering`: fold `from_f32(to_f32(x)) -> x`.
   - In `PositToF32OpLowering`: fold `to_f32(from_f32(x)) -> x`.
   - If needed, insert `unrealized_conversion_cast` to expected memref type.

Result after fix (same model/format):
- `mobilenetv2-12-qdq-p8e0.krnl.mlir` metrics:
  - `from_f32`: `169 -> 106`
  - `to_f32`: `100 -> 100` (unchanged)
  - local ping-pong pairs: `63 -> 27`
- Build/run status:
  - `.so` generation successful.
  - `limit=1` successful (`samples_failed=0`).
  - Main entrypoint symbol present and runnable.

Notes:
- This update reduces redundant conversions without changing current runtime metric profile yet.
- Accuracy/top1 still needs separate numeric-path tuning (scale/zp semantics and kernel numeric behavior), but this pass removes non-essential conversion churn first.

## 2026-03-17: Apply focus fixes #3/#4/#5 (no mixed-precision)

Scope:
- Keep single-format posit path (no mixed precision).
- Apply only:
  - #3 reduce redundant conversion churn,
  - #4 align dequantize axis broadcasting path,
  - #5 keep boundary decode default on f32 side for stable metric comparisons.

Code changes:
1) `Conversion/ONNXToPosit/Pattern/Math.cpp`
- `DQ(Q(x))` default switched to strict QDQ semantics:
  - `preferFractionalQDomain = false`

2) `Conversion/PositToKrnl/Pattern/Math.cpp`
- Added `stripMemrefAndUnrealizedCast(...)` and applied to `PositFromF32OpLowering` / `PositToF32OpLowering` fold checks.
- Added `makeF32Dense1D(...)`.
- DQ axis-path selection updated:
  - from `numElements > 1`
  - to presence of `scale_values` / `zero_point_values` attrs.
- When axis-path selected but `scale_values` missing, generate scalar dense scale global.

3) `run_time.cpp`
- Changed default `mainOutType`:
  - from `OutType::P8E0`
  - to `OutType::F32`

Rebuild:
- Rebuilt `onnx-mlir-opt`.
- Rebuilt 11 `.so` sets for:
  - `temp/mobilenet11_temp`
  - `temp/resnet50-11_temp`

Validation (`limit=1`, sample `img_00000.txt`):
- Mobilenet (baseline=`nqdq-f32`):
  - `p8e0`: Top1Match 0%, GT Top1 0%
  - `p16e1`: Top1Match 100%, GT Top1 100%
  - `p32e2`: Top1Match 100%, GT Top1 100%
  - `qdq-f32(int8 path)`: Top1Match 100%, GT Top1 100%
  - `nqdq-f32`: GT Top1 100%

- ResNet50 (baseline=`nqdq-f32`):
  - `p8e0`: Top1Match 0%, GT Top1 0%
  - `p16e1`: Top1Match 100%, GT Top1 100%
  - `p32e2`: no valid sample parsed in this run (`rc=143`, manually terminated due very long runtime)
  - `qdq-f32(int8 path)`: Top1Match 100%, GT Top1 100%
  - `nqdq-f32`: GT Top1 100%

## 2026-03-19: SoftPosit format coverage + p8 quire path update

Scope:
- Keep SoftPosit as default backend.
- Restore default posit format set to 9 formats: `p8e0~p32e2`.
- Make `p8e0/p8e1/p8e2` all support quire-mode switch (`--quire on|off`).

Code changes:
1) `posit_runtime.cpp`
- Added SoftPosit dynamic format handlers:
  - `FmtP16E2ViaPX2` (x=16)
  - `FmtP32E1ViaPX1` (x=32)
- Added `DotAccumulator<FmtP8E1ViaPX1>`:
  - `--quire on`: high-precision accumulator (`long double`) and single cast-back.
  - `--quire off`: original posit-domain `acc += a*b`.
- Added optional universal fallback type wrappers (`POSIT_USE_UNIVERSAL_FALLBACK`) for:
  - `p16e0`, `p32e0`
- Exported/forwarded runtime symbols now cover all 9 formats in softposit+fallback build.

2) `run_time.cpp`
- Added softposit decode support for:
  - `p16e2` via `pX2_to_pX2(...,16)`
  - `p32e1` via `pX1_to_pX1(...,32)`
- Added optional universal fallback decode path for:
  - `p16e0`, `p32e0` (`POSIT_USE_UNIVERSAL_FALLBACK`)
- Kept `--quire on|off` toggle path.

3) `bash/build_model11_sos.sh`
- Default formats changed back to 9:
  - `p8e0,p8e1,p8e2,p16e0,p16e1,p16e2,p32e0,p32e1,p32e2`
- Added option:
  - `--softposit-universal-fallback on|off` (default `on`)
- SoftPosit backend behavior:
  - Uses SoftPosit for available formats.
  - Uses fallback (`POSIT_USE_UNIVERSAL_FALLBACK`) for `p16e0/p32e0` if requested.
  - If fallback is off and requested formats include `p16e0/p32e0`, script errors with guidance.

Validation:
- Rebuilt `src/.deps/softposit-px1/libsoftposit_full.so` successfully.
- Library symbols confirmed:
  - present: `pX1_add`, `pX1_to_pX1`, `qX2_fdp_add`, `qX2_to_pX2`
  - missing: `qX1_fdp_add`, `qX1_to_pX1` (header declarations only; no C implementation in current SoftPosit tree).
- Compiled `posit_runtime.cpp` + `run_time.cpp` successfully with:
  - `-DPOSIT_USE_SOFTPOSIT_PX1 -DPOSIT_USE_SOFTPOSIT_PX2 -DPOSIT_USE_UNIVERSAL_FALLBACK`
- Export symbol check on temp runtime `.so` confirms 9-format symbol availability (from/to/gemm/conv2d entries).
