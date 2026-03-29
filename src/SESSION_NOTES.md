# SESSION_NOTES (Posit + ONNX-MLIR)

Last updated: 2026-03-29
Owner: tungtung9487

## 1) Current Git Branches (for resume on another machine)

- onnx-mlir fork:
  - Repo: https://github.com/tungtung9487/onnx-mlir
  - Branch: `tungtung9487/posit-work-20260329`
  - Commit: `04391764`
- llvm-project fork:
  - Repo: https://github.com/tungtung9487/llvm-project
  - Branch: `tungtung9487/llvm-snapshot-20260329`
  - Commit (HEAD when pushed): `113f01aa82d0`

## 2) Toolchain Actually Used

From current environment:

- `ONNX_MLIR_BIN=/home/lai/mlir_toy/onnx-mlir/build/Release/bin/onnx-mlir`
- `ONNX_MLIR_OPT_BIN=/home/lai/mlir_toy/onnx-mlir/build/Release/bin/onnx-mlir-opt`
- `MLIR_TRANSLATE_BIN=/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate`

Important:
- Must use matching `mlir-translate` from the same LLVM build family.
- Older `/usr/local/bin/mlir-translate` can fail on syntax like `inbounds|nuw`.

## 3) Main Build/Lowering Flow

### 3.1 ONNX -> ONNX MLIR (preferred import mode used by user)

```bash
./onnx-mlir --EmitONNXIR --mlir-elide-resource-strings-if-larger=1000000000 -o <out_base> <model.onnx>
```

### 3.2 ONNX/Posit -> Krnl

```bash
../build/Release/bin/onnx-mlir-opt <model>.onnx.mlir \
  --shape-inference \
  --convert-onnx-to-posit \
  --convert-posit-to-krnl \
  --canonicalize \
  --convert-onnx-to-krnl \
  --convert-posit-to-krnl \
  --canonicalize \
  --posit-format=p8e0 \
  -o <model>.krnl.mlir
```

### 3.3 Krnl -> LLVM dialect

```bash
../build/Release/bin/onnx-mlir-opt <model>.krnl.mlir \
  --canonicalize \
  --convert-krnl-to-affine \
  --convert-krnl-to-llvm \
  --reconcile-unrealized-casts \
  -o <model>.llvm.mlir
```

### 3.4 LLVM dialect -> .ll

```bash
/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate \
  --mlir-to-llvmir <model>.llvm.mlir -o <model>.ll
```

### 3.5 .ll -> .so

Handled by build scripts with `clang++` + `src/posit_runtime.cpp`.

## 4) Build Scripts in Use

- Core: `src/bash/build_model11_sos.sh`
- Wrappers:
  - `src/bash/build_mobilenet11_sos.sh`
  - `src/bash/build_resnet50_11_sos.sh`
- Extra formats:
  - `src/bash/build_model_extra_sos.sh`
  - `src/bash/build_mobilenet_extra_sos.sh`
  - `src/bash/build_resnet50_extra_sos.sh`
- Runtime benchmark/compare:
  - `src/bash/time_model11_dataset_parallel.sh`
  - `src/bash/time_mobilenet11_dataset_parallel.sh`
  - `src/bash/time_resnet50_11_dataset_parallel.sh`

## 5) Major Errors Encountered and Fixes Applied

Detailed history is in:
- `src/posit_mobilenet_resnet_fix_log.md`

Key points:

1. `onnx.Clip` / `onnx.Dim` unsupported in ONNXToPosit on large dynamic graphs
- Fix: dynamic-shape fallback path in `Conversion/ONNXToPosit/ONNXToPosit.cpp`.

2. `onnx.Constant` illegal in PositToKrnl when no posit ops exist
- Fix: early no-op guard in `Conversion/PositToKrnl/PositToKrnl.cpp`.

3. `krnl.define_loops` illegal in krnl->llvm
- Fix: pipeline adds `--convert-krnl-to-affine` before `--convert-krnl-to-llvm`.

4. `builtin.unrealized_conversion_cast` remained in LLVM dialect
- Fix: add `--reconcile-unrealized-casts`.

5. `mlir-translate` parse mismatch (`inbounds|nuw`)
- Fix: use `/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate`.

6. Script pipeline missing ONNX->Krnl lowering stage
- Fix: `build_model11_sos.sh` now includes `--convert-onnx-to-krnl` in stage A.

7. Binary/runtime path brittleness
- Fix: auto-detection for onnx-mlir / onnx-mlir-opt / mlir-translate and runtime source fallback.

8. Wrapper scripts only assumed dataset ONNX path
- Fix: wrappers now auto-select ONNX if available, else local `src/*.onnx.mlir`.

9. Ping-pong (`posit.to_f32` <-> `posit.from_f32`) too frequent
- Fix: fold rules in `Conversion/PositToKrnl/Pattern/Math.cpp`; measured reductions logged.

10. QDQ axis/scale/zp handling alignment
- Fix: dequantize axis path and scale/zero-point handling updated in ONNXToPosit / PositToKrnl paths.

## 6) Current Known Practical Status

- ResNet/MobileNet:
  - Stable path is to use checked-in or user-generated `--EmitONNXIR` MLIR as input to build scripts.
- MNIST (int8 QDQ import path):
  - still has known legalize issues around `posit.dequantize_linear` in generic `build_model11_sos.sh` path.

## 7) Temp Upload Policy Used for Git Push

For the latest push:
- Included: code + logs + notes + selected `src/temp` text/code artifacts.
- Excluded: over-100MB artifacts that can break GitHub push.

Observed >100,000,000-byte non-`.mlir/.ll` in `src/temp`:
- `src/temp/model/resnet50-v1-12.onnx` (102,576,593)
- `src/temp/resnet50-11_temp/resnet50-v1-12-nqdq-f32.so` (102,321,016)
- `src/temp/resnet50-11_quire/resnet50-v1-12-nqdq-f32.so` (102,321,016)
- `src/temp/resnet50-11_final/resnet50-v1-12-nqdq-f32.so` (102,321,016)
- `src/temp/resnet50-11/resnet50-v1-12-nqdq-f32.so` (102,321,016)

If needed later: use Git LFS for those binaries.

## 8) Quick Resume Checklist (new machine)

1. Clone both forks + checkout branches in section 1.
2. Build/use matching toolchain binaries:
   - onnx-mlir + onnx-mlir-opt from onnx-mlir build
   - mlir-translate from llvm-project build
3. Verify env:

```bash
echo "$ONNX_MLIR_BIN"
echo "$ONNX_MLIR_OPT_BIN"
echo "$MLIR_TRANSLATE_BIN"
```

4. Run model build scripts from `src/bash`.
5. Run dataset compare scripts with `run_time_sp`.

