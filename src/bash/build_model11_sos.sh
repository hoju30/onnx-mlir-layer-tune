#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  build_model11_sos.sh --model-name NAME \
    [--qdq-mlir PATH | --qdq-onnx PATH] \
    [--qdq-f32-mlir PATH] \
    [--nqdq-mlir PATH | --nqdq-onnx PATH] \
    [--shape-info STR] [--use-onnx-ir] \
    [--strict-qdq-mode] [--non-strict-qdq-mode] \
    [--align-to-int8-qdomain] [--no-align-to-int8-qdomain] \
    [--allow-qdq-fallback-to-nqdq] \
    [--continue-on-posit-fail] \
    [--keep-stage-logs] [--no-stage-logs] \
    [--skip-f32-baselines] \
    [--posit-source qdq|nqdq|both] [--posit-compact-constants] [--no-posit-compact-constants] \
    [--runtime-format-scope full|single] \
    [--runtime-qalign-mode full|alps-only] \
    [--runtime-mixed-accum runtime|off|p16e2|p32e2] \
    [--runtime-output-alps off|sampled|full|offline] \
    [--keep-ir] [--ir-dir PATH] \
    [--out-dir PATH] [--cruntime-lib-dir PATH] \
    [--backend softposit|universal] [--posit-formats CSV] \
    [--softposit-include-dir PATH] [--softposit-lib PATH] \
    [--universal-include-dir PATH]

Notes:
  - Non-strict QDQ mode is ON by default (prefer direct f32->posit from Q source).
  - Use --strict-qdq-mode to prioritize strict ONNX QDQ lowering.
  - INT8 q-domain alignment is OFF by default.
  - Use --align-to-int8-qdomain to require strict q-domain materialization.
  - Stage logs are OFF by default to save disk space.
  - Use --keep-stage-logs to retain stage logs and pass IR dumps.

Output:
  <out-dir>/<NAME>-qdq-<posit-format>.so    (QDQ posit variants)
  <out-dir>/<NAME>-nqdq-<posit-format>.so   (NQDQ/f32 posit variants when --posit-source nqdq|both)
  <out-dir>/<NAME>-qdq-f32.so
  <out-dir>/<NAME>-nqdq-f32.so
  <out-dir>/run_time_sp
  (IR files are NOT kept by default; use --keep-ir to preserve.)
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${src_root}/.." && pwd)"
deps_root="${src_root}/.deps"
workspace_root="$(cd "${project_root}/.." && pwd)"

model_name=""
qdq_mlir=""
qdq_onnx=""
qdq_f32_mlir=""
nqdq_mlir=""
nqdq_onnx=""
out_dir=""
posit_source="${POSIT_SOURCE:-qdq}"
posit_compact_constants="${POSIT_COMPACT_CONSTANTS:-auto}"
softposit_root="${SOFTPOSIT_ROOT:-${deps_root}/SoftPosit}"
universal_inc="${UNIVERSAL_INCLUDE_DIR:-${deps_root}/universal/include/sw}"
softposit_inc="${SOFTPOSIT_INCLUDE_DIR:-${softposit_root}/source/include}"
softposit_lib="${SOFTPOSIT_LIB_PATH:-}"
backend="${POSIT_BACKEND:-universal}"
posit_formats_csv="${POSIT_FORMATS:-p8e0,p8e1,p8e2,p16e0,p16e1,p16e2,p32e0,p32e1,p32e2}"
runtime_format_scope="${POSIT_RUNTIME_FORMAT_SCOPE:-full}"
runtime_qalign_mode="${POSIT_RUNTIME_QALIGN_MODE:-full}"
runtime_mixed_accum="${POSIT_RUNTIME_MIXED_ACCUM:-runtime}"
runtime_output_alps="${POSIT_RUNTIME_OUTPUT_ALPS_MODE:-off}"
cruntime_lib_dir="${CRUNTIME_LIB_DIR:-}"
shape_info=""
# Default to ONNXIR import. The basic importer is smaller, but ONNXIR has been
# much more robust for recent QDQ/NQDQ ResNet flows, especially mixed Posit
# lowering where tensor shape/type detail matters later in the pipeline.
import_mode="ir"
continue_on_posit_fail=0
allow_qdq_fallback_to_nqdq=0
align_to_int8_qdomain=0
strict_qdq_mode=0
skip_f32_baselines=0
keep_ir=0
keep_stage_logs=0
ir_dir_override=""
onnx_mlir_bin="${ONNX_MLIR_BIN:-}"
onnx_mlir_opt_bin="${ONNX_MLIR_OPT_BIN:-}"
mlir_translate_bin="${MLIR_TRANSLATE_BIN:-}"
clang_bin="${CLANG_BIN:-}"
clangxx_bin="${CLANGXX_BIN:-}"
cxx_bin="${CXX_BIN:-}"
mlir_include_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-name)
    [[ $# -ge 2 ]] || { echo "ERROR: --model-name needs a value"; exit 2; }
    model_name="$2"
    shift 2
    ;;
  --qdq-mlir)
    [[ $# -ge 2 ]] || { echo "ERROR: --qdq-mlir needs a path"; exit 2; }
    qdq_mlir="$2"
    shift 2
    ;;
  --qdq-onnx)
    [[ $# -ge 2 ]] || { echo "ERROR: --qdq-onnx needs a path"; exit 2; }
    qdq_onnx="$2"
    shift 2
    ;;
  --qdq-f32-mlir)
    [[ $# -ge 2 ]] || { echo "ERROR: --qdq-f32-mlir needs a path"; exit 2; }
    qdq_f32_mlir="$2"
    shift 2
    ;;
  --nqdq-mlir)
    [[ $# -ge 2 ]] || { echo "ERROR: --nqdq-mlir needs a path"; exit 2; }
    nqdq_mlir="$2"
    shift 2
    ;;
  --nqdq-onnx)
    [[ $# -ge 2 ]] || { echo "ERROR: --nqdq-onnx needs a path"; exit 2; }
    nqdq_onnx="$2"
    shift 2
    ;;
  --shape-info)
    [[ $# -ge 2 ]] || { echo "ERROR: --shape-info needs a value"; exit 2; }
    shape_info="$2"
    shift 2
    ;;
  --use-onnx-ir)
    import_mode="ir"
    shift
    ;;
  --continue-on-posit-fail)
    continue_on_posit_fail=1
    shift
    ;;
  --keep-stage-logs)
    keep_stage_logs=1
    shift
    ;;
  --no-stage-logs)
    keep_stage_logs=0
    shift
    ;;
  --allow-qdq-fallback-to-nqdq)
    allow_qdq_fallback_to_nqdq=1
    shift
    ;;
  --strict-qdq-mode)
    strict_qdq_mode=1
    align_to_int8_qdomain=1
    shift
    ;;
  --non-strict-qdq-mode)
    strict_qdq_mode=0
    shift
    ;;
  --align-to-int8-qdomain)
    align_to_int8_qdomain=1
    shift
    ;;
  --no-align-to-int8-qdomain)
    align_to_int8_qdomain=0
    shift
    ;;
  --skip-f32-baselines)
    skip_f32_baselines=1
    shift
    ;;
  --posit-source)
    [[ $# -ge 2 ]] || { echo "ERROR: --posit-source needs qdq|nqdq|both"; exit 2; }
    posit_source="$2"
    shift 2
    ;;
  --posit-compact-constants)
    posit_compact_constants="1"
    shift
    ;;
  --no-posit-compact-constants)
    posit_compact_constants="0"
    shift
    ;;
  --runtime-format-scope)
    [[ $# -ge 2 ]] || { echo "ERROR: --runtime-format-scope needs full|single"; exit 2; }
    runtime_format_scope="$2"
    shift 2
    ;;
  --runtime-qalign-mode)
    [[ $# -ge 2 ]] || { echo "ERROR: --runtime-qalign-mode needs full|alps-only"; exit 2; }
    runtime_qalign_mode="$2"
    shift 2
    ;;
  --runtime-mixed-accum)
    [[ $# -ge 2 ]] || { echo "ERROR: --runtime-mixed-accum needs runtime|off|p16e2|p32e2"; exit 2; }
    runtime_mixed_accum="$2"
    shift 2
    ;;
  --runtime-output-alps)
    [[ $# -ge 2 ]] || { echo "ERROR: --runtime-output-alps needs off|sampled|full|offline"; exit 2; }
    runtime_output_alps="$2"
    shift 2
    ;;
  --keep-ir)
    keep_ir=1
    shift
    ;;
  --ir-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --ir-dir needs a path"; exit 2; }
    ir_dir_override="$2"
    keep_ir=1
    shift 2
    ;;
  --out-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --out-dir needs a path"; exit 2; }
    out_dir="$2"
    shift 2
    ;;
  --universal-include-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --universal-include-dir needs a path"; exit 2; }
    universal_inc="$2"
    shift 2
    ;;
  --softposit-include-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --softposit-include-dir needs a path"; exit 2; }
    softposit_inc="$2"
    shift 2
    ;;
  --softposit-lib)
    [[ $# -ge 2 ]] || { echo "ERROR: --softposit-lib needs a path"; exit 2; }
    softposit_lib="$2"
    shift 2
    ;;
  --backend)
    [[ $# -ge 2 ]] || { echo "ERROR: --backend needs a value"; exit 2; }
    backend="$2"
    shift 2
    ;;
  --posit-formats)
    [[ $# -ge 2 ]] || { echo "ERROR: --posit-formats needs CSV"; exit 2; }
    posit_formats_csv="$2"
    shift 2
    ;;
  --cruntime-lib-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --cruntime-lib-dir needs a path"; exit 2; }
    cruntime_lib_dir="$2"
    shift 2
    ;;
  --onnx-mlir)
    [[ $# -ge 2 ]] || { echo "ERROR: --onnx-mlir needs a path"; exit 2; }
    onnx_mlir_bin="$2"
    shift 2
    ;;
  --onnx-mlir-opt)
    [[ $# -ge 2 ]] || { echo "ERROR: --onnx-mlir-opt needs a path"; exit 2; }
    onnx_mlir_opt_bin="$2"
    shift 2
    ;;
  --mlir-translate)
    [[ $# -ge 2 ]] || { echo "ERROR: --mlir-translate needs a path"; exit 2; }
    mlir_translate_bin="$2"
    shift 2
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown arg: $1"
    usage
    exit 2
    ;;
  esac
done

if [[ -z "${onnx_mlir_bin}" ]]; then
  for cand in \
    "${project_root}/build/Release/bin/onnx-mlir" \
    "${project_root}/build/Debug/bin/onnx-mlir" \
    "${src_root}/build/Release/bin/onnx-mlir" \
    "${src_root}/build/Debug/bin/onnx-mlir" \
    "${project_root}/onnx-mlir" \
    "${src_root}/onnx-mlir"
  do
    if [[ -x "${cand}" ]]; then
      onnx_mlir_bin="${cand}"
      break
    fi
  done
fi

if [[ -z "${onnx_mlir_opt_bin}" ]]; then
  for cand in \
    "${project_root}/build/Release/bin/onnx-mlir-opt" \
    "${project_root}/build/Debug/bin/onnx-mlir-opt" \
    "${src_root}/build/Release/bin/onnx-mlir-opt" \
    "${src_root}/build/Debug/bin/onnx-mlir-opt" \
    "${project_root}/onnx-mlir-opt" \
    "${src_root}/onnx-mlir-opt"
  do
    if [[ -x "${cand}" ]]; then
      onnx_mlir_opt_bin="${cand}"
      break
    fi
  done
fi

if [[ -z "${mlir_translate_bin}" ]]; then
  for cand in \
    "${workspace_root}/llvm-project/build/bin/mlir-translate" \
    "/home/lai/mlir_toy/llvm-project/build/bin/mlir-translate" \
    "${project_root}/build/Release/bin/mlir-translate" \
    "${project_root}/build/Debug/bin/mlir-translate" \
    "${src_root}/build/Release/bin/mlir-translate" \
    "${src_root}/build/Debug/bin/mlir-translate"
  do
    if [[ -x "${cand}" ]]; then
      mlir_translate_bin="${cand}"
      break
    fi
  done
  if [[ -z "${mlir_translate_bin}" ]] && command -v mlir-translate >/dev/null 2>&1; then
    mlir_translate_bin="$(command -v mlir-translate)"
  fi
fi

if [[ -z "${clangxx_bin}" ]]; then
  for cand in \
    "${workspace_root}/llvm-project/build/bin/clang++" \
    "${project_root}/build/Release/bin/clang++" \
    "${project_root}/build/Debug/bin/clang++"
  do
    if [[ -x "${cand}" ]]; then
      clangxx_bin="${cand}"
      break
    fi
  done
  if [[ -z "${clangxx_bin}" ]] && command -v clang++ >/dev/null 2>&1; then
    clangxx_bin="$(command -v clang++)"
  fi
fi

if [[ -z "${cxx_bin}" ]]; then
  if [[ -n "${clangxx_bin}" ]]; then
    cxx_bin="${clangxx_bin}"
  elif command -v c++ >/dev/null 2>&1; then
    cxx_bin="$(command -v c++)"
  else
    cxx_bin="$(command -v g++)"
  fi
fi

if [[ -z "${model_name}" ]]; then
  usage
  exit 2
fi
if [[ -z "${qdq_mlir}" && -z "${qdq_onnx}" ]]; then
  echo "ERROR: need --qdq-mlir or --qdq-onnx"
  exit 2
fi
if [[ -z "${nqdq_mlir}" && -z "${nqdq_onnx}" ]]; then
  echo "ERROR: need --nqdq-mlir or --nqdq-onnx"
  exit 2
fi
if [[ -z "${out_dir}" ]]; then
  out_dir="${project_root}/build/${model_name}-11"
fi

if [[ -n "${qdq_mlir}" && ! -f "${qdq_mlir}" ]]; then
  echo "ERROR: QDQ mlir not found: ${qdq_mlir}"
  exit 2
fi
if [[ -n "${qdq_f32_mlir}" && ! -f "${qdq_f32_mlir}" ]]; then
  echo "ERROR: qdq f32 mlir not found: ${qdq_f32_mlir}"
  exit 2
fi
if [[ -n "${qdq_onnx}" && ! -f "${qdq_onnx}" ]]; then
  echo "ERROR: QDQ onnx not found: ${qdq_onnx}"
  exit 2
fi
if [[ -n "${nqdq_mlir}" && ! -f "${nqdq_mlir}" ]]; then
  echo "ERROR: non-QDQ mlir not found: ${nqdq_mlir}"
  exit 2
fi
if [[ -n "${nqdq_onnx}" && ! -f "${nqdq_onnx}" ]]; then
  echo "ERROR: non-QDQ onnx not found: ${nqdq_onnx}"
  exit 2
fi
if [[ ! -x "${onnx_mlir_bin}" ]]; then
  echo "ERROR: onnx-mlir not found/executable: ${onnx_mlir_bin}"
  exit 2
fi
if [[ ! -x "${onnx_mlir_opt_bin}" ]]; then
  echo "ERROR: onnx-mlir-opt not found/executable: ${onnx_mlir_opt_bin}"
  exit 2
fi
if [[ ! -x "${mlir_translate_bin}" ]]; then
  echo "ERROR: mlir-translate not found/executable: ${mlir_translate_bin}"
  exit 2
fi
if [[ ! -x "${clangxx_bin}" ]]; then
  echo "ERROR: clang++ not found/executable: ${clangxx_bin}"
  exit 2
fi
if [[ ! -x "${cxx_bin}" ]]; then
  echo "ERROR: C++ compiler not found/executable: ${cxx_bin}"
  exit 2
fi

for cand in \
  "${workspace_root}/llvm-project/build/include" \
  "${workspace_root}/llvm-project/mlir/include" \
  "${workspace_root}/llvm-project/llvm/include"
do
  if [[ -d "${cand}" ]]; then
    mlir_include_args+=(-I"${cand}")
  fi
done

if [[ -z "${cruntime_lib_dir}" ]]; then
  for cand in \
    "${project_root}/build/Release/lib" \
    "${src_root}/build/Release/lib" \
    "${project_root}/build/lib" \
    "${src_root}/build/lib"
  do
    if [[ -f "${cand}/libcruntime.a" || -f "${cand}/libcruntime.so" ]]; then
      cruntime_lib_dir="${cand}"
      break
    fi
  done
fi
if [[ -z "${cruntime_lib_dir}" ]]; then
  echo "ERROR: cannot find libcruntime.{a,so}; set --cruntime-lib-dir"
  exit 2
fi
cruntime_static_lib="${cruntime_lib_dir}/libcruntime.a"
cruntime_shared_lib="${cruntime_lib_dir}/libcruntime.so"
if [[ -f "${cruntime_static_lib}" ]]; then
  cruntime_link_args=("${cruntime_static_lib}")
elif [[ -f "${cruntime_shared_lib}" ]]; then
  cruntime_link_args=(-L"${cruntime_lib_dir}" -lcruntime "-Wl,-rpath,${cruntime_lib_dir}")
else
  echo "ERROR: cannot find usable libcruntime in ${cruntime_lib_dir}"
  exit 2
fi

if [[ "${backend}" != "softposit" && "${backend}" != "universal" ]]; then
  echo "ERROR: --backend must be softposit or universal"
  exit 2
fi

if [[ "${posit_source}" != "qdq" && "${posit_source}" != "nqdq" && "${posit_source}" != "both" ]]; then
  echo "ERROR: --posit-source must be qdq, nqdq, or both"
  exit 2
fi

if [[ "${backend}" == "universal" ]]; then
  if [[ ! -f "${universal_inc}/universal/number/posit/posit_fwd.hpp" ]]; then
    for cand in \
      "${deps_root}/universal/include/sw" \
      "${project_root}/.deps/universal/include/sw" \
      "/usr/local/include/sw"
    do
      if [[ -f "${cand}/universal/number/posit/posit_fwd.hpp" ]]; then
        universal_inc="${cand}"
        break
      fi
    done
  fi
  if [[ ! -f "${universal_inc}/universal/number/posit/posit_fwd.hpp" ]]; then
    echo "ERROR: universal headers not found under: ${universal_inc}"
    echo "       Try: ${script_dir}/install_posit_deps.sh"
    exit 2
  fi
else
  if [[ ! -f "${softposit_inc}/softposit.h" ]]; then
    for cand in \
      "${softposit_root}/source/include" \
      "${deps_root}/SoftPosit/source/include"
    do
      if [[ -f "${cand}/softposit.h" ]]; then
        softposit_inc="${cand}"
        break
      fi
    done
  fi
  if [[ ! -f "${softposit_inc}/softposit.h" ]]; then
    echo "ERROR: softposit header not found: ${softposit_inc}/softposit.h"
    echo "       Try: ${script_dir}/install_posit_deps.sh"
    exit 2
  fi
  if [[ -z "${softposit_lib}" ]]; then
    for cand in \
      "${deps_root}/softposit-px1/libsoftposit.a" \
      "${src_root}/.deps/softposit-px1/libsoftposit.so" \
      "${src_root}/.deps/softposit-px1/libsoftposit_full.so" \
      "${softposit_root}/build/Linux-x86_64-GCC/libsoftposit.a" \
      "${softposit_root}/build/Linux_x86_64_GCC/libsoftposit.a" \
      "/home/lai/mlir_toy/SoftPosit/SoftPosit/build/Linux-x86_64-GCC/libsoftposit.so" \
      "/home/lai/mlir_toy/SoftPosit/SoftPosit/build/Linux_x86_64_GCC/libsoftposit.so"
    do
      if [[ -f "${cand}" ]]; then
        softposit_lib="${cand}"
        break
      fi
    done
  fi
  if [[ -z "${softposit_lib}" || ! -f "${softposit_lib}" ]]; then
    echo "ERROR: softposit library not found; set --softposit-lib"
    echo "       Try: ${script_dir}/install_posit_deps.sh"
    exit 2
  fi
fi

formats=()
IFS=',' read -r -a _fmt_raw <<< "${posit_formats_csv}"
for f in "${_fmt_raw[@]}"; do
  ff="${f//[[:space:]]/}"
  [[ -n "${ff}" ]] && formats+=("${ff}")
done
if [[ ${#formats[@]} -eq 0 ]]; then
  echo "ERROR: --posit-formats is empty"
  exit 2
fi

case "${runtime_format_scope}" in
  full|single) ;;
  *)
    echo "ERROR: --runtime-format-scope must be full or single"
    exit 2
    ;;
esac
case "${runtime_qalign_mode}" in
  full|alps-only) ;;
  *)
    echo "ERROR: --runtime-qalign-mode must be full or alps-only"
    exit 2
    ;;
esac
case "${runtime_mixed_accum}" in
  runtime|off|p16e2|p32e2) ;;
  *)
    echo "ERROR: --runtime-mixed-accum must be runtime, off, p16e2, or p32e2"
    exit 2
    ;;
esac
case "${runtime_output_alps}" in
  off|sampled|full|offline) ;;
  *)
    echo "ERROR: --runtime-output-alps must be off, sampled, full, or offline"
    exit 2
    ;;
esac

has_softposit_unsupported_format=0
for fmt in "${formats[@]}"; do
  if [[ "${fmt}" == "p16e0" || "${fmt}" == "p32e0" ]]; then
    has_softposit_unsupported_format=1
    break
  fi
done

echo "[config] backend=${backend}"
echo "[config] formats=${formats[*]}"
echo "[config] strict_qdq_mode=${strict_qdq_mode}"
echo "[config] posit_source=${posit_source}"
echo "[config] posit_compact_constants=${posit_compact_constants}"
echo "[config] runtime_format_scope=${runtime_format_scope}"
echo "[config] runtime_qalign_mode=${runtime_qalign_mode}"
echo "[config] runtime_mixed_accum=${runtime_mixed_accum}"
echo "[config] runtime_output_alps=${runtime_output_alps}"
echo "[config] align_to_int8_qdomain=${align_to_int8_qdomain}"
echo "[config] skip_f32_baselines=${skip_f32_baselines}"
if [[ "${skip_f32_baselines}" -eq 0 ]]; then
  echo "[config] include_f32_baselines=qdq-f32,nqdq-f32"
else
  echo "[config] include_f32_baselines=none"
fi
echo "[config] keep_stage_logs=${keep_stage_logs}"
echo "[config] mlir_translate=${mlir_translate_bin}"
echo "[config] clang++=${clangxx_bin}"
echo "[config] cxx=${cxx_bin}"
if [[ ${#mlir_include_args[@]} -gt 0 ]]; then
  echo "[config] mlir_includes=${mlir_include_args[*]}"
fi
if [[ "${backend}" == "softposit" ]]; then
  echo "[config] softposit_inc=${softposit_inc}"
  echo "[config] softposit_lib=${softposit_lib}"
  if [[ ${has_softposit_unsupported_format} -eq 1 ]]; then
    echo "ERROR: strict softposit mode does not support p16e0/p32e0."
    echo "       Use formats from: p8e0,p8e1,p8e2,p16e1,p16e2,p32e1,p32e2"
    echo "       Or switch to --backend universal if you need e0 for 16/32-bit."
    exit 2
  fi
fi

runtime_format_define_for() {
  case "$1" in
    p4e0) echo "POSIT_RUNTIME_FMT_P4E0" ;;
    p4e1) echo "POSIT_RUNTIME_FMT_P4E1" ;;
    p4e2) echo "POSIT_RUNTIME_FMT_P4E2" ;;
    p4e3) echo "POSIT_RUNTIME_FMT_P4E3" ;;
    p5e0) echo "POSIT_RUNTIME_FMT_P5E0" ;;
    p5e1) echo "POSIT_RUNTIME_FMT_P5E1" ;;
    p5e2) echo "POSIT_RUNTIME_FMT_P5E2" ;;
    p5e3) echo "POSIT_RUNTIME_FMT_P5E3" ;;
    p6e0) echo "POSIT_RUNTIME_FMT_P6E0" ;;
    p6e1) echo "POSIT_RUNTIME_FMT_P6E1" ;;
    p6e2) echo "POSIT_RUNTIME_FMT_P6E2" ;;
    p6e3) echo "POSIT_RUNTIME_FMT_P6E3" ;;
    p7e0) echo "POSIT_RUNTIME_FMT_P7E0" ;;
    p7e1) echo "POSIT_RUNTIME_FMT_P7E1" ;;
    p7e2) echo "POSIT_RUNTIME_FMT_P7E2" ;;
    p7e3) echo "POSIT_RUNTIME_FMT_P7E3" ;;
    p8e0) echo "POSIT_RUNTIME_FMT_P8E0" ;;
    p8e1) echo "POSIT_RUNTIME_FMT_P8E1" ;;
    p8e2) echo "POSIT_RUNTIME_FMT_P8E2" ;;
    p9e0) echo "POSIT_RUNTIME_FMT_P9E0" ;;
    p9e1) echo "POSIT_RUNTIME_FMT_P9E1" ;;
    p9e2) echo "POSIT_RUNTIME_FMT_P9E2" ;;
    p9e3) echo "POSIT_RUNTIME_FMT_P9E3" ;;
    p16e0) echo "POSIT_RUNTIME_FMT_P16E0" ;;
    p16e1) echo "POSIT_RUNTIME_FMT_P16E1" ;;
    p16e2) echo "POSIT_RUNTIME_FMT_P16E2" ;;
    p32e0) echo "POSIT_RUNTIME_FMT_P32E0" ;;
    p32e1) echo "POSIT_RUNTIME_FMT_P32E1" ;;
    p32e2) echo "POSIT_RUNTIME_FMT_P32E2" ;;
    *) return 1 ;;
  esac
}

append_runtime_string_define() {
  local macro="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    runtime_compile_args+=("-D${macro}=\"${value}\"")
  fi
}

ir_dir="${out_dir}/ir"
cleanup_ir_dir=0
cleanup_stage_log_dir=0
stage_log_root=""
mkdir -p "${out_dir}"
if [[ "${keep_ir}" -eq 1 ]]; then
  if [[ -n "${ir_dir_override}" ]]; then
    ir_dir="${ir_dir_override}"
  fi
  mkdir -p "${ir_dir}"
else
  ir_dir="$(mktemp -d "/tmp/${model_name}.ir.XXXXXX")"
  cleanup_ir_dir=1
fi

cleanup_ir() {
  if [[ "${cleanup_ir_dir}" -eq 1 && -n "${ir_dir}" && -d "${ir_dir}" ]]; then
    rm -rf "${ir_dir}"
  fi
  if [[ "${cleanup_stage_log_dir}" -eq 1 && -n "${stage_log_root}" && -d "${stage_log_root}" ]]; then
    rm -rf "${stage_log_root}"
  fi
}
trap cleanup_ir EXIT

import_onnx_to_mlir() {
  local src_onnx="$1"
  local dst_base="$2"
  local mode_flag="--EmitONNXBasic"
  if [[ "${import_mode}" == "ir" ]]; then
    mode_flag="--EmitONNXIR"
  fi

  local cmd=("${onnx_mlir_bin}" "${mode_flag}" -o "${dst_base}")
  cmd+=(
    "--mlir-elide-resource-strings-if-larger=1000000000"
    "--mlir-elide-elementsattrs-if-larger=1000000000"
  )
  if [[ -n "${shape_info}" ]]; then
    cmd+=("--shapeInformation=${shape_info}")
  fi
  cmd+=("${src_onnx}")

  "${cmd[@]}"
}

if [[ -z "${qdq_mlir}" && -n "${qdq_onnx}" ]]; then
  qdq_import_base="${ir_dir}/${model_name}-qdq-import"
  qdq_mlir="${qdq_import_base}.onnx.mlir"
  echo "[import] qdq (${import_mode})"
  import_onnx_to_mlir "${qdq_onnx}" "${qdq_import_base}"
fi

if [[ -z "${nqdq_mlir}" && -n "${nqdq_onnx}" ]]; then
  nqdq_import_base="${ir_dir}/${model_name}-nqdq-import"
  nqdq_mlir="${nqdq_import_base}.onnx.mlir"
  echo "[import] non-qdq (${import_mode})"
  import_onnx_to_mlir "${nqdq_onnx}" "${nqdq_import_base}"
fi

if [[ ! -f "${qdq_mlir}" ]]; then
  echo "ERROR: qdq mlir missing after import: ${qdq_mlir}"
  exit 2
fi
if [[ ! -f "${nqdq_mlir}" ]]; then
  echo "ERROR: non-qdq mlir missing after import: ${nqdq_mlir}"
  exit 2
fi

effective_qdq_mlir="${qdq_mlir}"
effective_qdq_onnx="${qdq_onnx}"
if [[ "${allow_qdq_fallback_to_nqdq}" -eq 1 && -n "${qdq_mlir}" && -n "${nqdq_mlir}" ]]; then
  if grep -Eq '"onnx.QLinearConv"|function_name = "QLinearAdd"' "${qdq_mlir}"; then
    echo "WARN: qdq MLIR contains ops not currently supported in this posit pipeline."
    echo "      Fallback enabled: use non-qdq MLIR as qdq input for build continuity."
    echo "      qdq source: ${qdq_mlir}"
    echo "      fallback : ${nqdq_mlir}"
    effective_qdq_mlir="${nqdq_mlir}"
    effective_qdq_onnx=""
  fi
fi

if [[ "${keep_stage_logs}" -eq 1 ]]; then
  stage_log_root="${out_dir}/stage_logs"
  mkdir -p "${stage_log_root}"
else
  stage_log_root="$(mktemp -d "/tmp/${model_name}.stage.XXXXXX")"
  cleanup_stage_log_dir=1
fi

align_qdomain_opt=()
if [[ "${align_to_int8_qdomain}" -eq 1 ]]; then
  align_qdomain_opt+=(--align-to-int8-qdomain)
fi
strict_qdq_opt=()
if [[ "${strict_qdq_mode}" -eq 1 ]]; then
  strict_qdq_opt+=(--strict-qdq-lowering)
fi

posit_sources=()
if [[ "${posit_source}" == "qdq" || "${posit_source}" == "both" ]]; then
  posit_sources+=("qdq:${effective_qdq_mlir}")
fi
if [[ "${posit_source}" == "nqdq" || "${posit_source}" == "both" ]]; then
  posit_sources+=("nqdq:${nqdq_mlir}")
fi

build_total_start_ns="$(date +%s%N)"
declare -A build_time_sec_by_target

for source_spec in "${posit_sources[@]}"; do
  source_label="${source_spec%%:*}"
  source_mlir="${source_spec#*:}"
  source_strict_qdq_opt=()
  source_align_qdomain_opt=()
  compact_env=(env -u ONNX_MLIR_POSIT_COMPACT_CONSTANTS -u POSIT_COMPACT_CONSTANTS \
                   -u ONNX_MLIR_POSIT_FORCE_NQDQ -u ONNX_MLIR_POSIT_FORCE_ALL_OPS \
                   -u POSIT_FORCE_NQDQ_POSIT)
  if [[ "${source_label}" == "qdq" ]]; then
    source_strict_qdq_opt=("${strict_qdq_opt[@]}")
    source_align_qdomain_opt=("${align_qdomain_opt[@]}")
  else
    # For the f32/nqdq posit path, enable compile-time compact posit constants
    # by default. This stores large f32 weights as p8/p16 raw-bit constants
    # when supported by the selected format.
    if [[ "${posit_compact_constants}" == "auto" || "${posit_compact_constants}" == "1" || "${posit_compact_constants}" == "on" || "${posit_compact_constants}" == "true" ]]; then
      compact_env=(env ONNX_MLIR_POSIT_COMPACT_CONSTANTS=1 POSIT_COMPACT_CONSTANTS=1 \
                   ONNX_MLIR_POSIT_FORCE_NQDQ=1 POSIT_FORCE_NQDQ_POSIT=1)
    fi
  fi

for fmt in "${formats[@]}"; do
  fmt_start_ns="$(date +%s%N)"
  echo "[build] ${model_name} ${source_label} posit ${fmt}"
  stage_prefix="${model_name}-${source_label}-${fmt}"
  fmt_log_dir="${stage_log_root}/${stage_prefix}"
  mkdir -p "${fmt_log_dir}"

  posit_mlir="${ir_dir}/${stage_prefix}.posit.mlir"
  krnl_mlir="${ir_dir}/${stage_prefix}.krnl.mlir"
  llvm_mlir="${ir_dir}/${stage_prefix}.llvm.mlir"
  ll="${ir_dir}/${stage_prefix}.ll"
  so="${out_dir}/${model_name}-${source_label}-${fmt}.so"
  log="${fmt_log_dir}/00.pipeline.opt.log"
  s123_log="${fmt_log_dir}/01_03.posit_pipeline.log"
  s4_log="${fmt_log_dir}/04.krnl_to_llvm.log"
  s5_log="${fmt_log_dir}/05.mlir_translate.log"
  s6_log="${fmt_log_dir}/06.link_so.log"
  s123_ir_dir="${fmt_log_dir}/01_03.pass_ir"
  s4_ir_dir="${fmt_log_dir}/04.pass_ir"
  s123_trace_opts=()
  s4_trace_opts=()

  : > "${log}"

  if [[ "${keep_stage_logs}" -eq 1 ]]; then
    rm -rf "${s123_ir_dir}" "${s4_ir_dir}"
    mkdir -p "${s123_ir_dir}" "${s4_ir_dir}"
    s123_trace_opts=(
      --mlir-print-ir-module-scope
      --mlir-print-ir-after=convert-onnx-to-posit
      --mlir-print-ir-after=convert-posit-to-krnl
      --mlir-print-ir-after=convert-onnx-to-krnl
      --mlir-print-ir-tree-dir="${s123_ir_dir}"
    )
    s4_trace_opts=(
      --mlir-print-ir-module-scope
      --mlir-print-ir-after=convert-krnl-to-affine
      --mlir-print-ir-after=convert-krnl-to-llvm
      --mlir-print-ir-after=reconcile-unrealized-casts
      --mlir-print-ir-tree-dir="${s4_ir_dir}"
    )
  fi

  if [[ "${source_label}" == "nqdq" ]]; then
    if ! "${compact_env[@]}" "${onnx_mlir_opt_bin}" "${source_mlir}" \
      --mlir-disable-threading \
      --shape-inference \
      --convert-onnx-to-posit \
      --posit-format="${fmt}" \
      "${s123_trace_opts[@]}" \
      -o "${posit_mlir}" > "${s123_log}" 2>&1; then
      echo "ERROR: posit pipeline (onnx->posit) failed for ${fmt}"
      illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${s123_log}" | head -n 1 || true)"
      if [[ -n "${illegal_op}" ]]; then
        echo "Hint: unsupported op during posit lowering: ${illegal_op}"
      fi
      cat "${s123_log}" >> "${log}" || true
      echo "Stage log: ${s123_log}"
      if [[ "${keep_stage_logs}" -eq 1 ]]; then
        echo "Stage IR dir: ${s123_ir_dir}"
      fi
      tail -n 40 "${s123_log}" || true
      if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
        echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
        continue
      fi
      exit 2
    fi
    cat "${s123_log}" >> "${log}" || true

    if ! "${onnx_mlir_opt_bin}" "${posit_mlir}" \
      --mlir-disable-threading \
      --canonicalize \
      --shape-inference \
      --convert-onnx-to-krnl \
      --convert-posit-to-krnl \
      --canonicalize \
      -o "${krnl_mlir}" >> "${s123_log}" 2>&1; then
      echo "ERROR: posit pipeline (posit/onnx->krnl) failed for ${fmt}"
      illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${s123_log}" | tail -n 1 || true)"
      if [[ -n "${illegal_op}" ]]; then
        echo "Hint: unsupported op during posit/onnx->krnl: ${illegal_op}"
      fi
      cat "${s123_log}" >> "${log}" || true
      echo "Stage log: ${s123_log}"
      if [[ "${keep_stage_logs}" -eq 1 ]]; then
        echo "Stage IR dir: ${s123_ir_dir}"
      fi
      tail -n 60 "${s123_log}" || true
      if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
        echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
        continue
      fi
      exit 2
    fi
  else
    if ! "${compact_env[@]}" "${onnx_mlir_opt_bin}" "${source_mlir}" \
      --mlir-disable-threading \
      --shape-inference \
      --convert-onnx-to-posit \
      "${source_strict_qdq_opt[@]}" \
      "${source_align_qdomain_opt[@]}" \
      --posit-format="${fmt}" \
      "${s123_trace_opts[@]}" \
      -o "${posit_mlir}" > "${s123_log}" 2>&1; then
      echo "ERROR: posit pipeline (onnx->posit) failed for ${fmt}"
      illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${s123_log}" | head -n 1 || true)"
      if [[ -n "${illegal_op}" ]]; then
        echo "Hint: unsupported op during posit lowering: ${illegal_op}"
      fi
      cat "${s123_log}" >> "${log}" || true
      echo "Stage log: ${s123_log}"
      if [[ "${keep_stage_logs}" -eq 1 ]]; then
        echo "Stage IR dir: ${s123_ir_dir}"
      fi
      tail -n 40 "${s123_log}" || true
      if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
        echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
        continue
      fi
      exit 2
    fi
    cat "${s123_log}" >> "${log}" || true

    if ! "${onnx_mlir_opt_bin}" "${posit_mlir}" \
      --mlir-disable-threading \
      --convert-posit-to-krnl \
      --canonicalize \
      --convert-onnx-to-krnl \
      --convert-posit-to-krnl \
      --canonicalize \
      -o "${krnl_mlir}" >> "${s123_log}" 2>&1; then
      echo "ERROR: posit pipeline (posit/onnx->krnl) failed for ${fmt}"
      illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${s123_log}" | tail -n 1 || true)"
      if [[ -n "${illegal_op}" ]]; then
        echo "Hint: unsupported op during posit/onnx->krnl: ${illegal_op}"
      fi
      cat "${s123_log}" >> "${log}" || true
      echo "Stage log: ${s123_log}"
      if [[ "${keep_stage_logs}" -eq 1 ]]; then
        echo "Stage IR dir: ${s123_ir_dir}"
      fi
      tail -n 60 "${s123_log}" || true
      if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
        echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
        continue
      fi
      exit 2
    fi
  fi
  cat "${s123_log}" >> "${log}" || true

  if ! "${onnx_mlir_opt_bin}" "${krnl_mlir}" \
    --mlir-disable-threading \
    --canonicalize \
    --convert-krnl-to-affine \
    --convert-krnl-to-llvm \
    --reconcile-unrealized-casts \
    "${s4_trace_opts[@]}" \
    -o "${llvm_mlir}" > "${s4_log}" 2>&1; then
    echo "ERROR: krnl->llvm lowering failed for ${fmt}"
    illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${s4_log}" | tail -n 1 || true)"
    if [[ -n "${illegal_op}" ]]; then
      echo "Hint: unsupported op during krnl->llvm: ${illegal_op}"
    fi
    cat "${s4_log}" >> "${log}" || true
    echo "Stage log: ${s4_log}"
    if [[ "${keep_stage_logs}" -eq 1 ]]; then
      echo "Stage IR dir: ${s4_ir_dir}"
    fi
    tail -n 40 "${s4_log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      continue
    fi
    exit 2
  fi
  cat "${s4_log}" >> "${log}" || true

  if ! "${mlir_translate_bin}" --mlir-to-llvmir "${llvm_mlir}" -o "${ll}" \
    > "${s5_log}" 2>&1; then
    echo "ERROR: mlir-translate failed for ${fmt}"
    cat "${s5_log}" >> "${log}" || true
    echo "Stage log: ${s5_log}"
    tail -n 40 "${s5_log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      continue
    fi
    exit 2
  fi
  cat "${s5_log}" >> "${log}" || true

  # Prefer the project-level runtime source. Avoid accidentally linking a stale
  # src/bash/posit_runtime.cpp when testing different runtime versions.
  posit_runtime_cpp="${POSIT_RUNTIME_CPP_PATH:-${src_root}/posit_runtime.cpp}"
  if [[ ! -f "${posit_runtime_cpp}" ]]; then
    if [[ -f "${script_dir}/posit_runtime.cpp" ]]; then
      echo "WARN: using fallback runtime source from script dir: ${script_dir}/posit_runtime.cpp"
      posit_runtime_cpp="${script_dir}/posit_runtime.cpp"
    else
      echo "ERROR: posit_runtime.cpp not found. Expected: ${src_root}/posit_runtime.cpp"
      echo "       Or set POSIT_RUNTIME_CPP_PATH=/path/to/posit_runtime.cpp"
      exit 2
    fi
  fi
  echo "[build] runtime source for ${fmt}: ${posit_runtime_cpp}"
  runtime_obj="${out_dir}/.${model_name}-${fmt}.posit_runtime.o"
  runtime_compile_args=(-std=c++20 -O3 -fPIC -c "${posit_runtime_cpp}" -o "${runtime_obj}")
  runtime_compile_args+=("${mlir_include_args[@]}")
  # Compile the posit runtime with hidden/default-pruned sections so format-
  # specific builds do not export or keep unrelated template instantiations.
  runtime_compile_args+=(
    -ffunction-sections
    -fdata-sections
    -fvisibility=hidden
    -fvisibility-inlines-hidden
  )
  if [[ "${runtime_format_scope}" == "single" ]]; then
    if runtime_fmt_define="$(runtime_format_define_for "${fmt}")"; then
      runtime_compile_args+=(-DPOSIT_RUNTIME_SINGLE_FORMAT=1 "-D${runtime_fmt_define}=1")
      # Some low-bit QDQ graphs still lower per-axis reference dequantization
      # through the p8e0 helper path. Keep that helper available in single-
      # scope builds so the generated .so can resolve its runtime symbols.
      if [[ "${source_label}" == "qdq" ]]; then
        case "${fmt}" in
          p4e*|p5e*|p6e*|p7e*|p9e*|p8e1|p8e2)
            runtime_compile_args+=(-DPOSIT_RUNTIME_FMT_P8E0=1)
            ;;
        esac
      fi
    else
      echo "ERROR: no runtime format define mapping for ${fmt}"
      exit 2
    fi
  fi
  if [[ "${runtime_qalign_mode}" == "alps-only" ]]; then
    runtime_compile_args+=(-DPOSIT_RUNTIME_QALIGN_ALPS_ONLY=1)
  fi
  case "${runtime_mixed_accum}" in
    off)
      runtime_compile_args+=(-DPOSIT_BUILD_MIXED_OFF=1)
      ;;
    p16e2)
      runtime_compile_args+=(-DPOSIT_BUILD_MIXED_P16E2=1)
      ;;
    p32e2)
      runtime_compile_args+=(-DPOSIT_BUILD_MIXED_P32E2=1)
      ;;
  esac
  case "${runtime_output_alps}" in
    sampled)
      runtime_compile_args+=(-DPOSIT_RUNTIME_OUTPUT_ALPS_SAMPLED=1)
      ;;
    full)
      runtime_compile_args+=(-DPOSIT_RUNTIME_OUTPUT_ALPS_FULL=1)
      ;;
    offline)
      runtime_compile_args+=(-DPOSIT_RUNTIME_OUTPUT_ALPS_OFFLINE=1)
      ;;
  esac
  runtime_compile_args+=(
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MIN=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN:-0.25}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MAX=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX:-4}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_STEPS=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS:-9}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_TARGET=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET:-1.0}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_PERCENTILE=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE:-0.95}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MIN_GAIN=${ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN:-0.0}"
    "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MAX_SAMPLES=${ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES:-4096}"
  )
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8 "${POSIT_GP_RS_VALUES_P8:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8 "${POSIT_GP_SC_VALUES_P8:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0 "${POSIT_GP_RS_VALUES_P8E0:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0 "${POSIT_GP_SC_VALUES_P8E0:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1 "${POSIT_GP_RS_VALUES_P8E1:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1 "${POSIT_GP_SC_VALUES_P8E1:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2 "${POSIT_GP_RS_VALUES_P8E2:-}"
  append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2 "${POSIT_GP_SC_VALUES_P8E2:-}"
  if [[ "${backend}" == "universal" ]]; then
    runtime_compile_args+=(-DPOSIT_USE_UNIVERSAL -I"${universal_inc}" -I/usr/local/include)
  else
    runtime_compile_args+=(
      -DPOSIT_USE_SOFTPOSIT_PX1
      -DPOSIT_USE_SOFTPOSIT_PX2
      -I"${softposit_inc}"
      -I/usr/local/include
    )
  fi
  if ! "${clangxx_bin}" "${runtime_compile_args[@]}" > "${s6_log}" 2>&1; then
    echo "ERROR: compiling posit runtime failed for ${fmt}"
    cat "${s6_log}" >> "${log}" || true
    echo "Stage log: ${s6_log}"
    tail -n 40 "${s6_log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      rm -f "${runtime_obj}"
      continue
    fi
    exit 2
  fi
  cat "${s6_log}" >> "${log}" || true

  cxx_args=(-std=c++20 -O3 -fPIC -shared "${ll}" "${runtime_obj}")
  cxx_args+=("${mlir_include_args[@]}")
  cxx_args+=(-Wl,--gc-sections)
  if [[ "${backend}" == "universal" ]]; then
    cxx_args+=(-DPOSIT_USE_UNIVERSAL -I"${universal_inc}" -I/usr/local/include)
  else
    softposit_rpath_dir="$(cd "$(dirname "${softposit_lib}")" && pwd)"
    cxx_args+=(
      -DPOSIT_USE_SOFTPOSIT_PX1
      -DPOSIT_USE_SOFTPOSIT_PX2
      -I"${softposit_inc}"
      -I/usr/local/include
      "${softposit_lib}"
      "-Wl,-rpath,${softposit_rpath_dir}"
    )
  fi
  cxx_args+=("${cruntime_link_args[@]}")
  cxx_args+=(-o "${so}")
  if ! "${clangxx_bin}" "${cxx_args[@]}" >> "${s6_log}" 2>&1; then
    echo "ERROR: linking .so failed for ${fmt}"
    cat "${s6_log}" >> "${log}" || true
    echo "Stage log: ${s6_log}"
    tail -n 40 "${s6_log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      rm -f "${runtime_obj}"
      continue
    fi
    exit 2
  fi
  cat "${s6_log}" >> "${log}" || true
  rm -f "${runtime_obj}"

  if [[ "${keep_ir}" -eq 0 ]]; then
    rm -f "${posit_mlir}" "${krnl_mlir}" "${llvm_mlir}" "${ll}"
  fi
  fmt_end_ns="$(date +%s%N)"
  fmt_sec="$(awk -v a="${fmt_start_ns}" -v b="${fmt_end_ns}" 'BEGIN{print (b-a)/1e9}')"
  build_time_sec_by_target["${model_name}-${source_label}-${fmt}.so"]="${fmt_sec}"
  echo "[build] done ${model_name}-${source_label}-${fmt}.so wall_time_sec=${fmt_sec}"
done

done

echo "[build] runner"
runner_start_ns="$(date +%s%N)"
# Prefer the project-level runner source. Allow an explicit override when testing.
runner_cpp="${RUN_TIME_CPP_PATH:-${src_root}/run_time.cpp}"
if [[ ! -f "${runner_cpp}" ]]; then
  if [[ -f "${script_dir}/run_time.cpp" ]]; then
    echo "WARN: using fallback runner source from script dir: ${script_dir}/run_time.cpp"
    runner_cpp="${script_dir}/run_time.cpp"
  else
    echo "ERROR: run_time.cpp not found. Expected: ${src_root}/run_time.cpp"
    echo "       Or set RUN_TIME_CPP_PATH=/path/to/run_time.cpp"
    exit 2
  fi
fi
echo "[build] runner source: ${runner_cpp}"
runner_args=(-std=c++20 -O3 "${runner_cpp}")
runner_args+=("${mlir_include_args[@]}")
if [[ "${backend}" == "universal" ]]; then
  runner_args+=(-DPOSIT_USE_UNIVERSAL -I"${universal_inc}" -I/usr/local/include)
else
  softposit_rpath_dir="$(cd "$(dirname "${softposit_lib}")" && pwd)"
  runner_args+=(
    -DPOSIT_USE_SOFTPOSIT_PX1
    -DPOSIT_USE_SOFTPOSIT_PX2
    -I"${softposit_inc}"
    -I/usr/local/include
    "${softposit_lib}"
    "-Wl,-rpath,${softposit_rpath_dir}"
  )
fi
runner_args+=(-ldl -o "${out_dir}/run_time_sp")
"${cxx_bin}" "${runner_args[@]}"
runner_end_ns="$(date +%s%N)"
runner_sec="$(awk -v a="${runner_start_ns}" -v b="${runner_end_ns}" 'BEGIN{print (b-a)/1e9}')"
build_time_sec_by_target["run_time_sp"]="${runner_sec}"
echo "[build] done run_time_sp wall_time_sec=${runner_sec}"

echo "[build] output-alps calibrate tool"
calib_tool_start_ns="$(date +%s%N)"
calib_tool="${out_dir}/output_alps_calibrate"
calib_args=(-std=c++20 -O3 "${posit_runtime_cpp}" -o "${calib_tool}")
calib_args+=("${mlir_include_args[@]}")
calib_args+=(-DPOSIT_RUNTIME_OUTPUT_ALPS_CALIB_TOOL=1)
calib_args+=(
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MIN=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN:-0.25}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_MAX=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX:-4}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_THETA_STEPS=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS:-9}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_TARGET=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET:-1.0}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GAMMA_PERCENTILE=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE:-0.95}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MIN_GAIN=${ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN:-0.0}"
  "-DPOSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_MAX_SAMPLES=${ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES:-4096}"
)
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8 "${POSIT_GP_RS_VALUES_P8:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8 "${POSIT_GP_SC_VALUES_P8:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E0 "${POSIT_GP_RS_VALUES_P8E0:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E0 "${POSIT_GP_SC_VALUES_P8E0:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E1 "${POSIT_GP_RS_VALUES_P8E1:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E1 "${POSIT_GP_SC_VALUES_P8E1:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_RS_VALUES_P8E2 "${POSIT_GP_RS_VALUES_P8E2:-}"
append_runtime_string_define POSIT_RUNTIME_OUTPUT_ALPS_DEFAULT_GP_SC_VALUES_P8E2 "${POSIT_GP_SC_VALUES_P8E2:-}"
if [[ "${backend}" == "universal" ]]; then
  calib_args+=(-DPOSIT_USE_UNIVERSAL -I"${universal_inc}" -I/usr/local/include)
else
  softposit_rpath_dir="$(cd "$(dirname "${softposit_lib}")" && pwd)"
  calib_args+=(
    -DPOSIT_USE_SOFTPOSIT_PX1
    -DPOSIT_USE_SOFTPOSIT_PX2
    -I"${softposit_inc}"
    -I/usr/local/include
    "${softposit_lib}"
    "-Wl,-rpath,${softposit_rpath_dir}"
  )
fi
"${cxx_bin}" "${calib_args[@]}"
calib_tool_end_ns="$(date +%s%N)"
calib_tool_sec="$(awk -v a="${calib_tool_start_ns}" -v b="${calib_tool_end_ns}" 'BEGIN{print (b-a)/1e9}')"
build_time_sec_by_target["output_alps_calibrate"]="${calib_tool_sec}"
echo "[build] done output_alps_calibrate wall_time_sec=${calib_tool_sec}"

if [[ "${skip_f32_baselines}" -eq 0 ]]; then
  echo "[build] qdq f32"
  qdq_f32_start_ns="$(date +%s%N)"
  qdq_src="${effective_qdq_mlir}"
  if [[ -n "${qdq_f32_mlir}" ]]; then
    qdq_src="${qdq_f32_mlir}"
  elif [[ -n "${effective_qdq_onnx}" ]]; then
    qdq_src="${effective_qdq_onnx}"
  fi
  "${onnx_mlir_bin}" "${qdq_src}" -O3 -L "${cruntime_lib_dir}" \
    -o "${out_dir}/${model_name}-qdq-f32"
  qdq_f32_end_ns="$(date +%s%N)"
  qdq_f32_sec="$(awk -v a="${qdq_f32_start_ns}" -v b="${qdq_f32_end_ns}" 'BEGIN{print (b-a)/1e9}')"
  build_time_sec_by_target["${model_name}-qdq-f32.so"]="${qdq_f32_sec}"
  echo "[build] done ${model_name}-qdq-f32.so wall_time_sec=${qdq_f32_sec}"

  echo "[build] non-qdq f32"
  nqdq_f32_start_ns="$(date +%s%N)"
  nqdq_src="${nqdq_mlir}"
  if [[ -n "${nqdq_onnx}" ]]; then
    nqdq_src="${nqdq_onnx}"
  fi
  "${onnx_mlir_bin}" "${nqdq_src}" -O3 -L "${cruntime_lib_dir}" \
    -o "${out_dir}/${model_name}-nqdq-f32"
  nqdq_f32_end_ns="$(date +%s%N)"
  nqdq_f32_sec="$(awk -v a="${nqdq_f32_start_ns}" -v b="${nqdq_f32_end_ns}" 'BEGIN{print (b-a)/1e9}')"
  build_time_sec_by_target["${model_name}-nqdq-f32.so"]="${nqdq_f32_sec}"
  echo "[build] done ${model_name}-nqdq-f32.so wall_time_sec=${nqdq_f32_sec}"
fi

so_count="$(find "${out_dir}" -maxdepth 1 -type f -name '*.so' | wc -l | tr -d '[:space:]')"
expected_so_count=$(( ${#formats[@]} * ${#posit_sources[@]} ))
if [[ "${skip_f32_baselines}" -eq 0 ]]; then
  expected_so_count=$(( expected_so_count + 2 ))
fi
echo
echo "Built .so files (${so_count}):"
find "${out_dir}" -maxdepth 1 -type f -name '*.so' | sort
echo "Runner:"
echo "  ${out_dir}/run_time_sp"
echo "Calibration tool:"
echo "  ${out_dir}/output_alps_calibrate"
if [[ "${keep_stage_logs}" -eq 1 ]]; then
  echo "Stage logs:"
  echo "  ${out_dir}/stage_logs"
else
  echo "Stage logs:"
  echo "  not kept by default (use --keep-stage-logs to enable)"
fi
if [[ "${keep_ir}" -eq 1 ]]; then
  echo "IR dir:"
  echo "  ${ir_dir}"
else
  echo "IR dir:"
  echo "  not kept (temporary build-only files)"
fi
if [[ "${so_count}" != "${expected_so_count}" ]]; then
  echo "WARNING: expected ${expected_so_count} .so files, got ${so_count}"
fi

build_total_end_ns="$(date +%s%N)"
build_total_sec="$(awk -v a="${build_total_start_ns}" -v b="${build_total_end_ns}" 'BEGIN{print (b-a)/1e9}')"
build_posit_config="${out_dir}/build_posit_config.log"
{
  echo "# build_posit_config"
  echo "model_name=${model_name}"
  echo "posit_source=${posit_source}"
  echo "backend=${backend}"
  echo "posit_formats=${posit_formats_csv}"
  echo "import_mode=${import_mode}"
  echo "strict_qdq_mode=${strict_qdq_mode}"
  echo "align_to_int8_qdomain=${align_to_int8_qdomain}"
  echo "posit_compact_constants=${posit_compact_constants}"
  echo "runtime_format_scope=${runtime_format_scope}"
  echo "runtime_qalign_mode=${runtime_qalign_mode}"
  echo "runtime_mixed_accum=${runtime_mixed_accum}"
  echo "runtime_output_alps=${runtime_output_alps}"
  echo "const_alps_enabled=${ONNX_MLIR_POSIT_CONST_ALPS:-${POSIT_CONST_ALPS:-0}}"
  echo "const_alps_jobs=${ONNX_MLIR_POSIT_CONST_ALPS_JOBS:-${POSIT_CONST_ALPS_JOBS:-<unset>}}"
  echo "const_alps_theta_min=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN:-0.25}"
  echo "const_alps_theta_max=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX:-4}"
  echo "const_alps_theta_steps=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS:-9}"
  echo "const_alps_gamma_target=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET:-1.0}"
  echo "const_alps_gamma_percentile=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE:-0.95}"
  echo "const_alps_min_gain=${ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN:-0.0}"
  echo "const_alps_max_samples=${ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES:-4096}"
  echo "gp_experimental_formats=${POSIT_GP_EXPERIMENTAL_FORMATS:-<unset>}"
  echo "gp_rs_values_p8=${POSIT_GP_RS_VALUES_P8:-<unset>}"
  echo "gp_sc_values_p8=${POSIT_GP_SC_VALUES_P8:-<unset>}"
  echo "gp_rs_values_p8e0=${POSIT_GP_RS_VALUES_P8E0:-<unset>}"
  echo "gp_sc_values_p8e0=${POSIT_GP_SC_VALUES_P8E0:-<unset>}"
  echo "gp_rs_values_p8e1=${POSIT_GP_RS_VALUES_P8E1:-<unset>}"
  echo "gp_sc_values_p8e1=${POSIT_GP_SC_VALUES_P8E1:-<unset>}"
  echo "gp_rs_values_p8e2=${POSIT_GP_RS_VALUES_P8E2:-<unset>}"
  echo "gp_sc_values_p8e2=${POSIT_GP_SC_VALUES_P8E2:-<unset>}"
  echo "runtime_output_alps_default_theta_min=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN:-0.25}"
  echo "runtime_output_alps_default_theta_max=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX:-4}"
  echo "runtime_output_alps_default_theta_steps=${ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS:-9}"
  echo "runtime_output_alps_default_gamma_target=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET:-1.0}"
  echo "runtime_output_alps_default_gamma_percentile=${ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE:-0.95}"
  echo "runtime_output_alps_default_min_gain=${ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN:-0.0}"
  echo "runtime_output_alps_default_max_samples=${ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES:-4096}"
  echo "runtime_output_alps_default_gp_rs_values_p8=${POSIT_GP_RS_VALUES_P8:-<unset>}"
  echo "runtime_output_alps_default_gp_sc_values_p8=${POSIT_GP_SC_VALUES_P8:-<unset>}"
  echo "runtime_output_alps_default_gp_rs_values_p8e0=${POSIT_GP_RS_VALUES_P8E0:-<unset>}"
  echo "runtime_output_alps_default_gp_sc_values_p8e0=${POSIT_GP_SC_VALUES_P8E0:-<unset>}"
  echo "runtime_output_alps_default_gp_rs_values_p8e1=${POSIT_GP_RS_VALUES_P8E1:-<unset>}"
  echo "runtime_output_alps_default_gp_sc_values_p8e1=${POSIT_GP_SC_VALUES_P8E1:-<unset>}"
  echo "runtime_output_alps_default_gp_rs_values_p8e2=${POSIT_GP_RS_VALUES_P8E2:-<unset>}"
  echo "runtime_output_alps_default_gp_sc_values_p8e2=${POSIT_GP_SC_VALUES_P8E2:-<unset>}"
  echo "runtime_output_alps_defaults_from_build_env=1"
  echo "weight_vs_activation_same_search_space=1"
  echo "weight_vs_activation_same_final_params=0"
} > "${build_posit_config}"
build_time_summary="${out_dir}/build_time_summary.log"
{
  echo "Build time summary"
  for target in $(printf '%s\n' "${!build_time_sec_by_target[@]}" | sort); do
    echo "  ${target}: ${build_time_sec_by_target[$target]} sec"
  done
  echo "  total_wall_time_sec=${build_total_sec}"
  echo "  build_posit_config=${build_posit_config}"
} | tee "${build_time_summary}"
