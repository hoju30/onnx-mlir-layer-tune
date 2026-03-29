#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  build_model11_sos.sh --model-name NAME \
    [--qdq-mlir PATH | --qdq-onnx PATH] \
    [--nqdq-mlir PATH | --nqdq-onnx PATH] \
    [--shape-info STR] [--use-onnx-ir] \
    [--continue-on-posit-fail] \
    [--skip-f32-baselines] \
    [--keep-ir] [--ir-dir PATH] \
    [--out-dir PATH] [--cruntime-lib-dir PATH] \
    [--backend softposit|universal] [--posit-formats CSV] \
    [--softposit-include-dir PATH] [--softposit-lib PATH] \
    [--universal-include-dir PATH]

Output:
  <out-dir>/<NAME>-qdq-<posit-format>.so   (posit variants)
  <out-dir>/<NAME>-qdq-f32.so
  <out-dir>/<NAME>-nqdq-f32.so
  <out-dir>/run_time_sp
  (IR files are NOT kept by default; use --keep-ir to preserve.)
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${src_root}/.." && pwd)"

model_name=""
qdq_mlir=""
qdq_onnx=""
nqdq_mlir=""
nqdq_onnx=""
out_dir=""
universal_inc="${UNIVERSAL_INCLUDE_DIR:-/home/lai/mlir_toy/universal/universal/include/sw}"
softposit_inc="${SOFTPOSIT_INCLUDE_DIR:-/home/lai/mlir_toy/SoftPosit/SoftPosit/source/include}"
softposit_lib="${SOFTPOSIT_LIB_PATH:-}"
backend="${POSIT_BACKEND:-universal}"
posit_formats_csv="${POSIT_FORMATS:-p8e0,p8e1,p8e2,p16e0,p16e1,p16e2,p32e0,p32e1,p32e2}"
cruntime_lib_dir="${CRUNTIME_LIB_DIR:-}"
shape_info=""
import_mode="basic"
continue_on_posit_fail=0
skip_f32_baselines=0
keep_ir=0
ir_dir_override=""
onnx_mlir_bin="${ONNX_MLIR_BIN:-}"
onnx_mlir_opt_bin="${ONNX_MLIR_OPT_BIN:-}"
mlir_translate_bin="${MLIR_TRANSLATE_BIN:-}"

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
  --skip-f32-baselines)
    skip_f32_baselines=1
    shift
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

if [[ -z "${cruntime_lib_dir}" ]]; then
  for cand in \
    "/home/lai/onnx-mlir/build/Release/lib" \
    "/home/lai/mlir_toy/onnx-mlir/build/Release/lib" \
    "${project_root}/build/Release/lib" \
    "${src_root}/build/Release/lib"
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

if [[ "${backend}" != "softposit" && "${backend}" != "universal" ]]; then
  echo "ERROR: --backend must be softposit or universal"
  exit 2
fi

if [[ "${backend}" == "universal" ]]; then
  if [[ ! -f "${universal_inc}/universal/number/posit/posit_fwd.hpp" ]]; then
    echo "ERROR: universal headers not found under: ${universal_inc}"
    exit 2
  fi
else
  if [[ ! -f "${softposit_inc}/softposit.h" ]]; then
    echo "ERROR: softposit header not found: ${softposit_inc}/softposit.h"
    exit 2
  fi
  if [[ -z "${softposit_lib}" ]]; then
    for cand in \
      "${src_root}/.deps/softposit-px1/libsoftposit.so" \
      "${src_root}/.deps/softposit-px1/libsoftposit_full.so" \
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

has_softposit_unsupported_format=0
for fmt in "${formats[@]}"; do
  if [[ "${fmt}" == "p16e0" || "${fmt}" == "p32e0" ]]; then
    has_softposit_unsupported_format=1
    break
  fi
done

echo "[config] backend=${backend}"
echo "[config] formats=${formats[*]}"
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

ir_dir="${out_dir}/ir"
cleanup_ir_dir=0
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

for fmt in "${formats[@]}"; do
  echo "[build] ${model_name} posit ${fmt}"
  krnl_mlir="${ir_dir}/${model_name}-qdq-${fmt}.krnl.mlir"
  llvm_mlir="${ir_dir}/${model_name}-qdq-${fmt}.llvm.mlir"
  ll="${ir_dir}/${model_name}-qdq-${fmt}.ll"
  so="${out_dir}/${model_name}-qdq-${fmt}.so"
  log="${ir_dir}/${model_name}-qdq-${fmt}.opt.log"

  if ! "${onnx_mlir_opt_bin}" "${qdq_mlir}" \
    --shape-inference \
    --convert-onnx-to-posit \
    --convert-posit-to-krnl \
    --canonicalize \
    --convert-onnx-to-krnl \
    --convert-posit-to-krnl \
    --canonicalize \
    --posit-format="${fmt}" \
    -o "${krnl_mlir}" > "${log}" 2>&1; then
    echo "ERROR: posit->krnl lowering failed for ${fmt}"
    illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${log}" | head -n 1 || true)"
    if [[ -n "${illegal_op}" ]]; then
      echo "Hint: unsupported op during posit lowering: ${illegal_op}"
    fi
    if rg -q "failed to legalize operation 'func.func'" "${log}"; then
      echo "Hint: this IR may not be supported by convert-onnx-to-posit yet."
      echo "      Try --qdq-onnx with default import mode (ONNXBasic)."
    fi
    tail -n 40 "${log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      continue
    fi
    exit 2
  fi

  if ! "${onnx_mlir_opt_bin}" "${krnl_mlir}" \
    --canonicalize \
    --convert-krnl-to-affine \
    --convert-krnl-to-llvm \
    --reconcile-unrealized-casts \
    -o "${llvm_mlir}" >> "${log}" 2>&1; then
    echo "ERROR: krnl->llvm lowering failed for ${fmt}"
    illegal_op="$(sed -n "s/.*failed to legalize operation '\\([^']*\\)'.*/\\1/p" "${log}" | tail -n 1 || true)"
    if [[ -n "${illegal_op}" ]]; then
      echo "Hint: unsupported op during krnl->llvm: ${illegal_op}"
    fi
    tail -n 40 "${log}" || true
    if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
      echo "WARN: skip posit ${fmt} and continue due to --continue-on-posit-fail"
      continue
    fi
    exit 2
  fi

  "${mlir_translate_bin}" --mlir-to-llvmir "${llvm_mlir}" -o "${ll}"

  posit_runtime_cpp="${script_dir}/posit_runtime.cpp"
  if [[ ! -f "${posit_runtime_cpp}" ]]; then
    posit_runtime_cpp="${src_root}/posit_runtime.cpp"
  fi
  cxx_args=(-std=c++20 -O3 -fPIC -shared "${ll}" "${posit_runtime_cpp}")
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
  cxx_args+=(-o "${so}")
  clang++ "${cxx_args[@]}"

  if [[ "${keep_ir}" -eq 0 ]]; then
    rm -f "${krnl_mlir}" "${llvm_mlir}" "${ll}" "${log}"
  fi
done

echo "[build] runner"
runner_cpp="${script_dir}/run_time.cpp"
if [[ ! -f "${runner_cpp}" && -f "${src_root}/run_time.cpp" ]]; then
  runner_cpp="${src_root}/run_time.cpp"
fi
runner_args=(-std=c++20 -O3 "${runner_cpp}")
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
g++ "${runner_args[@]}"

if [[ "${skip_f32_baselines}" -eq 0 ]]; then
  echo "[build] qdq f32"
  qdq_src="${qdq_mlir}"
  if [[ -n "${qdq_onnx}" ]]; then
    qdq_src="${qdq_onnx}"
  fi
  "${onnx_mlir_bin}" "${qdq_src}" -O3 -L "${cruntime_lib_dir}" \
    -o "${out_dir}/${model_name}-qdq-f32"

  echo "[build] non-qdq f32"
  nqdq_src="${nqdq_mlir}"
  if [[ -n "${nqdq_onnx}" ]]; then
    nqdq_src="${nqdq_onnx}"
  fi
  "${onnx_mlir_bin}" "${nqdq_src}" -O3 -L "${cruntime_lib_dir}" \
    -o "${out_dir}/${model_name}-nqdq-f32"
fi

so_count="$(find "${out_dir}" -maxdepth 1 -type f -name '*.so' | wc -l | tr -d '[:space:]')"
expected_so_count="${#formats[@]}"
if [[ "${skip_f32_baselines}" -eq 0 ]]; then
  expected_so_count=$(( expected_so_count + 2 ))
fi
echo
echo "Built .so files (${so_count}):"
find "${out_dir}" -maxdepth 1 -type f -name '*.so' | sort
echo "Runner:"
echo "  ${out_dir}/run_time_sp"
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
