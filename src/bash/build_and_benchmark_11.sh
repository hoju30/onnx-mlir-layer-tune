#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  build_and_benchmark_11.sh --model-name NAME \
    [--qdq-mlir PATH | --qdq-onnx PATH] \
    [--nqdq-mlir PATH | --nqdq-onnx PATH] \
    [--shape-info STR] [--out-dir PATH] \
    [--input-shape NxCxHxW] [--input-txt PATH | --zeros | --random [seed]] \
    [--warmup N] [--iters N] [--label CLASS] [--timeout-sec N] [--continue-on-posit-fail] \
    [--strict-qdq-mode] [--non-strict-qdq-mode] \
    [--align-to-int8-qdomain] [--no-align-to-int8-qdomain]

Output:
  1) Build 11 shared libraries via build_model11_sos.sh
  2) Run automatic comparison and write:
     <out-dir>/benchmark_report.txt
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
shape_info=""
input_shape="1x3x224x224"

input_mode="random"
input_txt=""
random_seed="12345"

warmup="20"
iters="200"
label=""
timeout_sec="0"
continue_on_posit_fail=0

extra_build_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      model_name="$2"; shift 2;;
    --qdq-mlir)
      qdq_mlir="$2"; shift 2;;
    --qdq-onnx)
      qdq_onnx="$2"; shift 2;;
    --nqdq-mlir)
      nqdq_mlir="$2"; shift 2;;
    --nqdq-onnx)
      nqdq_onnx="$2"; shift 2;;
    --shape-info)
      shape_info="$2"; shift 2;;
    --out-dir)
      out_dir="$2"; shift 2;;
    --input-shape)
      input_shape="$2"; shift 2;;
    --input-txt)
      input_mode="txt"; input_txt="$2"; shift 2;;
    --zeros)
      input_mode="zeros"; shift;;
    --random)
      input_mode="random"
      if [[ $# -ge 2 && "$2" != --* ]]; then
        random_seed="$2"; shift 2
      else
        shift
      fi
      ;;
    --warmup)
      warmup="$2"; shift 2;;
    --iters)
      iters="$2"; shift 2;;
    --label)
      label="$2"; shift 2;;
    --timeout-sec)
      timeout_sec="$2"; shift 2;;
    --continue-on-posit-fail)
      continue_on_posit_fail=1; shift;;
    --universal-include-dir|--cruntime-lib-dir|--onnx-mlir|--onnx-mlir-opt|--mlir-translate|--use-onnx-ir|--align-to-int8-qdomain|--no-align-to-int8-qdomain|--strict-qdq-mode|--non-strict-qdq-mode)
      if [[ "$1" == "--use-onnx-ir" ]]; then
        extra_build_args+=("$1")
        shift
      elif [[ "$1" == "--align-to-int8-qdomain" || "$1" == "--no-align-to-int8-qdomain" ||
              "$1" == "--strict-qdq-mode" || "$1" == "--non-strict-qdq-mode" ]]; then
        extra_build_args+=("$1")
        shift
      else
        extra_build_args+=("$1" "$2")
        shift 2
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${model_name}" ]]; then
  echo "ERROR: --model-name is required"
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

build_cmd=("${script_dir}/build_model11_sos.sh" "--model-name" "${model_name}" "--out-dir" "${out_dir}")

if [[ -n "${qdq_mlir}" ]]; then
  build_cmd+=("--qdq-mlir" "${qdq_mlir}")
else
  build_cmd+=("--qdq-onnx" "${qdq_onnx}")
fi

if [[ -n "${nqdq_mlir}" ]]; then
  build_cmd+=("--nqdq-mlir" "${nqdq_mlir}")
else
  build_cmd+=("--nqdq-onnx" "${nqdq_onnx}")
fi

if [[ -n "${shape_info}" ]]; then
  build_cmd+=("--shape-info" "${shape_info}")
fi

if [[ "${continue_on_posit_fail}" -eq 1 ]]; then
  build_cmd+=("--continue-on-posit-fail")
fi

if [[ ${#extra_build_args[@]} -gt 0 ]]; then
  build_cmd+=("${extra_build_args[@]}")
fi

echo "[1/2] build 11 shared libraries"
"${build_cmd[@]}"

runner="${out_dir}/run_time_sp"
if [[ ! -x "${runner}" ]]; then
  echo "ERROR: runner not found: ${runner}"
  exit 3
fi

qdq_f32_so="${out_dir}/${model_name}-qdq-f32.so"
nqdq_f32_so="${out_dir}/${model_name}-nqdq-f32.so"

for f in "${qdq_f32_so}" "${nqdq_f32_so}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: required baseline so missing: ${f}"
    exit 3
  fi
done

detect_entry_symbol() {
  local so="$1"
  local sym
  sym="$(nm -D "${so}" 2>/dev/null | awk '/_mlir_ciface_main_graph/ {print $3; exit}')"
  if [[ -z "${sym}" ]]; then
    return 1
  fi
  printf '%s' "${sym}"
}

qdq_entry="$(detect_entry_symbol "${qdq_f32_so}")" || {
  echo "ERROR: cannot detect main entry symbol in ${qdq_f32_so}"
  exit 3
}
nqdq_entry="$(detect_entry_symbol "${nqdq_f32_so}")" || {
  echo "ERROR: cannot detect main entry symbol in ${nqdq_f32_so}"
  exit 3
}

report="${out_dir}/benchmark_report.txt"
: > "${report}"

input_args=("--shape" "${input_shape}")
case "${input_mode}" in
  txt)
    if [[ ! -f "${input_txt}" ]]; then
      echo "ERROR: input txt not found: ${input_txt}"
      exit 4
    fi
    input_args+=("${input_txt}")
    ;;
  zeros)
    input_args+=("--zeros")
    ;;
  random)
    input_args+=("--random" "${random_seed}")
    ;;
esac

common_args=("--warmup" "${warmup}" "--iters" "${iters}" "--stats" "--quiet")
if [[ -n "${label}" ]]; then
  common_args+=("--label" "${label}")
fi
runner_prefix=()
if [[ "${timeout_sec}" != "0" ]]; then
  runner_prefix=("timeout" "${timeout_sec}")
fi

formats=(p8e0 p8e1 p8e2 p16e0 p16e1 p16e2 p32e0 p32e1 p32e2)

echo "[2/2] run benchmark and write report: ${report}"
{
  echo "model=${model_name}"
  echo "out_dir=${out_dir}"
  echo "input_shape=${input_shape}"
  echo "warmup=${warmup} iters=${iters}"
  if [[ -n "${label}" ]]; then
    echo "label=${label}"
  fi
  echo

  echo "=== Baseline: qdq-f32 vs nqdq-f32 ==="
  if ! "${runner_prefix[@]}" "${runner}" "${qdq_f32_so}" \
    "${input_args[@]}" \
    "${common_args[@]}" \
    --entry "${qdq_entry}" \
    --out-type f32 \
    --cmp "${nqdq_f32_so}:f32:${nqdq_entry}" \
    --baseline main; then
    echo "WARN: baseline run failed (qdq-f32)"
  fi
  echo

  for fmt in "${formats[@]}"; do
    so="${out_dir}/${model_name}-qdq-${fmt}.so"
    if [[ ! -f "${so}" ]]; then
      echo "=== ${fmt} ==="
      echo "SKIP: missing ${so}"
      echo
      continue
    fi

    echo "=== ${fmt} (baseline: qdq-f32) ==="
    pos_entry="$(detect_entry_symbol "${so}")" || {
      echo "SKIP: cannot detect main entry symbol in ${so}"
      echo
      continue
    }
    if ! "${runner_prefix[@]}" "${runner}" "${so}" \
      "${input_args[@]}" \
      "${common_args[@]}" \
      --entry "${pos_entry}" \
      --out-type f32 \
      --cmp "${qdq_f32_so}:f32:${qdq_entry}" \
      --cmp "${nqdq_f32_so}:f32:${nqdq_entry}" \
      --baseline cmp:1; then
      echo "WARN: benchmark failed for ${fmt}"
    fi
    echo
  done
} | tee -a "${report}"

echo "Done. Report: ${report}"
