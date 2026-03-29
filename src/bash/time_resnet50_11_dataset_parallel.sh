#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  time_resnet50_11_dataset_parallel.sh [txt_dir] [options for time_model11_dataset_parallel.sh]

Defaults:
  txt_dir = ./temp/imagenette_val_224
  out_dir = ./temp
  shape   = 1x3x224x224

Examples:
  time_resnet50_11_dataset_parallel.sh --jobs 4 --limit 20
  time_resnet50_11_dataset_parallel.sh ./temp/imagenette_val_224 --out-dir /tmp/resnet50-11-test3 --jobs 6
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
default_txt_dir="${src_root}/temp/imagenette_val_224"
default_out_dir="${src_root}/temp"

txt_dir="${default_txt_dir}"
out_dir="${default_out_dir}"
extra_args=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      extra_args=("$@")
      ;;
    *)
      txt_dir="$1"
      shift
      extra_args=("$@")
      ;;
  esac
fi

for ((i = 0; i < ${#extra_args[@]}; ++i)); do
  if [[ "${extra_args[$i]}" == "--txt-dir" && $((i + 1)) -lt ${#extra_args[@]} ]]; then
    txt_dir="${extra_args[$((i + 1))]}"
  fi
  if [[ "${extra_args[$i]}" == "--out-dir" && $((i + 1)) -lt ${#extra_args[@]} ]]; then
    out_dir="${extra_args[$((i + 1))]}"
  fi
done

"${script_dir}/time_model11_dataset_parallel.sh" \
  --model-name resnet50-v1-12 \
  --out-dir "${out_dir}" \
  --txt-dir "${txt_dir}" \
  --shape "1x3x224x224" \
  "${extra_args[@]}"
