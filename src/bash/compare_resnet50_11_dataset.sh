#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  compare_resnet50_11_dataset.sh [txt_dir] [options for compare_model11_dataset.sh]

Defaults (if no args):
  txt_dir = ./temp/imagenette_val_224
  out_dir = ./temp
  shape   = 1x3x224x224

Examples:
  compare_resnet50_11_dataset.sh
  compare_resnet50_11_dataset.sh ./temp/imagenette_val_224 --limit 100 --iters 1
  compare_resnet50_11_dataset.sh --out-dir /tmp/resnet50-11 --txt-dir ./temp/imagenette_val_224
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

"${script_dir}/compare_model11_dataset.sh" \
  --model-name resnet50-v1-12 \
  --out-dir "${out_dir}" \
  --txt-dir "${txt_dir}" \
  --shape "1x3x224x224" \
  "${extra_args[@]}"
