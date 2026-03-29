#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${src_root}/.." && pwd)"

out_dir="${project_root}/build/resnet50-extra"
extra_args=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == --* ]]; then
    extra_args=("$@")
  else
    out_dir="$1"
    shift
    extra_args=("$@")
  fi
fi

qdq_onnx="${project_root}/datasets/models/onnx/resnet50-v1-12-qdq.onnx"
nqdq_onnx="${project_root}/datasets/models/onnx/resnet50-v1-12.onnx"
qdq_mlir="${src_root}/resnet50-v1-12-qdq.onnx.mlir"
nqdq_mlir="${src_root}/resnet50-v1-12.onnx.mlir"

source_args=()
if [[ -f "${qdq_onnx}" && -f "${nqdq_onnx}" ]]; then
  source_args=(
    --qdq-onnx "${qdq_onnx}"
    --nqdq-onnx "${nqdq_onnx}"
  )
else
  source_args=(
    --qdq-mlir "${qdq_mlir}"
    --nqdq-mlir "${nqdq_mlir}"
  )
fi

"${script_dir}/build_model_extra_sos.sh" \
  --model-name "resnet50-v1-12" \
  "${source_args[@]}" \
  --shape-info "0:1x3x224x224" \
  --out-dir "${out_dir}" \
  "${extra_args[@]}"

