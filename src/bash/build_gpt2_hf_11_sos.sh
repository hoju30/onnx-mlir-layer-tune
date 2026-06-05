#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${src_root}/.." && pwd)"
workspace_root="$(cd "${project_root}/.." && pwd)"
gpt2_root="${workspace_root}/ImageNet100/model/gpt2_onnx_community/onnx"

out_dir="${workspace_root}/ImageNet100/build_gpt2_hf_posit11"
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

qdq_onnx="${gpt2_root}/model_int8.onnx"
nqdq_onnx="${gpt2_root}/model.onnx"

if [[ ! -f "${qdq_onnx}" ]]; then
  echo "ERROR: cannot find GPT-2 QDQ ONNX: ${qdq_onnx}"
  exit 2
fi
if [[ ! -f "${nqdq_onnx}" ]]; then
  echo "ERROR: cannot find GPT-2 non-QDQ ONNX: ${nqdq_onnx}"
  exit 2
fi

"${script_dir}/build_model11_sos.sh" \
  --model-name "gpt2-hf-debug" \
  --qdq-onnx "${qdq_onnx}" \
  --nqdq-onnx "${nqdq_onnx}" \
  --backend universal \
  --out-dir "${out_dir}" \
  "${extra_args[@]}"
