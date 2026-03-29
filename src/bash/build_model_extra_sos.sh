#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  build_model_extra_sos.sh [all args from build_model11_sos.sh]
    (defaults to universal backend + extra posit formats + no f32 baselines)

Default extra formats (exclude original 11 set):
  p4e0,p4e1,p4e2,p4e3,
  p5e0,p5e1,p5e2,p5e3,
  p6e0,p6e1,p6e2,p6e3,
  p7e0,p7e1,p7e2,p7e3,
  p9e0,p9e1,p9e2,p9e3

Examples:
  build_model_extra_sos.sh --model-name mobilenetv2-12 \
    --qdq-mlir ./mobilenetv2-12-qdq.onnx.mlir \
    --nqdq-mlir ./mobilenetv2-12.onnx.mlir \
    --shape-info 0:1x3x224x224 --out-dir ./temp/mobilenet-extra
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_formats="p4e0,p4e1,p4e2,p4e3,p5e0,p5e1,p5e2,p5e3,p6e0,p6e1,p6e2,p6e3,p7e0,p7e1,p7e2,p7e3,p9e0,p9e1,p9e2,p9e3"

has_flag=0
has_backend=0
has_formats=0

for ((i = 1; i <= $#; ++i)); do
  a="${!i}"
  if [[ "${a}" == "-h" || "${a}" == "--help" ]]; then
    usage
    exit 0
  fi
  if [[ "${a}" == "--skip-f32-baselines" ]]; then
    has_flag=1
  elif [[ "${a}" == "--backend" ]]; then
    has_backend=1
    ((i += 1))
  elif [[ "${a}" == "--posit-formats" ]]; then
    has_formats=1
    ((i += 1))
  fi
done

forward_args=("$@")
if [[ "${has_backend}" -eq 0 ]]; then
  forward_args+=(--backend universal)
fi
if [[ "${has_formats}" -eq 0 ]]; then
  forward_args+=(--posit-formats "${default_formats}")
fi
if [[ "${has_flag}" -eq 0 ]]; then
  forward_args+=(--skip-f32-baselines)
fi

exec "${script_dir}/build_model11_sos.sh" "${forward_args[@]}"

