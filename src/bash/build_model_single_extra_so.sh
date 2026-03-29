#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  build_model_single_extra_so.sh --format pNeM [args from build_model11_sos.sh]

Allowed extra formats:
  p4e0..p4e3, p5e0..p5e3, p6e0..p6e3, p7e0..p7e3, p9e0..p9e3

Notes:
  - This script builds exactly one extra-format qdq .so.
  - It defaults to universal backend and skips qdq-f32 / nqdq-f32 baselines.
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
format=""
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
  --format)
    [[ $# -ge 2 ]] || { echo "ERROR: --format needs a value"; exit 2; }
    format="$2"
    shift 2
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    forward_args+=("$1")
    shift
    ;;
  esac
done

if [[ -z "${format}" ]]; then
  echo "ERROR: missing --format"
  usage
  exit 2
fi

if [[ ! "${format}" =~ ^p(4|5|6|7|9)e[0-3]$ ]]; then
  echo "ERROR: unsupported extra format: ${format}"
  echo "       expected one of p4/p5/p6/p7/p9 with e0..e3"
  exit 2
fi

has_skip=0
has_backend=0
has_formats=0
for ((i = 0; i < ${#forward_args[@]}; ++i)); do
  a="${forward_args[$i]}"
  if [[ "${a}" == "--skip-f32-baselines" ]]; then
    has_skip=1
  elif [[ "${a}" == "--backend" ]]; then
    has_backend=1
    ((i += 1))
  elif [[ "${a}" == "--posit-formats" ]]; then
    has_formats=1
    ((i += 1))
  fi
done

if [[ "${has_backend}" -eq 0 ]]; then
  forward_args+=(--backend universal)
fi
if [[ "${has_formats}" -eq 0 ]]; then
  forward_args+=(--posit-formats "${format}")
fi
if [[ "${has_skip}" -eq 0 ]]; then
  forward_args+=(--skip-f32-baselines)
fi

exec "${script_dir}/build_model11_sos.sh" "${forward_args[@]}"

