#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
project_root="$(cd "${src_root}/.." && pwd)"

in_csv="${src_root}/temp/mobilenet_qdomain/qdq_probe_input_conv0.csv"
out_dir="${src_root}/temp/mobilenet_alpha_align_probe"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --in-csv)
    in_csv="$2"
    shift 2
    ;;
  --out-dir)
    out_dir="$2"
    shift 2
    ;;
  -h|--help)
    cat <<'EOF'
Usage:
  probe_qdq_alpha_align.sh [--in-csv PATH] [--out-dir DIR]

Default input:
  src/temp/mobilenet_qdomain/qdq_probe_input_conv0.csv

Outputs:
  <out-dir>/qdq_probe_input_conv0_alpha_align.csv
  <out-dir>/qdq_probe_input_conv0_alpha_align.txt
EOF
    exit 0
    ;;
  *)
    echo "Unknown arg: $1"
    exit 2
    ;;
  esac
done

mkdir -p "${out_dir}"
bin="${out_dir}/qdq_alpha_align_probe"
cpp="${src_root}/temp/probe/qdq_alpha_align_probe.cpp"
uinc="${src_root}/.deps/universal/include/sw"

if [[ ! -f "${cpp}" ]]; then
  echo "Missing source: ${cpp}"
  exit 1
fi
if [[ ! -d "${uinc}" ]]; then
  echo "Missing universal include dir: ${uinc}"
  exit 1
fi
if [[ ! -f "${in_csv}" ]]; then
  echo "Missing input CSV: ${in_csv}"
  exit 1
fi

g++ -std=c++20 -O2 -I"${uinc}" "${cpp}" -o "${bin}"

out_csv="${out_dir}/qdq_probe_input_conv0_alpha_align.csv"
out_txt="${out_dir}/qdq_probe_input_conv0_alpha_align.txt"
"${bin}" "${in_csv}" "${out_csv}" "${out_txt}"

echo "Wrote:"
echo "  ${out_csv}"
echo "  ${out_txt}"
