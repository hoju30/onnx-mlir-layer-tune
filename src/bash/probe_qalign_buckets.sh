#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  probe_qalign_buckets.sh [--format p8e0|p8e1|p8e2] [--mode off|alps|any]
                          [--bucket-limit N] [--samples-per-bucket N]
                          [--collect-csv PATH] [--qalign-csv PATH]
                          [--detail-csv PATH] [--out-prefix PATH]

Default paths target the current mobilenet11_dualref workspace.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
probe_src="${src_root}/temp/probe/qalign_bucket_probe.cpp"
uinc="${src_root}/.deps/universal/include/sw"

format="p8e0"
mode="off"
bucket_limit=3
samples_per_bucket=12
collect_csv="${src_root}/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/collect_upto_75.csv"
qalign_csv="${src_root}/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/calib_upto_75/qalign_p8e0.csv"
detail_csv="${src_root}/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/calib_upto_75/qalign_p8e0_detail.csv"
out_prefix="${src_root}/temp/mobilenet11_dualref/qalign_auto/mobilenetv2-12/p8e0/probe_first_off_buckets"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      format="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --bucket-limit)
      bucket_limit="$2"
      shift 2
      ;;
    --samples-per-bucket)
      samples_per_bucket="$2"
      shift 2
      ;;
    --collect-csv)
      collect_csv="$2"
      shift 2
      ;;
    --qalign-csv)
      qalign_csv="$2"
      shift 2
      ;;
    --detail-csv)
      detail_csv="$2"
      shift 2
      ;;
    --out-prefix)
      out_prefix="$2"
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

mkdir -p "$(dirname "${out_prefix}")"
bin="${out_prefix}.bin"

g++ -std=c++20 -O2 -I"${uinc}" "${probe_src}" -o "${bin}"
"${bin}" "${format}" "${collect_csv}" "${qalign_csv}" "${detail_csv}" \
  "${mode}" "${bucket_limit}" "${samples_per_bucket}" "${out_prefix}"

echo "Wrote:"
echo "  ${out_prefix}.csv"
echo "  ${out_prefix}.txt"
