#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  time_resnet50_extra_dataset_parallel.sh [txt_dir] [options]

Purpose:
  Run resnet50 extra posit formats in parallel:
    p4e0..p4e3, p5e0..p5e3, p6e0..p6e3, p7e0..p7e3, p9e0..p9e3

Options:
  --out-dir DIR         directory containing .so and run_time_sp
  --txt-dir DIR         input txt dataset directory
  --shape SHAPE         default 1x3x224x224
  --formats CSV         override formats, e.g. "p6e0,p6e1,p9e2"
  --jobs N              parallel worker count
  --limit N             sample limit (0 => all)
  --warmup N
  --iters N
  --timeout-sec N
  --progress N
  --label-map FILE
  --baseline auto|none|nqdq-f32|qdq-f32
  --no-benchmark|--with-benchmark
  --quire on|off
  --log-dir DIR         per-format logs dir
  --master-log FILE     master log file (default: <log-dir>/resnet50-v1-12-extra-parallel.log)
  --console on|off      print progress to stdout (default: off)

Defaults:
  txt_dir   = ./temp/imagenette_val_224
  out_dir   = ./temp/resnet50_extra
  shape     = 1x3x224x224
  jobs      = 1
  baseline  = auto (use nqdq-f32 if present, otherwise none)
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"

default_formats_csv="p4e0,p4e1,p4e2,p4e3,p5e0,p5e1,p5e2,p5e3,p6e0,p6e1,p6e2,p6e3,p7e0,p7e1,p7e2,p7e3,p9e0,p9e1,p9e2,p9e3"

model_name="resnet50-v1-12"
txt_dir="${src_root}/temp/imagenette_val_224"
out_dir="${src_root}/temp/resnet50_extra"
shape="1x3x224x224"
formats_csv="${default_formats_csv}"
jobs=1
limit=0
warmup=0
iters=1
timeout_sec=0
progress=0
label_map=""
baseline="auto"
no_benchmark=1
quire_mode="on"
log_dir=""
master_log=""
console_mode="off"

if [[ $# -gt 0 ]]; then
  case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  --*)
    ;;
  *)
    txt_dir="$1"
    shift
    ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
  --out-dir)
    out_dir="$2"
    shift 2
    ;;
  --txt-dir)
    txt_dir="$2"
    shift 2
    ;;
  --shape)
    shape="$2"
    shift 2
    ;;
  --formats)
    formats_csv="$2"
    shift 2
    ;;
  --jobs)
    jobs="$2"
    shift 2
    ;;
  --limit)
    limit="$2"
    shift 2
    ;;
  --warmup)
    warmup="$2"
    shift 2
    ;;
  --iters)
    iters="$2"
    shift 2
    ;;
  --timeout-sec)
    timeout_sec="$2"
    shift 2
    ;;
  --progress)
    progress="$2"
    shift 2
    ;;
  --label-map)
    label_map="$2"
    shift 2
    ;;
  --baseline)
    baseline="$2"
    shift 2
    ;;
  --no-benchmark)
    no_benchmark=1
    shift
    ;;
  --with-benchmark)
    no_benchmark=0
    shift
    ;;
  --quire)
    quire_mode="$2"
    shift 2
    ;;
  --log-dir)
    log_dir="$2"
    shift 2
    ;;
  --master-log)
    master_log="$2"
    shift 2
    ;;
  --console)
    console_mode="$2"
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

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

for n in "${jobs}" "${limit}" "${warmup}" "${iters}" "${timeout_sec}" "${progress}"; do
  if ! is_int "${n}"; then
    echo "ERROR: numeric args must be integers"
    exit 2
  fi
done
if [[ "${jobs}" -lt 1 ]]; then
  echo "ERROR: --jobs must be >= 1"
  exit 2
fi
if [[ "${iters}" -lt 1 ]]; then
  echo "ERROR: --iters must be >= 1"
  exit 2
fi
if [[ "${limit}" -lt 0 || "${warmup}" -lt 0 || "${timeout_sec}" -lt 0 || "${progress}" -lt 0 ]]; then
  echo "ERROR: limit/warmup/timeout-sec/progress must be >= 0"
  exit 2
fi
if [[ ! -d "${txt_dir}" ]]; then
  echo "ERROR: txt dir not found: ${txt_dir}"
  exit 2
fi
if [[ ! -f "${out_dir}/run_time_sp" ]]; then
  echo "ERROR: runner not found: ${out_dir}/run_time_sp"
  exit 2
fi
if [[ -n "${label_map}" && ! -f "${label_map}" ]]; then
  echo "ERROR: label map not found: ${label_map}"
  exit 2
fi

if [[ -z "${log_dir}" ]]; then
  log_dir="${out_dir}/extra_parallel_logs"
fi
mkdir -p "${log_dir}"
if [[ -z "${master_log}" ]]; then
  master_log="${log_dir}/${model_name}-extra-parallel.log"
fi
: > "${master_log}"
case "${console_mode}" in
on|off) ;;
*)
  echo "ERROR: --console must be on|off"
  exit 2
  ;;
esac

log_msg() {
  local msg="$1"
  echo "${msg}" >> "${master_log}"
  if [[ "${console_mode}" == "on" ]]; then
    echo "${msg}"
  fi
}

formats=()
IFS=',' read -r -a _raw <<<"${formats_csv}"
for f in "${_raw[@]}"; do
  ff="${f//[[:space:]]/}"
  [[ -n "${ff}" ]] || continue
  if [[ ! "${ff}" =~ ^p(4|5|6|7|9)e[0-3]$ ]]; then
    echo "ERROR: unsupported extra format: ${ff}"
    exit 2
  fi
  formats+=("${ff}")
done
if [[ ${#formats[@]} -eq 0 ]]; then
  echo "ERROR: empty --formats"
  exit 2
fi

baseline_effective="${baseline}"
if [[ "${baseline}" == "auto" ]]; then
  if [[ -f "${out_dir}/${model_name}-nqdq-f32.so" ]]; then
    baseline_effective="nqdq-f32"
  else
    baseline_effective="none"
  fi
fi
case "${baseline_effective}" in
none|nqdq-f32|qdq-f32) ;;
*)
  echo "ERROR: --baseline must be auto|none|nqdq-f32|qdq-f32"
  exit 2
  ;;
esac

log_msg "[config] model=${model_name}"
log_msg "[config] out_dir=${out_dir}"
log_msg "[config] txt_dir=${txt_dir}"
log_msg "[config] jobs=${jobs} limit=${limit} warmup=${warmup} iters=${iters}"
log_msg "[config] baseline=${baseline_effective} (requested=${baseline})"
log_msg "[config] formats=${formats[*]}"
log_msg "[config] logs=${log_dir}"
log_msg "[config] master_log=${master_log}"

batch_start_ns="$(date +%s%N)"

declare -a all_pids=()
declare -A pid_fmt
declare -A fmt_log
declare -A fmt_console
declare -A fmt_rc

for fmt in "${formats[@]}"; do
  so="${out_dir}/${model_name}-qdq-${fmt}.so"
  if [[ ! -f "${so}" ]]; then
    log_msg "WARN: skip missing ${so}"
    continue
  fi

  lf="${log_dir}/${model_name}-${fmt}.dataset.log"
  cf="${log_dir}/${model_name}-${fmt}.console.log"
  fmt_log["${fmt}"]="${lf}"
  fmt_console["${fmt}"]="${cf}"

  cmd=(
    "${script_dir}/time_model_single_format.sh"
    --model-name "${model_name}"
    --format "${fmt}"
    --out-dir "${out_dir}"
    --txt-dir "${txt_dir}"
    --shape "${shape}"
    --limit "${limit}"
    --warmup "${warmup}"
    --iters "${iters}"
    --timeout-sec "${timeout_sec}"
    --progress "${progress}"
    --baseline "${baseline_effective}"
    --quire "${quire_mode}"
    --log-file "${lf}"
  )
  if [[ -n "${label_map}" ]]; then
    cmd+=(--label-map "${label_map}")
  fi
  if [[ "${no_benchmark}" -eq 1 ]]; then
    cmd+=(--no-benchmark)
  else
    cmd+=(--with-benchmark)
  fi

  log_msg "[launch] ${fmt}"
  "${cmd[@]}" >"${cf}" 2>&1 &
  pid=$!
  all_pids+=("${pid}")
  pid_fmt["${pid}"]="${fmt}"

  while true; do
    running="$(jobs -pr | wc -l | tr -d '[:space:]')"
    if [[ "${running}" -lt "${jobs}" ]]; then
      break
    fi
    sleep 0.2
  done
done

if [[ ${#all_pids[@]} -eq 0 ]]; then
  log_msg "ERROR: no runnable extra-format .so found in ${out_dir}"
  echo "ERROR: no runnable extra-format .so found in ${out_dir}"
  exit 2
fi

failed=0
done_idx=0
total_tasks="${#all_pids[@]}"
for pid in "${all_pids[@]}"; do
  fmt="${pid_fmt["${pid}"]}"
  if wait "${pid}"; then
    rc=0
  else
    rc=$?
    failed=$((failed + 1))
  fi
  fmt_rc["${fmt}"]="${rc}"
  log_msg "[done] ${fmt} rc=${rc}"
  done_idx=$((done_idx + 1))
  if [[ "${rc}" -eq 0 ]]; then
    echo "{done ${done_idx}/${total_tasks}}"
  else
    echo "{done ${done_idx}/${total_tasks}} rc=${rc}"
  fi
done

extract_val() {
  local file="$1"
  local key="$2"
  sed -n "s/^  ${key}=//p" "${file}" | tail -n1
}

extract_metric() {
  local file="$1"
  local key="$2"
  sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p" "${file}" | tail -n1
}

extract_gt_top1() {
  local file="$1"
  sed -n 's/^  GT Top1(%)=\([^ ]*\) Top5(%)=.*/\1/p' "${file}" | tail -n1
}

extract_gt_top5() {
  local file="$1"
  sed -n 's/^  GT Top1(%)=[^ ]* Top5(%)=\([^ ]*\).*/\1/p' "${file}" | tail -n1
}

log_msg ""
log_msg "Summary (resnet50 extra formats)"
header="$(printf "%-8s %-4s %-12s %-10s %-10s %-10s %-10s" "format" "rc" "lat_us" "Top1(%)" "Top5Ov" "MAE" "Cosine")"
log_msg "${header}"
for fmt in "${formats[@]}"; do
  rc="${fmt_rc["${fmt}"]:-skip}"
  lf="${fmt_log["${fmt}"]:-}"
  if [[ -z "${lf}" || ! -f "${lf}" ]]; then
    line="$(printf "%-8s %-4s %-12s %-10s %-10s %-10s %-10s" "${fmt}" "${rc}" "-" "-" "-" "-" "-")"
    log_msg "${line}"
    continue
  fi
  lat="$(extract_val "${lf}" "avg_latency_us")"
  t1="$(extract_metric "${lf}" "Top1Match\\(%\\)")"
  t5="$(extract_metric "${lf}" "Top5Overlap")"
  mae="$(extract_metric "${lf}" "MAE")"
  cos="$(extract_metric "${lf}" "Cosine")"
  [[ -n "${lat}" ]] || lat="-"
  [[ -n "${t1}" ]] || t1="-"
  [[ -n "${t5}" ]] || t5="-"
  [[ -n "${mae}" ]] || mae="-"
  [[ -n "${cos}" ]] || cos="-"
  line="$(printf "%-8s %-4s %-12s %-10s %-10s %-10s %-10s" "${fmt}" "${rc}" "${lat}" "${t1}" "${t5}" "${mae}" "${cos}")"
  log_msg "${line}"
done

log_msg ""
log_msg "Summary (resnet50 extra formats, GT)"
header_gt="$(printf "%-8s %-4s %-12s %-12s %-12s" "format" "rc" "lat_us" "GT Top1(%)" "GT Top5(%)")"
log_msg "${header_gt}"
for fmt in "${formats[@]}"; do
  rc="${fmt_rc["${fmt}"]:-skip}"
  lf="${fmt_log["${fmt}"]:-}"
  if [[ -z "${lf}" || ! -f "${lf}" ]]; then
    line_gt="$(printf "%-8s %-4s %-12s %-12s %-12s" "${fmt}" "${rc}" "-" "-" "-")"
    log_msg "${line_gt}"
    continue
  fi
  lat="$(extract_val "${lf}" "avg_latency_us")"
  gt1="$(extract_gt_top1 "${lf}")"
  gt5="$(extract_gt_top5 "${lf}")"
  [[ -n "${lat}" ]] || lat="-"
  [[ -n "${gt1}" ]] || gt1="-"
  [[ -n "${gt5}" ]] || gt5="-"
  line_gt="$(printf "%-8s %-4s %-12s %-12s %-12s" "${fmt}" "${rc}" "${lat}" "${gt1}" "${gt5}")"
  log_msg "${line_gt}"
done

log_msg ""
log_msg "Per-format logs:"
log_msg "  ${log_dir}"
batch_end_ns="$(date +%s%N)"
total_wall_sec="$(awk -v a="${batch_start_ns}" -v b="${batch_end_ns}" 'BEGIN{print (b-a)/1e9}')"
log_msg "[time] total_wall_time_sec=${total_wall_sec}"
echo "Done. master log: ${master_log}"
echo "Total wall time (sec): ${total_wall_sec}"

if [[ "${failed}" -ne 0 ]]; then
  log_msg "WARN: ${failed} format(s) failed"
  echo "WARN: ${failed} format(s) failed. see ${master_log}"
  exit 1
fi
