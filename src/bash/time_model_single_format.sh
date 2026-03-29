#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  time_model_single_format.sh --model-name NAME --format FORMAT --txt-dir DIR
    [--out-dir DIR] [--shape NxCxHxW|N,C,H,W] [--limit N]
    [--warmup N] [--iters N] [--timeout-sec N] [--progress N]
    [--baseline nqdq-f32|qdq-f32|none] [--label-map FILE]
    [--log-file FILE] [--no-benchmark|--with-benchmark] [--quire on|off]

Formats:
  p*e*        -> use <out-dir>/<model-name>-qdq-p*e*.so
  qdq-f32     -> use <out-dir>/<model-name>-qdq-f32.so
  nqdq-f32    -> use <out-dir>/<model-name>-nqdq-f32.so

Examples:
  time_model_single_format.sh --model-name mobilenetv2-12 --format p8e0 \
    --out-dir ./temp/mobilenet11 --txt-dir ./temp/imagenette_val_224 \
    --label-map ./temp/imagenette_val_224_labels.txt --limit 100 --no-benchmark
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"

model_name=""
format=""
out_dir="${src_root}/temp"
txt_dir=""
shape="1x3x224x224"
limit=0
warmup=0
iters=1
timeout_sec=0
progress=0
baseline="nqdq-f32"
label_map=""
log_file=""
no_benchmark=1
quire_mode="on"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-name)
    model_name="$2"
    shift 2
    ;;
  --format)
    format="$2"
    shift 2
    ;;
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
  --baseline)
    baseline="$2"
    shift 2
    ;;
  --label-map)
    label_map="$2"
    shift 2
    ;;
  --log-file)
    log_file="$2"
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: file not found: $1"
    exit 2
  fi
}

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_number() {
  local v="$1"
  local low="${v,,}"
  case "${low}" in
  nan|+nan|-nan|inf|+inf|-inf)
    return 1
    ;;
  esac
  [[ "${v}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

normalize_shape() {
  local s="$1"
  s="${s//,/x}"
  if [[ ! "${s}" =~ ^[0-9]+x[0-9]+x[0-9]+x[0-9]+$ ]]; then
    echo ""
  else
    echo "${s}"
  fi
}

detect_entry() {
  local so="$1"
  local sym
  sym="$(nm -D "${so}" 2>/dev/null | awk '$3 ~ /^_mlir_ciface_main_graph/ { print $3; exit }')"
  if [[ -z "${sym}" ]]; then
    echo "_mlir_ciface_main_graph"
  else
    echo "${sym}"
  fi
}

extract_field() {
  local line="$1"
  local key="$2"
  echo "${line}" | sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p"
}

extract_top1_match() {
  local line="$1"
  echo "${line}" | sed -n 's/.* Top1([^)]*)=\(match\|diff\).*/\1/p'
}

extract_top5_overlap() {
  local line="$1"
  echo "${line}" | sed -n 's/.* Top5Overlap=\([0-9]\+\).*/\1/p'
}

avg_value() {
  local sum="$1"
  local cnt="$2"
  if [[ "${cnt}" -le 0 ]]; then
    echo "nan"
  else
    awk -v s="${sum}" -v n="${cnt}" 'BEGIN{print s/n}'
  fi
}

checkpoint_summary() {
  local done_n="$1"
  {
    echo "[checkpoint] ${done_n}/${limit}"
    echo "  samples_used=${used}"
    echo "  samples_failed=${failed}"
    echo "  samples_missing_label=${missing_label}"
    echo "  avg_latency_us=$(avg_value "${lat_sum}" "${lat_cnt}")"
    if [[ "${use_cmp}" -eq 1 ]]; then
      local c_mae c_rmse c_maxabs c_cos c_rmae c_js c_t1 c_t1p c_t5
      c_mae="$(avg_value "${mae_sum}" "${mae_cnt}")"
      c_rmse="$(avg_value "${rmse_sum}" "${rmse_cnt}")"
      c_maxabs="$(avg_value "${maxabs_sum}" "${maxabs_cnt}")"
      c_cos="$(avg_value "${cos_sum}" "${cos_cnt}")"
      c_rmae="$(avg_value "${rmae_sum}" "${rmae_cnt}")"
      c_js="$(avg_value "${js_sum}" "${js_cnt}")"
      c_t1="$(avg_value "${top1_match_yes}" "${top1_match_cnt}")"
      c_t1p="$(awk -v r="${c_t1}" 'BEGIN{print r*100}')"
      c_t5="$(avg_value "${top5ov_sum}" "${top5ov_cnt}")"
      echo "  baseline=${baseline}"
      echo "  MAE=${c_mae} RMSE=${c_rmse} MaxAbs=${c_maxabs} Cosine=${c_cos} RMAE=${c_rmae} JS=${c_js} Top1Match(%)=${c_t1p} Top5Overlap=${c_t5}"
    fi
    if [[ "${label_enabled}" -eq 1 ]]; then
      local g1 g5 g1p g5p
      g1="$(avg_value "${gt_top1_yes}" "${gt_top1_cnt}")"
      g5="$(avg_value "${gt_top5_yes}" "${gt_top5_cnt}")"
      g1p="$(awk -v r="${g1}" 'BEGIN{print r*100}')"
      g5p="$(awk -v r="${g5}" 'BEGIN{print r*100}')"
      echo "  GT Top1(%)=${g1p} Top5(%)=${g5p}"
    fi
    echo
  } >> "${log_file}"
}

if [[ -z "${model_name}" || -z "${format}" || -z "${txt_dir}" ]]; then
  usage
  exit 2
fi

shape_x="$(normalize_shape "${shape}")"
if [[ -z "${shape_x}" ]]; then
  echo "ERROR: invalid --shape: ${shape}"
  exit 2
fi

for n in "${limit}" "${warmup}" "${iters}" "${timeout_sec}" "${progress}"; do
  if ! is_int "${n}"; then
    echo "ERROR: numeric args must be integers"
    exit 2
  fi
done
if [[ "${limit}" -lt 0 || "${warmup}" -lt 0 || "${timeout_sec}" -lt 0 || "${progress}" -lt 0 ]]; then
  echo "ERROR: limit/warmup/timeout-sec/progress must be >= 0"
  exit 2
fi
if [[ "${iters}" -lt 1 ]]; then
  echo "ERROR: --iters must be >= 1"
  exit 2
fi

runner="${out_dir}/run_time_sp"
require_file "${runner}"

if [[ "${no_benchmark}" -eq 1 ]]; then
  if ! "${runner}" --help 2>&1 | grep -q -- "--no-benchmark"; then
    echo "WARN: ${runner} does not support --no-benchmark, fallback to --with-benchmark"
    no_benchmark=0
  fi
fi

runner_supports_quire=1
if ! "${runner}" --help 2>&1 | grep -q -- "--quire"; then
  runner_supports_quire=0
  echo "WARN: ${runner} does not support --quire, ignore quire mode (${quire_mode})"
fi

target_so=""
target_type=""
if [[ "${format}" =~ ^p[0-9]+e[0-9]+$ ]]; then
  target_so="${out_dir}/${model_name}-qdq-${format}.so"
  # Current qdq-p*e* pipelines still return f32 logits at graph boundary.
  # Decode output as f32 here; posit is used internally in lowered kernels.
  target_type="f32"
elif [[ "${format}" == "qdq-f32" ]]; then
  target_so="${out_dir}/${model_name}-qdq-f32.so"
  target_type="f32"
elif [[ "${format}" == "nqdq-f32" ]]; then
  target_so="${out_dir}/${model_name}-nqdq-f32.so"
  target_type="f32"
else
  echo "ERROR: unsupported --format: ${format}"
  exit 2
fi
require_file "${target_so}"
target_entry="$(detect_entry "${target_so}")"

use_cmp=1
baseline_so=""
baseline_type="f32"
baseline_entry=""
case "${baseline}" in
none)
  use_cmp=0
  ;;
nqdq-f32)
  baseline_so="${out_dir}/${model_name}-nqdq-f32.so"
  baseline_type="f32"
  ;;
qdq-f32)
  baseline_so="${out_dir}/${model_name}-qdq-f32.so"
  baseline_type="f32"
  ;;
*)
  echo "ERROR: unsupported --baseline: ${baseline} (use nqdq-f32|qdq-f32|none)"
  exit 2
  ;;
esac

if [[ "${use_cmp}" -eq 1 ]]; then
  require_file "${baseline_so}"
  baseline_entry="$(detect_entry "${baseline_so}")"
fi

if [[ ! -d "${txt_dir}" ]]; then
  echo "ERROR: txt dir not found: ${txt_dir}"
  exit 2
fi
mapfile -t txts < <(find "${txt_dir}" -maxdepth 1 -type f -name '*.txt' | sort)
if [[ ${#txts[@]} -eq 0 ]]; then
  echo "ERROR: no txt samples in ${txt_dir}"
  exit 2
fi
if [[ "${limit}" -le 0 || "${limit}" -gt ${#txts[@]} ]]; then
  limit="${#txts[@]}"
fi

declare -A label_by_name
label_enabled=0
if [[ -n "${label_map}" ]]; then
  require_file "${label_map}"
  while IFS= read -r raw; do
    [[ -z "${raw}" ]] && continue
    [[ "${raw}" =~ ^[[:space:]]*# ]] && continue
    line="${raw//,/ }"
    read -r f l _rest <<<"${line}"
    [[ -z "${f}" || -z "${l}" ]] && continue
    is_int "${l}" || continue
    label_by_name["${f}"]="${l}"
    label_by_name["$(basename "${f}")"]="${l}"
  done < "${label_map}"
  if [[ ${#label_by_name[@]} -gt 0 ]]; then
    label_enabled=1
  else
    echo "WARN: no valid labels parsed from ${label_map}, continue without GT top1/top5"
  fi
fi

if [[ -z "${log_file}" ]]; then
  log_file="${out_dir}/${model_name}-${format}.dataset.log"
fi
mkdir -p "$(dirname "${log_file}")"
: > "${log_file}"

{
  echo "time_model_single_format config"
  echo "  model_name=${model_name}"
  echo "  format=${format}"
  echo "  target_so=${target_so}"
  echo "  target_type=${target_type}"
  echo "  txt_dir=${txt_dir}"
  echo "  limit=${limit}"
  echo "  shape=${shape_x}"
  echo "  warmup=${warmup}"
  echo "  iters=${iters}"
  echo "  timeout_sec=${timeout_sec}"
  echo "  baseline=${baseline}"
  if [[ "${use_cmp}" -eq 1 ]]; then
    echo "  baseline_so=${baseline_so}"
  fi
  if [[ "${label_enabled}" -eq 1 ]]; then
    echo "  label_map=${label_map}"
  fi
  if [[ "${no_benchmark}" -eq 1 ]]; then
    echo "  mode=no-benchmark"
  else
    echo "  mode=with-benchmark"
  fi
  echo
} | tee -a "${log_file}"

used=0
failed=0
missing_label=0
first_fail_printed=0

lat_sum=0
lat_cnt=0

mae_sum=0
mae_cnt=0
rmse_sum=0
rmse_cnt=0
maxabs_sum=0
maxabs_cnt=0
cos_sum=0
cos_cnt=0
rmae_sum=0
rmae_cnt=0
js_sum=0
js_cnt=0
top1_match_yes=0
top1_match_cnt=0
top5ov_sum=0
top5ov_cnt=0
gt_top1_yes=0
gt_top1_cnt=0
gt_top5_yes=0
gt_top5_cnt=0

script_start_ns="$(date +%s%N)"

for ((i = 0; i < limit; ++i)); do
  sample="${txts[$i]}"
  base="$(basename "${sample}")"
  lbl=""
  if [[ "${label_enabled}" -eq 1 ]]; then
    lbl="${label_by_name["${sample}"]:-${label_by_name["${base}"]:-}}"
    if [[ -z "${lbl}" ]]; then
      missing_label=$((missing_label + 1))
    fi
  fi

  cmd=(
    "${runner}" "${target_so}" "${sample}"
    --shape "${shape_x}"
    --out-type "${target_type}"
    --entry "${target_entry}"
    --warmup "${warmup}"
    --iters "${iters}"
    --quiet
  )
  if [[ "${runner_supports_quire}" -eq 1 ]]; then
    cmd+=(--quire "${quire_mode}")
  fi
  if [[ "${use_cmp}" -eq 1 ]]; then
    cmd+=(--cmp "${baseline_so}:${baseline_type}:${baseline_entry}" --baseline "cmp:1")
  fi
  if [[ "${no_benchmark}" -eq 1 ]]; then
    cmd+=(--no-benchmark)
  fi
  if [[ -n "${lbl}" ]]; then
    cmd+=(--label "${lbl}")
  fi

  tmp_log="$(mktemp /tmp/single_fmt.XXXXXX.log)"
  set +e
  if [[ "${timeout_sec}" -gt 0 ]]; then
    timeout "${timeout_sec}" "${cmd[@]}" > "${tmp_log}" 2>&1
  else
    "${cmd[@]}" > "${tmp_log}" 2>&1
  fi
  rc=$?
  set -e

  {
    echo "=== sample[$((i + 1))/${limit}] ${sample} rc=${rc} ==="
    cat "${tmp_log}"
    echo
  } >> "${log_file}"

  if [[ "${rc}" -ne 0 ]]; then
    failed=$((failed + 1))
    if [[ "${rc}" -eq 124 ]]; then
      echo "[hint] rc=124 timeout for sample=${sample}" | tee -a "${log_file}"
    elif [[ "${rc}" -eq 139 ]]; then
      echo "[hint] rc=139 segfault for sample=${sample}" | tee -a "${log_file}"
    elif [[ "${rc}" -eq 143 ]]; then
      echo "[hint] rc=143 SIGTERM for sample=${sample}" | tee -a "${log_file}"
    fi
    if [[ "${first_fail_printed}" -eq 0 ]]; then
      first_fail_printed=1
      echo "[first-fail] sample=${sample} rc=${rc}" | tee -a "${log_file}"
      sed -n '1,60p' "${tmp_log}" | tee -a "${log_file}"
    fi
    rm -f "${tmp_log}"
    continue
  fi

  used=$((used + 1))

  lat="$(sed -n 's/^MAIN type=[^ ]* avg=\([^ ]*\) us.*/\1/p' "${tmp_log}" | head -n1)"
  if is_number "${lat}"; then
    lat_sum="$(awk -v a="${lat_sum}" -v b="${lat}" 'BEGIN{print a+b}')"
    lat_cnt=$((lat_cnt + 1))
  fi

  if [[ "${use_cmp}" -eq 1 ]]; then
    c1="$(sed -n '/^  C1 target=/p' "${tmp_log}" | head -n1)"
    if [[ -n "${c1}" ]]; then
      mae="$(extract_field "${c1}" "MAE")"
      rmse="$(extract_field "${c1}" "RMSE")"
      maxabs="$(extract_field "${c1}" "MaxAbs")"
      cosine="$(extract_field "${c1}" "Cosine")"
      rmae="$(extract_field "${c1}" "RMAE")"
      js="$(extract_field "${c1}" "JS")"
      t1m="$(extract_top1_match "${c1}")"
      t5ov="$(extract_top5_overlap "${c1}")"

      if is_number "${mae}"; then
        mae_sum="$(awk -v a="${mae_sum}" -v b="${mae}" 'BEGIN{print a+b}')"
        mae_cnt=$((mae_cnt + 1))
      fi
      if is_number "${rmse}"; then
        rmse_sum="$(awk -v a="${rmse_sum}" -v b="${rmse}" 'BEGIN{print a+b}')"
        rmse_cnt=$((rmse_cnt + 1))
      fi
      if is_number "${maxabs}"; then
        maxabs_sum="$(awk -v a="${maxabs_sum}" -v b="${maxabs}" 'BEGIN{print a+b}')"
        maxabs_cnt=$((maxabs_cnt + 1))
      fi
      if is_number "${cosine}"; then
        cos_sum="$(awk -v a="${cos_sum}" -v b="${cosine}" 'BEGIN{print a+b}')"
        cos_cnt=$((cos_cnt + 1))
      fi
      if is_number "${rmae}"; then
        rmae_sum="$(awk -v a="${rmae_sum}" -v b="${rmae}" 'BEGIN{print a+b}')"
        rmae_cnt=$((rmae_cnt + 1))
      fi
      if is_number "${js}"; then
        js_sum="$(awk -v a="${js_sum}" -v b="${js}" 'BEGIN{print a+b}')"
        js_cnt=$((js_cnt + 1))
      fi
      if [[ "${t1m}" == "match" || "${t1m}" == "diff" ]]; then
        top1_match_cnt=$((top1_match_cnt + 1))
        if [[ "${t1m}" == "match" ]]; then
          top1_match_yes=$((top1_match_yes + 1))
        fi
      fi
      if is_number "${t5ov}"; then
        top5ov_sum="$(awk -v a="${top5ov_sum}" -v b="${t5ov}" 'BEGIN{print a+b}')"
        top5ov_cnt=$((top5ov_cnt + 1))
      fi
    fi
  fi

  if [[ -n "${lbl}" ]]; then
    gline="$(sed -n '/^  label=[0-9-]\+ top1_hit=\(yes\|no\) top5_hit=\(yes\|no\)$/p' "${tmp_log}" | head -n1)"
    if [[ -n "${gline}" ]]; then
      t1="$(echo "${gline}" | sed -n 's/.* top1_hit=\(yes\|no\).*/\1/p')"
      t5="$(echo "${gline}" | sed -n 's/.* top5_hit=\(yes\|no\).*/\1/p')"
      if [[ -n "${t1}" ]]; then
        gt_top1_cnt=$((gt_top1_cnt + 1))
        if [[ "${t1}" == "yes" ]]; then
          gt_top1_yes=$((gt_top1_yes + 1))
        fi
      fi
      if [[ -n "${t5}" ]]; then
        gt_top5_cnt=$((gt_top5_cnt + 1))
        if [[ "${t5}" == "yes" ]]; then
          gt_top5_yes=$((gt_top5_yes + 1))
        fi
      fi
    fi
  fi

  rm -f "${tmp_log}"

  done_n=$((i + 1))
  if [[ "${progress}" -gt 0 && $((done_n % progress)) -eq 0 ]]; then
    checkpoint_summary "${done_n}"
    echo "[progress] ${done_n}/${limit}" | tee -a "${log_file}"
  fi
done

script_end_ns="$(date +%s%N)"
total_sec="$(awk -v a="${script_start_ns}" -v b="${script_end_ns}" 'BEGIN{print (b-a)/1e9}')"

if [[ "${used}" -eq 0 ]]; then
  echo "ERROR: no valid sample parsed (failed=${failed})" | tee -a "${log_file}"
  exit 3
fi

avg_lat="$(avg_value "${lat_sum}" "${lat_cnt}")"

{
  echo
  echo "Summary"
  echo "  target=${model_name}-${format}"
  echo "  samples_used=${used}"
  echo "  samples_failed=${failed}"
  echo "  samples_missing_label=${missing_label}"
  echo "  avg_latency_us=${avg_lat}"
  echo "  total_wall_time_sec=${total_sec}"
  if [[ "${use_cmp}" -eq 1 ]]; then
    avg_mae="$(avg_value "${mae_sum}" "${mae_cnt}")"
    avg_rmse="$(avg_value "${rmse_sum}" "${rmse_cnt}")"
    avg_maxabs="$(avg_value "${maxabs_sum}" "${maxabs_cnt}")"
    avg_cos="$(avg_value "${cos_sum}" "${cos_cnt}")"
    avg_rmae="$(avg_value "${rmae_sum}" "${rmae_cnt}")"
    avg_js="$(avg_value "${js_sum}" "${js_cnt}")"
    t1_rate="$(avg_value "${top1_match_yes}" "${top1_match_cnt}")"
    t1_pct="$(awk -v r="${t1_rate}" 'BEGIN{print r*100}')"
    t5ov_avg="$(avg_value "${top5ov_sum}" "${top5ov_cnt}")"
    echo "  baseline=${baseline}"
    echo "  MAE=${avg_mae} RMSE=${avg_rmse} MaxAbs=${avg_maxabs} Cosine=${avg_cos} RMAE=${avg_rmae} JS=${avg_js} Top1Match(%)=${t1_pct} Top5Overlap=${t5ov_avg}"
  fi
  if [[ "${label_enabled}" -eq 1 ]]; then
    gt1_rate="$(avg_value "${gt_top1_yes}" "${gt_top1_cnt}")"
    gt5_rate="$(avg_value "${gt_top5_yes}" "${gt_top5_cnt}")"
    gt1_pct="$(awk -v r="${gt1_rate}" 'BEGIN{print r*100}')"
    gt5_pct="$(awk -v r="${gt5_rate}" 'BEGIN{print r*100}')"
    echo "  GT Top1(%)=${gt1_pct} Top5(%)=${gt5_pct}"
  fi
  echo "  log_file=${log_file}"
} | tee -a "${log_file}"
