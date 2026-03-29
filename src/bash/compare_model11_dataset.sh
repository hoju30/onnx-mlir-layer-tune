#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  compare_model11_dataset.sh --model-name NAME --out-dir DIR --txt-dir DIR \
    [--shape NxCxHxW|N,C,H,W] [--limit N] [--warmup N] [--iters N] [--progress N] \
    [--timeout-sec N] [--label-map FILE] [--no-benchmark|--with-benchmark] \
    [--quire on|off]

One-pass all-model dataset compare (baseline: nqdq-f32):
  main target: qdq-p8e0
  compares: qdq-p8e1/p8e2/p16e0/p16e1/p16e2/p32e0/p32e1/p32e2/qdq-f32/nqdq-f32

Notes:
  - label-map is optional.
  - label-map format: each line '<txt_filename_or_path> <label_id>'
    (comma-separated is also accepted)
  - default mode is --no-benchmark (each so runs one infer only)
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name=""
out_dir=""
txt_dir=""
shape="1x1x28x28"
limit=0
warmup=0
iters=1
progress=0
timeout_sec=0
label_map=""
no_benchmark=1
quire_mode="on"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-name)
    [[ $# -ge 2 ]] || { echo "ERROR: --model-name needs a value"; exit 2; }
    model_name="$2"
    shift 2
    ;;
  --out-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --out-dir needs a path"; exit 2; }
    out_dir="$2"
    shift 2
    ;;
  --txt-dir)
    [[ $# -ge 2 ]] || { echo "ERROR: --txt-dir needs a path"; exit 2; }
    txt_dir="$2"
    shift 2
    ;;
  --shape)
    [[ $# -ge 2 ]] || { echo "ERROR: --shape needs NxCxHxW or N,C,H,W"; exit 2; }
    shape="$2"
    shift 2
    ;;
  --limit)
    [[ $# -ge 2 ]] || { echo "ERROR: --limit needs a number"; exit 2; }
    limit="$2"
    shift 2
    ;;
  --warmup)
    [[ $# -ge 2 ]] || { echo "ERROR: --warmup needs a number"; exit 2; }
    warmup="$2"
    shift 2
    ;;
  --iters)
    [[ $# -ge 2 ]] || { echo "ERROR: --iters needs a number"; exit 2; }
    iters="$2"
    shift 2
    ;;
  --progress)
    [[ $# -ge 2 ]] || { echo "ERROR: --progress needs a number"; exit 2; }
    progress="$2"
    shift 2
    ;;
  --timeout-sec)
    [[ $# -ge 2 ]] || { echo "ERROR: --timeout-sec needs a number"; exit 2; }
    timeout_sec="$2"
    shift 2
    ;;
  --label-map)
    [[ $# -ge 2 ]] || { echo "ERROR: --label-map needs a path"; exit 2; }
    label_map="$2"
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
    [[ $# -ge 2 ]] || { echo "ERROR: --quire needs on/off"; exit 2; }
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

if [[ -z "${model_name}" || -z "${out_dir}" || -z "${txt_dir}" ]]; then
  usage
  exit 2
fi

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_number() {
  local v="$1"
  local low="${v,,}"
  case "${low}" in
    nan|+nan|-nan|inf|+inf|-inf) return 1 ;;
  esac
  [[ "${v}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

add_sum() {
  local -n sum_ref="$1"
  local -n cnt_ref="$2"
  local key="$3"
  local val="$4"
  if ! is_number "${val}"; then
    return
  fi
  local prev="${sum_ref[$key]:-0}"
  sum_ref[$key]="$(awk -v a="${prev}" -v b="${val}" 'BEGIN{print a+b}')"
  cnt_ref[$key]=$(( ${cnt_ref[$key]:-0} + 1 ))
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: file not found: $1"
    exit 2
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

shape_x="$(normalize_shape "${shape}")"
if [[ -z "${shape_x}" ]]; then
  echo "ERROR: invalid --shape: ${shape}"
  exit 2
fi

if ! is_int "${limit}" || [[ "${limit}" -lt 0 ]]; then
  echo "ERROR: --limit must be >= 0"
  exit 2
fi
if ! is_int "${warmup}" || [[ "${warmup}" -lt 0 ]]; then
  echo "ERROR: --warmup must be >= 0"
  exit 2
fi
if ! is_int "${iters}" || [[ "${iters}" -lt 1 ]]; then
  echo "ERROR: --iters must be >= 1"
  exit 2
fi
if ! is_int "${progress}" || [[ "${progress}" -lt 0 ]]; then
  echo "ERROR: --progress must be >= 0"
  exit 2
fi
if ! is_int "${timeout_sec}" || [[ "${timeout_sec}" -lt 0 ]]; then
  echo "ERROR: --timeout-sec must be >= 0"
  exit 2
fi

runner="${out_dir}/run_time_sp"
main_decode_type="f32"
main_so="${out_dir}/${model_name}-qdq-p8e0.so"
if [[ ! -f "${main_so}" ]]; then
  mapfile -t _posit_candidates < <(find "${out_dir}" -maxdepth 1 -type f -name "${model_name}-qdq-p*e*.so" | sort)
  if [[ ${#_posit_candidates[@]} -eq 0 ]]; then
    echo "ERROR: no posit qdq .so found in ${out_dir}"
    exit 2
  fi
  main_so="${_posit_candidates[0]}"
  echo "WARN: ${model_name}-qdq-p8e0.so missing, use main=${main_so}"
fi
main_key="$(basename "${main_so}")"

cmp_sos=()
cmp_types=()
candidate_suffixes=(
  qdq-p8e0 qdq-p8e1 qdq-p8e2
  qdq-p16e0 qdq-p16e1 qdq-p16e2
  qdq-p32e0 qdq-p32e1 qdq-p32e2
  qdq-f32 nqdq-f32
)
for suffix in "${candidate_suffixes[@]}"; do
  so="${out_dir}/${model_name}-${suffix}.so"
  [[ -f "${so}" ]] || continue
  [[ "${so}" == "${main_so}" ]] && continue
  cmp_sos+=("${so}")
  cmp_types+=("f32")
done

metric_keys=("${main_key}")
latency_keys=("${main_key}")
for so in "${cmp_sos[@]}"; do
  key="$(basename "${so}")"
  latency_keys+=("${key}")
  if [[ "${key}" != "${model_name}-nqdq-f32.so" ]]; then
    metric_keys+=("${key}")
  fi
done
cmp_latency_keys=("${latency_keys[@]:1}")

baseline_cmp_index=-1
for ((i = 0; i < ${#cmp_sos[@]}; ++i)); do
  if [[ "$(basename "${cmp_sos[$i]}")" == "${model_name}-nqdq-f32.so" ]]; then
    baseline_cmp_index=$((i + 1))
    break
  fi
done
if [[ "${baseline_cmp_index}" -lt 0 ]]; then
  echo "ERROR: baseline ${model_name}-nqdq-f32.so missing in ${out_dir}"
  exit 2
fi

require_file "${runner}"
require_file "${main_so}"
for so in "${cmp_sos[@]}"; do
  require_file "${so}"
done
if [[ ! -d "${txt_dir}" ]]; then
  echo "ERROR: txt dir not found: ${txt_dir}"
  exit 2
fi

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

mapfile -t txts < <(find "${txt_dir}" -maxdepth 1 -type f -name '*.txt' | sort)
if [[ ${#txts[@]} -eq 0 ]]; then
  echo "ERROR: no txt samples in ${txt_dir}"
  exit 2
fi
if [[ "${limit}" -le 0 || "${limit}" -gt ${#txts[@]} ]]; then
  limit="${#txts[@]}"
fi

main_entry="$(detect_entry "${main_so}")"
declare -a cmp_entries
for so in "${cmp_sos[@]}"; do
  cmp_entries+=("$(detect_entry "${so}")")
done

declare -A label_by_name
label_enabled=0
if [[ -n "${label_map}" ]]; then
  require_file "${label_map}"
  while IFS= read -r raw; do
    [[ -z "${raw}" ]] && continue
    [[ "${raw}" =~ ^[[:space:]]*# ]] && continue
    line="${raw//,/ }"
    read -r f l _rest <<<"${line}"
    if [[ -z "${f}" || -z "${l}" ]]; then
      continue
    fi
    if ! is_int "${l}"; then
      continue
    fi
    label_by_name["${f}"]="${l}"
    label_by_name["$(basename "${f}")"]="${l}"
  done < "${label_map}"
  if [[ ${#label_by_name[@]} -gt 0 ]]; then
    label_enabled=1
  else
    echo "WARN: no valid labels parsed from ${label_map}, continue without GT top1/top5"
  fi
fi

declare -A lat_sum lat_cnt
declare -A mae_sum mae_cnt
declare -A rmse_sum rmse_cnt
declare -A maxabs_sum maxabs_cnt
declare -A cos_sum cos_cnt
declare -A rmae_sum rmae_cnt
declare -A js_sum js_cnt
declare -A top1_match_yes top1_match_cnt
declare -A top5ov_sum top5ov_cnt
declare -A gt_top1_yes gt_top1_cnt
declare -A gt_top5_yes gt_top5_cnt

script_start_ns="$(date +%s%N)"
used=0
failed=0
missing_label=0

for ((i = 0; i < limit; ++i)); do
  sample="${txts[$i]}"

  cmd=(
    "${runner}" "${main_so}" "${sample}"
    --shape "${shape_x}"
    --out-type "${main_decode_type}"
    --entry "${main_entry}"
    --warmup "${warmup}"
    --iters "${iters}"
    --quiet
  )
  if [[ "${runner_supports_quire}" -eq 1 ]]; then
    cmd+=(--quire "${quire_mode}")
  fi
  if [[ "${no_benchmark}" -eq 1 ]]; then
    cmd+=(--no-benchmark)
  fi

  if [[ "${label_enabled}" -eq 1 ]]; then
    base="$(basename "${sample}")"
    lbl="${label_by_name["${sample}"]:-${label_by_name["${base}"]:-}}"
    if [[ -n "${lbl}" ]]; then
      cmd+=(--label "${lbl}")
    else
      missing_label=$((missing_label + 1))
    fi
  fi

  for ((j = 0; j < ${#cmp_sos[@]}; ++j)); do
    cmd+=(--cmp "${cmp_sos[$j]}:${cmp_types[$j]}:${cmp_entries[$j]}")
  done
  cmd+=(--baseline "cmp:${baseline_cmp_index}")

  if [[ "${timeout_sec}" -gt 0 ]]; then
    set +e
    out="$(timeout "${timeout_sec}" "${cmd[@]}" 2>&1)"
    rc=$?
    set -e
  else
    set +e
    out="$("${cmd[@]}" 2>&1)"
    rc=$?
    set -e
  fi

  if [[ ${rc} -ne 0 ]]; then
    failed=$((failed + 1))
    if [[ ${rc} -eq 124 ]]; then
      echo "[hint] rc=124 timeout for sample=${sample}" >&2
    elif [[ ${rc} -eq 139 ]]; then
      echo "[hint] rc=139 segfault for sample=${sample}" >&2
    elif [[ ${rc} -eq 143 ]]; then
      echo "[hint] rc=143 SIGTERM (usually external timeout/kill) for sample=${sample}" >&2
    fi
    if [[ ${failed} -eq 1 ]]; then
      echo "[first-fail] sample=${sample} rc=${rc}" >&2
      if [[ -n "${out}" ]]; then
        echo "${out}" | sed -n '1,60p' >&2
      fi
    fi
    continue
  fi

  main_lat="$(echo "${out}" | sed -n 's/^MAIN type=[^ ]* avg=\([^ ]*\) us.*/\1/p' | head -n1)"
  add_sum lat_sum lat_cnt "${main_key}" "${main_lat}"

  for ((j = 1; j <= ${#cmp_latency_keys[@]}; ++j)); do
    key="${cmp_latency_keys[$((j - 1))]}"
    lat="$(echo "${out}" | sed -n "s/^CMP#${j} type=[^ ]* avg=\\([^ ]*\\) us so=.*/\\1/p" | head -n1)"
    add_sum lat_sum lat_cnt "${key}" "${lat}"
  done

  mapfile -t metric_lines < <(echo "${out}" | sed -n '/^  C[0-9]\+ target=/p')
  for ((j = 0; j < ${#metric_keys[@]} && j < ${#metric_lines[@]}; ++j)); do
    key="${metric_keys[$j]}"
    mline="${metric_lines[$j]}"

    mae="$(extract_field "${mline}" "MAE")"
    rmse="$(extract_field "${mline}" "RMSE")"
    maxabs="$(extract_field "${mline}" "MaxAbs")"
    rmae="$(extract_field "${mline}" "RMAE")"
    cosine="$(extract_field "${mline}" "Cosine")"
    js="$(extract_field "${mline}" "JS")"

    add_sum mae_sum mae_cnt "${key}" "${mae}"
    add_sum rmse_sum rmse_cnt "${key}" "${rmse}"
    add_sum maxabs_sum maxabs_cnt "${key}" "${maxabs}"
    add_sum rmae_sum rmae_cnt "${key}" "${rmae}"
    add_sum cos_sum cos_cnt "${key}" "${cosine}"
    add_sum js_sum js_cnt "${key}" "${js}"

    t1m="$(extract_top1_match "${mline}")"
    if [[ "${t1m}" == "match" || "${t1m}" == "diff" ]]; then
      top1_match_cnt["${key}"]=$(( ${top1_match_cnt["${key}"]:-0} + 1 ))
      if [[ "${t1m}" == "match" ]]; then
        top1_match_yes["${key}"]=$(( ${top1_match_yes["${key}"]:-0} + 1 ))
      fi
    fi

    t5ov="$(extract_top5_overlap "${mline}")"
    add_sum top5ov_sum top5ov_cnt "${key}" "${t5ov}"
  done

  if [[ "${label_enabled}" -eq 1 ]]; then
    mapfile -t gt_lines < <(echo "${out}" | sed -n '/^  label=[0-9-]\+ top1_hit=\(yes\|no\) top5_hit=\(yes\|no\)$/p')
    for ((j = 0; j < ${#latency_keys[@]} && j < ${#gt_lines[@]}; ++j)); do
      key="${latency_keys[$j]}"
      gline="${gt_lines[$j]}"
      t1="$(echo "${gline}" | sed -n 's/.* top1_hit=\(yes\|no\).*/\1/p')"
      t5="$(echo "${gline}" | sed -n 's/.* top5_hit=\(yes\|no\).*/\1/p')"
      if [[ -n "${t1}" ]]; then
        gt_top1_cnt["${key}"]=$(( ${gt_top1_cnt["${key}"]:-0} + 1 ))
        if [[ "${t1}" == "yes" ]]; then
          gt_top1_yes["${key}"]=$(( ${gt_top1_yes["${key}"]:-0} + 1 ))
        fi
      fi
      if [[ -n "${t5}" ]]; then
        gt_top5_cnt["${key}"]=$(( ${gt_top5_cnt["${key}"]:-0} + 1 ))
        if [[ "${t5}" == "yes" ]]; then
          gt_top5_yes["${key}"]=$(( ${gt_top5_yes["${key}"]:-0} + 1 ))
        fi
      fi
    done
  fi

  used=$((used + 1))
  if [[ "${progress}" -gt 0 && $((used % progress)) -eq 0 ]]; then
    echo "[progress] ${used}/${limit}"
  fi
done

script_end_ns="$(date +%s%N)"
total_sec="$(awk -v a="${script_start_ns}" -v b="${script_end_ns}" 'BEGIN{print (b-a)/1e9}')"
avg_sample_sec="$(avg_value "${total_sec}" "${used}")"

if [[ "${used}" -eq 0 ]]; then
  echo "ERROR: no valid sample parsed (failed=${failed})"
  exit 3
fi

echo
echo "Dataset summary"
echo "  directory=${txt_dir}"
echo "  samples_used=${used}"
echo "  samples_failed=${failed}"
echo "  shape=${shape_x}"
echo "  warmup=${warmup}"
echo "  iters=${iters}"
if [[ "${timeout_sec}" -gt 0 ]]; then
  echo "  timeout_sec=${timeout_sec}"
fi
if [[ "${no_benchmark}" -eq 1 ]]; then
  echo "  mode=no-benchmark (single infer per so)"
else
  echo "  mode=with-benchmark"
fi
if [[ "${label_enabled}" -eq 1 ]]; then
  echo "  label_map=${label_map}"
  echo "  samples_missing_label=${missing_label}"
fi
echo "  total_wall_time_sec=${total_sec}"
echo "  avg_wall_time_sec_per_sample=${avg_sample_sec}"

echo
echo "Latency summary (avg us)"
for key in "${latency_keys[@]}"; do
  avg_lat="$(avg_value "${lat_sum[$key]:-0}" "${lat_cnt[$key]:-0}")"
  echo "  ${key}: ${avg_lat}"
done

echo
echo "Metric summary vs baseline (${model_name}-nqdq-f32.so)"
for key in "${metric_keys[@]}"; do
  avg_mae="$(avg_value "${mae_sum[$key]:-0}" "${mae_cnt[$key]:-0}")"
  avg_rmse="$(avg_value "${rmse_sum[$key]:-0}" "${rmse_cnt[$key]:-0}")"
  avg_maxabs="$(avg_value "${maxabs_sum[$key]:-0}" "${maxabs_cnt[$key]:-0}")"
  avg_cos="$(avg_value "${cos_sum[$key]:-0}" "${cos_cnt[$key]:-0}")"
  avg_rmae="$(avg_value "${rmae_sum[$key]:-0}" "${rmae_cnt[$key]:-0}")"
  avg_js="$(avg_value "${js_sum[$key]:-0}" "${js_cnt[$key]:-0}")"
  t1_rate="$(avg_value "${top1_match_yes[$key]:-0}" "${top1_match_cnt[$key]:-0}")"
  t1_pct="$(awk -v r="${t1_rate}" 'BEGIN{print r*100}')"
  t5ov_avg="$(avg_value "${top5ov_sum[$key]:-0}" "${top5ov_cnt[$key]:-0}")"
  echo "  ${key}: MAE=${avg_mae} RMSE=${avg_rmse} MaxAbs=${avg_maxabs} Cosine=${avg_cos} RMAE=${avg_rmae} JS=${avg_js} Top1Match(%)=${t1_pct} Top5Overlap=${t5ov_avg}"
done

if [[ "${label_enabled}" -eq 1 ]]; then
  echo
  echo "GT accuracy summary (requires --label-map)"
  for key in "${latency_keys[@]}"; do
    gt1_rate="$(avg_value "${gt_top1_yes[$key]:-0}" "${gt_top1_cnt[$key]:-0}")"
    gt5_rate="$(avg_value "${gt_top5_yes[$key]:-0}" "${gt_top5_cnt[$key]:-0}")"
    gt1_pct="$(awk -v r="${gt1_rate}" 'BEGIN{print r*100}')"
    gt5_pct="$(awk -v r="${gt5_rate}" 'BEGIN{print r*100}')"
    echo "  ${key}: Top1(%)=${gt1_pct} Top5(%)=${gt5_pct}"
  done
fi
