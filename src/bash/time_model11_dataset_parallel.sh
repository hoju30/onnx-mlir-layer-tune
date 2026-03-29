#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  time_model11_dataset_parallel.sh --model-name NAME --out-dir DIR --txt-dir DIR \
    [--shape NxCxHxW|N,C,H,W] [--limit N] [--jobs N] [--progress N] \
    [--warmup N] [--iters N] [--timeout-sec N] [--label-map FILE] \
    [--no-benchmark|--with-benchmark] [--quire on|off] [--log-file FILE]

Purpose:
  Run available .so variants with multi-process scheduling.
  Idle process will pick next .so automatically.

Output:
  - Per-so average latency (us)
  - Per-so metrics vs baseline nqdq-f32:
      MAE RMSE MaxAbs Cosine RMAE JS Top1Match(%) Top5Overlap
  - Optional GT Top1/Top5 (if label-map provided)
  - Total wall time

Notes:
  - Default is --no-benchmark (single infer per so)
USAGE_EOF
}

model_name=""
out_dir=""
txt_dir=""
shape="1x1x28x28"
limit=0
jobs=1
progress=0
warmup=0
iters=1
timeout_sec=0
label_map=""
no_benchmark=1
quire_mode="on"
log_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-name)
    model_name="$2"
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
  --jobs)
    jobs="$2"
    shift 2
    ;;
  --progress)
    progress="$2"
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
  nan|+nan|-nan|inf|+inf|-inf)
    return 1
    ;;
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

for n in "${limit}" "${jobs}" "${progress}" "${warmup}" "${iters}" "${timeout_sec}"; do
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
if [[ "${warmup}" -lt 0 || "${limit}" -lt 0 || "${progress}" -lt 0 || "${timeout_sec}" -lt 0 ]]; then
  echo "ERROR: limit/progress/warmup/timeout-sec must be >= 0"
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

model_suffixes=(
  # qdq-p*e* models compute in posit internally but export f32 logits.
  "qdq-p8e0:f32"
  "qdq-p8e1:f32"
  "qdq-p8e2:f32"
  "qdq-p16e0:f32"
  "qdq-p16e1:f32"
  "qdq-p16e2:f32"
  "qdq-p32e0:f32"
  "qdq-p32e1:f32"
  "qdq-p32e2:f32"
  "qdq-f32:f32"
  "nqdq-f32:f32"
)

declare -a keys sos types entries
for spec in "${model_suffixes[@]}"; do
  suffix="${spec%%:*}"
  ty="${spec##*:}"
  key="${model_name}-${suffix}.so"
  so="${out_dir}/${key}"
  if [[ ! -f "${so}" ]]; then
    echo "WARN: skip missing ${so}"
    continue
  fi
  keys+=("${key}")
  sos+=("${so}")
  types+=("${ty}")
  entries+=("$(detect_entry "${so}")")
done

if [[ ${#keys[@]} -eq 0 ]]; then
  echo "ERROR: no runnable .so found under ${out_dir}"
  exit 2
fi

baseline_key="${model_name}-nqdq-f32.so"
baseline_so="${out_dir}/${baseline_key}"
baseline_entry="$(detect_entry "${baseline_so}")"

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

if [[ -z "${log_file}" ]]; then
  log_file="${out_dir}/${model_name}-11.dataset_parallel.log"
fi
mkdir -p "$(dirname "${log_file}")"
: > "${log_file}"

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

checkpoint_summary() {
  local done_n="$1"
  {
    echo "[checkpoint] ${done_n}/${limit}"
    echo "  samples_used=${used}"
    echo "  samples_failed=${failed}"
    echo "  samples_with_any_model_fail=${partial_fail}"
    if [[ "${label_enabled}" -eq 1 ]]; then
      echo "  samples_missing_label=${missing_label}"
    fi
    for key in "${keys[@]}"; do
      local c_lat c_mae c_top1 c_top1_pct
      c_lat="$(avg_value "${lat_sum[$key]:-0}" "${lat_cnt[$key]:-0}")"
      c_mae="$(avg_value "${mae_sum[$key]:-0}" "${mae_cnt[$key]:-0}")"
      c_top1="$(avg_value "${top1_match_yes[$key]:-0}" "${top1_match_cnt[$key]:-0}")"
      c_top1_pct="$(awk -v r="${c_top1}" 'BEGIN{print r*100}')"
      echo "  ${key}: Lat(us)=${c_lat} MAE=${c_mae} Top1Match(%)=${c_top1_pct}"
    done
    echo
  } >> "${log_file}"
}

{
  echo "time_model11_dataset_parallel config"
  echo "  model_name=${model_name}"
  echo "  out_dir=${out_dir}"
  echo "  txt_dir=${txt_dir}"
  echo "  shape=${shape_x}"
  echo "  jobs=${jobs}"
  echo "  limit=${limit}"
  echo "  warmup=${warmup}"
  echo "  iters=${iters}"
  echo "  timeout_sec=${timeout_sec}"
  if [[ "${no_benchmark}" -eq 1 ]]; then
    echo "  mode=no-benchmark"
  else
    echo "  mode=with-benchmark"
  fi
  if [[ "${label_enabled}" -eq 1 ]]; then
    echo "  label_map=${label_map}"
  fi
  echo "  baseline_so=${baseline_key}"
  echo "  log_file=${log_file}"
  echo
} >> "${log_file}"

script_start_ns="$(date +%s%N)"
used=0
failed=0
partial_fail=0
missing_label=0
first_fail_printed=0

for ((i = 0; i < limit; ++i)); do
  sample="${txts[$i]}"

  lbl=""
  if [[ "${label_enabled}" -eq 1 ]]; then
    base="$(basename "${sample}")"
    lbl="${label_by_name["${sample}"]:-${label_by_name["${base}"]:-}}"
    if [[ -z "${lbl}" ]]; then
      missing_label=$((missing_label + 1))
    fi
  fi

  tmpd="$(mktemp -d /tmp/model11_jobs.XXXXXX)"
  running=0

  for ((m = 0; m < ${#keys[@]}; ++m)); do
    key="${keys[$m]}"
    so="${sos[$m]}"
    ty="${types[$m]}"
    ent="${entries[$m]}"
    logf="${tmpd}/${m}.log"
    rcf="${tmpd}/${m}.rc"

    (
      set +e
      cmd=(
        "${runner}" "${so}" "${sample}"
        --shape "${shape_x}"
        --out-type "${ty}"
        --entry "${ent}"
        --warmup "${warmup}"
        --iters "${iters}"
        --quiet
        --cmp "${baseline_so}:f32:${baseline_entry}"
        --baseline "cmp:1"
      )
      if [[ "${runner_supports_quire}" -eq 1 ]]; then
        cmd+=(--quire "${quire_mode}")
      fi
      if [[ "${no_benchmark}" -eq 1 ]]; then
        cmd+=(--no-benchmark)
      fi
      if [[ -n "${lbl}" ]]; then
        cmd+=(--label "${lbl}")
      fi

      if [[ "${timeout_sec}" -gt 0 ]]; then
        timeout "${timeout_sec}" "${cmd[@]}"
      else
        "${cmd[@]}"
      fi
      echo "$?" > "${rcf}"
    ) > "${logf}" 2>&1 &

    running=$((running + 1))
    if [[ "${running}" -ge "${jobs}" ]]; then
      wait -n || true
      running=$((running - 1))
    fi
  done
  wait || true

  success_models=0
  failed_models=0

  for ((m = 0; m < ${#keys[@]}; ++m)); do
    key="${keys[$m]}"
    logf="${tmpd}/${m}.log"
    rcf="${tmpd}/${m}.rc"
    rc=1
    if [[ -f "${rcf}" ]]; then
      rc="$(cat "${rcf}")"
    fi

    if [[ "${rc}" -ne 0 ]]; then
      failed_models=$((failed_models + 1))
      if [[ "${rc}" -eq 124 ]]; then
        echo "[hint] rc=124 timeout for sample=${sample} so=${key}" >&2
      elif [[ "${rc}" -eq 139 ]]; then
        echo "[hint] rc=139 segfault for sample=${sample} so=${key}" >&2
      elif [[ "${rc}" -eq 143 ]]; then
        echo "[hint] rc=143 SIGTERM (usually external timeout/kill) for sample=${sample} so=${key}" >&2
      fi
      if [[ "${first_fail_printed}" -eq 0 ]]; then
        first_fail_printed=1
        echo "[first-fail] sample=${sample} so=${key} rc=${rc}" >&2
        sed -n '1,60p' "${logf}" >&2 || true
      fi
      continue
    fi

    success_models=$((success_models + 1))

    lat="$(sed -n 's/^MAIN type=[^ ]* avg=\([^ ]*\) us.*/\1/p' "${logf}" | head -n1)"
    add_sum lat_sum lat_cnt "${key}" "${lat}"

    c1="$(sed -n '/^  C1 target=/p' "${logf}" | head -n1)"
    if [[ -n "${c1}" ]]; then
      mae="$(extract_field "${c1}" "MAE")"
      rmse="$(extract_field "${c1}" "RMSE")"
      maxabs="$(extract_field "${c1}" "MaxAbs")"
      rmae="$(extract_field "${c1}" "RMAE")"
      cosine="$(extract_field "${c1}" "Cosine")"
      js="$(extract_field "${c1}" "JS")"

      add_sum mae_sum mae_cnt "${key}" "${mae}"
      add_sum rmse_sum rmse_cnt "${key}" "${rmse}"
      add_sum maxabs_sum maxabs_cnt "${key}" "${maxabs}"
      add_sum rmae_sum rmae_cnt "${key}" "${rmae}"
      add_sum cos_sum cos_cnt "${key}" "${cosine}"
      add_sum js_sum js_cnt "${key}" "${js}"

      t1m="$(extract_top1_match "${c1}")"
      if [[ "${t1m}" == "match" || "${t1m}" == "diff" ]]; then
        top1_match_cnt["${key}"]=$(( ${top1_match_cnt["${key}"]:-0} + 1 ))
        if [[ "${t1m}" == "match" ]]; then
          top1_match_yes["${key}"]=$(( ${top1_match_yes["${key}"]:-0} + 1 ))
        fi
      fi

      t5ov="$(extract_top5_overlap "${c1}")"
      add_sum top5ov_sum top5ov_cnt "${key}" "${t5ov}"
    fi

    if [[ -n "${lbl}" ]]; then
      gline="$(sed -n '/^  label=[0-9-]\+ top1_hit=\(yes\|no\) top5_hit=\(yes\|no\)$/p' "${logf}" | head -n1)"
      if [[ -n "${gline}" ]]; then
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
      fi
    fi
  done

  rm -rf "${tmpd}"

  if [[ "${success_models}" -gt 0 ]]; then
    used=$((used + 1))
  else
    failed=$((failed + 1))
  fi
  if [[ "${failed_models}" -gt 0 ]]; then
    partial_fail=$((partial_fail + 1))
  fi

  done_n=$((i + 1))
  if [[ "${progress}" -gt 0 && $((done_n % progress)) -eq 0 ]]; then
    checkpoint_summary "${done_n}"
    # Keep terminal output compact: print rc only when there is any model failure
    # for this sample batch point.
    if [[ "${failed_models}" -gt 0 ]]; then
      echo "{done ${done_n}/${limit}} rc=1"
    else
      echo "{done ${done_n}/${limit}}"
    fi
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
echo "  samples_with_any_model_fail=${partial_fail}"
echo "  shape=${shape_x}"
echo "  jobs=${jobs}"
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
echo "  baseline_so=${baseline_key}"
echo "  total_wall_time_sec=${total_sec}"
echo "  avg_wall_time_sec_per_sample=${avg_sample_sec}"

echo
echo "Per-SO latency summary (avg us)"
for key in "${keys[@]}"; do
  avg_lat="$(avg_value "${lat_sum[$key]:-0}" "${lat_cnt[$key]:-0}")"
  echo "  ${key}: ${avg_lat}"
done

echo
echo "Per-SO metric summary vs baseline (${baseline_key})"
for key in "${keys[@]}"; do
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
  echo "GT accuracy summary"
  for key in "${keys[@]}"; do
    gt1_rate="$(avg_value "${gt_top1_yes[$key]:-0}" "${gt_top1_cnt[$key]:-0}")"
    gt5_rate="$(avg_value "${gt_top5_yes[$key]:-0}" "${gt_top5_cnt[$key]:-0}")"
    gt1_pct="$(awk -v r="${gt1_rate}" 'BEGIN{print r*100}')"
    gt5_pct="$(awk -v r="${gt5_rate}" 'BEGIN{print r*100}')"
    echo "  ${key}: Top1(%)=${gt1_pct} Top5(%)=${gt5_pct}"
  done
fi
