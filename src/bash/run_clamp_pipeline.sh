#!/usr/bin/env bash
# run_clamp_pipeline.sh
#
# One-shot wrapper for the nqdq per-layer clamp experiment. It:
#   1) COLLECT : run the chosen posit .so on N images (parallel, f32 math) with
#                POSIT_COLLECT_LAYER_RANGES_FILE=<parts>/part_i.csv per image.
#   2) MERGE   : merge all part files + derive per-layer clamp [lo,hi] + ALPS theta
#                via merge_layer_ranges_to_clamp.py  -> <clamp_dir>/clamp.csv
#   3) EVAL    : run the full dataset eval via time_model11_dataset_parallel.sh
#                with POSIT_OUTPUT_CLAMP_FILE=<clamp_dir>/clamp.csv
#
# Mirrors the ALPS-offline auto-collect design (per-part files + merge), which is
# parallel-safe (the runner is one-process-per-image).
#
# Env knobs (all optional, sensible defaults):
#   CLAMP_DIR             output dir for parts + clamp.csv (default: <out-dir>/clamp_<suffix>)
#   CLAMP_COLLECT_LIMIT   #images for collection (default: 500)
#   CLAMP_COLLECT_JOBS    parallel jobs for collection (default: --jobs value)
#   CLAMP_SUFFIX          which posit suffix to collect on (default: first non-f32 in --suffixes)
#   CLAMP_PERCENTILE      percentile for [lo,hi] (default: 99)
#   CLAMP_MARGIN          extra fractional headroom (default: 0)
#   CLAMP_THETA_MODE      auto|from_max|fixed (default: auto)
#   CLAMP_FIXED_THETA     theta when mode=fixed (default: 0.1)
#   CLAMP_GAMMA           ALPS gamma (default: 1.0)
#   CLAMP_SYMMETRIC       1 to force symmetric bounds (default: 0)
#   CLAMP_SKIP_COLLECT    1 to reuse existing clamp.csv and skip collect+merge
#
# All other args are forwarded verbatim to time_model11_dataset_parallel.sh.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
imagenet_dir="/home/lai/onnx_mlir/ImageNet100"
eval_script="${script_dir}/time_model11_dataset_parallel.sh"
merge_py="${imagenet_dir}/merge_layer_ranges_to_clamp.py"

# --- Peek at forwarded args we need for the collect phase ---
model_name=""
out_dir=""
image_dir=""
image_preprocess_script=""
shape=""
suffixes=""
limit="5000"
jobs="1"
warmup="0"
iters="1"
no_benchmark=0
quire_mode="off"

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    --model-name) model_name="${args[$((i+1))]}";;
    --out-dir) out_dir="${args[$((i+1))]}";;
    --image-dir) image_dir="${args[$((i+1))]}";;
    --image-preprocess-script) image_preprocess_script="${args[$((i+1))]}";;
    --shape) shape="${args[$((i+1))]}";;
    --suffixes) suffixes="${args[$((i+1))]}";;
    --limit) limit="${args[$((i+1))]}";;
    --jobs) jobs="${args[$((i+1))]}";;
    --warmup) warmup="${args[$((i+1))]}";;
    --iters) iters="${args[$((i+1))]}";;
    --no-benchmark) no_benchmark=1;;
    --quire) quire_mode="${args[$((i+1))]}";;
  esac
  i=$((i+1))
done

if [[ -z "${out_dir}" || -z "${model_name}" || -z "${image_dir}" || -z "${shape}" ]]; then
  echo "ERROR: need --out-dir --model-name --image-dir --shape"
  exit 2
fi

# --- Pick the suffix to collect on ---
clamp_suffix="${CLAMP_SUFFIX:-}"
if [[ -z "${clamp_suffix}" ]]; then
  IFS=',' read -r -a _sfx <<< "${suffixes}"
  for s in "${_sfx[@]}"; do
    if [[ "${s}" != *"-f32" ]]; then clamp_suffix="${s}"; break; fi
  done
fi
if [[ -z "${clamp_suffix}" ]]; then
  echo "ERROR: no posit suffix found; set CLAMP_SUFFIX or include one in --suffixes"
  exit 2
fi

clamp_collect_limit="${CLAMP_COLLECT_LIMIT:-500}"
clamp_collect_jobs="${CLAMP_COLLECT_JOBS:-${jobs}}"
clamp_percentile="${CLAMP_PERCENTILE:-99}"
clamp_margin="${CLAMP_MARGIN:-0}"
clamp_theta_mode="${CLAMP_THETA_MODE:-auto}"   # auto|from_max|fixed|median
clamp_theta_target="${CLAMP_THETA_TARGET:-1.0}"  # for median mode: median→sweet spot
clamp_fixed_theta="${CLAMP_FIXED_THETA:-0.1}"
clamp_gamma="${CLAMP_GAMMA:-1.0}"
clamp_symmetric="${CLAMP_SYMMETRIC:-0}"
clamp_version_a="${CLAMP_VERSION_A:-0}"   # 1 = emit key,lo,hi only (use with ALPS on)
clamp_exclude_keys="${CLAMP_EXCLUDE_KEYS:-}"  # comma-sep keys to skip (e.g. classifier gemm2d)
clamp_dir="${CLAMP_DIR:-${out_dir}/clamp_${clamp_suffix}}"
parts_dir="${clamp_dir}/parts"
clamp_csv="${clamp_dir}/clamp.csv"

runner="${out_dir}/run_time_sp"
so="${out_dir}/${model_name}-${clamp_suffix}.so"
if [[ ! -x "${runner}" ]]; then echo "ERROR: runner missing: ${runner}"; exit 2; fi
if [[ ! -f "${so}" ]]; then echo "ERROR: .so missing: ${so}"; exit 2; fi

detect_entry() {
  local s="$1" sym
  sym="$(nm -D "${s}" 2>/dev/null | awk '$3 ~ /^_mlir_ciface_main_graph/ { print $3; exit }')"
  [[ -z "${sym}" ]] && echo "_mlir_ciface_main_graph" || echo "${sym}"
}

normalize_shape() { echo "$1" | tr ',' 'x'; }
shape_x="$(normalize_shape "${shape}")"
ent="$(detect_entry "${so}")"

runner_supports_quire=1
if ! "${runner}" --help 2>&1 | grep -q -- "--quire"; then runner_supports_quire=0; fi

if [[ "${CLAMP_SKIP_COLLECT:-0}" != "1" ]]; then
  echo "=== [clamp 1/3] COLLECT suffix=${clamp_suffix} limit=${clamp_collect_limit} jobs=${clamp_collect_jobs} ==="

  # Gather images (same convention as time_model11_dataset_parallel.sh)
  image_exts="jpg,jpeg,png,bmp"
  mapfile -t imgs < <(find "${image_dir}" -type f | awk -v csv="${image_exts}" '
    BEGIN { n=split(csv,a,","); for(i=1;i<=n;i++){gsub(/^[ \t.]+|[ \t]+$/,"",a[i]); exts[tolower(a[i])]=1} }
    { p=$0; n=split(p,parts,"."); ext=(n>1?tolower(parts[n]):""); if(ext in exts) print p }' | sort)
  if [[ ${#imgs[@]} -eq 0 ]]; then echo "ERROR: no images in ${image_dir}"; exit 2; fi

  n_collect="${clamp_collect_limit}"
  if [[ "${n_collect}" -le 0 || "${n_collect}" -gt ${#imgs[@]} ]]; then n_collect="${#imgs[@]}"; fi
  cj="${clamp_collect_jobs}"
  [[ "${cj}" -le 0 ]] && cj=1
  [[ "${cj}" -gt "${n_collect}" ]] && cj="${n_collect}"

  rm -rf "${parts_dir}"; mkdir -p "${parts_dir}"

  collect_one() {
    local idx="$1"
    local sample="${imgs[$idx]}"
    local part="${parts_dir}/part_${idx}.csv"
    local cmd=( "${runner}" "${so}" --shape "${shape_x}" --out-type "f32"
                --entry "${ent}" --warmup "${warmup}" --iters "${iters}" --quiet
                --image "${sample}" --image-preprocess-script "${image_preprocess_script}" )
    [[ "${runner_supports_quire}" -eq 1 ]] && cmd+=(--quire "${quire_mode}")
    [[ "${no_benchmark}" -eq 1 ]] && cmd+=(--no-benchmark)
    env POSIT_QOP_F32_MATH=on \
        "POSIT_COLLECT_LAYER_RANGES_FILE=${part}" \
        "${cmd[@]}" > /dev/null 2>&1 || true
  }

  if [[ "${cj}" -le 1 ]]; then
    for ((k=0; k<n_collect; ++k)); do collect_one "${k}"; done
  else
    for ((k=0; k<n_collect; ++k)); do
      collect_one "${k}" &
      while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${cj}" ]]; do wait -n; done
    done
    wait
  fi

  n_parts="$(find "${parts_dir}" -name 'part_*.csv' | wc -l | tr -d ' ')"
  echo "  collected ${n_parts} part files in ${parts_dir}"
  if [[ "${n_parts}" -eq 0 ]]; then echo "ERROR: no part files produced"; exit 4; fi

  echo "=== [clamp 2/3] MERGE -> ${clamp_csv} (percentile=${clamp_percentile} theta=${clamp_theta_mode}) ==="
  merge_args=( --parts-dir "${parts_dir}" --output "${clamp_csv}"
               --percentile "${clamp_percentile}" --margin "${clamp_margin}"
               --theta-mode "${clamp_theta_mode}" --fixed-theta "${clamp_fixed_theta}"
               --theta-target "${clamp_theta_target}"
               --gamma "${clamp_gamma}" )
  [[ "${clamp_symmetric}" == "1" ]] && merge_args+=(--symmetric)
  [[ "${clamp_version_a}" == "1" ]] && merge_args+=(--version-a)
  [[ -n "${clamp_exclude_keys}" ]] && merge_args+=(--exclude-keys "${clamp_exclude_keys}")
  python3 "${merge_py}" "${merge_args[@]}"
else
  echo "=== [clamp] CLAMP_SKIP_COLLECT=1, reuse ${clamp_csv} ==="
fi

if [[ ! -f "${clamp_csv}" ]]; then echo "ERROR: clamp csv missing: ${clamp_csv}"; exit 4; fi

echo "=== [clamp 3/3] EVAL with POSIT_OUTPUT_CLAMP_FILE=${clamp_csv} ==="
exec env "POSIT_OUTPUT_CLAMP_FILE=${clamp_csv}" bash "${eval_script}" "${args[@]}"
