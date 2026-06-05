#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  qalign_pipeline_model.sh --model-name NAME --out-dir DIR --input-txt FILE \
    [--build-script PATH] [--skip-build] \
    [--shape NxCxHxW|N,C,H,W] [--posit-formats CSV] [--collect-format FMT] \
    [--collect-csv PATH] [--calib-dir DIR] \
    [--collect-per-call N] [--collect-max-per-bucket N] \
    [--warmup N] [--iters N] [--entry SYMBOL] \
    [--no-benchmark|--with-benchmark] \
    [--apply-check on|off] [--apply-csv PATH] \
    [-- <extra args passed to build script>]

Purpose:
  1) (Optional) build model .so files
  2) collect qalign samples
  3) calibrate per-node/per-channel alpha
  4) (Optional) run A/B with and without POSIT_QALIGN_FILE

Defaults:
  posit-formats: p8e0,p8e1,p8e2
  collect-format: p8e0
  apply-check: on
USAGE_EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"

model_name=""
out_dir=""
input_txt=""
shape="1x3x224x224"
build_script=""
skip_build=0
posit_formats_csv="p8e0,p8e1,p8e2"
collect_format="p8e0"
collect_csv=""
calib_dir=""
collect_per_call=512
collect_max_per_bucket=1024
warmup=0
iters=1
entry=""
no_benchmark=1
apply_check="on"
apply_csv=""

build_extra_args=()

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
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

has_arg_in_array() {
  local needle="$1"
  shift
  local x
  for x in "$@"; do
    if [[ "${x}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

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
  --input-txt)
    input_txt="$2"
    shift 2
    ;;
  --shape)
    shape="$2"
    shift 2
    ;;
  --build-script)
    build_script="$2"
    shift 2
    ;;
  --skip-build)
    skip_build=1
    shift
    ;;
  --posit-formats)
    posit_formats_csv="$2"
    shift 2
    ;;
  --collect-format)
    collect_format="$2"
    shift 2
    ;;
  --collect-csv)
    collect_csv="$2"
    shift 2
    ;;
  --calib-dir)
    calib_dir="$2"
    shift 2
    ;;
  --collect-per-call)
    collect_per_call="$2"
    shift 2
    ;;
  --collect-max-per-bucket)
    collect_max_per_bucket="$2"
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
  --entry)
    entry="$2"
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
  --apply-check)
    apply_check="$2"
    shift 2
    ;;
  --apply-csv)
    apply_csv="$2"
    shift 2
    ;;
  --)
    shift
    while [[ $# -gt 0 ]]; do
      build_extra_args+=("$1")
      shift
    done
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

if [[ -z "${model_name}" ]]; then
  echo "ERROR: --model-name is required"
  exit 2
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="${src_root}/temp/${model_name}_qalign"
fi

if [[ -z "${input_txt}" ]]; then
  echo "ERROR: --input-txt is required"
  exit 2
fi
if [[ ! -f "${input_txt}" ]]; then
  echo "ERROR: input txt not found: ${input_txt}"
  exit 2
fi

shape_x="$(normalize_shape "${shape}")"
if [[ -z "${shape_x}" ]]; then
  echo "ERROR: invalid --shape: ${shape}"
  exit 2
fi

for n in "${collect_per_call}" "${collect_max_per_bucket}" "${warmup}" "${iters}"; do
  if ! is_int "${n}"; then
    echo "ERROR: collect/warmup/iters values must be integers >= 0"
    exit 2
  fi
done

if [[ "${iters}" -lt 1 ]]; then
  echo "ERROR: --iters must be >= 1"
  exit 2
fi

mkdir -p "${out_dir}"
if [[ -z "${collect_csv}" ]]; then
  collect_csv="${out_dir}/qalign_collect.csv"
fi
if [[ -z "${calib_dir}" ]]; then
  calib_dir="${out_dir}/qalign_calibration"
fi
mkdir -p "${calib_dir}"

if [[ "${skip_build}" -eq 0 ]]; then
  if [[ -z "${build_script}" ]]; then
    echo "ERROR: --build-script is required unless --skip-build"
    exit 2
  fi
  if [[ ! -f "${build_script}" ]]; then
    echo "ERROR: build script not found: ${build_script}"
    exit 2
  fi
  build_cmd=(bash "${build_script}" "${out_dir}")
  if ! has_arg_in_array "--posit-formats" "${build_extra_args[@]}"; then
    build_cmd+=(--posit-formats "${posit_formats_csv}")
  fi
  if [[ ${#build_extra_args[@]} -gt 0 ]]; then
    build_cmd+=("${build_extra_args[@]}")
  fi
  echo "[qalign] build command:"
  printf '  %q' "${build_cmd[@]}"
  echo
  "${build_cmd[@]}"
fi

runner="${out_dir}/run_time_sp"
so="${out_dir}/${model_name}-qdq-${collect_format}.so"
if [[ ! -x "${runner}" ]]; then
  echo "ERROR: runner not found: ${runner}"
  exit 3
fi
if [[ ! -f "${so}" ]]; then
  echo "ERROR: collect .so not found: ${so}"
  exit 3
fi

if [[ -z "${entry}" ]]; then
  entry="$(detect_entry "${so}")"
fi

echo "[qalign] collect samples"
collect_cmd=("${runner}" "${so}" "${input_txt}" \
  --shape "${shape_x}" \
  --out-type "${collect_format}" \
  --entry "${entry}" \
  --warmup "${warmup}" \
  --iters "${iters}" \
  --quiet)
if [[ "${no_benchmark}" -eq 1 ]]; then
  collect_cmd+=(--no-benchmark)
fi
POSIT_QALIGN_COLLECT_FILE="${collect_csv}" \
POSIT_QALIGN_COLLECT_PER_CALL="${collect_per_call}" \
POSIT_QALIGN_COLLECT_MAX_PER_BUCKET="${collect_max_per_bucket}" \
  "${collect_cmd[@]}"

if [[ ! -f "${collect_csv}" ]]; then
  echo "ERROR: collect CSV not generated: ${collect_csv}"
  echo "       runner/so may be built before qalign-collect runtime changes."
  echo "       try rebuild first (remove --skip-build), then run again."
  exit 4
fi

echo "[qalign] calibrate"
bash "${script_dir}/calibrate_qalign_pernode.sh" \
  --collect-csv "${collect_csv}" \
  --out-dir "${calib_dir}"

if [[ -z "${apply_csv}" ]]; then
  case "${collect_format}" in
  p8e0) apply_csv="${calib_dir}/qalign_p8e0.csv" ;;
  p8e1) apply_csv="${calib_dir}/qalign_p8e1.csv" ;;
  p8e2) apply_csv="${calib_dir}/qalign_p8e2.csv" ;;
  *) apply_csv="" ;;
  esac
fi

if [[ "${apply_check}" == "on" ]]; then
  if [[ -z "${apply_csv}" || ! -f "${apply_csv}" ]]; then
    echo "[qalign] apply-check skipped: apply csv is not available for format=${collect_format}"
  else
    echo "[qalign] apply A/B check"
    base_logits="${calib_dir}/logits_${collect_format}_baseline.csv"
    new_logits="${calib_dir}/logits_${collect_format}_qalign.csv"
    summary_txt="${calib_dir}/qalign_apply_ab_summary.txt"

    base_cmd=("${runner}" "${so}" "${input_txt}" \
      --shape "${shape_x}" \
      --out-type "${collect_format}" \
      --entry "${entry}" \
      --warmup 0 --iters 1 --quiet \
      --dump-logits "${base_logits}")
    if [[ "${no_benchmark}" -eq 1 ]]; then
      base_cmd+=(--no-benchmark)
    fi
    "${base_cmd[@]}"

    qalign_cmd=("${runner}" "${so}" "${input_txt}" \
      --shape "${shape_x}" \
      --out-type "${collect_format}" \
      --entry "${entry}" \
      --warmup 0 --iters 1 --quiet \
      --dump-logits "${new_logits}")
    if [[ "${no_benchmark}" -eq 1 ]]; then
      qalign_cmd+=(--no-benchmark)
    fi
    POSIT_QALIGN_FILE="${apply_csv}" "${qalign_cmd[@]}"

    python3 - "${base_logits}" "${new_logits}" "${summary_txt}" <<'PY'
import csv, math, sys
base, new, out = sys.argv[1], sys.argv[2], sys.argv[3]
def read(path):
    vals = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get("logit", "").strip().lower()
            if not s:
                continue
            vals.append(float("nan") if s == "nan" else float(s))
    return vals
b = read(base)
n = read(new)
m = min(len(b), len(n))
finite_abs = []
nonzero = 0
max_abs = -1.0
max_i = 0
for i in range(m):
    bi, ni = b[i], n[i]
    if math.isnan(bi) and math.isnan(ni):
        d = 0.0
    elif math.isnan(bi) or math.isnan(ni):
        d = float("inf")
    else:
        d = ni - bi
        finite_abs.append(abs(d))
        if d != 0.0:
            nonzero += 1
        if abs(d) > max_abs:
            max_abs = abs(d)
            max_i = i
mae = (sum(finite_abs) / len(finite_abs)) if finite_abs else float("nan")
with open(out, "w") as f:
    f.write("QALIGN apply A/B summary\n")
    f.write(f"baseline={base}\n")
    f.write(f"qalign={new}\n")
    f.write(f"count={m}\n")
    f.write(f"finite_mae={mae}\n")
    f.write(f"max_abs_delta={max_abs if max_abs >= 0 else 0.0} at_index={max_i}\n")
    f.write(f"nonzero_finite_delta={nonzero}\n")
print(out)
PY
  fi
fi

echo
echo "[qalign] done"
echo "  collect_csv: ${collect_csv}"
echo "  calibration: ${calib_dir}"
if [[ -n "${apply_csv}" ]]; then
  echo "  apply_csv:   ${apply_csv}"
fi
echo "  example apply:"
if [[ -n "${apply_csv}" ]]; then
  echo "    POSIT_QALIGN_FILE=${apply_csv} ${runner} ${so} ${input_txt} --shape ${shape_x} --out-type ${collect_format} --entry ${entry}"
else
  echo "    (no auto apply csv for format ${collect_format}; set --apply-csv manually)"
fi
