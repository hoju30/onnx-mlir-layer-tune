#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"

collect_csv=""
out_dir="${src_root}/temp/qalign_calibration"
formats="p8e0,p8e1,p8e2"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --collect-csv)
    collect_csv="$2"
    shift 2
    ;;
  --out-dir)
    out_dir="$2"
    shift 2
    ;;
  --formats)
    formats="$2"
    shift 2
    ;;
  -h|--help)
    cat <<'EOF'
Usage:
  calibrate_qalign_pernode.sh --collect-csv PATH [--out-dir DIR] [--formats CSV]

Input CSV format:
  key,channel,seen,sample_count,samples
  (from runtime env POSIT_QALIGN_COLLECT_FILE)
  or new dual-reference format:
  key,channel,seen_orig,sample_count_orig,orig_samples,seen_dq,sample_count_dq,dq_samples

Outputs:
  qalign_<fmt>.csv for requested formats in --formats (default: p8e0,p8e1,p8e2),
  plus detail CSV + summary txt under --out-dir.

Paper-referenced calibration knobs (env vars):
  QALIGN_PAPER_SIGMA            default 2
  QALIGN_AUTO_SIGMA             off | global | per-format (default off)
  QALIGN_SIGMA_MIN              auto-search lower bound (default -4)
  QALIGN_SIGMA_MAX              auto-search upper bound (default 8)
  QALIGN_SCORE_MAE_WEIGHT       default 0.25
  QALIGN_SCORE_DOWNSTREAM_WEIGHT default 0.25
  QALIGN_SCORE_ALPHA_REG        default 0.0
  QALIGN_FORCE_PAPER_ALPHA      default off
  QALIGN_COMPAND_MODE           off | alps (default alps)
  QALIGN_COMPAND_THETA_MIN      default 0.25
  QALIGN_COMPAND_THETA_MAX      default 16
  QALIGN_COMPAND_THETA_STEPS    default 17 (log2 grid)
  QALIGN_COMPAND_GAMMA_TARGET   default 1.0
  QALIGN_COMPAND_GAMMA_PERCENTILE default 0.99
  QALIGN_COMPAND_GAMMA_TARGET_LIST optional CSV, e.g. 0.5,0.75,1.0,1.5,2.0,3.0
  QALIGN_COMPAND_GAMMA_PERCENTILE_LIST optional CSV, e.g. 0.90,0.95,0.99
  QALIGN_COMPAND_MIN_GAIN       default 0.0
  QALIGN_FORCE_COMPAND          default off (on = ALPS-only, no off-baseline compare)
  QALIGN_GP_REFERENCE_KEY       optional fixed reference key for rs/sc propagation
  QALIGN_GP_REFERENCE_CHANNEL   optional fixed reference channel for rs/sc propagation
  QALIGN_CALIB_JOBS             default auto (0 = hardware_concurrency)

Notes:
  - If both `orig_samples` and `dq_samples` exist for a bucket, calibration uses:
      input = dq_samples
      target = orig_samples
    so runtime semantics stay aligned with strict QDQ (`int8_x_dq -> posit`)
    while scoring against the original pre-Q values.
  - Default mode is hybrid: compare legacy MAE-optimal theta and
    paper(center/log2 + sigma)-guided candidates by mixed score.
  - Paper-style ALPS uses:
      y = asinh(theta*x) / gamma
      y_q = snap_posit(y)
      x' = sinh(gamma*y_q) / theta
    and calibration searches theta while gamma is selected by grid search
    over target/percentile candidates.
  - For backward compatibility, QALIGN_COMPAND_BETA_{MIN,MAX,STEPS}
    are still accepted as aliases for THETA range.
    Output qalign CSV keeps columns theta,gamma,mode where:
      theta = former alpha, gamma = former beta.
  - Detail CSV also emits exploratory generalized-posit suggestions:
      gp_rs_local / gp_sc_local
      gp_rs_from_ref / gp_sc_from_ref
    These are paper-inspired analysis outputs only; current runtime still uses
    standard posit + qalign compander, not generalized-posit MAC.
  - Set QALIGN_FORCE_PAPER_ALPHA=on to force paper alpha baseline
    (for A/B study; may hurt accuracy on some models).
EOF
    exit 0
    ;;
  *)
    echo "Unknown arg: $1"
    exit 2
    ;;
  esac
done

if [[ -z "${collect_csv}" ]]; then
  echo "ERROR: --collect-csv is required"
  exit 2
fi
if [[ ! -f "${collect_csv}" ]]; then
  echo "ERROR: collect CSV not found: ${collect_csv}"
  exit 2
fi

mkdir -p "${out_dir}"
cpp="${src_root}/temp/probe/qalign_calibrate_from_collect.cpp"
bin="${out_dir}/qalign_calibrate_from_collect"
uinc="${src_root}/.deps/universal/include/sw"

if [[ ! -f "${cpp}" ]]; then
  echo "ERROR: missing source: ${cpp}"
  exit 2
fi
if [[ ! -d "${uinc}" ]]; then
  echo "ERROR: missing universal include: ${uinc}"
  exit 2
fi

g++ -std=c++20 -O2 -pthread -I"${uinc}" "${cpp}" -o "${bin}"
QALIGN_OUTPUT_FORMATS="${formats}" "${bin}" "${collect_csv}" "${out_dir}"

echo "Done. Calibration files:"
echo "  ${out_dir}/qalign_p8e0.csv"
echo "  ${out_dir}/qalign_p8e1.csv"
echo "  ${out_dir}/qalign_p8e2.csv"
