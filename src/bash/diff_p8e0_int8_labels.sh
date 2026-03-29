#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Usage:
  diff_p8e0_int8_labels.sh \
    --runner RUN_TIME_SP \
    --p8-so P8E0_SO \
    --int8-so INT8_BASELINE_SO \
    --input TXT \
    [--shape NxCxHxW] [--out-dir DIR] [--top-k N] \
    [--p8-entry SYMBOL] [--int8-entry SYMBOL]

Purpose:
  - Run one sample on p8e0 + int8/f32 baseline
  - Dump per-label logits
  - Print summary + Top-K largest label-wise diffs
USAGE_EOF
}

runner=""
p8_so=""
int8_so=""
input_txt=""
shape="1x3x224x224"
out_dir="./temp/logit_diff"
top_k=30
p8_entry="_mlir_ciface_main_graph"
int8_entry="_mlir_ciface_main_graph"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) runner="$2"; shift 2 ;;
    --p8-so) p8_so="$2"; shift 2 ;;
    --int8-so) int8_so="$2"; shift 2 ;;
    --input) input_txt="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --top-k) top_k="$2"; shift 2 ;;
    --p8-entry) p8_entry="$2"; shift 2 ;;
    --int8-entry) int8_entry="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -x "${runner}" ]] || { echo "ERROR: runner not executable: ${runner}"; exit 2; }
[[ -f "${p8_so}" ]] || { echo "ERROR: file not found: ${p8_so}"; exit 2; }
[[ -f "${int8_so}" ]] || { echo "ERROR: file not found: ${int8_so}"; exit 2; }
[[ -f "${input_txt}" ]] || { echo "ERROR: file not found: ${input_txt}"; exit 2; }

mkdir -p "${out_dir}"
p8_csv="${out_dir}/p8e0_logits.csv"
int8_csv="${out_dir}/int8_logits.csv"
diff_csv="${out_dir}/label_diff.csv"
top_csv="${out_dir}/label_diff_top${top_k}.csv"
summary_txt="${out_dir}/summary.txt"

echo "[run] p8e0 => ${p8_csv}"
"${runner}" "${p8_so}" "${input_txt}" \
  --shape "${shape}" \
  --posit p8e0 \
  --entry "${p8_entry}" \
  --warmup 0 --iters 1 --no-benchmark --quiet \
  --dump-logits "${p8_csv}" \
  > "${out_dir}/p8e0_run.log" 2>&1

echo "[run] int8/f32 baseline => ${int8_csv}"
"${runner}" "${int8_so}" "${input_txt}" \
  --shape "${shape}" \
  --out-type f32 \
  --entry "${int8_entry}" \
  --warmup 0 --iters 1 --no-benchmark --quiet \
  --dump-logits "${int8_csv}" \
  > "${out_dir}/int8_run.log" 2>&1

if [[ ! -s "${p8_csv}" || ! -s "${int8_csv}" ]]; then
  echo "ERROR: empty logits csv"
  exit 3
fi

awk -F, '
  NR==FNR {
    if (FNR==1) next;
    p8[$1]=$2;
    next;
  }
  FNR==1 { next; }
  {
    label=$1;
    b=$2+0.0;
    if (!(label in p8)) next;
    a=p8[label]+0.0;
    d=a-b;
    ad=(d<0?-d:d);
    n++;
    mae+=ad;
    rmse+=d*d;
    if (ad>maxabs) maxabs=ad;
    dot+=a*b;
    na2+=a*a;
    nb2+=b*b;
    printf "%s,%.17g,%.17g,%.17g,%.17g\n", label, a, b, d, ad;
  }
  END {
    if (n<=0) exit 10;
    mae/=n;
    rmse=sqrt(rmse/n);
    cosine=(na2>0 && nb2>0)?(dot/sqrt(na2*nb2)):0.0;
    printf "N=%d\nMAE=%.10g\nRMSE=%.10g\nMaxAbs=%.10g\nCosine=%.10g\n", n, mae, rmse, maxabs, cosine > "'"${summary_txt}"'";
  }
' "${p8_csv}" "${int8_csv}" > "${diff_csv}"

{
  echo "label,p8e0,int8,delta,abs_delta"
  sorted_tmp="${out_dir}/.label_diff_sorted.tmp"
  sort -t, -k5,5gr "${diff_csv}" > "${sorted_tmp}"
  head -n "${top_k}" "${sorted_tmp}"
  rm -f "${sorted_tmp}"
} > "${top_csv}"

echo "[done] summary: ${summary_txt}"
cat "${summary_txt}"
echo
echo "[done] top-${top_k} diff: ${top_csv}"
head -n $((top_k + 1)) "${top_csv}"
