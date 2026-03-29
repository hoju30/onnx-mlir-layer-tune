#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  analyze_mobilenet_layer_drift.sh
    [--runner PATH]
    [--p8-so PATH]
    [--ref-so PATH]
    [--proxy-so PATH]
    [--input PATH]
    [--shape NxCxHxW]
    [--krnl PATH]
    [--out-dir PATH]
    [--trace-elems N]

Default reference:
  --ref-so uses nqdq-f32 (f32 baseline).

Proxy (optional):
  --proxy-so can be a higher-precision posit model (ex: p32e2) to compute
  per-layer sample-MAE using POSIT_TRACE samples.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_root="$(cd "${script_dir}/.." && pwd)"
runner="${src_root}/temp/mobilenet11_temp/run_time_sp"
p8_so="${src_root}/temp/mobilenet11_temp/mobilenetv2-12-qdq-p8e0.so"
ref_so="${src_root}/temp/mobilenet11_temp/mobilenetv2-12-nqdq-f32.so"
proxy_so="${src_root}/temp/mobilenet11_temp/mobilenetv2-12-qdq-p32e2.so"
input_txt="${src_root}/temp/imagenette_val_224/img_00000.txt"
shape="1x3x224x224"
krnl_mlir="${src_root}/temp/mobilenet11_temp/ir/mobilenetv2-12-qdq-p8e0.krnl.mlir"
out_dir="${src_root}/temp/layer_drift_mobilenet"
trace_elems=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner) runner="$2"; shift 2 ;;
    --p8-so) p8_so="$2"; shift 2 ;;
    --ref-so) ref_so="$2"; shift 2 ;;
    --proxy-so) proxy_so="$2"; shift 2 ;;
    --input) input_txt="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --krnl) krnl_mlir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --trace-elems) trace_elems="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: file not found: $1"
    exit 2
  fi
}

require_exec() {
  if [[ ! -x "$1" ]]; then
    echo "ERROR: executable not found: $1"
    exit 2
  fi
}

detect_entry() {
  local so="$1"
  local sym
  sym="$(nm -D "$so" 2>/dev/null | awk '$3 ~ /^_mlir_ciface_main_graph/ { print $3; exit }')"
  if [[ -z "$sym" ]]; then
    echo "_mlir_ciface_main_graph"
  else
    echo "$sym"
  fi
}

require_exec "$runner"
require_file "$p8_so"
require_file "$ref_so"
require_file "$input_txt"
if [[ -n "$proxy_so" ]]; then
  require_file "$proxy_so"
fi
if [[ -n "$krnl_mlir" ]]; then
  require_file "$krnl_mlir"
fi

mkdir -p "$out_dir"

p8_entry="$(detect_entry "$p8_so")"
ref_entry="$(detect_entry "$ref_so")"
proxy_entry=""
if [[ -n "$proxy_so" ]]; then
  proxy_entry="$(detect_entry "$proxy_so")"
fi

echo "[run] p8e0 trace"
POSIT_TRACE=1 POSIT_TRACE_MAX_ELEMS="$trace_elems" \
  "$runner" "$p8_so" "$input_txt" \
  --shape "$shape" --out-type f32 --entry "$p8_entry" \
  --warmup 0 --iters 1 --quiet --no-benchmark \
  --dump-logits "$out_dir/p8_logits.csv" \
  > "$out_dir/p8.stdout.log" 2> "$out_dir/p8.trace.log"

echo "[run] nqdq-f32 baseline"
"$runner" "$ref_so" "$input_txt" \
  --shape "$shape" --out-type f32 --entry "$ref_entry" \
  --warmup 0 --iters 1 --quiet --no-benchmark \
  --dump-logits "$out_dir/ref_logits.csv" \
  > "$out_dir/ref.stdout.log" 2> "$out_dir/ref.stderr.log"

if [[ -n "$proxy_so" ]]; then
  echo "[run] proxy trace"
  POSIT_TRACE=1 POSIT_TRACE_MAX_ELEMS="$trace_elems" \
    "$runner" "$proxy_so" "$input_txt" \
    --shape "$shape" --out-type f32 --entry "$proxy_entry" \
    --warmup 0 --iters 1 --quiet --no-benchmark \
    --dump-logits "$out_dir/proxy_logits.csv" \
    > "$out_dir/proxy.stdout.log" 2> "$out_dir/proxy.trace.log"
fi

python3 - "$out_dir" "$krnl_mlir" <<'PY'
import csv
import math
import os
import re
import sys

out_dir = sys.argv[1]
krnl_mlir = sys.argv[2]

def read_logits(path):
    vals = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            vals.append(float(row[1]))
    return vals

def topk_idx(vals, k):
    return [i for i, _ in sorted(enumerate(vals), key=lambda x: x[1], reverse=True)[:k]]

def compare(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return None
    mae = 0.0
    mse = 0.0
    maxabs = 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        d = a[i] - b[i]
        ad = abs(d)
        mae += ad
        mse += d * d
        maxabs = max(maxabs, ad)
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    mae /= n
    rmse = math.sqrt(mse / n)
    cosine = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0
    return (n, mae, rmse, maxabs, cosine)

ptrace_re = re.compile(
    r'^\[PTRACE\] op=([^\s]+) rank=(\d+) shape=\[([^\]]*)\] n=(\d+) min=([^\s]+) max=([^\s]+) mean=([^\s]+) absmax=([^\s]+) sample=\[([^\]]*)\]'
)

def parse_trace(path):
    rows = []
    with open(path) as f:
        idx = 0
        for line in f:
            m = ptrace_re.match(line.strip())
            if not m:
                continue
            idx += 1
            sample_txt = m.group(9).strip()
            sample = []
            if sample_txt:
                for x in sample_txt.split(","):
                    x = x.strip()
                    if x:
                        try:
                            sample.append(float(x))
                        except ValueError:
                            pass
            rows.append({
                "idx": idx,
                "op": m.group(1),
                "rank": int(m.group(2)),
                "shape": m.group(3),
                "n": int(m.group(4)),
                "min": float(m.group(5)),
                "max": float(m.group(6)),
                "mean": float(m.group(7)),
                "absmax": float(m.group(8)),
                "sample": sample,
            })
    return rows

def parse_krnl_calls(path):
    if not path or not os.path.isfile(path):
        return []
    call_re = re.compile(r'^\s*call @_mlir_ciface_posit_([a-z0-9_]+)\(')
    calls = []
    with open(path) as f:
        idx = 0
        for ln, line in enumerate(f, start=1):
            m = call_re.search(line)
            if not m:
                continue
            idx += 1
            calls.append({"idx": idx, "krnl_line": ln, "op": m.group(1)})
    return calls

def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

p8_logits = read_logits(os.path.join(out_dir, "p8_logits.csv"))
ref_logits = read_logits(os.path.join(out_dir, "ref_logits.csv"))
final_cmp = compare(p8_logits, ref_logits)
if final_cmp is None:
    raise SystemExit("No logits parsed.")

p8_top1 = topk_idx(p8_logits, 1)[0] if p8_logits else -1
ref_top1 = topk_idx(ref_logits, 1)[0] if ref_logits else -1
p8_top5 = topk_idx(p8_logits, 5)
ref_top5 = topk_idx(ref_logits, 5)
top5_overlap = len(set(p8_top5).intersection(ref_top5))

p8_trace = parse_trace(os.path.join(out_dir, "p8.trace.log"))
calls = parse_krnl_calls(krnl_mlir)

# Join p8 trace with krnl call order.
joined = []
for r in p8_trace:
    krnl_line = ""
    krnl_op = ""
    i = r["idx"] - 1
    if 0 <= i < len(calls):
        krnl_line = calls[i]["krnl_line"]
        krnl_op = calls[i]["op"]
    joined.append([
        r["idx"], r["op"], r["rank"], r["shape"], r["n"], r["min"], r["max"],
        r["mean"], r["absmax"], "|".join(str(x) for x in r["sample"]),
        krnl_line, krnl_op
    ])
write_csv(
    os.path.join(out_dir, "p8_layer_trace.csv"),
    joined,
    ["idx", "trace_op", "rank", "shape", "n", "min", "max", "mean", "absmax",
     "sample", "krnl_line", "krnl_op"],
)

proxy_path = os.path.join(out_dir, "proxy.trace.log")
layer_drift_rows = []
if os.path.isfile(proxy_path):
    proxy_trace = parse_trace(proxy_path)
    m = min(len(p8_trace), len(proxy_trace))
    for i in range(m):
        a = p8_trace[i]
        b = proxy_trace[i]
        sa = a["sample"]
        sb = b["sample"]
        k = min(len(sa), len(sb))
        sample_mae = ""
        sample_maxabs = ""
        if k > 0:
            ds = [abs(sa[j] - sb[j]) for j in range(k)]
            sample_mae = sum(ds) / k
            sample_maxabs = max(ds)
        krnl_line = ""
        krnl_op = ""
        if i < len(calls):
            krnl_line = calls[i]["krnl_line"]
            krnl_op = calls[i]["op"]
        layer_drift_rows.append([
            i + 1, a["op"], b["op"], a["shape"], b["shape"],
            sample_mae, sample_maxabs, a["mean"], b["mean"], a["absmax"], b["absmax"],
            krnl_line, krnl_op
        ])

    # Sort by sample_mae desc (empty last)
    def keyfn(r):
        try:
            return float(r[5])
        except Exception:
            return -1.0
    top = sorted(layer_drift_rows, key=keyfn, reverse=True)[:30]
    write_csv(
        os.path.join(out_dir, "layer_drift_p8_vs_proxy_top30.csv"),
        top,
        ["idx", "p8_op", "proxy_op", "p8_shape", "proxy_shape",
         "sample_mae", "sample_maxabs", "p8_mean", "proxy_mean",
         "p8_absmax", "proxy_absmax", "krnl_line", "krnl_op"],
    )
    write_csv(
        os.path.join(out_dir, "layer_drift_p8_vs_proxy_all.csv"),
        layer_drift_rows,
        ["idx", "p8_op", "proxy_op", "p8_shape", "proxy_shape",
         "sample_mae", "sample_maxabs", "p8_mean", "proxy_mean",
         "p8_absmax", "proxy_absmax", "krnl_line", "krnl_op"],
    )

summary = os.path.join(out_dir, "summary.txt")
with open(summary, "w") as f:
    f.write("Final logits compare: p8e0 vs nqdq-f32\n")
    f.write(f"N={final_cmp[0]}\n")
    f.write(f"MAE={final_cmp[1]:.10g}\n")
    f.write(f"RMSE={final_cmp[2]:.10g}\n")
    f.write(f"MaxAbs={final_cmp[3]:.10g}\n")
    f.write(f"Cosine={final_cmp[4]:.10g}\n")
    f.write(f"Top1(p8,ref)=({p8_top1},{ref_top1})\n")
    f.write(f"Top5Overlap={top5_overlap}\n")
    f.write(f"PTRACE_rows={len(p8_trace)}\n")
    f.write(f"PTRACE_csv={os.path.join(out_dir, 'p8_layer_trace.csv')}\n")
    if layer_drift_rows:
        f.write("Layer drift proxy: p8e0 vs proxy posit trace samples\n")
        f.write(f"Proxy_rows={len(layer_drift_rows)}\n")
        f.write(f"Top30={os.path.join(out_dir, 'layer_drift_p8_vs_proxy_top30.csv')}\n")

print(open(summary).read(), end="")
PY

echo
echo "[done] outputs:"
echo "  $out_dir/summary.txt"
echo "  $out_dir/p8_layer_trace.csv"
if [[ -f "$out_dir/layer_drift_p8_vs_proxy_top30.csv" ]]; then
  echo "  $out_dir/layer_drift_p8_vs_proxy_top30.csv"
fi
