#!/usr/bin/env python3
"""Parallel GPT-2 perplexity eval — method A: independent-window parallelism.

GPT-2 scoring is autoregressive (token N needs token N-1's KV cache), so tokens
*within* a window must stay sequential. But the corpus can be cut into
non-overlapping windows that are scored INDEPENDENTLY (each from a fresh KV
cache), and those windows run in parallel processes — the GPT-2 analogue of the
CNN runner's per-sample `--jobs`. NLL/top1 are aggregated across all windows.

posit `.so` is single-threaded software posit, so `--jobs N` is the main way to
use multiple cores for a posit run.

NOTE: independent-window ppl differs slightly from one continuous sequence (each
window's first tokens have shorter context, so ppl is usually a touch higher).
This is fine for posit-vs-f32 *relative* comparison; just keep `--window` and
`--max-tokens` identical across the models you compare. Use `--jobs 1` to get the
non-parallel (single continuous-per-window) number for the accuracy-gap check you
mentioned.

Examples
--------
# posit p8e1, 8 workers, 128-token windows, first 2000 corpus tokens
python run_gpt2_text_eval_parallel.py \
  --model build_gpt2_nqdq_p8e1/gpt2-hf-debug-nqdq-p8e1.so \
  --text-file eval_text/wikitext2_test.txt \
  --window 128 --jobs 8 --max-tokens 2000 --progress 5
"""
import argparse
import math
import multiprocessing as mp
import os
import time

from run_gpt2_text_eval import (
    GPT2BPETokenizer,
    _guess_format_tag,
    append_result_row,
    build_runner,
    score_token_ids,
)

# Per-worker globals (each process builds its own runner once via the pool
# initializer; the .so / ORT session is NOT shared or forked half-initialized).
_RUNNER = None


def _init_worker(model_path: str) -> None:
    global _RUNNER
    _RUNNER = build_runner(model_path)


def _score_window(task: tuple[int, list[int]]) -> tuple[int, dict[str, object]]:
    idx, token_ids = task
    res = score_token_ids(_RUNNER, token_ids, progress=0)
    return idx, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .onnx or .so model")
    ap.add_argument(
        "--tokenizer-dir",
        default="/home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community",
        help="Directory containing GPT-2 tokenizer files",
    )
    ap.add_argument("--text-file", required=True,
                    help="Corpus to score (e.g. WikiText-2 test)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="Cap total corpus tokens used (0 = all)")
    ap.add_argument("--window", type=int, default=128,
                    help="Tokens per independent window (2..1024)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="Parallel worker processes (posit is single-threaded, "
                         "so this is how you use multiple cores)")
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="Only score the first N windows (0 = all)")
    ap.add_argument("--progress", type=int, default=10,
                    help="Print running ppl/top1/throughput every N completed "
                         "windows (0 = off)")
    ap.add_argument("--results-log", default="",
                    help="Append a TSV result row here. Default: "
                         "<model dir>/gpt2_text_eval_results.tsv ('off' disables).")
    ap.add_argument("--tag", default="",
                    help="free-form label stored in the results row.")
    args = ap.parse_args()

    window = max(2, min(args.window, 1024))

    tokenizer = GPT2BPETokenizer(args.tokenizer_dir)
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    if args.max_tokens > 0:
        ids = ids[: args.max_tokens]

    # Non-overlapping windows; drop a trailing <2-token remainder.
    windows = [ids[i:i + window] for i in range(0, len(ids), window)]
    windows = [w for w in windows if len(w) >= 2]
    if args.limit_windows > 0:
        windows = windows[: args.limit_windows]
    if not windows:
        raise SystemExit("ERROR: no scorable windows (need >=2 tokens)")

    total_tokens = sum(len(w) for w in windows)
    jobs = max(1, args.jobs)
    print(f"model {args.model}", flush=True)
    print(f"jobs {jobs}  window {window}  num_windows {len(windows)}  "
          f"corpus_tokens {total_tokens}", flush=True)

    tasks = list(enumerate(windows))
    tot_nll = 0.0
    tot_exact = 0
    tot_cmp = 0
    done = 0
    t0 = time.time()

    # spawn avoids fork-after-native-load issues (ctypes posit .so / ORT threads).
    ctx = mp.get_context("spawn")
    if jobs == 1:
        # Single-process path (no pool) — for the non-parallel baseline.
        _init_worker(args.model)
        results = (_score_window(t) for t in tasks)
        for _, r in results:
            tot_nll += r["nll_sum"]; tot_exact += r["exact"]; tot_cmp += r["compared"]
            done += 1
            if args.progress and done % args.progress == 0:
                el = time.time() - t0
                print(f"[progress] {done}/{len(windows)} win  "
                      f"ppl={math.exp(tot_nll / tot_cmp):.3f}  "
                      f"top1={tot_exact / tot_cmp:.3f}  "
                      f"{tot_cmp / el:.2f} tok/s  elapsed={el:.0f}s", flush=True)
    else:
        with ctx.Pool(jobs, initializer=_init_worker,
                      initargs=(args.model,)) as pool:
            for _, r in pool.imap_unordered(_score_window, tasks):
                tot_nll += r["nll_sum"]; tot_exact += r["exact"]; tot_cmp += r["compared"]
                done += 1
                if args.progress and done % args.progress == 0:
                    el = time.time() - t0
                    print(f"[progress] {done}/{len(windows)} win  "
                          f"ppl={math.exp(tot_nll / tot_cmp):.3f}  "
                          f"top1={tot_exact / tot_cmp:.3f}  "
                          f"{tot_cmp / el:.2f} tok/s  elapsed={el:.0f}s", flush=True)

    elapsed = time.time() - t0
    avg_nll = tot_nll / tot_cmp
    print("mode score_parallel")
    print("model", args.model)
    print("jobs", jobs)
    print("window", window)
    print("num_windows", len(windows))
    print("num_predicted", tot_cmp)
    print("avg_nll", avg_nll)
    print("ppl", math.exp(avg_nll))
    print("next_token_top1_acc", tot_exact / tot_cmp)
    print("elapsed_sec", round(elapsed, 2))
    print("tokens_per_sec", round(tot_cmp / elapsed, 3) if elapsed > 0 else 0.0)

    # Record the run (ppl / top1 / timing) like the CNN runner and the single
    # runner do, into the same TSV so windowed and continuous runs are comparable.
    if args.results_log.lower() != "off":
        fmt = _guess_format_tag(args.model)
        results_log = args.results_log or os.path.join(
            os.path.dirname(os.path.abspath(args.model)),
            "gpt2_text_eval_results.tsv")
        append_result_row(results_log, {
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": args.tag or f"{fmt}-w{window}",
            "model": os.path.basename(args.model),
            "format": fmt,
            "mode": f"score_parallel(win={window},jobs={jobs},nwin={len(windows)})",
            "num_tokens": "",
            "num_predicted": tot_cmp,
            "ppl": f"{math.exp(avg_nll):.6f}",
            "avg_nll": f"{avg_nll:.6f}",
            "next_token_top1_acc": f"{tot_exact / tot_cmp:.6f}",
            "elapsed_sec": f"{elapsed:.2f}",
            "tokens_per_sec": f"{(tot_cmp / elapsed) if elapsed > 0 else 0.0:.3f}",
            "omp_threads": os.environ.get("POSIT_OMP_THREADS", ""),
            "max_tokens": args.max_tokens,
            "text_file": os.path.basename(args.text_file),
        })
        print("results_log", results_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
