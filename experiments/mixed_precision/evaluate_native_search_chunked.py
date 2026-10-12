#!/usr/bin/env python3
"""benchmark_native_search.py --stage evaluate with each configuration's held-out inputs split over several
processes, and all configurations of a model compiled and run at once.

Every input still runs alone (batch 1) through the same compiled .so in a fresh process, so the logits are the ones a
single process produces (check: --verify); only the wall time changes. benchmark_native_search.py is used unchanged,
as frozen in the run's source/ folder: this wrapper only replaces erm.run_so and the evaluation pool size.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark_native_search as bns  # noqa: E402
import eval_random_mixed_precision as erm  # noqa: E402

single_run_so = erm.run_so


def chunked_run_so(chunks):
    def run_so(tools, so, inputs_path, out_path, timeout):
        x = np.load(inputs_path, mmap_mode="r")
        parts = [p for p in np.array_split(np.arange(len(x)), chunks) if len(p)]
        folder = Path(out_path).parent

        def one(k, idx):
            ip, op = folder / f"inputs_part{k}.npy", folder / f"logits_part{k}.npy"
            np.save(ip, x[idx[0]:idx[-1] + 1])
            z, ev = single_run_so(tools, so, ip, op, timeout)
            ip.unlink()
            op.unlink()
            return z, ev

        t0 = time.time()
        with ThreadPoolExecutor(len(parts)) as pool:
            res = list(pool.map(lambda a: one(*a), enumerate(parts)))
        z = np.concatenate([r[0] for r in res])
        np.save(out_path, z)
        return z, {"seconds": sum(r[1]["seconds"] for r in res), "n": sum(r[1]["n"] for r in res),
                   "peak_rss_gb": max(r[1]["peak_rss_gb"] for r in res), "wall_seconds": time.time() - t0,
                   "processes": len(parts)}
    return run_so


def verify(so, inputs_path, n, chunks, scratch):
    """Logits of the first n inputs: one process vs `chunks` processes."""
    tools = erm.Tools()
    scratch.mkdir(parents=True, exist_ok=True)
    sub = scratch / "verify_inputs.npy"
    np.save(sub, np.load(inputs_path, mmap_mode="r")[:n])
    a, _ = single_run_so(tools, so, sub, scratch / "verify_single.npy", 3600)
    b, _ = chunked_run_so(chunks)(tools, so, sub, scratch / "verify_chunked.npy", 3600)
    same = a.shape == b.shape and np.array_equal(a, b)
    print(json.dumps({"inputs": n, "chunks": chunks, "bit_identical": bool(same)}))
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--chunks", type=int, default=3, help="processes per configuration's held-out run")
    ap.add_argument("--config-workers", type=int, default=4, help="configurations compiled / run at once")
    ap.add_argument("--verify", nargs=2, metavar=("SO", "INPUTS"), help="only compare single vs chunked logits")
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.verify:
        sys.exit(0 if verify(Path(args.verify[0]), Path(args.verify[1]), 3 * args.chunks, args.chunks,
                             args.out_dir) else 1)
    erm.run_so = chunked_run_so(args.chunks)
    bns.ThreadPoolExecutor = lambda max_workers: ThreadPoolExecutor(max_workers=args.config_workers)
    note = args.out_dir / "evaluation_execution.json"
    log = json.loads(note.read_text()) if note.exists() else {}
    log.update({m: {"chunks": args.chunks, "config_workers": args.config_workers,
                    "runner": "evaluate_native_search_chunked.py"} for m in args.models.split(",")})
    note.write_text(json.dumps(log, indent=2) + "\n")
    bns.evaluate(args)


if __name__ == "__main__":
    main()
