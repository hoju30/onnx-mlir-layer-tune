#!/usr/bin/env python3
"""
Orchestrator for ml_predictor_dataset.md: exports/downloads all 22 model
instances (model_dataset_spec.py) into the model_dataset/<arch>/<variant>/
directory tree the spec's "Model Dataset Structure" section describes, and
writes a top-level manifest.json summarizing what landed where. This is pure
data acquisition -- no posit compilation, op-feature extraction, or accuracy
evaluation happens here; that is a separate later stage once these ONNX
files exist (see claude.md sections 5-20 and experiments/common/).

Resumable: an entry whose model.onnx + metadata.json already exist is
skipped unless --force is passed, so a partial run (e.g. killed mid-download)
can just be re-invoked.

Usage:
  python build_model_dataset.py --out-dir model_dataset
  python build_model_dataset.py --out-dir model_dataset --arch resnet18   # just one architecture
  python build_model_dataset.py --out-dir model_dataset --force           # re-export everything
"""
import argparse
import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_dataset_spec import MODELS, out_path


def already_done(onnx_path):
    meta_path = onnx_path.replace("model.onnx", "metadata.json")
    return os.path.exists(onnx_path) and os.path.exists(meta_path)


def run_entry(entry, onnx_path):
    if entry["source"] == "torchvision":
        cmd = [sys.executable, os.path.join(HERE, "export_torchvision_model.py"),
               "--model", entry["torchvision_model"], "--weights", entry["torchvision_weights"],
               "--out", onnx_path]
    elif entry["source"] == "timm":
        cmd = [sys.executable, os.path.join(HERE, "export_timm_model.py"),
               "--model", entry["timm_model"], "--out", onnx_path]
    elif entry["source"] == "onnx_model_zoo":
        cmd = [sys.executable, os.path.join(HERE, "download_onnx_zoo_model.py"),
               "--url", entry["zoo_url"], "--out", onnx_path]
    else:
        raise ValueError(f"unknown source {entry['source']!r}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "model_dataset"))
    ap.add_argument("--arch", default=None, help="only build this architecture (e.g. resnet18)")
    ap.add_argument("--force", action="store_true", help="re-export/download even if already present")
    args = ap.parse_args()

    entries = MODELS if args.arch is None else [m for m in MODELS if m["arch"] == args.arch]
    if not entries:
        print(f"no entries match --arch {args.arch!r}", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, entry in enumerate(entries, 1):
        onnx_path = out_path(args.out_dir, entry)
        tag = f"{entry['arch']}/{entry['variant']}"
        print(f"\n=== [{i}/{len(entries)}] {tag} ({entry['source']}) ===", flush=True)

        if not args.force and already_done(onnx_path):
            print(f"  already present at {onnx_path}, skipping (--force to redo)", flush=True)
            results.append({"arch": entry["arch"], "variant": entry["variant"], "status": "skipped",
                            "path": onnx_path})
            continue

        try:
            run_entry(entry, onnx_path)
            results.append({"arch": entry["arch"], "variant": entry["variant"], "status": "ok",
                            "path": onnx_path, "source": entry["source"], "note": entry.get("note")})
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"arch": entry["arch"], "variant": entry["variant"], "status": "failed",
                            "path": onnx_path, "error": str(e)})

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n=== {ok}/{len(results)} model instances present, {len(failed)} failed ===")
    for r in failed:
        print(f"  FAILED {r['arch']}/{r['variant']}: {r['error']}")
    print(f"wrote {manifest_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
