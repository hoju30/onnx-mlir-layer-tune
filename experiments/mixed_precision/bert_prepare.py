#!/usr/bin/env python3
"""BERT-base SST-2 (textattack/bert-base-uncased-SST-2) for the mixed-precision experiments: tokens, ONNX, checks.

Writes into --out-dir:
  model.onnx            wrapper taking only input_ids [1, L] (attention mask = input_ids != [PAD], token types 0),
                        returning logits [1, 2]; TorchScript exporter, opset 14, constant folding (as GPT-2)
  calib_tokens.npy      --num-calibration GLUE SST-2 train sentences (seeded), int64 [N, L], with calib_labels.npy
  val_tokens.npy        the 872 GLUE SST-2 validation sentences, int64 [872, L], with val_labels.npy
  prepare.json          sequence length, truncation count, weighted nodes, FP32 accuracies (PyTorch, ORT, onnx-mlir
                        on --compile-check sentences) and an onnx-mlir build of all-W8A8 per-channel weights
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402


class InputIdsOnly(torch.nn.Module):
    def __init__(self, model, pad_id):
        super().__init__()
        self.model, self.pad_id = model, pad_id

    def forward(self, input_ids):
        mask = (input_ids != self.pad_id).long()
        return self.model(input_ids=input_ids, attention_mask=mask,
                          token_type_ids=torch.zeros_like(input_ids)).logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, default=HERE / "bert_assets")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-calibration", type=int, default=10)
    ap.add_argument("--compile-check", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import BertForSequenceClassification, BertTokenizer
    tok = BertTokenizer.from_pretrained(args.assets / "model")
    model = BertForSequenceClassification.from_pretrained(args.assets / "model", attn_implementation="eager").eval()
    val = pq.read_table(args.assets / "data/sst2/validation-00000-of-00001.parquet").to_pydict()
    train = pq.read_table(args.assets / "data/sst2/train-00000-of-00001.parquet").to_pydict()
    pick = np.random.default_rng(0).choice(len(train["sentence"]), args.num_calibration, replace=False)
    cal = {k: [train[k][i] for i in pick] for k in ("sentence", "label")}
    lengths = [len(tok(s)["input_ids"]) for s in val["sentence"]]
    L = args.seq_len

    def encode(sentences):
        return np.array(tok(list(sentences), padding="max_length", truncation=True, max_length=L)["input_ids"], np.int64)

    vx, vy = encode(val["sentence"]), np.array(val["label"], np.int64)
    cx, cy = encode(cal["sentence"]), np.array(cal["label"], np.int64)
    for k, v in {"val_tokens": vx, "val_labels": vy, "calib_tokens": cx, "calib_labels": cy}.items():
        np.save(args.out_dir / f"{k}.npy", v)
    wrapper = InputIdsOnly(model, tok.pad_token_id).eval()
    with torch.no_grad():
        torch_logits = torch.cat([wrapper(torch.from_numpy(vx[i:i + 64])) for i in range(0, len(vx), 64)]).numpy()
    path = args.out_dir / "model.onnx"
    torch.onnx.export(wrapper, (torch.from_numpy(vx[:1]),), str(path), input_names=["input_ids"],
                      output_names=["logits"], opset_version=14, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_logits = np.concatenate([sess.run(None, {"input_ids": vx[i:i + 1]})[0] for i in range(len(vx))])
    nodes = erm.supported_nodes(graph)
    weighted = [n for n in nodes if n.get("weight_const")]
    ops = {}
    for n in weighted:
        ops[n["op"]] = ops.get(n["op"], 0) + 1
    report = {"seq_len": L, "max_tokens_validation": int(max(lengths)),
              "truncated_validation": int(sum(l > L for l in lengths)), "validation": len(vy), "calibration": len(cy),
              "weighted_nodes": len(weighted), "weighted_ops": ops, "supported_nodes": len(nodes),
              "torch_fp32_accuracy": float((torch_logits.argmax(1) == vy).mean()),
              "ort_fp32_accuracy": float((ort_logits.argmax(1) == vy).mean()),
              "ort_vs_torch_max_abs": float(np.abs(ort_logits - torch_logits).max())}
    print(json.dumps(report, indent=1), flush=True)

    # --- onnx-mlir: FP32 and all-W8A8 (per-channel weights) on a few sentences ---
    tools = erm.Tools()
    work = args.out_dir / "work"
    work.mkdir(exist_ok=True)
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
             "--mlir-elide-resource-strings-if-larger=1000000000",
             "--mlir-elide-elementsattrs-if-larger=1000000000", path])
    base = work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
    xs = vx[:args.compile_check]
    np.save(work / "inputs.npy", xs)
    ranges = erm.calibrate_ranges(graph, nodes, cx)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    w8a8 = {n["name"]: ({"w": "int8", "a": "int8"} if n.get("weight_const") else {"w": None, "a": "fp32"}) for n in nodes}
    for name, cfg in (("fp32", None), ("w8a8_per_channel", w8a8)):
        env = None if cfg is None else erm.node_formats_env(cfg, nodes, ranges)
        so, info = erm.build_so(tools, work, name, base, env, None)
        logits, ev = erm.run_so(tools, so, work / "inputs.npy", work / f"{name}.logits.npy", 3600)
        report[f"onnx_mlir_{name}"] = {
            "accuracy": float((logits.argmax(1) == vy[:len(xs)]).mean()),
            "max_abs_vs_ort_fp32": float(np.abs(logits - ort_logits[:len(xs)]).max()),
            "seconds_per_sentence": ev["seconds"] / ev["n"], "warnings": len(info["warnings"]),
            "warning_examples": info["warnings"][:3]}
        print(name, json.dumps(report[f"onnx_mlir_{name}"]), flush=True)
    (args.out_dir / "prepare.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
