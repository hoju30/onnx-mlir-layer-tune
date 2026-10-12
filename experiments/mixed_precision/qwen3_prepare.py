#!/usr/bin/env python3
"""Qwen3-0.6B (Qwen/Qwen3-0.6B, qwen3_assets/) for the mixed-precision experiments: tokens, ONNX, checks.

Same scheme as GPT-2 (run_gpt2_criteria.py): WikiText-2 raw, independent non-overlapping 128-token windows,
calibration from the validation split, evaluation from the test split; FP32 weights.

Writes into --out-dir:
  model.onnx (+ model.onnx.data)  wrapper taking input_ids [1, L] with a constant additive causal mask, returning
                        logits [1, L, vocab]; eager attention; TorchScript exporter, opset 14, constant folding
  validation_tokens.npy --calibration-windows windows, int64 [N, L]
  test_tokens.npy       --eval-windows windows, int64 [M, L]
  node_mapping.json     ONNX node -> parameter, for the 7 x 28 transformer projections (q/k/v/o, gate/up/down); the
                        embedding, its tied LM head and the norms stay FP32 (as GPT-2)
  prepare.json          FP32 agreement (PyTorch, ORT, onnx-mlir) and PPL on the evaluation windows (PyTorch)
"""
import argparse
import hashlib
import json
import sys
import time
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

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class CausalLM(torch.nn.Module):
    """input_ids -> logits, the causal mask a constant (eager attention adds a 4-D float mask as given)."""

    def __init__(self, model, seq):
        super().__init__()
        self.model = model
        mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)
        self.register_buffer("mask", mask[None, None], persistent=False)

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, attention_mask=self.mask, use_cache=False).logits


def fold_constants(model, max_bytes=1 << 26):
    """Every node whose inputs are all constants (initializers, Constant outputs, earlier folds) replaced by
    initializers, evaluated with ONNX's reference implementation at the model's own opsets. TorchScript leaves the
    RoPE tables and the Expand target shapes (ConstantOfShape -> Equal -> Where) as computations that onnx-mlir's
    shape inference does not see through, so every attention tensor would get a dynamic batch dimension (and the
    backward pass needs static shapes). Returns {op type: folded count}."""
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    values = {}

    def value(name):
        if name not in values:
            values[name] = numpy_helper.to_array(inits[name])
        return values[name]

    def evaluate(node, ins):
        m = helper.make_model(helper.make_graph(
            [node], "fold", [helper.make_tensor_value_info(x, onnx.TensorProto.UNDEFINED, None) for x in ins],
            [helper.make_tensor_value_info(y, onnx.TensorProto.UNDEFINED, None) for y in node.output]),
            opset_imports=list(model.opset_import))
        return [np.asarray(v) for v in ReferenceEvaluator(m).run(None, {x: value(x) for x in ins})]

    known, kept, folded = set(inits), [], {}
    for node in g.node:
        ins = [x for x in node.input if x]
        if node.op_type == "Constant":
            values[node.output[0]] = evaluate(node, [])[0]
            known.add(node.output[0])
            kept.append(node)
            continue
        if ins and all(x in known for x in ins):
            outs = evaluate(node, ins)
            if sum(v.nbytes for v in outs) <= max_bytes:
                for y, v in zip(node.output, outs):
                    values[y] = v
                    known.add(y)
                    g.initializer.append(numpy_helper.from_array(v, y))
                folded[node.op_type] = folded.get(node.op_type, 0) + 1
                continue
        kept.append(node)
    used = {x for n in kept for x in n.input} | {o.name for o in g.output}
    keep_inits = [i for i in g.initializer if i.name in used]
    del g.node[:]
    g.node.extend(kept)
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    return dict(sorted(folded.items()))


def token_nll(z, tokens):
    z = z[:, :-1].astype(np.float64)
    m = z.max(-1, keepdims=True)
    lse = np.log(np.exp(z - m).sum(-1)) + m[..., 0]
    return lse - np.take_along_axis(z, tokens[:, 1:, None], -1)[..., 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, default=HERE / "qwen3_assets")
    ap.add_argument("--sequence-length", type=int, default=128)
    ap.add_argument("--calibration-windows", type=int, default=10)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--compile-check", type=int, default=2, help="windows run through the onnx-mlir FP32 build")
    ap.add_argument("--tiny", action="store_true",
                    help="a random 2-layer Qwen3 of the same architecture (hidden 64, 4 heads / 2 KV heads, vocab "
                         "512, tokens taken mod 512), saved to <out-dir>/hf: checks the graph handling and backward "
                         "rules where float32 differences are not amplified by depth and outliers")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.assets / "model")
    if args.tiny:
        from transformers import Qwen3Config
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, tie_word_embeddings=True, rope_theta=1e6, rms_norm_eps=1e-6,
            attn_implementation="eager")).eval()
        model.save_pretrained(out / "hf")
    else:
        model = Qwen3ForCausalLM.from_pretrained(args.assets / "model", torch_dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    model.config.use_cache = False
    seq = args.sequence_length
    tokens = {}
    for split, count in (("validation", args.calibration_windows), ("test", args.eval_windows)):
        text = "\n\n".join(pq.read_table(HERE / "gpt2_assets/data" / f"{split}-00000-of-00001.parquet")["text"]
                           .to_pylist())
        ids = tok.encode(text, add_special_tokens=False)
        if args.tiny:
            ids = [i % model.config.vocab_size for i in ids]
        assert len(ids) >= count * seq
        tokens[split] = np.asarray(ids[:count * seq], np.int64).reshape(count, seq)
        np.save(out / f"{split}_tokens.npy", tokens[split])
    cx, ex = tokens["validation"], tokens["test"]

    wrapper = CausalLM(model, seq).eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(cx[:1]), use_cache=False).logits.numpy()       # HF's own causal mask
        wrapped = wrapper(torch.from_numpy(cx[:1])).numpy()
        test_logits = np.concatenate([wrapper(torch.from_numpy(ex[i:i + 1])).numpy() for i in range(len(ex))])
    np.testing.assert_allclose(wrapped, ref, rtol=1e-4, atol=1e-3)
    nll = token_nll(test_logits, ex)
    report = {"model": "tiny random Qwen3" if args.tiny else "Qwen/Qwen3-0.6B", "sequence_length": seq, "calibration_windows": len(cx),
              "eval_windows": len(ex), "wrapper_vs_hf_max_abs": float(np.abs(wrapped - ref).max()),
              "torch_fp32_test_ppl": float(np.exp(nll.mean())), "torch_fp32_test_nll": float(nll.mean())}
    log(f"PyTorch FP32: test PPL {report['torch_fp32_test_ppl']:.3f} on {len(ex)} windows; exporting")

    # Over protobuf's 2 GB: the exporter writes one external file per tensor; repack into model.onnx.data.
    tmp, path = out / "export_tmp", out / "model.onnx"
    tmp.mkdir(exist_ok=True)
    torch.onnx.export(wrapper, (torch.from_numpy(cx[:1]),), str(tmp / "model.onnx"), input_names=["input_ids"],
                      output_names=["logits"], opset_version=14, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(tmp / "model.onnx"))
    report["folded_constant_nodes"] = fold_constants(graph)
    log(f"folded constant subgraphs: {report['folded_constant_nodes']}")
    onnx.save_model(graph, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx.data", size_threshold=1024)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    graph = onnx.load(str(path))          # save_model turned the in-memory tensors into external references
    ops = {}
    for n in graph.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    report["onnx_ops"] = dict(sorted(ops.items()))
    report["onnx_bytes"] = path.stat().st_size + (out / "model.onnx.data").stat().st_size
    log(f"exported: {len(graph.graph.node)} nodes, {report['onnx_bytes'] / 1e9:.2f} GB, ops {report['onnx_ops']}")

    # ONNX node -> projection parameter. nn.Linear exports as MatMul(x, W^T) with W^T folded into a new initializer,
    # so match by value.
    params = {name: p.detach().numpy() for name, p in model.named_parameters()
              if name.startswith("model.layers.") and name.removesuffix(".weight").endswith(PROJECTIONS)}
    assert len(params) == 7 * model.config.num_hidden_layers, len(params)
    by_value = {}
    for name, w in params.items():
        wt = np.ascontiguousarray(w.T)
        by_value.setdefault((wt.shape, hashlib.sha256(wt.tobytes()).hexdigest()), []).append(name)
    inits = {i.name: i for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    nodes = erm.supported_nodes(graph)
    mapping = {}
    for n in nodes:
        if not n.get("weight_const"):
            continue
        a = onnx.numpy_helper.to_array(inits[by_name[n["name"]].input[1]])
        hit = by_value.get((a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))
        if hit:
            assert len(hit) == 1, hit
            mapping[n["name"]] = hit[0]
    assert sorted(mapping.values()) == sorted(params), (len(mapping), len(params))
    (out / "node_mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    report["projection_nodes"] = len(mapping)
    report["supported_nodes"] = len(nodes)
    report["weight_const_nodes"] = sum(1 for n in nodes if n.get("weight_const"))

    so_opts = ort.SessionOptions()
    so_opts.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(path), so_opts, providers=["CPUExecutionProvider"])
    k = args.compile_check
    ort_logits = np.concatenate([sess.run(None, {"input_ids": ex[i:i + 1]})[0] for i in range(k)])
    del sess
    report["ort_vs_torch_max_abs"] = float(np.abs(ort_logits - test_logits[:k]).max())
    report["ort_vs_torch_same_top1"] = float((ort_logits.argmax(-1) == test_logits[:k].argmax(-1)).mean())
    log(f"ORT vs PyTorch on {k} windows: max |dz| {report['ort_vs_torch_max_abs']:.2e}")

    tools = erm.Tools()
    work = out / "fp32_check"
    work.mkdir(exist_ok=True)
    _, report["onnx_mlir_import_s"] = erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / erm.SYMBOL_POSTFIX,
                                               "--mlir-elide-resource-strings-if-larger=1000000000",
                                               "--mlir-elide-elementsattrs-if-larger=1000000000", path])
    log(f"onnx-mlir import {report['onnx_mlir_import_s']:.0f}s; building FP32 .so")
    np.save(work / "inputs.npy", ex[:k])
    tb = time.time()
    # Krnl -O3 (tiled, vectorized MatMul), clang -O0: clang -O3 on this main_graph runs over 15 min, -O1 takes
    # 100 s, -O0 2 s, and all three run a window equally fast (24 s, the same logits).
    so, info = erm.build_so(tools, work, "fp32", work / f"{erm.SYMBOL_POSTFIX}.onnx.mlir", None, None,
                            opt="O0", krnl_opt="O3", constants_to_file=True)
    report["onnx_mlir_fp32_build_s"] = time.time() - tb
    report["onnx_mlir_fp32_build_steps"] = info["steps"]
    z, ev = erm.run_so(tools, so, work / "inputs.npy", work / "fp32.logits.npy", 3600)
    report["onnx_mlir_vs_ort_max_abs"] = float(np.abs(z - ort_logits).max())
    report["onnx_mlir_vs_torch_same_top1"] = float((z.argmax(-1) == test_logits[:k].argmax(-1)).mean())
    report["onnx_mlir_fp32_s_per_window"] = ev["seconds"] / ev["n"]
    report["onnx_mlir_fp32_peak_rss_gb"] = ev["peak_rss_gb"]
    log(f"onnx-mlir FP32: build {report['onnx_mlir_fp32_build_s']:.0f}s, {report['onnx_mlir_fp32_s_per_window']:.1f}"
        f" s/window, max |dz| vs ORT {report['onnx_mlir_vs_ort_max_abs']:.2e}")
    (out / "prepare.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("onnx_ops", "onnx_mlir_fp32_build_steps")},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
