#!/usr/bin/env python3
"""The native gradient statistics of Qwen3-0.6B (--mixedp-collect-stats grads, incl. the Slice / Expand rules of
RoPE and grouped-query attention) against PyTorch autograd -- an offline reference only.

Loss on both sides: the summed next-token cross-entropy over positions 0..L-2 of a calibration window (the native
seed (softmax - onehot) x row mask). Per window, compared per array (error = max|native - ref| / max|ref|):
  every quantization point (input x, output y of the 196 projections): sum(g^2), sum(g) per channel, g the total
  gradient of the tensor (q/k/v share their input tensor, so its gradient sums the three)
  every projection weight: sum(dW^2) per output channel
Qwen3 has no ReLU kinks (SiLU), so every window should agree to float32 accumulation error. Exits 1 if the median
error over arrays exceeds --tol-median or any array exceeds --tol-max.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from qwen3_prepare import CausalLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare-dir", type=Path, default=HERE / "runs/qwen3_prepare_20261011")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--tol-median", type=float, default=1e-3)
    ap.add_argument("--tol-max", type=float, default=5e-2)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = args.prepare_dir
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tools = erm.Tools()
    mapping = json.loads((src / "node_mapping.json").read_text())
    calib = np.load(src / "validation_tokens.npy")[:args.windows]
    model_dir = src / "hf" if (src / "hf").exists() else HERE / "qwen3_assets/model"   # qwen3_prepare.py --tiny
    seq, vocab = calib.shape[1], json.loads((model_dir / "config.json").read_text())["vocab_size"]

    # --- native: the grads graph of the projections, (Krnl -O3, clang -O0) as run_native_bc_pipeline's MatMul models
    inst, layout_path, so = out / "grads.onnx.mlir", out / "grads_layout.json", out / "grads.so"
    info = {"steps": "reused"}
    if not so.exists():                            # an earlier run's build is reused (same model, same pass)
        erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", out / erm.SYMBOL_POSTFIX,
                 "--mlir-elide-resource-strings-if-larger=1000000000",
                 "--mlir-elide-elementsattrs-if-larger=1000000000", src / "model.onnx"])
        erm.run([tools.opt, out / f"{erm.SYMBOL_POSTFIX}.onnx.mlir", "--shape-inference", "--mixedp-collect-stats",
                 "-o", inst], env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": ",".join(mapping),
                                   "MIXEDP_STATS_LAYOUT": str(layout_path)})
        tb = time.time()
        so, info = erm.build_so(tools, out, "grads", inst, None, None, opt="O0", krnl_opt="O3",
                                constants_to_file=True)
        for p in out.iterdir():
            if p.suffix in (".mlir", ".ll", ".bc"):
                p.unlink()
        log(f"grads .so built in {time.time() - tb:.0f}s ({ {k: round(v) for k, v in info['steps'].items()} })")
    layout = json.loads(layout_path.read_text())
    sys.path.insert(0, str(tools.lib))
    from PyRuntime import OMExecutionSession
    sess = OMExecutionSession(str(so), erm.SYMBOL_POSTFIX)
    native, run_s = [], []
    cache = out / f"native_{len(calib)}.npy"
    for ids in ([] if cache.exists() else calib):
        onehot = np.zeros((1, seq, vocab), np.float32)
        onehot[0, np.arange(seq - 1), ids[1:]] = 1.0
        tr = time.time()
        native.append(np.asarray(sess.run([np.ascontiguousarray(ids[None]), onehot])[layout["result_index"]],
                                 np.float64))
        run_s.append(time.time() - tr)
    if cache.exists():
        native = list(np.load(cache))
    else:
        np.save(cache, np.stack(native))
        log(f"native: {np.mean(run_s):.1f}s per window")

    # --- PyTorch reference
    from transformers import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                             attn_implementation="eager").eval()
    wrapper = CausalLM(model, seq).eval()
    modules = dict(model.named_modules())
    module_of = {node: modules[param.removesuffix(".weight")] for node, param in mapping.items()}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in module_of.values():
        m.weight.requires_grad_(True)
    model.model.embed_tokens.weight.requires_grad_(True)   # so layer 0's projection inputs carry gradients
    ref = []
    for ids in calib:
        cap = {}

        def hook(m, inp, outp):
            for t in (inp[0], outp):
                t.retain_grad()
            cap[m] = (inp[0], outp)

        hs = [m.register_forward_hook(hook) for m in module_of.values()]
        model.zero_grad(set_to_none=True)
        z = wrapper(torch.from_numpy(ids[None]))
        loss = torch.nn.functional.cross_entropy(z[0, :-1], torch.from_numpy(ids[1:]), reduction="sum")
        loss.backward()
        for h in hs:
            h.remove()
        r = {}
        for node, m in module_of.items():
            x, y = cap[m]
            for slot, t in (("x", x), ("y", y)):
                g = t.grad[0].double().numpy()                 # [L, channels]
                r[(node, slot)] = ((g ** 2).sum(0), g.sum(0))
            r[(node, "w")] = (m.weight.grad.double().numpy() ** 2).sum(1)   # torch [out, in]: per output channel
        ref.append(r)
    log("PyTorch reference done")

    # --- compare
    taps = {t["tap"]: t for t in layout["taps"]}
    errors = []                                    # (window, key, field, error)

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    for k, (v, r) in enumerate(zip(native, ref)):
        for p in layout["points"]:
            t = taps[p["tap"]]
            o, c = t["offset"], t["channels"]
            g2, gs = r[(p["node"], p["slot"])]
            errors.append((k, p["node"], p["slot"], "sum_g2", rel(v[o:o + c], g2)))
            errors.append((k, p["node"], p["slot"], "sum_g", rel(v[o + c:o + 2 * c], gs)))
        for w in layout["weights"]:
            errors.append((k, w["node"], "w", "sum_dW2", rel(v[w["offset"]:w["offset"] + w["channels"]],
                                                              r[(w["node"], "w")])))
    e = np.array([x[-1] for x in errors])
    worst = sorted(errors, key=lambda x: -x[-1] if np.isfinite(x[-1]) else -np.inf)[:5]
    report = {"windows": len(calib), "arrays": len(errors), "median_rel": float(np.median(e)),
              "p99_rel": float(np.percentile(e, 99)), "max_rel": float(np.max(e)),
              "by_field": {f: float(np.median([x[-1] for x in errors if x[3] == f]))
                           for f in ("sum_g2", "sum_g", "sum_dW2")},
              "worst": [list(x) for x in worst], "native_run_s_per_window": float(np.mean(run_s)) if run_s else None,
              "grads_build_steps": info["steps"]}
    ok = report["median_rel"] <= args.tol_median and report["max_rel"] <= args.tol_max
    report["pass"] = bool(ok)
    (out / "grad_check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "grads_build_steps"}, indent=1), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
