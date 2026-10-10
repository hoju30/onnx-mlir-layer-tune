#!/usr/bin/env python3
"""BERT-base SST-2: per-channel weights, criteria A / weight-B + activation-C / SQNR, MILP picks (for compiling).

Model and tokens from bert_prepare.py (--prep). The 74 weight-bearing nodes are 72 encoder MatMuls (weight [in, out],
output before the bias Add) and 2 Gemms (pooler, classifier; transB=1, output with bias). Int weights are quantized
per output channel (MatMul axis 1, Gemm axis 0, as the pass does with MIXEDP_WEIGHT_PER_CHANNEL=1); activations
use the pass's per-tensor a=. A channel of an activation is a hidden feature (last dimension).
  A     per-tensor Fisher, weight and activation terms
  BC    weight per-output-channel Fisher (B) + activation per-feature residual + coherent mean (C)
  SQNR  ||logits - logits_fp32||^2 on the calibration sentences with only that node quantized; computed with PyTorch
        hooks implementing the pass's arithmetic (spot-checked against the onnxruntime QDQ simulation), since each
        ORT run would copy the 440 MB model
MILP picks at budgets relative to all-int8 weight bytes and BOPs (MatMul BOPs x sequence length), checked in the
ORT simulation on the validation sentences; the picks are saved for the compiled evaluation.
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

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from bert_prepare import InputIdsOnly  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_criteria_cost_compare import RssSampler, match_modules  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant  # noqa: E402
from run_w8_per_channel_check import per_channel_int  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

AF = FORMATS[:-1]
CRITERIA = ("A", "BC", "SQNR")


def torch_weight_q(W, fmt):
    """Quantized torch weight [out, in]: int per output row (= the pass's per-channel axis for MatMul/Gemm)."""
    if fmt == "fp32":
        return W
    w = W.detach().numpy()
    q = per_channel_int(w, int(fmt[3:])) if fmt.startswith("int") else erm.np_fake_quant_weight(w, fmt, None)
    return torch.from_numpy(np.asarray(q, np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, default=HERE / "runs/bert_sst2_20261006")
    ap.add_argument("--assets", type=Path, default=HERE / "bert_assets")
    ap.add_argument("--budgets", default="0.6,0.7,0.8,1.0,1.2")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(b) for b in args.budgets.split(",")]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.prep / "model.onnx"))
    cx, cy = np.load(args.prep / "calib_tokens.npy"), np.load(args.prep / "calib_labels.npy")
    vx, vy = np.load(args.prep / "val_tokens.npy"), np.load(args.prep / "val_labels.npy")
    L = cx.shape[1]
    nodes = erm.supported_nodes(graph)
    weighted = [n for n in nodes if n.get("weight_const")]
    names = [n["name"] for n in weighted]
    op_of = {n["name"]: n["op"] for n in weighted}
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in weighted}
    ranges = erm.calibrate_ranges(graph, nodes, cx)

    from transformers import BertForSequenceClassification, BertTokenizer
    tok = BertTokenizer.from_pretrained(args.assets / "model")
    bert = BertForSequenceClassification.from_pretrained(args.assets / "model", attn_implementation="eager").eval()
    model = InputIdsOnly(bert, tok.pad_token_id).eval()
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    module_of, inits, by_name = match_modules(graph, weighted, linears)

    def y_of(n, out):   # the tensor the pass quantizes: MatMul output (before the bias Add), Gemm output (with bias)
        return out - module_of[n].bias if op_of[n] == "MatMul" else out

    def csum(t):
        return t.sum(dim=tuple(range(t.dim() - 1)))

    # --- Fisher tables ---
    cap, seen = {}, {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
             for n, m in module_of.items()]
    sampler = RssSampler().__enter__()
    g2w = {n: torch.zeros_like(module_of[n].weight, dtype=torch.float64) for n in names}
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]))
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += grads[3 * i].double() ** 2
                x, out = cap[n]
                for j, (s, t) in enumerate((("x", x), ("y", y_of(n, out)))):
                    g, a = grads[3 * i + 1 + j].double(), st[n][s]
                    lo, hi = seen.get((n, j), (float("inf"), float("-inf")))
                    seen[(n, j)] = (min(lo, float(t.min())), max(hi, float(t.max())))
                    a["g2"] = a["g2"] + csum(g ** 2)
                    a["gc"] = a["gc"] + csum(g) ** 2
                    a["n"] += t.numel() // t.shape[-1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + csum(d)
                        a["s2"][f] = a["s2"][f] + csum(d ** 2)
        cap.clear()
    CW, CA = {"A": {}, "B": {}}, {"A": {}, "C": {}}
    for n in names:
        W = module_of[n].weight
        G = g2w[n].numpy()                                     # [out, in]
        CW["A"][n], CW["B"][n], CA["A"][n], CA["C"][n] = {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}, {"fp32": 0.0}
        for f in AF:
            d2 = (torch_weight_q(W, f).double().numpy() - W.detach().double().numpy()) ** 2
            CW["A"][n][f] = float(G.mean() * d2.sum())
            CW["B"][n][f] = float((G.mean(1) * d2.sum(1)).sum())
            a_cost = c_cost = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                mu = a["s1"][f] / a["n"]
                a_cost += float(a["g2"].sum()) / (a["n"] * len(a["g2"])) * float(a["s2"][f].sum())
                c_cost += float((a["g2"] / a["n"] * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            CA["A"][n][f], CA["C"][n][f] = a_cost, c_cost
    sampler.__exit__(None, None, None)
    fisher_cost = sampler.report(f"{len(cy)} x (forward + backward), PyTorch CPU")
    for h in hooks:
        h.remove()
    mismatch = max(max(abs(seen[(n, j)][0] - ranges[tens[n][j]][0]), abs(seen[(n, j)][1] - ranges[tens[n][j]][1]))
                   / max(1e-6, ranges[tens[n][j]][1] - ranges[tens[n][j]][0]) for n in names for j in (0, 1))
    log(f"Fisher tables ready in {fisher_cost['seconds']:.1f} s; activation ranges match ONNX to {mismatch:.1e}")
    if mismatch > 1e-2:
        raise RuntimeError("PyTorch activations do not match the ONNX calibration ranges")
    del g2w

    # --- SQNR with PyTorch hooks ---
    current = {}

    def pre(mod, inp, n):
        return (act_fake_quant(inp[0], current[n][1], ranges[tens[n][0]]),)

    def post(mod, inp, out, n):
        a = current[n][1]
        if op_of[n] == "MatMul":
            return act_fake_quant(out - mod.bias, a, ranges[tens[n][1]]) + mod.bias
        return act_fake_quant(out, a, ranges[tens[n][1]])

    xc = torch.from_numpy(cx)
    with torch.no_grad():
        ref = model(xc).double()

    def sqnr_entry(n, w, a):
        m = module_of[n]
        orig = m.weight.data.clone()
        current[n] = (w, a)
        m.weight.data.copy_(torch_weight_q(orig, w))
        hs = [m.register_forward_pre_hook(lambda mod, inp, n=n: pre(mod, inp, n)),
              m.register_forward_hook(lambda mod, inp, out, n=n: post(mod, inp, out, n))]
        with torch.no_grad():
            z = model(xc).double()
        for h in hs:
            h.remove()
        m.weight.data.copy_(orig)
        return float(((z - ref) ** 2).sum()) if torch.isfinite(z).all() else None

    spath = args.out_dir / "sqnr_table.json"
    if spath.exists():
        saved = json.loads(spath.read_text())
        sq, sqnr_cost = saved["table"], saved["cost"]
    else:
        with RssSampler() as sampler:
            sq = {}
            for i, n in enumerate(names):
                sq[n] = {f"{w}|{a}": (0.0 if w == a == "fp32" else sqnr_entry(n, w, a)) for w in FORMATS for a in FORMATS}
                if (i + 1) % 10 == 0:
                    log(f"SQNR {i + 1}/{len(names)} nodes")
        sqnr_cost = sampler.report(f"{len(names) * 63} PyTorch forwards x {len(cy)} sentences")
        spath.write_text(json.dumps({"table": sq, "cost": sqnr_cost}) + "\n")
    log(f"SQNR table ready ({sqnr_cost['seconds']:.0f} s)")

    # spot check against the onnxruntime simulation (per-channel weights, each weight's own axis)
    axis = {}
    for n in names:
        node = by_name[n]
        axis[inits[node.input[1]].tobytes()] = 1 if op_of[n] == "MatMul" else 0
    original = erm.np_fake_quant_weight

    def pc_weight(w, fmt, oracle):
        if not fmt.startswith("int"):
            return original(w, fmt, oracle)
        ax = axis[np.asarray(w, np.float32).tobytes()]
        w = np.asarray(w, np.float32)
        return per_channel_int(w, int(fmt[3:])) if ax == 0 else per_channel_int(np.ascontiguousarray(w.T), int(fmt[3:])).T

    erm.np_fake_quant_weight = pc_weight
    ort_ref = erm.simulate_in_onnxruntime(graph, nodes, {}, ranges, cx, None).astype(np.float64)
    rng = np.random.default_rng(0)
    checks = []
    t_ort = []
    for n, w, a in [(names[i], ["int4", "int6", "int8", "f16"][j % 4], ["int4", "int6", "int8", "fp32"][(j + 1) % 4])
                    for j, i in enumerate(rng.choice(len(names), 6, replace=False))]:
        t1 = time.time()
        z = erm.simulate_in_onnxruntime(graph, nodes, {n: {"w": w, "a": a}}, ranges, cx, None)
        t_ort.append(time.time() - t1)
        o = float(((z - ort_ref) ** 2).sum())
        checks.append({"node": n, "w": w, "a": a, "torch": sq[n][f"{w}|{a}"], "ort": o})
    log("SQNR spot check torch vs ORT: " + "; ".join(f"{c['w']}/{c['a']} {c['torch']:.3g} vs {c['ort']:.3g}"
                                                     for c in checks)
        + f"; one ORT run {np.mean(t_ort):.1f} s -> full ORT table ~{np.mean(t_ort) * len(names) * 63 / 60:.0f} min")

    # --- MILP picks, ORT check on the validation sentences ---
    weight_elems = {n: int(module_of[n].weight.numel()) for n in names}
    flops = {n: weight_elems[n] * (L if op_of[n] == "MatMul" else 1) for n in names}
    int8_bytes, int8_bops = sum(weight_elems.values()), sum(64 * v for v in flops.values())

    def cost(c, n, w, a):
        if c == "SQNR":
            return sq[n][f"{w}|{a}"]
        return CW["B" if c == "BC" else "A"][n][w] + CA["A" if c == "A" else "C"][n][a]

    def evaluate(cfg):
        z = erm.simulate_in_onnxruntime(graph, nodes, cfg, ranges, vx, None)
        finite = np.isfinite(z).all(1)
        loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(vy),
                               reduction="none").numpy()
        return ((z.argmax(1) == vy) & finite).astype(float), loss

    fp32 = evaluate({})
    w8a8 = evaluate({n: {"w": "int8", "a": "int8"} for n in names})
    log(f"ORT simulation: FP32 acc {fp32[0].mean():.2%} CE {fp32[1].mean():.4f}; all W8A8 (per-channel) "
        f"acc {w8a8[0].mean():.2%} CE {w8a8[1].mean():.4f}")
    rows = []
    for f in budgets:
        for c in CRITERIA:
            cfg, _, status = milp_select(nodes, FORMATS, FORMATS, lambda n, w, a, c=c: cost(c, n, w, a), weight_elems,
                                         flops, f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1), (c, f, status)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
            correct, loss = evaluate(cfg)
            size, bops = config_cost(cfg, weight_elems, flops)
            rows.append({"budget": f, "criterion": c, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "sim_accuracy": float(correct.mean()), "sim_ce": float(loss.mean()),
                         "formats": cfg})
            log(f"f={f:.2f} {c:<5} ORT sim acc {correct.mean():6.2%}  CE {loss.mean():.4f}")
    erm.np_fake_quant_weight = original
    (args.out_dir / "results.json").write_text(json.dumps(
        {"model": "textattack/bert-base-uncased-SST-2", "seq_len": L, "nodes": len(names),
         "range_mismatch": mismatch, "cost": {"fisher_A_and_BC": fisher_cost, "sqnr_torch": sqnr_cost,
                                              "sqnr_ort_estimate_seconds": float(np.mean(t_ort) * len(names) * 63)},
         "sqnr_spot_check": checks, "sim_fp32": {"accuracy": float(fp32[0].mean()), "ce": float(fp32[1].mean())},
         "sim_w8a8": {"accuracy": float(w8a8[0].mean()), "ce": float(w8a8[1].mean())}, "rows": rows}, indent=2) + "\n")
    print("\nORT simulation, mean over budgets (accuracy / CE):")
    for c in CRITERIA:
        sel = [r for r in rows if r["criterion"] == c]
        print(f"  {c:<5} " + "  ".join(f"f={r['budget']:.2f} {r['sim_accuracy']:.2%}/{r['sim_ce']:.3f}" for r in sel)
              + f"   mean {np.mean([r['sim_accuracy'] for r in sel]):.2%}/{np.mean([r['sim_ce'] for r in sel]):.3f}")


if __name__ == "__main__":
    main()
