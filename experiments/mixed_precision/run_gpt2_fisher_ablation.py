#!/usr/bin/env python3
"""GPT-2 small: which term of the Fisher W+A criterion (run_gpt2_fisher_milp.py) needs element-wise alignment?

Each term is either element-wise, sum (g * delta)^2, or layer-scalar, mean(g^2) * ||delta||^2. Both are in loss
units, so swapping one term never reintroduces a scale mismatch; the only thing removed is the alignment of
gradient and quantization error inside the tensor. (Dropping a term instead would make the MILP treat that
tensor's quantization as free.) Terms: weight W, projection input x, projection output y (a= quantizes x and y).
Variants are selected by MILP at the same budgets and evaluated in ORT QDQ simulation on the test windows.
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
from transformers import GPT2LMHeadModel  # noqa: E402

from eval_random_mixed_precision import np_fake_quant_weight, simulate_in_onnxruntime, supported_nodes  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_gpt2_fisher_milp import FORMATS, act_fake_quant, token_nll  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

# (weight, input, output): True = element-wise, False = layer-scalar
VARIANTS = {"full (all element-wise)": (True, True, True), "weight -> scalar": (False, True, True),
            "input -> scalar": (True, False, True), "output -> scalar": (True, True, False),
            "all scalar": (False, False, False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", type=Path, default=HERE / "runs/gpt2_shared_ablation_20260930")
    ap.add_argument("--budgets", default="0.8,1.0,1.25,1.5")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(f) for f in args.budgets.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    src = args.source_run
    graph = onnx.load(str(src / "model.onnx"))
    mapping = json.loads((src / "node_mapping.json").read_text())
    ranges = {t: tuple(v) for t, v in json.loads((src / "calibration.json").read_text()).items()}
    calib, test = np.load(src / "validation_tokens.npy"), np.load(src / "test_tokens.npy")
    nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    assert len(nodes) == 48
    seq = calib.shape[1]

    model = GPT2LMHeadModel.from_pretrained(HERE / "gpt2_assets" / "model", local_files_only=True,
                                            attn_implementation="eager").eval()
    model.config.use_cache = False
    mods, params = dict(model.named_modules()), dict(model.named_parameters())
    cap = {}

    def hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    for n, p in mapping.items():
        mods[p.removesuffix(".weight")].register_forward_hook(hook(n))
    g2w = {n: torch.zeros_like(params[p], dtype=torch.float64) for n, p in mapping.items()}
    # per activation side: element-wise sum (g*d)^2 per format, sum g^2, element count, sum d^2 per format
    act = {n: {s: {"elem": dict.fromkeys(FORMATS, 0.0), "g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0)}
               for s in ("x", "y")} for n in mapping}
    for k, ids in enumerate(calib):
        model.zero_grad(set_to_none=True)
        ids_t = torch.from_numpy(ids[None])
        loss = model(ids_t, labels=ids_t, use_cache=False).loss
        loss.backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                g2w[n] += params[mapping[n]].grad.double() ** 2
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    a, g = act[n][side], t.grad.double()
                    a["g2"] += float((g ** 2).sum())
                    a["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        d = (act_fake_quant(t, f, ranges[rng_name]) - t).double()
                        a["elem"][f] += float(((g * d) ** 2).sum())
                        a["d2"][f] += float((d ** 2).sum())
        cap.clear()
        log(f"calibration window {k + 1}/{len(calib)}: CE {float(loss.detach()):.4f}")

    # term tables: term[n][side][mode][fmt], mode in {"elem", "scalar"}
    term, elems, flops = {}, {}, {}
    for n, p in mapping.items():
        W = params[p].detach().numpy()
        mean_g2w = float(g2w[n].mean())
        tw = {"elem": {"fp32": 0.0}, "scalar": {"fp32": 0.0}}
        for f in FORMATS[:-1]:
            dW2 = (np_fake_quant_weight(W, f, None).astype(np.float64) - W.astype(np.float64)) ** 2
            tw["elem"][f] = float((torch.from_numpy(dW2) * g2w[n]).sum())
            tw["scalar"][f] = mean_g2w * float(dW2.sum())
        term[n] = {"w": tw}
        for side in ("x", "y"):
            a = act[n][side]
            mean_g2 = a["g2"] / a["count"]
            term[n][side] = {"elem": {"fp32": 0.0, **{f: a["elem"][f] for f in FORMATS[:-1]}},
                             "scalar": {"fp32": 0.0, **{f: mean_g2 * a["d2"][f] for f in FORMATS[:-1]}}}
        elems[n], flops[n] = W.size, W.size * seq
    del g2w, model
    log("element-wise and layer-scalar Fisher tables ready")

    def value_fn(flags):
        mw, mx, my = ("elem" if f else "scalar" for f in flags)
        return lambda n, w, a: term[n]["w"][mw][w] + term[n]["x"][mx][a] + term[n]["y"][my][a]

    ref_size, ref_bops = sum(elems.values()), sum(64 * v for v in flops.values())
    nll, rows = {}, []
    for i, f in enumerate(budgets):
        for label, flags in VARIANTS.items():
            cfg, obj, status = milp_select(nodes, FORMATS, FORMATS, value_fn(flags), elems, flops,
                                           f * ref_size, f * ref_bops)
            assert status == 0, (label, f, status)
            size, bops = config_cost(cfg, elems, flops)
            z = simulate_in_onnxruntime(graph, nodes, cfg, ranges, test, None)
            assert np.isfinite(z).all(), (label, f)
            key = (f, label)
            nll[key] = token_nll(z, test)
            m = float(nll[key].mean())
            rows.append({"budget": f, "variant": label, "elementwise": dict(zip(("w", "x", "y"), flags)),
                         "size": size / ref_size, "bops": bops / ref_bops, "nll": m, "perplexity": float(np.exp(m)),
                         "top1": float((z[:, :-1, :].argmax(-1) == test[:, 1:]).mean()),
                         "f16_activations": sum(v["a"] == "f16" for v in cfg.values()), "formats": cfg})
            log(f"f={f:.2f} {label:<24} PPL {np.exp(m):8.2f}")
            del z
    for r in rows:
        d = nll[(r["budget"], r["variant"])] - nll[(r["budget"], "full (all element-wise)")]
        r["dnll_vs_full"], r["dnll_vs_full_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "source_run": str(src), "budgets": budgets, "rows": rows}, indent=2) + "\n")

    print(f"\n{'f':>5} {'variant':<24} {'PPL':>8} {'top1':>7} {'dNLL vs full':>17} {'size':>5} {'BOPs':>5} {'f16 act':>8}")
    for r in rows:
        d = "" if r["variant"].startswith("full") else f"{r['dnll_vs_full']:+.4f}±{r['dnll_vs_full_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant']:<24} {r['perplexity']:8.2f} {r['top1']:7.2%} {d:>17} "
              f"{r['size']:5.2f} {r['bops']:5.2f} {r['f16_activations']:8d}")


if __name__ == "__main__":
    main()
