#!/usr/bin/env python3
"""Qwen3.5-0.8B: do BC's Gate / Up misjudgments make the automatic knee configurations worse?

The two knees of runs/qwen35_bc_20261015 (FFN-only, FFN + attention projections) fix, per layer, the weight formats
of gate_proj and up_proj (same shape: any redistribution of their weight bits with the same total keeps M and B).
Each such redistribution is a Gate / Up decision BC made. The test, pre-registered in plan.json:

  select    on fresh selection windows, the whole knee configuration quantized (PyTorch replay, GPU), every
            alternative Gate / Up split of every layer (int4..int8, same bit total, activations kept); a layer's
            best alternative is adopted when it beats the knee in >= 95% of 2000 window bootstraps
  evaluate  the corrected configuration (all adopted splits) against the knee on the final-test windows: PyTorch
            replay and, if it differs from the knee, native onnx-mlir compilation (as the main run's evaluate)

Stages (one --out-dir): plan, simcheck (the PyTorch replay of the knees and FP32 against the native final-test
results), select, evaluate (--native for the compiled check).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import qwen35_common as q35  # noqa: F401  (transformers 5 first)
import numpy as np
import torch

from run_qwen35_bc import jdump, jload

HERE = Path(__file__).resolve().parent
MAIN = HERE / "runs/qwen35_bc_20261015"
SEQ = 128
SELECT = ("validation", 24576, 30)       # unused so far (main 0-5119, 8192-14335; 16384-17663; 20480-24319)
VERSIONS = ("ffn", "ffn_attn")
INT = {"int4": 4, "int5": 5, "int6": 6, "int7": 7, "int8": 8}


def splits(g, u):
    """Alternative (gate, up) int weight formats with the same bit total."""
    s = INT[g] + INT[u]
    return [(f"int{a}", f"int{s - a}") for a in range(4, 9) if 4 <= s - a <= 8 and (f"int{a}", f"int{s - a}") != (g, u)]


def gate_up_nodes():
    m = jload(MAIN / "node_mapping.json")
    by = {(v["layer"], v["kind"]): n for n, v in m.items()}
    return {l: (by[(l, "gate_proj")], by[(l, "up_proj")]) for l in range(24)}


def stage_plan(args, out, log):
    from qwen3_swiglu_common import sha256
    split, first, n = SELECT
    toks = np.asarray(q35.wikitext_ids(split)[first:first + n * SEQ], np.int64).reshape(n, SEQ)
    np.save(out / "tokens_select.npy", toks)
    gu = gate_up_nodes()
    moves = {}
    for ver in VERSIONS:
        f = jload(MAIN / f"selected_formats_{ver}.json")
        moves[ver] = []
        for l, (g, u) in gu.items():
            if f[g]["w"] in INT and f[u]["w"] in INT:
                for a, b in splits(f[g]["w"], f[u]["w"]):
                    moves[ver].append({"layer": l, "gate": [a, f[g]["a"]], "up": [b, f[u]["a"]],
                                       "knee": [f"{f[g]['w']}/{f[g]['a']}", f"{f[u]['w']}/{f[u]['a']}"]})
    jdump(out / "moves.json", moves)
    jdump(out / "plan.json", {
        "question": "do BC's Gate / Up misjudgments make the automatic knee configurations worse?",
        "knees": {v: {"file": f"selected_formats_{v}.json", "sha256": sha256(MAIN / f"selected_formats_{v}.json")} for v in VERSIONS},
        "moves": "per layer, every alternative split of the gate / up int weight bits with the same total (int4..int8), "
                 "activations unchanged: equal M and B. " + ", ".join(f"{v}: {len(moves[v])}" for v in VERSIONS),
        "selection_windows": {"split": split, "tokens": [first, first + n * SEQ], "windows": n,
                              "sha256": sha256(out / "tokens_select.npy")},
        "replay": "PyTorch, whole configuration quantized as the pass does (per-output-channel int weights; int / posit "
                  "/ f16 activations of every quantized node, the main run's ranges), GPU FP32 TF32 off; checked against "
                  "the native final-test results first (simcheck)",
        "selection_rule": "per layer, the alternative with the lowest mean loss on the selection windows is adopted if it "
                          "is lower than the knee in >= 95% of 2000 window bootstraps; corrected = knee + all adopted",
        "evaluation": "final test (main run tokens_final_test.npy, 64 windows): corrected vs knee, per-window NLL "
                      "difference, paired bootstrap; native compile of a corrected configuration that differs",
        "material_if": "corrected - knee final-test dNLL has 95% upper bound < 0 AND |mean| >= 10% of the knee's dNLL vs "
                       "FP32 (native main run: ffn 0.00727, ffn_attn 0.02096)",
        "seeds": {"bootstrap": 0}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan: moves {', '.join(f'{v} {len(moves[v])}' for v in VERSIONS)}")


class Replay:
    """The adapter on the GPU with a whole configuration quantized; weights quantized once per (node, format)."""

    def __init__(self, device, threads):
        from qwen3_swiglu_common import Quantizers
        import copy
        self.q = Quantizers(MAIN, formats=("p6e1", "p8e1"))
        for f, lut in self.q.posit.items():
            bad, n = lut.verify()
            if bad:
                raise SystemExit(f"posit LUT {f}: {bad} of {n} differ")
        self.qa = copy.copy(self.q)
        self.qa.posit = {}
        for f, lut in self.q.posit.items():
            self.qa.posit[f] = copy.copy(lut)
            self.qa.posit[f].t_values, self.qa.posit[f].t_bounds = lut.t_values.to(device), lut.t_bounds.to(device)
        if device != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        self.device = device
        self.hf = q35.load_hf(torch.float32)
        self.ad = q35.TextExport(self.hf, SEQ).eval().to(device)
        self.mods = dict(self.hf.named_modules())
        self.mapping = jload(MAIN / "node_mapping.json")
        self.ranges = jload(MAIN / "ranges_by_point.json")
        self.orig = {n: self.mod(n).weight.data.cpu().clone() for n in self.mapping}
        self.cache = {}
        self.act = {}                       # node -> activation format (hooks read it)
        for n in self.mapping:
            m = self.mod(n)
            m.register_forward_pre_hook(lambda mod, i, n=n: (self.qa.act(i[0], self.act[n], self.ranges[n]["x"]),)
                                        if self.act.get(n, "fp32") != "fp32" else None)
            m.register_forward_hook(lambda mod, i, o, n=n: self.qa.act(o, self.act[n], self.ranges[n]["y"])
                                    if self.act.get(n, "fp32") != "fp32" else None)

    def mod(self, n):
        return self.mods[self.mapping[n]["param"].removesuffix(".weight")]

    def set(self, cfg):
        """cfg: {node: {"w", "a"}}; nodes not in cfg are FP32."""
        for n in self.mapping:
            w, a = (cfg[n]["w"], cfg[n]["a"]) if n in cfg else ("fp32", "fp32")
            w = w or "fp32"
            if (n, w) not in self.cache:
                self.cache[(n, w)] = self.orig[n] if w == "fp32" else self.q.weight(self.orig[n], w)
            self.mod(n).weight.data = self.cache[(n, w)].to(self.device)
            self.act[n] = a

    def ce(self, toks, batch=10):
        """Summed next-token CE per window."""
        x = torch.from_numpy(toks).to(self.device)
        r = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                z = self.ad(x[i:i + batch])
                r.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]), x[i:i + batch, 1:].reshape(-1),
                                                           reduction="none").view(z.shape[0], -1).double().sum(1))
        return torch.cat(r).cpu().numpy()


def stage_simcheck(args, out, log):
    """The replay of FP32 and both knees on the final-test windows against the native per-window NLL."""
    rp = Replay(args.device, args.threads)
    toks = np.load(MAIN / "tokens_final_test.npy")
    nat = jload(MAIN / "results.json")
    res = {}
    for ver in ("fp32", *VERSIONS):
        t0 = time.time()
        rp.set({} if ver == "fp32" else jload(MAIN / f"selected_formats_{ver}.json"))
        nll = rp.ce(toks) / (SEQ - 1)
        nv = np.array(nat[ver]["per_window_nll"])
        res[ver] = {"sim_nll": float(nll.mean()), "native_nll": float(nv.mean()), "per_window": nll.tolist(),
                    "max_abs_window_diff": float(np.abs(nll - nv).max()), "seconds": time.time() - t0}
        log(f"{ver}: sim {nll.mean():.5f} native {nv.mean():.5f}, max window |diff| {res[ver]['max_abs_window_diff']:.2e}")
    for ver in VERSIONS:
        ds = np.array(res[ver]["per_window"]) - np.array(res["fp32"]["per_window"])
        dn = np.array(nat[ver]["per_window_nll"]) - np.array(nat["fp32"]["per_window_nll"])
        res[ver]["dnll_vs_fp32"] = {"sim": float(ds.mean()), "native": float(dn.mean()),
                                    "per_window_corr": float(np.corrcoef(ds, dn)[0, 1])}
        log(f"{ver}: dNLL vs FP32 sim {ds.mean():+.5f} native {dn.mean():+.5f}, per-window corr {res[ver]['dnll_vs_fp32']['per_window_corr']:.3f}")
    jdump(out / f"simcheck_{args.device}.json", res)


def stage_select(args, out, log):
    from qwen3_swiglu_common import sha256
    plan = jload(out / "plan.json")
    if sha256(out / "tokens_select.npy") != plan["selection_windows"]["sha256"]:
        raise SystemExit("selection windows changed")
    table = jload(MAIN / "table.json")
    gu = gate_up_nodes()
    moves = jload(out / "moves.json")
    toks = np.load(out / "tokens_select.npy")
    rp = Replay(args.device, args.threads)
    rng = np.random.default_rng(plan["seeds"]["bootstrap"])
    boot = rng.integers(0, len(toks), (2000, len(toks)))
    res = {}
    for ver in VERSIONS:
        t0 = time.time()
        knee = jload(MAIN / f"selected_formats_{ver}.json")
        rp.set(knee)
        base = rp.ce(toks) / (SEQ - 1)
        rows = []
        for mv in moves[ver]:
            g, u = gu[mv["layer"]]
            cfg = {**knee, g: {"w": mv["gate"][0], "a": mv["gate"][1]}, u: {"w": mv["up"][0], "a": mv["up"][1]}}
            rp.set(cfg)
            d = rp.ce(toks) / (SEQ - 1) - base
            bc = sum(table["B"][n][cfg[n]["w"]] + table["C"][n][cfg[n]["a"]] - table["B"][n][knee[n]["w"]] - table["C"][n][knee[n]["a"]]
                     for n in (g, u))
            rows.append({**mv, "dnll": float(d.mean()), "better_frac": float((d[boot].mean(1) < 0).mean()),
                         "per_window": d.tolist(), "bc_delta": float(bc)})
        adopted = {}
        for l in sorted({r["layer"] for r in rows}):
            best = min((r for r in rows if r["layer"] == l), key=lambda r: r["dnll"])
            if best["dnll"] < 0 and best["better_frac"] >= 0.95:
                adopted[l] = best
        corrected = dict(knee)
        for l, r in adopted.items():
            g, u = gu[l]
            corrected[g] = {"w": r["gate"][0], "a": r["gate"][1]}
            corrected[u] = {"w": r["up"][0], "a": r["up"][1]}
        rp.set(corrected)
        dc = rp.ce(toks) / (SEQ - 1) - base
        jdump(out / f"corrected_formats_{ver}.json", corrected)
        res[ver] = {"knee_select_nll": float(base.mean()), "moves": rows,
                    "adopted": {str(l): {k: r[k] for k in ("gate", "up", "knee", "dnll", "better_frac", "bc_delta")}
                                for l, r in adopted.items()},
                    "corrected_minus_knee_select": float(dc.mean()), "seconds": time.time() - t0,
                    "corrected_sha256": sha256(out / f"corrected_formats_{ver}.json")}
        log(f"{ver}: {len(rows)} moves, {sum(r['dnll'] < 0 for r in rows)} lower, adopted {len(adopted)} "
            f"(layers {sorted(adopted)}), corrected - knee on selection {dc.mean():+.5f}, {time.time() - t0:.0f}s")
    res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    jdump(out / "selection.json", res)


def native_eval(out, name, fmts, chunks, log):
    """As run_qwen35_bc.stage_evaluate: per-channel int weights, Krnl -O3 / clang -O0, the main run's ranges."""
    import os
    import onnx
    from mp_auto import config_hash
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm, qwen35_spec
    from evaluate_native_search_chunked import chunked_run_so
    from run_qwen35_bc import native_import
    tools, _, base = native_import(MAIN, log)
    toks = np.load(MAIN / "tokens_final_test.npy")
    graph = onnx.load(str(MAIN / "model.onnx"), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in jload(MAIN / "ranges_by_point.json").items():
        for slot, r in slots.items():
            n = by_name[node]
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    spec = qwen35_spec(out / "unused", MAIN)
    ev = out / "eval"
    ev.mkdir(exist_ok=True)
    target = ev / name
    target.mkdir(exist_ok=True)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    posit_obj = erm.build_posit_runtime(tools, ev)
    t0 = time.time()
    cfg = {n["name"]: fmts.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes}
    env = erm.node_formats_env(cfg, spec.nodes, ranges)
    (target / "compile_env.json").write_text(json.dumps({"MIXEDP_NODE_FORMATS": env, "MIXEDP_WEIGHT_PER_CHANNEL": "1",
                                                         "config_hash": config_hash(fmts)}, indent=1) + "\n")
    so, info = erm.build_so(tools, target, name, base, env, posit_obj, opt="O0", krnl_opt="O3", constants_to_file=True,
                            bytecode=True)
    build_s = time.time() - t0
    if info["warnings"]:
        raise RuntimeError(f"{name}: compilation warnings {info['warnings']}")
    for p in target.iterdir():
        if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
            p.unlink()
    np.save(ev / "final_test.npy", toks)
    z, run = chunked_run_so(chunks)(tools, so, ev / "final_test.npy", target / "logits.npy", 36000)
    (target / "logits.npy").unlink()
    nll = np.stack([cross_entropy(z[i, :-1], toks[i, 1:]) for i in range(len(toks))])
    for p in (so, target / "model.constants.bin"):
        p.unlink(missing_ok=True)
    return {"nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())), "per_window_nll": nll.mean(1).tolist(),
            "all_logits_finite": bool(np.isfinite(z).all()), "build_s": build_s, "run_wall_s": run["wall_seconds"]}


def paired(d, seed=0):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def stage_evaluate(args, out, log):
    """Corrected vs knee on the final-test windows: the PyTorch replay on --device (run once per device) and, with
    --native, the native compilation; the amended materiality rule (plan.json amendments) once all three exist."""
    from qwen3_swiglu_common import sha256
    sel = jload(out / "selection.json")
    plan = jload(out / "plan.json")
    res = jload(out / "evaluation.json") if (out / "evaluation.json").exists() else {}
    toks = np.load(MAIN / "tokens_final_test.npy")
    nat = jload(MAIN / "results.json")
    knee_dnll = {"ffn": 0.00727, "ffn_attn": 0.02096}
    floor = {v: x["spread_of_mean_dnll_gpu_cpu_native"] for v, x in plan["amendments"][0]["implementation_noise_floor"].items()}
    rp = None
    for ver in VERSIONS:
        if sha256(out / f"corrected_formats_{ver}.json") != sel[ver]["corrected_sha256"]:
            raise SystemExit(f"corrected_formats_{ver}.json changed after the selection")
        corrected = jload(out / f"corrected_formats_{ver}.json")
        knee = jload(MAIN / f"selected_formats_{ver}.json")
        same = corrected == knee
        r = res.setdefault(ver, {"corrected_equals_knee": same})
        key = f"sim_{args.device}"
        if key not in r and not same:
            rp = rp or Replay(args.device, args.threads)
            rp.set(knee)
            k = rp.ce(toks) / (SEQ - 1)
            rp.set(corrected)
            c = rp.ce(toks) / (SEQ - 1)
            r[key] = {"knee_nll": float(k.mean()), "corrected_nll": float(c.mean()), "corrected_minus_knee": paired(c - k)}
            jdump(out / "evaluation.json", res)
            log(f"{ver} {key}: corrected - knee {r[key]['corrected_minus_knee']}")
        if args.native and not same and "native" not in r:
            r["native"] = native_eval(out, f"corrected_{ver}", corrected, args.chunks, log)
            d = np.array(r["native"]["per_window_nll"]) - np.array(nat[ver]["per_window_nll"])
            r["native"]["corrected_minus_knee"] = paired(d)
            jdump(out / "evaluation.json", res)
            log(f"{ver} native: corrected NLL {r['native']['nll']:.5f} vs knee {nat[ver]['nll']:.5f}, diff {r['native']['corrected_minus_knee']}")
        if same:
            r["material"] = {"verdict": False, "reason": "no layer adopted a different Gate / Up split"}
        elif all(k in r for k in ("sim_cuda", "sim_cpu", "native")):
            d = r["native"]["corrected_minus_knee"]
            signs = {np.sign(r[k]["corrected_minus_knee"]["mean"]) for k in ("sim_cuda", "sim_cpu", "native")}
            original = bool(d["ci95"][1] < 0 and abs(d["mean"]) >= 0.1 * knee_dnll[ver])
            r["material"] = {"original_rule": original, "above_noise_floor": bool(abs(d["mean"]) > floor[ver]),
                             "same_sign_gpu_cpu_native": len(signs) == 1, "floor": floor[ver],
                             "verdict": bool(original and abs(d["mean"]) > floor[ver] and len(signs) == 1)}
    jdump(out / "evaluation.json", res)
    log("material: " + ", ".join(f"{v} {res[v].get('material', {}).get('verdict', 'pending')}" for v in VERSIONS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "simcheck", "select", "evaluate"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--native", action="store_true", help="evaluate: also compile corrected configurations natively")
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    if args.stage == "plan":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "plan.json").exists():
        raise SystemExit("run --stage plan first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    import resource
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
    timing["runs"].append({"stage": args.stage, "device": args.device, "wall_s": time.time() - t0,
                           "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                           "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20})
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
