#!/usr/bin/env python3
"""OLMoE fixed route: the MILP knee (R 0.707, = runs/olmoe_multipoint_20261015/configs/bc_p125.json) against the knee
that LP relaxation + rounding + repair selects (runs/solver_regression_20261016/configs/lp_knee.json), with FP32 as
the baseline: measured whole-model quantization damage, and each configuration's actual weight storage and BOPs.

Nothing is selected here; both configurations are read (hash-checked) from the runs that made them. Loss: per-window
summed next-token CE with every layer's cached FP32 route (run_olmoe_multipoint.replay, GPU FP32, TF32 off) on two
window sets that both already have routes:
  A  the multipoint evaluation windows (validation + test from token 131072, 32 + 32)
  B  the solver-regression evaluation windows (validation + test from token 163840, 32 + 32); replayed again here and
     compared with that run's truth.json (determinism check)
Resources:
  searched scope (the 3051 expert projections the MILP assigns): M = sum elements x weight bits / 8 and B = sum
     calibration routed rows x elements x weight bits x activation bits, against the knee budget (M <= R M8, B <= R B8)
  whole-model weight storage: searched projections at their formats (+ one FP32 scale per output channel for int
     weights, reported separately), the 7 unobserved experts as bf16, every other tensor FP32 (FP32 model: all FP32)
  BOPs actually executed on the evaluation tokens: every token's 8 routed experts per layer (eval routes) x elements x
     weight bits x activation bits, plus the FP32 linear layers (attention projections, router, LM head; 32 x 32 bits);
     attention score / value products are activation x activation, FP32 in every configuration, and not counted

Stages (one --out-dir): plan, resources (CPU), truth (GPU), analyze.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MP = HERE / "runs/olmoe_multipoint_20261015"
REG = HERE / "runs/solver_regression_20261016"
SEQ = 128
SETS = {"A": MP, "B": REG}
CONFIGS = {"milp_0707": MP / "configs/bc_p125.json", "lp_knee": REG / "configs/lp_knee.json"}
PROJ = ("gate_proj", "up_proj", "down_proj")


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    from mp_auto import config_hash
    reg = jload(REG / "regress.json")["olmoe"]
    names = sorted(jload(CONFIGS["lp_knee"]))
    (out / "configs").mkdir(exist_ok=True)
    for k, src in CONFIGS.items():
        shutil.copyfile(src, out / "configs" / f"{k}.json")
    milp, lp = jload(CONFIGS["milp_0707"]), jload(CONFIGS["lp_knee"])
    checks = {
        "milp_0707_is_regression_reference_knee": config_hash({n: milp[n] for n in names}) == reg["reference"]["knee"]["config_hash"],
        "milp_0707_is_regression_milp_knee_file": sha256(CONFIGS["milp_0707"]) == sha256(REG / "configs/milp_knee.json")
        or jload(REG / "configs/milp_knee.json") == milp,
        "lp_knee_is_lp_repair_knee": config_hash({n: lp[n] for n in names}) == reg["variants"]["lp_repair"]["knee"]["config_hash"],
        "lp_knee_is_lp_first_knee": config_hash({n: lp[n] for n in names}) == reg["variants"]["lp_first"]["knee"]["config_hash"],
        "same_searched_names": sorted(milp) == names,
        "nodes_differing": sum(milp[n] != lp[n] for n in names)}
    if not all(v for k, v in checks.items() if k != "nodes_differing"):
        raise SystemExit(f"configuration identity check failed: {checks}")
    shutil.copyfile(MP / "ranges.json", out / "ranges.json")
    jdump(out / "plan.json", {
        "question": "measured quantization damage and actual size / BOPs of the MILP knee (R 0.707) and the LP + "
                    "rounding + repair knee, against FP32, fixed route",
        "configs": {"fp32": "every tensor FP32 (BF16 checkpoint cast to FP32), FP32 route",
                    **{k: {"source": str(v.relative_to(HERE)), "sha256": sha256(v)} for k, v in CONFIGS.items()},
                    "identity_checks": checks},
        "unsearched": "experts with no calibration token: bf16 weights + FP32 activations (as in the runs that selected "
                      "these configurations)",
        "windows": {k: {"source": str(v.relative_to(HERE)), "tokens_sha256": sha256(v / "tokens_eval.npy"),
                        "routes_manifest_sha256": sha256(v / "routes_manifest.json")} for k, v in SETS.items()},
        "already_seen": "set A: bc_p125 and FP32 were measured in olmoe_multipoint (dNLL +0.0175); set B: all three were "
                        "measured in solver_regression (milp +0.0098, lp +0.0107, paired +0.0009 [-0.0013, +0.0031]). "
                        "Both sets are replayed again here in one code path; nothing is selected from them",
        "truth": "per window summed next-token CE (127 predictions per window), run_olmoe_multipoint.replay, GPU FP32, "
                 "TF32 off, ranges.json of the multipoint run",
        "analysis": {"per set and pooled (128 windows)": "dNLL vs FP32 per token, PPL, paired lp_knee - milp_0707",
                     "bootstrap": "2000 resamples stratified by the four segments (A-V, A-T, B-V, B-T; 32 each), seed 0",
                     "determinism": "set B summed CE against solver_regression truth.json",
                     "criteria": "none (a measurement); the solver regression's equivalence margin +-0.002 is quoted "
                                 "for reference only"},
        "resources": "see the module docstring",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan: {checks}")


def stage_resources(args, out, log):
    import olmoe_common as oc
    from htq_pareto_search import FORMAT_BITS as FB
    from run_olmoe_multipoint import problem
    cfg = oc.config()
    ck = oc.Checkpoint()
    _, meta, names, _, elems, rows, flops = problem(MP)
    L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
    shapes = {n: tuple(ck.handles[ck.map[n]].get_slice(n).get_shape()) for n in ck.map}
    expert_t = {f"model.layers.{l}.mlp.experts.{e}.{p}.weight": (l, e, p) for l in range(L) for e in range(E) for p in PROJ}
    other_elems = sum(int(np.prod(s)) for n, s in shapes.items() if n not in expert_t)
    searched = set(names)
    key = lambda l, e, p: f"L{l}.E{e}.{p}"
    unobserved = [k for (l, e, p) in expert_t.values() if (k := key(l, e, p)) not in searched]
    n_out = {p: shapes[f"model.layers.0.mlp.experts.0.{p}.weight"][0] for p in PROJ}
    pe = 2048 * 1024
    assert all(int(np.prod(shapes[t])) == pe for t in expert_t)

    # actual routed rows of every (layer, expert) on each evaluation set
    act_rows = {}
    for s, d in SETS.items():
        man = jload(d / "routes_manifest.json")["eval"]
        cnt = np.zeros((L, E), np.int64)
        for k in range(man["windows"]):
            ids = np.load(d / "routes" / f"eval_{k:03d}.npz")["ids"]          # [L, T, K]
            for l in range(L):
                cnt[l] += np.bincount(ids[l].ravel().astype(np.int64), minlength=E)
        act_rows[s] = {"counts": cnt, "tokens": man["windows"] * SEQ}
        assert cnt.sum() == L * K * man["windows"] * SEQ

    attn = sum(int(np.prod(shapes[f"model.layers.{l}.self_attn.{p}_proj.weight"])) for l in range(L) for p in "qkvo")
    router = sum(int(np.prod(shapes[f"model.layers.{l}.mlp.gate.weight"])) for l in range(L))
    head = int(np.prod(shapes["lm_head.weight"]))

    def fmt_of(fm, k):
        return fm[k] if fm is not None and k in fm else ({"w": "fp32", "a": "fp32"} if fm is None else {"w": "bf16", "a": "fp32"})

    res = {"model": {"parameters": int(sum(int(np.prod(s)) for s in shapes.values())), "expert_projections": len(expert_t),
                     "searched": len(names), "unobserved": len(unobserved), "other_tensor_elements": other_elems,
                     "linear_fp32_elements_per_token": {"attention_qkvo": attn, "router": router, "lm_head": head}},
           "configs": {}}
    M8 = sum(elems[n] for n in names)
    B8 = sum(64 * flops[n] for n in names)
    for c in ("fp32", "milp_0707", "lp_knee"):
        fm = None if c == "fp32" else jload(out / "configs" / f"{c}.json")
        r = {}
        if fm is not None:
            M = sum(elems[n] * FB[fm[n]["w"]] / 8 for n in names)
            B = sum(flops[n] * FB[fm[n]["w"]] * FB[fm[n]["a"]] for n in names)
            r["searched_scope"] = {"M_bytes": M, "M_over_M8": M / M8, "B_calibration": B, "B_over_B8": B / B8,
                                   "R": max(M / M8, B / B8), "within_R_0.7071067811865476":
                                       bool(M <= 0.7071067811865476 * M8 and B <= 0.7071067811865476 * B8)}
        # whole-model weight storage
        exp_bytes, scale_bytes, wcount = 0.0, 0, {}
        for (l, e, p) in expert_t.values():
            f = fmt_of(fm, key(l, e, p))["w"]
            exp_bytes += pe * FB[f] / 8
            wcount[f] = wcount.get(f, 0) + 1
            if f.startswith("int"):
                scale_bytes += n_out[p] * 4
        r["storage"] = {"expert_bytes": exp_bytes, "other_fp32_bytes": other_elems * 4, "int_scale_bytes": scale_bytes,
                        "total_bytes": exp_bytes + other_elems * 4 + scale_bytes,
                        "weight_formats": dict(sorted(wcount.items()))}
        if fm is not None:
            r["activation_formats"] = dict(sorted({a: sum(fm[n]["a"] == a for n in names) for a in
                                                   {fm[n]["a"] for n in names}}.items()))
        # BOPs executed on each evaluation set
        r["bops"] = {}
        for s, ar in act_rows.items():
            ex = sum(int(ar["counts"][l, e]) * pe * FB[fmt_of(fm, key(l, e, p))["w"]] * FB[fmt_of(fm, key(l, e, p))["a"]]
                     for (l, e, p) in expert_t.values())
            lin = ar["tokens"] * (attn + router + head) * 32 * 32
            r["bops"][s] = {"experts": float(ex), "fp32_linear": float(lin), "total": float(ex + lin),
                            "tokens": ar["tokens"]}
        res["configs"][c] = r
    f32 = res["configs"]["fp32"]
    for c in ("milp_0707", "lp_knee"):
        r = res["configs"][c]
        r["vs_fp32"] = {"storage_ratio": r["storage"]["total_bytes"] / f32["storage"]["total_bytes"],
                        "expert_storage_ratio": (r["storage"]["expert_bytes"] + r["storage"]["int_scale_bytes"])
                        / f32["storage"]["expert_bytes"],
                        **{f"bops_{s}_ratio": r["bops"][s]["total"] / f32["bops"][s]["total"] for s in SETS},
                        **{f"expert_bops_{s}_ratio": r["bops"][s]["experts"] / f32["bops"][s]["experts"] for s in SETS}}
    a, b = res["configs"]["lp_knee"], res["configs"]["milp_0707"]
    res["lp_minus_milp"] = {"M_bytes": a["searched_scope"]["M_bytes"] - b["searched_scope"]["M_bytes"],
                            "B_calibration_rel": a["searched_scope"]["B_calibration"] / b["searched_scope"]["B_calibration"] - 1,
                            "storage_bytes": a["storage"]["total_bytes"] - b["storage"]["total_bytes"],
                            **{f"bops_{s}_rel": a["bops"][s]["total"] / b["bops"][s]["total"] - 1 for s in SETS},
                            **{f"expert_bops_{s}_rel": a["bops"][s]["experts"] / b["bops"][s]["experts"] - 1 for s in SETS}}
    fm_m, fm_l = jload(out / "configs/milp_0707.json"), jload(out / "configs/lp_knee.json")
    res["differing_nodes"] = [{"node": n, "milp_0707": fm_m[n], "lp_knee": fm_l[n], "calibration_rows": rows[n]}
                              for n in names if fm_m[n] != fm_l[n]]
    jdump(out / "resources.json", res)
    for c, r in res["configs"].items():
        log(f"{c}: {json.dumps({k: v for k, v in r.items() if k != 'storage'}, default=str)[:600]}")
        log(f"   storage {r['storage']}")
    log(f"lp - milp: {res['lp_minus_milp']}")


def stage_truth(args, out, log):
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    from run_olmoe_multipoint import replay
    gpu_exact(args.device)
    pl = jload(out / "plan.json")
    for k in ("milp_0707", "lp_knee"):
        if sha256(out / "configs" / f"{k}.json") != pl["configs"][k]["sha256"]:
            raise SystemExit(f"{k} changed after the plan")
    f = out / "truth.json"
    done = jload(f) if f.exists() else {}
    for s, d in SETS.items():
        if s in done:
            continue
        if sha256(d / "tokens_eval.npy") != pl["windows"][s]["tokens_sha256"]:
            raise SystemExit(f"set {s} tokens changed")
        rts, _, _ = load_routes(d, "eval", oc.config(), args.device)
        ce, sec = replay(out, ["fp32", "milp_0707", "lp_knee"], np.load(d / "tokens_eval.npy"), rts, args.device, log,
                         f"set {s}")
        done[s] = {"summed_ce": ce, "seconds": sec}
        jdump(f, done)
        log(f"set {s}: {sec:.0f}s")


def stage_analyze(args, out, log):
    tr = jload(out / "truth.json")
    T = SEQ - 1
    rng = np.random.default_rng(0)
    draws = np.concatenate([q * 32 + rng.integers(0, 32, (2000, 32)) for q in range(4)], 1)   # A-V, A-T, B-V, B-T
    ce = {c: np.concatenate([np.array(tr[s]["summed_ce"][c]) for s in ("A", "B")]) for c in ("fp32", "milp_0707", "lp_knee")}
    parts = {"A": slice(0, 64), "B": slice(64, 128), "all": slice(0, 128)}

    def st(d, part):
        idx = draws[:, :64] if part == "A" else draws[:, 64:] if part == "B" else draws
        b = d[idx].mean(1)
        x = d[parts[part]]
        return {"mean": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                "windows": int(len(x))}
    res = {"sets": {}, "determinism_set_B": None}
    for part in ("A", "B", "all"):
        sl = parts[part]
        n = len(ce["fp32"][sl])
        row = {"fp32_nll": float(ce["fp32"][sl].sum() / (n * T)), "fp32_ppl": float(np.exp(ce["fp32"][sl].sum() / (n * T)))}
        for c in ("milp_0707", "lp_knee"):
            row[c] = {"nll": float(ce[c][sl].sum() / (n * T)), "ppl": float(np.exp(ce[c][sl].sum() / (n * T))),
                      "dnll": st((ce[c] - ce["fp32"]) / T, part)}
            row[c]["ppl_rel"] = row[c]["ppl"] / row["fp32_ppl"] - 1
        row["lp_minus_milp"] = st((ce["lp_knee"] - ce["milp_0707"]) / T, part)
        res["sets"][part] = row
    old = jload(REG / "truth.json")["summed_ce"]
    res["determinism_set_B"] = {new: float(np.abs(np.array(tr["B"]["summed_ce"][new]) - np.array(old[o])).max())
                                for new, o in (("fp32", "fp32"), ("milp_0707", "milp_knee"), ("lp_knee", "lp_knee"))}
    mp_old = jload(MP / "truth.json")["summed_ce"]
    res["vs_multipoint_set_A"] = {new: float(np.abs(np.array(tr["A"]["summed_ce"][new]) - np.array(mp_old[o])).max())
                                  for new, o in (("fp32", "fp32"), ("milp_0707", "bc_p125"))}
    jdump(out / "analysis.json", res)
    for part, row in res["sets"].items():
        log(f"{part}: FP32 NLL {row['fp32_nll']:.4f} PPL {row['fp32_ppl']:.3f} | MILP dNLL {row['milp_0707']['dnll']['mean']:+.4f} "
            f"{row['milp_0707']['dnll']['ci95']} PPL {row['milp_0707']['ppl']:.3f} | LP dNLL {row['lp_knee']['dnll']['mean']:+.4f} "
            f"{row['lp_knee']['dnll']['ci95']} PPL {row['lp_knee']['ppl']:.3f} | LP-MILP {row['lp_minus_milp']['mean']:+.4f} "
            f"{row['lp_minus_milp']['ci95']}")
    log(f"determinism set B (max |summed CE| diff vs solver_regression): {res['determinism_set_B']}; set A vs "
        f"multipoint: {res['vs_multipoint_set_A']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "resources", "truth", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    status = "ok"
    try:
        globals()[f"stage_{args.stage}"](args, out, log)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
        timing["runs"].append({"stage": args.stage, "device": args.device, "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
