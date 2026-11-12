#!/usr/bin/env python3
"""OLMoE fixed route: closing validation of the faster search (pre-registered in plan.json, one --out-dir).

Fixed route only (every layer: the window's cached FP32 expert ids and routing weights); BC, tables, ranges, formats and
the selection rule are unchanged; nothing here changes mp_milp / mp_auto or any earlier output.

Experiment A (CPU / GPU disagreement on LP knee - MILP knee), diagnostics on the solver-regression windows (set B):
  a_repro    the earlier common subset replayed again with the earlier code (run_olmoe_multipoint.replay): CPU
             (16 windows) in the CPU venv, GPU (all 64) in the GPU env; CPU repeat on 4 windows
  a_checks   tokens, loss reduction, routes, weights, ranges / scale / zero-point, posit LUTs, dtypes, versions
  a_dd       per-window LP - MILP on CPU and GPU, difference of differences, trace windows (rule in plan.json)
  a_trace    one process, CPU and GPU in lockstep over the trace windows: every QDQ site's input / output on both
             devices, the device quantizer applied to the CPU input (control), layer / MoE outputs, loss
  a_ablate   GPU replays: this script's replay with the earlier quantizer (must reproduce a_repro bitwise), with the
             corrected quantizer if the control finds a quantizer difference, and with a 1-ulp embedding perturbation
Experiment B (complete re-run of the corrected search), CPU, methods one after the other, nothing else running:
  b_problem  the selection problem (table, exact dominance pruning) and its build cost
  b_search   --method milp | lp | lpfirst: mp_auto.auto_select with the per-budget solver replaced in-process
  b_controls knee-related budgets that a method did not sample, solved by it (timed separately)
  b_analyze
Experiment C (new windows), after A and B:
  c_freeze, c_windows (CPU venv: tokenize, disjointness), c_capture (GPU routes), c_truth (GPU), c_cpu, c_analyze
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MP = HERE / "runs/olmoe_multipoint_20261015"
REG = HERE / "runs/solver_regression_20261016"
KC = HERE / "runs/olmoe_knee_check_20261016"
SPEED = HERE / "runs/olmoe_milp_speed_20261016"
STAGE1 = HERE / "runs/olmoe_fixed_route_20261015"
SEQ = 128
T_PRED = SEQ - 1
A_SUBSET = list(range(0, 8)) + list(range(32, 40))          # the solver regression's CPU subset of set B
A_REPEAT = [0, 1, 32, 33]
C_EVAL = {"V": ("validation", 196608, 256), "T": ("test", 196608, 256)}
C_CPU = list(range(0, 8)) + list(range(256, 264))
TIME_LIMIT = 300.0
REL_TOL, ABS_TOL = 1e-3, 1e-6
EQUIV = 0.002
BOOT = 10000
A_CONFIGS = {"milp_0707": MP / "configs/bc_p125.json", "lp_knee": REG / "configs/lp_knee.json"}
SCRIPTS = ["run_olmoe_search_final.py", "olmoe_common.py", "run_olmoe_multipoint.py", "run_olmoe_fixed_route.py",
           "run_solver_regression.py", "run_olmoe_milp_speed.py", "mp_auto.py", "mp_milp.py", "qwen3_swiglu_common.py",
           "eval_random_mixed_precision.py", "htq_pareto_search.py", "run_native_bc_pipeline.py"]


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(v):
    return float(v) if v is not None and math.isfinite(float(v)) else None


def script_hashes():
    return {s: sha256(HERE / s) for s in SCRIPTS}


def env_info(device=None):
    import platform
    import scipy
    info = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, torch_cuda=torch.version.cuda, threads=torch.get_num_threads(),
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    matmul_precision=torch.get_float32_matmul_precision())
        if device and device != "cpu" and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    (out / "configs").mkdir(exist_ok=True)
    for k, src in A_CONFIGS.items():
        shutil.copyfile(src, out / "configs" / f"{k}.json")
    shutil.copyfile(MP / "ranges.json", out / "ranges.json")
    inputs = {"bc_table": MP / "table.json", "table_meta": MP / "table_meta.json", "ranges": MP / "ranges.json",
              "milp_0707_config": A_CONFIGS["milp_0707"], "lp_knee_config": A_CONFIGS["lp_knee"],
              "setB_tokens": REG / "tokens_eval.npy", "setB_routes_manifest": REG / "routes_manifest.json",
              "setB_gpu_truth_regression": REG / "truth.json", "setB_cpu_truth_regression": REG / "truth_cpu.json",
              "setB_gpu_truth_knee_check": KC / "truth.json", "multipoint_selection_raw": MP / "selection_raw.json",
              "speed_study_repair": SPEED / "repair.json", "stage1_plan": STAGE1 / "plan.json",
              "multipoint_plan": MP / "plan.json", "regression_plan": REG / "plan.json"}
    jdump(out / "plan.json", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "OLMoE-1B-7B, fixed route only (cached FP32 expert ids and unnormalized routing weights per layer); no "
                 "dynamic route; BC criterion, BC table, calibration, ranges, posit LUTs, candidate formats, resource "
                 "model, auto_select budgets and knee rule unchanged; mp_milp / mp_auto / pipeline defaults unchanged",
        "background_verified": {
            "1": "3051 searched expert projections (problem(): table keys) - confirmed",
            "2": "MILP 0.707 (bc_p125) and the LP knee differ in 7 projections - confirmed (olmoe_knee_check)",
            "3": "128 windows LP - MILP +0.00046 nat/token, 95% CI [-0.0012, +0.0022] (2000 stratified resamples) - "
                 "confirmed; equivalence within +-0.002 not shown",
            "4": "CPU subset (16 windows of set B; only the two knees were replayed on CPU, FP32 was not) paired mean "
                 "+0.0076 vs GPU +0.00006 on the same windows - confirmed; cause not established",
            "5": "the MILP 0.707 reference knee = recorded MILP candidates + the speed study's repaired R 0.5 point, knee "
                 "recomputed; refinements were placed around the old knee R 0.569 - confirmed; no complete corrected "
                 "auto_select exists",
            "6": "pure LP + rounding + repair auto_select: 16.6 s (speed study) and 14.4 s (solver regression, run "
                 "concurrently with another regression process); LP first with fallback: 53.6 s (solver regression, "
                 "concurrent; its fallback was one MILP + repair, not the retry path); different methods, different "
                 "conditions - confirmed",
            "differences_found": ["multipoint report.md explains the R 0.5 failure as 'no solution within the time "
                                  "limit'; the speed study showed the 1 ppm retry made the problem infeasible (attempt "
                                  "1 had a solution 2.5e-7 over the budget)"]},
        "pre_plan_observation": "probe in the GPU env (torch 2.14.0+cu130): x / (0-dim CPU tensor) on CUDA differed "
                                "from the CPU division in 0.21% of 4M random elements, x / (CUDA tensor) in 0%. "
                                "qwen3_swiglu_common.int_act_qdq divides by a 0-dim CPU tensor; DevQuant's weight path "
                                "was verified bitwise, its activation path was not. Whether quantized codes and the "
                                "loss change is not known; experiment A tests it",
        "inputs_sha256": {k: sha256(v) for k, v in inputs.items()},
        "scripts_sha256": script_hashes(),
        "experiment_A": {
            "hypotheses": {"H0": "a setup difference between the CPU and GPU replays (tokens, loss reduction, routes, "
                                 "weights, ranges / scale / zero-point, posit LUT, dtype)",
                           "H1": "the GPU activation quantizer gives different outputs than the CPU quantizer for "
                                 "bitwise identical inputs",
                           "H2": "upstream floating-point differences (attention / matmul / reduction order) change "
                                 "the quantizer inputs; the resulting output differences propagate"},
            "data": "set B (solver-regression evaluation windows); CPU subset " + str(A_SUBSET) + "; CPU repeat "
                    + str(A_REPEAT) + "; configs fp32, milp_0707, lp_knee (copies in configs/)",
            "steps": ["a_repro: earlier code (run_olmoe_multipoint.replay), CPU venv on the CPU subset (12 threads), "
                      "GPU env on all 64 windows; must reproduce solver_regression / olmoe_knee_check bitwise; CPU "
                      "repeat on 4 windows",
                      "a_checks: listed in the hypotheses H0",
                      "a_dd: per window d = (LP - MILP)/127 per backend, dd = d_CPU - d_GPU; segment means; stratified "
                      f"paired bootstrap ({BOOT} resamples, seed 0) of mean d_CPU, d_GPU, dd; largest |dd| window",
                      "a_trace windows: the largest |dd| window and the smallest |dd| window (contrast); one process "
                      "(GPU env), CPU and GPU in lockstep; per QDQ site: input bitwise-equal fraction and max |diff|, "
                      "output mismatch count, mismatches with identical inputs, control = GPU quantizer applied to the "
                      "CPU input vs the CPU quantizer (and the corrected GPU quantizer), QDQ error; per layer: hidden / "
                      "MoE output differences, LP - MILP hidden difference per device; final CE; the 7 differing "
                      "projections reported separately; earliest layer of divergence",
                      "a_ablate (GPU, set B, 64 windows, fp32 / milp_0707 / lp_knee): (i) this script's replay with "
                      "the earlier quantizer (must equal a_repro GPU bitwise), (ii) corrected quantizer (only if the "
                      "control shows quantizer mismatches on identical inputs), (iii) corrected (or earlier, if no "
                      "correction) quantizer with the embedding output multiplied by (1 + s 2^-23), s = +-1 per "
                      "element, numpy seed 0: sensitivity of the quantized pipeline to 1-ulp upstream differences; "
                      "CPU subset compared with (i) and (ii)",
                      "optional: switching the 7 projections one at a time (sorted by name) from MILP to LP on the "
                      "trace windows, CPU and GPU - only if the trace leaves the configuration effect unlocated; "
                      "diagnostic only, single effects not assumed additive"],
            "correction_rule": "if the control shows that the GPU and CPU quantizers give different outputs for bitwise "
                               "identical inputs, that is an implementation error of the evaluation path (the CPU "
                               "Quantizers / ONNX QuantizeLinear divide by the scale): a corrected GPU activation "
                               "quantizer (scale and zero-point as tensors on the input's device) is implemented in "
                               "this script only, verified bitwise against the CPU quantizer on the traced inputs, "
                               "recorded as an amendment; earlier results stay as they are and are reported separately; "
                               "the BC table (computed with the earlier GPU quantizer) is not recomputed",
            "stop": "the listed steps; if the cause is not located, report what is excluded and what is not"},
        "experiment_B": {
            "problem": "runs/olmoe_multipoint_20261015 table.json + table_meta.json: 3051 projections, GRID11 x GRID11, "
                       "exact dominance pruning (run_olmoe_multipoint.nondominated), M = sum elements x weight bits / 8, "
                       "B = sum calibration routed rows x elements x weight bits x activation bits, R = max(M/M8, B/B8)",
            "search": "mp_auto.auto_select(samples=17, refinements=2), endpoints, feasibility check and knee as is; the "
                      "per-budget solver replaced in-process; no loss used",
            "methods": {
                "milp": "corrected MILP (minimal change of mp_milp.milp_select): attempt 1 at the exact budget, if its "
                        "answer exceeds the exact check the 1 ppm retry, both as before; only if the retry gives no "
                        "solution or an over-budget answer and attempt 1 had a solution, attempt 1's answer is repaired "
                        "into the exact budget by option swaps (run_solver_regression.repair), labelled "
                        "'repaired_after_retry_failure', returned with status 1 (not proven optimal); attempt 1 without "
                        "a solution is a failure as before (no cheapest-configuration substitute)",
                "lp": "pure LP relaxation + rounding (fractional node: its least-resource option in the LP support) + "
                      "repair, status 1 (not proven optimal); bound = LP optimum",
                "lpfirst": f"lp, accepted if objective - LP bound <= max({REL_TOL:g} objective, {ABS_TOL:g} x objective "
                           "scale), objective scale = max |cost| / 1e4; otherwise the corrected MILP above"},
            "solver": f"scipy {__import__('scipy').__version__} milp (HiGHS 1.12.0), objective scaled to max |cost| = "
                      f"1e4, constraint rows divided by the budget, mip_rel_gap 0, time_limit {TIME_LIMIT:g} s, presolve "
                      "default, disp on (log captured per attempt); LP: the same matrix, integrality 0, default options",
            "timing": "methods one after the other in separate processes, nothing else heavy running; end-to-end "
                      "auto_select seconds; per call: build, LP, rounding + repair, attempt 1, retry, repair, MILP "
                      "fallback; per MILP attempt from HiGHS's log: presolve end, first incumbent, status, primal / dual "
                      "bound, gap, nodes, warnings ('unavailable' when not printed); problem build (table load, "
                      "pruning) reported separately; one measurement each (no repeat, no stable speed-up ratio claimed)",
            "controls": "knee-related budgets = for each method: its final knee's search budget, its knee's left and "
                        "right frontier neighbours' budgets, and its last refinement round's budgets; every such budget "
                        "a method did not attempt (relative 1e-12) is solved by that method afterwards, timed "
                        "separately, not fed back into any selection; at most 10 MILP control solves (nearest to the "
                        "MILP knee first)",
            "analysis": ["sampled budgets and frontier per method",
                         "same-budget comparison: BC, M, B, feasibility, number of differing projections",
                         "per round: knee, neighbours, next sampling range (reconstructed from the solve order and "
                         "checked against the attempted budgets)",
                         "if the knee differs: same-budget solution quality, rounding slack, near-duplicate points "
                         "changing neighbours, near-tied knee scores changing the refinement direction, missing "
                         "endpoint",
                         "knee score of the best point and the runner-ups (R differs > 0.5%; immediate neighbours)"],
            "numerics": "a lower bound is not a proof: flagged 'suspect' when it exceeds the method's own objective, or "
                        "the objective of any exactly feasible configuration known at that budget (any method, any "
                        "budget <= it, the endpoints), by more than 1e-9 x max(1, objective); objectives below 1e-6 x "
                        "objective scale (HiGHS's absolute gap) are 'below solver resolution', reported with absolute "
                        "differences; HiGHS warnings recorded; no per-point retuning of tolerances",
            "stop": "if the MILP auto_select runs over 3 h, stop and report the completed part"},
        "experiment_C": {
            "order": "after A and B: freeze the final configurations of the three methods (identical ones evaluated "
                     "once), then tokenize the new windows, check disjointness, capture routes, evaluate",
            "windows": {k: {"split": v[0], "first_token": v[1], "windows": v[2]} for k, v in C_EVAL.items()},
            "windows_disjoint_from": "stage 1 calibration / diagnostic / check, multipoint evaluation (token 131072), "
                                     "solver-regression evaluation (token 163840): character spans",
            "sample_size": "SD of the per-window LP - MILP difference on the 128 knee-check windows = 0.0100 nat/token "
                           "(assumed for the new configurations; larger if they differ in more projections); 95% CI "
                           "half-width 1.96 x 0.0100 / sqrt(n): n = 384 gives 0.0010, n = 512 gives 0.00087; 512 "
                           "chosen (GPU replay about 5 min); fixed before evaluation, never increased",
            "backend": "primary GPU FP32, TF32 off, run_olmoe_multipoint's replay semantics; GPU activation quantizer: "
                       "the corrected one if experiment A applied the correction rule, else the earlier one; CPU "
                       "cross-check subset " + str(C_CPU) + " (first 8 windows of each segment), all evaluated "
                       "configurations, reported separately",
            "configs": "fp32, milp (experiment B corrected MILP knee), lp (pure LP knee), lpfirst (only if different)",
            "metrics": "NLL, PPL, dNLL vs FP32, paired dNLL of each faster method's configuration vs the MILP one, per "
                       f"segment; bootstrap {BOOT} resamples stratified by segment, seed 0; resources (searched scope M "
                       "/ B / R, whole-model storage estimate, BOPs on the evaluation tokens) with their differences",
            "decision": {"equivalent": f"95% CI inside [-{EQUIV}, +{EQUIV}]",
                         "materially different": f"95% CI entirely above +{EQUIV} or below -{EQUIV}",
                         "different, equivalence not shown": "95% CI excludes 0 but overlaps the margin",
                         "inconclusive": "95% CI contains 0 and extends beyond the margin",
                         "multiplicity": "per method the 95% CI decides; a joint statement that both faster methods "
                                         "are equivalent needs both Bonferroni 97.5% CIs inside the margin (only when "
                                         "lp and lpfirst give different configurations); CI containing 0 is not "
                                         "equivalence",
                         "cpu": "the CPU subset is reported on its own, never averaged with the GPU result"}},
        "not_claimed": ["BOPs reduction = hardware speed-up", "format storage estimate = runtime memory",
                        "OLMoE result generalizes to other models", "a new default search method"]})
    log("plan written")


# --------------------------------------------------------------------------- #
# quantizers and replay
# --------------------------------------------------------------------------- #
def _act_corrected(self, t, fmt, rng):
    """Quantizers.act with the int scale / zero-point as tensors on t's device (true division on every device)."""
    import torch
    import eval_random_mixed_precision as erm
    if fmt.startswith("int"):
        bits = int(fmt[3:])
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        scale, zp = erm.int_qparams(float(rng[0]), float(rng[1]), bits)
        s = torch.tensor(np.float32(scale), dtype=t.dtype, device=t.device)
        z = torch.tensor(np.float32(zp), dtype=t.dtype, device=t.device)
        q = torch.clamp(torch.round(t / s + z), qmin, qmax)
        return (q - z) * s
    return self.dev.act(t, fmt, rng)


def dev_quant(out, device, corrected=False):
    from run_olmoe_multipoint import DevQuant
    dq = DevQuant(out, device)
    if corrected:
        dq.act = types.MethodType(_act_corrected, dq)
    return dq


def quantized_layer(w, fm, l, dq, ranges, E, record=None):
    """run_olmoe_multipoint.replay's per-layer weights and activation hooks; record(key, slot, x, y) if given."""
    import olmoe_common as oc
    wq = dict(w)
    hk = oc.ExpertHooks()
    for p in oc.PROJ:
        wp = w[p].clone()
        for e in range(E):
            key = f"L{l}.E{e}.{p}"
            f = fm[key] if key in fm else {"w": "bf16", "a": "fp32"}
            if f["w"] != "fp32":
                wp[e] = dq.weight(w[p][e], f["w"])
            if f["a"] != "fp32":
                for s in ("x", "y"):
                    if record is None:
                        hk.act[(l, e, p, s)] = (lambda t, a=f["a"], r=ranges[key][s]: dq.act(t, a, r))
                    else:
                        def fn(t, a=f["a"], r=ranges[key][s], k=key, sl=s):
                            y = dq.act(t, a, r)
                            record(k, sl, a, r, t, y)
                            return y
                        hk.act[(l, e, p, s)] = fn
        wq[p] = wp
    return wq, hk


def replay(out, cfgs, toks_np, routes, dev, dq, log, label, perturb_seed=None, chunk=128):
    """Per-window summed CE of every configuration ({name: formats or None = FP32}), layer-major like
    run_olmoe_multipoint.replay (one weight load and one weight quantization per layer and configuration), windows in
    chunks; perturb_seed: embedding output x (1 + s 2^-23), s = +-1 per element (numpy seed)."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    cfg = oc.config()
    ck = oc.Checkpoint()
    ranges = jload(out / "ranges.json")
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    nW = len(toks_np)
    chunks = [(c, min(nW, c + chunk)) for c in range(0, nW, chunk)]
    t0 = time.time()
    H = {}
    with torch.no_grad():
        for a, b in chunks:
            toks = torch.from_numpy(toks_np[a:b]).to(dev)
            h0 = F.embedding(toks, g["embed"])
            if perturb_seed is not None:
                s = np.random.default_rng(perturb_seed + a).choice([-1.0, 1.0], size=tuple(h0.shape)).astype(np.float32)
                h0 = h0 * (1 + torch.from_numpy(s).to(dev) * 2.0 ** -23)
            for n in cfgs:
                H[n, a] = h0.clone()
        for l in range(cfg.num_hidden_layers):
            w = ck.layer(l, dev, torch.float32)
            for n, fm in cfgs.items():
                if fm is None:
                    wq, hk = w, None
                else:
                    wq, hk = quantized_layer(w, fm, l, dq, ranges, cfg.num_experts)
                for a, b in chunks:
                    rows = slice(a * SEQ, b * SEQ)
                    H[n, a] = oc.layer_fixed(wq, H[n, a], cfg, cos, sin, routes[l][0][rows], routes[l][1][rows], l, hk)[0]
                del wq
            del w
            log(f"{label}: layer {l} done ({time.time() - t0:.0f}s)")
        res = {}
        for n in cfgs:
            res[n] = []
            for a, b in chunks:
                res[n] += head_ce(g, H[n, a], torch.from_numpy(toks_np[a:b]).to(dev), cfg).cpu().numpy().tolist()
    return res, time.time() - t0


def load_cfgs(out, names):
    return {n: (None if n == "fp32" else jload(out / "configs" / f"{n}.json")) for n in names}


def setB_routes(dev, windows=None):
    import torch
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    rts, ids, ws = load_routes(REG, "eval", oc.config(), dev)
    if windows is not None:
        rows = torch.cat([torch.arange(k * SEQ, (k + 1) * SEQ) for k in windows]).to(dev)
        rts = [(i[rows], w[rows]) for i, w in rts]
    return rts, ids, ws


# --------------------------------------------------------------------------- #
# experiment A
# --------------------------------------------------------------------------- #
def stage_a_repro(args, out, log):
    import torch
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    from run_olmoe_multipoint import replay as replay_old
    d = out / "A"
    d.mkdir(exist_ok=True)
    names = ["fp32", "milp_0707", "lp_knee"]
    toks = np.load(REG / "tokens_eval.npy")
    if args.device == "cpu":
        f = d / "repro_cpu.json"
        res = jload(f) if f.exists() else {"env": env_info("cpu")}
        rts, _, _ = load_routes(REG, "eval", oc.config(), "cpu")
        for tag, wins in (("subset", A_SUBSET), ("repeat", A_REPEAT)):
            if tag in res:
                continue
            rows = torch.cat([torch.arange(k * SEQ, (k + 1) * SEQ) for k in wins])
            r = [(i[rows], w[rows]) for i, w in rts]
            ce, sec = replay_old(out, names, toks[wins], r, "cpu", log, f"cpu {tag}")
            res[tag] = {"windows": wins, "summed_ce": ce, "seconds": sec}
            jdump(f, res)
    else:
        gpu_exact(args.device)
        f = d / "repro_gpu.json"
        if f.exists():
            return
        rts, _, _ = load_routes(REG, "eval", oc.config(), args.device)
        ce, sec = replay_old(out, names, toks, rts, args.device, log, "gpu")
        ce2, sec2 = replay_old(out, names, toks, rts, args.device, log, "gpu repeat")
        jdump(f, {"env": env_info(args.device), "summed_ce": ce, "seconds": sec, "repeat_summed_ce": ce2,
                  "repeat_seconds": sec2})


def stage_a_checks(args, out, log):
    """H0 checks in one process (GPU env), CPU and CUDA tensors side by side."""
    import torch
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact("cuda")
    d = out / "A"
    cfg = oc.config()
    ck = oc.Checkpoint()
    res = {"env": env_info("cuda")}
    # tokens and loss: both backends read the same file and use the same code path
    res["tokens"] = {"file_sha256": sha256(REG / "tokens_eval.npy"),
                     "same_array_cpu_cuda": bool(np.array_equal(np.load(REG / "tokens_eval.npy"),
                                                                torch.from_numpy(np.load(REG / "tokens_eval.npy")).cuda().cpu().numpy())),
                     "loss": "olmoe_common.summed_ce: F.cross_entropy(logits[:, :-1], ids[:, 1:], reduction='none') "
                             "summed per window in float64; 127 predictions per window on both backends; causal mask = "
                             "finfo.min above the diagonal, explicit softmax (no SDPA kernel) on both"}
    # routes: the cached arrays on both devices
    rc, ids, ws = setB_routes("cpu")
    rg, _, _ = setB_routes("cuda")
    res["routes"] = {"manifest_sha256": sha256(REG / "routes_manifest.json"), "layers": len(rc),
                     "ids_equal": all(torch.equal(a[0], b[0].cpu()) for a, b in zip(rc, rg)),
                     "weights_bitwise_equal": all(torch.equal(a[1], b[1].cpu()) for a, b in zip(rc, rg)),
                     "ids_dtype": str(rc[0][0].dtype), "weights_dtype": str(rc[0][1].dtype)}
    # checkpoint tensors cast to FP32 on each device
    eq, dtypes = True, set()
    for l in range(cfg.num_hidden_layers):
        wc, wg = ck.layer(l, "cpu", torch.float32), ck.layer(l, "cuda", torch.float32)
        eq &= all(torch.equal(wc[k], wg[k].cpu()) for k in wc)
        try:
            dtypes |= {str(ck.handles[ck.map[n]].get_slice(n).get_dtype()) for n in ck.map if f"layers.{l}." in n}
        except AttributeError:
            dtypes.add("unavailable")
    gc, gg = ck.global_("cpu", torch.float32), ck.global_("cuda", torch.float32)
    eq &= all(torch.equal(gc[k], gg[k].cpu()) for k in gc)
    res["weights"] = {"fp32_cast_bitwise_equal_all_layers": bool(eq), "checkpoint_dtypes": sorted(dtypes),
                      "compute_dtype": "float32"}
    # quantizers: weights (all projections of the two configurations), posit LUTs, int qparams
    dqc, dqg = dev_quant(d, "cpu"), dev_quant(d, "cuda")
    lut = {f: {"values_equal": bool(torch.equal(dqc.dev.posit[f].t_values, dqg.dev.posit[f].t_values.cpu())),
               "bounds_equal": bool(torch.equal(dqc.dev.posit[f].t_bounds, dqg.dev.posit[f].t_bounds.cpu())),
               "values": int(len(dqc.dev.posit[f].t_values))} for f in dqc.dev.posit}
    res["posit_lut"] = lut
    cfgs = load_cfgs(out, ["milp_0707", "lp_knee"])
    wmis = {}
    for l in range(cfg.num_hidden_layers):
        wc, wg = ck.layer(l, "cpu", torch.float32), ck.layer(l, "cuda", torch.float32)
        for n, fm in cfgs.items():
            for p in oc.PROJ:
                for e in range(cfg.num_experts):
                    key = f"L{l}.E{e}.{p}"
                    f = fm[key]["w"] if key in fm else "bf16"
                    a, b = dqc.weight(wc[p][e], f), dqg.weight(wg[p][e], f).cpu()
                    if not torch.equal(a, b):
                        wmis.setdefault(n, []).append({"node": key, "format": f, "elements": int((a != b).sum())})
        log(f"checks: layer {l} weights compared")
    res["quantized_weights"] = {"projections_compared": 2 * 3 * cfg.num_hidden_layers * cfg.num_experts,
                                "mismatching": wmis}
    import eval_random_mixed_precision as erm
    rng = jload(out / "ranges.json")
    res["ranges"] = {"file_sha256": sha256(out / "ranges.json"), "same_file_both_backends": True,
                     "int_qparams": "erm.int_qparams in Python floats from the same ranges (identical on both "
                                    "backends by construction); scale / zero-point enter the quantizer as float32 "
                                    "0-dim tensors created on the CPU (torch.tensor(np.float32(.)))",
                     "example": {k: list(erm.int_qparams(*rng[k]["x"], 4)) for k in list(rng)[:2]}}
    res["qdq_semantics"] = "Quantizers.act (qwen3_swiglu_common): int: q = clamp(round(x / s + z)), (q - z) s; posit: "\
                           "LUT searchsorted on float order keys; f16 / bf16: casts; applied to each projection's input "\
                           "(x) and output (y) before the routing-weight product, identical code on both backends"
    jdump(d / "checks.json", res)
    log(f"checks: routes {res['routes']['ids_equal']}/{res['routes']['weights_bitwise_equal']}, weights "
        f"{eq}, LUT {lut}, quantized weight mismatches {sum(len(v) for v in wmis.values())}")


def boot_draws(n_per_seg, nseg, seed=0, B=BOOT):
    rng = np.random.default_rng(seed)
    return np.concatenate([q * n_per_seg + rng.integers(0, n_per_seg, (B, n_per_seg)) for q in range(nseg)], 1)


def ci(d, draws):
    b = d[draws].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n": int(len(d))}


def stage_a_dd(args, out, log):
    d = out / "A"
    cpu, gpu = jload(d / "repro_cpu.json"), jload(d / "repro_gpu.json")
    sub = np.array(A_SUBSET)
    old_cpu, old_gpu = jload(REG / "truth_cpu.json")["summed_ce"], jload(KC / "truth.json")["B"]["summed_ce"]
    c, g = cpu["subset"]["summed_ce"], gpu["summed_ce"]
    res = {"reproduction": {
        "cpu_vs_regression_cpu_max_abs": {k: float(np.abs(np.array(c[k]) - np.array(old_cpu[o])).max())
                                          for k, o in (("milp_0707", "milp_knee"), ("lp_knee", "lp_knee"))},
        "gpu_vs_knee_check_max_abs": {k: float(np.abs(np.array(g[k]) - np.array(old_gpu[k])).max()) for k in g},
        "gpu_repeat_max_abs": {k: float(np.abs(np.array(g[k]) - np.array(gpu["repeat_summed_ce"][k])).max()) for k in g},
        "cpu_repeat_max_abs": {k: float(np.abs(np.array(cpu["repeat"]["summed_ce"][k])
                                              - np.array(c[k])[[A_SUBSET.index(w) for w in A_REPEAT]]).max()) for k in c}}}
    gs = {k: np.array(v)[sub] for k, v in g.items()}
    cs = {k: np.array(v) for k, v in c.items()}
    draws = boot_draws(8, 2)
    dC = (cs["lp_knee"] - cs["milp_0707"]) / T_PRED
    dG = (gs["lp_knee"] - gs["milp_0707"]) / T_PRED
    dd = dC - dG
    res["per_window"] = [{"window": int(w), "segment": "V" if w < 32 else "T", "d_cpu": float(dC[i]), "d_gpu": float(dG[i]),
                          "dd": float(dd[i]),
                          **{f"cpu_minus_gpu_{k}": float((cs[k][i] - gs[k][i]) / T_PRED) for k in cs}}
                         for i, w in enumerate(A_SUBSET)]
    res["summary"] = {"d_cpu": ci(dC, draws), "d_gpu": ci(dG, draws), "dd": ci(dd, draws),
                      "segments": {s: {"d_cpu": float(dC[sl].mean()), "d_gpu": float(dG[sl].mean()), "dd": float(dd[sl].mean())}
                                   for s, sl in (("V", slice(0, 8)), ("T", slice(8, 16)))},
                      "sd": {"d_cpu": float(dC.std(ddof=1)), "d_gpu": float(dG.std(ddof=1)), "dd": float(dd.std(ddof=1))},
                      "corr_d_cpu_d_gpu": float(np.corrcoef(dC, dG)[0, 1]),
                      "cpu_minus_gpu_per_config": {k: ci((cs[k] - gs[k]) / T_PRED, draws) for k in cs},
                      "cpu_minus_gpu_abs_max": {k: float(np.abs(cs[k] - gs[k]).max() / T_PRED) for k in cs},
                      "gpu_all64_d": ci((np.array(g["lp_knee"]) - np.array(g["milp_0707"])) / T_PRED, boot_draws(32, 2))}
    i_max, i_min = int(np.argmax(np.abs(dd))), int(np.argmin(np.abs(dd)))
    res["trace_windows"] = {"largest_abs_dd": A_SUBSET[i_max], "smallest_abs_dd": A_SUBSET[i_min]}
    jdump(d / "dd.json", res)
    log(json.dumps({k: res[k] for k in ("reproduction", "summary", "trace_windows")})[:3000])


def stage_a_cpu_repeat(args, out, log):
    """Amendment 1: same-shape CPU repeats (A_REPEAT windows as one batch, fp32 / milp_0707 / lp_knee, as a_repro's
    'repeat'), tagged by --tag; MKL_CBWR from the environment is recorded; plus a matmul micro-test."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    from run_olmoe_multipoint import replay as replay_old
    d = out / "A"
    f = d / "cpu_repeat.json"
    res = jload(f) if f.exists() else {}
    if args.tag in res:
        return
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1024, 2048, generator=g)
    W = torch.randn(1024, 2048, generator=g)
    a = F.linear(x, W)
    b = F.linear(x.clone(), W.clone())
    buf = torch.empty(1024 * 2048 + 1)
    xs = buf[1:].view(1024, 2048)
    xs.copy_(x)
    c = F.linear(xs, W)
    micro = {"same_buffers_equal": bool(torch.equal(a, F.linear(x, W))), "fresh_copies_equal": bool(torch.equal(a, b)),
             "misaligned_copy_equal": bool(torch.equal(a, c)), "misaligned_max_abs": float((a - c).abs().max()),
             "x_data_ptr_mod64": [x.data_ptr() % 64, xs.data_ptr() % 64]}
    rts, _, _ = load_routes(REG, "eval", oc.config(), "cpu")
    wins = A_SUBSET if args.tag.startswith("subset") else A_REPEAT
    rows = torch.cat([torch.arange(k * SEQ, (k + 1) * SEQ) for k in wins])
    r = [(i[rows], w[rows]) for i, w in rts]
    cnames = ["milp_0707", "lp_knee"] if "twoconfigs" in args.tag else ["fp32", "milp_0707", "lp_knee"]
    ce, sec = replay_old(out, cnames, np.load(REG / "tokens_eval.npy")[wins], r, "cpu", log, f"cpu {args.tag}")
    res = jload(f) if f.exists() else {}
    res[args.tag] = {"windows": wins, "OMP_DYNAMIC": os.environ.get("OMP_DYNAMIC"), "MKL_DYNAMIC": os.environ.get("MKL_DYNAMIC"), "summed_ce": ce, "seconds": sec, "MKL_CBWR": os.environ.get("MKL_CBWR"),
                     "micro_matmul": micro, "env": env_info("cpu")}
    jdump(f, res)
    log(f"{args.tag}: micro {micro}")


def stage_a_trace(args, out, log):
    """CPU and GPU in lockstep (one process) over each trace window; see plan.json."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact("cuda")
    d = out / "A"
    tw = jload(d / "dd.json")["trace_windows"]
    cfg = oc.config()
    ck = oc.Checkpoint()
    ranges = jload(out / "ranges.json")
    dqc, dqg, dqf = dev_quant(d, "cpu"), dev_quant(d, "cuda"), dev_quant(d, "cuda", corrected=True)
    names = ["fp32", "milp_0707", "lp_knee"]
    cfgs = load_cfgs(out, names)
    fm_m, fm_l = cfgs["milp_0707"], cfgs["lp_knee"]
    diff7 = sorted(k for k in fm_m if fm_m[k] != fm_l[k])
    toks_all = np.load(REG / "tokens_eval.npy")
    res = {"windows": {}, "differing_projections": diff7}
    for role, win in tw.items():
        rc, _, _ = setB_routes("cpu", [win])
        rg, _, _ = setB_routes("cuda", [win])
        toks = toks_all[win:win + 1]
        tc, tg = torch.from_numpy(toks), torch.from_numpy(toks).cuda()
        cos_c, sin_c = oc.rope(cfg, SEQ, "cpu", torch.float32)
        cos_g, sin_g = oc.rope(cfg, SEQ, "cuda", torch.float32)
        gc_, gg_ = ck.global_("cpu", torch.float32), ck.global_("cuda", torch.float32)
        with torch.no_grad():
            Hc = {n: F.embedding(tc, gc_["embed"]) for n in names}
            Hg = {n: F.embedding(tg, gg_["embed"]) for n in names}
            layers = []
            sites7 = []
            for l in range(cfg.num_hidden_layers):
                wc, wg = ck.layer(l, "cpu", torch.float32), ck.layer(l, "cuda", torch.float32)
                row = {"layer": l, "configs": {}}
                for n in names:
                    recc, recg = [], []
                    if cfgs[n] is None:
                        oc_out, mc = oc.layer_fixed(wc, Hc[n], cfg, cos_c, sin_c, *rc[l], l)
                        og_out, mg = oc.layer_fixed(wg, Hg[n], cfg, cos_g, sin_g, *rg[l], l)
                    else:
                        wqc, hkc = quantized_layer(wc, cfgs[n], l, dqc, ranges, cfg.num_experts,
                                                   lambda *a: recc.append(a))
                        wqg, hkg = quantized_layer(wg, cfgs[n], l, dqg, ranges, cfg.num_experts,
                                                   lambda *a: recg.append(a))
                        oc_out, mc = oc.layer_fixed(wqc, Hc[n], cfg, cos_c, sin_c, *rc[l], l, hkc)
                        og_out, mg = oc.layer_fixed(wqg, Hg[n], cfg, cos_g, sin_g, *rg[l], l, hkg)
                        del wqc, wqg
                    agg = {"sites": 0, "elements": 0, "in_bitwise_equal": 0, "out_mismatch": 0,
                           "out_mismatch_with_equal_input": 0, "control_gpu_quantizer_on_cpu_input_mismatch": 0,
                           "control_corrected_quantizer_mismatch": 0, "in_max_abs": 0.0, "by_format": {}}
                    assert len(recc) == len(recg)
                    for (k1, s1, a1, r1, xc, yc), (k2, s2, a2, r2, xg, yg) in zip(recc, recg):
                        assert (k1, s1) == (k2, s2)
                        xg, yg = xg.cpu(), yg.cpu()
                        same_in = (xc.view(torch.int32) == xg.view(torch.int32))
                        mis = yc.view(torch.int32) != yg.view(torch.int32)
                        ctl = dqg.act(xc.cuda(), a1, r1).cpu().view(torch.int32) != yc.view(torch.int32)
                        ctf = dqf.act(xc.cuda(), a1, r1).cpu().view(torch.int32) != yc.view(torch.int32)
                        st = {"elements": int(xc.numel()), "in_bitwise_equal": int(same_in.sum()), "out_mismatch": int(mis.sum()),
                              "out_mismatch_with_equal_input": int((mis & same_in).sum()),
                              "control_gpu_quantizer_on_cpu_input_mismatch": int(ctl.sum()),
                              "control_corrected_quantizer_mismatch": int(ctf.sum())}
                        agg["sites"] += 1
                        for kk, vv in st.items():
                            agg[kk] += vv
                        agg["in_max_abs"] = max(agg["in_max_abs"], float((xc - xg).abs().max()) if xc.numel() else 0.0)
                        bf = agg["by_format"].setdefault(a1, {kk: 0 for kk in st})
                        for kk, vv in st.items():
                            bf[kk] += vv
                        if k1 in diff7:
                            sites7.append({"layer": l, "config": n, "node": k1, "slot": s1, "format": a1, **st,
                                           "in_max_abs": float((xc - xg).abs().max()) if xc.numel() else 0.0,
                                           "qdq_err_cpu": float((yc - xc).abs().mean()) if xc.numel() else 0.0,
                                           "qdq_err_gpu": float((yg - xg).abs().mean()) if xc.numel() else 0.0})
                    og_c, mg_c = og_out.cpu(), mg.cpu()
                    agg.update(layer_out_max_abs=float((oc_out - og_c).abs().max()),
                               layer_out_rel_l2=float((oc_out - og_c).norm() / oc_out.norm()),
                               moe_out_rel_l2=float((mc - mg_c).norm() / mc.norm()),
                               layer_out_bitwise_equal_frac=float((oc_out.view(torch.int32) == og_c.view(torch.int32)).float().mean()))
                    row["configs"][n] = agg
                    Hc[n], Hg[n] = oc_out, og_out
                # the configuration difference on each device
                dc = (Hc["lp_knee"] - Hc["milp_0707"])
                dg_ = (Hg["lp_knee"] - Hg["milp_0707"]).cpu()
                row["lp_minus_milp"] = {"cpu_rel_l2": float(dc.norm() / Hc["milp_0707"].norm()),
                                        "gpu_rel_l2": float(dg_.norm() / Hg["milp_0707"].norm().cpu()),
                                        "cpu_vs_gpu_of_difference_rel": float((dc - dg_).norm() / max(dc.norm(), 1e-30))}
                layers.append(row)
                log(f"trace {role} w{win} layer {l}: " + "; ".join(
                    f"{n}: out mismatch {a['out_mismatch']}/{a['elements']} (eq-input {a['out_mismatch_with_equal_input']}, "
                    f"ctl {a['control_gpu_quantizer_on_cpu_input_mismatch']}, ctl-fix {a['control_corrected_quantizer_mismatch']}), "
                    f"layer rel {a['layer_out_rel_l2']:.2e}" for n, a in row["configs"].items()))
            ce = {}
            for n in names:
                zc = oc.summed_ce(oc.head(gc_, Hc[n], cfg), tc)
                zg = oc.summed_ce(oc.head(gg_, Hg[n], cfg), tg).cpu()
                ce[n] = {"cpu": float(zc[0]), "gpu": float(zg[0])}
        res["windows"][role] = {"window": win, "layers": layers, "differing_projection_sites": sites7, "summed_ce": ce,
                                "dd_per_token": ((ce["lp_knee"]["cpu"] - ce["milp_0707"]["cpu"])
                                                 - (ce["lp_knee"]["gpu"] - ce["milp_0707"]["gpu"])) / T_PRED}
        jdump(d / "trace.json", res)


def stage_a_ablate(args, out, log):
    import torch
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact(args.device)
    d = out / "A"
    f = d / "ablate.json"
    res = jload(f) if f.exists() else {"env": env_info(args.device)}
    names = ["fp32", "milp_0707", "lp_knee"]
    cfgs = load_cfgs(out, names)
    toks = np.load(REG / "tokens_eval.npy")
    rts, _, _ = setB_routes(args.device)
    variants = [("earlier_quantizer", False, None)]
    if args.corrected:
        variants += [("corrected_quantizer", True, None), ("corrected_quantizer_perturbed", True, 0)]
    else:
        variants += [("earlier_quantizer_perturbed", False, 0)]
    for tag, corr, pert in variants:
        if tag in res:
            continue
        dq = dev_quant(d, args.device, corrected=corr)
        ce, sec = replay(out, cfgs, toks, rts, args.device, dq, log, tag, perturb_seed=pert)
        res[tag] = {"summed_ce": ce, "seconds": sec}
        jdump(f, res)
        log(f"{tag}: {sec:.0f}s")


def stage_a_analyze(args, out, log):
    d = out / "A"
    ab = jload(d / "ablate.json")
    g0 = jload(d / "repro_gpu.json")["summed_ce"]
    cpu = jload(d / "repro_cpu.json")["subset"]["summed_ce"]
    sub = np.array(A_SUBSET)
    draws16, draws64 = boot_draws(8, 2), boot_draws(32, 2)
    res = {"replay_reimplementation_equals_earlier": {k: float(np.abs(np.array(ab["earlier_quantizer"]["summed_ce"][k])
                                                                      - np.array(g0[k])).max()) for k in g0}}
    for tag in [t for t in ab if t != "env"]:
        ce = {k: np.array(v) for k, v in ab[tag]["summed_ce"].items()}
        r = {"vs_earlier_gpu_per_config": {k: {"max_abs_window_per_token": float(np.abs(ce[k] - np.array(g0[k])).max() / T_PRED),
                                               "mean_per_token": float((ce[k] - np.array(g0[k])).mean() / T_PRED)} for k in ce},
             "d_all64": ci((ce["lp_knee"] - ce["milp_0707"]) / T_PRED, draws64)}
        dg = (ce["lp_knee"][sub] - ce["milp_0707"][sub]) / T_PRED
        dc = (np.array(cpu["lp_knee"]) - np.array(cpu["milp_0707"])) / T_PRED
        r["cpu_subset"] = {"cpu_minus_gpu_per_config": {k: {"max_abs_window_per_token": float(np.abs(np.array(cpu[k]) - ce[k][sub]).max() / T_PRED),
                                                            "mean_per_token": float((np.array(cpu[k]) - ce[k][sub]).mean() / T_PRED),
                                                            "windows_bitwise_equal": int((np.array(cpu[k]) == ce[k][sub]).sum())}
                                                        for k in cpu},
                           "d_gpu": ci(dg, draws16), "d_cpu": ci(dc, draws16), "dd": ci(dc - dg, draws16)}
        if "perturbed" in tag:
            base = "corrected_quantizer" if "corrected" in tag else "earlier_quantizer"
            b = {k: np.array(v) for k, v in ab[base]["summed_ce"].items()}
            delta = {k: (ce[k] - b[k]) / T_PRED for k in ce}
            r["perturbation_effect"] = {k: {"sd_per_window": float(v.std(ddof=1)), "max_abs": float(np.abs(v).max()),
                                            "mean": float(v.mean())} for k, v in delta.items()}
            dd_p = delta["lp_knee"] - delta["milp_0707"]
            r["perturbation_effect"]["on_lp_minus_milp"] = {"sd_per_window": float(dd_p.std(ddof=1)),
                                                            "mean": ci(dd_p, draws64)}
        res[tag] = r
    jdump(d / "analysis.json", res)
    log(json.dumps(res)[:4000])


# --------------------------------------------------------------------------- #
# experiment B: solvers
# --------------------------------------------------------------------------- #
_libc = ctypes.CDLL(None)


def captured(fn, path):
    """Run fn with the process's stdout (HiGHS's log) sent to path."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(path, "w") as fh:
        os.dup2(fh.fileno(), 1)
        try:
            return fn()
        finally:
            _libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)


def parse_highs(path):
    txt = Path(path).read_text()
    r = {"warnings": [l.strip() for l in txt.splitlines() if l.strip().startswith("WARNING")]}
    m = re.search(r"Cost\s+\[([^,\]]+),\s*([^\]]+)\]", txt)
    r["cost_range_scaled"] = [float(m.group(1)), float(m.group(2))] if m else "unavailable"
    ps = txt.find("Presolving model")
    pe = max(txt.find("Presolve reductions"), txt.find("Solving MIP model with"))
    if ps >= 0 and pe > ps:
        ts = re.findall(r"nonzeros\s+([\d.]+)s", txt[ps:pe])
        r["presolve_end_s"] = float(ts[-1]) if ts else "unavailable"
    else:
        r["presolve_end_s"] = "unavailable"
    red = re.search(r"Presolve reductions: rows (\d+)\(-?(\d+)\); columns (\d+)\(-?(\d+)\)", txt)
    r["presolve_removed_columns"] = int(red.group(4)) if red else "unavailable"
    first = None
    for line in txt.splitlines():
        t = line.split()
        if not t or not t[-1].endswith("s"):
            continue
        try:
            sec = float(t[-1][:-1])
        except ValueError:
            continue
        pct = [i for i, x in enumerate(t) if x.endswith("%")]
        if not pct or len(t) < pct[0] + 4:
            continue
        try:
            bs = float(t[pct[0] + 2])
        except ValueError:
            continue
        if math.isfinite(bs):
            first = {"time_s": sec, "src": t[0] if t[0].isalpha() else "", "objective_scaled": bs}
            break
    r["first_incumbent"] = first if first else "unavailable"
    for key, pat in (("status", r"Status\s+(\S+)"), ("primal_bound", r"Primal bound\s+(\S+)"),
                     ("dual_bound", r"Dual bound\s+(\S+)"), ("gap", r"\n\s+Gap\s+(\S+)"), ("timing_s", r"Timing\s+(\S+)"),
                     ("nodes", r"\n\s+Nodes\s+(\S+)"), ("row_violation", r"(\S+) \(row viol\.\)"),
                     ("lp_iterations", r"LP iterations\s+(\S+)")):
        m = re.search(pat, txt)
        r[key] = m.group(1) if m else "unavailable"
    return r


def make_solver(method, records, logdir, time_limit=TIME_LIMIT, tag="search"):
    """A drop-in for mp_milp.milp_select (same arguments and return) solving with `method`; one record per call."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from run_solver_regression import arrays, constraint, feasible, repair
    counter = [0]

    def rel(pick, size, bops, mb, mo):
        return {"M_over_budget": float(size[pick].sum() / mb - 1), "B_over_budget": float(bops[pick].sum() / mo - 1),
                "feasible": feasible(pick, size, bops, mb, mo)}

    def select(nodes, weight_formats, act_formats, value, weight_elems, flops_proxy, max_bytes, max_bops,
               time_limit=time_limit, info=None):
        counter[0] += 1
        call = counter[0]
        t0 = time.perf_counter()
        weighted, opts, off, cost, size, bops = arrays(nodes, weight_formats, act_formats, value, weight_elems,
                                                       flops_proxy)
        ref_bytes = sum(weight_elems[n] for n in weighted)
        scale = float(np.abs(cost).max()) / 1e4 or 1.0
        cons = constraint(off, size, bops, max_bytes, max_bops)
        rec = {"tag": tag, "call": call, "method": method, "budget": max_bytes / ref_bytes, "scale": scale,
               "build_s": time.perf_counter() - t0, "steps": []}
        pick = path = status = lb = None

        def milp_attempt(shrink, name):
            hi = np.r_[cons.ub[:-2], 1 - shrink, 1 - shrink]
            lg = logdir / f"{tag}_{method}_{call:03d}_{name}.log"
            ts = time.perf_counter()
            res = captured(lambda: milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
                                        constraints=LinearConstraint(cons.A, cons.lb, hi),
                                        options={"mip_rel_gap": 0.0, "time_limit": time_limit, "disp": True}), lg)
            st = {"step": name, "seconds": time.perf_counter() - ts, "status": int(res.status),
                  "message": str(res.message), "highs": parse_highs(lg), "log": lg.name}
            p = None
            if res.x is not None:
                p = np.array([off[k] + int(np.argmax(res.x[off[k]:off[k + 1]])) for k in range(len(weighted))])
                db = finite(res.get("mip_dual_bound"))
                st.update(objective=float(cost[p].sum()), dual_bound=db * scale if db is not None else None,
                          mip_gap=finite(res.get("mip_gap")), **rel(p, size, bops, max_bytes, max_bops))
            rec["steps"].append(st)
            return p, res, st

        def corrected_milp():
            p1, r1, s1 = milp_attempt(0.0, "attempt1")
            if p1 is None:
                return None, None, None, None, f"MILP found no solution: {r1.message}"
            if s1["feasible"]:
                return p1, "milp", int(r1.status), s1.get("dual_bound"), None
            p2, r2, s2 = milp_attempt(1e-6, "retry_1ppm")
            if p2 is not None and s2["feasible"]:
                return p2, "milp_retry", int(r2.status), s2.get("dual_bound"), None
            ts = time.perf_counter()
            p3, steps = repair(p1, cost, size, bops, off, max_bytes, max_bops)
            rec["steps"].append({"step": "repair_attempt1", "seconds": time.perf_counter() - ts, "swaps": steps,
                                 "retry_outcome": "no solution" if p2 is None else "over budget",
                                 **({"objective": float(cost[p3].sum()), **rel(p3, size, bops, max_bytes, max_bops)}
                                    if p3 is not None else {"failed": True})})
            if p3 is None:
                return None, None, None, None, "attempt 1 could not be repaired into the budget"
            return p3, "milp_repaired_after_retry_failure", 1, s1.get("dual_bound"), None

        err = None
        if method in ("lp", "lpfirst"):
            lg = logdir / f"{tag}_{method}_{call:03d}_lp.log"
            ts = time.perf_counter()
            res = captured(lambda: milp(cost / scale, integrality=np.zeros(len(cost)), bounds=Bounds(0, 1),
                                        constraints=cons, options={"disp": True}), lg)
            st = {"step": "lp", "seconds": time.perf_counter() - ts, "status": int(res.status), "log": lg.name}
            rec["steps"].append(st)
            if res.x is not None:
                ts = time.perf_counter()
                p0, frac = [], 0
                for k in range(len(weighted)):
                    seg = res.x[off[k]:off[k + 1]]
                    nz = np.nonzero(seg > 1e-9)[0]
                    if len(nz) > 1:
                        frac += 1
                        j = nz[np.lexsort((bops[off[k] + nz], size[off[k] + nz]))[0]]
                    else:
                        j = int(np.argmax(seg))
                    p0.append(off[k] + j)
                p0 = np.array(p0)
                before = rel(p0, size, bops, max_bytes, max_bops)
                p1, steps = repair(p0, cost, size, bops, off, max_bytes, max_bops)
                lpb = float(res.fun) * scale
                st2 = {"step": "round_repair", "seconds": time.perf_counter() - ts, "fractional_nodes": frac,
                       "swaps": steps, "rounded": {"objective": float(cost[p0].sum()), **before}, "lp_bound": lpb}
                if p1 is not None:
                    o = float(cost[p1].sum())
                    tol = max(REL_TOL * o, ABS_TOL * scale)
                    st2.update(objective=o, gap=o - lpb, tolerance=tol, accepted=bool(o - lpb <= tol),
                               **rel(p1, size, bops, max_bytes, max_bops))
                    if method == "lp" or o - lpb <= tol:
                        pick, path, status, lb = p1, "lp", 1, lpb
                rec["steps"].append(st2)
            if pick is None and method == "lp":
                err = f"LP route found no solution: {res.message}"
        if pick is None and err is None:
            pick, path, status, lb, err = corrected_milp()
            if path is not None and method == "lpfirst":
                path = "lp->" + path
        rec["seconds"] = time.perf_counter() - t0
        if err is not None:
            rec["failure"] = err
            records.append(rec)
            raise RuntimeError(err)
        obj = float(cost[pick].sum())
        rec.update(path=path, status=status, objective=obj, lower_bound=lb, gap_abs=(obj - lb) if lb is not None else None,
                   M_rel=float(size[pick].sum() / (ref_bytes)), **rel(pick, size, bops, max_bytes, max_bops))
        records.append(rec)
        if info is not None:
            info.update(path=path, status=status, objective=obj, seconds=rec["seconds"])
        formats = {n["name"]: {"w": None, "a": "fp32"} for n in nodes if not n.get("weight_const")}
        for k, name in enumerate(weighted):
            w, a = opts[k][pick[k] - off[k]]
            formats[name] = {"w": w, "a": a}
        return formats, obj, status
    return select


def olmoe_problem():
    import olmoe_common  # noqa: F401
    from run_olmoe_multipoint import nondominated, problem, pruned
    t0 = time.perf_counter()
    table, _, names, nodes, elems, _, flops = problem(MP)
    t1 = time.perf_counter()
    bc = lambda n, w, a: table["B"][n][w] + table["C"][n][a]
    keep = nondominated(names, bc)
    t2 = time.perf_counter()
    return nodes, names, pruned(bc, keep), elems, flops, {"load_table_s": t1 - t0, "dominance_pruning_s": t2 - t1,
                                                          "options_kept": sum(len(v) for v in keep.values())}


def stage_b_problem(args, out, log):
    d = out / "B"
    d.mkdir(exist_ok=True)
    nodes, names, value, elems, flops, cost = olmoe_problem()
    allv = np.array([v for n in names for v in (value(n, w, a) for w, a in
                                                [(w, a) for w in __import__('run_native_bc_pipeline').GRID11
                                                 for a in __import__('run_native_bc_pipeline').GRID11]) if v is not None])
    jdump(d / "problem.json", {"nodes": len(names), **cost, "max_abs_cost": float(np.abs(allv).max()),
                               "objective_scale": float(np.abs(allv).max()) / 1e4, "min_positive_cost": float(allv[allv > 0].min()),
                               "table_sha256": sha256(MP / "table.json"), "table_meta_sha256": sha256(MP / "table_meta.json"),
                               "env": env_info()})
    log(f"problem: {len(names)} nodes, {cost}")


def stage_b_search(args, out, log):
    import mp_auto
    from run_native_bc_pipeline import GRID11
    d = out / "B"
    f = d / f"search_{args.method}.json"
    if f.exists():
        raise SystemExit(f"{f.name} exists")
    (d / "logs").mkdir(exist_ok=True)
    nodes, names, value, elems, flops, pcost = olmoe_problem()
    recs, slog = [], []
    orig = mp_auto.milp_select
    mp_auto.milp_select = make_solver(args.method, recs, d / "logs")
    t0 = time.perf_counter()
    try:
        sel = mp_auto.auto_select(nodes, GRID11, GRID11, value, elems, flops, samples=17, refinements=2,
                                  time_limit=TIME_LIMIT, solver_log=slog)
    finally:
        mp_auto.milp_select = orig
    sec = time.perf_counter() - t0
    jdump(f, {"method": args.method, "auto_select_seconds": sec, "problem_build": pcost, "env": env_info(),
              "scripts_sha256": script_hashes(), "selection": sel, "calls": recs, "names": names})
    k = sel["selected"]
    log(f"{args.method}: auto_select {sec:.1f}s, {len(recs)} calls, failures {len(sel['failures'])}, knee R "
        f"{k['resource']:.6f} BC {k['objective']:.6f} budget {k['search_budget']}")


def frontier_rounds(sel):
    """auto_select's rounds re-run from its candidates: round 0 = endpoints + the 17 samples, then each refinement
    (budgets already attempted are skipped by auto_select, so a round can add fewer than 3); checked against the
    attempted budgets."""
    from mp_auto import _frontier, _knee
    cands = sel["candidates"]
    att = sel["attempted_budgets"]
    ends = [c for c in cands if c["origin"] != "milp"]
    solved = [c for c in cands if c["origin"] == "milp"]
    n, out = 17, []
    for r in range(3):
        prefix = [c for c in solved if any(c["search_budget"] == a for a in att[:n])]
        fr = _frontier(ends + prefix)
        knee, scores = _knee(fr)
        i = fr.index(knee)
        left, right = fr[max(0, i - 1)], fr[min(len(fr) - 1, i + 1)]
        row = {"after_round": r, "attempted_so_far": n,
               "knee": {"resource": knee["resource"], "objective": knee["objective"], "search_budget": knee["search_budget"],
                        "score": scores[i]},
               "left": {"resource": left["resource"], "objective": left["objective"], "search_budget": left.get("search_budget")},
               "right": {"resource": right["resource"], "objective": right["objective"], "search_budget": right.get("search_budget")},
               "top_scores": [{"resource": fr[j]["resource"], "objective": fr[j]["objective"], "search_budget": fr[j].get("search_budget"),
                               "score": scores[j]} for j in sorted(range(len(fr)), key=lambda j: -scores[j])[:5]]}
        if r < 2:
            nxt = [float(b) for b in np.geomspace(left["resource"], right["resource"], 5)[1:-1]]
            new = []
            for b in nxt:
                if not any(math.isclose(b, a, rel_tol=1e-12) for a in list(att[:n]) + new):
                    new.append(b)
            row["next_budgets"] = nxt
            row["next_new_budgets"] = new
            row["matches_attempted"] = all(math.isclose(b, a, rel_tol=1e-12) for b, a in zip(new, att[n:n + len(new)]))
            n += len(new)
        out.append(row)
    return out


def knee_related_budgets(sel):
    """The final knee's budget, its frontier neighbours' budgets, and the last refinement round's budgets."""
    fr = sel["frontier"]
    k = sel["selected"]
    i = next(j for j, r in enumerate(fr) if r["resource"] == k["resource"] and r["objective"] == k["objective"])
    b = {k["search_budget"]}
    for j in (i - 1, i + 1):
        if 0 <= j < len(fr) and fr[j].get("search_budget") is not None:
            b.add(fr[j]["search_budget"])
    rounds = frontier_rounds(sel)
    n1 = rounds[1]["attempted_so_far"]
    b |= set(sel["attempted_budgets"][n1:n1 + len(rounds[1]["next_new_budgets"])])
    return sorted(x for x in b if x is not None)


def stage_b_controls(args, out, log):
    import mp_auto
    from run_native_bc_pipeline import GRID11
    d = out / "B"
    S = {m: jload(d / f"search_{m}.json") for m in ("milp", "lp", "lpfirst")}
    related = {m: knee_related_budgets(s["selection"]) for m, s in S.items()}
    allb = sorted(set(b for v in related.values() for b in v))
    todo = []
    for m, s in S.items():
        att = s["selection"]["attempted_budgets"]
        for b in allb:
            if not any(math.isclose(b, a, rel_tol=1e-12) for a in att):
                todo.append((m, b))
    mk = S["milp"]["selection"]["selected"]["search_budget"]
    milp_todo = sorted([t for t in todo if t[0] == "milp"], key=lambda t: abs(math.log(t[1] / mk)))
    skipped = milp_todo[10:]
    todo = [t for t in todo if t[0] != "milp"] + milp_todo[:10]
    f = d / "controls.json"
    res = jload(f) if f.exists() else {"related_budgets": related, "planned": todo, "skipped": skipped, "solves": []}
    nodes, names, value, elems, flops, _ = olmoe_problem()
    ref_b = sum(elems[n] for n in names)
    ref_o = sum(64 * flops[n] for n in names)
    lookup_cost = value
    for m, b in todo:
        if any(r["method"] == m and math.isclose(r["budget"], b, rel_tol=1e-12) for r in res["solves"]):
            continue
        recs = []
        solver = make_solver(m, recs, d / "logs", tag="control")
        t0 = time.perf_counter()
        try:
            cfg, obj, status = solver(nodes, GRID11, GRID11, lookup_cost, elems, flops, b * ref_b, b * ref_o)
            fm = {n: cfg[n] for n in names}
            from htq_pareto_search import FORMAT_BITS as FB
            size = sum(elems[n] * FB[fm[n]["w"]] / 8 for n in names) / ref_b
            bops = sum(flops[n] * FB[fm[n]["w"]] * FB[fm[n]["a"]] for n in names) / ref_o
            row = {"method": m, "budget": b, "seconds": time.perf_counter() - t0, "objective": obj, "status": status,
                   "size": size, "bops": bops, "resource": max(size, bops), "formats": fm, "call": recs[-1]}
        except RuntimeError as exc:
            row = {"method": m, "budget": b, "seconds": time.perf_counter() - t0, "failure": str(exc), "call": recs[-1] if recs else None}
        res["solves"].append(row)
        jdump(f, res)
        log(f"control {m} R {b:.6f}: {row.get('objective')} ({row['seconds']:.1f}s)")


def stage_b_analyze(args, out, log):
    from mp_auto import _frontier, _knee, config_hash
    d = out / "B"
    S = {m: jload(d / f"search_{m}.json") for m in ("milp", "lp", "lpfirst")}
    names = S["milp"]["names"]
    ctl = jload(d / "controls.json") if (d / "controls.json").exists() else {"solves": []}
    pj = jload(d / "problem.json")
    scale = pj["objective_scale"]
    h = lambda fm: config_hash({n: fm[n] for n in names})
    res = {"methods": {}, "same_budget": [], "knees": {}, "numerics": {}, "timing": {}}
    # every exactly feasible solution known (method, budget, objective, resource) for the bound checks
    known = []
    for m, s in S.items():
        for c in s["selection"]["candidates"]:
            known.append((c["resource"], c["objective"]))
    for r in ctl["solves"]:
        if "objective" in r:
            known.append((r["resource"], r["objective"]))
    for m, s in S.items():
        sel = s["selection"]
        calls = s["calls"]
        res["methods"][m] = {
            "auto_select_seconds": s["auto_select_seconds"], "problem_build": s["problem_build"], "calls": len(calls),
            "failures": sel["failures"], "paths": {p: sum(c.get("path") == p for c in calls) for p in sorted({c.get("path") for c in calls if c.get("path")})},
            "attempted_budgets": sel["attempted_budgets"],
            "frontier": [{"resource": f["resource"], "size": f["size"], "bops": f["bops"], "objective": f["objective"],
                          "search_budget": f["search_budget"], "origin": f["origin"], "status": f["solver_status"]} for f in sel["frontier"]],
            "knee": {"resource": sel["selected"]["resource"], "size": sel["selected"]["size"], "bops": sel["selected"]["bops"],
                     "objective": sel["selected"]["objective"], "search_budget": sel["selected"]["search_budget"],
                     "config_hash": h(sel["selected"]["formats"])},
            "rounds": frontier_rounds(sel),
            "time_breakdown": {k: float(sum(st["seconds"] for c in calls for st in c["steps"] if st["step"] == k))
                               for k in ("lp", "round_repair", "attempt1", "retry_1ppm", "repair_attempt1")},
            "build_s_total": float(sum(c["build_s"] for c in calls))}
        flags = []
        for c in calls:
            if "objective" not in c:
                continue
            b = c["budget"]
            lb = c.get("lower_bound")
            best_known = min([o for (rr, o) in known if rr <= b + 1e-12] + [c["objective"]])
            fl = {"budget": b, "path": c["path"], "objective": c["objective"], "lower_bound": lb,
                  "gap_abs": c.get("gap_abs"), "best_known_feasible": best_known,
                  "below_solver_resolution": bool(c["objective"] < ABS_TOL * scale)}
            if lb is not None:
                fl["bound_above_own_objective"] = bool(lb > c["objective"] + 1e-9 * max(1.0, abs(c["objective"])))
                fl["bound_above_known_feasible"] = bool(lb > best_known + 1e-9 * max(1.0, abs(best_known)))
                fl["suspect"] = fl["bound_above_own_objective"] or fl["bound_above_known_feasible"]
                fl["gap_rel"] = (c["objective"] - lb) / c["objective"] if c["objective"] > 0 else None
            warn = sorted({w for st in c["steps"] for w in st.get("highs", {}).get("warnings", [])})
            fl["highs_warnings"] = warn
            flags.append(fl)
        res["numerics"][m] = flags
    # same-budget comparisons (search + control solves)
    rows = {}
    for m, s in S.items():
        for c in s["selection"]["candidates"]:
            if c["origin"] == "milp":
                rows.setdefault(m, []).append(c)
    for r in ctl["solves"]:
        if "objective" in r:
            rows.setdefault(r["method"], []).append({**r, "search_budget": r["budget"], "origin": "control"})
    budgets = sorted({c["search_budget"] for v in rows.values() for c in v})
    merged = []
    for b in budgets:
        if merged and math.isclose(merged[-1], b, rel_tol=1e-12):
            continue
        merged.append(b)
    for b in merged:
        e = {"budget": b}
        for m in S:
            c = next((x for x in rows.get(m, []) if math.isclose(x["search_budget"], b, rel_tol=1e-12)), None)
            if c is not None:
                e[m] = {"objective": c["objective"], "size": c["size"], "bops": c["bops"], "resource": c["resource"],
                        "origin": c["origin"], "hash": h(c["formats"])}
        if "milp" in e:
            fm0 = next(x for x in rows["milp"] if math.isclose(x["search_budget"], b, rel_tol=1e-12))["formats"]
            for m in ("lp", "lpfirst"):
                if m in e:
                    fm = next(x for x in rows[m] if math.isclose(x["search_budget"], b, rel_tol=1e-12))["formats"]
                    e[m]["differing_projections_vs_milp"] = sum(fm[n] != fm0[n] for n in names)
                    e[m]["bc_minus_milp"] = e[m]["objective"] - e["milp"]["objective"]
                    e[m]["bc_rel_vs_milp"] = e[m]["bc_minus_milp"] / e["milp"]["objective"] if e["milp"]["objective"] else None
        res["same_budget"].append(e)
    # knees and the union frontier
    K = {m: s["selection"]["selected"] for m, s in S.items()}
    for m in ("lp", "lpfirst"):
        a, b = K[m], K["milp"]
        res["knees"][f"{m}_vs_milp"] = {"identical": h(a["formats"]) == h(b["formats"]),
                                        "differing_projections": sum(a["formats"][n] != b["formats"][n] for n in names),
                                        "bc_rel": (a["objective"] - b["objective"]) / b["objective"],
                                        "R_rel": (a["resource"] - b["resource"]) / b["resource"],
                                        "budgets": [a["search_budget"], b["search_budget"]]}
    allc = [c for s in S.values() for c in s["selection"]["candidates"]] + \
           [{**r, "origin": "control", "solver_status": r["status"], "search_budget": r["budget"]} for r in ctl["solves"] if "objective" in r]
    fu = _frontier(allc)
    ku, su = _knee(fu)
    res["knees"]["union_frontier"] = {"points": len(fu), "knee": {"resource": ku["resource"], "objective": ku["objective"],
                                                                  "search_budget": ku["search_budget"], "origin": ku["origin"],
                                                                  "hash": h(ku["formats"])},
                                      "top_scores": sorted([{"resource": f["resource"], "objective": f["objective"],
                                                             "search_budget": f["search_budget"], "origin": f["origin"],
                                                             "score": s_} for f, s_ in zip(fu, su)], key=lambda x: -x["score"])[:6]}
    for m in S:
        r0 = res["methods"][m]["rounds"][-1]
        ts = r0["top_scores"]
        best = ts[0]
        runner = next((t for t in ts[1:] if abs(t["resource"] / best["resource"] - 1) > 0.005), None)
        res["knees"][f"{m}_score_margin"] = {"best": best, "runner_up_R_differs_0.5pct": runner,
                                             "margin": best["score"] - runner["score"] if runner else None}
    t_m = S["milp"]["auto_select_seconds"]
    res["timing"] = {m: {"auto_select_s": s["auto_select_seconds"], "speedup_vs_corrected_milp": t_m / s["auto_select_seconds"]}
                     for m, s in S.items()}
    res["timing"]["controls_s"] = float(sum(r["seconds"] for r in ctl["solves"]))
    res["timing"]["note"] = "one measurement per method, methods run one after the other with nothing else heavy"
    jdump(d / "analysis.json", res)
    for m in S:
        k = res["methods"][m]["knee"]
        log(f"{m}: {res['methods'][m]['auto_select_seconds']:.1f}s, paths {res['methods'][m]['paths']}, knee R "
            f"{k['resource']:.6f} BC {k['objective']:.6f}, failures {len(res['methods'][m]['failures'])}")
    log(f"knees: {json.dumps({k: v for k, v in res['knees'].items() if 'vs' in k})}")


# --------------------------------------------------------------------------- #
# experiment C
# --------------------------------------------------------------------------- #
def stage_c_freeze(args, out, log):
    d = out / "C"
    d.mkdir(exist_ok=True)
    if (d / "freeze.json").exists():
        raise SystemExit("frozen")
    from mp_auto import config_hash
    S = {m: jload(out / "B" / f"search_{m}.json") for m in ("milp", "lp", "lpfirst")}
    names = S["milp"]["names"]
    (d / "configs").mkdir(exist_ok=True)
    hashes, evals = {}, {}
    for m, s in S.items():
        fm = {n: s["selection"]["selected"]["formats"][n] for n in names}
        hh = config_hash(fm)
        hashes[m] = hh
        same = next((k for k, v in evals.items() if v == hh), None)
        if same:
            log(f"{m} knee = {same} knee: evaluated once")
            continue
        evals[m] = hh
        jdump(d / "configs" / f"{m}.json", fm)
    shutil.copyfile(out / "ranges.json", d / "ranges.json")
    a = jload(out / "A" / "analysis.json") if (out / "A" / "analysis.json").exists() else {}
    corrected = bool(args.corrected)
    jdump(d / "freeze.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "knee_hashes": hashes,
                              "evaluated": list(evals), "configs_sha256": {m: sha256(d / "configs" / f"{m}.json") for m in evals},
                              "ranges_sha256": sha256(d / "ranges.json"), "gpu_quantizer": "corrected" if corrected else "earlier",
                              "experiment_A_analysis_sha256": sha256(out / "A" / "analysis.json") if a else None,
                              "scripts_sha256": script_hashes()})
    log(f"frozen: {list(evals)}; GPU quantizer {'corrected' if corrected else 'earlier'}")


def stage_c_windows(args, out, log):
    import olmoe_common as oc
    d = out / "C"
    fz = jload(d / "freeze.json")
    if (d / "tokens_eval.npy").exists():
        raise SystemExit("windows exist")
    tok = oc.tokenizer()
    s1 = jload(STAGE1 / "plan.json")["data"]
    prev = [(s1[n]["split"], s1[n]["char_spans"]) for n in ("calibration", "diagnostic", "check")]
    for src in (MP, REG):
        p = jload(src / "plan.json")
        segs = p["eval_windows"]["segments"] if "eval_windows" in p else p["olmoe_eval"]["windows"]["segments"]
        prev += [(v["split"], [v["char_spans"]]) for v in segs.values()]
    toks, segs = [], {}
    for k, (split, first, n) in C_EVAL.items():
        t, spans, ntok, _ = oc.windows(split, first, n, SEQ, tok)
        for sp_split, sp in prev:
            if sp_split == split and spans[0][0] < sp[-1][1] and spans[-1][1] > sp[0][0]:
                raise SystemExit(f"segment {k} overlaps an earlier window set")
        toks.append(t)
        segs[k] = {"split": split, "tokens": [first, first + n * SEQ], "windows": n,
                   "char_spans": [spans[0][0], spans[-1][1]], "split_tokens": ntok}
    np.save(d / "tokens_eval.npy", np.concatenate(toks))
    jdump(d / "windows.json", {"segments": segs, "sha256": sha256(d / "tokens_eval.npy"), "frozen_before": fz["time"],
                               "disjoint_checked_against": len(prev)})
    log(f"windows: {segs}")


def stage_c_capture(args, out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, head_ce
    gpu_exact(args.device)
    d = out / "C"
    if (d / "routes_manifest.json").exists():
        return
    (d / "routes").mkdir(exist_ok=True)
    cfg = oc.config()
    ck = oc.Checkpoint()
    toks_np = np.load(d / "tokens_eval.npy")
    g = ck.global_(args.device, torch.float32)
    cos, sin = oc.rope(cfg, SEQ, args.device, torch.float32)
    hashes, ces, t0 = [], [], time.time()
    for a in range(0, len(toks_np), 64):
        tk = torch.from_numpy(toks_np[a:a + 64]).to(args.device)
        B = len(tk)
        idxs, rws = [], []
        with torch.no_grad():
            h = F.embedding(tk, g["embed"])
            for l in range(cfg.num_hidden_layers):
                w = ck.layer(l, args.device, torch.float32)
                h, idx, rw = oc.layer_dynamic(w, h, cfg, cos, sin)
                idxs.append(idx.view(B, SEQ, -1).to(torch.int16).cpu().numpy())
                rws.append(rw.view(B, SEQ, -1).float().cpu().numpy())
                del w
            ces += head_ce(g, h, tk, cfg).cpu().numpy().tolist()
        ia, ra = np.stack(idxs, 1), np.stack(rws, 1)
        for k in range(B):
            np.savez(d / "routes" / f"eval_{a + k:03d}.npz", ids=ia[k], weights=ra[k], tokens=toks_np[a + k])
            hashes.append(oc.route_hash(ia[k], ra[k]))
        log(f"capture: windows {a}-{a + B - 1} ({time.time() - t0:.0f}s)")
    jdump(d / "routes_manifest.json", {"eval": {"windows": len(toks_np), "dtype_ids": "int16", "dtype_weights": "float32",
                                                "hash": hashes, "tokens_sha256": sha256(d / "tokens_eval.npy"),
                                                "model_revision": oc.REVISION, "summed_ce_dynamic": ces,
                                                "seconds": time.time() - t0, "env": env_info(args.device)}})


def c_check(out):
    d = out / "C"
    fz = jload(d / "freeze.json")
    for m, hh in fz["configs_sha256"].items():
        if sha256(d / "configs" / f"{m}.json") != hh:
            raise SystemExit(f"{m} changed after the freeze")
    return fz


def stage_c_truth(args, out, log):
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    gpu_exact(args.device)
    d = out / "C"
    fz = c_check(out)
    if (d / "truth.json").exists():
        return
    names = ["fp32"] + fz["evaluated"]
    cfgs = load_cfgs(d, names)
    rts, _, _ = load_routes(d, "eval", oc.config(), args.device)
    dq = dev_quant(d, args.device, corrected=fz["gpu_quantizer"] == "corrected")
    ce, sec = replay(d, cfgs, np.load(d / "tokens_eval.npy"), rts, args.device, dq, log, "C truth")
    jdump(d / "truth.json", {"summed_ce": ce, "seconds": sec, "gpu_quantizer": fz["gpu_quantizer"], "env": env_info(args.device)})


def stage_c_cpu(args, out, log):
    import torch
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    d = out / "C"
    fz = c_check(out)
    if (d / "truth_cpu.json").exists():
        return
    names = ["fp32"] + fz["evaluated"]
    cfgs = load_cfgs(d, names)
    rts, _, _ = load_routes(d, "eval", oc.config(), "cpu")
    rows = torch.cat([torch.arange(k * SEQ, (k + 1) * SEQ) for k in C_CPU])
    rts = [(i[rows], w[rows]) for i, w in rts]
    dq = dev_quant(d, "cpu")
    ce, sec = replay(d, cfgs, np.load(d / "tokens_eval.npy")[C_CPU], rts, "cpu", dq, log, "C cpu", chunk=16)
    jdump(d / "truth_cpu.json", {"subset": C_CPU, "summed_ce": ce, "seconds": sec, "env": env_info("cpu")})


def resources_for(cfgs, routes_dir, nW):
    import olmoe_common as oc
    from htq_pareto_search import FORMAT_BITS as FB
    from run_olmoe_multipoint import problem
    cfg = oc.config()
    ck = oc.Checkpoint()
    _, _, names, _, elems, rows, flops = problem(MP)
    L, E = cfg.num_hidden_layers, cfg.num_experts
    shapes = {n: tuple(ck.handles[ck.map[n]].get_slice(n).get_shape()) for n in ck.map}
    expert_t = [(l, e, p) for l in range(L) for e in range(E) for p in oc.PROJ]
    pe = 2048 * 1024
    other = sum(int(np.prod(s)) for n, s in shapes.items() if ".mlp.experts." not in n)
    nout = {p: shapes[f"model.layers.0.mlp.experts.0.{p}.weight"][0] for p in oc.PROJ}
    cnt = np.zeros((L, E), np.int64)
    for k in range(nW):
        ids = np.load(routes_dir / "routes" / f"eval_{k:03d}.npz")["ids"]
        for l in range(L):
            cnt[l] += np.bincount(ids[l].ravel().astype(np.int64), minlength=E)
    lin = sum(int(np.prod(shapes[f"model.layers.{l}.self_attn.{p}_proj.weight"])) for l in range(L) for p in "qkvo") + \
        sum(int(np.prod(shapes[f"model.layers.{l}.mlp.gate.weight"])) for l in range(L)) + int(np.prod(shapes["lm_head.weight"]))
    M8, B8 = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
    res = {}
    for c, fm in cfgs.items():
        get = (lambda k: {"w": "fp32", "a": "fp32"}) if fm is None else (lambda k: fm.get(k, {"w": "bf16", "a": "fp32"}))
        r = {}
        if fm is not None:
            M = sum(elems[n] * FB[fm[n]["w"]] / 8 for n in names)
            B = sum(flops[n] * FB[fm[n]["w"]] * FB[fm[n]["a"]] for n in names)
            r["searched"] = {"M_bytes": M, "M_rel": M / M8, "B_rel": B / B8, "R": max(M / M8, B / B8)}
        eb = sum(pe * FB[get(f"L{l}.E{e}.{p}")["w"]] / 8 for (l, e, p) in expert_t)
        sb = sum(nout[p] * 4 for (l, e, p) in expert_t if get(f"L{l}.E{e}.{p}")["w"].startswith("int"))
        r["storage_bytes"] = {"experts": eb, "int_scales": sb, "other_fp32": other * 4, "total": eb + sb + other * 4}
        ex = sum(int(cnt[l, e]) * pe * FB[get(f"L{l}.E{e}.{p}")["w"]] * FB[get(f"L{l}.E{e}.{p}")["a"]] for (l, e, p) in expert_t)
        r["bops_eval"] = {"experts": float(ex), "fp32_linear": float(nW * SEQ * lin * 1024), "total": float(ex + nW * SEQ * lin * 1024)}
        res[c] = r
    return res


def stage_c_analyze(args, out, log):
    d = out / "C"
    fz = c_check(out)
    tr = jload(d / "truth.json")["summed_ce"]
    ev = fz["evaluated"]
    nV = C_EVAL["V"][2]
    draws = boot_draws(nV, 2)
    ce = {k: np.array(v) for k, v in tr.items()}
    n = len(ce["fp32"])
    res = {"gpu": {"fp32": {"nll": float(ce["fp32"].sum() / (n * T_PRED)), "ppl": float(np.exp(ce["fp32"].sum() / (n * T_PRED)))}},
           "decisions": {}, "cpu": None}
    segs = {"V": slice(0, nV), "T": slice(nV, n)}
    for m in ev:
        x = (ce[m] - ce["fp32"]) / T_PRED
        res["gpu"][m] = {"nll": float(ce[m].sum() / (n * T_PRED)), "ppl": float(np.exp(ce[m].sum() / (n * T_PRED))),
                         "dnll": ci(x, draws), "segments": {s: float(x[sl].mean()) for s, sl in segs.items()}}

    def decide(c):
        lo, hi = c
        if -EQUIV <= lo and hi <= EQUIV:
            return "equivalent"
        if lo > EQUIV or hi < -EQUIV:
            return "materially different"
        if lo > 0 or hi < 0:
            return "different, equivalence not shown"
        return "inconclusive"
    hk = fz["knee_hashes"]
    for m in ("lp", "lpfirst"):
        src = m if m in ev else next(k for k in ev if hk[k] == hk[m])
        if src == "milp":
            res["decisions"][m] = {"same_configuration_as_milp": True, "paired": None, "decision": "identical configuration"}
            continue
        x = (ce[src] - ce["milp"]) / T_PRED
        b = x[draws].mean(1)
        c95 = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        c975 = [float(np.percentile(b, 1.25)), float(np.percentile(b, 98.75))]
        res["decisions"][m] = {"evaluated_as": src, "paired": {"mean": float(x.mean()), "ci95": c95, "ci97_5": c975,
                                                               "sd_per_window": float(x.std(ddof=1)),
                                                               "segments": {s: float(x[sl].mean()) for s, sl in segs.items()}},
                               "decision": decide(c95), "decision_bonferroni_97_5": decide(c975)}
    distinct = len({hk["lp"], hk["lpfirst"]}) == 2 and all(hk[m] != hk["milp"] for m in ("lp", "lpfirst"))
    res["joint"] = {"two_distinct_faster_configurations": distinct,
                    "both_equivalent": (all(res["decisions"][m]["decision_bonferroni_97_5"] == "equivalent" for m in ("lp", "lpfirst"))
                                        if distinct else None)}
    if (d / "truth_cpu.json").exists():
        cpu = jload(d / "truth_cpu.json")
        cc = {k: np.array(v) for k, v in cpu["summed_ce"].items()}
        sub = np.array(cpu["subset"])
        d16 = boot_draws(8, 2)
        r = {"subset": cpu["subset"], "per_config": {}}
        for k in cc:
            r["per_config"][k] = {"cpu_nll": float(cc[k].sum() / (len(sub) * T_PRED)), "gpu_nll_same_windows": float(ce[k][sub].sum() / (len(sub) * T_PRED)),
                                  "max_abs_window_diff_per_token": float(np.abs(cc[k] - ce[k][sub]).max() / T_PRED)}
            if k != "fp32":
                r["per_config"][k]["dnll_cpu"] = ci((cc[k] - cc["fp32"]) / T_PRED, d16)
                r["per_config"][k]["dnll_gpu_same_windows"] = ci((ce[k][sub] - ce["fp32"][sub]) / T_PRED, d16)
        for m in [m for m in ev if m != "milp"]:
            xc = (cc[m] - cc["milp"]) / T_PRED
            xg = (ce[m][sub] - ce["milp"][sub]) / T_PRED
            r[f"{m}_minus_milp"] = {"cpu": ci(xc, d16), "gpu_same_windows": ci(xg, d16), "cpu_minus_gpu": ci(xc - xg, d16)}
        res["cpu"] = r
    cfgs = load_cfgs(d, ["fp32"] + ev)
    res["resources"] = resources_for(cfgs, d, n)
    jdump(d / "analysis.json", res)
    log(json.dumps({k: res[k] for k in ("gpu", "decisions", "joint")})[:3000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--method", choices=("milp", "lp", "lpfirst"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--corrected", action="store_true", help="a_ablate / c_freeze: use the corrected GPU quantizer")
    ap.add_argument("--tag", default=None, help="a_cpu_repeat: name of the repeat")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage.startswith(("a_", "c_")) and args.stage not in ("a_dd", "a_analyze", "c_freeze", "c_windows"):
        import torch
        torch.set_num_threads(args.threads)
    t0 = time.time()
    logf = open(out / f"{args.stage}{'_' + args.method if args.method else ''}.log", "a")

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
        timing["runs"].append({"stage": args.stage, "method": args.method, "device": args.device, "status": status,
                               "wall_s": time.time() - t0, "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "script_sha256": sha256(HERE / "run_olmoe_search_final.py")})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
