#!/usr/bin/env python3
"""OLMoE fixed route: is BC's single-point mis-ranking (stage 1, runs/olmoe_fixed_route_20261015) stable, or does the
measured ranking depend on numerical conditions? Pre-registered in plan.json (one --out-dir).

Fixed: stage 1's 1080 candidates (9 experts x {w, a, wa} x 40), their frozen BC scores, calibration ranges, routes,
resource model and the 2583 same-resource pairs. Each candidate quantizes only its own expert projection(s) (single
point), the rest of the model FP32, the FP32 route of every window, exactly as stage 1's truth stage.

Conditions (one truth pass each): quantizer old (int activation QDQ dividing by a 0-dim CPU scale on the GPU, as stage 1
ran) or corrected (shared int_act_qdq since runs/fix_integration_20261016); perturbation none or the embedding output x
(1 + m s), s = +-1 per element (numpy seed), shared by every candidate and by the condition's own FP32 baseline.

Stages:
  plan       plan.json (+ hashes)
  windows    (CPU venv) the new windows B: tokenization, character-span disjointness
  capture    (GPU) FP32 routes of B
  truth      --set A|B --cond U|m23s0..|m20s0.. --quant corrected|old [--layers 8,15] [--tag rep2] [--device]
  mech       local mechanism diagnostic on the CPU/GPU subset (QDQ code flips at the candidate's own sites)
  analyze
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / "runs/olmoe_fixed_route_20261015"
SEQ = 128
T_PRED = SEQ - 1
B_FIRST, B_WINDOWS = 98304, 30                     # validation tokens of the new windows (B)
MAGS = {"m23": 2.0 ** -23, "m20": 2.0 ** -20}
SEEDS = (0, 1, 2)
PERTURBED = [f"{m}s{s}" for m in MAGS for s in SEEDS]
BOOT, BLOCK = 2000, 3


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_gpu_subset(cands):
    """Fixed before any result: per (expert, mode) the same-resource pair gate=int4 vs up=int4 (single projection)."""
    out = []
    for c in cands:
        f = c["formats"]
        if c["set"] in ("gate_proj", "up_proj") and f[c["set"]][0 if c["mode"] != "a" else 1] == "int4":
            out.append(c["name"])
    return out


# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    cands = jload(S1 / "candidates.json")
    sub = cpu_gpu_subset(cands)
    files = {k: S1 / k for k in ("candidates.json", "table.json", "tables.json", "ranges.json", "truth_diagnostic.json",
                                 "tokens_diagnostic.npy", "routes_manifest.json", "plan.json", "freeze.json")}
    jdump(out / "plan.json", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": "H1 BC stably mis-ranks; H2 the measured ranking is sensitive to numerical conditions; H3 both, by "
                    "expert / mode",
        "background_verified": {
            "1": "9 experts (L0, L8, L15 x high / mid / low traffic), 40 candidates per expert and mode (gate, up, down x 5 "
                 "formats + gate+up 25 combinations), modes w, a, wa: 1080 candidates - confirmed",
            "2": "2583 same-resource pairs (same expert, mode, M and calibration BOPs), 461 decisive (>= 95% of 2000 iid "
                 "window bootstrap draws, seed 0, same sign), BC right on 262 (56.8%), no BC ties - recomputed exactly "
                 "from truth_diagnostic.json",
            "3": "pairs share candidates and windows: not independent - confirmed (the original intervals are window "
                 "bootstraps)",
            "4": "the strong amplification of tiny FP differences was observed on whole quantized configurations "
                 "(runs/olmoe_search_final_20261016: about 10^3 code flips at layer 0 cascading over every quantized "
                 "expert); single-point candidates quantize one expert - not assumed to behave the same",
            "5": "stage 1 truth ran on the GPU with the old int activation QDQ (division by a 0-dim CPU scale); the "
                 "correction (runs/fix_integration_20261016) changes only int activation formats (int4 / int6 / int8 "
                 "in modes a, wa); weights and posit / f16 / bf16 paths are unchanged"},
        "inputs_sha256": {k: sha256(v) for k, v in files.items()},
        "scripts_sha256": {s: sha256(HERE / s) for s in ("run_olmoe_bc_numerics.py", "run_olmoe_fixed_route.py",
                                                          "olmoe_common.py", "qwen3_swiglu_common.py")},
        "scope": "OLMoE fixed route only; FP32 route ids / mixing weights fixed per window; BC scores, calibration, ranges, "
                 "candidates, resource model fixed; no BC change, no search, no dynamic route",
        "candidates": "all 1080 candidates and all 2583 same-resource pairs (no subset for the main analysis)",
        "single_point": "each candidate quantizes only its own expert projection(s) (stage 1's truth semantics: weights "
                        "per output channel, x and y QDQ of each listed projection), the rest FP32; replay from the "
                        "expert's layer with the FP32 hidden state of that layer's input under the same condition",
        "conditions": {
            "quantizer": {"old": "stage 1's GPU path (reproduction and the old-vs-corrected comparison only)",
                          "corrected": "every main condition"},
            "perturbation": {"U": "none", **{f"{m}s{s}": f"embedding output x (1 + {v:g} s), s = +-1 per element, numpy "
                                                             f"seed {s}" for m, v in MAGS.items() for s in SEEDS}},
            "magnitudes": "2^-23 (one FP32 ulp, relative) and 2^-20 (about the largest measured CPU/GPU FP32 layer-output "
                          "difference, 1e-6 relative L2, runs/olmoe_search_final_20261016 A4); fixed, never increased",
            "shared": "within a condition every candidate and the FP32 baseline start from the same perturbed embedding; "
                      "routes, calibration and ranges are not recomputed; measured perturbation size recorded (max abs, "
                      "relative L2 at the embedding and at the input of layers 0, 8, 15)",
            "weight_only": "no activation QDQ is added to w candidates"},
        "data": {"A": "stage 1's 30 diagnostic windows (validation tokens 1280-5119): reproduction and mechanism",
                 "B": f"{B_WINDOWS} new validation windows from token {B_FIRST}, character spans disjoint from every "
                      "earlier OLMoE window set; read only after A is analysed and the method frozen (freeze_B.json)",
                 "pairing": "the same windows, candidates and seeds are paired; seeds are not extra data windows"},
        "runs": {"A": ["old quantizer on the layer-8 and layer-15 candidates (720): must equal stage 1's stored truth "
                       "bitwise; then the stored truth is the old-quantizer baseline of all 1080",
                       "corrected U (all 1080); corrected U again on the layer-8 and layer-15 candidates (determinism)",
                       "corrected, 6 perturbed conditions (all 1080)"],
                 "B": ["corrected U and the 6 perturbed conditions (all 1080)"],
                 "cpu_gpu": "CPU (OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE, 12 threads) corrected U on the fixed subset: per "
                            "expert and mode the single gate=int4 and up=int4 candidates (54 candidates, 27 same-resource "
                            "pairs) on A",
                 "mechanism": "on the subset's a / wa candidates: activation QDQ code flips at the candidate's own sites "
                              "(GPU m23s0 vs GPU U; CPU U vs GPU U; identical inputs into the CPU and GPU quantizers), "
                              "the layer output and the final loss - local diagnostic only, not a main result"},
        "measures": {"D_AB": "per window and condition: (summed CE of A - summed CE of B) / 127, signed",
                     "BC_direction": "sign(C_A - C_B); BC tie (equal scores): excluded, counted",
                     "per_pair": ["mean direction and decisiveness per condition", "flip of the mean direction vs U per "
                                  "perturbed condition", "share of windows whose per-window sign differs from U",
                                  "BC agreement per condition", "SD over seeds of the mean difference / |mean under U|",
                                  "A vs B", "CPU vs GPU (subset)"]},
        "decisive": {"primary": f"block bootstrap: {30 // BLOCK} non-overlapping blocks of {BLOCK} consecutive windows "
                                f"(consecutive text), {BOOT} draws, seed 1, >= 95% of draws with the sign of the mean",
                     "secondary": f"stage 1's iid window bootstrap ({BOOT} draws, seed 0), for comparability"},
        "classification": {
            "stable_right / stable_wrong": "decisive under corrected U (primary rule) and the mean direction is the same "
                                           "in all 6 perturbed conditions; right / wrong = BC agrees / disagrees",
            "sensitive": "decisive under U but the mean direction flips in >= 1 perturbed condition; or not decisive under "
                         "U but decisive with opposite directions in two conditions (U and the 6 perturbed)",
            "undetermined": "everything else (difference too small for these windows)",
            "applied": "to A and to B separately with the same rule; nothing is tuned after A"},
        "primary_endpoints": ["E1: BC agreement on the fixed 461 pairs under corrected U, with a window-block bootstrap "
                              "95% CI (pairs re-evaluated jointly on each resample)",
                              "E2: classes of the 199 originally wrong pairs among the 461 (A), and the same pairs' "
                              "classes on B",
                              "E3: E2 by mode (w, a, wa)"],
        "secondary": ["all 2583 pairs; the new decisive set under corrected U; per expert, mode and pair class",
                      "old vs corrected quantizer: candidate dNLL change, pair order changes, BC agreement change",
                      "CPU vs GPU on the subset"],
        "statistics": "no independent-Bernoulli intervals over pairs; intervals only by window-block resampling (pairs "
                      "jointly); class shares are descriptive counts, reported per expert (9 experts do not represent all "
                      "experts); no multiplicity correction (descriptive endpoints, none tested for significance)",
        "exploratory": "anything not listed above (e.g. per-format patterns, links to BC components) is labelled "
                       "exploratory",
        "cpu_gpu_subset": sub,
        "limits": {"gpu_time_cap_h": 5.0, "stop": "if a full condition takes > 30 min, B's perturbed conditions are cut "
                                                   "to seed 0 of each magnitude and the cut is reported"}})
    log(f"plan written; subset {len(sub)} candidates")


# --------------------------------------------------------------------------- #
# windows B
# --------------------------------------------------------------------------- #
def stage_windows(args, out, log):
    import olmoe_common as oc
    d = out / "B"
    d.mkdir(exist_ok=True)
    if (d / "tokens.npy").exists():
        raise SystemExit("exists")
    s1 = jload(S1 / "plan.json")["data"]
    prev = [(s1[n]["split"], s1[n]["char_spans"]) for n in ("calibration", "diagnostic", "check")]
    for src, key in ((HERE / "runs/olmoe_multipoint_20261015/plan.json", "eval_windows"),
                     (HERE / "runs/solver_regression_20261016/plan.json", "olmoe_eval")):
        p = jload(src)
        segs = p[key]["segments"] if key == "eval_windows" else p[key]["windows"]["segments"]
        prev += [(v["split"], [v["char_spans"]]) for v in segs.values()]
    w = jload(HERE / "runs/olmoe_search_final_20261016/C/windows.json")["segments"]
    prev += [(v["split"], [v["char_spans"]]) for v in w.values()]
    t, spans, ntok, _ = oc.windows("validation", B_FIRST, B_WINDOWS, SEQ)
    for split, sp in prev:
        if split == "validation" and spans[0][0] < sp[-1][1] and spans[-1][1] > sp[0][0]:
            raise SystemExit("overlap with an earlier window set")
    np.save(d / "tokens.npy", t)
    jdump(d / "windows.json", {"split": "validation", "tokens": [B_FIRST, B_FIRST + B_WINDOWS * SEQ],
                               "char_spans": [spans[0][0], spans[-1][1]], "checked_against": len(prev),
                               "sha256": sha256(d / "tokens.npy")})
    log(f"B windows chars {spans[0][0]}-{spans[-1][1]}")


def stage_capture(args, out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, head_ce
    gpu_exact(args.device)
    d = out / "B"
    (d / "routes").mkdir(exist_ok=True)
    if (d / "routes_manifest.json").exists():
        return
    cfg, ck = oc.config(), oc.Checkpoint()
    toks_np = np.load(d / "tokens.npy")
    tk = torch.from_numpy(toks_np).to(args.device)
    g = ck.global_(args.device, torch.float32)
    cos, sin = oc.rope(cfg, SEQ, args.device, torch.float32)
    idxs, rws = [], []
    with torch.no_grad():
        h = F.embedding(tk, g["embed"])
        for l in range(cfg.num_hidden_layers):
            h, idx, rw = oc.layer_dynamic(ck.layer(l, args.device, torch.float32), h, cfg, cos, sin)
            idxs.append(idx.view(len(tk), SEQ, -1).to(torch.int16).cpu().numpy())
            rws.append(rw.view(len(tk), SEQ, -1).float().cpu().numpy())
        ce = head_ce(g, h, tk, cfg).cpu().numpy()
    ia, ra = np.stack(idxs, 1), np.stack(rws, 1)
    hashes = []
    for k in range(len(tk)):
        np.savez(d / "routes" / f"new_{k:03d}.npz", ids=ia[k], weights=ra[k], tokens=toks_np[k])
        hashes.append(oc.route_hash(ia[k], ra[k]))
    jdump(d / "routes_manifest.json", {"new": {"windows": len(tk), "hash": hashes, "tokens_sha256": sha256(d / "tokens.npy"),
                                               "summed_ce_dynamic": ce.tolist()}})
    log("B routes captured")


# --------------------------------------------------------------------------- #
# truth
# --------------------------------------------------------------------------- #
def _old_int_act(x, scale, zp, bits):
    """stage 1's int activation QDQ (scale / zero-point as 0-dim CPU tensors)."""
    import torch
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    s = torch.tensor(np.float32(scale), dtype=x.dtype)
    z = torch.tensor(np.float32(zp), dtype=x.dtype)
    return (torch.clamp(torch.round(x / s + z), qmin, qmax) - z) * s


def quantizers(work, dev, quant):
    from qwen3_swiglu_common import Quantizers
    import eval_random_mixed_precision as erm
    q = Quantizers(work, formats=("p6e1", "p8e1"))
    qa = copy.copy(q)
    qa.posit = {}
    for f_, lut in q.posit.items():
        qa.posit[f_] = copy.copy(lut)
        qa.posit[f_].t_values, qa.posit[f_].t_bounds = lut.t_values.to(dev), lut.t_bounds.to(dev)
    if quant == "old":
        base_act = qa.act

        def act(x, fmt, rng=None):
            if fmt.startswith("int"):
                bits = int(fmt[3:])
                scale, zp = erm.int_qparams(float(rng[0]), float(rng[1]), bits)
                return _old_int_act(x, scale, zp, bits)
            return base_act(x, fmt, rng)
        qa.act = act
    return q, qa


def set_data(out, name, dev):
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    cfg = oc.config()
    if name == "A":
        routes, _, _ = load_routes(S1, "diagnostic", cfg, dev)
        return np.load(S1 / "tokens_diagnostic.npy"), routes
    routes, _, _ = load_routes(out / "B", "new", cfg, dev)
    return np.load(out / "B" / "tokens.npy"), routes


def run_truth(out, set_name, cond, quant, cands, dev, log, label, record_sites=None, work_name="work"):
    """stage 1's truth stage under (quantizer, perturbation); per-window summed CE of each candidate minus the condition's
    own FP32 baseline. record_sites(name, site, tensor_in, tensor_out) captures the candidate's own QDQ sites."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import head_ce
    cfg, ck = oc.config(), oc.Checkpoint()
    ranges = jload(S1 / "ranges.json")
    work = out / work_name                     # one oracle build directory per concurrent process
    work.mkdir(exist_ok=True)
    q, qa = quantizers(work, dev, quant)
    toks_np, routes = set_data(out, set_name, dev)
    toks = torch.from_numpy(toks_np).to(dev)
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    t0 = time.time()
    pert = {}
    with torch.no_grad():
        h0 = F.embedding(toks, g["embed"])
        hp = h0
        if cond != "U":
            m = MAGS[cond[:3]]
            s = np.random.default_rng(int(cond[-1])).choice([-1.0, 1.0], size=tuple(h0.shape)).astype(np.float32)
            hp = h0 * (1 + torch.from_numpy(s).to(dev) * m)
            pert["embedding"] = {"max_abs": float((hp - h0).abs().max()), "rel_l2": float((hp - h0).norm() / h0.norm())}
        hs, hu = [hp], [h0]
        for l in range(cfg.num_hidden_layers):
            w = ck.layer(l, dev, torch.float32)
            hs.append(oc.layer_fixed(w, hs[-1], cfg, cos, sin, *routes[l], l)[0])
            if cond != "U":
                hu.append(oc.layer_fixed(w, hu[-1], cfg, cos, sin, *routes[l], l)[0])
            del w
        base = head_ce(g, hs[-1], toks, cfg).cpu().numpy()
        if cond != "U":
            base_u = head_ce(g, hu[-1], toks, cfg).cpu().numpy()
            for L in (0, 8, 15):
                pert[f"layer{L}_input"] = {"max_abs": float((hs[L] - hu[L]).abs().max()),
                                           "rel_l2": float((hs[L] - hu[L]).norm() / hu[L].norm())}
            pert["fp32_base_change_per_token"] = {"max_abs": float(np.abs(base - base_u).max() / T_PRED),
                                                  "mean": float((base - base_u).mean() / T_PRED)}
            del hu
        rows = {}
        for L, mode in sorted({(c["layer"], c["mode"]) for c in cands}):   # <= 120 candidates on the device, as stage 1
            group = [c for c in cands if c["layer"] == L and c["mode"] == mode]
            H = {c["name"]: hs[L].clone() for c in group}
            for l in range(L, cfg.num_hidden_layers):
                w = ck.layer(l, dev, torch.float32)
                for c in group:
                    if l == L:
                        e = c["expert"]
                        saved = {}
                        hk = oc.ExpertHooks()
                        for p, (wf, af) in c["formats"].items():
                            if wf != "fp32":
                                saved[p] = w[p][e].clone()
                                w[p][e] = q.weight(w[p][e].cpu(), wf).to(dev)
                            if af != "fp32":
                                for sl in ("x", "y"):
                                    r = ranges[f"{c['node']}.{p}"][sl]
                                    if record_sites is None:
                                        hk.act[(l, e, p, sl)] = (lambda t, af=af, r=r: qa.act(t, af, r))
                                    else:
                                        def fn(t, af=af, r=r, nm=c["name"], site=f"{p}.{sl}"):
                                            y = qa.act(t, af, r)
                                            record_sites(nm, site, t, y)
                                            return y
                                        hk.act[(l, e, p, sl)] = fn
                        H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *routes[l], l, hk)[0]
                        for p, v in saved.items():
                            w[p][e] = v
                    else:
                        H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *routes[l], l)[0]
                del w
            for c in group:
                rows[c["name"]] = (head_ce(g, H.pop(c["name"]), toks, cfg).cpu().numpy() - base).tolist()
            log(f"{label}: layer {L} {mode}: {len(group)} candidates ({time.time() - t0:.0f}s)")
    return {"rows": rows, "base": base.tolist(), "perturbation": pert, "seconds": time.time() - t0}


def stage_truth(args, out, log):
    import torch
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact(args.device)
    if args.set == "B" and not (out / "B" / "freeze_B.json").exists():
        raise SystemExit("B is read only after freeze_B.json")
    d = out / args.set
    d.mkdir(exist_ok=True)
    tag = f"{args.cond}_{args.quant}_{args.device.split(':')[0]}" + (f"_{args.tag}" if args.tag else "")
    f = d / f"truth_{tag}.json"
    if f.exists():
        raise SystemExit(f"{f.name} exists")
    cands = jload(S1 / "candidates.json")
    if args.layers:
        cands = [c for c in cands if c["layer"] in {int(x) for x in args.layers.split(",")}]
    if args.subset:
        sub = set(jload(out / "plan.json")["cpu_gpu_subset"])
        cands = [c for c in cands if c["name"] in sub]
    r = run_truth(out, args.set, args.cond, args.quant, cands, args.device, log, tag)
    import olmoe_common  # noqa: F401
    r.update(set=args.set, cond=args.cond, quant=args.quant, device=args.device, candidates=len(cands),
             torch=torch.__version__, threads=torch.get_num_threads(),
             tf32=bool(torch.backends.cuda.matmul.allow_tf32), script_sha256=sha256(HERE / "run_olmoe_bc_numerics.py"))
    jdump(f, r)
    log(f"{tag}: {len(cands)} candidates in {r['seconds']:.0f}s; perturbation {r['perturbation']}")


def stage_mech(args, out, log):
    """QDQ code flips at the subset's a / wa candidates' own sites: GPU m23s0 vs GPU U, CPU U vs GPU U, and identical
    inputs (the CPU's) into the GPU quantizer."""
    import torch
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact("cuda")
    sub = set(jload(out / "plan.json")["cpu_gpu_subset"])
    cands = [c for c in jload(S1 / "candidates.json") if c["name"] in sub and c["mode"] in ("a", "wa")]
    rec = {}

    def recorder(key):
        def f(nm, site, t, y):
            rec.setdefault(key, {}).setdefault(nm, {})[site] = (t.detach().cpu().clone(), y.detach().cpu().clone())
        return f
    runs = {}
    for key, cond, dev in (("gpu_U", "U", "cuda"), ("gpu_m23s0", "m23s0", "cuda"), ("cpu_U", "U", "cpu")):
        runs[key] = run_truth(out, "A", cond, "corrected", cands, dev, log, f"mech {key}", recorder(key), "work_mech")
    _, qa_g = quantizers(out / "work_mech", "cuda", "corrected")
    ranges = jload(S1 / "ranges.json")
    res = {}
    for c in cands:
        nm = c["name"]
        r = {}
        for site in rec["gpu_U"][nm]:
            xg, yg = rec["gpu_U"][nm][site]
            xp, yp = rec["gpu_m23s0"][nm][site]
            xc, yc = rec["cpu_U"][nm][site]
            p, sl = site.split(".")
            af = c["formats"][p][1]
            yg_on_c = qa_g.act(xc.cuda(), af, ranges[f"{c['node']}.{p}"][sl]).cpu()
            v = lambda a, b: float((a.view(torch.int32) != b.view(torch.int32)).float().mean()) if a.numel() else 0.0
            r[site] = {"format": af, "elements": int(xg.numel()),
                       "perturbed_vs_U": {"input_rel_l2": float((xp - xg).norm() / max(xg.norm(), 1e-30)),
                                          "output_mismatch_share": v(yp, yg)},
                       "cpu_vs_gpu": {"input_bitwise_equal_share": 1 - v(xc, xg), "output_mismatch_share": v(yc, yg)},
                       "same_input_gpu_vs_cpu_quantizer_mismatch": v(yg_on_c, yc)}
        r["dnll_window_max_abs_diff"] = {
            "perturbed_vs_U": float(np.abs(np.array(runs["gpu_m23s0"]["rows"][nm]) - np.array(runs["gpu_U"]["rows"][nm])).max() / T_PRED),
            "cpu_vs_gpu": float(np.abs(np.array(runs["cpu_U"]["rows"][nm]) - np.array(runs["gpu_U"]["rows"][nm])).max() / T_PRED)}
        res[nm] = r
    jdump(out / "A" / "mech.json", {"candidates": res, "runs": {k: {"rows": v["rows"], "perturbation": v["perturbation"]}
                                                                for k, v in runs.items()}})
    log(f"mech: {len(res)} candidates")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def pair_list(cands):
    from run_qwen35_bc import _pair_class
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["node"], c["mode"]), []).append(i)
    P = []
    for (node, mode), ix in groups.items():
        for a, i in enumerate(ix):
            for j in ix[a + 1:]:
                ci, cj = cands[i], cands[j]
                if ci["M_bytes"] == cj["M_bytes"] and ci["BOPs_calibration"] == cj["BOPs_calibration"]:
                    P.append({"i": i, "j": j, "node": node, "mode": mode, "coverage": ci["coverage"],
                              "class": _pair_class(ci, cj)})
    return P


def draws(nw, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "iid":
        return rng.integers(0, nw, (BOOT, nw))
    nb = nw // BLOCK
    b = rng.integers(0, nb, (BOOT, nb))
    return (b[:, :, None] * BLOCK + np.arange(BLOCK)[None, None, :]).reshape(BOOT, nb * BLOCK)


def pair_stats(D, P, dr):
    """mean difference, decisiveness (>= 95% of draws with the mean's sign) for every pair; D [cand, window]."""
    I, J = np.array([p["i"] for p in P]), np.array([p["j"] for p in P])
    d = D[I] - D[J]
    m = d.mean(1)
    Db = np.stack([d[:, b].mean(1) for b in dr], 1)
    agree = ((Db * np.sign(m)[:, None]) > 0).mean(1)
    return d, m, agree >= 0.95, Db


def stage_analyze(args, out, log):
    cands = jload(S1 / "candidates.json")
    names = [c["name"] for c in cands]
    bc = np.array([c["bc"] for c in cands])
    P = pair_list(cands)
    I, J = np.array([p["i"] for p in P]), np.array([p["j"] for p in P])
    bcdir = np.sign(bc[I] - bc[J])
    res = {"pairs": len(P), "bc_ties": int((bcdir == 0).sum()), "sets": {}}

    def load(set_name, tag):
        f = out / set_name / f"truth_{tag}.json"
        if not f.exists():
            return None
        r = jload(f)
        return r, np.array([r["rows"][n] for n in names]) / T_PRED if len(r["rows"]) == len(names) else None
    # stage 1 stored (old quantizer) and the fixed 461
    st = jload(S1 / "truth_diagnostic.json")
    D_old = np.array([st["rows"][n] for n in names]) / T_PRED
    _, m_old, dec_old, _ = pair_stats(D_old, P, draws(30, "iid", 0))
    fixed = np.where(dec_old)[0]
    orig_right = (np.sign(m_old) == bcdir)
    res["original"] = {"decisive": int(dec_old.sum()), "right": int(orig_right[fixed].sum())}
    # reproduction of the old quantizer
    rep = load("A", "U_old_cuda")
    if rep:
        r = rep[0]
        diffs = [np.abs(np.array(r["rows"][n]) - np.array(st["rows"][n])).max() for n in r["rows"]]
        res["reproduction_old_quantizer"] = {"candidates": len(diffs), "max_abs_summed_ce_diff": float(max(diffs)),
                                             "bitwise_equal": int(sum(x == 0 for x in diffs))}
    # determinism
    u = load("A", "U_corrected_cuda")
    u2 = load("A", "U_corrected_cuda_rep2")
    if u and u2:
        res["determinism"] = {"candidates": len(u2[0]["rows"]), "max_abs": float(max(
            np.abs(np.array(u2[0]["rows"][n]) - np.array(u[0]["rows"][n])).max() for n in u2[0]["rows"]))}
    for set_name in ("A", "B"):
        U = load(set_name, "U_corrected_cuda")
        if not U or U[1] is None:
            continue
        D_U = U[1]
        nw = D_U.shape[1]
        drb, dri = draws(nw, "block", 1), draws(nw, "iid", 0)
        dU, mU, decU, DbU = pair_stats(D_U, P, drb)
        _, _, decU_iid, _ = pair_stats(D_U, P, dri)
        conds = {c: load(set_name, f"{c}_corrected_cuda") for c in PERTURBED}
        conds = {c: v for c, v in conds.items() if v and v[1] is not None}
        S = {"windows": nw, "conditions": ["U"] + list(conds), "fp32_base_nll": float(np.mean(U[0]["base"]) / T_PRED)}
        M = {"U": mU}
        DEC = {"U": decU}
        WF = {}
        for c, (r, Dc) in conds.items():
            dc, mc, decc, _ = pair_stats(Dc, P, drb)
            M[c], DEC[c] = mc, decc
            WF[c] = (np.sign(dc) != np.sign(dU)).mean(1)          # share of windows whose per-window sign differs
            S.setdefault("perturbation", {})[c] = r["perturbation"]
        allc = list(M)
        flips = np.stack([np.sign(M[c]) != np.sign(mU) for c in conds], 1) if conds else np.zeros((len(P), 0), bool)
        anyflip = flips.any(1)
        opp = np.zeros(len(P), bool)
        for a_ in allc:
            for b_ in allc:
                opp |= DEC[a_] & DEC[b_] & (np.sign(M[a_]) != np.sign(M[b_]))
        cls = np.where(decU & ~anyflip, np.where(np.sign(mU) == bcdir, "stable_right", "stable_wrong"),
                       np.where((decU & anyflip) | opp, "sensitive", "undetermined"))
        cls = np.where(bcdir == 0, "bc_tie", cls)
        S["class_counts_all"] = {k: int((cls == k).sum()) for k in np.unique(cls)}
        # endpoints
        def acc_ci(sel, D, dr):
            if len(sel) == 0:
                return None
            d = D[I[sel]] - D[J[sel]]
            a = float((np.sign(d.mean(1)) == bcdir[sel]).mean())
            b = np.array([(np.sign(d[:, x].mean(1)) == bcdir[sel]).mean() for x in dr])
            return {"pairs": int(len(sel)), "agreement": a, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        S["E1_fixed461_agreement"] = {"U": acc_ci(fixed, D_U, drb), **{c: acc_ci(fixed, conds[c][1], drb) for c in conds}}
        S["all_pairs_agreement"] = {"U": acc_ci(np.arange(len(P)), D_U, drb), **{c: acc_ci(np.arange(len(P)), conds[c][1], drb) for c in conds}}
        newdec = np.where(decU)[0]
        newdec_iid = np.where(decU_iid)[0]
        S["new_decisive_set"] = {"block": acc_ci(newdec, D_U, drb), "iid": acc_ci(newdec_iid, D_U, dri),
                                 "overlap_with_fixed461": int(len(set(newdec) & set(fixed))),
                                 "iid_overlap_with_fixed461": int(len(set(newdec_iid) & set(fixed)))}
        wrong = fixed[~orig_right[fixed]]
        right = fixed[orig_right[fixed]]

        def counts(sel):
            return {k: int((cls[sel] == k).sum()) for k in ("stable_right", "stable_wrong", "sensitive", "undetermined", "bc_tie")}
        S["E2_original_wrong_pairs"] = {"pairs": int(len(wrong)), **counts(wrong)}
        S["original_right_pairs"] = {"pairs": int(len(right)), **counts(right)}
        S["E3_by_mode"] = {mo: {"orig_wrong": counts(np.array([k for k in wrong if P[k]["mode"] == mo], int)),
                                "orig_right": counts(np.array([k for k in right if P[k]["mode"] == mo], int)),
                                "all": counts(np.array([k for k in range(len(P)) if P[k]["mode"] == mo], int))}
                           for mo in ("w", "a", "wa")}
        S["by_expert"] = {nd: {mo: counts(np.array([k for k in range(len(P)) if P[k]["node"] == nd and P[k]["mode"] == mo], int))
                               for mo in ("w", "a", "wa")} for nd in sorted({p["node"] for p in P})}
        S["by_class"] = {cl: counts(np.array([k for k in range(len(P)) if P[k]["class"] == cl], int))
                         for cl in sorted({p["class"] for p in P})}
        # perturbation size relative to the pair difference
        if conds:
            for mg in MAGS:
                cs = [c for c in conds if c.startswith(mg)]
                if len(cs) >= 2:
                    sd = np.stack([M[c] for c in cs], 1).std(1, ddof=1)
                    ratio = sd / np.maximum(np.abs(mU), 1e-30)
                    shift = np.abs(np.stack([M[c] for c in cs], 1) - mU[:, None]).mean(1) / np.maximum(np.abs(mU), 1e-30)
                    S.setdefault("perturbation_vs_difference", {})[mg] = {
                        mo: {"median_sd_over_abs_mean": float(np.median(ratio[[p["mode"] == mo for p in P]])),
                             "median_mean_shift_over_abs_mean": float(np.median(shift[[p["mode"] == mo for p in P]])),
                             "mean_flip_share": float(flips[[p["mode"] == mo for p in P]][:, [list(conds).index(c) for c in cs]].mean()),
                             "window_flip_share_median": float(np.median(np.stack([WF[c] for c in cs], 1)[[p["mode"] == mo for p in P]].mean(1)))}
                        for mo in ("w", "a", "wa")}
        S["per_pair_file"] = f"pairs_{set_name}.csv"
        import csv
        with open(out / set_name / f"pairs_{set_name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "node", "mode", "class", "bc_dir", "fixed461", "orig_right", "mean_U", "decisive_U_block",
                        "decisive_U_iid", "flips", "class_new", *[f"mean_{c}" for c in conds]])
            for k, p in enumerate(P):
                w.writerow([names[p["i"]], names[p["j"]], p["node"], p["mode"], p["class"], int(bcdir[k]), int(dec_old[k]),
                            int(orig_right[k]), mU[k], int(decU[k]), int(decU_iid[k]), int(flips[k].sum()), cls[k],
                            *[M[c][k] for c in conds]])
        np.save(out / set_name / f"classes_{set_name}.npy", cls)
        res["sets"][set_name] = S
    # old vs corrected (A)
    if "A" in res["sets"]:
        D_U = load("A", "U_corrected_cuda")[1]
        ch = np.abs(D_U - D_old).max(1)
        _, mU, decU_iid, _ = pair_stats(D_U, P, draws(30, "iid", 0))
        res["old_vs_corrected"] = {
            "candidate_max_abs_dnll_change_by_mode": {mo: float(ch[[c["mode"] == mo for c in cands]].max()) for mo in ("w", "a", "wa")},
            "candidates_changed": {mo: int((ch[[c["mode"] == mo for c in cands]] > 0).sum()) for mo in ("w", "a", "wa")},
            "pair_mean_sign_flips_all": int((np.sign(mU) != np.sign(m_old)).sum()),
            "pair_mean_sign_flips_fixed461": int((np.sign(mU[fixed]) != np.sign(m_old[fixed])).sum()),
            "fixed461_agreement_old_vs_corrected": [float(orig_right[fixed].mean()), float((np.sign(mU[fixed]) == bcdir[fixed]).mean())],
            "decisive_iid_old_vs_corrected": [int(dec_old.sum()), int(decU_iid.sum())]}
    # A vs B
    if "A" in res["sets"] and "B" in res["sets"]:
        ca, cb = np.load(out / "A" / "classes_A.npy"), np.load(out / "B" / "classes_B.npy")
        wrong = fixed[~orig_right[fixed]]
        tab = {}
        for a_ in np.unique(ca):
            tab[a_] = {b_: int(((ca == a_) & (cb == b_)).sum()) for b_ in np.unique(cb)}
        res["A_vs_B_all_pairs"] = tab
        res["A_vs_B_original_wrong"] = {a_: {b_: int(((ca[wrong] == a_) & (cb[wrong] == b_)).sum()) for b_ in np.unique(cb)}
                                        for a_ in np.unique(ca[wrong])}
    # CPU / GPU subset
    cpu = out / "A" / "truth_U_corrected_cpu_subset.json"
    if cpu.exists():
        rc = jload(cpu)["rows"]
        ug = load("A", "U_corrected_cuda")[0]["rows"]
        sub = list(rc)
        diff = {n: float(np.abs(np.array(rc[n]) - np.array(ug[n])).max() / T_PRED) for n in sub}
        sp = [k for k, p in enumerate(P) if names[p["i"]] in rc and names[p["j"]] in rc]
        agree = []
        for k in sp:
            a_, b_ = names[P[k]["i"]], names[P[k]["j"]]
            dc = (np.array(rc[a_]) - np.array(rc[b_])).mean() / T_PRED
            dg = (np.array(ug[a_]) - np.array(ug[b_])).mean() / T_PRED
            agree.append({"pair": [a_, b_], "mode": P[k]["mode"], "cpu": float(dc), "gpu": float(dg), "same_sign": bool(np.sign(dc) == np.sign(dg))})
        res["cpu_gpu"] = {"candidates": len(sub), "max_window_abs_dnll_diff_by_mode": {
            mo: max([v for n, v in diff.items() if f"|{mo}|" in n] or [0.0]) for mo in ("w", "a", "wa")},
            "pairs": len(agree), "same_sign": sum(x["same_sign"] for x in agree),
            "same_sign_by_mode": {mo: [sum(x["same_sign"] for x in agree if x["mode"] == mo), sum(x["mode"] == mo for x in agree)] for mo in ("w", "a", "wa")},
            "detail": agree}
    jdump(out / "analysis.json", res)
    log(json.dumps({k: v for k, v in res.items() if k != "sets"}, default=str)[:3000])
    for s, S in res["sets"].items():
        log(f"{s}: E1 {S['E1_fixed461_agreement']['U']}, E2 {S['E2_original_wrong_pairs']}, classes {S['class_counts_all']}")


def stage_freeze_b(args, out, log):
    if (out / "B" / "freeze_B.json").exists():
        raise SystemExit("frozen")
    (out / "B").mkdir(exist_ok=True)
    jdump(out / "B" / "freeze_B.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "plan_sha256": sha256(out / "plan.json"),
                                        "script_sha256": sha256(HERE / "run_olmoe_bc_numerics.py"),
                                        "analysis_A_sha256": sha256(out / "analysis.json"),
                                        "note": "method, perturbations and classification frozen; B read after this"})
    log("B frozen")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--cond", default="U")
    ap.add_argument("--quant", default="corrected", choices=("corrected", "old"))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("truth", "mech", "capture"):
        import torch
        torch.set_num_threads(args.threads)
    t0 = time.time()
    tag = "_".join(x for x in (args.stage, args.set if args.stage == "truth" else None,
                                args.cond if args.stage == "truth" else None,
                                args.quant if args.stage == "truth" else None, args.tag) if x)
    (out / "logs").mkdir(exist_ok=True)
    logf = open(out / "logs" / f"{tag}.log", "a")

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
        tf = out / "timing.json"
        timing = jload(tf) if tf.exists() else {"runs": []}
        timing["runs"].append({"tag": tag, "device": args.device, "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
        jdump(tf, timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
