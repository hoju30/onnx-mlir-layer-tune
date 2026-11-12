#!/usr/bin/env python3
"""OLMoE-1B-7B, fixed route, stage 2: whole-model expert configurations chosen by the unchanged automatic pipeline.

Two questions, pre-registered in plan.json:
  (1) different compression levels: does the BC total of the frontier configurations follow the measured damage?
  (2) equal resources: at the same budget, does BC give the bits to the right experts? The same MILP with the same
      budget is solved with BC and with two controls that keep the layer / projection structure but drop BC's
      expert-specific information: BC shuffled among the experts of each (layer, projection) group, and the group's
      mean BC scaled by each expert's routed-token count.

Scope: every expert projection (gate / up / down) of all 16 layers whose expert has calibration tokens; experts
never routed in the calibration windows have no statistics and stay at bf16 weights (exact for this checkpoint) and
FP32 activations, outside the search. Format grid GRID11 x GRID11, MILP -> 2-D Pareto -> normalized chord knee
(mp_auto.auto_select), R = max(M / M8, B / B8): M the stored weight bytes of the searched projections, B the BOPs with
the calibration routed rows. The route is the FP32 route of each window (olmoe_common), as in stage 1
(runs/olmoe_fixed_route_20261015, whose calibration windows, routes and conventions are reused read-only).

Stages (one --out-dir, resumable): plan, capture (evaluation routes), table (BC of every projection, PyTorch,
GPU), search (frontier, points, controls; freeze), truth (whole-configuration replay, GPU), cpu (the pre-registered
cross-check subset on the CPU), analyze.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import olmoe_common as oc  # noqa: F401  (transformers 5 first)
import numpy as np
import torch
import torch.nn.functional as F

from run_olmoe_fixed_route import gpu_exact, head_ce, jdump, jload, load_routes, sha256

HERE = Path(__file__).resolve().parent
STAGE1 = HERE / "runs/olmoe_fixed_route_20261015"
SEQ = 128
EVAL = {"V": ("validation", 131072, 32), "T": ("test", 131072, 32)}
CPU_SUBSET = list(range(0, 8)) + list(range(32, 40))       # 8 windows of each segment
POINT_TARGETS = {"p060": 0.6, "p080": 0.8, "knee": 1.0, "p125": 1.25, "p160": 1.6}
BUDGET_POINTS = ("p080", "knee", "p125")
CONTROLS = ("shuffle0", "shuffle1", "traffic")


# --------------------------------------------------------------------------- #
# quantizers on the device, identical to qwen3_swiglu_common.Quantizers
# --------------------------------------------------------------------------- #
class DevQuant:
    """Weights: per-output-channel int (run_w8_per_channel_check.per_channel_int's arithmetic: float32 division,
    float64 round-half-away, float32 product), Universal posit LUTs, f16 / bf16 casts. Activations: Quantizers.act."""

    def __init__(self, out, device):
        from qwen3_swiglu_common import Quantizers
        import copy
        self.cpu = Quantizers(out, formats=("p4e1", "p6e1", "p8e1"))
        for f, lut in self.cpu.posit.items():
            bad, n = lut.verify()
            if bad:
                raise SystemExit(f"posit LUT {f}: {bad} of {n} differ from Universal")
        self.dev = copy.copy(self.cpu)
        self.dev.posit = {}
        for f, lut in self.cpu.posit.items():
            self.dev.posit[f] = copy.copy(lut)
            self.dev.posit[f].t_values, self.dev.posit[f].t_bounds = lut.t_values.to(device), lut.t_bounds.to(device)

    def weight(self, w, fmt):
        """w [..., out, in] (any leading dims), per output channel."""
        if fmt == "fp32":
            return w
        if fmt.startswith("int"):
            b = int(fmt[3:])
            qmax, qmin = (1 << (b - 1)) - 1, -(1 << (b - 1))
            m = w.abs().amax(-1, keepdim=True)
            s = torch.where(m > 0, m / torch.tensor(qmax, dtype=torch.float32, device=w.device), torch.ones_like(m))
            t = (w / s).double()
            q = torch.clamp(torch.sign(t) * torch.floor(t.abs() + 0.5), qmin, qmax)
            return q.float() * s
        if fmt == "f16":
            return w.half().float()
        if fmt == "bf16":
            return w.bfloat16().float()
        return self.dev.posit[fmt](w)

    def act(self, t, fmt, rng):
        return self.dev.act(t, fmt, rng)


def verify_devquant(dq, ck, device, log):
    """Bitwise: the device weight quantizer against the CPU Quantizers on three expert projections."""
    res = []
    w = ck.layer(7, device, torch.float32)
    for p, e in (("gate_proj", 3), ("up_proj", 40), ("down_proj", 61)):
        W = w[p][e]
        for f in ("int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16"):
            a = dq.weight(W, f).cpu()
            b = dq.cpu.weight(W.cpu(), f)
            res.append({"proj": p, "expert": e, "format": f, "bitwise_equal": bool(torch.equal(a, b)),
                        "max_abs": float((a - b).abs().max())})
    ok = all(r["bitwise_equal"] for r in res)
    log(f"device quantizer vs CPU Quantizers: {sum(r['bitwise_equal'] for r in res)}/{len(res)} bitwise equal")
    return {"checks": res, "pass": ok}


# --------------------------------------------------------------------------- #
# plan, routes
# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    tok = oc.tokenizer()
    toks, segs = [], {}
    s1 = jload(STAGE1 / "plan.json")["data"]
    for k, (split, first, n) in EVAL.items():
        t, spans, ntok, nchar = oc.windows(split, first, n, SEQ, tok)
        toks.append(t)
        used = [s1[name]["char_spans"] for name in ("calibration", "diagnostic", "check") if s1[name]["split"] == split]
        for sp in used:
            if spans[0][0] < sp[-1][1] and spans[-1][1] > sp[0][0]:
                raise SystemExit(f"segment {k} overlaps an earlier window set")
        segs[k] = {"split": split, "tokens": [first, first + n * SEQ], "windows": n, "char_spans": [spans[0][0], spans[-1][1]],
                   "split_tokens": ntok}
    toks = np.concatenate(toks)
    np.save(out / "tokens_eval.npy", toks)
    jdump(out / "plan.json", {
        "questions": {"1": "different compression levels: does the BC total of the frontier configurations follow the "
                           "measured whole-model damage?",
                      "2": "equal resources: at the same MILP budget, does BC allocate the bits to the right experts "
                           "(compared with controls that drop BC's expert-specific information)?"},
        "model": {"repo": oc.REPO, "revision": oc.REVISION},
        "fixed_route": "as stage 1 (runs/olmoe_fixed_route_20261015): every layer uses the window's cached FP32 top-8 ids "
                       "and unnormalized routing weights as constants",
        "calibration": {"source": "stage 1 tokens_calibration.npy and its route caches (read only, hash-checked)",
                        "tokens_sha256": sha256(STAGE1 / "tokens_calibration.npy"),
                        "routes_manifest_sha256": sha256(STAGE1 / "routes_manifest.json")},
        "eval_windows": {"segments": segs, "seq": SEQ, "count": int(len(toks)), "sha256": sha256(out / "tokens_eval.npy"),
                         "tokenizer": "OLMoE's own, no special tokens, WikiText-2 raw joined with blank lines",
                         "disjoint": "character spans disjoint from stage 1's calibration, diagnostic and check windows",
                         "chosen": "fixed offsets before any result; never changed"},
        "scope": {"searched": "gate_proj, up_proj, down_proj of every (layer, expert) with >= 1 calibration routed token",
                  "unobserved": "experts with no calibration token: bf16 weights (exact) + FP32 activations, not searched, "
                                "their storage counted as fixed",
                  "fp32": "router, attention, embedding, LM head, norms"},
        "bc": "unchanged (stage 1 definitions: C = W + A, MoE routed-row counts, shared gate / up input with total "
              "gradient, per-consumer QDQ, QDQ before the routing-weight product); PyTorch, fixed route, FP32 GPU",
        "search": {"policy": "mp_auto.auto_select (MILP -> 2-D Pareto -> normalized chord knee), unchanged",
                   "grid": "GRID11 x GRID11", "samples": 17, "refinements": 2, "time_limit_s": args.time_limit,
                   "resources": "M = sum elements x weight bits / 8 over the searched projections; B = sum calibration "
                                "routed rows x elements x weight bits x activation bits; R = max(M / M8, B / B8) relative "
                                "to all-int8"},
        "test1_points": {"targets_x_R_knee": POINT_TARGETS,
                         "rule": "the frontier point (configuration different from the others) closest to target x R_knee "
                                 "on the target's side of the knee (ties: lower R); if that is the frontier's lowest-R "
                                 "endpoint or has BC = 0, the nearest point on that side instead"},
        "test2_controls": {"budgets": "the actual R of the test-1 points " + ", ".join(BUDGET_POINTS) + " (M <= R M8, "
                                      "B <= R B8: the BC point satisfies both)",
                           "shuffle<s>": "BC's per-option values permuted among the searched experts of each (layer, "
                                         "projection) group, numpy seed s (0, 1); same MILP, same budget",
                           "traffic": "per (layer, projection) group, the group's mean BC per option times the expert's "
                                      "routed rows / the group's mean routed rows; same MILP, same budget",
                           "comparison": "BC configuration vs each control at the same budget; resources reported; a "
                                         "comparison counts only if both configurations' M and B are within 1% of each other"},
        "truth": "per window summed next-token CE of the whole quantized configuration with the fixed FP32 route, "
                 "minus FP32's (same route); GPU FP32, TF32 off; 64 evaluation windows",
        "metrics": {"bootstrap": "2000 resamples stratified by segment (32 drawn with replacement in each), the same "
                                 "indices for every configuration and difference; seed 0",
                    "test1": "order of BC totals vs measured dNLL; adjacent paired differences and CIs; per segment",
                    "test2": "dNLL(BC) - dNLL(control) per budget, paired CI; BC better if the CI is below 0"},
        "cross_check": {"subset": "windows " + str(CPU_SUBSET), "configs": "FP32, every configuration at the knee "
                                                                           "budget (BC knee and its three controls)",
                        "device": "CPU replay of the same code; reported per window against the GPU"},
        "not_claimed": ["no native result for the full model (stage 1: FP32 27.7 GB, per-window graphs)",
                        "no hardware speed-up", "BC scale differs from dNLL: no absolute prediction error"],
        "seeds": {"bootstrap": 0, "shuffle": [0, 1]}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"plan: eval {segs}")


def stage_capture(args, out, log):
    """FP32 dynamic forward of the evaluation windows: route caches (stage 1's code path)."""
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    (out / "routes").mkdir(exist_ok=True)
    if (out / "routes_manifest.json").exists():
        return
    toks_np = np.load(out / "tokens_eval.npy")
    t0 = time.time()
    toks = torch.from_numpy(toks_np).to(dev)
    B = len(toks_np)
    g = ck.global_(dev, torch.float32)
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    idxs, rws = [], []
    with torch.no_grad():
        h = F.embedding(toks, g["embed"])
        for l in range(cfg.num_hidden_layers):
            w = ck.layer(l, dev, torch.float32)
            h, idx, rw = oc.layer_dynamic(w, h, cfg, cos, sin)
            idxs.append(idx.view(B, SEQ, -1).to(torch.int16).cpu().numpy())
            rws.append(rw.view(B, SEQ, -1).float().cpu().numpy())
            del w
        ce = head_ce(g, h, toks, cfg).cpu().numpy()
    ia, ra = np.stack(idxs, 1), np.stack(rws, 1)
    for k in range(B):
        np.savez(out / "routes" / f"eval_{k:03d}.npz", ids=ia[k], weights=ra[k], tokens=toks_np[k])
    jdump(out / "routes_manifest.json", {"eval": {"windows": B, "shape_ids": list(ia.shape), "dtype_ids": "int16",
                                                   "dtype_weights": "float32",
                                                   "hash": [oc.route_hash(ia[k], ra[k]) for k in range(B)],
                                                   "tokens_sha256": sha256(out / "tokens_eval.npy"),
                                                   "model_revision": oc.REVISION, "summed_ce_dynamic": ce.tolist(),
                                                   "seconds": time.time() - t0}})
    log(f"eval routes: {B} windows in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# BC table of every expert projection
# --------------------------------------------------------------------------- #
def stage_table(args, out, log):
    from run_native_bc_pipeline import QUANT
    gpu_exact(args.device)
    cfg = oc.config()
    ck = oc.Checkpoint()
    dev = args.device
    dq = DevQuant(out, dev)
    check = verify_devquant(dq, ck, dev, log)
    if not check["pass"]:
        raise SystemExit("device quantizer differs from the CPU quantizer")
    routes, ids_np, _ = load_routes(STAGE1, "calibration", cfg, dev)
    toks = torch.from_numpy(np.load(STAGE1 / "tokens_calibration.npy")).to(dev)
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    t0 = time.time()
    with torch.no_grad():
        hs = [F.embedding(toks, g["embed"])]
        for l in range(cfg.num_hidden_layers):
            hs.append(oc.layer_fixed(ck.layer(l, dev, torch.float32), hs[-1], cfg, cos, sin, *routes[l], l)[0])
    hT = hs[-1].clone().requires_grad_(True)
    head_ce(g, hT, toks, cfg).sum().backward()
    gn = hT.grad
    B_, C_, ranges, meta = {}, {}, {}, {}
    for l in reversed(range(cfg.num_hidden_layers)):
        w = ck.layer(l, dev, torch.float32)
        cap = {}
        hk = oc.ExpertHooks()

        def c(ll, e, p, s, t, cap=cap):
            t.retain_grad()
            cap[(e, p, s)] = t
        hk.capture = c
        hi = hs[l].clone().requires_grad_(True)
        o, _ = oc.layer_fixed(w, hi, cfg, cos, sin, *routes[l], l, hk)
        o.backward(gn)
        gn = hi.grad
        idx = routes[l][0]
        mask_all = F.one_hot(idx, num_classes=cfg.num_experts + 1).permute(2, 1, 0)
        with torch.no_grad():
            for e in range(cfg.num_experts):
                node = f"L{l}.E{e}"
                if (e, "gate_proj", "x") not in cap:
                    meta[node] = {"layer": l, "expert": e, "routed_rows": 0, "windows": 0}
                    continue
                kpos, tok = torch.where(mask_all[e])
                win = tok // SEQ
                wins = sorted(set(win.tolist()))
                n = len(tok)
                meta[node] = {"layer": l, "expert": e, "routed_rows": int(n), "windows": len(wins)}
                X, GX = cap[(e, "gate_proj", "x")].detach(), cap[(e, "gate_proj", "x")].grad
                taps = {("gate_proj", "x"): (X, GX), ("up_proj", "x"): (X, GX)}
                for p, s in (("gate_proj", "y"), ("up_proj", "y"), ("down_proj", "x"), ("down_proj", "y")):
                    t = cap[(e, p, s)]
                    taps[(p, s)] = (t.detach(), t.grad)
                for p in oc.PROJ:
                    name = f"{node}.{p}"
                    W = w[p][e]
                    Xp, _ = taps[(p, "x")]
                    _, GY = taps[(p, "y")]
                    G = torch.zeros(W.shape[0], dtype=torch.float64, device=dev)
                    for k in wins:
                        m = win == k
                        G += ((GY[m].double().t() @ Xp[m].double()) ** 2).sum(1)
                    B_[name] = {"fp32": 0.0}
                    for f in QUANT:
                        d2 = ((dq.weight(W, f) - W).double() ** 2).sum(1)
                        B_[name][f] = float((G / W.shape[1] * d2).sum())
                    rng_ = {s: [float(taps[(p, s)][0].min()), float(taps[(p, s)][0].max())] for s in ("x", "y")}
                    ranges[name] = rng_
                    C_[name] = {"fp32": 0.0}
                    gstats = {}
                    for s in ("x", "y"):
                        t, gt = taps[(p, s)]
                        gt = gt.double()
                        G2 = (gt ** 2).sum(0)
                        GS = torch.stack([gt[win == k].sum(0) for k in wins]).pow(2).sum(0)
                        gstats[s] = (t, G2, GS)
                    for f in QUANT:
                        cc = 0.0
                        for s in ("x", "y"):
                            t, G2, GS = gstats[s]
                            err = (dq.act(t, f, rng_[s]) - t).double()
                            mu = err.sum(0) / n
                            cc += float((G2 / n * ((err ** 2).sum(0) - n * mu ** 2)).sum() + (GS * mu ** 2).sum())
                        C_[name][f] = cc
        del w, cap
        log(f"layer {l}: {sum(1 for k in meta if k.startswith(f'L{l}.') and meta[k]['routed_rows'] > 0)} observed experts "
            f"({time.time() - t0:.0f}s)")
    nonfinite = [(n, f) for t in (B_, C_) for n, r in t.items() for f, v in r.items() if not np.isfinite(v)]
    jdump(out / "table.json", {"B": B_, "C": C_, "naming": "B = W, C = A; node L<l>.E<e>.<proj>",
                               "implementation": "PyTorch fixed-route FP32 (GPU); quantizers bitwise equal to the CPU ones"})
    jdump(out / "ranges.json", ranges)
    jdump(out / "table_meta.json", {"experts": meta, "nonfinite": nonfinite, "devquant_check": check,
                                    "seconds": time.time() - t0})
    log(f"table: {len(B_)} projections, {sum(1 for v in meta.values() if v['routed_rows'] == 0)} unobserved experts, "
        f"{len(nonfinite)} non-finite, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# search, points, controls, freeze
# --------------------------------------------------------------------------- #
def problem(out):
    table = jload(out / "table.json")
    meta = jload(out / "table_meta.json")["experts"]
    names = sorted(table["B"], key=lambda n: (int(n.split(".")[0][1:]), int(n.split(".")[1][1:]), n.split(".")[2]))
    nodes = [{"name": n, "weight_const": True} for n in names]
    elems = {n: 2048 * 1024 for n in names}
    rows = {n: meta[".".join(n.split(".")[:2])]["routed_rows"] for n in names}
    flops = {n: elems[n] * rows[n] for n in names}
    return table, meta, names, nodes, elems, rows, flops


def resources(fm, names, elems, flops):
    from htq_pareto_search import FORMAT_BITS
    M = sum(elems[n] * FORMAT_BITS[fm[n]["w"]] / 8 for n in names)
    B = sum(flops[n] * FORMAT_BITS[fm[n]["w"]] * FORMAT_BITS[fm[n]["a"]] for n in names)
    M8, B8 = sum(elems[n] for n in names), sum(64 * flops[n] for n in names)
    return {"M_bytes": M, "BOPs": B, "M_rel": M / M8, "B_rel": B / B8, "R": max(M / M8, B / B8)}


def nondominated(names, cost):
    """Per node, the (w, a) options not strictly dominated by another option of the same node in (cost, weight bits,
    weight bits x activation bits); the node's bytes and BOPs are these bits times positive per-node constants, so a
    dominated option is never needed by an optimal MILP solution (exact pruning; ties in all three are all kept)."""
    from htq_pareto_search import FORMAT_BITS
    from run_native_bc_pipeline import GRID11
    keep = {}
    for n in names:
        opts = [(w, a) for w in GRID11 for a in GRID11]
        o = np.array([(cost(n, w, a), FORMAT_BITS[w], FORMAT_BITS[w] * FORMAT_BITS[a]) for w, a in opts])
        keep[n] = {opts[i] for i in range(len(o))
                   if not np.any(np.all(o <= o[i], axis=1) & np.any(o < o[i], axis=1))}
    return keep


def pruned(cost, keep):
    return lambda n, w, a: cost(n, w, a) if (w, a) in keep[n] else None


def stage_search(args, out, log):
    from mp_auto import auto_select, config_hash
    from mp_milp import milp_select
    from run_native_bc_pipeline import GRID11
    from collections import Counter
    if (out / "freeze.json").exists():
        raise SystemExit("already frozen")
    plan = jload(out / "plan.json")
    tl = next((a["time_limit_s"] for a in reversed(plan.get("amendments", [])) if "time_limit_s" in a),
              plan["search"]["time_limit_s"])
    table, meta, names, nodes, elems, rows, flops = problem(out)
    bc = lambda n, w, a: table["B"][n][w] + table["C"][n][a]
    log(f"search: {len(names)} projections, {len(GRID11) ** 2} options each")
    t0 = time.time()
    raw = out / "selection_raw.json"
    keep_bc = nondominated(names, bc)
    if raw.exists():
        sel = jload(raw)
        slog = sel.pop("solver_log_")
        t_auto = sel.pop("seconds_")
        log("auto_select result reused from selection_raw.json")
    else:
        slog = []
        sel = auto_select(nodes, GRID11, GRID11, pruned(bc, keep_bc), elems, flops, samples=17, refinements=2,
                          time_limit=tl, solver_log=slog)
        t_auto = time.time() - t0
        jdump(raw, {**sel, "solver_log_": slog, "seconds_": t_auto,
                    "pruned_options": {"kept": sum(len(v) for v in keep_bc.values()), "of": 121 * len(names)}})
        log(f"auto_select: {len(slog)} solves in {t_auto:.0f}s")
    F_, K = sel["frontier"], sel["selected"]
    h = lambda f: config_hash({n: f["formats"][n] for n in names})
    rk, kh = K["resource"], h(K)
    rmin = min(f["resource"] for f in F_)
    picked, notes = {"knee": K}, {"knee": "the automatic knee"}
    for name, x in sorted(POINT_TARGETS.items(), key=lambda kv: abs(kv[1] - 1)):   # nearest targets first
        if name == "knee":
            continue
        used = {h(p) for p in picked.values()}
        side = [f for f in F_ if h(f) not in used and ((f["resource"] < rk) if x < 1 else (f["resource"] > rk))]
        p = min(side, key=lambda f: (abs(f["resource"] - x * rk), f["resource"]))
        notes[name] = f"closest to {x * rk:.4f}"
        if p["resource"] == rmin or p["objective"] == 0:
            p = min(side, key=lambda f: (abs(f["resource"] - rk), f["resource"]))
            notes[name] = "degenerate target; nearest on that side"
        picked[name] = p
    if len({h(p) for p in picked.values()}) != len(picked):
        raise SystemExit("two test-1 points are the same configuration")
    cfgs = {}
    (out / "configs").mkdir(exist_ok=True)
    for name, p in picked.items():
        fm = {n: p["formats"][n] for n in names}
        cfgs[f"bc_{name}"] = {"test": 1, "budget_point": name, "criterion": "BC", "formats": fm, "bc_total": p["objective"],
                              "frontier_R": p["resource"], "rule": notes[name], "solver_status": p.get("solver_status")}
    # test 2 controls at the budgets of three test-1 points
    groups = {}
    for n in names:
        groups.setdefault((n.split(".")[0], n.split(".")[2]), []).append(n)
    from run_native_bc_pipeline import GRID11 as G
    costs = {}
    for s in (0, 1):
        perm = {}
        rs = np.random.default_rng(s)
        for gk, ns in groups.items():
            pi = rs.permutation(len(ns))
            for i, n in enumerate(ns):
                perm[n] = ns[pi[i]]
        costs[f"shuffle{s}"] = (lambda perm: lambda n, w, a: bc(perm[n], w, a))(perm)
    mean_rows = {gk: np.mean([rows[n] for n in ns]) for gk, ns in groups.items()}
    gmean = {gk: {(w, a): float(np.mean([bc(n, w, a) for n in ns])) for w in G for a in G} for gk, ns in groups.items()}
    gof = {n: gk for gk, ns in groups.items() for n in ns}
    costs["traffic"] = lambda n, w, a: gmean[gof[n]][(w, a)] * rows[n] / mean_rows[gof[n]]
    ref_b = sum(elems[n] for n in names)
    ref_o = sum(64 * flops[n] for n in names)
    keep = {cn: nondominated(names, costs[cn]) for cn in CONTROLS}
    clog = []
    for bp in BUDGET_POINTS:
        R = resources(cfgs[f"bc_{bp}"]["formats"], names, elems, flops)["R"]
        for cn in CONTROLS:
            info = {}
            t1 = time.time()
            fm, obj, status = milp_select(nodes, G, G, pruned(costs[cn], keep[cn]), elems, flops, R * ref_b, R * ref_o,
                                          time_limit=tl, info=info)
            fm = {n: fm[n] for n in names}
            clog.append({"budget_point": bp, "control": cn, "R_budget": R, "status": status, "seconds": time.time() - t1, **info})
            cfgs[f"{cn}_{bp}"] = {"test": 2, "budget_point": bp, "criterion": cn, "formats": fm, "R_budget": R,
                                  "control_objective": obj, "bc_total": float(sum(bc(n, fm[n]["w"], fm[n]["a"]) for n in names)),
                                  "solver_status": status}
            log(f"control {cn} at {bp} (R {R:.4f}): status {status}, {time.time() - t1:.0f}s")
    for k, v in cfgs.items():
        v.update(resources(v["formats"], names, elems, flops))
        v["config_hash"] = config_hash(v["formats"])
        v["w_formats"] = dict(Counter(f["w"] for f in v["formats"].values()))
        v["a_formats"] = dict(Counter(f["a"] for f in v["formats"].values()))
        jdump(out / "configs" / f"{k}.json", v["formats"])
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "formats"} for k, v in cfgs.items()}
    jdump(out / "configs.json", summary)
    jdump(out / "selection.json", {**{k: sel[k] for k in ("policy", "bounds", "reference", "attempted_budgets", "failures",
                                                           "excluded_nonfinite_options", "samples", "refinements")},
                                   "frontier": [{k: f[k] for k in ("objective", "size", "bops", "resource", "search_budget",
                                                                    "solver_status", "origin")} for f in F_],
                                   "knee": {k: K[k] for k in ("objective", "size", "bops", "resource")},
                                   "knee_scores": sel["knee_scores"], "solver_log": slog, "control_log": clog,
                                   "seconds": {"auto_select": t_auto, "total": time.time() - t0}})
    jdump(out / "freeze.json", {"configs_sha256": {k: sha256(out / "configs" / f"{k}.json") for k in cfgs},
                                "configs_json_sha256": sha256(out / "configs.json"), "table_sha256": sha256(out / "table.json"),
                                "ranges_sha256": sha256(out / "ranges.json"), "tokens_eval_sha256": sha256(out / "tokens_eval.npy"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("frozen: " + "; ".join(f"{k} R {v['R']:.3f} (M {v['M_rel']:.3f}, B {v['B_rel']:.3f}) BC {v['bc_total']:.4g}"
                               for k, v in summary.items()))


def check_freeze(out):
    fz = jload(out / "freeze.json")
    for k, hsh in fz["configs_sha256"].items():
        if sha256(out / "configs" / f"{k}.json") != hsh:
            raise SystemExit(f"{k} changed after the freeze")
    if sha256(out / "table.json") != fz["table_sha256"] or sha256(out / "ranges.json") != fz["ranges_sha256"]:
        raise SystemExit("table / ranges changed after the freeze")
    return list(fz["configs_sha256"])


# --------------------------------------------------------------------------- #
# whole-configuration replay
# --------------------------------------------------------------------------- #
def replay(out, names_cfg, toks_np, routes, dev, log, label):
    """Per-window summed CE of FP32 (name 'fp32') and of each configuration, all layers' routes fixed; layer-major
    (one weight load per layer), configurations' hidden states kept on the device."""
    cfg = oc.config()
    ck = oc.Checkpoint()
    dq = DevQuant(out, dev)
    ranges = jload(out / "ranges.json")
    toks = torch.from_numpy(toks_np).to(dev)
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    fms = {n: (None if n == "fp32" else jload(out / "configs" / f"{n}.json")) for n in names_cfg}
    t0 = time.time()
    with torch.no_grad():
        h0 = F.embedding(toks, g["embed"])
        H = {n: h0.clone() for n in names_cfg}
        for l in range(cfg.num_hidden_layers):
            w = ck.layer(l, dev, torch.float32)
            for n in names_cfg:
                if fms[n] is None:
                    H[n] = oc.layer_fixed(w, H[n], cfg, cos, sin, *routes[l], l)[0]
                    continue
                wq = {k: v for k, v in w.items()}
                hk = oc.ExpertHooks()
                for p in oc.PROJ:
                    wp = w[p].clone()
                    for e in range(cfg.num_experts):
                        key = f"L{l}.E{e}.{p}"
                        f = fms[n][key] if key in fms[n] else {"w": "bf16", "a": "fp32"}   # unobserved: bf16 / FP32
                        if f["w"] != "fp32":
                            wp[e] = dq.weight(w[p][e], f["w"])
                        if f["a"] != "fp32":
                            for s in ("x", "y"):
                                hk.act[(l, e, p, s)] = (lambda t, a=f["a"], r=ranges[key][s]: dq.act(t, a, r))
                    wq[p] = wp
                H[n] = oc.layer_fixed(wq, H[n], cfg, cos, sin, *routes[l], l, hk)[0]
                del wq
            del w
            log(f"{label}: layer {l} done ({time.time() - t0:.0f}s)")
        return {n: head_ce(g, H[n], toks, cfg).cpu().numpy().tolist() for n in names_cfg}, time.time() - t0


def stage_truth(args, out, log):
    gpu_exact(args.device)
    names_all = check_freeze(out)
    f = out / "truth.json"
    done = jload(f) if f.exists() else {"summed_ce": {}, "seconds": 0.0}
    todo = [n for n in ["fp32"] + names_all if n not in done["summed_ce"]]
    if args.configs:
        todo = [n for n in todo if n in args.configs.split(",")]
    if not todo:
        return
    cfg = oc.config()
    rts, _, _ = load_routes(out, "eval", cfg, args.device)
    toks = np.load(out / "tokens_eval.npy")
    for i in range(0, len(todo), args.group):
        part = todo[i:i + args.group]
        ce, sec = replay(out, part, toks, rts, args.device, log, f"group {i // args.group}")
        done["summed_ce"].update(ce)
        done["seconds"] += sec
        jdump(f, done)
        log(f"truth: {part} ({sec:.0f}s)")


def stage_cpu(args, out, log):
    names_all = check_freeze(out)
    f = out / "truth_cpu.json"
    done = jload(f) if f.exists() else {"subset": CPU_SUBSET, "summed_ce": {}, "seconds": 0.0}
    sub = ["fp32"] + [n for n in names_all if n.endswith("_knee")]
    todo = [n for n in sub if n not in done["summed_ce"]]
    if args.configs:
        todo = [n for n in todo if n in args.configs.split(",")]
    cfg = oc.config()
    rts, _, _ = load_routes(out, "eval", cfg, "cpu")
    T = SEQ
    rows = torch.cat([torch.arange(k * T, (k + 1) * T) for k in CPU_SUBSET])
    rts = [(i[rows], w[rows]) for i, w in rts]
    toks = np.load(out / "tokens_eval.npy")[CPU_SUBSET]
    for n in todo:
        ce, sec = replay(out, [n], toks, rts, "cpu", log, f"cpu {n}")
        done["summed_ce"].update(ce)
        done["seconds"] += sec
        jdump(f, done)
        log(f"cpu: {n} ({sec:.0f}s)")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def stage_analyze(args, out, log):
    import csv
    plan = jload(out / "plan.json")
    cfgs = jload(out / "configs.json")
    tr = jload(out / "truth.json")["summed_ce"]
    T = SEQ - 1
    base = np.array(tr["fp32"])
    n = len(base)
    rng = np.random.default_rng(plan["metrics"]["bootstrap"] if isinstance(plan["metrics"].get("bootstrap"), int) else 0)
    draws = np.concatenate([rng.integers(0, 32, (2000, 32)), 32 + rng.integers(0, 32, (2000, 32))], 1)
    seg = {"V": np.arange(0, 32), "T": np.arange(32, 64)}

    def st(d, part=None):
        if part is None:
            b, m = d[draws].mean(1), d.mean()
        else:
            b = d[draws[:, :32] if part == "V" else draws[:, 32:]].mean(1)
            m = d[seg[part]].mean()
        return {"mean": float(m), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    D = {k: (np.array(tr[k]) - base) / T for k in cfgs if k in tr}
    res = {"fp32_nll": float(base.sum() / (n * T)), "fp32_ppl": float(np.exp(base.sum() / (n * T))), "configs": {}, "test1": {},
           "test2": {}, "cross_impl": {}}
    for k, v in cfgs.items():
        if k not in D:
            continue
        res["configs"][k] = {**{kk: v[kk] for kk in ("test", "budget_point", "criterion", "bc_total", "M_bytes", "BOPs", "M_rel",
                                                     "B_rel", "R", "w_formats", "a_formats")},
                             "nll": float(np.array(tr[k]).sum() / (n * T)), "ppl": float(np.exp(np.array(tr[k]).sum() / (n * T))),
                             "dnll": {"all": st(D[k]), "V": st(D[k], "V"), "T": st(D[k], "T")}}
    pts = [f"bc_{p}" for p in sorted(POINT_TARGETS, key=lambda p: POINT_TARGETS[p])]
    pts = [p for p in pts if p in D]
    bcs = [cfgs[p]["bc_total"] for p in pts]
    dn = [float(D[p].mean()) for p in pts]
    res["test1"] = {"points_by_R": pts, "R": [cfgs[p]["R"] for p in pts], "bc_total": bcs, "dnll": dn,
                    "bc_decreasing_with_R": bool(all(a > b for a, b in zip(bcs, bcs[1:]))),
                    "dnll_decreasing_with_R": bool(all(a > b for a, b in zip(dn, dn[1:]))),
                    "adjacent": {f"{a}-{b}": {"all": st(D[a] - D[b]), "V": st(D[a] - D[b], "V"), "T": st(D[a] - D[b], "T")}
                                 for a, b in zip(pts, pts[1:])}}
    for bp in BUDGET_POINTS:
        b = f"bc_{bp}"
        row = {}
        for cn in CONTROLS:
            c = f"{cn}_{bp}"
            if b not in D or c not in D:
                continue
            rb, rc = cfgs[b], cfgs[c]
            same = abs(rb["M_bytes"] / rc["M_bytes"] - 1) <= 0.01 and abs(rb["BOPs"] / rc["BOPs"] - 1) <= 0.01
            d = D[b] - D[c]
            row[cn] = {"bc_minus_control": {"all": st(d), "V": st(d, "V"), "T": st(d, "T")}, "resources_within_1pct": bool(same),
                       "M_ratio_bc_over_control": rb["M_bytes"] / rc["M_bytes"], "B_ratio_bc_over_control": rb["BOPs"] / rc["BOPs"],
                       "control_bc_total": rc["bc_total"], "bc_bc_total": rb["bc_total"]}
            ci = row[cn]["bc_minus_control"]["all"]["ci95"]
            row[cn]["verdict"] = ("BC better" if ci[1] < 0 else "control better" if ci[0] > 0 else "no difference resolved")
        res["test2"][bp] = row
    if (out / "truth_cpu.json").exists():
        cpu = jload(out / "truth_cpu.json")
        sub = np.array(cpu["subset"])
        cb = np.array(cpu["summed_ce"]["fp32"])
        for k, v in cpu["summed_ce"].items():
            gv = np.array(tr[k])[sub]
            res["cross_impl"][k] = {"max_abs_window_diff_per_token": float(np.abs(np.array(v) - gv).max() / T),
                                    "subset_mean_diff_per_token": float((np.array(v) - gv).mean() / T)}
            if k != "fp32":
                dg = (gv - base[sub]) / T
                dc = (np.array(v) - cb) / T
                res["cross_impl"][k].update({"subset_dnll_gpu": float(dg.mean()), "subset_dnll_cpu": float(dc.mean()),
                                             "dnll_window_corr": float(np.corrcoef(dg, dc)[0, 1])})
        kb = "bc_knee"
        if kb in cpu["summed_ce"]:
            for cn in CONTROLS:
                c = f"{cn}_knee"
                if c in cpu["summed_ce"]:
                    g_ = (np.array(tr[kb])[sub] - np.array(tr[c])[sub]).mean() / T
                    c_ = (np.array(cpu["summed_ce"][kb]) - np.array(cpu["summed_ce"][c])).mean() / T
                    res["cross_impl"][f"knee|bc-{cn}"] = {"gpu": float(g_), "cpu": float(c_), "same_sign": bool(np.sign(g_) == np.sign(c_))}
    jdump(out / "analysis.json", res)
    with open(out / "per_window.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "segment", "window", "summed_ce", "tokens", "dnll_per_token"])
        for k in ["fp32"] + list(D):
            for i in range(n):
                w.writerow([k, "V" if i < 32 else "T", i, tr[k][i], T, 0.0 if k == "fp32" else float(D[k][i])])
    figures(out, cfgs, res)
    log("test1: BC " + ", ".join(f"{b:.4g}" for b in bcs) + " | dNLL " + ", ".join(f"{d:+.4f}" for d in dn))
    for bp, row in res["test2"].items():
        log(f"test2 {bp}: " + "; ".join(f"{cn}: {r['bc_minus_control']['all']['mean']:+.4f} {r['verdict']}" for cn, r in row.items()))


def figures(out, cfgs, res):
    """BC total and measured dNLL against weight bytes and BOPs (relative to all-int8), test-1 points and test-2
    controls; markers with 95% CIs, no lines through unevaluated resources."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    col = {"BC": "#2a78d6", "shuffle0": "#eb6834", "shuffle1": "#eb6834", "traffic": "#1baf7a"}
    mk = {"BC": "o", "shuffle0": "v", "shuffle1": "^", "traffic": "s"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": surface,
                         "axes.facecolor": surface, "text.color": ink})
    sel = jload(out / "selection.json")
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    for j, (key, lab) in enumerate((("M_rel", "weight bytes / all-int8 (searched expert projections)"),
                                    ("B_rel", "BOPs / all-int8 (calibration routed rows)"))):
        a = ax[0, j]
        fk = "size" if key == "M_rel" else "bops"
        xmax = 0.9 if fk == "size" else 1.0
        fr = [f for f in sel["frontier"] if f["objective"] > 0 and f[fk] <= xmax]
        a.scatter([f[fk] for f in fr], [f["objective"] for f in fr], s=20, color="#9b9a95",
                  label=f"frontier (BC, not evaluated; x > {xmax} not shown)")
        for k, v in res["configs"].items():
            crit = v["criterion"]
            a.scatter([v[key]], [v["bc_total"]], s=60 if crit == "BC" else 40, marker=mk[crit], color=col[crit],
                      edgecolor=surface, linewidth=1.5, zorder=3, label=None)
        a.set_yscale("log")
        a.set_xlabel(lab)
        a.set_ylabel("BC total (log)")
        a.grid(True, color=grid, linewidth=0.6)
        b = ax[1, j]
        seen = set()
        for k, v in res["configs"].items():
            crit = v["criterion"]
            d = v["dnll"]["all"]
            lbl = {"BC": "BC configurations", "shuffle0": "control: BC shuffled (seed 0)", "shuffle1": "control: BC shuffled (seed 1)",
                   "traffic": "control: traffic x group-mean BC"}[crit] if crit not in seen else None
            seen.add(crit)
            b.errorbar([v[key]], [d["mean"]], yerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt=mk[crit],
                       color=col[crit], ms=7, capsize=3, lw=1.6, mec=surface, mew=1.2, label=lbl)
        b.axhline(0, color=ink2, lw=0.8)
        b.set_xlabel(lab)
        b.set_ylabel("dNLL vs FP32 (nat/token), 95% CI")
        b.grid(True, color=grid, linewidth=0.6)
    ax[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[1, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("OLMoE-1B-7B, fixed route, all expert projections: BC (top) and measured damage on 64 new windows (bottom)",
                 fontsize=11, color=ink)
    fig.savefig(out / "fig_multipoint.png", dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "capture", "table", "search", "truth", "cpu", "analyze"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=30.0, help="plan: seconds per MILP solve")
    ap.add_argument("--configs", default=None, help="truth / cpu: restrict to these names")
    ap.add_argument("--group", type=int, default=6, help="truth: configurations replayed together")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
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
        import resource
        timing = jload(out / "timing.json") if (out / "timing.json").exists() else {"runs": []}
        timing["runs"].append({"stage": args.stage, "device": args.device, "configs": args.configs, "status": status,
                               "wall_s": time.time() - t0, "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "peak_rss_gb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None})
        jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
