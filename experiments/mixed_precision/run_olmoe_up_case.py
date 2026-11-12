#!/usr/bin/env python3
"""OLMoE fixed route, one reproducible BC mis-ranking (runs/olmoe_crosstext_20261017): L15.E17 up_proj weights int8 vs
p8e1 (weight-only, everything else FP32). Diagnostic of a single fixed case, not a general claim.

Questions: (1) how do the two weight errors dW = Q(W) - W differ in direction; (2) does the first-order term g . dW explain
the reversed order; (3) after removing it, does the remainder still oppose BC's g^2 score; (4) along each error direction
(W + alpha dW), at which alpha does the order become reversed.

Data: stage 1's 10 calibration windows (BC's statistics source) and the cross-text sets D and V (16 articles x 4 windows
each), every window with its own FP32 route; layer 15 is the last layer, so every variant is the FP32 model up to layer
15's input, then layer 15 with the modified expert, then the FP32 head. Loss = summed next-token CE / 127 per window;
segment = article (D, V), resampling unit for intervals.

Stages: plan, run (GPU), analyze.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / "runs/olmoe_fixed_route_20261015"
CT = HERE / "runs/olmoe_crosstext_20261017"
L, E, PROJ = 15, 17, "up_proj"
FORMATS = ("int8", "p8e1")
ALPHAS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0)
SEQ, T_PRED, WPS, CHUNK = 128, 127, 4, 32
BOOT = 10000


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_plan(args, out, log):
    if (out / "plan.json").exists():
        raise SystemExit("exists")
    jdump(out / "plan.json", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case": "L15.E17 up_proj, weight-only int8 vs p8e1; candidates L15.E17|w|up=int8/fp32 and L15.E17|w|up=p8e1/fp32 "
                "(reproducible BC error of runs/olmoe_crosstext_20261017: BC W(int8) 0.000796 < W(p8e1) 0.0309, measured "
                "int8 - p8e1 = +1.5e-4 (D), +2.3e-4 (V) nat/token)",
        "nature": "diagnostic of one case chosen from the D -> V confirmation; no threshold, no criterion change, no general claim",
        "fixed": "stage 1 weight quantizer (per output channel int8; p8e1 Universal LUT), FP32 elsewhere, each window's FP32 "
                 "route, the shared corrected quantizer code; GPU FP32, TF32 off",
        "data": {"C": "stage 1 calibration (10 windows, one text)", "D": "cross-text D (16 articles x 4 windows)",
                 "V": "cross-text V (16 articles x 4 windows)"},
        "measures": {
            "Q1": "||dW||, cosine(dW_int8, dW_p8e1), sign agreement, projection on W (dW = beta W + r), per-channel error "
                  "mean and norm, relative error by |W| decile, cosine of each dW with the gradient",
            "Q2": "first-order term per window <dL/dW, dW> / 127 (per-window gradients, FP32), summed per article; against the "
                  "measured damage at alpha = 1; also the calibration gradient applied to D / V (deployable sign)",
            "Q3": "remainder = measured - first-order; its order against BC's order; BC-form W(dW) = sum_c (G_c / n_in) "
                  "sum_in dW^2 re-estimated on C, D, V from per-window gradients (G_c = sum_window sum_in g^2), against "
                  "stage 1's table value",
            "Q4": f"alpha in {list(ALPHAS)}: measured damage of W + alpha dW for both formats; order per alpha; least-squares "
                  "fit damage(alpha) = b alpha + c alpha^2 per article (b vs the first-order term, c = directional curvature)"},
        "intervals": f"article bootstrap ({BOOT} draws, D seed 2, V seed 3, the same indices for every quantity of a set)",
        "check": "alpha = 1 must equal the cross-text truth of the two candidates (same chunking) and stage 1's calibration truth",
        "inputs_sha256": {"table": sha256(S1 / "table.json"), "candidates": sha256(S1 / "candidates.json"),
                          "D_truth": sha256(CT / "D/truth.json"), "V_truth": sha256(CT / "V/truth.json")},
        "script_sha256": sha256(HERE / "run_olmoe_up_case.py")})
    log("plan written")


def sets():
    return {"C": (S1, "calibration", np.load(S1 / "tokens_calibration.npy"), 10),
            "D": (CT / "D", "w", np.load(CT / "D/tokens.npy"), CHUNK),
            "V": (CT / "V", "w", np.load(CT / "V/tokens.npy"), CHUNK)}


def stage_run(args, out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_bc_numerics import quantizers
    from run_olmoe_fixed_route import gpu_exact, head_ce, load_routes
    gpu_exact("cuda")
    dev = "cuda"
    cfg, ck = oc.config(), oc.Checkpoint()
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    (out / "work").mkdir(exist_ok=True)
    q, _ = quantizers(out / "work", dev, "corrected")
    w15 = ck.layer(L, dev, torch.float32)
    W = w15[PROJ][E].clone()
    Q = {f: q.weight(W.cpu(), f).to(dev) for f in FORMATS}          # the quantized weight itself (alpha = 1 uses it)
    dW = {f: Q[f] - W for f in FORMATS}
    torch.save({"W": W.cpu(), **{f"dW_{f}": v.cpu() for f, v in dW.items()}}, out / "weights.pt")
    res = {}
    for name, (d, rname, toks_np, chunk) in sets().items():
        rts, _, _ = load_routes(d, rname, cfg, dev)
        n = len(toks_np)
        base, meas = [], {f: {a: [] for a in ALPHAS} for f in FORMATS}
        fo = {f: [] for f in FORMATS}                               # per-window <g, dW>
        g2 = []                                                     # per window, per output channel sum_in g^2
        gsum = torch.zeros_like(W)
        t0 = time.time()
        for a0 in range(0, n, chunk):
            sl = slice(a0 * SEQ, min(n, a0 + chunk) * SEQ)
            r = [(i[sl], w_[sl]) for i, w_ in rts]
            tk = torch.from_numpy(toks_np[a0:a0 + chunk]).to(dev)
            with torch.no_grad():
                h = F.embedding(tk, g["embed"])
                for l in range(L):
                    h = oc.layer_fixed(ck.layer(l, dev, torch.float32), h, cfg, cos, sin, *r[l], l)[0]
                h15 = h
                base_c = head_ce(g, oc.layer_fixed(w15, h15, cfg, cos, sin, *r[L], L)[0], tk, cfg)
                base += base_c.cpu().numpy().tolist()
                for f in FORMATS:
                    for al in ALPHAS:
                        wq = dict(w15)
                        up = w15[PROJ].clone()
                        up[E] = Q[f] if al == 1.0 else (W + al * dW[f])
                        wq[PROJ] = up
                        ce = head_ce(g, oc.layer_fixed(wq, h15, cfg, cos, sin, *r[L], L)[0], tk, cfg)
                        meas[f][al] += (ce - base_c).cpu().numpy().tolist()
                        del wq, up
            # per-window gradients of the summed CE w.r.t. the expert's up_proj weight (FP32 model)
            for k in range(len(tk)):
                Wg = W.clone().requires_grad_(True)
                up = torch.cat([w15[PROJ][:E], Wg[None], w15[PROJ][E + 1:]])
                wg = dict(w15)
                wg[PROJ] = up
                rows = slice(k * SEQ, (k + 1) * SEQ)
                rr = [(i[rows], w_[rows]) for i, w_ in r]
                with torch.enable_grad():
                    hk = oc.layer_fixed(wg, h15[k:k + 1], cfg, cos, sin, *rr[L], L)[0]
                    loss = oc.summed_ce(oc.head(g, hk, cfg), tk[k:k + 1]).sum()
                    gr, = torch.autograd.grad(loss, Wg)
                gsum += gr
                for f in FORMATS:
                    fo[f].append(float((gr * dW[f]).sum()))
                g2.append((gr ** 2).sum(1).cpu().numpy())
                del Wg, up, wg, gr
        res[name] = {"windows": n, "base": base, "measured": {f: {str(a): v for a, v in meas[f].items()} for f in FORMATS},
                     "first_order": fo, "seconds": time.time() - t0}
        np.save(out / f"g2_{name}.npy", np.array(g2))
        torch.save(gsum.cpu(), out / f"gsum_{name}.pt")
        log(f"{name}: {n} windows ({time.time() - t0:.0f}s)")
    jdump(out / "run.json", res)


def stage_analyze(args, out, log):
    import torch
    run = jload(out / "run.json")
    wt = torch.load(out / "weights.pt")
    W = wt["W"].double().numpy()
    dW = {f: wt[f"dW_{f}"].double().numpy() for f in FORMATS}
    tbl = jload(S1 / "table.json")["B"][f"L{L}.E{E}.{PROJ}"]
    res = {"bc_table": {f: tbl[f] for f in FORMATS}}
    # Q1: error direction
    n_in = W.shape[1]
    q1 = {}
    for f in FORMATS:
        d = dW[f]
        beta = float((d * W).sum() / (W * W).sum())
        r = d - beta * W
        aw = np.abs(W).ravel()
        dec = np.quantile(aw, np.linspace(0, 1, 11))
        rel = []
        for k in range(10):
            m = (aw >= dec[k]) & (aw <= dec[k + 1]) & (aw > 0)
            rel.append(float(np.mean(np.abs(d.ravel()[m]) / aw[m])))
        q1[f] = {"fro": float(np.linalg.norm(d)), "rel_fro": float(np.linalg.norm(d) / np.linalg.norm(W)),
                 "beta_projection_on_W": beta, "residual_fro": float(np.linalg.norm(r)),
                 "zero_error_share": float((d == 0).mean()), "mean_error": float(d.mean()),
                 "channel_mean_error_abs_median": float(np.median(np.abs(d.mean(1)))),
                 "channel_norm_cv": float(np.linalg.norm(d, axis=1).std() / np.linalg.norm(d, axis=1).mean()),
                 "rel_error_by_absW_decile": rel, "shrink_share": float(((np.abs(W + d) < np.abs(W)) & (W != 0)).mean()),
                 "grow_share": float(((np.abs(W + d) > np.abs(W)) & (W != 0)).mean())}
    a, b = dW["int8"].ravel(), dW["p8e1"].ravel()
    q1["cosine_int8_p8e1"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    q1["sign_agreement_nonzero"] = float((np.sign(a) == np.sign(b))[(a != 0) & (b != 0)].mean())
    q1["fro_ratio_p8_over_int8"] = q1["p8e1"]["fro"] / q1["int8"]["fro"]
    for s in ("C", "D", "V"):
        gs = torch.load(out / f"gsum_{s}.pt").double().numpy().ravel()
        q1[f"cosine_with_gradient_{s}"] = {f: float(gs @ dW[f].ravel() / (np.linalg.norm(gs) * np.linalg.norm(dW[f]))) for f in FORMATS}
        q1[f"cosine_gradient_with_W_{s}"] = float(gs @ W.ravel() / (np.linalg.norm(gs) * np.linalg.norm(W)))
    res["Q1"] = q1
    # per-set quantities
    gC = torch.load(out / "gsum_C.pt").double().numpy()
    nC = run["C"]["windows"]
    for s in ("C", "D", "V"):
        R = run[s]
        n = R["windows"]
        meas = {f: {a_: np.array(R["measured"][f][str(a_)]) / T_PRED for a_ in ALPHAS} for f in FORMATS}
        fo = {f: np.array(R["first_order"][f]) / T_PRED for f in FORMATS}
        seg = (lambda x: x.reshape(-1, WPS).mean(1)) if s != "C" else (lambda x: np.array([x.mean()]))
        nseg = 1 if s == "C" else n // WPS
        idx = np.random.default_rng(2 if s == "D" else 3).integers(0, nseg, (BOOT, nseg)) if s != "C" else None

        def st(x):
            v = seg(x)
            o = {"mean": float(v.mean())}
            if idx is not None:
                bb = v[idx].mean(1)
                o.update(ci95=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], segments_positive=int((v > 0).sum()),
                         segments=int(len(v)))
            return o
        r = {"measured_alpha1": {f: st(meas[f][1.0]) for f in FORMATS},
             "measured_alpha1_int8_minus_p8": st(meas["int8"][1.0] - meas["p8e1"][1.0]),
             "first_order": {f: st(fo[f]) for f in FORMATS},
             "first_order_int8_minus_p8": st(fo["int8"] - fo["p8e1"]),
             "remainder": {f: st(meas[f][1.0] - fo[f]) for f in FORMATS},
             "remainder_int8_minus_p8": st((meas["int8"][1.0] - fo["int8"]) - (meas["p8e1"][1.0] - fo["p8e1"])),
             "first_order_explains_share": {f: float(fo[f].sum() / meas[f][1.0].sum()) if meas[f][1.0].sum() else None for f in FORMATS},
             "window_corr_first_order_vs_measured": {f: float(np.corrcoef(fo[f], meas[f][1.0])[0, 1]) for f in FORMATS}}
        if s != "C":
            # deployable: the calibration gradient (mean per calibration window) applied to every window of this set
            r["calibration_first_order_per_token"] = {f: float((gC * dW[f]).sum() / nC / T_PRED) for f in FORMATS}
        # Q4: alpha scan, order per alpha, quadratic fit per segment
        r["alpha_scan"] = {str(a_): {"int8": st(meas["int8"][a_]), "p8e1": st(meas["p8e1"][a_]),
                                     "int8_minus_p8": st(meas["int8"][a_] - meas["p8e1"][a_])} for a_ in ALPHAS}
        A = np.array([[a_, a_ * a_] for a_ in ALPHAS])
        fit = {}
        for f in FORMATS:
            Y = np.stack([seg(meas[f][a_]) for a_ in ALPHAS])       # [alpha, segment]
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)              # [2, segment]
            bvals, cvals = coef[0], coef[1]
            pred = A @ coef
            fit[f] = {"b": float(bvals.mean()), "c": float(cvals.mean()),
                      "first_order_term": float(seg(fo[f]).mean()),
                      "fit_max_abs_residual": float(np.abs(pred - Y).max()),
                      **({"b_ci95": [float(np.percentile(bvals[idx].mean(1), 2.5)), float(np.percentile(bvals[idx].mean(1), 97.5))],
                          "c_ci95": [float(np.percentile(cvals[idx].mean(1), 2.5)), float(np.percentile(cvals[idx].mean(1), 97.5))]} if idx is not None else {})}
        fit["c_ratio_p8_over_int8"] = fit["p8e1"]["c"] / fit["int8"]["c"] if fit["int8"]["c"] else None
        r["quadratic_fit"] = fit
        # BC-form second-order term re-estimated on this set: sum_c (G_c / n_in) sum_in dW^2
        g2 = np.load(out / f"g2_{s}.npy").astype(np.float64)       # [window, out channel]
        Gc = g2.sum(0)
        r["bc_form_W"] = {f: float((Gc / n_in * (dW[f] ** 2).sum(1)).sum()) for f in FORMATS}
        r["bc_form_ratio_p8_over_int8"] = r["bc_form_W"]["p8e1"] / r["bc_form_W"]["int8"]
        res[s] = r
    res["bc_table_ratio_p8_over_int8"] = tbl["p8e1"] / tbl["int8"]
    # consistency with the cross-text truth and stage 1's calibration truth
    chk = {}
    for s, f_ in (("D", CT / "D/truth.json"), ("V", CT / "V/truth.json")):
        tr = jload(f_)["rows"]
        for f in FORMATS:
            chk[f"{s}_{f}"] = float(np.abs(np.array(run[s]["measured"][f]["1.0"]) - np.array(tr[f"L15.E17|w|up={f}/fp32"])).max())
    trc = jload(S1 / "truth_calibration.json")["rows"]
    for f in FORMATS:
        chk[f"C_{f}"] = float(np.abs(np.array(run["C"]["measured"][f]["1.0"]) - np.array(trc[f"L15.E17|w|up={f}/fp32"])).max())
    res["consistency_max_abs_summed_ce"] = chk
    jdump(out / "analysis.json", res)
    log(json.dumps(res["Q1"], indent=0)[:2500])
    for s in ("C", "D", "V"):
        r = res[s]
        log(f"{s}: measured int8 {r['measured_alpha1']['int8']['mean']:+.2e} p8 {r['measured_alpha1']['p8e1']['mean']:+.2e}; "
            f"first-order int8 {r['first_order']['int8']['mean']:+.2e} p8 {r['first_order']['p8e1']['mean']:+.2e}; remainder "
            f"int8 {r['remainder']['int8']['mean']:+.2e} p8 {r['remainder']['p8e1']['mean']:+.2e}; fit {r['quadratic_fit']}; "
            f"bc-form ratio {r['bc_form_ratio_p8_over_int8']:.1f}")
    log(f"consistency {chk}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        with open(out / f"{args.stage}.log", "a") as fh:
            fh.write(line + "\n")
    globals()[f"stage_{args.stage}"](args, out, log)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
