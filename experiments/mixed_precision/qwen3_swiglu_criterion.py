#!/usr/bin/env python3
"""Qwen3-0.6B gate/up criterion aggregation: native statistics vs PyTorch statistics, each aggregated here by an
implementation independent of run_native_bc_pipeline.native_tables, against that pipeline's table.json.

Notation (report): W weight term, A activation term, C = W + A; i node, c output channel (weights) or channel
(activations), s calibration window, j element of channel c.
  S_ic   = sum_s mean_{j in c} gW_isj^2                        W_i(w) = sum_c S_ic * sum_{j in c} dW_ij(w)^2
  G2_ic  = sum_s sum_j g_iscj^2       GS_ic = sum_s (sum_j g_iscj)^2   (per window first, squared, then summed)
  A_i(a) = sum_{t in {x, y}} sum_c [G2_tc var(dX_tc(a)) + GS_tc mean(dX_tc(a))^2],  n_t = N_cal * positions
Stages (--stage):
  native   run the statistics libraries of a finished native run (--native-run: stats_grads, stats_errors*) on its
           calibration windows; raw per-window results to <out>/native_raw
  torch    PyTorch FP32 autograd + the pass's QDQ (int params taken from the errors layouts) on the same windows,
           gate/up of all 28 layers; also the branch part of the shared input gradient (g_zg Wg, g_zu Wu)
  compare  W, A, C of both against table.json; format rankings; per-channel contributions; shared-input variants
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qwen3_swiglu_common import PREP, VOCAB, Quantizers, compare, environment, erm, load_model, node_mapping, \
    summed_ce, torch

HERE = Path(__file__).resolve().parent
QUANT = ("int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16")


def native_libs(run):
    libs = {"grads": (run / "stats_grads/grads.so", json.loads((run / "stats_grads/grads_layout.json").read_text()))}
    for d in sorted(run.glob("stats_errors*")):
        name = d.name.removeprefix("stats_")
        libs[name] = (d / f"{name}.so", json.loads((d / f"{name}_layout.json").read_text()))
    return libs


def stage_native(args, out, log):
    raw = out / "native_raw"
    raw.mkdir(parents=True)
    tools = erm.Tools()
    calib = np.load(args.native_run / "calib_inputs.npy")
    np.save(raw / "calib.npy", calib)
    for i, ids in enumerate(calib):
        oh = np.zeros((1, calib.shape[1], VOCAB), np.float32)
        oh[0, np.arange(calib.shape[1] - 1), ids[1:]] = 1.0
        np.save(raw / f"onehot_{i}.npy", oh)
    libs = native_libs(args.native_run)
    (out / "native_libs.json").write_text(json.dumps({k: {"so": str(v[0]), "so_sha256": __import__(
        "qwen3_swiglu_common").sha256(v[0])} for k, v in libs.items()}, indent=1) + "\n")
    jobs = [(name, i) for name in libs for i in range(len(calib))]

    def one(job):
        name, i = job
        so, lay = libs[name]
        r, _ = erm.run([sys.executable, HERE / "run_native_bc_pipeline.py", "--_stats-child", so, erm.SYMBOL_POSTFIX,
                        tools.lib, raw / "calib.npy", i, raw / f"onehot_{i}.npy" if name == "grads" else "-",
                        lay["result_index"], raw / f"{name}_w{i}.npy"])
        return json.loads(r.stdout.strip().splitlines()[-1])["run_s"]
    with ThreadPoolExecutor(args.workers) as pool:
        t = list(pool.map(one, jobs))
    for p in raw.glob("onehot_*.npy"):
        p.unlink()
    log(f"native statistics: {len(jobs)} runs, mean {np.mean(t):.0f}s")


def native_stats(args, nodes):
    """Per gate/up node: {"x": (G2, GS, positions), "y": ..., "w": S_c, "act": {slot: {fmt: (s1, s2)}}}."""
    raw = args.out_dir / "native_raw"
    libs = native_libs(args.native_run)
    n_cal = len(np.load(raw / "calib.npy"))
    g = np.stack([np.load(raw / f"grads_w{i}.npy") for i in range(n_cal)])
    lay = libs["grads"][1]
    taps = {t["tap"]: t for t in lay["taps"]}
    res = {n: {"act": {}} for n in nodes}
    for p in lay["points"]:
        if p["node"] in res:
            t = taps[p["tap"]]
            o, c = t["offset"], t["channels"]
            res[p["node"]][p["slot"]] = (g[:, o:o + c].sum(0), (g[:, o + c:o + 2 * c] ** 2).sum(0), t["positions"])
    for w in lay["weights"]:
        if w["node"] in res:
            res[w["node"]]["w"] = g[:, w["offset"]:w["offset"] + w["channels"]].sum(0) / w["elements_per_channel"]
            res[w["node"]]["w_axis"] = (w["channel_axis"], w["channels"])
            if "x_branch_offset" in w:            # MIXEDP_STATS_INPUT_GRAD=branch: this node's input-edge gradient
                o, c = w["x_branch_offset"], w["x_channels"]
                res[w["node"]]["x_branch"] = (g[:, o:o + c].sum(0), (g[:, o + c:o + 2 * c] ** 2).sum(0))
    for name, (_, el) in libs.items():
        if name == "grads":
            continue
        e = np.stack([np.load(raw / f"{name}_w{i}.npy") for i in range(n_cal)])
        etaps = {t["tap"]: t for t in el["taps"]}
        for p in el["points"]:
            if p["node"] in res:
                t = etaps[p["tap"]]
                c = t["channels"]
                res[p["node"]]["act"][p["slot"]] = {
                    f: (e[:, t["errors"][f]:t["errors"][f] + c].sum(0), e[:, t["errors"][f] + c:t["errors"][f] + 2 * c].sum(0))
                    for f in QUANT}
                res[p["node"]]["int_params"] = res[p["node"]].get("int_params", {})
                res[p["node"]]["int_params"][p["slot"]] = t["int_params"]
    return res, n_cal


def int_params(args):
    """(node, slot) -> {fmt: (scale, zp)} from the errors layouts: exactly what the native QDQ used."""
    out = {}
    for name, (_, el) in native_libs(args.native_run).items():
        if name == "grads":
            continue
        etaps = {t["tap"]: t for t in el["taps"]}
        for p in el["points"]:
            out[(p["node"], p["slot"])] = etaps[p["tap"]]["int_params"]
    return out


def stage_torch(args, out, log):
    mapping = node_mapping()
    targets = {n: v for n, v in mapping.items() if v[1] in ("gate", "up")}
    params = int_params(args)
    q = Quantizers(out)
    for f, lut in q.posit.items():
        bad, n = lut.verify()
        log(f"posit LUT {f}: {bad} of {n} corpus values differ from Universal")
        if bad:
            sys.exit("posit LUT does not reproduce Universal")
    model, wrapper = load_model(torch.float32)
    modules = dict(model.named_modules())
    mod = {n: modules[p.removesuffix(".weight")] for n, (_, _, p) in targets.items()}
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    calib = np.load(args.native_run / "calib_inputs.npy")

    def qdq(x, node, slot, f):
        if f.startswith("int"):
            scale, zp = params[(node, slot)][f]
            from qwen3_swiglu_common import int_act_qdq
            return int_act_qdq(x, scale, zp, int(f[3:]))
        return q.act(x, f)

    acc = {n: {"x_G2": 0, "x_GS": 0, "xb_G2": 0, "xb_GS": 0, "y_G2": 0, "y_GS": 0, "S": 0,
               "x_s1": {f: 0 for f in QUANT}, "x_s2": {f: 0 for f in QUANT},
               "y_s1": {f: 0 for f in QUANT}, "y_s2": {f: 0 for f in QUANT}} for n in targets}
    t0 = time.time()
    for ids in calib:
        cap = {}

        def hook(m, inp, outp):
            for t in (inp[0], outp):
                t.retain_grad()
            cap[m] = (inp[0], outp)
        hs = [m.register_forward_hook(hook) for m in mod.values()]
        loss = summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))
        loss.backward()
        for h in hs:
            h.remove()
        with torch.no_grad():
            for n, m in mod.items():
                x, y = cap[m]
                gx, gy = x.grad[0].double(), y.grad[0].double()
                gb = gy @ m.weight.double()                         # this branch's part of g_x
                a = acc[n]
                a["x_G2"] = a["x_G2"] + (gx ** 2).sum(0).numpy()
                a["x_GS"] = a["x_GS"] + gx.sum(0).numpy() ** 2
                a["xb_G2"] = a["xb_G2"] + (gb ** 2).sum(0).numpy()
                a["xb_GS"] = a["xb_GS"] + gb.sum(0).numpy() ** 2
                a["y_G2"] = a["y_G2"] + (gy ** 2).sum(0).numpy()
                a["y_GS"] = a["y_GS"] + gy.sum(0).numpy() ** 2
                gw = gy.T @ x[0].detach().double()                  # [out, in]
                a["S"] = a["S"] + (gw ** 2).mean(1).numpy()
                for slot, t in (("x", x[0].detach()), ("y", y[0].detach())):
                    for f in QUANT:
                        d = (qdq(t, n, slot, f) - t).double()
                        a[f"{slot}_s1"][f] = a[f"{slot}_s1"][f] + d.sum(0).numpy()
                        a[f"{slot}_s2"][f] = a[f"{slot}_s2"][f] + (d ** 2).sum(0).numpy()
        model.zero_grad(set_to_none=True)
    log(f"PyTorch statistics: {(time.time() - t0) / len(calib):.1f}s per window")
    err2 = {}
    for n, m in mod.items():
        w = m.weight.detach()
        err2[n] = {f: ((q.weight(w, f) - w).double() ** 2).sum(1).numpy() for f in QUANT}   # per output channel
    np.savez_compressed(out / "torch_stats.npz", **{
        f"{n}|{k}" if not isinstance(v, dict) else f"{n}|{k}|{f}": (v if not isinstance(v, dict) else v[f])
        for n, a in acc.items() for k, v in a.items() for f in (v if isinstance(v, dict) else [None])} | {
        f"{n}|err2|{f}": v for n, e in err2.items() for f, v in e.items()})
    (out / "torch_meta.json").write_text(json.dumps({"windows": len(calib), "positions": int(calib.shape[1])}) + "\n")


def aggregate_A(G2, GS, s1, s2, n):
    mean = s1 / n
    var = s2 / n - mean ** 2
    per_c = G2 * var + GS * mean ** 2
    return float(per_c.sum()), per_c


def stage_compare(args, out, log):
    mapping = node_mapping()
    targets = [n for n, v in mapping.items() if v[1] in ("gate", "up")]
    table = json.loads((args.native_run / "table.json").read_text())
    tbranch = json.loads((args.native_run / "table_branch.json").read_text())
    werr = {e["node"]: e for e in json.loads((args.native_run / "weight_err.json").read_text())["nodes"]}
    nat, n_cal = native_stats(args, targets)
    tz = np.load(out / "torch_stats.npz")
    meta = json.loads((out / "torch_meta.json").read_text())
    from scipy.stats import spearmanr
    rows = []
    for n in targets:
        layer, kind, _ = mapping[n]
        N = nat[n]
        P = N["x"][2]
        cnt = n_cal * P
        assert N["w_axis"] == (werr[n]["channel_axis"], werr[n]["channels"]), "weight channel axes differ"
        r = {"node": n, "layer": layer, "kind": kind, "w_axis": N["w_axis"]}
        nW, nA, nAb, nAx, nAxb, tW, tA, tAb, tAx, tAxb = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        pcA = {}
        for f in QUANT:
            e2 = np.array([np.inf if v is None else v for v in werr[n]["err2"][f]])
            nW[f] = float((N["w"] * e2).sum())
            nA[f] = 0.0
            for slot in ("x", "y"):
                s1, s2 = N["act"][slot][f]
                v, pc = aggregate_A(N[slot][0], N[slot][1], s1, s2, cnt)
                nA[f] += v
                if slot == "x":
                    nAx[f] = v
                if f == "int4" and slot == "x":
                    pcA["native_x"] = pc
            nAxb[f] = aggregate_A(N["x_branch"][0], N["x_branch"][1], *N["act"]["x"][f], cnt)[0]
            nAb[f] = nA[f] - nAx[f] + nAxb[f]
            tw = float((tz[f"{n}|S"] * tz[f"{n}|err2|{f}"]).sum())
            tW[f] = tw
            tcnt = meta["windows"] * meta["positions"]
            ax, pcx = aggregate_A(tz[f"{n}|x_G2"], tz[f"{n}|x_GS"], tz[f"{n}|x_s1|{f}"], tz[f"{n}|x_s2|{f}"], tcnt)
            axb, _ = aggregate_A(tz[f"{n}|xb_G2"], tz[f"{n}|xb_GS"], tz[f"{n}|x_s1|{f}"], tz[f"{n}|x_s2|{f}"], tcnt)
            ay, _ = aggregate_A(tz[f"{n}|y_G2"], tz[f"{n}|y_GS"], tz[f"{n}|y_s1|{f}"], tz[f"{n}|y_s2|{f}"], tcnt)
            tA[f], tAb[f], tAx[f], tAxb[f] = ax + ay, axb + ay, ax, axb
            if f == "int4":
                pcA["torch_x"] = pcx
        r["W"] = {f: {"table": table["B"][n][f], "native_reagg": nW[f], "torch": tW[f]} for f in QUANT}
        r["A"] = {f: {"table": table["C"][n][f], "native_reagg": nA[f], "torch": tA[f],
                      "table_branch": tbranch["C"][n][f], "native_branch_reagg": nAb[f], "torch_branch_x": tAb[f],
                      "native_x_only": nAx[f], "native_branch_x_only": nAxb[f],
                      "torch_x_only": tAx[f], "torch_branch_x_only": tAxb[f]} for f in QUANT}
        r["reagg_vs_table"] = {"W": max(abs(nW[f] - table["B"][n][f]) / max(abs(table["B"][n][f]), 1e-300) for f in QUANT),
                               "A": max(abs(nA[f] - table["C"][n][f]) / max(abs(table["C"][n][f]), 1e-300) for f in QUANT),
                               "A_branch": max(abs(nAb[f] - tbranch["C"][n][f]) / max(abs(tbranch["C"][n][f]), 1e-300)
                                               for f in QUANT)}
        r["torch_vs_native"] = {
            "W_rel": {f: abs(tW[f] - nW[f]) / max(abs(nW[f]), 1e-300) for f in QUANT},
            "A_rel": {f: abs(tA[f] - nA[f]) / max(abs(nA[f]), 1e-300) for f in QUANT},
            "C_rel": {f: abs(tW[f] + tA[f] - nW[f] - nA[f]) / max(abs(nW[f] + nA[f]), 1e-300) for f in QUANT},
            "A_branch_rel": {f: abs(tAb[f] - nAb[f]) / max(abs(nAb[f]), 1e-300) for f in QUANT}}
        Cn = np.array([[nW[w] + nA[a] for a in QUANT] for w in QUANT]).ravel()
        Ct = np.array([[tW[w] + tA[a] for a in QUANT] for w in QUANT]).ravel()
        r["rank"] = {"W_spearman": float(spearmanr([nW[f] for f in QUANT], [tW[f] for f in QUANT])[0]),
                     "A_spearman": float(spearmanr([nA[f] for f in QUANT], [tA[f] for f in QUANT])[0]),
                     "C_pairs_spearman": float(spearmanr(Cn, Ct)[0]),
                     "C_best_same": int(np.argmin(Cn)) == int(np.argmin(Ct))}
        r["per_channel_int4_x"] = compare(pcA["native_x"], pcA["torch_x"])
        r["shared_input"] = {f: {"A_x_total_over_branch_torch": tAx[f] / max(tAxb[f], 1e-300),
                                 "A_x_total_over_branch_native": nAx[f] / max(nAxb[f], 1e-300)} for f in QUANT}
        rows.append(r)
    (out / "criterion_compare.json").write_text(json.dumps(rows, default=float) + "\n")
    def worst(key, sub):
        return max(max(r[key][sub].values()) if isinstance(r[key][sub], dict) else r[key][sub] for r in rows)
    log(f"re-aggregation vs table.json: W {max(r['reagg_vs_table']['W'] for r in rows):.1e}, "
        f"A {max(r['reagg_vs_table']['A'] for r in rows):.1e}, A (branch table) "
        f"{max(r['reagg_vs_table']['A_branch'] for r in rows):.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("native", "torch", "compare"), required=True)
    ap.add_argument("--native-run", type=Path, default=HERE / "runs/native_bytecode_20261012/qwen3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.native_run = args.native_run.resolve()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    if args.stage == "native":
        (args.out_dir / "environment.json").write_text(json.dumps(environment(
            ["experiments/mixed_precision/qwen3_swiglu_criterion.py"]), indent=1) + "\n")
    {"native": stage_native, "torch": stage_torch, "compare": stage_compare}[args.stage](args, args.out_dir, log)


if __name__ == "__main__":
    main()
