#!/usr/bin/env python3
"""Qwen3-0.6B gate / up quantized alone and together: what the criterion predicts against what happens.

For layers --layers, targets gate | up | both, modes w (weight only) | a (activation only: the node's input and
output) | wa, formats --formats; every other node FP32. Calibration windows only (validation_tokens), summed
next-token CE as in the criterion.
  --stage sim     PyTorch FP32 with the pass's QDQ semantics: per-output-channel int / posit / cast weights, int
                  activations with the native (scale, zp) of the errors layouts, posit via the Universal-exact LUT;
                  each branch quantizes ITS OWN copy of the shared input (as the pass inserts one QDQ per consumer)
  --stage native  the same configurations compiled by onnx-mlir (--convert-onnx-to-mixed-precision, the evaluation
                  build levels) and run on the calibration windows, for --native-configs, plus FP32
  --stage analyze tables, correlations, joint vs single
Per configuration (sim): predicted W, A, C (native table.json; A also split into its x and y parts from the native
statistics), signed loss delta per window and summed, first-order f_s = sum g_y . dy (dy: the node output change
with FP32 inputs upstream, i.e. the local perturbation) and its parts through the weight / input / output, sum_s f_s
and sum_s f_s^2 (what the criterion approximates), and for both: the cross term 2 sum_s f_gate f_up and the SwiGLU
second-order term sum_s g_h . SiLU'(zg) dzg dzu.
"""
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qwen3_swiglu_common import PREP, Quantizers, environment, erm, int_act_qdq, load_model, node_mapping, \
    summed_ce, torch
from qwen3_swiglu_criterion import QUANT, int_params, native_stats, aggregate_A

HERE = Path(__file__).resolve().parent


def configs(layers, formats):
    for layer, target, mode, fmt in itertools.product(layers, ("gate", "up", "both"), ("w", "a", "wa"), formats):
        yield {"layer": layer, "target": target, "mode": mode, "fmt": fmt,
               "name": f"L{layer}_{target}_{mode}_{fmt}"}


def nodes_of(cfg, by_layer):
    return [by_layer[(cfg["layer"], k)] for k in (("gate", "up") if cfg["target"] == "both" else (cfg["target"],))]


def stage_sim(args, out, log):
    mapping = node_mapping()
    by_layer = {(l, k): n for n, (l, k, _) in mapping.items()}
    params = int_params(args)
    q = Quantizers(out)
    for f in q.posit:
        bad, n = q.posit[f].verify()
        if bad:
            sys.exit(f"posit LUT {f}: {bad} of {n} differ from Universal")
    model, wrapper = load_model(torch.float32)
    mods = dict(model.named_modules())
    mod = {n: mods[p.removesuffix(".weight")] for n, (_, _, p) in mapping.items()}
    calib = np.load(args.native_run / "calib_inputs.npy")
    layers = [int(x) for x in args.layers.split(",")]
    formats = args.formats.split(",")

    def qdq(x, node, slot, f):
        if f.startswith("int"):
            s, z = params[(node, slot)][f]
            return int_act_qdq(x, s, z, int(f[3:]))
        return q.act(x, f)

    # base pass: losses, and the tensors / gradients of the studied layers
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    base, t0 = [], time.time()
    for ids in calib:
        cap = {}
        hooks = []
        for l in layers:
            mlp = mods[f"model.layers.{l}.mlp"]
            for kind, m in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
                hooks.append(m.register_forward_hook(lambda mm, i, o, l=l, kind=kind: cap.update(
                    {(l, kind): (i[0], o)})))
        x0 = None
        z = wrapper(torch.from_numpy(ids[None]))
        for (l, kind), (i, o) in cap.items():
            i.retain_grad()
            o.retain_grad()
        loss = summed_ce(z, torch.from_numpy(ids))
        loss.backward()
        for h in hooks:
            h.remove()
        r = {"loss": float(loss)}
        for (l, kind), (i, o) in cap.items():
            r[(l, kind)] = {"x": i[0].detach().clone(), "y": o[0].detach().clone(), "gx": i.grad[0].clone(),
                            "gy": o.grad[0].clone()}
        base.append(r)
        model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"base pass: {(time.time() - t0) / len(calib):.1f}s per window")

    rows = []
    t0 = time.time()
    for cfg in configs(layers, formats):
        nodes = nodes_of(cfg, by_layer)
        f, mode = cfg["fmt"], cfg["mode"]
        saved, hooks = {}, []
        for n in nodes:
            m = mod[n]
            if "w" in mode:
                saved[n] = m.weight.data.clone()
                m.weight.data = q.weight(m.weight.data, f)
            if "a" in mode:
                hooks.append(m.register_forward_pre_hook(lambda mm, inp, n=n: (qdq(inp[0], n, "x", f),)))
                hooks.append(m.register_forward_hook(lambda mm, inp, o, n=n: qdq(o, n, "y", f)))
        losses = []
        with torch.no_grad():
            for ids in calib:
                losses.append(float(summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))))
        for h in hooks:
            h.remove()
        # first order, from the base tensors (local perturbation of each node's output)
        parts = {n: {"f": [], "fW": [], "fX": [], "fY": []} for n in nodes}
        dys = {}
        with torch.no_grad():
            for s, b in enumerate(base):
                for n in nodes:
                    _, kind, _ = mapping[n]
                    t = b[(cfg["layer"], kind)]
                    W = mod[n].weight.data                       # quantized now if "w" in mode
                    W0 = saved.get(n, W)
                    xq = qdq(t["x"], n, "x", f) if "a" in mode else t["x"]
                    yprime = xq @ W.T
                    yq = qdq(yprime, n, "y", f) if "a" in mode else yprime
                    gy = t["gy"].double()
                    dy = (yq - t["y"]).double()
                    dys[(s, n)] = dy
                    parts[n]["f"].append(float((gy * dy).sum()))
                    parts[n]["fW"].append(float((gy * (t["x"] @ (W - W0).T).double()).sum()))
                    parts[n]["fX"].append(float((gy * ((xq - t["x"]) @ W0.T).double()).sum()))
                    parts[n]["fY"].append(float((gy * (yq - yprime).double()).sum()))
        for n in saved:
            mod[n].weight.data = saved[n]
        dL = np.array(losses) - np.array([b["loss"] for b in base])
        ftot = np.sum([parts[n]["f"] for n in nodes], axis=0)
        row = dict(cfg, nodes=nodes, dL=dL.tolist(), dL_sum=float(dL.sum()), dL_abs_sum=float(np.abs(dL).sum()),
                   f=ftot.tolist(), f_sum=float(ftot.sum()), f2_sum=float((ftot ** 2).sum()),
                   parts=parts)                        # per node: per-window f, fW, fX, fY
        if cfg["target"] == "both":
            g, u = nodes
            fg, fu = np.array(parts[g]["f"]), np.array(parts[u]["f"])
            row["cross_2fgfu"] = float(2 * (fg * fu).sum())
            inter = 0.0                                          # sum_s g_h . SiLU'(zg) dzg dzu
            for s, b in enumerate(base):
                zg = b[(cfg["layer"], "gate")]["y"].double()
                sg = torch.sigmoid(zg)
                dsilu = sg + zg * sg * (1 - sg)
                gh = b[(cfg["layer"], "down")]["gx"].double()
                inter += float((gh * dsilu * dys[(s, g)] * dys[(s, u)]).sum())
            row["swiglu_second_order"] = inter
        rows.append(row)
    log(f"{len(rows)} configurations: {(time.time() - t0) / len(rows):.1f}s each")
    (out / "sim.json").write_text(json.dumps(rows) + "\n")


def native_configs(spec):
    out = []
    for s in spec.split(";"):
        layer, target, mode, fmt = s.split(":")
        out.append({"layer": int(layer), "target": target, "mode": mode, "fmt": fmt,
                    "name": f"L{layer}_{target}_{mode}_{fmt}"})
    return out


def stage_native(args, out, log):
    from evaluate_native_search_chunked import chunked_run_so
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, imported_model, qwen3_spec
    import onnx
    mapping = node_mapping()
    by_layer = {(l, k): n for n, (l, k, _) in mapping.items()}
    spec = qwen3_spec(out)
    tools = erm.Tools()
    graph = onnx.load(str(spec.model_path), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in json.loads((args.native_run / "ranges_by_point.json").read_text()).items():
        for slot, r in slots.items():
            n = by_name[node]
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    calib = np.load(args.native_run / "calib_inputs.npy")
    np.save(out / "calib_inputs.npy", calib)
    base = imported_model(args.native_run.parent / args.native_run.name)
    posit_obj = erm.build_posit_runtime(tools, out)
    import os
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"

    def one(cfg):
        target = out / cfg["name"]
        target.mkdir(exist_ok=True)
        env = None
        if cfg["target"] != "fp32":
            sel = set(nodes_of(cfg, by_layer))
            full = {n["name"]: ({"w": cfg["fmt"] if "w" in cfg["mode"] else None,
                                 "a": cfg["fmt"] if "a" in cfg["mode"] else "fp32"} if n["name"] in sel
                                else {"w": None, "a": "fp32"}) for n in spec.nodes}
            env = erm.node_formats_env(full, spec.nodes, ranges)
        t = time.time()
        so, info = erm.build_so(tools, target, "m", base, env, posit_obj, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True,
                                uses_posit=None if env else False, uses_f16=None if env else False)
        build_s = time.time() - t
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, ev = chunked_run_so(args.chunks)(tools, so, out / "calib_inputs.npy", target / "logits.npy", 36000)
        losses = [float(cross_entropy(z[s, :-1], calib[s, 1:]).sum()) for s in range(len(calib))]
        (target / "logits.npy").unlink()
        for p in (so, target / "model.constants.bin"):    # 3 GB per configuration: keep only the results
            p.unlink(missing_ok=True)
        res = dict(cfg, losses=losses, build_s=build_s, warnings=info["warnings"], eval_wall_s=ev["wall_seconds"])
        (target / "result.json").write_text(json.dumps(res) + "\n")
        log(f"native {cfg['name']}: summed loss {sum(losses):.4f} (build {build_s:.0f}s)")
        return res
    cfgs = [{"layer": -1, "target": "fp32", "mode": "-", "fmt": "fp32", "name": "fp32"}] + native_configs(args.native_configs)
    with ThreadPoolExecutor(args.parallel) as pool:
        res = list(pool.map(one, cfgs))
    (out / "native.json").write_text(json.dumps(res) + "\n")


def stage_analyze(args, out, log):
    """Predicted (native table, and its x / y / weight parts from the native statistics; the input part also from
    PyTorch with the total and with the branch gradient of the shared input) against the simulated first order and
    loss deltas; joint vs single; rank correlations."""
    from scipy.stats import spearmanr, pearsonr
    mapping = node_mapping()
    table = json.loads((args.native_run / "table.json").read_text())
    tbranch = json.loads((args.native_run / "table_branch.json").read_text())
    nat, n_cal = native_stats(args, [n for n, v in mapping.items() if v[1] in ("gate", "up")])
    crit = {r["node"]: r for r in json.loads((args.criterion_dir / "criterion_compare.json").read_text())}
    sim = json.loads((out / "sim.json").read_text())

    def a_parts(n, f):
        cnt = n_cal * nat[n]["x"][2]
        ax = aggregate_A(nat[n]["x"][0], nat[n]["x"][1], *nat[n]["act"]["x"][f], cnt)[0]
        axb = aggregate_A(nat[n]["x_branch"][0], nat[n]["x_branch"][1], *nat[n]["act"]["x"][f], cnt)[0]
        ay = aggregate_A(nat[n]["y"][0], nat[n]["y"][1], *nat[n]["act"]["y"][f], cnt)[0]
        return ax, ay, axb
    rows = []
    for r in sim:
        f, mode, nodes = r["fmt"], r["mode"], r["nodes"]
        e = {k: r[k] for k in ("name", "layer", "target", "mode", "fmt", "dL_sum", "dL_abs_sum", "f_sum", "f2_sum")}
        W = sum(table["B"][n][f] for n in nodes) if "w" in mode else 0.0
        A = sum(table["C"][n][f] for n in nodes) if "a" in mode else 0.0
        Ax = sum(a_parts(n, f)[0] for n in nodes) if "a" in mode else 0.0
        Ay = sum(a_parts(n, f)[1] for n in nodes) if "a" in mode else 0.0
        Axb = sum(a_parts(n, f)[2] for n in nodes) if "a" in mode else 0.0
        Ab = sum(tbranch["C"][n][f] for n in nodes) if "a" in mode else 0.0
        e.update(W=W, A=A, C=W + A, A_x=Ax, A_y=Ay, A_x_branch=Axb, A_branch=Ab, C_branch=W + Ab)
        if "a" in mode:                 # PyTorch: input part with the total vs the branch gradient of the shared X
            e["A_x_torch_total"] = sum(crit[n]["A"][f]["torch_x_only"] for n in nodes)
            e["A_x_torch_branch"] = sum(crit[n]["A"][f]["torch_branch_x_only"] for n in nodes)
        P = r["parts"]
        for k in ("fW", "fX", "fY"):          # joint: per window summed over the nodes, then squared and summed
            joint = np.sum([P[n][k] for n in nodes], axis=0)
            e[f"{k}2"] = float((joint ** 2).sum())
            e[f"{k}2_additive"] = float(sum((np.array(P[n][k]) ** 2).sum() for n in nodes))
        if r["target"] == "both":
            e["cross_2fgfu"], e["swiglu_second_order"] = r["cross_2fgfu"], r["swiglu_second_order"]
        rows.append(e)
    by = {e["name"]: e for e in rows}
    for e in rows:                                # joint vs single
        if e["target"] == "both":
            g = by[e["name"].replace("_both_", "_gate_")]
            u = by[e["name"].replace("_both_", "_up_")]
            e["dL_gate_plus_up"] = g["dL_sum"] + u["dL_sum"]
            e["dL_joint_minus_sum"] = e["dL_sum"] - e["dL_gate_plus_up"]
            e["f2_gate_plus_up"] = g["f2_sum"] + u["f2_sum"]
            e["C_gate_plus_up"] = g["C"] + u["C"]
            e["C_branch_gate_plus_up"] = g["C_branch"] + u["C_branch"]
    corr = {}
    for mode, crit_key in itertools.product(("w", "a", "wa", "all"), ("C", "C_branch")):
        sel = [e for e in rows if mode == "all" or e["mode"] == mode]
        C = np.array([e[crit_key] for e in sel])
        for key in ("dL_sum", "dL_abs_sum", "f2_sum"):
            v = np.array([e[key] for e in sel])
            ok = (C > 0) & (np.abs(v) > 0)
            corr[f"{mode}|{crit_key}~{key}"] = {
                "spearman": float(spearmanr(C, v)[0]),
                "spearman_vs_abs": float(spearmanr(C, np.abs(v))[0]),
                "pearson_loglog_abs": float(pearsonr(np.log(C[ok]), np.log(np.abs(v[ok])))[0]) if ok.sum() > 2 else None,
                "n": len(sel)}
    corr["ceiling|f2_sum~dL_abs_sum"] = {"spearman": float(spearmanr([e["f2_sum"] for e in rows],
                                                                     [abs(e["dL_sum"]) for e in rows])[0]),
                                         "n": len(rows)}
    # decisions inside one layer: which single branch hurts more (same mode and format), by C, C_branch and the
    # simulated first order; the w mode is the same for both attributions
    decisions = {}
    for key in ("C", "C_branch", "f2_sum"):
        for modes in (("w",), ("a", "wa")):
            pairs = [(e, by[e["name"].replace("_gate_", "_up_")]) for e in rows
                     if e["target"] == "gate" and e["mode"] in modes]
            miss = [[g["name"], g["dL_sum"], u["dL_sum"], g[key], u[key]] for g, u in pairs
                    if (g[key] > u[key]) != (g["dL_sum"] > u["dL_sum"])]
            decisions[f"gate_vs_up|{key}|{'+'.join(modes)}"] = {"agree": len(pairs) - len(miss), "n": len(pairs),
                                                                "misses": miss}
    both = [e for e in rows if e["target"] == "both"]
    single_max = {e["name"]: max(by[e["name"].replace("_both_", s)]["dL_sum"] for s in ("_gate_", "_up_"))
                  for e in both}
    decisions["joint_below_a_single"] = [      # an additive criterion always ranks the joint above each single
        [e["name"], e["dL_sum"], by[e["name"].replace("_both_", "_gate_")]["dL_sum"],
         by[e["name"].replace("_both_", "_up_")]["dL_sum"]] for e in both
        if e["dL_sum"] < single_max[e["name"]] and single_max[e["name"]] > 1]   # loss deltas below 1: noise level
    big = [e for e in both if abs(e["dL_joint_minus_sum"]) > 1]
    decisions["sign_nonadditivity_eq_sign_cross"] = {
        "agree": int(sum(np.sign(e["dL_joint_minus_sum"]) == np.sign(e["cross_2fgfu"]) for e in big)), "n": len(big)}
    # the shared input quantized on both edges with the same quantizer: one dX, so the joint first order of the
    # input part is (f_X,gate + f_X,up); against the criterion's input part counted once (total gradient) or
    # summed over the branches
    xs = [e for e in both if "a" in e["mode"]]
    decisions["joint_input_part"] = {
        "fX2_joint_over_additive_median": float(np.median([e["fX2"] / e["fX2_additive"] for e in xs])),
        "A_x_total_once_over_fX2_joint_median": float(np.median([e["A_x"] / 2 / e["fX2"] for e in xs])),
        "A_x_branch_sum_over_fX2_joint_median": float(np.median([e["A_x_branch"] / e["fX2"] for e in xs])),
        "A_x_total_twice_over_fX2_joint_median": float(np.median([e["A_x"] / e["fX2"] for e in xs])),
        "rows": [[e["name"], e["fX2"], e["fX2_additive"], e["A_x"] / 2, e["A_x_branch"]] for e in xs]}
    (out / "analysis.json").write_text(json.dumps({"rows": rows, "correlations": corr, "decisions": decisions},
                                                  indent=1) + "\n")
    import csv
    keys = []
    for e in rows:
        keys += [k for k in e if k not in keys]
    with open(out / "analysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for e in rows:
            w.writerow(e)
    log(json.dumps(corr, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("sim", "native", "analyze"), required=True)
    ap.add_argument("--native-run", type=Path, default=HERE / "runs/native_bytecode_20261012/qwen3")
    ap.add_argument("--layers", default="2,0,14,27")
    ap.add_argument("--formats", default="int4,int8,p6e1")
    ap.add_argument("--native-configs", default="2:gate:wa:int4;2:up:wa:int4;2:both:wa:int4;2:both:a:p6e1;"
                                                "2:both:w:int4;27:both:wa:int4;0:both:a:int8")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--criterion-dir", type=Path, help="qwen3_swiglu_criterion.py output (native_raw)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.native_run = args.native_run.resolve()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    if args.stage == "analyze":                    # native_stats() reads <out_dir>/native_raw
        args.out_dir_saved, args.out_dir = args.out_dir, args.criterion_dir.resolve()
        out = args.out_dir_saved
    else:
        out = args.out_dir
        (out / f"environment_{args.stage}.json").write_text(json.dumps(environment(
            ["experiments/mixed_precision/qwen3_swiglu_quant.py"]), indent=1) + "\n")
    {"sim": stage_sim, "native": stage_native, "analyze": stage_analyze}[args.stage](args, out, log)


if __name__ == "__main__":
    main()
