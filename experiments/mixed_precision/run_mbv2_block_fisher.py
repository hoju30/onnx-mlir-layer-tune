#!/usr/bin/env python3
"""MobileNetV2: BRECQ-style block sensitivity in HTQ's Eq. 5 -- does scoring a block's layers together help?

Formats are still chosen per layer (one (w, a) per node, as MIXEDP_NODE_FORMATS needs); only the cost changes:
  node_full     per-tensor Fisher per node, all 8x8 options, summed over nodes (the current criterion)
  node_pruned   the same, restricted to each node's candidate list (below), summed over nodes
  block         per block b (timm InvertedResidual / DepthwiseSeparable: its 2-3 convs together) and combination
                c_b of its nodes' candidates: S_b * sum ||z_b(c_b) - z_b||^2, z_b the block output computed from the
                FP32 block input with every conv's weight, input and output fake-quantized as the pass does,
                S_b = mean((dL/dz_b)^2) (BRECQ: block-diagonal Fisher, blocks additive). Stem, head and classifier
                are blocks of one node (z = the node's output), so every cost is on the same scale.
  sqnr          reference: the saved SQNR MILP config (runs/mobilenetv2_timm_ra_criteria_milp)
Candidate list of a node: the formats the node_full MILP picks over budgets 0.50..0.95, filled up to --candidates
with that node's (Fisher cost, bytes, BOPs) Pareto options nearest in resource use.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

from eval_random_mixed_precision import (calibrate_ranges, np_fake_quant_weight, preprocess_by_source,  # noqa: E402
                                         simulate_in_onnxruntime, supported_nodes)
from htq_pareto_search import FORMAT_BITS  # noqa: E402
from mp_milp import milp_select  # noqa: E402
from run_mbv2_fisher_term_ablation import FORMATS, act_fake_quant, fold_batchnorms  # noqa: E402
from sample_configs_htq import compute_weight_elems_and_flops_proxy  # noqa: E402
from sample_configs_sqnr import config_cost  # noqa: E402

OPTIONS = [(w, a) for w in FORMATS for a in FORMATS]


def select_groups(groups, budget_bytes, budget_bops, time_limit=60.0):
    """groups: list of option lists [(cost, bytes, bops, payload)]; pick one option per group, minimize cost
    within both budgets. Returns (payloads, status)."""
    cost = np.array([o[0] for g in groups for o in g])
    size = np.array([o[1] for g in groups for o in g])
    bops = np.array([o[2] for g in groups for o in g])
    lens = [len(g) for g in groups]
    off = np.cumsum([0] + lens)
    rows = np.concatenate([np.repeat(np.arange(len(groups)), lens), np.full(len(cost), len(groups)),
                           np.full(len(cost), len(groups) + 1)])
    cols = np.concatenate([np.arange(len(cost))] * 3)
    vals = np.concatenate([np.ones(len(cost)), size / budget_bytes, bops / budget_bops])
    A = coo_matrix((vals, (rows, cols)), shape=(len(groups) + 2, len(cost))).tocsr()
    scale = float(np.abs(cost).max()) / 1e4 or 1.0
    res = milp(cost / scale, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, np.r_[np.ones(len(groups)), -np.inf, -np.inf],
                                            np.r_[np.ones(len(groups)), 1 - 1e-9, 1 - 1e-9]),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(res.message)
    picks = [groups[k][int(np.argmax(res.x[off[k]:off[k + 1]]))][3] for k in range(len(groups))]
    return picks, int(res.status)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=HERE / "../imagenet100/model_dataset/mobilenetv2/timm_ra")
    ap.add_argument("--timm-name", default="mobilenetv2_100.ra_in1k")
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--candidates", type=int, default=6, help="per-node candidate formats for the block search")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budgets = [float(f) for f in args.budgets.split(",")]
    t0 = time.time()
    torch.set_num_threads(8)

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]]
    cx, cy = preprocess_by_source(args.images_dir, {"images": calib}, meta), np.array([e["label"] for e in calib])
    ex, ey = preprocess_by_source(args.images_dir, {"images": held}, meta), np.array([e["label"] for e in held])
    all_nodes = supported_nodes(graph)
    nodes = [n for n in all_nodes if n.get("weight_const")]
    ranges = calibrate_ranges(graph, all_nodes, cx)
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}

    import timm
    model = timm.create_model(args.timm_name, pretrained=True).eval()
    fold_batchnorms(model)
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(layers) == len(nodes)
    module_of, path_of = {}, {}
    for n, (path, m) in zip(nodes, layers):
        np.testing.assert_allclose(m.weight.detach().numpy(), inits[by_name[n["name"]].input[1]], rtol=1e-4, atol=1e-5)
        module_of[n["name"]], path_of[n["name"]] = m, path
    names = list(module_of)
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    # blocks: nodes sharing a parent module under "blocks."; every other node is a block of its own (its output)
    parent = {n: path_of[n].rsplit(".", 1)[0] if path_of[n].startswith("blocks.") else path_of[n] for n in names}
    block_paths = sorted(set(parent.values()), key=lambda p: names.index(next(n for n in names if parent[n] == p)))
    block_nodes = {p: [n for n in names if parent[n] == p] for p in block_paths}
    mods = dict(model.named_modules())
    log(f"{len(names)} nodes in {len(block_paths)} blocks: " + ", ".join(f"{p} ({len(v)})" for p, v in block_nodes.items()))

    # --- gradients: per-tensor Fisher of every node, block-output Fisher of every block ---
    cap, blk = {}, {}

    def node_hook(name):
        def fn(mod, inp, out):
            inp[0].retain_grad()
            out.retain_grad()
            cap[name] = (inp[0], out)
        return fn

    def block_hook(path):
        def fn(mod, inp, out):
            out.retain_grad()
            blk[path] = (inp[0].detach().clone(), out)
        return fn

    handles = [module_of[n].register_forward_hook(node_hook(n)) for n in names]
    handles += [mods[p].register_forward_hook(block_hook(p)) for p in block_paths]
    acc = {n: {"g2w": 0.0, **{s: {"g2": 0.0, "count": 0, "d2": dict.fromkeys(FORMATS, 0.0)} for s in ("x", "y")}}
           for n in names}
    bacc = {p: {"g2": 0.0, "count": 0, "inputs": []} for p in block_paths}
    for k in range(len(cy)):
        model.zero_grad(set_to_none=True)
        x0 = torch.from_numpy(cx[k:k + 1]).requires_grad_(True)
        F.cross_entropy(model(x0), torch.tensor(cy[k:k + 1])).backward()
        with torch.no_grad():
            for n, (x, y) in cap.items():
                acc[n]["g2w"] += float((module_of[n].weight.grad.double() ** 2).sum())
                for side, t, rng_name in (("x", x, tens[n][0]), ("y", y, tens[n][1])):
                    s = acc[n][side]
                    s["g2"] += float((t.grad.double() ** 2).sum())
                    s["count"] += t.numel()
                    for f in FORMATS[:-1]:
                        s["d2"][f] += float(((act_fake_quant(t, f, ranges[rng_name]) - t).double() ** 2).sum())
            for p, (bx, bz) in blk.items():
                b = bacc[p]
                b["g2"] += float((bz.grad.double() ** 2).sum())
                b["count"] += bz.numel()
                b["inputs"].append(bx)
        cap.clear()
        blk.clear()
    for h in handles:
        h.remove()
    T, qw = {}, {}
    for n in names:
        W = inits[by_name[n].input[1]]
        a = acc[n]
        Wm = module_of[n].weight.detach().clone()
        qw[n] = {"fp32": Wm, **{f: torch.from_numpy(np_fake_quant_weight(Wm.numpy(), f, None)) for f in FORMATS[:-1]}}
        T[n] = {"w": {"fp32": 0.0, **{f: a["g2w"] / W.size * float(((np_fake_quant_weight(W, f, None).astype(np.float64)
                                                                      - W.astype(np.float64)) ** 2).sum())
                                      for f in FORMATS[:-1]}},
                **{s: {"fp32": 0.0, **{f: a[s]["g2"] / a[s]["count"] * a[s]["d2"][f] for f in FORMATS[:-1]}}
                   for s in ("x", "y")}}

    def node_cost(n, w, a):
        return T[n]["w"][w] + T[n]["x"][a] + T[n]["y"][a]

    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)

    def res_of(n, w, a):
        return weight_elems[n] * FORMAT_BITS[w] / 8, flops_proxy[n] * FORMAT_BITS[w] * FORMAT_BITS[a]

    # --- candidate formats per node ---
    picked = {n: set() for n in names}
    for f in sorted(set(np.round(np.arange(0.5, 0.951, 0.05), 2).tolist()) | set(budgets)):
        cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, node_cost, weight_elems, flops_proxy,
                                     f * int8_bytes, f * int8_bops, time_limit=60)
        for n in names:
            picked[n].add((cfg[n]["w"], cfg[n]["a"]))
    cand = {}
    for n in names:
        pts = [(node_cost(n, w, a), *res_of(n, w, a), (w, a)) for w, a in OPTIONS]
        pareto = [p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and q[:3] != p[:3]
                                            for q in pts)]
        chosen = list(picked[n])
        load = lambda o: FORMAT_BITS[o[0]] / 8 + FORMAT_BITS[o[0]] * FORMAT_BITS[o[1]] / 64  # noqa: E731
        extra = sorted((p[3] for p in pareto if p[3] not in chosen),
                       key=lambda o: min(abs(load(o) - load(c)) for c in chosen))
        cand[n] = (chosen + extra)[:max(args.candidates, len(chosen))]
    log("per-node candidates: " + ", ".join(f"{len(cand[n])}" for n in names))

    # --- block costs: every combination of the block's candidates, block output from the FP32 block input ---
    current = {}

    def qpre(name):
        def fn(mod, inp):
            return (act_fake_quant(inp[0], current[name][1], ranges[tens[name][0]]),) + tuple(inp[1:])
        return fn

    def qpost(name):
        def fn(mod, inp, out):
            return act_fake_quant(out, current[name][1], ranges[tens[name][1]])
        return fn

    block_opts = {}
    with torch.no_grad():
        for p in block_paths:
            ns = block_nodes[p]
            orig = {n: module_of[n].weight.data.clone() for n in ns}
            xin = torch.cat(bacc[p]["inputs"])
            zref = mods[p](xin).double()
            hs = [module_of[n].register_forward_pre_hook(qpre(n)) for n in ns] + \
                 [module_of[n].register_forward_hook(qpost(n)) for n in ns]
            S_b = bacc[p]["g2"] / bacc[p]["count"]
            opts = []
            for combo in itertools.product(*(cand[n] for n in ns)):
                for n, (w, a) in zip(ns, combo):
                    current[n] = (w, a)
                    module_of[n].weight.data.copy_(qw[n][w])
                z = mods[p](xin).double()
                size = sum(res_of(n, w, a)[0] for n, (w, a) in zip(ns, combo))
                bops = sum(res_of(n, w, a)[1] for n, (w, a) in zip(ns, combo))
                opts.append((S_b * float(((z - zref) ** 2).sum()), size, bops, dict(zip(ns, combo)),
                             sum(node_cost(n, w, a) for n, (w, a) in zip(ns, combo))))
            for n in ns:
                module_of[n].weight.data.copy_(orig[n])
            for h in hs:
                h.remove()
            block_opts[p] = opts
            ratio = [o[0] / o[4] for o in opts if o[4] > 0]
            log(f"{p}: {len(opts)} combinations, block / node-sum cost median {np.median(ratio):.2f} "
                f"(range {min(ratio):.2f}-{max(ratio):.2f})")
    log(f"block costs: {sum(len(v) for v in block_opts.values())} block combinations scored")

    def groups(kind):
        return [[((o[0] if kind == "block" else o[4]), o[1], o[2], o[3]) for o in block_opts[p]] for p in block_paths]

    sq = [c for c in json.loads((HERE / "runs/mobilenetv2_timm_ra_criteria_milp/configs.json").read_text())["configs"]
          if c["criterion"] == "sqnr"]
    res, rows = {}, []
    for f in budgets:
        configs = {}
        cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, node_cost, weight_elems, flops_proxy,
                                     f * int8_bytes, f * int8_bops, time_limit=60)
        configs["node_full"] = ({n: (cfg[n]["w"], cfg[n]["a"]) for n in names}, status)
        for kind in ("node_pruned", "block"):
            picks, status = select_groups(groups("block" if kind == "block" else "node"), f * int8_bytes, f * int8_bops)
            configs[kind] = ({n: wa for d in picks for n, wa in d.items()}, status)
        s = next(c for c in sq if abs(c["budget"] - f) < 1e-9)
        configs["sqnr"] = ({n: (s["formats"][n]["w"], s["formats"][n]["a"]) for n in names}, 0)
        for kind, (pick, status) in configs.items():
            fmts = {n: {"w": w, "a": a} for n, (w, a) in pick.items()}
            size, bops = config_cost({**{n["name"]: {"w": None, "a": "fp32"} for n in all_nodes if not n.get("weight_const")},
                                      **fmts}, weight_elems, flops_proxy)
            z = simulate_in_onnxruntime(graph, all_nodes, fmts, ranges, ex, None)
            finite = np.isfinite(z).all(1)
            correct = ((z.argmax(1) == ey) & finite).astype(float)
            loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(ey),
                                   reduction="none").numpy()
            res[(f, kind)] = (correct, loss)
            rows.append({"budget": f, "variant": kind, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "formats": {n: list(wa) for n, wa in pick.items()}})
            log(f"f={f:.2f} {kind:<12} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in rows:
        c, l = res[(r["budget"], r["variant"])]
        cb, lb = res[(r["budget"], "node_pruned")]
        d1, dl = (c - cb) * 100, l - lb
        r.update(dtop1_pp=float(d1.mean()), dtop1_se=float(d1.std(ddof=1) / np.sqrt(len(d1))),
                 dloss=float(dl.mean()), dloss_se=float(dl.std(ddof=1) / np.sqrt(len(dl))))
    (args.out_dir / "results.json").write_text(json.dumps(
        {"backend": "ORT QDQ simulation", "model": args.timm_name, "budgets": budgets, "candidates": args.candidates,
         "candidate_formats": {n: [list(o) for o in cand[n]] for n in names}, "blocks": block_nodes, "rows": rows},
        indent=2) + "\n")
    print(f"\n{'f':>5} {'variant':<12} {'top1':>7} {'loss':>7} {'d top1 vs node_pruned':>22} {'d loss':>17} {'size':>5} {'BOPs':>5}")
    for r in rows:
        d = "" if r["variant"] == "node_pruned" else f"{r['dtop1_pp']:+6.2f}±{r['dtop1_se']:.2f}"
        dl = "" if r["variant"] == "node_pruned" else f"{r['dloss']:+.4f}±{r['dloss_se']:.4f}"
        print(f"{r['budget']:5.2f} {r['variant'] + ('' if r['milp_optimal'] else '*'):<12} {r['top1']:7.1%} "
              f"{r['loss']:7.4f} {d:>22} {dl:>17} {r['size']:5.2f} {r['bops']:5.2f}")


if __name__ == "__main__":
    main()
