#!/usr/bin/env python3
"""C (channel bias-aware Fisher) vs SQNR on one ImageNet model: cost, per-node prediction, MILP picks.

Criteria (per node and (w, a); the per-tensor weight term mean(g_W^2) * ||Q(W) - W||^2 in A and C):
  A     per-tensor Fisher: activation term mean(g^2) * sum delta^2 per tensor (input X, output Y)
  C     channel bias-aware: sum_c [ mean_c(g^2) * sum_{j in c} (delta_j - mu_c)^2 + mu_c^2 * sum_img (sum_{j in c} g_j)^2 ]
        (run_mbv2_channel_fisher.py); one forward + backward per calibration image (PyTorch, BatchNorm folded)
  SQNR  sum ||logits - logits_fp32||^2 with only that node quantized (sqnr_tables.py; one ORT run per node and option)
Cost: each criterion table is built in its own child process (--phase fisher / --phase sqnr) on the same calibration
images, after the shared loading (model, images, activation ranges): wall time of the table, and the RSS sampled
every 20 ms while it is built (peak, and peak minus the RSS just before).
Prediction: single-node held-out loss increase at w/a = int4/int8, int8/int4, int6/int6 (ORT), all-pairs Spearman
and IQR of log10(pred / dloss) (the all-pairs numbers track the MILP outcome; per-config ones do not).
MILP: budgets relative to all-int8 weight bytes and BOPs, ORT QDQ simulation on the held-out images.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_mbv2_fisher_term_ablation import FORMATS  # noqa: E402

CONFIGS = (("int4", "int8"), ("int8", "int4"), ("int6", "int6"))
AF = FORMATS[:-1]


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


class RssSampler:
    """Peak resident set size while the `with` block runs, sampled every 20 ms."""
    def __enter__(self):
        import threading
        self.before = self.peak = rss_mb()
        self.stop = threading.Event()

        def run():
            while not self.stop.wait(0.02):
                self.peak = max(self.peak, rss_mb())
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, rss_mb())

    def report(self, passes):
        return {"seconds": self.seconds, "rss_before_mb": self.before, "peak_rss_mb": self.peak,
                "increment_mb": self.peak - self.before, "process_peak_rss_mb": peak_rss_mb(), "passes": passes}


def common(args, n_eval=None):
    import onnx
    from eval_random_mixed_precision import calibrate_ranges, preprocess_by_source, supported_nodes
    graph = onnx.load(str(args.model_dir / "model.onnx"))
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads((args.images_dir / "manifest.json").read_text())
    calib = [e for e in manifest["images"] if e["calibration"]]
    held = [e for e in manifest["images"] if not e["calibration"]][:n_eval]
    cx, cy = preprocess_by_source(args.images_dir, {"images": calib}, meta), np.array([e["label"] for e in calib])
    all_nodes = supported_nodes(graph)
    ranges = calibrate_ranges(graph, all_nodes, cx)
    return graph, meta, held, cx, cy, all_nodes, ranges


def torch_model(meta):
    import torch.nn as nn
    from run_mbv2_fisher_term_ablation import fold_batchnorms
    if meta["source"] == "timm":
        import timm
        model = timm.create_model(meta["timm_model"], pretrained=True)
    elif meta["source"] == "torchvision":
        import torchvision
        model = getattr(torchvision.models, meta["torchvision_model"])(weights=meta["torchvision_weights"])
    else:
        raise ValueError(f"no PyTorch model for source {meta['source']}")
    model.eval()
    fold_batchnorms(model)
    for m in model.modules():   # an in-place activation (e.g. timm SE's act1) would overwrite a captured conv output
        if hasattr(m, "inplace"):
            m.inplace = False
    return model, [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def match_modules(graph, nodes, layers):
    """ONNX weighted node -> PyTorch module with the same (folded) weight; raises if any node is unmatched."""
    import onnx
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    by_name = {n.name: n for n in graph.graph.node}
    free, out = list(layers), {}
    for n in nodes:
        W = inits[by_name[n["name"]].input[1]]
        best = None
        for m in free:
            w = m.weight.detach().numpy()
            for cand in (w, w.T) if w.ndim == 2 else (w,):
                if cand.shape == W.shape:
                    err = float(np.abs(cand - W).max() / (np.abs(W).max() + 1e-12))
                    if best is None or err < best[0]:
                        best = (err, m)
        if best is None or best[0] > 1e-3:
            raise RuntimeError(f"{n['name']}: no PyTorch layer matches its weight (best rel. error "
                               f"{None if best is None else best[0]})")
        out[n["name"]] = best[1]
        free.remove(best[1])
    return out, inits, by_name


def phase_fisher(args):
    import torch
    import torch.nn.functional as F
    from eval_random_mixed_precision import np_fake_quant_weight
    from run_mbv2_act_lowbit_check import chan_sum
    from run_mbv2_fisher_term_ablation import act_fake_quant
    graph, meta, _, cx, cy, all_nodes, ranges = common(args, 0)
    nodes = [n for n in all_nodes if n.get("weight_const")]
    model, layers = torch_model(meta)
    module_of, inits, by_name = match_modules(graph, nodes, layers)
    names = [n["name"] for n in nodes]
    tens = {n["name"]: (n["act_tensors"][0], n["act_tensors"][-1]) for n in nodes}
    sampler = RssSampler().__enter__()
    cap = {}
    seen = {}
    for n, m in module_of.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, (inp[0], out)))
    g2w = dict.fromkeys(names, 0.0)
    st = {n: {s: {"g2": 0.0, "gc": 0.0, "n": 0, "s1": dict.fromkeys(AF, 0.0), "s2": dict.fromkeys(AF, 0.0)}
              for s in ("x", "y")} for n in names}
    for k in range(len(cy)):
        z = model(torch.from_numpy(cx[k:k + 1]).requires_grad_(True))
        targets = [t for n in names for t in (module_of[n].weight, *cap[n])]
        grads = torch.autograd.grad(F.cross_entropy(z, torch.tensor([int(cy[k])])), targets)
        with torch.no_grad():
            for i, n in enumerate(names):
                g2w[n] += float((grads[3 * i].double() ** 2).sum())
                for j, s in enumerate(("x", "y")):
                    t, g, a = cap[n][j], grads[3 * i + 1 + j].double(), st[n][s]
                    lo, hi = seen.get((n, j), (float("inf"), float("-inf")))
                    seen[(n, j)] = (min(lo, float(t.min())), max(hi, float(t.max())))
                    a["g2"] = a["g2"] + chan_sum(g ** 2)
                    a["gc"] = a["gc"] + chan_sum(g) ** 2
                    a["n"] += t.numel() // t.shape[1]
                    for f in AF:
                        d = (act_fake_quant(t, f, ranges[tens[n][j]]) - t).double()
                        a["s1"][f] = a["s1"][f] + chan_sum(d)
                        a["s2"][f] = a["s2"][f] + chan_sum(d ** 2)
        cap.clear()
    tables = {"CW": {}, "CA_A": {}, "CA_C": {}}
    for n in names:
        W = inits[by_name[n].input[1]]
        tables["CW"][n] = {"fp32": 0.0, **{f: g2w[n] / W.size * float(((np_fake_quant_weight(W, f, None).astype(
            np.float64) - W.astype(np.float64)) ** 2).sum()) for f in AF}}
        tables["CA_A"][n], tables["CA_C"][n] = {"fp32": 0.0}, {"fp32": 0.0}
        for f in AF:
            a_cost = c_cost = 0.0
            for s in ("x", "y"):
                a = st[n][s]
                S_c = a["g2"] / a["n"]
                mu = a["s1"][f] / a["n"]
                a_cost += float(a["g2"].sum()) / (a["n"] * len(a["g2"])) * float(a["s2"][f].sum())
                c_cost += float((S_c * (a["s2"][f] - a["n"] * mu ** 2)).sum() + (a["gc"] * mu ** 2).sum())
            tables["CA_A"][n][f], tables["CA_C"][n][f] = a_cost, c_cost
    sampler.__exit__(None, None, None)
    mismatch = max(max(abs(seen[(n, j)][0] - ranges[tens[n][j]][0]), abs(seen[(n, j)][1] - ranges[tens[n][j]][1]))
                   / max(1e-6, ranges[tens[n][j]][1] - ranges[tens[n][j]][0]) for n in names for j in (0, 1))
    if mismatch > 1e-2:
        raise RuntimeError(f"PyTorch activations do not match the ONNX calibration ranges (rel. {mismatch:.2e})")
    kinds = {n: ("fc" if m.weight.dim() == 2 else "dw" if m.groups > 1 and m.groups == m.in_channels
                 else "pw" if m.kernel_size == (1, 1) else "conv") for n, m in module_of.items()}
    (args.out_dir / "fisher_tables.json").write_text(json.dumps(
        {"tables": tables, "kind": kinds, "range_mismatch": mismatch,
         "cost": {**sampler.report(f"{len(cy)} x (forward + backward), PyTorch CPU"),
                  "threads": torch.get_num_threads()}}) + "\n")
    print(f"fisher: {len(names)} nodes, {sampler.seconds:.1f} s, peak RSS {sampler.peak:.0f} MB "
          f"(+{sampler.peak - sampler.before:.0f} MB); activation ranges match ONNX to {mismatch:.1e}")


def phase_sqnr(args):
    from sqnr_tables import sqnr_table
    graph, meta, _, cx, cy, all_nodes, ranges = common(args, 0)
    with RssSampler() as sampler:
        table = sqnr_table(graph, all_nodes, FORMATS, FORMATS, ranges, cx, None, log=lambda m: print(m, flush=True))
    n_runs = sum(1 for r in table.values() for k in r if k != "fp32|fp32")
    (args.out_dir / "sqnr_table.json").write_text(json.dumps(
        {"table": table, "cost": sampler.report(f"{n_runs + 1} ORT runs x {len(cy)} images")}) + "\n")
    print(f"sqnr: {len(table)} nodes, {sampler.seconds:.1f} s, peak RSS {sampler.peak:.0f} MB "
          f"(+{sampler.peak - sampler.before:.0f} MB)")


def driver(args):
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr
    from eval_random_mixed_precision import preprocess_by_source, simulate_in_onnxruntime
    from mp_milp import milp_select
    from sample_configs_htq import compute_weight_elems_and_flops_proxy
    from sample_configs_sqnr import config_cost
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    for phase, out in (("fisher", "fisher_tables.json"), ("sqnr", "sqnr_table.json")):
        if not (args.out_dir / out).exists():
            with open(args.out_dir / f"{phase}.log", "w") as fh:
                subprocess.run([sys.executable, __file__, "--phase", phase, "--model-dir", str(args.model_dir),
                                "--images-dir", str(args.images_dir), "--out-dir", str(args.out_dir)],
                               stdout=fh, stderr=subprocess.STDOUT, check=True)
        log((args.out_dir / f"{phase}.log").read_text().strip().splitlines()[-1])
    fis = json.loads((args.out_dir / "fisher_tables.json").read_text())
    sq = json.loads((args.out_dir / "sqnr_table.json").read_text())
    T, kind = fis["tables"], fis["kind"]

    def cost(v, n, w, a):
        if v == "SQNR":
            return sq["table"][n][f"{w}|{a}"]
        return T["CW"][n][w] + T["CA_" + v][n][a]

    graph, meta, held, cx, cy, all_nodes, ranges = common(args, args.num_eval)
    names = [n["name"] for n in all_nodes if n.get("weight_const")]
    ex = preprocess_by_source(args.images_dir, {"images": held}, meta)
    ey = np.array([e["label"] for e in held])

    def per_image(cfg, x, y):
        z = simulate_in_onnxruntime(graph, all_nodes, cfg, ranges, x, None)
        finite = np.isfinite(z).all(1)
        loss = F.cross_entropy(torch.from_numpy(np.where(finite[:, None], z, 0.0)), torch.from_numpy(y),
                               reduction="none").numpy()
        return ((z.argmax(1) == y) & finite).astype(float), loss

    # --- per-node prediction ---
    if args.truth_from or (args.out_dir / "truth.json").exists():
        src = args.truth_from or args.out_dir / "truth.json"
        truth = [r for r in json.loads(src.read_text())["rows"]]
        log(f"single-node truth from {src}")
    else:
        xt, yt = ex[:args.num_truth], ey[:args.num_truth]
        _, base = per_image({}, xt, yt)
        truth = []
        for i, n in enumerate(names):
            for w, a in CONFIGS:
                d = per_image({n: {"w": w, "a": a}}, xt, yt)[1] - base
                truth.append({"node": n, "w": w, "a": a, "dloss": float(d.mean()),
                              "dloss_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
            if (i + 1) % 10 == 0:
                log(f"single-node truth: {i + 1}/{len(names)} nodes")
        (args.out_dir / "truth.json").write_text(json.dumps({"num_images": len(yt), "rows": truth}, indent=1) + "\n")
    y = np.array([r["dloss"] for r in truth])
    sig = np.array([r["dloss"] > 2 * r["dloss_se"] for r in truth])
    pred_stats = {}
    for v in ("A", "C", "SQNR"):
        p = np.array([cost(v, r["node"], r["w"], r["a"]) for r in truth], dtype=float)
        lr = np.log10(np.clip(p[sig], 1e-300, None) / y[sig])
        cen = lr - np.median(lr)
        srows = [r for r, s in zip(truth, sig) if s]
        pred_stats[v] = {"spearman": float(spearmanr(p, y)[0]), "spearman_sig": float(spearmanr(p[sig], y[sig])[0]),
                         "iqr": float(np.subtract(*np.percentile(lr, [75, 25]))),
                         "by_config": {f"{w}/{a}": float(spearmanr(p[ix], y[ix])[0]) for w, a in CONFIGS
                                       for ix in [np.array([(r["w"], r["a"]) == (w, a) for r in truth])]},
                         "kind_bias": {k: float(np.median(cen[[kind[r["node"]] == k for r in srows]]))
                                       for k in ("conv", "pw", "dw", "fc") if any(kind[r["node"]] == k for r in srows)}}

    # --- MILP picks ---
    weight_elems, flops_proxy = compute_weight_elems_and_flops_proxy(graph, all_nodes)
    int8_bytes, int8_bops = sum(weight_elems[n] for n in names), sum(64 * flops_proxy[n] for n in names)
    res, rows = {}, []
    for f in [float(b) for b in args.budgets.split(",")]:
        for v in ("A", "C", "SQNR"):
            cfg, _, status = milp_select(all_nodes, FORMATS, FORMATS, lambda n, w, a, v=v: cost(v, n, w, a),
                                         weight_elems, flops_proxy, f * int8_bytes, f * int8_bops, time_limit=60)
            assert status in (0, 1), (v, f, status)
            cfg = {n: {"w": cfg[n]["w"], "a": cfg[n]["a"]} for n in names}
            correct, loss = per_image(cfg, ex, ey)
            res[(f, v)] = (correct, loss)
            size, bops = config_cost(cfg, weight_elems, flops_proxy)
            rows.append({"budget": f, "variant": v, "milp_optimal": status == 0, "size": size / int8_bytes,
                         "bops": bops / int8_bops, "top1": float(correct.mean()), "loss": float(loss.mean()),
                         "formats": cfg})
            log(f"f={f:.2f} {v:<5} top1 {correct.mean():6.1%}  loss {loss.mean():.4f}"
                + ("" if status == 0 else "  (MILP time limit)"))
    for r in rows:
        c, l = res[(r["budget"], r["variant"])]
        for ref in ("A", "SQNR"):
            cb, lb = res[(r["budget"], ref)]
            r[f"vs_{ref}"] = {"dtop1_pp": float(((c - cb) * 100).mean()), "dloss": float((l - lb).mean()),
                              "dloss_se": float((l - lb).std(ddof=1) / np.sqrt(len(l)))}
    _, fp32_loss = per_image({}, ex, ey)
    fp32_top1 = float(per_image({}, ex, ey)[0].mean())
    summary = {"model": str(args.model_dir), "nodes": len(names), "num_eval": len(ey), "num_truth_pairs": len(truth),
               "fp32": {"top1": fp32_top1, "loss": float(fp32_loss.mean())},
               "cost": {"fisher_A_and_C": fis["cost"], "sqnr": sq["cost"]}, "prediction": pred_stats, "rows": rows}
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")

    fc, sc = fis["cost"], sq["cost"]
    print(f"\n=== {args.model_dir} ({len(names)} weighted nodes; FP32 top-1 {fp32_top1:.1%}, loss {fp32_loss.mean():.4f})")
    for label, c in (("A/C ", fc), ("SQNR", sc)):
        print(f"cost  {label}: {c['seconds']:8.1f} s   peak RSS {c['peak_rss_mb']:6.0f} MB (+{c['increment_mb']:.0f} MB "
              f"over {c['rss_before_mb']:.0f} MB loaded)   [{c['passes']}]")
    print(f"      time ratio SQNR / C = {sc['seconds'] / fc['seconds']:.0f}x")
    print(f"prediction ({len(truth)} single-node pairs, {sig.sum()} with dloss > 2 SE): all-pairs Spearman / sig / IQR; "
          f"per config; per-kind bias")
    for v, s in pred_stats.items():
        print(f"  {v:<5} {s['spearman']:.3f} / {s['spearman_sig']:.3f} / {s['iqr']:.2f}   "
              + " ".join(f"{k} {x:.3f}" for k, x in s["by_config"].items()) + "   "
              + " ".join(f"{k} {b:+.2f}" for k, b in s["kind_bias"].items()))
    print("MILP picks, held-out: top-1 / loss (d loss vs SQNR)")
    for v in ("A", "C", "SQNR"):
        sel = [r for r in rows if r["variant"] == v]
        print(f"  {v:<5} " + "  ".join(f"f={r['budget']:.2f} {r['top1']:5.1%}/{r['loss']:.3f}" for r in sel)
              + f"   mean {np.mean([r['top1'] for r in sel]):5.1%}/{np.mean([r['loss'] for r in sel]):.3f}"
              + ("" if v == "SQNR" else f"  d loss vs SQNR {np.mean([r['vs_SQNR']['dloss'] for r in sel]):+.3f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("driver", "fisher", "sqnr"), default="driver")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images")
    ap.add_argument("--budgets", default="0.6,0.65,0.7,0.75,0.8")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--num-truth", type=int, default=300, help="held-out images for the single-node truth")
    ap.add_argument("--truth-from", type=Path, help="reuse a single-node truth (rows with node, w, a, dloss, dloss_se)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.phase == "driver":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        driver(args)
    else:
        {"fisher": phase_fisher, "sqnr": phase_sqnr}[args.phase](args)


if __name__ == "__main__":
    main()
