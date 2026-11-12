#!/usr/bin/env python3
"""Qwen3-0.6B: does curvature measured on the calibration windows transfer to new data, and is it worth its cost?

Six fixed cases: layers 0, 2, 27 x Gate-only / Up-only with an int4 activation (option (fp32, int4) of
runs/qwen3_block_criterion_20261014: the pass's per-consumer QDQ, the calibration (scale, zp)); error direction
e = Y_hat - Y_fp32 at the block output Y (Down FP32). The six have equal model size and BOPs (Gate and Up projections
have one shape), so every pair among them is an equal-resource decision; three are Gate-vs-Up.

Scores (calibration windows 0-9 only, at Y, summed over windows; a2 = second-order coefficient of the loss along e):
  Fisher (squared gradient)  1/2 sum_s (g_s . e_s)^2      Fisher diag  1/2 sum g^2 e^2   (g = gY, one backward)
  HVP                        1/2 sum_s e_s^T H_s e_s       (Hessian-vector product, double backward)
  FD(h)                      sum_s (dL_s(h) + dL_s(-h)) / 2h^2,  h = 1/8, 1/16 (primary), 1/32
  BC                         the native table's activation cost of the node (practical reference)
Data: calibration windows 0-9 (scores); windows 10-29 were analyzed by earlier studies and are not used; new
validation windows 30-89 (never analyzed) evaluate the frozen scores against the actual loss change of the full
quantization (alpha = 1). The new windows' own curvature is measured only as a transfer diagnostic, never as a score.
Cost: one process, 12 threads, 10 windows, 6 candidates; per method wall time, peak RSS (VmHWM reset per method),
and counts of downstream forward / backward / HVP window passes; shared setup listed separately.

Stages (one new --out-dir): plan, calib (scores + costs, then freeze), fresh (new windows; needs the freeze),
evaluate, tables.
"""
import argparse
import itertools
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_qwen_block_criterion import (CAL, SEQ, TABLES, Block, Downstream, Setup, jdump, jload,  # noqa: E402
                                      load_cache)

BLOCK = HERE / "runs/qwen3_block_criterion_20261014"
CASES = [(l, k) for l in (0, 2, 27) for k in ("gate", "up")]
LAYERS3 = (0, 2, 27)
OPT = ("fp32", "int4")
HS = (1 / 8, 1 / 16, 1 / 32)
H_PRIMARY = 1 / 16
NEW = (30, 90)                       # validation windows 30..89
HVP_CHUNK = 2


def case_name(l, k):
    return f"L{l}|{k}|int4"


def vm_hwm_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 2 ** 20
    return None


@contextmanager
def meter(costs, name, counters=None):
    """Wall time and peak RSS of a block (VmHWM reset to the current RSS on entry)."""
    open("/proc/self/clear_refs", "w").write("5")
    t0 = time.time()
    c = counters if counters is not None else {}
    yield c
    costs[name] = {"seconds": time.time() - t0, "peak_rss_gb": vm_hwm_gb(), **c}


def ce_per_window(S, logits, ids):
    torch = S.torch
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                             reduction="none").view(logits.shape[0], -1).sum(1)


def downstream_logits(S, D, h):
    x = h
    for layer in S.model.model.layers[D.layer + 1:]:
        x = layer(x, **D.kw)
    return S.model.lm_head(S.model.model.norm(x))


def hvp_a2(S, D, hidden, e, ids, chunk=HVP_CHUNK):
    """Per window: a2 = 1/2 e^T H e and the first order g . e, by double backward through the downstream."""
    torch = S.torch
    a2, f1 = [], []
    for i in range(0, hidden.shape[0], chunk):
        h = hidden[i:i + chunk].clone().requires_grad_(True)
        with torch.enable_grad():
            ce = ce_per_window(S, downstream_logits(S, D, h), ids[i:i + chunk])
            g, = torch.autograd.grad(ce.sum(), h, create_graph=True)
            ec = e[i:i + chunk]
            Hv, = torch.autograd.grad((g * ec).sum(), h)
        a2.append(0.5 * (Hv.double() * ec.double()).sum((1, 2)))
        f1.append((g.detach().double() * ec.double()).sum((1, 2)))
    return torch.cat(a2).numpy(), torch.cat(f1).numpy()


# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    from qwen3_swiglu_common import environment, sha256
    jdump(out / "plan.json", {
        "cases": [case_name(l, k) for l, k in CASES], "option": list(OPT), "boundary": "Y = down_proj output, Down FP32",
        "h": list(HS), "h_primary": H_PRIMARY, "calibration_windows": [0, CAL - 1], "unused_windows": [CAL, 29],
        "new_windows": [NEW[0], NEW[1] - 1],
        "scores": ["fisher_dir", "fisher_diag", "hvp", "fd_1/8", "fd_1/16 (primary)", "fd_1/32", "bc"],
        "evaluation": {"truth": "sum over new windows of the summed next-token CE change of the full quantization",
                       "pairs": "all 15 pairs (equal M and B); the 3 Gate-vs-Up pairs", "spearman": "over the 6 cases",
                       "decisiveness": "window bootstrap share of replicates keeping each pair's order"},
        "rules": "scores frozen (sha256 in freeze.json) before stage fresh; the new windows' own curvature is a "
                 "transfer diagnostic only"})
    jdump(out / "environment.json", {**environment(["experiments/mixed_precision/run_qwen_curvature_transfer.py",
                                                    "experiments/mixed_precision/run_qwen_block_criterion.py"]),
                                     "inputs_sha256": {p.name: sha256(p) for p in (BLOCK / "cache.npz", BLOCK / "tokens.npy",
                                                                                  TABLES / "table.json")}})
    log("plan frozen")


def stage_calib(args, out, log):
    from qwen3_swiglu_common import summed_ce
    costs = {}
    with meter(costs, "shared: model load + quantizers"):
        S = Setup(out, args.threads)
    torch = S.torch
    with meter(costs, "shared: cache load"):
        cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")[:CAL]
    ids = torch.from_numpy(toks)
    # gradients at Y for the three layers: one full forward + backward per window (shared by every candidate)
    with meter(costs, "fisher: gradients (full forward + backward per window)", {"full_forward": 0, "full_backward": 0}) as c:
        S.model.model.embed_tokens.weight.requires_grad_(True)
        cap, gy = {}, {l: [] for l in LAYERS3}

        def hook(l):
            def f(m, i, o):
                o.retain_grad()
                cap[l] = o
            return f
        hs = [S.mods[f"model.layers.{l}.mlp"].register_forward_hook(hook(l)) for l in LAYERS3]
        for s in range(CAL):
            loss = summed_ce(S.wrapper(ids[s][None]), ids[s])
            loss.backward()
            c["full_forward"] += 1
            c["full_backward"] += 1
            for l in LAYERS3:
                gy[l].append(cap[l].grad[0].clone())
            S.model.zero_grad(set_to_none=True)
        for h in hs:
            h.remove()
        S.model.model.embed_tokens.weight.requires_grad_(False)
        gy = {l: torch.stack(v).double() for l, v in gy.items()}
    checks = {f"L{l}_gy_vs_cache_rel_max": float((gy[l] - torch.from_numpy(cache[f"L{l}|gy"])).abs().max()
                                                 / torch.from_numpy(cache[f"L{l}|gy"]).abs().max()) for l in LAYERS3}
    with meter(costs, "shared: block replay (e for 6 cases)"):
        blocks = {l: Block(S, cache, l, slice(0, CAL)) for l in LAYERS3}
        E = {}
        with torch.no_grad():
            for l, k in CASES:
                B = blocks[l]
                E[(l, k)] = (B.y(OPT, None) if k == "gate" else B.y(None, OPT)) - B.y_ref
    res = {c: {} for c in map(lambda x: case_name(*x), CASES)}
    with meter(costs, "fisher: scores"):
        for l, k in CASES:
            ge = gy[l][:, :SEQ - 1] * E[(l, k)].double()[:, :SEQ - 1]
            F = ge.sum((1, 2))
            r = res[case_name(l, k)]
            r["F"] = F.tolist()
            r["fisher_dir"] = float(0.5 * (F ** 2).sum())
            r["fisher_diag"] = float(0.5 * (ge ** 2).sum())
    for h in HS:
        with meter(costs, f"fd: h={h:g}", {"downstream_forward_windows": 0}) as c:
            for l in LAYERS3:
                D = Downstream(S, l, toks)
                hmid = torch.from_numpy(cache[f"L{l}|hmid"][:CAL])
                with torch.no_grad():
                    base = D.run(hmid + blocks[l].y_ref)
                    c["downstream_forward_windows"] += CAL
                    for k in ("gate", "up"):
                        e = E[(l, k)]
                        p = D.run(hmid + blocks[l].y_ref + h * e) - base
                        m = D.run(hmid + blocks[l].y_ref - h * e) - base
                        c["downstream_forward_windows"] += 2 * CAL
                        res[case_name(l, k)][f"fd_{h:g}"] = {"a2": ((p + m) / (2 * h * h)).tolist(),
                                                             "a1": ((p - m) / (2 * h)).tolist()}
    with meter(costs, "hvp", {"downstream_forward_windows": 0, "downstream_backward_windows": 0, "hvp_windows": 0}) as c:
        for l in LAYERS3:
            D = Downstream(S, l, toks)
            hmid = torch.from_numpy(cache[f"L{l}|hmid"][:CAL])
            for k in ("gate", "up"):
                a2, f1 = hvp_a2(S, D, hmid + blocks[l].y_ref, E[(l, k)], ids)
                c["downstream_forward_windows"] += CAL
                c["downstream_backward_windows"] += CAL
                c["hvp_windows"] += CAL
                res[case_name(l, k)]["hvp"] = {"a2": a2.tolist(), "F": f1.tolist()}
    tab = jload(TABLES / "table.json")
    for l, k in CASES:
        r = res[case_name(l, k)]
        r["bc"] = tab["C"][S.node[(l, k)]]["int4"]
        r["scores"] = {"fisher_dir": r["fisher_dir"], "fisher_diag": r["fisher_diag"], "hvp": float(sum(r["hvp"]["a2"])),
                       **{f"fd_{h:g}": float(sum(r[f"fd_{h:g}"]["a2"])) for h in HS}, "bc": r["bc"]}
    jdump(out / "calib.json", {"cases": res, "costs": costs, "checks": checks, "threads": args.threads,
                               "windows": CAL, "candidates": len(CASES), "hvp_chunk": HVP_CHUNK})
    from qwen3_swiglu_common import sha256
    scores = {n: r["scores"] for n, r in res.items()}
    jdump(out / "scores.json", scores)
    jdump(out / "freeze.json", {"scores_sha256": sha256(out / "scores.json"), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(json.dumps({"checks": checks, "costs": {k: round(v["seconds"], 1) for k, v in costs.items()}}))


def stage_fresh(args, out, log):
    """The new windows: FP32 tensors at the three layers, the actual loss change of each case (alpha = 1), and, as a
    transfer diagnostic only, the new windows' own FD curvature (h primary)."""
    from qwen3_swiglu_common import sha256, summed_ce
    from run_qwen_branch_ablation import tokenize_validation
    fz = jload(out / "freeze.json")
    if fz["scores_sha256"] != sha256(out / "scores.json"):
        raise SystemExit("scores changed after the freeze")
    S = Setup(out, args.threads)
    torch = S.torch
    t0 = time.time()
    toks, _ = tokenize_validation(NEW[1])
    old = np.load(BLOCK / "tokens.npy")
    if not np.array_equal(toks[:old.shape[0]], old):
        raise RuntimeError("re-tokenized windows 0-29 differ from the block study's")
    new = toks[NEW[0]:NEW[1]]
    np.save(out / "new_tokens.npy", new)
    store = {f"L{l}|{k}": [] for l in LAYERS3 for k in ("x", "zg", "zu", "y", "hmid")}
    cap = {}

    def keep(name):
        def f(m, i, o):
            cap[name] = (i[0], o)
        return f

    def keep_pre(name):
        def f(m, i):
            cap[name] = i[0]
        return f
    hs = []
    for l in LAYERS3:
        mlp = S.mods[f"model.layers.{l}.mlp"]
        hs += [mlp.register_forward_hook(keep((l, "mlp"))), mlp.gate_proj.register_forward_hook(keep((l, "gate"))),
               mlp.up_proj.register_forward_hook(keep((l, "up"))),
               S.mods[f"model.layers.{l}.post_attention_layernorm"].register_forward_pre_hook(keep_pre((l, "hmid")))]
    losses = []
    with torch.no_grad():
        for ids in new:
            t = torch.from_numpy(ids)
            losses.append(float(summed_ce(S.wrapper(t[None]), t)))
            for l in LAYERS3:
                x, y = cap[(l, "mlp")]
                for k, v in (("x", x), ("zg", cap[(l, "gate")][1]), ("zu", cap[(l, "up")][1]), ("y", y), ("hmid", cap[(l, "hmid")])):
                    store[f"L{l}|{k}"].append(v[0].numpy().copy())
    for h in hs:
        h.remove()
    cache = {k: np.stack(v) for k, v in store.items()}
    fp32_s = time.time() - t0
    res, checks = {}, {}
    n = NEW[1] - NEW[0]
    with torch.no_grad():
        for l in LAYERS3:
            B = Block(S, cache, l, slice(0, n))
            checks[f"L{l}_y_replay_rel_max"] = float((B.y_ref - torch.from_numpy(cache[f"L{l}|y"])).abs().max()
                                                    / torch.from_numpy(cache[f"L{l}|y"]).abs().max())
            D = Downstream(S, l, new)
            hmid = torch.from_numpy(cache[f"L{l}|hmid"])
            base = D.run(hmid + B.y_ref)
            checks[f"L{l}_downstream_vs_model_loss_max_abs"] = float(np.abs(base - np.array(losses)).max())
            for k in ("gate", "up"):
                e = (B.y(OPT, None) if k == "gate" else B.y(None, OPT)) - B.y_ref
                dl = D.run(hmid + B.y_ref + e) - base
                p = D.run(hmid + B.y_ref + H_PRIMARY * e) - base
                m = D.run(hmid + B.y_ref - H_PRIMARY * e) - base
                res[case_name(l, k)] = {"dL": dl.tolist(), "a2_new": ((p + m) / (2 * H_PRIMARY ** 2)).tolist(),
                                        "a1_new": ((p - m) / (2 * H_PRIMARY)).tolist()}
            log(f"layer {l}: done")
    jdump(out / "fresh.json", {"cases": res, "checks": checks, "windows": [NEW[0], NEW[1] - 1], "fp32_pass_s": fp32_s,
                               "seconds": time.time() - t0})
    log(json.dumps(checks))


def stage_evaluate(args, out, log):
    from scipy.stats import spearmanr
    cal, fr = jload(out / "calib.json"), jload(out / "fresh.json")
    names = [case_name(l, k) for l, k in CASES]
    n_new = NEW[1] - NEW[0]
    dl = np.array([fr["cases"][c]["dL"] for c in names])
    S_new = dl.sum(1)
    scores = {s: np.array([cal["cases"][c]["scores"][s] for c in names]) for s in cal["cases"][names[0]]["scores"]}
    pairs = list(itertools.combinations(range(len(names)), 2))
    gu = [(names.index(case_name(l, "gate")), names.index(case_name(l, "up"))) for l in LAYERS3]
    rng = np.random.default_rng(0)
    boot = np.array([dl[:, rng.integers(0, n_new, n_new)].sum(1) for _ in range(2000)])
    decis = {f"{names[i]} vs {names[j]}": float((np.sign(boot[:, i] - boot[:, j]) == np.sign(S_new[i] - S_new[j])).mean())
             for i, j in pairs}
    res = {"truth": {c: {"S_new": float(S_new[k]), "mean_nll_per_token": float(S_new[k] / (n_new * 127))} for k, c in enumerate(names)},
           "pair_decisiveness": decis, "scores": {}}
    for s, v in scores.items():
        right = [bool(np.sign(v[i] - v[j]) == np.sign(S_new[i] - S_new[j])) for i, j in pairs]
        gright = [bool(np.sign(v[i] - v[j]) == np.sign(S_new[i] - S_new[j])) for i, j in gu]
        res["scores"][s] = {"pairs_right": int(sum(right)), "pairs": len(pairs), "gate_up_right": gright,
                            "spearman_6": float(spearmanr(v, S_new).statistic),
                            "predicted_over_actual": {c: float(v[k] * n_new / CAL / S_new[k]) for k, c in enumerate(names)},
                            "decisive_pairs_right": int(sum(r for r, (i, j) in zip(right, pairs)
                                                            if decis[f"{names[i]} vs {names[j]}"] >= 0.95)),
                            "decisive_pairs": int(sum(decis[f"{names[i]} vs {names[j]}"] >= 0.95 for i, j in pairs))}
    # calibration reliability: FD over h, FD vs HVP, first order vs gradient
    rel = {}
    for c in names:
        r = cal["cases"][c]
        a2 = {h: np.array(r[f"fd_{h:g}"]["a2"]) for h in HS}
        hv = np.array(r["hvp"]["a2"])
        rel[c] = {"fd_sum": {f"{h:g}": float(a2[h].sum()) for h in HS}, "hvp_sum": float(hv.sum()),
                  "fd_over_hvp": {f"{h:g}": float(a2[h].sum() / hv.sum()) for h in HS},
                  "window_sign_agree_fd_hvp": {f"{h:g}": float((np.sign(a2[h]) == np.sign(hv)).mean()) for h in HS},
                  "window_corr_fd_hvp": {f"{h:g}": float(np.corrcoef(a2[h], hv)[0, 1]) for h in HS},
                  "negative_windows_hvp": int((hv < 0).sum()),
                  "a1_vs_F_rel": {f"{h:g}": float(np.abs(np.array(r[f"fd_{h:g}"]["a1"]) - np.array(r["F"])).sum()
                                                  / np.abs(np.array(r["F"])).sum()) for h in HS},
                  "hvp_F_vs_full_backward_F_rel": float(np.abs(np.array(r["hvp"]["F"]) - np.array(r["F"])).sum()
                                                       / np.abs(np.array(r["F"])).sum()),
                  "fisher_dir_over_hvp": float(r["scores"]["fisher_dir"] / hv.sum())}
    res["calibration_reliability"] = rel
    res["rank_across_h"] = {f"{a:g} vs {b:g}": float(spearmanr(scores[f"fd_{a:g}"], scores[f"fd_{b:g}"]).statistic)
                            for a, b in itertools.combinations(HS, 2)}
    res["rank_fd_vs_hvp"] = {f"{h:g}": float(spearmanr(scores[f"fd_{h:g}"], scores["hvp"]).statistic) for h in HS}
    # transfer diagnostic: calibration curvature per window vs the new windows' own
    a2new = np.array([np.mean(fr["cases"][c]["a2_new"]) for c in names])
    a2cal = {s: np.array([np.mean(cal["cases"][c][s]["a2"]) if s == "hvp" else np.mean(cal["cases"][c]["fd_0.0625"]["a2"])
                          for c in names]) for s in ("hvp", "fd")}
    loc_new = np.array([np.sum(fr["cases"][c]["a1_new"]) + np.sum(fr["cases"][c]["a2_new"]) for c in names])
    res["transfer_diagnostic"] = {
        "per_window_a2_new_over_cal_fd": {c: float(a2new[k] / a2cal["fd"][k]) for k, c in enumerate(names)},
        "spearman_cal_fd_vs_new_fd": float(spearmanr(a2cal["fd"], a2new).statistic),
        "new_local_prediction_over_actual": {c: float(loc_new[k] / S_new[k]) for k, c in enumerate(names)}}
    # costs
    costs = cal["costs"]
    res["costs"] = costs
    jdump(out / "summary.json", res)
    log("summary written")


def stage_tables(args, out, log):
    s = jload(out / "summary.json")
    cal = jload(out / "calib.json")
    names = list(s["truth"])
    L = ["### calibration 曲率可靠度（窗口 0–9）", "",
         "| 案例 | FD h=1/8 | FD h=1/16 | FD h=1/32 | HVP | FD/HVP（1/8, 1/16, 1/32） | 窗口相關 FD–HVP（1/16） | HVP 負窗口 | a1 vs 梯度 F（1/16） | Fisher／HVP |",
         "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for c in names:
        r = s["calibration_reliability"][c]
        L.append(f"| {c} | " + " | ".join(f"{r['fd_sum'][f'{h:g}']:.1f}" for h in HS) + f" | {r['hvp_sum']:.1f} | "
                 + ", ".join(f"{r['fd_over_hvp'][f'{h:g}']:.3f}" for h in HS)
                 + f" | {r['window_corr_fd_hvp']['0.0625']:.3f} | {r['negative_windows_hvp']} | {r['a1_vs_F_rel']['0.0625']:.3f} | "
                   f"{r['fisher_dir_over_hvp']:.2f} |")
    L += ["", f"FD 不同 h 之間的 6 案例 Spearman：{s['rank_across_h']}；FD 對 HVP：{s['rank_fd_vs_hvp']}", "",
          "### 新窗口（30–89，從未分析）上的實際損傷", "", "| 案例 | S_new（summed CE） | 每 token 平均 NLL 變化 |", "|---|---:|---:|"]
    for c, v in s["truth"].items():
        L.append(f"| {c} | {v['S_new']:+.1f} | {v['mean_nll_per_token']:+.4f} |")
    L += ["", "### 凍結分數對新窗口的排序", "", "| 分數 | 15 個同資源配對對 | 其中決定性配對（bootstrap ≥ 0.95）對 | Gate／Up（L0, L2, L27） | 6 案例 Spearman |",
          "|---|---:|---:|---|---:|"]
    for k, v in s["scores"].items():
        L.append(f"| {k} | {v['pairs_right']}/{v['pairs']} | {v['decisive_pairs_right']}/{v['decisive_pairs']} | "
                 + "、".join("對" if x else "錯" for x in v["gate_up_right"]) + f" | {v['spearman_6']:.3f} |")
    L += ["", "### 量級：預測（×6，換算成 60 窗口）÷ 新窗口實際", "", "| 分數 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k, v in s["scores"].items():
        L.append(f"| {k} | " + " | ".join(f"{v['predicted_over_actual'][c]:.2f}" for c in names) + " |")
    t = s["transfer_diagnostic"]
    L += ["", "### 轉移診斷（新窗口自己的曲率；不當預測）", "", "| 案例 | 新窗口 a2 ÷ calibration a2（每窗口平均，FD 1/16） | 新窗口局部模型 ÷ 實際 |", "|---|---:|---:|"]
    for c in names:
        L.append(f"| {c} | {t['per_window_a2_new_over_cal_fd'][c]:.2f} | {t['new_local_prediction_over_actual'][c]:.2f} |")
    L.append(f"\ncalibration 與新窗口 a2 的 6 案例 Spearman：{t['spearman_cal_fd_vs_new_fd']:.3f}")
    L += ["", "### 建立成本（12 threads；10 窗口；6 候選）", "", "| 項目 | 秒 | 峰值 RSS（GB） | 計數 |", "|---|---:|---:|---|"]
    for k, v in s["costs"].items():
        cnt = {a: b for a, b in v.items() if a not in ("seconds", "peak_rss_gb")}
        L.append(f"| {k} | {v['seconds']:.1f} | {v['peak_rss_gb']:.2f} | {cnt if cnt else '—'} |")
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


PRECISION_CASES = [(0, "up"), (2, "up"), (27, "up")]
PRECISION_HS = (1 / 32, 1 / 128, 1 / 512)


def stage_precision(args, out, log):
    """Which of the float32 HVP and the float32 finite differences is right where they disagree? The downstream in
    float64 (model converted after the block replay, which stays as in calib): gradient F, HVP a2, and FD a1 / a2 at
    smaller h, calibration windows. Diagnostic for the curvature reliability; not a score."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")[:CAL]
    ids = torch.from_numpy(toks)
    E, base_in = {}, {}
    with torch.no_grad():
        for l in sorted({l for l, _ in PRECISION_CASES}):
            B = Block(S, cache, l, slice(0, CAL))
            base_in[l] = (torch.from_numpy(cache[f"L{l}|hmid"][:CAL]) + B.y_ref).double()
            for ll, k in PRECISION_CASES:
                if ll == l:
                    E[(l, k)] = ((B.y(OPT, None) if k == "gate" else B.y(None, OPT)) - B.y_ref).double()
    S.model.double()
    S.wrapper.double()
    S._layer_kwargs = None
    res, t0 = {}, time.time()
    for l, k in PRECISION_CASES:
        D = Downstream(S, l, toks)
        a2h, f1 = hvp_a2(S, D, base_in[l], E[(l, k)], ids)
        with torch.no_grad():
            base = D.run(base_in[l])
            fd = {}
            for h in PRECISION_HS:
                p = D.run(base_in[l] + h * E[(l, k)]) - base
                m = D.run(base_in[l] - h * E[(l, k)]) - base
                fd[f"{h:g}"] = {"a2": float(((p + m) / (2 * h * h)).sum()), "a1": float(((p - m) / (2 * h)).sum())}
        res[case_name(l, k)] = {"hvp64_a2": float(a2h.sum()), "grad64_F": float(f1.sum()), "fd64": fd,
                                "hvp64_a2_windows": a2h.tolist()}
        log(f"{case_name(l, k)}: {json.dumps(res[case_name(l, k)]['fd64'])} hvp {a2h.sum():.1f} F {f1.sum():.2f} "
            f"({time.time() - t0:.0f}s)")
    cal = jload(out / "calib.json")["cases"]
    for c, r in res.items():
        r["float32"] = {"hvp_a2": float(sum(cal[c]["hvp"]["a2"])), "F": float(sum(cal[c]["F"])),
                        **{f"fd_{h:g}": {"a2": float(sum(cal[c][f"fd_{h:g}"]["a2"])), "a1": float(sum(cal[c][f"fd_{h:g}"]["a1"]))}
                           for h in HS}}
    jdump(out / "precision.json", {"cases": res, "seconds": time.time() - t0})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("plan", "calib", "fresh", "evaluate", "tables", "precision"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "plan":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "plan.json").exists():
        raise SystemExit("run --stage plan first")
    if args.stage == "fresh" and not (out / "freeze.json").exists():
        raise SystemExit("scores are not frozen (stage calib)")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    timing = jload(out / "timing.json") if (out / "timing.json").exists() else {}
    timing[args.stage] = time.time() - t0
    jdump(out / "timing.json", timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
