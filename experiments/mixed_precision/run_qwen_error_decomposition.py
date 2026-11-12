#!/usr/bin/env python3
"""Qwen3-0.6B Gate/Up activation-error decomposition (diagnostic, after run_qwen_block_criterion.py): which part of
the quantization error carries the loss -- the per-channel variance term or the per-channel mean (bias) term, and
at which positions -- at the node taps (the BC criterion's activation term A) and at the block output Y.

Node taps (Gate / Up of layer L, per-consumer QDQ): x (the branch's copy of X) and y (zg / zu), d = QDQ(t) - t of the
FP32 tensor t (as the native errors statistics), gradients from the cached gY through the FP32 block:
  gh = gY Wd, g_zg = gh * zu * SiLU'(zg), g_zu = gh * SiLU(zg), g_x(gate) = g_zg Wg, g_x(up) = g_zu Wu,
  g_x(total) = g_x(gate) + g_x(up)  (X = the post-attention RMSNorm output feeds only Gate and Up)
per channel c over the calibration windows s and the positions P (n = windows * |P|):
  mu_c = sum d / n, var_c = sum d^2 / n - mu_c^2, G2_c = sum g^2, Gsum_c = sum_s (sum_t g)^2
  var term = sum_c G2_c var_c,  mean term = sum_c Gsum_c mu_c^2   (A = var + mean, the native C)
  diag = sum g^2 d^2 (elementwise; additive over positions)
P = all 128 positions (the native statistics; checked against table.json / table_branch.json) or 1..127 (no
position 0). Block output Y: e = Y_hat - Y_fp32 with the same var / mean / diag split on gY (positions 0..126; the
last position has no target), plain MSE, position-0 shares.
Intervention: the block output error of a candidate applied at position 0 only (Y_hat at t = 0, Y_fp32 elsewhere)
or at positions 1.. only, the downstream replayed (run_qwen_block_criterion.Downstream), dL per window on the
calibration and tuning windows.

Stages: decompose (no loss evaluation), intervene (downstream replays), tables, attribution (which term of the
BC activation cost orders candidate pairs wrongly). Post-hoc: the candidates and the
loss changes are those of runs/qwen3_block_criterion_20261014, whose tuning results were already seen.
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

from run_qwen_block_criterion import (CAL, LAYERS, OPTIONS, SEQ, TUNE, Block, Downstream, Setup, bc_tables,  # noqa: E402
                                      candidates, groups, jdump, jload, load_cache, load_rows, spearman)

BLOCK = HERE / "runs/qwen3_block_criterion_20261014"
ACT_FORMATS = ("int4", "int8", "p6e1", "p8e1", "bf16")
POSITIONS = {"all": slice(None), "no0": slice(1, None)}
INTERVENE = [(l, "a", k) for l in LAYERS for k in ("gate", "up")] + [(27, "w", k) for k in ("gate", "up")]


def terms(d, g, pos):
    """var / mean / diag terms of an error d against gradients g ([windows, positions, channels], float64)."""
    d, g = d[:, pos], g[:, pos]
    n = d.shape[0] * d.shape[1]
    mu = d.sum((0, 1)) / n
    var = (d ** 2).sum((0, 1)) / n - mu ** 2
    v, m = (g ** 2).sum((0, 1)) * var, (g.sum(1) ** 2).sum(0) * mu ** 2
    return {"var": float(v.sum()), "mean": float(m.sum()), "diag": float(((g * d) ** 2).sum()),
            "var_top_channel_share": float(v.max() / max(v.sum(), 1e-300)),
            "mean_top_channel_share": float(m.max() / max(m.sum(), 1e-300))}


def position0_shares(d, g):
    return {"d2_pos0_share": float((d[:, 0] ** 2).sum() / max((d ** 2).sum(), 1e-300)),
            "diag_pos0_share": float(((g[:, 0] * d[:, 0]) ** 2).sum() / max(((g * d) ** 2).sum(), 1e-300))}


def stage_decompose(args, out, log):
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    tot, br = bc_tables()
    node_terms, check = {}, []
    t0 = time.time()
    for layer in LAYERS:
        f = lambda k: torch.from_numpy(cache[f"L{layer}|{k}"][:CAL]).double()
        X, zg, zu, gy = f("x"), f("zg"), f("zu"), f("gy")
        W = {k: S.mlp(layer, k).weight.double() for k in ("gate", "up", "down")}
        with torch.no_grad():
            gh = gy @ W["down"]
            sg = torch.sigmoid(zg)
            g_z = {"gate": gh * zu * (sg + zg * sg * (1 - sg)), "up": gh * zg * sg}
            g_x = {k: g_z[k] @ W[k] for k in ("gate", "up")}
            g_x["total"] = g_x["gate"] + g_x["up"]
            Xf = torch.from_numpy(cache[f"L{layer}|x"][:CAL])
            Zf = {"gate": torch.from_numpy(cache[f"L{layer}|zg"][:CAL]), "up": torch.from_numpy(cache[f"L{layer}|zu"][:CAL])}
            for kind, a in itertools.product(("gate", "up"), ACT_FORMATS):
                dx = (S.act(Xf, layer, kind, "x", a) - Xf).double()
                dy = (S.act(Zf[kind], layer, kind, "y", a) - Zf[kind]).double()
                rec = {}
                for pname, pos in POSITIONS.items():
                    rec[pname] = {"x_total": terms(dx, g_x["total"], pos), "x_branch": terms(dx, g_x[kind], pos),
                                  "y": terms(dy, g_z[kind], pos)}
                rec["shares"] = {"x_total": position0_shares(dx, g_x["total"]), "y": position0_shares(dy, g_z[kind])}
                node_terms[f"L{layer}|{kind}|{a}"] = rec
                node = S.node[(layer, kind)]
                a_all = rec["all"]
                mine_t = sum(a_all[t]["var"] + a_all[t]["mean"] for t in ("x_total", "y"))
                mine_b = sum(a_all[t]["var"] + a_all[t]["mean"] for t in ("x_branch", "y"))
                check.append([f"L{layer}|{kind}|{a}", mine_t, tot["C"][node][a], mine_b, br["C"][node][a]])
        log(f"layer {layer}: node terms done")
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    validation = {"entries": len(check),
                  "total_rel_median": float(np.median([rel(r[1], r[2]) for r in check])),
                  "total_rel_max": float(max(rel(r[1], r[2]) for r in check)),
                  "branch_rel_median": float(np.median([rel(r[3], r[4]) for r in check])),
                  "branch_rel_max": float(max(rel(r[3], r[4]) for r in check)), "rows": check}
    log(f"A = var + mean (all positions) vs native C: total median {validation['total_rel_median']:.2e} max "
        f"{validation['total_rel_max']:.2e}; branch median {validation['branch_rel_median']:.2e} max "
        f"{validation['branch_rel_max']:.2e}")
    # block output Y: the exact joint error of every activation-carrying candidate
    y_terms = {}
    for layer in LAYERS:
        B = Block(S, cache, layer, slice(0, CAL))
        gy = torch.from_numpy(cache[f"L{layer}|gy"]).double()[:, :SEQ - 1]
        with torch.no_grad():
            for c in [c for c in candidates() if c["layer"] == layer and "a" in c["mode"]]:
                e = (B.y(c["gate"], c["up"]) - B.y_ref).double()[:, :SEQ - 1]
                rec = {p: {**terms(e, gy, pos), "mse": float((e[:, pos] ** 2).mean())} for p, pos in POSITIONS.items()}
                rec["shares"] = position0_shares(e, gy)
                y_terms[c["name"]] = rec
        log(f"layer {layer}: block output terms done")
    jdump(out / "decomposition.json", {"node_terms": node_terms, "y_terms": y_terms, "validation": validation,
                                       "seconds": time.time() - t0, "post_hoc": True})


def stage_intervene(args, out, log):
    """Block output error at position 0 only / positions 1.. only, downstream replayed (calibration + tuning)."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")
    loss = {}
    for layer in LAYERS:
        loss.update(jload(BLOCK / f"loss_L{layer}.json")["rows"])
    res, t0 = {}, time.time()
    for layer in LAYERS:
        B = Block(S, cache, layer, slice(0, CAL + TUNE))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"])
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            for c in candidates():
                if c["layer"] != layer or c["target"] == "both" or (layer, c["mode"], c["target"]) not in INTERVENE:
                    continue
                yh = B.y(c["gate"], c["up"])
                y0 = B.y_ref.clone()
                y0[:, 0] = yh[:, 0]
                yr = yh.clone()
                yr[:, 0] = B.y_ref[:, 0]
                d0, dr = D.run(hmid + y0) - base, D.run(hmid + yr) - base
                res[c["name"]] = {"pos0_only": d0.tolist(), "rest_only": dr.tolist(), "all": loss[c["name"]]}
        log(f"layer {layer}: {sum(n.startswith(f'L{layer}|') for n in res)} candidates, {D.seconds:.0f}s downstream")
    jdump(out / "intervention.json", {"rows": res, "seconds": time.time() - t0, "post_hoc": True})


def stage_tables(args, out, log):
    dec = jload(out / "decomposition.json")
    nt, yt, v = dec["node_terms"], dec["y_terms"], dec["validation"]
    tot, _ = bc_tables()
    data = [r for r in load_rows(BLOCK) if "a" in r["mode"]]
    L = [f"### 驗證：A = 變異項 + 平均偏差項（全部位置）重現 native C", "",
         f"total：median 相對差 {v['total_rel_median']:.2e}，max {v['total_rel_max']:.2e}；branch：median "
         f"{v['branch_rel_median']:.2e}，max {v['branch_rel_max']:.2e}（{v['entries']} 個 node × 格式）"]
    # node terms by layer / format (gate and up)
    L += ["", "### 節點層級：A 的組成（total 梯度，x + y tap 相加）", "",
          "| layer | node | 格式 | A（全部） | 變異項占比 | 平均偏差項占比 | 排除 position 0 後 A／原 A | 其中變異項 | 平均偏差項 | "
          "x 的 d² 在 position 0 占比 | 平均偏差項最大 channel 占比 |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer, kind, a in itertools.product(LAYERS, ("gate", "up"), ACT_FORMATS):
        r = nt[f"L{layer}|{kind}|{a}"]
        al = {k: r["all"]["x_total"][k] + r["all"]["y"][k] for k in ("var", "mean")}
        no = {k: r["no0"]["x_total"][k] + r["no0"]["y"][k] for k in ("var", "mean")}
        A, A0 = al["var"] + al["mean"], no["var"] + no["mean"]
        top = max(r["all"]["x_total"]["mean_top_channel_share"], r["all"]["y"]["mean_top_channel_share"])
        L.append(f"| {layer} | {kind} | {a} | {A:.4g} | {al['var'] / A:.3f} | {al['mean'] / A:.3f} | {A0 / A:.3f} | "
                 f"{no['var'] / max(al['var'], 1e-300):.3f} | {no['mean'] / max(al['mean'], 1e-300):.3f} | "
                 f"{r['shares']['x_total']['d2_pos0_share']:.3f} | {top:.3f} |")
    # predictors from the decomposition, within (layer, mode) groups of the activation modes
    def node_pred(r, part, pos):
        s = 0.0
        for kind in ("gate", "up"):
            o = r[kind]
            if o == "-":
                continue
            a = o.split("/")[1]
            t = nt[f"L{r['layer']}|{kind}|{a}"][pos]
            parts = ("var", "mean") if part == "A" else (part,)
            s += sum(t[tap][p] for tap in ("x_total", "y") for p in parts)
        if "w" in r["mode"]:
            s += sum(tot["B"][S_node[(r["layer"], k)]][r[k].split("/")[0]] for k in ("gate", "up") if r[k] != "-")
        return s
    from qwen3_swiglu_common import node_mapping
    S_node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    preds = {"BC_total（table）": lambda r: r["BC_total"]}
    for pos in ("all", "no0"):
        for part in ("A", "var", "mean", "diag"):
            preds[f"節點 {part}（{pos}）"] = lambda r, part=part, pos=pos: node_pred(r, part, pos)
    for pos in ("all", "no0"):
        for part in ("mse", "var", "mean", "diag"):
            preds[f"Y {part}（{pos}）"] = lambda r, part=part, pos=pos: yt[r["name"]][pos][part]
    gs = groups(data)
    L += ["", "### 組內 Spearman（activation-only 與 weight＋activation，8 組；事後診斷）", "",
          "| 預測量 | tuning S mean | calibration S mean | " + " | ".join(f"L{l} {m}" for (l, m) in sorted(gs)) + " |",
          "|---|---:|---:|" + "---:|" * len(gs)]
    summary = {}
    for name, fn in preds.items():
        vals = {k: spearman([fn(r) for r in rs], [r["S_tune"] for r in rs]) for k, rs in sorted(gs.items())}
        cal = [spearman([fn(r) for r in rs], [r["S_cal"] for r in rs]) for rs in gs.values()]
        tm = float(np.mean([x for x in vals.values() if x is not None]))
        cm = float(np.mean([x for x in cal if x is not None]))
        summary[name] = {"tuning": tm, "calibration": cm, "by_group": {f"L{l}|{m}": x for (l, m), x in vals.items()}}
        L.append(f"| {name} | {tm:.3f} | {cm:.3f} | " + " | ".join("—" if x is None else f"{x:.2f}" for x in vals.values()) + " |")
    # block output: where the error energy and weighted terms sit
    L += ["", "### block 輸出 Y：position 0 的占比（候選中位數）", "", "| layer | 模式 | e² 在 position 0 | gY²e² 在 position 0 | "
          "平均偏差項／(變異＋平均)（全部） | 同（排除 position 0） |", "|---:|---|---:|---:|---:|---:|"]
    for (l, m), rs in sorted(gs.items()):
        y = [yt[r["name"]] for r in rs]
        mf = lambda p: float(np.median([t[p]["mean"] / max(t[p]["var"] + t[p]["mean"], 1e-300) for t in y]))
        L.append(f"| {l} | {m} | {np.median([t['shares']['d2_pos0_share'] for t in y]):.3f} | "
                 f"{np.median([t['shares']['diag_pos0_share'] for t in y]):.3f} | {mf('all'):.3f} | {mf('no0'):.3f} |")
    if (out / "intervention.json").exists():
        iv = jload(out / "intervention.json")["rows"]
        L += ["", "### 介入：誤差只放在 position 0／只放在 position 1 之後（下游重播，summed CE 變化）", "",
              "| 候選 | S_cal 全部 | 只有 pos 0 | 只有 pos ≥1 | S_tune 全部 | 只有 pos 0 | 只有 pos ≥1 | tuning 交互（全部 − 兩者和） | "
              "Y 的 e² 在 pos 0 占比 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        frac = []
        for name, r in iv.items():
            a, p0, pr = (np.array(r[k]) for k in ("all", "pos0_only", "rest_only"))
            sc = lambda x: x[:CAL].sum()
            st = lambda x: x[CAL:].sum()
            share = yt[name]["shares"]["d2_pos0_share"] if name in yt else None
            L.append(f"| {name.replace('|', ' ')} | {sc(a):+.2f} | {sc(p0):+.2f} | {sc(pr):+.2f} | {st(a):+.2f} | "
                     f"{st(p0):+.2f} | {st(pr):+.2f} | {st(a) - st(p0) - st(pr):+.2f} | {'—' if share is None else f'{share:.3f}'} |")
            if abs(st(a)) > 1:
                frac.append(st(p0) / st(a))
        L.append(f"\n|S_tune| > 1 的候選：只有 position 0 的損傷／全部損傷，中位數 {np.median(frac):.3f}，"
                 f"範圍 {min(frac):+.3f}～{max(frac):+.3f}（{len(frac)} 個）")
    jdump(out / "predictors.json", summary)
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def stage_attribution(args, out, log):
    """Which term of the BC activation cost makes the wrong orders: inside each (layer, mode) group of the activation
    modes, every candidate pair whose loss change differs by more than TAU, ordered by
      BC = var + mean (+ B),  var only (+ B),  mean only (+ B),  B only (weight+activation groups),
    with var / mean the node terms over all positions (the native statistics) and B the table's weight term. A pair
    BC orders wrongly is "mean flipped" if var (+ B) alone orders it right, "var wrong" if var (+ B) is wrong too;
    a pair BC orders right is "mean rescued" if var (+ B) alone is wrong. Also the subset of pairs with equal
    weight bytes and BOPs (the decisions a resource-limited search faces). Post-hoc."""
    from qwen3_swiglu_common import node_mapping
    from run_qwen_block_criterion import TAU
    nt = jload(out / "decomposition.json")["node_terms"]
    tot, _ = bc_tables()
    node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    data = [r for r in load_rows(BLOCK) if "a" in r["mode"]]

    def parts(r, pos="all"):
        v = m = b = 0.0
        for kind in ("gate", "up"):
            if r[kind] == "-":
                continue
            w, a = r[kind].split("/")
            t = nt[f"L{r['layer']}|{kind}|{a}"][pos]
            v += t["x_total"]["var"] + t["y"]["var"]
            m += t["x_total"]["mean"] + t["y"]["mean"]
            if w != "fp32":
                b += tot["B"][node[(r["layer"], kind)]][w]
        return v, m, b
    P = {r["name"]: parts(r) for r in data}
    check = max(abs(sum(P[r["name"]]) - r["BC_total"]) / r["BC_total"] for r in data if r["BC_total"] > 0)
    res = {"max_rel_diff_var_plus_mean_plus_B_vs_BC_total": check, "tau": TAU}
    for tgt in ("S_tune", "S_cal"):
        for subset in ("all_pairs", "equal_resource"):
            agg, by = {}, {}
            for (l, mo), rs in sorted(groups(data).items()):
                c = dict.fromkeys(("pairs", "bc_right", "bc_wrong", "var_right", "mean_right", "b_right",
                                   "wrong_var_wrong", "wrong_mean_flipped", "right_mean_rescued",
                                   "wrong_var_wrong_mean_right"), 0)
                for a, b in itertools.combinations(rs, 2):
                    if subset == "equal_resource" and tuple(a["resources"]) != tuple(b["resources"]):
                        continue
                    dt = a[tgt] - b[tgt]
                    if abs(dt) <= TAU:
                        continue
                    (va, ma, ba), (vb, mb, bb) = P[a["name"]], P[b["name"]]
                    ok = lambda x: bool(np.sign(x) == np.sign(dt))
                    bc, vo, mo_, bo = ok(va + ma + ba - vb - mb - bb), ok(va + ba - vb - bb), ok(ma + ba - mb - bb), ok(ba - bb)
                    c["pairs"] += 1
                    c["bc_right"] += bc
                    c["bc_wrong"] += not bc
                    c["var_right"] += vo
                    c["mean_right"] += mo_
                    c["b_right"] += bo
                    c["wrong_var_wrong"] += (not bc) and (not vo)
                    c["wrong_mean_flipped"] += (not bc) and vo
                    c["right_mean_rescued"] += bc and not vo
                    c["wrong_var_wrong_mean_right"] += (not bc) and (not vo) and mo_
                by[f"L{l}|{mo}"] = c
                for k, v in c.items():
                    agg[k] = agg.get(k, 0) + v
            res[f"{tgt}|{subset}"] = {"total": agg, "by_group": by}
    jdump(out / "attribution.json", res)
    L = ["### 錯誤排序來自哪一項（同組配對，|ΔS| > 1；activation-only 與 weight＋activation 8 組；事後）", "",
         f"var + mean + B 與 table BC_total 的最大相對差：{check:.1e}", "",
         "| 資料 | 配對 | 配對數 | BC 對 | 只用變異項（+B）對 | 只用平均偏差項（+B）對 | BC 錯：變異項也錯 | BC 錯：平均偏差項翻錯 | "
         "BC 對：平均偏差項救回 | BC 錯且變異項錯，但平均偏差項單獨對 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tgt, lab in (("S_tune", "tuning"), ("S_cal", "calibration")):
        for subset, slab in (("all_pairs", "全部"), ("equal_resource", "相同資源")):
            t = res[f"{tgt}|{subset}"]["total"]
            L.append(f"| {lab} | {slab} | {t['pairs']} | {t['bc_right']} | {t['var_right']} | {t['mean_right']} | "
                     f"{t['wrong_var_wrong']} | {t['wrong_mean_flipped']} | {t['right_mean_rescued']} | "
                     f"{t['wrong_var_wrong_mean_right']} |")
    L += ["", "| 組（tuning，全部配對） | 配對數 | BC 錯 | 變異項也錯 | 平均偏差項翻錯 | 平均偏差項救回 |", "|---|---:|---:|---:|---:|---:|"]
    for g, c in res["S_tune|all_pairs"]["by_group"].items():
        L.append(f"| {g} | {c['pairs']} | {c['bc_wrong']} | {c['wrong_var_wrong']} | {c['wrong_mean_flipped']} | "
                 f"{c['right_mean_rescued']} |")
    text = "\n".join(L) + "\n"
    (out / "attribution.md").write_text(text)
    print(text)


OUTLIER_FACTOR = 10.0     # outlier channel: FP32 max |t| over the calibration windows > 10 x the median channel's


def stage_variance(args, out, log):
    """Where the variance term sum_c G2_c var_c goes wrong (post-hoc). Per node, format and tap (x with the total
    gradient, y), per channel: G2 over all positions or without position 0, var over all positions or without position
    0, the elementwise diag sum g^2 d^2, and outlier channels defined from the FP32 tensor alone. Candidate variants
    (summed over the quantized branches, + B in weight+activation groups) against the variance-only baseline:
    pairs ordered right, fixed and broken, inside the 8 activation groups, |dS| > TAU."""
    from qwen3_swiglu_common import node_mapping
    from run_qwen_block_criterion import TAU
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    t0 = time.time()
    V = {}
    for layer in LAYERS:
        f = lambda k: torch.from_numpy(cache[f"L{layer}|{k}"][:CAL]).double()
        zg, zu, gy = f("zg"), f("zu"), f("gy")
        W = {k: S.mlp(layer, k).weight.double() for k in ("gate", "up", "down")}
        with torch.no_grad():
            gh = gy @ W["down"]
            sg = torch.sigmoid(zg)
            g_z = {"gate": gh * zu * (sg + zg * sg * (1 - sg)), "up": gh * zg * sg}
            g_xt = g_z["gate"] @ W["gate"] + g_z["up"] @ W["up"]
            Xf = torch.from_numpy(cache[f"L{layer}|x"][:CAL])
            Zf = {"gate": torch.from_numpy(cache[f"L{layer}|zg"][:CAL]), "up": torch.from_numpy(cache[f"L{layer}|zu"][:CAL])}
            for kind, a in itertools.product(("gate", "up"), ACT_FORMATS):
                rec = {}
                for tap, T, g in (("x", Xf, g_xt), ("y", Zf[kind], g_z[kind])):
                    d = (S.act(T, layer, kind, tap, a) - T).double()
                    def stats(pos):
                        dd, gg = d[:, pos], g[:, pos]
                        n = dd.shape[0] * dd.shape[1]
                        mu = dd.sum((0, 1)) / n
                        return (gg ** 2).sum((0, 1)), (dd ** 2).sum((0, 1)) / n - mu ** 2, ((gg * dd) ** 2).sum((0, 1))
                    G2a, va, Da = stats(slice(None))
                    G2n, vn, Dn = stats(slice(1, None))
                    mag = T.abs().amax((0, 1)).double()
                    out_ch = mag > OUTLIER_FACTOR * mag.median()
                    rec[tap] = {"all": float((G2a * va).sum()), "g_no0": float((G2n * va).sum()),
                                "d_no0": float((G2a * vn).sum()), "no0": float((G2n * vn).sum()),
                                "diag": float(Da.sum()), "diag_no0": float(Dn.sum()),
                                "outlier_ch": float((G2a * va)[out_ch].sum()), "rest_ch": float((G2a * va)[~out_ch].sum()),
                                "n_outlier_ch": int(out_ch.sum()),
                                "pos0_g2_share": float(((G2a - G2n).sum() / max(G2a.sum(), 1e-300)))}
                V[f"L{layer}|{kind}|{a}"] = rec
        log(f"layer {layer}: variance parts done")
    tot, _ = bc_tables()
    node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    data = [r for r in load_rows(BLOCK) if "a" in r["mode"]]
    variants = {"變異項（基準，x+y）": lambda t: t["x"]["all"] + t["y"]["all"],
                "只有 x tap": lambda t: t["x"]["all"], "只有 y tap": lambda t: t["y"]["all"],
                "去掉 pos 0 的梯度": lambda t: t["x"]["g_no0"] + t["y"]["g_no0"],
                "去掉 pos 0 的誤差": lambda t: t["x"]["d_no0"] + t["y"]["d_no0"],
                "去掉 pos 0（梯度與誤差）": lambda t: t["x"]["no0"] + t["y"]["no0"],
                "只有 outlier channel": lambda t: t["x"]["outlier_ch"] + t["y"]["outlier_ch"],
                "去掉 outlier channel": lambda t: t["x"]["rest_ch"] + t["y"]["rest_ch"],
                "逐元素 Σg²d²（全部）": lambda t: t["x"]["diag"] + t["y"]["diag"],
                "逐元素 Σg²d²（去掉 pos 0）": lambda t: t["x"]["diag_no0"] + t["y"]["diag_no0"]}

    def value(r, fn):
        v = 0.0
        for kind in ("gate", "up"):
            if r[kind] == "-":
                continue
            w, a = r[kind].split("/")
            v += fn(V[f"L{r['layer']}|{kind}|{a}"])
            if w != "fp32":
                v += tot["B"][node[(r["layer"], kind)]][w]
        return v
    vals = {k: {r["name"]: value(r, fn) for r in data} for k, fn in variants.items()}
    base_key = next(iter(variants))
    res = {"outlier_factor": OUTLIER_FACTOR, "parts": V, "variants": {}}
    gs = groups(data)
    for tgt in ("S_tune", "S_cal"):
        for subset in ("all_pairs", "equal_resource"):
            pairs = []
            for (l, mo), rs in sorted(gs.items()):
                for a, b in itertools.combinations(rs, 2):
                    if subset == "equal_resource" and tuple(a["resources"]) != tuple(b["resources"]):
                        continue
                    dt = a[tgt] - b[tgt]
                    if abs(dt) > TAU:
                        pairs.append((f"L{l}|{mo}", a["name"], b["name"], dt))
            right = {k: [bool(np.sign(vals[k][a] - vals[k][b]) == np.sign(dt)) for _, a, b, dt in pairs] for k in variants}
            for k in variants:
                fixed = sum(r and not b0 for r, b0 in zip(right[k], right[base_key]))
                broken = sum(b0 and not r for r, b0 in zip(right[k], right[base_key]))
                by_layer = {}
                for (g, *_), r, b0 in zip(pairs, right[k], right[base_key]):
                    e = by_layer.setdefault(g, [0, 0])
                    e[0] += r and not b0
                    e[1] += b0 and not r
                res["variants"].setdefault(k, {})[f"{tgt}|{subset}"] = {
                    "pairs": len(pairs), "right": int(sum(right[k])), "fixed": int(fixed), "broken": int(broken),
                    "fixed_broken_by_group": by_layer}
            if tgt == "S_tune" and subset == "all_pairs":       # size of the baseline's wrong pairs
                wrong = [abs(dt) for (_, a, b, dt), r in zip(pairs, right[base_key]) if not r]
                allp = [abs(dt) for *_, dt in pairs]
                bins = [1, 5, 20, 100, 1e9]
                res["baseline_wrong_by_size"] = [
                    {"abs_dS": f"{lo:g}-{hi:g}", "wrong": int(sum(lo < x <= hi for x in wrong)),
                     "pairs": int(sum(lo < x <= hi for x in allp))} for lo, hi in zip(bins[:-1], bins[1:])]
    for k, fn in variants.items():
        res["variants"][k]["within_group_spearman_tuning"] = float(np.mean([
            v for v in (spearman([vals[k][r["name"]] for r in rs], [r["S_tune"] for r in rs]) for rs in gs.values())
            if v is not None]))
    jdump(out / "variance.json", res)
    L = ["### 變異項拆解：哪個部分造成錯誤排序（8 組，|ΔS| > 1；事後）", "",
         f"outlier channel：FP32 max|t| > {OUTLIER_FACTOR:g} × channel 中位數（只由張量大小決定）", "",
         "| 變體（皆 + B） | tuning 全部配對對 | 修正／弄壞 | 相同資源對 | 修正／弄壞 | calibration 全部對 | 修正／弄壞 | 組內 Spearman（tuning） |",
         "|---|---:|---|---:|---|---:|---|---:|"]
    for k in variants:
        e = res["variants"][k]
        a_, b_, c_ = e["S_tune|all_pairs"], e["S_tune|equal_resource"], e["S_cal|all_pairs"]
        L.append(f"| {k} | {a_['right']}/{a_['pairs']} | +{a_['fixed']}／−{a_['broken']} | {b_['right']}/{b_['pairs']} | "
                 f"+{b_['fixed']}／−{b_['broken']} | {c_['right']}/{c_['pairs']} | +{c_['fixed']}／−{c_['broken']} | "
                 f"{e['within_group_spearman_tuning']:.3f} |")
    L += ["", "| 基準排錯的配對，依 abs(ΔS_tune) | 錯 | 配對數 | 錯誤率 |", "|---|---:|---:|---:|"]
    for b in res["baseline_wrong_by_size"]:
        L.append(f"| {b['abs_dS']} | {b['wrong']} | {b['pairs']} | {b['wrong'] / max(b['pairs'], 1):.3f} |")
    L += ["", "| 變體 | 各組 修正／弄壞（tuning，全部配對） |", "|---|---|"]
    for k in list(variants)[1:]:
        g = res["variants"][k]["S_tune|all_pairs"]["fixed_broken_by_group"]
        L.append(f"| {k} | " + "，".join(f"{n} +{x[0]}/−{x[1]}" for n, x in g.items() if x[0] or x[1]) + " |")
    L += ["", "| node | 格式 | tap | pos 0 佔 G2 | outlier channel 數 | outlier channel 佔變異項 | 去掉 pos 0 梯度後／原 | 去掉 pos 0 誤差後／原 |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, rec in V.items():
        if key.split("|")[2] not in ("int4", "int8", "p6e1"):
            continue
        for tap in ("x", "y"):
            t = rec[tap]
            L.append(f"| {key.rsplit('|', 1)[0]} | {key.rsplit('|', 1)[1]} | {tap} | {t['pos0_g2_share']:.3f} | {t['n_outlier_ch']} | "
                     f"{t['outlier_ch'] / max(t['all'], 1e-300):.3f} | {t['g_no0'] / max(t['all'], 1e-300):.3f} | "
                     f"{t['d_no0'] / max(t['all'], 1e-300):.3f} |")
    text = "\n".join(L) + "\n"
    (out / "variance.md").write_text(text)
    print(text)
    log(f"done in {time.time() - t0:.0f}s")


SCALING = [(l, k) for l in (0, 2, 27) for k in ("gate", "up")]      # int4 activation, single branch
ALPHAS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def stage_scaling(args, out, log):
    """Is position 0's loss response to the quantization error still in the local quadratic regime? For the int4
    activation single-branch candidates of SCALING, the block output error e = Y_hat - Y_fp32 is scaled by alpha and
    applied at position 0 only or at positions 1.. only, the downstream replayed (calibration + tuning windows).
    Gradient-based predictions on the calibration windows (gY only exists there), for the same positions P:
      first order F = sum_s sum_{t in P} <gY, e>,  Fisher (per window) Q_dir = 1/2 sum_s (sum_{t in P} <gY, e>)^2,
      Fisher (elementwise) Q_diag = 1/2 sum gY^2 e^2.
    Fit dL(alpha) = a1 alpha + a2 alpha^2 on the three smallest alphas (calibration), extrapolate to alpha = 1. Post-hoc."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")
    res, t0 = {}, time.time()
    for layer in sorted({l for l, _ in SCALING}):
        B = Block(S, cache, layer, slice(0, CAL + TUNE))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"])
        gy = torch.from_numpy(cache[f"L{layer}|gy"]).double()
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            for kind in [k for l, k in SCALING if l == layer]:
                opt = ("fp32", "int4")
                yh = B.y(opt, None) if kind == "gate" else B.y(None, opt)
                e = yh - B.y_ref
                for pname, sel in (("pos0", slice(0, 1)), ("rest", slice(1, None))):
                    mask = torch.zeros_like(e)
                    mask[:, sel] = 1.0
                    curve = {}
                    for a in ALPHAS:
                        curve[str(a)] = (D.run(hmid + B.y_ref + a * mask * e) - base).tolist()
                    ec = (e * mask)[:CAL].double()[:, :SEQ - 1]
                    g = gy[:, :SEQ - 1]
                    per = (g * ec).sum((1, 2))
                    res[f"L{layer}|{kind}|{pname}"] = {
                        "curve": curve, "first_order_F": float(per.sum()),
                        "Q_dir": float(0.5 * (per ** 2).sum()), "Q_diag": float(0.5 * ((g * ec) ** 2).sum())}
        log(f"layer {layer}: done ({D.seconds:.0f}s downstream)")
    # fits on calibration
    for k, r in res.items():
        a = np.array(ALPHAS)
        y = np.array([sum(r["curve"][str(x)][:CAL]) for x in ALPHAS])
        A = np.stack([a[:3], a[:3] ** 2], 1)
        a1, a2 = np.linalg.lstsq(A, y[:3], rcond=None)[0]
        r["calibration_dL"] = y.tolist()
        r["tuning_dL"] = [sum(r["curve"][str(x)][CAL:]) for x in ALPHAS]
        r["fit_a1"], r["fit_a2"] = float(a1), float(a2)
        r["extrapolated_alpha1"] = float(a1 + a2)
        r["ratio_actual_to_extrapolated"] = float(y[-1] / (a1 + a2)) if abs(a1 + a2) > 1e-9 else None
        r["exponent_half_to_one"] = float(np.log2(y[-1] / y[-2])) if y[-1] > 0 and y[-2] > 0 else None
    jdump(out / "scaling.json", {"rows": res, "alphas": list(ALPHAS), "seconds": time.time() - t0, "post_hoc": True})
    L = ["### position 0 的響應是否仍在二次範圍（int4 激活單支；誤差乘 α 後只放在指定位置；事後）", "",
         "| 案例 | 位置 | calibration ΔL：α=1/16, 1/8, 1/4, 1/2, 1 | tuning ΔL：α=1/2, 1 | 由小 α 擬合外插到 α=1 | 實際／外插 | "
         "1/2→1 的指數 | 擬合 a1 | 梯度一階 F | 擬合 a2 | Fisher Q_dir | Fisher Q_diag |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in res.items():
        l, kind, pname = k.split("|")
        f = lambda x: "—" if x is None else f"{x:.3g}"
        L.append(f"| {l} {kind} | {pname} | " + ", ".join(f"{x:.2f}" for x in r["calibration_dL"]) + " | "
                 + f"{r['tuning_dL'][-2]:.1f}, {r['tuning_dL'][-1]:.1f} | {r['extrapolated_alpha1']:.1f} | "
                 f"{f(r['ratio_actual_to_extrapolated'])} | {f(r['exponent_half_to_one'])} | {r['fit_a1']:.3g} | "
                 f"{r['first_order_F']:.3g} | {r['fit_a2']:.3g} | {r['Q_dir']:.3g} | {r['Q_diag']:.3g} |")
    text = "\n".join(L) + "\n"
    (out / "scaling.md").write_text(text)
    print(text)


def loss_and_grad(S, D, hidden, ids):
    """Per-window summed CE of the downstream on `hidden` and its gradient w.r.t. hidden (= w.r.t. Y)."""
    torch = S.torch
    h = hidden.clone().requires_grad_(True)
    x = h
    with torch.enable_grad():
        for layer in S.model.model.layers[D.layer + 1:]:
            x = layer(x, **D.kw)
        logits = S.model.lm_head(S.model.model.norm(x))
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                               reduction="none").view(h.shape[0], -1).sum(1)
        ce.sum().backward()                 # windows are independent: d(sum)/dh_s = d(CE_s)/dh_s
    return ce.detach().double(), h.grad.double()


def stage_trapezoid(args, out, log):
    """Gradient evaluated at the quantized point (one more backward): trapezoid dL ~ 1/2 (g(0) + g(e)) . e and
    Simpson dL ~ 1/6 (g(0) + 4 g(e/2) + g(e)) . e against the actual dL and the first order g(0) . e, for the int4
    activation single-branch cases of SCALING, with the block output error at position 0 only, at positions 1.. only
    and at all positions. Gradients on the calibration windows only (as the criterion); the Gate-vs-Up order is also
    compared with the tuning loss. Post-hoc."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(BLOCK)
    toks = np.load(BLOCK / "tokens.npy")
    ids = torch.from_numpy(toks[:CAL])
    loss = {}
    for layer in LAYERS:
        loss.update(jload(BLOCK / f"loss_L{layer}.json")["rows"])
    res, checks, t0 = {}, {}, time.time()
    for layer in sorted({l for l, _ in SCALING}):
        B = Block(S, cache, layer, slice(0, CAL))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"][:CAL])
        ce0, g0 = loss_and_grad(S, D, hmid + B.y_ref, ids)
        gy = torch.from_numpy(cache[f"L{layer}|gy"]).double()
        checks[f"L{layer}"] = {"g0_vs_cached_gY_rel_max": float((g0 - gy).abs().max() / gy.abs().max())}
        for kind in [k for l, k in SCALING if l == layer]:
            opt = ("fp32", "int4")
            with torch.no_grad():
                e = (B.y(opt, None) if kind == "gate" else B.y(None, opt)) - B.y_ref
            for pname, sel in (("pos0", slice(0, 1)), ("rest", slice(1, None)), ("all", slice(None))):
                mask = torch.zeros_like(e)
                mask[:, sel] = 1.0
                em = e * mask
                ce1, g1 = loss_and_grad(S, D, hmid + B.y_ref + em, ids)
                _, gh = loss_and_grad(S, D, hmid + B.y_ref + 0.5 * em, ids)
                ed = em.double()
                dot = lambda g: float((g * ed).sum())
                res[f"L{layer}|{kind}|{pname}"] = {
                    "actual": float((ce1 - ce0).sum()), "first_order": dot(g0), "trapezoid": 0.5 * (dot(g0) + dot(g1)),
                    "simpson": (dot(g0) + 4 * dot(gh) + dot(g1)) / 6,
                    "end_gradient_term": dot(g1)}
            name = f"L{layer}|a|g=fp32/int4|u=-" if kind == "gate" else f"L{layer}|a|g=-|u=fp32/int4"
            res[f"L{layer}|{kind}|all"]["S_cal_block_study"] = float(sum(loss[name][:CAL]))
            res[f"L{layer}|{kind}|all"]["S_tune_block_study"] = float(sum(loss[name][CAL:]))
        log(f"layer {layer}: done ({time.time() - t0:.0f}s)")
    V = jload(out / "variance.json")["parts"]
    decisions = {}
    for layer in sorted({l for l, _ in SCALING}):
        g, u = res[f"L{layer}|gate|all"], res[f"L{layer}|up|all"]
        var = {k: V[f"L{layer}|{k}|int4"]["x"]["all"] + V[f"L{layer}|{k}|int4"]["y"]["all"] for k in ("gate", "up")}
        truth_cal = np.sign(g["S_cal_block_study"] - u["S_cal_block_study"])
        truth_tune = np.sign(g["S_tune_block_study"] - u["S_tune_block_study"])
        dec = {"gate_minus_up": {"S_cal": g["S_cal_block_study"] - u["S_cal_block_study"],
                                 "S_tune": g["S_tune_block_study"] - u["S_tune_block_study"]}}
        for est, (vg, vu) in {"variance_term": (var["gate"], var["up"]),
                              "first_order": (g["first_order"], u["first_order"]),
                              "trapezoid": (g["trapezoid"], u["trapezoid"]),
                              "simpson": (g["simpson"], u["simpson"])}.items():
            sg = np.sign(vg - vu)
            dec[est] = {"gate": vg, "up": vu, "right_cal": bool(sg == truth_cal), "right_tune": bool(sg == truth_tune)}
        decisions[f"L{layer}"] = dec
    jdump(out / "trapezoid.json", {"rows": res, "decisions": decisions, "checks": checks,
                                   "seconds": time.time() - t0, "post_hoc": True})
    L = ["### 在量化後的點再算一次梯度：梯形法與 Simpson（int4 激活單支，calibration；事後）", "",
         f"檢查：下游 backward 的 g(0) 與快取 gY 最大相對差 " + "，".join(f"{k} {v['g0_vs_cached_gY_rel_max']:.1e}" for k, v in checks.items()), "",
         "| 案例 | 位置 | 實際 ΔL | 一階 g(0)·e | 梯形 | Simpson | 梯形／實際 | Simpson／實際 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, r in res.items():
        l, kind, pname = k.split("|")
        rt = lambda x: f"{x / r['actual']:.2f}" if abs(r["actual"]) > 1 else "—"
        L.append(f"| {l} {kind} | {pname} | {r['actual']:+.1f} | {r['first_order']:+.1f} | {r['trapezoid']:+.1f} | "
                 f"{r['simpson']:+.1f} | {rt(r['trapezoid'])} | {rt(r['simpson'])} |")
    L += ["", "| 層 | 實際 Gate − Up（cal／tune） | 變異項 | 一階 | 梯形 | Simpson |", "|---|---|---|---|---|---|"]
    for l, d in decisions.items():
        mark = lambda x: ("對" if x["right_cal"] else "錯") + "／" + ("對" if x["right_tune"] else "錯")
        L.append(f"| {l} | {d['gate_minus_up']['S_cal']:+.0f}／{d['gate_minus_up']['S_tune']:+.0f} | "
                 + " | ".join(f"{d[e]['gate']:.0f} vs {d[e]['up']:.0f}（{mark(d[e])}）"
                              for e in ("variance_term", "first_order", "trapezoid", "simpson")) + " |")
    text = "\n".join(L) + "\n"
    (out / "trapezoid.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("decompose", "intervene", "tables", "attribution", "variance",
                                                       "scaling", "trapezoid"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "decompose" and out.exists() and (out / "decomposition.json").exists():
        raise SystemExit(f"{out} already has decomposition.json: use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    if args.stage not in ("tables", "attribution", "variance"):
        from qwen3_swiglu_common import environment
        jdump(out / f"environment_{args.stage}.json",
              environment(["experiments/mixed_precision/run_qwen_error_decomposition.py",
                           "experiments/mixed_precision/run_qwen_block_criterion.py"]))
    globals()[f"stage_{args.stage}"](args, out, log)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
