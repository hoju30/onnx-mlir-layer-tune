#!/usr/bin/env python3
"""Markdown tables for the Qwen3 SwiGLU gate/up study, read from the JSON results of qwen3_swiglu_*.py and
swiglu_backward_check.py under one run folder (default runs/qwen3_swiglu_20261013). Every number in the report
comes from here."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fmt(x, p=2):
    return "–" if x is None else f"{x:.{p}e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4)) else f"{x:.{p + 1}g}"


def grads_table(run):
    out = ["| 版本 | 比較 | median rel_max | p99 rel_max | max rel_max | median rel_l2 | min cosine |", "|---|---|---:|---:|---:|---:|---:|"]
    for tag, d in (("修正前", "phase1_grads"), ("修正後", "phase1_grads_fixed")):
        s = json.loads((run / d / "summary.json").read_text())["summary"]
        for pair in ("native_vs_fp64", "fp32_vs_fp64", "native_vs_fp32"):
            v = s[pair]
            out.append(f"| {tag} | {pair} | {fmt(v['median_rel_max'])} | {fmt(v['p99_rel_max'])} | "
                       f"{fmt(v['max_rel_max'])} | {fmt(v['median_rel_l2'])} | {v['min_cosine']:.6f} |")
    return "\n".join(out)


def per_layer_table(run):
    rows = {d: json.loads((run / d / "arrays.json").read_text()) for d in ("phase1_grads", "phase1_grads_fixed")}
    out = ["| layer | gate x/y/w 修正前 | up y/w 修正前 | gate x/y/w 修正後 | up y/w 修正後 | PyTorch FP32 (all fields) |",
           "|---:|---|---|---|---|---:|"]
    for l in range(28):
        def m(d, kind, slot):
            v = [r["native_vs_fp64"]["rel_max"] for r in rows[d] if r["layer"] == l and r["kind"] == kind and r["slot"] == slot]
            return max(v)
        f32 = max(r["fp32_vs_fp64"]["rel_max"] for r in rows["phase1_grads_fixed"] if r["layer"] == l and r["kind"] in ("gate", "up"))
        cells = []
        for d in ("phase1_grads", "phase1_grads_fixed"):
            cells.append(" / ".join(fmt(m(d, "gate", s), 1) for s in ("x", "y", "w")))
            cells.append(" / ".join(fmt(m(d, "up", s), 1) for s in ("y", "w")))
        out.append(f"| {l} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {fmt(f32, 1)} |")
    return "\n".join(out)


def forward_table(run):
    out = ["| 版本 | native median / max rel_l2 | ORT median / max | PyTorch FP32 median / max | logits native rel_l2 |",
           "|---|---|---|---|---:|"]
    for tag, d in (("修正前", "phase3_forward"), ("修正後", "phase3_forward_fixed")):
        rows = json.loads((run / d / "forward.json").read_text())
        mlp = [r for r in rows if r["tensor"] != "logits"]
        cells = []
        for name in ("native", "ort", "torch32"):
            v = np.array([r[f"{name}_vs_fp64"]["rel_l2"] for r in mlp])
            cells.append(f"{fmt(np.median(v), 1)} / {fmt(v.max(), 1)}")
        lg = max(r["native_vs_fp64"]["rel_l2"] for r in rows if r["tensor"] == "logits")
        out.append(f"| {tag} | {cells[0]} | {cells[1]} | {cells[2]} | {fmt(lg, 1)} |")
    return "\n".join(out)


def swiglu_table(run):
    res = json.loads((run / "phase2_swiglu" / "swiglu_backward.json").read_text())
    out = ["| case | native vs FP64 autograd (worst field) | closed form vs autograd | FD vs autograd | shared tap vs total | shared tap vs doubled |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in res:
        worst = max(v for d in r["native_vs_autograd"].values() for v in d.values())
        out.append(f"| L={r['L']}, residual={r['residual']} | {fmt(worst, 1)} | {fmt(max(r['closed_form_vs_autograd'].values()), 1)} | "
                   f"{fmt(r['fd_max_rel'], 1)} | {fmt(r['shared_input']['tap_vs_total_autograd'], 1)} | "
                   f"{fmt(r['shared_input']['tap_vs_branches_doubled'], 1)} |")
    return "\n".join(out)


def criterion_table(run):
    rows = json.loads((run / "phase4_criterion" / "criterion_compare.json").read_text())
    out = ["| 項目 | 數值 |", "|---|---:|"]
    out.append(f"| native 重新聚合 vs table.json：W 最大相對差 | {fmt(max(r['reagg_vs_table']['W'] for r in rows), 1)} |")
    out.append(f"| native 重新聚合 vs table.json：A 最大相對差 | {fmt(max(r['reagg_vs_table']['A'] for r in rows), 1)} |")
    out.append(f"| native 重新聚合 vs table_branch.json：A 最大相對差 | {fmt(max(r['reagg_vs_table']['A_branch'] for r in rows), 1)} |")
    for key, name in (("W_rel", "W"), ("A_rel", "A（總梯度）"), ("A_branch_rel", "A（分支梯度）"), ("C_rel", "C")):
        v = np.array([x for r in rows for x in r["torch_vs_native"][key].values() if np.isfinite(x)])
        out.append(f"| PyTorch vs native {name}：median / max 相對差（56 節點 × 10 格式） | {fmt(np.median(v), 1)} / {fmt(v.max(), 1)} |")
    for key, name in (("W_spearman", "W 格式排序 Spearman（min）"), ("A_spearman", "A 格式排序 Spearman（min）"),
                      ("C_pairs_spearman", "121 組 (w,a) 的 C 排序 Spearman（min）")):
        out.append(f"| {name} | {min(r['rank'][key] for r in rows):.6f} |")
    out.append(f"| 最小 C 的格式組合相同的節點數 | {sum(r['rank']['C_best_same'] for r in rows)}/{len(rows)} |")
    v = [r["per_channel_int4_x"]["cosine"] for r in rows]
    out.append(f"| int4 輸入項逐 channel 貢獻 cosine（min） | {min(v):.6f} |")
    out.append(f"| 權重 channel 軸（weights pass = grads） | {rows[0]['w_axis']} |")
    return "\n".join(out)


def shared_input_table(run):
    rows = json.loads((run / "phase4_criterion" / "criterion_compare.json").read_text())
    out = ["| layer | 節點 | A_x(總)/A_x(分支) int4 | int8 | p6e1 |", "|---:|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["layer"], r["kind"])):
        if r["layer"] in (0, 2, 14, 27):
            si = r["shared_input"]
            out.append(f"| {r['layer']} | {r['kind']} | " + " | ".join(
                f"{si[f]['A_x_total_over_branch_native']:.3g}" for f in ("int4", "int8", "p6e1")) + " |")
    allr = np.array([r["shared_input"][f]["A_x_total_over_branch_native"] for r in rows for f in ("int4", "int8", "p6e1")])
    out.append(f"| 全部 28 層 | gate/up | median {np.median(allr):.3g}（範圍 {allr.min():.3g}–{allr.max():.3g}） | | |")
    return "\n".join(out)


def quant_rows(run):
    return json.loads((run / "phase5_quant" / "analysis.json").read_text())


def layer_detail_table(run, layer=2):
    a = quant_rows(run)
    out = ["| 對象 | 模式 | 格式 | C（總梯度） | C（分支） | Σ_s f_s²（一階擾動平方和） | Σ_s ΔL_s（帶號） | Σ_s abs(ΔL_s) |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for e in a["rows"]:
        if e["layer"] == layer:
            out.append(f"| {e['target']} | {e['mode']} | {e['fmt']} | {fmt(e['C'])} | {fmt(e['C_branch'])} | "
                       f"{fmt(e['f2_sum'])} | {fmt(e['dL_sum'])} | {fmt(e['dL_abs_sum'])} |")
    return "\n".join(out)


def proxy_table(run):
    a = quant_rows(run)["rows"]
    out = ["| 項目 | 設定 | 比值 median | 範圍 | n |", "|---|---|---:|---|---:|"]

    def add(name, sel, num, den):
        r = np.array([e[num] / e[den] for e in sel if e[den] > 0 and e[num] > 0])
        if len(r):
            out.append(f"| {name} | | {r.size and np.median(r):.3g} | {r.min():.3g}–{r.max():.3g} | {len(r)} |")
    single = [e for e in a if e["target"] != "both"]
    both = [e for e in a if e["target"] == "both"]
    for grp, sel in (("單獨 (gate 或 up)", single), ("聯合 (gate+up)", both)):
        out.append(f"| **{grp}** | | | | |")
        add("W / Σ(f_W)²", [e for e in sel if "w" in e["mode"]], "W", "fW2")
        add("A_x(總梯度) / Σ(f_X)²", [e for e in sel if "a" in e["mode"]], "A_x", "fX2")
        add("A_x(分支梯度) / Σ(f_X)²", [e for e in sel if "a" in e["mode"]], "A_x_branch", "fX2")
        add("A_y / Σ(f_Y)²", [e for e in sel if "a" in e["mode"]], "A_y", "fY2")
        add("C(總梯度) / Σf²", sel, "C", "f2_sum")
        add("C(分支) / Σf²", sel, "C_branch", "f2_sum")
    return "\n".join(out)


def joint_table(run):
    a = quant_rows(run)["rows"]
    out = ["| layer | 模式 | 格式 | ΔL 聯合 | ΔL gate + ΔL up | 差 | Σf² 聯合 | Σf² gate + up | 2Σ f_g f_u | SwiGLU 二階項 | C 聯合（分支）| C gate+up（分支） |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in a:
        if e["target"] == "both":
            out.append(f"| {e['layer']} | {e['mode']} | {e['fmt']} | {fmt(e['dL_sum'])} | {fmt(e['dL_gate_plus_up'])} | "
                       f"{fmt(e['dL_joint_minus_sum'])} | {fmt(e['f2_sum'])} | {fmt(e['f2_gate_plus_up'])} | "
                       f"{fmt(e['cross_2fgfu'])} | {fmt(e['swiglu_second_order'])} | {fmt(e['C_branch'])} | "
                       f"{fmt(e['C_branch_gate_plus_up'])} |")
    return "\n".join(out)


def corr_table(run):
    c = quant_rows(run)["correlations"]
    out = ["| 模式 | criterion | Spearman(C, Σf²) | Spearman(C, abs ΔL) | Spearman(C, ΔL 帶號) | log-log Pearson(C, abs ΔL) | n |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for mode in ("w", "a", "wa", "all"):
        for ck in ("C", "C_branch"):
            f2, dl = c[f"{mode}|{ck}~f2_sum"], c[f"{mode}|{ck}~dL_sum"]
            out.append(f"| {mode} | {'總梯度' if ck == 'C' else '分支'} | {f2['spearman']:.3f} | {dl['spearman_vs_abs']:.3f} | "
                       f"{dl['spearman']:.3f} | {fmt(dl['pearson_loglog_abs'])} | {dl['n']} |")
    ceil = c["ceiling|f2_sum~dL_abs_sum"]
    out.append(f"| all | 上限：Σf² 本身 | — | {ceil['spearman']:.3f} | — | — | {ceil['n']} |")
    return "\n".join(out)


def decisions_table(run):
    d = quant_rows(run)["decisions"]
    out = ["| 決策 | 依據 | 正確 / 總數 |", "|---|---|---:|"]
    for key, label in (("C|w", "C（w 模式，兩種歸屬相同）"), ("C|a+wa", "C 總梯度（a、wa 模式）"),
                       ("C_branch|a+wa", "C 分支（a、wa 模式）"), ("f2_sum|w", "模擬一階 Σf²（w）"),
                       ("f2_sum|a+wa", "模擬一階 Σf²（a、wa）")):
        r = d[f"gate_vs_up|{key}"]
        out.append(f"| gate 與 up 哪支傷害較大 | {label} | {r['agree']} / {r['n']} |")
    r = d["sign_nonadditivity_eq_sign_cross"]
    out.append(f"| 聯合 ΔL 比單獨和大或小（abs 差 > 1） | 一階交叉項 2Σ f_g f_u 的正負號 | {r['agree']} / {r['n']} |")
    out.append(f"| 聯合 ΔL 低於某一單獨 ΔL（單獨 ΔL > 1） | 加總型 criterion 恆預測聯合 ≥ 單獨 | "
               f"{len(d['joint_below_a_single'])} 組違反 |")
    j = d["joint_input_part"]
    out += ["", "| 聯合量化時輸入項（同一 dX）| median |", "|---|---:|",
            f"| Σ(f_X,gate+f_X,up)² / [Σf_X,gate² + Σf_X,up²] | {j['fX2_joint_over_additive_median']:.2f} |",
            f"| A_x(總梯度，算一次) / Σ(f_X 聯合)² | {j['A_x_total_once_over_fX2_joint_median']:.2f} |",
            f"| A_x(分支 gate + up) / Σ(f_X 聯合)² | {j['A_x_branch_sum_over_fX2_joint_median']:.2f} |",
            f"| A_x(總梯度，算兩次；現行) / Σ(f_X 聯合)² | {j['A_x_total_twice_over_fX2_joint_median']:.2f} |"]
    return "\n".join(out)


def native_table(run):
    nat = json.loads((run / "phase5_native" / "native.json").read_text())
    sim = {e["name"]: e for e in json.loads((run / "phase5_quant" / "sim.json").read_text())}
    base = np.array(next(r for r in nat if r["name"] == "fp32")["losses"])
    out = ["| 設定 | native ΔL（onnx-mlir 編譯執行） | PyTorch 模擬 ΔL | 差 | 相對差 | 每窗口最大 abs 差 |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in nat:
        if r["name"] == "fp32":
            continue
        dn = np.array(r["losses"]) - base
        ds = np.array(sim[r["name"]]["dL"])
        d, s = float(dn.sum()), float(ds.sum())
        out.append(f"| {r['name']} | {fmt(d)} | {fmt(s)} | {fmt(d - s)} | {fmt(abs(d - s) / max(abs(s), 1e-12), 1)} | "
                   f"{fmt(float(np.abs(dn - ds).max()))} |")
    out.append(f"\nFP32 基準（native）：10 窗口 summed CE 合計 {base.sum():.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=HERE / "runs/qwen3_swiglu_20261013")
    args = ap.parse_args()
    for name, fn in (("gradient summary", grads_table), ("per layer", per_layer_table),
                     ("forward", forward_table), ("swiglu", swiglu_table),
                     ("criterion", criterion_table), ("shared input", shared_input_table),
                     ("layer 2 detail", layer_detail_table), ("proxy", proxy_table), ("joint", joint_table),
                     ("correlations", corr_table), ("decisions", decisions_table),
                     ("native validation", native_table)):
        try:
            print(f"### {name}\n\n{fn(args.run)}\n")
        except FileNotFoundError as e:
            print(f"### {name}: missing {e.filename}\n")


if __name__ == "__main__":
    main()
