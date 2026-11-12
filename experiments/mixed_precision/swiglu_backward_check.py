#!/usr/bin/env python3
"""SwiGLU backward of the native mixed-precision statistics pass (MixedPrecisionBackward) on a small standalone
model, against PyTorch autograd (FP64), the closed-form gradients and central finite differences.

Graph (ONNX built here, the ops TorchScript exports for Qwen3's MLP):
  zg = x Wg ("gate"), zu = x Wu ("up"), h = (zg * Sigmoid(zg)) * zu, y = h Wd ("down"),
  r = y [+ x if --residual], z = r Wh ("head"), loss = sum over rows 0..L-2 of CE(z_row, target_row)
Native: --mixedp-collect-stats grads (seed (softmax - onehot) x row mask), per tap sum g^2 / sum g per channel and
per weight sum dW^2 per output channel. Gate's and up's input is one tensor (x): its tap must hold the SUM of the
two branch gradients (plus the residual path), once.
  closed form   g_zg = g_h * zu * SiLU'(zg),  SiLU'(z) = s + z s (1 - s),  s = sigmoid(z);   g_zu = g_h * SiLU(zg)
                g_x  = g_zg Wg^T + g_zu Wu^T [+ g_r]
With L = 1 the per-channel sums are the elements themselves (sign and value compared element by element). zg is
placed on a grid of negative / near-zero / positive / large values by solving x Wg = target exactly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402

ZG_GRID = [-40.0, -12.0, -4.0, -1.0, -0.1, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 0.1, 1.0, 4.0, 12.0, 40.0, 90.0]


def make_case(L, D, F, V, residual, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((L, D))
    target = np.array([ZG_GRID[(i + 3 * l) % len(ZG_GRID)] for l in range(L) for i in range(F)]).reshape(L, F)
    wg = x.T @ np.linalg.solve(x @ x.T, target)            # x Wg = target exactly (L <= D)
    wu = rng.standard_normal((D, F)) * 1.5
    wd = rng.standard_normal((F, D)) * 0.3
    wh = rng.standard_normal((D, V)) * 0.5
    labels = rng.integers(0, V, L)
    c = {k: np.asarray(v, np.float32) for k, v in dict(x=x, wg=wg, wu=wu, wd=wd, wh=wh).items()}
    c.update(labels=labels, residual=residual, L=L, D=D, F=F, V=V)
    return c


def onnx_model(c):
    nodes = [helper.make_node("MatMul", ["x", "wg"], ["zg"], name="gate"),
             helper.make_node("MatMul", ["x", "wu"], ["zu"], name="up"),
             helper.make_node("Sigmoid", ["zg"], ["sg"], name="sig"),
             helper.make_node("Mul", ["zg", "sg"], ["silu"], name="silu"),
             helper.make_node("Mul", ["silu", "zu"], ["h"], name="glu"),
             helper.make_node("MatMul", ["h", "wd"], ["y"], name="down")]
    r = "y"
    if c["residual"]:
        nodes.append(helper.make_node("Add", ["y", "x"], ["r"], name="res"))
        r = "r"
    nodes.append(helper.make_node("MatMul", [r, "wh"], ["z"], name="head"))
    L, D, V = c["L"], c["D"], c["V"]
    g = helper.make_graph(nodes, "swiglu", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, L, D])],
                          [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, L, V])],
                          [numpy_helper.from_array(c[k], k) for k in ("wg", "wu", "wd", "wh")])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)])


def forward64(c, dzg=None, dzu=None, wg=None, wu=None, x=None):
    """FP64 loss with optional additive perturbations (finite differences)."""
    x = c["x"].astype(np.float64) if x is None else x
    zg = x @ (c["wg"].astype(np.float64) if wg is None else wg)
    zu = x @ (c["wu"].astype(np.float64) if wu is None else wu)
    if dzg is not None:
        zg = zg + dzg
    if dzu is not None:
        zu = zu + dzu
    h = zg / (1 + np.exp(-zg)) * zu
    r = h @ c["wd"].astype(np.float64) + (x if c["residual"] else 0)
    z = r @ c["wh"].astype(np.float64)
    m = z.max(1, keepdims=True)
    lse = np.log(np.exp(z - m).sum(1)) + m[:, 0]
    rows = np.arange(c["L"] - 1) if c["L"] > 1 else np.arange(1)
    return float((lse - z[np.arange(c["L"]), c["labels"]])[rows].sum())


def torch64(c):
    import torch
    t = {k: torch.tensor(c[k], dtype=torch.float64, requires_grad=True) for k in ("x", "wg", "wu", "wd", "wh")}
    zg, zu = t["x"] @ t["wg"], t["x"] @ t["wu"]
    s = torch.sigmoid(zg)
    silu = zg * s
    h = silu * zu
    y = h @ t["wd"]
    r = y + t["x"] if c["residual"] else y
    z = r @ t["wh"]
    for v in (zg, zu, h, y, r):
        v.retain_grad()
    rows = slice(0, c["L"] - 1) if c["L"] > 1 else slice(0, 1)
    loss = torch.nn.functional.cross_entropy(z[rows], torch.as_tensor(c["labels"][rows]), reduction="sum")
    loss.backward()
    g = {k: v.grad.numpy() for k, v in dict(zg=zg, zu=zu, h=h, y=y, r=r).items()}
    g.update({f"d{k}": t[k].grad.numpy() for k in ("x", "wg", "wu", "wd", "wh")})
    g["loss"] = float(loss)
    # closed forms from the autograd g_h
    zg64, zu64 = zg.detach().numpy(), zu.detach().numpy()
    s64 = 1 / (1 + np.exp(-zg64))
    g["zg_closed"] = g["h"] * zu64 * (s64 + zg64 * s64 * (1 - s64))
    g["zu_closed"] = g["h"] * zg64 * s64
    g["x_gate"] = g["zg"] @ c["wg"].astype(np.float64).T
    g["x_up"] = g["zu"] @ c["wu"].astype(np.float64).T
    g["x_closed"] = g["x_gate"] + g["x_up"] + (g["r"] if c["residual"] else 0)
    g["zg"], g["zu"] = g["zg"], g["zu"]
    g["zg_value"] = zg64
    return g


def finite_differences(c, g, rng, n=6, eps=1e-6):
    """Central differences in FP64 on random elements of zg, zu, Wg, Wu, x (step eps * max(1, |element|)); the error
    is taken relative to the largest |gradient| of that array, as the native comparisons are (an element whose
    gradient is ~0, e.g. zg = -40 where SiLU' ~ 4e-17, would otherwise divide rounding noise by ~0)."""
    out = []
    for name, shape, kw in (("zg", (c["L"], c["F"]), "dzg"), ("zu", (c["L"], c["F"]), "dzu")):
        for _ in range(n):
            i = tuple(rng.integers(0, s) for s in shape)
            d = np.zeros(shape)
            h = eps * max(1.0, abs(float(g["zg_value"][i])) if name == "zg" else 1.0)
            d[i] = h
            fd = (forward64(c, **{kw: d}) - forward64(c, **{kw: -d})) / (2 * h)
            out.append({"what": f"{name}{list(map(int, i))}", "fd": fd, "autograd": float(g[name][i]),
                        "array_max": float(np.abs(g[name]).max())})
    for name, key in (("wg", "dwg"), ("wu", "dwu"), ("x", "dx")):
        base = c[name].astype(np.float64)
        for _ in range(n):
            i = tuple(rng.integers(0, s) for s in base.shape)
            p, m = base.copy(), base.copy()
            h = eps * max(1.0, abs(float(base[i])))
            p[i] += h
            m[i] -= h
            fd = (forward64(c, **{name: p}) - forward64(c, **{name: m})) / (2 * h)
            out.append({"what": f"{name}{list(map(int, i))}", "fd": fd, "autograd": float(g[key][i]),
                        "array_max": float(np.abs(g[key]).max())})
    for r in out:
        r["abs"] = abs(r["fd"] - r["autograd"])
        r["rel"] = r["abs"] / max(r["array_max"], 1e-300)
    return out


def native(c, work, tools, branch=False):
    work.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model(c), str(work / "swiglu.onnx"))
    erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / "model", work / "swiglu.onnx"])
    layout_path, inst = work / "layout.json", work / "grads.onnx.mlir"
    erm.run([tools.opt, work / "model.onnx.mlir", "--shape-inference", "--mixedp-collect-stats", "-o", inst],
            env={"MIXEDP_STATS_MODE": "grads", "MIXEDP_STATS_NODES": "gate,up,down,head",
                 "MIXEDP_STATS_LAYOUT": str(layout_path), **({"MIXEDP_STATS_INPUT_GRAD": "branch"} if branch else {})})
    so, _ = erm.build_so(tools, work, "grads", inst, None, None, opt="O0", krnl_opt="O3")
    layout = json.loads(layout_path.read_text())
    oh = np.zeros((1, c["L"], c["V"]), np.float32)
    rows = range(c["L"] - 1) if c["L"] > 1 else range(1)
    for l in rows:
        oh[0, l, c["labels"][l]] = 1.0
    np.save(work / "x.npy", c["x"][None])
    np.save(work / "oh.npy", oh)
    out = work / "stats.npy"
    erm.run([sys.executable, HERE / "run_native_bc_pipeline.py", "--_stats-child", so, "model", tools.lib,
             work / "x.npy", 0, work / "oh.npy", layout["result_index"], out])
    v = np.load(out)
    taps = {t["tap"]: t for t in layout["taps"]}
    stats = {}
    for p in layout["points"]:
        t = taps[p["tap"]]
        o, ch = t["offset"], t["channels"]
        stats[(p["node"], p["slot"])] = (v[o:o + ch], v[o + ch:o + 2 * ch])
    for w in layout["weights"]:
        stats[(w["node"], "w")] = v[w["offset"]:w["offset"] + w["channels"]]
        if "x_branch_offset" in w:
            o, ch = w["x_branch_offset"], w["x_channels"]
            stats[(w["node"], "x_branch")] = (v[o:o + ch], v[o + ch:o + 2 * ch])
    return stats, layout


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check_case(c, work, tools, rng):
    g = torch64(c)
    stats, layout = native(c, work, tools, branch=True)
    res = {"L": c["L"], "residual": c["residual"], "loss_fp64": g["loss"]}
    # shared input: one tap for gate.x and up.x
    tap = {(p["node"], p["slot"]): p["tap"] for p in layout["points"]}
    res["gate_up_share_one_tap"] = tap[("gate", "x")] == tap[("up", "x")]
    ref = {("gate", "x"): g["dx"], ("up", "x"): g["dx"], ("gate", "y"): g["zg"], ("up", "y"): g["zu"],
           ("down", "x"): g["h"], ("down", "y"): g["y"], ("head", "x"): g["r"]}
    checks = {}
    for key, gr in ref.items():
        g2, gs = stats[key]
        checks[f"{key[0]}.{key[1]}"] = {"sum_g2": rel(g2, (gr ** 2).sum(0)), "sum_g": rel(gs, gr.sum(0))}
    dW = {"gate": g["dwg"], "up": g["dwu"], "down": g["dwd"], "head": g["dwh"]}
    for n, d in dW.items():
        checks[f"{n}.w"] = {"sum_dW2": rel(stats[(n, "w")], (d ** 2).sum(0))}   # ONNX W[in, out]: out = axis 1
    res["native_vs_autograd"] = checks
    # input-edge (branch) gradients: gate / up each see only their own share; down / head have one consumer
    branch_ref = {"gate": g["x_gate"], "up": g["x_up"], "down": g["h"], "head": g["r"]}
    res["branch_native_vs_autograd"] = {
        n: {"sum_g2": rel(stats[(n, "x_branch")][0], (r ** 2).sum(0)), "sum_g": rel(stats[(n, "x_branch")][1], r.sum(0))}
        for n, r in branch_ref.items()}
    res["closed_form_vs_autograd"] = {"g_zg": rel(g["zg_closed"], g["zg"]), "g_zu": rel(g["zu_closed"], g["zu"]),
                                      "g_x": rel(g["x_closed"], g["dx"])}
    # the shared-input tap holds the branch sum once: compare with 1x and 2x the gate/up parts
    gs_tap = stats[("gate", "x")][1]
    two_branch = (g["x_gate"] + g["x_up"]).sum(0)
    res["shared_input"] = {
        "tap_vs_total_autograd": rel(gs_tap, g["dx"].sum(0)),
        "tap_vs_gate_branch_only": rel(gs_tap, g["x_gate"].sum(0)),
        "tap_vs_up_branch_only": rel(gs_tap, g["x_up"].sum(0)),
        "tap_vs_branches_doubled": rel(gs_tap, (2 * (g["x_gate"] + g["x_up"]) + (g["r"] if c["residual"] else 0)).sum(0)),
        "branch_sum_vs_total_minus_residual": rel(two_branch, (g["dx"] - (g["r"] if c["residual"] else 0)).sum(0))}
    if c["L"] == 1:          # element by element, per zg regime
        zg = g["zg_value"][0]
        ng = stats[("gate", "y")][1]
        nu = stats[("up", "y")][1]
        by = []
        for v in sorted(set(np.round(zg, 9))):
            m = np.isclose(zg, v, rtol=1e-6, atol=1e-12)
            by.append({"zg": float(v), "n": int(m.sum()),
                       "g_zg_rel": float(np.max(np.abs(ng[m] - g["zg"][0][m]) / np.maximum(np.abs(g["zg"][0][m]), 1e-30))),
                       "g_zg_abs": float(np.max(np.abs(ng[m] - g["zg"][0][m]))),
                       "g_zu_rel": float(np.max(np.abs(nu[m] - g["zu"][0][m]) / np.maximum(np.abs(g["zu"][0][m]), 1e-30))),
                       "sign_agree": bool(np.all(np.sign(ng[m]) == np.sign(g["zg"][0][m])))})
        res["by_zg_value"] = by
    res["finite_differences"] = finite_differences(c, g, rng)
    res["fd_max_rel"] = max(r["rel"] for r in res["finite_differences"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = erm.Tools()
    rng = np.random.default_rng(1)
    results = []
    for L, residual in ((1, False), (1, True), (4, False), (4, True)):
        c = make_case(L, 8, 64 if L == 1 else 16, 12, residual, seed=10 * L + residual)
        r = check_case(c, out / f"L{L}_res{int(residual)}", tools, rng)
        results.append(r)
        worst = max(v for d in r["native_vs_autograd"].values() for v in d.values())
        print(f"L={L} residual={residual}: native vs FP64 autograd worst rel {worst:.2e}; closed form "
              f"{max(r['closed_form_vs_autograd'].values()):.1e}; FD {r['fd_max_rel']:.1e}; shared tap vs total "
              f"{r['shared_input']['tap_vs_total_autograd']:.1e}, vs doubled {r['shared_input']['tap_vs_branches_doubled']:.1e}; "
              f"branch stats vs autograd {max(max(d.values()) for d in r['branch_native_vs_autograd'].values()):.1e}",
              flush=True)
    (out / "swiglu_backward.json").write_text(json.dumps(results, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
