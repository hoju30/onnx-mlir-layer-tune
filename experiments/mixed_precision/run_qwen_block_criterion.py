#!/usr/bin/env python3
"""Qwen3-0.6B: a joint Gate/Up block criterion observed at the FFN output Y (down_proj's output), against the per-node
(additive) criterion, as predictors of the whole-model loss change.

Block (layer L, Down kept FP32):  zg = X Wg, zu = X Wu, h = SiLU(zg) * zu, Y = h Wd
Gate / Up use the mixed-precision pass's per-consumer QDQ: a branch with activation format a quantizes its own copy
of X and its own projection output (zg / zu, before SiLU), int formats with the calibration (scale, zp) of
runs/native_eps_fix_20261014/qwen3/ranges_by_point.json, weights per output channel as the pass does. A candidate
(Gate option, Up option) is replayed on cached FP32 tensors: Y_hat = (SiLU(zg_hat) * zu_hat) Wd, e = Y_hat - Y_fp32.

Scores phi(e), all on the calibration windows s = 0..9 and positions t = 0..126 (position 127 has no target and,
causal, affects no loss), channels j = 0..1023, accumulated in float64:
  mse  mean over (s, t, j) of e^2                                    (plain output MSE)
  gw   sum over (s, t, j) of gY^2 e^2      gY = d(summed CE of window s)/dY   (elementwise sensitivity weighted)
  dir  sum over s of (sum over (t, j) of gY e)^2                       (per-window first order, squared; with Down
                                                                         FP32 and linear this equals the same score at h)
Structures under one phi: additive phi(eg) + phi(eu), linear joint phi(eg + eu), exact joint phi(egu), with eg / eu
the Gate-only / Up-only output errors and egu the joint one (a single-branch candidate has all three equal).
External baselines: the per-node BC of table.json (total input gradient) and table_branch.json (branch), summed.

Loss impact (PyTorch, FP32 model with only this block's Y replaced): the residual stream before the MLP (h_mid) is
cached, layers L+1.. / final norm / LM head are replayed on h_mid + Y, dL_s = CE_s(Y_hat) - CE_s(Y_fp32) per window
(summed next-token CE). Over a window set: signed S = sum_s dL_s, |S| = abs(sum_s dL_s), A = sum_s abs(dL_s).
Calibration windows 0-9 build the cache, ranges and score tables; tuning windows 10-29 (validation split, not used
for anything else) validate; the test split is not read.

Stages (one --out-dir):
  cache     FP32 pass with hooks (calibration: X, zg, zu, h, Y, gY, h_mid; tuning: X, zg, zu, Y, h_mid), FP32 replay
            checks (block replay vs the model's Y, downstream replay vs the model's window losses), environment
  plan      the candidate set, scores, targets and comparisons, frozen (plan.json) before any loss evaluation
  table     block replay scores for every candidate (calibration only)
  loss      downstream replay dL per window, calibration and tuning, for every candidate (PyTorch simulation)
  analyze   the planned comparisons
  native    representative candidates compiled by onnx-mlir and run on the same 30 windows (native measurement)
  explore   post-hoc diagnostics (noise ceiling, position-0 outlier, signed first order), not part of the plan
  tables    the report's Markdown tables
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

TABLES = HERE / "runs/native_eps_fix_20261014/qwen3"
ABLATION = HERE / "runs/qwen3_branch_ablation_20261014"       # eval_tokens.npy (windows 0-29), native FP32 losses
LAYERS = (0, 2, 14, 27)
CAL, TUNE, SEQ = 10, 20, 128
EXTRA_FILES = ["experiments/mixed_precision/run_qwen_block_criterion.py",
               "experiments/mixed_precision/qwen3_swiglu_common.py", "src/Dialect/ONNX/Transforms/Recompose.cpp"]


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n")


def jload(path):
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Model, quantizers, cached tensors
# --------------------------------------------------------------------------- #
class Setup:
    """The FP32 model (no gradients unless asked), the pass-equivalent quantizers and the int (scale, zp) of every
    Gate / Up tensor from the calibration ranges."""

    def __init__(self, out, threads):
        import torch
        from qwen3_swiglu_common import Quantizers, load_model, node_mapping
        from run_native_bc_pipeline import erm
        torch.set_num_threads(threads)
        self.torch, self.erm = torch, erm
        self.q = Quantizers(out, formats=("p6e1", "p8e1"))
        self.model, self.wrapper = load_model(torch.float32)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mods = dict(self.model.named_modules())
        mapping = node_mapping()
        self.node = {(l, k): n for n, (l, k, _) in mapping.items()}
        self.ranges = jload(TABLES / "ranges_by_point.json")

    def mlp(self, layer, kind):
        return self.mods[f"model.layers.{layer}.mlp.{kind}_proj"]

    def act(self, x, layer, kind, slot, fmt):
        if fmt == "fp32":
            return x
        return self.q.act(x, fmt, self.ranges[self.node[(layer, kind)]][slot])

    def int_param_check(self):
        """(scale, zp) from the ranges (what the compile environment uses) against the errors layouts' records."""
        from types import SimpleNamespace
        from qwen3_swiglu_criterion import int_params
        rec = int_params(SimpleNamespace(native_run=TABLES))
        bad = n = 0
        for layer, kind, slot, fmt in itertools.product(LAYERS, ("gate", "up"), ("x", "y"), ("int4", "int8")):
            node = self.node[(layer, kind)]
            lo, hi = self.ranges[node][slot]
            mine = self.erm.int_qparams(float(lo), float(hi), int(fmt[3:]))
            theirs = rec[(node, slot)][fmt]
            n += 1
            bad += not (np.isclose(float(mine[0]), float(theirs[0]), rtol=0, atol=0) and int(mine[1]) == int(theirs[1]))
        return {"checked": n, "mismatches": bad}


def load_cache(out):
    z = np.load(out / "cache.npz")
    return {k: z[k] for k in z.files}


def stage_cache(args, out, log):
    from qwen3_swiglu_common import environment, sha256, summed_ce
    S = Setup(out, args.threads)
    torch = S.torch
    toks = np.load(ABLATION / "eval_tokens.npy")
    val = np.load(TABLES / "calib_inputs.npy").reshape(CAL, SEQ)
    if toks.shape != (CAL + TUNE, SEQ) or not np.array_equal(toks[:CAL], val):
        raise RuntimeError("eval_tokens.npy is not calibration windows 0-9 + tuning windows 10-29")
    np.save(out / "tokens.npy", toks)
    S.model.model.embed_tokens.weight.requires_grad_(True)   # a graph to differentiate through
    store = {f"L{l}|{k}": [] for l in LAYERS for k in ("x", "zg", "zu", "h", "y", "gy", "hmid")}
    losses, captured = [], {}

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = (inputs[0], output)
            if name[1] == "mlp" and output.requires_grad:
                output.retain_grad()
        return hook

    def keep_pre(name):
        def hook(module, inputs):
            captured[name] = inputs[0]
        return hook
    hooks = []
    for l in LAYERS:
        mlp = S.mods[f"model.layers.{l}.mlp"]
        hooks += [mlp.register_forward_hook(keep((l, "mlp"))),
                  mlp.gate_proj.register_forward_hook(keep((l, "gate"))),
                  mlp.up_proj.register_forward_hook(keep((l, "up"))),
                  mlp.down_proj.register_forward_hook(keep((l, "down"))),
                  S.mods[f"model.layers.{l}.post_attention_layernorm"].register_forward_pre_hook(keep_pre((l, "hmid")))]
    t0 = time.time()
    for s, ids in enumerate(toks):
        calib = s < CAL
        with torch.set_grad_enabled(calib):
            loss = summed_ce(S.wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))
            if calib:
                loss.backward()
        losses.append(float(loss.detach()))
        for l in LAYERS:
            x, y = captured[(l, "mlp")]
            xg, zg = captured[(l, "gate")]
            xu, zu = captured[(l, "up")]
            h, yd = captured[(l, "down")]
            if not (torch.equal(x, xg) and torch.equal(x, xu) and torch.equal(y, yd)):
                raise RuntimeError("block boundary: Gate / Up input or Down output is not the MLP's")
            for k, v in (("x", x), ("zg", zg), ("zu", zu), ("h", h), ("y", y), ("hmid", captured[(l, "hmid")])):
                store[f"L{l}|{k}"].append(v[0].detach().numpy().copy())
            if calib:
                store[f"L{l}|gy"].append(y.grad[0].numpy().copy())
        S.model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    pass_s = time.time() - t0
    np.savez(out / "cache.npz", **{k: np.stack(v) for k, v in store.items() if v})
    cache = load_cache(out)
    # FP32 replay checks: the block from the cached X with the model's weights, and the downstream from h_mid + Y
    checks = {}
    with torch.no_grad():
        for l in LAYERS:
            X = torch.from_numpy(cache[f"L{l}|x"])
            W = {k: S.mlp(l, k).weight for k in ("gate", "up", "down")}
            zg, zu = X @ W["gate"].T, X @ W["up"].T
            h = torch.nn.functional.silu(zg) * zu
            y = h @ W["down"].T
            ref = torch.from_numpy(cache[f"L{l}|y"])
            hs = torch.from_numpy(cache[f"L{l}|zg"])
            yrep = (torch.nn.functional.silu(hs) * torch.from_numpy(cache[f"L{l}|zu"])) @ W["down"].T
            checks[f"L{l}"] = {
                "zg_rel_max": rel_max(zg, torch.from_numpy(cache[f"L{l}|zg"])),
                "y_from_x_rel_max": rel_max(y, ref), "y_from_cached_zg_zu_rel_max": rel_max(yrep, ref),
                "h_rel_max": rel_max(torch.nn.functional.silu(hs) * torch.from_numpy(cache[f"L{l}|zu"]),
                                     torch.from_numpy(cache[f"L{l}|h"])),
                "gy_last_position_zero": bool(np.all(cache[f"L{l}|gy"][:, -1] == 0)),
                "y_abs_max": float(np.abs(cache[f"L{l}|y"]).max()),
                "y_position0_share_of_sq": float((cache[f"L{l}|y"][:, 0] ** 2).sum() / (cache[f"L{l}|y"] ** 2).sum())}
            D = Downstream(S, l, toks)
            ce = D.run(torch.from_numpy(cache[f"L{l}|hmid"] + cache[f"L{l}|y"]))
            checks[f"L{l}"]["downstream_vs_model_loss_max_abs"] = float(np.abs(ce - np.array(losses)).max())
            log(f"layer {l}: {json.dumps(checks[f'L{l}'])} (downstream {D.seconds:.1f}s for 30 windows)")
            checks[f"L{l}"]["downstream_s_30_windows"] = D.seconds
    checks["int_params_vs_errors_layouts"] = S.int_param_check()
    jdump(out / "baseline.json", {"losses": losses, "windows": {"calibration": [0, CAL - 1],
                                                                "tuning": [CAL, CAL + TUNE - 1]},
                                  "checks": checks, "fp32_pass_s": pass_s})
    jdump(out / "environment.json", {**environment(EXTRA_FILES), "cache_sha256": sha256(out / "cache.npz"),
                                     "tokens_sha256": sha256(out / "tokens.npy"),
                                     "table_sha256": sha256(TABLES / "table.json"),
                                     "table_branch_sha256": sha256(TABLES / "table_branch.json"),
                                     "ranges_sha256": sha256(TABLES / "ranges_by_point.json"),
                                     "threads": args.threads, "test_split_read": False})
    log(f"cache written ({pass_s:.0f}s FP32 pass)")


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


class Downstream:
    """Layers L+1.., the final norm and the LM head on a [windows, 128, 1024] residual stream after layer L's MLP;
    per-window summed next-token CE. The decoder layers get the arguments the model itself passes them (captured
    from one FP32 forward), batched over windows (mask and RoPE tables broadcast)."""

    def __init__(self, S, layer, toks, chunk=10):
        self.S, self.layer, self.chunk = S, layer, chunk
        self.ids = S.torch.from_numpy(toks)
        self.kw = S.__dict__.setdefault("_layer_kwargs", None) or self._capture(toks)
        S._layer_kwargs = self.kw
        self.seconds = 0.0

    def _capture(self, toks):
        torch, kw = self.S.torch, {}

        def hook(module, args, kwargs):
            kw.update({k: v for k, v in kwargs.items()})
        h = self.S.model.model.layers[1].register_forward_pre_hook(hook, with_kwargs=True)
        with torch.no_grad():
            self.S.wrapper(torch.from_numpy(toks[:1]))
        h.remove()
        return kw

    def run(self, hidden):
        torch = self.S.torch
        t0 = time.time()
        out = []
        with torch.no_grad():
            for i in range(0, hidden.shape[0], self.chunk):
                h = hidden[i:i + self.chunk]
                for layer in self.S.model.model.layers[self.layer + 1:]:
                    h = layer(h, **self.kw)
                logits = self.S.model.lm_head(self.S.model.model.norm(h))
                ids = self.ids[i:i + h.shape[0]]
                ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                       ids[:, 1:].reshape(-1), reduction="none")
                out.append(ce.view(h.shape[0], -1).double().sum(1).numpy())
        self.seconds += time.time() - t0
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Candidates, block replay, scores
# --------------------------------------------------------------------------- #
# per-node options (weight format, activation format) by mode; int6 / p8e1 are the knee's Gate / Up formats
OPTIONS = {"w": [("int4", "fp32"), ("int6", "fp32"), ("int8", "fp32"), ("p6e1", "fp32")],
           "a": [("fp32", "int4"), ("fp32", "int8"), ("fp32", "p6e1"), ("fp32", "p8e1"), ("fp32", "bf16")],
           "wa": [("int4", "int4"), ("int6", "p8e1"), ("int8", "int8"), ("p6e1", "p6e1")]}
SCORES, STRUCTURES = ("mse", "gw", "dir"), ("additive", "linear", "exact")
TAU = 1.0                      # summed-CE difference below which a pairwise decision is a tie (as the earlier study)


def opt_name(o):
    return "-" if o is None else f"{o[0]}/{o[1]}"


def candidates():
    out = []
    for layer, mode in itertools.product(LAYERS, OPTIONS):
        opts = OPTIONS[mode]
        for g, u in [(o, None) for o in opts] + [(None, o) for o in opts] + list(itertools.product(opts, opts)):
            target = "both" if g and u else "gate" if g else "up"
            out.append({"name": f"L{layer}|{mode}|g={opt_name(g)}|u={opt_name(u)}", "layer": layer, "mode": mode,
                        "target": target, "gate": g, "up": u})
    return out


def resources(c):
    """(weight bits summed over Gate / Up, sum of bits(w) * bits(a)): Gate and Up have the same shape, so equal
    tuples mean equal weight bytes and BOPs (an unquantized branch counts as fp32 / fp32)."""
    from htq_pareto_search import FORMAT_BITS
    opts = [c["gate"] or ("fp32", "fp32"), c["up"] or ("fp32", "fp32")]
    return (sum(FORMAT_BITS[w] for w, _ in opts), sum(FORMAT_BITS[w] * FORMAT_BITS[a] for w, a in opts))


class Block:
    """One layer's block on cached FP32 tensors: branch outputs per option, Y_hat per (Gate, Up) option pair."""

    def __init__(self, S, cache, layer, windows):
        torch = S.torch
        self.S, self.layer, self.sl = S, layer, windows
        self.x = torch.from_numpy(cache[f"L{layer}|x"][windows])
        self.z = {k: torch.from_numpy(cache[f"L{layer}|z{k[0]}"][windows]) for k in ("gate", "up")}
        self.wd = S.mlp(layer, "down").weight
        self.wq, self.branch_cache = {}, {}
        with torch.no_grad():
            self.y_ref = self.y(None, None)

    def branch(self, kind, opt):
        if opt is None:
            return self.z[kind]
        key = (kind, opt)
        if key not in self.branch_cache:
            w, a = opt
            if (kind, w) not in self.wq:
                self.wq[(kind, w)] = self.S.q.weight(self.S.mlp(self.layer, kind).weight, w)
            x = self.S.act(self.x, self.layer, kind, "x", a)
            self.branch_cache[key] = self.S.act(x @ self.wq[(kind, w)].T, self.layer, kind, "y", a)
        return self.branch_cache[key]

    def y(self, g, u):
        F = self.S.torch.nn.functional
        return (F.silu(self.branch("gate", g)) * self.branch("up", u)) @ self.wd.T


def phi(e, gy):
    """Scores of an output error e [windows, 128, 1024] (float64) on positions 0..126."""
    e, g = e[:, :SEQ - 1], gy[:, :SEQ - 1]
    ge = g * e
    return {"mse": float((e ** 2).mean()), "gw": float((ge ** 2).sum()), "dir": float((ge.sum((1, 2)) ** 2).sum())}


def bc_tables():
    t, b = jload(TABLES / "table.json"), jload(TABLES / "table_branch.json")
    return t, b


def bc_cost(table, node, opt):
    return 0.0 if opt is None else table["B"][node][opt[0]] + table["C"][node][opt[1]]


def stage_plan(args, out, log):
    """The candidates, scores, targets and comparisons, written before any loss is evaluated."""
    from qwen3_swiglu_common import sha256
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists: the plan is frozen once")
    cands = candidates()
    plan = {
        "candidates": len(cands), "options": {m: [opt_name(o) for o in v] for m, v in OPTIONS.items()},
        "layers": list(LAYERS), "modes": list(OPTIONS),
        "scores": {"mse": "mean over calibration windows 0-9, positions 0-126, channels of e^2",
                   "gw": "sum over the same of gY^2 e^2 (gY: gradient of the window's summed CE w.r.t. Y)",
                   "dir": "sum over windows of (sum over positions 0-126, channels of gY e)^2"},
        "structures": {"additive": "phi(eg) + phi(eu)", "linear": "phi(eg + eu)", "exact": "phi(egu)"},
        "baselines": {"BC_total": "table.json B[w] + C[a] summed over the quantized branches",
                      "BC_branch": "table_branch.json, the same"},
        "targets": {"S": "sum_s dL_s (signed)", "absS": "abs(sum_s dL_s)", "A": "sum_s abs(dL_s)",
                    "dL_s": "summed next-token CE of window s with this block's Y replaced, minus with Y_fp32"},
        "windows": {"calibration": [0, CAL - 1], "tuning": [CAL, CAL + TUNE - 1], "test": "not used"},
        "primary": {"target": "tuning S", "statistic": "Spearman inside each (layer, mode) group over all its "
                    "candidates (single and joint), mean and median over the 12 groups",
                    "question": "does the exact joint score rank better than the additive score of the same phi, "
                                "and than BC_total / BC_branch"},
        "secondary": ["the same on calibration S; on tuning A and |S|",
                      "joint candidates only, inside each group",
                      f"pairs inside a group with equal weight bytes and BOPs, |delta S| > {TAU}: share ordered right",
                      "of those, the Gate/Up swaps (a, b) vs (b, a)",
                      "non-additivity N = S_joint - S_gate_only - S_up_only against exact - additive (and linear - "
                      f"additive): Spearman, and sign agreement where |N| > {TAU}",
                      "pooled Spearman over all candidates (reported, not used for the conclusion)"],
        "tie_threshold_summed_ce": TAU,
    }
    jdump(out / "plan.json", plan)
    log(f"plan frozen: {len(cands)} candidates, sha256 {sha256(out / 'plan.json')[:16]}")


def stage_table(args, out, log):
    """Scores of every candidate from the block replay on the calibration windows (no loss evaluated)."""
    from qwen3_swiglu_common import sha256
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(out)
    tot, br = bc_tables()
    rows, t0 = [], time.time()
    timing = {}
    for layer in LAYERS:
        t1 = time.time()
        B = Block(S, cache, layer, slice(0, CAL))
        gy = torch.from_numpy(cache[f"L{layer}|gy"]).double()
        single = {}
        with torch.no_grad():
            for c in [c for c in candidates() if c["layer"] == layer]:
                key_g, key_u = ("gate", c["gate"]), ("up", c["up"])
                for kind, opt in (key_g, key_u):
                    if opt is not None and (kind, opt) not in single:
                        yk = B.y(opt, None) if kind == "gate" else B.y(None, opt)
                        single[(kind, opt)] = (yk - B.y_ref).double()
                zero = torch.zeros_like(B.y_ref, dtype=torch.float64)
                eg = single[key_g] if c["gate"] is not None else zero
                eu = single[key_u] if c["up"] is not None else zero
                egu = (B.y(c["gate"], c["up"]) - B.y_ref).double() if c["target"] == "both" else eg + eu
                pg, pu = phi(eg, gy), phi(eu, gy)
                sc = {"additive": {k: pg[k] + pu[k] for k in SCORES}, "linear": phi(eg + eu, gy),
                      "exact": phi(egu, gy)}
                ng, nu = S.node[(layer, "gate")], S.node[(layer, "up")]
                e64 = egu[:, :SEQ - 1]
                rows.append({**c, "gate": opt_name(c["gate"]), "up": opt_name(c["up"]), "resources": resources(c),
                             "scores": sc,
                             "BC_total": bc_cost(tot, ng, c["gate"]) + bc_cost(tot, nu, c["up"]),
                             "BC_branch": bc_cost(br, ng, c["gate"]) + bc_cost(br, nu, c["up"]),
                             "mse_position0_share": float((e64[:, 0] ** 2).sum() / max((e64 ** 2).sum(), 1e-300))})
        timing[f"L{layer}"] = time.time() - t1
        log(f"layer {layer}: {sum(r['layer'] == layer for r in rows)} candidates scored in {timing[f'L{layer}']:.1f}s")
    jdump(out / "scores.json", {"rows": rows, "seconds": timing, "total_s": time.time() - t0,
                                "plan_sha256": sha256(out / "plan.json")})


def stage_loss(args, out, log):
    """Whole-model loss change per window (PyTorch, FP32 model with this block's Y replaced), calibration and
    tuning, for every planned candidate. Resumes per layer file."""
    from qwen3_swiglu_common import sha256
    if not (out / "plan.json").exists():
        raise SystemExit("freeze the plan first (stage plan)")
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(out)
    toks = np.load(out / "tokens.npy")
    for layer in [int(x) for x in args.layers.split(",")]:
        f = out / f"loss_L{layer}.json"
        done = jload(f) if f.exists() else {"rows": {}, "seconds": 0.0}
        t1 = time.time()
        B = Block(S, cache, layer, slice(0, CAL + TUNE))
        D = Downstream(S, layer, toks)
        hmid = torch.from_numpy(cache[f"L{layer}|hmid"])
        with torch.no_grad():
            base = D.run(hmid + B.y_ref)
            done["fp32_replay_ce"] = base.tolist()
            for c in [c for c in candidates() if c["layer"] == layer]:
                if c["name"] in done["rows"]:
                    continue
                ce = D.run(hmid + B.y(c["gate"], c["up"]))
                done["rows"][c["name"]] = (ce - base).tolist()
                if len(done["rows"]) % 10 == 0:
                    jdump(f, done)
        done["seconds"] += time.time() - t1
        done["downstream_s"] = D.seconds
        done["plan_sha256"] = sha256(out / "plan.json")
        jdump(f, done)
        log(f"layer {layer}: {len(done['rows'])} candidates, {time.time() - t1:.0f}s")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


PREDICTORS = [f"{p}.{s}" for p in SCORES for s in STRUCTURES] + ["BC_total", "BC_branch"]


def predictor(r, key):
    if key.startswith("BC_"):
        return r[key]
    p, s = key.split(".")
    return r["scores"][s][p]


def load_rows(out):
    rows = jload(out / "scores.json")["rows"]
    dl = {}
    for layer in LAYERS:
        f = out / f"loss_L{layer}.json"
        if f.exists():
            dl.update(jload(f)["rows"])
    data = []
    for r in rows:
        if r["name"] not in dl:
            continue
        d = np.array(dl[r["name"]])
        r = dict(r, dL=d.tolist())
        for name, sl in (("cal", slice(0, CAL)), ("tune", slice(CAL, CAL + TUNE))):
            r[f"S_{name}"], r[f"absS_{name}"], r[f"A_{name}"] = (float(d[sl].sum()), float(abs(d[sl].sum())),
                                                                  float(np.abs(d[sl]).sum()))
        data.append(r)
    return data


def groups(data, joint_only=False):
    g = {}
    for r in data:
        if joint_only and r["target"] != "both":
            continue
        g.setdefault((r["layer"], r["mode"]), []).append(r)
    return g


def within(data, key, target, joint_only=False):
    vals = {f"L{l}|{m}": spearman([predictor(r, key) for r in rs], [r[target] for r in rs])
            for (l, m), rs in sorted(groups(data, joint_only).items())}
    v = [x for x in vals.values() if x is not None]
    return {"mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None,
            "groups": len(v), "by_group": vals}


def pairs(data, key, target, swaps_only=False):
    right = wrong = ties = 0
    for rs in groups(data).values():
        for a, b in itertools.combinations(rs, 2):
            if tuple(a["resources"]) != tuple(b["resources"]):
                continue
            if swaps_only and not (a["target"] == b["target"] == "both" and a["gate"] == b["up"]
                                   and a["up"] == b["gate"] and a["gate"] != a["up"]):
                continue
            dt = a[target] - b[target]
            if abs(dt) <= TAU:
                ties += 1
                continue
            dp = predictor(a, key) - predictor(b, key)
            right += np.sign(dp) == np.sign(dt)
            wrong += np.sign(dp) != np.sign(dt)
    return {"right": int(right), "wrong": int(wrong), "ties_below_tau": int(ties),
            "accuracy": float(right / (right + wrong)) if right + wrong else None}


def nonadditivity(data, phi_name, target):
    by = {(r["layer"], r["mode"], r["target"], r["gate"], r["up"]): r for r in data}
    N, I, X = [], [], []
    for r in data:
        if r["target"] != "both":
            continue
        g = by.get((r["layer"], r["mode"], "gate", r["gate"], "-"))
        u = by.get((r["layer"], r["mode"], "up", "-", r["up"]))
        if g is None or u is None:
            continue
        N.append(r[target] - g[target] - u[target])
        sc = r["scores"]
        I.append(sc["exact"][phi_name] - sc["additive"][phi_name])
        X.append(sc["linear"][phi_name] - sc["additive"][phi_name])
    N, I, X = map(np.array, (N, I, X))
    big = np.abs(N) > TAU
    return {"cases": int(len(N)), "positive_N": int((N > TAU).sum()), "negative_N": int((N < -TAU).sum()),
            "spearman_exact_minus_additive": spearman(I, N), "spearman_linear_minus_additive": spearman(X, N),
            "sign_agree_exact_minus_additive": int((np.sign(I[big]) == np.sign(N[big])).sum()),
            "sign_agree_linear_minus_additive": int((np.sign(X[big]) == np.sign(N[big])).sum()),
            "sign_cases": int(big.sum())}


def bootstrap_primary(data, reps=2000, seed=0):
    """Tuning windows resampled with replacement: the mean within-group Spearman of each predictor and the
    exact - additive difference per phi (added after the plan, before the tuning results were looked at)."""
    rng = np.random.default_rng(seed)
    d = np.array([r["dL"][CAL:] for r in data])
    gl = [[i for i, r in enumerate(data) if (r["layer"], r["mode"]) == k] for k in sorted(groups(data))]
    P = {k: np.array([predictor(r, k) for r in data]) for k in PREDICTORS}
    from scipy.stats import rankdata
    PR = {k: [rankdata(P[k][ix]) for ix in gl] for k in PREDICTORS}
    stats = {k: [] for k in PREDICTORS}
    for _ in range(reps):
        S = d[:, rng.integers(0, TUNE, TUNE)].sum(1)
        for k in PREDICTORS:
            v = []
            for ix, pr in zip(gl, PR[k]):
                t = rankdata(S[ix])
                if np.ptp(pr) and np.ptp(t):
                    v.append(np.corrcoef(pr, t)[0, 1])
            stats[k].append(np.mean(v))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {k: {"ci95": ci(v)} for k, v in stats.items()}
    for p in SCORES:
        for other in (f"{p}.additive", "BC_total", "BC_branch"):
            diff = np.array(stats[f"{p}.exact"]) - np.array(stats[other])
            out[f"{p}.exact-{other}"] = {"mean": float(diff.mean()), "ci95": ci(diff),
                                         "share_positive": float((diff > 0).mean())}
    return out


def phase5_crosscheck(data):
    """Same-format candidates against the earlier whole-model simulation (forward hooks, batch 1) on the
    calibration windows (runs/qwen3_swiglu_20261013/phase5_quant/sim.json)."""
    sim = {r["name"]: r for r in jload(HERE / "runs/qwen3_swiglu_20261013/phase5_quant/sim.json")}
    out = []
    for r in data:
        g = r["gate"] if r["gate"] != "-" else None
        u = r["up"] if r["up"] != "-" else None
        o = g or u
        w, a = o.split("/")
        fmt = w if r["mode"] == "w" else a if r["mode"] == "a" else (w if w == a else None)
        if fmt is None or fmt not in ("int4", "int8", "p6e1") or (g and u and g != u):
            continue
        name = f"L{r['layer']}_{r['target']}_{r['mode']}_{fmt}"
        if name in sim:
            out.append([r["name"], r["S_cal"], sim[name]["dL_sum"]])
    d = np.array([[a, b] for _, a, b in out]).reshape(-1, 2)
    return {"matched": len(out), "max_abs_diff": float(np.abs(d[:, 0] - d[:, 1]).max(initial=0.0)),
            "max_rel_diff_abs_S_gt_1": float(max((abs(a - b) / abs(b) for _, a, b in out if abs(b) > 1), default=0)),
            "rows": out}


def stage_analyze(args, out, log):
    from qwen3_swiglu_common import sha256
    data = load_rows(out)
    res = {"candidates_with_loss": len(data), "plan_sha256": sha256(out / "plan.json")}
    for tgt in ("S_tune", "S_cal", "A_tune", "absS_tune", "A_cal"):
        res[f"within_group|{tgt}"] = {k: within(data, k, tgt) for k in PREDICTORS}
        res[f"within_group_joint|{tgt}"] = {k: within(data, k, tgt, joint_only=True) for k in PREDICTORS}
        res[f"pooled|{tgt}"] = {k: spearman([predictor(r, k) for r in data], [r[tgt] for r in data])
                                for k in PREDICTORS}
    for tgt in ("S_tune", "S_cal"):
        res[f"equal_resource_pairs|{tgt}"] = {k: pairs(data, k, tgt) for k in PREDICTORS}
        res[f"swap_pairs|{tgt}"] = {k: pairs(data, k, tgt, swaps_only=True) for k in PREDICTORS}
        res[f"nonadditivity|{tgt}"] = {p: nonadditivity(data, p, tgt) for p in SCORES}
    # exact vs additive per group (primary, tuning S)
    cmp = {}
    for p in SCORES:
        for other in (f"{p}.additive", "BC_total", "BC_branch"):
            a = res["within_group|S_tune"][f"{p}.exact"]["by_group"]
            b = res["within_group|S_tune"][other]["by_group"]
            d = [a[g] - b[g] for g in a if a[g] is not None and b[g] is not None]
            cmp[f"{p}.exact vs {other}"] = {"mean_diff": float(np.mean(d)), "groups_better": int(sum(x > 0 for x in d)),
                                             "groups_worse": int(sum(x < 0 for x in d)), "groups": len(d)}
    res["primary_group_comparison"] = cmp
    res["bootstrap_tuning"] = bootstrap_primary(data)
    res["phase5_crosscheck"] = phase5_crosscheck(data)
    # the interaction diagnostic I_output = exact - additive, by mode and sign
    io = {}
    for m in OPTIONS:
        js = [r for r in data if r["mode"] == m and r["target"] == "both"]
        for p in SCORES if js else ():
            v = np.array([r["scores"]["exact"][p] - r["scores"]["additive"][p] for r in js])
            rel = v / np.array([r["scores"]["additive"][p] for r in js])
            io[f"{m}|{p}"] = {"positive": int((v > 0).sum()), "negative": int((v < 0).sum()),
                              "median_relative": float(np.median(rel)), "min_relative": float(rel.min()),
                              "max_relative": float(rel.max())}
    res["I_output"] = io
    jdump(out / "analysis.json", res)
    import csv
    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["name", "layer", "mode", "target", "gate", "up", "bits_w", "bits_wa", "BC_total", "BC_branch",
                *[f"{p}.{s}" for s in STRUCTURES for p in SCORES], "mse_position0_share",
                "S_cal", "absS_cal", "A_cal", "S_tune", "absS_tune", "A_tune"]
        w.writerow(head)
        for r in data:
            w.writerow([r["name"], r["layer"], r["mode"], r["target"], r["gate"], r["up"], *r["resources"],
                        r["BC_total"], r["BC_branch"], *[r["scores"][s][p] for s in STRUCTURES for p in SCORES],
                        r["mse_position0_share"], r["S_cal"], r["absS_cal"], r["A_cal"], r["S_tune"],
                        r["absS_tune"], r["A_tune"]])
    with open(out / "per_window.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "window", "set", "dL"])
        for r in data:
            for i, v in enumerate(r["dL"]):
                w.writerow([r["name"], i, "calibration" if i < CAL else "tuning", v])
    log(json.dumps({k: {kk: vv["mean"] for kk, vv in res[f"within_group|{k}"].items()} for k in ("S_tune", "S_cal")},
                   indent=1))


# --------------------------------------------------------------------------- #
# Native measurement of representative candidates
# --------------------------------------------------------------------------- #
NATIVE = ["L2|a|g=fp32/int4|u=fp32/int8", "L2|a|g=fp32/int8|u=fp32/int4",
          "L27|a|g=fp32/p6e1|u=fp32/int8", "L14|wa|g=int8/int8|u=p6e1/p6e1"]


def stage_native(args, out, log):
    """onnx-mlir builds of NATIVE (Gate / Up of one layer quantized, everything else FP32) as
    run_qwen_branch_ablation.py evaluates (per-channel int weights, Krnl -O3 / clang -O0, constants to file, the
    calibration ranges), run on the same 30 windows; FP32 reference: that study's native FP32 result (same compiler
    binary and windows, checked)."""
    from concurrent.futures import ThreadPoolExecutor
    import onnx
    from evaluate_native_search_chunked import chunked_run_so
    from qwen3_swiglu_common import PREP, environment, sha256
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm, imported_model
    tools = erm.Tools()
    fp = jload(ABLATION / "eval/fp32/result.json")
    if fp["environment"]["onnx_mlir_opt_sha256"] != sha256(tools.opt) or \
            fp["environment"]["eval_tokens_sha256"] != sha256(ABLATION / "eval_tokens.npy") or \
            not np.array_equal(np.load(ABLATION / "eval_tokens.npy"), np.load(out / "tokens.npy")):
        raise RuntimeError("the native FP32 reference was built by another compiler or on other windows")
    graph = onnx.load(str(PREP / "model.onnx"), load_external_data=False)
    mapping = jload(PREP / "node_mapping.json")
    nodes = [n for n in erm.supported_nodes(graph) if n["name"] in mapping]
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in jload(TABLES / "ranges_by_point.json").items():
        for slot, r in slots.items():
            n = by_name[node]
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    from qwen3_swiglu_common import node_mapping
    node = {(l, k): n for n, (l, k, _) in node_mapping().items()}
    cands = {c["name"]: c for c in candidates()}
    toks = np.load(out / "tokens.npy")
    inputs = out / "native_inputs.npy"
    np.save(inputs, toks)
    base = imported_model(TABLES)
    posit_obj = erm.build_posit_runtime(tools, out)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    jdump(out / "environment_native.json", environment(EXTRA_FILES))

    def one(name):
        c = cands[name]
        target = out / "native" / name.replace("|", "_").replace("/", "-").replace("=", "")
        result = target / "result.json"
        if result.exists():
            return jload(result)
        target.mkdir(parents=True, exist_ok=True)
        cfg = {n["name"]: {"w": None, "a": "fp32"} for n in nodes}
        for kind in ("gate", "up"):
            o = c[kind]
            if o is not None:
                cfg[node[(c["layer"], kind)]] = {"w": None if o[0] == "fp32" else o[0], "a": o[1]}
        env = erm.node_formats_env(cfg, nodes, ranges)
        (target / "compile_env.json").write_text(json.dumps({"MIXEDP_NODE_FORMATS": env,
                                                             "MIXEDP_WEIGHT_PER_CHANNEL": "1"}, indent=1) + "\n")
        t = time.time()
        so, info = erm.build_so(tools, target, "m", base, env, posit_obj, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True)
        build_s = time.time() - t
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, ev = chunked_run_so(args.chunks)(tools, so, inputs, target / "logits.npy", 36000)
        ce = [float(cross_entropy(z[s, :-1], toks[s, 1:]).sum()) for s in range(len(toks))]
        finite = bool(np.isfinite(z).all())
        (target / "logits.npy").unlink()
        for p in (so, target / "model.constants.bin"):
            p.unlink(missing_ok=True)
        r = {"name": name, "summed_ce": ce, "dL": (np.array(ce) - np.array(fp["summed_ce"])).tolist(),
             "all_logits_finite": finite, "warnings": info["warnings"], "build_s": build_s,
             "eval_wall_s": ev["wall_seconds"], "onnx_mlir_opt_sha256": sha256(tools.opt)}
        jdump(result, r)
        log(f"native {name}: S_cal {sum(r['dL'][:CAL]):+.4f}, S_tune {sum(r['dL'][CAL:]):+.4f} "
            f"(build {build_s:.0f}s, run {ev['wall_seconds']:.0f}s)")
        return r
    with ThreadPoolExecutor(args.parallel) as pool:
        res = list(pool.map(one, NATIVE))
    jdump(out / "native.json", {"fp32_reference": str(ABLATION / "eval/fp32/result.json"), "results": res})


def stage_explore(args, out, log):
    """Post-hoc diagnostics of the planned result (not part of the plan; explain, do not select):
    the noise ceiling (calibration S against tuning S of the same candidates, inside groups), the position-0 outlier
    (MSE without position 0), the signed first order F = sum_s <gY, e> (calibration windows), per-group values."""
    S = Setup(out, args.threads)
    torch = S.torch
    cache = load_cache(out)
    data = load_rows(out)
    by = {r["name"]: r for r in data}
    extra = {}
    with torch.no_grad():
        for layer in LAYERS:
            B = Block(S, cache, layer, slice(0, CAL))
            gy = torch.from_numpy(cache[f"L{layer}|gy"]).double()
            g = gy[:, :SEQ - 1]
            nop = lambda e: {"mse_no_pos0": float((e[:, 1:] ** 2).mean()),
                             "gw_no_pos0": float(((g[:, 1:] * e[:, 1:]) ** 2).sum())}
            single = {}
            for c in [c for c in candidates() if c["layer"] == layer]:
                e = (B.y(c["gate"], c["up"]) - B.y_ref).double()[:, :SEQ - 1]
                if c["target"] != "both":
                    single[(c["target"], c["gate"] or c["up"])] = e
                eg = single[("gate", c["gate"])] if c["gate"] is not None else 0 * e
                eu = single[("up", c["up"])] if c["up"] is not None else 0 * e
                ex, ad_g, ad_u = nop(e), nop(eg), nop(eu)
                extra[c["name"]] = {"first_order_cal": float((g * e).sum()),
                                    **{f"{k}.exact": v for k, v in ex.items()},
                                    **{f"{k}.additive": ad_g[k] + ad_u[k] for k in ex}}
    res = {"note": "post-hoc, not in plan.json"}
    keys = {**{f"{k}.{st}": (lambda r, k=k, st=st: extra[r["name"]][f"{k}.{st}"])
               for k in ("mse_no_pos0", "gw_no_pos0") for st in ("additive", "exact")},
            "first_order_cal (signed)": lambda r: extra[r["name"]]["first_order_cal"],
            "S_cal (noise ceiling)": lambda r: r["S_cal"],
            "mse.exact": lambda r: r["scores"]["exact"]["mse"], "gw.exact": lambda r: r["scores"]["exact"]["gw"],
            "BC_total": lambda r: r["BC_total"]}
    for k, fn in keys.items():
        per = {f"L{l}|{m}": spearman([fn(r) for r in rs], [r["S_tune"] for r in rs])
               for (l, m), rs in sorted(groups(data).items())}
        v = [x for x in per.values() if x is not None]
        res[k] = {"mean": float(np.mean(v)), "median": float(np.median(v)), "by_group": per}
    for k in ("mse_no_pos0", "gw_no_pos0"):
        d = [res[f"{k}.exact"]["by_group"][g] - res[f"{k}.additive"]["by_group"][g] for g in res[f"{k}.exact"]["by_group"]]
        res[f"{k}: exact - additive"] = {"mean": float(np.mean(d)), "groups_better": int(sum(x > 0 for x in d)),
                                         "groups_worse": int(sum(x < 0 for x in d))}
        for st in ("additive", "exact"):
            for tgt in ("S_tune", "S_cal"):
                right = wrong = 0
                for rs in groups(data).values():
                    for a, b in itertools.combinations(rs, 2):
                        if tuple(a["resources"]) != tuple(b["resources"]) or abs(a[tgt] - b[tgt]) <= TAU:
                            continue
                        ok = np.sign(extra[a["name"]][f"{k}.{st}"] - extra[b["name"]][f"{k}.{st}"]) == np.sign(a[tgt] - b[tgt])
                        right += ok
                        wrong += not ok
                res[f"equal_resource_pairs|{k}.{st}|{tgt}"] = f"{int(right)}/{int(right + wrong)}"
    for tgt in ("S_cal", "S_tune"):
        right = wrong = 0
        for rs in groups(data).values():
            for a, b in itertools.combinations(rs, 2):
                if tuple(a["resources"]) != tuple(b["resources"]) or abs(a["S_tune"] - b["S_tune"]) <= TAU:
                    continue
                if tgt == "S_cal":
                    ok = np.sign(a["S_cal"] - b["S_cal"]) == np.sign(a["S_tune"] - b["S_tune"])
                    right += ok
                    wrong += not ok
        if tgt == "S_cal":
            res["equal_resource_pairs|S_cal predicts S_tune (noise ceiling)"] = f"{int(right)}/{int(right + wrong)}"
    # first order against signed calibration S (same windows): does the signed first order carry the sign?
    fo = np.array([extra[r["name"]]["first_order_cal"] for r in data])
    sc = np.array([r["S_cal"] for r in data])
    res["first_order_vs_S_cal"] = {"spearman": spearman(fo, sc),
                                   "median_abs_ratio_first_order_to_S": float(np.median(np.abs(fo) / np.maximum(np.abs(sc), 1e-9)))}
    # position-0 share of the exact output MSE, by layer
    res["mse_position0_share_median_by_layer"] = {
        f"L{l}": float(np.median([r["mse_position0_share"] for r in data if r["layer"] == l])) for l in LAYERS}
    jdump(out / "explore.json", res)
    log(json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in res.items()}, indent=1))


def stage_tables(args, out, log):
    """The report's Markdown tables (tables.md) from baseline / scores / analysis / native."""
    a, b = jload(out / "analysis.json"), jload(out / "baseline.json")
    data = {r["name"]: r for r in load_rows(out)}
    f = lambda x, p=3: "—" if x is None else f"{x:.{p}g}"
    L = ["### FP32 replay checks", "", "| layer | zg rel_max | Y（由 X）rel_max | Y（由快取 zg/zu）rel_max | 下游 vs 模型 loss 最大差 | "
         "Y 能量在 position 0 的比例 | 下游 30 窗口秒數 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for l in LAYERS:
        c = b["checks"][f"L{l}"]
        L.append(f"| {l} | {c['zg_rel_max']:.2e} | {c['y_from_x_rel_max']:.2e} | {c['y_from_cached_zg_zu_rel_max']:.2e} | "
                 f"{c['downstream_vs_model_loss_max_abs']:.2e} | {c['y_position0_share_of_sq']:.4f} | "
                 f"{c['downstream_s_30_windows']:.1f} |")
    x = a["phase5_crosscheck"]
    L += ["", f"int (scale, zp) 與 errors layouts：{b['checks']['int_params_vs_errors_layouts']}；"
          f"與第 5 階段完整模型模擬（calibration S）：{x['matched']} 組，最大絕對差 {x['max_abs_diff']:.2e}，"
          f"|S| > 1 時最大相對差 {x['max_rel_diff_abs_S_gt_1']:.2e}"]

    def wg_table(title, key):
        rows = ["", f"### {title}", "", "| 預測量 | tuning S mean | median | calibration S mean | median | tuning A mean | "
                "tuning abs(S) mean |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k in PREDICTORS:
            t, c = a[f"{key}|S_tune"][k], a[f"{key}|S_cal"][k]
            rows.append(f"| {k} | {f(t['mean'])} | {f(t['median'])} | {f(c['mean'])} | {f(c['median'])} | "
                        f"{f(a[f'{key}|A_tune'][k]['mean'])} | {f(a[f'{key}|absS_tune'][k]['mean'])} |")
        return rows
    L += wg_table("within-group Spearman（12 組 = 4 層 × 3 模式；單獨 + 聯合候選）", "within_group")
    L += ["", "### 每組比較（tuning S）", "", "| 比較 | 平均差 | 較好組數 | 較差組數 | bootstrap 95% CI（平均 Spearman 差） | 為正的比例 |",
          "|---|---:|---:|---:|---|---:|"]
    bs = a["bootstrap_tuning"]
    for k, v in a["primary_group_comparison"].items():
        bk = k.replace(" vs ", "-")
        bb = bs.get(bk)
        L.append(f"| {k} | {v['mean_diff']:+.3f} | {v['groups_better']} | {v['groups_worse']} | "
                 f"{'[%+.3f, %+.3f]' % tuple(bb['ci95']) if bb else '—'} | {f(bb['share_positive']) if bb else '—'} |")
    L += wg_table("joint-only within-group Spearman（只有 Gate＋Up 候選）", "within_group_joint")
    L += ["", "### 相同資源的成對決策（|ΔS| > 1；正確／錯誤，tie 數）", "",
          "| 預測量 | 全部相同資源配對 tuning | calibration | Gate／Up 互換 tuning | calibration |", "|---|---|---|---|---|"]
    for k in PREDICTORS:
        e = lambda d: f"{d['right']}/{d['right'] + d['wrong']}（{f(d['accuracy'], 2)}；tie {d['ties_below_tau']}）"
        L.append(f"| {k} | {e(a['equal_resource_pairs|S_tune'][k])} | {e(a['equal_resource_pairs|S_cal'][k])} | "
                 f"{e(a['swap_pairs|S_tune'][k])} | {e(a['swap_pairs|S_cal'][k])} |")
    L += ["", "### 非加成性 N = S_joint − S_gate − S_up", "",
          "| phi | 資料 | 聯合候選 | N > 1 | N < −1 | Spearman(exact−additive, N) | Spearman(linear−additive, N) | "
          "正負號一致 exact−additive | linear−additive |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for tgt, lab in (("S_tune", "tuning"), ("S_cal", "calibration")):
        for p_ in SCORES:
            v = a[f"nonadditivity|{tgt}"][p_]
            L.append(f"| {p_} | {lab} | {v['cases']} | {v['positive_N']} | {v['negative_N']} | "
                     f"{f(v['spearman_exact_minus_additive'])} | {f(v['spearman_linear_minus_additive'])} | "
                     f"{v['sign_agree_exact_minus_additive']}/{v['sign_cases']} | "
                     f"{v['sign_agree_linear_minus_additive']}/{v['sign_cases']} |")
    L += ["", "### I_output = C_exact − C_additive（聯合候選）", "", "| 模式 | phi | > 0 | < 0 | 相對 additive 的中位數 | 最小 | 最大 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for k, v in a["I_output"].items():
        m, p_ = k.split("|")
        L.append(f"| {m} | {p_} | {v['positive']} | {v['negative']} | {v['median_relative']:+.3f} | "
                 f"{v['min_relative']:+.3f} | {v['max_relative']:+.3f} |")
    L += ["", "### pooled Spearman（全部候選；只供參考，主要反映格式強弱）", "", "| 預測量 | tuning S | calibration S |", "|---|---:|---:|"]
    for k in PREDICTORS:
        L.append(f"| {k} | {f(a['pooled|S_tune'][k])} | {f(a['pooled|S_cal'][k])} |")
    # mixed datatypes
    L += ["", "### 不同 datatype 的聯合候選（activation-only 與 weight＋activation）", "",
          "| 候選 | S_cal | S_tune | A_tune | N_tune | mse exact／additive | gw exact／additive | BC_total |",
          "|---|---:|---:|---:|---:|---|---|---:|"]
    by = {(r["layer"], r["mode"], r["target"], r["gate"], r["up"]): r for r in data.values()}
    picks = [("a", "fp32/int4", "fp32/int8"), ("a", "fp32/int8", "fp32/int4"), ("a", "fp32/p6e1", "fp32/int8"),
             ("a", "fp32/int8", "fp32/p6e1"), ("wa", "int4/int4", "int8/int8"), ("wa", "int8/int8", "int4/int4"),
             ("wa", "p6e1/p6e1", "int8/int8"), ("wa", "int8/int8", "p6e1/p6e1")]
    for l in LAYERS:
        for m, g, u in picks:
            r = by.get((l, m, "both", g, u))
            if r is None:
                continue
            gs, us = by.get((l, m, "gate", g, "-")), by.get((l, m, "up", "-", u))
            n = r["S_tune"] - gs["S_tune"] - us["S_tune"] if gs and us else None
            sc = r["scores"]
            L.append(f"| {r['name'].replace('|', ' ')} | {r['S_cal']:+.2f} | {r['S_tune']:+.2f} | {r['A_tune']:.2f} | {f(n)} | "
                     f"{sc['exact']['mse']:.3g}／{sc['additive']['mse']:.3g} | {sc['exact']['gw']:.3g}／{sc['additive']['gw']:.3g} | "
                     f"{r['BC_total']:.3g} |")
    if (out / "native.json").exists():
        L += ["", "### native（onnx-mlir 編譯執行）vs PyTorch 模擬", "",
              "| 候選 | native S_cal | 模擬 S_cal | native S_tune | 模擬 S_tune | 每窗口最大 abs 差 | finite | warnings |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
        for r in jload(out / "native.json")["results"]:
            sim = np.array(data[r["name"]]["dL"])
            nd = np.array(r["dL"])
            L.append(f"| {r['name'].replace('|', ' ')} | {nd[:CAL].sum():+.4f} | {sim[:CAL].sum():+.4f} | {nd[CAL:].sum():+.4f} | "
                     f"{sim[CAL:].sum():+.4f} | {np.abs(nd - sim).max():.4f} | {r['all_logits_finite']} | {len(r['warnings'])} |")
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("cache", "plan", "table", "loss", "analyze", "native",
                                                       "explore", "tables"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)), help="loss: layers to evaluate this run")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.stage == "cache":
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: use a new directory")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "cache.npz").exists():
        raise SystemExit("run --stage cache first")
    t0 = time.time()
    logf = open(out / f"{args.stage}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    globals()[f"stage_{args.stage}"](args, out, log)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
