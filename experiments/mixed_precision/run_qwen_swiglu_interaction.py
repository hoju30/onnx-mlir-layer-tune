#!/usr/bin/env python3
"""Calibration-only feasibility study of Gate/Up interaction-aware block costs.

The FP32 model and Down projection stay unchanged. Each branch uses the existing
per-consumer QDQ rules and native calibration parameters. Replayed block costs
are compared with the existing 108 full-model calibration experiments; eight
mixed-format cases at layer 2 are evaluated anew. No test data or search policy
is used or changed. This is a PyTorch diagnostic, not a native inference run.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
STUDY = HERE / "runs/qwen3_swiglu_20261013"
TABLES = HERE / "runs/native_eps_fix_20261014/qwen3"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def interaction_scan(rows):
    """Diagnose signed loss interactions; keep loss and loss-squared distinct."""
    pairs = [r for r in rows if r["target"] == "both"]
    result = {"joint_cases": len(pairs), "positive": sum(r["dL_joint_minus_sum"] > 0 for r in pairs),
              "negative": sum(r["dL_joint_minus_sum"] < 0 for r in pairs), "predictors": {}}
    for key in ("cross_2fgfu", "swiglu_second_order"):
        real = np.array([r["dL_joint_minus_sum"] for r in pairs])
        pred = np.array([r[key] for r in pairs])
        mask = np.abs(real) > 1
        result["predictors"][key] = {
            "spearman_signed_interaction": correlation(pred, real),
            "direction_correct_abs_I_gt_1": int((np.sign(pred[mask]) == np.sign(real[mask])).sum()),
            "direction_cases": int(mask.sum()),
            "units": "loss_squared" if key == "cross_2fgfu" else "loss",
        }
        if key != "cross_2fgfu":
            result["predictors"][key].update(
                median_relative_error_abs_I_gt_1=float(np.median(np.abs(pred[mask] - real[mask]) / np.abs(real[mask]))),
                mae=float(np.abs(pred - real).mean()), zero_prediction_mae=float(np.abs(real).mean()))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()

    def log(message):
        print(f"[{time.time() - start:7.1f}s] {message}", flush=True)

    analysis = json.loads((STUDY / "phase5_quant/analysis.json").read_text())
    originals = {r["name"]: r for r in analysis["rows"]}
    scan = interaction_scan(analysis["rows"])
    dump(out / "existing_interactions.json", scan)
    if args.scan_only:
        log(json.dumps(scan))
        return

    from qwen3_swiglu_common import (Quantizers, environment, int_act_qdq,
                                    load_model, node_mapping, sha256, summed_ce, torch)
    from qwen3_swiglu_criterion import int_params
    from types import SimpleNamespace

    torch.set_num_threads(args.threads)
    params = int_params(SimpleNamespace(native_run=TABLES))
    mapping = node_mapping()
    by_layer = {(layer, kind): node for node, (layer, kind, _) in mapping.items()}
    tokens = np.load(TABLES / "calib_inputs.npy")
    if tokens.shape != (10, 128):
        raise RuntimeError(f"Expected the existing ten calibration windows, got {tokens.shape}")
    layers, formats, modes = (0, 2, 14, 27), ("int4", "int8", "p6e1"), ("w", "a", "wa")
    dump(out / "environment.json", {
        **environment(["experiments/mixed_precision/run_qwen_swiglu_interaction.py"]),
        "study_sim_sha256": sha256(STUDY / "phase5_quant/sim.json"),
        "study_analysis_sha256": sha256(STUDY / "phase5_quant/analysis.json"),
        "table_sha256": sha256(TABLES / "table.json"),
        "branch_table_sha256": sha256(TABLES / "table_branch.json"),
        "calibration_sha256": sha256(TABLES / "calib_inputs.npy"),
        "threads": args.threads, "used_test_data": False,
    })
    q = Quantizers(out, formats=("p6e1",))
    bad, corpus_size = q.posit["p6e1"].verify()
    if bad:
        raise RuntimeError(f"Posit6 LUT verification failed: {bad}/{corpus_size}")
    model, wrapper = load_model(torch.float32)
    mods = dict(model.named_modules())
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)

    # One fresh baseline pass. Save exactly the tensors missing from the old
    # scalar JSON; no old .so or gradient cache is reused.
    base, losses = {layer: [] for layer in layers}, []
    for window, ids in enumerate(tokens):
        captured, hooks = {}, []

        def projection_hook(layer, kind):
            def hook(module, inputs, output):
                captured[(layer, kind)] = (inputs[0], output)
                if kind == "down":
                    inputs[0].retain_grad()
            return hook

        for layer in layers:
            mlp = mods[f"model.layers.{layer}.mlp"]
            for kind in ("gate", "up", "down"):
                hooks.append(getattr(mlp, f"{kind}_proj").register_forward_hook(projection_hook(layer, kind)))
        try:
            loss = summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids))
            loss.backward()
            losses.append(float(loss.detach()))
            for layer in layers:
                x, zg = captured[(layer, "gate")]
                xu, zu = captured[(layer, "up")]
                h = captured[(layer, "down")][0]
                if not torch.equal(x, xu):
                    raise RuntimeError("Gate/Up no longer share their FP32 input")
                reconstructed = torch.nn.functional.silu(zg) * zu
                torch.testing.assert_close(reconstructed, h, rtol=1e-6, atol=1e-6)
                base[layer].append({"x": x[0].detach().clone(), "zg": zg[0].detach().clone(),
                                    "zu": zu[0].detach().clone(), "h": h[0].detach().clone(),
                                    "gh": h.grad[0].detach().clone()})
        finally:
            for hook in hooks:
                hook.remove()
            model.zero_grad(set_to_none=True)
        log(f"Baseline {window + 1}/{len(tokens)}")
    for p in model.parameters():
        p.requires_grad_(False)
    np.savez(out / "fp32_block_cache.npz", **{
        f"L{layer}|{key}": np.stack([b[key].numpy() for b in samples])
        for layer, samples in base.items() for key in ("x", "zg", "zu", "h", "gh")})
    dump(out / "baseline.json", {"losses": losses, "windows": len(tokens), "tokens_per_window": 127,
                                 "posit6_corpus_checked": corpus_size})
    baseline_s = time.time() - start

    def qdq(x, node, slot, fmt):
        if fmt.startswith("int"):
            scale, zp = params[(node, slot)][fmt]
            return int_act_qdq(x, scale, zp, int(fmt[3:]))
        return q.act(x, fmt)

    def phi(gh, delta):
        """All scores are nonnegative loss-squared proxies, not signed loss."""
        g, d = gh.double(), torch.stack(delta).double()   # [window, position, channel]
        effect = (g * d).sum((1, 2))
        mean = d.mean((0, 1))
        variance = ((d - mean) ** 2).mean((0, 1))
        g2, gs = (g ** 2).sum((0, 1)), (g.sum(1) ** 2).sum(0)
        return {"directional": float((effect ** 2).sum()),
                "diagonal": float(((g * d) ** 2).sum()),
                "moments": float((g2 * variance + gs * mean ** 2).sum())}

    results = []
    mixed_pairs = (("int4", "int8"), ("int8", "int4"), ("p6e1", "int8"), ("int8", "p6e1"))
    for layer in layers:
        samples, branch_outputs = base[layer], {}
        gh = torch.stack([b["gh"] for b in samples])
        modules = {kind: mods[f"model.layers.{layer}.mlp.{kind}_proj"] for kind in ("gate", "up")}
        with torch.no_grad():
            for kind, fmt in itertools.product(("gate", "up"), formats):
                node, module = by_layer[(layer, kind)], modules[kind]
                qw = q.weight(module.weight, fmt)
                for mode in modes:
                    values = []
                    for b in samples:
                        x = qdq(b["x"], node, "x", fmt) if "a" in mode else b["x"]
                        y = x @ (qw if "w" in mode else module.weight).T
                        values.append(qdq(y, node, "y", fmt) if "a" in mode else y)
                    branch_outputs[(kind, mode, fmt)] = values
            cases = [(target, mode, fmt, fmt) for target, mode, fmt in itertools.product(
                ("gate", "up", "both"), modes, formats)]
            if layer == 2:
                cases += [("both", mode, g, u) for mode, (g, u) in itertools.product(("a", "wa"), mixed_pairs)]
            for target, mode, gate_fmt, up_fmt in cases:
                mixed = gate_fmt != up_fmt
                name = (f"L{layer}_{target}_{mode}_{gate_fmt}" if not mixed else
                        f"L{layer}_both_{mode}_gate_{gate_fmt}_up_{up_fmt}")
                dg_only, du_only, joint, residual, taylor = [], [], [], [], []
                identity_errors, identity_scales, fp32_roundoff = [], [], []
                for s, b in enumerate(samples):
                    zg = branch_outputs[("gate", mode, gate_fmt)][s] if target != "up" else b["zg"]
                    zu = branch_outputs[("up", mode, up_fmt)][s] if target != "gate" else b["zu"]
                    silu0, silug = torch.nn.functional.silu(b["zg"]), torch.nn.functional.silu(zg)
                    dg_only.append(silug * b["zu"] - b["h"])
                    du_only.append(silu0 * zu - b["h"])
                    joint.append(silug * zu - b["h"])
                    # Verify the algebra in FP64: the FP32 h values can be
                    # thousands while INT8 deltas are tiny, so subtracting
                    # four rounded products does not obey an exact identity.
                    zg0d, zu0d = b["zg"].double(), b["zu"].double()
                    zgd, zud = zg.double(), zu.double()
                    sg0d = torch.nn.functional.silu(zg0d)
                    sgd = torch.nn.functional.silu(zgd)
                    hd = sg0d * zu0d
                    egd, eud = sgd * zu0d - hd, sg0d * zud - hd
                    ejd = sgd * zud - hd
                    rd = (sgd - sg0d) * (zud - zu0d)
                    residual.append(rd)
                    identity_errors.append(float((ejd - egd - eud - rd).abs().max()))
                    identity_scales.append(max(1.0, float(ejd.abs().max())))
                    fp32_roundoff.append(float((joint[-1].double() - ejd).abs().max()))
                    sig = torch.sigmoid(b["zg"].double())
                    derivative = sig + b["zg"].double() * sig * (1 - sig)
                    taylor.append(derivative * (zg - b["zg"]).double() * (zu - b["zu"]).double())
                sg, su, sj = phi(gh, dg_only), phi(gh, du_only), phi(gh, joint)
                linear = phi(gh, [g + u for g, u in zip(dg_only, du_only)])
                absolute_identity, scale = max(identity_errors), max(identity_scales)
                if absolute_identity > 1e-10 * scale:
                    raise RuntimeError(f"Mixed difference identity failed: {name}: {absolute_identity}/{scale}")
                row = {"name": name, "layer": layer, "target": target, "mode": mode,
                       "gate_fmt": gate_fmt, "up_fmt": up_fmt, "mixed_formats": mixed,
                       "scores_additive": {k: sg[k] + su[k] for k in sg},
                       "scores_joint_linear": linear, "scores_joint_exact": sj,
                       "interaction_local_exact": sum(float((b["gh"].double() * d.double()).sum())
                                                      for b, d in zip(samples, residual)),
                       "interaction_local_taylor": sum(float((b["gh"].double() * d).sum())
                                                       for b, d in zip(samples, taylor)),
                       "identity_max_abs": absolute_identity, "identity_relative_scale": absolute_identity / scale,
                       "fp32_vs_fp64_local_delta_max_abs": max(fp32_roundoff)}
                if not mixed:
                    old = originals[name]
                    row.update(dL_sum=old["dL_sum"], dL_abs_sum=old["dL_abs_sum"],
                               BC_total=old["C"], BC_branch=old["C_branch"])
                    if target == "both":
                        row["interaction_true"] = old["dL_joint_minus_sum"]
                results.append(row)
        log(f"Replayed layer {layer}, {len(results)} cases so far")
    replay_s = time.time() - start - baseline_s

    # True calibration loss for the new mixed-format cases only. Singles come
    # from the old, same-checkpoint study; this limitation is recorded below.
    t0 = time.time()
    with torch.no_grad():
        for row in [r for r in results if r["mixed_formats"]]:
            layer, mode = row["layer"], row["mode"]
            saved, hooks = {}, []
            try:
                for kind in ("gate", "up"):
                    module = mods[f"model.layers.{layer}.mlp.{kind}_proj"]
                    fmt, node = row[f"{kind}_fmt"], by_layer[(layer, kind)]
                    if "w" in mode:
                        saved[kind] = module.weight.data
                        module.weight.data = q.weight(module.weight.data, fmt)
                    if "a" in mode:
                        hooks.append(module.register_forward_pre_hook(
                            lambda mm, inp, n=node, f=fmt: (qdq(inp[0], n, "x", f),)))
                        hooks.append(module.register_forward_hook(
                            lambda mm, inp, output, n=node, f=fmt: qdq(output, n, "y", f)))
                quant_losses = [float(summed_ce(wrapper(torch.from_numpy(ids[None])), torch.from_numpy(ids)))
                                for ids in tokens]
            finally:
                for hook in hooks:
                    hook.remove()
                for kind, weight in saved.items():
                    mods[f"model.layers.{layer}.mlp.{kind}_proj"].weight.data = weight
            delta = np.array(quant_losses) - np.array(losses)
            gate = originals[f"L{layer}_gate_{mode}_{row['gate_fmt']}"]
            up = originals[f"L{layer}_up_{mode}_{row['up_fmt']}"]
            row.update(dL_sum=float(delta.sum()), dL_abs_sum=float(np.abs(delta).sum()),
                       dL_per_window=delta.tolist(), BC_total=gate["C"] + up["C"],
                       BC_branch=gate["C_branch"] + up["C_branch"],
                       interaction_true=float(delta.sum() - gate["dL_sum"] - up["dL_sum"]))
            log(f"Mixed-format full-model calibration: {row['name']}, delta={row['dL_sum']:.3f}")
    dump(out / "probe_rows.json", results)

    # Compare like-for-like metrics: additive, joint linear, joint nonlinear.
    summary = {"units": {"costs": "loss_squared", "interaction_predictors": "loss"},
               "used_test_data": False, "independent_tuning_evaluated": False,
               "native_executed": False, "groups": {}, "interaction": {},
               "timing_s": {"baseline_and_setup": baseline_s, "local_replay": replay_s,
                            "mixed_full_model_eval": time.time() - t0, "total": time.time() - start}}
    for group in ("existing_all", "existing_joint", "mixed_joint"):
        rows = [r for r in results if (r["mixed_formats"] if group == "mixed_joint" else
                not r["mixed_formats"] and (group == "existing_all" or r["target"] == "both"))]
        metrics = {}
        for target in ("abs_net_loss", "sum_absolute_window_loss"):
            y = [abs(r["dL_sum"]) if target == "abs_net_loss" else r["dL_abs_sum"] for r in rows]
            metrics[target] = {"BC_total": correlation([r["BC_total"] for r in rows], y),
                               "BC_branch": correlation([r["BC_branch"] for r in rows], y)}
            for structure, metric in itertools.product(("additive", "joint_linear", "joint_exact"),
                                                       ("directional", "diagonal", "moments")):
                metrics[target][f"{structure}.{metric}"] = correlation(
                    [r[f"scores_{structure}"][metric] for r in rows], y)
        summary["groups"][group] = {"cases": len(rows), "spearman": metrics}
    pairs = [r for r in results if r["target"] == "both"]
    for key in ("interaction_local_taylor", "interaction_local_exact"):
        real = np.array([r["interaction_true"] for r in pairs])
        pred = np.array([r[key] for r in pairs])
        mask = np.abs(real) > 1
        summary["interaction"][key] = {"cases": len(pairs), "spearman": correlation(pred, real),
            "direction_correct_abs_I_gt_1": int((np.sign(pred[mask]) == np.sign(real[mask])).sum()),
            "direction_cases": int(mask.sum()), "mae": float(np.abs(pred - real).mean()),
            "zero_prediction_mae": float(np.abs(real).mean())}
    dump(out / "summary.json", summary)
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
