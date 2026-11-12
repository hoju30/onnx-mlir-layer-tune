#!/usr/bin/env python3
"""Separate W/X/Y term balance from within-term layer allocation using saved calibration tables.

For each term, scale mean|g| * squared_error so its sum across layers at int8 equals
the Fisher scalar sum at int8. Scales use calibration scores only, never test metrics.
This is a diagnostic normalization, not a Fisher estimator or a theoretical loss bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from eval_random_mixed_precision import (
    calibrate_ranges, np_fake_quant_weight, preprocess_by_source, simulate_in_onnxruntime, supported_nodes,
)
from mp_milp import milp_select
from sample_configs_htq import compute_weight_elems_and_flops_proxy
from sample_configs_sqnr import config_cost

HERE = Path(__file__).resolve().parent
FORMATS = ["int4", "int5", "int6", "int7", "int8", "f16", "bf16", "fp32"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["gpt2", "mbv2"], required=True)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budgets", help="Default: all budgets of source run")
    ap.add_argument("--variants", default="abs_matched_totals",
                    help="abs_matched_totals; abs_w/x/y_matched; mean_abs_squared_w[_matched]")
    ap.add_argument("--milp-time-limit", type=float, default=60.)
    args = ap.parse_args()
    variants = args.variants.split(",")
    if any(v not in ("abs_matched_totals", "abs_w_matched", "abs_x_matched", "abs_y_matched",
                     "mean_abs_squared_w", "mean_abs_squared_w_matched") for v in variants):
        ap.error("Unknown variant")
    meta = json.loads((args.source_run / "results.json").read_text())
    tables = json.loads((args.source_run / "criterion_tables.json").read_text())
    base, absolute = tables["2nd_tensor"], tables["abs_g_sq_error"]
    scales = {t: sum(v["int8"] for v in base[t].values()) /
              sum(v["int8"] for v in absolute[t].values()) for t in ("w", "x", "y")}
    matched = {t: {n: {f: scales[t] * v[f] for f in FORMATS} for n, v in absolute[t].items()}
               for t in scales}
    for t in scales:
        np.testing.assert_allclose(sum(v["int8"] for v in matched[t].values()),
                                   sum(v["int8"] for v in base[t].values()), rtol=1e-12)
    print("Calibration-only int8 matching factors:", scales, flush=True)
    budgets = [float(x) for x in args.budgets.split(",")] if args.budgets else meta["budgets"]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "matched_tables.json").write_text(json.dumps(matched) + "\n")

    if args.model == "gpt2":
        src = Path(meta["source_run"])
        graph = onnx.load(str(src / "model.onnx"))
        mapping = json.loads((src / "node_mapping.json").read_text())
        nodes = [n for n in supported_nodes(graph) if n["name"] in mapping]
        ranges = json.loads((src / "calibration.json").read_text())
        inputs = np.load(src / "test_tokens.npy")
        init = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
        by_name = {n.name: n for n in graph.graph.node}
        elems = {n["name"]: init[by_name[n["name"]].input[1]].size for n in nodes}
        flops = {n: size * inputs.shape[1] for n, size in elems.items()}
    else:
        from run_mbv2_fisher_term_ablation import torch
        torch.set_num_threads(4)
        model_dir, images_dir = Path(meta["model_dir"]), Path(meta["images_dir"])
        graph = onnx.load(str(model_dir / "model.onnx"))
        model_meta = json.loads((model_dir / "metadata.json").read_text())
        manifest = json.loads((images_dir / "manifest.json").read_text())
        calib = [e for e in manifest["images"] if e["calibration"]]
        held = [e for e in manifest["images"] if not e["calibration"]]
        inputs = preprocess_by_source(images_dir, {"images": held}, model_meta)
        labels = np.array([e["label"] for e in held])
        nodes = supported_nodes(graph)
        ranges = calibrate_ranges(graph, nodes, preprocess_by_source(images_dir, {"images": calib}, model_meta))
        elems, flops = compute_weight_elems_and_flops_proxy(graph, nodes)
    names = list(base["w"])
    assert set(names) == {n["name"] for n in nodes if n.get("weight_const")}
    moment_w, moment_scales = {}, {}
    if any(v.startswith("mean_abs_squared") for v in variants):
        # Original tables use G1=sum_calibration mean_elements|g| and
        # G2=sum_calibration mean_elements(g^2). Recover these from one nonzero
        # quantization error, then compare D*(G1/D)^2 with G2 at fixed error.
        count = meta["calibration_windows" if args.model == "gpt2" else "calibration_images"]
        init = {i.name: onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
        by_name = {n.name: n for n in graph.graph.node}
        stats = {}
        for n in names:
            weight = init[by_name[n].input[1]]
            error = np_fake_quant_weight(weight, "int4", None).astype(np.float64) - weight.astype(np.float64)
            energy = float(np.square(error).sum())
            if energy <= 0:
                raise ValueError(f"Cannot recover gradient moments from zero error: {n}")
            g1, g2 = absolute["w"][n]["int4"] / energy, base["w"][n]["int4"] / energy
            assert g1 * g1 / count <= g2 * (1 + 1e-6), (n, g1, g2)  # Jensen consistency
            # Check that int8 independently recovers the same sensitivity.
            e8 = np_fake_quant_weight(weight, "int8", None).astype(np.float64) - weight.astype(np.float64)
            np.testing.assert_allclose(g1 * float(np.square(e8).sum()), absolute["w"][n]["int8"], rtol=1e-5)
            moment_w[n] = {f: absolute["w"][n][f] * (g1 / count) for f in FORMATS}
            stats[n] = {"mean_abs_gradient": g1 / count, "mean_gradient_squared": g2 / count,
                        "concentration_ratio": g2 * count / (g1 * g1) if g1 else None}
        moment_scale = sum(v["int8"] for v in base["w"].values()) / sum(v["int8"] for v in moment_w.values())
        moment_scales = {"weight_total_matching_factor": moment_scale, "calibration_count": count}
        (args.out_dir / "weight_gradient_moments.json").write_text(json.dumps(stats, indent=2) + "\n")
        (args.out_dir / "mean_abs_squared_weight_tables.json").write_text(json.dumps(moment_w) + "\n")
        print("Weight mean-absolute-gradient squared matching factor:", moment_scale, flush=True)
    ref_size = sum(elems[n] for n in names)
    ref_bops = sum(64 * flops[n] for n in names)
    rows = []
    result = {"backend": "ORT QDQ simulation", "model": args.model, "source_run": str(args.source_run),
              "normalization": "Variants ending in matched or matched_totals match int8 totals using calibration only",
              "scales": scales, "gradient_moment_diagnostic": moment_scales,
              "budgets": budgets, "variants": variants, "rows": rows}
    for budget in budgets:
        for variant in variants:
            terms = {t: matched[t] if variant in ("abs_matched_totals", f"abs_{t}_matched") else base[t]
                     for t in ("w", "x", "y")}
            if variant.startswith("mean_abs_squared"):
                factor = moment_scale if variant.endswith("_matched") else 1.0
                terms["w"] = {n: {f: factor * v[f] for f in FORMATS} for n, v in moment_w.items()}
            def value(n, w, a):
                return terms["w"][n][w] + terms["x"][n][a] + terms["y"][n][a]
            cfg, obj, status = milp_select(nodes, FORMATS, FORMATS, value, elems, flops,
                                           budget * ref_size, budget * ref_bops, args.milp_time_limit)
            size, bops = config_cost(cfg, elems, flops)
            assert size <= budget * ref_size and bops <= budget * ref_bops
            row = {"budget": budget, "variant": variant, "milp_optimal": status == 0,
                   "size": size / ref_size, "bops": bops / ref_bops, "objective": obj, "formats": cfg}
            reuse = next((r for r in meta["rows"] if r["formats"] == cfg), None)
            if reuse:
                for key in ("nll", "perplexity", "top1", "loss", "nonfinite_images"):
                    if key in reuse:
                        row[key] = reuse[key]
                row["reused_metrics_from"] = {"budget": reuse["budget"], "variant": reuse["variant"]}
            else:
                z = simulate_in_onnxruntime(graph, nodes, cfg, ranges, inputs, None)
                assert np.isfinite(z).all(), (budget, variant)
                if args.model == "gpt2":
                    from scipy.special import logsumexp
                    losses = np.concatenate([logsumexp(s.astype(np.float64), axis=-1) - s[np.arange(len(t)), t]
                                             for s, t in zip(z[:, :-1], inputs[:, 1:])])
                    np.save(args.out_dir / f"nll_f{budget:g}_{variant}.npy", losses)
                    row.update(nll=float(losses.mean()), perplexity=float(np.exp(losses.mean())),
                               top1=float((z[:, :-1].argmax(-1) == inputs[:, 1:]).mean()))
                else:
                    correct = (z.argmax(1) == labels).astype(float)
                    losses = torch.nn.functional.cross_entropy(torch.from_numpy(z), torch.from_numpy(labels),
                                                               reduction="none").numpy()
                    np.savez(args.out_dir / f"metrics_f{budget:g}_{variant}.npz", correct=correct, loss=losses)
                    row.update(top1=float(correct.mean()), loss=float(losses.mean()), nonfinite_images=0)
                del z
            rows.append(row)
            (args.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print({k: v for k, v in row.items() if k != "formats"}, flush=True)


if __name__ == "__main__":
    main()
