"""Rebuild report figures from existing results; does not run model experiments."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"


def read(name):
    return json.loads((RUNS / name / "results.json").read_text())


plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
tree = read("mobilenetv2_timm_ra_tree_vs_milp_v2")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
for menu, color in [("small (3x3)", "#2878b5"), ("full (8x8)", "#cf6234")]:
    rows = sorted((r for r in tree["rows"] if r["menu"] == menu), key=lambda r: r["budget"])
    x = [r["budget"] for r in rows]
    for solver, style in [("tree", "--"), ("milp", "-")]:
        axes[0].plot(x, [100 * r[solver + "_top1"] for r in rows], style,
                     marker="o", color=color, label=f"{menu}: {solver.upper()}")
    axes[1].plot(x, [r["tree_objective"] / r["milp_objective"] for r in rows],
                 marker="o", color=color, label=menu)
axes[0].axhline(100 * tree["fp32_sim_top1"], color="gray", linestyle=":", label="FP32")
axes[0].set(ylabel="Top-1 accuracy (%)", ylim=(0, 78), title="Same criterion: shared_v2")
axes[0].legend(fontsize=8)
axes[1].axhline(1, color="gray", linestyle=":")
axes[1].set(ylabel="Tree objective / MILP objective", title="Search error (lower is better)")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("Budget / all-int8 cost")
    ax.set_xticks(x)
    ax.grid(alpha=.2)
fig.suptitle("MobileNetV2 | ORT simulation | 1,000 held-out images | beam cap 2,000 per axis", fontsize=11)
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"search_comparison.{suffix}", dpi=200)
plt.close(fig)

ablation = read("mobilenetv2_timm_ra_shared_fix")
compiled = read("mobilenetv2_timm_ra_criteria_milp")
compiled_v2 = read("mobilenetv2_timm_ra_shared_v2_compiled")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
colors = {"shared": "#777777", "shared_v1": "#a864a8", "shared_v2": "#2878b5", "sqnr": "#278355"}
for criterion in colors:
    rows = sorted((r for r in ablation["rows"] if r["criterion"] == criterion), key=lambda r: r["budget"])
    axes[0].plot([r["budget"] for r in rows], [100*r["sim_top1"] for r in rows],
                 marker="o", color=colors[criterion], label=criterion)
budgets = [.6, .65, .7, .75, .8]
compiled_rows = []
for criterion in ("shared", "split_connection", "split_direct", "shared_v2", "sqnr"):
    source = compiled_v2 if criterion == "shared_v2" else compiled
    acc = []
    for i, budget in enumerate(budgets):
        held = source["variants"][f"{criterion}_{i:03d}"]["held_out"]
        assert held["n"] == 1000
        acc.append(100 * held["top1"])
        compiled_rows.append({"criterion": criterion, "budget": budget, "n": held["n"], "top1_percent": acc[-1]})
    axes[1].plot(budgets, acc, marker="o", label=criterion, color=colors.get(criterion, {"split_connection": "#a864a8", "split_direct": "#cf6234"}.get(criterion)))
for ax in axes:
    ax.axhline(72.1, color="gray", linestyle=":", label="FP32 72.1%")
    ax.set(xlabel="Budget / all-int8 cost", ylabel="Top-1 accuracy (%)", ylim=(0, 78), xticks=budgets)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_title("Perturbation ablation (ORT simulation)")
axes[1].set_title("Separate saved runs (compiled QDQ)")
fig.suptitle("MobileNetV2 | Fixed solver: MILP | 10 calibration / 1,000 held-out images", fontsize=11)
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"perturbation_comparison.{suffix}", dpi=200)
plt.close(fig)
with (HERE / "compiled_accuracy.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(compiled_rows[0]))
    writer.writeheader()
    writer.writerows(compiled_rows)
with (HERE / "search_comparison.csv").open("w", newline="") as f:
    rows = [dict(r, objective_ratio=r["tree_objective"] / r["milp_objective"]) for r in tree["rows"]]
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote 2 PNGs, 2 PDFs, and 2 CSVs to {HERE}")
