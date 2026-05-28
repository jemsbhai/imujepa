"""
Generate publication-quality figures and tables for the paper.

Reads experiment results from experiments/exp_*/results.json and produces:
  - figures/*.pdf (matplotlib, camera-ready quality)
  - figures/*.tex (LaTeX table source, paste into Overleaf)

Usage:
    python -m scripts.generate_paper_assets
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "mae": "#2196F3",
    "jepa": "#FF5722",
    "hybrid": "#4CAF50",
}

METHOD_LABELS = {
    "mae": "MAE",
    "jepa": "JEPA (SIGReg)",
    "hybrid": "Hybrid (ours)",
}

DATASET_LABELS = {
    "uci_har": "UCI-HAR",
    "pamap2": "PAMAP2",
    "dsads": "DSADS",
    "mhealth": "MHealth",
    "usc_had": "USC-HAD",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_alpha_ablation_figure(exp_dir, fig_dir):
    """Figure 2: Alpha ablation curve (inverted-U with peak at 0.5)."""
    results = load_json(exp_dir / "exp_006_alpha_ablation" / "results.json")

    # Also include pure JEPA (alpha=0) and pure MAE (alpha=inf)
    exp005 = load_json(exp_dir / "exp_005_hybrid_cross_dataset" / "results.json")
    mae_avg = exp005["summary"]["avg_macro_f1"]["mae"]
    jepa_avg = exp005["summary"]["avg_macro_f1"]["jepa"]

    alphas = [0.0]
    f1s = [jepa_avg]

    for a_str, avg in sorted(results["averages"].items(), key=lambda x: float(x[0])):
        alphas.append(float(a_str))
        f1s.append(avg)

    # Add MAE as alpha -> infinity (plot at 1.0 for visualization)
    alphas.append(1.0)
    f1s.append(mae_avg)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    ax.plot(alphas, f1s, "o-", color=COLORS["hybrid"], linewidth=2, markersize=7, zorder=5)

    # Annotate key points
    ax.annotate("Pure JEPA", (0.0, jepa_avg), textcoords="offset points",
                xytext=(12, -12), fontsize=8, color=COLORS["jepa"])
    ax.annotate("Pure MAE", (1.0, mae_avg), textcoords="offset points",
                xytext=(-55, -12), fontsize=8, color=COLORS["mae"])

    # Highlight best
    best_idx = np.argmax(f1s)
    ax.plot(alphas[best_idx], f1s[best_idx], "*", color=COLORS["hybrid"],
            markersize=15, zorder=6)
    ax.annotate(f"Best: α={alphas[best_idx]}", (alphas[best_idx], f1s[best_idx]),
                textcoords="offset points", xytext=(10, 8), fontsize=8,
                fontweight="bold", color=COLORS["hybrid"])

    ax.set_xlabel("Reconstruction weight (α)")
    ax.set_ylabel("Avg. cross-dataset macro-F1")
    ax.set_xticks([0, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0\n(JEPA)", "0.1", "0.25", "0.5", "1.0\n(MAE)"])

    fig.savefig(fig_dir / "alpha_ablation.pdf")
    fig.savefig(fig_dir / "alpha_ablation.png")
    plt.close(fig)
    print("  Saved alpha_ablation.pdf")


def generate_cross_dataset_bar_chart(exp_dir, fig_dir):
    """Figure: Grouped bar chart of cross-dataset FT results."""
    exp007 = load_json(exp_dir / "exp_007_finetune" / "results.json")

    methods = ["mae", "jepa", "hybrid"]
    datasets = list(DATASET_LABELS.keys())

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(datasets))
    width = 0.25

    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            if ds in exp007["cross_dataset"].get(method, {}):
                vals.append(exp007["cross_dataset"][method][ds]["ft_macro_f1"])
            else:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method],
                      color=COLORS[method], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Macro-F1 (fine-tuned)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0.4, 1.0)

    fig.savefig(fig_dir / "cross_dataset_ft_bars.pdf")
    fig.savefig(fig_dir / "cross_dataset_ft_bars.png")
    plt.close(fig)
    print("  Saved cross_dataset_ft_bars.pdf")


def generate_within_dataset_table(exp_dir, fig_dir):
    """LaTeX table: within-dataset UCI-HAR results (3 seeds)."""
    exp007 = load_json(exp_dir / "exp_007_finetune" / "results.json")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Within-dataset results on UCI-HAR (3 seeds, mean $\pm$ std).}",
        r"\label{tab:within}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & LP Macro-F1 & FT Macro-F1 \\",
        r"\midrule",
    ]

    for method in ["mae", "jepa", "hybrid"]:
        wd = exp007["within_dataset"][method]
        lp = f"{wd['lp_mean']:.3f} $\\pm$ {wd['lp_std']:.3f}"
        ft = f"{wd['ft_mean']:.3f} $\\pm$ {wd['ft_std']:.3f}"
        label = METHOD_LABELS[method]
        if method == "hybrid":
            label = r"\textbf{" + label + "}"
            lp = r"\textbf{" + lp + "}"
            ft = r"\textbf{" + ft + "}"
        lines.append(f"{label} & {lp} & {ft} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = fig_dir / "table_within_dataset.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path.name}")


def generate_cross_dataset_table(exp_dir, fig_dir):
    """LaTeX table: cross-dataset FT results."""
    exp007 = load_json(exp_dir / "exp_007_finetune" / "results.json")

    methods = ["mae", "jepa", "hybrid"]
    datasets = list(DATASET_LABELS.keys())

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-dataset leave-one-out results (macro-F1). Pretrain on 4 datasets, evaluate on the held-out dataset.}",
        r"\label{tab:cross}",
        r"\begin{tabular}{l" + "cc" * len(methods) + "}",
        r"\toprule",
        r"& \multicolumn{2}{c}{MAE} & \multicolumn{2}{c}{JEPA (SIGReg)} & \multicolumn{2}{c}{\textbf{Hybrid (ours)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Held-out & LP & FT & LP & FT & LP & FT \\",
        r"\midrule",
    ]

    ft_avgs = {m: [] for m in methods}
    lp_avgs = {m: [] for m in methods}

    for ds in datasets:
        row = DATASET_LABELS[ds]
        for m in methods:
            cd = exp007["cross_dataset"][m][ds]
            lp_val = cd["lp_macro_f1"]
            ft_val = cd["ft_macro_f1"]
            lp_avgs[m].append(lp_val)
            ft_avgs[m].append(ft_val)
            row += f" & {lp_val:.3f} & {ft_val:.3f}"
        row += r" \\"
        lines.append(row)

    # Average row
    lines.append(r"\midrule")
    row = r"\textbf{Average}"
    for m in methods:
        lp_avg = np.mean(lp_avgs[m])
        ft_avg = np.mean(ft_avgs[m])
        if m == "hybrid":
            row += f" & \\textbf{{{lp_avg:.3f}}} & \\textbf{{{ft_avg:.3f}}}"
        else:
            row += f" & {lp_avg:.3f} & {ft_avg:.3f}"
    row += r" \\"
    lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    tex_path = fig_dir / "table_cross_dataset.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path.name}")


def generate_alpha_ablation_table(exp_dir, fig_dir):
    """LaTeX table: alpha ablation."""
    results = load_json(exp_dir / "exp_006_alpha_ablation" / "results.json")
    exp005 = load_json(exp_dir / "exp_005_hybrid_cross_dataset" / "results.json")

    datasets = list(DATASET_LABELS.keys())
    mae_f1 = exp005["results_macro_f1"]["mae"]
    jepa_f1 = exp005["results_macro_f1"]["jepa"]

    # Build alpha columns: 0 (JEPA), 0.1, 0.25, 0.5, 1.0 (MAE)
    alpha_keys = sorted(results["averages"].keys(), key=float)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hybrid mixing weight ablation (cross-dataset LP, macro-F1).}",
        r"\label{tab:alpha}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Held-out & $\alpha$=0 & $\alpha$=0.1 & $\alpha$=0.25 & $\alpha$=\textbf{0.5} & $\alpha$=1.0 \\",
        r" & (JEPA) & & & & (MAE) \\",
        r"\midrule",
    ]

    for ds in datasets:
        row = DATASET_LABELS[ds]
        row += f" & {jepa_f1[ds]:.3f}"
        for a_str in alpha_keys:
            val = results["results_macro_f1"][a_str][ds]
            row += f" & {val:.3f}"
        row += f" & {mae_f1[ds]:.3f}"
        row += r" \\"
        lines.append(row)

    # Averages
    lines.append(r"\midrule")
    row = r"\textbf{Average}"
    jepa_avg = np.mean([jepa_f1[ds] for ds in datasets])
    row += f" & {jepa_avg:.3f}"
    for a_str in alpha_keys:
        avg = results["averages"][a_str]
        if a_str == "0.5":
            row += f" & \\textbf{{{avg:.3f}}}"
        else:
            row += f" & {avg:.3f}"
    mae_avg = np.mean([mae_f1[ds] for ds in datasets])
    row += f" & {mae_avg:.3f}"
    row += r" \\"
    lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = fig_dir / "table_alpha_ablation.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path.name}")


def generate_fewshot_figure(exp_dir, fig_dir):
    """Figure 3: Few-shot label efficiency curves."""
    results_path = exp_dir / "exp_008_fewshot" / "results.json"
    if not results_path.exists():
        print("  Skipping fewshot figure (EXP-008 not complete)")
        return

    results = load_json(results_path)
    methods = ["mae", "jepa", "hybrid"]
    fractions = [float(f) for f in results["fractions"]]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for method in methods:
        f1s = [results["results"][method][str(f)]["macro_f1"] for f in fractions]
        ax.plot(fractions, f1s, "o-", color=COLORS[method],
                label=METHOD_LABELS[method], linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Fraction of labeled training data")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(fractions)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0%}" if x >= 0.1 else f"{x:.0%}"))
    ax.legend(loc="lower right", framealpha=0.9)

    fig.savefig(fig_dir / "fewshot_curves.pdf")
    fig.savefig(fig_dir / "fewshot_curves.png")
    plt.close(fig)
    print("  Saved fewshot_curves.pdf")


def generate_noise_figure(exp_dir, fig_dir):
    """Figure 4: Noise robustness curves."""
    results_path = exp_dir / "exp_009_noise" / "results.json"
    if not results_path.exists():
        print("  Skipping noise figure (EXP-009 not complete)")
        return

    results = load_json(results_path)
    methods = ["mae", "jepa", "hybrid"]
    snr_labels = results["snr_levels_db"]  # ["clean", 20, 10, 5, 0]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for method in methods:
        f1s = []
        for snr in snr_labels:
            key = str(snr) if snr != "clean" else "clean"
            f1s.append(results["results"][method][key]["macro_f1"])
        # x-axis: clean as rightmost (best), 0dB as leftmost (worst)
        # Plot left-to-right: 0, 5, 10, 20, clean
        x_positions = list(range(len(snr_labels)))
        ax.plot(x_positions, f1s, "o-", color=COLORS[method],
                label=METHOD_LABELS[method], linewidth=2, markersize=6)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Macro-F1")
    x_labels = [str(s) if s != "clean" else "Clean" for s in snr_labels]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.savefig(fig_dir / "noise_robustness.pdf")
    fig.savefig(fig_dir / "noise_robustness.png")
    plt.close(fig)
    print("  Saved noise_robustness.pdf")


def generate_dataset_table(fig_dir):
    """LaTeX table: dataset overview."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HAR benchmark datasets used in our experiments.}",
        r"\label{tab:datasets}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & Subj. & Act. & Ch. & Location & Hz & Windows \\",
        r"\midrule",
        r"UCI-HAR & 30 & 6 & 6 & Waist & 50 & 10,299 \\",
        r"PAMAP2 & 9 & 12 & 6 & Hand & 50 & 7,541 \\",
        r"DSADS & 8 & 19 & 6 & Torso & 50 & 9,120 \\",
        r"MHealth & 10 & 12 & 6 & Wrist & 50 & 2,677 \\",
        r"USC-HAD & 14 & 12 & 6 & Hip & 50 & 10,543 \\",
        r"\midrule",
        r"\textbf{Total} & \textbf{71} & & & & & \textbf{40,180} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = fig_dir / "table_datasets.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path.name}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    fig_dir = project_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Paper Assets")
    print("=" * 60)

    print("\nTables (LaTeX):")
    generate_dataset_table(fig_dir)
    generate_within_dataset_table(exp_dir, fig_dir)
    generate_cross_dataset_table(exp_dir, fig_dir)
    generate_alpha_ablation_table(exp_dir, fig_dir)

    print("\nFigures (PDF + PNG):")
    generate_alpha_ablation_figure(exp_dir, fig_dir)
    generate_cross_dataset_bar_chart(exp_dir, fig_dir)
    generate_fewshot_figure(exp_dir, fig_dir)
    generate_noise_figure(exp_dir, fig_dir)

    print(f"\nAll assets saved to {fig_dir}/")
    print("Copy .tex files into Overleaf, .pdf figures into figures/ in Overleaf.")


if __name__ == "__main__":
    main()
