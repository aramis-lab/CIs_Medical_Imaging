import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from ..df_loaders import extract_df_classif_width
from ..plot_utils import method_labels, method_colors, upload_to_overleaf


def _plot_metric_panel(ax, df, metric, panel_title):
    """Draw the CI-width boxplots for a single metric onto the given axis."""
    df_all = df[df["metric"] == metric]
    methods = sorted(df_all["method"].unique())
    preferred_order = ["basic", "bca", "percentile"]
    preferred = [m for m in preferred_order if m in methods]
    others = [m for m in methods if m not in preferred_order]
    methods = preferred + others

    max_width = 2 if metric == "mcc" else 1

    for i, n in enumerate(np.sort(df_all["n"].unique())):
        for j, method in enumerate(methods):
            widths = df_all[(df_all["n"] == n) & (df_all["method"] == method)]["width"]
            widths = widths.fillna(0.0)  # NaN widths correspond to degenerate CIs with width 0
            pos = (len(methods) + 2) * i + j
            if metric != "accuracy":
                pos = pos + 1
            ax.boxplot(widths, positions=[pos], widths=0.8, patch_artist=True,
                       boxprops=dict(facecolor=method_colors[method]),
                       flierprops=dict(marker='o', markersize=3, markerfacecolor=method_colors[method],
                                        markeredgewidth=1.5, markeredgecolor="black"),
                       medianprops=dict(color="white"), sym=method_colors[method])

    legend_handles = [mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods]

    ax.set_xlabel("Sample size", weight="bold")
    ax.set_ylabel("CI width", weight="bold")
    ax.set_ylim(0.0, max_width + 0.01)
    ax.set_title(panel_title, weight="bold")
    ax.set_xticks([(len(methods) + 2) * i + 2 for i in range(len(df_all["n"].unique()))])
    ax.set_xticklabels([f"{int(n)}" for n in np.sort(df_all["n"].unique())])
    ax.set_yticks(np.arange(0, max_width + 0.01, step=0.1*max_width))
    ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='black', linewidth=0.6)
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.25, 0.5))
    ax.set_xlim(-1, (len(methods) + 2) * len(df_all["n"].unique()))


def plot_all_width_classif(root_folder: str, output_path: str, upload_overleaf: bool = False):

    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro = ["accuracy", "auc", "f1_score", "ap"]
    averages_micro = ["micro"]

    df_micro = extract_df_classif_width(folder_path_micro, file_prefix_micro, metrics_micro, averages_micro)
    df_mcc = extract_df_classif_width(folder_path_micro, file_prefix_micro, ["mcc"], ["none"])

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro = ["balanced_accuracy", "auc", "f1_score", "ap"]
    averages_macro = ["macro"]

    df_macro = extract_df_classif_width(folder_path_macro, file_prefix_macro, metrics_macro, averages_macro)

    # Set Nature-style: clean, minimal, sans-serif, no grid, no top/right spines
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "legend.fontsize": 21,
        "axes.edgecolor": "black",
        "axes.linewidth": 2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })

    # 3x3 grid layout:
    #   row 1: accuracy (micro) | balanced_accuracy (macro) | MCC
    #   row 2: AUC (micro)      | F1 score (micro)           | AP (micro)
    #   row 3: AUC (macro)      | F1 score (macro)           | AP (macro)
    grid_spec = [
        [(df_micro, "accuracy", "ACCURACY"),
         (df_macro, "balanced_accuracy", "BALANCED ACCURACY"),
         (df_mcc, "mcc", "MCC")],
        [(df_micro, "auc", "AUC (micro)"),
         (df_micro, "f1_score", "F1 SCORE (micro)"),
         (df_micro, "ap", "AP (micro)")],
        [(df_macro, "auc", "AUC (macro)"),
         (df_macro, "f1_score", "F1 SCORE (macro)"),
         (df_macro, "ap", "AP (macro)")],
    ]

    fig, axs = plt.subplots(3, 3, figsize=(21 * 3, 15 * 3))

    for row_idx, row in enumerate(grid_spec):
        for col_idx, (df, metric, panel_title) in enumerate(row):
            _plot_metric_panel(axs[row_idx, col_idx], df, metric, panel_title)

    plt.tight_layout()

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

    if upload_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/{os.path.basename(output_path)}",
                            commit_msg="Update Supp Fig width of all methods and metrics for classification")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Supp Figure width of all methods and metrics for classification.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    parser.add_argument("--upload_overleaf", action="store_true", help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/width_classif.pdf")

    plot_all_width_classif(root_folder, output_path, upload_overleaf=args.upload_overleaf)


if __name__ == "__main__":
    main()