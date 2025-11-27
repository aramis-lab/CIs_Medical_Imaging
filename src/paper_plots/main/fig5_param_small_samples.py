import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import argparse

from ..plot_utils import method_labels, method_colors, metric_labels
from ..df_loaders import extract_df_classif_cov, extract_df_segm_cov


def plot_fig5_param_small_samples(root_folder:str, output_path:str):

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

    folder_path_classif = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_classif = "aggregated_results"
    metrics_classif = ["accuracy"]

    df_classif = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)

    # Remove matching rows
    preferred_order = ["basic", "bca", "percentile"]
    methods = df_classif["method"].unique()
    preferred = [m for m in preferred_order if m in methods]
    others = sorted([m for m in methods if m not in preferred_order])

    methods = preferred+others

    fig, axs = plt.subplots(2, 1, figsize=(27, 27))

    ax = axs[0]
    ax.hlines(np.arange(0, 1.01, 0.05), 0, (len(methods)+2)*len(df_classif["n"].unique()-2), colors="black", linewidths=0.6, linestyles=(0, (5,10)))

    arrow_legend_down = False
    arrow_legend_up = False
    n_values = [n for n in np.sort(df_classif["n"].unique()) if n<=100]
    for i, n in enumerate(n_values):
        for j, method in enumerate(methods):
            coverages = df_classif[(df_classif["n"]==n) & (df_classif["method"]==method)]["coverage"]
            pos = (len(methods)+2)*i+j
            ax.boxplot(coverages, positions=[pos], widths=0.8, patch_artist=True,
                        boxprops=dict(facecolor=method_colors[method]),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=method_colors[method],
                                            markeredgewidth=1.5, markeredgecolor="black"),
                        medianprops=dict(color="white"), sym=method_colors[method])

            count_below = (coverages < 0.5).sum()
            if count_below > 0:
                arrow_legend_down = True
                triangle = mpatches.Polygon(
                    [[pos, 0.49], [pos - 0.25, 0.5], [pos + 0.25, 0.5]],
                    closed=True,
                    facecolor=method_colors[method],
                    edgecolor='black',
                    alpha=0.8,
                    zorder=10
                )
                ax.add_patch(triangle)
                ax.text(pos, 0.502, f'{count_below}',
                        ha='center', fontsize=9, color=method_colors[method], 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))

            count_above = (coverages > 0.995).sum()
            if count_above > 0:
                arrow_legend_up = True
                triangle = mpatches.Polygon(
                    [[pos, 1.01], [pos - 0.25, 1], [pos + 0.25, 1]],
                    closed=True,
                    facecolor=method_colors[method],
                    edgecolor='black',
                    alpha=0.8,
                    zorder=10
                )
                ax.add_patch(triangle)
                ax.text(pos, 1.012, f'{count_above}',
                        ha='center', fontsize=9, color=method_colors[method],
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))

    legend_handles_ax1 = [
        mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods
    ]
    if arrow_legend_down:
        arrow_handle = Line2D([0], [0], color='black', marker='v', linestyle='None', alpha=0.5, markersize=21,
                                markeredgecolor="black", markeredgewidth=1.5, label='Count below \n50% coverage')
        legend_handles_ax1.append(arrow_handle)
    if arrow_legend_up:
        circle_handle = Line2D([0], [0], color='black', marker='^', linestyle='None', alpha=0.5, markersize=21,
                                markeredgecolor="black", markeredgewidth=1.5, label='Count above \n99.5% coverage')
        legend_handles_ax1.append(circle_handle)

    ax.set_ylabel("Coverage (%)", weight="bold")
    ax.set_xlabel("Sample size", weight="bold")
    ax.set_ylim(0.49, 1.02)
    ax.set_title(f"{metric_labels[metrics_classif[0]]}".replace("_", " "), weight="bold")
    ax.set_xticks([(len(methods)+2)*i+2 for i in range(len(n_values))])
    ax.set_xticklabels([f"{int(n)}" for n in np.sort(n_values)])
    ax.set_yticks(np.arange(0.5, 1.01, 0.05))
    ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
    ax.legend(handles=legend_handles_ax1, loc="lower right", bbox_to_anchor=(1.2, 0.5))
    ax.set_xlim(-1, (len(methods)+2)*len(n_values))

    ####### Second plot : mean of DSC ######

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    # Remove matching rows
    preferred_order = ["basic", "bca", "percentile"]
    methods = df_segm["method"].unique()
    preferred = [m for m in preferred_order if m in methods]
    others = sorted([m for m in methods if m not in preferred_order])

    methods = preferred+others
    ax = axs[1]
    ax.hlines(np.arange(0, 1.01, 0.05), 0, (len(methods)+2)*len(df_segm["n"].unique()-2), colors="black", linewidths=0.6, linestyles=(0, (5,10)))

    arrow_legend_down = False
    arrow_legend_up = False
    n_values = [n for n in np.sort(df_segm["n"].unique()) if n<=100]
    for i, n in enumerate(n_values):
        for j, method in enumerate(methods):
            coverages = df_segm[(df_segm["n"]==n) & (df_segm["method"]==method)]["coverage"]
            pos = (len(methods)+2)*i+j
            ax.boxplot(coverages, positions=[pos], widths=0.8, patch_artist=True,
                        boxprops=dict(facecolor=method_colors[method]),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=method_colors[method],
                                            markeredgewidth=1.5, markeredgecolor="black"),
                        medianprops=dict(color="white"), sym=method_colors[method])

            count_below = (coverages < 0.5).sum()
            if count_below > 0:
                arrow_legend_down = True
                triangle = mpatches.Polygon(
                    [[pos, 0.49], [pos - 0.25, 0.5], [pos + 0.25, 0.5]],
                    closed=True,
                    facecolor=method_colors[method],
                    edgecolor='black',
                    alpha=0.8,
                    zorder=10
                )
                ax.add_patch(triangle)
                ax.text(pos, 0.502, f'{count_below}',
                        ha='center', fontsize=9, color=method_colors[method], 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))

            count_above = (coverages > 0.995).sum()
            if count_above > 0:
                arrow_legend_up = True
                triangle = mpatches.Polygon(
                    [[pos, 1.01], [pos - 0.25, 1], [pos + 0.25, 1]],
                    closed=True,
                    facecolor=method_colors[method],
                    edgecolor='black',
                    alpha=0.8,
                    zorder=10
                )
                ax.add_patch(triangle)
                ax.text(pos, 1.012, f'{count_above}',
                        ha='center', fontsize=9, color=method_colors[method],
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))

    legend_handles_ax1 = [
        mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods
    ]
    if arrow_legend_down:
        arrow_handle = Line2D([0], [0], color='black', marker='v', linestyle='None', alpha=0.5, markersize=21,
                                markeredgecolor="black", markeredgewidth=1.5, label='Count below \n50% coverage')
        legend_handles_ax1.append(arrow_handle)
    if arrow_legend_up:
        circle_handle = Line2D([0], [0], color='black', marker='^', linestyle='None', alpha=0.5, markersize=21,
                                markeredgecolor="black", markeredgewidth=1.5, label='Count above \n99.5% coverage')
        legend_handles_ax1.append(circle_handle)

    ax.set_ylabel("Coverage (%)", weight="bold")
    ax.set_xlabel("Sample size", weight="bold")
    ax.set_ylim(0.49, 1.02)
    ax.set_title(f"Metric : {metric_labels[metrics_segm[0]]}, Stat: Mean".replace("_", " "), weight="bold")
    ax.set_xticks([(len(methods)+2)*i+2 for i in range(len(n_values))])
    ax.set_xticklabels([f"{int(n)}" for n in np.sort(n_values)])
    ax.set_yticks(np.arange(0.5, 1.01, 0.05))
    ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
    ax.legend(handles=legend_handles_ax1, loc="lower right", bbox_to_anchor=(1.2, 0.5))
    ax.set_xlim(-1, (len(methods)+2)*len(n_values))

    for ax, letter in zip(axs, ['A', 'B']):
        ax.text(0.98, 0.02, letter, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='bottom', ha='right')

    plt.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.4)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5 parametric methods on very small samples.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig5_param_small_samples.pdf")

    # Call your plotting function
    plot_fig5_param_small_samples(root_folder, output_path)

if __name__ == "__main__":
    main()