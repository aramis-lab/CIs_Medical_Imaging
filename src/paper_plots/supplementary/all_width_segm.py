import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from ..df_loaders import extract_df_segm_width
from ..plot_utils import method_labels, method_colors

def plot_all_width_segm(root_folder: str, output_path: str):
    
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats_segm = ["mean", "median", "trimmed_mean", "std", "iqr_length"]

    df_segm = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

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

    fig, axs = plt.subplots(len(stats_segm), len(metrics_segm), figsize=(18*len(metrics_segm), 15*len(stats_segm)))

    preferred_order = ["basic", "bca", "percentile"]
    for idx1, summary_statistic in enumerate(stats_segm):
        for idx2, metric in enumerate(metrics_segm):
            ax = axs[idx1][idx2]
            df_all = df_segm[(df_segm["metric"]==metric) & (df_segm["stat"]==summary_statistic)]
            methods = sorted(df_all["method"].unique())
            preferred = [m for m in preferred_order if m in methods]
            others = [m for m in methods if m not in preferred_order]

            methods = preferred+others
            max_width = df_all["width"].max()
            for i, n in enumerate(np.sort(df_all["n"].unique())):
                for j, method in enumerate(methods):
                    widths = df_all[(df_all["n"]==n) & (df_all["method"]==method)]["width"]
                    widths = widths.fillna(0.0) # NaN widths correspond to degenerate CIs with width 0
                    pos = (len(methods)+2)*i+j
                    if summary_statistic != "mean":
                        pos = pos + 1
                    ax.boxplot(widths, positions=[pos], widths=0.8, patch_artist=True,
                                boxprops=dict(facecolor=method_colors[method]),
                                flierprops=dict(marker='o', markersize=3, markerfacecolor=method_colors[method],
                                                markeredgewidth=1.5, markeredgecolor="black"),
                                medianprops=dict(color="white"), sym=method_colors[method])

            legend_handles_ax1 = [
                mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods
            ]

            ax.set_ylabel(summary_statistic.upper(), weight="bold")
            ax.set_xlabel("Sample size", weight="bold")
            ax.set_title(f"{metric.upper()}".replace("_", " "), weight="bold")
            ax.set_xticks([(len(methods)+2)*i+2 for i in range(len(df_all["n"].unique()))])
            ax.set_xticklabels([f"{int(n)}" for n in np.sort(df_all["n"].unique())])
            if metric in ["hd", "hd_perc", "masd", "assd"]:
                ax.set_yticks(np.arange(0, max(1,max_width)*1.01, step=max(max_width*1.01//10, 0.1)))
            else:
                ax.set_yticks(np.arange(0, max(1,max_width)*1.01, step=0.1))
            ax.grid(which='major', axis='y', linestyle=(0, (5,10)), color='black', linewidth=0.6)
            ax.legend(handles=legend_handles_ax1, loc="lower right", bbox_to_anchor=(1.35, 0.5))
            ax.set_xlim(-1, (len(methods)+2)*len(df_all["n"].unique()))
            ax.set_ylim(0.0, max(1, max_width)*1.01)

    plt.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.4)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Supp Figure width of all methods and metrics for segmentation.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/all_width_segm.pdf")

    plot_all_width_segm(root_folder, output_path)

if __name__ == "__main__":
    main()