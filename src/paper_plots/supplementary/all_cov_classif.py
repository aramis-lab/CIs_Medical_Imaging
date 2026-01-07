from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from ..df_loaders import extract_df_classif_cov
from ..plot_utils import method_labels, method_colors

def plot_all_cov_classif(root_folder: str, output_path: str):
    
    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro = ["accuracy", "auc", "f1_score", "ap"]

    df_micro = extract_df_classif_cov(folder_path_micro, file_prefix_micro, metrics_micro)
    df_mcc = extract_df_classif_cov(folder_path_micro, file_prefix_micro, ["mcc"])

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro = ["balanced_accuracy", "auc", "f1_score", "ap"]

    df_macro = extract_df_classif_cov(folder_path_macro, file_prefix_macro, metrics_macro)

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

    # Define metrics for each figure
    metrics_list = [
        ("Micro", metrics_micro, df_micro),
        ("Macro", metrics_macro, df_macro),
        ("MCC", ["mcc"], df_mcc)
    ]
    
    for fig_name, metrics, df in metrics_list:
        fig, axs = plt.subplots(len(metrics), 1, figsize=(21,15*len(metrics)))
        if len(metrics) == 1:
            axs = [axs]
        
        preferred_order = ["basic", "bca", "percentile"]
        
        for idx, metric in enumerate(metrics):
            ax = axs[idx]
            df_all = df[df["metric"] == metric]
            methods = sorted(df_all["method"].unique())
            preferred = [m for m in preferred_order if m in methods]
            others = [m for m in methods if m not in preferred_order]
            methods = preferred + others
            
            ax.hlines(np.arange(0, 1.01, 0.05), 0, (len(methods)+2)*len(df_all["n"].unique()-2), colors="black", linewidths=0.6, linestyles=(0, (5,10)))
            
            arrow_legend_down = False
            arrow_legend_up = False
            
            for i, n in enumerate(np.sort(df_all["n"].unique())):
                for j, method in enumerate(methods):
                    coverages = df_all[(df_all["n"]==n) & (df_all["method"]==method)]["coverage"]
                    pos = (len(methods)+2)*i+j
                    if metric != "accuracy":
                        pos = pos + 1
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
                            closed=True, facecolor=method_colors[method], edgecolor='black', alpha=0.8, zorder=10
                        )
                        ax.add_patch(triangle)
                        ax.text(pos, 0.502, f'{count_below}', ha='center', fontsize=9, color=method_colors[method], 
                                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
                    
                    count_above = (coverages > 0.995).sum()
                    if count_above > 0:
                        arrow_legend_up = True
                        triangle = mpatches.Polygon(
                            [[pos, 1.01], [pos - 0.25, 1], [pos + 0.25, 1]],
                            closed=True, facecolor=method_colors[method], edgecolor='black', alpha=0.8, zorder=10
                        )
                        ax.add_patch(triangle)
                        ax.text(pos, 1.012, f'{count_above}', ha='center', fontsize=9, color=method_colors[method],
                                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
            
            legend_handles = [mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods]
            if arrow_legend_down:
                legend_handles.append(Line2D([0], [0], color='black', marker='v', linestyle='None', alpha=0.5, markersize=21,
                                            markeredgecolor="black", markeredgewidth=1.5, label='Count below\n 50% coverage'))
            if arrow_legend_up:
                legend_handles.append(Line2D([0], [0], color='black', marker='^', linestyle='None', alpha=0.5, markersize=21,
                                            markeredgecolor="black", markeredgewidth=1.5, label='Count above\n 99.5% coverage'))
            
            ax.set_xlabel("Sample size", weight="bold")
            ax.set_ylabel("Coverage (%)", weight="bold")
            ax.set_ylim(0.49, 1.02)
            ax.set_title(f"{metric.upper()}".replace("_", " "), weight="bold")
            ax.set_xticks([(len(methods)+2)*i+2 for i in range(len(df_all["n"].unique()))])
            ax.set_xticklabels([f"{int(n)}" for n in np.sort(df_all["n"].unique())])
            ax.set_yticks(np.arange(0.5, 1.01, 0.05))
            ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
            ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.3, 0.5))
            ax.set_xlim(-1, (len(methods)+2)*len(df_all["n"].unique()))
        
        plt.tight_layout()
        output_file = output_path.replace(".pdf", f"_{fig_name.lower()}.pdf")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        plt.savefig(output_file)
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Supp Figure coverage of all methods and metrics for classification.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/all_cov_classif.pdf")

    plot_all_cov_classif(root_folder, output_path)

if __name__ == "__main__":
    main()