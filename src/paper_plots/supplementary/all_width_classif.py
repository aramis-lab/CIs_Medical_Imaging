from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from ..df_loaders import extract_df_classif_width
from ..plot_utils import method_labels, method_colors

def plot_all_width_classif(root_folder: str, output_path: str):
    
    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro = ["accuracy", "auc", "f1_score", "ap"]

    df_micro = extract_df_classif_width(folder_path_micro, file_prefix_micro, metrics_micro)
    df_mcc = extract_df_classif_width(folder_path_micro, file_prefix_micro, ["mcc"])

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro = ["accuracy", "auc", "f1_score", "ap"]

    df_macro = extract_df_classif_width(folder_path_macro, file_prefix_macro, metrics_macro)

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
        fig, axs = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 12))
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
            
            for i, n in enumerate(np.sort(df_all["n"].unique())):
                for j, method in enumerate(methods):
                    widths = df_all[(df_all["n"]==n) & (df_all["method"]==method)]["width"]
                    pos = (len(methods)+2)*i+j
                    ax.boxplot(widths, positions=[pos], widths=0.8, patch_artist=True,
                                boxprops=dict(facecolor=method_colors[method]),
                                flierprops=dict(marker='o', markersize=3, markerfacecolor=method_colors[method],
                                                markeredgewidth=1.5, markeredgecolor="black"),
                                medianprops=dict(color="white"), sym=method_colors[method])
            
            legend_handles = [mpatches.Patch(color=method_colors[method], label=method_labels[method]) for method in methods]
            
            ax.set_xlabel("Sample size", weight="bold")
            ax.set_ylim(0.49, 1.02)
            ax.set_title(f"{metric.upper()}".replace("_", " "), weight="bold")
            ax.set_xticks([(len(methods)+2)*i+2 for i in range(len(df_all["n"].unique()))])
            ax.set_xticklabels([f"{int(n)}" for n in np.sort(df_all["n"].unique())])
            ax.set_yticks(np.arange(0.5, 1.01, 0.05))
            ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
            ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.35, 0.5))
            ax.set_xlim(-1, (len(methods)+2)*len(df_all["n"].unique()))
        
        fig.suptitle(f"{fig_name}", fontsize=32, weight="bold", y=0.98)
        plt.tight_layout()
        output_file = output_path.replace(".pdf", f"_{fig_name.lower()}.pdf")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        plt.savefig(output_file)
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Supp Figure width of all methods and metrics for classification.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/all_width_classif.pdf")

    plot_all_width_classif(root_folder, output_path)

if __name__ == "__main__":
    main()