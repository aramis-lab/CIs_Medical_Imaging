import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

from ..df_loaders import extract_df_segm_cov
from ..plot_utils import metric_labels, stat_labels


def plot_central_vs_dispersion(root_folder: str, output_path:str):
    
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats_segm = ["mean", "std"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm["method"] == "percentile"]

    fig, axes = plt.subplots(3, 3, figsize=(21, 18), sharex=False)
    axes = axes.flatten()  # flatten to 1D for easy indexing

    for i, metric in enumerate(metrics_segm):
        df_all_metric = df_segm[df_segm['metric'] == metric]
        data_method = df_all_metric[df_all_metric['method'] == 'percentile']

        ax = axes[i]

        # central plot
        for stat, df_stat in data_method.groupby('stat'):
            grouped = df_stat.groupby('n')
            n_vals = grouped['coverage'].median().index.values
            medians = grouped['coverage'].median().values
            q1 = grouped['coverage'].quantile(0.25).values
            q3 = grouped['coverage'].quantile(0.75).values

            ax.plot(n_vals, medians, marker='o', label=stat_labels[stat],
                    linewidth=2, markersize=8)
            ax.fill_between(n_vals, q1, q3, alpha=0.2)

        ax.set_title(f'Metric: {metric_labels[metric]}', weight='bold', fontsize=18)
        ax.set_xlabel('Sample size', weight='bold', fontsize=14)
        ax.set_ylabel('Coverage (%)', weight='bold', fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_yticks(np.arange(0.5, 1.01, 0.05))
        ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
        ax.set_ylim(0.49, 1.01)
        ax.grid(True, axis='y', linestyle=(0, (5,10)), color='black', linewidth=0.6)
        ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot central vs dispersion statistics for medical imaging CIs.")
    parser.add_argument('--root_folder', type=str, required=True, help='Root folder containing results.')
    parser.add_argument('--output_path', type=str, required=False, help='Output path for the plot.')
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/central_vs_dispersion.pdf")

    plot_central_vs_dispersion(root_folder, output_path)

if __name__ == "__main__":
    main()