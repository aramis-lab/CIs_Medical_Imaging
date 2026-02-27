import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

from ..df_loaders import extract_df_segm_cov
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels

def plot_bca_fail(root_folder: str, output_path: str):

    plt.rcdefaults()

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats_segm = [ "median"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    _, axs = plt.subplots(3, 3, figsize=(25, 15))
    axs = axs.flatten()

    for i, metric in enumerate(metrics_segm):

        df_all_metric=df_segm[df_segm['metric']==metric]
        data_method=df_all_metric[df_all_metric['method'].isin(['bca', 'percentile']) & df_all_metric['stat'].isin(['mean', 'median'])]
        ax=axs[i]
        medians = data_method.groupby(['n', 'method', 'stat'])['coverage'].median().reset_index()
        q1 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.25).reset_index()
        q3 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.75).reset_index()
        df_plot = medians.merge(q1, on=['n', 'method', 'stat'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method', 'stat'])
        df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)
        
        for (method, stat), df_group in df_plot.groupby(['method', 'stat']):
            linestyle = '-' if stat == 'median' else '-'
            ax.plot(
                df_group['n'], df_group['coverage_median'],
                label=f"{method_labels[method]}",
                color=method_colors[method],
                marker='o',
                linestyle=linestyle,
                linewidth=4,
                markersize=10
            )
            ax.fill_between(df_group['n'], df_group['coverage_q1'], df_group['coverage_q3'],
            color=method_colors[method],
            alpha=0.2)
    
        ax.set_title(f'Metric: {metric_labels[metric]}, Summary statistic: {stat_labels[stat]}', weight='bold', fontsize=22)
        ax.set_xlabel('Sample size',weight='bold', fontsize=20)
        ax.set_ylabel('Coverage (%)', weight='bold', fontsize=20)
        ax.set_yticks(np.arange(0.5, 1.01, 0.05))
        ax.set_yticklabels((np.arange(0.5, 1.01, 0.05)*100).astype(int))
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_ylim(0.8, 1.01)
        ax.grid(True, axis='y', linestyle=(0, (5,10)), color='black', linewidth=0.6)

        ax.legend(fontsize=20)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Supp Figure failure of BCa.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/bca_fail.pdf")

    # Call your plotting function
    plot_bca_fail(root_folder, output_path)

if __name__ == "__main__":
    main()