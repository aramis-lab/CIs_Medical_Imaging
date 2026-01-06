import os
import matplotlib.pyplot as plt
import argparse

from ..df_loaders import extract_df_segm_cov
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels


def plot_fig4_bca(root_folder: str, output_path: str):

    plt.rcdefaults()

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["median"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    data_method=df_segm[df_segm['method'].isin(['bca', 'percentile']) & df_segm['stat'].isin(['median'])]
    medians = data_method.groupby(['n', 'method', 'stat'])['coverage'].median().reset_index()
    q1 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.25).reset_index()
    q3 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.75).reset_index()
    df_plot = medians.merge(q1, on=['n', 'method', 'stat'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method', 'stat'])
    df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)

    plt.figure(figsize=(20, 16))
    
    for (method, stat), df_group in df_plot.groupby(['method', 'stat']):
        linestyle = '-' if stat == 'median' else '-'
        plt.plot(
            df_group['n'], df_group['coverage_median'],
            label=f"{method_labels[method]}",
            color=method_colors[method],
            marker='o',
            linestyle=linestyle,
            linewidth=4,
            markersize=10
        )
        plt.fill_between(
            df_group['n'],
            df_group['coverage_q1'],
            df_group['coverage_q3'],
            color=method_colors[method],
            alpha=0.2
        )
    
    plt.title(f'Metric: {metric_labels[metrics_segm[0]]}, Summary statistic: Median', weight='bold', fontsize=40)
    plt.xlabel('Sample size',weight='bold', fontsize=32)
    plt.ylabel('Coverage (%)', weight='bold', fontsize=32)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    plt.tick_params(axis='y', labelsize=28)
    plt.tick_params(axis='x', labelsize=28)
    plt.ylim(0.79, 1.01)
    plt.grid(True, axis='y')

    plt.legend(fontsize= 32)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4 failure of BCa.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig4_bca.pdf")

    # Call your plotting function
    plot_fig4_bca(root_folder, output_path)

if __name__ == "__main__":
    main()