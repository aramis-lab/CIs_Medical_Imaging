import os
import matplotlib.pyplot as plt
from ..df_loaders import extract_df_segm_cov
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels


def plot_fig4_bca(root_folder: str, output_path: str):

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["mean", "median"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    data_method=df_segm[df_segm['method'].isin(['bca', 'percentile']) & df_segm['stat'].isin(['mean', 'median'])]
    medians = data_method.groupby(['n', 'method', 'stat'])['coverage'].median().reset_index()
    q1 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.25).reset_index()
    q3 = data_method.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.75).reset_index()
    df_plot = medians.merge(q1, on=['n', 'method', 'stat'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method', 'stat'])
    df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)
    
    for (method, stat), df_group in df_plot.groupby(['method', 'stat']):
        linestyle = '--' if stat == 'median' else '-'
        plt.plot(
            df_group['n'], df_group['coverage_median'],
            label=f"{method_labels[method]} ({stat_labels[stat]})",
            color=method_colors[method],
            marker='o',
            linestyle=linestyle,
            linewidth=2
        )
    
    plt.title(f'Metric: {metric_labels[metrics_segm[0]]}', weight='bold')
    plt.xlabel('Sample size',weight='bold', fontsize=16)
    plt.ylabel('Coverage', weight='bold', fontsize=16)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.ylim(None, 1.01)
    plt.grid(True, axis='y')

    plt.legend(fontsize= 20)
    plt.legend(fontsize= 20)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)

if __name__ == "__main__":
    root_folder = "C:/Users/Charles/Desktop/ICM"
    output_path = os.path.join(root_folder, 'clean_figs/main/fig4_bca.pdf')
    plot_fig4_bca(root_folder, output_path)