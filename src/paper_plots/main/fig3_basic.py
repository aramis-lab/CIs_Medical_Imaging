import os
import matplotlib.pyplot as plt
import argparse

from ..df_loaders import extract_df_segm_cov, extract_df_classif_cov
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels

def plot_fig3_basic(root_folder: str, output_path: str):

    plt.rcdefaults()
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    folder_path_classif = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_classif = "aggregated_results"
    metrics_classif = ["accuracy"]

    df_classif = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)

    fig, axs = plt.subplots(1, 2, figsize=(30, 15), sharey=True)

    ax = axs[0]
    metric = "dsc"
    stat = "mean"
    df_stat=df_segm[df_segm['method'].isin(['bca', 'percentile', 'basic'])]

    medians = df_stat.groupby(['n', 'method', 'stat'])['coverage'].median().reset_index()
    q1 = df_stat.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.25).reset_index()
    q3 = df_stat.groupby(['n', 'method', 'stat'])['coverage'].quantile(0.75).reset_index()
    df_plot = medians.merge(q1, on=['n', 'method', 'stat'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method', 'stat'])
    df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)

    for method, df_group in df_plot.groupby("method"):
        if method not in ['bca', 'percentile', 'basic']:
            pass
        ax.plot(
            df_group['n'], df_group['coverage_median'],
            label=f"{method_labels[method]}",
            color=method_colors[method],
            marker='o',
            linewidth=2
        )
        ax.fill_between(df_group['n'], df_group['coverage_q1'], df_group['coverage_q3'],
            color=method_colors[method],
            alpha=0.2)

    ax.set_title(f'Metric: {metric_labels[metric]}, Stat: {stat_labels[stat]}', weight='bold', fontsize=40)
    ax.set_xlabel('Sample size',weight='bold', fontsize=32)
    ax.set_ylabel('Coverage', weight='bold', fontsize=32)
    ax.tick_params(axis='y', labelsize=28)
    ax.tick_params(axis='x', labelsize=28)
    ax.set_ylim(0.8,1)

    ax.grid(True, axis='y')

    ax.legend(fontsize= 32)

    ax= axs[1]

    metric = "accuracy"

    df_metric=df_classif[df_classif['method'].isin(['bca', 'percentile', 'basic'])]

    medians = df_metric.groupby(['n', 'method'])['coverage'].median().reset_index()
    q1 = df_metric.groupby(['n', 'method'])['coverage'].quantile(0.25).reset_index()
    q3 = df_metric.groupby(['n', 'method'])['coverage'].quantile(0.75).reset_index()
    df_plot = medians.merge(q1, on=['n', 'method'], suffixes=('_median', '_q1')).merge(q3, on=['n', 'method'])
    df_plot.rename(columns={'coverage': 'coverage_q3'}, inplace=True)

    for method, df_group in df_plot.groupby("method"):
        if method not in ['bca', 'percentile', 'basic']:
            pass
        ax.plot(
            df_group['n'], df_group['coverage_median'],
            label=f"{method_labels[method]}",
            color=method_colors[method],
            marker='o',
            linewidth=2
        )
        ax.fill_between(df_group['n'], df_group['coverage_q1'], df_group['coverage_q3'],
            color=method_colors[method],
            alpha=0.2)
        # ax.hline(0.95, alpha=0.7, linestyle="--")

    ax.set_title(f'Metric: {metric_labels[metric]}', weight='bold', fontsize=40)
    ax.set_xlabel('Sample size',weight='bold', fontsize=32)
    ax.set_ylabel('Coverage', weight='bold', fontsize=32)
    ax.tick_params(axis='y', labelsize=28)
    ax.tick_params(axis='x', labelsize=28)
    ax.set_ylim(0.8,1)
    ax.grid(True, axis='y')

    ax.legend(fontsize= 32)

    for ax, letter in zip(axs, ['A', 'B']):
        ax.text(0.98, 0.02, letter, transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3 basic is the worst bootstrap.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig3_basic.pdf")

    # Call your plotting function
    plot_fig3_basic(root_folder, output_path)

if __name__ == "__main__":
    main()