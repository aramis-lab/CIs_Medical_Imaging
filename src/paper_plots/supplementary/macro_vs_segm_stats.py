import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse

from ..plot_utils import metric_labels, stat_labels
from ..df_loaders import extract_df_classif_cov, extract_df_segm_cov, extract_df_segm_width, extract_df_classif_width


def plot_macro_vs_segm_stats(root_folder:str, output_path:str):

        folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
        file_prefix_segm = "aggregated_results"
        metrics_segm = ["boundary_iou", "cldice", "dsc", "iou", "nsd"]
        stats_segm = ["mean", "median", "trimmed_mean", "std", "iqr_length"]

        df_segm_cov = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
        df_segm_width = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

        folder_path_classif = os.path.join(root_folder, "results_metrics_classif_macro")
        file_prefix_classif = "aggregated_results"
        metrics_classif = ["auc", "ap", "balanced_accuracy", "f1_score"]

        df_classif_cov = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)
        df_classif_width = extract_df_classif_width(folder_path_classif, file_prefix_classif, metrics_classif)

        # Color palettes
        palette_segm = sns.color_palette("Blues", len(metrics_segm)+4)
        color_dict_segm = dict(zip(metrics_segm, palette_segm))
        metrics_classif = ["balanced_accuracy", "ap", "auc", "f1_score"]
        palette_classif = sns.color_palette("Reds", len(metrics_classif))
        color_dict_classif = dict(zip(metrics_classif, palette_classif))

        df_classif_perc_cov = df_classif_cov[(df_classif_cov['n'] <= 250) & (df_classif_cov['method'] == 'percentile')]
        df_segm_perc_cov = df_segm_cov[(df_segm_cov['n'] <= 250) & (df_segm_cov['method'] == 'percentile')]
        df_segm_perc_width = df_segm_width[(df_segm_width['n'] <= 250) & (df_segm_width['method'] == 'percentile')]
        df_classif_perc_width = df_classif_width[(df_classif_width['n'] <= 250) & (df_classif_width['method'] == 'percentile')]
        n_stats = len(stats_segm)
        fig, axs = plt.subplots(n_stats, 2, figsize=(36, 15 * n_stats))
        if n_stats == 1:
                axs = np.array([axs])  # keep consistent 2D structure

        for row, stat in enumerate(stats_segm):

                # ============================================================
                # COVERAGE (left column)
                # ============================================================
                ax = axs[row, 0]

                # --- CLASSIFICATION ---
                medians = df_classif_perc_cov.groupby(['n', 'stat'])['value'].median().reset_index()
                q1 = df_classif_perc_cov.groupby(['n', 'stat'])['value'].quantile(0.25).reset_index()
                q3 = df_classif_perc_cov.groupby(['n', 'stat'])['value'].quantile(0.75).reset_index()
                df_plot = (
                        medians.merge(q1, on=['n', 'stat'], suffixes=('_median', '_q1'))
                        .merge(q3, on=['n', 'stat'])
                        .rename(columns={'value': 'value_q3'})
                )

                for stat_classif in df_plot['stat'].unique():
                        df_stat = df_plot[df_plot['stat'] == stat_classif]
                        ax.plot(df_stat['n'], df_stat['value_median'], marker='o',
                                color=color_dict_classif[stat_classif], linewidth=2,
                                label=metric_labels[stat_classif])
                        ax.plot(df_stat['n'], df_stat['value_q1'], linestyle="--",
                                color=color_dict_classif[stat_classif], linewidth=1)
                        ax.plot(df_stat['n'], df_stat['value_q3'], linestyle="--",
                                color=color_dict_classif[stat_classif], linewidth=1)
                        ax.fill_between(df_stat['n'], df_stat['value_q1'], df_stat['value_q3'],
                                        alpha=0.2, color=color_dict_classif[stat_classif])

                # --- SEGMENTATION ---
                df_segm_stat = df_segm_perc_cov[df_segm_perc_cov['stat'] == stat]
                medians = df_segm_stat.groupby(['n', 'metric'])['coverage'].median().reset_index()
                q1 = df_segm_stat.groupby(['n', 'metric'])['coverage'].quantile(0.25).reset_index()
                q3 = df_segm_stat.groupby(['n', 'metric'])['coverage'].quantile(0.75).reset_index()
                df_plot = (
                        medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
                        .merge(q3, on=['n', 'metric'])
                        .rename(columns={'coverage': 'coverage_q3'})
                )

                for metric in df_plot['metric'].unique():
                        df_metric = df_plot[df_plot['metric'] == metric]
                        ax.plot(df_metric['n'], df_metric['coverage_median'], marker='o',
                                color=color_dict_segm[metric], linewidth=2, label=metric_labels[metric])
                        ax.plot(df_metric['n'], df_metric['coverage_q1'], linestyle="--",
                                color=color_dict_segm[metric], linewidth=1)
                        ax.plot(df_metric['n'], df_metric['coverage_q3'], linestyle="--",
                                color=color_dict_segm[metric], linewidth=1)
                        ax.fill_between(df_metric['n'], df_metric['coverage_q1'], df_metric['coverage_q3'],
                                        alpha=0.6, color=color_dict_segm[metric])

                ax.set_title(f"Coverage — {stat_labels[stat]}", fontsize=44, weight="bold")
                ax.set_xlabel("Sample size", fontsize=32)
                ax.set_ylabel("Coverage", fontsize=32)
                ax.grid(True, axis="y")
                ax.set_ylim(None, 1.01)
                # ===== SEPARATE LEGENDS =====
                handles_all, labels_all = ax.get_legend_handles_labels()

                # classification first
                handles_classif = [h for h, lab in zip(handles_all, labels_all) if lab in metric_labels
                                and lab in metrics_classif]
                labels_classif = [metric_labels[lab] for lab in labels_all if lab in metrics_classif]

                # segmentation next
                handles_segm = [h for h, lab in zip(handles_all, labels_all) if lab in metric_labels
                                and lab in metrics_segm]
                labels_segm = [metric_labels[lab] for lab in labels_all if lab in metrics_segm]

                # draw them in different corners
                leg1 = ax.legend(handles_classif, labels_classif, title="Classification metrics",
                                fontsize=24, title_fontsize=28, loc="upper right")
                ax.add_artist(leg1)

                ax.legend(handles_segm, labels_segm, title="Segmentation metrics",
                        fontsize=24, title_fontsize=28, loc="center right")

                # ============================================================
                # WIDTH (right column)
                # ============================================================
                ax = axs[row, 1]

                # --- CLASSIFICATION ---
                medians = df_classif_perc_width.groupby(['n', 'stat'])['width'].median().reset_index()
                q1 = df_classif_perc_width.groupby(['n', 'stat'])['width'].quantile(0.25).reset_index()
                q3 = df_classif_perc_width.groupby(['n', 'stat'])['width'].quantile(0.75).reset_index()
                df_plot = (
                        medians.merge(q1, on=['n', 'stat'], suffixes=('_median', '_q1'))
                        .merge(q3, on=['n', 'stat'])
                        .rename(columns={'width': 'width_q3'})
                )

                for stat_classif in df_plot['stat'].unique():
                        df_stat = df_plot[df_plot['stat'] == stat_classif]
                        ax.plot(df_stat['n'], df_stat['width_median'], marker='o',
                                color=color_dict_classif[stat_classif], linewidth=2,
                                label=metric_labels[stat_classif])
                        ax.plot(df_stat['n'], df_stat['width_q1'], linestyle="--",
                                color=color_dict_classif[stat_classif], linewidth=1)
                        ax.plot(df_stat['n'], df_stat['width_q3'], linestyle="--",
                                color=color_dict_classif[stat_classif], linewidth=1)
                        ax.fill_between(df_stat['n'], df_stat['width_q1'], df_stat['width_q3'],
                                        alpha=0.2, color=color_dict_classif[stat_classif])

                # --- SEGMENTATION ---
                df_segm_mean = df_segm_perc_width[df_segm_perc_width['stat'] == stat]
                medians = df_segm_mean.groupby(['n', 'metric'])['width'].median().reset_index()
                q1 = df_segm_mean.groupby(['n', 'metric'])['width'].quantile(0.25).reset_index()
                q3 = df_segm_mean.groupby(['n', 'metric'])['width'].quantile(0.75).reset_index()
                df_plot = (
                        medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
                        .merge(q3, on=['n', 'metric'])
                        .rename(columns={'width': 'width_q3'})
                )

                for metric in df_plot['metric'].unique():
                        df_metric = df_plot[df_plot['metric'] == metric]
                        ax.plot(df_metric['n'], df_metric['width_median'], marker='o',
                                color=color_dict_segm[metric], linewidth=2, label=metric_labels[metric])
                        ax.plot(df_metric['n'], df_metric['width_q1'], linestyle="--",
                                color=color_dict_segm[metric], linewidth=1)
                        ax.plot(df_metric['n'], df_metric['width_q3'], linestyle="--",
                                color=color_dict_segm[metric], linewidth=1)
                        ax.fill_between(df_metric['n'], df_metric['width_q1'], df_metric['width_q3'],
                                        alpha=0.7, color=color_dict_segm[metric])

                ax.set_title(f"Width — {stat_labels[stat]}", fontsize=44, weight="bold")
                ax.set_xlabel("Sample size", fontsize=32)
                ax.set_ylabel("Width", fontsize=32)
                ax.grid(True, axis="y")

                # ===== SEPARATE LEGENDS =====
                handles_all, labels_all = ax.get_legend_handles_labels()

                # classification first
                handles_classif = [h for h, lab in zip(handles_all, labels_all) if lab in metric_labels
                                and lab in metrics_classif]
                labels_classif = [metric_labels[lab] for lab in labels_all if lab in metrics_classif]

                # segmentation next
                handles_segm = [h for h, lab in zip(handles_all, labels_all) if lab in metric_labels
                                and lab in metrics_segm]
                labels_segm = [metric_labels[lab] for lab in labels_all if lab in metrics_segm]

                # draw them in different corners
                leg1 = ax.legend(handles_classif, labels_classif, title="Classification metrics",
                                fontsize=24, title_fontsize=28, loc="upper right")
                ax.add_artist(leg1)

                ax.legend(handles_segm, labels_segm, title="Segmentation metrics",
                        fontsize=24, title_fontsize=28, loc="center right")

        # Global layout
        plt.suptitle("CI Comparison: Classification Macro vs Segmentation", fontsize=30, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
        plt.savefig(output_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure classification macro vs each segmentation summary statistic.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/macro_vs_segm_stats.pdf")

    # Call your plotting function
    plot_macro_vs_segm_stats(root_folder, output_path)

if __name__ == "__main__":
    main()
