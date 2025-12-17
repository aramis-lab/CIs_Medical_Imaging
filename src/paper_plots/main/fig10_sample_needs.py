import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import argparse

from ..plot_utils import metric_labels, stat_labels
from ..df_loaders import extract_df_classif_cov, extract_df_segm_cov, extract_df_segm_width, extract_df_classif_width


def plot_fig10_sample_needs(root_folder:str, output_path:str):

    plt.rcdefaults()

    # Define consistent font sizes
    TITLE_FONTSIZE = 40
    LABEL_FONTSIZE = 32
    TICK_FONTSIZE = 28 
    LEGEND_FONTSIZE = 34
    LEGEND_TITLE_FONTSIZE = 36

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["mean"]

    df_segm_cov = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm_width = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    folder_path_classif = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_classif = "aggregated_results"
    metrics_classif = ["auc"]

    df_classif_cov = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)
    df_classif_width = extract_df_classif_width(folder_path_classif, file_prefix_classif, metrics_classif)

    # Color palettes
    palette_segm = sns.color_palette("Blues", len(metrics_segm))
    color_dict_segm = dict(zip(metrics_segm, palette_segm))
    palette_classif = sns.color_palette("Reds", len(metrics_classif))
    color_dict_classif = dict(zip(metrics_classif, palette_classif))

    df_segm_cov_perc = df_segm_cov[(df_segm_cov["method"]=="percentile")]
    df_segm_width_perc = df_segm_width[(df_segm_width["method"]=="percentile")]
    df_classif_cov_perc = df_classif_cov[(df_classif_cov["method"]=="percentile")]
    df_classif_width_perc = df_classif_width[(df_classif_width["method"]=="percentile")]
    
    fig, axs = plt.subplots(2, 2, figsize=(36, 30))

    stat = stats_segm[0]

    # ============================================================
    # COVERAGE (left column)
    # ============================================================
    ax = axs[0,0]

    ax.axhline(y=0.94, color='black', linestyle='--', linewidth=4, label="Target Coverage (94%)")

    # --- CLASSIFICATION ---
    medians = df_classif_cov_perc.groupby(['n', 'metric'])['coverage'].median().reset_index()
    q1 = df_classif_cov_perc.groupby(['n', 'metric'])['coverage'].quantile(0.25).reset_index()
    q3 = df_classif_cov_perc.groupby(['n', 'metric'])['coverage'].quantile(0.75).reset_index()
    df_plot = (
            medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
            .merge(q3, on=['n', 'metric'])
            .rename(columns={'coverage': 'coverage_q3'})
    )

    for classif_metric in df_plot['metric'].unique():
            df_metric = df_plot[df_plot['metric'] == classif_metric]
            ax.plot(df_metric['n'], df_metric['coverage_median'], marker='o',
                    color=color_dict_classif[classif_metric], linewidth=4,
                    markersize=10,
                    label=classif_metric)
            ax.plot(df_metric['n'], df_metric['coverage_q1'], linestyle="--",
                    color=color_dict_classif[classif_metric], linewidth=1)
            ax.plot(df_metric['n'], df_metric['coverage_q3'], linestyle="--",
                    color=color_dict_classif[classif_metric], linewidth=1)
            ax.fill_between(df_metric['n'], df_metric['coverage_q1'], df_metric['coverage_q3'],
                            alpha=0.2, color=color_dict_classif[classif_metric])

    # --- SEGMENTATION ---
    df_segm_stat = df_segm_cov_perc[df_segm_cov_perc['stat'] == stat]
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
                    color=color_dict_segm[metric], linewidth=4,  markersize=10, label=metric)
            ax.plot(df_metric['n'], df_metric['coverage_q1'], linestyle="--",
                    color=color_dict_segm[metric], linewidth=1)
            ax.plot(df_metric['n'], df_metric['coverage_q3'], linestyle="--",
                    color=color_dict_segm[metric], linewidth=1)
            ax.fill_between(df_metric['n'], df_metric['coverage_q1'], df_metric['coverage_q3'],
                            alpha=0.6, color=color_dict_segm[metric])

    ax.set_title(f"Coverage", fontsize=TITLE_FONTSIZE, weight="bold")
    ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax.set_ylim(0.79, 1.01)
    ax.set_xlim(-5, 605)
    # ===== SEPARATE LEGENDS =====
    handles_all, labels_all = ax.get_legend_handles_labels()

    labels_all = [metric_labels[label] if label in metric_labels else label for label in labels_all]

    ax.legend(handles_all, labels_all,
            fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc="upper right")

    # ============================================================
    # WIDTH (right column)
    # ============================================================
    ax = axs[0,1]

    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=4, label="Target Width (0.1)")

    # --- CLASSIFICATION ---
    medians = df_classif_width_perc.groupby(['n', 'metric'])['width'].median().reset_index()
    q1 = df_classif_width_perc.groupby(['n', 'metric'])['width'].quantile(0.25).reset_index()
    q3 = df_classif_width_perc.groupby(['n', 'metric'])['width'].quantile(0.75).reset_index()
    df_plot = (
            medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
            .merge(q3, on=['n', 'metric'])
            .rename(columns={'width': 'width_q3'})
    )

    for classif_metric in df_plot['metric'].unique():
            df_metric = df_plot[df_plot['metric'] == classif_metric]
            ax.plot(df_metric['n'], df_metric['width_median'], marker='o',
                    color=color_dict_classif[classif_metric], linewidth=4,
                    markersize=10,
                    label=classif_metric)
            ax.plot(df_metric['n'], df_metric['width_q1'], linestyle="--",
                    color=color_dict_classif[classif_metric], linewidth=1)
            ax.plot(df_metric['n'], df_metric['width_q3'], linestyle="--",
                    color=color_dict_classif[classif_metric], linewidth=1)
            ax.fill_between(df_metric['n'], df_metric['width_q1'], df_metric['width_q3'],
                            alpha=0.2, color=color_dict_classif[classif_metric])

    # --- SEGMENTATION ---
    df_segm_stat = df_segm_width_perc[df_segm_width_perc['stat'] == stat]
    medians = df_segm_stat.groupby(['n', 'metric'])['width'].median().reset_index()
    q1 = df_segm_stat.groupby(['n', 'metric'])['width'].quantile(0.25).reset_index()
    q3 = df_segm_stat.groupby(['n', 'metric'])['width'].quantile(0.75).reset_index()
    df_plot = (
            medians.merge(q1, on=['n', 'metric'], suffixes=('_median', '_q1'))
            .merge(q3, on=['n', 'metric'])
            .rename(columns={'width': 'width_q3'})
    )

    for metric in df_plot['metric'].unique():
            df_metric = df_plot[df_plot['metric'] == metric]
            ax.plot(df_metric['n'], df_metric['width_median'], marker='o',
                    color=color_dict_segm[metric], linewidth=4, markersize=10, label=metric)
            ax.plot(df_metric['n'], df_metric['width_q1'], linestyle="--",
                    color=color_dict_segm[metric], linewidth=1)
            ax.plot(df_metric['n'], df_metric['width_q3'], linestyle="--",
                    color=color_dict_segm[metric], linewidth=1)
            ax.fill_between(df_metric['n'], df_metric['width_q1'], df_metric['width_q3'],
                            alpha=0.7, color=color_dict_segm[metric])

    ax.set_title(f"Width", fontsize=TITLE_FONTSIZE, weight="bold")
    ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Width", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y")
    ax.set_xlim(-5, 605)

    handles_all, labels_all = ax.get_legend_handles_labels()

    labels_all = [metric_labels[label] if label in metric_labels else label for label in labels_all]

    ax.legend(handles_all, labels_all,
            fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc="upper right")


    # ============================================================
    # SEGMENTATION COVERAGE DETAIL (bottom left)

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "masd"]
    stats_segm = ["mean"]

    df_segm_cov = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm_width = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    metrics_segm =['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice', 'assd', 'masd', 'hd', 'hd_perc']

    df_segm_cov_perc = df_segm_cov[(df_segm_cov["method"]=="percentile")]
    df_segm_width_perc = df_segm_width[(df_segm_width["method"]=="percentile")]

    palette = sns.color_palette("colorblind", len(metrics_segm))
    color_dict_segm = dict(zip(metrics_segm, palette))
    color_dict_segm.update({
        "iou": (31/255, 119/255, 180/255),        # #1f77b4 -> RGB normalized
        "boundary_iou": (74/255, 144/255, 226/255),  # #4a90e2 -> RGB normalized
        "cldice": (1/255, 104/255, 4/255)         # #016804 -> RGB normalized
    })

    ax = axs[1,0]
    ax.axhline(y=0.94, color='black', linestyle='--', linewidth=4, label="Target Coverage (94%)")
    for metric, df_metric in df_segm_cov_perc.groupby('metric'):
        medians = df_segm_cov_perc[df_segm_cov_perc['metric']==metric].groupby('n')['coverage'].median().values
        q1 = df_segm_cov_perc[df_segm_cov_perc['metric']==metric].groupby('n')['coverage'].quantile(0.25).values
        q3 = df_segm_cov_perc[df_segm_cov_perc['metric']==metric].groupby('n')['coverage'].quantile(0.75).values
        ax.plot(df_metric['n'].unique(), medians, marker='o', label=metric_labels[metric], color=color_dict_segm[metric], linewidth=4, markersize=10)
        ax.fill_between(df_metric['n'].unique(), q1, q3, alpha=0.2, color=color_dict_segm[metric])

    ax.set_title(f"Coverage", weight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Sample size', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Coverage (%)', fontsize=LABEL_FONTSIZE)
    ax.grid(True, axis='y')
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax.set_ylim(0.79, 1.01)

    ax.legend(title_fontsize=LEGEND_TITLE_FONTSIZE, fontsize=LEGEND_FONTSIZE, loc="lower right")

    axs[1,1].axis('off')

    # ============================================================
    for ax, letter in zip(axs.flatten(), ['A', 'B', 'C']):
            ax.text(0.5, 0.98, letter, transform=ax.transAxes,
                    fontsize=48, fontweight='bold', va='top', ha='center')

    # Global layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 10: Sample Size Needs")
    parser.add_argument("--root_folder", type=str, required=True,
                        help="Root folder containing the results folders.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output figure. If not provided, defaults to 'clean_figs/main/fig10_sample_needs.pdf' inside the root folder.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig10_sample_needs.pdf")

    # Call your plotting function
    plot_fig10_sample_needs(root_folder, output_path)

if __name__=="__main__":
    main()