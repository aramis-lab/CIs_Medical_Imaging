import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import itertools
import argparse

from ..plot_utils import metric_labels
from ..df_loaders import extract_df_classif_cov, extract_df_segm_cov, extract_df_segm_width, extract_df_classif_width

def plot_with_iqr(ax, df, y, label, marker):
    med = df.groupby("n")[y].median()
    q1 = df.groupby("n")[y].quantile(0.25)
    q3 = df.groupby("n")[y].quantile(0.75)

    ax.plot(med.index, med.values, marker=marker, linewidth=4, label=label)
    ax.fill_between(med.index, q1.values, q3.values, alpha=0.25)


def plot_fig10_sample_needs(root_folder:str, output_path:str, scale_log:bool=True):

    plt.rcdefaults()

    # Define consistent font sizes
    TITLE_FONTSIZE = 40
    LABEL_FONTSIZE = 32
    TICK_FONTSIZE = 28 
    LEGEND_FONTSIZE = 34
    LEGEND_TITLE_FONTSIZE = 36

    # Define metric lists
    metrics_segm = ["dsc", "nsd", "iou", "boundary_iou", "cldice", "hd", "hd_perc", "masd", "assd"]
    bounded_metrics_segm = ["dsc", "nsd", "iou", "boundary_iou", "cldice"]
    metrics_classif = ["balanced_accuracy", "auc", "ap", "f1_score"]
    stats_segm = ["mean"]

    # Load segmentation data
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    df_segm_cov = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm_width = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    # Load classification data
    folder_path_classif = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_classif = "aggregated_results"
    df_classif_cov = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)
    df_classif_width = extract_df_classif_width(folder_path_classif, file_prefix_classif, metrics_classif)

    # Color palettes
    palette_segm = sns.color_palette("Blues", 1)
    color_dict_segm = dict(zip(metrics_segm, [palette_segm[0]] * len(metrics_segm)))
    palette_classif = sns.color_palette("Reds", 1)
    color_dict_classif = dict(zip(metrics_classif, [palette_classif[0]] * len(metrics_classif)))

    # Filter by percentile method
    df_segm_cov_perc = df_segm_cov[df_segm_cov["method"] == "percentile"]
    df_segm_width_perc = df_segm_width[df_segm_width["method"] == "percentile"]
    df_classif_cov_perc = df_classif_cov[df_classif_cov["method"] == "percentile"]
    df_classif_width_perc = df_classif_width[df_classif_width["method"] == "percentile"]

    stat = stats_segm[0]

    # df_segm_stat = df_segm_width_perc[(df_segm_width_perc['stat'] == stat) & (df_segm_width_perc['metric'] == 'dsc')]
    # medians = df_segm_stat.groupby(['n'])['width'].median().reset_index()
    # q1 = df_segm_stat.groupby(['n'])['width'].quantile(0.25).reset_index()
    # q3 = df_segm_stat.groupby(['n'])['width'].quantile(0.75).reset_index()
    # df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'width': 'width_q3'})
    # plt.figure(figsize=(16, 12)) 
    # plt.plot(df_plot['n'], df_plot['width_median'], marker='s',
    #         color='orange', linewidth=4, markersize=10,
    #         label=metric_labels['dsc'])
    # plt.fill_between(df_plot['n'], df_plot['width_q1'], df_plot['width_q3'],
    #                 alpha=0.6, color='orange')
    
    # plt.title(f"Width", fontsize=TITLE_FONTSIZE, weight="bold")
    # plt.xlabel("Sample size", fontsize=LABEL_FONTSIZE)
    # plt.ylabel("Width", fontsize=LABEL_FONTSIZE)
    # plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    # plt.grid(True, axis="y")
    # plt.ylim(-0.05, 0.4)
    # plt.axhline(y=0.05, color='black', linestyle='--', linewidth=4, label=f"Width= 0.05")

    # plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc="upper right")

    # # Save figure (optional)
    
    # output_dir = os.path.join(root_folder, "clean_figs")
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, f"width_dsc.pdf"), bbox_inches="tight")
    
    # =================================================================
    # LOOP OVER ALL PAIRS OF METRICS
    # =================================================================
    for segm_metric, classif_metric in itertools.product(metrics_segm, metrics_classif):
        
        if segm_metric not in bounded_metrics_segm:
            target_widths = [""]
        else:
            target_widths = [0.1, 0.05, 0.01]
        
        target_coverages = [0.94, 0.925, (0.925, 0.975)]

        for target_coverage in target_coverages:
            for target_width in target_widths:
                
                if target_width == "":
                    fig, axs = plt.subplots(1, 1, figsize=(24, 14))
                    axs = [axs]
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(40, 16))
                fig.suptitle(f"{metric_labels[segm_metric]} (Segm) vs {metric_labels[classif_metric]} (Classif)", fontsize=TITLE_FONTSIZE + 4, weight="bold")

                # ============================================================
                # COVERAGE (left plot)
                # ============================================================
                ax = axs[0]
                if isinstance(target_coverage, tuple):
                    ax.axhline(y=target_coverage[0], color='black', linestyle='--', linewidth=4, label=f"Coverage= ({(target_coverage[0]*100):.1f}%, {(target_coverage[1]*100):.1f}%)")
                    ax.axhline(y=target_coverage[1], color='black', linestyle='--', linewidth=4)
                else:
                    ax.axhline(y=target_coverage, color='black', linestyle='--', linewidth=4, label=f"Coverage= ({(target_coverage*100):.1f}%)")

                # --- CLASSIFICATION ---
                df_classif = df_classif_cov_perc[df_classif_cov_perc["metric"] == classif_metric]
                medians = df_classif.groupby(['n'])['coverage'].median().reset_index()
                q1 = df_classif.groupby(['n'])['coverage'].quantile(0.25).reset_index()
                q3 = df_classif.groupby(['n'])['coverage'].quantile(0.75).reset_index()
                df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'coverage': 'coverage_q3'})

                ax.plot(df_plot['n'], df_plot['coverage_median'], marker='o',
                        color="tab:blue", linewidth=4, markersize=10,
                        label=metric_labels[classif_metric])
                ax.fill_between(df_plot['n'], df_plot['coverage_q1'], df_plot['coverage_q3'],
                                alpha=0.2, color="tab:blue")

                # --- SEGMENTATION ---
                df_segm_stat = df_segm_cov_perc[(df_segm_cov_perc['stat'] == stat) & (df_segm_cov_perc['metric'] == segm_metric)]
                medians = df_segm_stat.groupby(['n'])['coverage'].median().reset_index()
                q1 = df_segm_stat.groupby(['n'])['coverage'].quantile(0.25).reset_index()
                q3 = df_segm_stat.groupby(['n'])['coverage'].quantile(0.75).reset_index()
                df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'coverage': 'coverage_q3'})

                ax.plot(df_plot['n'], df_plot['coverage_median'], marker='s',
                        color="tab:orange", linewidth=4, markersize=10,
                        label=metric_labels[segm_metric])
                ax.fill_between(df_plot['n'], df_plot['coverage_q1'], df_plot['coverage_q3'],
                                alpha=0.2, color="tab:orange")

                ax.set_title("Coverage", fontsize=TITLE_FONTSIZE, weight="bold")
                ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
                ax.set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                ax.grid(True, axis="y")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
                
                if scale_log:
                    ax.set_xscale("log")
                ax.set_ylim(0.79, 1.01)
                ax.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)

                # ============================================================
                # WIDTH (right plot)
                # ============================================================
                if target_width != "":
                    ax = axs[1]
                    ax.axhline(y=target_width, color='black', linestyle='--', linewidth=4, label=f"Width= ({target_width})")

                    # --- CLASSIFICATION ---
                    df_classif = df_classif_width_perc[df_classif_width_perc["metric"] == classif_metric]
                    medians = df_classif.groupby(['n'])['width'].median().reset_index()
                    q1 = df_classif.groupby(['n'])['width'].quantile(0.25).reset_index()
                    q3 = df_classif.groupby(['n'])['width'].quantile(0.75).reset_index()
                    df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'width': 'width_q3'})

                    ax.plot(df_plot['n'], df_plot['width_median'], marker='o',
                            color="tab:blue", linewidth=4, markersize=10,
                            label=metric_labels[classif_metric])
                    ax.fill_between(df_plot['n'], df_plot['width_q1'], df_plot['width_q3'],
                                    alpha=0.2, color="tab:blue")

                    # --- SEGMENTATION ---
                    df_segm_stat = df_segm_width_perc[(df_segm_width_perc['stat'] == stat) & (df_segm_width_perc['metric'] == segm_metric)]
                    medians = df_segm_stat.groupby(['n'])['width'].median().reset_index()
                    q1 = df_segm_stat.groupby(['n'])['width'].quantile(0.25).reset_index()
                    q3 = df_segm_stat.groupby(['n'])['width'].quantile(0.75).reset_index()
                    df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'width': 'width_q3'})

                    ax.plot(df_plot['n'], df_plot['width_median'], marker='s',
                            color="tab:orange", linewidth=4, markersize=10,
                            label=metric_labels[segm_metric])
                    ax.fill_between(df_plot['n'], df_plot['width_q1'], df_plot['width_q3'],
                                    alpha=0.2, color="tab:orange")

                    ax.set_title(f"Width", fontsize=TITLE_FONTSIZE, weight="bold")
                    ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
                    ax.set_ylabel("Width", fontsize=LABEL_FONTSIZE)
                    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                    ax.grid(True, axis="y")
                    ax.set_ylim(-0.05, 1.05)
                    if scale_log:
                        ax.set_xscale("log")
                    ax.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc="upper right")

                # Save figure (optional)
                
                output_dir = os.path.join(root_folder, "clean_figs/comparison_plots_segm_classif_log" if scale_log else "clean_figs/comparison_plots_segm_classif_linear")
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, f"{segm_metric}_vs_{classif_metric}_{target_coverage}_{target_width}.pdf"), bbox_inches="tight")
                plt.close(fig)


    # ============================================================
    # SEGMENTATION COVERAGE DETAIL (bottom left)

    # --- Segmentation metrics
    metrics_segm = ['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice', 'assd', 'masd', 'hd', 'hd_perc']
    bounded_metrics = ['dsc', 'nsd', 'iou', 'boundary_iou', 'cldice']  # only these have bounded [0,1] range
    stats_segm = ['mean']

    # --- Load data
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    df_segm_cov = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm_width = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    # --- Filter percentile method
    df_segm_cov_perc = df_segm_cov[df_segm_cov["method"] == "percentile"]
    df_segm_width_perc = df_segm_width[df_segm_width["method"] == "percentile"]

    # --- Colors
    palette = sns.color_palette("colorblind", len(metrics_segm))
    color_dict_segm = dict(zip(metrics_segm, palette))
    color_dict_segm.update({
        "iou": (31/255, 119/255, 180/255),
        "boundary_iou": (74/255, 144/255, 226/255),
        "cldice": (1/255, 104/255, 4/255)
    })

    # ========================================================================
    # LOOP OVER ALL PAIRS OF SEGMENTATION METRICS
    # ========================================================================
    for metric_a, metric_b in itertools.combinations(metrics_segm, 2):

        both_bounded = (metric_a in bounded_metrics) and (metric_b in bounded_metrics)
        if both_bounded:
            target_widths = [0.1, 0.05, 0.01]
        else:
            target_widths = [""]
        target_coverages = [0.94, 0.925, (0.925, 0.975)]
        # Create figure: 1 or 2 subplots depending on boundedness
        for target_coverage in target_coverages:
            for target_width in target_widths:
                ncols = 2 if both_bounded else 1
                fig, axs = plt.subplots(1, ncols, figsize=(24 if not both_bounded else 40, 14 if not both_bounded else 16))
                if ncols == 1:
                    axs = [axs]
                fig.suptitle(f"{metric_labels[metric_a]} vs {metric_labels[metric_b]}", fontsize=TITLE_FONTSIZE + 4, weight="bold")

                # ============================================================
                # COVERAGE PLOT
                # ============================================================
                ax = axs[0]
                if isinstance(target_coverage, tuple):
                    ax.axhline(y=target_coverage[0], color='black', linestyle='--', linewidth=4, label=f"Coverage= ({(target_coverage[0]*100):.1f}%, {(target_coverage[1]*100):.1f}%)")
                    ax.axhline(y=target_coverage[1], color='black', linestyle='--', linewidth=4)
                else:
                    ax.axhline(y=target_coverage, color='black', linestyle='--', linewidth=4, label=f"Coverage= ({(target_coverage*100):.1f}%)")

                for i, metric in enumerate([metric_a, metric_b]):
                    df_metric = df_segm_cov_perc[df_segm_cov_perc["metric"] == metric]
                    medians = df_metric.groupby('n')['coverage'].median().reset_index()
                    q1 = df_metric.groupby('n')['coverage'].quantile(0.25).reset_index()
                    q3 = df_metric.groupby('n')['coverage'].quantile(0.75).reset_index()
                    df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'coverage': 'coverage_q3'})

                    ax.plot(df_plot['n'], df_plot['coverage_median'],
                            marker='o', label=metric_labels[metric], color="tab:blue" if i==0 else "tab:orange",
                            linewidth=4, markersize=10)
                    ax.fill_between(df_plot['n'], df_plot['coverage_q1'], df_plot['coverage_q3'],
                                    alpha=0.2, color="tab:blue" if i==0 else "tab:orange")

                ax.set_title("Coverage", fontsize=TITLE_FONTSIZE, weight="bold")
                ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
                ax.set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
                ax.grid(True, axis='y')
                if scale_log:
                    ax.set_xscale("log")
                ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
                ax.set_ylim(0.79, 1.01)
                ax.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)

                # ============================================================
                # WIDTH PLOT (only for bounded)
                # ============================================================
                if both_bounded:
                    ax = axs[1]
                    ax.axhline(y=target_width, color='black', linestyle='--', linewidth=4, label=f"Width= ({target_width})")

                    for i, metric in enumerate([metric_a, metric_b]):
                        df_metric = df_segm_width_perc[df_segm_width_perc["metric"] == metric]
                        medians = df_metric.groupby('n')['width'].median().reset_index()
                        q1 = df_metric.groupby('n')['width'].quantile(0.25).reset_index()
                        q3 = df_metric.groupby('n')['width'].quantile(0.75).reset_index()
                        df_plot = medians.merge(q1, on='n', suffixes=('_median', '_q1')).merge(q3, on='n').rename(columns={'width': 'width_q3'})

                        ax.plot(df_plot['n'], df_plot['width_median'],
                                marker='o', label=metric_labels[metric], color="tab:blue" if i==0 else "tab:orange",
                                linewidth=4, markersize=10)
                        ax.fill_between(df_plot['n'], df_plot['width_q1'], df_plot['width_q3'],
                                        alpha=0.2, color="tab:blue" if i==0 else "tab:orange")

                    ax.set_title("Width", fontsize=TITLE_FONTSIZE, weight="bold")
                    ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
                    ax.set_ylabel("Width", fontsize=LABEL_FONTSIZE)
                    ax.grid(True, axis='y')
                    if scale_log:
                        ax.set_xscale("log")
                    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
                    ax.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc="upper right")

                # ------------------------------------------------------------
                # SAVE FIGURE
                # ------------------------------------------------------------
                output_dir = os.path.join(root_folder, "clean_figs/comparison_plots_segm_segm_log" if scale_log else "clean_figs/comparison_plots_segm_segm_linear")
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{metric_a}_vs_{metric_b}_{target_coverage}_{target_width}.pdf")
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)

    # ============================================================
    # ADDITIONAL BIG FIGURE: SELECTED METRICS
    # ============================================================

    import string  # for letters A-F

    plt.rcdefaults()  # reset to default colors (blue/orange)

    # layout: 2 rows, 3 columns
    fig, axes = plt.subplots(3, 2, figsize=(30, 42), sharex=False,
                            gridspec_kw={"height_ratios": [1, 1, 1]})

    # Define selected pairs
    SELECTED_PAIRS = [
        ("dsc", "auc"),                  # coverage + width
        ("dsc", "balanced_accuracy"),    # coverage + width
        ("iou", "assd"),                 # coverage only
    ]

    subplot_letters = list(string.ascii_uppercase[:6])  # ['A', 'B', 'C', 'D', 'E', 'F']
    letter_idx = 0  # track the current letter

    for row, (segm_metric, other_metric) in enumerate(SELECTED_PAIRS):

        # -------------------
        # Coverage (top row)
        # -------------------
        ax = axes[row, 0]

        # Other metric
        if other_metric in metrics_classif:
            df_other = df_classif_cov_perc[df_classif_cov_perc["metric"] == other_metric]
            marker = "o"
        else:  # segmentation metric (ASSD)
            df_other = df_segm_cov_perc[(df_segm_cov_perc["metric"] == other_metric) &
                                        (df_segm_cov_perc["stat"] == "mean")]
            marker = "o"

        plot_with_iqr(ax, df_other, "coverage", metric_labels[other_metric], marker=marker)

        # Segmentation metric
        df_segm = df_segm_cov_perc[(df_segm_cov_perc["metric"] == segm_metric) &
                                (df_segm_cov_perc["stat"] == "mean")]
        plot_with_iqr(ax, df_segm, "coverage", metric_labels[segm_metric], marker="s")
        ax.axhline(y=0.925, color='black', linestyle='--', linewidth=4, label=f"Coverage= 92.5%")

        ax.set_title(f"{metric_labels[segm_metric]} vs {metric_labels[other_metric]}", fontsize=34, weight="bold")
        ax.grid(True, axis="y")
        ax.set_ylim(.80, 1)

        ax.set_xlabel("Sample size", fontsize=30)
        if row < 2:
            ax.set_xscale("log")
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        else:
            ax.set_xscale("linear")
            ax.set_xlim(0, 250)
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    
        ax.set_ylabel("Coverage (%)", fontsize=30)
        if row==0:
            ax.set_title("Coverage")
        ax.legend(fontsize=24)

        # Add subplot letter
        ax.text(0.5, 1, subplot_letters[letter_idx], transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top', ha='right')
        letter_idx += 1

        # -------------------
        # Width (bottom row)
        # -------------------
        ax = axes[row, 1]

        if row < 2:  # only first two pairs have width
            df_other_w = df_classif_width_perc[df_classif_width_perc["metric"] == other_metric]
            plot_with_iqr(ax, df_other_w, "width", metric_labels[other_metric], marker="o")
            df_segm_w = df_segm_width_perc[(df_segm_width_perc["metric"] == segm_metric) &
                                        (df_segm_width_perc["stat"] == "mean")]
            plot_with_iqr(ax, df_segm_w, "width", metric_labels[segm_metric], marker="s")

            ax.axhline(y=0.05, color='black', linestyle='--', linewidth=4, label=f"Width=0.05")

            ax.set_ylabel("Width", fontsize=30)
            ax.grid(True, axis="y")

            ax.set_xscale("log")
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
            ax.grid(True, axis="y")
            if row==0:
                ax.set_title("Width")


        else:
            df_segm_w = df_segm_width_perc[(df_segm_width_perc["metric"] == 'dsc') &
                                        (df_segm_width_perc["stat"] == "mean")]
            plot_with_iqr(ax, df_segm_w, "width", 'DSC', marker="s")
            ax.axhline(y=0.05, color='black', linestyle='--', linewidth=4, label=f"Width= ({target_width})")

            ax.set_ylabel("Width", fontsize=30)
            ax.grid(True, axis="y")
            ax.tick_params(labelsize=26)
            ax.set_xscale("linear")
            ax.set_xlim(0, 250)
        ax.legend(fontsize=24)
        ax.set_xlabel("Sample size", fontsize=30)

        # Add subplot letter
        ax.text(0.5, 1, subplot_letters[letter_idx], transform=ax.transAxes,
                fontsize=40, fontweight='bold', va='top', ha='right')
        letter_idx += 1

    # -------------------
    # Final formatting
    # -------------------
    plt.tight_layout(rect=[0, 0, 1, 0.94])


    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, bbox_inches="tight")
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
    # plot_fig10_sample_needs(root_folder, output_path, scale_log=True)
    plot_fig10_sample_needs(root_folder, output_path, scale_log=False)

if __name__=="__main__":
    main()