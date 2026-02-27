import matplotlib.pyplot as plt
import os
import argparse
import string 
from ..plot_utils import metric_labels
from ..df_loaders import extract_df_classif_cov, extract_df_segm_cov, extract_df_segm_width, extract_df_classif_width

def plot_with_iqr(ax, df, y, label, marker):
    med = df.groupby("n")[y].median()
    q1 = df.groupby("n")[y].quantile(0.25)
    q3 = df.groupby("n")[y].quantile(0.75)

    ax.plot(med.index, med.values, marker=marker, linewidth=4, label=label)
    ax.fill_between(med.index, q1.values, q3.values, alpha=0.25)


def plot_fig10_sample_needs(root_folder:str, output_path:str):

    plt.rcdefaults()

    # Define consistent font sizes
    TITLE_FONTSIZE = 40
    LABEL_FONTSIZE = 32
    TICK_FONTSIZE = 28 
    LEGEND_FONTSIZE = 34

    # Define metric lists
    metrics_segm = ["dsc", "nsd", "iou", "boundary_iou", "cldice", "hd", "hd_perc", "masd", "assd"]
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

 
    # Filter by percentile method
    df_segm_cov_perc = df_segm_cov[df_segm_cov["method"] == "percentile"]
    df_segm_width_perc = df_segm_width[df_segm_width["method"] == "percentile"]
    df_classif_cov_perc = df_classif_cov[df_classif_cov["method"] == "percentile"]
    df_classif_width_perc = df_classif_width[df_classif_width["method"] == "percentile"]

 
    #plt.rcdefaults()  # reset to default colors (blue/orange)

    # layout: 2 rows, 3 columns
    _, axes = plt.subplots(3, 2, figsize=(30, 42), sharex=False,
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

        # ax.set_title(f"{metric_labels[segm_metric]} vs {metric_labels[other_metric]}", fontsize=TITLE_FONTSIZE, weight="bold")
        ax.grid(True, axis="y")
        ax.set_ylim(.80, 1)

        ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)
        if row < 2:
            ax.set_xscale("log")
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
        else:
            ax.set_xscale("linear")
            ax.set_xlim(0, 250)
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))

    
        ax.set_ylabel("Coverage (%)", fontsize=LABEL_FONTSIZE)
        if row==0:
            ax.set_title("Coverage (%)", fontsize=TITLE_FONTSIZE, weight='bold')
        ax.legend(fontsize=LEGEND_FONTSIZE)

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

            ax.set_ylabel("Width", fontsize=LABEL_FONTSIZE)
            ax.grid(True, axis="y")

            ax.set_xscale("log")
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}'))
            ax.grid(True, axis="y")
            if row==0:
                ax.set_title("Width", fontsize=TITLE_FONTSIZE, weight='bold')


        else:
            df_segm_w = df_segm_width_perc[(df_segm_width_perc["metric"] == 'dsc') &
                                        (df_segm_width_perc["stat"] == "mean")]
            plot_with_iqr(ax, df_segm_w, "width", 'DSC', marker="s")
            ax.axhline(y=0.05, color='black', linestyle='--', linewidth=4, label=f"Width= 0.05")

            ax.set_ylabel("Width", fontsize=LABEL_FONTSIZE)
            ax.grid(True, axis="y")
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.set_xscale("linear")
            ax.set_xlim(0, 250)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel("Sample size", fontsize=LABEL_FONTSIZE)

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
    plot_fig10_sample_needs(root_folder, output_path)

if __name__=="__main__":
    main()