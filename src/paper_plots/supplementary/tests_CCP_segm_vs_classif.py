import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
import argparse
from ..df_loaders import extract_df_segm_cov, extract_df_classif_cov
from ..plot_utils import metric_labels, stat_labels, method_labels, upload_to_overleaf
from .test_basic import format_p
def perform_fits_segm(df_segm, metrics, stats):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in metrics:
                for stat in stats:
                    df_metric_stat = df_algo[(df_algo['metric'] == metric) & (df_algo['stat']==stat)]
                    for method in df_metric_stat['method'].unique():
                        df_metric_stat_method = df_metric_stat[df_metric_stat['method'] == method]
                        df_metric_stat_method = df_metric_stat_method.sort_values(by='n')
                        n_values = df_metric_stat_method['n'].to_numpy()
                        coverages = df_metric_stat_method['coverage'].to_numpy()
                        Y = 0.95 - coverages
                        X = np.vstack([1/n_values]).T
                        beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                        rel_error = np.sqrt(res[0]) / np.linalg.norm(coverages)
                        new_row = {
                            'task': task,
                            'algo': algo,
                            'metric': metric,
                            'stat': stat,
                            'method': method,
                            'beta2': beta2[0],
                            'R2': rel_error
                        }
                        results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def perform_fits_classif(df_classif):
    results = []
    for task in df_classif['task'].unique():
        df_task = df_classif[df_classif['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
                df_metric = df_algo[df_algo['metric'] == metric]
                for method in df_metric['method'].unique():
                    df_metric_method = df_metric[df_metric['method'] == method]
                    df_metric_method = df_metric_method.sort_values(by='n')
                    n_values = df_metric_method['n'].to_numpy()
                    coverages = df_metric_method['coverage'].to_numpy()
                    Y = 0.95 - coverages
                    X = np.vstack([1/n_values]).T
                    beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                    rel_error = np.sqrt(res[0]) / np.linalg.norm(coverages)
                    new_row = {
                        'task': task,
                        'algo': algo,
                        'metric': metric,
                        'method': method,
                        'beta2': beta2[0],
                        'rel_error': rel_error
                    }
                    results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def perform_pairwise_tests_segm_classif(df_fit_results, df_fit_results_classif):

    segm_metrics = ['dsc', 'nsd', 'cldice', 'iou', 'boundary_iou']
    classif_metrics = df_fit_results_classif['metric'].unique()
    n_values = df_fit_results['n'].unique()

    p_values = {str(n): {m : {m2: None for m2 in segm_metrics} for m in classif_metrics} for n in n_values}
    for n in n_values:
        for metric1 in classif_metrics:
            for metric2 in segm_metrics:
                data_metric1 = df_fit_results_classif[(df_fit_results_classif["method"]=='percentile')& (df_fit_results_classif['metric'] == metric1) &( df_fit_results_classif['n']==n)]
                data_metric2 = df_fit_results[(df_fit_results["stat"]=='mean')&(df_fit_results["method"]=='percentile') & (df_fit_results['metric'] == metric2)&( df_fit_results['n']==n)]

                def statistic(x, y):
                    return np.mean(x) - np.mean(y)

                res = permutation_test(
                    (data_metric1['value'].to_numpy(), data_metric2['value'].to_numpy()),
                    statistic,
                    vectorized=False,
                    n_resamples=100000,
                    alternative='less'
                )
                pval = res.pvalue

                p_values[str(n)][metric1][metric2] = pval

    return p_values

def get_pvalues_segm_classif(p_vals):
    pval_list = []
    locations = []

    for n, metric1_dict in p_vals.items():
        for metric1, metric2_dict in metric1_dict.items():
            for metric2, p_val in metric2_dict.items():
                if p_val is not None:
                    pval_list.append(p_val)
                    locations.append(
                        (n, metric1, metric2)
                    )

    pval_array = np.asarray(pval_list)
    return(pval_array, locations)

def reconstruct_segm_classif(qvals, locations,p_vals,alphas):
   
    significance = {
        n: {
            metric1: {
                    metric2: None
                        for metric2 in metric2_dict
                    }
                for metric1,metric2_dict in metric1_dict.items()
        }
        for n, metric1_dict in p_vals.items()
    }
    qvalues = {
        n: {
            metric1: {
                    metric2:  None
                        for metric2 in metric2_dict
                    }
                for metric1,metric2_dict in metric1_dict.items()
        }
        for n, metric1_dict in p_vals.items()
    }
    # Fill significance levels using q-values
    for (n, metric1, metric2), q in zip(locations, qvals):
        significance[n][metric1][metric2] = np.sum(q < alphas)
        qvalues[n][metric1][metric2] = q

    # Fill missing values
    for n, metric1_dict in p_vals.items():
        for metric1, metric2_dict in metric1_dict.items():
            for metric2, p_val in metric2_dict.items():
                if p_val is None:
                    significance[n][metric1][metric2] = 0

    return qvalues,significance

def tell_significance(p_vals, alphas=np.array([0.01, 0.05]), bonferroni_correction=True):
    num_comparisons = sum(
        p_val is not None
        for method_dict in p_vals.values()
        for stat_dict in method_dict.values()
        for metric1_dict in stat_dict.values()
        for p_val in metric1_dict.values()
    )

    if bonferroni_correction and num_comparisons > 0:
        alphas_corrected = alphas / num_comparisons
    else:
        alphas_corrected = alphas

    significance = {}
    for method, stat_dict in p_vals.items():
        significance[method] = {}
        for stat, metric1_dict in stat_dict.items():
            significance[method][stat] = {}
            for metric1, metric2_dict in metric1_dict.items():
                significance[method][stat][metric1] = {}
                for metric2, p_val in metric2_dict.items():
                    if p_val is not None:
                        significance[method][stat][metric1][metric2] = np.sum(p_val < alphas_corrected)
                    else:
                        significance[method][stat][metric1][metric2] = 0
    return significance


def plot_significance_matrix_segm_classif(
    significance: dict,
    p_values: dict,
    output_path: str,
    upload_overleaf: bool = False,
):
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    color_map_dict = {
        -1: "#F5F5F5",   # not available
        0: "#F8CC80FF",  # not significant
        1: "#D55E00",    # significant
    }

    n_values = list(significance.keys())
    metrics_classif = list(next(iter(significance.values())).keys())
    metrics_segm = list(next(iter(next(iter(significance.values())).values())).keys())

    classif_ticklabels = [metric_labels.get(m, m) for m in metrics_classif]
    segm_ticklabels = [metric_labels.get(m, m) for m in metrics_segm]

    fig, axes = plt.subplots(1, len(n_values), figsize=(15, 8), sharey=True)
    axes = np.atleast_1d(axes)

    for col, n in enumerate(n_values):
        ax = axes[col]
        sign_n = significance[n]
        p_values_n = p_values[n]

        global_matrix = np.zeros((len(metrics_segm), len(metrics_classif)))
        pval_matrix = []

        for j, metric_segm in enumerate(metrics_segm):
            pval_row = []
            for k, metric_classif in enumerate(metrics_classif):
                val = sign_n.get(metric_classif, {}).get(metric_segm, None)
                if val is None:
                    global_matrix[j, k] = -1
                elif val == 0:
                    global_matrix[j, k] = 0
                else:
                    global_matrix[j, k] = 1

                p_val = p_values_n.get(metric_classif, {}).get(metric_segm, None)
                pval_row.append("" if p_val is None else format_p(p_val))
            pval_matrix.append(pval_row)

        cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix)])

        sns.heatmap(
            global_matrix,
            xticklabels=classif_ticklabels,
            yticklabels=segm_ticklabels,
            annot=pval_matrix,
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt="",
            annot_kws={"fontsize": 5},
        )
        ax.tick_params(axis="x", rotation=90, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        ax.set_title(f"n={int(float(n))}", fontsize=14)

    legend_elements = [
        mpatches.Patch(facecolor="#D55E00", edgecolor="k",
                       label="Significant \n(FDR-adjusted p < 0.05)"),
        mpatches.Patch(facecolor="#F8CC80FF", edgecolor="k",
                       label="Not significant"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="k",
                       label="Not available"),
    ]
    legend = axes[-1].legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=10,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=10,
    )
    # exclude the legend from the layout engine but keep it for saving
    legend.set_in_layout(False)

    # tight_layout, leaving free space on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    # re-apply the desired spacing (tight_layout overrides it)
    fig.subplots_adjust(left=0.07, right=0.85, wspace=0.07, hspace=0)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            "Preprint/supp_figs/tests/cov_segm_classif.pdf",
            commit_msg="Update figure test cov segm classif",
        )


def main():
    """
    Standalone entry point for the segm-vs-classif coverage figure.

    Note: the BH-FDR correction applied here pools p-values from this test only.
    `make_correction_fdr.py` instead pools across all tests before correcting, so
    the q-values — and therefore the figure — differ between the two paths.
    """
    parser = argparse.ArgumentParser(
        description="Perform pairwise significance tests on segmentation CI coverage fits "
                    "vs classification macro CI coverage fits."
    )
    parser.add_argument("--root_folder", type=str, required=True,
                        help="Root folder containing results_metrics_segm and results_metrics_classif_macro.")
    parser.add_argument("--output_path", type=str, required=False,
                        help="Output path for the significance matrix plot.")
    parser.add_argument("--upload_overleaf", action="store_true",
                        help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, "clean_figs/supplementary/test_results/cov_segm_classif/all_n.pdf"
    )

    file_prefix = "aggregated_results"

    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice"]
    stats_segm = ["mean", "median", "trimmed_mean", "std", "iqr_length"]
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm["method"] == "percentile"]

    metrics_classif = ["balanced_accuracy", "ap", "auc", "f1_score"]
    averages_classif = ["macro"]
    folder_path_classif = os.path.join(root_folder, "results_metrics_classif_macro")
    df_classif = extract_df_classif_cov(folder_path_classif, file_prefix, metrics_classif, averages_classif)
    df_classif = df_classif[df_classif["method"] == "percentile"]

    print("Data loaded. Performing fits...")
    df_fit_results_segm = perform_fits_segm(df_segm, metrics_segm, stats_segm)
    df_fit_results_classif = perform_fits_classif(df_classif)
    print("Fitting completed.")

    p_values = perform_pairwise_tests_segm_classif(df_fit_results_segm, df_fit_results_classif)
    print("Pairwise tests completed.")

    alphas = np.array([0.001, 0.01, 0.05])
    pvals, locations = get_pvalues_segm_classif(p_values)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_values, significance = reconstruct_segm_classif(qvals, locations, p_values, alphas)

    print("Making plot...")
    plot_significance_matrix_segm_classif(
        significance, p_values, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()