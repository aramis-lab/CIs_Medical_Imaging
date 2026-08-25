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
from .test_basic import format_p
from ..df_loaders import extract_df_segm_cov
from ..plot_utils import metric_labels, stat_labels, method_labels, upload_to_overleaf

def perform_fits(df_segm, stats):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
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

def perform_pairwise_tests_segm(df_fit_results):
    
    metrics = df_fit_results['metric'].unique()
    methods = df_fit_results['method'].unique()
    stats = ['mean', 'std']
    p_values = {met : {s : {m : {m2: None for m2 in metrics} for m in metrics} for s in stats} for met in methods}

    for method in methods:
        for stat in stats:

            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                continue

            for i in range(len(metrics)):

                for j in range(i + 1, len(metrics)):
                    metric1 = metrics[i]
                    metric2 = metrics[j]

                    data_metric1 = df_fit_results[(df_fit_results["method"]==method) & (df_fit_results["stat"]==stat) & (df_fit_results['metric'] == metric1)]
                    data_metric2 = df_fit_results[(df_fit_results["method"]==method) & (df_fit_results["stat"]==stat) & (df_fit_results['metric'] == metric2)]

                    grp1 = (
                        data_metric1
                        .groupby(['task', 'algo'])['beta2']
                        .mean()
                        .reset_index(name='beta1')
                    )
                    grp2 = (
                        data_metric2
                        .groupby(['task', 'algo'])['beta2']
                        .mean()
                        .reset_index(name='beta2')
                    )

                    merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

                    merged = merged.dropna(subset=['beta1', 'beta2'])

                    if len(merged) < 2:
                        pval = None
                    else:
                        def statistic(x, y):
                            return np.mean(x) - np.mean(y)

                        res = permutation_test(
                            (merged['beta1'].to_numpy(), merged['beta2'].to_numpy()),
                            statistic,permutation_type='samples',
                            vectorized=False,
                            n_resamples=100000,
                            alternative='two-sided'
                        )
                        pval = res.pvalue

                    p_values[method][stat][metric1][metric2] = pval
                    p_values[method][stat][metric2][metric1] = pval

    return p_values

def get_pvalues_segm(p_vals):
    pval_list = []
    locations = []

    for method, stat_dict in p_vals.items():
        for stat, metric1_dict in stat_dict.items():
            for metric1, metric2_dict in metric1_dict.items():
                for metric2, p_val in metric2_dict.items():
                    if p_val is not None:
                        pval_list.append(p_val)
                        locations.append(
                            (method, stat, metric1, metric2)
                        )

    pval_array = np.asarray(pval_list)
    return(pval_array, locations)

def reconstruct_segm(qvals, locations,p_vals,alphas):
    significance = {
        method: {
            stat: {
                metric1:{}
                for metric1 in stat_dict
            }
            for stat, stat_dict in method_dict.items()
        }
        for method, method_dict in p_vals.items()
    }
    qvalues = {
            method: {
                stat: {
                    metric1: {}
                    for metric1 in stat_dict
                }
                for stat, stat_dict in method_dict.items()
            }
            for method, method_dict in p_vals.items()
        }
    # Fill significance levels using q-values
    for (method, stat, metric1, metric2), q in zip(locations, qvals):
      
        significance[method][stat][metric1][metric2] = q if q is None else np.sum(q < alphas)
        qvalues[method][stat][metric1][metric2] = q

    # Fill missing values
    # for method, stat_dict in p_vals.items():
    #     for stat, metric1_dict in stat_dict.items():
    #         for metric1, metric2_dict in metric1_dict.items():
    #             for metric2, p_val in metric2_dict.items():
    #                 if p_val is None:
    #                     significance[method][stat][metric1][metric2] = 0

    return qvalues,significance

def tell_significance(p_vals, alphas=np.array([0.01, 0.05]), bonferroni_correction=True):
    
    m = len(p_vals) # number of methods
    n = len(next(iter(p_vals.values()))) # number of stats
    o = len(next(iter(next(iter(p_vals.values())).values()))) # number of metrics
    p = len(next(iter(next(iter(next(iter(p_vals.values())).values())).values()))) # number of metrics
    num_comparisons = m*n*o*(p-1)/2 # number of pairwise comparisons per method and stat, multiplied by number of methods and stats
    
    if bonferroni_correction:
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
                        significance[method][stat][metric1][metric2] = p_val
    return significance


def plot_significance_matrix_segm(
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

    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    param_methods = ["param_z", "param_t"]

    color_map_dict = {
        -1: "#F5F5F5",   # not available
        0: "#F8CC80FF",  # not significant
        1: "#D55E00",    # significant
    }

    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metric_ticklabels = [metric_labels.get(m, m) for m in metrics_segm]

    # constrained_layout removed: it is incompatible with tight_layout
    fig, axes = plt.subplots(
        len(methods), len(stats),
        figsize=(15 * len(stats), 15 * len(methods)),
        sharey=True,
        squeeze=False,
    )

    last_visible_ax = None
    for row, method in enumerate(methods):
        for col, stat in enumerate(stats):
            ax = axes[row][col]

            if (stat != "mean") and (method in param_methods):
                ax.axis("off")
                continue
            last_visible_ax = ax

            method_stat_significance = significance.get(method, {}).get(stat, {})
            p_values_method_stat = p_values.get(method, {}).get(stat, {})

            global_matrix = np.zeros((len(metrics_segm), len(metrics_segm)))
            pval_matrix = []

            for i, metric1 in enumerate(metrics_segm):
                pval_row = []
                for j, metric2 in enumerate(metrics_segm):
                    val = method_stat_significance.get(metric2, {}).get(metric1, None)
                    if val is None:
                        global_matrix[i, j] = -1
                    elif val == 0:
                        global_matrix[i, j] = 0
                    else:
                        global_matrix[i, j] = 1

                    p_val = p_values_method_stat.get(metric2, {}).get(metric1, None)
                    pval_row.append("" if p_val is None else format_p(p_val))
                pval_matrix.append(pval_row)

            # only the lower triangle is drawn, so the colormap must be built from it alone
            mask = np.triu(np.ones_like(global_matrix, dtype=bool), k=1)
            cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix[~mask])])

            sns.heatmap(
                global_matrix,
                xticklabels=metric_ticklabels,
                yticklabels=metric_ticklabels,
                annot=pval_matrix,
                mask=mask,
                cmap=cmap,
                cbar=False,
                ax=ax,
                square=True,
                linewidths=0.4,
                fmt="",
                annot_kws={"fontsize": 12},
            )
            ax.tick_params(axis="x", rotation=45, labelsize=14)
            ax.tick_params(axis="y", rotation=45, labelsize=14)
            ax.set_title(
                f"Stat : {stat_labels.get(stat, stat)}, Method: {method_labels.get(method, method)}",
                fontsize=16,
            )

    legend_elements = [
        mpatches.Patch(facecolor="#D55E00", edgecolor="k",
                       label="Significant \n(FDR-adjusted p < 0.05)"),
        mpatches.Patch(facecolor="#F8CC80FF", edgecolor="k",
                       label="Not significant"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="k",
                       label="Not available"),
    ]
    legend_ax = last_visible_ax if last_visible_ax is not None else axes[-1][-1]
    legend = legend_ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=16,
    )
    # exclude the legend from the layout engine but keep it for saving
    legend.set_in_layout(False)

    # tight_layout, leaving free space on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            "Preprint/supp_figs/tests/cov_segm.pdf",
            commit_msg="Update figure test cov segm",
        )


def main():
    """
    Standalone entry point for the pairwise segmentation-metric coverage figure.

    Note: the BH-FDR correction applied here pools p-values from this test only.
    `make_correction_fdr.py` instead pools across all tests before correcting, so
    the q-values — and therefore the figure — differ between the two paths.
    """
    parser = argparse.ArgumentParser(
        description="Perform pairwise significance tests on segmentation CI coverage fits."
    )
    parser.add_argument("--root_folder", type=str, required=True,
                        help="Root folder containing results_metrics_segm.")
    parser.add_argument("--output_path", type=str, required=False,
                        help="Output path for the significance matrix plot.")
    parser.add_argument("--upload_overleaf", action="store_true",
                        help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, "clean_figs/supplementary/test_results/coverage_segm_metrics/test_segm.pdf"
    )

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    # perform_pairwise_tests_segm compares these two stats
    stats = ["mean", "std"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats)
    print("Data loaded. Performing fits...")

    df_fit_results = perform_fits(df_segm, stats)
    print("Fitting completed.")

    p_values = perform_pairwise_tests_segm(df_fit_results)
    print("Pairwise tests completed.")

    alphas = np.array([0.001, 0.01, 0.05])
    pvals, locations = get_pvalues_segm(p_values)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_values, significance = reconstruct_segm(qvals, locations, p_values, alphas)

    print("Making plot...")
    plot_significance_matrix_segm(
        significance, p_values, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()