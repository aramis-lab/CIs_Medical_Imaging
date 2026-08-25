import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import permutation_test
import argparse

from .test_basic import format_p
from ..df_loaders import extract_df_segm_cov, extract_df_segm_width   # replaces the cov-only import
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
                        df_metric_stat_method = (
                        df_metric_stat[df_metric_stat['method'] == method]
                        .sort_values(by='n'))

                        n_values = df_metric_stat_method['n'].to_numpy()
                        width_norms = df_metric_stat_method['width_norm'].to_numpy()

                        Y = width_norms
                        X = np.vstack([1 / np.sqrt(n_values)]).T

                        beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]

                        rel_error = np.sqrt(res[0]) / np.linalg.norm(width_norms)

                        new_row = {
                            'task': task,
                            'algo': algo,
                            'metric': metric,
                            'stat': stat,
                            'method': method,
                            'width_decay_pace': beta2[0],
                            'R2': rel_error
                        }
                        results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def perform_pairwise_tests(df_fit_results):
    
    metrics = df_fit_results['metric'].unique()
    methods = df_fit_results['method'].unique()
    stats = df_fit_results['stat'].unique()
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
                        .groupby(['task', 'algo'])['width_decay_pace']
                        .mean()
                        .reset_index(name='beta1')
                    )
                    grp2 = (
                        data_metric2
                        .groupby(['task', 'algo'])['width_decay_pace']
                        .mean()
                        .reset_index(name='width_decay_pace')
                    )

                    merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

                    merged = merged.dropna(subset=['beta1', 'width_decay_pace'])

                    if len(merged) < 2:
                        pval = None
                    else:
                        def statistic(x, y):
                            return np.mean(x) - np.mean(y)

                        res = permutation_test(
                            (merged['beta1'].to_numpy(), merged['width_decay_pace'].to_numpy()),
                            statistic,
                            vectorized=False,
                            n_resamples=50000,
                            alternative='two-sided'
                        )
                        pval = res.pvalue

                    p_values[method][stat][metric1][metric2] = pval
                    p_values[method][stat][metric2][metric1] = pval

    return p_values

def tell_significance(p_vals, alphas=np.array([0.01, 0.05, 0.1]), bonferroni_correction=True):
    
    m = len(next(iter(next(iter(p_vals.values())).values())).keys())
    num_comparisons = m - 1

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
                        significance[method][stat][metric1][metric2] = 0
    return significance

def plot_significance_matrix_wdp_segm(
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

    metric_order = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    param_methods = ["param_z", "param_t"]

    # significance is np.sum(p < alphas) with alphas = [0.01, 0.05, 0.1],
    # so 3 means p < 0.01, 2 means p < 0.05, 1 means p < 0.1
    color_map_dict = {
        -1: "#000000",   # diagonal
        0: "#d9d9d9",    # not significant
        1: "#fee08b",    # 10%
        2: "#fdae61",    # 5%
        3: "#d73027",    # 1%
    }

    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metrics_all = list(next(iter(next(iter(significance.values())).values())).keys())
    metrics_all = [m for m in metric_order if m in metrics_all]
    metric_ticklabels = [metric_labels.get(m, m) for m in metrics_all]

    fig, axes = plt.subplots(
        len(stats), len(methods),
        figsize=(15 * len(methods), 12 * len(stats)),
        squeeze=False,
    )

    last_visible_ax = None
    for row, stat in enumerate(stats):
        for col, method in enumerate(methods):
            ax = axes[row][col]

            if (stat != "mean") and (method in param_methods):
                ax.axis("off")
                continue
            last_visible_ax = ax

            method_stat_significance = significance.get(method, {}).get(stat, {})
            p_values_method_stat = p_values.get(method, {}).get(stat, {})

            global_matrix = np.zeros((len(metrics_all), len(metrics_all)))
            pval_matrix = []

            for i, metric1 in enumerate(metrics_all):
                pval_row = []
                for j, metric2 in enumerate(metrics_all):
                    val = method_stat_significance.get(metric1, {}).get(metric2, None)
                    global_matrix[i, j] = min(3, val) if val is not None else 0

                    p_val = p_values_method_stat.get(metric1, {}).get(metric2, None)
                    pval_row.append("" if p_val is None else format_p(p_val))
                global_matrix[i, i] = -1
                pval_row[i] = ""
                pval_matrix.append(pval_row)

            cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix)])

            sns.heatmap(
                global_matrix,
                xticklabels=metric_ticklabels,
                yticklabels=metric_ticklabels,
                annot=pval_matrix,
                cmap=cmap,
                cbar=False,
                ax=ax,
                fmt="",
                annot_kws={"fontsize": 16},
            )
            ax.tick_params(axis="x", rotation=45, labelsize=14)
            ax.tick_params(axis="y", rotation=45, labelsize=14)
            ax.set_title(
                f"Stat : {stat_labels.get(stat, stat)}, Method: {method_labels.get(method, method)}",
                fontsize=16,
            )

    legend_elements = [
        mpatches.Patch(facecolor="#d73027", edgecolor="k", label="1%"),
        mpatches.Patch(facecolor="#fdae61", edgecolor="k", label="5%"),
        mpatches.Patch(facecolor="#fee08b", edgecolor="k", label="10%"),
        mpatches.Patch(facecolor="#d9d9d9", edgecolor="k", label="Not significant"),
    ]
    legend_ax = last_visible_ax if last_visible_ax is not None else axes[-1][-1]
    legend = legend_ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith Bonferroni correction",
        title_fontsize=16,
    )
    # exclude the legend from the layout engine but keep it for saving
    legend.set_in_layout(False)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            "Preprint/supp_figs/tests/width_segm.pdf",
            commit_msg="Update figure test width segm",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Perform pairwise significance tests on segmentation CI width fits."
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
        root_folder, "clean_figs/supplementary/tests_WDP_segm.pdf"
    )

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats = ["mean"]

    df_segm = extract_df_segm_width(folder_path_segm, file_prefix_segm, metrics_segm, stats)
    print("Data loaded. Performing fits...")

    df_fit_results = perform_fits(df_segm, stats)
    print("Fitting completed.")

    p_values = perform_pairwise_tests(df_fit_results)
    print("Pairwise tests completed.")

    significance = tell_significance(p_values, bonferroni_correction=True)

    print("Making plot...")
    plot_significance_matrix_wdp_segm(
        significance, p_values, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()