import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import permutation_test
import argparse

from ..df_loaders import extract_df_segm_cov
from ..plot_utils import metric_labels, stat_labels, method_labels

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
                        rel_error = np.linalg.norm(X @ beta2 - Y) / np.linalg.norm(Y)
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
                            statistic,
                            vectorized=False,
                            n_resamples=10000,
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

def plot_significance_matrix(significance, p_vals, output_path):

    metric_order = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metrics_all = list(next(iter(next(iter(significance.values())).values())).keys())
    metrics_all = [m for m in metric_order if m in metrics_all]

    fig, axes = plt.subplots(len(stats), len(methods), figsize=(15 * len(methods), 12 * len(stats)))

    for row, stat in enumerate(stats):
        for col, method in enumerate(methods):
            ax = axes[row, col] if len(stats) > 1 else axes[col]

            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                ax.axis('off')
                continue

            # Extract significance for the specific method and stat
            method_stat_significance = significance.get(method, {}).get(stat, {})
            global_matrix = np.zeros((len(metrics_all), len(metrics_all)))

            for i, metric1 in enumerate(metrics_all):
                for j, metric2 in enumerate(metrics_all):
                    val = method_stat_significance.get(metric1, {}).get(metric2, None)
                    global_matrix[i, j] = min(3, val) if val is not None else 0
                global_matrix[i, i] = -1

            # Create p_val matrix for heatap 
            pval_matrix = np.full((len(metrics_all), len(metrics_all)), 0.0)
            for i, metric1 in enumerate(metrics_all):
                for j, metric2 in enumerate(metrics_all):
                    p_val = p_vals.get(method, {}).get(stat, {}).get(metric1, {}).get(metric2, None)
                    if p_val is not None:
                        pval_matrix[i, j] = p_val.round(4)
                    else:
                        pval_matrix[i, j] = 0.0
                pval_matrix[i, i] = 0.0  # Diagonal
            
            values = np.unique(global_matrix)

            # full mapping dictionary
            color_map_dict = {
                -1: '#000000',
                0: '#d9d9d9',
                1: '#fee08b',
                2: '#fdae61',
                3: '#d73027',
            }
            # extract only the colors for values that appear
            colors = [color_map_dict[v] for v in values]

            # build colormap
            cmap = ListedColormap(colors)
            
            # Plot heatmap
            labels = [metric_labels.get(m, m) for m in metrics_all]
            sns.heatmap(
                global_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=pval_matrix,
                cmap=cmap,
                cbar=False,
                ax=ax,
                annot_kws={"fontsize": 16}
            )
            ax.tick_params(axis='x', rotation=45, labelsize=14)

            ax.tick_params(axis='y', rotation=45, labelsize=14)

            ax.set_title(f"Stat : {stat_labels[stat]}, Method: {method_labels[method]}", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1% (Red)'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5% (Orange)'),
        mpatches.Patch(facecolor='#fee08b', edgecolor='k', label='10% (Yellow)'),
        mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant (Gray)')
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels with Bonferroni correction",
        title_fontsize=16
    )
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perform pairwise significance tests on segmentation CI coverage fits.")
    parser.add_argument('--root_folder', type=str, required=True, help='Root folder containing results_metrics_segm')
    parser.add_argument('--output_path', type=str, required=False, help='Output path for the significance matrix plot.')
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/tests_CCP_segm.pdf")

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats)

    print("Data loaded. Performing fits...")

    df_fit_results = perform_fits(df_segm, stats)
    print("Fitting completed.")
    p_values = perform_pairwise_tests(df_fit_results)
    print("Pairwise tests completed.")
    significance = tell_significance(p_values, bonferroni_correction=True)
    plot_significance_matrix(significance, p_values, output_path)

if __name__ == "__main__":
    main()