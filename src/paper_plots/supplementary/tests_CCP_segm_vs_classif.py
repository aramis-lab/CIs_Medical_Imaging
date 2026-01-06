import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import permutation_test
import argparse

from ..df_loaders import extract_df_segm_cov, extract_df_classif_cov
from ..plot_utils import metric_labels, stat_labels, method_labels

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

def perform_pairwise_tests(df_fit_results, df_fit_results_classif):

    segm_metrics = df_fit_results['metric'].unique()
    classif_metrics = df_fit_results_classif['metric'].unique()
    methods = df_fit_results_classif['method'].unique()
    stats = df_fit_results['stat'].unique()
    p_values = {met : {s : {m : {m2: None for m2 in segm_metrics} for m in classif_metrics} for s in stats} for met in methods}

    for method in methods:
        for stat in stats:
            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                continue
            for metric1 in classif_metrics:
                for metric2 in segm_metrics:
                    data_metric1 = df_fit_results_classif[(df_fit_results_classif["method"]==method) & (df_fit_results_classif['metric'] == metric1)]
                    data_metric2 = df_fit_results[(df_fit_results["method"]==method) & (df_fit_results['metric'] == metric2)]

                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)

                    res = permutation_test(
                        (data_metric1['beta2'].to_numpy(), data_metric2['beta2'].to_numpy()),
                        statistic,
                        vectorized=False,
                        n_resamples=50000,
                        alternative='two-sided'
                    )
                    pval = res.pvalue

                    p_values[method][stat][metric1][metric2] = pval

    return p_values

def tell_significance(p_vals, alphas=np.array([0.01, 0.05]), bonferroni_correction=True):
    
    m = len(next(iter(next(iter(p_vals.values())).values())).keys())
    n = len(next(iter(next(iter(next(iter(p_vals.values())).values())).values())).keys())
    num_comparisons = max(m, n)

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

def plot_significance_matrix_segm_vs_classif(root_folder:str, output_path:str):

    plt.rcdefaults()

    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice"]
    stats_segm = ["mean", "median", "trimmed_mean", "std", "iqr_length"]

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm['method'] == 'percentile']

    metrics_classif = ["balanced_accuracy", "ap", "auc", "f1_score"]

    folder_path_classif = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_classif = "aggregated_results"
    df_classif = extract_df_classif_cov(folder_path_classif, file_prefix_classif, metrics_classif)
    df_classif = df_classif[df_classif['method'] == 'percentile']

    df_fit_results_segm = perform_fits_segm(df_segm, metrics_segm, stats_segm)
    df_fit_results_classif = perform_fits_classif(df_classif)
    median_segm = df_fit_results_segm.groupby(['method', 'stat', 'metric'])['beta2'].median().reset_index()
    median_classif = df_fit_results_classif.groupby(['method', 'metric'])['beta2'].median().reset_index()

    order_segm = median_segm[median_segm['stat'] == 'mean'].sort_values('beta2')['metric'].tolist()
    order_classif = median_classif.sort_values('beta2')['metric'].tolist()
    print("Fitting completed.")
    p_values = perform_pairwise_tests(df_fit_results_segm, df_fit_results_classif)
    print("Pairwise tests completed.")
    significance = tell_significance(p_values, bonferroni_correction=True)

    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metrics_classif = list(next(iter(next(iter(significance.values())).values())).keys())
    metrics_segm = list(next(iter(next(iter(next(iter(significance.values())).values())).values())).keys())

    metrics_classif = [m for m in order_classif if m in metrics_classif]
    metrics_segm = [m for m in order_segm if m in metrics_segm]

    fig, axes = plt.subplots(len(stats), len(methods), figsize=(15 * len(methods), 12 * len(stats)))

    for row, stat in enumerate(stats):
        for col, method in enumerate(methods):
            if len(stats) == 1 and len(methods) == 1:
                ax = axes
            elif len(stats) == 1 or len(methods) == 1:
                ax = axes[max(row, col)]
            else:
                ax = axes[row, col]

            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                ax.axis('off')
                continue

            # Extract significance for the specific method and stat
            method_stat_significance = significance.get(method, {}).get(stat, {})
            global_matrix = np.zeros((len(metrics_segm), len(metrics_classif)))

            for i, metric1 in enumerate(metrics_segm):
                for j, metric2 in enumerate(metrics_classif):
                    val = method_stat_significance.get(metric2, {}).get(metric1, None)
                    global_matrix[i, j] = min(2, val) if val is not None else 0

            # Create p_val matrix for heatap 
            pval_matrix = []
            for i, metric1 in enumerate(metrics_segm):
                pval_row = []
                for j, metric2 in enumerate(metrics_classif):
                    p_val = p_values.get(method, {}).get(stat, {}).get(metric2, {}).get(metric1, None)
                    if p_val is not None:
                        pval_row.append(f"{p_val.round(4)}" if p_val >= 0.0001 else "<0.0001")
                    else:
                        pval_row.append("0")
                pval_matrix.append(pval_row)
            
            values = np.unique(global_matrix)

            # full mapping dictionary
            color_map_dict = {
                -1: '#000000',
                0: '#d9d9d9',
                1: '#fdae61',
                2: '#d73027',
            }
            # extract only the colors for values that appear
            colors = [color_map_dict[v] for v in values]

            # build colormap
            cmap = ListedColormap(colors)
            
            # Plot heatmap
            labels_x = [metric_labels.get(m, m) for m in metrics_classif]
            labels_y = [metric_labels.get(m, m) for m in metrics_segm]
            sns.heatmap(
                global_matrix,
                xticklabels=labels_x,
                yticklabels=labels_y,
                annot=pval_matrix,
                cmap=cmap,
                cbar=False,
                ax=ax,
                fmt='',
                annot_kws={"fontsize": 16}
            )
            ax.tick_params(axis='x', rotation=45, labelsize=14)

            ax.tick_params(axis='y', rotation=45, labelsize=14)

            ax.set_title(f"Stat : {stat_labels[stat]}, Method: {method_labels[method]}", fontsize=16)

            legend_elements = [
                mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1%, <0.0025'),
                mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%, <0.0125'),
                mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant')
            ]
            ax.legend(
                handles=legend_elements,
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                ncol=1,
                fontsize=16,
                frameon=True,
                title="Significance levels \nwith Bonferroni correction",
                title_fontsize=16
            )
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Perform pairwise significance tests on segmentation CI coverage fits vs classification macro CI coverage fits.")
    parser.add_argument('--root_folder', type=str, required=True, help='Root folder containing results_metrics_segm and results_metrics_classif_macro')
    parser.add_argument('--output_path', type=str, required=False, help='Output path for the significance matrix plot.')
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/tests_CCP_segm_vs_classif.pdf")

    plot_significance_matrix_segm_vs_classif(root_folder, output_path)

if __name__ == "__main__":
    main()