import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import scipy
from .test_basic_classif import format_p
import seaborn as sns
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels, upload_to_overleaf

def fit_ccp(segm_path):
    results = []
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats = ["mean", "median", "trimmed_mean", "std", "iqr_length"]
    methods=['basic', 'bca', 'percentile','param_z', 'param_t']
    for metric in metrics_segm:
        for stat in stats:

            path=os.path.join(segm_path,f'aggregated_results_{metric}_{stat}.csv')
            
            df_metric_stat = pd.read_csv(path)
            for task in df_metric_stat['subtask'].unique():
                df_task = df_metric_stat[df_metric_stat['subtask'] == task]
                for algo in df_task['alg_name'].unique():
                        df_algo = df_task[df_task['alg_name'] == algo]
                        for method in methods:
                            if (method in ['param_z', 'param_t']) & (stat!='mean'):
                                continue
                            else:
                                n_values = df_algo['n'].to_numpy()
                                coverages = df_algo[f'contains_true_stat_{method}'].to_numpy()
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
                                    'coverage': beta2[0],
                                    'R2': rel_error
                                }
                                results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)


def perform_pairwise_tests_basic(df_results):

    metrics = df_results['metric'].unique()
    methods = ['bca', 'percentile', 'param_z', 'param_t']
    stats = df_results['stat'].unique()
    n_values=df_results['n'].unique()
    p_values = {str(n): {stat : {metric : {m : None for m in methods} for metric in metrics} for stat in stats} for n in n_values}
    for n in n_values:
        print(n)
        for stat in stats:
        
            for metric in metrics:
            
                for j in methods:

                    
                    if (j in ['param_z', 'param_t']) & (stat!='mean'):
                        continue 
                    data_basic = df_results[(df_results["method"]=='basic') & (df_results["stat"]==stat) & (df_results['metric'] == metric)& (df_results['n']==n)]
                    data_methods= df_results[(df_results["method"]==j) & (df_results["stat"]==stat) & (df_results['metric'] == metric)& (df_results['n']==n)]
                    grp1 = (
                        data_basic
                        .groupby(['task', 'algo'])['value']
                        .mean()
                        .reset_index(name='beta1')
                    )
             
                    grp2 = (
                        data_methods
                        .groupby(['task', 'algo'])['value']
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
                            alternative='less'
                        )
                        pval = res.pvalue
                    p_values[str(n)][stat][metric][j] = pval
                

    return p_values

def get_pvalues_basic(p_values):
    pvals = []
    keys = []
    for n,stat_dict in p_values.items():
        for stat, metric_dict in stat_dict.items():
            for metric, method_dict in metric_dict.items():
                for method, pval in method_dict.items():

                    if pval is not None and not np.isnan(pval):
                        pvals.append(pval)
                        keys.append((n,stat, metric, method))

    pvals = np.asarray(pvals)
    return(pvals, keys)


def reconstruct_basic(qvals, keys,pvalues,alphas):
    q_values = {
        n:{
            stat: {
                metric: {
                    method: None
                    for method in method_dict
                }
                for metric, method_dict in metric_dict.items()
            }
            for stat, metric_dict in stat_dict.items()
        }
        for n, stat_dict in pvalues.items()
    }

    significant = {
        n: {
            stat: {
                metric: {
                    method: None
                    for method in method_dict
                }
                for metric, method_dict in metric_dict.items()
            }
            for stat, metric_dict in stat_dict.items()
        }
        for n, stat_dict in pvalues.items()
    }
    
    for (n,stat, metric, method), qval in zip(keys, qvals):
        q_values[n][stat][metric][method] = qval

        significant[n][stat][metric][method] = qval if qval is None else np.sum(qval < alphas)

    return q_values,significant
    


def tell_significance(p_values, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []

    for stat, metric_dict in p_values.items():
        for metric, method_dict in metric_dict.items():
            for method, pval in method_dict.items():

                if pval is not None and not np.isnan(pval):
                    pvals.append(pval)
                    keys.append((stat, metric, method))

    pvals = np.asarray(pvals)

    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )

    # Reconstruct nested dictionaries
    q_values = {
        stat: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for stat, metric_dict in p_values.items()
    }

    significant = {
        stat: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for stat, metric_dict in p_values.items()
    }
    
    for (stat, metric, method), qval in zip(keys, qvals):
        q_values[stat][metric][method] = qval
        significant[stat][metric][method] = np.sum(qval < alphas)

    return q_values,significant



def plot_significance_matrix_basic(
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

    metrics_all = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    main_methods = ["bca", "percentile"]
    param_methods = ["param_z", "param_t"]

    color_map_dict = {
        -1: "#F5F5F5",   # not available
        0: "#F8CC80FF",  # not significant
        1: "#D55E00",    # significant
    }

    n_values = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metric_ticklabels = [metric_labels.get(m, m) for m in metrics_all]

    # the "mean" column carries the two extra parametric methods, so it needs more width
    width_ratios = [2 if stat == "mean" else 1 for stat in stats]

    fig, axes = plt.subplots(
        len(n_values), len(stats),
        figsize=(18, 12 * len(n_values)),
        sharey=True,
        squeeze=False,
        width_ratios=width_ratios,
    )

    legend_elements = [
        mpatches.Patch(facecolor="#D55E00", edgecolor="k",
                       label="Significant \n (FDR-adjusted p < 0.05)"),
        mpatches.Patch(facecolor="#F8CC80FF", edgecolor="k",
                       label="Not significant"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="k",
                       label="Not available"),
    ]

    legends = []   # keep track of all legends so they are never cropped
    for row, n in enumerate(n_values):
        sign_n = significance[n]
        p_values_n = p_values[n]

        for col, stat in enumerate(stats):
            ax = axes[row][col]
            methods = main_methods + param_methods if stat == "mean" else main_methods
            method_ticklabels = [method_labels.get(m, m) for m in methods]

            stat_significance = sign_n.get(stat, {})
            global_matrix = np.zeros((len(metrics_all), len(methods)))
            pval_matrix = []

            for i, metric in enumerate(metrics_all):
                pval_row = []
                for j, method in enumerate(methods):
                    val = stat_significance.get(metric, {}).get(method, None)
                    if val is None:
                        global_matrix[i, j] = -1
                    elif val == 0:
                        global_matrix[i, j] = 0
                    else:
                        global_matrix[i, j] = 1

                    p_val = p_values_n.get(stat, {}).get(metric, {}).get(method)
                    pval_row.append("" if p_val is None else format_p(p_val))
                pval_matrix.append(pval_row)

            cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix)])

            sns.heatmap(
                global_matrix,
                annot=pval_matrix,
                xticklabels=method_ticklabels,
                yticklabels=metric_ticklabels,
                cmap=cmap,
                cbar=False,
                ax=ax,
                fmt="",
                linewidths=1,
                square=True,
                linecolor="white",
                annot_kws={"fontsize": 14},
            )
            ax.tick_params(axis="x", rotation=45, labelsize=14)
            ax.tick_params(axis="y", rotation=45, labelsize=14)
            if stat == "mean":
                ax.set_title(f"{stat_labels[stat]}, n={int(float(n))}", fontsize=14)
            else:
                ax.set_title(f"{stat_labels[stat]}", fontsize=14)

        # one legend per row, anchored on the rightmost heatmap of that row
        legend = axes[row][-1].legend(
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
        legends.append(legend)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    # re-apply the desired spacing (tight_layout overrides it)
    fig.subplots_adjust(right=0.75, top=1, bottom=0, hspace=0)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=tuple(legends))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            "Preprint/supp_figs/tests/cov_basic_segm.pdf",
            commit_msg="Update figure test basic segm",
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Supp Figure significance matrix of basic vs other methods for segmentation."
    )
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    parser.add_argument("--upload_overleaf", action="store_true", help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, "clean_figs/supplementary/test_results/cov_basic_segm/all_n.pdf"
    )

    segm_path = os.path.join(root_folder, "results_metrics_segm")

    print("fitting ccp")
    df_fit_results = fit_ccp(segm_path)
    valid_fits = df_fit_results[df_fit_results["R2"] <= 0.1]
    print("performing tests")
    p_values = perform_pairwise_tests_basic(valid_fits)
    print("significance")
    q_values, significance = tell_significance(p_values)

    print("making plot")
    plot_significance_matrix_basic(
        significance, p_values, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()