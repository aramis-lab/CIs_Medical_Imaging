import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import seaborn as sns
from .test_basic import format_p
from ..plot_utils import metric_labels, stat_labels, method_labels,upload_to_overleaf

def fit_wdp_segm(segm_path):
    results = []
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice"]
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
                                width_norms = df_algo[f'width_{method}'].to_numpy()
                                Y = width_norms
                                X = np.vstack([1 /np.sqrt(n_values)]).T
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
    return(df_fit_results)


def fit_wdp_classif(classif_path,agreg_type):
    results = []
    if agreg_type=='micro':

        metrics= ["accuracy","ap", "auc", "f1_score"]
    else:
        metrics= ["balanced_accuracy","ap", "auc", "f1_score"]
    methods=['basic', 'bca', 'percentile']
    for metric in metrics:
        
        path=os.path.join(classif_path,f'aggregated_results_{metric}.csv')
        
        df_metric_stat = pd.read_csv(path)
        for task in df_metric_stat['subtask'].unique():
            df_task = df_metric_stat[df_metric_stat['subtask'] == task]
            for algo in df_task['alg_name'].unique():
                    df_algo = df_task[df_task['alg_name'] == algo]
                    for method in methods:

                        n_values = df_algo['n'].to_numpy()
                        width_norms = df_algo[f'width_{method}'].to_numpy()
                        Y = width_norms
                        X = np.vstack([1 /np.sqrt(n_values)]).T
                        beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                        rel_error = np.sqrt(res[0]) / np.linalg.norm(width_norms)
                        new_row = {
                            'task': task,
                            'algo': algo,
                            'metric': metric,
                            'method': method,
                            'width_decay_pace': beta2[0],
                            'R2': rel_error
                        }
                        results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)


def perform_pairwise_tests_wdp_segm_classif(df_fit_results, df_fit_results_classif):

    segm_metrics = ['dsc', 'nsd', 'iou', 'boundary_iou', 'cldice']
    classif_metrics = df_fit_results_classif['metric'].unique()
    methods = ['basic', 'bca', 'percentile']
    stats = df_fit_results['stat'].unique()
    n_values = df_fit_results['n'].unique()
    p_values = {str(n): {m : {m2: None for m2 in segm_metrics} for m in classif_metrics} for n in n_values}
    for n in n_values:
        print(n)
        for metric1 in classif_metrics:
            for metric2 in segm_metrics:
                data_metric1 = df_fit_results_classif[(df_fit_results_classif["method"]=='percentile') & (df_fit_results_classif['metric'] == metric1)& (df_fit_results_classif['n'] == n)]
                data_metric2 = df_fit_results[(df_fit_results["method"]=='percentile') & (df_fit_results['metric'] == metric2)& (df_fit_results['stat'] == 'mean')& (df_fit_results['n'] == n)]
                def statistic(x, y):
                    return np.mean(x) - np.mean(y)
                res = permutation_test(
                    (data_metric1['value'].to_numpy(), data_metric2['value'].to_numpy()),
                    statistic,
                    vectorized=False,
                    n_resamples=100000,
                    alternative='greater'
                )
                pval = res.pvalue

                p_values[str(n)][metric1][metric2] = pval
    return p_values

def get_pvalues_wdp_segm_classif(p_vals):
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

def reconstruct_wdp_segm_classif(qvals, locations,p_vals,alphas):
    significance = {
        n: {
            metric1: {metric2: None 
                        for metric2 in metric2_dict}
                for metric1, metric2_dict in metric1_dict.items()
        }
        for n, metric1_dict in p_vals.items()
    }
    qvalues = {
        n: {
            metric1: {metric2: None 
                        for metric2 in metric2_dict}
                for metric1, metric2_dict in metric1_dict.items()
        }
        for n, metric1_dict in p_vals.items()
    }
    # Fill significance levels using q-values
    for (n, metric1, metric2), q in zip(locations, qvals):
        significance[n][metric1][metric2] = np.sum(q < alphas)
        qvalues[n][metric1][metric2] = q

    for n, metric1_dict in p_vals.items():
            for metric1, metric2_dict in metric1_dict.items():
                for metric2, p_val in metric2_dict.items():
                    if p_val is None:

                        significance[n][metric1][metric2] = 0

    return qvalues,significance


def tell_significance(
    p_vals,
    alphas=np.array([0.001, 0.01, 0.05])
):
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

    _, qvals, _, _ = multipletests(
        pval_array,
        method="fdr_bh"
    )

    significance = {
        method: {
            stat: {
                metric1: None
                for metric1 in stat_dict
            }
            for stat, stat_dict in method_dict.items()
        }
        for method, method_dict in p_vals.items()
    }

    # Fill significance levels using q-values
    for (method, stat, metric1, metric2), q in zip(locations, qvals):
        significance[method][stat][metric1][metric2] = np.sum(q < alphas)

    # Fill missing values
    for method, stat_dict in p_vals.items():
        for stat, metric1_dict in stat_dict.items():
            for metric1, metric2_dict in metric1_dict.items():
                for metric2, p_val in metric2_dict.items():
                    if p_val is None:
                        significance[method][stat][metric1][metric2] = 0

    return significance


def plot_significance_matrix_wdp_segm_classif(
    significance: dict,
    p_values: dict,
    agreg_type: str,
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

    fig, axes = plt.subplots(1, len(n_values), figsize=(18, 12), sharey=True)
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
                val = sign_n.get(metric_classif, {}).get(metric_segm)
                if val is None:
                    global_matrix[j, k] = -1
                elif val == 0:
                    global_matrix[j, k] = 0
                else:
                    global_matrix[j, k] = 1

                p_val = p_values_n.get(metric_classif, {}).get(metric_segm)
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
        ax.set_title(f"n={int(float(n))}", fontsize=12)

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
        fontsize=12,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=12,
    )
    # exclude the legend from the layout engine but keep it for saving
    legend.set_in_layout(False)

    # tight_layout, leaving free space on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            f"Preprint/supp_figs/tests/width_segm_classif_{agreg_type}.pdf",
            commit_msg="Update figure test width segm classif",
        )


def main():
    """
    Standalone entry point for the width-decay-pace segm-vs-classif figure.

    Note: the BH-FDR correction applied here pools p-values from this test only.
    `make_correction_fdr.py` instead pools across all tests before correcting, so
    the q-values — and therefore the figure — differ between the two paths.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Supp Figure significance matrix of CI width decay pace, segmentation vs classification."
    )
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--agreg_type", default="micro", choices=["micro", "macro"],
                        help="Aggregation type of the classification metrics.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    parser.add_argument("--upload_overleaf", action="store_true", help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    agreg_type = args.agreg_type
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, f"clean_figs/supplementary/test_results/width_segm_classif_{agreg_type}/all_n.pdf"
    )

    segm_path = os.path.join(root_folder, "results_metrics_segm")
    if agreg_type == "micro":
        classif_path = os.path.join(root_folder, "results_metrics_classif")
    else:
        classif_path = os.path.join(root_folder, "results_metrics_classif_macro")

    print("fitting wdp")
    df_fit_results_segm = fit_wdp_segm(segm_path)
    df_fit_results_classif = fit_wdp_classif(classif_path, agreg_type)

    print("performing tests")
    p_values = perform_pairwise_tests_wdp_segm_classif(df_fit_results_segm, df_fit_results_classif)

    print("significance")
    alphas = np.array([0.001, 0.01, 0.05])
    pvals, locations = get_pvalues_wdp_segm_classif(p_values)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_values, significance = reconstruct_wdp_segm_classif(qvals, locations, p_values, alphas)

    print("making plot")
    plot_significance_matrix_wdp_segm_classif(
        significance, p_values, agreg_type, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()