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
from .test_basic import fit_ccp as fit_ccp_segm
from .test_basic_classif import format_p, fit_ccp as fit_ccp_classif
from .test_bca_classif import perform_pairwise_tests_bca_classif
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels, upload_to_overleaf

import seaborn as sns




def perform_pairwise_tests_bca(df_results):

    metrics = df_results['metric'].unique()
   
    n_values=df_results['n'].unique()
    p_values = {str(n): {metric : None for metric in metrics} for n in n_values}
    for n in n_values:
    
        for metric in metrics:
                
            data_bca = df_results[(df_results["method"]=='bca') & (df_results["stat"]=='mean') & (df_results['metric'] == metric)& (df_results['n']==n)]
            data_percentile= df_results[(df_results["method"]=='percentile') & (df_results["stat"]=='mean') & (df_results['metric'] == metric)& (df_results['n']==n)]
            grp1 = (
                data_bca
                .groupby(['task', 'algo'])['value']
                .mean()
                .reset_index(name='beta1')
            )
        
            grp2 = (
                data_percentile
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
                    alternative='greater'
                )
                pval = res.pvalue
            p_values[str(n)][metric] = pval
            

    return p_values






def get_pvalues_bca(p_values):
    pvals = []
    keys = []
    for n,metric_dict in p_values.items():
       
        for metric, pval in metric_dict.items():
            
            if pval is not None and not np.isnan(pval):
                pvals.append(pval)
                keys.append((n, metric))

    pvals = np.asarray(pvals)
    return(pvals, keys)


def reconstruct_bca(qvals, keys,pvalues,alphas):
    q_values = {
        n:{
            metric: None
                for metric in metric_dict
        }
        for n, metric_dict in pvalues.items()
    }

    significant = {
        n: {
                metric: None
                for metric in metric_dict
        }
        for n, metric_dict in pvalues.items()
    }
    
    for (n, metric), qval in zip(keys, qvals):
        q_values[n][metric] = qval
        significant[n][metric] = np.sum(qval < alphas)

    return q_values,significant
    

def plot_significance_matrix_bca(
    significance: dict,
    p_values: dict,
    task: str,
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
    metrics = list(next(iter(significance.values())).keys())
    metric_ticklabels = [metric_labels.get(m, m) for m in metrics]

    fig, axes = plt.subplots(len(n_values), 1, figsize=(15, 10), sharex=True)
    axes = np.atleast_1d(axes)

    for row, n in enumerate(n_values):
        ax = axes[row]
        sign_n = significance[n]
        p_values_n = p_values[n]

        global_matrix = []
        pval_matrix = []
        for metric in metrics:
            val = sign_n.get(metric)
            if val is None:
                global_matrix.append(-1)
            elif val == 0:
                global_matrix.append(0)
            else:
                global_matrix.append(1)

            p_val = p_values_n.get(metric)
            pval_matrix.append("" if p_val is None else format_p(p_val))

        cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix)])

        sns.heatmap(
            [global_matrix],
            annot=[pval_matrix],
            xticklabels=metric_ticklabels,
            yticklabels=[f"n={int(float(n))}"],
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt="",
            annot_kws={"fontsize": 12},
        )
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", rotation=45, labelsize=14)

    legend_elements = [
        mpatches.Patch(facecolor="#D55E00", edgecolor="k",
                       label="Significant \n (FDR-adjusted p < 0.05)"),
        mpatches.Patch(facecolor="#F8CC80FF", edgecolor="k",
                       label="Not significant"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="k",
                       label="Not available"),
    ]
    legend = axes[-1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.8),
        ncol=1,
        fontsize=10,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=10,
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
            f"Preprint/supp_figs/tests/cov_bca_{task}.pdf",
            commit_msg=f"Update figure test bca {task}",
        )

def main():
    """
    Standalone entry point for the BCa-vs-percentile figure.

    Note: the BH-FDR correction applied here pools p-values from this test only.
    `make_correction_fdr.py` instead pools across all tests before correcting, so
    the q-values — and therefore the figure — differ between the two paths.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Supp Figure significance matrix of BCa vs percentile coverage."
    )
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--task", default="segm", choices=["segm", "classif"],
                        help="Whether to test segmentation or classification metrics.")
    parser.add_argument("--agreg_type", default="micro", choices=["micro", "macro"],
                        help="Aggregation type of the classification metrics (ignored when task is segm).")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    parser.add_argument("--upload_overleaf", action="store_true", help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    task = args.task
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, f"clean_figs/supplementary/test_results/cov_bca_{task}/all_n.pdf"
    )

    print("fitting ccp")
    if task == "segm":
        segm_path = os.path.join(root_folder, "results_metrics_segm")
        df_fit_results = fit_ccp_segm(segm_path)
        valid_fits = df_fit_results[df_fit_results["R2"] <= 0.1]
    else:
        if args.agreg_type == "micro":
            classif_path = os.path.join(root_folder, "results_metrics_classif")
        else:
            classif_path = os.path.join(root_folder, "results_metrics_classif_macro")
        df_fit_results = fit_ccp_classif(classif_path, args.agreg_type)
        valid_fits = df_fit_results[df_fit_results["R2"] <= 0.2]

    print("performing tests")
    if task == "segm":
        p_values = perform_pairwise_tests_bca(valid_fits)
    else:
        p_values = perform_pairwise_tests_bca_classif(valid_fits)

    print("significance")
    alphas = np.array([0.001, 0.01, 0.05])
    pvals, keys = get_pvalues_bca(p_values)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_values, significance = reconstruct_bca(qvals, keys, p_values, alphas)

    print("making plot")
    plot_significance_matrix_bca(
        significance, p_values, task, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()