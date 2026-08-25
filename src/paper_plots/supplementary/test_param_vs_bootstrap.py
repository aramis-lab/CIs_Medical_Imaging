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
from .test_basic_classif import format_p

from ..plot_utils import metric_labels, stat_labels, method_labels,upload_to_overleaf
from ..df_loaders import extract_df_segm_cov, extract_df_segm_width, extract_df_classif_cov


def get_data(task):
    res=[]
    main_methods=['basic', 'bca', 'percentile']
    if task =='segm': 
        metrics = ["dsc", "iou", "boundary_iou", "nsd", "cldice", 'assd','masd','hd','hd_perc']
        methods=main_methods +['param_z', 'param_t']
    else:
        metrics= ['accuracy']
        methods=main_methods +["wilson","agresti_coull" ,"wald",'exact']
    path = f'../../../../results_metrics_{task}'
    
    for metric in metrics:
        if task=='segm':
            path_segm=os.path.join(path,f'aggregated_results_{metric}_mean.csv')
        else:
            path_segm=os.path.join(path,f'aggregated_results_{metric}.csv')
        df_metric = pd.read_csv(path_segm)
        for method in methods:
            values=df_metric[df_metric['n']==10][f'contains_true_stat_{method}'].values
            res.append({'metric':metric, 
                        'method':method, 
                        'coverage_values':values})
    
    return(pd.DataFrame(res))



def perform_pairwise_tests_param_boot(df_results_cov,task):
    
    
    bootstrap_methods=['basic', 'bca', 'percentile']
    
    
    if task=='segm':
        metrics=df_results_cov['metric'].unique()
 
    else:
        metrics=['accuracy']
    n_values = df_results_cov['n'].unique()
    p_values =  {str(n) : {m : {m2: None for m2 in bootstrap_methods} for m in metrics} for n in n_values}
    for n in n_values:
        print(n)
        for metric in metrics:

            for method2 in bootstrap_methods:
                if task=='segm':
                    df_boot = df_results_cov[(df_results_cov["method"]==method2) & (df_results_cov['stat']=='mean')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]

                    df_param= df_results_cov[(df_results_cov["method"]=='param_t')& (df_results_cov['stat']=='mean')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                else:
                    df_param= df_results_cov[(df_results_cov["method"]=='wilson')& (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                    df_boot = df_results_cov[(df_results_cov["method"]==method2) & (df_results_cov['metric'] == metric)&(df_results_cov['n'] == n)]
                grp1 = (
                        df_param
                        .groupby(['task', 'algo'])['value']
                        .mean()
                        .reset_index(name='beta1')
                    )
             
                grp2 = (
                    df_boot
                    .groupby(['task', 'algo'])['value']
                    .mean()
                    .reset_index(name='beta2')
                )

                merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

                merged = merged.dropna(subset=['beta1', 'beta2'])
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
                p_values[str(n)][metric][method2] = pval
    return p_values

def get_pvalues_param_boot(pvalues):
    pval_list = []
    locations = []

    for n, metric_dict in pvalues.items():
        for metric, method_dict in metric_dict.items():
            for method, p_val in method_dict.items():
                if p_val is not None:
                    pval_list.append(p_val)
                    locations.append(
                        (n, metric, method)
                    )

    pval_array = np.asarray(pval_list)
    return(pval_array, locations)

def reconstruct_param_boot(qvals, locations,p_vals,alphas):
    significance = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in p_vals.items()
    }
    qvalues = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in p_vals.items()
    }
    # Fill significance levels using q-values
    for (n, metric, method), q in zip(locations, qvals):
        significance[n][metric][method]= np.sum(q < alphas)
        qvalues[n][metric][method]= q

    # Fill missing values
    for n, metric_dict in p_vals.items():
        for metric, method_dict in metric_dict.items():
            for method, p_val in method_dict.items():

                if p_val is None:
                    significance[n][metric][method] = 0

    return qvalues,significance

def tell_significance(
    p_vals,
    alphas=np.array([0.001, 0.01, 0.05])
):
    pval_list = []
    locations = []

    for metric, boot_dict in p_vals.items():
        for method, param_dict in boot_dict.items():
            for param, p_val in param_dict.items():
                if p_val is not None:
                    pval_list.append(p_val)
                    locations.append(
                        (metric, boot, param)
                    )

    pval_array = np.asarray(pval_list)

    _, qvals, _, _ = multipletests(
        pval_array,
        method="fdr_bh"
    )

    significance = {
        metric: {
            boot: {
                param: None
                for param in boot_dict
            }
            for boot, boot_dict in metric_dict.items()
        }
        for metric, metric_dict in p_vals.items()
    }

    # Fill significance levels using q-values
    for (metric, boot, param), q in zip(locations, qvals):
        significance[metric][boot][param]= np.sum(q < alphas)

    # Fill missing values
    for metric, metric_dict in p_vals.items():
        for boot, boot_dict in metric_dict.items():
            for param, param_dict in boot_dict.items():

                if p_val is None:
                    significance[metric][boot][param] = 0

    return significance


def plot_significance_matrix_param_boot(
    significance: dict,
    p_values: dict,
    task: str,
    quantity: str,
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
    methods = list(next(iter(next(iter(significance.values())).values())).keys())

    metric_ticklabels = [metric_labels.get(m, m) for m in metrics]
    method_ticklabels = [method_labels.get(m, m) for m in methods]

    if task == "segm":
        fig, axes = plt.subplots(1, len(n_values), figsize=(15, 10), sharey=True)
        annot_fontsize = 6
        ytick_rotation = 0
        ytick_fontsize = 14
    else:
        fig, axes = plt.subplots(len(n_values), 1, figsize=(12, 20), sharex=True)
        annot_fontsize = 16
        ytick_rotation = 45
        ytick_fontsize = 16
    axes = np.atleast_1d(axes)

    for idx, n in enumerate(n_values):
        ax = axes[idx]
        sign_n = significance[n]
        p_values_n = p_values[n]

        global_matrix = np.zeros((len(metrics), len(methods)))
        pval_matrix = []

        for j, metric in enumerate(metrics):
            pval_row = []
            metric_significance = sign_n.get(metric, {})
            p_values_metric = p_values_n.get(metric, {})
            for k, method in enumerate(methods):
                val = metric_significance.get(method)
                if val is None:
                    global_matrix[j, k] = -1
                elif val == 0:
                    global_matrix[j, k] = 0
                else:
                    global_matrix[j, k] = 1

                p_val = p_values_metric.get(method)
                pval_row.append("" if p_val is None else format_p(p_val))
            pval_matrix.append(pval_row)

        cmap = ListedColormap([color_map_dict[v] for v in np.unique(global_matrix)])

        sns.heatmap(
            global_matrix,
            xticklabels=method_ticklabels,
            yticklabels=metric_ticklabels if task == "segm" else [f"n={int(float(n))}"],
            annot=pval_matrix,
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt="",
            annot_kws={"fontsize": annot_fontsize},
        )
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", rotation=ytick_rotation, labelsize=ytick_fontsize)
        ax.set_title(f"n={int(float(n))}" if task == "segm" else "", fontsize=16)

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
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    # re-apply the desired spacing (tight_layout overrides it)
    if task == "segm":
        fig.subplots_adjust(left=0.1, right=0.8, wspace=0.03)
    else:
        fig.subplots_adjust(left=0.08, right=0.8)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            f"Preprint/supp_figs/tests/{quantity}_param_boot_{task}.pdf",
            commit_msg=f"Update figure test {quantity}_param_boot_{task}",
        )


def main():
    """
    Standalone entry point for the parametric-vs-bootstrap figure.

    Note: the BH-FDR correction applied here pools p-values from this test only.
    `make_correction_fdr.py` instead pools across all tests before correcting, so
    the q-values — and therefore the figure — differ between the two paths.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Supp Figure significance matrix of parametric vs bootstrap intervals."
    )
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--task", default="segm", choices=["segm", "classif"],
                        help="Whether to test segmentation or classification metrics.")
    parser.add_argument("--quantity", default="cov", choices=["cov", "width"],
                        help="Whether to test coverage or interval width (width is segm only).")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    parser.add_argument("--upload_overleaf", action="store_true", help="Upload the plot to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    task = args.task
    quantity = args.quantity
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(
        root_folder, f"clean_figs/supplementary/test_results/{quantity}_param_boot_{task}/all_n.pdf"
    )

    file_prefix = "aggregated_results"
    print("get data")
    if task == "segm":
        folder_path = os.path.join(root_folder, "results_metrics_segm")
        metrics = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "assd", "masd", "hd", "hd_perc"]
        stats = ["mean"]
        if quantity == "width":
            df_results = extract_df_segm_width(folder_path, file_prefix, metrics, stats)
        else:
            df_results = extract_df_segm_cov(folder_path, file_prefix, metrics, stats)
    else:
        folder_path = os.path.join(root_folder, "results_metrics_classif")
        metrics = ["accuracy"]
        averages = ["micro"]
        df_results = extract_df_classif_cov(folder_path, file_prefix, metrics, averages)

    print("performing tests")
    p_values = perform_pairwise_tests_param_boot(df_results, task)

    print("significance")
    alphas = np.array([0.001, 0.01, 0.05])
    pvals, locations = get_pvalues_param_boot(p_values)
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_values, significance = reconstruct_param_boot(qvals, locations, p_values, alphas)

    print("making plot")
    plot_significance_matrix_param_boot(
        significance, p_values, task, quantity, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()