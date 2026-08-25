import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
# from ..plot_utils import metric_labels, stat_labels, method_labels
# from mlxtend.evaluate import permutation_test
from ..plot_utils import method_labels, method_colors, metric_labels, stat_labels, upload_to_overleaf

import seaborn as sns

def fit_ccp(classif_path,agreg_type):
    results = []
    if agreg_type=='micro':

        metrics= ["accuracy","ap", "auc", "f1_score"]
    else:
        metrics= ["balanced_accuracy","ap", "auc", "f1_score"]
    methods=['basic', 'bca', 'percentile',"wilson","agresti_coull" ,"wald"]
    for metric in metrics:
        
        path=os.path.join(classif_path,f'aggregated_results_{metric}.csv')
        
        df_metric_stat = pd.read_csv(path)
        for task in df_metric_stat['subtask'].unique():
            df_task = df_metric_stat[df_metric_stat['subtask'] == task]
            for algo in df_task['alg_name'].unique():
                    df_algo = df_task[df_task['alg_name'] == algo]
                    for method in methods:
                        if (method in ["wilson","agresti_coull" ,"wald"]) & (metric!='accuracy'):
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
                                'method': method,
                                'beta2': beta2[0],
                                'R2': rel_error
                            }
                            results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)


def perform_pairwise_tests_basic_classif(df_fit_results):
    
    
    metrics = df_fit_results['metric'].unique()
    methods = ['bca', 'percentile',"wald", "exact","agresti_coull", "wilson"]
    n_values = df_fit_results['n'].unique()
    p_values = {str(n):{metric : {m : None for m in methods} for metric in metrics} for n in n_values}

    for n in n_values: 
        for metric in metrics:
            # print(metric)
            for j in ['bca', 'percentile',"wald", "exact","agresti_coull", "wilson"]:
            
                
                if (j in ["wilson","agresti_coull" ,"wald", "exact"]) & (metric!='accuracy'):
                    continue
                data_basic = df_fit_results[(df_fit_results["method"]=='basic') & (df_fit_results['metric'] == metric)& (df_fit_results['n']==n)]
                
                data_methods= df_fit_results[(df_fit_results["method"]==j) & (df_fit_results['metric'] == metric)& (df_fit_results['n']==n)]
                # print(data_basic['beta2'].mean(),data_methods['beta2'].mean())
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
                    # res = permutation_test(
                    #     merged['beta1'].to_numpy(), merged['beta2'].to_numpy(),
                    #     paired=True,
                    #     func=statistic,
                    #     seed=0, num_rounds=50000
                    # )

                    pval = res.pvalue
                p_values[str(n)][metric][j] = pval

    return p_values

def get_pvalues_basic_classif(pvalues):

    pvals = []
    keys = []
    for n, metric_dict in pvalues.items():
        for metric, method_dict in metric_dict.items():
            for method, pval in method_dict.items():

                if pval is not None and not np.isnan(pval):
                    pvals.append(pval)
                    keys.append( (n,metric, method))

    pvals = np.asarray(pvals)
    return(pvals, keys)

def reconstruct_basic_classif(qvals, keys,pvalues,alphas):
    q_values = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        } for n, metric_dict in pvalues.items()
    }
 

    significant = {
        n: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for n, metric_dict in pvalues.items()
    }
    
    for (n, metric, method), qval in zip(keys, qvals):
        q_values[n][metric][method] = qval
        if qval is None:
            significant[n][metric][method]=qval
        else:
            significant[n][metric][method] = np.sum(qval < alphas)

    return q_values,significant


def tell_significance(p_values, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []

    for metric, method_dict in p_values.items():
        for method, pval in method_dict.items():

            if pval is not None and not np.isnan(pval):
                pvals.append(pval)
                keys.append((metric, method))

    pvals = np.asarray(pvals)

    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )

    # Reconstruct nested dictionaries
    q_values = {
       
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in p_values.items()
        }
 

    significant = {
      
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in p_values.items()
       
    }
    
    for (metric, method), qval in zip(keys, qvals):
        print(qval)
        if qval is None:
     
            significant[metric][method]=qval
        else:
            significant[metric][method] = np.sum(qval < alphas)

        q_values[metric][method] = qval


    return q_values,significant


def format_p(p):
    if p is None:
        return ""
    elif p < 1e-4:
        return f"{p:.1e}"      # 2.3e-06
    elif p < 1e-3:
        return f"{p:.4f}"      # 0.0007
    elif p < 0.01:
        return f"{p:.3f}"      # 0.008
    else:
        return f"{p:.2f}"  
    
        # 0.03, 0.27
def plot_significance_matrix_basic_classif(
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

    main_methods = ["bca", "percentile"]
    if agreg_type == "macro":
        metrics_all = ["balanced_accuracy", "ap", "auc", "f1_score"]
        methods = main_methods
        annot_fontsize = 8
        tick_fontsize = 12
        xtick_rotation = 45
        wspace = 0.02
    else:
        metrics_all = ["accuracy", "ap", "auc", "f1_score"]
        methods = main_methods + ["wilson", "agresti_coull", "wald"]
        annot_fontsize = 4
        tick_fontsize = 7
        xtick_rotation = 90
        wspace = 0.03

    color_map_dict = {
        -1: "#F5F5F5",   # not available
        0: "#F8CC80FF",  # not significant
        1: "#D55E00",    # significant
    }

    n_values = list(significance.keys())
    method_ticklabels = [method_labels.get(m, m) for m in methods]
    metric_ticklabels = [metric_labels.get(m, m) for m in metrics_all]

    fig, axes = plt.subplots(1, len(n_values), figsize=(18, 12), sharey=True)
    axes = np.atleast_1d(axes)

    for col, n in enumerate(n_values):
        ax = axes[col]
        sign_n = significance[n]
        p_values_n = p_values[n]

        global_matrix = np.zeros((len(metrics_all), len(methods)))
        pval_matrix = []

        for i, metric in enumerate(metrics_all):
            pval_row = []
            for j, method in enumerate(methods):
                val = sign_n.get(metric, {}).get(method, None)
                if val is None:
                    global_matrix[i, j] = -1
                elif val == 0:
                    global_matrix[i, j] = 0
                else:
                    global_matrix[i, j] = 1

                p_val = p_values_n.get(metric, {}).get(method)
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
            annot_kws={"fontsize": annot_fontsize},
        )
        ax.tick_params(axis="x", rotation=xtick_rotation, labelsize=tick_fontsize)
        ax.tick_params(axis="y", rotation=45, labelsize=tick_fontsize)
        ax.set_title(f"n={int(float(n))}", fontsize=12)

    legend_elements = [
        mpatches.Patch(facecolor="#D55E00", edgecolor="k",
                       label="Significant (FDR-adjusted p < 0.05)"),
        mpatches.Patch(facecolor="#F8CC80FF", edgecolor="k",
                       label="Not significant"),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor="k",
                       label="Not available"),
    ]
    legend = axes[-1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.8),
        ncol=1,
        fontsize=10,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=10,
    )
    # keep the legend out of the tight_layout computation but reserve room for it
    legend.set_in_layout(False)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    # re-apply the desired spacing between the heatmaps (tight_layout resets it)
    fig.subplots_adjust(wspace=wspace)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)

    if upload_overleaf:
        upload_to_overleaf(
            output_path,
            f"Preprint/supp_figs/tests/cov_basic_classif_{agreg_type}.pdf",
            commit_msg="Update figure test basic classif",
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Supp Figure significance matrix of basic vs other methods for classification."
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
        root_folder, f"clean_figs/supplementary/test_results/cov_basic_classif_{agreg_type}/all_n.pdf"
    )

    if agreg_type == "micro":
        classif_path = os.path.join(root_folder, "results_metrics_classif")
    else:
        classif_path = os.path.join(root_folder, "results_metrics_classif_macro")

    print("fitting ccp")
    df_fit_results = fit_ccp(classif_path, agreg_type)
    valid_fits = df_fit_results[df_fit_results["R2"] <= 0.2]
    print("performing tests")
    p_values = perform_pairwise_tests_basic_classif(valid_fits)
    print("significance")
    q_values, significance = tell_significance(p_values)

    print("making plot")
    plot_significance_matrix_basic_classif(
        significance, p_values, agreg_type, output_path, upload_overleaf=args.upload_overleaf
    )


if __name__ == "__main__":
    main()