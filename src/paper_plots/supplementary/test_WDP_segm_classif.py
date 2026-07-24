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


def plot_significance_matrix_wdp_segm_classif(significance,p_values,task, to_overleaf=False):
    plt.rcdefaults()
    n_values=significance.keys()
    

    fig, axes = plt.subplots(1,len(n_values), figsize=(18, 12), sharey=True)

    
    
    for i,(n, sign_n) in enumerate(significance.items()):
        ax= axes[i]
        p_values_n=p_values[n]
        metrics_classif =sign_n.keys()
        metrics_segm = list(next(iter(sign_n.values())).keys())
        global_matrix = np.zeros((len(metrics_segm), len(metrics_classif)))
        pval_matrix= []
        for j, metric1 in enumerate(metrics_segm):
            pval_row=[]
            for k, metric2 in enumerate(metrics_classif):
                val = sign_n.get(metric2, {}).get(metric1)
                
                p_val = p_values_n.get(metric2, {}).get(metric1)

                if val is None:
                    global_matrix[j,k]=(-1)      # N/A
                elif val == 0:
                    global_matrix[j,k]=(0)       # Not significant
                else:
                    global_matrix[j,k]=(1)

                if p_val is None:
                    pval_row.append("")
                
                else:
                    pval_row.append(format_p(p_val))
            pval_matrix.append(pval_row)
    
        
        values = np.unique(global_matrix)

        # full mapping dictionary
        color_map_dict = {
       -1: "#F5F5F5",      
            0: "#F8CC80FF",     
            1: "#D55E00"
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
            square=True,
            linewidths=1,
            fmt='',
            annot_kws={"fontsize": 5}
        )
        ax.tick_params(axis='x', rotation=90, labelsize=10)

        ax.tick_params(axis='y', rotation=0, labelsize=10)

        ax.set_title(f"n={int(float(n))}", fontsize=12)

        legend_elements = [
                mpatches.Patch(facecolor="#D55E00",
                            edgecolor='k',
                            label="Significant \n(FDR-adjusted p < 0.05)"),
                mpatches.Patch(facecolor="#F8CC80FF",
                            edgecolor='k',
                            label="Not significant"),
                mpatches.Patch(facecolor="#F5F5F5",
                            edgecolor='k',
                            label="Not available"),
            ]
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=12,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=12
    )
    
    
    # fig.subplots_adjust(
    # left=0.07,   # more space for y tick labels
    # right=0.8,  # space for legend
    # # top=1,
    # # bottom=0,
    # wspace=0.05,
    # hspace=0
    # )
    plt.tight_layout()
    output_path=f'../clean_figs/supplementary/test_results/width_segm_classif_{task}/all_n.pdf'
    fig.savefig(output_path)
    if to_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/width_segm_classif_{task}.pdf", commit_msg="Update figure test width segm classif")
    else:

        plt.show()
    



def main():
    segm_path='../results_metrics_segm'
    classif_path='../results_metrics_classif'
    print('performing tests')
    p_values=perform_pairwise_tests_wdp_segm_classif(segm_path, classif_path)
    print('significance')
    significance=tell_significance(p_values)
    print('making plot')
    plot_significance_matrix_wdp_segm_classif(significance, p_values)

