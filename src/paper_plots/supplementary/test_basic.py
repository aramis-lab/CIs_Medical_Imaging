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



def plot_significance_matrix_basic(significance, p_values, to_overleaf=False):

    plt.rcdefaults()
    
    metric_order = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    main_methods = [ 'bca', 'percentile']
    n_values = list(significance.keys())
    metrics_all = metric_order
    stats= list(next(iter(significance.values())).keys())    
    
    fig, axes = plt.subplots(len(n_values), len(stats),figsize=(18 ,12* len(n_values) ), sharey=True,width_ratios=(2,1,1,1,1) )
    for i,(n,sign_n) in enumerate(significance.items()):
       
        p_values_n=p_values[n]
        for row, stat in enumerate(stats):
            if stat=='mean':
                methods=main_methods+['param_z', 'param_t']
            else:
                methods=main_methods
            ax = axes[i,row] 

            # Extract significance for the specific method and stat
            stat_significance = sign_n.get(stat, {})
            global_matrix = np.zeros((len(metrics_all), len(methods)))
            
            for k, metric in enumerate(metrics_all):
        
                for j, method in enumerate(methods):
                    val = stat_significance.get(metric, {}).get(method, None)
                    if val is None:
                        global_matrix[k, j] = -1      # N/A
                    elif val == 0:
                        global_matrix[k, j] = 0       # Not significant
                    else:
                        global_matrix[k, j] = 1
                    # global_matrix[i, j] = min(1, val) if val is not None else -1
                

            pval_matrix = []

            for metric in metrics_all:
                pval_row = []

                for method in methods:

                    p_val = p_values_n.get(stat, {}).get(metric, {}).get(method)
                    if p_val is None:
                        pval_row.append("")
                    
                    else:
                        pval_row.append(format_p(p_val))
                    # if p_val is None:
                    #     pval_row.append("0")
                    # else:
                    #     pval_row.append(
                    #         f"{p_val:.6f}" if p_val >= 0.05 else "<0.05"
                    #     )

                pval_matrix.append(pval_row)
            
            values = np.unique(global_matrix)

            # full mapping dictionary
            color_map_dict = {
        
            -1: "#F5F5F5",      # missing
            0: "#F8CC80FF",      # non-significant
            1: "#D55E00",
        
        }   
            
            # extract only the colors for values that appear
            colors = [color_map_dict[v] for v in values]

            # build colormap
            cmap = ListedColormap(colors)
            mabels = [method_labels.get(m, m) for m in methods]
            metlabels=[metric_labels.get(m, m) for m in metrics_all]
            # Plot heatma
            sns.heatmap(
                global_matrix,
                annot=pval_matrix,
                xticklabels=mabels,
                yticklabels=metlabels,
                cmap=cmap,
                cbar=False,
                ax=ax,
                fmt='',
                linewidths=1,
                square=True,
                linecolor="white",
                annot_kws={"fontsize": 14}
            )
            ax.tick_params(axis='x', rotation=45, labelsize=14)

            ax.tick_params(axis='y', rotation=45, labelsize=14)
            if stat=='mean':
                ax.set_title(f"{stat_labels[stat]}, n={n}", fontsize=14)
            else:
                ax.set_title(f"{stat_labels[stat]}", fontsize=14)

        legend_elements = [
                    mpatches.Patch(facecolor="#D55E00",
                                edgecolor='k',
                                label="Significant \n (FDR-adjusted p < 0.05)"),
                    mpatches.Patch(facecolor="#F8CC80FF",
                                edgecolor='k',
                                label="Not significant"),
                    mpatches.Patch(facecolor="#F5F5F5",
                                edgecolor='k',
                                label="Not available"),
                ]
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            fontsize=16,
            frameon=True,
            title="Significance levels \nwith FDR correction",
            title_fontsize=16
        )
    fig.subplots_adjust(
    # left=0.08,   
    right=0.75,  
    top=1,
    bottom=0,
  
    hspace=0
    )
    output_path=f'../clean_figs/supplementary/test_results/cov_basic_segm/all_n.pdf'
    fig.savefig(output_path)
    if to_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/cov_basic_segm.pdf", commit_msg=f"Update figure test basic segm")
    else:
        plt.show()
    
    
def main():
    segm_path='../../../../results_metrics_segm'
    print('fitting ccp')
    df_fit_results=fit_ccp(segm_path)
    valid_fits=df_fit_results[df_fit_results['R2']<=0.1]
    print('performing tests')
    p_values=perform_pairwise_tests_basic(valid_fits)
    print('significance')
    q_values,significance=tell_significance(p_values)

    print('making plot')
    plot_significance_matrix_basic(significance, p_values)

# main()