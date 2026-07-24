import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from .test_basic import format_p
from ..plot_utils import metric_labels, stat_labels, method_labels,upload_to_overleaf

import seaborn as sns


def fit_ccp(segm_path):
    results = []
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats = ["mean", "median", "std", "iqr_length"]
    methods=['basic', 'bca', 'percentile']
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
                                    'beta2': beta2[0],
                                    'R2': rel_error
                                }
                                results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)

def perform_pairwise_tests_spread_central(df_fit_results):
    
    metrics = df_fit_results['metric'].unique()
    methods = ['bca', 'basic', 'percentile']
    stats = [['mean', 'std'],['median', 'iqr_length']]
    stat_couples=['mean vs std','median vs iqr_length']
    n_values=df_fit_results['n'].unique()
    p_values = {str(n): {metric :  {stat : None for stat in stat_couples} for metric in metrics} for n in n_values}
    for n in n_values:
        for metric in metrics:
            for name, stat in zip(stat_couples,stats):

                data_central = df_fit_results[(df_fit_results["method"]=='percentile') & (df_fit_results["stat"]==stat[0]) & (df_fit_results['metric'] == metric)& (df_fit_results['n'] == n)]
                data_disp= df_fit_results[(df_fit_results["method"]=='percentile') & (df_fit_results["stat"]==stat[1]) & (df_fit_results['metric'] == metric) & (df_fit_results['n'] == n)]
             
                grp1 = (
                    data_central
                    .groupby(['task', 'algo'])['value']
                    .mean()
                    .reset_index(name='beta1')
                )
                grp2 = (
                    data_disp
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
                p_values[str(n)][metric][name] = pval

    return p_values

def get_pvalues_spread_central(p_values):
    pvals = []
    keys = []

    for n, metric_dict in p_values.items():
        for metric, stat_dict in metric_dict.items():
            for stat, pval in stat_dict.items():
                if pval is not None:
                    pvals.append(pval)
                    keys.append((n, metric, stat))

    pvals = np.asarray(pvals)
    return(pvals, keys)

def reconstruct_spread_central(qvals, keys,p_values,alphas):
    q_values = {
        n: {
            metric: {
                stat: None
                for stat in stat_dict
            }
            for metric, stat_dict in metric_dict.items()
        }
        for n, metric_dict in p_values.items()
    }

    significant = {
        n: {
            metric: {
                stat: {}
                for stat in stat_dict
            }
            for metric, stat_dict in metric_dict.items()
        }
        for n, metric_dict in p_values.items()
    }

    for (n, metric, stat), qval in zip(keys, qvals):
        q_values[n][metric][stat] = qval
        significant[n][metric][stat] = np.sum(qval < alphas)

    return q_values,significant


def tell_significance(p_values, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []

    for metric, metric_dict in p_values.items():
        for method, method_dict in metric_dict.items():
            for stat, pval in method_dict.items():
                if pval is not None:
                    pvals.append(pval)
                    keys.append((metric, method, stat))

    pvals = np.asarray(pvals)

    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )

    # Reconstruct nested dictionaries
    q_values = {
        metric: {
            method: {
                stat: None
                for stat in stat_dict
            }
            for method, stat_dict in method_dict.items()
        }
        for metric, method_dict in p_values.items()
    }

    significant = {
        metric: {
            method: {
                stat: None
                for stat in stat_dict
            }
            for method, stat_dict in method_dict.items()
        }
        for metric, method_dict in p_values.items()
    }

    for (metric, method, stat), qval in zip(keys, qvals):
        q_values[metric][method][stat] = qval
        significant[metric][method][stat] = np.sum(qval < alphas)

    return q_values,significant


def plot_significance_matrix_spread_central(significance, p_values, to_overleaf=False):

    plt.rcdefaults()
    n_values=significance.keys()

    metric_order = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    
    fig, axes = plt.subplots(1,len(n_values), figsize=(18,15), sharey=True)
    
    

    
    for i,(n, sign_n) in enumerate(significance.items()):
        ax=axes[i]
        pvalues_n=p_values[n]
        
        metrics =sign_n.keys()
        stats = list(next(iter(sign_n.values())).keys())
        global_matrix = np.zeros((len(metric_order), len(stats)))
        pval_matrix = []
        for j, metric in enumerate(metrics):
        
            pval_row=[]
            for k,stat in enumerate(stats):
                
                val = sign_n.get(metric, {}).get(stat)
                p_val = pvalues_n.get(metric, {}).get(stat)

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
        metlabels=[metric_labels.get(m, m) for m in metric_order]
        # Plot heatma
        sns.heatmap(
            global_matrix,
            annot=pval_matrix,
            xticklabels=stats,
            yticklabels=metlabels,
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt='',
            annot_kws={"fontsize": 10}
        )
        ax.tick_params(axis='x', rotation=90, labelsize=14)

        ax.tick_params(axis='y', rotation=0, labelsize=14,)

        ax.set_title(f'n={n}', fontsize=16)

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
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=16
    )
    plt.tight_layout()
    # fig.subplots_adjust(
    # left=0.1,   # more space for y tick labels
    # right=0.75,  # space for legend
    # # top=1,
    # # bottom=0,
    # wspace=0,
    # # hspace=0.01
    # )
    output_path=f'../clean_figs/supplementary/test_results/cov_spread_central/all_n.pdf'
    fig.savefig(output_path)
    if to_overleaf: 
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/cov_spread_central.pdf", commit_msg="Update figure test spread central")
   
    else:
        plt.show()

 

def main():
    segm_path='../../../../results_metrics_segm'
    print('fitting ccp')
    df_fit_results=fit_ccp(segm_path)
    valid_fits=df_fit_results[df_fit_results['R2']<=0.1]
    print('performing tests')
    p_values=perform_pairwise_tests_spread_central(valid_fits)
    print('significance')
    q_values,significance=tell_significance(p_values)

    print('making plot')
    plot_significance_matrix_spread_central(significance, p_values)

# main()
    