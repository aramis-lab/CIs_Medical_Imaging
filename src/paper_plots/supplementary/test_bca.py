import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from ..plot_utils import metric_labels, stat_labels, method_labels
import scipy
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
            metric: {}
                for metric in metric_dict
        }
        for n, metric_dict in pvalues.items()
    }

    significant = {
        n: {
                metric: {}
                for metric in metric_dict
        }
        for n, metric_dict in pvalues.items()
    }
    
    for (n, metric), qval in zip(keys, qvals):
        q_values[n][metric] = qval
        significant[n][metric] = np.sum(qval < alphas)

    return q_values,significant
    

def plot_significance_matrix_bca(significance, p_values, n, task):

    plt.rcdefaults()
    
    
    metrics = list(significance.keys())
    fig, ax = plt.subplots(1,1, figsize=(10*len(metrics) ,12))
    
    global_matrix = []
    pval_matrix = []

    for metric in metrics:

        val = significance.get(metric)
        p_val = p_values.get(metric)

        if val is None:
            global_matrix.append(-1)      # N/A
        elif val == 0:
            global_matrix.append(0)       # Not significant
        else:
            global_matrix.append(1)

        if p_val is None:
            pval_matrix.append("")
        elif p_val < 0.05:
            pval_matrix.append("<0.05")
        else:
            pval_matrix.append(f"{p_val:.3f}")
            
        values = np.unique(global_matrix)

        # full mapping dictionary
        color_map_dict = {
       
        -1: "#161515",
        0: "#d9d9d9",
        1: "#fdae61",
       
    }   
        
        
        # extract only the colors for values that appear
        colors = [color_map_dict[v] for v in values]

        # build colormap
        cmap = ListedColormap(colors)
        metlabels=[metric_labels.get(m, m) for m in metrics]
        # Plot heatma
        sns.heatmap(
            [global_matrix],
            annot=[pval_matrix],
            xticklabels=metlabels,
            cmap=cmap,
            cbar=False,
            ax=ax,
            fmt='',
            annot_kws={"fontsize": 16}
        )
        ax.tick_params(axis='x', rotation=45, labelsize=14)

        ax.tick_params(axis='y', rotation=45, labelsize=14)

        ax.set_title(f"Test cov BCa > cov percentile", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor="#fdae61", edgecolor='k', label='5%'),
        mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant')
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=16
    )
    plt.tight_layout()
    plt.savefig(f'../clean_figs/supplementary/test_results/cov_bca_{task}/{n}.pdf')
  