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
    

def plot_significance_matrix_bca(significance, p_values,task, to_overleaf=False):

    plt.rcdefaults()
    n_values=significance.keys()


    metrics = list(next(iter(significance.values())).keys())    

    
    fig, axes = plt.subplots(len(n_values),1, figsize=(15, 10), sharex=True) 

    
    
    
    for i,(n, sign_n) in enumerate(significance.items()):
     
        p_values_n=p_values[n]
        ax=axes[i]
        global_matrix = []
        pval_matrix = []
        for metric in metrics:

            val = sign_n.get(metric)
            p_val = p_values_n.get(metric)

            if val is None:
                global_matrix.append(-1)      # N/A
            elif val == 0:
                global_matrix.append(0)       # Not significant
            else:
                global_matrix.append(1)

            if p_val is None:
                pval_matrix.append("")
            else:
                pval_matrix.append(format_p(p_val))
                
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
        metlabels=[metric_labels.get(m, m) for m in metrics]
        # Plot heatma
        sns.heatmap(
            [global_matrix],
            annot=[pval_matrix],
            xticklabels=metlabels,
            yticklabels= [f"n={int(float(n))}"],
            cmap=cmap,
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1,
            fmt='',
            annot_kws={"fontsize": 12}
        )
        ax.tick_params(axis='x', rotation=45, labelsize=14)

        ax.tick_params(axis='y', rotation=45, labelsize=14)

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
        bbox_to_anchor=(1.02, 0.8),
        ncol=1,
        fontsize=10,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=10
    )
    
    plt.tight_layout()

    # fig.subplots_adjust(
    # left=0.08,   # more space for y tick labels
    # right=0.75,  # space for legend
    # top=1,
    # bottom=0,
    # wspace=0.07,
    # hspace=0.01
    # )
    output_path=f'../clean_figs/supplementary/test_results/cov_bca_{task}/all_n.pdf'
    fig.savefig(output_path)
    if to_overleaf:
        upload_to_overleaf(output_path, f"Preprint/supp_figs/Tests/cov_bca_{task}.pdf", commit_msg=f"Update figure test bca {task}")

    else:

        plt.show()
    
  