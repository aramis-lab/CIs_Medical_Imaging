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

import seaborn as sns
metric_labels = {
    'dsc': 'DSC',
    'iou': 'IoU',
    'nsd': 'NSD',
    'boundary_iou': 'Boundary IoU',
    'cldice': 'clDice',
    'assd': 'ASSD',
    'masd' : 'MASD',
    'hd': 'HD',
    'hd_perc': 'HD95',
    'balanced_accuracy': 'Balanced Accuracy',
    'ap': 'AP',
    'auc': 'AUC',
    'f1_score': 'F1 Score',
    'accuracy': 'Accuracy',
    "mcc": "MCC"
}

stat_labels = {
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'trimmed_mean': 'Trimmed Mean',
    'iqr_length': 'IQR Length'
}

method_labels = {
    "basic": "Basic",
    "percentile": "Percentile",
    "bca": "BCa",
    "delong": "DeLong",
    "logit_transform": "Logit Transform",
    "wilson": "Wilson",
    "agresti_coull" : "Agresti-Coull",
    "exact" : "Exact \n(Clopper-Pearson)",
    "wald" : 'Wald',
    "param_t" : "Parametric t",
    "param_z" : "Parametric z"
}

method_colors = {
    "basic": "#D4461F",
    "percentile": "#8E5EE8", 
    "bca" : "#FF9742",
    "wilson" : "#DFCF3E", 
    "agresti_coull" : "#5D9336", 
    "exact" : "#DB4ADB", 
    "wald" : "#367F9C",
    "param_t" : "#999999", 
    "param_z" : "#A7C7E7"}

def fit_ccp(classif_path,agreg_type):
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


def perform_pairwise_tests_micro_macro(df_fit_results, df_fit_results_macro):
    metrics_micro= df_fit_results['metric'].unique()
    metrics_macro= df_fit_results_macro['metric'].unique()
    couples=zip(metrics_micro, metrics_macro)
    
    n_values=df_fit_results['n'].unique()
    couple_names=[couple[0] + "_vs_" + couple[1] for couple in zip(metrics_micro, metrics_macro)]
    p_values = {str(n) : {couple : None for couple in couple_names} for n in n_values} 
    
   
    for n in n_values:
        print(n)
        for couple in zip(metrics_micro, metrics_macro):
            print(couple)
            couple_name=couple[0] + "_vs_" + couple[1]
            
            data_micro = df_fit_results[(df_fit_results["method"]=='percentile') & (df_fit_results['metric'] == couple[0])& (df_fit_results['n'] == n)]
            data_macro= df_fit_results_macro[(df_fit_results_macro["method"]=='percentile') & (df_fit_results_macro['metric'] == couple[1])& (df_fit_results_macro['n'] == n)]
            grp1 = (
                data_micro
                .groupby(['task', 'algo'])['value']
                .mean()
                .reset_index(name='beta1')
            )
            grp2 = (
                data_macro
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
            p_values[str(n)][couple_name] = pval

    return p_values


def get_pvalues_micro_macro(pvalues):
    pvals = []
    keys = []

    for n, metric_dict in pvalues.items():
        for couple, pval in metric_dict.items():
            
            if pval is not None and not np.isnan(pval):
                pvals.append(pval)
                keys.append((n, couple))

    pvals = np.asarray(pvals)
    return(pvals, keys)

def reconstruct_micro_macro(qvals, keys,pvalues,alphas):
    q_values = {
       
            n: {
                couple: None
                for couple in couple_dict
            }
            for n, couple_dict in pvalues.items()
        }
 

    significant = {
       
            n: {
                couple: None
                for couple in couple_dict
            }
            for n, couple_dict in pvalues.items()
        }
 

    
    for (n, couple), qval in zip(keys, qvals):
        q_values[n][couple] = qval
        significant[n][couple] = np.sum(qval < alphas)

    return q_values,significant




def tell_significance(p_values, alphas=np.array([0.001, 0.01, 0.05])):
   

    pvals = []
    keys = []

    for method, metric_dict in p_values.items():
        for couple, pval in metric_dict.items():
            
            if pval is not None and not np.isnan(pval):
                pvals.append(pval)
                keys.append((method, couple))

    pvals = np.asarray(pvals)
    # BH-FDR correction
    reject, qvals, _, _ = multipletests(
        pvals,
        method="fdr_bh"
    )

    # Reconstruct nested dictionaries
    q_values = {
       
            method: {
                couple: None
                for couple in couple_dict
            }
            for method, couple_dict in p_values.items()
        }
 

    significant = {
       
            method: {
                couple: None
                for couple in couple_dict
            }
            for method, couple_dict in p_values.items()
        }
 

    
    for (method, couple), qval in zip(keys, qvals):
        q_values[method][couple] = qval
        significant[method][couple] = np.sum(qval < alphas)

    return q_values,significant



def plot_significance_matrix_micro_macro(significance, p_values,n):

    plt.rcdefaults()
    metrics_all = list(iter(significance.keys()))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6*len(metrics_all)))

    

    global_matrix = []
    pval_matrix = []

    for metric in metrics_all:

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
            # global_matrix[i, j] = min(3, val) if val is not None else 0
        
    # pval_matrix = []

    # for metric in metrics_all:

    #     p_val = p_values.get(metric)
    #     if p_val is None:
    #         pval_matrix.append("")
    #     elif p_val < 0.05:
    #         pval_matrix.append("<0.05")
    #     else:
    #         pval_matrix.append(f"{p_val:.3f}")
            # if p_val is None:
            #     pval_row.append("0")
            # else:
            #     pval_row.append(
            #         f"{p_val:.6f}" if p_val >= 0.0001 else "<0.0001"
            #     )

   
    values = np.unique(global_matrix)
    # full mapping dictionary
    color_map_dict = {
        -1: '#000000',
        0: '#d9d9d9',
        1: '#fdae61',
       
    }
    # extract only the colors for values that appear
    colors = [color_map_dict[v] for v in values]

    # build colormap
    cmap = ListedColormap(colors)
    metlabels=[metric_labels.get(m, m) for m in metrics_all]
    # Plot heatma
    
    sns.heatmap(
        [global_matrix],
        annot=[pval_matrix],
        xticklabels=metlabels,
        cmap=cmap,
        cbar=False,
        ax=ax,
        fmt='',
        annot_kws={"fontsize": 12}
    )

    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.set_title('Test of micro vs macro', fontsize=16)
    legend_elements = [
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%'),
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
    plt.savefig(f'../clean_figs/supplementary/test_results/cov_micro_macro/{n}.pdf')
def main():

    path_micro='../results_metrics_classif'

    path_macro='../results_metrics_classif_macro'
    print('fitting ccp')
    df_fit_results=fit_ccp(path_micro,'micro')
    valid_fits=df_fit_results[df_fit_results['R2']<=0.1]
    df_fit_results_macro=fit_ccp(path_macro,'macro')
    valid_fits_macro=df_fit_results_macro[df_fit_results_macro['R2']<=0.1]
    
    print('performing tests')
    p_values,i=perform_pairwise_tests_micro_macro(valid_fits,valid_fits_macro)
    print(p_values)
    print('significance')
    q_values,significance=tell_significance(p_values)

    print('making plot')
    plot_significance_matrix_micro_macro(significance, p_values)

# main()