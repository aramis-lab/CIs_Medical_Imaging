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

import seaborn as sns
# metric_labels = {
#     'dsc': 'DSC',
#     'iou': 'IoU',
#     'nsd': 'NSD',
#     'boundary_iou': 'Boundary IoU',
#     'cldice': 'clDice',
#     'assd': 'ASSD',
#     'masd' : 'MASD',
#     'hd': 'HD',
#     'hd_perc': 'HD95',
#     'balanced_accuracy': 'Balanced Accuracy',
#     'ap': 'AP',
#     'auc': 'AUC',
#     'f1_score': 'F1 Score',
#     'accuracy': 'Accuracy',
#     "mcc": "MCC"
# }

# stat_labels = {
#     'mean': 'Mean',
#     'median': 'Median',
#     'std': 'Standard Deviation',
#     'trimmed_mean': 'Trimmed Mean',
#     'iqr_length': 'IQR Length'
# }

# method_labels = {
#     "basic": "Basic",
#     "percentile": "Percentile",
#     "bca": "BCa",
#     "delong": "DeLong",
#     "logit_transform": "Logit Transform",
#     "wilson": "Wilson",
#     "agresti_coull" : "Agresti-Coull",
#     "exact" : "Exact \n(Clopper-Pearson)",
#     "wald" : 'Wald',
#     "param_t" : "Parametric t",
#     "param_z" : "Parametric z"
# }

# method_colors = {
#     "basic": "#D4461F",
#     "percentile": "#8E5EE8", 
#     "bca" : "#FF9742",
#     "wilson" : "#DFCF3E", 
#     "agresti_coull" : "#5D9336", 
#     "exact" : "#DB4ADB", 
#     "wald" : "#367F9C",
#     "param_t" : "#999999", 
#     "param_z" : "#A7C7E7"}

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
                                    'beta2': beta2[0],
                                    'R2': rel_error
                                }
                                results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return(df_fit_results)


def perform_pairwise_tests_basic(df_fit_results):

    metrics = df_fit_results['metric'].unique()
    methods = ['bca', 'percentile', 'param_z', 'param_t']
    stats = df_fit_results['stat'].unique()
    p_values = {stat : {metric : {m : None for m in methods} for metric in metrics} for stat in stats}

    for stat in stats:
       
        for metric in metrics:
           
            for j in methods:

                
                if (j in ['param_z', 'param_t']) & (stat!='mean'):
                    continue 
                data_basic = df_fit_results[(df_fit_results["method"]=='basic') & (df_fit_results["stat"]==stat) & (df_fit_results['metric'] == metric)]
                data_methods= df_fit_results[(df_fit_results["method"]==j) & (df_fit_results["stat"]==stat) & (df_fit_results['metric'] == metric)]
             
                grp1 = (
                    data_basic
                    .groupby(['task', 'algo'])['beta2']
                    .mean()
                    .reset_index(name='beta1')
                )
                grp2 = (
                    data_methods
                    .groupby(['task', 'algo'])['beta2']
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
                        n_resamples=50000,
                        alternative='greater'
                    )
                    pval = res.pvalue
                p_values[stat][metric][j] = pval
               

    return p_values

def get_pvalues_basic(p_values):
    pvals = []
    keys = []

    for stat, metric_dict in p_values.items():
        for metric, method_dict in metric_dict.items():
            for method, pval in method_dict.items():

                if pval is not None and not np.isnan(pval):
                    pvals.append(pval)
                    keys.append((stat, metric, method))

    pvals = np.asarray(pvals)
    return(pvals, keys)


def reconstruct_basic(qvals, keys,pvalues,alphas):
    q_values = {
        stat: {
            metric: {
                method: None
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for stat, metric_dict in pvalues.items()
    }

    significant = {
        stat: {
            metric: {
                method: False
                for method in method_dict
            }
            for metric, method_dict in metric_dict.items()
        }
        for stat, metric_dict in pvalues.items()
    }
    
    for (stat, metric, method), qval in zip(keys, qvals):
        q_values[stat][metric][method] = qval
        significant[stat][metric][method] = np.sum(qval < alphas)

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
                method: False
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



def plot_significance_matrix_basic(significance, p_values):

    plt.rcdefaults()
    
    metric_order = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    main_methods = [ 'bca', 'percentile']
    stats = list(significance.keys())
    metrics_all = metric_order
    fig, axes = plt.subplots(1,len(stats), figsize=(10 * len(stats),12), width_ratios=[2,1,1,1,1])
    
    for row, stat in enumerate(stats):
        if stat=='mean':
            methods=main_methods+['param_z', 'param_t']
        else:
            methods=main_methods
        ax = axes[row] 

        # Extract significance for the specific method and stat
        stat_significance = significance.get(stat, {})
        global_matrix = np.zeros((len(metrics_all), len(methods)))
        
        for i, metric in enumerate(metrics_all):
       
            for j, method in enumerate(methods):
                val = stat_significance.get(metric, {}).get(method, None)
                global_matrix[i, j] = min(3, val) if val is not None else 0
            

        pval_matrix = []

        for metric in metrics_all:
            pval_row = []

            for method in methods:

                p_val = p_values.get(stat, {}).get(metric, {}).get(method, None)
                if p_val is None:
                    pval_row.append("0")
                else:
                    pval_row.append(
                        f"{p_val:.6f}" if p_val >= 0.0001 else "<0.0001"
                    )

            pval_matrix.append(pval_row)
        
        values = np.unique(global_matrix)

        # full mapping dictionary
        color_map_dict = {
            -1: '#000000',
            0: '#d9d9d9',
            1: '#fee08b',
            2: '#fdae61',
            3: '#d73027',
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
            annot_kws={"fontsize": 16}
        )
        ax.tick_params(axis='x', rotation=45, labelsize=14)

        ax.tick_params(axis='y', rotation=45, labelsize=14)

        ax.set_title(f"Stat : {stat_labels[stat]}", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1%'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%'),
        mpatches.Patch(facecolor='#fee08b', edgecolor='k', label='10%'),
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
    plt.savefig('../clean_figs/supplementary/test_basic_segm.pdf')
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