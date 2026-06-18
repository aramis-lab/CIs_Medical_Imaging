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
from .make_fit import df_fit_results, df_fit_results_micro, df_fit_results_macro,df_fit_results_wdp,df_fit_results_classif_wdp, segm_path, micro_path

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

    segm_metrics = df_fit_results['metric'].unique()
    classif_metrics = df_fit_results_classif['metric'].unique()
    methods = ['basic', 'bca', 'percentile']
    stats = df_fit_results['stat'].unique()
    p_values = {met : {s : {m : {m2: None for m2 in segm_metrics} for m in classif_metrics} for s in stats} for met in methods}

    for method in methods:

        for stat in stats:
        
            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                continue
            for metric1 in classif_metrics:
                for metric2 in segm_metrics:
                    data_metric1 = df_fit_results_classif[(df_fit_results_classif["method"]==method) & (df_fit_results_classif['metric'] == metric1)]
                    data_metric2 = df_fit_results[(df_fit_results["method"]==method) & (df_fit_results['metric'] == metric2)& (df_fit_results['stat'] == stat)]
                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)

                    res = permutation_test(
                        (data_metric1['width_decay_pace'].to_numpy(), data_metric2['width_decay_pace'].to_numpy()),
                        statistic,
                        vectorized=False,
                        n_resamples=50000,
                        alternative='greater'
                    )
                    pval = res.pvalue

                    p_values[method][stat][metric1][metric2] = pval

    return p_values
def get_pvalues_wdp_segm_classif(p_vals):
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
    return(pval_array, locations)

def reconstruct_wdp_segm_classif(qvals, locations,p_vals,alphas):
    significance = {
        method: {
            stat: {
                metric1: {}
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
                metric1: {}
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


def plot_significance_matrix_wdp_segm_classif(significance,p_values):

    plt.rcdefaults()

    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metrics_classif = list(next(iter(next(iter(significance.values())).values())).keys())
    metrics_segm = list(next(iter(next(iter(next(iter(significance.values())).values())).values())).keys())

    fig, axes = plt.subplots(len(methods), len(stats), figsize=(15 * len(stats), 12 * len(methods)))

    for col, stat in enumerate(stats):
        for row, method in enumerate(methods):
            if len(stats) == 1 and len(methods) == 1:
                ax = axes
            elif len(stats) == 1 or len(methods) == 1:
                ax = axes[max(row, col)]
            else:
                ax = axes[row, col]

            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                ax.axis('off')
                continue

            # Extract significance for the specific method and stat
            method_stat_significance = significance.get(method, {}).get(stat, {})
            global_matrix = np.zeros((len(metrics_segm), len(metrics_classif)))

            for i, metric1 in enumerate(metrics_segm):
                for j, metric2 in enumerate(metrics_classif):
                    val = method_stat_significance.get(metric2, {}).get(metric1, None)
                    global_matrix[i, j] = min(3, val) if val is not None else 0

            # Create p_val matrix for heatap 
            pval_matrix = []
            for i, metric1 in enumerate(metrics_segm):
                pval_row = []
                for j, metric2 in enumerate(metrics_classif):
                    p_val = p_values.get(method, {}).get(stat, {}).get(metric2, {}).get(metric1, None)
                    if p_val is not None:
                        pval_row.append(f"{p_val:.6f}" if p_val >= 0.0001 else "<0.0001")
                    else:
                        pval_row.append("0")
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
                fmt='',
                annot_kws={"fontsize": 16}
            )
            ax.tick_params(axis='x', rotation=45, labelsize=14)

            ax.tick_params(axis='y', rotation=45, labelsize=14)

            ax.set_title(f"Stat : {stat_labels[stat]}, Method: {method_labels[method]}", fontsize=16)

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1%'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5%'),
        mpatches.Patch(facecolor='#fee08b', edgecolor='k', label='10%'),
        mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant')
    ]
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=16,
        frameon=True,
        title="Significance levels \nwith FDR correction",
        title_fontsize=16
    )
    plt.tight_layout()
    
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    plt.savefig('../clean_figs/supplementary/test_WDP_segm_classif.pdf')
    plt.show()



def main():
    segm_path='../results_metrics_segm'
    classif_path='../results_metrics_classif'
    print('fitting ccp')
    # df_fit_results=fit_wdp_segm(segm_path)
    valid_fits=df_fit_results_wdp[df_fit_results_wdp['R2']<=0.1]
    # df_fit_results_classif=fit_wdp_classif(classif_path, 'micro')
    valid_fits_classif=df_fit_results_classif_wdp[df_fit_results_classif_wdp['R2']<=0.1]
    print('performing tests')
    p_values=perform_pairwise_tests_wdp_segm_classif(valid_fits, valid_fits_classif)
    print('significance')
    significance=tell_significance(p_values)
    print('making plot')
    plot_significance_matrix_wdp_segm_classif(significance, p_values)

