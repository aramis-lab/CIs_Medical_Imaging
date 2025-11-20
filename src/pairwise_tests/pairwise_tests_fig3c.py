import pandas as pd
import numpy as np
import os
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

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
    'accuracy': 'Accuracy'
}

def extract_coverage_data(folder_path, file_prefix, metrics, stats, methods):

    all_values=[]
    for metric in metrics:
        for stat in stats:

            file_path = os.path.join(folder_path, f"{file_prefix}_{metric}_{stat}.csv")
            data = pd.read_csv(file_path)
            n_subset=data['n'].unique()
            tasks=data['subtask'].unique()
    
            algos=data['alg_name'].unique()
            for task in tasks: 
                data_task=data[data['subtask']==task]
                for algo in algos: 
                    data_algo=data_task[data_task['alg_name']==algo]
                    
                    for n in n_subset:  # Show only selected n values
                        data_n = data_algo[data_algo['n'] == n]
                        method_dict = {
                            'basic': 'contains_true_stat_basic',
                            'bca': 'contains_true_stat_bca',
                            'percentile': 'contains_true_stat_percentile',
                        }

                        # Add parametric methods only for stat == 'mean'
                        if stat == 'mean':
                            method_dict.update({
                                'param_z': 'contains_true_stat_param_z',
                                'param_t': 'contains_true_stat_param_t'
                            })
                        for method, col in method_dict.items():
                            if method in methods:
                                for val in data_n[col]:
                                    
                                    all_values.append({
                                        'metric': metric,
                                        'stat': stat,
                                        'task':task, 
                                        'algo':algo,
                                        'n': n,
                                        'method': method,
                                        'coverage': val
                                    })
    df_segm=pd.DataFrame(all_values)
    return df_segm

def perform_fits(df_segm):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
                df_metric_stat = df_algo[(df_algo['metric'] == metric) & (df_algo['stat']=='mean')]
                for method in df_metric_stat['method'].unique():
                    df_metric_stat_method = df_metric_stat[df_metric_stat['method'] == method]
                    df_metric_stat_method = df_metric_stat_method.sort_values(by='n')
                    n_values = df_metric_stat_method['n'].to_numpy()
                    coverages = df_metric_stat_method['coverage'].to_numpy()
                    Y = 0.95 - coverages
                    X = np.vstack([1/n_values]).T
                    beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                    rel_error = np.linalg.norm(X @ beta2 - Y) / np.linalg.norm(Y)
                    new_row = {
                        'task': task,
                        'algo': algo,
                        'metric': metric,
                        'stat': "mean",
                        'method': method,
                        'beta2': beta2[0],
                        'R2': rel_error
                    }
                    results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results


def perform_pairwise_tests(df_fit_results, metrics):
    
    p_values = {m : {m2: None for m2 in metrics} for m in metrics}

    for i in range(len(metrics)):

        for j in range(i + 1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]

            data_metric1 = df_fit_results[df_fit_results['metric'] == metric1]
            data_metric2 = df_fit_results[df_fit_results['metric'] == metric2]

            grp1 = (
                data_metric1
                .groupby(['task', 'algo'])['beta2']
                .mean()
                .reset_index(name='beta1')
            )
            grp2 = (
                data_metric2
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
                    statistic,
                    vectorized=False,
                    n_resamples=100000,
                    alternative='two-sided'
                )
                pval = res.pvalue

            p_values[metric1][metric2] = pval
            p_values[metric2][metric1] = pval

    return p_values

def tell_significance(p_vals, alphas=np.array([0.001, 0.01, 0.05]), bonferroni_correction=True):
    
    m = len(p_vals)
    num_comparisons = m - 1

    if bonferroni_correction:
        alphas_corrected = alphas / num_comparisons
    else:
        alphas_corrected = alphas

    significance = {}
    for metric1, metric2_dict in p_vals.items():
        significance[metric1] = {}
        for metric2, p_val in metric2_dict.items():
            if p_val is not None:
                significance[metric1][metric2] = np.sum(p_val < alphas_corrected)
            else:
                significance[metric1][metric2] = 0
    return significance

def plot_significance_matrix(significance, output_folder):

    annot_mapping = {-1: "", 0: 'ns', 1: '*', 2: '**', 3: '***'}

    # Build global significance matrix: True only if significance is True for all values of n
    metrics_all = list(significance.keys())
    global_matrix = np.zeros((len(metrics_all), len(metrics_all)))

    for i, metric1 in enumerate(metrics_all):
        for j, metric2 in enumerate(metrics_all):
            val = significance.get(metric1, {}).get(metric2, None)
            global_matrix[i, j] = min(3, val)
        global_matrix[i, i] = -1

    # define a discrete colormap for the four significance levels (ns, *, **, ***)
    # your matrix
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
    annot_matrix = np.vectorize(annot_mapping.get)(global_matrix)
    plt.figure(figsize=(11, 8))
    labels = [metric_labels.get(m, m) for m in metrics]
    sns.heatmap(global_matrix, xticklabels=labels, yticklabels=labels, annot=annot_matrix, cmap=cmap, cbar=False, fmt="")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='center')

    # update the heatmap image to use the new colormap and fixed color limits
    ax = plt.gca()
    images = ax.get_images()
    if images:
        img = images[0]
        img.set_cmap(cmap)
        img.set_clim(0, 3)

    # add a legend explaining the star annotations and their significance levels
    labels_dict = {0:'ns', 1:'* (5%)', 2:'** (1%)', 3:'*** (0.1%)'}
    labels = [labels_dict[v] for v in np.unique(global_matrix) if v != -1]
    patches = [mpatches.Patch(color=cmap.colors[i+1], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, title='Significance', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title('Metric CCP equality tests w/o Bonferroni correction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'significance_matrix_CCP.png'))
    plt.close()
    
if __name__ == "__main__":
    folder_path = "../results_metrics_segm"
    file_prefix = "aggregated_results"
    methods = ['percentile']
    metrics=["boundary_iou", "iou", "cldice", "dsc", "nsd", "hd", "hd_perc", "masd", "assd"]
    stats=['mean']
    
    df_segm = extract_coverage_data(folder_path, file_prefix, metrics, stats, methods)

    df_fit_results = perform_fits(df_segm)
    
    p_vals = perform_pairwise_tests(df_fit_results, metrics)

    significance = tell_significance(p_vals, bonferroni_correction=False)

    output_folder = "../significance_matrices"
    plot_significance_matrix(significance, output_folder=output_folder)