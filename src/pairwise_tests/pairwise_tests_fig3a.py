import pandas as pd
import numpy as np
import os
from scipy.stats import binomtest
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

def perform_pairwise_tests(df_segm, metrics):
    
    n_values = df_segm['n'].unique()
    p_values = {n: {m : {m2: None for m2 in metrics} for m in metrics} for n in n_values}

    for n in n_values:
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                metric1 = metrics[i]
                metric2 = metrics[j]

                data_metric1 = df_segm[(df_segm['metric'] == metric1) & (df_segm['n'] == n)]
                data_metric2 = df_segm[(df_segm['metric'] == metric2) & (df_segm['n'] == n)]

                methods = df_segm['method'].unique()
                for method in methods:
                    # Pair by task and algo: aggregate coverage per (task, algo) then run paired test across those pairs
                    grp1 = data_metric1[data_metric1['method'] == method].groupby(['task', 'algo'])['coverage'].mean().reset_index(name='cov1')
                    grp2 = data_metric2[data_metric2['method'] == method].groupby(['task', 'algo'])['coverage'].mean().reset_index(name='cov2')

                    merged = pd.merge(grp1, grp2, on=['task', 'algo'], how='inner')

                    diff = merged['cov1'] - merged['cov2']

                    diff = diff[diff != 0].dropna()

                    n_pos  = np.sum(diff > 0)
                    n_tot = len(diff)

                    res = binomtest(n_pos, n_tot, p=0.5, alternative='two-sided')

                    p_values[n][metric1][metric2] = res.pvalue
                    p_values[n][metric2][metric1] = res.pvalue
    
    return p_values

def tell_significance(p_vals, alphas=np.array([0.001, 0.01, 0.05]), bonferroni_correction=True):
    
    n_values = len(p_vals)
    m = len(next(iter(p_vals.values())))
    num_comparisons = n_values * (m - 1)

    if bonferroni_correction:
        alphas_corrected = alphas / num_comparisons
    else:
        alphas_corrected = alphas

    significance = {}
    for n, metrics_dict in p_vals.items():
        significance[n] = {}
        for metric1, metric2_dict in metrics_dict.items():
            significance[n][metric1] = {}
            for metric2, p_val in metric2_dict.items():
                if p_val is not None:
                    significance[n][metric1][metric2] = np.sum(p_val < alphas_corrected)
                else:
                    significance[n][metric1][metric2] = 0
    return significance

def plot_significance_matrix(significance, output_folder):

    n_values = list(significance.keys())

    annot_mapping = {-1: "", 0: 'ns', 1: '*', 2: '**', 3: '***'}

    # define a discrete colormap for the four significance levels (ns, *, **, ***)
    cmap = ListedColormap(['#000000', '#d9d9d9', '#fee08b', '#fdae61', '#d73027'])  # grey -> yellow -> orange -> red
    for n in n_values:
        metrics = list(significance[n].keys())
        matrix = np.zeros((len(metrics), len(metrics)))

        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                matrix[i, j] = significance[n][metric1][metric2]
        
            matrix[i, i] = -1
        
        annot_matrix = np.vectorize(annot_mapping.get)(matrix)
        plt.figure(figsize=(11, 8))
        # use friendly metric names for ticks
        labels = [metric_labels.get(m, m) for m in metrics]
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=annot_matrix, cmap=cmap, cbar=False, fmt="")
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
        labels = ['ns', '* (5%)', '** (1%)', '*** (0.1%)']
        patches = [mpatches.Patch(color=cmap.colors[i+1], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=patches, title='Significance', bbox_to_anchor=(1.02, 1), loc='upper left')

        # make room for the legend and avoid clipping
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.title(f'Paired tests for coverage difference between metrics for n={n} w/ Bonferroni correction')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'significance_matrix_n{n}_corrected.png'))
        plt.close()

    # Build global significance matrix: True only if significance is True for all values of n
    first_n_key = next(iter(significance))
    metrics_all = list(significance[first_n_key].keys())
    global_matrix = np.zeros((len(metrics_all), len(metrics_all)))

    for i, metric1 in enumerate(metrics_all):
        for j, metric2 in enumerate(metrics_all):
            all_sig = 3
            for n_key in significance.keys():
                val = significance[n_key].get(metric1, {}).get(metric2, None)
                all_sig = min(all_sig, val)
            global_matrix[i, j] = all_sig
        global_matrix[i, i] = -1

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
    labels = ['ns', '* (5%)', '** (1%)', '*** (0.1%)']
    patches = [mpatches.Patch(color=cmap.colors[i+1], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, title='Significance', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title('Paired tests for coverage difference between metrics (all n values) w/ Bonferroni correction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'significance_matrix_global_corrected.png'))
    plt.close()
    
if __name__ == "__main__":
    folder_path = "../results_metrics_segm"
    file_prefix = "aggregated_results"
    methods = ['percentile']
    metrics=["boundary_iou", "iou", "cldice", "dsc", "nsd", "hd", "hd_perc", "masd", "assd"]
    stats=['mean']
    
    df_segm = extract_coverage_data(folder_path, file_prefix, metrics, stats, methods)
    
    p_vals = perform_pairwise_tests(df_segm, metrics)

    significance = tell_significance(p_vals, bonferroni_correction=True)

    output_folder = "../significance_matrices"
    plot_significance_matrix(significance, output_folder=output_folder)