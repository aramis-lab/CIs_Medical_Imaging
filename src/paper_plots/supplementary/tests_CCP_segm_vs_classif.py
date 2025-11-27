tick_fontsize=14
title_fontsize=18
label_fontsize=16

segm_metrics=["boundary_iou", "iou", "cldice", "dsc", "nsd"]
classif_metrics=["balanced_accuracy", "ap", "auc", "f1_score"]
stats=['mean', "median", "trimmed_mean", "std", "iqr_length"]

def extract_coverage_data_classif(folder_path, file_prefix, metrics, methods):
    all_values=[]
    for metric in metrics:

        file_path = os.path.join(folder_path, f"{file_prefix}_{metric}.csv")
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

                    for method, col in method_dict.items():
                        if method in methods:
                            for val in data_n[col]:

                                all_values.append({
                                    'metric': metric,
                                    'task':task, 
                                    'algo':algo,
                                    'n': n,
                                    'method': method,
                                    'coverage': val
                                })
    df_classif=pd.DataFrame(all_values)
    return df_classif

def perform_fits_segm(df_segm, stats):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in segm_metrics:
                for stat in stats:
                    df_metric_stat = df_algo[(df_algo['metric'] == metric) & (df_algo['stat']==stat)]
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
                            'stat': stat,
                            'method': method,
                            'beta2': beta2[0],
                            'R2': rel_error
                        }
                        results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def perform_fits_classif(df_classif):
    results = []
    for task in df_classif['task'].unique():
        df_task = df_classif[df_classif['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
                df_metric = df_algo[df_algo['metric'] == metric]
                for method in df_metric['method'].unique():
                    df_metric_method = df_metric[df_metric['method'] == method]
                    df_metric_method = df_metric_method.sort_values(by='n')
                    n_values = df_metric_method['n'].to_numpy()
                    coverages = df_metric_method['coverage'].to_numpy()
                    Y = 0.95 - coverages
                    X = np.vstack([1/n_values]).T
                    beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                    rel_error = np.linalg.norm(X @ beta2 - Y) / np.linalg.norm(Y)
                    new_row = {
                        'task': task,
                        'algo': algo,
                        'metric': metric,
                        'method': method,
                        'beta2': beta2[0],
                        'rel_error': rel_error
                    }
                    results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def perform_pairwise_tests(df_fit_results, df_fit_results_classif):

    segm_metrics = df_fit_results['metric'].unique()
    classif_metrics = df_fit_results_classif['metric'].unique()
    methods = df_fit_results_classif['method'].unique()
    stats = df_fit_results['stat'].unique()
    p_values = {met : {s : {m : {m2: None for m2 in segm_metrics} for m in classif_metrics} for s in stats} for met in methods}

    for method in methods:
        for stat in stats:
            if (stat != 'mean') and (method in ['param_z', 'param_t']):
                continue
            for metric1 in classif_metrics:
                for metric2 in segm_metrics:
                    data_metric1 = df_fit_results_classif[(df_fit_results_classif["method"]==method) & (df_fit_results_classif['metric'] == metric1)]
                    data_metric2 = df_fit_results[(df_fit_results["method"]==method) & (df_fit_results['metric'] == metric2)]

                    def statistic(x, y):
                        return np.mean(x) - np.mean(y)

                    res = permutation_test(
                        (data_metric1['beta2'].to_numpy(), data_metric2['beta2'].to_numpy()),
                        statistic,
                        vectorized=False,
                        n_resamples=10000,
                        alternative='two-sided'
                    )
                    pval = res.pvalue

                    p_values[method][stat][metric1][metric2] = pval

    return p_values

def tell_significance(p_vals, alphas=np.array([0.001, 0.01, 0.05]), bonferroni_correction=True):
    
    m = len(next(iter(next(iter(p_vals.values())).values())).keys())
    n = len(next(iter(next(iter(next(iter(p_vals.values())).values())).values())).keys())
    num_comparisons = max(m, n)

    if bonferroni_correction:
        alphas_corrected = alphas / num_comparisons
    else:
        alphas_corrected = alphas

    significance = {}
    for method, stat_dict in p_vals.items():
        significance[method] = {}
        for stat, metric1_dict in stat_dict.items():
            significance[method][stat] = {}
            for metric1, metric2_dict in metric1_dict.items():
                significance[method][stat][metric1] = {}
                for metric2, p_val in metric2_dict.items():
                    if p_val is not None:
                        significance[method][stat][metric1][metric2] = np.sum(p_val < alphas_corrected)
                    else:
                        significance[method][stat][metric1][metric2] = 0
    return significance

def plot_significance_matrix(significance, p_vals, output_folder, title_fontsize=18, tick_fontsize=14):

    methods = list(significance.keys())
    stats = list(next(iter(significance.values())).keys())
    metrics_classif = list(next(iter(next(iter(significance.values())).values())).keys())
    metrics_segm = list(next(iter(next(iter(next(iter(significance.values())).values())).values())).keys())

    fig, axes = plt.subplots(len(methods), len(stats), figsize=(11 * len(stats), 8 * len(methods)))

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
            pval_matrix = np.full((len(metrics_segm), len(metrics_classif)), 0.0)
            for i, metric1 in enumerate(metrics_segm):
                for j, metric2 in enumerate(metrics_classif):
                    p_val = p_vals.get(method, {}).get(stat, {}).get(metric2, {}).get(metric1, None)
                    if p_val is not None:
                        pval_matrix[i, j] = p_val.round(4)
                    else:
                        pval_matrix[i, j] = 0.0
            
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
                annot_kws={"fontsize": label_fontsize}
            )
            ax.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)

            ax.tick_params(axis='y', rotation=45, labelsize=tick_fontsize)

            ax.set_title(f"Stat : {stat_labels[stat]}, Method: {method_labels[method]}", fontsize=title_fontsize)

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='k', label='1% (Red)'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='k', label='5% (Orange)'),
        mpatches.Patch(facecolor='#fee08b', edgecolor='k', label='10% (Yellow)'),
        mpatches.Patch(facecolor='#d9d9d9', edgecolor='k', label='Not significant (Gray)')
    ]
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=label_fontsize,
        frameon=True,
        title="Significance levels with Bonferroni correction",
        title_fontsize=label_fontsize
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pairwise_comp_segm_classif.pdf'))
    plt.close()

df_classif = extract_coverage_data_classif("../../../results_metrics_classif_macro", "aggregated_results", ["balanced_accuracy", "ap", "auc", "f1_score"], ["percentile"])
df_fit_results_segm = perform_fits_segm(df_segm, stats)
df_fit_results_classif = perform_fits_classif(df_classif)
print("Fitting completed.")
p_values = perform_pairwise_tests(df_fit_results_segm, df_fit_results_classif)
print("Pairwise tests completed.")
significance = tell_significance(p_values, bonferroni_correction=True)
output_folder = '../../../clean_figs'
plot_significance_matrix(significance, p_values, output_folder)