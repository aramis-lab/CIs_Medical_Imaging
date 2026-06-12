import numpy as np
import pandas as pd
import os

def fit_ccp_classif(classif_path,agreg_type):
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

def fit_ccp_segm(segm_path):
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

micro_path='../results_metrics_classif'
macro_path = '../results_metrics_classif_macro'
segm_path = '../results_metrics_segm'

df_fit_results=fit_ccp_segm(segm_path)

df_fit_results_micro=fit_ccp_classif(micro_path, 'micro')
df_fit_results_macro=fit_ccp_classif(macro_path, 'macro')

df_fit_results_wdp=fit_wdp_segm(segm_path)
df_fit_results_classif_wdp=fit_wdp_classif(micro_path, 'micro')