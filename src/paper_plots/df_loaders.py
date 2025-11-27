import numpy as np
import pandas as pd
import os


def extract_df_segm_cov(folder_path:str, file_prefix:str, metrics:list[str], stats:list[str])->pd.DataFrame:
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
                            for val in data_n[col]:
                                
                                all_values.append({
                                    'metric': metric,
                                    'stat': stat,
                                    'task':task, 
                                    'algo':algo,
                                    'n': n,
                                    'method': method,
                                    'coverage': val,
                                    'x_group': f"{metric}\nn={n}"
                                })
    df_segm_cov=pd.DataFrame(all_values)
    return df_segm_cov

def extract_df_segm_width(folder_path:str, file_prefix:str, metrics:list[str], stats:list[str])->pd.DataFrame:
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
                            'basic': 'width_basic',
                            'bca': 'width_bca',
                            'percentile': 'width_percentile',
                        }

                        # Add parametric methods only for stat == 'mean'
                        if stat == 'mean':
                            method_dict.update({
                                'param_z': 'width_param_z',
                                'param_t': 'width_param_t'
                            })
                        for method, col in method_dict.items():
                            for val in data_n[col]:
                                
                                all_values.append({
                                    'metric': metric,
                                    'stat': stat,
                                    'task':task, 
                                    'algo':algo,
                                    'n': n,
                                    'method': method,
                                    'width': val,
                                    'x_group': f"{metric}\nn={n}"
                                })
    df_segm_width=pd.DataFrame(all_values)
    return df_segm_width

def extract_df_classif_cov(folder_path:str, file_prefix:str, metrics:list[str])->pd.DataFrame:
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

                    # Add parametric methods only for stat == 'mean'
                    if metric == 'accuracy':
                        method_dict.update({
                            'agresti_coull':'contains_true_stat_agresti_coull',
                            'wilson':'contains_true_stat_wilson',
                            'exact':'contains_true_stat_exact',
                            'wald': 'contains_true_stat_wald'
                        })
                    for method, col in method_dict.items():
                        for val in data_n[col]:
                            all_values.append({
                                'subtask':task, 
                                'algo':algo,
                                'stat': metric,
                                'n': n,
                                'method': method,
                                'coverage': val,
                                'x_group': f"n={n}"
                            })
    df_classif_cov=pd.DataFrame(all_values)
    return df_classif_cov

def extract_df_classif_width(folder_path:str, file_prefix:str, metrics:list[str])->pd.DataFrame:
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
                        'basic': 'width_basic',
                        'bca': 'width_bca',
                        'percentile': 'width_percentile',
                    }

                    # Add parametric methods only for stat == 'mean'
                    if metric == 'accuracy':
                        method_dict.update({
                            'agresti_coull':'width_agresti_coull',
                            'wilson':'width_wilson',
                            'exact':'width_exact',
                            'wald': 'width_wald'
                        })
                    for method, col in method_dict.items():
                        for val in data_n[col]:
                            all_values.append({
                                'subtask':task, 
                                'algo':algo,
                                'stat': metric,
                                'n': n,
                                'method': method,
                                'width': val,
                                'x_group': f"n={n}"
                            })
    df_classif_width=pd.DataFrame(all_values)
    return df_classif_width