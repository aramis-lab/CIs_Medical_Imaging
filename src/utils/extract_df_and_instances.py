import os
import pandas as pd
import numpy as np

def extract_df(path, metric, task):
    metric = "DCS" if metric == "DSC" else "HSD" if metric == "NSD" else metric
    df = pd.read_csv(path)
    df = df[(df["score"] == metric) & (df["subtask"] == task)]
    return df.drop(["score", "subtask", "task", "case", "alg_number", "phase"], axis=1)

def get_benchmark_instances(BASE_DIR, cfg):
    benchmark_instances = []
    for task in cfg.tasks:
        df_task = extract_df(os.path.join(BASE_DIR, cfg.relative_data_path), cfg.metric, task)
        print(df_task.shape)
        algos = df_task["alg_name"].unique()
        for algo in algos:
            benchmark_instances.append((task, algo))
    # Sort by task
    benchmark_instances.sort(key=lambda x: (x[0], x[1]))
    # Remove duplicates
    benchmark_instances = list(dict.fromkeys(benchmark_instances))
    
    return np.array(benchmark_instances)