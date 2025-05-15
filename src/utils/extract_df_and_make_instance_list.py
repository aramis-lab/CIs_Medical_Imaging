import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import os

def extract_df(path, metric, task):
    df = pd.read_csv(path)
    df = df[(df["score"] == metric) & (df["subtask"] == task)]
    return df.drop(["score", "subtask", "task", "case", "alg_number", "phase"], axis=1)

def get_benchmark_instances(BASE_DIR, cfg):
    benchmark_instances = []
    for task in cfg.tasks:
        df_task = extract_df(os.path.join(BASE_DIR, cfg.relative_data_path), cfg.metric, task)
        algos = df_task["alg_name"].unique()
        for algo in algos:
            benchmark_instances.append((task, algo))
    # Sort by task
    benchmark_instances.sort(key=lambda x: (x[0], x[1]))
    # Remove duplicates
    benchmark_instances = list(dict.fromkeys(benchmark_instances))
    
    return np.array(benchmark_instances)

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg_path = "../cfg/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    instances = get_benchmark_instances(BASE_DIR, cfg)

    with open("../benchmark_list.txt", "w") as f:
        for task, algo in instances:
            f.write(f"{task} {algo}\n")