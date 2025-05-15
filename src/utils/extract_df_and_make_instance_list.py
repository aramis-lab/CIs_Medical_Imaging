import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
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

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def export_benchmark_list(cfg: DictConfig):
    BASE_DIR = BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    instances = get_benchmark_instances(BASE_DIR, cfg)
    print(instances)
    if not os.path.exists(os.path.join(BASE_DIR, "/instances_list")):
        os.makedirs(os.path.join(BASE_DIR, "/instances_list"))
    with open(os.path.join(BASE_DIR, f"/instances_list/{cfg.metric}.txt"), "w") as f:
        for task, algo in instances:
            f.write(f"{task} {algo}\n")

if __name__ == "__main__":
    export_benchmark_list()