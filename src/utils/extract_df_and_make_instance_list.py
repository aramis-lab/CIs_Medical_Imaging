import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
import os

def extract_df(path, metric, task):
    df = pd.read_csv(path)
    df = df[df["subtask"] == task]
    if "score" in df.columns:
        df = df[df["score"] == metric]
    if "alg_name" in df.columns and "value" in df.columns:
        print(f"Extracting values for metric '{metric}' and task '{task}'")
        return df[["alg_name", "value"]]
    elif "alg_name" in df.columns and "logits" in df.columns and "target" in df.columns:
        print(f"Extracting logits and targets for metric '{metric}' and task '{task}'")
        return df[["alg_name", "logits", "target"]]
    else:
        raise ValueError(f"DataFrame does not contain required columns for metric '{metric}' and task '{task}'.")

def get_benchmark_instances(BASE_DIR, cfg):
    benchmark_instances = []
    df_all = pd.read_csv(os.path.join(BASE_DIR, cfg.relative_data_path))
    tasks = df_all["subtask"].unique()
    for task in tasks:
        df_task = extract_df(os.path.join(BASE_DIR, cfg.relative_data_path), cfg.metric, task)
        algos = df_task["alg_name"].unique()
        for algo in algos:
            benchmark_instances.append((task, algo))
    # Sort by task
    benchmark_instances.sort(key=lambda x: (x[0], x[1]))
    # Remove duplicates
    benchmark_instances = list(dict.fromkeys(benchmark_instances))
    
    return np.array(benchmark_instances)

@hydra.main(config_path="../cfg", version_base="1.3.2")
def export_benchmark_list(cfg: DictConfig):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    instances = get_benchmark_instances(BASE_DIR, cfg)
    if "task" in cfg:
        task_filter = cfg.task
        # Ensure task_filter is a list
        if isinstance(task_filter, str):
            task_filter = [task_filter]
        instances = np.array([inst for inst in instances if inst[0] in task_filter])
    if not os.path.exists(os.path.join(BASE_DIR, "instances_list")):
        os.makedirs(os.path.join(BASE_DIR, "instances_list"))
    if not os.path.exists(os.path.join(BASE_DIR, f"instances_list/{cfg.metric}.txt")):
        with open(os.path.join(BASE_DIR, f"instances_list/{cfg.metric}.txt"), "w") as f:
            for task, algo in instances:
                f.write(f"{task} {algo}\n")

if __name__ == "__main__":
    export_benchmark_list()