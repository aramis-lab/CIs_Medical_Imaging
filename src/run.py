import pandas as pd
import numpy as np
import joblib
import hydra
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde
from summary_stats import get_statistic
from intervals_and_metrics import compute_CIs, compute_metrics, get_bounds
from kernels import get_kernel
import os

from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def extract_df(df, metric, task):
    metric = "DCS" if metric == "DSC" else "HSD" if metric == "NSD" else metric
    df = df[(df["score"] == metric) & (df["subtask"] == task)]
    return df.drop(["score", "subtask", "task", "case", "alg_number", "phase"], axis=1)

def get_benchmark_instances(cfg):
    benchmark_instances = []
    df = pd.read_csv(os.path.join(BASE_DIR, cfg.relative_data_path))
    for task in cfg.tasks:
        df_task = extract_df(df, cfg.metric, task)
        algos = df_task["alg_name"].unique()
        for algo in algos:
            benchmark_instances.append((task, algo))
    # Sort by task
    benchmark_instances.sort(key=lambda x: (x[0], x[1]))
    # Remove duplicates
    benchmark_instances = list(dict.fromkeys(benchmark_instances))

    return np.array(benchmark_instances)

def make_kdes_and_compute_metrics(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = config.ci_methods
    statistic = lambda x, axis=None: get_statistic(config.summary_stat)(x, config.trimmed_mean_threshold, axis=axis)
    results = pd.DataFrame()

    a, b = get_bounds(config.metric)

    kernel = get_kernel(config.kernel)

    DSCs = df[df["alg_name"] == algo]["value"].to_numpy()

    values_span = np.max(DSCs) - np.min(DSCs)
    # Define the grid for KDE
    if np.isinf(a):
        min_val = np.min(DSCs) - 0.1 * values_span
    else:
        min_val = a
    
    if np.isinf(b):
        max_val = np.max(DSCs) + 0.1 * values_span
    else:
        max_val = b
    x = np.linspace(min_val, max_val, 10000)  # You can change the resolution of x
    alphas = np.ones(len(DSCs))

    dist_to_bounds = np.min([DSCs-a, b-DSCs], axis=0)

    # Iterative weighted KDE estimation
    for _ in range(1):
        y = weighted_kde(DSCs, x, dist_to_bounds, kernel, alphas)
        indices = np.searchsorted(x, DSCs)
        initial_estimates = y[indices]
        log_g = np.mean(np.log(initial_estimates))
        g = np.exp(log_g)
        alphas = (initial_estimates / g) ** (-1/2) * (1-np.exp(-1e6*dist_to_bounds))
    
    y = weighted_kde(DSCs, x, dist_to_bounds, kernel, alphas)
    samples = sample_weighted_kde(y, x, 1000000)

    # Compute true statistic
    true_value = statistic(samples)

    for n in tqdm(config.sample_sizes):
        new_row = {"subtask": task, "alg_name": algo, "n": n}
        samples = sample_weighted_kde(y,x, config.n_samples*n).reshape(config.n_samples, n)

        for method in ci_methods:
            CIs, nan_proportion = compute_CIs(samples, method, statistic)
            coverage, proportion_oob, width = compute_metrics(CIs, true_value)
            new_row.update({f"{method}_coverage": coverage, f"{method}_proportion_oob": proportion_oob, f"{method}_width": width, f"{method}_nans": nan_proportion})
        results = pd.concat([results, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    return results

def process_subtask(task, algo, cfg):
    print(f"Running KDE for metric {cfg.metric}, subtask {task} and algorithm {algo}")
    df = pd.read_csv(os.path.join(BASE_DIR, cfg.relative_data_path))
    df = extract_df(df, cfg.metric, task)
    results = make_kdes_and_compute_metrics(df, task, algo, cfg)
    return results

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    aggreggated_results = pd.DataFrame()

    benchmark_instances = get_benchmark_instances(cfg)

    # Use joblib to parallelize tasks
    with joblib.Parallel(n_jobs=-1) as parallel:
        results = parallel(joblib.delayed(process_subtask)(task, algo, cfg) for task, algo in benchmark_instances[-19:])

    # Combine results from all tasks
    for result in results:
        aggreggated_results = pd.concat([aggreggated_results, result], ignore_index=True)
    
    RESULTS_DIR = os.path.join(BASE_DIR, cfg.relative_output_dir)
    aggreggated_results.to_csv(os.path.join(RESULTS_DIR, f"results_{cfg.metric}_{cfg.summary_stat}.csv"))

if __name__ == "__main__":
    main()