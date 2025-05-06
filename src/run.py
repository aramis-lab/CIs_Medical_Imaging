import pandas as pd
import numpy as np
import hydra
import submitit
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde
from summary_stats import get_statistic
from intervals_and_metrics import compute_CIs, get_bounds, get_authorized_methods
from kernels import get_kernel
from utils import get_benchmark_instances, extract_df
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_kdes_classification(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(config.summary_stat, config.metric))
    results = pd.DataFrame()

    a, b = get_bounds(config.metric)

    results = None
    return results

def make_kdes_segmentation(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(config.summary_stat, config.metric))
    statistic = lambda x, axis=None: get_statistic(config.summary_stat)(x, config.trimmed_mean_threshold, axis=axis)
    results = pd.DataFrame()

    a, b = get_bounds(config.metric)

    kernel = get_kernel(config.kernel)

    values = df[df["alg_name"] == algo]["value"].to_numpy()

    values_span = np.max(values) - np.min(values)
    # Define the grid for KDE
    if np.isinf(a):
        min_val = np.min(values) - 0.1 * values_span
    else:
        min_val = a
    
    if np.isinf(b):
        max_val = np.max(values) + 0.1 * values_span
    else:
        max_val = b
    x = np.linspace(min_val, max_val, 10000)  # You can change the resolution of x
    alphas = np.ones(len(values))

    dist_to_bounds = np.min([values-a, b-values], axis=0)

    # Iterative weighted KDE estimation
    y = weighted_kde(values, x, dist_to_bounds, kernel, alphas)
    indices = np.searchsorted(x, values)
    initial_estimates = y[indices]
    log_g = np.mean(np.log(initial_estimates))
    g = np.exp(log_g)
    alphas = (initial_estimates / g) ** (-1/2)
    
    y = weighted_kde(values, x, dist_to_bounds, kernel, alphas)
    samples = sample_weighted_kde(y, x, 1000000)

    # Compute true statistic
    true_value = statistic(samples)

    for n in config.sample_sizes:
        samples = sample_weighted_kde(y,x, config.n_samples*n).reshape(config.n_samples, n)

        method_CIs = {
        method: compute_CIs(samples, method, statistic)
        for method in ci_methods
        }
        for sample_index in range(config.n_samples):
            row = {
                "subtask": task,
                "alg_name": algo,
                "n": n,
                "sample_index": sample_index
            }

            for method, CIs in method_CIs.items():
                CI = CIs[:, sample_index]
                row.update({
                    f"lower_bound_{method}": CI[0],
                    f"upper_bound_{method}": CI[1],
                    f"contains_true_stat_{method}": CI[0] <= true_value <= CI[1],
                    f"width_{method}": CI[1] - CI[0],
                    f"proportion_oob_{method}": ((CI[0] < 0) * (-CI[0]) + (CI[1] > 1) * (CI[1] - 1)) / (CI[1] - CI[0]),
                })

            results = pd.concat([results, pd.DataFrame(row, index=[0])], ignore_index=True)
    return results

def process_instance(task, algo, cfg):
    print(f"Running KDE for metric {cfg.metric}, subtask {task} and algorithm {algo}")
    path = os.path.join(BASE_DIR, cfg.relative_data_path)
    df = extract_df(path, cfg.metric, task)
    if cfg.metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "mcc", "ap", "auroc", "auc"]:
        results = make_kdes_classification(df, task, algo, cfg)
    else:
        results = make_kdes_segmentation(df, task, algo, cfg)
    return results

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    aggreggated_results = pd.DataFrame()

    benchmark_instances = get_benchmark_instances(BASE_DIR, cfg)
    # Initialize Submitit executor
    executor = submitit.AutoExecutor(folder="submitit_logs/")
    executor.update_parameters(slurm_partition="cpu_p1", slurm_account="qhn@cpu")

    # Launch jobs
    jobs = []
    with executor.batch():
        for task, algo in benchmark_instances:
            jobs.append(executor.submit(process_instance, task, algo, cfg))

    # Collect results
    aggregated_results = pd.DataFrame()
    for job in jobs:
        result = job.result()
        aggregated_results = pd.concat([aggregated_results, result], ignore_index=True)

    
    RESULTS_DIR = os.path.join(BASE_DIR, cfg.relative_output_dir)
    aggreggated_results.to_csv(os.path.join(RESULTS_DIR, f"results_{cfg.metric}_{cfg.summary_stat}.csv"))

if __name__ == "__main__":
    main()