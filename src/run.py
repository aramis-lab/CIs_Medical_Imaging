import pandas as pd
import numpy as np
import hydra
from collections import defaultdict
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde
from summary_stats import get_statistic
from intervals_and_metrics import compute_CIs, get_bounds, get_authorized_methods
from kernels import get_kernel
from utils import get_benchmark_instances, extract_df
import os

from tqdm import tqdm
from joblib import Parallel, delayed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_kdes_classification(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(config.summary_stat, config.metric))
    results = pd.DataFrame()

    a, b = get_bounds(config.metric)

    results = None
    return results

def make_kdes_segmentation(df, task, algo, config):
    RESULTS_DIR = os.path.join(BASE_DIR, config.relative_output_dir)
    if os.path.exists(os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}.csv")):
        existing_results = pd.read_csv(os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}.csv"))
        if not existing_results[(existing_results["subtask"]==task) & (existing_results["alg_name"]==algo) &
                                (existing_results["n"]==n) & (existing_results["sample_index"]==sample_index)].empty:
            print(f"Skipping task {task}, algorithm {algo}, sample size {n}, sample index {sample_index} as it already exists.")
            return pd.DataFrame()
    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(config.summary_stat, config.metric))
    statistic = lambda x, axis=None: get_statistic(config.summary_stat)(x, config.trimmed_mean_threshold, axis=axis)
    results = pd.DataFrame(columns=["subtask", "alg_name", "n", "sample_index"] + [f"{method}_{stat}" for method in ci_methods for stat in ["lower_bound", "upper_bound", "contains_true_stat", "width", "proportion_oob"]])

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
    all_rows = defaultdict(dict)

    for n in tqdm(config.sample_sizes):
        samples = sample_weighted_kde(y, x, config.n_samples * n).reshape(config.n_samples, n)

        batch_size = 50
        for method in ci_methods:
            for batch_start in range(0, config.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.n_samples)
                batch_samples = samples[batch_start:batch_end]
                CIs = compute_CIs(batch_samples, method, statistic)

                # Precompute vectorized components for speed
                lower_bounds = CIs[:, 0]
                upper_bounds = CIs[:, 1]
                widths = upper_bounds - lower_bounds
                contains_true = (lower_bounds <= true_value) & (true_value <= upper_bounds)
                proportion_oob = ((lower_bounds < 0) * (-lower_bounds) + (upper_bounds > 1) * (upper_bounds - 1)) / widths

                for sample_index in range(batch_start, batch_end):
                    key = (task, algo, n, sample_index)
                    all_rows[key].update({
                    "subtask": task,
                    "alg_name": algo,
                    "n": n,
                    "sample_index": sample_index,
                    f"lower_bound_{method}": lower_bounds[sample_index - batch_start],
                    f"upper_bound_{method}": upper_bounds[sample_index - batch_start],
                    f"contains_true_stat_{method}": contains_true[sample_index - batch_start],
                    f"width_{method}": widths[sample_index - batch_start],
                    f"proportion_oob_{method}": proportion_oob[sample_index - batch_start],
                    })
    results = pd.DataFrame(all_rows)

    if os.path.exists(os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}.csv")):
        existing_results = pd.read_csv(os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}.csv"))
        existing_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        existing_results = results
    existing_results.to_csv(os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}.csv"), index=False)


def process_instance(task, algo, cfg):
    print(f"Running KDE for metric {cfg.metric}, subtask {task} and algorithm {algo}")
    path = os.path.join(BASE_DIR, cfg.relative_data_path)
    df = extract_df(path, cfg.metric, task)
    if cfg.metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "mcc", "ap", "auroc", "auc"]:
        make_kdes_classification(df, task, algo, cfg)
    else:
        make_kdes_segmentation(df, task, algo, cfg)

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):

    benchmark_instances = get_benchmark_instances(BASE_DIR, cfg)
    # Process instances in parallel using joblib
    Parallel(n_jobs=-1)(
        delayed(process_instance)(task, algo, cfg) for task, algo in benchmark_instances
    )

if __name__ == "__main__":
    main()