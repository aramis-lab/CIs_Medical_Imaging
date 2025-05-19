import pandas as pd
import numpy as np
import hydra
import sys
from collections import defaultdict
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde
from summary_stats import get_statistic
from intervals_and_metrics import compute_CIs, get_bounds, get_authorized_methods
from kernels import get_kernel
from utils import extract_df
import os

from tqdm import tqdm

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
    results = pd.DataFrame(columns=["subtask", "alg_name", "n", "sample_index"] + [f"{stat}_{method}" for method in ci_methods for stat in ["lower_bound", "upper_bound", "contains_true_stat", "width", "proportion_oob"]])

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

    RESULTS_DIR = os.path.join(BASE_DIR, config.relative_output_dir)
    for n in tqdm(config.sample_sizes):
        output_path = os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}_{task}_{algo}_{n}.csv")
        if os.path.exists(output_path):
            existing_results = pd.read_csv(output_path)
            if existing_results.shape[0]==config.n_samples: # Already computed
                print(f"Skipping n = {n}, results already exist")
                return None
            else:
                print(f"Computing CIs for n = {n}")
            del existing_results
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

        results = pd.concat([results, pd.DataFrame(data = all_rows.values())], ignore_index=True)

        results = results.drop_duplicates(["subtask", "alg_name", "n", "sample_index"])
        results.to_csv(output_path, index=False)

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):

    print(f"Running KDE for metric {cfg.metric}, subtask {cfg.task} and algorithm {cfg.algo}")
    path = os.path.join(BASE_DIR, cfg.relative_data_path)
    df = extract_df(path, cfg.metric, cfg.task)
    if cfg.metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "mcc", "ap", "auroc", "auc"]:
        make_kdes_classification(df, cfg.task, str(cfg.algo), cfg)
    else:
        make_kdes_segmentation(df, cfg.task, str(cfg.algo), cfg)

if __name__ == "__main__":
    main()