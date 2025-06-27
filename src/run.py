import pandas as pd
import numpy as np
import hydra
from collections import defaultdict
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde, sample_weighted_kde_multivariate
from summary_stats import get_statistic
from intervals_and_metrics import get_metric, is_continuous, compute_CIs_segmentation, compute_CIs_classification, get_bounds, get_authorized_methods
from kernels import get_kernel
from utils import extract_df
import os
from scipy.special import softmax

from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_kdes_classification(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(None, config.metric))
    metric = get_metric(config.metric)
    results = pd.DataFrame()

    # Convert string representations of sets to 2D numpy array
    logits_str = df[df["alg_name"] == algo]["logits"]
    values = [list(eval(v, {"nan": np.nan})) for v in logits_str]
    lengths = np.array([len(v) for v in values])
    good_length = round(np.mean(lengths))
    indices = np.where(lengths==good_length)
    values = np.array([v for v in values if len(v)==good_length])
    labels = df[df["alg_name"] == algo]["target"].to_numpy()[indices]

    if len(values) < 50:
        print(f"Not enough values for {task} {algo} ({len(values)}), skipping KDE")
        return

    if np.any(np.isnan(values)): # There should be no NaNs in the logits, but just in case
        print("There are NaNs in the data, skipping to next instance")
        return

    kernel = get_kernel(config.kernel)

    # Define the grid for KDE
    alphas = np.ones(len(values))

    # Iterative weighted KDE estimation
    if config.adaptive_bandwidth:
        initial_estimates = kernel(values, values, alphas)
        initial_estimates = np.mean(initial_estimates, axis=1)
        log_g = np.mean(np.log(initial_estimates))
        g = np.exp(log_g)
        alphas = (initial_estimates / g) ** (-1/2)
    samples, sim_labels = sample_weighted_kde_multivariate(values, labels, config.kernel, 100000, alphas) # Shapes (1000000, 5) and (1000000,), not binary

    if (values.shape[1]>2 and config.metric not in ["accuracy", "f1_score","mcc","balanced_accuracy"]): # Multi-class problem, only bootstrap works properly
        ci_methods = ci_methods.intersection(["basic", "percentile", "bca"])
    true_value = metric(sim_labels, samples, average=config.average)
    all_rows = defaultdict(dict)
    RESULTS_DIR = os.path.join(BASE_DIR, config.relative_output_dir)
    for n in tqdm(config.sample_sizes):
        output_path = os.path.join(RESULTS_DIR, f"results_{config.metric}_{task}_{algo}_{n}.csv")
        if os.path.exists(output_path):
            existing_results = pd.read_csv(output_path)
            if existing_results.shape[0]>=config.n_samples: # Already computed
                print(f"Skipping n = {n}, results already exist")
                continue
            else:
                print(f"Computing CIs for n = {n}")
            del existing_results
        samples, sim_labels = sample_weighted_kde_multivariate(values, labels, config.kernel, config.n_samples * n, alphas)
        samples = samples.reshape(config.n_samples, n, -1)
        sim_labels = sim_labels.reshape(config.n_samples, n)
        batch_size = 50
        for method in ci_methods:
            for batch_start in range(0, config.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.n_samples)
                batch_samples = samples[batch_start:batch_end]
                batch_labels = sim_labels[batch_start:batch_end]
                CIs = compute_CIs_classification(batch_labels, batch_samples, config.metric, method, average=config.average)

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
                    "true_value" : true_value,
                    f"lower_bound_{method}": lower_bounds[sample_index - batch_start],
                    f"upper_bound_{method}": upper_bounds[sample_index - batch_start],
                    f"contains_true_stat_{method}": contains_true[sample_index - batch_start],
                    f"width_{method}": widths[sample_index - batch_start],
                    f"proportion_oob_{method}": proportion_oob[sample_index - batch_start],
                    })

        results = pd.DataFrame(data = all_rows.values())

        results = results.drop_duplicates(["subtask", "alg_name", "n", "sample_index"])
        results.to_csv(output_path, index=False)

def make_kdes_segmentation(df, task, algo, config):
    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods(config.summary_stat, config.metric))
    def statistic(x, axis=None):
        return get_statistic(config.summary_stat)(x, config.trimmed_mean_threshold, axis=axis)
    results = pd.DataFrame(columns=["subtask", "alg_name", "n", "sample_index"] + [f"{stat}_{method}" for method in ci_methods for stat in ["lower_bound", "upper_bound", "contains_true_stat", "width", "proportion_oob"]])

    a, b = get_bounds(config.metric)

    kernel = get_kernel(config.kernel)
    
    values = df[df["alg_name"] == algo]["value"].to_numpy()
    values = values[~np.isnan(values)]  # Remove NaN values
    if len(values) < 50:
        print(f"Not enough values for {task} {algo} ({len(values)}), skipping KDE")
        return

    if not is_continuous(config.metric):
        samples = np.random.choice(values, size=1000000, replace=True)
    else:
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
        if config.adaptive_bandwidth:
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
            # Check if results already exist
            print(f"Output path {output_path} already exists, checking if results are sufficient")
            existing_results = pd.read_csv(output_path)
            if existing_results.shape[0]>=config.n_samples: # Already computed
                print(f"Skipping n = {n}, results already exist")
                continue
            else:
                print(f"Computing CIs for n = {n}")
            del existing_results
        if not is_continuous(config.metric): # For discrete metrics, KDE makes no sense, we sample uniformly
            samples = np.random.choice(values, size=config.n_samples * n, replace=True).reshape(config.n_samples, n)
        else:
            samples = sample_weighted_kde(y, x, config.n_samples * n).reshape(config.n_samples, n)

        batch_size = 50
        for method in ci_methods:
            for batch_start in range(0, config.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.n_samples)
                batch_samples = samples[batch_start:batch_end]
                CIs = compute_CIs_segmentation(batch_samples, method, config.summary_stat, statistic, config.trimmed_mean_threshold)

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

        results = pd.DataFrame(data = all_rows.values())

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