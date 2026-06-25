import pandas as pd
import numpy as np
import hydra
from collections import defaultdict
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde, sample_weighted_kde_multivariate
from summary_stats import get_statistic
from intervals_and_metrics import get_metric, is_continuous, compute_CIs_segmentation, compute_CIs_classification, get_bounds, get_authorized_methods_classification, get_authorized_methods_segmentation, softmax, label_binarize_vectorized
from kernels import get_kernel
from utils import extract_df
import os

from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_kdes_classification(df, task, algo, config):

    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods_classification(config.metric, config.average))
    metric = get_metric(config.metric)
    results = pd.DataFrame()
    
    # Convert string representations of sets to 2D numpy array
    logits_str = df[df["alg_name"].astype(str) == algo]["logits"]
    values = [list(eval(v, {"nan": np.nan})) for v in logits_str]
    if len(values) == 0:
        print(f"Not enough values for {task} {algo} ({len(values)}), skipping KDE")
        return
    lengths = np.array([len(v) for v in values])
    good_length = round(np.mean(lengths))
    indices = np.where(lengths==good_length)
    values = np.array([v for v in values if len(v)==good_length])
    labels = df[df["alg_name"].astype(str) == algo]["target"].to_numpy()[indices]

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
    y_score, y_true = sample_weighted_kde_multivariate(values, labels, config.kernel, 100000, alphas) # Shapes (1000000, n_classes) and (1000000,), not binary
    y_score = softmax(y_score)
    
    n_classes = y_score.shape[-1]
    y_pred = np.argmax(y_score, axis=-1)

    correct_pred = (y_pred==y_true)[..., None] # To allow bootstrapping metric arguments

    y_true_bin = label_binarize_vectorized(y_true, n_classes)
    y_pred_bin = label_binarize_vectorized(y_pred, n_classes)

    tp = (y_true_bin==1) & (y_pred_bin==1)
    fp = (y_true_bin==0) & (y_pred_bin==1)
    tn = (y_true_bin==0) & (y_pred_bin==0)
    fn = (y_true_bin==1) & (y_pred_bin==0)

    metric_arguments = {"accuracy": ["correct_pred"],
                        "precision" : ["tp", "fp"],
                        "recall" : ["tp", "fn"],
                        "f1_score" : ["tp", "fp", "fn"],
                        "fbeta_score" : ["tp", "fp", "fn"],
                        "npv" : ["tn", "fn"],
                        "ppv" : ["tp", "fp"],
                        "sensitivity" : ["tp", "fn"],
                        "specificity" : ["tn", "fp"],
                        "balanced_accuracy" : ["tp", "fp", "tn", "fn"],
                        "mcc" : ["tp", "fp", "fn"],
                        "auroc" : ["y_score", "y_true_bin"],
                        "auc" : ["y_score", "y_true_bin"],
                        "ap" : ["y_score", "y_true_bin"]
    }

    original_arguments = {a: locals()[a] for a in metric_arguments[config.metric]}
    true_value = metric(average=config.average, **original_arguments)
    all_rows = defaultdict(dict)
    RESULTS_DIR = os.path.join(BASE_DIR, config.relative_output_dir)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    raw_output_path = os.path.join(RESULTS_DIR, f"results_{config.metric}__{config.average}_{task}_{algo}.csv")
    agg_output_path = os.path.join(RESULTS_DIR, f"aggregated_results_{config.metric}_{config.average}_{task}_{algo}.csv")

    # Load existing raw results if available
    existing_results = None
    if os.path.exists(raw_output_path):
        existing_results = pd.read_csv(raw_output_path)

    for n in tqdm(config.sample_sizes):
        # Check if this n already has enough results in the existing file
        if existing_results is not None:
            existing_for_n = existing_results[
                (existing_results["subtask"] == task)
                & (existing_results["alg_name"] == algo)
                & (existing_results["n"] == n)
            ]
            if existing_for_n.shape[0] >= config.n_samples:
                print(f"Skipping n = {n}, results already exist")
                continue
            else:
                print(f"Computing CIs for n = {n}")

        samples, sim_labels = sample_weighted_kde_multivariate(
            values, labels, config.kernel, config.n_samples * n, alphas
        )
        samples = samples.reshape(config.n_samples, n, -1)
        samples = softmax(samples)
        sim_labels = sim_labels.reshape(config.n_samples, n)
        batch_size = 5
        for method in ci_methods:
            print(method)
            for batch_start in range(0, config.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.n_samples)

                CIs = compute_CIs_classification(
                    sim_labels[batch_start:batch_end],
                    samples[batch_start:batch_end],
                    config.metric,
                    method,
                    average=config.average,
                    n_bootstrap=config.n_bootstrap,
                )

                lower_bounds = CIs[:, 0]
                upper_bounds = CIs[:, 1]
                widths = upper_bounds - lower_bounds
                contains_true = (lower_bounds <= true_value) & (true_value <= upper_bounds)
                proportion_oob = (
                    (lower_bounds < 0) * (-lower_bounds)
                    + (upper_bounds > 1) * (upper_bounds - 1)
                ) / widths

                for sample_index in range(batch_start, batch_end):
                    key = (task, algo, n, sample_index)
                    all_rows[key].update(
                        {
                            "subtask": task,
                            "alg_name": algo,
                            "n": n,
                            "sample_index": sample_index,
                            "true_value": true_value,
                            f"lower_bound_{method}": lower_bounds[sample_index - batch_start],
                            f"upper_bound_{method}": upper_bounds[sample_index - batch_start],
                            f"contains_true_stat_{method}": contains_true[sample_index - batch_start],
                            f"width_{method}": widths[sample_index - batch_start],
                            f"proportion_oob_{method}": proportion_oob[sample_index - batch_start],
                        }
                    )

    # ── Merge new results with existing, deduplicate, sort, and write ──

    new_results = pd.DataFrame(data=all_rows.values())

    if existing_results is not None and not new_results.empty:
        # New rows override old rows on the same key (keep="last")
        results = pd.concat([existing_results, new_results], ignore_index=True)
    elif existing_results is not None:
        results = existing_results
    elif not new_results.empty:
        results = new_results
    else:
        results = pd.DataFrame()

    if not results.empty:
        key_cols = ["subtask", "alg_name", "n", "sample_index"]
        results = results.drop_duplicates(subset=key_cols, keep="last")
        results = results.sort_values(by=key_cols).reset_index(drop=True)

        # ── Aggregated results (recomputed from the full merged raw data) ──
        group_cols = ["subtask", "alg_name", "n"]
        avg_dfs = []
        for method in ci_methods:
            avg_df = (
                results.groupby(group_cols)
                .agg(
                    {
                        f"contains_true_stat_{method}": "mean",
                        f"width_{method}": "mean",
                        f"proportion_oob_{method}": "mean",
                    }
                )
                .reset_index()
                .rename(columns={f"contains_true_stat_{method}": f"coverage_{method}"})
            )
            avg_dfs.append(avg_df)

        if avg_dfs:
            average_results = avg_dfs[0]
            for df in avg_dfs[1:]:
                average_results = pd.merge(average_results, df, on=group_cols, how="outer")
            average_results = average_results.sort_values(by=group_cols).reset_index(drop=True)
        else:
            average_results = pd.DataFrame()

        # ── Write both files ──
        results.to_csv(raw_output_path, index=False)
        average_results.to_csv(agg_output_path, index=False)

def make_kdes_segmentation(df, task, algo, config):
    # Retrieve configuration and set up variables
    ci_methods = set(config.ci_methods).intersection(get_authorized_methods_segmentation(config.summary_stat, config.metric))
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

        dist_to_bounds = np.inf * np.ones_like(values)
        if config.trim_bandwidth:
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

        samples = sample_weighted_kde(y, x, 1000000, a, b)

    # Compute true statistic
    true_value = statistic(samples)
    all_rows = defaultdict(dict)

    RESULTS_DIR = os.path.join(BASE_DIR, config.relative_output_dir)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    raw_output_path = os.path.join(RESULTS_DIR, f"results_{config.metric}_{config.summary_stat}_{task}_{algo}.csv")
    agg_output_path = os.path.join(RESULTS_DIR, f"aggregated_results_{config.metric}_{config.summary_stat}_{task}_{algo}.csv")

    # Load existing raw results if available
    existing_results = None
    if os.path.exists(raw_output_path):
        existing_results = pd.read_csv(raw_output_path)

    for n in tqdm(config.sample_sizes):
        # Check if this n already has enough results in the existing file
        if existing_results is not None:
            existing_for_n = existing_results[
                (existing_results["subtask"] == task)
                & (existing_results["alg_name"] == algo)
                & (existing_results["n"] == n)
            ]
            if existing_for_n.shape[0] >= config.n_samples:
                print(f"Skipping n = {n}, results already exist")
                continue
            else:
                print(f"Computing CIs for n = {n}")

        if not is_continuous(config.metric):
            samples = np.random.choice(values, size=config.n_samples * n, replace=True).reshape(config.n_samples, n)
        else:
            samples = sample_weighted_kde(y, x, config.n_samples * n, a, b).reshape(config.n_samples, n)

        batch_size = 50
        for method in ci_methods:
            for batch_start in range(0, config.n_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.n_samples)
                batch_samples = samples[batch_start:batch_end]
                CIs = compute_CIs_segmentation(
                    batch_samples, method, config.summary_stat,
                    statistic, config.trimmed_mean_threshold, a, b
                )

                lower_bounds = CIs[:, 0]
                upper_bounds = CIs[:, 1]
                widths = upper_bounds - lower_bounds
                contains_true = (lower_bounds <= true_value) & (true_value <= upper_bounds)
                proportion_oob = (
                    (lower_bounds < 0) * (-lower_bounds)
                    + (upper_bounds > 1) * (upper_bounds - 1)
                ) / widths

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

    # ── Merge new results with existing, deduplicate, sort, and write ──

    new_results = pd.DataFrame(data=all_rows.values())

    if existing_results is not None and not new_results.empty:
        results = pd.concat([existing_results, new_results], ignore_index=True)
    elif existing_results is not None:
        results = existing_results
    elif not new_results.empty:
        results = new_results
    else:
        results = pd.DataFrame()

    if not results.empty:
        key_cols = ["subtask", "alg_name", "n", "sample_index"]
        results = results.drop_duplicates(subset=key_cols, keep="last")
        results = results.sort_values(by=key_cols).reset_index(drop=True)

        # ── Aggregated results (recomputed from the full merged raw data) ──
        group_cols = ["subtask", "alg_name", "n"]
        avg_dfs = []
        for method in ci_methods:
            avg_df = (
                results.groupby(group_cols)
                .agg({
                    f"contains_true_stat_{method}": "mean",
                    f"width_{method}": "mean",
                    f"proportion_oob_{method}": "mean",
                })
                .reset_index()
                .rename(columns={f"contains_true_stat_{method}": f"coverage_{method}"})
            )
            avg_dfs.append(avg_df)

        if avg_dfs:
            average_results = avg_dfs[0]
            for df in avg_dfs[1:]:
                average_results = pd.merge(average_results, df, on=group_cols, how="outer")
            average_results = average_results.sort_values(by=group_cols).reset_index(drop=True)
        else:
            average_results = pd.DataFrame()

        # ── Write both files ──
        results.to_csv(raw_output_path, index=False)
        average_results.to_csv(agg_output_path, index=False)

@hydra.main(config_path="cfg", version_base="1.3.2")
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