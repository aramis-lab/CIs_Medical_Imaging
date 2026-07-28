"""Core bootstrap coverage-evaluation pipeline for classification and
segmentation confidence intervals.

This module contains the two workhorse functions that drive every
experiment, whether launched from a SLURM array job, a Hydra multirun, or
the single-experiment script :mod:`run_single`:

* :func:`make_kdes_classification` — fits a multivariate KDE to the logit
  vectors of a single (task, algorithm) pair, draws synthetic datasets at
  each requested sample size, computes confidence intervals with every
  authorized CI method, and records per-sample coverage and width.
* :func:`make_kdes_segmentation` — does the same for scalar per-case
  segmentation metrics, using a 1-D KDE (or empirical resampling for
  discrete metrics).

Both functions persist **raw** per-sample results and an **aggregated**
coverage summary as CSV files.  Existing results are loaded and merged
incrementally so that interrupted runs can be resumed without re-computing
sample sizes that have already been completed.

The Hydra entry point :func:`main` at the bottom of the file inspects the
configured metric name to dispatch to the appropriate function.
"""

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
    """Run the bootstrap CI coverage experiment for one classification instance.

    The function fits a multivariate adaptive-bandwidth KDE to the raw
    logit vectors of a single (task, algorithm) pair, computes a
    ground-truth metric value from a large synthetic population
    (100 000 samples), then — for every requested sample size — draws
    ``n_samples`` synthetic datasets, constructs confidence intervals with
    each authorized CI method, and checks whether they cover the ground
    truth.

    Results are written to two CSV files:

    * **Raw** — one row per (sample-size, replicate, method), containing
      bounds, coverage flag, width, and proportion out of bounds.
    * **Aggregated** — one row per (sample-size, method), containing mean
      coverage, mean width, and mean proportion out of bounds.

    If the raw file already exists, finished sample sizes are skipped and
    new rows are merged in, allowing safe resumption of interrupted runs.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered DataFrame for the desired task, as returned by
        :func:`utils.extract_df`.  Must contain ``"alg_name"``,
        ``"logits"``, and ``"target"`` columns.
    task : str
        Subtask name (used for filtering and as a key in output files).
    algo : str
        Algorithm name (used for filtering and as a key in output files).
    config : omegaconf.DictConfig
        Experiment configuration.  Expected fields:

        * ``ci_methods`` — list of CI method names to evaluate.
        * ``metric`` — classification metric name.
        * ``average`` — averaging strategy (``"micro"`` or ``"macro"``).
        * ``kernel`` — kernel name passed to :func:`kernels.get_kernel`.
        * ``adaptive_bandwidth`` — whether to use adaptive KDE bandwidths.
        * ``sample_sizes`` — list of sample sizes *n* to sweep over.
        * ``n_samples`` — number of synthetic replicates per sample size.
        * ``n_bootstrap`` — number of bootstrap resamples inside each CI
          method.
        * ``relative_output_dir`` — results directory relative to
          *BASE_DIR*.
        * ``relative_data_path`` — data CSV path relative to *BASE_DIR*.
    """
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
    indices = np.where(lengths == good_length)
    values = np.array([v for v in values if len(v) == good_length])
    labels = df[df["alg_name"].astype(str) == algo]["target"].to_numpy()[indices]

    if len(values) < 50:
        print(f"Not enough values for {task} {algo} ({len(values)}), skipping KDE")
        return

    if np.any(np.isnan(values)):  # There should be no NaNs in the logits, but just in case
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
        alphas = (initial_estimates / g) ** (-1 / 2)
    y_score, y_true = sample_weighted_kde_multivariate(values, labels, config.kernel, 100000, alphas)  # Shapes (1000000, n_classes) and (1000000,), not binary
    y_score = softmax(y_score)

    n_classes = y_score.shape[-1]
    y_pred = np.argmax(y_score, axis=-1)

    correct_pred = (y_pred == y_true)[..., None]  # To allow bootstrapping metric arguments

    y_true_bin = label_binarize_vectorized(y_true, n_classes)
    y_pred_bin = label_binarize_vectorized(y_pred, n_classes)

    tp = (y_true_bin == 1) & (y_pred_bin == 1)
    fp = (y_true_bin == 0) & (y_pred_bin == 1)
    tn = (y_true_bin == 0) & (y_pred_bin == 0)
    fn = (y_true_bin == 1) & (y_pred_bin == 0)

    metric_arguments = {"accuracy": ["correct_pred"],
                        "precision": ["tp", "fp"],
                        "recall": ["tp", "fn"],
                        "f1_score": ["tp", "fp", "fn"],
                        "fbeta_score": ["tp", "fp", "fn"],
                        "npv": ["tn", "fn"],
                        "ppv": ["tp", "fp"],
                        "sensitivity": ["tp", "fn"],
                        "specificity": ["tn", "fp"],
                        "balanced_accuracy": ["tp", "fp", "tn", "fn"],
                        "mcc": ["tp", "fp", "fn"],
                        "auroc": ["y_score", "y_true_bin"],
                        "auc": ["y_score", "y_true_bin"],
                        "ap": ["y_score", "y_true_bin"]
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
    """Run the bootstrap CI coverage experiment for one segmentation instance.

    The function fits a 1-D adaptive-bandwidth KDE to the per-case metric
    values of a single (task, algorithm) pair (or falls back to empirical
    resampling for discrete metrics), computes a ground-truth summary
    statistic from a large synthetic population (1 000 000 samples), then
    — for every requested sample size — draws ``n_samples`` synthetic
    datasets, constructs confidence intervals with each authorized CI
    method, and checks whether they cover the ground truth.

    Results are written to two CSV files with the same structure and
    incremental-merge logic described in
    :func:`make_kdes_classification`.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-filtered DataFrame for the desired task, as returned by
        :func:`utils.extract_df`.  Must contain ``"alg_name"`` and
        ``"value"`` columns.
    task : str
        Subtask name (used for filtering and as a key in output files).
    algo : str
        Algorithm name (used for filtering and as a key in output files).
    config : omegaconf.DictConfig
        Experiment configuration.  Expected fields:

        * ``ci_methods`` — list of CI method names to evaluate.
        * ``metric`` — segmentation metric name (e.g. ``"dsc"``,
          ``"hd"``).
        * ``summary_stat`` — summary statistic applied to each synthetic
          dataset (e.g. ``"mean"``, ``"median"``).
        * ``trimmed_mean_threshold`` — proportion of observations trimmed
          from each tail when ``summary_stat`` is ``"trimmed_mean"``.
        * ``trim_bandwidth`` — whether to clamp each observation's KDE
          bandwidth to its distance to the metric domain boundary.
        * ``kernel`` — kernel name passed to :func:`kernels.get_kernel`.
        * ``adaptive_bandwidth`` — whether to use adaptive KDE bandwidths.
        * ``sample_sizes`` — list of sample sizes *n* to sweep over.
        * ``n_samples`` — number of synthetic replicates per sample size.
        * ``relative_output_dir`` — results directory relative to
          *BASE_DIR*.
        * ``relative_data_path`` — data CSV path relative to *BASE_DIR*.
    """
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

        dist_to_bounds = np.inf * np.ones(len(values))
        if config.trim_bandwidth:
            dist_to_bounds = np.min([values - a, b - values], axis=0)

        # Iterative weighted KDE estimation
        y = weighted_kde(values, x, dist_to_bounds, kernel, alphas)
        if config.adaptive_bandwidth:
            indices = np.searchsorted(x, values)
            initial_estimates = y[indices]
            log_g = np.mean(np.log(initial_estimates))
            g = np.exp(log_g)
            alphas = (initial_estimates / g) ** (-1 / 2)
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
    """Hydra entry point: dispatch a single (task, algorithm) experiment.

    Loads the benchmark CSV, extracts the rows for the configured task and
    metric, and delegates to :func:`make_kdes_classification` or
    :func:`make_kdes_segmentation` depending on whether the metric name
    belongs to the set of known classification metrics.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.  Must contain at least ``metric``,
        ``task``, ``algo``, ``relative_data_path``, and every field
        expected by the downstream ``make_kdes_*`` function.
    """
    print(f"Running KDE for metric {cfg.metric}, subtask {cfg.task} and algorithm {cfg.algo}")
    path = os.path.join(BASE_DIR, cfg.relative_data_path)
    df = extract_df(path, cfg.metric, cfg.task)
    if cfg.metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "mcc", "ap", "auroc", "auc"]:
        make_kdes_classification(df, cfg.task, str(cfg.algo), cfg)
    else:
        make_kdes_segmentation(df, cfg.task, str(cfg.algo), cfg)


if __name__ == "__main__":
    main()