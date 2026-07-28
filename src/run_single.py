"""Run a single confidence-interval experiment from one self-contained YAML config.

This bypasses the SLURM / Hydra multirun machinery used by ``run_all.sh``
and lets you evaluate the coverage of confidence-interval methods for a
single metric, on a single task, for one algorithm (or every algorithm
found for that task), directly from the command line.

The config file must be a flat YAML containing every field normally spread
across ``default_classif.yaml`` / ``default_segm.yaml``, plus:

* ``task`` — *(required)* must match a ``"subtask"`` value in your data.
* ``algo`` — *(optional)* if omitted, every algorithm found for that
  task / metric in the dataset is run, and the coverage table aggregates
  across all of them.

See ``src/cfg/single_experiment_example.yaml`` (classification) and
``src/cfg/single_experiment_example_segm.yaml`` (segmentation) for
templates.

Usage
-----
::

    python src/run_single.py --config path/to/my_experiment.yaml
"""

import argparse
import os
import sys

import pandas as pd
from omegaconf import OmegaConf

# Make sibling packages (kde, kernels, utils, intervals_and_metrics, run, ...)
# importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import make_kdes_classification, make_kdes_segmentation, BASE_DIR  # noqa: E402
from utils import extract_df  # noqa: E402

CLASSIF_METRICS = {
    "accuracy", "npv", "ppv", "precision", "recall", "sensitivity",
    "specificity", "balanced_accuracy", "f1_score", "mcc", "ap", "auroc", "auc",
}

REQUIRED_COMMON = [
    "metric", "task", "kernel", "adaptive_bandwidth", "ci_methods",
    "sample_sizes", "relative_data_path", "relative_output_dir",
    "n_samples", "n_bootstrap",
]
REQUIRED_CLASSIF = ["average"]
REQUIRED_SEGM = ["summary_stat", "trimmed_mean_threshold", "trim_bandwidth"]


def load_config(path):
    """Load and validate a single-experiment YAML config.

    The metric name is used to infer whether the experiment is a
    classification or segmentation task.  The config is then checked for
    all fields required by the corresponding pipeline; a clear error
    message is raised if any are missing.

    Parameters
    ----------
    path : str
        Filesystem path to the YAML configuration file.

    Returns
    -------
    cfg : omegaconf.DictConfig
        The loaded configuration object.
    is_classif : bool
        *True* if the configured metric belongs to :data:`CLASSIF_METRICS`,
        *False* otherwise (segmentation path).

    Raises
    ------
    ValueError
        If one or more required fields are absent from the config.
    """
    cfg = OmegaConf.load(path)

    is_classif = cfg.get("metric") in CLASSIF_METRICS
    required = REQUIRED_COMMON + (REQUIRED_CLASSIF if is_classif else REQUIRED_SEGM)
    missing = [k for k in required if k not in cfg]
    if missing:
        kind = "classification" if is_classif else "segmentation"
        raise ValueError(
            f"Config '{path}' is missing required field(s) for a {kind} metric: {missing}. "
            f"See src/cfg/single_experiment_example{'' if is_classif else '_segm'}.yaml."
        )
    return cfg, is_classif


def resolve_algos_and_df(cfg):
    """Load the relevant data slice and determine which algorithms to run.

    If the config specifies an ``algo`` field the returned list contains
    only that single algorithm.  Otherwise every algorithm present in the
    dataset for the configured task and metric is included, allowing a
    single invocation to sweep over all of them.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration; must contain ``relative_data_path``,
        ``metric``, ``task``, and optionally ``algo``.

    Returns
    -------
    df : pd.DataFrame
        The filtered DataFrame returned by :func:`utils.extract_df`.
    algos : list of str
        Algorithm name(s) to evaluate.

    Raises
    ------
    ValueError
        If no algorithms are found for the given task / metric
        combination in the dataset.
    """
    data_path = os.path.join(BASE_DIR, cfg.relative_data_path)
    df = extract_df(data_path, cfg.metric, cfg.task)

    if cfg.get("algo") is not None:
        return df, [str(cfg.algo)]

    algos = sorted(df["alg_name"].astype(str).unique().tolist())
    if not algos:
        raise ValueError(
            f"No algorithm found for task='{cfg.task}' and metric='{cfg.metric}' "
            f"in {data_path}. Check that 'task' matches a 'subtask' value in your data."
        )
    print(f"No 'algo' specified in config: running all {len(algos)} algorithm(s) "
          f"found for task '{cfg.task}': {algos}")
    return df, algos


def run_experiment(cfg, is_classif, df, algos):
    """Execute the bootstrap coverage experiment for every requested algorithm.

    Delegates to :func:`run.make_kdes_classification` or
    :func:`run.make_kdes_segmentation` depending on the metric type.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    is_classif : bool
        *True* to use the classification pipeline, *False* for
        segmentation.
    df : pd.DataFrame
        Pre-filtered dataset (output of :func:`resolve_algos_and_df`).
    algos : list of str
        Algorithm name(s) to iterate over.
    """
    for algo in algos:
        print(f"\n=== Running {cfg.metric} / task={cfg.task} / algo={algo} ===")
        if is_classif:
            make_kdes_classification(df, cfg.task, algo, cfg)
        else:
            make_kdes_segmentation(df, cfg.task, algo, cfg)


def aggregated_paths(cfg, is_classif, algos):
    """Build the expected filesystem paths for the per-algorithm result CSVs.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration; must contain ``relative_output_dir``,
        ``metric``, ``task``, and either ``average`` (classification) or
        ``summary_stat`` (segmentation).
    is_classif : bool
        *True* for classification naming, *False* for segmentation.
    algos : list of str
        Algorithm name(s) whose result paths are needed.

    Returns
    -------
    list of str
        One absolute path per algorithm, following the naming convention
        ``aggregated_results_{metric}_{grouping}_{task}_{algo}.csv``.
    """
    results_dir = os.path.join(BASE_DIR, cfg.relative_output_dir)
    paths = []
    for algo in algos:
        if is_classif:
            fname = f"aggregated_results_{cfg.metric}_{cfg.average}_{cfg.task}_{algo}.csv"
        else:
            fname = f"aggregated_results_{cfg.metric}_{cfg.summary_stat}_{cfg.task}_{algo}.csv"
        paths.append(os.path.join(results_dir, fname))
    return paths


def print_coverage_summary(cfg, is_classif, algos):
    """Load result CSVs and print a human-readable coverage table to stdout.

    All per-algorithm result files are concatenated and the coverage
    columns (those prefixed with ``coverage_``) are displayed alongside
    ``subtask``, ``alg_name``, and ``n``.  When more than one algorithm
    was evaluated an additional table showing the mean coverage across
    algorithms per sample size is printed.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Experiment configuration.
    is_classif : bool
        *True* for classification naming, *False* for segmentation.
    algos : list of str
        Algorithm name(s) whose results should be summarised.
    """
    paths = aggregated_paths(cfg, is_classif, algos)
    dfs = []
    for p, algo in zip(paths, algos):
        if not os.path.exists(p):
            print(f"[warn] No results file for algo='{algo}' "
                  f"(not enough data, or nothing new to compute) — {p} not found.")
            continue
        dfs.append(pd.read_csv(p))

    if not dfs:
        print("\nNo coverage results were produced.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    coverage_cols = [c for c in combined.columns if c.startswith("coverage_")]
    display_cols = ["subtask", "alg_name", "n"] + coverage_cols

    group_label = f"average={cfg.average}" if is_classif else f"summary_stat={cfg.summary_stat}"
    print("\n" + "=" * 70)
    print(f"Coverage summary — metric={cfg.metric}, {group_label}, task={cfg.task}")
    print("=" * 70)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(combined[display_cols].sort_values(["alg_name", "n"]).to_string(index=False))

    if len(algos) > 1:
        print("\n--- Mean coverage across all algorithms, per n ---")
        mean_by_n = combined.groupby("n")[coverage_cols].mean().reset_index()
        with pd.option_context("display.width", 140):
            print(mean_by_n.to_string(index=False))


def main():
    """Parse CLI arguments and orchestrate the single-experiment pipeline.

    Sequentially calls :func:`load_config`, :func:`resolve_algos_and_df`,
    :func:`run_experiment`, and :func:`print_coverage_summary` to load
    the config, determine the algorithms, run the bootstrap coverage
    evaluation, and display the results.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a single-experiment YAML config.")
    args = parser.parse_args()

    cfg, is_classif = load_config(args.config)
    df, algos = resolve_algos_and_df(cfg)
    run_experiment(cfg, is_classif, df, algos)
    print_coverage_summary(cfg, is_classif, algos)


if __name__ == "__main__":
    main()