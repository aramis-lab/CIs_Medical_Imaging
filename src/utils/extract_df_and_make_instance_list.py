"""Extract and export the list of benchmark instances (task, algorithm)
for a given metric.

This script reads a CSV benchmark file, identifies every unique
(subtask, algorithm) pair for the requested metric, and writes them to a
simple text file under ``instances_list/``.  The resulting file is consumed
downstream by :mod:`build_task_list` to fan SLURM array jobs across all
relevant instances.

The script is driven by a Hydra configuration that must supply at least
``relative_data_path`` (path to the CSV relative to the project root),
``metric`` (the score column value to filter on), and optionally ``task``
(a whitelist of subtasks to keep).
"""

import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
import os


def extract_df(path, metric, task):
    """Load a benchmark CSV and return the rows for one (metric, task) pair.

    Depending on the columns present in the file the function returns
    either per-sample *values* (pre-computed metric scores) or raw
    *logits* and *targets* (for metrics that must be computed on the fly,
    such as AUC).

    Parameters
    ----------
    path : str
        Filesystem path to the benchmark CSV file.
    metric : str
        The metric name used to filter the ``"score"`` column, if that
        column exists.
    task : str
        The subtask name used to filter the ``"subtask"`` column.

    Returns
    -------
    pd.DataFrame
        A two- or three-column DataFrame:

        * ``["alg_name", "value"]`` when pre-computed scores are available.
        * ``["alg_name", "logits", "target"]`` when raw predictions are
          available instead.

    Raises
    ------
    ValueError
        If the filtered DataFrame contains neither the
        ``("alg_name", "value")`` nor the
        ``("alg_name", "logits", "target")`` column sets.
    """
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
    """Collect every unique (task, algorithm) pair for the configured metric.

    The function reads the full benchmark CSV once to discover all
    subtasks, optionally filters them to those listed in ``cfg.task``,
    then iterates over each subtask to collect the algorithms that
    appear.  The resulting list is sorted lexicographically by
    ``(task, algorithm)`` and deduplicated.

    Parameters
    ----------
    BASE_DIR : str
        Absolute path to the project root directory.
    cfg : DictConfig
        Hydra configuration object.  Expected keys:

        * ``relative_data_path`` — path to the CSV relative to
          *BASE_DIR*.
        * ``metric`` — metric name passed to :func:`extract_df`.
        * ``task`` (optional) — whitelist of subtask names to keep.

    Returns
    -------
    np.ndarray, shape ``(n, 2)``
        Each row is a ``(task, algorithm)`` string pair, sorted
        lexicographically and without duplicates.
    """
    benchmark_instances = []
    df_all = pd.read_csv(os.path.join(BASE_DIR, cfg.relative_data_path))
    tasks = df_all["subtask"].unique()
    if "task" in cfg:
        tasks = [task for task in tasks if task in cfg.task]
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
    """Write the (task, algorithm) instance list for the configured metric.

    Creates the ``instances_list/`` directory under the project root if it
    does not exist and writes one ``<metric>.txt`` file containing one
    ``task algorithm`` line per benchmark instance.  If the file already
    exists the function returns without overwriting it, ensuring
    idempotent execution across repeated runs.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object forwarded to
        :func:`get_benchmark_instances`.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    instances = get_benchmark_instances(BASE_DIR, cfg)
    if not os.path.exists(os.path.join(BASE_DIR, "instances_list")):
        os.makedirs(os.path.join(BASE_DIR, "instances_list"))
    if not os.path.exists(os.path.join(BASE_DIR, f"instances_list/{cfg.metric}.txt")):
        with open(os.path.join(BASE_DIR, f"instances_list/{cfg.metric}.txt"), "w") as f:
            for task, algo in instances:
                f.write(f"{task} {algo}\n")


if __name__ == "__main__":
    export_benchmark_list()