"""Fuse per-(task, algorithm) aggregated CSV files into a single CSV per
(metric, grouping) pair.

When the SLURM array pipeline finishes, each job writes its own small CSV
(e.g. ``aggregated_results_dice_mean_liver_nnunet.csv``).  This script
collects all such files that belong to the same (metric, summary_stat) or
(metric, average) pair, concatenates and deduplicates them, and produces
one consolidated CSV (e.g. ``aggregated_results_dice_mean.csv``).  The
originals can optionally be deleted after fusion.

The set of pairs to fuse is read from the same sweep YAML consumed by
:mod:`build_task_list`, keeping the two scripts consistent.

Usage
-----
::

    python src/utils/merge_dataframes.py \\
        --results_dir results/segm/ablations/gaussian_adaptive \\
        --sweep_file src/cfg/sweep/segm_all_pairs.yaml \\
        --task_type segm \\
        --delete_files
"""

import os
import glob
import argparse

import pandas as pd
import yaml


def load_yaml(path):
    """Read a YAML file and return its contents as a native Python object.

    Parameters
    ----------
    path : str
        Filesystem path to the YAML file.

    Returns
    -------
    dict or list
        The deserialised YAML content (typically a list of dicts or a
        single dict wrapping such a list).
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fuse_aggregated_results(results_dir, pattern, output_name, delete_files=False):
    """Concatenate, deduplicate, and write all CSVs matching a glob pattern.

    Files that match *pattern* inside *results_dir* are read into
    DataFrames, concatenated, and deduplicated on the key columns
    ``["subtask", "alg_name", "n"]`` (keeping the last occurrence).  If
    the fused output file already exists it is loaded and merged with the
    new data so that incremental re-runs accumulate results rather than
    overwriting them.

    Parameters
    ----------
    results_dir : str
        Directory in which to search for CSV files.
    pattern : str
        Glob pattern (relative to *results_dir*) selecting the per-job
        CSV files, e.g.
        ``"aggregated_results_dice_mean_*.csv"``.
    output_name : str
        Filename (not path) for the fused output CSV, written inside
        *results_dir*.  This file is excluded from the glob so it is
        never read as input to itself.
    delete_files : bool, optional
        If *True*, delete every matched source file after successful
        fusion.  Defaults to *False*.
    """
    matched_files = sorted(glob.glob(os.path.join(results_dir, pattern)))

    # Don't pick up the fused output file itself
    output_path = os.path.join(results_dir, output_name)
    matched_files = [
        f for f in matched_files
        if os.path.abspath(f) != os.path.abspath(output_path)
    ]

    if not matched_files:
        print(f"  No files found for pattern: {pattern}")
        return

    dfs = []
    for fpath in matched_files:
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")

    if not dfs:
        print(f"  No valid data for pattern: {pattern}")
        return

    combined = pd.concat(dfs, ignore_index=True)

    key_cols = ["subtask", "alg_name", "n"]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(by=key_cols).reset_index(drop=True)

    # If the fused file already exists, merge with it
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined = combined.sort_values(by=key_cols).reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"  Wrote fused file: {output_path} ({len(combined)} rows)")

    # Delete original per-(task, algo) files
    if delete_files:
        for fpath in matched_files:
            os.remove(fpath)
            print(f"    Deleted: {fpath}")


def extract_pairs(sweep_file, task_type):
    """Read a sweep YAML and return unique (metric, grouping-value) pairs.

    The sweep file is expected to contain a list of dicts, each with a
    ``"metric"`` key and a grouping key that depends on the task type
    (``"summary_stat"`` for segmentation, ``"average"`` for
    classification).  The list may appear at the top level or nested
    under a single wrapper key.

    Parameters
    ----------
    sweep_file : str
        Filesystem path to the sweep YAML file.
    task_type : str
        Either ``"segm"`` (segmentation — looks for ``"summary_stat"``)
        or ``"classif"`` (classification — looks for ``"average"``).

    Returns
    -------
    pairs : list of tuple[str, str]
        Unique ``(metric, grouping_value)`` pairs in the order they first
        appear in the file.
    group_key : str
        The grouping key used (``"summary_stat"`` or ``"average"``).

    Raises
    ------
    ValueError
        If *task_type* is not ``"segm"`` or ``"classif"``, or if the YAML
        structure cannot be interpreted as a list of pair dicts.
    """
    entries = load_yaml(sweep_file)

    # Accept either a bare list or a dict with a single top-level key
    if isinstance(entries, dict):
        # e.g. {"pairs": [...]} or {"sweep": [...]}
        if len(entries) == 1:
            entries = list(entries.values())[0]
        elif "pairs" in entries:
            entries = entries["pairs"]
        else:
            raise ValueError(f"Cannot interpret sweep file structure: {list(entries.keys())}")

    if task_type == "segm":
        group_key = "summary_stat"
    elif task_type == "classif":
        group_key = "average"
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    seen = set()
    pairs = []
    for entry in entries:
        metric = entry["metric"]
        group_val = entry[group_key]
        key = (metric, group_val)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    return pairs, group_key


def main():
    """Parse CLI arguments and fuse CSVs for every (metric, grouping) pair.

    Iterates over the pairs extracted from the sweep file and delegates
    each one to :func:`fuse_aggregated_results`, using a glob pattern
    that matches the per-(task, algorithm) CSV naming convention produced
    by the SLURM array pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Fuse per-(task, algo) aggregated CSVs into one CSV per (metric, summary_stat/average)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing the aggregated_results CSV files.",
    )
    parser.add_argument(
        "--sweep_file",
        type=str,
        required=True,
        help="Path to the sweep YAML listing the (metric, summary_stat/average) pairs.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["segm", "classif"],
        help="Type of task: 'segm' (uses summary_stat) or 'classif' (uses average).",
    )
    parser.add_argument(
        "--delete_files",
        action="store_true",
        help="If set, delete the original per-(task, algo) CSV files after fusion. (Default: False)",
    )
    args = parser.parse_args()

    pairs, group_key = extract_pairs(args.sweep_file, args.task_type)

    print(f"Found {len(pairs)} unique (metric, {group_key}) pairs to fuse.\n")

    for metric, group_val in pairs:
        print(f"[{args.task_type}] Fusing metric={metric}, {group_key}={group_val}")

        # Matches: aggregated_results_{metric}_{group_val}_{task}_{algo}.csv
        # but NOT: aggregated_results_{metric}_{group_val}.csv (the fused output)
        pattern = f"aggregated_results_{metric}_{group_val}_*.csv"
        output_name = f"aggregated_results_{metric}_{group_val}.csv"

        fuse_aggregated_results(args.results_dir, pattern, output_name, delete_files=args.delete_files)
        print()


if __name__ == "__main__":
    main()