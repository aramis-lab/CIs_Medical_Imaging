import os
import glob
import argparse

import pandas as pd
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fuse_aggregated_results(results_dir, pattern, output_name):
    """
    Find all aggregated CSVs matching `pattern`, concatenate them,
    deduplicate, sort, write one fused file, and delete the originals.
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
    for fpath in matched_files:
        os.remove(fpath)
        print(f"    Deleted: {fpath}")


def extract_pairs(sweep_file, task_type):
    """
    Read the sweep YAML and extract unique (metric, group_key) pairs.

    Expected sweep YAML structure (list of dicts), e.g.:

    Segmentation:
      - metric: dice
        summary_stat: mean
      - metric: hausdorff
        summary_stat: median

    Classification:
      - metric: f1_score
        average: macro
      - metric: precision
        average: micro
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
    args = parser.parse_args()

    pairs, group_key = extract_pairs(args.sweep_file, args.task_type)

    print(f"Found {len(pairs)} unique (metric, {group_key}) pairs to fuse.\n")

    for metric, group_val in pairs:
        print(f"[{args.task_type}] Fusing metric={metric}, {group_key}={group_val}")

        # Matches: aggregated_results_{metric}_{group_val}_{task}_{algo}.csv
        # but NOT: aggregated_results_{metric}_{group_val}.csv (the fused output)
        pattern = f"aggregated_results_{metric}_{group_val}_*.csv"
        output_name = f"aggregated_results_{metric}_{group_val}.csv"

        fuse_aggregated_results(args.results_dir, pattern, output_name)
        print()


if __name__ == "__main__":
    main()