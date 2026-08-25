import os

import pandas as pd

from .make_correction_fdr import tell_significance, tests, skipped_kdes


def get_leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from get_leaves(value)
    else:
        yield obj


def collect_corrected_pvalues(ablation_dir, tests=tests):
    """
    Flatten the FDR-corrected p-values of every test, for every KDE ablation.

    Returns a long DataFrame with columns: kde, test, qvalue. Leaves that are None
    (comparisons that were never run) are dropped.
    """
    rows = []
    for kde in os.listdir(ablation_dir):
        if kde.startswith(".") or kde in skipped_kdes:
            continue

        pvalues_folder = os.path.join(ablation_dir, kde, "pvalues")
        if not os.path.isdir(pvalues_folder):
            continue

        print(kde)
        significance = tell_significance(tests, pvalues_folder)

        for _, row in significance.iterrows():
            for qvalue in get_leaves(row["pvalues_corrected"]):
                if qvalue is None:
                    continue
                rows.append({
                    "kde": kde,
                    "test": row["test"],
                    "qvalue": qvalue,
                })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect the FDR-corrected p-values of every test, for every KDE ablation."
    )
    parser.add_argument("--ablation_dir", required=True,
                        help="Folder containing one subfolder per KDE variant.")
    args = parser.parse_args()

    df_qvalues = collect_corrected_pvalues(args.ablation_dir)
    if df_qvalues.empty:
        print("No p-value files found.")
        return

    print(df_qvalues)
    print(
        df_qvalues.groupby(["kde", "test"])["qvalue"]
        .agg(["count", "min", "median", "max"])
    )


if __name__ == "__main__":
    main()