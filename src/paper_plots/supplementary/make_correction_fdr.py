import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .test_basic import get_pvalues_basic, reconstruct_basic
from .test_basic_classif import get_pvalues_basic_classif, reconstruct_basic_classif
from .test_bca import get_pvalues_bca, reconstruct_bca
from .test_micro_vs_macro import get_pvalues_micro_macro, reconstruct_micro_macro
from .test_param_vs_bootstrap import get_pvalues_param_boot, reconstruct_param_boot
from .test_spread_vs_central import get_pvalues_spread_central, reconstruct_spread_central
from .test_WDP_segm_classif import get_pvalues_wdp_segm_classif, reconstruct_wdp_segm_classif
from .tests_CCP_segm import get_pvalues_segm, reconstruct_segm
from .tests_CCP_segm_vs_classif import get_pvalues_segm_classif, reconstruct_segm_classif

tests = [
    "basic_classif",
    "basic",
    "micro_macro",
    "param_boot",
    "spread_central",
    "wdp_segm_classif",
    "segm",
    "segm_classif",
    "bca",
    "bca_classif",
]

# For each test family: the flatten/reconstruct pair, and the (emitted test name, p-value
# filename) of every variant it produces. A family is skipped unless all of its files exist.
test_specs = {
    "basic_classif": {
        "get_pvalues": get_pvalues_basic_classif,
        "reconstruct": reconstruct_basic_classif,
        "variants": [
            ("basic_classif_micro", "pvalues_basic_classif_micro_by_n.json"),
            ("basic_classif_macro", "pvalues_basic_classif_macro_by_n.json"),
        ],
    },
    "basic": {
        "get_pvalues": get_pvalues_basic,
        "reconstruct": reconstruct_basic,
        "variants": [("basic", "pvalues_basic_by_n.json")],
    },
    "micro_macro": {
        "get_pvalues": get_pvalues_micro_macro,
        "reconstruct": reconstruct_micro_macro,
        "variants": [("micro_macro", "pvalues_micro_macro_by_n.json")],
    },
    "param_boot": {
        "get_pvalues": get_pvalues_param_boot,
        "reconstruct": reconstruct_param_boot,
        "variants": [
            ("param_boot_segm", "pvalues_param_boot_segm_by_n.json"),
            ("param_boot_classif", "pvalues_param_boot_classif_by_n.json"),
            ("param_boot_segm_width", "pvalues_param_boot_segm_width_by_n.json"),
        ],
    },
    "spread_central": {
        "get_pvalues": get_pvalues_spread_central,
        "reconstruct": reconstruct_spread_central,
        "variants": [("spread_central", "pvalues_spread_central_by_n.json")],
    },
    "wdp_segm_classif": {
        "get_pvalues": get_pvalues_wdp_segm_classif,
        "reconstruct": reconstruct_wdp_segm_classif,
        "variants": [
            ("wdp_segm_classif", "pvalues_segm_classif_width_by_n.json"),
            ("wdp_segm_classif_macro", "pvalues_segm_classif_macro_width_by_n.json"),
        ],
    },
    "segm": {
        "get_pvalues": get_pvalues_segm,
        "reconstruct": reconstruct_segm,
        "variants": [("segm", "pvalues_segm.json")],
    },
    "segm_classif": {
        "get_pvalues": get_pvalues_segm_classif,
        "reconstruct": reconstruct_segm_classif,
        "variants": [("segm_classif", "pvalues_segm_classif_by_n.json")],
    },
    "bca": {
        "get_pvalues": get_pvalues_bca,
        "reconstruct": reconstruct_bca,
        "variants": [("bca", "pvalues_bca_by_n.json")],
    },
    "bca_classif": {
        "get_pvalues": get_pvalues_bca,
        "reconstruct": reconstruct_bca,
        "variants": [("bca_classif", "pvalues_bca_classif_by_n.json")],
    },
}

# Variants that are reported as a single row in the ablation table
test_mapping = {
    "basic_classif_micro": "basic",
    "basic_classif_macro": "basic",
    "basic": "basic",

    "wdp_segm_classif": "wdp_segm_classif",
    "wdp_segm_classif_macro": "wdp_segm_classif",

    "bca": "bca",
    "bca_classif": "bca",
}

kde_mapping = {
    "epanechnikov_adaptive_trimmed": "epan.adapt.trimmed",
    "gaussian_adaptive": "gauss.adapt.",
    "epanechnikov_scott_trimmed": "epan.scott.trimmed",
    "gaussian_scott": "gauss.scott",
}

test_order = [
    "basic",
    "bca",
    "param_boot_segm",
    "param_boot_segm_width",
    "param_boot_classif",
    "segm_classif",
    "wdp_segm_classif",
    "segm",
    "micro_macro",
    "spread_central",
]

skipped_kdes = ["epanechnikov_scott_old", "epanechnikov_adaptive"]


def tell_significance(tests, pvalues_folder, alphas=np.array([0.001, 0.01, 0.05])):
    """
    Pool the p-values of every available test in `pvalues_folder`, apply a single
    BH-FDR correction across all of them, and reconstruct per-test nested dicts.

    Returns a DataFrame with columns: test, significance, pvalues, pvalues_corrected.
    """
    loaded = []
    for test in tests:
        spec = test_specs[test]
        paths = [os.path.join(pvalues_folder, filename) for _, filename in spec["variants"]]

        # a family contributes all of its variants or none of them
        if not all(os.path.exists(path) for path in paths):
            continue

        for (test_name, _), path in zip(spec["variants"], paths):
            with open(path, "r") as f:
                pvalues = json.load(f)
            flat_pvalues, keys = spec["get_pvalues"](pvalues)
            loaded.append({
                "test": test_name,
                "pvalues": pvalues,
                "flat_pvalues": flat_pvalues,
                "keys": keys,
                "reconstruct": spec["reconstruct"],
            })

    if not loaded:
        return pd.DataFrame(columns=["test", "significance", "pvalues", "pvalues_corrected"])

    all_pvalues = np.concatenate([row["flat_pvalues"] for row in loaded])
    _, qvalues, _, _ = multipletests(all_pvalues, method="fdr_bh")

    significance = []
    start = 0
    for row in loaded:
        length = len(row["flat_pvalues"])
        qvalues_test = qvalues[start:start + length]
        start += length

        q_vals_dict, significant_dict = row["reconstruct"](
            qvalues_test, row["keys"], row["pvalues"], alphas
        )
        significance.append({
            "test": row["test"],
            "significance": significant_dict,
            "pvalues": row["pvalues"],
            "pvalues_corrected": q_vals_dict,
        })

    return pd.DataFrame(significance)


def sum_leaves(obj):
    if isinstance(obj, dict):
        return np.nansum([sum_leaves(v) for v in obj.values()])
    else:
        return obj if obj is not None else 0


def count_non_none_leaves(obj):
    if isinstance(obj, dict):
        return sum(count_non_none_leaves(v) for v in obj.values())
    else:
        return 0 if obj is None else 1


def run_ablation_sweep(ablation_dir, tests=tests, alphas=np.array([0.05])):
    """
    Run the pooled FDR correction once per KDE variant found in `ablation_dir`,
    and count how many comparisons come out significant for each test.
    """
    results_ablation = []
    for kde in os.listdir(ablation_dir):
        if kde.startswith(".") or kde in skipped_kdes:
            continue

        pvalues_folder = os.path.join(ablation_dir, kde, "pvalues")
        if not os.path.isdir(pvalues_folder):
            continue

        print(kde)
        significance = tell_significance(tests, pvalues_folder, alphas=alphas)

        for _, row in significance.iterrows():
            significance_dict = row["significance"]
            total = count_non_none_leaves(significance_dict)
            significant = sum_leaves(significance_dict)
            results_ablation.append({
                "kde": kde,
                "test": row["test"],
                "sum of significance": significant,
                "total_number of tests": total,
                "proportion of not significance": 1 - significant / total if total else np.nan,
            })

    results_ablation_df = pd.DataFrame(results_ablation)
    if results_ablation_df.empty:
        return results_ablation_df

    results_ablation_df["test_grouped"] = results_ablation_df["test"].replace(test_mapping)
    results_ablation_df["kde"] = results_ablation_df["kde"].replace(kde_mapping)
    return results_ablation_df


def summarize_ablation(results_ablation_df):
    """Merge the grouped tests and pivot into the KDE-by-test table used in the paper."""
    df_merged = (
        results_ablation_df.groupby(["kde", "test_grouped"], as_index=False)
        .agg({
            "sum of significance": "sum",
            "total_number of tests": "sum",
        })
    )
    df_merged["proportion of not significance"] = round(
        (df_merged["sum of significance"] / df_merged["total_number of tests"]) * 100, 1
    )

    table = df_merged.pivot(
        index="test_grouped",
        columns="kde",
        values="proportion of not significance",
    )
    table = table.reindex(test_order)
    return df_merged, table


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Apply a pooled BH-FDR correction across all tests, for every KDE ablation."
    )
    parser.add_argument("--ablation_dir", required=True,
                        help="Folder containing one subfolder per KDE variant.")
    args = parser.parse_args()

    results_ablation_df = run_ablation_sweep(args.ablation_dir)
    if results_ablation_df.empty:
        print("No p-value files found.")
        return

    df_merged, table = summarize_ablation(results_ablation_df)
    print(df_merged)
    print(table.to_latex(float_format="%.1f"))
    print(table)


if __name__ == "__main__":
    main()