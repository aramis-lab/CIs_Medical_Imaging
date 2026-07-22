# plot_sunbursts.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

from ..df_loaders import extract_df_segm_cov, extract_df_classif_cov
from .sunburst import draw_sunburst

METHODS = ["percentile", "basic", "bca", "param_t", "wilson"]


def extract_and_merge_data(root_folder: str):
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "boundary_iou", "nsd", "cldice", "hd", "hd_perc", "masd", "assd"]
    stats_segm = ["mean", "median"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm["task_type"] = "segmentation"
    df_segm = df_segm[df_segm["method"].isin(METHODS)]

    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro = ["accuracy", "auc", "f1_score", "ap"]
    averages_micro = ["micro"]

    df_micro = extract_df_classif_cov(folder_path_micro, file_prefix_micro, metrics_micro, averages_micro)
    df_micro["task_type"] = "classification"
    df_micro["aggregation"] = "micro"
    df_micro = df_micro[df_micro["method"].isin(METHODS)]

    df_mcc = extract_df_classif_cov(folder_path_micro, file_prefix_micro, ["mcc"], ["none"])
    df_mcc["task_type"] = "classification"
    df_mcc["aggregation"] = None
    df_mcc = df_mcc[df_mcc["method"].isin(METHODS)]

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro = ["balanced_accuracy", "auc", "f1_score", "ap"]
    averages_macro = ["macro"]

    df_macro = extract_df_classif_cov(folder_path_macro, file_prefix_macro, metrics_macro, averages_macro)
    df_macro["task_type"] = "classification"
    df_macro["aggregation"] = "macro"
    df_macro = df_macro[df_macro["method"].isin(METHODS)]

    df_all = pd.concat([
        df_segm[["n", "method", "task_type", "stat", "metric", "task", "algo", "coverage"]],
        df_micro[["n", "method", "task_type", "aggregation", "metric", "task", "algo", "coverage"]],
        df_macro[["n", "method", "task_type", "aggregation", "metric", "task", "algo", "coverage"]],
        df_mcc[["n", "method", "task_type", "aggregation", "metric", "task", "algo", "coverage"]]
    ])

    group_cols = [c for c in df_all.columns if c not in ["task", "algo", "coverage"]]
    df_all = df_all.groupby(group_cols, dropna=False)["coverage"].mean().reset_index()

    df_all = df_all[df_all["n"]<=250]

    return df_all


def plot_all_sunbursts(root_folder: str, output_path: str):
    df_all = extract_and_merge_data(root_folder)

    regimes = sorted(df_all["n"].unique())
    nb_regime_groups = 3
    regime_groups = np.array_split(regimes, nb_regime_groups)

    n_groups = len(regime_groups)

    # ── One figure with one sunburst per regime group ──────────────
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(16, 16 * n_groups),
        subplot_kw=dict(aspect='equal'),
    )
    if n_groups == 1:
        axes = [axes]

    for ax, regime_group in zip(axes, regime_groups):
        df_group = df_all[df_all["n"].isin(regime_group)]

        # Average coverage across the n values within this regime group
        group_cols = [c for c in df_group.columns if c not in ["n", "coverage"]]
        df_group = df_group.groupby(group_cols, dropna=False)["coverage"].mean().reset_index()

        # Build the centre label:  "a ≤ n ≤ b"
        a, b = int(min(regime_group)), int(max(regime_group))
        if a == b:
            center_label = f"n = {a}"
        else:
            center_label = f"${a} ≤ n ≤ {b}$"

        draw_sunburst(
            df_group,
            fig=fig,
            ax=ax,
            center_label=center_label,
            center_fontsize=26,
        )

    plt.tight_layout()
    stem, ext = os.path.splitext(output_path)
    ext = ext or '.pdf'
    plt.savefig(output_path, format=ext.lstrip('.'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--output_path", required=False)
    parser.add_argument("--upload_overleaf", action="store_true")
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(
        root_folder, "clean_figs/supplementary/sunburst.pdf"
    )
    plot_all_sunbursts(root_folder, output_path)