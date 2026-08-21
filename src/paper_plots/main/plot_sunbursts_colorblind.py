# plot_sunbursts_colorblind.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse

from ..plot_utils import upload_to_overleaf
from ..df_loaders import extract_df_segm_cov, extract_df_classif_cov
from .sunburst_colorblind import draw_sunburst

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


def _build_axes(fig, n_groups):
    """Lay out ``n_groups`` sunburst axes as an inverted triangle
    (2 on top, 1 centered below) when there are exactly 3 groups.
    Falls back to a tightly-stacked single column otherwise.
    """
    if n_groups == 3:
        gs = fig.add_gridspec(2, 4, hspace=0.02, wspace=0.02)
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        gs = fig.add_gridspec(n_groups, 1, hspace=0.02)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_groups)]

    for ax in axes:
        ax.set_aspect('equal')
    return axes


def plot_all_sunbursts(root_folder: str, output_path: str, upload_overleaf: bool = False):
    df_all = extract_and_merge_data(root_folder)

    regimes = sorted(df_all["n"].unique())
    nb_regime_groups = 3
    regime_groups = np.array_split(regimes, nb_regime_groups)

    n_groups = len(regime_groups)

    # ── One figure, sunbursts arranged in an inverted triangle,
    #    with a single coverage colorbar next to the bottom plot only ──
    n_rows = 2 if n_groups == 3 else n_groups
    fig = plt.figure(figsize=(36, 16 * n_rows))
    axes = _build_axes(fig, n_groups)

    for i, (ax, regime_group) in enumerate(zip(axes, regime_groups)):
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
            add_colorbar=(i == len(axes) - 1),
        )

    stem, ext = os.path.splitext(output_path)
    ext = ext or '.pdf'
    plt.savefig(output_path, format=ext.lstrip('.'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    if upload_overleaf:
            upload_to_overleaf(output_path, f"Preprint/main_figs/{os.path.basename(output_path)}", commit_msg="Update Sunburst Plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--output_path", required=False)
    parser.add_argument("--upload_overleaf", action="store_true")
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(
        root_folder, "clean_figs/main/sunburst_colorblind.pdf"
    )
    upload_overleaf = args.upload_overleaf
    plot_all_sunbursts(root_folder, output_path, upload_overleaf=upload_overleaf)