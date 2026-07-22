import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from ...kde import sample_weighted_kde_multivariate
from ...kernels import get_kernel
from ...intervals_and_metrics import softmax
from itertools import product
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

kernel_name = "gaussian"  # For multivariate KDE, we use Gaussian kernel
# ─────────────────────────────────────────────────────────────
#  KDE sampling (unchanged from your code)
# ─────────────────────────────────────────────────────────────

def compute_instance(df_, task, algo):
    """Run adaptive KDE for one (task, algo) pair."""
    df = df_[(df_["subtask"] == task) & (df_["alg_name"] == algo)]

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
        return None

    if np.any(np.isnan(values)): # There should be no NaNs in the logits, but just in case
        print("There are NaNs in the data, skipping to next instance")
        return

    kernel = get_kernel(kernel_name)

    # Define the grid for KDE
    alphas = np.ones(len(values))

    # Iterative weighted KDE estimation
    initial_estimates = kernel(values, values, alphas)
    initial_estimates = np.mean(initial_estimates, axis=1)
    log_g = np.mean(np.log(initial_estimates))
    g = np.exp(log_g)
    alphas = (initial_estimates / g) ** (-1/2)
    y_score, y_true = sample_weighted_kde_multivariate(values, labels, kernel_name, 100000, alphas)
    y_score = softmax(y_score)
    return y_score, y_true


# ─────────────────────────────────────────────────────────────
#  Per-class rank-preservation evaluation for one (algo, task)
# ─────────────────────────────────────────────────────────────


def evaluate_rank_preservation_per_class(
    scores_orig, labels_orig,
    scores_kde, labels_kde,
    grid_size=200,
):
    """
    For one (task, algo), evaluate per-class (one-vs-rest) rank
    preservation between original and KDE-sampled scores.

    For each class c: label==c is "positive", every other label is
    "negative" — this matches how sklearn computes multiclass
    AUROC/AP under the hood, instead of collapsing all non-1 labels
    into a single negative bucket.

    Computes per-class AUROC/AP (and their |orig - kde| error, used
    later to report an MAE annotation on the histograms) as well as
    the *entire* ROC and PR curves for both original and KDE scores,
    interpolated onto a common grid and integrated as the pointwise
    absolute difference between the two curves — this is what's
    actually histogrammed downstream.

    Parameters
    ----------
    scores_orig : (N, C) original scores (softmax'd)
    labels_orig : (N,) integer class labels
    scores_kde  : (M, C) KDE-sampled scores (softmax'd)
    labels_kde  : (M,) integer class labels
    grid_size   : number of points used for curve interpolation/integration

    Returns
    -------
    list of dicts, one per class that had enough samples (possibly empty)
    """
    n_classes = scores_orig.shape[1]
    grid_fpr = np.linspace(0, 1, grid_size)
    grid_recall = np.linspace(0, 1, grid_size)

    rows = []
    for c in range(n_classes):
        y_orig = (labels_orig == c).astype(int)
        y_kde = (labels_kde == c).astype(int)

        n_pos_orig, n_neg_orig = y_orig.sum(), len(y_orig) - y_orig.sum()
        n_pos_kde, n_neg_kde = y_kde.sum(), len(y_kde) - y_kde.sum()

        if n_pos_orig < 5 or n_neg_orig < 5:
            continue
        if n_pos_kde < 5 or n_neg_kde < 5:
            continue

        s_orig = scores_orig[:, c]
        s_kde = scores_kde[:, c]

        # ── AUROC / AP (per-class, one-vs-rest) ─────────────
        auroc_orig = roc_auc_score(y_orig, s_orig)
        auroc_kde = roc_auc_score(y_kde, s_kde)
        ap_orig = average_precision_score(y_orig, s_orig)
        ap_kde = average_precision_score(y_kde, s_kde)

        # ── ROC curve: integral of |ΔTPR| over FPR ∈ [0,1] ──
        fpr_o, tpr_o, _ = roc_curve(y_orig, s_orig)
        fpr_k, tpr_k, _ = roc_curve(y_kde, s_kde)
        tpr_o_grid = np.interp(grid_fpr, fpr_o, tpr_o)
        tpr_k_grid = np.interp(grid_fpr, fpr_k, tpr_k)
        roc_integral_diff = np.trapezoid(np.abs(tpr_o_grid - tpr_k_grid), grid_fpr)

        # ── PR curve: integral of |ΔPrecision| over recall ∈ [0,1] ──
        prec_o, rec_o, _ = precision_recall_curve(y_orig, s_orig)
        prec_k, rec_k, _ = precision_recall_curve(y_kde, s_kde)
        order_o = np.argsort(rec_o)
        order_k = np.argsort(rec_k)
        prec_o_grid = np.interp(grid_recall, rec_o[order_o], prec_o[order_o])
        prec_k_grid = np.interp(grid_recall, rec_k[order_k], prec_k[order_k])
        pr_integral_diff = np.trapezoid(np.abs(prec_o_grid - prec_k_grid), grid_recall)

        rows.append({
            "class": c,
            "n_classes": n_classes,
            "auroc_orig": auroc_orig,
            "auroc_kde": auroc_kde,
            "auroc_abs_err": abs(auroc_orig - auroc_kde),
            "ap_orig": ap_orig,
            "ap_kde": ap_kde,
            "ap_abs_err": abs(ap_orig - ap_kde),
            "roc_integral_diff": roc_integral_diff,
            "pr_integral_diff": pr_integral_diff,
        })

    return rows


# ─────────────────────────────────────────────────────────────
#  Dashboard and summary
# ─────────────────────────────────────────────────────────────

def aggregate_and_plot(results_df, output_folder="."):
    """
    2×n_dims diagnostic dashboard, stratified by task dimensionality
    (n_classes) rather than class index — n_dims is inferred from the
    data (max n_classes seen).

    Row 0 (ROC): histogram of roc_integral_diff
                 = ∫|TPR_orig(f) - TPR_kde(f)| df
    Row 1 (PR):  histogram of pr_integral_diff
                 = ∫|P_orig(r) - P_kde(r)| dr

    Each subplot is annotated with the MAE that the (metric_orig,
    metric_kde) scatter plot / first-bisector comparison would show
    for that dimension, i.e. mean(|AUROC_orig - AUROC_kde|) for the
    ROC row and mean(|AP_orig - AP_kde|) for the PR row — even though
    that scatter plot itself is no longer drawn.
    """
    os.makedirs(output_folder, exist_ok=True)

    if len(results_df) == 0 or "n_classes" not in results_df.columns:
        raise ValueError("results_df is empty or missing 'n_classes' column")

    n_dims = int(results_df["n_classes"].max())
    fig, axes = plt.subplots(
        2, n_dims-1, figsize=(3.2 * (n_dims-1), 7), sharex=True, sharey="row",
        squeeze=False,
    )

    summary_rows = []

    max_roc_diff = results_df["roc_integral_diff"].max()
    max_pr_diff = results_df["pr_integral_diff"].max()

    max_diff = max(max_roc_diff, max_pr_diff)

    bins = np.linspace(0, max_diff, 30)

    for col in range(n_dims-1):
        dim = col + 2  # dim = 2, 3, ..., n_dims
        dim_df = results_df[results_df["n_classes"] == dim]

        # ── Row 0: ROC curve integral-of-|Δ| histogram ──────────
        ax = axes[0, col]
        if len(dim_df) > 0:
            roc_diff = dim_df["roc_integral_diff"]
            auroc_mae = dim_df["auroc_abs_err"].mean()

            ax.hist(roc_diff, bins=bins, density=True, alpha=0.7, edgecolor="k")
            ax.axvline(
                roc_diff.median(), color="red", ls="--",
                label=f"Med = {roc_diff.median():.4f}"
            )
            ax.text(
                0.97, 0.95, f"MAE(AUROC)\n= {auroc_mae:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
            )
            ax.legend(fontsize=7, loc="upper left")

            summary_rows.append({
                "n_classes": dim, "curve": "ROC",
                "n": len(dim_df),
                "median_integral_diff": roc_diff.median(),
                "p95_integral_diff": np.percentile(roc_diff, 95),
                "mae_auroc": auroc_mae,
            })
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_title(f"{dim} classes")
        ax.set_xlabel(r"ROC: $\int_0^1 |TPR_{orig} - TPR_{KDE}|\,df$")

        # ── Row 1: PR curve integral-of-|Δ| histogram ───────────
        ax = axes[1, col]
        if len(dim_df) > 0:
            pr_diff = dim_df["pr_integral_diff"]
            ap_mae = dim_df["ap_abs_err"].mean()

            ax.hist(pr_diff, bins=bins, density=True, alpha=0.7, edgecolor="k")
            ax.axvline(
                pr_diff.median(), color="red", ls="--",
                label=f"Med = {pr_diff.median():.4f}"
            )
            ax.text(
                0.97, 0.95, f"MAE(AP)\n= {ap_mae:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
            )
            ax.legend(fontsize=7, loc="upper left")

            summary_rows.append({
                "n_classes": dim, "curve": "PR",
                "n": len(dim_df),
                "median_integral_diff": pr_diff.median(),
                "p95_integral_diff": np.percentile(pr_diff, 95),
                "mae_ap": ap_mae,
            })
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_xlabel(r"PR: $\int_0^1 |P_{orig} - P_{KDE}|\,dr$")

    plt.suptitle(
        "ROC and PR l1 norm between original samples and KDE-based curves, stratified by task dimensionality\n"
        "(inset: MAE of the AUROC/AP orig-vs-KDE point comparison)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(
        os.path.join(output_folder, "rank_preservation.pdf"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    summary = pd.DataFrame(summary_rows)
    return summary


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    df_name = "data_matrix_classification.csv"
    df_path = os.path.join(BASE_DIR, df_name)

    parser = argparse.ArgumentParser(
        description="KDE Rank-Preservation Diagnostics (per-class AUROC/AP)"
    )
    parser.add_argument("--output_folder", type=str, default=".",
                        help="Folder to save output plots and summary")
    args = parser.parse_args()

    df_all = pd.read_csv(df_path)
    tasks = df_all["subtask"].unique()
    algos = df_all["alg_name"].unique()

    results = []

    combos = list(product(tasks, algos))

    for task, algo in tqdm(combos, desc="Rank preservation"):

        # ── Get original values ──
        logits_str = df_all[(df_all["alg_name"].astype(str) == algo) & (df_all["subtask"] == task)]["logits"]
        orig_values = [list(eval(v, {"nan": np.nan})) for v in logits_str]
        if len(orig_values) == 0:
            print(f"Not enough values for {task} {algo} ({len(orig_values)}), skipping KDE")
            continue
        lengths = np.array([len(v) for v in orig_values])
        good_length = round(np.mean(lengths))
        indices = np.where(lengths==good_length)
        orig_values = np.array([v for v in orig_values if len(v)==good_length])
        orig_scores = softmax(orig_values)
        orig_labels = df_all[(df_all["alg_name"].astype(str) == algo) & (df_all["subtask"] == task) ]["target"].to_numpy()[indices]

        # ── Get KDE samples ──
        kde_scores, kde_labels = compute_instance(df_all, task, str(algo))
        if kde_scores is None:
            continue

        # ── Evaluate, per class ──
        # NOTE: use orig_scores (softmax'd), not orig_values (raw
        # logits), so that original and KDE scores live in the same
        # space — kde_scores are already softmax'd upstream.
        per_class_rows = evaluate_rank_preservation_per_class(
            orig_scores, orig_labels,
            kde_scores, kde_labels
        )
        if not per_class_rows:
            continue

        for res in per_class_rows:
            res["task"] = task
            res["algo"] = str(algo)
            res["n_samples"] = len(orig_values)
            results.append(res)

    # ── Aggregate ──
    results_df = pd.DataFrame(results)

    out_dir = os.path.join(
        args.output_folder, "clean_figs/supplementary"
    )
    summary = aggregate_and_plot(results_df, output_folder=out_dir)