import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from ...kde import sample_weighted_kde_multivariate
from ...kernels import get_kernel
from ...intervals_and_metrics import softmax, label_binarize_vectorized
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

    In addition to per-class AUROC/AP, this computes the *entire*
    ROC and PR curves for both original and KDE scores, interpolates
    them onto a common grid, and integrates the pointwise absolute
    difference between the two curves. This is a stronger notion of
    "shape preservation" than comparing a handful of quantiles: it
    is literally the area between the orig and KDE curves.

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
        # fpr_* is already monotonically non-decreasing, safe for np.interp
        tpr_o_grid = np.interp(grid_fpr, fpr_o, tpr_o)
        tpr_k_grid = np.interp(grid_fpr, fpr_k, tpr_k)
        roc_integral_diff = np.trapezoid(np.abs(tpr_o_grid - tpr_k_grid), grid_fpr)

        # ── PR curve: integral of |ΔPrecision| over recall ∈ [0,1] ──
        prec_o, rec_o, _ = precision_recall_curve(y_orig, s_orig)
        prec_k, rec_k, _ = precision_recall_curve(y_kde, s_kde)
        # precision_recall_curve returns recall in decreasing order;
        # np.interp needs the x-coordinates sorted ascending
        order_o = np.argsort(rec_o)
        order_k = np.argsort(rec_k)
        prec_o_grid = np.interp(grid_recall, rec_o[order_o], prec_o[order_o])
        prec_k_grid = np.interp(grid_recall, rec_k[order_k], prec_k[order_k])
        pr_integral_diff = np.trapezoid(np.abs(prec_o_grid - prec_k_grid), grid_recall)

        rows.append({
            "class": c,
            "auroc_orig": auroc_orig,
            "auroc_kde": auroc_kde,
            "auroc_abs_err": abs(auroc_orig - auroc_kde),
            "ap_orig": ap_orig,
            "ap_kde": ap_kde,
            "ap_abs_err": abs(ap_orig - ap_kde),
            "roc_integral_diff": roc_integral_diff,
            "pr_integral_diff": pr_integral_diff,
            "n_pos_orig": int(n_pos_orig),
            "n_neg_orig": int(n_neg_orig),
        })

    return rows


# ─────────────────────────────────────────────────────────────
#  Dashboard and summary
# ─────────────────────────────────────────────────────────────

def aggregate_and_plot(results_df, output_folder="."):
    """
    2×2 diagnostic dashboard (per-class evaluation).

        (a) AUROC_orig vs AUROC_kde scatter
        (c) ROC curve area difference: ∫|TPR_orig(f) − TPR_kde(f)| df
        (d) AP_orig vs AP_kde scatter
        (f) PR curve area difference: ∫|P_orig(r) − P_kde(r)| dr
    """
    os.makedirs(output_folder, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ═══════════════════  (a) AUROC scatter  ═══════════════════
    ax = axes[0, 0]
    ax.scatter(results_df["auroc_orig"], results_df["auroc_kde"],
               s=8, alpha=0.3, rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5)
    r_auroc, _ = stats.pearsonr(results_df["auroc_orig"],
                                results_df["auroc_kde"])
    ax.set_xlabel("AUROC (original)")
    ax.set_ylabel("AUROC (KDE)")
    ax.set_title(f"(a) AUROC: orig vs KDE, per-class  (r = {r_auroc:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # ═══════════════  (c) ROC curve area difference  ═══════════
    ax = axes[0, 1]
    roc_diff = results_df["roc_integral_diff"]
    ax.hist(roc_diff, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(roc_diff.median(), color="red", ls="--",
               label=f"Median = {roc_diff.median():.4f}")
    ax.axvline(np.percentile(roc_diff, 95), color="orange", ls=":",
               label=f"95th pctl = {np.percentile(roc_diff, 95):.4f}")
    ax.set_xlabel(r"$\int_0^1 |TPR_{orig}(f) - TPR_{KDE}(f)|\,df$")
    ax.set_title("(b) ROC curve area difference, per-class")
    ax.legend()

    # ═══════════════════  (d) AP scatter  ═══════════════════════
    ax = axes[1, 0]
    ax.scatter(results_df["ap_orig"], results_df["ap_kde"],
               s=8, alpha=0.3, rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5)
    r_ap, _ = stats.pearsonr(results_df["ap_orig"],
                             results_df["ap_kde"])
    ax.set_xlabel("AP (original)")
    ax.set_ylabel("AP (KDE)")
    ax.set_title(f"(c) AP: orig vs KDE, per-class  (r = {r_ap:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # ═══════════════  (f) PR curve area difference  ═════════════
    ax = axes[1, 1]
    pr_diff = results_df["pr_integral_diff"]
    ax.hist(pr_diff, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(pr_diff.median(), color="red", ls="--",
               label=f"Median = {pr_diff.median():.4f}")
    ax.axvline(np.percentile(pr_diff, 95), color="orange", ls=":",
               label=f"95th pctl = {np.percentile(pr_diff, 95):.4f}")
    ax.set_xlabel(r"$\int_0^1 |P_{orig}(r) - P_{KDE}(r)|\,dr$")
    ax.set_title("(d) PR curve area difference, per-class")
    ax.legend()

    plt.suptitle("KDE Rank-Preservation Diagnostics (per-class AUROC / AP)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, "rank_preservation.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Summary table ────────────────────────────────────────
    summary = pd.DataFrame({
        "Metric": [
            "Number of (algo, task, class) evaluations",
            "Median |ΔAUROC|",
            "95th-pctl |ΔAUROC|",
            "Max |ΔAUROC|",
            "Pearson r (AUROC_orig, AUROC_kde)",
            "Median ROC curve area difference",
            "95th-pctl ROC curve area difference",
            "Median |ΔAP|",
            "95th-pctl |ΔAP|",
            "Max |ΔAP|",
            "Pearson r (AP_orig, AP_kde)",
            "Median PR curve area difference",
            "95th-pctl PR curve area difference",
        ],
        "Value": [
            f"{len(results_df)}",
            f"{results_df['auroc_abs_err'].median():.4f}",
            f"{np.percentile(results_df['auroc_abs_err'], 95):.4f}",
            f"{results_df['auroc_abs_err'].max():.4f}",
            f"{r_auroc:.4f}",
            f"{roc_diff.median():.4f}",
            f"{np.percentile(roc_diff, 95):.4f}",
            f"{results_df['ap_abs_err'].median():.4f}",
            f"{np.percentile(results_df['ap_abs_err'], 95):.4f}",
            f"{results_df['ap_abs_err'].max():.4f}",
            f"{r_ap:.4f}",
            f"{pr_diff.median():.4f}",
            f"{np.percentile(pr_diff, 95):.4f}",
        ],
    })
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