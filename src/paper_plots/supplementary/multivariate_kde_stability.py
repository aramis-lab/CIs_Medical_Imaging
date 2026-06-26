"""
rank_preservation.py

Verify that KDE-sampled logits preserve the score-ordering properties
(class-conditional CDFs, concordance probability) that determine
AUROC and AP, for each (algo, task) pair.
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score

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
#  Rank-preservation evaluation for one (algo, task)
# ─────────────────────────────────────────────────────────────


def evaluate_rank_preservation_binary(
    scores_orig, labels_orig,
    scores_kde, labels_kde
):
    """
    For one classification (task, algo):
    check whether the ordering structure that determines AUROC/AP
    is preserved by KDE sampling.

    Parameters
    ----------
    scores_orig  : (N,C) original scalar scores (e.g., logit for class c)
    labels_orig  : (N,) labels
    scores_kde   : (M,C) KDE-sampled scores
    labels_kde   : (M,) corresponding labels

    Returns
    -------
    dict of rank-preservation metrics, or None if degenerate.
    """
    pos_orig = scores_orig[labels_orig == 1]
    neg_orig = scores_orig[labels_orig == 0]
    pos_kde = scores_kde[labels_kde == 1]
    neg_kde = scores_kde[labels_kde == 0]

    if len(pos_orig) < 5 or len(neg_orig) < 5:
        return None
    if len(pos_kde) < 5 or len(neg_kde) < 5:
        return None

    labels_orig_bin = label_binarize_vectorized(labels_orig, scores_orig.shape[1])
    labels_kde_bin = label_binarize_vectorized(labels_kde, scores_kde.shape[1])

    # ── 1. AUROC ────────────────────────────────────────────
    auroc_orig = roc_auc_score(labels_orig_bin, scores_orig)
    auroc_kde = roc_auc_score(labels_kde_bin, scores_kde)

    # ── 2. Average Precision ────────────────────────────────
    ap_orig = average_precision_score(labels_orig_bin, scores_orig)
    ap_kde = average_precision_score(labels_kde_bin, scores_kde)

    # ── 3. Class prior ──────────────────────────────────────
    prevalence_orig = labels_orig_bin.mean()
    prevalence_kde = labels_kde_bin.mean()

    # ── 4. ROC shape: TPR at FPR quantiles ──────────────────
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    neg_thresholds_orig = np.quantile(neg_orig, quantiles)
    neg_thresholds_kde = np.quantile(neg_kde, quantiles)

    tpr_at_q_orig = np.array([
        (pos_orig > t).mean() for t in neg_thresholds_orig
    ])
    tpr_at_q_kde = np.array([
        (pos_kde > t).mean() for t in neg_thresholds_kde
    ])
    tpr_mae = np.mean(np.abs(tpr_at_q_orig - tpr_at_q_kde))
    tpr_max_err = np.max(np.abs(tpr_at_q_orig - tpr_at_q_kde))

    # ── 5. PR shape: Precision at recall quantiles ──────────
    #    To achieve recall = r, threshold = (1-r)-quantile
    #    of positive scores.  Precision at that threshold =
    #    #pos_above / (#pos_above + #neg_above).
    recall_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    pos_thresholds_orig = np.quantile(pos_orig, 1 - recall_levels)
    pos_thresholds_kde = np.quantile(pos_kde, 1 - recall_levels)

    def _precision_at_threshold(pos, neg, t):
        n_pos = (pos > t).sum()
        n_neg = (neg > t).sum()
        total = n_pos + n_neg
        return n_pos / total if total > 0 else 1.0

    prec_at_r_orig = np.array([
        _precision_at_threshold(pos_orig, neg_orig, t)
        for t in pos_thresholds_orig
    ])
    prec_at_r_kde = np.array([
        _precision_at_threshold(pos_kde, neg_kde, t)
        for t in pos_thresholds_kde
    ])
    prec_mae = np.mean(np.abs(prec_at_r_orig - prec_at_r_kde))
    prec_max_err = np.max(np.abs(prec_at_r_orig - prec_at_r_kde))

    return {
        "auroc_orig": auroc_orig,
        "auroc_kde": auroc_kde,
        "auroc_abs_err": abs(auroc_orig - auroc_kde),
        "ap_orig": ap_orig,
        "ap_kde": ap_kde,
        "ap_abs_err": abs(ap_orig - ap_kde),
        "prevalence_orig": prevalence_orig,
        "prevalence_kde": prevalence_kde,
        "prevalence_abs_err": abs(prevalence_orig - prevalence_kde),
        "tpr_at_quantile_mae": tpr_mae,
        "tpr_at_quantile_max_err": tpr_max_err,
        "prec_at_recall_mae": prec_mae,
        "prec_at_recall_max_err": prec_max_err,
    }


# ─────────────────────────────────────────────────────────────
#  Dashboard and summary
# ─────────────────────────────────────────────────────────────

def aggregate_and_plot(results_df, output_folder="."):
    """
    2×3 diagnostic dashboard.

    Top row (AUROC):
        (a) AUROC_orig vs AUROC_kde scatter
        (b) |ΔAUROC| histogram
        (c) ROC shape preservation (TPR-at-FPR-quantile MAE)

    Bottom row (AP):
        (d) AP_orig vs AP_kde scatter
        (e) |ΔAP| histogram
        (f) PR shape preservation (Precision-at-Recall-quantile MAE)
    """
    os.makedirs(output_folder, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ═══════════════════  TOP ROW: AUROC  ═════════════════════

    # (a) AUROC scatter ──────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(results_df["auroc_orig"], results_df["auroc_kde"],
               s=8, alpha=0.3, rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5)
    r_auroc, _ = stats.pearsonr(results_df["auroc_orig"],
                                results_df["auroc_kde"])
    ax.set_xlabel("AUROC (original)")
    ax.set_ylabel("AUROC (KDE)")
    ax.set_title(f"(a) AUROC: orig vs KDE  (r = {r_auroc:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # (b) |ΔAUROC| histogram ─────────────────────────────────
    ax = axes[0, 1]
    auroc_errs = results_df["auroc_abs_err"]
    ax.hist(auroc_errs, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(auroc_errs.median(), color="red", ls="--",
               label=f"Median = {auroc_errs.median():.4f}")
    ax.axvline(np.percentile(auroc_errs, 95), color="orange", ls=":",
               label=f"95th pctl = {np.percentile(auroc_errs, 95):.4f}")
    ax.set_xlabel("|AUROC$_{\\rm orig}$ − AUROC$_{\\rm KDE}$|")
    ax.set_title("(b) AUROC absolute error")
    ax.legend()

    # (c) ROC shape preservation ──────────────────────────────
    ax = axes[0, 2]
    tpr_mae = results_df["tpr_at_quantile_mae"]
    ax.hist(tpr_mae, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(tpr_mae.median(), color="red", ls="--",
               label=f"Median = {tpr_mae.median():.4f}")
    ax.set_xlabel("MAE of TPR at FPR quantiles")
    ax.set_title("(c) ROC shape preservation")
    ax.legend()

    # ═══════════════════  BOTTOM ROW: AP  ═════════════════════

    # (d) AP scatter ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(results_df["ap_orig"], results_df["ap_kde"],
               s=8, alpha=0.3, rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5)
    r_ap, _ = stats.pearsonr(results_df["ap_orig"],
                             results_df["ap_kde"])
    ax.set_xlabel("AP (original)")
    ax.set_ylabel("AP (KDE)")
    ax.set_title(f"(d) AP: orig vs KDE  (r = {r_ap:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # (e) |ΔAP| histogram ────────────────────────────────────
    ax = axes[1, 1]
    ap_errs = results_df["ap_abs_err"]
    ax.hist(ap_errs, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(ap_errs.median(), color="red", ls="--",
               label=f"Median = {ap_errs.median():.4f}")
    ax.axvline(np.percentile(ap_errs, 95), color="orange", ls=":",
               label=f"95th pctl = {np.percentile(ap_errs, 95):.4f}")
    ax.set_xlabel("|AP$_{\\rm orig}$ − AP$_{\\rm KDE}$|")
    ax.set_title("(e) AP absolute error")
    ax.legend()

    # (f) PR shape preservation ───────────────────────────────
    ax = axes[1, 2]
    prec_mae = results_df["prec_at_recall_mae"]
    ax.hist(prec_mae, bins=60, density=True, alpha=0.7, edgecolor="k")
    ax.axvline(prec_mae.median(), color="red", ls="--",
               label=f"Median = {prec_mae.median():.4f}")
    ax.set_xlabel("MAE of Precision at Recall quantiles")
    ax.set_title("(f) PR shape preservation")
    ax.legend()

    plt.suptitle("KDE Rank-Preservation Diagnostics (AUROC / AP)",
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
            "Median TPR-at-quantile MAE (ROC shape)",
            "Median |ΔAP|",
            "95th-pctl |ΔAP|",
            "Max |ΔAP|",
            "Pearson r (AP_orig, AP_kde)",
            "Median Prec-at-recall MAE (PR shape)",
        ],
        "Value": [
            f"{len(results_df)}",
            f"{auroc_errs.median():.4f}",
            f"{np.percentile(auroc_errs, 95):.4f}",
            f"{auroc_errs.max():.4f}",
            f"{r_auroc:.4f}",
            f"{tpr_mae.median():.4f}",
            f"{ap_errs.median():.4f}",
            f"{np.percentile(ap_errs, 95):.4f}",
            f"{ap_errs.max():.4f}",
            f"{r_ap:.4f}",
            f"{prec_mae.median():.4f}",
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
        description="KDE Rank-Preservation Diagnostics (AUROC/AP)"
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

        median_orig = np.median(orig_values)

        # ── Evaluate ──
        res = evaluate_rank_preservation_binary(
            orig_values, orig_labels,
            kde_scores, kde_labels
        )
        if res is None:
            continue

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