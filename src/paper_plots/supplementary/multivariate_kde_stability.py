import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

from ...kde import sample_weighted_kde_multivariate
from ...kernels import get_kernel
from ...intervals_and_metrics import softmax, get_metric, label_binarize_vectorized
from ...utils import extract_df


# ─────────────────────────────────────────────────────────────
#  Curve computation helpers
# ─────────────────────────────────────────────────────────────

N_GRID_POINTS = 100


def compute_roc_points(pos_scores, neg_scores, n_thresholds=2000):
    """Memory-safe ROC via searchsorted. Returns (fpr, tpr)."""
    lo = min(pos_scores.min(), neg_scores.min()) - 1e-10
    hi = max(pos_scores.max(), neg_scores.max()) + 1e-10
    thresholds = np.linspace(hi, lo, n_thresholds)
    sorted_pos = np.sort(pos_scores)
    sorted_neg = np.sort(neg_scores)
    n_pos, n_neg = len(sorted_pos), len(sorted_neg)
    tpr = (n_pos - np.searchsorted(sorted_pos, thresholds, side='left')) / n_pos
    fpr = (n_neg - np.searchsorted(sorted_neg, thresholds, side='left')) / n_neg
    return fpr, tpr


def compute_precision_at_fraction(pos_scores, neg_scores, n_thresholds=2000):
    """
    Precision curve parameterized by fraction of data examined.
    Returns (fractions, precisions), analogous to (fpr, tpr) for ROC.
    """
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    n = len(all_scores)
    sorted_labels = labels[np.argsort(-all_scores)]
    cumsum = np.cumsum(sorted_labels)

    # Evaluate at n_thresholds evenly spaced k values
    k_values = np.unique(np.linspace(1, n, n_thresholds).astype(int))
    fractions = k_values / n
    precisions = cumsum[k_values - 1] / k_values

    return fractions, precisions


def _split_pos_neg_micro(scores, labels_bin):
    """Flatten scores and labels for micro averaging."""
    flat_s = scores.ravel()
    flat_l = labels_bin.ravel().astype(int)
    return flat_s[flat_l == 1], flat_s[flat_l == 0]


def _interpolate_on_grid(x_raw, y_raw, grid):
    """Sort by x_raw and interpolate y_raw onto a common grid."""
    idx = np.argsort(x_raw)
    return np.interp(grid, x_raw[idx], y_raw[idx])


# ─────────────────────────────────────────────────────────────
#  Per-instance: compute curve values on shared grids (micro)
# ─────────────────────────────────────────────────────────────

def compute_micro_curve_points(scores, labels_bin, kde_scores, kde_labels_bin):
    """
    For one instance, compute original and KDE curve values
    interpolated onto shared grids.

    ROC:  shared FPR grid  → TPR values
    P@K:  shared fraction grid → Precision values

    Returns
    -------
    roc_orig, roc_kde : (N_GRID_POINTS,) — TPR on shared FPR grid
    pk_orig,  pk_kde  : (N_GRID_POINTS,) — Precision on shared fraction grid
    """
    o_pos, o_neg = _split_pos_neg_micro(scores, labels_bin)
    k_pos, k_neg = _split_pos_neg_micro(kde_scores, kde_labels_bin)

    # ── ROC: raw curves → interpolate onto shared FPR grid ──
    fpr_grid = np.linspace(0, 1, N_GRID_POINTS)

    fpr_o, tpr_o = compute_roc_points(o_pos, o_neg)
    roc_orig = _interpolate_on_grid(fpr_o, tpr_o, fpr_grid)

    fpr_k, tpr_k = compute_roc_points(k_pos, k_neg)
    roc_kde = _interpolate_on_grid(fpr_k, tpr_k, fpr_grid)

    # ── P@K: raw curves → interpolate onto shared fraction grid ──
    frac_grid = np.linspace(0.005, 1.0, N_GRID_POINTS)

    frac_o, prec_o = compute_precision_at_fraction(o_pos, o_neg)
    pk_orig = _interpolate_on_grid(frac_o, prec_o, frac_grid)

    frac_k, prec_k = compute_precision_at_fraction(k_pos, k_neg)
    pk_kde = _interpolate_on_grid(frac_k, prec_k, frac_grid)

    return roc_orig, roc_kde, pk_orig, pk_kde


# ─────────────────────────────────────────────────────────────
#  Aggregate plot: overlay + deviation (4 panels)
# ─────────────────────────────────────────────────────────────

def plot_aggregate_curves(roc_orig_mat, roc_kde_mat,
                          pk_orig_mat, pk_kde_mat,
                          save_path=None):
    """
    Four-panel figure summarizing curve agreement across all instances.

    Top row:    Mean curve overlay (Original vs KDE) with ±1 std bands
    Bottom row: Pointwise deviation (KDE − Original) with median + IQR

    Parameters
    ----------
    roc_orig_mat, roc_kde_mat : (N_instances, N_GRID_POINTS)
        TPR values on shared FPR grid.
    pk_orig_mat, pk_kde_mat : (N_instances, N_GRID_POINTS)
        Precision values on shared fraction grid.
    """
    n_inst = roc_orig_mat.shape[0]

    fpr_grid  = np.linspace(0, 1, N_GRID_POINTS)
    frac_grid = np.linspace(0.005, 1.0, N_GRID_POINTS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ═════════════════════════════════════════════════════════
    #  Top row: Mean curve overlay
    # ═════════════════════════════════════════════════════════

    def _overlay(ax, grid, orig_mat, kde_mat, xlabel, ylabel, title):
        mean_orig = np.nanmean(orig_mat, axis=0)
        std_orig  = np.nanstd(orig_mat, axis=0)
        mean_kde  = np.nanmean(kde_mat, axis=0)
        std_kde   = np.nanstd(kde_mat, axis=0)

        ax.fill_between(grid, mean_orig - std_orig, mean_orig + std_orig,
                        alpha=0.2, color='blue')
        ax.plot(grid, mean_orig, 'b-', lw=2, label='Original (mean ± std)')

        ax.fill_between(grid, mean_kde - std_kde, mean_kde + std_kde,
                        alpha=0.2, color='red')
        ax.plot(grid, mean_kde, 'r--', lw=2, label='KDE (mean ± std)')

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    _overlay(axes[0, 0], fpr_grid, roc_orig_mat, roc_kde_mat,
             'FPR', 'TPR', 'ROC — Curve Overlay')
    axes[0, 0].plot([0, 1], [0, 1], 'k:', alpha=0.3)

    _overlay(axes[0, 1], frac_grid, pk_orig_mat, pk_kde_mat,
             'Fraction of data examined', 'Precision',
             'Precision@K — Curve Overlay')

    # ═════════════════════════════════════════════════════════
    #  Bottom row: Pointwise deviation (KDE − Original)
    # ═════════════════════════════════════════════════════════

    def _deviation(ax, grid, orig_mat, kde_mat, xlabel, title):
        diff_mat = kde_mat - orig_mat  # (N_instances, N_GRID_POINTS)

        med = np.nanmedian(diff_mat, axis=0)
        q1  = np.nanpercentile(diff_mat, 25, axis=0)
        q3  = np.nanpercentile(diff_mat, 75, axis=0)
        p5  = np.nanpercentile(diff_mat, 5, axis=0)
        p95 = np.nanpercentile(diff_mat, 95, axis=0)

        ax.fill_between(grid, p5, p95, alpha=0.15, color='dodgerblue',
                        label='5th–95th percentile')
        ax.fill_between(grid, q1, q3, alpha=0.3, color='dodgerblue',
                        label='IQR (Q1–Q3)')
        ax.plot(grid, med, '-', color='dodgerblue', lw=2.5,
                label='Median')
        ax.axhline(0, color='black', ls='--', lw=1.5, alpha=0.6)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('KDE − Original', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    _deviation(axes[1, 0], fpr_grid, roc_orig_mat, roc_kde_mat,
               'FPR', 'ROC — Pointwise Deviation')

    _deviation(axes[1, 1], frac_grid, pk_orig_mat, pk_kde_mat,
               'Fraction of data examined',
               'Precision@K — Pointwise Deviation')

    fig.suptitle(
        f'KDE Score Ordering Preservation ({n_inst} task × algorithm instances)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Single-instance KDE + curve computation
# ─────────────────────────────────────────────────────────────

def compute_instance(df_path, task, algo):
    """
    Run KDE for one (task, algo) pair and return micro curve points.

    Returns
    -------
    tuple of 4 arrays (roc_orig, roc_kde, pk_orig, pk_kde), each (N_GRID_POINTS,)
    or None on failure.
    """
    try:
        df = extract_df(df_path, "auc", task)

        logits_str = df[df["alg_name"].astype(str) == algo]["logits"]
        values = [list(eval(v, {"nan": np.nan})) for v in logits_str]
        if len(values) == 0:
            print(f"  no data, skipping")
            return None

        lengths = np.array([len(v) for v in values])
        good_length = round(np.mean(lengths))
        indices = np.where(lengths == good_length)
        values = np.array([v for v in values if len(v) == good_length])
        labels = df[df["alg_name"].astype(str) == algo]["target"].to_numpy()[indices]

        kernel = get_kernel("epanechnikov")
        alphas = np.ones(len(values))

        # Adaptive bandwidth
        initial_estimates = kernel(values, values, alphas)
        initial_estimates = np.mean(initial_estimates, axis=1)
        log_g = np.mean(np.log(initial_estimates))
        g = np.exp(log_g)
        alphas = (initial_estimates / g) ** (-1 / 2)

        kde_values, kde_labels = sample_weighted_kde_multivariate(
            values, labels, "epanechnikov", 100_000, alphas
        )
        kde_scores = softmax(kde_values)
        n_classes = kde_scores.shape[-1]
        kde_labels_bin = label_binarize_vectorized(kde_labels, n_classes)
        labels_bin = label_binarize_vectorized(labels, n_classes)
        scores = softmax(values)

        return compute_micro_curve_points(
            scores, labels_bin, kde_scores, kde_labels_bin
        )

    except Exception as e:
        print(f"  error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    df_name = "data_matrix_classification.csv"
    df_path = os.path.join(BASE_DIR, df_name)

    parser = argparse.ArgumentParser(
        description="KDE score ordering preservation — aggregate analysis"
    )
    parser.add_argument("--output_folder", type=str, default=BASE_DIR)
    args = parser.parse_args()

    df = pd.read_csv(df_path)
    tasks = df["subtask"].unique()
    algos = df["alg_name"].unique()

    # ── Collect curve points across all (task, algo) instances ──
    all_roc_orig, all_roc_kde = [], []
    all_pk_orig,  all_pk_kde  = [], []

    for task in tasks:
        for algo in algos:
            print(f"  {task} / {algo} …", end="")
            result = compute_instance(df_path, task, str(algo))
            if result is not None:
                roc_o, roc_k, pk_o, pk_k = result
                all_roc_orig.append(roc_o)
                all_roc_kde.append(roc_k)
                all_pk_orig.append(pk_o)
                all_pk_kde.append(pk_k)
                print(" ✓")
            else:
                print()

    # ── Aggregate and plot ──
    if len(all_roc_orig) == 0:
        print("\nNo valid instances found.")
    else:
        roc_orig_mat = np.vstack(all_roc_orig)  # (N_instances, N_GRID_POINTS)
        roc_kde_mat  = np.vstack(all_roc_kde)
        pk_orig_mat  = np.vstack(all_pk_orig)
        pk_kde_mat   = np.vstack(all_pk_kde)

        print(f"\n  Collected {roc_orig_mat.shape[0]} valid instances.")

        plot_aggregate_curves(
            roc_orig_mat, roc_kde_mat,
            pk_orig_mat,  pk_kde_mat,
            save_path=os.path.join(args.output_folder,
                                   "score_ordering_preservation.pdf")
        )