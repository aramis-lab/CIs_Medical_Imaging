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

def precision_at_fraction_curve(pos_scores, neg_scores, fractions):
    """Precision evaluated at specified fractions of the ranked list."""
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    n = len(all_scores)
    sorted_labels = labels[np.argsort(-all_scores)]
    cumsum = np.cumsum(sorted_labels)
    k_values = np.unique(np.clip((fractions * n).astype(int), 1, n))
    precisions = cumsum[k_values - 1] / k_values
    actual_fractions = k_values / n
    return actual_fractions, precisions


def compute_roc_points(pos_scores, neg_scores, n_thresholds=2000):
    """Memory-safe ROC via searchsorted."""
    lo = min(pos_scores.min(), neg_scores.min()) - 1e-10
    hi = max(pos_scores.max(), neg_scores.max()) + 1e-10
    thresholds = np.linspace(hi, lo, n_thresholds)
    sorted_pos = np.sort(pos_scores)
    sorted_neg = np.sort(neg_scores)
    n_pos, n_neg = len(sorted_pos), len(sorted_neg)
    tpr = (n_pos - np.searchsorted(sorted_pos, thresholds, side='left')) / n_pos
    fpr = (n_neg - np.searchsorted(sorted_neg, thresholds, side='left')) / n_neg
    return fpr, tpr


def _split_pos_neg_micro(scores, labels_bin):
    """Flatten scores and labels for micro averaging."""
    flat_s = scores.ravel()
    flat_l = labels_bin.ravel().astype(int)
    return flat_s[flat_l == 1], flat_s[flat_l == 0]


# ─────────────────────────────────────────────────────────────
#  Per-instance: compute 100 QQ points (micro only)
# ─────────────────────────────────────────────────────────────

N_QQ_POINTS = 100


def compute_micro_qq_points(scores, labels_bin, kde_scores, kde_labels_bin):
    """
    At 100 evaluation grid points, compute (original, KDE) value pairs
    for ROC (TPR) and Precision@K, using micro averaging.

    For ROC: 100 common FPR grid points → 100 (TPR_orig, TPR_kde) pairs.
    For P@K: 100 common fraction grid points → 100 (P_orig, P_kde) pairs.

    Returns
    -------
    tpr_orig, tpr_kde : (100,) — ROC QQ points
    pk_orig, pk_kde   : (100,) — Precision@K QQ points
    """
    o_pos, o_neg = _split_pos_neg_micro(scores, labels_bin)
    k_pos, k_neg = _split_pos_neg_micro(kde_scores, kde_labels_bin)

    # ── ROC: evaluate at 100 common FPR grid points ──
    fpr_grid = np.linspace(0, 1, N_QQ_POINTS)

    fpr_o, tpr_o = compute_roc_points(o_pos, o_neg)
    idx = np.argsort(fpr_o)
    tpr_orig = np.interp(fpr_grid, fpr_o[idx], tpr_o[idx])

    fpr_k, tpr_k = compute_roc_points(k_pos, k_neg)
    idx = np.argsort(fpr_k)
    tpr_kde = np.interp(fpr_grid, fpr_k[idx], tpr_k[idx])

    # ── Precision@K: evaluate at 100 common fraction grid points ──
    frac_grid = np.linspace(0.005, 1.0, N_QQ_POINTS)

    f_o, p_o = precision_at_fraction_curve(o_pos, o_neg, frac_grid)
    pk_orig = np.interp(frac_grid, f_o, p_o)

    f_k, p_k = precision_at_fraction_curve(k_pos, k_neg, frac_grid)
    pk_kde = np.interp(frac_grid, f_k, p_k)

    return tpr_orig, tpr_kde, pk_orig, pk_kde


# ─────────────────────────────────────────────────────────────
#  Aggregate QQ plots (2 panels: ROC + Precision@K)
# ─────────────────────────────────────────────────────────────

def plot_aggregate_qq(tpr_orig_mat, tpr_kde_mat, pk_orig_mat, pk_kde_mat,
                      save_path=None):
    """
    Two-panel QQ plot summarizing score ordering preservation
    across all (task, algorithm) instances.

    At each of the 100 shared grid points, we have one (orig, kde) pair
    per instance. The QQ plot shows orig on x-axis, kde on y-axis.
    Perfect preservation → all points on y = x diagonal.

    Displays:
      - Faint per-instance curves (sorted by orig for clean rendering)
      - Median curve across instances
      - IQR (Q1–Q3) band across instances

    Parameters
    ----------
    tpr_orig_mat, tpr_kde_mat : (N_instances, 100) — ROC TPR values
    pk_orig_mat, pk_kde_mat   : (N_instances, 100) — Precision@K values
    """
    n_instances = tpr_orig_mat.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    def _qq(ax, orig_mat, kde_mat, title, axis_label):
        n_inst, n_pts = orig_mat.shape

        # ── Per-instance faint curves ──
        alpha = max(0.03, min(0.3, 5.0 / n_inst))
        for i in range(n_inst):
            order = np.argsort(orig_mat[i])
            ax.plot(orig_mat[i, order], kde_mat[i, order],
                    color='grey', alpha=alpha, lw=0.5, rasterized=True)

        # ── Summary: at each grid point, percentiles across instances ──
        med_orig = np.nanmedian(orig_mat, axis=0)
        med_kde  = np.nanmedian(kde_mat, axis=0)
        q1_kde   = np.nanpercentile(kde_mat, 25, axis=0)
        q3_kde   = np.nanpercentile(kde_mat, 75, axis=0)

        # Sort by median-orig so fill_between renders correctly
        order = np.argsort(med_orig)
        x     = med_orig[order]
        y_med = med_kde[order]
        y_q1  = q1_kde[order]
        y_q3  = q3_kde[order]

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6,
                label='Perfect preservation')
        ax.fill_between(x, y_q1, y_q3, alpha=0.3, color='dodgerblue',
                        label='IQR (Q1–Q3)')
        ax.plot(x, y_med, '-', color='dodgerblue', lw=2.5, label='Median')

        ax.set_xlabel(f'{axis_label} (Original)', fontsize=11)
        ax.set_ylabel(f'{axis_label} (KDE)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.legend(fontsize=9, loc='lower right')

    _qq(axes[0], tpr_orig_mat, tpr_kde_mat,
        'ROC — QQ Plot', 'TPR')

    _qq(axes[1], pk_orig_mat, pk_kde_mat,
        'Precision@K — QQ Plot', 'Precision')

    fig.suptitle(
        f'KDE Score Ordering Preservation ({n_instances} task × algorithm instances)',
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
#  Single-instance KDE + QQ computation
# ─────────────────────────────────────────────────────────────

def compute_instance(df_path, task, algo):
    """
    Run KDE for one (task, algo) pair and return micro QQ points.

    Returns
    -------
    tuple of 4 arrays (tpr_orig, tpr_kde, pk_orig, pk_kde), each (100,)
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

        return compute_micro_qq_points(scores, labels_bin, kde_scores, kde_labels_bin)

    except Exception as e:
        print(f" error: {e}")
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
        description="KDE score ordering preservation — aggregate QQ analysis"
    )
    parser.add_argument("--output_folder", type=str, default=BASE_DIR)
    args = parser.parse_args()

    df = pd.read_csv(df_path)
    tasks = df["subtask"].unique()
    algos = df["alg_name"].unique()

    # ── Collect QQ points across all (task, algo) instances ──
    all_tpr_orig, all_tpr_kde = [], []
    all_pk_orig,  all_pk_kde  = [], []

    for task in tasks:
        for algo in algos:
            print(f"  {task} / {algo} …", end="")
            result = compute_instance(df_path, task, str(algo))
            if result is not None:
                tpr_o, tpr_k, pk_o, pk_k = result
                all_tpr_orig.append(tpr_o)
                all_tpr_kde.append(tpr_k)
                all_pk_orig.append(pk_o)
                all_pk_kde.append(pk_k)
                print(" ✓")
            else:
                print()

    # ── Aggregate and plot ──
    if len(all_tpr_orig) == 0:
        print("\nNo valid instances found. Cannot produce QQ plots.")
    else:
        tpr_orig_mat = np.vstack(all_tpr_orig)   # (N_instances, 100)
        tpr_kde_mat  = np.vstack(all_tpr_kde)
        pk_orig_mat  = np.vstack(all_pk_orig)
        pk_kde_mat   = np.vstack(all_pk_kde)

        print(f"\n  Collected {tpr_orig_mat.shape[0]} valid instances.")

        plot_aggregate_qq(
            tpr_orig_mat, tpr_kde_mat,
            pk_orig_mat,  pk_kde_mat,
            save_path=os.path.join(args.output_folder, "score_ordering_qq.pdf")
        )