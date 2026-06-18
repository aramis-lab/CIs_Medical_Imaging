import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

from ...kde import weighted_kde, sample_weighted_kde
from ...kernels import get_kernel
from ...intervals_and_metrics import get_bounds, is_continuous
from ...utils import extract_df


# ─────────────────────────────────────────────────────────────
#  Curve computation helpers
# ─────────────────────────────────────────────────────────────

N_GRID_POINTS = 100


def compute_ecdf(values, grid):
    """
    Compute the empirical CDF of `values` evaluated at `grid` points.
    Returns (grid, cdf) — analogous to compute_roc_points returning (fpr, tpr).
    """
    sorted_vals = np.sort(values)
    cdf = np.searchsorted(sorted_vals, grid, side='right') / len(sorted_vals)
    return grid, cdf


def _interpolate_on_grid(x_raw, y_raw, grid):
    """Sort by x_raw and interpolate y_raw onto a common grid."""
    idx = np.argsort(x_raw)
    return np.interp(grid, x_raw[idx], y_raw[idx])


# ─────────────────────────────────────────────────────────────
#  Per-instance: compute CDF values on shared normalized grid
# ─────────────────────────────────────────────────────────────

def compute_cdf_curve_points(orig_values, kde_samples):
    """
    For one instance, compute original and KDE ECDFs
    interpolated onto a shared normalized grid.

    The value range is determined from the union of both samples,
    then mapped to [0, 1] so instances with different value ranges
    can be aggregated.

    Returns
    -------
    cdf_orig, cdf_kde : (N_GRID_POINTS,) — CDF values on shared grid
    """
    # Determine range from both samples
    lo = min(orig_values.min(), kde_samples.min())
    hi = max(orig_values.max(), kde_samples.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-12

    # Dense raw grid in original value space
    raw_grid = np.linspace(lo, hi, 2000)

    # Compute raw ECDFs
    raw_grid_o, cdf_o = compute_ecdf(orig_values, raw_grid)
    raw_grid_k, cdf_k = compute_ecdf(kde_samples, raw_grid)

    # Normalized grid [0, 1] for cross-instance aggregation
    norm_grid = np.linspace(0, 1, N_GRID_POINTS)

    # Map raw grid to [0, 1] and interpolate onto shared grid
    raw_norm = (raw_grid - lo) / (hi - lo)
    cdf_orig = _interpolate_on_grid(raw_norm, cdf_o, norm_grid)
    cdf_kde  = _interpolate_on_grid(raw_norm, cdf_k, norm_grid)

    return cdf_orig, cdf_kde


# ─────────────────────────────────────────────────────────────
#  Aggregate plot: overlay + deviation (2 panels)
# ─────────────────────────────────────────────────────────────

def plot_aggregate_cdf_curves(cdf_orig_mat, cdf_kde_mat, save_path=None):
    """
    Two-panel figure summarizing CDF agreement across all instances.

    Left:  Mean CDF overlay (Original vs KDE) with ±1 std bands
    Right: Pointwise deviation (KDE − Original) with median + IQR

    Parameters
    ----------
    cdf_orig_mat, cdf_kde_mat : (N_instances, N_GRID_POINTS)
        CDF values on shared normalized grid.
    """
    n_inst = cdf_orig_mat.shape[0]
    norm_grid = np.linspace(0, 1, N_GRID_POINTS)

    fig = plt.figure(figsize=(12, 5))

    # ── Right: Pointwise deviation ──
    diff_mat = cdf_kde_mat - cdf_orig_mat

    med = np.nanmedian(diff_mat, axis=0)
    q1  = np.nanpercentile(diff_mat, 25, axis=0)
    q3  = np.nanpercentile(diff_mat, 75, axis=0)
    p5  = np.nanpercentile(diff_mat, 5, axis=0)
    p95 = np.nanpercentile(diff_mat, 95, axis=0)

    plt.fill_between(norm_grid, p5, p95, alpha=0.15, color='dodgerblue',
                    label='5th–95th percentile')
    plt.fill_between(norm_grid, q1, q3, alpha=0.3, color='dodgerblue',
                    label='IQR (Q1–Q3)')
    plt.plot(norm_grid, med, '-', color='dodgerblue', lw=2.5,
            label='Median')
    plt.axhline(0, color='black', ls='--', lw=1.5, alpha=0.6)

    plt.xlabel('Normalized metric value', fontsize=11)
    plt.ylabel('CDF(KDE) − CDF(Original)', fontsize=11)
    plt.title('CDF — Pointwise Deviation', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)

    fig.suptitle(
        f'KDE CDF Preservation ({n_inst} task × algorithm instances)',
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
#  Single-instance KDE + CDF computation
# ─────────────────────────────────────────────────────────────

def compute_instance(df_path, metric, task, algo):
    """
    Run KDE for one (task, algo) pair and return CDF curve points.

    Returns
    -------
    (cdf_orig, cdf_kde) each (N_GRID_POINTS,), or None on failure.
    """
    try:
        df = extract_df(df_path, metric, task)

        values = df[df["alg_name"] == algo]["value"].to_numpy()
        values = values[~np.isnan(values)]

        if len(values) < 50:
            print(f"  not enough values ({len(values)}), skipping")
            return None

        if not is_continuous(metric):
            samples = np.random.choice(values, size=100_000, replace=True)
        else:
            a, b = get_bounds(metric)
            kernel = get_kernel("epanechnikov")

            # ── KDE grid setup (mirrors user's segmentation code) ──
            values_span = np.max(values) - np.min(values)
            min_val = np.min(values) - 0.1 * values_span if np.isinf(a) else a
            max_val = np.max(values) + 0.1 * values_span if np.isinf(b) else b

            x = np.linspace(min_val, max_val, 10000)
            alphas = np.ones(len(values))
            dist_to_bounds = np.min([values-a, b-values], axis=0)

            # ── Initial KDE ──
            y = weighted_kde(values, x, dist_to_bounds, kernel, alphas)

            # ── Adaptive bandwidth ──
            indices = np.clip(np.searchsorted(x, values), 0, len(y) - 1)
            initial_estimates = np.maximum(y[indices], 1e-300)
            log_g = np.mean(np.log(initial_estimates))
            g = np.exp(log_g)
            alphas = (initial_estimates / g) ** (-1 / 2)
            y = weighted_kde(values, x, dist_to_bounds, kernel, alphas)

            # ── Sample from KDE ──
            samples = sample_weighted_kde(y, x, 100_000, a, b)

        return compute_cdf_curve_points(values, samples)

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
    df_name = "data_matrix_grandchallenge_all.csv"
    df_path = os.path.join(BASE_DIR, df_name)

    parser = argparse.ArgumentParser(
        description="KDE CDF preservation — aggregate analysis for segmentation"
    )
    parser.add_argument("--output_folder", type=str, default=BASE_DIR)
    args = parser.parse_args()

    df = pd.read_csv(df_path)
    tasks = df["subtask"].unique()
    algos = df["alg_name"].unique()
    metrics = df["score"].unique()

    # ── Collect CDF curve points across all (task, algo) instances ──
    all_cdf_orig = []
    all_cdf_kde  = []

    for task in tasks:
        for algo in algos:
            print(f"  {task} / {algo} …", end="")
            for metric in metrics:
                result = compute_instance(df_path, metric, task, str(algo))
                if result is not None:
                    cdf_o, cdf_k = result
                    all_cdf_orig.append(cdf_o)
                    all_cdf_kde.append(cdf_k)
                    print(" ✓")
            else:
                print()

    # ── Aggregate and plot ──
    if len(all_cdf_orig) == 0:
        print("\nNo valid instances found.")
    else:
        cdf_orig_mat = np.vstack(all_cdf_orig)  # (N_instances, N_GRID_POINTS)
        cdf_kde_mat  = np.vstack(all_cdf_kde)

        print(f"\n  Collected {cdf_orig_mat.shape[0]} valid instances.")

        plot_aggregate_cdf_curves(
            cdf_orig_mat, cdf_kde_mat,
            save_path=os.path.join(args.output_folder,
                                   "clean_figs/supplementary/cdf_preservation_segmentation.pdf")
        )