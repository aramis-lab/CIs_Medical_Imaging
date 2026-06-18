import numpy as np
from scipy.special import ndtr, ndtri          # ← (1) raw C ufuncs, no Python overhead
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from joblib import Parallel, delayed            # ← (5) multicore parallelism

from ...intervals_and_metrics import get_bounds
from ...utils import extract_df
from ...kernels import get_kernel
from ...kde import weighted_kde, sample_weighted_kde

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)    # ← (3) precomputed constant


# ──────────────────────────────────────────────
#  (2) Precompute KDE params ONCE, reuse everywhere
# ──────────────────────────────────────────────
def _precompute_kde(values, a, b):
    n = len(values)
    h = np.std(values, ddof=1) * n ** (-0.2)
    alpha = (a - values) / h
    beta  = (b - values) / h
    cdf_alpha = ndtr(alpha)                     # ← (1) ndtr instead of norm.cdf
    Z = ndtr(beta) - cdf_alpha
    return h, cdf_alpha, Z


def kde_renormalized(values, a, b, n_samples, eval_points=None):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    h, cdf_alpha, Z = _precompute_kde(values, a, b)

    # ---- Sampling via inverse-CDF of truncated Gaussians ----
    indices = np.random.randint(0, n, size=n_samples)
    u = np.random.uniform(0.0, 1.0, size=n_samples)
    samples = values[indices] + h * ndtri(      # ← (1) ndtri instead of norm.ppf
        cdf_alpha[indices] + u * Z[indices]
    )
    samples = np.clip(samples, a, b)

    if eval_points is not None:
        eval_points = np.asarray(eval_points, dtype=np.float64)
        # (4) Fully vectorized density — replaces the Python for-loop over n
        diff = (eval_points[:, None] - values[None, :]) / h        # (m, n)
        # (3) Manual Gaussian PDF avoids norm.pdf overhead
        kernels = _INV_SQRT_2PI * np.exp(-0.5 * diff * diff)
        kernels /= (h * Z[None, :])
        density = kernels.mean(axis=1)
        density[(eval_points < a) | (eval_points > b)] = 0.0
        return samples, density

    return samples


# ──────────────────────────────────────────────
#  Worker executed in parallel by joblib
# ──────────────────────────────────────────────
def _single_rep_truncated(values, h, cdf_alpha, Z, a, b,
                n_s, n_bootstrap, ci_alpha, seed):
    """One repetition: KDE sample → bootstrap → percentile CI."""
    rng = np.random.default_rng(seed)           # ← thread-safe, fast Generator
    n = len(values)

    # --- KDE sampling ---
    idx = rng.integers(0, n, size=n_s)
    u   = rng.uniform(size=n_s)
    samples = values[idx] + h * ndtri(cdf_alpha[idx] + u * Z[idx])

    # --- Bootstrap means  (6) memory-capped chunking ---
    # Prevents allocating multi-GB arrays for large n_s
    max_chunk = max(1, 500_000_000 // (2 * n_s * 8))
    boot_means = np.empty(n_bootstrap)
    for s in range(0, n_bootstrap, max_chunk):
        e  = min(s + max_chunk, n_bootstrap)
        bi = rng.integers(0, n_s, size=(e - s, n_s))
        boot_means[s:e] = samples[bi].mean(axis=1)

    lo = np.percentile(boot_means, 100 * ci_alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - ci_alpha / 2))
    return lo, hi

def sample_custom_kde(values, a, b, n_samples, eval_points=None, return_density=False):
    kernel = get_kernel("epanechnikov")

    # ── KDE grid setup (mirrors user's segmentation code) ──
    values_span = np.max(values) - np.min(values)
    min_val = np.min(values) - 0.1 * values_span if np.isinf(a) else a
    max_val = np.max(values) + 0.1 * values_span if np.isinf(b) else b

    if eval_points is None:
        x = np.linspace(min_val, max_val, 10000)
    else:
        x = eval_points

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
    samples = sample_weighted_kde(y, x, n_samples, a, b)
    if return_density:
        density = y / np.trapezoid(y, x)
        return samples, density
    return samples

def _single_rep_custom(values, a, b, n_s, n_bootstrap, ci_alpha, seed):
    rng = np.random.default_rng(seed)
    samples = sample_custom_kde(values, a, b, n_s)

    # --- Bootstrap means  (6) memory-capped chunking ---
    # Prevents allocating multi-GB arrays for large n_s
    max_chunk = max(1, 500_000_000 // (2 * n_s * 8))
    boot_means = np.empty(n_bootstrap)
    for s in range(0, n_bootstrap, max_chunk):
        e  = min(s + max_chunk, n_bootstrap)
        bi = rng.integers(0, n_s, size=(e - s, n_s))
        boot_means[s:e] = samples[bi].mean(axis=1)

    lo = np.percentile(boot_means, 100 * ci_alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - ci_alpha / 2))
    return lo, hi


def compute_bootstrap_percentile_intervals_varying_sample_sizes_truncated_kde(
        values, a, b, sample_sizes,
        n_bootstrap=1000, n_repetitions=1000, alpha=0.05,
        n_jobs=-1):                             # ← (5) n_jobs for parallelism
    values = np.asarray(values, dtype=np.float64)
    h, cdf_alpha, Z = _precompute_kde(values, a, b)   # ← (2) computed ONCE

    intervals = {}
    ss = np.random.SeedSequence()

    for n_s in tqdm(sample_sizes[::-1], desc="Computing intervals"):
        seeds = ss.spawn(n_repetitions)

        # (5) All repetitions for this sample size run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_rep_truncated)(
                values, h, cdf_alpha, Z, a, b,
                n_s, n_bootstrap, alpha, seed
            )
            for seed in seeds
        )

        lbs, ubs = zip(*results)
        intervals[n_s] = (np.mean(lbs), np.mean(ubs))

    return intervals

def compute_bootstrap_percentile_intervals_varying_sample_sizes_custom_kde(
        values, a, b, sample_sizes,
        n_bootstrap=1000, n_repetitions=1000, alpha=0.05,
        n_jobs=-1):                             # ← (5) n_jobs for parallelism
    values = np.asarray(values, dtype=np.float64)
    intervals = {}
    ss = np.random.SeedSequence()

    for n_s in tqdm(sample_sizes[::-1], desc="Computing intervals"):
        seeds = ss.spawn(n_repetitions)

        # (5) All repetitions for this sample size run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_rep_custom)(
                values, a, b, n_s, n_bootstrap, alpha, seed
            )
            for seed in seeds
        )

        lbs, ubs = zip(*results)
        intervals[n_s] = (np.mean(lbs), np.mean(ubs))

    return intervals

def plot_sample_KDE_dis_CIs(values, a, b, sample_sizes, save_path,
                             n_bootstrap=999, n_repetitions=1000,
                             alpha=0.05, n_jobs=-1):
    orig_mean = np.mean(values)
    x = np.linspace(a, b, 1000)
    samples_truncated, density_truncated = kde_renormalized(values, a, b, 10_000_000, eval_points=x)
    density_truncated /= np.trapezoid(density_truncated, x)
    truncated_kde_mean = np.mean(samples_truncated)

    samples_custom, density_custom = sample_custom_kde(values, a, b, 10_000_000, eval_points=x, return_density=True)
    density_custom /= np.trapezoid(density_custom, x)
    custom_kde_mean = np.mean(samples_custom)


    intervals_truncated = compute_bootstrap_percentile_intervals_varying_sample_sizes_truncated_kde(
        values, a, b, sample_sizes, n_bootstrap, n_repetitions, alpha, n_jobs
    )

    intervals_custom = compute_bootstrap_percentile_intervals_varying_sample_sizes_custom_kde(
        values, a, b, sample_sizes, n_bootstrap, n_repetitions, alpha, n_jobs
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(values, bins=50, density=True, alpha=0.5,
                 color='gray', label='KDE Histogram')
    axes[0].plot(x, density_truncated, color='blue', label='Truncated KDE Estimate')
    axes[0].plot(x, density_custom, color='red', label='Proposed KDE Estimate', linestyle='--')
    for n_s in sample_sizes:
        lo, hi = intervals_truncated[n_s]
        axes[1].plot([lo, hi], [n_s*1.1, n_s*1.1], color='deepskyblue', lw=3)
        lo, hi = intervals_custom[n_s]
        axes[1].plot([lo, hi], [n_s, n_s], color='orange', lw=3)
    axes[0].set_title('KDE Estimates');  axes[0].set_xlabel('Value');  axes[0].set_ylabel('Density')
    axes[0].legend(loc="upper left")
    axes[1].set_title('Bootstrap Percentile Confidence Intervals')
    axes[1].set_xlabel('KDE Estimate');  axes[1].set_ylabel('Sample Size')
    axes[1].axvline(orig_mean, color='black', linestyle='--', label=f'Original Mean: {100*orig_mean:.2f}%')
    axes[1].axvline(truncated_kde_mean,  color='blue',   linestyle='-',  label=f'Truncated KDE Mean: {100*truncated_kde_mean:.2f}%')
    axes[1].axvline(custom_kde_mean,  color='red',   linestyle='--', label=f'Proposed KDE Mean: {100*custom_kde_mean:.2f}%')
    # ── Add legend entries for the CI intervals ──
    axes[1].plot([], [], color='orange',      lw=3, label='Proposed KDE CI')
    axes[1].plot([], [], color='deepskyblue', lw=3, label='Truncated KDE CI')
    axes[1].set_yscale('log')
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility of the sampling
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

    task = "Task03_Liver_L1"
    algo = "isarasua"
    metric = "dsc"

    df = extract_df(df_path, metric, task)
    values = df[df['alg_name'] == algo]['value'].to_numpy()
    a, b = get_bounds(metric)
    sample_sizes = [10, 32, 100, 316, 1000, 3162, 10000, 31623, 100000]
    plot_sample_KDE_dis_CIs(values, a, b, sample_sizes, save_path = os.path.join(args.output_folder, f"clean_figs/supplementary/bias_renormalized_interval.pdf"),
                             n_bootstrap=999, n_repetitions=100, alpha=0.05)