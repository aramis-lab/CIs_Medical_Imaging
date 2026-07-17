import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from ...kde import weighted_kde, sample_weighted_kde
from ...kernels import get_kernel
from ...intervals_and_metrics import get_bounds, is_continuous
from itertools import product
from tqdm import tqdm
from scipy.stats import energy_distance

# ─────────────────────────────────────────────────────────────
#  Single-instance KDE + CDF computation
# ─────────────────────────────────────────────────────────────

def compute_instance(df_, metric, task, algo):
    """
    Run KDE for one (task, algo) pair and return CDF curve points.

    Returns
    -------
    (cdf_orig, cdf_kde) each (N_GRID_POINTS,), or None on failure.
    """
    try:
        df = df_[(df_["subtask"] == task) & (df_["score"] == metric)]

        values = df[df["alg_name"] == algo]["value"].to_numpy()
        values = values[~np.isnan(values)]

        if len(values) < 50:
            return None

        if not is_continuous(metric):
            samples = np.random.choice(values, size=1000000, replace=True)
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
            samples = sample_weighted_kde(y, x, 1000000, a, b)

        return samples

    except Exception as e:
        print(f"  error: {e}")
        return None

def evaluate_kde_2sample(original, kde_samples):
    n = len(original)
    e_dist = energy_distance(original, kde_samples)
    mean_err = abs(np.mean(kde_samples) - np.mean(original)) / (np.std(original) + 1e-10)
    std_ratio = np.std(kde_samples) / (np.std(original) + 1e-10)
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q_orig = np.quantile(original, probs)
    q_kde = np.quantile(kde_samples, probs)
    max_q_err = np.max(np.abs(q_kde - q_orig) / (np.abs(q_orig) + 1e-10))
    median_q_err = np.median(np.abs(q_kde - q_orig) / (np.abs(q_orig) + 1e-10))
    return {
        'n': n,
        'energy_dist': e_dist, 'mean_err': mean_err, 'std_ratio': std_ratio,
        'max_q_err': max_q_err, 'median_q_err': median_q_err,
    }

def plot(df, all_originals, all_kde_samples, output_folder="."):

    # ── Dashboard ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    axes[0].hist(df['energy_dist'], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[0].axvline(df['energy_dist'].median(), color='red', ls='--',
                        label=f"Median = {df['energy_dist'].median():.4f}")
    axes[0].set_xlabel('Energy Distance'); axes[0].set_title('(a) Energy Distance')
    axes[0].legend()

    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    all_q_orig, all_q_kde = [], []
    for orig, kde_s in zip(all_originals, all_kde_samples):
        if len(orig) < 10:
            continue
        mu, sigma = np.mean(orig), np.std(orig) + 1e-10
        all_q_orig.append((np.quantile(orig, probs) - mu) / sigma)
        all_q_kde.append((np.quantile(kde_s, probs) - mu) / sigma)
    all_q_orig = np.concatenate(all_q_orig)
    all_q_kde = np.concatenate(all_q_kde)
    axes[1].plot([np.nanmin(all_q_orig), np.nanmax(all_q_orig)], [np.nanmin(all_q_orig), np.nanmax(all_q_orig)], color='red', ls='--', lw=2)
    axes[1].scatter(all_q_orig, all_q_kde, s=1, alpha=0.3)
    axes[1].set_xlabel('Standardised Empirical Quantiles')
    axes[1].set_ylabel('Standardised KDE Quantiles')
    axes[1].set_title('(b) Pooled Q-Q (all distributions)')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/kde_validation_2sample.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Summary table ──
    summary = pd.DataFrame({
        'Metric': [
            'Energy distance (median)',
            '|d_mu|/sigma (median)', 'sigma_KDE/sigma_data (median)',
            'Max quantile rel. error (median)',
        ],
        'Value': [
            f"{df['energy_dist'].median():.4f}",
            f"{df['mean_err'].median():.4f}",
            f"{df['std_ratio'].median():.4f}",
            f"{df['max_q_err'].median():.4f}",
        ]
    })

    return summary

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    df_name = "data_matrix_grandchallenge_all.csv"
    df_path = os.path.join(BASE_DIR, df_name)

    parser = argparse.ArgumentParser(description="KDE Representativeness Analysis")
    parser.add_argument("--output_folder", type=str, default=".", help="Folder to save output plots and summary")
    args = parser.parse_args()

    df = pd.read_csv(df_path)
    tasks = df["subtask"].unique()
    algos = df["alg_name"].unique()
    metrics = df["score"].unique()

    # ── Collect CDF curve points across all (task, algo) instances ──
    all_cdf_orig = []
    all_cdf_kde  = []

    means = []
    vars = []
    skews = []
    kurts = []

    results = []
    all_originals = []
    all_kde_samples = []
    for i, (task, algo, metric) in enumerate(tqdm(product(tasks, algos, metrics), total=len(tasks)*len(algos)*len(metrics))):
        kde_samples = compute_instance(df, metric, task, str(algo))
        original_values = df[(df["subtask"] == task) & (df["alg_name"] == algo) & (df["score"] == metric)]["value"].to_numpy()
        if original_values.size == 0 or kde_samples is None:
            continue
        res = evaluate_kde_2sample(original_values, kde_samples)
        res['dist_id'] = i
        results.append(res)
        all_originals.append(original_values)
        all_kde_samples.append(kde_samples)

    df = pd.DataFrame(results)

    summary = plot(df, all_originals, all_kde_samples, output_folder=os.path.join(args.output_folder, "clean_figs/supplementary"))