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
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp, epps_singleton_2samp, energy_distance
from statsmodels.stats.multitest import multipletests

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
    ks_stat, ks_pval = ks_2samp(original, kde_samples)
    cvm_res = cramervonmises_2samp(original, kde_samples)
    ad_stat, _, ad_pval = anderson_ksamp([original, kde_samples])
    try:
        es_stat, es_pval = epps_singleton_2samp(original, kde_samples)
    except Exception:
        es_stat, es_pval = np.nan, np.nan
    e_dist = energy_distance(original, kde_samples)
    mean_err = abs(np.mean(kde_samples) - np.mean(original)) / (np.std(original) + 1e-10)
    std_ratio = np.std(kde_samples) / (np.std(original) + 1e-10)
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q_orig = np.quantile(original, probs)
    q_kde = np.quantile(kde_samples, probs)
    max_q_err = np.max(np.abs(q_kde - q_orig) / (np.abs(q_orig) + 1e-10))
    median_q_err = np.median(np.abs(q_kde - q_orig) / (np.abs(q_orig) + 1e-10))
    return {
        'n': n, 'ks_stat': ks_stat, 'ks_pval': ks_pval,
        'cvm_stat': cvm_res.statistic, 'cvm_pval': cvm_res.pvalue,
        'ad_stat': ad_stat, 'ad_pval': ad_pval,
        'es_stat': es_stat, 'es_pval': es_pval,
        'energy_dist': e_dist, 'mean_err': mean_err, 'std_ratio': std_ratio,
        'max_q_err': max_q_err, 'median_q_err': median_q_err,
    }

def multiple_tests_and_plot(df, all_originals, all_kde_samples, output_folder="."):
    # ── Multiple testing correction ──
    rejected_bh, pvals_bh, _, _ = multipletests(df['ks_pval'], alpha=0.05, method='fdr_bh')
    rejected_bonf, _, _, _ = multipletests(df['ks_pval'], alpha=0.05, method='bonferroni')
    df['ks_pval_bh'] = pvals_bh
    df['rejected_bh'] = rejected_bh
    print(f"Raw:  {(df['ks_pval'] < 0.05).sum()}/{len(df)} rejected at alpha=0.05")
    print(f"BH:   {rejected_bh.sum()}/{len(df)} rejected after FDR correction")
    print(f"Bonf: {rejected_bonf.sum()}/{len(df)} rejected after Bonferroni correction")

    # ── Dashboard ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].hist(df['ks_stat'], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[0, 0].axvline(df['ks_stat'].median(), color='red', ls='--',
                        label=f"Median = {df['ks_stat'].median():.4f}")
    axes[0, 0].set_xlabel('2-Sample KS Statistic'); axes[0, 0].set_title('(a) KS Statistics')
    axes[0, 0].legend()

    axes[0, 1].hist(df['ks_pval'], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[0, 1].set_xlabel('KS p-value'); axes[0, 1].set_title('(b) KS p-values')
    axes[0, 1].legend()

    axes[0, 2].hist(df['energy_dist'], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[0, 2].axvline(df['energy_dist'].median(), color='red', ls='--',
                        label=f"Median = {df['energy_dist'].median():.4f}")
    axes[0, 2].set_xlabel('Energy Distance'); axes[0, 2].set_title('(c) Energy Distance')
    axes[0, 2].legend()

    axes[1, 0].hist(df['std_ratio'], bins=50, alpha=0.7, edgecolor='k')
    axes[1, 0].axvline(1.0, color='red', ls='--', lw=2, label='Ideal = 1.0')
    axes[1, 0].set_xlabel('sigma_KDE / sigma_data'); axes[1, 0].set_title('(d) Variance Preservation')
    axes[1, 0].legend()

    axes[1, 1].hist(df['mean_err'], bins=50, alpha=0.7, edgecolor='k')
    axes[1, 1].axvline(0, color='red', ls='--')
    axes[1, 1].set_xlabel('|mu_KDE - mu_data| / sigma_data'); axes[1, 1].set_title('(e) Mean Discrepancy')

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
    axes[1, 2].scatter(all_q_orig, all_q_kde, s=1, alpha=0.3)
    lims = [min(all_q_orig.min(), all_q_kde.min()), max(all_q_orig.max(), all_q_kde.max())]
    axes[1, 2].plot(lims, lims, 'r--', lw=2)
    axes[1, 2].set_xlabel('Standardised Empirical Quantiles')
    axes[1, 2].set_ylabel('Standardised KDE Quantiles')
    axes[1, 2].set_title('(f) Pooled Q-Q (all distributions)')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/kde_validation_2sample.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Summary table ──
    summary = pd.DataFrame({
        'Metric': [
            'KS statistic (median)', 'KS p > 0.05 (%)', 'KS p > 0.05 after BH (%)',
            'CvM p > 0.05 (%)', 'AD p > 0.05 (%)', 'Energy distance (median)',
            '|d_mu|/sigma (median)', 'sigma_KDE/sigma_data (median)',
            'Max quantile rel. error (median)',
        ],
        'Value': [
            f"{df['ks_stat'].median():.4f}",
            f"{(df['ks_pval'] > 0.05).mean()*100:.1f}",
            f"{(~df['rejected_bh']).mean()*100:.1f}",
            f"{(df['cvm_pval'] > 0.05).mean()*100:.1f}",
            f"{(df['ad_pval'] > 0.05).mean()*100:.1f}",
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

    summary = multiple_tests_and_plot(df, all_originals, all_kde_samples, output_folder=os.path.join(args.output_folder, "clean_figs/supplementary"))

    print("\nSummary of KDE representativeness across all distributions:")
    print(summary.to_string(index=False))