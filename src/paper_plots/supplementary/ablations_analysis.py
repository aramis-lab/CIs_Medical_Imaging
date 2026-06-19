import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from CIs_project.CIs_Medical_Imaging.src.kde import weighted_kde, sample_weighted_kde
from CIs_project.CIs_Medical_Imaging.src.kernels import get_kernel
from CIs_project.CIs_Medical_Imaging.src.intervals_and_metrics import get_bounds, is_continuous
from itertools import product
from tqdm import tqdm


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


BASE_DIR = "CIs_project"
df_name = "data_matrix_grandchallenge_all.csv"
df_path = os.path.join(BASE_DIR, df_name)

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

for task, algo, metric in tqdm(product(tasks, algos, metrics), total=len(tasks)*len(algos)*len(metrics)):
    kde_samples = compute_instance(df, metric, task, str(algo))
    original_values = df[(df["subtask"] == task) & (df["alg_name"] == algo) & (df["score"] == metric)]["value"].to_numpy()
    if original_values.size == 0 or kde_samples is None:
        continue
    
    means.append((np.nanmean(original_values) - np.nanmean(kde_samples))/np.nanmean(original_values))
    vars.append((np.nanvar(original_values) - np.nanvar(kde_samples))/np.nanvar(original_values))
    skews.append((pd.Series(original_values).skew() - pd.Series(kde_samples).skew())/pd.Series(original_values).skew())
    kurts.append((pd.Series(original_values).kurtosis() - pd.Series(kde_samples).kurtosis())/pd.Series(original_values).kurtosis())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(means, bins=30, alpha=0.7, color='black')
axes[0, 1].hist(vars, bins=30, alpha=0.7, color='black')
axes[1, 0].hist(skews, bins=30, alpha=0.7, color='black')
axes[1, 1].hist(kurts, bins=30, alpha=0.7, color='black')

axes[0, 0].set_title('Mean Differences')
axes[0, 1].set_title('Variance Differences')
axes[1, 0].set_title('Skewness Differences')
axes[1, 1].set_title('Kurtosis Differences')

plt.tight_layout()
plt.show()