import numpy as np
from scipy.stats import t, norm, bootstrap
from scipy.special import lambertw

def compute_CIs_segmentation(samples, method, summary_stat_name, statistic, threshold, a=-np.inf, b=np.inf, alpha=0.05):
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=0)
    if method == "param_t":
        return param_t_interval(samples, summary_stat_name, threshold, alpha)
    elif method == "param_z":
        return param_z_interval(samples, summary_stat_name, threshold, alpha)
    elif method in ["basic", "percentile", "bca"]:
        return compute_bootstrap_CI(samples, statistic, alpha, method)
    elif (summary_stat_name == "mean") and (method in ["hoeffding", "benett"]) and np.isfinite(a) and np.isfinite(b):
        return concentration_interval(samples, method, alpha, a, b)
    else:
        print(summary_stat_name, method, a, b)
        return np.empty((samples.shape[0], 2))

def param_z_interval(data, summary_stat_name, threshold, alpha=0.05):
    data = np.sort(data, axis=1)
    # If summary_stat_name is "trimmed_mean", we trim the data
    # by removing the lowest and highest threshold percent of samples
    if summary_stat_name == "trimmed_mean":
        lowercut = int(threshold * data.shape[1])
        uppercut = data.shape[1] - lowercut
        data = data[:, lowercut:uppercut]
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    z_score = norm.ppf(1 - alpha / 2)
    return np.vstack([means - z_score * std_errors, means + z_score * std_errors]).T

def param_t_interval(data, summary_stat_name, threshold, alpha=0.05):
    data = np.sort(data, axis=1)
    # If summary_stat_name is "trimmed_mean", we trim the data
    # by removing the lowest and highest threshold percent of samples
    if summary_stat_name == "trimmed_mean":
        lowercut = int(threshold * data.shape[1])
        uppercut = data.shape[1] - lowercut
        data = data[:, lowercut:uppercut]
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    t_score = t.ppf(1 - alpha / 2, df=data.shape[1] - 1)
    return np.vstack([means - t_score * std_errors, means + t_score * std_errors]).T

def h_inv(x):
    return np.exp(1+np.real(lambertw((x-1)/np.e)))

def concentration_interval(data, method, alpha=0.05, a=-np.inf, b=np.inf):
    n = data.shape[1]
    M = (b - a)
    if method == "hoeffding":
        means = np.mean(data, axis=1)
        radius = np.sqrt((np.log(2 / alpha) * M ** 2) / (2 * n))
    elif method == "benett":
        means = np.mean(data, axis=1)
        radius = (M**2) / 4 * h_inv(4/(n*M**2) * np.log(2/alpha))
    else:
        raise ValueError(f"Unknown concentration inequality method: {method}")
    return np.vstack([means - radius, means + radius]).T

def compute_bootstrap_CI(data, statistic, alpha=0.05, method="percentile"):
    bootstrap_ci = bootstrap((data,), statistic=statistic, vectorized=True, axis=1, batch=1, confidence_level=1 - alpha, n_resamples=9999, method=method).confidence_interval
    ci_bounds = np.array([bootstrap_ci.low, bootstrap_ci.high])
    return ci_bounds.squeeze().T