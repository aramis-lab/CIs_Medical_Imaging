import numpy as np
from scipy.stats import t, norm, bootstrap

def compute_CIs(samples, method, summary_stat_name, statistic, threshold, alpha=0.05):
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=0)
    if method == "param_t":
        return param_t_interval(samples, summary_stat_name, threshold, alpha)
    elif method == "param_z":
        return param_z_interval(samples, summary_stat_name, threshold, alpha)
    elif method in ["basic", "percentile", "bca"]:
        return compute_bootstrap_CI(samples, statistic, alpha, method)
    else:
        return np.empty((samples.shape[0], 2))

def param_z_interval(data, summary_stat_name, threshold, alpha=0.05):
    if summary_stat_name == "trimmed_mean":
        lowercut = int(threshold * len(data))
        uppercut = len(data) - lowercut
        data = data[lowercut:uppercut]
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    z_score = norm.ppf(1 - alpha / 2)
    return np.vstack([means - z_score * std_errors, means + z_score * std_errors]).T

def param_t_interval(data, summary_stat_name, threshold, alpha=0.05):
    if summary_stat_name == "trimmed_mean":
        lowercut = int(threshold * len(data))
        uppercut = len(data) - lowercut
        data = data[lowercut:uppercut]
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    t_score = t.ppf(1 - alpha / 2, df=data.shape[1] - 1)
    return np.vstack([means - t_score * std_errors, means + t_score * std_errors]).T

def compute_bootstrap_CI(data, statistic, alpha=0.05, method="percentile"):
    bootstrap_ci = bootstrap((data,), statistic=statistic, vectorized=True, axis=1, batch=1, confidence_level=1 - alpha, n_resamples=9999, method=method).confidence_interval
    ci_bounds = np.array([bootstrap_ci.low, bootstrap_ci.high])
    return ci_bounds.squeeze().T