import numpy as np
from scipy.stats import t, norm, bootstrap

def compute_CIs(samples, method, statistic, alpha=0.05):
    if method == "param_t":
        return param_t_interval(samples, alpha)
    elif method == "param_z":
        return param_z_interval(samples, alpha)
    elif method == "studentized":
        return studentized_interval(samples, statistic, alpha)
    elif method in ["basic", "percentile", "BCa"]:
        return compute_bootstrap_CI(samples, statistic, alpha, method)

def param_z_interval(data, alpha=0.05):
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    z_score = norm.ppf(1 - alpha / 2)
    return np.vstack([means - z_score * std_errors, means + z_score * std_errors])

def param_t_interval(data, alpha=0.05):
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])
    t_score = t.ppf(1 - alpha / 2, df=data.shape[1] - 1)
    return np.vstack([means - t_score * std_errors, means + t_score * std_errors])

def compute_bootstrap_CI(data, statistic, alpha=0.05, method="percentile"):
    bootstrap_ci = bootstrap((data,), statistic=statistic, vectorized=True, axis=1, batch=1, confidence_level=1 - alpha, n_resamples=9999, method=method).confidence_interval
    ci_bounds = np.array([bootstrap_ci.low, bootstrap_ci.high])
    return ci_bounds

def studentized_interval(samples, statistic, alpha, n_resamples=9999):
    samples_statistics = np.expand_dims(statistic(samples, axis=-1), -2)
    samples_stds = np.std(samples, axis=-1, keepdims=True)
    bootstrap_samples = np.stack([np.random.choice(samples[i], size=(n_resamples, samples.shape[1]), replace=True) for i in range(samples.shape[0])])
    bootstrap_statistics = statistic(bootstrap_samples, axis=-1)
    boostrap_stds = np.std(bootstrap_samples, axis=-1, keepdims=True)
    studentized = (bootstrap_statistics - samples_statistics) / boostrap_stds
    lower_bound = np.percentile(studentized, 100 * alpha / 2, axis=1)
    upper_bound = np.percentile(studentized, 100 * (1 - alpha / 2), axis=1)
    return np.vstack([samples_statistics - lower_bound * samples_stds, samples_statistics - upper_bound * samples_stds])
