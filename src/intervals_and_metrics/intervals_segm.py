"""Confidence-interval methods for segmentation summary statistics (mean,
median, trimmed mean, etc. of a per-sample metric like Dice score).

Dispatches to parametric (t/z) intervals, bootstrap intervals (basic,
percentile, BCa), or concentration-inequality bounds (Hoeffding, Bennett),
depending on the requested ``method``.
"""

import numpy as np
from scipy.stats import t, norm, bootstrap
from scipy.special import lambertw

def compute_CIs_segmentation(samples, method, summary_stat_name, statistic, threshold, a=-np.inf, b=np.inf, alpha=0.05):
    """Compute a confidence interval for a segmentation summary statistic.

    Parameters
    ----------
    samples : numpy.ndarray
        Array of shape ``(n_batches, n_samples)`` (or 1-D, which is treated
        as a single batch of shape ``(1, n_samples)``) holding per-sample
        metric values (e.g. per-image Dice scores).
    method : str
        CI method: ``"param_t"``, ``"param_z"``, ``"basic"``, ``"percentile"``,
        ``"bca"``, ``"hoeffding"``, or ``"benett"``.
    summary_stat_name : str
        Name of the summary statistic being estimated (``"mean"``,
        ``"median"``, ``"trimmed_mean"``, ...). Only affects behavior for
        the parametric methods, where ``"trimmed_mean"`` trims the data
        before computing the mean/std.
    statistic : callable
        The statistic function to bootstrap, passed through to
        :func:`compute_bootstrap_CI` for the ``"basic"``/``"percentile"``/``"bca"``
        methods.
    threshold : float
        Trimming threshold (fraction removed from each tail) used when
        ``summary_stat_name == "trimmed_mean"``.
    a, b : float, default -inf, inf
        Theoretical bounds of the metric, required (finite) for the
        concentration-inequality methods.
    alpha : float, default 0.05
        Significance level; the returned interval has nominal coverage
        ``1 - alpha``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_batches, 2)`` with the lower and upper bound for
        each batch, or an empty array of that shape if the requested
        ``method``/``summary_stat_name``/bounds combination isn't supported.
    """
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
    """Parametric confidence interval for the mean, using the normal (z) quantile.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_batches, n_samples)``.
    summary_stat_name : str
        If ``"trimmed_mean"``, the lowest/highest ``threshold`` fraction of
        samples (per batch, after sorting) is removed before computing the
        mean and standard error.
    threshold : float
        Fraction of samples trimmed from each tail when
        ``summary_stat_name == "trimmed_mean"``.
    alpha : float, default 0.05
        Significance level; the returned interval has nominal coverage
        ``1 - alpha``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_batches, 2)`` with the lower and upper bound for
        each batch.
    """
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
    """Parametric confidence interval for the mean, using the Student-t quantile.

    Same as :func:`param_z_interval` but uses the t-distribution (with
    ``n - 1`` degrees of freedom) instead of the normal distribution, which
    is more appropriate for small sample sizes.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_batches, n_samples)``.
    summary_stat_name : str
        If ``"trimmed_mean"``, trims each batch as in :func:`param_z_interval`.
    threshold : float
        Trimming fraction when ``summary_stat_name == "trimmed_mean"``.
    alpha : float, default 0.05
        Significance level; the returned interval has nominal coverage
        ``1 - alpha``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_batches, 2)`` with the lower and upper bound for
        each batch.
    """
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
    """Inverse of the function ``h(u) = (1+u) * log(1+u) - u``, used in Bennett's inequality.

    Computed in closed form via the Lambert W function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s).

    Returns
    -------
    numpy.ndarray or float
        ``h^{-1}(x)``, real-valued.
    """
    return np.exp(1+np.real(lambertw((x-1)/np.e)))

def concentration_interval(data, method, alpha=0.05, a=-np.inf, b=np.inf):
    """Confidence interval for the mean via a concentration inequality.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_batches, n_samples)``.
    method : {"hoeffding", "benett"}
        Which concentration inequality to use.
    alpha : float, default 0.05
        Significance level; the returned interval has nominal coverage
        at least ``1 - alpha``.
    a, b : float
        Finite theoretical bounds of the underlying random variable; the
        interval radius scales with ``M = b - a``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_batches, 2)`` with the lower and upper bound for
        each batch, centered on the sample mean.

    Raises
    ------
    ValueError
        If ``method`` is not ``"hoeffding"`` or ``"benett"``.
    """
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
    """Bootstrap confidence interval for an arbitrary statistic, via ``scipy.stats.bootstrap``.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_batches, n_samples)``; bootstrap resampling is
        performed independently per batch, along ``axis=1``.
    statistic : callable
        Vectorized statistic function, called as ``statistic(data, axis=1)``.
    alpha : float, default 0.05
        Significance level; the returned interval has nominal coverage
        ``1 - alpha``.
    method : {"percentile", "basic", "bca"}, default "percentile"
        Bootstrap CI method, forwarded to ``scipy.stats.bootstrap``.
        Uses 9999 resamples in all cases.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_batches, 2)`` with the lower and upper bound for
        each batch.
    """
    bootstrap_ci = bootstrap((data,), statistic=statistic, vectorized=True, axis=1, batch=1, confidence_level=1 - alpha, n_resamples=9999, method=method).confidence_interval
    ci_bounds = np.array([bootstrap_ci.low, bootstrap_ci.high])
    return ci_bounds.squeeze().T