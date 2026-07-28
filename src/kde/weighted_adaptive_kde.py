"""Weighted kernel density estimation and sampling for univariate and
multivariate data with adaptive, per-sample bandwidths.

In the univariate case the base bandwidth follows Silverman's rule of thumb
and is then scaled per observation by a user-supplied factor (``alphas``) and
clamped so that it never exceeds the distance from the observation to the
domain boundary. In the multivariate case the bandwidth matrix is derived
from the sample covariance (Scott's rule), regularised to be positive
semi-definite, and each drawn perturbation is rescaled by the corresponding
per-observation weight.

These routines are used downstream to build smooth bootstrap resamples from
empirical metric samples — either by evaluating and then inverting a density
(univariate path) or by perturbing selected data points directly
(multivariate path).
"""

import numpy as np
from ..kernels import get_sampling_function


def weighted_kde(data: np.ndarray, x_points: np.ndarray, dist_to_bounds: np.ndarray, kernel=None, alphas=None):
    """Evaluate a 1-D adaptive-bandwidth weighted KDE on a grid.

    For every observation in *data* the bandwidth is set to Silverman's
    rule-of-thumb value scaled by the matching entry in *alphas* and then
    clamped to *dist_to_bounds* so that the kernel never leaks outside the
    domain.  When the effective bandwidth is zero or no *kernel* is supplied
    the observation is placed into the nearest grid bin as a Dirac-like
    spike whose height preserves the density normalisation.

    Parameters
    ----------
    data : np.ndarray, shape ``(n,)``
        The observed 1-D sample values.
    x_points : np.ndarray, shape ``(m,)``
        A uniformly spaced evaluation grid on which the density is
        estimated.  The spacing ``x_points[1] - x_points[0]`` is assumed
        constant and is used for the Dirac fallback.
    dist_to_bounds : np.ndarray, shape ``(n,)``
        Maximum allowable bandwidth for each observation, typically the
        distance from the observation to the nearest domain boundary.
    kernel : callable or None, optional
        A kernel function with signature ``kernel(centre, x_points, bw)``
        returning an array of shape ``(m,)``.  If *None* every observation
        falls back to the Dirac-like placement.
    alphas : np.ndarray, shape ``(n,)`` or None, optional
        Per-observation multiplicative scaling factors applied to the base
        bandwidth.  Defaults to all ones (uniform scaling).

    Returns
    -------
    np.ndarray, shape ``(m,)``
        The estimated density evaluated at each point in *x_points*,
        normalised by *n*.
    """
    n = len(data)
    bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)

    if alphas is None:
        alphas = np.ones(n)

    bandwidths = bandwidth * alphas
    bandwidths = np.min([bandwidths, dist_to_bounds], axis=0)

    density = np.zeros_like(x_points)

    for i in range(n):
        if bandwidths[i] > 0 and kernel is not None:
            current_density = kernel(data[i], x_points, bandwidths[i]).squeeze()

            # Ensure total density integrates properly
            density += current_density

        else:
            idx = np.searchsorted(x_points, data[i])
            idx = np.clip(idx, 0, len(x_points) - 1)
            density[idx] += 1 / (x_points[1] - x_points[0])

    return density / n


def sample_weighted_kde(y, x, n_samples, a=0, b=1):
    """Draw random samples from a 1-D KDE via inverse-CDF sampling.

    The density values *y* evaluated on the grid *x* are turned into an
    empirical CDF, which is then inverted at uniformly drawn quantiles to
    produce samples.  The results are clipped to the interval ``[a, b]`` to
    ensure they remain within the metric domain.

    Parameters
    ----------
    y : np.ndarray, shape ``(m,)``
        Density values on the evaluation grid (need not be normalised;
        normalisation is handled internally via ``cumsum / sum``).
    x : np.ndarray, shape ``(m,)``
        The evaluation grid corresponding to *y*.
    n_samples : int
        Number of samples to draw.
    a : float, optional
        Lower bound of the support; drawn samples are clipped to this
        value.  Defaults to ``0``.
    b : float, optional
        Upper bound of the support; drawn samples are clipped to this
        value.  Defaults to ``1``.

    Returns
    -------
    np.ndarray, shape ``(n_samples,)``
        The drawn samples, each lying in ``[a, b]``.
    """
    cdf = np.cumsum(y) / np.sum(y)
    values = np.random.rand(n_samples)

    indices = np.searchsorted(cdf, values)
    inv_cdf = x[indices]

    inv_cdf = np.clip(inv_cdf, a, b)

    return inv_cdf


def sample_weighted_kde_multivariate(data, labels, kernel_name, n_samples, alphas=None):
    """Draw random samples from a multivariate KDE with per-observation scaling.

    Observations are selected with replacement from *data*, then each is
    perturbed by noise drawn from the kernel identified by *kernel_name*
    and shaped by the square root of the bandwidth matrix (Scott's rule
    applied to the sample covariance, regularised to be positive
    semi-definite).  The perturbation for each selected observation is
    additionally scaled by its corresponding entry in *alphas*, allowing
    observations near domain boundaries or with lower local density to
    receive smaller perturbations.

    Parameters
    ----------
    data : np.ndarray, shape ``(n, d)``
        The observed *d*-dimensional samples.
    labels : np.ndarray, shape ``(n,)``
        Labels (e.g. class indices) associated with each observation;
        returned alongside the drawn samples so that downstream code can
        keep track of which class each synthetic point belongs to.
    kernel_name : str
        Name of the kernel whose sampling function is retrieved via
        :func:`get_sampling_function`.  The sampling function must have
        signature ``f(n_samples, d)`` and return an array of shape
        ``(n_samples, d)``.
    n_samples : int
        Number of samples to draw.
    alphas : np.ndarray, shape ``(n,)`` or None, optional
        Per-observation multiplicative scaling factors applied to the
        perturbation.  Defaults to all ones (uniform scaling).

    Returns
    -------
    weighted_samples : np.ndarray, shape ``(n_samples, d)``
        The drawn samples, each formed as an observed data point plus a
        scaled kernel perturbation.
    sampled_labels : np.ndarray, shape ``(n_samples,)``
        The labels corresponding to the data points that were selected
        (before perturbation).
    """
    n, d = data.shape

    # Select indices with replacement
    indices = np.round(np.random.rand(n_samples) * (n - 1)).astype(int)

    # Covariance and bandwidth
    covariance = np.cov(data, rowvar=False)
    factor = 1.06 * n ** (-1.0 / (d + 4))
    bandwidth_matrix = factor * covariance

    # Ensure PSD (positive semidefinite)
    eigvals, eigvecs = np.linalg.eigh(bandwidth_matrix)
    eigvals[eigvals < 1e-10] = 1e-10  # Replace small/negative eigenvalues

    # Square root of safe_bandwidth
    bandwidth_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Sampling
    if alphas is None:
        alphas = np.ones(n)
    samples = data[indices]
    weights = alphas[indices]
    sampled_labels = labels[indices]

    sampling_function = get_sampling_function(kernel_name)
    norm_samples = sampling_function(n_samples, d)

    weighted_samples = samples + (norm_samples @ bandwidth_sqrt.T) * weights[:, np.newaxis]
    return weighted_samples, sampled_labels