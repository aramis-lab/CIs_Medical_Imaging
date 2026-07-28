"""Exponential (Laplacian) kernel density evaluator and multivariate sampler.

The exponential kernel is an unbounded, product-form kernel whose profile
decays as the negative exponential of the L1 norm.  In one dimension it
reduces to the standard (symmetric) Laplace density.  Because the kernel
factorises along each coordinate axis, multivariate samples are obtained
simply by drawing independent one-dimensional symmetric-exponential
variates — no rejection step is required.

Two entry points are provided:

* :func:`exponential_kernel` — evaluates the kernel density contributions
  on an arbitrary set of query points.
* :func:`sample_exponential_multivariate` — draws independent samples from
  the multivariate symmetric-exponential distribution.
"""

import numpy as np
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def exponential_kernel(x, y, h):
    """Evaluate the *d*-dimensional exponential kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = \\frac{1}{(2h)^d}
        \\exp\\!\\bigl(-\\|\\mathbf{u}\\|_1\\bigr)

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h` and
    :math:`\\|\\cdot\\|_1` denotes the L1 (Manhattan) norm.

    Parameters
    ----------
    x : array_like
        Centre(s) of the kernel.  Broadcast-compatible with *y* after
        preprocessing by :func:`preprocess_kernel_inputs`.
    y : array_like
        Query point(s) at which the kernel is evaluated.
    h : float or array_like
        Bandwidth (scalar or per-dimension).

    Returns
    -------
    np.ndarray
        Kernel density contributions at each query point, with the same
        leading shape as the broadcast of *x* and *y*.
    """
    x, y, h = preprocess_kernel_inputs(x, y, h)
    u = compute_scaled_differences(x, y, h)
    d = u.shape[-1]
    return np.exp(-np.linalg.norm(u, ord=1, axis=-1)) / (2 * h) ** d


def sample_exponential_multivariate(n_samples=1, d=1):
    """Draw independent samples from the *d*-dimensional symmetric-exponential distribution.

    Each coordinate is sampled independently as a standard exponential
    variate multiplied by a random sign (±1 with equal probability),
    yielding a symmetric Laplace marginal along every axis.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to draw.  Defaults to ``1``.
    d : int, optional
        Dimensionality of the samples.  Defaults to ``1``.

    Returns
    -------
    np.ndarray, shape ``(n_samples, d)``
        The drawn samples, each coordinate distributed as a standard
        symmetric exponential (Laplace with scale 1).
    """
    signs = np.random.choice([-1, 1], size=(n_samples, d))
    values = np.random.exponential(scale=1.0, size=(n_samples, d))
    return signs * values