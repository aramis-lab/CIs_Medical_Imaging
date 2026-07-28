"""Gaussian kernel density evaluator and multivariate sampler.

The Gaussian (normal) kernel is the most widely used kernel in
non-parametric density estimation.  It has infinite support and a smooth,
bell-shaped profile that factorises along each coordinate axis.  Because
multivariate standard-normal variates are readily available, sampling
requires no rejection step.

Two entry points are provided:

* :func:`gaussian_kernel` — evaluates the kernel density contributions on
  an arbitrary set of query points.
* :func:`sample_gaussian_multivariate` — draws independent samples from the
  standard multivariate normal distribution.
"""

import numpy as np
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def gaussian_kernel(x, y, h):
    """Evaluate the *d*-dimensional Gaussian kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = \\frac{1}{(h\\sqrt{2\\pi})^d}
        \\exp\\!\\bigl(-\\tfrac{1}{2}\\|\\mathbf{u}\\|^2\\bigr)

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h`.

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
    norms = np.linalg.norm(u, axis=-1)
    constant = 1 / (h * np.sqrt(2 * np.pi)) ** d
    return constant * np.exp(-0.5 * norms ** 2)


def sample_gaussian_multivariate(n_samples=1, d=1):
    """Draw independent samples from the *d*-dimensional standard normal distribution.

    Each coordinate is sampled independently from a standard normal
    distribution (mean 0, variance 1).

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to draw.  Defaults to ``1``.
    d : int, optional
        Dimensionality of the samples.  Defaults to ``1``.

    Returns
    -------
    np.ndarray, shape ``(n_samples, d)``
        The drawn samples, each coordinate distributed as
        :math:`\\mathcal{N}(0, 1)`.
    """
    return np.random.normal(size=(n_samples, d))