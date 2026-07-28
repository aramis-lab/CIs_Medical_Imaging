"""Epanechnikov kernel density evaluator and multivariate sampler.

The Epanechnikov kernel is a compactly supported, parabolic kernel defined
on the unit ball.  It is optimal in a mean-integrated-squared-error sense
among all kernels of a given order, making it a popular choice for
non-parametric density estimation.  The normalisation constant is derived
from the volume of the *d*-dimensional unit ball so that the kernel
integrates to unity in any dimensionality.

Two entry points are provided:

* :func:`epanechnikov_kernel` — evaluates the kernel density contributions
  on an arbitrary set of query points.
* :func:`sample_epanechnikov_multivariate` — draws independent samples from
  the Epanechnikov distribution via rejection sampling inside the unit
  ball.
"""

import numpy as np
from scipy.special import gamma
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def epanechnikov_kernel(x, y, h):
    """Evaluate the *d*-dimensional Epanechnikov kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = C_d \\;\\bigl(1 - \\|\\mathbf{u}\\|^2\\bigr)
        \\;\\mathbf{1}_{\\|\\mathbf{u}\\| \\le 1}

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h` and
    :math:`C_d` is the normalisation constant
    :math:`\\tfrac{d+2}{2}\\,\\Gamma\\!\\bigl(\\tfrac{d+2}{2}\\bigr)
    \\,\\pi^{-d/2}`.

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
        leading shape as the broadcast of *x* and *y*.  Points outside the
        unit ball receive a value of zero.
    """
    x, y, h = preprocess_kernel_inputs(x, y, h)
    u = compute_scaled_differences(x, y, h)
    d = u.shape[-1]
    norms = np.linalg.norm(u, axis=-1)
    # Check if the points are admissible
    admissible = norms <= 1
    normalization_constant = (d + 2) / 2 * gamma((d + 2) / 2) / np.pi ** (d / 2)
    return normalization_constant * (1 - norms ** 2) / (h ** d) * admissible


def sample_epanechnikov_multivariate(n_samples=1, d=1):
    """Draw independent samples from the *d*-dimensional Epanechnikov distribution.

    Sampling is performed by rejection: candidate points are drawn
    uniformly inside the *d*-dimensional hypercube ``[-1, 1]^d`` and
    accepted with probability :math:`1 - \\|\\mathbf{z}\\|^2`, restricted
    to the unit ball.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to draw.  Defaults to ``1``.
    d : int, optional
        Dimensionality of the samples.  Defaults to ``1``.

    Returns
    -------
    np.ndarray, shape ``(n_samples, d)``
        The drawn samples, each lying inside the *d*-dimensional unit
        ball.
    """
    samples = []
    while len(samples) < n_samples:
        z = np.random.uniform(-1, 1, d)
        norm_sq = np.sum(z ** 2)
        if norm_sq <= 1 and np.random.rand() < 1 - norm_sq:
            samples.append(z)
    return np.array(samples)