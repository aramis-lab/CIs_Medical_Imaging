"""Cosine kernel density evaluator and multivariate sampler.

The cosine kernel is a compactly supported kernel defined on the unit ball:
it equals zero whenever the (Euclidean) distance from the centre exceeds 1,
and follows a cosine profile otherwise.  The normalisation constant is
derived from the surface area of the *d*-dimensional unit ball so that the
kernel integrates to unity in any dimensionality.

Two entry points are provided:

* :func:`cosine_kernel` — evaluates the kernel density contributions on an
  arbitrary set of query points.
* :func:`sample_cosine_multivariate` — draws independent samples from the
  cosine distribution via rejection sampling inside the unit ball.
"""

import numpy as np
from scipy.special import gamma

from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def cosine_kernel(x, y, h):
    """Evaluate the *d*-dimensional cosine kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = C_d \\;\\cos\\!\\bigl(\\tfrac{\\pi}{2}
        \\|\\mathbf{u}\\|\\bigr) \\;\\mathbf{1}_{\\|\\mathbf{u}\\| \\le 1}

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h` and
    :math:`C_d` is the normalisation constant
    :math:`\\Gamma(d/2+1)\\,\\pi^{-d/2}\\,\\pi/4`.

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
    admissible = norms <= 1
    normalization_constant = gamma(d / 2 + 1) / np.pi ** (d / 2) * np.pi / 4
    return admissible * normalization_constant / h ** d * np.cos(np.pi / 2 * norms)


def sample_cosine_multivariate(n_samples=1, d=1):
    """Draw independent samples from the *d*-dimensional cosine distribution.

    Sampling is performed by rejection: candidate points are drawn
    uniformly inside the *d*-dimensional hypercube ``[-1, 1]^d`` and
    accepted with probability proportional to
    :math:`\\cos(\\tfrac{\\pi}{2} \\|\\mathbf{z}\\|)`, restricted to the
    unit ball.

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
        norm = np.linalg.norm(z)
        if norm <= 1 and np.random.rand() < (np.pi / 4) * np.cos((np.pi / 2) * norm):
            samples.append(z)
    return np.array(samples)