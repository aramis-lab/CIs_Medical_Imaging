"""Linear (triangular) kernel density evaluator and multivariate sampler.

The linear kernel is a compactly supported kernel defined on the unit ball
whose profile decreases linearly from its centre to zero at the boundary.
In one dimension it reduces to the familiar triangular kernel.  The
normalisation constant accounts for the *d*-dimensional geometry of the
unit ball so that the kernel integrates to unity in any dimensionality.

Two entry points are provided:

* :func:`linear_kernel` — evaluates the kernel density contributions on an
  arbitrary set of query points.
* :func:`sample_linear_multivariate` — draws independent samples from the
  linear distribution by combining uniform random directions with radii
  sampled via the inverse-CDF method.
"""

import numpy as np
from scipy.special import gamma
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def linear_kernel(x, y, h):
    """Evaluate the *d*-dimensional linear kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = \\frac{C_d}{h^d}
        \\bigl(1 - \\|\\mathbf{u}\\|\\bigr)
        \\;\\mathbf{1}_{\\|\\mathbf{u}\\| \\le 1}

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h` and
    :math:`C_d = 2^d\\,\\Gamma(d+1)\\;/\\;
    \\bigl[\\Gamma\\!\\bigl(\\tfrac{d+2}{2}\\bigr)\\,\\pi^{-d/2}\\bigr]`.

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
    c_d = gamma((d + 2) / 2) / np.pi ** (d / 2)
    normalization_constant = 2 ** d * gamma(d + 1) / c_d
    return (1 - norms) * admissible / h ** d * normalization_constant


def sample_linear_multivariate(n_samples=1, d=1):
    """Draw independent samples from the *d*-dimensional linear distribution.

    Sampling proceeds in three steps:

    1. A uniform random direction on the *d*-dimensional unit sphere is
       drawn for each sample.
    2. A radius is drawn from the radial distribution
       :math:`f(r) \\propto r^{d-1}(1-r)` for :math:`r \\in [0, 1]` using
       the inverse-CDF method.
    3. Each direction is scaled by its corresponding radius.

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
    directions = np.random.randn(n_samples, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Step 2: Sample radii from the radial distribution
    # PDF: f(r) ∝ r^{d-1} (1 - r), for r in [0,1]
    u = np.random.rand(n_samples)
    radii = 1 - (1 - u) ** (1 / (d + 1))  # Inverse CDF method

    # Step 3: Multiply radius by direction
    samples = directions * radii[:, None]
    return samples