"""Top-hat (uniform) kernel density evaluator and multivariate sampler.

The top-hat kernel is the simplest compactly supported kernel: it assigns a
constant density to every point inside the unit ball and zero outside.  The
normalisation constant equals the reciprocal of the volume of the
*d*-dimensional unit ball so that the kernel integrates to unity in any
dimensionality.

Two entry points are provided:

* :func:`tophat_kernel` — evaluates the kernel density contributions on an
  arbitrary set of query points.
* :func:`sample_tophat_multivariate` — draws independent samples uniformly
  from the *d*-dimensional unit ball via rejection sampling.
"""

import numpy as np
from scipy.special import gamma
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs


def tophat_kernel(x, y, h):
    """Evaluate the *d*-dimensional top-hat kernel centred at *x* on the points *y*.

    The kernel is defined as

    .. math::

        K(\\mathbf{u}) = \\frac{1}{h^d \\, V_d}
        \\;\\mathbf{1}_{\\|\\mathbf{u}\\| \\le 1}

    where :math:`\\mathbf{u} = (\\mathbf{y} - \\mathbf{x}) / h` and
    :math:`V_d = \\Gamma\\!\\bigl(\\tfrac{d+2}{2}\\bigr)\\,\\pi^{-d/2}`
    is the volume-related normalisation constant for the *d*-dimensional
    unit ball.

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
    return 1 / h ** d / c_d * admissible


def sample_tophat_multivariate(n_samples=1, d=1):
    """Draw independent samples uniformly from the *d*-dimensional unit ball.

    Sampling is performed by rejection: candidate points are drawn
    uniformly inside the *d*-dimensional hypercube ``[-1, 1]^d`` and
    accepted only if their Euclidean norm does not exceed 1.

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
        if norm <= 1:
            samples.append(z)
    return np.array(samples)