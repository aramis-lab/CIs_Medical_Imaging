"""Low-level input preprocessing and scaled-difference computation shared
by all kernel implementations.

Every kernel module delegates two preliminary tasks to this module before
evaluating its specific profile:

1. **Input canonicalisation** — scalar, 1-D, and 2-D inputs for centres,
   query points, and bandwidths are promoted to a uniform 2-D shape so
   that downstream code can rely on consistent array ranks.
2. **Scaled-difference tensor** — the pair-wise differences between centres
   and query points are computed and divided by the bandwidth, yielding the
   dimensionless quantity :math:`\\mathbf{u}` that every kernel profile is
   expressed in terms of.
"""

import numpy as np


def preprocess_kernel_inputs(x, y, h):
    """Canonicalise kernel inputs to 2-D arrays with compatible shapes.

    Scalar and 1-D inputs are promoted so that *x* has shape ``(n, d)``,
    *y* has shape ``(m, d)``, and *h* has shape ``(n, 1)`` (or ``(n, d)``),
    where *n* is the number of centres, *m* the number of query points, and
    *d* the dimensionality.

    Parameters
    ----------
    x : array_like
        Kernel centre(s).  Scalars and 1-D arrays are reshaped to
        ``(n, 1)``.
    y : array_like
        Query point(s) at which the kernel will be evaluated.  Scalars and
        1-D arrays are reshaped to ``(m, 1)``.
    h : array_like
        Bandwidth(s).  Scalars and 1-D arrays are reshaped to ``(n, 1)``.

    Returns
    -------
    x : np.ndarray, shape ``(n, d)``
        Canonicalised centres.
    y : np.ndarray, shape ``(m, d)``
        Canonicalised query points.
    h : np.ndarray, shape ``(n, 1)`` or ``(n, d)``
        Canonicalised bandwidths.

    Raises
    ------
    ValueError
        If the dimensionality of *x* and *y* (second axis) do not match,
        or if the number of centres in *x* and bandwidths in *h* (first
        axis) do not match.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.asarray(h)

    if x.ndim == 0:
        x = np.array([x])
    if y.ndim == 0:
        y = np.array([y])
    if h.ndim == 0:
        h = np.array([h])
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of dimensions.")
    if x.shape[0] != h.shape[0]:
        raise ValueError("x and h must have the same number of data points.")

    return x, y, h


def compute_scaled_differences(x, y, h):
    """Compute the bandwidth-scaled difference tensor between centres and query points.

    After canonicalising the inputs via :func:`preprocess_kernel_inputs`,
    the function broadcasts *x* and *y* against each other and divides by
    *h* to produce the dimensionless displacement

    .. math::

        u_{i,j} = \\frac{x_i - y_j}{h_i}

    for every centre *i* and query point *j*.

    Parameters
    ----------
    x : array_like
        Kernel centre(s), passed through :func:`preprocess_kernel_inputs`.
    y : array_like
        Query point(s), passed through :func:`preprocess_kernel_inputs`.
    h : array_like
        Bandwidth(s), passed through :func:`preprocess_kernel_inputs`.

    Returns
    -------
    np.ndarray, shape ``(n, m, d)``
        The scaled differences, where *n* is the number of centres, *m*
        the number of query points, and *d* the dimensionality.
    """
    x, y, h = preprocess_kernel_inputs(x, y, h)
    x_exp = x[:, None, :]       # (n, 1, d)
    y_exp = y[None, :, :]       # (1, m, d)
    h_exp = h[:, None]          # (n, 1, 1)
    u = (x_exp - y_exp) / h_exp # (n, m, d)
    return u