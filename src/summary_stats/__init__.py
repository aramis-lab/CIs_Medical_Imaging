"""Summary-statistic functions and a factory for retrieving them by name.

Each function in this module computes a single summary statistic from an
array of metric values (e.g. per-sample Dice scores).  They all share a
uniform signature ``(x, threshold, axis)`` so that higher-level code can
call any of them interchangeably — even when a particular statistic does
not use the *threshold* parameter.  The factory function
:func:`get_statistic` maps a human-readable name to the corresponding
callable, serving as the single look-up point used by the bootstrap and
confidence-interval pipelines.
"""

import numpy as np


def mean(x, threshold, axis=None):
    """Compute the arithmetic mean along an axis.

    Parameters
    ----------
    x : array_like
        Input data.
    threshold : float
        Unused.  Accepted for signature compatibility with other summary
        statistics (e.g. :func:`trimmed_mean`).
    axis : int or None, optional
        Axis along which the mean is computed.  Defaults to *None*
        (mean of the flattened array).

    Returns
    -------
    np.ndarray
        The arithmetic mean, with the reduced axis kept as a size-1
        dimension (``keepdims=True``).
    """
    return np.mean(x, axis=axis, keepdims=True)


def median(x, threshold, axis=None):
    """Compute the median along an axis.

    Parameters
    ----------
    x : array_like
        Input data.
    threshold : float
        Unused.  Accepted for signature compatibility with other summary
        statistics (e.g. :func:`trimmed_mean`).
    axis : int or None, optional
        Axis along which the median is computed.  Defaults to *None*
        (median of the flattened array).

    Returns
    -------
    np.ndarray
        The median, with the reduced axis kept as a size-1 dimension
        (``keepdims=True``).
    """
    return np.median(x, axis=axis, keepdims=True)


def trimmed_mean(x, threshold, axis=None):
    """Compute the trimmed (truncated) mean along an axis.

    A fraction *threshold* of observations is removed from **each** tail
    before computing the arithmetic mean of the remaining central
    observations.

    Parameters
    ----------
    x : array_like
        Input data.
    threshold : float
        Fraction of observations to trim from each tail (e.g. ``0.1``
        removes the lowest 10 % and the highest 10 %).
    axis : int or None, optional
        Axis along which the trimmed mean is computed.  If *None* the
        array is flattened first and the computation proceeds along
        axis 0.

    Returns
    -------
    np.ndarray
        The trimmed mean, with the reduced axis kept as a size-1
        dimension (``keepdims=True``).

    Raises
    ------
    ValueError
        If *threshold* is so large that the lower cut index exceeds
        the upper cut index, leaving no observations to average.
    """
    a = np.asarray(x)

    if a.size == 0:
        return np.nan

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(threshold * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = np.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return np.mean(atmp[tuple(sl)], axis=axis, keepdims=True)


def std(x, threshold, axis=None):
    """Compute the sample standard deviation along an axis.

    Uses one degree-of-freedom correction (``ddof=1``) to produce an
    unbiased estimator of the population standard deviation.

    Parameters
    ----------
    x : array_like
        Input data.
    threshold : float
        Unused.  Accepted for signature compatibility with other summary
        statistics (e.g. :func:`trimmed_mean`).
    axis : int or None, optional
        Axis along which the standard deviation is computed.  Defaults to
        *None* (standard deviation of the flattened array).

    Returns
    -------
    np.ndarray
        The sample standard deviation, with the reduced axis kept as a
        size-1 dimension (``keepdims=True``).
    """
    return np.std(x, axis=axis, keepdims=True, ddof=1)


def IQR_length(x, threshold, axis=None):
    """Compute the interquartile range (IQR) along an axis.

    The IQR is defined as the difference between the 75th and 25th
    percentiles.

    Parameters
    ----------
    x : array_like
        Input data.
    threshold : float
        Unused.  Accepted for signature compatibility with other summary
        statistics (e.g. :func:`trimmed_mean`).
    axis : int or None, optional
        Axis along which the IQR is computed.  Defaults to *None*
        (IQR of the flattened array).

    Returns
    -------
    np.ndarray
        The interquartile range, with the reduced axis kept as a size-1
        dimension (``keepdims=True``).
    """
    q3 = np.percentile(x, 75, axis=axis, keepdims=True)
    q1 = np.percentile(x, 25, axis=axis, keepdims=True)
    return q3 - q1


def get_statistic(statistic):
    """Return the summary-statistic function associated with a name.

    Parameters
    ----------
    statistic : str
        Name of the desired statistic, one of ``"mean"``, ``"median"``,
        ``"trimmed_mean"``, ``"std"``, or ``"iqr_length"``.

    Returns
    -------
    callable or None
        A function with signature ``f(x, threshold, axis)`` that computes
        the requested summary statistic, or *None* if *statistic* does not
        match any registered name.
    """
    statistic_dict = {
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed_mean,
        "std": std,
        "iqr_length": IQR_length
    }

    return statistic_dict.get(statistic, None)