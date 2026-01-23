import numpy as np

def mean(x, threshold, axis=None):
    return np.mean(x, axis=axis, keepdims=True)

def median(x, threshold, axis=None):
    return np.median(x, axis=axis, keepdims=True)

def trimmed_mean(x, threshold, axis=None):
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
    return np.sqrt(np.var(x, axis=axis, keepdims=True))

def IQR_length(x, threshold, axis=None):
    q3 = np.percentile(x, 75, axis=axis, keepdims=True)
    q1 = np.percentile(x, 25, axis=axis, keepdims=True)
    return q3 - q1

def get_statistic(statistic):

    statistic_dict = {
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed_mean,
        "std": std,
        "iqr_length": IQR_length
    }

    return statistic_dict.get(statistic, None)