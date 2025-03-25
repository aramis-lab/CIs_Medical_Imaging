import numpy as np
from scipy.stats import trim_mean


def mean(x, threshold, axis=None):
    return np.mean(x, axis=axis)

def median(x, threshold, axis=None):
    return np.median(x, axis=axis)

def trimmed_mean(x, threshold, axis=None):
    return trim_mean(x, threshold, axis=axis)

def std(x, threshold, axis=None):
    return np.sqrt(np.var(x, axis=axis))

def IQR_width(x, threshold, axis=None):
    q3 = np.percentile(x, 75, axis=axis)
    q1 = np.percentile(x, 25, axis=axis)
    return q3 - q1

def get_statistic(statistic):

    statistic_dict = {
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed_mean,
        "std": std,
        "IQR_width": IQR_width
    }

    return statistic_dict.get(statistic, None)