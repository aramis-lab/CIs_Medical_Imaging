import numpy as np
from scipy.stats import trim_mean


def mean(x, threshold, axis=None):
    return np.mean(x, axis=axis)

def median(x, threshold, axis=None):
    return np.median(x, axis=axis)

def trimmed_mean(x, threshold, axis=None):
    return trim_mean(x, threshold, axis=axis)

def get_statistic(statistic):

    statistic_dict = {
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed_mean
    }

    return statistic_dict.get(statistic, None)