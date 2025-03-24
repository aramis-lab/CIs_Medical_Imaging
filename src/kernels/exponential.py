import numpy as np


def exponential_kernel(x, y, h):
    return np.exp(-np.abs(x - y) / h)/h