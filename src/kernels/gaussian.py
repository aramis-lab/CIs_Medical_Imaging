import numpy as np

def gaussian_kernel(x, y, h):
    return np.exp(-np.sum((x - y) ** 2) / (2 * h ** 2)) / np.sqrt(2 * np.pi * h ** 2)