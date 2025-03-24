import numpy as np

def linear_kernel(x, y, h):
    admissible = np.abs(x - y) <= h / 2
    return (1 - np.abs(x - y) / h) * admissible / h