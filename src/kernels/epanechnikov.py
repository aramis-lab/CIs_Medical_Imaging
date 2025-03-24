import numpy as np

def epanechnikov_kernel(x, y, h):
    admissible = np.abs(x - y) <= h
    return 0.75 * (1 - ((x - y) / h) ** 2) / h * admissible