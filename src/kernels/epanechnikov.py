import numpy as np
from scipy.special import gamma

def epanechnikov_kernel(x, y, h):
    if len(np.asarray(x))>0:
        d = x.shape[0]
    else:
        d = 1
    admissible = np.abs(np.linalg.norm(x - y)) <= h
    normalization_constant = (d+2)/ 2 * gamma((d + 2) / 2) / np.pi**(d / 2)
    return normalization_constant * (1 - np.linalg.norm((x - y) / h) ** 2) / (h**d) * admissible

def sample_epanechnikov_multivariate(n_samples=1, d=1):
    samples = []
    while len(samples) < n_samples:
        z = np.random.uniform(-1, 1, d)
        norm_sq = np.sum(z ** 2)
        if norm_sq <= 1 and np.random.rand() < 1 - norm_sq:
            samples.append(z)
    return np.array(samples)