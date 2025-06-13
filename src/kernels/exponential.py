import numpy as np
from . import compute_scaled_differences

def exponential_kernel(x, y, h):
    u = compute_scaled_differences(x, y, h)
    d = x.shape[1]
    return np.exp(-np.linalg.norm(u, 1))/(2*h)**d

def sample_exponential_multivariate(n_samples = 1, d = 1):
    signs = np.random.choice([-1, 1], size=(n_samples, d))
    values = np.random.exponential(scale=1.0, size=(n_samples, d))
    return signs * values