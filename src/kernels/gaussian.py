import numpy as np
from . import compute_scaled_differences

def gaussian_kernel(x, y, h):
    u = compute_scaled_differences(x, y, h)
    d = x.shape[1]
    norms = np.linalg.norm(u, axis=-1)
    constant = 1 / (h * np.sqrt(2 * np.pi))**d
    return constant * np.exp(-0.5*norms**2)

def sample_gaussian_multivariate(n_samples=1, d=1):
    return np.random.normal(size=(n_samples, d))