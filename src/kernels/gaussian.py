import numpy as np
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs

def gaussian_kernel(x, y, h):
    x,y,h = preprocess_kernel_inputs(x, y, h)
    u = compute_scaled_differences(x, y, h)
    d = u.shape[-1]
    norms = np.linalg.norm(u, axis=-1)
    constant = 1 / (h * np.sqrt(2 * np.pi))**d
    return constant * np.exp(-0.5*norms**2)

def sample_gaussian_multivariate(n_samples=1, d=1):
    return np.random.normal(size=(n_samples, d))