import numpy as np
from scipy.special import gamma
from .kernels_preprocessing_utils import compute_scaled_differences, preprocess_kernel_inputs

def tophat_kernel(x, y, h):
    x,y,h = preprocess_kernel_inputs(x, y, h)
    u = compute_scaled_differences(x, y, h)
    d = u.shape[-1]
    norms = np.linalg.norm(u, axis=-1)
    admissible = norms <= 1
    c_d = gamma((d + 2) / 2) / np.pi**(d / 2)
    return 1/ h**d / c_d * admissible

def sample_tophat_multivariate(n_samples=1, d=1):
    samples = []
    while len(samples) < n_samples:
        z = np.random.uniform(-1, 1, d)
        norm = np.linalg.norm(z)
        if norm <= 1:
            samples.append(z)
    return np.array(samples)