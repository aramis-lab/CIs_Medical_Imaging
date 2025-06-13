import numpy as np
from scipy.special import gamma

from . import compute_scaled_differences

def cosine_kernel(x, y, h):
    u = compute_scaled_differences(x, y, h)
    d = u.shape[-1]
    norms = np.linalg.norm(u, axis=-1)
    admissible = norms <= 1
    normalization_constant = gamma(d / 2 + 1) / np.pi**(d / 2) * np.pi / 4
    return admissible * normalization_constant / h**d * np.cos(np.pi/2 * norms)

def sample_cosine_multivariate(n_samples=1, d=1):
    samples = []
    while len(samples) < n_samples:
        z = np.random.uniform(-1, 1, d)
        norm = np.linalg.norm(z)
        if norm <= 1 and np.random.rand() < (np.pi / 4) * np.cos((np.pi / 2) * norm):
            samples.append(z)
    return np.array(samples)