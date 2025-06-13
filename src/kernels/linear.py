import numpy as np
from scipy.special import gamma
from . import compute_scaled_differences

def linear_kernel(x, y, h):
    u = compute_scaled_differences(x, y, h)
    d = x.shape[1]
    norms = np.linalg.norm(u, axis=-1)
    admissible = norms <= 1
    c_d = gamma((d + 2) / 2) / np.pi**(d / 2)
    normalization_constant = 2**d * gamma(d + 1) / c_d
    return (1 - norms) * admissible / h**d * normalization_constant

def sample_linear_multivariate(n_samples=1, d=1):
    directions = np.random.randn(n_samples, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Step 2: Sample radii from the radial distribution
    # PDF: f(r) ∝ r^{d-1} (1 - r), for r in [0,1]
    u = np.random.rand(n_samples)
    radii = 1 - (1 - u) ** (1 / (d + 1))  # Inverse CDF method

    # Step 3: Multiply radius by direction
    samples = directions * radii[:, None]
    return samples