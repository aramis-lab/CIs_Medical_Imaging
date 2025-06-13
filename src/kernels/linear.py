import numpy as np
from scipy.special import gamma

def linear_kernel(x, y, h):
    if len(np.asarray(x))>0:
        d = x.shape[0]
    else:
        d = 1
    admissible = np.linalg.norm(x - y) <= h
    c_d = gamma((d + 2) / 2) / np.pi**(d / 2)
    normalization_constant = 2**d * gamma(d + 1) / c_d
    return (1 - np.linalg.norm((x - y) / h)) * admissible / h**d * normalization_constant

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