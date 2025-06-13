import numpy as np
from scipy.special import gamma

def tophat_kernel(x, y, h):
    if len(np.asarray(x))>0:
        d = x.shape[0]
    else:
        d = 1
    admissible = np.linalg.norm(x - y) <= h
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