import numpy as np

def exponential_kernel(x, y, h):
    if len(np.asarray(x))>0:
        d = x.shape[0]
    else:
        d = 1
    return np.exp(-np.linalg.norm((x - y) / h, 1))/(2*h)**d

def sample_exponential_multivariate(n_samples = 1, d = 1):
    signs = np.random.choice([-1, 1], size=(n_samples, d))
    values = np.random.exponential(scale=1.0, size=(n_samples, d))
    return signs * values