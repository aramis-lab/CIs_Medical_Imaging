import numpy as np

def gaussian_kernel(x, y, h):
    if len(np.asarray(x))>0:
        d = x.shape[0]
    else:
        d = 1
    constant = 1 / (h * np.sqrt(2 * np.pi))**d
    return constant * np.exp(-0.5*np.linalg.norm((x - y)/h)**2)

def sample_gaussian_multivariate(n_samples=1, d=1):
    return np.random.normal(size=(n_samples, d))