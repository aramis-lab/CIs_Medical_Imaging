import numpy as np
from src.kernels import get_sampling_function

# Weighted KDE estimation for 1D data with adaptive bandwidth
def weighted_kde(data: np.ndarray, x_points: np.ndarray, dist_to_bounds: np.ndarray, kernel=None, alphas=None):

    n = len(data)
    bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)

    if alphas is None:
        alphas = np.ones(n)

    bandwidths = bandwidth * alphas
    bandwidths = np.min([bandwidths, dist_to_bounds], axis=0)

    density = np.zeros_like(x_points)

    for i in range(n):
        if bandwidths[i]>0 and kernel is not None:
            current_density = kernel(x_points, data[i], bandwidths[i])

            # Ensure total density integrates properly
            density += current_density
        
        else : 
            idx = np.searchsorted(x_points, data[i])
            idx = np.clip(idx, 0, len(x_points) - 1)
            density[idx] += 1 / (x_points[1] - x_points[0])

    return density / n

# Sampling from KDE with adaptive bandwidth
def sample_weighted_kde(y, x, n_samples):
    """Samples from KDE with adaptive bandwidth."""
    cdf = np.cumsum(y) / np.sum(y)
    values = np.random.rand(n_samples)

    indices = np.searchsorted(cdf, values)
    inv_cdf = x[indices]

    inv_cdf = np.clip(inv_cdf, 0, 1)
    
    return inv_cdf

# Sample from multivariate KDE
def sample_weighted_kde_multivariate(data, labels, kernel_name, n_samples, alphas=None):
    n, d = data.shape

    # Select indices with replacement
    indices = np.round(np.random.rand(n_samples) * (n - 1)).astype(int)

    # Covariance and bandwidth
    covariance = np.cov(data, rowvar=False)
    factor = 1.06 * n ** (-1.0 / (d + 4))
    bandwidth_matrix = factor * covariance

    # Ensure PSD (positive semidefinite)
    eigvals, eigvecs = np.linalg.eigh(bandwidth_matrix)
    eigvals[eigvals < 1e-10] = 1e-10  # Replace small/negative eigenvalues

    # Square root of safe_bandwidth
    bandwidth_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Sampling
    if alphas is None:
        alphas = np.ones(n)
    samples = data[indices]
    weights = alphas[indices]
    sampled_labels = labels[indices]

    sampling_function = get_sampling_function(kernel_name)
    norm_samples = sampling_function(n_samples, d)

    weighted_samples = samples + (norm_samples @ bandwidth_sqrt.T) * weights[:, np.newaxis]
    return weighted_samples, sampled_labels