import numpy as np
from ..kernels import get_kernel

# Weighted KDE estimation
def weighted_kde(data: np.ndarray, x_points: np.ndarray, dist_to_bounds: np.ndarray, alphas=None, kernel_name="epanechnikov"):
    n = len(data)
    bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)

    if alphas is None:
        alphas = np.ones(n)

    bandwidths = bandwidth * alphas
    bandwidths = np.min([bandwidths, dist_to_bounds], axis=0)

    kernel = get_kernel(kernel_name)  # Ensure this function returns a valid kernel

    density = np.zeros_like(x_points)

    for i in range(n):
        if bandwidths[i]>0:
            current_density = kernel(x_points, data[i], bandwidths[i])

            # Ensure total density integrates properly
            density += current_density
        
        else : 
            idx = -round(data[i])
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

