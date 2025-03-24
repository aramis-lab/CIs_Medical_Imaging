import numpy as np
from kernels import get_kernel
from scipy.integrate import trapezoid

# Weighted KDE estimation
def weighted_kde(data: np.ndarray, x_points: np.ndarray, alphas=None, kernel_name="epanechnikov") -> np.ndarray:
    n = len(data)
    bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)

    if alphas is None:
        alphas = np.ones(n)
    
    bandwidths = bandwidth * alphas

    kernel = get_kernel(kernel_name)

    area_values = []
    for i in range(n):
        density = kernel(x_points, data[i], bandwidths[i])
        area_values.append(trapezoid(density, x_points, axis=0))

    density = np.zeros_like(x_points)
    for i in range(n):
        density += kernel(x_points, data[i], bandwidths[i]) / area_values[i] / n

    return density

# Sampling from KDE with adaptive bandwidth
def sample_weighted_kde(y, x, n_samples):
    """Samples from KDE with adaptive bandwidth."""
    cdf = np.cumsum(y) / np.sum(y)
    values = np.random.rand(n_samples)

    indices = np.searchsorted(cdf, values)
    inv_cdf = x[indices]
    
    return inv_cdf

