import numpy as np
from ..kernels import get_kernel
from scipy.integrate import trapezoid

# Weighted KDE estimation
def weighted_kde(data: np.ndarray, x_points: np.ndarray, alphas=None, kernel_name="epanechnikov"):
    n = len(data)
    bandwidth = 1.06 * np.std(data) * n ** (-1 / 5)

    if alphas is None:
        alphas = np.ones(n)
    
    bandwidths = bandwidth * alphas

    kernel = get_kernel(kernel_name)  # Ensure this function returns a valid kernel

    density = np.zeros_like(x_points)

    for i in range(n):
        current_density = kernel(x_points, data[i], bandwidths[i])

        # Compute lost mass at the left boundary
        x_lower = np.linspace(min(x_points) - bandwidth, min(x_points), 1000)
        lost_mass_lower = trapezoid(kernel(x_lower, data[i], bandwidths[i]), x_lower)

        # Compute lost mass at the right boundary
        x_upper = np.linspace(max(x_points), max(x_points) + bandwidth, 1000)
        lost_mass_upper = trapezoid(kernel(x_upper, data[i], bandwidths[i]), x_upper)

        # Transpose mass to the nearest boundary
        if lost_mass_lower > 0:
            current_density[0] += lost_mass_lower  # Move to the first point
        if lost_mass_upper > 0:
            current_density[-1] += lost_mass_upper  # Move to the last point

        # Ensure total density integrates properly
        density += current_density

    return density / n

# Sampling from KDE with adaptive bandwidth
def sample_weighted_kde(y, x, n_samples):
    """Samples from KDE with adaptive bandwidth."""
    cdf = np.cumsum(y) / np.sum(y)
    values = np.random.rand(n_samples)

    indices = np.searchsorted(cdf, values)
    inv_cdf = x[indices]
    
    return inv_cdf

