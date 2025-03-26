import numpy as np
import scipy.special as sp

def gamma_kernel(x_points, data, bandwidth_factor=1.0):
    """Gamma Kernel Density Estimation for bounded data in [0,1]."""
    
    # Estimate shape (α) and scale (β) parameters for each point

    shape = (data/bandwidth_factor) +1
    scale = bandwidth_factor
    
    coeff = 1/(scale**shape * sp.gamma(shape))
    return coeff * x_points ** (shape-1) * np.exp(-x_points/scale)