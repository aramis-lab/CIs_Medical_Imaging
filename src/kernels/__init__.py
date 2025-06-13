import numpy as np

from .cosine import *
from .epanechnikov import *
from .exponential import *
from .gaussian import *
from .linear import *
from .tophat import *

def get_kernel(kernel_name):
    kernel_dict = {
        "cosine": cosine_kernel,
        "epanechnikov": epanechnikov_kernel,
        "exponential": exponential_kernel,
        "gaussian": gaussian_kernel,
        "linear": linear_kernel,
        "tophat": tophat_kernel
    }
    return kernel_dict.get(kernel_name, None)

def get_sampling_function(kernel_name):
    sampling_function_dict = {
        "cosine": sample_cosine_multivariate,
        "epanechnikov": sample_epanechnikov_multivariate,
        "exponential": sample_exponential_multivariate,
        "gaussian": sample_gaussian_multivariate,
        "linear": sample_linear_multivariate,
        "tophat": sample_tophat_multivariate
    }
    return sampling_function_dict.get(kernel_name, None)

def preprocess_kernel_inputs(x,y,h):
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.asarray(h)

    if x.ndim==0:
        x = np.array([x])
    if y.ndim==0:
        y = np.array([y])
    if h.ndim==0:
        h = np.array([h])
    if x.ndim==1:
        x = x.reshape(-1, 1)
    if y.ndim==1:
        y = y.reshape(-1, 1)
    if h.ndim==1:
        h = h.reshape(-1, 1)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of dimensions.")
    if x.shape[0] != h.shape[0]:
        raise ValueError("x and h must have the same number of data points.")
    
    return x, y, h

def compute_scaled_differences(x, y, h):
    x, y, h = preprocess_kernel_inputs(x, y, h)
    x_exp = x[:, None, :]       # (n, 1, d)
    y_exp = y[None, :, :]       # (1, m, d)
    h_exp = h[:, None, None]    # (n, 1, 1)
    u = (x_exp - y_exp) / h_exp # (n, m, d)
    return u
