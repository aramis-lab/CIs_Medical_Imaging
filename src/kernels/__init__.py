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