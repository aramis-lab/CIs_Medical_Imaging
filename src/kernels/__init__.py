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