"""Registry of available kernel functions and their multivariate sampling
counterparts.

Each supported kernel (cosine, Epanechnikov, exponential, Gaussian, linear,
top-hat) is implemented in its own submodule and re-exported here via
wildcard imports.  The two factory functions :func:`get_kernel` and
:func:`get_sampling_function` act as a single point of access: given a
kernel name they return the matching 1-D density evaluator or multivariate
sampler respectively, insulating the rest of the codebase from the concrete
import paths.
"""

from .cosine import *
from .epanechnikov import *
from .exponential import *
from .gaussian import *
from .linear import *
from .tophat import *


def get_kernel(kernel_name):
    """Return the 1-D kernel density function associated with a kernel name.

    Parameters
    ----------
    kernel_name : str
        Name of the desired kernel, one of ``"cosine"``,
        ``"epanechnikov"``, ``"exponential"``, ``"gaussian"``,
        ``"linear"``, or ``"tophat"``.

    Returns
    -------
    callable or None
        The kernel function with signature
        ``kernel(centre, x_points, bandwidth)`` returning an array of
        density contributions, or *None* if *kernel_name* does not match
        any registered kernel.
    """
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
    """Return the multivariate sampling function associated with a kernel name.

    Parameters
    ----------
    kernel_name : str
        Name of the desired kernel, one of ``"cosine"``,
        ``"epanechnikov"``, ``"exponential"``, ``"gaussian"``,
        ``"linear"``, or ``"tophat"``.

    Returns
    -------
    callable or None
        A sampling function with signature ``f(n_samples, d)`` that draws
        *n_samples* noise vectors in *d* dimensions from the specified
        kernel distribution, or *None* if *kernel_name* does not match any
        registered kernel.
    """
    sampling_function_dict = {
        "cosine": sample_cosine_multivariate,
        "epanechnikov": sample_epanechnikov_multivariate,
        "exponential": sample_exponential_multivariate,
        "gaussian": sample_gaussian_multivariate,
        "linear": sample_linear_multivariate,
        "tophat": sample_tophat_multivariate
    }
    return sampling_function_dict.get(kernel_name, None)