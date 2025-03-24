import numpy as np

def tophat_kernel(x, y, h):
    admissible = np.abs(x - y) <= h / 2
    return 1/ h * admissible