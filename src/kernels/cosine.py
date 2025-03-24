import numpy as np

def cosine_kernel(x, y, h):
    admissible = np.abs(x - y) <= h
    return admissible * np.pi / 4 / h * np.cos(np.pi / 2 / h * (x - y))
