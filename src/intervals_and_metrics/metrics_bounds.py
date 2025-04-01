import numpy as np

def get_bounds(metric):
    if metric == "DSC":
        return 0, 1
    elif metric == "NSD":
        return 0, 1
    else:
        raise ValueError(f"Unknown metric: {metric}")