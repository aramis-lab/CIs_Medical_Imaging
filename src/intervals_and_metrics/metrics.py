import numpy as np

def compute_metrics(CIs, true_value):
    coverage = np.mean(((CIs[0] <= true_value) & (CIs[1] >= true_value)))
    width = CIs[1] - CIs[0]
    proportion_oob = ((CIs[0] < 0) * (-CIs[0]) + (CIs[1] > 1) * (CIs[1] - 1)) / width
    return coverage, np.nanmean(proportion_oob), np.nanmean(width)