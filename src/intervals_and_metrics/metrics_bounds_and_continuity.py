"""Static metadata about each supported metric: its theoretical bounds, and
whether it takes continuous or discrete values (which determines whether the
segmentation pipeline builds a KDE over the metric's values or resamples
them directly)."""

import numpy as np

def get_bounds(metric):
    """Return the theoretical lower/upper bounds of a metric.

    Parameters
    ----------
    metric : str
        Metric name (case-insensitive). Supported: ``"dsc"``, ``"nsd"``,
        ``"boundary_iou"``, ``"iou"``, ``"cldice"``, ``"accuracy"``, ``"auc"``,
        ``"auroc"``, ``"ap"``, ``"balanced_accuracy"``, ``"f1_score"``,
        ``"npv"``, ``"ppv"``, ``"sensitivity"``, ``"specificity"``,
        ``"precision"``, ``"recall"`` (bounded in ``[0, 1]``); ``"mcc"``
        (bounded in ``[-1, 1]``); ``"hd"``, ``"hd_95"``, ``"hd_perc"``,
        ``"assd"``, ``"masd"`` (bounded in ``[0, inf)``).

    Returns
    -------
    tuple of float
        The ``(lower_bound, upper_bound)`` pair for this metric, or ``None``
        if the metric is not recognized.
    """
    if metric.lower() in ["dsc", "nsd", "boundary_iou", "iou", "cldice", "accuracy", "auc", "auroc", "ap", "balanced_accuracy", "f1_score", "npv", "ppv", "sensitivity", "specificity",
                        "precision", "recall"]:
        return (0,1)
    elif metric.lower() in ["mcc"]:
        return (-1, 1)
    elif metric.lower() in ["hd", "hd_95", "hd_perc", "assd", "masd"]:
        return (0, np.inf)
    
def is_continuous(metric):
    """Return whether a metric takes continuous (rather than discrete) values.

    Parameters
    ----------
    metric : str
        Metric name (case-insensitive).

    Returns
    -------
    bool
        ``True`` if the metric is continuous (e.g. Dice score, Hausdorff
        distance), ``False`` otherwise (e.g. an unrecognized/discrete metric).
        Used to decide whether the segmentation pipeline estimates a KDE
        over the metric's values, or resamples the observed values directly.
    """
    continuous_metrics = ["dsc", "nsd", "boundary_iou", "iou", "cldice", "accuracy", "auc", "auroc", "ap", "balanced_accuracy", "f1_score", "npv", "ppv", "sensitivity", 
                          "specificity", "precision", "recall", "mcc", "masd", "assd"]
    return metric.lower() in continuous_metrics