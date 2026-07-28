"""Lookup tables mapping (metric, grouping) pairs to the confidence-interval
methods that are statistically valid for them.

Not every CI method applies to every metric: parametric normal/beta intervals
only make sense for proportions computed under micro-averaging, DeLong-style
AUC intervals only apply to micro-averaged AUC, and so on. These two functions
are the single source of truth for that compatibility matrix, used both by
the sweep pipeline (``build_task_list.py``) and by the single-experiment
runner to filter the ``ci_methods`` requested in a config down to the ones
that are actually authorized.
"""


def get_authorized_methods_segmentation(summary_stat, metric):
    """Return the CI methods authorized for a segmentation metric/summary-stat pair.

    Parameters
    ----------
    summary_stat : str
        The summary statistic being estimated, one of ``"mean"``, ``"median"``,
        ``"iqr_length"``, ``"std"``, or ``"trimmed_mean"``.
    metric : str
        The segmentation metric, one of ``"nsd"``, ``"boundary_iou"``, ``"cldice"``,
        ``"dsc"``, ``"iou"``, ``"assd"``, ``"hd"``, ``"hd_perc"``, or ``"masd"``.

    Returns
    -------
    set of str
        The set of CI method names that are valid for this
        (metric, summary_stat) combination.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported segmentation metrics.
    """
    if metric in ["nsd", "boundary_iou", "cldice", "dsc", "iou"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca", "hoeffding", "benett"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}
    elif metric in ["assd", "hd", "hd_perc", "masd"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}

    raise ValueError(f"The following metric : {metric} is not supported")

def get_authorized_methods_classification(metric, average):
    """Return the CI methods authorized for a classification metric/average pair.

    Parameters
    ----------
    metric : str
        The classification metric, one of ``"accuracy"``, ``"npv"``, ``"ppv"``,
        ``"precision"``, ``"recall"``, ``"sensitivity"``, ``"specificity"``,
        ``"ap"``, ``"mcc"``, ``"balanced_accuracy"``, ``"f1_score"``,
        ``"fbeta_score"``, ``"auroc"``, or ``"auc"``.
    average : str
        The averaging strategy, ``"macro"`` or ``"micro"``.

    Returns
    -------
    set of str
        The set of CI method names that are valid for this
        (metric, average) combination. Parametric proportion intervals
        (``agresti_coull``, ``wilson``, ``wald``, ``exact``) and AUC-specific
        methods (``logit_transform``, ``delong``) are only authorized under
        micro-averaging, since only then can the metric be reduced to a
        single binary proportion or a single binary AUC problem.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported classification metrics.
    """
    if metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity"]:
        if average=="micro" or metric=="accuracy":
            return {"percentile", "basic", "bca", "agresti_coull", "wilson", "wald", "exact"}
        else:
            return {"percentile", "basic", "bca"}
    elif metric in ["ap", "mcc", "balanced_accuracy", "f1_score", "fbeta_score"]:
        return {"percentile", "basic", "bca"}
    elif metric in ["auroc", "auc"]:
        if average=="micro":
            return {"percentile", "basic", "bca", "logit_transform", "delong"}
        else:
            return {"percentile", "basic", "bca"}
    raise ValueError(f"The following metric : {metric} is not supported")