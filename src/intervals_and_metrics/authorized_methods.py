def get_authorized_methods(summary_stat, metric):
    if metric in ["nsd", "boundary_iou", "cldice", "dsc", "iou"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}
    elif metric in ["assd", "hd", "hd_95", "masd"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}
    elif metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "fbeta_score", "mcc"]:
        return {"percentile", "basic", "bca", "agresti_coull", "wilson", "wald", "param_z", "cloper_pearson", "exact"}
    elif metric in ["ap", "auroc", "auc"]:
        return {"percentile", "basic", "bca", "logit_transform", "empirical_likelihood", "delong", "param_z"}
    raise ValueError("The following metric is not supported")