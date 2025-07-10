def get_authorized_methods_segmentation(summary_stat, metric):
    if metric in ["nsd", "boundary_iou", "cldice", "dsc", "iou"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}
    elif metric in ["assd", "hd", "hd_perc", "masd"]:
        if summary_stat in ["mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca"}
        elif summary_stat in ["median", "iqr_length", "std", "trimmed_mean"]:
            return {"percentile", "basic", "bca"}
    
    raise ValueError(f"The following metric : {metric} is not supported")

def get_authorized_methods_classification(metric, average):
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