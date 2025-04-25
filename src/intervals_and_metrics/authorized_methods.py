def get_authorized_methods(summary_stat, metric):
    if metric in ["nsd", "boundary_iou", "cldice", "dsc", "iou", "nsd"]:
        if summary_stat in ["mean", "trimmed_mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca", "studentized"}
        elif summary_stat in ["median", "iqr_length"]:
            return {"percentile", "basic", "bca", "studentized"}
        elif summary_stat in ["standard_deviation"]:
            return
    elif metric in ["assd", "hd", "95th_hd", "masd"]:
        if summary_stat in ["mean", "trimmed_mean"]:
            return {"param_t", "param_z", "percentile", "basic", "bca", "studentized"}
        elif summary_stat in ["median"]:
            return {"percentile", "basic", "bca", "studentized"}
    elif metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "ba", "f1_score", "mcc"]:
        return {"percentile", "basic", "bca", "studentized", "agresti_coull", "wilson", "wald", "param_z", "cloper_pearson", "exact"}
    elif metric in ["ap", "auroc", "auc"]:
        return {"percentile", "basic", "bca", "studentized", "logit_transform", "empirical_likelihood", "delong", "param_z"}
