import numpy as np

def get_bounds(metric):
    if metric.lower() in ["dsc", "nsd", "boundary iou", "iou", "cldice", "accuracy", "auc", "auroc", "ap", "balanced accuracy", "f1 score", "npv", "ppv", "sensitivity", "specificity",
                        "precision", "recall"]:
        return (0,1)
    elif metric.lower() in ["mcc"]:
        return (-1, 1)
    elif metric.lower() in ["hd", "hd_95", "hd perc", "assd", "masd"]:
        return (0, np.inf)
    
def is_continuous(metric):
    continuous_metrics = ["dsc", "nsd", "boundary iou", "iou", "cldice", "accuracy", "auc", "auroc", "ap", "balanced accuracy", "f1 score", "npv", "ppv", "sensitivity", 
                          "specificity", "precision", "recall", "mcc"]
    return metric.lower() in continuous_metrics