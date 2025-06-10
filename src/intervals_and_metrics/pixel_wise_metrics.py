import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

def accuracy(y_true, y_pred, average="micro"):
    return np.mean(y_true == y_pred)

def npv(y_true, y_pred, average="micro"):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def ppv(y_true, y_pred, average="micro"):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def precision(y_true, y_pred, average="micro"):
    return precision_score(y_true, y_pred, average=average)

def recall(y_true, y_pred, average="micro"):
    return recall_score(y_true, y_pred, average=average)

def sensitivity(y_true, y_pred, average="micro"):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity(y_true, y_pred, average="micro"):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tn = np.sum((y_true == 0) & (y_pred == 0), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def balanced_accuracy(y_true, y_pred, average="micro"):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    sensitivity_scores = sensitivity(y_true, y_pred, average="micro")
    specificity_scores = specificity(y_true, y_pred, average="micro")
    
    return np.mean(sensitivity_scores + specificity_scores) / 2

def f1(y_true, y_pred, average="micro"):
    return f1_score(y_true, y_pred, average=average)

def fbeta_score(y_true, y_pred, beta=1.0, average="micro"):
    return f1_score(y_true, y_pred, beta=beta, average=average)

def mcc(y_true, y_pred, average="micro"):
    return matthews_corrcoef(y_true, y_pred)

def ap(y_true, y_score, average="micro"):
    y_true = label_binarize(y_true, classes=np.unique(y_true))
    return average_precision_score(y_true, y_score, average=average)

def auroc(y_true, y_score, average="micro"):
    y_true = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_true, y_score, average=average, multi_class='ovr')

def get_metric(metric):

    metric_dict = {
        "accuracy": accuracy,
        "npv": npv,
        "ppv": ppv,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1,
        "fbeta_score": fbeta_score,
        "mcc": mcc,
        "ap" : ap,
        "auroc" : auroc,
        "auc" : auroc,
    }

    return metric_dict.get(metric, None)