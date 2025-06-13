import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

def softmax_to_predictions(softmax_scores):
    """Convert softmax scores to class predictions."""
    return np.argmax(softmax_scores, axis=1)

def accuracy(y_true, y_pred, average="micro"):
    """Calculate accuracy."""
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(y_true == y_pred)

def npv(y_true, y_pred, average="micro"):
    """Calculate Negative Predictive Value (NPV)."""
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    # Binarize labels for multi-class
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tn = np.sum((y_true == 0) & (y_pred == 0), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    npvs = np.zeros(len(tn))
    for i in range(len(tn)):
        if tn[i] + fn[i]>0:
            npvs = tn[i] / (tn[i]+fn[i])
    return npvs

def ppv(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    ppvs = np.zeros(len(tp))
    for i in range(len(tp)):
        if tp[i] + fp[i]>0:
            ppvs = tp[i] / (tp[i]+fp[i])
    return ppvs

def precision(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    return precision_score(y_true, y_pred, average=average)

def recall(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    return recall_score(y_true, y_pred, average=average)

def sensitivity(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)
    sens = np.zeros(len(tp))
    for i in range(len(tp)):
        if tp[i] + fn[i]>0:
            sens = tp[i] / (tp[i]+fn[i])
    return sens

def specificity(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    tn = np.sum((y_true == 0) & (y_pred == 0), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    spec = np.zeros(len(tn))
    for i in range(len(tn)):
        if tn[i] + fp[i]>0:
            spec = tn[i] / (tn[i]+fp[i])
    return spec

def balanced_accuracy(y_true, y_pred, average="micro"):
    sensitivity_scores = sensitivity(y_true, y_pred, average="micro")
    specificity_scores = specificity(y_true, y_pred, average="micro")
    
    return np.mean(sensitivity_scores + specificity_scores) / 2

def f1(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    return f1_score(y_true, y_pred, average=average)

def fbeta_score(y_true, y_pred, beta=1.0, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
    return f1_score(y_true, y_pred, beta=beta, average=average)

def mcc(y_true, y_pred, average="micro"):
    y_pred = softmax_to_predictions(y_pred) if y_pred.ndim > 1 else y_pred
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