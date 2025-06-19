import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, fbeta_score
)
from sklearn.preprocessing import label_binarize

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_to_predictions(logits):
    """Convert logits to predicted class indices using softmax."""
    return np.argmax(softmax(logits, axis=-1), axis=-1)

def apply_metric_batchwise(metric_fn, y_true, y_pred, **kwargs):
    if y_true.ndim == 2:  # batch
        return np.array([metric_fn(y_t, y_p, **kwargs) for y_t, y_p in zip(y_true, y_pred)])
    else:
        return metric_fn(y_true, y_pred, **kwargs)

def accuracy(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)
    return apply_metric_batchwise(lambda yt, yp: np.mean(yt == yp), y_true, y_pred)

def precision(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    def _precision(yt, yp):
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return apply_metric_batchwise(_precision, y_true, y_pred)

def recall(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    def _recall(yt, yp):
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return apply_metric_batchwise(_recall, y_true, y_pred)

def f1(y_true, y_pred, average="micro"):
    def _f1(yt, yp):
        p = precision(yt, yp)
        r = recall(yt, yp)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return apply_metric_batchwise(_f1, y_true, y_pred)

def fbeta(y_true, y_pred, beta=1.0, average="micro"):
    def _fbeta(yt, yp):
        p = precision(yt, yp)
        r = recall(yt, yp)
        beta2 = beta ** 2
        denom = beta2 * p + r
        return (1 + beta2) * p * r / denom if denom > 0 else 0.0

    return apply_metric_batchwise(_fbeta, y_true, y_pred)

def npv(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    def _npv(yt, yp):
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return apply_metric_batchwise(_npv, y_true, y_pred)

def sensitivity(y_true, y_pred, average="micro"):
    return recall(y_true, y_pred, average)

def specificity(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    def _specificity(yt, yp):
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return apply_metric_batchwise(_specificity, y_true, y_pred)

def balanced_accuracy(y_true, y_pred, average="micro"):
    sens = sensitivity(y_true, y_pred, average)
    spec = specificity(y_true, y_pred, average)
    return (sens + spec) / 2

def mcc(y_true, y_pred, average="micro"):
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    def _mcc(yt, yp):
        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0

    return apply_metric_batchwise(_mcc, y_true, y_pred)

def ap(y_true, y_pred, average="macro"):
    y_score = softmax(y_pred, axis=-1)

    def _ap(yt, ys):
        yt_bin = label_binarize(yt, classes=np.unique(yt))
        return average_precision_score(yt_bin, ys, average=average)

    return apply_metric_batchwise(_ap, y_true, y_score)

def auroc(y_true, y_pred, average="macro"):
    y_score = softmax(y_pred, axis=-1)

    def _auroc(yt, ys):
        yt_bin = label_binarize(yt, classes=np.unique(yt))
        return roc_auc_score(yt_bin, ys, average=average, multi_class='ovr')

    return apply_metric_batchwise(_auroc, y_true, y_score)

def get_metric(metric):
    metric_dict = {
        "accuracy": accuracy,
        "npv": npv,
        "ppv": precision,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1,
        "fbeta_score": fbeta,
        "mcc": mcc,
        "ap": ap,
        "auroc": auroc,
        "auc": auroc,
    }
    return metric_dict.get(metric, None)