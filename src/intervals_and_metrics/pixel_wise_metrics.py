import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_to_predictions(logits):
    """Convert logits to predicted class indices using softmax."""
    return np.argmax(softmax(logits, axis=-1), axis=-1)

def apply_metric_batchwise(metric_fn, y_true, y_pred, **kwargs):
    if y_true.ndim == 3:  # batch
        return np.array([metric_fn(y_t, y_p, **kwargs) for y_t, y_p in zip(y_true, y_pred)])
    else:
        return metric_fn(y_true, y_pred, **kwargs)

def accuracy(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _accuracy(yt, yp, average):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)
        tn = np.sum((yt == 0) & (yp == 0), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)
        
        if average=="micro":
            tp = np.sum(tp)
            fp = np.sum(fp)
            tn = np.sum(tn)
            fn = np.sum(fn)
            return (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn)>0 else 0.0
        
        # Average is 'macro'
        class_metrics = np.where((tp+fp+tn+fn)>0, (tp+tn)/(tp+fp+tn+fn), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_accuracy, y_true, y_pred, average=average)

def precision(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _precision(yt, yp, average):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)
        
        if average=="micro":
            tp = np.sum(tp)
            fp = np.sum(fp)
            return tp/(tp+fp) if (tp+fp)>0 else 0.0
        
        # Average is 'macro'
        class_metrics = np.where((tp+fp)>0, tp/(tp+fp), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_precision, y_true, y_pred, average=average)

def recall(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _recall(yt, yp, average):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)

        if average == "micro":
            tp = np.sum(tp)
            fn = np.sum(fn)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        class_metrics = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_recall, y_true, y_pred, average=average)

def f1(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _f1(yt, yp, average):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)

        if average == "micro":
            tp = np.sum(tp)
            fp = np.sum(fp)
            fn = np.sum(fn)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        class_metrics = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_f1, y_true, y_pred, average=average)

def fbeta(y_true, y_pred, beta=1.0, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _fbeta(yt, yp, average, beta):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)
        beta2 = beta ** 2

        if average == "micro":
            tp = np.sum(tp)
            fp = np.sum(fp)
            fn = np.sum(fn)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = beta2 * p + r
            return (1 + beta2) * p * r / denom if denom > 0 else 0.0

        p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = beta2 * p + r
        class_metrics = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_fbeta, y_true, y_pred, average=average, beta=beta)

def npv(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _npv(yt, yp, average):
        tn = np.sum((yt == 0) & (yp == 0), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)

        if average == "micro":
            tn = np.sum(tn)
            fn = np.sum(fn)
            return tn / (tn + fn) if (tn + fn) > 0 else 0.0

        class_metrics = np.where((tn + fn) > 0, tn / (tn + fn), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_npv, y_true, y_pred, average=average)

def sensitivity(y_true, y_pred, average="micro"):
    return recall(y_true, y_pred, average)

def specificity(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _specificity(yt, yp, average):
        tn = np.sum((yt == 0) & (yp == 0), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)

        if average == "micro":
            tn = np.sum(tn)
            fp = np.sum(fp)
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0

        class_metrics = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_specificity, y_true, y_pred, average=average)

def balanced_accuracy(y_true, y_pred, average="micro"):
    sens = sensitivity(y_true, y_pred, average)
    spec = specificity(y_true, y_pred, average)
    return (sens + spec) / 2

def mcc(y_true, y_pred, average="micro"):
    classes = np.arange(y_pred.shape[-1])
    if y_pred.ndim > y_true.ndim:
        y_pred = softmax_to_predictions(y_pred)

    y_pred = label_binarize(y_pred, classes=classes)
    y_true = label_binarize(y_true, classes=classes)

    def _mcc(yt, yp, average):
        tp = np.sum((yt == 1) & (yp == 1), axis=1)
        tn = np.sum((yt == 0) & (yp == 0), axis=1)
        fp = np.sum((yt == 0) & (yp == 1), axis=1)
        fn = np.sum((yt == 1) & (yp == 0), axis=1)

        if average == "micro":
            tp = np.sum(tp)
            tn = np.sum(tn)
            fp = np.sum(fp)
            fn = np.sum(fn)
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return numerator / denominator if denominator > 0 else 0.0

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        class_metrics = np.where(denominator > 0, numerator / denominator, 0.0)
        return np.mean(class_metrics)

    return apply_metric_batchwise(_mcc, y_true, y_pred, average=average)

def ap(y_true, y_pred, average="macro"):
    n_classes = y_pred.shape[-1]
    y_score = softmax(y_pred, axis=-1)

    def _ap(yt, ys):
        yt_bin = label_binarize(yt, classes=np.arange(n_classes))
        return average_precision_score(yt_bin, ys, average=average)

    return apply_metric_batchwise(_ap, y_true, y_score)

def auroc(y_true, y_pred, average="macro"):
    n_classes = y_pred.shape[-1]
    y_score = softmax(y_pred, axis=-1)

    def _auroc(yt, ys):
        yt_bin = label_binarize(yt, classes=np.arange(n_classes))
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