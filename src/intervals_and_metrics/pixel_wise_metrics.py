import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def label_binarize_vectorized(y, n_classes): # Vectorized version of label_binarize
    """Binarize labels in a vectorized way."""

    y = np.asarray(y)
    if y.ndim < 2:
        raise ValueError("y must have more than 2 dimensions")
    n_classes = int(n_classes)
    shape = y.shape + (n_classes,)
    y_onehot = np.zeros(shape, dtype=np.int32)
    # Create indices for all axes except the last (class axis)
    idx = np.indices(y.shape)
    y_onehot[(*idx, y)] = 1

    return y_onehot

def format_data(y_true, y_pred):
    """Puts y_true and y_pred in the format (batch_size, n_samples, n_classes) for metric calculations. If n_classes is 2, it will be (batch_size, n_samples, 1)."""

    n_classes = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=-1) # Convert to class labels because y_pred is (..., n_samples, n_classes)

    if y_true.ndim == 1:  # Single sample
        y_true = y_true[None, :]
        y_pred = y_pred[None, :]
    
    y_true = label_binarize_vectorized(y_true.astype(np.int32), n_classes)  # Convert to one-hot encoding
    y_pred = label_binarize_vectorized(y_pred.astype(np.int32), n_classes)  # Convert to one-hot encoding
    
    return y_true, y_pred

def accuracy(y_true, y_pred, average=None):
    
    y_pred = np.argmax(y_pred, axis=-1)

    if y_true.ndim == 1:  # Single sample
        y_true = y_true[None, :]
        y_pred = y_pred[None, :]

    return np.count_nonzero(y_pred==y_true, axis=-1) / y_pred.shape[-1]

def precision(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tp = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 1), axis=-2)
    fp = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 1), axis=-2)

    if average == "micro":
        tp = np.sum(tp, axis=-1)
        fp = np.sum(fp, axis=-1)
        denom = tp + fp
        prec = tp / denom
        return np.where(denom > 0, prec, 0.0)

    class_metrics = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    return np.mean(class_metrics, axis=-1)

def recall(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tp = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 1), axis=-2)
    fn = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 0), axis=-2)

    if average == "micro":
        tp = np.sum(tp, axis=-1)
        fn = np.sum(fn, axis=-1)
        denom = tp + fn
        rec = tp / denom
        return np.where(denom > 0, rec, 0.0)

    class_metrics = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    return np.mean(class_metrics, axis=-1)

def f1(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tp = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 1), axis=-2)
    fp = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 1), axis=-2)
    fn = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 0), axis=-2)

    if average == "micro":
        tp = np.sum(tp, axis=-1)
        fp = np.sum(fp, axis=-1)
        fn = np.sum(fn, axis=-1)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        denom = p + r
        f1_score = 2 * p * r / denom
        return np.where(denom > 0, f1_score, 0.0)

    p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = p + r
    class_metrics = np.where(denom > 0, 2 * p * r / denom, 0.0)
    return np.mean(class_metrics, axis=-1)

def fbeta(y_true, y_pred, beta=1.0, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)
    beta2 = beta ** 2

    tp = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 1), axis=-2)
    fp = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 1), axis=-2)
    fn = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 0), axis=-2)

    if average == "micro":
        tp = np.sum(tp, axis=-1)
        fp = np.sum(fp, axis=-1)
        fn = np.sum(fn, axis=-1)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        denom = beta2 * p + r
        fbeta_score = (1 + beta2) * p * r / denom
        return np.where(denom > 0, fbeta_score, 0.0)

    p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = beta2 * p + r
    class_metrics = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)
    return np.mean(class_metrics, axis=-1)

def npv(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tn = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 0), axis=-2)
    fn = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 0), axis=-2)

    if average == "micro":
        tn = np.sum(tn, axis=-1)
        fn = np.sum(fn, axis=-1)
        denom = tn + fn
        npv_score = tn / denom
        return np.where(denom > 0, npv_score, 0.0)

    class_metrics = np.where((tn + fn) > 0, tn / (tn + fn), 0.0)
    return np.mean(class_metrics, axis=-1)

def sensitivity(y_true, y_pred, average="micro"):
    return recall(y_true, y_pred, average)

def specificity(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tn = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 0), axis=-2)
    fp = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 1), axis=-2)

    if average == "micro":
        tn = np.sum(tn, axis=-1)
        fp = np.sum(fp, axis=-1)
        denom = tn + fp
        spec = tn / denom
        return np.where(denom > 0, spec, 0.0)

    class_metrics = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return np.mean(class_metrics, axis=-1)

def balanced_accuracy(y_true, y_pred, average="micro"):
    sens = sensitivity(y_true, y_pred, average)
    spec = specificity(y_true, y_pred, average)
    return (sens + spec) / 2

def mcc(y_true, y_pred, average="micro"):
    y_true, y_pred = format_data(y_true, y_pred)

    tp = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 1), axis=-2)
    tn = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 0), axis=-2)
    fp = np.count_nonzero(np.bitwise_and(y_true == 0, y_pred == 1), axis=-2)
    fn = np.count_nonzero(np.bitwise_and(y_true == 1, y_pred == 0), axis=-2)

    if average == "micro":
        tp = np.sum(tp, axis=-1)
        tn = np.sum(tn, axis=-1)
        fp = np.sum(fp, axis=-1)
        fn = np.sum(fn, axis=-1)
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_score = numerator / denominator
        return np.where(denominator > 0, mcc_score, 0.0)

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    class_metrics = np.where(denominator > 0, numerator / denominator, 0.0)
    return np.mean(class_metrics, axis=-1)

def ap(y_true, y_pred, average="macro"):
    # y_pred is expected to be logits or probabilities (batch_size, n_samples, n_classes)
    # y_true is expected to be (batch_size, n_samples) or (batch_size, n_samples, n_classes)
    if y_pred.ndim == 2:
        y_pred = y_pred[None, ...]
    if y_true.ndim == 1:
        y_true = y_true[None, ...]
    if y_true.ndim == 2:
        y_true = label_binarize_vectorized(y_true, y_pred.shape[-1])
    y_score = softmax(y_pred, axis=-1)
    batch_size = y_true.shape[0]
    scores = []
    for i in range(batch_size):
        try:
            score = average_precision_score(y_true[i], y_score[i], average=average)
        except Exception:
            score = 0.0
        scores.append(score)
    return np.array(scores)

def auroc(y_true, y_pred, average="macro"):
    if y_pred.ndim == 2:
        y_pred = y_pred[None, ...]
    if y_true.ndim == 1:
        y_true = y_true[None, ...]
    if y_true.ndim == 2:
        y_true = label_binarize_vectorized(y_true, y_pred.shape[-1])
    y_score = softmax(y_pred, axis=-1)
    batch_size = y_true.shape[0]
    scores = []
    for i in range(batch_size):
        try:
            score = roc_auc_score(y_true[i], y_score[i], average=average, multi_class='ovr')
        except Exception:
            score = 0.0
        scores.append(score)
    return np.array(scores)

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