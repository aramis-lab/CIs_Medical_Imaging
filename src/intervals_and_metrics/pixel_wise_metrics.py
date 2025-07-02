import numpy as np

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

def ap(y_true, y_scores, average="micro"):
    """
    Compute average precision for each sample and label.
    y_true: array-like of shape (..., n_samples, n_labels), binary {0,1}
    y_scores: array-like of shape (..., n_samples, n_labels), float
    Returns: array of shape (..., n_labels)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    n_classes = y_scores.shape[-1]

    y_true = label_binarize_vectorized(y_true.astype(np.int32), n_classes)  # Convert to one-hot encoding
    y_scores = softmax(y_scores)
    
    if average == "micro":
        orig_shape = y_true.shape[:-2]

        y_true = y_true.reshape(orig_shape + (-1,))
        y_scores = y_scores.reshape(orig_shape + (-1,))
        desc_order = np.argsort(y_scores, axis=-1)[..., ::-1]

        y_true_sorted = np.take_along_axis(y_true, desc_order, axis=-1)

        tp_cumsum = np.cumsum(y_true_sorted, axis=-1)
        total_positives = np.sum(y_true_sorted, axis=-1, keepdims=True)

        mask = (total_positives>0).squeeze(-1)

        precision = tp_cumsum / (np.arange(1, y_true_sorted.shape[-1] + 1))
        recall = tp_cumsum / total_positives

        recall_diff = np.diff(np.concatenate([np.zeros(recall.shape[:-1]+(1,)), recall], axis=-1), axis=-1)

        ap = np.sum(precision * recall_diff, axis=-1)

        ap[~mask] = 0.0

        return ap


    elif average == "macro":

        desc_order = np.argsort(y_scores, axis=-2)[..., ::-1, :]

        y_true_sorted = np.take_along_axis(y_true, desc_order, axis=-2)
        # Cumulative sum of true positives
        tp_cumsum = np.cumsum(y_true_sorted, axis=-2)
        total_positives = np.sum(y_true_sorted, axis=-2, keepdims=True)
        # Avoid division by zero
        mask = (total_positives > 0).squeeze(-2)

        # Precision and recall
        precision = tp_cumsum / (np.arange(1, y_true_sorted.shape[-2] + 1)[..., None])
        recall = tp_cumsum / total_positives

        recall_shape = np.array(recall.shape)
        recall_shape[-2] = 1
        recall_diff = np.diff(np.concatenate([np.zeros(recall_shape), recall], axis=-2), axis=-2)

        ap = np.sum(precision * recall_diff, axis=-2)

        ap[~mask] = 0.0

        return np.mean(ap, axis=-1)
    
    else:
        raise ValueError(f"The averaging method {average} is not supported for this implementation of average precision score.")

def auroc(y_true, y_scores, average="micro"):
    """
    Compute AUC for each sample and label.
    y_true: array-like of shape (..., n_samples, n_labels), binary {0,1}
    y_scores: array-like of shape (..., n_samples, n_labels), float
    Returns: array of shape (..., n_labels)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_scores.shape[-1]

    y_true = label_binarize_vectorized(y_true.astype(np.int32), n_classes)
    y_scores = softmax(y_scores)

    if average == "micro":
        orig_shape = y_true.shape[:-2]
        y_true = y_true.reshape(orig_shape + (-1,))
        y_scores = y_scores.reshape(orig_shape + (-1,))
        desc_order = np.argsort(y_scores, axis=-1)[..., ::-1]
        y_true_sorted = np.take_along_axis(y_true, desc_order, axis=-1)

        total_positives = np.sum(y_true_sorted, axis=-1, keepdims=True)
        total_negatives = y_true_sorted.shape[-1] - total_positives

        tps = np.cumsum(y_true_sorted, axis=-1)
        fps = np.cumsum(1 - y_true_sorted, axis=-1)

        tpr = tps / np.maximum(total_positives, 1)
        fpr = fps / np.maximum(total_negatives, 1)

        d_fpr = np.diff(np.concatenate([np.zeros(fpr.shape[:-1] + (1,)), fpr], axis=-1), axis=-1)
        auc = np.sum(tpr * d_fpr, axis=-1)
        mask = (total_positives > 0).squeeze(-1) & (total_negatives > 0).squeeze(-1)
        auc[~mask] = 0.0
        return auc

    elif average == "macro":
        desc_order = np.argsort(y_scores, axis=-2)[..., ::-1, :]
        y_true_sorted = np.take_along_axis(y_true, desc_order, axis=-2)

        total_positives = np.sum(y_true_sorted, axis=-2, keepdims=True)
        total_negatives = y_true_sorted.shape[-2] - total_positives

        tps = np.cumsum(y_true_sorted, axis=-2)
        fps = np.cumsum(1 - y_true_sorted, axis=-2)

        tpr = tps / np.maximum(total_positives, 1)
        fpr = fps / np.maximum(total_negatives, 1)

        recall_shape = np.array(fpr.shape)
        recall_shape[-2] = 1
        d_fpr = np.diff(np.concatenate([np.zeros(recall_shape), fpr], axis=-2), axis=-2)
        auc = np.sum(tpr * d_fpr, axis=-2)
        mask = (total_positives > 0).squeeze(-2) & (total_negatives > 0).squeeze(-2)
        auc[~mask] = 0.0
        return np.mean(auc, axis=-1)
    else:
        raise ValueError(f"The averaging method {average} is not supported for this implementation of AUROC.")

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