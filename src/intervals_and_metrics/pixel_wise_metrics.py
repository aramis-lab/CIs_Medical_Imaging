import numpy as np
from scipy.stats import rankdata

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def label_binarize_vectorized(y, n_classes): # Vectorized version of label_binarize
    """Binarize labels in a vectorized way."""

    y = np.asarray(y)
    if y.ndim == 0:
        raise ValueError("y must be at least a vector.")
    n_classes = int(n_classes)
    shape = y.shape + (n_classes,)
    y_onehot = np.zeros(shape, dtype=np.int32)
    # Create indices for all axes except the last (class axis)
    idx = np.indices(y.shape)
    y_onehot[(*idx, y)] = 1

    return y_onehot

def accuracy(correct_pred, average=None):
    
    return np.count_nonzero(correct_pred, axis=(-1,-2)) / correct_pred.shape[-2]

def precision(tp, fp, average="micro"):

    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        denom = tp + fp
        prec = tp / denom
        return np.where(denom > 0, prec, 0.0)
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)

    class_prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    return np.mean(class_prec, axis=-1)

def recall(tp, fn, average="micro"):

    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        denom = tp + fn
        rec = tp / denom
        return np.where(denom > 0, rec, 0.0)

    tp = np.count_nonzero(tp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)
    class_recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    return np.mean(class_recall, axis=-1)

def f1(tp, fp, fn, average="micro"):

    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        denom = p + r
        f1_score = 2 * p * r / denom
        return np.where(denom > 0, f1_score, 0.0)
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)

    p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = p + r
    class_f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    return np.mean(class_f1, axis=-1)

def fbeta(tp, fp, fn, beta=1.0, average="micro"):
    beta2 = beta ** 2

    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        denom = beta2 * p + r
        fbeta_score = (1 + beta2) * p * r / denom
        return np.where(denom > 0, fbeta_score, 0.0)
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)

    p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    denom = beta2 * p + r
    class_fbeta = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)
    return np.mean(class_fbeta, axis=-1)

def npv(tn, fn, average="micro"):

    if average == "micro":
        tn = np.count_nonzero(tn, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        denom = tn + fn
        npv_score = tn / denom
        return np.where(denom > 0, npv_score, 0.0)
    
    tn = np.count_nonzero(tn, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)

    class_npv = np.where((tn + fn) > 0, tn / (tn + fn), 0.0)
    return np.mean(class_npv, axis=-1)

def sensitivity(tp, fn, average="micro"):
    return recall(tp, fn, average)

def specificity(tn, fp, average="micro"):

    if average == "micro":
        tn = np.count_nonzero(tn, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        denom = tn + fp
        spec = tn / denom
        return np.where(denom > 0, spec, 0.0)
    
    tn = np.count_nonzero(tn, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)

    class_spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
    return np.mean(class_spec, axis=-1)

def balanced_accuracy(tp, fp, tn, fn, average=None):
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)
    tn = np.count_nonzero(tn, axis=-2)
    
    class_bal_acc = np.where((tp + fn) > 0, tp/(tp+fn), 0.0)
    
    return np.mean(class_bal_acc, axis=-1)

def mcc(tp, fp, fn, average=None):
    
    N = tp.shape[-2]
    S = np.count_nonzero(tp, axis=(-2, -1))
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)

    T = (tp + fn)/N
    P = (tp + fp)/N

    numerator = S/N - np.sum(T*P, axis=-1)
    denom = np.sqrt((1 - np.sum(P**2, axis=-1)) * (1 - np.sum(T**2, axis=-1)))
    return np.where(denom>0, numerator/denom, 0.0)


def ap(y_score, y_true_bin, average="micro"):
    """
    Compute average precision (AP) for each sample and label.
    y_score : scores, shape (..., N, D)
    y_true_bin : one_hot encoding of labels, shape (..., N, D)
    average : "micro" or "macro"

    Returns:
      - micro: AP with flattened labels, shape (...)
      - macro: AP averaged over classes, shape (...)
    """

    def _ap_binary(scores, targets, axis=-1):
        # Sort scores descending along axis
        sorted_indices = np.argsort(-scores, axis=axis)
        sorted_targets = np.take_along_axis(targets, sorted_indices, axis=axis)

        # Cumulative sum of true labels: precision numerator
        cumsum = np.cumsum(sorted_targets, axis=axis)

        # Create a broadcasted index for position (1-based)
        idx = np.arange(1, scores.shape[axis]+1)
        if axis==-1:
            idx = np.expand_dims(idx, tuple(range(sorted_targets.ndim - 1)))
        elif axis ==-2:
            idx = np.expand_dims(idx, tuple(range(sorted_targets.ndim - 2)) + (sorted_targets.ndim-1,))

        # Precision at each threshold
        precision = cumsum / idx

        # Multiply precision by sorted_targets to zero-out non-relevant positions
        precision_at_hits = precision * sorted_targets

        # Sum precision at relevant positions and normalize by number of positives
        total_positives = np.sum(targets, axis=axis)
        ap = np.sum(precision_at_hits, axis=axis) / np.clip(total_positives, a_min=1, a_max=None)

        # If no positives, set AP to NaN (optional: could also be 0)
        ap = np.where(total_positives > 0, ap, np.nan)
        return ap

    if average == "micro":
        flat_scores = y_score.reshape(*y_score.shape[:-2], -1)
        flat_targets = y_true_bin.reshape(*y_true_bin.shape[:-2], -1)
        return _ap_binary(flat_scores, flat_targets, axis=-1)

    elif average == "macro":
        class_ap = _ap_binary(y_score, y_true_bin, axis=-2)
        return np.mean(class_ap, axis=-1)

    else:
        raise ValueError(f"Unsupported average method '{average}' for AP calculation.")

def auroc(y_score, y_true_bin, average="micro"):
    """
    Compute AUC for each sample and label.
    y_score : scores, shape (...,N,D)
    y_true_bin : one_hot encoding of labels, shape (...,N,D)
    """

    def _auroc_binary(scores, targets, axis=-1):
        ranks = rankdata(scores, axis=axis)
        rank_sum = np.sum(ranks*targets, axis=axis)
        P = np.count_nonzero(targets, axis=axis)
        N = targets.shape[axis]
        return np.where(P*(N-P)>0, (rank_sum - P * (P + 1) / 2)/(P*(N-P)), np.nan)

    if average == "micro":
        y_score = y_score.reshape(*y_score.shape[:-2], -1)
        y_true_bin = y_true_bin.reshape(*y_true_bin.shape[:-2], -1)
        return _auroc_binary(y_score, y_true_bin, axis=-1)
    
    elif average == "macro":
        class_AUCs = _auroc_binary(y_score, y_true_bin, axis=-2)
        return np.mean(class_AUCs, axis=-1)
    
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