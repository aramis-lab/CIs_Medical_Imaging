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

def balanced_accuracy(tp, fp, tn, fn, average="micro"):
    
    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        tn = np.count_nonzero(tn, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        denom_spec = tn + fp
        spec = tn / denom_spec
        denom_sens = tp + fn
        sens = tp / denom_sens
        bal_acc = (spec + sens) / 2
        return np.where((denom_spec > 0) & (denom_sens > 0), bal_acc, 0.0)
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)
    tn = np.count_nonzero(tn, axis=-2)
    
    class_bal_acc = np.where(((tn + fp) > 0) & ((tp + fn) > 0), (tn/(tn+fp) + tp/(tp+fn))/2, 0.0)
    
    return np.mean(class_bal_acc, axis=-1)

def mcc(tp, fp, tn, fn, average="micro"):

    if average == "micro":
        tp = np.count_nonzero(tp, axis=(-2, -1))
        tn = np.count_nonzero(tn, axis=(-2, -1))
        fp = np.count_nonzero(fp, axis=(-2, -1))
        fn = np.count_nonzero(fn, axis=(-2, -1))
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_score = numerator / denominator
        return np.where(denominator > 0, mcc_score, 0.0)
    
    tp = np.count_nonzero(tp, axis=-2)
    fp = np.count_nonzero(fp, axis=-2)
    fn = np.count_nonzero(fn, axis=-2)
    tn = np.count_nonzero(tn, axis=-2)

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    class_mcc = np.where(denominator > 0, numerator / denominator, 0.0)
    return np.mean(class_mcc, axis=-1)

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

    if y_true.ndim == 1:
        y_true = y_true[None, :]

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

def auroc(y_score, y_true_bin, average="micro"):
    """
    Compute AUC for each sample and label.
    y_score : scores, shape (...,N,D)
    y_true_bin : one_hot encoding of labels, shape (...,N,D)
    """

    if average == "micro":
        y_score = y_score.reshape(*y_score.shape[:-2], -1)
        y_true_bin = y_true_bin.reshape(*y_true_bin.shape[:-2], -1)
        ranks = rankdata(y_score, axis=-1)
        rank_sum = np.sum(ranks*y_true_bin, axis=-1)
        n_pos = np.count_nonzero(y_true_bin, axis=-1)
        n_neg = y_true_bin.shape[-1] - n_pos
        return (rank_sum - n_pos * (n_pos + 1) / 2)/(n_pos * n_neg)
    
    elif average == "macro":
        ranks = rankdata(y_score, axis=-2)
        rank_sum = np.sum(ranks*y_true_bin, axis=-2)
        n_pos = np.count_nonzero(y_true_bin, axis=-2)
        n_neg = y_true_bin.shape[-2] - n_pos
        class_AUCs = (rank_sum - n_pos * (n_pos + 1) / 2)/(n_pos * n_neg)
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