
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2, rankdata
from scipy.special import ndtr, ndtri
from scipy.optimize import root_scalar
from .pixel_wise_metrics import get_metric, label_binarize_vectorized, auroc
from numba import njit

def compute_CIs_classification(y_true, y_pred, metric, method, average=None, alpha=0.05, stratified=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If y_true is binary and last dimension is 2, collapse to single binary array
    if y_true.ndim > 1 and y_true.shape[-1] == 2:
        # Collapse one-hot or probability to class labels
        # One-hot encoding
        y_true = y_true[..., 0]
        y_pred = y_pred[..., 0]

    if y_true.ndim == 1:  # single sample
        y_true = y_true[None, ...]
        y_pred = y_pred[None, ...]
    elif y_pred.ndim == y_true.ndim + 1:  # shape (B, N, C)
        pass  # already batched
    elif y_pred.ndim == y_true.ndim:  # likely shape (B, N)
        pass
    else:
        raise ValueError("Input dimension mismatch or unsupported format.")

    if metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "fbeta_score", "mcc"]:
        batch_results = CI_accuracy(y_true, y_pred, metric, method, alpha, average, stratified)
    elif metric in ['ap', 'auc', 'auroc']:
        batch_results = CI_AUC(y_true, y_pred, metric, method, alpha, average, stratified)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return np.stack(batch_results, axis=0)

def CI_accuracy(y_true, y_pred, metric, method, alpha, average, stratified):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if method in ["wald", "param_z"]:
        method = "normal"
    elif method in ["cloper_pearson", "exact"]:
        method = "beta"
    
    if method in ["normal","agresti_coull","beta","wilson"] and average=="micro": # If average is micro, we can consider the problem as a binary classification problem
        n_classes = y_pred.shape[-1]
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred_bin = label_binarize_vectorized(y_pred, n_classes)
        y_true_bin = label_binarize_vectorized(y_true, n_classes)

        tp = np.count_nonzero((y_true_bin == 1) & (y_pred_bin == 1), axis=(-2,-1))
        fn = np.count_nonzero((y_true_bin == 1) & (y_pred_bin == 0), axis=(-2,-1))
        tn = np.count_nonzero((y_true_bin == 0) & (y_pred_bin == 0), axis=(-2,-1))
        fp = np.count_nonzero((y_true_bin == 0) & (y_pred_bin == 1), axis=(-2,-1))

        if metric == "accuracy":
            correct_pred = y_pred==y_true
            value = np.count_nonzero(correct_pred, axis=-1)
            total = correct_pred.shape[-1]
        elif metric in ["precision", "ppv"]:
            value = tp
            total = np.where((tp + fp) > 0, tp + fp, 1)
        elif metric in ["recall", "sensitivity"]:
            value = tp
            total = np.where((tp + fn) > 0, tp + fn, 1)
        elif metric == "specificity":
            value = tn
            total = np.where((tn + fp) > 0, tn + fp, 1)
        elif metric == "npv":
            value = tn
            total = np.where((tn + fn) > 0, tn + fn, 1)
        else:
            raise ValueError(f"Unknown metric for parametric methods: {metric}")
        return np.array(proportion_confint(value, total, alpha=alpha, method= method)).T
    elif method in ['percentile', 'basic', 'bca']:
        return stratified_bootstrap_CI(y_true, y_pred, metric_name=metric, average=average, n_bootstrap=9999, alpha=alpha, stratified=stratified, method=method)
    else:
        if average!="micro":
            raise ValueError("Non-bootstrap CI methods are not defined for multi-class if average is not 'micro'.")
        else:
            raise NotImplementedError(f"The following method is not implemented : {method}. Currently, 'percentile', 'basic', 'bca', 'agresti_coull', 'wilson', 'wald', 'normal', 'param_z', 'cloper_pearson' and 'exact' are implemented.")

def CI_AUC(y_true, y_pred, metric, method, alpha, average, stratified):
    if method in ['percentile', 'basic', 'bca']:
        return stratified_bootstrap_CI(y_true, y_pred, metric_name=metric, average=average, n_bootstrap=9999, alpha=alpha, method=method, stratified=stratified)
    elif method in ["delong", "logit_transform", "empirical_likelihood"] and metric in ['auc', 'auroc'] and average=="micro":
        y_true = label_binarize_vectorized(y_true, n_classes=y_pred.shape[-1]) # Shape (batch_size, n_samples, n_classes)
        AUC = auroc(y_pred, y_true)
        y_pred = y_pred.reshape(y_pred.shape[:-2] + (-1,))
        y_true = y_true.reshape(y_true.shape[:-2] + (-1,)) # Shape (batch_size, n_samples*n_classes)
        N = y_true.shape[-1]
        n = np.count_nonzero(y_true, axis=-1)
        m = N-n
        ranks = rankdata(y_pred, axis=-1)
        rank_pos = ranks*y_true
        rank_neg = ranks*(1-y_true)

        bar_R=np.sum(rank_neg, axis=-1)/m
        bar_S=np.sum(rank_pos, axis=-1)/n
        ideal_R = np.zeros_like(rank_neg)
        ideal_S = np.zeros_like(rank_pos)

        for i in range(y_true.shape[0]):
            ideal_R[i, y_true[i] == 0] = np.arange(1, m[i] + 1)
            ideal_S[i, y_true[i] == 1] = np.arange(1, n[i] + 1)
        S_10=(1/((m-1)*n**2))*(np.sum((rank_neg-ideal_R)**2,axis=-1)-m*(bar_R-(m+1)/2)**2)
        S_01=(1/((n-1)*n**2))*(np.sum((rank_pos-ideal_S)**2,axis=-1)-n*(bar_S-(n+1)/2)**2)
        S=np.sqrt((m*S_01+n*S_10)/(m+n))
        if method == "delong": 
            return CI_DL(y_pred, y_true,AUC, m,n)
        elif method == "logit_transform":
            return CI_LT(AUC, m, n, S)
        elif method == "empirical_likelihood":
            intervals = []
            for i in range(y_true.shape[0]):
                Y = y_pred[i][y_true[i]]  # diseased
                X = y_pred[i][1-y_true[i]]
                intervals.append(el_auc_confidence_interval(Y, X, S[i],AUC[i], alpha))
            return np.array(intervals)
        else:
            raise NotImplementedError(f"The following method is not implemented : {method}. Currently, 'percentile', 'basic', 'bca', 'delong', 'logit_transform' and 'empirical_likelihood' are implemented.")
    else:
        raise ValueError("Non-bootstrap CI methods are not defined for multi-class AUC.")

def CI_LT(AUC, m, n, S):
    LL=np.log(AUC/(1-AUC))-1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
    UL=np.log(AUC/(1-AUC))+1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
    LT_low=np.exp(LL)/(1+np.exp(LL))
    LT_high=np.exp(UL)/(1+np.exp(UL))
    LT_low = np.where((AUC==0)|(AUC==1), np.nan, LT_low)
    LT_high = np.where((AUC==0)|(AUC==1), np.nan, LT_high)
    return np.array([LT_low, LT_high]).T

def CI_DL(y_pred, y,AUC, m,n):

    X = y_pred * (1-y)
    Y = y_pred * y

    binary_comp = 1 - (Y[...,None,:] < X[...,None])
    D10 = (np.count_nonzero(binary_comp, axis=-1) - m[...,None])/n[...,None]
    D01 = (np.count_nonzero(binary_comp, axis=-2) - n[...,None])/m[...,None]
    
    var=(1/(m*(m-1)))*(np.sum((D10-AUC[...,None])**2,axis=-1)-n*AUC**2) + (1/(n*(n-1)))*(np.sum((D01-AUC[...,None])**2,axis=-1)-m*AUC**2)
    
    DL_low=AUC-1.96*np.sqrt(var)
    DL_high=AUC+1.96*np.sqrt(var)
    return np.array([DL_low, DL_high]).T

def el_auc_confidence_interval(Y, X, S, AUC, alpha=0.05, tol=1e-8):
    """
    Compute empirical likelihood-based confidence interval for AUC.

    Parameters:
    Y : array-like, shape (n,)
        Scores for positive class.
    X : array-like, shape (m,)
        Scores for negative class.
    S : float
        Standard error estimate of AUC.
    AUC : float
        Observed AUC.
    alpha : float, optional
        Significance level (default 0.05).
    tol : float, optional
        Tolerance for root finding and bisection (default 1e-8).

    Returns:
    ci_low, ci_high : float
        Lower and upper confidence limits.
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    n = Y.size
    m = X.size

    # Sort X once and compute empirical CDF values for Y
    X_sorted = np.sort(X)
    # Fhat_X(y) = proportion of X <= y
    idx = np.searchsorted(X_sorted, Y, side='right')
    F = idx / m       # Fhat_X(Y)
    U = 1 - F        # U_hat
    v = F            # rename for consistency

    # Edge cases: all U's are 0 or 1
    sumU = U.sum()
    if sumU == 0:
        return np.array([1.0, 1.0])
    if sumU == n:
        return np.array([0.0, 0.0])

    # Precompute constants
    c_const = m / ((m + n) * n * S * S)
    chi2_crit = chi2.ppf(1 - alpha, df=1)

    def test_stat(delta):
        # Solve mean((v - delta)/(1 + lambda*(v - delta))) = 0 for lambda
        dev = v - delta
        # Bracket for lambda: avoid zeros in denom
        max_dev = np.max(dev)
        # Denominator >0 => lambda > -1/max_dev and lambda < -1/min_dev
        lower = -1.0 / (max_dev + 1e-12)
        upper = 1.0 / (max_dev + 1e-12)

        def eq_fn(lmbda):
            denom = 1.0 + lmbda * dev
            return np.sum(dev / denom)

        sol = root_scalar(eq_fn, bracket=[lower, upper], method='brentq', xtol=tol)
        if not sol.converged:
            return np.inf
        lam = sol.root

        l_val = 2.0 * np.sum(np.log1p(lam * dev))
        # Test statistic
        return c_const * (dev @ dev) * l_val

    # Function to find bound via bisection
    def find_bound(a, b, increasing):
        fa = test_stat(a) - chi2_crit
        fb = test_stat(b) - chi2_crit
        # Ensure root exists
        if fa * fb > 0:
            return a if fa < fb else b
        while b - a > tol:
            mid = 0.5 * (a + b)
            fm = test_stat(mid) - chi2_crit
            if (fm <= 0) == increasing:
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    # Compute confidence bounds
    ci_low = find_bound(0.0, AUC, increasing=True)
    ci_high = find_bound(AUC, 1.0, increasing=False)

    return np.array([ci_low, ci_high])

@njit
def stratified_bootstrap_numba(class_indices, class_sizes, n_bootstrap):
    n_classes = len(class_indices)
    n_total = sum(class_sizes)
    result = np.empty((n_bootstrap, n_total), dtype=np.int32)
    for b in range(n_bootstrap):
        pos = 0
        for c in range(n_classes):
            indices = class_indices[c]
            size = class_sizes[c]
            for i in range(size):
                idx = np.random.randint(0, len(indices))
                result[b, pos] = indices[idx]
                pos += 1
    return result

# Timing wrapper for stratified_bootstrap_CI
def stratified_bootstrap_CI(y_true, y_score, metric_name='auc', average='micro', n_bootstrap=9999, alpha=0.05, method='percentile', stratified=True):

    y_true = np.array(y_true)
    n_classes = y_score.shape[-1]
    batch_size, n_samples = y_true.shape
    classes = np.arange(n_classes)
    y_pred = np.argmax(y_score, axis=-1)

    correct_pred = (y_pred==y_true)[..., None] # To allow bootstrapping metric arguments

    y_true_bin = label_binarize_vectorized(y_true, n_classes)
    y_pred_bin = label_binarize_vectorized(y_pred, n_classes)

    tp = (y_true_bin==1) & (y_pred_bin==1)
    fp = (y_true_bin==0) & (y_pred_bin==1)
    tn = (y_true_bin==0) & (y_pred_bin==0)
    fn = (y_true_bin==1) & (y_pred_bin==0)

    metric_arguments = {"accuracy": ["correct_pred"],
                        "precision" : ["tp", "fp"],
                        "recall" : ["tp", "fn"],
                        "f1_score" : ["tp", "fp", "fn"],
                        "fbeta_score" : ["tp", "fp", "fn"],
                        "npv" : ["tn", "fn"],
                        "ppv" : ["tp", "fp"],
                        "sensitivity" : ["tp", "fn"],
                        "specificity" : ["tn", "fp"],
                        "balanced_accuracy" : ["tp", "fp", "tn", "fn"],
                        "mcc" : ["tp", "fp", "fn"],
                        "auroc" : ["y_score", "y_true_bin"],
                        "auc" : ["y_score", "y_true_bin"],
                        "ap" : ["y_score", "y_true_bin"]
    }

    metric = get_metric(metric_name)
    original_arguments = {a : locals()[a] for a in metric_arguments[metric_name]}
    original_stat = metric(average=average, **original_arguments)

    bootstrapped_arguments = {a : [] for a in metric_arguments[metric_name]}

    for i in range(len(y_true)):
        sample = y_true[i]

        if stratified:
            class_indices = [np.flatnonzero(sample == c) for c in classes]

            # Separate bootstrap resampling from metric calculation (vectorized)
            resampled_indices = stratified_bootstrap_numba(class_indices, [len(class_indices[cls]) for cls in classes], n_bootstrap)
        else:
            resampled_indices = np.random.randint(0, n_samples, (n_bootstrap, n_samples))

        for a in bootstrapped_arguments:
            value = locals()[a][i]
            bootstrapped_arguments[a].append(value[resampled_indices])
    
    bootstrapped_arguments = {a : np.array(bootstrapped_arguments[a]) for a in bootstrapped_arguments}
        
    bootstrap_distribution = metric(average=average, **bootstrapped_arguments)

    lower = np.percentile(bootstrap_distribution, 100 * alpha / 2, axis=-1)
    upper = np.percentile(bootstrap_distribution, 100 * (1 - alpha / 2), axis=-1)

    if method == 'percentile':
        return np.stack([lower, upper], axis=1)

    elif method == 'basic':
        low = 2 * original_stat - upper
        high = 2 * original_stat - lower
        return np.stack([low, high], axis=1)

    elif method == 'bca':
        jack_stats_all = []
        for i in range(len(y_true)):
            n_samples = y_true[i].shape[0]
            jack_idx = np.array([np.delete(np.arange(n_samples), i) for i in range(n_samples)])
            jackknife_arguments = {a : [] for a in metric_arguments[metric_name]}
            for a in jackknife_arguments:
                value = locals()[a][i]
                jackknife_arguments[a] = value[jack_idx]
            jack_stat = metric(average=average, **jackknife_arguments)
            jack_stats_all.append(jack_stat)
        jack_stats_all = np.array(jack_stats_all)
        jack_mean = np.mean(jack_stats_all, axis=1, keepdims=True)
        accel = np.sum((jack_mean - jack_stats_all) ** 3, axis=1) / (6 * (np.sum((jack_mean - jack_stats_all) ** 2, axis=1) ** 1.5))
        z0 = ndtri(np.mean(bootstrap_distribution < original_stat[:, None], axis=1))

        z_low = ndtri(alpha / 2)
        z_high = - z_low
        x_low = z0 + z_low
        x_high = z0 + z_high

        # Define coefficients to compute percentiles, account for 0 acceleration to prevent 0 division error
        pct1_coeff = np.where((accel == 0)|(np.isnan(accel)), x_low, 1 / accel * (1 / (1 - accel * x_low) - 1))
        pct2_coeff = np.where((accel == 0)|(np.isnan(accel)), x_high, 1 / accel * (1 / (1 - accel * x_high) - 1))

        # Define percentiles to take for BCa interval
        pct1 = ndtr(z0 + pct1_coeff)
        pct2 = ndtr(z0 + pct2_coeff)

        # Take percentiles for each sample in batch
        bca_low = [np.percentile(bootstrap_distribution[i], 100 * pct1[i]) for i in range(len(bootstrap_distribution))]
        bca_high = [np.percentile(bootstrap_distribution[i], 100 * pct2[i]) for i in range(len(bootstrap_distribution))]

        bca_low = np.where(np.isnan(accel), np.nan, bca_low) # Correct for acceleration error
        bca_high = np.where(np.isnan(accel), np.nan, bca_high)
        return np.stack([bca_low, bca_high], axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")