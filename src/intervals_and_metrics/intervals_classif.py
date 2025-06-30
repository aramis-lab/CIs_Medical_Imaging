
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2, norm
from scipy.optimize import root_scalar
from .pixel_wise_metrics import get_metric, label_binarize_vectorized
from numba import njit

from sklearn.preprocessing import label_binarize

def compute_CIs_classification(y_true, y_pred, metric, method, average=None, alpha=0.05):
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
        batch_results = CI_accuracy(y_true, y_pred, metric, method, average, alpha)
    elif metric in ['ap', 'auc', 'auroc']:
        batch_results = CI_AUC(y_true, y_pred, metric, method, alpha, average)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return np.stack(batch_results, axis=0)

def CI_accuracy(y_true, y_pred, metric, method, average, alpha):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if method in ["wald", "param_z"]:
        method = "normal"
    elif method in ["cloper_pearson", "exact"]:
        method = "beta"
    
    if method in ["normal","agresti_coull","beta","wilson"] and average=="micro": # If average is micro, we can consider the problem as a binary classification problem
        classes = np.unique(y_true)
        y_pred = np.argmax(y_pred, axis=-1) if y_pred.ndim > y_true.ndim else y_pred
        y_pred_bin = label_binarize(y_pred.ravel(), classes=classes).reshape(y_pred.shape[0], -1)
        y_true_bin = label_binarize(y_true.ravel(), classes=classes).reshape(y_true.shape[0], -1)
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1), axis=1)
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0), axis=1)
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0), axis=1)
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1), axis=1)

        if metric == "accuracy":
            value = (tp + tn)
            total = tp + fp + fn + tn
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
        return stratified_bootstrap_CI(y_true, y_pred, metric_name=metric, average=average, n_bootstrap=9999, alpha=alpha, method=method)
    else:
        if average!="micro":
            raise ValueError("Non-bootstrap CI methods are not defined for multi-class if average is not 'micro'.")
        else:
            raise NotImplementedError(f"The following method is not implemented : {method}. Currently, 'percentile', 'basic', 'bca', 'agresti_coull', 'wilson', 'wald', 'normal', 'param_z', 'cloper_pearson' and 'exact' are implemented.")

def CI_AUC(y_true, y_pred, metric, method, alpha, average):
    if method in ['percentile', 'basic', 'bca']: 
        return stratified_bootstrap_CI(y_true, y_pred, metric_name=metric, average=average, n_bootstrap=9999, alpha=alpha, method=method)
    elif y_pred.ndim==1 and metric in ['auc', 'auroc']:
        y_true_bin = label_binarize(y_true, classes=np.arange(y_pred.shape[1]))
        N=len(y_true_bin)
        n=np.sum(y_true)
        m=N-n
        AUC=roc_auc_score(y_true_bin, y_pred, average=average, multi_class="ovr")
        sorted_indices = np.argsort(y_pred)
        y_sorted = y_true[sorted_indices]
        positive_in_sorted = np.where(y_sorted == 1)[0]
        negative_in_sorted=np.where(y_sorted == 0)[0]
        bar_R=np.mean(negative_in_sorted)
        bar_S=np.mean(positive_in_sorted)
        ideal_R = np.arange(1, len(negative_in_sorted) + 1)
        ideal_S=np.arange(1, len(positive_in_sorted) + 1)
        S_10=(1/((m-1)*n**2))*(np.sum((negative_in_sorted-ideal_R)**2)-m*(bar_R-(m+1)/2)**2)
        S_01=(1/((n-1)*n**2))*(np.sum((positive_in_sorted-ideal_S)**2)-n*(bar_S-(n+1)/2)**2)
        S=np.sqrt((m*S_01+n*S_10)/(m+n))
        Y = y_pred[y_true == 1]  # diseased
        X = y_pred[y_true == 0] 
        if method == "delong": 
            return CI_DL(y_pred, y_true,AUC, m,n)
        elif method == "logit_transform":
            return CI_LT(AUC, m, n, S)
        elif method == "empirical_likelihood":
            return el_auc_confidence_interval(Y, X, S,AUC, alpha)
        else:
            raise NotImplementedError(f"The following method is not implemented : {method}. Currently, 'percentile', 'basic', 'bca', 'delong', 'logit_transform' and 'empirical_likelihood' are implemented.")
    else:
        raise ValueError("Non-bootstrap CI methods are not defined for multi-class AUC.")

def CI_LT(AUC, m, n, S):
    if AUC !=0 and AUC !=1:
        LL=np.log(AUC/(1-AUC))-1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
        UL=np.log(AUC/(1-AUC))+1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
        LT_low=np.exp(LL)/(1+np.exp(LL))
        LT_high=np.exp(UL)/(1+np.exp(UL))
        return(np.array([LT_low, LT_high]))
    else:
        return( np.array([np.nan, np.nan]))

def CI_DL(y_pred, y,AUC, m,n):
    positive_preds = y_pred[y == 1]
    negative_preds = y_pred[y == 0]
   
    # Get indices of negative samples
    negative_indices = np.where(y == 0)[0]
    positive_indices= np.where(y == 1)[0]
    # Compute proportion for each negative
    D10 = []
    D01=[]
    for idx in negative_indices:
        pred = y_pred[idx]
        proportion = np.mean(positive_preds >= pred)
        D10.append( proportion)
    for idx in positive_indices:
        pred = y_pred[idx]
        proportion = np.mean(negative_preds <= pred)
        D01.append(proportion)
   
    var=(1/(m*(m-1)))*np.sum((D10-AUC)**2) + (1/(n*(n-1)))*np.sum((D01-AUC)**2)
    
    DL_low=AUC-1.96*np.sqrt(var)
    DL_high=AUC+1.96*np.sqrt(var)
    return(np.array([DL_low, DL_high]))

def el_auc_confidence_interval(Y, X, S,AUC, alpha):
    n, m = len(Y), len(X)
    tol=1e-8
    # Compute U_hat
    U_hat = 1 - np.array([np.mean(X <= yj) for yj in Y])
    
    if np.sum(U_hat) == 0:
        return np.array([1.0, 1.0])
    if np.sum(U_hat) == len(U_hat):
        return np.array([0.0, 0.0])

    def test_stat(delta):
        def estimating_equation(lmbda):
            denom = 1 + lmbda * (1 - U_hat - delta)
            if np.any(denom <= 0):
                return np.nan
            return np.mean((1 - U_hat - delta) / denom)
        
        # Find lambda via root finding
        grid = np.linspace(-1000, 1000, 10000)
        f_vals = np.array([estimating_equation(lmbda) for lmbda in grid])
        valid = ~np.isnan(f_vals)
        sign_change = np.where(np.diff(np.sign(f_vals[valid])))[0]
        
        if len(sign_change) == 0:
            return np.inf  # invalid delta

        idx = sign_change[0]
        lam_left = grid[valid][idx]
        lam_right = grid[valid][idx + 1]
        
        sol = root_scalar(estimating_equation, bracket=[lam_left, lam_right], method='brentq')
        if not sol.converged:
            return np.inf

        lmbda = sol.root
        l_val = 2 * np.sum(np.log(1 + lmbda * (1 - U_hat - delta)))
        r_delta = (m / (m + n)) * np.sum((1 - U_hat - delta)**2) / (n * S**2)
        return r_delta * l_val

    chi2_crit = chi2.ppf(1 - alpha, df=1)

    def in_conf(delta):
        return test_stat(delta) - chi2_crit

    # Bisection to find lower bound
    def find_bound(low, high, increasing):
        while high - low > tol:
            mid = (low + high) / 2
            val = in_conf(mid)
            if (val <= 0)==increasing:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    # Bracket the confidence region
    try:
        # Find feasible region by coarse search
        # coarse_grid = np.linspace(0, 1, 100)
        # feasible = [delta for delta in coarse_grid if in_conf(delta) <= 0]
        # if not feasible:
        #     raise ValueError("No valid confidence interval found.")

        # delta_low_start = min(feasible)
        # delta_high_start = max(feasible)

        ci_low = find_bound(0.0, AUC, increasing=True)
        ci_high = find_bound(AUC, 1.0, increasing=False)
        return np.array([ci_low, ci_high])
    except Exception as e:
        raise ValueError(f"Failed to compute confidence interval: {e}")

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
def stratified_bootstrap_CI(y_true, y_score, metric_name='auc', average='micro', n_bootstrap=9999, alpha=0.05, method='percentile'):

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    metric = get_metric(metric_name)
    original_stat = metric(y_true, y_score, average)

    y_samples = []
    y_score_samples = []
    classes_b = np.arange(y_score.shape[-1])
    if y_true.ndim == 1:  # single sample
        y_true = y_true[None, ...]
        y_score = y_score[None, ...]
    for i in range(len(y_true)):
        sample = y_true[i]
        n_samples = len(sample)
        
        class_indices = [np.flatnonzero(sample == c) for c in classes_b]

        # Separate bootstrap resampling from metric calculation (vectorized)
        resampled_indices = stratified_bootstrap_numba(class_indices, [len(class_indices[cls]) for cls in classes_b], n_bootstrap)

        y_samples.append(y_true[i, resampled_indices])
        y_score_samples.append(y_score[i, resampled_indices])
    
    stats = metric(np.array(y_samples), np.array(y_score_samples), average)

    lower = np.percentile(stats, 100 * alpha / 2, axis=-1)
    upper = np.percentile(stats, 100 * (1 - alpha / 2), axis=-1)

    if method == 'percentile':
        return np.stack([lower, upper], axis=1)

    elif method == 'basic':
        low = 2 * original_stat - upper
        high = 2 * original_stat - lower
        return np.stack([low, high], axis=1)

    elif method == 'bca':
        jack_stats_all = []
        for i in range(len(y_true)):
            jack_stats = []
            n_samples = y_true[i].shape[0]
            for j in range(n_samples):
                jack_idx = np.delete(np.arange(n_samples), j)
                y_jack_true = y_true[i][jack_idx]
                y_jack_score = y_score[i][jack_idx]
                jack_stat = metric(y_jack_true, y_jack_score, average)
                jack_stats.append(jack_stat)
            jack_stats_all.append(jack_stats)
        jack_stats_all = np.array(jack_stats_all).squeeze()
        jack_mean = np.mean(jack_stats_all, axis=1, keepdims=True)
        accel = np.sum((jack_mean - jack_stats_all) ** 3, axis=1) / (6 * (np.sum((jack_mean - jack_stats_all) ** 2, axis=1) ** 1.5)).flatten()
        z0 = norm.ppf(np.mean(stats < original_stat[:, None], axis=1))
        z_low = norm.ppf(alpha / 2)
        z_high = norm.ppf(1 - alpha / 2)
        x_low = z0 + z_low
        x_high = z0 + z_high
        pct1_coeff = np.where((accel == 0)|(np.isnan(accel)), x_low, 1 / accel * (1 / (1 - accel * x_low) - 1))
        pct2_coeff = np.where((accel == 0)|(np.isnan(accel)), x_high, 1 / accel * (1 / (1 - accel * x_high) - 1))
        pct1 = norm.cdf(z0 + pct1_coeff)
        pct2 = norm.cdf(z0 + pct2_coeff)
        bca_low = np.percentile(stats, 100 * pct1)
        bca_high = np.percentile(stats, 100 * pct2)
        bca_low = np.where(np.isnan(accel), np.nan, bca_low)
        bca_high = np.where(np.isnan(accel), np.nan, bca_high)
        return np.stack([bca_low, bca_high], axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")