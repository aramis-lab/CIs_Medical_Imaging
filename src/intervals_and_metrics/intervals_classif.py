
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2, norm
from scipy.optimize import root_scalar
from .pixel_wise_metrics import get_metric

from sklearn.preprocessing import label_binarize

def compute_CIs_classification(y_true, y_pred, metric, method, average=None, alpha=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:  # single sample
        y_true = y_true[None, ...]
        y_pred = y_pred[None, ...]
    elif y_pred.ndim == y_true.ndim + 1 and metric != 'auc':  # shape (B, N, C)
        pass  # already batched
    elif y_pred.ndim == y_true.ndim:  # likely shape (B, N)
        pass
    else:
        raise ValueError("Input dimension mismatch or unsupported format.")

    batch_results = []
    for yt, yp in zip(y_true, y_pred):
        if metric in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "fbeta_score", "mcc"]:
            res = CI_accuracy(yt, yp, metric, method, average, alpha)
        elif metric in ['ap', 'auc', 'auroc']:
            res = CI_AUC(yt, yp, method, alpha, average)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        batch_results.append(res)

    return np.stack(batch_results, axis=0)


def CI_accuracy(y_true, y_pred, metric, method, average, alpha):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if method in ["wald", "param_z"]:
        method = "normal"
    elif method in ["cloper_pearson", "exact"]:
        method = "beta"
    

    if method in ["normal","agresti_coull","beta","wilson"] and average=="micro":
        classes = np.unique(y_true)
        y_pred = np.argmax(y_pred, axis=-1) if y_pred.ndim > y_true.ndim else y_pred
        y_pred_bin = label_binarize(y_pred.ravel(), classes=classes).reshape(y_pred.shape[0], -1)
        y_true_bin = label_binarize(y_true.ravel(), classes=classes).reshape(y_true.shape[0], -1)
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))

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
        return stratified_bootstrap_CI(y_true, y_pred, metric_name=metric, average=average, n_bootstrap=1000, alpha=1-alpha, method=method)
    else:
        raise NotImplementedError(f"The following method is not implemented : {method}. Currently, 'percentile', 'basic', 'bca', 'agresti_coull', 'wilson', 'wald', 'normal', 'param_z', 'cloper_pearson' and 'exact' are implemented.")

def CI_AUC(y_true, y_pred, method, alpha, average):
   
    if method in ['percentile', 'basic', 'bca']: 
        return stratified_bootstrap_CI(y_true, y_pred, metric_name='auc', average=average, n_bootstrap=1000, alpha=alpha, method=method)
    else:
        y_true_bin = label_binarize(y_true, classes=y_pred.shape[1])
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

def stratified_bootstrap_CI(y_true, y_score, metric_name='auc', average='micro', n_bootstrap=1000, alpha=0.05, method='percentile'):
    y_true = np.array(y_true)
   
    y_score = np.array(y_score)
    
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize if needed for AUC
    if metric_name == 'auc':
        y_true_bin = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
        if n_classes == 2:
            y_score = y_score.ravel()
            y_true_bin = y_true_bin.ravel()
        original_stat = roc_auc_score(y_true_bin, y_score, average=average, multi_class='ovr')
    elif metric_name in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "fbeta_score", "mcc"]:
        metric = get_metric(metric_name)
        y_pred = np.argmax(y_score, axis=-1) if y_score.ndim > y_true.ndim else y_score
        original_stat = metric(y_true, y_pred, average=average)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    stats = []
    original_stat = np.array(original_stat)

    n_samples = len(y_true)
    stats = []

    # Stratified resampling, per batch
    for _ in range(99):

        classes_b = np.unique(y_true)

        # Get class indices for the current batch
        class_indices = {cls: np.where(y_true == cls)[0] for cls in classes_b}

        # Resample with stratification
        resample_idx = []
        for cls in classes_b:
            resample_idx.extend(np.random.choice(class_indices[cls], size=len(class_indices[cls]), replace=True))
        resample_idx = np.array(resample_idx)
        np.random.shuffle(resample_idx)

        if metric_name == 'auc':
            y_sample_bin = y_true_bin[resample_idx]
            y_sample_score = y_score[resample_idx]
            try:
                stat = roc_auc_score(y_sample_bin, y_sample_score, average=average, multi_class='ovr')
            except ValueError:
                stat = np.nan
        elif metric_name in ["accuracy", "npv", "ppv", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1_score", "fbeta_score", "mcc"]:
            y_sample = y_true[resample_idx]
            y_sample_score = y_score[resample_idx]
            stat = metric(y_sample, y_sample_score, average=average)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        stats.append(stat)

    stats = np.array(stats)  # shape: (n_bootstraps, batch_size)

    # Confidence interval calculation (percentile method)
    lower = np.percentile(stats, 100 * alpha / 2, axis=0)
    upper = np.percentile(stats, 100 * (1 - alpha / 2), axis=0)

    if method == 'percentile':
        return np.array([lower, upper])

    elif method == 'basic':
        low = 2 * original_stat - upper
        high = 2 * original_stat - lower
        return np.array([low, high])

    elif method == 'bca':

        jack_stats = []
        for i in range(n_samples):
            jack_idx = np.delete(np.arange(n_samples), i)

            if metric_name == 'auc':
                y_true_bin_jack = y_true_bin[jack_idx]
                jack_stat = roc_auc_score(y_true_bin_jack, y_score[jack_idx], average=average, multi_class='ovr')
            else:
                y_jack_pred = np.argmax(y_score[jack_idx], axis=1) if y_score.ndim > y_true.ndim else y_score[jack_idx]
                jack_stat = metric(y_true[jack_idx], y_jack_pred, average=average)

            jack_stats.append(jack_stat)

        jack_stats = np.array(jack_stats)
        jack_mean = np.mean(jack_stats)
        accel = np.sum((jack_mean - jack_stats) ** 3) / (6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5))

        z0 = norm.ppf(np.mean(stats < original_stat))
        z_low = norm.ppf(alpha / 2)
        z_high = norm.ppf(1 - alpha / 2)

        x_low = z0 + z_low
        x_high = z0 + z_high
        
        if np.isnan(accel):
            return np.array([np.nan, np.nan])
        elif accel == 0:
            pct1 = norm.cdf(z0 + x_low)
            pct2 = norm.cdf(z0 + x_high)
        else:
            pct1 = norm.cdf(z0 + 1/accel*(1/(1-accel*x_low) - 1))
            pct2 = norm.cdf(z0 + 1/accel*(1/(1-accel*x_high) - 1))

        bca_low = np.percentile(stats, 100 * pct1)
        bca_high = np.percentile(stats, 100 * pct2)

        return np.array([bca_low, bca_high])

    else:
        raise ValueError(f"Unknown method: {method}")
