
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2, norm
from scipy.optimize import root_scalar

from sklearn.preprocessing import label_binarize

def CI_accuracy(y_true, y_pred, method, alpha):
    
    if method in ["normal","agresti_coull","beta","wilson"]:
        n_success=np.sum(y_true==y_pred)
        n=len(y_pred)
        return  proportion_confint(n_success, n, alpha=alpha, method= method)
    if method in ['percentile', 'basic', 'bca']:
        return stratified_bootstrap_metric(y_true, y_pred, metric='accuracy', average=None, n_bootstrap=1000, alpha=1-alpha, method=method)


def scipy_bootstrap_ci(data, method='percentile',statistic=np.mean, alpha=0.05, n_resamples=9999):

    data = np.array(data).astype(float)
    ci = bootstrap(
        data=(data,),
        statistic=np.mean,
        confidence_level=1 - alpha,
        method=method,
        n_resamples=n_resamples,
        vectorized=True,
    ).confidence_interval
    return  np.array([ci.low, ci.high]).squeeze().T

def studentized_interval_accuracy(samples, alpha, n_resamples=9999):
    samples = np.array(samples).astype(float)
    samples_statistics = np.mean(samples, axis=-1)
    
    samples_stds = np.std(samples, axis=-1).squeeze()
    bootstrap_samples = np.stack([np.random.choice(samples[i], size=(n_resamples, samples.shape[1]), replace=True) for i in range(samples.shape[0])])
    bootstrap_statistics = np.mean(bootstrap_samples, axis=-1).squeeze()
    
    boostrap_stds = np.std(bootstrap_samples, axis=-1)
    studentized = (bootstrap_statistics - samples_statistics) / boostrap_stds
    
    lower_bound = np.percentile(studentized, 100 * alpha / 2, axis=1)
    
    upper_bound = np.percentile(studentized, 100 * (1 - alpha / 2), axis=1)

    return np.vstack([samples_statistics - upper_bound * samples_stds, samples_statistics - lower_bound * samples_stds])





def CI_AUC(y_true, y_pred, method, alpha, average):
   
    if method in ['percentile', 'basic', 'bca']: 
        
        return stratified_bootstrap_metric(y_true, y_pred, metric='auc', average=average, n_bootstrap=1000, alpha=alpha, method=method)
    else: 
        N=len(y_true)
        n=np.sum(y_true)
        m=N-n
        AUC=roc_auc_score(y_true, y_pred)
        sorted_indices = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_indices]
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
        if method == "DeLong": 
            return CI_DL(y_pred, y_true,AUC, m,n)
        elif method == "Logit Transform":
            return CI_LT(AUC, m, n, S)
        elif method == "Empirical Likelihood":
            return el_auc_confidence_interval(Y, X, S,AUC, alpha)
       


def auc_statistic(y, y_pred, axis=None, average=None ):
    if average=="weighted":
        return roc_auc_score(y, y_pred, average="weighted")
    else: 
        return roc_auc_score(y, y_pred)


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
        # print(in_conf(ci_low), in_conf(ci_high))
        return np.array([ci_low, ci_high])
    except Exception as e:
        raise ValueError(f"Failed to compute confidence interval: {e}")




def stratified_bootstrap_metric(y_true, y_score, metric='auc', average='micro', n_bootstrap=1000, alpha=0.05, method='percentile'):
    y_true = np.array(y_true)
   
    y_score = np.array(y_score)
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    n = len(y_true)

    # Binarize if needed for AUC
    if metric == 'auc':
        y_true_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_score = y_score.ravel()
            y_true_bin = y_true_bin.ravel()
        original_stat = roc_auc_score(y_true_bin, y_score, average=average, multi_class='ovr')
    elif metric == 'accuracy':
 
        y_pred = np.argmax(y_score, axis=1) if y_score.ndim > 1 else y_score
        original_stat = accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    stats = []
    
    # Stratified resampling
    class_indices = {cls: np.where(y_true == cls)[0] for cls in classes}

    for _ in range(n_bootstrap):
        resample_idx = []
        for cls in classes:
            resample_idx.extend(np.random.choice(class_indices[cls], size=len(class_indices[cls]), replace=True))
        resample_idx = np.array(resample_idx)
        np.random.shuffle(resample_idx)

        if metric == 'auc':
            y_sample_bin = y_true_bin[resample_idx]
            y_sample_score = y_score[resample_idx]
            try:
                stat = roc_auc_score(y_sample_bin, y_sample_score, average=average, multi_class='ovr')
                stats.append(stat)
            except ValueError:
                continue
        elif metric == 'accuracy':
            y_sample = y_true[resample_idx]
            
            y_sample_score = y_score[resample_idx]
         
            y_sample_pred = np.argmax(y_sample_score, axis=1) if y_sample_score.ndim > 1 else y_sample_score
            stat = accuracy_score(y_sample, y_sample_pred)
           
            stats.append(stat)

    stats = np.array(stats)
   
    # Confidence interval calculation
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))

    if method == 'percentile':
        return [lower, upper]

    elif method == 'basic':
        low = 2 * original_stat - upper
        high = 2 * original_stat - lower
        return [low, high]

    elif method == 'bca':
        jack_stats = []
        for i in range(n):
            jack_idx = np.delete(np.arange(n), i)
            if metric == 'auc':
                y_true_jack = y_true[jack_idx]
                y_true_bin_jack = label_binarize(y_true_jack, classes=classes)
                jack_stat = roc_auc_score(y_true_bin_jack, y_score[jack_idx], average=average, multi_class='ovr')
            else:
                y_jack_pred = np.argmax(y_score[jack_idx], axis=1) if y_score.ndim > 1 else y_score[jack_idx]
                jack_stat = accuracy_score(y_true[jack_idx], y_jack_pred)
            jack_stats.append(jack_stat)

        jack_stats = np.array(jack_stats)
        jack_mean = np.mean(jack_stats)
        acc = np.sum((jack_mean - jack_stats) ** 3) / (6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5))

        z0 = norm.ppf(np.mean(stats < original_stat))
        z_low = norm.ppf(alpha / 2)
        z_high = norm.ppf(1 - alpha / 2)

        pct1 = norm.cdf(z0 + (z0 + z_low) / (1 - acc * (z0 + z_low)))
        pct2 = norm.cdf(z0 + (z0 + z_high) / (1 - acc * (z0 + z_high)))

        bca_low = np.percentile(stats, 100 * pct1)
        bca_high = np.percentile(stats, 100 * pct2)
        return [bca_low, bca_high]

    else:
        raise ValueError(f"Unknown method: {method}")
print( CI_accuracy(np.array([1, 0, 1, 0, 1, 1, 0, 1]), np.array([1, 1, 1, 0, 0, 0, 0, 1]) ,method= "studentized", alpha=0.95))