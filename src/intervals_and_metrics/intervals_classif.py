import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import binomtest, bootstrap
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2
from scipy.optimize import root_scalar


def CI_accuracy(y_true, y_pred, method, alpha):
    n_success=np.sum(y_true==y_pred)
    n=len(y_pred)
    if method in ["normal","agresti_coull","beta","wilson"]:
        return  np.array(proportion_confint(n_success, n, alpha=alpha, method= method)).squeeze().T
    elif method in ['percentile', 'basic', 'bca']:
        return scipy_bootstrap_ci((y_true==y_pred), method=method)
    elif method == 'studentized':
        return studentized_interval_accuracy((y_true==y_pred,), alpha=alpha)


def scipy_bootstrap_ci(data, method='percentile', alpha=0.05, n_resamples=9999):
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
    return np.vstack([samples_statistics.squeeze() - upper_bound * samples_stds, samples_statistics.squeeze() - lower_bound * samples_stds]).squeeze().T

def CI_AUC(y_true, y_pred, method, alpha):
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
    Y = y_pred[y == 1] # diseased
    X = y_pred[y == 0] 
    if method == "DeLong": 
        return CI_DL(y_pred, y_true,AUC, m,n)
    elif method == "Logit Transform":
        return CI_LT(AUC, m, n, S)
    elif method == "Empirical Likelihood":
        return el_auc_confidence_interval(Y, X, S)
    elif method in ['percentile', 'basic', 'bca']:
        res=bootstrap((y, y_pred),
                statistic=auc_statistic,
                vectorized=False,  # because roc_auc_score isn't vectorized
                paired=True,
                confidence_level=0.95,
                n_resamples=1000,
                method=method,
                random_state=42).confidence_interval
        return np.array([res.low, res.high])
    elif method == 'studentized':
        return studentized_interval_accuracy(y_true, y_pred,alpha=alpha, outer_resamples=1000, inner_resamples=500)
    

CI_AUC(y, y_pred, 'bca', alpha)
def auc_statistic(y, y_pred, axis=None):
    return roc_auc_score(y, y_pred)

def CI_LT(AUC, m, n, S):
    if AUC !=0 and AUC !=1:
        LL=np.log(AUC/(1-AUC))-1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
        UL=np.log(AUC/(1-AUC))+1.96*np.sqrt((m+n)*S**2/(m*n))/(AUC*(1-AUC))
        LT_low=np.exp(LL)/(1+np.exp(LL))
        LT_high=np.exp(UL)/(1+np.exp(UL))
        return(np.arrayy([LT_low, LT_high]))
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

def el_auc_confidence_interval(Y, X, S):
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


def studentized_bootstrap_auc(y_true, y_score, alpha=0.05, outer_resamples=1000, inner_resamples=500):
    rng = np.random.default_rng(42)

    n = len(y_true)
    
    # Original AUC
    auc_hat = roc_auc_score(y_true, y_score)
    
    t_values = []
    auc_boots = []
    
    for _ in range(outer_resamples):
        # Outer bootstrap: sample indices with replacement
        indices = rng.choice(n, size=n, replace=True)
        y_b = y_true[indices]
        s_b = y_score[indices]
        
        try:
            auc_b = roc_auc_score(y_b, s_b)
        except ValueError:
            # Not enough class variety
            continue
        
        # Inner bootstrap to estimate std of AUC_b
        inner_aucs = []
        for _ in range(inner_resamples):
            inner_indices = rng.choice(n, size=n, replace=True)
            y_i = y_b[inner_indices]
            s_i = s_b[inner_indices]
            try:
                inner_auc = roc_auc_score(y_i, s_i)
                inner_aucs.append(inner_auc)
            except ValueError:
                continue

        if len(inner_aucs) < 2:
            continue

        sigma_b = np.std(inner_aucs, ddof=1)
        t = (auc_b - auc_hat) / sigma_b
        t_values.append(t)
        auc_boots.append(auc_b)
    
    t_values = np.array(t_values)
    t_low = np.percentile(t_values, 100 * (1 - alpha / 2))
    t_high = np.percentile(t_values, 100 * (alpha / 2))

    sigma_hat = np.std(auc_boots, ddof=1)

    # Invert studentized statistic to get CI
    lower = auc_hat - t_low * sigma_hat
    upper = auc_hat - t_high * sigma_hat

    return np.array([lower, upper])
        