import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
from scipy.stats import skew
import argparse
from tqdm import tqdm

from ..df_loaders import extract_df_segm_cov

def perform_fits(df_segm, stats):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
                for stat in stats:
                    df_metric_stat = df_algo[(df_algo['metric'] == metric) & (df_algo['stat']==stat)]
                    for method in df_metric_stat['method'].unique():
                        df_metric_stat_method = df_metric_stat[df_metric_stat['method'] == method]
                        df_metric_stat_method = df_metric_stat_method.sort_values(by='n')
                        n_values = df_metric_stat_method['n'].to_numpy()
                        coverages = df_metric_stat_method['coverage'].to_numpy()
                        Y = 0.95 - coverages
                        X = np.vstack([1/n_values]).T
                        beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                        rel_error = np.sqrt(res[0]) / np.linalg.norm(coverages)
                        new_row = {
                            'task': task,
                            'algo': algo,
                            'metric': metric,
                            'stat': stat,
                            'method': method,
                            'beta2': beta2[0],
                            'relative_error': rel_error
                        }
                        results.append(new_row)
    df_fit_results = pd.DataFrame(results)
    return df_fit_results

def compute_descriptive_stats(root_folder:str):

    data = pd.read_csv(os.path.join(root_folder, "data_matrix_grandchallenge_all.csv"))

    results=[]
    metrics=data['score'].unique()

    for score in metrics:
        df=data[data['score']==score]
        algos=df['alg_name'].unique()
        score=df['score'].unique()[0]
        for alg in algos:

            df_alg= df[df['alg_name']==alg]
            tasks = df_alg['subtask'].unique()
            for task in tasks:
                if score=='cldice' and task not in  ['Task08_HepaticVessel_L1','Task08_HepaticVessel_L2']:
                    continue
                else:
                    values = df_alg[df_alg['subtask'] == task]['value'].dropna()
                    if len(values)<50:
                        continue
                    value={
                    'metric':score,
                    'algo': alg,
                    'task': task,
                    'skewness': skew(values)
                    }
                    results.append(value)
        
    results_df=pd.DataFrame(results)
    return results_df

def combine_fit_and_stats(df_fit_results, df_stats):
    df_merged = pd.merge(df_fit_results, df_stats, left_on=['algo', 'task', 'metric'], right_on=['algo', 'task', 'metric'], how='inner')
    return df_merged

def compute_correlation_skew_CCP(root_folder:str):

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc", "iou", "nsd", "boundary_iou", "cldice", "assd", "masd", "hd", "hd_perc"]
    stats_segm = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm['method'] == 'percentile']
    df_fit_results = perform_fits(df_segm, stats_segm)

    df_stats = compute_descriptive_stats(root_folder)

    df_fit_results = combine_fit_and_stats(df_fit_results, df_stats)

    df_fit_results['skewness_sq'] = df_fit_results['skewness'] ** 2 
    df_fit_results["task_metric"] = df_fit_results["task"] + "_" + df_fit_results["metric"] 
    df_fit_results = df_fit_results.reset_index(drop=True)
    df_fit_results["task_metric"] = df_fit_results["task_metric"].astype(str)
    df_fit_results = df_fit_results.dropna(subset=["beta2", "skewness_sq", "task_metric"])
    # --- Base model ---
    model = smf.mixedlm("beta2 ~ skewness", df_fit_results, groups=df_fit_results["task_metric"])
    fit = model.fit()

    # --- Function to compute marginal R² ---
    def compute_r2_marginal(fit, df):
        X_fixed = pd.DataFrame({
        "Intercept": 1.0,
        "skewness": df["skewness"].astype(float)
        }, index=df.index)
        mu_fixed = X_fixed.dot(fit.fe_params)            # fixed-effect linear predictor
        var_fixed = np.var(mu_fixed, ddof=1)             # σ_f^2

        # random intercept variance (if random intercept only)
        var_random = float(fit.cov_re.iloc[0, 0])        # σ^2_random

        # residual variance
        var_resid = float(fit.scale)                     # σ^2_resid

        # Marginal R2 and implied correlation (includes intercept variance)
        R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
        return np.sign(fit.fe_params["skewness"]) * np.sqrt(R2_marginal)

    r2_orig = compute_r2_marginal(fit, df_fit_results)
    # -------------------
    # Bootstrap CI (two-level)
    # -------------------
    n_boot = 9999
    boot_r2 = []
    groups = df_fit_results["task_metric"].unique()
    rng = np.random.default_rng(seed=42)

    for b in tqdm(range(n_boot)):
        # Step 1: sample groups with replacement
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        boot_samples = []

        # Step 2: within each group, resample observations with replacement
        for g in sampled_groups:
            df_group = df_fit_results[df_fit_results["task_metric"] == g]
            df_group_boot = df_group.sample(n=len(df_group), replace=True, random_state=rng.integers(1e9))
            df_group_boot["task_metric"] = g  # keep same group label
            boot_samples.append(df_group_boot)

        # Combine into one bootstrap dataset
        df_boot = pd.concat(boot_samples, ignore_index=True)

        try:
            model_boot = smf.mixedlm("beta2 ~ skewness", df_boot, groups=df_boot["task_metric"])
            fit_boot = model_boot.fit()
            r2_boot = compute_r2_marginal(fit_boot, df_boot)
            boot_r2.append(r2_boot)
        except Exception as e:
            print(e)
            continue  # skip failed fits

    # --- Bootstrap CI ---
    r2_ci_lower = np.percentile(boot_r2, 2.5)
    r2_ci_upper = np.percentile(boot_r2, 97.5)
    print(f"Bootstrap 95% CI for R²_marginal {r2_orig:.4f}: [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")

def main():
    parser = argparse.ArgumentParser(description="Compute correlation between skewness and CCP convergence rate.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data.")
    args = parser.parse_args()
    compute_correlation_skew_CCP(args.root_folder)

if __name__ == "__main__":
    main()