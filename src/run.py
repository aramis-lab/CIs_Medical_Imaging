import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import hydra
from omegaconf import DictConfig
from kde import weighted_kde, sample_weighted_kde
from summary_stats import get_statistic
from intervals_and_metrics import compute_CIs, compute_metrics
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "data_matrix_grandchallenge.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def extract_df(df, metric, task):
    metric = "DCS" if metric == "DSC" else "HSD" if metric == "NSD" else metric
    df = df[(df["score"] == metric) & (df["subtask"] == task)]
    return df.drop(["score", "subtask", "task", "case", "alg_number", "phase"], axis=1)

def make_kdes_and_compute_metrics(df, task, config):

    # Retrieve configuration and set up variables
    ci_methods = config.ci_methods
    statistic = lambda x, axis=None: get_statistic(config.summary_stat)(x, config.trimmed_mean_threshold, axis=axis)
    results = pd.DataFrame()

    # Iterate over algorithms
    for algo in tqdm(df["alg_name"].unique()):
        DSCs = df[df["alg_name"] == algo]["value"].to_numpy()

        # Define the grid for KDE
        x = np.linspace(0, 1, 1000)  # You can change the resolution of x
        alphas = np.ones(len(DSCs))

        dist_to_bounds = np.min([DSCs - np.min(DSCs), np.max(DSCs)-DSCs], axis=0)

        # Iterative weighted KDE estimation
        for _ in range(1):
            y = weighted_kde(DSCs, x, alphas, config.kernel)
            indices = np.searchsorted(x, DSCs)
            initial_estimates = y[indices]
            log_g = np.mean(np.log(initial_estimates))
            g = np.exp(log_g)
            alphas = (initial_estimates / g) ** (-1/2) * (1-np.exp(-1e6*dist_to_bounds))
        
        y = weighted_kde(DSCs, x, alphas, config.kernel)
        samples = sample_weighted_kde(y, x, 1000000)

        # Compute true statistic
        true_value = statistic(samples)

        for n in tqdm([10, 25, 50, 75, 100, 125, 150, 200, 250]):
            new_row = {"subtask": task, "alg_name": algo, "n": n}
            samples = sample_weighted_kde(y,x, config.n_samples*n).reshape(config.n_samples, n)

            for method in ci_methods:
                CIs, nan_proportion = compute_CIs(samples, method, statistic)
                coverage, proportion_oob, width = compute_metrics(CIs, true_value)
                new_row.update({f"{method}_coverage": coverage, f"{method}_proportion_oob": proportion_oob, f"{method}_width": width, f"{method}_nans": nan_proportion})
            results = pd.concat([results, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    return results

def process_subtask(task, cfg):
    print(f"Running KDE for metric {cfg.metric} and subtask {task}")
    df = pd.read_csv(DATA_PATH)
    df = extract_df(df, cfg.metric, task)
    results = make_kdes_and_compute_metrics(df, task, cfg)
    return results

@hydra.main(config_path="cfg", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig):
    aggreggated_results = pd.DataFrame()

    # Use joblib to parallelize tasks
    with joblib.Parallel(n_jobs=-1) as parallel:
        results = parallel(joblib.delayed(process_subtask)(task, cfg) for task in cfg.tasks)

    # Combine results from all tasks
    for result in results:
        aggreggated_results = pd.concat([aggreggated_results, result], ignore_index=True)
    
    aggreggated_results.to_csv(os.path.join(RESULTS_DIR, f"results_{cfg.metric}_{cfg.summary_stat}.csv"))

if __name__ == "__main__":
    main()