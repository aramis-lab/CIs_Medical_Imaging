import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ..df_loaders import extract_df_segm_cov


def plot_cov_fail_dsc_mean(root_folder:str, output_path:str):

    plt.rcdefaults()
    
    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm = ["dsc"]
    stats_segm = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm['method']=='percentile']
    
    data = pd.read_csv(os.path.join(root_folder, "data_matrix_grandchallenge_all.csv"))

    algo_dict = {algo: f"Algo_{i+1}" for i, algo in enumerate(data['alg_name'].unique())}
    task_dict = {task: f"Task_{i+1}" for i, task in enumerate(data['subtask'].unique())}

    df_n = df_segm[df_segm['n'] == 50]
    data_score = data[data['score'] == 'dsc'].copy()

    Q1 = df_n['coverage'].quantile(0.25)
    Q3 = df_n['coverage'].quantile(0.75)
    IQR = Q3 - Q1
    # outliers: below Q1 - 1.5IQR or above Q3 + 1.5IQR

    bad_cov = df_n[(df_n['coverage'] < Q1 - 1.5*IQR)].copy()
    # Worst 25% coverage
    bad_cov = df_n[df_n['coverage'] <= np.percentile(df_n['coverage'], 25)].copy()

    bad_cov['algo_anon'] = bad_cov['algo'].replace(algo_dict)
    bad_cov['task_anon'] = bad_cov['task'].replace(task_dict)
    data_score['algo_anon'] = data_score['alg_name'].replace(algo_dict)
    data_score['task_anon'] = data_score['subtask'].replace(task_dict)
    # Collect (task, algo, coverage) triples

    triples = []
    for _, row in bad_cov.iterrows():
        task = row['task_anon']
        algo = row['algo_anon']
        cov_value = row['coverage']

        dist = data_score[
            (data_score['task_anon'] == task) & 
            (data_score['algo_anon'] == algo)
        ]
        if len(dist) > 0:
            triples.append((task, algo, cov_value))

    # Drop duplicates (keep first occurrence) and sort by coverage

    triples = list({(t, a): (t, a, c) for t, a, c in triples}.values())
    triples.sort(key=lambda x: x[2])  # ascending order (lowest coverage first)

    triples = triples[:15]  # Keep only top 15 worst coverage

    n_pairs = len(triples)
    n_col = 5
    n_row = 3

    fig, axes = plt.subplots(n_row, n_col, figsize=(6*n_col, 5*n_row), squeeze=False)

    for idx, (task, algo, cov_value) in enumerate(triples):
        i, j = divmod(idx, n_col)
        ax = axes[i, j]
        dist = data_score[
            (data_score['task_anon'] == task) & 
            (data_score['algo_anon'] == algo)
        ]

        sns.histplot(
            dist['value'], bins=20, kde=False,
            color="skyblue", edgecolor="black", alpha=0.7, ax=ax
        )

        ax.set_title(f"Task: {task}, Algo: {algo}\nn={len(dist)}", fontsize=12)
        ax.legend([f"Coverage={cov_value:.3f}"], loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Hide unused subplots

    for k in range(n_pairs, n_row*n_col):
        i, j = divmod(k, n_col)
        axes[i, j].axis("off")

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Supp Figure coverage failure DSC mean.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/cov_fail_dsc_mean.pdf")

    # Call your plotting function
    plot_cov_fail_dsc_mean(root_folder, output_path)

if __name__ == "__main__":
    main()