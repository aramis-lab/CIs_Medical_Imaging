import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

unbounded_metrics = ["masd", "assd", "hd", "hd_perc"]
bounded_metrics = ["dsc", "nsd", "iou", "boundary_iou", "cldice"]
all_metrics = unbounded_metrics + bounded_metrics
classif_metrics = ["accuracy", "balanced_accuracy", "f1_score", "ap", "auc", "mcc"]

def list_files_recursively(path, metrics):
    all_files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            all_files += list_files_recursively(full_path, metrics)  # recurse into subdir
        else:
            if entry.endswith(".csv") and any(metric in entry for metric in metrics):
                all_files.append(full_path)
    return all_files

ref = "epanechnikov_adaptive_trimmed"
segm_ablations = ["epanechnikov_adaptive", "epanechnikov_scott", "gaussian_adaptive", "gaussian_scott"]
classif_ablations = ["epanechnikov_scott", "gaussian_adaptive", "gaussian_scott"]

def process_ablation_results_diff(ablations, ref, metrics, plot_histograms=True):
    paths = {ref: list_files_recursively(f"CIs_project/results_ablations/{ref}", metrics)}
    for ablation in ablations:
        paths[ablation] = list_files_recursively(f"CIs_project/results_ablations/{ablation}", metrics)

    dfs = {}
    for ablation, files in paths.items():
        dfs[ablation] = {}
        for file in files:
            df = pd.read_csv(file)
            base_path, filename = os.path.split(file)
            dfs[ablation][filename] = df

    abs_diffs = {ablation: [] for ablation in ablations}

    key_cols = ["subtask", "alg_name", "n"]
    methods = ["basic", "percentile", "bca"]

    for ablation in ablations:
        for file in paths[ablation]:
            base_path, filename = os.path.split(file)
            if filename in dfs[ref]:
                df_ref = dfs[ref][filename].sort_values(by=key_cols).reset_index(drop=True)
                non_key_cols = [col for col in df_ref.columns if col not in key_cols]
                df_ablation = dfs[ablation][filename].sort_values(by=key_cols).reset_index(drop=True)
                diff_df = df_ref.copy()
                for col in non_key_cols:
                    diff_df[col] = (df_ref[col] - df_ablation[col])
                    diff_df[col] = diff_df[col].abs()
                abs_diffs[ablation].append(diff_df)

    cov_diffs = {ablation: {method: [] for method in methods} for ablation in ablations}
    width_diffs = {ablation: {method: [] for method in methods} for ablation in ablations}

    for ablation, diff_dfs in abs_diffs.items():
        for diff_df in diff_dfs:
            numerical_cols = diff_df[non_key_cols]
            meandiffs = numerical_cols.mean()
            for method in methods:
                cov_diffs[ablation][method].append(meandiffs[f'coverage_{method}'])
                width_diffs[ablation][method].append(meandiffs[f'width_{method}'])

    if plot_histograms:
        fig, axes = plt.subplots(len(ablations), 2, figsize=(16, 24))

        for i, ablation in enumerate(ablations):
            max_diff = max(max(cov_diffs[ablation][method]) for method in methods)
            bins = np.linspace(0, max_diff+1e-6, 50)
            for method in methods:
                axes[i, 0].hist(cov_diffs[ablation][method], label=f'Coverage {method}', alpha=0.3, bins=bins)
            axes[i, 0].set_title(f'Coverage Differences for {ablation}')
            axes[i, 0].set_xlabel('Mean Coverage Difference (1 value per metric and average/summary statistic)')
            axes[i, 0].legend()

            max_diff = max(max(width_diffs[ablation][method]) for method in methods)
            bins = np.linspace(0, max_diff+1e-6, 50)
            for method in methods:
                axes[i, 1].hist(width_diffs[ablation][method], label=f'Width {method}', alpha=0.3, bins=bins)
            axes[i, 1].set_title(f'Width Differences for {ablation}')
            axes[i, 1].set_xlabel('Mean Width Difference (1 value per metric and average/summary statistic)')
            axes[i, 1].legend()
        plt.show()
        plt.close()

    # Build a Markdown table with means ± std
    header = "| Ablation | Method | Coverage Diff | Width Diff |\n"
    header += "|----------|--------|---------------|------------|\n"

    body = ""
    for ablation in ablations:
        for method in methods:
            mean_cov_diff = np.mean(cov_diffs[ablation][method])
            std_cov_diff = np.std(cov_diffs[ablation][method])
            mean_width_diff = np.mean(width_diffs[ablation][method])
            std_width_diff = np.std(width_diffs[ablation][method])
            body += (
                f"| {ablation} | {method} | "
                f"{mean_cov_diff:.4f} ± {std_cov_diff:.4f} | "
                f"{mean_width_diff:.4f} ± {std_width_diff:.4f} |\n"
            )

    markdown_table = header + body
    print(markdown_table)

def process_ablation_results_relative_diff(ablations, ref, metrics, plot_histograms=True):
    paths = {ref: list_files_recursively(f"CIs_project/results_ablations/{ref}", metrics)}
    for ablation in ablations:
        paths[ablation] = list_files_recursively(f"CIs_project/results_ablations/{ablation}", metrics)

    dfs = {}
    for ablation, files in paths.items():
        dfs[ablation] = {}
        for file in files:
            df = pd.read_csv(file)
            df = df[df["n"] <= 250]  # Filter out rows where n > 250
            base_path, filename = os.path.split(file)
            dfs[ablation][filename] = df

    abs_diffs = {ablation: [] for ablation in ablations}

    key_cols = ["subtask", "alg_name", "n"]
    methods = ["basic", "percentile", "bca"]

    for ablation in ablations:
        for file in paths[ablation]:
            base_path, filename = os.path.split(file)
            if filename in dfs[ref]:
                df_ref = dfs[ref][filename].sort_values(by=key_cols).reset_index(drop=True)
                non_key_cols = [col for col in df_ref.columns if col not in key_cols]
                df_ablation = dfs[ablation][filename].sort_values(by=key_cols).reset_index(drop=True)
                diff_df = df_ref.copy()
                for col in non_key_cols:
                    diff_df[col] = (df_ref[col] - df_ablation[col])
                    if "width" in col:
                        diff_df[col] = diff_df[col] / df_ref[col].replace(0, np.nan)  # Avoid division by zero
                    diff_df[col] = diff_df[col].abs()
                if "balanced_accuracy" in file and ablation=="gaussian_adaptive":
                    indices = diff_df.sort_values(by='width_bca', ascending=False).index[:20]
                    print(f"Top 20 differences for {ablation} in {file}:")
                    print(diff_df.loc[indices, ["subtask", "alg_name", "n", "width_bca"]])
                    print(df_ref.loc[indices, ["subtask", "alg_name", "n", "width_bca"]])
                abs_diffs[ablation].append(diff_df)

    cov_diffs = {ablation: {method: [] for method in methods} for ablation in ablations}
    width_diffs = {ablation: {method: [] for method in methods} for ablation in ablations}

    for ablation, diff_dfs in abs_diffs.items():
        for diff_df in diff_dfs:
            numerical_cols = diff_df[non_key_cols]
            meandiffs = numerical_cols.mean()
            for method in methods:
                cov_diffs[ablation][method].append(meandiffs[f'coverage_{method}'])
                width_diffs[ablation][method].append(meandiffs[f'width_{method}'])

    if plot_histograms:
        fig, axes = plt.subplots(len(ablations), 2, figsize=(16, 24))

        for i, ablation in enumerate(ablations):
            max_diff = max(max(cov_diffs[ablation][method]) for method in methods)
            bins = np.linspace(0, max_diff+1e-6, 50)
            for method in methods:
                axes[i, 0].hist(cov_diffs[ablation][method], label=f'Coverage {method}', alpha=0.3, bins=bins)
            axes[i, 0].set_title(f'Coverage Differences for {ablation}')
            axes[i, 0].set_xlabel('Mean Coverage Difference (1 value per metric and average/summary statistic)')
            axes[i, 0].legend()
            
            max_diff = max(max(width_diffs[ablation][method]) for method in methods)
            bins = np.linspace(0, max_diff+1e-6, 50)
            for method in methods:
                axes[i, 1].hist(width_diffs[ablation][method], label=f'Width {method}', alpha=0.3, bins=bins)
            axes[i, 1].set_title(f'Relative Width Differences for {ablation}')
            axes[i, 1].set_xlabel('Mean Relative Width Difference (1 value per metric and average/summary statistic)')
            axes[i, 1].legend()
        plt.show()
        plt.close()

    # Build a Markdown table with means ± std
    header = "| Ablation | Method | Coverage Diff | Width Relative Diff |\n"
    header += "|----------|--------|---------------|------------|\n"

    body = ""
    for ablation in ablations:
        for method in methods:
            mean_cov_diff = np.mean(cov_diffs[ablation][method])
            std_cov_diff = np.std(cov_diffs[ablation][method])
            mean_width_diff = np.mean(width_diffs[ablation][method])
            std_width_diff = np.std(width_diffs[ablation][method])
            body += (
                f"| {ablation} | {method} | "
                f"{mean_cov_diff:.4f} ± {std_cov_diff:.4f} | "
                f"{mean_width_diff:.4f} ± {std_width_diff:.4f} |\n"
            )

    markdown_table = header + body
    print(markdown_table)

if __name__ == "__main__":

    print("Ablation results for bounded metrics (width not normalized):\n")
    process_ablation_results_diff(segm_ablations, ref, bounded_metrics, plot_histograms=False)
    print("Ablation results for unbounded metrics (width not normalized):\n")
    process_ablation_results_diff(segm_ablations, ref, unbounded_metrics, plot_histograms=False)
    print("Ablation results for all segmentation metrics (width not normalized):\n")
    process_ablation_results_diff(segm_ablations, ref, all_metrics, plot_histograms=False)
    print("Ablation results for bounded metrics (width normalized):\n")
    process_ablation_results_relative_diff(segm_ablations, ref, bounded_metrics, plot_histograms=False)
    print("Ablation results for unbounded metrics (width normalized):\n")
    process_ablation_results_relative_diff(segm_ablations, ref, unbounded_metrics, plot_histograms=False)
    print("Ablation results for all segmentation metrics (width normalized):\n")
    process_ablation_results_relative_diff(segm_ablations, ref, all_metrics, plot_histograms=False)


    print("Ablation results for classif metrics (width not normalized):\n")
    process_ablation_results_diff(classif_ablations, ref, classif_metrics, plot_histograms=False)
    print("Ablation results for classif metrics (width normalized):\n")
    process_ablation_results_relative_diff(classif_ablations, ref, classif_metrics, plot_histograms=False)