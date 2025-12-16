import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

from ..df_loaders import extract_df_segm_cov
from ..plot_utils import metric_labels

def perform_fits_segm(df_segm, metrics, stats):
    results = []
    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in metrics:
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

def plot_rel_error_CCP_segm(root_folder:str, output_path:str):

    plt.rcdefaults()

    metrics_segm = ['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice', 'assd', 'masd', 'hd', 'hd_perc']
    palette = sns.color_palette("colorblind", len(metrics_segm))
    color_dict = dict(zip(metrics_segm, palette))
    color_dict.update({
        "iou": (31/255, 119/255, 180/255),        # #1f77b4 -> RGB normalized
        "boundary_iou": (74/255, 144/255, 226/255),  # #4a90e2 -> RGB normalized
        "cldice": (1/255, 104/255, 4/255)         # #016804 -> RGB normalized
    })
    stats_segm = ["mean"]

    def darken_color(color, amount=0.8):
        return tuple(min(max(c * amount, 0), 1) for c in color)

    dark_color_dict = {k: darken_color(v, 0.75) for k, v in color_dict.items()}

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)
    df_segm = df_segm[df_segm["method"]=="percentile"]

    df_fit_segm = perform_fits_segm(df_segm, metrics_segm, stats_segm)

    sorted_metrics = df_fit_segm.groupby('metric')["beta2"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # Plot R2
    sns.boxplot(x='metric', y='relative_error', data=df_fit_segm, order=sorted_metrics, hue="metric", showfliers=False, palette=color_dict, linewidth=1, ax=ax)
    sns.stripplot(x='metric', y='relative_error', data=df_fit_segm, order=sorted_metrics, hue="metric", jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax)

    ax.set_title('', weight='bold', fontsize=16)
    ax.set_ylabel('Relative error', weight='bold', fontsize=14)
    ax.set_xlabel('Metric', weight='bold', fontsize=14)
    ax.set_ylim(0, 0.06)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Adjust x-axis labels for both subplots
    xticks = ax.get_xticklabels()
    ax.set_xticklabels([metric_labels[metric.get_text()] for metric in xticks])
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True,
                        help='Root folder containing results_metrics_segm directory')
    parser.add_argument('--output_path', type=str,help='Path to save the output plot')
    args = parser.parse_args()

    root_folder = args.root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs", "supplementary", "relative_error_CCP_segm.pdf")

    plot_rel_error_CCP_segm(root_folder, output_path)

if __name__ == "__main__":
    main()