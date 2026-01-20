import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
import os
import argparse

from ..plot_utils import metric_labels, stat_labels
from ..df_loaders import extract_df_segm_cov

def plot_fig8_metrics(root_folder:str, output_path:str):

    plt.rcdefaults()

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")
    file_prefix_segm = "aggregated_results"
    metrics_segm =['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice', 'assd', 'masd', 'hd', 'hd_perc']
    stats_segm = ["mean"]

    df_segm = extract_df_segm_cov(folder_path_segm, file_prefix_segm, metrics_segm, stats_segm)

    palette = sns.color_palette("colorblind", len(metrics_segm))
    color_dict = dict(zip(metrics_segm, palette))
    color_dict.update({
        "iou": (31/255, 119/255, 180/255),        # #1f77b4 -> RGB normalized
        "boundary_iou": (74/255, 144/255, 226/255),  # #4a90e2 -> RGB normalized
        "cldice": (1/255, 104/255, 4/255)         # #016804 -> RGB normalized
    })

    fig = plt.figure(figsize=(25,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.7, 1])

    ax_left = fig.add_subplot(gs[:, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bot_right = fig.add_subplot(gs[1, 1])

    data_method=df_segm[df_segm['method']=='percentile']

    for metric, df_metric in data_method.groupby('metric'):
        medians = data_method[data_method['metric']==metric].groupby('n')['coverage'].median().values
        q1 = data_method[data_method['metric']==metric].groupby('n')['coverage'].quantile(0.25).values
        q3 = data_method[data_method['metric']==metric].groupby('n')['coverage'].quantile(0.75).values
        ax_left.plot(df_metric['n'].unique(), medians, marker='o', label=metric_labels[metric], color=color_dict[metric], linewidth=4, markersize=10)
        ax_left.fill_between(df_metric['n'].unique(), q1, q3, alpha=0.2, color=color_dict[metric])
    sorted_metrics = (
    data_method.groupby('metric')["coverage"]
    .median()
    .sort_values(ascending=False)
    .index
    )

    ax_left.set_title(f'Summary statistic : {stat_labels["mean"]}', weight='bold', fontsize=28)
    ax_left.set_xlabel('Sample size', weight='bold', fontsize=26)
    ax_left.set_ylabel('Coverage (%)', weight='bold',fontsize=26)
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    ax_left.grid(True, axis='y')
    # Sort the legend by median value at n=125
    legend_order = (
        data_method[data_method['n'] == 125]
        .groupby('metric')['coverage']
        .median()
        .sort_values(ascending=False)
        .index
    )
    legend_order = pd.Index([metric_labels[m] for m in legend_order])

    handles, labels = ax_left.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: legend_order.get_loc(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax_left.legend(sorted_handles, sorted_labels, fontsize=25, loc="lower right")
    ax_left.tick_params(axis='x', labelsize=28)
    ax_left.tick_params(axis='y', labelsize=28)

    data= pd.read_csv(os.path.join(root_folder, "data_matrix_grandchallenge_all.csv"), sep=';')
    results=[]
    metrics=data['score'].unique()
    for score in metrics:
        df=data[data['score']==score]
        algos=df['alg_name'].unique()
        score=df['score'].unique()[0]
        count_total=0
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
                    count_total+=1
                    value={
                    'Metric':score,
                    'algorithm': alg,
                    'subtask': task,
                    'skewness': skew(values),
                    'kurtosis': kurtosis(values),
                    'Mean': np.mean(values), 
                    'Standard error': np.std(values, ddof=1),
                    "total_number": count_total
                    }  
        
                    results.append(value)
    results_df= pd.DataFrame(results)
    for score in metrics: 
        mask = results_df['Metric'] == score
        max_val = results_df.loc[mask, 'total_number'].max()
        results_df.loc[mask, 'total_number'] = max_val
    results = []

    for task in df_segm['task'].unique():
        df_task = df_segm[df_segm['task'] == task]
        for algo in df_task['algo'].unique():
            df_algo = df_task[df_task['algo'] == algo]
            for metric in df_algo['metric'].unique():
                df_metric_stat = df_algo[(df_algo['metric'] == metric) & (df_algo['stat']=='mean')]
                for method in df_metric_stat['method'].unique():
                    df_metric_stat_method = df_metric_stat[df_metric_stat['method'] == method]
                    df_metric_stat_method = df_metric_stat_method.sort_values(by='n')
                    n_values = df_metric_stat_method['n'].to_numpy()
                    coverages = df_metric_stat_method['coverage'].to_numpy()
                    Y = 0.95 - coverages
                    X = np.vstack([1/n_values]).T
                    beta2, res = np.linalg.lstsq(X, Y, rcond=None)[:2]
                    rel_error = np.abs(res[0]) / np.sum(Y**2)
                    new_row = {
                        'task': task,
                        'algo': algo,
                        'metric': metric,
                        'stat': "mean",
                        'method': method,
                        'beta2': beta2[0],
                        'R2': rel_error
                    }
                    results.append(new_row)
    df_fit_results = pd.DataFrame(results)

    # Add skewness to df_fit_results
    df_fit_results = df_fit_results.merge(results_df[['Metric', 'algorithm', 'subtask', 'skewness', "kurtosis"]], 
                                        left_on=['metric', 'algo', 'task'], 
                                        right_on=['Metric', 'algorithm', 'subtask'], 
                                        how='left')

    df_fit_results.drop(columns=['Metric', 'algorithm', 'subtask'], inplace=True)

    df_fit_results = df_fit_results[df_fit_results['method']=="percentile"]

    plt.rcdefaults()
    sns.reset_defaults()

    def darken_color(color, amount=0.8):
        return tuple(min(max(c * amount, 0), 1) for c in color)

    dark_color_dict = {k: darken_color(v, 0.75) for k, v in color_dict.items()}

    df_fit_results["abs_skewness"] = df_fit_results["skewness"].abs()
    # Sort metrics by descending absolute skewness
    sorted_metrics = df_fit_results.groupby('metric')["abs_skewness"].median().sort_values(ascending=False).index

    label_fontsize = 23
    legend_fontsize = 18
    title_fontsize = 25
    tick_fontsize = 18

    # Plot Absolute Skewness
    ax_top_right.axhspan(0, 1, color='#009409', alpha=0.3, label='Normal range')
    ax_top_right.axhline(2, color='red', linestyle='--', label='Highly skewed', linewidth=1.7)
    sns.boxplot(x='metric', y='abs_skewness', data=df_fit_results, order=sorted_metrics, hue="metric", showfliers=False, palette=color_dict, linewidth=1, ax=ax_top_right)
    sns.stripplot(x='metric', y='abs_skewness', data=df_fit_results, order=sorted_metrics, hue="metric", jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax_top_right)

    ax_top_right.set_title('Absolute skewness values', weight='bold', fontsize=title_fontsize)
    ax_top_right.set_ylabel('Absolute skewness', weight='bold', fontsize=label_fontsize)
    ax_top_right.set_xlabel('Metric', weight='bold', fontsize=label_fontsize)
    ax_top_right.legend(title="Typical values",
                loc='upper right',
                frameon=False,
                fontsize=legend_fontsize,
                title_fontproperties=FontProperties(weight='bold'))
    ax_top_right.tick_params(axis='y', labelsize=tick_fontsize)
    ax_top_right.set_ylim(-0.05, 10)
    ax_top_right.tick_params(axis='x', labelsize=tick_fontsize-2)
    ax_top_right.tick_params(axis='y', labelsize=tick_fontsize)
    xticks = ax_top_right.get_xticklabels()
    ax_top_right.set_xticklabels([metric_labels[metric.get_text()] for metric in xticks])

    sorted_metrics = df_fit_results.groupby('metric')["beta2"].median().sort_values(ascending=False).index
    # Plot Coverage Error Constant
    sns.boxplot(x='metric', y='beta2', data=df_fit_results, order=sorted_metrics, hue="metric", showfliers=False, palette=color_dict, linewidth=1, ax=ax_bot_right)
    sns.stripplot(x='metric', y='beta2', data=df_fit_results, order=sorted_metrics, hue="metric", jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax_bot_right)

    sorted_metrics_ccp = df_fit_results.groupby('metric')["beta2"].median().sort_values(ascending=False).index

    # Plot Coverage Error Constant
    sns.boxplot(x='metric', y='beta2', data=df_fit_results, order=sorted_metrics_ccp, hue="metric", showfliers=False, palette=color_dict, linewidth=1, ax=ax_bot_right)
    sns.stripplot(x='metric', y='beta2', data=df_fit_results, order=sorted_metrics_ccp, hue="metric", jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax_bot_right)

    ax_bot_right.set_title('CCP values \n percentile method for CI of the mean', weight='bold', fontsize=title_fontsize)
    ax_bot_right.set_ylabel('Coverage convergence pace', weight='bold', fontsize=label_fontsize)
    ax_bot_right.set_xlabel('Metric', weight='bold', fontsize=label_fontsize)
    ax_bot_right.set_ylim(-0.05, 6)
    ax_bot_right.tick_params(axis='x', labelsize=tick_fontsize-2)
    ax_bot_right.tick_params(axis='y', labelsize=tick_fontsize)
    # Adjust x-axis labels for both subplots
    xticks = ax_bot_right.get_xticklabels()
    ax_bot_right.set_xticklabels([metric_labels[metric.get_text()] for metric in xticks])

    for ax, letter in zip([ax_left, ax_top_right, ax_bot_right], ['A', 'B', 'C']):
        ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                fontsize=36, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 7 segmentation metrics.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig8_metrics.pdf")

    # Call your plotting function
    plot_fig8_metrics(root_folder, output_path)

if __name__=="__main__":
    main()