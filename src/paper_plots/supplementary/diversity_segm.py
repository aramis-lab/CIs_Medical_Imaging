import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import skew, kurtosis
from matplotlib.font_manager import FontProperties

from ..plot_utils import metric_labels

def compute_descriptive_stats(root_folder:str):

    data = pd.read_csv(os.path.join(root_folder, "data_matrix_grandchallenge_all.csv"))

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
                    'kurtosis': kurtosis(values, fisher=False),
                    'Mean': np.mean(values), 
                    'Standard deviation': np.std(values, ddof=1),
                    "total_number": count_total
                    }  
        
                    results.append(value)
        
    results_df=pd.DataFrame(results)
    return results_df


def plot_descriptive_stats_segm(root_folder:str, output_path:str):

    plt.rcdefaults()

    results_df = compute_descriptive_stats(root_folder)
    metrics=results_df['Metric'].unique()

    bounded_metrics = ['dsc', 'iou', 'nsd', 'boundary_iou', 'cldice']
    unbounded_metrics = ['assd', 'masd', 'hd' , 'hd_perc']
    metrics_order = bounded_metrics + unbounded_metrics

    results_df['Metric'] = pd.Categorical(results_df['Metric'], categories=metrics_order, ordered=True)
    results_df = results_df.sort_values('Metric')

    palette = sns.color_palette("colorblind", len(metrics))
    color_dict = dict(zip(metrics_order, palette))
    color_dict.update({
        "iou": (31/255, 119/255, 180/255),        # #1f77b4 -> RGB normalized
        "boundary_iou": (74/255, 144/255, 226/255),  # #4a90e2 -> RGB normalized
        "cldice": (1/255, 104/255, 4/255)         # #016804 -> RGB normalized
    })
    def darken_color(color, amount=0.8):
        return tuple(min(max(c * amount, 0), 1) for c in color)

    dark_color_dict = {k: darken_color(v, 0.75) for k, v in color_dict.items()}

    _, axs = plt.subplots(4,1, figsize=(16, 24))

    ### --- Plot Mean --- ###
    ax = axs[0]

    # Primary axis (left): bounded metrics only
    sns.boxplot(x='Metric', y='Mean', data=results_df[results_df['Metric'].isin(bounded_metrics)], hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax, dodge=False)
    sns.stripplot(x='Metric', y='Mean', data=results_df[results_df['Metric'].isin(bounded_metrics)], hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax, dodge=False)
    ax.set_ylabel('Mean (bounded metrics)', fontsize=20)
    ax.set_ylim(0, 1.01)

    # Twin axis (right): unbounded metrics
    ax2 = ax.twinx()
    sns.boxplot(x='Metric', y='Mean', data=results_df[results_df['Metric'].isin(unbounded_metrics)], hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax2, dodge=False)
    sns.stripplot(x='Metric', y='Mean', data=results_df[results_df['Metric'].isin(unbounded_metrics)], hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax2, dodge=False)
    ax2.set_ylabel('Mean (unbounded metrics)', fontsize=20)
    ax2.set_ylim(0, None)
    ax.vlines([len(bounded_metrics)-0.5], ymin=ax.get_ylim()[0], ymax=ax2.get_ylim()[1], color='gray', linestyle='--', linewidth=1.5)

    # Common formatting
    ax.set_xticklabels([metric_labels[m] for m in metrics_order], fontsize=18)
    ax.set_title('Mean values across all instances',weight='bold', fontsize=20)
    ax.set_xlabel('Metric', fontsize=20)

    ### --- Plot Standard Error --- ###
    ax = axs[1]
    # Primary axis (left): bounded metrics only
    sns.boxplot(x='Metric', y='Standard deviation', data=results_df[results_df['Metric'].isin(bounded_metrics)], hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax, dodge=False)
    sns.stripplot(x='Metric', y='Standard deviation', data=results_df[results_df['Metric'].isin(bounded_metrics)], hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax, dodge=False)
    ax.set_ylabel('Standard Deviation (bounded metrics)',fontsize=20)
    ax.set_ylim(0, 1.01)

    # Twin axis (right): unbounded metrics
    ax2 = ax.twinx()
    sns.boxplot(x='Metric', y='Standard deviation', data=results_df[results_df['Metric'].isin(unbounded_metrics)], hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax2, dodge=False)
    sns.stripplot(x='Metric', y='Standard deviation', data=results_df[results_df['Metric'].isin(unbounded_metrics)], hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax2, dodge=False)
    ax2.set_ylabel('Standard Deviation (unbounded metrics)',fontsize=20)
    ax2.set_ylim(0, None)
    ax.vlines([len(bounded_metrics)-0.5], ymin=ax.get_ylim()[0], ymax=ax2.get_ylim()[1], color='gray', linestyle='--', linewidth=1.5)

    ax.set_xticklabels([metric_labels[m] for m in metrics_order], fontsize=18)
    ax.set_title('Standard Deviation values across all instances',weight='bold',fontsize=20)
    ax.set_xlabel('Metric', fontsize=20)

    # Plot Skewness
    ax = axs[2]
    ax.axhspan(-1, 1, color='#009409', alpha=0.3, label='Normal range')
    ax.axhline(2, color='red', linestyle='--', label='Highly skewed on the left', linewidth=1.7)
    ax.axhline(-2, color='red', linestyle='-', label='Highly skewed on the right', linewidth=1.7)
    sns.boxplot(x='Metric', y='skewness', data=results_df, hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax)
    sns.stripplot(x='Metric', y='skewness', data=results_df, hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax)

    ax.legend(title="Typical skewness values",
          
               frameon=False,
               fontsize=18,
               title_fontproperties=FontProperties(weight='bold', size=18), loc='upper left'
            #    bbox_to_anchor=(1.23, 0.8)
              )
    ax.set_xticklabels([metric_labels[m] for m in metrics_order], fontsize=18)
    ax.set_title('Skewness values across all instances', weight='bold', fontsize=20)
    ax.set_ylabel('Skewness',fontsize=20)
    ax.set_xlabel('Metric',fontsize=20)
    ax.set_ylim(-6, 10)

    # Plot Kurtosis
    ax = axs[3]
    ax.axhspan(2, 7, color='#009409', alpha=0.3, label='Normal range')
    ax.axhline(7, color='red', linestyle='--', label='Heavy tails', linewidth=1.7)
    ax.axhline(2, color='red', linestyle='-', label='Light tails', linewidth=1.7)
    sns.boxplot(x='Metric', y='kurtosis', data=results_df, hue='Metric', showfliers=False, palette=color_dict, linewidth=1, ax=ax)
    sns.stripplot(x='Metric', y='kurtosis', data=results_df, hue='Metric', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, ax=ax)

    ax.legend(title="Typical kurtosis values",
             
               frameon=False,
               fontsize=18,
               title_fontproperties=FontProperties(weight='bold', size=18),loc='upper left'
            #    bbox_to_anchor=(1.18, 0.8)
              )
    ax.set_xticklabels([metric_labels[m] for m in metrics_order], fontsize=18)
    ax.set_title('Kurtosis values across all instances', weight='bold', fontsize=20)
    ax.set_ylabel('Kurtosis',fontsize=20)
    ax.set_xlabel('Metric',fontsize=20)
    ax.set_ylim(0, 50)

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute and plot descriptive statistics for segmentation metrics.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data matrix CSV file.")
    parser.add_argument("--output_path", type=str, required=False, help="Output path for the descriptive statistics plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/skew_kurt_segm.pdf")

    plot_descriptive_stats_segm(root_folder, output_path)

if __name__ == "__main__":
    main()