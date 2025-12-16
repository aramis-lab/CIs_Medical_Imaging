import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.font_manager import FontProperties


def multivariate_skewness_kurtosis(logits, eps=1e-5):
    X = np.asarray(logits)
    n_samples, d = X.shape

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Regularized covariance
    cov = np.cov(X_centered, rowvar=False)
    cov += eps * np.eye(d)
    
    # Use Cholesky to compute whitening matrix
    L = np.linalg.cholesky(cov)
    cov_inv_sqrt = np.linalg.inv(L).T
    X_standardized = X_centered @ cov_inv_sqrt

    # α²: skewness
    alpha_squared = 0.0
    for i in range(d):
        for j in range(d):
            for k in range(d):
                e_ijk = np.mean(X_standardized[:, i] * X_standardized[:, j] * X_standardized[:, k])
                alpha_squared += e_ijk ** 2

    # β*: kurtosis
    norm_squared = np.sum(X_standardized ** 2, axis=1)
    beta_star = np.mean(norm_squared ** 2) 

    return {
        'Skewness': np.sqrt(alpha_squared),
        'Kurtosis': beta_star
    }
def parse_vector_string(vec_str):
    # Remove curly braces and split by comma
    vec_str = vec_str.strip('{}')
    parts = vec_str.split(',')
    # Convert each part to float (strip whitespace just in case)
    return [float(x.strip()) for x in parts]

def compute_descriptive_stats(root_folder:str):

    data = pd.read_csv(os.path.join(root_folder, "data_matrix_classification.csv"))

    results = []

    subtasks=data['subtask'].unique()

    for task in subtasks: 
        data_task=data[data['subtask']==task]
        algos=data_task['alg_name'].unique()
        for alg in algos:
            data_alg=data_task[data_task['alg_name']==alg]
            logits=data_alg['logits']
    
            list_of_lists = logits.apply(parse_vector_string).tolist()
            max_len = max(len(vec) for vec in list_of_lists)
            list_of_lists = [vec for vec in list_of_lists if len(vec) == max_len]

            matrix_logits = np.array(list_of_lists)
            
            # Convert list of lists to numpy matrix
        
            stats = multivariate_skewness_kurtosis(matrix_logits)
        
            results.append({
                "task": task,
                "algo": alg,
                "skewness":stats['Skewness'], 
                "Kurtosis":stats['Kurtosis'], 
                "dimension": matrix_logits.shape[1]
            })
        
    results_df=pd.DataFrame(results)
    return results_df


def plot_descriptive_stats_classif(root_folder:str, output_path:str):

    plt.rcdefaults()

    results_df = compute_descriptive_stats(root_folder)
    dimensions = results_df['dimension'].unique()
    dimension_order = sorted(results_df['dimension'].unique())

    palette = sns.color_palette("colorblind", len(dimensions))
    color_dict = dict(zip(dimensions, palette))
    def darken_color(color, amount=0.8):
        return tuple(min(max(c * amount, 0), 1) for c in color)

    dark_color_dict = {k: darken_color(v, 0.75) for k, v in color_dict.items()}

    # Plot Skewness
    fig, axs = plt.subplots(2,1, figsize=(15, 12))
    ax = axs[0]
    ax.axhspan(0, 1, color='#009409', alpha=0.3, label='Normal range (0-1)')
    sns.boxplot(x='dimension', y='skewness', data=results_df, hue='dimension', showfliers=False, palette=color_dict, linewidth=1, ax=ax, legend=False, order=dimension_order)
    sns.stripplot(x='dimension', y='skewness', data=results_df, hue='dimension', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, order=dimension_order, ax=ax)

    ax.legend(title="Typical skewness values",
            loc='upper right',
            frameon=False,
            fontsize=10,
            title_fontproperties=FontProperties(weight='bold'),
            bbox_to_anchor=(1.23, 0.8)
        )
    ax.set_title('Skewness values across all dimensions', weight='bold', fontsize=15)
    ax.set_ylabel('Skewness', weight='bold')
    ax.set_xlabel('Dimension', weight='bold')
    ax.set_ylim(-1,12)

    # Plot Kurtosis
    ax = axs[1]
    # Plot dynamic normal range [0, d(d+2)] with per-dimension colors
    for i, d in enumerate(dimension_order):
        d=int(d)
        d_kurt_max = d * (d + 2)
        band_width = d_kurt_max * 0.1
        ax.axhspan(d_kurt_max-band_width, d_kurt_max+band_width, xmin=(i ) / len(dimensions), xmax=(i + 1) / len(dimensions),
                    color=color_dict[d], alpha=0.3, label=f'Normal range for d={d}')

    sns.boxplot(x='dimension', y='Kurtosis', data=results_df, hue='dimension',
            showfliers=False, palette=color_dict, legend=False, order=dimension_order, ax=ax)
    sns.stripplot(x='dimension', y='Kurtosis', data=results_df, hue='dimension',
                jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, order= dimension_order, ax=ax)

    ax.legend(title="Typical kurtosis values",
        loc='upper right',
        frameon=False,
        fontsize=10,
        title_fontproperties=FontProperties(weight='bold'),
        bbox_to_anchor=(1.18, 0.8)
    )

    ax.set_title('Kurtosis values across all dimensions', weight='bold', fontsize=15)
    ax.set_ylabel('Kurtosis', weight='bold')
    ax.set_xlabel('Dimension', weight='bold')
    ax.set_ylim(-1,150)

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute and plot descriptive statistics for classification metrics.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data matrix CSV file.")
    parser.add_argument("--output_path", type=str, required=False, help="Output path for the descriptive statistics plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/skew_kurt_classif.pdf")

    plot_descriptive_stats_classif(root_folder, output_path)

if __name__ == "__main__":
    main()