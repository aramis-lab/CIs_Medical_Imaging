import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.font_manager import FontProperties
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.special import softmax
import matplotlib.gridspec as gridspec
from ..plot_utils import metric_labels


def multivariate_skewness_kurtosis(logits, eps=1e-5):
    X = np.asarray(logits)
    _, d = X.shape

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
            to_exclude = [i for i, vec in enumerate(list_of_lists) if len(vec) < max_len]
            list_of_lists = [vec for vec in list_of_lists if len(vec) == max_len]

            matrix_logits = np.array(list_of_lists)
            y_score = softmax(matrix_logits, axis=1)
            if y_score.shape[1]==2:
                y_score = y_score[:,1]
                y_pred = (y_score >= 0.5).astype(int)
            else:
                y_pred = np.argmax(y_score, axis=1)
            targets=data_alg['target'].values
            targets = np.array([t for i, t in enumerate(targets) if i not in to_exclude])

            stats = multivariate_skewness_kurtosis(matrix_logits)
            acc = accuracy_score(targets, y_pred)
            bal_acc = balanced_accuracy_score(targets, y_pred)

            micro_auc = roc_auc_score(targets, y_score, multi_class='ovr', average='micro')
            macro_auc = roc_auc_score(targets, y_score, multi_class='ovr', average='macro')

            micro_f1 = f1_score(targets, y_pred, average='micro')
            macro_f1 = f1_score(targets, y_pred, average='macro')

            micro_ap = average_precision_score(targets, y_score, average='micro')
            macro_ap = average_precision_score(targets, y_score, average='macro')

            mcc = matthews_corrcoef(targets, y_pred)
        
            results.append({
                "task": task,
                "algo": alg,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "micro_auc": micro_auc,
                "macro_auc": macro_auc,
                "micro_f1_score": micro_f1,
                "macro_f1_score": macro_f1,
                "micro_ap": micro_ap,
                "macro_ap": macro_ap,
                "mcc": abs(mcc),
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
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])  # first row a bit taller

    ax = fig.add_subplot(gs[0, :])
    # Boxplot + jitterplot for each metric
    metrics = [
        'accuracy', 'balanced_accuracy',
        'micro_auc', 'macro_auc',
        'micro_f1_score', 'macro_f1_score',
        'micro_ap', 'macro_ap',
        'mcc'
    ]
    long_df = results_df.melt(
        id_vars=['dimension'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

    sns.boxplot(
        x='metric', y='value',
        data=long_df,
        hue='dimension',
        showfliers=False,
        palette=color_dict,
        linewidth=1,
        order=metrics,
        hue_order=dimension_order,
        ax=ax,
        legend=True
    )
    sns.stripplot(
        x='metric', y='value',
        data=long_df,
        hue='dimension',
        dodge=True,
        jitter=True,
        alpha=0.6,
        palette=dark_color_dict,
        order=metrics,
        hue_order=dimension_order,
        ax=ax,
        legend=False
    )

    ax.set_title('Classification metric distributions', weight='bold', fontsize=15)
    ax.set_ylabel('Score', weight='bold', fontsize=14)
    ax.set_xlabel('', weight='bold', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    labels = []
    for label in ax.get_xticklabels():
        if "micro" in label.get_text():
            metric = label.get_text().replace("micro_", "")
            labels.append("Micro " + metric_labels.get(metric, metric).replace("_", " "))
        elif "macro" in label.get_text():
            metric = label.get_text().replace("macro_", "")
            labels.append("Macro " + metric_labels.get(metric, metric).replace("_", " "))
        else:
            labels.append(metric_labels.get(label.get_text(), label.get_text()).replace("_", " "))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Dimension",
        loc='upper right',
        frameon=False,
        fontsize=14,
        title_fontproperties=FontProperties(weight='bold')
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.axhspan(0, 1, color='#009409', alpha=0.3, label='Normal range (0-1)')
    sns.boxplot(x='dimension', y='skewness', data=results_df, hue='dimension', showfliers=False, palette=color_dict, linewidth=1, ax=ax, legend=False, order=dimension_order)
    sns.stripplot(x='dimension', y='skewness', data=results_df, hue='dimension', jitter=True, alpha=0.6, palette=dark_color_dict, legend=False, order=dimension_order, ax=ax)

    ax.legend(title="Typical skewness values",
            loc='upper left',
            frameon=False,
            fontsize=10,
            title_fontproperties=FontProperties(weight='bold')
        )
    ax.set_title('Skewness values across all dimensions', weight='bold', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel('Skewness', weight='bold', fontsize=14)
    ax.set_xlabel('Dimension', weight='bold', fontsize=14)
    ax.set_ylim(-1,12)

    # Plot Kurtosis
    ax = fig.add_subplot(gs[1, 1])
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
        loc='upper left',
        frameon=False,
        fontsize=10,
        title_fontproperties=FontProperties(weight='bold')
    )

    ax.set_title('Kurtosis values across all dimensions', weight='bold', fontsize=15)
    ax.set_ylabel('Kurtosis', weight='bold', fontsize=14)
    ax.set_xlabel('Dimension', weight='bold', fontsize=14)
    ax.set_ylim(-1,150)
    ax.tick_params(axis='both', which='major', labelsize=12)

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