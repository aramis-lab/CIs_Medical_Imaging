import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

from ..plot_utils import metric_labels
from ..df_loaders import extract_df_classif_cov, extract_df_classif_width


def plot_micro_vs_macro_all(root_folder:str, output_path:str):
    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro =['accuracy', "auc", "f1_score", "ap"]

    df_micro = extract_df_classif_cov(folder_path_micro, file_prefix_micro, metrics_micro)
    df_micro_width = extract_df_classif_width(folder_path_micro, file_prefix_micro, metrics_micro)

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro =['balanced_accuracy', "auc", "f1_score", "ap"]

    df_macro = extract_df_classif_cov(folder_path_macro, file_prefix_macro, metrics_macro)
    df_macro_width = extract_df_classif_width(folder_path_macro, file_prefix_macro, metrics_macro)

    # Choose layout: each metric gets 2 columns (coverage + width)
    means_macro_df_n=df_macro[(df_macro['method']=='percentile') & (df_macro['n']<=250)].sort_values(by=['stat', 'n'])
    means_micro_df_n=df_micro[(df_micro['method']=='percentile') & (df_micro['n']<=250)].sort_values(by=['stat', 'n'])

    means_macro_df_n_width =df_macro_width[(df_macro_width['method']=='percentile') & (df_macro_width['n']<=250)].sort_values(by=['stat', 'n'])
    means_micro_df_n_width =df_micro_width[(df_micro_width['method']=='percentile') & (df_micro_width['n']<=250)].sort_values(by=['stat', 'n'])

    num_metrics = len([m for m in metrics_macro if m not in ['accuracy', 'balanced_accuracy']]) + 1
    fig, axes = plt.subplots(
        num_metrics, 2, 
        figsize=(36, 15 * num_metrics), 
        sharex=True
    )

    # If only one row, force axes to be 2D
    if num_metrics == 1:
        axes = np.array([axes])

    row = 0

    for metric in metrics_micro:
        if metric in ['accuracy', 'balanced_accuracy']:
            continue

        ax_cov = axes[row, 0]
        ax_width = axes[row, 1]

        # Coverage ----------------------------------------------------------
        macro_data = means_macro_df_n[means_macro_df_n['stat'] == metric]
        micro_data = means_micro_df_n[means_micro_df_n['stat'] == metric]

        for data, label in [(macro_data, "Macro"), (micro_data, "Micro")]:
            if not data.empty:
                med = data.groupby("n")["value"].median()
                q1 = data.groupby("n")["value"].quantile(0.25)
                q3 = data.groupby("n")["value"].quantile(0.75)
                n_vals = med.index.values

                ax_cov.plot(n_vals, med.values, marker="o", label=label)
                ax_cov.fill_between(n_vals, q1.values, q3.values, alpha=0.2)

        ax_cov.set_title(f'Coverage for micro-aggregated vs macro-aggregated {metric_labels[metric]}', weight='bold', fontsize=32)
        ax_cov.set_xlabel('Sample size', weight='bold', fontsize=28)
        ax_cov.set_ylabel('Coverage', weight='bold', fontsize=28)
        ax_cov.tick_params(axis='y', labelsize=24)
        ax_cov.tick_params(axis='x', labelsize=24)
        ax_cov.set_ylim(0, 1.05)
        ax_cov.grid(True, axis='y')
        ax_cov.legend(fontsize=24)

        # Width subplot (right)
        # Macro
        macro_width = means_macro_df_n_width[means_macro_df_n_width['stat'] == metric]
        if not macro_width.empty:
            medians = macro_width.groupby('n')['width'].median().values
            q1 = macro_width.groupby('n')['width'].quantile(0.25).values
            q3 = macro_width.groupby('n')['width'].quantile(0.75).values
            n_vals = macro_width['n'].unique()
            ax_width.plot(n_vals, medians, marker='o', label='Macro', linewidth=2, markersize=8)
            ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
        # Micro
        micro_width = means_micro_df_n_width[means_micro_df_n_width['stat'] == metric]
        if not micro_width.empty:
            medians = micro_width.groupby('n')['width'].median().values
            q1 = micro_width.groupby('n')['width'].quantile(0.25).values
            q3 = micro_width.groupby('n')['width'].quantile(0.75).values
            n_vals = micro_width['n'].unique()
            ax_width.plot(n_vals, medians, marker='o', label='Micro', linewidth=2, markersize=8)
            ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
        ax_width.set_title(f'Width for micro-aggregated vs macro-aggregated {metric_labels[metric]}', weight='bold', fontsize=32)
        ax_width.set_xlabel('Sample size', weight='bold', fontsize=28)
        ax_width.set_ylabel('Width', weight='bold', fontsize=28)
        ax_width.tick_params(axis='y', labelsize=24)
        ax_width.tick_params(axis='x', labelsize=24)
        ax_width.set_ylim(-0.01, None)
        ax_width.grid(True, axis='y')
        ax_width.legend()

        row += 1
    # ----------- Balanced Accuracy vs Accuracy (final row) ------------------

    ax_cov = axes[row, 0]
    ax_width = axes[row, 1]

    # Coverage
    macro_data = means_macro_df_n[means_macro_df_n['stat'] == "balanced_accuracy"]
    micro_data = means_micro_df_n[means_micro_df_n['stat'] == "accuracy"]

    # Coverage subplot (left)
    # Macro
    if not macro_data.empty:
        medians = macro_data.groupby('n')['coverage'].median().values
        q1 = macro_data.groupby('n')['coverage'].quantile(0.25).values
        q3 = macro_data.groupby('n')['coverage'].quantile(0.75).values
        n_vals = macro_data['n'].unique()
        ax_cov.plot(n_vals, medians, marker='o', label='Balanced Accuracy', linewidth=2, markersize=8)
        ax_cov.fill_between(n_vals, q1, q3, alpha=0.2)
    # Micro
    if not micro_data.empty:
        medians = micro_data.groupby('n')['coverage'].median().values
        q1 = micro_data.groupby('n')['coverage'].quantile(0.25).values
        q3 = micro_data.groupby('n')['coverage'].quantile(0.75).values
        n_vals = micro_data['n'].unique()
        ax_cov.plot(n_vals, medians, marker='o', label='Accuracy', linewidth=2, markersize=8)
        ax_cov.fill_between(n_vals, q1, q3, alpha=0.2)
    ax_cov.set_title(f'Coverage for {metric_labels["balanced_accuracy"]} vs Accuracy', weight='bold', fontsize=32)
    ax_cov.set_xlabel('Sample size', weight='bold', fontsize=28)
    ax_cov.set_ylabel('Coverage', weight='bold', fontsize=28)
    ax_cov.tick_params(axis='y', labelsize=24)
    ax_cov.tick_params(axis='x', labelsize=24)
    ax_cov.set_ylim(0, 1.05)
    ax_cov.grid(True, axis='y')
    ax_cov.legend(fontsize=24)
    # Width subplot (right)
    # Macro
    macro_width = means_macro_df_n_width[means_macro_df_n_width['stat'] == "balanced_accuracy"]
    if not macro_width.empty:
        medians = macro_width.groupby('n')['width'].median().values
        q1 = macro_width.groupby('n')['width'].quantile(0.25).values
        q3 = macro_width.groupby('n')['width'].quantile(0.75).values
        n_vals = macro_width['n'].unique()
        ax_width.plot(n_vals, medians, marker='o', label='Balanced Accuracy', linewidth=2, markersize=8)
        ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
    # Micro
    micro_width = means_micro_df_n_width[means_micro_df_n_width['stat'] == "accuracy"]
    if not micro_width.empty:
        medians = micro_width.groupby('n')['width'].median().values
        q1 = micro_width.groupby('n')['width'].quantile(0.25).values
        q3 = micro_width.groupby('n')['width'].quantile(0.75).values
        n_vals = micro_width['n'].unique()
        ax_width.plot(n_vals, medians, marker='o', label='Accuracy', linewidth=2, markersize=8)
        ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
    ax_width.set_title(f'Width for {metric_labels["balanced_accuracy"]} vs Accuracy', weight='bold', fontsize=32)
    ax_width.set_xlabel('Sample size', weight='bold', fontsize=28)
    ax_width.set_ylabel('Width', weight='bold', fontsize=28)
    ax_width.tick_params(axis='y', labelsize=24)
    ax_width.tick_params(axis='x', labelsize=24)
    ax_width.set_ylim(-0.01, None)
    ax_width.grid(True, axis='y')
    ax_width.legend()

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure micro vs macro for all classification metrics.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/micro_vs_macro.pdf")

    # Call your plotting function
    plot_micro_vs_macro_all(root_folder, output_path)

if __name__ == "__main__":
    main()