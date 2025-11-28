import matplotlib.pyplot as plt
import os
import argparse

from ..plot_utils import metric_labels
from ..df_loaders import extract_df_classif_cov, extract_df_classif_width


def plot_fig8_micro_vs_macro(root_folder:str, output_path:str):
    folder_path_micro = os.path.join(root_folder, "results_metrics_classif")
    file_prefix_micro = "aggregated_results"
    metrics_micro =['accuracy']

    df_micro = extract_df_classif_cov(folder_path_micro, file_prefix_micro, metrics_micro)
    df_micro_width = extract_df_classif_width(folder_path_micro, file_prefix_micro, metrics_micro)

    folder_path_macro = os.path.join(root_folder, "results_metrics_classif_macro")
    file_prefix_macro = "aggregated_results"
    metrics_macro =['balanced_accuracy']

    df_macro = extract_df_classif_cov(folder_path_macro, file_prefix_macro, metrics_macro)
    df_macro_width = extract_df_classif_width(folder_path_macro, file_prefix_macro, metrics_macro)


    # Choose layout: each metric gets 2 columns (coverage + width)
    macro_data=df_macro[(df_macro['method']=='percentile') & (df_macro['n']<=250)].sort_values(by=['metric', 'n'])
    micro_data=df_micro[(df_micro['method']=='percentile') & (df_micro['n']<=250)].sort_values(by=['metric', 'n'])

    macro_width =df_macro_width[(df_macro_width['method']=='percentile') & (df_macro_width['n']<=250)].sort_values(by=['metric', 'n'])
    micro_width =df_micro_width[(df_micro_width['method']=='percentile') & (df_micro_width['n']<=250)].sort_values(by=['metric', 'n'])

    fig, axes = plt.subplots(
        1, 2, 
        figsize=(36, 15), 
        sharex=True
    )

    ax_cov = axes[0]

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
    ax_width = axes[1]

    # Width subplot (right)
    # Macro
    if not macro_width.empty:
        medians = macro_width.groupby('n')['width'].median().values
        q1 = macro_width.groupby('n')['width'].quantile(0.25).values
        q3 = macro_width.groupby('n')['width'].quantile(0.75).values
        n_vals = macro_width['n'].unique()
        ax_width.plot(n_vals, medians, marker='o', label='Balanced Accuracy', linewidth=2, markersize=8)
        ax_width.fill_between(n_vals, q1, q3, alpha=0.2)
    # Micro
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
    ax_width.legend(fontsize=24)

    for ax, letter in zip([ax_cov, ax_width], ['A', 'B']):
        ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                fontsize=36, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 8 micro vs macro.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path for the output PDF file.")

    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/main/fig8_micro_vs_macro.pdf")

    # Call your plotting function
    plot_fig8_micro_vs_macro(root_folder, output_path)

if __name__ == "__main__":
    main()
