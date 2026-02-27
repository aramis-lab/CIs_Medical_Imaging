import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.stats import t
from ..df_loaders import extract_df_segm_width

def plot_fig6_hoeffding_vs_param_t(root_folder:str, output_path:str):

    plt.rcdefaults()

    n_values = np.linspace(10, 1000, 200)

    hoeffding_width = 2 * np.sqrt(np.log(2 / 0.05) / (2 * n_values))

    typical_param_t_width = 2 * t.ppf(0.975, n_values - 1) * 0.220090 / np.sqrt(n_values)
    empirical_bernstein_width = 2 * (0.220090*np.sqrt(2  * np.log(4 / 0.05) / n_values)+ 7 * np.log(4 / 0.05) / (3 * (n_values - 1)))

    folder_path_segm = os.path.join(root_folder, "results_metrics_segm")

    df_segm_width = extract_df_segm_width(folder_path_segm, "aggregated_results", ["dsc"], ["mean"])

    df_percentile = df_segm_width[df_segm_width["method"] == "percentile"]
    df_param_t = df_segm_width[df_segm_width["method"] == "param_t"]

    df_percentile_median = df_percentile.groupby("n")["width"].median()
    df_param_t_median = df_param_t.groupby("n")["width"].median()
    df_param_t_Q1 = df_param_t.groupby("n")["width"].quantile(0.25)
    df_param_t_Q3 = df_param_t.groupby("n")["width"].quantile(0.75)
    var_Q1 = (df_param_t_Q1 * np.sqrt(df_param_t_Q1.index) / 4) ** 2
    var_Q3 = (df_param_t_Q3 * np.sqrt(df_param_t_Q3.index) / 4) ** 2
    inferred_width_Q1 = 2 * t.ppf(0.975, n_values - 1) * np.sqrt(var_Q1.median()) / np.sqrt(n_values)
    inferred_width_Q3 = 2 * t.ppf(0.975, n_values - 1) * np.sqrt(var_Q3.median()) / np.sqrt(n_values)
    plt.figure(figsize=(8, 6))
   # plt.plot(n_values, worst_param_t_width, label="Worst-case Parametric t Width", color='#DE8F05')
    plt.plot(
    n_values,
    inferred_width_Q1,
    linestyle="--",
    color="#CC78BC",
    linewidth=2,
    label=fr"Parametric t Q1 ($\hat\sigma={np.sqrt(var_Q1.median()):.4f}$)"
    )

    plt.plot(
        n_values,
        inferred_width_Q3,
        linestyle="-.",
        color="#CC78BC",
        linewidth=2,
        alpha=0.7,
        label=fr"Parametric t Q3 ($\hat\sigma={np.sqrt(var_Q3.median()):.4f}$)"
    )
    
    plt.plot(n_values, hoeffding_width, label="Hoeffding", color='#0173B2', linestyle='--')
    plt.plot(n_values, empirical_bernstein_width, label="Empirical Bernstein", color='#DE8F05', linestyle='-.')
    plt.plot(n_values, typical_param_t_width, label="Median Parametric t", color='#CC78BC')
    plt.scatter(df_percentile_median.index, df_percentile_median.values, label="Percentile, CIs of mean of DSC", color='#029E73', marker='D',zorder=5)
    plt.scatter(df_param_t_median.index, df_param_t_median.values, label="Parametric t, CIs of mean of DSC", color="#3C220C", marker='x', zorder=5)
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Width of Confidence Interval")
    plt.title("Comparison of concentration inequalities, parametric t and percentile bootstrap")
    plt.legend(title="Method")
    plt.ylim(-0.01, 1.01)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Hoeffding vs worst-case Parametric t widths.")
    parser.add_argument("--root_folder", type=str, required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the output plot.")
    args = parser.parse_args()
    root_folder = args.root_folder
    output_path = args.output_path if args.output_path else f"{args.root_folder}/clean_figs/main/fig6_hoeffding_vs_param_t.pdf"
    plot_fig6_hoeffding_vs_param_t(root_folder, output_path)


if __name__ == "__main__":
    main()