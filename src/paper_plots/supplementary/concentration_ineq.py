import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t


def plot_hoeffding_eb_t_ci_widths(output_path: str):

    alpha = 0.05

    n_min = 10
    n_max = 10**5

    sigma_t=0.22

    n = np.logspace(np.log10(n_min), np.log10(n_max), num=5000).astype(int)

    width_hoeffding = 2 * np.sqrt(np.log(2 / alpha) / (2 * n))
    width_empirical_bernstein = 2 * (
        sigma_t * np.sqrt(2  * np.log(4 / alpha) / n)
        + 7 * np.log(4 / alpha) / (3 * (n - 1))
    )
    tcrit = t.ppf(1 - alpha / 2, df=n - 1)
    width_t = 2 * tcrit * (sigma_t / np.sqrt(n))

    plt.plot(n, width_hoeffding, label="Hoeffding", color='#0173B2', linestyle='-')
    plt.plot(n, width_empirical_bernstein, label=fr"Empirical Bernstein (median $\hat{{\sigma}}=0.22$)", color='#DE8F05', linestyle='-')
    plt.plot(n, width_t, label=fr"Parametric t (median $\hat{{\sigma}}=0.22$)", color='#CC78BC')

    plt.xlabel("n")
    plt.ylabel("CI width")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.25)
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def plot_hoeffding_eb_t_ci_width_ratios(output_path: str):

    alpha = 0.05

    n_min = 10
    n_max = 10**9

    sigma_t=0.22

    n = np.logspace(np.log10(n_min), np.log10(n_max), num=5000).astype(int)


    width_hoeffding = 2 * np.sqrt(np.log(2 / alpha) / (2 * n))

    width_empirical_bernstein = 2 * (
        sigma_t * np.sqrt(2  * np.log(4 / alpha) / n)
        + 7 * np.log(4 / alpha) / (3 * (n - 1))
    )

    tcrit = t.ppf(1 - alpha / 2, df=n - 1)
    width_t = 2 * tcrit * (sigma_t / np.sqrt(n))

    width_hoeffding_ratio = width_hoeffding / width_t
    width_empirical_bernstein_ratio = width_empirical_bernstein / width_t

    plt.plot(n, width_hoeffding_ratio, label=fr"Hoeffding / Parametric t (median $\hat{{\sigma}}=0.2200$)", color='#0173B2', linestyle='-')
    plt.plot(n, width_empirical_bernstein_ratio, label=fr"Empirical Bernstein / Parametric t (independent of $\hat{{\sigma}}$)", color='#DE8F05', linestyle='-')

    plt.xlabel("n")
    plt.ylabel("CI Width Ratio")
    plt.xscale('log')
    plt.legend()
    plt.ylim(0, 10)
    plt.grid(True)
    plt.title(f"CI Width Ratios")
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Supp Figure coverage of all methods and metrics for classification.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder.")
    parser.add_argument("--output_path", required=False, help="Path to save the output plot.")
    args = parser.parse_args()

    root_folder = args.root_folder
    # If output_path not provided, default inside root_folder
    output_path = args.output_path or os.path.join(root_folder, "clean_figs/supplementary/concentration_ineq.pdf")

    plot_hoeffding_eb_t_ci_widths(output_path)
    plot_hoeffding_eb_t_ci_width_ratios(output_path.replace(".pdf", "_ratios.pdf"))

if __name__ == "__main__":
    main()