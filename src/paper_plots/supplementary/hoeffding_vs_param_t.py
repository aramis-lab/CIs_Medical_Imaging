import numpy as np
import argparse
import matplotlib.pyplot as plt

def plot_hoeffding_vs_param_t(output_path:str):

    plt.rcdefaults()

    n_values = np.linspace(10, 1000, 200)

    hoeffding_width = 2 * np.sqrt(np.log(2 / 0.05) / (2 * n_values))
    worst_param_t_width = 2 / np.sqrt(n_values)
    typical_param_t_width = 4 * 0.2 / np.sqrt(n_values)

    plt.figure(figsize=(8, 6))
    plt.plot(n_values, hoeffding_width, label="Hoeffding Width", color='blue', linestyle='--')
    plt.plot(n_values, worst_param_t_width, label="Worst-case Parametric t Width", color='green')
    plt.plot(n_values, typical_param_t_width, label="Typical Parametric t Width", color='red')
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Width of Confidence Interval")
    plt.title("Comparison between Hoeffding's, worst-case and typical Parametric t widths")
    plt.legend(title="Method")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Hoeffding vs worst-case Parametric t widths.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output plot.")
    args = parser.parse_args()
    output_path = f"{args.output_folder}/hoeffding_vs_param_t.pdf"
    plot_hoeffding_vs_param_t(output_path)


if __name__ == "__main__":
    main()