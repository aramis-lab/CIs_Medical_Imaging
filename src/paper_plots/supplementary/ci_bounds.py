import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from ..plot_utils import method_labels, method_colors

def plot_ci_bounds(root_folder: str, output_path: str):

    task = "Task03_Liver_L2"
    alg_name = "17111010008"

    df = pd.read_csv(os.path.join(root_folder, "data_matrix_grandchallenge_all.csv"))
    df = df[df["score"]=="dsc"]
    df = df[(df["alg_name"]==alg_name) & (df["subtask"]==task)]

    true_value = df["value"].to_numpy().mean()

    df = pd.read_csv(os.path.join(root_folder, f"results_dsc_median/results_dsc_median_{task}_{alg_name}_50.csv"))

    df = df[(df["n"]==50)]

    _, axs = plt.subplots(1, 3, figsize=(24, 8))

    lower_all = {}
    upper_all = {}
    for i, method in enumerate(["percentile", "basic", "bca"]):
        ax = axs[i]
        upper = df[f"upper_bound_{method}"].to_numpy()
        lower = df[f"lower_bound_{method}"].to_numpy()
        center = (upper+lower)/2
        lower_all[method] = lower
        upper_all[method] = upper
    

        indices = np.argsort(lower)
        lower = lower[indices]
        upper = upper[indices]

        coverage = np.mean((lower <= true_value) & (upper >= true_value))

        ax.fill_betweenx(np.arange(len(indices)), lower, upper, color=method_colors[method])
        ax.vlines(true_value, 0,10000, colors="red", linestyles="--", label="True summary statistic value")
        ax.vlines(np.mean(center), 0,10000, colors="black", linestyles="--", label="Intervals mean center")
        ax.axis()
        ax.set_title("CI bounds " + method_labels[method] + " vs True Value, Coverage: " + f"{coverage:.2f}", fontsize=16)
        ax.set_xlabel("Confidence Interval Bounds", fontsize=14)
        ax.set_ylabel("Interval Index (lower bound sorted)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(0, 1)
        ax.legend()

    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot CI bounds for a specific task and algorithm.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data.")
    parser.add_argument("--output_path", type=str, required=False, help="Output path for the plot PDF.")
    args = parser.parse_args()
    root_folder = args.root_folder
    output_path = args.output_path if args.output_path else os.path.join(root_folder, "clean_figs/supplementary/ci_bounds.pdf")

    plot_ci_bounds(root_folder, output_path)

if __name__ == "__main__":
    main()