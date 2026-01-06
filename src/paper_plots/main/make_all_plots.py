import os
import argparse

from .fig3_basic import plot_fig3_basic
from .fig4_bca import plot_fig4_bca
from .fig5_param_small_samples import plot_fig5_param_small_samples
from .fig6_hoeffding_vs_param_t import plot_fig6_hoeffding_vs_param_t
from .fig7_macro_vs_segm_mean import plot_fig7_macro_vs_segm_mean
from .fig7bis_micro_vs_segm_mean import plot_fig7bis_micro_vs_segm_mean
from .fig8_metrics import plot_fig8_metrics
from .fig9_micro_vs_macro import plot_fig9_micro_vs_macro
from .fig10_sample_needs import plot_fig10_sample_needs

def make_all_plots(root_folder:str, output_folder:str):
    plot_fig3_basic(root_folder, os.path.join(output_folder, "fig3_basic.pdf"))
    plot_fig4_bca(root_folder, os.path.join(output_folder, "fig4_bca.pdf"))
    plot_fig5_param_small_samples(root_folder, os.path.join(output_folder, "fig5_param_small_samples.pdf"))
    plot_fig6_hoeffding_vs_param_t(root_folder, os.path.join(output_folder, "fig6_hoeffding_vs_param_t.pdf"))
    plot_fig7_macro_vs_segm_mean(root_folder, os.path.join(output_folder, "fig7_macro_vs_segm_mean.pdf"))
    plot_fig7bis_micro_vs_segm_mean(root_folder, os.path.join(output_folder, "fig7bis_micro_vs_segm_mean.pdf"))
    plot_fig8_metrics(root_folder, os.path.join(output_folder, "fig8_metrics.pdf"))
    plot_fig9_micro_vs_macro(root_folder, os.path.join(output_folder, "fig9_micro_vs_macro.pdf"))
    plot_fig10_sample_needs(root_folder, os.path.join(output_folder, "fig10_sample_needs.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all paper plots.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder containing results.")
    parser.add_argument("--output_folder", required=False, help="Path to the output folder for saving plots.")
    
    args = parser.parse_args()
    
    root_folder = args.root_folder
    output_folder = args.output_folder or os.path.join(root_folder, "clean_figs/main/")
    
    os.makedirs(output_folder, exist_ok=True)
    
    make_all_plots(root_folder, output_folder)
