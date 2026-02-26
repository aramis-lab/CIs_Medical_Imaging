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

def make_all_plots(root_folder:str, output_folder:str, export_format:str="pdf"):
    plot_fig3_basic(root_folder, os.path.join(output_folder, f"fig3_basic.{export_format}"))
    plot_fig4_bca(root_folder, os.path.join(output_folder, f"fig4_bca.{export_format}"))
    plot_fig5_param_small_samples(root_folder, os.path.join(output_folder, f"fig5_param_small_samples.{export_format}"))
    plot_fig6_hoeffding_vs_param_t(root_folder, os.path.join(output_folder, f"fig6_hoeffding_vs_param_t.{export_format}"))
    plot_fig7_macro_vs_segm_mean(root_folder, os.path.join(output_folder, f"fig7_macro_vs_segm_mean.{export_format}"))
    plot_fig7bis_micro_vs_segm_mean(root_folder, os.path.join(output_folder, f"fig7bis_micro_vs_segm_mean.{export_format}"))
    plot_fig8_metrics(root_folder, os.path.join(output_folder, f"fig8_metrics.{export_format}"))
    plot_fig9_micro_vs_macro(root_folder, os.path.join(output_folder, f"fig9_micro_vs_macro.{export_format}"))
    plot_fig10_sample_needs(root_folder, os.path.join(output_folder, f"fig10_sample_needs.{export_format}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all paper plots.")
    parser.add_argument("--root_folder", required=True, help="Path to the root folder containing results.")
    parser.add_argument("--output_folder", required=False, help="Path to the output folder for saving plots.")
    parser.add_argument("--export_format", required=False, default="pdf", help="Format for exported plots (e.g., pdf, png).")
    
    args = parser.parse_args()
    
    root_folder = args.root_folder
    output_folder = args.output_folder or os.path.join(root_folder, "clean_figs/main/")
    export_format = args.export_format
    
    os.makedirs(output_folder, exist_ok=True)
    
    make_all_plots(root_folder, output_folder, export_format)
