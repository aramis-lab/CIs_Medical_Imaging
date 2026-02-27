import os

from .all_cov_classif import plot_all_cov_classif
from .all_width_classif import plot_all_width_classif
from .all_cov_segm import plot_all_cov_segm
from .all_width_segm import plot_all_width_segm
from .bca_fail import plot_bca_fail
from .central_vs_dispersion import plot_central_vs_dispersion
from .ci_bounds import plot_ci_bounds
from .cov_fail_dsc_mean import plot_cov_fail_dsc_mean
from .coverages_metrics_segm import plot_coverage_metrics_segm
from .diversity_classif import plot_descriptive_stats_classif
from .diversity_segm import plot_descriptive_stats_segm
from .macro_vs_segm_stats import plot_macro_vs_segm_stats
from .micro_vs_segm_stats import plot_micro_vs_segm_stats
from .micro_vs_macro import plot_micro_vs_macro_all
from .relative_error_CCP import plot_rel_error_CCP_segm
from .tests_CCP_segm_vs_classif import plot_significance_matrix_segm_vs_classif
from .tests_CCP_segm import plot_significance_matrix_segm
from .concentration_ineq import plot_hoeffding_eb_t_ci_widths, plot_hoeffding_eb_t_ci_width_ratios

from .sample_needs_all import plot_fig10_sample_needs

def make_all_plots(root_folder: str, output_folder: str, export_format: str = "pdf"):
    output_path = os.path.join(output_folder, f"all_cov_classif.{export_format}")
    plot_all_cov_classif(root_folder, output_path)
    output_path = os.path.join(output_folder, f"all_width_classif.{export_format}")
    plot_all_width_classif(root_folder, output_path)
    output_path = os.path.join(output_folder, f"all_cov_segm.{export_format}")
    plot_all_cov_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"all_width_segm.{export_format}")
    plot_all_width_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"bca_fail.{export_format}")
    plot_bca_fail(root_folder, output_path)
    output_path = os.path.join(output_folder, f"central_vs_dispersion.{export_format}")
    plot_central_vs_dispersion(root_folder, output_path)
    plot_fig10_sample_needs(root_folder, output_folder=output_folder)
    output_path = os.path.join(output_folder, f"ci_bounds.{export_format}")
    plot_ci_bounds(root_folder, output_path)
    output_path = os.path.join(output_folder, f"tests_CCP_segm.{export_format}")
    plot_significance_matrix_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"cov_fail_dsc_mean.{export_format}")
    plot_cov_fail_dsc_mean(root_folder, output_path)
    output_path = os.path.join(output_folder, f"cov_segm_metrics.{export_format}")
    plot_coverage_metrics_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"skew_kurt_classif.{export_format}")
    plot_descriptive_stats_classif(root_folder, output_path)
    output_path = os.path.join(output_folder, f"skew_kurt_segm.{export_format}")
    plot_descriptive_stats_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"macro_vs_segm_stats.{export_format}")
    plot_macro_vs_segm_stats(root_folder, output_path)
    output_path = os.path.join(output_folder, f"micro_vs_segm_stats.{export_format}")
    plot_micro_vs_segm_stats(root_folder, output_path)
    output_path = os.path.join(output_folder, f"micro_vs_macro.{export_format}")
    plot_micro_vs_macro_all(root_folder, output_path)
    output_path = os.path.join(output_folder, f"relative_error_CCP_segm.{export_format}")
    plot_rel_error_CCP_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"tests_CCP_segm_vs_classif.{export_format}")
    plot_significance_matrix_segm_vs_classif(root_folder, output_path)
    output_path = os.path.join(output_folder, f"tests_CCP_segm.{export_format}")
    plot_significance_matrix_segm(root_folder, output_path)
    output_path = os.path.join(output_folder, f"concentration_ineq.{export_format}")
    plot_hoeffding_eb_t_ci_widths(output_path)
    output_path = os.path.join(output_folder, f"concentration_ineq_ratios.{export_format}")
    plot_hoeffding_eb_t_ci_width_ratios(output_path)
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all plots for the paper.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for saving plots.")
    parser.add_argument("--export_format", type=str, default="pdf", help="Format for exported plots (e.g., pdf, png).")
    args = parser.parse_args()

    root_folder = args.root_folder
    output_folder = args.output_folder or os.path.join(root_folder, "clean_figs/main/")
    export_format = args.export_format

    make_all_plots(root_folder, output_folder, export_format)
