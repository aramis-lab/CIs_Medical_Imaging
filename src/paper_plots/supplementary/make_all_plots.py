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
from .tests_CCP_segm_vs_classif import plot_significance_matrix_segm_classif
from .tests_CCP_segm import plot_significance_matrix_segm
from .concentration_ineq import plot_hoeffding_eb_t_ci_widths, plot_hoeffding_eb_t_ci_width_ratios

from .sample_needs_all import plot_fig10_sample_needs

def make_all_plots(root_folder: str, output_folder: str, upload_overleaf: bool = False, export_format: str = "pdf"):
    # output_path = os.path.join(output_folder, f"coverages_classif.{export_format}")
    # plot_all_cov_classif(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"width_classif.{export_format}")
    # plot_all_width_classif(root_folder, output_path, upload_overleaf=upload_overleaf)
    output_path = os.path.join(output_folder, f"coverages_segm.{export_format}")
    plot_all_cov_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"width_segm.{export_format}")
    # plot_all_width_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"bca_fail.{export_format}")
    # plot_bca_fail(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"spread_vs_central_all.{export_format}")
    # plot_central_vs_dispersion(root_folder, output_path, upload_overleaf=upload_overleaf)
    # plot_fig10_sample_needs(root_folder, output_folder=output_folder)
    # output_path = os.path.join(output_folder, f"ci_bounds.{export_format}")
    # plot_ci_bounds(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"fail_mean_dsc_percentile.{export_format}")
    # plot_cov_fail_dsc_mean(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"cov_segm_metrics.{export_format}")
    # plot_coverage_metrics_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"skew_kurt_classif.{export_format}")
    # plot_descriptive_stats_classif(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"skew_kurt_segm.{export_format}")
    # plot_descriptive_stats_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"macro_vs_segm_stats.{export_format}")
    # plot_macro_vs_segm_stats(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"micro_vs_segm_stats.{export_format}")
    # plot_micro_vs_segm_stats(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"micro_vs_macro.{export_format}")
    # plot_micro_vs_macro_all(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"relative_errors.{export_format}")
    # plot_rel_error_CCP_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"pairwise_comp_classif_segm.{export_format}")
    # plot_significance_matrix_segm_vs_classif(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"pairwise_comp_segm_segm.{export_format}")
    # plot_significance_matrix_segm(root_folder, output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"concentration_ineq.{export_format}")
    # plot_hoeffding_eb_t_ci_widths(output_path, upload_overleaf=upload_overleaf)
    # output_path = os.path.join(output_folder, f"concentration_ineq_ratios.{export_format}")
    # plot_hoeffding_eb_t_ci_width_ratios(output_path, upload_overleaf=upload_overleaf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all plots for the paper.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data.")
    parser.add_argument("--output_folder", type=str, help="Output folder for saving plots.")
    parser.add_argument("--export_format", type=str, default="pdf", help="Format for exported plots (e.g., pdf, png).")
    parser.add_argument("--upload_overleaf", action="store_true", help="Whether to upload the generated plots to Overleaf.")
    args = parser.parse_args()

    root_folder = args.root_folder
    output_folder = args.output_folder or os.path.join(root_folder, "clean_figs/supplementary/")
    export_format = args.export_format
    upload_overleaf = args.upload_overleaf

    make_all_plots(root_folder, output_folder, upload_overleaf, export_format)
