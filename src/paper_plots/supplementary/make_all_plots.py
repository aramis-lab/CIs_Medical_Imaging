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
from .micro_vs_macro import plot_micro_vs_macro_all
from .relative_error_CCP import plot_rel_error_CCP_segm
from .tests_CCP_segm_vs_classif import plot_significance_matrix_segm_vs_classif
from .tests_CCP_segm import plot_significance_matrix_segm


def make_all_plots(root_folder: str):
    output_path = os.path.join(root_folder, "clean_figs/supplementary/all_cov_classif.pdf")
    plot_all_cov_classif(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/all_width_classif.pdf")
    plot_all_width_classif(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/all_cov_segm.pdf")
    plot_all_cov_segm(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/all_width_segm.pdf")
    plot_all_width_segm(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/bca_fail.pdf")
    plot_bca_fail(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/central_vs_dispersion.pdf")
    plot_central_vs_dispersion(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/ci_bounds.pdf")
    plot_ci_bounds(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/cov_fail_dsc_mean.pdf")
    plot_cov_fail_dsc_mean(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/coverage_metrics_segm.pdf")
    plot_coverage_metrics_segm(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/skew_kurt_classif.pdf")
    plot_descriptive_stats_classif(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/skew_kurt_segm.pdf")
    plot_descriptive_stats_segm(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/macro_vs_segm_stats.pdf")
    plot_macro_vs_segm_stats(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/micro_vs_macro.pdf")
    plot_micro_vs_macro_all(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/relative_error_CCP_segm.pdf")
    plot_rel_error_CCP_segm(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/tests_CCP_segm_vs_classif.pdf")
    plot_significance_matrix_segm_vs_classif(root_folder, output_path)
    output_path = os.path.join(root_folder, "clean_figs/supplementary/tests_CCP_segm.pdf")
    plot_significance_matrix_segm(root_folder, output_path)
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all plots for the paper.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the data.")
    args = parser.parse_args()

    make_all_plots(args.root_folder)
