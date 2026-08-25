import os

from .make_correction_fdr import tell_significance, tests
from .test_basic import plot_significance_matrix_basic
from .test_basic_classif import plot_significance_matrix_basic_classif
from .test_bca import plot_significance_matrix_bca
from .test_micro_vs_macro import plot_significance_matrix_micro_macro
from .test_param_vs_bootstrap import plot_significance_matrix_param_boot
from .test_spread_vs_central import plot_significance_matrix_spread_central
from .test_WDP_segm_classif import plot_significance_matrix_wdp_segm_classif
from .tests_CCP_segm import plot_significance_matrix_segm
from .tests_CCP_segm_vs_classif import plot_significance_matrix_segm_classif

# For each test emitted by tell_significance: the plotting function, the extra positional
# arguments it takes between p_values and output_path, and its output path relative to
# the output folder.
plot_specs = {
    "basic_classif_micro": {
        "plot": plot_significance_matrix_basic_classif,
        "args": ("micro",),
        "output_path": "cov_basic_classif_micro/all_n.pdf",
    },
    "basic_classif_macro": {
        "plot": plot_significance_matrix_basic_classif,
        "args": ("macro",),
        "output_path": "cov_basic_classif_macro/all_n.pdf",
    },
    "basic": {
        "plot": plot_significance_matrix_basic,
        "args": (),
        "output_path": "cov_basic_segm/all_n.pdf",
    },
    "bca": {
        "plot": plot_significance_matrix_bca,
        "args": ("segm",),
        "output_path": "cov_bca_segm/all_n.pdf",
    },
    "bca_classif": {
        "plot": plot_significance_matrix_bca,
        "args": ("classif",),
        "output_path": "cov_bca_classif/all_n.pdf",
    },
    "micro_macro": {
        "plot": plot_significance_matrix_micro_macro,
        "args": (),
        "output_path": "cov_micro_macro/all_n.pdf",
    },
    "param_boot_segm": {
        "plot": plot_significance_matrix_param_boot,
        "args": ("segm", "cov"),
        "output_path": "cov_param_boot_segm/all_n.pdf",
    },
    "param_boot_classif": {
        "plot": plot_significance_matrix_param_boot,
        "args": ("classif", "cov"),
        "output_path": "cov_param_boot_classif/all_n.pdf",
    },
    "param_boot_segm_width": {
        "plot": plot_significance_matrix_param_boot,
        "args": ("segm", "width"),
        "output_path": "width_param_boot_segm/all_n.pdf",
    },
    "spread_central": {
        "plot": plot_significance_matrix_spread_central,
        "args": (),
        "output_path": "cov_spread_central/all_n.pdf",
    },
    "wdp_segm_classif": {
        "plot": plot_significance_matrix_wdp_segm_classif,
        "args": ("micro",),
        "output_path": "width_segm_classif_micro/all_n.pdf",
    },
    "wdp_segm_classif_macro": {
        "plot": plot_significance_matrix_wdp_segm_classif,
        "args": ("macro",),
        "output_path": "width_segm_classif_macro/all_n.pdf",
    },
    "segm": {
        "plot": plot_significance_matrix_segm,
        "args": (),
        "output_path": "coverage_segm_metrics/test_segm.pdf",
    },
    "segm_classif": {
        "plot": plot_significance_matrix_segm_classif,
        "args": (),
        "output_path": "cov_segm_classif/all_n.pdf",
    },
}


def make_plots(significance, output_folder: str, upload_overleaf: bool = False):
    """
    Draw the significance heatmap of every test present in `significance`.

    `significance` is the DataFrame returned by
    `make_correction_fdr.tell_significance`, so the q-values it carries are
    FDR-corrected across all tests jointly.
    """
    for test in significance["test"].unique():
        spec = plot_specs.get(test)
        if spec is None:
            print(f"No plotting function registered for test '{test}', skipping.")
            continue

        print(test)
        row = significance[significance["test"] == test].iloc[0]
        significance_test = row["significance"]
        p_values = row["pvalues_corrected"]

        output_path = os.path.join(output_folder, spec["output_path"])
        spec["plot"](
            significance_test,
            p_values,
            *spec["args"],
            output_path,
            upload_overleaf=upload_overleaf,
        )


def make_all_heatmaps(
    ablation_dir: str,
    output_folder: str = None,
    upload_overleaf_for: str = None,
    tests=tests,
):
    """
    Draw the significance heatmaps of every ablation found in `ablation_dir`.

    Each subfolder of `ablation_dir` is one ablation. A subfolder is processed only
    if it contains a `pvalues` folder, and is skipped otherwise. The FDR correction
    is pooled across all tests *within* an ablation, never across ablations.

    If `output_folder` is None, each ablation's figures are written inside its own
    subfolder. Otherwise they go to `output_folder/<ablation>/`.

    `upload_overleaf_for` is the name of the single ablation whose figures are pushed
    to Overleaf; all ablations share the same Overleaf destinations, so only one can
    be uploaded per run.
    """
    for ablation in sorted(os.listdir(ablation_dir)):
        if ablation.startswith("."):
            continue

        ablation_path = os.path.join(ablation_dir, ablation)
        pvalues_folder = os.path.join(ablation_path, "pvalues")
        if not os.path.isdir(pvalues_folder):
            print(f"No pvalues folder in '{ablation}', skipping.")
            continue

        print(f"=== {ablation} ===")
        significance = tell_significance(tests, pvalues_folder)
        if significance.empty:
            print(f"No p-value files in '{ablation}', skipping.")
            continue

        if output_folder is None:
            ablation_output_folder = os.path.join(
                ablation_path, "clean_figs/supplementary/test_results"
            )
        else:
            ablation_output_folder = os.path.join(output_folder, ablation)

        make_plots(
            significance,
            ablation_output_folder,
            upload_overleaf=(ablation == upload_overleaf_for),
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate the significance heatmaps of all tests, for every ablation."
    )
    parser.add_argument("--ablation_dir", required=True,
                        help="Folder containing one subfolder per ablation.")
    parser.add_argument("--output_folder", required=False,
                        help="Folder to save the output plots. Defaults to inside each ablation subfolder.")
    parser.add_argument("--upload_overleaf_for", required=False,
                        help="Name of the single ablation whose figures are uploaded to Overleaf.")
    args = parser.parse_args()

    make_all_heatmaps(
        args.ablation_dir,
        output_folder=args.output_folder,
        upload_overleaf_for=args.upload_overleaf_for,
    )


if __name__ == "__main__":
    main()