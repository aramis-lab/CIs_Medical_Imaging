metric_labels = {
    'dsc': 'DSC',
    'iou': 'IoU',
    'nsd': 'NSD',
    'boundary_iou': 'Boundary IoU',
    'cldice': 'clDice',
    'assd': 'ASSD',
    'masd' : 'MASD',
    'hd': 'HD',
    'hd_perc': 'HD95',
    'balanced_accuracy': 'Balanced Accuracy',
    'ap': 'AP',
    'auc': 'AUC',
    'f1_score': 'F1 Score',
    'accuracy': 'Accuracy',
    "mcc": "MCC"
}

stat_labels = {
    'mean': 'Mean',
    'median': 'Median',
    'std': 'Standard Deviation',
    'trimmed_mean': 'Trimmed Mean',
    'iqr_length': 'IQR Length'
}

method_labels = {
    "basic": "Basic",
    "percentile": "Percentile",
    "bca": "BCa",
    "delong": "DeLong",
    "logit_transform": "Logit Transform",
    "wilson": "Wilson",
    "agresti_coull" : "Agresti-Coull",
    "exact" : "Exact \n(Clopper-Pearson)",
    "wald" : 'Wald',
    "param_t" : "Parametric t",
    "param_z" : "Parametric z"
}

method_colors = {
    "basic": "#D4461F",
    "percentile": "#8E5EE8", 
    "bca" : "#FF9742",
    "wilson" : "#DFCF3E", 
    "agresti_coull" : "#5D9336", 
    "exact" : "#DB4ADB", 
    "wald" : "#367F9C",
    "param_t" : "#999999", 
    "param_z" : "#A7C7E7"}

import subprocess
import shutil
from pathlib import Path

OVERLEAF_REPO = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()).parent.parent / "overleaf_CI_project"

def upload_to_overleaf(src_path, dest_path, commit_msg=None):
    """
    Copy a local file into the Overleaf project and push it.

    Args:
        src_path:  Path to the local figure (e.g. "results/plot.pdf")
        dest_path: Target path inside the Overleaf project (e.g. "figures/plot.pdf")
        commit_msg: Optional commit message
    """
    src = Path(src_path)
    dest = OVERLEAF_REPO / dest_path

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    # 1. Pull latest changes from Overleaf
    subprocess.run(["git", "pull", "-q", "--rebase"], cwd=OVERLEAF_REPO, check=True)

    # 2. Copy the file into the repo
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied {src} → {dest}")

    # 3. Stage, commit, push
    subprocess.run(["git", "add", str(dest)], cwd=OVERLEAF_REPO, check=True)

    msg = commit_msg or f"Update {dest_path}"
    result = subprocess.run(
        ["git", "commit", "-m", msg, "-q"],
        cwd=OVERLEAF_REPO,
        capture_output=True, text=True
    )

    if "nothing to commit" in result.stdout:
        print("File unchanged — nothing to push.")
        return

    subprocess.run(["git", "push", "-q"], cwd=OVERLEAF_REPO, check=True)
    print(f"Pushed {dest_path} to Overleaf!")