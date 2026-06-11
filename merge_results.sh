#!/bin/bash
set -euo pipefail

module load python
eval "$(conda shell.bash hook)"
conda activate CI

CONFIG_PATH=${1:?Usage: ./merge_results.sh <config_path e.g. classif/config_classif>}

CFG_ROOT="src/cfg"
CONFIG_FILE="${CFG_ROOT}/${CONFIG_PATH}.yaml"
CONFIG_DIR=$(dirname "$CONFIG_PATH")          # e.g. "classif"
ABLATION_GROUP="${CONFIG_DIR}/ablations"       # e.g. "classif/ablations"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ── Extract from main config (mirrors run_all.sh) ──────────────────
mapfile -t EXPERIMENTS < <(
    python src/utils/extract_sweep.py run_plan.experiments \
        --config "$CONFIG_FILE"
)

SWEEP_FILE_REL=$(python src/utils/extract_sweep.py run_plan.sweep_file \
    --config "$CONFIG_FILE")
SWEEP_FILE="${CFG_ROOT}/${SWEEP_FILE_REL}"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "ERROR: Sweep file not found: $SWEEP_FILE"
    exit 1
fi

# Determine task type from config directory name
if [[ "$CONFIG_DIR" == *"classif"* ]]; then
    TASK_TYPE="classif"
elif [[ "$CONFIG_DIR" == *"segm"* ]]; then
    TASK_TYPE="segm"
else
    echo "ERROR: Cannot infer task type (classif/segm) from config path: $CONFIG_DIR"
    exit 1
fi

echo "========================================"
echo "  Config file:    $CONFIG_FILE"
echo "  Ablation group: $ABLATION_GROUP"
echo "  Sweep file:     $SWEEP_FILE"
echo "  Task type:      $TASK_TYPE"
echo "  Experiments:    ${EXPERIMENTS[*]}"
echo "========================================"
echo ""

for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    echo "──────────────────────────────────────"
    echo "  Merging: $EXPERIMENT"
    echo "──────────────────────────────────────"

    # relative_output_dir is overridden in each ablation config
    ABLATION_CFG="${CFG_ROOT}/${ABLATION_GROUP}/${EXPERIMENT}.yaml"

    if [[ ! -f "$ABLATION_CFG" ]]; then
        echo "  WARNING: Ablation config not found: $ABLATION_CFG — skipping."
        continue
    fi

    RESULTS_DIR=$(python src/utils/extract_sweep.py relative_output_dir \
        --config "$ABLATION_CFG")

    echo "  Results dir: $RESULTS_DIR"

    python src/utils/merge_dataframes.py \
        --results_dir "$RESULTS_DIR" \
        --sweep_file "$SWEEP_FILE" \
        --task_type "$TASK_TYPE"

    echo ""
done

echo "========================================"
echo "  All merges complete."
echo "========================================"