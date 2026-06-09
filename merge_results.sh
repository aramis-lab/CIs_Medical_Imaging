#!/bin/bash
set -euo pipefail

module load python
eval "$(conda shell.bash hook)"
conda activate CI

CONFIG_PATH=${1:?Usage: ./merge_results.sh <config_path e.g. classif/config_classif>}

CFG_ROOT="src/cfg"
CONFIG_FILE="${CFG_ROOT}/${CONFIG_PATH}.yaml"
CONFIG_DIR=$(dirname "$CONFIG_PATH")          # e.g. "classif" or "segm"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract the results directory from the config
RESULTS_DIR=$(python src/utils/extract_sweep.py relative_output_dir \
    --config "$CONFIG_FILE")

# Extract the sweep file (contains the metric pairs)
SWEEP_FILE_REL=$(python src/utils/extract_sweep.py run_plan.sweep_file \
    --config "$CONFIG_FILE")
SWEEP_FILE="${CFG_ROOT}/${SWEEP_FILE_REL}"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "ERROR: Sweep file not found: $SWEEP_FILE"
    exit 1
fi

echo "========================================"
echo "  Config file:   $CONFIG_FILE"
echo "  Config dir:    $CONFIG_DIR"
echo "  Sweep file:    $SWEEP_FILE"
echo "  Results dir:   $RESULTS_DIR"
echo "========================================"

# Determine task type from config directory name
if [[ "$CONFIG_DIR" == *"classif"* ]]; then
    TASK_TYPE="classif"
elif [[ "$CONFIG_DIR" == *"segm"* ]]; then
    TASK_TYPE="segm"
else
    echo "ERROR: Cannot infer task type (classif/segm) from config path: $CONFIG_DIR"
    exit 1
fi

echo "  Task type:     $TASK_TYPE"
echo ""

python src/utils/merge_dataframes.py \
    --results_dir "$RESULTS_DIR" \
    --sweep_file "$SWEEP_FILE" \
    --task_type "$TASK_TYPE"

echo "========================================"
echo "  Merge complete."
echo "========================================"