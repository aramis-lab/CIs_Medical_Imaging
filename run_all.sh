#!/bin/bash
set -euo pipefail

module load python
eval "$(conda shell.bash hook)"
conda activate CI

CONFIG_PATH=${1:?Usage: ./run_all.sh <config_path e.g. classif/config_classif>}

CFG_ROOT="src/cfg"
CONFIG_FILE="${CFG_ROOT}/${CONFIG_PATH}.yaml"
CONFIG_DIR=$(dirname "$CONFIG_PATH")          # e.g. "classif"
ABLATION_GROUP="${CONFIG_DIR}/ablations"       # e.g. "classif/ablations"
HYDRA_CONFIG_NAME="$CONFIG_PATH"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================"
echo "  Config file:     $CONFIG_FILE"
echo "  Ablation group:  $ABLATION_GROUP"
echo "========================================"

mapfile -t EXPERIMENTS < <(
    python src/utils/extract_sweep.py run_plan.experiments \
        --config "$CONFIG_FILE"
)
mapfile -t ALL_METRICS < <(
    python src/utils/extract_sweep.py run_plan.all_metrics \
        --config "$CONFIG_FILE"
)

SWEEP_FILE_REL=$(python src/utils/extract_sweep.py run_plan.sweep_file \
    --config "$CONFIG_FILE")
SWEEP_FILE="${CFG_ROOT}/${SWEEP_FILE_REL}"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "ERROR: Sweep file not found: $SWEEP_FILE"
    exit 1
fi

echo "Experiments : ${EXPERIMENTS[*]}"
echo "All metrics : ${ALL_METRICS[*]}"
echo "Sweep file  : $SWEEP_FILE"
echo ""

metrics_csv=$(IFS=','; echo "${ALL_METRICS[*]}")
echo "Generating instance lists for: $metrics_csv"
python src/utils/extract_df_and_make_instance_list.py \
    --config-name="$CONFIG_PATH" -m \
    metric="$metrics_csv"
echo ""

mkdir -p task_lists logs

for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    echo "──────────────────────────────────────"
    echo "  Experiment: $EXPERIMENT"
    echo "──────────────────────────────────────"

    TASK_LIST="task_lists/${EXPERIMENT}.txt"

    python src/utils/build_task_list.py \
        --ablation-name "$EXPERIMENT" \
        --ablation-group "$ABLATION_GROUP" \
        --sweep "$SWEEP_FILE" \
        --instance-dir instances_list \
        --output "$TASK_LIST"

    NUM_TASKS=$(wc -l < "$TASK_LIST")

    if [[ "$NUM_TASKS" -eq 0 ]]; then
        echo "  No tasks found, skipping."
        continue
    fi

    echo "  Submitting $NUM_TASKS tasks..."
    sbatch --array=0-$((NUM_TASKS - 1)) \
           --job-name="exp_${EXPERIMENT}" \
           --export=ALL,HYDRA_CONFIG_NAME="$HYDRA_CONFIG_NAME",TASK_LIST="$TASK_LIST" \
           array_job.sh

    echo ""
done

echo "========================================"
echo "  All experiments submitted."
echo "========================================"