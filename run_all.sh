#!/bin/bash
# ================================================================
#  run_all.sh — Read experiments from YAML, preprocess, submit jobs
#
#  Usage:
#      ./run_all.sh config_classif
#      ./run_all.sh config_seg
# ================================================================

set -euo pipefail

module load python
conda activate CI

CONFIG_NAME=${1:?Usage: ./run_all.sh <config_name>}

echo "========================================"
echo "  Config: $CONFIG_NAME"
echo "========================================"

# ── 1. Extract experiments and metrics from YAML ────────────
mapfile -t EXPERIMENTS < <(
    python src/utils/extract_sweep.py run_plan.experiments \
        --config-name="$CONFIG_NAME"
)
mapfile -t ALL_METRICS < <(
    python src/utils/extract_sweep.py run_plan.all_metrics \
        --config-name="$CONFIG_NAME"
)

echo "Experiments : ${EXPERIMENTS[*]}"
echo "All metrics : ${ALL_METRICS[*]}"
echo ""

# ── 2. Preprocess: generate instance lists (once for all) ───
metrics_csv=$(IFS=','; echo "${ALL_METRICS[*]}")

echo "Generating instance lists for: $metrics_csv"
python src/utils/extract_df_and_make_instance_list.py \
    --config-name="$CONFIG_NAME" -m \
    metric="$metrics_csv"
echo ""

# ── 3. For each experiment: build task list → sbatch ─────────
mkdir -p task_lists logs

for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    echo "──────────────────────────────────────"
    echo "  Experiment: $EXPERIMENT"
    echo "──────────────────────────────────────"

    TASK_LIST="task_lists/${EXPERIMENT}.txt"

    # Build task list (sweep pairs × instance lists)
    python src/utils/build_task_list.py \
        --config-name="$CONFIG_NAME" \
        --experiment="$EXPERIMENT" \
        --instance-dir=instances_list \
        --output="$TASK_LIST"

    # Count tasks
    NUM_TASKS=$(wc -l < "$TASK_LIST")

    if [[ "$NUM_TASKS" -eq 0 ]]; then
        echo "  No tasks found, skipping."
        continue
    fi

    echo "  Submitting $NUM_TASKS tasks..."
    sbatch --array=0-$((NUM_TASKS - 1)) \
           --job-name="exp_${EXPERIMENT}" \
           --export=ALL,CONFIG_NAME="$CONFIG_NAME",TASK_LIST="$TASK_LIST" \
           array_job.sh

    echo ""
done

echo "========================================"
echo "  All experiments submitted."
echo "========================================"