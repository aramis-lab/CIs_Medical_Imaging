#!/bin/bash
set -euo pipefail

module load python
conda activate CI

CONFIG_PATH=${1:?Usage: ./run_all.sh <config_path e.g. classif/config_classif>}

# ── Resolve paths ───────────────────────────────────────────
CFG_ROOT="src/cfg"                              # ← Fix here
CONFIG_FILE="${CFG_ROOT}/${CONFIG_PATH}.yaml"
CONFIG_DIR=$(dirname "$CONFIG_FILE")
ABLATION_DIR="${CONFIG_DIR}/ablations"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================"
echo "  Config file:   $CONFIG_FILE"
echo "  Ablation dir:  $ABLATION_DIR"
echo "========================================"

# ── 1. Extract from YAML (OmegaConf.load, no Hydra) ────────
mapfile -t EXPERIMENTS < <(
    python src/utils/extract_sweep.py run_plan.experiments \
        --config "$CONFIG_FILE"
)
mapfile -t ALL_METRICS < <(
    python src/utils/extract_sweep.py run_plan.all_metrics \
        --config "$CONFIG_FILE"
)

echo "Experiments : ${EXPERIMENTS[*]}"
echo "All metrics : ${ALL_METRICS[*]}"
echo ""

# ── 2. Preprocess: generate instance lists ──────────────────
metrics_csv=$(IFS=','; echo "${ALL_METRICS[*]}")

echo "Generating instance lists for: $metrics_csv"
python src/utils/extract_df_and_make_instance_list.py \
    --config-name="$CONFIG_PATH" -m \
    metric="$metrics_csv"
echo ""

# ── 3. For each experiment: build task list → sbatch ────────
mkdir -p task_lists logs

for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    echo "──────────────────────────────────────"
    echo "  Experiment: $EXPERIMENT"
    echo "──────────────────────────────────────"

    ABLATION_FILE="${ABLATION_DIR}/${EXPERIMENT}.yaml"
    TASK_LIST="task_lists/${EXPERIMENT}.txt"

    if [[ ! -f "$ABLATION_FILE" ]]; then
        echo "  ERROR: Ablation file not found: $ABLATION_FILE"
        continue
    fi

    # Build task list — reads YAML directly, no Hydra
    python src/utils/build_task_list.py \
        --ablation "$ABLATION_FILE" \
        --config-name "$CONFIG_PATH" \
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
           --export=ALL,CONFIG_PATH="$CONFIG_PATH",TASK_LIST="$TASK_LIST" \
           array_job.sh

    echo ""
done

echo "========================================"
echo "  All experiments submitted."
echo "========================================"