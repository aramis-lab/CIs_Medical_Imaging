#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH -A zcd@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --qos=qos_cpu-t3
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

module load python
conda activate CI

# Load all task-algo pairs
# METRICS=(accuracy f1_score balanced_accuracy mcc auc ap)
METRICS=(accuracy)
AVERAGE=micro
ALL_PAIRS=()

for METRIC in "${METRICS[@]}"; do
  LIST_FILE="instances_list/${METRIC}.txt"
  if [[ -f "$LIST_FILE" ]]; then
    mapfile -t LINES < "$LIST_FILE"
    for LINE in "${LINES[@]}"; do
      ALL_PAIRS+=("$METRIC;$LINE")
    done
  fi
done

PAIR="${ALL_PAIRS[$SLURM_ARRAY_TASK_ID]}"
IFS=";" read -r METRIC TASK_ALGO <<< "$PAIR"
read -r TASK ALGO <<< "$TASK_ALGO"

python src/run.py config_name=config_classif -m \
  metric="$METRIC" \
  kernel=epanechnikov \
  +task="$TASK" +algo="$ALGO"