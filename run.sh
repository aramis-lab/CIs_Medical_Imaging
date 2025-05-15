#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=20:00:00
#SBATCH --nodes=20
#SBATCH -A qhn@cpu
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=40
#SBATCH --qos=qos_cpu-t3
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

module load python
conda activate CI

METRICS=(dsc nsd)

# Extract metrics into comma-separated string
metrics_csv=$(IFS=','; echo "${METRICS[*]}")

python src/utils/extract_df_and_make_instance_list.py -m \
    metric="$metrics_csv" \
    kernel=epanechnikov \
    summary_stat=mean,median,trimmed_mean,std,iqr_length

for METRIC in "${METRICS[@]}"; do
  LIST_FILE="instances_list/${METRIC}.txt"
  mapfile -t TASKS_AND_ALGOS < "$LIST_FILE"
  PAIR="${TASKS_AND_ALGOS[$SLURM_ARRAY_TASK_ID]}"
  read -r TASK ALGO <<< "$PAIR"

  python src/run.py -m \
    metric="$METRIC" \
    kernel=epanechnikov \
    summary_stat=mean,median,trimmed_mean,std,iqr_length \
    +task="$TASK" +algo="$ALGO"
done
