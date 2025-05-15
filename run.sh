#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=50:00:00
#SBATCH --nodes=20
#SBATCH -A qhn@cpu
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=40
#SBATCH --qos=qos_cpu-t4
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

module load python
conda activate CI

mapfile -t TASKS_AND_ALGOS < benchmark_list.txt
PAIR="${TASKS_AND_ALGOS[$SLURM_ARRAY_TASK_ID]}"
TASK=$(echo $PAIR | cut -d ' ' -f1)
ALGO=$(echo $PAIR | cut -d ' ' -f2)

# Run hydra sweep over configurations for this task+algo
python src/run_instance.py "$TASK" "$ALGO" -m \
  metric=dsc,nsd \
  kernel=epanechnikov \
  summary_stat=mean,median,trimmed_mean,std,iqr_length