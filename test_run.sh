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

python src/run.py -m \
  metric=nsd \
  kernel=epanechnikov \
  summary_stat=iqr_length \
  +task=pancreas_l1 +algo=17111010008