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
  metric=dsc \
  kernel=epanechnikov \
  summary_stat=trimmed_mean \
  +task=pancreas_L1 +algo=17111010008