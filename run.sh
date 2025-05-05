#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=20:00:00
#SBATCH --nodes=40
#SBATCH --account=qhn@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

module load python
conda activate CI

python src/run.py -m \
  --config-path cfg \
  --config-name config.yaml \
  metric=dsc,nsd \
  kernel=epanechnikov \
  summary_stat=mean,median,trimmed_mean,std,iqr_length
