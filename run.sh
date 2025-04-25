#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem150G
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
