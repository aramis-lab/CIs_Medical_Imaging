#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=your_partition_name

module load python
conda activate CI

python src/run.py -m \
  --config-path cfg \
  --config-name config.yaml \
  metric=dsc,nsd \
  kernel=epanechnikov \
  summary_stat=mean,median,trimmed_mean,std,iqr_length
