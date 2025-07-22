#!/bin/bash

module load python
conda activate CI

# METRICS=(accuracy f1_score balanced_accuracy mcc auc ap)
METRICS=(mcc)
metrics_csv=$(IFS=','; echo "${METRICS[*]}")

# Preprocess instance lists
python src/utils/extract_df_and_make_instance_list.py --config-name=config_classif -m\
  metric="$metrics_csv" \
  +task=chexpert_pleural_effusion \
  kernel=epanechnikov

# Count total task-algo pairs
count=0
for METRIC in "${METRICS[@]}"; do
  FILE="instances_list/${METRIC}.txt"
  if [[ -f "$FILE" ]]; then
    lines=$(wc -l < "$FILE")
    count=$((count + lines))
  fi
done

# Submit SLURM array job
echo "Submitting array job with $count tasks..."
sbatch --array=0-$((count - 1)) array_job_classif.sh