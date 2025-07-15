#!/bin/bash

module load python
conda activate CI

# METRICS=(boundary_iou assd cldice hd hd_perc iou masd)
METRICS=(masd assd)
metrics_csv=$(IFS=','; echo "${METRICS[*]}")

# Preprocess instance lists
python src/utils/extract_df_and_make_instance_list.py -m --config-name=config_segm \
  metric="$metrics_csv" \
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
sbatch --array=0-$((count - 1)) array_job_segm.sh