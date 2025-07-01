#!/bin/bash

module load python
conda activate CI

# METRICS=(boundary_iou assd cldice hd hd_perc iou masd)
METRICS=(boundary_iou hd_perc iou masd)
metrics_csv=$(IFS=','; echo "${METRICS[*]}")

# Preprocess instance lists
python src/utils/extract_df_and_make_instance_list.py -m \
  metric="$metrics_csv" \
  kernel=epanechnikov \
  summary_stat=mean,median,trimmed_mean,std,iqr_length

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
sbatch --array=0-$((count - 1)) array_job.sh