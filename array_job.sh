#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH -A zcd@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --qos=qos_cpu-t3
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

# ── CONFIG_PATH and TASK_LIST come from --export in run_all.sh ──

module load python

eval "$(conda shell.bash hook)"
conda activate CI

# ── Read the override line for this array task ──────────────
#    Lines are 1-indexed in sed, SLURM_ARRAY_TASK_ID is 0-indexed
OVERRIDES=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

echo "Config:    $HYDRA_CONFIG_NAME"
echo "Task list: $TASK_LIST"
echo "Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Overrides: $OVERRIDES"
echo ""

# ── Run single Hydra job ────────────────────────────────────
#    eval handles shlex-quoted values (e.g. +task='my task')
eval python src/run.py --config-name="$HYDRA_CONFIG_NAME" $OVERRIDES