#!/bin/bash
#SBATCH --job-name=hydra_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH -A zcd@cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=qos_cpu-t3
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread

module load python
eval "$(conda shell.bash hook)"
conda activate CI

OVERRIDES=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

echo "Hydra config: $HYDRA_CONFIG_NAME"
echo "Task list:    $TASK_LIST"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Overrides:    $OVERRIDES"
echo ""

# ── Start background memory monitor ────────────────────────
PEAK_MEM=0
monitor_memory() {
    while kill -0 "$1" 2>/dev/null; do
        # RSS in KB from /proc
        MEM_KB=$(ps -o rss= -p "$1" 2>/dev/null || echo 0)
        # Include child processes
        CHILDREN_KB=$(ps -o rss= --ppid "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}')
        TOTAL_KB=$((MEM_KB + CHILDREN_KB))
        if (( TOTAL_KB > PEAK_MEM )); then
            PEAK_MEM=$TOTAL_KB
        fi
        sleep 5
    done
}

# ── Run the job ─────────────────────────────────────────────
START_TIME=$(date +%s)

eval python src/run.py --config-name="$HYDRA_CONFIG_NAME" $OVERRIDES &
PID=$!

monitor_memory "$PID" &
MONITOR_PID=$!

wait "$PID"
EXIT_CODE=$?

kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Print summary ──────────────────────────────────────────
echo ""
echo "========================================"
echo "  JOB SUMMARY"
echo "========================================"
echo "  Exit code:    $EXIT_CODE"
echo "  Wall time:    ${ELAPSED}s ($(date -ud @${ELAPSED} +%H:%M:%S))"
echo "  Peak memory:  $((PEAK_MEM / 1024)) MB ($((PEAK_MEM / 1048576)) GB)"
echo "  CPUs:         $SLURM_CPUS_PER_TASK"
echo "========================================"

# ── Also get SLURM's own accounting ────────────────────────
echo ""
echo "SLURM accounting:"
sstat -j "${SLURM_JOB_ID}.batch" \
    --format=AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize \
    2>/dev/null || echo "  (sstat not available)"

exit $EXIT_CODE