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

module load python
eval "$(conda shell.bash hook)"
conda activate CI

OVERRIDES=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

echo "Hydra config: $HYDRA_CONFIG_NAME"
echo "Task list:    $TASK_LIST"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Overrides:    $OVERRIDES"
echo ""

# ── Memory monitor using cgroup (SLURM-reliable) ───────────
MEM_LOG=$(mktemp /tmp/memlog.XXXXXX)

monitor_memory() {
    local pid=$1
    local peak=0
    while kill -0 "$pid" 2>/dev/null; do
        # Method 1: cgroup (most reliable on SLURM)
        if [[ -f /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.usage_in_bytes ]]; then
            mem=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.usage_in_bytes 2>/dev/null || echo 0)
            mem_mb=$((mem / 1048576))
        # Method 2: cgroup v2
        elif [[ -f /sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}/memory.current ]]; then
            mem=$(cat /sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}/memory.current 2>/dev/null || echo 0)
            mem_mb=$((mem / 1048576))
        # Method 3: sum all descendant processes
        else
            mem_kb=0
            for p in $(pgrep -P "$pid" 2>/dev/null) "$pid"; do
                rss=$(ps -o rss= -p "$p" 2>/dev/null || echo 0)
                mem_kb=$((mem_kb + rss))
            done
            mem_mb=$((mem_kb / 1024))
        fi

        if (( mem_mb > peak )); then
            peak=$mem_mb
        fi

        # Log every sample for debugging
        echo "$(date +%H:%M:%S) ${mem_mb}MB" >> "$MEM_LOG"

        sleep 2  # Sample every 2 seconds (faster than before)
    done
    echo "$peak" > "${MEM_LOG}.peak"
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

# ── Read peak memory ───────────────────────────────────────
PEAK_MEM=0
if [[ -f "${MEM_LOG}.peak" ]]; then
    PEAK_MEM=$(cat "${MEM_LOG}.peak")
fi

# ── Print summary ──────────────────────────────────────────
echo ""
echo "========================================"
echo "  JOB SUMMARY"
echo "========================================"
echo "  Exit code:    $EXIT_CODE"
echo "  Wall time:    ${ELAPSED}s ($(date -ud @${ELAPSED} +%H:%M:%S))"
echo "  Peak memory:  ${PEAK_MEM} MB ($((PEAK_MEM / 1024)) GB)"
echo "  CPUs:         $SLURM_CPUS_PER_TASK"

if [[ $EXIT_CODE -eq 137 ]]; then
    echo "  STATUS:       *** OOM KILLED ***"
    echo "  Last 5 memory samples:"
    tail -5 "$MEM_LOG" 2>/dev/null | sed 's/^/    /'
fi

echo "========================================"

# ── Cleanup ────────────────────────────────────────────────
rm -f "$MEM_LOG" "${MEM_LOG}.peak"

exit $EXIT_CODE