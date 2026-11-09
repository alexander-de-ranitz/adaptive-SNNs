#!/bin/bash
# Shared Snellius job template. Each experiment's job.sh should set its own #SBATCH
# header, set the variables below, then source this file.
#
# Required:
#   JOB_NAME       label used for the copied-back results directory
#   LAUNCH_SCRIPT  path, relative to the repo root, invoked with
#                  --output_dir "$TMPDIR/output_dir"
#   JAX_PLATFORM   "cpu" or "cuda"
# Optional:
#   REPO_DIR       repo checkout on the cluster (default: $HOME/adaptive_SNNs)
#   RESULT_PREFIX  results directory prefix (default: $JOB_NAME)
#   COPY_ON_EXIT   "1" to copy results back from an EXIT/INT/TERM trap instead
#                  of after a clean run (use for long GPU jobs that may be killed)

set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"
RESULT_PREFIX="${RESULT_PREFIX:-$JOB_NAME}"

echo "Starting job $JOB_NAME on $(hostname) at $(date +%Y%m%d_%H%M%S)"
echo "Loading modules..."
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source ~/venvs/adaptive_snns/bin/activate
cd "$REPO_DIR"

mkdir -p "$TMPDIR/output_dir/logs" "$TMPDIR/output_dir/results"

copy_results() {
    local dest="$REPO_DIR/results/${RESULT_PREFIX}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dest"
    cp -r "$TMPDIR/output_dir/." "$dest/"
}

if [ "${COPY_ON_EXIT:-0}" = "1" ]; then
    cleanup() {
        local exit_code=$?
        echo "Cleanup running with exit code $exit_code at $(date)"
        find "$TMPDIR/output_dir" -type f 2>/dev/null | head -20
        copy_results || echo "Copy failed with exit code $?"
        exit "$exit_code"
    }
    trap cleanup EXIT INT TERM
fi

# Prevent BLAS/OpenMP oversubscription when running many parallel Python jobs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=1  # 64-bit precision matters for numerical stability here
export JAX_PLATFORMS="$JAX_PLATFORM"

echo "Running simulations at $(date)..."
python -u "$REPO_DIR/$LAUNCH_SCRIPT" --output_dir "$TMPDIR/output_dir"
echo "Simulations completed at $(date)."

if [ "${COPY_ON_EXIT:-0}" != "1" ]; then
    copy_results
fi
