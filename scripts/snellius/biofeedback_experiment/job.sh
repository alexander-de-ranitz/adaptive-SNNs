#!/bin/bash
#SBATCH -J biofeedback
#SBATCH -t 53
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=START,END,FAIL

# Load necessary modules
echo "Starting job on $(hostname) at $(date +%Y%m%d_%H%M%S)"
echo "Job $SLURM_JOBID started at `date`"
echo "Loading modules..."
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

echo "Modules loaded, starting job..."

echo "Activating venv..."
REPO_DIR="$HOME/adaptive_SNNs"

# Install required Python packages in user space
source ~/venvs/adaptive_snns/bin/activate

echo "Environment setup complete, setting up directories..."
cd "$REPO_DIR"

# Make output directory in TMPDIR
mkdir -p "$TMPDIR/output_dir"
mkdir -p "$TMPDIR/output_dir/logs"
mkdir -p "$TMPDIR/output_dir/results"

cleanup() {
    local exit_code=$?
    echo "Cleanup running with exit code $exit_code at $(date)"
    echo "Contents of $TMPDIR/output_dir:"
    find "$TMPDIR/output_dir" -type f 2>/dev/null | head -20

    DEST_DIR="$REPO_DIR/results/biofeedback_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$DEST_DIR"
    cp -rv "$TMPDIR/output_dir/." "$DEST_DIR/" || echo "Copy failed with exit code $?"
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

# Prevent BLAS/OpenMP oversubscription when running many Python jobs in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=1 # Enable 64-bit precision in JAX, which is important for numerical stability in our simulations
export JAX_PLATFORMS=cuda # Use CUDA backend for GPU acceleration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "Running simulations at $(date)..."
set -x
python -u "$REPO_DIR/scripts/snellius/biofeedback_experiment/launch.py" \
    --output_dir "$TMPDIR/output_dir"

echo "Simulations completed, copying results back to home directory..."
