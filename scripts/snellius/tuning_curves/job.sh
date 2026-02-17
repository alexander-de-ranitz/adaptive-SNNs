#!/bin/bash
#SBATCH -J tuning_curves
#SBATCH -t 25
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --mem=64GB
#SBATCH --gpus=1

# Load necessary modules
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

# Prevent BLAS/OpenMP oversubscription when running many Python jobs in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running tuning curve simulations..."
python "$REPO_DIR/scripts/snellius/tuning_curves/launch.py" \
    --output_dir "$TMPDIR/output_dir"

echo "Simulations completed, copying results back to home directory..."
# Copy results back to home directory
DEST_DIR="$REPO_DIR/results/tuning_curves_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST_DIR"
cp -r "$TMPDIR/output_dir/." "$DEST_DIR/"
