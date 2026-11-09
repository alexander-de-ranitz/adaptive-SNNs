#!/bin/bash
#SBATCH -J single_synapse_learning_extended_performance
#SBATCH -t 180
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=72
#SBATCH --gpus=0
#SBATCH --mem=240G
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="single_synapse_learning_extended_performance"
export LAUNCH_SCRIPT="scripts/snellius/single_synapse_learning/extended_performance/launch.py"
export JAX_PLATFORM="cpu"
export COPY_ON_EXIT="1"  # Copy results back even if the job is killed (e.g. OOM)

source "$REPO_DIR/scripts/snellius/_templates/job_template.sh"
