#!/bin/bash
#SBATCH -J single_synapse_learning_extended
#SBATCH -t 45
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus=1
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="single_synapse_learning"
export LAUNCH_SCRIPT="scripts/snellius/single_synapse_learning/launch.py"
export JAX_PLATFORM="cpu"

source "$REPO_DIR/scripts/snellius/templates/job_template.sh"
