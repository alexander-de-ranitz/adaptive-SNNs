#!/bin/bash
#SBATCH -J single_synapse_learning_extended_firing_statistics
#SBATCH -t 20
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=72
#SBATCH --gpus=0
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="single_synapse_learning_extended_firing_statistics"
export LAUNCH_SCRIPT="scripts/snellius/single_synapse_learning/extended_firing_statistics/launch.py"
export JAX_PLATFORM="cpu"

source "$REPO_DIR/scripts/snellius/_templates/job_template.sh"
