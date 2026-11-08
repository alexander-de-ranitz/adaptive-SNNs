#!/bin/bash
#SBATCH -J assumption_verification
#SBATCH -t 150
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=72
#SBATCH --gpus=0
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=START,END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="assumption_verification"
export LAUNCH_SCRIPT="scripts/snellius/assumption_verification/launch.py"
export JAX_PLATFORM="cpu"
export COPY_ON_EXIT="1"

source "$REPO_DIR/scripts/snellius/_templates/job_template.sh"
