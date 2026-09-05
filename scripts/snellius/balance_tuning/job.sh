#!/bin/bash
#SBATCH -J delta_v_tuning
#SBATCH -t 60
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=72
#SBATCH --gpus=1
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="balance_tuning"
export LAUNCH_SCRIPT="scripts/snellius/balance_tuning/launch.py"
export JAX_PLATFORM="cpu"

source "$REPO_DIR/scripts/snellius/templates/job_template.sh"
