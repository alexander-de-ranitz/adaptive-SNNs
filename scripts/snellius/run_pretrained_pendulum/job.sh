#!/bin/bash
#SBATCH -J run_pretrained_pendulum
#SBATCH -t 100
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="pendulum_pretrained"
export LAUNCH_SCRIPT="scripts/snellius/run_pretrained_pendulum/run.py"
export JAX_PLATFORM="cuda"
export COPY_ON_EXIT="1"

source "$REPO_DIR/scripts/snellius/templates/job_template.sh"
