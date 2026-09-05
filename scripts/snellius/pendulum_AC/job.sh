#!/bin/bash
#SBATCH -J pendulum_sim
#SBATCH -t 360
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=END,FAIL

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

export JOB_NAME="pendulum_AC_spiking_input"
export LAUNCH_SCRIPT="scripts/snellius/pendulum_AC/launch.py"
export JAX_PLATFORM="cuda"
export COPY_ON_EXIT="1"

source "$REPO_DIR/scripts/snellius/templates/job_template.sh"
