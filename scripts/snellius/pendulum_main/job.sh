#!/bin/bash
#SBATCH -J pendulum_main_run
#SBATCH -t 800
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --mail-user=alexanderderanitz@gmail.com
#SBATCH --mail-type=START,END,FAIL    

REPO_DIR="${REPO_DIR:-$HOME/adaptive_SNNs}"

scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING

export JOB_NAME="pendulum_main_run_final"
export LAUNCH_SCRIPT="scripts/snellius/pendulum_main/launch.py"
export JAX_PLATFORM="cuda"
export COPY_ON_EXIT="1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

source "$REPO_DIR/scripts/snellius/_templates/job_template.sh"
