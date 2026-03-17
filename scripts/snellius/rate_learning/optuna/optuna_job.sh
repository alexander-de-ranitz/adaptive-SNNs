#!/bin/bash
#SBATCH -J rate_learning
#SBATCH -t 400
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --gpus=1

# Load necessary modules
echo "Loading modules..."
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load PostgreSQL/17.5-GCCcore-14.2.0

echo "Modules loaded, starting job..."

# Set up PostgreSQL database for Optuna
# Use node-local temp space for the DB (fast; cleaned up after job)
PGDIR="$TMPDIR/pgdata"
PGSOCK="$TMPDIR/pgsock"
PGPORT=55432
mkdir -p "$PGDIR" "$PGSOCK"
chmod 700 "$PGDIR" "$PGSOCK"

# Initialize DB with "trust" auth (safe because we bind to 127.0.0.1 and socket dir is private)
initdb -D "$PGDIR" --auth-local=trust --auth-host=trust >/dev/null

# Start postgres:
# - listen only on localhost
# - increase max connections above 128
# - put unix socket in $PGSOCK
pg_ctl -D "$PGDIR" -l "$TMPDIR/postgres.log" -w start -o "\
  -h 127.0.0.1 \
  -p $PGPORT \
  -k $PGSOCK \
  -N 256"

# Create a database for Optuna (optional but cleaner than using "postgres")
createdb -h 127.0.0.1 -p $PGPORT optuna_db || true

# Optuna storage URL (TCP)
export OPTUNA_STORAGE="postgresql+psycopg2://$USER@127.0.0.1:$PGPORT/optuna_db"
export OPTUNA_STUDY="rate_learning_study"

echo "Activating venv..."
REPO_DIR="$HOME/adaptive_SNNs"

# Set up output directory for live trial logging
OUTPUT_DIR="$REPO_DIR/results/optuna_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
export OPTUNA_OUTPUT_DIR="$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Activate the virtual environment
source ~/venvs/adaptive_snns/bin/activate

# Install/update required dependencies
echo "Installing dependencies..."
pip install --quiet optuna psycopg2-binary pandas

echo "Environment setup complete, setting up directories..."
cd "$REPO_DIR"

# Add repository root to PYTHONPATH so scripts module can be imported
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Prevent BLAS/OpenMP oversubscription when running many Python jobs in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=1 # Enable 64-bit precision in JAX, which is important for numerical stability in our simulations

# Create the Optuna study before launching workers to avoid race conditions
echo "Creating Optuna study..."
python -c "
import optuna
import os

storage = os.environ['OPTUNA_STORAGE']
study_name = os.environ['OPTUNA_STUDY']

sampler = optuna.samplers.TPESampler(
    multivariate=True,
    group=False,
    seed=1234,
)

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction='maximize',
    sampler=sampler,
    load_if_exists=True,
)
print(f'Study {study_name} created/loaded successfully')
"

echo "Running rate learning hyperparameter study..."
srun -n 64 python "$REPO_DIR/scripts/snellius/rate_learning/optuna/worker.py"

# After the study is complete, export results and clean up
echo "Exporting Optuna study results..."
python -c "
import optuna
import pandas as pd
import os

storage = os.environ['OPTUNA_STORAGE']
study_name = os.environ['OPTUNA_STUDY']
study = optuna.load_study(study_name=study_name, storage=storage)

# Export to CSV
df = study.trials_dataframe()
df.to_csv('$OUTPUT_DIR/trials.csv', index=False)

# Save best parameters
with open('$OUTPUT_DIR/best_params.txt', 'w') as f:
    f.write(f'Best value: {study.best_value}\n')
    f.write(f'Best parameters: {study.best_params}\n')
"

# Dump the Optuna database to a file for safekeeping
pg_dump -h 127.0.0.1 -p $PGPORT optuna_db > $OUTPUT_DIR/optuna_rate_learning.sql

# Stop PostgreSQL server
pg_ctl -D "$PGDIR" -m fast stop
