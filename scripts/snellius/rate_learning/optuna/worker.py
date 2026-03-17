# worker.py
import csv
import os
from pathlib import Path

import diffrax as dfx
import optuna
from jax import numpy as jnp

from adaptive_SNN.models.networks.eligibility_LIF import EligibilityLIFNetwork
from scripts.helpers import SimulationConfig, run_simulation


def objective(trial: optuna.Trial) -> float:
    # Example hyperparameters
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    nl = trial.suggest_float("nl", 1e-3, 0.5, log=False)

    # Run your CPU-heavy code here
    # Return a single scalar to minimize/maximize
    score = run_training(lr=lr, noise_level=nl, trial_number=trial.number)

    return score  # e.g. validation loss to minimize


def run_training(lr, noise_level, trial_number):
    initial_weights = jnp.linspace(1, 10, 10)
    mean_rewards = []
    for weight_idx, w in enumerate(initial_weights):
        t0 = 0
        t1 = 1000
        n_saves = 1000
        # Create unique seed for each simulation: trial_number * 1000 + weight_index
        # This ensures no correlations between different simulations
        unique_seed = trial_number * 1000 + weight_idx
        print(
            f"Running simulation with lr={lr}, noise_level={noise_level}, initial_weight={w}, trial={trial_number}, seed={unique_seed}"
        )
        save_at = dfx.SaveAt(
            ts=jnp.linspace((t1 - t0) * 0.75, t1, n_saves),
            fn=lambda t, x, args: x.reward_signal,
        )  # Only save reward signal to reduce memory usage
        cfg = SimulationConfig(
            model=EligibilityLIFNetwork,
            t0=t0,
            t1=t1,
            save_at=save_at,  # No need to save intermediate results for optimization
            save_results=False,  # Don't save results to disk during optimization
            network_output_fn=lambda t, agent_state, args: jnp.squeeze(
                agent_state.noisy_network.network_state.S[0]
            ),
            lr=lr,
            noise_level=noise_level,
            initial_weight=w,
            key_seed=unique_seed,  # Use unique seed for each simulation
        )

        result, _ = run_simulation(cfg)
        mean_rewards.append(jnp.mean(result.ys))

    score = jnp.mean(
        jnp.array(mean_rewards)
    )  # Example: average reward across different initial weights
    return score.item()  # Convert JAX scalar to Python float for Optuna


def log_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback to log trial completion to a CSV file for recovery purposes.
    Appends each trial result to trials_live.csv in real-time.
    """
    # Get output directory from environment variable or use default
    output_dir = os.environ.get("OPTUNA_OUTPUT_DIR", "./optuna_results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "trials_live.csv"

    # Check if file exists to determine if we need to write headers
    file_exists = csv_file.exists()

    # Prepare trial data
    row = {
        "number": trial.number,
        "value": trial.value,
        "datetime_start": trial.datetime_start.isoformat()
        if trial.datetime_start
        else "",
        "datetime_complete": trial.datetime_complete.isoformat()
        if trial.datetime_complete
        else "",
        "duration": str(trial.duration) if trial.duration else "",
        "state": trial.state.name,
    }

    # Add all parameters
    for param_name, param_value in trial.params.items():
        row[f"params_{param_name}"] = param_value

    # Add user attributes if any
    for attr_name, attr_value in trial.user_attrs.items():
        row[f"user_attrs_{attr_name}"] = attr_value

    # Write to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    # Print a simple confirmation message
    print(
        f"Trial {trial.number} complete: value={trial.value:.6f}, params={trial.params} "
        f"(logged to {csv_file})",
        flush=True,
    )


def main():
    storage_url = os.environ["OPTUNA_STORAGE"]
    study_name = os.environ.get("OPTUNA_STUDY", "rate-learning-study")

    # Load the existing study (should already be created by the job script)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
    )

    # Each worker runs a chunk of trials, sequentially
    # The callback logs trial results after each completion for recovery purposes
    study.optimize(
        objective,
        n_trials=2,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[log_trial_callback],
    )


if __name__ == "__main__":
    main()
