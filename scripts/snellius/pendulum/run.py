import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

from jax import random as jr

from adaptive_SNN.simulation_configs.pendulum_config import create_pendulum_config
from adaptive_SNN.utils.runner import run_simulation


def main():
    print(f"Jax is using device: {jax.devices()[0]}")

    parser = argparse.ArgumentParser(
        description="Run network simulation with specified parameters"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/biofeedback_experiment.npz",
        help="Path to save the simulation results",
    )
    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )

    args = parser.parse_args()
    cfg = create_pendulum_config(N_neurons=1000, key=jr.PRNGKey(args.key_seed))
    cfg.save_file = args.output_file

    start = time.time()
    sol, model = run_simulation(cfg, save_results=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
