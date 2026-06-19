import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_SNN.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_SNN.utils.runner import run_batched_simulation


def main():
    print(f"Jax is using device: {jax.devices()[0]}")

    parser = argparse.ArgumentParser(
        description="Run network simulation with specified parameters"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/default_folder/results/pendulum",
        help="Path to save the simulation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model type to use for the simulation (e.g., 'default', 'gated')",
    )

    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )

    args = parser.parse_args()
    key = jr.PRNGKey(args.key_seed)
    N_parallel = 1
    model = GatedLIFNetwork if args.model == "gated" else EligibilityLIFNetwork
    configs = [
        create_pendulum_AC_config(N_neurons=2, model_cls=model, key=jr.fold_in(key, i))
        for i in range(N_parallel)
    ]

    def save_fn(t, x: SystemState, args):
        return (
            x.environment_state,
            x.reward_signal,
            x.agent_state.reward_predictor_state.value,
            x.agent_state.RPE,
        )

    for i, cfg in enumerate(configs):
        cfg.save_file = args.output_file + f"_{i}"
        cfg.t1 = 2000
        cfg.lr = 0
        cfg.save_at = SaveAt(
            ts=jnp.linspace(cfg.t0, cfg.t1, int(250 * cfg.t1)), fn=save_fn
        )

    start = time.time()
    sol, models = run_batched_simulation(
        configs, save_results=True, return_final_state=True
    )

    final_weights = sol.ys[1].agent_state.reward_predictor_state.weights
    jnp.save(args.output_file + "_final_weights.npy", final_weights)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
