import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr

from adaptive_snn.models.agent_env_system import SystemState
from adaptive_snn.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.utils.runner import run_batched_simulation


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
        "--balance",
        type=float,
        default=jnp.nan,
        help="Balance parameter for the network",
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
        create_pendulum_AC_config(
            N_neurons=1000, model_cls=model, key=jr.fold_in(key, i)
        )
        for i in range(N_parallel)
    ]

    def save_fn(t, x: SystemState, args):
        total_charge = jnp.abs(x.agent_state.network_state.charge_in) + jnp.abs(
            x.agent_state.network_state.charge_out
        )
        balance = jnp.where(
            (x.agent_state.network_state.charge_in != 0.0)
            & (x.agent_state.network_state.charge_out != 0.0),
            (
                x.agent_state.network_state.charge_in
                + x.agent_state.network_state.charge_out
            )
            / total_charge,
            jnp.nan,
        )
        return (
            x.agent_state.network_state.S.astype(jnp.bool_),
            balance,
            jnp.nanmean(x.agent_state.network_state.W, axis=1),
            jnp.mean(x.agent_state.network_state.V),
            x.environment_state[2],
        )

    lrs = [0.0] * N_parallel
    init_w = [2.5] * N_parallel
    for i, cfg in enumerate(configs):
        lr = jnp.concat(
            [
                jnp.ones((cfg.N_neurons, cfg.N_neurons)) * lrs[i],
                jnp.ones((cfg.N_neurons, cfg.N_inputs)) / 10 * lrs[i],
            ],
            axis=1,
        )
        cfg.lr = lr
        cfg.save_file = args.output_file + f"_w_{init_w[i]}_i_{i}"
        cfg.t1 = 105
        # cfg.save_at = SaveAt(
        #     ts=jnp.linspace(0.0, cfg.t1, int(50*1e2)), fn=save_fn
        # )
        cfg.save_at = SaveAt(
            ts=jnp.linspace(100, cfg.t1, int((cfg.t1 - 100) * 100)), fn=save_fn
        )
        cfg.get_balance_rate = lambda t, state, args: jnp.max(
            jnp.array([10 - t * 9.0 / 100, 1.0])
        )
        cfg.balance = args.balance
        cfg.initial_input_weight = init_w[i]
        cfg.initial_rec_weight = init_w[i]

    start = time.time()
    sol, models = run_batched_simulation(
        configs, save_results=True, return_final_state=True
    )
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
