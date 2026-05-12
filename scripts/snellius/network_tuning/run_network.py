import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

from jax import numpy as jnp
from jax import random as jr

from adaptive_SNN.simulation_configs.network_config import create_network_config
from adaptive_SNN.utils.runner import run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Run network simulation with specified parameters"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="Balance parameter to adjust the ratio of inhibitory to excitatory synaptic weights",
    )
    parser.add_argument(
        "--N_external_inputs",
        type=int,
        default=200,
        help="Number of external inputs to the network",
    )
    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )
    parser.add_argument(
        "--rec_weight",
        type=float,
        default=1.0,
        help="Initial recurrent synaptic weight",
    )
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    cfg = create_network_config(N_neurons=1000, key=jr.PRNGKey(args.key_seed))
    cfg.connection_prob_E = 0.1
    cfg.t1 = 2.5
    cfg.initial_rec_weight = args.rec_weight
    cfg.initial_input_weight = 1.0
    cfg.save_file = args.output_file

    cfg.balance = args.balance

    N_E_in = args.N_external_inputs
    cfg.args.update({"N_simulated_E_inputs": N_E_in})

    input_rate = jnp.array([N_E_in * 10])  # High frequency background input

    spike_key = jr.fold_in(cfg.key, 1337)

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint((t - cfg.t0) / cfg.dt), dtype=jnp.int64)
        return jr.poisson(
            jr.fold_in(spike_key, step_idx),
            input_rate * cfg.dt,
            shape=(cfg.N_neurons, cfg.N_inputs),
        )

    cfg.input_spike_fn = input_spike_fn

    start = time.time()
    sol, model = run_simulation(cfg, save_results=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
