import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_SNN.models import SystemState
from adaptive_SNN.models.networks import LIFNetwork
from adaptive_SNN.simulation_configs.single_synapse_learning_AC import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.metrics import compute_CV_ISI
from adaptive_SNN.utils.runner import run_simulation


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file name"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=1e-9,
        help="Min noise std for the synaptic noise",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=jnp.nan,
        help="Balance parameter for the network",
    )
    parser.add_argument(
        "--initial_weight",
        type=float,
        default=0.0,
        help="Initial weight for the background E synapse",
    )
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")

    args = parser.parse_args()

    cfg = create_single_synapse_learning_config(
        network_cls=LIFNetwork,
        reward_noise_jump_rate=1.0,
        key=jr.PRNGKey(args.key_seed),
    )
    cfg.min_noise_std = args.noise_level
    cfg.noise_level = 0.0
    cfg.balance = args.balance
    cfg.initial_weight_matrix = jnp.tile(
        jnp.array(
            [jnp.nan] * cfg.N_neurons
            + [args.initial_weight, args.initial_weight * 8, args.initial_weight]
        ),
        (cfg.N_neurons, 1),
    )
    cfg.save_file = args.output_file

    def save_fn(t, x: SystemState, args):
        charge_in = x.agent_state.network_state.charge_in
        charge_out = x.agent_state.network_state.charge_out
        balance = jnp.where(
            (charge_in != 0.0) | (charge_out != 0.0),
            (charge_in + charge_out) / (jnp.abs(charge_in) + jnp.abs(charge_out)),
            jnp.nan,
        )
        return (
            x.agent_state.network_state.V,
            x.agent_state.network_state.S.astype(jnp.bool_),
            balance,
        )

    cfg.t1 = 600
    start_save = 100
    cfg.save_at = SaveAt(
        ts=jnp.linspace(start_save, cfg.t1, int(1e4 * (cfg.t1 - start_save))),
        fn=save_fn,
    )

    sol, model = run_simulation(cfg, save_results=False)

    V, S, balance = sol.ys
    ts = sol.ts

    CV_ISI = compute_CV_ISI(S, ts)
    mean_V = jnp.mean(V)
    firing_rate = jnp.sum(S) / (cfg.t1 - start_save)
    mean_balance = jnp.nanmean(balance)

    jnp.savez(
        args.output_file,
        CV_ISI=CV_ISI,
        mean_V=mean_V,
        firing_rate=firing_rate,
        mean_balance=mean_balance,
    )

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
