import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_snn.models import SystemState
from adaptive_snn.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_snn.simulation_configs.single_synapse_learning_AC import (
    create_single_synapse_learning_config,
)
from adaptive_snn.utils.runner import run_simulation


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta_V", type=float, default=0.001, help="Steepness of the gating function"
    )

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

    if args.delta_V == 0.0:
        model_cls = EligibilityLIFNetwork
    else:
        model_cls = GatedLIFNetwork

    cfg = create_single_synapse_learning_config(
        network_cls=model_cls,
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

    def save_fn(t, x: SystemState, args):
        RPE = x.agent_state.RPE.astype(jnp.float32)
        reward_noise = x.environment_state.reward_noise.astype(jnp.float32)
        eligibility = x.agent_state.network_state.features.eligibility[0, -1].astype(
            jnp.float32
        )
        return (RPE, reward_noise, eligibility)

    cfg.t1 = 10100
    start_save = 100
    cfg.save_at = SaveAt(
        ts=jnp.linspace(start_save, cfg.t1, int(1e3 * (cfg.t1 - start_save))),
        fn=save_fn,
    )

    cfg.save_file = args.output_file

    cfg.delta_V = args.delta_V

    sol, model = run_simulation(cfg, save_results=False)

    RPE, reward_noise, eligibility = (
        sol.ys[0].squeeze(),
        sol.ys[1].squeeze(),
        sol.ys[2].squeeze(),
    )
    dW_task = (eligibility * RPE).squeeze()
    dW_noise = (eligibility * reward_noise).squeeze()

    alignment = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_task))
    snr = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_noise))
    snr_raw = jnp.sum(jnp.abs(dW_task)) / jnp.sum(jnp.abs(dW_noise))

    jnp.savez(
        args.output_file,
        alignment=alignment,
        snr=snr,
        snr_raw=snr_raw,
    )

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
