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
from adaptive_SNN.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_SNN.simulation_configs.single_synapse_config import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta_V", type=float, default=0.001, help="Steepness of the gating function"
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file name"
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

    def save_fn(t, x: SystemState, args):
        reward = x.environment_state.reward.astype(jnp.float32)
        reward_noise = x.environment_state.reward_noise.astype(jnp.float32)
        eligibility = x.agent_state.network_state.features.eligibility.astype(
            jnp.float32
        )
        return (reward, reward_noise, eligibility)

    cfg.t1 = 2500
    cfg.save_at = SaveAt(
        ts=jnp.linspace(cfg.t0, cfg.t1, 1e3 * cfg.t1),
        fn=save_fn,
    )

    cfg.args.update({"delta_V": args.delta_V})

    sol, model = run_simulation(cfg, save_results=True)

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
