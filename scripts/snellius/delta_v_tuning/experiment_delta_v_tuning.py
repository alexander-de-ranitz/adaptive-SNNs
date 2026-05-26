import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_SNN.models.networks.coupled import (
    CoupledNoiseEligibilityLIFNetwork,
    CoupledNoiseGatedLIFNetwork,
)
from adaptive_SNN.simulation_configs.single_synapse_learning import (
    create_default_config_single_synapse_task,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


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

    cfg = create_default_config_single_synapse_task(
        RPE_noise_rate=1.0,
        key=jr.PRNGKey(args.key_seed),
    )

    cfg.save_at = SaveAt(
        ts=jnp.linspace(cfg.t0, cfg.t1, 50000),
        fn=lambda t, x, args: save_part_of_state(
            x,
            RPE=True,
            reward_noise=True,
            eligibility=True,
        ),
    )

    cfg.t1 = 300
    cfg.args.update({"delta_V": args.delta_V})
    if args.delta_V == 0.0:
        print("NO GATING")
        model_cls = CoupledNoiseEligibilityLIFNetwork
    else:
        model_cls = CoupledNoiseGatedLIFNetwork

    file_suffix = "_gated" if model_cls == CoupledNoiseGatedLIFNetwork else "_default"
    cfg.save_file = (
        args.output_file if args.output_file is not None else ""
    ) + file_suffix
    print(f"Running simulation with model class {model_cls.__name__}...")
    cfg.network_cls = model_cls
    sol, model = run_simulation(cfg, save_results=True)
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
