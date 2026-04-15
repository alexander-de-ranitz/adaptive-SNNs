import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import jax.random as jr

from adaptive_SNN.models.networks.coupled import (
    CoupledNoiseEligibilityLIFNetwork,
    CoupledNoiseGatedLIFNetwork,
)
from adaptive_SNN.utils.runner import run_simulation
from scripts.simulation_configs.single_synapse_learning import (
    create_default_config_single_synapse_task,
)


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--noise_level", type=float, default=0.0, help="Noise level hyperparameter"
    )

    parser.add_argument(
        "--reward_noise_rate",
        type=float,
        default=0.0,
        help="Reward noise level hyperparameter",
    )
    parser.add_argument(
        "--min_noise_std",
        type=float,
        default=1e-9,
        help="Minimum noise standard deviation",
    )
    parser.add_argument(
        "--reward_noise_std",
        type=float,
        default=1.0,
        help="Reward noise jump standard deviation",
    )
    parser.add_argument(
        "--input_rate",
        type=float,
        default=50.0,
        help="Input spike rate for the third input channel",
    )
    parser.add_argument(
        "--RPE_decay_tau", type=float, default=0.05, help="RPE decay time constant"
    )
    parser.add_argument(
        "--delta_V", type=float, default=0.001, help="Steepness of the gating function"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0,
        help="Learning rate for weight updates",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file name"
    )
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")

    args = parser.parse_args()

    cfg = create_default_config_single_synapse_task(
        lr=args.learning_rate,
        noise_level=args.noise_level,
        RPE_noise_rate=args.reward_noise_rate,
        key=jr.PRNGKey(args.key_seed),
    )

    cfg.args.update({"delta_V": args.delta_V, "tau_RPE": args.RPE_decay_tau})

    for model_cls in [CoupledNoiseGatedLIFNetwork, CoupledNoiseEligibilityLIFNetwork]:
        file_prefix = "gated" if model_cls == CoupledNoiseGatedLIFNetwork else "default"
        cfg.save_file = file_prefix + (
            args.output_file if args.output_file is not None else ""
        )
        print(f"Running simulation with model class {model_cls.__name__}...")
        cfg.base_network_cls = model_cls
        sol, model = run_simulation(cfg, save_results=True)
        end = time.time()
        print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
