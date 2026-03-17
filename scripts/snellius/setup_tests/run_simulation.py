import argparse
import time

from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import LIFNetwork
from adaptive_SNN.utils.save_helper import save_part_of_state
from scripts.helpers import SimulationConfig, run_simulation


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_weight", type=float, default=5.0, help="Initial weight factor"
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.0, help="Noise level hyperparameter"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=2.0,
        help="Desired balance of excitatory/inhibitory input",
    )
    parser.add_argument("--output_file", type=str, help="Output file name")
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")

    args = parser.parse_args()

    print(
        f"Running simulation with initial_weight={args.initial_weight}, balance={args.balance}, noise_level={args.noise_level}, key_seed={args.key_seed}. File will be saved to {args.output_file}"
    )

    def save_fn(t, x, args):
        return save_part_of_state(x, environment_state=True, reward_signal=True)

    run_simulation(
        SimulationConfig(
            model=LIFNetwork,
            t0=0,
            t1=250,
            dt=1e-4,
            N_neurons=1,
            N_inputs=500,
            fraction_excitatory_input=0.8,
            input_types=None,
            rates=10.0,
            warmup_time=50,
            reward_rate=0.1,
            save_at=SaveAt(ts=jnp.linspace(50, 250, 200), fn=save_fn),
            save_file=args.output_file,
            balance=args.balance,
            noise_level=args.noise_level,
            lr=0.0,
            initial_weight=args.initial_weight,
            weight_std=0.0,
            key_seed=args.key_seed,
            save_results=True,
        ),
    )

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
