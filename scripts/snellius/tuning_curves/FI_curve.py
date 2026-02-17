import argparse
import time

import jax
from diffrax import SaveAt

from adaptive_SNN.utils.save_helper import save_part_of_state
from scripts.helpers import run_noisy_network_simulation


def main():
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

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
        f"Running simulation with initial_weight={args.initial_weight}, balance={args.balance}, noise_level={args.noise_level}, key_seed={args.key_seed}"
    )

    def save_fn(t, x, args):
        return save_part_of_state(x, S=True)

    run_noisy_network_simulation(
        t0=0,
        t1=5,
        dt=1e-4,
        save_at=SaveAt(steps=True, fn=save_fn),
        output_file=args.output_file,
        balance=args.balance,
        noise_level=args.noise_level,
        key_seed=args.key_seed,
        initial_weight=args.initial_weight,
    )
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
