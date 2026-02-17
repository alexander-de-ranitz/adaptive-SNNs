import argparse
import time

from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import LIFNetwork
from adaptive_SNN.utils.save_helper import save_part_of_state
from scripts.helpers import run_rate_learning_simulation_fixed_I_weight


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
        return save_part_of_state(x, S=True, environment_state=True, reward=True)

    # run_rate_learning_simulation(
    #     t0=0,
    #     t1=200,
    #     dt=1e-4,
    #     save_at=SaveAt(steps=True, fn=save_fn),
    #     output_file=args.output_file,
    #     balance=args.balance,
    #     noise_level=args.noise_level,
    #     key_seed=args.key_seed,
    #     initial_weight=args.initial_weight,
    #     lr=0.0,
    #     iterations=1,
    # )

    default_weight = 7.5
    curr_balance = (
        7.5
        * jnp.abs(LIFNetwork.reversal_potential_I - LIFNetwork.resting_potential)
        * LIFNetwork.tau_I
    ) / (
        default_weight
        * jnp.abs(LIFNetwork.reversal_potential_E - LIFNetwork.resting_potential)
        * LIFNetwork.tau_E
        + 1e-12
    )
    adjust_ratio = args.balance / curr_balance
    I_weight = adjust_ratio * default_weight

    new_balance = (
        I_weight
        * jnp.abs(LIFNetwork.reversal_potential_I - LIFNetwork.resting_potential)
        * LIFNetwork.tau_I
        + 1e-12
    ) / (
        args.initial_weight
        * jnp.abs(LIFNetwork.reversal_potential_E - LIFNetwork.resting_potential)
        * LIFNetwork.tau_E
    )

    print(
        f"I weight = {I_weight:.4f}, E weight = {args.initial_weight:.4f}, new balance = {new_balance:.4f}, desired balance = {args.balance:.4f}"
    )

    run_rate_learning_simulation_fixed_I_weight(
        t0=0,
        t1=200,
        dt=1e-4,
        save_at=SaveAt(steps=True, fn=save_fn),
        output_file=args.output_file,
        noise_level=args.noise_level,
        key_seed=args.key_seed,
        initial_weight_E=args.initial_weight,
        initial_weight_I=I_weight,
        lr=0.0,
        iterations=1,
    )
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
