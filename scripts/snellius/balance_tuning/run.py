import time

import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability
import argparse

import diffrax as dfx
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.networks import LIFNetwork
from adaptive_SNN.models.networks.network_handler import NetworkHandler
from adaptive_SNN.solver import solve_ODE
from adaptive_SNN.utils.metrics import compute_CV_ISI


def main():
    parser = argparse.ArgumentParser(
        description="Run single neuron balancing simulation with specified parameters"
    )
    parser.add_argument(
        "--output_file",
        type=str,
    )
    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1.025,
        help="Balance parameter for the simulation",
    )
    parser.add_argument(
        "--initial_weight",
        type=float,
        default=1.4,
        help="Initial weight for the simulation",
    )
    script_args = parser.parse_args()

    t0 = 0
    t1 = 1000
    dt0 = 1e-4
    key = jr.PRNGKey(script_args.key_seed)
    balance = script_args.balance

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    min_noise_std = 0.0
    noise_level = 0.0
    N_neurons = 1
    N_inputs = 2

    initial_weight_matrix = jnp.array(
        [[jnp.nan, script_args.initial_weight, script_args.initial_weight * 10]]
    )
    input_types = jnp.array([1, 0])  # 1 for excitatory, 0 for inhibitory

    # Set up models
    model = NetworkHandler(
        network=LIFNetwork(
            N_neurons=N_neurons,
            N_inputs=N_inputs,
            fully_connected_input=True,
            fraction_excitatory_input=0.5,
            input_types=input_types,
            min_noise_std=min_noise_std,
            initial_weight_matrix=initial_weight_matrix,
            rec_weight_std=0.0,
            input_weight_std=0.0,
            key=key,
            dt=dt0,
            balance_rate=1.0,
        )
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    def get_spikes(t, x, args):
        return jr.poisson(
            jr.fold_in(key, jnp.rint(t / dt0)),
            rates * dt0,
            shape=(N_neurons, N_inputs),
        )

    args = {
        "input_spike_fn": get_spikes,
        "get_desired_balance": lambda t, x, args: jnp.array([balance]),
        "noise_scale_hyperparam": noise_level,
    }

    def save_fn(t, state, args):
        return (state.V, state.S, state.charge_in, state.charge_out)

    start = time.time()
    sol = solve_ODE(
        model,
        solver,
        t0,
        t1,
        dt0,
        init_state,
        save_at=SaveAt(ts=jnp.arange(100, t1, dt0), fn=save_fn),
        args=args,
    )
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")

    V, S, charge_in, charge_out = sol.ys
    charge_ratio = jnp.mean(
        jnp.where(
            (jnp.abs(charge_out) > 0) & (jnp.abs(charge_in) > 0),
            charge_in / jnp.abs(charge_out),
            1.0,
        )
    )
    cv_isi = compute_CV_ISI(S, sol.ts)
    firing_rate = jnp.sum(S, axis=0) / (t1 - t0)
    mean_voltage = jnp.mean(V) * 1e3

    jnp.savez(
        script_args.output_file,
        CV_ISI=cv_isi,
        firing_rate=firing_rate,
        charge_ratio=charge_ratio,
        mean_voltage=mean_voltage,
    )


if __name__ == "__main__":
    main()
