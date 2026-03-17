import time

import diffrax as dfx
import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_simulate_SNN_results


def main():
    t0 = 0
    t1 = 10
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    min_noise_std = 5e-9
    noise_level = 0.1
    N_neurons = 1
    N_inputs = 2

    input_weight = 20.0

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        fully_connected_input=True,
        fraction_excitatory_input=0.5,
        input_types=jnp.array([1, 0]),
        weight_std=0.0,
        input_weight=input_weight,
        key=key,
        dt=dt0,
    )

    key, _ = jr.split(key)
    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model, noise_model=noise_model, min_noise_std=min_noise_std
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
        "get_input_spikes": get_spikes,
        "get_desired_balance": lambda t, x, args: jnp.array([1.75]),
        "noise_scale_hyperparam": noise_level,
    }

    def save_fn(t, state, args):
        return save_part_of_state(
            state,
            V=True,
            G=True,
            W=True,
            S=True,
            noise_state=True,
            mean_E_conductance=True,
            var_E_conductance=True,
        )

    start = time.time()
    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        init_state,
        save_at=SaveAt(t0=True, t1=True, steps=True, fn=save_fn),
        args=args,
    )
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")

    plot_simulate_SNN_results(sol, model, split_noise=True)


if __name__ == "__main__":
    main()
