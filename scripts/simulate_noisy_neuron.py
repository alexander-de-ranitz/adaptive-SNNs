import diffrax as dfx
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.visualization import plot_simulate_SNN_results


def main():
    t0 = 0
    t1 = 25
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    min_noise_std = 5e-9
    N_neurons = 1
    N_inputs = 2

    input_weight = 5.0

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        fully_connected_input=True,
        fraction_excitatory_input=0.5,
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

    args = {
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.fold_in(jr.PRNGKey(0), jnp.int32(jnp.round(t / dt0))),
            rates * dt0,
            shape=(N_neurons, N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([5.0]),
        "noise_scale_hyperparam": 0.2,
    }

    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        init_state,
        save_at=SaveAt(t0=True, t1=True, steps=True),
        args=args,
    )

    plot_simulate_SNN_results(sol, model, split_noise=True)


if __name__ == "__main__":
    main()
