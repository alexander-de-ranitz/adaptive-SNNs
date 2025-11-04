import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.visualization import plot_simulate_SNN_results


def main():
    t0 = 0
    t1 = 0.5
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    N_neurons = 1
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=0, fully_connected_input=True, key=key, dt=dt0
    )
    key, _ = jr.split(key)
    noise_model = OUP(tau=neuron_model.tau_E, noise_scale=2e-16, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    args = {
        "get_input_spikes": lambda t, x, a: jnp.zeros((model.base_network.N_inputs,)),
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_SNN_results(sol, model, t0, t1, dt0)


if __name__ == "__main__":
    main()
