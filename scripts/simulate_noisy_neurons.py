import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.utils.plotting import plot_simulate_SNN_results
from adaptive_SNN.utils.solver import simulate_noisy_SNN


def main():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 1
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=0, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=50.0, noise_scale=50e-9, mean=25 * 1e-9, dim=N_neurons)
    noise_I_model = OUP(theta=50.0, noise_scale=50e-9, mean=50 * 1e-9, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    args = {
        "get_input_spikes": lambda t, x, a: jnp.zeros((model.base_network.N_inputs,)),
        "get_learning_rate": lambda t, x, a: jnp.array([0.0]),
        "get_desired_balance": lambda t, x, a: 0.0,  # = no balancing
        "RPE": jnp.array([0.0]),
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_SNN_results(sol, model, t0, t1, dt0)


if __name__ == "__main__":
    main()
