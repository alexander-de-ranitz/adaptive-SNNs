import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.utils.plotting import plot_simulate_SNN_results
from adaptive_SNN.utils.solver import simulate_noisy_SNN


def main():
    t0 = 0
    t1 = 1
    dt0 = 0.0001
    key = jr.PRNGKey(1)

    N_neurons = 1
    N_inputs = 2
    input_types = jnp.concatenate(
        [jnp.ones((int(N_inputs * 0.8),)), jnp.zeros((int(N_inputs * 0.2),))], axis=0
    )  # First half excitatory, second half inhibitory
    input_types = jnp.array([1, 0])  # First excitatory, second inhibitory
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=input_types,
        fully_connected_input=True,
        key=key,
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=0.0, noise_scale=0e-8, mean=0e-8, dim=N_neurons)
    noise_I_model = OUP(theta=0.0, noise_scale=0e-8, mean=0e-8, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    def get_desired_balance(t, x, args):
        return jnp.where(
            t < 0.25, 0.0, jnp.where(t < 0.5, 5.0, jnp.where(t < 0.75, 10.0, 15.0))
        )

    # Input spikes: Poisson with rate 20 Hz
    rate = 100  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process
    args = {
        "get_input_spikes": lambda t, x, args: jr.bernoulli(
            jr.PRNGKey((t / dt0).astype(int)), p=p, shape=(N_inputs,)
        ),
        "get_desired_balance": get_desired_balance,
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_SNN_results(sol, model, t0, t1, dt0)


if __name__ == "__main__":
    main()
