import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    OUP,
    LIFNetwork,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.analytics import compute_required_input_weight
from adaptive_SNN.utils.metrics import compute_CV_ISI
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization.utils.components import _plot_noise_distribution_STA


def get_default_args():
    """Returns a dictionary of all zero default args for simulate_noisy_SNN."""
    return {
        "get_learning_rate": lambda t, x, args: 0.0,
        "get_input_spikes": lambda t, x, args: jnp.zeros(
            shape=(x.W.shape[1] - x.W.shape[0],)
        ),
        "get_desired_balance": lambda t, x, args: 0.0,
        "RPE": jnp.array(0.0),
    }


def main():
    noise_levels = [0.05, 0.1, 0.5, 2.0]
    fig, axs = plt.subplots(len(noise_levels), 1, figsize=(6, 8))

    for ax, noise_level in zip(axs, noise_levels):
        print(f"Simulating noisy neuron with noise level: {noise_level}")
        simulate_noisy_neuron(ax, noise_level)
    plt.tight_layout()
    plt.show()


def simulate_noisy_neuron(ax, noise_level: float):
    t0 = 0
    t1 = 100
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    N_neurons = 1
    N_inputs = 2

    input_weight = compute_required_input_weight(
        target_mean_g_syn=50e-9,
        N_inputs=1.0,
        tau=LIFNetwork.tau_E,
        input_rate=exc_rate,
        synaptic_increment=LIFNetwork.synaptic_increment,
    )

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        fully_connected_input=True,
        input_weight=input_weight,
        fraction_excitatory_input=0.5,
        key=key,
        dt=dt0,
    )

    key, _ = jr.split(key)
    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    args = {
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey(jnp.int32(jnp.round(t / dt0))),
            rates * dt0,
            shape=(N_neurons, N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([5.0]),
        "noise_scale_hyperparam": noise_level,
    }

    def save_fn(state: NoisyNetworkState):
        return save_part_of_state(state, S=True, noise_state=True)

    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        init_state,
        save_every_n_steps=1,
        args=args,
        save_fn=save_fn,
    )

    noise_std = model.compute_desired_noise_std(0, model.initial, args)

    CV_ISI = compute_CV_ISI(sol.ys.network_state.S)[0]

    _plot_noise_distribution_STA(
        ax,
        sol,
        model,
        noise_std=noise_std,
    )

    ax.set_title(f"Noise Level: {noise_level}, CV_ISI: {CV_ISI:.2f}")


if __name__ == "__main__":
    main()
