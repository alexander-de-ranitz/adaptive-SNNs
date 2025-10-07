import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models.models import (
    OUP,
    LIFNetwork,
    LIFState,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.utils.solver import simulate_noisy_SNN


def main():
    t0 = 0
    t1 = 1
    dt0 = 0.0001
    key = jr.PRNGKey(1)
    N_neurons = 1
    N_inputs = 2

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=jnp.array([1, 0]),
        fully_connected_input=True,
        key=key,
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    solver = dfx.EulerHeun()
    init_state = model.initial

    # Input spikes: Poisson with rate 20 Hz
    rate = 500  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process

    balances = [0.5, 1, 2, 5]
    weights = [5, 10, 15, 20]
    data = []
    for b in balances:
        for w in weights:
            network_state, E, I = (
                model.initial.network_state,
                model.initial.noise_E_state,
                model.initial.noise_I_state,
            )
            init_state = NoisyNetworkState(
                LIFState(
                    network_state.V,
                    network_state.S,
                    network_state.W * w,
                    network_state.G,
                ),
                E,
                I,
            )

            print(f"Simulating for balance {b} and weight {w}")

            # Define args
            args = {
                "get_input_spikes": lambda t, x, args: jr.bernoulli(
                    jr.PRNGKey((t / dt0).astype(int)), p=p, shape=(N_inputs,)
                ),
                "get_desired_balance": lambda t, x, args: jnp.array(
                    b
                ),  # Desired E/I balance
            }

            sol = simulate_noisy_SNN(
                model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
            )
            voltage_trace = sol.ys.network_state.V[:, 0]
            spikes = sol.ys.network_state.S[:, 0]
            data.append((b, w, voltage_trace, spikes))

    # Plot voltage traces
    fig, ax = plt.subplots(
        len(balances), len(weights), figsize=(15, 10), sharex=True, sharey=True
    )
    for b_idx, b in enumerate(balances):
        for w_idx, w in enumerate(weights):
            idx = b_idx * len(weights) + w_idx
            t = sol.ts
            V = data[idx][2]
            S = data[idx][3]
            spike_times = t[S > 0]
            ax[b_idx, w_idx].vlines(spike_times, V[S > 0] * 1e3, -40, color="k")
            ax[b_idx, w_idx].plot(t, V * 1e3, c="k")
            ax[b_idx, w_idx].set_title(f"Balance {b}, Weight {w}")
            ax[b_idx, w_idx].set_xlabel("Time (s)")
            ax[b_idx, w_idx].set_ylabel("Membrane Potential (mV)")
            ax[b_idx, w_idx].label_outer()
    fig.suptitle("Neuron Membrane Potentials for Different I/E Balances and Weights")
    plt.tight_layout()
    plt.show()

    def make_heatmap_with_values(
        ax,
        matrix,
        x_labels,
        y_labels,
        title,
        xlabel,
        ylabel,
        value_format="{:.1f}",
        cmap="viridis",
    ):
        # cax = ax.matshow(matrix, cmap=cmap)
        ax.matshow(matrix, cmap=cmap)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                ax.text(
                    j,
                    i,
                    value_format.format(val),
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=9,
                )
        # fig.colorbar(cax)
        ax.set_xticks(jnp.arange(len(x_labels)))
        ax.set_yticks(jnp.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Heatmap of firing rates
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))
    firing_rates = jnp.array(
        [
            [
                jnp.sum(data[b_idx * len(weights) + w_idx][3]) / (t1 - t0)
                for w_idx in range(len(weights))
            ]
            for b_idx in range(len(balances))
        ]
    )
    mean_potentials = jnp.array(
        [
            [
                jnp.mean(data[b_idx * len(weights) + w_idx][2])
                for w_idx in range(len(weights))
            ]
            for b_idx in range(len(balances))
        ]
    )
    stddev_potentials = jnp.array(
        [
            [
                jnp.std(data[b_idx * len(weights) + w_idx][2])
                for w_idx in range(len(weights))
            ]
            for b_idx in range(len(balances))
        ]
    )

    make_heatmap_with_values(
        axs[0],
        firing_rates,
        weights,
        balances,
        "Firing Rates (Hz)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    make_heatmap_with_values(
        axs[1],
        mean_potentials * 1e3,
        weights,
        balances,
        "Mean Membrane Potential (mV)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    make_heatmap_with_values(
        axs[2],
        stddev_potentials * 1e3,
        weights,
        balances,
        "Std Dev of Membrane Potential (mV)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
