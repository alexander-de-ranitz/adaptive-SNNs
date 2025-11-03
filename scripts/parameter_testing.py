import diffrax as dfx
import equinox as eqx
import jax.random as jr
import matplotlib as mpl
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    OUP,
    LIFNetwork,
    NoisyNetwork,
)
from adaptive_SNN.models.metrics import compute_charge_ratio
from adaptive_SNN.solver import simulate_noisy_SNN

mpl.rcParams["savefig.directory"] = "../figures"


def main():
    t0 = 0
    t1 = 1.0
    dt0 = 1e-5
    key = jr.PRNGKey(1)
    N_neurons = 1
    N_inputs = 2

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt0,
        input_neuron_types=jnp.array([1, 0]),
        fully_connected_input=True,
        key=key,
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(tau=250.0, noise_scale=100e-9, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(tau=250.0, noise_scale=100e-9, mean=0.0, dim=N_neurons)

    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    solver = dfx.EulerHeun()

    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rate = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    balances = [0.05, 0.1, 0.5, 1, 2.0]
    weight_factor = [0.1, 0.5, 0.65, 0.8, 1]
    data = []
    for b in balances:
        for w in weight_factor:
            init_state = eqx.tree_at(
                lambda x: x.network_state.W,
                model.initial,
                model.initial.network_state.W * w,
            )

            print(f"Simulating for balance {b} and weight {w}")

            # Define args
            args = {
                "get_input_spikes": lambda t, x, args: jr.poisson(
                    jr.PRNGKey((t / dt0).astype(int)), rate * dt0, shape=(N_inputs,)
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
            balance_ratio = compute_charge_ratio(
                sol.ts,
                sol.ys,
                model,
            )[0]
            data.append((b, w, voltage_trace, spikes, balance_ratio))

    # Plot voltage traces
    fig, ax = plt.subplots(
        len(balances), len(weight_factor), figsize=(15, 10), sharex=True, sharey=True
    )
    for b_idx, b in enumerate(balances):
        for w_idx, w in enumerate(weight_factor):
            idx = b_idx * len(weight_factor) + w_idx
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
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    firing_rates = jnp.array(
        [
            [
                jnp.sum(data[b_idx * len(weight_factor) + w_idx][3]) / (t1 - t0)
                for w_idx in range(len(weight_factor))
            ]
            for b_idx in range(len(balances))
        ]
    )
    mean_potentials = jnp.array(
        [
            [
                jnp.mean(data[b_idx * len(weight_factor) + w_idx][2])
                for w_idx in range(len(weight_factor))
            ]
            for b_idx in range(len(balances))
        ]
    )
    stddev_potentials = jnp.array(
        [
            [
                jnp.std(data[b_idx * len(weight_factor) + w_idx][2])
                for w_idx in range(len(weight_factor))
            ]
            for b_idx in range(len(balances))
        ]
    )
    balance_ratios = jnp.array(
        [
            [
                data[b_idx * len(weight_factor) + w_idx][4]
                for w_idx in range(len(weight_factor))
            ]
            for b_idx in range(len(balances))
        ]
    )

    make_heatmap_with_values(
        axs[0][0],
        firing_rates,
        weight_factor,
        balances,
        "Firing Rates (Hz)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    make_heatmap_with_values(
        axs[0][1],
        mean_potentials * 1e3,
        weight_factor,
        balances,
        "Mean Membrane Potential (mV)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    make_heatmap_with_values(
        axs[1][0],
        stddev_potentials * 1e3,
        weight_factor,
        balances,
        "Std Dev of Membrane Potential (mV)",
        "Synaptic Weight",
        "I/E Balance",
        value_format="{:.1f}",
    )
    make_heatmap_with_values(
        axs[1][1],
        balance_ratios,
        weight_factor,
        balances,
        "Measured I/E Balance",
        "Synaptic Weight",
        "Desired I/E Balance",
        value_format="{:.2f}",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
