import jax.numpy as jnp
import matplotlib as mpl
from diffrax import Solution
from jaxtyping import Array
from matplotlib import pyplot as plt

from adaptive_SNN.models.models import NoisyNeuronModel

mpl.rcParams["savefig.directory"] = "../figures"


def plot_simulate_noisy_SNN_results(
    sol: Solution,
    spikes: Array,
    model: NoisyNeuronModel,
    t0: float,
    t1: float,
    dt0: float,
):
    # Get results
    t = sol.ts
    (V, W, G), noise_E, noise_I = sol.ys

    G_inhibitory = (
        jnp.sum(G * jnp.invert(model.network.excitatory_mask[None, None, :]), axis=-1)
        + noise_I
    )
    G_excitatory = (
        jnp.sum(G * model.network.excitatory_mask[None, None, :], axis=-1) + noise_E
    )
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot membrane potentials
    for i in range(model.network.N_neurons):
        spike_times = t[spikes[:, i] > 0]
        ax1.vlines(
            spike_times,
            V[:, i][spikes[:, i] > 0] * 1e3,
            -40,
        )
        ax1.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V")
    ax1.set_ylabel("Membrane Potential (mV)")
    ax1.set_title("Neuron Membrane Potential")

    # Plot total conductance of neuron 0
    ax2.plot(t, G_excitatory[:, 0], label="Total E Conductance", color="g")
    ax2.plot(t, G_inhibitory[:, 0], label="Total I Conductance", color="r")
    ax2.plot(t, noise_E[:, 0], label="Noise E Conductance", color="g", linestyle="--")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Total Conductance (S)")
    ax2.set_title("Total Conductances")

    # Plot spikes as raster plot
    spike_times_per_neuron = [
        jnp.nonzero(spikes[:, i])[0] * dt0 for i in range(spikes.shape[1])
    ][::-1]
    ax3.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    ax3.set_yticks(range(len(spike_times_per_neuron)))
    ax3.set_ylabel("Neuron")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Spike Raster Plot")

    if model.network.N_inputs > 0:
        # Shade background to distinguish input vs. main neurons
        N_input = model.network.N_inputs
        ax3.axhspan(-0.5, N_input - 0.5, facecolor="lightgray", alpha=0.3)

    # Set x-axis limits and ticks for all subplots
    xticks = jnp.linspace(t0, t1, 6)  # 6 evenly spaced ticks
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(t0, t1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax.label_outer()

    plt.tight_layout()
    plt.show()
