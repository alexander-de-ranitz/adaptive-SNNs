import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.signal import welch

from adaptive_SNN.visualization.utils import (
    get_LIF_state,
    plot_membrane_potential,
    plot_voltage_distribution,
)


def _plot_frequency_spectrum(ax, t, state, model, neurons_to_plot=None):
    V = get_LIF_state(state).V
    N_timepoints = t.shape[0]
    dt = t[1] - t[0]

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    for neuron_idx in neurons_to_plot:
        f, Pxx = welch(V[:, neuron_idx], fs=1 / dt, nperseg=min(1 / dt, N_timepoints))
        mask = f < 150  # Limit to 150 Hz for better visualization
        ax.plot(f[mask], Pxx[mask], label=f"Neuron {neuron_idx}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log scale)")
    ax.set_yscale("log")
    ax.legend()


def plot_voltage_spectrum(
    sol,
    model,
    t0: float,
    t1: float,
    dt0: float,
    neurons_to_plot: jnp.ndarray | None = None,
):
    t = sol.ts
    state = sol.ys
    V = get_LIF_state(state).V

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    fig, axs = plt.subplots(3, 1, figsize=(8, 5))

    # Plot membrane potential
    plot_membrane_potential(axs[0], t, state, model, neurons_to_plot=neurons_to_plot)
    axs[0].set_title("Neuron Membrane Potential")

    # Plot voltage distribution
    plot_voltage_distribution(axs[1], t, state, model, neurons_to_plot=neurons_to_plot)
    axs[1].set_title("Voltage Distribution")

    # Plot freq spectrum
    _plot_frequency_spectrum(axs[2], t, state, model, neurons_to_plot=neurons_to_plot)
    axs[2].set_title("Voltage Frequency Spectrum")

    plt.tight_layout()
    plt.show()
