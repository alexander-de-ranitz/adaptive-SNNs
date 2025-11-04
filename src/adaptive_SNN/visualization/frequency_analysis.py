import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.signal import welch

from adaptive_SNN.visualization.utils import (
    _plot_membrane_potential,
    _plot_voltage_distribution,
    get_LIF_model,
    get_LIF_state,
    get_noisy_network_model,
    get_noisy_network_state,
)


def _plot_frequency_spectrum(ax, t, signals, labels):
    N_timepoints = t.shape[0]
    dt = t[1] - t[0]

    for i, signal in enumerate(signals):
        f, Pxx = welch(signal, fs=1 / dt, nperseg=min(1 / dt, N_timepoints))
        f = f[1:]  # Remove DC component
        Pxx = Pxx[1:]
        mask = f < 150  # Limit to 150 Hz for better visualization, remove DC component
        ax.plot(f[mask], Pxx[mask] / jnp.sum(Pxx[mask]), label=labels[i])

    ax.set_xlabel("Log Frequency (Hz)")
    ax.set_ylabel("Log Power Density")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


def _plot_voltage_frequency_spectrum(ax, t, state, model, neurons_to_plot=None):
    V = get_LIF_state(state).V
    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])
    signals = [V[:, i] for i in neurons_to_plot]
    labels = [f"Neuron {i + 1}" for i in neurons_to_plot]
    _plot_frequency_spectrum(ax, t, signals, labels)


def _plot_conductance_frequency_spectrum(
    ax, t, state, model, neurons_to_plot=None, plot_noise=False
):
    lif_state = get_LIF_state(state)
    G = lif_state.G
    W = jnp.where(~jnp.isfinite(lif_state.W), 0.0, lif_state.W)

    exc_mask = get_LIF_model(model).excitatory_mask
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(weighed_G_excitatory.shape[1])

    signals = [weighed_G_excitatory[:, i] for i in neurons_to_plot]
    labels = [f"Neuron {i + 1} Syn E Conductance" for i in neurons_to_plot]
    _plot_frequency_spectrum(ax, t, signals, labels)

    # Plot the empirical and expected spectrum of the noise
    if plot_noise:
        noisy_network_state = get_noisy_network_state(state)
        noise = noisy_network_state.noise_state
        noise_signals = [noise[:, i] for i in neurons_to_plot]
        noise_labels = [f"Neuron {i + 1} Noise E Conductance" for i in neurons_to_plot]
        _plot_frequency_spectrum(ax, t, noise_signals, noise_labels)

        # Plot expected PSD for OU process
        noise_model = get_noisy_network_model(model).noise_model
        expected_PSD = (
            lambda f: 2
            * noise_model.noise_scale
            * noise_model.tau**2
            / (1 + (2 * jnp.pi * f * noise_model.tau) ** 2)
        )
        f_vals = jnp.linspace(0, 150, 150)
        ax.plot(
            f_vals,
            expected_PSD(f_vals) / jnp.sum(expected_PSD(f_vals)),
            label="Expected OU PSD",
            linestyle="--",
            color="k",
        )
    ax.set_title("Excitatory Conductance PSD")
    ax.legend()


def plot_frequency_analysis(
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

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    # Plot membrane potential
    _plot_membrane_potential(axs[0], t, state, model, neurons_to_plot=neurons_to_plot)
    axs[0].set_title("Neuron Membrane Potential")

    # Plot voltage distribution
    _plot_voltage_distribution(axs[1], t, state, model, neurons_to_plot=neurons_to_plot)
    axs[1].set_title("Voltage Distribution")

    # Plot freq spectrum
    _plot_conductance_frequency_spectrum(
        axs[2], t, state, model, neurons_to_plot=neurons_to_plot, plot_noise=True
    )
    axs[2].set_title("Conductance Frequency Spectrum")

    plt.tight_layout()
    plt.show()
