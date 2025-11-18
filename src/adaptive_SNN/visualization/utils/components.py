import jax.numpy as jnp
from diffrax import Solution
from jaxtyping import Array
from matplotlib.pyplot import Axes
from scipy.signal import welch

from adaptive_SNN.models import LIFNetwork
from adaptive_SNN.visualization.utils.adapters import (
    get_LIF_model,
    get_LIF_state,
    get_noisy_network_model,
    get_noisy_network_state,
)


def _plot_membrane_potential(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    **plot_kwargs,
):
    lif_state = get_LIF_state(sol.ys)
    V = lif_state.V
    S = lif_state.S
    t = sol.ts

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    # Plot membrane potentials
    for i in neurons_to_plot:
        spike_times = t[S[:, i] > 0]
        ax.vlines(
            spike_times,
            V[:, i][S[:, i] > 0] * 1e3,
            -40,
        )
        ax.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V", **plot_kwargs)
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title("Neuron Membrane Potential")


def _plot_spikes_raster(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    **plot_kwargs,
):
    lif_network = get_LIF_model(model)
    lif_state = get_LIF_state(sol.ys)

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_network.N_neurons)

    spikes = lif_state.S[:, neurons_to_plot]

    t = sol.ts
    dt = t[1] - t[0]

    # Build spike times per neuron (reverse order to show I neurons at bottom)
    spike_times_per_neuron = [
        jnp.nonzero(spikes[:, i])[0] * dt for i in range(spikes.shape[1])
    ][::-1]
    ax.set_yticks(range(len(spike_times_per_neuron)))
    ax.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8, **plot_kwargs)
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Spike Raster Plot")


def _plot_conductances(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    split_noise: bool = False,
    **plot_kwargs,
):
    base_network = get_LIF_model(model)
    network_state = get_LIF_state(sol.ys)

    N_neurons = base_network.N_neurons
    W = jnp.where(~jnp.isfinite(network_state.W), 0.0, network_state.W)
    G = network_state.G
    exc_mask = base_network.excitatory_mask

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(N_neurons)

    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    t = sol.ts

    if split_noise:
        if isinstance(model, LIFNetwork):
            raise ValueError(
                "Cannot split noise conductances for LIFNetwork model; no noise present."
            )

        noisy_network_state = get_noisy_network_state(sol.ys)
        noise = noisy_network_state.noise_state

        ax.plot(
            t,
            weighed_G_excitatory[:, neurons_to_plot],
            label="Synaptic E Conductance",
            color="g",
            **plot_kwargs,
        )
        ax.plot(
            t,
            weighed_G_inhibitory[:, neurons_to_plot],
            label="Synaptic I Conductance",
            color="r",
            **plot_kwargs,
        )
        ax.plot(
            t,
            noise[:, neurons_to_plot],
            label="Noise E Conductance",
            color="g",
            linestyle=":",
            **plot_kwargs,
        )
    else:
        if isinstance(model, LIFNetwork):
            total_G_excitatory = weighed_G_excitatory
            total_G_inhibitory = weighed_G_inhibitory
        else:
            noisy_network_state = get_noisy_network_state(sol.ys)
            noise = noisy_network_state.noise_state
            total_G_excitatory = weighed_G_excitatory + noise
            total_G_inhibitory = weighed_G_inhibitory

        ax.plot(
            t,
            total_G_excitatory[:, neurons_to_plot],
            label="Total E Conductance",
            color="g",
            **plot_kwargs,
        )
        ax.plot(
            t,
            total_G_inhibitory[:, neurons_to_plot],
            label="Total I Conductance",
            color="r",
            **plot_kwargs,
        )
    ax.set_ylabel("Total Conductance (S)")
    ax.set_title("Total Conductances")


def _plot_voltage_distribution(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    **plot_kwargs,
):
    V = get_LIF_state(sol.ys).V

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    for i in neurons_to_plot:
        ax.hist(
            V[:, i] * 1e3,
            bins=50,
            alpha=0.5,
            label=f"Neuron {i + 1}",
            density=True,
            **plot_kwargs,
        )
    ax.set_xlabel("Membrane Potential (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Membrane Potential Distribution")
    ax.set_xlim(jnp.min(V) * 1e3 - 5, jnp.max(V) * 1e3 + 5)


def _plot_ISI_distribution(
    ax: Axes, sol: Solution, model, neurons_to_plot: Array | None = None, **plot_kwargs
):
    """Plot the inter-spike interval (ISI) distribution for the network."""
    lif_state = get_LIF_state(sol.ys)
    lif_model = get_LIF_model(model)
    dt = sol.ts[1] - sol.ts[0]

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_model.N_neurons)

    spikes = lif_state.S[:, neurons_to_plot]
    spike_times = jnp.flatnonzero(spikes)
    ISIs = jnp.diff(spike_times) * dt  # Convert to time using dt
    ax.hist(ISIs, bins=50, density=True, **plot_kwargs)
    ax.set_xlabel("Inter-Spike Interval (s)")
    ax.set_ylabel("Density")


def _plot_spike_rates(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    moving_window_duration: float = 0.1,
    **plot_kwargs,
):
    """Plot the mean network spike rate over time.

    The spike rate is computed using a moving window centered at the current timepoint.
    """
    lif_state = get_LIF_state(sol.ys)
    lif_model = get_LIF_model(model)

    t = sol.ts
    dt = t[1] - t[0]

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_model.N_neurons)

    spikes = lif_state.S[:, neurons_to_plot]
    total_rec_spikes_per_timestep = jnp.sum(spikes, axis=1).squeeze()
    rec_spike_rate = (
        jnp.convolve(
            total_rec_spikes_per_timestep,
            jnp.ones(shape=(int(1 / dt * moving_window_duration))),
            mode="same",
        )
        / moving_window_duration
        / jnp.size(neurons_to_plot)
    )
    ax.plot(t, rec_spike_rate, **plot_kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike Rate (Hz)")
    ax.set_title("Mean Network Spike Rate")


def _plot_spike_rate_distributions(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    **plot_kwargs,
):
    """Plot the distribution of mean spike rates across neurons."""
    lif_state = get_LIF_state(sol.ys)
    lif_model = get_LIF_model(model)

    t = sol.ts
    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_model.N_neurons)

    spikes = lif_state.S[:, neurons_to_plot]
    spike_rates = jnp.sum(spikes, axis=0) / (t[-1] - t[0])
    ax.hist(
        spike_rates,
        bins=jnp.arange(0, jnp.ceil(jnp.max(spike_rates)), step=1 / (t[-1] - t[0])),
        density=True,
        **plot_kwargs,
    )
    ax.set_xlabel("Spike Rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Spike Rate Distribution")


def _plot_frequency_spectrum(
    ax: Axes,
    sol: Solution,
    signals,
    labels,
    **plot_kwargs,
):
    t = sol.ts
    N_timepoints = t.shape[0]
    dt = t[1] - t[0]

    for i, signal in enumerate(signals):
        f, Pxx = welch(signal, fs=1 / dt, nperseg=min(1 / dt, N_timepoints))
        f = f[1:]  # Remove DC component
        Pxx = Pxx[1:]
        mask = f < 150  # Limit to 150 Hz for better visualization, remove DC component
        ax.plot(f[mask], Pxx[mask] / jnp.sum(Pxx[mask]), label=labels[i], **plot_kwargs)

    ax.set_xlabel("Log Frequency (Hz)")
    ax.set_ylabel("Log Power Density")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


def _plot_voltage_frequency_spectrum(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    **plot_kwargs,
):
    V = get_LIF_state(sol.ys).V
    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])
    signals = [V[:, i] for i in neurons_to_plot]
    labels = [f"Neuron {i + 1}" for i in neurons_to_plot]
    _plot_frequency_spectrum(ax, sol, signals, labels, **plot_kwargs)


def _plot_conductance_frequency_spectrum(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    plot_noise: bool = False,
    **plot_kwargs,
):
    lif_state = get_LIF_state(sol.ys)
    G = lif_state.G
    W = jnp.where(~jnp.isfinite(lif_state.W), 0.0, lif_state.W)

    exc_mask = get_LIF_model(model).excitatory_mask
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(weighed_G_excitatory.shape[1])

    signals = [weighed_G_excitatory[:, i] for i in neurons_to_plot]
    labels = [f"Neuron {i + 1} Syn E Conductance" for i in neurons_to_plot]
    _plot_frequency_spectrum(ax, sol, signals, labels, **plot_kwargs)

    # Plot the empirical and expected spectrum of the noise
    if plot_noise:
        noisy_network_state = get_noisy_network_state(sol.ys)
        noise = noisy_network_state.noise_state
        noise_signals = [noise[:, i] for i in neurons_to_plot]
        noise_labels = [f"Neuron {i + 1} Noise E Conductance" for i in neurons_to_plot]
        _plot_frequency_spectrum(ax, sol, noise_signals, noise_labels, **plot_kwargs)

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
