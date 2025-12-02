import jax.numpy as jnp
from diffrax import Solution
from jaxtyping import Array
from matplotlib.pyplot import Axes
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.stats import norm

from adaptive_SNN.models import LIFNetwork
from adaptive_SNN.utils.metrics import compute_network_firing_rate
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
        offset = i * 20  # Offset different neurons for visibility
        spike_times = t[S[:, i] > 0]
        ax.vlines(
            spike_times, offset + V[:, i][S[:, i] > 0] * 1e3, offset - 40, colors="k"
        )
        ax.plot(
            t, offset + V[:, i] * 1e3, c="k", label=f"Neuron {i + 1} V", **plot_kwargs
        )
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

    # Build spike times per neuron (reverse order to show I neurons at bottom)
    spike_times_per_neuron = [
        sol.ts[jnp.nonzero(spikes[:, i])[0]] for i in range(spikes.shape[1])
    ][::-1]
    if len(spike_times_per_neuron) < 10:
        ax.set_yticks(range(len(spike_times_per_neuron)))
    else:
        ax.set_yticks([])
    ax.eventplot(
        spike_times_per_neuron,
        colors="black",
        linelengths=0.8,
        linewidths=0.4,
        **plot_kwargs,
    )
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
    split_recurrent: bool = False,
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

    if split_recurrent:
        weighed_G_excitatory = W * G * exc_mask[None, :]
        weighed_G_inhibitory = W * G * jnp.invert(exc_mask[None, :])

        if weighed_G_excitatory.ndim == 2:
            weighed_G_excitatory = weighed_G_excitatory[:, None, :]
            weighed_G_inhibitory = weighed_G_inhibitory[:, None, :]

        total_G_excitatory_recurrent = jnp.sum(
            weighed_G_excitatory[:, :, :N_neurons], axis=-1
        )
        total_G_excitatory_input = jnp.sum(
            weighed_G_excitatory[:, :, N_neurons:], axis=-1
        )
        total_G_inhibitory_recurrent = jnp.sum(
            weighed_G_inhibitory[:, :, :N_neurons], axis=-1
        )
        total_G_inhibitory_input = jnp.sum(
            weighed_G_inhibitory[:, :, N_neurons:], axis=-1
        )
        neurons_to_plot = jnp.array(neurons_to_plot)
        ax.plot(
            sol.ts,
            total_G_excitatory_recurrent[:, neurons_to_plot],
            label="Recurrent Synaptic E Conductance",
            color="g",
            **plot_kwargs,
        )
        ax.plot(
            sol.ts,
            total_G_inhibitory_recurrent[:, neurons_to_plot],
            label="Recurrent Synaptic I Conductance",
            color="r",
            **plot_kwargs,
        )
        ax.plot(
            sol.ts,
            total_G_excitatory_input[:, neurons_to_plot],
            label="Input Synaptic E Conductance",
            color="g",
            linestyle="--",
            **plot_kwargs,
        )
        ax.plot(
            sol.ts,
            total_G_inhibitory_input[:, neurons_to_plot],
            label="Input Synaptic I Conductance",
            color="r",
            linestyle="--",
            **plot_kwargs,
        )
        ax.legend()
        return

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
    ax.set_ylabel("Conductance (S)")
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
        neurons_to_plot = jnp.array([0])

    for i in neurons_to_plot:
        counts, bin_edges, _ = ax.hist(
            V[:, i] * 1e3,
            bins=50,
            alpha=0.5,
            density=True,
            histtype="stepfilled",
            color="darkgreen",
            label="Measured distribution",
            **plot_kwargs,
        )

        # Fit a Gaussian to the voltage distribution
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def gaussian(
            x,
            mu,
            sigma,
        ):
            return norm.pdf(x, mu, sigma)

        (mu, sigma), _ = curve_fit(
            gaussian,
            bin_centers,
            counts,
            p0=[jnp.mean(V[:, i] * 1e3), jnp.std(V[:, i] * 1e3)],
        )
        x = jnp.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        pdf = (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
        ax.plot(
            x,
            pdf,
            linestyle="--",
            c="k",
            label="Gaussian least squares fit",
            **plot_kwargs,
        )
    ax.set_xlabel("Membrane Potential (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Membrane Potential Distribution")
    ax.legend()
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

    ISIs = jnp.array([])
    for i in neurons_to_plot:
        spikes = lif_state.S[:, i]
        spike_times = jnp.nonzero(spikes)[0] * dt
        ISIs = jnp.append(ISIs, jnp.diff(spike_times).flatten())
    ax.hist(ISIs, bins=50, density=True, **plot_kwargs)
    ax.set_xlabel("Inter-Spike Interval (s)")
    ax.set_ylabel("Density")


def _plot_spike_rates(
    ax: Axes,
    sol: Solution,
    model,
    neurons_to_plot: Array | None = None,
    moving_window_duration: float = 0.01,
    **plot_kwargs,
):
    """Plot the mean network spike rate over time.

    The spike rate is computed using a moving window centered at the current timepoint.
    """
    lif_state = get_LIF_state(sol.ys)
    lif_model = get_LIF_model(model)

    t = sol.ts

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_model.N_neurons)

    rec_spike_rate = compute_network_firing_rate(
        lif_state.S[:, neurons_to_plot],
        t,
        moving_window_duration,
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
        # Note: PSD is not normalized
        expected_PSD = lambda f: noise_model.tau**2 / (
            1 + (2 * jnp.pi * f * noise_model.tau) ** 2
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


def _plot_noise_distribution_STA(
    ax: Axes,
    sol,
    model,
    neurons_to_plot: jnp.ndarray | None = None,
    noise_std: float | None = None,
    **plot_kwargs,
):
    """Plot the spike-triggered average (STA) of the noise process

    Providing the noise_level allows overlaying the analytical noise distribution for comparison.
    """
    state = sol.ys
    lif_state = get_LIF_state(state)
    noise_state = get_noisy_network_state(state).noise_state

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(lif_state.S.shape[1])

    spike_times = jnp.nonzero(lif_state.S[:, neurons_to_plot])
    data = []
    for i, neuron_idx in enumerate(neurons_to_plot):
        neuron_spike_times = spike_times[i]
        neuron_noise = noise_state[:, neuron_idx]
        noise_values_at_spikes = neuron_noise[neuron_spike_times]
        data.append(noise_values_at_spikes)

    ax.hist(
        jnp.concatenate(data),
        bins=31,
        density=True,
        histtype="stepfilled",
        label="Noise distribution at spike times",
        alpha=0.7,
        color="darkgreen",
    )

    # Plot the analytical noise distribution for comparison
    # note that this assumes constant noise std over time
    if noise_std is not None:
        x = jnp.linspace(-4 * noise_std, 4 * noise_std, 100)
        pdf = (1 / (noise_std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
            -0.5 * (x / noise_std) ** 2
        )
        ax.plot(
            x, pdf, color="k", linestyle="--", label="Analytical Noise Distribution"
        )
        ax.legend()

    ax.set_title("Distribution of Noise Values at Spike Times")
    ax.set_xlabel("Noise Value")
    ax.set_ylabel("Density")
