import jax.numpy as jnp
import matplotlib as mpl
from diffrax import Solution
from matplotlib import pyplot as plt

from adaptive_SNN.models.models import (
    AgentSystem,
    LIFNetwork,
    NoisyNetwork,
)

mpl.rcParams["savefig.directory"] = "../figures"


def _plot_membrane_potential(ax, t, state, model, neurons_to_plot=None):
    if isinstance(model, LIFNetwork):
        base_network = model
        network_state = state
    elif isinstance(model, NoisyNetwork):
        base_network = model.base_network
        network_state = state.network_state

    V = network_state.V
    S = network_state.S

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[0])

    # Plot membrane potentials
    for i in neurons_to_plot:
        spike_times = t[S[:, i] > 0]
        ax.vlines(
            spike_times,
            V[:, i][S[:, i] > 0] * 1e3,
            -40,
        )
        ax.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title("Neuron Membrane Potential")


def _plot_spikes(ax, t, state, model, neurons_to_plot=None):
    if isinstance(model, LIFNetwork):
        base_network = model
        network_state = state
    elif isinstance(model, NoisyNetwork):
        base_network = model.base_network
        network_state = state.network_state
    elif isinstance(model, AgentSystem):
        base_network = model.noisy_network.base_network
        network_state = state[0].network_state

    N_neurons = base_network.N_neurons
    N_inputs = base_network.N_inputs
    exc_mask = base_network.excitatory_mask
    spikes = network_state.S

    dt = t[1] - t[0]
    spike_times_per_neuron = [
        jnp.nonzero(spikes[:, i])[0] * dt for i in range(spikes.shape[1])
    ][::-1]
    ax.set_yticks(range(len(spike_times_per_neuron)))
    ax.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spike Raster Plot")

    if N_inputs > 0:
        # Shade background to distinguish input vs. main neurons
        # This assumes that all exc/inh inputs are grouped toghether at the end of the neuron list
        N_exc_input = jnp.sum(exc_mask[N_neurons:])
        N_inh_input = N_inputs - N_exc_input
        ax.axhspan(-0.5, N_inh_input - 0.5, facecolor="#E8BFB5", alpha=0.3)
        ax.axhspan(
            N_inh_input - 0.5,
            N_inh_input + N_exc_input - 0.5,
            facecolor="#B5D6E8",
            alpha=0.3,
        )


def _plot_conductances(ax, t, state, model, neurons_to_plot=None, split_noise=False):
    if isinstance(model, LIFNetwork):
        base_network = model
        network_state = state

        noise_E = jnp.zeros_like(N_neurons)
        noise_I = jnp.zeros_like(N_neurons)
    elif isinstance(model, NoisyNetwork):
        base_network = model.base_network
        network_state = state.network_state

        noise_E = state.noise_E_state
        noise_I = state.noise_I_state

    N_neurons = base_network.N_neurons
    N_inputs = base_network.N_inputs
    W = network_state.W
    G = network_state.G
    exc_mask = base_network.excitatory_mask

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(N_neurons)

    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    if split_noise:
        ax.plot(
            t,
            weighed_G_excitatory[:, neurons_to_plot],
            label="Synaptic E Conductance",
            color="g",
        )
        ax.plot(
            t,
            weighed_G_inhibitory[:, neurons_to_plot],
            label="Synaptic I Conductance",
            color="r",
        )
        ax.plot(
            t,
            noise_E[:, neurons_to_plot],
            label="Noise E Conductance",
            color="g",
            linestyle="--",
        )
        ax.plot(
            t,
            noise_I[:, neurons_to_plot],
            label="Noise I Conductance",
            color="r",
            linestyle="--",
        )
    else:
        ax.plot(
            t,
            weighed_G_excitatory[:, neurons_to_plot] + noise_E[:, neurons_to_plot],
            label="Total E Conductance",
            color="g",
        )
        ax.plot(
            t,
            weighed_G_inhibitory[:, neurons_to_plot] + noise_I[:, neurons_to_plot],
            label="Total I Conductance",
            color="r",
        )
    ax.legend(loc="upper right")
    ax.set_ylabel("Total Conductance (S)")
    ax.set_title("Total Conductances")


def plot_simulate_SNN_results(
    sol: Solution,
    model: LIFNetwork | NoisyNetwork,
    t0: float,
    t1: float,
    dt0: float,
):
    # Get results
    t = sol.ts
    state = sol.ys

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    _plot_membrane_potential(ax1, t, state, model, [0])
    _plot_conductances(ax2, t, state, model, [0])
    _plot_spikes(ax3, t, state, model)

    # Set x-axis limits and ticks for all subplots
    xticks = jnp.linspace(t0, t1, 6)  # 6 evenly spaced ticks
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(t0, t1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax.label_outer()

    plt.tight_layout()
    plt.show()


def plot_learning_results(
    sol: Solution,
    model: AgentSystem,
    t0: float,
    t1: float,
    dt0: float,
    args: dict = None,
):
    # Get results
    t = sol.ts
    state = sol.ys
    network_state, reward_state, env_state = state

    # Compute reward prediction error if possible
    if args is not None and "reward_fn" in args:
        rewards = jnp.array(
            [args["reward_fn"](ti, env_state[i], args) for i, ti in enumerate(t)]
        )
        RPE = rewards - jnp.squeeze(reward_state)
    else:
        RPE = None

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    axs[0].plot(t, env_state, label="Environment State", color="m")
    axs[0].set_title("Environment State Over Time")
    axs[0].set_ylabel("Environment State")

    axs[1].plot(t, reward_state, label="Reward State", color="b")
    axs[1].plot(t, rewards, label="Instant Rewards", color="k", linestyle="--")
    axs[1].set_title("Rewards Over Time")
    axs[1].set_ylabel("Reward")

    axs[2].plot(t, RPE, label="Reward Prediction Error", color="r")
    axs[2].set_title("Reward Prediction Error Over Time")
    axs[2].set_ylabel("RPE")

    # Plot spikes as raster plot
    _plot_spikes(axs[3], t, state, model)

    # Set x-axis limits and ticks for all subplots
    xticks = jnp.linspace(t0, t1, 6)  #
    for ax in axs:
        ax.set_xlim(t0, t1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax.label_outer()

    plt.tight_layout()
    plt.show()
