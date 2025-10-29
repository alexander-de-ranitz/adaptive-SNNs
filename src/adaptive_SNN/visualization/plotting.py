import jax
import jax.numpy as jnp
import matplotlib as mpl
from diffrax import Solution
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    Agent,
    AgentEnvSystem,
    AgentState,
    LIFNetwork,
    NoisyNetwork,
    NoisyNetworkState,
    SystemState,
)

mpl.rcParams["savefig.directory"] = "../figures"


def _get_LIF_state(state):
    if isinstance(state, LIFNetwork):
        return state
    elif isinstance(state, NoisyNetworkState):
        return state.network_state
    elif isinstance(state, AgentState):
        return state.noisy_network.network_state
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network.network_state
    else:
        raise ValueError("Unsupported state type")


def _get_LIF_model(model):
    if isinstance(model, LIFNetwork):
        return model
    elif isinstance(model, NoisyNetwork):
        return model.base_network
    elif isinstance(model, Agent):
        return model.noisy_network.base_network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.noisy_network.base_network
    else:
        raise ValueError("Unsupported model type")


def _get_noisy_network_state(state):
    if isinstance(state, NoisyNetworkState):
        return state
    elif isinstance(state, AgentState):
        return state.noisy_network
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network
    else:
        raise ValueError("Unsupported state type")


def _plot_membrane_potential(ax, t, state, model, neurons_to_plot=None):
    lif_state = _get_LIF_state(state)
    V = lif_state.V
    S = lif_state.S

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
    lif_network = _get_LIF_model(model)
    lif_state = _get_LIF_state(state)

    N_neurons = lif_network.N_neurons
    N_inputs = lif_network.N_inputs
    exc_mask = lif_network.excitatory_mask
    spikes = lif_state.S

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
    base_network = _get_LIF_model(model)
    network_state = _get_LIF_state(state)

    N_neurons = base_network.N_neurons
    W = network_state.W
    G = network_state.G
    exc_mask = base_network.excitatory_mask

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(N_neurons)

    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    if split_noise:
        noisy_network_state = _get_noisy_network_state(state)
        noise_E = noisy_network_state.noise_E_state
        noise_I = noisy_network_state.noise_I_state
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
            linestyle=":",
        )
        ax.plot(
            t,
            noise_I[:, neurons_to_plot],
            label="Noise I Conductance",
            color="r",
            linestyle=":",
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


def _plot_voltage_distribution(ax, t, state, model, neurons_to_plot=None):
    V = _get_LIF_state(state).V

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    for i in neurons_to_plot:
        ax.hist(
            V[:, i] * 1e3, bins=50, alpha=0.5, label=f"Neuron {i + 1}", density=True
        )
    ax.set_xlabel("Membrane Potential (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Membrane Potential Distribution")
    ax.set_xlim(jnp.min(V) * 1e3 - 5, jnp.max(V) * 1e3 + 5)


def plot_simulate_SNN_results(
    sol: Solution,
    model: LIFNetwork | NoisyNetwork,
    t0: float,
    t1: float,
    dt0: float,
    split_noise: bool = False,
    plot_spikes: bool = True,
    plot_voltage_distribution: bool = False,
    neurons_to_plot: jnp.ndarray | None = None,
    save_path: str | None = None,
):
    # Get results
    t = sol.ts
    state = sol.ys

    n_axs = 2 + int(plot_spikes) + int(plot_voltage_distribution)

    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 8), sharex=False)

    for ax in axs:
        ax.set_xlim(t0, t1)

    _plot_membrane_potential(axs[0], t, state, model, neurons_to_plot=neurons_to_plot)
    _plot_conductances(
        axs[1],
        t,
        state,
        model,
        neurons_to_plot=neurons_to_plot,
        split_noise=split_noise,
    )
    if plot_spikes:
        _plot_spikes(axs[2], t, state, model, neurons_to_plot=neurons_to_plot)
    if plot_voltage_distribution:
        _plot_voltage_distribution(
            axs[-1], t, state, model, neurons_to_plot=neurons_to_plot
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_learning_results(
    sols: Solution | list[Solution],
    model,
    t0: float,
    t1: float,
    dt0: float,
    args: dict = None,
    target_state: float = 10.0,
    save_path: str | None = None,
):
    if isinstance(sols, Solution):
        sols = [sols]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    colors = ["b", "g", "m", "c", "r", "y", "k"]

    for i, sol in enumerate(sols):
        # Get results
        t = sol.ts
        state: SystemState = sol.ys
        agent_state, env_state = state.agent_state, state.environment_state
        network_state, reward_state = agent_state.noisy_network, agent_state.reward

        # Compute reward prediction error if possible
        if args is not None and "reward_fn" in args:
            rewards = jnp.array(
                [
                    args["reward_fn"](
                        ti, jax.tree.map(lambda arr: arr[i], env_state), args
                    )
                    for i, ti in enumerate(t)
                ]
            )
            RPE = jnp.squeeze(rewards) - jnp.squeeze(reward_state)
        else:
            RPE = None

        for ax in axs:
            ax.set_xlim(t0, t1)

        axs[0].plot(t, env_state, label="Environment State", color=colors[i])
        if i == 0:
            axs[0].axhline(
                target_state, color="k", linestyle="--", label="Target State"
            )
        axs[0].set_title("Environment State ")
        axs[0].set_ylabel("Environment State")

        axs[1].plot(t, reward_state, label="Reward State", color=colors[i])
        # axs[1].plot(t, rewards, label="Instant Rewards", color="k", linestyle="--")
        # axs[1].legend(loc="upper right")
        axs[1].set_title("Rewards Over Time")
        axs[1].set_ylabel("Reward")

        axs[2].plot(t, RPE, label="Reward Prediction Error", color=colors[i])
        axs[2].set_title("Reward Prediction Error")
        axs[2].set_ylabel("RPE")

        # # Plot spikes as raster plot
        # _plot_spikes(axs[3], t, state, model)

        # Plot exc synaptic weights over time for first neuron
        axs[3].plot(t, network_state.network_state.W[:, 0, 1], color=colors[i])
        axs[3].set_title("Synaptic Weight")
        axs[3].set_ylabel("Weight")
        axs[3].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
