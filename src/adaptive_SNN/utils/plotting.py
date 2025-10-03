import jax.numpy as jnp
import matplotlib as mpl
from diffrax import Solution
from matplotlib import pyplot as plt

from adaptive_SNN.models.models import AgentSystem, NoisyNetwork, NoisyNetworkState

mpl.rcParams["savefig.directory"] = "../figures"


def plot_simulate_noisy_SNN_results(
    sol: Solution,
    model: NoisyNetwork,
    t0: float,
    t1: float,
    dt0: float,
):
    # Get results
    t = sol.ts
    result: NoisyNetworkState = sol.ys
    network_state, noise_E, noise_I = (
        result.network_state,
        result.noise_E_state,
        result.noise_I_state,
    )

    V, S, W, G = network_state.V, network_state.S, network_state.W, network_state.G

    weighed_G_inhibitory = (
        jnp.sum(
            W * G * jnp.invert(model.base_network.excitatory_mask[None, :]), axis=-1
        )
        + noise_I
    )
    weighed_G_excitatory = (
        jnp.sum(W * G * model.base_network.excitatory_mask[None, :], axis=-1) + noise_E
    )
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot membrane potentials
    for i in range(model.base_network.N_neurons):
        spike_times = t[S[:, i] > 0]
        ax1.vlines(
            spike_times,
            V[:, i][S[:, i] > 0] * 1e3,
            -40,
        )
        ax1.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V")
    ax1.set_ylabel("Membrane Potential (mV)")
    ax1.set_title("Neuron Membrane Potential")

    # Plot total conductance of neuron 0
    ax2.plot(t, weighed_G_excitatory[:, 0], label="Total E Conductance", color="g")
    ax2.plot(t, weighed_G_inhibitory[:, 0], label="Total I Conductance", color="r")
    ax2.plot(t, noise_E[:, 0], label="Noise E Conductance", color="g", linestyle="--")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Total Conductance (S)")
    ax2.set_title("Total Conductances")

    # Plot spikes as raster plot
    spike_times_per_neuron = [jnp.nonzero(S[:, i])[0] * dt0 for i in range(S.shape[1])][
        ::-1
    ]
    ax3.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    ax3.set_yticks(range(len(spike_times_per_neuron)))
    ax3.set_ylabel("Neuron")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Spike Raster Plot")

    if model.base_network.N_inputs > 0:
        # Shade background to distinguish input vs. main neurons
        N_input = model.base_network.N_inputs
        N_exc_input = jnp.sum(model.base_network.excitatory_mask[:N_input])
        ax3.axhspan(-0.5, N_input - N_exc_input - 0.5, facecolor="#E8BFB5", alpha=0.3)
        ax3.axhspan(
            N_input - N_exc_input - 0.5, N_input - 0.5, facecolor="#B5D6E8", alpha=0.3
        )

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
    network_state, reward_state, env_state = sol.ys

    # Compute reward prediction error if possible
    if args is not None and "compute_reward" in args:
        rewards = jnp.array(
            [args["compute_reward"](ti, env_state[i], args) for i, ti in enumerate(t)]
        )
        RPE = rewards - jnp.squeeze(reward_state)
        print("RPE shape:", RPE.shape)
        print("Rewards shape:", rewards.shape)
        print("Reward state shape:", reward_state.shape)
    else:
        RPE = None

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    axs[0].plot(t, reward_state, label="Reward State", color="b")
    axs[0].set_title("Reward State Over Time")
    axs[0].set_ylabel("Reward")

    axs[1].plot(t, env_state, label="Environment State", color="m")
    axs[1].set_title("Environment State Over Time")
    axs[1].set_ylabel("Environment State")

    if RPE is not None:
        axs[2].plot(t, RPE, label="Reward Prediction Error", color="r")
        axs[2].set_title("Reward Prediction Error Over Time")
        axs[2].set_ylabel("RPE")

    # Plot spikes as raster plot
    spike_times_per_neuron = [
        jnp.nonzero(network_state[0].S[:, i])[0] * dt0
        for i in range(network_state[0].S.shape[1])
    ][::-1]
    axs[3].eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    axs[3].set_yticks(range(len(spike_times_per_neuron)))
    axs[3].set_ylabel("Neuron")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_title("Spike Raster Plot")

    N_input = model.noisy_network.base_network.N_inputs
    if N_input > 0:
        # Shade background to distinguish input vs. main neurons and excitatory vs. inhibitory
        N_exc_input = jnp.sum(model.base_network.excitatory_mask[:N_input])
        axs[3].axhspan(
            -0.5, N_input - N_exc_input - 0.5, facecolor="#E8BFB5", alpha=0.3
        )
        axs[3].axhspan(
            N_input - N_exc_input - 0.5, N_input - 0.5, facecolor="#B5D6E8", alpha=0.3
        )

    # Set x-axis limits and ticks for all subplots
    xticks = jnp.linspace(t0, t1, 6)  #
    for ax in axs:
        ax.set_xlim(t0, t1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax.label_outer()

    plt.tight_layout()
    plt.show()
