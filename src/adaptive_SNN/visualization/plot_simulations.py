import jax
import jax.numpy as jnp
import matplotlib as mpl
from diffrax import Solution
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    LIFNetwork,
    NoisyNetwork,
    SystemState,
)
from adaptive_SNN.visualization.utils import (
    _plot_conductances,
    _plot_membrane_potential,
    _plot_spikes_raster,
)

mpl.rcParams["savefig.directory"] = "../figures"


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
        _plot_spikes_raster(axs[2], t, state, model, neurons_to_plot=neurons_to_plot)
    if plot_voltage_distribution:
        plot_voltage_distribution(
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
