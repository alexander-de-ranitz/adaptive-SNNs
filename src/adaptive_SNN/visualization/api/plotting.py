import jax
import matplotlib as mpl
from diffrax import Solution
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    SystemState,
)
from adaptive_SNN.utils.metrics import compute_CV_ISI
from adaptive_SNN.visualization.utils.adapters import get_LIF_model, get_LIF_state
from adaptive_SNN.visualization.utils.components import (
    _plot_conductance_frequency_spectrum,
    _plot_conductances,
    _plot_ISI_distribution,
    _plot_membrane_potential,
    _plot_spike_rate_distributions,
    _plot_spike_rates,
    _plot_spikes_raster,
    _plot_voltage_distribution,
)

mpl.rcParams["savefig.directory"] = "../figures"


def plot_simulate_SNN_results(
    sol: Solution,
    model: LIFNetwork | NoisyNetwork,
    split_noise: bool = False,
    plot_spikes: bool = True,
    plot_voltage_distribution: bool = False,
    neurons_to_plot: jnp.ndarray | None = None,
    save_path: str | None = None,
    **plot_kwargs,
):
    # Get results
    t = sol.ts
    t0 = t[0]
    t1 = t[-1]

    n_axs = 2 + int(plot_spikes) + int(plot_voltage_distribution)

    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 8), sharex=False)

    for ax in axs:
        ax.set_xlim(t0, t1)

    _plot_membrane_potential(
        axs[0], sol, model, neurons_to_plot=neurons_to_plot, **plot_kwargs
    )
    _plot_conductances(
        axs[1],
        sol,
        model,
        neurons_to_plot=neurons_to_plot,
        split_noise=split_noise,
    )
    if plot_spikes:
        _plot_spikes_raster(
            axs[2], sol, model, neurons_to_plot=neurons_to_plot, **plot_kwargs
        )
    if plot_voltage_distribution:
        _plot_voltage_distribution(
            axs[-1], sol, model, neurons_to_plot=neurons_to_plot, **plot_kwargs
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_learning_results(
    sols: Solution | list[Solution],
    model: AgentEnvSystem,
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
        t0 = t[0]
        t1 = t[-1]

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


def plot_network_stats(
    sol: Solution,
    model,
    save_path: str | None = None,
):
    """Plot various network statistics including ISI distribution."""
    lif_model = get_LIF_model(model)

    fig, axs = plt.subplots(3, 2, figsize=(6, 6))
    _plot_ISI_distribution(
        axs[0][0],
        sol,
        model,
        neurons_to_plot=jnp.arange(lif_model.N_neurons),
        color="blue",
        label="Recurrent Neurons",
        alpha=0.7,
    )

    CV_ISI = compute_CV_ISI(get_LIF_state(sol.ys).S)
    CV_ISI = CV_ISI[~jnp.isnan(CV_ISI)]
    axs[0][1].hist(CV_ISI, bins=20, color="k")
    axs[0][1].set_title("CV of ISI Distribution")
    axs[0][1].set_xlabel("CV of ISI")
    axs[0][1].set_ylabel("Count")

    _plot_spike_rate_distributions(
        axs[1][0],
        sol,
        model,
    )

    _plot_spikes_raster(axs[2][0], sol, model)

    _plot_spike_rates(
        axs[2][1],
        sol,
        model,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_frequency_analysis(
    sol,
    model,
    neurons_to_plot: jnp.ndarray | None = None,
):
    state = sol.ys
    V = get_LIF_state(state).V

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    # Plot membrane potential
    _plot_membrane_potential(axs[0], sol, model, neurons_to_plot=neurons_to_plot)
    axs[0].set_title("Neuron Membrane Potential")

    # Plot voltage distribution
    _plot_voltage_distribution(axs[1], sol, model, neurons_to_plot=neurons_to_plot)
    axs[1].set_title("Voltage Distribution")

    # Plot freq spectrum
    _plot_conductance_frequency_spectrum(
        axs[2], sol, model, neurons_to_plot=neurons_to_plot, plot_noise=True
    )
    axs[2].set_title("Conductance Frequency Spectrum")

    plt.tight_layout()
    plt.show()
