import equinox as eqx
import jax
import matplotlib as mpl
from diffrax import Solution
from jax import numpy as jnp
from jaxtyping import Array
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    SystemState,
)
from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.utils.metrics import compute_CV_ISI
from adaptive_SNN.visualization.utils.adapters import (
    get_LIF_model,
    get_LIF_state,
    get_noisy_network_state,
)
from adaptive_SNN.visualization.utils.components import (
    _plot_conductance_frequency_spectrum,
    _plot_conductances,
    _plot_ISI_distribution,
    _plot_membrane_potential,
    _plot_noise_distribution_STA,
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
    split_recurrent: bool = False,
    plot_spikes: bool = True,
    plot_voltage_distribution: bool = False,
    neurons_to_plot: jnp.ndarray | None = None,
    save_path: str | None = None,
    **plot_kwargs,
):
    """Plot the results of a noisy SNN simulation.

    Solution state must contain the LIFState's V, S, W, and G. If NoisyNetwork, also noise_state.
    """
    # Get results
    t = sol.ts
    t0 = t[0]
    t1 = t[-1]

    n_axs = 2 + int(plot_spikes) + int(plot_voltage_distribution)

    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 8), sharex=True)

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
        split_recurrent=split_recurrent,
        **plot_kwargs,
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


def plot_rate_learning_results(
    sols: Solution | list[Solution],
    model: AgentEnvSystem | None = None,
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
        network_state, predicted_reward, reward_signal = (
            agent_state.noisy_network,
            agent_state.predicted_reward.squeeze(),
            state.reward_signal,
        )
        RPE = reward_signal - predicted_reward

        for ax in axs:
            ax.set_xlim(t0, t1)

        axs[0].plot(t, env_state, label="Environment State", color=colors[i])
        if i == 0:
            axs[0].axhline(
                target_state, color="k", linestyle="--", label="Target State"
            )
        axs[0].set_title("Environment State ")
        axs[0].set_ylabel("Environment State")

        axs[1].plot(t, predicted_reward, label="Predicted Reward", color=colors[i])
        axs[1].plot(t, reward_signal, label="Reward Signal", color="k", linestyle="--")
        axs[1].legend(loc="upper right")
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


def plot_SDI_results(
    sol: Solution,
    model: AgentEnvSystem,
    args: dict,
    save_path: str | None = None,
):
    if isinstance(sol, Solution):
        sol = [sol]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    colors = ["k", "b", "g", "m", "c", "r", "y", "k"]

    for i, sol in enumerate(sol):
        # Get results
        t = sol.ts
        t0 = t[0]
        t1 = t[-1]

        state: SystemState = sol.ys
        agent_state, env_state = state.agent_state, state.environment_state
        predicted_reward = agent_state.predicted_reward
        reward_signal = state.reward_signal
        RPE = reward_signal.squeeze() - predicted_reward.squeeze()

        for ax in axs:
            ax.set_xlim(t0, t1)

        axs[0].plot(t, env_state, label=["Position", "Velocity"])
        axs[0].legend()
        axs[0].set_title("Environment State ")
        axs[0].set_ylabel("Environment State")

        axs[1].plot(t, predicted_reward, label="Predicted Reward", color=colors[i])
        axs[1].plot(t, reward_signal, label="Reward Signal", color="k", linestyle="--")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Rewards Over Time")
        axs[1].set_ylabel("Reward")

        axs[2].plot(t, RPE, label="Reward Prediction Error", color=colors[i])
        axs[2].set_title("Reward Prediction Error")
        axs[2].set_ylabel("RPE")

        if args is not None and "network_output_fn" in args:
            outputs = jnp.array(
                [
                    args["network_output_fn"](
                        ti, jax.tree.map(lambda arr: arr[i], agent_state), args
                    )
                    for i, ti in enumerate(t)
                ]
            )
            axs[3].plot(t, outputs, label="Network Output", color=colors[i])
            axs[3].set_title("Network Output Over Time")
            axs[3].set_ylabel("Network Output")
            axs[3].set_xlabel("Time (s)")

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

    CV_ISI = compute_CV_ISI(get_LIF_state(sol.ys).S, sol.ts)
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


def plot_noise_STA(
    sols,
    model,
    neurons_to_plot: jnp.ndarray | None = None,
    noise_levels: float | list | None = None,
    save_path: str | None = None,
):
    """Plot the spike-triggered average (STA) of the noise process

    Providing the noise_level allows overlaying the analytical noise distribution for comparison.
    """
    if not isinstance(sols, list):
        sols = [sols]
        noise_levels = [noise_levels]

    fig, axs = plt.subplots(len(sols), 1, figsize=(3.05, 1 * len(sols)))

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(get_LIF_model(model).N_neurons)

    for i, sol in enumerate(sols):
        ax = axs[i] if len(sols) > 1 else axs

        # We need to compute the (mean) desired noise std, we take the mean over the second half of the simulation to reduce transient effects
        args = {
            "noise_scale_hyperparam": noise_levels[i]
            if noise_levels is not None
            else None
        }
        var_E_conductance = get_LIF_state(sol.ys).var_E_conductance
        mean_var_E = jnp.mean(
            var_E_conductance[jnp.size(sol.ts) // 2 :, neurons_to_plot], axis=0
        )
        last_state = jax.tree.map(lambda v: v[-1], get_noisy_network_state(sol.ys))
        last_state = eqx.tree_at(
            lambda s: s.network_state.var_E_conductance, last_state, mean_var_E
        )
        if not isinstance(model, NoisyNetwork):
            raise NotImplementedError(
                "Model must be NoisyNetwork to compute noise std."
            )
        noise_std = model.compute_desired_noise_std(0.0, last_state, args)

        _plot_noise_distribution_STA(
            ax, sol, model, neurons_to_plot=neurons_to_plot, noise_std=noise_std
        )

        lif_state = get_LIF_state(sol.ys)
        noise_state = get_noisy_network_state(sol.ys).noise_state

        CV_ISI = compute_CV_ISI(lif_state.S, sol.ts)[
            0
        ]  # Compute CV ISI for first neuron
        corr = jnp.corrcoef(noise_state.flatten(), lif_state.V[:, 0].flatten())[0, 1]
        ax.set_title(
            rf"Noise Level = {noise_levels[i]} $\sigma_{{\mathrm{{syn}}}}$ | CV ISI =  {CV_ISI:.2f} | Corr(V, Noise) = {corr:.2f}"
        )
        ax.set_xlabel("Noise Value")
        ax.set_ylabel("Density")
        ax.label_outer()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_learning_detailed(
    sol: Solution,
    model: AgentEnvSystem | None = None,
    args: dict = {},
    neurons_to_plot: Array | None = None,
    save_path: str | None = None,
    target_state: float | None = None,
    **plot_kwargs,
):
    # Get results
    t = sol.ts
    t0 = t[0]
    t1 = t[-1]

    fig, axs = plt.subplots(
        7
        if not isinstance(model.agent.noisy_network.base_network, GatedLIFNetwork)
        else 8,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    for ax in axs:
        ax.set_xlim(t0, t1)

    state: SystemState = sol.ys
    agent_state, env_state = state.agent_state, state.environment_state
    network_state, predicted_reward = (
        agent_state.noisy_network,
        agent_state.predicted_reward,
    )
    reward_signal = state.reward_signal
    eligibility_trace = agent_state.noisy_network.network_state.features.eligibility
    noise_state = agent_state.noisy_network.noise_state
    var_E_conductance = agent_state.noisy_network.network_state.var_E_conductance
    RPE = reward_signal.squeeze() - predicted_reward.squeeze()

    axs[0].plot(t, env_state, label="Environment State")
    if target_state is not None:
        axs[0].axhline(target_state, color="k", linestyle="--", label="Target State")
    axs[0].set_title("Environment State ")
    axs[0].set_ylabel("Environment State")

    axs[1].plot(t, RPE, label="Reward Prediction Error")
    axs[1].plot(
        t,
        reward_signal.squeeze(),
        label="Reward Signal",
        color="r",
        linestyle="--",
        linewidth=0.5,
    )
    axs[1].plot(
        t,
        predicted_reward.squeeze(),
        label="Predicted Reward",
        color="b",
        linestyle="--",
        linewidth=0.5,
    )
    axs[1].set_title("Reward Prediction Error")
    axs[1].set_ylabel("RPE")

    _plot_membrane_potential(
        axs[2], sol, model, neurons_to_plot=neurons_to_plot, **plot_kwargs
    )
    if model is not None:
        _plot_conductances(
            axs[3],
            sol,
            model,
            neurons_to_plot=neurons_to_plot,
            split_noise=True,
        )

    axs[4].plot(
        t,
        eligibility_trace[:, 0, model.agent.noisy_network.base_network.excitatory_mask],
        label="Eligibility Trace",
        alpha=0.5,
    )
    axs[4].set_title("Eligibility Trace")
    axs[4].set_ylabel("Eligibility Trace")
    axs[4].set_xlabel("Time (s)")

    E_weights = network_state.network_state.W[
        :, 0, model.agent.noisy_network.base_network.excitatory_mask
    ]
    I_weights = network_state.network_state.W[
        :, 0, ~model.agent.noisy_network.base_network.excitatory_mask
    ]
    # Plot exc synaptic weights over time for first neuron
    axs[5].plot(t, E_weights, label="Excitatory Weights", c="g", alpha=0.1)
    axs[5].plot(t, I_weights, label="Inhibitory Weights", c="r", alpha=0.1)
    axs[5].set_title("Synaptic Weight")
    axs[5].set_ylabel("Weight")
    axs[5].set_xlabel("Time (s)")

    axs[6].plot(t, noise_state[:, 0], label="Noise State")
    axs[6].set_title("Noise State")
    axs[6].set_ylabel("Noise State")
    axs[6].set_xlabel("Time (s)")

    axs[6].plot(t, jnp.sqrt(var_E_conductance[:, 0]), label="Var E Conductance")

    if isinstance(model.agent.noisy_network.base_network, GatedLIFNetwork):
        print("Plotting gating function value over time for GatedLIFNetwork...")
        gating_values = model.agent.noisy_network.base_network.gating_function(
            get_LIF_state(sol.ys).V[:, 0]
        )
        axs[7].plot(t, gating_values, label="Gating Function Value")
        axs[7].set_title("Gating Function Value")
        axs[7].set_ylabel("Gating Value")
        axs[7].set_xlabel("Time (s)")

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_weight_distribution_over_time(
    sols: Solution | list[Solution],
    model: AgentEnvSystem,
    t_range: tuple[float, float] | None = None,
    plot_reward: bool = True,
    neurons_to_plot: Array | None = None,
    n_bins: int = 5,
    save_path: str | None = None,
    **plot_kwargs,
):
    """Plots the distribution of synaptic weights as a ridgeline plot over time.

    Arguments:
        sols: The solution object(s) containing the simulation results.
        model: The AgentEnvSystem model used in the simulation.
        t_range: A tuple specifying the time range to use for the plot (start, end). If None, plots the entire range.
        plot_reward: Whether to plot the probability density of the reward as well.
        neurons_to_plot: An array of neuron indices to include in the weight distribution. If None, includes all neurons.
        n_bins: Number of time bins to divide the simulation into for the ridgeline plot.
        save_path: Path to save the figure. If None, displays the figure instead.
    """

    if isinstance(sols, Solution):
        sols = [sols]

    fig, axs = plt.subplots(1, 1 if not plot_reward else 2, figsize=(3, 2))
    if not plot_reward:
        axs = [axs]

    # Collect weight and reward data from all solutions
    all_weight_bins = [[] for _ in range(n_bins)]
    all_reward_bins = [[] for _ in range(n_bins)]

    for sol in sols:
        # Get results
        t = sol.ts
        state: SystemState = sol.ys

        if t_range is not None:
            t_start, t_end = t_range
        else:
            t_start, t_end = t[0], t[-1]

        # Create time bins
        bin_edges = jnp.linspace(t_start, t_end, n_bins + 1)

        for i in range(n_bins):
            bin_start_idx = jnp.searchsorted(t, bin_edges[i])
            bin_end_idx = jnp.searchsorted(t, bin_edges[i + 1])

            # Extract weights for this time bin
            weight_data = state.agent_state.noisy_network.network_state.W[
                bin_start_idx:bin_end_idx, :, :
            ]
            if neurons_to_plot is not None:
                weight_data = weight_data[:, neurons_to_plot]

            # Remove -inf values
            weight_data = weight_data[jnp.isfinite(weight_data)]

            all_weight_bins[i].append(weight_data.flatten())

            if plot_reward:
                # Extract rewards for this time bin
                reward_data = state.agent_state.predicted_reward[
                    bin_start_idx:bin_end_idx
                ]
                all_reward_bins[i].append(reward_data.flatten())

    # Concatenate data from all solutions
    for i in range(n_bins):
        all_weight_bins[i] = jnp.concatenate(all_weight_bins[i])
        if plot_reward:
            all_reward_bins[i] = jnp.concatenate(all_reward_bins[i])

    # Plot weight distribution ridgeline
    _plot_ridgeline(
        axs[0], all_weight_bins, bin_edges, "Synaptic Weight", **plot_kwargs
    )

    # Plot reward distribution ridgeline
    if plot_reward:
        _plot_ridgeline(
            axs[1], all_reward_bins, bin_edges, "Predicted Reward", **plot_kwargs
        )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def _plot_ridgeline(
    ax, data_bins, bin_edges, xlabel, n_points=200, overlap=0.7, **plot_kwargs
):
    """Helper function to create a ridgeline plot.

    Arguments:
        ax: Matplotlib axis to plot on.
        data_bins: List of arrays, each containing data for one time bin.
        bin_edges: Array of time bin edges.
        xlabel: Label for the x-axis.
        n_points: Number of points to use for KDE evaluation.
        overlap: Amount of vertical overlap between ridges (0 = no overlap, 1 = full height overlap).
    """
    from scipy.stats import gaussian_kde

    n_bins = len(data_bins)

    # Determine global x range for all bins
    all_data = jnp.concatenate(data_bins)
    x_min, x_max = jnp.min(all_data), jnp.max(all_data)
    x_range = x_max - x_min
    x = jnp.linspace(x_min, x_max, n_points)

    # Compute KDE for each time bin and find max density for scaling
    kdes = []
    max_density = 0
    for i, data in enumerate(data_bins):
        if len(data) > 1:
            kde = gaussian_kde(data)
            density = kde(x)
            kdes.append(density)
            max_density = max(max_density, jnp.max(density))
        else:
            kdes.append(jnp.zeros_like(x))

    # Normalize densities and plot ridgelines
    colors = plt.cm.viridis(jnp.linspace(0, 1, n_bins))

    # Scale factor to prevent overlaps - distributions should fit within allocated space
    ridge_height = (1 - overlap) * 0.90

    for i in range(n_bins):
        # Normalize density to fit within allocated ridge height
        y = kdes[i] / max_density * ridge_height if max_density > 0 else kdes[i]
        y = jnp.clip(
            y, 0, ridge_height
        )  # Ensure densities don't exceed allocated height
        # Vertical offset with overlap (reversed so oldest is at top)
        y_offset = (n_bins - 1 - i) * (1 - overlap)
        y_shifted = y + y_offset

        # set the values of y_fill to 0.0 at the edges of the fill_mask to create a sharp cutoff
        ax.fill_between(
            x, y_offset, y_shifted, color=colors[i], alpha=0.7, linewidth=0.0
        )
        ax.plot(x, y_shifted, color="k", linewidth=0.3)
        # ax.hlines(y_offset, x_min, x_max, color='k', linewidth=1)  # Baseline for each ridge

        # Add time label showing the time range
        t_start = bin_edges[i]
        t_end = bin_edges[i + 1]

        # Position label
        label_y = y_offset + 0.25 * (1 - overlap)
        ax.text(
            x_max - 0.02 * x_range,
            label_y,
            f"t={t_start:.0f}-{t_end:.0f}s",
            verticalalignment="center",
            fontsize=8,
            horizontalalignment="right",
            color="k",
        )

    # Add text next to y-axis to indicate time progression. It should say "<- Time" with an arrow pointing downwards, indicating that time progresses from top to bottom.
    ax.text(
        x_min - 0.05 * x_range,
        (n_bins - 1) * (1 - overlap) / 2,
        r"$\leftarrow$ Time",
        rotation=90,
        verticalalignment="center",
        fontsize=10,
        color="k",
    )

    ax.set_xlabel(xlabel)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.001, n_bins * (1 - overlap) - 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Remove x-tick markers, keep only the numbers
    ax.tick_params(axis="x", which="both", length=0)


def plot_optuna_results(
    results_dir: str,
    study_name: str = "optuna_rate_learning",
    metric_name: str = "Mean Reward",
    save_path: str | None = None,
):
    """Plot the results of an Optuna hyperparameter optimization study using Matplotlib.

    This creates a heatmap of the hyperparameter values vs. the objective metric, which can be useful for visualizing the relationship between hyperparameters and performance.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.interpolate import griddata

    # Load the trials CSV file
    results_path = Path(results_dir)
    csv_file = results_path / "trials.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"No trials.csv found in {results_dir}")

    trials = pd.read_csv(csv_file)

    # Create a heatmap with hyperparameters on the axes and the metric as the color
    hyperparams = [col for col in trials.columns if "params_" in col]
    if len(hyperparams) != 2:
        raise ValueError(
            f"Expected exactly 2 hyperparameters for heatmap plotting. Got {len(hyperparams)}: {hyperparams}"
        )

    x_param, y_param = hyperparams

    # Extract parameter values and objectives
    x_values = trials[x_param].values
    y_values = trials[y_param].values
    z_values = trials["value"].values

    # Filter outliers using IQR method to maintain contrast
    q1, q3 = np.percentile(z_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr  # Use 3*IQR for outlier detection
    upper_bound = q3 + 3 * iqr

    # Create mask for non-outliers
    mask = (z_values >= lower_bound) & (z_values <= upper_bound)

    # Use only non-outliers for interpolation and color scaling
    x_values_filtered = x_values[mask]
    y_values_filtered = y_values[mask]
    z_values_filtered = z_values[mask]

    # Create a regular grid for interpolation
    n_points = 100
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    xi = np.linspace(x_min, x_max, n_points)
    yi = np.linspace(y_min, y_max, n_points)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate values onto the regular grid using filtered data
    zi_grid = griddata(
        (x_values_filtered, y_values_filtered),
        z_values_filtered,
        (xi_grid, yi_grid),
        method="cubic",
        fill_value=np.nan,
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot interpolated heatmap with color limits based on filtered data
    vmin, vmax = z_values_filtered.min(), z_values_filtered.max()
    cax = ax.pcolormesh(
        xi, yi, zi_grid, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax
    )

    # Overlay scatter points showing actual trial locations (filtered)
    ax.scatter(
        x_values_filtered,
        y_values_filtered,
        c=z_values_filtered,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.5,
        s=30,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    # Mark outliers with a different symbol (if any)
    if mask.sum() < len(mask):
        ax.scatter(
            x_values[~mask],
            y_values[~mask],
            marker="x",
            c="red",
            s=50,
            linewidths=2,
            label="Outliers (excluded)",
            zorder=10,
        )

    ax.set_xlabel(x_param.replace("params_", ""))
    ax.set_ylabel(y_param.replace("params_", ""))
    ax.set_title(f"{metric_name} Heatmap")
    fig.colorbar(cax, label=metric_name, ax=ax)

    # Add legend if outliers were filtered
    if mask.sum() < len(mask):
        ax.legend()
        print(
            f"Filtered {len(mask) - mask.sum()} outlier(s) with values: {z_values[~mask]}"
        )

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_weights_over_time(
    sol: Solution,
    model: AgentEnvSystem,
    neurons_to_plot: Array | None = None,
    save_path: str | None = None,
):
    """Plot the synaptic weights over time for the specified neurons."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    t = sol.ts
    state: SystemState = sol.ys

    weight_data = state.agent_state.noisy_network.network_state.W

    # Extract weights over time
    if neurons_to_plot is not None:
        weight_data = weight_data[:, neurons_to_plot, :]

    I_mask = weight_data[1, :, :] > 5
    E_mask = weight_data[1, :, :] <= 5

    axs[0].plot(
        t,
        weight_data[:, I_mask].squeeze(),
        color="r",
        alpha=0.1,
        label="Inhibitory Weights",
    )
    axs[1].plot(
        t,
        weight_data[:, E_mask].squeeze(),
        color="g",
        alpha=0.1,
        label="Excitatory Weights",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Synaptic Weight")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Synaptic Weight")

    # ax.plot(t, environment_state, label='Environment State', color='r', linestyle='--')

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
