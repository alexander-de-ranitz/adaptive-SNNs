import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from adaptive_SNN.models import SystemState


def smooth_signal(signal, sigma=5):
    """Smooth a 1D signal using Gaussian filter.

    Args:
        signal: 1D array to smooth
        sigma: Standard deviation for Gaussian kernel (larger = smoother)

    Returns:
        Smoothed signal
    """
    return gaussian_filter1d(signal, sigma=sigma)


def parse_filename(filename):
    """Extract hyperparameters from filename.

    Expected format: rate_learning_iw{initial_weight}_nl{noise_level}_lr{learning_rate}.npz
    """
    pattern = r".*_iw([\d.]+)_nl([\d.]+)_lr([\d.]+)\.npz"
    match = re.match(pattern, filename)
    if match:
        iw = float(match.group(1))
        nl = float(match.group(2))
        lr = float(match.group(3))
        return iw, nl, lr
    else:
        raise ValueError(f"Could not parse filename: {filename}")


def load_results(results_dir):
    """Load all simulation results from directory.

    Returns:
        Dictionary with structure: {(nl, lr, iw): (times, reward_data, weight_data, env_state)}
    """
    results_path = Path(results_dir)
    data = {}

    for file in sorted(results_path.glob("*.npz")):
        try:
            # Parse hyperparameters from filename
            iw, nl, lr = parse_filename(file.name)

            # Load the data
            npz_data = np.load(file, allow_pickle=True)
            sol = npz_data["sol"].item()
            times = sol.ts
            state: SystemState = sol.ys

            # Extract reward data
            reward_data = state.reward_signal

            # Extract weight data (the learned weight is at index 2)
            weight_data = state.agent_state.noisy_network.network_state.W

            # Extract environment state
            env_state = state.environment_state

            # Store results
            data[(nl, lr, iw)] = (times, reward_data, weight_data, env_state)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue

    return data


def calculate_performance(times, reward_data, last_n_seconds=100):
    """Calculate performance as mean reward over the last N seconds.

    Args:
        times: Array of time points
        last_n_seconds: Number of seconds to average over (default: 100)

    Returns:
        Mean reward over the last N seconds
    """
    # Find the cutoff time
    cutoff_time = times[-1] - last_n_seconds

    # Find index corresponding to cutoff time
    cutoff_idx = np.searchsorted(times, cutoff_time)

    # Calculate mean reward over last N seconds
    if cutoff_idx < len(times) - 1:
        performance = np.mean(reward_data[cutoff_idx:])
    else:
        # If simulation is too short, use all data
        performance = np.mean(reward_data)

    return performance


def plot_heatmap(data, output_file=None):
    """Plot heatmap of performance averaged over initial weights.

    Args:
        data: Dictionary with structure: {(nl, lr, iw): (times, reward_data, weight_data)}
        output_file: Optional path to save the figure
    """
    if len(data) == 0:
        print("No data to plot!")
        return

    # Organize data by (nl, lr), averaging over iw
    performance_data = defaultdict(list)

    for (nl, lr, iw), (times, reward_data, weight_data, env_state) in data.items():
        perf = calculate_performance(times, reward_data, last_n_seconds=1000)
        performance_data[(nl, lr)].append(perf)

    # Get unique noise levels and learning rates
    all_nl = sorted(set(nl for nl, lr, iw in data.keys()))
    all_lr = sorted(set(lr for nl, lr, iw in data.keys()))

    # Create matrix for heatmap (averaged over initial weights)
    performance_matrix = np.zeros((len(all_lr), len(all_nl)))

    for i, lr in enumerate(all_lr):
        for j, nl in enumerate(all_nl):
            if (nl, lr) in performance_data:
                # Average over different initial weights
                performance_matrix[i, j] = np.mean(performance_data[(nl, lr)])
            else:
                performance_matrix[i, j] = np.nan

    outlier_threshold_upper = 0
    outlier_threshold_lower = -20

    # Create a mask for outliers
    outlier_mask = (performance_matrix > outlier_threshold_upper) | (
        performance_matrix < outlier_threshold_lower
    )

    # Create a copy for plotting (set outliers to NaN for color mapping)
    plot_matrix = performance_matrix.copy()
    plot_matrix[outlier_mask] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use pcolormesh for better control over axis labels
    im = ax.pcolormesh(
        np.arange(len(all_nl) + 1),
        np.arange(len(all_lr) + 1),
        plot_matrix,
        cmap="viridis",
        shading="flat",
    )

    # Overlay outliers with a distinct color (gray)
    outlier_plot_matrix = np.full_like(performance_matrix, np.nan)
    outlier_plot_matrix[outlier_mask] = 1.0  # Just a dummy value for coloring

    ax.pcolormesh(
        np.arange(len(all_nl) + 1),
        np.arange(len(all_lr) + 1),
        outlier_plot_matrix,
        cmap="Greys",
        shading="flat",
        vmin=0,
        vmax=2,
        alpha=0.7,
    )

    # Set ticks at the center of each cell
    ax.set_xticks(np.arange(len(all_nl)) + 0.5)
    ax.set_yticks(np.arange(len(all_lr)) + 0.5)
    ax.set_xticklabels([f"{nl}" for nl in all_nl])
    ax.set_yticklabels([f"{lr}" for lr in all_lr])

    ax.set_xlabel("Noise Level", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.set_title(
        "Performance (Mean Reward over Last 100s)\nAveraged over Initial Weights (Outliers shown in gray)",
        fontsize=15,
        fontweight="bold",
    )

    # Add colorbar with original values
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Reward", fontsize=12)

    # Add text annotations showing the actual performance values
    for i, lr in enumerate(all_lr):
        for j, nl in enumerate(all_nl):
            if not np.isnan(performance_matrix[i, j]):
                value = performance_matrix[i, j]

                # Check if it's an outlier
                if outlier_mask[i, j]:
                    # Show outlier with different formatting
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=8,
                    )
                else:
                    # Normal value
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {output_file}")
        plt.close(fig)
    else:
        plt.show()


def plot_learning_curves(data, output_file=None):
    """Plot grid of learning curves.

    Creates a 6x6 grid of subplots, one for each (noise_level, learning_rate) combination.
    Each subplot shows two learning curves (one for each initial weight) and environment state.

    Args:
        data: Dictionary with structure: {(nl, lr, iw): (times, reward_data, weight_data, env_state)}
        output_file: Optional path to save the figure
    """
    if len(data) == 0:
        print("No data to plot!")
        return

    # Get unique noise levels and learning rates
    all_nl = sorted(set(nl for nl, lr, iw in data.keys()))
    all_lr = sorted(set(lr for nl, lr, iw in data.keys()))
    all_iw = sorted(set(iw for nl, lr, iw in data.keys()))

    # Create figure with subplots
    fig, axes = plt.subplots(
        len(all_lr), len(all_nl), figsize=(20, 20), sharex=True, sharey=False
    )

    # Store secondary axes for sharing y-axis per row
    secondary_axes = {}

    # Plot learning curves
    for i, lr in enumerate(all_lr):
        for j, nl in enumerate(all_nl):
            # Reverse row indexing so smallest lr is at bottom (matches heatmap)
            row_idx = len(all_lr) - 1 - i
            if len(all_lr) > 1 and len(all_nl) > 1:
                ax = axes[row_idx, j]
            elif len(all_lr) > 1 and len(all_nl) == 1:
                ax = axes[row_idx]
            elif len(all_lr) == 1 and len(all_nl) > 1:
                ax = axes[j]
            else:
                ax = axes
            # Create secondary y-axis for environment state
            ax2 = ax.twinx()

            # Store secondary axis for this row
            if row_idx not in secondary_axes:
                secondary_axes[row_idx] = []
            secondary_axes[row_idx].append(ax2)

            # Plot weight curves for each initial weight on primary axis
            for iw_idx, iw in enumerate(all_iw):
                if (nl, lr, iw) in data:
                    times, reward_data, weight_data, env_state = data[(nl, lr, iw)]

                    # Select excitatory weights (excluding -inf inhibitory weights)
                    # Use np.isfinite to exclude -inf values
                    E_mask = (weight_data[1, :, :] < 5) & np.isfinite(
                        weight_data[1, :, :]
                    )

                    E_weights = weight_data[:, E_mask]
                    ax.plot(
                        times[1:], E_weights[1:], color="g", alpha=0.1, linewidth=0.5
                    )

                    # Plot mean weight over time for excitatory weights (mean over weights dimension)
                    mean_weights = np.mean(E_weights[1:], axis=1)
                    ax.plot(
                        times[1:],
                        mean_weights,
                        color="k",
                        alpha=1,
                        linewidth=0.5,
                        zorder=10,
                    )

                    # Plot environment state on secondary axis (only once per subplot)
                    # Flatten env_state if needed
                    if len(reward_data.shape) > 1:
                        reward_data = reward_data.squeeze()
                    # Smooth the environment state for better visualization
                    reward_smooth = smooth_signal(reward_data, sigma=100)
                    ax2.plot(
                        times[1:],
                        reward_smooth[1:],
                        color="b",
                        alpha=1,
                        label="reward",
                        linewidth=1,
                        linestyle="--",
                    )
                    ax2.plot(times[1:], reward_data[1:], color="b", alpha=0.3)

            # Set title for each subplot (noise level and learning rate)
            title = f"nl={nl}, lr={lr}"
            ax.set_title(title, fontsize=10)

            # Add grid
            ax.grid(True, alpha=0.3)

            # Set labels for edge subplots
            if j == 0:
                ax.set_ylabel("Weight", fontsize=10)
            if j == len(all_nl) - 1:
                ax2.set_ylabel("Reward (Smoothed)", fontsize=10)
            else:
                ax2.set_yticklabels([])  # Hide y-tick labels for non-edge subplots

            if i == len(all_lr) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)

            ax.label_outer()  # Hide inner labels

    # Synchronize secondary y-axis limits per row
    for row_idx, ax2_list in secondary_axes.items():
        if len(ax2_list) > 0:
            # Get the min and max y-values across all secondary axes in this row
            all_ylims = [ax2.get_ylim() for ax2 in ax2_list]
            global_ymin = min(ylim[0] for ylim in all_ylims)
            global_ymax = max(ylim[1] for ylim in all_ylims)

            # Set all secondary axes in this row to use the same limits
            for ax2 in ax2_list:
                ax2.set_ylim(global_ymin, global_ymax)

    # Add overall title
    fig.suptitle(
        "Weight Evolution: Rate Learning Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to {output_file}")
        plt.close(fig)
    else:
        plt.show()


def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

    # Automatically find the most recent results directory
    results_base_dir = base_dir / "results"
    name = "rate_learning_eligibility_LIF_20260312_154312"
    dirs = sorted(results_base_dir.glob(name), key=os.path.getmtime, reverse=True)
    if len(dirs) == 0:
        print(
            f"No rate learning results directories found in results folder {results_base_dir}!"
        )
        return
    results_dir = dirs[0] / "results"

    # Set up output directory
    output_dir = base_dir / "figures" / "rate_learning"
    os.makedirs(output_dir, exist_ok=True)

    output_file_heatmap = output_dir / f"{name}_performance_heatmap.png"
    output_file_curves = output_dir / f"{name}_learning_curves.png"

    print(f"Loading results from {results_dir}...")
    data = load_results(results_dir)

    if len(data) == 0:
        print("No data found!")
        return

    print(f"Found {len(data)} result files")

    # Get unique values
    all_nl = sorted(set(nl for nl, lr, iw in data.keys()))
    all_lr = sorted(set(lr for nl, lr, iw in data.keys()))
    all_iw = sorted(set(iw for nl, lr, iw in data.keys()))

    print(f"Noise levels: {all_nl}")
    print(f"Learning rates: {all_lr}")
    print(f"Initial weights: {all_iw}")

    # Plot heatmap
    print("\nCreating heatmap...")
    plot_heatmap(data, output_file_heatmap)

    # Plot learning curves
    print("\nCreating learning curves...")
    plot_learning_curves(data, output_file_curves)


if __name__ == "__main__":
    main()
