import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_filename(filename):
    """Extract hyperparameters from filename.

    Expected format: FI_curve_iw{initial_weight}_balance{balance}_nl{noise_level}.npz
    """
    pattern = r"FI_curve_iw([\d.]+)_balance([\d.]+)_nl([\d.]+)\.npz"
    match = re.match(pattern, filename)
    if match:
        iw = float(match.group(1))
        balance = float(match.group(2))
        nl = float(match.group(3))
        return iw, balance, nl
    else:
        raise ValueError(f"Could not parse filename: {filename}")


def calculate_firing_rate(spike_data, times):
    """Calculate average firing rate from spike data.

    Args:
        spike_data: Array of shape (n_timesteps, n_neurons) with binary spike indicators
        times: Array of time points

    Returns:
        Average firing rate in Hz
    """
    # Sum all spikes across time
    total_spikes = np.sum(spike_data)

    # Calculate total time duration
    duration = times[-1] - times[0]

    # Calculate firing rate in Hz
    firing_rate = total_spikes / duration if duration > 0 else 0.0

    return firing_rate


def load_results(results_dir):
    """Load all simulation results from directory.

    Returns:
        Dictionary with structure: {(balance, noise_level): [(initial_weight, firing_rate), ...]}
    """
    results_path = Path(results_dir)
    data = defaultdict(list)

    for file in sorted(results_path.glob("FI_curve_*.npz")):
        try:
            # Parse hyperparameters from filename
            iw, balance, nl = parse_filename(file.name)

            # Load the data
            npz_data = np.load(file, allow_pickle=True)
            times = npz_data["times"]
            sol = npz_data["sol"].item()  # .item() to get the dict from 0-d array

            # Extract spike data from the state structure
            # sol is a NoisyNetworkState, so we access network_state.S
            if hasattr(sol, "network_state"):
                spike_data = sol.network_state.S
            elif isinstance(sol, dict) and "network_state" in sol:
                spike_data = sol["network_state"]["S"]
            else:
                print(f"Warning: Unexpected data structure in {file.name}")
                continue

            # Calculate firing rate
            firing_rate = calculate_firing_rate(spike_data, times)

            # Store results grouped by (balance, noise_level)
            data[(balance, nl)].append((iw, firing_rate))

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue

    return data


def plot_FI_curves(data, output_file=None):
    """Plot FI curves for different parameter combinations.

    Args:
        data: Dictionary with structure: {(balance, noise_level): [(initial_weight, firing_rate), ...]}
        output_file: Optional path to save the figure
    """
    if len(data) == 0:
        print("No data to plot!")
        return

    # Reorganize data by noise level, then by balance
    # Structure: {noise_level: {balance: [(weight, rate), ...]}}
    data_by_noise = defaultdict(lambda: defaultdict(list))
    for (balance, nl), points in data.items():
        data_by_noise[nl][balance] = sorted(points)  # Sort by weight

    # Get unique noise levels and balances
    noise_levels = sorted(data_by_noise.keys())
    all_balances = sorted(set(balance for (balance, nl) in data.keys()))

    # Create subplots (one for each noise level)
    n_plots = len(noise_levels)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    # Use a colormap for different balance levels
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(all_balances))]

    for idx, nl in enumerate(noise_levels):
        ax = axes[idx]

        # Plot each balance level as a different colored line
        for balance_idx, balance in enumerate(all_balances):
            if balance in data_by_noise[nl]:
                points = data_by_noise[nl][balance]
                if len(points) == 0:
                    continue

                weights = [p[0] for p in points]
                rates = [p[1] for p in points]

                ax.plot(
                    weights,
                    rates,
                    "o-",
                    color=colors[balance_idx],
                    linewidth=2,
                    markersize=6,
                    label=f"Balance={balance:.2f}",
                )

        ax.set_xlabel("Initial Weight", fontsize=12)
        ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
        ax.set_title(f"Noise Level={nl:.2f}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_file}")

    plt.show()


def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent

    # Automatically find the most recent results directory
    results_base_dir = base_dir / "results"
    tuning_curve_dirs = sorted(
        results_base_dir.glob("tuning_curves_*"), key=os.path.getmtime, reverse=True
    )
    if len(tuning_curve_dirs) == 0:
        print(
            f"No tuning curve results directories found in results folder {results_base_dir}!"
        )
        return
    results_dir = tuning_curve_dirs[0] / "results"
    output_file = base_dir / "figures" / "tuning_curves" / "FI_curves.png"

    os.makedirs(output_file.parent, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    data = load_results(results_dir)

    if len(data) == 0:
        print("No data found!")
        return

    print(f"Found data for {len(data)} parameter combinations")
    for (balance, nl), points in sorted(data.items()):
        print(f"  Balance={balance:.2f}, Noise={nl:.2f}: {len(points)} data points")

    # Plot results
    plot_FI_curves(data, output_file)


if __name__ == "__main__":
    main()
