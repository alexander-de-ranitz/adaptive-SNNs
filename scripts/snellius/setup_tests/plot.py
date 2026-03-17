import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_SNN.models import SystemState


def parse_filename(filename):
    """Extract hyperparameters from filename.

    Expected format: setup_tests_iw{initial_weight}_balance{balance}_nl{noise_level}.npz
    """
    pattern = r"setup_tests_iw([\d.]+)_balance([\d.]+)_nl([\d.]+)\.npz"
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
        Dictionary with structure: {(balance, initial_weight): [(firing_rate, reward, environment_state, times), ...]}
    """
    results_path = Path(results_dir)
    data = defaultdict(list)

    for file in sorted(results_path.glob("setup_tests_*.npz")):
        try:
            # Parse hyperparameters from filename
            iw, balance, nl = parse_filename(file.name)

            # Load the data
            npz_data = np.load(file, allow_pickle=True)
            sol = npz_data["sol"].item()
            times = sol.ts

            state: SystemState = sol.ys
            # Extract data from the state structure
            reward_data = state.reward_signal
            environment_state = state.environment_state

            # # Calculate firing rate
            # firing_rate = calculate_firing_rate(spike_data, times)

            # Store results grouped by (balance, weight), including times for later processing
            data[(balance, iw)].append((reward_data, environment_state, times))

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue

    return data


def plot_curves(data, output_file=None):
    """Plot the reward, environment state, and firing rate as a function of initial weight for each balance level.

    This should result in three subplots:
    1. Firing rate vs initial weight, with one line per balance level
    2. Mean reward vs initial weight, with one line per balance level
    3. Mean environment state vs initial weight, with one line per balance level

    For all of these, the first 5 seconds of the simulation should be ignored to avoid transient effects, and the mean should be taken over the remaining time steps.

    Args:
        data: Dictionary with structure: {(balance, initial_weight): [(firing_rate, reward, environment_state, times), ...]}
        output_file: Optional path to save the figure
    """
    if len(data) == 0:
        print("No data to plot!")
        return

    # Reorganize data by balance level
    # Structure: {balance: {initial_weight: (avg_firing_rate, avg_reward, avg_env_state)}}
    data_by_balance = defaultdict(dict)

    for (balance, iw), results in data.items():
        # Average over multiple runs (different noise levels)
        mean_rewards = []
        mean_env_states = []

        for reward_data, environment_state, times in results:
            # Find index corresponding to t=5 seconds
            cutoff_idx = np.searchsorted(times, 0.0)

            if cutoff_idx < len(times) - 1:
                mean_rewards.append(np.mean(reward_data[cutoff_idx:]))
                mean_env_states.append(np.mean(environment_state[cutoff_idx:]))
            else:
                mean_rewards.append(np.mean(reward_data))
                mean_env_states.append(np.mean(environment_state))

        # Average over runs
        data_by_balance[balance][iw] = (
            np.mean(mean_rewards),
            np.mean(mean_env_states),
        )

    # Get unique balance levels
    all_balances = sorted(data_by_balance.keys())
    # I_weights = [compute_I_weight_for_balance(balance, 7.5) for balance in all_balances]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Use a colormap for different balance levels
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(all_balances))]

    # Plot 1: env state vs initial weight
    ax = axes[0]
    for balance_idx, balance in enumerate(all_balances):
        iws = sorted(data_by_balance[balance].keys())
        env_state = [data_by_balance[balance][iw][1] for iw in iws]
        ax.plot(
            iws,
            env_state,
            label=f"Balance={balance:.2f}",
            color=colors[balance_idx],
            marker="o",
        )
    ax.set_xlabel("Initial Weight")
    ax.set_ylabel("Mean Environment State")
    ax.legend()

    # Plot 2: Mean reward vs initial weight
    ax = axes[1]
    for balance_idx, balance in enumerate(all_balances):
        iws = sorted(data_by_balance[balance].keys())
        rewards = [data_by_balance[balance][iw][0] for iw in iws]
        ax.plot(
            iws,
            rewards,
            label=f"Balance={balance:.2f}",
            color=colors[balance_idx],
            marker="o",
        )
    ax.set_xlabel("Initial Weight")
    ax.set_ylabel("Mean Reward")

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Figure saved to {output_file}")

    plt.show()


def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent

    # Automatically find the most recent results directory
    results_base_dir = base_dir / "results"
    dirs = sorted(
        results_base_dir.glob("setup_tests_square_error_20260311_140120"),
        key=os.path.getmtime,
        reverse=True,
    )
    if len(dirs) == 0:
        print(
            f"No setup tests results directories found in results folder {results_base_dir}!"
        )
        return
    results_dir = dirs[0] / "results"
    output_file = base_dir / "figures" / "setup_tests" / "setup_tests.png"

    os.makedirs(output_file.parent, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    data = load_results(results_dir)

    if len(data) == 0:
        print("No data found!")
        return

    print(f"Found data for {len(data)} parameter combinations")
    for (balance, iw), points in sorted(data.items()):
        print(
            f"  Balance={balance:.2f}, Initial Weight={iw:.2f}: {len(points)} data points"
        )

    # Plot results
    plot_curves(data, output_file)


if __name__ == "__main__":
    main()
