import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import numpy as jnp

from adaptive_SNN.utils.metrics import (
    compute_CV_ISI,
    compute_synchrony,
)

DATA_FOLDER = Path("results/network_tuning_only_e_input_20260423_025145/results")


def load_data(overwrite=False) -> pd.DataFrame:
    if (
        os.path.exists(DATA_FOLDER.parent / "network_tuning_results.pkl")
        and not overwrite
    ):
        return pd.read_pickle(DATA_FOLDER.parent / "network_tuning_results.pkl")

    all_data = []
    n_files = len([f for f in os.listdir(DATA_FOLDER) if f.endswith(".npz")])
    i = 0
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".npz"):
            print(f"Processing file {i + 1}/{n_files}", end="\r")
            i += 1
            N = int(file.split("_")[1])
            b = float(file.split("_")[3])
            data = jnp.load(os.path.join(DATA_FOLDER, file), allow_pickle=True)
            S = data["sol"].item().ys
            ts = data["sol"].item().ts
            synchrony = compute_synchrony(S)
            cv_isi = compute_CV_ISI(S, ts)
            mean_cv_isi = jnp.nanmean(cv_isi)
            neuron_firing_rates = jnp.sum(S, axis=0) / (ts[-1] - ts[0])  # in Hz
            mean_firing_rate = jnp.mean(neuron_firing_rates)
            std_neuron_firing_rate = jnp.std(neuron_firing_rates)
            # firing_rate_over_time = compute_network_firing_rate(S, ts)

            data_row = {
                "N_inputs": N,
                "balance": b,
                "synchrony": synchrony,
                "mean_cv_isi": float(mean_cv_isi),
                "mean_firing_rate": float(mean_firing_rate),
                "std_neuron_firing_rate": float(std_neuron_firing_rate),
                # "firing_rate_over_time": firing_rate_over_time,
            }
            all_data.append(data_row)
    df = pd.DataFrame(all_data)
    df.to_pickle(DATA_FOLDER.parent / "network_tuning_results.pkl")
    return df


def plot_results():
    df = load_data()
    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))

    print(df.head(3))

    flat_axs = axs.flatten()
    for i, metric in enumerate(["mean_cv_isi", "mean_firing_rate", "synchrony"]):
        ax = flat_axs[i]
        pivot = df.pivot(index="N_inputs", columns="balance", values=metric)

        ax.imshow(pivot.values)
        ax.set_xticks(
            np.arange(len(pivot.columns))[::2],
            labels=[f"{b:.2f}" for b in pivot.columns][::2],
        )
        ax.set_yticks(np.arange(len(pivot.index))[::2], labels=pivot.index[::2])

        ax.set_title(metric)
        ax.set_xlabel("Balance")
        ax.set_ylabel("Number of Inputs")
        fig.colorbar(ax.images[0], ax=ax, fraction=0.05, pad=0.03)
    plt.show()


def plot_spike_raster(file):
    data = jnp.load(file, allow_pickle=True)
    S = data["sol"].item().ys
    ts = data["sol"].item().ts

    synchrony = compute_synchrony(S)
    cv_isi = compute_CV_ISI(S, ts)
    mean_cv_isi = jnp.nanmean(cv_isi)
    neuron_firing_rates = jnp.sum(S, axis=0) / (ts[-1] - ts[0])  # in Hz
    mean_firing_rate = jnp.mean(neuron_firing_rates)
    std_neuron_firing_rate = jnp.std(neuron_firing_rates)

    print(
        f"Synchrony: {synchrony:.4f}, Mean CV ISI: {mean_cv_isi:.4f}, Mean Firing Rate: {mean_firing_rate:.2f} Hz, Std Firing Rate: {std_neuron_firing_rate:.2f} Hz"
    )

    plt.figure(figsize=(6, 4))
    for neuron_idx in range(S.shape[1]):
        spike_times = ts[S[:, neuron_idx] == 1]
        # Plot spikes using eventplot
        plt.eventplot(
            spike_times, lineoffsets=neuron_idx, colors="black", linelengths=0.8
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    plt.title("Spike Raster Plot")
    plt.show()


if __name__ == "__main__":
    plot_results()
    # plot_spike_raster(DATA_FOLDER / "N_150_b_0.750000_seed_0.npz")
