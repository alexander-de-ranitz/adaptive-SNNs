import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from jax import numpy as jnp
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from adaptive_SNN.utils.metrics import compute_CV_ISI, compute_synchrony
from adaptive_SNN.utils.runner import _load_existing_solution
from adaptive_SNN.visualization.api.plotting import plot_spike_raster

RESULTS_DIR = "results/pendulum_AC_no_network_reset_20260729_142000/results/"


def load_pendulum_results(file_path):
    sol, _ = _load_existing_solution(file_path)
    return sol


def load_all_data():
    df = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".npz"):
            sol = load_pendulum_results(os.path.join(RESULTS_DIR, filename))
            w = float(re.search(r"w_(\d+\.?\d*)", filename).group(1))
            b = float(re.search(r"b_(\d+\.?\d*)", filename).group(1))
            df.append(
                {"filename": filename, "solution": sol, "weight": w, "balance": b}
            )
    return pd.DataFrame(df)


def colorbar(ax: Axes, im: AxesImage):
    """
    Add a color bar aligned to `im` neater than `fig.colorbar(im)`.
    https://stackoverflow.com/a/39938019/8954109
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    return ax.figure.colorbar(im, cax=cax, orientation="vertical")


def plot_full_dynamics():
    df = load_all_data()

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    df["error"] = jnp.nan  # Initialize the error column with NaN values
    df["firing_rate"] = jnp.nan  # Initialize the firing rate column with NaN values
    for i, (balance, subset) in enumerate(
        df.sort_values(by="balance").groupby(["balance"])
    ):
        for idx, row in subset.iterrows():
            sol = row["solution"]
            t = sol.ts
            charge_in, charge_out, E, I, W, firing_rate = sol.ys[0]
            # I, E, V = sol.ys[0]

            print(f"Weight: {row['weight']}, Balance: {row['balance']}")

            balance_ratio = (charge_in + charge_out) / (
                jnp.abs(charge_in) + jnp.abs(charge_out)
            )
            error = jnp.nanmean((balance_ratio[t > 100] - row["balance"]) ** 2)
            df.iloc[idx, df.columns.get_loc("error")] = error
            df.iloc[idx, df.columns.get_loc("firing_rate")] = jnp.nanmean(
                firing_rate[t > 100]
            )
            print(f"Weight: {row['weight']}, Balance: {row['balance']}, Error: {error}")

            axs[0].plot(
                t, charge_in, label=f"Charge In {row['weight']} {row['balance']}"
            )
            axs[0].plot(
                t, -charge_out, label=f"Charge Out {row['weight']} {row['balance']}"
            )
            axs[0].legend()

            axs[i].plot(
                t,
                balance_ratio,
                label=f"Balance Ratio {row['weight']} {row['balance']}",
            )
            axs[i].legend()
            axs[i].hlines(
                y=row["balance"], xmin=t[0], xmax=t[-1], color="r", linestyle="--"
            )

    plt.show()

    fig, axs = plt.subplots(1, 2)
    df["log_error"] = jnp.log(
        df["error"].to_numpy() + 1e-10
    )  # Add a small constant to avoid log(0)
    pivot = df.pivot(index="weight", columns="balance", values="log_error")
    axs[0].imshow(pivot, aspect="auto", origin="lower")
    colorbar(axs[0], axs[0].images[0])
    axs[0].set_xlabel("Balance")
    axs[0].set_xticks(
        ticks=jnp.arange(len(pivot.columns)), labels=[f"{b:.3f}" for b in pivot.columns]
    )
    axs[0].set_yticks(
        ticks=jnp.arange(len(pivot.index)), labels=[f"{w:.2f}" for w in pivot.index]
    )
    axs[0].set_ylabel("Initial Weight")
    axs[0].set_title("Error Heatmap")

    axs[1].imshow(
        df.pivot(index="weight", columns="balance", values="firing_rate"),
        aspect="auto",
        origin="lower",
    )
    colorbar(axs[1], axs[1].images[0])
    axs[1].set_xlabel("Balance")
    axs[1].set_xticks(
        ticks=jnp.arange(len(pivot.columns)), labels=[f"{b:.3f}" for b in pivot.columns]
    )
    axs[1].set_yticks(
        ticks=jnp.arange(len(pivot.index)), labels=[f"{w:.2f}" for w in pivot.index]
    )
    axs[1].set_ylabel("Initial Weight")
    axs[1].set_title("Firing Rate Heatmap")

    plt.show()


def plot_spiking():
    df = load_all_data()
    if len(df) > 1:
        print("Multiple results found. Using the first one for plotting.")
    sol = df.iloc[0]["solution"]
    ts = sol.ts
    spikes, balance, mean_W, mean_V, env_time = sol.ys[0]

    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    # Plot spike raster
    plot_spike_raster(ts, spikes, ax=ax1)
    y_min, y_max = 0, spikes.shape[1]
    ax1.fill_between(
        ts,
        y_min - 10,
        y_max + 10,
        where=(env_time < 0.25),
        color="gray",
        alpha=0.3,
        edgecolor="none",
    )
    # ax1.fill_between(ts, 1000, 900, color='darkblue', alpha=0.1, edgecolor='none')
    # ax1.fill_between(ts, 900, 800, color='darkred', alpha=0.1, edgecolor='none')
    ax1.set_xticks(
        ticks=jnp.linspace(ts[0], ts[-1], 11),
        labels=[f"{x:.1f}" for x in jnp.linspace(0, 2, 11)],
    )
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("Neuron Index")
    ax1.set_yticks(ticks=jnp.linspace(y_min, y_max, 11))

    # Compute and plot CV of ISI
    cv_isi = compute_CV_ISI(spikes, ts)
    cv_isi = cv_isi[jnp.isfinite(cv_isi)]  # Filter out NaN values
    print(cv_isi, jnp.sum(jnp.isnan(cv_isi)))
    ax2.hist(cv_isi, bins=30, color="gray", edgecolor="black")
    ax2.set_xlabel("CV of ISI")
    ax2.set_ylabel("Count")
    synchrony = compute_synchrony(spikes)
    ax2.legend()
    print(
        f"Synchrony: {synchrony}, mean voltage: {jnp.nanmean(mean_V)}, mean CV ISI: {jnp.nanmean(cv_isi)}"
    )

    # Plot firing rate histogram
    firing_rates = jnp.sum(spikes, axis=0) / (ts[-1] - ts[0])
    print(
        f"Mean firing rate: {jnp.mean(firing_rates)}, std firing rate: {jnp.std(firing_rates)}"
    )
    ax3.hist(firing_rates, bins=30, color="gray", edgecolor="black")
    ax3.set_xlabel("Firing Rate (Hz)")
    ax3.set_ylabel("Count")

    # Plot balance over time
    ax4.plot(ts, balance, color="k", alpha=0.2)
    ax4.plot(ts, jnp.mean(balance, axis=1), color="k", label="Mean Balance")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Balance")
    plt.show()


if __name__ == "__main__":
    # plot_spiking()
    plot_full_dynamics()
