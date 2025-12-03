import os

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from joblib import memory

from adaptive_SNN.utils.metrics import (
    compute_CV_ISI,
    compute_network_firing_rate,
    compute_synchrony,
)

mem = memory.Memory(location="cachedir", verbose=0)


@mem.cache
def load_data():
    data = {}
    n_files = len(os.listdir("data/network_tuning"))
    curr = 0
    for file in os.listdir("data/network_tuning"):
        if file.endswith(".npz"):
            parts = file[:-4].split("_")
            weight = float(parts[5])
            balance = float(parts[7])
            results = jnp.load(os.path.join("data/network_tuning", file))
            ts = results["ts"]
            S = results["S"]

            print(
                "\033[2K\r" + f"Processing file {file}. Progress = {curr}/{n_files}",
                end="",
            )
            curr += 1

            synchrony = compute_synchrony(S)
            cv_isi = compute_CV_ISI(S, ts)
            mean_cv_isi = jnp.nanmean(cv_isi)
            neuron_firing_rates = jnp.sum(S, axis=0) / (ts[-1] - ts[0])  # in Hz
            mean_neuron_firing_rate = jnp.mean(neuron_firing_rates)
            std_neuron_firing_rate = jnp.std(neuron_firing_rates)
            population_firing_rate = compute_network_firing_rate(S, ts)
            mean_population_firing_rate = jnp.mean(population_firing_rate)
            std_population_firing_rate = jnp.std(population_firing_rate)

            data[(weight, balance)] = {
                "synchrony": float(synchrony),
                "mean_cv_isi": float(mean_cv_isi),
                "mean_firing_rate": float(mean_neuron_firing_rate),
                "std_neuron_firing_rate": float(std_neuron_firing_rate),
                "mean_population_firing_rate": float(mean_population_firing_rate),
                "std_population_firing_rate": float(std_population_firing_rate),
            }
    return data


def plot_results():
    data = load_data()
    if not data:
        raise RuntimeError("No .npz files found in data/network_tuning")

    weights = sorted({w for (w, _) in data.keys()})
    balances = sorted({b for (_, b) in data.keys()})
    w_index = {value: idx for idx, value in enumerate(weights)}
    b_index = {value: idx for idx, value in enumerate(balances)}

    metric_names = [
        "synchrony",
        "mean_cv_isi",
        "mean_firing_rate",
        "std_neuron_firing_rate",
    ]
    n_cols = 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    label_formatter = lambda values: [f"{value:.1f}" for value in values]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        heatmap = np.full((len(weights), len(balances)), np.nan, dtype=float)
        for (w, b), metrics in data.items():
            heatmap[w_index[w], b_index[b]] = metrics[metric]

        im = ax.imshow(heatmap, origin="lower", cmap="Greys_r")
        im.cmap.set_bad(color="#f0f0f0")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(range(len(balances)))
        ax.set_xticklabels(label_formatter(balances))
        ax.set_xlabel("Balance")
        ax.set_yticks(range(len(weights)))
        ax.set_yticklabels(label_formatter(weights))
        ax.set_ylabel("Weight")

        # # Uncomment to add cell values in text form
        # valid_mask = np.isfinite(heatmap)
        # if valid_mask.any():
        #     vmin = np.nanmin(heatmap)
        #     vmax = np.nanmax(heatmap)
        #     midpoint = vmin + (vmax - vmin) / 2
        # else:
        #     vmin = vmax = midpoint = 0.0
        # for row in range(len(balances)):
        #     for col in range(len(weights)):
        #         value = heatmap[row, col]
        #         if np.isnan(value):
        #             continue
        #         text_color = "black" if vmax > vmin and value >= midpoint else "white"
        #         ax.text(
        #             col,
        #             row,
        #             f"{value:.2f}",
        #             ha="center",
        #             va="center",
        #             color=text_color,
        #             fontsize=9,
        #         )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes created by the grid layout
    for ax in axes[len(metric_names) :]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
