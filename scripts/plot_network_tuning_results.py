import math
import os

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from adaptive_SNN.utils.metrics import (
    compute_CV_ISI,
    compute_network_firing_rate,
    compute_synchrony,
)


def plot_results():
    data = {}
    for file in os.listdir("data/network_tuning"):
        if file.endswith(".npz"):
            iw = float(file.split("_")[3])
            rw = float(file.split("_")[5])
            results = jnp.load(os.path.join("data/network_tuning", file))
            ts = results["ts"]
            S = results["S"]

            synchrony = compute_synchrony(S)
            cv_isi = compute_CV_ISI(S, ts)
            mean_cv_isi = jnp.nanmean(cv_isi)
            neuron_firing_rates = jnp.sum(S, axis=0) / (ts[-1] - ts[0])  # in Hz
            mean_neuron_firing_rate = jnp.mean(neuron_firing_rates)
            std_neuron_firing_rate = jnp.std(neuron_firing_rates)
            population_firing_rate = compute_network_firing_rate(S, ts)
            mean_population_firing_rate = jnp.mean(population_firing_rate)
            std_population_firing_rate = jnp.std(population_firing_rate)

            data[(iw, rw)] = {
                "synchrony": float(synchrony),
                "mean_cv_isi": float(mean_cv_isi),
                "mean_firing_rate": float(mean_neuron_firing_rate),
                "std_neuron_firing_rate": float(std_neuron_firing_rate),
                "mean_population_firing_rate": float(mean_population_firing_rate),
                "std_population_firing_rate": float(std_population_firing_rate),
            }
    if not data:
        raise RuntimeError("No .npz files found in data/network_tuning")

    iw_values = sorted({iw for (iw, _) in data.keys()})
    rw_values = sorted({rw for (_, rw) in data.keys()})
    iw_index = {value: idx for idx, value in enumerate(iw_values)}
    rw_index = {value: idx for idx, value in enumerate(rw_values)}

    metric_names = list(next(iter(data.values())).keys())
    n_cols = min(3, len(metric_names)) or 1
    n_rows = math.ceil(len(metric_names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    label_formatter = lambda values: [f"{value:g}" for value in values]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        heatmap = np.full((len(rw_values), len(iw_values)), np.nan, dtype=float)
        for (iw, rw), metrics in data.items():
            heatmap[rw_index[rw], iw_index[iw]] = metrics[metric]

        im = ax.imshow(heatmap, origin="lower", cmap="viridis")
        im.cmap.set_bad(color="#f0f0f0")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(range(len(iw_values)))
        ax.set_xticklabels(label_formatter(iw_values))
        ax.set_xlabel("iw")
        ax.set_yticks(range(len(rw_values)))
        ax.set_yticklabels(label_formatter(rw_values))
        ax.set_ylabel("rw")

        valid_mask = np.isfinite(heatmap)
        if valid_mask.any():
            vmin = np.nanmin(heatmap)
            vmax = np.nanmax(heatmap)
            midpoint = vmin + (vmax - vmin) / 2
        else:
            vmin = vmax = midpoint = 0.0
        for row in range(len(rw_values)):
            for col in range(len(iw_values)):
                value = heatmap[row, col]
                if np.isnan(value):
                    continue
                text_color = "black" if vmax > vmin and value >= midpoint else "white"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes created by the grid layout
    for ax in axes[len(metric_names) :]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
