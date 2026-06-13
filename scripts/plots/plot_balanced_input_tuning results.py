import os
import re

import numpy as np
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

DATA_DIR = "results/balance_tuning_20260605_101130/results"


def load_results():
    df = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".npz"):
            data = np.load(os.path.join(DATA_DIR, f))
            cv_isi = float(data["CV_ISI"].squeeze())
            firing_rate = float(data["firing_rate"].squeeze())
            charge_ratio = float(data["charge_ratio"].squeeze())
            mean_voltage = float(data["mean_voltage"].squeeze())
            balance = float(re.search(r"b_(\d+\.\d+)", f).group(1))
            init_weight = float(re.search(r"w_(\d+\.\d+)", f).group(1))
            df.append(
                {
                    "balance": balance,
                    "init_weight": init_weight,
                    "CV_ISI": cv_isi,
                    "firing_rate": firing_rate,
                    "charge_ratio": charge_ratio,
                    "mean_voltage": mean_voltage,
                }
            )
    df = pd.DataFrame(df)
    return df


def main():
    df = load_results()

    init_weights = jnp.array(sorted(df["init_weight"].unique()))
    balances = jnp.array(sorted(df["balance"].unique()))

    def add_pixel_outline(ax, mask, color="white", linewidth=1.5):
        mask_np = np.asarray(mask, dtype=bool)
        n_rows, n_cols = mask_np.shape
        segments = []

        for row in range(n_rows):
            for col in range(n_cols):
                if not mask_np[row, col]:
                    continue

                x0, x1 = col - 0.5, col + 0.5
                y0, y1 = row - 0.5, row + 0.5

                if row == 0 or not mask_np[row - 1, col]:
                    segments.append([(x0, y0), (x1, y0)])
                if row == n_rows - 1 or not mask_np[row + 1, col]:
                    segments.append([(x0, y1), (x1, y1)])
                if col == 0 or not mask_np[row, col - 1]:
                    segments.append([(x0, y0), (x0, y1)])
                if col == n_cols - 1 or not mask_np[row, col + 1]:
                    segments.append([(x1, y0), (x1, y1)])

        if segments:
            ax.add_collection(
                LineCollection(segments, colors=color, linewidths=linewidth)
            )

    metric_matrices = [
        df.pivot(index="balance", columns="init_weight", values="CV_ISI").to_numpy(),
        df.pivot(
            index="balance", columns="init_weight", values="firing_rate"
        ).to_numpy(),
        df.pivot(
            index="balance", columns="init_weight", values="charge_ratio"
        ).to_numpy(),
        df.pivot(
            index="balance", columns="init_weight", values="mean_voltage"
        ).to_numpy(),
    ]

    # Do not show CV ISI for neurons with firing rate < 1 Hz, as CV ISI is not meaningful for very low firing rates.
    cv_isi = metric_matrices[0].copy()
    cv_isi[metric_matrices[1] < 1.0] = jnp.nan
    metric_matrices[0] = cv_isi

    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    axs = axs.flatten()
    images = [
        ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest")
        for ax, matrix in zip(axs, metric_matrices)
    ]

    titles = [
        "CV of ISI",
        "Firing Rate",
        "E/I Ratio",
        "Mean Voltage",
    ]

    target_ranges = [
        (0.9, 1.1),
        (5.0, 20.0),
        None,
        (-60, jnp.inf),
    ]

    x_tick_labels = [f"{float(w):.1f}" for w in init_weights][::2]
    y_tick_labels = [f"{b:.2f}" for b in balances][::2]

    for ax, title in zip(axs, titles):
        ax.set_xticks(jnp.arange(len(x_tick_labels)) * 2)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(jnp.arange(len(y_tick_labels)) * 2)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("E Weight")
        ax.set_ylabel("Balance (E/I Ratio)")
        ax.set_title(rf"\textbf{{{title}}}")

    target_masks = []
    for matrix, target in zip(metric_matrices, target_ranges):
        if target is None:
            target_masks.append(None)
            continue

        lower, upper = target
        mask = ~jnp.isnan(matrix) & (matrix >= lower) & (matrix <= upper)
        target_masks.append(mask)

    # valid_target_masks = [mask for mask in target_masks if mask is not None]
    # common_target_mask = jnp.logical_and.reduce(jnp.stack(valid_target_masks), axis=0)

    # for ax, mask in zip(axs, target_masks):
    #     if mask is not None and bool(jnp.any(mask)):
    #         add_pixel_outline(ax, mask, color="lightgray", linewidth=1.5)

    #     if bool(jnp.any(common_target_mask)):
    #         add_pixel_outline(ax, common_target_mask, color="white", linewidth=2.0)

    # Add colorbars
    for ax, img in zip(axs, images):
        fig.colorbar(img, ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
