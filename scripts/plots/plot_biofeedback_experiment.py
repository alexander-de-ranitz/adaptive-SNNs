from pathlib import Path

import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

DATA_DIR = Path("results/biofeedback_20260519_155837/results")
OUTPUT_PATH = Path("figures/biofeedback")


def parse_run_file(file: Path):
    parts = file.stem.split("_")
    method = parts[0]
    lr = float(parts[2].replace("seed", ""))
    assert method in {"gated", "eligibility"}, method
    return method, lr


def load_run_arrays(file: Path) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    with jnp.load(file, allow_pickle=True) as data:
        stats = data["sol"].item().ys
        ts = data["sol"].item().ts
    metadata = parse_run_file(file)
    return stats, metadata, ts


def load_all_data(data_dir: Path) -> pd.DataFrame:
    run_files = []
    for file in data_dir.glob("*.npz"):
        data, metadata, ts = load_run_arrays(file)
        run_files.append([*metadata, *data, ts])
    return pd.DataFrame(
        run_files,
        columns=["method", "lr", "S", "RPE", "W_target", "W_mean", "env_state", "ts"],
    )


def main():
    df = load_all_data(DATA_DIR)
    print(df.head())
    fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True)
    df = df.loc[df["lr"] != 0.0]
    for lr, group in df.groupby("lr"):
        for method, method_group in group.groupby("method"):
            ts = method_group["ts"].values[0]
            color = "#1D8F4D" if method == "gated" else "#747474"

            # Plot environment state (firing rate) for the target neuron and the entire network
            target_env_state = method_group["env_state"].values[0][:, 0]
            network_env_state = jnp.mean(method_group["env_state"].values[0], axis=1)

            # Smooth the signal
            target_env_state = jnp.convolve(
                target_env_state, jnp.ones(1000) / 1000, mode="same"
            )
            axs[0].plot(
                ts[1:],
                target_env_state[1:],
                label=f"Target Neuron ({method}, lr={lr})",
                color=color,
            )
            axs[0].plot(
                ts[1:],
                network_env_state[1:],
                label=f"Network ({method}, lr={lr})",
                linestyle="--",
                color=color,
            )

            # Plot mean synaptic weight ot the target neuron and the entire network
            mean_w = method_group["W_mean"].values[0]
            target_w = jnp.nanmean(method_group["W_target"].values[0], axis=1)

            axs[1].plot(
                ts[1:],
                target_w[1:],
                label=f"Target Neuron Mean Weight ({method}, lr={lr})",
                color=color,
            )
            axs[1].plot(
                ts[1:],
                mean_w[1:],
                label=f"Network Mean Weight ({method}, lr={lr})",
                linestyle="--",
                color=color,
            )

    for ax in axs:
        ax.legend()

    plt.xlabel("Time")
    plt.show()


if __name__ == "__main__":
    main()
