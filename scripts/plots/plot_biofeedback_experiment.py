from pathlib import Path

import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

DATA_DIR = Path("results/biofeedback_20260520_173251/results")
OUTPUT_PATH = Path("figures/biofeedback")


def parse_run_file(file: Path):
    parts = file.stem.split("_")
    method = parts[0]
    lr = float(parts[2].replace("seed", ""))
    if method == "eligibility":
        method = "default"
    assert method in {"gated", "default"}, method
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
        columns=["method", "lr", "env_state", "target_w", "mean_w", "std_w", "ts"],
    )


def main():
    df = load_all_data(DATA_DIR)
    print(df.head())
    fig, axs = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    df = df.loc[df["lr"] != 0.0]
    for lr, group in df.groupby("lr"):
        for method, method_group in group.groupby("method"):
            ts = method_group["ts"].values[0]
            color = "#1D8F4D" if method == "gated" else "#747474"

            # Plot environment state (firing rate) for the target neuron and the entire network
            target_env_state = method_group["env_state"].values[0][:, 0]
            network_env_state = jnp.mean(method_group["env_state"].values[0], axis=1)

            axs[0].plot(
                ts[1:],
                target_env_state[1:],
                label=f"{method.capitalize()}",
                color=color,
            )
            axs[0].plot(
                ts[1:],
                network_env_state[1:],
                linestyle="--",
                color=color,
            )

            # Plot mean synaptic weight ot the target neuron and the entire network
            mean_w = method_group["mean_w"].values[0]
            target_w = method_group["target_w"].values[0]
            if jnp.any(~jnp.isfinite(target_w)):
                target_w = jnp.where(jnp.isfinite(target_w), target_w, jnp.nan)
            mean_w_target = jnp.nanmean(target_w, axis=1)

            axs[1].plot(
                ts[1:],
                mean_w_target[1:],
                label=f"{method.capitalize()}",
                color=color,
            )
            axs[1].plot(
                ts[1:],
                mean_w[1:],
                linestyle="--",
                color=color,
            )

    axs[0].set_ylabel("Firing Rate")
    axs[1].set_ylabel("Mean Synaptic Weight")
    for ax in axs:
        ax.legend()

    plt.xlabel("Time")
    plt.show()


if __name__ == "__main__":
    main()
