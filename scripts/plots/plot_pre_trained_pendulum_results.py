# Add scripts/ to the path so that we can import from scripts
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from jax import numpy as jnp

from adaptive_SNN.utils.runner import _load_existing_solution

RESULTS_DIR = "results/pendulum_pretrained_20260904_140811/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    sol, _ = _load_existing_solution(file_path)
    ts = sol.ts
    saved_state = sol.ys[0]
    try:
        lr = float(re.search(r"_lr_(\d+\.?\d*)_", file_path).group(1))
    except:
        lr = None
    try:
        balance_rate = float(re.search(r"_balance_(\d+\.?\d*)_", file_path).group(1))
    except:
        balance_rate = None
    final_state = sol.ys[1]
    return {
        "file_path": file_path,
        "state": saved_state,
        "ts": ts,
        "final_state": final_state,
        "model": model,
        "lr": lr,
        "balance_rate": balance_rate,
    }


def load_all_data():
    df = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".npz"):
            print("Loading file: ", filename)
            if filename.__contains__("final_weights"):
                continue
            file_path = os.path.join(RESULTS_DIR, filename)
            df.append(load_pendulum_results(file_path))
    return pd.DataFrame(df)


def plot_pretrained_run():
    full_df = load_all_data()
    full_df = full_df.sort_values(["model", "lr", "balance_rate"])

    for (m, lr, b), group in full_df.groupby(["model", "lr", "balance_rate"]):
        if len(group) != 1:
            raise ValueError("More than one file in group!")
        print(f"Plotting dynamics for model: {m}, lr: {lr}, balance_rate: {b}")
        saved_state = group["state"].iloc[0]
        (
            env_state,
            agent_output,
            value,
            RPE,
            reward,
            hist,
            L_rate,
            R_rate,
            H_rate,
            min_bal,
            max_bal,
            mean_bal,
            var_bal,
            mean_w_in,
            mean_w_rec,
            var_w_in,
            var_w_rec,
        ) = saved_state

        # Plotting the results
        fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=True)
        axs[0, 0].plot(group["ts"].iloc[0], env_state[:, 0], label="Pendulum Angle")
        axs[0, 0].plot(group["ts"].iloc[0], env_state[:, 1], label="Pendulum Velocity")
        axs[0, 0].set_title("Pendulum State")
        axs[0, 0].legend()

        axs[0, 1].plot(group["ts"].iloc[0], agent_output[:, 0], label="Agent Output")
        axs[0, 1].set_title("Agent Output")
        axs[0, 1].legend()

        # axs[1, 0].plot(group["ts"].iloc[0], value, label="Value")
        # axs[1, 0].set_title("Value Function")
        # axs[1, 0].legend()
        axs[1, 0].plot(group["ts"].iloc[0], mean_bal, label="Mean Balance")
        axs[1, 0].fill_between(
            group["ts"].iloc[0],
            mean_bal - jnp.sqrt(var_bal),
            mean_bal + jnp.sqrt(var_bal),
            color="gray",
            alpha=0.3,
        )
        axs[1, 0].plot(group["ts"].iloc[0], min_bal, label="Min Balance")
        axs[1, 0].plot(group["ts"].iloc[0], max_bal, label="Max Balance")
        axs[1, 0].set_title("Balance Metrics")
        axs[1, 0].legend()

        axs[1, 1].plot(group["ts"].iloc[0], RPE, label="RPE")
        axs[1, 1].set_title("Reward Prediction Error")
        axs[1, 1].legend()

        filter_size = 1000
        filtered_reward = jnp.convolve(
            reward.squeeze(), jnp.ones(filter_size) / filter_size, mode="valid"
        )
        axs[2, 0].plot(
            group["ts"].iloc[0], reward, label="Reward", linewidth=0.5, alpha=0.5
        )
        axs[2, 0].plot(
            group["ts"].iloc[0][filter_size // 2 : -filter_size // 2 + 1],
            filtered_reward,
            label="Filtered Reward",
        )
        axs[2, 0].set_title("Reward")
        axs[2, 0].legend()

        axs[2, 1].plot(group["ts"].iloc[0], L_rate, label="Left Rate", alpha=0.5)
        axs[2, 1].plot(group["ts"].iloc[0], R_rate, label="Right Rate", alpha=0.5)
        axs[2, 1].plot(group["ts"].iloc[0], H_rate, label="Hidden Rate", alpha=0.5)
        axs[2, 1].set_title("Neuron Firing Rates")
        axs[2, 1].legend()

        bins = jnp.linspace(0.0, 0.05, 15)
        axs[3, 1].imshow(
            hist.T,
            aspect="auto",
            extent=[group["ts"].iloc[0][0], group["ts"].iloc[0][-1], 0, hist.shape[1]],
            origin="lower",
        )
        axs[3, 1].set_title("Balance Histogram")
        axs[3, 1].set_xlabel("Time")
        axs[3, 1].set_ylabel("Balance Bins")
        axs[3, 1].set_yticks(jnp.arange(len(bins) - 1))
        axs[3, 1].set_yticklabels([f"{bins[i]:.2f}" for i in range(len(bins) - 1)])

        plt.show()


if __name__ == "__main__":
    plot_pretrained_run()
