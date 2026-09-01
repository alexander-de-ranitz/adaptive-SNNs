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

RESULTS_DIR = "results/pendulum_pretrained_20260813_171520/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    sol, _ = _load_existing_solution(file_path)
    ts = sol.ts
    saved_state = sol.ys[0]
    lr = float(re.search(r"_lr_(\d+\.?\d*)_", file_path).group(1))
    balance_rate = float(re.search(r"_balance_(\d+\.?\d*)_", file_path).group(1))
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
            mean_w_in,
            mean_w_rec,
            var_w_in,
            var_w_rec,
        ) = saved_state

        # Plotting the results
        fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex=False)
        axs[0, 0].plot(group["ts"].iloc[0], env_state[:, 0], label="Pendulum Angle")
        axs[0, 0].plot(group["ts"].iloc[0], env_state[:, 1], label="Pendulum Velocity")
        axs[0, 0].set_title("Pendulum State")
        axs[0, 0].legend()

        axs[0, 1].plot(group["ts"].iloc[0], agent_output[:, 0], label="Agent Output")
        axs[0, 1].set_title("Agent Output")
        axs[0, 1].legend()

        axs[1, 0].plot(group["ts"].iloc[0], value, label="Value")
        axs[1, 0].set_title("Value Function")
        axs[1, 0].legend()

        axs[1, 1].plot(group["ts"].iloc[0], RPE, label="RPE")
        axs[1, 1].set_title("Reward Prediction Error")
        axs[1, 1].legend()

        axs[2, 0].plot(group["ts"].iloc[0], reward, label="Reward")
        axs[2, 0].set_title("Reward")
        axs[2, 0].legend()

        axs[2, 1].plot(group["ts"].iloc[0], mean_w_in, label="Mean Input Weights")
        axs[2, 1].plot(group["ts"].iloc[0], mean_w_rec, label="Mean Recurrent Weights")
        axs[2, 1].set_title("Mean Weights")
        axs[2, 1].legend()

        axs[3, 0].plot(group["ts"].iloc[0], var_w_in, label="Var Input Weights")
        axs[3, 0].plot(group["ts"].iloc[0], var_w_rec, label="Var Recurrent Weights")
        axs[3, 0].set_title("Variance Weights")
        axs[3, 0].legend()

        bins = jnp.linspace(0.0, 0.05, 15)
        aggregate_hist = jnp.sum(hist, axis=0) / jnp.sum(hist)
        first_hist = hist[0] / jnp.sum(hist[0])
        last_hist = hist[-1] / jnp.sum(hist[-1])
        axs[3, 1].bar(
            bins[:-1],
            first_hist,
            width=bins[1] - bins[0],
            align="edge",
            alpha=0.5,
            label="First Time Step",
            fill=False,
            edgecolor="blue",
        )
        axs[3, 1].bar(
            bins[:-1],
            last_hist,
            width=bins[1] - bins[0],
            align="edge",
            alpha=0.5,
            label="Last Time Step",
            fill=False,
            edgecolor="red",
        )
        axs[3, 1].bar(
            bins[:-1],
            aggregate_hist,
            width=bins[1] - bins[0],
            align="edge",
            label="Aggregate",
            alpha=0.5,
            fill=False,
            edgecolor="black",
        )
        axs[3, 1].set_title("Balance Distribution")
        axs[3, 1].legend()
        plt.show()


if __name__ == "__main__":
    plot_pretrained_run()
