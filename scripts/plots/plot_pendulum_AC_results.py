import os

import matplotlib.pyplot as plt
import pandas as pd
from jax import numpy as jnp

from adaptive_SNN.utils.runner import _load_existing_solution

RESULTS_DIR = "results/pendulum_AC_20260618_154401/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    sol, _ = _load_existing_solution(file_path)
    ts = sol.ts
    ys = sol.ys[0]
    env_state, reward_signal, predicted_value, RPE = ys
    final_state = sol.ys[1]
    return {
        "ts": ts,
        "env_state": env_state,
        "reward_signal": reward_signal,
        "predicted_value": predicted_value,
        "RPE": RPE,
        "model": model,
        "final_state": final_state,
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


def plot_full_dynamics():
    full_df = load_all_data()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 12))

    # Plot reward signal
    axs[0].plot(full_df.iloc[0]["ts"], full_df.iloc[0]["RPE"], label="RPE", alpha=0.7)
    axs[0].plot(
        full_df.iloc[0]["ts"], full_df.iloc[0]["reward_signal"], label="Reward Signal"
    )
    axs[0].set_title("Reward Signal and RPE")
    axs[0].legend()
    axs[0].set_ylabel("Value")
    axs[0].set_xlabel("Time (s)")

    print(
        f"Mean reward = {jnp.mean(full_df.iloc[0]['reward_signal'])}, Mean RPE = {jnp.mean(full_df.iloc[0]['RPE'])}"
    )

    # Plot predicted value
    axs[1].plot(
        full_df.iloc[0]["ts"],
        full_df.iloc[0]["predicted_value"],
        label="Predicted Value",
        color="orange",
    )
    axs[1].set_title("Predicted Value")
    axs[1].set_ylabel("Value")
    axs[1].set_xlabel("Time (s)")

    # Plot environment state
    axs[2].plot(
        full_df.iloc[0]["ts"],
        full_df.iloc[0]["env_state"][:, 0],
        label="Pendulum Angle",
    )
    axs[2].plot(
        full_df.iloc[0]["ts"],
        full_df.iloc[0]["env_state"][:, 1],
        label="Pendulum Angular Velocity",
    )
    axs[2].set_title("Environment State")
    axs[2].set_ylabel("Value")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()

    plt.show()

    weights = full_df.iloc[0]["final_state"].agent_state.reward_predictor_state.weights[
        :-1
    ]
    plt.imshow(
        weights.reshape(16, 16).T,
        aspect="auto",
        origin="lower",
        extent=[0, 16, 0, 16],
        interpolation="none",
    )
    plt.colorbar(label="Weight Value")
    plt.xlabel("Angle Encoding Neuron Index")
    plt.ylabel("Angular Velocity Encoding Neuron Index")
    plt.title("Learned Critic Weights at Final Timepoint")
    plt.show()

    # Show the mean RPE binned by state angle
    angle_bins = jnp.linspace(-0.3, 0.3, 30)
    mean_rpe_by_angle = []
    for i in range(len(angle_bins) - 1):
        mask = (full_df.iloc[0]["env_state"][:, 0] >= angle_bins[i]) & (
            full_df.iloc[0]["env_state"][:, 0] < angle_bins[i + 1]
        )
        mean_rpe_by_angle.append(jnp.mean(full_df.iloc[0]["RPE"][mask]))
    plt.bar(
        angle_bins[:-1],
        mean_rpe_by_angle,
        width=angle_bins[1] - angle_bins[0],
        align="edge",
    )
    plt.xlabel("Pendulum Angle")
    plt.ylabel("Mean RPE")
    plt.title("Mean RPE by Pendulum Angle")
    plt.show()


if __name__ == "__main__":
    plot_full_dynamics()
