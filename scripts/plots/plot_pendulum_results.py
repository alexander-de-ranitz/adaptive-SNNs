import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import numpy as jnp

RESULTS_DIR = "results/pendulum_run_20260609_104725/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    data = np.load(file_path, allow_pickle=True)
    ts = data["ts"].item()
    ys = data["ys"]
    # env_state, reward_signal, reward_predictor_state, network_output, weights = ys
    (
        env_state,
        reward_signal,
        reward_predictor_state,
        network_output,
        rec_firing_rate,
        left_firing_rate,
        right_firing_rate,
        weights,
    ) = ys
    return {
        "ts": ts,
        "env_state": env_state,
        "reward_signal": reward_signal,
        "reward_predictor_state": reward_predictor_state,
        "network_output": network_output,
        "rec_firing_rate": rec_firing_rate,
        "left_firing_rate": left_firing_rate,
        "right_firing_rate": right_firing_rate,
        "weights": weights,
        "model": model,
    }


def load_all_data():
    df = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".npz"):
            file_path = os.path.join(RESULTS_DIR, filename)
            df.append(load_pendulum_results(file_path))
    return pd.DataFrame(df)


def plot_full_dynamics():
    full_df = load_all_data()

    for model, df in full_df.groupby("model"):
        print(f"Plotting results for model: {model}")
        fig, axs = plt.subplots(len(df), 2, figsize=(6, 6), sharex=True)
        for i in range(len(df)):
            for j in range(2):
                ts = df.iloc[i]["ts"]
                weights = df.iloc[i]["weights"]
                indices = jnp.nonzero(~jnp.isnan(weights[0, j, :]))[0]
                for k in indices:
                    # c = "lightblue" if k < 25 else ("darkblue" if k < 50 else ("lightcoral" if k < 75 else "darkred"))
                    # label = "left angle" if k < 25 else ("right angle" if k < 50 else ("left ang vel" if k < 75 else "right ang vel"))
                    c = "b" if k < 33 else ("r" if k < 66 else "g")
                    label = (
                        "Recurrent"
                        if k < 33
                        else ("Angle" if k < 66 else "Angular Velocity")
                    )
                    axs[i, j].plot(
                        ts, weights[:, j, k], alpha=0.5, color=c, label=label
                    )
        axs[0, 0].set_title("Left Output Neuron")
        axs[0, 1].set_title("Recurrent Neuron")
        axs[0, 0].legend()
        plt.show()

        fig, axs = plt.subplots(len(df), 1, figsize=(6, 6), sharex=True)
        for i in range(len(df)):
            ts = df.iloc[i]["ts"]
            reward_signal = df.iloc[i]["reward_signal"]
            reward_predictor_state = df.iloc[i]["reward_predictor_state"]
            axs[i].plot(ts, reward_signal, label="Reward Signal", color="k")
            axs[i].plot(
                ts, reward_predictor_state, label="Reward Predictor State", color="r"
            )
            axs[i].set_title(f"Run {i + 1}")
            axs[i].legend()
        plt.show()

        fig, axs = plt.subplots(len(df), 1, figsize=(6, 6), sharex=True)
        for i in range(len(df)):
            ts = df.iloc[i]["ts"]
            reward_signal = df.iloc[i]["reward_signal"]
            reward_predictor_state = df.iloc[i]["reward_predictor_state"]
            rpe = reward_signal.squeeze() - reward_predictor_state.squeeze()
            axs[i].plot(
                ts,
                jnp.convolve(rpe, jnp.ones(1000) / 1000, mode="same"),
                label="RPE",
                color="k",
            )
            axs[i].set_title(f"Run {i + 1}")
            axs[i].legend()
        plt.show()

        n_runs = len(df)
        fig, axs = plt.subplots(n_runs, 1, figsize=(6, 6), sharex=True)
        for i, col in enumerate(
            [
                "rec_firing_rate",
                "left_firing_rate",
                "right_firing_rate",
                "network_output",
            ]
        ):
            axs[i].set_title(f"{col}")
            for j in range(n_runs):
                ts = df.iloc[j]["ts"]
                data = df.iloc[j][col]
                if col == "env_state":
                    data = data[:, 0]  # Plot only the angle of the pendulum
                axs[i].plot(ts, data)
        plt.show()

        rs = []
        rpes = []
        ts = df.iloc[0]["ts"]
        for i, r in enumerate(df["reward_signal"].to_numpy()):
            r = jnp.asarray(r).squeeze()
            rs.append(r)
            r_pred = df.iloc[i]["reward_predictor_state"].squeeze()
            pred_corr = jnp.corrcoef(r.squeeze(), r_pred)[0, 1]
            rpes.append(r - r_pred)
            print(
                f"Correlation between reward signal and predictor state: {pred_corr:.3f}"
            )
        rs_stacked = jnp.vstack(rs)
        rpes_stacked = jnp.vstack(rpes)
        print(f"Mean RPE across runs: {jnp.mean(rpes_stacked):.3f}")
        mean_rs = jnp.mean(rs_stacked, axis=0)
        filter = jnp.ones(1000) / 1000
        plt.plot(
            ts[filter.shape[0] // 2 : -filter.shape[0] // 2 + 1],
            jnp.convolve(mean_rs, filter, mode="valid"),
            c="k",
        )
        for r in rs:
            plt.plot(
                ts[filter.shape[0] // 2 : -filter.shape[0] // 2 + 1],
                jnp.convolve(r, filter, mode="valid"),
                alpha=0.1,
                c="k",
                linewidth=0.5,
            )
        plt.xlabel("Time")
        plt.ylabel("Reward Signal")
        plt.title("Average Reward Signal Across Runs")
        plt.show()

        # Plot reward distribution and predicted reward distribution
        plt.figure(figsize=(6, 6))
        for i in range(n_runs):
            reward_signal = df.iloc[i]["reward_signal"]
            reward_predictor_state = df.iloc[i]["reward_predictor_state"]
            rpe = reward_signal - reward_predictor_state
            plt.scatter(reward_signal, reward_predictor_state, alpha=0.5, c="k")
        plt.xlabel("Reward Signal")
        plt.ylabel("Predicted Reward")
        plt.title("Reward Signal vs. Predicted Reward")
        plt.show()

        for i in range(len(df)):
            ts = df.iloc[i]["ts"]
            env_state = df.iloc[i]["env_state"]
            t_env = env_state[:, -1]  # Time points for the environment state
            t_diff = jnp.diff(t_env)
            n_completed = jnp.sum(t_diff < -4.9)
            print(jnp.sort(t_diff)[:10])
            print(f"Run {i + 1}: Completed {n_completed} trials")


if __name__ == "__main__":
    plot_full_dynamics()
