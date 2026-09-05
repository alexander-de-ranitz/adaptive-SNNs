import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from jax import numpy as jnp

from adaptive_snn.utils.runner import load_final_state
from adaptive_snn.utils.save_helper import load_named_result

RESULTS_DIR = "results/pendulum_AC_spiking_input_20260903_145905/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    result = load_named_result(file_path)
    ts = result["ts"]
    try:
        iter = int(re.search(r"_lr_\d+\.?\d*_(\d+)", file_path).group(1))
    except:
        try:
            iter = int(re.search(r"_(\d+)_lr_", file_path).group(1))
        except:
            iter = None
    chunk = int(re.search(r"_chunk_(\d+)", file_path).group(1))
    seed = int(re.search(r"_seed_(\d+)", file_path).group(1))
    lr = float(re.search(r"_lr_(\d+\.?\d*)_", file_path).group(1))
    final_state = load_final_state(file_path)
    return {
        "file_path": file_path,
        "state": result,
        "ts": ts,
        "iter": iter,
        "chunk": chunk,
        "final_state": final_state,
        "model": model,
        "lr": lr,
        "seed": seed,
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
    full_df = full_df.sort_values("file_path")

    for (m, lr, i, s), subset in full_df.sort_values(
        ["model", "lr", "iter", "seed"]
    ).groupby(["model", "lr", "iter", "seed"]):
        print(f"Plotting dynamics for model: {m}, lr: {lr}, iteration: {i}, seed: {s}")
        print(f"Files: {subset['file_path'].values}")
        subset = subset.sort_values("chunk")
        fig, axs = plt.subplots(5, 2, sharex=True, figsize=(5, 6))
        all_rewards = []
        all_RPEs = []
        for row in subset.itertuples():
            state = row.state
            all_rewards.append(state["mean_reward"])
            all_RPEs.append(state["mean_RPE"])

            ts = row.ts
            axs[0, 0].plot(ts, state["mean_reward"])
            axs[0, 0].set_title("Mean Reward")
            axs[1, 0].plot(ts, state["mean_RPE"])
            axs[1, 0].set_title("Mean RPE")
            axs[2, 0].plot(ts, state["mean_W_actor_input"])
            axs[2, 0].set_title("Mean Actor Input Weights")
            axs[3, 0].plot(ts, state["mean_W_actor_recurrent"])
            axs[3, 0].set_title("Mean Actor Recurrent Weights")
            axs[4, 0].plot(ts, state["mean_W_critic"])
            axs[4, 0].set_title("Mean Critic Weights")
            axs[4, 0].set_xlabel("Time")

            axs[0, 1].plot(ts, state["mean_filtered_spikes_L"], label="Left")
            axs[0, 1].plot(ts, -state["mean_filtered_spikes_R"], label="Right")
            axs[0, 1].plot(
                ts,
                state["mean_filtered_spikes_L"] - state["mean_filtered_spikes_R"],
                label="Difference",
            )
            axs[0, 1].legend()
            axs[0, 1].set_title("Mean Filtered Outputs")
            axs[1, 1].plot(ts, state["var_RPE"])
            axs[1, 1].set_title("Variance RPE")
            axs[2, 1].plot(ts, state["var_W_actor_recurrent"])
            axs[2, 1].set_title("Variance Actor Recurrent Weights")
            axs[3, 1].plot(ts, state["fraction_clipped_dW"])
            axs[3, 1].set_title("Fraction Clipped dW")
            axs[4, 1].plot(ts, state["mean_balance"])
            axs[4, 1].set_title("Mean Balance")
            axs[4, 1].axhline(
                y=0.01,
                xmin=ts[0],
                xmax=ts[-1],
                color="red",
                linestyle="--",
                label="Target Balance",
            )
            axs[4, 1].fill_between(
                ts,
                state["mean_balance"] - jnp.sqrt(state["var_balance"]),
                state["mean_balance"] + jnp.sqrt(state["var_balance"]),
                alpha=0.3,
            )
            print(
                f"Mean balance after warmup: {jnp.mean(state['mean_balance'][int(len(state['mean_balance']) / 4) :])}"
            )
            print(
                f"Std balance after warmup: {jnp.mean(jnp.sqrt(state['var_balance'][int(len(state['var_balance']) / 4) :]))}"
            )
        all_rewards = jnp.concatenate(all_rewards)
        all_rewards_trend = jnp.convolve(
            all_rewards.squeeze(), jnp.ones(100) / 100, mode="same"
        )
        axs[0, 0].plot(ts, all_rewards_trend, color="black", linewidth=2, label="Trend")

        all_RPEs = jnp.concatenate(all_RPEs)
        all_RPEs_trend = jnp.convolve(
            all_RPEs.squeeze(), jnp.ones(100) / 100, mode="same"
        )
        axs[1, 0].plot(ts, all_RPEs_trend, color="black", linewidth=2, label="Trend")
        print("Mean RPE after warmup: ", jnp.mean(all_RPEs[int(len(all_RPEs) / 4) :]))

        plt.show()

        # try:
        #     balance_rate = re.search(
        #         r"balance_rate_(\d+\.\d+)", subset["file_path"].values[0]
        #     ).group(1)
        # except AttributeError:
        #     balance_rate = None
        # try:
        #     tau = re.search(
        #         r"tau_charge_(\d+\.\d+)", subset["file_path"].values[0]
        #     ).group(1)
        # except AttributeError:
        #     tau = None
        # FIGURES_DIR = REPO_ROOT / "figures" / "pendulum" / "learned_I_weights"
        # FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # plt.savefig(
        #     FIGURES_DIR / f"{m}_lr_{lr}_balance_rate_{balance_rate}.png"
        # )


if __name__ == "__main__":
    plot_full_dynamics()
