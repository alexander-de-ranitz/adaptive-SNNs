import jax

jax.config.update("jax_enable_x64", True)

# Add scripts/ to the path so that we can import from scripts
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import re
import time

import diffrax as dfx
import pandas as pd
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_snn.models.agent_env_system import SystemState
from adaptive_snn.models.networks.eligibility_LIF import EligibilityLIFNetwork
from adaptive_snn.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.utils.runner import load_final_state, run_simulation
from adaptive_snn.utils.save_helper import load_named_result, save_part_of_state

RESULTS_DIR = "results/pendulum_AC_noiseless_input_20260808_043000/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    result = load_named_result(file_path)
    ts = result["ts"]
    iter = int(re.search(r"_lr_\d+\.?\d*_(\d+)", file_path).group(1))
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


def main():
    df = load_all_data()

    selected_run = df.loc[
        (df["model"] == "gated") & (df["lr"] == 1000) & (df["iter"] == 1)
    ].iloc[0]
    y0 = selected_run["final_state"]
    y0: SystemState = jax.tree.map(
        lambda x: x.astype(jnp.float64), y0
    )  # We store in 32 bit but need 64 bit in the simulation
    model = selected_run["model"]
    ts = selected_run["ts"]

    charge_in = y0.agent_state.network_state.charge_in
    charge_out = y0.agent_state.network_state.charge_out
    balance = (charge_in + charge_out) / (jnp.abs(charge_in) + jnp.abs(charge_out))
    print("Mean balance = ", jnp.mean(balance), " std balance = ", jnp.std(balance))
    plt.hist(balance)
    plt.show()
    return

    t0 = ts[-1]
    delta_t = 1.0
    t1 = t0 + delta_t
    key = jr.fold_in(jr.PRNGKey(int(selected_run["seed"])), int(selected_run["iter"]))
    config = create_pendulum_AC_config(
        N_neurons=1000,
        key=key,
        model_cls=GatedLIFNetwork if model == "gated" else EligibilityLIFNetwork,
    )
    config.lr = 0.0
    config.t0 = t0
    config.critic_lr = 0.0
    config.t1 = t1
    config.save_at = dfx.SaveAt(
        ts=jnp.linspace(config.t0, config.t1, int(1e3 * delta_t)),
        fn=lambda t, state, args: (
            save_part_of_state(
                state,
                environment_state=True,
                agent_output=True,
                reward_signal=True,
                reward_predictor_state=True,
                RPE=True,
            ),
            jnp.max(state.agent_state.network_state.features.eligibility),
        ),
    )
    config.save_file = "results/pendulum_analysis/reward_prediction_test.npz"

    print(
        f"Running simulation from t0={t0} to t1={t1} with initial state from {selected_run['file_path']}."
    )
    start = time.time()
    sol, model = run_simulation(
        config, save_results=True, return_final_state=True, overwrite=False, y0=y0
    )
    end = time.time()
    print(f"Simulation time: {end - start}")

    # Extract fields
    state: SystemState = sol.ys[0][0]
    max_eligibility = sol.ys[0][1]
    sim_ts = sol.ts
    reward = state.reward_signal
    predicted_reward = state.agent_state.reward_predictor_state.value
    RPE = state.agent_state.RPE
    environment_state = state.environment_state
    agent_output = state.agent_output
    max_dW = RPE.squeeze() * max_eligibility.squeeze()

    # Compute actual vs predicted reward
    # Actual reward is the discounted sum of future rewards
    tau_discount = 0.1
    t_step = jnp.linspace(0, 0.5, int(1e3 * 0.5)) / config.dt
    discount_kernel = jnp.exp(-config.dt / tau_discount * t_step)
    discounted_reward = (
        jnp.convolve(reward.squeeze(), jnp.flip(discount_kernel.squeeze()), mode="same")
        * 10
    )

    fig, axs = plt.subplots(4, 1, figsize=(10, 5))
    ax = axs[0]
    ax.plot(sim_ts, discounted_reward, label="Discounted Reward")
    ax.plot(sim_ts, predicted_reward, label="Predicted Reward")
    ax2 = ax.twinx()
    ax2.plot(sim_ts, RPE, label="RPE", color="red")
    ax2.set_ylabel("RPE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.legend()

    axs[1].plot(sim_ts, environment_state[:, 0], label="Pendulum Angle")
    axs[1].plot(sim_ts, environment_state[:, 1], label="Pendulum Angular Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Pendulum State")
    axs[1].legend()

    axs[2].plot(sim_ts, agent_output, label="Agent Output")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Agent Output")
    axs[2].legend()

    axs[3].plot(sim_ts, max_dW, label="Max dW")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Max dW")
    axs[3].legend()
    plt.show()


if __name__ == "__main__":
    main()
