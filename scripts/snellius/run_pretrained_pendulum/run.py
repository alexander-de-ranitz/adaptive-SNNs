import jax

jax.config.update("jax_enable_x64", True)

# Add scripts/ to the path so that we can import from scripts
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import argparse
import os
import re
import time

import diffrax as dfx
import pandas as pd
from jax import numpy as jnp
from jax import random as jr

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks.eligibility_LIF import EligibilityLIFNetwork
from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_SNN.utils.runner import (
    _load_existing_solution,
    run_batched_simulation,
)

RESULTS_DIR = "results/pendulum_AC_noiseless_input_20260812_203939/results/"


def load_pendulum_results(file_path):
    model = "gated" if "gated" in file_path else "default"
    sol, _ = _load_existing_solution(file_path)
    ts = sol.ts
    saved_state = sol.ys[0]
    iter = int(re.search(r"_lr_\d+\.?\d*_(\d+)", file_path).group(1))
    chunk = int(re.search(r"_chunk_(\d+)", file_path).group(1))
    seed = int(re.search(r"_seed_(\d+)", file_path).group(1))
    lr = float(re.search(r"_lr_(\d+\.?\d*)_", file_path).group(1))
    final_state = sol.ys[1]
    return {
        "file_path": file_path,
        "state": saved_state,
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
    parser = argparse.ArgumentParser(
        description="Continue a simulation from a pretrained pendulum solution"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files",
    )
    args = parser.parse_args()

    df = load_all_data()

    selected_run = df.loc[(df["model"] == "gated") & (df["lr"] == 1000)].iloc[0]
    y0 = selected_run["final_state"]
    y0 = jax.tree.map(
        lambda x: x.astype(jnp.float64), y0
    )  # We store in 32 bit but want 64 bit in the simulation
    model = selected_run["model"]
    ts = selected_run["ts"]

    N_parallel = 4
    y0s = [y0 for _ in range(N_parallel)]
    t0 = ts[-1]
    delta_t = 250.0
    t1 = t0 + delta_t
    key = jr.fold_in(jr.PRNGKey(int(selected_run["seed"])), int(selected_run["iter"]))
    configs = [
        create_pendulum_AC_config(
            N_neurons=1000,
            key=key,
            model_cls=GatedLIFNetwork if model == "gated" else EligibilityLIFNetwork,
        )
        for _ in range(N_parallel)
    ]
    N_neurons = 1000

    params = [
        {"lr": 0.0, "balance_rate": 0.0},
        {"lr": 0.0, "balance_rate": 0.1},
        {"lr": 1e3, "balance_rate": 0.0},
        {"lr": 1e3, "balance_rate": 0.1},
    ]

    def save_fn(t, state: SystemState, args):
        charge_in = state.agent_state.network_state.charge_in
        charge_out = state.agent_state.network_state.charge_out
        balance = (charge_in + charge_out) / (jnp.abs(charge_in) + jnp.abs(charge_out))
        return (
            state.environment_state.astype(jnp.float32),
            state.agent_output.astype(jnp.float32),
            state.agent_state.reward_predictor_state.value.astype(jnp.float32),
            state.agent_state.RPE.astype(jnp.float32),
            state.reward_signal.astype(jnp.float32),
            jnp.histogram(balance, bins=jnp.linspace(0.0, 0.05, 15))[0].astype(
                jnp.float32
            ),
            jnp.nanmean(state.agent_state.network_state.W[:, N_neurons:]),
            jnp.nanmean(state.agent_state.network_state.W[:, :N_neurons]),
            jnp.nanvar(state.agent_state.network_state.W[:, N_neurons:]),
            jnp.nanvar(state.agent_state.network_state.W[:, :N_neurons]),
        )

    for p, config in zip(params, configs):
        config.lr = jnp.asarray(p["lr"])
        config.t0 = t0
        config.t1 = t1
        config.args["final_balance_rate"] = jnp.asarray(p["balance_rate"])
        config.args["get_balance_rate"] = lambda t, state, args: args[
            "final_balance_rate"
        ]
        config.save_at = dfx.SaveAt(
            ts=jnp.linspace(config.t0, config.t1, int(1e2 * delta_t)), fn=save_fn
        )
        config.save_file = os.path.join(
            args.output_dir,
            "results",
            "pendulum_AC_continued_{}_lr_{}_balance_{}_{}.npz".format(
                model, p["lr"], p["balance_rate"], int(selected_run["seed"])
            ),
        )

    print(
        f"Running simulation from t0={t0} to t1={t1} with initial state from {selected_run['file_path']}."
    )
    start = time.time()
    sol, model = run_batched_simulation(
        configs, save_results=True, return_final_state=True, y0s=y0s
    )
    end = time.time()
    print(f"Simulation finished in: {end - start} seconds")


if __name__ == "__main__":
    main()
