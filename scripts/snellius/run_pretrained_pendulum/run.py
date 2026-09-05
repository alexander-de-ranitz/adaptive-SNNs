import jax

jax.config.update("jax_enable_x64", True)

import argparse
import os
import re
import time

import diffrax as dfx
import equinox as eqx
import pandas as pd
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array

from adaptive_snn.models.agent_env_system import SystemState
from adaptive_snn.models.networks.eligibility_LIF import EligibilityLIFNetwork
from adaptive_snn.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.utils.runner import load_final_state, run_batched_simulation
from adaptive_snn.utils.save_helper import load_named_result

RESULTS_DIR = "results/pendulum_pretrained_20260903_202317/results/"


class Result(eqx.Module):
    environment_state: Array
    agent_output: Array
    critic_value: Array
    RPE: Array
    reward_signal: Array
    balance_hist: Array
    filtered_spikes_L: Array
    filtered_spikes_R: Array
    filtered_spikes_H: Array
    balance_min: Array
    balance_max: Array
    balance_mean: Array
    balance_var: Array
    W_actor_input_mean: Array
    W_actor_recurrent_mean: Array
    W_actor_input_var: Array
    W_actor_recurrent_var: Array


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
            try:
                iter = int(re.search(r"_iter_(\d+)", file_path).group(1))
            except:
                iter = None
    try:
        chunk = int(re.search(r"_chunk_(\d+)", file_path).group(1))
    except:
        chunk = None
    try:
        seed = int(re.search(r"_seed_(\d+)", file_path).group(1))
    except:
        seed = None
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
    # t1 = 974
    # delta_t = t1 - t0
    key = jr.fold_in(jr.PRNGKey(int(selected_run["seed"])), int(selected_run["iter"]))
    print("Using key: ", key)
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
        {"lr": 0.0, "balance_rate": 0.0, "tau_charge": 10.0},
        {"lr": 1e3, "balance_rate": 1.0, "tau_charge": 1.0},
        {"lr": 1e3, "balance_rate": 0.1, "tau_charge": 1.0},
        {"lr": 1e3, "balance_rate": 0.01, "tau_charge": 1.0},
        {"lr": 1e3, "balance_rate": 1.0, "tau_charge": 10.0},
        {"lr": 1e3, "balance_rate": 0.1, "tau_charge": 10.0},
        {"lr": 1e3, "balance_rate": 0.01, "tau_charge": 10.0},
        {"lr": 0.0, "balance_rate": 1.0, "tau_charge": 1.0},
        {"lr": 0.0, "balance_rate": 0.1, "tau_charge": 1.0},
        {"lr": 0.0, "balance_rate": 0.01, "tau_charge": 1.0},
        {"lr": 0.0, "balance_rate": 1.0, "tau_charge": 10.0},
        {"lr": 0.0, "balance_rate": 0.1, "tau_charge": 10.0},
        {"lr": 0.0, "balance_rate": 0.01, "tau_charge": 10.0},
    ]

    def save_fn(t, state: SystemState, args):
        charge_in = state.agent_state.network_state.charge_in
        charge_out = state.agent_state.network_state.charge_out
        balance = (charge_in + charge_out) / (jnp.abs(charge_in) + jnp.abs(charge_out))
        return Result(
            environment_state=state.environment_state,
            agent_output=state.agent_output,
            critic_value=state.agent_state.reward_predictor_state.value,
            RPE=state.agent_state.RPE,
            reward_signal=state.reward_signal,
            balance_hist=jnp.histogram(balance, bins=jnp.linspace(0.0, 0.05, 15))[0],
            filtered_spikes_L=jnp.mean(
                state.agent_state.network_state.filtered_spike_trains[
                    : N_neurons // 10
                ],
                axis=0,
            ),
            filtered_spikes_R=jnp.mean(
                state.agent_state.network_state.filtered_spike_trains[
                    N_neurons // 10 : N_neurons // 5
                ],
                axis=0,
            ),
            filtered_spikes_H=jnp.mean(
                state.agent_state.network_state.filtered_spike_trains[N_neurons // 5 :],
                axis=0,
            ),
            balance_min=jnp.min(balance),
            balance_max=jnp.max(balance),
            balance_mean=jnp.nanmean(balance),
            balance_var=jnp.nanvar(balance),
            W_actor_input_mean=jnp.nanmean(
                state.agent_state.network_state.W[:, N_neurons:]
            ),
            W_actor_recurrent_mean=jnp.nanmean(
                state.agent_state.network_state.W[:, :N_neurons]
            ),
            W_actor_input_var=jnp.nanvar(
                state.agent_state.network_state.W[:, N_neurons:]
            ),
            W_actor_recurrent_var=jnp.nanvar(
                state.agent_state.network_state.W[:, :N_neurons]
            ),
        )

    for p, config in zip(params, configs):
        config.lr = jnp.asarray(p["lr"])
        config.t0 = t0
        config.t1 = t1
        config.balance_rate = jnp.asarray(p["balance_rate"])
        config.tau_charge = jnp.asarray(p["tau_charge"])
        config.get_balance_rate = lambda t, state, args: args["final_balance_rate"]
        config.save_at = dfx.SaveAt(
            ts=jnp.linspace(config.t0, config.t1, int(1e2 * delta_t)), fn=save_fn
        )
        config.save_file = os.path.join(
            args.output_dir,
            "results",
            f"pendulum_AC_continued_{model}_lr_{p['lr']}_balance_{p['balance_rate']}_tau_charge_{p['tau_charge']}_seed_{int(selected_run['seed'])}.npz",
        )

    print(
        f"Running simulation from t0={t0} to t1={t1} with initial state from {selected_run['file_path']}."
    )
    start = time.time()
    sol, model = run_batched_simulation(
        configs, save_results=True, return_final_state=True, y0s=y0s, named_result=True
    )
    end = time.time()
    print(f"Simulation finished in: {end - start} seconds")


if __name__ == "__main__":
    main()
