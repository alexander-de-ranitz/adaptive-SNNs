import re

import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

import equinox as eqx
from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array

from adaptive_snn.models.agent_env_system import SystemState
from adaptive_snn.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.utils.runner import run_batched_simulation


class PendulumSavedState(eqx.Module):
    mean_V: Array
    mean_RPE: Array
    var_RPE: Array
    mean_reward: Array
    mean_balance: Array
    var_balance: Array
    agent_output: Array
    mean_filtered_spikes_L: Array
    mean_filtered_spikes_R: Array
    mean_filtered_spikes_H: Array
    var_filtered_spikes_L: Array
    var_filtered_spikes_R: Array
    var_filtered_spikes_H: Array
    mean_W_actor_input: Array
    mean_W_actor_recurrent: Array
    var_W_actor_input: Array
    var_W_actor_recurrent: Array
    mean_W_critic: Array
    var_W_critic: Array
    fraction_clipped_dW: Array


def main():
    print(f"Jax is using device: {jax.devices()[0]}")

    parser = argparse.ArgumentParser(
        description="Run network simulation with specified parameters"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/default_folder/results/pendulum",
        help="Path to save the simulation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model type to use for the simulation (e.g., 'default', 'gated')",
    )

    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )

    parser.add_argument(
        "--N_parallel",
        type=int,
        default=1,
        help="Number of parallel simulations to run",
    )

    args = parser.parse_args()
    key = jr.PRNGKey(args.key_seed)
    N_parallel = args.N_parallel
    N_neurons = 1000
    model = GatedLIFNetwork if args.model == "gated" else EligibilityLIFNetwork
    configs = [
        create_pendulum_AC_config(
            N_neurons=N_neurons, model_cls=model, key=jr.fold_in(key, i)
        )
        for i in range(N_parallel)
    ]

    def save_fn(t, x: SystemState, args):
        total_charge = jnp.abs(x.agent_state.network_state.charge_in) + jnp.abs(
            x.agent_state.network_state.charge_out
        )
        balance = jnp.where(
            (x.agent_state.network_state.charge_in != 0.0)
            & (x.agent_state.network_state.charge_out != 0.0),
            (
                x.agent_state.network_state.charge_in
                + x.agent_state.network_state.charge_out
            )
            / total_charge,
            jnp.nan,
        )
        mean_balance = jnp.nanmean(balance)
        var_balance = jnp.nanvar(balance)
        fraction_clipped_dW = jnp.mean(
            jnp.abs(
                x.agent_state.network_state.features.eligibility * x.agent_state.RPE
            )
            > args["gradient_clip"]
        )

        return PendulumSavedState(
            mean_RPE=x.agent_state.mean_RPE,
            var_RPE=x.agent_state.var_RPE,
            mean_reward=x.mean_reward,
            mean_balance=mean_balance,
            var_balance=var_balance,
            agent_output=x.agent_output,
            mean_V=jnp.mean(x.agent_state.network_state.V),
            mean_filtered_spikes_L=jnp.mean(
                x.agent_state.network_state.filtered_spike_trains[
                    0 : int(N_neurons / 10)
                ]
            ),
            mean_filtered_spikes_R=jnp.mean(
                x.agent_state.network_state.filtered_spike_trains[
                    int(N_neurons / 10) : int(N_neurons / 5)
                ]
            ),
            mean_filtered_spikes_H=jnp.mean(
                x.agent_state.network_state.filtered_spike_trains[int(N_neurons / 5) :]
            ),
            var_filtered_spikes_L=jnp.var(
                x.agent_state.network_state.filtered_spike_trains[
                    0 : int(N_neurons / 10)
                ]
            ),
            var_filtered_spikes_R=jnp.var(
                x.agent_state.network_state.filtered_spike_trains[
                    int(N_neurons / 10) : int(N_neurons / 5)
                ]
            ),
            var_filtered_spikes_H=jnp.var(
                x.agent_state.network_state.filtered_spike_trains[int(N_neurons / 5) :]
            ),
            mean_W_actor_input=jnp.nanmean(
                x.agent_state.network_state.W[:, N_neurons:]
            ),
            mean_W_actor_recurrent=jnp.nanmean(
                x.agent_state.network_state.W[:, :N_neurons]
            ),
            var_W_actor_input=jnp.nanvar(x.agent_state.network_state.W[:, N_neurons:]),
            var_W_actor_recurrent=jnp.nanvar(
                x.agent_state.network_state.W[:, :N_neurons]
            ),
            mean_W_critic=jnp.nanmean(x.agent_state.reward_predictor_state.weights),
            var_W_critic=jnp.nanvar(x.agent_state.reward_predictor_state.weights),
            fraction_clipped_dW=fraction_clipped_dW,
        )

    params = [
        {
            "lr": 1e3,
            "balance_rate": 0.0,
            "gradient_clip": jnp.inf,
            "tau_charge": 1.0,
            "learn_I_weights": True,
        },
        {
            "lr": 250,
            "balance_rate": 0.0,
            "gradient_clip": jnp.inf,
            "tau_charge": 1.0,
            "learn_I_weights": True,
        },
        {
            "lr": 500,
            "balance_rate": 0.1,
            "gradient_clip": jnp.inf,
            "tau_charge": 1.0,
            "learn_I_weights": True,
        },
        {
            "lr": 1e3,
            "balance_rate": 1.0,
            "gradient_clip": jnp.inf,
            "tau_charge": 1.0,
            "learn_I_weights": True,
        },
    ]
    for i, cfg in enumerate(configs):
        p = params[i % len(params)]
        lr = jnp.concat(
            [
                jnp.ones((cfg.N_neurons, cfg.N_neurons)) * p["lr"],
                jnp.ones((cfg.N_neurons, cfg.N_inputs)) * p["lr"],
            ],
            axis=1,
        )
        cfg.lr = lr
        cfg.balance_rate = jnp.asarray(p["balance_rate"])
        cfg.get_balance_rate = lambda t, state, args: jnp.max(
            jnp.array([10 - t * 9.0 / 100, args["final_balance_rate"]])
        )
        cfg.gradient_clip = jnp.asarray(p["gradient_clip"])
        cfg.tau_charge = jnp.asarray(p["tau_charge"])
        cfg.learn_I_weights = p["learn_I_weights"]
        cfg.save_file = (
            args.output_file
            + f"_lr_{p['lr']}_{i}_clip_{p['gradient_clip']}_tau_charge_{p['tau_charge']}_balance_rate_{p['balance_rate']}_learn_I_{p['learn_I_weights']}"
        )
        cfg.t1 = 2000

    start_time = time.time()
    N_chunks = 1
    t_prev = configs[0].t0
    full_t1 = configs[0].t1
    y0s = None
    print(f"Starting simulation with {N_parallel} parallel runs in {N_chunks} chunks.")
    for chunk in range(N_chunks):
        chunk_t0 = t_prev
        chunk_t1 = (chunk + 1) * (full_t1) / N_chunks
        for cfg in configs:
            cfg.t0 = chunk_t0
            cfg.t1 = chunk_t1
            cfg.save_at = SaveAt(
                ts=jnp.linspace(cfg.t0, cfg.t1, int(2 * (cfg.t1 - cfg.t0))), fn=save_fn
            )
            if re.search(r"_chunk_\d+$", cfg.save_file):
                cfg.save_file = re.sub(r"_chunk_\d+$", f"_chunk_{chunk}", cfg.save_file)
            else:
                cfg.save_file = cfg.save_file + f"_chunk_{chunk}"
        t_prev = chunk_t1
        start_chunk = time.time()
        sol, models = run_batched_simulation(
            configs,
            save_results=True,
            return_final_state=True,
            named_result=True,
            y0s=y0s,
        )
        end_chunk = time.time()
        print(
            f"Simulation ({chunk + 1}/{N_chunks}) took {end_chunk - start_chunk:.2f} seconds."
        )

        final_states = sol.ys[1]
        y0s = final_states

    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
