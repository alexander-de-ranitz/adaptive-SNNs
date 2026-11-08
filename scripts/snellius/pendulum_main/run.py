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
    W_input_L_minus_R_map: (
        Array  # (grid) mean input weight of left minus right output population
    )
    W_input_H_map: Array  # (grid) mean input weight of the hidden population
    episode_count: Array
    episode_length_sum: Array


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
        "--delta_V_exponent",
        type=str,
        default="None",
        help="Exponent for the gating function steepness (delta_V = 2^delta_V_exponent)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=750.0,
        help="Learning rate for the network",
    )

    args = parser.parse_args()
    N_neurons = 1000
    if args.delta_V_exponent != "None":
        # delta_V == inf is used to indicate the spike-gated variant of the gated learning rule
        if float(args.delta_V_exponent) == jnp.inf:
            delta_V = jnp.inf
        else:
            delta_V = jnp.power(
                jnp.float64(2), jnp.float64(float(args.delta_V_exponent))
            )
        model = GatedLIFNetwork
    else:
        delta_V = None
        model = EligibilityLIFNetwork
    encoding_grid_shape = (16, 16)

    params = [
        {"lr": float(args.lr), "noise_floor": 20e-9, "key_seed": 10101},
        {"lr": float(args.lr), "noise_floor": 20e-9, "key_seed": 20202},
        {"lr": float(args.lr), "noise_floor": 20e-9, "key_seed": 30303},
        {"lr": float(args.lr), "noise_floor": 20e-9, "key_seed": 40404},
        {"lr": float(args.lr), "noise_floor": 20e-9, "key_seed": 50505},
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
        W_input = x.agent_state.network_state.W[:, N_neurons:]

        # Compute the mean weight for each input neuron to the left, right, and hidden populations
        # This gives a rough idea of the state -> output mapping per population that is learned by the network
        pop_map = lambda rows: jnp.nanmean(W_input[rows], axis=0).reshape(
            encoding_grid_shape
        )
        W_input_L_minus_R_map = pop_map(slice(0, N_neurons // 10)) - pop_map(
            slice(N_neurons // 10, N_neurons // 5)
        )
        W_input_H_map = pop_map(slice(N_neurons // 5, N_neurons))

        return PendulumSavedState(
            mean_RPE=x.agent_state.mean_RPE,
            var_RPE=x.agent_state.var_RPE,
            mean_reward=x.analytics.mean_reward,
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
            W_input_L_minus_R_map=W_input_L_minus_R_map,
            W_input_H_map=W_input_H_map,
            episode_count=x.analytics.episode_count,
            episode_length_sum=x.analytics.episode_length_sum,
        )

    configs = []
    env_kwargs = {
        "initial_angle_range": (-5 * jnp.pi / 180, 5 * jnp.pi / 180),
        "initial_angular_velocity_range": (-10 * jnp.pi / 180, 10 * jnp.pi / 180),
        "max_allowed_angle": 30 * jnp.pi / 180,
        "max_allowed_angular_velocity": 90 * jnp.pi / 180,
        "max_episode_time": 2.0,
    }

    for i, p in enumerate(params):
        cfg = create_pendulum_AC_config(
            N_neurons=N_neurons,
            model_cls=model,
            encoding_grid_shape=encoding_grid_shape,
            key=jr.PRNGKey(p["key_seed"]),
            **env_kwargs,
        )
        cfg.additional_args["subtract_mean_perturbation"] = jnp.asarray(True)
        cfg.lr = jnp.asarray(p["lr"])
        cfg.noise_floor = jnp.asarray(p["noise_floor"])
        cfg.base_network_kwargs["tau_spike_filter"] = jnp.asarray(100e-3)
        cfg.balance_rate = jnp.asarray(0.1)
        cfg.get_balance_rate = lambda t, state, args: jnp.max(
            jnp.array([15 - t * 15 / 100, args["balance_rate"]])
        )
        cfg.critic_lr = 0.05
        cfg.reward_predictor_kwargs["tau_filtered_input"] = 0.1
        if delta_V is not None:
            cfg.delta_V = jnp.asarray(delta_V)
        cfg.tau_charge = jnp.asarray(1.0)
        cfg.save_file = (
            args.output_file + f"_cfg_{i}_" + "_".join(f"{k}_{v}" for k, v in p.items())
        )
        cfg.t1 = 3000

        configs.append(cfg)

    start_time = time.time()
    chunk_times = [0.0, 1000, 2000, 3000]
    N_chunks = len(chunk_times) - 1
    t_prev = chunk_times[0]
    y0s = None

    print(
        f"Starting simulation with {len(configs)} parallel runs in {N_chunks} chunks."
    )
    for chunk in range(N_chunks):
        chunk_t0 = t_prev
        chunk_t1 = chunk_times[chunk + 1]  # (chunk + 1) * (full_t1) / N_chunks
        for cfg in configs:
            cfg.t0 = chunk_t0
            cfg.t1 = chunk_t1
            cfg.save_at = SaveAt(
                ts=jnp.linspace(cfg.t0, cfg.t1, int(5 * (cfg.t1 - cfg.t0))), fn=save_fn
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
        y0s = [
            jax.tree.map(lambda arr: arr[i], final_states) for i in range(len(configs))
        ]

    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
