import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import time

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt
from matplotlib import pyplot as plt

from adaptive_snn.models import SystemState
from adaptive_snn.models.networks import EligibilityLIFNetwork
from adaptive_snn.simulation_configs.single_synapse_learning_AC import (
    create_single_synapse_learning_config,
)
from adaptive_snn.utils.runner import run_simulation
from adaptive_snn.utils.save_helper import save_part_of_state


def main():
    start = time.time()

    model_cls = EligibilityLIFNetwork

    cfg = create_single_synapse_learning_config(
        network_cls=model_cls,
        reward_noise_jump_rate=1.0,
        key=jr.PRNGKey(12),
    )
    cfg.balance = 0.01
    cfg.min_noise_std = 1e-9
    cfg.initial_weight_matrix = jnp.tile(
        jnp.array([jnp.nan] * cfg.N_neurons + [0.5, 2, 0.5]),
        (cfg.N_neurons, 1),
    )

    def save_fn(t, x: SystemState, args):
        return save_part_of_state(
            x,
            charge_in=True,
            charge_out=True,
            mean_E_conductance=True,
            mean_I_conductance=True,
            filtered_spike_trains=True,
            V=True,
            reward_signal=True,
            reward_noise=True,
            RPE=True,
            perturbations=True,
            eligibility=True,
            reward_predictor_state=True,
        )

    cfg.t0 = 0
    cfg.t1 = 100
    cfg.save_at = SaveAt(
        ts=jnp.linspace(20, cfg.t1, int(1e3 * (cfg.t1 - 20))),
        fn=save_fn,
    )
    cfg.save_file = "results/single_synapse_learning.npz"
    cfg.args["delta_V"] = jnp.pow(2.0, -13)

    sol, model = run_simulation(cfg, save_results=False)

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")

    state: SystemState = sol.ys

    # Plot conductance, charge, and V over time
    fig, axs = plt.subplots(7, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(
        sol.ts,
        state.agent_state.network_state.V,
        label="Membrane Potential (V)",
        c="purple",
    )
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Membrane Potential (V)")
    axs[0].legend()

    axs[1].plot(
        sol.ts,
        state.agent_state.network_state.perturbations,
        label="Perturbations",
        c="b",
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Perturbations")
    axs[1].legend()

    axs[2].plot(
        sol.ts,
        state.agent_state.network_state.mean_E_conductance,
        label="Mean E Conductance",
        c="g",
    )
    axs[2].plot(
        sol.ts,
        state.agent_state.network_state.mean_I_conductance,
        label="Mean I Conductance",
        c="r",
    )
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Conductance")
    axs[3].legend()

    axs[3].plot(
        sol.ts, state.agent_state.RPE, label="Reward Prediction Error (RPE)", c="m"
    )
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("RPE")
    axs[3].legend()

    axs[4].plot(
        sol.ts,
        state.agent_state.network_state.features.eligibility[:, 0, -1],
        label="Eligibility",
    )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Eligibility")
    axs[4].legend()

    dw = (
        state.agent_state.network_state.features.eligibility[:, 0, -1]
        * state.agent_state.RPE.squeeze()
    )
    axs[5].plot(sol.ts, dw, label="Weight Change (dW)", c="orange")
    axs[5].set_xlabel("Time (s)")
    axs[5].set_ylabel("Weight Change (dW)")
    axs[5].legend()

    alignment = jnp.sum(dw) / jnp.sum(jnp.abs(dw))
    SNR = jnp.sum(jnp.abs(dw)) / jnp.sum(
        jnp.abs(
            state.agent_state.network_state.features.eligibility[:, 0, -1]
            * state.environment_state.reward_noise.squeeze()
        )
    )
    print(f"Alignment: {alignment:.4f}")
    print(f"SNR: {SNR:.4f}")
    plt.show()

    # reward, reward_noise, eligibility, V, filtered_spike_trains, S, W = sol.ys[0].squeeze(), sol.ys[1].squeeze(), sol.ys[2].squeeze(), sol.ys[3].squeeze(), sol.ys[4].squeeze(), sol.ys[5].squeeze(), sol.ys[6].squeeze()

    # print(jnp.sum(S, axis=0))

    # fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    # axs[0].plot(sol.ts, reward, label="Reward", c='g')
    # axs[0].plot(sol.ts, reward_noise, label="Reward Noise", c='gray')
    # axs[0].set_ylabel("Reward")
    # axs[0].legend()

    # axs[1].plot(sol.ts, eligibility, label="Eligibility", c='b')
    # axs[1].set_ylabel("Eligibility")
    # axs[1].legend()

    # axs[2].plot(sol.ts, V, label="Membrane Potential", c='r')
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Membrane Potential (V)")
    # axs[2].legend()

    # axs[3].plot(sol.ts, filtered_spike_trains, label="Filtered Spike Trains", c='k')
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Filtered Spikes")
    # axs[3].legend()

    # axs[4].plot(sol.ts, W[:, -3], label="E weight", c='g')
    # axs[4].plot(sol.ts, W[:, -2], label="I weight", c='r')
    # axs[4].set_xlabel("Time (s)")
    # axs[4].set_ylabel("Weight")
    # axs[4].legend()
    # plt.show()

    # reward, reward_noise, eligibility = state.environment_state.reward.squeeze(), state.environment_state.reward_noise.squeeze(), state.agent_state.network_state.features.eligibility[:, 0, -1].squeeze()

    # dW_task = (eligibility * reward).ravel()
    # dW_noise = (eligibility * reward_noise).ravel()

    # alignment = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_task))
    # snr = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_noise))

    # print(f"Mean reward: {jnp.mean(reward)}, Mean reward noise: {jnp.mean(reward_noise)}")
    # print(f"Mean eligibility: {jnp.mean(eligibility)}")
    # print(f"Alignment: {alignment:.4f}, SNR: {snr:.4f}")

    # zero_mean_elig = eligibility - jnp.mean(eligibility)
    # dw_task = (zero_mean_elig * reward).ravel()
    # dw_noise = (zero_mean_elig * reward_noise).ravel()
    # alignment = jnp.sum(dw_task) / jnp.sum(jnp.abs(dw_task))
    # snr = jnp.sum(dw_task) / jnp.sum(jnp.abs(dw_noise))
    # print(f"Alignment with zero-mean eligibility: {alignment:.4f}, SNR: {snr:.4f}")

    # eligibility_pos_only = jnp.clip(eligibility, min = 0.0)
    # dW_task_pos = (eligibility_pos_only * reward).ravel()
    # dW_noise_pos = (eligibility_pos_only * reward_noise).ravel()

    # alignment_pos = jnp.sum(dW_task_pos) / jnp.sum(jnp.abs(dW_task_pos))
    # snr_pos = jnp.sum(dW_task_pos) / jnp.sum(jnp.abs(dW_noise_pos))
    # print(f"Alignment pos elig only: {alignment_pos:.4f}, SNR: {snr_pos:.4f}")

    # eligibility_neg_only = jnp.clip(eligibility, max = 0.0)
    # dW_task_neg = (eligibility_neg_only * reward).ravel()
    # dW_noise_neg = (eligibility_neg_only * reward_noise).ravel()

    # alignment_neg = jnp.sum(dW_task_neg) / jnp.sum(jnp.abs(dW_task_neg))
    # snr_neg = jnp.sum(dW_task_neg) / jnp.sum(jnp.abs(dW_noise_neg))
    # print(f"Alignment neg elig only: {alignment_neg:.4f}, SNR: {snr_neg:.4f}")

    # dW_task = (eligibility * jnp.clip(reward, min=0.0)).ravel()
    # dW_noise = (eligibility * jnp.clip(reward_noise, min=0.0)).ravel()

    # alignment = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_task))
    # snr = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_noise))

    # print(f"Alignment pos reward only: {alignment:.4f}, SNR: {snr:.4f}")

    # dW_task = (eligibility * jnp.clip(reward, max=0.0)).ravel()
    # dW_noise = (eligibility * jnp.clip(reward_noise, max=0.0)).ravel()

    # alignment = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_task))
    # snr = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_noise))

    # print(f"Alignment neg reward only: {alignment:.4f}, SNR: {snr:.4f}")


if __name__ == "__main__":
    main()
