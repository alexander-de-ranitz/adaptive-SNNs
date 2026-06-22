import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import time

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt
from matplotlib import pyplot as plt

from adaptive_SNN.models import SystemState
from adaptive_SNN.models.networks import EligibilityLIFNetwork
from adaptive_SNN.simulation_configs.single_synapse_config import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    start = time.time()

    model_cls = EligibilityLIFNetwork

    cfg = create_single_synapse_learning_config(
        network_cls=model_cls,
        reward_noise_jump_rate=1.0,
        key=jr.PRNGKey(12),
    )
    cfg.balance = 1.05

    def save_fn(t, x: SystemState, args):
        return save_part_of_state(
            x,
            W=True,
            G=True,
            charge_in=True,
            charge_out=True,
            mean_E_conductance=True,
            mean_I_conductance=True,
            perturbations=True,
            V=True,
            filtered_spike_trains=True,
        )

    # def save_fn(t, x: SystemState, args):
    #     reward = x.environment_state.reward.astype(jnp.float32)
    #     reward_noise = x.environment_state.reward_noise.astype(jnp.float32)
    #     eligibility = x.agent_state.network_state.features.eligibility[0, -1].astype(
    #         jnp.float32
    #     )
    #     return (reward, reward_noise, eligibility)

    cfg.t1 = 100.0
    cfg.save_at = SaveAt(
        ts=jnp.linspace(95, cfg.t1, int(1e3 * (cfg.t1 - 95))),
        fn=save_fn,
    )
    cfg.save_file = "results/single_synapse_learning.npz"
    cfg.args["delta_V"] = jnp.pow(2.0, -12)
    cfg.base_network_kwargs["balance_rate"] = 1.0
    cfg.base_network_kwargs["tau_spike_filter"] = 1.0
    cfg.base_network_kwargs["tau_low_pass"] = 1.0

    sol, model = run_simulation(cfg, save_results=True, overwrite=True)

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")

    state: SystemState = sol.ys
    # reward, reward_noise, eligibility = sol.ys[0].squeeze(), sol.ys[1].squeeze(), sol.ys[2].squeeze()

    # reward, reward_noise, eligibility = state.environment_state.reward.squeeze(), state.environment_state.reward_noise.squeeze(), state.agent_state.network_state.features.eligibility[:, 0, -1].squeeze()

    # dW_task = (eligibility * reward).ravel()
    # dW_noise = (eligibility * reward_noise).ravel()

    # alignment = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_task))
    # snr = jnp.sum(dW_task) / jnp.sum(jnp.abs(dW_noise))

    # print(f"Alignment: {alignment:.4f}, SNR: {snr:.4f}")

    balance_over_time = jnp.array(
        [
            model.agent.network.compute_balance(
                0.0,
                jax.tree.map(lambda v: v[i], state.agent_state.network_state),
                cfg.args,
            )
            for i in range(len(sol.ts))
        ]
    )
    print("Mean balance over time: ", jnp.mean(balance_over_time, axis=0))

    plt.plot(sol.ts, state.agent_state.network_state.V[:, 0])
    plt.show()
    # fig, axs = plt.subplots(1, 6, figsize=(12, 4), sharex=True)
    # axs[0].plot(sol.ts, state.agent_state.network_state.W[:, 0, -3], label="E weight", c='g')
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Weight")
    # axs[0].plot(sol.ts, state.agent_state.network_state.W[:, 0, -2], label="I weight", c='r')
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Weight")

    # charge_diff = state.agent_state.network_state.charge_in[:, 0] + state.agent_state.network_state.charge_out[:, 0]
    # axs[1].plot(sol.ts, charge_diff, label="Charge Difference (In - Out)", c='k')
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Charge Difference")
    # axs[1].legend()

    # axs[2].plot(sol.ts, state.agent_state.network_state.charge_in[:, 0], label="Charge In", c='g')
    # axs[2].plot(sol.ts, state.agent_state.network_state.charge_out[:, 0], label="Charge Out", c='r')
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Charge")
    # axs[2].legend()

    # axs[3].plot(sol.ts, state.agent_state.network_state.mean_E_conductance[:, 0], label="Mean E Conductance", c='g')
    # axs[3].plot(sol.ts, state.agent_state.network_state.mean_I_conductance[:, 0], label="Mean I Conductance", c='r')
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Conductance")
    # axs[3].legend()

    # axs[4].plot(sol.ts, balance_over_time[:, 0], label="E/I Balance", c='m')
    # axs[4].set_xlabel("Time (s)")
    # axs[4].set_ylabel("Balance")
    # axs[4].legend()

    # axs[5].plot(sol.ts, state.agent_state.network_state.filtered_spike_trains[:, 0], label="Filtered Spike Trains", c='k')
    # axs[5].set_xlabel("Time (s)")
    # axs[5].set_ylabel("Filtered Spikes")
    # axs[5].legend()
    # plt.show()


if __name__ == "__main__":
    main()
