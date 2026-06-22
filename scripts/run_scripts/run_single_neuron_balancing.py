import jax

jax.config.update("jax_enable_x64", True)

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.simulation_configs.single_synapse_config import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation


def main():
    cfg = create_single_synapse_learning_config(key=jr.PRNGKey(0))
    cfg.save_file = "results/single_neuron_balancing.npz"

    cfg.balance = 1.04
    cfg.base_network_kwargs["balance_rate"] = 1.0
    cfg.base_network_kwargs["tau_spike_filter"] = 1.0
    cfg.min_noise_std = 0.0

    def save_fn(t, x, args):
        return (
            x.agent_state.network_state.charge_in[0],
            x.agent_state.network_state.charge_out[0],
            x.agent_state.network_state.filtered_spike_trains[0],
            x.agent_state.network_state.W[0],
            x.agent_state.network_state.V[0],
            x.agent_state.network_state.S[0],
            x.agent_state.network_state.G[0],
        )

    cfg.t1 = 100
    cfg.save_at = SaveAt(
        ts=jnp.linspace(cfg.t0, cfg.t1, int(10000 * cfg.t1)), fn=save_fn
    )
    cfg.initial_weight_matrix = jnp.tile(
        jnp.array([jnp.nan] * cfg.N_neurons + [0.6, 3.5, 0.0]),
        (cfg.N_neurons, 1),
    )
    sol, model = run_simulation(cfg, overwrite=True)

    # Plot results
    ts = sol.ts
    charge_in = sol.ys[0]
    charge_out = sol.ys[1]
    filtered_spike_trains = sol.ys[2]
    weights = sol.ys[3]
    voltages = sol.ys[4]

    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(ts, charge_in, label="Charge In")
    axs[0].plot(ts, charge_out, label="Charge Out")
    axs[0].plot(ts, charge_in + charge_out, label="Charge In + Charge Out", c="gray")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Charge")
    axs[0].set_title("Charge In and Out")
    axs[0].legend()

    balance_ratio = jnp.where(
        (jnp.abs(charge_out) > 0) & (jnp.abs(charge_in) > 0),
        charge_in / jnp.abs(charge_out),
        1.0,
    )

    axs[1].plot(ts, balance_ratio, label="Charge In/Out Ratio")
    axs[1].axhline(cfg.balance, c="k", linestyle="--", label="Target Balance")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Ratio")
    axs[1].set_title("Charge In/Out Ratio")
    axs[1].set_ylim(0.99, 1.05)

    axs[2].plot(ts, filtered_spike_trains, label="Filtered Spike Train")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Filtered Spike Train")
    axs[2].set_title("Filtered Spike Train")
    axs[2].legend()

    axs[3].plot(ts, weights[:, -2], label="I weight")
    axs[3].plot(ts, weights[:, -3], label="E weight")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Weights")
    axs[3].set_title("Synaptic Weights")
    axs[3].legend()

    axs[4].plot(ts, voltages, label="Voltage")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Voltage")
    axs[4].set_title("Membrane Voltage")
    axs[4].legend()

    plt.show()


if __name__ == "__main__":
    main()
