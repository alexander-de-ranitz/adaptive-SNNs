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

    cfg.dt = 1e-4
    cfg.balance = 0.01
    cfg.base_network_kwargs["balance_rate"] = 5.0
    cfg.base_network_kwargs["tau_spike_filter"] = 1.0
    cfg.min_noise_std = 1e-9

    def save_fn(t, x, args):
        return (
            x.agent_state.network_state.charge_in[0],
            x.agent_state.network_state.charge_out[0],
            x.agent_state.network_state.filtered_spike_trains[0],
            x.agent_state.network_state.W[0],
            x.agent_state.network_state.V[0],
            x.agent_state.network_state.S[0],
            x.agent_state.network_state.G[0],
            x.agent_state.network_state.perturbations[0],
        )

    cfg.t1 = 200
    cfg.save_at = SaveAt(
        ts=jnp.linspace(cfg.t0, cfg.t1, int(1000 * cfg.t1)), fn=save_fn
    )
    cfg.initial_weight_matrix = jnp.tile(
        jnp.array([jnp.nan] * cfg.N_neurons + [1.0, 8, 0.0]),
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
    conductances = sol.ys[6]
    perturbations = sol.ys[7]

    print(f"Mean E conductance: {jnp.mean(weights[:, -3] * conductances[:, -3])}")
    print(f"Mean syn. I conductance: {jnp.mean(weights[:, -2] * conductances[:, -2])}")

    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(ts, charge_in, label="Charge In")
    axs[0].plot(ts, charge_out, label="Charge Out")
    axs[0].plot(ts, charge_in + charge_out, label="Charge In + Charge Out", c="gray")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Charge")
    axs[0].set_title("Charge In and Out")
    axs[0].legend()

    total_charge = jnp.abs(charge_in) + jnp.abs(charge_out)
    balance = jnp.where(
        (charge_in != 0.0) & (charge_out != 0.0),
        (charge_in + charge_out) / total_charge,
        jnp.nan,
    )
    print(f"Average balance: {jnp.nanmean(balance)}")
    print(f"Mean voltage: {jnp.mean(voltages)}")
    print(f"mean filtered spike train: {jnp.mean(filtered_spike_trains)}")

    axs[1].plot(ts, balance, label="Balance")
    axs[1].axhline(cfg.balance, c="k", linestyle="--", label="Target Balance")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("au")
    axs[1].set_title("Balance")

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
    print("Correlation between voltage and perturbations:")
    print(jnp.corrcoef(voltages, perturbations)[0, 1])
    plt.show()


if __name__ == "__main__":
    main()
