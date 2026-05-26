import jax

jax.config.update("jax_enable_x64", True)

import time

from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.simulation_configs.network_config import create_network_config
from adaptive_SNN.utils.runner import run_simulation


def main():
    cfg = create_network_config(N_neurons=1000, key=jr.PRNGKey(98765))
    cfg.t1 = 0.25
    cfg.initial_rec_weight = jnp.ones(cfg.N_neurons)
    # cfg.initial_rec_weight = cfg.initial_rec_weight.at[0].set(10.0)
    cfg.initial_input_weight = 1.0
    cfg.balance = 0.5

    N_E_in = 150
    N_I_in = 0  # N_E_in // 4
    cfg.args.update({"N_simulated_I_inputs": N_I_in, "N_simulated_E_inputs": N_E_in})
    # save file from my manual run
    cfg.save_file = "results/network_tuning_manual/firing_rates_w_10.npz"

    def save(t, x: SystemState, args):
        # return x.environment_state.astype(jnp.float32)
        return (
            x.agent_state.network_state.network_state.W[:10].astype(jnp.float32),
            x.agent_state.network_state.network_state.G[:10].astype(jnp.float32),
        )

    cfg.network_output_fn = lambda t, agent_state, args: agent_state.network_state.S

    cfg.save_at = SaveAt(ts=jnp.linspace(0.0, cfg.t1, 200), fn=save)

    rate = jnp.array([N_E_in * 10])  # High frequency background input

    spike_key = jr.fold_in(cfg.key, 1337)

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint((t - cfg.t0) / cfg.dt), dtype=jnp.int64)
        return jr.poisson(
            jr.fold_in(spike_key, step_idx),
            rate * cfg.dt,
            shape=(cfg.N_neurons, cfg.N_inputs),
        )

    cfg.input_spike_fn = input_spike_fn

    start = time.time()
    sol, model = run_simulation(cfg, overwrite=False, save_results=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")

    estimate_conductance_statistics(cfg.balance, 1.0, 1.0, 6)

    W, G = sol.ys
    G = G * W
    excitatory_mask = model.agent.network.base_network.excitatory_mask

    print(f"G shape is {G.shape}")

    G_exc = jnp.nansum(G[:, :, excitatory_mask], axis=-1)
    G_inh = jnp.nansum(G[:, :, ~excitatory_mask], axis=-1)

    print(f"Mean G_exc: {jnp.nanmean(G_exc)}, Mean G_inh: {jnp.nanmean(G_inh)}")

    print(f"G_exc shape: {G_exc.shape}, G_inh shape: {G_inh.shape}")
    plt.plot(sol.ts, G_exc[:, 0], label="Excitatory Conductance", c="g")
    plt.plot(sol.ts, G_inh[:, 0], label="Inhibitory Conductance", c="r")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Synaptic Trace (G)")
    plt.title("Synaptic Traces")
    plt.tight_layout()
    plt.show()


def plot_spike_raster(t, spikes):
    neuron_firing_rates = jnp.sum(spikes, axis=0) / (t[-1] - t[0])  # in Hz
    mean_firing_rate = jnp.mean(neuron_firing_rates)
    print(f"Mean firing rate: {mean_firing_rate:.2f} Hz")

    fig, ax = plt.subplots(figsize=(8, 4))
    # Build spike times per neuron (reverse order to show I neurons at bottom)
    spike_times_per_neuron = [
        t[jnp.nonzero(spikes[:, i])[0]] for i in range(spikes.shape[1])
    ][::-1]
    if len(spike_times_per_neuron) < 10:
        ax.set_yticks(range(len(spike_times_per_neuron)))
    else:
        ax.set_yticks([])
    ax.eventplot(
        spike_times_per_neuron,
        colors="black",
        linelengths=0.8,
        linewidths=0.4,
    )
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Spike Raster Plot")
    plt.show()


def estimate_conductance_statistics(balance, w_rec, w_input, rec_firing_rate):
    input_firing_rate = 10
    N_rec = 1000
    N_input = 150
    p_E = 0.1
    p_I = 0.2
    tau_syn = 6e-3

    total_E_weight = N_rec * p_E * 0.8 * w_rec + N_input * w_input

    E_driving_force = jnp.abs(0.0 - (-70e-3))
    I_driving_force = jnp.abs(-80e-3 - (-70e-3))
    # balance = (total_I_weight * I_driving_force) / (total_E_weight * E_driving_force)
    total_I_weight = balance * (total_E_weight * E_driving_force) / (I_driving_force)
    I_weight = total_I_weight / (N_rec * p_I * 0.2)

    print(f"Expected excitatory weight: {w_rec}")
    print(f"Expected inhibitory weight: {I_weight}")

    expected_rec_conductance = (
        rec_firing_rate * N_rec * p_E * 0.8 * w_rec * tau_syn * 1e-9
    )
    expected_input_conductance = input_firing_rate * N_input * w_input * tau_syn * 1e-9
    expected_conductance = expected_rec_conductance + expected_input_conductance
    print(f"Expected excitatory conductance: {expected_conductance}")

    expected_rec_conductance_I = (
        rec_firing_rate * N_rec * p_I * 0.2 * I_weight * tau_syn * 1e-9
    )
    print(f"Expected inhibitory conductance: {expected_rec_conductance_I}")


if __name__ == "__main__":
    main()
