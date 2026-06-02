import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import jax.numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import SystemState
from adaptive_SNN.models.environments.pendulum import PendulumEnvironment
from adaptive_SNN.simulation_configs.network_config import create_network_config
from adaptive_SNN.simulation_configs.pendulum_config import (
    compute_rates,
    create_pendulum_config,
)
from adaptive_SNN.utils.runner import run_simulation


def plot_encoding_scheme():
    """Visualize the tuning curves of the encoding population for the pendulum environment."""
    env = PendulumEnvironment()

    # Create env states, evenly spaced, to visualize the tuning curves of the encoding population
    N_states = 14
    max_state_value = jnp.array(
        [env.max_allowed_angle, env.max_allowed_angular_velocity]
    )
    env_states = jnp.linspace(-max_state_value, max_state_value, N_states)
    rates = jnp.array([compute_rates(env_state) for env_state in env_states])
    print(jnp.sum(rates, axis=1))
    plt.figure(figsize=(6, 3))
    plt.imshow(
        rates.T,
        aspect="auto",
        extent=[0, N_states, 0, rates.shape[1]],
        interpolation="none",
    )
    plt.vlines(jnp.arange(0, N_states), ymin=0, ymax=rates.shape[1], colors="white")
    plt.colorbar(label="Firing Rate (Hz)")
    plt.show()


def plot_network_input():
    """Visualize the input to the network"""

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    for i, cfg in enumerate(
        [create_pendulum_config(N_neurons=1000), create_network_config(N_neurons=1000)]
    ):
        if i == 0:
            cfg.environment_kwargs["initial_angle_range"] = (0.0, 0.0)
        cfg.initial_input_weight = 1.0
        cfg.input_weight_std = 0.0
        cfg.t1 = 1e-4
        cfg.save_at = dfx.SaveAt(t1=True, ts=None)
        sol, model = run_simulation(cfg, save_results=False)

        state: SystemState = sol.ys
        input_W = state.agent_state.network_state.W[0, :, cfg.N_neurons :]

        total_conductance_change = jnp.zeros(cfg.N_neurons)
        total_spikes = 0
        for t in jnp.arange(0, 1000 * 1e-4, 1e-4):
            input_spikes = cfg.input_spike_fn(t, state, cfg.args)
            total_spikes += jnp.sum(input_spikes)
            conductance_increase = (
                jnp.where(jnp.isnan(input_W), 0.0, input_W) * input_spikes
            )
            conductance_per_neuron = jnp.sum(conductance_increase, axis=1)
            total_conductance_change += conductance_per_neuron

        axs[i].hist(total_conductance_change, bins=50)
        axs[i].set_xlabel("Total Conductance Increase")
        axs[i].set_ylabel("Number of Neurons")
        print(
            f"Total conductance change: mean={jnp.mean(total_conductance_change):.2f}, std={jnp.std(total_conductance_change):.2f}"
        )
        print(f"Total spikes: {total_spikes}")

    plt.show()


if __name__ == "__main__":
    plot_encoding_scheme()
    plot_network_input()
