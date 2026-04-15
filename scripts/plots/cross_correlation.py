import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import SaveAt

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state
from scripts.simulation_configs.single_neuron_simulation import (
    create_single_neuron_config_extra_synapse,
)


def plot_cross_correlation():
    config = create_single_neuron_config_extra_synapse(N_neurons=2)
    config.t1 = 20
    t_start_saving = 2.0

    config.save_at = SaveAt(
        ts=jnp.linspace(
            t_start_saving, config.t1, int((config.t1 - t_start_saving) // config.dt)
        ),
        fn=lambda t, x, args: save_part_of_state(
            x,
            V=True,
            S=True,
            noise_state=True,
        ),
    )
    config.args["use_noise"] = jnp.array([True, False])
    config.min_noise_std = 5e-9

    key = jr.PRNGKey(15105)
    config.key = key

    sol, model = run_simulation(config, save_results=False)
    state: SystemState = sol.ys

    # Remove data around spike times to avoid the influence of spiking activity on the correlation analysis
    spike_idx = jnp.where(state.agent_state.noisy_network.network_state.S[:, 0] == 1)[0]
    V_0 = state.agent_state.noisy_network.network_state.V[:, 0]
    V_1 = state.agent_state.noisy_network.network_state.V[:, 1]
    noise = state.agent_state.noisy_network.noise_state[:, 0]

    remove_spikes = True
    if remove_spikes:
        for spike_id in spike_idx:
            WINDOW_BUFFER = 10e-3
            start_idx = max(0, spike_id - int(WINDOW_BUFFER / config.dt))
            end_idx = min(
                state.agent_state.noisy_network.network_state.V.shape[0],
                spike_id + int(WINDOW_BUFFER / config.dt),
            )
            V_0 = V_0.at[start_idx:end_idx].set(jnp.nan)
            V_1 = V_1.at[start_idx:end_idx].set(jnp.nan)
            noise = noise.at[start_idx:end_idx].set(jnp.nan)

        if jnp.isnan(V_0).all() or jnp.isnan(V_1).all() or jnp.isnan(noise).all():
            print(
                "All data points are NaN after removing spike windows. Cannot compute correlation."
            )
            return

        # Compute lagged correlation between noise and voltage difference.
        # Drop masked-out samples first, otherwise NaNs propagate and make correlations undefined.
        valid = jnp.isfinite(V_0) & jnp.isfinite(V_1) & jnp.isfinite(noise)
        V_0 = V_0[valid]
        V_1 = V_1[valid]
        noise = noise[valid]

        if jnp.sum(valid) < 100:
            print(
                "Not enough valid data points after removing spike windows. Cannot compute correlation."
            )
            return

    voltage_diff = V_0 - V_1
    apply_whitening = True
    if apply_whitening:

        def prewhiten_ar1(signal):
            signal_np = jnp.asarray(signal)
            signal_centered = signal_np - jnp.mean(signal_np)
            prev = signal_centered[:-1]
            nxt = signal_centered[1:]
            denom = jnp.dot(prev, prev)
            phi = 0.0 if denom == 0 else jnp.dot(prev, nxt) / denom
            residual = nxt - phi * prev
            return residual, phi

        noise_pw, phi = prewhiten_ar1(noise)
        voltage_diff_np = jnp.asarray(voltage_diff)
        voltage_diff_centered = voltage_diff_np - jnp.mean(voltage_diff_np)
        voltage_diff_pw = voltage_diff_centered[1:] - phi * voltage_diff_centered[:-1]
        noise = jnp.asarray(noise_pw)
        voltage_diff = jnp.asarray(voltage_diff_pw)

    max_lag = jnp.round(0.1 / config.dt).astype(int)  # maximum lag of 100 ms
    step = jnp.round(0.0001 / config.dt).astype(int)  # compute correlation every 1 ms
    lags = jnp.arange(-max_lag, max_lag + 1, step)
    corrs = []
    for lag in lags:
        corr = jnp.corrcoef(
            voltage_diff[max_lag:-max_lag],
            noise[max_lag + lag : noise.shape[0] + -max_lag + lag],
        )[0, 1]
        corrs.append(corr)
    plt.figure(figsize=(3.5, 2))
    plt.plot(lags * config.dt, corrs, c="k")
    plt.xlim(lags[0] * config.dt, lags[-1] * config.dt)
    plt.xticks(
        jnp.arange(-0.1, 0.11, 0.05),
        labels=[f"{int(x * 1000)}" for x in jnp.arange(-0.1, 0.11, 0.05)],
    )
    plt.xlabel("Lag (ms)")
    plt.ylabel("Cross-correlation")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    plot_cross_correlation()
