import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import SaveAt

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks import LIFNetwork, LIFState
from adaptive_SNN.simulation_configs.single_synapse_config import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


class ExternalNoiseStd(LIFNetwork):
    def compute_desired_noise_std(self, t, state: LIFState, args):
        return args.get("external_noise_std")(t, state, args)

    def update(self, t, state, args):
        state = super().update(t, state, args)
        external_noise_std = args.get("external_noise_std")(t, state, args)
        perturbation = state.perturbations
        new_perturbation = jnp.where(
            external_noise_std > 0, perturbation, jnp.zeros_like(perturbation)
        )
        return eqx.tree_at(lambda s: s.perturbations, state, new_perturbation)


def plot_perturbation_distribution_over_time(ax):
    config = create_single_synapse_learning_config(
        initial_synapse_weight=5.0, key=jr.PRNGKey(125)
    )

    config.N_neurons = 2
    t0 = 0.0
    t1 = 2.04
    t_start_saving = 2.0
    t_onset = 2.01
    t_offset = 2.02
    external_noise_std = 1e-9

    config.network_cls = ExternalNoiseStd
    config.t0 = t0
    config.t1 = t1

    config.save_at = SaveAt(
        ts=jnp.arange(t_start_saving, t1, config.dt),
        fn=lambda t, x, args: save_part_of_state(
            x,
            V=True,
            S=True,
        ),
    )

    config.args["external_noise_std"] = lambda t, x, args: jnp.where(
        t > t_onset,
        jnp.where(
            t < t_offset,
            jnp.zeros((config.N_neurons,)).at[0].set(external_noise_std),
            jnp.zeros((config.N_neurons,)),
        ),
        jnp.zeros((config.N_neurons,)),
    )
    config.args["use_noise"] = jnp.array([True, False])

    n_iterations = 100
    key = jr.PRNGKey(2001)
    V_diff = None
    for i in range(n_iterations):
        print(f"Running simulation {i + 1}/{n_iterations}...", end="\r")
        key = jr.fold_in(key, i)
        cfg_key, spike_key = jr.split(key, 2)
        config.key = cfg_key
        config.save_file = "results/perturbation_dist/run_" + str(i)

        def input_spike_fn(t, x, args):
            step_idx = jnp.asarray(jnp.rint((t - t0) / config.dt), dtype=jnp.int64)
            spikes_1d = jr.poisson(
                jr.fold_in(spike_key, step_idx),
                jnp.array([5000, 1250, 10]) * config.dt,
                shape=(1, config.N_inputs),
            )
            return jnp.tile(spikes_1d, (config.N_neurons, 1))

        config.input_spike_fn = input_spike_fn

        sol, model = run_simulation(config, save_results=True)
        state: SystemState = sol.ys

        if jnp.sum(state.agent_state.network_state.S) > 0:
            print(
                f"Run {i}: Spikes detected during perturbation window, skipping this run."
            )
            continue  # Skip this run if there are any spikes, as we want to analyze the voltage distribution without the influence of spiking activity

        if not jnp.allclose(jnp.diff(sol.ts), config.dt):
            diff = jnp.diff(sol.ts)
            print(diff.max(), diff.min())

        V_diff = (
            state.agent_state.network_state.V[:, 0]
            - state.agent_state.network_state.V[:, 1]
            if V_diff is None
            else jnp.vstack(
                (
                    V_diff,
                    state.agent_state.network_state.V[:, 0]
                    - state.agent_state.network_state.V[:, 1],
                )
            )
        )

    ax.plot(
        sol.ts,
        V_diff.T * 1e3,
        color="darkgreen",
        alpha=0.3,
        label="Voltage Difference Samples",
    )

    y0, y1 = ax.get_ylim()
    y_extreme = max(abs(y0), abs(y1))
    y0, y1 = -y_extreme, y_extreme
    ax.vlines(
        [t_onset, t_offset],
        ymin=y0,
        ymax=y1,
        color="lightgray",
        linestyle="--",
        label="Perturbation Window",
    )
    ax.fill_betweenx([y0, y1], x1=t_onset, x2=t_offset, color="lightgray", alpha=0.5)
    ax.set_ylim(y0, y1)
    ax.set_xticks(
        jnp.linspace(t_start_saving, t1, 5),
        labels=[
            f"{jnp.round((t - t_start_saving) * 1000).astype(int)}"
            for t in jnp.linspace(t_start_saving, t1, 5)
        ],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage Difference (mV)")


def plot_cross_correlation(ax):
    config = create_single_synapse_learning_config(
        key=jr.PRNGKey(15105), initial_synapse_weight=10.0
    )
    config.t1 = 20
    t_start_saving = 2.0

    config.save_at = SaveAt(
        ts=jnp.arange(t_start_saving, config.t1, config.dt),
        fn=lambda t, x, args: save_part_of_state(
            x,
            V=True,
            S=True,
            perturbations=True,
        ),
    )
    config.args["use_noise"] = jnp.array([True, False])
    config.min_noise_std = 1e-9

    config.save_file = "results/perturbation_dist/correlation_run"
    sol, model = run_simulation(config, save_results=True)
    state: SystemState = sol.ys

    # Remove data around spike times to avoid the influence of spiking activity on the correlation analysis
    spike_idx = jnp.where(state.agent_state.network_state.S[:, 0] == 1)[0]
    V_0 = state.agent_state.network_state.V[:, 0]
    V_1 = state.agent_state.network_state.V[:, 1]
    noise = state.agent_state.network_state.perturbations[:, 0]

    remove_spikes = True
    if remove_spikes:
        for spike_id in spike_idx:
            WINDOW_BUFFER = 10e-3
            start_idx = max(0, spike_id - int(WINDOW_BUFFER / config.dt))
            end_idx = min(
                state.agent_state.network_state.V.shape[0],
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
    ax.plot(lags * config.dt, corrs, c="k")
    ax.set_xlim(lags[0] * config.dt, lags[-1] * config.dt)
    ax.set_xticks(
        jnp.arange(-0.1, 0.11, 0.05),
        labels=[f"{int(x * 1000)}" for x in jnp.arange(-0.1, 0.11, 0.05)],
    )
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.0))
    plot_perturbation_distribution_over_time(axs[0])
    plot_cross_correlation(axs[1])

    for ax, label in zip(axs, [r"\textbf{A}", r"\textbf{B}"]):
        ax.text(
            -0.12,
            1.12,  # x, y in axes coordinates (outside top-left)
            label,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
        )
    plt.show()
