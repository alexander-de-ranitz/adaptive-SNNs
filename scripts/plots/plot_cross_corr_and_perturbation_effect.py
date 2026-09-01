import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from diffrax import SaveAt
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg

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
        initial_synapse_weight=0.0, key=jr.PRNGKey(125)
    )

    config.N_neurons = 2
    config.balance = 0.01
    t0 = 0.0
    t1 = 12.04
    t_start_saving = 12.0
    t_onset = 12.01
    t_offset = 12.02
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

    max_iterations = 1000
    N_target = 100
    n_succes = 0
    key = jr.PRNGKey(2001)
    V_diff = None
    for i in range(max_iterations):
        print(
            f"Running simulation {i + 1}/{max_iterations}. Collected {n_succes}/{N_target} successful runs.",
            end="\r",
        )
        key = jr.fold_in(key, i)
        cfg_key, spike_key = jr.split(key, 2)
        config.key = cfg_key
        config.save_file = "results/perturbation_dist/run_" + str(i)

        def input_spike_fn(t, x, args):
            step_idx = jnp.asarray(jnp.rint((t - t0) / config.dt), dtype=jnp.int64)
            spikes_1d = jr.poisson(
                jr.fold_in(spike_key, step_idx),
                jnp.array([5000, 1250, 0.0]) * config.dt,
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
        n_succes += 1
        if n_succes >= N_target:
            break

    # Fit an exponential curve to the voltage difference after the perturbation offset
    def exponential_decay(t, A, tau, C):
        return A * jnp.exp(-t / tau) + C

    V_after_offset = V_diff[:, sol.ts >= t_offset]
    ts_after_offset = sol.ts[sol.ts >= t_offset] - t_offset
    initial_guess = [1e-3, 0.01, 0.0]  # A, tau, C
    estimated_params = []
    print(ts_after_offset.shape, V_after_offset.shape)
    for i in range(V_after_offset.shape[0]):
        try:
            popt, _ = curve_fit(
                lambda t, A, tau, C: exponential_decay(t, A, tau, C),
                ts_after_offset.squeeze(),
                V_after_offset[i].squeeze(),
                p0=initial_guess,
                bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
            )
            estimated_params.append(popt)
        except RuntimeError:
            print(f"Run {i}: Curve fitting did not converge, skipping this run.")
            continue
    estimated_params = jnp.vstack(estimated_params)
    mean_params = jnp.nanmean(estimated_params, axis=0)
    tau_std = jnp.nanstd(estimated_params[:, 1])
    print(
        f"Estimated exponential decay parameters (mean across runs): A={mean_params[0]:.4f}, tau={mean_params[1]:.4f}, C={mean_params[2]:.4f}"
    )
    print(f"Standard deviation of tau across runs: {tau_std:.4f}")
    # Plot the voltage difference over time for all runs
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
    config = create_single_synapse_learning_config(key=jr.PRNGKey(15105))
    config.initial_weight_matrix = jnp.tile(
        jnp.array([jnp.nan] * config.N_neurons + [0.5, 2.0, 0.5]),
        (config.N_neurons, 1),
    )
    config.t1 = 600
    t_start_saving = 100.0

    config.save_at = SaveAt(
        ts=jnp.arange(t_start_saving, config.t1, config.dt),
        fn=lambda t, x, args: save_part_of_state(
            x, V=True, S=True, perturbations=True, G=True, W=True
        ),
    )

    config.args["use_noise"] = jnp.array([True, False])
    config.min_noise_std = 10e-9
    config.balance = 0.05

    config.save_file = "results/perturbation_dist/correlation_run"
    sol, _ = run_simulation(
        config, save_results=False, overwrite=False, load_if_exists=False
    )
    state: SystemState = sol.ys

    # # Convert to np arrays for consistency, as loaded data is in np format
    V_0 = np.array(state.agent_state.network_state.V[:, 0], copy=True)
    V_1 = np.array(state.agent_state.network_state.V[:, 1], copy=True)
    noise = np.array(state.agent_state.network_state.perturbations[:, 0], copy=True)
    # G = np.asarray(state.agent_state.network_state.G[:, 0, :] * state.agent_state.network_state.W[:, 0, :])
    # G_total = np.nanmean(np.abs(G)) + LIFNetwork.leak_conductance
    # print("Estimated tau_eff = ", LIFNetwork.membrane_capacitance/G_total)

    V_diff = V_0 - V_1

    # Fit AR(1) model
    V_diff = V_diff - jnp.mean(V_diff)
    noise = noise - jnp.mean(noise)
    fit = AutoReg(np.asarray(noise), lags=1, trend="n").fit()
    phi_hat = fit.params[-1]  # AR(1) coefficient

    apply_filter = lambda x: x[1:] - phi_hat * x[:-1]
    V_diff_filtered = apply_filter(V_diff)
    noise_filtered = apply_filter(noise)

    # Remove data around spike times to avoid the influence of spiking activity on the correlation analysis
    # not doing this results in qualitatively similar results, with tau=0.0018, but the correlation function is more noisy and less smooth
    spike_idx = jnp.where(state.agent_state.network_state.S[:, 0] == 1)[0]
    for spike_id in spike_idx:
        WINDOW_BUFFER = 10e-3
        # start_idx = max(0, spike_id - int(WINDOW_BUFFER / config.dt))
        start_idx = max(0, spike_id - 1)
        end_idx = min(
            state.agent_state.network_state.V.shape[0],
            spike_id + int(WINDOW_BUFFER / config.dt),
        )
        V_diff_filtered = V_diff_filtered.at[start_idx:end_idx].set(np.nan)
        noise_filtered = noise_filtered.at[start_idx:end_idx].set(np.nan)

    # Define lags for cross-correlation computation
    max_lag = jnp.round(20e-3 / config.dt).astype(int)  # maximum lag of 20 ms
    step = jnp.round(1e-4 / config.dt).astype(int)  # compute correlation every 0.1 ms
    lags = jnp.arange(-max_lag, max_lag + 1, step)

    # Compute cross-correlation for the filtered signals
    corrs = []
    for lag in lags:
        X = V_diff_filtered[max_lag:-max_lag]
        Y = noise_filtered[max_lag + lag : noise_filtered.shape[0] - max_lag + lag]
        valid = jnp.isfinite(X) & jnp.isfinite(Y)
        if jnp.sum(valid) < 100:
            print(f"Not enough valid data points for lag {lag}. Skipping this lag.")
            continue
        corr = jnp.corrcoef(
            X[valid],
            Y[valid],
        )[0, 1]
        corrs.append(corr)

    ax.plot(lags * config.dt, corrs, c="k")
    ax.set_xlim(lags[0] * config.dt, lags[-1] * config.dt)
    ax.set_xticks(
        jnp.arange(-0.02, 0.02, 0.005),
        labels=[f"{int(x * 1000)}" for x in jnp.arange(-0.02, 0.02, 0.005)],
    )
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.grid(alpha=0.3)

    # Fit an exponential decay to the negative lags of the correlation function
    negative_mask = np.asarray(lags < 0)
    fit_lags = np.asarray(lags[negative_mask] * config.dt)
    fit_corrs = np.asarray(corrs)[negative_mask]
    valid = np.isfinite(fit_lags) & np.isfinite(fit_corrs)
    fit_lags = fit_lags[valid]
    fit_corrs = fit_corrs[valid]

    def exponential_decay(x, amplitude, tau, offset):
        return amplitude * np.exp(-x / tau) + offset

    if fit_lags.size >= 3:
        fit_x = np.asarray(-fit_lags, dtype=float)
        fit_corrs = np.asarray(fit_corrs, dtype=float)
        initial_amplitude = jnp.max(fit_corrs) - jnp.min(fit_corrs)
        initial_tau = 10e-3
        initial_offset = jnp.mean(fit_corrs)
        tau_lower_bound = 1e-5
        tau_upper_bound = 1

        try:
            popt, _ = curve_fit(
                exponential_decay,
                fit_x,
                fit_corrs,
                p0=[initial_amplitude, initial_tau, initial_offset],
                bounds=(
                    [-np.inf, tau_lower_bound, -np.inf],
                    [np.inf, tau_upper_bound, np.inf],
                ),
                maxfev=10000,
            )
            print(
                f"Fitted exponential parameters: amplitude={popt[0]:.4f}, tau={popt[1]:.6f}, offset={popt[2]:.4f}"
            )
            fitted_corrs = exponential_decay(fit_x, *popt)
            ax.plot(
                fit_lags,
                fitted_corrs,
                color="tab:red",
                linestyle="--",
                linewidth=2,
                label="Exponential fit",
            )
            ax.legend(loc="upper right")
        except RuntimeError:
            print("Exponential fit did not converge.")


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
